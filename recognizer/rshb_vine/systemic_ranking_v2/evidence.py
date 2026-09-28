"""Per-target candidate evidence shared by receipt replay and the live selector.

One builder over the actual frozen pool (4 visual Top20 + supplied 8175 cards). Text evidence reuses the
selector-v4 tokenizer, typed lexicon, claim extraction and typed states; what changes is the aggregation:

- token weights are catalogue specificities (how many producers / products carry a token), so generic
  words (winery, vino, muskat) weigh little without a hand list;
- producer and name are specificity-weighted sums of the claimed tokens that were read (capped), not an
  any-token flag and not a fraction, so unread words stay neutral; each read token carries the best native
  score per reader (Vision quantized 0.3/0.5/1.0 and Paddle continuous stay separate
  dimensions, never a cross-reader max);
- rivals are candidates of a different accepted (snapshot-03) product; other cards of the same product never
  make a token non-exclusive or a producer foreign;
- name tokens that are typed atoms (grape/colour/sugar/style words) stay in the name evidence only with
  their catalogue specificity (Muskat, Cabernet weigh little); the typed role reads them separately and
  names are kept whole because a partial grape reading (КАБЕРНЕ) is not typed evidence at all;
- repeated reads of one text (several crops, several readers) enter once through a max, never a sum;
- typed roles keep support / contradiction / unknown claim apart; a contradiction needs a comparable claim,
  an unread or unknown claim is neutral, a grape list is never contradicted (no source states it is complete;
  grape support is kept), and a contradiction counts as source-verified only when a contradicted comparable
  claim is itself verified.

No GT, no expected words and no model output are read here. Full raw evidence is returned next to the features.
"""
from collections import defaultdict
import math

from rshb_vine.interaction_ranker_v1 import evidence as E1
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text

SCHEMA_VERSION = 'systemic-ranking-v2-features-v1'
CHANNELS = V4.CHANNELS
TYPED = V4.TYPED_ROLES
MATCHED = ('exact', 'homoglyph', 'translit')
VERIFIED = ('source_verified',)
CAP = 2.

VISUAL = (*(f'vis.{c}.rr' for c in CHANNELS), *(f'vis.{c}.gap' for c in CHANNELS),
          'vis.top1_count', 'vis.label_agree', 'vis.context_agree',
          'vis.small_label_x_B0_label_rr', 'vis.small_label_x_B3_label_rr')
READERS = ('v', 'p')
READER_SHORT = {'vision_r3_macos': 'v', 'paddle_v5_cpu': 'p'}
FEATURE_NAMES = (
    *VISUAL, 'pool.control_proposal',
    'prod.read_w', 'prod.read_g.v', 'prod.read_g.p', 'prod.phrase', 'prod.other_g.v', 'prod.other_g.p',
    'name.read_w', 'name.read_g.v', 'name.read_g.p', 'name.phrase', 'name.exclusive_w', 'name.fuzzy_w',
    'series.read_w',
    *(f'typ.{r}.{k}.{x}' for r in TYPED for k in ('support_g', 'contra_g') for x in READERS
      if not (r == 'grape_blend' and k == 'contra_g')),
    'typ.contra_verified_g.v', 'typ.contra_verified_g.p',
)
# +1: adding this evidence can only raise the candidate's own score; -1: can only lower it; 0: free.
SIGNS = {
    **{f'vis.{c}.rr': 1 for c in CHANNELS}, **{f'vis.{c}.gap': -1 for c in CHANNELS},
    'vis.top1_count': 1, 'vis.label_agree': 1, 'vis.context_agree': 1,
    'vis.small_label_x_B0_label_rr': 0, 'vis.small_label_x_B3_label_rr': 0, 'pool.control_proposal': 1,
    **{n: (-1 if '.other_g.' in n or '.contra' in n else 1) for n in FEATURE_NAMES if n.startswith(('prod.', 'name.', 'series.', 'typ.'))},
}
CONTRADICTION_FEATURES = tuple(n for n in FEATURE_NAMES if SIGNS[n] < 0 and not n.endswith('.gap'))


def _spec(df, n):
    return 1. if n <= 1 else min(1., max(0., math.log(n / max(df, 1)) / math.log(n)))


class CatalogStats:
    """Token specificities over the selection-pool registry: producers by producer key, names by product."""

    def __init__(self, registry, context):
        producer_df, name_df = defaultdict(set), defaultdict(set)
        producers, products = set(), set()
        for slug, card in registry.cards.items():
            profile = registry.claims.get(slug)
            key = text.base_norm(card.get('raw_metadata', {}).get('Винодельня', '')) or 'card:' + slug
            forms = V4._producer_forms(card, profile, context)
            tokens = {t.skeletons[0] for f in forms for t in text.tokenize(f)}
            producers.add(key)
            for s in tokens:
                producer_df[s].add(key)
            values = V4._claim_values(profile, 'commercial_name')[0] or [card.get('raw_metadata', {}).get('Название вина', '')]
            products.add(card['product_id'])
            for t in V4._name_tokens(values, tokens, context):
                name_df[t.skeletons[0]].add(card['product_id'])
        self.n_producers, self.n_products = len(producers), len(products)
        self.producer = {s: _spec(len(v), self.n_producers) for s, v in producer_df.items()}
        self.name = {s: _spec(len(v), self.n_products) for s, v in name_df.items()}

    def producer_weight(self, token):
        return self.producer.get(token.skeletons[0], 1.)

    def name_weight(self, token):
        return self.name.get(token.skeletons[0], 1.)

    def describe(self):
        return {'producers': self.n_producers, 'products': self.n_products,
                'producer_tokens': len(self.producer), 'name_tokens': len(self.name)}


class _Index(V4._QueryIndex):
    def matches(self, claim_norm):
        """{line_id: best match kind} of one claim token over every line (same candidate places as v4)."""
        key = ('m', claim_norm)
        if key in self.cache:
            return self.cache[key]
        skeleton = text.skeleton(claim_norm)
        places = set(self.exact.get(claim_norm, ())) | set(self.skeleton.get(skeleton, ()))
        if skeleton and len(skeleton) >= 5:
            places |= {p for p in self.by_initial.get(skeleton[0], ())
                       if abs(len(self.lines[p[0]]['tokens'][p[1]].skeletons[0]) - len(skeleton)) <= 2}
        found = {}
        for line_id, pos in places:
            kind = text.token_match(self.lines[line_id]['tokens'][pos], claim_norm)
            if kind is not None and (line_id not in found or V4.KIND_RANK[kind] < V4.KIND_RANK[found[line_id]]):
                found[line_id] = kind
        self.cache[key] = found
        return found


def _line_grades(lines, provenance):
    """{line_id: {reader tag: best native score of that reader on the line}}; scales are never mixed."""
    grades = {}
    for line in lines:
        g = dict.fromkeys(READERS, 0.)
        for i in line['observation_ids']:
            record = provenance[i]
            score = record['native_score']
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                tag = E1.READER_TAGS[record['reader']]
                g[READER_SHORT[tag]] = max(g[READER_SHORT[tag]], float(score))
        grades[line['line_id']] = g
    return grades


def _best_grade(line_ids, grades):
    return {r: max((grades[l][r] for l in line_ids), default=0.) for r in READERS}


def _token_reads(index, grades, token):
    """Per-reader grades of the best non-fuzzy reads, fuzzy-only flag, supporting line ids."""
    hits = index.matches(token.forms[0])
    strong = [l for l, k in hits.items() if k in MATCHED]
    weak = [l for l, k in hits.items() if k == 'fuzzy']
    return _best_grade(strong, grades), bool(weak and not strong), sorted(strong or weak)


def _coverage(index, grades, tokens, weight):
    """Specificity-weighted coverage of claim tokens; a token counts once, at its best read per reader."""
    total = sum(weight(t) for t in tokens)
    empty = {'cover': 0., 'read_w': 0., 'fuzzy_w': 0., 'tokens': [], **{f'read_g.{r}': 0. for r in READERS}}
    if not tokens or total <= 0:
        return empty
    out = dict(empty, tokens=[])
    for t in tokens:
        w = weight(t)
        g, fuzzy, lines = _token_reads(index, grades, t)
        read = any(g.values())
        out['cover'] += w * read / total
        out['read_w'] += w * read
        out['fuzzy_w'] += w * fuzzy
        for r in READERS:
            out[f'read_g.{r}'] += w * g[r]
        out['tokens'].append({'token': t.forms[0], 'weight': round(w, 4), 'grades': g, 'fuzzy_only': fuzzy,
                              'lines': lines})
    return out


def _atoms(tokens, lexicon):
    """Positions of name tokens that the typed lexicon reads as grape/colour/sugar/style."""
    spans = set()
    for found in lexicon.entities(tuple(tokens)).values():
        for item in found:
            spans.update(range(item['start'], item['end']))
    return spans


def _members(candidate, context):
    members, keys = E1._members(candidate, context)
    for m in members.values():
        atoms = _atoms(m['name'], context.lexicon)
        m['core'] = [t for i, t in enumerate(m['name']) if i not in atoms]
        m['atoms'] = [t for i, t in enumerate(m['name']) if i in atoms]
    return members, keys


def _rank(record):
    return (record['read_w'], sum(record[f'read_g.{r}'] for r in READERS))


def _cap(value):
    return min(value, CAP) / CAP


def _comparable(role, claims):
    """Claims a reading can contradict: v4 comparable claims. No catalogue source grants negative inference from
    an absent grape (profiles carry supports_negative_inference_from_absence=false), so grape lists never are."""
    return [] if role == 'grape_blend' else [c for c in claims if c['keys']]


def build(base, v4, provenance, context, stats, control_slug, product_of):
    """Feature rows aligned with ``v4['candidates']`` and the raw evidence behind every value.

    ``product_of(slug)`` is the accepted product identity used to decide who is a rival: other aliases/cards
    of the same product never count as competing text.
    """
    lines = V4._lines(base['query_evidence']['observations'])
    index = _Index(lines)
    if len(provenance) != len(base['query_evidence']['observations']) or any(r is None for r in provenance):
        raise ValueError('Provenance side-car does not cover the observation packet')
    grades = _line_grades(lines, provenance)
    typed = E1.field_evidence(base, context)['candidates']
    candidates = v4['candidates']
    if [c['candidate_id'] for c in candidates] != [c['candidate_id'] for c in base['candidates']]:
        raise ValueError('v4 and base pools differ')
    per, keys, products = {}, {}, {}
    for c in base['candidates']:
        per[c['candidate_id']], keys[c['candidate_id']] = _members(c, context)
        products[c['candidate_id']] = {product_of(s) for s in c['card_slugs']}
    producer = {}
    for cid, members in per.items():
        reads = [dict(_coverage(index, grades, text.tokenize(form), stats.producer_weight), form=form)
                 for m in members.values() for form in m['producers']]
        producer[cid] = max(reads, key=_rank, default=None) or dict(_coverage(index, grades, [], stats.producer_weight), form=None)
    claimed = {cid: {t.skeletons[0] for m in members.values() for t in m['name']} for cid, members in per.items()}
    rows, raw = [], []
    for c in candidates:
        cid, v4f = c['candidate_id'], c['features']
        f = {n: float(v4f[n]) for n in VISUAL}
        f['pool.control_proposal'] = float(control_slug in c['card_slugs'])
        own = producer[cid]
        f['prod.read_w'] = _cap(own['read_w'])
        for r in READERS:
            f[f'prod.read_g.{r}'] = _cap(own[f'read_g.{r}'])
        f['prod.phrase'] = float(v4f['txt.producer.full_any'])
        rivals = [o for o in per if o != cid and not products[o] & products[cid]]
        other = [producer[o] for o in rivals if keys[o] and not keys[o] & keys[cid]]
        rival_producer = {}
        for r in READERS:
            best = max(other, key=lambda x: x[f'read_g.{r}'], default=None)
            f[f'prod.other_g.{r}'] = max(0., _cap(best[f'read_g.{r}']) - _cap(own[f'read_g.{r}'])) if best else 0.
            rival_producer[r] = best and {'form': best['form'], 'read_g': best[f'read_g.{r}']}
        rival_tokens = set().union(*(claimed[o] for o in rivals)) if rivals else set()
        names, series = [], []
        for slug, m in per[cid].items():
            read = _coverage(index, grades, m['name'], stats.name_weight)
            exclusive = sum(x['weight'] for x, t in zip(read['tokens'], m['name'])
                            if any(x['grades'].values()) and t.skeletons[0] not in rival_tokens)
            names.append(dict(read, slug=slug, exclusive=exclusive, atoms=[t.forms[0] for t in m['atoms']]))
            for tokens in m['series']:
                if tokens:
                    series.append(dict(_coverage(index, grades, tokens, stats.name_weight), slug=slug))
        name = max(names, key=_rank, default=None) or _coverage(index, grades, [], stats.name_weight)
        f['name.read_w'] = _cap(name['read_w'])
        for r in READERS:
            f[f'name.read_g.{r}'] = _cap(name[f'read_g.{r}'])
        f['name.phrase'] = float(v4f['txt.name.full_any'])
        f['name.exclusive_w'] = _cap(max((n['exclusive'] for n in names), default=0.))
        f['name.fuzzy_w'] = _cap(max((n['fuzzy_w'] for n in names), default=0.))
        f['series.read_w'] = _cap(max((x['read_w'] for x in series), default=0.))
        typed_raw, verified = {}, dict.fromkeys(READERS, 0.)
        for role in TYPED:
            t = typed[cid]['typed'][role]
            claims = [m['typed'][role] for m in per[cid].values()]
            contradicted = _comparable(role, claims) if t['state']['contra_any'] else []
            sg = _best_grade(t['support'], grades)
            cg = _best_grade(t['contra'], grades) if contradicted else dict.fromkeys(READERS, 0.)
            for r in READERS:
                f[f'typ.{role}.support_g.{r}'] = sg[r]
                if role != 'grape_blend':
                    f[f'typ.{role}.contra_g.{r}'] = cg[r]
                if any(cl['status'] in VERIFIED for cl in contradicted):
                    verified[r] = max(verified[r], cg[r])
            typed_raw[role] = {'state': t['state'].get('kind'), 'support_lines': sorted(t['support']),
                               'contra_lines': sorted(t['contra']), 'disputed': t['disputed'],
                               'contradicted_claims': len(contradicted),
                               'claims': [{'keys': sorted({k['key'] for k in cl['keys']}), 'status': cl['status'],
                                           'source': cl['source']} for cl in claims]}
        for r in READERS:
            f[f'typ.contra_verified_g.{r}'] = verified[r]
        if set(f) != set(FEATURE_NAMES) or len(f) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in f.values()):
            raise ValueError('systemic-ranking-v2 feature invariant failed')
        rows.append({n: f[n] for n in FEATURE_NAMES})
        raw.append({'candidate_id': cid, 'producer_keys': sorted(keys[cid]), 'products': sorted(products[cid]),
                    'producer': {k: own.get(k) for k in ('form', 'cover', 'read_w', 'tokens', *(f'read_g.{r}' for r in READERS))},
                    'producer_rival': rival_producer,
                    'name': [{k: n[k] for k in ('slug', 'cover', 'read_w', 'exclusive', 'tokens', 'atoms',
                                                 *(f'read_g.{r}' for r in READERS))} for n in names],
                    'series': [{k: x[k] for k in ('slug', 'cover', 'tokens')} for x in series],
                    'typed': typed_raw, 'visual_ranks': c['v4_evidence']['visual_ranks']})
    query = {'lines': [{'line_id': l['line_id'], 'raw': l['raw'], 'grades': grades[l['line_id']],
                        'observation_ids': l['observation_ids'],
                        'sources': sorted({provenance[i]['reader'] + ':' + provenance[i]['crop_source']
                                           for i in l['observation_ids']})} for l in lines],
             'observed_typed': v4['query_evidence']['observed_typed'], 'small_label': v4['query_evidence']['small_label']}
    return rows, raw, query
