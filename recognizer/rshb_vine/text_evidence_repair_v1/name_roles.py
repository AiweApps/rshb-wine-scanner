"""Producer-printed origin words in NAME evidence (component C).

A commercial-name word can be an origin phrase that a producer prints on the labels of its other products whose
catalogue names lack it (``ДОЛИНА ДОНА`` on Ведерниковъ labels, also their Регион value). Reading such a word
cannot tell the product that names it from its siblings, yet systemic name evidence credits only the naming card.
The role table is built offline from catalogue names/Регион and existing gallery reference OCR (machine output,
used as catalogue-side evidence of what labels print, never as query truth). A role needs origin evidence (the
producer's Регион value or an origin-designation line); words of any typed lexicon entry, variant alias or the
generic vocabulary never get it, so abbreviated typed phrases keep their identity. At request time ``build`` is
``E.build`` with the flagged words out of the NAME tokens of every card of that producer, so own coverage/grades,
fuzzy weight and rival exclusivity change together; the integrator's pre-guard stage takes only the NAME
features from it. Nothing is removed for other producers.
"""
from collections import defaultdict
from pathlib import Path

from rshb_vine.io import digest, read_json, sha256, verify
from rshb_vine.positive_variant_text import ALIASES
from rshb_vine.producer_role_v2.roles import origin_line
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E
from rshb_vine.text_evidence_repair_v1 import producer_names as P

RULE = {
    'version': 'producer-printed-origin-name-words-v1',
    'scope': 'per producer key (raw Винодельня base_norm); roles never cross producers',
    'tokens': 'core commercial-name tokens (systemic name tokens minus typed grape/colour/sugar/style atoms), '
              'skeleton >= min_token_skeleton, skeleton not a word of any typed lexicon entry (curated aliases and '
              'catalogue typed forms, every word of multiword entries), of any positive_variant_text.ALIASES entry '
              '(reserve included) or of the generic vocabulary',
    'min_token_skeleton': 3,
    'observed_products': 'products of the producer with at least one gallery reference that has media-audit OCR',
    'printed': 'a reference OCR line reads the token non-fuzzy (exact/homoglyph/translit) at native score >= '
               'min_reference_score',
    'min_reference_score': 0.5,
    'role': 'printed_origin_word when unnamed printers >= min_unnamed_printers, unnamed printers >= observed '
            'products whose name carries the token, and origin evidence holds',
    'origin_evidence': 'the token is a word of the Регион value of a card of this producer, or an unnamed printing '
                       'reference line is an origin-designation line (producer_role_v2.origin_line markers)',
    'min_unnamed_printers': 2,
    'effect': 'core (non-atom) occurrences of the token leave NAME tokens of every card of that producer in systemic '
              'evidence; typed grape/colour/sugar/style atoms keep the whole name; nothing else changes',
}
TABLE = 'runs/text-evidence-improve-v1/identity/name-roles-v1.json'
SOURCES = (P.GALLERY, P.REFERENCE_OCR, 'rshb_vine/systemic_ranking_v2/evidence.py',
           'rshb_vine/interaction_ranker_v1/evidence.py')


def input_checksums(root, context):
    root = Path(root)
    return {**{p: sha256(root / p) for p in SOURCES}, **context.source_checksums}


def load_table(root, context, registry, expected_checksum):
    """The sealed table, refused unless its checksum is the pinned one and every input still has its SHA."""
    table = verify(read_json(Path(root) / TABLE))
    if table['checksum'] != expected_checksum or table['rule'] != RULE \
            or table['registry_checksum'] != registry.checksum:
        raise ValueError('Name-role table differs from the pinned table, rule or registry')
    if table['inputs_sha256'] != input_checksums(root, context):
        raise ValueError('Name-role table inputs changed since the table was built')
    return table


def _member(selection, slug):
    card, profile = selection.registry.cards[slug], selection.registry.claims.get(slug)
    pseudo = {'card_slugs': [slug], 'provenance': {'cards': {slug: card}, 'claims': {slug: profile}}}
    return E._members(pseudo, selection.context)[0][slug]


def build_table(root, selection):
    """Every core name token of every producer with its naming/printing products and the role decision."""
    root = Path(root)
    registry = selection.registry
    by_key, product_of = defaultdict(list), {}
    for slug, card in registry.cards.items():
        by_key[P._producer_key(card, slug)].append(slug)
        product_of[slug] = card['product_id']
    refs = defaultdict(set)
    for r in read_json(root / P.GALLERY)['references']:
        if r['slug'] in registry.cards:
            refs[r['slug']].add(r['image_sha256'])
    ocr = P._reference_ocr(root, {s for v in refs.values() for s in v})
    typed = {text.skeleton(n) for e in selection.context.lexicon._entries for n in e.norms}
    typed |= {t.skeletons[0] for aliases in ALIASES.values() for a in aliases for t in text.tokenize(a)}
    typed |= {text.skeleton(g) for g in selection.context.generic}
    candidates, roles, typed_words = [], defaultdict(list), defaultdict(list)
    for key in sorted(by_key):
        slugs = sorted(by_key[key])
        members = {s: _member(selection, s) for s in slugs}
        observed = {product_of[s] for s in slugs if any(h in ocr for h in refs[s])}
        region = {x for s in slugs for x in P._skeletons(registry.cards[s].get('raw_metadata', {}).get('Регион', ''))}
        tokens = {}
        for s in slugs:
            for t in members[s]['core']:
                if len(t.skeletons[0]) >= RULE['min_token_skeleton']:
                    tokens.setdefault(t.skeletons[0], t)
        for skel, token in sorted(tokens.items()):
            if skel in typed:
                typed_words[key].append(token.forms[0])
                continue
            named = {product_of[s] for s in slugs if skel in {t.skeletons[0] for t in members[s]['name']}}
            printers = defaultdict(list)
            for s in slugs:
                if product_of[s] in named:
                    continue
                for sha in sorted(refs[s]):
                    if sha not in ocr:
                        continue
                    lines, index, _ = ocr[sha]
                    for line_id, kind in sorted(index.matches(token.forms[0]).items()):
                        score = lines[line_id]['max_score'] or 0.
                        if kind in E.MATCHED and score >= RULE['min_reference_score']:
                            printers[product_of[s]].append({'slug': s, 'image_sha256': sha, 'kind': kind,
                                                            'raw': lines[line_id]['raw'], 'score': round(score, 4),
                                                            'origin_line': origin_line(lines[line_id]['tokens'])})
            named_observed = named & observed
            origin = {'region_field': skel in region,
                      'origin_line': any(x['origin_line'] for v in printers.values() for x in v)}
            role = (len(printers) >= RULE['min_unnamed_printers'] and len(printers) >= len(named_observed)
                    and any(origin.values()))
            record = {'producer_key': key, 'token': token.forms[0], 'skeleton': skel,
                      'named_products': sorted(named), 'named_observed_products': sorted(named_observed),
                      'unnamed_printing_products': {p: v for p, v in sorted(printers.items())},
                      'observed_products': len(observed), 'producer_products': len({product_of[s] for s in slugs}),
                      'origin_evidence': origin, 'role': 'printed_origin_word' if role else 'name'}
            if printers:
                candidates.append(record)
            if role:
                roles[key].append(skel)
    return {
        'kind': 'name-roles', 'rule': RULE, 'registry_checksum': registry.checksum,
        'inputs_sha256': input_checksums(root, selection.context),
        'gallery_references_with_ocr': len(ocr),
        'typed_lexicon_words': len(typed),
        'excluded_typed_words': {k: sorted(v) for k, v in sorted(typed_words.items())},
        'candidates': candidates, 'roles': {k: sorted(v) for k, v in sorted(roles.items())},
    }


class NameRoles:
    """Per-producer printed range words; ``members`` is ``E._members`` with those words out of NAME tokens."""

    def __init__(self, selection, table):
        table = verify(table)
        if table['rule'] != RULE or table['registry_checksum'] != selection.registry.checksum:
            raise ValueError('Name-role table was built for another rule or registry')
        self.roles = {k: frozenset(v) for k, v in table['roles'].items()}
        self.table_checksum = table['checksum']
        self.checksum = digest({'rule': RULE, 'table': self.table_checksum})

    def describe(self):
        return {'rule': RULE['version'], 'table_checksum': self.table_checksum, 'checksum': self.checksum,
                'producers': len(self.roles), 'tokens': sum(map(len, self.roles.values()))}

    def members(self, candidate, context):
        members, keys = E._members(candidate, context)
        cards = candidate['provenance']['cards']
        removed = {}
        for slug, m in members.items():
            words = self.roles.get(P._producer_key(cards[slug], slug), frozenset())
            drop = [t for t in m['core'] if t.skeletons[0] in words]
            if drop:
                m['name'] = [t for t in m['name'] if not any(t is d for d in drop)]
                m['core'] = [t for t in m['core'] if not any(t is d for d in drop)]
                removed[slug] = [t.forms[0] for t in drop]
        return members, keys, removed


def build(base, v4, provenance, context, stats, control_slug, product_of, roles):
    """``E.build`` with NAME tokens from ``roles.members``; the parent itself when no pool card is affected.

    Returns rows, raw, query and the per-candidate removed words (empty when the parent ran).
    """
    members = {c['candidate_id']: roles.members(c, context) for c in base['candidates']}
    removed = {cid: m[2] for cid, m in members.items() if m[2]}
    if not removed:
        rows, raw, query = E.build(base, v4, provenance, context, stats, control_slug, product_of)
        return rows, raw, query, {}
    lines = V4._lines(base['query_evidence']['observations'])
    index = E._Index(lines)
    if len(provenance) != len(base['query_evidence']['observations']) or any(r is None for r in provenance):
        raise ValueError('Provenance side-car does not cover the observation packet')
    grades = E._line_grades(lines, provenance)
    typed = E.E1.field_evidence(base, context)['candidates']
    candidates = v4['candidates']
    if [c['candidate_id'] for c in candidates] != [c['candidate_id'] for c in base['candidates']]:
        raise ValueError('v4 and base pools differ')
    per, keys, products = {}, {}, {}
    for c in base['candidates']:
        per[c['candidate_id']], keys[c['candidate_id']], _ = members[c['candidate_id']]
        products[c['candidate_id']] = {product_of(s) for s in c['card_slugs']}
    producer = {}
    for cid, members_ in per.items():
        reads = [dict(E._coverage(index, grades, text.tokenize(form), stats.producer_weight), form=form)
                 for m in members_.values() for form in m['producers']]
        producer[cid] = max(reads, key=E._rank, default=None) or dict(E._coverage(index, grades, [], stats.producer_weight), form=None)
    claimed = {cid: {t.skeletons[0] for m in members_.values() for t in m['name']} for cid, members_ in per.items()}
    rows, raw = [], []
    for c in candidates:
        cid, v4f = c['candidate_id'], c['features']
        f = {n: float(v4f[n]) for n in E.VISUAL}
        f['pool.control_proposal'] = float(control_slug in c['card_slugs'])
        own = producer[cid]
        f['prod.read_w'] = E._cap(own['read_w'])
        for r in E.READERS:
            f[f'prod.read_g.{r}'] = E._cap(own[f'read_g.{r}'])
        f['prod.phrase'] = float(v4f['txt.producer.full_any'])
        rivals = [o for o in per if o != cid and not products[o] & products[cid]]
        other = [producer[o] for o in rivals if keys[o] and not keys[o] & keys[cid]]
        rival_producer = {}
        for r in E.READERS:
            best = max(other, key=lambda x: x[f'read_g.{r}'], default=None)
            f[f'prod.other_g.{r}'] = max(0., E._cap(best[f'read_g.{r}']) - E._cap(own[f'read_g.{r}'])) if best else 0.
            rival_producer[r] = best and {'form': best['form'], 'read_g': best[f'read_g.{r}']}
        rival_tokens = set().union(*(claimed[o] for o in rivals)) if rivals else set()
        names, series = [], []
        for slug, m in per[cid].items():
            read = E._coverage(index, grades, m['name'], stats.name_weight)
            exclusive = sum(x['weight'] for x, t in zip(read['tokens'], m['name'])
                            if any(x['grades'].values()) and t.skeletons[0] not in rival_tokens)
            names.append(dict(read, slug=slug, exclusive=exclusive, atoms=[t.forms[0] for t in m['atoms']]))
            for tokens in m['series']:
                if tokens:
                    series.append(dict(E._coverage(index, grades, tokens, stats.name_weight), slug=slug))
        name = max(names, key=E._rank, default=None) or E._coverage(index, grades, [], stats.name_weight)
        f['name.read_w'] = E._cap(name['read_w'])
        for r in E.READERS:
            f[f'name.read_g.{r}'] = E._cap(name[f'read_g.{r}'])
        f['name.phrase'] = float(v4f['txt.name.full_any'])
        f['name.exclusive_w'] = E._cap(max((n['exclusive'] for n in names), default=0.))
        f['name.fuzzy_w'] = E._cap(max((n['fuzzy_w'] for n in names), default=0.))
        f['series.read_w'] = E._cap(max((x['read_w'] for x in series), default=0.))
        typed_raw, verified = {}, dict.fromkeys(E.READERS, 0.)
        for role in E.TYPED:
            t = typed[cid]['typed'][role]
            claims = [m['typed'][role] for m in per[cid].values()]
            contradicted = E._comparable(role, claims) if t['state']['contra_any'] else []
            sg = E._best_grade(t['support'], grades)
            cg = E._best_grade(t['contra'], grades) if contradicted else dict.fromkeys(E.READERS, 0.)
            for r in E.READERS:
                f[f'typ.{role}.support_g.{r}'] = sg[r]
                if role != 'grape_blend':
                    f[f'typ.{role}.contra_g.{r}'] = cg[r]
                if any(cl['status'] in E.VERIFIED for cl in contradicted):
                    verified[r] = max(verified[r], cg[r])
            typed_raw[role] = {'state': t['state'].get('kind'), 'support_lines': sorted(t['support']),
                               'contra_lines': sorted(t['contra']), 'disputed': t['disputed'],
                               'contradicted_claims': len(contradicted),
                               'claims': [{'keys': sorted({k['key'] for k in cl['keys']}), 'status': cl['status'],
                                           'source': cl['source']} for cl in claims]}
        for r in E.READERS:
            f[f'typ.contra_verified_g.{r}'] = verified[r]
        if set(f) != set(E.FEATURE_NAMES) or len(f) != len(E.FEATURE_NAMES) or not all(E.math.isfinite(v) for v in f.values()):
            raise ValueError('systemic-ranking-v2 feature invariant failed')
        rows.append({n: f[n] for n in E.FEATURE_NAMES})
        raw.append({'candidate_id': cid, 'producer_keys': sorted(keys[cid]), 'products': sorted(products[cid]),
                    'producer': {k: own.get(k) for k in ('form', 'cover', 'read_w', 'tokens', *(f'read_g.{r}' for r in E.READERS))},
                    'producer_rival': rival_producer,
                    'name': [{k: n[k] for k in ('slug', 'cover', 'read_w', 'exclusive', 'tokens', 'atoms',
                                                 *(f'read_g.{r}' for r in E.READERS))} for n in names],
                    'name_role_removed': removed.get(cid, {}),
                    'series': [{k: x[k] for k in ('slug', 'cover', 'tokens')} for x in series],
                    'typed': typed_raw, 'visual_ranks': c['v4_evidence']['visual_ranks']})
    query = {'lines': [{'line_id': l['line_id'], 'raw': l['raw'], 'grades': grades[l['line_id']],
                        'observation_ids': l['observation_ids'],
                        'sources': sorted({provenance[i]['reader'] + ':' + provenance[i]['crop_source']
                                           for i in l['observation_ids']})} for l in lines],
             'observed_typed': v4['query_evidence']['observed_typed'], 'small_label': v4['query_evidence']['small_label']}
    return rows, raw, query, removed


if __name__ == '__main__':
    import json
    from rshb_vine.io import seal, write_json
    ROOT = Path(__file__).resolve().parents[2]
    table = seal(build_table(ROOT, P.catalogue_selection(ROOT)))
    write_json(ROOT / TABLE, table)
    print(json.dumps({'table': TABLE, 'checksum': table['checksum'], 'candidates': len(table['candidates']),
                      'roles': table['roles']}, ensure_ascii=False))
