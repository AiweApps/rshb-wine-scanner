"""Symmetric per-candidate evidence for selector v4 over the actual frozen pool.

Every candidate is compared with the same OCR lines and its own catalogue claims;
nothing is computed relative to the 8175 answer, which stays provenance only.
Typed states keep support, contradiction, missing observation and unknown claim
apart. Catalogue claims are soft evidence with their status; there is no veto.
Standalone years never enter product features and are reported separately.
"""
from collections import defaultdict
from copy import deepcopy
import math

from rshb_vine.system_selection_v4 import text
from rshb_vine.system_selection_v4.context import RAW_FIELDS

SCHEMA_VERSION = 'system-selection-v4-features-v1'
CHANNELS = ('B0_label', 'B3_label', 'B0_context', 'B3_context')
TYPED_ROLES = ('grape_blend', 'color', 'sugar', 'style')
SINGLE_VALUED = ('color', 'sugar', 'style')
HIGH = .85
SMALL_LABEL_PX = 90
LABEL_VIEW_KINDS = ('detected_label', 'front_label')
FEATURE_NAMES = (
    *(f'vis.{c}.{f}' for c in CHANNELS for f in ('present', 'rr', 'gap')),
    'vis.support', 'vis.top1_count', 'vis.top3_count', 'vis.label_agree', 'vis.context_agree',
    'vis.b0_label_only_top1', 'vis.small_label_x_B0_label_rr', 'vis.small_label_x_B3_label_rr',
    'txt.producer.full_any', 'txt.producer.full_high', 'txt.producer.cover_any', 'txt.producer.cover_high',
    'txt.producer.other_read_high',
    'txt.name.full_any', 'txt.name.full_high', 'txt.name.cover_any', 'txt.name.cover_high',
    'txt.name.distinct_any', 'txt.name.distinct_high', 'txt.name.fuzzy_only', 'txt.name.distinct_unread',
    'txt.series.full_any', 'txt.series.full_high', 'txt.series.cover_any', 'txt.series.cover_high',
    *(f'typ.{r}.{f}' for r in TYPED_ROLES
      for f in ('support_any', 'support_high', 'contra_any', 'contra_high', 'unknown_when_observed')),
    'pool.extra_source_rr',
)
# Sign constraints of the fit (weight >= 0 or <= 0 in normalized space); others are free.
MONOTONE = {
    **{f'vis.{c}.rr': 1 for c in CHANNELS}, **{f'vis.{c}.gap': -1 for c in CHANNELS},
    'vis.support': 1, 'vis.top1_count': 1, 'vis.top3_count': 1, 'vis.label_agree': 1, 'vis.context_agree': 1,
    **{n: 1 for n in FEATURE_NAMES if n.startswith(('txt.', 'typ.')) and any(
        n.endswith(s) for s in ('full_any', 'full_high', 'cover_any', 'cover_high', 'distinct_any',
                                'distinct_high', 'support_any', 'support_high'))},
    **{n: -1 for n in FEATURE_NAMES if n.endswith(('contra_any', 'contra_high'))},
    'txt.producer.other_read_high': -1, 'txt.name.distinct_unread': -1, 'txt.name.fuzzy_only': -1,
}
KIND_RANK = {kind: i for i, kind in enumerate(text.MATCH_ORDER)}


def _lines(observations):
    groups = {}
    for index, observation in enumerate(observations):
        raw = observation.get('raw_text', '')
        tokens = text.tokenize(raw)
        score = observation.get('score')
        high = isinstance(score, (int, float)) and not isinstance(score, bool) and score >= HIGH
        key = ' '.join(t.forms[0] for t in tokens)
        years = [t.forms[0] for t in text.tokenize(raw, drop_years=False) if text.is_year(t.forms[0])]
        if not tokens and not years:
            continue
        line = groups.setdefault(key, {'tokens': tokens, 'observation_ids': [], 'high': False,
                                       'raw': [], 'max_score': None, 'years': set()})
        line['observation_ids'].append(index)
        line['raw'].append(raw)
        line['high'] |= high
        line['years'].update(years)
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            line['max_score'] = score if line['max_score'] is None else max(line['max_score'], score)
    return [dict(line, line_id=i, years=sorted(line['years'])) for i, line in enumerate(groups.values())]


class _QueryIndex:
    def __init__(self, lines):
        self.lines = lines
        self.exact, self.skeleton, self.by_initial = defaultdict(list), defaultdict(list), defaultdict(list)
        for line in lines:
            for pos, token in enumerate(line['tokens']):
                place = (line['line_id'], pos)
                for form in token.forms:
                    self.exact[form].append(place)
                for skeleton in token.skeletons:
                    self.skeleton[skeleton].append(place)
                    if skeleton:
                        self.by_initial[skeleton[0]].append(place)
        self.cache = {}

    def token(self, claim_norm):
        """Best reading of one claim token: {'kind', 'high', 'line_id', 'pos'} per confidence tier."""
        if claim_norm in self.cache:
            return self.cache[claim_norm]
        skeleton = text.skeleton(claim_norm)
        places = set(self.exact.get(claim_norm, ())) | set(self.skeleton.get(skeleton, ()))
        if skeleton and len(skeleton) >= 5:
            places |= {p for p in self.by_initial.get(skeleton[0], ())
                       if abs(len(self.lines[p[0]]['tokens'][p[1]].skeletons[0]) - len(skeleton)) <= 2}
        best = {}
        for line_id, pos in places:
            kind = text.token_match(self.lines[line_id]['tokens'][pos], claim_norm)
            if kind is None:
                continue
            tier = 'high' if self.lines[line_id]['high'] else 'low'
            found = {'kind': kind, 'line_id': line_id, 'pos': pos}
            for key in {tier, 'any'}:
                if key not in best or KIND_RANK[kind] < KIND_RANK[best[key]['kind']]:
                    best[key] = found
        self.cache[claim_norm] = best
        return best

    def phrase(self, claim_tokens):
        norms = tuple(t.forms[0] for t in claim_tokens)
        result = {}
        if not norms:
            return result
        for line in self.lines:
            found = text.phrase_match(line['tokens'], norms)
            if found is None:
                continue
            tier = 'high' if line['high'] else 'low'
            for key in {tier, 'any'}:
                if key not in result or KIND_RANK[found['weakest']] < KIND_RANK[result[key]['weakest']]:
                    result[key] = dict(found, line_id=line['line_id'])
        return result


def _raw_text(item):
    raw = item.get('raw')
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # source.raw is the catalogue text; value is sometimes already a normalized key.
        for value in (raw.get('source', {}).get('raw'), raw.get('value')):
            if isinstance(value, str) and value:
                return value
    return None


def _claim_values(profile, role):
    """Name roles use the extracted normalized phrases; typed roles use the raw catalogue text."""
    claim = (profile or {}).get('roles', {}).get(role, {})
    values = []
    for item in claim.get('claims', []):
        if role in TYPED_ROLES:
            value = _raw_text(item)
            values += [value] if value else []
        else:
            normalized = item.get('normalized')
            values += [normalized] if isinstance(normalized, str) else [v for v in normalized or [] if isinstance(v, str)]
    return values, claim.get('status', 'missing'), bool(claim.get('unknown', True))


def _producer_forms(card, profile, context):
    raw = card.get('raw_metadata', {}).get('Винодельня', '')
    forms = set(_claim_values(profile, 'manufacturer')[0]) | set(_claim_values(profile, 'visible_brand')[0])
    forms |= context.producer_aliases.get(text.base_norm(raw), set())
    if raw:
        forms.add(raw)
    return sorted(f for f in forms if text.tokenize(f))


def _name_tokens(values, producer_tokens, context):
    tokens, seen = [], set()
    for value in values:
        for token in text.tokenize(value):
            norm = token.forms[0]
            if norm in context.generic or norm in seen or token.skeletons[0] in producer_tokens:
                continue
            seen.add(norm)
            tokens.append(token)
    return tokens


def _typed_claim(role, slug, card, profile, context):
    """Claim keys in the lexicon key space, with where they came from."""
    values, status, unknown = _claim_values(profile, role)
    keys, source = [], 'normalized_claim'
    if not unknown:
        for value in values:
            keys += [dict(k, raw_value=value) for k in context.lexicon.claim_keys(role, value)]
    if role == 'sugar' and slug in context.sugar_facts:
        fact = context.sugar_facts[slug]
        keys = [{'key': fact['value'], 'raw_piece': fact['value'], 'mapping': 'verified_sugar_ledger'}]
        status, source = 'source_verified', 'sugar_ledger'
    if status in ('conflict', 'unresolved', 'ambiguous'):
        return {'keys': [], 'status': status, 'source': source, 'disputed_keys': keys}
    if not keys:
        for field in RAW_FIELDS[role]:
            value = card.get('raw_metadata', {}).get(field, '')
            if value:
                derived = context.lexicon.claim_keys(role, value)
                if role in SINGLE_VALUED and len({k['key'] for k in derived}) > 1:
                    derived = []
                keys += [dict(k, raw_value=value, field=field) for k in derived]
            if keys:
                status, source = 'catalog_card_field_derived', 'card_raw_metadata'
                break
    return {'keys': keys, 'status': status, 'source': source}


def _observed_typed(lines, lexicon, explained):
    """Typed readings per tier; tokens inside a read producer phrase (Красная Горка) are not typed evidence."""
    observed = {role: {'high': {}, 'low': {}} for role in TYPED_ROLES}
    for line in lines:
        tier = 'high' if line['high'] else 'low'
        for role, found in lexicon.entities(line['tokens']).items():
            if role not in observed:
                continue
            for item in found:
                span = {(line['line_id'], i) for i in range(item['start'], item['end'])}
                if span & explained or role in SINGLE_VALUED and item['match_kind'] == 'fuzzy':
                    continue
                observed[role][tier].setdefault(item['key'], []).append(
                    {'line_id': line['line_id'], 'form': item['form'], 'match_kind': item['match_kind'],
                     'mapping': item.get('mapping')})
    for role in TYPED_ROLES:
        for key in set(observed[role]['high']) & set(observed[role]['low']):
            observed[role]['low'].pop(key)
    return observed


def _key_tokens(lexicon):
    """Token sets of every grape key, so a partial reading (Красностоп) meets its fuller claim."""
    result = defaultdict(list)
    for _, mapping in lexicon.alias_sources['grape_blend']:
        for key, aliases in mapping.items():
            for alias in aliases:
                tokens = frozenset(t.skeletons[0] for t in text.tokenize(alias))
                if tokens:
                    result[key].append(tokens)
    return result


def _grape_compatible(a, b, key_tokens):
    def forms(key):
        return key_tokens.get(key) or ([frozenset(key[4:].split('_'))] if key.startswith('raw:') else [])
    return a == b or any(x <= y or y <= x for x in forms(a) for y in forms(b))


def _typed_state(role, observed, claims, key_tokens):
    """support/contradiction per tier; contradiction needs a comparable claim and no support."""
    high, low = set(observed['high']), set(observed['low'])
    claim_sets = [{k['key'] for k in c['keys']} for c in claims if c['keys']]
    if role == 'grape_blend':
        # A reading compatible with a claimed grape by token containment counts as that claim.
        high = {next((k for keys in claim_sets for k in keys if _grape_compatible(o, k, key_tokens)), o) for o in high}
        low = {next((k for keys in claim_sets for k in keys if _grape_compatible(o, k, key_tokens)), o) for o in low}
    state = {'support_any': 0., 'support_high': 0., 'contra_any': 0., 'contra_high': 0.,
             'unknown_when_observed': float(bool(high | low) and not claim_sets)}
    if not claim_sets or not (high | low):
        state['kind'] = 'unknown_claim' if claim_sets == [] and (high | low) else 'not_observed'
        return state
    supported_high = any(keys & high for keys in claim_sets)
    supported_any = supported_high or any(keys & low for keys in claim_sets)

    # Catalogue raw forms (Олег, Молдова, Цветочный) may be ordinary words: support only.
    curated = {tier: {k for k, hits in observed[tier].items() if any(h.get('mapping') == 'alias' for h in hits)}
               for tier in ('high', 'low')}
    if role == 'grape_blend':
        curated = {tier: {next((k for keys in claim_sets for k in keys if _grape_compatible(o, k, key_tokens)), o)
                          for o in values} for tier, values in curated.items()}

    def contradicts(seen):
        if not seen or (role in SINGLE_VALUED and len(seen) != 1):
            return False
        return all(seen.isdisjoint(keys) for keys in claim_sets)

    contra_high = not supported_any and contradicts(curated['high'])
    contra_any = contra_high or (not supported_any and not curated['high'] and contradicts(curated['low']))
    state.update(support_any=float(supported_any), support_high=float(supported_high),
                 contra_any=float(contra_any), contra_high=float(contra_high),
                 kind='support' if supported_any else 'contradiction' if contra_any else 'ambiguous_observation')
    return state


def _label_min_side(raw_visual, arm):
    rows = raw_visual.get('arms', {}).get(arm, {}).get('retrieval', {}).get('channel_top20', {}).get('front_label', [])
    views = raw_visual.get('views', [])
    index = rows[0].get('query_view_index') if rows else None
    if type(index) is not int or not 0 <= index < len(views) or views[index].get('kind') not in LABEL_VIEW_KINDS:
        return None
    box = views[index].get('bbox')
    if not isinstance(box, list) or len(box) != 4:
        return None
    return float(min(box[2] - box[0], box[3] - box[1]))


def build_features(base, raw_visual, context, cards, claims, extra_candidates=None):
    """v4 feature output over the frozen pool of ``base`` (build_candidate_features output).

    ``extra_candidates``: optional [{'slug','source','rank','score','evidence'}] from a separately
    admitted candidate generator. A product already in the pool only gains provenance; a new
    product enters with zero visual evidence and ``publish_requires_extension=True``.
    """
    lines = _lines(base['query_evidence']['observations'])
    index = _QueryIndex(lines)
    channels = {}
    for arm in ('B0', 'B3'):
        top20 = raw_visual.get('arms', {}).get(arm, {}).get('retrieval', {}).get('channel_top20', {})
        for kind, suffix in (('front_label', 'label'), ('context', 'context')):
            channels[arm + '_' + suffix] = top20.get(kind, [])
    sizes = {arm: _label_min_side(raw_visual, arm) for arm in ('B0', 'B3')}
    small = {arm: float(size is not None and size < SMALL_LABEL_PX) for arm, size in sizes.items()}
    candidates = [dict(c) for c in base['candidates']]
    by_slug = {s: c for c in candidates for s in c['card_slugs']}
    for item in extra_candidates or ():
        slug = item['slug']
        record = {k: deepcopy(item.get(k)) for k in ('source', 'rank', 'score', 'evidence')} | {'slug': slug}
        if slug not in cards:
            raise ValueError('Extra candidate absent from identity registry: ' + slug)
        card = cards[slug]
        key = card['product_id'] if card['binding_status'] == 'source_admitted_product' else 'unresolved-card:' + slug
        existing = by_slug.get(slug) or next((c for c in candidates if c['candidate_id'] == key), None)
        if existing is not None:
            existing['extra_sources'] = [*existing.get('extra_sources', []), record]
            continue
        new = {'candidate_id': key, 'product_id': card['product_id'], 'identity_status': card['binding_status'],
               'card_slugs': [slug], 'representative_slug': slug, 'extra_sources': [record],
               'publish_requires_extension': True,
               'provenance': {'visual': [], 'control_candidates': [], 'control_proposal_slug': None,
                              'cards': {slug: deepcopy(card)}, 'claims': {slug: deepcopy(claims.get(slug))},
                              'geometry': {}}}
        candidates.append(new)
        by_slug[slug] = new
    per = {}
    producer_keys = {}
    for candidate in candidates:
        member_cards, member_claims = candidate['provenance']['cards'], candidate['provenance']['claims']
        members = {}
        for slug in candidate['card_slugs']:
            card, profile = member_cards[slug], member_claims.get(slug)
            producers = _producer_forms(card, profile, context)
            producer_tokens = {t.skeletons[0] for f in producers for t in text.tokenize(f)}
            name_values = _claim_values(profile, 'commercial_name')[0] or [card.get('raw_metadata', {}).get('Название вина', '')]
            series_values = _claim_values(profile, 'series')[0]
            members[slug] = {'producers': producers,
                             'name': _name_tokens(name_values, producer_tokens, context),
                             'series': [_name_tokens([v], producer_tokens, context) for v in series_values],
                             'typed': {role: _typed_claim(role, slug, card, profile, context) for role in TYPED_ROLES}}
        per[candidate['candidate_id']] = members
        producer_keys[candidate['candidate_id']] = {text.base_norm(member_cards[s].get('raw_metadata', {}).get('Винодельня', ''))
                                                    for s in candidate['card_slugs']} - {''}
    siblings = defaultdict(set)
    for cid, keys in producer_keys.items():
        for key in keys:
            siblings[key].add(cid)

    def name_set(cid):
        return {t.skeletons[0] for m in per[cid].values() for t in m['name']}

    typed_atoms = {}
    for cid, members in per.items():
        atoms = set()
        for m in members.values():
            for item in (x for found in context.lexicon.entities(tuple(m['name'])).values() for x in found):
                atoms.update(t.skeletons[0] for t in m['name'][item['start']:item['end']])
        typed_atoms[cid] = atoms
    producer_reads = {}
    for cid, members in per.items():
        best = {}
        for m in members.values():
            for form in m['producers']:
                found = index.phrase(text.tokenize(form))
                for tier, hit in found.items():
                    if tier not in best or KIND_RANK[hit['weakest']] < KIND_RANK[best[tier]['weakest']]:
                        best[tier] = dict(hit, form=form)
        producer_reads[cid] = best
    high_producers = {k for cid, reads in producer_reads.items() if 'high' in reads for k in producer_keys[cid]}
    explained = set()
    for form in sorted({f for members in per.values() for m in members.values() for f in m['producers']}):
        tokens = tuple(t.forms[0] for t in text.tokenize(form))
        for line in lines:
            found = text.phrase_match(line['tokens'], tokens) if tokens else None
            if found:
                explained |= {(line['line_id'], i) for i in range(found['start'], found['end'])}
    observed = _observed_typed(lines, context.lexicon, explained)
    key_tokens = getattr(context, 'grape_key_tokens', None)
    if key_tokens is None:
        key_tokens = context.grape_key_tokens = _key_tokens(context.lexicon)
    output = []
    for candidate in candidates:
        cid = candidate['candidate_id']
        provenance = candidate['provenance']
        f = dict.fromkeys(FEATURE_NAMES, 0.)
        visual = {}
        for channel in CHANNELS:
            rows = channels[channel]
            hits = [e for e in provenance['visual'] if e['channel'] == channel]
            if not rows:
                continue
            top = float(rows[0]['score'])
            if hits:
                best = min(hits, key=lambda e: e['rank'])
                visual[channel] = best['rank']
                f[f'vis.{channel}.present'] = 1.
                f[f'vis.{channel}.rr'] = 1. / best['rank']
                f[f'vis.{channel}.gap'] = top - float(best['raw']['score'])
            else:
                f[f'vis.{channel}.gap'] = top - float(rows[-1]['score'])
        ranks = list(visual.values())
        f['vis.support'] = len(ranks) / 4
        f['vis.top1_count'] = sum(r == 1 for r in ranks) / 4
        f['vis.top3_count'] = sum(r <= 3 for r in ranks) / 4
        f['vis.label_agree'] = float(visual.get('B0_label') == 1 and visual.get('B3_label') == 1)
        f['vis.context_agree'] = float(visual.get('B0_context') == 1 and visual.get('B3_context') == 1)
        f['vis.b0_label_only_top1'] = float(visual.get('B0_label') == 1 and not any(
            r <= 3 for c, r in visual.items() if c != 'B0_label'))
        f['vis.small_label_x_B0_label_rr'] = small['B0'] * f['vis.B0_label.rr']
        f['vis.small_label_x_B3_label_rr'] = small['B3'] * f['vis.B3_label.rr']
        members = per[cid]
        reads = producer_reads[cid]
        f['txt.producer.full_any'] = float('any' in reads)
        f['txt.producer.full_high'] = float('high' in reads)
        producer_cover = {'any': 0., 'high': 0.}
        for m in members.values():
            for form in m['producers']:
                tokens = text.tokenize(form)
                for tier in producer_cover:
                    got = sum(tier in index.token(t.forms[0]) for t in tokens)
                    producer_cover[tier] = max(producer_cover[tier], got / len(tokens))
        f['txt.producer.cover_any'], f['txt.producer.cover_high'] = producer_cover['any'], producer_cover['high']
        f['txt.producer.other_read_high'] = float(bool(high_producers - producer_keys[cid]) and 'any' not in reads)
        own = name_set(cid)
        family = set().union(*(siblings[k] for k in producer_keys[cid])) if producer_keys[cid] else {cid}
        if len(family) > 1:
            common = set.intersection(*(name_set(other) for other in family))
            distinct = own - common
        else:
            distinct = own - typed_atoms[cid]
        name_trace = {'distinct_tokens': sorted(distinct), 'family_size': len(family), 'members': {}}
        best_name = {'full_any': 0., 'full_high': 0., 'cover_any': 0., 'cover_high': 0.}
        matched_distinct = {'any': set(), 'high': set()}
        kinds, high_name_hits = set(), 0
        for slug, m in members.items():
            tokens = m['name']
            if not tokens:
                continue
            reads_name = index.phrase(tokens)
            hits = {t.forms[0]: index.token(t.forms[0]) for t in tokens}
            best_name['full_any'] = max(best_name['full_any'], float('any' in reads_name))
            best_name['full_high'] = max(best_name['full_high'], float('high' in reads_name))
            for tier in ('any', 'high'):
                best_name['cover_' + tier] = max(best_name['cover_' + tier],
                                                 sum(tier in h for h in hits.values()) / len(tokens))
            for t in tokens:
                hit = hits[t.forms[0]]
                if 'any' in hit:
                    kinds.add(hit['any']['kind'])
                if t.skeletons[0] in distinct:
                    for tier in ('any', 'high'):
                        if tier in hit:
                            matched_distinct[tier].add(t.skeletons[0])
            high_name_hits = max(high_name_hits, sum('high' in h for h in hits.values()))
            name_trace['members'][slug] = {'tokens': [t.forms[0] for t in tokens],
                                           'reads': {k: v for k, v in hits.items() if v},
                                           'full_phrase': reads_name}
        for key, value in best_name.items():
            f['txt.name.' + key] = value
        f['txt.name.distinct_any'] = min(len(matched_distinct['any']), 3) / 3
        f['txt.name.distinct_high'] = min(len(matched_distinct['high']), 3) / 3
        f['txt.name.fuzzy_only'] = float(bool(kinds) and kinds == {'fuzzy'})
        well_read = 'any' in reads or high_name_hits >= 2
        f['txt.name.distinct_unread'] = float(bool(distinct) and not matched_distinct['any'] and well_read)
        series_trace = {}
        for slug, m in members.items():
            for tokens in m['series']:
                if not tokens:
                    continue
                reads_series = index.phrase(tokens)
                f['txt.series.full_any'] = max(f['txt.series.full_any'], float('any' in reads_series))
                f['txt.series.full_high'] = max(f['txt.series.full_high'], float('high' in reads_series))
                for tier in ('any', 'high'):
                    f['txt.series.cover_' + tier] = max(f['txt.series.cover_' + tier], sum(
                        tier in index.token(t.forms[0]) for t in tokens) / len(tokens))
                series_trace.setdefault(slug, []).append({'tokens': [t.forms[0] for t in tokens],
                                                         'full_phrase': reads_series})
        typed_trace = {}
        for role in TYPED_ROLES:
            claims = [m['typed'][role] for m in members.values()]
            state = _typed_state(role, observed[role], claims, key_tokens)
            for key in ('support_any', 'support_high', 'contra_any', 'contra_high', 'unknown_when_observed'):
                f[f'typ.{role}.{key}'] = state[key]
            typed_trace[role] = {'state': state['kind'], 'claims': dict(zip(members, claims)),
                                 'observed_high': sorted(observed[role]['high']),
                                 'observed_low': sorted(observed[role]['low'])}
        extra = candidate.get('extra_sources', [])
        ranks = [e['rank'] for e in extra if type(e.get('rank')) is int and e['rank'] >= 1]
        f['pool.extra_source_rr'] = 1. / min(ranks) if ranks else 0.
        if not all(math.isfinite(v) for v in f.values()) or tuple(f) != FEATURE_NAMES:
            raise ValueError('v4 feature vector invariant failed')
        output.append({'candidate_id': cid, 'product_id': candidate['product_id'],
                       'identity_status': candidate['identity_status'], 'card_slugs': list(candidate['card_slugs']),
                       'representative_slug': candidate['representative_slug'], 'features': f,
                       'provenance': provenance, 'extra_sources': extra,
                       'publish_requires_extension': bool(candidate.get('publish_requires_extension')),
                       'v4_evidence': {'visual_ranks': visual, 'producer': {'forms': sorted({x for m in members.values() for x in m['producers']}),
                                                                             'reads': reads, 'keys': sorted(producer_keys[cid])},
                                       'name': name_trace, 'series': series_trace, 'typed': typed_trace,
                                       'hard_veto': False}})
    query = deepcopy({k: v for k, v in base['query_evidence'].items() if k != 'aligned_readings'})
    query.update(lines=[{'line_id': l['line_id'], 'tokens': [t.forms[0] for t in l['tokens']],
                         'observation_ids': l['observation_ids'], 'high': l['high'], 'max_score': l['max_score'],
                         'raw': l['raw']} for l in lines],
                 observed_typed={role: {tier: sorted(v) for tier, v in observed[role].items()} for role in TYPED_ROLES},
                 observed_years=sorted({y for l in lines for y in l['years']}),
                 label_min_side_px=sizes, small_label=small, confidence_threshold=HIGH,
                 confidence='reader_specific_uncalibrated')
    return {'schema_version': SCHEMA_VERSION, 'feature_names': list(FEATURE_NAMES), 'candidates': output,
            'query_evidence': query, 'pool_policy': 'frozen_visual_4xTop20_plus_supplied8175' + (
                '_plus_extra_sources' if extra_candidates else ''),
            'control_role': 'provenance_and_card_choice_only_not_a_score_feature',
            'vintage_in_product_features': False, 'is_selection': False, 'hard_veto_used': False, 'probability': None}
