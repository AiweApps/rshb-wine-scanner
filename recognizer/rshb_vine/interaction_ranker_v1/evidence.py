"""Per-field OCR support with reader/crop provenance for every candidate of the actual runtime pool.

Matching reuses the selector-v4 matchers on the same grouped lines, so support/contradiction states equal
the v4 features. The aggregation differs: every supporting line is kept, and each line is mapped through
the OCR provenance side-car to readers, crop sources and native scores. No 0.85 tier is used here;
Vision's quantized native grades (0.3/0.5/1.0) and Paddle's continuous score stay separate per reader.
A line read twice by the same reader on the same crop is one source. Standalone years never enter.
"""
from collections import defaultdict

from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text

READER_TAGS = {'apple_vision': 'vision_r3_macos', 'paddle_ppocrv5_cpu': 'paddle_v5_cpu'}
READER_SCOPE = {
    'vision_r3_macos': 'Apple Vision revision 3, accurate, ru-RU/en-US, quantized native confidence; verified on the '
                       'current macOS runtime only (M1 Max, macOS 26.3); no Linux quality claim',
    'paddle_v5_cpu': 'PaddleOCR PP-OCRv5 CPU sparse alternative read, continuous uncalibrated native score',
}
TAGS = tuple(READER_TAGS.values())
TEXT_FIELDS = ('producer', 'name', 'name_distinct', 'series')
TYPED = V4.TYPED_ROLES
DISPUTED = ('conflict', 'unresolved', 'ambiguous')
VISUAL = (*(f'vis.{c}.{k}' for c in V4.CHANNELS for k in ('rr', 'gap')), 'vis.top1_count', 'vis.support',
          'vis.label_agree', 'vis.context_agree', 'vis.small_label_x_B0_label_rr', 'vis.small_label_x_B3_label_rr')
FEATURE_NAMES = (
    *VISUAL,
    *(f'txt.{f}.{k}' for f in TEXT_FIELDS for k in ('support', 'sources', 'exclusive', *(f'best.{t}' for t in TAGS))),
    'txt.producer.other_read', *(f'txt.producer.other.best.{t}' for t in TAGS), 'txt.name.distinct_unread',
    *(f'typ.{r}.{k}' for r in TYPED for k in (
        'support', 'contra', 'unknown_claim', 'disputed_claim', 'support.sources', 'support.exclusive',
        *(f'support.best.{t}' for t in TAGS), *(f'contra.best.{t}' for t in TAGS))),
    'q.sources', 'q.small_label',
)
SCHEMA_VERSION = 'interaction-ranker-v1-features-v1'


class _Index(V4._QueryIndex):
    def token_lines(self, claim_norm):
        """Every line holding a token that matches ``claim_norm`` (same candidate places as v4 ``token``)."""
        skeleton = text.skeleton(claim_norm)
        places = set(self.exact.get(claim_norm, ())) | set(self.skeleton.get(skeleton, ()))
        if skeleton and len(skeleton) >= 5:
            places |= {p for p in self.by_initial.get(skeleton[0], ())
                       if abs(len(self.lines[p[0]]['tokens'][p[1]].skeletons[0]) - len(skeleton)) <= 2}
        return {line_id for line_id, pos in places
                if text.token_match(self.lines[line_id]['tokens'][pos], claim_norm) is not None}

    def phrase_lines(self, claim_tokens):
        norms = tuple(t.forms[0] for t in claim_tokens)
        if not norms:
            return set()
        return {line['line_id'] for line in self.lines if text.phrase_match(line['tokens'], norms) is not None}


def _members(candidate, context):
    member_cards, member_claims = candidate['provenance']['cards'], candidate['provenance']['claims']
    members = {}
    for slug in candidate['card_slugs']:
        card, profile = member_cards[slug], member_claims.get(slug)
        producers = V4._producer_forms(card, profile, context)
        producer_tokens = {t.skeletons[0] for f in producers for t in text.tokenize(f)}
        name_values = V4._claim_values(profile, 'commercial_name')[0] or [card.get('raw_metadata', {}).get('Название вина', '')]
        series_values = V4._claim_values(profile, 'series')[0]
        members[slug] = {'producers': producers, 'name': V4._name_tokens(name_values, producer_tokens, context),
                         'series': [V4._name_tokens([v], producer_tokens, context) for v in series_values],
                         'typed': {role: V4._typed_claim(role, slug, card, profile, context) for role in TYPED}}
    keys = {text.base_norm(member_cards[s].get('raw_metadata', {}).get('Винодельня', '')) for s in candidate['card_slugs']} - {''}
    return members, keys


def _distinct(per, producer_keys, lexicon):
    """Distinct name skeletons per candidate, exactly as v4 derives them (family = same producer key)."""
    siblings = defaultdict(set)
    for cid, keys in producer_keys.items():
        for key in keys:
            siblings[key].add(cid)

    def name_set(cid):
        return {t.skeletons[0] for m in per[cid].values() for t in m['name']}

    result = {}
    for cid, members in per.items():
        family = set().union(*(siblings[k] for k in producer_keys[cid])) if producer_keys[cid] else {cid}
        if len(family) > 1:
            result[cid] = name_set(cid) - set.intersection(*(name_set(o) for o in family))
        else:
            atoms = set()
            for m in members.values():
                for item in (x for found in lexicon.entities(tuple(m['name'])).values() for x in found):
                    atoms.update(t.skeletons[0] for t in m['name'][item['start']:item['end']])
            result[cid] = name_set(cid) - atoms
    return result, siblings


def _typed_lines(role, observed, claims, key_tokens, state):
    claim_sets = [{k['key'] for k in c['keys']} for c in claims if c['keys']]
    claimed = set().union(*claim_sets) if claim_sets else set()

    def mapped(key):
        if role == 'grape_blend':
            return next((k for keys in claim_sets for k in keys if V4._grape_compatible(key, k, key_tokens)), key)
        return key

    support, contra = set(), set()
    for tier in ('high', 'low'):
        for key, hits in observed[tier].items():
            if mapped(key) in claimed:
                support |= {h['line_id'] for h in hits}
            elif state['contra_any'] and any(h.get('mapping') == 'alias' for h in hits):
                contra |= {h['line_id'] for h in hits if h.get('mapping') == 'alias'}
    return support, contra


def field_evidence(base, context):
    """{'lines', 'candidates': {cid: {field: {'support': set(line ids), ...}}}, 'v4_states'} over base['candidates']."""
    lines = V4._lines(base['query_evidence']['observations'])
    index = _Index(lines)
    per, producer_keys = {}, {}
    for c in base['candidates']:
        per[c['candidate_id']], producer_keys[c['candidate_id']] = _members(c, context)
    distinct, siblings = _distinct(per, producer_keys, context.lexicon)
    explained = set()
    for form in sorted({f for members in per.values() for m in members.values() for f in m['producers']}):
        tokens = tuple(t.forms[0] for t in text.tokenize(form))
        for line in lines:
            found = text.phrase_match(line['tokens'], tokens) if tokens else None
            if found:
                explained |= {(line['line_id'], i) for i in range(found['start'], found['end'])}
    observed = V4._observed_typed(lines, context.lexicon, explained)
    key_tokens = getattr(context, 'grape_key_tokens', None)
    if key_tokens is None:
        key_tokens = context.grape_key_tokens = V4._key_tokens(context.lexicon)
    producer_lines = {cid: set().union(*(index.phrase_lines(text.tokenize(f)) for m in members.values()
                                         for f in m['producers'])) for cid, members in per.items()}
    output = {}
    for cid, members in per.items():
        name_lines, distinct_lines = set(), set()
        for m in members.values():
            for t in m['name']:
                found = index.token_lines(t.forms[0])
                name_lines |= found
                if t.skeletons[0] in distinct[cid]:
                    distinct_lines |= found
        series_lines = set().union(*(index.phrase_lines(tokens) for m in members.values() for tokens in m['series'] if tokens))
        own = producer_lines[cid]
        other = set() if own else set().union(*(producer_lines[o] for o in per
                                               if o != cid and producer_keys[o] - producer_keys[cid]))
        record = {'producer': {'support': own}, 'name': {'support': name_lines},
                  'name_distinct': {'support': distinct_lines}, 'series': {'support': series_lines},
                  'producer_other': other, 'has_distinct': bool(distinct[cid]),
                  'producer_keys': sorted(producer_keys[cid]), 'typed': {}}
        for role in TYPED:
            claims = [m['typed'][role] for m in members.values()]
            state = V4._typed_state(role, observed[role], claims, key_tokens)
            support, contra = _typed_lines(role, observed[role], claims, key_tokens, state)
            record['typed'][role] = {'support': support, 'contra': contra, 'state': state,
                                     'disputed': any(c['status'] in DISPUTED for c in claims)}
        output[cid] = record
    return {'lines': lines, 'candidates': output, 'siblings': {k: sorted(v) for k, v in siblings.items()}}


def _sources(line_ids, lines, provenance):
    best, sources = dict.fromkeys(TAGS, 0.), set()
    for line_id in line_ids:
        for i in lines[line_id]['observation_ids']:
            record = provenance[i]
            tag = READER_TAGS[record['reader']]
            score = record['native_score']
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                best[tag] = max(best[tag], float(score))
            sources.add((record['reader'], record['crop_source']))
    return best, sources


def _exclusive(cid, field_lines, n):
    """1 - share of other pool candidates supported by any of the same lines; 0 without support."""
    own = field_lines[cid]
    if not own or n <= 1:
        return 0.
    shared = sum(1 for other, lines in field_lines.items() if other != cid and lines & own)
    return 1. - shared / (n - 1)


def candidate_features(v4_output, evidence, provenance):
    """Feature dicts aligned with ``v4_output['candidates']``; ``provenance[i]`` describes observation i."""
    lines = evidence['lines']
    if len(provenance) != len(v4_output['query_evidence']['observations']) or any(r is None for r in provenance):
        raise ValueError('Provenance side-car does not cover the observation packet')
    candidates = v4_output['candidates']
    n = len(candidates)
    per = evidence['candidates']
    text_lines = {f: {c['candidate_id']: per[c['candidate_id']][f]['support'] for c in candidates} for f in TEXT_FIELDS}
    typed_lines = {r: {c['candidate_id']: per[c['candidate_id']]['typed'][r]['support'] for c in candidates} for r in TYPED}
    packet_sources = {(p['reader'], p['crop_source']) for p in provenance}
    small = float(any(v4_output['query_evidence'].get('small_label', {}).values()))
    rows = []
    for c in candidates:
        cid, v4f, e = c['candidate_id'], c['features'], per[c['candidate_id']]
        f = {name: float(v4f[name]) for name in VISUAL}
        for field in TEXT_FIELDS:
            best, sources = _sources(e[field]['support'], lines, provenance)
            f[f'txt.{field}.support'] = float(bool(e[field]['support']))
            f[f'txt.{field}.sources'] = float(min(len(sources), 3))
            f[f'txt.{field}.exclusive'] = _exclusive(cid, text_lines[field], n)
            for tag in TAGS:
                f[f'txt.{field}.best.{tag}'] = best[tag]
        best, _ = _sources(e['producer_other'], lines, provenance)
        f['txt.producer.other_read'] = float(bool(e['producer_other']))
        for tag in TAGS:
            f[f'txt.producer.other.best.{tag}'] = best[tag]
        f['txt.name.distinct_unread'] = float(e['has_distinct'] and not e['name_distinct']['support']
                                              and bool(e['producer']['support']))
        for role in TYPED:
            t = e['typed'][role]
            sbest, ssources = _sources(t['support'], lines, provenance)
            cbest, _ = _sources(t['contra'], lines, provenance)
            f[f'typ.{role}.support'] = t['state']['support_any']
            f[f'typ.{role}.contra'] = t['state']['contra_any']
            f[f'typ.{role}.unknown_claim'] = t['state']['unknown_when_observed']
            f[f'typ.{role}.disputed_claim'] = float(t['disputed'])
            f[f'typ.{role}.support.sources'] = float(min(len(ssources), 3))
            f[f'typ.{role}.support.exclusive'] = _exclusive(cid, typed_lines[role], n)
            for tag in TAGS:
                f[f'typ.{role}.support.best.{tag}'] = sbest[tag]
                f[f'typ.{role}.contra.best.{tag}'] = cbest[tag]
        f['q.sources'] = float(min(len(packet_sources), 3))
        f['q.small_label'] = small
        if set(f) != set(FEATURE_NAMES) or len(f) != len(FEATURE_NAMES):
            raise ValueError('interaction feature schema invariant failed')
        rows.append({n: f[n] for n in FEATURE_NAMES})
    return rows


def v4_consistency(v4_output, evidence):
    """Support/contradiction flags recomputed here must equal the v4 feature values of every candidate."""
    failures = []
    for c in v4_output['candidates']:
        f, e = c['features'], evidence['candidates'][c['candidate_id']]
        checks = {'producer': (f['txt.producer.full_any'] > 0) == bool(e['producer']['support']),
                  'name_distinct': (f['txt.name.distinct_any'] > 0) == bool(e['name_distinct']['support']),
                  'series': (f['txt.series.full_any'] > 0) == bool(e['series']['support'])}
        for role in TYPED:
            state = e['typed'][role]['state']
            checks[role] = all(f[f'typ.{role}.{k}'] == state[k] for k in
                               ('support_any', 'support_high', 'contra_any', 'contra_high', 'unknown_when_observed'))
            checks[role + '_lines'] = bool(e['typed'][role]['support']) == bool(state['support_any'])
        failures += [(c['candidate_id'], k) for k, ok in checks.items() if not ok]
    return failures
