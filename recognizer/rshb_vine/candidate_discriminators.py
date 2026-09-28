"""Source-preserving lexical detail over an actual frozen candidate pool.

Existing runtime aliases, no wine-specific additions or truth input. These are
soft candidate features, never proofs of identity or missing-word vetoes.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

from rshb_vine.learned_selection_features import FEATURE_NAMES as BASE_NAMES, SCHEMA_VERSION as BASE_SCHEMA
from rshb_vine.positive_variant_text import ALIASES
from rshb_vine.resolution.identity import COLOR, SUGAR, STYLE
from rshb_vine.gallery_variant_text import STYLE_ALIASES
from rshb_vine.frozen_candidate_selector import GENERIC
from rshb_vine.typed_catalog_lexicon import normalize, TypedCatalogLexicon
from rshb_vine.product_first.partial_evidence import NAMES, contains_tokens

SCHEMA_VERSION = 'candidate-discriminators-v2'
NAME_DETAILS = ('high_claim_coverage', 'high_distinctive_count',
                'low_distinctive_count', 'high_distinctive_pool_mass')
FEATURE_NAMES = (*BASE_NAMES,
    *(f'text_detail.{role}.{field}' for role in NAMES for field in NAME_DETAILS),
    'text_detail.producer_variant.high_support_count', 'text_detail.producer_variant.low_support_count')
SOURCE_PATHS = (
    'rshb_vine/candidate_discriminators.py', 'rshb_vine/learned_selection_features.py',
    'rshb_vine/positive_variant_text.py', 'rshb_vine/resolution/identity.py',
    'rshb_vine/gallery_variant_text.py', 'rshb_vine/frozen_candidate_selector.py',
    'rshb_vine/typed_catalog_lexicon.py', 'rshb_vine/product_first/partial_evidence.py',
    'rshb_vine/catalog_text_retrieval.py', 'config/producer-brand-aliases.json',
)

# Reuse the runtime's full public grape lexicon, including Muscat families.
# An empty catalogue means no learned/source-dependent name inventory is built;
# the curated general aliases are the same ones used by existing8175.
GRAPE_ALIASES = TypedCatalogLexicon([], Path(__file__).resolve().parents[1]).grapes


def _plain_tokens(text):
    # Product/year boundary: preserve other numbers, remove year-like tokens.
    return tuple(t for t in normalize(text).split()
                 if not (t.isdigit() and len(t) == 4 and 1800 <= int(t) <= 2099))


def _forms():
    rows = []
    # Prefer the specific style to the generic colour "white" for a phrase
    # such as blanc de blancs. Sugar/style share a semantic key when identical.
    for priority, source, mapping in ((0, 'positive_variant_text.ALIASES', ALIASES),
            (0, 'TypedCatalogLexicon.grapes', GRAPE_ALIASES),
            (0, 'identity.STYLE', STYLE), (0, 'gallery_variant_text.STYLE_ALIASES', STYLE_ALIASES),
            (1, 'identity.SUGAR', SUGAR), (2, 'identity.COLOR', COLOR)):
        for key, forms in mapping.items():
            for form in forms:
                phrase = _plain_tokens(form)
                if phrase:
                    rows.append((phrase, 'zz' + key.replace('_', '') + 'zz', source, priority))
    return tuple(sorted(rows, key=lambda row: (-len(row[0]), row[3], row[1], row[2], row[0])))


FORMS = _forms()
GENERIC_WORDS = frozenset(t for word in GENERIC for t in _plain_tokens(word))


@lru_cache(maxsize=16384)
def canonical_tokens(text):
    """Longest whole-token aliases with their original library provenance."""
    words = _plain_tokens(text)
    result, applied = [], []
    i = 0
    while i < len(words):
        match = next((row for row in FORMS if words[i:i + len(row[0])] == row[0]), None)
        if match is None:
            result.append(words[i])
            i += 1
        else:
            phrase, atom, source, _ = match
            result.append(atom)
            applied.append((i, phrase, atom, source))
            i += len(phrase)
    return tuple(result), tuple(applied)


def _distinctive(tokens):
    # Grape, sugar, colour, style and reserve stay in their own evidence,
    # not suddenly a uniquely identifying commercial name.
    return {t for t in tokens if len(t) >= 3 and not t.isdigit()
            and t not in GENERIC_WORDS and not (t.startswith('zz') and t.endswith('zz'))}


def enrich_candidate_features(base):
    if base.get('schema_version') != BASE_SCHEMA or base.get('feature_names') != list(BASE_NAMES):
        raise ValueError('Expected actual frozen85 feature output')
    result = deepcopy(base)
    result['schema_version'] = SCHEMA_VERSION
    result['feature_names'] = list(FEATURE_NAMES)
    observations = result['query_evidence']['observations']
    groups = {}
    for i, observation in enumerate(observations):
        tokens, applied = canonical_tokens(observation.get('raw_text', ''))
        if not tokens: continue
        score = observation.get('score')
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError('Invalid OCR confidence')
        high = isinstance(score, (int, float)) and score >= .85
        group = groups.setdefault(tokens, {'tokens': list(tokens), 'line_ids': [], 'high': False, 'aliases': []})
        group['line_ids'].append(i)
        group['high'] |= high
        group['aliases'].append({'line_id': i, 'mappings': applied})
    lines = list(groups.values())
    high_tokens = {t for line in lines if line['high'] for t in line['tokens']}
    low_tokens = {t for line in lines if not line['high'] for t in line['tokens']} - high_tokens
    role_candidates = {}
    for candidate in result['candidates']:
        roles = {}
        for role in NAMES:
            member_rows = []
            for slug, profile in candidate['provenance']['claims'].items():
                claim = (profile or {}).get('roles', {}).get(role, {})
                if not claim.get('comparison_allowed') or claim.get('status', 'unknown') == 'unknown': continue
                for value in claim.get('values', []):
                    tokens, applied = canonical_tokens(str(value))
                    if not tokens: continue
                    expected = set(tokens)
                    member_rows.append({'slug': slug, 'catalog_value': value, 'tokens': list(tokens),
                        'aliases': applied, 'catalog_status': claim.get('status'),
                        'expected_distinctive': sorted(_distinctive(tokens)),
                        'matched_high': sorted(expected & high_tokens),
                        'matched_low': sorted(expected & low_tokens),
                        'source_claims': deepcopy(claim.get('claims', [])),
                        'line_matches': [{'line_ids': line['line_ids'], 'high': line['high'],
                            'shared': sorted(expected & set(line['tokens'])),
                            'full_phrase': contains_tokens(line['tokens'], list(tokens))}
                            for line in lines if expected & set(line['tokens'])]})
            roles[role] = member_rows
        role_candidates[candidate['candidate_id']] = roles

    frequency = {role: Counter(t for roles in role_candidates.values()
        for t in {t for row in roles[role] for t in row['expected_distinctive']}) for role in NAMES}
    # An isolated Reserve coefficient has no positive examples in admitted
    # train65. The existing runtime already compares variant attributes within
    # a producer. Represent that general comparison (grape/style/reserve/etc.)
    # instead; attribute type and library source remain in lexical evidence.
    producers = {c['candidate_id']: {normalize(card.get('raw_metadata', {}).get('Винодельня', ''))
                 for card in c['provenance']['cards'].values()} - {''} for c in result['candidates']}
    control_producers = set().union(*(producers[c['candidate_id']] for c in result['candidates']
                                    if c['features']['control.is_proposal']))
    variant_sets = {}
    for cid, roles in role_candidates.items():
        name_rows = roles['commercial_name'] + roles['series']
        if len(producers[cid]) == 1 and producers[cid] == control_producers and name_rows:
            variant_sets[cid] = {t for row in name_rows for t in row['tokens']
                                if t.startswith('zz') and t.endswith('zz')}
    variant_frequency = Counter(t for tokens in variant_sets.values() for t in tokens)
    n = len(result['candidates'])
    for candidate in result['candidates']:
        features = candidate['features']
        roles = role_candidates[candidate['candidate_id']]
        for role, rows in roles.items():
            # Preserve literal evidence and add aligned aliases; never erase a
            # partial literal reading just because a longer alias was not read.
            for high, suffix in ((True, 'high'), (False, 'low')):
                matches = [m for row in rows for m in row['line_matches'] if m['high'] == high]
                features[f'text.{role}.{suffix}_full'] = max(features[f'text.{role}.{suffix}_full'], float(any(m['full_phrase'] for m in matches)))
                features[f'text.{role}.{suffix}_partial'] = max(features[f'text.{role}.{suffix}_partial'], float(any(not m['full_phrase'] for m in matches)))
            # Maximum over actual member claims; do not fabricate one exact
            # card by combining parts of several reference/year variants.
            features[f'text_detail.{role}.high_claim_coverage'] = max(
                (len(row['matched_high']) / len(set(row['tokens'])) for row in rows), default=0.)
            high_sets = [set(row['matched_high']) & set(row['expected_distinctive']) for row in rows]
            low_sets = [set(row['matched_low']) & set(row['expected_distinctive']) for row in rows]
            features[f'text_detail.{role}.high_distinctive_count'] = float(max(map(len, high_sets), default=0))
            features[f'text_detail.{role}.low_distinctive_count'] = float(max(map(len, low_sets), default=0))
            features[f'text_detail.{role}.high_distinctive_pool_mass'] = max((sum(
                math.log((n + 1) / (frequency[role][token] + 1)) / math.log(n + 1)
                for token in tokens) for tokens in high_sets), default=0.) if n else 0.
        name_rows = roles['commercial_name'] + roles['series']
        variant_atoms = variant_sets.get(candidate['candidate_id'], set())
        distinguishing_atoms = {t for t in variant_atoms if 0 < variant_frequency[t] < len(variant_sets)}
        for suffix in ('high', 'low'):
            features[f'text_detail.producer_variant.{suffix}_support_count'] = float(max(
                (len(set(row['matched_' + suffix]) & distinguishing_atoms) for row in name_rows), default=0))
        assert set(features) == set(FEATURE_NAMES)
        assert all(math.isfinite(float(v)) for v in features.values())
        candidate['features'] = {k: float(features[k]) for k in FEATURE_NAMES}
        candidate['discriminating_text_evidence'] = {'roles': roles,
            'qualifier_identity_alone': False, 'hard_veto': False,
            'same_control_producer': bool(candidate['candidate_id'] in variant_sets),
            'known_name_candidates_in_producer_comparison': len(variant_sets),
            'distinguishing_typed_atoms': sorted(distinguishing_atoms),
            'coverage_semantics': 'Soft union of words on this physical target; not a reconstructed OCR line or identity proof.'}
    result['query_evidence']['aligned_readings'] = lines
    result['discriminator_policy'] = {'version': SCHEMA_VERSION, 'year_in_product_ranking': False,
        'new_wine_specific_aliases': False, 'missing_word_veto': False,
        'generic_terms_are_identity': False, 'full_phrase_requires_one_line': True,
        'alias_sources': sorted({row[2] for row in FORMS}),
        'probability': None}
    return result
