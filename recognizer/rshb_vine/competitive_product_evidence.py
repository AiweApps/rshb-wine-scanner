"""One fixed, symmetric evidence challenger over the actual product pool.

Consumes the existing103 evidence contract; it extracts no new text, consults
no labels, fits no weights and changes no model scores. The caller owns the
binding of OCR and visual candidates to one physical target/receipt.
"""
from copy import deepcopy

from rshb_vine.candidate_discriminators import SCHEMA_VERSION
from rshb_vine.learned_selection_features import VERIFIED_STATUSES
from rshb_vine.product_first.partial_evidence import NAMES, contains_tokens


POLICY_VERSION = 'competitive-product-evidence-v1'
PRODUCER_ROLES = ('manufacturer', 'visible_brand')
IDENTITY_ROLES = ('commercial_name', 'series')
TYPED_ROLES = ('grape_blend', 'color', 'sugar', 'style')
COMPARABLE_STATUSES = frozenset(('catalog_asserted', *VERIFIED_STATUSES))
POLICY = {
    'version': POLICY_VERSION,
    'scope': 'all_actual_pool_products_on_one_physical_target',
    'required_anchor': 'high_full_manufacturer_or_visible_brand_on_same_card',
    'required_identity': 'high_full_commercial_name_or_distinctive_series_on_same_card',
    'ocr_threshold': .85,
    'threshold_origin': 'unchanged_existing103_high_reading_boundary',
    'winner': 'unique_candidate_with_card_witness_dominating_every_other_product',
    'retention': 'keep_prior_answer_if_no_unique_supported_winner',
    'unknown_is_contradiction': False,
    'missing_word_veto': False,
    'short_name_beats_unread_long_name': False,
    'qualifier_alone_is_identity': False,
    'year_in_product_comparison': False,
    'new_aliases_or_extraction': False,
    'model_scores_changed': False,
    'probability': None,
}


def _usable(claim):
    return bool(claim.get('comparison_allowed') and not claim.get('unknown')
                and claim.get('values') and claim.get('status') in COMPARABLE_STATUSES)


def _name_rows(candidate, slug, role, observations):
    claim = (candidate['provenance']['claims'].get(slug) or {}).get('roles', {}).get(role, {})
    if not _usable(claim):
        return []
    output = []
    for row in candidate['discriminating_text_evidence']['roles'].get(role, []):
        if row['slug'] != slug:
            continue
        high_full, low_full = set(), set()
        for match in row['line_matches']:
            if not match['full_phrase']:
                continue
            for index in match['line_ids']:
                if type(index) is not int or not 0 <= index < len(observations):
                    raise ValueError('Name witness is outside this target OCR packet')
                score = observations[index].get('score')
                # Repeated low readings do not inherit the confidence of a
                # different high reading in the same canonical reading group.
                high = isinstance(score, (int, float)) and not isinstance(score, bool) and score >= .85
                (high_full if high else low_full).add(index)
        output.append({
            'role': role, 'catalog_value': row['catalog_value'],
            'tokens': list(row['tokens']), 'catalog_status': claim['status'],
            'distinctive_tokens': list(row['expected_distinctive']),
            'high_distinctive_support': sorted(set(row['matched_high']) & set(row['expected_distinctive'])),
            'high_full_line_ids': sorted(high_full), 'low_full_line_ids': sorted(low_full),
            'catalog_sources': [deepcopy(c.get('source')) for c in row['source_claims']],
        })
    return output


def _card_evidence(candidate, slug, observations):
    profile = candidate['provenance']['claims'].get(slug) or {}
    roles = profile.get('roles', {})
    names = {role: _name_rows(candidate, slug, role, observations) for role in NAMES}
    producer = [row for role in PRODUCER_ROLES for row in names[role]
                if row['high_full_line_ids'] and row['distinctive_tokens']]
    producer_tokens = set(t for row in producer for t in row['tokens'])
    identity = [row for role in IDENTITY_ROLES for row in names[role]
                if row['high_full_line_ids'] and set(row['tokens']) - producer_tokens
                and (role == 'commercial_name' or row['distinctive_tokens'])]
    typed, conflicts = {}, []
    matches = candidate['text_evidence']['card_matches'].get(slug, {}).get('roles', {})
    for role in TYPED_ROLES:
        claim = roles.get(role, {})
        usable = _usable(claim)
        findings = matches.get(role, {}).get('findings', [])
        supported = [deepcopy(f) for f in findings if usable and f['threshold_admitted']]
        for finding in supported:
            index = finding['line_id']
            if type(index) is not int or not 0 <= index < len(observations):
                raise ValueError('Typed witness is outside this target OCR packet')
        typed[role] = {
            'catalog_values': list(claim.get('values', [])) if usable else [],
            'catalog_status': claim.get('status', 'unknown'), 'comparable': usable,
            'high_supported_values': sorted({f['catalog_value'] for f in supported}),
            'high_findings': supported,
            'catalog_sources': [deepcopy(c.get('source')) for c in claim.get('claims', [])],
        }
        # Only positive, unambiguous reading against an established field can
        # conflict. A partial grape list is never a claim that other grapes
        # are absent. A catalog-only disagreement remains a recorded risk.
        observed = candidate['text_evidence']['typed_disagreements'].get(role, {}).get('high_observed', [])
        expected = typed[role]['catalog_values']
        if role != 'grape_blend' and usable and len(observed) == len(expected) == 1 and observed != expected:
            conflicts.append({'role': role, 'observed': list(observed), 'catalog_values': expected,
                              'catalog_status': claim['status'],
                              'source_verified': claim['status'] in VERIFIED_STATUSES})
    reasons = []
    if not producer:
        reasons.append('no_high_full_distinctive_producer_or_brand')
    if not identity:
        reasons.append('no_high_full_identity_beyond_producer')
    if any(c['source_verified'] for c in conflicts):
        reasons.append('source_verified_typed_conflict')
    return {'slug': slug, 'names': names, 'typed': typed,
            'producer_anchor': producer, 'identity_witnesses': identity,
            'typed_conflicts': conflicts, 'eligible': not reasons,
            'ineligible_reasons': reasons}


def _rows(card, roles):
    return [row for role in roles for row in card['names'][role]]


def _preservation(left, right):
    """Can left retain every positive high witness of right? UNKNOWN is no veto.

    Unknown claims cannot assert that an observed right-hand witness has been
    retained. This blocks a promotion, but is never labelled a contradiction.
    """
    lost = []
    for group in (PRODUCER_ROLES, IDENTITY_ROLES):
        left_rows = _rows(left, group)
        available_tokens = {t for row in left_rows for t in row['tokens']}
        for row in _rows(right, group):
            if row['high_full_line_ids'] and not any(
                contains_tokens(other['tokens'], row['tokens']) for other in left_rows
            ):
                lost.append({'kind': 'full_phrase', 'role': row['role'],
                             'catalog_value': row['catalog_value'], 'tokens': row['tokens'],
                             'line_ids': row['high_full_line_ids'], 'missing_claim_is_unknown': not left_rows})
            missing = set(row['high_distinctive_support']) - available_tokens
            if missing:
                lost.append({'kind': 'partial_distinctive_tokens', 'role': row['role'],
                             'tokens': sorted(missing), 'missing_claim_is_unknown': not left_rows})
    for role in TYPED_ROLES:
        missing = set(right['typed'][role]['high_supported_values']) - set(left['typed'][role]['catalog_values'])
        if missing:
            lost.append({'kind': 'typed_positive_value', 'role': role, 'values': sorted(missing),
                         'missing_claim_is_unknown': not left['typed'][role]['comparable']})
    return lost


def _card_dominance(left, right):
    lost = _preservation(left, right)
    right_identity = _rows(right, IDENTITY_ROLES)
    # An observed short name is compatible with an unobserved longer sibling;
    # lack of that suffix must not become evidence against the sibling.
    distinguishing = [row for row in left['identity_witnesses'] if right_identity and not any(
        contains_tokens(other['tokens'], row['tokens']) for other in right_identity)]
    right_producer = _rows(right, PRODUCER_ROLES)
    producer_difference = [row for row in left['producer_anchor'] if right_producer and not any(
        contains_tokens(other['tokens'], row['tokens']) for other in right_producer)]
    typed_advantages = [{'role': role, 'values': sorted(set(left['typed'][role]['high_supported_values'])
                         - set(right['typed'][role]['catalog_values']))}
                        for role in TYPED_ROLES if right['typed'][role]['comparable'] and
                        set(left['typed'][role]['high_supported_values']) - set(right['typed'][role]['catalog_values'])]
    # Missing metadata does not disprove right. Positive producer+name evidence
    # can nevertheless rank above a candidate with no identity witnesses at
    # all. Keep that weaker basis explicit, separate from typed contradictions.
    unopposed_positive = bool(not right_identity and not right['identity_witnesses'])
    advantage = bool(distinguishing or producer_difference or typed_advantages or unopposed_positive)
    reasons = []
    if not left['eligible']:
        reasons.extend(left['ineligible_reasons'])
    if lost:
        reasons.append('would_lose_positive_evidence')
    if not advantage:
        reasons.append('compatible_identity_including_unread_longer_name')
    return {'left_slug': left['slug'], 'right_slug': right['slug'],
            'dominates': not reasons, 'reasons': reasons, 'lost_positive_evidence': lost,
            'different_full_identity': [{'role': row['role'], 'catalog_value': row['catalog_value'],
                                         'line_ids': row['high_full_line_ids']} for row in distinguishing],
            'different_full_producer': [{'role': row['role'], 'catalog_value': row['catalog_value'],
                                         'line_ids': row['high_full_line_ids']} for row in producer_difference],
            'positive_typed_advantages': typed_advantages,
            'unopposed_positive_bundle_unknown_competitor_identity': unopposed_positive,
            'missing_observation_is_contradiction': False}


class CompetitiveProductEvidence:
    def compare(self, features):
        """Compare actual evidence before consulting any current/model answer."""
        if features.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Competitive evidence requires the existing enriched103 contract')
        query = features['query_evidence']
        if query.get('instance_id') is None or query.get('ocr_threshold') != .85:
            raise ValueError('Missing physical target binding or changed OCR confidence boundary')
        candidates = features['candidates']
        ids = [c['candidate_id'] for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate pool candidate ID')
        for candidate in candidates:
            slugs = candidate['card_slugs']
            if not slugs or len(slugs) != len(set(slugs)):
                raise ValueError('Every pool product requires distinct actual card members')
            if (set(slugs) != set(candidate['provenance']['cards'])
                    or set(slugs) != set(candidate['provenance']['claims'])):
                raise ValueError('Pool product/card provenance differs')
        rows = {c['candidate_id']: [_card_evidence(c, slug, query['observations']) for slug in c['card_slugs']]
                for c in candidates}
        comparisons, winners, winning_cards = [], [], {}
        for cid in sorted(rows):
            viable_cards = {card['slug'] for card in rows[cid] if card['eligible']}
            if not viable_cards:
                continue
            for other in sorted(rows):
                if cid == other:
                    continue
                forward = [_card_dominance(a, b) for a in rows[cid] for b in rows[other]]
                reverse = [_card_dominance(b, a) for a in rows[cid] for b in rows[other]]
                # One real card hypothesis must establish the whole comparison.
                # Do not splice producer/name/typed fields from group members.
                card_winners = {card['slug'] for card in rows[cid] if card['eligible'] and all(
                    f['dominates'] for f in forward if f['left_slug'] == card['slug'])}
                dominates = bool(card_winners)
                comparisons.append({'left_candidate_id': cid, 'right_candidate_id': other,
                                    'left_dominates': dominates, 'left_to_right': forward,
                                    'right_to_left': reverse})
                viable_cards &= card_winners
            if viable_cards:
                winners.append(cid)
                winning_cards[cid] = sorted(viable_cards)
        return {'policy': deepcopy(POLICY), 'instance_id': query['instance_id'],
                 'same_instance_binding': 'supplied_existing103_target_packet; caller authenticates receipt lineage',
                 'producer_name_same_card_required': True,
                 'qualified_candidates': [cid for cid in sorted(rows) if any(c['eligible'] for c in rows[cid])],
                 'dominating_candidates': winners, 'winning_card_witnesses': winning_cards, 'candidates': rows,
                 'pairwise_comparisons': comparisons,
                 'model_score_is_probability': False, 'exact_vintage_claimed': False}

    def resolve(self, features, proposal):
        ranked = proposal['ranked_candidates']
        ids = [c['candidate_id'] for c in features['candidates']]
        if set(ids) != {r['candidate_id'] for r in ranked} or len(ids) != len(ranked):
            raise ValueError('Proposal and actual evidence pool must preserve the same candidates')
        trace = self.compare(features)
        winners = trace['dominating_candidates']
        selected = winners[0] if len(winners) == 1 else proposal.get('candidate_id')
        changed = selected != proposal.get('candidate_id')
        output = deepcopy(proposal)
        trace.update(prior_candidate_id=proposal.get('candidate_id'), selected_candidate_id=selected,
                     changed=changed, reason=(
                         'unique_supported_competitor' if changed else 'prior_answer_has_unique_support'
                         if len(winners) == 1 else 'insufficient_or_ambiguous_competing_evidence'))
        output['competitive_product_evidence'] = trace
        if not changed:
            return output
        winner = next(row for row in ranked if row['candidate_id'] == selected)
        output.update(candidate_id=winner['candidate_id'], product_id=winner['product_id'],
                      representative_slug=winner['representative_slug'], best_candidate=winner['representative_slug'],
                      representative_source=winner['representative_source'], score=winner['score'],
                      score_margin=None, probability=None, exact_slug=None,
                      vintage={'value': None, 'status': 'unknown'},
                      reason='competitive_product_evidence', release_admitted=False)
        # An earlier describe_year result belongs to its earlier product. The
        # shared runtime must describe the selected product afresh afterwards.
        output.pop('product_resolution', None)
        for row in output['ranked_candidates']:
            row['pre_competition_rank'] = row['rank']
        output['ranked_candidates'].sort(key=lambda row: row['candidate_id'] != selected)
        for rank, row in enumerate(output['ranked_candidates'], 1):
            row['rank'] = rank
        return output
