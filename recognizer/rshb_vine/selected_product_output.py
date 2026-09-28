"""Publish already chosen product proposals into the legacy card response.

No model calls, new candidates, identity inference or geometry changes. Input
selections are real ProductSelection outputs, keyed by the physical instance.
After this projection the caller refreshes product/year descriptions and calls
the existing control.refresh_aliases(result). Exact/calibrated slug stays None.
"""
from copy import deepcopy
import math

from rshb_vine.io import digest


POLICY = 'selected-product-output-v1'
CHANNEL_ORDER = ('B0_label', 'B0_context', 'B3_label', 'B3_context')


def _finite_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Selection score must be a finite uncalibrated number')
    return float(value)


def _source_row(slug, candidate, old_rows):
    provenance = candidate['provenance']
    controls = [row for row in provenance['control_candidates'] if row['slug'] == slug]
    visual = [row for row in provenance['visual'] if row['raw']['slug'] == slug]
    for row in visual:
        if row['channel'] not in CHANNEL_ORDER or type(row['rank']) is not int or row['rank'] < 1:
            raise ValueError('Invalid actual visual candidate provenance')
    if slug in old_rows:
        if not any(row == old_rows[slug] for row in controls):
            raise ValueError('Legacy card differs from the actual selector input')
        result = deepcopy(old_rows[slug])
        source = 'actual8175_returned_card'
    elif controls:
        result = deepcopy(controls[0])
        source = 'actual8175_supplied_card'
    elif visual:
        raw = min(visual, key=lambda row: (row['rank'], CHANNEL_ORDER.index(row['channel'])))
        result = deepcopy(raw['raw'])
        source = 'actual_raw_visual_card:' + raw['channel']
    else:
        raise ValueError('Product member has no actual card evidence: ' + slug)
    if result.get('slug') != slug:
        raise ValueError('Source row changed card identity')
    # These are small raw rank/reference rows, not full feature or claim copies.
    # Original card score/channels remain unchanged; selection_score is separate.
    result['selection_card_source'] = source
    result['selection_visual_evidence'] = deepcopy(visual)
    return result


def _project_target(target, selection):
    sid = str(target['instance_id'])
    features, proposal = selection['features'], selection['proposal']
    if str(features['query_evidence']['instance_id']) != sid:
        raise ValueError('Selection belongs to another physical target')
    candidates = features['candidates']
    by_id = {c['candidate_id']: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError('Duplicate feature candidate IDs')
    ranked = proposal['ranked_candidates']
    if len(ranked) != len(by_id) or {r['candidate_id'] for r in ranked} != set(by_id):
        raise ValueError('Selection changed the actual product pool')
    retrieval = target['retrieval']
    original_rows = retrieval.get('ranked_candidates', [])
    old_rows = {r['slug']: r for r in original_rows}
    if len(old_rows) != len(original_rows):
        raise ValueError('Duplicate legacy card rows')
    control = retrieval.get('best_candidate')
    supplied_control = [c['provenance']['control_proposal_slug'] for c in candidates
                        if c['provenance'].get('control_proposal_slug') is not None]
    if supplied_control != ([control] if control is not None else []):
        raise ValueError('Selection was built from another control answer')
    all_cards = set()
    for candidate in candidates:
        members = candidate['card_slugs']
        if not members or len(members) != len(set(members)) or all_cards.intersection(members):
            raise ValueError('Actual card must belong to exactly one nonempty pool product')
        all_cards.update(members)
        cards = candidate['provenance']['cards']
        if set(cards) != set(members) or any(
            cards[s]['slug'] != s or cards[s]['product_id'] != candidate['product_id'] for s in members
        ):
            raise ValueError('Product/card registry provenance changed')
    if not set(old_rows) <= all_cards:
        raise ValueError('Selector pool lost a previous control card')
    selected_id = proposal.get('candidate_id')
    selected_slug = proposal.get('representative_slug')
    if ranked:
        if selected_id != ranked[0]['candidate_id'] or selected_slug not in by_id[selected_id]['card_slugs']:
            raise ValueError('Selected product/card disagrees with the final candidate order')
        if proposal.get('product_id') != by_id[selected_id]['product_id'] or proposal.get('best_candidate') != selected_slug:
            raise ValueError('Selected product and representative card disagree')
    elif any(proposal.get(key) is not None for key in ('candidate_id', 'product_id', 'representative_slug', 'best_candidate')):
        raise ValueError('Empty pool cannot publish a selected card')
    projected = []
    for rank, item in enumerate(ranked, 1):
        candidate = by_id[item['candidate_id']]
        members = candidate['card_slugs']
        if (item['rank'] != rank or item['product_id'] != candidate['product_id']
                or set(item['card_slugs']) != set(members)):
            raise ValueError('Final product rank or card membership changed')
        representative = selected_slug if item['candidate_id'] == selected_id else item['representative_slug']
        if representative not in members:
            raise ValueError('Product representative is not an actual member')
        order = [representative]
        order.extend(row['slug'] for row in original_rows if row['slug'] in members and row['slug'] not in order)
        order.extend(slug for slug in members if slug not in order)
        score = _finite_score(item['score'])
        for slug in order:
            row = _source_row(slug, candidate, old_rows)
            row.update(selection_score=score, selection_score_kind='uncalibrated_linear_selector_score',
                       selection_product_rank=rank, selection_candidate_id=item['candidate_id'],
                       product_id=item['product_id'])
            projected.append(row)
    if {row['slug'] for row in projected} != all_cards or len(projected) != len(all_cards):
        raise ValueError('Public card projection lost or duplicated actual evidence')
    if (projected[0]['slug'] if projected else None) != selected_slug:
        raise ValueError('Public first card differs from selected representative')
    trace = {'instance_id': sid, 'before': control, 'after': selected_slug,
             'changed': control != selected_slug, 'candidate_id': selected_id,
             'product_id': proposal.get('product_id'),
             'selection_score': _finite_score(proposal['score']) if ranked else None,
             'score_kind': 'uncalibrated_linear_selector_score',
             'model_checksum': proposal.get('model_checksum'),
             'identity_registry_checksum': proposal.get('identity_registry_checksum'),
             'feature_digest': digest(features), 'proposal_digest': digest(proposal),
             'selection_reason': proposal.get('reason'),
             'previous_candidate_union': deepcopy(retrieval.get('candidate_union', [])),
             'previous_related_catalog_slugs': deepcopy(retrieval.get('related_catalog_slugs', [])),
             'product_candidates': len(ranked), 'card_candidates': len(projected),
             'card_score_fields_preserved': True, 'calibrated': False, 'exact_slug': None}
    retrieval.update(best_candidate=selected_slug, ranked_candidates=projected,
                     candidate_union=sorted(all_cards), slug=None, probability_correct=None,
                     decision='uncertain' if selected_slug is not None else 'unknown',
                     selection_score=trace['selection_score'], related_catalog_slugs=[],
                     selected_product_output=trace)
    # All raw visual evidence, including visual_ranked_candidates and
    # channel_top20, remains in its original fields and its original order.
    target.pop('product_resolution', None)
    target.pop('product_candidates', None)
    return trace


def publish_selected_products(baseline, selections_by_instance):
    """Return a copy with card ordering projected from each selected product.

    ``selections_by_instance`` contains one real ``{features, proposal}`` value
    for every actual target. A supplied ROI does not license choosing one of
    several remaining targets. Call ``control.refresh_aliases(output)`` and
    regenerate product/year descriptions after this function, before serving.
    """
    targets = baseline.get('targets', [])
    target_ids = [str(t['instance_id']) for t in targets]
    if len(target_ids) != len(set(target_ids)) or set(target_ids) != set(selections_by_instance):
        raise ValueError('Public output must preserve the exact physical target roster')
    if baseline.get('selected_product_output') is not None:
        raise ValueError('Cannot publish over an already projected selection')
    if not targets and baseline.get('best_candidate') is not None:
        raise ValueError('A global card without a physical target cannot be projected')
    result = deepcopy(baseline)
    traces = [_project_target(target, selections_by_instance[str(target['instance_id'])])
              for target in result.get('targets', [])]
    if len(targets) == 1:
        retrieval = result['targets'][0]['retrieval']
        result.update(best_candidate=retrieval['best_candidate'], ranked_candidates=retrieval['ranked_candidates'],
                      decision='uncertain' if retrieval['best_candidate'] is not None else 'insufficient_evidence',
                      requires_target_selection=False, selection_score=retrieval['selection_score'])
    else:
        result.update(best_candidate=None, ranked_candidates=[], selection_score=None,
                      requires_target_selection=len(targets) > 1)
        if len(targets) > 1:
            result['decision'] = 'ambiguous_target'
    result.update(slug=None, probability_correct=None)
    result.pop('product_resolution', None)
    if 'catalog_additions' in result:
        result['catalog_additions']['related_catalog_slugs'] = []
    result['selected_product_output'] = {
        'policy': POLICY, 'targets': traces, 'geometry_changed': False, 'roster_changed': False,
        'source_best_candidate': baseline.get('best_candidate'),
        'source_result_digest': digest(baseline), 'public_representative_card': result.get('best_candidate'),
        'exact_slug': None, 'probability': None, 'calibrated': False,
        'requires_alias_refresh': True, 'requires_product_year_refresh': True}
    return result
