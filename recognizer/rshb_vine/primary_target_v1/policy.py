"""Deterministic primary-target projection over an already published response.

Pure function of the response geometry and the request ROI: no model call, no card score, no GT.
Every target, its answer and its evidence stay byte-identical; for more than one target the public
card becomes the primary target's card as a suggested default, while decision/requires_target_selection
keep saying that the target is ambiguous and probability stays null.
"""
from copy import deepcopy
import math

from rshb_vine.io import digest

POLICY = 'primary-target-central-default-v1'
SPEC = {
    'geometry': "target['bbox'] in EXIF-oriented original pixels (the published target box)",
    'frame_size': "the single original_size shared by every target's retrieval views; otherwise no primary",
    'anchor_without_roi': 'frame centre (width/2, height/2)',
    'anchor_with_roi': 'ROI centre, applied only to the targets the runtime ROI parent selection returned',
    'order': 'Euclidean pixel distance of the bbox centre to the anchor ascending, then bbox area descending, '
             'then instance_id string ascending',
    'coefficients': 'none; no ambiguity margin, no card/selection score, no GT',
    'single_target': 'unchanged response; basis single_target or explicit_roi',
    'zero_targets': 'unchanged response; no primary, no full-frame fallback',
    'public_projection': 'best_candidate, ranked_candidates, selection_score and '
                         'catalog_additions.related_catalog_slugs copied from the primary target; decision '
                         'ambiguous_target and requires_target_selection=true kept; slug and probability null; '
                         'selected_product_output.public_representative_card follows the suggestion with '
                         'public_representative_basis=primary_target_suggested_default, accepted=false',
}
PROJECTED_FIELDS = ('best_candidate', 'ranked_candidates', 'selection_score', 'catalog_additions.related_catalog_slugs',
                    'selected_product_output.public_representative_card')


def frame_size(result):
    sizes = {tuple(v['original_size']) for t in result.get('targets', [])
             for v in t['retrieval'].get('views', []) if v.get('original_size')}
    return list(sizes.pop()) if len(sizes) == 1 else None


def ranking(targets, anchor):
    rows = []
    for t in targets:
        x1, y1, x2, y2 = t['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rows.append({'instance_id': str(t['instance_id']), 'centre': [cx, cy],
                     'distance_px': math.hypot(cx - anchor[0], cy - anchor[1]),
                     'area_px': max(0, x2 - x1) * max(0, y2 - y1)})
    rows.sort(key=lambda r: (r['distance_px'], -r['area_px'], r['instance_id']))
    return rows


def choose(result, roi=None):
    """(basis, primary instance_id or None, ranking, anchor) from response geometry only."""
    targets = result.get('targets') or []
    prefix = 'explicit_roi' if roi is not None else None
    if not targets:
        return (prefix + '_no_target' if prefix else 'no_target'), None, [], None
    if len(targets) == 1:
        return (prefix or 'single_target'), str(targets[0]['instance_id']), [], None
    if roi is not None:
        anchor = [(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2]
        basis = 'explicit_roi_centre_default'
    else:
        size = frame_size(result)
        if size is None:
            return 'central_default_unavailable_frame_size', None, [], None
        anchor = [size[0] / 2, size[1] / 2]
        basis = 'central_default'
    rows = ranking(targets, anchor)
    return basis, rows[0]['instance_id'], rows, anchor


def apply(result, roi=None, profile_checksum=None):
    """Annotate (and for >1 target project) the response in place; targets are never touched."""
    if result.get('decision') == 'invalid_image':
        return result
    targets = result.get('targets') or []
    before = digest(targets)
    basis, primary, rows, anchor = choose(result, roi)
    block = {'policy': POLICY, 'profile_checksum': profile_checksum, 'basis': basis, 'primary_instance_id': primary,
             'target_count': len(targets), 'roi': list(roi) if roi is not None else None, 'anchor': anchor,
             'frame_size': frame_size(result), 'ranking': rows, 'suggested_default': False, 'projected': False,
             'targets_digest': before, 'calibrated': False, 'probability': None, 'exact_slug': None}
    if len(targets) > 1 and primary is not None:
        contract = (result.get('best_candidate') is None and result.get('requires_target_selection') is True
                    and result.get('decision') == 'ambiguous_target')
        if not contract:
            block['not_projected_reason'] = 'source_multi_target_contract_unexpected'
        else:
            target = next(t for t in targets if str(t['instance_id']) == primary)
            retrieval = target['retrieval']
            additions = result.get('catalog_additions')
            block['source_public'] = {'best_candidate': None, 'selection_score': result.get('selection_score'),
                                      'ranked_candidates_digest': digest(result.get('ranked_candidates')),
                                      'related_catalog_slugs': (additions or {}).get('related_catalog_slugs')}
            result.update(best_candidate=retrieval.get('best_candidate'),
                          ranked_candidates=deepcopy(retrieval.get('ranked_candidates', [])),
                          selection_score=retrieval.get('selection_score'), slug=None, probability_correct=None)
            if additions is not None:
                additions['related_catalog_slugs'] = list(retrieval.get('related_catalog_slugs', []))
            trace = result.get('selected_product_output')
            if trace is not None:
                block['source_public']['public_representative_card'] = trace.get('public_representative_card')
                trace.update(public_representative_card=retrieval.get('best_candidate'),
                             public_representative_basis='primary_target_suggested_default',
                             public_representative_accepted=False)
            block.update(suggested_default=True, projected=True, primary_card=retrieval.get('best_candidate'),
                         public_decision='ambiguous_target', requires_target_selection=True)
    if digest(targets) != before:
        raise ValueError('Primary-target projection changed a target')
    result['primary_target'] = primary
    result['target_selection'] = block
    return result
