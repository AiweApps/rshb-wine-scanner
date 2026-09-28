"""Physical target contract: a pure function of parent responses (geometry + request ROI only).

No model call, no OCR, no card score, no GT. The parent response is never modified; the block lists every
physical bottle with its own unchanged answer, the ROI-addressed bottle, and a geometric primary suggestion
that is never published as a product.
"""
import math

from rshb_vine.bottle_instances import iou
from rshb_vine.io import digest

POLICY = 'target-contract-v2'
ELIGIBLE_AREA_RATIO = 0.5
TIE_MARGIN = 0.10
ADDRESS_IOU = 0.9
SPEC = {
    'geometry': 'target bottle_bbox, else context_bbox, else label bbox (EXIF-oriented original pixels)',
    'frame_size': "the single original_size shared by the targets' retrieval views",
    'eligible': f'area >= {ELIGIBLE_AREA_RATIO} * largest target area',
    'distance': 'sqrt(((cx - W/2) / W)^2 + ((cy - H/2) / H)^2) of the geometry centre',
    'suggestion': f'nearest eligible target only if the second-nearest eligible is >= {TIE_MARGIN} farther, '
                  'else ambiguous_tie; order ties by larger area, then instance_id',
    'roi': 'the parent ROI pass is authoritative; its single target is the addressed bottle, matched into the '
           f'no-ROI bottle list by geometry IoU >= {ADDRESS_IOU}',
    'publication': 'top-level fields are exactly the parent response for the same request; the suggestion is '
                   'never copied into best_candidate/slug and is never confirmed',
}


def geometry(target):
    for key in ('bottle_bbox', 'context_bbox', 'bbox'):
        if target.get(key):
            return list(target[key]), key
    return None, None


def frame_size(result):
    sizes = {tuple(v['original_size']) for t in result.get('targets') or []
             for v in (t.get('retrieval') or {}).get('views', []) if v.get('original_size')}
    return list(sizes.pop()) if len(sizes) == 1 else None


def detector_parents(result):
    trace = result.get('localization_trace')
    return None if trace is None else {str(p['parent_id']) for p in trace}


def selectability(target, parents):
    """Whether sending this box back as target_roi can address the target (ROI selects detector regions only)."""
    parent = str(target.get('parent_id'))
    if parents is not None and parent in parents:
        return 'detector_region'
    if parent.startswith('orphan-'):
        return 'not_addressable_orphan_label_without_detector_parent'
    if parent.startswith('closeup-'):
        return 'not_addressable_closeup_route_without_detector_region'
    return 'not_addressable_no_detector_region'


def bottle(target, parents=None):
    box, source = geometry(target)
    retrieval = target.get('retrieval') or {}
    selectable = selectability(target, parents)
    return {'instance_id': str(target['instance_id']), 'parent_id': target.get('parent_id'), 'kind': target.get('kind'),
            'bottle_bbox': target.get('bottle_bbox'), 'label_bbox': target.get('bbox'), 'geometry': box,
            'geometry_source': source, 'selectable': selectable,
            'select_roi': None if selectable.startswith('not_addressable') else box,
            'reasons': list(retrieval.get('reasons') or []), 'card': retrieval.get('best_candidate'),
            'product_id': (target.get('product_resolution') or {}).get('product_id'),
            'target_decision': retrieval.get('decision'), 'probability_correct': None, 'confirmed': False,
            'card_source': 'this_request', 'addressed': False, 'primary': False}


def central(entries, size):
    """Frozen geometric primary suggestion over the listed bottles."""
    if len(entries) < 2:
        return {'basis': 'not_multi_target', 'instance_id': None, 'ranking': []}
    if size is None:
        return {'basis': 'frame_size_unavailable', 'instance_id': None, 'ranking': []}
    width, height = size
    rows = []
    for e in entries:
        x1, y1, x2, y2 = e['geometry']
        rows.append({'instance_id': e['instance_id'], 'area_px': max(0, x2 - x1) * max(0, y2 - y1),
                     'distance': math.hypot(((x1 + x2) / 2 - width / 2) / width, ((y1 + y2) / 2 - height / 2) / height)})
    largest = max(r['area_px'] for r in rows)
    for r in rows:
        r['area_ratio'] = r['area_px'] / largest if largest else 0.0
        r['eligible'] = largest > 0 and r['area_ratio'] >= ELIGIBLE_AREA_RATIO
    rows.sort(key=lambda r: (not r['eligible'], r['distance'], -r['area_px'], r['instance_id']))
    eligible = [r for r in rows if r['eligible']]
    if not eligible:
        return {'basis': 'no_eligible_target', 'instance_id': None, 'ranking': rows}
    margin = eligible[1]['distance'] - eligible[0]['distance'] if len(eligible) > 1 else None
    basis = 'central_suggestion' if margin is None or margin >= TIE_MARGIN else 'ambiguous_tie'
    return {'basis': basis, 'instance_id': eligible[0]['instance_id'] if basis == 'central_suggestion' else None,
            'nearest_instance_id': eligible[0]['instance_id'], 'margin': margin, 'ranking': rows}


def _roi_reason(result):
    reason = (result.get('roi_selection') or {}).get('reason')
    return {'unique_maximum': 'roi_parent_without_target', 'no_overlap': 'roi_no_overlap',
            'tied_overlap': 'roi_tied_overlap'}.get(reason, 'roi_' + str(reason))


def build(result, roi=None, full=None):
    """result: parent response published for this request; full: parent no-ROI response for the same image."""
    listed = full if roi is not None and full is not None else result
    parents = detector_parents(listed)
    entries = [bottle(t, parents) for t in listed.get('targets') or []]
    size = frame_size(listed) or frame_size(result)
    suggestion = central(entries, size)
    block = {'policy': POLICY, 'spec': SPEC, 'mode': 'explicit_roi' if roi is not None else 'automatic',
             'roi': list(roi) if roi is not None else None,
             'bottles_scope': 'all' if roi is None or full is not None else 'addressed_only',
             'frame_size': size, 'central': suggestion, 'addressed_instance_id': None, 'roi_match': None,
             'roi_observation': None, 'addressed_physical_linkage': None, 'calibrated': False, 'probability': None,
             'published': {'card': result.get('best_candidate'), 'decision': result.get('decision'),
                           'slug': result.get('slug'), 'source': 'parent_response_unchanged'},
             'result_targets_digest': digest(result.get('targets')),
             'full_targets_digest': digest(full.get('targets')) if full is not None else None}
    if roi is None:
        if not entries:
            primary, basis = None, 'no_target'
        elif len(entries) == 1:
            primary, basis = entries[0]['instance_id'], 'single_target'
        else:
            primary, basis = suggestion['instance_id'], suggestion['basis']
    else:
        roi_targets = result.get('targets') or []
        primary, basis = None, _roi_reason(result) if not roi_targets else 'roi_multiple_targets'
        if len(roi_targets) == 1:
            addressed = bottle(roi_targets[0], detector_parents(result))
            addressed.update(card_source='roi_pass', roi_pass_instance_id=addressed['instance_id'])
            match = None
            if full is not None and addressed['geometry']:
                scored = sorted(((iou(e['geometry'], addressed['geometry']), e['instance_id']) for e in entries
                                 if e['geometry']), reverse=True)
                ambiguous = len(scored) > 1 and scored[1][0] >= ADDRESS_IOU
                if scored and scored[0][0] >= ADDRESS_IOU and not ambiguous:
                    match = scored[0]
                block['roi_match'] = {'iou': [{'instance_id': i, 'iou': s} for s, i in scored[:3]],
                                      'matched_instance_id': match[1] if match else None,
                                      'reason': 'matched' if match else 'ambiguous' if ambiguous else 'below_threshold'}
            elif full is None:
                block['roi_match'] = {'reason': 'addressed_only_no_full_pass'}
            linkage = 'roi_pass_only' if full is None else 'linked' if match else 'unresolved'
            if full is None:
                entry = entries[0]
                entry.update(card_source='roi_pass', roi_pass_instance_id=entry['instance_id'])
            elif match:
                entry = next(e for e in entries if e['instance_id'] == match[1])
                entry.update(full_pass_card=entry['card'], full_pass_product_id=entry['product_id'],
                             card=addressed['card'], product_id=addressed['product_id'],
                             target_decision=addressed['target_decision'], card_source='roi_pass',
                             roi_pass_instance_id=addressed['instance_id'])
            else:
                addressed['instance_id'] = 'roi-' + addressed['instance_id']
                addressed['physical_linkage'] = 'unresolved_not_counted_as_distinct_bottle'
                block['roi_observation'] = addressed
                entry = addressed
            primary, basis = entry['instance_id'], 'explicit_roi'
            block['addressed_instance_id'] = primary
            block['addressed_physical_linkage'] = linkage
    for e in entries:
        e['primary'] = e['instance_id'] == primary
        e['addressed'] = roi is not None and e['instance_id'] == primary
    block.update(bottles=entries, bottle_count=len(entries), primary_instance_id=primary, primary_basis=basis,
                 primary_confirmed=False,
                 selectable_bottles=sum(e['select_roi'] is not None for e in entries),
                 suggested_card=next((e['card'] for e in entries if e['instance_id'] == primary), None)
                 if basis == 'central_suggestion' else None,
                 requires_target_selection=len(entries) > 1 and basis != 'explicit_roi'
                 or (roi is not None and basis != 'explicit_roi' and bool(entries)))
    return block


def compact(result, block):
    """What /v2/targets returns: the contract plus the unchanged published fields."""
    return {'target_contract': block, 'decision': result.get('decision'), 'best_candidate': result.get('best_candidate'),
            'slug': result.get('slug'), 'probability_correct': result.get('probability_correct'),
            'requires_target_selection': result.get('requires_target_selection'),
            'reasons': list(result.get('reasons') or []), 'calibrated': False,
            'recognition_profile': result.get('recognition_profile'), 'parent_response_digest': digest(result)}
