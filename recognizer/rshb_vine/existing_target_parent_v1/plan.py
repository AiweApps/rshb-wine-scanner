"""Which frozen orphan-label trials beside an already published target may receive a detector parent (N1).

Pure functions over one control-level result after the frozen orphan stage (or a saved API response for a census).
The frozen orphan stage itself adds physically parented labels beside existing targets; it fails for a truncated
close-up bottle because its stripe search runs on the raw column and finds no proposal. N1 revisits only the trials
that G (atlas-geometry-trusted-region-v4) classifies label_only apart from its zero-target restriction and asks the
frozen stripe detector once more on the frozen close-up canvas. A parent is claimed only for a unique mapped proposal
that contains the label (N1). Only when no proposal contains the label at all, the label itself becomes an unlocalized
target whose object gate context is the label region, as G does for label_only beside another bottle (N1b); ambiguous
parents, conflicts and a physical retry rejected downstream never fall back. Every constant is an existing frozen one.
"""
from rshb_vine.atlas_geometry_repair_v1 import plan as G
from rshb_vine.bottle_instances import area, intersection
from rshb_vine.target_misses_v1.diagnose import CANVAS_SCALE

POLICY = 'existing-target-physical-parent-v1'
PHYSICAL, LABEL_ONLY = G.PHYSICAL, G.LABEL_ONLY
STRIPE_PAD = 0.1
CONTAINED = G.CONTAINED
SAME_PARENT_IOU = G.SAME_PARENT_IOU
PROVENANCE_IOU = G.PROVENANCE_IOU
DETECTOR_THRESHOLD, DETECTOR_NMS, DETECTOR_LIMIT = 0.2, 0.6, 16
TRIALS_MAX = 2
CONSTANTS = {'policy': POLICY, 'stripe': 'frozen orphan stripe: label x -+ %.1f of label width, full frame height' % STRIPE_PAD,
             'canvas': 'frozen target_misses_v1 centred white canvas, scale %d' % CANVAS_SCALE,
             'detector': 'frozen orphan-stage LowBottleDetector (threshold %.1f, NMS %.1f, max %d), fresh wrapper per call'
                         % (DETECTOR_THRESHOLD, DETECTOR_NMS, DETECTOR_LIMIT),
             'parent_containment_min': CONTAINED, 'same_parent_iou': SAME_PARENT_IOU,
             'retry_label_iou_with_region_min': PROVENANCE_IOU, 'trials_per_request_max': TRIALS_MAX,
             'retries_per_request_max': TRIALS_MAX, 'retry_targets_required': 1,
             'acts_only_if': ['roi is None', 'top-level request pixels', 'no G inject/canvas/orphan/split context',
                              'the pass already has >=1 target'],
             'existing_targets': 'kept; every appended target is a new physical bottle',
             'text_identity_used': False,
             'object_gate': 'frozen B0 gate inside the unchanged retry: context = parent (N1) or label region (N1b)',
             'label_only_fallback': 'only when E1 has no containing proposal; physical_bottle_localized false, no bottle box',
             'no_fallback_if': ['ambiguous containing proposals', 'conflict with an existing target',
                                'physical retry rejected downstream']}


def stripe(box, size):
    pad = STRIPE_PAD * (box[2] - box[0])
    return [max(0, int(box[0] - pad)), 0, min(size[0], int(box[2] + pad)), size[1]]


def contained(label, box):
    return intersection(label, box) / max(area(label), 1)


def map_proposal(bbox, column, offset, height):
    x1, y1, x2, y2 = bbox
    dx, dy = offset
    return [max(0., x1 - dx) + column[0], max(0., y1 - dy), min(column[2] - column[0], x2 - dx) + column[0],
            min(float(height), y2 - dy)]


def trials(result):
    """G-classified label_only trials of a pass that already has targets; ``eligible`` rows get one E1 call."""
    trace = result.get('orphan_label_recovery') or {}
    targets = result.get('targets') or []
    if not trace.get('attempted') or result.get('decision') == 'invalid_image' or not targets:
        return []
    rows = []
    for row in G.candidates(result):
        if row.get('kind') != LABEL_ONLY or row.get('skip') != 'request_has_targets':
            continue
        row = {k: row[k] for k in ('trial', 'label_bbox', 'frozen_reason', 'stripe_attempted', 'stripe_proposals',
                                   'association_reason', 'label_score')}
        conflict, other = G._conflict(LABEL_ONLY, row['label_bbox'], None, targets)
        row.update(eligible=conflict is None, skip=conflict, conflicting_instance_id=other)
        rows.append(row)
    budget = 0
    for row in rows:
        if row['eligible']:
            budget += 1
            if budget > TRIALS_MAX:
                row.update(eligible=False, skip='trials_per_request_budget')
    return rows


def parent(label, proposals, targets):
    """(parent record or None, reason) from the mapped E1 proposals of one label."""
    holding = [p for p in proposals if p['inside_column'] and contained(label, p['bbox']) >= CONTAINED]
    if len(holding) != 1:
        return None, 'no_containing_proposal' if not holding else 'ambiguous_containing_proposals'
    best = holding[0]
    conflict, _ = G._conflict(PHYSICAL, label, best['bbox'], targets)
    if conflict:
        return None, conflict
    return {'source': 'existing_target_parent_v1_canvas_stripe', 'bottle_id': None, 'bottle_bbox': list(best['bbox']),
            'label_bbox': None, 'status': None, 'wine_score': None, 'detector_score': best['score'],
            'canvas_bbox': list(best['canvas_bbox']), 'label_containment': contained(label, best['bbox'])}, 'unique_parent'
