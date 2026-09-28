"""Which frozen orphan-label trials G may hand to the downstream stages as an already-found region.

Pure functions over one recognition result (the control-level result after the frozen orphan stage, or a saved API
response for the census). Rows are only observed during a complete frozen pass; G acts in a second, armed pass only when
that whole frozen pipeline (including later fallbacks such as the canvas close-up route) returned no target, so a frozen
target is never pre-empted. Only a request that still has no target is eligible: beside an existing target the gate
context and the label detector are contaminated by that bottle (v1 diagnostic: a painting parent passed the gate, and
the detector mostly fires on faces, papers, food and pictures there). Two trial classes, both rejected by the frozen
stage:

  physical    reason not_one_valid_wine_target with exactly one containing parent: the frozen retry sent the label
              crop back through the bottle detector, which cannot find a bottle inside a label crop. The region keeps
              that detector parent (instances, bottle_rescue.retry_instances or the trial's own stripe proposals);
              physical_bottle_localized is claimed.
  label_only  reason no_unique_physical_parent, no containing parent, the frozen stripe search over the label's column
              returned no bottle proposal at all and the frozen beneath-association was not used. No bottle geometry
              is claimed. The object-gate context is the frame, as in the frozen close-up convention, only when
              the frame holds no other bottle geometry; otherwise it is the label region itself, so that another
              bottle cannot vouch for a paper or box the detector fired on.

Everything else (ambiguous parents, stripe proposals that do not contain the label, association retries on a
synthetic union parent, fragments of one bottle) stays frozen. Constants are the frozen association constants.
"""
from rshb_vine.bottle_instances import area, intersection, iou
from rshb_vine.roskachestvo_geometry_v2 import association as A

POLICY = 'atlas-geometry-trusted-region-v4'
PHYSICAL, LABEL_ONLY = 'physical', 'label_only'
SAME_PARENT_IOU = A.SAME_PARENT_IOU
PROVENANCE_IOU = A.PROVENANCE_IOU
CONTAINED = 0.8
ORPHAN_OVERLAP = 0.2
MAX_RETRIES_PER_REQUEST = 2
MAX_FULL_PASSES = 2
CONSTANTS = {'same_parent_iou': SAME_PARENT_IOU, 'retry_label_iou_with_region_min': PROVENANCE_IOU,
             'label_on_existing_bottle_containment': CONTAINED, 'label_overlap_with_existing_target_max': ORPHAN_OVERLAP,
             'retries_per_request_max': MAX_RETRIES_PER_REQUEST, 'retry_targets_required': 1,
             'requests': 'full-pipeline zero-target only: observe pass, at most one armed pass; stop after the first recovered target',
             'full_passes_max': MAX_FULL_PASSES,
             'armed_pass_requires': ['roi is None', 'observe pass valid image', 'observe pass has 0 targets',
                                     '>=1 eligible trial observed in the observe pass'],
             'armed_pass_accepted_only_if': 'targets non-empty and all added by armed G, else observe result',
             'object_gate_threshold': 'frozen (unchanged)', 'label_detector_threshold': 'frozen (unchanged)',
             'observe_expanded_memo': 'request-local: first observe control.expanded call (original bytes, no roi/G inject/'
                                      'canvas/orphan/split context, 0 targets) served once as a deep copy to the first armed '
                                      'call with the same key; front-label split traces replayed; no cross-request cache'}


def parent_pool(result):
    pool = []
    for source, rows in (('instances', result.get('instances') or []),
                         ('bottle_rescue.retry_instances', (result.get('bottle_rescue') or {}).get('retry_instances') or [])):
        for row in rows:
            if row.get('bottle_bbox'):
                pool.append({'source': source, 'bottle_id': row.get('bottle_id'), 'bottle_bbox': list(row['bottle_bbox']),
                             'label_bbox': row.get('label_bbox'), 'status': row.get('status'),
                             'wine_score': row.get('wine_score')})
    return pool


def _parent_record(pool, box, trial):
    pool = pool + [{'source': 'orphan_stripe_proposals', 'bottle_id': None, 'bottle_bbox': list(p['bbox']),
                    'label_bbox': None, 'status': None, 'wine_score': None, 'detector_score': p.get('score')}
                   for p in trial.get('proposals') or []]
    best = max(pool, key=lambda p: iou(p['bottle_bbox'], box), default=None)
    return dict(best) if best is not None and iou(best['bottle_bbox'], box) > SAME_PARENT_IOU else None


def _conflict(kind, label, parent, targets):
    for t in targets:
        own = t.get('bbox')
        body = t.get('bottle_bbox')
        if own and intersection(label, own) / max(area(label), 1) >= ORPHAN_OVERLAP:
            return 'overlaps_existing_target_label', t.get('instance_id')
        if body and intersection(label, body) / max(area(label), 1) >= CONTAINED:
            return 'label_on_existing_physical_target', t.get('instance_id')
        if kind == PHYSICAL and iou(parent, body or t.get('context_bbox') or own) > SAME_PARENT_IOU:
            return 'existing_physical_target', t.get('instance_id')
    return None, None


def label_only_context(result, label, size):
    if parent_pool(result) or result.get('targets'):
        return list(label), 'label_region_other_bottle_or_target_in_frame'
    return [0, 0, size[0], size[1]], 'frame_no_bottle_geometry'


def candidates(result):
    """Every frozen orphan trial with its G classification; ``eligible`` rows are the ones G may retry."""
    trace = result.get('orphan_label_recovery') or {}
    if not trace.get('attempted') or result.get('decision') == 'invalid_image':
        return []
    pool, targets, rows = parent_pool(result), result.get('targets') or [], []
    for k, trial in enumerate(trace.get('trials') or []):
        label = list(trial['label_bbox'])
        association = trial.get('geometry_association') or {}
        parents = trial.get('parents') or []
        row = {'trial': k, 'label_bbox': label, 'frozen_reason': trial.get('reason'), 'parents': [list(p) for p in parents],
               'stripe_attempted': bool(trial.get('stripe_attempted')), 'stripe_proposals': len(trial.get('proposals') or []),
               'association_reason': association.get('reason'), 'association_used': bool(association.get('used')),
               'label_score': association.get('orphan_label_detector_score'), 'eligible': False}
        rows.append(row)
        if trial.get('reason') == 'recovered' or association.get('used'):
            row['skip'] = 'frozen_recovered'
            continue
        if trial.get('reason') == 'not_one_valid_wine_target' and len(parents) == 1 and not association:
            parent = _parent_record(pool, parents[0], trial)
            if parent is None:
                row['skip'] = 'parent_not_in_roster'
                continue
            row.update(kind=PHYSICAL, parent=parent)
        elif (trial.get('reason') == 'no_unique_physical_parent' and not parents and trial.get('stripe_attempted')
              and not trial.get('proposals') and association.get('reason') == 'no_parent_beneath'):
            row.update(kind=LABEL_ONLY, parent=None)
        else:
            row['skip'] = 'not_trusted_geometry'
            continue
        if targets:
            row['skip'] = 'request_has_targets'
            continue
        conflict, other = _conflict(row['kind'], label, row['parent'] and row['parent']['bottle_bbox'], targets)
        if conflict:
            row.update(skip=conflict, conflicting_instance_id=other)
            continue
        row['eligible'] = True
    return rows
