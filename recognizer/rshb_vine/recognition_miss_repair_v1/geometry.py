"""CanvasCloseupRoute successor: a lone label-less bottle proposal rejected by the frozen wine gate is not geometry.

The frozen route (target-misses-v1) refuses a retry whenever bottle rescue leaves any label-less
proposal smaller than the frame. When that proposal's own wine score is below the frozen object
threshold of the same request, it is not a wine bottle and cannot own a label, so the frame is
treated like a proposal-free close-up. The retry, canvas label detector and every downstream gate
are the unchanged route code.
"""
import copy
import time

from rshb_vine.bottle_instances import iou
from rshb_vine.preprocessing import decode
from rshb_vine.target_misses_v1.route import POLICY as PARENT_POLICY
from rshb_vine.target_misses_v1.route import PROVENANCE_IOU, CanvasCloseupRoute, eligible, zero_target

POLICY = 'recognition-miss-repair-v1-nonwine-proposal-canvas'


def eligible_v2(baseline, size):
    ok, reason = eligible(baseline, size)
    if ok or reason != 'has_bottle_proposal':
        return ok, reason
    rescue = baseline.get('bottle_rescue') or {}
    proposals = rescue.get('retry_instances') or []
    threshold = (baseline.get('object_filter') or {}).get('threshold')
    if threshold is None or len(proposals) != 1 or (rescue.get('raw_nms_proposals') or 0) != 1:
        return ok, reason
    p = proposals[0]
    if p.get('label_bbox') is None and p.get('wine_score') is not None and p['wine_score'] < threshold:
        return True, 'eligible_nonwine_label_less_proposal'
    return ok, reason


class NonWineProposalCanvasRoute(CanvasCloseupRoute):
    def recognize(self, data, roi=None):
        # Same body as the frozen CanvasCloseupRoute.recognize; only the eligibility function differs.
        started = time.perf_counter()
        baseline = self.runtime.recognize(data, roi)
        baseline_ms = (time.perf_counter() - started) * 1000
        if not zero_target(baseline, roi):
            baseline['canvas_closeup_route'] = {'policy': PARENT_POLICY, 'attempted': False,
                                                'reason': 'not_zero_target_request', 'recovered': False}
            return baseline
        image, _ = decode(data)
        ok, reason = eligible_v2(baseline, image.size)
        trace = {'policy': PARENT_POLICY, 'attempted': ok, 'reason': reason, 'recovered': False,
                 'baseline_ms': baseline_ms, 'eligibility_policy': POLICY}
        if not ok:
            baseline['canvas_closeup_route'] = trace
            return baseline
        retry_started = time.perf_counter()
        with self._active(image) as request:
            retry = self.runtime.recognize(data, None)
        trace.update(canvas_fired=request['fired'], canvas_labels=request['labels'],
                     retry_ms=(time.perf_counter() - retry_started) * 1000)
        targets = retry.get('targets', [])
        if not request['fired']:
            trace['reason'] = 'canvas_not_reached'
        elif len(targets) != 1:
            trace['reason'] = 'no_canvas_label' if not request['labels'] else f'retry_targets_{len(targets)}'
        elif targets[0].get('physical_bottle_localized') or not any(
                iou(targets[0]['bbox'], label['bbox']) >= PROVENANCE_IOU for label in request['labels']):
            trace['reason'] = 'target_not_from_canvas_label'
        else:
            trace.update(reason='recovered', recovered=True)
        chosen = copy.deepcopy(retry) if trace['recovered'] else baseline
        if trace['recovered']:
            target = chosen['targets'][0]
            target['geometry_source'] = PARENT_POLICY
            target['physical_bottle_localized'] = False
            trace['retry_decision'] = retry.get('decision')
        chosen['canvas_closeup_route'] = trace
        timing = chosen.setdefault('timing_ms', {})
        timing['canvas_closeup_route_total'] = (time.perf_counter() - started) * 1000
        timing['total'] = timing['canvas_closeup_route_total']
        return chosen
