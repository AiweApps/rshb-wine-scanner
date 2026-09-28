"""Branch (a) of causal-crops-v1 over the loaded frozen CanvasCloseupRoute instance, without patching any global.

The frozen route computes the baseline and its own eligibility unchanged. Only when it refused with
``has_bottle_proposal`` and every bottle proposal is label-less with a finite wine-object score below the finite
object threshold of the same response, the adapter runs the one canvas retry through the frozen route's
request-local context and publishes it under the frozen acceptance (canvas fired, exactly one target, not
physically localized, IoU >= 0.9 with a canvas label). It acts only inside ``request_scope`` of a request with no
ROI (bottles=addressed and bottles=all are then the same single pass), at most once per request; a ROI request,
including its bottles=all no-ROI pass, stays frozen. The adapter object itself holds no per-request state.
"""
from contextlib import contextmanager
import contextvars
import copy
import math
import time

from rshb_vine.bottle_instances import iou
from rshb_vine.preprocessing import decode
from rshb_vine.target_misses_v1 import route as R

POLICY = 'zero-target-release-v1-a'
RULE = {'id': POLICY, 'parent_route': R.POLICY, 'applies': 'outer request without ROI (addressed or all), zero targets, '
        'frozen route reason has_bottle_proposal; ROI requests including their bottles=all pass unchanged', 'ignored': 'every retry proposal label-less with finite wine_score < finite '
        'object_filter.threshold of the same response; raw_nms_proposals == len(retry_instances) >= 1',
        'acceptance': 'frozen: canvas fired, one target, physical_bottle_localized false, IoU >= %s with a canvas label'
        % R.PROVENANCE_IOU, 'retries_per_request': 1, 'threshold_changed': False, 'branch_b': False}
_SCOPE = contextvars.ContextVar('zero_target_release_v1_scope', default=None)


@contextmanager
def request_scope(roi, bottles):
    token = _SCOPE.set({'enabled': roi is None, 'bottles': bottles, 'retries': 0, 'events': []})
    try:
        yield _SCOPE.get()
    finally:
        _SCOPE.reset(token)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ignored_proposals(result):
    """Proposals that (a) ignores, or None when the response is outside (a)."""
    trace = result.get('canvas_closeup_route') or {}
    if (trace.get('policy') != R.POLICY or trace.get('attempted') is not False or trace.get('reason') != 'has_bottle_proposal'
            or result.get('decision') == 'invalid_image' or result.get('targets')):
        return None
    if result.get('instances') or result.get('rejected_instances') or result.get('diagnostic_low_wine_score_instances'):
        return None
    threshold = (result.get('object_filter') or {}).get('threshold')
    rescue = result.get('bottle_rescue') or {}
    proposals = rescue.get('retry_instances') or []
    if not _finite(threshold) or not proposals or rescue.get('raw_nms_proposals') != len(proposals):
        return None
    if any(p.get('label_bbox') is not None or not _finite(p.get('wine_score')) or p['wine_score'] >= threshold
           for p in proposals):
        return None
    return [{'bottle_bbox': p.get('bottle_bbox'), 'wine_score': p['wine_score']} for p in proposals]


class ZeroTargetRoute:
    def __init__(self, frozen):
        self.frozen = frozen
        self.runtime = frozen.runtime
        self.manifest = frozen.manifest

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        result = self.frozen.recognize(data, roi)
        scope = _SCOPE.get()
        if scope is None or not scope['enabled'] or roi is not None or scope['retries']:
            return result
        ignored = ignored_proposals(result)
        if ignored is None:
            return result
        scope['retries'] += 1
        image, _ = decode(data)
        event = {'policy': POLICY, 'object_threshold': result['object_filter']['threshold'], 'ignored_proposals': ignored,
                 'baseline_ms': (time.perf_counter() - started) * 1000, 'recovered': False}
        retry_started = time.perf_counter()
        with self.frozen._active(image) as request:
            retry = self.frozen.runtime.recognize(data, None)
        event.update(canvas_fired=request['fired'], canvas_labels=request['labels'],
                     retry_ms=(time.perf_counter() - retry_started) * 1000)
        targets = retry.get('targets', [])
        if not request['fired']:
            event['reason'] = 'canvas_not_reached'
        elif len(targets) != 1:
            event['reason'] = 'no_canvas_label' if not request['labels'] else f'retry_targets_{len(targets)}'
        elif targets[0].get('physical_bottle_localized') or not any(
                iou(targets[0]['bbox'], label['bbox']) >= R.PROVENANCE_IOU for label in request['labels']):
            event['reason'] = 'target_not_from_canvas_label'
        else:
            event.update(reason='recovered', recovered=True, retry_decision=retry.get('decision'))
        if event['recovered']:
            chosen = copy.deepcopy(retry)
            target = chosen['targets'][0]
            target['geometry_source'] = R.POLICY
            target['physical_bottle_localized'] = False
            chosen['canvas_closeup_route'] = {
                'policy': R.POLICY, 'attempted': True, 'reason': 'recovered', 'recovered': True,
                'baseline_ms': event['baseline_ms'], 'eligibility_policy': POLICY, 'canvas_fired': True,
                'canvas_labels': request['labels'], 'retry_ms': event['retry_ms'], 'retry_decision': event['retry_decision']}
        else:
            chosen = result
        event['total_ms'] = (time.perf_counter() - started) * 1000
        scope['events'].append(event)
        chosen['zero_target_route'] = copy.deepcopy(event)
        timing = chosen.setdefault('timing_ms', {})
        timing['zero_target_route_total'] = event['total_ms']
        timing['total'] = event['total_ms']
        return chosen


def install(release):
    """Swap the one frozen route instance of a freshly loaded repair-v2 chain for the adapter; nothing global."""
    from rshb_vine.coherent_challenger_v1.active import CoherentRecognition
    from rshb_vine.geometry_loop import GeometryLocalizer
    from rshb_vine.label_rescue import LabelRescue
    from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
    from rshb_vine.systemic_ranking_v2.runtime import SystemicRecognition
    from rshb_vine.target_contract_v2.runtime import TargetContractRecognition
    geometry, target = release.base.geometry, release.base.target
    systemic = target.inner
    coherent = getattr(systemic, 'parent', None)
    frozen = getattr(coherent, 'route', None)
    if (type(geometry) is not GeometryReleaseRecognition or geometry.inner is not target
            or type(target) is not TargetContractRecognition or type(systemic) is not SystemicRecognition
            or type(coherent) is not CoherentRecognition or type(frozen) is not R.CanvasCloseupRoute
            or frozen.runtime is not coherent.runtime or release.base.runtime is not coherent.runtime):
        raise ValueError('Unexpected runtime graph for the zero-target route adapter')
    rescue = coherent.runtime.control.expanded.core.parent.parent.parent.base.detector
    if (not isinstance(rescue, LabelRescue) or type(rescue.parent) is not R.CanvasCloseupLocalizer
            or not isinstance(rescue.parent.parent, GeometryLocalizer)):
        raise ValueError('Frozen canvas close-up localizer delegate is not installed')
    coherent.route = ZeroTargetRoute(frozen)
    return {'policy': POLICY, 'route_path': 'base.target.inner.parent.route', 'frozen_route': frozen.manifest['checksum'],
            'frozen_route_class': R.CanvasCloseupRoute.__module__ + '.CanvasCloseupRoute'}
