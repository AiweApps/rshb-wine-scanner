"""R: one bottle-context label view for exactly one surviving label-less wine bottle of a zero-target request.

Acts after every frozen rescue and after the zero-target adapter (a), only inside ``request_scope`` of an outer
request without ROI (addressed and all are then one pass), at most once per request. Eligible: no target, exactly
one instance in ``instances`` (the frozen wine-object survivors), with a physical RT-DETR ``bottle_bbox``, no
``label_bbox`` and a finite wine score >= the finite object threshold of the same response. Rejected instances
may exist; two or more survivors are excluded. Instance-free responses stay with the canvas route.

The retry reruns the unchanged runtime once while a request-local hook on the frozen LabelRescue turns only the
``bottle_context_no_label`` region whose context box equals that bottle box into a label view of the same parent
(bbox = bottle box, no raw label regions). Every model, gate and downstream stage is the frozen one, so B3, OCR
and selection read the same parent. Accepted only with exactly one retry target on that parent and bottle box and
one survivor; otherwise the baseline is returned. The published target and instance say that no label was
localized; the baseline instances are kept verbatim in the trace.
"""
from contextlib import contextmanager
import contextvars
import copy
import hashlib
import math
import time

from rshb_vine.io import sha256
from rshb_vine.preprocessing import checked_box, decode

POLICY = 'reading-recovery-bottle-context-v1'
SOURCE = 'bottle_context_label_view'
_SCOPE = contextvars.ContextVar('reading_recovery_v1_scope', default=None)
_REQUEST = contextvars.ContextVar('reading_recovery_v1_request', default=None)


@contextmanager
def request_scope(roi, bottles):
    token = _SCOPE.set({'enabled': roi is None, 'bottles': bottles, 'retries': 0, 'events': []})
    try:
        yield _SCOPE.get()
    finally:
        _SCOPE.reset(token)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value):
    return [float(v) for v in value]


def _fingerprint(image):
    return image.size, hashlib.sha256(image.tobytes()).hexdigest()


def plan(result):
    if result.get('decision') == 'invalid_image':
        return {'eligible': False, 'reason': 'invalid_image'}
    if result.get('targets'):
        return {'eligible': False, 'reason': 'has_targets'}
    threshold = (result.get('object_filter') or {}).get('threshold')
    if not _finite(threshold):
        return {'eligible': False, 'reason': 'no_object_threshold'}
    survivors = result.get('instances') or []
    counts = {'surviving_instances': len(survivors), 'rejected_instances': len(result.get('rejected_instances') or [])}
    if not survivors:
        return {**counts, 'eligible': False, 'reason': 'no_surviving_instance'}
    if len(survivors) > 1:
        return {**counts, 'eligible': False, 'reason': 'multiple_surviving_instances'}
    instance = survivors[0]
    if instance.get('label_bbox') is not None:
        return {**counts, 'eligible': False, 'reason': 'instance_has_label'}
    if instance.get('bottle_bbox') is None:
        return {**counts, 'eligible': False, 'reason': 'no_physical_bottle'}
    if not _finite(instance.get('wine_score')) or instance['wine_score'] < threshold:
        return {**counts, 'eligible': False, 'reason': 'below_object_gate'}
    return {**counts, 'eligible': True, 'reason': 'single_label_less_wine_bottle', 'bottle_id': str(instance['bottle_id']),
            'bottle_bbox': _box(instance['bottle_bbox']), 'wine_score': instance['wine_score'], 'object_threshold': threshold}


class BottleContextView:
    """Instance-level ``detect`` of the one frozen LabelRescue object (shared by every holder); inert unless an R
    request for this image is active."""

    def __init__(self, rescue):
        self.rescue = rescue
        self.frozen = type(rescue).detect

    def __call__(self, image):
        regions = self.frozen(self.rescue, image)
        request = _REQUEST.get()
        if request is None or _fingerprint(image) != request['fingerprint']:
            return regions
        request['calls'] += 1
        matches = [i for i, r in enumerate(regions) if r.get('source') == 'bottle_context_no_label'
                   and r.get('context_bbox') is not None
                   and _box(checked_box(r['context_bbox'], image.size)) == request['bottle_bbox']]
        if len(matches) != 1:
            request['unmatched_calls'] += 1
            return regions
        i = matches[0]
        region = regions[i]
        view = {**region, 'kind': 'detected_label', 'bbox': list(request['bottle_bbox']), 'raw_label_regions': [],
                'source': SOURCE, 'label_localized': False, 'reading_recovery': POLICY}
        request['converted'].append({'call': request['calls'], 'parent_id': region.get('parent_id'),
                                     'bbox': list(request['bottle_bbox'])})
        return regions[:i] + [view] + regions[i + 1:]


class BottleContextRoute:
    def __init__(self, inner):
        self.inner = inner
        self.runtime = inner.runtime
        self.manifest = {'policy': POLICY, 'inner': type(inner).__module__ + '.' + type(inner).__name__,
                         'retries_per_request': 1, 'threshold_changed': False, 'roi_requests': 'unchanged',
                         'source_sha256': sha256(__file__), 'calibrated': False}

    @contextmanager
    def _active(self, image, bottle_bbox):
        token = _REQUEST.set({'fingerprint': _fingerprint(image), 'bottle_bbox': bottle_bbox, 'calls': 0,
                              'unmatched_calls': 0, 'converted': []})
        try:
            yield _REQUEST.get()
        finally:
            _REQUEST.reset(token)

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        result = self.inner.recognize(data, roi)
        scope = _SCOPE.get()
        if scope is None or not scope['enabled'] or roi is not None or scope['retries']:
            return result
        decision = plan(result)
        if not decision['eligible']:
            if decision['reason'] != 'has_targets':
                result['bottle_context_route'] = {'policy': POLICY, 'attempted': False, **decision}
            return result
        scope['retries'] += 1
        image, _ = decode(data)
        event = {'policy': POLICY, 'attempted': True, **decision, 'eligibility': decision['reason'], 'recovered': False,
                 'baseline_ms': (time.perf_counter() - started) * 1000,
                 'baseline_instances': copy.deepcopy(result.get('instances') or []),
                 'baseline_rejected_instances': copy.deepcopy(result.get('rejected_instances') or [])}
        retry_started = time.perf_counter()
        with self._active(image, decision['bottle_bbox']) as request:
            retry = self.runtime.recognize(data, None)
        event.update(localizer_calls=request['calls'], converted=request['converted'],
                     unmatched_calls=request['unmatched_calls'], retry_ms=(time.perf_counter() - retry_started) * 1000)
        targets = retry.get('targets') or []
        survivors = retry.get('instances') or []
        bbox = decision['bottle_bbox']
        if not request['converted']:
            event['reason'] = 'bottle_context_not_reached'
        elif len(targets) != 1:
            event['reason'] = f'retry_targets_{len(targets)}'
        elif (targets[0].get('bottle_bbox') is None or _box(targets[0]['bottle_bbox']) != bbox
              or _box(targets[0].get('bbox') or []) != bbox or not targets[0].get('physical_bottle_localized')):
            event['reason'] = 'target_not_on_eligible_bottle'
        elif len(survivors) != 1 or survivors[0].get('bottle_bbox') is None or _box(survivors[0]['bottle_bbox']) != bbox:
            event['reason'] = 'retry_survivors_differ'
        else:
            event.update(reason='recovered', recovered=True, retry_decision=retry.get('decision'))
        if event['recovered']:
            chosen = copy.deepcopy(retry)
            target = chosen['targets'][0]
            target.update(geometry_source=POLICY, label_localized=False, label_view_source=SOURCE)
            survivor = chosen['instances'][0]
            survivor.update(label_bbox=None, label_view_bbox=list(bbox), label_view_source=SOURCE,
                            status=event['baseline_instances'][0].get('status'))
        else:
            chosen = result
        event['total_ms'] = (time.perf_counter() - started) * 1000
        scope['events'].append({k: event[k] for k in ('reason', 'recovered', 'total_ms')})
        chosen['bottle_context_route'] = event
        timing = chosen.setdefault('timing_ms', {})
        timing['bottle_context_route_total'] = event['total_ms']
        timing['total'] = event['total_ms']
        return chosen


def install(release):
    """Wrap the zero-target route of the loaded graph and the frozen LabelRescue localizer it retries through."""
    from rshb_vine.label_rescue import LabelRescue
    from rshb_vine.target_misses_v1.route import CanvasCloseupLocalizer
    from rshb_vine.zero_target_release_v1.route import ZeroTargetRoute
    coherent = release.base.target.inner.parent
    if type(coherent.route) is not ZeroTargetRoute or coherent.route.runtime is not coherent.runtime:
        raise ValueError('Zero-target route adapter is not the route of this graph')
    rescue = coherent.runtime.control.expanded.core.parent.parent.parent.base.detector
    if (not isinstance(rescue, LabelRescue) or type(rescue.parent) is not CanvasCloseupLocalizer
            or 'detect' in vars(rescue)):
        raise ValueError('Frozen label rescue / canvas localizer chain is not installed or already wrapped')
    rescue.detect = BottleContextView(rescue)
    route = BottleContextRoute(coherent.route)
    coherent.route = route
    return {'policy': POLICY, 'route_path': 'base.target.inner.parent.route',
            'localizer_path': 'base.target.inner.parent.runtime.control.expanded.core.parent.parent.parent.base.detector'
                              '.detect (instance attribute of the one LabelRescue object)',
            'manifest': route.manifest}
