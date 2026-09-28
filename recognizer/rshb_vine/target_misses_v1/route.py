"""Opt-in canvas close-up label route for zero-target requests without any bottle geometry.

Cause (runs/target-misses-v1/diagnosis.json): when a label fills the whole frame, RT-DETR proposes
no bottle and the frozen full-frame label detector scores the label far below its 0.3 threshold;
the same detector finds it once the frame is centred on a larger neutral canvas. The route reruns
the unchanged runtime once and, only for the request's own pixels, lets the existing close-up
branch see the canvas detection. Weights, thresholds and all downstream gates stay frozen.
"""
from contextlib import contextmanager
import contextvars
import copy
import hashlib
import time

from rshb_vine.bottle_instances import area, iou
from rshb_vine.io import seal, sha256
from rshb_vine.preprocessing import InvalidImage, checked_box, decode
from rshb_vine.target_misses_v1.diagnose import CANVAS_SCALE, centered_canvas, stack

POLICY = 'canvas-closeup-label-route-v1'
FRAME_PROPOSAL = 0.9
PROVENANCE_IOU = 0.9
_REQUEST = contextvars.ContextVar('target_misses_v1_request', default=None)


def _fingerprint(image):
    return image.size, hashlib.sha256(image.tobytes()).hexdigest()


class CanvasCloseupLocalizer:
    """Delegate for the frozen GeometryLocalizer; inert unless a route request is active."""

    def __init__(self, parent):
        object.__setattr__(self, 'parent', parent)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def __setattr__(self, name, value):
        # BottleRescueProfile swaps `bottles` on the localizer it reaches through this delegate.
        setattr(self.parent, name, value)

    def detect(self, image):
        regions = self.parent.detect(image)
        request = _REQUEST.get()
        if request is None or regions or self.parent.last_trace:
            return regions
        if _fingerprint(image) != request['fingerprint']:
            return regions
        if request['fired']:
            return copy.deepcopy(request['regions'])
        request['fired'] = True
        canvas, dx, dy = centered_canvas(image)
        found = []
        for region in self.parent.labels.detect(canvas):
            x1, y1, x2, y2 = region['bbox']
            try:
                box = checked_box([max(0, x1 - dx), max(0, y1 - dy), min(image.width, x2 - dx),
                                   min(image.height, y2 - dy)], image.size)
            except InvalidImage:
                continue
            found.append({**region, 'bbox': [float(v) for v in box], 'canvas_closeup_route': POLICY})
        request['labels'] = [{'bbox': r['bbox'], 'detector_score': r['detector_score']} for r in found]
        request['regions'] = copy.deepcopy(found)
        return found


def zero_target(baseline, roi):
    return roi is None and baseline.get('decision') != 'invalid_image' and not baseline.get('targets')


def eligible(baseline, size):
    if baseline.get('instances') or baseline.get('rejected_instances') or baseline.get('diagnostic_low_wine_score_instances'):
        return False, 'has_instances'
    rescue = baseline.get('bottle_rescue') or {}
    proposals = rescue.get('retry_instances') or []
    if (rescue.get('raw_nms_proposals') or 0) != len(proposals) or len(proposals) > 1:
        return False, 'has_bottle_proposal'
    for p in proposals:
        # A label-less proposal spanning the frame is the frame itself, not separate bottle geometry.
        box = p.get('bottle_bbox')
        if p.get('label_bbox') is not None or box is None or area(box) < FRAME_PROPOSAL * size[0] * size[1]:
            return False, 'has_bottle_proposal'
    return True, 'eligible'


class CanvasCloseupRoute:
    def __init__(self, runtime):
        self.runtime = runtime
        rescue, localizer = stack(runtime)
        if not isinstance(localizer, CanvasCloseupLocalizer):
            rescue.parent = CanvasCloseupLocalizer(localizer)
        self.manifest = seal({'kind': POLICY, 'runtime': runtime.manifest['checksum'], 'canvas_scale': CANVAS_SCALE,
                              'canvas_fill': 'white', 'frame_sized_proposal_area': FRAME_PROPOSAL, 'label_threshold': 'frozen default 0.3', 'weights_changed': False,
                              'accept': 'exactly one target from the unchanged downstream stages, else baseline',
                              'source_sha256': sha256(__file__), 'calibrated': False})

    @contextmanager
    def _active(self, image):
        token = _REQUEST.set({'fingerprint': _fingerprint(image), 'fired': False, 'labels': []})
        try:
            yield _REQUEST.get()
        finally:
            _REQUEST.reset(token)

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        baseline = self.runtime.recognize(data, roi)
        baseline_ms = (time.perf_counter() - started) * 1000
        if not zero_target(baseline, roi):
            baseline['canvas_closeup_route'] = {'policy': POLICY, 'attempted': False, 'reason': 'not_zero_target_request',
                                                'recovered': False}
            return baseline
        image, _ = decode(data)
        ok, reason = eligible(baseline, image.size)
        trace = {'policy': POLICY, 'attempted': ok, 'reason': reason, 'recovered': False, 'baseline_ms': baseline_ms}
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
            target['geometry_source'] = POLICY
            target['physical_bottle_localized'] = False
            trace['retry_decision'] = retry.get('decision')
        chosen['canvas_closeup_route'] = trace
        timing = chosen.setdefault('timing_ms', {})
        timing['canvas_closeup_route_total'] = (time.perf_counter() - started) * 1000
        timing['total'] = timing['canvas_closeup_route_total']
        return chosen
