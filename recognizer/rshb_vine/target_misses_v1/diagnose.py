"""Per-stage causal trace for zero-target requests: bottle boxes, raw label boxes, receipt stages."""
from pathlib import Path
import time

from PIL import Image

from rshb_vine.io import read_json, seal, sha256, write_json
from rshb_vine.preprocessing import decode
from rshb_vine.target_misses_v1.scope import OUT, load

CANVAS_SCALE = 2


def stack(runtime):
    """Locate the frozen localizer objects used by the current runtime; fail if the graph changed."""
    from rshb_vine.geometry_loop import GeometryLocalizer
    from rshb_vine.label_rescue import LabelRescue
    rescue = runtime.control.expanded.core.parent.parent.parent.base.detector
    if not isinstance(rescue, LabelRescue) or not isinstance(rescue.parent, GeometryLocalizer):
        raise ValueError('Unexpected localizer graph')
    return rescue, rescue.parent


def centered_canvas(image, scale=CANVAS_SCALE):
    w, h = image.size
    canvas = Image.new('RGB', (w * scale, h * scale), 'white')
    dx, dy = (canvas.width - w) // 2, (canvas.height - h) // 2
    canvas.paste(image, (dx, dy))
    return canvas, dx, dy


def _labels(detector, image, frame):
    w, h = frame
    rows = detector.detect(image, 0.0)
    rows = sorted(rows, key=lambda r: -r['detector_score'])[:3]
    return [{'bbox': [round(v, 1) for v in r['bbox']], 'score': round(r['detector_score'], 4),
             'area_fraction': round((r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1]) / (w * h), 3)}
            for r in rows]


def _receipt_stages(result):
    br = result.get('bottle_rescue') or {}
    orphan = result.get('orphan_label_recovery') or {}
    return {'decision': result.get('decision'), 'targets': len(result.get('targets', [])),
            'instances': len(result.get('instances', [])), 'rejected_by_object_gate': len(result.get('rejected_instances', [])),
            'low_wine_score_instances': len(result.get('diagnostic_low_wine_score_instances', [])),
            'bottle_rescue_raw_nms_proposals': br.get('raw_nms_proposals'),
            'bottle_rescue_retry_instances': len(br.get('retry_instances', [])),
            'orphan_trials': [t.get('reason') for t in orphan.get('trials', [])],
            'square_rescue': (result.get('square_rescue') or {}).get('reason'),
            'oversized_label_rescue_attempted': (result.get('oversized_label_rescue') or {}).get('attempted')}


def run(root, runtime):
    root = Path(root).resolve()
    scope = load(root)
    rescue, localizer = stack(runtime)
    rows = []
    for case in scope['misses']:
        image, _ = decode((root / case['image_path']).read_bytes())
        started = time.perf_counter()
        bottles = localizer.bottles.detect(image)
        low = runtime.control.orphans.bottles.detect(image)
        frame = _labels(localizer.labels, image, image.size)
        spare = _labels(rescue.rescue, image, image.size)
        canvas, dx, dy = centered_canvas(image)
        on_canvas = _labels(localizer.labels, canvas, image.size)
        for r in on_canvas:
            r['bbox_original'] = [round(max(0, min(image.width, r['bbox'][0] - dx)), 1),
                                  round(max(0, min(image.height, r['bbox'][1] - dy)), 1),
                                  round(max(0, min(image.width, r['bbox'][2] - dx)), 1),
                                  round(max(0, min(image.height, r['bbox'][3] - dy)), 1)]
        receipt = read_json(root / case['intake_receipt'])
        rows.append({'item_id': case['item_id'], 'image_sha256': case['image_sha256'], 'size': list(image.size),
                     'bottles_main_threshold': len(bottles), 'bottles_rescue_threshold': len(low),
                     'frame_label_top3_primary': frame, 'frame_label_top3_rejected_rescue_weights': spare,
                     'canvas_label_top3_primary': on_canvas, 'receipt_stages': _receipt_stages(receipt['result']),
                     'ms': (time.perf_counter() - started) * 1000})
    doc = seal({'kind': 'target-misses-v1-diagnosis', 'scope_checksum': scope['checksum'],
                'runtime_manifest': runtime.manifest['checksum'], 'canvas_scale': CANVAS_SCALE,
                'primary_label_threshold': 0.3, 'bottle_threshold_main': 0.4, 'bottle_threshold_rescue': 0.2,
                'note': 'scores below the frozen thresholds are diagnostic only; nothing is accepted here',
                'source_sha256': sha256(__file__), 'rows': rows})
    write_json(root / OUT / 'diagnosis.json', doc)
    return doc
