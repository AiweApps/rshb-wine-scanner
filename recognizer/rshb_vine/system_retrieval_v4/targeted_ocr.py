"""Bounded targeted label-crop OCR reread for targets whose selector OCR has no confident line."""
import re
import time

from PIL import Image

LABEL_KINDS = ('detected_label', 'front_label', 'partial_label')
HIGH = 0.85
KEEP = 0.5
MAX_TARGETS = 2
MARGIN = 0.08
TARGET_SHORT_SIDE = 320
MAX_SCALE = 3.0
MAX_LONG_SIDE = 2048
ROTATIONS = (0, 90, 270)


def _letters(text):
    return len(re.findall(r'[^\W\d_]', str(text)))


def has_confident_line(observations):
    return any(float(o.get('score') or 0) >= HIGH and _letters(o.get('raw_text', '')) >= 3 for o in observations or [])


def needs_reread(observations, label_bbox):
    return label_bbox is not None and not has_confident_line(observations)


def _crop(image, bbox, rotation_ccw):
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    box = [max(0, int(x1 - mx)), max(0, int(y1 - my)), min(image.width, int(x2 + mx + 0.999)),
           min(image.height, int(y2 + my + 0.999))]
    crop = image.crop(box)
    if rotation_ccw in (90, 180, 270):
        crop = crop.rotate(rotation_ccw, expand=True)
    scale = max(1.0, min(MAX_SCALE, TARGET_SHORT_SIDE / max(1, min(crop.size)), MAX_LONG_SIDE / max(crop.size)))
    if scale > 1.0:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    return crop, box, scale


def _strength(observations):
    return sum(_letters(o['raw_text']) for o in observations if float(o['score']) >= HIGH)


class TargetedLabelReread:
    def __init__(self, reader):
        self.reader = reader

    def reread(self, image, label_bbox, rotation_ccw=0, instance_id=None):
        crop, box, scale = _crop(image, label_bbox, rotation_ccw)
        reads = []
        for relative in ROTATIONS:
            view = crop if relative == 0 else crop.rotate(relative, expand=True)
            started = time.perf_counter()
            observations = self.reader.read(view)
            reads.append({'instance_id': instance_id, 'bbox': box, 'rotation_ccw': (rotation_ccw + relative) % 360,
                          'relative_rotation': relative, 'scale': round(scale, 4), 'crop_size': list(view.size),
                          'observations': observations, 'ms': round((time.perf_counter() - started) * 1000, 2)})
            high = [o for o in observations if float(o['score']) >= HIGH]
            if relative == 0 and len(high) >= 2 and _strength(observations) >= 8:
                break
        chosen = max(reads, key=lambda r: (_strength(r['observations']), -ROTATIONS.index(r['relative_rotation'])))
        supplemental = [{'raw_text': o['raw_text'], 'score': o['score'], 'polygon': []}
                        for o in chosen['observations'] if float(o['score']) >= KEEP]
        return {'reads': reads, 'chosen_relative_rotation': chosen['relative_rotation'],
                'supplemental_observations': supplemental, 'ms': round(sum(r['ms'] for r in reads), 2)}

    def apply(self, image, result):
        """Runtime adapter: add a separate targeted_label_ocr record to a recognition result without changing others."""
        record = {'performed': False, 'limit_targets': MAX_TARGETS, 'reads': [], 'targets': [], 'calibrated': False}
        evidence = (result.get('product_identity_evidence') or {}).get('targets') or []
        by_instance = {str(t.get('instance_id')): t for t in evidence}
        for target in result.get('targets') or []:
            if len(record['targets']) >= MAX_TARGETS:
                break
            instance = str(target.get('instance_id'))
            views = (target.get('retrieval') or {}).get('views') or []
            label = next((v for v in views if v.get('kind') in LABEL_KINDS), None)
            observations = (by_instance.get(instance) or {}).get('ocr_observations') or []
            if not needs_reread(observations, label and label.get('bbox')):
                continue
            outcome = self.reread(image, label['bbox'], target.get('retrieval_pixel_rotation_ccw', 0) or 0, instance)
            record['performed'] = True
            record['reads'].extend(outcome['reads'])
            record['targets'].append({'instance_id': instance, 'chosen_relative_rotation': outcome['chosen_relative_rotation'],
                                      'supplemental_observations': outcome['supplemental_observations'], 'ms': outcome['ms']})
        result['targeted_label_ocr'] = record
        return result
