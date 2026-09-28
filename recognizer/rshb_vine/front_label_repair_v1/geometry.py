"""Two-detector front-label boundary: RT-DETR served label united with an overlapping SSDlite main-label box.

The served RT-DETR label box can stop inside a large close-up label (brand/logo above it left out). The
independent SSDlite main-label detector, already run on every request by orphan recovery, often bounds the same
physical label differently. When both boxes are the same label (IoU), their union, clipped to the physical
parent (or the frame when no bottle is localized), replaces the served box as the label-channel view.
"""
from rshb_vine.bottle_instances import area, iou

POLICY = 'two-detector-label-boundary-v2'
SSD_SCORE = 0.3
# IoU, not overlap/min-area: an SSD box that merely contains the label (e.g. up to the shoulder) is not that label.
SAME_LABEL_IOU = 0.5
MIN_GROWTH = 1.1


def clip(box, bound):
    return [max(box[0], bound[0]), max(box[1], bound[1]), min(box[2], bound[2]), min(box[3], bound[3])]


def recovered_label(label, parent, ssd_boxes):
    """Return (box, trace); box is None when no SSDlite detection of the same label enlarges the served box."""
    trace = {'policy': POLICY, 'served_label': [int(round(v)) for v in label], 'parent': list(parent), 'matches': []}
    union = list(label)
    for s in ssd_boxes:
        if s['score'] < SSD_SCORE:
            continue
        overlap = iou(label, s['bbox'])
        if overlap >= SAME_LABEL_IOU:
            trace['matches'].append({'bbox': [int(round(v)) for v in s['bbox']], 'score': round(s['score'], 4),
                                     'iou': round(overlap, 4), 'source': s.get('source')})
            union = [min(union[0], s['bbox'][0]), min(union[1], s['bbox'][1]),
                     max(union[2], s['bbox'][2]), max(union[3], s['bbox'][3])]
    box = [int(round(v)) for v in clip(union, parent)]
    growth = area(box) / max(area(label), 1)
    trace.update(recovered=box, growth=round(growth, 4))
    if not trace['matches'] or growth < MIN_GROWTH:
        trace['added'] = False
        trace['reason'] = 'no_same_label_detection' if not trace['matches'] else 'no_material_growth'
        return None, trace
    trace['added'] = True
    return box, trace
