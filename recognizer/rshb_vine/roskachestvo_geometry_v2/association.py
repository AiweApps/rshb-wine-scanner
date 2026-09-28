"""V1 association with one change: the retry context is union(short_parent, orphan_label) padded 10% of union width
left/right, 2% of frame height above and 10% of frame height below, clamped to the frame.

One predeclared association policy on top of the frozen orphan-label recovery v2.

The frozen v2 pass runs first and is kept byte-for-byte. Only its trials that ended with zero containing parents
(``no_unique_physical_parent`` and ``parents == []``, stripe included) are revisited: the orphan label is joined to
the single label-less bottle proposal that ends at the label's top edge and overlaps it horizontally. The physical
parent becomes union(proposal, label); the unchanged recognize_without_orphans (detector, label detector, wine gate,
retrieval, OCR) runs once on a fixed full-body context crop around that union. The retry must return exactly
one target with visual retrieval whose label overlaps the orphan label; the public slug stays unconfirmed.
"""
from copy import deepcopy
import io
import math
import time

from rshb_vine.bottle_instances import iou
from rshb_vine.orphan_label_recovery_v2 import OrphanLabelRecovery, finalize_text_contract, publish_recovered_text
from rshb_vine.preprocessing import checked_box, decode

POLICY = 'orphan-label-beneath-short-parent-context-v2'
X_OVERLAP = 0.8
TOP_GAP_FRACTION_OF_HEIGHT = 0.05
CONTEXT_PAD_X_OF_UNION_WIDTH = 0.1
CONTEXT_TOP_OF_FRAME_HEIGHT = 0.02
CONTEXT_BOTTOM_OF_FRAME_HEIGHT = 0.1
PROVENANCE_IOU = 0.5
SAME_PARENT_IOU = 0.65
PAYLOAD_LIMIT = 20 * 1024 * 1024
CONSTANTS = {'x_overlap_min_of_label_and_parent_width': X_OVERLAP,
             'abs(label_top - parent_bottom) <= fraction_of_image_height': TOP_GAP_FRACTION_OF_HEIGHT,
             'label_bottom > parent_bottom': True, 'parent_label_bbox': None,
             'context_pad_x_fraction_of_union_width': CONTEXT_PAD_X_OF_UNION_WIDTH,
             'context_top_fraction_of_frame_height': CONTEXT_TOP_OF_FRAME_HEIGHT,
             'context_bottom_fraction_of_frame_height': CONTEXT_BOTTOM_OF_FRAME_HEIGHT,
             'retry_label_iou_with_orphan_min': PROVENANCE_IOU,
             'same_parent_iou': SAME_PARENT_IOU, 'retry_targets_required': 1, 'labels_per_request': 'frozen v2 (<=2)'}


class CapturingLabels:
    """Keeps the last full-frame label detections so the association trace can cite the detector score."""

    def __init__(self, detector):
        self.detector = detector
        self.last = []

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect(self, image, *args, **kwargs):
        self.last = self.detector.detect(image, *args, **kwargs)
        return self.last


def detection_score(detections, label):
    for d in detections:
        b = d['bbox']
        if [math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3])] == label:
            return d.get('detector_score')
    return None


def geometry(parent, label, height):
    overlap = max(0, min(parent[2], label[2]) - max(parent[0], label[0]))
    facts = {'x_overlap_of_label': overlap / max(label[2] - label[0], 1),
             'x_overlap_of_parent': overlap / max(parent[2] - parent[0], 1),
             'top_gap_fraction': (label[1] - parent[3]) / height,
             'label_extends_below': label[3] > parent[3]}
    facts['beneath'] = (facts['x_overlap_of_label'] >= X_OVERLAP and facts['x_overlap_of_parent'] >= X_OVERLAP
                        and abs(facts['top_gap_fraction']) <= TOP_GAP_FRACTION_OF_HEIGHT and facts['label_extends_below'])
    return facts


def associate(parents, label, height):
    matches, checked = [], []
    for p in parents:
        box = p.get('bottle_bbox')
        if not box or p.get('label_bbox') is not None:
            continue
        facts = geometry(box, label, height)
        checked.append({'bottle_id': p.get('bottle_id'), 'bottle_bbox': list(box), 'status': p.get('status'),
                        'wine_score': p.get('wine_score'), **facts})
        if facts['beneath'] and not any(iou(box, m['bottle_bbox']) > SAME_PARENT_IOU for m in matches):
            matches.append(p)
    return matches, checked


def padded(box, size):
    dx = CONTEXT_PAD_X_OF_UNION_WIDTH * (box[2] - box[0])
    top, bottom = CONTEXT_TOP_OF_FRAME_HEIGHT * size[1], CONTEXT_BOTTOM_OF_FRAME_HEIGHT * size[1]
    return checked_box([max(0, box[0] - dx), max(0, box[1] - top), min(size[0], box[2] + dx),
                        min(size[1], box[3] + bottom)], size)


class GeometryOrphanRecovery(OrphanLabelRecovery):
    def __init__(self, frozen):
        super().__init__(CapturingLabels(frozen.labels), frozen.bottles, frozen.retry)
        self.events = []

    def apply(self, data, baseline, roi=None):
        self.labels.last = []
        result = super().apply(data, baseline, roi)
        trace = result['orphan_label_recovery']
        pending = [(k, t) for k, t in enumerate(trace['trials'])
                   if t.get('reason') == 'no_unique_physical_parent' and not t.get('parents')]
        if not pending:
            return result
        started = time.perf_counter()
        image, _ = decode(data)
        parents = baseline.get('instances', []) + baseline.get('bottle_rescue', {}).get('retry_instances', [])
        added = []
        for k, trial in pending:
            label = trial['label_bbox']
            matches, checked = associate(parents, label, image.height)
            event = {'policy': POLICY, 'trial': k, 'orphan_label_bbox': list(label),
                     'orphan_label_detector_score': detection_score(self.labels.last, label),
                     'frozen_reason': trial['reason'], 'parents_checked': checked, 'used': False}
            trial['geometry_association'] = event
            self.events.append(event)
            if len(matches) != 1:
                event['reason'] = 'no_parent_beneath' if not matches else 'ambiguous_parent_beneath'
                continue
            source = matches[0]
            short = list(source['bottle_bbox'])
            union = checked_box([min(short[0], label[0]), min(short[1], label[1]),
                                 max(short[2], label[2]), max(short[3], label[3])], image.size)
            context = padded(union, image.size)
            event.update(original_parent={'bottle_id': source.get('bottle_id'), 'bottle_bbox': short,
                                          'status': source.get('status'), 'wine_score': source.get('wine_score')},
                         physical_bbox=union, context_bbox=context)
            if any(iou(union, t.get('bottle_bbox') or t.get('context_bbox') or t['bbox']) > SAME_PARENT_IOU
                   for t in result.get('targets', [])):
                event['reason'] = 'existing_physical_target'
                continue
            buffer = io.BytesIO()
            image.crop(context).save(buffer, format='PNG', compress_level=1)
            if len(buffer.getvalue()) > PAYLOAD_LIMIT:
                event['reason'] = 'payload_limit'
                continue
            retry = self.retry(buffer.getvalue())
            found = retry.get('targets', [])
            event['retry'] = {'decision': retry.get('decision'), 'targets': len(found),
                              'instances': [{k2: i.get(k2) for k2 in ('bottle_id', 'bottle_bbox', 'label_bbox', 'status',
                                                                        'wine_score')} for i in retry.get('instances', [])]}
            if len(found) != 1:
                event['reason'] = 'not_one_valid_wine_target'
                continue
            t = deepcopy(found[0])

            def restore(b):
                return [max(0, min(image.width, b[0] + context[0])), max(0, min(image.height, b[1] + context[1])),
                        max(0, min(image.width, b[2] + context[0])), max(0, min(image.height, b[3] + context[1]))]

            t['bbox'] = restore(t['bbox'])
            provenance = iou(t['bbox'], label)
            event['retry'].update(label_bbox=t['bbox'], label_iou_with_orphan=provenance,
                                  instance_id=found[0].get('instance_id'), wine_score=t.get('wine_score'),
                                  bottle_bbox=restore(t['bottle_bbox']) if t.get('bottle_bbox') else None,
                                  physical_bottle_localized=found[0].get('physical_bottle_localized'))
            if provenance < PROVENANCE_IOU:
                event['reason'] = 'retry_target_not_orphan_label'
                continue
            if not t.get('retrieval', {}).get('views'):
                event['reason'] = 'no_visual_retrieval'
                continue
            sid = f'orphan-geometry-{k}'
            t.update(instance_id=sid, parent_id=sid, physical_bottle_localized=True, bottle_bbox=union,
                     context_bbox=context, geometry_source=POLICY,
                     geometry_association={'original_parent': event['original_parent'], 'orphan_label_bbox': list(label),
                                           'orphan_label_detector_score': event['orphan_label_detector_score'],
                                           'retry_instance_id': found[0].get('instance_id'),
                                           'retry_label_iou_with_orphan': provenance})
            for view in t['retrieval'].get('views', []):
                if view.get('bbox'):
                    view['bbox'] = restore(view['bbox'])
                view['original_size'] = [image.width, image.height]
            result.setdefault('targets', []).append(t)
            instance = next((p for p in result.get('instances', []) if p.get('bottle_bbox') == short
                             and p.get('label_bbox') is None), None)
            if instance is None:
                instance = {'bottle_id': sid}
                result.setdefault('instances', []).append(instance)
            instance['geometry_association'] = {'policy': POLICY, 'original_bottle_bbox': short,
                                                'original_status': instance.get('status'),
                                                'original_wine_score': instance.get('wine_score'),
                                                'orphan_label_bbox': list(label)}
            instance.update(bottle_bbox=union, label_bbox=t['bbox'], label_id=sid + '-label', status='label_available',
                            wine_score=t.get('wine_score'))
            t['parent_id'] = instance['bottle_id']
            publish_recovered_text(result, baseline, retry, sid, context)
            event.update(reason='recovered', used=True, instance_id=sid, answer=t['retrieval']['best_candidate'])
            trace['added'].append(sid)
            added.append(sid)
        if added:
            finalize_text_contract(result)
            if len(result['targets']) == 1:
                t = result['targets'][0]
                result.update(best_candidate=t['retrieval']['best_candidate'],
                              ranked_candidates=t['retrieval']['ranked_candidates'], slug=None, decision='uncertain',
                              requires_target_selection=False)
            else:
                result.update(best_candidate=None, slug=None, ranked_candidates=[], decision='ambiguous_target',
                              requires_target_selection=True)
        trace['geometry_association_ms'] = 1000 * (time.perf_counter() - started)
        return result
