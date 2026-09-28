"""One extra label pass on physical bottles left without a label by the baseline."""
from rshb_vine.io import digest
from rshb_vine.preprocessing import checked_box


class LabelRescue:
    def __init__(self, parent, rescue):
        self.parent = parent
        self.rescue = rescue
        self.model_id = digest({'parent': parent.model_id, 'rescue': rescue.model_id,
                                'policy': 'empty-physical-parent-only-upright-v1'})
        self.last_trace = []

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def detect(self, image):
        original = self.parent.detect(image)
        self.last_trace = [dict(t) for t in self.parent.last_trace]
        if len(self.last_trace) > self.parent.max_bottles:
            return original
        traces = {str(t['parent_id']): t for t in self.last_trace}
        result = []
        for region in original:
            if region.get('source') != 'bottle_context_no_label':
                result.append(region)
                continue
            box = checked_box(region['context_bbox'], image.size)
            found = self.rescue.detect(image.crop(box))
            trace = traces[str(region['parent_id'])]
            trace['label_rescue_attempted'] = True
            trace['label_rescue_count'] = len(found)
            if not found:
                result.append(region)
                continue
            labels = []
            for label in found:
                x1, y1, x2, y2 = checked_box(label['bbox'], (box[2]-box[0], box[3]-box[1]))
                labels.append({'bbox': [x1+box[0], y1+box[1], x2+box[0], y2+box[1]],
                               'detector_score': label['detector_score']})
            groups = [labels] if self.parent.merge_parent_labels else self.parent.components(labels)
            for group in groups:
                bounds = [min(r['bbox'][0] for r in group), min(r['bbox'][1] for r in group),
                          max(r['bbox'][2] for r in group), max(r['bbox'][3] for r in group)]
                result.append({**region, 'bbox': bounds, 'kind': 'detected_label',
                               'detector_score': max(r['detector_score'] for r in group),
                               'raw_label_regions': group, 'source': 'rescue_label_inside_bottle'})
            # New label geometry came from the upright crop, not a failed rotated attempt.
            trace.update(label_count=len(labels), selected_rotation=0)
        return result
