"""Experimental explicit-ROI parent selection, independent of SKU predictions."""
from rshb_vine.bottle_instances import iou
from rshb_vine.oriented_geometry import OrientedGeometryPipeline


class SelectingDetector:
    def __init__(self, detector):
        self.detector = detector
        self.roi = None
        self.selection = None

    def __getattr__(self, key):
        return getattr(self.detector, key)

    def detect(self, image):
        regions = self.detector.detect(image)
        self.selection = None
        if self.roi is None:
            return regions
        parents = {}
        keys = []
        for i, region in enumerate(regions):
            key = region.get('parent_id', f'closeup-{i}')
            keys.append(key)
            box = region.get('bottle_bbox') or region.get('context_bbox') or [0, 0, *image.size]
            parents[key] = box
        scores = sorted([(iou(box, self.roi), str(key), key) for key, box in parents.items()], reverse=True)
        trace = {'roi': list(self.roi), 'policy': 'unique-positive-parent-iou-v1',
                 'scores': [{'parent_id': key, 'iou': score, 'bbox': parents[key]} for score, _, key in scores],
                 'selected_parent_id': None}
        self.selection = trace
        if not scores or scores[0][0] <= 0:
            trace['reason'] = 'no_overlap'
            return []
        if len(scores) > 1 and abs(scores[0][0] - scores[1][0]) <= 1e-12:
            trace['reason'] = 'tied_overlap'
            return []
        chosen = scores[0][2]
        trace.update(selected_parent_id=chosen, reason='unique_maximum')
        return [r for key, r in zip(keys, regions) if key == chosen]


class RoiSelectionPipeline(OrientedGeometryPipeline):
    def __init__(self, baseline, artifact):
        super().__init__(baseline, artifact)
        self.detector = SelectingDetector(self.detector)

    def recognize_image(self, image, roi=None):
        self.detector.roi = roi
        try:
            # Explicit ROI already selected a parent; do not apply the old center filter.
            result = super().recognize_image(image, None)
            result['roi_selection'] = self.detector.selection
            return result
        finally:
            self.detector.roi = None
