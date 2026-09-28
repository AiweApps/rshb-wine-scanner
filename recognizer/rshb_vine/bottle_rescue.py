"""Bounded second detection pass only after the selected pipeline has no target."""
import torch
from rshb_vine.bottle_instances import iou
from rshb_vine.wine_object_profile import WineObjectProfile


class LowBottleDetector:
    def __init__(self, parent):
        self.parent = parent

    def detect(self, image):
        parent = self.parent
        inputs = parent.processor(images=image, return_tensors='pt').to(parent.device)
        with torch.inference_mode():
            output = parent.model(**inputs)
        found = parent.processor.post_process_object_detection(
            output, threshold=.2, target_sizes=[image.size[::-1]])[0]
        candidates = []
        for label, score, box in zip(found['labels'], found['scores'], found['boxes']):
            if parent.model.config.id2label[int(label)] != 'bottle':
                continue
            x1, y1, x2, y2 = box.cpu().tolist()
            bounds = [max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)]
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                candidates.append({'bbox': bounds, 'score': float(score)})
        kept = []
        for row in sorted(candidates, key=lambda r: -r['score']):
            if not any(iou(row['bbox'], other['bbox']) > .6 for other in kept):
                kept.append(row)
        self.proposal_count = len(kept)
        return kept[:16]


class BottleRescueProfile(WineObjectProfile):
    def recognize_image(self, image, roi=None):
        baseline = super().recognize_image(image, roi)
        baseline['bottle_rescue'] = {'attempted': False}
        if baseline['targets'] or roi is not None:
            return baseline
        localizer = self.detector.detector.parent
        original = localizer.bottles
        retry_detector = LowBottleDetector(original)
        try:
            localizer.bottles = retry_detector
            recovered = super().recognize_image(image, roi)
        finally:
            localizer.bottles = original
        trace = {'attempted': True, 'threshold': .2,
                 'raw_nms_proposals': retry_detector.proposal_count,
                 'proposal_limit': 16,
                 'baseline_instances': baseline['instances'],
                 'retry_instances': recovered['instances'],
                 'retry_targets': len(recovered['targets'])}
        # Return the new recognition only when it yielded actual SKU targets.
        result = recovered if recovered['targets'] else baseline
        result['bottle_rescue'] = trace
        if result is recovered:
            for key, value in baseline['timing_ms'].items():
                if isinstance(value, (int, float)):
                    result['timing_ms'][key] = result['timing_ms'].get(key, 0) + value
        else:
            for key, value in recovered['timing_ms'].items():
                if isinstance(value, (int, float)):
                    result['timing_ms'][key] = result['timing_ms'].get(key, 0) + value
        return result
