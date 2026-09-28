"""Coarse bottle localization followed by main-label localization on bottle crops."""
from pathlib import Path
from rshb_vine.io import digest, read_json, verify, sha256, local_path
from rshb_vine.preprocessing import checked_box


class BottleLabelLocalizer:
    def __init__(self, bottles, labels, bottle_model_id, max_bottles=16, merge_parent_labels=False):
        self.bottles, self.labels, self.max_bottles = bottles, labels, max_bottles
        self.merge_parent_labels = merge_parent_labels
        self.model_id = digest({'policy': 'bottle-label-parent-envelope-v1' if merge_parent_labels else 'bottle-label-overlap-components-v1',
                                'bottle_model': bottle_model_id, 'label_model': labels.model_id,
                                'min_intersection_over_smaller': .1, 'max_bottles': max_bottles})

    @classmethod
    def from_artifacts(cls, root, label_artifact, device='mps', merge_parent_labels=False):
        from rshb_vine.models import Detector
        from rshb_vine.label_detector import LabelDetector
        root = Path(root)
        manifest = verify(read_json(root / 'models/manifest.json'))
        files = {k: v for k, v in manifest['files'].items() if k.startswith('detector/')}
        if not files or any(sha256(local_path(root / 'models', k)) != v for k, v in files.items()):
            raise ValueError('Bottle detector files changed')
        return cls(Detector(root, device), LabelDetector(label_artifact, device),
                   digest({'files': files, 'adapter': sha256(root / 'rshb_vine/models.py')}),
                   merge_parent_labels=merge_parent_labels)

    @staticmethod
    def components(regions):
        def linked(a, b):
            a, b = a['bbox'], b['bbox']
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
            return intersection / area >= .1
        pending = list(regions)
        groups = []
        while pending:
            group = [pending.pop(0)]
            changed = True
            while changed:
                changed = False
                for candidate in list(pending):
                    if any(linked(candidate, member) for member in group):
                        group.append(candidate)
                        pending.remove(candidate)
                        changed = True
            groups.append(group)
        return groups

    def detect(self, image):
        bottles = self.bottles.detect(image)
        if not bottles:
            return self.labels.detect(image)
        result = []
        for i, bottle in enumerate(bottles):
            parent = checked_box(bottle['bbox'], image.size)
            common = {'parent_id': i, 'context_bbox': parent, 'bottle_score': float(bottle['score'])}
            labels = []
            if len(bottles) <= self.max_bottles:
                crop = image.crop(parent)
                for region in self.labels.detect(crop):
                    box = checked_box(region['bbox'], crop.size)
                    labels.append({'bbox': [box[0]+parent[0], box[1]+parent[1],
                                             box[2]+parent[0], box[3]+parent[1]],
                                   'detector_score': region['detector_score']})
            if not labels:
                result.append({**common, 'bbox': parent, 'kind': 'bottle_context',
                               'detector_score': float(bottle['score']), 'raw_label_regions': [],
                               'source': 'bottle_context_no_label'})
            groups = [labels] if self.merge_parent_labels and labels else self.components(labels)
            for group in groups:
                box = [min(r['bbox'][0] for r in group), min(r['bbox'][1] for r in group),
                       max(r['bbox'][2] for r in group), max(r['bbox'][3] for r in group)]
                result.append({**common, 'bbox': box, 'kind': 'detected_label',
                               'detector_score': max(r['detector_score'] for r in group),
                               'raw_label_regions': group, 'source': 'label_inside_detected_bottle'})
        return result
