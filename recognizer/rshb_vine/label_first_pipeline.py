"""Shared automatic image path: decode once, localize, one batched visual encode."""
import math
import time
import hashlib
from dataclasses import asdict

import numpy as np

from rshb_vine.preprocessing import decode, checked_box, InvalidImage
from rshb_vine.visual_core import View, validate_vectors
from rshb_vine.io import seal, digest


REGION_POLICY = 'confident-enclosing-label-v1:ioa0.95:score0.9:area_ratio12'


def consolidate_regions(regions):
    """Suppress a lower-confidence fragment inside a confident enclosing label.

    Separate/partially overlapping targets remain separate. Keep suppressed
    evidence so this geometric heuristic is auditable; it asserts no identity.
    """
    def area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    kept, suppressed = [], []
    for region in sorted(regions, key=lambda r: -area(r['bbox'])):
        box = region['bbox']
        parent = None
        for candidate in kept:
            if (candidate.get('parent_id') != region.get('parent_id')
                    or candidate.get('kind') == 'bottle_context' or region.get('kind') == 'bottle_context'):
                continue
            outer = candidate['bbox']
            intersection = max(0, min(box[2], outer[2]) - max(box[0], outer[0])) * max(0, min(box[3], outer[3]) - max(box[1], outer[1]))
            if (intersection / area(box) >= .95 and area(outer) / area(box) <= 12
                    and candidate['detector_score'] >= max(.9, region['detector_score'])):
                parent = candidate
                break
        if parent is None:
            kept.append(region)
        else:
            suppressed.append({**region, 'reason': 'nested_label_fragment', 'enclosing_bbox': parent['bbox']})
    # Preserve detector order for targets that survived.
    retained = {id(r) for r in kept}
    return [r for r in regions if id(r) in retained], suppressed


class LabelFirstPipeline:
    def __init__(self, encoder, index, detector, encoder_id, max_targets=16, conditional_ocr=None):
        if encoder_id != index.encoder_id:
            raise ValueError('Query/gallery encoder mismatch')
        if max_targets < 1:
            raise ValueError('Positive target limit required')
        self.encoder, self.index, self.detector = encoder, index, detector
        self.encoder_id, self.max_targets = encoder_id, max_targets
        self.conditional_ocr = conditional_ocr
        self.manifest = seal({'kind': 'label_first_automatic_core_v1', 'encoder_id': encoder_id,
                              'detector_id': detector.model_id, 'max_targets': max_targets,
                              'region_policy': REGION_POLICY,
                              'conditional_ocr_id': conditional_ocr.policy_id if conditional_ocr else None,
                              'references_checksum': digest(index.references),
                              'vectors_sha256': hashlib.sha256(index.vectors.tobytes()).hexdigest()})

    def recognize(self, data, context_bbox=None):
        started = time.perf_counter()
        try:
            image, metadata = decode(data)
            decoded = time.perf_counter()
            result = self.recognize_image(image, context_bbox=context_bbox)
        except InvalidImage as exc:
            return {'decision': 'invalid_image', 'best_candidate': None, 'slug': None,
                    'probability_correct': None, 'reasons': [str(exc)],
                    'timing_ms': {'total': (time.perf_counter() - started) * 1000}}
        result['decode'] = metadata
        result['timing_ms']['decode'] = (decoded - started) * 1000
        result['timing_ms']['total'] = (time.perf_counter() - started) * 1000
        return result

    def recognize_image(self, image, *, context_bbox=None):
        """Use the already oriented original; regions never become new decoded inputs."""
        if context_bbox is not None and not isinstance(context_bbox, (list, tuple)):
            raise InvalidImage('ROI must be a coordinate array')
        context = checked_box(context_bbox if context_bbox is not None else [0, 0, *image.size], image.size)
        context_view = View('context', context, 'target_roi' if context_bbox is not None else 'full_frame',
                            True, list(image.size), cropper_version='automatic-main-label-v1')
        started = time.perf_counter()
        detections = self.detector.detect(image)
        detector_ms = (time.perf_counter() - started) * 1000
        accepted, rejected = [], []
        for i, detection in enumerate(detections):
            try:
                bbox = checked_box(detection['bbox'], image.size)
                score = float(detection['detector_score'])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError('Invalid detector score')
                region = {'bbox': bbox, 'detector_score': score}
                if 'context_bbox' in detection:
                    region.update(context_bbox=checked_box(detection['context_bbox'], image.size),
                                  parent_id=detection['parent_id'], kind=detection.get('kind', 'detected_label'),
                                  raw_label_regions=detection.get('raw_label_regions', []))
                    if region['kind'] not in ('detected_label', 'bottle_context'):
                        raise ValueError('Unsupported localized view')
            except (KeyError, TypeError, ValueError, InvalidImage):
                rejected.append({'index': i, 'reason': 'invalid_detector_region'})
                continue
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            if context_bbox is not None and not (context[0] <= cx < context[2] and context[1] <= cy < context[3]):
                rejected.append({'index': i, 'reason': 'outside_requested_roi'})
                continue
            accepted.append(region)
        accepted, suppressed = consolidate_regions(accepted)
        base = {'decision': 'ambiguous_target' if len(accepted) > 1 else 'uncertain', 'slug': None, 'probability_correct': None,
                'best_candidate': None, 'targets': [], 'context_fallback': None,
                'ranked_candidates': [],
                'context_view': asdict(context_view), 'detector_id': self.detector.model_id,
                'encoder_id': self.encoder_id, 'rejected_regions': rejected,
                'suppressed_regions': suppressed, 'region_policy': REGION_POLICY,
                'requires_target_selection': len(accepted) > 1,
                'reasons': ['confidence_not_calibrated'],
                'timing_ms': {'detector': detector_ms, 'encode': 0., 'retrieval': 0., 'ocr': 0.}}
        if len(accepted) > self.max_targets:
            # Return every localized region for selection; never encode a silently
            # truncated subset or pick the wine with the highest object score.
            base['unscored_regions'] = accepted
            base['reasons'].append('target_limit_requires_roi')
            return base
        views = [context_view]
        target_views = []
        for region in accepted:
            ids = [0]
            if 'context_bbox' in region:
                ids = [len(views)]
                views.append(View('context', region['context_bbox'], 'detected_bottle', True,
                                  list(image.size), cropper_version=self.detector.model_id))
            if region.get('kind') != 'bottle_context':
                ids.append(len(views))
                views.append(View('detected_label', region['bbox'], 'automatic_main_label', None,
                                  list(image.size), cropper_version=self.detector.model_id))
            target_views.append(ids)
        if accepted:
            used = sorted({i for ids in target_views for i in ids})
            positions = {old: new for new, old in enumerate(used)}
            views = [views[i] for i in used]
            target_views = [[positions[i] for i in ids] for ids in target_views]
        crops = [image.crop(view.bbox) for view in views]
        started = time.perf_counter()
        vectors = np.asarray(self.encoder.encode(crops), dtype='float32')
        validate_vectors(vectors, len(views))
        base['timing_ms']['encode'] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        if not accepted:
            fallback = self.index.search(vectors, views, self.encoder_id)
            base['context_fallback'] = fallback
            base['best_candidate'] = fallback['best_candidate']
            base['ranked_candidates'] = fallback['ranked_candidates']
            base['decision'] = fallback['decision']
            base['reasons'].append('no_valid_label_context_fallback')
        else:
            for region, ids in zip(accepted, target_views):
                result = self.index.search(vectors[ids], [views[i] for i in ids], self.encoder_id)
                base['targets'].append({**region, 'retrieval': result})
            if len(accepted) == 1:
                base['best_candidate'] = base['targets'][0]['retrieval']['best_candidate']
                base['ranked_candidates'] = base['targets'][0]['retrieval']['ranked_candidates']
                base['decision'] = base['targets'][0]['retrieval']['decision']
            else:
                base['reasons'].append('multiple_label_regions')
        base['timing_ms']['retrieval'] = (time.perf_counter() - started) * 1000
        if self.conditional_ocr is not None:
            if len(base['targets']) == 1 and base['targets'][0].get('kind') != 'bottle_context':
                started = time.perf_counter()
                target = base['targets'][0]
                target['retrieval'] = self.conditional_ocr.apply(image, target['bbox'], target['retrieval'])
                result = target['retrieval']
                base['best_candidate'] = result['best_candidate']
                base['ranked_candidates'] = result['ranked_candidates']
                base['reasons'] = result['reasons']
                base['ocr'] = result['ocr']
                base['timing_ms']['ocr'] = (time.perf_counter() - started) * 1000
            else:
                base['ocr'] = {'performed': False, 'reason': 'select_target_before_ocr' if len(base['targets']) > 1 else 'no_localized_label'}
        return base
