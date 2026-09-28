"""Shared real-image OCR stage for the bounded positive-variant challenger."""
import time
from PIL import Image
from rshb_vine.oriented_geometry import OrientedGeometryPipeline
from rshb_vine.preprocessing import checked_box


class VariantTextPipeline(OrientedGeometryPipeline):
    def __init__(self, baseline, verifier_artifact, policy, reader):
        super().__init__(baseline, verifier_artifact)
        self.variant_policy = policy
        self.variant_reader = reader

    def recognize_image(self, image, roi=None):
        result = super().recognize_image(image, roi)
        trace = {'performed': False, 'reason': 'not_single_target'}
        result['variant_text'] = trace
        if len(result['targets']) != 1:
            return result
        target = result['targets'][0]
        box = checked_box(target.get('context_bbox', target['bbox']), image.size)
        angle = target.get('retrieval_pixel_rotation_ccw', 0)
        crop = image.crop(box)
        if angle in (90, 270):
            crop = crop.transpose(Image.Transpose.ROTATE_90 if angle == 90 else Image.Transpose.ROTATE_270)
        trace.update(reason='single_target', bbox_original=list(box), rotation_ccw=angle,
                     polygon_coordinate_space='rotated_bottle_crop_pixels')
        started = time.perf_counter()
        try:
            observations = self.variant_reader.read(crop)
            ranked, evidence = self.variant_policy.rerank(result['ranked_candidates'], observations)
        except Exception as exc:
            trace.update(reason='ocr_unavailable', error_type=type(exc).__name__)
        else:
            trace.update(performed=True, observations=observations, evidence=evidence)
            result['visual_ranked_candidates'] = result['ranked_candidates']
            result['ranked_candidates'] = ranked
            result['best_candidate'] = ranked[0]['slug'] if ranked else None
            target['retrieval']['visual_ranked_candidates'] = target['retrieval']['ranked_candidates']
            target['retrieval']['ranked_candidates'] = ranked
            target['retrieval']['best_candidate'] = result['best_candidate']
        result['timing_ms']['variant_text'] = (time.perf_counter() - started) * 1000
        # Preserve the existing uncalibrated decision and object-selection contract.
        result['slug'] = None
        result['probability_correct'] = None
        return result
