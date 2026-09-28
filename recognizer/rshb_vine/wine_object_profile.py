"""Filter SKU targets before orientation and OCR, retaining raw object evidence."""
from rshb_vine.crop_stage_pipeline import CropStagePipeline
from rshb_vine.proposal_verifier import ProposalVerifierPipeline
from rshb_vine.recognition_profile import RecognitionProfile


class WineObjectStage(CropStagePipeline):
    def recognize_image(self, image, roi=None):
        result = ProposalVerifierPipeline.recognize_image(self, image, roi)
        structure = result.pop('structure_only')
        # A rejected target remains inspectable. Filtering is not evidence that
        # the physical object was never detected or that a catalogue SKU exists.
        result['instances'] = structure['instances']
        result['object_filter'] = {
            'policy': 'frozen-feature-wine-object-v1',
            'threshold': self.spec['threshold'],
            'targets_before': len(structure['targets']),
            'targets_after': len(result['targets']),
            'rejected_instance_ids': [i['bottle_id'] for i in result['rejected_instances']],
        }
        result['diagnostic_low_wine_score_instances'] = result['rejected_instances']
        result['verifier_policy'] = 'wine_object_target_gate'
        return result


class WineObjectProfile(RecognitionProfile, WineObjectStage):
    """ROI → geometry → object gate → orientation → text, using the same core."""
