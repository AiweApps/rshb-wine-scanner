"""Crop-trained label localization with verifier scores retained as diagnostics."""
from rshb_vine.proposal_verifier import ProposalVerifierPipeline
from rshb_vine.io import seal

class CropStagePipeline(ProposalVerifierPipeline):
    def __init__(self,baseline,artifact):
        super().__init__(baseline,artifact)
        self.manifest=seal({**{k:v for k,v in self.manifest.items() if k!='checksum'},'object_gate':'label_required; wine_score_diagnostic_only','actual_localizer_id':self.detector.model_id})

    def recognize_image(self,image,roi=None):
        diagnostic=super().recognize_image(image,roi)
        result=diagnostic['structure_only']
        result['timing_ms']=diagnostic['timing_ms']
        result['verifier_policy']='diagnostic_only'
        result['diagnostic_low_wine_score_instances']=diagnostic['rejected_instances']
        return result
