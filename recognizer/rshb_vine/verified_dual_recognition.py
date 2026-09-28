"""Integrated candidate with evidence-preserving B0/B3 rank fusion."""
from copy import deepcopy
import time
from rshb_vine.admitted_evidence_recognition import AdmittedEvidenceRecognition
from rshb_vine.alternative_b3_consensus import AlternativeB3Consensus
from rshb_vine.rank_fused_b3_consensus import RankFusedB3Consensus
from rshb_vine.preprocessing import decode, checked_box
from rshb_vine.visual_core import View
from rshb_vine.io import seal


class VerifiedDualRecognition(AdmittedEvidenceRecognition):
    def __init__(self, root, protocol_path):
        super().__init__(root, protocol_path)
        core = self.expanded.core
        self.dual = RankFusedB3Consensus(
            AlternativeB3Consensus(core.single, core.multi, core.guard,
                                   self.isolation, self.expanded.aliases),
            self.sugar.ledger)
        self.manifest = seal({**{k: v for k, v in self.manifest.items() if k != 'checksum'},
                              'kind': 'verified-dual-recognition-v1',
                              'B3_visual_review': 'all_physical_targets',
                              'ranking': 'equal_reciprocal_top20_strict_improvement'})

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        baseline = super().recognize(data, roi)
        if not baseline.get('targets'):
            return baseline
        image, _ = decode(data)
        core = self.expanded.core
        raw = {'targets': []}
        for target in baseline['targets']:
            views = [View(**v) for v in target['retrieval']['views']]
            images = [image.crop(checked_box(v.bbox, image.size)) for v in views]
            angle = target.get('retrieval_pixel_rotation_ccw', 0)
            if angle not in (0, 90, 270):
                raise ValueError('Unsupported frozen pixel rotation')
            if angle:
                images = [im.rotate(angle, expand=True) for im in images]
            retrieval = core.index.search(core.encoder.encode(images), views, core.eid)
            retrieval['visual_ranked_candidates'] = deepcopy(retrieval['ranked_candidates'])
            raw['targets'].append({'instance_id': target['instance_id'], 'retrieval': retrieval})
        result = self.dual.apply(baseline, raw)
        self.refresh_aliases(result)
        result['verified_dual_recognition'] = {'manifest': self.manifest['checksum'],
                                              'B3_targets': len(raw['targets']),
                                              'weights_changed': False}
        result.setdefault('timing_ms', {})['total'] = 1000 * (time.perf_counter() - started)
        return result
