"""Isolated B3 label-specialist ablation; preserves8151."""
from pathlib import Path
import sys
import time
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, read_jsonl, verify, sha256, seal
from rshb_vine.label_specialist_consensus import VariantGuard, combine
from rshb_vine.within_producer_title import WithinProducerTitle
from rshb_vine.catalog_constrained_title import CatalogConstrainedTitle
from rshb_vine.product_name_selector import ProductNameSelector
from rshb_vine.domain_retrieval_replay import replay
from rshb_vine.preprocessing import decode
from rshb_vine.encoder_artifact import load_encoder
from rshb_vine.visual_core import LabelFirstIndex


class GuardedConsensus:
    def __init__(self, parent):
        self.parent = parent
        run = ROOT / 'runs/label-specialist-consensus-v1'
        validation = verify(read_json(run / 'validation/report.json'))
        protocol = verify(read_json(run / 'validation/protocol.json'))
        if not validation['passed'] or protocol['code_sha'] != sha256(ROOT / 'rshb_vine/label_specialist_consensus.py'):
            raise ValueError('Validation or code freeze failed')
        groups = verify(read_json(run / 'source-groups.json'))['records']
        catalog = list(read_jsonl(ROOT / 'data/normalized/catalog.jsonl'))
        sig = verify(read_json(ROOT / 'runs/gallery-variant-source-review-v1/gallery/signatures.json'))['signatures']
        self.single = CatalogConstrainedTitle(catalog, sig)
        self.multi = ProductNameSelector(catalog, sig)
        self.title = WithinProducerTitle(self.single)
        self.guard = VariantGuard(catalog, sig, groups)
        self.eid, self.encoder = load_encoder(ROOT, ROOT / 'runs/domain-training-v1/B3-round-evaluation/B3-provisional/encoder', 'mps')
        gallery = verify(read_json(ROOT / 'runs/B3-current-pipeline-v1/gallery.json'))
        vp = ROOT / 'runs/B3-current-pipeline-v1/vectors.npy'
        if self.eid != gallery['encoder_id'] or sha256(vp) != gallery['vectors_sha']:
            raise ValueError('B3 index mismatch')
        self.index = LabelFirstIndex(np.load(vp), gallery['references'], self.eid)
        self.manifest = seal({'kind': 'label-specialist-consensus-v1', 'parent': parent.manifest,
                              'validation': validation['checksum'], 'runner_sha': sha256(__file__),
                              'calibrated': False, 'conditional_B3': True})

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        baseline = self.parent.recognize(data, roi)
        if not baseline.get('targets'):
            return baseline
        needed = False
        for target in baseline['targets']:
            ret = target['retrieval']
            ch = ret.get('channel_top20', {})
            label, context = ch.get('front_label', []), ch.get('context', [])
            if label and context and label[0]['slug'] != context[0]['slug'] and ret['best_candidate'] == label[0]['slug']:
                needed = True
        competitor = baseline
        if needed:
            image, _ = decode(data)
            competitor = replay(image, baseline, self.encoder, self.index, self.eid, self.single)
            if len(competitor['targets']) > 1:
                obs = {str(t['instance_id']): t['observations'] for t in baseline.get('instance_text', {}).get('targets', []) if t.get('performed')}
                for target in competitor['targets']:
                    if str(target['instance_id']) in obs:
                        ret = target['retrieval']
                        rows, _ = self.multi.rerank(ret['ranked_candidates'], obs[str(target['instance_id'])])
                        ret.update(ranked_candidates=rows, best_candidate=rows[0]['slug'] if rows else None)
        result = combine(baseline, competitor, self.guard, self.title)
        result['guarded_model_consensus']['B3_performed'] = needed
        result['timing_ms'] = {'total': (time.perf_counter() - started) * 1000}
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8152)
    parser.add_argument('--startup-receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.port < 8152:
        raise ValueError('Preserve previous profiles')
    from scripts.run_fanagoria_gallery_candidate import load
    from rshb_vine.square_rescue import SquareRescue
    from rshb_vine.api import create_app
    import uvicorn
    uvicorn.run(create_app(GuardedConsensus(SquareRescue(load(args.startup_receipt)))), host='127.0.0.1', port=args.port)
