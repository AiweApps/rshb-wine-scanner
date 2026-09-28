"""Experimental bounded label rescue over the selected object-filter profile."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, verify, sha256, seal


def load():
    from scripts.run_wine_object_threshold import load as load_parent
    from rshb_vine.label_detector import LabelDetector
    from rshb_vine.label_rescue import LabelRescue
    plan = verify(read_json(ROOT / 'runs/label-rescue-v1/plan.json'))
    artifact = ROOT / 'runs/occlusion-geometry-v1/label-model'
    manifest = verify(read_json(artifact / 'manifest.json'))
    trial = verify(read_json(artifact / 'trial.json'))
    assert manifest['checksum'] == plan['rescue_weights']
    assert manifest['trial_checksum'] == trial['checksum']
    for name, expected in trial['source_sha256'].items():
        assert sha256(ROOT / name) == expected
    parent = load_parent()
    assert parent.manifest['checksum'] == plan['parent_runtime']
    localizer = parent.detector.detector
    assert parent.base.detector is localizer
    wrapper = LabelRescue(localizer, LabelDetector(artifact, 'mps'))
    parent.detector.detector = wrapper
    parent.base.detector = wrapper
    parent.manifest = seal({'kind': 'label-rescue-v1', 'parent_runtime': parent.manifest,
        'plan_checksum': plan['checksum'], 'localizer_id': wrapper.model_id,
        'sources': {name: sha256(ROOT / name) for name in
                    ['rshb_vine/label_rescue.py', 'scripts/run_label_rescue.py']},
        'calibrated': False})
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8110)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
