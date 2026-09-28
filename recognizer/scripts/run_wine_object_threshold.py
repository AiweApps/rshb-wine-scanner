"""Source-reviewed actual-proposal threshold over frozen object classifier."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, verify, seal, sha256


def load():
    from scripts.run_wine_object_profile import load as load_parent
    parent = load_parent()
    source = ROOT / 'runs/wine-object-threshold-v2'
    threshold = verify(read_json(source / 'threshold.json'))
    plan = verify(read_json(source / 'plan.json'))
    assert threshold['plan_checksum'] == plan['checksum']
    assert threshold['verifier_checksum'] == parent.spec['checksum']
    assert threshold['negative_upper'] < threshold['threshold'] < threshold['positive_lower']
    assert threshold['threshold'] == (threshold['positive_lower'] + threshold['negative_upper']) / 2
    old = parent.manifest
    parent.spec = seal({**{k: v for k, v in parent.spec.items() if k != 'checksum'},
                        'threshold': threshold['threshold'],
                        'threshold_admission_checksum': threshold['checksum']})
    parent.manifest = seal({'kind': 'wine-object-actual-proposal-threshold-v2',
                            'parent_runtime': old, 'threshold_checksum': threshold['checksum'],
                            'runner_sha': sha256(__file__), 'calibrated': False})
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8109)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
