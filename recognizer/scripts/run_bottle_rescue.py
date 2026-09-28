"""Run the bounded low-confidence bottle rescue candidate."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, verify, seal, sha256


def load():
    from scripts.run_label_rescue import load as load_parent
    from rshb_vine.bottle_rescue import BottleRescueProfile
    parent = load_parent()
    plan = verify(read_json(ROOT / 'runs/bottle-rescue-v1/plan.json'))
    assert parent.manifest['checksum'] == plan['parent_runtime']
    parent.__class__ = BottleRescueProfile
    parent.manifest = seal({'kind': 'bottle-rescue-v1', 'parent_runtime': parent.manifest,
                            'plan_checksum': plan['checksum'], 'calibrated': False,
                            'sources': {name: sha256(ROOT / name) for name in
                                        ['rshb_vine/bottle_rescue.py', 'scripts/run_bottle_rescue.py']}})
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8111)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
