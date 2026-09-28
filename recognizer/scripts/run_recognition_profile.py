"""One integrated local recognition profile, preserving the frozen controls."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, verify, seal, sha256


def load(mode='integrated'):
    from scripts.run_vision_text import load as load_vision
    from rshb_vine.recognition_profile import RecognitionProfile
    vision = verify(read_json(ROOT/'runs/vision-text-v1/decision.json'))
    roi = verify(read_json(ROOT/'runs/roi-selection-v1/decision.json'))
    roi_protocol = verify(read_json(ROOT/'runs/roi-selection-v1/replay-resume-v1/protocol.json'))
    plan = verify(read_json(ROOT/'runs/recognition-profile-v1/plan.json'))
    assert plan['vision_decision'] == vision['checksum'] and plan['roi_decision'] == roi['checksum']
    assert sha256(ROOT/'rshb_vine/roi_selection.py') == roi_protocol['source_sha']
    for name, expected in plan['sources'].items():
        assert sha256(ROOT/name) == expected
    parent = load_vision()
    assert parent.manifest == vision['runtime']
    assert parent.manifest['visual_runtime'] == roi_protocol['runtime']
    pipe = RecognitionProfile(parent.base, ROOT/'runs/crop-label-v1/verifier/verifier.json',
                              parent.variant_policy, parent.variant_reader)
    pipe.manifest = seal({'kind': 'integrated-recognition-profile-v1',
        'vision_runtime': parent.manifest, 'roi_source': roi_protocol['source_sha'],
        'plan_checksum': plan['checksum'], 'sources': plan['sources'],
        'stage_order': ['explicit_roi_parent_selection', 'visual_retrieval', 'positive_variant_text'],
        'calibrated': False})
    return pipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['replay', 'serve'])
    parser.add_argument('--port', type=int, default=8104)
    args = parser.parse_args()
    if args.action == 'replay':
        from scripts.run_geometry_loop import replay
        replay('integrated', 'all', 'recognition-profile-v1', load)
    else:
        import uvicorn
        from rshb_vine.api import create_app
        uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
