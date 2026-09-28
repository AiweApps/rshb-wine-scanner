"""Load the admitted wine-object candidate for full pipeline comparison."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rshb_vine.io import read_json, verify, seal, sha256


def load():
    from scripts.run_catalog_phrase_profile import load as load_parent
    from rshb_vine.wine_object_profile import WineObjectProfile
    run = ROOT / 'runs/wine-object-fit-v1'
    admission = verify(read_json(run / 'candidate-admission.json'))
    spec = verify(read_json(run / 'verifier.json'))
    audit = verify(read_json(run / 'feature-audit.json'))
    assert admission['status'] == 'admitted_for_full_pipeline_comparison_only'
    assert spec['checksum'] == admission['verifier_checksum']
    assert audit['checksum'] == admission['feature_audit_checksum']
    assert not audit['rows_above_tolerance'] and audit['n'] == 823
    for name, expected in admission['source_sha'].items():
        assert sha256(ROOT / name) == expected
    parent = load_parent()
    plan = verify(read_json(run / 'plan.json'))
    assert parent.manifest['checksum'] == plan['runtime_baseline']
    assert spec['encoder_id'] == audit['encoder_id'] == parent.base.encoder_id
    assert spec['baseline_runtime'] == parent.base.manifest['runtime_descriptor_checksum']
    parent.__class__ = WineObjectProfile
    parent.spec = spec
    parent.coef = np.array(spec['coef'])
    parent.bias = spec['intercept']
    parent.manifest = seal({
        'kind': 'wine-object-profile-v1', 'parent_runtime': parent.manifest,
        'verifier_checksum': spec['checksum'], 'admission_checksum': admission['checksum'],
        'calibrated': False, 'sources': {
            name: sha256(ROOT / name) for name in
            ['rshb_vine/wine_object_profile.py', 'scripts/run_wine_object_profile.py']},
    })
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8108)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
