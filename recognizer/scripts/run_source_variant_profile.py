"""Verified catalogue variant attributes over the integrated recognition profile."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, verify, sha256, seal


def load():
    from scripts.run_recognition_profile import load as load_parent
    from rshb_vine.gallery_variant_text import GalleryVariantText
    run = ROOT / 'runs/gallery-variant-source-review-v1'
    admission = verify(read_json(run / 'admission.json'))
    signatures = verify(read_json(run / 'gallery/signatures.json'))
    screen = verify(read_json(run / 'cached/protocol.json'))
    report = verify(read_json(run / 'cached/report.json'))
    code_sha = sha256(ROOT / 'rshb_vine/gallery_variant_text.py')
    assert admission['code_sha'] == signatures['parser_sha'] == screen['code_sha'] == code_sha
    assert signatures['source_admission_checksum'] == admission['checksum']
    assert screen['signature_checksum'] == signatures['checksum']
    assert report['protocol_checksum'] == screen['checksum']
    assert all(s['regressions'] == 0 for s in report['summary'].values())
    assert sum(s['fixes'] for s in report['summary'].values()) > 0
    parent = load_parent()
    release = verify(read_json(ROOT / 'runs/recognition-profile-v1/decision.json'))
    assert parent.manifest['checksum'] == release['runtime_snapshot']
    parent.variant_policy = GalleryVariantText(parent.variant_policy, signatures['signatures'])
    parent.manifest = seal({'kind': 'source-variant-profile-v1', 'parent_runtime': parent.manifest,
                            'signature_checksum': signatures['checksum'],
                            'admission_checksum': admission['checksum'],
                            'screen_checksum': report['checksum'], 'calibrated': False,
                            'sources': {'rshb_vine/gallery_variant_text.py': code_sha,
                                        'scripts/run_source_variant_profile.py': sha256(__file__)}})
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8105)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
