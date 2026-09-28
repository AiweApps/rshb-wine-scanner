"""Run the frozen catalogue phrase candidate over the source variant profile."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, read_jsonl, verify, seal, sha256


def load():
    from scripts.run_source_variant_profile import load as load_parent
    from rshb_vine.catalog_phrase_text import CatalogPhrase
    admission = verify(read_json(ROOT / 'runs/catalog-phrase-release-v1/admission-v2.json'))
    protocol = verify(read_json(ROOT / 'runs/catalog-phrase-v2/cached/protocol.json'))
    report = verify(read_json(ROOT / 'runs/catalog-phrase-v2/cached/report.json'))
    assert report['checksum'] == admission['screen_checksum']
    assert report['protocol_checksum'] == protocol['checksum'] == admission['screen_protocol_checksum']
    assert all(r['regressions'] == 0 for r in report['summary'].values())
    assert sum(r['fixes'] for r in report['summary'].values()) > 0
    source = ROOT / 'rshb_vine/catalog_phrase_text.py'
    catalog = ROOT / 'data/normalized/catalog.jsonl'
    assert sha256(source) == protocol['code_sha'] == admission['policy_sha']
    assert sha256(catalog) == admission['catalog_sha']
    parent = load_parent()
    assert parent.manifest['checksum'] == admission['parent_snapshot']
    parent.variant_policy = CatalogPhrase(parent.variant_policy, read_jsonl(catalog))
    parent.manifest = seal({
        'kind': 'catalog-phrase-profile-v1', 'parent_runtime': parent.manifest,
        'admission_checksum': admission['checksum'], 'calibrated': False,
        'sources': {'rshb_vine/catalog_phrase_text.py': sha256(source),
                    'scripts/run_catalog_phrase_profile.py': sha256(__file__)},
    })
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8107)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
