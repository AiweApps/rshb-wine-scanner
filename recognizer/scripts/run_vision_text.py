"""Guarded macOS Vision text challenger over frozen upright visual recognition."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rshb_vine.io import read_json, read_jsonl, verify, seal, sha256


def load():
    run = ROOT/'runs/vision-text-v1'
    plan = verify(read_json(run/'plan.json'))
    transport = verify(read_json(run/'transport-v2.json'))
    assert transport['parent_plan'] == plan['checksum']
    protocol = verify(read_json(run/'current-crop/protocol.json'))
    report = verify(read_json(run/'current-crop/report.json'))
    assert protocol['vision_plan'] == plan['checksum']
    assert report['protocol_checksum'] == protocol['checksum']
    summary = report['summary']
    assert {k: v['n'] for k, v in summary.items()} == {
        'kultovo_open': 390, 'single_main_bottle': 21, 'multiple_visible_bottles': 15}
    assert summary['kultovo_open']['after_top1'] >= 364
    assert summary['single_main_bottle']['after_top1'] >= 17
    assert all(v['errors'] == 0 for v in summary.values())
    for name, expected in plan['code_sha256'].items():
        if name != 'rshb_vine/vision_ocr.py':
            assert sha256(ROOT/name) == expected
    for name, expected in transport['sources'].items():
        assert sha256(ROOT/name) == expected
    executable = run/'vision-ocr'
    assert sha256(executable) == plan['binary_sha256']
    catalog = ROOT/'data/normalized/catalog.jsonl'
    references = ROOT/'data/index/label-first-adapted-real-v2/references.json'
    screen = verify(read_json(ROOT/'runs/positive-variant-text-v2/cached-screen/protocol.json'))
    assert sha256(catalog) == screen['catalog_sha']
    assert sha256(references) == screen['reference_sha']
    from scripts.run_oriented_geometry import load as load_parent
    from rshb_vine.vision_ocr import VisionOCR
    from rshb_vine.positive_variant_text import PositiveVariantText
    from rshb_vine.variant_text_pipeline import VariantTextPipeline
    parent = load_parent()
    assert parent.manifest == verify(read_json(ROOT/'runs/geometry-upright-v1/decision.json'))['runtime']
    reader = VisionOCR(executable)
    assert reader.info == plan['reader_info']
    pipe = VariantTextPipeline(parent.base, ROOT/'runs/crop-label-v1/verifier/verifier.json',
        PositiveVariantText(read_jsonl(catalog), {r['slug'] for r in read_json(references)}), reader)
    pipe.manifest = seal({'kind': 'vision-positive-variant-v1', 'visual_runtime': parent.manifest,
        'plan': plan['checksum'], 'current_crop_report': report['checksum'],
        'transport': transport['checksum'],
        'reader_info': reader.info, 'worker_sha': plan['binary_sha256'],
        'sources': {name: sha256(ROOT/name) for name in [
            'scripts/run_vision_text.py', 'rshb_vine/variant_text_pipeline.py',
            'rshb_vine/vision_ocr.py', 'rshb_vine/positive_variant_text.py']},
        'calibrated': False, 'platform': 'macOS only'})
    return pipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8103)
    args = parser.parse_args()
    import uvicorn
    from rshb_vine.api import create_app
    uvicorn.run(create_app(load()), host='127.0.0.1', port=args.port)
