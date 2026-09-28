"""Reference additions v1 (component D): describe / build / verify / freeze / install-check over bde4fa52.

``describe`` and ``verify`` load no model. ``build`` encodes the 4 admitted views with the frozen B4343 encoder and
writes the immutable build folder once; ``freeze`` seals the descriptor over that build; ``install-check`` loads the
bde4fa52 graph from its immutable profile (models load, no query inference) and runs the full preflight, then installs
D in memory only unless ``--preflight-only``; the report (ready, failed checks or partial steps) is written once. No
service, pointer, factory, profile or frozen source is written; root owns HTTP admission and activation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import local_path, write_json  # noqa: E402
from rshb_vine.reference_additions_v1 import admission as A  # noqa: E402
from rshb_vine.reference_additions_v1 import component as D  # noqa: E402
from rshb_vine.reference_additions_v1 import gallery as G  # noqa: E402


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('describe')
    build = sub.add_parser('build')
    build.add_argument('--device', default='mps')
    build.add_argument('--out', default=G.OUT)
    verify = sub.add_parser('verify')
    verify.add_argument('--out', default=G.OUT)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--out', default=G.OUT)
    freeze.add_argument('--descriptor', default=D.DESCRIPTOR)
    check = sub.add_parser('install-check')
    check.add_argument('--descriptor', default=D.DESCRIPTOR)
    check.add_argument('--report', default='runs/release-next-v1/references/install-check-v5.json')
    check.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'describe':
        config, report = A.load(ROOT)
        gallery, _, _ = G.parent(ROOT)
        _print({'admission': report, 'planned_rows': G.new_rows(ROOT, config, G.OUT),
                'parent_rows': len(gallery['references']), 'sources_sha256': D.sources_sha(ROOT), 'model_loaded': False})
    elif args.command == 'build':
        _print(G.build(ROOT, args.device, args.out))
    elif args.command == 'verify':
        doc, array, receipt = G.load(ROOT, args.out)
        _print({'rows': len(doc['references']), 'shape': list(array.shape), 'gallery_sha256': receipt['gallery_sha256'],
                'vectors_sha256': receipt['vectors_sha256'], 'prefix_byte_identical': True, 'model_loaded': False})
    elif args.command == 'freeze':
        path = local_path(ROOT, args.descriptor)
        if path.exists():
            raise SystemExit('Descriptor already exists: ' + args.descriptor)
        descriptor = D.freeze(ROOT, args.out)
        write_json(path, descriptor)
        _print({'descriptor': args.descriptor, 'checksum': descriptor['checksum']})
    else:
        from rshb_vine.atlas_repair_release_v1.release import AtlasRepairRelease
        descriptor = D.load_descriptor(ROOT, args.descriptor)
        report = local_path(ROOT, args.report)
        if report.exists():
            raise SystemExit('Report already exists: ' + args.report)
        graph = AtlasRepairRelease(ROOT, D.PARENT_PROFILE)
        try:
            if args.preflight_only:
                result = dict(D.preflight(graph, ROOT, descriptor)[0], mode='preflight_only')
            else:
                state, record = D.install(graph, ROOT, descriptor)
                result = dict(record, mode='install', full_preflight=state['preflight'])
        except D.PreflightError as error:
            result = dict(error.report, mode='install', status='preflight_failed')
        except Exception as error:
            result = dict(getattr(graph, D.BLOCK, None) or {}, mode='install', status='failed',
                          error='%s: %s' % (type(error).__name__, error))
        write_json(report, result)
        _print(result)


if __name__ == '__main__':
    main()
