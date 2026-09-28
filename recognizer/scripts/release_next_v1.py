"""Release next v1: candidate draft/freeze/describe/serve/replay of bde4fa52 + a root-named D/N recipe, then
preflight/stage/status/verify/activate/rollback/recover/inventory of the factory F12 release through scripts/recognize.py.

Candidate (pre-activation): loopback 8225 under the live factory F11, loading the static bde4fa52 profile file; factory,
receipts, recognize.py, the current pointer and the services on 8175/8187 (and the N standalone on 8224) are never
touched. ``freeze --mode diagnostic --admission PATH`` needs a sealed root diagnostic protocol and is never stageable;
``freeze --mode quality_gate --admission PATH`` needs the root gate (decision quality_gate_passed). Both name the recipe
(ordered subset of D, N) and the paths/checksums of the component descriptors. ``draft`` only previews; component
descriptor paths for it are given as ``--component D=PATH --component N=PATH``.
Release: factory F12 and receipts C12 (drafts in runs/release-next-v1/integration/dispatch) add the dispatch12 B3-only
base and the release-next kinds; ``preflight`` is read-only over config/; stage needs the root stage decision, a
quality_gate descriptor and the live atlas_repair_active state (F11, R3, C11, pointer 766d4acd), seals a restoration
snapshot of those bytes, archives F12/C12 and writes the immutable dispatch12 chain, gallery migration v8, dispatch
migration v11, the release profile and its manifest. A switch journals its full plan before the first write. The
service on 8175 must be stopped around activate/rollback/recover.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import read_json, seal, verify, write_json  # noqa: E402
from rshb_vine.release_next_v1 import candidate as C  # noqa: E402
from rshb_vine.release_next_v1 import recipe as K  # noqa: E402


def _write_once(root, path, doc):
    target = root / path
    if target.exists():
        raise SystemExit(path + ' already exists; immutable file preserved')
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, doc)


def _paths(values):
    out = {}
    for value in values or ():
        name, _, path = value.partition('=')
        if name not in K.COMPONENTS or not path or name in out:
            raise SystemExit('--component must be D=PATH or N=PATH, once each: ' + value)
        out[name] = path
    return out


# ---- candidate ----

def serve(root, port, descriptor):
    from rshb_vine.target_contract_v2.runtime import create_app
    if port in C.FORBIDDEN_PORTS:
        raise SystemExit('Port %d belongs to the live service or another candidate' % port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = C.ReleaseNextCandidate(root, descriptor)
        import uvicorn
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()


def replay(root, descriptor, source, output, roi, bottles, capture):
    if output.exists():
        raise SystemExit('replay output already exists; immutable receipt preserved')
    data = source.read_bytes()
    pipeline = C.ReleaseNextCandidate(root, descriptor, capture=capture)
    started = time.perf_counter()
    result = pipeline.recognize(data, roi, bottles)
    receipt = seal({'kind': 'release-next-v1-candidate-replay', 'image_path': str(source),
                    'request_sha256': hashlib.sha256(data).hexdigest(), 'target_roi': roi, 'target_bottles': bottles,
                    'descriptor_checksum': pipeline.descriptor['checksum'], 'runtime_checksum': pipeline.manifest['checksum'],
                    'result': result, 'captured_n_scope': pipeline.last_scope if capture else None,
                    'seconds': time.perf_counter() - started, 'new_visual_or_OCR_inference': True,
                    'quality_evaluation': False})
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == '.gz':
        with gzip.open(output, 'wt') as f:
            json.dump(receipt, f, ensure_ascii=False)
    else:
        write_json(output, receipt)
    block = result.get(K.BLOCK) or {}
    return {'receipt': str(output), 'checksum': receipt['checksum'], 'decision': result.get('decision'),
            'best_candidate': result.get('best_candidate'),
            K.BLOCK: {k: block.get(k) for k in ('recipe', 'profile_checksum', 'admission_mode', 'components')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('draft', 'freeze', 'describe', 'serve', 'replay', 'preflight', 'stage', 'status',
                                           'verify', 'activate', 'rollback', 'recover', 'inventory'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int)
    parser.add_argument('--mode', choices=tuple(C.MODES), help='freeze: diagnostic or quality_gate')
    parser.add_argument('--admission', help='freeze: root diagnostic protocol or root gate path')
    parser.add_argument('--component', action='append', help='draft: D=PATH or N=PATH component descriptor')
    parser.add_argument('--descriptor', help='serve/replay/describe: candidate descriptor path (default quality_gate one)')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--roi', help='replay: optional original-coordinate target ROI as a JSON array')
    parser.add_argument('--bottles', choices=('addressed', 'all'), default='addressed')
    parser.add_argument('--capture', action='store_true', help='replay: keep the N request scope')
    parser.add_argument('--decision-checksum', help='stage: checksum of the root stage decision')
    parser.add_argument('--confirm', help='activate: checksum of the staged release profile')
    args = parser.parse_args()
    root = args.root.resolve()
    descriptor = args.descriptor or C.DESCRIPTOR
    if args.action == 'draft':
        doc = C.draft(root, _paths(args.component))
        (root / C.DRAFT).parent.mkdir(parents=True, exist_ok=True)
        write_json(root / C.DRAFT, doc, replace=True)
        print(json.dumps({'draft': C.DRAFT, 'checksum': doc['checksum'], 'servable': False,
                          'components': {c: r['checksum'] for c, r in doc['components'].items()},
                          'components_blocked': doc['components_blocked']}, ensure_ascii=False))
        return
    if args.action == 'freeze':
        if not (args.mode and args.admission):
            raise SystemExit('freeze needs --mode and --admission')
        doc = C.freeze(root, args.mode, args.admission)
        path = args.descriptor or C.default_path(args.mode, doc['recipe'])
        _write_once(root, path, doc)
        print(json.dumps({'descriptor': path, 'checksum': doc['checksum'], 'recipe': doc['recipe'],
                          'mode': args.mode, 'release_admissible': doc['release_admissible']}))
        return
    if args.action == 'describe':
        loaded = C.load_descriptor(root, descriptor) if (root / descriptor).is_file() else None
        draft = verify(read_json(root / C.DRAFT)) if (root / C.DRAFT).is_file() else None
        print(json.dumps({'candidate_descriptor': loaded, 'draft_checksum': draft and draft['checksum'],
                          'parent': C.check_parent(root), 'staged_release': release_staged(root),
                          'models_loaded': False}, ensure_ascii=False, indent=2))
        return
    if args.action == 'serve':
        serve(root, args.port or C.PORT, descriptor)
        return
    if args.action == 'replay':
        if args.input is None or args.output is None:
            raise SystemExit('replay needs --input and a new --output')
        roi = json.loads(args.roi) if args.roi is not None else None
        if roi is not None and not isinstance(roi, list):
            raise SystemExit('ROI must be a JSON array')
        print(json.dumps(replay(root, descriptor, args.input.resolve(), args.output, roi, args.bottles, args.capture),
                         ensure_ascii=False))
        return
    release_main(root, args)


def release_staged(root):
    from rshb_vine.release_next_v1 import lifecycle
    return lifecycle.staged(root)


def release_main(root, args):
    """Release lifecycle lives in lifecycle.py, outside the candidate source pins."""
    from rshb_vine.release_next_v1 import lifecycle
    lifecycle.main(root, args)


if __name__ == '__main__':
    main()
