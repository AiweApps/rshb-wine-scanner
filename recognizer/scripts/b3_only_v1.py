"""B3-only SKU candidate: freeze | describe | replay | serve (loopback, default 8184). Never touches the current pointer."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'describe', 'replay', 'serve'))
    parser.add_argument('--profile', default='config/recognition-b3-only-v1-candidate-v3.json')
    parser.add_argument('--port', type=int, default=8184)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, help='New immutable replay receipt; never overwritten')
    parser.add_argument('--roi', help='Optional original-coordinate target ROI as a JSON array')
    parser.add_argument('--bottles', choices=('addressed', 'all'), default='addressed')
    parser.add_argument('--load', action='store_true', help='describe: also load models and print the ownership proof')
    args = parser.parse_args()
    from rshb_vine.io import seal, write_json
    from rshb_vine.b3_only_v1 import runtime as B
    if args.action == 'freeze':
        path = ROOT / args.profile
        if path.exists():
            parser.error('profile exists; immutable profile preserved')
        profile = seal(B.freeze_body(ROOT))
        write_json(path, profile)
        print(json.dumps({'profile': args.profile, 'checksum': profile['checksum']}))
        return
    if args.action == 'describe':
        out = B.describe(ROOT, args.profile)
        if args.load:
            pipeline = B.B3OnlyRecognition(ROOT, args.profile)
            out.update(models_loaded=True, runtime_checksum=pipeline.manifest['checksum'],
                       ownership=pipeline.ownership, arm=pipeline.manifest['arm'])
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if args.action == 'serve':
        if not 1 <= args.port <= 65535:
            parser.error('port must be between 1 and 65535')
        from rshb_vine.target_contract_v2.runtime import create_app
        import uvicorn
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(('127.0.0.1', args.port))
            listener.listen(128)
            pipeline = B.B3OnlyRecognition(ROOT, args.profile)
            uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=args.port)).run(sockets=[listener])
        finally:
            listener.close()
        return
    if args.input is None or args.output is None:
        parser.error('replay requires --input and --output')
    if args.output.exists():
        parser.error('replay output already exists; immutable receipt preserved')
    roi = json.loads(args.roi) if args.roi is not None else None
    if roi is not None and not isinstance(roi, list):
        parser.error('ROI must be a JSON array')
    source = args.input.resolve()
    data = source.read_bytes()[:20 * 1024 * 1024 + 1]
    loaded = time.perf_counter()
    pipeline = B.B3OnlyRecognition(ROOT, args.profile)
    loaded = time.perf_counter() - loaded
    started = time.perf_counter()
    result = pipeline.recognize(data, roi, args.bottles)
    receipt = seal({'kind': 'b3-only-v1-single-image-replay', 'profile_checksum': pipeline.profile['checksum'],
                    'runtime_checksum': pipeline.manifest['checksum'], 'image_path': str(source),
                    'request_sha256': sha256(data).hexdigest(), 'target_roi': roi, 'target_bottles': args.bottles,
                    'result': result, 'load_seconds': loaded, 'seconds': time.perf_counter() - started,
                    'new_visual_or_OCR_inference': True, 'quality_evaluation': False})
    write_json(args.output, receipt)
    print(json.dumps({'receipt': str(args.output), 'checksum': receipt['checksum'], 'decision': result['decision'],
                      'best_candidate': result.get('best_candidate'),
                      'role_counters': result.get('b3_only_v1', {}).get('role_counters')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
