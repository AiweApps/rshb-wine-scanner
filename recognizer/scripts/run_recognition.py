"""Describe, serve or replay one explicit, version-bound recognition profile."""
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
    parser.add_argument('action', choices=('serve', 'replay', 'describe'))
    parser.add_argument('--profile', type=Path, default=ROOT / 'config/recognition-current.json')
    parser.add_argument('--port', type=int, default=8175)
    parser.add_argument('--input', type=Path, help='One original image for an explicit end-to-end replay')
    parser.add_argument('--output', type=Path, help='New immutable replay receipt; never overwritten')
    parser.add_argument('--roi', help='Optional original-coordinate target ROI as a JSON array')
    args = parser.parse_args()
    from rshb_vine.recognition_runtime import RecognitionRuntime, describe_profile
    from rshb_vine.io import seal, write_json
    if args.action == 'describe':
        if args.input is not None or args.output is not None or args.roi is not None:
            parser.error('describe accepts no image/output/ROI')
        print(json.dumps(describe_profile(ROOT, args.profile), ensure_ascii=False, indent=2))
        return
    if args.action == 'serve':
        if args.input is not None or args.output is not None or args.roi is not None:
            parser.error('serve accepts no image/output/ROI')
        if not 1 <= args.port <= 65535:
            parser.error('port must be between 1 and 65535')
        # Reserve the actual socket before loading models. A failed bind leaves
        # existing processes intact and avoids a preflight/startup race.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(('127.0.0.1', args.port))
            listener.listen(128)
            pipeline = RecognitionRuntime(ROOT, args.profile)
            from rshb_vine.recognition_api import create_app
            import uvicorn
            config = uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=args.port)
            uvicorn.Server(config).run(sockets=[listener])
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
    with source.open('rb') as stream:
        data = stream.read(20 * 1024 * 1024 + 1)
    pipeline = RecognitionRuntime(ROOT, args.profile)
    started = time.perf_counter()
    result = pipeline.recognize(data, roi)
    receipt = seal({'kind': 'recognition-single-image-replay-v1',
        'profile_checksum': pipeline.profile['checksum'], 'runtime_checksum': pipeline.manifest['checksum'],
        'image_path': str(source), 'request_sha256': sha256(data).hexdigest(), 'request_bytes': len(data),
        'file_bytes': source.stat().st_size, 'target_roi': roi, 'result': result,
        'seconds': time.perf_counter() - started, 'new_visual_or_OCR_inference': True,
        'quality_evaluation': False})
    write_json(args.output, receipt)
    print(json.dumps({'receipt': str(args.output), 'checksum': receipt['checksum'],
                      'decision': result['decision'], 'best_candidate': result.get('best_candidate')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
