"""Run the separate in-process catalogue/evidence/geometry candidate."""
from pathlib import Path
import argparse
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8173)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'runs/verified-dual-recognition-v1/protocol.json')
    parser.add_argument('--check-only', action='store_true',
                        help='Check source hashes without loading models or serving.')
    args = parser.parse_args()
    from rshb_vine.verified_dual_recognition import VerifiedDualRecognition
    from rshb_vine.unified_recognition import verify_protocol
    protocol = verify_protocol(ROOT, args.protocol)
    if args.check_only:
        print('Verified unified protocol:', protocol['checksum'])
        sys.exit(0)
    if args.port != 8173:
        raise ValueError('This candidate reserves port 8173; keep existing services unchanged')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Port already occupied; do not start a duplicate service')
    from rshb_vine.api import create_app
    import uvicorn
    uvicorn.run(create_app(VerifiedDualRecognition(ROOT, args.protocol)),
                host='127.0.0.1', port=args.port)
