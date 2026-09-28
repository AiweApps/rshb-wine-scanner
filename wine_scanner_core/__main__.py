"""Recognition core: python -m wine_scanner_core {describe|verify|preflight|serve} (configuration from WINE_SCANNER_CORE_*).

describe  model-free runtime descriptor (the gateway's WINE_SCANNER_EXPECTED_PROFILE)
verify    describe plus SHA-256 of every vendored source and asset file
preflight verify plus the vendored profile chain and receipt guards, without loading models
serve     verify, load the recognizer and serve /health/live, /health/ready and /v1/recognize
"""
import json
import os
import re
import sys

CHECKSUM = re.compile(r'^[0-9a-f]{64}$')


def settings(env=os.environ):
    sha = env.get('WINE_SCANNER_CORE_ASSETS_SHA256', '').strip()
    if not CHECKSUM.match(sha):
        raise ValueError('WINE_SCANNER_CORE_ASSETS_SHA256 must be a lowercase 64-hex SHA-256')
    expected = env.get('WINE_SCANNER_CORE_EXPECTED', '').strip() or None
    if expected is not None and not CHECKSUM.match(expected):
        raise ValueError('WINE_SCANNER_CORE_EXPECTED must be a lowercase 64-hex SHA-256')
    port, threads = int(env.get('WINE_SCANNER_CORE_PORT', '8375')), int(env.get('WINE_SCANNER_CORE_THREADS', '2'))
    if not 1 <= port <= 65535 or not 1 <= threads <= 2:
        raise ValueError('WINE_SCANNER_CORE_PORT must be 1..65535 and WINE_SCANNER_CORE_THREADS 1..2 '
                         '(PP-OCRv6 predict crashes with 4 Paddle threads on Linux/arm64)')
    return {'root': env.get('WINE_SCANNER_CORE_ROOT', '/srv/core'),
            'manifest': env.get('WINE_SCANNER_CORE_MANIFEST', '/srv/pack/manifest.json'), 'sha': sha,
            'expected': expected, 'host': env.get('WINE_SCANNER_CORE_HOST', '127.0.0.1'), 'port': port,
            'threads': threads}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else 'serve'
    if command not in ('describe', 'verify', 'preflight', 'serve'):
        sys.exit(__doc__)
    try:
        config = settings()
    except ValueError as error:
        sys.exit(f'wine_scanner_core: {error}')
    sys.path.insert(0, config['root'])
    from wine_scanner_core.core import Core
    core = Core(config['root'], config['manifest'], config['sha'], config['threads'])
    if config['expected'] and core.descriptor['checksum'] != config['expected']:
        sys.exit('wine_scanner_core: descriptor %s is not WINE_SCANNER_CORE_EXPECTED' % core.descriptor['checksum'])
    report = {'descriptor_checksum': core.descriptor['checksum'], 'descriptor': core.descriptor}
    if command != 'describe':
        report['verified'] = core.verify()
    if command == 'preflight':
        report['preflight'] = core.preflight()
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    if command != 'serve':
        return
    import uvicorn
    from wine_scanner_core.server import CoreRecognition, create_app
    pipeline = CoreRecognition(core, config['threads'])
    print(json.dumps({'loaded': core.descriptor['checksum'], 'load_seconds': pipeline.runtime.load_seconds}), flush=True)
    uvicorn.run(create_app(pipeline), host=config['host'], port=config['port'], access_log=False,
                proxy_headers=False, server_header=False)


if __name__ == '__main__':
    main()
