"""Bind an exported web asset pack to the recognition core's runtime descriptor.

The Linux core serves the same catalogue and gallery as its parent release, so its cards and thumbnails are the
parent pack's files; only the manifest changes: it names the core descriptor and records the pack it came from.
Every source file is SHA-checked; files are cloned (copy-on-write where the filesystem supports it), never linked.

  python tools/bind_web_assets.py --source ASSETS/web --source-sha256 SHA --profile CORE_DESCRIPTOR --out ASSETS/web-core
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

KIND = 'wine-scanner-web-assets-v1'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clone(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'darwin' and subprocess.run(['cp', '-c', str(src), str(dst)]).returncode == 0:
        return
    shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--profile', required=True, help='Core runtime descriptor (python -m wine_scanner_core describe)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = (args.source / 'manifest.json').read_bytes()
    if sha256(raw) != args.source_sha256:
        raise SystemExit('Source manifest does not match --source-sha256')
    manifest = json.loads(raw)
    if manifest.get('kind') != KIND:
        raise SystemExit('Unsupported source manifest kind')
    staging = args.out.with_name(args.out.name + '.partial')
    if args.out.exists() or staging.exists():
        raise SystemExit(f'Refusing to overwrite {args.out} (or its .partial staging directory)')
    for name, entry in manifest['files'].items():
        src, dst = args.source / name, staging / name
        if sha256(src.read_bytes()) != entry['sha256']:
            raise SystemExit('Source file changed: ' + name)
        clone(src, dst)
        if sha256(dst.read_bytes()) != entry['sha256']:
            raise SystemExit('Copy differs: ' + name)
    bound = dict(manifest, profile_checksum=args.profile,
                 bound_from={'manifest_sha256': args.source_sha256, 'profile_checksum': manifest['profile_checksum'],
                             'reason': 'same catalogue and gallery; the recognizer runs as the Linux CPU core'})
    data = json.dumps(bound, ensure_ascii=False, sort_keys=True, indent=1).encode() + b'\n'
    (staging / 'manifest.json').write_bytes(data)
    staging.rename(args.out)
    print(json.dumps({'out': str(args.out), 'manifest_sha256': sha256(data), 'profile_checksum': args.profile}))


if __name__ == '__main__':
    main()
