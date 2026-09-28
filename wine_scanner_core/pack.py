"""Serving asset pack and vendored source ledger: every file the recognizer reads is listed with its SHA-256.

The pack manifest is accepted only when its bytes hash to the configured SHA-256. Every listed file is hashed
before the recognizer is built, and a file present in the pack or in the vendored source tree but absent from its
list is an error, so nothing unreviewed is loaded.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath

PACK_KIND = 'wine-scanner-core-assets-v1'
LEDGER_KIND = 'wine-scanner-core-source-ledger-v1'
LEDGER = 'SOURCES.json'
IGNORED_PARTS = ('__pycache__',)
BLOCK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('', '.', '..') for p in path.parts):
        raise ValueError('Listed path is not a clean relative path: ' + name)
    return path.as_posix()


def listed_tree(root, skip=()):
    root = Path(root)
    found = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symbolic link in a verified tree: ' + str(path.relative_to(root)))
        if path.is_file() and not any(p in IGNORED_PARTS for p in path.relative_to(root).parts):
            found.add(path.relative_to(root).as_posix())
    return found - set(skip)


def check_files(root, entries, workers):
    root = Path(root).resolve(strict=True)

    def one(item):
        name, entry = item
        path = (root / relative(name)).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('Listed path escapes its tree: ' + name)
        size = path.stat().st_size
        if size != entry['bytes'] or sha256_file(path) != entry['sha256']:
            raise ValueError('File differs from its manifest entry: ' + name)
        return size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(one, sorted(entries.items(), key=lambda kv: -kv[1]['bytes'])))


class AssetPack:
    """A sealed manifest; its files sit at their relative paths under ``tree`` (the pack's top-level folders are
    mounted read-only into the vendored root, so the vendored code finds them where it always has)."""

    def __init__(self, manifest_path, tree, manifest_sha256, workers=4):
        raw = Path(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError('Core asset manifest does not match WINE_SCANNER_CORE_ASSETS_SHA256')
        manifest = json.loads(raw)
        if manifest.get('kind') != PACK_KIND or manifest.get('checksum') != canonical(
                {k: v for k, v in manifest.items() if k != 'checksum'}):
            raise ValueError('Unsupported or unsealed core asset manifest')
        self.manifest, self.manifest_sha256 = manifest, manifest_sha256
        self.tree = Path(tree).resolve(strict=True)
        self.files = manifest['files']
        self.writable = [relative(p) for p in manifest.get('writable_dirs', [])]
        self.workers = workers

    def verify(self):
        tops = set(self.top_level())
        extra = {p for p in listed_tree(self.tree) if p.split('/')[0] in tops} - set(self.files)
        extra -= {p for p in extra if any(p.startswith(w + '/') for w in self.writable)}
        if extra:
            raise ValueError('Unlisted files in the asset pack: %s' % sorted(extra)[:5])
        return {'files': len(self.files), 'bytes': check_files(self.tree, self.files, self.workers)}

    def top_level(self):
        return sorted({relative(name).split('/')[0] for name in self.files} | {p.split('/')[0] for p in self.writable})


class SourceLedger:
    """``SOURCES.json`` of the vendored serving tree: exported SHA and the private original it was taken from."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        ledger = json.loads((self.root / LEDGER).read_text())
        if ledger.get('kind') != LEDGER_KIND or ledger.get('checksum') != canonical(
                {k: v for k, v in ledger.items() if k != 'checksum'}):
            raise ValueError('Unsupported or unsealed vendored source ledger')
        self.ledger, self.files = ledger, ledger['files']

    def verify(self, mounted=()):
        present = listed_tree(self.root, skip=(LEDGER,))
        present = {p for p in present if p.split('/')[0] not in set(mounted)}
        if present != set(self.files):
            raise ValueError('Vendored tree differs from its ledger: extra %s missing %s'
                             % (sorted(present - set(self.files))[:5], sorted(set(self.files) - present)[:5]))
        check_files(self.root, self.files, 4)
        return {'files': len(self.files), 'changed': sorted(p for p, e in self.files.items()
                                                            if e['sha256'] != e['original_sha256'])}

    def aliases(self):
        """Exported SHA -> original SHA for vendored files the export had to change; empty when all are byte copies."""
        return {p: (e['sha256'], e['original_sha256']) for p, e in self.files.items()
                if e['sha256'] != e['original_sha256']}
