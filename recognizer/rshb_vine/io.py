"""Canonical hashes and immutable, atomic artifact publication."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    from rshb_vine.evaluation.reset_access import check_read
    check_read(path)
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    from rshb_vine.evaluation.reset_access import check_read
    check_read(path)
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)  # atomic no-overwrite publication
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def seal(payload):
    return dict(payload, checksum=digest(payload))


def verify(document):
    if document.get('checksum') != digest({k: v for k, v in document.items() if k != 'checksum'}):
        raise ValueError('Manifest checksum mismatch')
    return document


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes dataset root')
    return path
