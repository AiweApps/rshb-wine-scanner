"""Single fail-closed reader for current protection metadata; no implicit legacy fallback."""
import hashlib
import json
import re
from pathlib import Path


def load_protection(repo_root=None, *, require_final=True):
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
    pointer_path = root / 'data/ranker-corpus-inventory-v1/protection-current.json'
    pointer_bytes = pointer_path.read_bytes()
    pointer = json.loads(pointer_bytes)
    if pointer.get('schema_version') != 'protection-current-v1':
        raise ValueError('Unsupported protection pointer schema')
    if require_final and pointer.get('status') != 'reviewed_metadata_closure':
        raise ValueError('Current protection closure is not finalized')
    def checked(spec):
        path = (root / spec['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Protection artifact escaped repository')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Protection checksum mismatch: ' + spec['path'])
        return raw
    summary = json.loads(checked(pointer['summary']))
    memberships = [json.loads(line) for line in checked(pointer['membership']).splitlines() if line.strip()]
    derivatives = [json.loads(line) for line in checked(pointer['derivatives']).splitlines() if line.strip()]
    for row in memberships + derivatives:
        sha = row.get('sha256')
        if not isinstance(sha, str) or re.fullmatch(r'[0-9a-f]{64}', sha) is None:
            raise ValueError('Malformed protected image SHA256')
    for row in derivatives:
        parent = row.get('parent_sha256')
        if not isinstance(parent, str) or re.fullmatch(r'[0-9a-f]{64}', parent) is None:
            raise ValueError('Malformed protected derivative parent SHA256')
    if len(derivatives) != summary['derivative_count']:
        raise ValueError('Protection derivative count mismatch')
    for path, sha in summary['sources'].items():
        checked({'path': path, 'sha256': sha})
    if len(memberships) != summary['memberships'] or len({r['sha256'] for r in memberships}) != summary['unique_sha']:
        raise ValueError('Protection counts mismatch')
    return {'memberships': memberships, 'derivatives': derivatives,
            'forbidden_sha256': sorted({r['sha256'] for r in memberships + derivatives}),
            'pointer': pointer, 'pointer_sha256': hashlib.sha256(pointer_bytes).hexdigest(),
            'summary': summary}
