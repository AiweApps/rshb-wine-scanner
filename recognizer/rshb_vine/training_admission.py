"""Fit-role preflight for every row, including inherited training records.

Passing this check is necessary, not sufficient: identity, feature provenance,
gallery exposure and the experiment's group split still require admission.
Reads role metadata only, never protected predictions or query pixels.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from rshb_vine.data_protection import load_protection

BASE = 'data/ranker-corpus-inventory-v1/readiness-v2/'
TABLE = 'runs/ranker-dataset-v1-root-admission01/admitted-table.jsonl'


def _groups(row):
    result = set()
    for key in ('capture_group_ids', 'source_group_ids'):
        result.update(str(x) for x in row.get(key, []) if x)
    if row.get('capture_group'):
        result.add(str(row['capture_group']))
    return result


def audit_training_rows(rows, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    protection = load_protection(root)
    denied, groups, sources = defaultdict(set), defaultdict(set), {}

    def read(path):
        raw = (root / path).read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        return ([json.loads(line) for line in raw.splitlines() if line.strip()]
                if path.endswith('.jsonl') else json.loads(raw))

    def block(row, reason, sha_key='image_sha256'):
        sha = row[sha_key]
        if not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError('Invalid role-source image SHA: ' + reason)
        denied[sha].add(reason)
        for group in _groups(row):
            groups[group].add(reason)

    for sha in protection['forbidden_sha256']:
        denied[sha].add('protection-current')
    for role in ('validation', 'test', 'protected_hold'):
        path = f'data/evaluation/web232-v1/{role}.json'
        for row in read(path)['records']:
            block(row, 'web232:' + role, 'sha256')
    for row in read(BASE + 'B/internet-reusable-feature-rows.jsonl'):
        if row['source_partition'] in {
            'slug_whole71_reserved', 'slug_new52_reserved', 'internet21_open_validation'
        }:
            block(row, 'reserved:' + row['source_partition'])
    for row in read(BASE + 'D/internet-source-bindings-v1.jsonl'):
        if row['split_role'] == 'open_evaluation_only_no_fit':
            block(row, 'internet:open_evaluation_only_no_fit', 'original_sha256')
    for row in read(BASE + 'A/kultovo-availability-incremental.jsonl'):
        if any(any(term in role for term in ('calibration_candidate', 'final_test_candidate', 'locked', 'consumed'))
               for role in row['split_role']):
            block(row, 'kultovo:reserved_role', 'original_sha256')
    for row in read(TABLE):
        if row['ranker_split'] == 'development_hold':
            block(row, 'ranker:development_hold')

    decisions, ids = [], set()
    for row in rows:
        qid, sha = row.get('query_id'), row.get('image_sha256', '')
        reasons = set()
        if not qid or qid in ids:
            reasons.add('missing_or_duplicate_query_id')
        ids.add(qid)
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            reasons.add('invalid_image_sha256')
        else:
            reasons.update(denied.get(sha, ()))
        for ancestor in row.get('derivative_sha256', []):
            reasons.update('ancestor:' + r for r in denied.get(ancestor, ()))
        for group in _groups(row):
            reasons.update('group:' + r for r in groups.get(group, ()))
        role = row.get('ranker_split')
        if role not in ('train', 'fit'):
            reasons.add('row_not_explicit_train')
        decisions.append({'query_id': qid, 'image_sha256': sha,
                          'allowed_by_role': not reasons, 'reasons': sorted(reasons)})
    result = {'kind': 'training-role-admission-v1', 'rows': len(decisions),
              'rejected': sum(not r['allowed_by_role'] for r in decisions),
              'denied_image_count': len(denied), 'decisions': decisions,
              'sources_sha256': sources, 'protection_pointer_sha256': protection['pointer_sha256'],
              'identity_or_fit_admission_granted': False}
    result['checksum'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def require_training_admission(rows, repo_root=None):
    """Fail before optimisation if any supplied fit row has a role conflict."""
    result = audit_training_rows(rows, repo_root)
    if result['rejected']:
        rejected = [r['query_id'] for r in result['decisions'] if not r['allowed_by_role']]
        raise ValueError(f"Training role admission rejected {len(rejected)} rows: {rejected[:8]}")
    return result
