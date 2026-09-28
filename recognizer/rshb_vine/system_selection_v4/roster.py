"""Train roster, root role preflight and grouped fit/internal-selection split for selector v4.

Every candidate row is passed to the root preflight; rejected rows stay in the roster
with their reasons instead of disappearing. Rows whose per-case evidence was already
read while designing features are forced into fit, so internal selection holds only
components nobody has examined case by case. Assignment uses a v4-only hash salt and
no model output or label outcome.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from rshb_vine.io import read_json, read_jsonl, seal, sha256
from rshb_vine.ranker_integration_v3.split import components, row_keys

REAL_TABLE = 'runs/ranker-dataset-v1-root-admission01/admitted-table.jsonl'
EXTRA_ROWS = 'runs/ranker-completion-v2/admitted-extra-rows.jsonl'
PENDING_INTERNET = 'data/ranker-integration-v3/internet/candidate-rows.jsonl'
EXAMINED_SOURCES = {
    'error_register': 'runs/current-error-causes-v1/roles.json',
    'v3_distinguishability': 'runs/ranker-integration-v3/distinguishability/summary.json',
    'v3_evaluation': 'runs/ranker-integration-v3/evaluation.json',
}
SALT = 'system-selection-v4/internal-selection'
SELECTION_ROW_SHARE = 0.2
SOURCES = (REAL_TABLE, EXTRA_ROWS, *EXAMINED_SOURCES.values(), 'rshb_vine/training_admission.py',
           'rshb_vine/ranker_integration_v3/split.py', 'rshb_vine/system_selection_v4/roster.py')


def candidate_rows(root):
    root = Path(root)
    table = read_jsonl(root / REAL_TABLE)
    rows = [dict(r, cohort='real430') for r in table if r['ranker_split'] == 'train']
    rows += [dict(r, cohort='real_extra_v2') for r in read_jsonl(root / EXTRA_ROWS)]
    dev = [dict(r, cohort='development108') for r in table if r['ranker_split'] == 'development_hold']
    return rows, dev


def examined_query_ids(root, rows):
    """Rows whose individual evidence was already read by earlier feature-design work."""
    root = Path(root)
    by_sha = defaultdict(set)
    for row in rows:
        by_sha[row['image_sha256']].add(row['query_id'])
    examined = defaultdict(set)
    for item in read_json(root / EXAMINED_SOURCES['error_register'])['rows']:
        for qid in by_sha.get(item.get('sha256'), ()):
            examined[qid].add('error_register:' + str(item['n']))
    ids = {r['query_id'] for r in rows}
    summary = read_json(root / EXAMINED_SOURCES['v3_distinguishability'])
    named = [r['query_id'] for r in summary.get('wrong_rows_top_negative', [])]
    named += list(summary.get('wrong_side_rows', []))
    named += [r['query_id'] for r in summary.get('weak_rows_detail', []) if isinstance(r, dict)]
    for qid in named:
        if qid in ids:
            examined[qid].add('v3_distinguishability')
    for row in read_json(root / EXAMINED_SOURCES['v3_evaluation'])['rows']:
        if row.get('outcome') != 'both_right' and row['query_id'] in ids:
            examined[row['query_id']].add('v3_evaluation:' + row['outcome'])
    return {qid: sorted(reasons) for qid, reasons in examined.items()}


def _h(text):
    return hashlib.sha256((SALT + ':' + text).encode()).hexdigest()


def build_roster(root):
    """Return the sealed roster; raises only when the admitted remainder still fails preflight."""
    from rshb_vine.training_admission import audit_training_rows, require_training_admission
    root = Path(root)
    rows, dev = candidate_rows(root)
    audit = audit_training_rows(rows, root)
    decisions = {d['query_id']: d for d in audit['decisions']}
    admitted = [r for r in rows if decisions[r['query_id']]['allowed_by_role']]
    excluded = [{'query_id': r['query_id'], 'image_sha256': r['image_sha256'], 'cohort': r['cohort'],
                 'reasons': decisions[r['query_id']]['reasons']} for r in rows
                if not decisions[r['query_id']]['allowed_by_role']]
    required = require_training_admission(admitted, root)
    forbidden = {k for r in dev for k in row_keys(r) if not k.startswith('producer:')}
    touching = sorted({r['query_id'] for r in admitted if forbidden & set(row_keys(r))})
    if touching:
        raise ValueError('Admitted train row shares a development key: ' + ','.join(touching[:3]))
    comp = components(admitted)
    members = defaultdict(list)
    for r in admitted:
        members[comp[r['query_id']]].append(r['query_id'])
    examined = examined_query_ids(root, admitted)
    locked = {comp[q] for q in examined}
    target = SELECTION_ROW_SHARE * len(admitted)
    selection, size = set(), 0
    for c in sorted(set(members) - locked, key=_h):
        if size >= target:
            break
        if size + len(members[c]) <= target * 1.25:
            selection.add(c)
            size += len(members[c])
    assignment = {r['query_id']: 'internal_selection' if comp[r['query_id']] in selection else 'fit'
                  for r in admitted}
    records = [{'query_id': r['query_id'], 'image_sha256': r['image_sha256'], 'cohort': r['cohort'],
                'component': comp[r['query_id']], 'split': assignment[r['query_id']],
                'examined_before_v4': examined.get(r['query_id'], []),
                'receipt_path': r['receipt_path'], 'receipt_sha256': r.get('receipt_sha256'),
                'loss_ignore_candidate_ids': sorted(r.get('loss_ignore_candidate_ids', [])),
                'ground_truth_products': sorted(r['ground_truth']['current_products']),
                'retrieval_miss': bool(r.get('retrieval_miss'))}
               for r in sorted(admitted, key=lambda r: r['query_id'])]
    pending = [r['query_id'] for r in read_jsonl(root / PENDING_INTERNET)] if (root / PENDING_INTERNET).exists() else []
    return seal({
        'kind': 'system-selection-v4-roster', 'salt': SALT, 'selection_row_share': SELECTION_ROW_SHARE,
        'candidate_rows': len(rows), 'admitted_rows': len(admitted), 'excluded_rows': excluded,
        'role_audit_checksum': audit['checksum'], 'required_admission_checksum': required['checksum'],
        'role_audit_sources_sha256': audit['sources_sha256'],
        'protection_pointer_sha256': audit['protection_pointer_sha256'],
        'identity_or_fit_admission_granted': False,
        'components': len(members), 'examined_components_forced_to_fit': len(locked),
        'examined_rows': len(examined), 'selection_components': len(selection),
        'counts': dict(Counter(assignment.values())),
        'counts_by_cohort': {s: dict(Counter(r['cohort'] for r in admitted if assignment[r['query_id']] == s))
                             for s in ('fit', 'internal_selection')},
        'development108': sorted(r['query_id'] for r in dev),
        'pending_not_train': {'path': PENDING_INTERNET, 'rows': len(pending),
                              'reason': 'source/identity admission pending with root; not passed as train'},
        'sources_sha256': {p: sha256(root / p) for p in SOURCES},
        'rows': records,
        'label_or_model_outcome_used_for_assignment': False,
    })


def load_rows(root, roster, split=None):
    """Full admitted rows for the requested split, re-checked against the sealed roster."""
    root = Path(root)
    rows, dev = candidate_rows(root)
    by_id = {r['query_id']: r for r in rows + dev}
    if split == 'development108':
        return [by_id[q] for q in roster['development108']]
    wanted = [r for r in roster['rows'] if split is None or r['split'] == split]
    result = []
    for record in wanted:
        row = by_id[record['query_id']]
        if row['image_sha256'] != record['image_sha256'] or row['receipt_path'] != record['receipt_path']:
            raise ValueError('Roster row changed since freeze: ' + record['query_id'])
        result.append(dict(row, ranker_split='train', v4_split=record['split'], component=record['component']))
    return result
