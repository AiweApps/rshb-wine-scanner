"""Immutable scope of the 20 no-runtime-target intake rows and their allowed controls."""
from pathlib import Path

from rshb_vine.io import read_json, read_jsonl, seal, sha256, verify, write_json

OUT = 'runs/target-misses-v1'
INTAKE = 'runs/runtime-training-intake-v1'
LEDGER = INTAKE + '/ledger.jsonl'
SUMMARY = INTAKE + '/summary.json'
SNAPSHOT_ROWS = 'data/runtime-training-intake-v1/input-snapshot.jsonl'
NEGATIVES = 'data/bottle-instance-v1/admitted-v2.json'
PROFILE = 'config/recognition-current.json'
STRAPI = 'data/extracted/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads'
MISS = 'no_runtime_target_e2e_miss'
CONTROL_OUTCOMES = ('rankable_draft_eligible', 'held_rankable_row')
CONTROLS_PER_SCOPE = 10


def _protected(root):
    from rshb_vine.data_protection import load_protection
    protection = load_protection(root)
    return set(protection['forbidden_sha256']), protection['pointer_sha256']


def build(root):
    root = Path(root).resolve()
    summary = read_json(root / SUMMARY)
    ledger_sha = sha256(root / LEDGER)
    if summary['artifacts'][LEDGER] != ledger_sha:
        raise ValueError('Intake ledger differs from its sealed summary')
    ledger = read_jsonl(root / LEDGER)
    rows = {r['item_id']: r for r in read_jsonl(root / SNAPSHOT_ROWS)}
    forbidden, pointer_sha = _protected(root)

    def item(entry):
        row = rows[entry['item_id']]
        path = root / row['image_path']
        if sha256(path) != entry['image_sha256']:
            raise ValueError('Source bytes changed: ' + entry['item_id'])
        if entry['image_sha256'] in forbidden or row['protected_sha_now']:
            raise ValueError('Protected image in scope: ' + entry['item_id'])
        return {'item_id': entry['item_id'], 'image_sha256': entry['image_sha256'], 'image_path': row['image_path'],
                'confirmed_slug': row['confirmed_slug'], 'product_id': entry['product_id'],
                'target_scope': row['target_scope'], 'target_description': row['target_description'],
                'identity_qa_grade': row['identity_qa_grade'], 'split_role_at_source': row['split_role_at_source'],
                'prior_source_use_statuses': row['prior_source_use_statuses'], 'use_flags': row['use_flags'],
                'intake_outcome': entry['outcome'], 'intake_receipt': entry['receipt'],
                'intake_receipt_sha256': sha256(root / entry['receipt']), 'holds': entry['holds']}

    misses = [item(e) for e in ledger if e['outcome'] == MISS]
    if len(misses) != summary['outcomes'][MISS]:
        raise ValueError('Miss count differs from summary')
    # Deterministic ordinary controls: first N by SHA per target scope, receipts reused for parity.
    controls = []
    for scope in ('label_closeup', 'single_bottle'):
        pool = sorted((e for e in ledger if e['outcome'] in CONTROL_OUTCOMES and e.get('receipt')
                       and rows[e['item_id']]['target_scope'] == scope), key=lambda e: e['image_sha256'])
        controls += [item(e) for e in pool[:CONTROLS_PER_SCOPE]]
    negatives = []
    frames = verify(read_json(root / NEGATIVES))['frames']
    for frame in frames:
        if frame['split'] != 'selection' or frame['instances']:
            continue
        name = Path(frame['image_path']).name
        found = [p for p in [root / frame['image_path'], *(root / 'data/source-media').glob(frame['image_sha256'] + '.*'),
                             root / STRAPI / name] if p.is_file() and sha256(p) == frame['image_sha256']]
        if not found:
            raise ValueError('Negative bytes unavailable: ' + frame['image_sha256'])
        if frame['image_sha256'] in forbidden:
            raise ValueError('Protected negative')
        negatives.append({'image_sha256': frame['image_sha256'], 'image_path': str(found[0].relative_to(root)),
                          'admitted_image_path': frame['image_path'],
                          'negative_review_status': frame['negative_review_status']})
    doc = seal({'kind': 'target-misses-v1-scope', 'fit_admitted': False,
                'semantics': 'confirmed_slug is a source-product binding, not physical target GT',
                'inputs_sha256': {LEDGER: ledger_sha, SUMMARY: sha256(root / SUMMARY),
                                  SNAPSHOT_ROWS: sha256(root / SNAPSHOT_ROWS), NEGATIVES: sha256(root / NEGATIVES),
                                  PROFILE: sha256(root / PROFILE)},
                'protection_pointer_sha256': pointer_sha,
                'misses': misses, 'ordinary_controls': controls, 'nonwine_controls': negatives})
    write_json(root / OUT / 'scope.json', doc)
    return doc


def load(root):
    root = Path(root).resolve()
    doc = verify(read_json(root / OUT / 'scope.json'))
    for path, digest_ in doc['inputs_sha256'].items():
        if path != LEDGER and sha256(root / path) != digest_:
            raise ValueError('Scope input changed: ' + path)
    for entry in doc['misses'] + doc['ordinary_controls'] + doc['nonwine_controls']:
        if sha256(root / entry['image_path']) != entry['image_sha256']:
            raise ValueError('Scope image changed: ' + entry['image_path'])
    return doc
