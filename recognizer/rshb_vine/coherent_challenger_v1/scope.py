"""Finite image-level evaluation scope over the allowed open cohorts, sealed before any model run.

Cohorts (first listed wins on duplicate SHA): operational development, operational web232 train,
ranker-completion real_extra_v2, runtime-training-intake-v1 usable rows, verified nonwine negatives.
Eval guard: the fit-role audit plus the protection closure; only open development/eval-only roles
pass. web232 validation/test/hold, reserved whole71/new52/open-validation, kultovo reserved roles,
closed/consumed/locked protection states, their derivatives and shared groups are excluded.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from rshb_vine.catalog.product_registry import ProductRegistry
from rshb_vine.combined_ranker_v1.identity import FROZEN_BUNDLE
from rshb_vine.data_protection import load_protection
from rshb_vine.io import read_json, seal, sha256, write_json
from rshb_vine.training_admission import audit_training_rows

OUT = 'runs/coherent-challenger-v1'
CURRENT_PROFILE = 'bd939433388ebfb5723f92c90a1a4b1d11ff29569abc5fdde389570c74c888be'
OPERATIONAL = 'data/unified-corpus-v1/operational-v1/'
INTAKE_LEDGER = 'runs/runtime-training-intake-v1/ledger.jsonl'
EXTRA_ROWS = 'runs/ranker-completion-v2/admitted-extra-rows.jsonl'
EXTRA_MANIFESTS = ('runs/ranker-completion-v2/capture-manifest.json', 'runs/corpus-completion-v1/real7-capture-manifest.json',
                   'runs/ranker-integration-v3/internet/eligibility.json')
DEV108_ROSTER = 'runs/catalog-additions-alpha-repair-v1/full-pipeline-diagnostic-v3/roster.json'
DEV108_SAVED = 'runs/catalog-additions-alpha-repair-v1/full-pipeline-diagnostic-v3/paired-v1/current_saved'
NEGATIVES = 'data/bottle-instance-v1/admitted-v2.json'
RECOVERED_IMAGES = 'data/evaluation-source-recovery-v1/images'
TARGET_MISSES_SCOPE = 'runs/target-misses-v1/scope.json'
RECEIPT_DIRS = ('runs/runtime-training-intake-v1/receipts', 'runs/ranker-integration-v3/internet/capture-runtime',
                'runs/ranker-completion-v2/capture-runtime', 'runs/corpus-completion-v1/real7-runtime')
USABLE_INTAKE = ('rankable_draft_eligible', 'held_rankable_row', 'no_runtime_target_e2e_miss',
                 'held_multibottle_no_source_bbox', 'retrieval_miss_e2e_only')
OPEN_PROTECTION = ('open_evaluation_only_no_fit', 'open_synthetic_evaluation_only_no_fit')
EVAL_OK_REASONS = ('row_not_explicit_train', 'ranker:development_hold', 'internet:open_evaluation_only_no_fit',
                   'group:ranker:development_hold', 'group:internet:open_evaluation_only_no_fit')
INTAKE_EXCLUDING_HOLDS = ('reserved_lineage_component:protected', 'prior_duplicate_or_derivative_status:exclude_previous_exact',
                          'prior_duplicate_or_derivative_status:exclude_derivative',
                          'prior_duplicate_or_derivative_status:derived_duplicate',
                          'prior_duplicate_or_derivative_status:duplicate_reference')


def _jsonl(root, path):
    return [json.loads(line) for line in (root / path).read_text().splitlines() if line.strip()]


def receipt_index(root):
    """sha -> saved full current-profile result (path), only receipts produced by the pinned current profile."""
    index = {}
    for folder in RECEIPT_DIRS:
        for path in sorted((root / folder).glob('*.json')):
            if path.name.endswith('.attempt.json'):
                continue
            doc = json.loads(path.read_text())
            result = doc.get('result') or {}
            if (doc.get('profile_checksum') != CURRENT_PROFILE or
                    (result.get('recognition_runtime') or {}).get('profile_checksum') != CURRENT_PROFILE):
                continue
            sha = doc.get('request_sha256') or doc.get('source_sha256')
            index.setdefault(sha, str(path.relative_to(root)))
    for path in sorted((root / DEV108_SAVED).glob('*.json')):
        doc = json.loads(path.read_text())
        if (doc.get('result') or {}).get('recognition_runtime', {}).get('profile_checksum') == CURRENT_PROFILE \
                and doc.get('request_roi') is None:
            index.setdefault(doc['source_sha256'], str(path.relative_to(root)))
    return index


def _closed_shas(protection):
    states = {}
    for row in protection['memberships']:
        states.setdefault(row['sha256'], set()).add(row['protected_state'])
    closed = {s for s, st in states.items() if st - set(OPEN_PROTECTION)}
    changed = True
    while changed:
        changed = False
        for row in protection['derivatives']:
            if row['parent_sha256'] in closed and row['sha256'] not in closed:
                closed.add(row['sha256'])
                changed = True
    return closed, states


def build(root):
    root = Path(root).resolve()
    frozen = ProductRegistry.from_bundle(root, FROZEN_BUNDLE)
    queries, duplicates, excluded = {}, [], Counter()

    def add(qid, sha, path, cohort, condition, slugs, status, target_scope, extra=None):
        if sha in queries:
            duplicates.append({'query_id': qid, 'kept': queries[sha]['query_id']})
            return
        unknown = [s for s in slugs if s not in frozen.cards]
        queries[sha] = {'query_id': qid, 'image_sha256': sha, 'image_path': path, 'cohort': cohort,
                        'condition': condition, 'target_scope': target_scope,
                        'ground_truth': {'acceptable_slugs': sorted(slugs), 'status': status,
                                         'frozen_products': sorted({frozen.product_id(s) for s in slugs if s in frozen.cards}),
                                         'unknown_slugs': unknown}, **(extra or {})}

    for role, cohort in (('development', 'base_development'), ('train', 'base_web232_train')):
        doc = read_json(root / OPERATIONAL / f'{role}.json')
        by_sha = {}
        for r in doc['records']:
            by_sha.setdefault(r['sha256'], []).append(r)
        for sha, rows in by_sha.items():
            path = rows[0]['image_path']
            path = str(Path(path).relative_to(root)) if Path(path).is_absolute() else path
            if not (root / path).is_file():
                path = next((str(p.relative_to(root)) for p in (root / RECOVERED_IMAGES).glob(sha + '.*')), path)
            slugs = sorted({s for r in rows for s in r['acceptable_slugs']})
            add(rows[0]['id'], sha, path, cohort, rows[0]['source'] + ':' + rows[0]['cohort'], slugs,
                'operational_' + role, 'single' if len(slugs) == 1 else 'multi_product',
                {'source_record_ids': [r['id'] for r in rows]})
    paths = {}
    for manifest in EXTRA_MANIFESTS:
        for r in read_json(root / manifest)['records']:
            sha = r.get('sha256') or r.get('image_sha256') or r.get('source_sha256')
            if sha and r.get('image_path'):
                paths.setdefault(sha, r['image_path'])
    for r in _jsonl(root, EXTRA_ROWS):
        if r['image_sha256'] not in paths:
            excluded['real_extra_no_image_path'] += 1
            continue
        add(r['query_id'], r['image_sha256'], paths[r['image_sha256']], 'real_extra_v2', r.get('cohort') or r['source'],
            r['ground_truth']['acceptable_slugs'], r['ground_truth']['status'], 'single')
    miss_shas = {c['image_sha256'] for c in read_json(root / TARGET_MISSES_SCOPE)['misses']}
    for r in _jsonl(root, INTAKE_LEDGER):
        if r['outcome'] not in USABLE_INTAKE:
            excluded['intake:' + r['outcome']] += 1
            continue
        if set(r['holds']) & set(INTAKE_EXCLUDING_HOLDS):
            excluded['intake:lineage_or_duplicate_hold'] += 1
            continue
        receipt = json.loads((root / r['receipt']).read_text()) if r.get('receipt') else {}
        path = receipt.get('image_path') or paths.get(r['image_sha256'])
        if not path:
            excluded['intake_no_image_path'] += 1
            continue
        scope = 'multibottle_unlocated' if r['target_policy'] == 'multibottle_no_source_bbox' else 'single'
        flags = sorted({h.split(':')[0] for h in r['holds']})
        add(r['item_id'], r['image_sha256'], path, 'runtime_intake_v1', r['cohort'], [r['slug']],
            'intake:' + r['identity_qa_grade'], scope,
            {'intake_outcome': r['outcome'], 'intake_holds': r['holds'], 'hold_families': flags,
             'owner_label_only': any(h.startswith('owner_collection_exclusion') for h in r['holds']),
             'route_development_exposed': r['image_sha256'] in miss_shas})
    negatives = [f for f in read_json(root / NEGATIVES)['frames']
                 if f.get('source_kind') == 'negative_original' and f.get('negative_review_status') == 'visually_verified']
    for f in negatives:
        add('nonwine:' + f['image_sha256'][:16], f['image_sha256'], f'data/source-media/{f["image_sha256"]}.webp',
            'nonwine_verified', 'negative_split:' + f['split'], [], 'visually_verified_no_wine', 'nonwine')

    protection = load_protection(root)
    closed, states = _closed_shas(protection)
    audit = audit_training_rows([{'query_id': q['query_id'], 'image_sha256': sha, 'ranker_split': 'train'}
                                 for sha, q in queries.items()], root)
    reasons = {d['image_sha256']: d['reasons'] for d in audit['decisions']}
    receipts = receipt_index(root)
    kept, denied = [], []
    for sha, q in queries.items():
        why = [r for r in reasons[sha] if r not in EVAL_OK_REASONS and r != 'protection-current']
        if sha in closed:
            why.append('protection:' + ','.join(sorted(states.get(sha, {'derivative_of_closed'}))))
        if not (root / q['image_path']).is_file() or sha256(root / q['image_path']) != sha:
            why.append('image_bytes_unavailable_or_changed')
        if why:
            denied.append({'query_id': q['query_id'], 'image_sha256': sha, 'cohort': q['cohort'], 'reasons': sorted(why)})
            continue
        q['eval_guard_reasons_allowed'] = [r for r in reasons[sha] if r != 'row_not_explicit_train']
        q['protection_state'] = sorted(states.get(sha, []))
        q['current_receipt'] = receipts.get(sha)
        kept.append(q)
    kept.sort(key=lambda q: (q['cohort'], q['query_id']))
    counts = Counter(q['cohort'] for q in kept)
    doc = seal({'kind': 'coherent-challenger-v1-scope', 'queries': kept, 'denied': denied,
                'duplicates': duplicates, 'excluded_before_guard': dict(excluded),
                'counts': {'kept': dict(counts), 'kept_total': len(kept), 'denied': dict(Counter(d['cohort'] for d in denied)),
                           'with_saved_current_receipt': sum(bool(q['current_receipt']) for q in kept),
                           'needs_current_capture': sum(not q['current_receipt'] for q in kept)},
                'guard': {'eval_ok_reasons': list(EVAL_OK_REASONS), 'open_protection_states': list(OPEN_PROTECTION),
                          'protection_pointer_sha256': protection['pointer_sha256'],
                          'role_audit_checksum': audit['checksum'], 'role_audit_sources': audit['sources_sha256']},
                'frozen_registry': frozen.checksum, 'current_profile': CURRENT_PROFILE,
                'request_policy': 'no supplied ROI for every query (common replay policy)',
                'sources_sha256': {p: sha256(root / p) for p in (OPERATIONAL + 'development.json', OPERATIONAL + 'train.json',
                                   INTAKE_LEDGER, EXTRA_ROWS, NEGATIVES, TARGET_MISSES_SCOPE, *EXTRA_MANIFESTS)},
                'labels_or_outcomes_used_for_scope': False})
    write_json(root / OUT / 'scope.json', doc)
    return doc
