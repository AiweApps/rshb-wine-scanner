"""Combined roster: frozen v4 base (477) + intake draft-eligible (166) + audited text-only positives.

Roles are inherited, never re-drawn: v4 fit/internal_selection and the intake split draft
(train_draft→fit, internal_validation_draft→internal_selection). Text-miss additions are
grouped with every existing component before any metric exists; a component touching one
role joins it, a fresh one is placed by a salted hash, a bridging one or any development108
key is held. Additions enter only with an ``admit_as_text_positive`` evidence verdict and no
ledger hold. Nothing here grants fit admission; the fit guard calls the role preflight again.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

from rshb_vine.io import read_json, read_jsonl, seal, sha256, verify
from rshb_vine.ranker_integration_v3.split import _Union, row_keys
from rshb_vine.system_selection_v4 import roster as v4_roster

V4_ROSTER = 'runs/system-selection-v4/roster.json'
INTAKE_HANDOFF = 'runs/runtime-training-intake-v1/handoff.json'
INTAKE_ROWS = 'data/runtime-training-intake-v1/rows.jsonl'
INTAKE_TABLE = 'data/runtime-training-intake-v1/v4-table.jsonl'
INTAKE_SPLIT = 'runs/runtime-training-intake-v1/split-draft.json'
INTAKE_LEDGER = 'runs/runtime-training-intake-v1/ledger.jsonl'
ADMITTED_TABLE = 'runs/ranker-dataset-v1-root-admission01/admitted-table.jsonl'
ROLE_PROBE = 'runs/root-review-claude-20260923/text-positive-role-probe.json'
EVIDENCE = 'runs/combined-ranker-v1/evidence/text-positive-audit.json'
CONDITIONAL_LINKS = 'data/identity-link-review-v1/conditional-rows.json'
REGISTRY_BUNDLE = 'config/product-identity-current.json'
INTAKE_SNAPSHOT = 'data/runtime-training-intake-v1/input-snapshot.jsonl'
INTAKE_PREFLIGHT = 'runs/runtime-training-intake-v1/preflight.json'
PHOTO_DECISIONS = ('data/photo-product-completion-claude-v2/source-decisions.jsonl',
                   'data/photo-product-completion-claude-v2/target-decisions.jsonl')
REFERENCE_DERIVATIVE_BITS = 3
PAGE_LEVEL = '/browser/by-slug/'
IDENTITY_PRIOR_STATUSES = {'hold_variant', 'reject_wrong_product', 'reject_previous'}
IDENTITY_RAW_CONFLICT = {'conflict', 'proposed_conflict'}
SOL_CONTRADICTION = {'existing_derivative_not_new_query', 'foreign_out_of_catalog', 'terminal_source_excluded',
                     'unresolved_variant_or_identity_conflict', 'unresolved_variant_or_mismatch',
                     'unresolved_variant_or_catalog', 'conditional_candidate_unresolved'}
IDENTITY_FEATURES = ('producer', 'name_or_line', 'color_or_style', 'sugar', 'grape_or_blend')
SALT = 'combined-ranker-v1/text-positive-placement'
SELECTION_SHARE = 0.2
TEXT_INTAKE = ('rti1-7f21250397a303a4/target/closeup-0', 'rti1-db1bf4369d82e732/target/0',
               'rti1-ecf13e2934ffd8b8/target/closeup-0')
TEXT_HISTORICAL = ('historical865/web232:W0185/target/closeup-0',)
ROOT_HELD = {'rti1-2fa388aff7b78269/target/1': 'root: hold_possible_derivative + development108_producer_family'}
INTAKE_ROLE = {'train_draft': 'fit', 'internal_validation_draft': 'internal_selection'}
SOURCES = ('runs/identity-link-review-v1/candidate-bundle-handoff.json',
           'data/product-identity-v1/snapshot-03-candidate/manifest.json', V4_ROSTER, INTAKE_HANDOFF, INTAKE_ROWS, INTAKE_TABLE, INTAKE_SPLIT, INTAKE_LEDGER, ADMITTED_TABLE,
           ROLE_PROBE, EVIDENCE, CONDITIONAL_LINKS, REGISTRY_BUNDLE, INTAKE_SNAPSHOT, INTAKE_PREFLIGHT, *PHOTO_DECISIONS, 'rshb_vine/combined_ranker_v1/roster.py',
           'rshb_vine/training_admission.py', 'rshb_vine/ranker_integration_v3/split.py')


def _hash_share(text):
    return int(hashlib.sha256((SALT + ':' + text).encode()).hexdigest()[:8], 16) / 2 ** 32


def hidden_holds(root, rows):
    """Pre-registered holds the intake ledger did not carry; same rule for every intake row, no metric input.

    Identity: a prior variant/wrong-product status, a raw source-identity conflict, an earlier independent
    (Sol) verdict that contradicts the binding, a final review feature mismatch, or a redesign inference
    below high confidence. Derivative: dHash <= 3 bits to a gallery reference of the same product.
    """
    from rshb_vine.catalog.product_registry import ProductRegistry
    registry = ProductRegistry.from_bundle(root, REGISTRY_BUNDLE)
    snapshot = {r['item_id']: r for r in read_jsonl(root / INTAKE_SNAPSHOT)}
    decisions = {}
    for path in PHOTO_DECISIONS:
        for r in read_jsonl(root / path):
            decisions.setdefault(r['item_id'], r)
    near = defaultdict(list)
    for p in verify(read_json(root / INTAKE_PREFLIGHT))['near_pairs']:
        if p['b_kind'] == 'gallery' and p['bits'] <= REFERENCE_DERIVATIVE_BITS:
            near[p['a']].append(p)
    output = {}
    for row, item_id in rows:
        snap, decision = snapshot[item_id], decisions.get(item_id, {})
        review = decision.get('claude_review') or {}
        confidence = next((x.get('confidence') for x in (decision.get('claude_adjudication'), decision.get('claude_qa'),
                                                          review) if isinstance(x, dict) and x.get('confidence')), None)
        sol = decision.get('sol') or {}
        reasons = [f'prior_identity_status:{x}' for x in sorted(set(snap.get('prior_source_use_statuses') or [])
                                                               & IDENTITY_PRIOR_STATUSES)]
        if snap.get('source_identity_status_raw') in IDENTITY_RAW_CONFLICT:
            reasons.append('source_identity_raw:' + snap['source_identity_status_raw'])
        if isinstance(sol, dict) and sol.get('final_status') in SOL_CONTRADICTION:
            reasons.append('earlier_review_contradiction:' + sol['final_status'])
        mismatches = [f for f in IDENTITY_FEATURES if (review.get('features') or {}).get(f) == 'mismatch']
        if mismatches:
            reasons.append('review_feature_mismatch:' + ','.join(mismatches))
        if snap.get('label_design') == 'redesign_same_product' and confidence != 'high':
            reasons.append('redesign_inference_confidence:' + str(confidence))
        positives = set(row['ground_truth']['current_products'])
        for p in near.get(row['image_sha256'], []):
            if positives & {registry.product_id(s) for s in p.get('b_slugs') or [] if s in registry.cards}:
                reasons.append(f'gallery_reference_derivative:{p["b"][:16]}:{p["bits"]}bits')
                break
        output[row['query_id']] = {'reasons': reasons, 'item_id': item_id,
                                   'prior_source_use_statuses': snap.get('prior_source_use_statuses') or [],
                                   'source_identity_status_raw': snap.get('source_identity_status_raw'),
                                   'sol_final_status': sol.get('final_status') if isinstance(sol, dict) else None,
                                   'review_confidence': confidence, 'label_design': snap.get('label_design'),
                                   'source_page_group': ('page:' + str(Path(snap['image_path']).parent)
                                                         if PAGE_LEVEL in snap['image_path'] else None)}
    return output


def _intake(root):
    handoff = verify(read_json(root / INTAKE_HANDOFF))
    for path in (INTAKE_ROWS, INTAKE_TABLE, INTAKE_SPLIT, INTAKE_LEDGER):
        if sha256(root / path) != handoff['artifacts'].get(path, sha256(root / path)):
            raise ValueError('Intake artefact changed since its handoff: ' + path)
    rows = {r['query_id']: r for r in read_jsonl(root / INTAKE_ROWS)}
    table = {r['query_id']: r for r in read_jsonl(root / INTAKE_TABLE)}
    split = verify(read_json(root / INTAKE_SPLIT))['assignment']
    ledger = {r['query_id']: r for r in read_jsonl(root / INTAKE_LEDGER) if r.get('query_id')}
    return rows, table, split, ledger


def load_candidates(root):
    """(base, intake, additions, held) as full rows with provenance fields; roles only from frozen sources."""
    root = Path(root)
    frozen = verify(read_json(root / V4_ROSTER))
    if v4_roster.build_roster(root)['checksum'] != frozen['checksum']:
        raise ValueError('v4 roster inputs changed since freeze')
    base = [dict(r, combined_role=r['v4_split'], combined_source='v4_roster', identity_qa_grade='base_admitted',
                 role_basis='frozen v4 roster ' + frozen['checksum'][:12])
            for r in v4_roster.load_rows(root, frozen)]
    rows, table, split, ledger = _intake(root)
    intake, held = [], []
    eligible = [qid for qid, record in sorted(table.items()) if record['intake_outcome'] == 'rankable_draft_eligible']
    audit = hidden_holds(root, [(rows[q], ledger[q]['item_id']) for q in eligible + list(TEXT_INTAKE)])
    for qid in eligible:
        record, found = table[qid], audit[qid]
        extra = {'source_group_ids': sorted(set(rows[qid].get('source_group_ids') or [])
                                           | ({found['source_page_group']} - {None})),
                 'hidden_hold_audit': found}
        if found['reasons']:
            held.append({'query_id': qid, 'image_sha256': rows[qid]['image_sha256'], 'hold_reasons': found['reasons'],
                         'combined_source': 'runtime_training_intake_v1', 'intake_draft_role': split[qid]['split'],
                         'identity_qa_grade': record['identity_qa_grade']})
            continue
        role = INTAKE_ROLE[split[qid]['split']]
        intake.append(dict(rows[qid], **extra, ranker_split='train' if role == 'fit' else 'internal_selection',
                           combined_role=role, combined_source='runtime_training_intake_v1',
                           component=None, identity_qa_grade=record['identity_qa_grade'],
                           role_basis='intake split draft: ' + split[qid]['reason']))
    verdicts = {r['query_id']: r for r in read_json(root / EVIDENCE)['rows']}
    admitted = {r['query_id']: r for r in read_jsonl(root / ADMITTED_TABLE)}
    additions = []
    for qid in TEXT_INTAKE + TEXT_HISTORICAL:
        if qid in TEXT_INTAKE:
            found = audit[qid]
            row = dict(rows[qid], source_group_ids=sorted(set(rows[qid].get('source_group_ids') or [])
                                                          | ({found['source_page_group']} - {None})), hidden_hold_audit=found)
            entry = ledger[qid]
            holds = list(entry['holds']) + list(entry['blocks'] or []) + found['reasons']
            grade, source = entry['identity_qa_grade'], 'runtime_training_intake_v1_text_miss'
        else:
            row = admitted[qid]
            holds = [] if row['ranker_split'] == 'train_retrieval_miss_e2e_only' and row.get('A_source_role') == 'train' \
                else ['historical_role_not_train:' + str(row['ranker_split'])]
            grade, source = 'historical_admitted_table', 'admitted_table_train_retrieval_miss'
        verdict = verdicts.get(qid, {}).get('verdict')
        if verdict != 'admit_as_text_positive':
            holds.append('evidence_verdict:' + str(verdict))
        item = dict(row, combined_source=source, component=None, identity_qa_grade=grade,
                    text_positive_candidate=True, evidence_verdict=verdict)
        if holds:
            held.append({'query_id': qid, 'image_sha256': row['image_sha256'], 'hold_reasons': holds,
                         'combined_source': source})
        else:
            additions.append(item)
    for qid, reason in ROOT_HELD.items():
        held.append({'query_id': qid, 'image_sha256': rows[qid]['image_sha256'], 'hold_reasons': [reason],
                     'combined_source': 'runtime_training_intake_v1_text_miss'})
    return frozen, base, intake, additions, held


def place(base, intake, additions, dev):
    """Group every row; additions inherit a touching role or a salted-hash role; returns {qid: (role, reason)}."""
    union = _Union()
    for r in base + intake + additions:
        for k in row_keys(r):
            union.join('q:' + r['query_id'], k)
    forbidden = {k for r in dev for k in row_keys(r) if not k.startswith('producer:')}
    roles = defaultdict(set)
    for r in base + intake:
        roles[union.find('q:' + r['query_id'])].add(r['combined_role'])
    bridging = sorted(c for c, s in roles.items() if len(s) > 1)
    placement = {}
    for r in additions:
        comp = union.find('q:' + r['query_id'])
        if forbidden & set(row_keys(r)):
            placement[r['query_id']] = ('held', 'shares_development108_key')
        elif len(roles.get(comp, ())) > 1:
            placement[r['query_id']] = ('held', 'bridges_fit_and_internal_selection')
        elif roles.get(comp):
            role = next(iter(roles[comp]))
            placement[r['query_id']] = (role, 'joins_existing_' + role + '_component')
        else:
            members = sorted(x['query_id'] for x in additions if union.find('q:' + x['query_id']) == comp)
            role = 'internal_selection' if _hash_share(members[0]) < SELECTION_SHARE else 'fit'
            placement[r['query_id']] = (role, 'fresh_component_salted_hash')
    components = {r['query_id']: union.find('q:' + r['query_id']) for r in base + intake + additions}
    return placement, components, bridging


def build(root):
    from rshb_vine.training_admission import audit_training_rows
    root = Path(root)
    frozen, base, intake, additions, held = load_candidates(root)
    dev = v4_roster.load_rows(root, frozen, 'development108')
    placement, components, bridging = place(base, intake, additions, dev)
    placed = []
    for r in additions:
        role, reason = placement[r['query_id']]
        if role == 'held':
            held.append({'query_id': r['query_id'], 'image_sha256': r['image_sha256'], 'hold_reasons': [reason],
                         'combined_source': r['combined_source']})
            continue
        placed.append(dict(r, ranker_split='train' if role == 'fit' else 'internal_selection', combined_role=role,
                           role_basis='combined placement: ' + reason))
    rows = base + intake + placed
    identity_audit, rows, held = candidate_identity_audit(root, rows, dev, held)
    keys = Counter((r['image_sha256'], str(r['instance_id'])) for r in rows)
    duplicates = sorted(k for k, n in keys.items() if n > 1)
    sha_counts = Counter(r['image_sha256'] for r in rows)
    if duplicates or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate SHA/target in combined roster')
    conditional = {r['source_sha256'] for r in read_json(root / CONDITIONAL_LINKS)['rows']}
    if conditional & {r['image_sha256'] for r in rows}:
        raise ValueError('Combined roster contains a conditional identity-link row')
    comp_roles = defaultdict(set)
    for r in rows:
        comp_roles[components[r['query_id']]].add(r['combined_role'])
    crossing = sorted(c for c, s in comp_roles.items() if len(s) > 1)
    forbidden = {k for r in dev for k in row_keys(r) if not k.startswith('producer:')}
    touching_dev = sorted(r['query_id'] for r in rows if forbidden & set(row_keys(r)))
    fit_rows = [r for r in rows if r['combined_role'] == 'fit']
    selection_rows = [r for r in rows if r['combined_role'] == 'internal_selection']
    fit_audit = audit_training_rows(fit_rows, root)
    selection_audit = audit_training_rows([dict(r, ranker_split='train') for r in selection_rows], root)
    records = [{'query_id': r['query_id'], 'image_sha256': r['image_sha256'], 'instance_id': str(r['instance_id']),
                'role': r['combined_role'], 'source': r['combined_source'], 'role_basis': r['role_basis'],
                'identity_qa_grade': r['identity_qa_grade'], 'component': components[r['query_id']],
                'receipt_path': r['receipt_path'], 'receipt_sha256': r.get('receipt_sha256'),
                'ground_truth_products': sorted(r['ground_truth']['current_products']),
                'ground_truth_slugs': sorted(r['ground_truth']['acceptable_slugs']),
                'candidate_positive_products': r['candidate_positive_products'],
                'retrieval_miss': bool(r.get('retrieval_miss')),
                'text_positive_candidate': bool(r.get('text_positive_candidate')),
                'source_provenance': {k: v for k, v in (r.get('hidden_hold_audit') or {}).items() if k != 'reasons'} or None,
                'loss_ignore_candidate_ids': sorted(r.get('loss_ignore_candidate_ids', []))}
               for r in sorted(rows, key=lambda r: r['query_id'])]
    by_role = lambda items: dict(Counter(r['combined_role'] for r in items))
    return seal({
        'kind': 'combined-ranker-v1-roster', 'v4_roster_checksum': frozen['checksum'],
        'counts': {'base': by_role(base), 'intake': by_role(intake), 'text_positive_additions': by_role(placed),
                   'total': by_role(rows),
                   'by_grade': {role: dict(Counter(r['identity_qa_grade'] for r in rows if r['combined_role'] == role))
                                for role in ('fit', 'internal_selection')}},
        'placement': {q: {'role': p[0], 'reason': p[1]} for q, p in sorted(placement.items())},
        'held': held,
        'held_counts': dict(Counter(reason.split(':')[0] for h in held for reason in h['hold_reasons'])),
        'held_rows_by_intake_role': dict(Counter(h.get('intake_draft_role', 'text_or_root') for h in held)),
        'checks': {'duplicate_sha_target': duplicates, 'images_with_multiple_rows': sum(n > 1 for n in sha_counts.values()),
                   'conditional_identity_rows': 0, 'components': len(comp_roles),
                   'components_crossing_roles': crossing, 'components_bridging_before_additions': bridging,
                   'rows_sharing_development108_key': touching_dev,
                   'fit_role_audit': {'checksum': fit_audit['checksum'], 'rejected': fit_audit['rejected'],
                                      'rejections': [d for d in fit_audit['decisions'] if not d['allowed_by_role']]},
                   'internal_selection_role_audit': {'checksum': selection_audit['checksum'],
                                                     'rejected': selection_audit['rejected'],
                                                     'rejections': [d for d in selection_audit['decisions']
                                                                    if not d['allowed_by_role']]},
                   'unlabelled_targets': 'only the labelled instance of a frame is a row; other targets are never rows'},
        'excluded_by_design': ['development108 (regression reference only)', 'intake held_rankable_row 101',
                               'intake multibottle/no-target/owner-excluded/blocked', '43 v3 rows pending root',
                               'photo-v3 and later source bindings (no runtime admission)',
                               '47 conditional identity-link rows', 'protected/locked/consumed'],
        'candidate_identity_audit': identity_audit,
        'salt': SALT, 'selection_share_for_fresh_components': SELECTION_SHARE,
        'sources_sha256': {p: sha256(root / p) for p in SOURCES},
        'rows': records, 'label_or_model_outcome_used_for_assignment': False,
        'identity_or_fit_admission_granted': False})


def candidate_identity_audit(root, rows, dev, held):
    """Split impact of the challenger identity: label migration and GT products shared across roles.

    Rows whose labels cannot migrate are held. A candidate product that is GT in both fit and internal
    selection holds the internal-selection rows (fit keeps its evidence); one shared with development108
    is reported, since development108 is a historical regression reference only.
    """
    from rshb_vine.catalog.product_registry import ProductRegistry
    from rshb_vine.combined_ranker_v1.identity import candidate_labels, load_candidate
    handoff, candidate = load_candidate(root)
    frozen = ProductRegistry.from_bundle(root, REGISTRY_BUNDLE)
    labels = {r['query_id']: candidate_labels(r, frozen, candidate) for r in rows}
    status = Counter(v[1] for v in labels.values())
    kept, held = [], list(held)
    for r in rows:
        products, state = labels[r['query_id']]
        if state in ('split_requires_label_review', 'acceptable_slug_outside_migrated_product'):
            held.append({'query_id': r['query_id'], 'image_sha256': r['image_sha256'],
                         'hold_reasons': ['candidate_identity:' + state], 'combined_source': r['combined_source']})
        else:
            kept.append(r)
    fit_products = {p for r in kept if r['combined_role'] == 'fit' for p in labels[r['query_id']][0]}
    crossing = [r for r in kept if r['combined_role'] == 'internal_selection'
                and fit_products & set(labels[r['query_id']][0])]
    for r in crossing:
        held.append({'query_id': r['query_id'], 'image_sha256': r['image_sha256'],
                     'hold_reasons': ['candidate_identity:gt_product_also_in_fit'], 'combined_source': r['combined_source'],
                     'intake_draft_role': 'internal_selection'})
    kept = [r for r in kept if r not in crossing]
    dev_products = set()
    for r in dev:
        dev_products.update(labels.get(r['query_id'], candidate_labels(r, frozen, candidate))[0])
    shared_dev = sorted(r['query_id'] for r in kept if dev_products & set(labels[r['query_id']][0]))
    changed = sorted(r['query_id'] for r in kept if labels[r['query_id']][1] == 'migrated_changed_id')
    audit = {'handoff_checksum': handoff['checksum'], 'candidate_registry_checksum': candidate.checksum,
             'candidate_bundle_checksum': candidate.bundle_checksum, 'frozen_registry_checksum': frozen.checksum,
             'label_status': dict(status), 'rows_with_changed_product_id': changed,
             'held_label_migration': sorted(q for q, (_, st) in labels.items()
                                            if st in ('split_requires_label_review', 'acceptable_slug_outside_migrated_product')),
             'held_internal_selection_sharing_fit_product': sorted(r['query_id'] for r in crossing),
             'rows_sharing_development108_product': shared_dev}
    for r in kept:
        r['candidate_positive_products'] = labels[r['query_id']][0]
    return audit, kept, held


def load_rows(root, roster, role):
    """Full rows of one role, re-checked against the sealed combined roster."""
    root = Path(root)
    _, base, intake, additions, _ = load_candidates(root)
    by_id = {r['query_id']: r for r in base + intake + additions}
    output = []
    for record in roster['rows']:
        if record['role'] != role:
            continue
        row = by_id[record['query_id']]
        if row['image_sha256'] != record['image_sha256'] or row['receipt_path'] != record['receipt_path']:
            raise ValueError('Combined roster row changed: ' + record['query_id'])
        output.append(dict(row, ranker_split='train' if role == 'fit' else 'internal_selection',
                           combined_role=role, component=record['component'],
                           identity_qa_grade=record['identity_qa_grade']))
    return output
