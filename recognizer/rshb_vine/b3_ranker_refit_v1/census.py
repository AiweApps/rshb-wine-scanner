"""Read-only census for a B3-only ranker refit: roster exposure, cross-pool membership and saved-response B3 ranks.

No models, no HTTP, no selector replay. Validation/test/protected/closed rosters are read for SHA/group membership
only; their prediction or label fields are never read. Open ranks come from saved open-http-gate responses.
"""
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rshb_vine.io import read_json, sha256

FIT471 = 'runs/systemic-ranking-v2/tables/fit.jsonl'
QUARANTINE = 'runs/systemic-ranking-v2/fit-cohort-intersections.json'
GATE = 'runs/recognition-repair-20260926/open-http-gate'
OPEN_ROWS = 'runs/systemic-release-v3/selector/replay-v1/rows.jsonl'
LEDGER = 'runs/release-last-mile-v1/regressions/root-reviewed/ledger.jsonl'
UNIFIED = 'data/unified-corpus-v1/ledger.json'
RTI1_SNAPSHOT = 'data/runtime-training-intake-v1/input-snapshot.jsonl'
SLUG1000 = 'data/internet-slug1000-20260922'
B3_PARENT = 'data/domain-training-v1/B3-admission-v1/manifest.json'
FINGERPRINTS = 'runs/runtime-training-intake-v1/fingerprints.json'
DHASH_NEAR_BITS = 6
EXPOSED_GROUPS = 'b3_stage5_exposed_fit471_groups'
STAGE5_STEPS = 'data/catalog-training-data-v2/stage5/steps.jsonl'
WEB232 = 'data/evaluation/web232-v1/'
CLOSED = ('data/internet-evaluation-intake-20260921/locked-final/manifest.json',
          'data/internet-evaluation-intake-20260921/extension-v2/locked-final/manifest.json')
SLICES = {'product_exact_web232': 'exact_train', 'product_admitted': 'real_confirmed',
          'product_source_candidate_not_exact': 'real_provisional', 'product_visual_candidate_not_exact': 'real_provisional',
          'product_intended_reference_conditioned': 'synthetic', 'agent_audited_text_link_not_admitted': 'roskachestvo_weak'}
OPEN_STATES = ('open_evaluation_only_no_fit', 'open_synthetic_evaluation_only_no_fit')
CHANNELS = {'label': 'front_label', 'context': 'context'}


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def membership(root):
    """SHA / group / capture-group sets of every pool that must stay outside any fit (membership fields only)."""
    from rshb_vine.data_protection import load_protection
    out = {}
    for role in ('validation', 'test', 'protected_hold'):
        recs = read_json(root / (WEB232 + role + '.json'))['records']
        out['web232_' + role] = {'sha': {r['sha256'] for r in recs}, 'group': {r['group_id'] for r in recs},
                                 'capture': {c for r in recs for c in r.get('capture_group_ids') or []}}
    closed = [r for p in CLOSED for r in read_json(root / p)['records']]
    out['closed75'] = {'sha': {r.get('image_sha256') or r['sha256'] for r in closed},
                       'group': {r['split_group_id'] for r in closed if r.get('split_group_id')},
                       'capture': {r['capture_session_id'] for r in closed if r.get('capture_session_id')}}
    protection = load_protection(root)
    rows = protection['memberships'] + protection['derivatives']
    out['protected_no_read'] = {'sha': {r['sha256'] for r in rows if r['protected_state'] not in OPEN_STATES},
                                'group': set(), 'capture': set()}
    return out


def fit_prohibited_open(root):
    """Readable open evaluation rows whose protection forbids ranker/encoder fit (controls, never fit)."""
    from rshb_vine.data_protection import load_protection
    return {r['sha256'] for r in load_protection(root)['memberships'] if r['protected_state'] in OPEN_STATES}


def b3_stage5_query_sha(root):
    text = (root / STAGE5_STEPS).read_text()
    return set(re.findall(r'"query_source_sha256":"([0-9a-f]{64})"', text))


def b3_parent_real_sha(root):
    real = read_json(root / B3_PARENT)['real']
    return {r['image_sha256'] for r in real} | {a for r in real for a in r.get('source_ancestor_sha') or []}


def unified(root):
    return {r['sha256']: r for r in read_json(root / UNIFIED)['records']}


def image_index(root):
    """Candidate local paths by SHA: rti1 snapshot paths and sha-prefixed internet slug1000 files (verified on use)."""
    index = defaultdict(list)
    for r in jsonl(root / RTI1_SNAPSHOT):
        index[r['image_sha256']].append(r['image_path'])
    for f in (root / SLUG1000).rglob('*'):
        m = re.search(r'([0-9a-f]{10})\.[a-z]+$', f.name)
        if m and f.is_file():
            index[m.group(1)].append(str(f.relative_to(root)))
    return index


def local_image(root, sha, record, index):
    candidates = list((record or {}).get('paths', [])) + index.get(sha, []) + index.get(sha[:10], [])
    candidates += [str(p.relative_to(root)) for p in sorted((root / 'data/source-media').glob(sha + '.*'))]
    for p in candidates:
        path = Path(p) if p.startswith('/') else root / p
        if path.is_file() and sha256(path) == sha:
            return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    return None


def hits(sha, groups, pools):
    return sorted(name for name, m in pools.items()
                  if sha in m['sha'] or groups & (m['group'] | m['capture']))


def roster_audit(root):
    root = Path(root)
    fit = jsonl(root / FIT471)
    q_sha = {r.get('image_sha256') for r in read_json(root / QUARANTINE)['rows']}
    pools, b3q, uni = membership(root), b3_stage5_query_sha(root), unified(root)
    no_fit, index, b3p = fit_prohibited_open(root), image_index(root), b3_parent_real_sha(root)
    gate = {r['sha256']: r for r in read_json(root / (GATE + '/protocol.json'))['rows']}
    train137 = {r['sha256']: r for r in read_json(root / (WEB232 + 'train.json'))['records']}
    rows, counts = [], Counter()
    for r in fit:
        sha = r['image_sha256']
        u = uni.get(sha)
        exp = (u or {}).get('model_exposure', {})
        groups = {r['group'], r['lineage_group']}
        if sha in train137:
            groups |= {train137[sha]['group_id'], *(train137[sha].get('capture_group_ids') or [])}
        cohort = r['query_id'].split('/')[1].split(':')[0] if r['query_id'].startswith('historical865/') else r['query_id'].split('-')[0]
        rec = {'query_id': r['query_id'], 'instance_id': r['instance_id'], 'image_sha256': sha, 'cohort': cohort,
               'source_kind': r['source_kind'], 'group': r['group'], 'cv_fold': r['cv_fold'],
               'quarantine26': sha in q_sha, 'fit_prohibited_open': sha in no_fit,
               'b3_stage5_query_fit': sha in b3q,
               'b3_parent_real_fit': sha in b3p, 'unified_b3_direct_fit': exp.get('B3_encoder_direct_fit'),
               'in_open593': sha in gate, 'open593_set': gate[sha]['set'] if sha in gate else None,
               'closed_or_protected_hits': hits(sha, groups, pools),
               'local_image': local_image(root, sha, u, index) if sha not in gate else gate[sha]['path'],
               'unified_role': (u or {}).get('historical_roles')}
        rows.append(rec)
        counts[(cohort, rec['source_kind'], rec['in_open593'], rec['b3_stage5_query_fit'], rec['quarantine26'],
                bool(rec['closed_or_protected_hits']), rec['local_image'] is not None)] += 1
    return rows, counts


def open_rank(order, gt, mapping):
    """(GT product rank, top1 status): top1 is 'correct' / 'wrong', 'absent' for an empty channel and
    'mapping_unknown' when the first retrieved slug has no snapshot-03 product; unmapped slugs never shift ranks."""
    if not order:
        return None, 'absent'
    seen = []
    for slug in order:
        p = mapping.get(slug)
        if p and p not in seen:
            seen.append(p)
    top = mapping.get(order[0])
    status = 'mapping_unknown' if top is None else 'correct' if top in gt else 'wrong'
    return next((i + 1 for i, p in enumerate(seen) if p in gt), None), status


def open_matrix(root):
    """Every labelled focal target of the saved open593 replay: B3 label/context GT ranks versus the current answer."""
    from rshb_vine.catalog_training_evaluation_v2 import pins
    root = Path(root)
    mapping = pins.product03()
    pools, b3q = membership(root), b3_stage5_query_sha(root)
    fit_sha = {r['image_sha256'] for r in jsonl(root / FIT471)}
    no_fit, b3p = fit_prohibited_open(root), b3_parent_real_sha(root)
    protocol = {r['key']: r for r in read_json(root / (GATE + '/protocol.json'))['rows']}
    ledger = {r['key']: r for r in jsonl(root / (GATE + '/receipts/ledger.jsonl'))}
    train137 = {r['sha256']: r for r in read_json(root / (WEB232 + 'train.json'))['records']}
    replay = [r for r in jsonl(root / OPEN_ROWS) if r['gt_products'] and r['focal']]
    bodies, out = {}, []
    for r in replay:
        prow, rec = protocol[r['key']], ledger[r['key']]
        sha = prow['sha256']
        groups = set()
        if sha in train137:
            groups = {train137[sha]['group_id'], *(train137[sha].get('capture_group_ids') or [])}
        blocked = hits(sha, groups, pools)
        entry = {'key': r['key'], 'instance_id': r['instance_id'], 'cohort': r['cohort'], 'set': r['set'],
                 'slice': SLICES.get(r['gt_level'], r['gt_level']), 'gt_level': r['gt_level'], 'focal': r['focal'],
                 'image_sha256': sha, 'group': (train137[sha]['group_id'] if sha in train137 else None),
                 'b3_stage5_query_fit': sha in b3q, 'ranker_fit471': sha in fit_sha, 'fit_prohibited_open': sha in no_fit,
                 'b3_parent_real_fit': sha in b3p, 'closed_or_protected_hits': blocked}
        if blocked:
            entry['status'] = 'blocked_membership_not_read'
            out.append(entry)
            continue
        if r['key'] not in bodies:
            path = root / rec['reused_from'] if rec.get('reused_from') else root / GATE / 'receipts/responses' / rec['file']
            if sha256(path) != rec['response_sha256']:
                raise ValueError('saved response SHA differs: ' + r['key'])
            bodies = {r['key']: json.load(gzip.open(path, 'rt'))['body']}
        target = next(t for t in bodies[r['key']]['product_identity_evidence']['targets']
                      if str(t['instance_id']) == r['instance_id'])
        gt = set(r['gt_products'])
        arm = target['raw_visual']['arms']['B3']['retrieval']['channel_top20']
        for name, channel in CHANNELS.items():
            order = [x['slug'] for x in arm.get(channel, [])]
            entry[name + '_gt_rank'], entry[name + '_top1'] = open_rank(order, gt, mapping)
            entry[name + '_top1_is_selected'] = (mapping.get(order[0]) == r['baseline']['product03']) if order and mapping.get(order[0]) else None
        entry['current_correct'] = bool(r['baseline']['correct'])
        entry['current_gt_rank'] = r['baseline']['gt_rank']
        entry['current_selected_product'] = r['baseline']['product03']
        entry['guard'] = r['baseline_guard']
        entry['geometry'] = r['baseline_geometry']
        entry['status'] = 'read'
        out.append(entry)
    return out


def summarize_matrix(rows):
    table = defaultdict(Counter)
    for r in rows:
        if r['status'] != 'read':
            table[r['slice']]['blocked'] += 1
            continue
        mark = {'correct': '+', 'wrong': '-', 'absent': '0', 'mapping_unknown': '?'}
        cell = 'L' + mark[r['label_top1']] + 'C' + mark[r['context_top1']] + ('F+' if r['current_correct'] else 'F-')
        table[r['slice']][cell] += 1
        table[r['slice']]['n'] += 1
        table[r['slice']]['b3_stage5_query_fit'] += r['b3_stage5_query_fit']
        table[r['slice']]['b3_parent_real_fit'] += r['b3_parent_real_fit']
        table[r['slice']]['ranker_fit471'] += r['ranker_fit471']
        table[r['slice']]['fit_prohibited_open'] += r['fit_prohibited_open']
        if r['b3_stage5_query_fit']:
            table[r['slice'] + '|b3fit']['n'] += 1
            table[r['slice'] + '|b3fit'][cell] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}


def full_fit_rows(root):
    from rshb_vine.combined_ranker_v1 import roster as combined
    from rshb_vine.io import verify
    frozen = verify(read_json(Path(root) / 'runs/combined-ranker-v1/roster.json'))
    return {r['query_id']: r for r in combined.load_rows(root, frozen, 'fit')}


def row_groups(row):
    return {g for g in [row.get('capture_group'), *(row.get('capture_group_ids') or []), *(row.get('source_group_ids') or [])] if g}


def lineage_components(uni):
    """Undirected declared-ancestor components of the unified ledger (derivatives, crops, same-original siblings)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for sha, r in uni.items():
        for a in r.get('ancestors') or []:
            parent[find(sha)] = find(a)
    members = defaultdict(set)
    for x in list(parent):
        members[find(x)].add(x)
    return find, members


def crosspool(root, roster):
    """Clean fit rows versus every held pool by SHA, declared lineage component and capture/source group."""
    from rshb_vine.data_protection import load_protection
    root = Path(root)
    uni, full, pools = unified(root), full_fit_rows(root), membership(root)
    protection = load_protection(root)
    find, members = lineage_components({**uni, **{d['sha256']: {'ancestors': [d['parent_sha256']]} for d in protection['derivatives']}})
    sha_pools = {name: m['sha'] for name, m in pools.items()}
    sha_pools['b3_stage5_query'] = b3_stage5_query_sha(root)
    sha_pools['b3_parent_real'] = b3_parent_real_sha(root)
    sha_pools['gallery_reference'] = ({s for s, r in uni.items() if 'rshb_gallery_reference' in r['sources']}
                                      | set(re.findall(r'"source_sha256":"([0-9a-f]{64})"', (root / STAGE5_STEPS).read_text())))
    sha_pools['unified_held_validation_test'] = {s for s, r in uni.items() if set(r.get('held_relations') or []) & {'validation', 'test'}}
    sha_pools['open_eval_fit_prohibited'] = fit_prohibited_open(root)
    group_pools = {name: m['group'] | m['capture'] for name, m in pools.items()}
    group_pools['protection_membership_groups'] = {g for r in protection['memberships'] for g in r.get('group_ids') or []}
    exposed = [r for r in roster if r['b3_stage5_query_fit']]
    group_pools[EXPOSED_GROUPS] = {g for r in exposed for g in row_groups(full[r['query_id']])}
    web_train = {r['sha256']: r for r in read_json(root / (WEB232 + 'train.json'))['records']}
    group_pools[EXPOSED_GROUPS] |= {g for r in exposed if r['image_sha256'] in web_train
                                                       for g in [web_train[r['image_sha256']]['group_id'],
                                                                 *(web_train[r['image_sha256']].get('capture_group_ids') or [])]}
    from rshb_vine.runtime_training_intake_v1.intake import _dhash
    prints = read_json(root / FINGERPRINTS)['items']
    gallery = [(v['dhash'], set(v['slugs'])) for v in prints.values() if v['kind'] == 'gallery']
    out = []
    for r in roster:
        if r['b3_stage5_query_fit'] or r['b3_parent_real_fit'] or r['quarantine26'] or not r['local_image']:
            continue
        sha, row = r['image_sha256'], full[r['query_id']]
        h = prints[sha]['dhash'] if sha in prints else _dhash(root / r['local_image'])
        gt = set(row['ground_truth']['acceptable_slugs'])
        same = min([bin(h ^ g).count('1') for g, slugs in gallery if slugs & gt] or [None], key=lambda v: 99 if v is None else v)
        component = members.get(find(sha), {sha}) | {sha}
        groups = row_groups(row)
        if sha in web_train:
            groups |= {web_train[sha]['group_id'], *(web_train[sha].get('capture_group_ids') or [])}
        out.append({'query_id': r['query_id'], 'image_sha256': sha, 'cohort': r['cohort'], 'cv_fold': r['cv_fold'],
                    'group': r['group'], 'lineage_component_size': len(component),
                    'sha_hits': sorted(n for n, s in sha_pools.items() if sha in s),
                    'lineage_hits': sorted(n for n, s in sha_pools.items() if (component - {sha}) & s),
                    'group_hits': sorted(n for n, g in group_pools.items() if groups & g),
                    'gallery_same_product_dhash_bits': same,
                    'gallery_same_product_near_duplicate': same is not None and same <= DHASH_NEAR_BITS,
                    'groups': sorted(groups)})
    return out
