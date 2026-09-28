"""Zero-target release v1: candidate freeze/serve/replay, OPEN593 class census, paired HTTP protocol and admission,
then stage/activate/rollback/recover of the factory F8 release through scripts/recognize.py.

Candidate (pre-activation): the frozen release 99451df0 plus the branch (a) route adapter on loopback 8196 under the
live factory F7. Release: factory F8 and receipts C8 add the dispatch8 B3-only base and the zero-target release
kinds, so the F7-pinned dispatch7 chain no longer loads; the release runs over byte-identical dispatch8 copies. The
state is the tuple (factory, recognize, receipts, pointer) sha. Stage seals a restoration snapshot of the live
repair-v2 bytes (F7, R3, C7, pointer fc55103d), so rollback and recover read only that snapshot, the staged archive
and the sealed manifest. A switch journals its full plan before the first write. The service on 8175 must be
stopped around activate/rollback; config/local-release-v1/services.json is unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import digest, read_json, seal, sha256, verify, write_json  # noqa: E402

OUT = 'runs/next-cycle-v1/target'
PROTOCOL = OUT + '/protocol.json'
HTTP = OUT + '/http'
RECORDS = HTTP + '/records.jsonl'
CONCURRENCY = HTTP + '/concurrency.jsonl'
ADMISSION = OUT + '/admission.json'
CENSUS = OUT + '/class-census.json'
DECISION = OUT + '/root-decision.json'
OLD_SCOPE = 'runs/causal-crops-v1/live/protocol.json'
OLD_SCOPE_CHECKSUM = '133776ef371a417860bb9d76218a3c55d92240e0c66de043b0582f42b89f5ac6'
OPEN593 = 'runs/recognition-repair-20260926/open-http-gate/receipts'
WEB232 = 'data/evaluation/web232-v1'
BUDGET_SECONDS = 900
REQUEST_TIMEOUT = 90
EXTRA_LIMIT_S = 5.0
MAX_PIXELS = 24_000_000
MAX_BYTES = 20 * 1024 * 1024
TOTAL_LIMIT_S = 8.0
TIMING_KEYS = {'timing_ms', 'timing', 'ms', 'seconds'}
PROBES = HTTP + '/probes.jsonl'
IDENTITY = '<effective-profile>'

FACTORY = 'rshb_vine/recognition_factory.py'
RECOGNIZE = 'scripts/recognize.py'
RECEIPTS = 'rshb_vine/recognition_receipts.py'
CURRENT = 'config/recognition-current.json'
CODE = (FACTORY, RECOGNIZE, RECEIPTS)
COMPOSITES = ('config/recognition-coherent-v1.json', 'config/recognition-coherent-v1-dispatch2.json',
              'config/recognition-coherent-v1-dispatch3.json', 'config/recognition-coherent-v1-dispatch4.json',
              'config/recognition-coherent-v1-dispatch5.json', 'config/recognition-coherent-v1-dispatch6.json',
              'config/recognition-coherent-v1-dispatch7.json')
DISPATCH7 = COMPOSITES[-1]
SYSTEMIC7 = 'config/recognition-systemic-v2-dispatch7.json'
TARGET7 = 'config/recognition-target-contract-v2-dispatch7.json'
GEOMETRY7 = 'config/recognition-geometry-v2-dispatch7.json'
BASE7 = 'config/recognition-b3-only-v1-dispatch7.json'
DISPATCH8 = 'config/recognition-coherent-v1-dispatch8.json'
SYSTEMIC8 = 'config/recognition-systemic-v2-dispatch8.json'
TARGET8 = 'config/recognition-target-contract-v2-dispatch8.json'
GEOMETRY8 = 'config/recognition-geometry-v2-dispatch8.json'
BASE8 = 'config/recognition-b3-only-v1-dispatch8.json'
V2_RELEASE = 'config/recognition-repair-v2-release.json'
V2_MANIFEST = 'config/recognition-repair-v2-release-manifest.json'
RELEASE = 'config/zero-target-release-v1-profile.json'
MANIFEST = 'config/zero-target-release-v1-manifest.json'
MIGRATION = 'config/recognition-dispatch-migration-v7.json'
MIGRATIONS_OLD = tuple('config/recognition-dispatch-migration-v%d.json' % v for v in range(1, 7))
GALLERY_MIGRATION_OLD = 'config/b3-only-v1-gallery-code-migration-v3.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration-v4.json'
GALLERY_MIGRATIONS_OLD = ('config/b3-only-v1-gallery-code-migration.json', 'config/b3-only-v1-gallery-code-migration-v2.json',
                          GALLERY_MIGRATION_OLD)
DRAFTS = {FACTORY: OUT + '/dispatch/recognition_factory.py', RECEIPTS: OUT + '/dispatch/recognition_receipts.py'}
ARCHIVE_NAMES = {FACTORY: 'config/dispatch-archive/recognition_factory.%s.py',
                 RECEIPTS: 'config/dispatch-archive/recognition_receipts.%s.py'}
RESTORATION = OUT + '/release/restoration.json'
RESTORE_DIR = OUT + '/release/restore'
JOURNAL = OUT + '/release/journal.jsonl'
F1 = '9d9d3ac003ff0cfc13f4d15f5d81beddea99e5c42f9278e411463fab87aea87b'
F2 = 'be2c15d4b8380d89791e22fb6e984f1111455b45b6f78172f5f33a103b9f778f'
F3 = '08b4ab3fbcc31278458ce52c5bb0e17e0098ca06796b91016bfe4485da4b5a35'
F4 = 'f16a78f8ec91296c0f50f1ba1c9309798704a13125a85dcab748d893836723b0'
F5 = '0c3c7f20121817b5e0c1dbf22d8c4bc4af46a91cc872783d40c8fdf1a914e8a6'
F6 = '6d58e20208ced9e9c0a53b6dac193b138fa30dd810a790d575b12278f878dd43'
F7 = '2ab82e69084e05fdd6ef5fcaf727a31e6f1e15512658a00732c188f9ad9b97ca'
R3 = '6faabf40f02e265e73b26d6b813a88c54fc4cd42daaa90516002e924843b4a77'
C7 = 'de77f44d528a1478d8742970dfe468efb029c390eb55b0e0f8f2506a9103af86'
ARCHIVE = {F1: 'config/dispatch-archive/recognition_factory.9d9d3ac0.py',
           F2: 'config/dispatch-archive/recognition_factory.be2c15d4.py',
           F3: 'config/dispatch-archive/recognition_factory.08b4ab3f.py',
           F4: 'config/dispatch-archive/recognition_factory.f16a78f8.py',
           F5: 'config/dispatch-archive/recognition_factory.0c3c7f20.py',
           F6: 'config/dispatch-archive/recognition_factory.6d58e202.py',
           F7: 'config/dispatch-archive/recognition_factory.2ab82e69.py',
           R3: 'config/dispatch-archive/recognize.6faabf40.py',
           C7: 'config/dispatch-archive/recognition_receipts.de77f44d.py'}
V2_POINTER_SHA256 = 'fc55103d3d6502eb3be2bc341e96d43a6cf294324d567eedc82cd0beb1f1a075'
V2_CHECKSUM = '99451df062a5a9675a18ebc23d673f98e44c1e721483faaf6a6c78876fbd7687'
V2_STATE = [F7, R3, C7, V2_POINTER_SHA256]
ROLLBACK = ('scripts/zero_target_release_v1.py rollback  (restores repair_v2_active from the sealed snapshot: '
            'factory 2ab8, receipts de77, pointer fc55103d = repair-v2 release 99451df0)')


def _gz_read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def _gz_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Response file exists; immutable: ' + str(path))
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def scrub(value, drop=()):
    """Response without declared timing fields (timing_ms, timing, ms, seconds, *_ms, *_seconds) and the named blocks."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()
                if k not in TIMING_KEYS and k not in drop and not k.endswith('_ms') and not k.endswith('_seconds')}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def normalized(body, candidate, triggered=False):
    """Timing-free response with only the declared identity paths normalized and the declared added paths removed."""
    from rshb_vine.zero_target_release_v1 import release as ZR
    out = json.loads(json.dumps(body))
    for name, key in ZR.IDENTITY_PATHS:
        if isinstance(out.get(name), dict) and key in out[name]:
            out[name][key] = IDENTITY
    if candidate:
        for name, key in ZR.ADDED_PATHS:
            if isinstance(out.get(name), dict):
                out[name].pop(key, None)
        out.pop(ZR.BLOCK, None)
        if triggered:
            out.pop('zero_target_route', None)
    return scrub(out)


def first_diff(a, b, path=''):
    if type(a) is not type(b):
        return path or '/'
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return path + '/' + k
            found = first_diff(a[k], b[k], path + '/' + k)
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return path + '[len]'
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_diff(x, y, '%s[%d]' % (path, i))
            if found:
                return found
        return None
    return None if a == b else (path or '/')


def answer(result):
    """Public answer summary: decision, candidate, ranked slugs and every target's box and top candidates."""
    if not isinstance(result, dict):
        return result
    return {'decision': result.get('decision'), 'best_candidate': result.get('best_candidate'),
            'ranked': [c.get('slug') for c in result.get('ranked_candidates') or []],
            'targets': [{'instance_id': t.get('instance_id'), 'bbox': [round(v, 1) for v in t.get('bbox') or []],
                         'physical': t.get('physical_bottle_localized'), 'geometry_source': t.get('geometry_source'),
                         'top': [c.get('slug') for c in (t.get('candidates') or [])[:5]]}
                        for t in result.get('targets') or []]}


# ---- census (metadata only) ----

def _saved_body(saved, key):
    if not isinstance(saved, dict) or set(saved) not in ({'http_status', 'body'}, {'http_status', 'body', 'request'}):
        raise ValueError('Unexpected saved response envelope: ' + key)
    if saved['http_status'] != 200 or not isinstance(saved['body'], dict):
        raise ValueError('Saved 200 response without a body dict: ' + key)
    return saved['body']


def census(root):
    from rshb_vine.zero_target_release_v1 import route as Z
    base = root / OPEN593
    ledger = _rows(base / 'ledger.jsonl')
    if len({r['key'] for r in ledger}) != len(ledger):
        raise ValueError('Duplicate OPEN593 ledger keys')
    rows, counts, envelopes = [], {}, {}
    for row in ledger:
        entry = {'key': row['key'], 'set': row['set'], 'http_status': row['http_status']}
        if row['http_status'] == 200:
            source = root / row['reused_from'] if row.get('reused_from') else base / 'responses' / row['file']
            raw = gzip.decompress(source.read_bytes())
            if row['response_sha256'] not in (hashlib.sha256(raw).hexdigest(), sha256(source)):
                raise ValueError('Saved response bytes differ from the ledger: ' + row['key'])
            saved = json.loads(raw)
            name = '+'.join(sorted(saved)) if isinstance(saved, dict) else type(saved).__name__
            envelopes[name] = envelopes.get(name, 0) + 1
            result = _saved_body(saved, row['key'])
            rescue = result.get('bottle_rescue') or {}
            proposals = rescue.get('retry_instances') or []
            route = result.get('canvas_closeup_route') or {}
            entry.update(targets=len(result.get('targets') or []), route_reason=route.get('reason'),
                         route_recovered=route.get('recovered'), raw_nms_proposals=rescue.get('raw_nms_proposals'),
                         proposals=len(proposals), instances=len(result.get('instances') or []),
                         object_threshold=(result.get('object_filter') or {}).get('threshold'),
                         proposal_facts=[{'label_bbox': p.get('label_bbox') is not None, 'wine_score': p.get('wine_score')}
                                         for p in proposals],
                         branch_a_eligible=Z.ignored_proposals(result) is not None)
        key = '|'.join(map(str, (entry['set'], entry.get('targets') == 0, entry.get('route_reason'),
                                 entry.get('branch_a_eligible'))))
        counts[key] = counts.get(key, 0) + 1
        rows.append(entry)
    ok = [r for r in rows if r['http_status'] == 200]
    eligible = [r for r in ok if r['branch_a_eligible']]
    by_n = {}
    for r in eligible:
        by_n[str(r['proposals'])] = by_n.get(str(r['proposals']), 0) + 1
    return seal({'kind': 'zero-target-release-v1-class-census', 'source': OPEN593,
                 'ledger_sha256': sha256(base / 'ledger.jsonl'), 'rows': len(rows), 'http_200': len(ok),
                 'http_other': len(rows) - len(ok), 'envelopes': envelopes,
                 'fields': 'targets count, canvas route trace, object threshold, bottle proposals (label presence, wine '
                           'score), instance count; no pixels, GT, slugs or inference',
                 'zero_target_200': sum(1 for r in ok if r['targets'] == 0),
                 'with_target_200': sum(1 for r in ok if r['targets'] > 0),
                 'branch_a_eligible': len(eligible), 'branch_a_eligible_by_proposals': by_n,
                 'branch_a_eligible_by_set': {s: sum(1 for r in eligible if r['set'] == s) for s in ('train137', 'supplemental456')},
                 'counts_set_zero_reason_eligible': counts, 'eligible_rows': eligible,
                 'population': 'saved OPEN593 HTTP responses of repair-v1 013c845a (release 51eedd7f); the geometry and '
                               'canvas route stages upstream of the selector are unchanged in 99451df0, but this is not '
                               'the live 99451df0 population; W0353 route trace confirmed live by root '
                               '(runs/open-case-consult-20260926/W0353-current.json)',
                 'rule': Z.RULE, 'not_quality_evidence': True})


# ---- protocol ----

EXCLUDED_SOURCES = (
    ('closed75_base', 'data/internet-evaluation-intake-20260921/locked-final/manifest.json', 'records', 'sha256', None, 19),
    ('closed75_extension', 'data/internet-evaluation-intake-20260921/extension-v2/locked-final/manifest.json', 'records',
     'sha256', None, 56),
    ('reset_validation_locked', 'data/evaluation/reset-v1/membership.json', 'members', 'image_sha256',
     ('validation', 'locked_test'), 196),
    ('web232_validation', WEB232 + '/validation.json', 'records', 'sha256', None, 46),
    ('web232_test', WEB232 + '/test.json', 'records', 'sha256', None, 46),
    ('web232_protected_hold', WEB232 + '/protected_hold.json', 'records', 'sha256', None, 3),
    ('calibration_current_pool', 'data/calibration-intake-v1/pool-reviewed-v5.json', 'records', 'image_sha256', None, 13),
    ('calibration_real_unknown', 'data/real-unknown-calibration-v1/calibration-pool.json', 'records', 'image_sha256', None, 3),
    ('calibration_expansion_rshb', 'runs/calibration-expansion-v1/rshb-roster.json', 'records', 'image_sha256', None, 67),
    ('calibration_expansion_kultovo', 'runs/calibration-expansion-v1/kultovo-source-audit.json', 'records',
     'image_sha256', None, 55))
CALIBRATION_POINTER = 'data/calibration-intake-v1/current.json'


def excluded_shas(root, protection):
    """Explicit closed/held/calibration SHA sets (metadata only), expanded through protected derivatives."""
    if read_json(root / CALIBRATION_POINTER)['manifest'] != 'data/calibration-intake-v1/pool-reviewed-v5.json':
        raise ValueError('Calibration pointer moved; update the explicit exclusion list')
    sets, audit = {}, {}
    for name, path, field, key, splits, expected in EXCLUDED_SOURCES:
        doc = read_json(root / path)
        shas = {r[key] for r in doc[field] if splits is None or r.get('split') in splits}
        if len(shas) != expected:
            raise ValueError('Exclusion source count changed: %s %d != %d' % (name, len(shas), expected))
        sets[name] = shas
        audit[name] = {'path': path, 'sha256': sha256(root / path), 'images': len(shas)}
    for name in sets:
        changed = True
        while changed:
            changed = False
            for row in protection['derivatives']:
                if row['parent_sha256'] in sets[name] and row['sha256'] not in sets[name]:
                    sets[name].add(row['sha256'])
                    changed = True
    return sets, audit


def build_protocol(root):
    from rshb_vine.coherent_challenger_v1.scope import _closed_shas
    from rshb_vine.data_protection import load_protection
    from rshb_vine.zero_target_release_v1 import release as ZR
    old = verify(read_json(root / OLD_SCOPE))
    if old['checksum'] != OLD_SCOPE_CHECKSUM:
        raise ValueError('Old causal-crops scope differs')
    descriptor = ZR.load_candidate_descriptor(root)
    protection = load_protection(root)
    closed, states = _closed_shas(protection)
    explicit, audit = excluded_shas(root, protection)
    items, excluded = [], []
    for it in old['scope']:
        sha = it['image_sha256']
        hits = sorted(name for name, shas in explicit.items() if sha in shas) + (['protection_closed'] if sha in closed else [])
        if hits:
            excluded.append({'query_id': it['query_id'], 'role': it['role'], 'image_sha256': sha, 'reasons': hits})
            continue
        if sha256(root / it['image_path']) != sha:
            raise ValueError('Image SHA mismatch: ' + it['image_path'])
        from PIL import Image
        with Image.open(root / it['image_path']) as image:
            pixels = image.width * image.height
        oversized = pixels > MAX_PIXELS or (root / it['image_path']).stat().st_size > MAX_BYTES
        items.append({'key': digest([it['role'], it['query_id'], sha])[:16], 'role': it['role'],
                      'query_id': it['query_id'], 'image_path': it['image_path'], 'image_sha256': sha,
                      'pixels': pixels, 'expected_http': 422 if oversized else 200,
                      'protection_state': sorted(states.get(sha, [])), 'development_expected_slugs': it['expected_slugs']})
    if len({i['key'] for i in items}) != len(items) or len({i['image_sha256'] for i in items}) != len(items):
        raise ValueError('Duplicate scope items')
    if len(items) + len(excluded) != len(old['scope']):
        raise ValueError('Scope coverage count mismatch')
    return seal({
        'kind': 'zero-target-release-v1-http-protocol', 'written_before_run': True,
        'candidate_descriptor': descriptor['checksum'], 'candidate_sources_sha256': descriptor['sources_sha256'],
        'rule': descriptor['rule'], 'scope_source': {'path': OLD_SCOPE, 'checksum': OLD_SCOPE_CHECKSUM}, 'items': items,
        'scope_counts': {r: sum(1 for i in items if i['role'] == r) for r in sorted({i['role'] for i in items})},
        'membership_audit': {'protection_pointer_sha256': protection['pointer_sha256'], 'source_items': len(old['scope']),
                             'included': len(items), 'excluded': excluded,
                             'rule': 'exclusion only by a prior protection/closed/held/calibration membership; no GT-based selection',
                             'explicit_sources': audit, 'sha_verified': len(items),
                             'open_no_fit_states_kept_as_labels': sorted({s for i in items for s in i['protection_state']}),
                             'note': 'open/no-fit diagnostic holds are labels only, distinct from protected locked; '
                                     'no held-GT positive claims'},
        'arms': {'baseline': 'live 8175 current 99451df0 (unchanged)', 'candidate': '8196 ZeroTargetCandidate'},
        'procedure': 'health 200+ready and profile on both servers; one excluded warm-up per server; per item an attempt '
                     'record, image SHA and frozen pins re-verified, then one POST /v1/recognize (no ROI) to baseline and '
                     'then candidate; timeout %ds; any non-200 or transport error stops without retry; resume refuses '
                     'after any error or an attempt without receipt; cumulative budget %ds including warm-up; then 3 waves '
                     'x 3 POSTs to the candidate, first submitter rotated, 100 ms stagger' % (REQUEST_TIMEOUT, BUDGET_SECONDS),
        'parity': 'exact equality after dropping timing_ms, timing, ms, seconds, *_ms, *_seconds; only the declared '
                  'identity paths are normalized (%s) and only the declared candidate additions removed (%s, block %s, '
                  'and zero_target_route on triggered rows); answer-level equality reported separately'
                  % (['.'.join(p) for p in ZR.IDENTITY_PATHS], ['.'.join(p) for p in ZR.ADDED_PATHS], ZR.BLOCK),
        'probes': [{'id': 'P1', 'item': 'web232:W0353', 'endpoint': '/v1/recognize', 'roi': None, 'bottles': 'all',
                    'check': 'each arm equals its own serial addressed response (timing only); candidate still recovered'},
                   {'id': 'P2', 'item': 'first multi_control', 'endpoint': '/v1/recognize',
                    'roi': 'integer-rounded bbox of the first target of the baseline serial response', 'bottles': 'all',
                    'check': 'baseline vs candidate parity; candidate adapter disabled, no event'},
                   {'id': 'P3', 'item': 'web232:W0353', 'endpoint': '/v1/recognize', 'roi': 'full frame [0,0,w,h]',
                    'bottles': 'addressed', 'check': 'baseline vs candidate parity; candidate adapter disabled, no event'},
                   {'id': 'P4', 'item': 'web232:W0353', 'endpoint': '/v2/targets', 'roi': None, 'bottles': 'addressed',
                    'check': 'each arm compact decision/best_candidate/requires_target_selection/reasons and '
                             'target_contract equal its own serial /v1 response (parent_response_digest excluded)'}],
        'probe_requests': 8,
        'expected_invalid': 'items with pixels > %d or bytes > %d (decode limits) must return the same 422 body on both '
                            'arms; declared from file metadata before the run' % (MAX_PIXELS, MAX_BYTES),
        'gates': {'T': 'every request HTTP 200 on both arms with profile 99451df0 (and the candidate descriptor), '
                       'or the declared identical 422 for expected_invalid items',
                  'G1': '0 targets on every nonwine candidate response',
                  'G2': 'exact parity on every untriggered row (includes every baseline with a target) and on every '
                        'triggered row that did not recover',
                  'G3': '>=1 recovered row; each: exactly one event, reason recovered, canvas fired, non-empty canvas '
                        'labels, published target IoU >= 0.9 with a canvas label, baseline 0 targets, candidate 1 target, '
                        'physical_bottle_localized false, geometry_source canvas route; causal review per row by root; '
                        'development_expected_slugs only reported',
                  'G4': 'at most one retry (one zero_target_route event) per request',
                  'G5': 'per triggered row: candidate - baseline HTTP seconds <= %.1f and candidate <= %.1f s; '
                        'reported per row, no aggregate over untriggered rows' % (EXTRA_LIMIT_S, TOTAL_LIMIT_S),
                  'G7': 'all 4 contract probes pass (8 predeclared requests, no retry)',
                  'G6': 'HTTP serialized busy/parity check, no simultaneous-inference claim: exactly 9 unique (wave, item) requests; each of the 3 items accepted (200) at least '
                        'once and each 200 equals its serial candidate response (parity rule); every other status is '
                        '503 Inference busy'},
        'stop': 'any gate failure -> no stage; each failure analysed; no threshold/rule change after results',
        'not_gt': 'development_expected_slugs are stored development roles, not GT; no tuning',
        'budget_seconds': BUDGET_SECONDS})


def _post(url, path, timeout=REQUEST_TIMEOUT, endpoint='/v1/recognize', form=None):
    import httpx
    data = Path(path).read_bytes()
    started = time.perf_counter()
    try:
        response = httpx.post(url + endpoint, files={'image': (Path(path).name, data)}, data=form or {}, timeout=timeout)
    except httpx.HTTPError as exc:
        return {'status': None, 'error': type(exc).__name__, 'seconds': time.perf_counter() - started,
                'request_sha256': hashlib.sha256(data).hexdigest()}, None
    try:
        body = response.json()
    except ValueError:
        body = None
    return {'status': response.status_code, 'seconds': time.perf_counter() - started,
            'request_sha256': hashlib.sha256(data).hexdigest(),
            'response_sha256': digest(body) if body is not None else None}, body


def _health(url):
    import httpx
    response = httpx.get(url + '/health/ready', timeout=10)
    body = response.json() if response.status_code == 200 else {}
    if response.status_code != 200 or body.get('ready') is not True:
        raise SystemExit('Server not ready: ' + url)
    return body


def _pins(root, protocol):
    from rshb_vine.zero_target_release_v1 import release as ZR
    if verify(read_json(root / PROTOCOL))['checksum'] != protocol['checksum']:
        raise SystemExit('Protocol changed during the run')
    if ZR.sources_sha(root) != protocol['candidate_sources_sha256']:
        raise SystemExit('Candidate sources changed since the protocol')
    if sha256(root / ZR.PARENT_RELEASE) != V2_POINTER_SHA256 or _live(root) != V2_STATE:
        raise SystemExit('Parent release or live dispatch/pointer changed')


def _response_profile(arm, body, protocol):
    body = body or {}
    contract = body.get('target_contract') or {}
    release = body.get('recognition_repair_release_v2') or {}
    block = body.get('zero_target_release_v1')
    if arm == 'baseline':
        if contract.get('profile_checksum') != V2_CHECKSUM or release.get('profile_checksum') != V2_CHECKSUM:
            return 'baseline profile is not 99451df0'
        return 'baseline carries the candidate block' if block is not None else None
    descriptor = protocol['candidate_descriptor']
    if contract.get('profile_checksum') != descriptor or release.get('profile_checksum') != descriptor:
        return 'candidate identity is not the protocol descriptor'
    if release.get('parent_release_profile_checksum') != V2_CHECKSUM:
        return 'candidate lacks the 99451df0 parent lineage'
    if (block or {}).get('descriptor_checksum') != descriptor or (block or {}).get('profile_checksum') != descriptor:
        return 'candidate block does not name the protocol descriptor'
    return None


def run_http(root, baseline_port, candidate_port):
    protocol = verify(read_json(root / PROTOCOL))
    _pins(root, protocol)
    base_url, cand_url = 'http://127.0.0.1:%d' % baseline_port, 'http://127.0.0.1:%d' % candidate_port
    records = root / RECORDS
    history = _rows(records)
    if history:
        raise SystemExit('HTTP records already exist; no resume after a partial or failed run (new admission required)')
    receipts, used = set(), 0.0

    def health():
        found = {'baseline': _health(base_url), 'candidate': _health(cand_url)}
        if found['baseline'].get('runtime_descriptor_checksum') != V2_CHECKSUM:
            raise SystemExit('Baseline server is not 99451df0')
        if found['candidate'].get('runtime_descriptor_checksum') != protocol['candidate_descriptor']:
            raise SystemExit('Candidate server does not serve the protocol descriptor')
        return found

    def request(item, arm, url, event):
        nonlocal used
        if sha256(root / item['image_path']) != item['image_sha256']:
            _append(records, {'event': 'error', 'key': item['key'], 'reason': 'image SHA changed'})
            raise SystemExit('Image SHA changed: ' + item['image_path'])
        meta, body = _post(url, root / item['image_path'])
        used += meta['seconds']
        _append(records, {'event': event, 'key': item['key'], 'arm': arm, **meta})
        problem = None
        if meta['status'] != item['expected_http'] or body is None:
            problem = 'HTTP %s (expected %s) %s' % (meta['status'], item['expected_http'], meta.get('error') or '')
        elif meta['request_sha256'] != item['image_sha256']:
            problem = 'request bytes differ from the item SHA'
        elif item['expected_http'] == 200:
            problem = _response_profile(arm, body, protocol)
        if problem:
            _append(records, {'event': 'error', 'key': item['key'], 'arm': arm, 'reason': problem})
            raise SystemExit('Stopped on %s/%s: %s; no retry' % (item['key'], arm, problem))
        return meta, body

    _append(records, {'event': 'start', 'protocol': protocol['checksum'], 'health': health()})
    warm = next(i for i in protocol['items'] if i['expected_http'] == 200)
    for arm, url in (('baseline', base_url), ('candidate', cand_url)):
        request(warm, arm, url, 'warmup')
    for item in protocol['items']:
        if used > BUDGET_SECONDS:
            _append(records, {'event': 'budget_stop', 'used_seconds': used,
                              'remaining': [i['key'] for i in protocol['items'] if i['key'] not in receipts]})
            raise SystemExit('Budget exhausted; recorded budget_stop')
        _pins(root, protocol)
        health()
        _append(records, {'event': 'attempt', 'key': item['key']})
        row = {'event': 'item', 'key': item['key'], 'role': item['role'], 'query_id': item['query_id']}
        for arm, url in (('baseline', base_url), ('candidate', cand_url)):
            meta, body = request(item, arm, url, 'request')
            _gz_write(root / HTTP / 'responses' / arm / (item['key'] + '.json.gz'), body)
            row[arm] = meta
        _append(records, row)
        receipts.add(item['key'])
    return len(receipts)


def run_concurrency(root, candidate_port, waves=3, stagger=0.1):
    protocol = verify(read_json(root / PROTOCOL))
    _pins(root, protocol)
    url = 'http://127.0.0.1:%d' % candidate_port
    path = root / CONCURRENCY
    if _rows(path):
        raise SystemExit('Concurrency already recorded; immutable')
    pick = {}
    for item in protocol['items']:
        if item['query_id'] == 'web232:W0353':
            pick['recovery'] = item
        elif item['role'] == 'nonwine' and item['expected_http'] == 200 and 'nonwine' not in pick:
            pick['nonwine'] = item
        elif item['role'] == 'single_control' and 'control' not in pick:
            pick['control'] = item
    names = ['recovery', 'nonwine', 'control']
    if set(pick) != set(names):
        raise SystemExit('Concurrency items not found in the protocol')
    for wave in range(waves):
        order = names[wave:] + names[:wave]
        with ThreadPoolExecutor(3) as pool:
            futures = {}
            for name in order:
                futures[name] = pool.submit(_post, url, root / pick[name]['image_path'])
                time.sleep(stagger)
            for name in order:
                meta, body = futures[name].result()
                file = None
                if body is not None and meta['status'] == 200:
                    file = '%s/concurrency/w%d-%s.json.gz' % (HTTP, wave, pick[name]['key'])
                    _gz_write(root / file, body)
                _append(path, {'event': 'concurrent', 'wave': wave, 'order': order.index(name), 'name': name,
                               'key': pick[name]['key'], **meta, 'file': file,
                               'detail': (body or {}).get('detail') if meta['status'] != 200 else None})
    return _rows(path)


def run_probes(root, baseline_port, candidate_port):
    """The 4 predeclared contract probes (8 requests) after the serial run; no retry, immutable."""
    from PIL import Image
    protocol = verify(read_json(root / PROTOCOL))
    _pins(root, protocol)
    path = root / PROBES
    if _rows(path):
        raise SystemExit('Probes already recorded; immutable')
    items = {i['query_id']: i for i in protocol['items']}
    w0353 = items['web232:W0353']
    multi = next(i for i in protocol['items'] if i['role'] == 'multi_control')
    serial_base = _gz_read(root / HTTP / 'responses' / 'baseline' / (multi['key'] + '.json.gz'))
    first = (serial_base.get('targets') or [{}])[0].get('bbox')
    if not first:
        raise SystemExit('First multi_control baseline response has no target box')
    with Image.open(root / w0353['image_path']) as image:
        frame = [0, 0, image.width, image.height]
    plans = (('P1', w0353, '/v1/recognize', None, 'all'), ('P2', multi, '/v1/recognize', [round(v) for v in first], 'all'),
             ('P3', w0353, '/v1/recognize', frame, 'addressed'), ('P4', w0353, '/v2/targets', None, 'addressed'))
    urls = {'baseline': 'http://127.0.0.1:%d' % baseline_port, 'candidate': 'http://127.0.0.1:%d' % candidate_port}
    for pid, item, endpoint, roi, bottles in plans:
        form = {'bottles': bottles, **({'target_roi': json.dumps(roi)} if roi is not None else {})}
        for arm in ('baseline', 'candidate'):
            if sha256(root / item['image_path']) != item['image_sha256']:
                raise SystemExit('Image SHA changed: ' + item['image_path'])
            meta, body = _post(urls[arm], root / item['image_path'], endpoint=endpoint, form=form)
            file = None
            if body is not None:
                file = '%s/probes/%s-%s.json.gz' % (HTTP, pid, arm)
                _gz_write(root / file, body)
            _append(path, {'event': 'probe', 'id': pid, 'arm': arm, 'key': item['key'], 'endpoint': endpoint,
                           'roi': roi, 'bottles': bottles, **meta, 'file': file})
            if meta['status'] != 200:
                raise SystemExit('Probe %s/%s HTTP %s; no retry' % (pid, arm, meta['status']))
    return _rows(path)


def _probe_gate(root):
    rows = {(r['id'], r['arm']): r for r in _rows(root / PROBES) if r.get('event') == 'probe'}
    if len(rows) != 8:
        return ['probes not run or incomplete: %d/8' % len(rows)], {}
    failures, facts = [], {}
    load = lambda pid, arm: _gz_read(root / rows[(pid, arm)]['file'])
    serial = lambda arm, key: _gz_read(root / HTTP / 'responses' / arm / (key + '.json.gz'))
    for (pid, arm), r in rows.items():
        if r['status'] != 200 or digest(load(pid, arm)) != r['response_sha256']:
            failures.append({'probe': pid, 'arm': arm, 'reason': 'status or response digest'})
    for arm in ('baseline', 'candidate'):
        body, own = load('P1', arm), serial(arm, rows[('P1', arm)]['key'])
        diff = first_diff(scrub(own), scrub(body))
        if diff:
            failures.append({'probe': 'P1', 'arm': arm, 'first_diff': diff})
    facts['P1_candidate_recovered'] = bool((load('P1', 'candidate').get('zero_target_release_v1') or {}).get('recovered'))
    if not facts['P1_candidate_recovered']:
        failures.append({'probe': 'P1', 'reason': 'candidate no-ROI bottles=all did not recover'})
    for pid in ('P2', 'P3'):
        base, cand = load(pid, 'baseline'), load(pid, 'candidate')
        diff = first_diff(normalized(base, False), normalized(cand, True))
        block = cand.get('zero_target_release_v1') or {}
        if diff or block.get('enabled') is not False or block.get('events') or 'zero_target_route' in cand:
            failures.append({'probe': pid, 'first_diff': diff, 'enabled': block.get('enabled'), 'events': len(block.get('events') or [])})
    for arm in ('baseline', 'candidate'):
        compact, own = load('P4', arm), serial(arm, rows[('P4', arm)]['key'])
        fields = ('decision', 'best_candidate', 'requires_target_selection', 'reasons')
        if any(compact.get(f) != own.get(f) for f in fields) or first_diff(scrub(compact['target_contract']),
                                                                           scrub(own['target_contract'])):
            failures.append({'probe': 'P4', 'arm': arm, 'reason': 'compact differs from its serial /v1 response'})
    return failures, facts


# ---- admission (computed gates; the decision is root's) ----

def _recovery_ok(base, cand, events):
    from rshb_vine.bottle_instances import iou
    target = (cand.get('targets') or [{}])[0]
    event = events[0] if len(events) == 1 else {}
    labels = event.get('canvas_labels') or []
    checks = {'one_event': len(events) == 1, 'event_recovered': event.get('recovered') is True
              and event.get('reason') == 'recovered', 'canvas_fired': event.get('canvas_fired') is True,
              'labels_nonempty': bool(labels),
              'iou_ge_0.9': bool(target.get('bbox')) and any(iou(target['bbox'], l['bbox']) >= 0.9 for l in labels),
              'baseline_zero': not base.get('targets'), 'candidate_one': len(cand.get('targets') or []) == 1,
              'not_physical': target.get('physical_bottle_localized') is False,
              'geometry_source': target.get('geometry_source') == 'canvas-closeup-label-route-v1',
              'eligibility_policy': (cand.get('canvas_closeup_route') or {}).get('eligibility_policy') == 'zero-target-release-v1-a'}
    return all(checks.values()), checks


def admission(root):
    protocol = verify(read_json(root / PROTOCOL))
    items = {i['key']: i for i in protocol['items']}
    history = _rows(root / RECORDS)
    rows = [r for r in history if r.get('event') == 'item']
    keys = [r['key'] for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(items) or len(rows) != len(items):
        raise SystemExit('HTTP records are not exactly one receipt per protocol item: %d/%d' % (len(rows), len(items)))
    errors = [r for r in history if r['event'] in ('error', 'budget_stop')]
    table, gates = [], {g: [] for g in ('T', 'G1', 'G2', 'G3', 'G4', 'G5')}
    gates['T'].extend(errors)
    serial = {}
    for r in rows:
        item = items[r['key']]
        bodies = {}
        for arm in ('baseline', 'candidate'):
            meta = r[arm]
            body = _gz_read(root / HTTP / 'responses' / arm / (r['key'] + '.json.gz'))
            if (meta['status'] != item['expected_http'] or digest(body) != meta['response_sha256']
                    or meta['request_sha256'] != item['image_sha256']):
                gates['T'].append({'key': r['key'], 'arm': arm, 'status': meta['status']})
            if item['expected_http'] == 200 and _response_profile(arm, body, protocol):
                gates['T'].append({'key': r['key'], 'arm': arm, 'profile': _response_profile(arm, body, protocol)})
            bodies[arm] = body
        base, cand = bodies['baseline'], bodies['candidate']
        if item['expected_http'] != 200:
            if base != cand:
                gates['T'].append({'key': r['key'], 'reason': 'expected_invalid bodies differ'})
            table.append({'key': r['key'], 'role': item['role'], 'query_id': item['query_id'], 'triggered': False,
                          'expected_invalid': True, 'status': [r['baseline']['status'], r['candidate']['status']],
                          'answer_equal': base == cand})
            continue
        serial[r['key']] = cand
        block = cand.get('zero_target_release_v1') or {}
        events = block.get('events') or []
        triggered = bool(events) or 'zero_target_route' in cand
        recovered = bool(block.get('recovered'))
        row = {'key': r['key'], 'role': item['role'], 'query_id': item['query_id'], 'triggered': triggered,
               'recovered': recovered, 'baseline_targets': len(base.get('targets') or []),
               'candidate_targets': len(cand.get('targets') or []),
               'baseline_s': r['baseline']['seconds'], 'candidate_s': r['candidate']['seconds'],
               'answer_equal': answer(base) == answer(cand)}
        if item['role'] == 'nonwine' and row['candidate_targets']:
            gates['G1'].append(r['key'])
        if len(events) > 1:
            gates['G4'].append(r['key'])
        if not recovered:
            diff = first_diff(normalized(base, False), normalized(cand, True, triggered))
            row['exact_parity'] = diff is None
            if diff is not None:
                row['first_diff'] = diff
                gates['G2'].append(r['key'])
        else:
            ok, checks = _recovery_ok(base, cand, events)
            row.update(recovery_contract=ok, recovery_checks=checks, candidate_best=cand.get('best_candidate'),
                       development_expected_slugs=item['development_expected_slugs'], event=events[0] if events else None)
            if not ok:
                gates['G3'].append(r['key'])
        if triggered:
            extra = row['candidate_s'] - row['baseline_s']
            first = events[0] if events else {}
            row.update(extra_s=extra, retry_ms=first.get('retry_ms'), route_total_ms=first.get('total_ms'),
                       reason=first.get('reason'))
            if extra > EXTRA_LIMIT_S or row['candidate_s'] > TOTAL_LIMIT_S:
                gates['G5'].append(r['key'])
        table.append(row)
    recovered = [t for t in table if t.get('recovered')]
    if not recovered:
        gates['G3'].append('no_recovered_row')
    concurrent = [c for c in _rows(root / CONCURRENCY) if c.get('event') == 'concurrent']
    g6 = []
    pairs = {(c['wave'], c['key']) for c in concurrent}
    if len(concurrent) != 9 or len(pairs) != 9:
        g6.append('not exactly 9 unique (wave, item) requests')
    accepted = {c['name'] for c in concurrent if c['status'] == 200}
    if accepted != {'recovery', 'nonwine', 'control'}:
        g6.append('accepted item types: %s' % sorted(accepted))
    for c in concurrent:
        if c['status'] == 200:
            same = serial.get(c['key']) is not None and first_diff(
                scrub(serial[c['key']]), scrub(_gz_read(root / c['file']))) is None
            if not same:
                g6.append({'wave': c['wave'], 'name': c['name'], 'reason': 'differs from serial'})
        elif not (c['status'] == 503 and 'busy' in str(c.get('detail'))):
            g6.append({'wave': c['wave'], 'name': c['name'], 'status': c['status']})
    gates['G6'] = g6
    gates['G7'], probe_facts = _probe_gate(root)
    triggered = [t for t in table if t.get('triggered')]
    return seal({'kind': 'zero-target-release-v1-computed-admission', 'protocol': protocol['checksum'],
                 'records_sha256': sha256(root / RECORDS),
                 'concurrency_sha256': sha256(root / CONCURRENCY) if concurrent else None,
                 'gates': {g: {'ok': not v, 'failures': v} for g, v in gates.items()},
                 'counts': {'rows': len(table), 'triggered': len(triggered), 'recovered': len(recovered),
                            'answer_equal_untriggered': sum(1 for t in table if not t['triggered'] and t['answer_equal']),
                            'untriggered': sum(1 for t in table if not t['triggered']),
                            'expected_invalid': sum(1 for t in table if t.get('expected_invalid'))},
                 'triggered_rows': triggered, 'rows': table, 'probe_facts': probe_facts,
                 'probes_sha256': sha256(root / PROBES) if (root / PROBES).is_file() else None,
                 'concurrency': {'requests': len(concurrent), 'ok_200': sum(1 for c in concurrent if c['status'] == 200),
                                 'busy_503': sum(1 for c in concurrent if c['status'] == 503)},
                 'decision': 'root_required', 'calibrated': False, 'independent_accuracy_claim': False})


# ---- stage / activate / rollback ----

def _atomic_bytes(path, data):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_once(root, path, doc):
    target = root / path
    if target.exists():
        if verify(read_json(target)) != doc:
            raise ValueError(path + ' exists with different content; staged descriptors are immutable')
        return
    write_json(target, doc)


def _copy_once(root, data, path):
    target = root / path
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(path + ' exists with different bytes; staged copies are immutable')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(target, data)


def _body(doc, *drop):
    return {k: v for k, v in doc.items() if k not in ('checksum', *drop)}


def _live(root):
    return [sha256(root / p) for p in (*CODE, CURRENT)]


def _check_before(root, decision_checksum):
    if _live(root) != V2_STATE:
        raise ValueError('Stage requires the live repair_v2_active state (F7, R3, C7, pointer fc55103d)')
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    for path in (*COMPOSITES, SYSTEMIC7, TARGET7, GEOMETRY7, BASE7, *MIGRATIONS_OLD, *GALLERY_MIGRATIONS_OLD):
        verify(read_json(root / path))
    v2 = verify(read_json(root / V2_RELEASE))
    if v2['checksum'] != V2_CHECKSUM or sha256(root / V2_RELEASE) != V2_POINTER_SHA256:
        raise ValueError('Frozen repair-v2 release 99451df0 differs')
    for path, expected in verify(read_json(root / V2_MANIFEST))['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Repair-v2 release file changed since its stage: ' + path)
    from rshb_vine.zero_target_release_v1 import release as ZR
    descriptor = ZR.load_candidate_descriptor(root)
    decision = verify(read_json(root / DECISION))
    if (decision['checksum'] != decision_checksum or decision.get('kind') != 'zero-target-release-v1-root-decision'
            or decision.get('decision') != 'admitted_for_stage' or decision.get('candidate_descriptor') != descriptor['checksum']):
        raise ValueError('Root decision does not admit this candidate for stage')
    for draft in DRAFTS.values():
        if not (root / draft).is_file():
            raise ValueError('Missing staged dispatch draft: ' + draft)
    return v2, descriptor, decision


def _chain(root, source, name, parent_path, parent_doc, what):
    doc = _body(source, 'supersedes', 'rollback')
    doc.update({'name': name, 'parent_profile': parent_path, 'parent_profile_checksum': parent_doc['checksum'],
                'parent_profile_sha256': sha256(root / parent_path),
                'supersedes': {'path': what[0], 'sha256': sha256(root / what[0]), 'checksum': source['checksum'],
                               'same': what[1]},
                'release_status': 'staged_pending_activation', 'rollback': ROLLBACK})
    return seal(doc)


def _snapshot(root):
    files = {}
    for path, sha in zip((*CODE, CURRENT), V2_STATE):
        copy = '%s/%s.%s' % (RESTORE_DIR, Path(path).name, sha[:16])
        _copy_once(root, (root / path).read_bytes(), copy)
        if sha256(root / copy) != sha:
            raise ValueError('Restoration copy differs from live bytes: ' + path)
        files[path] = {'sha256': sha, 'copy': copy}
    restoration = seal({'kind': 'recognition-release-restoration-v1', 'state': 'repair_v2_active', 'files': files,
                        'profile_checksum': V2_CHECKSUM, 'profile_path': V2_RELEASE})
    _write_once(root, RESTORATION, restoration)
    return restoration


def stage(root, decision_checksum):
    from rshb_vine.catalog_training_evaluation_v2 import pins
    from rshb_vine.zero_target_release_v1 import release as ZR
    from rshb_vine.zero_target_release_v1 import route as Z
    v2, descriptor, decision = _check_before(root, decision_checksum)
    restoration = _snapshot(root)
    staged = {}
    for logical, draft in DRAFTS.items():
        sha = sha256(root / draft)
        if sha in (F7, C7):
            raise ValueError('Draft ' + draft + ' equals the live repair-v2 dispatch')
        path = ARCHIVE_NAMES[logical] % sha[:8]
        _copy_once(root, (root / draft).read_bytes(), path)
        staged[logical] = {'sha256': sha, 'archive': path}
    F8, C8 = staged[FACTORY]['sha256'], staged[RECEIPTS]['sha256']

    dispatch7 = verify(read_json(root / DISPATCH7))
    if dispatch7['pins_sha256'].get(FACTORY) != F7 or dispatch7['pins_sha256'].get(RECOGNIZE) != R3:
        raise ValueError('coherent-v1-dispatch7 does not pin the expected dispatch')
    dispatch8 = _body(dispatch7, 'migrated_from')
    dispatch8.update({
        'name': 'coherent-v1-dispatch8', 'version': 8, 'pins_sha256': dict(dispatch7['pins_sha256'], **{FACTORY: F8}),
        'migrated_from': {'path': DISPATCH7, 'sha256': sha256(root / DISPATCH7), 'checksum': dispatch7['checksum'],
                          'changed_pins': {FACTORY: {'from': F7, 'to': F8}},
                          'behavior': 'identical: factory v8 adds the dispatch8 B3-only base and zero-target release kinds'},
        'release_status': 'dispatch_migration_staged'})
    dispatch8 = seal(dispatch8)
    _write_once(root, DISPATCH8, dispatch8)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory F7 -> F8)'
    systemic8 = _chain(root, verify(read_json(root / SYSTEMIC7)), 'systemic-ranking-v2-dispatch8', DISPATCH8, dispatch8,
                       (SYSTEMIC7, same))
    _write_once(root, SYSTEMIC8, systemic8)
    target8 = _chain(root, verify(read_json(root / TARGET7)), 'target-contract-v2-dispatch8', SYSTEMIC8, systemic8,
                     (TARGET7, same))
    _write_once(root, TARGET8, target8)
    geometry8 = _chain(root, verify(read_json(root / GEOMETRY7)), 'roskachestvo-geometry-v2-dispatch8', TARGET8, target8,
                       (GEOMETRY7, same))
    _write_once(root, GEOMETRY8, geometry8)

    old = verify(read_json(root / GALLERY_MIGRATION_OLD))
    if old['expected_sha256'] != F4 or old['replacement_sha256'] != F7:
        raise ValueError('Gallery migration v3 is not F4 -> F7')
    gallery_migration = seal(dict(_body(old), replacement_sha256=F8, version=4,
                                  supersedes={'path': GALLERY_MIGRATION_OLD, 'checksum': old['checksum']}))
    _write_once(root, GALLERY_MIGRATION, gallery_migration)

    base7 = verify(read_json(root / BASE7))
    own = {p: sha256(root / p) for p in (*ZR.SOURCES, *ZR.FROZEN_ROUTE_SOURCES, GALLERY_MIGRATION, BASE7)}
    base = _body(base7, 'supersedes', 'rollback')
    base.update({'kind': ZR.BASE_KIND, 'name': 'b3-only-v1-dispatch8',
                 'parent_profile': GEOMETRY8, 'parent_profile_sha256': sha256(root / GEOMETRY8),
                 'parent_profile_checksum': geometry8['checksum'], 'target_profile_checksum': target8['checksum'],
                 'gallery_code_migration_checksum': gallery_migration['checksum'],
                 'pins_sha256': dict(base7['pins_sha256'], **own),
                 'supersedes': {'path': BASE7, 'sha256': sha256(root / BASE7), 'checksum': base7['checksum'],
                                'same': 'identity fields and equivalence to 83e97cb3; parent chain differs only in '
                                        'dispatch pins F7 -> F8 and gallery migration v3 -> v4'},
                 'release_status': 'development_release', 'rollback': ROLLBACK})
    base = seal(base)
    _write_once(root, BASE8, base)

    release_pins = dict(v2['pins_sha256'])
    release_pins.update({p: sha256(root / p) for p in (*ZR.SOURCES, *ZR.FROZEN_ROUTE_SOURCES, BASE8, V2_RELEASE)})
    release = _body(v2, 'supersedes', 'rollback', 'predecessor_release')
    release.update({
        'kind': ZR.KIND, 'name': 'zero-target-release-v1',
        'parent_profile': BASE8, 'parent_profile_sha256': sha256(root / BASE8), 'parent_profile_checksum': base['checksum'],
        'composition': v2['composition'] + ' -> ' + Z.POLICY + ' (canvas close-up route adapter)',
        'components': dict(v2['components'], zero_target_route=Z.POLICY),
        'zero_target': {'rule': Z.RULE, 'sources_sha256': ZR.sources_sha(root),
                        'candidate_descriptor': descriptor['checksum'], 'root_decision': decision['checksum'],
                        'protocol': decision.get('protocol')},
        'predecessor_release': {'path': V2_RELEASE, 'sha256': V2_POINTER_SHA256, 'checksum': V2_CHECKSUM},
        'pins_sha256': release_pins, 'release_status': 'development_release', 'rollback': ROLLBACK})
    release = seal(release)
    _write_once(root, RELEASE, release)

    entries = []
    for profile, expected in zip(COMPOSITES, (F1, F2, F3, F4, F5, F6, F7)):
        doc = verify(read_json(root / profile))
        if doc['pins_sha256'].get(FACTORY) != expected:
            raise ValueError('Unexpected factory pin: ' + profile)
        entries.append({'profile': profile, 'profile_sha256': sha256(root / profile), 'profile_checksum': doc['checksum'],
                        'logical_path': FACTORY, 'expected_sha256': expected, 'archive': ARCHIVE[expected],
                        'replacement_sha256': F8})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 7, 'dispatch_paths': [FACTORY],
                      'scope': 'factory 9d9d/be2c/08b4/f16a/0c3c/6d58/2ab8 -> F8; recognize.py entries of v2..v4 remain valid (6faa live)',
                      'entries': entries})
    _write_once(root, MIGRATION, migration)

    archives = {**ARCHIVE, F8: staged[FACTORY]['archive'], C8: staged[RECEIPTS]['archive']}
    files = [*ZR.SOURCES, *ZR.FROZEN_ROUTE_SOURCES, DECISION, ZR.CANDIDATE_DESCRIPTOR, V2_RELEASE, V2_MANIFEST,
             *COMPOSITES, SYSTEMIC7, TARGET7, GEOMETRY7, BASE7, DISPATCH8, SYSTEMIC8, TARGET8, GEOMETRY8, BASE8, RELEASE,
             MIGRATION, *MIGRATIONS_OLD, GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *release_pins,
             *archives.values(), RESTORATION, *(f['copy'] for f in restoration['files'].values())]
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'zero-target-release-v1',
        'files_sha256_at_stage': {p: sha256(root / p) for p in dict.fromkeys(files)},
        'staged_dispatch': {v['archive']: {'logical_path': k, 'sha256': v['sha256'], 'draft': DRAFTS[k]}
                            for k, v in staged.items()},
        'profiles': {'release': {'path': RELEASE, 'checksum': release['checksum']},
                     'base_dispatch8': {'path': BASE8, 'checksum': base['checksum']},
                     'geometry_dispatch8': {'path': GEOMETRY8, 'checksum': geometry8['checksum']},
                     'target_dispatch8': {'path': TARGET8, 'checksum': target8['checksum']},
                     'systemic_dispatch8': {'path': SYSTEMIC8, 'checksum': systemic8['checksum']},
                     'dispatch8': {'path': DISPATCH8, 'checksum': dispatch8['checksum']},
                     'repair_v2_release': {'path': V2_RELEASE, 'checksum': V2_CHECKSUM}},
        'model_checksum': release['ablated_ranker_checksum'],
        'states': {'repair_v2_active': V2_STATE, 'zero_target_active': [F8, R3, C8, sha256(root / RELEASE)]},
        'sources': {'repair_v2_active': {p: f['copy'] for p, f in restoration['files'].items()},
                    'zero_target_active': {FACTORY: archives[F8], RECOGNIZE: restoration['files'][RECOGNIZE]['copy'],
                                           RECEIPTS: archives[C8], CURRENT: RELEASE}},
        'restoration': {'path': RESTORATION, 'checksum': restoration['checksum']},
        'code_sha_at_stage': pins.code_sha(),
        'historical_reproduction': 'repair-v2 99451df0 and older profiles bind through migration v7 without loading '
                                   'models; rollback restores the sealed 2ab8/6faa/de77 bytes and pointer fc55103d, '
                                   'after which scripts/recognition_repair_release_v2.py manages older states',
        'calibrated': False, 'weights_changed': False, 'fit_run': False, 'release_status': 'staged_pending_activation'})
    _write_once(root, MANIFEST, manifest)
    return manifest


def _manifest(root):
    manifest = verify(read_json(root / MANIFEST))
    for path, expected in manifest['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Release file changed since stage: ' + path)
    _sources(root, manifest)
    return manifest


def _sources(root, manifest):
    for name, paths in manifest['sources'].items():
        for path, sha in zip((*CODE, CURRENT), manifest['states'][name]):
            if sha256(root / paths[path]) != sha:
                raise ValueError('Switch source bytes differ: ' + paths[path])


def _restore_manifest(root):
    """Rollback/recover need only the sealed state table, the restoration snapshot and the staged sources."""
    manifest = verify(read_json(root / MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    if (manifest['states']['repair_v2_active'] != V2_STATE or restoration['checksum'] != manifest['restoration']['checksum']
            or [restoration['files'][p]['sha256'] for p in (*CODE, CURRENT)] != V2_STATE
            or manifest['sources']['repair_v2_active'] != {p: f['copy'] for p, f in restoration['files'].items()}):
        raise ValueError('Sealed rollback state differs from F7/R3/C7/fc55103d')
    for path, sha in zip((*CODE, CURRENT), V2_STATE):
        if sha256(root / restoration['files'][path]['copy']) != sha:
            raise ValueError('Restoration copy changed: ' + path)
    return manifest


def state(root, manifest):
    live = _live(root)
    return next((name for name, pair in manifest['states'].items() if pair == live), 'unknown')


def _pending(root):
    pending = None
    for event in _rows(root / JOURNAL):
        if event['event'] == 'switch_started':
            pending = event
        elif event['event'] in ('switch_done', 'recovered'):
            pending = None
    return pending


def _port_free(port):
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(('127.0.0.1', port)) != 0


def _journal(root, event):
    _append(root / JOURNAL, dict(event, ts=time.time()))


def _describe(root):
    out = subprocess.run([sys.executable, str(root / RECOGNIZE), 'describe'], cwd=root, capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout)


def _source(root, manifest, path, sha):
    for name, pair in manifest['states'].items():
        for p, d in zip((*CODE, CURRENT), pair):
            if p == path and d == sha:
                source = root / manifest['sources'][name][path]
                if sha256(source) != sha:
                    raise ValueError('Switch source bytes differ: ' + str(source))
                return source.read_bytes()
    raise ValueError('No sealed source carries ' + path + ' ' + sha)


def _apply(root, manifest, plan):
    for path, (_, to) in plan.items():
        if sha256(root / path) != to:
            _atomic_bytes(root / path, _source(root, manifest, path, to))


def _switch(root, manifest, target, port):
    if not _port_free(port):
        raise SystemExit('Port %d is serving; stop the service before switching code+profile' % port)
    if _pending(root):
        raise SystemExit('An interrupted switch is recorded; run recover first')
    before = state(root, manifest)
    if before == 'unknown':
        raise SystemExit('Unknown code/profile state; inspect before switching')
    if before == target:
        raise SystemExit('Already in state ' + target)
    plan = {p: [a, b] for p, a, b in zip((*CODE, CURRENT), manifest['states'][before], manifest['states'][target])}
    _journal(root, {'event': 'switch_started', 'from': before, 'to': target, 'plan': plan, 'manifest': manifest['checksum']})
    _apply(root, manifest, plan)
    after = state(root, manifest)
    if after != target:
        raise ValueError('Switch verification failed: ' + after + '; run recover')
    described = _describe(root)
    expected = manifest['profiles']['release' if target == 'zero_target_active' else 'repair_v2_release']['checksum']
    if described['profile_checksum'] != expected:
        raise ValueError('Pointer describes %s, expected %s; run recover' % (described['profile_checksum'], expected))
    _journal(root, {'event': 'switch_done', 'state': after, 'profile_checksum': described['profile_checksum']})
    return after, described['profile_checksum']


def recover(root, manifest, port):
    if not _port_free(port):
        raise SystemExit('Port %d is serving; stop the service before recovery' % port)
    pending = _pending(root)
    if pending is None:
        raise SystemExit('No interrupted switch recorded')
    plan = pending['plan']
    for path, pair in plan.items():
        if sha256(root / path) not in pair:
            raise SystemExit('Unrecognized bytes outside the recorded switch: ' + path)
    _apply(root, manifest, {p: [b, a] for p, (a, b) in plan.items()})
    after = state(root, manifest)
    if after != pending['from']:
        raise ValueError('Recovery verification failed: ' + after)
    _journal(root, {'event': 'recovered', 'state': after, 'abandoned': pending['to']})
    return after


# ---- candidate ----

def serve(root, port):
    from rshb_vine.recognition_factory import create_app
    from rshb_vine.zero_target_release_v1 import release as ZR
    if port in ZR.FORBIDDEN_PORTS:
        raise SystemExit('Port %d belongs to the live service' % port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = ZR.ZeroTargetCandidate(root)
        import uvicorn
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'describe', 'serve', 'replay', 'census', 'protocol', 'run',
                                           'concurrency', 'probes', 'admission', 'stage', 'status', 'verify', 'activate',
                                           'rollback', 'recover'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int)
    parser.add_argument('--baseline-port', type=int, default=8175)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--decision-checksum', help='stage: checksum of ' + DECISION)
    parser.add_argument('--confirm', help='activate: checksum of the staged release profile')
    args = parser.parse_args()
    root = args.root.resolve()
    from rshb_vine.zero_target_release_v1 import release as ZR
    if args.action == 'freeze':
        descriptor = ZR.freeze_candidate(root)
        _write_once(root, ZR.CANDIDATE_DESCRIPTOR, descriptor)
        print(json.dumps({'descriptor': ZR.CANDIDATE_DESCRIPTOR, 'checksum': descriptor['checksum']}))
        return
    if args.action == 'describe':
        from rshb_vine.recognition_factory import describe
        descriptor = ZR.load_candidate_descriptor(root)
        print(json.dumps({'candidate_descriptor': descriptor, 'parent': describe(root, ZR.PARENT_RELEASE),
                          'staged_release': (root / MANIFEST).is_file(), 'models_loaded': False},
                         ensure_ascii=False, indent=2))
        return
    if args.action == 'serve':
        serve(root, args.port or ZR.CANDIDATE_PORT)
        return
    if args.action == 'replay':
        if args.input is None or args.output is None or args.output.exists():
            raise SystemExit('replay needs --input and a new --output')
        data = args.input.read_bytes()
        pipeline = ZR.ZeroTargetCandidate(root)
        started = time.perf_counter()
        result = pipeline.recognize(data)
        write_json(args.output, seal({'kind': 'zero-target-release-v1-candidate-replay', 'image_path': str(args.input),
                                      'request_sha256': hashlib.sha256(data).hexdigest(),
                                      'runtime_checksum': pipeline.manifest['checksum'], 'result': result,
                                      'seconds': time.perf_counter() - started, 'quality_evaluation': False}))
        print(json.dumps({'receipt': str(args.output), 'decision': result['decision'],
                          'zero_target': {k: v for k, v in result.get('zero_target_release_v1', {}).items() if k != 'events'}}))
        return
    if args.action == 'census':
        doc = census(root)
        _write_once(root, CENSUS, doc)
        print(json.dumps({k: doc[k] for k in ('rows', 'zero_target_200', 'branch_a_eligible', 'branch_a_eligible_by_proposals',
                                              'branch_a_eligible_by_set', 'checksum')}))
        return
    if args.action == 'protocol':
        doc = build_protocol(root)
        _write_once(root, PROTOCOL, doc)
        print(json.dumps({'protocol': PROTOCOL, 'checksum': doc['checksum'], 'scope_counts': doc['scope_counts']}))
        return
    if args.action == 'run':
        print(json.dumps({'items_done': run_http(root, args.baseline_port, args.port or ZR.CANDIDATE_PORT)}))
        return
    if args.action == 'concurrency':
        rows = run_concurrency(root, args.port or ZR.CANDIDATE_PORT)
        print(json.dumps({'requests': len(rows), 'statuses': [r['status'] for r in rows]}))
        return
    if args.action == 'probes':
        rows = run_probes(root, args.baseline_port, args.port or ZR.CANDIDATE_PORT)
        print(json.dumps({'requests': len(rows), 'statuses': [r['status'] for r in rows]}))
        return
    if args.action == 'admission':
        doc = admission(root)
        write_json(root / ADMISSION, doc, replace=True)
        print(json.dumps({'gates': {g: v['ok'] for g, v in doc['gates'].items()}, 'counts': doc['counts'],
                          'checksum': doc['checksum']}))
        return
    port = args.port or 8175
    if args.action == 'stage':
        if not args.decision_checksum:
            raise SystemExit('stage needs --decision-checksum')
        manifest = stage(root, args.decision_checksum)
        print(json.dumps({'manifest': manifest['checksum'], 'state': state(root, manifest),
                          'release': manifest['profiles']['release']}))
        return
    if not (root / MANIFEST).is_file():
        print(json.dumps({'state': 'repair_v2_active' if _live(root) == V2_STATE else 'unknown', 'staged': False,
                          'live': dict(zip((*CODE, CURRENT), _live(root)))}))
        raise SystemExit(0 if args.action == 'status' else 'Release not staged: ' + MANIFEST)
    manifest = _manifest(root) if args.action in ('verify', 'activate') else _restore_manifest(root)
    if args.action in ('status', 'verify'):
        current, pending = state(root, manifest), _pending(root)
        print(json.dumps({'state': current, 'manifest': manifest['checksum'], 'pending_switch': pending,
                          'live': dict(zip((*CODE, CURRENT), _live(root)))}))
        if args.action == 'verify' and (current == 'unknown' or pending):
            raise SystemExit('verify failed')
        return
    if args.action == 'recover':
        print(json.dumps({'state': recover(root, manifest, port)}))
        return
    if args.action == 'activate':
        if args.confirm != manifest['profiles']['release']['checksum']:
            raise SystemExit('--confirm must name the staged release profile checksum')
        print(json.dumps(_switch(root, manifest, 'zero_target_active', port)))
        return
    print(json.dumps(_switch(root, manifest, 'repair_v2_active', port)))


if __name__ == '__main__':
    main()
