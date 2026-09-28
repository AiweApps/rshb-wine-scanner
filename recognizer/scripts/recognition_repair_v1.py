"""Recognition repair v1: combined candidate (reference-conflict filter + OCR injection/guard v2 + front-label 577a575e).

describe -- profile, components and base release description (no models loaded)
freeze   -- seal the candidate-v2 profile (pins own/composed sources and data inputs)
plan     -- sealed paired HTTP plan v2 (<=24 rows): filter-affected, OCR/guard fixes, past guard regressions,
            Aratti, Markotkh, negative, ROI, crowded and unaffected controls; admitted open sources only
serve    -- loopback candidate service (default 8189); 8175/8188 and the current pointer are never touched
http     -- one append-only paired pass: health identity first, per-row response identity, sealed budget, resume
            only on a ledger of the same plan and profiles
History: the first profile (combined/candidate) and 8-row plan (combined/http) are kept unchanged.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/recognition-repair-20260926/combined'
PLAN = OUT / 'http-v3/plan.json'
FAILED_PLAN = 'runs/recognition-repair-20260926/combined/http-v2/plan.json'
ROSTER = 'runs/system-capability-snapshot-v1/protocol/roster.json'
MAIN183 = 'runs/b3-only-integration-v1/candidate-http/report.json'
HISTORY_PLAN = 'runs/recognition-repair-20260926/combined/http/plan.json'
SAVED = 'runs/system-capability-snapshot-v1/receipts/responses'
BUDGET = 48
BASELINE_CHECKSUM = '83e97cb3f6e8d1a89451b1ffcdd8de4e20f6417706b488f1e4d426c538cafbbb'
ROWS = (
    *(('filter_affected', k) for k in ('real_new52:S1000-f2a0c2d83f1c10', 'real_open38:internet-v2-77501a85ee384fd94f89',
                                       'real_organizer:organizer-087', 'real_whole71:S1000-cb5f07d983f04d',
                                       'real_whole71:S1000-f56f3fbb15954c')),
    ('ocr_injection_fix', 'real_open38:internet-v2-26c1090da01c0291dcd2'),
    ('ocr_injection_recall_only', 'real_whole71:S1000-d44a878e9e2d9c'),
    ('guard_fix', 'real_open38:internet-4959e59964129e2c4731'),
    ('guard_fix', 'real_open38:internet-v2-d6941394b665ae01af88'),
    ('guard_fix', 'real_whole71:S1000-08bf52eac64b28'),
    ('guard_fix', 'synthetic_hard36:09-angle'),
    ('guard_v1_past_regression', 'W0072'),
    ('guard_v1_past_regression', 'real_whole71:S1000-cbb2c591a740f2'),
    ('aratti_crop', 'real_organizer:organizer-022'),
    ('aratti_crop', 'real_organizer:organizer-024'),
    ('markotkh', 'real_whole71:S1000-5b3f0369aaa994'),
    ('blend_explanation_multi_target', 'real_organizer:organizer-010'),
    ('negative', 'negative_nonwine:nonwine:01af816ec94899dd'),
    ('supplied_roi', 'real_supplied_roi:real_open426:rshb-photo-Q0112-target-0'),
    ('crowded_multi_target', 'synthetic_hard36:09-crowded'),
    ('unaffected_control', 'real_open38:internet-4811d4ab68b321ea3ff1'),
    ('unaffected_control', 'real_whole71:S1000-01d59ba0671138'),
)
CRITERIA = {
    'filter_affected': 'card A not first through its contradicted reference; B or explicit A/B ambiguity; slug/probability None',
    'ocr_injection_fix': 'GT product first (replay: sary-pandas -> Chernyy Polkovnik)',
    'ocr_injection_recall_only': 'GT card present among candidates; top-1 not required (replay rank 3)',
    'guard_fix': 'GT product first; guard trace blocked=true with the replay reason',
    'guard_v1_past_regression': 'GT product first as in baseline; guard not blocked',
    'aratti_crop': 'crop repair traces published; GT rank not worse than baseline (022 selector issue may remain)',
    'markotkh': 'no claim; OCR unclear; report ranks and traces only',
    'blend_explanation_multi_target': 'no rubin-golodrigi injection (explained_by_pool_blend); targets unchanged',
    'negative': 'no SKU targets',
    'supplied_roi': 'same addressed target as baseline; no new targets',
    'crowded_multi_target': 'target count/geometry unchanged; changed cards per bottle listed, not scored',
    'unaffected_control': 'targets and top-1 identical to baseline',
    'v2_targets': 'compact /v2/targets target_contract.profile_checksum names the served profile',
    'all': 'HTTP status and target count equal; public slug/probability None; every changed selection reviewed',
}
V2_TARGETS_KEY = 'real_organizer:organizer-010'


def write_new(path, obj):
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def _sha(path):
    from rshb_vine.io import sha256
    return sha256(path)


def cmd_describe(a):
    from rshb_vine.recognition_repair_v1 import runtime
    print(json.dumps(runtime.describe(ROOT), ensure_ascii=False, indent=1))


def cmd_freeze(a):
    from rshb_vine.io import seal, write_json
    from rshb_vine.recognition_repair_v1 import runtime
    path = ROOT / runtime.PROFILE
    if path.exists():
        raise SystemExit('profile exists; immutable profile preserved')
    profile = seal(runtime.freeze_body(ROOT))
    write_json(path, profile)
    runtime.load_profile(ROOT)
    print(json.dumps({'profile': runtime.PROFILE, 'checksum': profile['checksum'], 'components': profile['components']}))


def cmd_plan(a):
    from rshb_vine.io import read_json, seal, verify
    from rshb_vine.recognition_repair_v1 import runtime
    profile = runtime.load_profile(ROOT)
    roster = read_json(ROOT / ROSTER)
    snap = {r['request_key']: r for r in roster['rows']}
    main = {r['id']: r for r in verify(read_json(ROOT / MAIN183))['rows']}
    rows = []
    for role, key in ROWS:
        if key in main:
            r = main[key]
            if r['split'] != 'train':
                raise SystemExit('only web232 train rows are admitted: ' + key)
            entry = {'key': key, 'cohort': 'web232_train', 'path': r['path'], 'sha256': r['sha256'],
                     'acceptable_slugs': None, 'gt_products': r['gt'], 'gt_level': 'product_admitted', 'roi': None}
        else:
            r = snap[key]
            entry = {'key': key, 'cohort': r['cohort'], 'path': r['path'], 'sha256': r['sha256'],
                     'acceptable_slugs': r['acceptable_slugs'], 'gt_level': r['gt_level'], 'roi': r['roi']}
        if _sha(ROOT / entry['path']) != entry['sha256']:
            raise SystemExit('query SHA differs: ' + key)
        rows.append(dict(entry, role=role, bottles=None, criterion=CRITERIA[role]))
    probe = next(dict(r, role='v2_targets', criterion=CRITERIA['v2_targets'], route='/v2/targets')
                 for r in rows if r['key'] == V2_TARGETS_KEY)
    total = 2 * (len(rows) + 1)
    if len({r['key'] for r in rows}) != len(rows) or total > BUDGET:
        raise SystemExit('plan rows duplicate or exceed the request budget')
    body = {'kind': 'recognition-repair-v1-http-plan-v2', 'candidate_profile': runtime.PROFILE,
            'candidate_profile_checksum': profile['checksum'], 'baseline_profile_checksum': BASELINE_CHECKSUM,
            'history_plan': {'path': HISTORY_PLAN, 'sha256': _sha(ROOT / HISTORY_PLAN)},
            'failed_plan': {'path': FAILED_PLAN, 'sha256': _sha(ROOT / FAILED_PLAN),
                            'failure': 'candidate-v2 0afdf67f answered 503 (missing selection.model); kept as receipt'},
            'sources_sha256': {ROSTER: _sha(ROOT / ROSTER), MAIN183: _sha(ROOT / MAIN183)},
            'selection': 'fixed role list from saved replay/crop/filter work; validation46, locked, protected and '
                         'closed rows excluded; fixed before any candidate-v2 request',
            'baseline': 'live 8175 (release 83e97cb3) on the same bytes, same session',
            'candidate': 'recognition-repair-v1 candidate-v3 on 8189', 'requests_total': total,
            'budget_total': BUDGET, 'criteria': CRITERIA, 'rows': rows, 'v2_targets_probe': probe}
    write_new(PLAN, seal(body))
    print(json.dumps({'plan': str(PLAN.relative_to(ROOT)), 'rows': len(rows), 'requests': total,
                      'candidate_profile_checksum': profile['checksum']}))


def cmd_serve(a):
    import socket
    import uvicorn
    from rshb_vine.recognition_repair_v1 import runtime
    from rshb_vine.target_contract_v2.runtime import create_app
    if a.port in runtime.FORBIDDEN_PORTS:
        raise SystemExit('%d is reserved (current release / crop candidate); the combined candidate never binds it' % a.port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', a.port))
        listener.listen(128)
        pipeline = runtime.RecognitionRepairV1(ROOT)
        log = PLAN.parent / 'serve-errors.log'
        recognize = pipeline.recognize

        def logged(*args, **kwargs):
            try:
                return recognize(*args, **kwargs)
            except Exception:
                import traceback
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, 'a') as stream:
                    stream.write(traceback.format_exc() + '\n')
                raise
        pipeline.recognize = logged
        print(json.dumps({'runtime': pipeline.manifest['checksum'], 'installation': pipeline.installation},
                         ensure_ascii=False), flush=True)
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=a.port)).run(sockets=[listener])
    finally:
        listener.close()


def _identity(side, body, expected, route):
    """Response names the pinned profile; candidate traces are complete and bound to this response's targets."""
    contract = body.get('target_contract') or {}
    if contract.get('profile_checksum') != expected:
        return False
    if route == '/v2/targets':
        return True
    if side == 'baseline':
        return (body.get('b3_only_v1') or {}).get('profile_checksum') == expected
    block = body.get('recognition_repair_v1') or {}
    if block.get('profile_checksum') != expected:
        return False
    targets = sorted(str(t['instance_id']) for t in (body.get('product_identity_evidence') or {}).get('targets', []))
    records = sorted(str(r['instance_id']) for r in (body.get('systemic_ranking_v2') or {}).get('targets', []))
    injected = sorted(str(r['instance_id']) for r in block.get('ocr_candidate_injection', []))
    guarded = sorted(str(r['instance_id']) for r in block.get('discriminator_guard', []))
    return targets == records == injected == guarded


def cmd_http(a):
    import time
    import requests
    from rshb_vine.io import read_json, sha256, verify
    from rshb_vine.recognition_repair_v1 import runtime
    plan = verify(read_json(PLAN))
    profile = runtime.load_profile(ROOT)
    if (plan['kind'] != 'recognition-repair-v1-http-plan-v2' or plan['candidate_profile_checksum'] != profile['checksum']
            or plan['baseline_profile_checksum'] != BASELINE_CHECKSUM or plan['requests_total'] != 2 * (len(plan['rows']) + 1)
            or plan['requests_total'] > min(plan['budget_total'], BUDGET)):
        raise SystemExit('plan seal, profile or budget does not match this runner')
    expected = {'baseline': BASELINE_CHECKSUM, 'candidate': profile['checksum']}
    out = PLAN.parent
    ledger = out / 'ledger.jsonl'
    rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    for r in rows:
        if r['plan_checksum'] != plan['checksum'] or r['profile_checksum'] != expected[r['side']] or not r['identity_ok']:
            raise SystemExit('existing ledger belongs to another plan/profile or has an identity failure; not resuming')
    done = {(r['side'], r['route'], r['key']) for r in rows}
    if len(done) != len(rows) or len(rows) > plan['requests_total']:
        raise SystemExit('ledger has duplicate rows or exceeds the plan budget')
    for side, port in (('baseline', a.baseline_port), ('candidate', a.port)):
        url = 'http://127.0.0.1:%d' % port
        health = requests.get(url + '/health/ready', timeout=10).json()
        (out / 'health').mkdir(parents=True, exist_ok=True)
        with open(out / 'health' / (side + '.jsonl'), 'a') as f:
            f.write(json.dumps(dict(health, at=time.time(), port=port)) + '\n')
        if health.get('runtime_descriptor_checksum') != expected[side]:
            raise SystemExit(side + ' health runtime_descriptor_checksum differs from ' + expected[side])
        (out / 'responses' / side).mkdir(parents=True, exist_ok=True)
        for row in [*plan['rows'], plan['v2_targets_probe']]:
            route = row.get('route', '/v1/recognize')
            if (side, route, row['key']) in done:
                continue
            path = ROOT / row['path']
            if sha256(path) != row['sha256']:
                raise SystemExit('query SHA changed: ' + row['path'])
            form = {}
            if row['roi']:
                form['target_roi'] = json.dumps(row['roi'])
            if row.get('bottles'):
                form['bottles'] = row['bottles']
            started = time.time()
            r = requests.post(url + route, files={'image': (path.name, path.read_bytes(), 'application/octet-stream')},
                              data=form, timeout=300)
            body = r.json()
            name = ('v2targets_' if route == '/v2/targets' else '') + row['key'].replace(':', '_').replace('/', '_') + '.json.gz'
            with gzip.open(out / 'responses' / side / name, 'wt') as f:
                json.dump({'http_status': r.status_code, 'body': body}, f, ensure_ascii=False)
            ok = r.status_code != 200 or body.get('decision') == 'invalid_image' or _identity(side, body, expected[side], route)
            rec = {'side': side, 'route': route, 'key': row['key'], 'role': row['role'], 'http_status': r.status_code,
                   'seconds': time.time() - started, 'file': f'responses/{side}/{name}',
                   'response_sha256': sha256(out / 'responses' / side / name), 'image_sha256': row['sha256'],
                   'plan_checksum': plan['checksum'], 'profile_checksum': expected[side], 'identity_ok': ok,
                   'health_snapshot': health.get('snapshot')}
            with open(ledger, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(side, rec['key'], rec['http_status'], '%.1fs' % rec['seconds'], flush=True)
            if not ok:
                raise SystemExit('response identity differs from the pinned profile: ' + side + ' ' + row['key'])
            if r.status_code != 200:
                raise SystemExit('stopping at the first non-200 response: %s %s %d' % (side, row['key'], r.status_code))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('describe').set_defaults(fn=cmd_describe)
    sub.add_parser('freeze').set_defaults(fn=cmd_freeze)
    sub.add_parser('plan').set_defaults(fn=cmd_plan)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=8189)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser('http')
    s.add_argument('--port', type=int, default=8189)
    s.add_argument('--baseline-port', type=int, default=8175)
    s.set_defaults(fn=cmd_http)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
