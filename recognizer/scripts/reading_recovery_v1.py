"""Reading recovery v1 (R bottle-context view, O rotated sparse reread): census, freeze, serve, paired HTTP run, compare.

census              -- saved open593 bodies only: R/O eligibility and reasons; no model or OCR
plan C              -- exact pre-result HTTP roster of component C (R|O) from the census and the root roster decision
freeze C            -- candidate descriptor of C (sources, parent fa317ef2, census); no model
serve C [--port]    -- ROOT: candidate C on loopback (default 8198) under the current factory app
run C               -- ROOT, only with candidate-C/root-authorization.json naming the plan: one paired pass
                       baseline 8175 vs candidate, no retry/resume
compare C           -- saved bodies of the run only: recovery, added OCR text, selection changes, latency
describe            -- source hashes and the fixed rules
Every output is write-once; there are no sweeps, retries or reruns.
"""
import argparse
import gzip
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/evidence-consistency-v1/reading'
GATE = 'runs/recognition-repair-20260926/open-http-gate'
BASELINE_PORT = 8175
CANDIDATE_PORT = 8198
FORBIDDEN_PORTS = (8175, 8187, 8196, 8197)
MAX_PAIRED = 80
WALL_SECONDS = 1500
O_SAMPLE = 52
O_SALT = 'reading-recovery-v1-O'
ZERO_TARGET_KEY = 'main183:W0353'


def C():
    from rshb_vine.reading_recovery_v1 import candidate
    return candidate


def write_once(path, doc):
    from rshb_vine.io import write_json
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, doc)


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def folder(component):
    return OUT / f'candidate-{component}'


def cmd_census(a):
    from rshb_vine.reading_recovery_v1.census import run
    result = run(ROOT)
    write_once(OUT / 'census.json', result)
    summary = {arm: {k: v for k, v in x.items() if k not in ('eligible', 'controls')} for arm, x in result['arms'].items()}
    print(json.dumps({'checksum': result['checksum'], 'saved_bodies': result['saved_bodies'],
                      'invalid_uploads': len(result['invalid_uploads']), 'arms': summary}, ensure_ascii=False, indent=1))


# ---- plan ----

def _item(row, role, why, expected):
    return {'key': row['key'], 'path': row['path'], 'sha256': row['sha256'], 'roi': row['roi'], 'upload': row['upload'],
            'cohort': row['cohort'], 'set': row['set'], 'gt_level': row.get('gt_level'),
            'gt_products': row.get('gt_products') or [], 'acceptable_slugs': row.get('acceptable_slugs') or [],
            'role': role, 'why': why, 'expected_http': expected}


def build_plan(component):
    from rshb_vine.io import seal, sha256, verify, read_json
    census = verify(read_json(OUT / 'census.json'))
    protocol = read_json(ROOT / GATE / 'protocol.json')
    if sha256(ROOT / GATE / 'protocol.json') != census['roster_sha256']:
        raise SystemExit('open593 roster changed since the census')
    rows = {r['key']: r for r in protocol['rows']}
    cen = {r['key']: r for r in census['rows']}
    arm = census['arms'][component]
    eligible = [rows[e['key']] for e in arm['eligible']]
    if component == 'O':
        roi_eligible = [r for r in eligible if r['roi'] is not None]
        rest = sorted((r for r in eligible if r['roi'] is None),
                      key=lambda r: hashlib.sha256((r['sha256'] + O_SALT).encode()).hexdigest())
        chosen = roi_eligible + rest[:O_SAMPLE - len(roi_eligible)]
        rule = ('all eligible supplied-ROI images, then the remaining eligible by sha256(image_sha256 + %r) ascending, '
                '%d in total' % (O_SALT, O_SAMPLE))
    else:
        chosen = []
        rule = ('census has 0 R-eligible open593 images: controls plus the 5 root-admitted gallery reference '
                'probes (functional route check only, gallery role stays reference); no development benefit claim')
    items = [_item(r, 'eligible', cen[r['key']][component]['reason'], 200) for r in chosen]
    used = {i['key'] for i in items}

    def add(row, why):
        if row['key'] not in used:
            used.add(row['key'])
            items.append(_item(row, 'control', why, 200))

    valid = {r['key'] for r in census['rows']}
    negatives = sorted(k for k in valid if rows[k]['cohort'] == 'negative_nonwine')
    for key in sorted(k for k in valid if rows[k]['roi'] is not None):
        add(rows[key], 'real_supplied_roi')
    labelled = sorted(k for k in valid if rows[k]['cohort'] != 'negative_nonwine'
                      and not rows[k]['cohort'].startswith('synthetic') and rows[k].get('gt_level') not in (None, 'none')
                      and rows[k]['roi'] is None and cen[k]['targets'] and not cen[k][component]['eligible'])
    for key in labelled[:10]:
        add(rows[key], 'unchanged_labelled_real_first_by_key')
    for key in (negatives if component == 'R' else negatives[:5]):
        add(rows[key], 'negative_nonwine_valid' if component == 'R' else 'negative_nonwine_valid_first_by_key')
    add(rows[ZERO_TARGET_KEY], 'zero_target_route_retention')
    admission = None
    if component == 'R':
        candidates = verify(read_json(OUT / 'r-positive-probe-candidates.json'))
        admission = verify(read_json(OUT / 'root-reference-probe-admission.json'))
        if admission.get('decision') != 'admitted_functional_probe_only' \
                or admission.get('candidate_checksum') != candidates['checksum']:
            raise SystemExit('reference probe admission does not admit the candidate list')
        for probe in admission['items']:
            if sha256(ROOT / probe['path']) != probe['image_sha256'] or probe['role'] != 'reference_route_functional_probe':
                raise SystemExit('admitted reference probe changed: ' + probe['path'])
            items.append({'key': 'reference_probe:' + probe['slug'], 'path': probe['path'], 'sha256': probe['image_sha256'],
                          'roi': None, 'upload': probe['upload'], 'cohort': 'reference_route_functional_probe',
                          'set': 'gallery_reference', 'gt_level': None, 'gt_products': [], 'acceptable_slugs': [],
                          'reference_slug': probe['slug'], 'role': 'reference_probe',
                          'why': 'root_admitted_functional_probe_only', 'expected_http': 200})
    if len(items) > MAX_PAIRED:
        raise SystemExit('roster exceeds %d paired images' % MAX_PAIRED)
    probes = [_item(rows[i['key']], 'upload_contract_probe', 'invalid_upload_422_no_ocr', i['http_status'])
              for i in census['invalid_uploads']]
    return seal({'kind': 'reading-recovery-v1-http-plan', 'component': component, 'census_checksum': census['checksum'],
                 'roster': GATE + '/protocol.json', 'roster_sha256': census['roster_sha256'],
                 'baseline': {'port': BASELINE_PORT, 'profile_checksum': C().PARENT_CHECKSUM},
                 'candidate': {'port': CANDIDATE_PORT, 'descriptor': C().descriptor_path(component)},
                 'eligible_rule': rule, 'control_rule': 'all 12 supplied ROI, 10 unchanged labelled real first by key, '
                 + ('all 45 valid negative_nonwine' if component == 'R' else '5 valid negative_nonwine first by key')
                 + ', W0353; the 7 invalid 422 uploads are separate upload-contract probes',
                 'reference_probe_admission': None if admission is None else {
                     'path': str((OUT / 'root-reference-probe-admission.json').relative_to(ROOT)),
                     'checksum': admission['checksum']},
                 'counts': {'eligible': len(chosen), 'paired': len(items), 'probes': len(probes),
                            'reference_probes': sum(i['role'] == 'reference_probe' for i in items)},
                 'items': items, 'probes': probes,
                 'procedure': 'health on both; one excluded warm-up per arm on the first item; per item: image SHA '
                              're-verified, one POST /v1/recognize per arm with the open593 upload semantics (filename, '
                              'MIME, target_roi when ROI, no bottles field, open593 timeout), arm order alternating '
                              '(even index baseline first); then the probes on both arms; then POST /v2/targets on W0353 '
                              'to both arms; every body gz write-once; any unexpected status, transport error or '
                              'identity mismatch stops without retry; no resume',
                 'gates': {'L1': 'candidate p95 <= max(1.2 * baseline p95, baseline p95 + 0.3 s), paired items, client '
                                 'seconds, nearest-rank',
                           'safety': '0 labelled correct -> wrong; controls and ineligible items keep targets, selection and '
                                     'OCR packet; negatives gain no SKU target',
                           'benefit': 'descriptive only: recovered physical targets (R), added rotated lines and '
                                      'producer reads (O), selection changes reviewed by root'},
                 'no_change_after_results': True, 'written_before_run': True})


def cmd_plan(a):
    plan = build_plan(a.component)
    write_once(folder(a.component) / 'plan.json', plan)
    print(json.dumps({'checksum': plan['checksum'], **plan['counts'], 'rule': plan['eligible_rule']}, indent=1))


def cmd_freeze(a):
    descriptor = C().freeze_descriptor(ROOT, a.component)
    write_once(ROOT / C().descriptor_path(a.component), descriptor)
    print(json.dumps({'checksum': descriptor['checksum'], 'sources': descriptor['sources_sha256']}, indent=1))


# ---- serve / run (root) ----

def cmd_serve(a):
    from rshb_vine.target_contract_v2.runtime import create_app
    if a.port in FORBIDDEN_PORTS:
        raise SystemExit('Port %d belongs to the live service or an older candidate' % a.port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', a.port))
        listener.listen(128)
        pipeline = C().ReadingRecoveryCandidate(ROOT, a.component)
        import uvicorn
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=a.port)).run(sockets=[listener])
    finally:
        listener.close()


def _post(url, endpoint, item):
    import httpx
    data = (ROOT / item['path']).read_bytes()
    up = item['upload']
    form = {'target_roi': json.dumps(item['roi'])} if item['roi'] is not None else {}
    started = time.perf_counter()
    try:
        response = httpx.post(url + endpoint, files={'image': (up['filename'], data, up['mime'])}, data=form,
                              timeout=up['timeout'])
    except httpx.HTTPError as exc:
        return {'status': None, 'error': type(exc).__name__, 'seconds': time.perf_counter() - started}, None
    seconds = time.perf_counter() - started
    try:
        body = response.json()
    except ValueError:
        body = {'_raw_text': response.text[:2000]}
    return {'status': response.status_code, 'seconds': seconds, 'request_sha256': hashlib.sha256(data).hexdigest()}, body


def _health(url):
    import httpx
    response = httpx.get(url + '/health/ready', timeout=15)
    body = response.json() if response.status_code == 200 else {}
    if response.status_code != 200 or body.get('ready') is not True:
        raise SystemExit('Server not ready: ' + url)
    return body


def _identity_problem(arm, body, plan, descriptor):
    if body.get('slug') is not None or body.get('probability_correct') is not None:
        return 'public slug/probability not None'
    block = body.get(C().BLOCK)
    if arm == 'baseline':
        return 'baseline response carries a reading-recovery block' if block is not None else None
    if not block or block.get('descriptor_checksum') != descriptor['checksum'] or block.get('component') != plan['component'] \
            or block.get('calibrated') is not False or block.get('release_admitted') is not False:
        return 'candidate block does not name the frozen descriptor/component or claims calibration/admission'
    return None


def cmd_run(a):
    from rshb_vine.io import read_json, sha256, verify
    base = folder(a.component)
    plan = verify(read_json(base / 'plan.json'))
    auth = verify(read_json(base / 'root-authorization.json'))
    if auth.get('plan_checksum') != plan['checksum'] or auth.get('component') != a.component:
        raise SystemExit('root authorization does not name this plan')
    descriptor = C().load_descriptor(ROOT, plan['candidate']['descriptor'])
    journal = base / 'http/journal.jsonl'
    if journal.exists():
        raise SystemExit('journal exists; no resume after a partial or failed run')
    urls = {'baseline': 'http://127.0.0.1:%d' % plan['baseline']['port'],
            'candidate': 'http://127.0.0.1:%d' % plan['candidate']['port']}
    started, sent, snapshots = time.monotonic(), 0, {}
    budget = 2 * (len(plan['items']) + len(plan['probes']) + 2)

    def healthy():
        found = {arm: _health(url) for arm, url in urls.items()}
        if found['candidate'].get('runtime_descriptor_checksum') != descriptor['checksum']:
            raise SystemExit('candidate server does not serve the frozen descriptor')
        if found['baseline'].get('runtime_descriptor_checksum') != plan['baseline']['profile_checksum']:
            raise SystemExit('baseline server is not fa317ef2')
        now = {arm: h.get('snapshot') for arm, h in found.items()}
        if not all(now.values()) or (snapshots and now != snapshots):
            raise SystemExit('health snapshot missing or changed during the run: %r' % now)
        snapshots.update(now)
        return found

    def stop(reason, **extra):
        append(journal, {'event': 'stop', 'reason': reason, 'elapsed': time.monotonic() - started, 'sent': sent, **extra})
        raise SystemExit('stopped: %s; no retry' % reason)

    def request(item, arm, event, endpoint='/v1/recognize'):
        nonlocal sent
        if time.monotonic() - started > WALL_SECONDS:
            stop('wall_cap', key=item['key'])
        if sent >= budget:
            stop('request_budget', key=item['key'])
        if sha256(ROOT / item['path']) != item['sha256']:
            stop('image SHA changed', key=item['key'])
        sent += 1
        meta, body = _post(urls[arm], endpoint, item)
        file = None
        if body is not None:
            name = hashlib.sha256(item['key'].encode()).hexdigest()[:24]
            file = base / 'http/responses' / arm / ('%s%s.json.gz' % ({'warmup': 'warmup-', 'compact': 'compact-'}.get(event, ''), name))
            if file.exists():
                stop('response file exists', key=item['key'])
            file.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(file, 'wt') as f:
                json.dump({'key': item['key'], 'http_status': meta['status'], 'body': body,
                           'request': {'endpoint': endpoint, 'roi': item['roi'], **item['upload']}}, f, ensure_ascii=False)
            meta['response_sha256'] = sha256(file)
            file = str(file.relative_to(ROOT))
        append(journal, {'event': event, 'key': item['key'], 'arm': arm, 'endpoint': endpoint, 'file': file, **meta})
        problem = None
        if meta['status'] != item['expected_http']:
            problem = 'HTTP %s (expected %s) %s' % (meta['status'], item['expected_http'], meta.get('error') or '')
        elif meta.get('request_sha256') != item['sha256']:
            problem = 'request bytes differ from the item SHA'
        elif meta['status'] == 200 and endpoint == '/v1/recognize':
            problem = _identity_problem(arm, body, plan, descriptor)
        if problem:
            stop(problem, key=item['key'], arm=arm)

    append(journal, {'event': 'start', 'plan': plan['checksum'], 'authorization': auth['checksum'],
                     'health': healthy(), 'at': time.time()})
    items = plan['items']
    for arm in ('baseline', 'candidate'):
        request(items[0], arm, 'warmup')
    for index, item in enumerate(items + plan['probes']):
        healthy()
        order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
        for arm in order:
            request(item, arm, 'request' if item['role'] != 'upload_contract_probe' else 'probe')
    zero = next(i for i in items if i['key'] == ZERO_TARGET_KEY)
    for arm in ('baseline', 'candidate'):
        request(zero, arm, 'compact', '/v2/targets')
    append(journal, {'event': 'done', 'health': healthy(), 'elapsed': time.monotonic() - started, 'sent': sent})
    print(json.dumps({'sent': sent, 'elapsed': time.monotonic() - started}))


# ---- compare (saved bodies only) ----

def _p95(values):
    values = sorted(values)
    return values[max(0, -(-95 * len(values) // 100) - 1)] if values else None


def _p50(values):
    values = sorted(values)
    return values[(len(values) - 1) // 2] if values else None


def _targets(body):
    out = []
    for t in body.get('targets') or []:
        product = (t.get('product_resolution') or {}).get('product_id')
        out.append({'instance_id': str(t.get('instance_id')), 'bbox': t.get('bbox'), 'bottle_bbox': t.get('bottle_bbox'),
                    'physical': bool(t.get('physical_bottle_localized')), 'label_localized': t.get('label_localized', True),
                    'best': (t.get('retrieval') or {}).get('best_candidate'), 'product_id': product})
    return out


def _packet(body):
    return [(o.get('raw_text'), o.get('score')) for o in (body.get('variant_text') or {}).get('observations', [])] \
        if len(body.get('targets') or []) == 1 else None


def _correct(item, target):
    if not target:
        return None
    if item['gt_products']:
        return target['product_id'] in item['gt_products']
    if item['acceptable_slugs']:
        return target['best'] in item['acceptable_slugs']
    return None


def _subsequence(small, big):
    it = iter(big)
    return all(x in it for x in small)


def cmd_compare(a):
    import collections
    from rshb_vine.io import read_json, seal, sha256, verify
    base = folder(a.component)
    plan = verify(read_json(base / 'plan.json'))
    journal = jsonl(base / 'http/journal.jsonl')
    if not journal or journal[-1].get('event') != 'done':
        raise SystemExit('run is not complete')
    files = {}
    for row in journal:
        if row['event'] in ('request', 'probe') and row.get('file'):
            if sha256(ROOT / row['file']) != row['response_sha256']:
                raise SystemExit('response changed: ' + row['file'])
            files[(row['key'], row['arm'])] = (row, json.load(gzip.open(ROOT / row['file'], 'rt'))['body'])
    rows, seconds = [], {'baseline': [], 'candidate': []}
    for item in plan['items']:
        (mb, b), (mc, c) = files[(item['key'], 'baseline')], files[(item['key'], 'candidate')]
        seconds['baseline'].append(mb['seconds'])
        seconds['candidate'].append(mc['seconds'])
        tb, tc = _targets(b), _targets(c)
        cb = _correct(item, tb[0]) if len(tb) == 1 else None
        cc = _correct(item, tc[0]) if len(tc) == 1 else None
        pb, pc = _packet(b), _packet(c)
        route, rotated = c.get('bottle_context_route') or {}, c.get('rotated_label_ocr') or {}
        added = [o['raw_text'] for r in rotated.get('reads', []) for o in r['raw_observations']][:40]
        rows.append({'key': item['key'], 'role': item['role'], 'why': item['why'], 'cohort': item['cohort'],
                     'gt_level': item['gt_level'], 'targets': [len(tb), len(tc)],
                     'physical_targets': [sum(t['physical'] for t in tb), sum(t['physical'] for t in tc)],
                     'selected': [[t['product_id'] or t['best'] for t in tb], [t['product_id'] or t['best'] for t in tc]],
                     'selection_changed': [t['product_id'] or t['best'] for t in tb] != [t['product_id'] or t['best'] for t in tc],
                     'correct': [cb, cc],
                     'packet_preserved': None if pb is None or pc is None else _subsequence(pb, pc),
                     'packet_changed': pb != pc,
                     'R': {k: route.get(k) for k in ('attempted', 'reason', 'recovered', 'retry_ms', 'converted')} if route else None,
                     'O': {'eligible': rotated.get('eligible'), 'reason': rotated.get('reason'), 'performed': rotated.get('performed'),
                           'added_lines': rotated.get('added_lines'), 'ms': rotated.get('ms'), 'read_texts': added,
                           'producer_hits_after': None} if rotated else None,
                     'seconds': [mb['seconds'], mc['seconds']]})
    probes = [{'key': p['key'], 'status': [files.get((p['key'], arm), ({'status': None},))[0].get('status')
                                           for arm in ('baseline', 'candidate')]} for p in plan['probes']]
    pb, pc = _p95(seconds['baseline']), _p95(seconds['candidate'])
    limit = max(1.2 * pb, pb + 0.3)
    labelled = [r for r in rows if r['correct'][0] is not None or r['correct'][1] is not None]
    summary = seal({
        'kind': 'reading-recovery-v1-http-compare', 'component': a.component, 'plan_checksum': plan['checksum'],
        'items': len(rows), 'probes': probes,
        'latency': {'baseline_p95': pb, 'candidate_p95': pc, 'limit': limit, 'L1': pc <= limit,
                    'by_role': {role: {'n': len(sub), 'baseline_p50': _p50([r['seconds'][0] for r in sub]),
                                       'candidate_p50': _p50([r['seconds'][1] for r in sub]),
                                       'baseline_p95': _p95([r['seconds'][0] for r in sub]),
                                       'candidate_p95': _p95([r['seconds'][1] for r in sub]),
                                       'added_seconds': sorted(r['seconds'][1] - r['seconds'][0] for r in sub)}
                                for role, sub in (('eligible', [r for r in rows if r['role'] == 'eligible']),
                                                  ('control', [r for r in rows if r['role'] == 'control']),
                                                  ('performed', [r for r in rows if (r['O'] or {}).get('performed')
                                                                 or (r['R'] or {}).get('attempted')]))}},
        'labelled': {'correct_to_wrong': [r['key'] for r in labelled if r['correct'] == [True, False]],
                     'wrong_to_correct': [r['key'] for r in labelled if r['correct'] == [False, True]],
                     'baseline_correct': sum(r['correct'][0] is True for r in labelled),
                     'candidate_correct': sum(r['correct'][1] is True for r in labelled)},
        'changed': {'selection': [r['key'] for r in rows if r['selection_changed']],
                    'targets': [r['key'] for r in rows if r['targets'][0] != r['targets'][1]],
                    'packet_not_preserved': [r['key'] for r in rows if r['packet_preserved'] is False],
                    'controls_changed': [r['key'] for r in rows if r['role'] == 'control'
                                         and (r['selection_changed'] or r['targets'][0] != r['targets'][1] or r['packet_changed'])],
                    'eligible_not_performed': [r['key'] for r in rows if r['role'] == 'eligible' and not (
                        (r['O'] or {}).get('performed') or (r['R'] or {}).get('attempted'))],
                    'negative_targets': [r['key'] for r in rows if r['cohort'] == 'negative_nonwine' and r['targets'][1] > r['targets'][0]]},
        'R': {'attempted': sum(bool(r['R'] and r['R'].get('attempted')) for r in rows),
              'recovered': [r['key'] for r in rows if r['R'] and r['R'].get('recovered')],
              'reference_probes': [{'key': r['key'], 'targets': r['targets'], 'R': r['R']} for r in rows
                                   if r['role'] == 'reference_probe']},
        'O': {'performed': sum(bool(r['O'] and r['O'].get('performed')) for r in rows),
              'added_lines': sum((r['O'] or {}).get('added_lines') or 0 for r in rows),
              'reasons': dict(collections.Counter((r['O'] or {}).get('reason') for r in rows)),
              'ms_p95': _p95([r['O']['ms'] for r in rows if r['O'] and r['O'].get('ms') is not None])},
        'rows': rows,
        'limits': 'open593 development roster; no independent accuracy, no calibration; every change needs root review'})
    write_once(base / 'compare.json', summary)
    print(json.dumps({k: summary[k] for k in ('latency', 'labelled', 'R', 'O')}, indent=1))
    print(json.dumps({k: v if len(v) < 40 else len(v) for k, v in summary['changed'].items()}, indent=1))


def cmd_describe(a):
    from rshb_vine.io import sha256
    from rshb_vine.reading_recovery_v1 import route as R, rotated_ocr as O
    print(json.dumps({'sources_sha256': {p: sha256(ROOT / p) for p in C().SOURCES},
                      'R': {'policy': R.POLICY, 'rule': R.__doc__.strip()},
                      'O': {'policy': O.POLICY, 'extra_rotations': O.EXTRA_ROTATIONS, 'rule': O.__doc__.strip()}},
                     ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('census').set_defaults(fn=cmd_census)
    sub.add_parser('describe').set_defaults(fn=cmd_describe)
    for name, fn in (('plan', cmd_plan), ('freeze', cmd_freeze), ('run', cmd_run), ('compare', cmd_compare)):
        s = sub.add_parser(name)
        s.add_argument('component', choices=('R', 'O'))
        s.set_defaults(fn=fn)
    s = sub.add_parser('serve')
    s.add_argument('component', choices=('R', 'O'))
    s.add_argument('--port', type=int, default=CANDIDATE_PORT)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
