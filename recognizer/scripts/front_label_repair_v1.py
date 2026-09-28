"""Front-label repair v1: fixed-bounds probe, geometry class scan and candidate service.

probe   -- one bounded B3-4343 counterfactual on predefined physical crop bounds (3 images)
scan    -- rule class on stored train137 + audit25 geometry (SSDlite on CPU only)
freeze  -- seal the candidate profile (pins own sources, base release profile, SSDlite manifest)
serve   -- loopback candidate service (default 8188); 8175 and the current pointer are never touched
plan    -- fixed HTTP request list: every scanned target the rule fires on + non-firing Markotkh control
plan2   -- v2 list: the same v1 requests + one clean supplied-ROI request + the most crowded synthetic scene
http    -- one append-only pass of the plan against a running candidate; request budget enforced
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/recognition-repair-20260926/crop'

ARATTI = ['aratti-kaberne-sovinon-2020-krasnoe-suhoe']
PROBE_CASES = [
    {'id': 'organizer-022', 'path': 'data/real-photos/92.6_07-09-2026_11-04-40.webp',
     'sha256': '3c1e06bc461fe048e78dd010f8d990b5387895b02126aa02e01ff8d914582dce', 'gt': ARATTI,
     'related': ['aratti-kaberne-sovinon-2021-krasnoe-suhoe', 'vinodelnya-begildeeva-kaberne-sovinon-krasnoe-suhoe-14'],
     'variants': [
         {'name': 'served_label', 'channel': 'label', 'bbox': [738, 948, 2331, 3577], 'basis': 'served detected_label view'},
         {'name': 'orphan_stage_label', 'channel': 'label', 'bbox': [738, 356, 2331, 3716], 'basis': 'orphan-stage label detector box (sent to retry)'},
         {'name': 'body_width_served_y', 'channel': 'label', 'bbox': [246, 948, 2592, 3577], 'basis': 'parent bottle x-range x served label y-range'},
         {'name': 'body_width_orphan_y', 'channel': 'label', 'bbox': [246, 356, 2592, 3716], 'basis': 'parent bottle x-range x orphan-stage label y-range'},
         {'name': 'manual_full_label', 'channel': 'label', 'bbox': [246, 6, 2592, 3716], 'basis': 'visual physical label bounds: body width, frame top, label bottom edge'},
         {'name': 'parent_bottle_as_label', 'channel': 'label', 'bbox': [246, 6, 2592, 4000], 'basis': 'physical parent bottle box'},
         {'name': 'served_context', 'channel': 'context', 'bbox': [738, 356, 2331, 3716], 'basis': 'served context view'},
         {'name': 'parent_bottle_context', 'channel': 'context', 'bbox': [246, 6, 2592, 4000], 'basis': 'physical parent bottle box'}]},
    {'id': 'organizer-024', 'path': 'data/real-photos/92.96_19-08-2026_21-07-12.webp',
     'sha256': '120946ab5cf6c2b84bd9de87e82b76c916f7ebba6154759649eaaf266bbc61ca', 'gt': ARATTI,
     'related': ['aratti-kaberne-sovinon-2021-krasnoe-suhoe', 'vinodelnya-begildeeva-kaberne-sovinon-krasnoe-suhoe-14'],
     'variants': [
         {'name': 'served_label', 'channel': 'label', 'bbox': [551, 983, 2781, 3994], 'basis': 'served detected_label view'},
         {'name': 'manual_full_label', 'channel': 'label', 'bbox': [672, 0, 2563, 3994], 'basis': 'visual physical label bounds: body edges, frame top'},
         {'name': 'served_x_full_height', 'channel': 'label', 'bbox': [551, 0, 2781, 4032], 'basis': 'served label x-range x full frame height'},
         {'name': 'full_frame_as_label', 'channel': 'label', 'bbox': [0, 0, 3024, 4032], 'basis': 'unlocalized close-up frame'},
         {'name': 'served_context', 'channel': 'context', 'bbox': [0, 0, 3024, 4032], 'basis': 'served context view'}]},
    {'id': 'S1000-5b3f0369aaa994',
     'path': 'data/internet-slug1000-20260922/by-slug/usadba-markoth-shardone-beloe-suhoe-12/a3e8680951d054fe0668f5c5ce95cf2d51b3dc37b247bc8561413d53f7241e68.jpg',
     'sha256': 'a3e8680951d054fe0668f5c5ce95cf2d51b3dc37b247bc8561413d53f7241e68',
     'gt': ['usadba-markoth-shardone-beloe-suhoe-12'],
     'related': ['vinodelnya-myshako-shardone-beloe-suhoe-123', 'usadba-markoth-kyuve-blan-shardone-beloe-suhoe-12'],
     'variants': [
         {'name': 'served_label', 'channel': 'label', 'bbox': [358, 970, 718, 1494], 'basis': 'served detected_label view (visually complete label)'},
         {'name': 'label_margin5', 'channel': 'label', 'bbox': [340, 944, 736, 1520], 'basis': 'served label +5% each side'},
         {'name': 'body_width_served_y', 'channel': 'label', 'bbox': [322, 970, 743, 1494], 'basis': 'parent bottle x-range x served label y-range'},
         {'name': 'parent_bottle_as_label', 'channel': 'label', 'bbox': [322, 4, 743, 1594], 'basis': 'physical parent bottle box'},
         {'name': 'served_context', 'channel': 'context', 'bbox': [322, 4, 743, 1594], 'basis': 'served context view'},
         {'name': 'full_frame_context', 'channel': 'context', 'bbox': [0, 0, 1067, 1600], 'basis': 'full frame'}]},
]


def write_new(path, obj):
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def cmd_probe(a):
    from rshb_vine.front_label_repair_v1 import probe
    out = OUT / 'probe'
    target = out / 'probe.json'
    if target.exists():
        raise SystemExit('probe already recorded: ' + str(target))
    (out / 'crops').mkdir(parents=True, exist_ok=True)
    result = probe.run(ROOT, PROBE_CASES, out / 'crops', a.device)
    result['kind'] = 'front-label-repair-v1-fixed-bounds-probe'
    result['variants_predefined_in'] = 'scripts/front_label_repair_v1.py PROBE_CASES'
    write_new(target, result)
    for r in result['rows']:
        print(r['id'][:12], r['variant'].ljust(24), r['channel'].ljust(11), 'gt_rank', r['gt_rank'],
              'gt_score', r['gt_score'], 'top1', r['top3'][0], r['related_ranks'])
    print('crops', result['crops_encoded'], 'load %.1fs infer %.1fs' % (result['load_seconds'], result['inference_seconds']))


def stored_targets():
    """Allowed stored receipts: train137 of the B3-only open183 HTTP run and the 25-case source audit (456 baseline)."""
    import gzip
    b3 = ROOT / 'runs/b3-only-integration-v1/candidate-http'
    roster = {r['id']: r for r in json.loads((b3 / 'roster.json').read_text())['rows']}
    for r in json.loads((b3 / 'report.json').read_text())['rows']:
        if roster[r['id']]['split'] == 'train':
            body = json.load(gzip.open(b3 / r['file']))
            yield 'train137', r['id'], roster[r['id']]['path'], roster[r['id']]['sha256'], r['gt'], body.get('body', body)
    base = ROOT / 'runs/regression-audit-20260926/baseline/receipts'
    ledger = {json.loads(l)['request_key']: json.loads(l) for l in open(base / 'ledger.jsonl')}
    rows = {r['request_key']: r for r in json.loads((ROOT / 'runs/system-capability-snapshot-v1/protocol/roster.json').read_text())['rows']}
    cases = json.loads((ROOT / 'runs/b3-loss-source-audit-20260926/visual/evidence.json').read_text())['cases']
    for c in cases:
        k = c['request_key']
        if k in ledger:
            body = json.load(gzip.open(base / 'responses' / ledger[k]['file']))['body']
            yield 'audit25', k, rows[k]['path'], rows[k]['sha256'], rows[k]['acceptable_slugs'], body


def cmd_scan(a):
    """Rule class on stored geometry; SSDlite (CPU, the orphan-stage artifact) is the only model run."""
    from rshb_vine.front_label_repair_v1 import geometry
    from rshb_vine.io import sha256
    from rshb_vine.label_detector import LabelDetector
    from rshb_vine.preprocessing import decode
    target = OUT / 'scan' / 'class-scan.json'
    if target.exists():
        raise SystemExit('scan already recorded: ' + str(target))
    ssd = LabelDetector(ROOT / 'runs/label-detector-real-v2', 'cpu')
    rows, images = [], 0
    for src, key, path, sha, gt, body in stored_targets():
        if sha256(ROOT / path) != sha:
            raise ValueError('Stored query SHA changed: ' + path)
        image, _ = decode((ROOT / path).read_bytes())
        images += 1
        for t in body.get('targets', []):
            views = t['retrieval']['views']
            label = next(v['bbox'] for v in views if v['kind'] != 'context')
            context = next(v['bbox'] for v in views if v['kind'] == 'context')
            orphan = str(t['instance_id']).startswith('orphan')
            # Orphan targets come from a retry on the orphan label crop: frame = that crop (the served context view).
            frame = context if orphan else [0, 0, *image.size]
            parent = frame if orphan or t.get('bottle_bbox') is None else context
            crop = image.crop(frame)
            ssd_boxes = [{'bbox': [r['bbox'][0] + frame[0], r['bbox'][1] + frame[1], r['bbox'][2] + frame[0],
                                   r['bbox'][3] + frame[1]], 'score': r['detector_score']} for r in ssd.detect(crop)]
            if orphan:
                ssd_boxes.append({'bbox': list(frame), 'score': geometry.SSD_SCORE, 'source': 'orphan_stage_ssd_crop'})
            box, trace = geometry.recovered_label(label, parent, ssd_boxes)
            rows.append({'src': src, 'key': key, 'instance_id': str(t['instance_id']), 'orphan': orphan,
                         'unlocalized': t.get('bottle_bbox') is None, 'path': path, 'sha256': sha, 'gt': gt,
                         'size': list(image.size), 'served_best': t['retrieval'].get('best_candidate'),
                         'rotation_ccw': t.get('retrieval_pixel_rotation_ccw', 0), **trace})
    fired = [r for r in rows if r['added']]
    write_new(target, {'kind': 'front-label-repair-v1-class-scan', 'policy': geometry.POLICY,
                       'constants': {'ssd_score': geometry.SSD_SCORE, 'same_label_iou': geometry.SAME_LABEL_IOU,
                                     'min_growth': geometry.MIN_GROWTH},
                       'model_run': 'SSDlite main-label detector runs/label-detector-real-v2 on CPU only',
                       'images': images, 'targets': len(rows), 'fired': len(fired), 'rows': rows})
    print('images', images, 'targets', len(rows), 'fired', len(fired))
    for r in fired:
        print(r['src'], r['key'][-26:], r['instance_id'], 'L', r['served_label'], '->', r['recovered'],
              'growth', r['growth'], 'P', r['parent'], r['size'])


BUDGET_TOTAL = 24
PROBE_REQUESTS = 3
CONTROL = 'real_whole71:S1000-5b3f0369aaa994'


def cmd_freeze(a):
    from rshb_vine.front_label_repair_v1 import runtime
    from rshb_vine.io import seal, write_json
    path = ROOT / runtime.PROFILE
    if path.exists():
        raise SystemExit('profile exists; immutable profile preserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = seal(runtime.freeze_body(ROOT))
    write_json(path, profile)
    print(json.dumps({'profile': runtime.PROFILE, 'checksum': profile['checksum']}))


def cmd_serve(a):
    import socket
    import uvicorn
    from rshb_vine.front_label_repair_v1 import runtime
    from rshb_vine.target_contract_v2.runtime import create_app
    if a.port == 8175:
        raise SystemExit('8175 is the current release; the candidate never binds it')
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', a.port))
        listener.listen(128)
        pipeline = runtime.FrontLabelRepairRecognition(ROOT)
        print(json.dumps({'runtime': pipeline.manifest['checksum'], 'installation': pipeline.installation}), flush=True)
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=a.port)).run(sockets=[listener])
    finally:
        listener.close()


def cmd_plan(a):
    scan = json.loads((OUT / 'scan/class-scan.json').read_text())
    target = OUT / 'http/plan.json'
    keys, rows = [], {}
    for r in scan['rows']:
        if r['added'] or r['key'] == CONTROL:
            if r['key'] not in rows:
                keys.append(r['key'])
                rows[r['key']] = {'key': r['key'], 'src': r['src'], 'path': r['path'], 'sha256': r['sha256'],
                                  'gt': r['gt'], 'fired_instances': []}
            if r['added']:
                rows[r['key']]['fired_instances'].append(r['instance_id'])
    if len(keys) + PROBE_REQUESTS > BUDGET_TOTAL:
        raise SystemExit('plan exceeds the initial request budget: %d' % len(keys))
    write_new(target, {'kind': 'front-label-repair-v1-http-plan', 'scan_sha256': _sha(OUT / 'scan/class-scan.json'),
                       'rule': 'all scanned targets where the frozen rule adds a view, plus one non-firing control',
                       'control': CONTROL, 'requests': len(keys), 'probe_requests': PROBE_REQUESTS,
                       'budget_total': BUDGET_TOTAL, 'rows': [rows[k] for k in keys]})
    print(len(keys), 'requests planned;', len(keys) + PROBE_REQUESTS, 'of', BUDGET_TOTAL)


V2_EXTRA = ['real_supplied_roi:real_open426:rshb-photo-Q0288-target-0', 'synthetic_hard36:11-angle']


def cmd_plan2(a):
    v1 = json.loads((OUT / 'v1/http/plan.json').read_text())
    roster = {r['request_key']: r for r in json.loads((ROOT / 'runs/system-capability-snapshot-v1/protocol/roster.json').read_text())['rows']}
    rows = [dict(r, roi=None) for r in v1['rows']]
    for key in V2_EXTRA:
        r = roster[key]
        rows.append({'key': key, 'src': 'v2_gate', 'path': r['path'], 'sha256': r['sha256'],
                     'gt': r['acceptable_slugs'], 'fired_instances': None, 'roi': r['roi']})
    if len(rows) > BUDGET_TOTAL:
        raise SystemExit('plan exceeds the v2 request budget')
    write_new(OUT / 'http/plan.json', {'kind': 'front-label-repair-v2-http-plan', 'v1_plan_sha256': _sha(OUT / 'v1/http/plan.json'),
                                       'rule': 'same v1 requests + supplied ROI + most crowded synthetic_hard36 scene (max targets)',
                                       'requests': len(rows), 'budget_total': BUDGET_TOTAL, 'validation46': 'not opened',
                                       'rows': rows})
    print(len(rows), 'requests planned')


def _sha(path):
    from rshb_vine.io import sha256
    return sha256(path)


def cmd_http(a):
    import gzip
    import time
    import requests
    plan = json.loads((OUT / 'http/plan.json').read_text())
    out = OUT / 'http'
    (out / 'responses').mkdir(parents=True, exist_ok=True)
    ledger = out / 'ledger.jsonl'
    done = {json.loads(l)['key'] for l in open(ledger)} if ledger.exists() else set()
    url = 'http://127.0.0.1:%d' % a.port
    ready = requests.get(url + '/health/ready', timeout=10).json()
    for row in plan['rows']:
        if row['key'] in done:
            continue
        path = ROOT / row['path']
        if _sha(path) != row['sha256']:
            raise SystemExit('query SHA changed: ' + row['path'])
        data = path.read_bytes()
        started = time.time()
        form = {'target_roi': json.dumps(row['roi'])} if row.get('roi') else None
        r = requests.post(url + '/v1/recognize', files={'image': (path.name, data, 'application/octet-stream')},
                          data=form, timeout=180)
        name = row['key'].replace(':', '_').replace('/', '_') + '.json.gz'
        with gzip.open(out / 'responses' / name, 'wt') as f:
            json.dump({'http_status': r.status_code, 'body': r.json()}, f, ensure_ascii=False)
        rec = {'key': row['key'], 'http_status': r.status_code, 'seconds': time.time() - started, 'file': name,
               'response_sha256': _sha(out / 'responses' / name), 'runtime': ready.get('snapshot')}
        with open(ledger, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(rec['key'], rec['http_status'], '%.1fs' % rec['seconds'], flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('probe')
    s.add_argument('--device', default='mps')
    s.set_defaults(fn=cmd_probe)
    s = sub.add_parser('scan')
    s.set_defaults(fn=cmd_scan)
    s = sub.add_parser('freeze')
    s.set_defaults(fn=cmd_freeze)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=8188)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser('plan')
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser('plan2')
    s.set_defaults(fn=cmd_plan2)
    s = sub.add_parser('http')
    s.add_argument('--port', type=int, default=8188)
    s.set_defaults(fn=cmd_http)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
