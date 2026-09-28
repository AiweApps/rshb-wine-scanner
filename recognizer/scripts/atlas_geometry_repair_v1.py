"""Atlas geometry repair G: census of saved responses, candidate freeze/describe/serve/replay and saved-response compare.

census   CPU only: classify frozen orphan trials of the saved atlas521 (cd0b9910) and open593 (013c845a) responses.
freeze   staged descriptor of the candidate (own source SHA + immutable F10 profile); no active pointer is written.
describe descriptor + source checks, no models loaded.
serve    loopback candidate (default 8221) through the same target-contract HTTP app as the live F10 kind.
replay   in-process candidate over explicit inputs; each response gz + journal row; optional nested-retry capture.
compare  candidate responses against the saved baseline responses of the same input bytes.
Root runs every command that loads models; this script never touches 8175/8187, the factory or the current pointer.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import read_json, seal, sha256, verify, write_json  # noqa: E402
from rshb_vine.atlas_geometry_repair_v1 import plan as P  # noqa: E402

OUT = 'runs/atlas-repair-v1/geometry'
DESCRIPTOR = OUT + '/candidate-v4/descriptor.json'
ATLAS = 'runs/wine-atlas-20260927/stable-run/per-image-evaluation.jsonl'
OPEN_PROTOCOL = 'runs/recognition-repair-20260926/open-http-gate/protocol.json'
OPEN_PROTOCOL_SHA256 = 'a958747c7ec0271fc4dd0115303a7da2298c7640d09a8a424c2e4b76bc25e2e7'
OPEN_LEDGER = 'runs/recognition-repair-20260926/open-http-gate/receipts/ledger.jsonl'
OPEN_RESPONSES = 'runs/recognition-repair-20260926/open-http-gate/receipts/responses'
FORBIDDEN_PORTS = (8175, 8187, 8189, 8192, 8196, 8197, 8198, 8199)


def _gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def _rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _summary(result):
    return {'decision': result.get('decision'), 'targets': len(result.get('targets') or []),
            'target_ids': [str(t.get('instance_id')) for t in result.get('targets') or []],
            'physical_targets': sum(1 for t in result.get('targets') or [] if t.get('physical_bottle_localized')),
            'best_candidate': result.get('best_candidate')}


def _census_row(key, source, response_path, response_sha, result, extra):
    rows = P.candidates(result)
    if extra.get('roi') is not None:
        for row in rows:
            if row['eligible']:
                row.update(eligible=False, skip='roi_request')
    for row in rows:
        if row.get('kind') == P.LABEL_ONLY:
            row['gate_context'] = P.label_only_context(result, row['label_bbox'], (0, 0))[1]
    eligible = [r['kind'] for r in rows if r['eligible']]
    armed = extra.get('roi') is None and result.get('decision') != 'invalid_image' and not result.get('targets') and bool(eligible)
    return {'key': key, 'source': source, 'response': response_path, 'response_sha256': response_sha, **extra,
            **_summary(result), 'trials': rows, 'eligible': eligible, 'armed_pass_expected': armed}


def census(root):
    root = Path(root)
    out = []
    for r in _rows(root / ATLAS):
        path = r['response_path']
        if sha256(root / path) != r['response_sha256']:
            raise SystemExit('Saved atlas response changed: ' + path)
        out.append(_census_row('atlas:%04d' % r['index'], 'atlas521_cd0b9910', path, r['response_sha256'], _gz(root / path),
                               {'input_path': r['input_path'], 'input_sha256': r['input_sha256'], 'role': r.get('role'),
                                'group': r.get('group')}))
    if sha256(root / OPEN_PROTOCOL) != OPEN_PROTOCOL_SHA256:
        raise SystemExit('open593 roster differs from the audited roster (runs/evidence-consistency-v1/open593-admission.json)')
    protocol = read_json(root / OPEN_PROTOCOL)
    rows = {row['key']: row for row in protocol['rows']}
    for entry in _rows(root / OPEN_LEDGER):
        path = entry['reused_from'] if 'reused_from' in entry else f"{OPEN_RESPONSES}/{entry['file']}"
        if entry.get('http_status') != 200:
            continue
        if sha256(root / path) != entry['response_sha256']:
            raise SystemExit('Saved open593 response changed: ' + path)
        row = rows[entry['key']]
        saved = _gz(root / path)
        out.append(_census_row('open:' + entry['key'], 'open593_013c845a', path, entry['response_sha256'], saved.get('body', saved),
                               {'input_path': row['path'], 'input_sha256': row['sha256'], 'roi': row.get('roi'),
                                'cohort': row.get('cohort'), 'set': row.get('set')}))
    eligible = [r for r in out if r['eligible']]
    reasons = {}
    for r in out:
        for t in r['trials']:
            name = '%s|%s' % (t['frozen_reason'], t.get('kind') or t.get('skip'))
            reasons[name] = reasons.get(name, 0) + 1
    doc = seal({'kind': 'atlas-geometry-repair-v1-census', 'policy': P.POLICY, 'constants': P.CONSTANTS,
                'plan_sha256': sha256(root / 'rshb_vine/atlas_geometry_repair_v1/plan.py'),
                'sources': {'atlas521': ATLAS, 'open593': OPEN_LEDGER}, 'responses': len(out),
                'with_orphan_trials': sum(1 for r in out if r['trials']), 'trial_classes': dict(sorted(reasons.items())),
                'eligible_responses': len(eligible),
                'eligible_by_source_kind': {f"{s}|{k}": sum(1 for r in eligible if r['source'] == s and k in r['eligible'])
                                            for s in ('atlas521_cd0b9910', 'open593_013c845a') for k in (P.PHYSICAL, P.LABEL_ONLY)},
                'eligible_with_existing_targets': [r['key'] for r in eligible if r['targets']],
                'armed_pass_expected': sum(1 for r in out if r['armed_pass_expected']),
                'note': 'saved responses only, no inference; open593 responses are 013c845a (same frozen orphan/geometry '
                        'stages, older selector); a ROI request is never eligible at runtime; the saved response is the '
                        'complete frozen pass, whose final orphan trace approximates the observe pass (a canvas retry '
                        'may observe further trials)', 'rows': out})
    write_json(root / OUT / ('census-%s.json' % doc['plan_sha256'][:8]), doc)
    return {k: doc[k] for k in ('checksum', 'responses', 'with_orphan_trials', 'trial_classes', 'eligible_responses',
                                'eligible_by_source_kind', 'eligible_with_existing_targets', 'armed_pass_expected')}


def _census(root, name):
    docs = sorted((root / OUT).glob('census-*.json'))
    doc = verify(read_json(root / name)) if name else None
    if doc is None:
        want = 'census-%s.json' % sha256(root / 'rshb_vine/atlas_geometry_repair_v1/plan.py')[:8]
        if not (root / OUT / want).exists():
            raise SystemExit('No census for the current plan.py; run census first (found: %s)' % [d.name for d in docs])
        doc = verify(read_json(root / OUT / want))
    return doc, {r['key']: r for r in doc['rows']}


def freeze(root, path):
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    doc = G.freeze(root)
    write_json(root / path, doc)
    return {'descriptor': path, 'checksum': doc['checksum'], 'sources_sha256': doc['sources_sha256'],
            'parent_release': doc['parent_release']['checksum']}


def describe(root, path):
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    doc = G.load_descriptor(root, path)
    return {'descriptor': path, 'checksum': doc['checksum'], 'policy': doc['policy'], 'constants': doc['constants'],
            'parent_release': doc['parent_release'], 'port': doc['port'], 'activated': doc['activated'],
            'release_admitted': doc['release_admitted'], 'probability': None, 'models_loaded': False}


def serve(root, port, path):
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    from rshb_vine.target_contract_v2.runtime import create_app
    if port in FORBIDDEN_PORTS:
        raise SystemExit('Port %d belongs to the live service or another candidate' % port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = G.AtlasGeometryCandidate(root, path)
        import uvicorn
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()


def _write_gz(path, value):
    with gzip.open(path, 'wt') as f:
        json.dump(value, f, ensure_ascii=False)


def replay(root, name, keys, images, path, capture, limit):
    """In-process candidate over census keys and/or explicit images; plan sealed before the first inference."""
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    run = root / OUT / 'replay' / name
    if run.exists():
        raise SystemExit('Replay run exists; immutable results preserved: ' + str(run))
    doc, rows = _census(root, None)
    items = []
    for key in keys:
        if key not in rows:
            raise SystemExit('Unknown census key ' + key)
        r = rows[key]
        items.append({'key': key, 'input_path': r['input_path'], 'input_sha256': r['input_sha256'], 'roi': r.get('roi'),
                      'baseline_response': r['response'], 'baseline_response_sha256': r['response_sha256'],
                      'baseline_source': r['source']})
    for image in images:
        items.append({'key': 'image:' + Path(image).name, 'input_path': image, 'input_sha256': sha256(Path(image)), 'roi': None})
    if not items or len(items) > limit:
        raise SystemExit('Replay needs 1..%d requests, got %d' % (limit, len(items)))
    for item in items:
        if sha256(Path(item['input_path'])) != item['input_sha256']:
            raise SystemExit('Input bytes differ from census: ' + item['key'])
    descriptor = G.load_descriptor(root, path)
    run.mkdir(parents=True)
    plan = seal({'kind': 'atlas-geometry-repair-v1-replay-plan', 'descriptor': descriptor['checksum'], 'census': doc['checksum'],
                 'items': items, 'capture_nested': capture, 'requests': len(items), 'bottles': 'addressed',
                 'repeat_rule': 'no automatic repeat; a failed request is recorded, not retried', 'fit': False})
    write_json(run / 'started.json', plan)
    pipeline = G.AtlasGeometryCandidate(root, path, capture=capture)
    out = []
    for n, item in enumerate(items):
        data = Path(item['input_path']).read_bytes()
        started = time.perf_counter()
        row = {'n': n, 'key': item['key'], 'input_sha256': hashlib.sha256(data).hexdigest()}
        try:
            result = pipeline.recognize(data, item['roi'], 'addressed')
        except Exception as exc:
            row.update(error=type(exc).__name__ + ': ' + str(exc)[:300], seconds=time.perf_counter() - started)
        else:
            file = run / ('%02d.json.gz' % n)
            _write_gz(file, result)
            row.update(seconds=time.perf_counter() - started, response=str(file.relative_to(root)), response_sha256=sha256(file),
                       summary=_summary(result), recovered=(result.get(G.BLOCK) or {}).get('recovered'))
            if capture and pipeline.last_captured:
                nested = run / ('%02d-nested.json.gz' % n)
                _write_gz(nested, pipeline.last_captured)
                row.update(nested=str(nested.relative_to(root)), nested_sha256=sha256(nested))
        with (run / 'journal.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        out.append(row)
        print(json.dumps({k: row.get(k) for k in ('n', 'key', 'seconds', 'summary', 'recovered', 'error')}, ensure_ascii=False), flush=True)
    write_json(run / 'done.json', seal({'plan': plan['checksum'], 'rows': out, 'runtime': pipeline.manifest['checksum']}))
    return {'run': str(run.relative_to(root)), 'requests': len(out), 'errors': sum(1 for r in out if 'error' in r)}


def _projection(result):
    return [(str(t.get('instance_id')), [round(v) for v in t['bbox']], t.get('bottle_bbox') and [round(v) for v in t['bottle_bbox']],
             bool(t.get('physical_bottle_localized')), t['retrieval'].get('best_candidate')) for t in result.get('targets') or []]


def _saved(root, path):
    value = _gz(root / path)
    return value.get('body', value) if 'http_status' in value and 'body' in value else value


def compare(root, responses):
    """Candidate responses (replay run dir or a JSONL of {key, response}) against the saved baseline of the same bytes."""
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    doc, rows = _census(root, None)
    source = root / responses
    if source.is_dir():
        pairs = [(r['key'], r['response']) for r in _rows(source / 'journal.jsonl') if 'response' in r]
    else:
        pairs = [(r['key'], r['response']) for r in _rows(source)]
    out = []
    for key, response in pairs:
        after = _saved(root, response)
        row = rows.get(key)
        before = _saved(root, row['response']) if row else None
        block = after.get(G.BLOCK) or {}
        item = {'key': key, 'baseline_source': row and row['source'], 'response': response, 'identity': block.get('profile_checksum'),
                'recovered': block.get('recovered'), 'retries': block.get('retries'),
                'second_pass': block.get('second_pass'), 'selected_pass': block.get('selected_pass'),
                'passes': [{k: p.get(k) for k in ('phase', 'targets', 'eligible_observed', 'added', 'canvas_closeup_route', 'ms')}
                           for p in block.get('passes') or []],
                'attempts': [{'phase': e.get('phase')} | {k: a.get(k) for k in ('trial', 'kind', 'outcome', 'gate_context', 'label_score', 'answer')}
                             | {'retry_decision': (a.get('retry') or {}).get('decision'),
                                'retry_targets': (a.get('retry') or {}).get('targets'),
                                'rejected_wine': (a.get('retry') or {}).get('rejected_instances'),
                                'retry_ms': (a.get('retry') or {}).get('ms')}
                             for e in block.get('events') or [] for a in e.get('attempts') or []],
                'after': _summary(after), 'after_targets': _projection(after),
                'slug': after.get('slug'), 'probability': after.get('probability_correct')}
        if before is not None:
            item.update(before=_summary(before), before_targets=_projection(before))
            kept = [t for t in item['after_targets'] if not t[0].startswith('atlas-geometry-')]
            item['existing_targets_unchanged'] = kept == item['before_targets']
            item['changed'] = item['after_targets'] != item['before_targets'] or item['after']['decision'] != item['before']['decision']
        out.append(item)
    doc = seal({'kind': 'atlas-geometry-repair-v1-compare', 'census': doc['checksum'], 'responses': responses, 'rows': out,
                'note': 'open593 baselines are 013c845a responses (older selector): existing-target parity is exact only for '
                        'atlas521 cd0b9910 baselines; no correctness is claimed here'})
    name = 'compare-%s.json' % hashlib.sha256(responses.encode()).hexdigest()[:8]
    write_json(root / OUT / name, doc)
    return {'compare': str(Path(OUT) / name), 'checksum': doc['checksum'], 'rows': len(out),
            'recovered': sum(1 for r in out if r['recovered']),
            'existing_changed': [r['key'] for r in out if r.get('existing_targets_unchanged') is False]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('census')
    for name in ('freeze', 'describe'):
        sub.add_parser(name).add_argument('--descriptor', default=DESCRIPTOR)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=8221)
    s.add_argument('--descriptor', default=DESCRIPTOR)
    r = sub.add_parser('replay')
    r.add_argument('--name', required=True)
    r.add_argument('--key', action='append', default=[])
    r.add_argument('--image', action='append', default=[])
    r.add_argument('--descriptor', default=DESCRIPTOR)
    r.add_argument('--capture-nested', action='store_true')
    r.add_argument('--limit', type=int, default=16)
    c = sub.add_parser('compare')
    c.add_argument('responses')
    args = parser.parse_args()
    if args.command == 'census':
        print(json.dumps(census(ROOT), ensure_ascii=False, indent=1))
    elif args.command == 'freeze':
        print(json.dumps(freeze(ROOT, args.descriptor), ensure_ascii=False, indent=1))
    elif args.command == 'describe':
        print(json.dumps(describe(ROOT, args.descriptor), ensure_ascii=False, indent=1))
    elif args.command == 'serve':
        serve(ROOT, args.port, args.descriptor)
    elif args.command == 'replay':
        print(json.dumps(replay(ROOT, args.name, args.key, args.image, args.descriptor, args.capture_nested, args.limit),
                         ensure_ascii=False))
    else:
        print(json.dumps(compare(ROOT, args.responses), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
