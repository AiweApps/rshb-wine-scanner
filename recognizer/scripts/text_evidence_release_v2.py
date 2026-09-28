"""Text-evidence release v2: candidate freeze/describe/serve/replay of 24b67962 + the admitted D recipe, then
stage/status/verify/activate/rollback/recover of the factory F9 release through scripts/recognize.py.

Candidate (pre-activation): loopback 8197 under the live factory F8; factory, receipts, recognize.py, the current
pointer and the services on 8175/8187 are never touched. Freeze needs the root-admitted v2 protocol and the root
gate decision (kind text-evidence-improve-v2-root-gate, decision quality_gate_passed, recipe [B, D] or [A, B, D, M]).
Release: factory F9 and receipts C9 (drafts in runs/text-evidence-improve-v2/release/dispatch) add the dispatch9
B3-only base and the text-evidence release kinds; stage needs the root stage decision and the live zero_target_active
state (F8, R3, C8, pointer c10ccfed), seals a restoration snapshot of those bytes, archives F9/C9 and writes the
immutable dispatch9 chain, gallery migration v5, dispatch migration v8, the release profile and its manifest. A switch
journals its full plan before the first write. The service on 8175 must be stopped around activate/rollback.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rshb_vine.io import read_json, seal, sha256, verify, write_json  # noqa: E402
from rshb_vine.text_evidence_repair_v2 import runtime as T  # noqa: E402

_spec = importlib.util.spec_from_file_location('_zero_target_release_v1_script', ROOT / 'scripts/zero_target_release_v1.py')
ZS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ZS)

FACTORY, RECOGNIZE, RECEIPTS, CURRENT = T.LIVE_CODE
CODE = (FACTORY, RECOGNIZE, RECEIPTS)
DECISION = T.OUT + '/root-stage-decision.json'
DRAFTS = {FACTORY: T.OUT + '/dispatch/recognition_factory.py', RECEIPTS: T.OUT + '/dispatch/recognition_receipts.py'}
RESTORATION = T.OUT + '/restoration.json'
RESTORE_DIR = T.OUT + '/restore'
JOURNAL = T.OUT + '/journal.jsonl'
F8 = '3d8672504aac374c32571c9552a6e2fa879dfd23383f30ec489871f034ea9f77'
C8 = '8d0326de373ff40939be7beb7bc8f6ae522e0046288545b99d5ab9f53b66d865'
ZT_POINTER_SHA256 = 'c10ccfed2cdb74135ff622127ecaa58b6f1fbad9fd375a455666524bd95859c3'
ZT_STATE = [F8, ZS.R3, C8, ZT_POINTER_SHA256]
ARCHIVE = {**ZS.ARCHIVE, F8: 'config/dispatch-archive/recognition_factory.3d867250.py',
           C8: 'config/dispatch-archive/recognition_receipts.8d0326de.py'}
FACTORY_PINS = (ZS.F1, ZS.F2, ZS.F3, ZS.F4, ZS.F5, ZS.F6, ZS.F7, F8)
COMPOSITES = (*ZS.COMPOSITES, ZS.DISPATCH8)
CHAIN = (ZS.SYSTEMIC7, ZS.TARGET7, ZS.GEOMETRY7, ZS.BASE7, ZS.SYSTEMIC8, ZS.TARGET8, ZS.GEOMETRY8, ZS.BASE8, ZS.V2_RELEASE,
         ZS.V2_MANIFEST)
ZT_MANIFEST = ZS.MANIFEST
DISPATCH9 = 'config/recognition-coherent-v1-dispatch9.json'
SYSTEMIC9 = 'config/recognition-systemic-v2-dispatch9.json'
TARGET9 = 'config/recognition-target-contract-v2-dispatch9.json'
GEOMETRY9 = 'config/recognition-geometry-v2-dispatch9.json'
MIGRATION = 'config/recognition-dispatch-migration-v8.json'
MIGRATIONS_OLD = (*ZS.MIGRATIONS_OLD, ZS.MIGRATION)
GALLERY_MIGRATIONS_OLD = (*ZS.GALLERY_MIGRATIONS_OLD, ZS.GALLERY_MIGRATION)
ACTIVE, PREVIOUS = 'text_evidence_active', 'zero_target_active'
ROLLBACK = ('scripts/text_evidence_release_v2.py rollback  (restores zero_target_active from the sealed snapshot: '
            'factory 3d86, receipts 8d03, pointer c10ccfed = zero-target release 24b67962)')
INVENTORY_ROLE = 'text-evidence-v2 activation/rollback sources and sealed decisions'


def TR():
    from rshb_vine.text_evidence_repair_v2 import release
    return release


def _describe_file(root, path):
    return {'path': path, 'checksum': verify(read_json(root / path))['checksum'], 'sha256': sha256(root / path)}


def serve(root, port):
    from rshb_vine.target_contract_v2.runtime import create_app
    if port in T.FORBIDDEN_PORTS:
        raise SystemExit('Port %d belongs to the live service or an older candidate' % port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = T.TextEvidenceCandidate(root)
        import uvicorn
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()


# ---- stage ----

def _check_before(root, decision_checksum):
    if ZS._live(root) != ZT_STATE:
        raise ValueError('Stage requires the live zero_target_active state (F8, R3, C8, pointer c10ccfed)')
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    for path in (*COMPOSITES, *CHAIN, *MIGRATIONS_OLD, *GALLERY_MIGRATIONS_OLD):
        verify(read_json(root / path))
    T.check_parent(root)
    descriptor = T.load_descriptor(root)
    decision = verify(read_json(root / DECISION))
    if (decision['checksum'] != decision_checksum or decision.get('kind') != 'text-evidence-release-v2-root-decision'
            or decision.get('decision') != 'admitted_for_stage' or decision.get('candidate_descriptor') != descriptor['checksum']
            or decision.get('gate') != descriptor['gate']['checksum'] or decision.get('recipe') != descriptor['recipe']):
        raise ValueError('Root decision does not admit this candidate descriptor, gate and recipe for stage')
    for draft in DRAFTS.values():
        if not (root / draft).is_file():
            raise ValueError('Missing staged dispatch draft: ' + draft)
    return verify(read_json(root / T.PARENT_PROFILE)), descriptor, decision


def _chain(root, source, name, parent_path, parent_doc, what):
    doc = ZS._body(source, 'supersedes', 'rollback')
    doc.update({'name': name, 'parent_profile': parent_path, 'parent_profile_checksum': parent_doc['checksum'],
                'parent_profile_sha256': sha256(root / parent_path),
                'supersedes': {'path': what[0], 'sha256': sha256(root / what[0]), 'checksum': source['checksum'],
                               'same': what[1]},
                'release_status': 'staged_pending_activation', 'rollback': ROLLBACK})
    return seal(doc)


def _snapshot(root):
    files = {}
    for path, sha in zip((*CODE, CURRENT), ZT_STATE):
        copy = '%s/%s.%s' % (RESTORE_DIR, Path(path).name, sha[:16])
        ZS._copy_once(root, (root / path).read_bytes(), copy)
        if sha256(root / copy) != sha:
            raise ValueError('Restoration copy differs from live bytes: ' + path)
        files[path] = {'sha256': sha, 'copy': copy}
    restoration = seal({'kind': 'recognition-release-restoration-v1', 'state': PREVIOUS, 'files': files,
                        'profile_checksum': T.PARENT_CHECKSUM, 'profile_path': T.PARENT_PROFILE})
    ZS._write_once(root, RESTORATION, restoration)
    return restoration


def _release_pins(root, parent, protocol, descriptor, decision):
    pins = dict(parent['pins_sha256'])
    added = {p: s for p, s in protocol['pins_sha256'].items() if p not in T.LIVE_CODE}
    added.update(descriptor['sources_sha256'])
    added.update(TR().sources_sha(root))
    added.update({p: sha256(root / p) for p in (TR().BASE_PROFILE, T.PARENT_PROFILE, T.DESCRIPTOR, descriptor['protocol']['path'],
                                                 descriptor['gate']['path'], DECISION)})
    clash = [p for p, s in added.items() if pins.get(p, s) != s]
    if clash:
        raise ValueError('Protocol/deploy pins disagree with 24b67962 pins: ' + ', '.join(clash[:3]))
    pins.update(added)
    live = [p for p in pins if p in T.LIVE_CODE]
    if live:
        raise ValueError('Release profile must not pin live dispatch or pointer: ' + ', '.join(live))
    return pins


def stage(root, decision_checksum):
    from rshb_vine.catalog_training_evaluation_v2 import pins as code_pins
    R = TR()
    parent, descriptor, decision = _check_before(root, decision_checksum)
    protocol = verify(read_json(root / descriptor['protocol']['path']))
    restoration = _snapshot(root)
    staged = {}
    for logical, draft in DRAFTS.items():
        sha = sha256(root / draft)
        if sha in (F8, C8):
            raise ValueError('Draft ' + draft + ' equals the live zero-target dispatch')
        path = ZS.ARCHIVE_NAMES[logical] % sha[:8]
        ZS._copy_once(root, (root / draft).read_bytes(), path)
        staged[logical] = {'sha256': sha, 'archive': path}
    F9, C9 = staged[FACTORY]['sha256'], staged[RECEIPTS]['sha256']

    dispatch8 = verify(read_json(root / ZS.DISPATCH8))
    if dispatch8['pins_sha256'].get(FACTORY) != F8 or dispatch8['pins_sha256'].get(RECOGNIZE) != ZS.R3:
        raise ValueError('coherent-v1-dispatch8 does not pin the expected dispatch')
    dispatch9 = ZS._body(dispatch8, 'migrated_from')
    dispatch9.update({
        'name': 'coherent-v1-dispatch9', 'version': 9, 'pins_sha256': dict(dispatch8['pins_sha256'], **{FACTORY: F9}),
        'migrated_from': {'path': ZS.DISPATCH8, 'sha256': sha256(root / ZS.DISPATCH8), 'checksum': dispatch8['checksum'],
                          'changed_pins': {FACTORY: {'from': F8, 'to': F9}},
                          'behavior': 'identical: factory v9 adds the dispatch9 B3-only base and text-evidence release kinds'},
        'release_status': 'dispatch_migration_staged'})
    dispatch9 = seal(dispatch9)
    ZS._write_once(root, DISPATCH9, dispatch9)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory F8 -> F9)'
    systemic9 = _chain(root, verify(read_json(root / ZS.SYSTEMIC8)), 'systemic-ranking-v2-dispatch9', DISPATCH9, dispatch9,
                       (ZS.SYSTEMIC8, same))
    ZS._write_once(root, SYSTEMIC9, systemic9)
    target9 = _chain(root, verify(read_json(root / ZS.TARGET8)), 'target-contract-v2-dispatch9', SYSTEMIC9, systemic9,
                     (ZS.TARGET8, same))
    ZS._write_once(root, TARGET9, target9)
    geometry9 = _chain(root, verify(read_json(root / ZS.GEOMETRY8)), 'roskachestvo-geometry-v2-dispatch9', TARGET9, target9,
                       (ZS.GEOMETRY8, same))
    ZS._write_once(root, GEOMETRY9, geometry9)

    old = verify(read_json(root / ZS.GALLERY_MIGRATION))
    if old['expected_sha256'] != ZS.F4 or old['replacement_sha256'] != F8:
        raise ValueError('Gallery migration v4 is not F4 -> F8')
    gallery_migration = seal(dict(ZS._body(old), replacement_sha256=F9, version=5,
                                  supersedes={'path': ZS.GALLERY_MIGRATION, 'checksum': old['checksum']}))
    ZS._write_once(root, R.GALLERY_MIGRATION, gallery_migration)

    base8 = verify(read_json(root / ZS.BASE8))
    own = {p: sha256(root / p) for p in (*R.SOURCES, R.GALLERY_MIGRATION, ZS.BASE8)}
    base = ZS._body(base8, 'supersedes', 'rollback')
    base.update({'kind': R.BASE_KIND, 'name': 'b3-only-v1-dispatch9',
                 'parent_profile': GEOMETRY9, 'parent_profile_sha256': sha256(root / GEOMETRY9),
                 'parent_profile_checksum': geometry9['checksum'], 'target_profile_checksum': target9['checksum'],
                 'gallery_code_migration_checksum': gallery_migration['checksum'],
                 'pins_sha256': dict(base8['pins_sha256'], **own),
                 'supersedes': {'path': ZS.BASE8, 'sha256': sha256(root / ZS.BASE8), 'checksum': base8['checksum'],
                                'same': 'identity fields and equivalence to 83e97cb3; parent chain differs only in '
                                        'dispatch pins F8 -> F9 and gallery migration v4 -> v5'},
                 'release_status': 'development_release', 'rollback': ROLLBACK})
    base = seal(base)
    ZS._write_once(root, R.BASE_PROFILE, base)

    recipe = descriptor['recipe']
    release_pins = _release_pins(root, parent, protocol, descriptor, decision)
    release = ZS._body(parent, 'supersedes', 'rollback', 'predecessor_release')
    release.update({
        'kind': R.KIND, 'name': 'text-evidence-release-v2',
        'parent_profile': R.BASE_PROFILE, 'parent_profile_sha256': sha256(root / R.BASE_PROFILE),
        'parent_profile_checksum': base['checksum'],
        'composition': parent['composition'] + ' -> text-evidence-repair-v2 selector [' + ', '.join(recipe) + ']',
        'components': dict(parent['components'], text_evidence_repair_v2=list(recipe)),
        'text_evidence': {'adapter': 'text-evidence-repair-v2-selection', 'recipe': list(recipe),
                          'tables': descriptor['tables'], 'sources_sha256': R.sources_sha(root),
                          'candidate_descriptor': _describe_file(root, T.DESCRIPTOR),
                          'protocol': _describe_file(root, descriptor['protocol']['path']),
                          'gate': _describe_file(root, descriptor['gate']['path']),
                          'root_decision': _describe_file(root, DECISION),
                          'protocol_pointer_pin': {'path': CURRENT, 'sha256': ZT_POINTER_SHA256},
                          'release_admitted': True, 'calibrated': False, 'probability': None},
        'predecessor_release': {'path': T.PARENT_PROFILE, 'sha256': ZT_POINTER_SHA256, 'checksum': T.PARENT_CHECKSUM},
        'pins_sha256': release_pins, 'release_status': 'development_release', 'rollback': ROLLBACK})
    release = seal(release)
    ZS._write_once(root, R.PROFILE, release)

    entries = []
    for profile, expected in zip(COMPOSITES, FACTORY_PINS):
        doc = verify(read_json(root / profile))
        if doc['pins_sha256'].get(FACTORY) != expected:
            raise ValueError('Unexpected factory pin: ' + profile)
        entries.append({'profile': profile, 'profile_sha256': sha256(root / profile), 'profile_checksum': doc['checksum'],
                        'logical_path': FACTORY, 'expected_sha256': expected, 'archive': ARCHIVE[expected],
                        'replacement_sha256': F9})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 8, 'dispatch_paths': [FACTORY],
                      'scope': 'factory 9d9d/be2c/08b4/f16a/0c3c/6d58/2ab8/3d86 -> F9; recognize.py entries of v2..v4 remain valid (6faa live)',
                      'entries': entries})
    ZS._write_once(root, MIGRATION, migration)

    archives = {**ARCHIVE, F9: staged[FACTORY]['archive'], C9: staged[RECEIPTS]['archive']}
    files = [*R.SOURCES, DECISION, T.DESCRIPTOR, T.PARENT_PROFILE, ZT_MANIFEST, *COMPOSITES, *CHAIN,
             DISPATCH9, SYSTEMIC9, TARGET9, GEOMETRY9, R.BASE_PROFILE, R.PROFILE, MIGRATION, *MIGRATIONS_OLD,
             R.GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *release_pins, *archives.values(), RESTORATION,
             *(f['copy'] for f in restoration['files'].values())]
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'text-evidence-release-v2',
        'files_sha256_at_stage': {p: sha256(root / p) for p in dict.fromkeys(files)},
        'staged_dispatch': {v['archive']: {'logical_path': k, 'sha256': v['sha256'], 'draft': DRAFTS[k]}
                            for k, v in staged.items()},
        'profiles': {'release': {'path': R.PROFILE, 'checksum': release['checksum']},
                     'base_dispatch9': {'path': R.BASE_PROFILE, 'checksum': base['checksum']},
                     'geometry_dispatch9': {'path': GEOMETRY9, 'checksum': geometry9['checksum']},
                     'target_dispatch9': {'path': TARGET9, 'checksum': target9['checksum']},
                     'systemic_dispatch9': {'path': SYSTEMIC9, 'checksum': systemic9['checksum']},
                     'dispatch9': {'path': DISPATCH9, 'checksum': dispatch9['checksum']},
                     'zero_target_release': {'path': T.PARENT_PROFILE, 'checksum': T.PARENT_CHECKSUM}},
        'model_checksum': release['ablated_ranker_checksum'],
        'states': {PREVIOUS: ZT_STATE, ACTIVE: [F9, ZS.R3, C9, sha256(root / R.PROFILE)]},
        'sources': {PREVIOUS: {p: f['copy'] for p, f in restoration['files'].items()},
                    ACTIVE: {FACTORY: archives[F9], RECOGNIZE: restoration['files'][RECOGNIZE]['copy'],
                             RECEIPTS: archives[C9], CURRENT: R.PROFILE}},
        'restoration': {'path': RESTORATION, 'checksum': restoration['checksum']},
        'code_sha_at_stage': code_pins.code_sha(),
        'historical_reproduction': 'zero-target 24b67962 and older profiles bind through migration v8 without loading '
                                   'models; rollback restores the sealed 3d86/6faa/8d03 bytes and pointer c10ccfed, '
                                   'after which scripts/zero_target_release_v1.py manages older states',
        'calibrated': False, 'weights_changed': False, 'fit_run': False, 'release_status': 'staged_pending_activation'})
    ZS._write_once(root, R.MANIFEST, manifest)
    return manifest


# ---- switch ----

def _manifest(root):
    manifest = verify(read_json(root / TR().MANIFEST))
    for path, expected in manifest['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Release file changed since stage: ' + path)
    ZS._sources(root, manifest)
    return manifest


def _restore_manifest(root):
    """Rollback/recover need only the sealed state table, the restoration snapshot and the staged sources."""
    manifest = verify(read_json(root / TR().MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    if (manifest['states'][PREVIOUS] != ZT_STATE or restoration['checksum'] != manifest['restoration']['checksum']
            or [restoration['files'][p]['sha256'] for p in (*CODE, CURRENT)] != ZT_STATE
            or manifest['sources'][PREVIOUS] != {p: f['copy'] for p, f in restoration['files'].items()}):
        raise ValueError('Sealed rollback state differs from F8/R3/C8/c10ccfed')
    for path, sha in zip((*CODE, CURRENT), ZT_STATE):
        if sha256(root / restoration['files'][path]['copy']) != sha:
            raise ValueError('Restoration copy changed: ' + path)
    return manifest


def _pending(root):
    pending = None
    for event in ZS._rows(root / JOURNAL):
        if event['event'] == 'switch_started':
            pending = event
        elif event['event'] in ('switch_done', 'recovered'):
            pending = None
    return pending


def _journal(root, event):
    ZS._append(root / JOURNAL, dict(event, ts=time.time()))


def _switch(root, manifest, target, port):
    if not ZS._port_free(port):
        raise SystemExit('Port %d is serving; stop the service before switching code+profile' % port)
    if _pending(root):
        raise SystemExit('An interrupted switch is recorded; run recover first')
    before = ZS.state(root, manifest)
    if before == 'unknown':
        raise SystemExit('Unknown code/profile state; inspect before switching')
    if before == target:
        raise SystemExit('Already in state ' + target)
    plan = {p: [a, b] for p, a, b in zip((*CODE, CURRENT), manifest['states'][before], manifest['states'][target])}
    _journal(root, {'event': 'switch_started', 'from': before, 'to': target, 'plan': plan, 'manifest': manifest['checksum']})
    ZS._apply(root, manifest, plan)
    after = ZS.state(root, manifest)
    if after != target:
        raise ValueError('Switch verification failed: ' + after + '; run recover')
    described = ZS._describe(root)
    expected = manifest['profiles']['release' if target == ACTIVE else 'zero_target_release']['checksum']
    if described['profile_checksum'] != expected:
        raise ValueError('Pointer describes %s, expected %s; run recover' % (described['profile_checksum'], expected))
    _journal(root, {'event': 'switch_done', 'state': after, 'profile_checksum': described['profile_checksum']})
    return after, described['profile_checksum']


def recover(root, manifest, port):
    if not ZS._port_free(port):
        raise SystemExit('Port %d is serving; stop the service before recovery' % port)
    pending = _pending(root)
    if pending is None:
        raise SystemExit('No interrupted switch recorded')
    plan = pending['plan']
    for path, pair in plan.items():
        if sha256(root / path) not in pair:
            raise SystemExit('Unrecognized bytes outside the recorded switch: ' + path)
    ZS._apply(root, manifest, {p: [b, a] for p, (a, b) in plan.items()})
    after = ZS.state(root, manifest)
    if after != pending['from']:
        raise ValueError('Recovery verification failed: ' + after)
    _journal(root, {'event': 'recovered', 'state': after, 'abandoned': pending['to']})
    return after


def inventory(root):
    """explicit_artifacts rows for config/local-release-v1/inventory-policy.json; root adds them after release."""
    manifest = verify(read_json(root / TR().MANIFEST))
    paths = [JOURNAL, RESTORATION, DECISION, T.DESCRIPTOR, *(verify(read_json(root / RESTORATION))['files'][p]['copy']
                                                            for p in (*CODE, CURRENT))]
    missing = [p for p in paths if not (root / p).is_file()]
    return {'manifest': manifest['checksum'], 'missing': missing,
            'explicit_artifacts': [{'path': p, 'role': INVENTORY_ROLE} for p in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'describe', 'serve', 'replay', 'stage', 'status', 'verify', 'activate',
                                           'rollback', 'recover', 'inventory'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int)
    parser.add_argument('--protocol', help='freeze: root-admitted v2 protocol path')
    parser.add_argument('--gate', help='freeze: root gate decision path')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--decision-checksum', help='stage: checksum of ' + DECISION)
    parser.add_argument('--confirm', help='activate: checksum of the staged release profile')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'freeze':
        if not args.protocol or not args.gate:
            raise SystemExit('freeze needs --protocol and --gate')
        descriptor = T.freeze(root, args.protocol, args.gate)
        ZS._write_once(root, T.DESCRIPTOR, descriptor)
        print(json.dumps({'descriptor': T.DESCRIPTOR, 'checksum': descriptor['checksum'], 'recipe': descriptor['recipe']}))
        return
    if args.action == 'describe':
        descriptor = T.load_descriptor(root) if (root / T.DESCRIPTOR).is_file() else None
        print(json.dumps({'candidate_descriptor': descriptor, 'parent': T.check_parent(root),
                          'staged_release': (root / TR().MANIFEST).is_file(), 'models_loaded': False},
                         ensure_ascii=False, indent=2))
        return
    if args.action == 'serve':
        serve(root, args.port or T.PORT)
        return
    if args.action == 'replay':
        if args.input is None or args.output is None or args.output.exists():
            raise SystemExit('replay needs --input and a new --output')
        data = args.input.read_bytes()
        pipeline = T.TextEvidenceCandidate(root)
        started = time.perf_counter()
        result = pipeline.recognize(data)
        write_json(args.output, seal({'kind': 'text-evidence-release-v2-candidate-replay', 'image_path': str(args.input),
                                      'request_sha256': hashlib.sha256(data).hexdigest(),
                                      'runtime_checksum': pipeline.manifest['checksum'], 'result': result,
                                      'seconds': time.perf_counter() - started, 'quality_evaluation': False}))
        print(json.dumps({'receipt': str(args.output), 'decision': result['decision'],
                          T.BLOCK: {k: v for k, v in result.get(T.BLOCK, {}).items() if k != 'selector_identity'}}))
        return
    port = args.port or 8175
    if args.action == 'stage':
        if not args.decision_checksum:
            raise SystemExit('stage needs --decision-checksum')
        manifest = stage(root, args.decision_checksum)
        print(json.dumps({'manifest': manifest['checksum'], 'state': ZS.state(root, manifest),
                          'release': manifest['profiles']['release']}))
        return
    if not (root / TR().MANIFEST).is_file():
        print(json.dumps({'state': PREVIOUS if ZS._live(root) == ZT_STATE else 'unknown', 'staged': False,
                          'live': dict(zip((*CODE, CURRENT), ZS._live(root)))}))
        raise SystemExit(0 if args.action == 'status' else 'Release not staged: ' + TR().MANIFEST)
    if args.action == 'inventory':
        print(json.dumps(inventory(root), ensure_ascii=False, indent=2))
        return
    manifest = _manifest(root) if args.action in ('verify', 'activate') else _restore_manifest(root)
    if args.action in ('status', 'verify'):
        current, pending = ZS.state(root, manifest), _pending(root)
        print(json.dumps({'state': current, 'manifest': manifest['checksum'], 'pending_switch': pending,
                          'live': dict(zip((*CODE, CURRENT), ZS._live(root)))}))
        if args.action == 'verify' and (current == 'unknown' or pending):
            raise SystemExit('verify failed')
        return
    if args.action == 'recover':
        print(json.dumps({'state': recover(root, manifest, port)}))
        return
    if args.action == 'activate':
        if args.confirm != manifest['profiles']['release']['checksum']:
            raise SystemExit('--confirm must name the staged release profile checksum')
        print(json.dumps(_switch(root, manifest, ACTIVE, port)))
        return
    print(json.dumps(_switch(root, manifest, PREVIOUS, port)))


if __name__ == '__main__':
    main()
