"""Evidence-consistency release v1: candidate draft/freeze/describe/serve/replay of fa317ef2 + a root-admitted I/G/O/R
recipe, then stage/status/verify/activate/rollback/recover/inventory of the factory F10 release through scripts/recognize.py.

Candidate (pre-activation): loopback 8199 under the live factory F9, loading the static fa317ef2 profile file; factory,
receipts, recognize.py, the current pointer and the services on 8175/8187 (and candidates on 8196/8197/8198) are never
touched. Freeze needs the root gate (kind evidence-consistency-v1-root-gate, decision quality_gate_passed, recipe an
ordered subset of I, G, O, R, with the sealed protocol/descriptor of each selected component). ``draft`` only previews.
Release: factory F10 and receipts C10 (drafts in runs/evidence-consistency-v1/release/dispatch) add the dispatch10
B3-only base and the evidence-consistency release kinds; stage needs the root stage decision and the live
text_evidence_active state (F9, R3, C9, pointer a1d2c27d), seals a restoration snapshot of those bytes, archives F10/C10
and writes the immutable dispatch10 chain, gallery migration v6, dispatch migration v9, the release profile and its
manifest. A switch journals its full plan before the first write. The service on 8175 must be stopped around
activate/rollback/recover.
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
from rshb_vine.evidence_consistency_release_v1 import candidate as C  # noqa: E402

_spec = importlib.util.spec_from_file_location('_text_evidence_release_v2_script', ROOT / 'scripts/text_evidence_release_v2.py')
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)
ZS = TS.ZS

FACTORY, RECOGNIZE, RECEIPTS, CURRENT = C.LIVE_CODE
CODE = (FACTORY, RECOGNIZE, RECEIPTS)
OUT = 'runs/evidence-consistency-v1/release'
DECISION = OUT + '/root-stage-decision.json'
DRAFTS = {FACTORY: OUT + '/dispatch/recognition_factory.py', RECEIPTS: OUT + '/dispatch/recognition_receipts.py'}
RESTORATION = OUT + '/restoration.json'
RESTORE_DIR = OUT + '/restore'
JOURNAL = OUT + '/journal.jsonl'
F9 = '2b919ac0d2246f625913554eecd8dd1c94232398f7d3e94f969ff19949db6838'
C9 = '2cf9c55ea02431748c571c710603c5ba44a41c19d91c7ca076c8ed5e648a1d5c'
TE_POINTER_SHA256 = 'a1d2c27d6639a90935a100567d39b85bea85263025a194724227d0ed0bc4d385'
TE_STATE = [F9, ZS.R3, C9, TE_POINTER_SHA256]
ARCHIVE = {**TS.ARCHIVE, F9: 'config/dispatch-archive/recognition_factory.2b919ac0.py',
           C9: 'config/dispatch-archive/recognition_receipts.2cf9c55e.py'}
FACTORY_PINS = (*TS.FACTORY_PINS, F9)
COMPOSITES = (*TS.COMPOSITES, TS.DISPATCH9)
TE_BASE = 'config/recognition-b3-only-v1-dispatch9.json'
TE_PROFILE = C.PARENT_PROFILE
TE_MANIFEST = C.PARENT_MANIFEST
CHAIN = (*TS.CHAIN, TS.T.PARENT_PROFILE, TS.ZT_MANIFEST, TS.SYSTEMIC9, TS.TARGET9, TS.GEOMETRY9, TE_BASE, TE_PROFILE, TE_MANIFEST)
DISPATCH10 = 'config/recognition-coherent-v1-dispatch10.json'
SYSTEMIC10 = 'config/recognition-systemic-v2-dispatch10.json'
TARGET10 = 'config/recognition-target-contract-v2-dispatch10.json'
GEOMETRY10 = 'config/recognition-geometry-v2-dispatch10.json'
MIGRATION = 'config/recognition-dispatch-migration-v9.json'
MIGRATIONS_OLD = (*TS.MIGRATIONS_OLD, TS.MIGRATION)
GALLERY_MIGRATION_OLD = 'config/b3-only-v1-gallery-code-migration-v5.json'
GALLERY_MIGRATIONS_OLD = (*TS.GALLERY_MIGRATIONS_OLD, GALLERY_MIGRATION_OLD)
ACTIVE, PREVIOUS = 'evidence_consistency_active', 'text_evidence_active'
ROLLBACK = ('scripts/evidence_consistency_release_v1.py rollback  (restores text_evidence_active from the sealed snapshot: '
            'factory 2b91, receipts 2cf9, pointer a1d2c27d = text-evidence release fa317ef2)')
INVENTORY_ROLE = 'evidence-consistency-v1 activation/rollback sources and sealed decisions'


def ER():
    from rshb_vine.evidence_consistency_release_v1 import release
    return release


def _describe_file(root, path):
    return {'path': path, 'checksum': verify(read_json(root / path))['checksum'], 'sha256': sha256(root / path)}


# ---- candidate ----

def serve(root, port):
    from rshb_vine.target_contract_v2.runtime import create_app
    if port in C.FORBIDDEN_PORTS:
        raise SystemExit('Port %d belongs to the live service or another candidate' % port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', port))
        listener.listen(128)
        pipeline = C.EvidenceConsistencyCandidate(root)
        import uvicorn
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=port)).run(sockets=[listener])
    finally:
        listener.close()


def replay(root, source, output, roi, bottles):
    if output.exists():
        raise SystemExit('replay output already exists; immutable receipt preserved')
    data = source.read_bytes()
    pipeline = C.EvidenceConsistencyCandidate(root)
    started = time.perf_counter()
    result = pipeline.recognize(data, roi, bottles)
    receipt = seal({'kind': 'evidence-consistency-release-v1-candidate-replay', 'image_path': str(source),
                    'request_sha256': hashlib.sha256(data).hexdigest(), 'target_roi': roi, 'target_bottles': bottles,
                    'descriptor_checksum': pipeline.descriptor['checksum'], 'runtime_checksum': pipeline.manifest['checksum'],
                    'result': result, 'seconds': time.perf_counter() - started, 'new_visual_or_OCR_inference': True,
                    'quality_evaluation': False})
    write_json(output, receipt)
    block = result.get(C.K.BLOCK) or {}
    return {'receipt': str(output), 'checksum': receipt['checksum'], 'decision': result['decision'],
            C.K.BLOCK: {k: block.get(k) for k in ('recipe', 'profile_checksum', 'route_events', 'rotated_ocr_performed')}}


# ---- stage ----

def _check_before(root, decision_checksum):
    if ZS._live(root) != TE_STATE:
        raise ValueError('Stage requires the live text_evidence_active state (F9, R3, C9, pointer a1d2c27d)')
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    for path in (*COMPOSITES, *CHAIN, *MIGRATIONS_OLD, *GALLERY_MIGRATIONS_OLD):
        verify(read_json(root / path))
    C.check_parent(root)
    descriptor = C.load_descriptor(root)
    decision = verify(read_json(root / DECISION))
    if (decision['checksum'] != decision_checksum or decision.get('kind') != ER().DECISION_KIND
            or decision.get('decision') != 'admitted_for_stage' or decision.get('candidate_descriptor') != descriptor['checksum']
            or decision.get('gate') != descriptor['gate']['checksum'] or decision.get('recipe') != descriptor['recipe']):
        raise ValueError('Root decision does not admit this candidate descriptor, gate and recipe for stage')
    for draft in DRAFTS.values():
        if not (root / draft).is_file():
            raise ValueError('Missing staged dispatch draft: ' + draft)
    return verify(read_json(root / TE_PROFILE)), descriptor, decision


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
    for path, sha in zip((*CODE, CURRENT), TE_STATE):
        copy = '%s/%s.%s' % (RESTORE_DIR, Path(path).name, sha[:16])
        ZS._copy_once(root, (root / path).read_bytes(), copy)
        if sha256(root / copy) != sha:
            raise ValueError('Restoration copy differs from live bytes: ' + path)
        files[path] = {'sha256': sha, 'copy': copy}
    restoration = seal({'kind': 'recognition-release-restoration-v1', 'state': PREVIOUS, 'files': files,
                        'profile_checksum': C.PARENT_CHECKSUM, 'profile_path': TE_PROFILE})
    ZS._write_once(root, RESTORATION, restoration)
    return restoration


def _release_pins(root, parent, descriptor):
    pins = dict(parent['pins_sha256'])
    added = {**descriptor['pins_sha256'], **descriptor['sources_sha256'], **ER().sources_sha(root)}
    added.update({p: sha256(root / p) for p in (ER().BASE_PROFILE, TE_PROFILE, C.DESCRIPTOR, DECISION,
                                                 *(r['path'] for r in C.admission_refs(descriptor)))})
    clash = [p for p, s in added.items() if pins.get(p, s) != s]
    if clash:
        raise ValueError('Admission/deploy pins disagree with fa317ef2 pins: ' + ', '.join(clash[:3]))
    pins.update(added)
    live = [p for p in pins if p in C.LIVE_CODE]
    if live:
        raise ValueError('Release profile must not pin live dispatch or pointer: ' + ', '.join(live))
    return pins


def stage(root, decision_checksum):
    from rshb_vine.catalog_training_evaluation_v2 import pins as code_pins
    E = ER()
    parent, descriptor, decision = _check_before(root, decision_checksum)
    restoration = _snapshot(root)
    staged = {}
    for logical, draft in DRAFTS.items():
        sha = sha256(root / draft)
        if sha in (F9, C9):
            raise ValueError('Draft ' + draft + ' equals the live text-evidence dispatch')
        path = ZS.ARCHIVE_NAMES[logical] % sha[:8]
        ZS._copy_once(root, (root / draft).read_bytes(), path)
        staged[logical] = {'sha256': sha, 'archive': path}
    F10, C10 = staged[FACTORY]['sha256'], staged[RECEIPTS]['sha256']

    dispatch9 = verify(read_json(root / TS.DISPATCH9))
    if dispatch9['pins_sha256'].get(FACTORY) != F9 or dispatch9['pins_sha256'].get(RECOGNIZE) != ZS.R3:
        raise ValueError('coherent-v1-dispatch9 does not pin the expected dispatch')
    dispatch10 = ZS._body(dispatch9, 'migrated_from')
    dispatch10.update({
        'name': 'coherent-v1-dispatch10', 'version': 10, 'pins_sha256': dict(dispatch9['pins_sha256'], **{FACTORY: F10}),
        'migrated_from': {'path': TS.DISPATCH9, 'sha256': sha256(root / TS.DISPATCH9), 'checksum': dispatch9['checksum'],
                          'changed_pins': {FACTORY: {'from': F9, 'to': F10}},
                          'behavior': 'identical: factory v10 adds the dispatch10 B3-only base and evidence-consistency release kinds'},
        'release_status': 'dispatch_migration_staged'})
    dispatch10 = seal(dispatch10)
    ZS._write_once(root, DISPATCH10, dispatch10)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory F9 -> F10)'
    systemic10 = _chain(root, verify(read_json(root / TS.SYSTEMIC9)), 'systemic-ranking-v2-dispatch10', DISPATCH10, dispatch10,
                        (TS.SYSTEMIC9, same))
    ZS._write_once(root, SYSTEMIC10, systemic10)
    target10 = _chain(root, verify(read_json(root / TS.TARGET9)), 'target-contract-v2-dispatch10', SYSTEMIC10, systemic10,
                      (TS.TARGET9, same))
    ZS._write_once(root, TARGET10, target10)
    geometry10 = _chain(root, verify(read_json(root / TS.GEOMETRY9)), 'roskachestvo-geometry-v2-dispatch10', TARGET10, target10,
                        (TS.GEOMETRY9, same))
    ZS._write_once(root, GEOMETRY10, geometry10)

    old = verify(read_json(root / GALLERY_MIGRATION_OLD))
    if old['expected_sha256'] != ZS.F4 or old['replacement_sha256'] != F9:
        raise ValueError('Gallery migration v5 is not F4 -> F9')
    gallery_migration = seal(dict(ZS._body(old), replacement_sha256=F10, version=6,
                                  supersedes={'path': GALLERY_MIGRATION_OLD, 'checksum': old['checksum']}))
    ZS._write_once(root, E.GALLERY_MIGRATION, gallery_migration)

    base9 = verify(read_json(root / TE_BASE))
    own = {p: sha256(root / p) for p in (*E.SOURCES, E.GALLERY_MIGRATION, TE_BASE)}
    base = ZS._body(base9, 'supersedes', 'rollback')
    base.update({'kind': E.BASE_KIND, 'name': 'b3-only-v1-dispatch10',
                 'parent_profile': GEOMETRY10, 'parent_profile_sha256': sha256(root / GEOMETRY10),
                 'parent_profile_checksum': geometry10['checksum'], 'target_profile_checksum': target10['checksum'],
                 'gallery_code_migration_checksum': gallery_migration['checksum'],
                 'pins_sha256': dict(base9['pins_sha256'], **own),
                 'supersedes': {'path': TE_BASE, 'sha256': sha256(root / TE_BASE), 'checksum': base9['checksum'],
                                'same': 'identity fields and equivalence to 83e97cb3; parent chain differs only in '
                                        'dispatch pins F9 -> F10 and gallery migration v5 -> v6'},
                 'release_status': 'development_release', 'rollback': ROLLBACK})
    base = seal(base)
    ZS._write_once(root, E.BASE_PROFILE, base)

    recipe = descriptor['recipe']
    release_pins = _release_pins(root, parent, descriptor)
    release = ZS._body(parent, 'supersedes', 'rollback', 'predecessor_release')
    release.update({
        'kind': E.KIND, 'name': 'evidence-consistency-release-v1',
        'parent_profile': E.BASE_PROFILE, 'parent_profile_sha256': sha256(root / E.BASE_PROFILE),
        'parent_profile_checksum': base['checksum'],
        'composition': parent['composition'] + ' -> evidence-consistency [' + ', '.join(recipe) + ']',
        'components': dict(parent['components'], evidence_consistency_v1=list(recipe)),
        'evidence_consistency': {'recipe': list(recipe), 'sources_sha256': E.sources_sha(root),
                                 'candidate_descriptor': _describe_file(root, C.DESCRIPTOR),
                                 'gate': _describe_file(root, descriptor['gate']['path']),
                                 'root_decision': _describe_file(root, DECISION),
                                 'admission': descriptor['admission'],
                                 'release_admitted': True, 'calibrated': False, 'probability': None},
        'predecessor_release': {'path': TE_PROFILE, 'sha256': TE_POINTER_SHA256, 'checksum': C.PARENT_CHECKSUM},
        'pins_sha256': release_pins, 'release_status': 'development_release', 'rollback': ROLLBACK})
    release = seal(release)
    ZS._write_once(root, E.PROFILE, release)

    entries = []
    for profile, expected in zip(COMPOSITES, FACTORY_PINS):
        doc = verify(read_json(root / profile))
        if doc['pins_sha256'].get(FACTORY) != expected:
            raise ValueError('Unexpected factory pin: ' + profile)
        entries.append({'profile': profile, 'profile_sha256': sha256(root / profile), 'profile_checksum': doc['checksum'],
                        'logical_path': FACTORY, 'expected_sha256': expected, 'archive': ARCHIVE[expected],
                        'replacement_sha256': F10})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 9, 'dispatch_paths': [FACTORY],
                      'scope': 'factory 9d9d/be2c/08b4/f16a/0c3c/6d58/2ab8/3d86/2b91 -> F10; recognize.py entries of v2..v4 remain valid (6faa live)',
                      'entries': entries})
    ZS._write_once(root, MIGRATION, migration)

    archives = {**ARCHIVE, F10: staged[FACTORY]['archive'], C10: staged[RECEIPTS]['archive']}
    files = [*E.SOURCES, DECISION, C.DESCRIPTOR, *(r['path'] for r in C.admission_refs(descriptor)), *COMPOSITES, *CHAIN,
             DISPATCH10, SYSTEMIC10, TARGET10, GEOMETRY10, E.BASE_PROFILE, E.PROFILE, MIGRATION, *MIGRATIONS_OLD,
             E.GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *release_pins, *archives.values(), RESTORATION,
             *(f['copy'] for f in restoration['files'].values())]
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'evidence-consistency-release-v1',
        'files_sha256_at_stage': {p: sha256(root / p) for p in dict.fromkeys(files)},
        'staged_dispatch': {v['archive']: {'logical_path': k, 'sha256': v['sha256'], 'draft': DRAFTS[k]}
                            for k, v in staged.items()},
        'profiles': {'release': {'path': E.PROFILE, 'checksum': release['checksum']},
                     'base_dispatch10': {'path': E.BASE_PROFILE, 'checksum': base['checksum']},
                     'geometry_dispatch10': {'path': GEOMETRY10, 'checksum': geometry10['checksum']},
                     'target_dispatch10': {'path': TARGET10, 'checksum': target10['checksum']},
                     'systemic_dispatch10': {'path': SYSTEMIC10, 'checksum': systemic10['checksum']},
                     'dispatch10': {'path': DISPATCH10, 'checksum': dispatch10['checksum']},
                     'text_evidence_release': {'path': TE_PROFILE, 'checksum': C.PARENT_CHECKSUM}},
        'recipe': list(recipe), 'model_checksum': release['ablated_ranker_checksum'],
        'states': {PREVIOUS: TE_STATE, ACTIVE: [F10, ZS.R3, C10, sha256(root / E.PROFILE)]},
        'sources': {PREVIOUS: {p: f['copy'] for p, f in restoration['files'].items()},
                    ACTIVE: {FACTORY: archives[F10], RECOGNIZE: restoration['files'][RECOGNIZE]['copy'],
                             RECEIPTS: archives[C10], CURRENT: E.PROFILE}},
        'restoration': {'path': RESTORATION, 'checksum': restoration['checksum']},
        'code_sha_at_stage': code_pins.code_sha(),
        'historical_reproduction': 'text-evidence fa317ef2 and older profiles bind through migration v9 without loading '
                                   'models; rollback restores the sealed 2b91/6faa/2cf9 bytes and pointer a1d2c27d, '
                                   'after which scripts/text_evidence_release_v2.py manages older states',
        'calibrated': False, 'weights_changed': False, 'fit_run': False, 'release_status': 'staged_pending_activation'})
    ZS._write_once(root, E.MANIFEST, manifest)
    return manifest


# ---- switch ----

def _manifest(root):
    manifest = verify(read_json(root / ER().MANIFEST))
    for path, expected in manifest['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Release file changed since stage: ' + path)
    ZS._sources(root, manifest)
    return manifest


def _restore_manifest(root):
    """Rollback/recover need only the sealed state table, the restoration snapshot and the staged sources."""
    manifest = verify(read_json(root / ER().MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    if (manifest['states'][PREVIOUS] != TE_STATE or restoration['checksum'] != manifest['restoration']['checksum']
            or [restoration['files'][p]['sha256'] for p in (*CODE, CURRENT)] != TE_STATE
            or manifest['sources'][PREVIOUS] != {p: f['copy'] for p, f in restoration['files'].items()}):
        raise ValueError('Sealed rollback state differs from F9/R3/C9/a1d2c27d')
    for path, sha in zip((*CODE, CURRENT), TE_STATE):
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
    expected = manifest['profiles']['release' if target == ACTIVE else 'text_evidence_release']['checksum']
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
    manifest = verify(read_json(root / ER().MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    paths = [JOURNAL, RESTORATION, DECISION, C.DESCRIPTOR, *(restoration['files'][p]['copy'] for p in (*CODE, CURRENT))]
    missing = [p for p in paths if not (root / p).is_file()]
    return {'manifest': manifest['checksum'], 'missing': missing,
            'explicit_artifacts': [{'path': p, 'role': INVENTORY_ROLE} for p in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('draft', 'freeze', 'describe', 'serve', 'replay', 'stage', 'status', 'verify',
                                           'activate', 'rollback', 'recover', 'inventory'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--port', type=int)
    parser.add_argument('--gate', help='freeze: root gate decision path')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--roi', help='replay: optional original-coordinate target ROI as a JSON array')
    parser.add_argument('--bottles', choices=('addressed', 'all'), default='addressed')
    parser.add_argument('--decision-checksum', help='stage: checksum of ' + DECISION)
    parser.add_argument('--confirm', help='activate: checksum of the staged release profile')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'draft':
        doc = C.draft(root)
        write_json(root / C.DRAFT, doc)
        print(json.dumps({'draft': C.DRAFT, 'checksum': doc['checksum'], 'servable': False}))
        return
    if args.action == 'freeze':
        if not args.gate:
            raise SystemExit('freeze needs --gate')
        descriptor = C.freeze(root, args.gate)
        ZS._write_once(root, C.DESCRIPTOR, descriptor)
        print(json.dumps({'descriptor': C.DESCRIPTOR, 'checksum': descriptor['checksum'], 'recipe': descriptor['recipe']}))
        return
    if args.action == 'describe':
        descriptor = C.load_descriptor(root) if (root / C.DESCRIPTOR).is_file() else None
        draft = verify(read_json(root / C.DRAFT)) if (root / C.DRAFT).is_file() else None
        print(json.dumps({'candidate_descriptor': descriptor, 'draft_checksum': draft and draft['checksum'],
                          'parent': C.check_parent(root), 'staged_release': (root / ER().MANIFEST).is_file(),
                          'models_loaded': False}, ensure_ascii=False, indent=2))
        return
    if args.action == 'serve':
        serve(root, args.port or C.PORT)
        return
    if args.action == 'replay':
        if args.input is None or args.output is None:
            raise SystemExit('replay needs --input and a new --output')
        roi = json.loads(args.roi) if args.roi is not None else None
        if roi is not None and not isinstance(roi, list):
            raise SystemExit('ROI must be a JSON array')
        print(json.dumps(replay(root, args.input.resolve(), args.output, roi, args.bottles), ensure_ascii=False))
        return
    port = args.port or 8175
    if args.action == 'stage':
        if not args.decision_checksum:
            raise SystemExit('stage needs --decision-checksum')
        manifest = stage(root, args.decision_checksum)
        print(json.dumps({'manifest': manifest['checksum'], 'state': ZS.state(root, manifest),
                          'release': manifest['profiles']['release']}))
        return
    if not (root / ER().MANIFEST).is_file():
        print(json.dumps({'state': PREVIOUS if ZS._live(root) == TE_STATE else 'unknown', 'staged': False,
                          'live': dict(zip((*CODE, CURRENT), ZS._live(root)))}))
        raise SystemExit(0 if args.action == 'status' else 'Release not staged: ' + ER().MANIFEST)
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
