"""Release lifecycle of the atlas repair release v1 (factory F11, receipts C11) through scripts/recognize.py.

Stage needs the root stage decision, the quality_gate candidate descriptor and the live evidence_consistency_active
state (F10, R3, C10, pointer 4b21e0fe = cd0b9910); it seals a restoration snapshot of those bytes, archives F11/C11
from runs/atlas-repair-v1/release/dispatch and writes the immutable dispatch11 chain, gallery migration v7, dispatch
migration v10, the release profile and its manifest. A switch journals its full plan before the first write; the
service on 8175 must be stopped around activate/rollback/recover. The helpers of the zero-target/evidence-consistency
lifecycles are reused as loaded, never modified.
"""
import importlib.util
import json
from pathlib import Path
import time

from rshb_vine.io import read_json, seal, sha256, verify
from rshb_vine.atlas_repair_release_v1 import candidate as C

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('_evidence_consistency_release_v1_script',
                                               _ROOT / 'scripts/evidence_consistency_release_v1.py')
ES = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ES)
ZS = ES.ZS

FACTORY, RECOGNIZE, RECEIPTS, CURRENT = C.LIVE_CODE
CODE = (FACTORY, RECOGNIZE, RECEIPTS)
OUT = C.OUT
MANIFEST = 'config/atlas-repair-release-v1-manifest.json'
DECISION = OUT + '/root-stage-decision.json'
DRAFTS = {FACTORY: OUT + '/dispatch/recognition_factory.py', RECEIPTS: OUT + '/dispatch/recognition_receipts.py'}
RESTORATION = OUT + '/restoration.json'
RESTORE_DIR = OUT + '/restore'
JOURNAL = OUT + '/journal.jsonl'
F10 = '6c62d2806c6da0c0b34ee35d6366eef8acccf1f631d6fd8a0ce53280d83e1733'
C10 = '2e8efa529083c65a7bf2259c68b84afcda16f15581147b24369726f8073684b3'
EC_POINTER_SHA256 = '4b21e0fe5f14f44f01fe9e80e1508b7b189ab5711126f3896f8c8cf2b3940499'
EC_STATE = [F10, ZS.R3, C10, EC_POINTER_SHA256]
ARCHIVE = {**ES.ARCHIVE, F10: 'config/dispatch-archive/recognition_factory.6c62d280.py',
           C10: 'config/dispatch-archive/recognition_receipts.2e8efa52.py'}
FACTORY_PINS = (*ES.FACTORY_PINS, F10)
COMPOSITES = (*ES.COMPOSITES, ES.DISPATCH10)
EC_BASE = 'config/recognition-b3-only-v1-dispatch10.json'
EC_PROFILE = C.PARENT_PROFILE
EC_MANIFEST = C.PARENT_MANIFEST
CHAIN = (*ES.CHAIN, ES.SYSTEMIC10, ES.TARGET10, ES.GEOMETRY10, EC_BASE, EC_PROFILE, EC_MANIFEST)
DISPATCH11 = 'config/recognition-coherent-v1-dispatch11.json'
SYSTEMIC11 = 'config/recognition-systemic-v2-dispatch11.json'
TARGET11 = 'config/recognition-target-contract-v2-dispatch11.json'
GEOMETRY11 = 'config/recognition-geometry-v2-dispatch11.json'
MIGRATION = 'config/recognition-dispatch-migration-v10.json'
MIGRATIONS_OLD = (*ES.MIGRATIONS_OLD, ES.MIGRATION)
GALLERY_MIGRATION_OLD = 'config/b3-only-v1-gallery-code-migration-v6.json'
GALLERY_MIGRATIONS_OLD = (*ES.GALLERY_MIGRATIONS_OLD, GALLERY_MIGRATION_OLD)
ACTIVE, PREVIOUS = 'atlas_repair_active', 'evidence_consistency_active'
ROLLBACK = ('scripts/atlas_repair_release_v1.py rollback  (restores evidence_consistency_active from the sealed snapshot: '
            'factory 6c62, receipts 2e8e, pointer 4b21e0fe = evidence-consistency release cd0b9910)')
INVENTORY_ROLE = 'atlas-repair-v1 activation/rollback sources and sealed decisions'


def AR():
    from rshb_vine.atlas_repair_release_v1 import release
    return release


def staged(root):
    return (Path(root) / MANIFEST).is_file()


def _describe_file(root, path):
    return {'path': path, 'checksum': verify(read_json(root / path))['checksum'], 'sha256': sha256(root / path)}


# ---- stage ----

def _check_before(root, decision_checksum):
    if ZS._live(root) != EC_STATE:
        raise ValueError('Stage requires the live evidence_consistency_active state (F10, R3, C10, pointer 4b21e0fe)')
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    for path in (*COMPOSITES, *CHAIN, *MIGRATIONS_OLD, *GALLERY_MIGRATIONS_OLD):
        verify(read_json(root / path))
    C.check_parent(root)
    descriptor = C.load_descriptor(root, C.DESCRIPTOR)
    if descriptor['admission']['mode'] != 'quality_gate' or descriptor.get('release_admissible') is not True:
        raise ValueError('Only a quality_gate candidate descriptor can be staged')
    gate = verify(read_json(root / descriptor['admission']['path']))
    diagnostic = AR().diagnostic_parity(root, gate, descriptor)
    decision = verify(read_json(root / DECISION))
    if (decision['checksum'] != decision_checksum or decision.get('kind') != AR().DECISION_KIND
            or decision.get('decision') != 'admitted_for_stage' or decision.get('candidate_descriptor') != descriptor['checksum']
            or decision.get('gate') != descriptor['admission']['checksum'] or decision.get('recipe') != descriptor['recipe']):
        raise ValueError('Root decision does not admit this candidate descriptor, gate and recipe for stage')
    for draft in DRAFTS.values():
        if not (root / draft).is_file():
            raise ValueError('Missing staged dispatch draft: ' + draft)
    return verify(read_json(root / EC_PROFILE)), descriptor, decision, diagnostic


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
    for path, sha in zip((*CODE, CURRENT), EC_STATE):
        copy = '%s/%s.%s' % (RESTORE_DIR, Path(path).name, sha[:16])
        ZS._copy_once(root, (root / path).read_bytes(), copy)
        if sha256(root / copy) != sha:
            raise ValueError('Restoration copy differs from live bytes: ' + path)
        files[path] = {'sha256': sha, 'copy': copy}
    restoration = seal({'kind': 'recognition-release-restoration-v1', 'state': PREVIOUS, 'files': files,
                        'profile_checksum': C.PARENT_CHECKSUM, 'profile_path': EC_PROFILE})
    ZS._write_once(root, RESTORATION, restoration)
    return restoration


def _release_pins(root, parent, descriptor, diagnostic):
    pins = dict(parent['pins_sha256'])
    added = {**descriptor['pins_sha256'], **descriptor['sources_sha256'], **AR().sources_sha(root)}
    added.update({p: sha256(root / p) for p in (AR().BASE_PROFILE, EC_PROFILE, C.DESCRIPTOR, DECISION, diagnostic['path'],
                                                 *(r['path'] for r in C.admission_refs(descriptor)))})
    clash = [p for p, s in added.items() if pins.get(p, s) != s]
    if clash:
        raise ValueError('Admission/deploy pins disagree with cd0b9910 pins: ' + ', '.join(clash[:3]))
    pins.update(added)
    live = [p for p in pins if p in C.LIVE_CODE]
    if live:
        raise ValueError('Release profile must not pin live dispatch or pointer: ' + ', '.join(live))
    return pins


def stage(root, decision_checksum):
    from rshb_vine.catalog_training_evaluation_v2 import pins as code_pins
    E = AR()
    parent, descriptor, decision, diagnostic = _check_before(root, decision_checksum)
    restoration = _snapshot(root)
    staged_dispatch = {}
    for logical, draft in DRAFTS.items():
        sha = sha256(root / draft)
        if sha in (F10, C10):
            raise ValueError('Draft ' + draft + ' equals the live evidence-consistency dispatch')
        path = ZS.ARCHIVE_NAMES[logical] % sha[:8]
        ZS._copy_once(root, (root / draft).read_bytes(), path)
        staged_dispatch[logical] = {'sha256': sha, 'archive': path}
    F11, C11 = staged_dispatch[FACTORY]['sha256'], staged_dispatch[RECEIPTS]['sha256']

    dispatch10 = verify(read_json(root / ES.DISPATCH10))
    if dispatch10['pins_sha256'].get(FACTORY) != F10 or dispatch10['pins_sha256'].get(RECOGNIZE) != ZS.R3:
        raise ValueError('coherent-v1-dispatch10 does not pin the expected dispatch')
    dispatch11 = ZS._body(dispatch10, 'migrated_from')
    dispatch11.update({
        'name': 'coherent-v1-dispatch11', 'version': 11, 'pins_sha256': dict(dispatch10['pins_sha256'], **{FACTORY: F11}),
        'migrated_from': {'path': ES.DISPATCH10, 'sha256': sha256(root / ES.DISPATCH10), 'checksum': dispatch10['checksum'],
                          'changed_pins': {FACTORY: {'from': F10, 'to': F11}},
                          'behavior': 'identical: factory v11 adds the dispatch11 B3-only base and atlas repair release kinds'},
        'release_status': 'dispatch_migration_staged'})
    dispatch11 = seal(dispatch11)
    ZS._write_once(root, DISPATCH11, dispatch11)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory F10 -> F11)'
    systemic11 = _chain(root, verify(read_json(root / ES.SYSTEMIC10)), 'systemic-ranking-v2-dispatch11', DISPATCH11,
                        dispatch11, (ES.SYSTEMIC10, same))
    ZS._write_once(root, SYSTEMIC11, systemic11)
    target11 = _chain(root, verify(read_json(root / ES.TARGET10)), 'target-contract-v2-dispatch11', SYSTEMIC11, systemic11,
                      (ES.TARGET10, same))
    ZS._write_once(root, TARGET11, target11)
    geometry11 = _chain(root, verify(read_json(root / ES.GEOMETRY10)), 'roskachestvo-geometry-v2-dispatch11', TARGET11,
                        target11, (ES.GEOMETRY10, same))
    ZS._write_once(root, GEOMETRY11, geometry11)

    old = verify(read_json(root / GALLERY_MIGRATION_OLD))
    if old['expected_sha256'] != ZS.F4 or old['replacement_sha256'] != F10:
        raise ValueError('Gallery migration v6 is not F4 -> F10')
    gallery_migration = seal(dict(ZS._body(old), replacement_sha256=F11, version=7,
                                  supersedes={'path': GALLERY_MIGRATION_OLD, 'checksum': old['checksum']}))
    ZS._write_once(root, E.GALLERY_MIGRATION, gallery_migration)

    base10 = verify(read_json(root / EC_BASE))
    own = {p: sha256(root / p) for p in (*E.SOURCES, E.GALLERY_MIGRATION, EC_BASE)}
    base = ZS._body(base10, 'supersedes', 'rollback')
    base.update({'kind': E.BASE_KIND, 'name': 'b3-only-v1-dispatch11',
                 'parent_profile': GEOMETRY11, 'parent_profile_sha256': sha256(root / GEOMETRY11),
                 'parent_profile_checksum': geometry11['checksum'], 'target_profile_checksum': target11['checksum'],
                 'gallery_code_migration_checksum': gallery_migration['checksum'],
                 'pins_sha256': dict(base10['pins_sha256'], **own),
                 'supersedes': {'path': EC_BASE, 'sha256': sha256(root / EC_BASE), 'checksum': base10['checksum'],
                                'same': 'identity fields and equivalence to 83e97cb3; parent chain differs only in '
                                        'dispatch pins F10 -> F11 and gallery migration v6 -> v7'},
                 'release_status': 'development_release', 'rollback': ROLLBACK})
    base = seal(base)
    ZS._write_once(root, E.BASE_PROFILE, base)

    recipe = descriptor['recipe']
    release_pins = _release_pins(root, parent, descriptor, diagnostic)
    release = ZS._body(parent, 'supersedes', 'rollback', 'predecessor_release')
    release.update({
        'kind': E.KIND, 'name': 'atlas-repair-release-v1',
        'parent_profile': E.BASE_PROFILE, 'parent_profile_sha256': sha256(root / E.BASE_PROFILE),
        'parent_profile_checksum': base['checksum'],
        'composition': parent['composition'] + ' -> atlas repair [' + ', '.join(recipe) + ']',
        'components': dict(parent['components'], atlas_repair_v1=list(recipe)),
        'atlas_repair': {'recipe': list(recipe), 'sources_sha256': E.sources_sha(root),
                         'candidate_descriptor': _describe_file(root, C.DESCRIPTOR),
                         'gate': _describe_file(root, descriptor['admission']['path']),
                         'root_decision': _describe_file(root, DECISION), 'diagnostic_descriptor': diagnostic,
                         'components': descriptor['components'],
                         'release_admitted': True, 'calibrated': False, 'probability': None},
        'predecessor_release': {'path': EC_PROFILE, 'sha256': EC_POINTER_SHA256, 'checksum': C.PARENT_CHECKSUM},
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
                        'replacement_sha256': F11})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 10, 'dispatch_paths': [FACTORY],
                      'scope': 'factory 9d9d/be2c/08b4/f16a/0c3c/6d58/2ab8/3d86/2b91/6c62 -> F11; recognize.py entries of v2..v4 remain valid (6faa live)',
                      'entries': entries})
    ZS._write_once(root, MIGRATION, migration)

    archives = {**ARCHIVE, F11: staged_dispatch[FACTORY]['archive'], C11: staged_dispatch[RECEIPTS]['archive']}
    files = [*E.SOURCES, DECISION, C.DESCRIPTOR, diagnostic['path'], *(r['path'] for r in C.admission_refs(descriptor)),
             *COMPOSITES, *CHAIN,
             DISPATCH11, SYSTEMIC11, TARGET11, GEOMETRY11, E.BASE_PROFILE, E.PROFILE, MIGRATION, *MIGRATIONS_OLD,
             E.GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *release_pins, *archives.values(), RESTORATION,
             *(f['copy'] for f in restoration['files'].values())]
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'atlas-repair-release-v1',
        'files_sha256_at_stage': {p: sha256(root / p) for p in dict.fromkeys(files)},
        'staged_dispatch': {v['archive']: {'logical_path': k, 'sha256': v['sha256'], 'draft': DRAFTS[k]}
                            for k, v in staged_dispatch.items()},
        'profiles': {'release': {'path': E.PROFILE, 'checksum': release['checksum']},
                     'base_dispatch11': {'path': E.BASE_PROFILE, 'checksum': base['checksum']},
                     'geometry_dispatch11': {'path': GEOMETRY11, 'checksum': geometry11['checksum']},
                     'target_dispatch11': {'path': TARGET11, 'checksum': target11['checksum']},
                     'systemic_dispatch11': {'path': SYSTEMIC11, 'checksum': systemic11['checksum']},
                     'dispatch11': {'path': DISPATCH11, 'checksum': dispatch11['checksum']},
                     'evidence_consistency_release': {'path': EC_PROFILE, 'checksum': C.PARENT_CHECKSUM}},
        'recipe': list(recipe), 'model_checksum': release['ablated_ranker_checksum'],
        'states': {PREVIOUS: EC_STATE, ACTIVE: [F11, ZS.R3, C11, sha256(root / E.PROFILE)]},
        'sources': {PREVIOUS: {p: f['copy'] for p, f in restoration['files'].items()},
                    ACTIVE: {FACTORY: archives[F11], RECOGNIZE: restoration['files'][RECOGNIZE]['copy'],
                             RECEIPTS: archives[C11], CURRENT: E.PROFILE}},
        'restoration': {'path': RESTORATION, 'checksum': restoration['checksum']},
        'code_sha_at_stage': code_pins.code_sha(),
        'historical_reproduction': 'evidence-consistency cd0b9910 and older profiles bind through migration v10 without '
                                   'loading models; rollback restores the sealed 6c62/6faa/2e8e bytes and pointer 4b21e0fe, '
                                   'after which scripts/evidence_consistency_release_v1.py manages older states',
        'calibrated': False, 'weights_changed': False, 'fit_run': False, 'release_status': 'staged_pending_activation'})
    ZS._write_once(root, MANIFEST, manifest)
    return manifest


# ---- switch ----

def _manifest(root):
    manifest = verify(read_json(root / MANIFEST))
    for path, expected in manifest['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Release file changed since stage: ' + path)
    ZS._sources(root, manifest)
    return manifest


def _restore_manifest(root):
    """Rollback/recover need only the sealed state table, the restoration snapshot and the staged sources."""
    manifest = verify(read_json(root / MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    if (manifest['states'][PREVIOUS] != EC_STATE or restoration['checksum'] != manifest['restoration']['checksum']
            or [restoration['files'][p]['sha256'] for p in (*CODE, CURRENT)] != EC_STATE
            or manifest['sources'][PREVIOUS] != {p: f['copy'] for p, f in restoration['files'].items()}):
        raise ValueError('Sealed rollback state differs from F10/R3/C10/4b21e0fe')
    for path, sha in zip((*CODE, CURRENT), EC_STATE):
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
    expected = manifest['profiles']['release' if target == ACTIVE else 'evidence_consistency_release']['checksum']
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
    manifest = verify(read_json(root / MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    paths = [JOURNAL, RESTORATION, DECISION, C.DESCRIPTOR, *(restoration['files'][p]['copy'] for p in (*CODE, CURRENT))]
    missing = [p for p in paths if not (root / p).is_file()]
    return {'manifest': manifest['checksum'], 'missing': missing,
            'explicit_artifacts': [{'path': p, 'role': INVENTORY_ROLE} for p in paths]}


def main(root, args):
    root = Path(root).resolve()
    port = args.port or 8175
    if args.action == 'stage':
        if not args.decision_checksum:
            raise SystemExit('stage needs --decision-checksum')
        manifest = stage(root, args.decision_checksum)
        print(json.dumps({'manifest': manifest['checksum'], 'state': ZS.state(root, manifest),
                          'release': manifest['profiles']['release']}))
        return
    if not staged(root):
        print(json.dumps({'state': PREVIOUS if ZS._live(root) == EC_STATE else 'unknown', 'staged': False,
                          'live': dict(zip((*CODE, CURRENT), ZS._live(root)))}))
        raise SystemExit(0 if args.action == 'status' else 'Release not staged: ' + MANIFEST)
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
    if args.action == 'rollback':
        print(json.dumps(_switch(root, manifest, PREVIOUS, port)))
        return
    raise SystemExit('Unsupported release action: ' + args.action)
