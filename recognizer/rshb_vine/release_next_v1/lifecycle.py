"""Release lifecycle of release next v1 (factory F12, receipts C12) through scripts/recognize.py.

``preflight`` is read-only over config/: it verifies the live atlas_repair_active state (F11, R3, C11, pointer 766d4acd
= bde4fa52), the archived dispatch bytes, the sealed sources of that state and the dispatch drafts, and writes only
runs/release-next-v1/integration/preflight.json. Stage needs the root stage decision, the quality_gate candidate
descriptor and the same live state; it seals a restoration snapshot of those bytes, archives F12/C12 from
runs/release-next-v1/integration/dispatch and writes the immutable dispatch12 chain, gallery migration v8, dispatch
migration v11, the release profile and its manifest. A switch journals its full plan before the first write; the
service on 8175 must be stopped around activate/rollback/recover. The helpers of the atlas/zero-target lifecycles are
reused as loaded, never modified.
"""
import json
from pathlib import Path
import time

from rshb_vine.io import read_json, seal, sha256, verify, write_json
from rshb_vine.atlas_repair_release_v1 import lifecycle as AL
from rshb_vine.release_next_v1 import candidate as C

ZS = AL.ZS
FACTORY, RECOGNIZE, RECEIPTS, CURRENT = C.LIVE_CODE
CODE = (FACTORY, RECOGNIZE, RECEIPTS)
OUT = C.OUT
MANIFEST = 'config/release-next-v1-manifest.json'
DECISION = OUT + '/root-stage-decision.json'
DRAFTS = {FACTORY: OUT + '/dispatch/recognition_factory.py', RECEIPTS: OUT + '/dispatch/recognition_receipts.py'}
PREFLIGHT = OUT + '/preflight.json'
RESTORATION = OUT + '/restoration.json'
RESTORE_DIR = OUT + '/restore'
JOURNAL = OUT + '/journal.jsonl'
F11 = '761155f5fd408486e61e6c873bad73cf76e0fff98fadbce14d0f78f534e3aba7'
C11 = '433de49458ddef3918dc50aa731b0f3e1ade29db2b1ce3abb0c524519570bff1'
ATLAS_POINTER_SHA256 = '766d4acdb2a92b7712df23ba26b9e0ad884df14b23e8c34be436d7be293e6d5b'
ATLAS_STATE = [F11, ZS.R3, C11, ATLAS_POINTER_SHA256]
ARCHIVE = {**AL.ARCHIVE, F11: 'config/dispatch-archive/recognition_factory.761155f5.py',
           C11: 'config/dispatch-archive/recognition_receipts.433de494.py'}
FACTORY_PINS = (*AL.FACTORY_PINS, F11)
COMPOSITES = (*AL.COMPOSITES, AL.DISPATCH11)
ATLAS_BASE = 'config/recognition-b3-only-v1-dispatch11.json'
ATLAS_PROFILE = C.PARENT_PROFILE
ATLAS_MANIFEST = C.PARENT_MANIFEST
CHAIN = (*AL.CHAIN, AL.SYSTEMIC11, AL.TARGET11, AL.GEOMETRY11, ATLAS_BASE, ATLAS_PROFILE, ATLAS_MANIFEST)
DISPATCH12 = 'config/recognition-coherent-v1-dispatch12.json'
SYSTEMIC12 = 'config/recognition-systemic-v2-dispatch12.json'
TARGET12 = 'config/recognition-target-contract-v2-dispatch12.json'
GEOMETRY12 = 'config/recognition-geometry-v2-dispatch12.json'
MIGRATION = 'config/recognition-dispatch-migration-v11.json'
MIGRATIONS_OLD = (*AL.MIGRATIONS_OLD, AL.MIGRATION)
GALLERY_MIGRATION_OLD = 'config/b3-only-v1-gallery-code-migration-v7.json'
GALLERY_MIGRATIONS_OLD = (*AL.GALLERY_MIGRATIONS_OLD, GALLERY_MIGRATION_OLD)
ACTIVE, PREVIOUS = 'release_next_active', 'atlas_repair_active'
ROLLBACK = ('scripts/release_next_v1.py rollback  (restores atlas_repair_active from the sealed snapshot: '
            'factory 7611, receipts 433d, pointer 766d4acd = atlas repair release bde4fa52)')
INVENTORY_ROLE = 'release-next-v1 activation/rollback sources and sealed decisions'


def RN():
    from rshb_vine.release_next_v1 import release
    return release


def staged(root):
    return (Path(root) / MANIFEST).is_file()


def _describe_file(root, path):
    return {'path': path, 'checksum': verify(read_json(root / path))['checksum'], 'sha256': sha256(root / path)}


def _check_live(root):
    """The live atlas_repair_active state, its archives, its sealed switch sources and every older chain document."""
    if ZS._live(root) != ATLAS_STATE:
        raise ValueError('Requires the live atlas_repair_active state (F11, R3, C11, pointer 766d4acd)')
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    atlas = verify(read_json(root / ATLAS_MANIFEST))
    if atlas['states'][PREVIOUS] != ATLAS_STATE:
        raise ValueError('Atlas manifest does not seal the live atlas_repair_active state')
    for path, sha in zip((*CODE, CURRENT), ATLAS_STATE):
        if sha256(root / atlas['sources'][PREVIOUS][path]) != sha:
            raise ValueError('Atlas sealed source differs: ' + atlas['sources'][PREVIOUS][path])
    for path in (*COMPOSITES, *CHAIN, *MIGRATIONS_OLD, *GALLERY_MIGRATIONS_OLD):
        verify(read_json(root / path))
    return C.check_parent(root)


def _check_drafts(root):
    out = {}
    for logical, draft in DRAFTS.items():
        if not (root / draft).is_file():
            raise ValueError('Missing dispatch draft: ' + draft)
        sha = sha256(root / draft)
        if sha in ATLAS_STATE:
            raise ValueError('Draft ' + draft + ' equals the live atlas dispatch')
        out[logical] = {'draft': draft, 'sha256': sha}
    return out


def preflight(root):
    """Read-only readiness record: live state, rollback sources, drafts, candidate descriptor and decision if present."""
    root = Path(root).resolve()
    record = {'kind': 'release-next-v1-preflight', 'ts': time.time(), 'state': None, 'blocked': []}
    try:
        record['parent'] = _check_live(root)
        record['state'] = PREVIOUS
        record['rollback_sources'] = verify(read_json(root / ATLAS_MANIFEST))['sources'][PREVIOUS]
    except (OSError, KeyError, ValueError) as error:
        record['blocked'].append('live: %s' % error)
    try:
        record['dispatch_drafts'] = _check_drafts(root)
    except (OSError, ValueError) as error:
        record['blocked'].append('drafts: %s' % error)
    for name, path in (('candidate_descriptor', C.DESCRIPTOR), ('root_stage_decision', DECISION)):
        record[name] = _describe_file(root, path) if (root / path).is_file() else None
        if record[name] is None:
            record['blocked'].append('%s absent: %s' % (name, path))
    record['staged'] = staged(root)
    record['stageable'] = not record['blocked'] and not record['staged']
    record = seal(record)
    write_json(root / PREFLIGHT, record, replace=True)
    return record


# ---- stage ----

def _check_before(root, decision_checksum):
    _check_live(root)
    descriptor = C.load_descriptor(root, C.DESCRIPTOR)
    if descriptor['admission']['mode'] != 'quality_gate' or descriptor.get('release_admissible') is not True:
        raise ValueError('Only a quality_gate candidate descriptor can be staged')
    gate = verify(read_json(root / descriptor['admission']['path']))
    diagnostic = RN().diagnostic_parity(root, gate, descriptor)
    decision = verify(read_json(root / DECISION))
    if (decision['checksum'] != decision_checksum or decision.get('kind') != RN().DECISION_KIND
            or decision.get('decision') != 'admitted_for_stage' or decision.get('candidate_descriptor') != descriptor['checksum']
            or decision.get('gate') != descriptor['admission']['checksum'] or decision.get('recipe') != descriptor['recipe']):
        raise ValueError('Root decision does not admit this candidate descriptor, gate and recipe for stage')
    _check_drafts(root)
    return verify(read_json(root / ATLAS_PROFILE)), descriptor, decision, diagnostic


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
    for path, sha in zip((*CODE, CURRENT), ATLAS_STATE):
        copy = '%s/%s.%s' % (RESTORE_DIR, Path(path).name, sha[:16])
        ZS._copy_once(root, (root / path).read_bytes(), copy)
        if sha256(root / copy) != sha:
            raise ValueError('Restoration copy differs from live bytes: ' + path)
        files[path] = {'sha256': sha, 'copy': copy}
    restoration = seal({'kind': 'recognition-release-restoration-v1', 'state': PREVIOUS, 'files': files,
                        'profile_checksum': C.PARENT_CHECKSUM, 'profile_path': ATLAS_PROFILE})
    ZS._write_once(root, RESTORATION, restoration)
    return restoration


def _release_pins(root, parent, descriptor, diagnostic):
    pins = dict(parent['pins_sha256'])
    added = {**descriptor['pins_sha256'], **descriptor['sources_sha256'], **RN().sources_sha(root)}
    added.update({p: sha256(root / p) for p in (RN().BASE_PROFILE, ATLAS_PROFILE, C.DESCRIPTOR, DECISION, diagnostic['path'],
                                                 *(r['path'] for r in C.admission_refs(descriptor)))})
    clash = [p for p, s in added.items() if pins.get(p, s) != s]
    if clash:
        raise ValueError('Admission/deploy pins disagree with bde4fa52 pins: ' + ', '.join(clash[:3]))
    pins.update(added)
    live = [p for p in pins if p in C.LIVE_CODE]
    if live:
        raise ValueError('Release profile must not pin live dispatch or pointer: ' + ', '.join(live))
    return pins


def stage(root, decision_checksum):
    from rshb_vine.catalog_training_evaluation_v2 import pins as code_pins
    E = RN()
    if staged(root):
        raise ValueError('Release already staged: ' + MANIFEST)
    parent, descriptor, decision, diagnostic = _check_before(root, decision_checksum)
    restoration = _snapshot(root)
    staged_dispatch = {}
    for logical, draft in DRAFTS.items():
        sha = sha256(root / draft)
        path = ZS.ARCHIVE_NAMES[logical] % sha[:8]
        ZS._copy_once(root, (root / draft).read_bytes(), path)
        staged_dispatch[logical] = {'sha256': sha, 'archive': path}
    F12, C12 = staged_dispatch[FACTORY]['sha256'], staged_dispatch[RECEIPTS]['sha256']

    dispatch11 = verify(read_json(root / AL.DISPATCH11))
    if dispatch11['pins_sha256'].get(FACTORY) != F11 or dispatch11['pins_sha256'].get(RECOGNIZE) != ZS.R3:
        raise ValueError('coherent-v1-dispatch11 does not pin the expected dispatch')
    dispatch12 = ZS._body(dispatch11, 'migrated_from')
    dispatch12.update({
        'name': 'coherent-v1-dispatch12', 'version': 12, 'pins_sha256': dict(dispatch11['pins_sha256'], **{FACTORY: F12}),
        'migrated_from': {'path': AL.DISPATCH11, 'sha256': sha256(root / AL.DISPATCH11), 'checksum': dispatch11['checksum'],
                          'changed_pins': {FACTORY: {'from': F11, 'to': F12}},
                          'behavior': 'identical: factory v12 adds the dispatch12 B3-only base and release-next kinds'},
        'release_status': 'dispatch_migration_staged'})
    dispatch12 = seal(dispatch12)
    ZS._write_once(root, DISPATCH12, dispatch12)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory F11 -> F12)'
    systemic12 = _chain(root, verify(read_json(root / AL.SYSTEMIC11)), 'systemic-ranking-v2-dispatch12', DISPATCH12,
                        dispatch12, (AL.SYSTEMIC11, same))
    ZS._write_once(root, SYSTEMIC12, systemic12)
    target12 = _chain(root, verify(read_json(root / AL.TARGET11)), 'target-contract-v2-dispatch12', SYSTEMIC12, systemic12,
                      (AL.TARGET11, same))
    ZS._write_once(root, TARGET12, target12)
    geometry12 = _chain(root, verify(read_json(root / AL.GEOMETRY11)), 'roskachestvo-geometry-v2-dispatch12', TARGET12,
                        target12, (AL.GEOMETRY11, same))
    ZS._write_once(root, GEOMETRY12, geometry12)

    old = verify(read_json(root / GALLERY_MIGRATION_OLD))
    if old['expected_sha256'] != ZS.F4 or old['replacement_sha256'] != F11:
        raise ValueError('Gallery migration v7 is not F4 -> F11')
    gallery_migration = seal(dict(ZS._body(old), replacement_sha256=F12, version=8,
                                  supersedes={'path': GALLERY_MIGRATION_OLD, 'checksum': old['checksum']}))
    ZS._write_once(root, E.GALLERY_MIGRATION, gallery_migration)

    base11 = verify(read_json(root / ATLAS_BASE))
    own = {p: sha256(root / p) for p in (*E.SOURCES, E.GALLERY_MIGRATION, ATLAS_BASE)}
    base = ZS._body(base11, 'supersedes', 'rollback')
    base.update({'kind': E.BASE_KIND, 'name': 'b3-only-v1-dispatch12',
                 'parent_profile': GEOMETRY12, 'parent_profile_sha256': sha256(root / GEOMETRY12),
                 'parent_profile_checksum': geometry12['checksum'], 'target_profile_checksum': target12['checksum'],
                 'gallery_code_migration_checksum': gallery_migration['checksum'],
                 'pins_sha256': dict(base11['pins_sha256'], **own),
                 'supersedes': {'path': ATLAS_BASE, 'sha256': sha256(root / ATLAS_BASE), 'checksum': base11['checksum'],
                                'same': 'identity fields and equivalence to 83e97cb3; parent chain differs only in '
                                        'dispatch pins F11 -> F12 and gallery migration v7 -> v8'},
                 'release_status': 'development_release', 'rollback': ROLLBACK})
    base = seal(base)
    ZS._write_once(root, E.BASE_PROFILE, base)

    recipe = descriptor['recipe']
    release_pins = _release_pins(root, parent, descriptor, diagnostic)
    release = ZS._body(parent, 'supersedes', 'rollback', 'predecessor_release')
    release.update({
        'kind': E.KIND, 'name': 'release-next-v1',
        'parent_profile': E.BASE_PROFILE, 'parent_profile_sha256': sha256(root / E.BASE_PROFILE),
        'parent_profile_checksum': base['checksum'],
        'composition': parent['composition'] + ' -> release next [' + ', '.join(recipe) + ']',
        'components': dict(parent['components'], release_next_v1=list(recipe)),
        'release_next': {'recipe': list(recipe), 'sources_sha256': E.sources_sha(root),
                         'candidate_descriptor': _describe_file(root, C.DESCRIPTOR),
                         'gate': _describe_file(root, descriptor['admission']['path']),
                         'root_decision': _describe_file(root, DECISION), 'diagnostic_descriptor': diagnostic,
                         'components': descriptor['components'],
                         'release_admitted': True, 'calibrated': False, 'probability': None},
        'predecessor_release': {'path': ATLAS_PROFILE, 'sha256': ATLAS_POINTER_SHA256, 'checksum': C.PARENT_CHECKSUM},
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
                        'replacement_sha256': F12})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 11, 'dispatch_paths': [FACTORY],
                      'scope': 'factory 9d9d/be2c/08b4/f16a/0c3c/6d58/2ab8/3d86/2b91/6c62/7611 -> F12; recognize.py entries of v2..v4 remain valid (6faa live)',
                      'entries': entries})
    ZS._write_once(root, MIGRATION, migration)

    archives = {**ARCHIVE, F12: staged_dispatch[FACTORY]['archive'], C12: staged_dispatch[RECEIPTS]['archive']}
    files = [*E.SOURCES, DECISION, C.DESCRIPTOR, diagnostic['path'], *(r['path'] for r in C.admission_refs(descriptor)),
             *COMPOSITES, *CHAIN,
             DISPATCH12, SYSTEMIC12, TARGET12, GEOMETRY12, E.BASE_PROFILE, E.PROFILE, MIGRATION, *MIGRATIONS_OLD,
             E.GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *release_pins, *archives.values(), RESTORATION,
             *(f['copy'] for f in restoration['files'].values())]
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'release-next-v1',
        'files_sha256_at_stage': {p: sha256(root / p) for p in dict.fromkeys(files)},
        'staged_dispatch': {v['archive']: {'logical_path': k, 'sha256': v['sha256'], 'draft': DRAFTS[k]}
                            for k, v in staged_dispatch.items()},
        'profiles': {'release': {'path': E.PROFILE, 'checksum': release['checksum']},
                     'base_dispatch12': {'path': E.BASE_PROFILE, 'checksum': base['checksum']},
                     'geometry_dispatch12': {'path': GEOMETRY12, 'checksum': geometry12['checksum']},
                     'target_dispatch12': {'path': TARGET12, 'checksum': target12['checksum']},
                     'systemic_dispatch12': {'path': SYSTEMIC12, 'checksum': systemic12['checksum']},
                     'dispatch12': {'path': DISPATCH12, 'checksum': dispatch12['checksum']},
                     'atlas_repair_release': {'path': ATLAS_PROFILE, 'checksum': C.PARENT_CHECKSUM}},
        'recipe': list(recipe), 'model_checksum': release['ablated_ranker_checksum'],
        'states': {PREVIOUS: ATLAS_STATE, ACTIVE: [F12, ZS.R3, C12, sha256(root / E.PROFILE)]},
        'sources': {PREVIOUS: {p: f['copy'] for p, f in restoration['files'].items()},
                    ACTIVE: {FACTORY: archives[F12], RECOGNIZE: restoration['files'][RECOGNIZE]['copy'],
                             RECEIPTS: archives[C12], CURRENT: E.PROFILE}},
        'restoration': {'path': RESTORATION, 'checksum': restoration['checksum']},
        'code_sha_at_stage': code_pins.code_sha(),
        'historical_reproduction': 'atlas repair bde4fa52 and older profiles bind through migration v11 without loading '
                                   'models; rollback restores the sealed 7611/6faa/433d bytes and pointer 766d4acd, after '
                                   'which scripts/atlas_repair_release_v1.py manages older states',
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
    if (manifest['states'][PREVIOUS] != ATLAS_STATE or restoration['checksum'] != manifest['restoration']['checksum']
            or [restoration['files'][p]['sha256'] for p in (*CODE, CURRENT)] != ATLAS_STATE
            or manifest['sources'][PREVIOUS] != {p: f['copy'] for p, f in restoration['files'].items()}):
        raise ValueError('Sealed rollback state differs from F11/R3/C11/766d4acd')
    for path, sha in zip((*CODE, CURRENT), ATLAS_STATE):
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
    expected = manifest['profiles']['release' if target == ACTIVE else 'atlas_repair_release']['checksum']
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
    if args.action == 'preflight':
        record = preflight(root)
        print(json.dumps({'preflight': PREFLIGHT, 'checksum': record['checksum'], 'state': record['state'],
                          'stageable': record['stageable'], 'blocked': record['blocked']}, ensure_ascii=False))
        return
    if args.action == 'stage':
        if not args.decision_checksum:
            raise SystemExit('stage needs --decision-checksum')
        manifest = stage(root, args.decision_checksum)
        print(json.dumps({'manifest': manifest['checksum'], 'state': ZS.state(root, manifest),
                          'release': manifest['profiles']['release']}))
        return
    if not staged(root):
        print(json.dumps({'state': PREVIOUS if ZS._live(root) == ATLAS_STATE else 'unknown', 'staged': False,
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
