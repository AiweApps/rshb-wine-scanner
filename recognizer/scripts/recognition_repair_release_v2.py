"""Stage, verify, activate, roll back and recover the recognition-repair v2 release through scripts/recognize.py.

Two dispatch files change together: factory (F6 -> F7: dispatch7 B3-only base and repair-v2 release kinds) and
recognition_receipts.py (C6 -> C7: dispatch7 chain + release, reads migration v1..v6); scripts/recognize.py stays
6faa. The dispatch6 composite pins factory F6, so the release runs over byte-identical dispatch7 copies of the
geometry-v2 chain; every older composite binds through config/recognition-dispatch-migration-v6.json and the B4343
gallery receipt through config/b3-only-v1-gallery-code-migration-v3.json. F7/C7 are drafted under
runs/recognition-repair-v2/release/dispatch and staged only as config/dispatch-archive copies. The state is the
tuple (factory, recognize, receipts, pointer) sha. Stage seals a restoration snapshot of the live repair-v1 bytes
(F6, R3, C6, pointer 1e9b6f13), so rollback and recover read only that snapshot, the staged archive and the sealed
manifest, never candidate modules. A switch journals its full plan before the first write; recover restores the
pre-switch bytes of an interrupted recorded switch only. ``stage`` needs the frozen v2 candidate profile and
refuses unless the live state is repair_v1_active.
"""
import argparse
import inspect
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

from rshb_vine.io import read_json, seal, sha256, verify, write_json  # noqa: E402

FACTORY = 'rshb_vine/recognition_factory.py'
RECOGNIZE = 'scripts/recognize.py'
RECEIPTS = 'rshb_vine/recognition_receipts.py'
CURRENT = 'config/recognition-current.json'
CODE = (FACTORY, RECOGNIZE, RECEIPTS)
COMPOSITES = ('config/recognition-coherent-v1.json', 'config/recognition-coherent-v1-dispatch2.json',
              'config/recognition-coherent-v1-dispatch3.json', 'config/recognition-coherent-v1-dispatch4.json',
              'config/recognition-coherent-v1-dispatch5.json', 'config/recognition-coherent-v1-dispatch6.json')
DISPATCH6 = COMPOSITES[-1]
SYSTEMIC6 = 'config/recognition-systemic-v2-dispatch6.json'
TARGET6 = 'config/recognition-target-contract-v2-dispatch6.json'
GEOMETRY6 = 'config/recognition-geometry-v2-dispatch6.json'
BASE6 = 'config/recognition-b3-only-v1-dispatch6.json'
DISPATCH7 = 'config/recognition-coherent-v1-dispatch7.json'
SYSTEMIC7 = 'config/recognition-systemic-v2-dispatch7.json'
TARGET7 = 'config/recognition-target-contract-v2-dispatch7.json'
GEOMETRY7 = 'config/recognition-geometry-v2-dispatch7.json'
BASE7 = 'config/recognition-b3-only-v1-dispatch7.json'
B3_RELEASE = 'config/recognition-b3-only-v1-release.json'
V1_RELEASE = 'config/recognition-repair-v1-release.json'
V1_MANIFEST = 'config/recognition-repair-v1-release-manifest.json'
RELEASE = 'config/recognition-repair-v2-release.json'
MIGRATION = 'config/recognition-dispatch-migration-v6.json'
MIGRATIONS_OLD = tuple('config/recognition-dispatch-migration-v%d.json' % v for v in range(1, 6))
GALLERY_MIGRATION_OLD = 'config/b3-only-v1-gallery-code-migration-v2.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration-v3.json'
GALLERY_MIGRATIONS_OLD = ('config/b3-only-v1-gallery-code-migration.json', GALLERY_MIGRATION_OLD)
MANIFEST = 'config/recognition-repair-v2-release-manifest.json'
RELEASE_CODE = ('rshb_vine/recognition_repair_v2/release.py', 'scripts/recognition_repair_release_v2.py')
INHERITED_CODE = ('rshb_vine/recognition_repair_v1/release.py',)
OUT = 'runs/recognition-repair-v2/release'
DRAFTS = {FACTORY: OUT + '/dispatch/recognition_factory.py', RECEIPTS: OUT + '/dispatch/recognition_receipts.py'}
ARCHIVE_NAMES = {FACTORY: 'config/dispatch-archive/recognition_factory.%s.py',
                 RECEIPTS: 'config/dispatch-archive/recognition_receipts.%s.py'}
RESTORATION = OUT + '/restoration.json'
RESTORE_DIR = OUT + '/restore'
JOURNAL = OUT + '/journal.jsonl'
F1 = '9d9d3ac003ff0cfc13f4d15f5d81beddea99e5c42f9278e411463fab87aea87b'
F2 = 'be2c15d4b8380d89791e22fb6e984f1111455b45b6f78172f5f33a103b9f778f'
F3 = '08b4ab3fbcc31278458ce52c5bb0e17e0098ca06796b91016bfe4485da4b5a35'
F4 = 'f16a78f8ec91296c0f50f1ba1c9309798704a13125a85dcab748d893836723b0'
F5 = '0c3c7f20121817b5e0c1dbf22d8c4bc4af46a91cc872783d40c8fdf1a914e8a6'
F6 = '6d58e20208ced9e9c0a53b6dac193b138fa30dd810a790d575b12278f878dd43'
R3 = '6faabf40f02e265e73b26d6b813a88c54fc4cd42daaa90516002e924843b4a77'
C6 = '4ebcdfa4959ff308ea80390b99c6e790c8f0fd1e57042432fd87301565f8edfb'
ARCHIVE = {F1: 'config/dispatch-archive/recognition_factory.9d9d3ac0.py',
           F2: 'config/dispatch-archive/recognition_factory.be2c15d4.py',
           F3: 'config/dispatch-archive/recognition_factory.08b4ab3f.py',
           F4: 'config/dispatch-archive/recognition_factory.f16a78f8.py',
           F5: 'config/dispatch-archive/recognition_factory.0c3c7f20.py',
           F6: 'config/dispatch-archive/recognition_factory.6d58e202.py',
           R3: 'config/dispatch-archive/recognize.6faabf40.py',
           C6: 'config/dispatch-archive/recognition_receipts.4ebcdfa4.py'}
V1_POINTER_SHA256 = '1e9b6f13eb1705a3abe3d44f3c0dbfb1546237423553ca2b79703c296c4d4966'
V1_CHECKSUM = '51eedd7ff128961c582194f3b52e0bd14a987299010889a9a9dffbe4595dcbe7'
V1_STATE = [F6, R3, C6, V1_POINTER_SHA256]
ROLLBACK = ('scripts/recognition_repair_release_v2.py rollback  (restores repair_v1_active from the sealed snapshot: '
            'factory 6d58, receipts 4ebc, pointer 1e9b6f13 = repair-v1 release 51eedd7f)')


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


def _check_before(root, candidate_path, candidate_checksum):
    if _live(root) != V1_STATE:
        raise ValueError('Stage requires the live repair_v1_active state (F6, R3, C6, pointer 1e9b6f13)')
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    for path in (*COMPOSITES, SYSTEMIC6, TARGET6, GEOMETRY6, BASE6, B3_RELEASE, *MIGRATIONS_OLD, *GALLERY_MIGRATIONS_OLD):
        verify(read_json(root / path))
    v1 = verify(read_json(root / V1_RELEASE))
    if v1['checksum'] != V1_CHECKSUM or sha256(root / V1_RELEASE) != V1_POINTER_SHA256:
        raise ValueError('Frozen repair-v1 release 51eedd7f differs')
    for path, expected in verify(read_json(root / V1_MANIFEST))['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Repair-v1 release file changed since its stage: ' + path)
    candidate = verify(read_json(root / candidate_path))
    if candidate['checksum'] != candidate_checksum:
        raise ValueError('Candidate profile checksum differs from --candidate-checksum')
    from rshb_vine.recognition_repair_v2 import runtime as C2
    from rshb_vine.recognition_repair_v2 import release as RR
    candidate = C2.load_profile(root, candidate_path)
    if 'repair' not in inspect.signature(C2.RecognitionRepairV2.__init__).parameters:
        raise ValueError('RecognitionRepairV2 lacks the keyword-only repair parameter')
    if (candidate['parent_release']['checksum'] != V1_CHECKSUM
            or candidate['repair_v1_profile']['checksum'] != RR.REPAIR_V1_CHECKSUM
            or candidate['repair_v1_profile']['path'] != v1['combined_profile']):
        raise ValueError('Candidate is not built over repair-v1 51eedd7f / 013c845a')
    for name, draft in DRAFTS.items():
        if not (root / draft).is_file():
            raise ValueError('Missing staged dispatch draft: ' + draft)
    return v1, candidate


def _chain(root, source, name, parent_path, parent_doc, what):
    doc = _body(source, 'supersedes', 'rollback')
    doc.update({'name': name, 'parent_profile': parent_path, 'parent_profile_checksum': parent_doc['checksum'],
                'parent_profile_sha256': sha256(root / parent_path),
                'supersedes': {'path': what[0], 'sha256': sha256(root / what[0]), 'checksum': source['checksum'],
                               'same': what[1]},
                'release_status': 'staged_pending_activation', 'rollback': ROLLBACK})
    return seal(doc)


def _stage_dispatch(root):
    staged = {}
    for logical, draft in DRAFTS.items():
        data = (root / draft).read_bytes()
        digest = sha256(root / draft)
        if digest in (F6, C6):
            raise ValueError('Draft ' + draft + ' equals the live repair-v1 dispatch')
        path = ARCHIVE_NAMES[logical] % digest[:8]
        _copy_once(root, data, path)
        staged[logical] = {'sha256': digest, 'archive': path}
    return staged


def _snapshot(root):
    """Content-addressed copies of the live repair-v1 bytes; rollback needs nothing else."""
    files = {}
    for path, digest in zip((*CODE, CURRENT), V1_STATE):
        copy = '%s/%s.%s' % (RESTORE_DIR, Path(path).name, digest[:16])
        _copy_once(root, (root / path).read_bytes(), copy)
        if sha256(root / copy) != digest:
            raise ValueError('Restoration copy differs from live bytes: ' + path)
        files[path] = {'sha256': digest, 'copy': copy}
    restoration = seal({'kind': 'recognition-release-restoration-v1', 'state': 'repair_v1_active', 'files': files,
                        'profile_checksum': V1_CHECKSUM, 'profile_path': V1_RELEASE})
    _write_once(root, RESTORATION, restoration)
    return restoration


def stage(root, candidate_path, candidate_checksum):
    v1, candidate = _check_before(root, candidate_path, candidate_checksum)
    from rshb_vine.catalog_training_evaluation_v2 import pins
    from rshb_vine.recognition_repair_v2 import release as RR
    restoration = _snapshot(root)
    staged = _stage_dispatch(root)
    F7, C7 = staged[FACTORY]['sha256'], staged[RECEIPTS]['sha256']

    dispatch6 = verify(read_json(root / DISPATCH6))
    if dispatch6['pins_sha256'].get(FACTORY) != F6 or dispatch6['pins_sha256'].get(RECOGNIZE) != R3:
        raise ValueError('coherent-v1-dispatch6 does not pin the expected dispatch')
    dispatch7 = _body(dispatch6, 'migrated_from')
    dispatch7.update({
        'name': 'coherent-v1-dispatch7', 'version': 7,
        'pins_sha256': dict(dispatch6['pins_sha256'], **{FACTORY: F7}),
        'migrated_from': {'path': DISPATCH6, 'sha256': sha256(root / DISPATCH6), 'checksum': dispatch6['checksum'],
                          'changed_pins': {FACTORY: {'from': F6, 'to': F7}},
                          'behavior': 'identical: factory v7 adds the dispatch7 B3-only base and repair-v2 release kinds'},
        'release_status': 'dispatch_migration_staged'})
    dispatch7 = seal(dispatch7)
    _write_once(root, DISPATCH7, dispatch7)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory F6 -> F7)'
    systemic7 = _chain(root, verify(read_json(root / SYSTEMIC6)), 'systemic-ranking-v2-dispatch7', DISPATCH7,
                       dispatch7, (SYSTEMIC6, same))
    _write_once(root, SYSTEMIC7, systemic7)
    target7 = _chain(root, verify(read_json(root / TARGET6)), 'target-contract-v2-dispatch7', SYSTEMIC7, systemic7,
                     (TARGET6, same))
    _write_once(root, TARGET7, target7)
    geometry7 = _chain(root, verify(read_json(root / GEOMETRY6)), 'roskachestvo-geometry-v2-dispatch7', TARGET7,
                       target7, (GEOMETRY6, same))
    _write_once(root, GEOMETRY7, geometry7)

    b3 = verify(read_json(root / B3_RELEASE))
    old = verify(read_json(root / GALLERY_MIGRATION_OLD))
    receipt = verify(read_json(root / b3['arm_inputs']['gallery_dir'] / 'receipt.json'))
    if receipt['code_sha256'].get(FACTORY) != F4 or old['gallery_receipt_checksum'] != receipt['checksum']:
        raise ValueError('B4343 gallery receipt does not bind factory f16a')
    gallery_migration = seal(dict(_body(old), replacement_sha256=F7, version=3,
                                  supersedes={'path': GALLERY_MIGRATION_OLD, 'checksum': old['checksum']}))
    _write_once(root, GALLERY_MIGRATION, gallery_migration)

    base6 = verify(read_json(root / BASE6))
    base_pins = dict(b3['pins_sha256'])
    base_pins.update({p: sha256(root / p) for p in (*INHERITED_CODE, *RELEASE_CODE, B3_RELEASE, GALLERY_MIGRATION)})
    base = _body(b3, 'equivalent_candidate', 'rollback', 'evidence')
    base.update({'kind': RR.BASE_KIND, 'name': 'b3-only-v1-dispatch7',
                 'parent_profile': GEOMETRY7, 'parent_profile_sha256': sha256(root / GEOMETRY7),
                 'parent_profile_checksum': geometry7['checksum'], 'target_profile_checksum': target7['checksum'],
                 'gallery_code_migration_checksum': gallery_migration['checksum'], 'pins_sha256': base_pins,
                 'equivalent_release': base6['equivalent_release'],
                 'supersedes': {'path': BASE6, 'sha256': sha256(root / BASE6), 'checksum': base6['checksum'],
                                'same': 'identity fields and equivalence to 83e97cb3; parent chain differs only in '
                                        'dispatch pins F6 -> F7 and gallery migration v2 -> v3'},
                 'release_status': 'development_release', 'rollback': ROLLBACK})
    base = seal(base)
    _write_once(root, BASE7, base)

    producer = RR._producer_pin(root, candidate)
    release_pins = dict(candidate['pins_sha256'])
    release_pins.update({p: sha256(root / p) for p in (*RELEASE_CODE, *INHERITED_CODE, candidate_path, BASE7,
                                                       v1['combined_profile'], RR.PRODUCER_PIN)})
    stages = [s['name'] for s in candidate['stages']]
    release = seal({
        'kind': RR.KIND, 'name': 'recognition-repair-v2-release',
        'parent_profile': BASE7, 'parent_profile_sha256': sha256(root / BASE7), 'parent_profile_checksum': base['checksum'],
        'candidate_profile': candidate_path, 'candidate_profile_checksum': candidate['checksum'],
        'repair_v1_profile': v1['combined_profile'], 'repair_v1_profile_checksum': v1['combined_profile_checksum'],
        'stages': stages, 'producer_pin': producer,
        'components': dict(v1['components'], repair_v2_stages=stages,
                           selector_registry_checksum=candidate['selector_registry_checksum']),
        'composition': 'recognition-repair-v1 013c845a -> recognition-repair-v2 (' + ', '.join(stages) + ')',
        'predecessor_release': {'path': V1_RELEASE, 'sha256': V1_POINTER_SHA256, 'checksum': V1_CHECKSUM},
        'parent_ranker_checksum': base['parent_ranker_checksum'], 'ablated_ranker_checksum': base['ablated_ranker_checksum'],
        'pins_sha256': release_pins, 'routes': base['routes'],
        'serve': 'scripts/recognize.py serve | replay [--roi R] [--bottles all] | describe',
        'calibrated': False, 'probability': None, 'public_slug': 'never confirmed', 'weights_changed': False,
        'fit_run': False, 'independent_test': False, 'release_status': 'development_release', 'rollback': ROLLBACK})
    _write_once(root, RELEASE, release)

    entries = []
    for profile, expected in zip(COMPOSITES, (F1, F2, F3, F4, F5, F6)):
        doc = verify(read_json(root / profile))
        if doc['pins_sha256'].get(FACTORY) != expected:
            raise ValueError('Unexpected factory pin: ' + profile)
        entries.append({'profile': profile, 'profile_sha256': sha256(root / profile), 'profile_checksum': doc['checksum'],
                        'logical_path': FACTORY, 'expected_sha256': expected, 'archive': ARCHIVE[expected],
                        'replacement_sha256': F7})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 6, 'dispatch_paths': [FACTORY],
                      'scope': 'factory 9d9d/be2c/08b4/f16a/0c3c/6d58 -> F7; recognize.py entries of v2..v4 remain valid (6faa live)',
                      'entries': entries})
    _write_once(root, MIGRATION, migration)

    archives = {**ARCHIVE, F7: staged[FACTORY]['archive'], C7: staged[RECEIPTS]['archive']}
    files = [*RELEASE_CODE, *INHERITED_CODE, candidate_path, *candidate['pins_sha256'], V1_RELEASE, V1_MANIFEST,
             *COMPOSITES, SYSTEMIC6, TARGET6, GEOMETRY6, BASE6, B3_RELEASE, DISPATCH7, SYSTEMIC7, TARGET7, GEOMETRY7,
             BASE7, RELEASE, MIGRATION, GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *MIGRATIONS_OLD, RR.PRODUCER_PIN,
             *archives.values(), RESTORATION, *(f['copy'] for f in restoration['files'].values())]
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'recognition-repair-v2-release',
        'files_sha256_at_stage': {p: sha256(root / p) for p in dict.fromkeys(files)},
        'staged_dispatch': {v['archive']: {'logical_path': k, 'sha256': v['sha256'], 'draft': DRAFTS[k]}
                            for k, v in staged.items()},
        'profiles': {'release': {'path': RELEASE, 'checksum': release['checksum']},
                     'candidate': {'path': candidate_path, 'checksum': candidate['checksum']},
                     'base_dispatch7': {'path': BASE7, 'checksum': base['checksum']},
                     'geometry_dispatch7': {'path': GEOMETRY7, 'checksum': geometry7['checksum']},
                     'target_dispatch7': {'path': TARGET7, 'checksum': target7['checksum']},
                     'systemic_dispatch7': {'path': SYSTEMIC7, 'checksum': systemic7['checksum']},
                     'dispatch7': {'path': DISPATCH7, 'checksum': dispatch7['checksum']},
                     'repair_v1_release': {'path': V1_RELEASE, 'checksum': V1_CHECKSUM}},
        'model_checksum': release['ablated_ranker_checksum'],
        'states': {'repair_v1_active': V1_STATE, 'repair_v2_active': [F7, R3, C7, sha256(root / RELEASE)]},
        'sources': {'repair_v1_active': {p: f['copy'] for p, f in restoration['files'].items()},
                    'repair_v2_active': {FACTORY: archives[F7], RECOGNIZE: restoration['files'][RECOGNIZE]['copy'],
                                         RECEIPTS: archives[C7], CURRENT: RELEASE}},
        'restoration': {'path': RESTORATION, 'checksum': restoration['checksum']},
        'required_runtime_assets': [MIGRATION, GALLERY_MIGRATION, *GALLERY_MIGRATIONS_OLD, *MIGRATIONS_OLD,
                                    *archives.values()],
        'code_sha_at_stage': pins.code_sha(),
        'historical_reproduction': '51eedd7f, 83e97cb3 and older profiles bind through migration v6 without loading '
                                   'models; rollback restores the sealed 6d58/6faa/4ebc bytes and pointer 1e9b6f13, '
                                   'after which scripts/recognition_repair_release_v1.py manages older states',
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
        for path, digest in zip((*CODE, CURRENT), manifest['states'][name]):
            if sha256(root / paths[path]) != digest:
                raise ValueError('Switch source bytes differ: ' + paths[path])


def _restore_manifest(root):
    """Rollback/recover need only the sealed state table, the restoration snapshot and the staged v2 sources, so an
    edited candidate source never blocks returning to repair_v1_active; activation keeps _manifest."""
    manifest = verify(read_json(root / MANIFEST))
    restoration = verify(read_json(root / RESTORATION))
    if (manifest['states']['repair_v1_active'] != V1_STATE or restoration['checksum'] != manifest['restoration']['checksum']
            or [restoration['files'][p]['sha256'] for p in (*CODE, CURRENT)] != V1_STATE
            or manifest['sources']['repair_v1_active'] != {p: f['copy'] for p, f in restoration['files'].items()}):
        raise ValueError('Sealed rollback state differs from F6/R3/C6/1e9b6f13')
    for path, digest in zip((*CODE, CURRENT), V1_STATE):
        if sha256(root / restoration['files'][path]['copy']) != digest:
            raise ValueError('Restoration copy changed: ' + path)
    return manifest


def state(root, manifest=None):
    manifest = manifest or _manifest(root)
    live = _live(root)
    return next((name for name, pair in manifest['states'].items() if pair == live), 'unknown')


def _pending(root):
    path = root / JOURNAL
    pending = None
    if path.is_file():
        for line in path.read_text().splitlines():
            event = json.loads(line)
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
    path = root / JOURNAL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(dict(event, ts=time.time()), sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _describe(root):
    out = subprocess.run([sys.executable, str(root / RECOGNIZE), 'describe'], cwd=root, capture_output=True,
                         text=True, check=True)
    return json.loads(out.stdout)


def _source(root, manifest, path, digest):
    for name, pair in manifest['states'].items():
        for p, d in zip((*CODE, CURRENT), pair):
            if p == path and d == digest:
                source = root / manifest['sources'][name][path]
                if sha256(source) != digest:
                    raise ValueError('Switch source bytes differ: ' + str(source))
                return source.read_bytes()
    raise ValueError('No sealed source carries ' + path + ' ' + digest)


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
    _journal(root, {'event': 'switch_started', 'from': before, 'to': target, 'plan': plan,
                    'manifest': manifest['checksum']})
    _apply(root, manifest, plan)
    after = state(root, manifest)
    if after != target:
        raise ValueError('Switch verification failed: ' + after + '; run recover')
    described = _describe(root)
    expected = manifest['profiles']['release' if target == 'repair_v2_active' else 'repair_v1_release']['checksum']
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'status', 'verify', 'activate', 'rollback', 'recover'))
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root (a temporary copy for rehearsal)')
    parser.add_argument('--candidate', help='stage: frozen repair-v2 candidate profile path')
    parser.add_argument('--candidate-checksum', help='stage: its sealed checksum')
    parser.add_argument('--confirm', help='activate: checksum of the release profile')
    parser.add_argument('--port', type=int, default=8175)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'stage':
        if not args.candidate or not args.candidate_checksum:
            raise SystemExit('stage needs --candidate and --candidate-checksum')
        print(json.dumps({'manifest': stage(root, args.candidate, args.candidate_checksum)['checksum'], 'state': state(root)}))
        return
    if not (root / MANIFEST).is_file():
        live = dict(zip((*CODE, CURRENT), _live(root)))
        print(json.dumps({'state': 'repair_v1_active' if _live(root) == V1_STATE else 'unknown', 'staged': False,
                          'live': live}))
        raise SystemExit(0 if args.action == 'status' else 'Release not staged: ' + MANIFEST)
    manifest = _manifest(root) if args.action in ('verify', 'activate') else _restore_manifest(root)
    if args.action in ('status', 'verify'):
        current, pending = state(root, manifest), _pending(root)
        print(json.dumps({'state': current, 'manifest': manifest['checksum'],
                          'pending_switch': pending, 'live': dict(zip((*CODE, CURRENT), _live(root)))}))
        if args.action == 'verify' and (current == 'unknown' or pending):
            raise SystemExit('verify failed: ' + ('interrupted switch recorded; run recover' if pending
                                                  else 'unknown code/profile state'))
        return
    if args.action == 'recover':
        print(json.dumps({'state': recover(root, manifest, args.port)}))
        return
    if args.action == 'activate':
        if args.confirm != manifest['profiles']['release']['checksum']:
            raise SystemExit('--confirm must name the staged release profile checksum')
        print(json.dumps(_switch(root, manifest, 'repair_v2_active', args.port)))
        return
    print(json.dumps(_switch(root, manifest, 'repair_v1_active', args.port)))


if __name__ == '__main__':
    main()
