"""Stage, verify, activate, roll back and recover the B3-only v1 release through scripts/recognize.py.

Two dispatch files change together: factory (f16a -> F5: b3-only release kind, result/selector checksum) and
recognition_receipts.py (7cfc -> C5: dispatch5 chain + release, reads migration v1..v4); scripts/recognize.py stays
6faa. The dispatch4 composite pins factory f16a, so the release runs over byte-identical dispatch5 copies of the
geometry-v2 chain; old profiles bind through config/recognition-dispatch-migration-v4.json, the B4343 gallery
receipt through config/b3-only-v1-gallery-code-migration.json. New bytes are staged only in config/dispatch-archive.
The state is the tuple (factory, recognize, receipts, pointer) sha. A switch journals its full plan before the first
write; recover restores the pre-switch bytes of an interrupted recorded switch and refuses any other bytes.
"""
import argparse
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
COHERENT = 'config/recognition-coherent-v1.json'
DISPATCH2 = 'config/recognition-coherent-v1-dispatch2.json'
DISPATCH3 = 'config/recognition-coherent-v1-dispatch3.json'
DISPATCH4 = 'config/recognition-coherent-v1-dispatch4.json'
SYSTEMIC4 = 'config/recognition-systemic-v2-dispatch4.json'
TARGET4 = 'config/recognition-target-contract-v2-dispatch4.json'
GEOMETRY = 'config/recognition-geometry-v2-release.json'
DISPATCH5 = 'config/recognition-coherent-v1-dispatch5.json'
SYSTEMIC5 = 'config/recognition-systemic-v2-dispatch5.json'
TARGET5 = 'config/recognition-target-contract-v2-dispatch5.json'
GEOMETRY5 = 'config/recognition-geometry-v2-dispatch5.json'
RELEASE = 'config/recognition-b3-only-v1-release.json'
CANDIDATE = 'config/recognition-b3-only-v1-candidate-v3.json'
MIGRATION = 'config/recognition-dispatch-migration-v4.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration.json'
MIGRATIONS_OLD = {'config/recognition-dispatch-migration-v1.json':
                  '2463756b282b3315dab4eed01150d542a1ed7f775ccb209f4c8b3c07d1e17544',
                  'config/recognition-dispatch-migration-v2.json':
                  '45676ed5465c51198dd6450a784335a94eb5a48879fa1780c8797c37e6abb469',
                  'config/recognition-dispatch-migration-v3.json':
                  '8defe4da0b058d171d04b4f4b0b0dc2bdc9989d00edd5609c9f658b953d8810f'}
MANIFEST = 'config/b3-only-v1-release-manifest.json'
GEOMETRY_MANIFEST = 'config/geometry-v2-release-manifest.json'
EVIDENCE = ('runs/b3-only-integration-v1/root-acceptance-before-candidate.json',
            'runs/b3-only-integration-v1/review-decision.json')
RELEASE_CODE = ('rshb_vine/b3_only_v1/release.py', 'scripts/b3_only_release.py')
F1 = '9d9d3ac003ff0cfc13f4d15f5d81beddea99e5c42f9278e411463fab87aea87b'
F2 = 'be2c15d4b8380d89791e22fb6e984f1111455b45b6f78172f5f33a103b9f778f'
F3 = '08b4ab3fbcc31278458ce52c5bb0e17e0098ca06796b91016bfe4485da4b5a35'
F4 = 'f16a78f8ec91296c0f50f1ba1c9309798704a13125a85dcab748d893836723b0'
F5 = '0c3c7f20121817b5e0c1dbf22d8c4bc4af46a91cc872783d40c8fdf1a914e8a6'
R3 = '6faabf40f02e265e73b26d6b813a88c54fc4cd42daaa90516002e924843b4a77'
C4 = '7cfc9d8d937be32db448f2a264572e5f4be3faeee36fa1ebbac028fccf7adf3f'
C5 = '9090924bbc38c74f33c7d0606df9cb5d8c07762bc49e09f42baa64fde489b3e8'
ARCHIVE = {F1: 'config/dispatch-archive/recognition_factory.9d9d3ac0.py',
           F2: 'config/dispatch-archive/recognition_factory.be2c15d4.py',
           F3: 'config/dispatch-archive/recognition_factory.08b4ab3f.py',
           F4: 'config/dispatch-archive/recognition_factory.f16a78f8.py',
           F5: 'config/dispatch-archive/recognition_factory.0c3c7f20.py',
           R3: 'config/dispatch-archive/recognize.6faabf40.py',
           C4: 'config/dispatch-archive/recognition_receipts.7cfc9d8d.py',
           C5: 'config/dispatch-archive/recognition_receipts.9090924b.py'}
LOGICAL = {F5: FACTORY, C5: RECEIPTS}
GEOMETRY_SHA256 = '226017a26ee9bbd43a41968d70bbccc66fb6078437881a9fcb8f50248c063e3a'
CANDIDATE_CHECKSUM = '30671615f3ccd5a8ac2824c60d46d4fbf65b08746cbfa8156756badfcbf63429'
EXPECTED = {COHERENT: '37f1f5d494e461bbca191beb8c4c07ef388913743e7a629afc088799d29a18e5',
            DISPATCH2: 'c59858d55308f4f57b5735b2aa513b333cfc0c173dc9e87cf89f9464ef04aecc',
            DISPATCH3: 'dd5bf76450b9bda6100b0d75a89645382c74bc7235829f6ba1bc5eaffc4b7872',
            DISPATCH4: '87ca1f058f97bc857cd448e0c097894eb6fbc916f3f4c4d86f78828593e13876',
            GEOMETRY: GEOMETRY_SHA256, **MIGRATIONS_OLD}
JOURNAL = 'runs/b3-only-integration-v1/release/journal.jsonl'
ROLLBACK = 'scripts/b3_only_release.py rollback  (restores geometry_v2_active: factory f16a, receipts 7cfc, pointer 226017a2)'


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


def _body(doc, *drop):
    return {k: v for k, v in doc.items() if k not in ('checksum', *drop)}


def _check_before(root):
    for path, expected in EXPECTED.items():
        if sha256(root / path) != expected:
            raise ValueError('Unexpected before-bytes: ' + path)
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archived dispatch bytes differ: ' + path)
    candidate = verify(read_json(root / CANDIDATE))
    if candidate['checksum'] != CANDIDATE_CHECKSUM or candidate['parent_profile_sha256'] != GEOMETRY_SHA256:
        raise ValueError('Frozen B3-only candidate v3 differs')
    for path, expected in candidate['pins_sha256'].items():
        if sha256(root / path) != expected:
            raise ValueError('Candidate v3 pinned file changed: ' + path)
    decision = verify(read_json(root / EVIDENCE[1]))
    if (decision['result'] != 'candidate_accepted_for_versioned_development_activation'
            or decision['candidate_profile']['checksum'] != CANDIDATE_CHECKSUM or not decision['ml_frozen']):
        raise ValueError('Review decision does not admit this candidate')
    return candidate


def _chain(root, source, name, parent_path, parent_doc, what):
    doc = _body(source, 'supersedes', 'rollback')
    doc.update({'name': name, 'parent_profile': parent_path, 'parent_profile_checksum': parent_doc['checksum'],
                'parent_profile_sha256': sha256(root / parent_path),
                'supersedes': {'path': what[0], 'sha256': sha256(root / what[0]), 'checksum': source['checksum'],
                               'same': what[1]},
                'release_status': 'staged_pending_activation', 'rollback': ROLLBACK})
    return seal(doc)


def stage(root):
    candidate = _check_before(root)
    from rshb_vine.catalog_training_evaluation_v2 import pins
    dispatch4 = verify(read_json(root / DISPATCH4))
    if dispatch4['pins_sha256'].get(FACTORY) != F4 or dispatch4['pins_sha256'].get(RECOGNIZE) != R3:
        raise ValueError('coherent-v1-dispatch4 does not pin the expected dispatch')
    dispatch5 = _body(dispatch4)
    dispatch5.update({
        'name': 'coherent-v1-dispatch5', 'version': 5,
        'pins_sha256': dict(dispatch4['pins_sha256'], **{FACTORY: F5}),
        'migrated_from': {'path': DISPATCH4, 'sha256': sha256(root / DISPATCH4), 'checksum': dispatch4['checksum'],
                          'changed_pins': {FACTORY: {'from': F4, 'to': F5}},
                          'behavior': 'identical: factory v5 adds the b3-only release kind'},
        'release_status': 'dispatch_migration_staged'})
    dispatch5 = seal(dispatch5)
    _write_once(root, DISPATCH5, dispatch5)
    same = 'model, sources and pins; parent differs only in dispatch pins (factory f16a -> F5)'
    systemic5 = _chain(root, verify(read_json(root / SYSTEMIC4)), 'systemic-ranking-v2-dispatch5', DISPATCH5,
                       dispatch5, (SYSTEMIC4, same))
    _write_once(root, SYSTEMIC5, systemic5)
    target5 = _chain(root, verify(read_json(root / TARGET4)), 'target-contract-v2-dispatch5', SYSTEMIC5, systemic5,
                     (TARGET4, same))
    _write_once(root, TARGET5, target5)
    geometry = verify(read_json(root / GEOMETRY))
    geometry5 = _chain(root, geometry, 'roskachestvo-geometry-v2-dispatch5', TARGET5, target5, (GEOMETRY, same))
    _write_once(root, GEOMETRY5, geometry5)
    folder = root / candidate['arm_inputs']['gallery_dir']
    receipt = verify(read_json(folder / 'receipt.json'))
    if receipt['code_sha256'].get(FACTORY) != F4:
        raise ValueError('B4343 gallery receipt does not bind factory f16a')
    gallery_migration = seal({
        'kind': 'b3-only-v1-gallery-code-migration', 'gallery_dir': candidate['arm_inputs']['gallery_dir'],
        'gallery_receipt_checksum': receipt['checksum'], 'logical_path': FACTORY, 'expected_sha256': F4,
        'archive': ARCHIVE[F4], 'replacement_sha256': F5,
        'scope': 'pins.code_sha() of the receipt must equal live code except this factory entry; receipt unchanged'})
    _write_once(root, GALLERY_MIGRATION, gallery_migration)
    from rshb_vine.b3_only_v1 import release as B, runtime as R
    pins_release = dict(candidate['pins_sha256'])
    pins_release.update({p: sha256(root / p) for p in (*RELEASE_CODE, *EVIDENCE, CANDIDATE, GALLERY_MIGRATION)})
    release = seal({
        'kind': B.KIND, 'policy': R.POLICY, 'name': 'b3-only-v1-release',
        'parent_profile': GEOMETRY5, 'parent_profile_sha256': sha256(root / GEOMETRY5),
        'parent_profile_checksum': geometry5['checksum'], 'target_profile_checksum': target5['checksum'],
        'arm_inputs': candidate['arm_inputs'], 'ablated_features': candidate['ablated_features'],
        'parent_ranker_checksum': candidate['parent_ranker_checksum'],
        'ablated_ranker_checksum': candidate['ablated_ranker_checksum'],
        'b3_encoder_id': 'b3c611ae6d478784ef70c6441c9bc776a4583a411aa836988f74285f33b0a6d6',
        'b0_gate_encoder_id': 'e2283412d540b566d41bccbabaf183751011164b519741cac602a196cb6590bc',
        'gallery_code_migration_checksum': gallery_migration['checksum'], 'device': candidate['device'],
        'pins_sha256': pins_release,
        'equivalent_candidate': {'path': CANDIDATE, 'sha256': sha256(root / CANDIDATE), 'checksum': candidate['checksum'],
                                 'same': 'gate, B3 4343 arm/gallery, split, single-arm dual/core, ablated ranker, '
                                         'response semantics; parent chain differs only in dispatch pins'},
        'evidence': {'acceptance': EVIDENCE[0], 'review_decision': EVIDENCE[1],
                     'http183': 'train 128/137, validation 44/46 (control 127, 42); gates 29/29'},
        'routes': candidate['routes'], 'serve': 'scripts/recognize.py serve | replay [--roi R] [--bottles all] | describe',
        'calibrated': False, 'probability': None, 'public_slug': 'never confirmed', 'weights_changed': False,
        'fit_run': False, 'independent_test': False, 'release_status': 'development_release', 'rollback': ROLLBACK})
    _write_once(root, RELEASE, release)
    entries = []
    for profile, expected in ((COHERENT, F1), (DISPATCH2, F2), (DISPATCH3, F3), (DISPATCH4, F4)):
        doc = verify(read_json(root / profile))
        if doc['pins_sha256'].get(FACTORY) != expected:
            raise ValueError('Unexpected factory pin: ' + profile)
        entries.append({'profile': profile, 'profile_sha256': sha256(root / profile), 'profile_checksum': doc['checksum'],
                        'logical_path': FACTORY, 'expected_sha256': expected, 'archive': ARCHIVE[expected],
                        'replacement_sha256': F5})
    migration = seal({'kind': 'recognition-dispatch-migration-v1', 'version': 4,
                      'dispatch_paths': [FACTORY, RECOGNIZE],
                      'scope': 'factory f16a/08b4/be2c/9d9d -> F5; recognize.py entries of v3 remain valid (6faa live)',
                      'entries': entries})
    _write_once(root, MIGRATION, migration)
    files = [*RELEASE_CODE, *EVIDENCE, CANDIDATE, COHERENT, DISPATCH2, DISPATCH3, DISPATCH4, SYSTEMIC4, TARGET4,
             GEOMETRY, DISPATCH5, SYSTEMIC5, TARGET5, GEOMETRY5, RELEASE, MIGRATION, GALLERY_MIGRATION,
             *MIGRATIONS_OLD, GEOMETRY_MANIFEST, *ARCHIVE.values(), 'rshb_vine/b3_only_v1/runtime.py',
             'rshb_vine/b3_only_v1/split.py']
    manifest = seal({
        'kind': 'recognition-release-manifest-v1', 'name': 'b3-only-v1-release',
        'files_sha256_at_stage': {p: sha256(root / p) for p in files},
        'staged_dispatch': {ARCHIVE[h]: {'logical_path': LOGICAL[h], 'sha256': h} for h in (F5, C5)},
        'profiles': {'release': {'path': RELEASE, 'checksum': release['checksum']},
                     'geometry_dispatch5': {'path': GEOMETRY5, 'checksum': geometry5['checksum']},
                     'target_dispatch5': {'path': TARGET5, 'checksum': target5['checksum']},
                     'systemic_dispatch5': {'path': SYSTEMIC5, 'checksum': systemic5['checksum']},
                     'dispatch5': {'path': DISPATCH5, 'checksum': dispatch5['checksum']},
                     'geometry_v2': {'path': GEOMETRY, 'checksum': geometry['checksum']}},
        'model_checksum': release['ablated_ranker_checksum'],
        'states': {'geometry_v2_active': [F4, R3, C4, GEOMETRY_SHA256],
                   'b3_only_active': [F5, R3, C5, sha256(root / RELEASE)]},
        'required_runtime_assets': [MIGRATION, GALLERY_MIGRATION, *MIGRATIONS_OLD, *ARCHIVE.values()],
        'code_sha_at_stage': pins.code_sha(),
        'historical_reproduction': 'dispatch4-chain profiles bind through migration v4; rollback restores exact '
                                   'f16a/6faa/7cfc bytes and pointer 226017a2, after which '
                                   'scripts/geometry_v2_release.py manages older states',
        'calibrated': False, 'weights_changed': False, 'fit_run': False, 'release_status': 'staged_pending_activation'})
    _write_once(root, MANIFEST, manifest)
    return manifest


def _manifest(root):
    manifest = verify(read_json(root / MANIFEST))
    for path, expected in manifest['files_sha256_at_stage'].items():
        if sha256(root / path) != expected:
            raise ValueError('Release file changed since stage: ' + path)
    for expected, path in ARCHIVE.items():
        if sha256(root / path) != expected:
            raise ValueError('Archive changed: ' + path)
    return manifest


def _live(root):
    return [sha256(root / p) for p in (*CODE, CURRENT)]


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
    if path == CURRENT:
        profile = next(p['path'] for p in manifest['profiles'].values() if sha256(root / p['path']) == digest)
        return (root / profile).read_bytes()
    return (root / ARCHIVE[digest]).read_bytes()


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
    _journal(root, {'event': 'switch_started', 'from': before, 'to': target, 'plan': plan})
    _apply(root, manifest, plan)
    after = state(root, manifest)
    if after != target:
        raise ValueError('Switch verification failed: ' + after + '; run recover')
    described = _describe(root)
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
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root (a temporary clone for rehearsal)')
    parser.add_argument('--confirm', help='activate: checksum of the release profile')
    parser.add_argument('--port', type=int, default=8175)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'stage':
        print(json.dumps({'manifest': stage(root)['checksum'], 'state': state(root)}))
        return
    manifest = _manifest(root)
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
        print(json.dumps(_switch(root, manifest, 'b3_only_active', args.port)))
        return
    print(json.dumps(_switch(root, manifest, 'geometry_v2_active', args.port)))


if __name__ == '__main__':
    main()
