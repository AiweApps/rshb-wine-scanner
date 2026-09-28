"""Recognition repair v2 candidate: repair-v1 composition with the composed selector stages at the live consumer.

Candidate mode (default, loopback 8189 under the live factory F6): the frozen release 51eedd7f is loaded through
its own RepairRelease (all pins verified), then its guarded OCR selector is replaced by RepairSelectionV2. Release
mode: the caller passes an already composed repair-v1 runtime (dispatch7 base) as ``repair``. Filter, crop, guard,
ranker, registry and records are inherited; public slug/probability stay None. The producer-role stage stays
declared-pending until an admitted API and source hashes are sealed into the profile.
"""
from pathlib import Path
import time

from rshb_vine.io import local_path, read_json, seal, sha256, verify

KIND = 'recognition-repair-v2-combined-profile'
PROFILE = 'runs/recognition-repair-v2/candidate/profile.json'
PARENT_RELEASE = 'config/recognition-repair-v1-release.json'
PARENT_RELEASE_CHECKSUM = '51eedd7ff128961c582194f3b52e0bd14a987299010889a9a9dffbe4595dcbe7'
REPAIR_V1_PROFILE = 'runs/recognition-repair-20260926/combined/candidate-v3/profile.json'
REPAIR_V1_CHECKSUM = '013c845afd71a5b2bee85726d5dda9d2a868aa40f1988c5077f90e3cb911d1c8'
SELECTOR_REGISTRY_CHECKSUM = '31e053df11ba7fffc38666d300cd8cfea6af7b4f58cc88abdba2580dcbd95baa'
PORT = 8189
FORBIDDEN_PORTS = (8175, 8188)
SOURCES = ('rshb_vine/recognition_repair_v2/__init__.py', 'rshb_vine/recognition_repair_v2/composition.py',
           'rshb_vine/recognition_repair_v2/geometry.py', 'rshb_vine/recognition_repair_v2/runtime.py',
           'scripts/recognition_repair_v2.py', PARENT_RELEASE, REPAIR_V1_PROFILE)
GEOMETRY_INPUTS = ('runs/catalog-training-data-v2/stage5/evaluator/actual-fit-v1/B4343/gallery/gallery.json',
                   'runs/b3-v2-evaluation-final/references.json', 'config/reference-conflicts-v1.json')


def freeze_body(root, geometry=True, producer=False):
    """``producer``: seal the producer_role_v2 pre-guard stage; only after its API/hash is admitted by root."""
    from rshb_vine.geometric_reference_probe_v1 import POLICY
    from rshb_vine.geometric_reference_probe_v1 import verifier as V
    from rshb_vine.geometric_reference_probe_v1.layout import CRITERIA
    from rshb_vine.recognition_repair_v2 import geometry as G
    root = Path(root).resolve()
    parent = verify(read_json(root / PARENT_RELEASE))
    if parent['checksum'] != PARENT_RELEASE_CHECKSUM or parent['combined_profile_checksum'] != REPAIR_V1_CHECKSUM:
        raise ValueError('Parent release is not 51eedd7f over repair-v1 013c845a')
    stages = []
    pins = {p: sha256(root / p) for p in SOURCES}
    if producer:
        from rshb_vine.producer_role_v2 import roles as R
        from rshb_vine.recognition_repair_v2 import producer as P
        P.check_pin(root)
        sources = {p: sha256(root / p) for p in (*P.SOURCES, P.PIN, 'rshb_vine/recognition_repair_v2/producer.py')}
        stages.append({'name': 'producer_role', 'stage': P.STAGE, 'position': 'pre_guard', 'rule': R.RULE['version'],
                       'flags': dict(R.FULL), 'pin_checksum': P.PIN_CHECKSUM, 'sources_sha256': sources})
        pins.update(sources)
    if geometry:
        stages.append({'name': 'layout_geometry', 'stage': G.STAGE, 'position': 'post_guard', 'trigger': V.TRIGGER,
                       'scope': G.SCOPE, 'criteria': CRITERIA, 'decision_rule': V.AMBIGUITY, 'policy': POLICY, 'sources_sha256': {p: sha256(root / p) for p in G.SOURCES}})
        pins.update({p: sha256(root / p) for p in (*G.SOURCES, *GEOMETRY_INPUTS)})
    return {'kind': KIND, 'name': 'recognition-repair-v2-candidate',
            'parent_release': {'path': PARENT_RELEASE, 'checksum': PARENT_RELEASE_CHECKSUM, 'sha256': sha256(root / PARENT_RELEASE)},
            'repair_v1_profile': {'path': REPAIR_V1_PROFILE, 'checksum': REPAIR_V1_CHECKSUM},
            'selector_registry_checksum': SELECTOR_REGISTRY_CHECKSUM, 'stages': stages,
            'producer_role': 'sealed pre_guard stage' if producer else 'pending: no admitted producer_role_v2 API/hash',
            'stage_order': 'injection -> frozen ranker -> resolver -> pre_guard -> guard v2 -> post_guard (geometry last)',
            'pins_sha256': pins, 'port': PORT, 'release_status': 'experimental_candidate_not_admitted',
            'activated': False, 'calibrated': False, 'probability': None, 'fit_run': False, 'weights_changed': False,
            'rollback': 'stop 8189; 8175, factory and config/recognition-current.json are never modified by this candidate'}


def load_profile(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if (profile.get('kind') != KIND or profile['parent_release']['checksum'] != PARENT_RELEASE_CHECKSUM
            or profile['repair_v1_profile']['checksum'] != REPAIR_V1_CHECKSUM):
        raise ValueError('Unsupported recognition-repair v2 profile')
    for path, value in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != value:
            raise ValueError('Recognition-repair v2 pinned file changed: ' + path)
    for stage in profile['stages']:
        if any(profile['pins_sha256'].get(p) != v for p, v in stage['sources_sha256'].items()):
            raise ValueError('Stage sources are not pinned by the profile: ' + stage['name'])
    return profile


def build_stages(root, profile, guarded):
    from rshb_vine.recognition_repair_v2 import geometry as G
    stages = []
    for spec in profile['stages']:
        if spec['name'] == 'layout_geometry':
            stage = G.GeometryStage(root, guarded.registry)
            if (stage.identity['sources_sha256'] != spec['sources_sha256'] or stage.identity['trigger'] != spec['trigger']['id']
                    or stage.identity['scope'] != spec['scope']['id'] or stage.identity['decision_rule'] != spec['decision_rule']['id']):
                raise ValueError('Geometry stage differs from the profile')
        elif spec['name'] == 'producer_role':
            from rshb_vine.recognition_repair_v2 import producer as P
            stage = P.ProducerRoleStage(root, guarded.inner)
            if (stage.identity['rule'] != spec['rule'] or stage.identity['flags'] != spec['flags']
                    or stage.identity['pin_checksum'] != spec['pin_checksum']
                    or any(spec['sources_sha256'][p] != v for p, v in stage.identity['sources_sha256'].items())):
                raise ValueError('Producer-role stage differs from the profile')
        else:
            raise ValueError('Unknown stage: ' + spec['name'])
        stages.append(stage)
    return stages


class RecognitionRepairV2:
    def __init__(self, root, profile_path=PROFILE, *, repair=None):
        from rshb_vine.recognition_repair_v2.composition import install
        root = Path(root).resolve()
        self.profile = load_profile(root, profile_path)
        if repair is None:
            from rshb_vine.recognition_repair_v1.release import RepairRelease
            self.outer = RepairRelease(root, PARENT_RELEASE)
            if self.outer.profile['checksum'] != PARENT_RELEASE_CHECKSUM:
                raise ValueError('Loaded parent release differs from 51eedd7f')
            repair, self.parent = self.outer.inner, {'release': PARENT_RELEASE_CHECKSUM}
        else:
            self.outer, self.parent = repair, {'repair_v1_over': repair.base['kind']}
        if repair.profile['checksum'] != REPAIR_V1_CHECKSUM:
            raise ValueError('Composed repair-v1 runtime is not 013c845a')
        if repair.selection.registry.checksum != self.profile['selector_registry_checksum']:
            raise ValueError('Selector registry differs from the profile')
        self.repair = repair
        self.selection = install(repair, build_stages(root, self.profile, repair.selection))
        self.manifest = seal({'kind': 'recognition-repair-v2-runtime', 'profile_checksum': self.profile['checksum'],
                              'runtime_descriptor_checksum': self.profile['checksum'],
                              'parent_runtime': self.outer.manifest['checksum'], 'parent': self.parent,
                              'repair_v1_profile_checksum': REPAIR_V1_CHECKSUM,
                              'stages': [s.identity for s in self.selection.stages],
                              'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        started = time.perf_counter()
        result = self.outer.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        records = (result.get('systemic_ranking_v2') or {}).get('targets', [])
        evidence = sorted(str(t['instance_id']) for t in (result.get('product_identity_evidence') or {}).get('targets', []))
        if sorted(str(r['instance_id']) for r in records) != evidence or any('recognition_repair_v2' not in r for r in records):
            raise RuntimeError('Published selector records are not the composed v2 selection of this response')
        for block in ('roskachestvo_geometry_v2', 'target_contract'):
            if result.get(block) is not None:
                result[block]['repair_v1_profile_checksum'] = result[block]['profile_checksum']
                result[block]['profile_checksum'] = self.profile['checksum']
        traces = [r['recognition_repair_v2'] for r in records]
        result['recognition_repair_v2'] = {
            'profile_checksum': self.profile['checksum'], 'runtime_checksum': self.manifest['checksum'],
            'parent': self.parent, 'repair_v1_profile_checksum': REPAIR_V1_CHECKSUM,
            'stages': [s.identity for s in self.selection.stages], 'producer_role': self.profile['producer_role'],
            'targets': traces,
            'applied': sorted(t['instance_id'] for t in traces
                              if any((v or {}).get('applied') for v in t['stages'].values())),
            'calibrated': False, 'probability': None, 'release_admitted': False,
            'seconds': time.perf_counter() - started}
        return result


def describe(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'parent_release': profile['parent_release'],
            'repair_v1_profile': profile['repair_v1_profile'], 'stages': [s['name'] for s in profile['stages']],
            'producer_role': profile['producer_role'], 'port': profile['port'],
            'public_answer': 'uncertain proposal; slug and probability stay None', 'models_loaded': False}
