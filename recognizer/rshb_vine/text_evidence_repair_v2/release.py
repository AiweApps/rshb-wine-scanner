"""Text-evidence release v2 (factory F9 kind): 24b67962 over dispatch9 copies of its chain with the admitted D recipe.

Factory F9 adds this kind and the dispatch9 B3-only base; any factory change stops the F8-pinned dispatch8 chain from
loading, so the release runs over byte-identical dispatch9 copies (only the factory pin F8 -> F9 and gallery migration
v4 -> v5 differ). The frozen D8 constructors hard-code their parent paths, kind and gallery migration, so
B3OnlyReleaseD9 and RepairV1D9 are explicit versioned copies with the same identity checks. TextEvidenceRelease is
ZeroTargetRelease composed on that base plus ``runtime.install_recipe``: the same selector installation, recipe and
tables as the candidate on 8197. No sha256, pin, module function or class attribute is patched. Public slug and
probability stay None.
"""
from pathlib import Path

import numpy as np

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.b3_only_v1 import runtime as R
from rshb_vine.b3_only_v1 import split as S
from rshb_vine.recognition_repair_v1 import release as V1
from rshb_vine.recognition_repair_v1 import runtime as C1
from rshb_vine.recognition_repair_v2 import release as RR
from rshb_vine.text_evidence_repair_v2 import runtime as T
from rshb_vine.zero_target_release_v1 import release as ZR
from rshb_vine.zero_target_release_v1 import route as Z

KIND = 'recognition-text-evidence-release-v2-profile'
BASE_KIND = 'recognition-b3-only-v1-dispatch9-profile'
PROFILE = 'config/text-evidence-release-v2-profile.json'
MANIFEST = 'config/text-evidence-release-v2-manifest.json'
BASE_PROFILE = 'config/recognition-b3-only-v1-dispatch9.json'
PARENT_PROFILE = 'config/recognition-geometry-v2-dispatch9.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration-v5.json'
FACTORY = ZR.FACTORY
SOURCES = (f'{T.PACKAGE}/release.py', *T.SOURCES, T.PACKAGE_INIT)
BLOCK = T.BLOCK


def sources_sha(root):
    return {p: sha256(Path(root) / p) for p in SOURCES}


def load_base_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != BASE_KIND or profile.get('policy') != R.POLICY:
        raise ValueError('Unsupported dispatch9 B3-only profile')
    if profile['parent_profile'] != PARENT_PROFILE or sha256(local_path(root, PARENT_PROFILE)) != profile['parent_profile_sha256']:
        raise ValueError('Pinned dispatch9 geometry profile bytes changed')
    V1._pinned(root, profile, 'Dispatch9 B3-only profile')
    equivalent = profile['equivalent_release']
    if (equivalent['path'] != RR.EQUIVALENT_RELEASE or equivalent['checksum'] != RR.EQUIVALENT_CHECKSUM
            or sha256(local_path(root, RR.EQUIVALENT_RELEASE)) != equivalent['sha256']):
        raise ValueError('Dispatch9 base is not bound to the frozen B3-only release 83e97cb3')
    if profile['arm_inputs'] != R.ARM_INPUTS or profile['ablated_features'] != list(R.ABLATED_FEATURES):
        raise ValueError('Dispatch9 B3-only inputs differ from the candidate protocol')
    return profile


def check_text_evidence(root, profile):
    """Recipe, tables, descriptor, protocol, gate and root decision of the release, all pinned by the profile."""
    te = profile['text_evidence']
    recipe = T._recipe(te['recipe'])
    if te['tables'] != {'B': T.SHORT_FORM_TABLE, 'D': T.SHORT_FORM_TABLE}:
        raise ValueError('B and D must read the one admitted short-form table')
    pins = profile['pins_sha256']
    for name in ('candidate_descriptor', 'protocol', 'gate', 'root_decision'):
        ref = te[name]
        if pins.get(ref['path']) != ref['sha256'] or verify(read_json(local_path(root, ref['path'])))['checksum'] != ref['checksum']:
            raise ValueError('Text-evidence ' + name + ' is not the pinned file')
    descriptor = verify(read_json(local_path(root, te['candidate_descriptor']['path'])))
    protocol = verify(read_json(local_path(root, te['protocol']['path'])))
    gate = verify(read_json(local_path(root, te['gate']['path'])))
    if (descriptor.get('kind') != T.DESCRIPTOR_KIND or tuple(descriptor['recipe']) != recipe or descriptor['tables'] != te['tables']
            or descriptor['protocol']['checksum'] != protocol['checksum'] or descriptor['gate']['checksum'] != gate['checksum']):
        raise ValueError('Release recipe/tables/admission differ from the frozen candidate descriptor')
    if (gate.get('kind') != T.GATE_KIND or gate.get('decision') != 'quality_gate_passed'
            or gate.get('protocol_checksum') != protocol['checksum'] or tuple(gate['recipe']) != recipe):
        raise ValueError('Root gate does not pass this recipe under this protocol')
    moved = {p: s for p, s in protocol['pins_sha256'].items() if p in T.LIVE_CODE}
    kept = {p: s for p, s in protocol['pins_sha256'].items() if p not in T.LIVE_CODE}
    if any(pins.get(p) != s for p, s in {**kept, **descriptor['sources_sha256'], **te['sources_sha256']}.items()):
        raise ValueError('Release profile does not pin the protocol algorithm and deploy sources')
    predecessor = profile['predecessor_release']
    if moved != {T.LIVE_CODE[-1]: predecessor['sha256']} or te['protocol_pointer_pin'] != {'path': T.LIVE_CODE[-1], 'sha256': predecessor['sha256']}:
        raise ValueError('The protocol pointer pin must be the predecessor 24b67962 bytes only')
    return recipe


def load_profile(root, profile_path):
    from rshb_vine.recognition_repair_v2 import runtime as C2
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile['zero_target']['rule'] != Z.RULE:
        raise ValueError('Unsupported text-evidence release profile')
    base = load_base_profile(root, profile['parent_profile'])
    if base['checksum'] != profile['parent_profile_checksum'] or sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
        raise ValueError('Text-evidence release base profile differs')
    V1._pinned(root, profile, 'Text-evidence release profile')
    if any(profile['pins_sha256'].get(p) != v for p, v in profile['zero_target']['sources_sha256'].items()):
        raise ValueError('Zero-target sources are not pinned by the profile')
    predecessor = profile['predecessor_release']
    if (predecessor['checksum'] != T.PARENT_CHECKSUM or predecessor['path'] != T.PARENT_PROFILE
            or profile['pins_sha256'].get(predecessor['path']) != predecessor['sha256']):
        raise ValueError('Text-evidence release is not over the pinned 24b67962')
    candidate = C2.load_profile(root, profile['candidate_profile'])
    if candidate['checksum'] != profile['candidate_profile_checksum'] or profile['candidate_profile'] not in profile['pins_sha256']:
        raise ValueError('Repair v2 candidate profile differs from the release profile')
    repair = C1.load_profile(root, profile['repair_v1_profile'])
    if repair['checksum'] != RR.REPAIR_V1_CHECKSUM or profile['repair_v1_profile_checksum'] != RR.REPAIR_V1_CHECKSUM:
        raise ValueError('Text-evidence release is not over repair-v1 013c845a')
    if [s['name'] for s in candidate['stages']] != profile['stages'] or RR._producer_pin(root, candidate) != profile['producer_pin']:
        raise ValueError('Candidate stages or producer pin differ from the release profile')
    check_text_evidence(root, profile)
    return profile


def describe_base(root, profile_path):
    root = Path(root).resolve()
    profile = load_base_profile(root, profile_path)
    parent = V1._describe_parent(root, profile)
    return {'kind': BASE_KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'], 'policy': R.POLICY,
            'composition': parent['composition'] + ' -> B3-only SKU retrieval (stage5 4343) + B0 context-only object gate',
            'public_answer': 'uncertain proposal; slug never confirmed', 'routes': profile['routes'],
            'model_checksum': profile['ablated_ranker_checksum'], 'parent_ranker_checksum': profile['parent_ranker_checksum'],
            'equivalent_release': profile['equivalent_release'], 'parent': parent,
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


def describe_release(root, profile_path):
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    parent = V1._describe_parent(root, profile)
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': parent['composition'] + ' -> ' + profile['composition'],
            'candidate_profile': profile['candidate_profile'], 'candidate_profile_checksum': profile['candidate_profile_checksum'],
            'repair_v1_profile_checksum': profile['repair_v1_profile_checksum'], 'stages': profile['stages'],
            'producer_pin': profile['producer_pin'], 'components': profile['components'],
            'zero_target': profile['zero_target'], 'text_evidence': profile['text_evidence'],
            'predecessor_release': profile['predecessor_release'],
            'public_answer': 'uncertain proposal; slug and probability stay None',
            'routes': profile['routes'], 'model_checksum': profile['ablated_ranker_checksum'], 'parent': parent,
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


def _gallery(root, profile, arm):
    """reindex.load_gallery checks with pins.code_sha() mapped through the sealed factory migration v5 only."""
    from rshb_vine.catalog_training_evaluation_v2 import pins, reindex
    migration = verify(read_json(local_path(root, GALLERY_MIGRATION)))
    if migration['checksum'] != profile['gallery_code_migration_checksum'] or migration['logical_path'] != FACTORY:
        raise ValueError('Gallery code migration differs from the dispatch9 profile')
    if sha256(local_path(root, migration['archive'])) != migration['expected_sha256']:
        raise ValueError('Archived factory bytes differ from the gallery code migration')
    folder = local_path(root, R.ARM_INPUTS['gallery_dir'])
    receipt = verify(read_json(folder / 'receipt.json'))
    gallery = verify(read_json(folder / 'gallery.json'))
    live = pins.code_sha()
    if live.get(FACTORY) != migration['replacement_sha256'] or receipt['code_sha256'].get(FACTORY) != migration['expected_sha256']:
        raise ValueError('Gallery receipt factory hash is not covered by the migration')
    if receipt['checksum'] != migration['gallery_receipt_checksum'] or dict(live, **{FACTORY: migration['expected_sha256']}) != receipt['code_sha256']:
        raise ValueError('Gallery receipt code provenance differs beyond the migrated factory')
    if (receipt.get('kind') != 'catalog-training-v2-b3-gallery-reindex-v1' or receipt['arm'] != arm
            or receipt['gallery_checksum'] != gallery['checksum'] or receipt['gallery_sha256'] != sha256(folder / 'gallery.json')
            or gallery['encoder_id'] != arm['encoder_id'] or receipt['references'] != reindex.PREFIX + reindex.ADDED
            or sha256(folder / 'vectors.npy') != gallery['vectors_sha'] or receipt['vectors_sha256'] != gallery['vectors_sha']):
        raise ValueError('Reindexed gallery provenance does not bind this arm')
    refs, _ = reindex.references()
    if gallery['references'] != refs:
        raise ValueError('Gallery reference order differs from serving B3 order')
    return gallery, np.load(folder / 'vectors.npy', allow_pickle=False)


class B3OnlyReleaseD9(RR.B3OnlyReleaseD7):
    """B3OnlyReleaseD8.__init__ with the dispatch9 parent, kind and gallery migration v5; recognize() inherited."""

    def __init__(self, root, profile_path):
        from rshb_vine.catalog_training_evaluation_v2 import adapter, reindex
        from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
        from rshb_vine.visual_core import LabelFirstIndex
        root = Path(root).resolve()
        profile = load_base_profile(root, profile_path)
        geometry = GeometryReleaseRecognition(root, PARENT_PROFILE)
        if geometry.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded geometry release differs from the pinned dispatch9 parent')
        target = geometry.inner
        if target.profile['checksum'] != profile['target_profile_checksum']:
            raise ValueError('Geometry parent is not the pinned dispatch9 target-contract profile')
        runtime = adapter.recognition_runtime(target)
        a = {k: str(root / v) for k, v in R.ARM_INPUTS.items()}
        arm, model_path = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                              a['fit_admission'])
        if arm['arm'] != 'B4343' or arm['encoder_id'] != profile['b3_encoder_id']:
            raise ValueError('Dispatch9 B3-only base is frozen to stage5 checkpoint 4343')
        gallery, vectors = _gallery(root, profile, arm)
        encoder = reindex.load_encoder_model(model_path, profile['device'])
        index = LabelFirstIndex(vectors, gallery['references'], arm['encoder_id'])
        substitution = adapter.substitute(target, arm, encoder, index, gallery)

        base, core = runtime.holders['B0'], runtime.holders['B3']
        instance = runtime.control.expanded.core.parent.parent.parent
        b0_eid = runtime.arms['B0'][0]
        if instance.base is not base or instance.spec['encoder_id'] != b0_eid or base.encoder_id != b0_eid:
            raise ValueError('Object gate is not bound to the original B0 holder')
        if b0_eid != profile['b0_gate_encoder_id'] or core.eid != arm['encoder_id'] or core.eid == b0_eid:
            raise ValueError('Gate/core encoder identities differ from the dispatch9 profile')
        b0_encoder_raw = getattr(base.encoder, '_encoder', base.encoder)
        b0_index_raw = getattr(base.index, '_index', base.index)
        b0_encoder_wrapped, b0_index_wrapped = base.encoder, base.index
        self.counters = S.RoleCounters()
        mro = S.install_split(instance, b0_encoder_wrapped, b0_eid, self.counters)
        base.encoder, base.encoder_id = core.encoder, core.eid
        base.index = S.CountingIndex(core.index, self.counters, 'b3_base_route_search_calls')
        S.guard_b0_encoder(b0_encoder_raw)
        S.install_single_arm_core(core, self.counters)
        S.tripwire(b0_index_raw, ('search', 'reuse'))
        if b0_index_wrapped is not b0_index_raw:
            S.tripwire(b0_index_wrapped, ('search', 'reuse'))
        control = runtime.control
        if (type(control.dual).__name__ != 'ReferencePhrasePreservation'
                or type(control.dual.parent).__name__ != 'RankFusedB3Consensus'):
            raise ValueError('Control dual-fusion ownership changed')
        control.dual.parent = S.SingleArmDual(self.counters)
        runtime.arms = {'B3': (core.eid, core.encoder, S.CountingIndex(core.index, self.counters, 'b3_selection_arm_searches', 'b3_selection_arm_reuses'))}

        selection = target.inner.selection
        if runtime.selection is not selection or type(selection).__name__ != 'SystemicSelection':
            raise ValueError('Systemic selection ownership changed')
        ranker = R.ablate(selection.ranker)
        if (selection.ranker['checksum'] != profile['parent_ranker_checksum']
                or ranker['checksum'] != profile['ablated_ranker_checksum']):
            raise ValueError('Systemic ranker or its fixed ablation differs from the profile')
        selection.use_ranker(ranker)

        refs, visited = R._graph_refs(geometry, {'b0_index_raw': b0_index_raw, 'b0_encoder_raw': b0_encoder_raw,
                                                 'b0_encoder_wrapped': b0_encoder_wrapped})
        stray = [r for r in refs if r['object'] == 'b0_index_raw' or not (
            r['path'].endswith('.gate_encoder') or '._owners[' in r['path']
            or r['object'] == 'b0_encoder_raw' and r['path'].endswith('.gate_encoder._encoder'))]
        if not any(r['path'].endswith('.gate_encoder') for r in refs):
            raise ValueError('Graph scan did not reach the object-gate encoder; proof incomplete')
        if stray:
            raise ValueError('Hidden B0 consumer remains in the loaded graph: ' + repr(stray[:3]))
        self.ownership = {
            'object_gate': {'encoder_id': b0_eid, 'spec_checksum': instance.spec['checksum'],
                            'threshold': instance.spec['threshold'], 'views': 'context only, original B0 encoder'},
            'instance_mro': mro,
            'base_holder': {'encoder_id': core.eid, 'shared_with_core_arm': base.encoder is core.encoder,
                            'gallery_checksum': gallery['checksum'], 'vectors_sha256': gallery['vectors_sha']},
            'core_B3_arm': {'encoder_id': core.eid, 'substitution_checksum': substitution['checksum']},
            'selection_raw_visual_arms': sorted(runtime.arms),
            'ranker': {'parent': profile['parent_ranker_checksum'], 'ablated': ranker['checksum'],
                       'zeroed': list(R.ABLATED_FEATURES)},
            'graph_scan': {'objects_visited': visited, 'b0_references': refs, 'stray': stray}}
        self.geometry, self.target, self.runtime, self.profile = geometry, target, runtime, profile
        self.manifest = seal({'kind': 'b3-only-v1-dispatch9-runtime', 'policy': R.POLICY,
                              'profile_checksum': profile['checksum'], 'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': geometry.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'equivalent_release': profile['equivalent_release']['checksum'],
                              'ownership': self.ownership, 'arm': arm, 'activated': True,
                              'calibration_status': 'unknown'})


def check_base(root, base):
    """ZR.check_base for the dispatch9 kind: identical encoders, arm, ranker and ablation to 83e97cb3."""
    original = verify(read_json(local_path(root, C1.BASE_PROFILE)))
    profile = base.profile
    equivalent = profile.get('equivalent_release') or {}
    if (type(base) is not B3OnlyReleaseD9 or profile.get('kind') != BASE_KIND or equivalent.get('path') != C1.BASE_PROFILE
            or equivalent.get('checksum') != C1.BASE_CHECKSUM
            or equivalent.get('sha256') != sha256(local_path(root, C1.BASE_PROFILE)) or original['checksum'] != C1.BASE_CHECKSUM):
        raise ValueError('Supplied base is not the dispatch9 equivalent of the frozen B3-only release 83e97cb3')
    differs = [k for k in C1.BASE_IDENTITY if profile.get(k) != original.get(k)]
    if differs:
        raise ValueError('Supplied base differs from 83e97cb3 in: ' + ', '.join(differs))
    own = base.ownership
    if (own['core_B3_arm']['encoder_id'] != original['b3_encoder_id'] or own['object_gate']['encoder_id'] != original['b0_gate_encoder_id']
            or own['ranker']['ablated'] != original['ablated_ranker_checksum'] or own['selection_raw_visual_arms'] != ['B3']):
        raise ValueError('Loaded base ownership differs from 83e97cb3')
    return {'kind': profile['kind'], 'checksum': profile['checksum'], 'equivalent_release': C1.BASE_CHECKSUM}


class RepairV1D9(RR.RepairV1D7):
    """RepairV1D8.__init__ over an explicit dispatch9 base; recognize() inherited."""

    def __init__(self, root, profile_path, base_release):
        from rshb_vine.ocr_candidate_repair_v1 import guarded as ocr_selection
        from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
        root = Path(root).resolve()
        self.profile = C1.load_profile(root, profile_path)
        self.guard = ReferenceConflicts(root)
        if self.guard.checksum != self.profile['reference_conflicts_checksum']:
            raise ValueError('Reference-conflict config differs from the profile')
        self.base = check_base(root, base_release)
        self.release = base_release
        self.filtered, filter_install = C1.install_reference_filter(self.release, self.guard)
        self.selection = ocr_selection.install(self.release, self.guard.injection_conflict)
        if type(self.selection) is not ocr_selection.GuardedOcrCandidateSelection:
            raise ValueError('OCR selection must be the guarded v2 adapter')
        self.front_label = None
        if self.profile['components']['front_label_repair']:
            from rshb_vine.front_label_repair_v1 import runtime as front_label
            self.front_label = front_label
            if front_label.load_profile(root, C1.FRONT_LABEL_PROFILE)['checksum'] != C1.FRONT_LABEL_CHECKSUM:
                raise ValueError('Front-label profile differs from the admitted 577a575e')
            front_install = front_label.install(self.release)
        else:
            front_install = None
        self.installation = {'reference_filter': filter_install, 'ocr_selection': type(self.selection).__module__,
                             'front_label_repair': front_install}
        self.manifest = seal({'kind': 'recognition-repair-v1-runtime', 'profile_checksum': self.profile['checksum'],
                              'runtime_descriptor_checksum': self.profile['checksum'],
                              'parent_runtime': self.release.manifest['checksum'], 'base_profile_checksum': self.base['checksum'],
                              'equivalent_base_release': C1.BASE_CHECKSUM,
                              'reference_conflicts_checksum': self.guard.checksum,
                              'installation': self.installation, 'activated': False, 'calibration_status': 'unknown'})


class TextEvidenceRelease:
    """ZeroTargetRelease over the dispatch9 base with the admitted D recipe installed once on its selector."""

    def __init__(self, root, profile_path=PROFILE):
        from rshb_vine.recognition_repair_v2.runtime import RecognitionRepairV2
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        base = B3OnlyReleaseD9(root, profile['parent_profile'])
        if base.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded dispatch9 base differs from the release profile')
        repair = RepairV1D9(root, profile['repair_v1_profile'], base)
        if repair.profile['checksum'] != RR.REPAIR_V1_CHECKSUM or repair.release is not base:
            raise ValueError('Repair-v1 runtime is not 013c845a over this base')
        inner = RecognitionRepairV2(root, profile['candidate_profile'], repair=repair)
        if (inner.profile['checksum'] != profile['candidate_profile_checksum'] or inner.repair is not repair
                or inner.outer is not repair or repair.selection is not inner.selection):
            raise ValueError('Repair v2 runtime is not the pinned composition over this repair-v1 runtime')
        if [s['name'] for s in inner.profile['stages']] != profile['stages']:
            raise ValueError('Repair v2 stages differ from the release profile')
        if base.runtime.selection is not inner.selection or base.target.inner.selection is not inner.selection:
            raise ValueError('Composed v2 selector is not the live consumer of this base')
        self.root, self.profile, self.base, self.repair, self.inner = root, profile, base, repair, inner
        self.route_installation = Z.install(self)
        self.recipe = tuple(profile['text_evidence']['recipe'])
        self.selection, self.installation = T.install_recipe(self, self.recipe, profile['text_evidence']['tables'])
        self.manifest = seal({'kind': 'text-evidence-release-v2-runtime', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': inner.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'candidate_profile_checksum': profile['candidate_profile_checksum'],
                              'repair_v1_profile_checksum': RR.REPAIR_V1_CHECKSUM,
                              'predecessor_release': profile['predecessor_release']['checksum'],
                              'route_installation': self.route_installation, 'installation': self.installation,
                              'activated': True, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        with Z.request_scope(roi, bottles) as scope:
            result = self.inner.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        v1, v2 = result['recognition_repair_v1'], result['recognition_repair_v2']
        if v1['base_profile_checksum'] != self.profile['parent_profile_checksum'] or v1['base_kind'] != BASE_KIND:
            raise RuntimeError('Repair-v1 runtime reports another base than the release profile')
        if v2['profile_checksum'] != self.profile['candidate_profile_checksum']:
            raise RuntimeError('Repair v2 runtime reports another candidate profile than the release profile')
        own, runtime = self.profile['checksum'], self.manifest['checksum']
        v1.update(release_profile_checksum=own, release_admitted=True)
        v2.update(release_profile_checksum=own, release_admitted=True)
        for name in ('roskachestvo_geometry_v2', 'target_contract'):
            if result.get(name) is not None:
                result[name]['candidate_profile_checksum'] = result[name]['profile_checksum']
                result[name]['profile_checksum'] = own
        predecessor = self.profile['predecessor_release']['checksum']
        result['recognition_repair_release_v2'] = {
            'profile_checksum': own, 'runtime_checksum': runtime,
            'base_profile_checksum': self.profile['parent_profile_checksum'],
            'candidate_profile_checksum': self.profile['candidate_profile_checksum'],
            'repair_v1_profile_checksum': RR.REPAIR_V1_CHECKSUM, 'predecessor_release': predecessor,
            'parent_release_profile_checksum': predecessor, 'equivalent_base_release': RR.EQUIVALENT_CHECKSUM,
            'calibrated': False, 'probability': None}
        result[ZR.BLOCK] = ZR.block(
            scope, own, runtime, base_profile_checksum=self.profile['parent_profile_checksum'],
            candidate_profile_checksum=self.profile['candidate_profile_checksum'],
            repair_v1_profile_checksum=RR.REPAIR_V1_CHECKSUM, parent_release_profile_checksum=predecessor,
            equivalent_base_release=RR.EQUIVALENT_CHECKSUM, release_admitted=True)
        result[BLOCK] = T.block(result, self.selection, own, runtime, predecessor, self.recipe, admitted=True)
        return result
