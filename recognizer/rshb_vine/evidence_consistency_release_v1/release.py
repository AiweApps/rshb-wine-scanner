"""Evidence-consistency release v1 (factory F10 kind): fa317ef2 plus the root-admitted recipe over a dispatch10 chain.

Factory F10 adds this kind and the dispatch10 B3-only base; any factory change stops the F9-pinned dispatch9 chain
from loading (dispatch9 composite pin and gallery migration v5 name F9), so the release runs over byte-identical
dispatch10 copies (only the factory pin F9 -> F10 and gallery migration v5 -> v6 differ). The frozen D9 constructors
hard-code their parent paths, kind and gallery migration, so B3OnlyReleaseD10 and RepairV1D10 are explicit versioned
copies with the same identity checks. EvidenceConsistencyRelease is the TextEvidenceRelease graph composed on that
base (zero-target route, the fa317ef2 A+B+D+M selector) plus ``recipe.install`` - the same installation as the
candidate on 8199. The predecessor fa317ef2 profile and its text-evidence admission are verified statically; its
inherited fields must be identical. No sha256, pin, module function or class attribute is patched. Public slug and
probability stay None.
"""
from pathlib import Path

import numpy as np

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.b3_only_v1 import runtime as R
from rshb_vine.b3_only_v1 import split as S
from rshb_vine.evidence_consistency_release_v1 import candidate as C
from rshb_vine.evidence_consistency_release_v1 import recipe as K
from rshb_vine.recognition_repair_v1 import release as V1
from rshb_vine.recognition_repair_v1 import runtime as C1
from rshb_vine.recognition_repair_v2 import release as RR
from rshb_vine.text_evidence_repair_v2 import release as TR
from rshb_vine.text_evidence_repair_v2 import runtime as T
from rshb_vine.zero_target_release_v1 import release as ZR
from rshb_vine.zero_target_release_v1 import route as Z

KIND = 'recognition-evidence-consistency-release-v1-profile'
BASE_KIND = 'recognition-b3-only-v1-dispatch10-profile'
DECISION_KIND = 'evidence-consistency-release-v1-root-decision'
PROFILE = 'config/evidence-consistency-release-v1-profile.json'
MANIFEST = 'config/evidence-consistency-release-v1-manifest.json'
BASE_PROFILE = 'config/recognition-b3-only-v1-dispatch10.json'
PARENT_PROFILE = 'config/recognition-geometry-v2-dispatch10.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration-v6.json'
FACTORY = ZR.FACTORY
SOURCES = (f'{K.PACKAGE}/release.py', *C.SOURCES)
INHERITED = ('candidate_profile', 'candidate_profile_checksum', 'repair_v1_profile', 'repair_v1_profile_checksum', 'stages',
             'producer_pin', 'zero_target', 'text_evidence', 'ablated_ranker_checksum', 'parent_ranker_checksum', 'routes')


def sources_sha(root):
    return {p: sha256(Path(root) / p) for p in SOURCES}


def load_base_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != BASE_KIND or profile.get('policy') != R.POLICY:
        raise ValueError('Unsupported dispatch10 B3-only profile')
    if profile['parent_profile'] != PARENT_PROFILE or sha256(local_path(root, PARENT_PROFILE)) != profile['parent_profile_sha256']:
        raise ValueError('Pinned dispatch10 geometry profile bytes changed')
    V1._pinned(root, profile, 'Dispatch10 B3-only profile')
    equivalent = profile['equivalent_release']
    if (equivalent['path'] != RR.EQUIVALENT_RELEASE or equivalent['checksum'] != RR.EQUIVALENT_CHECKSUM
            or sha256(local_path(root, RR.EQUIVALENT_RELEASE)) != equivalent['sha256']):
        raise ValueError('Dispatch10 base is not bound to the frozen B3-only release 83e97cb3')
    if profile['arm_inputs'] != R.ARM_INPUTS or profile['ablated_features'] != list(R.ABLATED_FEATURES):
        raise ValueError('Dispatch10 B3-only inputs differ from the candidate protocol')
    return profile


def check_evidence_consistency(root, profile):
    """Recipe, candidate descriptor, root gate and root stage decision, all pinned by the profile and consistent."""
    ec = profile['evidence_consistency']
    recipe = K.check(ec['recipe'])
    pins = profile['pins_sha256']
    docs = {}
    for name in ('candidate_descriptor', 'gate', 'root_decision'):
        ref = ec[name]
        docs[name] = verify(read_json(local_path(root, ref['path'])))
        if pins.get(ref['path']) != ref['sha256'] or docs[name]['checksum'] != ref['checksum']:
            raise ValueError('Evidence-consistency ' + name + ' is not the pinned file')
    descriptor, gate, decision = docs['candidate_descriptor'], docs['gate'], docs['root_decision']
    if (descriptor.get('kind') != C.DESCRIPTOR_KIND or tuple(descriptor['recipe']) != recipe
            or descriptor['gate']['checksum'] != gate['checksum']):
        raise ValueError('Release recipe/gate differ from the frozen candidate descriptor')
    if (gate.get('kind') != C.GATE_KIND or gate.get('decision') != 'quality_gate_passed'
            or gate.get('parent_checksum') != C.PARENT_CHECKSUM or tuple(gate['recipe']) != recipe):
        raise ValueError('Root gate does not pass this recipe over fa317ef2')
    if (decision.get('kind') != DECISION_KIND or decision.get('decision') != 'admitted_for_stage'
            or decision.get('candidate_descriptor') != descriptor['checksum'] or decision.get('gate') != gate['checksum']
            or decision.get('recipe') != list(recipe)):
        raise ValueError('Root decision does not admit this candidate descriptor, gate and recipe')
    for ref in C.admission_refs(descriptor):
        if pins.get(ref['path']) != ref['sha256']:
            raise ValueError('Release profile does not pin admission file ' + ref['path'])
    need = {**descriptor['pins_sha256'], **descriptor['sources_sha256'], **ec['sources_sha256']}
    if any(pins.get(p) != s for p, s in need.items()):
        raise ValueError('Release profile does not pin the admitted algorithm and deploy sources')
    return recipe


def load_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile['zero_target']['rule'] != Z.RULE:
        raise ValueError('Unsupported evidence-consistency release profile')
    base = load_base_profile(root, profile['parent_profile'])
    if base['checksum'] != profile['parent_profile_checksum'] or sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
        raise ValueError('Evidence-consistency release base profile differs')
    V1._pinned(root, profile, 'Evidence-consistency release profile')
    if any(profile['pins_sha256'].get(p) != v for p, v in profile['zero_target']['sources_sha256'].items()):
        raise ValueError('Zero-target sources are not pinned by the profile')
    predecessor = profile['predecessor_release']
    if (predecessor['checksum'] != C.PARENT_CHECKSUM or predecessor['path'] != C.PARENT_PROFILE
            or profile['pins_sha256'].get(predecessor['path']) != predecessor['sha256']):
        raise ValueError('Evidence-consistency release is not over the pinned fa317ef2')
    parent = TR.load_profile(root, predecessor['path'])
    differs = [k for k in INHERITED if profile.get(k) != parent.get(k)]
    if differs:
        raise ValueError('Fields inherited from fa317ef2 differ: ' + ', '.join(differs))
    check_evidence_consistency(root, profile)
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
            'evidence_consistency': profile['evidence_consistency'], 'predecessor_release': profile['predecessor_release'],
            'public_answer': 'uncertain proposal; slug and probability stay None',
            'routes': profile['routes'], 'model_checksum': profile['ablated_ranker_checksum'], 'parent': parent,
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


def _gallery(root, profile, arm):
    """reindex.load_gallery checks with pins.code_sha() mapped through the sealed factory migration v6 only."""
    from rshb_vine.catalog_training_evaluation_v2 import pins, reindex
    migration = verify(read_json(local_path(root, GALLERY_MIGRATION)))
    if migration['checksum'] != profile['gallery_code_migration_checksum'] or migration['logical_path'] != FACTORY:
        raise ValueError('Gallery code migration differs from the dispatch10 profile')
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


class B3OnlyReleaseD10(RR.B3OnlyReleaseD7):
    """B3OnlyReleaseD9.__init__ with the dispatch10 parent, kind and gallery migration v6; recognize() inherited."""

    def __init__(self, root, profile_path):
        from rshb_vine.catalog_training_evaluation_v2 import adapter, reindex
        from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
        from rshb_vine.visual_core import LabelFirstIndex
        root = Path(root).resolve()
        profile = load_base_profile(root, profile_path)
        geometry = GeometryReleaseRecognition(root, PARENT_PROFILE)
        if geometry.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded geometry release differs from the pinned dispatch10 parent')
        target = geometry.inner
        if target.profile['checksum'] != profile['target_profile_checksum']:
            raise ValueError('Geometry parent is not the pinned dispatch10 target-contract profile')
        runtime = adapter.recognition_runtime(target)
        a = {k: str(root / v) for k, v in R.ARM_INPUTS.items()}
        arm, model_path = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                              a['fit_admission'])
        if arm['arm'] != 'B4343' or arm['encoder_id'] != profile['b3_encoder_id']:
            raise ValueError('Dispatch10 B3-only base is frozen to stage5 checkpoint 4343')
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
            raise ValueError('Gate/core encoder identities differ from the dispatch10 profile')
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
        self.manifest = seal({'kind': 'b3-only-v1-dispatch10-runtime', 'policy': R.POLICY,
                              'profile_checksum': profile['checksum'], 'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': geometry.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'equivalent_release': profile['equivalent_release'], 'ownership': self.ownership,
                              'arm': arm, 'activated': True, 'calibration_status': 'unknown'})


def check_base(root, base):
    """ZR.check_base for the dispatch10 kind: identical encoders, arm, ranker and ablation to 83e97cb3."""
    original = verify(read_json(local_path(root, C1.BASE_PROFILE)))
    profile = base.profile
    equivalent = profile.get('equivalent_release') or {}
    if (type(base) is not B3OnlyReleaseD10 or profile.get('kind') != BASE_KIND or equivalent.get('path') != C1.BASE_PROFILE
            or equivalent.get('checksum') != C1.BASE_CHECKSUM
            or equivalent.get('sha256') != sha256(local_path(root, C1.BASE_PROFILE)) or original['checksum'] != C1.BASE_CHECKSUM):
        raise ValueError('Supplied base is not the dispatch10 equivalent of the frozen B3-only release 83e97cb3')
    differs = [k for k in C1.BASE_IDENTITY if profile.get(k) != original.get(k)]
    if differs:
        raise ValueError('Supplied base differs from 83e97cb3 in: ' + ', '.join(differs))
    own = base.ownership
    if (own['core_B3_arm']['encoder_id'] != original['b3_encoder_id'] or own['object_gate']['encoder_id'] != original['b0_gate_encoder_id']
            or own['ranker']['ablated'] != original['ablated_ranker_checksum'] or own['selection_raw_visual_arms'] != ['B3']):
        raise ValueError('Loaded base ownership differs from 83e97cb3')
    return {'kind': profile['kind'], 'checksum': profile['checksum'], 'equivalent_release': C1.BASE_CHECKSUM}


class RepairV1D10(RR.RepairV1D7):
    """RepairV1D9.__init__ over an explicit dispatch10 base; recognize() inherited."""

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


class EvidenceConsistencyRelease:
    """The fa317ef2 graph over the dispatch10 base with the admitted I/G/O/R recipe installed once."""

    def __init__(self, root, profile_path=PROFILE):
        from rshb_vine.recognition_repair_v2.runtime import RecognitionRepairV2
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        base = B3OnlyReleaseD10(root, profile['parent_profile'])
        if base.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded dispatch10 base differs from the release profile')
        repair = RepairV1D10(root, profile['repair_v1_profile'], base)
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
        self.text_recipe = tuple(profile['text_evidence']['recipe'])
        self.selection, self.text_installation = T.install_recipe(self, self.text_recipe, profile['text_evidence']['tables'])
        self.recipe = tuple(profile['evidence_consistency']['recipe'])
        self.installation = K.install(self, self.recipe, root)
        self.predecessor = profile['predecessor_release']['checksum']
        self.lineage = [self.predecessor, T.PARENT_CHECKSUM]
        self.manifest = seal({'kind': 'evidence-consistency-release-v1-runtime', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': inner.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'candidate_profile_checksum': profile['candidate_profile_checksum'],
                              'repair_v1_profile_checksum': RR.REPAIR_V1_CHECKSUM,
                              'predecessor_release': self.predecessor, 'recipe': list(self.recipe),
                              'route_installation': self.route_installation, 'text_installation': self.text_installation,
                              'installation': self.installation, 'activated': True, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        with K.scope(self.recipe, roi, bottles) as route_scope:
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
        own, runtime, predecessor = self.profile['checksum'], self.manifest['checksum'], self.predecessor
        for block in (v1, v2):
            block.update(release_profile_checksum=own, release_admitted=True, parent_release_profile_checksum=predecessor)
        for name in ('roskachestvo_geometry_v2', 'target_contract'):
            if result.get(name) is not None:
                result[name]['candidate_profile_checksum'] = result[name]['profile_checksum']
                result[name].update(profile_checksum=own, parent_release_profile_checksum=predecessor)
        lineage = dict(base_profile_checksum=self.profile['parent_profile_checksum'],
                       candidate_profile_checksum=self.profile['candidate_profile_checksum'],
                       repair_v1_profile_checksum=RR.REPAIR_V1_CHECKSUM, parent_release_profile_checksum=predecessor,
                       parent_lineage=list(self.lineage), equivalent_base_release=RR.EQUIVALENT_CHECKSUM)
        result['recognition_repair_release_v2'] = {'profile_checksum': own, 'runtime_checksum': runtime,
                                                   'predecessor_release': predecessor, **lineage,
                                                   'calibrated': False, 'probability': None}
        result[ZR.BLOCK] = ZR.block(scope, own, runtime, **lineage, release_admitted=True)
        result[TR.BLOCK] = dict(T.block(result, self.selection, own, runtime, predecessor, self.text_recipe, admitted=True),
                                parent_lineage=list(self.lineage))
        entries = K.label_view_contract(result, self.recipe)
        result[K.BLOCK] = K.block(result, self.recipe, route_scope, entries, stage='release', profile_checksum=own,
                                  runtime_checksum=runtime, parent_release_profile_checksum=predecessor,
                                  parent_lineage=list(self.lineage), release_admitted=True)
        return result
