"""Staged release of the combined recognition repair behind a versioned factory (F6), never the live chain.

Factory F6 adds two kinds, so the frozen dispatch5 chain and the B3-only release 83e97cb3 no longer load under it
(both pin or check factory F5). Base: the same B3-only composition, rebuilt over byte-identical dispatch6 copies
of the geometry-v2 chain. The B4343 gallery receipt binds pins.code_sha(); the only accepted difference is the
factory through gallery code migration v2 (archived F4 -> F6). Release: RecognitionRepairV1 (the reviewed
combined candidate) over that base, passed in as ``base_release``. Checksums, pins and ownership are checked here;
no sha256 or pin is patched. Public slug and probability stay None.
"""
from pathlib import Path
import time

import numpy as np

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.b3_only_v1 import runtime as R
from rshb_vine.b3_only_v1 import split as S

BASE_KIND = 'recognition-b3-only-v1-dispatch6-profile'
KIND = 'recognition-repair-v1-release-profile'
CURRENT_POINTER = 'config/recognition-current.json'
PARENT_PROFILE = 'config/recognition-geometry-v2-dispatch6.json'
TARGET_PROFILE = 'config/recognition-target-contract-v2-dispatch6.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration-v2.json'
EQUIVALENT_RELEASE = 'config/recognition-b3-only-v1-release.json'
EQUIVALENT_CHECKSUM = '83e97cb3f6e8d1a89451b1ffcdd8de4e20f6417706b488f1e4d426c538cafbbb'
FACTORY = 'rshb_vine/recognition_factory.py'


def _pinned(root, profile, what):
    if profile['parent_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError(what + ' must not depend on the mutable current pointer')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError(what + ' pinned file changed: ' + path)


def load_base_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != BASE_KIND or profile.get('policy') != R.POLICY:
        raise ValueError('Unsupported dispatch6 B3-only profile')
    if profile['parent_profile'] != PARENT_PROFILE or sha256(local_path(root, PARENT_PROFILE)) != profile['parent_profile_sha256']:
        raise ValueError('Pinned dispatch6 geometry profile bytes changed')
    _pinned(root, profile, 'Dispatch6 B3-only profile')
    equivalent = profile['equivalent_release']
    if (equivalent['path'] != EQUIVALENT_RELEASE or equivalent['checksum'] != EQUIVALENT_CHECKSUM
            or sha256(local_path(root, EQUIVALENT_RELEASE)) != equivalent['sha256']):
        raise ValueError('Dispatch6 base is not bound to the frozen B3-only release 83e97cb3')
    if profile['arm_inputs'] != R.ARM_INPUTS or profile['ablated_features'] != list(R.ABLATED_FEATURES):
        raise ValueError('Dispatch6 B3-only inputs differ from the candidate protocol')
    return profile


def load_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND:
        raise ValueError('Unsupported recognition-repair release profile')
    base = load_base_profile(root, profile['parent_profile'])
    if base['checksum'] != profile['parent_profile_checksum'] or sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
        raise ValueError('Repair release base profile differs')
    _pinned(root, profile, 'Repair release profile')
    combined = verify(read_json(local_path(root, profile['combined_profile'])))
    if combined['checksum'] != profile['combined_profile_checksum'] or profile['combined_profile'] not in profile['pins_sha256']:
        raise ValueError('Combined repair profile differs from the release profile')
    return profile


def _describe_parent(root, profile):
    from rshb_vine.recognition_factory import describe
    parent = describe(root, profile['parent_profile'])
    if parent['profile_checksum'] != profile['parent_profile_checksum']:
        raise ValueError('Parent profile differs from the pinned checksum')
    return parent


def describe_base(root, profile_path):
    root = Path(root).resolve()
    profile = load_base_profile(root, profile_path)
    parent = _describe_parent(root, profile)
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
    parent = _describe_parent(root, profile)
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': parent['composition'] + ' -> ' + profile['composition'],
            'combined_profile': profile['combined_profile'], 'combined_profile_checksum': profile['combined_profile_checksum'],
            'components': profile['components'], 'public_answer': 'uncertain proposal; slug and probability stay None',
            'routes': profile['routes'], 'model_checksum': profile['ablated_ranker_checksum'], 'parent': parent,
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


def _gallery(root, profile, arm):
    """reindex.load_gallery checks with pins.code_sha() mapped through the sealed factory migration v2 only."""
    from rshb_vine.catalog_training_evaluation_v2 import pins, reindex
    migration = verify(read_json(local_path(root, GALLERY_MIGRATION)))
    if migration['checksum'] != profile['gallery_code_migration_checksum'] or migration['logical_path'] != FACTORY:
        raise ValueError('Gallery code migration differs from the dispatch6 profile')
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


class B3OnlyReleaseD6:
    """The B3-only release composition of b3_only_v1.release.B3OnlyRelease over the dispatch6 chain."""

    def __init__(self, root, profile_path):
        from rshb_vine.catalog_training_evaluation_v2 import adapter, reindex
        from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
        from rshb_vine.visual_core import LabelFirstIndex
        root = Path(root).resolve()
        profile = load_base_profile(root, profile_path)
        geometry = GeometryReleaseRecognition(root, PARENT_PROFILE)
        if geometry.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded geometry release differs from the pinned dispatch6 parent')
        target = geometry.inner
        if target.profile['checksum'] != profile['target_profile_checksum']:
            raise ValueError('Geometry parent is not the pinned dispatch6 target-contract profile')
        runtime = adapter.recognition_runtime(target)
        a = {k: str(root / v) for k, v in R.ARM_INPUTS.items()}
        arm, model_path = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                              a['fit_admission'])
        if arm['arm'] != 'B4343' or arm['encoder_id'] != profile['b3_encoder_id']:
            raise ValueError('Dispatch6 B3-only base is frozen to stage5 checkpoint 4343')
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
            raise ValueError('Gate/core encoder identities differ from the dispatch6 profile')
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
        self.manifest = seal({'kind': 'b3-only-v1-dispatch6-runtime', 'policy': R.POLICY,
                              'profile_checksum': profile['checksum'], 'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': geometry.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'equivalent_release': profile['equivalent_release']['checksum'],
                              'ownership': self.ownership, 'arm': arm, 'activated': True,
                              'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        started = time.perf_counter()
        self.counters.reset()
        result = self.geometry.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        arms = [sorted(t.get('raw_visual', {}).get('arms', {}))
                for t in result.get('product_identity_evidence', {}).get('targets', [])]
        self.counters.add('b0_raw_visual_arms', sum('B0' in a for a in arms))
        counters = dict(self.counters.values)
        if counters['b0_index_search_calls'] or counters['b0_raw_visual_arms']:
            raise RuntimeError('B0 retrieval reached the dispatch6 B3-only base')
        parent = self.profile['parent_profile_checksum']
        geometry = result.get('roskachestvo_geometry_v2')
        if geometry is not None:
            geometry['geometry_release_profile_checksum'] = geometry['profile_checksum']
            geometry['profile_checksum'] = self.profile['checksum']
        contract = result.get('target_contract')
        if contract is not None:
            contract['geometry_release_profile_checksum'] = parent
            contract['profile_checksum'] = self.profile['checksum']
        result['b3_only_v1'] = {'policy': R.POLICY, 'profile_checksum': self.profile['checksum'],
                                'runtime_checksum': self.manifest['checksum'], 'parent_profile_checksum': parent,
                                'equivalent_release': self.profile['equivalent_release']['checksum'],
                                'b3_encoder_id': self.ownership['core_B3_arm']['encoder_id'],
                                'b0_gate_encoder_id': self.ownership['object_gate']['encoder_id'],
                                'ranker_checksum': self.ownership['ranker']['ablated'],
                                'selection_raw_visual_arms': arms, 'role_counters': counters,
                                'public_status': result.get('decision'), 'calibrated': False, 'probability': None,
                                'release_admitted': True, 'seconds': time.perf_counter() - started}
        return result


class RepairRelease:
    """RecognitionRepairV1 over the dispatch6 base; answers carry the release profile checksum."""

    def __init__(self, root, profile_path):
        from rshb_vine.recognition_repair_v1.runtime import RecognitionRepairV1
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        base = B3OnlyReleaseD6(root, profile['parent_profile'])
        if base.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded dispatch6 base differs from the release profile')
        inner = RecognitionRepairV1(root, profile['combined_profile'], base_release=base)
        if inner.profile['checksum'] != profile['combined_profile_checksum'] or inner.release is not base:
            raise ValueError('Combined repair runtime is not the pinned composition over this base')
        if inner.profile['components'] != profile['components']:
            raise ValueError('Combined repair components differ from the release profile')
        self.root, self.profile, self.base, self.inner = root, profile, base, inner
        self.manifest = seal({'kind': 'recognition-repair-v1-release-runtime', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': inner.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'combined_profile_checksum': profile['combined_profile_checksum'],
                              'activated': True, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        result = self.inner.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        inner = result['recognition_repair_v1']
        if inner['base_profile_checksum'] != self.profile['parent_profile_checksum']:
            raise RuntimeError('Combined runtime reports another base than the release profile')
        inner.update(release_profile_checksum=self.profile['checksum'], release_admitted=True)
        for block in ('roskachestvo_geometry_v2', 'target_contract'):
            if result.get(block) is not None:
                result[block]['combined_profile_checksum'] = result[block]['profile_checksum']
                result[block]['profile_checksum'] = self.profile['checksum']
        result['recognition_repair_release_v1'] = {
            'profile_checksum': self.profile['checksum'], 'runtime_checksum': self.manifest['checksum'],
            'base_profile_checksum': self.profile['parent_profile_checksum'],
            'combined_profile_checksum': self.profile['combined_profile_checksum'],
            'equivalent_base_release': EQUIVALENT_CHECKSUM, 'calibrated': False, 'probability': None}
        return result
