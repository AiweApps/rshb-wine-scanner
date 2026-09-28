"""Activatable B3-only release: the frozen candidate-v3 composition over the dispatch5 geometry-v2 chain.

Candidate v3 (runtime.B3OnlyRecognition) hardcodes the dispatch4 geometry parent, whose composite pins factory F4;
activation changes the factory, so the release rebuilds the same composition over byte-identical dispatch5 copies.
Gate, B3 arm, split, single-arm dual/core, ablated ranker, graph proof and response block are the v3 ones.
The B4343 gallery receipt binds pins.code_sha(); the only accepted difference is recognition_factory.py through
the sealed gallery code migration (archived F4 bytes -> live F5).
"""
from pathlib import Path
import time

import numpy as np

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.b3_only_v1 import runtime as R
from rshb_vine.b3_only_v1 import split as S

KIND = 'recognition-b3-only-v1-release-profile'
CURRENT_POINTER = 'config/recognition-current.json'
PARENT_PROFILE = 'config/recognition-geometry-v2-dispatch5.json'
TARGET_PROFILE = 'config/recognition-target-contract-v2-dispatch5.json'
GALLERY_MIGRATION = 'config/b3-only-v1-gallery-code-migration.json'
FACTORY = 'rshb_vine/recognition_factory.py'


def load_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile.get('policy') != R.POLICY:
        raise ValueError('Unsupported B3-only release profile')
    if profile['parent_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError('B3-only release must not depend on the mutable current pointer')
    if profile['parent_profile'] != PARENT_PROFILE or sha256(local_path(root, PARENT_PROFILE)) != profile['parent_profile_sha256']:
        raise ValueError('Pinned dispatch5 geometry profile bytes changed')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('B3-only release pinned file changed: ' + path)
    if profile['arm_inputs'] != R.ARM_INPUTS or profile['ablated_features'] != list(R.ABLATED_FEATURES):
        raise ValueError('B3-only release inputs differ from the candidate protocol')
    return profile


def describe_release(root, profile_path):
    from rshb_vine.recognition_factory import describe
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    parent = describe(root, profile['parent_profile'])
    if parent['profile_checksum'] != profile['parent_profile_checksum']:
        raise ValueError('Parent geometry profile differs from the B3-only release profile')
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'], 'policy': R.POLICY,
            'composition': parent['composition'] + ' -> B3-only SKU retrieval (stage5 4343) + B0 context-only object gate',
            'public_answer': 'uncertain proposal; slug never confirmed', 'routes': profile['routes'],
            'model_checksum': profile['ablated_ranker_checksum'], 'parent_ranker_checksum': profile['parent_ranker_checksum'],
            'equivalent_candidate': profile['equivalent_candidate'], 'parent': parent,
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


def _gallery(root, profile, arm):
    """reindex.load_gallery checks with pins.code_sha() mapped through the sealed factory migration only."""
    from rshb_vine.catalog_training_evaluation_v2 import pins, reindex
    migration = verify(read_json(local_path(root, GALLERY_MIGRATION)))
    if migration['checksum'] != profile['gallery_code_migration_checksum'] or migration['logical_path'] != FACTORY:
        raise ValueError('Gallery code migration differs from the release profile')
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


class B3OnlyRelease:
    def __init__(self, root, profile_path):
        from rshb_vine.catalog_training_evaluation_v2 import adapter, reindex
        from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
        from rshb_vine.visual_core import LabelFirstIndex
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        geometry = GeometryReleaseRecognition(root, PARENT_PROFILE)
        if geometry.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Loaded geometry release differs from the pinned dispatch5 parent')
        target = geometry.inner
        if target.profile['checksum'] != profile['target_profile_checksum']:
            raise ValueError('Geometry parent is not the pinned dispatch5 target-contract profile')
        runtime = adapter.recognition_runtime(target)
        a = {k: str(root / v) for k, v in R.ARM_INPUTS.items()}
        arm, model_path = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                              a['fit_admission'])
        if arm['arm'] != 'B4343' or arm['encoder_id'] != profile['b3_encoder_id']:
            raise ValueError('B3-only release is frozen to stage5 checkpoint 4343')
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
            raise ValueError('Gate/core encoder identities differ from the release profile')
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
        self.manifest = seal({'kind': 'b3-only-v1-release-runtime', 'policy': R.POLICY,
                              'profile_checksum': profile['checksum'], 'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': geometry.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'equivalent_candidate': profile['equivalent_candidate']['checksum'],
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
            raise RuntimeError('B0 retrieval reached the B3-only release')
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
                                'equivalent_candidate': self.profile['equivalent_candidate']['checksum'],
                                'b3_encoder_id': self.ownership['core_B3_arm']['encoder_id'],
                                'b0_gate_encoder_id': self.ownership['object_gate']['encoder_id'],
                                'ranker_checksum': self.ownership['ranker']['ablated'],
                                'selection_raw_visual_arms': arms, 'role_counters': counters,
                                'public_status': result.get('decision'), 'calibrated': False, 'probability': None,
                                'release_admitted': True, 'seconds': time.perf_counter() - started}
        return result
