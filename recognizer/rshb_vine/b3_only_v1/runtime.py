"""B3-only SKU candidate over the byte-pinned geometry-v2 release closure (never the mutable current pointer).

Ownership after construction:
  object gate      original B0 encoder on context views + frozen coef/threshold (split.SplitWineObjectStage)
  base holder      stage5 B3 encoder/index/eid (the same shared-cache delegates as the core arm)
  core B3 arm      adapter.substitute: stage5 B3 encoder/index; parent B3 objects tripwired
  control dual     B0xB3 rank fusion replaced by SingleArmDual (no B3xB3 self-agreement)
  selection        raw_visual carries only the B3 arm; systemic ranker is a fixed ablation (below)
"""
from copy import deepcopy
from pathlib import Path
import time

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.b3_only_v1 import split as S

KIND = 'recognition-b3-only-v1-candidate-profile'
POLICY = 'b3-only-sku-retrieval-ocr-with-b0-object-gate-v1'
PROFILE = 'config/recognition-b3-only-v1-candidate-v3.json'
CURRENT_POINTER = 'config/recognition-current.json'
PARENT_PROFILE = 'config/recognition-geometry-v2-release.json'
PARENT_SHA256 = '226017a26ee9bbd43a41968d70bbccc66fb6078437881a9fcb8f50248c063e3a'
PARENT_CHECKSUM = 'ccfdb7d9a1da6cf58ef3098838a4b4ad0ab610aed440df0af766954711c2bfbe'
STAGE5 = 'runs/catalog-training-data-v2/stage5'
ARM_INPUTS = {
    'encoder_dir': STAGE5 + '/evaluator/actual-fit-v1/B4343/encoder',
    'gallery_dir': STAGE5 + '/evaluator/actual-fit-v1/B4343/gallery',
    'export_receipt': STAGE5 + '/evaluator/actual-fit-v1/B4343-export.stdout',
    'recipe': STAGE5 + '/launch/recipe.json',
    'checkpoint': STAGE5 + '/launch/replacement-v1/download/fit/checkpoint-004343.pt',
    'fit_admission': STAGE5 + '/launch/root-fit-admission.json',
}
ABLATED_FEATURES = ('vis.B0_context.gap', 'vis.B0_context.rr', 'vis.B0_label.gap', 'vis.B0_label.rr',
                    'vis.small_label_x_B0_label_rr', 'vis.label_agree', 'vis.context_agree', 'vis.top1_count',
                    'pool.control_proposal')
SOURCES = ('rshb_vine/b3_only_v1/__init__.py', 'rshb_vine/b3_only_v1/split.py', 'rshb_vine/b3_only_v1/runtime.py',
           'scripts/b3_only_v1.py')
DEVICE = 'mps'


def ablate(model):
    """Fixed ablation of B0 visual, cross-encoder agreement and control-boost features; no refit."""
    names = model['features']
    missing = [n for n in ABLATED_FEATURES if n not in names]
    if missing:
        raise ValueError('Ablated features absent from the systemic schema: ' + ', '.join(missing))
    body = deepcopy({k: v for k, v in model.items() if k != 'checksum'})
    body['weights'] = [0. if n in ABLATED_FEATURES else w for n, w in zip(names, model['weights'])]
    body['effective_weights'] = {n: w / s for n, w, s in zip(names, body['weights'], model['scale'])}
    body['ablation'] = {'kind': 'b3-only-v1-fixed-weight-ablation', 'parent_model_checksum': model['checksum'],
                        'zeroed': list(ABLATED_FEATURES), 'refit': False,
                        'reason': 'B0 visual, B0xB3 agreement and control boost excluded from the final score'}
    return seal(body)


def load_profile(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile.get('policy') != POLICY:
        raise ValueError('Unsupported B3-only candidate profile')
    if profile['parent_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError('B3-only candidate must not depend on the mutable current pointer')
    if (profile['parent_profile'] != PARENT_PROFILE or profile['parent_profile_sha256'] != PARENT_SHA256
            or sha256(local_path(root, PARENT_PROFILE)) != PARENT_SHA256):
        raise ValueError('Pinned geometry-v2 release profile bytes changed')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('B3-only candidate pinned file changed: ' + path)
    if profile['arm_inputs'] != ARM_INPUTS or profile['ablated_features'] != list(ABLATED_FEATURES):
        raise ValueError('B3-only candidate inputs differ from the module protocol')
    return profile


def freeze_body(root):
    """Profile body pinning every source and model file; sealed by the CLI freeze action."""
    root = Path(root).resolve()
    pins = {p: sha256(root / p) for p in SOURCES}
    for key in ('export_receipt', 'recipe', 'checkpoint', 'fit_admission'):
        pins[ARM_INPUTS[key]] = sha256(root / ARM_INPUTS[key])
    for folder in ('encoder_dir', 'gallery_dir'):
        for f in sorted((root / ARM_INPUTS[folder]).iterdir()):
            if f.is_file():
                pins[ARM_INPUTS[folder] + '/' + f.name] = sha256(f)
    parent = verify(read_json(root / PARENT_PROFILE))
    systemic = verify(read_json(root / 'config/recognition-systemic-v2-dispatch4.json'))
    model = verify(read_json(root / systemic['model']))
    pins[systemic['model']] = sha256(root / systemic['model'])
    return {'kind': KIND, 'policy': POLICY, 'name': 'b3-only-v1-candidate',
            'parent_profile': PARENT_PROFILE, 'parent_profile_sha256': PARENT_SHA256,
            'parent_profile_checksum': parent['checksum'], 'arm_inputs': ARM_INPUTS,
            'ablated_features': list(ABLATED_FEATURES), 'parent_ranker_checksum': model['checksum'],
            'ablated_ranker_checksum': ablate(model)['checksum'], 'pins_sha256': pins, 'device': DEVICE,
            'port': 8184, 'release_status': 'experimental_candidate_not_admitted', 'activated': False,
            'calibrated': False, 'probability': None, 'public_slug': 'never confirmed',
            'rollback': 'stop 8184; config/recognition-current.json and 8175 are never modified by this candidate',
            'routes': parent['routes']}


def _graph_refs(root_obj, targets, limit=500000):
    """Attribute paths inside the loaded graph that still reach the given objects (hidden-consumer proof)."""
    seen, found, stack = set(), [], [(root_obj, 'pipeline')]
    ids = {id(t): name for name, t in targets.items()}
    while stack and len(seen) < limit:
        obj, path = stack.pop()
        if id(obj) in ids and path != 'pipeline':
            found.append({'path': path, 'object': ids[id(obj)]})
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            children = [(v, path + '[' + repr(k)[:40] + ']') for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            children = [(v, path + '[%d]' % i) for i, v in enumerate(obj)]
        elif _owned(obj):
            children = [(v, path + '.' + k) for k, v in vars(obj).items()]
        else:
            children = []
        stack.extend(c for c in children if _owned(c[0]) or isinstance(c[0], (dict, list, tuple))
                     and (len(c[0]) <= 256 or any(_owned(v) for v in (c[0].values() if isinstance(c[0], dict) else c[0]))))
    if stack:
        raise ValueError('Graph scan limit reached; ownership proof incomplete')
    return found, len(seen)


def _owned(obj):
    return (type(obj).__module__ or '').startswith(('rshb_vine', 'scripts.')) and hasattr(obj, '__dict__')


class B3OnlyRecognition:
    def __init__(self, root, profile_path=PROFILE):
        from rshb_vine.catalog_training_evaluation_v2 import adapter, pins, reindex
        from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        geometry = GeometryReleaseRecognition(root, PARENT_PROFILE)
        if geometry.profile['checksum'] != PARENT_CHECKSUM or profile['parent_profile_checksum'] != PARENT_CHECKSUM:
            raise ValueError('Loaded geometry release differs from the pinned parent')
        target = geometry.inner
        if target.profile['checksum'] != pins.PROFILE_CHECKSUM:
            raise ValueError('Geometry parent is not the target-contract profile bound by the B3 adapter')
        runtime = adapter.recognition_runtime(target)
        a = {k: str(root / v) for k, v in ARM_INPUTS.items()}
        arm, model_path = reindex.encoder_arm(a['encoder_dir'], a['export_receipt'], a['recipe'], a['checkpoint'],
                                              a['fit_admission'])
        if arm['arm'] != 'B4343':
            raise ValueError('B3-only candidate is frozen to stage5 checkpoint 4343')
        encoder, index, gallery = adapter.arm_components(arm, model_path, a['gallery_dir'], profile['device'])
        substitution = adapter.substitute(target, arm, encoder, index, gallery)

        base, core = runtime.holders['B0'], runtime.holders['B3']
        instance = runtime.control.expanded.core.parent.parent.parent
        b0_eid = runtime.arms['B0'][0]
        if instance.base is not base or instance.spec['encoder_id'] != b0_eid or base.encoder_id != b0_eid:
            raise ValueError('Object gate is not bound to the original B0 holder')
        if core.eid != arm['encoder_id'] or core.eid == b0_eid:
            raise ValueError('Core arm is not the substituted stage5 B3')
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
        ranker = ablate(selection.ranker)
        if (selection.ranker['checksum'] != profile['parent_ranker_checksum']
                or ranker['checksum'] != profile['ablated_ranker_checksum']):
            raise ValueError('Systemic ranker or its fixed ablation differs from the profile')
        selection.use_ranker(ranker)

        refs, visited = _graph_refs(geometry, {'b0_index_raw': b0_index_raw, 'b0_encoder_raw': b0_encoder_raw,
                                               'b0_encoder_wrapped': b0_encoder_wrapped})
        stray = [r for r in refs if r['object'] == 'b0_index_raw' or not (
            r['path'].endswith('.gate_encoder') or '._owners[' in r['path']
            or r['object'] == 'b0_encoder_raw' and r['path'].endswith('.gate_encoder._encoder'))]
        self.ownership = {
            'object_gate': {'encoder_id': b0_eid, 'spec_checksum': instance.spec['checksum'],
                            'threshold': instance.spec['threshold'], 'views': 'context only, original B0 encoder'},
            'instance_mro': mro,
            'base_holder': {'encoder_id': core.eid, 'shared_with_core_arm': base.encoder is core.encoder,
                            'gallery_checksum': gallery['checksum'], 'vectors_sha256': gallery['vectors_sha'],
                            'consumers': ['SplitWineObjectStage retrieval', 'OrientedGeometryPipeline rotated retrieval']},
            'control_dual': 'ReferencePhrasePreservation kept; its RankFused/Alternative parent replaced by SingleArmDual',
            'core_consensus': 'GuardedConsensus combine(baseline, baseline): title/variant guards kept, no second B3 vote',
            'b0_encoder_guard': 'raw B0 encode raises outside split gate context',
            'core_B3_arm': {'encoder_id': core.eid, 'substitution_checksum': substitution['checksum']},
            'selection_raw_visual_arms': sorted(runtime.arms),
            'control': 'computed with B3 upstream; feeds pool candidates only, pool.control_proposal weight 0',
            'ranker': {'parent': profile['parent_ranker_checksum'], 'ablated': ranker['checksum'],
                       'zeroed': list(ABLATED_FEATURES)},
            'graph_scan': {'objects_visited': visited, 'b0_references': refs, 'stray': stray,
                           'rule': 'B0 encoder reachable only as instance.gate_encoder or cache owner registry; raw B0 index unreachable'},
            'gallery_reference_order_equals_serving': True}
        if not any(r['path'].endswith('.gate_encoder') for r in refs):
            raise ValueError('Graph scan did not reach the object-gate encoder; proof incomplete')
        if stray:
            raise ValueError('Hidden B0 consumer remains in the loaded graph: ' + repr(stray[:3]))
        self.geometry, self.target, self.runtime, self.profile = geometry, target, runtime, profile
        self.manifest = seal({'kind': 'b3-only-v1-runtime', 'policy': POLICY, 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': geometry.manifest['checksum'],
                              'parent_profile_checksum': PARENT_CHECKSUM, 'ownership': self.ownership,
                              'arm': arm, 'activated': False, 'calibration_status': 'unknown'})

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
            raise RuntimeError('B0 retrieval reached the B3-only candidate')
        geometry = result.get('roskachestvo_geometry_v2')
        if geometry is not None:
            geometry['geometry_release_profile_checksum'] = geometry['profile_checksum']
            geometry['profile_checksum'] = self.profile['checksum']
        contract = result.get('target_contract')
        if contract is not None:
            contract['geometry_release_profile_checksum'] = PARENT_CHECKSUM
            contract['profile_checksum'] = self.profile['checksum']
        result['b3_only_v1'] = {'policy': POLICY, 'profile_checksum': self.profile['checksum'],
                                'runtime_checksum': self.manifest['checksum'],
                                'parent_profile_checksum': PARENT_CHECKSUM,
                                'b3_encoder_id': self.ownership['core_B3_arm']['encoder_id'],
                                'b0_gate_encoder_id': self.ownership['object_gate']['encoder_id'],
                                'ranker_checksum': self.ownership['ranker']['ablated'],
                                'selection_raw_visual_arms': arms, 'role_counters': counters,
                                'public_status': result.get('decision'), 'calibrated': False, 'probability': None,
                                'release_admitted': False, 'seconds': time.perf_counter() - started}
        return result


def describe(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    return {'kind': KIND, 'policy': POLICY, 'profile_checksum': profile['checksum'],
            'parent_profile': PARENT_PROFILE, 'parent_profile_checksum': PARENT_CHECKSUM,
            'arm_inputs': ARM_INPUTS, 'ablated_features': list(ABLATED_FEATURES),
            'parent_ranker_checksum': profile['parent_ranker_checksum'],
            'ablated_ranker_checksum': profile['ablated_ranker_checksum'], 'routes': profile['routes'],
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}
