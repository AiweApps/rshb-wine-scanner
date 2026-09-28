"""Opt-in integrated candidate over the byte-pinned coherent-v1 parent.

Composition: frozen old103 baseline -> NonWineProposalCanvasRoute (C geometry-only eligibility) ->
snapshot-03 identity projection -> primary-target projection (A; only when the descriptor lists it).
C's text pool and B's ranker are never imported. One RecognitionRuntime is the only model holder.
The parent pins are checked except the two dispatch files, so adding this kind to the factory
cannot invalidate the profile; the HTTP transport and every computing source stay pinned.
"""
from pathlib import Path
import time

from rshb_vine.io import local_path, read_json, seal, sha256, verify, write_json

KIND = 'recognition-integrated-profile-v1'
PARENT_KIND = 'recognition-composite-profile-v1'
PARENT_COMPOSITION = 'coherent-challenger-v1'
PARENT = 'config/recognition-coherent-v1.json'
CURRENT_POINTER = 'config/recognition-current.json'
DISPATCH = ('rshb_vine/recognition_factory.py', 'scripts/recognize.py')
COMPONENT_PROFILES = {
    'geometry': 'runs/recognition-miss-repair-v1/profiles/geometry-only-v1a1.json',
    'primary_target': 'config/recognition-primary-target-v1.json',
}
OWN_CODE = ('rshb_vine/integrated_candidate_v1/__init__.py', 'rshb_vine/integrated_candidate_v1/runtime.py',
            'rshb_vine/recognition_miss_repair_v1/geometry.py', 'rshb_vine/target_misses_v1/route.py',
            'rshb_vine/primary_target_v1/policy.py', 'rshb_vine/coherent_challenger_v1/identity.py',
            'rshb_vine/recognition_api.py')


def _check_pins(root, pins, what, skip=()):
    for path, expected in pins.items():
        if path not in skip and sha256(local_path(root, path)) != expected:
            raise ValueError(what + ' source changed: ' + path)


def load_parent(root, profile):
    parent_path = profile['parent_profile']
    if sha256(local_path(root, parent_path)) != profile['parent_profile_sha256']:
        raise ValueError('Parent composite profile bytes changed')
    parent = verify(read_json(local_path(root, parent_path)))
    if (parent['checksum'] != profile['parent_profile_checksum'] or parent.get('kind') != PARENT_KIND
            or parent.get('composition_id') != PARENT_COMPOSITION):
        raise ValueError('Parent composite profile differs from the integrated profile')
    if parent['baseline_profile'] == CURRENT_POINTER or CURRENT_POINTER in parent['pins_sha256']:
        raise ValueError('Parent depends on the mutable current pointer')
    if sha256(local_path(root, parent['baseline_profile'])) != parent['baseline_profile_sha256']:
        raise ValueError('Baseline profile bytes changed')
    _check_pins(root, parent['pins_sha256'], 'Parent', skip=DISPATCH)
    return parent


def load_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    components = profile.get('components') or []
    if profile.get('kind') != KIND or 'geometry' not in components or not set(components) <= set(COMPONENT_PROFILES):
        raise ValueError('Unsupported integrated recognition profile')
    if profile['parent_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError('Integrated profile must not depend on the mutable current pointer')
    if set(profile['component_profiles']) != set(components):
        raise ValueError('Component profiles do not match the component list')
    for name, ref in profile['component_profiles'].items():
        if ref['path'] != COMPONENT_PROFILES[name] or sha256(local_path(root, ref['path'])) != ref['sha256']:
            raise ValueError('Component profile bytes changed: ' + name)
        component = verify(read_json(local_path(root, ref['path'])))
        if component['checksum'] != ref['checksum']:
            raise ValueError('Component profile checksum changed: ' + name)
        _check_pins(root, component['pins_sha256'], 'Component ' + name)
    _check_pins(root, profile['pins_sha256'], 'Integrated profile')
    parent = load_parent(root, profile)
    return profile, parent


def describe(root, profile_path):
    from rshb_vine.recognition_runtime import describe_profile
    root = Path(root).resolve()
    profile, parent = load_profile(root, profile_path)
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': profile['composition'], 'components': profile['components'],
            'component_profiles': profile['component_profiles'], 'excluded': profile['excluded'],
            'public_answer': 'selected_product', 'public_answer_interpretation': profile['public_answer_interpretation'],
            'parent': {'kind': PARENT_KIND, 'profile_checksum': parent['checksum'], 'name': parent['name'],
                       'composition': parent['composition'],
                       'identity_registry_checksum': parent['identity']['registry_checksum'],
                       'baseline': describe_profile(root, parent['baseline_profile'])},
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


def write_profile(root, path, name, components, release_status):
    root = Path(root).resolve()
    out = root / path
    if out.exists():
        raise ValueError('Integrated profile exists; descriptors are immutable')
    parent = verify(read_json(root / PARENT))
    refs = {}
    for c in sorted(components):
        doc = verify(read_json(root / COMPONENT_PROFILES[c]))
        refs[c] = {'path': COMPONENT_PROFILES[c], 'sha256': sha256(root / COMPONENT_PROFILES[c]),
                   'checksum': doc['checksum'], 'name': doc['name']}
    steps = ['old103 baseline (frozen pool)', 'NonWineProposalCanvasRoute (C geometry-only eligibility)',
             'snapshot-03-candidate identity']
    if 'primary_target' in components:
        steps.append('primary-target central/ROI suggestion (A)')
    doc = seal({
        'kind': KIND, 'name': name, 'version': 1, 'composition': ' -> '.join(steps),
        'components': sorted(components), 'component_profiles': refs,
        'parent_profile': PARENT, 'parent_profile_checksum': parent['checksum'],
        'parent_profile_sha256': sha256(root / PARENT),
        'baseline_profile_checksum': parent['baseline_profile_checksum'],
        'selector_model_checksum': parent['selector_model_checksum'],
        'parent_pins_not_rechecked': {p: 'dispatch file (profile-kind routing only); rechecking it would make any '
                                         'factory extension invalidate the parent' for p in DISPATCH},
        'dispatch_sha256_at_freeze': {p: sha256(root / p) for p in DISPATCH},
        'pins_sha256': {p: sha256(root / p) for p in OWN_CODE if (root / p).is_file()},
        'excluded': {'text_pool': 'C text-only pool: diagnostic/shadow only, not loaded, cannot alter ranking',
                     'interaction_ranker_v1': 'B rejected (CV 457->454); no B code or weights loaded',
                     'selector': 'old103 selector unchanged'},
        'public_answer_interpretation': {
            'chosen': 'best_candidate (= /v1/eval/predict slug)',
            'accepted': 'chosen is not null and target_selection.suggested_default is not true; a primary-target '
                        'suggestion is chosen but not accepted (decision ambiguous_target, slug null, probability null)',
            'calibrated': False},
        'release_status': release_status, 'current': False, 'calibrated': False, 'probability': None,
        'fit_run': False, 'weights_changed': False, 'thresholds_changed': False,
        'ocr_reader': 'macOS Vision rev3 (parent profile); Linux not validated'})
    write_json(out, doc)
    return doc


def parent_marker(parent):
    return {'kind': PARENT_KIND, 'name': parent['name'], 'profile_checksum': parent['checksum'],
            'baseline_profile_checksum': parent['baseline_profile_checksum'], 'calibrated': False, 'probability': None}


def finish(result, profile, parent, roi, started=None):
    """Post-route steps shared by the live runtime and the saved-result replay."""
    from rshb_vine.primary_target_v1 import policy
    result['parent_recognition_profile'] = parent_marker(parent)
    timing = result.setdefault('timing_ms', {})
    if 'primary_target' in profile['components']:
        projected = time.perf_counter()
        policy.apply(result, roi, profile['checksum'])
        timing['primary_target'] = (time.perf_counter() - projected) * 1000
    if started is not None:
        timing['total'] = (time.perf_counter() - started) * 1000
    result['recognition_profile'] = {
        'kind': KIND, 'name': profile['name'], 'profile_checksum': profile['checksum'],
        'parent_profile_checksum': parent['checksum'], 'baseline_profile_checksum': parent['baseline_profile_checksum'],
        'components': {k: v['checksum'] for k, v in profile['component_profiles'].items()},
        'calibrated': False, 'probability': None}
    return result


class IntegratedRecognition:
    def __init__(self, root, profile_path):
        from rshb_vine.coherent_challenger_v1.identity import IdentityAdapter
        from rshb_vine.recognition_miss_repair_v1.geometry import POLICY as GEOMETRY_POLICY
        from rshb_vine.recognition_miss_repair_v1.geometry import NonWineProposalCanvasRoute
        from rshb_vine.recognition_runtime import RecognitionRuntime
        root = Path(root).resolve()
        profile, parent = load_profile(root, profile_path)
        runtime = RecognitionRuntime(root, root / parent['baseline_profile'])
        if (runtime.profile['checksum'] != parent['baseline_profile_checksum']
                or runtime.registry.checksum != parent['selection_pool_registry_checksum']
                or runtime.selection.model['checksum'] != parent['selector_model_checksum']):
            raise ValueError('Baseline runtime differs from the parent composite profile')
        self.runtime = runtime
        self.route = NonWineProposalCanvasRoute(runtime)
        self.adapter = IdentityAdapter(root, frozen_registry=runtime.registry)
        if (self.adapter.candidate.checksum != parent['identity']['registry_checksum']
                or self.adapter.handoff['checksum'] != parent['identity']['handoff_checksum']):
            raise ValueError('Candidate identity differs from the parent composite profile')
        self.profile, self.parent = profile, parent
        self.manifest = seal({'kind': 'integrated-recognition-runtime-v1', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_profile_checksum': parent['checksum'],
                              'baseline_runtime': runtime.manifest['checksum'], 'route': self.route.manifest['checksum'],
                              'eligibility_policy': GEOMETRY_POLICY, 'identity': self.adapter.descriptor,
                              'components': profile['components'], 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        result = self.route.recognize(data, roi)
        projected = time.perf_counter()
        result = self.adapter.project(result, copy=False)
        result.setdefault('timing_ms', {})['identity_projection'] = (time.perf_counter() - projected) * 1000
        return finish(result, self.profile, self.parent, roi, started)
