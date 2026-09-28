"""Activatable composite profile: explicit old103 baseline -> CanvasCloseupRoute -> snapshot-03 identity.

The parent is a byte-frozen baseline path, never the mutable current pointer, so switching
config/recognition-current.json to this profile cannot recurse or break its own pins.
"""
from pathlib import Path
import time

from rshb_vine.io import local_path, read_json, seal, sha256, verify

KIND = 'recognition-composite-profile-v1'
CURRENT_POINTER = 'config/recognition-current.json'


def load_composite(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile.get('composition_id') != 'coherent-challenger-v1':
        raise ValueError('Unsupported composite recognition profile')
    if profile['baseline_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError('Composite profile must not depend on the mutable current pointer')
    if sha256(local_path(root, profile['baseline_profile'])) != profile['baseline_profile_sha256']:
        raise ValueError('Baseline profile bytes changed')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('Composite profile source changed: ' + path)
    return profile


def describe_composite(root, profile_path):
    from rshb_vine.recognition_runtime import describe_profile
    profile = load_composite(root, profile_path)
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': profile['composition'], 'public_answer': 'selected_product',
            'baseline': describe_profile(root, profile['baseline_profile']),
            'identity_registry_checksum': profile['identity']['registry_checksum'],
            'route_policy': profile['route']['policy'], 'evidence': profile['evidence'],
            'release_status': profile['release_status'], 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


class CoherentRecognition:
    def __init__(self, root, profile_path):
        from rshb_vine.coherent_challenger_v1.identity import IdentityAdapter
        from rshb_vine.recognition_runtime import RecognitionRuntime
        from rshb_vine.target_misses_v1.route import CanvasCloseupRoute
        root = Path(root).resolve()
        profile = load_composite(root, profile_path)
        runtime = RecognitionRuntime(root, root / profile['baseline_profile'])
        if (runtime.profile['checksum'] != profile['baseline_profile_checksum']
                or runtime.registry.checksum != profile['selection_pool_registry_checksum']
                or runtime.selection.model['checksum'] != profile['selector_model_checksum']):
            raise ValueError('Baseline runtime differs from the composite profile')
        self.runtime = runtime
        self.route = CanvasCloseupRoute(runtime)
        self.adapter = IdentityAdapter(root, frozen_registry=runtime.registry)
        if (self.adapter.candidate.checksum != profile['identity']['registry_checksum']
                or self.adapter.handoff['checksum'] != profile['identity']['handoff_checksum']):
            raise ValueError('Candidate identity differs from the composite profile')
        self.profile = profile
        self.manifest = seal({'kind': 'coherent-recognition-runtime-v1', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'baseline_runtime': runtime.manifest['checksum'], 'route': self.route.manifest['checksum'],
                              'identity': self.adapter.descriptor, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        result = self.route.recognize(data, roi)
        projected = time.perf_counter()
        result = self.adapter.project(result, copy=False)
        timing = result.setdefault('timing_ms', {})
        timing['identity_projection'] = (time.perf_counter() - projected) * 1000
        timing['total'] = (time.perf_counter() - started) * 1000
        result['recognition_profile'] = {'kind': KIND, 'name': self.profile['name'],
                                         'profile_checksum': self.profile['checksum'],
                                         'baseline_profile_checksum': self.profile['baseline_profile_checksum'],
                                         'calibrated': False, 'probability': None}
        return result
