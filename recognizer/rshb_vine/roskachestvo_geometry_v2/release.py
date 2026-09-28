"""Activatable geometry-v2 release: byte-pinned dispatch4 target-contract profile with the frozen v2 association.

The parent is config/recognition-target-contract-v2-dispatch4.json, never the mutable current pointer. The frozen
runtime.locate and association.GeometryOrphanRecovery of the admitted candidate are reused unchanged; only the
parent profile differs from runtime.GeometryCandidateRecognition, whose hardcoded 0d1c2294 parent stays intact.
"""
from copy import deepcopy
from pathlib import Path

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.roskachestvo_geometry_v2 import association as A

KIND = 'recognition-geometry-v2-release-profile'
CURRENT_POINTER = 'config/recognition-current.json'


def load_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile.get('policy') != A.POLICY or profile.get('constants') != A.CONSTANTS:
        raise ValueError('Unsupported geometry release profile')
    if profile['parent_profile'] == CURRENT_POINTER or CURRENT_POINTER in profile['pins_sha256']:
        raise ValueError('Geometry release profile must not depend on the mutable current pointer')
    if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
        raise ValueError('Parent target-contract profile bytes changed')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('Geometry release source changed: ' + path)
    return profile


def describe_geometry_release(root, profile_path):
    from rshb_vine.recognition_factory import describe
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    parent = describe(root, profile['parent_profile'])
    if parent['profile_checksum'] != profile['parent_profile_checksum']:
        raise ValueError('Parent target-contract profile differs from the geometry release profile')
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': parent['composition'] + ' -> ' + A.POLICY + ' orphan-label association',
            'public_answer': 'parent_selected_product_unchanged', 'policy': profile['policy'],
            'constants': profile['constants'], 'routes': profile['routes'],
            'model_checksum': profile['parent_model_checksum'], 'parent': parent,
            'supersedes': profile.get('supersedes'), 'release_status': profile['release_status'],
            'identity_admitted': False, 'calibration_status': 'unknown', 'probability': None,
            'models_loaded': False}


class GeometryReleaseRecognition:
    def __init__(self, root, profile_path):
        from rshb_vine.roskachestvo_geometry_v2.runtime import locate
        from rshb_vine.target_contract_v2.runtime import TargetContractRecognition
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        inner = TargetContractRecognition(root, profile['parent_profile'])
        if inner.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Parent target-contract runtime differs from the geometry release profile')
        control, frozen = locate(inner)
        self.recovery = A.GeometryOrphanRecovery(frozen)
        control.orphans = self.recovery
        self.inner, self.profile = inner, profile
        self.manifest = seal({'kind': 'roskachestvo-geometry-v2-release-runtime', 'policy': A.POLICY,
                              'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent_runtime': inner.manifest['checksum'],
                              'parent_profile_checksum': profile['parent_profile_checksum'],
                              'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        self.recovery.events = []
        result = self.inner.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        result['roskachestvo_geometry_v2'] = {'policy': A.POLICY, 'profile_checksum': self.profile['checksum'],
                                              'descriptor_checksum': self.profile['checksum'],
                                              'events': deepcopy(self.recovery.events),
                                              'recovered': [e['instance_id'] for e in self.recovery.events if e['used']]}
        return result
