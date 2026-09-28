"""Opt-in challenger runtime: byte-frozen coherent-v1 composite with the systemic selector swapped in.

The parent composite (old103 -> CanvasCloseupRoute -> snapshot-03) loads through its own pin checks; only
its runtime's selection is replaced by SystemicSelection (same object used by receipt replay). The shared
factory/recognize.py dispatch is pinned by coherent-v1 and is not extended here, so no current pointer,
receipt binding or dispatch pin changes. Rollback is not loading this profile.
"""
from pathlib import Path
import time

from rshb_vine.io import local_path, read_json, seal, sha256, verify

KIND = 'recognition-systemic-candidate-profile-v1'


def load_profile(root, profile_path):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND:
        raise ValueError('Unsupported systemic candidate profile')
    if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
        raise ValueError('Parent composite profile bytes changed')
    for path, expected in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != expected:
            raise ValueError('Systemic candidate source changed: ' + path)
    return profile


class SystemicRecognition:
    def __init__(self, root, profile_path):
        from rshb_vine.coherent_challenger_v1.active import CoherentRecognition
        from rshb_vine.systemic_ranking_v2.selection import SystemicSelection, attach
        root = Path(root).resolve()
        profile = load_profile(root, profile_path)
        parent = CoherentRecognition(root, profile['parent_profile'])
        if parent.profile['checksum'] != profile['parent_profile_checksum']:
            raise ValueError('Parent composite differs from the systemic profile')
        legacy = parent.runtime.selection
        selection = SystemicSelection(root, legacy, model_path=profile['model'], public='systemic')
        if selection.ranker['checksum'] != profile['model_checksum']:
            raise ValueError('Systemic model differs from the profile')
        attach(parent.runtime, selection)
        self.parent, self.selection, self.profile = parent, selection, profile
        self.manifest = seal({'kind': 'systemic-recognition-runtime-v1', 'profile_checksum': profile['checksum'],
                              'runtime_descriptor_checksum': profile['checksum'],
                              'parent': parent.manifest['checksum'], 'model': profile['model_checksum'],
                              'calibration_status': 'unknown'})

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        self.selection.records.clear()
        result = self.parent.recognize(data, roi)
        result['systemic_ranking_v2'] = {'profile_checksum': self.profile['checksum'],
                                         'model_checksum': self.profile['model_checksum'],
                                         'targets': list(self.selection.records)}
        result['recognition_profile'] = {'kind': KIND, 'name': self.profile['name'],
                                         'profile_checksum': self.profile['checksum'],
                                         'parent_profile_checksum': self.profile['parent_profile_checksum'],
                                         'calibrated': False, 'probability': None}
        result.setdefault('timing_ms', {})['systemic_total'] = (time.perf_counter() - started) * 1000
        return result
