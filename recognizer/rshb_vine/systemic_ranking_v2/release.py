"""Describe the systemic release profile for scripts/recognize.py describe without loading models.

The frozen runtime.py keeps the loader; this module only reports the profile, its strictly verified parent
composite and the unchanged uncalibrated contract.
"""
from pathlib import Path

from rshb_vine.systemic_ranking_v2.runtime import KIND, load_profile


def describe_systemic(root, profile_path):
    from rshb_vine.coherent_challenger_v1.active import describe_composite
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    parent = describe_composite(root, profile['parent_profile'])
    if parent['profile_checksum'] != profile['parent_profile_checksum']:
        raise ValueError('Parent composite differs from the systemic profile')
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': parent['composition'] + ' -> systemic-ranking-v2 selector',
            'public_answer': 'selected_product', 'model_checksum': profile['model_checksum'],
            'evaluation_checksum': profile['evaluation_checksum'], 'parent': parent,
            'supersedes': profile.get('supersedes'), 'release_status': profile['release_status'],
            'calibration_status': 'unknown', 'probability': None, 'models_loaded': False}
