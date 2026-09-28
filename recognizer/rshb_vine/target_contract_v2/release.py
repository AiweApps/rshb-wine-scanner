"""Describe a target-contract profile for scripts/recognize.py describe without loading models.

runtime.py keeps the loader; this module reports the profile, its strictly verified systemic parent and the
unchanged uncalibrated contract.
"""
from pathlib import Path

from rshb_vine.target_contract_v2.runtime import KIND, load_profile


def describe_target_contract(root, profile_path):
    from rshb_vine.recognition_factory import describe
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    parent = describe(root, profile['parent_profile'])
    if parent['profile_checksum'] != profile['parent_profile_checksum']:
        raise ValueError('Parent systemic profile differs from the target-contract profile')
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'name': profile['name'],
            'composition': parent['composition'] + ' -> target-contract-v2 bottle list/ROI selection',
            'public_answer': 'parent_selected_product_unchanged', 'policy': profile['policy'],
            'routes': profile['routes'], 'model_checksum': profile['parent_model_checksum'], 'parent': parent,
            'supersedes': profile.get('supersedes'), 'release_status': profile['release_status'],
            'calibration_status': 'unknown', 'probability': None, 'models_loaded': False}
