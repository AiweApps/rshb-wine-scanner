"""Single profile-kind dispatch for HTTP, replay and describe; unknown kinds fail closed."""
from pathlib import Path

from rshb_vine.io import local_path, read_json

CURRENT = 'config/recognition-current.json'
RUNTIME_KIND = 'recognition-runtime-profile-v1'
COMPOSITE_KIND = 'recognition-composite-profile-v1'
SYSTEMIC_KIND = 'recognition-systemic-candidate-profile-v1'
TARGET_KIND = 'recognition-target-contract-v2-profile'
GEOMETRY_KIND = 'recognition-geometry-v2-release-profile'
B3_ONLY_KIND = 'recognition-b3-only-v1-release-profile'
B3_ONLY_D6_KIND = 'recognition-b3-only-v1-dispatch6-profile'
REPAIR_KIND = 'recognition-repair-v1-release-profile'
B3_ONLY_D7_KIND = 'recognition-b3-only-v1-dispatch7-profile'
REPAIR_V2_KIND = 'recognition-repair-v2-release-profile'
B3_ONLY_D8_KIND = 'recognition-b3-only-v1-dispatch8-profile'
ZERO_TARGET_KIND = 'recognition-zero-target-release-v1-profile'
B3_ONLY_D9_KIND = 'recognition-b3-only-v1-dispatch9-profile'
TEXT_EVIDENCE_KIND = 'recognition-text-evidence-release-v2-profile'
B3_ONLY_D10_KIND = 'recognition-b3-only-v1-dispatch10-profile'
EVIDENCE_CONSISTENCY_KIND = 'recognition-evidence-consistency-release-v1-profile'
B3_ONLY_D11_KIND = 'recognition-b3-only-v1-dispatch11-profile'
ATLAS_REPAIR_KIND = 'recognition-atlas-repair-release-v1-profile'
B3_ONLY_D12_KIND = 'recognition-b3-only-v1-dispatch12-profile'
RELEASE_NEXT_KIND = 'recognition-release-next-v1-profile'
BOTTLE_KINDS = (TARGET_KIND, GEOMETRY_KIND, B3_ONLY_KIND, B3_ONLY_D6_KIND, REPAIR_KIND, B3_ONLY_D7_KIND, REPAIR_V2_KIND,
                B3_ONLY_D8_KIND, ZERO_TARGET_KIND, B3_ONLY_D9_KIND, TEXT_EVIDENCE_KIND, B3_ONLY_D10_KIND,
                EVIDENCE_CONSISTENCY_KIND, B3_ONLY_D11_KIND, ATLAS_REPAIR_KIND, B3_ONLY_D12_KIND, RELEASE_NEXT_KIND)


def profile_kind(root, profile_path=CURRENT):
    return read_json(local_path(root, profile_path)).get('kind')


def load_runtime(root, profile_path=CURRENT):
    root = Path(root).resolve()
    kind = profile_kind(root, profile_path)
    if kind == RUNTIME_KIND:
        from rshb_vine.recognition_runtime import RecognitionRuntime
        return RecognitionRuntime(root, local_path(root, profile_path))
    if kind == COMPOSITE_KIND:
        from rshb_vine.coherent_challenger_v1.active import CoherentRecognition
        return CoherentRecognition(root, profile_path)
    if kind == SYSTEMIC_KIND:
        from rshb_vine.systemic_ranking_v2.runtime import SystemicRecognition
        return SystemicRecognition(root, profile_path)
    if kind == TARGET_KIND:
        from rshb_vine.target_contract_v2.runtime import TargetContractRecognition
        return TargetContractRecognition(root, profile_path)
    if kind == GEOMETRY_KIND:
        from rshb_vine.roskachestvo_geometry_v2.release import GeometryReleaseRecognition
        return GeometryReleaseRecognition(root, profile_path)
    if kind == B3_ONLY_KIND:
        from rshb_vine.b3_only_v1.release import B3OnlyRelease
        return B3OnlyRelease(root, profile_path)
    if kind == B3_ONLY_D6_KIND:
        from rshb_vine.recognition_repair_v1.release import B3OnlyReleaseD6
        return B3OnlyReleaseD6(root, profile_path)
    if kind == REPAIR_KIND:
        from rshb_vine.recognition_repair_v1.release import RepairRelease
        return RepairRelease(root, profile_path)
    if kind == B3_ONLY_D7_KIND:
        from rshb_vine.recognition_repair_v2.release import B3OnlyReleaseD7
        return B3OnlyReleaseD7(root, profile_path)
    if kind == REPAIR_V2_KIND:
        from rshb_vine.recognition_repair_v2.release import RepairV2Release
        return RepairV2Release(root, profile_path)
    if kind == B3_ONLY_D8_KIND:
        from rshb_vine.zero_target_release_v1.release import B3OnlyReleaseD8
        return B3OnlyReleaseD8(root, profile_path)
    if kind == ZERO_TARGET_KIND:
        from rshb_vine.zero_target_release_v1.release import ZeroTargetRelease
        return ZeroTargetRelease(root, profile_path)
    if kind == B3_ONLY_D9_KIND:
        from rshb_vine.text_evidence_repair_v2.release import B3OnlyReleaseD9
        return B3OnlyReleaseD9(root, profile_path)
    if kind == TEXT_EVIDENCE_KIND:
        from rshb_vine.text_evidence_repair_v2.release import TextEvidenceRelease
        return TextEvidenceRelease(root, profile_path)
    if kind == B3_ONLY_D10_KIND:
        from rshb_vine.evidence_consistency_release_v1.release import B3OnlyReleaseD10
        return B3OnlyReleaseD10(root, profile_path)
    if kind == EVIDENCE_CONSISTENCY_KIND:
        from rshb_vine.evidence_consistency_release_v1.release import EvidenceConsistencyRelease
        return EvidenceConsistencyRelease(root, profile_path)
    if kind == B3_ONLY_D11_KIND:
        from rshb_vine.atlas_repair_release_v1.release import B3OnlyReleaseD11
        return B3OnlyReleaseD11(root, profile_path)
    if kind == ATLAS_REPAIR_KIND:
        from rshb_vine.atlas_repair_release_v1.release import AtlasRepairRelease
        return AtlasRepairRelease(root, profile_path)
    if kind == B3_ONLY_D12_KIND:
        from rshb_vine.release_next_v1.release import B3OnlyReleaseD12
        return B3OnlyReleaseD12(root, profile_path)
    if kind == RELEASE_NEXT_KIND:
        from rshb_vine.release_next_v1.release import ReleaseNextRelease
        return ReleaseNextRelease(root, profile_path)
    raise ValueError('Unsupported recognition profile kind: ' + str(kind))


def describe(root, profile_path=CURRENT):
    root = Path(root).resolve()
    kind = profile_kind(root, profile_path)
    if kind == RUNTIME_KIND:
        from rshb_vine.recognition_runtime import describe_profile
        return describe_profile(root, local_path(root, profile_path))
    if kind == COMPOSITE_KIND:
        from rshb_vine.coherent_challenger_v1.active import describe_composite
        return describe_composite(root, profile_path)
    if kind == SYSTEMIC_KIND:
        from rshb_vine.systemic_ranking_v2.release import describe_systemic
        return describe_systemic(root, profile_path)
    if kind == TARGET_KIND:
        from rshb_vine.target_contract_v2.release import describe_target_contract
        return describe_target_contract(root, profile_path)
    if kind == GEOMETRY_KIND:
        from rshb_vine.roskachestvo_geometry_v2.release import describe_geometry_release
        return describe_geometry_release(root, profile_path)
    if kind == B3_ONLY_KIND:
        from rshb_vine.b3_only_v1.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D6_KIND:
        from rshb_vine.recognition_repair_v1.release import describe_base
        return describe_base(root, profile_path)
    if kind == REPAIR_KIND:
        from rshb_vine.recognition_repair_v1.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D7_KIND:
        from rshb_vine.recognition_repair_v2.release import describe_base
        return describe_base(root, profile_path)
    if kind == REPAIR_V2_KIND:
        from rshb_vine.recognition_repair_v2.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D8_KIND:
        from rshb_vine.zero_target_release_v1.release import describe_base
        return describe_base(root, profile_path)
    if kind == ZERO_TARGET_KIND:
        from rshb_vine.zero_target_release_v1.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D9_KIND:
        from rshb_vine.text_evidence_repair_v2.release import describe_base
        return describe_base(root, profile_path)
    if kind == TEXT_EVIDENCE_KIND:
        from rshb_vine.text_evidence_repair_v2.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D10_KIND:
        from rshb_vine.evidence_consistency_release_v1.release import describe_base
        return describe_base(root, profile_path)
    if kind == EVIDENCE_CONSISTENCY_KIND:
        from rshb_vine.evidence_consistency_release_v1.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D11_KIND:
        from rshb_vine.atlas_repair_release_v1.release import describe_base
        return describe_base(root, profile_path)
    if kind == ATLAS_REPAIR_KIND:
        from rshb_vine.atlas_repair_release_v1.release import describe_release
        return describe_release(root, profile_path)
    if kind == B3_ONLY_D12_KIND:
        from rshb_vine.release_next_v1.release import describe_base
        return describe_base(root, profile_path)
    if kind == RELEASE_NEXT_KIND:
        from rshb_vine.release_next_v1.release import describe_release
        return describe_release(root, profile_path)
    raise ValueError('Unsupported recognition profile kind: ' + str(kind))


def create_app(pipeline):
    """HTTP routes of the loaded profile: target-contract kinds add their routes, every other kind keeps the shared API."""
    if pipeline.profile.get('kind') in BOTTLE_KINDS:
        from rshb_vine.target_contract_v2.runtime import create_app as target_app
        return target_app(pipeline)
    from rshb_vine.recognition_api import create_app as shared_app
    return shared_app(pipeline)


def result_profile_checksum(result):
    """Profile that produced an HTTP/replay result: release-next/atlas-repair/evidence-consistency/text-evidence/zero-target/repair release, B3-only, geometry, target contract, composite, baseline."""
    release_next = result.get('release_next_v1')
    if release_next is not None and release_next.get('profile_checksum') is not None:
        return release_next['profile_checksum']
    atlas = result.get('atlas_repair_release_v1')
    if atlas is not None and atlas.get('profile_checksum') is not None:
        return atlas['profile_checksum']
    consistency = result.get('evidence_consistency_release_v1')
    if consistency is not None and consistency.get('profile_checksum') is not None:
        return consistency['profile_checksum']
    text_evidence = result.get('text_evidence_release_v2')
    if text_evidence is not None and text_evidence.get('profile_checksum') is not None:
        return text_evidence['profile_checksum']
    zero_target = result.get('zero_target_release_v1')
    if zero_target is not None and zero_target.get('profile_checksum') is not None:
        return zero_target['profile_checksum']
    repair_v2 = result.get('recognition_repair_release_v2')
    if repair_v2 is not None:
        return repair_v2['profile_checksum']
    candidate_v2 = result.get('recognition_repair_v2')
    if candidate_v2 is not None:
        return candidate_v2['profile_checksum']
    repair =result.get('recognition_repair_release_v1')
    if repair is not None:
        return repair['profile_checksum']
    b3_only = result.get('b3_only_v1')
    if b3_only is not None:
        return b3_only['profile_checksum']
    geometry = result.get('roskachestvo_geometry_v2')
    if geometry is not None and 'profile_checksum' in geometry:
        return geometry['profile_checksum']
    contract = result.get('target_contract')
    if contract is not None:
        return contract.get('profile_checksum')
    composite = result.get('recognition_profile')
    if composite is not None:
        return composite.get('profile_checksum')
    return result.get('recognition_runtime', {}).get('profile_checksum')


def selector_model_checksum(root, profile_path=CURRENT):
    from rshb_vine.io import verify
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') == COMPOSITE_KIND:
        return profile['selector_model_checksum']
    if profile.get('kind') == RUNTIME_KIND:
        return profile['selector']['model_checksum']
    if profile.get('kind') == SYSTEMIC_KIND:
        return profile['model_checksum']
    if profile.get('kind') in (B3_ONLY_KIND, B3_ONLY_D6_KIND, REPAIR_KIND, B3_ONLY_D7_KIND, REPAIR_V2_KIND, B3_ONLY_D8_KIND,
                               ZERO_TARGET_KIND, B3_ONLY_D9_KIND, TEXT_EVIDENCE_KIND, B3_ONLY_D10_KIND,
                               EVIDENCE_CONSISTENCY_KIND, B3_ONLY_D11_KIND, ATLAS_REPAIR_KIND, B3_ONLY_D12_KIND,
                               RELEASE_NEXT_KIND):
        return profile['ablated_ranker_checksum']
    if profile.get('kind') in BOTTLE_KINDS:
        return profile['parent_model_checksum']
    raise ValueError('Unsupported recognition profile kind: ' + str(profile.get('kind')))
