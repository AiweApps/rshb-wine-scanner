"""Bind saved captures/receipts to the exact profile they were produced under, not to the latest current.

A recorded (checksum, sha256) resolves only to an immutable versioned profile file with those bytes whose
own pins still verify; there is no fallback. Old old103 receipts therefore keep their old identity and
selector after the current pointer moves, and new captures bind the composite profile they ran on.
A pinned source may differ from live bytes only through one sealed dispatch-migration entry for that exact
profile, logical path and expected hash, whose archived bytes still carry the expected hash.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from rshb_vine.catalog.product_registry import ProductRegistry
from rshb_vine.io import digest, local_path, read_json, sha256, verify
from rshb_vine.recognition_factory import (ATLAS_REPAIR_KIND, B3_ONLY_D12_KIND, B3_ONLY_D6_KIND, B3_ONLY_D7_KIND, B3_ONLY_D8_KIND,
                                          B3_ONLY_D9_KIND, B3_ONLY_D10_KIND, B3_ONLY_D11_KIND, B3_ONLY_KIND,
                                          COMPOSITE_KIND, EVIDENCE_CONSISTENCY_KIND, GEOMETRY_KIND, REPAIR_KIND,
                                          REPAIR_V2_KIND, RUNTIME_KIND, TARGET_KIND, TEXT_EVIDENCE_KIND,
                                          RELEASE_NEXT_KIND, ZERO_TARGET_KIND, describe, result_profile_checksum)

INTEGRATED_KIND = 'recognition-integrated-profile-v1'
SYSTEMIC_KIND = 'recognition-systemic-candidate-profile-v1'
VERSIONED_PROFILES = ('config/recognition-old103-v1.json', 'config/recognition-coherent-v1.json',
                      'config/recognition-integrated-v1.json', 'config/recognition-integrated-geometry-v1.json',
                      'config/recognition-coherent-v1-dispatch2.json', 'config/recognition-systemic-v2-release.json',
                      'config/recognition-coherent-v1-dispatch3.json', 'config/recognition-systemic-v2-dispatch3.json',
                      'config/recognition-target-contract-v2-release.json',
                      'config/recognition-coherent-v1-dispatch4.json', 'config/recognition-systemic-v2-dispatch4.json',
                      'config/recognition-target-contract-v2-dispatch4.json', 'config/recognition-geometry-v2-release.json',
                      'config/recognition-coherent-v1-dispatch5.json', 'config/recognition-systemic-v2-dispatch5.json',
                      'config/recognition-target-contract-v2-dispatch5.json', 'config/recognition-geometry-v2-dispatch5.json',
                      'config/recognition-b3-only-v1-release.json',
                      'config/recognition-coherent-v1-dispatch6.json', 'config/recognition-systemic-v2-dispatch6.json',
                      'config/recognition-target-contract-v2-dispatch6.json', 'config/recognition-geometry-v2-dispatch6.json',
                      'config/recognition-b3-only-v1-dispatch6.json', 'config/recognition-repair-v1-release.json',
                      'config/recognition-coherent-v1-dispatch7.json', 'config/recognition-systemic-v2-dispatch7.json',
                      'config/recognition-target-contract-v2-dispatch7.json', 'config/recognition-geometry-v2-dispatch7.json',
                      'config/recognition-b3-only-v1-dispatch7.json', 'config/recognition-repair-v2-release.json',
                      'config/recognition-coherent-v1-dispatch8.json', 'config/recognition-systemic-v2-dispatch8.json',
                      'config/recognition-target-contract-v2-dispatch8.json', 'config/recognition-geometry-v2-dispatch8.json',
                      'config/recognition-b3-only-v1-dispatch8.json', 'config/zero-target-release-v1-profile.json',
                      'config/recognition-coherent-v1-dispatch9.json', 'config/recognition-systemic-v2-dispatch9.json',
                      'config/recognition-target-contract-v2-dispatch9.json', 'config/recognition-geometry-v2-dispatch9.json',
                      'config/recognition-b3-only-v1-dispatch9.json', 'config/text-evidence-release-v2-profile.json',
                      'config/recognition-coherent-v1-dispatch10.json', 'config/recognition-systemic-v2-dispatch10.json',
                      'config/recognition-target-contract-v2-dispatch10.json', 'config/recognition-geometry-v2-dispatch10.json',
                      'config/recognition-b3-only-v1-dispatch10.json', 'config/evidence-consistency-release-v1-profile.json',
                      'config/recognition-coherent-v1-dispatch11.json', 'config/recognition-systemic-v2-dispatch11.json',
                      'config/recognition-target-contract-v2-dispatch11.json', 'config/recognition-geometry-v2-dispatch11.json',
                      'config/recognition-b3-only-v1-dispatch11.json', 'config/atlas-repair-release-v1-profile.json',
                      'config/recognition-coherent-v1-dispatch12.json', 'config/recognition-systemic-v2-dispatch12.json',
                      'config/recognition-target-contract-v2-dispatch12.json', 'config/recognition-geometry-v2-dispatch12.json',
                      'config/recognition-b3-only-v1-dispatch12.json', 'config/release-next-v1-profile.json')
MIGRATION_KIND = 'recognition-dispatch-migration-v1'
MIGRATIONS = {'config/recognition-dispatch-migration-v1.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v2.json': ('rshb_vine/recognition_factory.py',
                                                                'scripts/recognize.py'),
              'config/recognition-dispatch-migration-v3.json': ('rshb_vine/recognition_factory.py',
                                                                'scripts/recognize.py'),
              'config/recognition-dispatch-migration-v4.json': ('rshb_vine/recognition_factory.py',
                                                                'scripts/recognize.py'),
              'config/recognition-dispatch-migration-v5.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v6.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v7.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v8.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v9.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v10.json': ('rshb_vine/recognition_factory.py',),
              'config/recognition-dispatch-migration-v11.json': ('rshb_vine/recognition_factory.py',)}


@dataclass(frozen=True)
class BoundProfile:
    path: str
    kind: str
    checksum: str
    sha256: str
    runtime_checksum: str
    product_bundle: str
    bundle_checksum: str
    selector_model_checksum: str

    def registry(self, root):
        registry = ProductRegistry.from_bundle(root, self.product_bundle)
        if registry.bundle_checksum != self.bundle_checksum:
            raise ValueError('Product bundle differs from the bound profile')
        return registry


def bind_profile(root, checksum, profile_sha256):
    root = Path(root).resolve()
    matches = [p for p in VERSIONED_PROFILES if (root / p).is_file() and sha256(root / p) == profile_sha256]
    if len(matches) != 1:
        raise ValueError('No unique verified versioned profile with the recorded SHA256')
    path = matches[0]
    if _verified_checksum(root, path) != checksum:
        raise ValueError('Versioned profile checksum differs from the recorded checksum')
    profile = verify(read_json(local_path(root, path)))
    if profile['kind'] == RUNTIME_KIND:
        if sha256(local_path(root, profile['product_bundle'])) != profile['sources'][profile['product_bundle']]:
            raise ValueError('Bound runtime profile bundle bytes changed')
        return BoundProfile(path, RUNTIME_KIND, checksum, profile_sha256, checksum, profile['product_bundle'],
                            profile['product_bundle_checksum'], profile['selector']['model_checksum'])
    if profile['kind'] == COMPOSITE_KIND:
        handoff = verify(read_json(local_path(root, profile['identity']['handoff'])))
        return BoundProfile(path, COMPOSITE_KIND, checksum, profile_sha256, profile['baseline_profile_checksum'],
                            profile['identity']['bundle'], handoff['bundle']['manifest_checksum'],
                            profile['selector_model_checksum'])
    if profile['kind'] == INTEGRATED_KIND:
        parent = verify(read_json(local_path(root, profile['parent_profile'])))
        handoff = verify(read_json(local_path(root, parent['identity']['handoff'])))
        return BoundProfile(path, INTEGRATED_KIND, checksum, profile_sha256, profile['baseline_profile_checksum'],
                            parent['identity']['bundle'], handoff['bundle']['manifest_checksum'],
                            profile['selector_model_checksum'])
    if profile['kind'] == SYSTEMIC_KIND:
        parent = verify(read_json(local_path(root, profile['parent_profile'])))
        handoff = verify(read_json(local_path(root, parent['identity']['handoff'])))
        return BoundProfile(path, SYSTEMIC_KIND, checksum, profile_sha256, parent['baseline_profile_checksum'],
                            parent['identity']['bundle'], handoff['bundle']['manifest_checksum'],
                            profile['model_checksum'])
    if profile['kind'] == TARGET_KIND:
        systemic = verify(read_json(local_path(root, profile['parent_profile'])))
        parent = verify(read_json(local_path(root, systemic['parent_profile'])))
        handoff = verify(read_json(local_path(root, parent['identity']['handoff'])))
        return BoundProfile(path, TARGET_KIND, checksum, profile_sha256, parent['baseline_profile_checksum'],
                            parent['identity']['bundle'], handoff['bundle']['manifest_checksum'],
                            profile['parent_model_checksum'])
    if profile['kind'] == GEOMETRY_KIND:
        target = verify(read_json(local_path(root, profile['parent_profile'])))
        systemic = verify(read_json(local_path(root, target['parent_profile'])))
        parent = verify(read_json(local_path(root, systemic['parent_profile'])))
        handoff = verify(read_json(local_path(root, parent['identity']['handoff'])))
        return BoundProfile(path, GEOMETRY_KIND, checksum, profile_sha256, parent['baseline_profile_checksum'],
                            parent['identity']['bundle'], handoff['bundle']['manifest_checksum'],
                            profile['parent_model_checksum'])
    if profile['kind'] in (B3_ONLY_KIND, B3_ONLY_D6_KIND, REPAIR_KIND, B3_ONLY_D7_KIND, REPAIR_V2_KIND, B3_ONLY_D8_KIND,
                           ZERO_TARGET_KIND, B3_ONLY_D9_KIND, TEXT_EVIDENCE_KIND, B3_ONLY_D10_KIND, EVIDENCE_CONSISTENCY_KIND,
                           B3_ONLY_D11_KIND, ATLAS_REPAIR_KIND, B3_ONLY_D12_KIND, RELEASE_NEXT_KIND):
        repair = profile['kind'] in (REPAIR_KIND, REPAIR_V2_KIND, ZERO_TARGET_KIND, TEXT_EVIDENCE_KIND, EVIDENCE_CONSISTENCY_KIND,
                                     ATLAS_REPAIR_KIND, RELEASE_NEXT_KIND)
        base = verify(read_json(local_path(root, profile['parent_profile']))) if repair else profile
        geometry = verify(read_json(local_path(root, base['parent_profile'])))
        target = verify(read_json(local_path(root, geometry['parent_profile'])))
        systemic = verify(read_json(local_path(root, target['parent_profile'])))
        parent = verify(read_json(local_path(root, systemic['parent_profile'])))
        handoff = verify(read_json(local_path(root, parent['identity']['handoff'])))
        return BoundProfile(path, profile['kind'], checksum, profile_sha256, parent['baseline_profile_checksum'],
                            parent['identity']['bundle'], handoff['bundle']['manifest_checksum'],
                            profile['ablated_ranker_checksum'])
    raise ValueError('Unsupported bound profile kind')


def _migrations(root):
    """(entry, dispatch paths of its sealed manifest) for every present migration manifest."""
    out = []
    for manifest, paths in MIGRATIONS.items():
        path = local_path(root, manifest)
        if not path.is_file():
            continue
        doc = verify(read_json(path))
        if doc.get('kind') != MIGRATION_KIND or tuple(doc.get('dispatch_paths', ())) != paths:
            raise ValueError('Unsupported dispatch migration manifest')
        out.extend((entry, paths) for entry in doc['entries'])
    return out


def _check_pins(root, profile_path, pins, what):
    """Every pin equals live bytes, or exactly one migration entry for this profile/path/expected hash covers it."""
    profile_sha256 = sha256(local_path(root, profile_path))
    checksum = read_json(local_path(root, profile_path)).get('checksum')
    for source, expected in pins.items():
        live = sha256(local_path(root, source))
        if live == expected:
            continue
        entries = [e for e, paths in _migrations(root)
                   if source in paths and e['profile'] == profile_path and e['profile_sha256'] == profile_sha256
                   and e['profile_checksum'] == checksum and e['logical_path'] == source
                   and e['expected_sha256'] == expected and e['replacement_sha256'] == live]
        if len(entries) != 1 or sha256(local_path(root, entries[0]['archive'])) != expected:
            raise ValueError(what + ' source changed: ' + source)


def _composite_checksum(root, path):
    profile = verify(read_json(local_path(root, path)))
    if profile.get('kind') != COMPOSITE_KIND:
        raise ValueError('Parent is not a composite profile')
    if sha256(local_path(root, profile['baseline_profile'])) != profile['baseline_profile_sha256']:
        raise ValueError('Baseline profile bytes changed')
    _check_pins(root, path, profile['pins_sha256'], 'Composite profile')
    return profile['checksum']


def _verified_checksum(root, path):
    profile = verify(read_json(local_path(root, path)))
    if profile.get('kind') == INTEGRATED_KIND:
        from rshb_vine.integrated_candidate_v1.runtime import load_profile
        checksum = load_profile(root, path)[0]['checksum']
        if _composite_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Integrated parent checksum differs')
        return checksum
    if profile.get('kind') == COMPOSITE_KIND:
        return _composite_checksum(root, path)
    if profile.get('kind') == SYSTEMIC_KIND:
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent composite profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Systemic profile')
        if _composite_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Systemic parent checksum differs')
        return profile['checksum']
    if profile.get('kind') == TARGET_KIND:
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent systemic profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Target-contract profile')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Target-contract parent checksum differs')
        return profile['checksum']
    if profile.get('kind') == GEOMETRY_KIND:
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent target-contract profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Geometry release profile')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Geometry release parent checksum differs')
        return profile['checksum']
    if profile.get('kind') in (B3_ONLY_KIND, B3_ONLY_D6_KIND, B3_ONLY_D7_KIND, B3_ONLY_D8_KIND, B3_ONLY_D9_KIND, B3_ONLY_D10_KIND,
                               B3_ONLY_D11_KIND, B3_ONLY_D12_KIND):
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent geometry profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'B3-only release profile')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('B3-only release parent checksum differs')
        return profile['checksum']
    if profile.get('kind') == REPAIR_KIND:
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent dispatch6 B3-only profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Repair release profile')
        if verify(read_json(local_path(root, profile['combined_profile'])))['checksum'] != profile['combined_profile_checksum']:
            raise ValueError('Combined repair profile checksum differs')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Repair release parent checksum differs')
        return profile['checksum']
    if profile.get('kind') == REPAIR_V2_KIND:
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent dispatch7 B3-only profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Repair v2 release profile')
        for name in ('candidate_profile', 'repair_v1_profile'):
            if verify(read_json(local_path(root, profile[name])))['checksum'] != profile[name + '_checksum']:
                raise ValueError('Repair v2 ' + name + ' checksum differs')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Repair v2 release parent checksum differs')
        return profile['checksum']
    if profile.get('kind') == ZERO_TARGET_KIND:
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent dispatch8 B3-only profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Zero-target release profile')
        for name in ('candidate_profile', 'repair_v1_profile'):
            if verify(read_json(local_path(root, profile[name])))['checksum'] != profile[name + '_checksum']:
                raise ValueError('Zero-target release ' + name + ' checksum differs')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Zero-target release parent checksum differs')
        return profile['checksum']
    if profile.get('kind') in (TEXT_EVIDENCE_KIND, EVIDENCE_CONSISTENCY_KIND, ATLAS_REPAIR_KIND, RELEASE_NEXT_KIND):
        if sha256(local_path(root, profile['parent_profile'])) != profile['parent_profile_sha256']:
            raise ValueError('Parent B3-only profile bytes changed')
        _check_pins(root, path, profile['pins_sha256'], 'Text-evidence release profile')
        for name in ('candidate_profile', 'repair_v1_profile'):
            if verify(read_json(local_path(root, profile[name])))['checksum'] != profile[name + '_checksum']:
                raise ValueError('Text-evidence release ' + name + ' checksum differs')
        predecessor = profile['predecessor_release']
        if _verified_checksum(root, predecessor['path']) != predecessor['checksum']:
            raise ValueError('Text-evidence release predecessor checksum differs')
        if _verified_checksum(root, profile['parent_profile']) != profile['parent_profile_checksum']:
            raise ValueError('Text-evidence release parent checksum differs')
        return profile['checksum']
    return describe(root, path)['profile_checksum']


def public_answer(result):
    """Chosen = best_candidate (the /v1/eval/predict slug); a primary-target suggestion is chosen, never accepted."""
    chosen = result.get('best_candidate')
    suggested = (result.get('target_selection') or {}).get('suggested_default') is True
    return {'chosen': chosen, 'accepted': chosen if chosen is not None and not suggested else None,
            'suggested_default': suggested, 'calibrated': False}


def bind_capture(root, capture):
    """The profile a capture protocol recorded (profile_checksum + profile_sha256)."""
    return bind_profile(root, capture['profile_checksum'], capture['profile_sha256'])


def bind_pointer(root, pointer='config/recognition-current.json'):
    """The profile the pointer names now; refuses a pointer whose bytes are not a versioned profile."""
    root = Path(root).resolve()
    return bind_profile(root, describe(root, pointer)['profile_checksum'], sha256(local_path(root, pointer)))


def result_matches(result, bound):
    runtime = result.get('recognition_runtime', {})
    return (result_profile_checksum(result) == bound.checksum and runtime.get('mode') == 'candidate_service'
            and runtime.get('profile_checksum') == bound.runtime_checksum)


def checked_receipt(capture_dir, capture, source_row, bound):
    """Same immutable receipt/attempt checks as the frozen scorer; profile check against the bound profile."""
    ident = source_row['id']
    key = digest(ident) + '.json'
    receipt_path = capture_dir / 'records' / key
    attempt_path = capture_dir / 'attempts' / key
    receipt = verify(read_json(receipt_path))
    attempt = verify(read_json(attempt_path))
    if (receipt.get('kind') != 'current-recognition-intake-receipt-v1'
            or receipt.get('protocol') != capture['checksum']
            or receipt.get('id') != ident
            or receipt.get('image_sha256') != source_row['sha256']
            or receipt.get('attempt_sha256') != sha256(attempt_path)
            or attempt.get('protocol') != capture['checksum']
            or attempt.get('id') != ident
            or attempt.get('image_sha256') != source_row['sha256']):
        raise ValueError('Capture receipt/attempt mismatch: ' + ident)
    if bound.checksum != capture['profile_checksum']:
        raise ValueError('Bound profile is not the capture profile')
    result = receipt.get('result')
    if receipt['status'] == 200 and receipt.get('valid_contract'):
        if not isinstance(result, dict) or not result_matches(result, bound):
            raise ValueError('Response profile mismatch: ' + ident)
    return receipt, sha256(receipt_path)


def describe_binding(bound):
    return json.loads(json.dumps(bound.__dict__))
