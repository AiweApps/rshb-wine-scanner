"""Combined candidate over a loaded B3-only base (83e97cb3 itself, or its dispatch6 equivalent passed as base_release).

Installed in this order, frozen sources untouched:
  ConflictFilteredIndex  replaces the raw B3 LabelFirstIndex under the request cache, so every consumer (object
                         base route, core arm, selection raw_visual) and every cache sees filtered results only
  GuardedOcrCandidate    ocr_candidate_repair_v1.guarded.install: catalogue-phrase injection + shared-name
                         visual-agreement guard v2; the conflict hook is informative, text evidence is never blocked
  front-label repair     profile 577a575e (two-detector label boundary v2; base.index wrapped after the filter)
Public slug/probability stay None; B3 remains the only raw-visual arm. History: the first combined profile
fad80ee5 (unguarded OCR, no front-label) and its 8-row plan stay under combined/candidate and combined/http.
"""
from pathlib import Path
import time
import types

from rshb_vine.io import local_path, read_json, seal, sha256, verify

KIND = 'recognition-repair-v1-combined-profile'
BASE_PROFILE = 'config/recognition-b3-only-v1-release.json'
BASE_CHECKSUM = '83e97cb3f6e8d1a89451b1ffcdd8de4e20f6417706b488f1e4d426c538cafbbb'
PROFILE = 'runs/recognition-repair-20260926/combined/candidate-v3/profile.json'
FAILED_PROFILE = {'path': 'runs/recognition-repair-20260926/combined/candidate-v2/profile.json',
                  'checksum': '0afdf67fb8cc1751e16f89e2f328e6417a8b73e2af491604adb33d5f84aff5cd',
                  'http': 'runs/recognition-repair-20260926/combined/http-v2',
                  'failure': 'HTTP 503 AttributeError: RecognitionRuntime._attach_products reads selection.model; '
                             'the OCR adapter lacked the diagnostic legacy-model passthrough'}
HISTORY_PROFILE = 'runs/recognition-repair-20260926/combined/candidate/profile.json'
BASE_D6_KIND = 'recognition-b3-only-v1-dispatch6-profile'
BASE_IDENTITY = ('policy', 'arm_inputs', 'ablated_features', 'parent_ranker_checksum', 'ablated_ranker_checksum',
                 'b3_encoder_id', 'b0_gate_encoder_id', 'device')
GUARD_RULE = 'shared-name-visual-agreement-guard-v2'
FRONT_LABEL_PROFILE = 'runs/recognition-repair-20260926/crop/candidate/profile.json'
FRONT_LABEL_CHECKSUM = '577a575eb51577bfdb4e3ac817facb9527d26a0c81a7b87bba8c1c50b02ce20c'
PORT = 8189
FORBIDDEN_PORTS = (8175, 8188)
SOURCES = ('rshb_vine/recognition_repair_v1/__init__.py', 'rshb_vine/recognition_repair_v1/runtime.py',
           'scripts/recognition_repair_v1.py',
           'rshb_vine/reference_conflicts_v1/__init__.py', 'rshb_vine/reference_conflicts_v1/guard.py',
           'rshb_vine/reference_conflicts_v1/runtime.py', 'config/reference-conflicts-v1.json',
           'rshb_vine/ocr_candidate_repair_v1/__init__.py', 'rshb_vine/ocr_candidate_repair_v1/injection.py',
           'rshb_vine/ocr_candidate_repair_v1/selection.py', 'rshb_vine/ocr_candidate_repair_v1/discriminator.py',
           'rshb_vine/ocr_candidate_repair_v1/guarded.py',
           'runs/catalog-training-data-v2/stage5/evaluator/actual-fit-v1/B4343/gallery/gallery.json',
           'data/catalog-additions-20260921/catalog.jsonl')
FRONT_LABEL_SOURCES = ('rshb_vine/front_label_repair_v1/__init__.py', 'rshb_vine/front_label_repair_v1/geometry.py',
                       'rshb_vine/front_label_repair_v1/runtime.py', 'runs/label-detector-real-v2/manifest.json',
                       'scripts/front_label_repair_v1.py', FRONT_LABEL_PROFILE)
SEMI_CARD = 'dva-serdtsa-muskat-muskat-belyy-beloe-polusuhoe-12'
SUGAR_PROVENANCE = {
    'card': f'provisional-card:{SEMI_CARD}',
    'frozen_runtime_claim': {'value': 'semidry', 'source': 'data/product-identity-v1/snapshot-01/claims.json',
                             'origin': 'external catalog candidate linked to a RESET locked_test photo; retained unchanged'},
    'admissible_alternative_origin': {'value': 'semidry', 'source': 'data/site-catalog-intake-20260921/bodies/'
                                      '71cb89c8109d7f250ea8f87eed6b1e5f8500d70a27bdedecc4cabf140fb2e36f',
                                      'field': 'category.name', 'raw': 'Белое полусухое',
                                      'body_sha256': 'bc7b306b04e0995dd1fd13ee81517424f16e18a33d18a287ef32d0f01eeaabcc'},
    'values_agree': True,
    'protected_metadata_read': 'data/external/kultovo_labels.jsonl line 662 was read during the 2026-09-26 audit; '
                               'its value is not used by this candidate',
}


def freeze_body(root, front_label=True):
    from rshb_vine.ocr_candidate_repair_v1 import discriminator
    root = Path(root).resolve()
    if discriminator.RULE['version'] != GUARD_RULE:
        raise ValueError('Discriminator guard is not the reviewed v2 rule')
    if front_label:
        from rshb_vine.front_label_repair_v1 import runtime as crop
        if crop.load_profile(root, FRONT_LABEL_PROFILE)['checksum'] != FRONT_LABEL_CHECKSUM:
            raise ValueError('Front-label profile is not the admitted 577a575e')
    base = verify(read_json(root / BASE_PROFILE))
    if base['checksum'] != BASE_CHECKSUM:
        raise ValueError('Base B3-only release profile changed')
    paths = SOURCES + (FRONT_LABEL_SOURCES if front_label else ())
    pins = {p: sha256(root / p) for p in paths}
    pins[BASE_PROFILE] = sha256(root / BASE_PROFILE)
    conflicts = verify(read_json(root / 'config/reference-conflicts-v1.json'))
    return {'kind': KIND, 'name': 'recognition-repair-v1-combined', 'base_profile': BASE_PROFILE,
            'base_checksum': BASE_CHECKSUM, 'pins_sha256': pins, 'reference_conflicts_checksum': conflicts['checksum'],
            'components': {'reference_conflict_filter': 'reference_conflicts_v1.runtime.ConflictFilteredIndex on the raw B3 index',
                           'ocr_selection': 'ocr_candidate_repair_v1.guarded.install (injection v2 + discriminator guard)',
                           'discriminator_guard': GUARD_RULE,
                           'front_label_repair': bool(front_label),
                           'front_label_profile_checksum': FRONT_LABEL_CHECKSUM if front_label else None},
            'sugar_provenance': SUGAR_PROVENANCE, 'port': PORT, 'release_status': 'experimental_candidate_not_admitted',
            'activated': False, 'calibrated': False, 'probability': None, 'fit_run': False,
            'training_guard': 'separate admission consumer (reference_conflicts_v1 admit); not applied by this runtime',
            'history': {'profile': HISTORY_PROFILE, 'checksum': 'fad80ee5b63f45514dcdcef64e4b9b89d2b7da384377bd8e36b321a1c81917d2',
                        'plan': 'runs/recognition-repair-20260926/combined/http/plan.json',
                        'note': 'first combined candidate (unguarded OCR, no front-label); kept unchanged, not loadable by this module'},
            'failed_candidate': FAILED_PROFILE,
            'rollback': 'stop 8189; 8175, factory and config/recognition-current.json are never modified by this candidate'}


def load_profile(root, profile_path=PROFILE):
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, profile_path)))
    if profile.get('kind') != KIND or profile['base_profile'] != BASE_PROFILE or profile['base_checksum'] != BASE_CHECKSUM:
        raise ValueError('Unsupported recognition-repair profile')
    expected = set(SOURCES + (FRONT_LABEL_SOURCES if profile['components']['front_label_repair'] else ()) + (BASE_PROFILE,))
    if set(profile['pins_sha256']) != expected:
        raise ValueError('Recognition-repair pin set differs from the module protocol')
    for path, value in profile['pins_sha256'].items():
        if sha256(local_path(root, path)) != value:
            raise ValueError('Recognition-repair pinned file changed: ' + path)
    if profile['components'].get('discriminator_guard') != GUARD_RULE:
        raise ValueError('Recognition-repair profile does not name the guard v2 rule')
    if profile['components']['front_label_repair'] and profile['components']['front_label_profile_checksum'] != FRONT_LABEL_CHECKSUM:
        raise ValueError('Recognition-repair profile does not bind front-label 577a575e')
    return profile


def check_base(root, base):
    """A supplied base must be the dispatch6 equivalent of 83e97cb3 with identical encoders, arm, ranker and ablation."""
    original = verify(read_json(local_path(root, BASE_PROFILE)))
    profile = base.profile
    equivalent = profile.get('equivalent_release') or {}
    if (profile.get('kind') != BASE_D6_KIND or equivalent.get('path') != BASE_PROFILE
            or equivalent.get('checksum') != BASE_CHECKSUM or equivalent.get('sha256') != sha256(local_path(root, BASE_PROFILE))
            or original['checksum'] != BASE_CHECKSUM):
        raise ValueError('Supplied base is not bound to the frozen B3-only release 83e97cb3')
    differs = [k for k in BASE_IDENTITY if profile.get(k) != original.get(k)]
    if differs:
        raise ValueError('Supplied base differs from 83e97cb3 in: ' + ', '.join(differs))
    own = base.ownership
    if (own['core_B3_arm']['encoder_id'] != original['b3_encoder_id'] or own['object_gate']['encoder_id'] != original['b0_gate_encoder_id']
            or own['ranker']['ablated'] != original['ablated_ranker_checksum'] or own['selection_raw_visual_arms'] != ['B3']):
        raise ValueError('Loaded base ownership differs from 83e97cb3')
    return {'kind': profile['kind'], 'checksum': profile['checksum'], 'equivalent_release': BASE_CHECKSUM}


def _slots(root_obj, target, limit=500000):
    """Every (container, key, path) in the owned object graph that holds ``target`` directly."""
    from rshb_vine.b3_only_v1.runtime import _owned
    seen, found, stack = set(), [], [(root_obj, None, None, 'release')]
    while stack and len(seen) < limit:
        obj, parent, key, path = stack.pop()
        if parent is not None and (obj is target or isinstance(obj, types.MethodType) and obj.__self__ is target):
            found.append((parent, key if obj is target else ('bound_method', key), path))
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            children = [(v, obj, k, path + '[' + repr(k)[:40] + ']') for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            children = [(v, obj, i, path + '[%d]' % i) for i, v in enumerate(obj)]
        elif _owned(obj):
            children = [(v, obj, ('attr', k), path + '.' + k) for k, v in vars(obj).items()]
        else:
            children = []
        stack.extend(c for c in children if c[0] is target or isinstance(c[0], types.MethodType) or _owned(c[0]) or isinstance(c[0], (dict, list, tuple))
                     and (len(c[0]) <= 256 or any(_owned(v) for v in (c[0].values() if isinstance(c[0], dict) else c[0]))))
    if stack:
        raise ValueError('Graph scan limit reached; index ownership proof incomplete')
    return found, len(seen)


def install_reference_filter(release, guard):
    from rshb_vine.reference_conflicts_v1.runtime import filter_index
    from rshb_vine.visual_core import LabelFirstIndex
    core = release.runtime.holders['B3']
    raw = getattr(core.index, '_index', None)
    if type(raw) is not LabelFirstIndex or raw.encoder_id != release.ownership['core_B3_arm']['encoder_id']:
        raise ValueError('Innermost shared B3 index is not the release LabelFirstIndex')
    slots, visited = _slots(release, raw)
    if not slots or any(type(p).__name__ != '_IndexDelegate' or k != ('attr', '_index') for p, k, _ in slots):
        raise ValueError('Raw B3 index must be held only by request-cache delegates: ' + repr([s[2] for s in slots]))
    filtered = filter_index(raw, guard)
    for parent, key, path in slots:
        if isinstance(parent, tuple):
            raise ValueError('Raw B3 index held in an immutable tuple: ' + path)
        if isinstance(key, tuple):
            setattr(parent, key[1], filtered)
        else:
            parent[key] = filtered
    after, _ = _slots(release, raw)
    if [p for p, _, _ in after] != [filtered]:
        raise ValueError('Raw B3 index still reachable outside the filter: ' + repr([s[2] for s in after]))
    if sorted(release.runtime.arms) != ['B3']:
        raise ValueError('Raw-visual arms changed; B3 must stay the only arm')
    return filtered, {'replaced_slots': [s[2] for s in slots], 'objects_visited': visited,
                      'raw_holders_after': [s[2] for s in after], 'excluded_references': filtered.excluded}


class RecognitionRepairV1:
    def __init__(self, root, profile_path=PROFILE, *, base_release=None):
        from rshb_vine.ocr_candidate_repair_v1 import guarded as ocr_selection
        from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
        root = Path(root).resolve()
        self.profile = load_profile(root, profile_path)
        self.guard = ReferenceConflicts(root)
        if self.guard.checksum != self.profile['reference_conflicts_checksum']:
            raise ValueError('Reference-conflict config differs from the profile')
        if base_release is None:
            from rshb_vine.b3_only_v1.release import B3OnlyRelease
            self.release = B3OnlyRelease(root, BASE_PROFILE)
            if self.release.profile['checksum'] != BASE_CHECKSUM:
                raise ValueError('Loaded B3-only release differs from the pinned base')
            self.base = {'kind': self.release.profile['kind'], 'checksum': BASE_CHECKSUM, 'equivalent_release': BASE_CHECKSUM}
        else:
            self.base = check_base(root, base_release)
            self.release = base_release
        self.filtered, filter_install = install_reference_filter(self.release, self.guard)
        self.selection = ocr_selection.install(self.release, self.guard.injection_conflict)
        if type(self.selection) is not ocr_selection.GuardedOcrCandidateSelection:
            raise ValueError('OCR selection must be the guarded v2 adapter')
        self.front_label = None
        if self.profile['components']['front_label_repair']:
            from rshb_vine.front_label_repair_v1 import runtime as front_label
            self.front_label = front_label
            if front_label.load_profile(root, FRONT_LABEL_PROFILE)['checksum'] != FRONT_LABEL_CHECKSUM:
                raise ValueError('Front-label profile differs from the admitted 577a575e')
            front_install = front_label.install(self.release)
        else:
            front_install = None
        self.installation = {'reference_filter': filter_install, 'ocr_selection': type(self.selection).__module__,
                             'front_label_repair': front_install}
        self.manifest = seal({'kind': 'recognition-repair-v1-runtime', 'profile_checksum': self.profile['checksum'],
                              'runtime_descriptor_checksum': self.profile['checksum'],
                              'parent_runtime': self.release.manifest['checksum'], 'base_profile_checksum': self.base['checksum'],
                              'equivalent_base_release': BASE_CHECKSUM,
                              'reference_conflicts_checksum': self.guard.checksum,
                              'installation': self.installation, 'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        started = time.perf_counter()
        traces = self.front_label.TRACES if self.front_label else None
        token = traces.set([]) if traces else None
        try:
            result = self.release.recognize(data, roi, bottles)
            split_calls = list(traces.get()) if traces else None
        finally:
            if traces:
                traces.reset(token)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        arms = result['b3_only_v1']['selection_raw_visual_arms']
        if any(a != ['B3'] for a in arms):
            raise RuntimeError('Raw-visual arms must be exactly one B3 arm per target')
        records = (result.get('systemic_ranking_v2') or {}).get('targets', [])
        if any(r.get('adapter') != 'ocr-candidate-repair-v1-selection' for r in records):
            raise RuntimeError('Published selector records are not from the OCR repair adapter')
        for block in ('roskachestvo_geometry_v2', 'target_contract'):
            if result.get(block) is not None:
                result[block]['b3_base_profile_checksum'] = result[block]['profile_checksum']
                result[block]['profile_checksum'] = self.profile['checksum']
        result['recognition_repair_v1'] = {
            'profile_checksum': self.profile['checksum'], 'runtime_checksum': self.manifest['checksum'],
            'base_profile_checksum': self.base['checksum'], 'base_kind': self.base['kind'],
            'equivalent_base_release': BASE_CHECKSUM, 'components': self.profile['components'],
            'reference_filter': {'config_checksum': self.guard.checksum,
                                 'excluded_references': self.filtered.excluded,
                                 'annotations': self.guard.annotate_response(result)},
            'ocr_candidate_injection': [dict(r['ocr_candidate_injection'], instance_id=r['instance_id']) for r in records],
            'front_label_repair': None if split_calls is None else {
                'profile_checksum': FRONT_LABEL_CHECKSUM, 'split_calls': split_calls,
                'views_added': sum(bool(t.get('added')) for t in split_calls),
                'targets': [{'instance_id': str(t['instance_id']),
                             'recovered_views': [v['bbox'] for v in t['retrieval'].get('views', [])
                                                 if v.get('source') == self.front_label.VIEW_SOURCE]}
                            for t in result.get('targets', [])]},
            'discriminator_guard': [dict(r['discriminator_guard'] or {}, instance_id=r['instance_id']) for r in records],
            'sugar_provenance': SUGAR_PROVENANCE, 'calibrated': False, 'probability': None,
            'release_admitted': False, 'seconds': time.perf_counter() - started}
        return result


def describe(root, profile_path=PROFILE):
    from rshb_vine.b3_only_v1.release import describe_release
    root = Path(root).resolve()
    profile = load_profile(root, profile_path)
    return {'kind': KIND, 'profile_checksum': profile['checksum'], 'components': profile['components'],
            'base': describe_release(root, BASE_PROFILE), 'port': profile['port'],
            'sugar_provenance': profile['sugar_provenance'], 'training_guard': profile['training_guard'],
            'public_answer': 'uncertain proposal; slug and probability stay None', 'models_loaded': False}
