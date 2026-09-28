"""N existing-target parent over a loaded bde4fa52-equivalent graph; per-instance adapters only.

ExistingTargetParent  wraps the installed orphan-stage ``apply`` (G TrustedRegionRecovery over the frozen
                      GeometryOrphanRecovery + front-label wrap). Only in G's observe pass, on the request's own bytes,
                      outside every nested retry context and only when that pass already has a target, each
                      plan.trials row gets one E1 call: the frozen orphan stripe detector on the frozen close-up canvas.
                      N1: a unique containing proposal becomes the physical parent (context = that parent).
                      N1b: only when E1 found no containing proposal at all, the label itself is the gate context and
                      no bottle is claimed (G label_only convention beside another bottle).
                      The region is served once by G's TrustedRegionLocalizer to the installed orphan retry on the
                      ORIGINAL bytes, so the B0 object gate, B3, front-label boundary, OCR and T/I run unchanged.
                      Existing targets and instances stay byte-identical; the retry target is appended exactly as the
                      frozen orphan stage appends a label beside existing targets. The injected localization row stays
                      in N's trace only: native ROI cannot re-address a region no detector pass produced.
                      N1b first scores the same checked label crop with the frozen B0 gate of the graph (same encoder,
                      formula and threshold); only a passing label pays for the retry, whose own gate still decides.
OriginalScopeGeometry per-instance wrapper of the layout GeometryStage: when N appended targets to a request whose
                      frozen pass had exactly one target, that target is evaluated as the single target it was;
                      every other call is the frozen stage unchanged.
run / trace           one request under its own context-local scope over any complete ``recognize`` callable.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
from pathlib import Path
import time

from rshb_vine.atlas_geometry_repair_v1 import runtime as G
from rshb_vine.bottle_instances import iou
from rshb_vine.existing_target_parent_v1 import plan as P
import math

from rshb_vine.io import digest, local_path, read_json, seal, sha256, verify
from rshb_vine.orphan_label_recovery_v2 import publish_recovered_text
from rshb_vine.preprocessing import checked_box, decode
from rshb_vine.roskachestvo_geometry_v2 import association as A

PACKAGE = 'rshb_vine/existing_target_parent_v1'
SOURCES = (f'{PACKAGE}/__init__.py', f'{PACKAGE}/plan.py', f'{PACKAGE}/runtime.py', f'{PACKAGE}/component.py',
           'scripts/existing_target_parent_v1.py')
DESCRIPTOR_KIND = 'existing-target-parent-v1-component-descriptor'
PARENT_PROFILE = 'config/atlas-repair-release-v1-profile.json'
PARENT_MANIFEST = 'config/atlas-repair-release-v1-manifest.json'
PARENT_CHECKSUM = 'bde4fa52d35b1c297e43996cacf11db2e00558a0273586639b3e01923ab468f3'
PARENT_RUNTIME = '54f09a202c69d8c2b0afd0688e39736a0d40f90b75826dde184ccb8b778bd7cd'
OUT = 'runs/release-next-v1/geometry'
DESCRIPTOR = OUT + '/component/descriptor-v3.json'
# (path, kind, checksum, decision or None): the sealed documents that admit and bound this component.
BINDING_DOCUMENTS = (
    ('runs/release-next-v1/scope.json', 'release-next-v1-root-scope',
     '8b4bf180432abed50d3823511174bc14e79ec198c7ec003605cf351c50e0cdd9', None),
    (OUT + '/root-admission.json', 'release-next-v1-geometry-root-admission',
     '18bd8f4ab3f214cf2ccfe64c5ac6aaecf03f696fd4245c9cffed1c9a443cc731', 'admit_N1_implementation_and_bounded_preprobe'),
    (OUT + '/root-n1b-admission.json', 'release-next-v1-N1b-root-admission',
     'f0a6fde30c7fa72fc3635b7ff5cb5d8c95df17240b04267ff0cbceabcb9f68cf',
     'admit_N1b_unlocalized_label_recovery_after_failed_E1'),
    (OUT + '/causal-plan.json', 'release-next-v1-geometry-causal-plan',
     'bd01be92b6506e6363dcd633c6bdcdfa072d64409429dbf29d21e74f9089e44f', None),
    ('runs/atlas-repair-v1/geometry/census-b0d8688c.json', 'atlas-geometry-repair-v1-census',
     '57c886c8c9b80615e09775588d7c599a47cb5358603615402eb744f8131875c4', None),
    (OUT + '/preprobe/plan.json', 'existing-target-parent-v1-preprobe-plan',
     '2c9b5fa3ea416c0d3bdcb7794866df123bf1dc6eafdd7585cc0f1832547af4ba', None),
    (OUT + '/preprobe/results.json', 'existing-target-parent-v1-preprobe-results',
     'c3a44e2223e53de9db5856ef705d489a3d5ad30142cec22e18a292799f54443e', None),
    (OUT + '/preprobe/summary.json', 'existing-target-parent-v1-preprobe-summary',
     '49be6fef1891fa28dc9c2ec2f3cd6a4753ea8d4e8bdf0a33e29b902f4284a4ef', None),
    (OUT + '/root-http-protocol.json', 'release-next-v1-N-root-http-protocol',
     '2f25ec1fded11501b222c9b860a52330514c3d9dc53ef643b321eae5a0688306', None),
    (OUT + '/root-correction-v2-admission.json', 'release-next-v1-N-correction-root-admission',
     '6713316863dfacbd37e233f18372548c6fa7721a2f356d0ed401ef06ce434008', 'admit_bounded_semantics_and_cost_correction'),
    (OUT + '/root-roi-contract-correction.json', 'release-next-v1-N-roi-contract-correction',
     'e0ccd095ad4b4c43f6668e68e9eeedf868dbfbb5070b33ced937f029ace12780', 'admit_remove_injected_native_addressability'),
)
PREDECESSOR = {'path': OUT + '/component/descriptor-v2.json',
               'checksum': '062659250497d68e302c111712a0e718eb9a269cd6f440e08dcd244a9e25f54c',
               'sources_snapshot': OUT + '/component/v2-archive',
               'superseded_because': ['injected localization row made recovered targets look ROI-addressable',
                                      'appended targets moved the original single target out of the layout stage scope',
                                      'N1b paid a full retry for labels the frozen B0 gate rejects'],
               'predecessor': {'path': OUT + '/component/descriptor.json',
                               'checksum': '2df34ce34c8cc6c069aa11e2d7bbeb31e92e06ad501148e74cdb9e5e701505dd',
                               'sources_snapshot': OUT + '/component/v1-archive'}}
SOURCE_TAG = {P.PHYSICAL: 'existing_target_parent_v1_canvas_stripe_parent',
              P.LABEL_ONLY: 'existing_target_parent_v1_unlocalized_label'}
EVIDENCE = {P.PHYSICAL: 'unique frozen stripe-detector proposal on the frozen close-up canvas containing the label',
            P.LABEL_ONLY: 'no bottle claimed: E1 returned no containing proposal; object gate on the label region only'}

_REQUEST = ContextVar('existing_target_parent_v1_request', default=None)


@contextmanager
def request_scope(data, roi, capture=False):
    token = _REQUEST.set({'enabled': roi is None, 'sha256': hashlib.sha256(data).hexdigest(), 'handled': False,
                          'events': [], 'skipped_invocations': [], 'e1_calls': 0, 'retries': 0, 'added': [],
                          'original_target_ids': None, 'n1b_gate_calls': 0, 'layout_scope': [],
                          'captured': [] if capture else None})
    try:
        yield _REQUEST.get()
    finally:
        _REQUEST.reset(token)


def _skip(data, result, roi, scope):
    from rshb_vine.front_label_repair_v1 import runtime as FL
    from rshb_vine.target_misses_v1 import route as R
    if roi is not None:
        return 'roi'
    if result.get('decision') == 'invalid_image':
        return 'invalid_image'
    for name, var in (('g_inject', G._INJECT), ('canvas_request', R._REQUEST), ('orphan_image', FL.ORPHAN_IMAGE),
                      ('orphan_frame', FL.ORPHAN_FRAME), ('split_image', FL.IMAGE)):
        if var.get() is not None:
            return name + '_active'
    g = G._REQUEST.get()
    if g is None or g['phase'] != G.OBSERVE:
        return 'not_g_observe_pass'
    if hashlib.sha256(data).hexdigest() != scope['sha256']:
        return 'not_request_bytes'
    if not result.get('targets'):
        return 'no_existing_target'
    if scope['handled']:
        return 'request_already_handled'
    return None


def _region(kind, label, parent, sid, score, size):
    label = [float(v) for v in checked_box(label, size)]
    physical = kind == P.PHYSICAL
    context = checked_box(parent['bottle_bbox'], size) if physical else checked_box(label, size)
    region = {'bbox': label, 'detector_score': float(score), 'kind': 'detected_label', 'parent_id': sid,
              'raw_label_regions': [{'bbox': list(label), 'detector_score': float(score)}], 'source': SOURCE_TAG[kind],
              'context_bbox': context}
    if not physical:
        region['unlocalized'] = True
    trace = [{'parent_id': sid, 'bottle_bbox': list(context) if physical else None, 'rotation_attempted': False,
              'selected_rotation': 0, 'label_count': 1, 'source': SOURCE_TAG[kind]}]
    return region, trace


class ExistingTargetParent:
    def __init__(self, orphan, inner_apply, retry, bottles_parent, gate):
        self.orphan, self.inner, self.retry, self.bottles_parent, self.gate = orphan, inner_apply, retry, bottles_parent, gate

    def _n1b_gate(self, image, label):
        """The frozen object gate of the graph on the checked label crop, exactly as the retry would score it."""
        import numpy as np
        from rshb_vine.b3_only_v1 import split as S
        gate = self.gate
        box = checked_box(checked_box(label, image.size), image.size)
        started = time.perf_counter()
        with S.GATE:
            vector = gate.gate_encoder.encode([image.crop(box)])[0]
        gate.gate_counters.add('b0_gate_encode_calls', 1)
        gate.gate_counters.add('b0_gate_encode_images', 1)
        score = float(1 / (1 + np.exp(-np.clip(np.dot(vector, gate.coef) + gate.bias, -40, 40))))
        threshold = gate.spec['threshold']
        return {'context': 'label_region', 'crop_bbox': list(box), 'score': score, 'threshold': threshold,
                'passes': score >= threshold, 'gate_encoder_id': gate.gate_encoder_id,
                'gate_spec_checksum': gate.spec['checksum'], 'counters': ['b0_gate_encode_calls', 'b0_gate_encode_images'],
                'ms': 1000 * (time.perf_counter() - started)}

    def _e1(self, image, label):
        from rshb_vine.bottle_rescue import LowBottleDetector
        from rshb_vine.target_misses_v1.diagnose import centered_canvas
        column = P.stripe(label, image.size)
        canvas, dx, dy = centered_canvas(image.crop(column), P.CANVAS_SCALE)
        detector = LowBottleDetector(self.bottles_parent)
        started = time.perf_counter()
        found = detector.detect(canvas)
        proposals = []
        for p in found:
            box = P.map_proposal(p['bbox'], column, (dx, dy), image.height)
            proposals.append({'canvas_bbox': list(p['bbox']), 'score': p['score'], 'bbox': box,
                              'inside_column': box[2] > box[0] and box[3] > box[1],
                              'label_containment': P.contained(label, box)})
        return {'column': column, 'canvas_size': [canvas.width, canvas.height], 'canvas_offset': [dx, dy],
                'raw_nms_proposals': detector.proposal_count, 'proposals': proposals,
                'ms': 1000 * (time.perf_counter() - started)}

    def _retry(self, scope, fingerprint, region, trace, data):
        scope['retries'] += 1
        inject = {'fingerprint': fingerprint, 'regions': [region], 'localization_trace': trace, 'served': False,
                  'calls': [], 'trace': []}
        token = G._INJECT.set(inject)
        started = time.perf_counter()
        try:
            retry = self.retry(data)
        finally:
            G._INJECT.reset(token)
        found = retry.get('targets') or []
        record = {'ms': 1000 * (time.perf_counter() - started), 'localizer_calls': inject['calls'], 'served': inject['served'],
                  'decision': retry.get('decision'), 'reasons': retry.get('reasons'), 'targets': len(found),
                  'instances': [{k: i.get(k) for k in ('bottle_id', 'bottle_bbox', 'label_bbox', 'status', 'wine_score')}
                                for i in retry.get('instances') or []],
                  'object_filter': deepcopy(retry.get('object_filter')),
                  'rejected_instances': [{k: i.get(k) for k in ('bottle_id', 'wine_score')}
                                         for i in retry.get('rejected_instances') or []]}
        if scope['captured'] is not None:
            scope['captured'].append({'instance_id': region['parent_id'], 'retry': deepcopy(retry)})
        return retry, record

    @staticmethod
    def _accepted(kind, retry, record, region, sid):
        found = retry.get('targets') or []
        if not record['served']:
            return None, 'region_not_served'
        if len(found) != 1:
            return None, 'retry_targets_%d' % len(found)
        target = deepcopy(found[0])
        if str(target.get('instance_id')) != sid:
            return None, 'retry_target_not_injected_region'
        record['label_iou_with_region'] = overlap = iou(target['bbox'], region['bbox'])
        if overlap < P.PROVENANCE_IOU:
            return None, 'retry_target_not_region'
        if not target.get('retrieval', {}).get('views'):
            return None, 'no_visual_retrieval'
        physical = kind == P.PHYSICAL
        if bool(target.get('physical_bottle_localized')) != physical or bool(target.get('bottle_bbox')) != physical:
            return None, 'physical_contract_mismatch'
        return target, 'recovered'

    def apply(self, data, baseline, roi=None):
        result = self.inner(data, baseline, roi)
        scope = _REQUEST.get()
        if scope is None or not scope['enabled']:
            return result
        reason = _skip(data, result, roi, scope)
        if reason is not None:
            scope['skipped_invocations'].append(reason)
            return result
        scope['handled'] = True
        started = time.perf_counter()
        rows = P.trials(result)
        event = {'policy': P.POLICY, 'trials': rows, 'attempts': [], 'added': [],
                 'existing_target_ids': [str(t.get('instance_id')) for t in result['targets']]}
        scope['events'].append(event)
        eligible = [r for r in rows if r['eligible']]
        if not eligible:
            event['ms'] = 1000 * (time.perf_counter() - started)
            return result
        image, _ = decode(data)
        fingerprint = G._fingerprint(image)
        old_targets = deepcopy(result['targets'])
        old_instances = deepcopy(result.get('instances') or [])
        preserved = [digest(t) for t in old_targets], digest(old_instances)
        before = {'targets': deepcopy(old_targets), 'variant_text': deepcopy(result.get('variant_text') or {})}
        for row in eligible:
            label = row['label_bbox']
            attempt = {'trial': row['trial'], 'label_bbox': label}
            event['attempts'].append(attempt)
            scope['e1_calls'] += 1
            attempt['e1'] = e1 = self._e1(image, label)
            parent, why = P.parent(label, e1['proposals'], result['targets'])
            attempt['parent_decision'] = why
            if parent is not None:
                kind = P.PHYSICAL
            elif why == 'no_containing_proposal':
                kind = P.LABEL_ONLY
            else:
                attempt['outcome'] = why
                continue
            attempt['kind'] = kind
            score = row['label_score'] if row['label_score'] is not None else A.detection_score(
                getattr(self.orphan.labels, 'last', None) or [], label)
            if score is None:
                attempt['outcome'] = 'label_score_unavailable'
                continue
            if kind == P.LABEL_ONLY:
                scope['n1b_gate_calls'] += 1
                attempt['n1b_gate'] = self._n1b_gate(image, label)
                if not attempt['n1b_gate']['passes']:
                    attempt['outcome'] = 'n1b_gate_rejected_before_retry'
                    continue
            sid = 'existing-parent-%d' % row['trial']
            region, trace = _region(kind, label, parent, sid, score, image.size)
            attempt.update(instance_id=sid, label_score=score, parent=deepcopy(parent), injected_region=deepcopy(region))
            retry, attempt['retry'] = self._retry(scope, fingerprint, region, trace, data)
            if kind == P.LABEL_ONLY:
                gated = next((i.get('wine_score') for i in (retry.get('instances') or []) + (retry.get('rejected_instances') or [])
                              if str(i.get('bottle_id')) == sid and isinstance(i.get('wine_score'), (int, float))), None)
                attempt['n1b_gate']['retry_wine_score'] = gated
                attempt['n1b_gate']['retry_parity'] = gated is not None and math.isclose(
                    gated, attempt['n1b_gate']['score'], rel_tol=1e-4, abs_tol=1e-6)
            target, outcome = self._accepted(kind, retry, attempt['retry'], region, sid)
            if target is None:
                attempt['outcome'] = outcome
                continue
            association = {'policy': P.POLICY, 'kind': kind, 'orphan_trial': row['trial'], 'orphan_label_bbox': list(label),
                           'orphan_label_detector_score': score, 'frozen_reason': row['frozen_reason'],
                           'frozen_stripe_proposals': row['stripe_proposals'], 'parent': deepcopy(parent),
                           'parent_evidence': EVIDENCE[kind], 'e1_decision': why,
                           'gate_context': 'detector_parent' if kind == P.PHYSICAL else 'label_region',
                           'retry_label_iou_with_region': attempt['retry']['label_iou_with_region'],
                           'text_identity_used': False}
            target.update(geometry_source=P.POLICY, geometry_association=association)
            result['targets'].append(target)
            instance = next((deepcopy(i) for i in retry.get('instances') or [] if str(i.get('bottle_id')) == sid),
                            {'bottle_id': sid, 'bottle_bbox': target.get('bottle_bbox'), 'label_bbox': target['bbox'],
                             'status': 'label_available', 'wine_score': target.get('wine_score')})
            instance['geometry_association'] = {'policy': P.POLICY, 'kind': kind, 'target_instance_id': sid,
                                                'parent_source': parent and parent['source']}
            result.setdefault('instances', []).append(instance)
            publish_recovered_text(result, before, retry, sid, [0, 0, image.width, image.height])
            packet = next(p for p in result['instance_text']['targets'] if str(p.get('instance_id')) == sid)
            packet['existing_target_parent_region'] = {'label_bbox': region['bbox'], 'context_bbox': region['context_bbox'],
                                                       'kind': kind, 'retry_input': 'original request frame, offset 0'}
            attempt['injected_localization_trace'] = [deepcopy(r) for r in retry.get('localization_trace') or []
                                                      if G._owned(r, sid)]
            public = {k: v for k, v in retry.items() if k != 'localization_trace'}
            attempt['copied_blocks'] = ['instance_text.targets'] + G._copy_instance_entries(public, result, sid)
            attempt.update(outcome='recovered', answer=target['retrieval'].get('best_candidate'))
            event['added'].append(sid)
            scope['added'].append(sid)
        if event['added']:
            scope['original_target_ids'] = [str(t.get('instance_id')) for t in old_targets]
            G._finalize(result, True, None)
            kept = [digest(t) for t in result['targets'][:len(old_targets)]]
            if kept != preserved[0] or digest(result['instances'][:len(old_instances)]) != preserved[1]:
                raise RuntimeError('N changed an existing target or instance')
        event['ms'] = 1000 * (time.perf_counter() - started)
        result.setdefault('timing_ms', {})['existing_target_parent'] = event['ms']
        return result


class _SingleTargetRequest:
    """The in-flight request as the frozen pass published it: only the original target in ``result['targets']``."""

    def __init__(self, request, target):
        self._request = request
        self.data = request.data
        self.result = dict(request.result, targets=[target])

    def target(self, instance_id):
        return self._request.target(instance_id)

    def image(self):
        return self._request.image()


class OriginalScopeGeometry:
    def __init__(self, stage):
        self.stage, self.frozen = stage, stage.apply

    def apply(self, selection, out, call, request):
        scope = _REQUEST.get()
        if scope is None or request is None or G._INJECT.get() is not None or not scope['added']:
            return self.frozen(selection, out, call, request)
        original = scope['original_target_ids'] or []
        sid = str(call['raw_visual']['instance_id'])
        targets = request.result.get('targets') or []
        ids = [str(t.get('instance_id')) for t in targets]
        if len(original) != 1 or sid != original[0] or ids != original + scope['added']:
            return self.frozen(selection, out, call, request)
        out = self.frozen(selection, out, call, _SingleTargetRequest(request, targets[0]))
        record = out.get(self.stage.trace_key) or {}
        scope['layout_scope'].append({'instance_id': sid, 'evaluated_as': 'original single-target request',
                                      'request_targets': ids, 'action': record.get('action'),
                                      'applied': record.get('applied')})
        return out


def run(stage, recognize, data, roi=None, bottles='addressed', capture=False):
    """(result, scope) of one request; ``recognize(data, roi, bottles)`` is one complete parent recognition."""
    with request_scope(data, roi, capture) as scope:
        result = recognize(data, roi, bottles)
    return result, scope


def trace(scope):
    if scope is None:
        return None
    return {'policy': P.POLICY, 'enabled': scope['enabled'], 'handled': scope['handled'], 'e1_calls': scope['e1_calls'],
            'n1b_gate_calls': scope['n1b_gate_calls'], 'retries': scope['retries'], 'events': deepcopy(scope['events']),
            'original_target_ids': scope['original_target_ids'], 'layout_scope_preserved': deepcopy(scope['layout_scope']),
            'skipped_invocations': list(scope['skipped_invocations']), 'recovered': list(scope['added'])}


def locate(graph):
    from rshb_vine.bottle_rescue import LowBottleDetector
    orphan = graph.base.runtime.control.orphans
    if type(orphan) is not A.GeometryOrphanRecovery or type(orphan.labels) is not A.CapturingLabels:
        raise ValueError('Loaded orphan stage is not the frozen GeometryOrphanRecovery with capturing labels')
    recovery = graph.recovery
    if not isinstance(recovery, G.TrustedRegionRecovery) or vars(orphan).get('apply') != recovery.apply:
        raise ValueError('Installed orphan apply is not the G TrustedRegionRecovery of this graph (or N is installed)')
    if vars(orphan).get('retry') is not recovery.retry:
        raise ValueError('Installed orphan retry is not the front-label retry G uses')
    bottles = orphan.bottles
    if type(bottles) is not LowBottleDetector or bottles.parent.processor.size != {'height': 640, 'width': 640}:
        raise ValueError('Orphan-stage bottle detector is not the frozen 640x640 LowBottleDetector')
    gate = graph.base.runtime.control.expanded.core.parent.parent.parent
    if any(not hasattr(gate, name) for name in ('gate_encoder', 'gate_encoder_id', 'gate_counters', 'coef', 'bias', 'spec')):
        raise ValueError('Object gate of the loaded graph is not the split B0 wine-object stage')
    if gate.gate_encoder_id != graph.base.profile['b0_gate_encoder_id']:
        raise ValueError('Object gate encoder differs from the release profile')
    return orphan, recovery, bottles.parent, gate, geometry_stage(graph)


def geometry_stage(graph):
    """The single layout GeometryStage instance shared by every live selector owner of the graph."""
    from rshb_vine.recognition_repair_v2.geometry import GeometryStage
    found = [s for s in getattr(graph.inner.selection, 'stages', ()) if isinstance(s, GeometryStage)]
    if len(found) != 1 or type(found[0]) is not GeometryStage:
        raise ValueError('Live selector does not hold exactly one frozen layout GeometryStage')
    if 'apply' in vars(found[0]):
        raise ValueError('Layout GeometryStage.apply is already wrapped on this instance')
    return found[0]


def install(graph):
    orphan, recovery, detector, gate, layout = locate(graph)
    stage = ExistingTargetParent(orphan, orphan.apply, orphan.retry, detector, gate)
    orphan.apply = stage.apply
    scope_keeper = OriginalScopeGeometry(layout)
    layout.apply = scope_keeper.apply
    return stage, {'policy': P.POLICY, 'constants': P.CONSTANTS, 'orphan_path': 'base.runtime.control.orphans.apply',
                   'wraps': recovery.apply.__module__ + '.' + recovery.apply.__qualname__,
                   'retry': orphan.retry.__module__ + '.' + orphan.retry.__qualname__,
                   'region_server': 'rshb_vine.atlas_geometry_repair_v1.runtime.TrustedRegionLocalizer (installed by G)',
                   'e1_detector': type(detector).__module__ + '.' + type(detector).__name__ + ' of the orphan stage',
                   'n1b_gate': {'path': 'base.runtime.control.expanded.core.parent.parent.parent',
                                'gate_encoder_id': gate.gate_encoder_id, 'spec_checksum': gate.spec['checksum'],
                                'threshold': gate.spec['threshold']},
                   'layout_scope': {'path': 'inner.selection.stages[GeometryStage].apply', 'trace_key': layout.trace_key}}


def _parent_ref(root):
    root = Path(root).resolve()
    profile = verify(read_json(root / PARENT_PROFILE))
    manifest = verify(read_json(root / PARENT_MANIFEST))
    if profile['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Parent profile is not the immutable atlas repair release bde4fa52')
    return {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM, 'sha256': sha256(root / PARENT_PROFILE),
            'manifest': {'path': PARENT_MANIFEST, 'checksum': manifest['checksum'], 'sha256': sha256(root / PARENT_MANIFEST)},
            'runtime_checksum': PARENT_RUNTIME}


def sources_sha(root):
    return {p: sha256(Path(root) / p) for p in SOURCES}


def binding_documents(root):
    """(pins_sha256, refs) of the binding documents; each must still carry its sealed kind, checksum and decision."""
    root = Path(root).resolve()
    pins, refs = {}, []
    for path, kind, checksum, decision in BINDING_DOCUMENTS:
        doc = verify(read_json(local_path(root, path)))
        if doc.get('kind') != kind or doc['checksum'] != checksum or (decision is not None and doc.get('decision') != decision):
            raise ValueError('Binding document differs from the admitted one: ' + path)
        pins[path] = sha256(local_path(root, path))
        refs.append({'path': path, 'kind': kind, 'checksum': checksum, 'decision': decision})
    return pins, refs


def freeze(root):
    root = Path(root).resolve()
    pins, refs = binding_documents(root)
    return seal({'kind': DESCRIPTOR_KIND, 'version': 2, 'policy': P.POLICY, 'constants': P.CONSTANTS,
                 'parent_release': _parent_ref(root), 'sources_sha256': sources_sha(root), 'pins_sha256': pins,
                 'binding_documents': refs, 'predecessor_descriptor': PREDECESSOR, 'activated': False,
                 'release_admitted': False, 'calibrated': False, 'probability': None, 'public_slug': None,
                 'weights_changed': False, 'fit_run': False,
                 'not_implemented': ['304 fragment merge (no physical evidence, no control population)']})


def load_descriptor(root, path=DESCRIPTOR):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if descriptor.get('kind') != DESCRIPTOR_KIND:
        raise ValueError('Unsupported existing-target parent descriptor')
    if _parent_ref(root) != descriptor['parent_release']:
        raise ValueError('Parent release bytes differ from the descriptor')
    if sources_sha(root) != descriptor['sources_sha256']:
        raise ValueError('Component sources changed since freeze')
    if descriptor['constants'] != P.CONSTANTS or descriptor['policy'] != P.POLICY:
        raise ValueError('Component constants differ from the descriptor')
    pins, refs = binding_documents(root)
    if descriptor.get('pins_sha256') != pins or descriptor.get('binding_documents') != refs:
        raise ValueError('Binding document pins differ from the descriptor')
    return descriptor
