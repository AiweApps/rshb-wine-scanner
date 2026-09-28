"""Atlas geometry repair G over the loaded immutable F10 release cd0b9910; per-instance adapters only.

TrustedRegionLocalizer  delegate installed at LabelRescue.parent (above the frozen CanvasCloseupLocalizer); inert
                        unless a G retry is active. During that retry its first detect() on the request's own pixels
                        returns the already-found region and every other call returns nothing, so neither the bottle
                        nor the label detector re-discovers what the frozen stages found.
TrustedRegionRecovery   wraps the installed orphan-stage ``apply`` (frozen GeometryOrphanRecovery + front-label wrap).
                        The frozen result is kept. In the observe phase it only records ``plan.candidates``; in the armed
                        phase, for each eligible trial it runs the installed orphan ``retry`` once on the ORIGINAL request
                        bytes with the region injected, so wine gate, B3, front-label boundary, OCR and conditional reads
                        run unchanged in original coordinates.
ObserveExpandedMemo     per-instance wrapper of control.expanded.recognize. The first call of the observe pass (plain original
                        query, no G/canvas/orphan context, zero targets) is kept for this request only; the armed pass's
                        first call with the same bytes and context gets a deep copy once, with its front-label split traces
                        replayed. Every other call, the injected G retry included, is computed.
recognize_with_recovery one request over any complete recognize callable: a frozen observe pass, then one armed pass only
                        when that whole pass (later frozen fallbacks included) returned no target and G observed an
                        eligible trial; the armed result is published only if all its targets are G's, else the observe
                        result is returned untouched. Shared by this candidate and the combined candidate/release.
AtlasGeometryCandidate  the F10 release loaded from its immutable profile, G installed, own descriptor identity;
                        the parent release and its lineage only under lineage keys. Slug/probability stay None.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
from pathlib import Path
import time

from rshb_vine.atlas_geometry_repair_v1 import plan as P
from rshb_vine.bottle_instances import iou
from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.orphan_label_recovery_v2 import publish_recovered_text
from rshb_vine.preprocessing import checked_box, decode
from rshb_vine.roskachestvo_geometry_v2 import association as A

PACKAGE = 'rshb_vine/atlas_geometry_repair_v1'
SOURCES = (f'{PACKAGE}/__init__.py', f'{PACKAGE}/plan.py', f'{PACKAGE}/runtime.py', 'scripts/atlas_geometry_repair_v1.py')
DESCRIPTOR_KIND = 'atlas-geometry-repair-v1-candidate-descriptor'
BLOCK = 'atlas_geometry_repair_v1'
PARENT_PROFILE = 'config/evidence-consistency-release-v1-profile.json'
PARENT_MANIFEST = 'config/evidence-consistency-release-v1-manifest.json'
PARENT_CHECKSUM = 'cd0b9910133f4c17df53d7e171ae3c0432a5b8a1252163d42413ef6808d81ece'
PARENT_RUNTIME = 'be6e2a50d9703d1d2d0607caf214dffb259c4f8890cc0ebb61c06083c067e985'
OUT = 'runs/atlas-repair-v1/geometry'
DESCRIPTOR = OUT + '/candidate-v4/descriptor.json'
PORT = 8221
SOURCE_TAG = 'atlas_geometry_repair_v1_trusted_region'
PER_INSTANCE_EXCLUDED = frozenset({'targets', 'instances', 'ranked_candidates', 'timing_ms', 'bottle_rescue',
                                   'orphan_label_recovery', 'object_filter', 'raw_detections', 'instance_text',
                                   'variant_text'})

OBSERVE, ARMED = 'observe', 'armed'
PHASES = (OBSERVE, ARMED)

_REQUEST = ContextVar('atlas_geometry_repair_v1_request', default=None)
_INJECT = ContextVar('atlas_geometry_repair_v1_inject', default=None)


def _fingerprint(image):
    return list(image.size), hashlib.sha256(image.tobytes()).hexdigest()


@contextmanager
def request_scope(roi, bottles, capture, phase):
    if phase not in PHASES:
        raise ValueError('G request phase must be one of %r' % (PHASES,))
    token = _REQUEST.set({'enabled': roi is None, 'phase': phase, 'bottles': bottles, 'retries': 0, 'events': [],
                          'passes': [], 'second_pass': None, 'selected_pass': 0, 'captured': [] if capture else None,
                          'expanded_calls': 0, 'memo': {'entry': None, 'stored': None, 'served': None, 'calls': []}})
    try:
        yield _REQUEST.get()
    finally:
        _REQUEST.reset(token)


class TrustedRegionLocalizer:
    def __init__(self, parent):
        object.__setattr__(self, 'parent', parent)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def __setattr__(self, name, value):
        # BottleRescueProfile swaps `bottles` on the localizer it reaches through this delegate.
        setattr(self.parent, name, value)

    @property
    def last_trace(self):
        inject = _INJECT.get()
        return inject['trace'] if inject is not None else self.parent.last_trace

    def detect(self, image):
        inject = _INJECT.get()
        if inject is None:
            return self.parent.detect(image)
        same = _fingerprint(image) == inject['fingerprint']
        served = same and not inject['served']
        inject['calls'].append({'size': list(image.size), 'request_pixels': same, 'regions_served': len(inject['regions']) if served else 0})
        if not served:
            inject['trace'] = []
            self._no_rescue_detection(inject['calls'][-1])
            return []
        inject['served'] = True
        inject['trace'] = deepcopy(inject['localization_trace'])
        return deepcopy(inject['regions'])

    def _no_rescue_detection(self, call):
        from rshb_vine.bottle_rescue import LowBottleDetector
        bottles = getattr(self.parent, 'bottles', None)
        # Frozen BottleRescueProfile reads proposal_count of its per-request detector; it was never called here.
        if isinstance(bottles, LowBottleDetector) and not hasattr(bottles, 'proposal_count'):
            bottles.proposal_count = 0
            call['low_bottle_detector_not_called'] = True


def _memo_context(data, roi, scope, index):
    """(bypass reason or None, key) of one expanded call; only an eligible call hashes the request bytes."""
    from rshb_vine.front_label_repair_v1 import runtime as FL
    from rshb_vine.target_misses_v1 import route as R
    if index:
        return 'not_first_call_of_pass', None
    if roi is not None:
        return 'roi', None
    for name, var in (('g_inject', _INJECT), ('canvas_request', R._REQUEST), ('orphan_image', FL.ORPHAN_IMAGE),
                      ('orphan_frame', FL.ORPHAN_FRAME), ('split_image', FL.IMAGE)):
        if var.get() is not None:
            return name + '_active', None
    return None, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'roi': None,
                  'bottles': scope['bottles'], 'front_label_traces': FL.TRACES.get() is not None}


class ObserveExpandedMemo:
    def __init__(self, frozen):
        self.frozen = frozen

    def recognize(self, data, roi=None):
        scope = _REQUEST.get()
        if scope is None or not scope['enabled']:
            return self.frozen(data, roi)
        from rshb_vine.front_label_repair_v1 import runtime as FL
        index, memo, phase = scope['expanded_calls'], scope['memo'], scope['phase']
        scope['expanded_calls'] += 1
        started = time.perf_counter()
        reason, key = _memo_context(data, roi, scope, index)
        call = {'full_pass': len(scope['passes']), 'phase': phase, 'call_index': index, 'bypass': reason}
        memo['calls'].append(call)
        entry = memo['entry']
        if phase == ARMED and reason is None:
            if entry is None:
                call['outcome'] = 'miss_nothing_stored'
            elif entry['key'] != key:
                call['outcome'] = 'miss_key_differs'
            else:
                memo['entry'] = None
                traces = FL.TRACES.get()
                if traces is not None:
                    traces.extend(deepcopy(entry['traces']))
                body = deepcopy(entry['body'])
                call.update(outcome='hit', key_sha256=key['sha256'], replayed_front_label_traces=len(entry['traces']),
                            saved_ms=entry['ms'], hit_ms=1000 * (time.perf_counter() - started))
                memo['served'] = dict(call)
                return body
        traces = FL.TRACES.get()
        before = len(traces) if traces is not None else None
        computed = time.perf_counter()
        result = self.frozen(data, roi)
        ms = 1000 * (time.perf_counter() - computed)
        call.update(outcome=call.get('outcome', 'computed'), ms=ms)
        if phase != OBSERVE or reason is not None or memo['stored'] is not None:
            return result
        targets = len(result.get('targets') or [])
        if result.get('decision') == 'invalid_image' or targets:
            memo['stored'] = {'stored': False, 'reason': 'invalid_image' if targets == 0 else 'has_targets', 'targets': targets}
            return result
        copy_started = time.perf_counter()
        memo['entry'] = {'key': key, 'body': deepcopy(result), 'ms': ms,
                         'traces': deepcopy(traces[before:]) if traces is not None else []}
        memo['stored'] = {'stored': True, 'key_sha256': key['sha256'], 'bytes': key['bytes'], 'bottles': key['bottles'],
                          'front_label_traces_mode': key['front_label_traces'], 'targets': 0, 'ms': ms,
                          'front_label_traces': len(memo['entry']['traces']),
                          'copy_ms': 1000 * (time.perf_counter() - copy_started)}
        call['outcome'] = 'stored'
        return result


def _region(row, sid, score, size, result):
    label = [float(v) for v in checked_box(row['label_bbox'], size)]
    common = {'bbox': label, 'detector_score': float(score), 'kind': 'detected_label', 'parent_id': sid,
              'raw_label_regions': [{'bbox': list(label), 'detector_score': float(score)}], 'source': SOURCE_TAG}
    if row['kind'] == P.PHYSICAL:
        context, context_kind = checked_box(row['parent']['bottle_bbox'], size), 'detector_parent'
        region = {**common, 'context_bbox': context}
    else:
        context, context_kind = P.label_only_context(result, label, size)
        context = checked_box(context, size)
        region = {**common, 'context_bbox': context, 'unlocalized': True}
    trace = [{'parent_id': sid, 'bottle_bbox': None if row['kind'] == P.LABEL_ONLY else list(context),
              'rotation_attempted': False, 'selected_rotation': 0, 'label_count': 1, 'source': SOURCE_TAG}]
    return region, trace, context_kind


def _owned(row, sid):
    return isinstance(row, dict) and sid in (str(row.get('instance_id')), str(row.get('parent_id')), str(row.get('bottle_id')))


def _copy_instance_entries(source, target, sid):
    """Every per-instance record of the retry keyed by the injected id, appended to the same block of the result."""
    copied = []
    for name, block in source.items():
        if name in PER_INSTANCE_EXCLUDED:
            continue
        if isinstance(block, list):
            rows = [r for r in block if _owned(r, sid)]
            if rows:
                target.setdefault(name, []).extend(deepcopy(rows))
                copied.append(name)
        elif isinstance(block, dict):
            for field, value in block.items():
                if not isinstance(value, list):
                    continue
                rows = [r for r in value if _owned(r, sid)]
                if rows:
                    holder = target.setdefault(name, {})
                    if not isinstance(holder, dict):
                        continue
                    holder.setdefault(field, []).extend(deepcopy(rows))
                    copied.append(name + '.' + field)
    return copied


def _finalize(result, had_targets, retry_reasons):
    packets = (result.get('instance_text') or {}).get('targets') or []
    reads = (result.get('conditional_label_ocr') or {}).get('reads')
    if reads is not None:
        result['conditional_label_ocr']['performed'] = bool(reads)
    targets = result['targets']
    if len(targets) == 1:
        sid = str(targets[0]['instance_id'])
        packet = next((p for p in packets if str(p.get('instance_id')) == sid), None)
        result['variant_text'] = deepcopy(packet) if packet is not None else {
            'performed': False, 'reason': 'no_instance_text_packet', 'evidence_location': 'instance_text.targets'}
        t = targets[0]
        result.update(best_candidate=t['retrieval']['best_candidate'], ranked_candidates=t['retrieval']['ranked_candidates'],
                      slug=None, decision='uncertain', requires_target_selection=False)
    else:
        result['variant_text'] = {'performed': False, 'reason': 'not_single_target', 'evidence_location': 'instance_text.targets'}
        result.update(best_candidate=None, slug=None, ranked_candidates=[], decision='ambiguous_target',
                      requires_target_selection=True)
    if not had_targets:
        result['reasons'] = list(retry_reasons)


class TrustedRegionRecovery:
    def __init__(self, orphan, frozen_apply, retry):
        self.orphan, self.frozen_apply, self.retry = orphan, frozen_apply, retry

    def apply(self, data, baseline, roi=None):
        result = self.frozen_apply(data, baseline, roi)
        scope = _REQUEST.get()
        if scope is None or not scope['enabled'] or roi is not None or result.get('decision') == 'invalid_image':
            return result
        started = time.perf_counter()
        rows = P.candidates(result)
        full_pass = len(scope['passes'])
        invocation = sum(1 for e in scope['events'] if e['full_pass'] == full_pass)
        event = {'policy': P.POLICY, 'phase': scope['phase'], 'full_pass': full_pass, 'pass': invocation, 'trials': rows,
                 'attempts': [], 'added': []}
        scope['events'].append(event)
        eligible = [r for r in rows if r['eligible']]
        if not eligible or scope['phase'] != ARMED:
            return result
        image, _ = decode(data)
        fingerprint = _fingerprint(image)
        detections = getattr(self.orphan.labels, 'last', None) or []
        had_targets = bool(result.get('targets'))
        before = {'targets': deepcopy(result.get('targets') or []), 'variant_text': deepcopy(result.get('variant_text') or {})}
        retry_reasons = None
        for row in eligible:
            attempt = {'trial': row['trial'], 'kind': row['kind'], 'label_bbox': row['label_bbox']}
            event['attempts'].append(attempt)
            if scope['retries'] >= P.MAX_RETRIES_PER_REQUEST:
                attempt['outcome'] = 'request_retry_budget'
                continue
            if had_targets or result.get('targets'):
                attempt['outcome'] = 'request_has_target'
                continue
            score = row['label_score'] if row['label_score'] is not None else A.detection_score(detections, row['label_bbox'])
            if score is None:
                attempt['outcome'] = 'label_score_unavailable'
                continue
            conflict, other = P._conflict(row['kind'], row['label_bbox'], row['parent'] and row['parent']['bottle_bbox'],
                                          result.get('targets') or [])
            if conflict:
                attempt.update(outcome=conflict, conflicting_instance_id=other)
                continue
            sid = 'atlas-geometry-%d-%d' % (event['pass'], row['trial'])
            region, trace, context_kind = _region(row, sid, score, image.size, result)
            attempt.update(instance_id=sid, label_score=score, injected_region=deepcopy(region), gate_context=context_kind,
                           parent=deepcopy(row['parent']))
            scope['retries'] += 1
            inject = {'fingerprint': fingerprint, 'regions': [region], 'localization_trace': trace, 'served': False,
                      'calls': [], 'trace': []}
            token = _INJECT.set(inject)
            retry_started = time.perf_counter()
            try:
                retry = self.retry(data)
            finally:
                _INJECT.reset(token)
            found = retry.get('targets') or []
            attempt['retry'] = {
                'ms': 1000 * (time.perf_counter() - retry_started), 'localizer_calls': inject['calls'],
                'decision': retry.get('decision'), 'reasons': retry.get('reasons'), 'targets': len(found),
                'instances': [{k: i.get(k) for k in ('bottle_id', 'bottle_bbox', 'label_bbox', 'status', 'wine_score')}
                              for i in retry.get('instances') or []],
                'object_filter': deepcopy(retry.get('object_filter')),
                'rejected_instances': [{k: i.get(k) for k in ('bottle_id', 'wine_score')} for i in retry.get('rejected_instances') or []],
                'oversized_label_rescue': deepcopy((retry.get('oversized_label_rescue') or {}).get('attempted')),
                'square_rescue_attempted': (retry.get('square_rescue') or {}).get('attempted'),
                'bottle_rescue_attempted': (retry.get('bottle_rescue') or {}).get('attempted')}
            if scope['captured'] is not None:
                scope['captured'].append({'full_pass': full_pass, 'instance_id': sid, 'retry': deepcopy(retry)})
            if not inject['served']:
                attempt['outcome'] = 'region_not_served'
                continue
            if len(found) != 1:
                attempt['outcome'] = 'retry_targets_%d' % len(found)
                continue
            target = deepcopy(found[0])
            if str(target.get('instance_id')) != sid:
                attempt['outcome'] = 'retry_target_not_injected_region'
                continue
            overlap = iou(target['bbox'], region['bbox'])
            attempt['retry']['label_iou_with_region'] = overlap
            if overlap < P.PROVENANCE_IOU:
                attempt['outcome'] = 'retry_target_not_region'
                continue
            if not target.get('retrieval', {}).get('views'):
                attempt['outcome'] = 'no_visual_retrieval'
                continue
            physical = row['kind'] == P.PHYSICAL
            if bool(target.get('physical_bottle_localized')) != physical or (not physical and target.get('bottle_bbox') is not None):
                attempt['outcome'] = 'physical_contract_mismatch'
                continue
            target.update(geometry_source=P.POLICY, geometry_association={
                'policy': P.POLICY, 'kind': row['kind'], 'orphan_trial': row['trial'], 'orphan_label_bbox': row['label_bbox'],
                'orphan_label_detector_score': score, 'frozen_reason': row['frozen_reason'],
                'parent': deepcopy(row['parent']), 'gate_context': context_kind,
                'retry_label_iou_with_region': overlap})
            result.setdefault('targets', []).append(target)
            association = {'policy': P.POLICY, 'kind': row['kind'], 'parent_source': row['parent'] and row['parent']['source'],
                           'parent_bottle_id': row['parent'] and row['parent']['bottle_id'], 'target_instance_id': sid}
            existing = next((i for i in result.setdefault('instances', []) if physical and i.get('bottle_bbox')
                             and i.get('label_bbox') is None and iou(i['bottle_bbox'], target['bottle_bbox']) > P.SAME_PARENT_IOU), None)
            if existing is not None:
                association.update(original_status=existing.get('status'), original_wine_score=existing.get('wine_score'))
                existing.update(label_bbox=target['bbox'], label_id=sid + '-label', status='label_available',
                                wine_score=target.get('wine_score'), geometry_association=association)
                target['parent_id'] = existing['bottle_id']
            else:
                instance = next((deepcopy(i) for i in retry.get('instances') or [] if str(i.get('bottle_id')) == sid),
                                {'bottle_id': sid, 'bottle_bbox': target.get('bottle_bbox'), 'label_bbox': target['bbox'],
                                 'status': 'label_available', 'wine_score': target.get('wine_score')})
                instance['geometry_association'] = association
                result['instances'].append(instance)
            if not (retry.get('variant_text') or {}).get('performed'):
                attempt['text'] = 'retry_variant_text_not_performed'
            # Exactly one retry target with the injected id: its variant_text is that target's packet (original pixels).
            publish_recovered_text(result, before, retry, sid, [0, 0, image.width, image.height])
            packet = next(p for p in result['instance_text']['targets'] if str(p.get('instance_id')) == sid)
            packet['atlas_geometry_repair_region'] = {'label_bbox': region['bbox'], 'context_bbox': region['context_bbox'],
                                                      'retry_input': 'original request frame, offset 0'}
            attempt['copied_blocks'] = ['instance_text.targets'] + _copy_instance_entries(retry, result, sid)
            attempt.update(outcome='recovered', answer=target['retrieval'].get('best_candidate'))
            event['added'].append(sid)
            retry_reasons = retry.get('reasons') or []
        if event['added']:
            _finalize(result, had_targets, retry_reasons)
        event['ms'] = 1000 * (time.perf_counter() - started)
        result.setdefault('timing_ms', {})['atlas_geometry_repair'] = event['ms']
        return result


def _pass_summary(scope, phase, result, first_event, ms):
    events = scope['events'][first_event:]
    targets = result.get('targets') or []
    return {'phase': phase, 'decision': result.get('decision'), 'reasons': deepcopy(result.get('reasons')),
            'targets': len(targets), 'target_ids': [str(t.get('instance_id')) for t in targets],
            'instances': len(result.get('instances') or []), 'rejected_instances': len(result.get('rejected_instances') or []),
            'orphan_invocations': len(events), 'eligible_observed': sum(1 for e in events for r in e['trials'] if r['eligible']),
            'added': [sid for e in events for sid in e['added']],
            'canvas_closeup_route': {k: (result.get('canvas_closeup_route') or {}).get(k)
                                     for k in ('attempted', 'reason', 'recovered')},
            'expanded_calls': scope['expanded_calls'], 'ms': ms}


def _full_pass(scope, recognize, data, roi, bottles, phase):
    scope['phase'], scope['expanded_calls'] = phase, 0
    first_event, started = len(scope['events']), time.perf_counter()
    try:
        result = recognize(data, roi, bottles)
    except Exception as error:
        scope['passes'].append({'phase': phase, 'error': type(error).__name__ + ': ' + str(error)[:300],
                                'ms': 1000 * (time.perf_counter() - started)})
        raise
    summary = _pass_summary(scope, phase, result, first_event, 1000 * (time.perf_counter() - started))
    scope['passes'].append(summary)
    if scope['captured'] is not None:
        scope['captured'].append({'full_pass': len(scope['passes']) - 1, 'phase': phase, 'full_result': deepcopy(result)})
    return result, summary


def _armed_skip(first, summary, roi):
    if roi is not None:
        return 'roi_request'
    if first.get('decision') == 'invalid_image':
        return 'invalid_image'
    if summary['targets']:
        return 'first_pass_has_targets'
    if not summary['eligible_observed']:
        return 'no_eligible_observed'
    return None


def _discard_armed(scope):
    for event in scope['events']:
        if event['phase'] == ARMED and event['added']:
            event['discarded'], event['added'] = event['added'], []


def recognize_with_recovery(recognize, data, roi=None, bottles='addressed', capture=False):
    """(result, scope) of one request; ``recognize(data, roi, bottles)`` is one complete frozen recognition."""
    with request_scope(roi, bottles, capture, OBSERVE) as scope:
        first, summary = _full_pass(scope, recognize, data, roi, bottles, OBSERVE)
        skip = _armed_skip(first, summary, roi)
        scope['second_pass'] = {'run': skip is None, 'skip_reason': skip}
        if skip is not None:
            return first, scope
        try:
            second, armed = _full_pass(scope, recognize, data, roi, bottles, ARMED)
        except Exception as error:
            _discard_armed(scope)
            scope['second_pass'].update(outcome='error', error=type(error).__name__ + ': ' + str(error)[:300])
            return first, scope
        ids = set(armed['target_ids'])
        if not ids:
            outcome = 'no_target'
        elif not ids <= set(armed['added']):
            outcome = 'unexpected_targets'
        else:
            outcome = 'recovered'
        scope['second_pass']['outcome'] = outcome
        if outcome != 'recovered':
            _discard_armed(scope)
            return first, scope
        scope['selected_pass'] = 1
        timing = second.setdefault('timing_ms', {})
        timing['atlas_geometry_repair_observe_pass'] = summary['ms']
        timing['atlas_geometry_repair_two_pass_total'] = summary['ms'] + armed['ms']
        return second, scope


def trace(scope):
    """Public G block of one request scope; ``recovered`` names only targets of the published result."""
    events, memo = deepcopy(scope['events']), scope['memo']
    return {'policy': P.POLICY, 'enabled': scope['enabled'], 'retries': scope['retries'], 'events': events,
            'passes': deepcopy(scope['passes']), 'second_pass': deepcopy(scope['second_pass']),
            'selected_pass': scope['selected_pass'], 'recovered': [sid for e in events for sid in e['added']],
            'expanded_memo': {'scope': 'one request, first observe call -> first armed call, once',
                              'stored': deepcopy(memo['stored']), 'served': deepcopy(memo['served']),
                              'calls': deepcopy(memo['calls'])}}


def locate(release):
    from rshb_vine.label_rescue import LabelRescue
    from rshb_vine.target_misses_v1.route import CanvasCloseupLocalizer
    control = release.base.runtime.control
    orphan = control.orphans
    if type(orphan) is not A.GeometryOrphanRecovery or type(orphan.labels) is not A.CapturingLabels:
        raise ValueError('Loaded orphan stage is not the frozen GeometryOrphanRecovery with capturing labels')
    if 'apply' not in vars(orphan) or 'retry' not in vars(orphan):
        raise ValueError('Front-label orphan wrap is not installed on the loaded orphan stage')
    if 'recognize' in vars(control.expanded):
        raise ValueError('control.expanded.recognize is already wrapped on this instance')
    rescue = control.expanded.core.parent.parent.parent.base.detector
    if not isinstance(rescue, LabelRescue) or type(rescue.parent) is not CanvasCloseupLocalizer:
        raise ValueError('Frozen LabelRescue -> CanvasCloseupLocalizer stack is not the loaded localizer')
    return control, orphan, rescue


def install(release):
    control, orphan, rescue = locate(release)
    frozen_apply, retry = orphan.apply, orphan.retry
    if frozen_apply.__qualname__.split('.')[-1] != 'orphan_apply' or retry.__qualname__.split('.')[-1] != 'orphan_retry':
        raise ValueError('Orphan apply/retry are not the front-label wrappers of F10')
    recovery = TrustedRegionRecovery(orphan, frozen_apply, retry)
    memo = ObserveExpandedMemo(control.expanded.recognize)
    rescue.parent = TrustedRegionLocalizer(rescue.parent)
    orphan.apply = recovery.apply
    control.expanded.recognize = memo.recognize
    return recovery, {'policy': P.POLICY, 'constants': P.CONSTANTS,
                      'localizer_path': 'base.runtime.control.expanded.core.parent.parent.parent.base.detector.parent',
                      'localizer_delegate_of': type(rescue.parent.parent).__module__ + '.CanvasCloseupLocalizer',
                      'orphan_path': 'base.runtime.control.orphans.apply',
                      'frozen_apply': frozen_apply.__module__ + '.' + frozen_apply.__qualname__,
                      'retry': retry.__module__ + '.' + retry.__qualname__,
                      'expanded_memo_path': 'base.runtime.control.expanded.recognize',
                      'expanded_memo_of': memo.frozen.__module__ + '.' + memo.frozen.__qualname__}


def _parent_ref(root):
    root = Path(root).resolve()
    profile = verify(read_json(root / PARENT_PROFILE))
    manifest = verify(read_json(root / PARENT_MANIFEST))
    if profile['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Parent profile is not the immutable F10 release cd0b9910')
    return {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM, 'sha256': sha256(root / PARENT_PROFILE),
            'manifest': {'path': PARENT_MANIFEST, 'checksum': manifest['checksum'], 'sha256': sha256(root / PARENT_MANIFEST)},
            'runtime_checksum': PARENT_RUNTIME}


def sources_sha(root):
    return {p: sha256(Path(root) / p) for p in SOURCES}


def freeze(root):
    root = Path(root).resolve()
    return seal({'kind': DESCRIPTOR_KIND, 'policy': P.POLICY, 'constants': P.CONSTANTS, 'parent_release': _parent_ref(root),
                 'sources_sha256': sources_sha(root), 'port': PORT, 'activated': False, 'release_admitted': False,
                 'calibrated': False, 'probability': None, 'public_slug': None, 'weights_changed': False, 'fit_run': False,
                 'scope': 'runs/atlas-repair-v1/scope.json', 'not_implemented': ['fragment grouping (root bridge msg5)'],
                 'predecessor_descriptor': {'path': OUT + '/candidate-v3/descriptor.json',
                                            'checksum': '57de5bf99cd9df59658d70213891664beb436bdbd327c54b72e2f42eaee51812',
                                            'sources_snapshot': OUT + '/v3-source'},
                 'rollback': 'stop %d; 8175, 8187, factory, receipts and config/recognition-current.json are never modified' % PORT})


def load_descriptor(root, path=DESCRIPTOR):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if descriptor.get('kind') != DESCRIPTOR_KIND:
        raise ValueError('Unsupported atlas geometry repair descriptor')
    if _parent_ref(root) != descriptor['parent_release']:
        raise ValueError('Parent F10 release bytes differ from the descriptor')
    if sources_sha(root) != descriptor['sources_sha256']:
        raise ValueError('Candidate sources changed since freeze')
    if descriptor['constants'] != P.CONSTANTS or descriptor['policy'] != P.POLICY:
        raise ValueError('Candidate constants differ from the descriptor')
    return descriptor


def relabel(result, checksum, runtime_checksum):
    """Own checksum wherever cd0b9910/be6e2a50 named itself; the parent only under lineage keys."""
    from rshb_vine.evidence_consistency_release_v1 import recipe as K
    from rshb_vine.text_evidence_repair_v2 import runtime as T
    from rshb_vine.text_evidence_repair_v2.release import BLOCK as TE_BLOCK
    ec, te = result[K.BLOCK], result[TE_BLOCK]
    old = list(ec.get('parent_lineage') or [])
    for block in (ec, te):
        block.update(profile_checksum=checksum, runtime_checksum=runtime_checksum,
                     parent_release_profile_checksum=PARENT_CHECKSUM, parent_runtime_checksum=PARENT_RUNTIME,
                     release_admitted=False)
    lineage = T.relabel(result, checksum, runtime_checksum, PARENT_CHECKSUM, PARENT_RUNTIME, admitted=False)
    if lineage[1:] != old:
        raise RuntimeError('Parent lineage of the F10 response differs from its evidence-consistency block')
    ec['parent_lineage'] = te['parent_lineage'] = lineage
    return lineage


class AtlasGeometryCandidate:
    def __init__(self, root, descriptor_path=DESCRIPTOR, capture=False):
        from rshb_vine.evidence_consistency_release_v1.release import EvidenceConsistencyRelease
        root = Path(root).resolve()
        self.descriptor = load_descriptor(root, descriptor_path)
        release = EvidenceConsistencyRelease(root, PARENT_PROFILE)
        if release.profile['checksum'] != PARENT_CHECKSUM or release.manifest['checksum'] != PARENT_RUNTIME:
            raise ValueError('Loaded F10 release differs from cd0b9910/be6e2a50')
        self.recovery, self.installation = install(release)
        self.release, self.profile, self.capture = release, self.descriptor, capture
        self.manifest = seal({'kind': 'atlas-geometry-repair-v1-candidate-runtime',
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': PARENT_RUNTIME, 'parent_release_profile_checksum': PARENT_CHECKSUM,
                              'installation': self.installation, 'activated': False, 'calibration_status': 'unknown'})
        self.last_captured = None

    def recognize(self, data, roi=None, bottles='addressed'):
        started = time.perf_counter()
        result, scope = recognize_with_recovery(self.release.recognize, data, roi, bottles, self.capture)
        self.last_captured = scope['captured']
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        own, runtime = self.descriptor['checksum'], self.manifest['checksum']
        lineage = relabel(result, own, runtime)
        result[BLOCK] = {**trace(scope), 'profile_checksum': own, 'descriptor_checksum': own, 'runtime_checksum': runtime,
                         'parent_release_profile_checksum': PARENT_CHECKSUM, 'parent_runtime_checksum': PARENT_RUNTIME,
                         'parent_lineage': lineage, 'seconds': time.perf_counter() - started, 'release_admitted': False,
                         'calibrated': False, 'probability': None}
        return result
