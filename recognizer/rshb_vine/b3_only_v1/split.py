"""Explicit object-gate / retrieval split installed per instance via __class__ (no frozen or global mutation).

SplitWineObjectStage sits directly before WineObjectStage in the loaded instance MRO, so the rescue, ROI,
orientation and instance-text layers above it are unchanged. Its verifier body follows the frozen
ProposalVerifierPipeline.recognize_image line for line except that the gate score uses the original B0
encoder on context views only, and retrieval encodes/searches the label-bearing views with ``self.base``,
which the runtime binds to the stage5 B3 encoder and index.
"""
from copy import deepcopy
import time

import numpy as np

from rshb_vine.bottle_instances import area, intersection, iou
from rshb_vine.label_first_pipeline import consolidate_regions
from rshb_vine.preprocessing import checked_box
from rshb_vine.visual_core import View
from rshb_vine.wine_object_profile import WineObjectStage


class GateContext:
    """The only scope in which the original B0 encoder may run."""

    def __init__(self):
        self.depth = 0

    def __enter__(self):
        self.depth += 1

    def __exit__(self, *_):
        self.depth -= 1


GATE = GateContext()


def guard_b0_encoder(raw):
    """Instance-level guard on the raw B0 model: any hidden B0 inference outside the gate raises."""
    original = raw.encode

    def encode(images):
        if not GATE.depth:
            raise RuntimeError('B3-only candidate: B0 encoder called outside the object gate')
        return original(images)
    raw.encode = encode


class RoleCounters:
    KEYS = ('b0_gate_encode_calls', 'b0_gate_encode_images', 'b3_base_route_search_calls',
            'b3_selection_arm_searches', 'b3_selection_arm_reuses', 'b0_index_search_calls',
            'b0_raw_visual_arms', 'dual_fusion_neutralized_targets', 'core_consensus_single_arm_requests')

    def __init__(self):
        self.values = dict.fromkeys(self.KEYS, 0)

    def add(self, key, value=1):
        self.values[key] += value

    def reset(self):
        self.values = dict.fromkeys(self.KEYS, 0)


def _batched(encoder, crops):
    return np.concatenate([encoder.encode(crops[j:j + 8]) for j in range(0, len(crops), 8)]) if crops else []


def split_verifier(self, image, roi=None):
    t = time.perf_counter(); regions = self.detector.detect(image); parents = {}
    regions, _ = consolidate_regions(regions)
    for j, r in enumerate(regions):
        if 'context_bbox' not in r:
            r = {**r, 'context_bbox': [0, 0, *image.size], 'parent_id': f'closeup-{j}', 'kind': 'detected_label', 'unlocalized': True}
        box = checked_box(r['context_bbox'], image.size)
        if roi and not (roi[0] <= (box[0] + box[2]) / 2 <= roi[2] and roi[1] <= (box[1] + box[3]) / 2 <= roi[3]):
            continue
        p = parents.setdefault(r['parent_id'], dict(parent_id=r['parent_id'], context_bbox=box, labels=[], unlocalized=r.get('unlocalized', False), detector_score=r['detector_score']))
        if r['kind'] == 'detected_label':
            p['labels'].append(r['bbox'])
    detector_ms = (time.perf_counter() - t) * 1000
    parents = list(parents.values()); views = []
    for p in parents:
        p['context_vector'] = len(views); views.append(View('context', p['context_bbox'], 'detected_bottle', True, list(image.size), self.detector.model_id))
        if p['labels']:
            bs = p['labels']; p['bbox'] = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]; p['label_vector'] = len(views)
            views.append(View('detected_label', checked_box(p['bbox'], image.size), 'automatic_main_label', None, list(image.size), self.detector.model_id))
    t = time.perf_counter()
    gate_ids = [p['context_vector'] for p in parents]
    with GATE:
        gate = _batched(self.gate_encoder, [image.crop(views[i].bbox) for i in gate_ids])
    self.gate_counters.add('b0_gate_encode_calls', (len(gate_ids) + 7) // 8)
    self.gate_counters.add('b0_gate_encode_images', len(gate_ids))
    gate_ms = (time.perf_counter() - t) * 1000
    retrieval_ids = [i for p in parents if p['labels'] for i in (p['context_vector'], p['label_vector'])]
    t = time.perf_counter()
    b3 = _batched(self.base.encoder, [image.crop(views[i].bbox) for i in retrieval_ids])
    row = {i: k for k, i in enumerate(retrieval_ids)}
    encode_ms = (time.perf_counter() - t) * 1000; t = time.perf_counter(); candidates = []; instances = []
    for n, p in enumerate(parents):
        score = float(1 / (1 + np.exp(-np.clip(np.dot(gate[n], self.coef) + self.bias, -40, 40))))
        ins = dict(bottle_id=str(p['parent_id']), bottle_bbox=None if p['unlocalized'] else p['context_bbox'], label_bbox=p.get('bbox'), wine_score=score, status='label_available' if p['labels'] else 'insufficient_evidence')
        instances.append(ins)
        if not p['labels']:
            continue
        ids = [p['context_vector'], p['label_vector']]
        retrieval = self.base.index.search(b3[[row[i] for i in ids]], [views[i] for i in ids], self.base.encoder_id)
        candidates.append(dict(instance_id=str(p['parent_id']), parent_id=p['parent_id'], bbox=p['bbox'], bottle_bbox=None if p['unlocalized'] else p['context_bbox'], context_bbox=p['context_bbox'], physical_bottle_localized=not p['unlocalized'], kind='detected_label', detector_score=p['detector_score'], wine_score=score, retrieval=retrieval))
    kept = []; duplicates = []
    for c in sorted(candidates, key=lambda r: -r['detector_score']):
        duplicate = next((k for k in kept if iou(c['bbox'], k['bbox']) > .5 and intersection(c['context_bbox'], k['context_bbox']) / max(min(area(c['context_bbox']), area(k['context_bbox'])), 1) > .9), None)
        if duplicate:
            duplicates.append(dict(instance_id=c['instance_id'], kept_instance_id=duplicate['instance_id']))
        else:
            kept.append(c)

    def resolve(targets, ins):
        r = dict(decision='insufficient_evidence', slug=None, best_candidate=None, probability_correct=None, ranked_candidates=[], targets=targets, instances=ins, requires_target_selection=len(targets) > 1, reasons=['confidence_not_calibrated'], suppressed_instances=duplicates)
        if len(targets) == 1:
            found = targets[0]['retrieval']; r.update(decision=found['decision'], best_candidate=found['best_candidate'], ranked_candidates=found['ranked_candidates'])
        elif len(targets) > 1:
            r['decision'] = 'ambiguous_target'
        else:
            r['reasons'].append('no_verified_bottle_with_label')
        return r
    structure = resolve(kept, instances)
    threshold = self.spec['threshold']; result = resolve([c for c in kept if c['wine_score'] >= threshold], [i for i in instances if i['wine_score'] >= threshold])
    result['rejected_instances'] = [i for i in instances if i['wine_score'] < threshold]; result['structure_only'] = structure
    result['timing_ms'] = dict(detector=detector_ms, encode=gate_ms + encode_ms, gate_encode=gate_ms, retrieval=(time.perf_counter() - t) * 1000, ocr=0.)
    return result


class SplitWineObjectStage(WineObjectStage):
    def recognize_image(self, image, roi=None):
        result = split_verifier(self, image, roi)
        structure = result.pop('structure_only')
        result['instances'] = structure['instances']
        result['object_filter'] = {
            'policy': 'frozen-feature-wine-object-v1',
            'threshold': self.spec['threshold'],
            'targets_before': len(structure['targets']),
            'targets_after': len(result['targets']),
            'rejected_instance_ids': [i['bottle_id'] for i in result['rejected_instances']],
            'gate_encoder_id': self.gate_encoder_id,
            'retrieval_encoder_id': self.base.encoder_id,
        }
        result['diagnostic_low_wine_score_instances'] = result['rejected_instances']
        result['verifier_policy'] = 'wine_object_target_gate'
        return result


def install_split(instance, gate_encoder, gate_encoder_id, counters):
    """Rebind one loaded instance to a class whose MRO places SplitWineObjectStage right before WineObjectStage."""
    cls = type(instance)
    split = type('B3Only' + cls.__name__, (cls, SplitWineObjectStage), {'__module__': __name__})
    mro = split.__mro__
    if mro.index(SplitWineObjectStage) + 1 != mro.index(WineObjectStage) or mro[1] is not cls:
        raise ValueError('Split stage is not directly before WineObjectStage in the instance MRO')
    instance.gate_encoder, instance.gate_encoder_id, instance.gate_counters = gate_encoder, gate_encoder_id, counters
    instance.__class__ = split
    return [c.__name__ for c in mro]


class CountingIndex:
    def __init__(self, index, counters, search_key, reuse_key=None):
        self._index, self.counters, self.search_key, self.reuse_key = index, counters, search_key, reuse_key

    def __getattr__(self, name):
        return getattr(self._index, name)

    def search(self, query, views, encoder_id):
        self.counters.add(self.search_key)
        return self._index.search(query, views, encoder_id)

    def reuse(self, images, views, encoder_id):
        found = self._index.reuse(images, views, encoder_id)
        if found is not None and self.reuse_key:
            self.counters.add(self.reuse_key)
        return found


def install_single_arm_core(core, counters):
    """GuardedConsensus keeps its title/variant guards but combines the B3 baseline with itself: no second B3 vote."""
    import sys
    cls = type(core)
    combine = sys.modules[cls.__module__].combine

    def recognize(self, data, roi=None):
        started = time.perf_counter()
        baseline = self.parent.recognize(data, roi)
        if not baseline.get('targets'):
            return baseline
        result = combine(baseline, baseline, self.guard, self.title)
        result['guarded_model_consensus']['B3_performed'] = False
        result['guarded_model_consensus']['b3_only_single_arm'] = 'competitor=baseline; no second B3 vote'
        counters.add('core_consensus_single_arm_requests')
        result['timing_ms'] = {'total': (time.perf_counter() - started) * 1000}
        return result
    core.__class__ = type('B3OnlySingleArm' + cls.__name__, (cls,), {'recognize': recognize, '__module__': __name__})


class SingleArmDual:
    """Parent of the kept ReferencePhrasePreservation: replaces B0xB3 rank fusion; no B3xB3 self-agreement."""

    def __init__(self, counters):
        self.counters = counters

    def apply(self, baseline, raw_b3):
        result = deepcopy(baseline)
        traces = []
        for target in result.get('targets', []):
            best = target['retrieval'].get('best_candidate')
            traces.append({'instance_id': str(target['instance_id']), 'before': best, 'after': best, 'changed': False,
                           'reason': 'b3_only_single_arm_no_dual_fusion'})
        self.counters.add('dual_fusion_neutralized_targets', len(traces))
        result['alternative_b3_consensus'] = {'policy': 'b3-only-single-arm-neutralized', 'targets': traces,
                                             'calibrated': False}
        result['rank_fused_b3_consensus'] = {'policy': 'neutralized', 'reason': 'B3 would be fused with itself'}
        result.pop('timing_ms', None)
        return result


def tripwire(obj, names):
    def fail(*_, **__):
        raise RuntimeError('B3-only candidate: original B0 index must not be searched')
    for name in names:
        setattr(obj, name, fail)
