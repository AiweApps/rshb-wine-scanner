"""One isolated bottle view may resolve an incomplete-label disagreement.

Keep the original physical target and candidate pool. A proposed replacement
must agree with the original context leader and preserve confident multiword
identity evidence observed on either the original target or isolated bottle.
"""
from copy import deepcopy
import io
import time

from rshb_vine.preprocessing import decode
from rshb_vine.resolution.identity import norm


def eligible_targets(result, roi=None):
    if roi is not None or result.get('decision') == 'invalid_image':
        return []
    found = []
    for target in result.get('targets', []):
        bottle = target.get('bottle_bbox')
        label = target['bbox']
        if not bottle or bottle[3] <= bottle[1]:
            continue
        height = bottle[3] - bottle[1]
        if ((label[1] - bottle[1]) / height >= .5
                and (label[3] - label[1]) / height <= .5
                and target['retrieval'].get('channel_disagreement')):
            found.append(target)
    return found


class BottleIsolationEvidence:
    def __init__(self, catalog, retry):
        self.catalog = {row['slug']: row['fields'] for row in catalog}
        self.retry = retry

    def lost_evidence(self, observations, before, after):
        # Missing metadata is not permission to discard positive old evidence.
        if before not in self.catalog or after not in self.catalog:
            return None
        fields = ('Винодельня', 'Название вина', 'Сорт винограда')
        old = [norm(self.catalog[before].get(k, '')) for k in fields]
        new = [norm(self.catalog[after].get(k, '')) for k in fields]
        lost = set()
        for observation in observations:
            if observation['score'] < .85:
                continue
            words = norm(observation['raw_text']).split()
            for size in range(2, len(words) + 1):
                for offset in range(len(words) - size + 1):
                    span = ' '.join(words[offset:offset + size])
                    if (any(' ' + span + ' ' in ' ' + phrase + ' ' for phrase in old)
                            and not any(' ' + span + ' ' in ' ' + phrase + ' ' for phrase in new)):
                        lost.add(span)
        return sorted(lost)

    def apply_retry(self, baseline, retry, roi=None):
        """Resolve a real retry result; also usable for immutable receipt replay."""
        result = deepcopy(baseline)
        targets = eligible_targets(baseline, roi)
        trace = {'policy': 'bottle-isolation-evidence-v1', 'attempted': False,
                 'eligible_targets': len(targets), 'changed': False}
        result['bottle_isolation_evidence'] = trace
        if len(targets) != 1:
            trace['reason'] = 'not_one_eligible_target'
            return result
        target = targets[0]
        sid = str(target['instance_id'])
        ret = target['retrieval']
        before = ret['best_candidate']
        context = ret.get('channel_top20', {}).get('context', [])
        leader = context[0]['slug'] if context else None
        proposed = retry.get('best_candidate')
        trace.update(attempted=True, instance_id=sid, before=before, after=before,
                     proposed=proposed, context_leader=leader,
                     source_bottle_bbox=target['bottle_bbox'],
                     retry_target_count=len(retry.get('targets', [])),
                     retry_decision=retry.get('decision'))
        if len(retry.get('targets', [])) != 1 or not leader or proposed != leader:
            trace['reason'] = 'retry_not_singleton_context_agreement'
            return result
        if proposed == before:
            trace['reason'] = 'same_answer'
            return result
        if proposed not in {row['slug'] for row in ret.get('ranked_candidates', [])}:
            trace['reason'] = 'proposal_not_in_existing_ranked_pool'
            return result
        observations = list(retry.get('variant_text', {}).get('observations', []))
        if len(baseline.get('targets', [])) == 1:
            original = baseline.get('variant_text', {}).get('observations', [])
        else:
            original = next((p.get('observations', [])
                             for p in baseline.get('instance_text', {}).get('targets', [])
                             if str(p['instance_id']) == sid), [])
        observations.extend(original)
        lost = self.lost_evidence(observations, before, proposed)
        trace.update(lost_evidence=lost,
                     retry_observations=retry.get('variant_text', {}).get('observations', []),
                     original_observations=original,
                     retry_coordinate_space='isolated bottle crop; not merged into original OCR')
        if lost is None or lost:
            trace['reason'] = 'missing_catalog_metadata' if lost is None else 'protected_identity_evidence'
            return result
        selected = next(t for t in result['targets'] if str(t['instance_id']) == sid)
        retrieval = selected['retrieval']
        retrieval['ranked_candidates'] = sorted(retrieval['ranked_candidates'],
                                                 key=lambda row: row['slug'] != proposed)
        retrieval['best_candidate'] = proposed
        if retrieval.get('slug') == before:
            retrieval['slug'] = proposed
        if len(result['targets']) == 1:
            result.update(best_candidate=proposed, ranked_candidates=retrieval['ranked_candidates'])
            if result.get('slug') == before:
                result['slug'] = proposed
        trace.update(after=proposed, changed=True, reason='context_agreement_preserves_read_identity')
        return result

    def apply(self, data, baseline, roi=None):
        targets = eligible_targets(baseline, roi)
        if len(targets) != 1:
            return self.apply_retry(baseline, {}, roi)
        started = time.perf_counter()
        image, _ = decode(data)
        # Match the original bounded experiment's PIL crop and compression.
        crop = image.crop(targets[0]['bottle_bbox'])
        stream = io.BytesIO()
        crop.save(stream, format='PNG', compress_level=1)
        if len(stream.getvalue()) > 20 * 1024 * 1024:
            result = deepcopy(baseline)
            result['bottle_isolation_evidence'] = {
                'policy': 'bottle-isolation-evidence-v1', 'attempted': False,
                'reason': 'payload_limit', 'changed': False, 'eligible_targets': 1}
            return result
        result = self.apply_retry(baseline, self.retry(stream.getvalue()), roi)
        result['bottle_isolation_evidence']['ms'] = 1000 * (time.perf_counter() - started)
        return result
