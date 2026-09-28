"""Fixed ablation: B3 label/context agreement may challenge a visual baseline.

Pure real-receipt policy: no model calls, no altered physical geometry or OCR.
"""
from copy import deepcopy


class AlternativeB3Consensus:
    def __init__(self, single, multi, guard, preservation, aliases=None):
        self.single, self.multi = single, multi
        self.guard, self.preservation = guard, preservation
        self.aliases = aliases or {}

    @staticmethod
    def observations(baseline, sid):
        if len(baseline.get('targets', [])) == 1:
            packet = baseline.get('variant_text', {})
        else:
            packet = next((p for p in baseline.get('instance_text', {}).get('targets', [])
                           if str(p['instance_id']) == sid), {})
        return packet.get('observations', []) if packet.get('performed') else []

    def apply(self, baseline, raw_b3):
        result = deepcopy(baseline)
        traces = []
        other = {str(t['instance_id']): t for t in raw_b3.get('targets', [])}
        if set(other) != {str(t['instance_id']) for t in baseline.get('targets', [])}:
            raise ValueError('B3 physical-target roster differs from source')
        for target in result.get('targets', []):
            sid = str(target['instance_id'])
            old = target['retrieval']
            new = deepcopy(other[sid]['retrieval'])
            before = old.get('best_candidate')
            visual = old.get('visual_ranked_candidates', [])
            baseline_visual = visual[0]['slug'] if visual else None
            observations = self.observations(baseline, sid)
            policy = self.single if len(result['targets']) == 1 else self.multi
            ranked, text = policy.rerank(new.get('visual_ranked_candidates', new['ranked_candidates']), observations)
            proposal = ranked[0]['slug'] if ranked else None
            channels = new.get('channel_top20', {})
            label = channels.get('front_label', [])
            context = channels.get('context', [])
            label_top = label[0]['slug'] if label else None
            context_top = context[0]['slug'] if context else None
            trace = {'instance_id': sid, 'before': before, 'after': before,
                     'baseline_visual_leader': baseline_visual, 'proposal': proposal,
                     'B3_front_top1': label_top, 'B3_context_top1': context_top,
                     'text_evidence': text, 'changed': False}
            traces.append(trace)
            if before != baseline_visual:
                trace['reason'] = 'preserve_baseline_text_override'
                continue
            if not proposal or proposal == before or proposal != label_top or proposal != context_top:
                trace['reason'] = 'no_different_B3_internal_agreement'
                continue
            allowed, reasons = self.guard.check(before, proposal, observations)
            lost = self.preservation.lost_evidence(observations, before, proposal)
            trace.update(guard=reasons, lost_evidence=lost)
            if not allowed or lost is None or lost:
                trace['reason'] = 'evidence_guard'
                continue
            new.update(ranked_candidates=ranked, best_candidate=proposal)
            new['related_catalog_slugs'] = sorted(a for a, c in self.aliases.items() if c == proposal)
            target['retrieval'] = new
            trace.update(after=proposal, changed=True, reason='B3_internal_agreement_preserves_evidence',
                         old_pool=[r['slug'] for r in old.get('ranked_candidates', [])],
                         new_pool=[r['slug'] for r in ranked])
        if len(result.get('targets', [])) == 1 and any(t['changed'] for t in traces):
            ret = result['targets'][0]['retrieval']
            result.update(best_candidate=ret['best_candidate'], ranked_candidates=ret['ranked_candidates'])
            if result.get('slug') is not None:
                result['slug'] = None  # No calibrated acceptance is introduced.
        result['alternative_b3_consensus'] = {'policy': 'alternative-b3-consensus-v1',
                                             'targets': traces, 'calibrated': False}
        result.pop('timing_ms', None)
        return result
