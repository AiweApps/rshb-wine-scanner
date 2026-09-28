"""One fixed equal-weight reciprocal-rank admission plus verified sugar veto."""
from copy import deepcopy
from fractions import Fraction
from rshb_vine.resolution.identity import values, SUGAR


class RankFusedB3Consensus:
    def __init__(self, parent, sugar_ledger):
        self.parent = parent
        self.sugar_ledger = sugar_ledger

    @staticmethod
    def reciprocal(rows, slug):
        return next((Fraction(1, rank) for rank, row in enumerate(rows[:20], 1)
                     if row['slug'] == slug), Fraction(0))

    def apply(self, baseline, raw_b3):
        candidate = self.parent.apply(baseline, raw_b3)
        result = deepcopy(baseline)
        original = {str(t['instance_id']): t for t in baseline.get('targets', [])}
        raw = {str(t['instance_id']): t for t in raw_b3.get('targets', [])}
        proposed = {str(t['instance_id']): t for t in candidate.get('targets', [])}
        traces = []
        for target, parent_trace in zip(result.get('targets', []), candidate['alternative_b3_consensus']['targets']):
            sid = str(target['instance_id'])
            trace = deepcopy(parent_trace)
            traces.append(trace)
            if not trace['changed']:
                continue
            before, after = trace['before'], trace['after']
            sums = {before: Fraction(0), after: Fraction(0)}
            channel_trace = []
            for label, retrieval in [('baseline', original[sid]['retrieval']), ('B3', raw[sid]['retrieval'])]:
                for kind in ['front_label', 'context']:
                    rows = retrieval.get('channel_top20', {}).get(kind, [])
                    a, b = self.reciprocal(rows, before), self.reciprocal(rows, after)
                    sums[before] += a
                    sums[after] += b
                    channel_trace.append({'model': label, 'channel': kind,
                                          'before_reciprocal': str(a), 'proposal_reciprocal': str(b)})
            observations = self.parent.observations(baseline, sid)
            seen = set().union(*(values(o['raw_text'], SUGAR)
                                     for o in observations if o['score'] >= .85))
            fact = self.sugar_ledger.get(after, {})
            contradiction = len(seen) == 1 and bool(fact.get('verified')) and fact['value'] not in seen
            trace.update(rank_fusion={'before_sum': str(sums[before]), 'proposal_sum': str(sums[after]),
                                      'channels': channel_trace, 'strictly_greater': sums[after] > sums[before]},
                         observed_sweetness=sorted(seen), proposal_sugar=fact,
                         verified_sugar_contradiction=contradiction)
            if sums[after] <= sums[before] or contradiction:
                trace.update(after=before, changed=False,
                             reason='verified_sugar_contradiction' if contradiction else 'rank_fusion_not_strictly_greater')
                continue
            target['retrieval'] = deepcopy(proposed[sid]['retrieval'])
            trace['reason'] = 'B3_agreement_strict_rank_fusion_preserves_evidence'
        if len(result.get('targets', [])) == 1 and any(t['changed'] for t in traces):
            retrieval = result['targets'][0]['retrieval']
            result.update(best_candidate=retrieval['best_candidate'], ranked_candidates=retrieval['ranked_candidates'])
            if result.get('slug') is not None:
                result['slug'] = None
            if 'catalog_additions' in result:
                result['catalog_additions']['related_catalog_slugs'] = retrieval.get('related_catalog_slugs', [])
        result['alternative_b3_consensus'] = {'policy': 'rank-fused-b3-consensus-v1', 'targets': traces, 'calibrated': False}
        result['rank_fused_b3_consensus'] = {'policy': 'rank-fused-b3-consensus-v1', 'channels': 4,
                                          'weights': 'equal', 'tie': 'keep_baseline', 'rank_limit': 20}
        result.pop('timing_ms', None)
        return result
