"""Experimental source-verified color selection within an explicit product family."""
from copy import deepcopy
import re
from rshb_vine.alternative_b3_consensus import AlternativeB3Consensus
from rshb_vine.resolution.identity import norm


def color_phrase(text):
    """Require wine style context, so names such as White Horse are not colors."""
    text = norm(text)
    matches = re.findall(r'\b(white|red|rose|rosé)\s+(?:wine|(?:semi\s*)?(?:sweet|dry)|extra\s+brut|brut)\b', text)
    return {'rose' if x == 'rosé' else x for x in matches}


class ReferenceColorEvidence:
    def __init__(self, facts):
        self.facts = {r['slug']: r for r in facts if r.get('verified')}

    @staticmethod
    def family(fact):
        return tuple(norm(fact[k]) for k in ('brand_text', 'product_text', 'sweetness'))

    def apply(self, baseline):
        result = deepcopy(baseline)
        traces = []
        for target in result.get('targets', []):
            sid = str(target['instance_id'])
            obs = [o for o in AlternativeB3Consensus.observations(baseline, sid) if o['score'] >= .85]
            seen = set().union(*(color_phrase(o['raw_text']) for o in obs))
            ret = target['retrieval']; before = ret['best_candidate']; old = self.facts.get(before)
            trace = {'instance_id': sid, 'before': before, 'after': before, 'observed_color': sorted(seen), 'reason': 'insufficient_or_noncontradicting_evidence'}
            traces.append(trace)
            if not old or len(seen) != 1 or old['color'] in seen:
                continue
            # Require the product name on this target; family equality also preserves brand and sweetness.
            texts = [' '+norm(o['raw_text'])+' ' for o in obs]
            if not any(' '+norm(old['product_text'])+' ' in text for text in texts):
                trace['reason'] = 'product_name_not_read'
                continue
            candidates = [r['slug'] for r in ret['ranked_candidates'] if r['slug'] in self.facts
                          and self.family(self.facts[r['slug']]) == self.family(old)
                          and self.facts[r['slug']]['color'] in seen]
            if len(candidates) != 1:
                trace['reason'] = 'no_unique_verified_family_candidate'
                continue
            chosen = candidates[0]
            ret['best_candidate'] = chosen
            ret['ranked_candidates'] = sorted(ret['ranked_candidates'], key=lambda r:r['slug'] != chosen)
            ret['slug'] = None
            ret['probability_correct'] = None
            ret['related_catalog_slugs'] = []
            trace.update(after=chosen, reason='explicit_color_verified_product_family')
        if len(result.get('targets', [])) == 1 and any(t['before'] != t['after'] for t in traces):
            ret = result['targets'][0]['retrieval']
            result.update(best_candidate=ret['best_candidate'], ranked_candidates=ret['ranked_candidates'], slug=None, probability_correct=None)
        result['reference_color_evidence'] = {'policy': 'reference-color-family-v1', 'targets': traces, 'calibrated': False}
        return result
