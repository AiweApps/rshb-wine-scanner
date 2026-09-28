"""Bounded preservation of a fully read, longer visual-leader product name.

Does not add candidates or interpret missing OCR as negative evidence. Only
reverses a title-only override that shortened an explicitly matched title.
"""
from copy import deepcopy
from rshb_vine.product_name_selector import ProductNameSelector
from rshb_vine.positive_variant_text import contains


class SeriesEvidenceChallenger:
    def __init__(self, catalog):
        self.names = ProductNameSelector(catalog, {})

    def apply(self, baseline):
        result = deepcopy(baseline)
        changes = []
        traces = baseline.get('guarded_model_consensus', {}).get('targets', [])
        for target in result.get('targets', []):
            sid = str(target['instance_id'])
            trace = next((t for t in traces if str(t['instance_id']) == sid), {})
            ret = target['retrieval']
            visual = ret.get('visual_ranked_candidates', [])
            current = ret.get('best_candidate')
            leader = visual[0]['slug'] if visual else None
            if not leader or current == leader:
                continue
            if not (trace.get('before') == leader == trace.get('consensus_proposal')
                    and trace.get('title_proposal') == current == trace.get('after')
                    and not trace.get('guard') and not trace.get('title_guard')):
                continue
            if len(result['targets']) == 1:
                packet = baseline.get('variant_text', {})
            else:
                packet = next((t for t in baseline.get('instance_text', {}).get('targets', [])
                               if str(t['instance_id']) == sid), {})
            if not packet.get('performed'):
                continue
            if leader not in self.names.items or current not in self.names.items:
                continue
            if self.names.items[leader]['producer'] != self.names.items[current]['producer']:
                continue
            _, evidence = self.names.rerank(visual, packet.get('observations', []))
            matched = evidence.get('matched', {})
            full, short = matched.get(leader), matched.get(current)
            if not full or not short or full == short or not contains(full, short):
                continue
            rows = ret.get('ranked_candidates', [])
            if not any(r['slug'] == leader for r in rows):
                continue
            ret['ranked_candidates'] = [r for r in rows if r['slug'] == leader] + [r for r in rows if r['slug'] != leader]
            ret['best_candidate'] = leader
            if ret.get('slug') == current:
                ret['slug'] = leader
            changes.append({'instance_id': sid, 'before': current, 'after': leader,
                            'preserved_name': full, 'shorter_name': short})
        if len(result.get('targets', [])) == 1:
            ret = result['targets'][0]['retrieval']
            result.update(best_candidate=ret['best_candidate'], ranked_candidates=ret['ranked_candidates'])
            if result.get('slug') == baseline.get('best_candidate'):
                result['slug'] = ret['best_candidate']
        result['series_evidence_challenger'] = {'policy': 'series-evidence-v1', 'changes': changes}
        return result
