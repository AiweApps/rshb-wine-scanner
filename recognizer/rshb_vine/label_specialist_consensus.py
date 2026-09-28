"""Isolated ablation: remove only mandatory B3-context agreement."""
import copy
from rshb_vine.resolution.identity import COLOR, GRAPES, SUGAR, values, compare
from rshb_vine.gallery_variant_text import signature


class VariantGuard:
    def __init__(self, catalog, signatures, product_groups):
        self.fields = {r['slug']: r['fields'] for r in catalog}
        self.signatures = signatures
        self.groups = product_groups

    def check(self, before, after, observations):
        if not after or after == before:
            return True, []
        usable = [o for o in observations if o['score'] >= .85]
        f = self.fields[after]
        reasons = []
        for key, aliases, field in [('color', COLOR, 'Категория'),
                                    ('grape', GRAPES, 'Сорт винограда'),
                                    ('sugar', SUGAR, 'Название вина')]:
            seen = set().union(*(values(o['raw_text'], aliases) for o in usable))
            expected = values(f.get(field, ''), aliases)
            if compare(seen, expected, subset=key == 'grape') == 'contradicted':
                reasons.append(key + ':explicit_conflict')
        query = signature(observations)
        ref = self.signatures.get(after, {})
        if query['style'] and ref.get('style') and query['style'] != ref['style']:
            reasons.append('style:explicit_conflict')
        if query['year'] and ref.get('year') and query['year'] != ref['year']:
            reasons.append('year:explicit_conflict')
        a, b = self.groups.get(before), self.groups.get(after)
        if (a and b and a['product_group'] == b['product_group']
                and a.get('year') and b.get('year') and a['year'] != b['year']
                and query['year'] != str(b['year'])):
            reasons.append('same_product_different_year:requires_positive_year')
        return not reasons, reasons


def consensus_candidate(b0, b3):
    def top(ret, kind):
        rows = ret.get('channel_top20', {}).get(kind, [])
        return rows[0]['slug'] if rows else None
    before, proposal = b0.get('best_candidate'), b3.get('best_candidate')
    if (proposal and proposal != before
            and proposal == top(b3, 'front_label') == top(b0, 'context')
            and before == top(b0, 'front_label')):
        return proposal
    return before


def combine(b0, b3, guard, title):
    """Keep physical targets fixed; return an uncalibrated candidate trace."""
    result = copy.deepcopy(b0)
    traces = []
    for target in result['targets']:
        sid = str(target['instance_id'])
        old = target['retrieval']
        other = next(t['retrieval'] for t in b3['targets'] if str(t['instance_id']) == sid)
        obs = b0.get('variant_text', {}).get('observations', []) if b0.get('variant_text', {}).get('performed') else []
        if len(result['targets']) > 1:
            obs = next((t.get('observations', []) for t in b0.get('instance_text', {}).get('targets', [])
                        if t.get('performed') and str(t['instance_id']) == sid), [])
        proposal = consensus_candidate(old, other)
        allowed, reasons = guard.check(old['best_candidate'], proposal, obs)
        selected = proposal if allowed else old['best_candidate']
        chosen = copy.deepcopy(other if selected != old['best_candidate'] else old)
        ranked, evidence = title.rerank(chosen['ranked_candidates'], obs)
        title_proposal = ranked[0]['slug'] if ranked else None
        title_allowed, title_reasons = guard.check(selected, title_proposal, obs)
        if title_allowed:
            chosen.update(ranked_candidates=ranked, best_candidate=title_proposal)
        target['retrieval'] = chosen
        traces.append({'instance_id': sid, 'before': old['best_candidate'],
                       'after': chosen['best_candidate'], 'consensus_proposal': proposal,
                       'guard': reasons, 'title_proposal': title_proposal,
                       'title_guard': title_reasons})
    if len(result['targets']) == 1:
        ret = result['targets'][0]['retrieval']
        result.update(best_candidate=ret['best_candidate'], ranked_candidates=ret['ranked_candidates'])
    result['guarded_model_consensus'] = {'policy': 'label-specialist-consensus-v1', 'targets': traces}
    result.pop('timing_ms', None)
    return result
