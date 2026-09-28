"""Counter-evidence guard: shared (non-exclusive) name text cannot overturn an agreed visual top-1.

The frozen ranker may put first a candidate whose only advantage over the product that is rank 1 in every
visual channel is a name reading whose tokens are also claimed by other products in the pool (a legal
category phrase, a grape named on a blend list, one script of a transliterated name). Such a reading does
not discriminate that candidate when the name is only partly read, or when its complete reading is a grape
that the agreed same-producer card lists in its blend; the agreed visual top-1 is then kept. The override
stands whenever the text winner has exclusive name, stronger producer, fewer contradictions or a complete
unexplained name reading. Scores and the full order are kept.
"""
from copy import deepcopy

from rshb_vine.systemic_ranking_v2 import evidence as E

RULE = {
    'version': 'shared-name-visual-agreement-guard-v2',
    'visual_agreement': 'one candidate is rank 1 in every visual channel that has a rank-1 row; at least two channels',
    'block_when_all': ['text winner exclusive name weight == 0',
                       'text winner name.read_w > agreed candidate name.read_w',
                       'text winner prod.read_w <= agreed candidate prod.read_w',
                       'agreed candidate has no contradiction feature larger than the text winner',
                       'text winner name is not completely read, or its complete name is a grape listed by a '
                       'same-producer agreed card whose own name lacks it (blend explanation)'],
}
COMPLETE = 1 - 1e-9
CONTRA = [n for n in E.CONTRADICTION_FEATURES]


def agreed_visual(evidence):
    channels = {ch for ev in evidence for ch, rank in (ev.get('visual_ranks') or {}).items() if rank == 1}
    if len(channels) < 2:
        return None
    agreed = [i for i, ev in enumerate(evidence) if all((ev.get('visual_ranks') or {}).get(ch) == 1 for ch in channels)]
    return (agreed[0], sorted(channels)) if len(agreed) == 1 else None


def _explained(profiles, winner_slugs, agreed_slugs):
    for y in winner_slugs:
        py = profiles(y)
        for x in agreed_slugs:
            px = profiles(x)
            if (py['producer_key'] and py['producer_key'] == px['producer_key'] and py['name']
                    and py['name'] <= px['grapes'] and not py['name'] <= px['name']):
                return [y, x]
    return None


def check(out, profiles):
    """Guard decision for one evaluated target; ``profiles(slug)`` gives producer key, grape and name skeletons."""
    raw = out.get('raw_proposal')
    trace = {'rule': RULE['version'], 'blocked': False}
    if not raw or not raw['ranked_candidates']:
        trace['reason'] = 'empty_pool'
        return trace
    agreed = agreed_visual(out['evidence'])
    winner = raw['ranked_candidates'][0]['input_pool_index']
    if agreed is None or agreed[0] == winner:
        trace['reason'] = 'no_agreed_visual' if agreed is None else 'winner_is_agreed_visual'
        return trace
    x, y = agreed[0], winner
    fx, fy = out['rows'][x], out['rows'][y]
    exclusive = max((n['exclusive'] for n in out['evidence'][y]['name']), default=0.)
    contra_x = {n: fx[n] for n in CONTRA if fx[n] > fy[n]}
    complete = max((n['cover'] for n in out['evidence'][y]['name']), default=0.) >= COMPLETE
    explained = _explained(profiles, out['base']['candidates'][y]['card_slugs'], out['base']['candidates'][x]['card_slugs'])
    conditions = {'winner_name_incomplete_or_blend_explained': not complete or explained is not None,'winner_name_exclusive_zero': exclusive == 0,
                  'winner_name_advantage': fy['name.read_w'] > fx['name.read_w'],
                  'producer_not_stronger': fy['prod.read_w'] <= fx['prod.read_w'],
                  'agreed_not_more_contradicted': not contra_x}
    trace.update(agreed_candidate=out['evidence'][x]['candidate_id'], agreed_channels=agreed[1],
                 text_winner=out['evidence'][y]['candidate_id'], winner_exclusive_name=exclusive,
                 conditions=conditions, agreed_extra_contradictions=contra_x, winner_name_complete=complete,
                 blend_explanation=explained,
                 name_read_w={'agreed': fx['name.read_w'], 'winner': fy['name.read_w']})
    trace['blocked'] = all(conditions.values())
    trace['reason'] = 'shared_name_text_cannot_override_agreed_visual' if trace['blocked'] else 'override_has_discriminating_evidence'
    return trace


def apply(out, resolver, profiles):
    """Return ``out`` with the guard applied through the unchanged resolver (same path for replay and live)."""
    trace = check(out, profiles)
    out['discriminator_guard'] = trace
    if not trace['blocked']:
        return out
    raw = deepcopy(out['raw_proposal'])
    for row in raw['ranked_candidates']:
        row['learned_rank'] = row['rank']
    raw['ranked_candidates'].sort(key=lambda r: r['candidate_id'] != trace['agreed_candidate'])
    for rank, row in enumerate(raw['ranked_candidates'], 1):
        row['rank'] = rank
    top = raw['ranked_candidates'][0]
    raw.update(candidate_id=top['candidate_id'], product_id=top['product_id'],
               representative_slug=top['representative_slug'], best_candidate=top['representative_slug'],
               representative_source=top['representative_source'], score=top['score'], score_margin=None,
               reason='shared_name_visual_agreement_guard', exact_slug=None, probability=None,
               release_admitted=False, discriminator_guard=trace)
    out['raw_proposal_before_guard'] = out['raw_proposal']
    out['raw_proposal'] = raw
    out['proposal'] = resolver.resolve(out['base'], raw)
    return out
