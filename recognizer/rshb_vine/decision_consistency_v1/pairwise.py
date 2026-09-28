"""Component G: the shared-name guard judges name exclusivity against the agreed visual candidate, not the whole pool.

Frozen guard v2 blocks a text winner when, among other conditions, none of its read name tokens is absent from the
names of every rival product in the pool. A varietal word (Рислинг) is then "shared" because unrelated producers'
Riesling cards are in the pool, although the agreed visual candidate (Muskat) does not claim it. G recomputes only
that condition, pairwise: a read (non-fuzzy) winner name token counts when the agreed candidate claims it nowhere
(full name incl. generic words, producer forms, series, grape values; catalogue normal form, script-independent).

Safeguards kept:
- colour/sugar/style atoms never count (typed roles and their verified contradictions own them); years are dropped;
- the catalogue generic list stays excluded from name tokens (frozen member build);
- a grape blend-listed by the agreed card is claimed by it; a grape absent from an agreed grape list is neutral
  (lists are open), unless every agreed card is a closed monosort whose single non-disputed grape key is also named
  in its own title AND every member card of the winner and of the agreed candidate carries one and the same non-empty
  catalogue producer key (Винодельня, base_norm); a grape word alone never moves the answer to another or an unknown
  producer; it is never a contradiction feature and never lowers any score; other name tokens keep the generic rule;
- disputed name claims (conflict/unresolved/ambiguous) or a shared product keep the frozen value;
- the other four guard conditions, including the incomplete-reading one, are the frozen ones.

Post_guard stage, first, release-only: the frozen evidence and the frozen ``discriminator_guard`` trace are never
modified. The frozen D.check runs on the pre-guard state with pairwise exclusivity in a copy. Only a frozen block can
change: when the pairwise decision does not block, the pre-guard proposal is restored through the resolver. A
pairwise block where the frozen guard allowed the proposal is traced (``pairwise_would_block``) and not applied.
``effective`` names which guard decided the published proposal.
"""
import time

from rshb_vine.ocr_candidate_repair_v1 import discriminator as D
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E

STAGE = 'decision-consistency-pairwise-guard-exclusivity-v1'
DISPUTED = ('conflict', 'unresolved', 'ambiguous')
TYPED_NEUTRAL = ('color', 'sugar', 'style')
RULE = {
    'version': 'pairwise-agreed-exclusivity-v1',
    'replaces': 'guard v2 condition winner_name_exclusive_zero: pool-global rival name tokens -> agreed candidate claims',
    'counted': 'read non-fuzzy winner name token absent from every agreed claim (name incl. generic, producer forms, '
               'series, grape values) by catalogue skeleton',
    'neutral': ['colour/sugar/style atoms', 'grape absent from an open agreed grape list', 'unread or fuzzy-only'],
    'grape_counts_only': 'every agreed card: exactly one non-disputed grape key, named in its own title, differing; '
                         'and all winner and agreed member cards share one non-empty catalogue producer key',
    'producer_relation': 'same = one non-empty key over all member cards of both; different = all keys non-empty and '
                         'disjoint; unknown = otherwise (empty key or partial overlap)',
    'kept': 'other four guard conditions, frozen evidence, frozen guard trace, weights, ranker, resolver',
    'not_applied': 'disputed name claim of an agreed card, winner and agreed share a product',
    'scope': 'release-only: reconsiders frozen blocks; never adds a block',
}


def _skeletons(values):
    return {s for v in values for t in text.tokenize(v) for s in t.skeletons}


def _roles(tokens, lexicon):
    roles = {}
    for role, found in lexicon.entities(tuple(tokens)).items():
        for item in found:
            for i in range(item['start'], item['end']):
                roles[i] = (role, item['key'])
    return roles


def agreed_claims(candidate, members, lexicon):
    """Everything the agreed candidate claims, per card, in catalogue skeleton space."""
    cards, profiles = candidate['provenance']['cards'], candidate['provenance']['claims']
    out = {}
    for slug, m in members.items():
        card, profile = cards[slug], profiles.get(slug)
        names, status, _ = V4._claim_values(profile, 'commercial_name')
        raw = card.get('raw_metadata', {})
        titles = [*names, raw.get('Название вина', '')]
        grapes = [*V4._claim_values(profile, 'grape_blend')[0], raw.get('Сорт винограда', '')]
        grape = m['typed']['grape_blend']
        keys = sorted({k['key'] for k in grape['keys']})
        title_keys = sorted({key for v in titles for role, key in _roles(text.tokenize(v), lexicon).values()
                             if role == 'grape_blend'})
        out[slug] = {'name_status': status, 'grape_status': grape['status'], 'grape_keys': keys,
                     'title_grape_keys': title_keys,
                     'closed_monosort': len(keys) == 1 and title_keys == keys and grape['status'] not in DISPUTED,
                     'skeletons': {'name': _skeletons(titles), 'producer': _skeletons(m['producers']),
                                   'series': _skeletons(V4._claim_values(profile, 'series')[0]),
                                   'grape': _skeletons(grapes)}}
    return out


def producer_relation(cx, cy):
    """Catalogue producer relation of the agreed (cx) and winner (cy) candidates over all member cards."""
    def keys(c):
        cards = c['provenance']['cards']
        return {s: text.base_norm(cards[s].get('raw_metadata', {}).get('Винодельня', '')) for s in c['card_slugs']}
    agreed, winner = keys(cx), keys(cy)
    a, w = set(agreed.values()), set(winner.values())
    if '' in a | w:
        relation = 'unknown'
    elif len(a | w) == 1:
        relation = 'same'
    elif not a & w:
        relation = 'different'
    else:
        relation = 'unknown'
    return {'relation': relation, 'agreed_keys': agreed, 'winner_keys': winner}


def _claimed_by(token, claims):
    where = sorted({src for c in claims.values() for src, sk in c['skeletons'].items() if set(token.skeletons) & sk})
    return where or None


def _rival_claims(out, context, product_of, y):
    """Frozen rival name tokens of the winner (E.build semantics) and each rival's products."""
    products = [{product_of(s) for s in c['card_slugs']} for c in out['base']['candidates']]
    claimed = {}
    for i, c in enumerate(out['base']['candidates']):
        if i != y and not products[i] & products[y]:
            claimed[i] = {t.skeletons[0] for m in E._members(c, context)[0].values() for t in m['name']}
    return claimed, products


def pairwise_evidence(out, x, y, context, product_of):
    """Winner name entries with pairwise exclusivity (a copy) and the per-token trace; frozen entries untouched."""
    lexicon = context.lexicon
    cx, cy = out['base']['candidates'][x], out['base']['candidates'][y]
    mx, my = E._members(cx, context)[0], E._members(cy, context)[0]
    claims = agreed_claims(cx, mx, lexicon)
    rivals, products = _rival_claims(out, context, product_of, y)
    rival_tokens = set().union(*rivals.values()) if rivals else set()
    producers = producer_relation(cx, cy)
    trace = {'agreed_claims': {s: {k: v for k, v in c.items() if k != 'skeletons'} for s, c in claims.items()},
             'producer_relation': producers, 'names': []}
    if products[x] & products[y]:
        trace['not_applied'] = 'winner_and_agreed_share_product'
        return None, trace
    disputed = sorted(s for s, c in claims.items() if c['name_status'] in DISPUTED)
    if disputed:
        trace['not_applied'] = 'agreed_name_claim_disputed:' + ','.join(disputed)
        return None, trace
    closed = all(c['closed_monosort'] for c in claims.values())
    agreed_keys = {k for c in claims.values() for k in c['grape_keys']}
    frozen = out['evidence'][y]['name']
    if [n['slug'] for n in frozen] != list(my):
        raise RuntimeError('winner name evidence is not aligned with its members')
    names = []
    for n, (slug, m) in zip(frozen, my.items()):
        if [r['token'] for r in n['tokens']] != [t.forms[0] for t in m['name']]:
            raise RuntimeError('winner name tokens are not aligned with the frozen evidence: ' + slug)
        status = V4._claim_values(cy['provenance']['claims'].get(slug), 'commercial_name')[1]
        roles = _roles(m['name'], lexicon)
        tokens, pool_check, pairwise = [], 0., 0.
        for i, (t, r) in enumerate(zip(m['name'], n['tokens'])):
            read = any(r['grades'].values())
            role, key = roles.get(i, (None, None))
            where = _claimed_by(t, claims) if read else None
            if not read:
                verdict = 'fuzzy_only_neutral' if r['fuzzy_only'] else 'unread'
            elif where:
                verdict = 'claimed_by_agreed:' + '+'.join(where)
            elif role in TYPED_NEUTRAL:
                verdict = 'typed_role_neutral:' + role
            elif role == 'grape_blend':
                if not closed or key in agreed_keys:
                    verdict = 'grape_absent_from_open_agreed_list_neutral'
                elif producers['relation'] == 'same':
                    verdict = 'counted_grape_vs_closed_monosort'
                else:
                    verdict = 'grape_vs_closed_monosort_producer_%s_neutral' % producers['relation']
            else:
                verdict = 'counted'
            pool_exclusive = read and t.skeletons[0] not in rival_tokens
            pool_check += r['weight'] * pool_exclusive
            if verdict.startswith('counted'):
                pairwise += r['weight']
            tokens.append({'token': r['token'], 'weight': r['weight'], 'read': read, 'role': role, 'key': key,
                           'verdict': verdict, 'pool_exclusive': pool_exclusive,
                           'pool_rival_products_claiming': len({p for o, sk in rivals.items() if t.skeletons[0] in sk
                                                                for p in products[o]})})
        if abs(pool_check - n['exclusive']) > 1e-9:
            raise RuntimeError('pool exclusivity recomputation differs from the frozen evidence: ' + slug)
        applied = status not in DISPUTED
        names.append(dict(n, exclusive=pairwise if applied else n['exclusive']))
        trace['names'].append({'slug': slug, 'name_status': status, 'applied': applied,
                               'exclusive_pool': n['exclusive'], 'exclusive_pairwise': pairwise, 'tokens': tokens})
    trace['agreed_closed_monosort'] = closed
    evidence = list(out['evidence'])
    evidence[y] = dict(out['evidence'][y], name=names)
    return evidence, trace


def classify(pool, pair, trace):
    """Change class of the pairwise decision; new_block_* are diagnostic only (release-only stage)."""
    verdicts = [t for n in trace['names'] for t in n['tokens']]
    counted = [t for t in verdicts if t['verdict'].startswith('counted')]
    lost = [t for t in verdicts if t['pool_exclusive'] and not t['verdict'].startswith('counted')]
    if pool['blocked'] and not pair['blocked']:
        if any(t['verdict'] == 'counted_grape_vs_closed_monosort' for t in counted):
            return 'released_closed_monosort_grape'
        return 'released_common_name' if any(t['pool_rival_products_claiming'] for t in counted) else 'released_other'
    if pair['blocked'] and not pool['blocked']:
        if any(t['verdict'].startswith('grape_vs_closed_monosort_producer_') for t in lost):
            return 'new_block_monosort_grape_producer_not_same'
        if any(t['verdict'] in ('grape_absent_from_open_agreed_list_neutral', 'claimed_by_agreed:grape') for t in lost):
            return 'new_block_open_or_blend_grape'
        if any(t['verdict'].startswith('claimed_by_agreed') for t in lost):
            return 'new_block_claimed_by_agreed'
        return 'new_block_typed_role' if any(t['verdict'].startswith('typed') for t in lost) else 'new_block_other'
    return 'same_decision'


def _guard_view(trace):
    return {k: trace.get(k) for k in ('blocked', 'reason', 'conditions', 'winner_exclusive_name')}


class PairwiseGuardStage:
    trace_key = 'pairwise_guard'
    position = 'post_guard'

    def __init__(self):
        self.identity = {'stage': STAGE, 'rule': RULE, 'guard': D.RULE['version'],
                         'position': 'first post_guard, before layout geometry'}
        self.last_seconds = None

    def apply(self, selection, out, call, request):
        started = time.perf_counter()
        trace = {'stage': STAGE, 'applied': False}
        out[self.trace_key] = trace
        try:
            return self._apply(selection, out, trace)
        finally:
            self.last_seconds = time.perf_counter() - started

    def _apply(self, selection, out, trace):
        frozen = out.get('discriminator_guard') or {}
        raw_pre = out.get('raw_proposal_before_guard') if frozen.get('blocked') else out.get('raw_proposal')
        if not raw_pre or not raw_pre['ranked_candidates']:
            trace['reason'] = 'empty_pool'
            return out
        agreed = D.agreed_visual(out['evidence'])
        y = raw_pre['ranked_candidates'][0]['input_pool_index']
        if agreed is None or agreed[0] == y:
            trace['reason'] = 'no_agreed_visual' if agreed is None else 'winner_is_agreed_visual'
            return out
        profiles, resolver = selection.index.profile, selection.legacy.resolver
        pre = dict(out, raw_proposal=raw_pre)
        pool = D.check(pre, profiles)
        if _guard_view(pool) != _guard_view(frozen):
            raise RuntimeError('frozen guard recomputation differs from the published guard trace')
        evidence, detail = pairwise_evidence(pre, agreed[0], y, selection.inner.context, selection.inner.identity.product_id)
        trace.update(agreed_candidate=pool['agreed_candidate'], text_winner=pool['text_winner'], pairwise=detail)
        if evidence is None:
            trace['reason'] = detail['not_applied']
            trace['effective'] = {'blocked': pool['blocked'], 'decided_by': 'frozen_guard_v2'}
            return out
        pair = D.check(dict(pre, evidence=evidence), profiles)
        trace.update(guard_pool=_guard_view(pool), guard_pairwise=_guard_view(pair),
                     changed_conditions=sorted(k for k, v in pair['conditions'].items() if pool['conditions'][k] != v))
        trace['class'] = classify(pool, pair, detail)
        if not pool['blocked']:
            trace['reason'] = 'frozen_guard_allowed_release_only' if pair['blocked'] else 'same_guard_decision'
            trace['effective'] = {'blocked': False, 'decided_by': 'frozen_guard_v2', 'pairwise_would_block': pair['blocked']}
            return out
        if pair['blocked']:
            trace['reason'] = 'same_guard_decision'
            trace['effective'] = {'blocked': True, 'decided_by': 'frozen_guard_v2_confirmed_pairwise'}
            return out
        out['raw_proposal_before_pairwise'], out['proposal_before_pairwise'] = out['raw_proposal'], out['proposal']
        out['raw_proposal'] = raw_pre
        out['proposal'] = resolver.resolve(out['base'], raw_pre)
        trace['reason'] = 'pairwise_released_frozen_block'
        trace['effective'] = {'blocked': False, 'decided_by': 'pairwise_guard_v1',
                              'frozen_guard_blocked': True, 'frozen_guard_trace': 'discriminator_guard (unchanged)'}
        trace.update(applied=out['proposal']['representative_slug'] != out['proposal_before_pairwise']['representative_slug'],
                     selected={'before': out['proposal_before_pairwise']['representative_slug'],
                               'after': out['proposal']['representative_slug']})
        return out
