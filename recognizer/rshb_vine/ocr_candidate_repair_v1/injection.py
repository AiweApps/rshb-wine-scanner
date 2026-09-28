"""Add catalogue cards whose identity phrase is read on the target's own crops to the frozen candidate pool.

The phrase index reuses the systemic evidence's member extraction (producer forms, commercial-name tokens,
typed-atom split), so an injected card is described by exactly the tokens the ranker later scores. A card
enters only when every core name token (commercial name minus producer and typed grape/colour/sugar/style
atoms; all name tokens when the name is only such atoms) and a complete specific producer form are read by a non-fuzzy match on target-bound OCR. A phrase
that fits more products than the budget injects nothing: that is recorded as ambiguity, never forced. A read
that a same-producer blend already in the pool lists as a grape is explained by that card, not a new product.
Injected cards get no visual rank, score or control flag; the frozen ranker and resolver decide.
"""
from rshb_vine.io import digest, read_json, read_jsonl, verify
from rshb_vine.learned_selection_features import _candidate_key
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E

RULE = {
    'version': 'ocr-catalog-phrase-injection-v2',
    'target_sources': ['bottle_context_crop', 'instance_label_crop', 'label_view_crop'],
    'match_kinds': list(E.MATCHED),
    'min_token_skeleton': 3,
    'identity_tokens': 'core name tokens; a name made only of typed atoms (a grape name) uses all its tokens',
    'min_core_name_weight': 0.6,
    'producer': 'every specific token (weight >= 0.5) of at least one producer form is read',
    'min_producer_token_weight': 0.5,
    'budget_products': 3,
    'blend_explanation': 'a same-producer pool card listing all identity tokens as grapes but not in its name explains the read',
    'ownership': 'a core-name line also read on another target of the same scene blocks injection',
    'conflicts': 'a card with a blocking reference conflict from the supplied guard is not injected',
}
GALLERY = 'runs/catalog-training-data-v2/stage5/evaluator/actual-fit-v1/B4343/gallery/gallery.json'
CATALOG = 'data/catalog-additions-20260921/catalog.jsonl'


def eligible_slugs(root, registry):
    """Registry cards that the served B3 gallery references and the catalogue does not exclude from retrieval."""
    gallery = {r['slug'] for r in read_json(root / GALLERY)['references']}
    excluded = {r['slug'] for r in read_jsonl(root / CATALOG) if r.get('excluded_from_retrieval')}
    return sorted((gallery & set(registry.cards)) - excluded)


def _producer_key(card):
    return text.base_norm(card.get('raw_metadata', {}).get('Винодельня', ''))


def _line_key(raw):
    return ' '.join(t.forms[0] for t in text.tokenize(raw))


def scene_line_keys(result, instance_id):
    """Line keys read on the other targets of one result (instance label crops and their re-reads)."""
    sid, keys = str(instance_id), set()
    for packet in (result.get('instance_text') or {}).get('targets', []):
        if str(packet.get('instance_id')) != sid:
            keys.update(_line_key(o.get('raw_text', '')) for o in packet.get('observations', []))
    for read in (result.get('conditional_label_ocr') or {}).get('reads', []):
        if str(read.get('instance_id')) != sid:
            keys.update(_line_key(o.get('raw_text', '')) for o in read.get('observations', []))
    return keys - {''}


class CatalogPhraseIndex:
    def __init__(self, selection, slugs):
        self.registry, self.context, self.stats = selection.registry, selection.context, selection.stats
        self.entries = {}
        for slug in slugs:
            card, profile = self.registry.cards[slug], self.registry.claims.get(slug)
            pseudo = {'card_slugs': [slug], 'provenance': {'cards': {slug: card}, 'claims': {slug: profile}}}
            member = E._members(pseudo, self.context)[0][slug]
            kind, tokens = ('core_name', member['core']) if member['core'] else ('typed_atom_name', member['name'])
            core = [t for t in tokens if len(t.skeletons[0]) >= RULE['min_token_skeleton']]
            weight = sum(self.stats.name_weight(t) for t in core)
            forms = []
            for form in member['producers']:
                tokens = [(t, self.stats.producer_weight(t)) for t in text.tokenize(form)]
                tokens = [(t, w) for t, w in tokens
                          if w >= RULE['min_producer_token_weight'] and len(t.skeletons[0]) >= RULE['min_token_skeleton']]
                if tokens and all(len(tokens) != len(f) or {t.skeletons[0] for t, _ in tokens} != {t.skeletons[0] for t, _ in f}
                                  for f in forms):
                    forms.append(tokens)
            if core and forms and weight >= RULE['min_core_name_weight']:
                self.entries[slug] = {'core': core, 'kind': kind, 'weight': weight, 'producer': forms,
                                      'key': _candidate_key(slug, self.registry.cards),
                                      'product_id': card['product_id'], 'producer_key': _producer_key(card)}
        self.profiles_cache = {}
        self.checksum = digest({'rule': RULE, 'slugs': sorted(self.entries),
                                'phrases': {s: [t.forms[0] for t in e['core']] for s, e in sorted(self.entries.items())}})

    def profile(self, slug):
        """Grape-list and name skeletons of one registry card (any pool member, indexed or not)."""
        if slug not in self.profiles_cache:
            card, profile = self.registry.cards[slug], self.registry.claims.get(slug)
            pseudo = {'card_slugs': [slug], 'provenance': {'cards': {slug: card}, 'claims': {slug: profile}}}
            member = E._members(pseudo, self.context)[0][slug]
            grapes = V4._claim_values(profile, 'grape_blend')[0] + [card.get('raw_metadata', {}).get('Сорт винограда', '')]
            self.profiles_cache[slug] = {'producer_key': _producer_key(card),
                                    'grapes': {t.skeletons[0] for g in grapes for t in text.tokenize(g or '')},
                                    'name': {t.skeletons[0] for t in member['name']}}
        return self.profiles_cache[slug]

    def _blend_explanation(self, entry, pool_slugs):
        identity = {t.skeletons[0] for t in entry['core']}
        return sorted(s for s in pool_slugs
                      if _candidate_key(s, self.registry.cards) != entry['key']
                      and self.profile(s)['producer_key'] == entry['producer_key'] and entry['producer_key']
                      and identity <= self.profile(s)['grapes'] and not identity <= self.profile(s)['name'])

    def describe(self):
        return {'rule': RULE, 'indexed_cards': len(self.entries), 'checksum': self.checksum}

    def propose(self, observations, provenance, pool_slugs, other_line_keys=(), conflicts=None):
        """Injection decision for one target; every match, skip and ambiguity is kept as evidence."""
        pool_keys = {_candidate_key(s, self.registry.cards) for s in pool_slugs}
        bound = [i for i, r in enumerate(provenance) if r is not None and r['crop_source'] in RULE['target_sources']]
        packet = [observations[i] for i in bound]
        lines = V4._lines(packet)
        index = E._Index(lines)
        grades = E._line_grades(lines, [provenance[i] for i in bound])
        matched = {}
        for slug, entry in self.entries.items():
            core = []
            for t in entry['core']:
                strong = {l: k for l, k in index.matches(t.forms[0]).items() if k in E.MATCHED}
                if not strong:
                    break
                core.append({'token': t.forms[0], 'lines': sorted(strong), 'kinds': sorted(set(strong.values()))})
            else:
                producer = None
                for form in entry['producer']:
                    reads = [{'token': t.forms[0], 'weight': round(w, 4),
                              'lines': sorted(l for l, k in index.matches(t.forms[0]).items() if k in E.MATCHED)}
                             for t, w in form]
                    if all(r['lines'] for r in reads):
                        producer = reads
                        break
                if producer:
                    matched[slug] = {'slug': slug, 'candidate_key': entry['key'], 'product_id': entry['product_id'],
                                     'name_kind': entry['kind'], 'core_name_weight': round(entry['weight'], 4),
                                     'core': core, 'producer': producer,
                                     'blend_explained_by': self._blend_explanation(entry, pool_slugs)}
        line_raw = {l['line_id']: {'raw': l['raw'], 'grades': grades[l['line_id']],
                                   'observation_ids': [bound[i] for i in l['observation_ids']]} for l in lines}
        for m in matched.values():
            used = sorted({l for part in (m['core'], m['producer']) for x in part for l in x['lines']})
            m['lines'] = {l: line_raw[l] for l in used}
            m['ownership_uncertain'] = any(_line_key(r) in other_line_keys
                                           for x in m['core'] for l in x['lines'] for r in line_raw[l]['raw'])
            m['conflict'] = conflicts(m['slug']) if conflicts else None
        keys = sorted({m['candidate_key'] for m in matched.values()})
        decision = {'rule': RULE['version'], 'index_checksum': self.checksum, 'bound_observations': len(bound),
                    'total_observations': len(observations), 'matched': sorted(matched.values(), key=lambda m: m['slug']),
                    'matched_products': len(keys), 'injected': [], 'skipped': [], 'status': 'no_phrase_match'}
        if not matched:
            return decision
        if len(keys) > RULE['budget_products']:
            decision['status'] = 'ambiguous_phrase_over_budget'
            decision['skipped'] = [{'slug': m['slug'], 'reason': 'ambiguous_phrase_over_budget'} for m in decision['matched']]
            return decision
        for m in decision['matched']:
            reason = ('already_in_pool' if m['candidate_key'] in pool_keys else
                      'ownership_uncertain' if m['ownership_uncertain'] else
                      'explained_by_pool_blend' if m['blend_explained_by'] else
                      'reference_conflict' if m['conflict'] and m['conflict'].get('blocks_injection') else None)
            if reason:
                decision['skipped'].append({'slug': m['slug'], 'reason': reason})
            else:
                decision['injected'].append(m['slug'])
        decision['status'] = 'injected' if decision['injected'] else 'matched_not_injected'
        return decision
