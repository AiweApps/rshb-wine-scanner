"""Component T: catalogue composite-identity candidate admission on top of the live injection proposer (B).

The frozen phrase index drops a card at build time when its core name (commercial name minus producer and typed
grape/colour/sugar/style atoms) is too common (weight < 0.6): ``Каберне Совиньон Резерв`` of Chateau de Talu keeps only
``rezerv``, a grape-only ``Мерло`` keeps nothing specific. T does not lower that gate. It admits a card when the target's
own crops read its complete identity: every name token (core and typed atoms together) and one complete specific
producer form, at OCR words disjoint from the name reads. Uniqueness comes from the same catalogue claims: siblings are
the cards whose producer tokens contain that form. A card with a more specific sibling (its name tokens a proper subset
of the sibling's, other product) is never added: the read may be a partial read of the sibling (series/line unknown).
Equal-name siblings (other product, same name tokens: style/sugar/vintage unknown) are added together, or none when
one of them has no gallery reference. A name token may be read through an attested bilingual alias group of the
running positive-variant text (``reserve``: резерв/reserve/reserva) and only there; no fuzzy match, no hand rule.

At request time the parent decision is kept verbatim; T tries only cards the parent did not match, keeps the parent
product budget, the ownership/blend/reference-conflict checks and records original, proposed and admitted separately.
Scores, ranker, resolver and guard profiles are untouched; probability and exact slug stay None.
"""
from collections import defaultdict

from rshb_vine.io import digest, read_jsonl
from rshb_vine.learned_selection_features import _candidate_key
from rshb_vine.ocr_candidate_repair_v1 import injection as I
from rshb_vine.positive_variant_text import ALIASES as POSITIVE_ALIASES
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E
from rshb_vine.text_evidence_repair_v1.producer_names import ShortProducerIndex

ALIAS_GROUPS = {'reserve': tuple(POSITIVE_ALIASES['reserve'])}
STYLE_ROLES = ('color', 'sugar', 'style')
CLAIM_ROLES = ('manufacturer', 'visible_brand', 'commercial_name', 'series', 'grape_blend', 'color', 'sugar', 'style')
RULE = {
    'version': 'catalogue-composite-identity-admission-v1',
    'parent_rules': [I.RULE['version'], 'catalogue-short-producer-forms-v1'],
    'name_tokens': 'every token of the card name (core and typed atoms) with skeleton >= min_token_skeleton',
    'min_token_skeleton': I.RULE['min_token_skeleton'],
    'name_weight_gate': 'none; the frozen 0.6 core-name gate is unchanged for the parent and not used by T',
    'name_rejected': 'names made only of colour/sugar/style atoms',
    'producer': 'one frozen producer form (tokens with weight >= 0.5, skeleton >= 3) holding >= 1 specific token '
                '(outside the generic vocabulary, not a typed atom, not a digit); every token read at OCR words '
                'that no name token read uses',
    'reads': list(E.MATCHED) + ['attested_alias'],
    'alias_groups': {k: list(v) for k, v in ALIAS_GROUPS.items()},
    'alias_rule': 'single-token groups only; the card token matches (non-fuzzy) one group form, the OCR token reads '
                  'a group form the card token does not match non-fuzzy (exact/homoglyph/translit of that form)',
    'siblings': 'registry cards (not excluded from retrieval) whose producer tokens contain the form, other candidate key',
    'more_specific_sibling': 'name tokens a proper subset of a sibling name: never added (read or unread sibling)',
    'equal_siblings': 'same name tokens: added together within budget; none when any equal sibling is not gallery-eligible',
    'budget_products': I.RULE['budget_products'],
    'request_time': 'parent decision verbatim; only cards the parent did not match; parent over budget -> nothing; '
                    'parent matched keys plus T keys over budget -> nothing; then already_in_pool, already_injected, '
                    'ownership_uncertain (name lines read on another target), explained_by_pool_blend, reference_conflict',
}


def _skeleton_set(tokens):
    return frozenset(t.skeletons[0] for t in tokens)


def _alias_forms(token):
    """Other forms of the attested group whose form this card token reads, or () when it is in no group."""
    for key, group in ALIAS_GROUPS.items():
        forms = [text.tokenize(a) for a in group]
        if any(len(f) != 1 for f in forms):
            raise ValueError('Alias group %s has a multi-token form' % key)
        forms = [f[0] for f in forms]
        if any(text.token_match(f, token.forms[0]) in E.MATCHED for f in forms):
            return key, tuple(f.forms[0] for f in forms if text.token_match(f, token.forms[0]) not in E.MATCHED)
    return None, ()


class CompositeIdentityIndex:
    """Composite-identity entries over the B proposer's own registry, context, stats and frozen index (read only)."""

    rule = RULE

    def __init__(self, parent, root):
        if type(parent) is not ShortProducerIndex:
            raise ValueError('T wraps the live short-producer proposer (component B) only')
        self.parent, self.frozen = parent, parent.base
        self.registry, self.context, self.stats = parent.registry, parent.context, parent.stats
        self.profile, self.last_trace = self.frozen.profile, None
        self.eligible = frozenset(I.eligible_slugs(root, self.registry))
        if not set(self.frozen.entries) <= self.eligible:
            raise ValueError('Frozen phrase index holds cards outside the eligible gallery slugs')
        excluded = {r['slug'] for r in read_jsonl(root / I.CATALOG) if r.get('excluded_from_retrieval')}
        self.universe = sorted(set(self.registry.cards) - excluded)
        generic = {text.skeleton(g) for g in self.context.generic}
        self.cards, by_token = {}, defaultdict(set)
        for slug in self.universe:
            card = self.cards[slug] = self._card(slug, generic)
            for s in card['producer_tokens']:
                by_token[s].add(slug)
        self.entries, self.census = {}, {}
        for slug in sorted(self.eligible):
            row = self._entry(slug, by_token)
            self.census[slug] = row
            if row['status'] == 'admitted':
                self.entries[slug] = self.cards[slug]
        self.checksum = digest({'rule': RULE, 'parent': parent.checksum, 'entries': {
            s: {'name': [t.forms[0] for t in c['name']], 'forms': [[t.forms[0] for t, _ in f] for f in c['forms']],
                'aliases': c['aliases_trace'], 'more_specific': self.census[s]['more_specific_siblings'],
                'equal': self.census[s]['equal_siblings']} for s, c in sorted(self.entries.items())}})

    def _card(self, slug, generic):
        card, profile = self.registry.cards[slug], self.registry.claims.get(slug)
        pseudo = {'card_slugs': [slug], 'provenance': {'cards': {slug: card}, 'claims': {slug: profile}}}
        member = E._members(pseudo, self.context)[0][slug]
        roles = ['core'] * len(member['name'])
        for role, found in self.context.lexicon.entities(tuple(member['name'])).items():
            for item in found:
                for i in range(item['start'], item['end']):
                    roles[i] = role
        keep = [i for i, t in enumerate(member['name']) if len(t.skeletons[0]) >= RULE['min_token_skeleton']]
        name, name_roles = [member['name'][i] for i in keep], [roles[i] for i in keep]
        forms, rejected_forms = [], []
        for value in member['producers']:
            tokens = [(t, self.stats.producer_weight(t)) for t in text.tokenize(value)]
            tokens = [(t, w) for t, w in tokens
                      if w >= I.RULE['min_producer_token_weight'] and len(t.skeletons[0]) >= I.RULE['min_token_skeleton']]
            if not tokens or any(_skeleton_set(t for t, _ in f) == _skeleton_set(t for t, _ in tokens) for f in forms):
                continue
            specific = [t.forms[0] for t, _ in tokens if t.skeletons[0] not in generic and not t.forms[0].isdigit()
                        and not E._atoms([t], self.context.lexicon)]
            (forms if specific else rejected_forms).append(tokens)
        aliases, trace = {}, {}
        for i, t in enumerate(name):
            key, other = _alias_forms(t)
            if other:
                aliases[i] = other
                trace[t.forms[0]] = {'group': key, 'forms': list(other)}
        return {'slug': slug, 'key': _candidate_key(slug, self.registry.cards), 'product_id': card['product_id'],
                'producer_key': I._producer_key(card), 'name': name, 'name_roles': name_roles,
                'name_set': _skeleton_set(name), 'forms': forms, 'rejected_forms': rejected_forms,
                'producer_tokens': {t.skeletons[0] for f in member['producers'] for t in text.tokenize(f)},
                'aliases': aliases, 'aliases_trace': trace, 'claims': self._claim_status(profile)}

    @staticmethod
    def _claim_status(profile):
        """Per role the registry's own status, unknown flag and comparison permission (they can disagree)."""
        roles = (profile or {}).get('roles', {})
        return {r: ({'status': roles[r]['status'], 'unknown': roles[r]['unknown'],
                     'comparison_allowed': roles[r]['comparison_allowed'], 'claims': len(roles[r]['claims'])}
                    if r in roles else {'status': 'role_absent'})
                for r in CLAIM_ROLES}

    def _entry(self, slug, by_token):
        c = self.cards[slug]
        reasons = []
        if not c['name']:
            reasons.append('no_name_tokens')
        elif all(r in STYLE_ROLES for r in c['name_roles']):
            reasons.append('name_only_colour_sugar_style')
        if not c['forms']:
            reasons.append('no_specific_producer_form' if c['rejected_forms'] else 'no_producer_form')
        siblings = set()
        for form in c['forms']:
            skel = [t.skeletons[0] for t, _ in form]
            siblings |= set.intersection(*(by_token.get(s, set()) for s in skel))
        siblings = sorted(s for s in siblings - {slug} if self.cards[s]['key'] != c['key'])
        more = [s for s in siblings if c['name_set'] < self.cards[s]['name_set']]
        equal = [s for s in siblings if c['name_set'] == self.cards[s]['name_set']]
        less = [s for s in siblings if self.cards[s]['name_set'] < c['name_set']]
        weight = sum(self.stats.name_weight(t) for t, r in zip(c['name'], c['name_roles']) if r == 'core') \
            if any(r == 'core' for r in c['name_roles']) else sum(self.stats.name_weight(t) for t in c['name'])
        status = 'rejected' if reasons else 'admitted'
        blocked = ('more_specific_sibling' if more else
                   'equal_sibling_not_eligible' if any(s not in self.eligible for s in equal) else None)
        return {'slug': slug, 'status': status, 'reasons': reasons, 'frozen_indexed': slug in self.frozen.entries,
                'frozen_gate_weight': round(weight, 4), 'name': [t.forms[0] for t in c['name']],
                'name_roles': c['name_roles'], 'producer_forms': [[t.forms[0] for t, _ in f] for f in c['forms']],
                'rejected_producer_forms': [[t.forms[0] for t, _ in f] for f in c['rejected_forms']],
                'aliases': c['aliases_trace'], 'more_specific_siblings': more, 'equal_siblings': equal,
                'equal_siblings_not_eligible': [s for s in equal if s not in self.eligible],
                'less_specific_siblings': less, 'never_added': blocked if status == 'admitted' else None,
                'claims': c['claims']}

    def describe(self):
        admitted = [r for r in self.census.values() if r['status'] == 'admitted']
        return dict(self.parent.describe(), composite_rule=RULE['version'], composite_checksum=self.checksum,
                    parent_proposer_checksum=self.parent.checksum, composite_entries=len(self.entries),
                    composite_addable=sum(1 for r in admitted if not r['never_added']), checksum=self.checksum)

    # ---- request time ----

    def _read_name(self, index, entry):
        """Per name token: non-fuzzy places, or places of an attested alias form; None when a token is unread."""
        reads, places = [], set()
        for i, t in enumerate(entry['name']):
            found, kind = ShortProducerIndex._places(index, t.forms[0]), None
            if found:
                kinds = sorted({text.token_match(index.lines[l]['tokens'][p], t.forms[0]) for l, p in found})
            else:
                for form in entry['aliases'].get(i, ()):
                    found |= ShortProducerIndex._places(index, form)
                kinds, kind = ['attested_alias'], 'attested_alias'
            if not found:
                return None, None
            places |= found
            reads.append({'token': t.forms[0], 'role': entry['name_roles'][i], 'lines': sorted({l for l, _ in found}),
                          'kinds': kinds, 'alias_forms': list(entry['aliases'][i]) if kind else None})
        return reads, places

    def _read_producer(self, index, entry, name_places):
        for form in entry['forms']:
            reads = []
            for t, w in form:
                places = ShortProducerIndex._places(index, t.forms[0]) - name_places
                if not places:
                    break
                reads.append({'token': t.forms[0], 'weight': round(w, 4), 'lines': sorted({l for l, _ in places})})
            else:
                return reads
        return None

    def propose(self, observations, provenance, pool_slugs, other_line_keys=(), conflicts=None):
        decision = self.parent.propose(observations, provenance, pool_slugs, other_line_keys, conflicts)
        trace = {'rule': RULE['version'], 'checksum': self.checksum,
                 'original': {k: decision.get(k) for k in ('status', 'injected', 'matched_products')}
                 | {'matched': [m['slug'] for m in decision['matched']], 'skipped': list(decision['skipped'])},
                 'proposed': [], 'admitted': [], 'skipped': [], 'status': 'no_composite_match', 'applied': False,
                 'counts': {'proposed': 0, 'admitted': 0, 'skipped': 0}, 'skip_reasons': []}
        decision['composite_identity'] = self.last_trace = trace
        short = decision.get('short_producer') or {}
        if decision['status'] == 'ambiguous_phrase_over_budget' or short.get('status') == 'short_form_over_budget':
            trace['status'] = 'parent_over_budget'
            return decision
        done = {m['slug'] for m in decision['matched']}
        bound = [i for i, r in enumerate(provenance) if r is not None and r['crop_source'] in I.RULE['target_sources']]
        lines = V4._lines([observations[i] for i in bound])
        index = E._Index(lines)
        found = {}
        for slug, entry in self.entries.items():
            if slug in done:
                continue
            name, places = self._read_name(index, entry)
            if name is None:
                continue
            producer = self._read_producer(index, entry, places)
            if producer is None:
                continue
            row = self.census[slug]
            found[slug] = {'slug': slug, 'candidate_key': entry['key'], 'product_id': entry['product_id'],
                           'name_kind': 'composite_identity', 'core_name_weight': row['frozen_gate_weight'],
                           'core': [{'token': r['token'], 'lines': r['lines'], 'kinds': r['kinds']} for r in name],
                           'name_reads': name, 'producer': producer, 'producer_form_kind': 'composite',
                           'frozen_indexed': row['frozen_indexed'],
                           'blend_explained_by': self.frozen._blend_explanation(
                               {'core': entry['name'], 'key': entry['key'], 'producer_key': entry['producer_key']},
                               pool_slugs)}
        if not found:
            return decision
        grades = E._line_grades(lines, [provenance[i] for i in bound])
        line_raw = {l['line_id']: {'raw': l['raw'], 'grades': grades[l['line_id']],
                                   'observation_ids': [bound[i] for i in l['observation_ids']]} for l in lines}
        for m in found.values():
            used = sorted({l for part in (m['core'], m['producer']) for x in part for l in x['lines']})
            m['lines'] = {l: line_raw[l] for l in used}
            m['ownership_uncertain'] = any(I._line_key(r) in other_line_keys
                                           for x in m['core'] for l in x['lines'] for r in line_raw[l]['raw'])
            m['conflict'] = conflicts(m['slug']) if conflicts else None
        trace['proposed'] = sorted(found)
        live, skipped = {}, []
        for slug in sorted(found):
            row = self.census[slug]
            if row['more_specific_siblings']:
                read = [s for s in row['more_specific_siblings'] if s in found or s in done
                        or self._sibling_read(index, s)]
                skipped.append({'slug': slug, 'reason': 'more_specific_sibling_read' if read
                                else 'more_specific_sibling_unread', 'siblings': row['more_specific_siblings'],
                                'read_siblings': read})
            elif row['equal_siblings_not_eligible']:
                skipped.append({'slug': slug, 'reason': 'equal_sibling_not_eligible',
                                'siblings': row['equal_siblings_not_eligible']})
            else:
                live[slug] = found[slug]
        parent_keys = {m['candidate_key'] for m in decision['matched']}
        keys = parent_keys | {m['candidate_key'] for m in live.values()}
        if live and len(keys) > I.RULE['budget_products']:
            skipped += [{'slug': s, 'reason': 'composite_over_budget'} for s in sorted(live)]
            live = {}
        pool_keys = {_candidate_key(s, self.registry.cards) for s in pool_slugs}
        injected_keys = {_candidate_key(s, self.registry.cards) for s in decision['injected']}
        for slug in sorted(live):
            m = live[slug]
            reason = ('already_in_pool' if m['candidate_key'] in pool_keys else
                      'already_injected' if m['candidate_key'] in injected_keys else
                      'ownership_uncertain' if m['ownership_uncertain'] else
                      'explained_by_pool_blend' if m['blend_explained_by'] else
                      'reference_conflict' if m['conflict'] and m['conflict'].get('blocks_injection') else None)
            if reason:
                skipped.append({'slug': slug, 'reason': reason})
            else:
                trace['admitted'].append(slug)
        trace['skipped'] = sorted(skipped, key=lambda s: s['slug'])
        trace['status'] = 'injected' if trace['admitted'] else 'matched_not_injected'
        trace['applied'] = bool(trace['admitted'])
        trace['counts'] = {'proposed': len(trace['proposed']), 'admitted': len(trace['admitted']),
                           'skipped': len(trace['skipped'])}
        trace['skip_reasons'] = sorted({s['reason'] for s in trace['skipped']})
        decision['matched'] = [*decision['matched'], *(found[s] for s in sorted(found))]
        decision['matched_products'] = len(parent_keys | {m['candidate_key'] for m in found.values()})
        decision['injected'] = [*decision['injected'], *trace['admitted']]
        decision['skipped'] = [*decision['skipped'], *({'slug': s['slug'], 'reason': s['reason']} for s in trace['skipped'])]
        decision['status'] = 'injected' if decision['injected'] else 'matched_not_injected'
        return decision

    def _sibling_read(self, index, slug):
        sibling = self.cards[slug]
        return all(ShortProducerIndex._places(index, t.forms[0])
                   or any(ShortProducerIndex._places(index, f) for f in sibling['aliases'].get(i, ()))
                   for i, t in enumerate(sibling['name']))
