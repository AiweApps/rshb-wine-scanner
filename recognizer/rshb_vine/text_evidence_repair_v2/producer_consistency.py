"""Component D: the catalogue-attested short producer forms of table 22a4 as producer forms of the scoring path.

Component B lets an admitted short form (Эльбузд) inject its card, but every scoring consumer reads producer forms
from ``system_selection_v4.features._producer_forms`` (manufacturer / visible_brand claims, verified brand aliases of
config/producer-brand-aliases.json, raw Винодельня). The injected card then reads its producer only as a partial legal
form: prod.read_w is credited, prod.phrase (v4 txt.producer.full_any, one whole form read contiguously) stays 0 and
producer-role-v2 finds no complete form. Verified brand aliases (BURNIER, SIKORY, LORIOT) already are whole forms of
that path; D adds the attested short forms as forms of the same kind, for every card of their producer.

AliasContext        per-instance read-only view of the frozen SelectionContext: producer_aliases = verified aliases
                    united with the admitted short forms, keyed by the producer key (base_norm Винодельня); every other
                    attribute is the frozen one; the frozen dict and sets are never mutated. Catalogue specificities
                    recomputed over the view must equal the frozen CatalogStats, otherwise the view is refused.
AliasProducerStage  in place of the repair-v2 producer_role stage (same trace key, position and producer_role_v2 pin):
                    1. v4 and systemic evidence rebuilt over the view from the same base, provenance and packet; every
                       systemic feature except prod.phrase must equal the frozen rows (else RuntimeError);
                    2. rows take prod.phrase of that alias-aware build (the initial E/v4 phrase);
                    3. frozen producer_role_v2.selection.apply with ProducerRoles over the view and FULL flags (origin,
                       ownership, anchor, complete non-fuzzy eligible reading): the phrase stays only with a complete
                       reading of one form, legal or alias, so the initial phrase and the stage read one form set;
                    4. the final rows must equal the frozen producer stage (frozen ProducerRoles, FULL, over the same
                       rows) in every feature except prod.phrase; every final phrase difference is traced;
                    5. one frozen ranker + resolver re-rank when the final rows differ from the pre-stage rows.
Injection, evidence, the v4 output, guard profile, weights, ranker, resolver and geometry stay frozen. The trace keeps,
for every candidate whose alias v4 phrase or final phrase differs, the pre-stage / alias-v4 / frozen-stage / final
phrase, the frozen and alias v4 reads and complete forms with their kind (catalogue_form | verified_alias |
attested_alias with its table rows); ``revoked`` lists alias v4 phrases the producer stage did not prove.
"""
import time
from types import MappingProxyType

from rshb_vine.io import digest, sha256
from rshb_vine.learned_selection_features import _candidate_key
from rshb_vine.ocr_candidate_repair_v1.selection import _relabel
from rshb_vine.producer_role_v2 import roles as R
from rshb_vine.producer_role_v2 import selection as S
from rshb_vine.recognition_repair_v2 import producer as RP
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E
from rshb_vine.systemic_ranking_v2.selection import feature_digest
from rshb_vine.text_evidence_repair_v1 import producer_names as P

STAGE = 'catalogue-attested-producer-aliases-v1'
PACKAGE = 'rshb_vine/text_evidence_repair_v2'
SOURCES = (f'{PACKAGE}/__init__.py', f'{PACKAGE}/producer_consistency.py',
           'rshb_vine/text_evidence_repair_v1/producer_names.py', P.TABLE)
PHRASE = 'prod.phrase'
RULE = {
    'version': STAGE,
    'table_rule': P.RULE['version'],
    'forms': 'each admitted short form of the pinned table (short tokens joined) is a producer form of every card whose '
             'producer key (base_norm Винодельня) is the row producer key; not added when a verified alias of that key '
             'has the same skeletons',
    'consumers': 'v4 producer phrase (txt.producer.full_any -> prod.phrase), systemic producer evidence and '
                 'producer_role_v2 coverage / completeness read the same form set',
    'invariant': 'catalogue specificities and every card name / core / atom / series token list and typed claim '
                 'over the view equal the frozen ones; the alias-aware build changes no systemic feature except '
                 'prod.phrase',
    'phrase': 'kept by producer_role_v2 only with a complete non-fuzzy eligible reading of one form (FULL flags)',
    'baseline': 'final rows equal the frozen producer stage (frozen ProducerRoles, FULL) except prod.phrase; every '
                'final phrase difference is traced, including a v4 phrase the frozen stage would revoke',
    'rerank': 'one frozen ranker + resolver re-rank when the final rows differ from the pre-stage rows; the pre-stage '
              'proposal is kept when they are equal',
    'unchanged': 'injection, evidence, v4 output, guard profile, weights, ranker, resolver, geometry, raw OCR',
}


def _skeletons(form):
    return sorted(t.skeletons[0] for t in text.tokenize(form))


def attested_aliases(table, registry, frozen):
    """({producer key: {alias form: [table rows]}}, [forms a verified alias already covers])."""
    keys = {P._producer_key(card, slug) for slug, card in registry.cards.items()}
    aliases, covered = {}, set()
    for row in table['admitted']:
        key, form = row['producer_key'], ' '.join(row['short_tokens'])
        if key.startswith('card:') or key not in keys:
            raise ValueError('Attested alias of no catalogue producer key: ' + key)
        if _skeletons(form) != row['skeletons']:
            raise ValueError('Attested alias does not re-tokenize to its skeletons: ' + form)
        if any(_skeletons(v) == row['skeletons'] for v in frozen.producer_aliases.get(key, ())):
            covered.add((key, form))
            continue
        aliases.setdefault(key, {}).setdefault(form, []).append(
            {k: row[k] for k in ('source_form', 'short_tokens', 'reference_support_images', 'cards')})
    return aliases, [{'producer_key': k, 'form': f} for k, f in sorted(covered)]


def name_token_parity(registry, frozen, view):
    """Every card keeps its name, core, atom and series tokens and typed claims under the view; its forms only grow."""
    cards, grown = 0, 0
    for slug, card in registry.cards.items():
        pseudo = {'card_slugs': [slug], 'provenance': {'cards': {slug: card}, 'claims': {slug: registry.claims.get(slug)}}}
        (a,), (b,) = E._members(pseudo, frozen)[0].values(), E._members(pseudo, view)[0].values()
        for k in ('name', 'core', 'atoms'):
            if [t.forms for t in a[k]] != [t.forms for t in b[k]]:
                raise ValueError('Attested aliases change %s tokens of %s' % (k, slug))
        if [[t.forms for t in s] for s in a['series']] != [[t.forms for t in s] for s in b['series']] \
                or a['typed'] != b['typed']:
            raise ValueError('Attested aliases change series tokens or typed claims of ' + slug)
        if not set(a['producers']) <= set(b['producers']):
            raise ValueError('Attested aliases drop a producer form of ' + slug)
        cards += 1
        grown += set(b['producers']) != set(a['producers'])
    return {'cards_checked': cards, 'cards_with_added_forms': grown}


class AliasContext:
    """The frozen SelectionContext with attested aliases added to a new, read-only producer_aliases mapping."""

    def __init__(self, frozen, aliases):
        merged = {k: frozenset(v) for k, v in frozen.producer_aliases.items()}
        for key, forms in aliases.items():
            merged[key] = merged.get(key, frozenset()) | frozenset(forms)
        self.frozen = frozen
        self.producer_aliases = MappingProxyType(merged)

    def __getattr__(self, name):
        return getattr(self.frozen, name)


class AliasProducerStage:
    trace_key = 'producer_role'
    position = 'pre_guard'

    def __init__(self, root, inner, table_checksum):
        RP.check_pin(root)
        table = P.load_table(root, inner.context, inner.registry, table_checksum)
        self.inner, self.frozen = inner, inner.context
        self.aliases, self.covered = attested_aliases(table, inner.registry, inner.context)
        self.context = AliasContext(inner.context, {k: set(v) for k, v in self.aliases.items()})
        view = E.CatalogStats(inner.registry, self.context)
        if view.producer != inner.stats.producer or view.name != inner.stats.name:
            raise ValueError('Attested aliases change catalogue specificities')
        self.name_tokens = name_token_parity(inner.registry, inner.context, self.context)
        self.roles = R.ProducerRoles(self.context, inner.stats)
        self.frozen_roles = R.ProducerRoles(inner.context, inner.stats)
        self.last_seconds = None
        self.checksum = digest({'rule': RULE, 'table': table['checksum'],
                                'aliases': {k: sorted(v) for k, v in sorted(self.aliases.items())}})
        self.identity = {'stage': STAGE, 'rule': STAGE, 'base_rule': R.RULE['version'], 'flags': dict(R.FULL),
                         'pin_checksum': RP.PIN_CHECKSUM, 'table': P.TABLE, 'table_checksum': table['checksum'],
                         'alias_checksum': self.checksum,
                         'producers_with_aliases': len(self.aliases), 'covered_by_verified_alias': self.covered,
                         'name_token_parity': self.name_tokens,
                         'sources_sha256': {p: sha256(root / p) for p in (*SOURCES, *RP.SOURCES)}}

    def form_kind(self, candidate, form):
        cards, claims = candidate['provenance']['cards'], candidate['provenance']['claims']
        for slug in candidate['card_slugs']:
            if form in V4._producer_forms(cards[slug], claims.get(slug), self.frozen):
                key = text.base_norm(cards[slug].get('raw_metadata', {}).get('Винодельня', ''))
                return {'kind': 'verified_alias' if form in self.frozen.producer_aliases.get(key, ()) else 'catalogue_form'}
        for slug in candidate['card_slugs']:
            rows = self.aliases.get(P._producer_key(cards[slug], slug), {}).get(form)
            if rows:
                return {'kind': 'attested_alias', 'table_rows': rows}
        raise RuntimeError('Producer form of no known kind: ' + form)

    def apply(self, selection, out, call, request):
        started = time.perf_counter()
        try:
            return self._apply(selection, out, call)
        finally:
            self.last_seconds = time.perf_counter() - started

    def _apply(self, selection, out, call):
        inner = self.inner
        if selection.inner is not inner:
            raise RuntimeError('Alias stage built for another selector')
        control_slug, ids, base = call['control_slug'], out['candidate_ids'], out['base']
        injected = {_candidate_key(s, selection.registry.cards) for s in out['ocr_candidate_injection']['injected']}

        def rescore(rows):
            raw = inner._propose(base, out['v4'], rows, control_slug)
            _relabel(raw, injected)
            return raw, selection.legacy.resolver.resolve(base, raw)

        v4 = V4.build_features(base, call['raw_visual'], self.context, inner.registry.cards, inner.registry.claims)
        built, _, _ = E.build(base, v4, call['provenance'], self.context, inner.stats, control_slug,
                              inner.identity.product_id)
        if [c['candidate_id'] for c in v4['candidates']] != ids:
            raise RuntimeError('Alias-aware v4 pool differs')
        frozen, frozen_digest = out['rows'], out['feature_digest']
        for cid, a, b in zip(ids, built, frozen):
            other = [n for n in a if n != PHRASE and a[n] != b[n]]
            if other:
                raise RuntimeError('Alias-aware build changed %s of %s' % (other, cid))
        target = R.TargetProducerText(base['query_evidence']['observations'], call['provenance'],
                                      call.get('other_line_keys', ()), R.FULL)
        baseline = [dict(row, **f) for (f, _), row in zip(self.frozen_roles.features(out, target, R.FULL), frozen)]
        phrase_rows = [dict(b, **{PHRASE: a[PHRASE]}) for a, b in zip(built, frozen)]
        added = [i for i, (a, b) in enumerate(zip(phrase_rows, frozen)) if a[PHRASE] != b[PHRASE]]
        before = (out.get('raw_proposal'), out.get('proposal'))
        if added:
            out['rows'], out['feature_digest'] = phrase_rows, feature_digest(phrase_rows, ids)
        trace = S.apply(out, self.roles, call['provenance'], call.get('other_line_keys', ()), rescore, R.FULL)
        final = out['rows']
        for cid, a, b in zip(ids, final, baseline):
            other = [n for n in a if n != PHRASE and a[n] != b[n]]
            if other:
                raise RuntimeError('Alias stage changed %s of %s against the frozen producer stage' % (other, cid))
        deltas = [i for i, (a, b) in enumerate(zip(final, baseline)) if a[PHRASE] != b[PHRASE]]
        alias = {'stage': STAGE, 'alias_checksum': self.checksum, 'applied': bool(deltas),
                 'alias_v4_added': [ids[i] for i in added], 'revoked': [ids[i] for i in added if not final[i][PHRASE]],
                 'phrase_changes': [], 'extra_rerank': False}
        if added:
            out['rows_alias_phrase'] = phrase_rows
            out['rows_before_producer_role'], out['feature_digest_before_producer_role'] = frozen, frozen_digest
            if final == frozen:
                out['raw_proposal'], out['proposal'] = before
                for k in ('raw_proposal_before_producer_role', 'proposal_before_producer_role'):
                    out.pop(k, None)
                trace['reranked'] = False
                trace.pop('winner', None)
            elif not trace['reranked'] and before[0] is not None:
                out['raw_proposal_before_producer_role'], out['proposal_before_producer_role'] = before
                out['raw_proposal'], out['proposal'] = rescore(final)
                alias['extra_rerank'] = True
                trace.update(reranked=True, winner={'before': before[0]['representative_slug'],
                                                    'after': out['raw_proposal']['representative_slug']})

        def reads(v4_out, i, candidate):
            return {tier: dict({k: hit[k] for k in ('form', 'weakest', 'kinds', 'line_id')},
                               **self.form_kind(candidate, hit['form']))
                    for tier, hit in sorted(v4_out['candidates'][i]['v4_evidence']['producer']['reads'].items())}

        def complete(roles, candidate):
            return [dict(form=p['form'], **self.form_kind(candidate, p['form']))
                    for p in roles.producers(candidate, target, True)[1] if p['complete']]

        for i in sorted(set(added) | set(deltas)):
            candidate = base['candidates'][i]
            alias['phrase_changes'].append({
                'candidate_id': ids[i], 'pre_stage': frozen[i][PHRASE], 'alias_v4': phrase_rows[i][PHRASE],
                'frozen_stage': baseline[i][PHRASE], 'final': final[i][PHRASE],
                'v4_reads_frozen': reads(out['v4'], i, candidate), 'v4_reads_alias': reads(v4, i, candidate),
                'complete_forms_frozen': complete(self.frozen_roles, candidate),
                'complete_forms_alias': complete(self.roles, candidate)})
        trace['alias'] = alias
        out[self.trace_key] = trace
        return out
