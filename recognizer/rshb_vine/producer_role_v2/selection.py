"""Selector adapter: OCR candidate injection -> producer-role features -> frozen ranker/resolver -> guard v2.

The same ``evaluate`` serves CPU replay and the live ``select`` inherited from the OCR adapter, so registry,
model, ranker, records, product resolution and published record fields stay those of the running selector
(the record keeps its adapter name, which the repair runtime checks) and gains a ``producer_role_v2`` trace.
Probability and exact slug stay None.
"""
from pathlib import Path

from rshb_vine.learned_selection_features import _candidate_key
from rshb_vine.ocr_candidate_repair_v1 import discriminator as D
from rshb_vine.ocr_candidate_repair_v1.guarded import GuardedOcrCandidateSelection
from rshb_vine.ocr_candidate_repair_v1.selection import OcrCandidateSelection, _relabel
from rshb_vine.producer_role_v2 import roles as R
from rshb_vine.systemic_ranking_v2.selection import feature_digest

ADAPTER_VERSION = 'producer-role-v2-selection'


class ProducerRoleSelection(OcrCandidateSelection):
    def __init__(self, root, inner, conflicts=None, *, index=None, flags=None):
        if index is None:
            super().__init__(root, inner, conflicts)
        else:
            self.root, self.inner = Path(root), inner
            self.registry, self.legacy, self.model = inner.registry, inner.legacy, inner.model
            self.index, self.conflicts, self.current_result, self.records = index, conflicts, None, []
        self.flags = dict(R.FULL, **(flags or {}))
        self.roles = R.ProducerRoles(inner.context, inner.stats)
        self.last_trace = None

    def _rescore(self, out, control_slug):
        def rescore(rows):
            raw = self.inner._propose(out['base'], out['v4'], rows, control_slug)
            _relabel(raw, {_candidate_key(s, self.registry.cards) for s in out['ocr_candidate_injection']['injected']})
            return raw, self.legacy.resolver.resolve(out['base'], raw)
        return rescore

    def evaluate(self, *, control_slug, raw_visual, observations, control_candidates, provenance, other_line_keys=()):
        out = super().evaluate(control_slug=control_slug, raw_visual=raw_visual, observations=observations,
                               control_candidates=control_candidates, provenance=provenance,
                               other_line_keys=other_line_keys)
        trace = apply(out, self.roles, provenance, other_line_keys, self._rescore(out, control_slug), self.flags)
        out = D.apply(out, self.legacy.resolver, self.index.profile)
        out['producer_role_v2'] = self.last_trace = trace
        return out

    def select(self, **kwargs):
        result = super().select(**kwargs)
        t = self.last_trace
        result['systemic_ranking_v2']['producer_role_v2'] = {
            'adapter': ADAPTER_VERSION, 'rule': t['rule'], 'excluded_lines': t['excluded_lines'],
            'feature_changes': [{k: c[k] for k in ('candidate_id', 'features', 'own_form', 'complete_forms')}
                                for c in t['changes']],
            'winner': t.get('winner')}
        return result


def apply(out, roles, provenance, other_line_keys, rescore, flags=R.FULL):
    """Pre-guard stage: replace the producer features of ``out`` in place and re-rank once through ``rescore``.

    ``out`` is an evaluated selection before the discriminator guard (rows, evidence, base, candidate_ids,
    feature_digest, raw_proposal, proposal); ``roles`` is ``R.ProducerRoles(context, stats)``; ``rescore(rows)``
    returns ``(raw_proposal, proposal)`` from the frozen ranker and resolver. The guard is not applied here.
    Returns the trace; without a feature change ``out`` is untouched.
    """
    target = R.TargetProducerText(out['base']['query_evidence']['observations'], provenance, other_line_keys, flags)
    changes, new = [], []
    for (f, ev), row, cid, old in zip(roles.features(out, target, flags), out['rows'], out['candidate_ids'], out['evidence']):
        new.append(dict(row, **f))
        diff = {n: [row[n], f[n]] for n in R.FEATURES if f[n] != row[n]}
        if diff:
            changes.append({'candidate_id': cid, 'features': diff, 'own_form': ev['own']['form'],
                            'own_anchored': ev['own']['anchored'], 'complete_forms': ev['complete_forms'],
                            'own_tokens': [{k: t[k] for k in ('token', 'weight', 'specific', 'status', 'lines')}
                                           for t in ev['own']['tokens']],
                            'excluded_tokens': ev['excluded_tokens'],
                            'original_producer': {k: old['producer'][k] for k in ('form', 'read_w', 'read_g.v', 'read_g.p')},
                            'rival': ev['rival'], 'original_rival': old['producer_rival']})
    trace = {'adapter': ADAPTER_VERSION, 'rule': R.RULE['version'], 'flags': dict(flags),
             'excluded_lines': target.trace(), 'changes': changes, 'reranked': False}
    if not changes:
        return trace
    out['rows_before_producer_role'] = out['rows']
    out['feature_digest_before_producer_role'] = out['feature_digest']
    out['rows'] = new
    out['feature_digest'] = feature_digest(new, out['candidate_ids'])
    if out.get('raw_proposal') is not None:
        out['raw_proposal_before_producer_role'] = out['raw_proposal']
        out['proposal_before_producer_role'] = out['proposal']
        out['raw_proposal'], out['proposal'] = rescore(new)
        trace['reranked'] = True
        trace['winner'] = {'before': out['raw_proposal_before_producer_role']['representative_slug'],
                           'after': out['raw_proposal']['representative_slug']}
    return trace


def install(repair):
    """Replace the guarded OCR selector of a constructed RecognitionRepairV1 (index, conflicts and hooks reused)."""
    from rshb_vine.systemic_ranking_v2.selection import attach
    guarded, runtime = repair.selection, repair.release.runtime
    if type(guarded) is not GuardedOcrCandidateSelection or runtime.selection is not guarded \
            or repair.release.target.inner.selection is not guarded:
        raise ValueError('Guarded OCR selection ownership changed')
    wrapper = ProducerRoleSelection(guarded.root, guarded.inner, guarded.conflicts, index=guarded.index)
    attach(runtime, wrapper)
    repair.release.target.inner.selection = wrapper
    repair.selection = wrapper
    return wrapper
