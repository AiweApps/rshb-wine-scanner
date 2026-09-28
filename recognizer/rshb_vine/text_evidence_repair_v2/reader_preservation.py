"""Component M: a typed reading already present before normalization A is not lost through A.

V4._observed_typed drops the low-tier (score < V4.HIGH) hits of a typed key once the same key is also read on a
high-tier line. When A turns a Latin-homoglyph line (MYCKAT) into a high-tier reading of that key, an unchanged
low-tier line of another reader (Vision «МУСКАТ» 0.5) disappears from the typed support/contradiction lines, and
its per-reader grade with it. The key set, the typed state and every other key keep their meaning; only this
provenance is lost.

M restores exactly those hits, and nothing else:
- only for a target whose packet A changed; otherwise the stage returns ``out`` untouched;
- only for a (role, key) that A made high-tier (a high hit on a line containing an A-changed observation) while
  the original packet read it only low-tier, so the original evidence itself was never tier-filtered;
- only low hits on effective lines that contain no A-changed observation and have exactly the observation set of
  an original low hit of the same (role, key), after the frozen producer-phrase exclusion of both packets;
- through the frozen typed logic: E1._typed_lines maps restored hits to support / contradiction with the
  candidate's effective typed state; if adding the hits would change that state (V4._typed_state), the
  candidate-role keeps the effective lines and the skip is traced;
- per-reader grades are E._best_grade over the united line sets (max per reader over native scores, never a sum);
  contradiction counts only where the frozen build counts it (contra_any and a comparable claim).

Only typ.* features may change; a recomputation without restored hits must equal the incoming rows, otherwise the
stage raises. One rescore through the frozen ranker and resolver. Raw and effective OCR are not modified.
"""
import time

from rshb_vine.interaction_ranker_v1 import evidence as E1
from rshb_vine.learned_selection_features import _candidate_key
from rshb_vine.ocr_candidate_repair_v1.selection import _relabel
from rshb_vine.system_selection_v4 import features as V4
from rshb_vine.system_selection_v4 import text
from rshb_vine.systemic_ranking_v2 import evidence as E
from rshb_vine.systemic_ranking_v2.selection import feature_digest
from rshb_vine.text_evidence_repair_v1.composition import check_effective

STAGE = 'text-evidence-reader-preservation-pre-guard-stage'
SOURCES = ('rshb_vine/text_evidence_repair_v2/reader_preservation.py',)
TYPED_FEATURES = tuple(n for n in E.FEATURE_NAMES if n.startswith('typ.'))
RULE = {
    'version': 'a-tier-filter-reader-preservation-v1',
    'scope': 'targets whose packet component A changed',
    'restored': 'original low-tier hits of a (role, key) that A made high-tier while the original read it only '
                'low-tier; same observation set, no A-changed observation, producer-phrase exclusion of both packets',
    'typed_logic': 'E1._typed_lines with the effective typed state; state must be unchanged by the restored hits',
    'grades': 'E._best_grade per reader over united line sets (max, no sums, no new coefficient)',
    'features': list(TYPED_FEATURES),
}


def _producer_forms(base, context):
    return sorted({f for c in base['candidates'] for m in E1._members(c, context)[0].values() for f in m['producers']})


def _explained(lines, forms):
    """Producer-phrase positions, as E1.field_evidence derives them."""
    explained = set()
    for form in forms:
        tokens = tuple(t.forms[0] for t in text.tokenize(form))
        for line in lines:
            found = text.phrase_match(line['tokens'], tokens) if tokens else None
            if found:
                explained |= {(line['line_id'], i) for i in range(found['start'], found['end'])}
    return explained


def _observed(lines, lexicon, forms):
    """(frozen tier-filtered observation, low-tier hits before the filter) over the same line ids."""
    explained = _explained(lines, forms)
    full = V4._observed_typed(lines, lexicon, explained)
    low = V4._observed_typed([l for l in lines if not l['high']], lexicon, explained)
    return full, {role: low[role]['low'] for role in low}


def _obs(lines, hit):
    return frozenset(lines[hit['line_id']]['observation_ids'])


def lost_hits(original, effective, changed, lexicon, forms):
    """{role: {key: [effective low hits]}} that the tier filter removed only because of A, and the observation."""
    changed_ids = {c['observation_id'] for c in changed}
    ol, el = V4._lines(original), V4._lines(effective)
    of, _ = _observed(ol, lexicon, forms)
    ef, elow = _observed(el, lexicon, forms)
    restored, trace = {}, []
    for role in V4.TYPED_ROLES:
        for key in sorted(set(ef[role]['high']) & set(elow[role])):
            if key in of[role]['high'] or key not in of[role]['low']:
                continue
            introduced = [h for h in ef[role]['high'][key] if _obs(el, h) & changed_ids]
            if not introduced:
                raise RuntimeError('typed key %s/%s became high-tier without an A-changed observation' % (role, key))
            before = {_obs(ol, h) for h in of[role]['low'][key]}
            hits = [h for h in elow[role][key] if _obs(el, h) in before and not _obs(el, h) & changed_ids]
            record = {'role': role, 'key': key,
                      'high_from_A': [sorted(_obs(el, h)) for h in introduced],
                      'original_low': sorted(sorted(s) for s in before),
                      'restored': [sorted(_obs(el, h)) for h in hits]}
            if len(hits) < len(before):
                record['not_restorable'] = 'original low observation set changed or merged under A'
            trace.append(record)
            if hits:
                restored.setdefault(role, {})[key] = hits
    return restored, ef, trace


def typed_rows(base, context, provenance, restored=None, observed=None):
    """typ.* rows per candidate as E.build computes them, with optional restored low hits; and per-candidate trace."""
    lines = V4._lines(base['query_evidence']['observations'])
    grades = E._line_grades(lines, provenance)
    typed = E1.field_evidence(base, context)['candidates']
    key_tokens = context.grape_key_tokens
    rows, traces = [], []
    for c in base['candidates']:
        cid = c['candidate_id']
        members, _ = E._members(c, context)
        f, verified, trace = {}, dict.fromkeys(E.READERS, 0.), {}
        for role in E.TYPED:
            t = typed[cid]['typed'][role]
            claims = [m['typed'][role] for m in members.values()]
            support, contra = set(t['support']), set(t['contra'])
            extra = (restored or {}).get(role)
            if extra:
                united = {'high': dict(observed[role]['high']), 'low': dict(observed[role]['low'])}
                for key, hits in extra.items():
                    united['low'][key] = united['low'].get(key, []) + hits
                if V4._typed_state(role, united, claims, key_tokens) != t['state']:
                    trace[role] = {'skipped': 'typed_state_would_change'}
                else:
                    rs, rc = E1._typed_lines(role, {'high': {}, 'low': extra}, claims, key_tokens, t['state'])
                    if rs - support or rc - contra:
                        trace[role] = {'support_added': sorted(rs - support), 'contra_added': sorted(rc - contra)}
                    support, contra = support | rs, contra | rc
            contradicted = E._comparable(role, claims) if t['state']['contra_any'] else []
            sg = E._best_grade(support, grades)
            cg = E._best_grade(contra, grades) if contradicted else dict.fromkeys(E.READERS, 0.)
            for r in E.READERS:
                f[f'typ.{role}.support_g.{r}'] = sg[r]
                if role != 'grape_blend':
                    f[f'typ.{role}.contra_g.{r}'] = cg[r]
                if any(cl['status'] in E.VERIFIED for cl in contradicted):
                    verified[r] = max(verified[r], cg[r])
            if role in trace:
                trace[role].update(support_lines=sorted(support), contra_lines=sorted(contra))
        for r in E.READERS:
            f[f'typ.contra_verified_g.{r}'] = verified[r]
        rows.append(f)
        traces.append(trace)
    return rows, traces


class ReaderPreservationStage:
    """Pre-guard stage after the producer stage. ``source()`` returns the normalization record of the target being
    evaluated ({'original': observations, 'effective': observations, ...}, as RecordingNormalization.last) or None
    when A is not enabled; the caller resets it before every evaluate."""

    trace_key = 'reader_preservation'
    position = 'pre_guard'

    def __init__(self, inner, source):
        if set(TYPED_FEATURES) - set(E.FEATURE_NAMES) or not TYPED_FEATURES:
            raise ValueError('typed features are not systemic features')
        self.inner, self.source = inner, source
        self.identity = {'stage': STAGE, 'rule': RULE['version'], 'features': list(TYPED_FEATURES),
                         'position': 'pre_guard after producer_role'}
        self.last_seconds = None

    def apply(self, selection, out, call, request):
        started = time.perf_counter()
        try:
            return self._apply(selection, out, call)
        finally:
            self.last_seconds = time.perf_counter() - started

    def _apply(self, selection, out, call):
        record = self.source()
        trace = {'stage': STAGE, 'rule': RULE['version'], 'applied': False, 'reranked': False}
        out[self.trace_key] = trace
        if record is None:
            trace['reason'] = 'a_disabled'
            return out
        original, effective = record['original'], call['observations']
        packet = out['base']['query_evidence']['observations']
        if record['effective'] is not effective or [o.get('raw_text') for o in packet] != [o.get('raw_text') for o in effective]:
            raise RuntimeError('reader preservation: normalization record is not the packet being evaluated')
        changed = check_effective(original, effective)
        if not changed:
            trace['reason'] = 'a_unchanged'
            return out
        context = self.inner.context
        base = out['base']
        restored, observed, lost = lost_hits(original, effective, changed, context.lexicon, _producer_forms(base, context))
        trace['lost'] = lost
        if not restored:
            trace['reason'] = 'no_tier_filtered_original_hit'
            return out
        plain, _ = typed_rows(base, context, call['provenance'])
        for cid, row, p in zip(out['candidate_ids'], out['rows'], plain):
            drift = [n for n in TYPED_FEATURES if row[n] != p[n]]
            if drift:
                raise RuntimeError('reader preservation: typed recomputation differs from incoming rows %s of %s' % (drift, cid))
        fresh, per = typed_rows(base, context, call['provenance'], restored, observed)
        new, changes = [], []
        for cid, row, f, t in zip(out['candidate_ids'], out['rows'], fresh, per):
            diff = {n: [row[n], f[n]] for n in TYPED_FEATURES if f[n] != row[n]}
            if diff or t:
                changes.append({'candidate_id': cid, 'features': diff, 'roles': t})
            new.append(dict(row, **{n: f[n] for n in TYPED_FEATURES}))
        trace['changes'] = changes
        if not any(c['features'] for c in changes):
            trace['reason'] = 'restored_lines_leave_grades_unchanged'
            return out
        evidence = []
        for ev, t in zip(out['evidence'], per):
            typed = dict(ev['typed'])
            for role, x in t.items():
                if 'support_lines' in x:
                    typed[role] = dict(typed[role], support_lines=x['support_lines'], contra_lines=x['contra_lines'],
                                       reader_preserved=True)
            evidence.append(dict(ev, typed=typed))
        out['rows_before_reader_preservation'], out['evidence_before_reader_preservation'] = out['rows'], out['evidence']
        out['feature_digest_before_reader_preservation'] = out['feature_digest']
        out['rows'], out['evidence'] = new, evidence
        out['feature_digest'] = feature_digest(new, out['candidate_ids'])
        trace['applied'] = True
        if out.get('raw_proposal') is not None:
            injected = {_candidate_key(s, selection.registry.cards) for s in out['ocr_candidate_injection']['injected']}
            raw = selection.inner._propose(base, out['v4'], new, call['control_slug'])
            _relabel(raw, injected)
            out['raw_proposal_before_reader_preservation'] = out['raw_proposal']
            out['proposal_before_reader_preservation'] = out['proposal']
            out['raw_proposal'] = raw
            out['proposal'] = selection.legacy.resolver.resolve(base, raw)
            trace.update(reranked=True, winner={'before': out['raw_proposal_before_reader_preservation']['representative_slug'],
                                                'after': raw['representative_slug']})
        return out
