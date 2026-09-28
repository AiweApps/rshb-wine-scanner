"""Selector composition at the actual consumer: guarded OCR selection of repair-v1 plus ordered, checked stages.

Order inside the one ``evaluate`` used by CPU replay and live ``select``: OCR injection -> frozen ranker ->
existing resolver -> pre-guard stages -> discriminator guard v2 -> post-guard stages. With no stages the output is
the guarded v1 selection itself. A stage may reorder the raw proposal only through the unchanged resolver; the pool,
learned scores, public probability (None) and the published record adapter stay those of the running selector.

Stage contract (a producer-role stage must implement it; none is loaded until its admitted API/hash is pinned):
  trace_key: str            key of its per-target trace in ``out`` and in the published record
  identity: dict            sealed into the profile (rule/version, sources_sha256)
  position: 'pre_guard' | 'post_guard'
  apply(selection, out, call, request) -> out
      selection  this adapter (inner, legacy.resolver, index.profile, registry)
      out        current evaluate output (base, rows, evidence, raw_proposal, proposal, ...)
      call       the evaluate keyword arguments (control_slug, raw_visual, observations, control_candidates,
                 provenance, other_line_keys)
      request    Request of the in-flight image (control result + bytes) or None
"""
from contextlib import contextmanager
from pathlib import Path

from rshb_vine.ocr_candidate_repair_v1 import discriminator as D
from rshb_vine.ocr_candidate_repair_v1.guarded import GuardedOcrCandidateSelection
from rshb_vine.ocr_candidate_repair_v1.selection import OcrCandidateSelection

ADAPTER_VERSION = 'recognition-repair-v2-selection'
POSITIONS = ('pre_guard', 'post_guard')


class Request:
    """The in-flight control result and request bytes; the image is decoded like the runtime, only on demand."""

    def __init__(self, result, data):
        self.result, self.data, self._image = result, data, None

    def target(self, instance_id):
        return next((t for t in self.result.get('targets', []) if str(t['instance_id']) == instance_id), None)

    def image(self):
        if self._image is None:
            from rshb_vine.preprocessing import decode
            self._image = decode(self.data)[0]
        return self._image


def _check(before, out, stage):
    ids = before['candidate_ids']
    if out['candidate_ids'] != ids or out['base'] is not before['base']:
        raise RuntimeError(stage.trace_key + ' changed the selector pool')
    raw, proposal = out.get('raw_proposal'), out.get('proposal')
    if raw is None:
        return
    if sorted(r['candidate_id'] for r in raw['ranked_candidates']) != sorted(ids):
        raise RuntimeError(stage.trace_key + ' changed the ranked candidate set')
    if proposal.get('probability') is not None or proposal.get('exact_slug') is not None:
        raise RuntimeError(stage.trace_key + ' published a probability or exact slug')
    if proposal is not before.get('proposal') and 'existing_evidence_resolution' not in proposal:
        raise RuntimeError(stage.trace_key + ' changed the proposal without the existing resolver')


class RepairSelectionV2(OcrCandidateSelection):
    def __init__(self, guarded, stages=()):
        if type(guarded) is not GuardedOcrCandidateSelection:
            raise ValueError('RepairSelectionV2 composes the guarded v2 OCR selection only')
        if any(s.position not in POSITIONS for s in stages) or len({s.trace_key for s in stages}) != len(stages):
            raise ValueError('Stage positions or trace keys invalid')
        self.root, self.inner = Path(guarded.root), guarded.inner
        self.registry, self.legacy, self.model = guarded.inner.registry, guarded.inner.legacy, guarded.inner.model
        self.index, self.conflicts = guarded.index, guarded.conflicts
        self.current_result, self.records = None, []
        self.stages = tuple(stages)
        self.request = None
        self.last_trace = None

    def _run(self, position, out, call):
        for stage in self.stages:
            if stage.position == position:
                before = {'candidate_ids': list(out['candidate_ids']), 'base': out['base'], 'proposal': out.get('proposal')}
                out = stage.apply(self, out, call, self.request)
                _check(before, out, stage)
        return out

    def evaluate(self, **call):
        out = super().evaluate(**call)
        out = self._run('pre_guard', out, call)
        out = D.apply(out, self.legacy.resolver, self.index.profile)
        out = self._run('post_guard', out, call)
        self.last_trace = {'adapter': ADAPTER_VERSION, 'instance_id': str(call['raw_visual']['instance_id']),
                           'stages': {s.trace_key: out.get(s.trace_key) for s in self.stages}}
        out['recognition_repair_v2'] = self.last_trace
        return out

    def select(self, **kwargs):
        self.last_trace = None
        result = super().select(**kwargs)
        if self.last_trace is None:
            raise RuntimeError('Live select did not run the composed evaluate')
        result['systemic_ranking_v2']['recognition_repair_v2'] = self.last_trace
        return result

    @contextmanager
    def replaying(self, result, data):
        """CPU replay of saved responses: the saved body stands in for the control result of the same bytes."""
        self.request = Request(result, data)
        try:
            yield self
        finally:
            self.request = None


def install(repair, stages=()):
    """Replace the guarded OCR selector of a constructed RecognitionRepairV1 (index, conflicts, hooks reused)."""
    from rshb_vine.systemic_ranking_v2.selection import attach
    guarded, runtime = repair.selection, repair.release.runtime
    if (type(guarded) is not GuardedOcrCandidateSelection or runtime.selection is not guarded
            or repair.release.target.inner.selection is not guarded):
        raise ValueError('Guarded OCR selection ownership changed')
    wrapper = RepairSelectionV2(guarded, stages)
    attach(runtime, wrapper)
    original = runtime._attach_products

    def hooked(result, data, control_seconds):
        wrapper.request = Request(result, data)
        try:
            return original(result, data, control_seconds)
        finally:
            wrapper.request = None

    runtime._attach_products = hooked
    repair.release.target.inner.selection = wrapper
    repair.selection = wrapper
    return wrapper
