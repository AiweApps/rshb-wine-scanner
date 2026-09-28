"""Text-evidence repair v2 at the actual consumer: v1 components A/B unchanged plus component D in the producer stage.

``TextEvidenceSelectionV2`` wraps the running RepairSelectionV2 like v1. With D, a per-instance copy of that parent
carries AliasProducerStage in place of its producer_role stage (same position; the original parent and its stage are
untouched), and the v1 wrapper (A normalization, B injection proxy over the frozen index and context) is built over
that copy. Without D the copy is not made and the wrapper equals the v1 wrapper of the same components (G0 parity
with no component). Component M (reader preservation of A, ``reader_preservation``) is a pre_guard stage right after
producer_role; it requires A and reads the normalization record of the current target through ``a_record``. ``parent``
stays the running RepairSelectionV2; the per-instance copy is ``twin``. Probability and exact slug stay None.
"""
import copy

from rshb_vine.recognition_repair_v2 import producer as RP
from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2
from rshb_vine.text_evidence_repair_v1 import composition as C1
from rshb_vine.text_evidence_repair_v2 import producer_consistency as D
from rshb_vine.text_evidence_repair_v2 import reader_preservation as M

ADAPTER_VERSION = 'text-evidence-repair-v2-selection'
PACKAGE = 'rshb_vine/text_evidence_repair_v2'
V1_COMPONENTS = ('A', 'B')
COMPONENTS = (*V1_COMPONENTS, 'D', 'M')
COMPONENT_SOURCES = {'A': C1.COMPONENT_SOURCES['A'], 'B': C1.COMPONENT_SOURCES['B'], 'D': D.SOURCES, 'M': M.SOURCES}
SOURCES = (f'{PACKAGE}/__init__.py', f'{PACKAGE}/composition.py', *C1.SOURCES)


def with_alias_stage(parent, table_checksum):
    """A copy of ``parent`` whose repair-v2 producer_role stage is replaced by the alias stage."""
    kinds = [type(s) for s in parent.stages]
    if kinds.count(RP.ProducerRoleStage) != 1 or [s.trace_key for s in parent.stages].count('producer_role') != 1:
        raise ValueError('Parent has no single repair-v2 producer_role stage to replace')
    stage = D.AliasProducerStage(parent.root, parent.inner, table_checksum)
    twin = copy.copy(parent)
    twin.stages = tuple(stage if type(s) is RP.ProducerRoleStage else s for s in parent.stages)
    twin.current_result, twin.records, twin.request, twin.last_trace = None, [], None, None
    return twin, stage


class RecordingNormalization:
    """Component A of v1 that keeps the original, effective and trace of the target being evaluated."""

    def __init__(self, inner):
        self.inner, self.name, self.identity, self.last = inner, inner.name, inner.identity, None
        self.module, self.lexicon, self.vocabulary = inner.module, inner.lexicon, inner.vocabulary

    def normalize(self, observations, provenance):
        effective, detail = self.inner.normalize(observations, provenance)
        self.last = {'original': observations, 'effective': effective, 'detail': detail}
        return effective, detail

    def counterpart_line_keys(self, result, instance_id):
        return self.inner.counterpart_line_keys(result, instance_id)


class TextEvidenceSelectionV2(C1.TextEvidenceSelection):
    def __init__(self, parent, components=(), tables=None):
        tables = tables or {}
        if len(set(components)) != len(components) or set(components) - set(COMPONENTS):
            raise ValueError('Unknown or repeated component: %r' % (components,))
        if 'M' in components and 'A' not in components:
            raise ValueError('Reader preservation M needs normalization A')
        if type(parent) is not RepairSelectionV2:
            raise ValueError('TextEvidenceSelectionV2 wraps the running RepairSelectionV2 instance only')
        running, self.alias_stage, self.twin = parent, None, None
        if 'D' in components:
            self.twin, self.alias_stage = with_alias_stage(parent, tables['D'])
            parent = self.twin
        super().__init__(parent, tuple(c for c in V1_COMPONENTS if c in components), tables)
        self.parent = running
        if self.normalization is not None:
            self.normalization = RecordingNormalization(self.normalization)
        self.components = tuple(c for c in COMPONENTS if c in components)
        if self.alias_stage is not None:
            self.identities.append(dict(self.alias_stage.identity, component='D', module=D.__name__))
        self.extra = []
        if 'M' in self.components:
            stage = M.ReaderPreservationStage(self.inner, self.a_record)
            keys = [s.trace_key for s in self.stages]
            if stage.position != 'pre_guard' or stage.trace_key in keys:
                raise ValueError('Reader preservation must be a new pre_guard stage')
            at = keys.index('producer_role') + 1
            self.stages = self.stages[:at] + (stage,) + self.stages[at:]
            self.extra.append(('M', stage))
            self.identities.append(dict(stage.identity, component='M', module=M.__name__))
        self.identity = {'adapter': ADAPTER_VERSION, 'v1_adapter': C1.ADAPTER_VERSION,
                         'components': list(self.components), 'identities': self.identities,
                         'stage_order': [s.trace_key for s in self.stages],
                         'producer_stage': type(next(s for s in self.stages if s.trace_key == 'producer_role')).__name__}

    def a_record(self):
        """Normalization record {'original', 'effective', 'detail'} of the target being evaluated; None without A."""
        if self.normalization is None:
            return None
        if self.normalization.last is None:
            raise RuntimeError('Normalization A has not run for this target')
        return self.normalization.last

    def evaluate(self, **call):
        timed = [('producer_alias_stage', self.alias_stage)] if self.alias_stage is not None else []
        timed += [(st.trace_key, st) for _, st in self.extra]
        for _, st in timed:
            st.last_seconds = None
        if self.normalization is not None:
            self.normalization.last = None
        out = super().evaluate(**call)
        trace = out[C1.TRACE_KEY]
        trace['adapter'], trace['components'] = ADAPTER_VERSION, list(self.components)
        if self.alias_stage is not None:
            alias = out['producer_role']['alias']
            trace['producer_aliases'] = alias
            trace['applied'] = trace['applied'] or alias['applied']
        for _, st in self.extra:
            trace[st.trace_key] = out.get(st.trace_key)
            trace['applied'] = trace['applied'] or bool((out.get(st.trace_key) or {}).get('applied'))
        if timed:
            seconds = {k: v for k, v in trace['seconds'].items() if k != 'components_total'}
            seconds.update({name: st.last_seconds for name, st in timed})
            trace['seconds'] = dict(seconds, components_total=sum(v for v in seconds.values() if v is not None))
        return out


def install(v2_runtime, components, tables=None):
    """Replace the RepairSelectionV2 of a constructed RecognitionRepairV2 on its own runtime graph (no globals)."""
    from rshb_vine.recognition_repair_v2.composition import Request
    from rshb_vine.systemic_ranking_v2.selection import attach
    parent, repair = v2_runtime.selection, v2_runtime.repair
    runtime = repair.release.runtime
    if (type(parent) is not RepairSelectionV2 or repair.selection is not parent or runtime.selection is not parent
            or repair.release.target.inner.selection is not parent):
        raise ValueError('RepairSelectionV2 ownership changed')
    wrapper = TextEvidenceSelectionV2(parent, components, tables)
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
    v2_runtime.selection = wrapper
    return wrapper
