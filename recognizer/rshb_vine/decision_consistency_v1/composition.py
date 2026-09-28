"""Decision consistency v1 at the actual consumer: components I and G over a built TextEvidenceSelectionV2.

``compose(selection, components)`` returns an instance-local shallow copy of the built v2 selector (A, B, D, M and
their state objects shared, frozen parents untouched, no global or module attribute patched):
  I  the copy's ``legacy`` is a copy of the running legacy selection whose resolver is CrossScriptResolution, so the
     guard, producer/alias/M rescoring and layout geometry resolve through it; a first pre_guard stage resolves the
     inner initial proposal again with it. The old103 ``legacy`` output of the inner selection stays frozen.
  G  a first post_guard stage (before layout geometry) re-decides the frozen guard with pairwise exclusivity.
Stage traces join the per-target text-evidence trace through ``extra`` (applied flag and seconds). Replay and live
``select`` run the same ``evaluate``. Probability and exact slug stay None.
"""
import copy

from rshb_vine.decision_consistency_v1 import identity as I
from rshb_vine.decision_consistency_v1 import pairwise as G
from rshb_vine.text_evidence_repair_v2.composition import TextEvidenceSelectionV2

ADAPTER_VERSION = 'decision-consistency-v1-selection'
PACKAGE = 'rshb_vine/decision_consistency_v1'
COMPONENTS = ('I', 'G')
BASE_RECIPE = ('A', 'B', 'D', 'M')
SOURCES = tuple(f'{PACKAGE}/{n}.py' for n in ('__init__', 'identity', 'pairwise', 'composition'))


def compose(selection, components):
    if type(selection) is not TextEvidenceSelectionV2 or tuple(selection.components) != BASE_RECIPE:
        raise ValueError('compose needs the built A+B+D+M TextEvidenceSelectionV2')
    if len(set(components)) != len(components) or set(components) - set(COMPONENTS):
        raise ValueError('Unknown or repeated component: %r' % (components,))
    components = tuple(c for c in COMPONENTS if c in components)
    twin = copy.copy(selection)
    twin.current_result, twin.records, twin.request, twin.last_trace, twin.last_text = None, [], None, None, None
    twin.extra, twin.identities = list(selection.extra), list(selection.identities)
    stages = list(selection.stages)
    if 'I' in components:
        twin.legacy = I.legacy_view(selection.legacy)
        stage = I.CrossScriptStage()
        stages.insert(0, stage)
        twin.extra.append(('I', stage))
        twin.identities.append(dict(stage.identity, component='I', module=I.__name__))
        twin.evaluate = _with_call_history(twin.evaluate, twin.legacy.resolver, stage.trace_key)
    if 'G' in components:
        stage = G.PairwiseGuardStage()
        at = next((i for i, s in enumerate(stages) if s.position == 'post_guard'), len(stages))
        stages.insert(at, stage)
        twin.extra.append(('G', stage))
        twin.identities.append(dict(stage.identity, component='G', module=G.__name__))
    keys = [s.trace_key for s in stages]
    if len(set(keys)) != len(keys):
        raise ValueError('Stage trace keys collide')
    twin.stages = tuple(stages)
    twin.components = (*selection.components, *components)
    twin.identity = dict(selection.identity, adapter=ADAPTER_VERSION, parent_adapter=selection.identity['adapter'],
                         components=list(twin.components), identities=twin.identities, stage_order=keys)
    return twin


def _with_call_history(evaluate, resolver, key):
    """Instance-local evaluate of the twin: one I call history per target, attached to its I stage trace."""

    def evaluated(**call):
        resolver.begin()
        try:
            out = evaluate(**call)
        finally:
            calls = resolver.end()
        trace = out.get(key)
        if trace is None:
            if calls:
                raise RuntimeError('I resolver ran without the I stage trace')
            return out
        trace['calls'] = calls
        trace['recovered_calls'] = len(I.recovered_calls(calls))
        return out

    return evaluated


def install(release, components):
    """Replace the installed v2 selector of a loaded text-evidence release graph by its composed copy.

    Owners include ``release.selection``: TextEvidenceRelease.recognize publishes its identity through T.block, so the
    composed identity (with ``parent_adapter``) is published, not the parent A+B+D+M one. Frozen parent sources untouched.
    """
    from rshb_vine.recognition_repair_v2.composition import Request
    from rshb_vine.systemic_ranking_v2.selection import attach
    inner, base = release.inner, release.base

    def owners():
        return (base.runtime.selection, base.target.inner.selection, inner.repair.selection, inner.selection,
                release.selection)

    parent = inner.selection
    if type(parent) is not TextEvidenceSelectionV2 or any(o is not parent for o in owners()):
        raise ValueError('v2 selector ownership differs from one TextEvidenceSelectionV2')
    wrapper = compose(parent, components)
    runtime = inner.repair.release.runtime
    attach(runtime, wrapper)
    original = runtime._attach_products

    def hooked(result, data, control_seconds):
        wrapper.request = Request(result, data)
        try:
            return original(result, data, control_seconds)
        finally:
            wrapper.request = None

    runtime._attach_products = hooked
    inner.repair.release.target.inner.selection = wrapper
    inner.repair.selection = wrapper
    inner.selection = wrapper
    release.selection = wrapper
    if any(o is not wrapper for o in owners()):
        raise ValueError('Composed selector is not the single live consumer')
    return wrapper
