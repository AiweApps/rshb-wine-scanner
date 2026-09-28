"""The one installation of an admitted I/G/O/R recipe on a loaded text-evidence graph, shared by candidate and release.

I and G go through ``decision_consistency_v1.composition.install`` (all five selector owners, ``release.selection``
included, so T.block publishes the composed identity); O and R through the ``reading_recovery_v1`` installers on their
own nodes (expanded.alternative; zero-target route + frozen LabelRescue). R's request scope wraps the outer request and
its truthful label-view contract runs only when R is in the recipe. Nothing global, pinned or module-level is patched.
"""
from contextlib import nullcontext
from copy import deepcopy

COMPONENTS = ('I', 'G', 'O', 'R')
PACKAGE = 'rshb_vine/evidence_consistency_release_v1'
BLOCK = 'evidence_consistency_release_v1'
RR = 'rshb_vine/reading_recovery_v1'
DC_SOURCES = tuple(f'rshb_vine/decision_consistency_v1/{n}.py' for n in ('__init__', 'identity', 'pairwise', 'composition'))
COMPONENT_SOURCES = {'I': DC_SOURCES, 'G': DC_SOURCES,
                     'O': (f'{RR}/__init__.py', f'{RR}/rotated_ocr.py',
                           'runs/text-evidence-improve-v1/identity/short-producer-forms-v1.json'),
                     'R': (f'{RR}/__init__.py', f'{RR}/route.py', f'{RR}/candidate.py')}


def check(recipe):
    """A non-empty subset of I, G, O, R written in that order; anything else is refused."""
    recipe = tuple(recipe or ())
    if not recipe or len(set(recipe)) != len(recipe) or set(recipe) - set(COMPONENTS):
        raise ValueError('Recipe must be a non-empty subset of I, G, O, R: %r' % (recipe,))
    if recipe != tuple(c for c in COMPONENTS if c in recipe):
        raise ValueError('Recipe must list components in the order I, G, O, R: %r' % (recipe,))
    return recipe


def component_sources(recipe):
    return tuple(dict.fromkeys(p for c in check(recipe) for p in COMPONENT_SOURCES[c]))


def install(release, recipe, root):
    """Install the recipe once on ``release`` (TextEvidenceRelease or its F10 copy); returns the installation record."""
    from rshb_vine.decision_consistency_v1 import composition as DC
    from rshb_vine.reading_recovery_v1 import route as R, rotated_ocr as O
    recipe = check(recipe)
    record = {'recipe': list(recipe)}
    selector = tuple(c for c in recipe if c in DC.COMPONENTS)
    if selector:
        parent = release.selection
        wrapper = DC.install(release, selector)
        if wrapper is parent or release.selection is not wrapper or tuple(wrapper.components)[-len(selector):] != selector:
            raise ValueError('Composed I/G selector is not the single live consumer of this graph')
        record['decision_consistency'] = {
            'adapter': DC.ADAPTER_VERSION, 'components': list(selector), 'identity': deepcopy(wrapper.identity),
            'owners': ['base.runtime.selection', 'base.target.inner.selection', 'inner.repair.selection',
                       'inner.selection', 'release.selection']}
    if 'O' in recipe:
        record['rotated_ocr'] = O.install(release, root)
    if 'R' in recipe:
        record['bottle_context_route'] = R.install(release)
    return record


def scope(recipe, roi, bottles):
    """R's outer request scope (bounded to no-ROI requests) when R is selected; otherwise no scope."""
    from rshb_vine.reading_recovery_v1 import route as R
    return R.request_scope(roi, bottles) if 'R' in recipe else nullcontext(None)


def label_view_contract(result, recipe):
    if 'R' not in recipe:
        return []
    from rshb_vine.reading_recovery_v1.candidate import ReadingRecoveryCandidate
    return ReadingRecoveryCandidate._label_view_contract(result)


def block(result, recipe, route_scope, contract_entries, **identity):
    """Public block: recipe, identity/lineage fields, R route events and O reads; slug/probability stay None."""
    rotated = result.get('rotated_label_ocr')
    return {**identity, 'recipe': list(recipe),
            'route_events': deepcopy(route_scope['events']) if route_scope is not None else [],
            'label_view_contract_entries': contract_entries,
            'rotated_ocr_performed': bool(rotated and rotated.get('performed')) if 'O' in recipe else None,
            'calibrated': False, 'probability': None}
