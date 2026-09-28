"""The one installation of an admitted G/T recipe on a loaded cd0b9910 graph, shared by candidate and release.

G is the frozen atlas geometry repair (``atlas_geometry_repair_v1.runtime.install``: trusted-region localizer delegate
and orphan-stage wrap on their own nodes; ``recognize_with_recovery`` runs the parent once observe-only and one armed
pass only when that result has zero targets and an eligible trial); T is the frozen composite-identity
proposer (``atlas_text_repair_v1.composition.install`` on the single composed I selector, all five owners). Each
component is bound by its own frozen document: G by its candidate descriptor (sources, constants, parent cd0b9910),
T by its sealed protocol d9bf34ed (pins, composite checksum, parent index). The parent recipe I stays as installed by
the F10 graph; old G/O/R of decision consistency and reading recovery are never installed. Nothing global, pinned or
module-level is patched.
"""
from copy import deepcopy

from rshb_vine.io import local_path, read_json, sha256, verify

COMPONENTS = ('G', 'T')
PACKAGE = 'rshb_vine/atlas_repair_release_v1'
BLOCK = 'atlas_repair_release_v1'
PARENT_CHECKSUM = 'cd0b9910133f4c17df53d7e171ae3c0432a5b8a1252163d42413ef6808d81ece'
PARENT_RECIPE = ['I']
G_DESCRIPTOR_KIND = 'atlas-geometry-repair-v1-candidate-descriptor'
T_PROTOCOL = 'runs/atlas-repair-v1/text/protocol.json'
T_PROTOCOL_KIND = 'atlas-text-repair-v1-protocol'
T_PROTOCOL_CHECKSUM = 'd9bf34ed0d84b89557d2dc1cb704a90f7dc65338d5c2632743d9b453dd632166'
T_SOURCES = ('rshb_vine/atlas_text_repair_v1/__init__.py', 'rshb_vine/atlas_text_repair_v1/index.py',
             'rshb_vine/atlas_text_repair_v1/composition.py')
OWNERS = ['base.runtime.selection', 'base.target.inner.selection', 'inner.repair.selection', 'inner.selection',
          'release.selection']


def check(recipe):
    """A non-empty subset of G, T written in that order; anything else is refused."""
    recipe = tuple(recipe or ())
    if not recipe or len(set(recipe)) != len(recipe) or set(recipe) - set(COMPONENTS):
        raise ValueError('Recipe must be a non-empty subset of G, T: %r' % (recipe,))
    if recipe != tuple(c for c in COMPONENTS if c in recipe):
        raise ValueError('Recipe must list components in the order G, T: %r' % (recipe,))
    return recipe


def parse(text):
    return check(tuple(c.strip() for c in text.split(',') if c.strip()))


def g_sources():
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    return tuple(G.SOURCES)


def component_sources(recipe):
    sources = {'G': g_sources() if 'G' in recipe else (), 'T': T_SOURCES}
    return tuple(dict.fromkeys(p for c in check(recipe) for p in sources[c]))


def _changed(root, pins):
    return [p for p, s in pins.items() if sha256(local_path(root, p)) != s]


def g_descriptor_default():
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    return G.DESCRIPTOR


def component_refs(root, recipe, g_descriptor=None):
    """Frozen document of every selected component, still live: (refs, pins) with pins = their sources/pins."""
    refs, pins = {}, {}
    if 'G' in recipe:
        g_descriptor = g_descriptor or g_descriptor_default()
        doc = verify(read_json(local_path(root, g_descriptor)))
        if doc.get('kind') != G_DESCRIPTOR_KIND or doc['parent_release']['checksum'] != PARENT_CHECKSUM:
            raise ValueError('G descriptor is not an atlas geometry descriptor over cd0b9910')
        if set(doc['sources_sha256']) != set(g_sources()):
            raise ValueError('G descriptor does not pin exactly the G sources')
        from rshb_vine.atlas_geometry_repair_v1 import plan as GP
        if doc['constants'] != GP.CONSTANTS or doc['policy'] != GP.POLICY:
            raise ValueError('G constants/policy differ from the frozen descriptor')
        changed = _changed(root, doc['sources_sha256'])
        if changed:
            raise ValueError('G sources changed since its descriptor: ' + ', '.join(changed))
        refs['G'] = {'path': g_descriptor, 'checksum': doc['checksum'], 'sha256': sha256(local_path(root, g_descriptor)),
                     'policy': doc['policy'], 'not_implemented': doc.get('not_implemented', [])}
        pins.update(doc['sources_sha256'])
    if 'T' in recipe:
        doc = verify(read_json(local_path(root, T_PROTOCOL)))
        if (doc.get('kind') != T_PROTOCOL_KIND or doc['checksum'] != T_PROTOCOL_CHECKSUM
                or doc.get('no_change_after_results') is not True or doc['parent']['profile'] != PARENT_CHECKSUM
                or doc['parent']['recipe'][-1:] != PARENT_RECIPE):
            raise ValueError('T protocol is not the sealed d9bf34ed over cd0b9910 recipe I')
        missing = [p for p in T_SOURCES if p not in doc['pins_sha256']]
        if missing:
            raise ValueError('T protocol does not pin ' + ', '.join(missing))
        changed = _changed(root, doc['pins_sha256'])
        if changed:
            raise ValueError('T protocol pins changed: ' + ', '.join(changed[:3]))
        refs['T'] = {'path': T_PROTOCOL, 'checksum': doc['checksum'], 'sha256': sha256(local_path(root, T_PROTOCOL)),
                     'composite_checksum': doc['component']['composite_checksum'],
                     'frozen_index_checksum': doc['parent']['frozen_index_checksum'],
                     'parent_index_checksum': doc['parent']['served_index_checksum']}
        pins.update(doc['pins_sha256'])
    return refs, pins


def install(release, recipe, root, refs):
    """Install the recipe once on ``release`` (EvidenceConsistencyRelease or its F11 copy); returns the record."""
    recipe = check(recipe)
    if tuple(release.recipe) != tuple(PARENT_RECIPE):
        raise ValueError('Parent graph recipe is not I: %r' % (release.recipe,))
    record = {'recipe': list(recipe), 'parent_recipe': list(release.recipe)}
    recovery = None
    if 'G' in recipe:
        from rshb_vine.atlas_geometry_repair_v1 import runtime as G
        recovery, geometry = G.install(release)
        record['geometry'] = dict(deepcopy(geometry), descriptor_checksum=refs['G']['checksum'])
    if 'T' in recipe:
        from rshb_vine.atlas_text_repair_v1 import composition as TC
        text = TC.install(release, root)
        identity = text['atlas_text_repair']
        expected = refs['T']
        if (identity['composite_checksum'] != expected['composite_checksum']
                or identity['frozen_index_checksum'] != expected['frozen_index_checksum']
                or identity['parent_index_checksum'] != expected['parent_index_checksum']):
            raise ValueError('Installed T index differs from the sealed T protocol')
        if text['owners'] != OWNERS:
            raise ValueError('T is not installed on all five selector owners')
        record['text'] = dict(deepcopy(text), protocol_checksum=expected['checksum'])
    return record, recovery


def run(recipe, recognize, data, roi, bottles, capture=False):
    """(result, G scope): G's observe-then-armed runner over ``recognize`` when G is selected, else one plain call."""
    if 'G' not in recipe:
        return recognize(data, roi, bottles), None
    from rshb_vine.atlas_geometry_repair_v1 import runtime as G
    return G.recognize_with_recovery(recognize, data, roi, bottles, capture=capture)


def text_traces(result):
    """The per-target T traces the live selector published (systemic_ranking_v2 records)."""
    traces = []
    for record in (result.get('systemic_ranking_v2') or {}).get('targets') or []:
        for holder in (record, record.get('systemic_ranking_v2') or {}):
            trace = holder.get('atlas_text_repair_v1') if isinstance(holder, dict) else None
            if isinstance(trace, dict):
                traces.append({'instance_id': str(record.get('instance_id')), 'status': trace.get('status'),
                               'applied': bool(trace.get('applied')), 'counts': deepcopy(trace.get('counts')),
                               'admitted': deepcopy(trace.get('admitted')), 'skip_reasons': deepcopy(trace.get('skip_reasons'))})
                break
    return traces


def block(recipe, geometry_scope, result, **identity):
    """Public block: recipe, identity/lineage, G events and T per-target traces; slug/probability stay None."""
    out = {**identity, 'recipe': list(recipe), 'parent_recipe': list(PARENT_RECIPE), 'calibrated': False, 'probability': None}
    if 'G' in recipe:
        from rshb_vine.atlas_geometry_repair_v1 import runtime as G
        out['geometry'] = G.trace(geometry_scope)
    if 'T' in recipe:
        traces = text_traces(result)
        records = (result.get('systemic_ranking_v2') or {}).get('targets') or []
        if len(traces) != len(records):
            raise RuntimeError('A published selector record lacks the T trace')
        out['text'] = {'targets': traces, 'applied': sorted(t['instance_id'] for t in traces if t['applied'])}
    return out
