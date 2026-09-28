"""The one installation of a root-named D/N recipe on a loaded bde4fa52 graph, shared by candidate and release.

D is ``rshb_vine.reference_additions_v1.component`` (gallery/text eligibility of source-verified references); N is
``rshb_vine.existing_target_parent_v1.component`` (physical parent beside an existing target). Each owner is bound
only by its own sealed component descriptor (kind, parent bde4fa52, policy, constants, sources and data pins, all
live-verified here). Owner contract:

  KIND, SOURCES, load_descriptor(root, path) -> sealed dict
  install(graph, root, descriptor) -> (state, record)            instance-level ownership only, once per graph
  N only: run(state, recognize, data, roi, bottles, capture) -> (result, scope); trace(scope) -> dict

``graph`` is the loaded AtlasRepairRelease (candidate) or ReleaseNextRelease (release); D installs before N. The
atlas G/T recipe of bde4fa52 stays as installed. Nothing global, pinned or module-level is patched here.
"""
from copy import deepcopy
import importlib

from rshb_vine.io import local_path, read_json, sha256, verify

COMPONENTS = ('D', 'N')
OWNERS = {'D': 'rshb_vine.reference_additions_v1.component', 'N': 'rshb_vine.existing_target_parent_v1.component'}
REQUIRED = {'D': ('KIND', 'SOURCES', 'load_descriptor', 'install'),
            'N': ('KIND', 'SOURCES', 'load_descriptor', 'install', 'run', 'trace')}
PACKAGE = 'rshb_vine/release_next_v1'
BLOCK = 'release_next_v1'
PARENT_CHECKSUM = 'bde4fa52d35b1c297e43996cacf11db2e00558a0273586639b3e01923ab468f3'
PARENT_RUNTIME = '54f09a202c69d8c2b0afd0688e39736a0d40f90b75826dde184ccb8b778bd7cd'
PARENT_ATLAS_RECIPE = ('G', 'T')
LINEAGE_KEYS = frozenset({'parent_release_profile_checksum', 'parent_runtime_checksum', 'parent_lineage',
                          'predecessor_release'})
INSTALLED = '_release_next_v1_installation'


def check(recipe):
    """A non-empty subset of D, N written in that order; anything else is refused."""
    recipe = tuple(recipe or ())
    if not recipe or len(set(recipe)) != len(recipe) or set(recipe) - set(COMPONENTS):
        raise ValueError('Recipe must be a non-empty subset of D, N: %r' % (recipe,))
    if recipe != tuple(c for c in COMPONENTS if c in recipe):
        raise ValueError('Recipe must list components in the order D, N: %r' % (recipe,))
    return recipe


def parse(text):
    return check(tuple(c.strip() for c in text.split(',') if c.strip()))


def owner(component):
    module = importlib.import_module(OWNERS[component])
    missing = [name for name in REQUIRED[component] if not hasattr(module, name)]
    if missing:
        raise ValueError('%s owner %s lacks: %s' % (component, OWNERS[component], ', '.join(missing)))
    if OWNERS[component].replace('.', '/') + '.py' not in module.SOURCES:
        raise ValueError('%s owner does not list its own component module in SOURCES' % component)
    return module


def component_sources(recipe):
    return tuple(dict.fromkeys(p for c in check(recipe) for p in owner(c).SOURCES))


def _changed(root, pins):
    return [p for p, s in pins.items() if sha256(local_path(root, p)) != s]


def component_ref(root, component, path):
    """(ref, pins) of one live component descriptor; the owner's loader and the file must agree byte-for-byte."""
    module = owner(component)
    doc = verify(read_json(local_path(root, path)))
    loaded = module.load_descriptor(root, path)
    if loaded != doc:
        raise ValueError('%s owner loader returned another document than %s' % (component, path))
    if doc.get('kind') != module.KIND or (doc.get('parent_release') or {}).get('checksum') != PARENT_CHECKSUM:
        raise ValueError('%s descriptor is not a %s over bde4fa52' % (component, module.KIND))
    if 'policy' not in doc or 'constants' not in doc:
        raise ValueError('%s descriptor does not state its policy and constants' % component)
    sources = doc.get('sources_sha256') or {}
    missing = [p for p in module.SOURCES if p not in sources]
    if missing:
        raise ValueError('%s descriptor does not pin its sources: %s' % (component, ', '.join(missing)))
    pins = {**sources, **(doc.get('pins_sha256') or {})}
    clash = [p for p in set(sources) & set(doc.get('pins_sha256') or {}) if sources[p] != doc['pins_sha256'][p]]
    if clash:
        raise ValueError('%s descriptor pins disagree with its sources: %s' % (component, ', '.join(clash[:3])))
    changed = _changed(root, pins)
    if changed:
        raise ValueError('%s pins changed since its descriptor: %s' % (component, ', '.join(changed[:3])))
    ref = {'path': path, 'checksum': doc['checksum'], 'sha256': sha256(local_path(root, path)), 'kind': doc['kind'],
           'policy': doc['policy'], 'not_implemented': doc.get('not_implemented', [])}
    return ref, pins


def component_refs(root, recipe, paths):
    """Refs and merged pins of every selected component; ``paths`` maps component -> descriptor path."""
    refs, pins = {}, {}
    for component in check(recipe):
        if not (paths or {}).get(component):
            raise ValueError('No descriptor path named for component ' + component)
        ref, own = component_ref(root, component, paths[component])
        clash = [p for p, s in own.items() if pins.get(p, s) != s]
        if clash:
            raise ValueError('Component pins disagree across D/N: ' + ', '.join(clash[:3]))
        refs[component] = ref
        pins.update(own)
    return refs, pins


def install(graph, recipe, root, refs):
    """Install the recipe once on a loaded bde4fa52-equivalent graph; returns (states, record)."""
    recipe = check(recipe)
    if tuple(graph.atlas_recipe) != PARENT_ATLAS_RECIPE or tuple(graph.recipe) != ('I',):
        raise ValueError('Graph is not the bde4fa52 composition (atlas G,T over evidence-consistency I)')
    if getattr(graph, INSTALLED, None) is not None:
        raise ValueError('release_next_v1 is already installed on this graph')
    states, record = {}, {'recipe': list(recipe), 'parent_atlas_recipe': list(graph.atlas_recipe)}
    for component in recipe:
        ref = refs[component]
        if sha256(local_path(root, ref['path'])) != ref['sha256']:
            raise ValueError('Component descriptor changed since freeze: ' + ref['path'])
        descriptor = verify(read_json(local_path(root, ref['path'])))
        if descriptor['checksum'] != ref['checksum']:
            raise ValueError('Component descriptor checksum differs: ' + ref['path'])
        state, installed = owner(component).install(graph, root, descriptor)
        states[component] = state
        record[component] = dict(deepcopy(installed), descriptor_checksum=ref['checksum'])
    setattr(graph, INSTALLED, record)
    return states, record


def run(recipe, states, recognize, data, roi, bottles, capture=False):
    """(result, N scope): N's runner over the complete parent ``recognize`` when N is selected, else one plain call."""
    if 'N' not in recipe:
        return recognize(data, roi, bottles), None
    return owner('N').run(states['N'], recognize, data, roi, bottles, capture)


def block(recipe, refs, record, n_scope, **identity):
    """Public block: recipe, identity/lineage, D installation evidence, N trace; slug/probability stay None."""
    out = {**identity, 'recipe': list(recipe), 'parent_atlas_recipe': list(PARENT_ATLAS_RECIPE),
           'components': {c: refs[c]['checksum'] for c in recipe}, 'calibrated': False, 'probability': None,
           'inherited_blocks': 'algorithm provenance of bde4fa52; with D the active gallery/text eligibility is the one '
                               'reported under reference_additions'}
    if 'D' in recipe:
        out['reference_additions'] = deepcopy(record['D'])
    if 'N' in recipe:
        out['existing_target_parent'] = owner('N').trace(n_scope)
    return out


def mentions(value, needles, path=()):
    """Paths where a needle appears as a string value outside the lineage keys."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in LINEAGE_KEYS:
                yield from mentions(v, needles, (*path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from mentions(v, needles, (*path, i))
    elif isinstance(value, str) and any(n in value for n in needles):
        yield '.'.join(map(str, path))


def relabel(result, checksum, runtime_checksum, *, admitted):
    """Own checksum wherever bde4fa52/54f09a20 named themselves; bde4fa52 and its lineage only under lineage keys."""
    from rshb_vine.atlas_repair_release_v1 import candidate as AC
    from rshb_vine.atlas_repair_release_v1 import recipe as AK
    atlas = result[AK.BLOCK]
    if atlas.get('profile_checksum') != PARENT_CHECKSUM or atlas.get('runtime_checksum') != PARENT_RUNTIME:
        raise RuntimeError('Response was not produced by bde4fa52/54f09a20')
    expected = [PARENT_CHECKSUM, *(atlas.get('parent_lineage') or [])]
    # The atlas block goes first: the inherited relabel scans the whole response for the parent before returning.
    atlas.update(profile_checksum=checksum, runtime_checksum=runtime_checksum, parent_release_profile_checksum=PARENT_CHECKSUM,
                 parent_runtime_checksum=PARENT_RUNTIME, parent_lineage=expected, release_admitted=admitted)
    lineage = AC.relabel(result, checksum, runtime_checksum, PARENT_CHECKSUM, PARENT_RUNTIME, admitted=admitted)
    if lineage != expected:
        raise RuntimeError('Parent lineage of the response differs from its atlas block')
    return lineage


def fail_closed(result, needles):
    left = list(mentions(result, needles))
    if left:
        raise RuntimeError('Response names the parent outside lineage: ' + ', '.join(left[:3]))
