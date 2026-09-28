"""Release-next candidate: the immutable F11 release bde4fa52 under live F11/C11 plus a root-named D/N recipe.

The parent is loaded from its immutable profile file (never the current pointer) exactly as factory F11 loads it; the
recipe is installed once through ``recipe.install``, the same code the release uses. A descriptor is frozen in one of
two modes, each naming a root document and the component descriptor checksums it covers:

  diagnostic    root diagnostic protocol (kind release-next-v1-root-diagnostic-protocol): servable for root HTTP/replay,
                ``release_admissible`` false, never stageable;
  quality_gate  root gate (kind release-next-v1-root-gate, decision quality_gate_passed): the only mode stage accepts.

``draft`` previews available components and is never servable. Every field in which bde4fa52/54f09a20 name
themselves carries the descriptor and candidate runtime checksums; bde4fa52 and its lineage stay under lineage keys
only, and a response naming them anywhere else fails closed. Slug and probability stay None. Never activated here.
"""
from pathlib import Path

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.release_next_v1 import recipe as K

DESCRIPTOR_KIND = 'release-next-v1-candidate-descriptor'
DRAFT_KIND = 'release-next-v1-candidate-draft'
GATE_KIND = 'release-next-v1-root-gate'
DIAGNOSTIC_KIND = 'release-next-v1-root-diagnostic-protocol'
PARENT_PROFILE = 'config/atlas-repair-release-v1-profile.json'
PARENT_MANIFEST = 'config/atlas-repair-release-v1-manifest.json'
PARENT_MANIFEST_CHECKSUM = 'b56df82ee0502e62b3c50a4dd78eaed6e1c8a228d2e573f27451f27013df3bcc'
PARENT_CHECKSUM = K.PARENT_CHECKSUM
PARENT_RUNTIME = K.PARENT_RUNTIME
PARENT_STATE = 'atlas_repair_active'
LIVE_CODE = ('rshb_vine/recognition_factory.py', 'scripts/recognize.py', 'rshb_vine/recognition_receipts.py',
             'config/recognition-current.json')
OUT = 'runs/release-next-v1/integration'
DESCRIPTOR = OUT + '/candidate/descriptor.json'
DIAGNOSTIC_DESCRIPTOR = OUT + '/candidate/diagnostic-%s.json'
DRAFT = OUT + '/candidate/draft.json'
PORT = 8225
FORBIDDEN_PORTS = (8175, 8187, 8196, 8197, 8198, 8199, 8221, 8222, 8223, 8224)
SOURCES = (f'{K.PACKAGE}/__init__.py', f'{K.PACKAGE}/recipe.py', f'{K.PACKAGE}/candidate.py',
           'scripts/release_next_v1.py')
MODES = {'diagnostic': DIAGNOSTIC_KIND, 'quality_gate': GATE_KIND}


def check_parent(root):
    """bde4fa52 bytes, its staged manifest and the live F11 state (factory, recognize, receipts, pointer) it runs under."""
    root = Path(root).resolve()
    profile = verify(read_json(root / PARENT_PROFILE))
    manifest = verify(read_json(root / PARENT_MANIFEST))
    if (profile['checksum'] != PARENT_CHECKSUM or manifest['checksum'] != PARENT_MANIFEST_CHECKSUM
            or manifest['profiles']['release']['checksum'] != PARENT_CHECKSUM):
        raise ValueError('Parent release is not bde4fa52 of manifest b56df82e')
    changed = [p for p, s in manifest['files_sha256_at_stage'].items() if sha256(root / p) != s]
    if changed:
        raise ValueError('bde4fa52 release files changed since its stage: ' + ', '.join(changed[:3]))
    live = [sha256(root / p) for p in LIVE_CODE]
    if live != manifest['states'][PARENT_STATE]:
        raise ValueError('Live factory/recognize/receipts/pointer is not the F11 state of bde4fa52')
    return {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM, 'sha256': sha256(root / PARENT_PROFILE),
            'runtime_checksum': PARENT_RUNTIME, 'atlas_recipe': list(profile['atlas_repair']['recipe']),
            'recipe': list(profile['evidence_consistency']['recipe']),
            'manifest': {'path': PARENT_MANIFEST, 'checksum': manifest['checksum'], 'sha256': sha256(root / PARENT_MANIFEST)},
            'live_state': dict(zip(LIVE_CODE, live))}


def bound_parent(root):
    """Model and registry of bde4fa52 as the receipts binder resolves them (no model loaded)."""
    from rshb_vine.recognition_receipts import bind_profile
    bound = bind_profile(root, PARENT_CHECKSUM, sha256(Path(root) / PARENT_PROFILE))
    return {'selector_model_checksum': bound.selector_model_checksum, 'product_bundle': bound.product_bundle,
            'bundle_checksum': bound.bundle_checksum, 'baseline_runtime_checksum': bound.runtime_checksum}


def sources_sha(root, recipe=K.COMPONENTS):
    root = Path(root)
    return {p: sha256(root / p) for p in (*SOURCES, *(K.component_sources(recipe) if recipe else ()))}


def check_admission(root, mode, path):
    """The root document of ``mode``: its recipe and the component descriptors it names, still live."""
    root = Path(root).resolve()
    if mode not in MODES:
        raise ValueError('Unknown admission mode: ' + str(mode))
    doc = verify(read_json(local_path(root, path)))
    if doc.get('kind') != MODES[mode] or doc.get('parent_checksum') != PARENT_CHECKSUM:
        raise ValueError('%s is not a %s over bde4fa52' % (path, MODES[mode]))
    if mode == 'quality_gate' and doc.get('decision') != 'quality_gate_passed':
        raise ValueError('Root gate does not pass this recipe')
    if mode == 'diagnostic' and doc.get('no_change_after_results') is not True:
        raise ValueError('Root diagnostic protocol is not sealed before results')
    recipe = K.check(doc['recipe'])
    named = doc.get('components') or {}
    if set(named) != set(recipe):
        raise ValueError('Root document must name exactly the descriptors of recipe %r' % (recipe,))
    refs, pins = K.component_refs(root, recipe, {c: named[c].get('path') for c in recipe})
    wrong = [c for c in recipe if named[c].get('checksum') != refs[c]['checksum']]
    if wrong:
        raise ValueError('Root document does not name the live component descriptor of: ' + ', '.join(wrong))
    ref = {'mode': mode, 'path': path, 'checksum': doc['checksum'], 'sha256': sha256(local_path(root, path))}
    return recipe, ref, refs, pins


def _body(root, recipe):
    return {'parent_release': check_parent(root), 'bound_parent': bound_parent(root),
            'sources_sha256': sources_sha(root, recipe), 'port': PORT, 'activated': False, 'calibrated': False,
            'probability': None, 'public_slug': None, 'weights_changed': False, 'fit_run': False,
            'rollback': 'stop %d; 8175, 8187, factory, receipts and config/recognition-current.json are never modified' % PORT}


def draft(root, paths=None):
    """Pre-admission preview: not servable, not a descriptor, no admission implied; unavailable parts are listed."""
    root = Path(root).resolve()
    refs, blocked = {}, {}
    for component in K.COMPONENTS:
        try:
            K.owner(component)
            refs.update(K.component_refs(root, (component,), paths)[0])
        except (ImportError, OSError, ValueError, KeyError) as error:
            blocked[component] = '%s: %s' % (type(error).__name__, error)
    available = tuple(c for c in K.COMPONENTS if c in refs)
    return seal({'kind': DRAFT_KIND, 'recipe': None, 'components_available': list(available), 'components': refs,
                 'components_blocked': blocked, 'admission': None, 'servable': False, 'root_documents': MODES,
                 **_body(root, available)})


def freeze(root, mode, path):
    root = Path(root).resolve()
    recipe, ref, refs, pins = check_admission(root, mode, path)
    body = _body(root, recipe)
    clash = [p for p, s in body['sources_sha256'].items() if pins.get(p, s) != s]
    if clash:
        raise ValueError('Component pins disagree with live sources: ' + ', '.join(clash[:3]))
    return seal({'kind': DESCRIPTOR_KIND, 'recipe': list(recipe), 'admission': ref, 'components': refs,
                 'pins_sha256': pins, 'release_admissible': mode == 'quality_gate', 'release_admitted': False, **body})


def default_path(mode, recipe):
    return DESCRIPTOR if mode == 'quality_gate' else DIAGNOSTIC_DESCRIPTOR % ''.join(recipe)


def admission_refs(descriptor):
    return [descriptor['admission'], *(descriptor['components'][c] for c in descriptor['recipe'])]


def load_descriptor(root, path=DESCRIPTOR):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if descriptor.get('kind') != DESCRIPTOR_KIND:
        raise ValueError('Unsupported release-next candidate descriptor (a draft is never servable)')
    recipe = K.check(descriptor['recipe'])
    if check_parent(root) != descriptor['parent_release']:
        raise ValueError('Parent release or live F11 state differs from the descriptor')
    for ref in admission_refs(descriptor):
        if sha256(local_path(root, ref['path'])) != ref['sha256']:
            raise ValueError('Admission file changed since freeze: ' + ref['path'])
    changed = [p for p, s in {**descriptor['pins_sha256'], **descriptor['sources_sha256']}.items()
               if sha256(local_path(root, p)) != s]
    if changed:
        raise ValueError('Candidate pins or sources changed: ' + ', '.join(changed[:3]))
    if sources_sha(root, recipe) != descriptor['sources_sha256']:
        raise ValueError('Candidate source set differs from the descriptor')
    return descriptor


class ReleaseNextCandidate:
    """bde4fa52 + D/N recipe on loopback 8225 under the live factory F11; never activated from here."""

    def __init__(self, root, descriptor_path=DESCRIPTOR, capture=False):
        from rshb_vine.atlas_repair_release_v1.release import AtlasRepairRelease
        root = Path(root).resolve()
        self.descriptor = load_descriptor(root, descriptor_path)
        self.recipe = tuple(self.descriptor['recipe'])
        graph = AtlasRepairRelease(root, PARENT_PROFILE)
        if graph.profile['checksum'] != PARENT_CHECKSUM or graph.manifest['checksum'] != PARENT_RUNTIME:
            raise ValueError('Loaded parent release differs from bde4fa52/54f09a20')
        self.states, self.installation = K.install(graph, self.recipe, root, self.descriptor['components'])
        self.graph, self.profile, self.capture, self.last_scope = graph, self.descriptor, capture, None
        self.manifest = seal({'kind': 'release-next-v1-candidate-runtime',
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': PARENT_RUNTIME, 'parent_release_profile_checksum': PARENT_CHECKSUM,
                              'recipe': list(self.recipe), 'admission_mode': self.descriptor['admission']['mode'],
                              'installation': self.installation, 'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        result, scope = K.run(self.recipe, self.states, self.graph.recognize, data, roi, bottles, self.capture)
        self.last_scope = scope if self.capture else None
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        own, runtime = self.descriptor['checksum'], self.manifest['checksum']
        lineage = K.relabel(result, own, runtime, admitted=False)
        result[K.BLOCK] = K.block(self.recipe, self.descriptor['components'], self.installation, scope, stage='candidate',
                                  profile_checksum=own, runtime_checksum=runtime, descriptor_checksum=own,
                                  admission_mode=self.descriptor['admission']['mode'],
                                  parent_release_profile_checksum=PARENT_CHECKSUM, parent_runtime_checksum=PARENT_RUNTIME,
                                  parent_lineage=lineage, release_admitted=False)
        K.fail_closed(result, (PARENT_CHECKSUM, PARENT_RUNTIME))
        return result
