"""Atlas repair candidate: the immutable F10 release cd0b9910 under live F10/C10 plus a root-named G/T recipe.

The parent is loaded from its immutable profile file (never the current pointer) exactly as factory F10 loads it; the
recipe is installed once through ``recipe.install``, the same code the F11 release uses. A descriptor is frozen in one
of two modes, each naming a root document and the frozen G descriptor / T protocol checksums it covers:

  diagnostic    root diagnostic protocol (kind atlas-repair-v1-root-diagnostic-protocol): servable for root HTTP/replay,
                ``release_admissible`` false, never stageable;
  quality_gate  root gate (kind atlas-repair-v1-root-gate, decision quality_gate_passed): the only mode stage accepts.

``draft`` previews both components and is never servable. Every field in which cd0b9910/be6e2a50 name themselves
carries the descriptor and candidate runtime checksums; cd0b9910 and its lineage stay under lineage keys only, and a
response naming them anywhere else fails closed. Slug and probability stay None. The candidate is never activated here.
"""
from pathlib import Path

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.atlas_repair_release_v1 import recipe as K

DESCRIPTOR_KIND = 'atlas-repair-release-v1-candidate-descriptor'
DRAFT_KIND = 'atlas-repair-release-v1-candidate-draft'
GATE_KIND = 'atlas-repair-v1-root-gate'
DIAGNOSTIC_KIND = 'atlas-repair-v1-root-diagnostic-protocol'
PARENT_PROFILE = 'config/evidence-consistency-release-v1-profile.json'
PARENT_MANIFEST = 'config/evidence-consistency-release-v1-manifest.json'
PARENT_CHECKSUM = K.PARENT_CHECKSUM
PARENT_RUNTIME = 'be6e2a50d9703d1d2d0607caf214dffb259c4f8890cc0ebb61c06083c067e985'
PARENT_STATE = 'evidence_consistency_active'
LIVE_CODE = ('rshb_vine/recognition_factory.py', 'scripts/recognize.py', 'rshb_vine/recognition_receipts.py',
             'config/recognition-current.json')
OUT = 'runs/atlas-repair-v1/release'
DESCRIPTOR = OUT + '/candidate/descriptor.json'
DIAGNOSTIC_DESCRIPTOR = OUT + '/candidate/diagnostic-%s.json'
DRAFT = OUT + '/candidate/draft.json'
PORT = 8222
FORBIDDEN_PORTS = (8175, 8187, 8196, 8197, 8198, 8199, 8221)
SOURCES = (f'{K.PACKAGE}/__init__.py', f'{K.PACKAGE}/recipe.py', f'{K.PACKAGE}/candidate.py',
           'scripts/atlas_repair_release_v1.py')
MODES = {'diagnostic': DIAGNOSTIC_KIND, 'quality_gate': GATE_KIND}


def check_parent(root):
    """cd0b9910 bytes, its staged manifest and the live F10 state (factory, recognize, receipts, pointer) it runs under."""
    root = Path(root).resolve()
    profile = verify(read_json(root / PARENT_PROFILE))
    manifest = verify(read_json(root / PARENT_MANIFEST))
    if profile['checksum'] != PARENT_CHECKSUM or manifest['profiles']['release']['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Parent release is not cd0b9910')
    changed = [p for p, s in manifest['files_sha256_at_stage'].items() if sha256(root / p) != s]
    if changed:
        raise ValueError('cd0b9910 release files changed since its stage: ' + ', '.join(changed[:3]))
    live = [sha256(root / p) for p in LIVE_CODE]
    if live != manifest['states'][PARENT_STATE]:
        raise ValueError('Live factory/recognize/receipts/pointer is not the F10 state of cd0b9910')
    return {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM, 'sha256': sha256(root / PARENT_PROFILE),
            'runtime_checksum': PARENT_RUNTIME, 'recipe': list(profile['evidence_consistency']['recipe']),
            'manifest': {'path': PARENT_MANIFEST, 'checksum': manifest['checksum'], 'sha256': sha256(root / PARENT_MANIFEST)},
            'live_state': dict(zip(LIVE_CODE, live))}


def bound_parent(root):
    """Model and registry of cd0b9910 as the receipts binder resolves them (no model loaded)."""
    from rshb_vine.recognition_receipts import bind_profile
    bound = bind_profile(root, PARENT_CHECKSUM, sha256(Path(root) / PARENT_PROFILE))
    return {'selector_model_checksum': bound.selector_model_checksum, 'product_bundle': bound.product_bundle,
            'bundle_checksum': bound.bundle_checksum, 'baseline_runtime_checksum': bound.runtime_checksum}


def sources_sha(root, recipe=K.COMPONENTS):
    root = Path(root)
    return {p: sha256(root / p) for p in (*SOURCES, *K.component_sources(recipe))}


def check_admission(root, mode, path):
    """The root document of ``mode``: its recipe and the G/T frozen-document checksums it names, still live."""
    root = Path(root).resolve()
    if mode not in MODES:
        raise ValueError('Unknown admission mode: ' + str(mode))
    doc = verify(read_json(local_path(root, path)))
    if doc.get('kind') != MODES[mode] or doc.get('parent_checksum') != PARENT_CHECKSUM:
        raise ValueError('%s is not a %s over cd0b9910' % (path, MODES[mode]))
    if mode == 'quality_gate' and doc.get('decision') != 'quality_gate_passed':
        raise ValueError('Root gate does not pass this recipe')
    if mode == 'diagnostic' and doc.get('no_change_after_results') is not True:
        raise ValueError('Root diagnostic protocol is not sealed before results')
    recipe = K.check(doc['recipe'])
    named = doc.get('components') or {}
    refs, pins = K.component_refs(root, recipe, (named.get('G') or {}).get('path'))
    wrong = [c for c in recipe if (named.get(c) or {}).get('checksum') != refs[c]['checksum']]
    if wrong or set(named) - set(recipe):
        raise ValueError('Root document does not name the frozen G descriptor / T protocol of: ' + ', '.join(wrong or named))
    ref = {'mode': mode, 'path': path, 'checksum': doc['checksum'], 'sha256': sha256(local_path(root, path))}
    return recipe, ref, refs, pins


def _body(root, recipe):
    return {'parent_release': check_parent(root), 'bound_parent': bound_parent(root),
            'sources_sha256': sources_sha(root, recipe), 'port': PORT, 'activated': False, 'calibrated': False,
            'probability': None, 'public_slug': None, 'weights_changed': False, 'fit_run': False,
            'rollback': 'stop %d; 8175, 8187, factory, receipts and config/recognition-current.json are never modified' % PORT}


def draft(root, g_descriptor=None):
    """Pre-admission preview over both components: not servable, not a descriptor, no admission implied."""
    root = Path(root).resolve()
    refs, blocked = {}, {}
    for component in K.COMPONENTS:
        try:
            refs.update(K.component_refs(root, (component,), g_descriptor)[0])
        except (OSError, ValueError, KeyError) as error:
            blocked[component] = str(error)
    return seal({'kind': DRAFT_KIND, 'recipe': None, 'components_available': sorted(refs), 'components': refs,
                 'components_blocked': blocked,
                 'admission': None, 'servable': False, 'root_documents': MODES, **_body(root, K.COMPONENTS)})


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
        raise ValueError('Unsupported atlas repair candidate descriptor (a draft is never servable)')
    recipe = K.check(descriptor['recipe'])
    if check_parent(root) != descriptor['parent_release']:
        raise ValueError('Parent release or live F10 state differs from the descriptor')
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


def relabel(result, checksum, runtime_checksum, parent_checksum, parent_runtime, *, admitted):
    """Own checksum wherever the parent release named itself; the parent and its lineage only under lineage keys."""
    from rshb_vine.evidence_consistency_release_v1 import recipe as EC
    from rshb_vine.text_evidence_repair_v2 import runtime as T
    from rshb_vine.text_evidence_repair_v2.release import BLOCK as TE_BLOCK
    ec, te = result[EC.BLOCK], result[TE_BLOCK]
    old = list(ec.get('parent_lineage') or [])
    for block in (ec, te):
        block.update(profile_checksum=checksum, runtime_checksum=runtime_checksum,
                     parent_release_profile_checksum=parent_checksum, parent_runtime_checksum=parent_runtime,
                     release_admitted=admitted)
    lineage = T.relabel(result, checksum, runtime_checksum, parent_checksum, parent_runtime, admitted=admitted)
    if lineage[1:] != old:
        raise RuntimeError('Parent lineage of the response differs from its evidence-consistency block')
    ec['parent_lineage'] = te['parent_lineage'] = lineage
    return lineage


class AtlasRepairCandidate:
    """cd0b9910 + G/T recipe on loopback 8222 under the live factory F10; never activated from here."""

    def __init__(self, root, descriptor_path=DESCRIPTOR, capture=False):
        from rshb_vine.evidence_consistency_release_v1.release import EvidenceConsistencyRelease
        root = Path(root).resolve()
        self.descriptor = load_descriptor(root, descriptor_path)
        self.recipe = tuple(self.descriptor['recipe'])
        release = EvidenceConsistencyRelease(root, PARENT_PROFILE)
        if release.profile['checksum'] != PARENT_CHECKSUM or release.manifest['checksum'] != PARENT_RUNTIME:
            raise ValueError('Loaded parent release differs from cd0b9910/be6e2a50')
        self.installation, self.recovery = K.install(release, self.recipe, root, self.descriptor['components'])
        self.release, self.profile, self.capture, self.last_captured = release, self.descriptor, capture, None
        self.manifest = seal({'kind': 'atlas-repair-release-v1-candidate-runtime',
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': PARENT_RUNTIME, 'parent_release_profile_checksum': PARENT_CHECKSUM,
                              'recipe': list(self.recipe), 'admission_mode': self.descriptor['admission']['mode'],
                              'installation': self.installation, 'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        result, geometry_scope = K.run(self.recipe, self.release.recognize, data, roi, bottles, self.capture)
        self.last_captured = geometry_scope and geometry_scope['captured']
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        own, runtime = self.descriptor['checksum'], self.manifest['checksum']
        lineage = relabel(result, own, runtime, PARENT_CHECKSUM, PARENT_RUNTIME, admitted=False)
        result[K.BLOCK] = K.block(self.recipe, geometry_scope, result, stage='candidate', profile_checksum=own,
                                  runtime_checksum=runtime, descriptor_checksum=own,
                                  admission_mode=self.descriptor['admission']['mode'],
                                  components={c: self.descriptor['components'][c]['checksum'] for c in self.recipe},
                                  parent_release_profile_checksum=PARENT_CHECKSUM, parent_runtime_checksum=PARENT_RUNTIME,
                                  parent_lineage=lineage, release_admitted=False)
        return result
