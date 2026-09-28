"""Evidence-consistency candidate: the static fa317ef2 TextEvidenceRelease under live F9 plus a root-admitted recipe.

The parent is loaded from its immutable profile file (never the current pointer) exactly as factory F9 loads it; the
recipe is installed once through ``recipe.install``, the same code the F10 release uses. Every field in which
fa317ef2 names itself carries the candidate descriptor checksum and every runtime checksum the candidate runtime
manifest; fa317ef2 and 24b67962 stay only under explicit parent-lineage keys, and a response naming them anywhere else
fails closed. Raw OCR, detections, ROI/all/compact and the target contract are those of the graph. Slug and
probability stay None. The candidate is never activated from here.
"""
from pathlib import Path

from rshb_vine.io import local_path, read_json, seal, sha256, verify
from rshb_vine.evidence_consistency_release_v1 import recipe as K

DESCRIPTOR_KIND = 'evidence-consistency-release-v1-candidate-descriptor'
DRAFT_KIND = 'evidence-consistency-release-v1-candidate-draft'
GATE_KIND = 'evidence-consistency-v1-root-gate'
DC_PROTOCOL_KIND = 'decision-consistency-v1-protocol'
RR_DESCRIPTOR_KIND = 'reading-recovery-v1-candidate-descriptor'
PARENT_PROFILE = 'config/text-evidence-release-v2-profile.json'
PARENT_MANIFEST = 'config/text-evidence-release-v2-manifest.json'
PARENT_CHECKSUM = 'fa317ef284b0cc0ffc93e38a7297f4ed45139fb7e544d2888d53e246c92ff62a'
PARENT_STATE = 'text_evidence_active'
LIVE_CODE = ('rshb_vine/recognition_factory.py', 'scripts/recognize.py', 'rshb_vine/recognition_receipts.py',
             'config/recognition-current.json')
OUT = 'runs/evidence-consistency-v1/integration'
DESCRIPTOR = OUT + '/candidate/descriptor.json'
DRAFT = OUT + '/draft-descriptor.json'
PORT = 8199
FORBIDDEN_PORTS = (8175, 8187, 8196, 8197, 8198)
SOURCES = (f'{K.PACKAGE}/__init__.py', f'{K.PACKAGE}/recipe.py', f'{K.PACKAGE}/candidate.py',
           'scripts/evidence_consistency_release_v1.py')


def check_parent(root):
    """fa317ef2 bytes, its staged manifest and the live F9 state (factory, recognize, receipts, pointer) it runs under."""
    root = Path(root).resolve()
    profile = verify(read_json(root / PARENT_PROFILE))
    manifest = verify(read_json(root / PARENT_MANIFEST))
    if profile['checksum'] != PARENT_CHECKSUM or manifest['profiles']['release']['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Parent release is not fa317ef2')
    changed = [p for p, s in manifest['files_sha256_at_stage'].items() if sha256(root / p) != s]
    if changed:
        raise ValueError('fa317ef2 release files changed since its stage: ' + ', '.join(changed[:3]))
    live = [sha256(root / p) for p in LIVE_CODE]
    if live != manifest['states'][PARENT_STATE]:
        raise ValueError('Live factory/recognize/receipts/pointer is not the F9 state of fa317ef2')
    return {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM, 'sha256': sha256(root / PARENT_PROFILE),
            'manifest': {'path': PARENT_MANIFEST, 'checksum': manifest['checksum'], 'sha256': sha256(root / PARENT_MANIFEST)},
            'live_state': dict(zip(LIVE_CODE, live))}


def bound_parent(root):
    """Model and registry of fa317ef2 as the receipts binder resolves them (no model loaded)."""
    from rshb_vine.recognition_receipts import bind_profile
    bound = bind_profile(root, PARENT_CHECKSUM, sha256(Path(root) / PARENT_PROFILE))
    return {'selector_model_checksum': bound.selector_model_checksum, 'product_bundle': bound.product_bundle,
            'bundle_checksum': bound.bundle_checksum, 'baseline_runtime_checksum': bound.runtime_checksum}


def _ref(root, path, kind):
    doc = verify(read_json(local_path(root, path)))
    if doc.get('kind') != kind:
        raise ValueError('%s is not a %s' % (path, kind))
    return doc, {'path': path, 'checksum': doc['checksum'], 'sha256': sha256(local_path(root, path))}


def _live_pins(root, pins, what):
    changed = [p for p, s in pins.items() if sha256(local_path(root, p)) != s]
    if changed:
        raise ValueError(what + ' changed: ' + ', '.join(changed[:3]))
    return dict(pins)


def check_gate(root, gate_path):
    """The root gate naming the recipe, with the sealed protocol/descriptors of each selected component still live."""
    root = Path(root).resolve()
    gate, gate_ref = _ref(root, gate_path, GATE_KIND)
    if gate.get('decision') != 'quality_gate_passed' or gate.get('parent_checksum') != PARENT_CHECKSUM:
        raise ValueError('Root gate does not pass a recipe over fa317ef2')
    recipe = K.check(gate['recipe'])
    admission, pins = {}, {}
    if {'I', 'G'} & set(recipe):
        ref = gate.get('decision_consistency_protocol') or {}
        protocol, own = _ref(root, ref.get('path', ''), DC_PROTOCOL_KIND)
        if own['checksum'] != ref.get('checksum') or protocol.get('no_change_after_results') is not True:
            raise ValueError('Gate does not bind the sealed decision-consistency protocol')
        missing = [p for p in K.DC_SOURCES if p not in protocol['pins_sha256']]
        if missing:
            raise ValueError('Decision-consistency protocol does not pin ' + ', '.join(missing))
        admission['decision_consistency_protocol'] = own
        pins.update(_live_pins(root, protocol['pins_sha256'], 'Decision-consistency protocol pins'))
    for component in ('O', 'R'):
        if component not in recipe:
            continue
        ref = (gate.get('reading_recovery_descriptors') or {}).get(component) or {}
        descriptor, own = _ref(root, ref.get('path', ''), RR_DESCRIPTOR_KIND)
        if (own['checksum'] != ref.get('checksum') or descriptor.get('component') != component
                or descriptor.get('parent_checksum') != PARENT_CHECKSUM):
            raise ValueError('Gate does not bind the frozen reading-recovery %s descriptor over fa317ef2' % component)
        admission.setdefault('reading_recovery_descriptors', {})[component] = own
        pins.update(_live_pins(root, descriptor['sources_sha256'], 'Reading-recovery %s sources' % component))
    return gate, gate_ref, recipe, admission, pins


def sources_sha(root, recipe=K.COMPONENTS):
    root = Path(root)
    return {p: sha256(root / p) for p in (*SOURCES, *K.component_sources(recipe))}


def _body(root, recipe):
    return {'parent_release': check_parent(root), 'bound_parent': bound_parent(root),
            'sources_sha256': sources_sha(root, recipe), 'port': PORT, 'activated': False, 'calibrated': False,
            'probability': None, 'public_slug': None, 'weights_changed': False, 'fit_run': False,
            'rollback': 'stop %d; 8175, 8187, factory, receipts and config/recognition-current.json are never modified' % PORT}


def draft(root):
    """Pre-gate preview over all four components: not servable, not a descriptor, no admission implied."""
    root = Path(root).resolve()
    return seal({'kind': DRAFT_KIND, 'recipe': None, 'components_available': list(K.COMPONENTS), 'gate': None,
                 'servable': False, 'root_gate_required': GATE_KIND, **_body(root, K.COMPONENTS)})


def freeze(root, gate_path):
    root = Path(root).resolve()
    gate, gate_ref, recipe, admission, pins = check_gate(root, gate_path)
    body = _body(root, recipe)
    clash = [p for p, s in body['sources_sha256'].items() if pins.get(p, s) != s]
    if clash:
        raise ValueError('Admission pins disagree with live sources: ' + ', '.join(clash[:3]))
    return seal({'kind': DESCRIPTOR_KIND, 'recipe': list(recipe), 'gate': gate_ref, 'admission': admission,
                 'pins_sha256': pins, **body})


def admission_refs(descriptor):
    admission = descriptor['admission']
    protocol = [admission['decision_consistency_protocol']] if 'decision_consistency_protocol' in admission else []
    return [descriptor['gate'], *protocol, *admission.get('reading_recovery_descriptors', {}).values()]


def load_descriptor(root, path=DESCRIPTOR):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if descriptor.get('kind') != DESCRIPTOR_KIND:
        raise ValueError('Unsupported evidence-consistency candidate descriptor (a draft is never servable)')
    recipe = K.check(descriptor['recipe'])
    if check_parent(root) != descriptor['parent_release']:
        raise ValueError('Parent release or live F9 state differs from the descriptor')
    for ref in admission_refs(descriptor):
        if sha256(local_path(root, ref['path'])) != ref['sha256']:
            raise ValueError('Admission file changed since freeze: ' + ref['path'])
    _live_pins(root, {**descriptor['pins_sha256'], **descriptor['sources_sha256']}, 'Candidate pins or sources')
    if sources_sha(root, recipe) != descriptor['sources_sha256']:
        raise ValueError('Candidate source set differs from the descriptor')
    return descriptor


def relabel(result, checksum, runtime_checksum, parent_checksum, parent_runtime, *, admitted):
    """Own checksum wherever the parent release named itself; the parent and its lineage only under lineage keys."""
    from rshb_vine.text_evidence_repair_v2 import runtime as T
    from rshb_vine.text_evidence_repair_v2.release import BLOCK as TE_BLOCK
    result[TE_BLOCK].update(profile_checksum=checksum, runtime_checksum=runtime_checksum,
                            parent_release_profile_checksum=parent_checksum, parent_runtime_checksum=parent_runtime,
                            release_admitted=admitted)
    lineage = T.relabel(result, checksum, runtime_checksum, parent_checksum, parent_runtime, admitted=admitted)
    result[TE_BLOCK]['parent_lineage'] = lineage
    return lineage


class EvidenceConsistencyCandidate:
    """fa317ef2 + admitted recipe on loopback 8199 under the live factory F9; never activated from here."""

    def __init__(self, root, descriptor_path=DESCRIPTOR):
        from rshb_vine.text_evidence_repair_v2.release import TextEvidenceRelease
        root = Path(root).resolve()
        self.descriptor = load_descriptor(root, descriptor_path)
        self.recipe = tuple(self.descriptor['recipe'])
        release = TextEvidenceRelease(root, PARENT_PROFILE)
        if release.profile['checksum'] != PARENT_CHECKSUM:
            raise ValueError('Loaded parent release differs from fa317ef2')
        self.installation = K.install(release, self.recipe, root)
        self.release, self.profile = release, self.descriptor
        self.manifest = seal({'kind': 'evidence-consistency-release-v1-candidate-runtime',
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': release.manifest['checksum'],
                              'parent_release_profile_checksum': PARENT_CHECKSUM, 'recipe': list(self.recipe),
                              'installation': self.installation, 'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        with K.scope(self.recipe, roi, bottles) as route_scope:
            result = self.release.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        entries = K.label_view_contract(result, self.recipe)
        own, runtime = self.descriptor['checksum'], self.manifest['checksum']
        lineage = relabel(result, own, runtime, PARENT_CHECKSUM, self.release.manifest['checksum'], admitted=False)
        result[K.BLOCK] = K.block(result, self.recipe, route_scope, entries, stage='candidate', profile_checksum=own,
                                  runtime_checksum=runtime, descriptor_checksum=own,
                                  parent_release_profile_checksum=PARENT_CHECKSUM,
                                  parent_runtime_checksum=self.release.manifest['checksum'], parent_lineage=lineage,
                                  release_admitted=False)
        return result
