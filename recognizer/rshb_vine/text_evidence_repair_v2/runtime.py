"""Text-evidence release v2 candidate: current 24b67962 with the admitted producer-consistency recipe at its selector.

The zero-target release 24b67962 is loaded exactly as factory F8 loads it (ZeroTargetRelease over the dispatch8
chain, all its pins and its staged manifest verified); then the D selector of ``text_evidence_repair_v2.composition``
is installed once on that same graph, where base.runtime.selection, base.target.inner.selection, repair.selection and
inner.selection are one object. The zero-target branch (a), ROI, bottles=all and the compact API stay those of the
graph; the canvas retry of branch (a) runs through the same runtime and therefore the same selector. Recipes are only
[B, D] or [A, B, D, M] (A always with its reader preservation M); C is refused. No global, pin or module attribute is patched.

Identity: every profile field in which 24b67962 names itself (geometry/target-contract blocks, repair v1/v2 release
fields, release and zero-target profile_checksum) carries the candidate descriptor checksum; every runtime_checksum
carries the candidate runtime manifest checksum (self.manifest, distinct from the descriptor). 24b67962 and its
runtime remain only as explicit parent lineage. A response still naming either anywhere else fails closed. Public
slug and probability stay None.
"""
from copy import deepcopy
from pathlib import Path

from rshb_vine.io import local_path, read_json, seal, sha256, verify

DESCRIPTOR_KIND = 'text-evidence-release-v2-candidate-descriptor'
GATE_KIND = 'text-evidence-improve-v2-root-gate'
PARENT_PROFILE = 'config/zero-target-release-v1-profile.json'
PARENT_MANIFEST = 'config/zero-target-release-v1-manifest.json'
PARENT_CHECKSUM = '24b67962c1032c2cd9e8cd381edb54a6448484933e952e9eb9802706120b5920'
PARENT_STATE = 'zero_target_active'
LIVE_CODE = ('rshb_vine/recognition_factory.py', 'scripts/recognize.py', 'rshb_vine/recognition_receipts.py',
             'config/recognition-current.json')
OUT = 'runs/text-evidence-improve-v2/release'
DESCRIPTOR = OUT + '/candidate/descriptor.json'
PORT = 8197
FORBIDDEN_PORTS = (8175, 8187, 8196)
RECIPES = (('B', 'D'), ('A', 'B', 'D', 'M'))
SHORT_FORM_TABLE = '22a4e4f8eb605b4a164d3da5b2139456e9554e7a813c59fb11fcf51db2de1566'
PACKAGE = 'rshb_vine/text_evidence_repair_v2'
SOURCES = (f'{PACKAGE}/runtime.py', 'scripts/text_evidence_release_v2.py')
SELECTOR_SOURCES = (f'{PACKAGE}/composition.py', f'{PACKAGE}/producer_consistency.py')
PACKAGE_INIT = f'{PACKAGE}/__init__.py'
READER_SOURCE = f'{PACKAGE}/reader_preservation.py'
BLOCK = 'text_evidence_release_v2'
PARENT_KEYS = frozenset({'parent_release_profile_checksum', 'parent_runtime_checksum', 'parent_lineage', 'predecessor_release'})


def _recipe(value):
    recipe = tuple(value)
    if 'C' in recipe or recipe not in RECIPES:
        raise ValueError('Recipe must be [B, D] or [A, B, D, M]; C is never released: %r' % (value,))
    return recipe


def check_parent(root):
    """24b67962 bytes, its staged manifest and the live F8 state it runs under."""
    root = Path(root).resolve()
    profile = verify(read_json(root / PARENT_PROFILE))
    manifest = verify(read_json(root / PARENT_MANIFEST))
    if profile['checksum'] != PARENT_CHECKSUM or manifest['profiles']['release']['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Parent release is not 24b67962')
    changed = [p for p, s in manifest['files_sha256_at_stage'].items() if sha256(root / p) != s]
    if changed:
        raise ValueError('24b67962 release files changed since its stage: ' + ', '.join(changed[:3]))
    live = [sha256(root / p) for p in LIVE_CODE]
    if live != manifest['states'][PARENT_STATE]:
        raise ValueError('Live factory/recognize/receipts/pointer is not the F8 state of 24b67962')
    return {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM, 'sha256': sha256(root / PARENT_PROFILE),
            'manifest': {'path': PARENT_MANIFEST, 'checksum': manifest['checksum'], 'sha256': sha256(root / PARENT_MANIFEST)},
            'live_state': dict(zip(LIVE_CODE, live))}


def check_admission(root, protocol_path, gate_path):
    """The root-admitted v2 protocol (all pins live) and the root gate decision naming the released recipe."""
    root = Path(root).resolve()
    protocol = verify(read_json(local_path(root, protocol_path)))
    gate = verify(read_json(local_path(root, gate_path)))
    if protocol.get('no_change_after_results') is not True or not protocol.get('pins_sha256'):
        raise ValueError('v2 protocol is not a pinned pre-result protocol')
    changed = [p for p, s in protocol['pins_sha256'].items() if sha256(local_path(root, p)) != s]
    if changed:
        raise ValueError('v2 protocol pins changed: ' + ', '.join(changed[:3]))
    if (gate.get('kind') != GATE_KIND or gate.get('decision') != 'quality_gate_passed'
            or gate.get('protocol_checksum') != protocol['checksum']):
        raise ValueError('Root gate does not pass this v2 protocol')
    recipe = _recipe(gate['recipe'])
    tables = {c: protocol['component_tables'][c] for c in recipe if c in protocol.get('component_tables', {})}
    algorithm = (*SELECTOR_SOURCES, *((READER_SOURCE,) if 'M' in recipe else ()))
    missing = [p for p in algorithm if p not in protocol['pins_sha256']]
    if missing:
        raise ValueError('v2 protocol does not pin the selector algorithm: ' + ', '.join(missing))
    if tables != {'B': SHORT_FORM_TABLE, 'D': SHORT_FORM_TABLE}:
        raise ValueError('B and D must read the one admitted short-form table: %r' % (tables,))
    return protocol, gate, recipe, tables


def sources_sha(root):
    """Deploy-only files, pinned by the descriptor itself: the v2 protocol pins the algorithm and harness only."""
    root = Path(root)
    return {p: sha256(root / p) for p in (*SOURCES, PACKAGE_INIT) if (root / p).is_file() or p != PACKAGE_INIT}


def freeze(root, protocol_path, gate_path):
    root = Path(root).resolve()
    parent = check_parent(root)
    protocol, gate, recipe, tables = check_admission(root, protocol_path, gate_path)
    return seal({'kind': DESCRIPTOR_KIND, 'parent_release': parent,
                 'protocol': {'path': protocol_path, 'checksum': protocol['checksum'],
                              'sha256': sha256(local_path(root, protocol_path))},
                 'gate': {'path': gate_path, 'checksum': gate['checksum'], 'sha256': sha256(local_path(root, gate_path))},
                 'recipe': list(recipe), 'tables': tables, 'pins_sha256': dict(protocol['pins_sha256']),
                 'sources_sha256': sources_sha(root), 'port': PORT, 'activated': False, 'calibrated': False,
                 'probability': None, 'weights_changed': False, 'fit_run': False,
                 'rollback': 'stop %d; 8175, factory, receipts and config/recognition-current.json are never modified' % PORT})


def load_descriptor(root, path=DESCRIPTOR):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if descriptor.get('kind') != DESCRIPTOR_KIND:
        raise ValueError('Unsupported text-evidence v2 candidate descriptor')
    _recipe(descriptor['recipe'])
    if check_parent(root) != descriptor['parent_release']:
        raise ValueError('Parent release or live F8 state differs from the descriptor')
    for ref in (descriptor['protocol'], descriptor['gate']):
        if sha256(local_path(root, ref['path'])) != ref['sha256']:
            raise ValueError('Admission file changed since freeze: ' + ref['path'])
    changed = [p for p, s in {**descriptor['pins_sha256'], **descriptor['sources_sha256']}.items()
               if sha256(local_path(root, p)) != s]
    if changed or sources_sha(root) != descriptor['sources_sha256']:
        raise ValueError('Candidate pins or sources changed since freeze: ' + ', '.join(changed[:3]))
    return descriptor


def install_recipe(release, recipe, tables):
    """The D selector installed once on the loaded 24b67962 graph; ownership checked before and after."""
    from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2
    from rshb_vine.text_evidence_repair_v2 import composition as T
    from rshb_vine.zero_target_release_v1.route import ZeroTargetRoute
    recipe = _recipe(recipe)
    inner, base = release.inner, release.base

    def owners():
        return (base.runtime.selection, base.target.inner.selection, inner.repair.selection, inner.selection)

    parent = inner.selection
    if type(parent) is not RepairSelectionV2 or any(o is not parent for o in owners()):
        raise ValueError('24b67962 selector ownership differs from one RepairSelectionV2')
    if type(base.target.inner.parent.route) is not ZeroTargetRoute:
        raise ValueError('Zero-target route adapter is not installed on this graph')
    wrapper = T.install(inner, recipe, tables)
    if wrapper is parent or any(o is not wrapper for o in owners()) or wrapper.parent is not parent:
        raise ValueError('D selector is not the single live consumer over the 24b67962 selector')
    stage_order = wrapper.identity.get('stage_order') or [s.trace_key for s in wrapper.stages]
    if tuple(wrapper.components) != recipe or 'name_roles' in stage_order:
        raise ValueError('Installed components differ from the recipe or include C')
    return wrapper, {'adapter': T.ADAPTER_VERSION, 'module': T.__name__, 'recipe': list(recipe), 'tables': tables,
                     'identity': wrapper.identity, 'parent_selector': type(parent).__module__ + '.RepairSelectionV2',
                     'owners': ['base.runtime.selection', 'base.target.inner.selection', 'inner.repair.selection',
                                'inner.selection'], 'zero_target_route': 'base.target.inner.parent.route'}


def _parent_mentions(value, needles, path=()):
    if isinstance(value, dict):
        for k, v in value.items():
            if k not in PARENT_KEYS:
                yield from _parent_mentions(v, needles, (*path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _parent_mentions(v, needles, (*path, i))
    elif isinstance(value, str) and value in needles:
        yield '.'.join(map(str, path))


def relabel(result, checksum, runtime_checksum, parent_checksum, parent_runtime, *, admitted):
    """Own checksum in every identity field the parent release wrote; the parent only under explicit lineage keys."""
    lineage = [parent_checksum, *(result['recognition_repair_release_v2'].get('parent_lineage')
                                  or [result['recognition_repair_release_v2']['predecessor_release']])]
    for name in ('recognition_repair_v1', 'recognition_repair_v2'):
        result[name].update(release_profile_checksum=checksum, release_admitted=admitted,
                            parent_release_profile_checksum=parent_checksum)
    for name in ('roskachestvo_geometry_v2', 'target_contract'):
        if result.get(name) is not None:
            result[name].update(profile_checksum=checksum, parent_release_profile_checksum=parent_checksum)
    release = result['recognition_repair_release_v2']
    release.update(profile_checksum=checksum, runtime_checksum=runtime_checksum, predecessor_release=parent_checksum,
                   parent_release_profile_checksum=parent_checksum, parent_runtime_checksum=parent_runtime,
                   parent_lineage=lineage)
    zero = result['zero_target_release_v1']
    zero.update(profile_checksum=checksum, runtime_checksum=runtime_checksum, parent_release_profile_checksum=parent_checksum,
                parent_runtime_checksum=parent_runtime, parent_lineage=lineage, release_admitted=admitted)
    left = list(_parent_mentions(result, {parent_checksum, parent_runtime}))
    if left:
        raise RuntimeError('Response still names the parent release outside lineage: ' + ', '.join(left[:3]))
    return lineage


def trace_key():
    """D keeps the v1 per-target trace key (augmented with producer_aliases) unless it declares its own."""
    from rshb_vine.text_evidence_repair_v1 import composition as V1
    from rshb_vine.text_evidence_repair_v2 import composition as T
    return getattr(T, 'TRACE_KEY', V1.TRACE_KEY)


def block(result, selection, checksum, runtime_checksum, parent_checksum, recipe, *, admitted):
    from rshb_vine.text_evidence_repair_v2 import composition as T
    records = (result.get('systemic_ranking_v2') or {}).get('targets', [])
    traces = [r.get(trace_key()) for r in records]
    if any(t is None for t in traces):
        raise RuntimeError('A published selector record lacks the D selector trace')
    return {'adapter': T.ADAPTER_VERSION, 'recipe': list(recipe), 'profile_checksum': checksum,
            'runtime_checksum': runtime_checksum, 'parent_release_profile_checksum': parent_checksum,
            'selector_identity': deepcopy(selection.identity),
            'applied': sorted(str(r['instance_id']) for r, t in zip(records, traces) if t.get('applied')),
            'release_admitted': admitted, 'calibrated': False, 'probability': None}


class TextEvidenceCandidate:
    """24b67962 + D recipe on loopback 8197 under the live factory F8; never activated from here."""

    def __init__(self, root, descriptor_path=DESCRIPTOR):
        from rshb_vine.zero_target_release_v1.release import ZeroTargetRelease
        root = Path(root).resolve()
        self.descriptor = load_descriptor(root, descriptor_path)
        self.recipe = tuple(self.descriptor['recipe'])
        release = ZeroTargetRelease(root, PARENT_PROFILE)
        if release.profile['checksum'] != PARENT_CHECKSUM:
            raise ValueError('Loaded parent release differs from 24b67962')
        self.selection, self.installation = install_recipe(release, self.recipe, self.descriptor['tables'])
        self.release, self.profile = release, self.descriptor
        self.manifest = seal({'kind': 'text-evidence-release-v2-candidate-runtime',
                              'runtime_descriptor_checksum': self.descriptor['checksum'],
                              'parent_runtime': release.manifest['checksum'],
                              'parent_release_profile_checksum': PARENT_CHECKSUM, 'installation': self.installation,
                              'activated': False, 'calibration_status': 'unknown'})

    def recognize(self, data, roi=None, bottles='addressed'):
        result = self.release.recognize(data, roi, bottles)
        if result.get('decision') == 'invalid_image':
            return result
        if result.get('slug') is not None or result.get('probability_correct') is not None:
            raise RuntimeError('Public slug/probability must stay None')
        own, runtime = self.descriptor['checksum'], self.manifest['checksum']
        relabel(result, own, runtime, PARENT_CHECKSUM, self.release.manifest['checksum'], admitted=False)
        result[BLOCK] = block(result, self.selection, own, runtime, PARENT_CHECKSUM, self.recipe, admitted=False)
        return result
