"""Text-evidence repair v1 at the actual consumer: the running RepairSelectionV2 with up to three versioned components.

The wrapper is built from the live (or replayed) RepairSelectionV2 instance; frozen modules, ranker weights, stages
and published record adapter stay those of that instance. Per target, in the one ``evaluate`` shared by CPU replay
and live ``select``:

  A  normalization   effective = normalize_observations(observations, provenance, lexicon): same count, order and
                     every field except ``raw_text``; provenance (native scores, readers, crops) is the one traced on
                     the ORIGINAL observations and is never recomputed. Scene ownership keys become the original keys
                     of the other targets united with their normalized keys (counterpart_line_keys), so a normalized
                     string can never hide a line owned by another bottle.
  B  producer names  injection only: the phrase index is replaced for this instance by a proxy whose ``propose`` is
                     the ShortProducerPhraseIndex over the sealed short-form table (built over the same eligible slugs,
                     parent checksum = frozen index) and whose ``profile`` (read by the guard) is the frozen index.
  C  name roles      NameRoleStage, pre_guard after producer_role and before guard v2: the name_roles build of the
                     same effective packet replaces only the five name.* features and the raw evidence (read by the
                     guard); all other features must equal the frozen build; one rescore through the frozen ranker
                     and the existing resolver.

Every consumer of the choice (injection, evidence/ranker rows, producer_role, name roles, guard, geometry trigger,
public product_resolution) reads the effective strings; the raw OCR packet of the response is untouched and the
trace keeps original/effective per changed observation. With no component enabled the wrapper is the parent
selection itself (G0 parity). Probability and exact slug stay None.
"""
import copy
import time

from rshb_vine.io import digest
from rshb_vine.ocr_candidate_repair_v1 import injection as I
from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2

ADAPTER_VERSION = 'text-evidence-repair-v1-selection'
TRACE_KEY = 'text_evidence_repair_v1'
PACKAGE = 'rshb_vine/text_evidence_repair_v1'
COMPONENT_SOURCES = {'A': (f'{PACKAGE}/normalization.py',),
                     'B': (f'{PACKAGE}/producer_names.py', 'runs/text-evidence-improve-v1/identity/short-producer-forms-v1.json'),
                     'C': (f'{PACKAGE}/name_roles.py', 'runs/text-evidence-improve-v1/identity/name-roles-v1.json')}
SOURCES = (f'{PACKAGE}/__init__.py', f'{PACKAGE}/composition.py')


def check_effective(original, effective):
    """Effective observations may differ from the originals only in ``raw_text``."""
    if len(effective) != len(original):
        raise RuntimeError('normalization changed the observation count')
    changed = []
    for i, (o, e) in enumerate(zip(original, effective)):
        if e is o:
            raise RuntimeError('normalization returned an original observation object')
        if set(o) != set(e) or any(o[k] != e[k] for k in o if k != 'raw_text'):
            raise RuntimeError('normalization changed a field other than raw_text at observation %d' % i)
        if e.get('raw_text') != o.get('raw_text'):
            changed.append({'observation_id': i, 'original': o.get('raw_text'), 'effective': e.get('raw_text')})
    return changed


class Normalization:
    """Component A over the selector's own typed lexicon (inner.context.lexicon), one vocabulary per instance."""

    name = 'A'

    def __init__(self, inner):
        from rshb_vine.text_evidence_repair_v1 import normalization as N
        self.module, self.lexicon = N, inner.context.lexicon
        self.vocabulary = N.Vocabulary(self.lexicon)
        self.identity = {'component': 'A', 'module': N.__name__, 'rule': N.RULE['version'],
                         'vocabulary_digest': self.vocabulary.digest}

    def normalize(self, observations, provenance):
        return self.module.normalize_observations(observations, provenance, self.lexicon, vocabulary=self.vocabulary)

    def counterpart_line_keys(self, result, instance_id):
        return set(self.module.counterpart_line_keys(result, instance_id, self.lexicon, vocabulary=self.vocabulary))


class ProducerNameIndex:
    """Injection index of this selection instance: component B proposes, the frozen index keeps the guard profile."""

    def __init__(self, frozen, proposer, rule):
        self.frozen, self.proposer, self.rule = frozen, proposer, rule
        self.checksum = digest({'frozen': frozen.checksum, 'component_b': proposer.checksum})
        self.last_seconds = None

    def profile(self, slug):
        return self.frozen.profile(slug)

    def propose(self, observations, provenance, pool_slugs, other_line_keys=(), conflicts=None):
        started = time.perf_counter()
        decision = self.proposer.propose(observations, provenance, pool_slugs, other_line_keys, conflicts)
        self.last_seconds = time.perf_counter() - started
        decision['index_checksum'] = self.checksum
        decision['frozen_index_checksum'] = self.frozen.checksum
        return decision

    def describe(self):
        return {'frozen': self.frozen.describe(), 'component_b': self.proposer.describe(), 'checksum': self.checksum}


def producer_name_index(selection, table_checksum):
    from rshb_vine.text_evidence_repair_v1 import producer_names as P
    inner = selection.inner
    proposer = P.build_injection_index(selection.index, inner.registry, inner.context, inner.stats,
                                       root=selection.root, table_checksum=table_checksum)
    if proposer.base is not selection.index:
        raise ValueError('Short-form index is not built over the frozen phrase index')
    identity = {'component': 'B', 'module': P.__name__, 'rule': P.RULE['version'], 'table': P.TABLE,
                'table_checksum': proposer.table_checksum, 'index_checksum': proposer.checksum,
                'cards_with_short_forms': len(proposer.short)}
    return ProducerNameIndex(selection.index, proposer, P.RULE['version']), identity


class TimedStage:
    """A stage with its own wall time per call; trace key, position and identity are the stage's."""

    def __init__(self, stage):
        self.stage, self.trace_key, self.position, self.identity = stage, stage.trace_key, stage.position, stage.identity
        self.last_seconds = None

    def apply(self, selection, out, call, request):
        started = time.perf_counter()
        try:
            return self.stage.apply(selection, out, call, request)
        finally:
            self.last_seconds = time.perf_counter() - started


NAME_FEATURES = ('name.read_w', 'name.read_g.v', 'name.read_g.p', 'name.exclusive_w', 'name.fuzzy_w')
NAME_STAGE = 'text-evidence-name-roles-pre-guard-stage'


class NameRoleStage:
    """Component C after producer_role: NAME features and evidence without producer-range printed words, one rescore.

    Only the five name.* read/exclusive/fuzzy features and the ``name`` entries of the raw evidence change; every
    other feature and evidence field of the C build must equal the frozen build of the same effective packet, and
    producer_role's features are kept. name.phrase stays the frozen v4 full-title phrase feature (a complete
    catalogue title claim read as a phrase, not token coverage), so C does not touch it.
    """

    trace_key = 'name_roles'
    position = 'pre_guard'

    def __init__(self, root, inner, table_checksum):
        from rshb_vine.systemic_ranking_v2 import evidence as E
        from rshb_vine.text_evidence_repair_v1 import name_roles as C
        if set(NAME_FEATURES) - set(E.FEATURE_NAMES):
            raise ValueError('NAME features are not systemic features')
        self.module, self.inner = C, inner
        self.roles = C.NameRoles(inner, C.load_table(root, inner.context, inner.registry, table_checksum))
        self.identity = {'stage': NAME_STAGE, 'rule': C.RULE['version'], 'table': C.TABLE,
                         'table_checksum': self.roles.table_checksum, 'roles_checksum': self.roles.checksum,
                         'features': list(NAME_FEATURES), 'kept': 'name.phrase (frozen v4 full-title phrase), all other features',
                         'position': 'pre_guard after producer_role'}

    def apply(self, selection, out, call, request):
        from rshb_vine.learned_selection_features import _candidate_key
        from rshb_vine.ocr_candidate_repair_v1.selection import _relabel
        from rshb_vine.systemic_ranking_v2.selection import feature_digest
        inner = self.inner
        rows, raw, _, removed = self.module.build(out['base'], out['v4'], call['provenance'], inner.context, inner.stats,
                                                  call['control_slug'], inner.identity.product_id, self.roles)
        trace = {'stage': NAME_STAGE, 'rule': self.identity['rule'], 'removed': removed, 'changes': [],
                 'applied': False, 'reranked': False}
        out[self.trace_key] = trace
        if not removed:
            return out
        frozen = out.get('rows_before_producer_role', out['rows'])
        for cid, a, b in zip(out['candidate_ids'], rows, frozen):
            other = [n for n in a if n not in NAME_FEATURES and a[n] != b[n]]
            if other:
                raise RuntimeError('name-role build changed non-name features %s of %s' % (other, cid))
        new, changes = [], []
        for cid, current, c_row in zip(out['candidate_ids'], out['rows'], rows):
            row = dict(current, **{n: c_row[n] for n in NAME_FEATURES})
            diff = {n: [current[n], row[n]] for n in NAME_FEATURES if row[n] != current[n]}
            if diff:
                changes.append({'candidate_id': cid, 'features': diff})
            new.append(row)
        trace['changes'] = changes
        if not changes:
            return out
        evidence = []
        for old, fresh in zip(out['evidence'], raw):
            other = [k for k in fresh if k not in ('name', 'name_role_removed') and fresh[k] != old.get(k)]
            if other or set(old) - set(fresh):
                raise RuntimeError('name-role build changed evidence fields %s of %s' % (other, old['candidate_id']))
            evidence.append(dict(old, name=fresh['name'], name_role_removed=fresh['name_role_removed']))
        out['rows_before_name_roles'], out['evidence_before_name_roles'] = out['rows'], out['evidence']
        out['feature_digest_before_name_roles'] = out['feature_digest']
        out['rows'], out['evidence'] = new, evidence
        out['feature_digest'] = feature_digest(new, out['candidate_ids'])
        trace['applied'] = True
        if out.get('raw_proposal') is not None:
            injected = {_candidate_key(s, selection.registry.cards) for s in out['ocr_candidate_injection']['injected']}
            proposal_raw = selection.inner._propose(out['base'], out['v4'], new, call['control_slug'])
            _relabel(proposal_raw, injected)
            out['raw_proposal_before_name_roles'], out['proposal_before_name_roles'] = out['raw_proposal'], out['proposal']
            out['raw_proposal'] = proposal_raw
            out['proposal'] = selection.legacy.resolver.resolve(out['base'], proposal_raw)
            trace.update(reranked=True, winner={'before': out['raw_proposal_before_name_roles']['representative_slug'],
                                                'after': proposal_raw['representative_slug']})
        return out


def name_role_stage(selection, table_checksum):
    stage = NameRoleStage(selection.root, selection.inner, table_checksum)
    if stage.trace_key in {s.trace_key for s in selection.stages}:
        raise ValueError('Name-role stage must be a new pre_guard stage')
    return TimedStage(stage), dict(stage.identity, component='C', module=stage.module.__name__)


class TextEvidenceSelection(RepairSelectionV2):
    def __init__(self, parent, components=(), tables=None):
        """``tables``: pinned table checksums {'B': ..., 'C': ...} of the enabled table-driven components."""
        tables = tables or {}
        if type(parent) is not RepairSelectionV2:
            raise ValueError('TextEvidenceSelection wraps the running RepairSelectionV2 instance only')
        if len(set(components)) != len(components) or set(components) - set(COMPONENT_SOURCES):
            raise ValueError('Unknown or repeated component: %r' % (components,))
        self.parent = parent
        self.root, self.inner = parent.root, parent.inner
        self.registry, self.legacy, self.model = parent.registry, parent.legacy, parent.model
        self.index, self.conflicts = parent.index, parent.conflicts
        self.current_result, self.records, self.request, self.last_trace = None, [], None, None
        self.stages = parent.stages
        self.components = tuple(c for c in ('A', 'B', 'C') if c in components)
        self.identities, self.normalization, self.name_stage = [], None, None
        if 'A' in self.components:
            self.normalization = Normalization(self.inner)
            self.identities.append(self.normalization.identity)
        if 'B' in self.components:
            self.index, identity = producer_name_index(self, tables['B'])
            self.identities.append(identity)
        if 'C' in self.components:
            keys = [s.trace_key for s in parent.stages]
            if 'producer_role' not in keys:
                raise ValueError('Name-role stage is placed after producer_role, which is not loaded')
            stage, identity = name_role_stage(self, tables['C'])
            at = keys.index('producer_role') + 1
            self.stages = parent.stages[:at] + (stage,) + parent.stages[at:]
            self.name_stage = stage
            self.identities.append(identity)
        self.identity = {'adapter': ADAPTER_VERSION, 'components': list(self.components), 'identities': self.identities,
                         'stage_order': [s.trace_key for s in self.stages]}
        self.last_text = None

    def _result(self):
        request = self.request.result if self.request is not None else None
        if request is not None and self.current_result is not None and request is not self.current_result:
            raise RuntimeError('Request and in-flight control result differ')
        result = request if request is not None else self.current_result
        if result is None:
            raise RuntimeError('Scene ownership needs the in-flight or replayed control result')
        return result

    def evaluate(self, **call):
        iid = str(call['raw_visual']['instance_id'])
        trace = {'adapter': ADAPTER_VERSION, 'instance_id': iid, 'components': list(self.components)}
        effective, seconds = call['observations'], {}
        if isinstance(self.index, ProducerNameIndex):
            self.index.last_seconds = None
        if self.name_stage is not None:
            self.name_stage.last_seconds = None
        if self.normalization is not None:
            started = time.perf_counter()
            original = call['observations']
            snapshot = copy.deepcopy(original)
            effective, detail = self.normalization.normalize(original, call['provenance'])
            if original != snapshot:
                raise RuntimeError('normalization mutated the original observations')
            changed = check_effective(original, effective)
            given = set(call.get('other_line_keys', ()))
            if given != I.scene_line_keys(self._result(), iid):
                raise RuntimeError('Supplied scene line keys are not those of the control result')
            keys = self.normalization.counterpart_line_keys(self._result(), iid)
            if not given <= keys:
                raise RuntimeError('Counterpart line keys dropped an original key of another target')
            call = dict(call, observations=effective, other_line_keys=keys)
            seconds['normalization'] = time.perf_counter() - started
            trace['normalization'] = {'changed': changed, 'trace': detail,
                                      'other_line_keys_added': sorted(keys - given)}
        out = super().evaluate(**call)
        if 'B' in self.components:
            decision = out['ocr_candidate_injection']
            short = set((decision.get('short_producer') or {}).get('matched') or [])
            trace['producer_names'] = dict({k: decision.get(k) for k in (
                'rule', 'status', 'injected', 'skipped', 'short_producer', 'index_checksum', 'frozen_index_checksum')},
                short_form_injected=sorted(short & set(decision['injected'])))
            seconds['producer_names_propose'] = self.index.last_seconds
        if self.name_stage is not None:
            trace['name_roles'] = out.get(self.name_stage.trace_key)
            seconds['name_roles'] = self.name_stage.last_seconds
        trace['applied'] = bool((trace.get('normalization') or {}).get('changed')
                                or (trace.get('producer_names') or {}).get('short_form_injected')
                                or (trace.get('name_roles') or {}).get('applied'))
        trace['seconds'] = dict(seconds, components_total=sum(v for v in seconds.values() if v is not None))
        out[TRACE_KEY] = self.last_text = trace
        out['effective_observations'] = effective
        return out

    def select(self, **kwargs):
        self.last_text = None
        result = super().select(**kwargs)
        trace = self.last_text
        if trace is None:
            raise RuntimeError('Live select did not run the text-evidence evaluate')
        record = result['systemic_ranking_v2']
        record[TRACE_KEY] = dict(trace, identity=self.identity)
        changed = (trace.get('normalization') or {}).get('changed')
        if changed:
            public = result['proposal']
            effective = copy.deepcopy(kwargs['observations'])
            for c in changed:
                effective[c['observation_id']]['raw_text'] = c['effective']
            record[TRACE_KEY]['product_resolution_raw_ocr'] = public['product_resolution']
            public['product_resolution'] = self.legacy.describe(
                public['representative_slug'], effective,
                {'kind': 'same_instance_ocr', 'instance_id': kwargs['raw_visual']['instance_id']})
        return result


def install(v2_runtime, components, tables=None):
    """Replace the RepairSelectionV2 of a constructed RecognitionRepairV2 on its own runtime graph (no globals)."""
    from rshb_vine.recognition_repair_v2.composition import Request
    from rshb_vine.systemic_ranking_v2.selection import attach
    parent, repair = v2_runtime.selection, v2_runtime.repair
    runtime = repair.release.runtime
    if (type(parent) is not RepairSelectionV2 or repair.selection is not parent or runtime.selection is not parent
            or repair.release.target.inner.selection is not parent):
        raise ValueError('RepairSelectionV2 ownership changed')
    wrapper = TextEvidenceSelection(parent, components, tables)
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
