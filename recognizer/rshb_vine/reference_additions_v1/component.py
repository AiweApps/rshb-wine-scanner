"""Component D at every actual consumer of a loaded bde4fa52 graph: 4 admitted B3 rows reach all gallery holders.

``install(graph, root, descriptor)`` runs after the frozen G/T and before anything else. It proves by an object-graph
scan that each gallery-derived structure has exactly one live holder and re-derives every checksum formula on the
untouched state (``preflight`` evaluates and reports all preconditions before any mutation), then extends in place (never replaces a slot, so T's ``select`` closure keeps its proposer):

  visual   the one LabelFirstIndex under ConflictFilteredIndex/_IndexDelegate: vectors, references, channel_ids;
           the filter's excluded rows are recomputed by the same guard and must stay unchanged;
  layout   the layout-geometry verifier's own reference copy, source paths, conflicts and identity;
  registry ``active_in_gallery`` of the two cards per registry owner; the snapshot-01 selector registry (shared by
           selector, layout and text) and the public snapshot-03 identity registry keep their own checksums;
  text     frozen CatalogPhraseIndex (entries for the new slugs by its own constructor, same rule and 0.6 gate),
           ShortProducerIndex (own constructor over the pinned table), CompositeIdentityIndex (eligible set, sibling
           census and entries recomputed over the whole universe), every ProducerNameIndex checksum and the T identity;
  policies every consumer catalogue text policy (FrozenCandidateSelector/PositiveVariantText) must cover the served
           gallery: the control-chain title.policy and instance.variant_policy, built over catalogue∩B0-gallery slugs, are
           rebuilt by their own constructors over the admitted-grape adapted catalogue and used unmodified.

Old rows, vectors, entries and policies stay byte-identical; the record publishes the new gallery/index identities
next to the old ones. Slug and probability stay None; nothing global, pinned or module-level is patched.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from rshb_vine.io import digest, local_path, read_json, read_jsonl, seal, sha256, verify
from rshb_vine.reference_additions_v1 import admission as A
from rshb_vine.reference_additions_v1 import gallery as G

KIND = 'reference-additions-release-next-v1-descriptor'
PACKAGE = 'rshb_vine/reference_additions_v1'
SOURCES = (*G.SOURCES, f'{PACKAGE}/component.py', 'scripts/reference_additions_v1.py')
DESCRIPTOR = 'runs/release-next-v1/references/descriptor-v5.json'
VERSION = 5
PARENT_PROFILE = 'config/atlas-repair-release-v1-profile.json'
PARENT_CHECKSUM = 'bde4fa52d35b1c297e43996cacf11db2e00558a0273586639b3e01923ab468f3'
BLOCK = 'reference_additions_v1'
SUPERSEDES = 'd90ad8b8b3c41e0fa1b881255100139a97dd2bdf301d9066f6c77987cae3e846'
FAILURE = 'runs/release-next-v1/references/archive-dd90/manifest.json'
RELEASE_NEXT_KINDS = ('recognition-release-next-v1-profile',)
SELECTOR_REGISTRY = '31e053df11ba7fffc38666d300cd8cfea6af7b4f58cc88abdba2580dcbd95baa'
PUBLIC_REGISTRY = '296d99bb47b9f5e31ceda65202a11aa8bbd40814bd31b3e6db479841d09a43dd'
REGISTRY_ROLES = {SELECTOR_REGISTRY: 'selector_pool_snapshot01', PUBLIC_REGISTRY: 'public_product_identity_snapshot03'}
POLICY = {'id': 'reference-additions-v1-source-only-append',
          'rule': 'append root-admitted catalogue card images as gallery-only B3 rows after the frozen parent rows; '
                  'no reordering, removal, re-encoding or weight change of existing rows',
          'text': 'catalogue text indexes rebuilt by their own constructors over the enlarged eligible set; '
                  'rules, gates and weights unchanged; no ratio or title heuristic',
          'registries': 'each registry keeps its own identity and checksum; selector, layout and text consumers must share '
                        'the one snapshot-01 selector registry; only the gallery-derived active_in_gallery flag of the added '
                        'cards is set, per registry owner, with provenance in the D record',
          'catalogue_policies': 'every consumer catalogue text policy must cover the served gallery slugs; an uncovered '
                                'ProductNameSelector/CatalogConstrainedTitle is rebuilt by its unchanged constructor over '
                                'the same admitted-grape adapted catalogue (inventory bound to the control receipt) and used '
                                'as constructed; old per-card items, grapes, product and producer names and grape words must '
                                'be unchanged; only the catalogue-wide distinct_names effect is allowed and reported',
          'fit': 'none', 'query_promotion': 'none', 'public_slug': None, 'probability': None}
CONSTANTS = {'parent_gallery_sha256': A.PARENT_GALLERY_SHA, 'parent_vectors_sha256': A.PARENT_VECTORS_SHA,
             'parent_rows': A.PARENT_ROWS, 'kinds': list(A.KINDS), 'encoder_preprocessing': G.PREPROCESSING,
             'decode': G.DECODE, 'max_new_rows': 4, 'registry_roles': REGISTRY_ROLES,
             'control_protocol': 'runs/color-verified-recognition-v1/protocol.json'}


def sources_sha(root):
    return {p: sha256(local_path(root, p)) for p in SOURCES}


def build_pins(root, out=G.OUT):
    target = local_path(root, out)
    pins = {f'{out}/{n}': sha256(target / n) for n in ('gallery.json', 'vectors.npy', 'receipt.json')}
    pins.update({f'{out}/images/{p.name}': sha256(p) for p in sorted((target / 'images').iterdir())})
    return pins


def freeze(root, out=G.OUT):
    """Sealed descriptor over a finished build: parent bde4fa52, admission, build bytes and D sources."""
    root = Path(root).resolve()
    profile = verify(read_json(local_path(root, PARENT_PROFILE)))
    if profile['checksum'] != PARENT_CHECKSUM:
        raise ValueError('Parent release profile is not bde4fa52')
    doc, array, receipt = G.load(root, out)
    config = verify(read_json(local_path(root, A.CONFIG)))
    pins = {A.CONFIG: sha256(local_path(root, A.CONFIG)), **build_pins(root, out)}
    for key in ('root_admission', 'source_audit', 'amendment', 'crop_manifest'):
        pins[config[key]['path']] = config[key]['sha256']
    return seal({'kind': KIND, 'version': VERSION, 'supersedes': {'descriptor_checksum': SUPERSEDES,
                 'failure': FAILURE}, 'parent_release': {'path': PARENT_PROFILE, 'checksum': PARENT_CHECKSUM,
                                                  'sha256': sha256(local_path(root, PARENT_PROFILE))},
                 'build': out, 'gallery_checksum': doc['checksum'], 'gallery_sha256': receipt['gallery_sha256'],
                 'vectors_sha256': receipt['vectors_sha256'], 'rows': len(doc['references']),
                 'added_slugs': sorted({r['slug'] for r in doc['references'][A.PARENT_ROWS:]}),
                 'policy': POLICY, 'constants': CONSTANTS, 'sources_sha256': sources_sha(root), 'pins_sha256': pins,
                 'gallery_only': True, 'fit_run': False, 'activated': False, 'calibrated': False,
                 'probability': None, 'public_slug': None})


def load_descriptor(root, path=DESCRIPTOR):
    root = Path(root).resolve()
    descriptor = verify(read_json(local_path(root, path)))
    if (descriptor.get('kind') != KIND or descriptor.get('version') != VERSION
            or descriptor['parent_release']['checksum'] != PARENT_CHECKSUM):
        raise ValueError('Not a v%d reference additions descriptor over bde4fa52' % VERSION)
    if descriptor.get('policy') != POLICY or descriptor.get('constants') != CONSTANTS:
        raise ValueError('D policy/constants differ from the frozen descriptor')
    if descriptor['rows'] - CONSTANTS['parent_rows'] > CONSTANTS['max_new_rows']:
        raise ValueError('Descriptor exceeds the admitted row budget')
    if sha256(local_path(root, PARENT_PROFILE)) != descriptor['parent_release']['sha256']:
        raise ValueError('Parent release profile bytes changed')
    changed = [p for p, s in {**descriptor['pins_sha256'], **descriptor['sources_sha256']}.items()
               if sha256(local_path(root, p)) != s]
    if changed or set(descriptor['sources_sha256']) != set(SOURCES):
        raise ValueError('Descriptor pins or sources changed: ' + ', '.join(changed[:3] or ['source set']))
    return descriptor


def holders(graph, kinds, limit=500000):
    """{class name: [(object, path)]} of every owned object of the given classes reachable from ``graph``."""
    from rshb_vine.b3_only_v1.runtime import _owned
    found, seen, stack = defaultdict(list), set(), [(graph, 'release')]
    while stack and len(seen) < limit:
        obj, path = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        name = type(obj).__name__
        if name in kinds and _owned(obj):
            found[name].append((obj, path))
        if isinstance(obj, dict):
            children = [(v, path + '[' + repr(k)[:40] + ']') for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            children = [(v, path + '[%d]' % i) for i, v in enumerate(obj)]
        elif _owned(obj):
            children = [(v, path + '.' + k) for k, v in vars(obj).items() if k not in DATA_ATTRS]
        else:
            children = []
        stack.extend(c for c in children if _owned(c[0]) or isinstance(c[0], (dict, list, tuple)) and (
            len(c[0]) <= 256 or any(_owned(v) for v in (c[0].values() if isinstance(c[0], dict) else c[0]))))
    if stack:
        raise ValueError('Graph scan limit reached; consumer proof incomplete')
    return found


def _one(found, name):
    rows = found.get(name, [])
    if len(rows) != 1:
        raise ValueError('%s must have exactly one live holder, found %s' % (name, [p for _, p in rows]))
    return rows[0][0]


class PreflightError(ValueError):
    def __init__(self, report):
        failed = [c['check'] for c in report['checks'] if not c['ok']]
        super().__init__('D preflight failed before any mutation: ' + ', '.join(failed))
        self.report = report


def _composite_checksum(index):
    from rshb_vine.atlas_text_repair_v1.index import RULE
    return digest({'rule': RULE, 'parent': index.parent.checksum, 'entries': {
        s: {'name': [t.forms[0] for t in c['name']], 'forms': [[t.forms[0] for t, _ in f] for f in c['forms']],
            'aliases': c['aliases_trace'], 'more_specific': index.census[s]['more_specific_siblings'],
            'equal': index.census[s]['equal_siblings']} for s, c in sorted(index.entries.items())}})


def _verifier_identity(verifier, registry):
    from rshb_vine.recognition_repair_v2 import geometry as LG
    return digest({'policy': LG.POLICY, 'criteria': LG.CRITERIA, 'ambiguity': LG.V.AMBIGUITY, 'trigger': LG.V.TRIGGER,
                   'gallery': verifier.gallery, 'code': verifier.code, 'registry': registry.checksum,
                   'conflicts': verifier.conflicts_checksum})


def _producer_checksum(index):
    return digest({'frozen': index.frozen.checksum, 'component_b': index.proposer.checksum})


# Per-row/per-card payloads of the scanned holders; they hold data, never another consumer object.
DATA_ATTRS = frozenset(('entries', 'census', 'cards', 'short', 'references', 'paths', 'profiles_cache', '_prepared',
                        'excluded', 'document', 'claims', 'products', 'members', 'universe', 'vectors', 'channel_ids'))
KINDS = ('LabelFirstIndex', 'ConflictFilteredIndex', '_IndexDelegate', 'SelectorBoundLayoutVerifier', 'GeometryStage',
         'ProductRegistry', 'CatalogPhraseIndex', 'ShortProducerIndex', 'CompositeIdentityIndex', 'ProducerNameIndex',
         'ReferenceConflicts', 'SystemicSelection', 'IdentityAdapter', 'ColorVerifiedRecognition')


def check_graph(graph, descriptor):
    """bde4fa52 itself, or a release-next profile whose predecessor is bde4fa52 and which names this descriptor as D."""
    profile = graph.profile
    if profile['checksum'] != PARENT_CHECKSUM:
        named = ((profile.get('release_next') or {}).get('components') or {}).get('D') or {}
        if (profile.get('kind') not in RELEASE_NEXT_KINDS
                or (profile.get('predecessor_release') or {}).get('checksum') != PARENT_CHECKSUM
                or named.get('checksum') != descriptor['checksum']):
            raise ValueError('D installs only on bde4fa52 or a release-next graph over it that names this descriptor')
    if 'T' not in tuple(graph.atlas_recipe):
        raise ValueError('D installs only after the G/T recipe with T')


CONTROL_PROTOCOL = 'runs/color-verified-recognition-v1/protocol.json'
REBUILDABLE = ('ProductNameSelector', 'CatalogConstrainedTitle')
DERIVED = ('product_names', 'producer_names')


def policy_slots(graph, limit=500000):
    """Every (owner, attr, path, policy) edge to a catalogue text policy, plus container edges that hold one."""
    from rshb_vine.b3_only_v1.runtime import _owned
    from rshb_vine.frozen_candidate_selector import FrozenCandidateSelector
    from rshb_vine.positive_variant_text import PositiveVariantText
    kinds = (FrozenCandidateSelector, PositiveVariantText)
    edges, containers, seen, stack = [], [], set(), [(graph, 'release')]
    while stack and len(seen) < limit:
        obj, path = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            children = [(v, path + '[' + repr(k)[:40] + ']') for k, v in obj.items()]
            containers += [p for v, p in children if isinstance(v, kinds)]
        elif isinstance(obj, (list, tuple)):
            children = [(v, path + '[%d]' % i) for i, v in enumerate(obj)]
            containers += [p for v, p in children if isinstance(v, kinds)]
        elif _owned(obj):
            children = []
            for k, v in vars(obj).items():
                if isinstance(v, kinds):
                    edges.append((obj, k, path + '.' + k, v))
                if k not in DATA_ATTRS:
                    children.append((v, path + '.' + k))
        else:
            children = []
        stack.extend(c for c in children if _owned(c[0]) or isinstance(c[0], (dict, list, tuple)) and (
            len(c[0]) <= 256 or any(_owned(v) for v in (c[0].values() if isinstance(c[0], dict) else c[0]))))
    if stack:
        raise ValueError('Graph scan limit reached; policy ownership proof incomplete')
    return edges, containers


def _top_level(edges):
    """Policies held by a consumer, not the internal base/parent layers of another policy."""
    from rshb_vine.catalog_phrase_text import CatalogPhrase
    from rshb_vine.frozen_candidate_selector import FrozenCandidateSelector
    from rshb_vine.gallery_variant_text import GalleryVariantText
    internal = (CatalogPhrase, GalleryVariantText, FrozenCandidateSelector)
    return [e for e in edges if not isinstance(e[0], internal)]


def _derived(policy, slugs):
    return {'items': {s: policy.items[s] for s in slugs}, 'grapes': {s: policy.parent.grapes[s] for s in slugs},
            **{k: {s: getattr(policy, k)[s] for s in slugs} for k in DERIVED},
            'distinct': sorted(set(policy.distinct_names) & set(slugs)), 'grape_words': sorted(policy.grape_words)}


def _find_receipt(value):
    if isinstance(value, dict):
        if isinstance(value.get('admitted_grape_evidence'), dict):
            return value['admitted_grape_evidence']
        for v in value.values():
            found = _find_receipt(v)
            if found:
                return found
    return None


def plan_policies(graph, root, gallery_slugs, control):
    """Rebuild plan for every consumer policy that does not cover the served gallery; nothing is installed here."""
    from rshb_vine.admitted_grape_evidence import AdmittedGrapeEvidence
    from rshb_vine.unified_recognition import verify_protocol
    edges, containers = policy_slots(graph)
    top = _top_level(edges)
    protocol = verify_protocol(root, local_path(root, CONTROL_PROTOCOL))
    catalog = read_jsonl(local_path(root, protocol['catalog_directory']) / 'catalog.jsonl')
    adapted, inventory = AdmittedGrapeEvidence(root, local_path(root, protocol['grape_protocol'])).adapt(catalog)
    receipt = _find_receipt(control.manifest) or {}
    excluded = {r['slug'] for r in adapted if r.get('excluded_from_retrieval')}
    target = set(gallery_slugs) - excluded
    by_policy = defaultdict(list)
    for owner, attr, path, policy in top:
        by_policy[id(policy)].append((owner, attr, path, policy))
    audit, plan = [], []
    for rows in by_policy.values():
        policy = rows[0][3]
        missing = sorted(target - set(policy.items))
        audit.append({'class': type(policy).__name__, 'slots': [p for _, _, p, _ in rows],
                      'items': len(policy.items), 'missing_gallery_slugs': missing})
        if not missing:
            continue
        if type(policy).__name__ not in REBUILDABLE or len(rows) != 1:
            raise ValueError('Uncovered catalogue policy without a rebuild contract: ' + ', '.join(p for _, _, p, _ in rows))
        owner, attr, path, _ = rows[0]
        old = sorted(policy.items)
        cls = type(policy)
        rebuilt_old = cls([r for r in adapted if r['slug'] in policy.items], policy.gallery)
        if _derived(rebuilt_old, old) != _derived(policy, old) or set(rebuilt_old.items) != set(old):
            raise ValueError('Live policy is not reproduced by its adapted-catalogue contract: ' + path)
        wanted = set(old) | target
        new = cls([r for r in adapted if r['slug'] in wanted], policy.gallery)
        if set(new.items) != wanted or new.distinct_names is not new.parent.base.distinct_names:
            raise ValueError('Rebuilt policy does not cover the served gallery: ' + path)
        before, after = _derived(policy, old), _derived(new, old)
        diff = {'items_changed': sorted(s for s in old if before['items'][s] != after['items'][s]),
                'grapes_changed': sorted(s for s in old if before['grapes'][s] != after['grapes'][s]),
                **{k + '_changed': sorted(s for s in old if before[k][s] != after[k][s]) for k in DERIVED},
                'distinct_lost': sorted(set(before['distinct']) - set(after['distinct'])),
                'distinct_gained': sorted(set(after['distinct']) - set(before['distinct'])),
                'grape_words_added': sorted(set(after['grape_words']) - set(before['grape_words'])),
                'grape_words_removed': sorted(set(before['grape_words']) - set(after['grape_words']))}
        frozen_parts = [k for k in ('items_changed', 'grapes_changed', 'product_names_changed', 'producer_names_changed',
                                    'grape_words_added', 'grape_words_removed') if diff[k]]
        if frozen_parts:
            raise ValueError('Rebuild would change old per-card entries or grape words (%s): %s' % (', '.join(frozen_parts), path))
        plan.append({'owner': owner, 'attr': attr, 'path': path, 'old': policy, 'new': new, 'old_slugs': old,
                     'added_slugs': sorted(wanted - set(old)), 'constructor_diff_on_old_slugs': diff})
    detail = {'policies': audit, 'container_held': containers, 'rebuild': [p['path'] for p in plan],
              'inventory_checksum': inventory['checksum'], 'receipt_inventory': receipt.get('inventory'),
              'control_protocol': CONTROL_PROTOCOL,
              'constructor_diff_on_old_slugs': {p['path']: p['constructor_diff_on_old_slugs'] for p in plan}}
    ok = not containers and receipt.get('inventory') == inventory['checksum']
    return ok, detail, plan


def apply_policies(plan):
    """Swap each single slot to its unchanged-constructor rebuild; no derived entry is edited."""
    records = []
    for step in plan:
        setattr(step['owner'], step['attr'], step['new'])
        records.append({'path': step['path'], 'class': type(step['new']).__name__, 'old_items': len(step['old_slugs']),
                        'new_items': len(step['new'].items), 'added_slugs': step['added_slugs'],
                        'constructor_output_unmodified': True,
                        'distinct_names_delta_on_old_slugs': {
                            'lost': step['constructor_diff_on_old_slugs']['distinct_lost'],
                            'gained': step['constructor_diff_on_old_slugs']['distinct_gained']}})
    return records


def _registry_holders(found):
    """[(registry, checksum, role, holder paths)] per distinct registry object, in scan order."""
    rows, index = [], {}
    for registry, path in found.get('ProductRegistry', []):
        if id(registry) not in index:
            index[id(registry)] = len(rows)
            rows.append((registry, registry.checksum, REGISTRY_ROLES.get(registry.checksum), []))
        rows[index[id(registry)]][3].append(path)
    return rows


def preflight(graph, root, descriptor):
    """Every precondition on the untouched graph, all evaluated; returns (report, context). Mutates nothing."""
    from rshb_vine.ocr_candidate_repair_v1.injection import CatalogPhraseIndex
    from rshb_vine.text_evidence_repair_v1 import producer_names as P
    root = Path(root).resolve()
    checks, ctx = [], {}

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as error:  # a failed precondition is reported, never raised mid-scan
            ok, detail = False, '%s: %s' % (type(error).__name__, error)
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})
        return ok

    def graph_kind():
        check_graph(graph, descriptor)
        return getattr(graph, BLOCK, None) is None, {'profile_checksum': graph.profile['checksum'],
                                                     'already_installed': getattr(graph, BLOCK, None) is not None}

    def build():
        doc, array, receipt = G.load(root, descriptor['build'])
        ctx.update(doc=doc, array=array, receipt=receipt, added=doc['references'][A.PARENT_ROWS:],
                   slugs=sorted({r['slug'] for r in doc['references'][A.PARENT_ROWS:]}))
        return (doc['checksum'] == descriptor['gallery_checksum'] and receipt['vectors_sha256'] == descriptor['vectors_sha256']
                and len(doc['references']) == descriptor['rows']), {'rows': len(doc['references'])}

    def scan():
        found = ctx['found'] = holders(graph, KINDS)
        counts = {k: [p for _, p in found.get(k, [])] for k in KINDS}
        for name, key in (('LabelFirstIndex', 'raw'), ('ConflictFilteredIndex', 'filtered'),
                          ('SelectorBoundLayoutVerifier', 'verifier'), ('GeometryStage', 'stage'),
                          ('CatalogPhraseIndex', 'frozen'), ('ShortProducerIndex', 'short'),
                          ('CompositeIdentityIndex', 'composite')):
            ctx[key] = _one(found, name)
        ctx['names'] = [n for n, _ in found.get('ProducerNameIndex', [])]
        ctx['delegates'] = [d for d, _ in found.get('_IndexDelegate', []) if d._index is ctx['filtered']]
        return True, counts

    def chain():
        raw, filtered, verifier, stage = ctx['raw'], ctx['filtered'], ctx['verifier'], ctx['stage']
        frozen, short, composite, names = ctx['frozen'], ctx['short'], ctx['composite'], ctx['names']
        links = {'filter_wraps_raw': filtered._index is raw, 'delegates': len(ctx['delegates']),
                 'stage_verifier': stage.verifier is verifier, 'short_over_frozen': short.base is frozen,
                 'composite_over_short': composite.parent is short and composite.frozen is frozen,
                 'producer_name_indexes': len(names), 'names_over_frozen': all(n.frozen is frozen for n in names),
                 'live_proposer_is_composite': graph.selection.index.proposer is composite}
        return all(links.values()), links

    def registries():
        rows = ctx['registries'] = _registry_holders(ctx['found'])
        selector = ctx['verifier'].registry
        consumers = {'layout_verifier': ctx['verifier'].registry, 'geometry_stage': ctx['stage'].registry,
                     'phrase_index': ctx['frozen'].registry, 'short_index': ctx['short'].registry,
                     'composite_index': ctx['composite'].registry}
        systemic = [(s, p) for s, p in ctx['found'].get('SystemicSelection', [])]
        adapters = [(a, p) for a, p in ctx['found'].get('IdentityAdapter', [])]
        detail = {'objects': [{'checksum': c, 'role': role, 'holders': paths[:12], 'holder_count': len(paths)}
                              for _, c, role, paths in rows],
                  'selector_consumers_share_one_object': all(r is selector for r in consumers.values()),
                  'selector_checksum': selector.checksum,
                  'systemic': [{'path': p, 'registry': s.registry.checksum, 'identity': s.identity.checksum,
                                'registry_is_selector': s.registry is selector} for s, p in systemic],
                  'identity_adapters': [{'path': p, 'frozen': a.frozen.checksum, 'candidate': a.candidate.checksum}
                                        for a, p in adapters]}
        ok = (detail['selector_consumers_share_one_object'] and selector.checksum == SELECTOR_REGISTRY
              and all(role is not None for _, _, role, _ in rows)
              and all(s.registry is selector and s.identity.checksum == PUBLIC_REGISTRY for s, _ in systemic)
              and all(a.frozen.checksum == SELECTOR_REGISTRY and a.candidate.checksum == PUBLIC_REGISTRY for a, _ in adapters))
        return ok, detail

    def cards():
        slugs = ctx['slugs']
        state = {'%s:%d' % (c[:8], i): {s: (s in r.cards and r.cards[s]['active_in_gallery']) if s in r.cards else 'absent'
                                         for s in slugs} for i, (r, c, _, _) in enumerate(ctx['registries'])}
        ok = all(v is False for row in state.values() for v in row.values()) and not set(slugs) & set(ctx['composite'].eligible)
        return ok, {'active_in_gallery_before': state, 'already_eligible': sorted(set(slugs) & set(ctx['composite'].eligible))}

    def prefix():
        raw, verifier, array = ctx['raw'], ctx['verifier'], ctx['array']
        parent_rows = ctx['doc']['references'][:A.PARENT_ROWS]
        detail = {'raw_rows_equal': raw.references == parent_rows,
                  'raw_vectors_byte_identical': raw.vectors.tobytes() == array[:A.PARENT_ROWS].tobytes(),
                  'verifier_rows_equal': verifier.references == parent_rows,
                  'verifier_gallery_sha': verifier.gallery['sha256'], 'shared_rows_list': raw.references is verifier.references}
        return (detail['raw_rows_equal'] and detail['raw_vectors_byte_identical'] and detail['verifier_rows_equal']
                and detail['verifier_gallery_sha'] == A.PARENT_GALLERY_SHA), detail

    def formulas():
        frozen, short, composite, verifier = ctx['frozen'], ctx['short'], ctx['composite'], ctx['verifier']
        ctx['table'] = P.load_table(root, short.context, short.registry, short.table_checksum)
        detail = {'composite': _composite_checksum(composite) == composite.checksum,
                  'verifier': _verifier_identity(verifier, verifier.registry) == verifier.identity,
                  'producer_name': all(_producer_checksum(n) == n.checksum for n in ctx['names']),
                  'phrase_rebuild': CatalogPhraseIndex(frozen, sorted(composite.eligible)).checksum == frozen.checksum,
                  'short_rebuild': P.ShortProducerIndex(frozen, ctx['table']).checksum == short.checksum}
        return all(detail.values()), detail

    def conflicts():
        rows = [*ctx['raw'].references, *ctx['added']]
        excluded = graph.repair.guard.excluded_references(rows)
        ctx['excluded'] = excluded
        return excluded == ctx['filtered'].excluded, {'excluded_after_append': len(excluded),
                                                      'excluded_now': len(ctx['filtered'].excluded)}

    def policies():
        slugs = {r['slug'] for r in ctx['doc']['references']}
        ok, detail, ctx['policy_plan'] = plan_policies(graph, root, slugs, _one(ctx['found'], 'ColorVerifiedRecognition'))
        ctx['policy_target'] = slugs
        return ok, detail

    for name, fn in (('graph', graph_kind), ('build', build), ('scan', scan), ('chain', chain),
                     ('registries', registries), ('cards', cards), ('prefix', prefix), ('formulas', formulas),
                     ('conflicts', conflicts), ('catalogue_policies', policies)):
        if not check(name, fn) and name in ('build', 'scan'):
            break
    report = {'descriptor_checksum': descriptor['checksum'], 'checks': checks,
              'ready': len(checks) == 10 and all(c['ok'] for c in checks), 'mutated': False}
    return report, ctx


def public_preflight(report):
    """The preflight as published in responses: the input graph's own profile checksum stays in the external receipt."""
    out = deepcopy(report)
    for row in out['checks']:
        if row['check'] == 'graph' and isinstance(row['detail'], dict):
            row['detail'] = {'already_installed': row['detail'].get('already_installed'),
                             'parent_release_binding': 'verified' if row['ok'] else 'failed'}
    return out


def install(graph, root, descriptor):
    """Extend every gallery consumer of ``graph`` once after a complete preflight; returns (state, record)."""
    from rshb_vine.atlas_text_repair_v1.composition import TRACE_KEY
    from rshb_vine.ocr_candidate_repair_v1.injection import CatalogPhraseIndex
    from rshb_vine.text_evidence_repair_v1 import producer_names as P
    from rshb_vine.visual_core import validate_vectors
    root = Path(root).resolve()
    report, ctx = preflight(graph, root, descriptor)
    if not report['ready']:
        raise PreflightError(report)
    raw, filtered, verifier, stage = ctx['raw'], ctx['filtered'], ctx['verifier'], ctx['stage']
    frozen, short, composite, names = ctx['frozen'], ctx['short'], ctx['composite'], ctx['names']
    doc, array, receipt, added, slugs = ctx['doc'], ctx['array'], ctx['receipt'], ctx['added'], ctx['slugs']
    before = {'gallery_sha256': A.PARENT_GALLERY_SHA, 'rows': A.PARENT_ROWS, 'phrase': frozen.checksum,
              'short': short.checksum, 'composite': composite.checksum, 'composite_entries': len(composite.entries),
              'producer_name': [n.checksum for n in names], 'verifier': verifier.identity,
              'excluded_references': deepcopy(filtered.excluded)}
    steps = []
    try:
        availability = []
        for registry, checksum, role, paths in ctx['registries']:
            for slug in slugs:
                registry.cards[slug]['active_in_gallery'] = True
            availability.append({'checksum': checksum, 'role': role, 'holder_count': len(paths), 'holders': paths[:12],
                                 'cards': slugs, 'field': 'active_in_gallery', 'before': False, 'after': True,
                                 'checksum_unchanged': registry.checksum == checksum,
                                 'provenance': {'descriptor_checksum': descriptor['checksum'],
                                                'gallery_sha256': receipt['gallery_sha256'],
                                                'reason': 'card now has appended B3 gallery rows'}})
        steps.append('registry_availability')
        vectors = np.ascontiguousarray(array, dtype='float32')
        validate_vectors(vectors, len(doc['references']))
        shared_rows = raw.references is verifier.references
        raw.references.extend(deepcopy(added))
        raw.vectors = vectors
        raw.channel_ids = {kind: np.array([i for i, r in enumerate(raw.references) if r['kind'] == kind], dtype=int)
                           for kind in ('context', 'front_label')}
        steps.append('visual_index')
        if not shared_rows:
            verifier.references.extend(deepcopy(added))
        for i, row in enumerate(added, A.PARENT_ROWS):
            verifier.paths[i] = row['image_path']
        verifier.excluded = {e['reference_index']: e for e in ctx['excluded']}
        verifier.gallery = {'path': descriptor['build'] + '/gallery.json', 'sha256': receipt['gallery_sha256'],
                            'checksum': doc['checksum'], 'appended_to': A.PARENT_GALLERY_SHA}
        verifier.identity = _verifier_identity(verifier, verifier.registry)
        stage.identity['verifier_identity'] = verifier.identity
        steps.append('layout_verifier')
        eligible = sorted(set(composite.eligible) | set(slugs))
        phrase = CatalogPhraseIndex(frozen, eligible)
        if set(frozen.entries) - set(phrase.entries) or any(
                [t.forms[0] for t in phrase.entries[s]['core']] != [t.forms[0] for t in e['core']] for s, e in frozen.entries.items()):
            raise ValueError('Phrase rebuild changed an existing entry')
        frozen.entries.update({s: phrase.entries[s] for s in slugs if s in phrase.entries})
        frozen.checksum = phrase.checksum
        steps.append('phrase_index')
        rebuilt = P.ShortProducerIndex(frozen, ctx['table'])
        if any(rebuilt.short.get(s) is None for s in short.short) or rebuilt.entries is not frozen.entries:
            raise ValueError('Short-form rebuild dropped an existing card')
        short.short, short.checksum = rebuilt.short, rebuilt.checksum
        steps.append('short_index')
        composite.eligible = frozenset(eligible)
        by_token = defaultdict(set)
        for slug in composite.universe:
            for s in composite.cards[slug]['producer_tokens']:
                by_token[s].add(slug)
        composite.entries, composite.census = {}, {}
        for slug in eligible:
            row = composite._entry(slug, by_token)
            composite.census[slug] = row
            if row['status'] == 'admitted':
                composite.entries[slug] = composite.cards[slug]
        if not set(frozen.entries) <= composite.eligible:
            raise ValueError('Phrase index holds cards outside the eligible gallery slugs')
        composite.checksum = _composite_checksum(composite)
        for index in names:
            index.checksum = _producer_checksum(index)
        identity = graph.selection.identity['atlas_text_repair']
        identity.update(composite_checksum=composite.checksum, index_checksum=graph.selection.index.checksum,
                        frozen_index_checksum=frozen.checksum, composite_entries=len(composite.entries))
        steps.append('composite_and_identity')
        policy_records = apply_policies(ctx['policy_plan'])
        steps.append('catalogue_policies')
        _, recheck, remaining = plan_policies(graph, root, ctx['policy_target'], _one(ctx['found'], 'ColorVerifiedRecognition'))
        if remaining or any(p['missing_gallery_slugs'] for p in recheck['policies']):
            raise RuntimeError('A catalogue policy still misses served gallery slugs')
        if not (len(raw.references) == len(verifier.references) == raw.vectors.shape[0] == descriptor['rows']
                and raw.references[A.PARENT_ROWS:] == added == verifier.references[A.PARENT_ROWS:]
                and all(d._index is filtered for d in ctx['delegates']) and graph.selection.index.proposer is composite):
            raise RuntimeError('A consumer does not see the appended gallery')
    except Exception as error:
        setattr(graph, BLOCK, {'component': 'D', 'status': 'partial_mutation_failed', 'completed_steps': steps,
                               'error': '%s: %s' % (type(error).__name__, error), 'preflight': public_preflight(report)})
        raise
    census = {s: {k: composite.census[s][k] for k in ('status', 'reasons', 'never_added', 'equal_siblings',
                                                     'more_specific_siblings', 'frozen_indexed')} for s in slugs}
    after = {'gallery_sha256': receipt['gallery_sha256'], 'gallery_checksum': doc['checksum'], 'rows': descriptor['rows'],
             'phrase': frozen.checksum, 'short': short.checksum, 'composite': composite.checksum,
             'composite_entries': len(composite.entries), 'producer_name': [n.checksum for n in names],
             'verifier': verifier.identity}
    record = {'component': 'D', 'status': 'installed', 'descriptor_checksum': descriptor['checksum'],
              'descriptor_version': VERSION, 'added_slugs': slugs, 'added_rows': len(added), 'before': before,
              'after': after, 'preflight': public_preflight(report), 'completed_steps': steps,
              'phrase_entries_added': sorted(s for s in slugs if s in frozen.entries),
              'short_forms_added': sorted(s for s in slugs if s in short.short), 'composite_census': census,
              'registry_availability': availability, 'catalogue_policies': policy_records,
              'catalogue_policy_audit_after': recheck['policies'],
              'visual_and_layout_rows_shared_list': shared_rows, 'excluded_references_unchanged': True,
              'old_rows_vectors_byte_identical': True,
              'T_protocol_values_superseded': {'protocol_checksum': graph.installation['text']['protocol_checksum'],
                                               'composite_checksum': before['composite'], 'frozen_index_checksum': before['phrase']},
              'calibrated': False, 'probability': None, 'public_slug': None}
    setattr(graph, BLOCK, record)
    state = {'gallery': doc, 'vectors': vectors, 'receipt': receipt, 'trace_key': TRACE_KEY, 'preflight': report}
    return state, record
