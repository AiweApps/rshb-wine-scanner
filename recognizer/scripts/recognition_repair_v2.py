"""Recognition repair v2 candidate: repair-v1 51eedd7f + composed selector stages (producer-role slot, sparse geometry).

describe  -- profile/stages (no models loaded)
freeze    -- seal the candidate profile (refuses to overwrite; --producer once producer_role_v2 is admitted)
protocol  -- sealed census/class protocol, written before any matching
census    -- trigger census over the saved current-equivalent 593 responses with the selector registry (no matching)
counterfactual63 -- ambiguity rule on the cached class63 verifier rows (no matching)
class     -- ONE CPU pass on every triggered target: saved evidence -> guarded v1 replay (parity) and composed v2
             replay (real SIFT matching on the request bytes) -> unchanged resolver; focal GT pairing
protocol-combined / scope / combined -- single-SKU-target geometry scope (cached) and ONE combined CPU pass
             (producer_role pre_guard + guard v2 + geometry) over all 593 saved responses with cached correspondences
History: class63 (protocol v2) in runs/recognition-repair-v2/class-v2; superseded registry census in .../class.
serve     -- loopback candidate on 8189 (8175/8188 refused); tracebacks logged, first failure answers 503
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/recognition-repair-v2'
GATE = ROOT / 'runs/recognition-repair-20260926/open-http-gate'
GATE_FILES = ('runs/recognition-repair-20260926/open-http-gate/protocol.json',
              'runs/recognition-repair-20260926/open-http-gate/receipts/ledger.jsonl',
              'runs/recognition-repair-20260926/open-http-gate/compare/paired-rows.jsonl')
MODEL_PATH = 'runs/systemic-ranking-v2/final/model.json'
PROTOCOL = OUT / 'class-v3/protocol.json'
PROTOCOL_V2 = 'runs/recognition-repair-v2/class-v2/protocol.json'
CENSUS = OUT / 'class-v2/census.json'
CLASS63 = OUT / 'class-v2/class-single-target.json'
COUNTERFACTUAL = OUT / 'class-v3/counterfactual63.json'
CLASS = OUT / 'class-v3/class-triggered.json'
MATCH_BUDGET_SECONDS = 600
COMBINED_DIR = OUT / 'combined-r2'
COMBINED_PROTOCOL = COMBINED_DIR / 'protocol.json'
COMBINED_FAILED = 'runs/recognition-repair-v2/combined'
SCOPE_FILE = OUT / 'combined/scope-counterfactual.json'


def write_new(path, obj):
    from rshb_vine.io import write_json
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, obj)


def roster():
    from rshb_vine.io import sha256
    protocol = json.loads((GATE / 'protocol.json').read_text())
    ledger = {json.loads(l)['key']: json.loads(l) for l in (GATE / 'receipts/ledger.jsonl').read_text().splitlines()}
    paired = {json.loads(l)['key']: json.loads(l) for l in (GATE / 'compare/paired-rows.jsonl').read_text().splitlines()}
    for row in protocol['rows']:
        if row['set'] not in ('train137', 'supplemental456'):
            raise SystemExit('roster row outside train137/supplemental456: ' + row['key'])
        rec = ledger[row['key']]
        if rec['http_status'] != 200:
            continue
        path = ROOT / rec['reused_from'] if rec.get('reused_from') else GATE / 'receipts/responses' / rec['file']
        if sha256(path) != rec['response_sha256']:
            raise SystemExit('saved response SHA differs: ' + str(path))
        yield row, json.load(gzip.open(path, 'rt'))['body'], paired[row['key']]


def selector_registry():
    """The registry of the live selector/publisher: old103 baseline product bundle (live checksum 31e053df)."""
    from rshb_vine.catalog.product_registry import ProductRegistry
    from rshb_vine.io import read_json, verify
    from rshb_vine.recognition_repair_v2 import runtime
    from rshb_vine.systemic_ranking_v2.selection import BASELINE_PROFILE
    release = verify(read_json(ROOT / runtime.PARENT_RELEASE))
    registry = ProductRegistry.from_bundle(ROOT, verify(read_json(ROOT / BASELINE_PROFILE))['product_bundle'])
    if registry.checksum != runtime.SELECTOR_REGISTRY_CHECKSUM:
        raise SystemExit('selector registry differs from the profile value')
    return registry, release


def cmd_describe(a):
    from rshb_vine.recognition_repair_v2 import runtime
    print(json.dumps(runtime.describe(ROOT, a.profile), ensure_ascii=False, indent=1))


def cmd_freeze(a):
    from rshb_vine.io import seal
    from rshb_vine.recognition_repair_v2 import runtime
    profile = seal(runtime.freeze_body(ROOT, geometry=not a.no_geometry, producer=a.producer))
    write_new(ROOT / a.profile, profile)
    runtime.load_profile(ROOT, a.profile)
    print(json.dumps({'profile': a.profile, 'checksum': profile['checksum'],
                      'stages': [s['name'] for s in profile['stages']], 'producer_role': profile['producer_role']}))


def cmd_protocol(a):
    from rshb_vine.geometric_reference_probe_v1 import POLICY
    from rshb_vine.geometric_reference_probe_v1 import verifier as V
    from rshb_vine.geometric_reference_probe_v1.layout import CRITERIA
    from rshb_vine.io import read_json, seal, sha256, verify
    from rshb_vine.recognition_repair_v2 import geometry as G, runtime
    census = verify(read_json(CENSUS))
    files = (*GATE_FILES, *G.SOURCES, *runtime.GEOMETRY_INPUTS, MODEL_PATH, runtime.PARENT_RELEASE,
             'rshb_vine/recognition_repair_v2/composition.py', 'rshb_vine/recognition_repair_v2/geometry.py',
             'rshb_vine/recognition_repair_v2/runtime.py', 'scripts/recognition_repair_v2.py',
             str(CENSUS.relative_to(ROOT)), str(CLASS63.relative_to(ROOT)), PROTOCOL_V2)
    body = {
        'kind': 'recognition-repair-v2-geometry-class-protocol-v3', 'written_before_matching': True,
        'authorization': 'root bridge 13/15 (2026-09-26): competing-product ambiguity rule, cached63 counterfactual, '
                         'then ONE full triggered class; CPU only, no models/HTTP/F7',
        'supersedes': {'path': PROTOCOL_V2, 'sha256': sha256(ROOT / PROTOCOL_V2),
                       'result': 'class63 pass=false: 2 wrong->correct, 1 correct->wrong (W0233 sibling)'},
        'history': {'verifier_v3': 'runs/recognition-repair-v2/history/geometric_reference_probe_v1-v3 (SHA256SUMS)',
                    'code_after_class63': 'runs/recognition-repair-v2/history/class-v2-code'},
        'change': V.AMBIGUITY, 'no_threshold_change': 'min_unique_inliers 8 and plausible homography are the existing '
                                                      'CRITERIA; ratio 1.5, hull .2, span 1/3 unchanged',
        'population': 'every 200 row of the current-equivalent 593 open HTTP gate; validation46/locked/protected never read',
        'registry': {'identity_registry_checksum': runtime.SELECTOR_REGISTRY_CHECKSUM,
                     'reason': 'registry of the live selector pool and selected_product_output'},
        'trigger': V.TRIGGER, 'criteria': CRITERIA, 'policy': POLICY,
        'census': {'path': str(CENSUS.relative_to(ROOT)), 'checksum': census['checksum'],
                   'triggered_targets': census['triggered_targets']},
        'step1_counterfactual63': 'apply the rule to the cached class63 verifier rows (sorted by unique inliers, so a rival '
                                  'outside the saved top5 cannot block when the 5th row has < 8); no matching',
        'step2_class_scope': 'every census-triggered target of every request (single and multi-target)',
        'class_path': 'saved evidence -> GuardedOcrCandidateSelection replay (must reproduce saved feature digest and '
                      'systemic_selected) and RepairSelectionV2 [geometry stage] replay on the request bytes; runtime '
                      'checked_box pixels; runtime prepared-reference cache (%d)' % G.PREPARED_LIMIT,
        'labels': 'focal GT only, as text_proof: single-target request -> its target; multi-target -> paired '
                  'cur.matched_target; other targets unlabeled background; gt_products are snapshot-03 ProductIDs '
                  'compared through pins.product03(); labels read after selection for pairing only',
        'pass_criteria': ['complete within the matching budget', 'baseline replay parity on every processed target',
                          'replay trigger equals census trigger on every processed target', 'no exception',
                          '0 focal labelled correct -> wrong changes'],
        'reported_not_gated': ['every selection change with case-level cause, every competing rival (not only top5)',
                               'background/unlabeled changes as unknown, not improvements', 'every focal GT-rank change',
                               'per-target stage CPU/wall, per-request added wall seconds, prepared-reference cache hits/misses'],
        'limits': ['layout is style/layout evidence, not exact vintage; vintage stays unresolved by this stage'],
        'matching_budget_seconds': MATCH_BUDGET_SECONDS,
        'no_threshold_or_criteria_change_after_results': True,
        'next': 'report to root; runtime admission, HTTP and release staging are root-owned',
        'files_sha256': {p: sha256(ROOT / p) for p in dict.fromkeys(files)}}
    write_new(PROTOCOL, seal(body))
    print(json.dumps({'protocol': str(PROTOCOL.relative_to(ROOT))}))


def _protocol():
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(PROTOCOL))
    for path, value in protocol['files_sha256'].items():
        if sha256(ROOT / path) != value:
            raise SystemExit('protocol file changed: ' + path)
    return protocol


def _saved_target(body, sid):
    return next(t for t in body['targets'] if str(t['instance_id']) == sid)


def _saved_injection(body, sid):
    return next((e for e in (body.get('recognition_repair_v1') or {}).get('ocr_candidate_injection', [])
                 if str(e['instance_id']) == sid), None)


def cmd_census(a):
    from rshb_vine.geometric_reference_probe_v1 import verifier as V
    from rshb_vine.io import seal
    from rshb_vine.recognition_repair_v2 import geometry as G
    protocol = _protocol()
    registry, _ = selector_registry()
    verifier = G.SelectorBoundLayoutVerifier(ROOT, registry)
    rows, requests = [], 0
    for row, body, paired in roster():
        requests += 1
        evidence = (body.get('product_identity_evidence') or {}).get('targets', [])
        for t in evidence:
            sid = str(t['instance_id'])
            target = _saved_target(body, sid)
            injection = _saved_injection(body, sid)
            eligible, conditions = V.sparse_trigger(body, target, registry)
            indices = verifier.eligible_references(target) if eligible else []
            products = {V.product_of(registry, verifier.references[i]['slug']) for i in indices}
            rows.append({'key': row['key'], 'cohort': row['cohort'], 'instance_id': sid,
                         'single_target': len(evidence) == 1, 'eligible': eligible, 'conditions': conditions,
                         'injection_present': injection is not None, 'references': len(indices),
                         'products': len(products), 'within_budget': 2 <= len(products) <= verifier.max_products})
    triggered = [r for r in rows if r['eligible']]
    single = [r for r in triggered if r['single_target']]
    fails = {k: sum(not r['conditions'][k] for r in rows) for k in ('label_view', 'visual_disagreement', 'no_full_text_identity')}
    out = seal({'kind': 'recognition-repair-v2-trigger-census-v1', 'protocol_checksum': protocol['checksum'],
                'identity_registry_checksum': registry.checksum, 'product_bundle_checksum': registry.bundle_checksum,
                'requests': requests, 'targets': len(rows),
                'triggered_targets': len(triggered), 'single_target_triggered': len(single),
                'single_target_within_budget': sum(r['within_budget'] for r in single),
                'single_target_pairs': sum(r['references'] for r in single if r['within_budget']),
                'failing_condition_counts': fails, 'rows': rows})
    write_new(CENSUS, out)
    print(json.dumps({k: out[k] for k in ('targets', 'triggered_targets', 'single_target_triggered',
                                          'single_target_within_budget', 'single_target_pairs')}))


class Replay:
    def __init__(self):
        from rshb_vine.b3_only_v1.runtime import ablate
        from rshb_vine.ocr_candidate_repair_v1.guarded import GuardedOcrCandidateSelection
        from rshb_vine.recognition_repair_v2 import geometry as G
        from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2
        from rshb_vine.reference_conflicts_v1.guard import ReferenceConflicts
        from rshb_vine.systemic_ranking_v2.selection import SystemicSelection
        registry, release = selector_registry()
        inner = SystemicSelection.from_profile(ROOT, model_path=MODEL_PATH, public='systemic')
        ranker = ablate(inner.ranker)
        if ranker['checksum'] != release['ablated_ranker_checksum']:
            raise SystemExit('ablated ranker differs from the release')
        inner.use_ranker(ranker)
        if inner.registry.checksum != registry.checksum:
            raise SystemExit('replay selector registry differs from the release registry')
        self.baseline = GuardedOcrCandidateSelection(ROOT, inner, ReferenceConflicts(ROOT).injection_conflict)
        self.stage = G.GeometryStage(ROOT, inner.registry)
        self.v2 = RepairSelectionV2(self.baseline, [self.stage])
        self.registry = inner.registry
        from rshb_vine.catalog_training_evaluation_v2 import pins
        self.mapping = pins.product03()

    def order(self, out):
        products = []
        for r in out['raw_proposal']['ranked_candidates']:
            for slug in sorted(r['card_slugs']):
                p = self.mapping.get(slug)
                if p and p not in products:
                    products.append(p)
        return products

    def summary(self, out, gt):
        order = self.order(out)
        selected = out['proposal']['representative_slug']
        product = self.mapping.get(selected)
        return {'selected': selected, 'product03': product, 'selector_product_id': out['proposal']['product_id'],
                'correct': (product in gt) if gt else None,
                'gt_rank': next((i + 1 for i, p in enumerate(order) if p in gt), None) if gt else None,
                'top3': [[r['representative_slug'], round(r['score'], 4), r.get('learned_rank', r['rank'])]
                         for r in out['raw_proposal']['ranked_candidates'][:3]]}


def cmd_counterfactual(a):
    from rshb_vine.geometric_reference_probe_v1 import verifier as V
    from rshb_vine.io import read_json, seal, verify
    protocol = _protocol()
    cached = verify(read_json(CLASS63))
    rows = []
    for r in cached['rows']:
        if r['action'] != 'promoted_through_resolver':
            rows.append({'key': r['key'], 'instance_id': r['instance_id'], 'before_action': r['action'],
                         'after_action': r['action'], 'change_before': r['change'], 'reason': 'rule only blocks promotions'})
            continue
        top5 = r['top5']
        if len(top5) == 5 and V.competing(top5[4]):
            raise SystemExit('a rival outside the cached top5 could compete; counterfactual needs new matching: ' + r['key'])
        rivals = [x for x in top5[1:] if V.competing(x)]
        rows.append({'key': r['key'], 'instance_id': r['instance_id'], 'before_action': r['action'],
                     'after_action': 'ambiguous_competing_layout' if rivals else r['action'],
                     'change_before': r['change'], 'change_after': None if rivals else r['change'],
                     'competing_rivals': rivals, 'selected_before': r['v2']['selected'],
                     'selected_after': r['baseline']['selected'] if rivals else r['v2']['selected']})
    promoted = [x for x in rows if x['before_action'] == 'promoted_through_resolver']
    out = seal({'kind': 'recognition-repair-v2-geometry-counterfactual63-v1', 'protocol_checksum': protocol['checksum'],
                'class63_checksum': cached['checksum'], 'new_matching': False,
                'promotions_before': len(promoted),
                'promotions_after': sum(x['after_action'] == 'promoted_through_resolver' for x in promoted),
                'fixes_kept': [x['key'] for x in promoted if x['change_after'] == 'wrong_to_correct'],
                'fixes_lost': [x['key'] for x in promoted if x['change_before'] == 'wrong_to_correct' and x['change_after'] != 'wrong_to_correct'],
                'regressions_removed': [x['key'] for x in promoted if x['change_before'] == 'correct_to_wrong' and x['change_after'] is None],
                'regressions_left': [x['key'] for x in promoted if x.get('change_after') == 'correct_to_wrong'],
                'rows': rows})
    write_new(COUNTERFACTUAL, out)
    print(json.dumps({k: out[k] for k in ('promotions_before', 'promotions_after', 'fixes_kept', 'fixes_lost',
                                          'regressions_removed', 'regressions_left')}))


def focal(n_targets, sid, paired):
    if n_targets == 1:
        return 'single'
    matched = (paired.get('cur') or {}).get('matched_target')
    return 'matched' if matched is not None and str(matched) == sid else None


def cmd_class(a):
    from rshb_vine.interaction_ranker_v1.source import trace_result
    from rshb_vine.io import read_json, seal, sha256, verify
    from rshb_vine.ocr_candidate_repair_v1 import injection as I
    from rshb_vine.recognition_repair_v2 import geometry as G
    protocol = _protocol()
    census = verify(read_json(CENSUS))
    scope = {(r['key'], r['instance_id']): r for r in census['rows'] if r['eligible']}
    rp = Replay()
    verifier = rp.stage.verifier
    cache = {'hits': 0, 'misses': 0}
    prepare = verifier._reference

    def counted(index):
        cache['hits' if index in verifier._prepared else 'misses'] += 1
        return prepare(index)
    verifier._reference = counted
    rows, requests, started, matching, stopped = [], [], time.perf_counter(), 0.0, False
    for row, body, paired in roster():
        evidence = (body.get('product_identity_evidence') or {}).get('targets', [])
        triggered = [t for t in evidence if (row['key'], str(t['instance_id'])) in scope]
        if not triggered:
            continue
        if sha256(ROOT / row['path']) != row['sha256']:
            raise SystemExit('query SHA differs: ' + row['key'])
        data = (ROOT / row['path']).read_bytes()
        gt = set(paired.get('gt_products') or [])
        request_wall, before_cache = 0.0, dict(cache)
        with rp.v2.replaying(body, data):
            for t in triggered:
                sid = str(t['instance_id'])
                kind = focal(len(evidence), sid, paired)
                labels = gt if kind else set()
                saved = next(r for r in body['systemic_ranking_v2']['targets'] if str(r['instance_id']) == sid)
                provenance, _ = trace_result(body, t['instance_id'], t['ocr_observations'])
                call = dict(control_slug=t['control_slug'], raw_visual=t['raw_visual'], observations=t['ocr_observations'],
                            control_candidates=t['control_candidates'], provenance=provenance,
                            other_line_keys=I.scene_line_keys(body, t['instance_id']))
                base = rp.baseline.evaluate(**call)
                parity = {'feature_digest': base['feature_digest'] == saved['systemic_feature_digest'],
                          'selected': base['proposal']['representative_slug'] == saved['systemic_selected']}
                if not all(parity.values()):
                    raise SystemExit('baseline replay parity failed: %s %s %s' % (row['key'], sid, parity))
                tick = time.perf_counter()
                out = rp.v2.evaluate(**call)
                wall = time.perf_counter() - tick
                matching += wall
                request_wall += wall
                trace = out['layout_geometry']
                if trace.get('trigger') is None or trace['trigger'] != scope[(row['key'], sid)]['conditions']:
                    raise SystemExit('replay trigger differs from census: %s %s' % (row['key'], sid))
                b, v = rp.summary(base, labels), rp.summary(out, labels)
                change = None
                if b['selected'] != v['selected']:
                    change = ('unlabeled' if not labels else 'correct_to_correct' if b['correct'] and v['correct']
                              else 'correct_to_wrong' if b['correct'] else 'wrong_to_correct' if v['correct'] else 'wrong_to_wrong')
                result = trace.get('verifier') or {}
                rows.append({'key': row['key'], 'cohort': row['cohort'], 'set': row['set'], 'instance_id': sid,
                             'n_targets': len(evidence), 'focal': kind, 'gt_products': sorted(labels),
                             'stage_cpu_seconds': trace.get('cpu_seconds'), 'stage_wall_seconds': round(wall, 3),
                             'references_cached_before': trace.get('references_cached_before'),
                             'parity': parity, 'baseline': b, 'v2': v, 'change': change, 'action': trace['action'],
                             'pixels': trace.get('pixels'), 'resolver': trace.get('resolver'),
                             'winner_to_next': result.get('winner_to_next'), 'pairs': result.get('pairs'),
                             'competing_rivals': result.get('competing_rivals'), 'top5': result.get('top5'),
                             'incumbent_row': result.get('incumbent_row')})
                print(row['key'], sid, trace['action'], change or '', '%.0fs' % matching, flush=True)
                if matching > MATCH_BUDGET_SECONDS:
                    stopped = True
                    break
        requests.append({'key': row['key'], 'n_targets': len(evidence), 'triggered': len(triggered),
                         'added_wall_seconds': round(request_wall, 3),
                         'cache_hits': cache['hits'] - before_cache['hits'],
                         'cache_misses': cache['misses'] - before_cache['misses']})
        if stopped:
            break
    complete = len(rows) == len(scope)
    counts = {'action': {}, 'change': {}, 'change_focal': {}}
    for r in rows:
        counts['action'][r['action']] = counts['action'].get(r['action'], 0) + 1
        if r['change']:
            counts['change'][r['change']] = counts['change'].get(r['change'], 0) + 1
            if r['focal']:
                counts['change_focal'][r['change']] = counts['change_focal'].get(r['change'], 0) + 1
    labelled = [r for r in rows if r['gt_products']]
    changes = [{k: r[k] for k in ('key', 'instance_id', 'n_targets', 'focal', 'change', 'action', 'winner_to_next',
                                  'resolver', 'competing_rivals')}
               | {'before': r['baseline']['selected'], 'after': r['v2']['selected'],
                  'gt_rank': [r['baseline']['gt_rank'], r['v2']['gt_rank']]} for r in rows if r['change']]
    rank_changes = [{'key': r['key'], 'instance_id': r['instance_id'], 'gt_rank': [r['baseline']['gt_rank'], r['v2']['gt_rank']]}
                    for r in labelled if r['baseline']['gt_rank'] != r['v2']['gt_rank']]
    cpu = sorted(r['stage_cpu_seconds'] for r in rows if r['stage_cpu_seconds'] is not None)
    added = sorted(q['added_wall_seconds'] for q in requests)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    out = seal({'kind': 'recognition-repair-v2-geometry-class-triggered-v1', 'protocol_checksum': protocol['checksum'],
                'census_checksum': census['checksum'], 'scope_targets': len(scope), 'processed': len(rows),
                'complete': complete, 'stopped_by_budget': stopped, 'matching_seconds': round(matching, 1),
                'total_seconds': round(time.perf_counter() - started, 1), 'counts': counts,
                'focal_labelled': len(labelled),
                'focal_labelled_baseline_correct': sum(bool(r['baseline']['correct']) for r in labelled),
                'focal_labelled_v2_correct': sum(bool(r['v2']['correct']) for r in labelled),
                'pass': complete and not counts['change_focal'].get('correct_to_wrong'),
                'changes': changes, 'focal_gt_rank_changes': rank_changes,
                'stage_cpu_seconds': {'n': len(cpu), 'median': pct(cpu, .5), 'p95': pct(cpu, .95), 'max': cpu[-1] if cpu else None,
                                      'sum': round(sum(cpu), 2)},
                'request_added_wall_seconds': {'requests': len(requests), 'median': pct(added, .5), 'p95': pct(added, .95),
                                               'max': added[-1] if added else None},
                'prepared_reference_cache': dict(cache, limit=G.PREPARED_LIMIT,
                                                 note='hits accumulate over this replay order; a fresh service starts cold'),
                'stage_identity': rp.stage.identity, 'requests': requests, 'rows': rows})
    write_new(CLASS, out)
    print(json.dumps({k: out[k] for k in ('scope_targets', 'processed', 'complete', 'matching_seconds', 'counts',
                                          'focal_labelled', 'focal_labelled_baseline_correct', 'focal_labelled_v2_correct',
                                          'pass', 'focal_gt_rank_changes', 'stage_cpu_seconds', 'request_added_wall_seconds',
                                          'prepared_reference_cache')}, ensure_ascii=False))
    print(json.dumps(out['changes'], ensure_ascii=False))


def cmd_protocol_combined(a):
    from rshb_vine.geometric_reference_probe_v1 import verifier as V
    from rshb_vine.io import read_json, seal, sha256, verify
    from rshb_vine.recognition_repair_v2 import geometry as G, producer as P, runtime
    P.check_pin(ROOT)
    class3 = verify(read_json(CLASS))
    files = (*GATE_FILES, *G.SOURCES, *P.SOURCES, P.PIN, *runtime.GEOMETRY_INPUTS, MODEL_PATH, runtime.PARENT_RELEASE,
             'rshb_vine/recognition_repair_v2/composition.py', 'rshb_vine/recognition_repair_v2/geometry.py',
             'rshb_vine/recognition_repair_v2/producer.py', 'rshb_vine/recognition_repair_v2/runtime.py',
             'scripts/recognition_repair_v2.py', str(CENSUS.relative_to(ROOT)), str(CLASS.relative_to(ROOT)),
             str(PROTOCOL.relative_to(ROOT)), str(SCOPE_FILE.relative_to(ROOT)))
    body = {
        'kind': 'recognition-repair-v2-combined-cpu-protocol-v1', 'written_before_run': True,
        'authorization': 'root bridge 17/18/20/22/23 (2026-09-26): single-SKU-target geometry scope, cached scope '
                         'counterfactual, ONE combined CPU selector pass on all 593 with cached correspondences; '
                         'no models/HTTP/F7',
        'class_v3': {'path': str(CLASS.relative_to(ROOT)), 'checksum': class3['checksum'],
                     'kept_as': 'raw latency evidence and geometry class result; not an independent quality claim'},
        'geometry_change_since_class_v3': G.SCOPE,
        'supersedes': {'path': COMBINED_FAILED + '/protocol.json', 'sha256': sha256(ROOT / COMBINED_FAILED / 'protocol.json'),
                       'rows_sha256': sha256(ROOT / COMBINED_FAILED / 'rows.jsonl'),
                       'failure': 'cache replayed class_v3 STAGE actions (promoted_through_resolver) as verifier actions, '
                                  'so the 2 layout promotions were traced but not applied; fixed by mapping stage actions '
                                  'back to layout_disagrees_with_incumbent and a per-target check that cached rows '
                                  'reproduce the class_v3 stage action',
                       'kept_valid': 'scope-counterfactual.json (no matching, unaffected)'},
        'producer': {'pin': P.PIN, 'pin_checksum': P.PIN_CHECKSUM, 'flags': 'R.FULL', 'position': 'pre_guard',
                     'scope': 'all targets'},
        'decision_rule': V.AMBIGUITY,
        'stage_order': 'injection -> frozen ranker -> resolver -> producer_role (pre_guard) -> guard v2 -> geometry '
                       '(post_guard, single SKU target pass only)',
        'scope_counterfactual': 'class_v3 rows with n_targets > 1 become outside_single_target_scope; no matching',
        'combined_population': 'every 200 row of the current-equivalent 593 gate, every product evidence target',
        'geometry_cache': 'class_v3 verifier output is reused only when image SHA, selector pool union, registry, '
                          'census trigger and incumbent (post producer+guard proposal) equal the class_v3 values; a '
                          'proposing row gets layout_evidence recomputed from the cached winner ProductID and the live '
                          'pool; otherwise the real verifier runs (counted as new matching); the resolver always reruns',
        'labels': 'focal GT only (single target, or paired cur.matched_target); snapshot-03 via pins.product03(); '
                  'read after selection',
        'outputs': ['combined/scope-counterfactual.json', 'combined/rows.jsonl (per target: selected, full raw order, '
                    'stage traces, focal GT metrics, baseline/combined diff)', 'combined/summary.json'],
        'gates': ['baseline replay parity on every target', 'no exception',
                  '0 focal labelled correct -> wrong', 'every change listed with stage cause; unlabeled changes unknown'],
        'no_threshold_or_criteria_change_after_results': True,
        'files_sha256': {p: sha256(ROOT / p) for p in dict.fromkeys(files)}}
    write_new(COMBINED_PROTOCOL, seal(body))
    print(json.dumps({'protocol': str(COMBINED_PROTOCOL.relative_to(ROOT))}))


def _combined_protocol():
    from rshb_vine.io import read_json, sha256, verify
    protocol = verify(read_json(COMBINED_PROTOCOL))
    for path, value in protocol['files_sha256'].items():
        if sha256(ROOT / path) != value:
            raise SystemExit('combined protocol file changed: ' + path)
    return protocol


def cmd_scope(a):
    from rshb_vine.io import read_json, seal, verify
    protocol = _combined_protocol()
    class3 = verify(read_json(CLASS))
    single = [r for r in class3['rows'] if r['n_targets'] == 1]
    multi = [r for r in class3['rows'] if r['n_targets'] > 1]
    requests = [q for q in class3['requests']]
    kept = [q['added_wall_seconds'] for q in requests if q['n_targets'] == 1]
    dropped = [q['added_wall_seconds'] for q in requests if q['n_targets'] > 1]
    pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else None
    selectable = {}
    for row, body, paired in roster():
        if any(r['key'] == row['key'] for r in single):
            n = (body.get('target_contract') or {}).get('selectable_bottles')
            selectable[str(n)] = selectable.get(str(n), 0) + 1
    out = seal({'kind': 'recognition-repair-v2-geometry-scope-counterfactual-v1', 'protocol_checksum': protocol['checksum'],
                'class_v3_checksum': class3['checksum'], 'new_matching': False, 'scope': 'single SKU target pass',
                'targets_in_scope': len(single), 'targets_out_of_scope': len(multi),
                'changes_in_scope': [[r['key'], r['instance_id'], r['change']] for r in single if r['change']],
                'changes_lost': [[r['key'], r['instance_id'], r['change']] for r in multi if r['change']],
                'actions_out_of_scope': {x: sum(r['action'] == x for r in multi) for x in {r['action'] for r in multi}},
                'in_scope_target_contract_selectable_bottles': selectable,
                'added_wall_seconds_kept_requests': {'n': len(kept), 'median': pct(kept, .5), 'p95': pct(kept, .95),
                                                     'max': max(kept, default=None)},
                'added_wall_seconds_removed_requests': {'n': len(dropped), 'median': pct(dropped, .5),
                                                        'p95': pct(dropped, .95), 'max': max(dropped, default=None)},
                'limitation': 'geometry does not run on multi-target passes; their residual layout errors stay unaddressed'})
    write_new(SCOPE_FILE, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('checksum',)}, ensure_ascii=False))


class CachedCorrespondence:
    """class_v3 verifier output for an unchanged target/incumbent; the real verifier otherwise."""

    def __init__(self, verifier, class3):
        self.verifier, self.real = verifier, verifier.evaluate_target
        self.rows = {(r['key'], r['instance_id']): r for r in class3['rows']}
        self.key, self.stats = None, {'reused': 0, 'matched': 0, 'reasons': {}}

    def __call__(self, image, body, target, force=False):
        from rshb_vine.geometric_reference_probe_v1 import verifier as V
        sid = str(target['instance_id'])
        row = self.rows.get((self.key, sid))
        incumbent = target['retrieval'].get('best_candidate')
        reason = ('no_class_v3_row' if row is None else 'incumbent_differs' if row['baseline']['selected'] != incumbent
                  else 'pairs_differ' if row['pairs'] is not None and row['pairs'] != len(self.verifier.eligible_references(target))
                  else None)
        if reason:
            self.stats['matched'] += 1
            self.stats['reasons'][reason] = self.stats['reasons'].get(reason, 0) + 1
            return dict(self.real(image, body, target, force), cache='matched:' + reason)
        self.stats['reused'] += 1
        action = G_PROPOSING if row['action'] in STAGE_AFTER_PROPOSAL else row['action']
        out = {'instance_id': sid, 'incumbent': incumbent, 'incumbent_retained': True, 'resolver_required': True,
               'action': action, 'winner_to_next': row['winner_to_next'], 'pairs': row['pairs'],
               'competing_rivals': row['competing_rivals'], 'top5': row['top5'], 'incumbent_row': row['incumbent_row'],
               'cache': 'class_v3'}
        if action in (G_PROPOSING, 'agrees_with_incumbent', 'ambiguous_competing_layout'):
            pid = row['top5'][0]['product_id']
            out['layout_evidence'] = {'product_id': pid, 'card_slugs_in_union': sorted(
                s for s in target['retrieval'].get('candidate_union', []) if V.product_of(self.verifier.registry, s) == pid),
                'kind': 'positive layout consistency, not exact identity or vintage'}
        return out


G_PROPOSING = 'layout_disagrees_with_incumbent'
STAGE_AFTER_PROPOSAL = ('promoted_through_resolver', 'abstain_existing_resolver_blocked',
                        'abstain_winner_not_one_pool_candidate')


def _raw_order(out):
    return [[r['candidate_id'], r['representative_slug'], round(r['score'], 6), r.get('learned_rank', r['rank'])]
            for r in out['raw_proposal']['ranked_candidates']]


def cmd_combined(a):
    from rshb_vine.interaction_ranker_v1.source import trace_result
    from rshb_vine.io import read_json, seal, sha256, verify
    from rshb_vine.ocr_candidate_repair_v1 import injection as I
    from rshb_vine.recognition_repair_v2 import producer as P
    from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2
    protocol = _combined_protocol()
    class3 = verify(read_json(CLASS))
    census = verify(read_json(CENSUS))
    triggered = {(r['key'], r['instance_id']) for r in census['rows'] if r['eligible']}
    rp = Replay()
    producer = P.ProducerRoleStage(ROOT, rp.baseline.inner)
    combined = RepairSelectionV2(rp.baseline, [producer, rp.stage])
    cache = CachedCorrespondence(rp.stage.verifier, class3)
    rp.stage.verifier.evaluate_target = cache
    path = COMBINED_DIR / 'rows.jsonl'
    if path.exists():
        raise SystemExit('refusing to overwrite ' + str(path.relative_to(ROOT)))
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    rows, started = [], time.perf_counter()
    with path.open('w') as stream:
        for row, body, paired in roster():
            evidence = (body.get('product_identity_evidence') or {}).get('targets', [])
            data = (ROOT / row['path']).read_bytes()
            if sha256(ROOT / row['path']) != row['sha256']:
                raise SystemExit('query SHA differs: ' + row['key'])
            gt = set(paired.get('gt_products') or [])
            cache.key = row['key']
            with combined.replaying(body, data):
                for t in evidence:
                    sid = str(t['instance_id'])
                    kind = focal(len(evidence), sid, paired)
                    labels = gt if kind else set()
                    saved = next(r for r in body['systemic_ranking_v2']['targets'] if str(r['instance_id']) == sid)
                    provenance, _ = trace_result(body, t['instance_id'], t['ocr_observations'])
                    call = dict(control_slug=t['control_slug'], raw_visual=t['raw_visual'],
                                observations=t['ocr_observations'], control_candidates=t['control_candidates'],
                                provenance=provenance, other_line_keys=I.scene_line_keys(body, t['instance_id']))
                    base = rp.baseline.evaluate(**call)
                    parity = {'feature_digest': base['feature_digest'] == saved['systemic_feature_digest'],
                              'selected': base['proposal']['representative_slug'] == saved['systemic_selected']}
                    if not all(parity.values()):
                        raise SystemExit('baseline replay parity failed: %s %s %s' % (row['key'], sid, parity))
                    out = combined.evaluate(**call)
                    b, v = rp.summary(base, labels), rp.summary(out, labels)
                    geometry, prod = out['layout_geometry'], out['producer_role']
                    cached = cache.rows.get((row['key'], sid))
                    if (geometry.get('verifier') or {}).get('cache') == 'class_v3' and geometry['action'] != cached['action']:
                        raise SystemExit('cached geometry did not reproduce the class_v3 stage action: %s %s' % (row['key'], sid))
                    change = None
                    if b['selected'] != v['selected']:
                        change = ('unlabeled' if not labels else 'correct_to_correct' if b['correct'] and v['correct']
                                  else 'correct_to_wrong' if b['correct'] else 'wrong_to_correct' if v['correct']
                                  else 'wrong_to_wrong')
                    cause = [n for n, hit in (('producer_role', prod.get('reranked') and prod['winner']['before'] != prod['winner']['after']),
                                              ('guard_v2', (out.get('discriminator_guard') or {}).get('blocked')),
                                              ('layout_geometry', geometry.get('applied'))) if hit]
                    rec = {'key': row['key'], 'cohort': row['cohort'], 'set': row['set'], 'instance_id': sid,
                           'n_targets': len(evidence), 'focal': kind, 'gt_products': sorted(labels),
                           'image_sha256': row['sha256'], 'census_triggered': (row['key'], sid) in triggered,
                           'parity': parity, 'baseline': b, 'combined': v, 'change': change, 'stage_causes': cause,
                           'baseline_raw_order': _raw_order(base), 'combined_raw_order': _raw_order(out),
                           'combined_feature_digest': out['feature_digest'],
                           'producer_role': {'reranked': prod.get('reranked'), 'winner': prod.get('winner'),
                                             'feature_changes': len(prod.get('changes', []))},
                           'guard_v2': {k: (out.get('discriminator_guard') or {}).get(k) for k in ('blocked', 'reason')},
                           'layout_geometry': {k: geometry.get(k) for k in ('action', 'applied', 'selected')}
                                              | {'cache': (geometry.get('verifier') or {}).get('cache')},
                           'resolver': (out['proposal'].get('existing_evidence_resolution') or {}).get('reason')}
                    stream.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    rows.append(rec)
    counts = {}
    for r in rows:
        if r['change']:
            key = r['change'] + (':focal' if r['focal'] else ':background')
            counts[key] = counts.get(key, 0) + 1
    labelled = [r for r in rows if r['gt_products']]
    ranks = [[r['key'], r['instance_id'], r['baseline']['gt_rank'], r['combined']['gt_rank']]
             for r in labelled if r['baseline']['gt_rank'] != r['combined']['gt_rank']]
    summary = seal({'kind': 'recognition-repair-v2-combined-cpu-summary-v1', 'protocol_checksum': protocol['checksum'],
                    'rows_sha256': sha256(path), 'requests': len({r['key'] for r in rows}), 'targets': len(rows),
                    'parity': sum(all(r['parity'].values()) for r in rows), 'changes': counts,
                    'changed': [[r['key'], r['instance_id'], r['focal'], r['change'], r['stage_causes'],
                                 r['baseline']['selected'], r['combined']['selected'],
                                 r['baseline']['gt_rank'], r['combined']['gt_rank']] for r in rows if r['change']],
                    'focal_labelled': len(labelled),
                    'focal_baseline_correct': sum(bool(r['baseline']['correct']) for r in labelled),
                    'focal_combined_correct': sum(bool(r['combined']['correct']) for r in labelled),
                    'focal_gt_rank_changes': ranks,
                    'geometry_actions': {x: sum(r['layout_geometry']['action'] == x for r in rows)
                                         for x in {r['layout_geometry']['action'] for r in rows}},
                    'geometry_cache': cache.stats,
                    'pass': not counts.get('correct_to_wrong:focal'),
                    'seconds': round(time.perf_counter() - started, 1)})
    write_new(COMBINED_DIR / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('requests', 'targets', 'parity', 'changes', 'focal_labelled',
                                              'focal_baseline_correct', 'focal_combined_correct', 'focal_gt_rank_changes',
                                              'geometry_actions', 'geometry_cache', 'pass', 'seconds')}, ensure_ascii=False))
    print(json.dumps(summary['changed'], ensure_ascii=False))


def cmd_serve(a):
    import socket
    import traceback
    import uvicorn
    from rshb_vine.recognition_repair_v2 import runtime
    from rshb_vine.target_contract_v2.runtime import create_app
    if a.port in runtime.FORBIDDEN_PORTS:
        raise SystemExit('%d is reserved; the v2 candidate never binds it' % a.port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(('127.0.0.1', a.port))
        listener.listen(128)
        pipeline = runtime.RecognitionRepairV2(ROOT, a.profile)
        log = OUT / 'candidate/serve-errors.log'
        recognize = pipeline.recognize

        def logged(*args, **kwargs):
            try:
                return recognize(*args, **kwargs)
            except Exception:
                log.parent.mkdir(parents=True, exist_ok=True)
                with open(log, 'a') as stream:
                    stream.write(time.strftime('%Y-%m-%dT%H:%M:%S ') + traceback.format_exc() + '\n')
                raise
        pipeline.recognize = logged
        print(json.dumps({'runtime': pipeline.manifest['checksum'], 'profile': pipeline.profile['checksum']}), flush=True)
        uvicorn.Server(uvicorn.Config(create_app(pipeline), host='127.0.0.1', port=a.port)).run(sockets=[listener])
    finally:
        listener.close()


def main():
    from rshb_vine.recognition_repair_v2 import runtime
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name, fn in (('describe', cmd_describe), ('freeze', cmd_freeze), ('protocol', cmd_protocol),
                     ('census', cmd_census), ('counterfactual63', cmd_counterfactual), ('class', cmd_class),
                     ('protocol-combined', cmd_protocol_combined), ('scope', cmd_scope), ('combined', cmd_combined),
                     ('serve', cmd_serve)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if name in ('describe', 'freeze', 'serve'):
            s.add_argument('--profile', default=runtime.PROFILE)
        if name == 'freeze':
            s.add_argument('--producer', action='store_true', help='seal the admitted producer_role_v2 pre-guard stage')
            s.add_argument('--no-geometry', action='store_true')
        if name == 'serve':
            s.add_argument('--port', type=int, default=runtime.PORT)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
