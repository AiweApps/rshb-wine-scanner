"""CPU replay harness of text-evidence repair v1 over the saved admitted open593 (890 product-evidence targets).

The consumer is the one of the current selector (24b679 = zero-target release over repair-v2 99451df0; the route
adapter acts before selection and is outside this replay): guarded OCR selection v1 with the 5c0be ranker,
RepairSelectionV2 stages [producer_role, layout_geometry], wrapped by TextEvidenceSelection with the arm's
components. Every target gets the saved raw OCR observations, the provenance traced on those originals and the
original scene line keys, exactly as the live ``select``; nothing in the saved body is edited.

Layout geometry: the G0 arm uses the class_v3 correspondence cache of the parity-proven repair-v2 replay. Other arms
reuse the verifier output of G0 for a target only when its verifier input fingerprint (incumbent, eligible
reference indices, sparse-trigger conditions, label view and rotation) equals the G0 one; otherwise the CPU SIFT
verifier runs and the call is logged. No GT, label or outcome is read by the selector path.
"""
import copy
import json
import sys
import time
from pathlib import Path

from rshb_vine.io import digest, read_json, sha256

ROOT = Path(__file__).resolve().parents[2]
GATE = 'runs/recognition-repair-20260926/open-http-gate'
COMBINED_R2 = 'runs/recognition-repair-v2/combined-r2/rows.jsonl'
OPEN_ROWS = 'runs/systemic-release-v3/selector/replay-v1/rows.jsonl'
LEDGER = 'runs/release-last-mile-v1/regressions/root-reviewed/ledger.jsonl'
CLASS3 = 'runs/recognition-repair-v2/class-v3/class-triggered.json'
ABLATED_CHECKSUM = '5c0be0ca42479779925e136220824f3f6b4367f85bc58081334050c07b7ce963'
OPEN_TARGETS = 890
OPEN_IMAGES = 593
MANDATORY = ('real_organizer:organizer-022', 'real_open38:internet-v2-26c1090da01c0291dcd2')
SLICES = {'product_exact_web232': 'exact_train', 'product_admitted': 'real_confirmed',
          'product_source_candidate_not_exact': 'real_provisional', 'product_visual_candidate_not_exact': 'real_provisional',
          'product_intended_reference_conditioned': 'synthetic', 'agent_audited_text_link_not_admitted': 'roskachestvo_weak'}
REAL_EXACT = ('exact_train', 'real_confirmed')
TEXT_FEATURES = ('prod.', 'name.', 'series.', 'typ.')
CLOSED_SYNTHETIC_DIR = 'runs/crop-label-v1/closed-synthetic'


def _scripts(name):
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    return __import__(name)


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ---- protected-role audit (SHA / group / capture membership only; no body, pixel, label or prediction read) ----

def audit_open593(root=ROOT):
    """Open593 image SHAs against every closed/held/calibration/consumed pool, before any saved body is read."""
    from rshb_vine.b3_ranker_refit_v1 import census as C
    from rshb_vine.coherent_challenger_v1.scope import _closed_shas
    from rshb_vine.data_protection import load_protection
    Z = _scripts('zero_target_release_v1')
    rows = read_json(root / GATE / 'protocol.json')['rows']
    if len(rows) != OPEN_IMAGES or {r['set'] for r in rows} != {'train137', 'supplemental456'}:
        raise ValueError('open593 roster differs from train137 + supplemental456')
    protection = load_protection(root)
    closed, states = _closed_shas(protection)
    explicit, sources = Z.excluded_shas(root, protection)
    member = C.membership(root)
    synthetic = sorted(p for p in (root / CLOSED_SYNTHETIC_DIR).iterdir() if p.is_file() and p.name != 'consumed.json')
    explicit['closed_synthetic20_consumed'] = {sha256(p) for p in synthetic}
    sources['closed_synthetic20_consumed'] = {'path': CLOSED_SYNTHETIC_DIR, 'images': len(synthetic),
                                              'marker_sha256': sha256(root / CLOSED_SYNTHETIC_DIR / 'consumed.json')}
    unified = C.unified(root)
    explicit['unified_held_validation_test'] = {s for s, r in unified.items()
                                                if set(r.get('held_relations') or []) & {'validation', 'test'}}
    train = {r['sha256']: r for r in read_json(root / (C.WEB232 + 'train.json'))['records']}
    hits, unknown = [], []
    for r in rows:
        sha = r['sha256']
        reasons = sorted(n for n, s in explicit.items() if sha in s)
        reasons += ['protection_closed'] if sha in closed else []
        reasons += sorted('%s:sha' % n for n, m in member.items() if sha in m['sha'])
        web = train.get(sha)
        if r['set'] == 'train137':
            if web is None:
                reasons.append('train137_not_in_web232_train')
            else:
                groups, captures = {web['group_id']}, set(web.get('capture_group_ids') or [])
                reasons += sorted('%s:group' % n for n, m in member.items() if groups & m['group'])
                reasons += sorted('%s:capture' % n for n, m in member.items() if captures & m['capture'])
        if reasons:
            hits.append({'key': r['key'], 'set': r['set'], 'sha256': sha, 'reasons': sorted(set(reasons))})
        if sha not in states and sha not in unified and web is None:
            unknown.append({'key': r['key'], 'set': r['set'], 'cohort': r['cohort']})
    return {'kind': 'text-evidence-repair-v1-open593-audit', 'roster': GATE + '/protocol.json',
            'roster_sha256': sha256(root / GATE / 'protocol.json'), 'images': len(rows),
            'protection_pointer_sha256': protection['pointer_sha256'], 'explicit_sources': sources,
            'pools': {n: len(s) for n, s in sorted(explicit.items())},
            'membership_pools': {n: {k: len(v) for k, v in m.items()} for n, m in sorted(member.items())},
            'protection_closed': len(closed), 'hits': hits,
            'unknown_role_images': {'count': len(unknown), 'rows': unknown,
                                    'meaning': 'no protection membership, unified-corpus record or web232 train record'},
            'fields_read': 'image SHA, set, cohort, web232 train group/capture ids; closed-synthetic files hashed as bytes',
            'passed': not hits}


# ---- consumer ----

class GeometryCache:
    """G0: class_v3 cache (parity-proven). Other arms: G0 verifier output under an identical input fingerprint."""

    def __init__(self, stage, registry, cached, reference=None):
        self.verifier, self.registry, self.cached = stage.verifier, registry, cached
        self.real, self.reference = cached.real, reference
        self.key, self.calls, self.outputs = None, [], {}
        self.stats = {'g0_reuse': 0, 'real': 0, 'class_v3_path': 0}

    def fingerprint(self, body, target, force):
        from rshb_vine.geometric_reference_probe_v1 import verifier as V
        view = V.label_view(target)
        retrieval = target['retrieval']
        return digest({'incumbent': retrieval.get('best_candidate'),
                       'candidate_union': sorted(retrieval.get('candidate_union') or []),
                       'eligible': self.verifier.eligible_references(target),
                       'trigger': V.sparse_trigger(body, target, self.registry)[1],
                       'label_view': [view['kind'], view['bbox'], view.get('source')],
                       'rotation': target.get('retrieval_pixel_rotation_ccw') or 0, 'force': bool(force)})

    def __call__(self, image, body, target, force=False):
        sid, fp = str(target['instance_id']), self.fingerprint(body, target, force)
        ref = self.reference.get((self.key, sid)) if self.reference is not None else None
        started = time.perf_counter()
        if self.reference is None:
            self.stats['class_v3_path'] += 1
            self.cached.key = self.key
            result = self.cached(image, body, target, force)
            source = result.get('cache', 'class_v3')
        elif ref is not None and ref['fingerprint'] == fp:
            self.stats['g0_reuse'] += 1
            result, source = dict(copy.deepcopy(ref['result']), cache='g0:' + str(ref['source'])), 'g0'
        else:
            self.stats['real'] += 1
            result = dict(self.real(image, body, target, force), cache='real:' + ('fingerprint_differs' if ref else 'no_g0_output'))
            source = 'real'
        seconds = round(time.perf_counter() - started, 3)
        self.outputs[(self.key, sid)] = {'fingerprint': fp, 'result': result, 'source': source, 'seconds': seconds}
        if source not in ('g0', 'class_v3'):
            self.calls.append({'key': self.key, 'instance_id': sid, 'incumbent': target['retrieval'].get('best_candidate'),
                               'fingerprint': fp, 'source': source, 'cache': result.get('cache'),
                               'action': result.get('action'), 'pairs': result.get('pairs'), 'seconds': seconds})
        return result


class Consumer:
    def __init__(self, components=(), geometry_reference=None, tables=None):
        from rshb_vine.io import verify
        from rshb_vine.recognition_repair_v2 import producer as P
        from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2
        from rshb_vine.text_evidence_repair_v1.composition import TextEvidenceSelection
        RR = _scripts('recognition_repair_v2')
        self.RR, self.rp = RR, RR.Replay()
        self.inner = self.rp.baseline.inner
        if self.inner.ranker['checksum'] != ABLATED_CHECKSUM:
            raise ValueError('replay ranker is not 5c0be')
        parent = RepairSelectionV2(self.rp.baseline, [P.ProducerRoleStage(RR.ROOT, self.inner), self.rp.stage])
        self.selection = TextEvidenceSelection(parent, components, tables)
        cached = RR.CachedCorrespondence(self.rp.stage.verifier, verify(read_json(ROOT / CLASS3)))
        self.geometry = GeometryCache(self.rp.stage, self.inner.registry, cached, geometry_reference)
        self.rp.stage.verifier.evaluate_target = self.geometry

    def calls(self, body):
        from rshb_vine.interaction_ranker_v1.source import trace_result
        from rshb_vine.ocr_candidate_repair_v1 import injection as I
        for t in (body.get('product_identity_evidence') or {}).get('targets', []):
            provenance, _ = trace_result(body, t['instance_id'], t['ocr_observations'])
            yield str(t['instance_id']), dict(
                control_slug=t['control_slug'], raw_visual=t['raw_visual'], observations=t['ocr_observations'],
                control_candidates=t['control_candidates'], provenance=provenance,
                other_line_keys=I.scene_line_keys(body, t['instance_id']))

    def evaluate(self, key, body, data, call):
        self.geometry.key = key
        self.geometry.outputs.pop((key, str(call['raw_visual']['instance_id'])), None)
        snapshot = copy.deepcopy(call['observations'])
        started = time.perf_counter()
        with self.selection.replaying(body, data):
            out = self.selection.evaluate(**call)
        if call['observations'] != snapshot:
            raise RuntimeError('saved raw OCR observations were modified: ' + key)
        out['replay_seconds'] = time.perf_counter() - started
        return out

    def outcome(self, out, gt):
        s = self.rp.summary(out, gt)
        proposal = out['proposal']
        s.update(raw_order=self.RR._raw_order(out), feature_digest=out['feature_digest'],
                 guard={k: (out.get('discriminator_guard') or {}).get(k) for k in ('blocked', 'reason')},
                 geometry=(out.get('layout_geometry') or {}).get('action'),
                 geometry_applied=bool((out.get('layout_geometry') or {}).get('applied')),
                 producer_reranked=(out.get('producer_role') or {}).get('reranked'),
                 injection={k: out['ocr_candidate_injection'].get(k) for k in ('status', 'injected')},
                 exact_or_probability=proposal.get('exact_slug') is not None or proposal.get('probability') is not None,
                 pool=len(out['candidate_ids']), seconds=round(out['replay_seconds'], 4))
        text = out['text_evidence_repair_v1']
        geometry = self.geometry.outputs.get((self.geometry.key, str(out['recognition_repair_v2']['instance_id'])))
        s['stage_seconds'] = {'components': round(text['seconds']['components_total'], 5),
                              **{k: v and round(v, 5) for k, v in text['seconds'].items() if k != 'components_total'},
                              'geometry_verifier': geometry and geometry['seconds'],
                              'geometry_source': geometry and geometry['source']}
        s['text'] = {'applied': text['applied'],
                     'changed_observations': (text.get('normalization') or {}).get('changed', []),
                     'other_line_keys_added': (text.get('normalization') or {}).get('other_line_keys_added', []),
                     'producer_names': text.get('producer_names'), 'name_roles': text.get('name_roles')}
        if gt:
            s['positive_text_rows'] = {
                cid: {n: v for n, v in row.items() if n.startswith(TEXT_FEATURES)}
                for cid, row, c in zip(out['candidate_ids'], out['rows'], out['base']['candidates'])
                if {self.rp.mapping.get(x) for x in c['card_slugs']} & gt}
        return s


def roster():
    """(row, body, bytes) of every saved 200 response of open593, all SHAs checked."""
    RR = _scripts('recognition_repair_v2')
    for row, body, _ in RR.roster():
        path = ROOT / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('query SHA differs: ' + row['key'])
        yield row, body, path.read_bytes()


def labels():
    open_rows = {(r['key'], r['instance_id']): r for r in jsonl(ROOT / OPEN_ROWS)}
    ledger = {(r['key'], r['instance_id']): r for r in jsonl(ROOT / LEDGER) if r.get('instance_id') is not None}
    reference = {(r['key'], r['instance_id']): r for r in jsonl(ROOT / COMBINED_R2)}
    return open_rows, ledger, reference


def run_arm(consumer, budget, sink):
    """Every open593 target through ``consumer``; ``sink(rows)`` receives the rows of each image as soon as it is done."""
    open_rows, ledger, reference = labels()
    rows = []
    for row, body, data in roster():
        image_rows = []
        for sid, call in consumer.calls(body):
            lab = open_rows[(row['key'], sid)]
            gt = set(lab['gt_products'])
            out = consumer.evaluate(row['key'], body, data, call)
            ref = reference[(row['key'], sid)]
            outcome = consumer.outcome(out, gt)
            image_rows.append({'key': row['key'], 'instance_id': sid, 'image_sha256': row['sha256'], 'set': row['set'],
                         'cohort': row['cohort'], 'focal': lab['focal'], 'gt_products': sorted(gt),
                         'gt_level': lab['gt_level'], 'slice': SLICES.get(lab['gt_level']) if gt else None,
                         'ledger_status': (ledger.get((row['key'], sid)) or {}).get('status'),
                         'repair_v1_fix_listed': bool((ledger.get((row['key'], sid)) or {}).get('repair_v1_fix_listed')),
                         'current_parity': {'selected': outcome['selected'] == ref['combined']['selected'],
                                            'raw_order': outcome['raw_order'] == ref['combined_raw_order'],
                                            'feature_digest': outcome['feature_digest'] == ref['combined_feature_digest']},
                         'outcome': outcome})
        rows.extend(image_rows)
        sink(row['key'], image_rows)
        budget.check('%d targets' % len(rows))
    if len(rows) != OPEN_TARGETS or {(r['key'], r['instance_id']) for r in rows} != set(reference):
        raise RuntimeError('replay does not cover the %d open targets' % OPEN_TARGETS)
    return rows


def _delta(before, after, gt):
    """Text-feature change on the positive (GT product) candidates present in both pools."""
    if not gt:
        return None
    b, a = before.get('positive_text_rows') or {}, after.get('positive_text_rows') or {}
    support = contra = 0
    changes = {}
    for cid in sorted(set(b) & set(a)):
        diff = {n: [b[cid][n], a[cid][n]] for n in a[cid] if a[cid][n] != b[cid].get(n)}
        if diff:
            changes[cid] = diff
            support += sum(1 for n, (x, y) in diff.items() if y > x and '.contra' not in n and '.other_g' not in n
                           and 'fuzzy' not in n)
            contra += sum(1 for n, (x, y) in diff.items() if y > x and ('.contra' in n or '.other_g' in n or 'fuzzy' in n))
    return {'positive_support_up': support, 'positive_contra_up': contra, 'changes': changes,
            'positive_added': sorted(set(a) - set(b)), 'positive_removed': sorted(set(b) - set(a))}


def compare(base_rows, arm_rows):
    """All-change ledger and gate counts of one arm against G0 (same 890 keys)."""
    from collections import Counter
    base = {(r['key'], r['instance_id']): r for r in base_rows}
    changes, by_slice, semantic = [], {}, Counter()
    for r in arm_rows:
        b = base[(r['key'], r['instance_id'])]
        bo, ao = b['outcome'], r['outcome']
        gt = set(r['gt_products'])
        kind = ('fix' if gt and ao['correct'] and not bo['correct'] else
                'regression' if gt and bo['correct'] and not ao['correct'] else
                'changed_same_label' if gt and ao['selected'] != bo['selected'] else
                'changed_unlabelled' if not gt and ao['selected'] != bo['selected'] else None)
        delta = _delta(bo, ao, gt)
        if delta:
            semantic['targets_positive_support_up'] += bool(delta['positive_support_up'])
            semantic['targets_positive_contra_up'] += bool(delta['positive_contra_up'])
        if gt:
            c = by_slice.setdefault(r['slice'], Counter())
            c['n'] += 1
            c['before_correct'] += bool(bo['correct'])
            c['after_correct'] += bool(ao['correct'])
            c[kind or 'unchanged'] += 1
        other = [n for n in ('raw_order', 'feature_digest', 'guard', 'geometry', 'injection', 'producer_reranked')
                 if bo[n] != ao[n]]
        if kind or other or ao['text']['applied']:
            changes.append({'key': r['key'], 'instance_id': r['instance_id'], 'slice': r['slice'], 'focal': r['focal'],
                            'cohort': r['cohort'], 'ledger_status': r['ledger_status'], 'kind': kind,
                            'fields_changed': other, 'gt_products': r['gt_products'],
                            'before': {k: bo[k] for k in ('selected', 'product03', 'correct', 'gt_rank', 'top3', 'guard',
                                                          'geometry', 'injection')},
                            'after': {k: ao[k] for k in ('selected', 'product03', 'correct', 'gt_rank', 'top3', 'guard',
                                                         'geometry', 'injection', 'text')},
                            'positive_text_delta': delta})
    protected = [r for r in arm_rows if base[(r['key'], r['instance_id'])]['outcome']['correct']
                 and (r['ledger_status'] in ('fixed_vs_old', 'recovered_after_b3only') or r['key'] in MANDATORY
                      or (base[(r['key'], r['instance_id'])]['outcome']['guard'] or {}).get('blocked'))]
    counts = {k: sum(c[k] for c in by_slice.values()) for k in ('fix', 'regression', 'changed_same_label')}
    return {'by_slice': {k: dict(v) for k, v in sorted(by_slice.items())}, 'counts': counts,
            'candidate_fixes_pending_front_scope': [[c['key'], c['instance_id'], c['slice']] for c in changes
                                                    if c['kind'] == 'fix' and c['slice'] in REAL_EXACT],
            'changed_unlabelled': sum(1 for c in changes if c['kind'] == 'changed_unlabelled'),
            'protected_rows': len(protected),
            'protected_broken': [(r['key'], r['instance_id']) for r in protected if not r['outcome']['correct']],
            'fix_listed_changed': [(r['key'], r['instance_id']) for r in arm_rows if r['repair_v1_fix_listed']
                                   and r['outcome']['selected'] != base[(r['key'], r['instance_id'])]['outcome']['selected']],
            'contract_violations': [(r['key'], r['instance_id']) for r in arm_rows if r['outcome']['exact_or_probability']],
            'semantic': dict(semantic), 'changes': changes,
            'text_applied_targets': sum(1 for r in arm_rows if r['outcome']['text']['applied'])}

