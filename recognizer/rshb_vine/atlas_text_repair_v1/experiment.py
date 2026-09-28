"""CPU replay of component T over saved actual responses: open593 (890 targets) and targeted atlas Talu bodies.

The consumer is the decision-consistency one (A+B+D+M over RepairSelectionV2 with the 5c0be ranker, built once);
arm ``I`` is its ``compose(('I',))`` copy, the current cd0b9910 selector, and must reproduce the saved I rows on every
parity field. Arm ``IT`` is a fresh I copy with T attached to that same object. Layout geometry (the current geometry
stage, not the rejected pairwise guard G): the saved verifier output of the text-evidence-v2 base arm (G0v2) under an
identical input fingerprint; a miss runs the same frozen CPU verifier only within the protocol's real-call budget,
otherwise the target is blocked, has no outcome and is listed. No NN or OCR. No GT, label or outcome is read by the
selector path.
"""
import copy
import gzip
import json
import time
from pathlib import Path

from rshb_vine.atlas_text_repair_v1 import composition as TC
from rshb_vine.decision_consistency_v1 import experiment as DCX
from rshb_vine.decision_consistency_v1.composition import compose
from rshb_vine.io import sha256

X, V2 = DCX.X, DCX.V2
PARITY_FIELDS = DCX.PARITY_FIELDS
ATLAS_ROWS = 'runs/wine-atlas-20260927/stable-run/per-image-evaluation.jsonl'


class GeometryMiss(Exception):
    pass


class SavedGeometry(X.GeometryCache):
    """G0v2 verifier outputs under an identical fingerprint; a miss is blocked unless the sealed protocol allows real
    calls, then the same frozen CPU verifier runs on the original pixels at most ``real_max`` times per arm and its
    output is cached by (target, fingerprint) for the later arms of the same process. Every real call is logged."""

    def __init__(self, stage, registry, cached, reference, real_max=0, computed=None):
        super().__init__(stage, registry, cached, reference or {})
        self.blocked, self.real_max = [], real_max
        self.computed = computed if computed is not None else {}

    def __call__(self, image, body, target, force=False):
        sid, fp = str(target['instance_id']), self.fingerprint(body, target, force)
        ref = self.reference.get((self.key, sid))
        started = time.perf_counter()
        if ref is not None and ref['fingerprint'] == fp:
            self.stats['g0_reuse'] += 1
            result, source = dict(copy.deepcopy(ref['result']), cache='g0:' + str(ref['source'])), 'g0'
        elif (self.key, sid, fp) in self.computed:
            self.stats['real_reuse'] = self.stats.get('real_reuse', 0) + 1
            result, source = copy.deepcopy(self.computed[(self.key, sid, fp)]), 'real_reuse'
        elif self.stats['real'] < self.real_max:
            self.stats['real'] += 1
            result = dict(self.real(image, body, target, force),
                          cache='real:' + ('fingerprint_differs' if ref else 'no_saved_output'))
            self.computed[(self.key, sid, fp)] = copy.deepcopy(result)
            source = 'real'
        else:
            self.blocked.append({'key': self.key, 'instance_id': sid, 'fingerprint': fp,
                                 'incumbent': target['retrieval'].get('best_candidate'),
                                 'reason': ('fingerprint_differs' if ref else 'no_saved_output')
                                 + ('' if not self.real_max else ':real_budget_exhausted')})
            raise GeometryMiss(self.blocked[-1]['reason'])
        seconds = round(time.perf_counter() - started, 3)
        self.outputs[(self.key, sid)] = {'fingerprint': fp, 'result': result, 'source': source, 'seconds': seconds}
        if source != 'g0':
            self.calls.append({'key': self.key, 'instance_id': sid, 'incumbent': target['retrieval'].get('best_candidate'),
                               'fingerprint': fp, 'source': source, 'cache': result.get('cache'),
                               'action': result.get('action'), 'pairs': result.get('pairs'), 'seconds': seconds})
        return result


class Consumer(DCX.Consumer):
    def __init__(self, geometry_reference, tables, root, real_geometry_max=0):
        super().__init__(geometry_reference, tables)
        self.root, self.real_geometry_max, self.computed = Path(root), real_geometry_max, {}

    def arm(self, name):
        """Fresh I copy of the base selector (``IT``: T attached to that copy) and a fresh geometry cache per arm."""
        self.selection = compose(self.base_selection, ('I',))
        self.t_identity = TC.attach(self.selection, self.root) if name == 'IT' else None
        self.geometry = SavedGeometry(self.rp.stage, self.inner.registry, self.geometry.cached, self.reference,
                                      self.real_geometry_max, self.computed)
        self.rp.stage.verifier.evaluate_target = self.geometry
        return self.selection

    def outcome(self, out, gt):
        s = super().outcome(out, gt)
        decision = out['ocr_candidate_injection']
        trace = decision.get('composite_identity')
        s['composite'] = trace and {k: trace[k] for k in ('status', 'applied', 'counts', 'skip_reasons', 'proposed',
                                                          'admitted', 'skipped', 'original')}
        s['injection_matched'] = [m['slug'] for m in decision['matched']]
        s['index_checksum'] = decision.get('index_checksum')
        return s


def open_rows(consumer, budget, sink):
    """Every open593 target; a geometry miss leaves the target blocked (no outcome) instead of computing it."""
    labels, ledger, reference = X.labels()
    rows = []
    for row, body, data in X.roster():
        image_rows = []
        for sid, call in consumer.calls(body):
            lab = labels[(row['key'], sid)]
            gt = set(lab['gt_products'])
            record = {'key': row['key'], 'instance_id': sid, 'image_sha256': row['sha256'], 'set': row['set'],
                      'cohort': row['cohort'], 'focal': lab['focal'], 'gt_products': sorted(gt),
                      'gt_level': lab['gt_level'], 'slice': X.SLICES.get(lab['gt_level']) if gt else None,
                      'ledger_status': (ledger.get((row['key'], sid)) or {}).get('status'),
                      'repair_v1_fix_listed': bool((ledger.get((row['key'], sid)) or {}).get('repair_v1_fix_listed'))}
            try:
                out = consumer.evaluate(row['key'], body, data, call)
            except GeometryMiss as miss:
                record.update(blocked=str(miss), outcome=None)
            else:
                record['outcome'] = consumer.outcome(out, gt)
            image_rows.append(record)
        rows.extend(image_rows)
        sink(row['key'], image_rows)
        budget.check('%d targets' % len(rows))
    if len(rows) != X.OPEN_TARGETS or {(r['key'], r['instance_id']) for r in rows} != set(reference):
        raise RuntimeError('replay does not cover the %d open targets' % X.OPEN_TARGETS)
    return rows


def parity(rows, saved_rows):
    """Arm I against the saved I rows of evidence-consistency v1: every parity field of every target, no blocked row."""
    from collections import Counter
    saved = {(r['key'], r['instance_id']): r['outcome'] for r in saved_rows}
    counts, differ = Counter(), []
    for r in rows:
        f = saved.get((r['key'], r['instance_id']))
        for k in PARITY_FIELDS:
            same = f is not None and r['outcome'] is not None and r['outcome'][k] == f[k]
            counts[f'{k}:{same}'] += 1
            if not same:
                differ.append([r['key'], r['instance_id'], k])
    blocked = [[r['key'], r['instance_id']] for r in rows if r['outcome'] is None]
    return {'counts': dict(counts), 'differ': differ[:50], 'blocked': blocked,
            'passed': len(saved) == len(rows) == X.OPEN_TARGETS and not differ and not blocked}


def compare(base_rows, arm_rows, producers):
    """v2 ledger over targets evaluated in both arms, blocked targets listed, and the T census of every trace."""
    base = {(r['key'], r['instance_id']): r for r in base_rows}
    blocked = [[r['key'], r['instance_id'], r['blocked']] for r in arm_rows if r['outcome'] is None]
    kept = [r for r in arm_rows if r['outcome'] is not None]
    result = V2.compare([base[(r['key'], r['instance_id'])] for r in kept], kept, producers)
    census = []
    for r in kept:
        o, b = r['outcome'], base[(r['key'], r['instance_id'])]['outcome']
        c = o.get('composite') or {}
        if c.get('status') not in (None, 'no_composite_match'):
            census.append({'key': r['key'], 'instance_id': r['instance_id'], 'slice': r['slice'],
                           'gt_products': r['gt_products'], 'composite': c,
                           'before': {k: b[k] for k in ('selected', 'correct', 'gt_rank', 'injection')},
                           'after': {k: o[k] for k in ('selected', 'correct', 'gt_rank', 'injection')},
                           'selection_changed': b['selected'] != o['selected']})
    for c in result['changes']:
        o = next(r['outcome'] for r in kept if (r['key'], r['instance_id']) == (c['key'], c['instance_id']))
        c['composite'] = o.get('composite')
    result['blocked'] = blocked
    result['census'] = {'targets_with_t_match': len(census),
                        'targets_t_injected': sum(1 for x in census if x['composite']['admitted']),
                        'targets_selection_changed': sum(1 for x in census if x['selection_changed']),
                        'skip_reasons': _reasons(census)}
    return result, census


def _reasons(census):
    from collections import Counter
    return dict(sorted(Counter(s['reason'] for x in census for s in x['composite']['skipped']).items()))


# ---- targeted atlas diagnostic (saved live cd0b9910 bodies; explanatory, never an independent quality claim) ----

def atlas_items(root, indices):
    rows = {r['index']: r for r in X.jsonl(Path(root) / ATLAS_ROWS)}
    out = []
    for n in indices:
        r = rows[n]
        if sha256(Path(root) / r['input_path']) != r['input_sha256'] or sha256(Path(root) / r['response_path']) != r['response_sha256']:
            raise ValueError('atlas input/response SHA differs: %d' % n)
        out.append(r)
    return out


def atlas_rows(consumer, root, items):
    """Per atlas target: saved live record, replayed selection; blocked on a geometry call (no saved cache there)."""
    rows = []
    for r in items:
        body = json.load(gzip.open(Path(root) / r['response_path'], 'rt'))
        data = (Path(root) / r['input_path']).read_bytes()
        live = {str(t['instance_id']): t for t in (body.get('systemic_ranking_v2') or {}).get('targets', [])}
        for sid, call in consumer.calls(body):
            saved = live.get(sid) or {}
            row = {'index': r['index'], 'group': r['group'], 'role': r['role'], 'gt_candidates': r['gt_candidates'],
                   'instance_id': sid, 'live': {'selected': saved.get('systemic_selected'),
                                                'feature_digest': saved.get('systemic_feature_digest'),
                                                'injection': saved.get('ocr_candidate_injection')}}
            started = time.perf_counter()
            try:
                out = consumer.evaluate('atlas:%d' % r['index'], body, data, call)
            except GeometryMiss as miss:
                row.update(blocked=str(miss), replay=None)
            else:
                decision = out['ocr_candidate_injection']
                row['replay'] = {'selected': out['proposal']['representative_slug'],
                                 'feature_digest': out['feature_digest'],
                                 'injection': {k: decision.get(k) for k in ('rule', 'index_checksum', 'status', 'injected',
                                                                           'skipped', 'matched_products')},
                                 'composite': decision.get('composite_identity'),
                                 'top5': [x['representative_slug'] if 'representative_slug' in x else x['candidate_id']
                                          for x in out['proposal'].get('ranked_candidates', [])[:5]],
                                 'guard': {k: (out.get('discriminator_guard') or {}).get(k) for k in ('blocked', 'reason')},
                                 'seconds': round(time.perf_counter() - started, 3)}
            rows.append(row)
    return rows
