"""CPU replay harness of text-evidence repair v2 over the saved admitted open593 (890 targets).

The consumer, inputs, roster, labels, geometry cache and comparison are those of the v1 harness; the selection is
TextEvidenceSelectionV2 over RepairSelectionV2(guarded v1 + 5c0be, [producer_role, layout_geometry]). With no
component it is the v1 G0 consumer. The v2 ledger adds, per labelled change, the positive-candidate text support
decreases and the GT score delta, not only increases.
"""
from rshb_vine.text_evidence_repair_v1 import experiment as X

jsonl, roster, labels, run_arm = X.jsonl, X.roster, X.labels, X.run_arm
OPEN_TARGETS, OPEN_IMAGES, ABLATED_CHECKSUM = X.OPEN_TARGETS, X.OPEN_IMAGES, X.ABLATED_CHECKSUM


class Consumer(X.Consumer):
    def __init__(self, components=(), geometry_reference=None, tables=None):
        from rshb_vine.io import read_json, verify
        from rshb_vine.recognition_repair_v2 import producer as P
        from rshb_vine.recognition_repair_v2.composition import RepairSelectionV2
        from rshb_vine.text_evidence_repair_v2.composition import TextEvidenceSelectionV2
        RR = X._scripts('recognition_repair_v2')
        self.RR, self.rp = RR, RR.Replay()
        self.inner = self.rp.baseline.inner
        if self.inner.ranker['checksum'] != ABLATED_CHECKSUM:
            raise ValueError('replay ranker is not 5c0be')
        parent = RepairSelectionV2(self.rp.baseline, [P.ProducerRoleStage(RR.ROOT, self.inner), self.rp.stage])
        self.selection = TextEvidenceSelectionV2(parent, components, tables)
        cached = RR.CachedCorrespondence(self.rp.stage.verifier, verify(read_json(X.ROOT / X.CLASS3)))
        self.geometry = X.GeometryCache(self.rp.stage, self.inner.registry, cached, geometry_reference)
        self.rp.stage.verifier.evaluate_target = self.geometry

    def outcome(self, out, gt):
        s = super().outcome(out, gt)
        text = out['text_evidence_repair_v1']
        s['text']['producer_aliases'] = text.get('producer_aliases')
        s['margin'] = out['raw_proposal']['score_margin']
        if gt:
            scores = {r['candidate_id']: r['score'] for r in out['raw_proposal']['ranked_candidates']}
            s['positive_scores'] = {cid: round(scores[cid], 6) for cid in s['positive_text_rows']}
        return s


def _delta(before, after, gt):
    """v1 delta plus support decreases and the best positive-candidate score change."""
    d = X._delta(before, after, gt)
    if d is None:
        return None
    down = 0
    for diff in d['changes'].values():
        down += sum(1 for n, (x, y) in diff.items() if y < x and '.contra' not in n and '.other_g' not in n
                    and 'fuzzy' not in n)
    b, a = before.get('positive_scores') or {}, after.get('positive_scores') or {}
    d['positive_support_down'] = down
    d['gt_score'] = {'before': max(b.values(), default=None), 'after': max(a.values(), default=None)}
    return d


class Producers:
    """Producer keys of a selected slug and of a GT product (all cards of that product), from the replay registry."""

    def __init__(self, registry, mapping):
        from rshb_vine.text_evidence_repair_v1 import producer_names as P
        self.key = {slug: P._producer_key(card, slug) for slug, card in registry.cards.items()}
        self.cards = {}
        for slug, product in mapping.items():
            self.cards.setdefault(product, set()).add(slug)

    def relation(self, change):
        gt = {self.key[s] for p in change['gt_products'] for s in self.cards.get(p, ()) if s in self.key}
        out = {}
        for side in ('before', 'after'):
            o = change[side]
            out[side] = {'same_producer_as_gt': self.key.get(o['selected']) in gt,
                         'wrong_sibling': not o['correct'] and self.key.get(o['selected']) in gt,
                         'gt_in_pool': o['gt_rank'] is not None}
        return out


def compare(base_rows, arm_rows, producers=None):
    """v1 comparison; every listed change and the semantic counts use the v2 delta; labelled changes get sibling tags."""
    result = X.compare(base_rows, arm_rows)
    base = {(r['key'], r['instance_id']): r for r in base_rows}
    arm = {(r['key'], r['instance_id']): r for r in arm_rows}
    for c in result['changes']:
        k = (c['key'], c['instance_id'])
        c['positive_text_delta'] = _delta(base[k]['outcome'], arm[k]['outcome'], set(c['gt_products']))
        c['margin'] = {'before': base[k]['outcome'].get('margin'), 'after': arm[k]['outcome'].get('margin')}
        if producers is not None and c['gt_products']:
            c['producer_relation'] = producers.relation(c)
    result['semantic']['targets_positive_support_down'] = sum(
        1 for k, r in arm.items()
        if (_delta(base[k]['outcome'], r['outcome'], set(r['gt_products'])) or {}).get('positive_support_down'))
    result['wrong_sibling_after'] = [[c['key'], c['instance_id'], c['kind']] for c in result['changes']
                                     if (c.get('producer_relation') or {}).get('after', {}).get('wrong_sibling')
                                     and c['kind'] in ('regression', 'changed_same_label')]
    return result
