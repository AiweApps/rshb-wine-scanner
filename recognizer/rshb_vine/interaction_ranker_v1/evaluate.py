"""Paired old103 vs interaction-ranker answers from saved receipts: CV out-of-fold, internal117, dev108, open register.

Selection runs on the frozen snapshot-01 pool; correctness, changed answers, GT sets and qualified coverage are
judged in the source-verified snapshot-03 product identity that the live coherent output and the register GT use
(same card roster; cards only re-grouped). Raw card slugs and frozen product ids stay in the diagnostics.
Every answer is product level (vintage excluded), before and after the unchanged resolver. Counts are reported in rows, components and lineage groups. The register is exposed and overlaps
training by design; its rows are split into in-fit / internal / dev108 / out-of-roster and never called a test.
Multi-target register receipts are not scored here (target choice belongs to the primary-target stage).
"""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path

from rshb_vine.interaction_ranker_v1.source import check_trace, row_input, trace_result
from rshb_vine.io import read_json, verify


class Projection:
    """snapshot-03 product projection of frozen-pool cards, loaded and source-checked like the coherent adapter."""

    def __init__(self, root, frozen):
        from rshb_vine.combined_ranker_v1.identity import load_candidate
        self.handoff, self.candidate = load_candidate(root)
        self.frozen = frozen
        if set(frozen.cards) != set(self.candidate.cards):
            raise ValueError('snapshot-03 must keep the frozen card roster')

    def product(self, slug):
        return self.candidate.product_id(slug) if slug else None

    def gt_from_frozen_products(self, products):
        return set(self.candidate.rebind_ids(products, self.frozen))

    def gt_from_slugs(self, slugs):
        return {self.candidate.product_id(s) for s in slugs if s in self.candidate.cards}

    def describe(self):
        return {'identity_registry': self.candidate.checksum, 'identity_bundle': self.candidate.bundle_checksum,
                'identity_handoff': self.handoff['checksum'], 'selection_pool_registry': self.frozen.checksum}


def paired(projection, out, positives):
    """``positives``: snapshot-03 product ids of the GT."""
    slugs = {k: out[k]['representative_slug'] for k in ('legacy', 'proposal', 'raw_proposal', 'learned')}
    p = {k: projection.product(v) for k, v in slugs.items()}
    return {'old_correct': p['legacy'] in positives, 'new_correct': p['proposal'] in positives,
            'old_raw_correct': p['learned'] in positives, 'new_raw_correct': p['raw_proposal'] in positives,
            'changed': p['legacy'] != p['proposal'], 'resolver_blocked': out['proposal']['existing_evidence_resolution']['blocked'],
            'qualified': any(projection.product(s) in positives for c in out['base']['candidates'] for s in c['card_slugs']),
            'old_slug': slugs['legacy'], 'new_slug': slugs['proposal'], 'old_product': p['legacy'], 'new_product': p['proposal'],
            'old_frozen_product': projection.frozen.product_id(slugs['legacy']),
            'new_frozen_product': projection.frozen.product_id(slugs['proposal']),
            'seconds': out['seconds']}


def tally(items):
    """items: [{'row': id, 'group', 'lineage', 'slice', **paired}] -> counts with independent units."""
    def units(rows):
        return {'rows': len(rows), 'components': len({r['group'] for r in rows}),
                'lineage_groups': len({r['lineage'] for r in rows})}

    result = {}
    for stage, (a, b) in {'post_resolver': ('old_correct', 'new_correct'),
                          'pre_resolver': ('old_raw_correct', 'new_raw_correct')}.items():
        fixes = [r for r in items if not r[a] and r[b]]
        regs = [r for r in items if r[a] and not r[b]]
        slices = defaultdict(lambda: Counter())
        for r in fixes:
            slices[r['slice']]['fix'] += 1
        for r in regs:
            slices[r['slice']]['regression'] += 1
        result[stage] = {'n': len(items), 'old_top1': sum(r[a] for r in items), 'new_top1': sum(r[b] for r in items),
                         'fixes': units(fixes), 'regressions': units(regs),
                         'fix_ids': sorted(r['row'] for r in fixes), 'regression_ids': sorted(r['row'] for r in regs),
                         'slices': {k: dict(v) for k, v in slices.items()}}
    result['qualified_candidate_coverage'] = sum(r['qualified'] for r in items)
    result['changed_answers'] = sum(r['changed'] for r in items)
    result['resolver_blocked'] = sum(r['resolver_blocked'] for r in items)
    return result


def roster_items(root, selection, rows, meta, projection):
    """Paired answers for roster rows (fit OOF, internal117, dev108); parity with the saved old103 answer required."""
    items, parity = [], Counter()
    for row in rows:
        payload, provenance, report, kind = row_input(root, row)
        if check_trace(provenance, report, payload['observations']):
            parity['trace_problem'] += 1
        out = selection.evaluate(**deepcopy(payload), provenance=provenance)
        ok = out['legacy']['representative_slug'] == row['selected_slug']
        parity['old103_selected:' + ('ok' if ok else 'FAIL')] += 1
        m = meta(row)
        items.append({'row': row['query_id'], 'group': m['group'], 'lineage': m['lineage'], 'slice': m['slice'],
                      'parity': ok, **paired(projection, out, projection.gt_from_frozen_products(row['ground_truth']['current_products']))})
    return items, dict(parity)


def _receipt(root, register_row):
    current = register_row.get('current') or {}
    first = current.get('first_pass') or ''
    path = first.split(':', 1)[1] if ':' in first else current.get('receipt')
    if not path:
        return None, 'no_receipt'
    full = Path(root) / path
    document = json.load(gzip.open(full)) if path.endswith('.gz') else read_json(full)
    if 'checksum' in document:
        verify(document)
    return document.get('result', document), None


def register_items(root, selection, rosters, *, with_labels, projection=None):
    """Single-target register receipts; ``with_labels=False`` computes evaluability and replay parity only."""
    register = verify(read_json(Path(root) / 'runs/all-open-errors-v1/register.json'))['rows']
    items, skipped = [], Counter()
    for r in register:
        result, reason = _receipt(root, r)
        targets = ((result or {}).get('product_identity_evidence') or {}).get('targets', [])
        executed = ((result or {}).get('learned_selector_execution') or {}).get('targets', [])
        if reason or len(targets) != 1 or len(executed) != 1:
            skipped[reason or ('multi_target' if len(targets) > 1 else 'no_selection_target')] += 1
            continue
        t = targets[0]
        payload = {'control_slug': t['control_slug'], 'raw_visual': t['raw_visual'],
                   'observations': t['ocr_observations'], 'control_candidates': t['control_candidates']}
        provenance, report = trace_result(result, t['instance_id'], payload['observations'])
        if check_trace(provenance, report, payload['observations']):
            skipped['provenance_trace'] += 1
            continue
        out = selection.evaluate(**deepcopy(payload), provenance=provenance)
        if out['legacy']['representative_slug'] != executed[0]['proposal']['representative_slug']:
            skipped['old103_replay_parity'] += 1
            continue
        membership = next((k for k, shas in rosters.items() if r['image_sha256'] in shas), 'out_of_roster')
        item = {'row': r['register_key'] + ':' + r['query_id'], 'group': r['lineage_group'], 'lineage': r['lineage_group'],
                'membership': membership, 'slice': r['data_type'], 'image_sha256': r['image_sha256']}
        if with_labels:
            gt = r['ground_truth']
            if r['status'] == 'GT_hold' or gt.get('status') != 'unambiguous':
                skipped['gt_hold_or_ambiguous'] += 1
                continue
            item.update(paired(projection, out, projection.gt_from_slugs(gt['acceptable_slugs'])),
                        status_at_freeze=r['status'], gt_slugs=sorted(gt['acceptable_slugs']))
        items.append(item)
    return items, dict(skipped)
