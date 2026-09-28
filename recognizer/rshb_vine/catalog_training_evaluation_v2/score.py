"""Image-level ProductID03 scorer for web232 validation46 and paired A/B comparison.

GT is only validation.json acceptable_slugs mapped through registry03. Validation has no bottle/ROI
annotation, so per-target GT-conditioned metrics are reported unavailable, never zero.
"""
from collections import Counter

from rshb_vine.catalog_training_evaluation_v2 import pins

TIMING_KEYS = {'timing_ms', 'seconds', 'elapsed_ms', 'latency_ms', 'ms', 'control_seconds', 'total_seconds',
               'encode_seconds', 'key_seconds'}
# digest(result) of a control result that itself carries timing_ms (recognition_runtime/selected_product_output).
TIMING_DIGESTS = {'/product_identity_evidence/source_control_digest', '/selected_product_output/source_result_digest'}


def validation_targets():
    doc = pins.read('validation')
    if doc['role'] != 'validation' or doc['n'] != 46 or len(doc['records']) != 46:
        raise ValueError('Validation46 manifest scope changed')
    mapping = pins.product03()
    rows = []
    for r in doc['records']:
        if r['split'] != 'validation' or r['fit_admitted'] is not False:
            raise ValueError('Closed or fit-admitted record in validation scope: ' + r['id'])
        unmapped = [s for s in r['acceptable_slugs'] if s not in mapping]
        rows.append({'id': r['id'], 'sha256': r['sha256'], 'image_kind': r['image_kind'],
                     'acceptable_slugs': r['acceptable_slugs'], 'unmapped_gt_slugs': unmapped,
                     'acceptable_products03': sorted({mapping[s] for s in r['acceptable_slugs'] if s in mapping}),
                     'gt_geometry': None, 'exact_variant_confirmed': r['exact_variant_confirmed']})
    return rows


def _b3_pool(result):
    """B3 arm retrieval rows if the published response exposes selector raw arms; else None."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            arms = node.get('arms')
            if isinstance(arms, dict) and isinstance(arms.get('B3'), dict) and 'retrieval' in arms['B3']:
                found.append(arms['B3']['retrieval'])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(result)
    return found or None


def score_record(target, result, mapping, request_sha256):
    if request_sha256 != target['sha256']:
        raise ValueError('Request bytes differ from the validation record: ' + target['id'])
    card = result.get('best_candidate')
    confirmed = result.get('slug')
    product = mapping.get(card) if card else None
    gt = set(target['acceptable_products03'])
    if card is None:
        status = 'null_best_candidate'
    elif product is None:
        status = 'unmapped_prediction'
    else:
        status = 'correct' if product in gt else 'wrong'
    targets = result.get('targets') or []
    any_hit = any(mapping.get((t.get('retrieval') or {}).get('best_candidate')) in gt for t in targets)
    return {'id': target['id'], 'request_sha256': request_sha256, 'decision': result.get('decision'), 'best_candidate': card,
            'best_product03': product, 'status': status, 'correct': status == 'correct',
            'confirmed_slug': confirmed, 'confirmed_correct': (mapping.get(confirmed) in gt) if confirmed else None,
            'n_targets': len(targets), 'any_target_gt_hit_diagnostic': any_hit,
            'b3_pool_exposed': _b3_pool(result) is not None}


def summarize(rows, denominator=46):
    status = Counter(r['status'] for r in rows)
    decisions = Counter(r['decision'] for r in rows)
    return {'photos_denominator': denominator, 'scored_photos': len(rows),
            'missing_photos': denominator - len(rows),
            'best_candidate_product03_correct': status['correct'], 'status_counts': dict(status),
            'decision_counts': dict(decisions),
            'ambiguous_target_decisions': decisions.get('ambiguous_target', 0),
            'confirmed_slug_returned': sum(r['confirmed_slug'] is not None for r in rows),
            'confirmed_slug_correct': sum(r['confirmed_correct'] is True for r in rows),
            'multi_target_photos': sum(r['n_targets'] > 1 for r in rows),
            'any_target_gt_hit_diagnostic_not_primary': sum(r['any_target_gt_hit_diagnostic'] for r in rows),
            'per_target_gt_conditioned': 'unavailable: validation46 has no bottle bbox/ROI annotation',
            'b3_pool_metrics': ('evidence_present_metrics_not_computed' if rows and all(r['b3_pool_exposed'] for r in rows)
                                else 'unavailable: published response does not expose B3 arm retrieval'),
            'exact_card_claim': False, 'independent_test': False}


def complete(rows):
    """Exactly the 46 validation IDs, once each, bound to their manifest image SHA."""
    targets = {t['id']: t['sha256'] for t in validation_targets()}
    ids = [r['id'] for r in rows]
    if len(ids) != len(targets) or set(ids) != set(targets) or any(r['request_sha256'] != targets[r['id']] for r in rows):
        raise ValueError('Arm report is not the complete exact validation46 set')
    return {r['id']: r for r in rows}


def paired(a_rows, b_rows):
    a, b = complete(a_rows), complete(b_rows)
    fixed = sorted(i for i in a if not a[i]['correct'] and b[i]['correct'])
    broken = sorted(i for i in a if a[i]['correct'] and not b[i]['correct'])
    new_wrong = sorted(i for i in a if a[i]['status'] != 'wrong' and b[i]['status'] == 'wrong')
    return {'fixed': fixed, 'regressed': broken, 'new_wrong_product': new_wrong,
            'net_correct': len(fixed) - len(broken)}


def select(a_rows, arms):
    """Frozen rule over selectable steps: most correct, then fewest new wrong, then earliest; must beat A."""
    steps = pins.checkpoint_steps()['selectable']
    base = sum(r['correct'] for r in complete(a_rows).values())
    if sorted(arms) != sorted(steps):
        raise ValueError('Frozen selection needs all selectable checkpoints scored: ' + str(steps))
    ranked = []
    for step in steps:
        rows = arms[step]
        pair = paired(a_rows, rows)
        ranked.append((-sum(r['correct'] for r in rows), len(pair['new_wrong_product']), step, pair))
    best = min(ranked, key=lambda x: x[:3])
    gain = -best[0] - base
    return {'selected': best[2] if gain > 0 else None, 'correct': -best[0], 'A_correct': base, 'gain': gain,
            'pair': best[3], 'reason': 'beats A' if gain > 0 else 'no Product gain over A; keep parent'}


def diff(left, right, path=''):
    """All differing leaf paths; timing keys are listed separately rather than dropped."""
    out = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right), key=str):
            sub = path + '/' + str(key)
            if key not in left or key not in right:
                out.append((sub, 'missing_on_' + ('left' if key not in left else 'right')))
            else:
                out.extend(diff(left[key], right[key], sub))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append((path, 'length %d vs %d' % (len(left), len(right))))
        for i, (a, b) in enumerate(zip(left, right)):
            out.extend(diff(a, b, path + '/' + str(i)))
    elif left != right:
        out.append((path, 'value'))
    return out


def classify(paths):
    timing = [p for p in paths if p[1] == 'value' and any(part in TIMING_KEYS for part in p[0].split('/'))]
    digests = [p for p in paths if p[1] == 'value' and p[0] in TIMING_DIGESTS]
    return {'timing_only': [p[0] for p in timing], 'timing_derived_digest': [p[0] for p in digests],
            'behavior': [list(p) for p in paths if p not in timing and p not in digests]}
