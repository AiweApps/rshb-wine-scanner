"""One linear listwise scorer over the v4 feature schema; no control prior, no resolver overlay.

Fit: multi-positive softmax loss, equal mass per group then per query, L2 towards zero,
sign bounds from ``features.MONOTONE`` and one deterministic L-BFGS-B call. Loss-ignored
candidates are removed from the fit pool only. Prediction never reads labels.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from rshb_vine.io import digest, seal, sha256, verify
from rshb_vine.system_selection_v4.features import CHANNELS, FEATURE_NAMES, MONOTONE, SCHEMA_VERSION

MODEL_VERSION = 'system-selection-v4-linear-listwise-v1'
PROPOSAL_VERSION = 'system-selection-v4-proposal-v1'
CODE_PATHS = ('rshb_vine/system_selection_v4/scorer.py', 'rshb_vine/system_selection_v4/features.py',
              'rshb_vine/system_selection_v4/text.py', 'rshb_vine/system_selection_v4/context.py')
TIE_POLICY = 'score_desc_then_visual_support_desc_then_candidate_id'


@dataclass(frozen=True)
class FitConfig:
    regularization: float = 1.0
    scale_floor: float = 0.05
    maxiter: int = 500
    maxfun: int = 5000
    maxcor: int = 10
    maxls: int = 20
    tol: float = 1e-9


def schema():
    return {'schema_version': SCHEMA_VERSION, 'feature_names': list(FEATURE_NAMES), 'monotone': MONOTONE}


def code_checksums(root):
    return {path: sha256(root / path) for path in CODE_PATHS}


def matrix(feature_output):
    if feature_output.get('schema_version') != SCHEMA_VERSION or feature_output.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('v4 feature schema mismatch')
    rows = [[float(c['features'][n]) for n in FEATURE_NAMES] for c in feature_output['candidates']]
    result = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURE_NAMES))
    if not np.isfinite(result).all():
        raise ValueError('Non-finite v4 feature')
    return result


def _bounds():
    return [(0.0, None) if MONOTONE.get(n) == 1 else (None, 0.0) if MONOTONE.get(n) == -1 else (None, None)
            for n in FEATURE_NAMES]


def _projected(weights, gradient):
    values = [0.0 if (lo is not None and w <= lo + 1e-12 and g > 0) or (hi is not None and w >= hi - 1e-12 and g < 0)
              else g for (lo, hi), w, g in zip(_bounds(), weights, gradient)]
    return float(np.max(np.abs(values)))


def listwise_objective(w, normalized, mass, regularization):
    from scipy.special import logsumexp
    loss = 0.5 * regularization * float(w @ w)
    grad = regularization * w
    for (x, p), m in zip(normalized, mass):
        s = x @ w
        a, b = logsumexp(s), logsumexp(s[p])
        loss += m * (a - b)
        grad = grad + m * (np.exp(s - a) @ x - np.exp(s[p] - b) @ x[p])
    return float(loss), grad


def prepare_fit(queries, config):
    """Masked, group-weighted, fit-normalized pools; skipped queries stay listed."""
    prepared, skipped = [], []
    for q in sorted(queries, key=lambda q: q['id']):
        keep = np.asarray(q['keep'], dtype=bool)
        x, positive = q['matrix'][keep], np.asarray(q['positive'], dtype=bool)[keep]
        reason = ('empty_pool' if not len(x) else 'positive_absent' if not positive.any()
                  else 'no_negative' if positive.all() else None)
        if reason:
            skipped.append({'id': q['id'], 'group': q['group'], 'reason': reason})
        else:
            prepared.append((q['id'], q['group'], x, positive))
    if not prepared:
        raise ValueError('No fit query has both positive and negative candidates')
    groups = {}
    for _, group, _, _ in prepared:
        groups[group] = groups.get(group, 0) + 1
    mass = np.asarray([1.0 / (len(groups) * groups[g]) for _, g, _, _ in prepared])
    mean = sum(m * x.mean(axis=0) for m, (_, _, x, _) in zip(mass, prepared))
    variance = sum(m * np.mean((x - mean) ** 2, axis=0) for m, (_, _, x, _) in zip(mass, prepared))
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), config.scale_floor)
    normalized = [((x - mean) / scale, p) for _, _, x, p in prepared]
    return prepared, skipped, groups, mass, mean, scale, normalized


def fit(queries, *, config, root):
    """``queries``: [{'id','group','matrix','positive','keep'}]; returns the unsealed model body."""
    from scipy.optimize import minimize
    import scipy
    prepared, skipped, groups, mass, mean, scale, normalized = prepare_fit(queries, config)

    def objective(w):
        return listwise_objective(w, normalized, mass, config.regularization)

    start = np.zeros(len(FEATURE_NAMES))
    options = {'maxiter': config.maxiter, 'maxfun': config.maxfun, 'maxcor': config.maxcor,
               'maxls': config.maxls, 'ftol': config.tol, 'gtol': config.tol}
    initial, _ = objective(start)
    result = minimize(objective, start, method='L-BFGS-B', jac=True, bounds=_bounds(), options=options)
    weights = np.asarray(result.x, dtype=np.float64)
    final, gradient = objective(weights)
    if not np.isfinite(weights).all():
        raise ValueError('Non-finite v4 weights')
    normalization = {'mean': mean.tolist(), 'scale': scale.tolist(), 'scale_floor': config.scale_floor}
    return {'version': MODEL_VERSION, 'feature_schema': schema(), 'feature_schema_checksum': digest(schema()),
            'code_checksums': code_checksums(root), 'config': asdict(config), 'normalization': normalization,
            'weights': weights.tolist(), 'weights_checksum': digest(weights.tolist()),
            'training': {'used_queries': len(prepared), 'groups': len(groups), 'skipped': skipped,
                         'query_weighting': '1/(groups*queries_in_group)', 'initialization': 'zeros',
                         'control_prior': None, 'randomness': 'none'},
            'optimizer': {'method': 'L-BFGS-B', 'bounds': 'features.MONOTONE signs', 'success': bool(result.success),
                          'status': int(result.status), 'message': str(result.message), 'iterations': int(result.nit),
                          'function_evaluations': int(result.nfev), 'initial_objective': initial,
                          'final_objective': final, 'projected_gradient_max_abs': _projected(weights, gradient),
                          'scipy': scipy.__version__, 'numpy': np.__version__},
            'tie_policy': TIE_POLICY, 'probability': None, 'calibration': 'absent', 'release_admitted': False}


def check_model(model, root):
    verify(model)
    if model.get('version') != MODEL_VERSION or model.get('feature_schema_checksum') != digest(schema()):
        raise ValueError('v4 model/schema mismatch')
    if model.get('code_checksums') != code_checksums(root):
        raise ValueError('v4 feature/scorer code changed since fit')
    if digest(model['weights']) != model['weights_checksum']:
        raise ValueError('v4 weights checksum mismatch')


def _representative(candidate):
    provenance = candidate['provenance']
    control = provenance.get('control_proposal_slug')
    if control is not None:
        return control, 'actual8175_proposal_member'
    visual = sorted(((e['rank'], CHANNELS.index(e['channel']), e['raw']['slug']) for e in provenance.get('visual', [])
                     if e['raw']['slug'] in candidate['card_slugs']))
    if visual:
        return visual[0][2], 'best_raw_visual_member'
    for row in provenance.get('control_candidates', []):
        if row['slug'] in candidate['card_slugs']:
            return row['slug'], 'actual8175_supplied_member'
    for row in candidate.get('extra_sources', []):
        return row['slug'], 'extra_source_member:' + str(row.get('source'))
    raise ValueError('Candidate has no actual member evidence')


def scores(feature_output, model):
    mean, scale = (np.asarray(model['normalization'][k]) for k in ('mean', 'scale'))
    return ((matrix(feature_output) - mean) / scale) @ np.asarray(model['weights'])


def rank(feature_output, score_values):
    candidates = feature_output['candidates']
    order = sorted(range(len(candidates)), key=lambda i: (
        -float(score_values[i]), -candidates[i]['features']['vis.support'], candidates[i]['candidate_id']))
    ranked = []
    for position, i in enumerate(order, 1):
        c = candidates[i]
        representative, source = _representative(c)
        ranked.append({'rank': position, 'input_pool_index': i, 'candidate_id': c['candidate_id'],
                       'product_id': c['product_id'], 'identity_status': c['identity_status'],
                       'card_slugs': list(c['card_slugs']), 'representative_slug': representative,
                       'representative_source': source, 'score': float(score_values[i]),
                       'is_control_proposal': c['provenance'].get('control_proposal_slug') is not None,
                       'visual_support': c['features']['vis.support'],
                       'publish_requires_extension': c.get('publish_requires_extension', False)})
    return ranked


def propose(feature_output, model, *, control_slug):
    values = scores(feature_output, model)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite v4 score')
    ranked = rank(feature_output, values)
    winner = ranked[0] if ranked else None
    margin = ranked[0]['score'] - ranked[1]['score'] if len(ranked) > 1 else None
    return {'schema_version': PROPOSAL_VERSION, 'model_checksum': model['checksum'],
            'feature_schema_checksum': model['feature_schema_checksum'], 'ranked_candidates': ranked,
            'candidate_id': winner and winner['candidate_id'], 'product_id': winner and winner['product_id'],
            'representative_slug': winner and winner['representative_slug'],
            'best_candidate': winner and winner['representative_slug'],
            'representative_source': winner and winner['representative_source'],
            'score': winner and winner['score'], 'score_margin': margin if margin is None or math.isfinite(margin) else None,
            'control_slug': control_slug,
            'control_agrees': bool(winner and control_slug in winner['card_slugs']),
            'exact_slug': None, 'vintage': {'value': None, 'status': 'resolved_after_selection'},
            'probability': None, 'confidence': {'status': 'uncalibrated_proposal', 'probability': None},
            'reason': 'v4_linear_listwise' if winner else 'empty_candidate_pool',
            'tie_policy': TIE_POLICY, 'resolver': 'none_symmetric_features', 'release_admitted': False}
