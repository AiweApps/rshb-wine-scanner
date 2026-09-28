"""Sign-constrained listwise conditional-logit ranker over the systemic evidence schema.

Score s_i = sum_j w_j x_ij / scale_j; per query the multi-positive softmax loss
L_q = logsumexp(s_kept) - logsumexp(s_positive) is weighted by the group mass m_q = Q / (G * n_group) and
fitted with L2 penalty lambda/2 |w|^2 by L-BFGS-B. Every feature with sign +1 has w >= 0 and every feature
with sign -1 has w <= 0 (bounds, not a penalty), so for a fixed candidate adding support can only raise and
adding a contradiction can only lower its score, all else equal. Features enter linearly; the only
interactions are the fixed small-label x label-rank products inherited from v4 (free sign). The 8175 proposal
indicator's effective weight is bounded to [0, 1] logit as a fixed experiment assumption; its actual
contribution is measured on real candidates after the fit. Scales are the
training-candidate standard deviations (floor 0.05) so that lambda acts on comparable units; they change
neither signs nor the feasible set. No intercept (softmax is shift invariant), no control offset.
"""
from dataclasses import asdict, dataclass

import numpy as np

MODEL_VERSION = 'systemic-ranking-v2-signed-listwise-logit'
TIE_POLICY = 'score_desc_then_visual_support_desc_then_candidate_id'


@dataclass(frozen=True)
class Config:
    l2: float = 1.0
    scale_floor: float = 0.05
    maxiter: int = 2000
    tol: float = 1e-10
    effective_upper: tuple = (('pool.control_proposal', 1.0),)


def prepare(queries):
    """Keep loss-kept candidates; skip queries without a positive or a negative (listed, never guessed)."""
    kept, skipped = [], []
    for q in queries:
        keep = np.asarray(q['keep'], dtype=bool)
        x, p = np.asarray(q['x'], dtype=np.float64)[keep], np.asarray(q['positive'], dtype=bool)[keep]
        reason = 'empty_pool' if not len(x) else 'positive_absent' if not p.any() else 'no_negative' if p.all() else None
        if reason:
            skipped.append({'id': q['id'], 'group': q['group'], 'reason': reason})
        else:
            kept.append({'id': q['id'], 'group': q['group'], 'x': x, 'positive': p})
    groups = {}
    for q in kept:
        groups[q['group']] = groups.get(q['group'], 0) + 1
    mass = np.asarray([len(kept) / (len(groups) * groups[q['group']]) for q in kept])
    return kept, skipped, mass, len(groups)


def _objective(w, blocks, l2):
    from scipy.special import logsumexp
    loss, grad = .5 * l2 * float(w @ w), l2 * w.copy()
    for x, pos, m in blocks:
        s = x @ w
        a, b = logsumexp(s), logsumexp(s[pos])
        loss += m * (a - b)
        p = np.exp(s - a)
        r = np.where(pos, np.exp(s - b), 0.)
        grad += m * (x.T @ (p - r))
    return loss, grad


def fit(queries, config, feature_names, signs):
    from scipy.optimize import minimize
    import scipy
    kept, skipped, mass, groups = prepare(queries)
    x_all = np.concatenate([q['x'] for q in kept])
    scale = np.maximum(x_all.std(axis=0), config.scale_floor)
    blocks = [(q['x'] / scale, q['positive'], m) for q, m in zip(kept, mass)]
    upper = dict(config.effective_upper)
    bounds = [(0., upper[n] * s if n in upper else None) if signs[n] > 0 else (None, 0.) if signs[n] < 0 else (None, None)
              for n, s in zip(feature_names, scale)]
    w0 = np.zeros(len(feature_names))
    start_loss = _objective(w0, blocks, config.l2)[0]
    result = minimize(_objective, w0, args=(blocks, config.l2), jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': config.maxiter, 'ftol': config.tol, 'gtol': 1e-8})
    w = result.x
    violations = [n for n, v, sc in zip(feature_names, w, scale) if signs[n] > 0 and v < 0 or signs[n] < 0 and v > 0
                  or n in upper and v / sc > upper[n] + 1e-12]
    weights = {n: float(v) for n, v in zip(feature_names, w)}
    return {'version': MODEL_VERSION, 'features': list(feature_names), 'signs': {n: signs[n] for n in feature_names},
            'config': asdict(config), 'scale': [float(v) for v in scale], 'weights': [float(v) for v in w],
            'effective_weights': {n: weights[n] / s for n, s in zip(feature_names, scale)},
            'active_at_bound': sorted(n for n, v in weights.items() if signs[n] != 0 and v == 0.),
            'at_upper_bound': sorted(n for n, v, sc in zip(feature_names, w, scale) if n in upper and v / sc >= upper[n] - 1e-9),
            'training': {'queries': len(kept), 'groups': groups, 'skipped': skipped, 'candidates': int(len(x_all)),
                         'query_mass': 'Q/(groups*rows_in_group)', 'control_offset': 'none',
                         'loss_start': float(start_loss), 'loss_end': float(result.fun),
                         'optimizer': {'method': 'L-BFGS-B', 'success': bool(result.success),
                                       'message': str(result.message), 'iterations': int(result.nit)},
                         'numpy': np.__version__, 'scipy': scipy.__version__},
            'sign_violations': violations, 'tie_policy': TIE_POLICY, 'probability': None, 'calibration': 'absent'}


def score(model, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, len(model['features']))
    return x @ (np.asarray(model['weights']) / np.asarray(model['scale']))


def check_loadable(model):
    if model.get('version') != MODEL_VERSION or model.get('sign_violations'):
        raise ValueError('systemic-ranking-v2 model version/sign check failed; inference blocked')
    if not model['training']['optimizer']['success']:
        raise ValueError('systemic-ranking-v2 model did not converge; inference blocked')
