"""One deterministic listwise Product-ID ranker over admitted real runtime rows.

This module does not choose data, roles, optimizer settings, threshold or release.
The caller must supply an immutable admission protocol and source-bound rows.
"""
from collections import Counter
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from rshb_vine.candidate_discriminators import FEATURE_NAMES as RUNTIME_103


EXTRA_NAMES = (
    'quality.label_min_side_log_x_B0_label_similarity',
    'quality.label_min_side_log_x_B3_label_similarity',
)
FEATURE_NAMES = (*RUNTIME_103, *EXTRA_NAMES)
SCHEMA_VERSION = 'ranker-real-runtime-features-105-proposal-v1'


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Non-finite or non-numeric ' + name)
    return float(value)


def candidate_vector(query, candidate):
    """Numeric inference features only; no positive Product ID or source GT read."""
    sparse = candidate['nonzero_features']
    if set(sparse) - set(RUNTIME_103):
        raise ValueError('Candidate has a feature outside frozen runtime103')
    values = {name: _finite(sparse.get(name, 0.0), name) for name in RUNTIME_103}
    # A detector target box may be a bottle fallback. Only actual label-channel
    # view geometry licenses a label-size interaction, with each arm checked.
    def label_size(arm):
        channel = query.get('raw_channels', {}).get(arm + '_label', [])
        views = query.get('views', [])
        index = channel[0].get('query_view_index') if channel else None
        if (type(index) is not int or not isinstance(views, list) or not 0 <= index < len(views)
                or views[index].get('kind') not in ('detected_label', 'front_label')):
            return None
        box = views[index].get('bbox')
        if (not isinstance(box, list) or len(box) != 4 or any(
                isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            return None
        return min(box[2] - box[0], box[3] - box[1])
    b0_size, b3_size = label_size('B0'), label_size('B3')
    b0_size_log = 0.0 if b0_size is None else math.log1p(b0_size) / math.log1p(128)
    b3_size_log = 0.0 if b3_size is None else math.log1p(b3_size) / math.log1p(128)
    b0 = values['visual.B0_label.raw_similarity']
    b3 = values['visual.B3_label.raw_similarity']
    extra = (
        b0_size_log * b0, b3_size_log * b3,
    )
    return np.asarray([values[name] for name in RUNTIME_103] + list(extra), dtype=np.float64)


def prepare_query(row, *, require_fit=False):
    if require_fit and (not row.get('fit_admitted') or row.get('ranker_split') != 'train'):
        raise ValueError('Training query lacks per-target admission')
    pool = row['candidate_pool']
    ids = [c['product_id'] for c in pool]
    candidate_ids = [c['candidate_id'] for c in pool]
    if len(set(ids)) != len(ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Candidate pool has duplicate Product or candidate IDs')
    positives = set(row['ground_truth']['current_products'])
    if not positives or not pool:
        raise ValueError('Training query needs a positive label and actual candidates')
    mask = np.asarray([pid in positives for pid in ids], dtype=bool)
    if not mask.any():
        raise ValueError('Retrieval miss cannot be inserted into ranker gradient')
    ignored_values = row.get('loss_ignore_candidate_ids', [])
    if not isinstance(ignored_values, list) or any(not isinstance(cid, str) for cid in ignored_values):
        raise ValueError('Loss-ignore candidates must be a list of actual candidate IDs')
    ignore = set(ignored_values)
    if len(ignore) != len(ignored_values):
        raise ValueError('Duplicate loss-ignore candidate ID')
    if not ignore <= set(candidate_ids) or any(c['candidate_id'] in ignore and positive
                                              for c,positive in zip(pool,mask)):
        raise ValueError('Loss-ignore set is outside actual negative candidates')
    keep = np.asarray([cid not in ignore for cid in candidate_ids], dtype=bool)
    if sum(keep) < 2 or not mask[keep].any() or mask[keep].all():
        raise ValueError('Loss-ignore set leaves no positive/negative ranking comparison')
    matrix = np.stack([candidate_vector(row, c) for c,kept in zip(pool,keep) if kept])
    if matrix.shape != (sum(keep), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid fixed feature matrix')
    return {'id':row['query_id'],'group':row['capture_group'],
            'producer_group':row.get('producer_group'),
            'matrix':matrix,'positive_mask':mask[keep],
            'candidate_ids':[cid for cid,kept in zip(candidate_ids,keep) if kept],
            'loss_ignored_candidate_ids':sorted(ignore),'actual_pool_candidates':len(pool)}


def weight_preview(queries):
    counts = Counter(q['group'] for q in queries)
    if None in counts or '' in counts or any(not q.get('producer_group') for q in queries):
        raise ValueError('Every training query needs capture/family and producer groups')
    raw = np.asarray([1.0 / counts[q['group']] for q in queries], dtype=np.float64)
    raw /= raw.sum()
    group_weights=Counter()
    producer_weights=Counter()
    for query,weight in zip(queries,raw):
        group_weights[query['group']]+=float(weight)
        producer_weights[query['producer_group']]+=float(weight)
    return raw,group_weights,producer_weights


def _weights(queries, max_query_weight, max_group_weight, max_producer_weight):
    raw,group_weights,producer_weights=weight_preview(queries)
    if raw.max() > max_query_weight:
        raise ValueError('One query dominates the admitted fit budget')
    if max(group_weights.values()) > max_group_weight+1e-12:
        raise ValueError('One capture/family group dominates the fit budget')
    if max(producer_weights.values()) > max_producer_weight+1e-12:
        raise ValueError('One producer dominates the fit budget')
    return raw


def _normalization(queries, weights, scale_floor):
    mean = sum(weight * q['matrix'].mean(axis=0) for weight,q in zip(weights,queries))
    variance = sum(weight * np.mean((q['matrix'] - mean) ** 2, axis=0)
                   for weight,q in zip(weights,queries))
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), scale_floor)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Non-finite train-only normalization')
    return mean, scale


def fit_once(queries, *, l2, maxiter, scale_floor, max_query_weight,
             max_group_weight, max_producer_weight):
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Empty or duplicate training ranking tasks')
    l2 = _finite(l2, 'l2')
    if not 0 < l2 <= 10 or type(maxiter) is not int or not 1 <= maxiter <= 1000:
        raise ValueError('Fit configuration outside fixed bounds')
    scale_floor = _finite(scale_floor, 'scale_floor')
    max_query_weight = _finite(max_query_weight, 'max_query_weight')
    max_group_weight = _finite(max_group_weight, 'max_group_weight')
    max_producer_weight = _finite(max_producer_weight, 'max_producer_weight')
    if not all(0 < value <= 1 for value in (scale_floor,max_query_weight,max_group_weight,max_producer_weight)):
        raise ValueError('Invalid normalization or group cap')
    weights = _weights(queries, max_query_weight, max_group_weight, max_producer_weight)
    mean, scale = _normalization(queries, weights, scale_floor)
    matrices = [(q['matrix'] - mean) / scale for q in queries]
    masks = [q['positive_mask'] for q in queries]

    def loss_grad(coef):
        loss = l2 * float(coef @ coef) / 2
        grad = l2 * coef.copy()
        for weight, matrix, mask in zip(weights, matrices, masks):
            scores = matrix @ coef
            loss += weight * (logsumexp(scores) - logsumexp(scores[mask]))
            grad += weight * (softmax(scores) @ matrix - softmax(scores[mask]) @ matrix[mask])
        return loss, grad

    result = minimize(loss_grad, np.zeros(len(FEATURE_NAMES), dtype=np.float64),
                      jac=True, method='L-BFGS-B', options={'maxiter':maxiter,'ftol':1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('One admitted optimizer attempt did not converge: ' + result.message)
    return {'kind':'real-runtime-listwise-product-ranker-v1','feature_schema':SCHEMA_VERSION,
        'feature_names':list(FEATURE_NAMES),'mean':mean.tolist(),'scale':scale.tolist(),
        'coefficients':result.x.tolist(),'l2':l2,'optimizer':'L-BFGS-B',
        'iterations':int(result.nit),'train_queries':len(queries),
        'train_groups':len({q['group'] for q in queries}),
        'objective':float(result.fun),'calibrated':False,'release_admitted':False,
        'tie_policy':'descending_score_then_candidate_id',
        'representative_policy':'existing_actual_control_then_best_visual_then_supplied_card'}


def rank(row, model):
    if model['feature_names'] != list(FEATURE_NAMES) or model.get('feature_schema') != SCHEMA_VERSION:
        raise ValueError('Model/feature schema mismatch')
    pool = row['candidate_pool']
    if not pool:
        return []
    matrix = np.stack([candidate_vector(row,c) for c in pool])
    mean = np.asarray(model['mean'], dtype=np.float64)
    scale = np.asarray(model['scale'], dtype=np.float64)
    coef = np.asarray(model['coefficients'], dtype=np.float64)
    if (mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape or coef.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(coef).all() or not (scale > 0).all()):
        raise ValueError('Invalid model normalization or coefficients')
    scores = (matrix - mean) / scale @ coef
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite ranking score')
    return sorted(({'product_id':c['product_id'],'candidate_id':c['candidate_id'],
                    'score':float(score),'representative_slug':c['representative_slug']}
                   for c,score in zip(pool,scores)),key=lambda r:(-r['score'],r['candidate_id']))
