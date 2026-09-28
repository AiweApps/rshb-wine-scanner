"""CPU linear listwise selection over the frozen85 candidate feature contract.

This module owns neither data admission nor release. ``fit`` accepts explicitly
admitted feature tables and a frozen config; ``predict`` has no truth argument
and accesses only candidate features and their actual member provenance.
Scores are uncalibrated. A representative card never proves an exact vintage.
"""
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np

from rshb_vine.io import digest, seal, sha256, verify
from rshb_vine.learned_selection_features import FEATURE_NAMES, SCHEMA_VERSION
from rshb_vine.product_first.visual_selector import CHANNELS


MODEL_VERSION = "learned-selection-linear-listwise-v1"
PREDICTION_VERSION = "learned-selection-linear-proposal-v1"
FEATURE_SCHEMA = {"schema_version": SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES)}
FEATURE_SCHEMA_CHECKSUM = digest(FEATURE_SCHEMA)
CONTROL_INDEX = FEATURE_NAMES.index("control.is_proposal")
CODE_PATHS = (
    "learned_selection_model.py", "learned_selection_features.py",
    "product_first/visual_selector.py", "product_first/partial_evidence.py",
    "typed_catalog_lexicon.py", "resolution/identity.py", "gallery_variant_text.py",
)


@dataclass(frozen=True)
class FitConfig:
    """Every numeric parameter is explicit; use None to disable the prior.

    ``control_prior_coefficient`` is the L2 centre in RAW control-indicator
    score units. Its standardized weight is coefficient * fit scale. L2 is
    one-half regularization times squared distance from that prior vector.
    ``tol`` supplies both L-BFGS-B ftol and gtol. No random seed is used because
    initialization and optimization are deterministic and contain no sampling.
    """

    regularization: float
    maxiter: int
    maxfun: int
    maxls: int
    maxcor: int
    tol: float
    scale_floor: float
    control_prior_coefficient: float | None

    def __post_init__(self):
        for name in ("maxiter", "maxfun", "maxls", "maxcor"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be an explicit positive integer")
        for name in ("regularization", "tol", "scale_floor"):
            value = _number(getattr(self, name), name)
            if value < 0 or (name != "regularization" and value == 0):
                raise ValueError("Invalid " + name)
        if self.control_prior_coefficient is not None:
            _number(self.control_prior_coefficient, "control_prior_coefficient")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _code_checksums():
    root = Path(__file__).resolve().parent
    return {"rshb_vine/" + path: sha256(root / path) for path in CODE_PATHS}


def _matrix(feature_output):
    """Project only the known inference contract, never labels or truth fields."""
    if feature_output.get("schema_version") != SCHEMA_VERSION or feature_output.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Frozen feature schema mismatch")
    candidates = feature_output.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Feature output must contain its ordered candidate list")
    seen = set()
    control_count = 0
    rows = []
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Candidate IDs must be nonempty and unique within a target")
        seen.add(cid)
        members = candidate.get("card_slugs")
        if not isinstance(members, list) or not members or any(not isinstance(slug, str) or not slug for slug in members) or len(set(members)) != len(members):
            raise ValueError("Candidate must preserve its distinct actual card members")
        cards = candidate.get("provenance", {}).get("cards", {})
        if set(cards) != set(members):
            raise ValueError("Candidate card registry provenance is incomplete")
        if candidate.get("identity_status") == "source_admitted_product":
            if cid != candidate.get("product_id") or any(
                cards[slug].get("binding_status") != "source_admitted_product"
                or cards[slug].get("product_id") != candidate["product_id"] for slug in members
            ):
                raise ValueError("Unadmitted cross-card grouping")
        elif (len(members) != 1 or cid != "unresolved-card:" + members[0]
              or cards[members[0]].get("binding_status") != candidate.get("identity_status")):
            raise ValueError("Unresolved identity must remain card-local")
        if any(cards[slug].get("slug") != slug or cards[slug].get("product_id") != candidate.get("product_id") for slug in members):
            raise ValueError("Candidate identity changed relative to its registry provenance")
        features = candidate.get("features", {})
        if set(features) != set(FEATURE_NAMES):
            raise ValueError("Candidate feature names differ from frozen85")
        values = [_number(features[name], name) for name in FEATURE_NAMES]
        control = candidate.get("provenance", {}).get("control_proposal_slug")
        if control is not None and control not in members:
            raise ValueError("Control proposal is not an actual member")
        if values[CONTROL_INDEX] != float(control is not None):
            raise ValueError("Control indicator differs from supplied8175 provenance")
        control_count += control is not None
        rows.append(values)
    if control_count > 1:
        raise ValueError("One physical target cannot have multiple8175 control proposals")
    matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURE_NAMES))
    return candidates, matrix


def _weighted_normalization(arrays, query_weights, scale_floor):
    # Equal mass per group, then per query, then per candidate: a larger
    # retrieval pool does not dominate normalization or the listwise objective.
    mean = sum(weight * array.mean(axis=0) for array, weight in zip(arrays, query_weights))
    variance = sum(weight * np.mean((array - mean) ** 2, axis=0) for array, weight in zip(arrays, query_weights))
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), scale_floor)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Non-finite fit normalization")
    return mean, scale


def fit(queries, *, config):
    """Fit ONCE and return a sealed JSON-compatible model; write no files.

    Each query is ``{id: str, group: str, feature_output: frozen85_output,
    positive_mask: list[bool|0|1]}``, with the mask aligned to candidate order.
    Multiple positives represent admitted acceptable product/card hypotheses;
    their softmax masses are summed, not forced into one arbitrary card.

    Empty pools, absent positives and all-positive pools carry no ranking
    supervision and are reported as skipped. They never receive synthetic
    candidates or labels. Admission and held-out comparisons belong to caller.
    """
    if not isinstance(config, FitConfig):
        raise TypeError("config must be an explicitly constructed frozen FitConfig")
    if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)) or not queries:
        raise ValueError("fit requires a nonempty admitted query sequence")
    prepared, skipped, input_records = [], [], []
    ids = set()
    for query in queries:
        qid, group = query.get("id"), query.get("group")
        if not isinstance(qid, str) or not qid or qid in ids or not isinstance(group, str) or not group:
            raise ValueError("Every fit query needs a unique id and an explicit group")
        ids.add(qid)
        candidates, matrix = _matrix(query["feature_output"])
        mask = query.get("positive_mask")
        if not isinstance(mask, (list, tuple, np.ndarray)) or len(mask) != len(candidates) or any(
            not isinstance(value, (bool, int, np.bool_, np.integer)) or value not in (0, 1) for value in mask
        ):
            raise ValueError("positive_mask must be aligned explicit boolean product truth")
        positive = np.asarray(mask, dtype=bool)
        input_records.append({"id": qid, "group": group, "candidate_ids": [row["candidate_id"] for row in candidates],
                              "features": matrix.tolist(), "positive_mask": positive.tolist()})
        reason = "empty_candidate_pool" if not candidates else "positive_absent_from_pool" if not positive.any() else "no_negative_candidate" if positive.all() else None
        if reason:
            skipped.append({"id": qid, "group": group, "reason": reason, "candidates": len(candidates), "positives": int(positive.sum())})
        else:
            prepared.append((qid, group, matrix, positive))
    prepared.sort(key=lambda row: row[0])
    input_records.sort(key=lambda row: row["id"])
    skipped.sort(key=lambda row: row["id"])
    if not prepared:
        raise ValueError("No fit query contains both a positive and a negative candidate")
    group_sizes = Counter(row[1] for row in prepared)
    group_count = len(group_sizes)
    query_weights = np.asarray([1.0 / (group_count * group_sizes[row[1]]) for row in prepared], dtype=np.float64)
    arrays = [row[2] for row in prepared]
    mean, scale = _weighted_normalization(arrays, query_weights, config.scale_floor)
    normalized = [(array - mean) / scale for array in arrays]
    positives = [row[3] for row in prepared]
    prior = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    if config.control_prior_coefficient is not None:
        prior[CONTROL_INDEX] = config.control_prior_coefficient * scale[CONTROL_INDEX]

    # SciPy is needed only for fit; prediction uses the sealed NumPy vectors.
    import scipy
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    def objective(weights):
        residual = weights - prior
        loss = 0.5 * config.regularization * float(residual @ residual)
        gradient = config.regularization * residual
        for matrix, positive, mass in zip(normalized, positives, query_weights):
            scores = matrix @ weights
            all_lse = logsumexp(scores)
            positive_lse = logsumexp(scores[positive])
            loss += mass * (all_lse - positive_lse)
            gradient += mass * (np.exp(scores - all_lse) @ matrix
                                - np.exp(scores[positive] - positive_lse) @ matrix[positive])
        if not math.isfinite(loss) or not np.isfinite(gradient).all():
            raise ValueError("Non-finite listwise objective or gradient")
        return float(loss), gradient

    options = {"maxiter": config.maxiter, "maxfun": config.maxfun, "maxls": config.maxls,
               "maxcor": config.maxcor, "ftol": config.tol, "gtol": config.tol}
    initial_loss, _ = objective(prior)
    result = minimize(objective, prior.copy(), method="L-BFGS-B", jac=True, options=options)
    weights = np.asarray(result.x, dtype=np.float64)
    final_loss, gradient = objective(weights)
    if weights.shape != (len(FEATURE_NAMES),) or not np.isfinite(weights).all():
        raise ValueError("Optimizer did not return a finite85 weight vector")
    normalization = {"mean": mean.tolist(), "scale": scale.tolist(),
                     "method": "fit_only_equal_group_then_query_then_candidate_weighted_population_std",
                     "scale_floor": config.scale_floor}
    config_data = asdict(config)
    weights_list, prior_list = weights.tolist(), prior.tolist()
    return seal({"version": MODEL_VERSION, "feature_schema": {"schema_version": SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES)},
                 "feature_schema_checksum": FEATURE_SCHEMA_CHECKSUM, "code_checksums": _code_checksums(),
                 "config": config_data, "config_checksum": digest(config_data),
                 "normalization": normalization, "normalization_checksum": digest(normalization),
                 "weights": weights_list, "weights_checksum": digest(weights_list),
                 "prior_weights": prior_list, "prior_weights_checksum": digest(prior_list),
                 "prior_semantics": "L2 centre; raw control coefficient multiplied by fit scale; no independent intercept",
                 "training": {"input_checksum": digest(input_records), "input_queries": len(queries), "used_queries": len(prepared),
                              "used_candidates": sum(len(array) for array in arrays), "groups": group_count,
                              "group_query_counts": dict(sorted(group_sizes.items())), "group_mass": 1.0 / group_count,
                              "query_weighting": "1/(used_group_count*used_queries_in_group)",
                              "skipped": skipped, "query_order": "sorted_unique_id", "initialization": "prior_weights", "randomness": "none"},
                 "optimizer": {"method": "L-BFGS-B", "analytic_gradient": True, "options": options,
                               "success": bool(result.success), "status": int(result.status), "message": str(result.message),
                               "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                               "total_objective_evaluations": int(result.nfev) + 2,
                               "budget_scope": "minimize limits plus one initial and one final objective audit",
                               "initial_objective": initial_loss, "final_objective": final_loss,
                               "gradient_max_abs": float(np.max(np.abs(gradient))), "restarts": 0},
                 "runtime": {"numpy": np.__version__, "scipy": scipy.__version__, "device": "CPU", "dtype": "float64"},
                 "tie_policy": "descending_score_then_actual_control_proposal_then_input_pool_order",
                 "card_policy": "control_proposal_member_then_best_raw_visual_member_then_first_actual_supplied8175_member",
                 "probability": None, "calibration": "absent", "release_admitted": False})


def _model_arrays(model):
    verify(model)
    if model.get("version") != MODEL_VERSION or model.get("feature_schema") != FEATURE_SCHEMA or model.get("feature_schema_checksum") != FEATURE_SCHEMA_CHECKSUM:
        raise ValueError("Model/feature schema mismatch")
    if model.get("code_checksums") != _code_checksums():
        raise ValueError("Model or frozen feature implementation changed")
    for value, checksum in (("config", "config_checksum"), ("normalization", "normalization_checksum"),
                            ("weights", "weights_checksum"), ("prior_weights", "prior_weights_checksum")):
        if digest(model[value]) != model.get(checksum):
            raise ValueError(value + " checksum mismatch")
    config = FitConfig(**model["config"])
    mean = np.asarray(model["normalization"]["mean"], dtype=np.float64)
    scale = np.asarray(model["normalization"]["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if any(array.shape != (len(FEATURE_NAMES),) or not np.isfinite(array).all() for array in (mean, scale, weights)):
        raise ValueError("Invalid model vector")
    if (scale < config.scale_floor).any():
        raise ValueError("Model scale violates fitted floor")
    return mean, scale, weights


def _representative(candidate):
    provenance, members = candidate["provenance"], set(candidate["card_slugs"])
    control = provenance.get("control_proposal_slug")
    if control is not None:
        if control not in members:
            raise ValueError("Control card is not a selected candidate member")
        return control, "actual8175_proposal"
    visual = []
    for order, evidence in enumerate(provenance.get("visual", [])):
        slug = evidence.get("raw", {}).get("slug")
        rank, channel = evidence.get("rank"), evidence.get("channel")
        if slug not in members or type(rank) is not int or rank < 1 or channel not in CHANNELS:
            raise ValueError("Invalid raw visual member provenance")
        visual.append(((rank, CHANNELS.index(channel), order), slug))
    if visual:
        return min(visual, key=lambda row: row[0])[1], "raw_visual_member"
    for row in provenance.get("control_candidates", []):
        if row.get("slug") not in members:
            raise ValueError("Supplied8175 card is not an actual candidate member")
        return row["slug"], "actual8175_supplied_candidate"
    raise ValueError("Candidate has no actual source member to represent it")


def predict(feature_output, model):
    """Score inference-only features; no labels, product truth or fit data read.

    Equal numeric scores prefer the actual8175 proposal, then the incoming pool
    order. Identifier strings only bind provenance; they are never numeric
    features or newly inferred cross-card equivalences. ``exact_slug`` remains
    None even when the representative card retains the original control slug.
    """
    mean, scale, weights = _model_arrays(model)
    candidates, matrix = _matrix(feature_output)
    scores = ((matrix - mean) / scale) @ weights
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite candidate ranking score")
    order = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]),
                   -candidates[index]["features"]["control.is_proposal"], index))
    ranked = []
    for rank, index in enumerate(order, 1):
        candidate = candidates[index]
        representative, source = _representative(candidate)
        ranked.append({"rank": rank, "input_pool_index": index, "candidate_id": candidate["candidate_id"],
                       "product_id": candidate["product_id"], "identity_status": candidate["identity_status"],
                       "card_slugs": list(candidate["card_slugs"]), "representative_slug": representative,
                       "representative_source": source, "score": float(scores[index]),
                       "is_control_proposal": bool(candidate["features"]["control.is_proposal"])})
    winner = ranked[0] if ranked else None
    return {"schema_version": PREDICTION_VERSION, "model_checksum": model["checksum"],
            "feature_schema_checksum": FEATURE_SCHEMA_CHECKSUM, "ranked_candidates": ranked,
            "candidate_id": winner["candidate_id"] if winner else None,
            "product_id": winner["product_id"] if winner else None,
            "representative_slug": winner["representative_slug"] if winner else None,
            "best_candidate": winner["representative_slug"] if winner else None,
            "representative_source": winner["representative_source"] if winner else None,
            "score": winner["score"] if winner else None,
            "score_margin": ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else None,
            "exact_slug": None, "vintage": {"value": None, "status": "unknown"},
            "probability": None, "confidence": {"status": "uncalibrated_proposal", "probability": None},
            "reason": "linear_listwise_product_proposal" if winner else "empty_candidate_pool",
            "tie_policy": model["tie_policy"], "card_policy": model["card_policy"], "release_admitted": False}
