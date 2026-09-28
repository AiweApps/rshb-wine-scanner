"""Schema-bound linear listwise selector for one separately admitted CPU fit.

The caller supplies the numeric feature schema and its source files, then calls
``head.fit(queries, config=FitConfig(...))`` or ``head.predict(features, model)``.
The feature output and candidate provenance follow learned_selection_features;
only the explicit numeric schema may vary. ``control.is_proposal`` is required
for the existing control prior and tie policy. No schema, optimizer parameters,
admission, files or runtime are selected or changed by this module.

The algorithm preserves frozen learned_selection_model v1: equal group/query/
candidate normalization, multi-positive listwise loss, L2 around a control prior,
one deterministic L-BFGS-B call and actual-source card representatives. Models
are sealed against this core, the reused FitConfig and the declared feature code.
"""
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from rshb_vine.io import digest, seal, sha256, verify
from rshb_vine.learned_selection_model import FitConfig
from rshb_vine.product_first.visual_selector import CHANNELS


__all__ = ["CandidateListwise", "FitConfig"]
MODEL_VERSION = "candidate-linear-listwise-v1"
PREDICTION_VERSION = "candidate-linear-listwise-proposal-v1"
CONTROL_FEATURE = "control.is_proposal"
ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE_PATHS = (
    "rshb_vine/candidate_listwise.py",
    "rshb_vine/learned_selection_model.py",  # Frozen FitConfig and its validation.
    "rshb_vine/product_first/visual_selector.py",  # Representative channel order.
    "rshb_vine/io.py",
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def _weighted_normalization(arrays, query_weights, scale_floor):
    mean = sum(weight * array.mean(axis=0) for array, weight in zip(arrays, query_weights))
    variance = sum(weight * np.mean((array - mean) ** 2, axis=0)
                   for array, weight in zip(arrays, query_weights))
    scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), scale_floor)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Non-finite fit normalization")
    return mean, scale


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


class CandidateListwise:
    """One explicit feature contract; source paths are files inside this repo.

    Paths may be absolute or repository-relative and are stored canonically as
    repository-relative paths. Include the feature builder and every dependency
    affecting its numeric features; the core dependencies above are added here.
    Fit admission, an immutable one-fit marker, and evaluation stay with caller.
    """

    def __init__(self, *, schema_version, feature_names, feature_source_paths):
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise ValueError("schema_version must be an explicit nonempty string")
        if (not isinstance(feature_names, Sequence) or isinstance(feature_names, (str, bytes))
                or not feature_names or any(not isinstance(name, str) or not name for name in feature_names)
                or len(set(feature_names)) != len(feature_names)):
            raise ValueError("feature_names must be an ordered nonempty sequence of distinct strings")
        if CONTROL_FEATURE not in feature_names:
            raise ValueError("Feature schema must preserve " + CONTROL_FEATURE)
        if (not isinstance(feature_source_paths, Sequence) or isinstance(feature_source_paths, (str, bytes))
                or not feature_source_paths):
            raise ValueError("feature_source_paths must explicitly list the feature implementation files")
        paths = set()
        for source in feature_source_paths:
            path = Path(source)
            path = (path if path.is_absolute() else ROOT / path).resolve()
            relative = path.relative_to(ROOT).as_posix()
            if not path.is_file():
                raise ValueError("Feature source is not a file: " + relative)
            paths.add(relative)
        self.schema_version = schema_version
        self.feature_names = tuple(feature_names)
        self.feature_source_paths = tuple(sorted(paths))
        self._control_index = self.feature_names.index(CONTROL_FEATURE)

    @property
    def feature_schema(self):
        return {"schema_version": self.schema_version, "feature_names": list(self.feature_names)}

    @property
    def feature_schema_checksum(self):
        return digest(self.feature_schema)

    def code_checksums(self):
        paths = sorted(set(CORE_SOURCE_PATHS) | set(self.feature_source_paths))
        return {path: sha256(ROOT / path) for path in paths}

    def _matrix(self, feature_output):
        """Read the declared inference contract, never labels or truth fields."""
        if (feature_output.get("schema_version") != self.schema_version
                or feature_output.get("feature_names") != list(self.feature_names)):
            raise ValueError("Frozen feature schema mismatch")
        candidates = feature_output.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Feature output must contain its ordered candidate list")
        seen, rows = set(), []
        control_count = 0
        for candidate in candidates:
            cid = candidate.get("candidate_id")
            if not isinstance(cid, str) or not cid or cid in seen:
                raise ValueError("Candidate IDs must be nonempty and unique within a target")
            seen.add(cid)
            members = candidate.get("card_slugs")
            if (not isinstance(members, list) or not members
                    or any(not isinstance(slug, str) or not slug for slug in members)
                    or len(set(members)) != len(members)):
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
            if any(cards[slug].get("slug") != slug
                   or cards[slug].get("product_id") != candidate.get("product_id") for slug in members):
                raise ValueError("Candidate identity changed relative to its registry provenance")
            features = candidate.get("features", {})
            if set(features) != set(self.feature_names):
                raise ValueError("Candidate features differ from the declared schema")
            values = [_number(features[name], name) for name in self.feature_names]
            control = candidate.get("provenance", {}).get("control_proposal_slug")
            if control is not None and control not in members:
                raise ValueError("Control proposal is not an actual member")
            if values[self._control_index] != float(control is not None):
                raise ValueError("Control indicator differs from supplied8175 provenance")
            control_count += control is not None
            rows.append(values)
        if control_count > 1:
            raise ValueError("One physical target cannot have multiple8175 control proposals")
        matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(self.feature_names))
        return candidates, matrix

    def fit(self, queries, *, config):
        """Run one optimizer call and return a sealed model; write no files.

        Queries contain id, group, feature_output and a separate positive_mask
        aligned to actual candidates. Multiple positive masses are summed.
        Empty/no-positive/all-positive pools are retained in skipped metadata;
        no candidate or label is invented. This method grants no admission.
        """
        if not isinstance(config, FitConfig):
            raise TypeError("config must be an explicitly constructed frozen FitConfig")
        if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)) or not queries:
            raise ValueError("fit requires a nonempty admitted query sequence")
        code_checksums = self.code_checksums()
        prepared, skipped, input_records = [], [], []
        ids = set()
        for query in queries:
            qid, group = query.get("id"), query.get("group")
            if not isinstance(qid, str) or not qid or qid in ids or not isinstance(group, str) or not group:
                raise ValueError("Every fit query needs a unique id and an explicit group")
            ids.add(qid)
            candidates, matrix = self._matrix(query["feature_output"])
            mask = query.get("positive_mask")
            if not isinstance(mask, (list, tuple, np.ndarray)) or len(mask) != len(candidates) or any(
                not isinstance(value, (bool, int, np.bool_, np.integer)) or value not in (0, 1) for value in mask
            ):
                raise ValueError("positive_mask must be aligned explicit boolean product truth")
            positive = np.asarray(mask, dtype=bool)
            input_records.append({"id": qid, "group": group,
                                  "candidate_ids": [row["candidate_id"] for row in candidates],
                                  "features": matrix.tolist(), "positive_mask": positive.tolist()})
            reason = ("empty_candidate_pool" if not candidates else "positive_absent_from_pool"
                      if not positive.any() else "no_negative_candidate" if positive.all() else None)
            if reason:
                skipped.append({"id": qid, "group": group, "reason": reason,
                                "candidates": len(candidates), "positives": int(positive.sum())})
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
        prior = np.zeros(len(self.feature_names), dtype=np.float64)
        if config.control_prior_coefficient is not None:
            prior[self._control_index] = config.control_prior_coefficient * scale[self._control_index]

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
        if weights.shape != (len(self.feature_names),) or not np.isfinite(weights).all():
            raise ValueError("Optimizer did not return a finite schema-sized weight vector")
        if self.code_checksums() != code_checksums:
            raise ValueError("Core or feature implementation changed during fit")
        normalization = {"mean": mean.tolist(), "scale": scale.tolist(),
                         "method": "fit_only_equal_group_then_query_then_candidate_weighted_population_std",
                         "scale_floor": config.scale_floor}
        config_data = asdict(config)
        weights_list, prior_list = weights.tolist(), prior.tolist()
        return seal({
            "version": MODEL_VERSION, "feature_schema": self.feature_schema,
            "feature_schema_checksum": self.feature_schema_checksum,
            "feature_source_paths": list(self.feature_source_paths), "code_checksums": code_checksums,
            "config": config_data, "config_checksum": digest(config_data),
            "normalization": normalization, "normalization_checksum": digest(normalization),
            "weights": weights_list, "weights_checksum": digest(weights_list),
            "prior_weights": prior_list, "prior_weights_checksum": digest(prior_list),
            "prior_semantics": "L2 centre; raw control coefficient multiplied by fit scale; no independent intercept",
            "training": {"input_checksum": digest(input_records), "input_queries": len(queries),
                         "used_queries": len(prepared), "used_candidates": sum(len(array) for array in arrays),
                         "groups": group_count, "group_query_counts": dict(sorted(group_sizes.items())),
                         "group_mass": 1.0 / group_count,
                         "query_weighting": "1/(used_group_count*used_queries_in_group)",
                         "skipped": skipped, "query_order": "sorted_unique_id",
                         "initialization": "prior_weights", "randomness": "none"},
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
            "probability": None, "calibration": "absent", "release_admitted": False,
        })

    def _model_arrays(self, model):
        verify(model)
        if (model.get("version") != MODEL_VERSION or model.get("feature_schema") != self.feature_schema
                or model.get("feature_schema_checksum") != self.feature_schema_checksum
                or model.get("feature_source_paths") != list(self.feature_source_paths)):
            raise ValueError("Model/feature schema or source contract mismatch")
        if model.get("code_checksums") != self.code_checksums():
            raise ValueError("Model or frozen feature implementation changed")
        for value, checksum in (("config", "config_checksum"), ("normalization", "normalization_checksum"),
                                ("weights", "weights_checksum"), ("prior_weights", "prior_weights_checksum")):
            if digest(model[value]) != model.get(checksum):
                raise ValueError(value + " checksum mismatch")
        config = FitConfig(**model["config"])
        mean = np.asarray(model["normalization"]["mean"], dtype=np.float64)
        scale = np.asarray(model["normalization"]["scale"], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        if any(array.shape != (len(self.feature_names),) or not np.isfinite(array).all()
               for array in (mean, scale, weights)):
            raise ValueError("Invalid model vector")
        if (scale < config.scale_floor).any():
            raise ValueError("Model scale violates fitted floor")
        return mean, scale, weights

    def predict(self, feature_output, model):
        """Return an uncalibrated proposal from features, without any GT input."""
        mean, scale, weights = self._model_arrays(model)
        candidates, matrix = self._matrix(feature_output)
        scores = ((matrix - mean) / scale) @ weights
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite candidate ranking score")
        order = sorted(range(len(candidates)), key=lambda index: (
            -float(scores[index]), -candidates[index]["features"][CONTROL_FEATURE], index))
        ranked = []
        for rank, index in enumerate(order, 1):
            candidate = candidates[index]
            representative, source = _representative(candidate)
            ranked.append({"rank": rank, "input_pool_index": index, "candidate_id": candidate["candidate_id"],
                           "product_id": candidate["product_id"], "identity_status": candidate["identity_status"],
                           "card_slugs": list(candidate["card_slugs"]), "representative_slug": representative,
                           "representative_source": source, "score": float(scores[index]),
                           "is_control_proposal": bool(candidate["features"][CONTROL_FEATURE])})
        winner = ranked[0] if ranked else None
        return {"schema_version": PREDICTION_VERSION, "model_checksum": model["checksum"],
                "feature_schema_checksum": self.feature_schema_checksum, "ranked_candidates": ranked,
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
