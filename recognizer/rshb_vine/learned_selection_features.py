"""Inference-only features for a bounded learned product selector.

The pool is the untruncated four visual Top20 union plus explicitly supplied
8175 proposals. This module accepts no ground truth, fits nothing, makes no
selection, and assigns no probability. Identifiers and raw provenance are kept
outside the finite numeric feature map. All text evidence remains soft.
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

from rshb_vine.product_first.partial_evidence import (
    MAPS, NAMES, extract_packet, match_card,
)
from rshb_vine.product_first.visual_selector import CHANNELS, select_visual_product
from rshb_vine.typed_catalog_lexicon import normalize


SCHEMA_VERSION = "learned-selection-features-v1"
ROLES = (*NAMES, "grape_blend", "color", "sugar", "style")
SOFT_DISAGREEMENT_ROLES = ("color", "sugar", "style")
VERIFIED_STATUSES = {"source_verified", "source_verified_alias"}
DISPUTED_STATUSES = {"conflict", "unresolved", "ambiguous"}
VISUAL_FIELDS = (
    "present", "reciprocal_rank", "raw_similarity", "top_score_gap",
    "reciprocal_rank_x_view_area_fraction",
)
ROLE_FIELDS = (
    "high_full", "high_partial", "low_full", "low_partial",
    "missing", "disputed", "source_verified",
)
FEATURE_NAMES = (
    *(f"visual.{channel}.{field}" for channel in CHANNELS for field in VISUAL_FIELDS),
    *(f"text.{role}.{field}" for role in ROLES for field in ROLE_FIELDS),
    "visual.label_top1_agreement", "visual.context_top1_agreement",
    "control.is_proposal", "control.in_supplied_candidates",
    "source.product_admitted", "source.profile_missing",
    *(f"text.{role}.soft_disagreement" for role in SOFT_DISAGREEMENT_ROLES),
)


def _finite(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {description}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {description}")
    return result


def _without_year(text):
    # Exclude year-like tokens even when a source name happens to contain one.
    return " ".join(token for token in normalize(text).split()
                    if not (token.isdigit() and len(token) == 4
                            and 1800 <= int(token) <= 2099))


def _text_packet(observations):
    raw = deepcopy(list(observations))
    for observation in raw:
        if not isinstance(observation.get("raw_text", ""), str):
            raise ValueError("OCR raw_text must be a string")
        if observation.get("score") is not None:
            _finite(observation["score"], "OCR score")
    extracted = extract_packet({"observations": raw}, {"kind": "supplied_query_observations"})
    groups = {}
    for line in extracted["lines"]:
        normalized = _without_year(line["raw"].get("raw_text", ""))
        line["normalized"] = line["same_reading_group"] = normalized
        line["tokens"] = normalized.split()
        if not normalized:
            continue
        group = groups.setdefault(normalized, {"lines": [], "selected": line})
        group["lines"].append(line["line_id"])
        # Repeated readings are one group; a high reading replaces a low one.
        # No votes accumulate across readers, crops, or repeated OCR packets.
        if line["legacy_threshold_admitted"] and not group["selected"]["legacy_threshold_admitted"]:
            group["selected"] = line
    extracted["lines"] = [group["selected"] for _, group in sorted(groups.items())]
    provenance = [{"normalized_without_year": key, "source_line_ids": group["lines"],
                   "feature_line_id": group["selected"]["line_id"],
                   "threshold_admitted": group["selected"]["legacy_threshold_admitted"]}
                  for key, group in sorted(groups.items())]
    return extracted, raw, provenance


def _feature_claim(profile):
    if profile is None:
        return None
    result = deepcopy(profile)
    result["roles"] = {role: result.get("roles", {}).get(role, {}) for role in ROLES}
    for role in NAMES:
        claim = result["roles"][role]
        claim["values"] = sorted({_without_year(str(value)) for value in claim.get("values", [])} - {""})
    return result


def _candidate_key(slug, cards):
    if slug not in cards:
        raise ValueError("Candidate absent from identity registry: " + slug)
    card = cards[slug]
    return card["product_id"] if card["binding_status"] == "source_admitted_product" else "unresolved-card:" + slug


def _view_area_fraction(row, views):
    index = row.get("query_view_index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(views):
        return 0.0, "missing_query_view"
    view = views[index]
    box, size = view.get("bbox"), view.get("original_size")
    if box is None or size is None:
        return 0.0, "missing_view_geometry"
    if len(box) != 4 or len(size) != 2:
        raise ValueError("Invalid view geometry shape")
    x1, y1, x2, y2 = [_finite(value, "view coordinate") for value in box]
    width, height = [_finite(value, "original image dimension") for value in size]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("View geometry is outside its original image")
    return ((x2 - x1) / width) * ((y2 - y1) / height), "recorded_geometry"


def _role_features(profiles, packet, features):
    matches = {slug: match_card(packet, profile) for slug, profile in profiles.items() if profile is not None}
    diagnostics = {"card_matches": matches, "typed_disagreements": {}}
    lines = {line["line_id"]: line for line in packet["lines"]}
    observed = {
        role: {confidence: sorted({entity["value"] for line in packet["lines"]
                                  if line["legacy_threshold_admitted"] == (confidence == "high")
                                  for entity in line["entities"].get(role, [])})
               for confidence in ("high", "low")}
        for role in MAPS
    }
    for role in ROLES:
        role_claims = [(profile or {}).get("roles", {}).get(role, {}) for profile in profiles.values()]
        usable = [bool(claim.get("comparison_allowed") and claim.get("values") and not claim.get("unknown"))
                  for claim in role_claims]
        prefix = f"text.{role}."
        features[prefix + "missing"] = float(not all(usable))
        features[prefix + "disputed"] = float(any(claim.get("status") in DISPUTED_STATUSES for claim in role_claims))
        features[prefix + "source_verified"] = float(all(usable) and all(
            claim.get("status") in VERIFIED_STATUSES for claim in role_claims))
        # Boolean any-member support describes a product hypothesis, never a
        # synthesized exact card. The per-card witnesses remain separately visible.
        for match in matches.values():
            for finding in match["roles"].get(role, {}).get("findings", []):
                confidence = "high" if finding["threshold_admitted"] else "low"
                extent = "partial" if finding["match_kind"] == "partial_tokens" else "full"
                features[prefix + confidence + "_" + extent] = 1.0
        if role not in MAPS:
            continue
        expected = [sorted(set(claim.get("values", []))) if allow else []
                    for claim, allow in zip(role_claims, usable)]
        seen = observed[role]["high"]
        # A grape list is usually incomplete. Other typed fields get only a
        # soft disagreement, with agreement required across all member cards.
        comparable = bool(all(usable) and all(len(value) == 1 for value in expected)
                          and all(value == expected[0] for value in expected))
        disagreement = (role in SOFT_DISAGREEMENT_ROLES and comparable
                        and len(seen) == 1 and seen != expected[0])
        if role in SOFT_DISAGREEMENT_ROLES:
            features[prefix + "soft_disagreement"] = float(disagreement)
        diagnostics["typed_disagreements"][role] = {
            "high_observed": seen, "low_observed": observed[role]["low"],
            "member_expected": dict(zip(profiles, expected)),
            "source_status": {slug: (profile or {}).get("roles", {}).get(role, {}).get("status", "missing")
                              for slug, profile in profiles.items()},
            "soft_disagreement": bool(disagreement), "hard_veto_allowed": False,
            "reason": "incomplete_grape_list_never_conflicts" if role == "grape_blend" else
                      "single_high_reading_vs_consistent_comparable_member_claims" if disagreement else "no_supported_disagreement",
        }
    diagnostics["support_line_ids"] = sorted({finding["line_id"] for match in matches.values()
                                              for role in match["roles"].values() for finding in role["findings"]})
    assert set(diagnostics["support_line_ids"]) <= set(lines)
    return diagnostics


def build_candidate_features(*, control_slug, raw_visual, observations, cards,
                             claims, control_candidates=None):
    """Return deterministic candidates with the fixed 85-name numeric schema.

    ``raw_visual`` is ONE target from product-first-v1/visual/<digest>.json,
    with ``arms.B0/B3.retrieval.channel_top20``, ``views``, and ``instance_id``.
    ``cards`` and ``claims`` are slug-keyed mappings from identity-registry-v1
    and normalized-card-claims-v2. A missing claim profile is allowed; a missing
    identity-registry card is an error. ``observations`` is the target's original
    OCR observation list; the existing >=.85 boundary is unchanged.

    ``control_slug`` is its actual frozen8175 proposal or None. Optional
    ``control_candidates`` is the actual supplied8175 candidate sequence, each
    a slug string or a mapping containing ``slug``. Callers must bind these
    inputs to the same physical target and receipt; this pure function cannot
    authenticate their external provenance. No other candidate source is used.

    The output is sorted by candidate_id for serialization, not learned rank.
    Features are finite floats, absent visual values are zero with present=0,
    missing/unknown profiles never produce conflict, and all original evidence
    is preserved separately. Geometry contributes only through a rank interaction.
    """
    if not isinstance(cards, Mapping) or not isinstance(claims, Mapping):
        raise TypeError("cards and claims must be slug-keyed mappings")
    channels = {}
    for arm in ("B0", "B3"):
        retrieval = raw_visual.get("arms", {}).get(arm, {}).get("retrieval", {})
        top20 = retrieval.get("channel_top20", {})
        for kind, suffix in (("front_label", "label"), ("context", "context")):
            rows = top20.get(kind, [])
            for row in rows:
                _finite(row.get("score"), "raw visual similarity")
            channels[arm + "_" + suffix] = rows
    visual = select_visual_product(channels, cards)
    pool = {candidate["candidate_id"]: deepcopy(candidate) for candidate in visual["ranked_products"]}
    control_rows = []
    if control_candidates is not None:
        if isinstance(control_candidates, (str, bytes, Mapping)) or not isinstance(control_candidates, Sequence):
            raise TypeError("control_candidates must be a sequence of actual8175 candidates")
        control_rows = [({"slug": row} if isinstance(row, str) else deepcopy(row)) for row in control_candidates]
    supplied_control_slugs = set()
    for row in control_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("slug"), str) or not row["slug"]:
            raise ValueError("Supplied8175 candidate lacks slug")
        supplied_control_slugs.add(row["slug"])
    if control_slug is not None and (not isinstance(control_slug, str) or not control_slug):
        raise ValueError("control_slug must be the actual proposal slug or None")
    all_control_slugs = supplied_control_slugs | ({control_slug} if control_slug else set())
    for slug in sorted(all_control_slugs):
        key = _candidate_key(slug, cards)
        card = cards[slug]
        candidate = pool.setdefault(key, {"candidate_id": key, "product_id": card["product_id"],
                                         "identity_status": card["binding_status"], "card_slugs": [],
                                         "channels": {}, "representative_slug": slug})
        if slug not in candidate["card_slugs"]:
            candidate["card_slugs"].append(slug)
    packet, raw_observations, reading_groups = _text_packet(observations)
    views = raw_visual.get("views", [])
    output = []
    for key, candidate in sorted(pool.items()):
        slugs = sorted(candidate["card_slugs"])
        for slug in slugs:
            profile = claims.get(slug)
            if profile is not None and (profile.get("slug") != slug
                                        or profile.get("product_id") != cards[slug]["product_id"]):
                raise ValueError("Claim profile identity mismatch: " + slug)
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        provenance = {"visual": [], "control_candidates": [row for row in control_rows if row["slug"] in slugs],
                      "control_proposal_slug": control_slug if control_slug in slugs else None,
                      "cards": {slug: deepcopy(cards[slug]) for slug in slugs},
                      "claims": {slug: deepcopy(claims.get(slug)) for slug in slugs}, "geometry": {}}
        for channel in CHANNELS:
            evidence = candidate["channels"].get(channel)
            provenance["visual"].extend({"channel": channel, "rank": rank, "raw": deepcopy(row)}
                                        for rank, row in enumerate(channels[channel], 1) if row["slug"] in slugs)
            if evidence is None:
                continue
            rank, row = evidence["rank"], evidence["raw"]
            area, status = _view_area_fraction(row, views)
            score = _finite(row["score"], "raw visual similarity")
            values = (1.0, 1.0 / rank, score, float(channels[channel][0]["score"]) - score, area / rank)
            for field, value in zip(VISUAL_FIELDS, values):
                features[f"visual.{channel}.{field}"] = value
            provenance["geometry"][channel] = {"status": status, "view_area_fraction": area,
                                                 "query_view_index": row.get("query_view_index")}
        for kind in ("label", "context"):
            features[f"visual.{kind}_top1_agreement"] = float(all(
                candidate["channels"].get(arm + "_" + kind, {}).get("rank") == 1 for arm in ("B0", "B3")))
        features["control.is_proposal"] = float(control_slug in slugs)
        features["control.in_supplied_candidates"] = float(bool(supplied_control_slugs.intersection(slugs)))
        features["source.product_admitted"] = float(candidate["identity_status"] == "source_admitted_product")
        features["source.profile_missing"] = float(any(slug not in claims or claims[slug] is None for slug in slugs))
        profiles = {slug: _feature_claim(claims.get(slug)) for slug in slugs}
        text_evidence = _role_features(profiles, packet, features)
        if tuple(features) != FEATURE_NAMES or not all(math.isfinite(value) for value in features.values()):
            raise ValueError("Feature schema or finite-vector invariant failed")
        output.append({"candidate_id": key, "product_id": candidate["product_id"],
                       "identity_status": candidate["identity_status"], "card_slugs": slugs,
                       "representative_slug": candidate["representative_slug"], "features": features,
                       "provenance": provenance, "text_evidence": text_evidence})
    return {"schema_version": SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES), "candidates": output,
            "pool_policy": "untruncated_visual_4xTop20_plus_supplied8175_only",
            "query_evidence": {"instance_id": raw_visual.get("instance_id"), "views": deepcopy(views),
                               "rotation_ccw": raw_visual.get("rotation_ccw"),
                               "arms": {arm: {key: deepcopy(value) for key, value in data.items() if key != "retrieval"}
                                        for arm, data in raw_visual.get("arms", {}).items()},
                               "observations": raw_observations, "reading_groups": reading_groups,
                               "ocr_threshold": .85, "confidence": "reader_specific_uncalibrated"},
            "is_selection": False, "hard_veto_used": False, "probability": None}
