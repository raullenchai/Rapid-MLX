"""Contrastive topic pruning; no absence-of-evidence pruning."""

import numpy as np


def contrastive(
    reap_halves,
    counts_halves,
    slice_names,
    roles,
    hot_threshold=0.25,
    cold_ratio=4.0,
    min_cold_routes=2,
    ratio_tolerance=0.5,
):
    r, counts = np.asarray(reap_halves, float), np.asarray(counts_halves)
    if (
        r.ndim != 4
        or r.shape[1] != 2
        or r.shape != counts.shape
        or len(slice_names) != len(r)
        or len(set(slice_names)) != len(r)
        or not np.isfinite(r).all()
        or not np.isfinite(counts).all()
        or np.any(r < 0)
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError("expected finite nonnegative [slice,2,layer,expert] arrays")
    if (
        not np.isfinite(
            [hot_threshold, cold_ratio, min_cold_routes, ratio_tolerance]
        ).all()
        or min_cold_routes != int(min_cold_routes)
        or isinstance(min_cold_routes, (bool, np.bool_))
        or not hot_threshold > 0
        or cold_ratio <= 1
        or min_cold_routes < 1
        or not 0 <= ratio_tolerance <= 1
    ):
        raise ValueError("invalid contrastive thresholds")
    if any(role not in ("hot", "cold", "reference") for role in roles.values()):
        raise ValueError("unsupported slice role")
    ids = {
        role: [i for i, name in enumerate(slice_names) if roles.get(name) == role]
        for role in ("hot", "cold", "reference")
    }
    if not ids["hot"] or not ids["cold"] or not ids["reference"]:
        raise ValueError("need hot, cold and language-matched general reference slices")
    if any(name not in roles for name in slice_names):
        raise ValueError("every calibrated slice needs an explicit role")
    # All slices/halves have equal token budgets. Normalize within each layer
    # for protection only; the RD objective below still uses raw REAP magnitudes.
    normalized = r / np.maximum(r.mean(axis=-1, keepdims=True), 1e-30)
    hot_halves = normalized[ids["hot"]].max(axis=0)
    hot_score = hot_halves.max(axis=0)
    protected = hot_score >= hot_threshold
    never = (counts == 0).all(axis=(0, 1))
    references = []
    for cold in ids["cold"]:
        _, separator, language = slice_names[cold].rpartition("_")
        expected = "general_reference_" + language
        matches = [i for i in ids["reference"] if slice_names[i] == expected]
        if not separator or not language or len(matches) != 1:
            raise ValueError(
                "each cold slice needs an explicit language suffix and one language-matched general reference"
            )
        references.append(matches[0])
    ref = r[references]
    ratios = r[ids["cold"]] / np.maximum(ref, 1e-30)
    # An expert the matched reference never routes has no ratio evidence.
    enough = (counts[ids["cold"]] >= min_cold_routes) & (counts[references] > 0)
    stable = (ratios.min(axis=1) >= cold_ratio) & enough.all(axis=1)
    stable &= (ratios.max(axis=1) - ratios.min(axis=1)) <= ratio_tolerance * np.maximum(
        ratios.max(axis=1), 1e-30
    )
    # A single cold topic must meet the ratio/count/stability tests in BOTH
    # halves. Maxima from different topics cannot fabricate a stable candidate.
    candidate = stable.any(axis=0) & ~protected & ~never
    return {
        "candidate": candidate,
        "protected": protected | never,
        "never_routed": never,
        "hot_score": hot_score,
        "cold_specificity": np.where(enough, ratios, 0).min(axis=1).max(axis=0),
        "sensitivity": r[ids["hot"]].sum(axis=1).max(axis=0),
        "candidate_count_by_layer": candidate.sum(axis=-1).tolist(),
    }
