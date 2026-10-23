"""Homogeneous layer/projection RD options with explicit hot-pruning fallback."""

from itertools import product

import numpy as np

PRESETS = {
    "T256": {
        "budget_bytes": 200_000_000_000,
        "max_prune_fraction": 0.10,
        "fallback": False,
        "allow_hot_pruning": False,
    },
    "T128": {
        "budget_bytes": 100_000_000_000,
        "max_prune_fraction": 0.60,
        "fallback": True,
        "allow_hot_pruning": True,
    },
}


def checked_byte_costs(costs):
    """Bound NumPy byte arithmetic before casting or summing any allocation."""
    if (
        np.any(costs >= 2**63)
        or sum(int(value) for value in costs[..., -1].flat) > np.iinfo(np.int64).max
    ):
        raise ValueError("byte costs exceed safe int64 accumulation range")
    return costs.astype(np.int64)


def router_row_bytes(costs):
    costs = np.asarray(costs)
    low, middle = int(costs[0, 1]), int(costs[0, 2])
    row = low - 5 * (middle - low) // 2
    if row < 0 or not np.all(costs == costs[0]):
        raise ValueError("layer allocator requires equal-shaped expert matrices")
    return row


def routed_and_router_bytes(costs, kept, bits, row=None):
    """Equal-shaped gate/up/down matrices, including one router row per expert."""
    costs = np.asarray(costs)
    if row is None:
        row = router_row_bytes(costs)
    return (
        sum(
            sum(int(value) for value in costs[kept, b - 1]) - len(kept) * row
            for b in bits
        )
        // 3
        + len(kept) * row
    )


def layer_allocate(
    selection,
    q,
    expert_costs,
    fixed_bytes,
    budget_bytes,
    max_prune_fraction=0.1,
    fallback=False,
    topk=6,
    projection_q=None,
    allow_hot_pruning=False,
):
    if allow_hot_pruning and not fallback:
        raise ValueError("--allow-hot-pruning requires --rd-fallback")
    if allow_hot_pruning and "never_routed" not in selection:
        raise ValueError("hot pruning requires an explicit never-routed mask")
    s = np.asarray(selection["sensitivity"], float)
    q, costs = np.asarray(q, float), np.asarray(expert_costs)
    if (
        s.ndim != 2
        or q.shape != (*s.shape, 4)
        or costs.shape != q.shape
        or not np.isfinite(q).all()
        or np.any(q < 0)
        or not np.isfinite(s).all()
        or np.any(s < 0)
        or not 0 <= max_prune_fraction < 1
        or not 1 <= topk <= s.shape[1]
        or not np.isfinite(costs).all()
        or np.any(costs != np.floor(costs))
        or not np.isfinite(fixed_bytes)
        or not np.isfinite(budget_bytes)
        or fixed_bytes != int(fixed_bytes)
        or budget_bytes != int(budget_bytes)
        or not np.all(q[..., 0] == 1)
        or fixed_bytes < 0
        or budget_bytes < 0
        or np.any(costs[..., 0] != 0)
        or np.any(costs[..., 1:] <= costs[..., :-1])
    ):
        raise ValueError("invalid layer RD inputs")
    fixed_bytes, budget_bytes = int(fixed_bytes), int(budget_bytes)
    costs = checked_byte_costs(costs)
    for key in ("candidate", "protected"):
        if np.asarray(selection[key]).shape != s.shape:
            raise ValueError("selection masks must match sensitivity")
    never = np.asarray(selection.get("never_routed", np.zeros(s.shape, bool)), bool)
    if never.shape != s.shape or np.any(never & selection["candidate"]):
        raise ValueError("invalid never-routed mask")
    if projection_q is not None:
        projection_q = np.asarray(projection_q)
        if (
            projection_q.shape != (*s.shape, 3, 3)
            or not np.isfinite(projection_q).all()
            or np.any(projection_q < 0)
        ):
            raise ValueError("projection errors must be [L,E,gate/up/down,2/3/4]")
    fallback_requested = fallback
    contrastive_minimum = int(fixed_bytes)
    for layer in range(len(s)):
        candidates = np.asarray(selection["candidate"][layer], bool)
        protected = np.asarray(selection["protected"][layer], bool)
        if np.any(candidates & protected):
            raise ValueError("contrastive candidates cannot be protected")
        removal_cap = min(
            int(s.shape[1] * max_prune_fraction),
            s.shape[1] - topk,
            int(candidates.sum()),
        )
        contrastive_minimum += (s.shape[1] - removal_cap) * int(costs[layer, 0, 1])
    # Fallback is permitted only if even all-2-bit contrastive-only choices
    # cannot meet the target. A request is not permission to prune extra hot
    # utility merely to purchase higher bit widths when contrastive fits.
    fallback = fallback and contrastive_minimum > budget_bytes
    options = []
    for layer in range(len(s)):
        candidate = selection["candidate"][layer]
        protected = selection["protected"][layer]
        # Hot protection can be relaxed only by an explicit fallback opt-in.
        # Never-routed experts remain immutable even under that opt-in.
        removable = sorted(
            np.flatnonzero(
                ~never[layer] & (~protected | (fallback and allow_hot_pruning))
            ),
            key=lambda e: (not candidate[e], bool(protected[e]), s[layer, e], int(e)),
        )
        if not fallback:
            removable = [e for e in removable if candidate[e]]
        cap = min(
            int(s.shape[1] * max_prune_fraction), s.shape[1] - topk, len(removable)
        )
        layer_options = []
        widths = (
            list(product((2, 3, 4), repeat=3))
            if projection_q is not None
            else [(b, b, b) for b in (2, 3, 4)]
        )
        # Equal matrix parameter counts are specific to this checkpoint.
        row = router_row_bytes(costs[layer])
        for bits in widths:
            if projection_q is None:
                keep_q = q[layer, :, bits[0] - 1]
            else:
                keep_q = sum(
                    projection_q[layer, :, i, b - 2] for i, b in enumerate(bits)
                )
            # Removing an expert trades s*q for s. Rank that incremental RD
            # penalty, not absolute saliency, while exhausting contrastive
            # candidates before touching the remaining unprotected experts.
            ordered = sorted(
                removable,
                key=lambda e: (
                    not candidate[e],
                    bool(protected[e]),
                    s[layer, e] * (1 - keep_q[e]),
                    int(e),
                ),
            )
            for removed in range(cap + 1):
                kept = np.array(
                    sorted(set(range(s.shape[1])) - set(ordered[:removed])), int
                )
                size = routed_and_router_bytes(costs[layer], kept, bits, row)
                loss = float(
                    s[layer, ordered[:removed]].sum()
                    + (s[layer, kept] * keep_q[kept]).sum()
                )
                layer_options.append((size, loss, kept.tolist(), bits, removed))
        options.append(layer_options)
    remaining = budget_bytes - fixed_bytes
    if sum(min(o[0] for o in opts) for opts in options) > remaining:
        raise ValueError(
            "infeasible under hot/never-routed protection and per-layer pruning caps; "
            "--rd-fallback --allow-hot-pruning explicitly permits hot expert removal; "
            "never-routed protection, caps and top-k floor remain enforced"
        )

    def solve(lam):
        chosen = [
            min(opts, key=lambda o: (o[1] + lam * o[0], o[0], o[3])) for opts in options
        ]
        return chosen, sum(o[0] for o in chosen), sum(o[1] for o in chosen)

    lo, hi = 0.0, 1.0
    while solve(hi)[1] > remaining:
        hi *= 2
    best, used, loss = solve(hi)
    lower = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        choice, size, error = solve(mid)
        lower = max(lower, error + mid * (size - remaining))
        if size > remaining:
            lo = mid
        else:
            hi = mid
            best, used, loss = choice, size, error
    while True:
        upgrades = [
            ((old[1] - new[1]) / (new[0] - old[0]), layer, new)
            for layer, (old, opts) in enumerate(zip(best, options))
            for new in opts
            if old[0] < new[0] <= old[0] + remaining - used and new[1] < old[1]
        ]
        if not upgrades:
            break
        _, layer, new = max(upgrades, key=lambda x: (x[0], -x[1]))
        used += new[0] - best[layer][0]
        loss += new[1] - best[layer][1]
        best[layer] = new
    return {
        "schema_version": 2,
        "granularity": "projection" if projection_q is not None else "layer",
        "budget_bytes": int(budget_bytes),
        "fixed_non_expert_bytes": int(fixed_bytes),
        "resident_tensor_bytes": int(fixed_bytes + used),
        "predicted_distortion": loss,
        "dual_lower_bound": lower,
        "duality_gap": max(0.0, loss - lower),
        "contrastive_count_by_layer": selection["candidate_count_by_layer"],
        "protected_count_by_layer": selection["protected"].sum(-1).tolist(),
        "max_prune_fraction": max_prune_fraction,
        "rd_fallback": fallback,
        "rd_fallback_requested": fallback_requested,
        "allow_hot_pruning": allow_hot_pruning,
        "contrastive_only_minimum_bytes": contrastive_minimum,
        "layers": {
            str(i): {
                "kept_expert_ids": o[2],
                "projection_bits": dict(
                    zip(("gate_proj", "up_proj", "down_proj"), o[3])
                ),
                "summed_kept_sensitivity": float(s[i, o[2]].sum()),
                "removed_experts": o[4],
                "removed_protected_experts": int(
                    np.asarray(selection["protected"][i], bool).sum()
                    - np.asarray(selection["protected"][i], bool)[o[2]].sum()
                ),
                "routed_and_router_bytes": o[0],
            }
            for i, o in enumerate(best)
        },
    }
