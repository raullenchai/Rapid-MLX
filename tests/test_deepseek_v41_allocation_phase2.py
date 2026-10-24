from __future__ import annotations

import numpy as np
import pytest

from scripts.deepseek_v41_contrastive import contrastive
from scripts.deepseek_v41_rd_common import affine_bytes
from scripts.deepseek_v41_rd_layer_allocate import layer_allocate


def slices():
    # Expert 0 specializes on one hot slice, expert 1 is cold-specific,
    # expert 2 is a syntax generalist, expert 3 is never routed.
    r = np.array(
        [
            [[8.0, 0.01, 1.0, 0.0]],
            [[0.01, 0.01, 1.0, 0.0]],
            [[0.1, 8.0, 1.0, 0.0]],
            [[1.0, 0.1, 1.0, 0.0]],
        ]
    )
    r = np.repeat(r[:, None], 2, axis=1)
    return (
        r,
        np.where(r > 0, 8, 0),
        ["chat", "medical", "cold_en", "general_reference_en"],
        {
            "chat": "hot",
            "medical": "hot",
            "cold_en": "cold",
            "general_reference_en": "reference",
        },
    )


def test_contrastive_specialist_generalist_never_and_ratio():
    args = slices()
    result = contrastive(*args)
    assert result["candidate"].tolist() == [[False, True, False, False]]
    assert result["protected"].tolist() == [[True, False, True, True]]
    # Multiplying every slice by a common layer magnitude cannot alter topic decisions.
    assert np.array_equal(
        result["candidate"], contrastive(args[0] * 100, *args[1:])["candidate"]
    )


def test_contrastive_needs_reference_routes_for_a_ratio():
    r, counts, names, roles = slices()
    counts = counts.copy()
    counts[3, :, 0, 1] = 0  # the matched reference never routes expert 1
    result = contrastive(r, counts, names, roles)
    assert not result["candidate"][0, 1]
    assert result["cold_specificity"][0, 1] == 0


def test_contrastive_requires_same_cold_topic_stable_in_both_halves():
    r, c, names, roles = slices()
    r[2, 1, 0, 1] = 0.1
    assert not contrastive(r, c, names, roles)["candidate"].any()
    r[2, 1, 0, 1] = 8
    c[2, 1, 0, 1] = 1
    assert not contrastive(r, c, names, roles)["candidate"].any()


def test_contrastive_experts_hot_in_either_half_are_protected():
    r, c, names, roles = slices()
    r[1, 1, 0, 1] = 10
    result = contrastive(r, c, names, roles)
    assert result["protected"][0, 1] and not result["candidate"][0, 1]


def test_contrastive_unknown_roles_and_bad_inputs_fail():
    r, c, names, roles = slices()
    with pytest.raises(ValueError):
        contrastive(r, c, names, {})
    r[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        contrastive(r, c, names, roles)


def test_layer_allocator_caps_protection_uniform_width_and_budget():
    sel = contrastive(*slices())
    cost = np.broadcast_to(
        [0] + [3 * affine_bytes((64, 64), b) + 136 for b in (2, 3, 4)], (1, 4, 4)
    )
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], cost.shape)
    result = layer_allocate(
        sel, q, cost, 100, 100 + 3 * int(cost[0, 0, 1]), 0.5, True, 1
    )
    assert result["layers"]["0"]["kept_expert_ids"] == [0, 2, 3]
    assert result["resident_tensor_bytes"] <= result["budget_bytes"]
    assert len(set(result["layers"]["0"]["projection_bits"].values())) == 1
    assert result["duality_gap"] >= 0
    with pytest.raises(ValueError, match="infeasible"):
        layer_allocate(sel, q, cost, 100, int(cost[0, 0, 1]), 0.9, True, 1)


def test_projection_allocator_prices_and_chooses_down_precision():
    sel = {
        "candidate": np.zeros((1, 2), bool),
        "protected": np.ones((1, 2), bool),
        "sensitivity": np.ones((1, 2)),
        "candidate_count_by_layer": [0],
    }
    cost = np.broadcast_to(
        [0] + [3 * affine_bytes((64, 64), b) + 136 for b in (2, 3, 4)], (1, 2, 4)
    )
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], cost.shape)
    pq = np.broadcast_to(
        [[0.001, 0.0009, 0.0008], [0.001, 0.0009, 0.0008], [0.8, 0.01, 0.009]],
        (1, 2, 3, 3),
    )
    budget = 2 * (2 * affine_bytes((64, 64), 2) + affine_bytes((64, 64), 3) + 136)
    result = layer_allocate(sel, q, cost, 0, budget, topk=1, projection_q=pq)
    assert result["layers"]["0"]["projection_bits"] == {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 3,
    }
    assert result["resident_tensor_bytes"] == budget


def test_fallback_only_after_contrastive_set_and_caps():
    sel = {
        "candidate": np.array([[True, False, False, False]]),
        "protected": np.array([[False, False, False, True]]),
        "sensitivity": np.array([[0.1, 0.01, 1.0, 1.0]]),
        "candidate_count_by_layer": [1],
    }
    cost = np.broadcast_to(
        [0] + [3 * affine_bytes((64, 64), b) + 136 for b in (2, 3, 4)], (1, 4, 4)
    )
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], cost.shape)
    budget = 2 * int(cost[0, 0, 1])
    with pytest.raises(ValueError, match="infeasible"):
        layer_allocate(sel, q, cost, 0, budget, 0.5, False, 1)
    result = layer_allocate(sel, q, cost, 0, budget, 0.5, True, 1)
    assert result["layers"]["0"]["kept_expert_ids"] == [2, 3]
    enough = layer_allocate(sel, q, cost, 0, 3 * int(cost[0, 0, 1]), 0.5, True, 1)
    assert not enough["rd_fallback"]
    assert enough["layers"]["0"]["removed_experts"] <= 1


def test_cold_ratio_uses_language_matched_reference():
    r, c, names, roles = slices()
    r = np.concatenate((r, 100 * r[-1:]), axis=0)
    c = np.concatenate((c, c[-1:]), axis=0)
    names = [
        "chat",
        "medical",
        "cold_archaeology_zh",
        "general_reference_en",
        "general_reference_zh",
    ]
    roles = dict(zip(names, ["hot", "hot", "cold", "reference", "reference"]))
    # Same cold magnitude no longer qualifies when the same-language reference
    # is 100x stronger. An unrelated English reference cannot create a topic.
    assert not contrastive(r, c, names, roles)["candidate"].any()
