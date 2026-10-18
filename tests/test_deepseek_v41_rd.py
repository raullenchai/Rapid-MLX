from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deepseek_v41_rd_allocate import allocate
from scripts.deepseek_v41_rd_common import (
    affine_bytes,
    aggregate_scores,
    bank_plan,
    masked_route,
)
from scripts.evaluate_deepseek_v41_rd import (
    compare,
    select_knee,
    token_metrics,
    validate_allocation,
)


def test_gate_free_scores_average_routed_tokens_and_protect_domain_max():
    shape = (2, 2, 1, 2)
    counts = np.ones(shape)
    counts[0, 0, 0, 0] = 2
    norms = np.ones(shape)
    norms[0, 0, 0, 0] = 8
    squared = norms**2
    reap = norms * 0.25
    result = aggregate_scores(reap, norms, squared, counts)
    assert result["man_by_slice"][0, 0, 0] == 3  # (8+1)/(2+1)
    assert result["msan_by_slice"][0, 0, 0] == pytest.approx(65 / 3)
    assert result["reap_by_slice"][0, 0, 0] == 2.25
    assert result["robust_man"][0, 0] == 3
    assert result["robust_man"][0, 1] == 1
    # Changing gate weights cannot change MAN/MSAN.
    changed = aggregate_scores(reap * 17, norms, squared, counts)
    np.testing.assert_array_equal(result["man_by_slice"], changed["man_by_slice"])


def test_never_routed_and_empty_half_are_finite():
    z = np.zeros((1, 2, 1, 3))
    result = aggregate_scores(z, z, z, z)
    assert all(np.isfinite(a).all() for a in result.values())
    assert not result["robust_reap"].any()
    with pytest.raises(ValueError):
        aggregate_scores(z - 1, z, z, z)


@pytest.mark.parametrize("bits,expected", [(2, 11059200), (3, 15482880), (4, 19906560)])
def test_exact_expert_bytes(bits, expected):
    assert 3 * affine_bytes((2304, 5120), bits) == expected


def test_bank_remap_and_router_slicing_design():
    bits = [0, 3, 2, 0, 4, 3, 2, 2]
    plan = bank_plan(bits, topk=3)
    assert plan["kept_expert_ids"] == [1, 2, 4, 5, 6, 7]
    assert plan["original_to_compact"] == [-1, 0, 1, -1, 2, 3, 4, 5]
    stack = np.arange(8 * 3 * 4).reshape(8, 3, 4)
    router = np.arange(8 * 4).reshape(8, 4)
    compact_router = router[plan["kept_expert_ids"]]
    for compact, original in enumerate(plan["kept_expert_ids"]):
        bank = str(plan["compact_to_bank"][compact])
        local = plan["compact_to_local"][compact]
        np.testing.assert_array_equal(
            stack[plan["banks"][bank]][local], stack[original]
        )
        np.testing.assert_array_equal(compact_router[compact], router[original])
    with pytest.raises(ValueError, match="fewer"):
        bank_plan([2, 0], topk=2)


def test_mask_changes_selection_without_biasing_weights_and_measures_mass():
    scores = np.array([[0.1, 0.2, 0.3, 0.4]])
    bias = np.array([100.0, 0.0, 0.0, 0.0])
    chosen, weights, mass = masked_route(scores, bias, [1, 2, 3], topk=2)
    assert chosen.tolist() == [[3, 2]]
    np.testing.assert_allclose(weights, [[4 / 7, 3 / 7]])
    np.testing.assert_allclose(mass, [0.8])  # original top2 are experts 0 and 3
    for bad in ([0], [1, 1], [-1, 2], [0, 4]):
        with pytest.raises(ValueError):
            masked_route(scores, bias, bad, 2)


def fixture_costs(layers=1, experts=3):
    return np.broadcast_to([0, 3, 5, 7], (layers, experts, 4)).copy()


def brute_optimum(s, q, c, budget, minimum):
    optimum = float("inf")
    for choices in itertools.product(range(4), repeat=s.size):
        state = np.array(choices).reshape(s.shape)
        if np.any(np.count_nonzero(state, axis=-1) < minimum):
            continue
        if np.take_along_axis(c, state[..., None], -1).sum() > budget:
            continue
        distortion = (s * np.take_along_axis(q, state[..., None], -1)[..., 0]).sum()
        optimum = min(optimum, distortion)
    return optimum


def test_equal_slope_supported_solution_is_globally_optimal():
    s = np.array([[4.0, 2.0, 1.0]])
    q = np.broadcast_to([1.0, 0.4, 0.15, 0.05], (1, 3, 4))
    costs = fixture_costs()
    # Compare every small integer budget; a zero dual gap must certify optimum.
    certified = 0
    for budget in range(3, 22):
        result = allocate(s, q, costs, 2, budget + 2, min_kept=1, topk=1)
        optimum = brute_optimum(s, q, costs, budget, 1)
        assert result["resident_tensor_bytes"] <= budget + 2
        assert result["dual_lower_bound"] <= optimum + 1e-9
        assert result["predicted_distortion"] >= optimum - 1e-9
        if result["duality_gap"] < 1e-8:
            assert result["predicted_distortion"] == pytest.approx(optimum)
            certified += 1
    assert certified >= 3


def test_allocator_layer_guards_shared_experts_and_infeasible_budget():
    s = np.array([[100.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], (2, 3, 4))
    result = allocate(s, q, fixture_costs(2), 10, 22, min_kept=2, topk=2)
    assert result["shared_experts"] == "unchanged"
    assert all(
        len(layer_id["kept_expert_ids"]) >= 2 for layer_id in result["layers"].values()
    )
    with pytest.raises(ValueError, match="infeasible"):
        allocate(s, q, fixture_costs(2), 10, 21, min_kept=2, topk=2)
    with pytest.raises(ValueError, match="guard"):
        allocate(s, q, fixture_costs(2), 0, 100, min_kept=1, topk=2)


def test_allocator_handles_ties_by_spending_feasible_marginal_steps():
    s = np.ones((1, 8))
    q = np.broadcast_to([1.0, 0.2, 0.1, 0.02], (1, 8, 4))
    result = allocate(s, q, fixture_costs(1, 8), 0, 24, min_kept=1, topk=1)
    assert result["resident_tensor_bytes"] == 24
    assert result["predicted_distortion"] == pytest.approx(1.6)


@pytest.mark.parametrize("bad", ["nan", "negative", "fractional_cost", "q0"])
def test_allocator_rejects_invalid_inputs(bad):
    s = np.ones((1, 3))
    q = np.broadcast_to([1.0, 0.3, 0.1, 0.01], (1, 3, 4)).copy()
    c = fixture_costs().astype(float)
    if bad == "nan":
        s[0, 0] = np.nan
    elif bad == "negative":
        s[0, 0] = -1
    elif bad == "fractional_cost":
        c[0, 0, 1] = 3.5
    else:
        q[0, 0, 0] = 0.9
    with pytest.raises(ValueError):
        allocate(s, q, c, 0, 21, 1, 1)


def test_kl_direction_shift_invariance_and_flip_rate():
    r = np.array([[1000.0, 1001.0], [-1000.0, 1000.0]])
    c = np.array([[1001.0, 1000.0], [-999.0, 999.0]])
    kl, flip = token_metrics(r, c)
    assert kl[0] == pytest.approx(np.tanh(0.5))
    assert flip.tolist() == [True, False]
    np.testing.assert_allclose(token_metrics(r, r + 777)[0], 0, atol=1e-12)
    with pytest.raises(ValueError):
        token_metrics(r, c[:, :1])


def write_cache(root, logits, mode, ids=(1, 2), corpus="hash"):
    from scripts.deepseek_v41_rd_common import file_hash

    root.mkdir()
    np.save(root / "domain-logits.npy", np.array(logits, np.float32))
    np.save(root / "domain-tokens.npy", np.array(ids))
    np.save(root / "domain-mass.npy", np.ones((len(ids), 2)) * 0.75)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "model_revision": "revision",
                "tokenizer_sha256": "tokenizer",
                "slices": {
                    "domain": {
                        "corpus_sha256": [corpus],
                        "logits_sha256": file_hash(root / "domain-logits.npy"),
                    }
                },
            }
        )
    )


def test_cache_comparison_validates_identity_and_reports_domains(tmp_path):
    r, c = tmp_path / "reference", tmp_path / "candidate"
    logits = [[1.0, 2.0], [3.0, 0.0]]
    write_cache(r, logits, "unpruned_reference")
    write_cache(c, logits, "pruning_mask_only")
    report = compare(r, c)
    assert report["slices"]["domain"]["kl_nats"] == 0
    assert report["slices"]["domain"]["mean_retained_routing_mass"] == 0.75
    np.save(c / "domain-tokens.npy", np.array([1, 3]))
    with pytest.raises(ValueError, match="token IDs"):
        compare(r, c)


def test_cache_comparison_rejects_corrupted_logits(tmp_path):
    r, c = tmp_path / "reference", tmp_path / "candidate"
    write_cache(r, [[1.0, 2.0], [3.0, 0.0]], "unpruned_reference")
    write_cache(c, [[1.0, 2.0], [3.0, 0.0]], "pruning_mask_only")
    np.save(c / "domain-logits.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="hash"):
        compare(r, c)


def test_knee_uses_safe_larger_neighbor_and_fixed_worst_domain_gate():
    points = [
        {"budget_gb": b, "worst_domain_kl_nats": k}
        for b, k in [(260, 0.001), (240, 0.002), (220, 0.004), (200, 0.04), (185, 0.1)]
    ]
    # Largest rise per GB is 200 ->185; safe-side 200 fails .02 ->220.
    assert select_knee(points, 0.02)["budget_gb"] == 220
    with pytest.raises(ValueError, match="no safe"):
        select_knee(points, 0)


@pytest.mark.requires_mlx
def test_quantization_errors_use_cpu_and_exact_mlx_payload():
    mx = pytest.importorskip("mlx.core")
    from scripts.deepseek_v41_expert_quant_error import quant_error

    with mx.stream(mx.cpu):
        rng = np.random.default_rng(5)
        weights = [
            mx.array((rng.normal(size=(64, 64)) / 8).astype(np.float32)).astype(
                mx.bfloat16
            )
            for _ in range(3)
        ]
        probes = mx.array(rng.normal(size=(8, 64)).astype(np.float32))
        errors, costs = quant_error(weights, probes)
        assert errors[0] == 1
        assert np.all(np.isfinite(errors))
        assert errors[3] < errors[1]
        assert costs.tolist() == [0] + [
            3 * affine_bytes((64, 64), b) for b in (2, 3, 4)
        ]


@pytest.mark.requires_mlx
def test_native_score_probe_preserves_moe_outputs_and_gate_free_norms():
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.moe import MoE
    from scripts.calibrate_deepseek_v41_reap import install_native_score_probe

    with mx.stream(mx.cpu):
        args = ModelArgs(
            dim=64, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2
        )
        moe = MoE(args)
        model = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)])
        x = mx.ones((1, 1, 64))
        expected = np.array(moe(x))
        seen = []
        install_native_score_probe(model, lambda *v: seen.append(v))
        actual = np.array(moe(x))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        assert len(seen) == 2
        assert sum(v[2] for v in seen) == pytest.approx(1.0)
        assert all(v[3] > 0 for v in seen)


@pytest.mark.requires_mlx
def test_native_mask_probe_preserves_shared_expert_and_topk():
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.moe import MoE
    from scripts.evaluate_deepseek_v41_rd import install_native_masks

    with mx.stream(mx.cpu):
        args = ModelArgs(
            dim=64, moe_inter_dim=64, n_routed_experts=4, n_activated_experts=2
        )
        moe = MoE(args)
        model = SimpleNamespace(layers=[SimpleNamespace(ffn=moe)], args=args)
        allocation = {
            "layers": {"0": {**bank_plan([2, 0, 3, 0], 2), "bit_widths": [2, 0, 3, 0]}}
        }
        masses = {0: []}
        install_native_masks(model, allocation, masses)
        bias = np.array(moe.gate.bias)
        weights, ids = moe.gate(mx.ones((1, 64)))
        assert set(ids.tolist()[0]) == {0, 2}
        assert weights.sum().item() == pytest.approx(args.route_scale)
        np.testing.assert_array_equal(np.array(moe.gate.bias), bias)
        assert 0 <= masses[0][0] <= 1
        bad = json.loads(json.dumps(allocation))
        bad["layers"]["0"]["kept_expert_ids"] = [0, 1]
        with pytest.raises(ValueError):
            validate_allocation(bad, 1, 4, 2)


def test_allocator_cli_checks_provenance_stability_and_payload_costs(
    tmp_path, monkeypatch
):
    import sys

    from scripts import deepseek_v41_rd_allocate as cli

    scores = tmp_path / "scores.npz"
    errors = tmp_path / "errors.npz"
    costs = tmp_path / "costs.json"
    sm = {"model_revision": "r", "routing_topk": 1}
    qm = {"model_revision": "r", "group_size": 64, "bits_order": [0, 2, 3, 4]}
    payload = fixture_costs(1, 2)
    charged = payload.copy()
    charged[..., 1:] += 1
    np.savez(
        scores,
        robust_reap=np.ones((1, 2)),
        split_keep_overlap=np.ones((1, 3)),
        metadata=json.dumps(sm),
    )
    np.savez(
        errors,
        q=np.broadcast_to([1.0, 0.3, 0.1, 0.01], (1, 2, 4)),
        completed=np.ones((1, 2), bool),
        expert_weight_bytes=payload,
        metadata=json.dumps(qm),
    )
    cost = {
        "model_revision": "r",
        "group_size": 64,
        "fixed_non_expert_bytes": 2,
        "router_row_bytes_by_layer": {"0": 1},
        "expert_bytes": charged.tolist(),
    }
    costs.write_text(json.dumps(cost))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "allocate",
            "--granularity", "legacy-expert",
            "--scores",
            str(scores),
            "--errors",
            str(errors),
            "--costs",
            str(costs),
            "--budgets-gb",
            "0.000000016",
            "--min-kept",
            "1",
            "--topk",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    cli.main()
    result = json.loads(next((tmp_path / "out").glob("*.json")).read_text())
    assert result["resident_tensor_bytes"] <= 16
    cost["expert_bytes"][0][0][1] += 1
    costs.write_text(json.dumps(cost))
    with pytest.raises(ValueError, match="measured MLX payloads"):
        cli.main()
    cost["model_revision"] = "other"
    costs.write_text(json.dumps(cost))
    with pytest.raises(ValueError, match="revision"):
        cli.main()
