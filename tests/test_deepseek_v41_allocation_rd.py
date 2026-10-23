from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.deepseek_v41_rd_allocate import allocate


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


def test_allocator_cli_checks_provenance_stability_and_payload_costs(
    tmp_path, monkeypatch
):
    import sys

    from scripts import deepseek_v41_rd_allocate as cli

    monkeypatch.setattr(cli, "validate_legacy_inventory", lambda *args: None)

    scores = tmp_path / "scores.npz"
    errors = tmp_path / "errors.npz"
    costs = tmp_path / "costs.json"
    sm = {
        "model_revision": "r",
        "routing_topk": 1,
        "config_sha256": "config",
        "slice_names": ["hot"],
    }
    qm = {
        "model_revision": "r",
        "config_sha256": "config",
        "group_size": 64,
        "bits_order": [0, 2, 3, 4],
    }
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
            "--granularity",
            "legacy-expert",
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


def fixture_costs(layers=1, experts=3):
    return np.broadcast_to([0, 3, 5, 7], (layers, experts, 4)).copy()
