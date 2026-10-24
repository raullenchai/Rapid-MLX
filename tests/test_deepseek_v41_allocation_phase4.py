from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts.deepseek_v41_rd_allocate import pruning_audit
from scripts.deepseek_v41_rd_build import mtp_byte_costs
from scripts.deepseek_v41_rd_common import file_hash
from scripts.deepseek_v41_rd_layer_allocate import PRESETS, layer_allocate
from scripts.repack_deepseek_v41_native import CheckpointIndex


def selection():
    return {
        "candidate": np.array([[True, False, False, False, False, False]]),
        "protected": np.array([[False, False, True, True, True, True]]),
        "never_routed": np.array([[False, False, False, False, False, True]]),
        # Max raw REAP over hot slices, as returned by contrastive().
        "sensitivity": np.array([[100.0, 50.0, 3.0, 1.0, 1.0, 0.0]]),
        "candidate_count_by_layer": [1],
    }


def solve(budget, **kwargs):
    costs = np.broadcast_to([0, 30, 42, 54], (1, 6, 4))
    q = np.broadcast_to([1.0, 0.0, 0.0, 0.0], costs.shape).copy()
    q[0, 2, 1:] = 0.9  # RD penalty .3, ahead of protected experts 3 and 4.
    return layer_allocate(
        selection(), q, costs, 0, budget, max_prune_fraction=0.8, topk=1, **kwargs
    )


@pytest.mark.parametrize("removed", [1, 2, 3, 4])
def test_hot_fallback_order_rd_penalty_and_stable_ties(removed):
    result = solve((6 - removed) * 30, fallback=True, allow_hot_pruning=True)
    spec = result["layers"]["0"]
    assert spec["kept_expert_ids"] == sorted(set(range(6)) - set(range(removed)))
    assert spec["removed_protected_experts"] == max(0, removed - 2)
    assert 5 in spec["kept_expert_ids"]


def test_hot_pruning_flag_gating_guards_and_fallback_activation():
    assert PRESETS["T128"]["allow_hot_pruning"]
    assert not PRESETS["T256"]["allow_hot_pruning"]
    with pytest.raises(ValueError, match="requires --rd-fallback"):
        solve(90, allow_hot_pruning=True)
    with pytest.raises(ValueError, match="--allow-hot-pruning"):
        solve(90, fallback=True)
    # Even with opt-in, never-routed and per-layer cap cannot be bypassed.
    with pytest.raises(ValueError, match="infeasible"):
        solve(30, fallback=True, allow_hot_pruning=True)
    result = solve(150, fallback=True, allow_hot_pruning=True)
    assert not result["rd_fallback"]
    assert result["layers"]["0"]["removed_protected_experts"] == 0
    costs = np.broadcast_to([0, 30, 42, 54], (1, 6, 4))
    q = np.broadcast_to([1.0, 0.0, 0.0, 0.0], costs.shape)
    with pytest.raises(ValueError, match="infeasible"):
        layer_allocate(
            selection(), q, costs, 0, 90, 0.9, True, 4, allow_hot_pruning=True
        )


def test_cli_rejects_hot_pruning_without_fallback_before_reading_inputs(monkeypatch):
    from scripts.deepseek_v41_rd_allocate import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "allocate",
            "--allow-hot-pruning",
            "--scores",
            "missing",
            "--errors",
            "missing",
            "--costs",
            "missing",
            "--output-dir",
            "missing",
        ],
    )
    with pytest.raises(ValueError, match="requires --rd-fallback"):
        main()


def test_cli_requires_explicit_prune_policy_without_preset(monkeypatch):
    from scripts.deepseek_v41_rd_allocate import main

    args = ["allocate", "--scores", "m", "--errors", "m", "--costs", "m"]
    monkeypatch.setattr(sys, "argv", [*args, "--output-dir", "m"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize(
    "override",
    [["--max-prune-fraction", "0.5"], ["--rd-fallback"], ["--allow-hot-pruning"]],
)
def test_cli_rejects_policy_overrides_with_preset(monkeypatch, capsys, override):
    from scripts.deepseek_v41_rd_allocate import main

    args = ["allocate", "--scores", "m", "--errors", "m", "--costs", "m"]
    preset = ["--preset", next(iter(PRESETS))]
    monkeypatch.setattr(sys, "argv", [*args, *preset, *override, "--output-dir", "m"])
    with pytest.raises(SystemExit):
        main()
    assert "fixes the pruning policy" in capsys.readouterr().err


def test_hot_slice_audit_counts_reap_and_empty_layers():
    result = {"layers": {"0": {"kept_expert_ids": [1]}, "1": {"kept_expert_ids": [0]}}}
    counts = np.array([[[9, 1], [0, 0]], [[2, 2], [3, 1]], [[1, 1], [1, 1]]])
    reap = np.array([[[1, 9], [0, 0]], [[1, 3], [2, 4]], [[1, 1], [1, 1]]])
    audit = pruning_audit(
        result,
        counts,
        reap,
        ["a", "b", "cold"],
        {"a": "hot", "b": "hot", "cold": "cold"},
    )
    assert set(audit["hot_slice_retention"]) == {"a", "b"}
    a = audit["hot_slice_retention"]["a"]
    assert a["route_mass_retained"] == 0.1
    assert a["reap_mass_retained"] == 0.9
    assert a["route_mass_retained_by_layer"] == [0.1, 1.0]
    assert audit["min_hot_route_mass_retained"] == 0.1
    assert audit["min_hot_reap_mass_retained"] == 0.5


def test_hot_sensitivity_is_max_over_slices_not_average():
    from scripts.deepseek_v41_contrastive import contrastive

    r = np.repeat(
        np.array(
            [
                [[1.0, 9.0, 4.0, 0.0]],
                [[1.0, 0.0, 4.0, 0.0]],
                [[1.0, 1.0, 1.0, 0.0]],
                [[1.0, 1.0, 1.0, 0.0]],
            ]
        )[:, None],
        2,
        axis=1,
    )
    sel = contrastive(
        r,
        np.where(r > 0, 8, 0),
        ["a", "b", "cold_en", "general_reference_en"],
        {
            "a": "hot",
            "b": "hot",
            "cold_en": "cold",
            "general_reference_en": "reference",
        },
    )
    assert sel["sensitivity"].tolist() == [[2.0, 18.0, 8.0, 0.0]]
    costs = np.broadcast_to([0, 30, 42, 54], (1, 4, 4))
    q = np.broadcast_to([1.0, 0.0, 0.0, 0.0], costs.shape)
    result = layer_allocate(sel, q, costs, 0, 60, 0.75, True, 1, allow_hot_pruning=True)
    assert result["layers"]["0"]["kept_expert_ids"] == [1, 3]


@pytest.fixture
def draft_source(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from rapid_mlx.models.deepseek_v41_native.dspark import _required_tensors
    from tests.test_deepseek_v41_dspark import _config
    from tests.test_deepseek_v41_quant_error import tiny_source

    mx.set_default_device(mx.cpu)
    source = tmp_path / "source"
    config = tiny_source(source)
    config.update(_config())
    config["text_config"]["num_experts_per_tok"] = 1
    # The strict DSpark loader requires release architecture metadata. Tensor
    # dimensions are deliberately tiny; this test executes only its MoE path.
    config["text_config"]["hidden_size"] = 5120
    raw = mx.load(str(source / "tiny.safetensors"))
    raw.pop("mtp.omitted.weight")
    bases = {
        n.rsplit(".", 1)[0]
        for n in _required_tensors(config)
        if n.endswith((".weight", ".scales", ".biases"))
    }
    for base in sorted(bases):
        if ".ffn.experts." in base:
            proj = base.rsplit(".", 1)[-1]
            for suffix in ("weight", "scale"):
                raw[base + "." + suffix] = raw[
                    f"layers.0.ffn.experts.0.{proj}.{suffix}"
                ]
        elif base.endswith("ffn.gate"):
            raw[base + ".weight"] = mx.zeros((128, 64), mx.bfloat16)
        elif "norm" in base:
            raw[base + ".weight"] = mx.ones((64,), mx.bfloat16)
        else:
            raw[base + ".weight"] = mx.full((64, 64), 0.01, mx.bfloat16)
    for name in _required_tensors(config):
        if not name.endswith((".weight", ".scales", ".biases")):
            raw[name] = mx.zeros(
                (128 if name.endswith("ffn.gate.bias") else 64,), mx.bfloat16
            )
    mx.save_safetensors(str(source / "tiny.safetensors"), raw)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "tiny.safetensors" for n in raw}})
    )
    (source / "config.json").write_text(json.dumps(config))
    return source, config


def draft_allocation(bits=None):
    a = {"keep_mtp": True, "layers": {"0": {"kept_expert_ids": [0, 1], "bits": 2}}}
    if bits is not None:
        a["mtp_expert_bits"] = bits
    return a


@pytest.mark.requires_mlx
@pytest.mark.parametrize("bits", [4, 8])
def test_allocator_cli_prices_selected_draft_even_with_old_inventory(
    draft_source, tmp_path, monkeypatch, bits
):
    from scripts import deepseek_v41_expert_quant_error as quant
    from scripts import deepseek_v41_rd_allocate as allocator

    source, config = draft_source
    costfile = tmp_path / "costs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inventory",
            "--source",
            str(source),
            "--inventory",
            str(costfile),
            "--inventory-only",
            "--release-policy",
            "--model-revision",
            "r",
        ],
    )
    quant.main()
    cost = json.loads(costfile.read_text())
    draft_costs = mtp_byte_costs(CheckpointIndex(source), config)
    assert cost["mtp_bytes_by_expert_bits"] == draft_costs
    assert cost["mtp_bytes"] == draft_costs["4"]
    # Emulate the read-only phase-3 inventory, which has only an 8-bit total.
    cost["mtp_bytes"] = draft_costs["8"]
    cost.pop("mtp_bytes_by_expert_bits")
    cost.pop("mtp_expert_bits")
    costfile.write_text(json.dumps(cost))
    roles = {"chat": "hot", "cold_en": "cold", "general_reference_en": "reference"}
    r = np.repeat(
        np.array([[[8.0, 0.01]], [[0.1, 8.0]], [[1.0, 0.1]]])[:, None], 2, axis=1
    )
    scores, errors, rolefile = (
        tmp_path / n for n in ("scores.npz", "errors.npz", "roles.json")
    )
    np.savez(
        scores,
        reap_halves=r,
        counts_halves_by_slice=np.full(r.shape, 8),
        split_keep_overlap=np.ones((3, 3)),
        metadata=json.dumps(
            {
                "model_revision": "r",
                "routing_topk": 1,
                "slice_names": list(roles),
                "config_sha256": file_hash(source / "config.json"),
            }
        ),
    )
    weight = np.array(cost["expert_bytes"])
    weight[..., 1:] -= cost["router_row_bytes_by_layer"]["0"]
    np.savez(
        errors,
        q=np.broadcast_to([1.0, 0.3, 0.1, 0.01], weight.shape),
        projection_q=np.broadcast_to([[0.1, 0.03, 0.01]] * 3, (1, 2, 3, 3)),
        expert_weight_bytes=weight,
        completed=np.ones((1, 2), bool),
        metadata=json.dumps(
            {
                "model_revision": "r",
                "group_size": 64,
                "bits_order": [0, 2, 3, 4],
                "config_sha256": file_hash(source / "config.json"),
                "index_sha256": file_hash(source / "model.safetensors.index.json"),
            }
        ),
    )
    rolefile.write_text(json.dumps(roles))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "allocate",
            "--source",
            str(source),
            "--scores",
            str(scores),
            "--errors",
            str(errors),
            "--costs",
            str(costfile),
            "--slice-roles",
            str(rolefile),
            "--preset",
            "T256",
            "--granularity",
            "projection",
            "--topk",
            "1",
            "--mtp-expert-bits",
            str(bits),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    allocator.main()
    result = json.loads((tmp_path / "out" / "allocation-200GB.json").read_text())
    assert result["mtp_bytes"] == draft_costs[str(bits)]
    assert result["mtp_expert_bits"] == bits
    assert (
        result["fixed_non_expert_bytes"]
        == cost["fixed_non_expert_bytes"] + draft_costs[str(bits)]
    )
    assert (
        result["resident_tensor_bytes"]
        == result["byte_breakdown"]["resident_tensor_bytes"]
    )
    assert result["byte_breakdown"]["components"]["mtp"] == draft_costs[str(bits)]
    assert "removed_protected_experts" in result["layers"]["0"]
    assert result["min_hot_route_mass_retained"] == 1.0

    from scripts import deepseek_v41_rd_build as builder

    original_plan = builder.plan

    def inconsistent_plan(*args, **kwargs):
        entries, metadata, report = original_plan(*args, **kwargs)
        report["resident_tensor_bytes"] += 1
        return entries, metadata, report

    monkeypatch.setattr(builder, "plan", inconsistent_plan)
    with pytest.raises(ValueError, match="conversion policy"):
        allocator.main()
    monkeypatch.setattr(builder, "plan", original_plan)
    config_path = source / "config.json"
    config_bytes = config_path.read_bytes()

    def changed_during_plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        config_path.write_bytes(config_bytes + b" ")
        return result

    monkeypatch.setattr(builder, "plan", changed_during_plan)
    with pytest.raises(ValueError, match="source checkpoint config"):
        allocator.main()
    assert (
        json.loads((tmp_path / "out" / "allocation-200GB.json").read_text()) == result
    )
    monkeypatch.setattr(builder, "plan", original_plan)
    config_path.write_bytes(config_bytes)
    config_path.write_bytes(config_bytes + b" ")
    with pytest.raises(ValueError, match="source checkpoint config"):
        allocator.main()
    config_path.write_bytes(config_bytes)
    index_path = source / "model.safetensors.index.json"
    index_path.write_bytes(index_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source checkpoint index"):
        allocator.main()
