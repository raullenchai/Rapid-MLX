from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.deepseek_v41_rd_allocate import pruning_audit
from scripts.deepseek_v41_rd_build import build, mtp_byte_costs, plan
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
        ["a", "b", "cold", "ref"],
        {"a": "hot", "b": "hot", "cold": "cold", "ref": "reference"},
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
def test_draft_build_accounting_contract_strict_load_and_dspark_moe(
    draft_source, tmp_path, bits
):
    import mlx.core as mx
    import mlx.nn as nn

    from rapid_mlx.models.deepseek_v41_native.dspark import DSpark, DSparkWeights
    from rapid_mlx.models.deepseek_v41_native.load import quant_predicate

    source, config = draft_source
    index = CheckpointIndex(source)
    a = draft_allocation(None if bits == 4 else bits)
    entries, modules, report = plan(index, config, a)
    base = "mtp.0.ffn.experts.0.w1"
    assert modules[base] == {"bits": bits, "group_size": 64}
    expert = next(e for e in entries if e.name == base + ".weight")
    assert expert.dtype == "U32" and expert.shape == (64, 64 * bits // 32)
    assert all(
        e.bits == 8
        for e in entries
        if ".shared_experts." in e.name and e.component == "mtp"
    )
    assert all(
        e.dtype == "BF16"
        for e in entries
        if "markov_head" in e.name or e.name == "mtp.2.norm.weight"
    )
    costs = mtp_byte_costs(index, config)
    assert report["components"]["mtp"] == costs[str(bits)]
    assert costs["8"] - costs["4"] == 3 * 128 * 3 * 64 * 64 // 2
    dry = build(source, tmp_path / "dry", tmp_path / "unused", a, dry_run=True)
    assert dry["resident_tensor_bytes"] == report["resident_tensor_bytes"]
    assert not (tmp_path / "dry").exists()
    build(source, tmp_path / "built", tmp_path / "shared", a)
    emitted = json.loads((tmp_path / "built" / "config.json").read_text())
    assert emitted["rapid_quantization"]["allocation"]["mtp_expert_bits"] == bits
    weights = DSparkWeights(tmp_path / "built" / "dspark")
    assert weights.quant_config(base) == {
        "bits": bits,
        "group_size": 64,
        "mode": "affine",
    }
    # Both runtime readers consume the emitted per-module contract.
    assert (
        quant_predicate(64, 8, None, weights.q["modules"])(base, nn.Linear(64, 64))
        == modules[base]
    )
    draft = DSpark(SimpleNamespace(w=weights, c=weights.config), pin_weights=False)
    x = mx.full((1, 64), 0.01, mx.bfloat16)
    actual = draft.adapter.moe("mtp.0.ffn", x)
    assert actual.shape == (1, 64) and mx.isfinite(actual).all().item()

    # Decode the emitted affine matrices independently of DSpark's matmul path.
    def linear(base, value):
        w = mx.dequantize(
            *(weights.read(base + "." + s) for s in ("weight", "scales", "biases")),
            **weights.quant_config(base),
        )
        return (value @ w.T).astype(value.dtype)

    b = "mtp.0.ffn.experts.0"
    gate = linear(b + ".w1", x).astype(mx.float32)
    up = linear(b + ".w3", x).astype(mx.float32)
    gate, up = mx.minimum(gate, 10), mx.clip(up, -10, 10)
    expected = linear(b + ".w2", (gate * mx.sigmoid(gate) * up).astype(x.dtype))
    np.testing.assert_allclose(
        np.array(draft.adapter.expert(b, x).astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.02,
        atol=0.02,
    )


@pytest.mark.parametrize("bits", [2, 3, 16, True, 4.0])
def test_invalid_draft_width_rejected(bits):
    with pytest.raises(ValueError, match="mtp_expert_bits"):
        plan(None, {}, draft_allocation(bits))


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
    roles = {"chat": "hot", "cold": "cold", "reference": "reference"}
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
            {"model_revision": "r", "routing_topk": 1, "slice_names": list(roles)}
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
            {"model_revision": "r", "group_size": 64, "bits_order": [0, 2, 3, 4]}
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


@pytest.mark.requires_mlx
def test_all_two_bit_no_draft_payload_unchanged(tmp_path):
    import mlx.core as mx

    from tests.test_deepseek_v41_quant_error import tiny_source

    mx.set_default_device(mx.cpu)
    source = tmp_path / "source"
    c = tiny_source(source)
    c["text_config"]["num_experts_per_tok"] = 1
    (source / "config.json").write_text(json.dumps(c))
    a = draft_allocation()
    a["keep_mtp"] = False
    entries, modules, report = plan(CheckpointIndex(source), c, a)
    for bits in (4, 8):
        assert plan(CheckpointIndex(source), c, {**a, "mtp_expert_bits": bits}) == (
            entries,
            modules,
            report,
        )
    build(source, tmp_path / "built", tmp_path / "shared", a)
    # Captured from the phase-3 implementation before phase-4 changes.
    expected = {
        "model-layer-0.safetensors": "1f343858474fa8398136e0355e4c84aebde527d91fc54755fc989a31d3b335a6",
        "model-top.safetensors": "4d9f379e576eeae43f3dd0f3401b175399cad5fe9eeef73adca6c85526bdc405",
        "model-engram-0.safetensors": "1976bd5292787b19e9704f97fcc8412201be592696739bd196db3866991deb92",
    }
    for name, digest in expected.items():
        assert (
            hashlib.sha256((tmp_path / "built" / name).read_bytes()).hexdigest()
            == digest
        )
