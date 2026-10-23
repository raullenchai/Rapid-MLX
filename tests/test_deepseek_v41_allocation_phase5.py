from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest

from scripts import deepseek_v41_rd_build as builder
from scripts.deepseek_v41_expert_quant_error import inventory
from scripts.deepseek_v41_rd_allocate import inventory_digest
from scripts.deepseek_v41_variant_alloc import variant_allocation
from scripts.repack_deepseek_v41_native import CheckpointIndex
from tests.test_deepseek_v41_quant_error import tiny_source


def test_variant_accounting_matches_builder_and_cli(source, tmp_path, monkeypatch):
    # inventory() follows convert.py's quantization rules, which import MLX.
    pytest.importorskip("mlx.core")
    from scripts.deepseek_v41_variant_alloc import main

    config = json.loads((source / "config.json").read_text())
    index = CheckpointIndex(source)
    cost = inventory(index, config)
    base = allocation()
    entries, _, report = builder.plan(index, config, base)
    cost["fixed_non_expert_bytes"] = sum(
        v
        for k, v in report["components"].items()
        if k not in ("router", "routed_experts", "engram_ssd", "mtp")
    )
    base.update(
        budget_bytes=report["resident_tensor_bytes"],
        resident_tensor_bytes=report["resident_tensor_bytes"],
        fixed_non_expert_bytes=cost["fixed_non_expert_bytes"]
        + report["components"]["mtp"],
        mtp_bytes=report["components"]["mtp"],
        byte_breakdown=builder.layout_report(entries, report),
        predicted_distortion=42,
        duality_gap=1,
    )
    base["provenance"] = {"costs_sha256": inventory_digest(cost)}
    before = copy.deepcopy(base)
    result = variant_allocation(base, cost, ["0=3/4/2"])
    _, _, actual = builder.plan(index, config, result)
    assert base == before
    assert result["resident_tensor_bytes"] == actual["resident_tensor_bytes"]
    assert result["byte_breakdown"]["components"] == actual["components"]
    assert (
        result["byte_breakdown"]["total_payload_bytes"] == actual["total_payload_bytes"]
    )
    assert result["budget_bytes"] is None
    assert result["variant"]["resident_byte_delta"] > 0
    assert "predicted_distortion" not in result and "duality_gap" not in result
    assert "safetensors_container_bytes" not in result["byte_breakdown"]
    basefile, costfile, output = [
        tmp_path / n for n in ("base.json", "cost.json", "out.json")
    ]
    basefile.write_text(json.dumps(base))
    costfile.write_text(json.dumps(cost))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "variant",
            "--base",
            str(basefile),
            "--costs",
            str(costfile),
            "--set-bits",
            "0=3/4/2",
            "--output",
            str(output),
        ],
    )
    main()
    assert (
        json.loads(output.read_text())["resident_tensor_bytes"]
        == actual["resident_tensor_bytes"]
    )


@pytest.mark.parametrize("override", ["bad", "0=1/3/3", "0=3/3", "9=3/3/3", "-1=3/3/3"])
def test_variant_invalid_overrides(source, override):
    pytest.importorskip("mlx.core")
    config = json.loads((source / "config.json").read_text())
    cost = inventory(CheckpointIndex(source), config)
    base = allocation()
    base.update(
        keep_mtp=False,
        fixed_non_expert_bytes=cost["fixed_non_expert_bytes"],
        resident_tensor_bytes=cost["fixed_non_expert_bytes"]
        + sum(row[1] for row in cost["expert_bytes"][0]),
    )
    base["provenance"] = {"costs_sha256": inventory_digest(cost)}
    with pytest.raises(ValueError):
        variant_allocation(base, cost, [override])


def test_variant_multiple_layers_repeated_overrides_and_pruning():
    base = {
        "layers": {
            "0": {"kept_expert_ids": [0, 1], "bits": 2},
            "1": {"kept_expert_ids": [1], "bits": 2},
        },
        "fixed_non_expert_bytes": 100,
        "resident_tensor_bytes": 190,
        "budget_bytes": 190,
    }
    cost = {
        "fixed_non_expert_bytes": 100,
        "expert_bytes": np.broadcast_to([0, 30, 42, 54], (2, 2, 4)).tolist(),
    }
    base["provenance"] = {"costs_sha256": inventory_digest(cost)}
    result = variant_allocation(base, cost, ["0,1=3/3/3", "1=4/3/2"])
    assert result["resident_tensor_bytes"] == 226
    assert result["variant"]["resident_byte_delta"] == 36
    assert result["layers"]["1"]["kept_expert_ids"] == [1]
    assert result["layers"]["0"]["routed_and_router_bytes"] == 84
    assert result["layers"]["1"]["routed_and_router_bytes"] == 42
    assert result["layers"]["1"]["projection_bits"]["gate_proj"] == 4
    assert result["budget_bytes"] is None


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    config = tiny_source(path)
    config["text_config"]["num_experts_per_tok"] = 1
    (path / "config.json").write_text(json.dumps(config))
    return path


def allocation(ids=(0, 1), bits=2):
    return {
        "keep_mtp": True,
        "layers": {"0": {"kept_expert_ids": list(ids), "bits": bits}},
    }
