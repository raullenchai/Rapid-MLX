from __future__ import annotations

import copy
import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import deepseek_v41_rd_build as builder
from scripts.deepseek_v41_expert_quant_error import inventory
from scripts.deepseek_v41_variant_alloc import variant_allocation
from scripts.evaluate_deepseek_v41_rd import compare
from scripts.repack_deepseek_v41_native import CheckpointIndex
from tests.test_deepseek_v41_quant_error import tiny_source
from tests.test_deepseek_v41_rd import write_cache


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


@pytest.fixture
def reuse_build(source, tmp_path):
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    original = tmp_path / "original"
    builder.build(source, original, tmp_path / "shared", allocation())
    return original


@pytest.mark.requires_mlx
@pytest.mark.parametrize("bits", [2, 3])
def test_reuse_matches_fresh_bytes_and_inodes(source, reuse_build, tmp_path, bits):
    fresh, reused = tmp_path / "fresh", tmp_path / "reused"
    a = allocation(bits=bits)
    builder.build(source, fresh, tmp_path / "shared", copy.deepcopy(a))
    dry = tmp_path / "nonexistent" / "dry"
    report = builder.build(
        source, dry, tmp_path / "unused", copy.deepcopy(a), True, reuse_build
    )
    assert not dry.parent.exists() and not (tmp_path / "unused").exists()
    expected = ["model-layer-0.safetensors"] if bits == 3 else []
    assert report["reuse"]["rewritten"] == expected
    builder.build(source, reused, tmp_path / "shared", a, reuse_from=reuse_build)
    for path in fresh.rglob("*"):
        if path.is_file():
            relative = path.relative_to(fresh)
            assert (reused / relative).read_bytes() == path.read_bytes(), relative
    for name in report["reuse"]["reused"]:
        assert (reused / name).stat().st_ino == (reuse_build / name).stat().st_ino
    for name in expected:
        assert (reused / name).stat().st_ino != (reuse_build / name).stat().st_ino
    provenance = json.loads((reused / "reuse-provenance.json").read_text())
    assert provenance == report["reuse"]
    assert len(provenance["allocation_sha256"]) == 64
    assert "dspark/model-mtp.safetensors" in provenance["reused"]
    assert "model-engram-0-weight.safetensors" in provenance["reused"]
    assert "model-engram-0-affine.safetensors" in provenance["reused"]


@pytest.mark.requires_mlx
@pytest.mark.parametrize("corruption", ["header", "size", "length", "missing"])
def test_reuse_verifies_header_and_size(source, reuse_build, tmp_path, corruption):
    path = reuse_build / "model-top.safetensors"
    if corruption == "missing":
        path.unlink()
    else:
        data = path.read_bytes()
        if corruption == "size":
            data = data[:-1]
        elif corruption == "length":
            data = struct.pack("<Q", 2**63) + data[8:]
        else:
            data = data.replace(b'"head.weight"', b'"dead.weight"')
        path.write_bytes(data)
    dry = builder.build(
        source, tmp_path / "dry", tmp_path / "unused", allocation(), True, reuse_build
    )
    assert dry["reuse"]["rewritten"] == [path.name]


@pytest.mark.requires_mlx
def test_equal_header_expert_swap_is_rewritten(source, tmp_path):
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    old = tmp_path / "old"
    builder.build(source, old, tmp_path / "shared", allocation((0,)))
    report = builder.build(
        source, tmp_path / "new", tmp_path / "shared", allocation((1,)), True, old
    )
    assert report["reuse"]["rewritten"] == ["model-layer-0.safetensors"]


@pytest.mark.requires_mlx
def test_reuse_source_and_filesystem_guards(source, reuse_build, tmp_path, monkeypatch):
    monkeypatch.setattr(
        builder,
        "existing_parent",
        lambda _: SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=-1)),
    )
    with pytest.raises(ValueError, match="same filesystem"):
        builder.build(
            source, tmp_path / "dry", tmp_path / "s", allocation(), True, reuse_build
        )
    config = json.loads((source / "config.json").read_text())
    config["changed"] = True
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source identity"):
        builder.build(
            source, tmp_path / "dry", tmp_path / "s", allocation(), True, reuse_build
        )
    assert not (tmp_path / "dry").exists()


def test_variant_accounting_matches_builder_and_cli(source, tmp_path, monkeypatch):
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
    config = json.loads((source / "config.json").read_text())
    cost = inventory(CheckpointIndex(source), config)
    with pytest.raises(ValueError):
        variant_allocation(allocation(), cost, [override])


@pytest.fixture
def caches(tmp_path):
    reference, candidate = tmp_path / "r", tmp_path / "c"
    logits = [[1, 2], [3, 4]]
    write_cache(reference, logits, "unpruned_reference")
    write_cache(candidate, logits, "packed_quantized")
    path = reference / "metadata.json"
    metadata = json.loads(path.read_text())
    # A slice absent from candidate need not have payload files read at all.
    metadata["slices"]["extra"] = metadata["slices"]["domain"]
    path.write_text(json.dumps(metadata))
    return reference, candidate


def test_subset_compare_default_unchanged_and_cli(caches, tmp_path, monkeypatch):
    from scripts.evaluate_deepseek_v41_rd import main

    r, c = caches
    with pytest.raises(ValueError, match="provenance/slice"):
        compare(r, c)
    result = compare(r, c, subset=True)
    assert result["compared_slices"] == ["domain"]
    assert result["mean_domain_kl_nats"] == 0
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "compare",
            "--reference",
            str(r),
            "--candidate",
            str(c),
            "--subset",
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize(
    "bad",
    ["empty", "unknown", "revision", "tokenizer", "corpus", "logits", "tokens", "mass"],
)
def test_subset_preserves_all_guards(caches, bad):
    r, c = caches
    path = c / "metadata.json"
    metadata = json.loads(path.read_text())
    if bad == "empty":
        metadata["slices"] = {}
    elif bad == "unknown":
        metadata["slices"]["unknown"] = metadata["slices"].pop("domain")
    elif bad in ("revision", "tokenizer"):
        metadata["model_revision" if bad == "revision" else "tokenizer_sha256"] = (
            "changed"
        )
    elif bad == "corpus":
        metadata["slices"]["domain"]["corpus_sha256"] = "changed"
    else:
        np.save(c / f"domain-{bad}.npy", np.zeros((2, 2)))
        if bad == "mass":
            np.save(c / "domain-mass.npy", np.full((2, 2), np.nan))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        compare(r, c, subset=True)


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
    result = variant_allocation(base, cost, ["0,1=3/3/3", "1=4/3/2"])
    assert result["resident_tensor_bytes"] == 226
    assert result["variant"]["resident_byte_delta"] == 36
    assert result["layers"]["1"]["kept_expert_ids"] == [1]
    assert result["layers"]["0"]["routed_and_router_bytes"] == 84
    assert result["layers"]["1"]["routed_and_router_bytes"] == 42
    assert result["layers"]["1"]["projection_bits"]["gate_proj"] == 4
    assert result["budget_bytes"] is None


@pytest.mark.requires_mlx
def test_changed_draft_width_rewrites_only_sidecar(source, tmp_path):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    tensors = mx.load(str(source / "tiny.safetensors"))
    for suffix in ("weight", "scale"):
        tensors[f"mtp.0.ffn.experts.0.w1.{suffix}"] = tensors[
            f"layers.0.ffn.experts.0.w1.{suffix}"
        ]
    mx.save_safetensors(str(source / "tiny.safetensors"), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "tiny.safetensors" for n in tensors}})
    )
    original, fresh, reused = [tmp_path / n for n in ("original", "fresh", "reused")]
    builder.build(source, original, tmp_path / "shared", allocation())
    a = {**allocation(), "mtp_expert_bits": 8}
    builder.build(source, fresh, tmp_path / "shared", copy.deepcopy(a))
    result = builder.build(source, reused, tmp_path / "shared", a, reuse_from=original)
    assert result["reuse"]["rewritten"] == ["dspark/model-mtp.safetensors"]
    for path in fresh.rglob("*"):
        if path.is_file():
            assert (reused / path.relative_to(fresh)).read_bytes() == path.read_bytes()
