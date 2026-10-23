from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.deepseek_v41_expert_quant_error import (
    inventory,
    load_expert,
    main,
    quant_error,
)
from scripts.deepseek_v41_rd_common import affine_bytes
from scripts.repack_deepseek_v41_native import CheckpointIndex

pytestmark = pytest.mark.requires_mlx


def tiny_source(root: Path, *, include_f32_aux=False):
    """Actual safetensors bytes, including source FP4 and exponent-only scales."""
    root.mkdir()
    tensors = {}
    rng = np.random.default_rng(8)
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            base = f"layers.0.ffn.experts.{expert}.{projection}"
            tensors[base + ".weight"] = (
                "U8",
                rng.integers(0, 256, (64, 32), dtype=np.uint8),
            )
            tensors[base + ".scale"] = ("U8", np.full((64, 2), 124, np.uint8))
    for projection in ("w1", "w2", "w3"):
        tensors[f"layers.0.ffn.shared_experts.{projection}.weight"] = (
            "BF16",
            np.zeros((64, 64), np.uint16),
        )
    tensors["layers.0.ffn.gate.weight"] = ("BF16", np.zeros((2, 64), np.uint16))
    tensors["layers.0.ffn.gate.bias"] = ("F16", np.zeros(2, np.float16))
    tensors["layers.0.ffn.gate.bias_vl"] = ("F16", np.zeros(2, np.float16))
    tensors["head.weight"] = ("BF16", np.zeros((64, 64), np.uint16))
    tensors["norm.weight"] = ("F32", np.zeros(64, np.float32))
    tensors["layers.0.hc_attn_base"] = ("F16", np.zeros(128, np.float16))
    if include_f32_aux:
        tensors["layers.0.hc_attn_fn"] = ("F32", np.zeros(4, np.float32))
        tensors["layers.0.attn.attn_sink"] = ("F32", np.zeros(2, np.float32))
    tensors["layers.0.engram.embed.weight"] = ("U8", np.zeros((2, 256), np.uint8))
    tensors["layers.0.engram.embed.scale"] = ("U8", np.full((2, 8), 127, np.uint8))
    tensors["mtp.omitted.weight"] = ("F32", np.ones(3, np.float32))
    tensors["vision.omitted.weight"] = ("F32", np.ones(3, np.float32))
    offset = 0
    header = {}
    for name, (dtype, array) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with (root / "tiny.safetensors").open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for _, array in tensors.values():
            output.write(array.tobytes())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "tiny.safetensors" for k in tensors}})
    )
    config = {
        "text_config": {
            "num_hidden_layers": 1,
            "n_routed_experts": 2,
            "hidden_size": 64,
            "swiglu_limit": 10.0,
        },
        "quantization_config": {"expert_dtype": "fp4"},
    }
    (root / "config.json").write_text(json.dumps(config))
    return config


def test_header_inventory_counts_fixed_shared_router_and_excludes_engram(tmp_path):
    config = tiny_source(tmp_path / "source")
    index = CheckpointIndex(tmp_path / "source")
    result = inventory(index, config)
    assert (
        result["fixed_non_expert_bytes"]
        == 4 * affine_bytes((64, 64), 4) + 64 * 4 + 128 * 4
    )
    assert result["excluded_engram_bytes"] == 528
    assert result["router_row_bytes_by_layer"] == {0: 136}
    expected = [0] + [3 * affine_bytes((64, 64), b) + 136 for b in (2, 3, 4)]
    assert result["expert_bytes"] == [[expected, expected]]


def test_plan_preserves_released_aux_dtype_and_f32_router_biases(tmp_path):
    from scripts import deepseek_v41_rd_build as rd_build

    source = tmp_path / "source"
    config = tiny_source(source, include_f32_aux=True)
    config["text_config"]["num_experts_per_tok"] = 2
    index = CheckpointIndex(source)
    for reference, bits, aux_dtype in ((False, 4, "BF16"), (True, 16, "F32")):
        allocation = {
            "reference": reference,
            "layers": {"0": {"kept_expert_ids": [0, 1], "bits": bits}},
        }
        entries, *_ = rd_build.plan(index, config, allocation)
        dtypes = {entry.name: entry.dtype for entry in entries}
        assert dtypes["layers.0.hc_attn_fn"] == aux_dtype
        assert dtypes["layers.0.attn.attn_sink"] == aux_dtype
        assert dtypes["layers.0.ffn.gate.bias"] == "F32"
        assert dtypes["layers.0.ffn.gate.bias_vl"] == "F32"


@pytest.mark.parametrize("target", ["plan", "inventory"])
@pytest.mark.parametrize(
    "dense_field", ["first_k_dense_replace", "first_k_dense", None]
)
def test_dense_ffn_layers_fail_closed(tmp_path, target, dense_field):
    if target == "inventory":
        pytest.importorskip("mlx.core")

    from scripts import deepseek_v41_rd_build as rd_build

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 2
    index = CheckpointIndex(source)
    if dense_field is not None:
        config["text_config"][dense_field] = 1
    else:
        index.tensors = {
            name: tensor
            for name, tensor in index.tensors.items()
            if ".ffn.experts." not in name
        }
    with pytest.raises(ValueError, match="dense FFN layers are not supported") as exc:
        if target == "plan":
            rd_build.plan(
                index,
                config,
                {"layers": {"0": {"kept_expert_ids": [0, 1], "bits": 4}}},
            )
        else:
            inventory(index, config)
    if dense_field is None:
        assert "layers without routed experts: [0]" in str(exc.value)


def test_tensor_stream_matches_native_fp4_decoder(tmp_path):
    import mlx.core as mx

    from rapid_mlx.models.deepseek_v41_native.dequant import dequant_fp4
    from scripts.deepseek_v41_expert_quant_error import read_tensor

    tiny_source(tmp_path / "source")
    index = CheckpointIndex(tmp_path / "source")
    with mx.stream(mx.cpu):
        weights = load_expert(index, 0, 0)
        base = "layers.0.ffn.experts.0.w1"
        expected = dequant_fp4(
            read_tensor(index.tensors[base + ".weight"]),
            read_tensor(index.tensors[base + ".scale"]),
            mx.bfloat16,
        )
        np.testing.assert_array_equal(
            np.array(weights[0].astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
        )
        assert all(w.shape == (64, 64) for w in weights)


def test_zero_energy_probes_fail_closed():
    import mlx.core as mx

    with mx.stream(mx.cpu), pytest.raises(ValueError, match="reference energy"):
        quant_error([mx.zeros((64, 64)) for _ in range(3)], mx.ones((2, 64)))


def test_partial_quantization_resume_and_provenance_guard(tmp_path, monkeypatch):
    tiny_source(tmp_path / "source")
    args = [
        "quant-error",
        "--source",
        str(tmp_path / "source"),
        "--inventory",
        str(tmp_path / "costs.json"),
        "--output",
        str(tmp_path / "errors.npz"),
        "--probes",
        "2",
        "--stop",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", args)
    main()
    with np.load(tmp_path / "errors.npz") as data:
        first = data["q"][0, 0].copy()
        assert data["completed"].tolist() == [[True, False]]
    args[-1] = "2"
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    main()
    with np.load(tmp_path / "errors.npz") as data:
        np.testing.assert_array_equal(data["q"][0, 0], first)
        assert data["completed"].all()
    (tmp_path / "costs.json").write_text("unchanged\n")
    monkeypatch.setattr(sys, "argv", args + ["--resume", "--seed", "9"])
    with pytest.raises(ValueError, match="provenance"):
        main()
    assert (tmp_path / "costs.json").read_text() == "unchanged\n"

    shard = tmp_path / "source" / "tiny.safetensors"
    payload = shard.read_bytes()
    shard.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    monkeypatch.setattr(sys, "argv", args + ["--resume"])
    with pytest.raises(ValueError, match="provenance"):
        main()


def test_empty_expert_range_does_not_write_output_or_inventory(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")

    tiny_source(tmp_path / "source")
    output = tmp_path / "errors.npz"
    inventory_path = tmp_path / "costs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quant-error",
            "--source",
            str(tmp_path / "source"),
            "--output",
            str(output),
            "--inventory",
            str(inventory_path),
            "--start",
            "0",
            "--stop",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="^invalid expert range$"):
        main()
    assert not output.exists()
    assert not inventory_path.exists()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_write_group_failure_removes_partial_and_retry_succeeds(
    tmp_path, monkeypatch, error_type
):
    mx = pytest.importorskip("mlx.core")

    from scripts import deepseek_v41_rd_build as rd_build

    source = tmp_path / "source"
    config = tiny_source(source)
    index = CheckpointIndex(source)
    path = tmp_path / "group.safetensors"
    entries = [
        rd_build.Entry(
            "norm.weight",
            (128,),
            "F32",
            ("norm.weight", "norm.weight"),
            "fixed",
        )
    ]
    transformed = rd_build.transformed
    calls = 0

    def fail_mid_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("injected failure")
        return transformed(*args, **kwargs)

    with mx.stream(mx.cpu):
        monkeypatch.setattr(rd_build, "transformed", fail_mid_write)
        with pytest.raises(error_type, match="injected failure"):
            rd_build.write_group(path, entries, index, config)
        assert calls == 2
        assert not list(path.parent.glob(path.name + ".partial-*"))
        assert not path.exists()

        monkeypatch.setattr(rd_build, "transformed", transformed)
        rd_build.write_group(path, entries, index, config)
    assert rd_build.matching_file(path, entries)
    assert not list(path.parent.glob(path.name + ".partial-*"))


def test_build_failure_removes_staging_and_retry_succeeds(tmp_path, monkeypatch):
    from scripts import deepseek_v41_rd_build as rd_build

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 2
    (source / "config.json").write_text(json.dumps(config))
    destination = tmp_path / "build"
    allocation = {
        "layers": {"0": {"kept_expert_ids": [0, 1], "bits": 4}},
    }
    copy_metadata_files = rd_build._copy_metadata_files

    def simulate_write_group(path, entries, *_args, **_kwargs):
        header = rd_build.header(entries)
        with path.open("wb") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            output.write(bytes(sum(entry.nbytes for entry in entries)))

    def fail_copy(*_args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(rd_build, "write_group", simulate_write_group)
    monkeypatch.setattr(rd_build, "_copy_metadata_files", fail_copy)
    with pytest.raises(RuntimeError, match="injected failure"):
        rd_build.build(source, destination, tmp_path / "shared", allocation)
    assert not destination.exists()
    assert not list(tmp_path.glob("build.partial-*"))

    monkeypatch.setattr(rd_build, "_copy_metadata_files", copy_metadata_files)
    rd_build.build(source, destination, tmp_path / "shared", allocation)
    assert destination.is_dir()
    assert (destination / "config.json").is_file()
    assert not list(tmp_path.glob("build.partial-*"))


@pytest.mark.parametrize(
    "existing", ["race", "payload mismatch", "size mismatch", "header mismatch"]
)
def test_build_reuses_concurrent_shared_group_or_rejects_mismatch(
    tmp_path, monkeypatch, existing
):
    from types import ModuleType

    # Only exercise file publication; tensor payloads are supplied below.
    mlx = ModuleType("mlx")
    mlx.core = ModuleType("mlx.core")
    mlx.core.clear_cache = lambda: None
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)

    from scripts import deepseek_v41_rd_build as rd_build

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 2
    (source / "config.json").write_text(json.dumps(config))
    destination = tmp_path / "build"
    shared = tmp_path / "shared"
    allocation = {"layers": {"0": {"kept_expert_ids": [0, 1], "bits": 4}}}
    monkeypatch.setattr(
        rd_build,
        "transformed",
        lambda _index, _config, entry, *_: bytes(entry.nbytes // len(entry.sources)),
    )
    link = rd_build.os.link
    winners = {}
    write_group = rd_build.write_group

    def preexisting_shared_group(path, entries, *args, **kwargs):
        if kwargs.get("publish_exclusive") and existing not in (
            "race",
            "payload mismatch",
        ):
            raw = rd_build.header(entries)
            data = (
                struct.pack("<Q", len(raw))
                + raw
                + bytes(sum(entry.nbytes for entry in entries))
            )
            if existing == "size mismatch":
                data = data[:-1]
            else:
                data = data[:8] + b"!" + data[9:]
            path.write_bytes(data)
        return write_group(path, entries, *args, **kwargs)

    monkeypatch.setattr(rd_build, "write_group", preexisting_shared_group)

    def competing_publish(temporary, path):
        if path.parent.parent == shared and existing in ("race", "payload mismatch"):
            # Our complete, fsynced temporary exists before the other builder wins.
            assert ".partial-" in temporary.name
            data = temporary.read_bytes()
            if existing == "payload mismatch":
                data = data[:-1] + b"\x01"
            path.write_bytes(data)
            winners[path.name] = (path.stat().st_ino, data)
        return link(temporary, path)

    monkeypatch.setattr(rd_build.os, "link", competing_publish)
    if existing == "race":
        rd_build.build(source, destination, shared, allocation)
        assert winners
        for name, (inode, data) in winners.items():
            assert (destination / name).stat().st_ino != inode
            assert (destination / name).read_bytes() == data
    else:
        with pytest.raises(ValueError, match=f"shared Engram (file )?{existing}"):
            rd_build.build(source, destination, shared, allocation)
        assert not destination.exists()
    assert not list(shared.rglob("*.partial-*"))
    assert not list(tmp_path.glob("build.partial-*"))


def test_shared_identity_distinguishes_tensor_offsets(tmp_path, monkeypatch):
    from scripts import deepseek_v41_rd_build as rd_build

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 2
    (source / "config.json").write_text(json.dumps(config))
    allocation = {"layers": {"0": {"kept_expert_ids": [0, 1], "bits": 4}}}

    def simulate_write_group(path, entries, *_args, **_kwargs):
        raw = rd_build.header(entries)
        path.write_bytes(
            struct.pack("<Q", len(raw))
            + raw
            + bytes(sum(entry.nbytes for entry in entries))
        )

    monkeypatch.setattr(rd_build, "write_group", simulate_write_group)
    first = tmp_path / "first"
    shared = tmp_path / "shared"
    rd_build.build(source, first, shared, allocation)

    # Exchange equal-size tensors in a real shard, preserving names, dtypes,
    # shapes, shard filename and config. Only their storage offsets change.
    shard = source / "tiny.safetensors"
    data = shard.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    left = header["layers.0.ffn.experts.0.w1.weight"]
    right = header["layers.0.ffn.experts.1.w1.weight"]
    left["data_offsets"], right["data_offsets"] = (
        right["data_offsets"],
        left["data_offsets"],
    )
    raw = json.dumps(header).encode().ljust(header_size, b" ")
    assert len(raw) == header_size
    shard.write_bytes(data[:8] + raw + data[8 + header_size :])

    second = tmp_path / "second"
    rd_build.build(source, second, shared, allocation)
    identities = [
        json.loads((build / "config.json").read_text())["rapid_quantization"][
            "shared_engram_identity"
        ]
        for build in (first, second)
    ]
    assert identities[0] != identities[1]
    assert {p.name for p in shared.iterdir()} == set(identities)
    with pytest.raises(ValueError, match="reuse build source identity mismatch"):
        rd_build.build(source, tmp_path / "reuse", shared, allocation, reuse_from=first)


def test_fp8_expert_stream_uses_native_decoder(tmp_path):
    import mlx.core as mx

    from scripts.repack_deepseek_v41_native import SourceTensor

    # e4m3 0x38 = 1; ue8m0 127 = scale 1. All projections share tiny slices.
    path = tmp_path / "fp8.bin"
    path.write_bytes(bytes([0x38]) * 4096 + bytes([127]) * 4)
    tensors = {}
    for projection in ("w1", "w3", "w2"):
        base = f"layers.0.ffn.experts.0.{projection}"
        tensors[base + ".weight"] = SourceTensor(
            base + ".weight", path, 0, 4096, "F8_E4M3", (64, 64)
        )
        tensors[base + ".scale"] = SourceTensor(
            base + ".scale", path, 4096, 4, "U8", (2, 2)
        )
    from types import SimpleNamespace

    with mx.stream(mx.cpu):
        weights = load_expert(
            SimpleNamespace(tensors=tensors), 0, 0, expert_dtype="fp8"
        )
        for weight in weights:
            np.testing.assert_array_equal(
                np.array(weight.astype(mx.float32)), np.ones((64, 64))
            )


def test_native_calibration_main_saves_domain_halves_hashes_and_revision(
    tmp_path, monkeypatch
):
    import mlx.core as mx
    import tokenizers

    import rapid_mlx.models.deepseek_v41_native.load as native_load
    from rapid_mlx.models.deepseek_v41_native.config import ModelArgs
    from rapid_mlx.models.deepseek_v41_native.moe import MoE
    from scripts import calibrate_deepseek_v41_reap as calibration

    model_path = tmp_path / "model"
    model_path.mkdir()
    config = {"num_hidden_layers": 1, "n_routed_experts": 4, "num_experts_per_tok": 2}
    (model_path / "config.json").write_text(json.dumps(config))
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "weights.safetensors"}})
    )
    (model_path / "weights.safetensors").write_bytes(b"fixture payload")
    for name in ("code", "math"):
        (tmp_path / f"{name}.txt").write_text(f"{name} distinct corpus")
    manifest = tmp_path / "slices.json"
    manifest.write_text(
        json.dumps({"slices": {"code": ["code.txt"], "math": ["math.txt"]}})
    )

    class TinyTokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def encode(self, text):
            from types import SimpleNamespace

            return SimpleNamespace(ids=[1, 2, 3, 4])

    class TinyModel:
        def __init__(self):
            from types import SimpleNamespace

            self.layers = [
                SimpleNamespace(
                    ffn=MoE(
                        ModelArgs(
                            dim=64,
                            moe_inter_dim=64,
                            n_routed_experts=4,
                            n_activated_experts=2,
                        )
                    )
                )
            ]
            self.caches = []

        def make_cache(self, **kwargs):
            cache = {}
            self.caches.append(cache)
            return cache

        def __call__(self, ids, cache, **kwargs):
            return self.layers[0].ffn(mx.ones((1, 1, 64)) * ids.item())

    with mx.stream(mx.cpu):
        model = TinyModel()
        monkeypatch.setattr(native_load, "load", lambda *args, **kwargs: (model, None))
        monkeypatch.setattr(tokenizers, "Tokenizer", TinyTokenizer)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "calibrate",
                "--model",
                str(model_path),
                "--native",
                "--slices",
                str(manifest),
                "--tokens",
                "4",
                "--keep-experts",
                "2",
                "--model-revision",
                "r",
                "--trust-checkpoint-runtime",
                "--output",
                str(tmp_path / "scores.npz"),
            ],
        )
        calibration.main()
    assert len(model.caches) == 2 and model.caches[0] is not model.caches[1]
    with np.load(tmp_path / "scores.npz") as result:
        metadata = json.loads(str(result["metadata"]))
        assert metadata["model_revision"] == "r"
        from scripts.deepseek_v41_rd_common import checkpoint_hashes

        assert metadata["checkpoint_sha256"] == checkpoint_hashes(model_path)
        assert metadata["tokens_per_slice"] == [4, 4]
        assert metadata["slice_names"] == ["code", "math"]
        assert (
            metadata["corpus"]["code"][0]["sha256"]
            != metadata["corpus"]["math"][0]["sha256"]
        )
        assert result["counts_by_slice"].sum() == 16
        assert result["reap_halves"].shape == (2, 2, 1, 4)
        assert result["split_keep_overlap"].shape == ()
        assert result["split_keep_overlap_by_domain"].shape == (2, 3)
        assert result["split_keep_overlap"] == pytest.approx(
            result["split_keep_overlap_by_domain"][:, 0].min()
        )
        assert metadata["split_keep_overlap"] == pytest.approx(
            result["split_keep_overlap"].item()
        )
        np.testing.assert_array_equal(
            metadata["split_keep_overlap_by_domain"],
            result["split_keep_overlap_by_domain"],
        )


def test_reused_payloads_are_verified_independent_copies(tmp_path, monkeypatch):
    from scripts import deepseek_v41_rd_build as build

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 2
    (source / "config.json").write_text(json.dumps(config))
    allocation = {"layers": {"0": {"kept_expert_ids": [0, 1], "bits": 4}}}
    monkeypatch.setattr(
        build, "transformed", lambda _i, _c, e, *_: bytes(e.nbytes // len(e.sources))
    )
    first, second, shared = tmp_path / "first", tmp_path / "second", tmp_path / "shared"
    build.build(source, first, shared, allocation)
    build.build(source, second, shared, allocation, reuse_from=first)
    payloads = json.loads((first / "config.json").read_text())["rapid_quantization"][
        "payload_sha256"
    ]
    assert payloads
    for name in payloads:
        assert (first / name).stat().st_ino != (second / name).stat().st_ino
    name = next(iter(payloads))
    previous = (second / name).read_bytes()
    original = first / name
    data = original.read_bytes()
    original.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert (second / name).read_bytes() == previous
    with pytest.raises(ValueError, match="reuse payload digest mismatch"):
        build.build(source, tmp_path / "third", shared, allocation, reuse_from=first)


def test_shared_identity_binds_equal_layout_payload(tmp_path):
    from scripts.deepseek_v41_rd_common import checkpoint_hashes

    tiny_source(tmp_path / "a")
    import shutil

    shutil.copytree(tmp_path / "a", tmp_path / "b")
    before = checkpoint_hashes(tmp_path / "a")
    assert before == checkpoint_hashes(tmp_path / "b")
    shard = tmp_path / "b" / "tiny.safetensors"
    data = shard.read_bytes()
    shard.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    after = checkpoint_hashes(tmp_path / "b")
    assert before["config.json"] == after["config.json"]
    assert (
        before["model.safetensors.index.json"] == after["model.safetensors.index.json"]
    )
    assert before["tiny.safetensors"] != after["tiny.safetensors"]


@pytest.mark.parametrize("reuse", [False, True])
def test_copy_mutation_refuses_publication(tmp_path, monkeypatch, reuse):
    from scripts import deepseek_v41_rd_build as build

    source = tmp_path / "source"
    config = tiny_source(source)
    config["text_config"]["num_experts_per_tok"] = 2
    (source / "config.json").write_text(json.dumps(config))
    allocation = {"layers": {"0": {"kept_expert_ids": [0, 1], "bits": 4}}}
    monkeypatch.setattr(
        build, "transformed", lambda _i, _c, e, *_: bytes(e.nbytes // len(e.sources))
    )
    first, shared = tmp_path / "first", tmp_path / "shared"
    build.build(source, first, shared, allocation)
    copy = build.shutil.copyfile

    def changed_copy(src, dst):
        copy(src, dst)
        data = dst.read_bytes()
        dst.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))

    monkeypatch.setattr(build.shutil, "copyfile", changed_copy)
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="payload changed during copy"):
        build.build(
            source, destination, shared, allocation, reuse_from=first if reuse else None
        )
    assert not destination.exists()
    assert not list(tmp_path.glob("failed.partial-*"))
