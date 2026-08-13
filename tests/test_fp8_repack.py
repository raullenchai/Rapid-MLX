# SPDX-License-Identifier: Apache-2.0
"""Tests for the load-time fp8 -> mxfp8 repack (``vllm_mlx/fp8_repack.py``).

Pins:

1. ``_repack_fp8`` is a pure byte move: mlx ``mx.dequantize(mode="mxfp8")``
   over the repacked (weight, scales) reproduces ``e4m3(byte) *
   2^(scale-127)`` with the source's 128x128 block scales broadcast to
   32-element groups — element-identical, no re-rounding.
2. ``is_fp8_block_checkpoint`` keys off ``quantization_config.quant_method``.
3. ``load_fp8_model_online`` on a synthetic fp8 bailing_hybrid checkpoint:
   fp8-shipped linears become mxfp8-quantized modules holding the exact
   source bytes, ``modules_to_not_convert``-style bf16 tensors stay plain,
   and the loaded model runs a forward pass.

Real-checkpoint validation (Ling-3.0-tiny-fp8, 4 spot tensors EXACT,
serve + correctness probe) was run out-of-band; these tests keep the
synthetic contract in CI.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402

from vllm_mlx.fp8_repack import (  # noqa: E402
    _repack_fp8,
    is_fp8_block_checkpoint,
    load_fp8_model_online,
)


def _e4m3_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0x0F
        m = b & 0x07
        if e == 0x0F and m == 0x07:
            lut[b] = np.nan
        elif e == 0:
            lut[b] = s * (m / 8.0) * 2.0**-6
        else:
            lut[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut


def test_repack_is_bit_exact():
    rng = np.random.default_rng(9)
    rows, cols, bs = 96, 160, 32  # 3x5 scale blocks, block 32 == group
    w = rng.integers(0, 0x7F, size=rows * cols, dtype=np.uint8)
    s = rng.integers(
        110, 140, size=(rows + bs - 1) // bs * ((cols + bs - 1) // bs), dtype=np.uint8
    )
    s_shape = [(rows + bs - 1) // bs, (cols + bs - 1) // bs]

    wq, scales = _repack_fp8(w, [rows, cols], s, s_shape, (bs, bs))
    got = np.array(
        mx.dequantize(wq, scales, group_size=32, bits=8, mode="mxfp8").astype(
            mx.float32
        )
    )

    ref = (
        _e4m3_lut()[w].reshape(rows, cols)
        * np.repeat(
            np.repeat(2.0 ** (s.reshape(s_shape).astype(np.int32) - 127), bs, 0)[:rows],
            bs,
            1,
        )[:, :cols]
    )
    np.testing.assert_array_equal(got, ref)


def test_repack_128_block_broadcast():
    """128x128 source blocks (the shipped configuration) cover 4 groups."""
    rng = np.random.default_rng(10)
    rows, cols = 256, 384  # 2x3 blocks of 128
    w = rng.integers(0, 0x7F, size=rows * cols, dtype=np.uint8)
    s = rng.integers(120, 133, size=6, dtype=np.uint8)

    wq, scales = _repack_fp8(w, [rows, cols], s, [2, 3], (128, 128))
    assert scales.shape == (rows, cols // 32)
    got = np.array(
        mx.dequantize(wq, scales, group_size=32, bits=8, mode="mxfp8").astype(
            mx.float32
        )
    )
    ref = _e4m3_lut()[w].reshape(rows, cols) * np.repeat(
        np.repeat(2.0 ** (s.reshape(2, 3).astype(np.int32) - 127), 128, 0), 128, 1
    )
    np.testing.assert_array_equal(got, ref)


def _write_safetensors_raw(path: Path, entries: dict):
    header, blobs, off = {}, [], 0
    for name, (raw, dtype, shape) in entries.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [off, off + len(raw)],
        }
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def _bf16_bytes(arr: np.ndarray) -> bytes:
    u = arr.astype(np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)).tobytes()


@pytest.fixture()
def fp8_bailing_checkpoint(tmp_path):
    """Tiny synthetic bailing_hybrid fp8 checkpoint (2 layers, 4 experts)."""
    rng = np.random.default_rng(21)
    H, HD, NH, NE, MOE_FFN, FFN, VOCAB, BS = 64, 16, 4, 4, 32, 128, 128, 32
    src = tmp_path / "fp8src"
    src.mkdir()
    entries = {}

    def add_bf16(name, *shape):
        entries[name] = (
            _bf16_bytes(rng.standard_normal(shape).astype(np.float32) * 0.02),
            "BF16",
            list(shape),
        )

    def add_f32(name, *shape):
        entries[name] = (
            (rng.standard_normal(shape).astype(np.float32) * 0.02).tobytes(),
            "F32",
            list(shape),
        )

    def add_fp8(name, rows, cols):
        w = rng.integers(0, 0x7F, size=rows * cols, dtype=np.uint8)
        s = rng.integers(120, 133, size=(rows // BS) * (cols // BS), dtype=np.uint8)
        entries[name + ".weight"] = (w.tobytes(), "F8_E4M3", [rows, cols])
        entries[name + ".weight_scale_inv"] = (
            s.tobytes(),
            "F8_E8M0",
            [rows // BS, cols // BS],
        )
        return w

    add_bf16("model.word_embeddings.weight", VOCAB, H)
    add_bf16("lm_head.weight", VOCAB, H)
    add_bf16("model.norm.weight", H)
    kept = {}
    for i in range(2):
        p = f"model.layers.{i}"
        add_bf16(f"{p}.input_layernorm.weight", H)
        add_bf16(f"{p}.post_attention_layernorm.weight", H)
        if i == 1:  # MLA layer (group of 2 for the tiny config)
            add_bf16(f"{p}.attention.q_a_proj.weight", 32, H)
            add_bf16(f"{p}.attention.q_a_layernorm.weight", 32)
            add_bf16(f"{p}.attention.q_b_proj.weight", NH * 24, 32)
            add_bf16(f"{p}.attention.kv_a_proj_with_mqa.weight", 40, H)
            add_bf16(f"{p}.attention.kv_a_layernorm.weight", 32)
            add_bf16(f"{p}.attention.kv_b_proj.weight", NH * 32, 32)
            kept[f"{p}.attention.dense"] = add_fp8(f"{p}.attention.dense", H, NH * HD)
            add_bf16(f"{p}.attention.g_proj.weight", NH, H)
        else:  # KDA layer
            for t in ("q", "k", "v", "f", "g"):
                kept[f"{p}.attention.{t}_proj"] = add_fp8(
                    f"{p}.attention.{t}_proj", NH * HD, H
                )
            for t in ("q", "k", "v"):
                add_bf16(f"{p}.attention.{t}_conv1d.weight", NH * HD, 1, 4)
            add_bf16(f"{p}.attention.b_proj.weight", NH, H)
            add_f32(f"{p}.attention.A_log", NH)
            add_f32(f"{p}.attention.dt_bias", NH * HD)
            add_bf16(f"{p}.attention.o_norm.weight", HD)
            kept[f"{p}.attention.o_proj"] = add_fp8(f"{p}.attention.o_proj", H, NH * HD)
        if i == 0:
            add_fp8(f"{p}.mlp.gate_proj", FFN, H)
            add_fp8(f"{p}.mlp.up_proj", FFN, H)
            add_fp8(f"{p}.mlp.down_proj", H, FFN)
        else:
            add_bf16(f"{p}.mlp.gate.weight", NE, H)
            add_f32(f"{p}.mlp.gate.expert_bias", NE)
            for e in range(NE):
                for t, r, c in (
                    ("gate", MOE_FFN, H),
                    ("up", MOE_FFN, H),
                    ("down", H, MOE_FFN),
                ):
                    add_fp8(f"{p}.mlp.experts.{e}.{t}_proj", r, c)
            for t, r, c in (
                ("gate", MOE_FFN, H),
                ("up", MOE_FFN, H),
                ("down", H, MOE_FFN),
            ):
                add_fp8(f"{p}.mlp.shared_experts.{t}_proj", r, c)

    _write_safetensors_raw(src / "model.safetensors", entries)
    config = {
        "model_type": "bailing_hybrid",
        "hidden_size": H,
        "num_hidden_layers": 2,
        "num_attention_heads": NH,
        "num_key_value_heads": NH,
        "head_dim": HD,
        "intermediate_size": FFN,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 6000000.0,
        "rope_interleave": True,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        "layer_group_size": 2,
        "short_conv_kernel_size": 4,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "num_experts": NE,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": MOE_FFN,
        "moe_shared_expert_intermediate_size": MOE_FFN,
        "num_shared_experts": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 2,
        "topk_group": 1,
        "score_function": "sigmoid",
        "moe_router_enable_expert_bias": True,
        "first_k_dense_replace": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [BS, BS],
        },
    }
    (src / "config.json").write_text(json.dumps(config))
    return src, kept


def test_is_fp8_block_checkpoint(fp8_bailing_checkpoint, tmp_path):
    src, _ = fp8_bailing_checkpoint
    assert is_fp8_block_checkpoint(src)
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert not is_fp8_block_checkpoint(plain)
    assert not is_fp8_block_checkpoint(tmp_path / "missing")

    # ``quant_method=fp8`` is not itself a wire format. Float-scaled and
    # otherwise unspecified FP8 checkpoints must stay on their normal loader
    # instead of being interpreted as e4m3 + ue8m0 bytes.
    for name, quantization in {
        "unspecified": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        "float_scales": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "float32",
            "weight_block_size": [128, 128],
        },
    }.items():
        candidate = tmp_path / name
        candidate.mkdir()
        (candidate / "config.json").write_text(
            json.dumps({"quantization_config": quantization})
        )
        assert not is_fp8_block_checkpoint(candidate)


def test_repack_rejects_payload_and_scale_shape_mismatches():
    with pytest.raises(ValueError, match="payload size"):
        _repack_fp8(
            np.zeros(31, dtype=np.uint8),
            [1, 32],
            np.ones(1, dtype=np.uint8),
            [1, 1],
            (32, 32),
        )
    with pytest.raises(ValueError, match="scale layout"):
        _repack_fp8(
            np.zeros(64 * 64, dtype=np.uint8),
            [64, 64],
            np.ones(1, dtype=np.uint8),
            [1, 1],
            (32, 32),
        )


def test_online_load_repacks_and_runs(fp8_bailing_checkpoint):
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    src, kept = fp8_bailing_checkpoint
    model = load_fp8_model_online(src)

    # fp8-shipped linears became one fused mxfp8 module holding the
    # exact source bytes row-concatenated in q|k|v order...
    qkv = model.model.layers[0].attention.qkv_proj
    assert qkv.mode == "mxfp8" and qkv.group_size == 32 and qkv.bits == 8
    src_bytes = np.concatenate(
        [kept[f"model.layers.0.attention.{t}_proj"] for t in "qkv"]
    )
    got_bytes = np.array(qkv.weight).view(np.uint8).reshape(-1)
    np.testing.assert_array_equal(got_bytes, src_bytes)

    # ...bf16-shipped modules stayed plain (checkpoint fidelity the
    # offline path can't offer for the router), INCLUDING the lm_head —
    # the affine8 head speedup is opt-in via env var, default faithful.
    assert not hasattr(model.model.layers[1].mlp.gate, "scales")
    assert not hasattr(model.model.layers[1].attention.q_a_proj, "scales")
    assert not hasattr(model.lm_head, "scales")

    # Forward runs and yields finite logits.
    logits = model(mx.array([[1, 2, 3]]))
    mx.eval(logits)
    assert logits.shape == (1, 3, 128)
    assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))


def test_opt_in_lm_head_affine8(fp8_bailing_checkpoint, monkeypatch):
    """RAPID_MLX_FP8_LM_HEAD_AFFINE8=1 quantizes ONLY the lm_head."""
    from vllm_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    src, _ = fp8_bailing_checkpoint
    monkeypatch.setenv("RAPID_MLX_FP8_LM_HEAD_AFFINE8", "1")
    model = load_fp8_model_online(src)
    head = model.lm_head
    assert hasattr(head, "scales") and head.bits == 8 and head.group_size == 64
    assert getattr(head, "mode", "affine") == "affine"
    # Everything else untouched: fp8 modules still mxfp8, router still fp.
    assert model.model.layers[0].attention.qkv_proj.mode == "mxfp8"
    assert not hasattr(model.model.layers[1].mlp.gate, "scales")
    logits = model(mx.array([[1, 2, 3]]))
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))
