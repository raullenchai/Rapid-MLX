# SPDX-License-Identifier: Apache-2.0
"""Online (load-time) repack of DeepSeek-style fp8 checkpoints into mxfp8.

Ling 3.0 fp8 checkpoints (``inclusionAI/Ling-3.0-tiny-fp8``) store weights
as safetensors ``F8_E4M3`` bytes plus a ``<name>_scale_inv`` companion
holding e8m0 (``ue8m0``) scales per 128x128 block. mlx has no fp8 dtype
(ml-explore/mlx#1670), so ``mx.load`` cannot even open the shards.

This module loads the ORIGINAL checkpoint directly, repacking at
load time into mlx's ``mxfp8`` quantized layout — **bit-losslessly**:

- mlx mxfp8 stores e4m3 bytes packed 4-per-uint32 plus one e8m0 byte per
  32-element row group. The source's e4m3 weight bytes are copied
  verbatim into the packed words, and each 128x128 block's e8m0 scale
  byte is broadcast to its 4 covered column groups (128/32) and 128
  covered rows. No value is decoded or re-rounded anywhere, so
  ``mx.dequantize`` reproduces ``e4m3(w) * 2^(scale-127)`` exactly
  (verified element-identical; contrast with offline
  ``mx.quantize(mode="mxfp8")`` which re-derives scales per 32-group and
  re-rounds, ~2^-12 max diff).
- Tensors the publisher kept unquantized (``modules_to_not_convert``:
  MLA q/kv LoRA projections, router, embeddings, lm_head, per-head
  gates …) load as plain bf16/f32 — byte-identical too. The router even
  stays bf16, which the offline path could not offer (mlx_lm quantizes
  it 8-bit affine via the model's quant_predicate).

One deliberate, opt-in deviation (default OFF — the default load is
fully faithful): setting ``RAPID_MLX_FP8_LM_HEAD_AFFINE8=1``
re-quantizes the untied ``lm_head`` to affine 8-bit (group 64, float
scale+bias) from its loaded bf16 values. Decode is dominated by the
bf16 head read (483 MB on tiny — measured 1.0 ms/token, ~90% of the
faithful-vs-offline gap); affine8 halves that while staying far more
accurate than the offline path's mxfp8 head (measured on real hidden
states: top-1 flips 0%, max |Δlogit| 0.08 on a ~41 logit range, vs
4.2% flips / 1.05 for mxfp8's power-of-two scales).

Memory: the model skeleton's random init and the ``nn.quantize`` graph
stay lazy (never evaluated); only the repacked arrays materialize, so
peak RSS is about the checkpoint size (~8.4 GB for tiny), not the bf16
size.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

_SCALE_SUFFIX = "_scale_inv"
_MXFP8_GROUP = 32  # mlx mxfp8 group size (fixed by the format)
_SUPPORTED_FP8_FORMAT = "e4m3"
_SUPPORTED_SCALE_FORMAT = "ue8m0"


def _supported_quantization_config(config: dict) -> bool:
    """Whether ``config`` describes the exact byte formats we can repack.

    ``quant_method=fp8`` alone is not a wire format: other publishers use
    float scales, per-channel layouts, or different FP8 encodings. Claiming
    those checkpoints here would divert them into a byte-level loader whose
    assumptions do not hold.
    """
    qc = config.get("quantization_config") or {}
    block = qc.get("weight_block_size")
    return (
        qc.get("quant_method") == "fp8"
        and str(qc.get("fmt", "")).lower() == _SUPPORTED_FP8_FORMAT
        and str(qc.get("scale_fmt", "")).lower() == _SUPPORTED_SCALE_FORMAT
        and isinstance(block, list | tuple)
        and len(block) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in block)
    )


def is_fp8_block_checkpoint(model_path: Path) -> bool:
    """True only for the supported e4m3 + ue8m0 block-FP8 wire format."""
    cfg = model_path / "config.json"
    if not cfg.exists():
        return False
    try:
        qc = json.loads(cfg.read_text()).get("quantization_config") or {}
    except (OSError, json.JSONDecodeError):
        return False
    return _supported_quantization_config({"quantization_config": qc})


class _ShardReader:
    """Minimal safetensors reader returning raw little-endian bytes.

    numpy-side only — deliberately no framework dtype mapping, since the
    whole point is that mlx (0.31.x) cannot represent F8_E4M3/F8_E8M0.
    """

    def __init__(self, path: Path):
        self.path = path
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"invalid safetensors file (truncated header): {path}")
            hlen = struct.unpack("<Q", prefix)[0]
            if hlen > file_size - 8:
                raise ValueError(f"invalid safetensors header length in {path}")
            self.header = json.loads(f.read(hlen))
        if not isinstance(self.header, dict):
            raise ValueError(f"invalid safetensors header in {path}")
        self.header.pop("__metadata__", None)
        self.data_start = 8 + hlen
        data_size = file_size - self.data_start
        for name, entry in self.header.items():
            try:
                off0, off1 = entry["data_offsets"]
                shape = entry["shape"]
                dtype = entry["dtype"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid safetensors entry {name!r} in {path}"
                ) from exc
            if (
                not isinstance(off0, int)
                or isinstance(off0, bool)
                or not isinstance(off1, int)
                or isinstance(off1, bool)
                or off0 < 0
                or off1 < off0
                or off1 > data_size
                or not isinstance(shape, list)
                or any(
                    not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                    for dim in shape
                )
                or not isinstance(dtype, str)
            ):
                raise ValueError(f"invalid safetensors entry {name!r} in {path}")

    def read(self, name: str) -> tuple[np.ndarray, str, list[int]]:
        ent = self.header[name]
        off0, off1 = ent["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + off0)
            payload = f.read(off1 - off0)
        if len(payload) != off1 - off0:
            raise ValueError(f"truncated safetensors payload {name!r} in {self.path}")
        raw = np.frombuffer(payload, dtype=np.uint8)
        return raw, ent["dtype"], ent["shape"]


def _open_shards(model_path: Path) -> tuple[dict[str, str], dict[str, _ShardReader]]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        single = model_path / "model.safetensors"
        weight_map = {n: single.name for n in _ShardReader(single).header}
    shards = {
        fname: _ShardReader(model_path / fname)
        for fname in sorted(set(weight_map.values()))
    }
    return weight_map, shards


def _raw_to_mx(raw: np.ndarray, dtype: str, shape: list[int]) -> mx.array:
    """Non-fp8 safetensors payload → mx array (byte-identical)."""
    elements = int(np.prod(shape, dtype=np.int64))
    bytes_per_element = {"BF16": 2, "F32": 4, "F16": 2}.get(dtype)
    if bytes_per_element is None:
        raise ValueError(f"unsupported non-fp8 dtype {dtype}")
    if raw.size != elements * bytes_per_element:
        raise ValueError(
            f"invalid {dtype} payload size: got {raw.size} bytes for shape {shape}"
        )
    if dtype == "BF16":
        return mx.array(raw.view(np.uint16)).view(mx.bfloat16).reshape(shape)
    if dtype == "F32":
        return mx.array(raw.view(np.float32)).reshape(shape)
    if dtype == "F16":
        return mx.array(raw.view(np.float16)).reshape(shape)
    raise AssertionError("unreachable")


def _repack_fp8(
    w_raw: np.ndarray,
    w_shape: list[int],
    s_raw: np.ndarray,
    s_shape: list[int],
    block: tuple[int, int],
) -> tuple[mx.array, mx.array]:
    """(e4m3 bytes, block e8m0 scales) → mlx mxfp8 (packed weight, scales).

    Pure byte moves: e4m3 bytes are packed 4-per-uint32 verbatim; each
    block scale byte is repeated over the rows/column-groups it covers.
    """
    rows, cols = w_shape
    bs_r, bs_c = block
    if cols % _MXFP8_GROUP:
        raise ValueError(f"cols {cols} not divisible by mxfp8 group {_MXFP8_GROUP}")
    if bs_c % _MXFP8_GROUP:
        raise ValueError(f"block width {bs_c} not divisible by {_MXFP8_GROUP}")
    if w_raw.size != rows * cols:
        raise ValueError(
            f"invalid F8_E4M3 payload size: got {w_raw.size} bytes for shape {w_shape}"
        )
    expected_scale_shape = [(rows + bs_r - 1) // bs_r, (cols + bs_c - 1) // bs_c]
    if s_shape != expected_scale_shape or s_raw.size != int(np.prod(s_shape)):
        raise ValueError(
            "invalid F8_E8M0 scale layout: "
            f"got shape {s_shape} for weight {w_shape}, block {block}; "
            f"expected {expected_scale_shape}"
        )

    wq = w_raw.reshape(rows, cols // 4, 4).view(np.uint32).reshape(rows, cols // 4)

    s = s_raw.reshape(s_shape)
    scales = np.repeat(np.repeat(s, bs_r, axis=0)[:rows], bs_c // _MXFP8_GROUP, axis=1)
    scales = scales[:, : cols // _MXFP8_GROUP]

    return mx.array(wq), mx.array(scales)


def load_fp8_model_online(model_path: Path) -> nn.Module:
    """Build the model and load an fp8-block checkpoint via mxfp8 repack."""
    config = json.loads((model_path / "config.json").read_text())
    qc = config.get("quantization_config") or {}
    if not _supported_quantization_config(config):
        raise ValueError(
            f"{model_path} is not a supported e4m3 + ue8m0 block-FP8 checkpoint"
        )
    bs = qc["weight_block_size"]
    block = (int(bs[0]), int(bs[1]))
    model_type = config["model_type"]

    arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    model = arch.Model(arch.ModelArgs.from_dict(config))

    weight_map, shards = _open_shards(model_path)

    def read(name: str):
        return shards[weight_map[name]].read(name)

    # Quantize exactly the modules the checkpoint shipped as fp8 — the
    # scale companion's presence is the source of truth. Module paths
    # mirror checkpoint names, except experts are a stacked SwitchLinear
    # (`…mlp.experts.gate_proj`) standing in for per-expert tensors.
    def has_fp8(name: str) -> bool:
        return f"{name}.weight{_SCALE_SUFFIX}" in weight_map

    def class_predicate(path: str, module: nn.Module) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        marker = ".mlp.experts."
        if marker in path:
            head, _, tail = path.partition(marker)
            return has_fp8(f"{head}{marker[:-1]}.0.{tail}")
        # Fused KDA serving modules (see BailingKDA/sanitize): quantized
        # iff their first source projection shipped fp8 — row concat of
        # same-scheme sources is what sanitize produces.
        if path.endswith(".qkv_proj"):
            return has_fp8(path[: -len("qkv_proj")] + "q_proj")
        if path.endswith(".fg_proj"):
            return has_fp8(path[: -len("fg_proj")] + "f_proj")
        return has_fp8(path)

    # Both the skeleton's random init and this quantization graph stay
    # lazy; load_weights below replaces every parameter before anything
    # is evaluated, so neither ever materializes.
    nn.quantize(
        model,
        group_size=_MXFP8_GROUP,
        bits=8,
        mode="mxfp8",
        class_predicate=class_predicate,
    )

    weights: dict[str, mx.array] = {}
    n_repacked = 0
    for name in weight_map:
        if name.endswith(_SCALE_SUFFIX):
            continue
        raw, dtype, shape = read(name)
        if dtype == "F8_E4M3":
            scale_name = name + _SCALE_SUFFIX
            if scale_name not in weight_map:
                raise ValueError(
                    f"{name} is F8_E4M3 but has no {_SCALE_SUFFIX} companion"
                )
            s_raw, s_dtype, s_shape = read(scale_name)
            if s_dtype != "F8_E8M0":
                raise ValueError(
                    f"{scale_name} is {s_dtype}; online mxfp8 repack supports ue8m0 "
                    "(power-of-two) scales only — f32-scale fp8 sources are not "
                    "supported yet"
                )
            wq, scales = _repack_fp8(raw, shape, s_raw, s_shape, block)
            base = name[: -len(".weight")]
            weights[f"{base}.weight"] = wq
            weights[f"{base}.scales"] = scales
            n_repacked += 1
        else:
            weights[name] = _raw_to_mx(raw, dtype, shape)

    logger.info(
        "fp8 online repack: %d tensors repacked to mxfp8 (bit-lossless), %d kept fp",
        n_repacked,
        len(weights) - 2 * n_repacked,
    )

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)

    # Opt-in (default OFF, keeping the load fully checkpoint-faithful):
    # RAPID_MLX_FP8_LM_HEAD_AFFINE8=1 quantizes the untied bf16 lm_head
    # to affine 8-bit from its just-loaded real values — ~+10% decode on
    # tiny for a measured-zero top-1 change (see module docstring).
    want_q = os.environ.get("RAPID_MLX_FP8_LM_HEAD_AFFINE8", "") == "1"
    if want_q and isinstance(getattr(model, "lm_head", None), nn.Linear):
        nn.quantize(
            model,
            class_predicate=lambda path, module: (
                {"group_size": 64, "bits": 8, "mode": "affine"}
                if path == "lm_head"
                else False
            ),
        )
        logger.info("fp8 online repack: lm_head re-quantized to affine8 (opt-in)")

    mx.eval(model.parameters())
    model.eval()
    return model
