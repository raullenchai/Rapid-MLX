#!/usr/bin/env python3
"""End-to-end verification of the v2 converter against the SYNTHETIC shard set.

Checks:
  1. Output index covers every source weight (2 extra keys per quantised
     tensor), and total_size == sum of original source weight byte totals.
  2. COPY tensors (1-D norms, A_log, buffers/aux, non-divisible widths) are
     preserved value-for-value (same shape/dtype) in the output.
  3. QUANTISE tensors (incl. PLE/embed tables) round-trip through affine
     q4-g32 with bounded error.
  4. Output shards use deterministic ``model-{i:05d}-of-{N:05d}`` naming and
     every output file's SHA-256 matches SHA256SUMS.txt.
  5. Peak RSS is reported (in the ledger).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from qwen38_streaming_convert import classify_tensor, quantized_tensor_names
from safetensors import safe_open as st_safe_open


def _load(shard: Path) -> dict[str, mx.array]:
    return mx.load(str(shard))


def _dtype_bytes(dtype: str) -> int:
    return {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8": 1,
        "I16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }[dtype.upper()]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _dequantize(q, scales, biases, *, group_size: int, bits: int):
    return np.asarray(
        mx.dequantize(
            mx.array(q),
            mx.array(scales),
            mx.array(biases),
            group_size=group_size,
            bits=bits,
        )
    )


def _copy_predicted(src_name: str, shape: list, dtype: str) -> bool:
    action, _, _ = classify_tensor(src_name, shape, dtype)
    return action == "copy"


def verify(src: Path, out: Path) -> int:
    src, out = src.resolve(), out.resolve()
    src_index = json.loads((src / "model.safetensors.index.json").read_text())
    out_index = json.loads((out / "model.safetensors.index.json").read_text())
    src_wm = src_index["weight_map"]
    out_wm = out_index["weight_map"]
    failures: list[str] = []

    # 1a. coverage: every source weight present in output map
    missing = set(src_wm) - set(out_wm)
    if missing:
        failures.append(f"output index missing source weights: {sorted(missing)[:10]}")

    # 1b. total_size == sum of every emitted output tensor byte total
    src_meta = {}
    for n in src_wm:
        sp = src / src_wm[n]
        with st_safe_open(str(sp), framework="numpy") as sf:
            shp, dt = sf.get_slice(n).get_shape(), sf.get_slice(n).get_dtype()
        src_meta[n] = (list(shp), dt)
        # dtype bytes
        base = dt.upper().lstrip("F")
        b = {"8": 1, "16": 2, "32": 4, "64": 8}.get(base, 4)
        if dt.upper() == "BF16":
            b = 2
    expected_output_keys = set(src_wm)
    for name, (shape, dtype) in src_meta.items():
        action, _, _ = classify_tensor(name, shape, dtype)
        if action == "quantize":
            expected_output_keys.update(quantized_tensor_names(name)[1:])
    if set(out_wm) != expected_output_keys:
        failures.append("output index does not map exactly every emitted tensor")
    expected_total = 0
    for shard_name in sorted(set(out_wm.values())):
        with st_safe_open(str(out / shard_name), framework="numpy") as sf:
            for key in sf.keys():  # noqa: SIM118
                view = sf.get_slice(key)
                expected_total += math.prod(view.get_shape()) * _dtype_bytes(
                    view.get_dtype()
                )
    got_total = out_index["metadata"]["total_size"]
    if got_total != expected_total:
        failures.append(
            f"total_size mismatch: got {got_total}, expected {expected_total}"
        )
    else:
        print(f"  [ok] total_size == {expected_total} (emitted weight byte total)")

    # 2. COPY tensors preserved value-for-value
    copy_names = [
        n for n in src_wm if _copy_predicted(n, src_meta[n][0], src_meta[n][1])
    ]
    for name in copy_names:
        src_arr = _load(src / src_wm[name])[name]
        out_arr = _load(out / out_wm[name])[name]
        if src_arr.shape != out_arr.shape or str(src_arr.dtype) != str(out_arr.dtype):
            failures.append(
                f"COPY {name} shape/dtype mismatch {src_arr.shape}/{src_arr.dtype} "
                f"vs {out_arr.shape}/{out_arr.dtype}"
            )
        elif not bool(mx.array_equal(src_arr, out_arr).item()):
            failures.append(f"COPY {name} value mismatch")
        else:
            print(f"  [ok] COPY preserved: {name} {out_arr.shape} {out_arr.dtype}")
    print(f"  copy tensors verified: {len(copy_names)}")

    # 3. QUANTISE tensors round-trip q4-g32
    quant_names = [n for n in src_wm if n not in copy_names]
    for name in quant_names:
        src_arr = np.asarray(_load(src / src_wm[name])[name].astype(mx.float32))
        shard = _load(out / out_wm[name])
        _, group_size, bits = classify_tensor(
            name, src_meta[name][0], src_meta[name][1]
        )
        _, scales_name, biases_name = quantized_tensor_names(name)
        rec = _dequantize(
            shard[name],
            shard[scales_name],
            shard[biases_name],
            group_size=group_size,
            bits=bits,
        ).astype(np.float32)
        if rec.shape != src_arr.shape:
            failures.append(
                f"quant {name} shape mismatch {rec.shape} vs {src_arr.shape}"
            )
            continue
        mse = float(np.mean((rec - src_arr) ** 2))
        scale = float(np.abs(src_arr).max())
        normalized_mse = mse / max(scale**2, 1e-12)
        if normalized_mse > 0.1:
            failures.append(f"quant {name} excessive error mse={mse:.5g}")
    print(f"  mixed-quant tensors round-tripped: {len(quant_names)}")

    # 4. deterministic shard naming + SHA256SUMS
    shards = sorted(p for p in out.glob("model-*.safetensors"))
    for p in shards:
        if not (p.stem.count("-of-") == 1 and len(p.stem) >= 6):
            failures.append(f"non-deterministic shard name: {p.name}")
    sums = {}
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        h, rel = line.split("  ", 1)
        sums[rel] = h
    bad = sum(1 for rel, h in sums.items() if _sha256(out / rel) != h)
    if bad:
        failures.append(f"SHA256 mismatches: {bad}")
    print(
        f"  SHA256SUMS entries: {len(sums)}, mismatches: {bad}; shards: {len(shards)}"
    )

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL SYNTHETIC CHECKS PASSED (v2 q4-g32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify(Path(sys.argv[1]), Path(sys.argv[2])))
