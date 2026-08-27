#!/usr/bin/env python3
"""FAIL-CLOSED streaming q4-g32 converter for Qwen3.8-Flash-Next (prototype v2).

Script-only lane helper for Vector's qwen4_exp port. Never edits model math.
Converts an HF safetensors snapshot to an MLX affine q4-g32 layout by
streaming shard-by-shard, quantising every quantisable weight (including the
~51B-param PLE embedding table) WITHOUT ever materialising the whole table in
RAM — each source tensor is read via mmap byte-slice, quantised in-memory one
tensor at a time, and written to bounded output shards.

Revision against Vector's review of cee4ef00 ("NOT safe for real weights"):
  1. PLE is QUANTISED q4-g32 (not preserved BF16). Still never materialised:
     per-tensor streaming keeps peak RSS bounded regardless of the 51B table.
  2. Explicit tensor-name contract (embed_tokens / mm.embedding / ple_embed.rows.*)
     instead of substring matching.
  3. Per-tensor quantisation predicate from the manifest classification: 1-D
     norms, ``A_log``, buffers, and widths not divisible by group_size are
     copied as-is (fp), never quantised.
  4. Output index ``total_size`` = sum of original weight byte totals (the
     loader's semantic model size, not source *.safetensors file sizes), and
     output shards use deterministic ``model-{i:05d}-of-{N:05d}`` naming with
     the real total ``N``.

Verified on a SYNTHETIC scaled-down shard set (same name/dtype/shape pattern
plus the flagged tensor classes), NOT on real weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import resource
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3
GUARD_EXPERT_SSD = "/Volumes/Extreme SSD"

# Buffers / special params copied as-is (never quantised).
_BUFFER_SUFFIXES = (".A_log",)
# dtypes that are aux/buffer-like and copied as-is rather than quantised.
_NON_QUANT_DTYPES = {"F64", "F8", "F4", "I8", "I16", "I32", "I64", "U8", "BOOL"}
_FLOAT_DTYPES = {"BF16", "F16", "F32"}

_PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
_PLE_SUFFIX = ".weight"
_ROUTING_GATE_SUFFIXES = (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")

_NP_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
_DTYPE_BYTES = {"F8": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype.upper(), 4)


# ---------------------------------------------------------------------------
# Manifest classification (Vector's per-tensor quantise / copy predicate)
# ---------------------------------------------------------------------------
def _is_ple_shard(name: str) -> bool:
    if not (name.startswith(_PLE_PREFIX) and name.endswith(_PLE_SUFFIX)):
        return False
    shard = name[len(_PLE_PREFIX) : -len(_PLE_SUFFIX)]
    return shard.isdecimal() and 0 <= int(shard) < 128


def classify_tensor(name: str, shape: list[int], dtype: str) -> tuple[str, int, int]:
    """Return ``(action, group_size, bits)`` for one source tensor.

    Explicit contract first, then the manifest predicate:
      * 1-D weights (norms, biases) are copy/fp.
      * ``A_log`` safegate params / buffers and non-fp aux dtypes are copy/fp.
      * widths whose last dim is not divisible by ``group_size`` are copy/fp.
      * everything else (2-D + divisible, incl. embed/PLE tables) is quantised.
    """
    up = dtype.upper()
    if len(shape) <= 1 or name.endswith(_BUFFER_SUFFIXES):
        return "copy", 0, 0
    if up in _NON_QUANT_DTYPES or up not in _FLOAT_DTYPES:
        return "copy", 0, 0
    if _is_ple_shard(name):
        if shape[-1] != 160 or shape[-1] % 32:
            raise RuntimeError(f"invalid Qwen4-Exp PLE shape for {name}: {shape}")
        return "quantize", 32, 4
    if name.endswith(_ROUTING_GATE_SUFFIXES):
        if shape[-1] % 64:
            raise RuntimeError(
                f"invalid Qwen4-Exp routing-gate shape for {name}: {shape}"
            )
        return "quantize", 64, 8
    if shape[-1] % 64:
        return "copy", 0, 0
    return "quantize", 64, 4


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
def _safe_source_shard(snapshot: Path, relative: Path) -> Path:
    """Resolve a flat HF shard name, including standard snapshot symlinks.

    HF snapshot files point into the repository cache's sibling ``blobs``
    directory.  The index is allowed to name one basename only, and the
    resolved target must remain inside that model-cache root.
    """
    if relative.is_absolute() or len(relative.parts) != 1:
        raise RuntimeError(f"source index contains a non-flat shard path: {relative}")
    candidate = snapshot / relative
    if not candidate.is_file():
        raise RuntimeError(f"missing source shard: {relative}")
    resolved = candidate.resolve()
    snapshot_root = snapshot.resolve()
    allowed_roots = [snapshot_root]
    # Standard HF cache layout: <repo>/snapshots/<revision> contains symlinks
    # into the same repo's sibling blobs directory. A plain local checkpoint
    # has no such exception and is confined to its own source directory.
    if snapshot.parent.name == "snapshots":
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_dir():
            allowed_roots.append(blobs.resolve())
    if not any(
        resolved == root or resolved.is_relative_to(root) for root in allowed_roots
    ):
        raise RuntimeError(f"source shard escapes model cache root: {relative}")
    return resolved


def _read_shard_layout(shard: Path) -> tuple[int, dict]:
    with (
        open(shard, "rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mem,
    ):
        header_len = int(np.frombuffer(mem[:8], dtype=np.uint64)[0])
        header = json.loads(mem[8 : 8 + header_len].decode("utf-8"))
    return header_len, header


def _tensor_bytes(shard: Path, header_len: int, header: dict, name: str) -> bytes:
    info = header.get(name)
    if info is None:
        raise RuntimeError(f"weight {name!r} not in source shard {shard}")
    begin = 8 + header_len + int(info["data_offsets"][0])
    end = 8 + header_len + int(info["data_offsets"][1])
    with (
        open(shard, "rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mem,
    ):
        return bytes(mem[begin:end])


def _mlx_from_bytes(data: bytes, dtype: str, shape: list[int]) -> mx.array:
    up = dtype.upper()
    if up == "BF16":
        return mx.array(np.frombuffer(data, dtype=np.uint16).reshape(shape)).view(
            mx.bfloat16
        )
    np_dtype = _NP_DTYPES.get(up)
    if np_dtype is None:
        raise RuntimeError(f"unsupported source dtype {dtype} for tensor")
    return mx.array(np.frombuffer(data, dtype=np_dtype).reshape(shape))


def quantize_affine(
    arr: mx.array, *, group_size: int, bits: int
) -> tuple[mx.array, mx.array, mx.array]:
    q, scales, biases = mx.quantize(
        arr, group_size=group_size, bits=bits, mode="affine"
    )
    mx.eval(q, scales, biases)
    return q, scales, biases


def quantized_tensor_names(name: str) -> tuple[str, str, str]:
    """Return the MLX parameter names for an affine-quantized tensor.

    MLX stores ``scales`` and ``biases`` beside a module's ``weight``.  Most
    checkpoint tensors therefore replace the terminal ``.weight`` component;
    Qwen4-Exp's fused expert tensors omit that component and retain the
    established append form until the model sanitizer splits them.
    """
    if name.endswith(".weight"):
        module = name.removesuffix(".weight")
        return name, f"{module}.scales", f"{module}.biases"
    return name, f"{name}.scales", f"{name}.biases"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Replace one metadata file only after its complete payload is durable."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_sha256sums(output: Path) -> None:
    lines = []
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{_sha256(p)}  {p.relative_to(output).as_posix()}\n")
    _atomic_write_bytes(output / "SHA256SUMS.txt", "".join(lines).encode())


_AUX_COPY_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_aux_metadata(source: Path, output: Path) -> list[str]:
    copied: list[str] = []
    for name in _AUX_COPY_NAMES:
        src = source / name
        if src.is_file():
            _safe_source_shard(source, Path(name))
            shutil.copyfile(src, output / name)
            copied.append(name)
    return copied


def _module_path_for_override(source_weight: str) -> str:
    key = source_weight.removesuffix(".weight")
    if key.startswith("model.language_model"):
        key = key.replace("model.language_model", "language_model.model", 1)
    if ".ngram_embedding.shard_" in key:
        prefix, shard = key.split(".ngram_embedding.shard_", 1)
        key = f"{prefix}.ngram_embedding.shards.{int(shard)}"
    return key


def _write_quantized_config(
    source: Path, output: Path, overrides: dict[str, dict]
) -> None:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise RuntimeError("source has no config.json")
    config = json.loads(config_path.read_text())
    quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
    quantization.update(dict(sorted(overrides.items())))
    config["quantization"] = quantization
    config["quantization_config"] = quantization
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def inspect_manifest(source: Path) -> dict:
    """Classify the real headers without reading tensor payloads."""
    source = source.resolve()
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("source has no model.safetensors.index.json")
    weight_map: dict[str, str] = json.loads(index_path.read_text()).get(
        "weight_map", {}
    )
    layouts: dict[str, tuple[int, dict]] = {}
    counts = {"q4_g32_ple": 0, "q8_g64_gate": 0, "q4_g64": 0, "copy": 0}
    source_bytes = 0
    for name in sorted(weight_map):
        shard = _safe_source_shard(source, Path(weight_map[name]))
        if str(shard) not in layouts:
            layouts[str(shard)] = _read_shard_layout(shard)
        _, header = layouts[str(shard)]
        info = header.get(name)
        if info is None:
            raise RuntimeError(f"weight {name!r} not in source shard {shard}")
        shape, dtype = list(info["shape"]), info["dtype"]
        source_bytes += math.prod(shape) * _dtype_bytes(dtype)
        action, group_size, bits = classify_tensor(name, shape, dtype)
        if action == "copy":
            counts["copy"] += 1
        elif (group_size, bits) == (32, 4):
            counts["q4_g32_ple"] += 1
        elif (group_size, bits) == (64, 8):
            counts["q8_g64_gate"] += 1
        elif (group_size, bits) == (64, 4):
            counts["q4_g64"] += 1
        else:
            raise RuntimeError(f"unrecognized classification for {name}")
    if sum(counts.values()) != len(weight_map):
        raise RuntimeError("manifest classification is not one-to-one")
    # 48 decoder layers carry two routing gates each; the checkpoint's one
    # MTP layer carries the same pair and is retained for milestone M2.
    if counts["q4_g32_ple"] != 128 or counts["q8_g64_gate"] != 98:
        raise RuntimeError(f"checkpoint contract count mismatch: {counts}")
    return {"weights": len(weight_map), "source_bytes": source_bytes, **counts}


# ---------------------------------------------------------------------------
# Bounded quantised-shard writer with deterministic naming + weight map
# ---------------------------------------------------------------------------
class _ShardWriter:
    """Accumulates per-base-weight safetensors records into bounded shards and
    records which shard each base weight lands in (incl. its .scales/.biases)."""

    def __init__(self, output: Path, max_bytes: int):
        self.output = output
        self.max_bytes = max_bytes
        self.index = 0
        self.files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        # current buffer: list of (base_name, tensors_dict), + running bytes
        self._buf: list[dict[str, mx.array]] = []
        self._buf_bytes = 0
        self.total_output_bytes = 0

    def add(self, tensors: dict[str, mx.array]) -> None:
        nbytes = sum(t.nbytes for t in tensors.values())
        if self._buf and self._buf_bytes + nbytes > self.max_bytes:
            self._flush()
        self._buf.append(tensors)
        self._buf_bytes += nbytes
        self.total_output_bytes += nbytes

    def _flush(self) -> None:
        if not self._buf:
            return
        self.index += 1
        path = self.output / f"model-{self.index:05d}-of-00000.safetensors"
        payload: dict[str, mx.array] = {}
        for tensors in self._buf:
            for key, value in tensors.items():
                payload[key] = value
                self.weight_map[key] = path.name
        mx.save_safetensors(str(path), payload, metadata={"format": "mlx"})
        self.files.append(path)
        self._buf = []
        self._buf_bytes = 0

    def finalize(self) -> list[Path]:
        self._flush()
        total = len(self.files)
        renamed: list[Path] = []
        for idx, path in enumerate(self.files, start=1):
            new = self.output / f"model-{idx:05d}-of-{total:05d}.safetensors"
            if new != path:
                path.replace(new)
            renamed.append(new)
            shard_name = new.name
            for base, target in list(self.weight_map.items()):
                if target == path.name:
                    self.weight_map[base] = shard_name
        self.files = renamed
        return renamed


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------
def _convert_into(
    source: Path,
    output: Path,
    *,
    max_shard_bytes: int,
    min_free_bytes: int = 140 * 1024**3,
    max_rss_bytes: int = int(220.0 * 1024**3),
) -> dict:
    source = source.resolve()
    output = output.expanduser().resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise RuntimeError("internal conversion staging directory is not empty")

    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("source has no model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError("index has empty weight_map")

    started = time.monotonic()

    shard_layouts: dict[str, tuple[int, dict]] = {}
    for n in weight_map:
        sh = _safe_source_shard(source, Path(weight_map[n]))
        if str(sh) not in shard_layouts:
            shard_layouts[str(sh)] = _read_shard_layout(sh)

    writer = _ShardWriter(output, max_shard_bytes)
    total_weight_bytes = 0  # Σ numel*dtype_bytes over ALL source weights
    n_quant = n_copy = 0
    quant_overrides: dict[str, dict] = {}

    for name in sorted(weight_map):
        if peak_rss_bytes() > max_rss_bytes:
            raise RuntimeError(
                f"process footprint aborted: peak RSS {peak_rss_bytes() / 1024**3:.1f} "
                f"GiB > guard {max_rss_bytes / 1024**3:.1f} GiB"
            )
        src = _safe_source_shard(source, Path(weight_map[name]))
        header_len, header = shard_layouts[str(src)]
        info = header.get(name)
        if info is None:
            raise RuntimeError(f"weight {name!r} not in source shard {src}")
        shape = list(info["shape"])
        dtype = info["dtype"]
        total_weight_bytes += math.prod(shape) * _dtype_bytes(dtype)

        action, group_size, bits = classify_tensor(name, shape, dtype)
        data = _tensor_bytes(src, header_len, header, name)
        if action == "copy":
            arr = _mlx_from_bytes(data, dtype, shape)
            mx.eval(arr)
            writer.add({name: arr})
            n_copy += 1
        else:
            arr = _mlx_from_bytes(data, dtype, shape)
            q, scales, biases = quantize_affine(arr, group_size=group_size, bits=bits)
            weight_name, scales_name, biases_name = quantized_tensor_names(name)
            writer.add(
                {
                    weight_name: q,
                    scales_name: scales,
                    biases_name: biases,
                },
            )
            if (group_size, bits) != (64, 4):
                quant_overrides[_module_path_for_override(name)] = {
                    "group_size": group_size,
                    "bits": bits,
                    "mode": "affine",
                }
            n_quant += 1
        del data, arr
        if peak_rss_bytes() > max_rss_bytes:
            raise RuntimeError(
                f"process footprint aborted after {name}: peak RSS "
                f"{peak_rss_bytes() / 1024**3:.1f} GiB > guard "
                f"{max_rss_bytes / 1024**3:.1f} GiB"
            )

    quant_shards = writer.finalize()

    # Canonical output index. total_size = original model byte total (loader's
    # semantic "model size"), NOT source *.safetensors file sizes (which carry
    # per-shard headers and would over-count / drift across re-bundles).
    out_index = {
        "metadata": {"total_size": writer.total_output_bytes},
        "weight_map": dict(sorted(writer.weight_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(out_index, indent=2)
    )
    aux = _copy_aux_metadata(source, output)
    _write_quantized_config(source, output, quant_overrides)
    _write_sha256sums(output)

    files = sorted(p for p in output.rglob("*") if p.is_file())
    return {
        "source": str(source),
        "output": str(output),
        "files": len(files),
        "output_bytes": sum(p.stat().st_size for p in files),
        "peak_rss_bytes": peak_rss_bytes(),
        "wall_s": round(time.monotonic() - started, 3),
        "shards": [p.name for p in files if p.name.startswith("model-")],
        "default_group_size": 64,
        "n_quant": n_quant,
        "n_copy": n_copy,
        "total_weight_bytes": total_weight_bytes,
        "total_output_weight_bytes": writer.total_output_bytes,
        "aux_copied": aux,
        "status": "ok",
    }


def convert(
    source: Path,
    output: Path,
    *,
    max_shard_bytes: int,
    min_free_bytes: int = 140 * 1024**3,
    max_rss_bytes: int = int(220.0 * 1024**3),
) -> dict:
    """Convert into a private sibling tree, publishing only a complete result."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if str(source).startswith(GUARD_EXPERT_SSD) or str(output).startswith(
        GUARD_EXPERT_SSD
    ):
        raise RuntimeError("Extreme SSD is outside this task")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(output.parent).free
    if available < min_free_bytes:
        raise RuntimeError(
            f"insufficient free space at {output.parent}: {available / 1024**3:.1f} "
            f"GiB < {min_free_bytes / 1024**3:.0f} GiB"
        )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        ledger = _convert_into(
            source,
            staging,
            max_shard_bytes=max_shard_bytes,
            min_free_bytes=min_free_bytes,
            max_rss_bytes=max_rss_bytes,
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    ledger["output"] = str(output)
    return ledger


def _renamed_quantized_aux_key(name: str) -> str:
    if name.endswith(".weight.scales"):
        return name.removesuffix(".weight.scales") + ".scales"
    if name.endswith(".weight.biases"):
        return name.removesuffix(".weight.biases") + ".biases"
    return name


def repair_quantized_aux_names(output: Path) -> dict:
    """Repair a converted tree without moving or rewriting tensor payloads.

    Safetensors permits JSON whitespace at the end of its fixed-size header.
    Renaming ``*.weight.{scales,biases}`` to sibling module parameters only
    shortens the JSON, so the original header length and every data offset stay
    unchanged.  The function fails closed on collisions or header growth.
    """
    output = output.expanduser().resolve()
    index_path = output / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing output index: {index_path}")

    # Phase 1: build and validate the complete tree-wide plan without opening
    # any model file for writing. In particular, an index collision may span
    # two different shards and must be rejected before the first header moves.
    plans: list[tuple[Path, int, bytes, bytes, int]] = []
    for shard in sorted(output.glob("model-*.safetensors")):
        with shard.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise RuntimeError(f"truncated safetensors prefix: {shard}")
            header_len = struct.unpack("<Q", raw_len)[0]
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                raise RuntimeError(f"truncated safetensors header: {shard}")
            header = json.loads(raw_header)
            renamed: dict[str, object] = {}
            shard_changes = 0
            for key, value in header.items():
                new_key = (
                    key if key == "__metadata__" else _renamed_quantized_aux_key(key)
                )
                if new_key in renamed:
                    raise RuntimeError(
                        f"quantized auxiliary rename collision in {shard}: {new_key}"
                    )
                renamed[new_key] = value
                shard_changes += int(new_key != key)
            if not shard_changes:
                continue
            encoded = json.dumps(
                renamed, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            if len(encoded) > header_len:
                raise RuntimeError(
                    f"repaired header grew in {shard}: {len(encoded)} > {header_len}"
                )
            padded = encoded + b" " * (header_len - len(encoded))
            plans.append((shard, header_len, raw_header, padded, shard_changes))

    index = json.loads(index_path.read_text())
    old_map: dict[str, str] = index.get("weight_map", {})
    if not isinstance(old_map, dict):
        raise RuntimeError("output index weight_map must be an object")
    new_map: dict[str, str] = {}
    for key, shard_name in old_map.items():
        new_key = _renamed_quantized_aux_key(key)
        if new_key in new_map:
            raise RuntimeError(f"output index rename collision: {new_key}")
        new_map[new_key] = shard_name
    index["weight_map"] = dict(sorted(new_map.items()))
    encoded_index = (json.dumps(index, indent=2) + "\n").encode()

    # Phase 2: commit the already-validated fixed-size headers, then metadata.
    # A multi-file atomic rename would duplicate the 98 GiB payload, so retain
    # the original headers/index/checksum and roll the transaction back if any
    # write fails. Every header write preserves offsets and payload bytes.
    old_index = index_path.read_bytes()
    sums_path = output / "SHA256SUMS.txt"
    old_sums = sums_path.read_bytes() if sums_path.is_file() else None
    applied: list[tuple[Path, bytes]] = []
    try:
        for shard, header_len, original, repaired, _ in plans:
            # Register rollback before the first mutable write so even a short
            # or failed header write restores this shard.
            applied.append((shard, original))
            with shard.open("r+b") as handle:
                handle.seek(8)
                handle.write(repaired)
                handle.flush()
                os.fsync(handle.fileno())
        _atomic_write_bytes(index_path, encoded_index)
        _write_sha256sums(output)
    except Exception as commit_error:
        rollback_errors = []
        for shard, original in reversed(applied):
            try:
                with shard.open("r+b") as handle:
                    handle.seek(8)
                    handle.write(original)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception as rollback_error:  # pragma: no cover - I/O failure
                rollback_errors.append(f"{shard}: {rollback_error}")
        try:
            _atomic_write_bytes(index_path, old_index)
            if old_sums is None:
                sums_path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(sums_path, old_sums)
        except Exception as rollback_error:  # pragma: no cover - I/O failure
            rollback_errors.append(f"metadata: {rollback_error}")
        if rollback_errors:  # pragma: no cover - double I/O failure
            raise RuntimeError(
                "quantized auxiliary repair failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from commit_error
        raise

    changed_keys = sum(plan[4] for plan in plans)
    payload_checks = {
        shard.name: (str(header_len), str(shard.stat().st_size))
        for shard, header_len, _, _, _ in plans
    }
    return {
        "status": "ok",
        "output": str(output),
        "changed_keys": changed_keys,
        "changed_shards": len(plans),
        "unchanged_payload_layout": payload_checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    ap.add_argument(
        "--inspect-only",
        action="store_true",
        help="validate and classify all headers without creating output",
    )
    ap.add_argument(
        "--repair-quantized-aux-names",
        action="store_true",
        help="repair an existing output tree's MLX quantized parameter names",
    )
    args = ap.parse_args()
    try:
        if args.inspect_only:
            print(json.dumps(inspect_manifest(args.source), indent=2))
            return 0
        if args.repair_quantized_aux_names:
            if args.output is not None:
                ap.error(
                    "--repair-quantized-aux-names uses --source as the output tree"
                )
            print(json.dumps(repair_quantized_aux_names(args.source), indent=2))
            return 0
        if args.output is None:
            ap.error("--output is required unless --inspect-only is set")
        ledger = convert(
            args.source,
            args.output,
            max_shard_bytes=args.max_shard_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"CONVERT FAILED (fail-closed): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
