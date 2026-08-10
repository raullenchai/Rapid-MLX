# SPDX-License-Identifier: Apache-2.0
"""Byte-offset safetensors reader for one MoE expert's weight bundle.

Ticket: ``.scratch/rapid-mlx-disk-stream/issues/01-registry-offset-reader-lfm25.md``.
Productionized from the spike's already bit-exact-verified
``.scratch/moe-disk-stream/scripts/02_offset_reader.py`` — same header
parsing, same byte-range math, same bf16 ``mx.view`` bit-reinterpret
trick — generalized to be driven by a
:class:`~vllm_mlx.registry.StreamingAdapter`'s tensor-naming template
instead of hardcoded LFM2.5 tensor names. Ticket 05
(``.scratch/rapid-mlx-disk-stream/issues/05-qwen2-moe-shared-expert-arch.md``)
added the ``"direct"``-layout branch (qwen2_moe-style,
per-expert-named-tensor, see :func:`fetch_expert_bundle`) reusing this
same module and its no-caching discipline.

No caching at this layer, by design (PRD: caching is ``expert_cache.py``,
a separate module/ticket) — every call does a fresh ``open()``/``seek()``/
``read()`` per tensor.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from vllm_mlx.registry import StreamingAdapter

# safetensors dtype string -> numpy dtype to read the raw bytes as. BF16 has
# no native numpy dtype: read as uint16 and bit-reinterpret via mx.view.
_DTYPE_MAP: dict[str, type] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def _read_header(path: Path) -> tuple[dict, int]:
    """Parse only the safetensors JSON header (8-byte little-endian length
    prefix + that many bytes of JSON) — never reads tensor data. Reopens
    the file every call; nothing is cached/memoized across fetches.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def _fetch_tensor_slice(
    path: Path,
    tensor_name: str,
    expert_id: int | None = None,
) -> mx.array:
    """Fetch one tensor (or, if ``expert_id`` is given, one axis-0 slice of
    a stacked-experts tensor) via a single seek+read of exactly its byte
    range. No ``mx.load``, no whole-file safetensors load, no caching.
    """
    header, data_start = _read_header(path)  # re-read + re-parse every call
    meta = header[tensor_name]
    dtype_str = meta["dtype"]
    shape = list(meta["shape"])
    start, end = meta["data_offsets"]

    if expert_id is not None:
        num_experts = shape[0]
        if (end - start) % num_experts != 0:
            raise ValueError(f"{tensor_name}: not evenly stacked on axis 0")
        per_expert_nbytes = (end - start) // num_experts
        start = start + expert_id * per_expert_nbytes
        end = start + per_expert_nbytes
        shape = shape[1:]

    np_dtype = _DTYPE_MAP[dtype_str]
    nbytes = end - start
    expected_nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
    if nbytes != expected_nbytes:
        raise ValueError(
            f"{tensor_name}: byte-range size {nbytes} != expected "
            f"{expected_nbytes} for shape {shape}"
        )

    with open(path, "rb") as f:  # fresh handle every call — no caching
        f.seek(data_start + start)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"{tensor_name}: short read: got {len(raw)}, wanted {nbytes}")

    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    mx_arr = mx.array(arr)
    if dtype_str == "BF16":
        mx_arr = mx.view(mx_arr, mx.bfloat16)  # bit-reinterpret uint16 -> bfloat16
    return mx_arr


def _resolve_shard_path(path: Path, tensor_name: str) -> Path:
    """Resolve the concrete safetensors file holding ``tensor_name``, for
    both the ``"stacked"`` and ``"direct"`` layouts — the directory-vs-file
    resolution is identical either way. A safetensors tensor is never split
    across shards, so this works unmodified for ``"stacked"``'s stacked
    tensor name too (if a ``"stacked"``-layout checkpoint were ever sharded,
    the whole stacked tensor — all experts included — would still live in
    exactly one shard, keyed by its own name in ``weight_map`` just like any
    other tensor).

    ``path`` may be:

    * a single safetensors file — used as-is (e.g. a single-shard test
      fixture).
    * a checkpoint *directory* — the shape
      ``vllm_mlx.utils.tokenizer._resolve_model_path`` hands
      ``disk_stream_patch.install`` in the real CLI flow. Resolved via
      that directory's ``model.safetensors.index.json`` ``weight_map``
      when present (qwen2_moe's real checkpoint ships 2 shards), else
      ``path/model.safetensors`` (single-file checkpoint in directory form).
    """
    if path.is_file():
        return path
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(f"{tensor_name!r} not found in {index_path}'s weight_map")
        # shard_name comes straight from the checkpoint's own (untrusted)
        # index.json. `path / shard_name` alone is not safe: if shard_name
        # is absolute, pathlib's `/` discards `path` entirely, and a
        # "../"-prefixed relative value walks out of the checkpoint
        # directory — either way the caller below would happily open a
        # file outside `path`.
        #
        # Reject traversal *lexically* on shard_name itself -- do NOT
        # `.resolve()` the candidate and compare it against a resolved
        # `path` (`is_relative_to`). Every real HuggingFace-cache
        # checkpoint directory is made entirely of symlinks into a
        # shared `blobs/` store one level up
        # (`snapshots/<sha>/model-...safetensors -> ../../blobs/<hash>`),
        # so resolving symlinks walks every legitimate shard "outside"
        # its own snapshot directory and rejects every real cached
        # checkpoint (confirmed against the actual local HF cache layout
        # for mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit). The actual
        # attack surface is the *shard_name string* -- an absolute path,
        # or a relative path containing a ".." component -- which a pure
        # parts-based check catches without ever touching the
        # filesystem's real symlink targets.
        shard_path = Path(shard_name)
        if shard_path.is_absolute() or ".." in shard_path.parts:
            raise ValueError(
                f"refusing to resolve tensor {tensor_name!r}: index.json "
                f"maps it to shard_name={shard_name!r}, which is an "
                f"absolute path or escapes the checkpoint directory via "
                f"'..'"
            )
        return path / shard_name
    return path / "model.safetensors"


def fetch_expert_bundle(
    adapter: StreamingAdapter,
    path: str | Path,
    layer_idx: int,
    expert_id: int,
) -> dict[str, dict[str, mx.array]]:
    """Fetch one routed expert's full 9-tensor weight bundle straight off
    ``path``, driven by ``adapter.tensor_template``.

    Returns ``{proj: {component: mx.array}}`` for every
    ``adapter.sub_projections`` x ``adapter.tensor_components`` pair — 9
    tensors for the default 3x3 (gate/up/down x weight/scales/biases).

    Two layouts are implemented:

    * ``"stacked"`` (LFM2.5-style) — every sub-projection/component is one
      on-disk tensor holding all experts stacked on axis 0; one expert is
      a contiguous byte-range slice computed from ``data_offsets`` and
      ``expert_id``.
    * ``"direct"`` (qwen2_moe-style) — each expert has its own separately
      named tensor per component; no byte-offset slicing, just 9 direct
      whole-tensor lookups by name (via
      ``adapter.tensor_template.tensor_name(..., expert_id=expert_id)``).

    Both layouts accept ``path`` as either a single safetensors file or a
    checkpoint directory (the shape the real CLI's
    ``_resolve_model_path`` -> ``disk_stream_patch.install`` flow actually
    hands in) — see :func:`_resolve_shard_path`.

    Any other ``adapter.tensor_template.layout`` value raises
    ``NotImplementedError``.
    """
    layout = adapter.tensor_template.layout
    if layout not in ("stacked", "direct"):
        raise NotImplementedError(
            f"offset_reader only implements the 'stacked' and 'direct' "
            f"tensor layouts; {adapter.model_type!r} declares layout="
            f"{layout!r}"
        )

    path = Path(path)
    bundle: dict[str, dict[str, mx.array]] = {}
    for proj in adapter.sub_projections:
        bundle[proj] = {}
        for component in adapter.tensor_components:
            if layout == "stacked":
                tensor_name = adapter.tensor_template.tensor_name(layer_idx, proj, component)
                shard_path = _resolve_shard_path(path, tensor_name)
                bundle[proj][component] = _fetch_tensor_slice(
                    shard_path, tensor_name, expert_id=expert_id
                )
            else:  # "direct"
                tensor_name = adapter.tensor_template.tensor_name(
                    layer_idx, proj, component, expert_id=expert_id
                )
                shard_path = _resolve_shard_path(path, tensor_name)
                bundle[proj][component] = _fetch_tensor_slice(
                    shard_path, tensor_name, expert_id=None
                )
    return bundle
