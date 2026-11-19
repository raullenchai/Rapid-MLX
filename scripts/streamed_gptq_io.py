"""Architecture-independent bounded tensor I/O and restart receipts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from scripts.deepseek_v41_rd_common import file_hash, read_slices
from scripts.repack_deepseek_v41_native import _DTYPE_BYTES, CheckpointIndex


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def artifact_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".gptq.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("conversion artifact is busy") from None
        yield


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TensorReader:
    """Validate local paths and payload bounds before mapping a tensor slice."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        weight_map = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )["weight_map"]
        for name in set(weight_map.values()):
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe checkpoint shard path")
        self.index = CheckpointIndex(self.root)
        for tensor in self.index.tensors.values():
            if (
                tensor.offset < 8
                or tensor.offset + tensor.nbytes > tensor.path.stat().st_size
            ):
                raise ValueError("truncated checkpoint tensor")

    def raw(self, name, *, expert=None, rows=None):
        t = self.index.tensors[name]
        dtype = {
            "BF16": "<u2",
            "F8_E4M3": "u1",
            "F8_E4M3FN": "u1",
            "F16": "<f2",
            "F32": "<f4",
            "I32": "<i4",
            "I64": "<i8",
            "U32": "<u4",
        }.get(t.dtype)
        if dtype is None:
            raise ValueError(f"unsupported tensor dtype: {t.dtype}")
        mapped = np.memmap(
            t.path, mode="r", dtype=dtype, offset=t.offset, shape=t.shape
        )
        selected = mapped
        if expert is not None:
            if len(t.shape) < 2 or not 0 <= expert < t.shape[0]:
                raise ValueError("invalid expert slice")
            selected = selected[expert]
        if rows is not None:
            start, stop = rows
            if (
                selected.ndim not in (1, 2)
                or not 0 <= start < stop <= selected.shape[0]
            ):
                raise ValueError("invalid row slice")
            selected = selected[start:stop]
        value = np.array(selected)
        del selected, mapped
        return value

    def array(self, name, **selection):
        import mlx.core as mx

        raw = self.raw(name, **selection)
        value = mx.array(raw)
        return (
            value.view(mx.bfloat16)
            if self.index.tensors[name].dtype == "BF16"
            else value
        )


class RangeSink:
    """Sequential leading-axis writes with exact dtype, shape and coverage checks."""

    def __init__(self, path, entries):
        self.path = Path(path)
        self.temporary = self.path.with_suffix(".partial")
        self.entries = entries
        self.positions = dict.fromkeys(entries, 0)
        self.offsets = {}
        header, offset = {}, 0
        for name, (dtype, shape) in entries.items():
            size = math.prod(shape) * _DTYPE_BYTES[dtype]
            header[name] = dict(
                dtype=dtype, shape=shape, data_offsets=[offset, offset + size]
            )
            self.offsets[name] = offset
            offset += size
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        self.file = self.temporary.open("w+b")
        self.file.write(struct.pack("<Q", len(raw)))
        self.file.write(raw)
        self.base = 8 + len(raw)
        self.file.truncate(self.base + offset)

    def put(self, name, start, value):
        dtype, shape = self.entries[name]
        raw = np.asarray(value)
        expected_dtype = {
            "BF16": np.dtype("uint16"),
            "F32": np.dtype("float32"),
            "U32": np.dtype("uint32"),
            "I32": np.dtype("int32"),
            "I64": np.dtype("int64"),
            "F16": np.dtype("float16"),
        }[dtype]
        if (
            raw.dtype != expected_dtype
            or raw.ndim != len(shape)
            or raw.shape[1:] != tuple(shape[1:])
        ):
            raise ValueError("output dtype or shape mismatch")
        stop = start + len(raw)
        if start != self.positions[name] or not start < stop <= shape[0]:
            raise ValueError("nonsequential or duplicate output range")
        row_bytes = math.prod(shape[1:]) * _DTYPE_BYTES[dtype]
        self.file.seek(self.base + self.offsets[name] + start * row_bytes)
        self.file.write(raw.tobytes())
        self.positions[name] = stop

    def finish(self):
        if any(self.positions[n] != s[0] for n, (_, s) in self.entries.items()):
            raise ValueError("incomplete output tensor")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        self.temporary.replace(self.path)

    def close(self):
        self.file.close()


def verify_receipt(root, receipt, identity):
    if receipt["identity"] != identity:
        raise ValueError("resume identity mismatch")
    referenced = set(receipt.get("weight_map", {}).values()) | set(
        receipt.get("boundaries", [])
    )
    if not referenced.issubset(receipt["files"]):
        raise ValueError("unbound resume artifact")
    for name, expected in receipt["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe receipt path")
        if file_hash(Path(root) / path) != expected:
            raise ValueError("resume artifact mismatch")


def slice_evidence(manifest):
    paths = read_slices(Path(manifest), [])
    hashes = {name: [file_hash(p) for p in files] for name, files in paths.items()}
    return paths, hashes


def mx_module():
    import mlx.core as mx

    return mx


def load_boundary(path):
    return mx_module().load(str(path))["hidden"]


def save_boundary(path, hidden):
    mx = mx_module()
    mx.eval(hidden)
    temporary = path.with_suffix(".partial.safetensors")
    mx.save_safetensors(str(temporary), {"hidden": hidden})
    temporary.replace(path)
