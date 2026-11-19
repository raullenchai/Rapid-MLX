"""Selected-row affine embeddings shared by native Engram runtimes."""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_MAX_SAFETENSORS_HEADER_SIZE = 100_000_000


class DiskQuantizedEngramEmbedding(nn.Module):
    """Selected-row affine Engram reader backed by safetensors mmaps.

    ``path`` is one shard holding all three affine tensors, or a mapping from
    each tensor key to its shard (a 4-bit table can exceed hub file limits).

    Only packed rows requested by the current n-gram hashes cross into MLX.
    A bounded raw-row LRU absorbs repeated decode lookups without retaining a
    device-sized view of the embedding table.
    """

    def __init__(
        self,
        path: str | Path | Mapping[str, str | Path],
        *,
        weight_key: str,
        scales_key: str,
        biases_key: str,
        num_embeddings: int,
        dim: int,
        group_size: int,
        bits: int,
        cache_rows: int = 16384,
    ):
        super().__init__()
        if bits not in (2, 3, 4, 6, 8):
            raise ValueError(f"unsupported affine Engram width: {bits}")
        if cache_rows < 0:
            raise ValueError("Engram cache_rows cannot be negative")
        self.group_size, self.bits = group_size, bits
        self.cache_rows = cache_rows
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = RLock()
        self._closed = False
        self._executor: ThreadPoolExecutor | None = None
        self._pending: tuple[np.ndarray, Future] | None = None
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._shards: list[tuple] = []
        self._weight = self._scales = self._biases = None
        try:
            opened: dict[Path, tuple] = {}

            def shard(key):
                source = Path(path[key] if isinstance(path, Mapping) else path)
                if source not in opened:
                    opened[source] = self._open_shard(source)
                return opened[source]

            self._weight = self._tensor_view(
                shard(weight_key),
                weight_key,
                dtype="U32",
                numpy_dtype=np.dtype("<u4"),
                shape=(num_embeddings, dim * bits // 32),
            )
            quant_shape = (num_embeddings, dim // group_size)
            self._scales = self._tensor_view(
                shard(scales_key),
                scales_key,
                dtype="BF16",
                numpy_dtype=np.dtype("<u2"),
                shape=quant_shape,
            )
            self._biases = self._tensor_view(
                shard(biases_key),
                biases_key,
                dtype="BF16",
                numpy_dtype=np.dtype("<u2"),
                shape=quant_shape,
            )
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rapid-engram"
            )
        except Exception:
            self._weight = self._scales = self._biases = None
            self._close_shards()
            raise

    def _open_shard(self, source: Path) -> tuple:
        """Open one shard; return ``(header, mapping, data_start)``."""
        file = source.open("rb")
        self._shards.append((file, None))
        length_raw = file.read(8)
        if len(length_raw) != 8:
            raise ValueError("truncated safetensors header")
        header_length = struct.unpack("<Q", length_raw)[0]
        file_size = file.seek(0, 2)
        if (
            header_length > _MAX_SAFETENSORS_HEADER_SIZE
            or header_length > file_size - 8
        ):
            raise ValueError("invalid safetensors header length")
        file.seek(8)
        header = json.loads(file.read(header_length))
        data_start = 8 + header_length
        self._validate_header_ranges(header, file_size - data_start)
        mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self._shards[-1] = (file, mapping)
        if hasattr(mapping, "madvise"):
            mapping.madvise(mmap.MADV_RANDOM)
        return header, mapping, data_start

    def _close_shards(self) -> None:
        shards, self._shards = self._shards, []
        for file, mapping in shards:
            try:
                if mapping is not None:
                    mapping.close()
            finally:
                file.close()

    @staticmethod
    def _validate_header_ranges(header, data_size: int) -> None:
        if not isinstance(header, dict):
            raise ValueError("invalid safetensors header")
        ranges = []
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"invalid safetensors tensor entry: {key}")
            offsets = entry.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"invalid safetensors tensor offsets: {key}")
            start, end = offsets
            if start < 0 or end < start or end > data_size:
                raise ValueError(f"invalid safetensors tensor offsets: {key}")
            ranges.append((start, end, key))
        ranges.sort()
        expected_start = 0
        for start, end, key in ranges:
            if start != expected_start:
                raise ValueError(f"non-contiguous safetensors tensor offsets: {key}")
            expected_start = end
        if expected_start != data_size:
            raise ValueError("safetensors tensor offsets do not cover the data buffer")

    def _tensor_view(self, shard, key, *, dtype, numpy_dtype, shape):
        header, mapping, data_start = shard
        entry = header.get(key)
        if entry is None or entry.get("dtype") != dtype:
            raise ValueError(f"invalid or missing Engram tensor: {key}")
        if tuple(entry.get("shape", ())) != shape:
            raise ValueError(f"invalid Engram tensor shape: {key}")
        start, end = entry.get("data_offsets", (-1, -1))
        expected = math.prod(shape) * numpy_dtype.itemsize
        if start < 0 or end - start != expected or data_start + end > len(mapping):
            raise ValueError(f"invalid Engram tensor byte length: {key}")
        return np.ndarray(
            shape,
            dtype=numpy_dtype,
            buffer=mapping,
            offset=data_start + start,
        )

    @staticmethod
    def _bfloat16(raw: np.ndarray) -> mx.array:
        values = (raw.astype(np.uint32) << 16).view(np.float32)
        return mx.array(values).astype(mx.bfloat16)

    def _dequantize(self, rows) -> mx.array:
        weights, scales, biases = rows
        return mx.dequantize(
            mx.array(weights),
            self._bfloat16(scales),
            self._bfloat16(biases),
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        ).astype(mx.float32)

    def _views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._weight is None or self._scales is None or self._biases is None:
            raise RuntimeError("Engram embedding is closed")
        return self._weight, self._scales, self._biases

    def _raw_rows(self, flat: np.ndarray):
        weight, scales, biases = self._views()
        return (
            np.array(weight[flat], copy=True),
            np.array(scales[flat], copy=True),
            np.array(biases[flat], copy=True),
        )

    def _gather_rows(self, flat: np.ndarray):
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram embedding is closed")
            weight, _, _ = self._views()
            if flat.size and (flat.min() < 0 or flat.max() >= weight.shape[0]):
                raise IndexError("Engram row outside table")
            if not flat.size or not self.cache_rows or flat.size > 4096:
                self.cache_misses += len(set(int(row) for row in flat))
                rows = self._raw_rows(flat)
            else:
                unique = list(dict.fromkeys(int(row) for row in flat))
                resolved = {
                    row: self._cache[row] for row in unique if row in self._cache
                }
                missing = [row for row in unique if row not in resolved]
                self.cache_misses += len(missing)
                if missing:
                    loaded = self._raw_rows(np.asarray(missing, dtype=np.int64))
                    for index, row in enumerate(missing):
                        value = tuple(value[index].copy() for value in loaded)
                        resolved[row] = value
                        self._cache[row] = value
                        while len(self._cache) > self.cache_rows:
                            self._cache.popitem(last=False)
                gathered = [resolved[int(row)] for row in flat]
                for row in unique:
                    if row in self._cache:
                        self._cache.move_to_end(row)
                self.cache_hits += flat.size - len(missing)
                rows = tuple(
                    np.stack([value[column] for value in gathered])
                    for column in range(3)
                )
        return rows

    @staticmethod
    def _discard_future(future: Future) -> None:
        def consume_error(done: Future) -> None:
            try:
                done.exception()
            except CancelledError:  # pragma: no cover - cancellation race
                pass

        future.add_done_callback(consume_error)
        future.cancel()

    def prefetch(self, indices: np.ndarray) -> None:
        """Start selected-row I/O before this embedding's layer executes."""
        flat = np.asarray(indices, dtype=np.int64).reshape(-1).copy()
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram embedding is closed")
            previous, self._pending = self._pending, None
        if previous is not None:
            self._discard_future(previous[1])
        executor = self._executor
        if executor is None:  # pragma: no cover - concurrent close
            raise RuntimeError("Engram embedding is closed")
        future = executor.submit(self._gather_rows, flat)
        with self._lock:
            if self._closed:  # pragma: no cover - concurrent close
                future.cancel()
                raise RuntimeError("Engram embedding is closed")
            self._pending = (flat, future)

    def __call__(self, indices: mx.array) -> mx.array:
        host = np.asarray(indices, dtype=np.int64)
        flat = host.reshape(-1)
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            requested, future = pending
            if np.array_equal(requested, flat):
                rows = future.result()
            else:
                self._discard_future(future)
                rows = self._gather_rows(flat)
        else:
            rows = self._gather_rows(flat)
        values = self._dequantize(rows)
        return values.reshape(*host.shape, values.shape[-1])

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending, self._pending = self._pending, None
        if pending is not None:
            pending[1].cancel()
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._cache.clear()
            self._weight = self._scales = self._biases = None
            self._close_shards()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def resolve_indexed_shard(model_path: str, filename: str) -> str:
    """Resolve an indexed shard without escaping its model/CAS repository."""
    root = Path(model_path).resolve()
    lexical = Path(os.path.abspath(os.path.join(root, filename)))
    if os.path.commonpath((root, lexical)) != str(root):
        raise ValueError("Engram shard must stay inside the model directory")
    resolved = lexical.resolve(strict=True)
    if os.path.commonpath((root, resolved)) == str(root):
        return str(resolved)
    # Hub snapshots legitimately symlink immutable files into the specific
    # repository's sibling blobs directory. Recognize the repository layout,
    # rather than trusting any parent directory merely named "snapshots".
    repository = root.parent.parent
    if root.parent.name == "snapshots" and repository.name.startswith("models--"):
        lexical_blobs = repository / "blobs"
        blobs = lexical_blobs.resolve()
        if (
            lexical_blobs.is_dir()
            and not lexical_blobs.is_symlink()
            and os.path.commonpath((repository, blobs)) == str(repository)
            and os.path.commonpath((blobs, resolved)) == str(blobs)
        ):
            return str(resolved)
    raise ValueError("Engram shard symlink escapes the model repository")
