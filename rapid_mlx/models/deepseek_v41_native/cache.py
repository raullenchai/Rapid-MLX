"""Incremental state for DeepSeek-V4.1 decode and chunked prefill.

Per layer:

* **window ring** — every layer: a circular buffer of ``window_size`` KV
  entries (position p at slot p % window), values already FP8 fake-quantized;
* **compressed KV** — kv_source layers only: one FP4-fake-quantized latent per
  complete group, append-only. Consumer layers hold a *reference* to their
  source's buffer, so a source's write is immediately visible downstream;
* **compressor partial group** — ratio>1 sources: fp32 kv/score rows of the
  open group;
* **index keys** — layers that are both kv and index sources: the FP4-fake-
  quantized index-key cache, read by every index source below them.

Model-level: the engram compressed-token-id history, and the global offset.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from .compressor import CompressorState
from .config import ModelArgs


class LayerCache:
    def __init__(
        self,
        bsz: int,
        args: ModelArgs,
        layer_id: int,
        max_seq_len: int,
        dtype=mx.float32,
    ):
        self.window = args.window_size
        self.ratio = args.compress_ratio(layer_id)
        self.is_kv_source = layer_id in args.kv_source_layers
        self.dtype = dtype

        self.win_kv = mx.zeros((bsz, self.window, args.head_dim), dtype=dtype)

        self.comp_kv: Any | None = None
        self.comp_state: CompressorState | None = None
        self.index_k: Any | None = None
        # Per-row committed group counts; set only on batched decode caches.
        self.comp_counts: Any | None = None
        if self.is_kv_source:
            n_comp = max_seq_len // self.ratio
            self.comp_kv = mx.zeros((bsz, n_comp, args.head_dim), dtype=dtype)
            if self.ratio > 1:
                self.comp_state = CompressorState(bsz, self.ratio, args.head_dim)
            if layer_id in args.index_source_layers:
                self.index_k = mx.zeros((bsz, n_comp, args.index_head_dim), dtype=dtype)

    # ---- window ring ----

    def window_chrono(self, pos: int) -> mx.array:
        """The cached window KV in chronological order: positions
        [pos - Wp, pos) where Wp = min(pos, window). [b, Wp, head_dim]."""
        w = self.window
        wp = min(pos, w)
        if wp == 0:
            return self.win_kv[:, :0]
        first = pos - wp
        slots = (first + mx.arange(wp)) % w
        return self.win_kv[:, slots]

    def write_window(self, pos: int, kv: mx.array):
        """Write chunk KV at positions [pos, pos+n) into the ring."""
        n = kv.shape[1]
        if n == 1:  # decode: one slice update, no index arrays
            slot = pos % self.window
            self.win_kv[:, slot : slot + 1] = kv.astype(self.dtype)
            return
        keep = min(n, self.window)
        tail = kv[:, n - keep :]
        slots = (pos + n - keep + mx.arange(keep)) % self.window
        self.win_kv[:, slots] = tail.astype(self.dtype)


class ModelCache:
    def __init__(
        self,
        args: ModelArgs,
        bsz: int = 1,
        max_seq_len: int | None = None,
        dtype=mx.float32,
    ):
        self.args = args
        self.bsz = bsz
        self.max_seq_len = max_seq_len or min(args.max_seq_len, 4096)
        self.offset = 0
        self.rollback_start: int | None = None
        self.layers = [
            LayerCache(bsz, args, i, self.max_seq_len, dtype)
            for i in range(args.n_layers)
        ]
        self.engram_ids = (
            np.zeros((bsz, self.max_seq_len), dtype=np.int64)
            if args.engram_layer_ids
            else None
        )

    def rollback(self, offset: int) -> None:
        """Discard a suffix from the most recent speculative target call."""
        if (
            self.rollback_start is None
            or not self.rollback_start <= offset <= self.offset
        ):
            raise ValueError(
                f"rollback offset {offset} outside latest forward "
                f"[{self.rollback_start}, {self.offset}]"
            )
        for layer in self.layers:
            if layer.comp_state is not None:
                layer.comp_state.rollback(offset)
        self.offset = offset
        self.disable_rollback()

    def begin_forward(self) -> None:
        """Open the only rollback window supported by this cache."""
        self.rollback_start = self.offset
        snapshots: list[Any] = []
        for layer in self.layers:
            if layer.comp_state is not None:
                layer.comp_state.begin_forward(self.offset)
                snapshots.extend(
                    (layer.comp_state.snapshot_kv, layer.comp_state.snapshot_score)
                )
        if snapshots:
            mx.eval(*snapshots)

    def disable_rollback(self) -> None:
        """Close any prior rollback window without synchronizing device work."""
        self.rollback_start = None
        for layer in self.layers:
            if layer.comp_state is not None:
                layer.comp_state.disable_rollback()


class BatchModelCache:
    """Decode-only rows at independent positions, without physical left padding.

    Buffers have a leading row axis; compressed buffers use the largest sequence
    capacity. ``offsets`` is host metadata and ``positions`` its device mirror.
    Merge/join/compaction copy state and never alias a caller's mutable cache.
    ``filter`` and ``extend`` mutate this cache and return it for convenience.
    """

    args: ModelArgs
    max_seq_len: int
    offsets: np.ndarray
    positions: mx.array
    layers: list[LayerCache]
    engram_ids: np.ndarray | None

    @classmethod
    def from_caches(
        cls, caches: list[ModelCache], max_seq_len: int | None = None
    ) -> BatchModelCache:
        if not caches:
            raise ValueError("from_caches requires at least one row")
        args = caches[0].args
        dtype = caches[0].layers[0].dtype if args.n_layers else mx.float32
        for cache in caches:
            if cache.args != args:
                raise ValueError(
                    "cannot batch caches with different model configurations"
                )
            if cache.bsz != 1:
                raise ValueError("from_caches accepts only single-row caches")
            if cache.engram_ids is not None and cache.engram_ids.shape[0] != 1:
                raise ValueError("from_caches accepts only single-row histories")
            if any(layer.dtype != dtype for layer in cache.layers):
                raise ValueError("cannot batch caches with different dtypes")
        capacity = (
            max_seq_len
            if max_seq_len is not None
            else max(cache.max_seq_len for cache in caches)
        )
        if capacity <= 0 or any(not 0 <= cache.offset <= capacity for cache in caches):
            raise ValueError("sequence capacity must cover every row's offset")
        result = cls()
        result.args, result.max_seq_len = args, capacity
        result.offsets = np.array([cache.offset for cache in caches], dtype=np.int64)
        result.positions = mx.array(result.offsets, dtype=mx.int32)
        result.layers = [
            LayerCache(len(caches), args, i, capacity, dtype)
            for i in range(args.n_layers)
        ]
        for i, target in enumerate(result.layers):
            target.win_kv = mx.concatenate([c.layers[i].win_kv for c in caches])
            target.comp_counts = None
            if target.comp_kv is not None:
                target.comp_counts = result.positions // target.ratio
                for row, source in enumerate(c.layers[i] for c in caches):
                    # Equal configurations allocate the same optional buffers.
                    assert source.comp_kv is not None
                    count = min(source.comp_kv.shape[1], target.comp_kv.shape[1])
                    target.comp_kv[row : row + 1, :count] = source.comp_kv[:, :count]
                    if target.index_k is not None:
                        assert source.index_k is not None
                        target.index_k[row : row + 1, :count] = source.index_k[
                            :, :count
                        ]
                if target.comp_state is not None:
                    for name in ("kv_state", "score_state"):
                        setattr(
                            target.comp_state,
                            name,
                            mx.concatenate(
                                [getattr(c.layers[i].comp_state, name) for c in caches]
                            ),
                        )
        result.engram_ids = None
        if args.engram_layer_ids:
            result.engram_ids = np.zeros((len(caches), capacity), dtype=np.int64)
            for row, cache in enumerate(caches):
                assert cache.engram_ids is not None
                count = min(capacity, cache.max_seq_len)
                result.engram_ids[row, :count] = cache.engram_ids[0, :count]
        return result

    def __len__(self):
        return len(self.offsets)

    def filter(self, keep_rows):
        """Keep integer row indices in the supplied order, including no rows."""
        rows = np.asarray(keep_rows)
        if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
            raise ValueError("keep_rows must be a vector of integer row indices")
        rows = rows.astype(np.int64)
        if np.any(rows < 0) or np.any(rows >= len(self)):
            raise IndexError("row outside batch")
        indices = mx.array(rows, dtype=mx.int32)
        self.offsets = self.offsets[rows].copy()
        self.positions = mx.array(self.offsets, dtype=mx.int32)
        if self.engram_ids is not None:
            self.engram_ids = self.engram_ids[rows].copy()
        for layer in self.layers:
            for name in ("win_kv", "comp_kv", "index_k", "comp_counts"):
                value = getattr(layer, name)
                if value is not None:
                    setattr(layer, name, value[indices])
            if layer.comp_state is not None:
                for name in ("kv_state", "score_state"):
                    setattr(
                        layer.comp_state, name, getattr(layer.comp_state, name)[indices]
                    )
        return self

    def extend(self, other: BatchModelCache):
        """Append rows, growing to the larger sequence capacity."""
        if self.args != other.args:
            raise ValueError("cannot join caches with different model configurations")
        if not len(other):
            return self
        rows = [self.extract(i) for i in range(len(self))]
        rows.extend(other.extract(i) for i in range(len(other)))
        merged = self.from_caches(rows, max(self.max_seq_len, other.max_seq_len))
        self.__dict__.update(merged.__dict__)
        return self

    def extract(self, row: int) -> ModelCache:
        """Return an independent single-stream cache at the current position."""
        if not 0 <= row < len(self):
            raise IndexError("row outside batch")
        dtype = self.layers[0].dtype if self.layers else mx.float32
        result = ModelCache(self.args, 1, self.max_seq_len, dtype)
        result.offset = int(self.offsets[row])
        for source, target in zip(self.layers, result.layers):
            for name in ("win_kv", "comp_kv", "index_k"):
                value = getattr(source, name)
                if value is not None:
                    setattr(target, name, mx.array(value[row : row + 1]))
            if source.comp_state is not None:
                for name in ("kv_state", "score_state"):
                    setattr(
                        target.comp_state,
                        name,
                        mx.array(getattr(source.comp_state, name)[row : row + 1]),
                    )
        if self.engram_ids is not None:
            result.engram_ids = self.engram_ids[row : row + 1].copy()
        return result

    def advance(self):
        self.offsets += 1
        self.positions = mx.array(self.offsets, dtype=mx.int32)

    def begin_forward(self):
        raise NotImplementedError(
            "batched decode does not support speculative rollback"
        )

    def rollback(self, offset):
        raise NotImplementedError(
            "batched decode does not support speculative rollback"
        )
