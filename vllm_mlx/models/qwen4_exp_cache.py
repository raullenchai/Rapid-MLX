# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp cache types for the MLX text decoder.

The cache follows the engine request-lifecycle contract: a fixed raw-key ring
feeds one compressed key per complete group, while the compressed cache is
owned independently from the main attention KV cache. It subclasses
``ArraysCache`` so mlx-lm's established batching path can prepare, merge,
filter, extend, and extract it without a model-specific scheduler patch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import mlx.core as mx

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import ArraysCache  # noqa: E402


class Qwen4ExpStateCache(ArraysCache):
    """Recurrent Qwen4 state with speculative-verify restore points.

    The four-slot PLE layer cache couples GDN convolution/state with PLE
    convolution/ngram history. A rejected speculative token must restore all
    four slots to the same accepted boundary; restoring GDN alone silently
    desynchronizes later PLE inputs.

    ``ArraysCache.extract`` constructs the base class explicitly. Override it
    here so continuous batching and prefix-cache reuse retain this cache's
    atomic rollback contract when a single request leaves a batch.
    """

    rollback_state: list[list[mx.array | None]] | None = None
    _rollback_slots: dict[int, list[mx.array]] | None = None

    def extract(self, idx):
        cache = type(self)(len(self.cache))
        cache.cache = [
            None if item is None else mx.contiguous(item[idx : idx + 1])
            for item in self.cache
        ]
        if self.left_padding is not None:
            cache.left_padding = mx.contiguous(self.left_padding[idx : idx + 1])
        if self.lengths is not None:
            cache.lengths = mx.contiguous(self.lengths[idx : idx + 1])
        return cache

    def record_slot_snapshots(
        self,
        slot: int,
        snapshots: list[mx.array],
        *,
        finalize: bool = False,
    ) -> None:
        """Stage per-position recurrent state and publish atomic boundaries."""
        if not snapshots:
            return
        if self._rollback_slots is None:
            self._rollback_slots = {}
        self._rollback_slots[slot] = snapshots
        if not finalize:
            return
        expected_slots = set(range(len(self.cache)))
        if set(self._rollback_slots) != expected_slots:
            raise AssertionError(
                "Qwen4 speculative cache snapshots do not cover every state slot"
            )
        lengths = {len(items) for items in self._rollback_slots.values()}
        if len(lengths) != 1:
            raise AssertionError("Qwen4 speculative cache snapshot lengths diverged")
        count = lengths.pop()
        self.rollback_state = [
            [self._rollback_slots[slot][position] for slot in range(len(self.cache))]
            for position in range(count)
        ]
        self._rollback_slots = None

    def restore_rollback(self, n_to_drop: int, verify_size: int) -> None:
        snapshots = self.rollback_state
        if not snapshots:
            raise AssertionError("Qwen4 verify rollback has no saved boundary")
        keep = verify_size - n_to_drop
        if keep < 1 or keep > len(snapshots):
            raise AssertionError(
                f"invalid Qwen4 rollback boundary: keep={keep}, "
                f"snapshots={len(snapshots)}"
            )
        self.cache = list(snapshots[keep - 1])
        self.rollback_state = None
        self._rollback_slots = None


class QSAIndexCache(ArraysCache):
    """Raw circular index keys plus persistent compressed-key state."""

    step = 256

    def __init__(
        self,
        compress_ratio: int,
        left_padding: Sequence[int] | None = None,
    ):
        if compress_ratio <= 0:
            raise ValueError("QSA compression ratio must be positive")
        super().__init__(size=2, left_padding=list(left_padding or []))
        self.compress_ratio = compress_ratio
        batch = len(left_padding) if left_padding else 1
        self._offsets = [0] * batch
        self._compressed_counts = [0] * batch
        self._valid_until: list[int] | None = None
        self._right_padding: list[int] | None = None
        self._pending_left_padding = [int(item) for item in (left_padding or [0])]

    def _ensure_batch(self, batch: int):
        """Adopt mlx-lm's fresh-batch size before state is committed."""
        if len(self._offsets) == batch:
            return
        if not self.empty() or self.raw_ring is not None:
            raise ValueError("QSA cache batch size changed outside cache lifecycle")
        if self.left_padding is None or self.left_padding.size != batch:
            raise ValueError("QSA cache batch metadata does not match input")
        self._offsets = [0] * batch
        self._compressed_counts = [0] * batch
        self._pending_left_padding = [int(item) for item in self.left_padding.tolist()]

    @property
    def raw_ring(self):
        return self.cache[0]

    @raw_ring.setter
    def raw_ring(self, value):
        self.cache[0] = value

    @property
    def compressed_keys(self):
        return self.cache[1]

    @compressed_keys.setter
    def compressed_keys(self, value):
        self.cache[1] = value

    @property
    def offset(self):
        if len(self._offsets) == 1 and self.left_padding is None:
            return self._offsets[0]
        return mx.array(self._offsets, dtype=mx.int32)

    @property
    def _compressed_count(self):
        """Compatibility scalar used by single-request diagnostics/tests."""
        if len(self._compressed_counts) != 1:
            raise AttributeError("batched QSA cache has per-row compressed counts")
        return self._compressed_counts[0]

    def prepare(self, lengths=None, right_padding=None, **_kwargs):
        batch = (
            int(self.left_padding.size)
            if self.left_padding is not None
            else len(self._offsets)
        )
        self._ensure_batch(batch)
        if lengths is not None:
            self._valid_until = [
                offset + int(length) for offset, length in zip(self._offsets, lengths)
            ]
        self._right_padding = (
            None if right_padding is None else [int(item) for item in right_padding]
        )

    def finalize(self):
        if self._right_padding is not None and self.left_padding is not None:
            self.left_padding += mx.array(self._right_padding, dtype=mx.int32)
        self._valid_until = None
        self._right_padding = None
        self.lengths = None
        # QSA state contains only logical tokens, so unlike the main KV cache
        # it needs no physical roll after a right-padded prefill.

    def valid_lengths(self, input_length: int) -> list[int]:
        return [count for _, count in self.valid_spans(input_length)]

    def valid_spans(self, input_length: int) -> list[tuple[int, int]]:
        starts = [min(input_length, pending) for pending in self._pending_left_padding]
        if self._valid_until is None:
            return [(start, input_length - start) for start in starts]
        return [
            (
                start,
                max(0, min(input_length - start, limit - offset)),
            )
            for start, limit, offset in zip(starts, self._valid_until, self._offsets)
        ]

    def update(
        self,
        raw_keys: mx.array,
        transform_group: Callable[[mx.array, int], mx.array],
        transform_groups: Callable[[mx.array, mx.array], mx.array] | None = None,
    ) -> mx.array:
        """Commit valid raw rows and cache every newly completed group.

        ``raw_keys`` is ``[batch, tokens, index_dim]``. ``transform_group``
        receives one row's raw mean ``[1, index_dim]`` and the group's logical
        first position. Per-row invocation is intentional: continuous batches
        may contain requests at different logical positions.
        """

        batch, length, dim = raw_keys.shape
        self._ensure_batch(batch)
        valid_spans = self.valid_spans(length)
        if (
            transform_groups is not None
            and batch == 1
            and valid_spans == [(0, length)]
            and self._offsets[0] % self.compress_ratio == 0
            and length // self.compress_ratio >= 2
        ):
            return self._update_aligned_single_row(raw_keys, transform_groups)
        if self.raw_ring is None:
            self.raw_ring = mx.zeros(
                (batch, self.compress_ratio, dim), dtype=raw_keys.dtype
            )
        elif self.raw_ring.shape[0] != batch or self.raw_ring.shape[-1] != dim:
            raise ValueError("QSA raw-key cache shape changed within one request")

        completed: list[list[mx.array]] = [[] for _ in range(batch)]
        for row, (input_start, valid_length) in enumerate(valid_spans):
            for token in range(valid_length):
                position = self._offsets[row] + token
                self.raw_ring[row, position % self.compress_ratio, :] = raw_keys[
                    row, input_start + token, :
                ]
                if (position + 1) % self.compress_ratio == 0:
                    pooled = mx.mean(
                        self.raw_ring[row : row + 1].astype(mx.float32), axis=1
                    ).astype(raw_keys.dtype)
                    completed[row].append(
                        transform_group(pooled, position + 1 - self.compress_ratio)[0]
                    )
            self._offsets[row] += valid_length
            self._pending_left_padding[row] -= input_start

        new_counts = [
            old + len(rows) for old, rows in zip(self._compressed_counts, completed)
        ]
        max_count = max(new_counts, default=0)
        if max_count:
            capacity = ((max_count + self.step - 1) // self.step) * self.step
            if self.compressed_keys is None:
                expanded = mx.zeros((batch, capacity, dim), dtype=raw_keys.dtype)
            elif capacity > self.compressed_keys.shape[1]:
                expanded = mx.zeros(
                    (batch, capacity, dim), dtype=self.compressed_keys.dtype
                )
                expanded[:, : self.compressed_keys.shape[1], :] = self.compressed_keys
            else:
                expanded = self.compressed_keys
            for row, rows in enumerate(completed):
                if rows:
                    start = self._compressed_counts[row]
                    expanded[row, start : start + len(rows), :] = mx.stack(rows)
            self.compressed_keys = expanded
        self._compressed_counts = new_counts

        if self.compressed_keys is None:
            return mx.zeros((batch, 0, dim), dtype=raw_keys.dtype)
        return self.compressed_keys[:, :max_count, :]

    def _update_aligned_single_row(
        self,
        raw_keys: mx.array,
        transform_groups: Callable[[mx.array, mx.array], mx.array],
    ) -> mx.array:
        """Batch complete ratio-sized groups for the common B=1 prefill path."""
        _, length, dim = raw_keys.shape
        ratio = self.compress_ratio
        complete_length = length // ratio * ratio
        group_count = complete_length // ratio
        # ``update`` enters this fast path only with at least two complete
        # groups, so keep the implementation free of unreachable empty-group
        # branches and make every cache transition explicit.
        groups = raw_keys[:, :complete_length, :].reshape(1, group_count, ratio, dim)
        pooled = mx.mean(groups.astype(mx.float32), axis=2).astype(raw_keys.dtype)
        starts = self._offsets[0] + mx.arange(group_count) * ratio
        transformed = transform_groups(pooled, starts)

        ring = raw_keys[:, complete_length - ratio : complete_length, :]
        remainder = length - complete_length
        if remainder:
            ring = mx.concatenate(
                [
                    raw_keys[:, complete_length:, :],
                    ring[:, remainder:, :],
                ],
                axis=1,
            )
        self.raw_ring = ring

        old_count = self._compressed_counts[0]
        new_count = old_count + group_count
        capacity = ((new_count + self.step - 1) // self.step) * self.step
        if self.compressed_keys is None:
            expanded = mx.zeros((1, capacity, dim), dtype=raw_keys.dtype)
        elif capacity > self.compressed_keys.shape[1]:
            expanded = mx.zeros((1, capacity, dim), dtype=self.compressed_keys.dtype)
            expanded[:, : self.compressed_keys.shape[1], :] = self.compressed_keys
        else:
            expanded = self.compressed_keys
        expanded[:, old_count:new_count, :] = transformed
        self.compressed_keys = expanded

        self._offsets[0] += length
        self._pending_left_padding[0] = 0
        self._compressed_counts[0] = new_count
        return self.compressed_keys[:, :new_count, :]

    def keys_for_blocks(self, row: int, block_count: int) -> mx.array:
        if block_count < 0 or block_count > self._compressed_counts[row]:
            raise ValueError("QSA compressed block request is out of range")
        if self.compressed_keys is None:
            if block_count:
                raise ValueError("QSA compressed cache is empty")
            return mx.zeros((0, 0))
        return self.compressed_keys[row, :block_count, :]

    @property
    def state(self):
        compressed = self.compressed_keys
        if compressed is not None:
            compressed = compressed[:, : max(self._compressed_counts, default=0)]
        return self.raw_ring, compressed

    @state.setter
    def state(self, value):
        self.raw_ring, self.compressed_keys = value

    @property
    def meta_state(self):
        return json.dumps(
            {
                "compress_ratio": self.compress_ratio,
                "offsets": self._offsets,
                "compressed_counts": self._compressed_counts,
            },
            separators=(",", ":"),
        )

    @meta_state.setter
    def meta_state(self, value):
        decoded = json.loads(value)
        self.compress_ratio = int(decoded["compress_ratio"])
        self._offsets = [int(item) for item in decoded["offsets"]]
        self._compressed_counts = [int(item) for item in decoded["compressed_counts"]]
        self._valid_until = None
        self._right_padding = None
        self._pending_left_padding = [0] * len(self._offsets)
        self.left_padding = (
            None
            if len(self._offsets) == 1
            else mx.zeros(len(self._offsets), dtype=mx.int32)
        )
        self.lengths = None

    @classmethod
    def from_state(cls, state, meta_state):
        cache = cls.__new__(cls)
        cache.cache = [None, None]
        cache.meta_state = meta_state
        cache.state = state
        return cache

    def is_trimmable(self):
        return self.can_trim(1)

    def can_trim(self, n: int) -> bool:
        """Amount-aware preflight consumed by composite cache rollback."""
        return all(self._can_trim_row(offset, n) for offset in self._offsets)

    def trim_checkpoint(self):
        return list(self._offsets), list(self._compressed_counts)

    def restore_trim_checkpoint(self, state):
        self._offsets, self._compressed_counts = state

    def _can_trim_row(self, offset: int, n: int) -> bool:
        if n < 0 or n > offset:
            return False
        if n == 0:
            return True
        remainder = offset % self.compress_ratio
        available = remainder if remainder else min(self.compress_ratio, offset)
        return n <= available

    def trim(self, n):
        # The raw ring retains exactly the current partial group, or the most
        # recently completed group at a boundary. Rewind only within that
        # recoverable window; the next update overwrites discarded rows and
        # deterministically recomputes a removed compressed block.
        if not all(self._can_trim_row(offset, n) for offset in self._offsets):
            return 0
        self._offsets = [offset - n for offset in self._offsets]
        self._compressed_counts = [
            offset // self.compress_ratio for offset in self._offsets
        ]
        return n

    def size(self):
        return max(self._offsets, default=0)

    def empty(self):
        return all(offset == 0 for offset in self._offsets)

    @property
    def nbytes(self):
        return sum(item.nbytes for item in self.cache if item is not None)

    def filter(self, batch_indices):
        indices = (
            batch_indices.tolist()
            if isinstance(batch_indices, mx.array)
            else list(batch_indices)
        )
        super().filter(batch_indices)
        self._offsets = [self._offsets[index] for index in indices]
        self._compressed_counts = [self._compressed_counts[index] for index in indices]
        self._pending_left_padding = [
            self._pending_left_padding[index] for index in indices
        ]
        if self._valid_until is not None:
            self._valid_until = [self._valid_until[index] for index in indices]
        # ArraysCache.filter already selected the matching physical-padding
        # rows. Preserve those values: they describe where each request's
        # logical token zero sits in the still-padded KV tensors. Recomputing
        # against the longest *surviving* request would incorrectly turn a
        # retained shorter row's padding into zero.

    def extend(self, other):
        rows = [self.extract(index) for index in range(len(self._offsets))]
        rows.extend(other.extract(index) for index in range(len(other._offsets)))
        merged = self.merge(rows)
        self.cache = list(merged.cache)
        self.left_padding = merged.left_padding
        self.lengths = None
        self._offsets = merged._offsets
        self._compressed_counts = merged._compressed_counts
        self._pending_left_padding = merged._pending_left_padding
        self._valid_until = self._right_padding = None

    def extract(self, idx):
        cache = QSAIndexCache(self.compress_ratio)
        count = self._compressed_counts[idx]
        if self.raw_ring is not None:
            cache.raw_ring = mx.contiguous(self.raw_ring[idx : idx + 1])
        if self.compressed_keys is not None and count:
            cache.compressed_keys = mx.contiguous(
                self.compressed_keys[idx : idx + 1, :count]
            )
        cache._offsets = [self._offsets[idx]]
        cache._compressed_counts = [count]
        cache._pending_left_padding = [0]
        return cache

    @classmethod
    def merge(cls, caches):
        if not caches:
            raise ValueError("cannot merge an empty QSA cache list")
        ratio = caches[0].compress_ratio
        if any(cache.compress_ratio != ratio for cache in caches):
            raise ValueError("QSA caches in a batch must share compression ratio")
        merged = cls(ratio, left_padding=[0] * len(caches))
        merged._offsets = [cache._offsets[0] for cache in caches]
        merged._compressed_counts = [cache._compressed_counts[0] for cache in caches]
        merged._pending_left_padding = [0] * len(caches)
        max_offset = max(merged._offsets, default=0)
        merged.left_padding = mx.array(
            [max_offset - offset for offset in merged._offsets], dtype=mx.int32
        )

        raw_template = next(
            (cache.raw_ring for cache in caches if cache.raw_ring is not None), None
        )
        if raw_template is not None:
            merged.raw_ring = mx.zeros(
                (len(caches), *raw_template.shape[1:]), dtype=raw_template.dtype
            )
            for row, cache in enumerate(caches):
                if cache.raw_ring is not None:
                    merged.raw_ring[row] = cache.raw_ring[0]

        compressed_template = next(
            (
                cache.compressed_keys
                for cache in caches
                if cache.compressed_keys is not None
            ),
            None,
        )
        max_count = max(merged._compressed_counts, default=0)
        if compressed_template is not None and max_count:
            merged.compressed_keys = mx.zeros(
                (len(caches), max_count, compressed_template.shape[-1]),
                dtype=compressed_template.dtype,
            )
            for row, cache in enumerate(caches):
                count = cache._compressed_counts[0]
                if count:
                    merged.compressed_keys[row, :count] = cache.compressed_keys[
                        0, :count
                    ]
        return merged
