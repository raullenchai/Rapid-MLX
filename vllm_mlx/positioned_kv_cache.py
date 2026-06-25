# SPDX-License-Identifier: Apache-2.0
"""Arbitrary-``position_ids`` KV-cache write path (R15 task #301).

The upstream ``mlx_lm.models.cache.KVCache`` /
``mlx_lm.models.cache.QuantizedKVCache`` implementations grow strictly
monotonically: every call to :py:meth:`KVCache.update_and_fetch` writes
``keys`` / ``values`` into slots ``[offset, offset+S)`` and bumps
``offset += S``. That contract is fine for vanilla prefill + decode, but
it is incompatible with every tree-based speculative decode pipeline
(EAGLE-3, DFlash, DDTree, MTP) and with DSA-style sparse gather.

A draft tree emits **non-monotonic** position ids — e.g. positions
``[10, 11, 11, 12, 12, 13, 13]`` while the verifier checks three
alternative branches at depths 11/12/13 — and the verifier picks one
branch at the end. The cache must support writing those branch
candidates into explicit slots so the verifier can later overwrite or
discard them; without that, no tree spec decode is even possible.

This module is the **purely additive** capability that unlocks Phase 5
of the 0.9 TODO (#302 MTP, DFlash-MLX, DDTree-MLX) and #296 disk-backed
KV checkpointing. It does NOT change the upstream class signatures and
does NOT touch the on-disk cache format used by ``save_prompt_cache`` /
the radix prefix cache (#303).

Design choices
--------------

* The capability is offered two ways: a standalone function
  :func:`positioned_update_and_fetch` that operates on any vanilla
  ``KVCache`` / ``QuantizedKVCache`` instance, AND thin subclasses
  :class:`PositionedKVCache` / :class:`PositionedQuantizedKVCache` that
  expose ``update_and_fetch(keys, values, position_ids=None)`` as the
  natural method-level extension. Callers that already hold a vanilla
  cache (the common case for spec decode that swaps the cache in
  per-request) can use the function; new callers that want the method
  signature can construct the subclass directly.
* Duplicate position ids resolve to **last-writer-wins by source-index
  order**. We pre-dedup ``position_ids`` Python-side and pass MLX the
  ``(position, last_source_index)`` pair set — without the dedup, MLX
  scatter into duplicate indices is racy (verified empirically on
  ``mlx.core 0.31.3`` — same-position writes land non-deterministically
  across heads). The dedup keeps the LAST occurrence of each position,
  so a tree verifier that places the accepted branch last in the
  ``keys`` tensor gets exactly that branch in the cache.
* ``self.offset`` advances to ``max(prev_offset, max(position_ids)+1)``
  so subsequent monotonic decode steps continue to read the correct
  causal window. Backward writes (positions below ``prev_offset``) do
  **not** rewind the offset.
* When ``position_ids is None`` the implementation delegates straight to
  the upstream ``update_and_fetch``; backward compatibility is
  bit-identical (locked by ``tests/test_kv_cache_position_ids.py``).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

__all__ = [
    "PositionedKVCache",
    "PositionedQuantizedKVCache",
    "positioned_update_and_fetch",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_position_ids(
    position_ids: Any,
) -> tuple[list[int], list[int], int]:
    """Normalize ``position_ids`` and pre-dedup to last-writer-wins order.

    Returns ``(unique_positions, source_indices, max_position)`` where:

    * ``unique_positions`` is the deduplicated, order-preserving list of
      cache slots to write into.
    * ``source_indices`` is the parallel list of source-row indices into
      ``keys`` / ``values``. For each unique position we keep the LAST
      occurrence in the original ``position_ids`` — this is the
      last-writer-wins semantics the tree verifier depends on.
    * ``max_position`` is ``max(unique_positions)``, used to size the
      cache buffer.

    Why we don't pass the raw ``mx.array`` directly to MLX scatter:
    ``mlx.core 0.31.3`` scatter assignment into duplicate destination
    indices is non-deterministic (the racing writes land in any order
    across heads). Pre-deduping Python-side makes the write trivially
    deterministic at the cost of one host-side dict pass — negligible
    on the spec-decode tree size (~64 candidates).

    Accepts an ``mx.array``, a Python list/tuple, or any iterable of
    ints.
    """
    if position_ids is None:  # pragma: no cover — caller guarded
        raise ValueError("position_ids must not be None")

    if isinstance(position_ids, mx.array):
        if position_ids.ndim != 1:
            raise ValueError(
                f"position_ids must be 1-D (got shape {position_ids.shape})"
            )
        # ``.tolist()`` forces a sync, but the dedup needs host-side
        # ints anyway so the eval is unavoidable.
        raw = [int(p) for p in position_ids.tolist()]
    else:
        raw = [int(p) for p in position_ids]

    if not raw:
        raise ValueError("position_ids must not be empty")
    for p in raw:
        if p < 0:
            raise ValueError(
                f"position_ids must be non-negative (got {p})"
            )

    # Dedup: keep the LAST source index for each unique position. ``dict``
    # preserves insertion order on Python 3.7+, so iterating ``last.items()``
    # gives a stable order.
    last: dict[int, int] = {}
    for i, p in enumerate(raw):
        last[p] = i  # overwrite — final assignment wins

    unique_positions = list(last.keys())
    source_indices = list(last.values())
    return unique_positions, source_indices, max(unique_positions)


def _validate_seq_dim(
    keys: mx.array, values: mx.array, raw_pos_len: int
) -> None:
    """Ensure ``position_ids`` length matches the seq dim of keys/values.

    ``raw_pos_len`` is the length of the original (pre-dedup)
    ``position_ids`` array. The validator is intentionally strict — a
    tree builder that emits the wrong number of positions has a real bug
    and must fail loud, not silently scatter into the wrong slots.
    """
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            "keys and values must be 4-D (B, n_kv_heads, S, head_dim); got "
            f"keys.ndim={keys.ndim}, values.ndim={values.ndim}"
        )
    if keys.shape[2] != raw_pos_len:
        raise ValueError(
            f"position_ids length {raw_pos_len} does not match keys seq dim "
            f"{keys.shape[2]}"
        )
    if values.shape[2] != raw_pos_len:
        raise ValueError(
            f"position_ids length {raw_pos_len} does not match values seq dim "
            f"{values.shape[2]}"
        )


def _grow_kv_cache(cache: KVCache, keys: mx.array, values: mx.array, required: int) -> None:
    """Grow a plain ``KVCache`` so its buffer covers at least ``required`` slots.

    Mirrors the upstream allocation logic in
    :py:meth:`KVCache.update_and_fetch` but sized to ``required`` rather
    than ``offset + S``. The cache buffer is rounded up to a multiple of
    :py:attr:`KVCache.step` to match the upstream allocation pattern;
    that keeps allocation noise off the hot path when many positioned
    writes land within the same allocation chunk.
    """
    step = cache.step
    if cache.keys is None:
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        n_steps = (step + required - 1) // step
        capacity = n_steps * step
        cache.keys = mx.zeros((B, n_kv_heads, capacity, k_head_dim), keys.dtype)
        cache.values = mx.zeros((B, n_kv_heads, capacity, v_head_dim), values.dtype)
        return

    current = cache.keys.shape[2]
    if required <= current:
        return

    # Trim any allocation slack past the current offset before growing —
    # mirrors upstream's ``prev % step != 0`` trim. Keeps capacity tight
    # and matches the layout the offset-based fast path expects.
    if cache.offset % step != 0 and cache.offset < current:
        cache.keys = cache.keys[..., : cache.offset, :]
        cache.values = cache.values[..., : cache.offset, :]

    deficit = required - cache.keys.shape[2]
    n_steps = (step + deficit - 1) // step
    pad = n_steps * step
    B, n_kv_heads, _, k_head_dim = cache.keys.shape
    v_head_dim = cache.values.shape[3]
    new_k = mx.zeros((B, n_kv_heads, pad, k_head_dim), cache.keys.dtype)
    new_v = mx.zeros((B, n_kv_heads, pad, v_head_dim), cache.values.dtype)
    cache.keys = mx.concatenate([cache.keys, new_k], axis=2)
    cache.values = mx.concatenate([cache.values, new_v], axis=2)


def _grow_quant_cache(
    cache: QuantizedKVCache, keys: mx.array, values: mx.array, required: int
) -> None:
    """Grow a ``QuantizedKVCache`` so its buffer covers at least ``required`` slots.

    The quantized cache stores ``(packed_uint32, scales, biases)`` triples
    rather than a single tensor. Growth has to expand each member of the
    triple in lock-step. Capacity rounds up to a multiple of ``step`` to
    match the upstream allocation pattern.
    """
    step = cache.step
    bits = cache.bits
    group_size = cache.group_size
    el_per_int = 8 * mx.uint32.size // bits

    if cache.keys is None:
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        n_steps = (step + required - 1) // step
        capacity = n_steps * step

        def init(dim: int):
            return (
                mx.zeros((B, n_kv_heads, capacity, dim // el_per_int), dtype=mx.uint32),
                mx.zeros((B, n_kv_heads, capacity, dim // group_size), dtype=keys.dtype),
                mx.zeros((B, n_kv_heads, capacity, dim // group_size), dtype=keys.dtype),
            )

        cache.keys = init(k_head_dim)
        cache.values = init(v_head_dim)
        return

    current = cache.keys[0].shape[2]
    if required <= current:
        return

    # Trim any allocation slack the same way upstream does in
    # ``QuantizedKVCache.update_and_fetch``.
    prev = cache.offset
    if prev % step != 0 and prev < current:
        def _trim(x):
            return x[..., :prev, :]
        cache.keys = tuple(_trim(x) for x in cache.keys)
        cache.values = tuple(_trim(x) for x in cache.values)

    deficit = required - cache.keys[0].shape[2]
    n_steps = (step + deficit - 1) // step
    pad = n_steps * step
    B, n_kv_heads, _, _ = cache.keys[0].shape

    def expand(triple):
        return tuple(
            mx.concatenate(
                [x, mx.zeros((B, n_kv_heads, pad, x.shape[-1]), dtype=x.dtype)],
                axis=2,
            )
            for x in triple
        )

    cache.keys = expand(cache.keys)
    cache.values = expand(cache.values)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def positioned_update_and_fetch(
    cache: KVCache | QuantizedKVCache,
    keys: mx.array,
    values: mx.array,
    position_ids: mx.array | list[int] | tuple[int, ...] | None = None,
):
    """Scatter ``keys`` / ``values`` into ``cache`` at explicit positions.

    Args:
        cache: A vanilla :class:`KVCache` or :class:`QuantizedKVCache`.
            Subclasses inherit the same behaviour; rotating/chunked/batch
            caches are out of scope and raise :class:`TypeError`.
        keys: Shape ``(B, n_kv_heads, S, k_head_dim)`` — the new key rows.
        values: Shape ``(B, n_kv_heads, S, v_head_dim)`` — matching V.
        position_ids: ``None`` to delegate to the upstream monotonic
            update_and_fetch path (bit-identical backward compat); or a
            length-``S`` 1-D ``mx.array`` / list of non-negative ints
            giving the explicit cache slot for each row. Duplicate
            positions resolve to "last-writer-wins by source-index order"
            (matches ``mlx.core`` scatter semantics).

    Returns:
        The same ``(keys_view, values_view)`` tuple the upstream method
        returns, sliced to ``[:cache.offset]``. For the quantized cache
        the views are 3-tuples ``(packed, scales, biases)`` matching the
        upstream contract.

    Raises:
        TypeError: If ``cache`` is neither :class:`KVCache` nor
            :class:`QuantizedKVCache` (or a subclass thereof).
        ValueError: If ``position_ids`` has the wrong shape, contains
            negative values, is empty, or its length does not match the
            seq dim of ``keys`` / ``values``.
    """
    if position_ids is None:
        # Delegate to the upstream method on the matching base class —
        # NOT to ``cache.update_and_fetch`` directly, which would recurse
        # back through the subclass override.
        if isinstance(cache, QuantizedKVCache):
            return QuantizedKVCache.update_and_fetch(cache, keys, values)
        if isinstance(cache, KVCache):
            return KVCache.update_and_fetch(cache, keys, values)
        raise TypeError(
            "positioned_update_and_fetch only supports KVCache and "
            f"QuantizedKVCache; got {type(cache).__name__}."
        )

    # Compute raw length BEFORE dedup so the validator catches "wrong
    # number of positions for this many rows" cleanly.
    if isinstance(position_ids, mx.array):
        raw_len = int(position_ids.shape[0]) if position_ids.ndim == 1 else -1
    else:
        try:
            raw_len = len(position_ids)
        except TypeError:
            raw_len = -1
    if raw_len < 0:
        raw_len = keys.shape[2]  # fall back; validator will still catch shape

    unique_positions, source_indices, max_pos = _coerce_position_ids(position_ids)
    _validate_seq_dim(keys, values, raw_len)

    # ``mlx.core 0.31.3`` scatter assignment into duplicate destination
    # indices is non-deterministic. ``unique_positions`` is already
    # deduplicated; gather the matching source rows so the destination
    # write has no duplicates.
    pos_arr = mx.array(unique_positions, dtype=mx.int32)
    src_arr = mx.array(source_indices, dtype=mx.int32)
    required = max(cache.offset, max_pos + 1)

    if isinstance(cache, QuantizedKVCache):
        _grow_quant_cache(cache, keys, values, required)
        # Quantize the new rows in the same dtype/group/bits the cache
        # was built with, then scatter each member of the triple. We
        # quantize the full ``keys`` tensor (cheap on tree-size inputs)
        # and gather the deduplicated source rows from the quantized
        # output, rather than quantizing twice.
        q_keys = mx.quantize(keys, group_size=cache.group_size, bits=cache.bits)
        q_values = mx.quantize(values, group_size=cache.group_size, bits=cache.bits)
        new_keys = list(cache.keys)
        new_values = list(cache.values)
        for i in range(3):
            # ``mx.take(x, src_arr, axis=2)`` is the cheap gather; it
            # produces a tensor whose seq dim matches ``pos_arr``.
            new_keys[i][..., pos_arr, :] = mx.take(q_keys[i], src_arr, axis=2)
            new_values[i][..., pos_arr, :] = mx.take(q_values[i], src_arr, axis=2)
        cache.keys = tuple(new_keys)
        cache.values = tuple(new_values)
        cache.offset = required
        view = tuple(x[..., : cache.offset, :] for x in cache.keys)
        v_view = tuple(x[..., : cache.offset, :] for x in cache.values)
        return view, v_view

    if isinstance(cache, KVCache):
        _grow_kv_cache(cache, keys, values, required)
        cache.keys[..., pos_arr, :] = mx.take(keys, src_arr, axis=2)
        cache.values[..., pos_arr, :] = mx.take(values, src_arr, axis=2)
        cache.offset = required
        return (
            cache.keys[..., : cache.offset, :],
            cache.values[..., : cache.offset, :],
        )

    raise TypeError(
        "positioned_update_and_fetch only supports KVCache and "
        f"QuantizedKVCache; got {type(cache).__name__}. Rotating/chunked/"
        "batch caches are out of scope for R15 #301 — file a follow-up if "
        "a tree spec decode pipeline needs them."
    )


class PositionedKVCache(KVCache):
    """``KVCache`` with an optional ``position_ids`` argument on update.

    Backward-compat note: when callers omit ``position_ids`` the result
    is bit-identical to ``KVCache.update_and_fetch`` (the implementation
    delegates straight to the parent). When provided, the new rows are
    scattered via :func:`positioned_update_and_fetch`.

    Persistence note: this subclass is NOT a drop-in replacement for the
    cache type registered with ``mlx_lm.models.cache.save_prompt_cache`` —
    the on-disk format records ``type(c).__name__`` and the loader looks
    that name up in the upstream module globals. Use the standalone
    :func:`positioned_update_and_fetch` for callers that interleave with
    the radix prefix cache / disk-backed KV checkpointing (#296, #303).
    """

    def update_and_fetch(  # type: ignore[override]
        self,
        keys: mx.array,
        values: mx.array,
        position_ids: mx.array | list[int] | tuple[int, ...] | None = None,
    ):
        return positioned_update_and_fetch(self, keys, values, position_ids)


class PositionedQuantizedKVCache(QuantizedKVCache):
    """``QuantizedKVCache`` with an optional ``position_ids`` on update.

    Same contract as :class:`PositionedKVCache` but for the int4/int8
    quantized path. ``--kv-cache-dtype int4`` is the default after R15
    #300 / #910, so this subclass is the one most callers will end up
    using when threading tree spec decode through production.

    Persistence note: same caveat as :class:`PositionedKVCache` — the
    on-disk format records the class name and the upstream loader does
    not know about this subclass. Use :func:`positioned_update_and_fetch`
    for callers that interleave with the radix prefix cache.
    """

    def update_and_fetch(  # type: ignore[override]
        self,
        keys: mx.array,
        values: mx.array,
        position_ids: mx.array | list[int] | tuple[int, ...] | None = None,
    ):
        return positioned_update_and_fetch(self, keys, values, position_ids)
