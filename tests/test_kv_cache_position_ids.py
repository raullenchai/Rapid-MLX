# SPDX-License-Identifier: Apache-2.0
"""Unit tests for arbitrary ``position_ids`` KV-cache writes (R15 task #301).

These tests pin the contract for :mod:`vllm_mlx.positioned_kv_cache`,
which is the foundational unlock for every tree-based speculative decode
pipeline (EAGLE-3, DFlash, DDTree, MTP — Phase 5 of the 0.9-TODO) and
DSA per-token gather. The tree decoders depend on:

* **Bit-identical backward compat** when ``position_ids=None`` — the
  existing monotonic-offset path must not move.
* **Last-writer-wins by source-index order** for duplicate positions —
  the tree verifier orders branch candidates so the accepted branch is
  the last row in ``keys``; the cache write must keep that one.
* **Sparse position support** ([3, 5, 7]) — DSA gather emits sparse
  positions and expects the in-between slots to retain whatever was
  previously there (zero on a fresh cache).
* **Quantized parity** — ``--kv-cache-dtype int4`` is the default after
  R15 #300 / #910; the positioned write must behave identically on the
  quantized path.
* **Validation** — shape mismatch / negative ids / empty ids reject
  loudly so a buggy speculative-decode tree builder can't silently
  corrupt the cache.

Run with::

    pytest tests/test_kv_cache_position_ids.py
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402

from vllm_mlx.positioned_kv_cache import (  # noqa: E402
    PositionedKVCache,
    PositionedQuantizedKVCache,
    positioned_update_and_fetch,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_keys_values(seq: int, *, head_dim: int = 8, n_kv: int = 2, seed: int = 0):
    """Return deterministic ``(keys, values)`` of the requested seq length."""
    keys = mx.random.normal((1, n_kv, seq, head_dim), key=mx.random.key(seed))
    values = mx.random.normal((1, n_kv, seq, head_dim), key=mx.random.key(seed + 1))
    return keys, values


def _make_marked_keys_values(values_seq, *, head_dim: int = 8, n_kv: int = 2):
    """Return ``(keys, values)`` with each row filled with the matching marker.

    Used by the dedup / last-writer-wins tests so a single ``[0]`` element
    inspection identifies which source row landed in a given slot.
    """
    k_rows = [mx.full((1, n_kv, head_dim), float(v)) for v in values_seq]
    v_rows = [mx.full((1, n_kv, head_dim), float(v) + 1000.0) for v in values_seq]
    return mx.stack(k_rows, axis=2), mx.stack(v_rows, axis=2)


# ---------------------------------------------------------------------------
# Backward compatibility — no position_ids → bit-identical to upstream
# ---------------------------------------------------------------------------


def test_unquantized_no_position_ids_is_bit_identical_to_upstream():
    """``position_ids=None`` must yield the EXACT same (keys, values, offset)
    as the upstream ``KVCache.update_and_fetch`` — bit-identical, not just
    close. This is the backward-compat anchor for every existing caller.
    """
    upstream = KVCache()
    positioned = PositionedKVCache()
    k, v = _make_keys_values(5)
    out_u_k, out_u_v = upstream.update_and_fetch(k, v)
    out_p_k, out_p_v = positioned.update_and_fetch(k, v)
    assert upstream.offset == positioned.offset == 5
    assert mx.array_equal(out_u_k, out_p_k).item()
    assert mx.array_equal(out_u_v, out_p_v).item()


def test_unquantized_monotonic_position_ids_match_default_append():
    """Passing ``[5, 6, 7]`` to a cache already at offset 5 is the same
    contract as the default monotonic append; the cache state must be
    bit-identical to a vanilla ``KVCache`` that consumed the same rows.
    """
    upstream = KVCache()
    positioned = PositionedKVCache()
    k0, v0 = _make_keys_values(5, seed=2)
    upstream.update_and_fetch(k0, v0)
    positioned.update_and_fetch(k0, v0)
    k1, v1 = _make_keys_values(3, seed=4)
    out_u_k, out_u_v = upstream.update_and_fetch(k1, v1)
    out_p_k, out_p_v = positioned.update_and_fetch(k1, v1, position_ids=[5, 6, 7])
    assert upstream.offset == positioned.offset == 8
    assert mx.array_equal(out_u_k, out_p_k).item()
    assert mx.array_equal(out_u_v, out_p_v).item()


def test_standalone_helper_matches_upstream_when_position_ids_omitted():
    """The standalone ``positioned_update_and_fetch`` helper must be the
    same as the method form when ``position_ids=None`` — it's the API the
    radix prefix cache / disk-backed KV path will call into without
    needing to swap the cache subclass.
    """
    upstream = KVCache()
    helper_cache = KVCache()
    k, v = _make_keys_values(4, seed=7)
    out_u_k, out_u_v = upstream.update_and_fetch(k, v)
    out_h_k, out_h_v = positioned_update_and_fetch(helper_cache, k, v)
    assert helper_cache.offset == upstream.offset
    assert mx.array_equal(out_u_k, out_h_k).item()
    assert mx.array_equal(out_u_v, out_h_v).item()


# ---------------------------------------------------------------------------
# Tree spec decode case — non-monotonic positions with duplicates
# ---------------------------------------------------------------------------


def test_unquantized_tree_duplicate_positions_last_writer_wins():
    """Tree case: cache at offset 5, write positions ``[6, 6, 7]`` (two
    branch candidates at depth 6, one at depth 7).

    Contract:
      * Position 6 holds the LAST source row at that position (last-writer
        wins by source-index order — matches MLX scatter semantics).
      * Position 7 holds the single source row at that position.
      * Offset advances to ``max(prev_offset, max(positions)+1) = 8``.
    """
    cache = PositionedKVCache()
    # Warm the cache to offset 5 with a known prefill.
    prefill_k, prefill_v = _make_keys_values(5, seed=11)
    cache.update_and_fetch(prefill_k, prefill_v)
    assert cache.offset == 5

    # 3 candidate rows; markers 1.0, 2.0, 3.0 so we can identify which
    # row landed where after the write.
    k_tree, v_tree = _make_marked_keys_values([1.0, 2.0, 3.0])
    cache.update_and_fetch(k_tree, v_tree, position_ids=[6, 6, 7])

    assert cache.offset == 8
    # Position 6 must hold the LAST source row mapped to it (2.0).
    assert cache.keys[0, 0, 6, 0].item() == pytest.approx(2.0)
    assert cache.values[0, 0, 6, 0].item() == pytest.approx(1002.0)
    # Position 7 holds the single source row at that position.
    assert cache.keys[0, 0, 7, 0].item() == pytest.approx(3.0)
    assert cache.values[0, 0, 7, 0].item() == pytest.approx(1003.0)


def test_unquantized_tree_non_adjacent_duplicate_positions():
    """Tree case where the duplicate position is not adjacent in the
    ``position_ids`` array: ``[6, 7, 6]`` with markers ``[10, 20, 30]``.

    Position 6 must hold ``30`` (the LAST source row mapped to it);
    position 7 must hold ``20``. This guards the operator's expectation
    that the verifier can place the accepted branch at any index in the
    tree as long as it's the LAST occurrence of that depth.
    """
    cache = PositionedKVCache()
    k_tree, v_tree = _make_marked_keys_values([10.0, 20.0, 30.0])
    cache.update_and_fetch(k_tree, v_tree, position_ids=[6, 7, 6])
    assert cache.offset == 8
    assert cache.keys[0, 0, 6, 0].item() == pytest.approx(30.0)
    assert cache.keys[0, 0, 7, 0].item() == pytest.approx(20.0)


def test_unquantized_dsa_sparse_positions_preserves_gaps():
    """DSA-like sparse positions ``[3, 5, 7]`` — the slots in between
    (4, 6) must retain whatever was there before (zero on a fresh
    cache). This is the contract DSA per-token gather depends on.
    """
    cache = PositionedKVCache()
    k_sparse, v_sparse = _make_marked_keys_values([100.0, 200.0, 300.0])
    cache.update_and_fetch(k_sparse, v_sparse, position_ids=[3, 5, 7])

    assert cache.offset == 8
    # Written slots
    assert cache.keys[0, 0, 3, 0].item() == pytest.approx(100.0)
    assert cache.keys[0, 0, 5, 0].item() == pytest.approx(200.0)
    assert cache.keys[0, 0, 7, 0].item() == pytest.approx(300.0)
    # Gap slots must be zero (fresh cache, never written)
    assert cache.keys[0, 0, 4, 0].item() == pytest.approx(0.0)
    assert cache.keys[0, 0, 6, 0].item() == pytest.approx(0.0)
    # Slots below the lowest position are also fresh-zero
    assert cache.keys[0, 0, 0, 0].item() == pytest.approx(0.0)


def test_unquantized_backward_write_does_not_rewind_offset():
    """Writing into a position BELOW the current offset must overwrite
    the slot but NOT rewind the offset — the radix prefix cache (#303)
    and disk-backed KV checkpointing (#296) rely on monotonic offset
    growth even when tree branches retroactively rewrite earlier slots.
    """
    cache = PositionedKVCache()
    k0, v0 = _make_keys_values(8, seed=23)
    cache.update_and_fetch(k0, v0)
    assert cache.offset == 8

    k_back, v_back = _make_marked_keys_values([42.0])
    cache.update_and_fetch(k_back, v_back, position_ids=[3])
    assert cache.offset == 8  # NOT rewound
    assert cache.keys[0, 0, 3, 0].item() == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Quantized (int4 / int8) path — the production default after R15 #300
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_no_position_ids_is_bit_identical_to_upstream(bits):
    """Backward compat anchor for the quantized path. Locks the contract
    that ``--kv-cache-dtype int4`` (the new default) keeps the same wire
    output as the upstream ``QuantizedKVCache`` when ``position_ids`` is
    not provided.
    """
    upstream = QuantizedKVCache(group_size=64, bits=bits)
    positioned = PositionedQuantizedKVCache(group_size=64, bits=bits)
    k, v = _make_keys_values(5, head_dim=64)
    out_u_k, out_u_v = upstream.update_and_fetch(k, v)
    out_p_k, out_p_v = positioned.update_and_fetch(k, v)
    assert upstream.offset == positioned.offset == 5
    # Quantized state is a tuple (packed, scales, biases) — compare all
    # three members for both keys and values.
    for i in range(3):
        assert mx.array_equal(out_u_k[i], out_p_k[i]).item()
        assert mx.array_equal(out_u_v[i], out_p_v[i]).item()


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_monotonic_position_ids_match_default_append(bits):
    """The ``[5, 6, 7]`` monotonic path must produce the same quantized
    state as the upstream default append. The quant op is non-trivial
    (uint32 packing + scales/biases), so this catches any drift between
    the positioned write path and the upstream blob layout.
    """
    upstream = QuantizedKVCache(group_size=64, bits=bits)
    positioned = PositionedQuantizedKVCache(group_size=64, bits=bits)
    k0, v0 = _make_keys_values(5, head_dim=64, seed=31)
    upstream.update_and_fetch(k0, v0)
    positioned.update_and_fetch(k0, v0)
    k1, v1 = _make_keys_values(3, head_dim=64, seed=33)
    out_u_k, out_u_v = upstream.update_and_fetch(k1, v1)
    out_p_k, out_p_v = positioned.update_and_fetch(k1, v1, position_ids=[5, 6, 7])
    assert upstream.offset == positioned.offset == 8
    for i in range(3):
        assert mx.array_equal(out_u_k[i], out_p_k[i]).item()
        assert mx.array_equal(out_u_v[i], out_p_v[i]).item()


def test_quantized_tree_duplicate_positions_last_writer_wins():
    """Quantized tree case — same contract as the unquantized version
    but exercising the int4 wrapper. Compared via dequantized values
    because the packed uint32 representation isn't directly
    interpretable.
    """
    cache = PositionedQuantizedKVCache(group_size=64, bits=4)
    prefill_k, prefill_v = _make_keys_values(5, head_dim=64, seed=41)
    cache.update_and_fetch(prefill_k, prefill_v)
    assert cache.offset == 5

    k_tree, v_tree = _make_marked_keys_values([1.0, 2.0, 3.0], head_dim=64)
    cache.update_and_fetch(k_tree, v_tree, position_ids=[6, 6, 7])
    assert cache.offset == 8

    # Dequantize and inspect the slots. Each row was filled with a
    # constant value, so a single column ``[0]`` is enough to identify.
    packed_k, scales_k, biases_k = cache.keys
    deq_k = mx.dequantize(packed_k, scales_k, biases_k, group_size=64, bits=4)
    # int4 dequantization introduces small rounding, so use approx with
    # a generous tolerance — we only care that the correct source row
    # landed in the slot.
    assert deq_k[0, 0, 6, 0].item() == pytest.approx(2.0, abs=0.1)
    assert deq_k[0, 0, 7, 0].item() == pytest.approx(3.0, abs=0.1)


def test_quantized_dsa_sparse_positions_preserves_gaps():
    """Quantized DSA sparse case — exercises the int4 scatter path with
    non-contiguous positions. Gap slots must remain zero (fresh cache),
    written slots dequantize to the marker value within int4 tolerance.
    """
    cache = PositionedQuantizedKVCache(group_size=64, bits=4)
    k_sparse, v_sparse = _make_marked_keys_values([100.0, 200.0, 300.0], head_dim=64)
    cache.update_and_fetch(k_sparse, v_sparse, position_ids=[3, 5, 7])
    assert cache.offset == 8

    packed_k, scales_k, biases_k = cache.keys
    deq_k = mx.dequantize(packed_k, scales_k, biases_k, group_size=64, bits=4)
    # Written slots — int4 has limited dynamic range; the marker values
    # 100/200/300 are well outside [-1, 1] but still distinguishable
    # within rough tolerance after dequant.
    assert deq_k[0, 0, 3, 0].item() == pytest.approx(100.0, rel=0.05)
    assert deq_k[0, 0, 5, 0].item() == pytest.approx(200.0, rel=0.05)
    assert deq_k[0, 0, 7, 0].item() == pytest.approx(300.0, rel=0.05)
    # Gap slots — the fresh capacity is zero-initialized, and "zero
    # quantized" stays zero after dequant.
    assert deq_k[0, 0, 4, 0].item() == pytest.approx(0.0, abs=1e-3)
    assert deq_k[0, 0, 6, 0].item() == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Validation — bad inputs must reject loudly
# ---------------------------------------------------------------------------


def test_position_ids_length_mismatch_rejects():
    """A buggy tree builder that emits the wrong number of positions
    must trip the validator, not silently corrupt the cache.
    """
    cache = PositionedKVCache()
    k, v = _make_keys_values(3)
    with pytest.raises(ValueError, match="position_ids length"):
        cache.update_and_fetch(k, v, position_ids=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="position_ids length"):
        cache.update_and_fetch(k, v, position_ids=[0, 1])


def test_position_ids_negative_rejects():
    """Negative positions are nonsensical for a 0-indexed cache slot;
    reject loudly so the speculative decode tree builder can't smuggle
    a sign-bit bug into production.
    """
    cache = PositionedKVCache()
    k, v = _make_keys_values(2)
    with pytest.raises(ValueError, match="non-negative"):
        cache.update_and_fetch(k, v, position_ids=[0, -1])


def test_position_ids_empty_rejects():
    """An empty position_ids array is almost certainly a tree-builder
    bug — fail loud, not silent.
    """
    cache = PositionedKVCache()
    k, v = _make_keys_values(0)
    with pytest.raises(ValueError, match="empty"):
        cache.update_and_fetch(k, v, position_ids=[])


def test_unsupported_cache_type_rejects():
    """The R15 #301 scope is plain ``KVCache`` + ``QuantizedKVCache``.
    Rotating / chunked / batch caches deliberately raise so callers know
    they need to either upgrade those classes or stick with the
    monotonic path.
    """
    from mlx_lm.models.cache import RotatingKVCache

    rot = RotatingKVCache(max_size=64)
    k, v = _make_keys_values(3)
    with pytest.raises(TypeError, match="only supports KVCache"):
        positioned_update_and_fetch(rot, k, v, position_ids=[0, 1, 2])


# ---------------------------------------------------------------------------
# Cache growth — capacity must expand past the upstream step boundary
# ---------------------------------------------------------------------------


def test_unquantized_growth_past_step_boundary():
    """A sparse write that lands beyond the current capacity (the
    upstream step is 256) must grow the buffer to fit. Without this,
    the scatter assignment trips an IndexError on the first tree write
    that emits a high-index position.
    """
    cache = PositionedKVCache()
    k, v = _make_marked_keys_values([7.0])
    cache.update_and_fetch(k, v, position_ids=[500])
    assert cache.offset == 501
    # Capacity rounds up to a multiple of step (256) — at minimum 512.
    assert cache.keys.shape[2] >= 501
    assert cache.keys[0, 0, 500, 0].item() == pytest.approx(7.0)


def test_quantized_growth_past_step_boundary():
    """Same growth contract on the quantized path. The triple
    ``(packed, scales, biases)`` must expand in lock-step; if any
    member lags, the next scatter is misaligned.
    """
    cache = PositionedQuantizedKVCache(group_size=64, bits=4)
    k, v = _make_marked_keys_values([7.0], head_dim=64)
    cache.update_and_fetch(k, v, position_ids=[500])
    assert cache.offset == 501
    assert cache.keys[0].shape[2] >= 501
    assert cache.keys[1].shape[2] >= 501
    assert cache.keys[2].shape[2] >= 501


# ---------------------------------------------------------------------------
# Helper-form parity — the standalone function and the subclass method
# must produce the same state for the same inputs.
# ---------------------------------------------------------------------------


def test_helper_function_and_subclass_method_produce_same_state():
    """The radix prefix cache will likely call the standalone helper
    (no need to swap cache types). The subclass method is a sugar
    wrapper; the two must produce identical state for the same
    sequence of operations.
    """
    via_subclass = PositionedKVCache()
    via_helper = KVCache()
    k0, v0 = _make_keys_values(5, seed=51)
    via_subclass.update_and_fetch(k0, v0)
    positioned_update_and_fetch(via_helper, k0, v0)

    k1, v1 = _make_marked_keys_values([1.0, 2.0, 3.0])
    via_subclass.update_and_fetch(k1, v1, position_ids=[6, 6, 7])
    positioned_update_and_fetch(via_helper, k1, v1, position_ids=[6, 6, 7])

    assert via_subclass.offset == via_helper.offset == 8
    assert mx.array_equal(
        via_subclass.keys[..., :8, :], via_helper.keys[..., :8, :]
    ).item()
    assert mx.array_equal(
        via_subclass.values[..., :8, :], via_helper.values[..., :8, :]
    ).item()
