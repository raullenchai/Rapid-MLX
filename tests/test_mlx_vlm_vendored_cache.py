# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the three documented upstream-bugfix deviations in the
vendored mlx-vlm cache module (see the package ``__init__.py`` provenance).
Each test fails against the byte-verbatim upstream 0.7.1 source.
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.models.mlx_vlm_vendored.cache import (
    BatchPoolingCache,
    BatchRotatingKVCache,
)


def test_batch_rotating_merge_with_content():
    """Upstream passed ``c.keys`` to the zero-arg in-place
    ``_temporal_order``, so every merge with content raised TypeError."""
    a = BatchRotatingKVCache(16, [0])
    keys_a = mx.arange(2 * 6 * 8, dtype=mx.float32).reshape(1, 2, 6, 8)
    a._update_concat(keys_a, mx.zeros((1, 2, 6, 8)))
    b = BatchRotatingKVCache(16, [0])
    b._update_concat(mx.zeros((1, 2, 4, 8)), mx.zeros((1, 2, 4, 8)))

    merged = BatchRotatingKVCache.merge([a, b])

    assert merged.batch_size == 2
    assert merged.keys.shape == (2, 2, 6, 8)
    # Row 0 carries cache a's tokens; row 1 is left-padded to the longest
    # member.
    assert mx.array_equal(merged.keys[0], keys_a[0])


def test_batch_rotating_merge_rotated_operand_is_temporal():
    """A wrapped (rotated) operand must land in the merged cache in
    chronological order, not ring order."""
    a = BatchRotatingKVCache(8, [0])
    # Fill the window with single-token decode steps, then wrap it: after
    # 12 steps the 8-token ring holds tokens 4..11 with the write cursor at 4.
    for t in range(12):
        k = mx.full((1, 2, 1, 4), float(t))
        a.update_and_fetch(k, mx.zeros((1, 2, 1, 4)))
    assert a.rotated

    merged = BatchRotatingKVCache.merge([a])

    # Temporal order restores tokens 4..11 in ascending sequence.
    expected = mx.arange(8, dtype=mx.float32).reshape(1, 8, 1) + 4.0
    expected = mx.broadcast_to(expected, (2, 8, 4))
    assert mx.array_equal(merged.keys[0], expected)


def test_batch_rotating_merge_unrotated_preallocation_uses_live_prefix():
    """Single-token decode preallocates capacity beyond the live prefix;
    merge must not copy the unused zero-filled tail."""
    cache = BatchRotatingKVCache(8, [0])
    for token in (11.0, 12.0, 13.0):
        item = mx.full((1, 2, 1, 4), token)
        cache.update_and_fetch(item, item)

    assert cache.rotated is False
    assert cache.keys.shape[2] == 8
    assert cache.size() == 3

    merged = BatchRotatingKVCache.merge([cache])

    expected = mx.array([11.0, 12.0, 13.0]).reshape(1, 3, 1)
    expected = mx.broadcast_to(expected, (2, 3, 4))
    assert mx.array_equal(merged.keys[0], expected)
    assert mx.array_equal(merged.values[0], expected)


def test_batch_pooling_make_mask_scalar_offset_matches_array_branch():
    """Upstream's scalar-offset branch added ``offset`` twice; the absolute
    query positions must match the ``mx.array`` branch semantics."""
    cache = BatchPoolingCache.__new__(BatchPoolingCache)
    cache.ratio = 4
    P = 3
    cache.pooled = mx.zeros((1, P, 2))
    cache._pool_lengths = [P]
    cache._lengths = [2**31]
    cache.remainder = [0]

    offset = 8
    scalar = cache.make_mask(L=3, offset=offset)
    array = cache.make_mask(L=3, offset=mx.array([offset]))

    assert scalar is not None and array is not None
    assert scalar.shape == array.shape == (1, 3, P)
    assert mx.array_equal(scalar, array)
    # Query 0 sits at absolute position 9: it sees pooled tokens with
    # index < 9 // 4 == 2. (Upstream's double-offset build admitted every
    # pooled token from the first query.)
    assert mx.array_equal(scalar[0, 0], mx.array([True, True, False]))
    # Query 2 sits at position 11: 11 // 4 == 2 — still index < 2.
    assert mx.array_equal(scalar[0, 2], mx.array([True, True, False]))


def test_batch_rotating_meta_state_roundtrip_restores_rotated_flag():
    """Upstream serialized ``rotated`` with ``str()`` but parsed it with
    ``bool()``, so the string "False" restored as True and the next
    ``_temporal_order`` would roll a cache that was never rotated."""
    plain = BatchRotatingKVCache(16, [0])
    plain._update_concat(mx.zeros((1, 2, 6, 8)), mx.zeros((1, 2, 6, 8)))
    assert plain.rotated is False
    restored = BatchRotatingKVCache.from_state(plain.state, plain.meta_state)
    assert restored.rotated is False
    assert restored._offset == plain._offset and restored._idx == plain._idx

    a = BatchRotatingKVCache(8, [0])
    for t in range(9):
        a.update_and_fetch(mx.full((1, 2, 1, 4), float(t)), mx.zeros((1, 2, 1, 4)))
    assert a.rotated is True
    restored = BatchRotatingKVCache.from_state(a.state, a.meta_state)
    assert restored.rotated is True
    assert restored._idx == a._idx and restored._offset == a._offset
