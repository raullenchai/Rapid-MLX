# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant KV cache (patches/turboquant_cache.py)."""

import mlx.core as mx
import numpy as np
import pytest

from vllm_mlx.patches.turboquant_cache import (
    TurboQuantKVCache,
    _dequantize,
    _load_codebook,
    _pack,
    _quantize,
    _rotation_matrix,
    _unpack,
)

# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    def test_orthogonality(self):
        Q = _rotation_matrix(128)
        product = np.array(Q @ Q.T, dtype=np.float32)
        np.testing.assert_allclose(product, np.eye(128), atol=1e-4)

    def test_deterministic(self):
        Q1 = np.array(_rotation_matrix(64, seed=42))
        Q2 = np.array(_rotation_matrix(64, seed=42))
        np.testing.assert_allclose(Q1, Q2, atol=1e-6)

    def test_different_seeds(self):
        Q1 = np.array(_rotation_matrix(64, seed=1))
        Q2 = np.array(_rotation_matrix(64, seed=2))
        assert not np.allclose(Q1, Q2)


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------


class TestCodebook:
    def test_3bit_size(self):
        c, b = _load_codebook(3, 128)
        assert c.shape == (8,)
        assert b.shape == (9,)  # boundaries include -5 and +5 sentinels

    def test_4bit_size(self):
        c, b = _load_codebook(4, 128)
        assert c.shape == (16,)
        assert b.shape == (17,)

    def test_scaling(self):
        c1, _ = _load_codebook(4, 64)
        c2, _ = _load_codebook(4, 128)
        # Larger dim = smaller scale
        assert float(mx.max(mx.abs(c1))) > float(mx.max(mx.abs(c2)))


# ---------------------------------------------------------------------------
# Pack / Unpack
# ---------------------------------------------------------------------------


class TestPackUnpack:
    def test_4bit_roundtrip(self):
        indices = mx.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=mx.uint8
        )
        packed = _pack(indices, 4)
        unpacked = _unpack(packed, 4, 16)
        np.testing.assert_array_equal(np.array(indices), np.array(unpacked))

    def test_3bit_roundtrip(self):
        indices = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1]], dtype=mx.uint8)
        packed = _pack(indices, 3)
        unpacked = _unpack(packed, 3, 10)
        np.testing.assert_array_equal(np.array(indices), np.array(unpacked))

    def test_2bit_roundtrip(self):
        indices = mx.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=mx.uint8)
        packed = _pack(indices, 2)
        unpacked = _unpack(packed, 2, 8)
        np.testing.assert_array_equal(np.array(indices), np.array(unpacked))


# ---------------------------------------------------------------------------
# Quantize / Dequantize roundtrip
# ---------------------------------------------------------------------------


class TestQuantizeDequantize:
    @pytest.fixture
    def setup_4bit(self):
        dim = 128
        c, b = _load_codebook(4, dim)
        R = _rotation_matrix(dim)
        return c, b, R, R.T, dim

    def test_4bit_roundtrip_quality(self, setup_4bit):
        c, b, R, Rt, dim = setup_4bit
        np.random.seed(0)
        vectors = mx.array(np.random.randn(1, 8, 32, dim).astype(np.float32))

        indices, norms = _quantize(vectors, Rt, b)
        reconstructed = _dequantize(indices, norms, R, c)

        orig = np.array(vectors.reshape(-1, dim))
        recon = np.array(reconstructed.reshape(-1, dim))
        cosines = np.sum(orig * recon, axis=-1) / (
            np.linalg.norm(orig, axis=-1) * np.linalg.norm(recon, axis=-1) + 1e-8
        )
        assert cosines.mean() > 0.95, f"4-bit cosine {cosines.mean():.4f} < 0.95"

    def test_3bit_roundtrip_quality(self):
        dim = 128
        c, b = _load_codebook(3, dim)
        R = _rotation_matrix(dim)
        np.random.seed(0)
        vectors = mx.array(np.random.randn(1, 8, 32, dim).astype(np.float32))

        indices, norms = _quantize(vectors, R.T, b)
        reconstructed = _dequantize(indices, norms, R, c)

        orig = np.array(vectors.reshape(-1, dim))
        recon = np.array(reconstructed.reshape(-1, dim))
        cosines = np.sum(orig * recon, axis=-1) / (
            np.linalg.norm(orig, axis=-1) * np.linalg.norm(recon, axis=-1) + 1e-8
        )
        assert cosines.mean() > 0.90, f"3-bit cosine {cosines.mean():.4f} < 0.90"


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------


class TestTurboQuantKVCache:
    def test_init(self):
        cache = TurboQuantKVCache(bits=4)
        assert cache.turbo_bits == 4
        assert cache.offset == 0
        assert cache.empty()

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            TurboQuantKVCache(bits=5)

    def test_update_and_fetch(self):
        cache = TurboQuantKVCache(bits=4)
        keys = mx.array(np.random.randn(1, 8, 16, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 8, 16, 128).astype(np.float32))

        out_k, out_v = cache.update_and_fetch(keys, values)
        assert out_k.shape == keys.shape
        assert out_v.shape == values.shape
        assert cache.offset == 16

    def test_incremental_update(self):
        cache = TurboQuantKVCache(bits=4)
        k1 = mx.array(np.random.randn(1, 4, 8, 64).astype(np.float32))
        v1 = mx.array(np.random.randn(1, 4, 8, 64).astype(np.float32))
        cache.update_and_fetch(k1, v1)
        assert cache.offset == 8

        k2 = mx.array(np.random.randn(1, 4, 1, 64).astype(np.float32))
        v2 = mx.array(np.random.randn(1, 4, 1, 64).astype(np.float32))
        out_k, out_v = cache.update_and_fetch(k2, v2)
        assert cache.offset == 9
        assert out_k.shape == (1, 4, 9, 64)

    def test_quality_after_update(self):
        cache = TurboQuantKVCache(bits=4)
        np.random.seed(0)
        keys = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float32))

        out_k, out_v = cache.update_and_fetch(keys, values)

        orig_k = np.array(keys.reshape(-1, 128))
        recon_k = np.array(out_k.reshape(-1, 128))
        cosines = np.sum(orig_k * recon_k, axis=-1) / (
            np.linalg.norm(orig_k, axis=-1) * np.linalg.norm(recon_k, axis=-1) + 1e-8
        )
        assert cosines.mean() > 0.95

    def test_memory_savings(self):
        cache = TurboQuantKVCache(bits=4)
        keys = mx.array(np.random.randn(1, 8, 64, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 8, 64, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)

        fp16_bytes = keys.nbytes + values.nbytes
        tq_bytes = cache.nbytes
        ratio = tq_bytes / fp16_bytes
        assert ratio < 0.50, f"Ratio {ratio:.2f} should be < 0.50"

    def test_trim(self):
        cache = TurboQuantKVCache(bits=4)
        keys = mx.array(np.random.randn(1, 4, 20, 64).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 20, 64).astype(np.float32))
        cache.update_and_fetch(keys, values)
        assert cache.offset == 20

        trimmed = cache.trim(5)
        assert trimmed == 5
        assert cache.offset == 15

    def test_is_trimmable(self):
        assert TurboQuantKVCache(bits=4).is_trimmable()

    def test_state_roundtrip(self):
        cache = TurboQuantKVCache(bits=4)
        keys = mx.array(np.random.randn(1, 4, 16, 64).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 16, 64).astype(np.float32))
        cache.update_and_fetch(keys, values)

        # Save state
        state = cache.state
        meta = cache.meta_state

        # Restore into new cache
        cache2 = TurboQuantKVCache(bits=4)
        cache2.meta_state = meta
        cache2.state = state

        assert cache2.offset == 16
        assert cache2.turbo_bits == 4

    def test_nbytes_empty(self):
        cache = TurboQuantKVCache(bits=4)
        assert cache.nbytes == 0

    def test_size(self):
        cache = TurboQuantKVCache(bits=4)
        assert cache.size() == 0
        keys = mx.array(np.random.randn(1, 4, 10, 64).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 10, 64).astype(np.float32))
        cache.update_and_fetch(keys, values)
        assert cache.size() == 10
