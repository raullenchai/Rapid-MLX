# SPDX-License-Identifier: Apache-2.0
"""Tests for TurboQuant KV cache compression."""

import mlx.core as mx
import numpy as np
import pytest

from vllm_mlx.turboquant import (
    LLOYD_MAX_BOUNDARIES,
    LLOYD_MAX_CODEBOOKS,
    TurboQuantConfig,
    TurboQuantKVCache,
    auto_select_bits,
    generate_rotation_matrix,
    turboquant_decode,
    turboquant_encode,
)

# ---------------------------------------------------------------------------
# TurboQuantConfig
# ---------------------------------------------------------------------------


class TestTurboQuantConfig:
    def test_valid_3bit(self):
        cfg = TurboQuantConfig(bits=3)
        assert cfg.bits == 3

    def test_valid_4bit(self):
        cfg = TurboQuantConfig(bits=4)
        assert cfg.bits == 4

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be 3 or 4"):
            TurboQuantConfig(bits=2)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            TurboQuantConfig(group_size=0)

    def test_defaults(self):
        cfg = TurboQuantConfig()
        assert cfg.bits == 3
        assert cfg.group_size == 32
        assert cfg.rotation_seed == 42


# ---------------------------------------------------------------------------
# auto_select_bits
# ---------------------------------------------------------------------------


class TestAutoSelectBits:
    def test_large_head_dim(self):
        assert auto_select_bits(128) == 3

    def test_medium_head_dim(self):
        assert auto_select_bits(96) == 3

    def test_small_head_dim(self):
        assert auto_select_bits(64) == 4

    def test_tiny_head_dim(self):
        assert auto_select_bits(32) == 4


# ---------------------------------------------------------------------------
# Lloyd-Max codebooks
# ---------------------------------------------------------------------------


class TestLloydMaxCodebooks:
    def test_3bit_size(self):
        assert LLOYD_MAX_CODEBOOKS[3].shape == (8,)

    def test_4bit_size(self):
        assert LLOYD_MAX_CODEBOOKS[4].shape == (16,)

    def test_3bit_boundaries_size(self):
        assert LLOYD_MAX_BOUNDARIES[3].shape == (7,)

    def test_4bit_boundaries_size(self):
        assert LLOYD_MAX_BOUNDARIES[4].shape == (15,)

    def test_codebook_sorted(self):
        for bits in (3, 4):
            cb = np.array(LLOYD_MAX_CODEBOOKS[bits])
            assert np.all(cb[:-1] <= cb[1:]), f"{bits}-bit codebook not sorted"

    def test_boundaries_sorted(self):
        for bits in (3, 4):
            bd = np.array(LLOYD_MAX_BOUNDARIES[bits])
            assert np.all(bd[:-1] <= bd[1:]), f"{bits}-bit boundaries not sorted"

    def test_codebook_symmetric(self):
        """Codebook should be approximately symmetric around 0."""
        for bits in (3, 4):
            cb = np.array(LLOYD_MAX_CODEBOOKS[bits])
            assert abs(cb.sum()) < 0.1, f"{bits}-bit codebook not symmetric"


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    def test_orthogonality(self):
        """Q @ Q.T should be identity."""
        Q = generate_rotation_matrix(128, seed=42)
        Q_np = np.array(Q, dtype=np.float32)
        product = Q_np @ Q_np.T
        np.testing.assert_allclose(product, np.eye(128), atol=1e-5)

    def test_deterministic(self):
        """Same seed and dim should produce same matrix."""
        Q1 = generate_rotation_matrix(64, seed=123)
        Q2 = generate_rotation_matrix(64, seed=123)
        np.testing.assert_array_equal(np.array(Q1), np.array(Q2))

    def test_different_seeds(self):
        """Different seeds should produce different matrices."""
        Q1 = generate_rotation_matrix(64, seed=1)
        Q2 = generate_rotation_matrix(64, seed=2)
        assert not np.allclose(np.array(Q1), np.array(Q2))

    def test_different_dims(self):
        Q64 = generate_rotation_matrix(64, seed=42)
        Q128 = generate_rotation_matrix(128, seed=42)
        assert Q64.shape == (64, 64)
        assert Q128.shape == (128, 128)

    def test_caching(self):
        """Second call should return cached result."""
        Q1 = generate_rotation_matrix(32, seed=99)
        Q2 = generate_rotation_matrix(32, seed=99)
        # Should be the exact same object (cached)
        assert Q1 is Q2


# ---------------------------------------------------------------------------
# Encode / Decode roundtrip
# ---------------------------------------------------------------------------


class TestEncodeDecode:
    @pytest.fixture
    def rotation_128(self):
        return generate_rotation_matrix(128, seed=42)

    @pytest.fixture
    def rotation_64(self):
        return generate_rotation_matrix(64, seed=42)

    @pytest.fixture
    def gaussian_data_128(self):
        """Simulate V tensor: (1, 8, 32, 128) — batch=1, 8 heads, 32 tokens, head_dim=128."""
        np.random.seed(0)
        return mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))

    @pytest.fixture
    def gaussian_data_64(self):
        np.random.seed(0)
        return mx.array(np.random.randn(1, 8, 32, 64).astype(np.float16))

    def test_4bit_roundtrip_quality_128(self, gaussian_data_128, rotation_128):
        indices, scales, zeros = turboquant_encode(
            gaussian_data_128, bits=4, group_size=32, rotation=rotation_128
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=4,
            group_size=32,
            rotation=rotation_128,
            head_dim=128,
        )

        # Cosine similarity per vector
        orig = np.array(gaussian_data_128.reshape(-1, 128), dtype=np.float32)
        recon = np.array(reconstructed.reshape(-1, 128), dtype=np.float32)
        cosines = np.sum(orig * recon, axis=-1) / (
            np.linalg.norm(orig, axis=-1) * np.linalg.norm(recon, axis=-1) + 1e-8
        )
        mean_cosine = cosines.mean()
        assert mean_cosine > 0.95, f"4-bit cosine {mean_cosine:.4f} < 0.95"

    def test_3bit_roundtrip_quality_128(self, gaussian_data_128, rotation_128):
        indices, scales, zeros = turboquant_encode(
            gaussian_data_128, bits=3, group_size=32, rotation=rotation_128
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=3,
            group_size=32,
            rotation=rotation_128,
            head_dim=128,
        )

        orig = np.array(gaussian_data_128.reshape(-1, 128), dtype=np.float32)
        recon = np.array(reconstructed.reshape(-1, 128), dtype=np.float32)
        cosines = np.sum(orig * recon, axis=-1) / (
            np.linalg.norm(orig, axis=-1) * np.linalg.norm(recon, axis=-1) + 1e-8
        )
        mean_cosine = cosines.mean()
        assert mean_cosine > 0.90, f"3-bit cosine {mean_cosine:.4f} < 0.90"

    def test_4bit_roundtrip_quality_64(self, gaussian_data_64, rotation_64):
        """head_dim=64 needs 4-bit for decent quality."""
        indices, scales, zeros = turboquant_encode(
            gaussian_data_64, bits=4, group_size=32, rotation=rotation_64
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=4,
            group_size=32,
            rotation=rotation_64,
            head_dim=64,
        )

        orig = np.array(gaussian_data_64.reshape(-1, 64), dtype=np.float32)
        recon = np.array(reconstructed.reshape(-1, 64), dtype=np.float32)
        cosines = np.sum(orig * recon, axis=-1) / (
            np.linalg.norm(orig, axis=-1) * np.linalg.norm(recon, axis=-1) + 1e-8
        )
        mean_cosine = cosines.mean()
        assert mean_cosine > 0.93, f"4-bit head_dim=64 cosine {mean_cosine:.4f} < 0.93"

    def test_4bit_mse(self, gaussian_data_128, rotation_128):
        """MSE should be low for 4-bit."""
        indices, scales, zeros = turboquant_encode(
            gaussian_data_128, bits=4, group_size=32, rotation=rotation_128
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=4,
            group_size=32,
            rotation=rotation_128,
            head_dim=128,
        )
        mse = float(mx.mean((gaussian_data_128 - reconstructed) ** 2))
        assert mse < 0.05, f"4-bit MSE {mse:.4f} > 0.05"

    def test_3bit_mse(self, gaussian_data_128, rotation_128):
        indices, scales, zeros = turboquant_encode(
            gaussian_data_128, bits=3, group_size=32, rotation=rotation_128
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=3,
            group_size=32,
            rotation=rotation_128,
            head_dim=128,
        )
        mse = float(mx.mean((gaussian_data_128 - reconstructed) ** 2))
        assert mse < 0.15, f"3-bit MSE {mse:.4f} > 0.15"

    def test_indices_dtype(self, gaussian_data_128, rotation_128):
        indices, _, _ = turboquant_encode(
            gaussian_data_128, bits=4, group_size=32, rotation=rotation_128
        )
        assert indices.dtype == mx.uint8

    def test_packed_indices_range_4bit(self, gaussian_data_128, rotation_128):
        """Packed indices are uint8 with nibble-packed values."""
        packed, _, _ = turboquant_encode(
            gaussian_data_128, bits=4, group_size=32, rotation=rotation_128
        )
        assert packed.dtype == mx.uint8
        # Each byte has high nibble + low nibble, each in [0,15]
        assert int(mx.max(packed)) <= 255

    def test_packed_indices_range_3bit(self, gaussian_data_128, rotation_128):
        packed, _, _ = turboquant_encode(
            gaussian_data_128, bits=3, group_size=32, rotation=rotation_128
        )
        assert packed.dtype == mx.uint8

    def test_output_shapes(self, gaussian_data_128, rotation_128):
        """Verify output shapes are correct (packed indices)."""
        packed, scales, zeros = turboquant_encode(
            gaussian_data_128, bits=3, group_size=32, rotation=rotation_128
        )
        # packed indices: last dim = ceil(head_dim/2) due to nibble packing
        assert packed.shape == (1, 8, 32, 64)  # 128 // 2
        # scales/zeros: (..., seq_len, n_groups)
        n_groups = 128 // 32  # = 4
        assert scales.shape == (1, 8, 32, n_groups)
        assert zeros.shape == (1, 8, 32, n_groups)

    def test_non_divisible_group_size(self):
        """head_dim not divisible by group_size should still work."""
        np.random.seed(0)
        data = mx.array(np.random.randn(1, 4, 16, 100).astype(np.float16))
        rotation = generate_rotation_matrix(100, seed=42)

        indices, scales, zeros = turboquant_encode(
            data, bits=4, group_size=32, rotation=rotation
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=4,
            group_size=32,
            rotation=rotation,
            head_dim=100,
        )
        assert reconstructed.shape == data.shape

    def test_single_token(self):
        """Single-token V should work."""
        np.random.seed(0)
        data = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float16))
        rotation = generate_rotation_matrix(128, seed=42)

        indices, scales, zeros = turboquant_encode(
            data, bits=4, group_size=32, rotation=rotation
        )
        reconstructed = turboquant_decode(
            indices,
            scales,
            zeros,
            bits=4,
            group_size=32,
            rotation=rotation,
            head_dim=128,
        )
        assert reconstructed.shape == data.shape


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------


class TestTurboQuantKVCache:
    @pytest.fixture
    def mock_kv_cache(self):
        """Create a mock KVCache-like object."""
        from unittest.mock import MagicMock

        kv = MagicMock()
        np.random.seed(0)
        kv.keys = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))
        kv.values = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))
        kv.offset = 32
        return kv

    @pytest.fixture
    def config(self):
        return TurboQuantConfig(bits=4, group_size=32)

    def test_from_kv_cache(self, mock_kv_cache, config):
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        assert tq.keys is not None
        assert tq.values_compressed[0] is not None  # indices
        assert tq.offset == 32
        assert tq.head_dim == 128

    def test_to_kv_cache_roundtrip(self, mock_kv_cache, config):
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        restored = tq.to_kv_cache()

        # Keys should be identical (FP16, no compression)
        np.testing.assert_array_equal(
            np.array(restored.keys), np.array(mock_kv_cache.keys)
        )

        # Values should be close (compressed + decompressed)
        orig = np.array(mock_kv_cache.values, dtype=np.float32)
        recon = np.array(restored.values, dtype=np.float32)
        cosines = np.sum(orig.reshape(-1, 128) * recon.reshape(-1, 128), axis=-1) / (
            np.linalg.norm(orig.reshape(-1, 128), axis=-1)
            * np.linalg.norm(recon.reshape(-1, 128), axis=-1)
            + 1e-8
        )
        assert cosines.mean() > 0.93

    def test_keys_unchanged(self, mock_kv_cache, config):
        """K must stay FP16, not be compressed."""
        original_keys = np.array(mock_kv_cache.keys)
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        np.testing.assert_array_equal(np.array(tq.keys), original_keys)

    def test_memory_savings(self, mock_kv_cache, config):
        """Compressed V should use less memory than FP16 V."""
        fp16_v_bytes = mock_kv_cache.values.nbytes
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)

        indices, scales, zeros = tq.values_compressed
        compressed_bytes = indices.nbytes + scales.nbytes + zeros.nbytes

        ratio = compressed_bytes / fp16_v_bytes
        # Nibble-packed indices (half size) + fp16 scales/zeros: ~31% of FP16 V
        assert ratio < 0.40, f"Compression ratio {ratio:.2f} > 0.40"

    def test_3bit_memory_savings(self, mock_kv_cache):
        config3 = TurboQuantConfig(bits=3, group_size=32)
        fp16_v_bytes = mock_kv_cache.values.nbytes
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config3)

        indices, scales, zeros = tq.values_compressed
        compressed_bytes = indices.nbytes + scales.nbytes + zeros.nbytes

        ratio = compressed_bytes / fp16_v_bytes
        assert ratio < 0.40, f"3-bit ratio {ratio:.2f} > 0.40"

    def test_is_trimmable(self, mock_kv_cache, config):
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        assert tq.is_trimmable()

    def test_trim(self, mock_kv_cache, config):
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        tq.trim(10)
        assert tq.offset == 22
        assert tq.keys.shape[-2] == 22

    def test_trim_all(self, mock_kv_cache, config):
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        tq.trim(100)  # More than offset
        assert tq.offset == 0

    def test_empty_cache(self, config):
        from unittest.mock import MagicMock

        kv = MagicMock()
        kv.keys = None
        kv.values = None
        kv.offset = 0

        tq = TurboQuantKVCache.from_kv_cache(kv, config)
        assert tq.keys is None
        assert tq.offset == 0

        restored = tq.to_kv_cache()
        assert restored.keys is None

    def test_memory_bytes_property(self, mock_kv_cache, config):
        tq = TurboQuantKVCache.from_kv_cache(mock_kv_cache, config)
        mem = tq.memory_bytes
        assert mem > 0
        # Should be less than FP16 keys + FP16 values
        fp16_total = mock_kv_cache.keys.nbytes + mock_kv_cache.values.nbytes
        assert mem < fp16_total


# ---------------------------------------------------------------------------
# Integration: memory_cache compress/decompress
# ---------------------------------------------------------------------------


class TestMemoryCacheIntegration:
    """Test TurboQuant wiring in memory_cache.py."""

    def _make_cache_list(self, n_layers=4, seq_len=32, n_heads=8, head_dim=128):
        """Create a list of real KVCache layers."""
        from mlx_lm.models.cache import KVCache

        cache = []
        np.random.seed(0)
        for _ in range(n_layers):
            kv = KVCache()
            kv.keys = mx.array(
                np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
            )
            kv.values = mx.array(
                np.random.randn(1, n_heads, seq_len, head_dim).astype(np.float16)
            )
            kv.offset = seq_len
            cache.append(kv)
        return cache

    def test_compress_decompress_roundtrip(self):
        """Compress then decompress should produce valid KVCache layers."""
        from vllm_mlx.memory_cache import (
            _turboquant_compress_cache,
            _turboquant_decompress_cache,
        )

        cache = self._make_cache_list()
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)

        # All layers should be TurboQuantKVCache
        for layer in compressed:
            assert isinstance(layer, TurboQuantKVCache)

        decompressed = _turboquant_decompress_cache(compressed)

        # All layers should have keys and values
        for layer in decompressed:
            assert layer.keys is not None
            assert layer.values is not None

    def test_compress_memory_reduction(self):
        """Compressed cache should use less total memory."""
        from vllm_mlx.memory_cache import (
            _turboquant_compress_cache,
            estimate_kv_cache_memory,
        )

        cache = self._make_cache_list()
        original_mem = sum(layer.keys.nbytes + layer.values.nbytes for layer in cache)

        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)
        compressed_mem = estimate_kv_cache_memory(compressed)

        # Compressed should be significantly smaller
        ratio = compressed_mem / original_mem
        assert ratio < 0.75, f"Compression ratio {ratio:.2f} > 0.75"
        assert compressed_mem > 0, "Memory estimate should not be 0"

    def test_none_layers_passthrough(self):
        """None layers should pass through unchanged."""
        from vllm_mlx.memory_cache import (
            _turboquant_compress_cache,
            _turboquant_decompress_cache,
        )

        cache = [None, None]
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)
        assert compressed == [None, None]

        decompressed = _turboquant_decompress_cache(compressed)
        assert decompressed == [None, None]

    def test_mixed_layers(self):
        """Non-KVCache layers should pass through unchanged."""
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import _turboquant_compress_cache

        # Create a mix: KVCache + non-KVCache
        kv = KVCache()
        np.random.seed(0)
        kv.keys = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))
        kv.values = mx.array(np.random.randn(1, 8, 32, 128).astype(np.float16))
        kv.offset = 32

        mamba = MagicMock()  # Not a KVCache instance

        cache = [kv, mamba, None]
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)

        assert isinstance(compressed[0], TurboQuantKVCache)
        assert compressed[1] is mamba  # Passed through
        assert compressed[2] is None  # Passed through

    def test_trim_cache_offset_with_turboquant(self):
        """_trim_cache_offset should trim TurboQuantKVCache without mutating original."""
        from vllm_mlx.memory_cache import (
            _trim_cache_offset,
            _turboquant_compress_cache,
        )

        cache = self._make_cache_list(n_layers=2, seq_len=32)
        compressed = _turboquant_compress_cache(cache, bits=4, group_size=32)

        # Save original offsets
        orig_offsets = [c.offset for c in compressed]
        orig_keys_shapes = [c.keys.shape for c in compressed]

        # Trim 10 tokens
        trimmed = _trim_cache_offset(compressed, trim_by=10)

        # Trimmed copies should have reduced offset
        for tc in trimmed:
            assert tc.offset == 22  # 32 - 10

        # Original entries must NOT be mutated
        for i, c in enumerate(compressed):
            assert c.offset == orig_offsets[i]
            assert c.keys.shape == orig_keys_shapes[i]

    def test_trim_cache_offset_mixed_layers(self):
        """_trim_cache_offset handles mixed TurboQuantKVCache + other layers."""
        from unittest.mock import MagicMock

        from mlx_lm.models.cache import KVCache

        from vllm_mlx.memory_cache import (
            _trim_cache_offset,
            _turboquant_compress_cache,
        )

        # Create KVCache + non-KVCache
        kv = KVCache()
        kv.keys = mx.array(np.random.randn(1, 4, 20, 128).astype(np.float16))
        kv.values = mx.array(np.random.randn(1, 4, 20, 128).astype(np.float16))
        kv.offset = 20

        other = MagicMock()

        compressed = _turboquant_compress_cache([kv], bits=4, group_size=32)
        mixed = compressed + [other, None]

        trimmed = _trim_cache_offset(mixed, trim_by=5)
        assert len(trimmed) == 3
        assert trimmed[0].offset == 15  # TurboQuantKVCache trimmed
        assert trimmed[2] is None
