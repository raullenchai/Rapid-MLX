# SPDX-License-Identifier: Apache-2.0
"""Tests for model registry and multi-engine scenarios."""

import pytest

pytest.importorskip("mlx")


import functools
import gc
import os
import socket

from vllm_mlx import (
    EngineConfig,
    EngineCore,
    ModelOwnershipError,
    SamplingParams,
    SchedulerConfig,
    get_registry,
)

pytestmark = [pytest.mark.real_hf_cache, pytest.mark.requires_mlx]

# Use a small model for fast tests
TEST_MODEL = "mlx-community/Qwen3-0.6B-8bit"


@functools.cache
def _cached_model_and_tokenizer():
    from mlx_lm import load

    return load(TEST_MODEL)


@pytest.fixture(scope="module", autouse=True)
def _release_cached_model_and_tokenizer():
    """Return model memory when this integration module finishes."""
    yield
    _cached_model_and_tokenizer.cache_clear()
    gc.collect()
    import mlx.core as mx

    mx.clear_cache()


@pytest.fixture
def model_and_tokenizer():
    """Load once, but only after the function-scoped hermetic guard is armed."""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert getattr(socket.socket.connect, "_hermetic_network_guard", False)
    return _cached_model_and_tokenizer()


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_acquire_and_release(self, model_and_tokenizer):
        """Test basic acquire and release."""
        model, tokenizer = model_and_tokenizer
        registry = get_registry()

        engine = EngineCore(model, tokenizer, engine_id="test-engine-1")

        # Verify ownership
        is_owned, owner_id = registry.is_owned(model)
        assert is_owned
        assert owner_id == "test-engine-1"

        # Release
        engine.close()

        # After close(), the engine should no longer own the model
        assert not engine._owns_model

        # Note: is_owned() may still return True if weak ref is still alive,
        # but the engine's _owns_model flag should be False
        # The registry will clean up stale entries on next cleanup() call

    def test_force_ownership_transfer(self, model_and_tokenizer):
        """Test that new engine can force ownership from existing one."""
        model, tokenizer = model_and_tokenizer
        registry = get_registry()

        # First engine
        engine1 = EngineCore(model, tokenizer, engine_id="engine-1")
        assert engine1._owns_model

        # Second engine should take ownership (force=True is default)
        engine2 = EngineCore(model, tokenizer, engine_id="engine-2")
        assert engine2._owns_model

        # First engine should have been reset
        assert engine1.scheduler.batch_generator is None

        # Verify ownership transferred
        is_owned, owner_id = registry.is_owned(model)
        assert is_owned
        assert owner_id == "engine-2"

        # Cleanup
        engine2.close()

    def test_no_force_raises_error(self, model_and_tokenizer):
        """Test that force=False raises error when model is owned."""
        model, tokenizer = model_and_tokenizer

        engine1 = EngineCore(model, tokenizer, engine_id="engine-1")

        try:
            with pytest.raises(ModelOwnershipError):
                EngineCore(
                    model, tokenizer, engine_id="engine-2", force_model_ownership=False
                )
        finally:
            engine1.close()


class TestMultiEngine:
    """Tests for multi-engine scenarios."""

    def test_sequential_engines_with_close(self, model_and_tokenizer):
        """Test creating multiple engines sequentially with explicit close."""
        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=10)

        for i in range(3):
            engine = EngineCore(model, tokenizer, engine_id=f"seq-engine-{i}")
            try:
                result = engine.generate_batch_sync(["Hello"], params)
                assert len(result) == 1
                assert result[0].completion_tokens > 0
            finally:
                engine.close()

    def test_sequential_engines_without_close(self, model_and_tokenizer):
        """Test that engines work even without explicit close (force=True)."""
        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=10)

        # Create engines without closing - should still work due to force=True
        for i in range(3):
            engine = EngineCore(model, tokenizer, engine_id=f"noclose-engine-{i}")
            result = engine.generate_batch_sync(["Hello"], params)
            assert len(result) == 1
            assert result[0].completion_tokens > 0
            # Don't close - next engine should force ownership

        # Clean up last engine
        engine.close()

    def test_engine_gc_releases_model(self, model_and_tokenizer):
        """Test that garbage collection releases model ownership."""
        model, tokenizer = model_and_tokenizer
        registry = get_registry()

        # Create engine and immediately delete reference
        engine = EngineCore(model, tokenizer, engine_id="gc-test-engine")
        assert engine._owns_model

        # Delete and force GC
        del engine
        gc.collect()

        # Ownership should be released
        is_owned, _ = registry.is_owned(model)
        assert not is_owned


class TestCacheRecovery:
    """Tests for automatic cache error recovery."""

    def test_recovery_from_simulated_cache_corruption(self, model_and_tokenizer):
        """Test that scheduler recovers from cache corruption."""
        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=10)

        engine = EngineCore(model, tokenizer)

        try:
            # First generation should work
            result1 = engine.generate_batch_sync(["Hello"], params)
            assert result1[0].completion_tokens > 0

            # Simulate corruption by clearing batch generator
            engine.scheduler.batch_generator = None
            engine.scheduler._current_sampler_params = None

            # Should recover automatically
            result2 = engine.generate_batch_sync(["World"], params)
            assert result2[0].completion_tokens > 0
        finally:
            engine.close()

    def test_cache_validation_rejects_invalid(self, model_and_tokenizer):
        """Test that invalid caches are rejected."""
        model, tokenizer = model_and_tokenizer

        engine = EngineCore(model, tokenizer)

        try:
            # Test None cache
            assert not engine.scheduler._validate_cache(None)

            # Test empty list
            assert not engine.scheduler._validate_cache([])

            # Test list with None
            assert not engine.scheduler._validate_cache([None, None])

            # Note: valid caches are harder to test without actual generation
        finally:
            engine.close()


class TestBenchmarkScenario:
    """Tests simulating the benchmark script scenario."""

    def test_benchmark_like_usage(self, model_and_tokenizer):
        """Test usage pattern similar to benchmark_all_models.py."""
        model, tokenizer = model_and_tokenizer

        prompts = ["Hello", "World", "Test"]
        params = SamplingParams(max_tokens=20)

        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=256,
                prefill_batch_size=8,
                completion_batch_size=32,
            )
        )

        engine = EngineCore(model, tokenizer, config)

        try:
            # Single requests
            for p in prompts[:2]:
                result = engine.generate_batch_sync([p], params)[0]
                assert result.completion_tokens > 0

            # Reset and batch
            engine.scheduler.reset()
            results = engine.generate_batch_sync(prompts, params)
            assert len(results) == 3
            assert all(r.completion_tokens > 0 for r in results)

            # Reset and another batch
            engine.scheduler.reset()
            results = engine.generate_batch_sync(prompts, params)
            assert len(results) == 3
        finally:
            engine.close()

    def test_multiple_models_sequentially(self):
        """Test benchmarking multiple models in sequence."""
        from mlx_lm import load

        # Use the same small model twice to simulate different models
        models = [TEST_MODEL, TEST_MODEL]
        params = SamplingParams(max_tokens=10)

        for model_name in models:
            model, tokenizer = load(model_name)
            engine = EngineCore(model, tokenizer)

            try:
                result = engine.generate_batch_sync(["Hello"], params)
                assert result[0].completion_tokens > 0
            finally:
                engine.close()

            # Force GC between models
            del model, tokenizer, engine
            gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
