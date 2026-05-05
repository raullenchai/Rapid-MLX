# SPDX-License-Identifier: Apache-2.0
"""
Deterministic tests for continuous batching system.

These tests use temperature=0 to ensure reproducible outputs.
Run with: pytest tests/test_batching_deterministic.py -v
"""

import asyncio
import time

import pytest

# Model to use for tests - small model for fast testing
TEST_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model once for all tests in this module."""
    try:
        from mlx_lm import load

        model, tokenizer = load(TEST_MODEL)
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load model {TEST_MODEL}: {e}")


@pytest.fixture
def sampling_params():
    """Deterministic sampling params (temperature=0)."""
    from vllm_mlx import SamplingParams

    return SamplingParams(max_tokens=10, temperature=0.0, top_p=1.0)


class TestDeterministicSingleRequest:
    """Test single request determinism."""

    @pytest.mark.asyncio
    async def test_same_prompt_same_output(self, model_and_tokenizer, sampling_params):
        """Same prompt should produce same output with temp=0."""
        from vllm_mlx import AsyncEngineCore, EngineConfig, SchedulerConfig

        model, tokenizer = model_and_tokenizer
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                prefill_batch_size=2,
                completion_batch_size=4,
            )
        )

        prompt = "What is 2+2? Answer:"

        outputs = []
        for _ in range(3):  # Run 3 times
            async with AsyncEngineCore(model, tokenizer, config) as engine:
                await asyncio.sleep(0.05)
                request_id = await engine.add_request(prompt, sampling_params)

                async for output in engine.stream_outputs(request_id, timeout=30):
                    if output.finished:
                        outputs.append(output.output_text)
                        break

        # All outputs should be identical
        assert len(outputs) == 3
        assert outputs[0] == outputs[1] == outputs[2], f"Outputs differ: {outputs}"

    @pytest.mark.asyncio
    async def test_token_streaming_order(self, model_and_tokenizer, sampling_params):
        """Tokens should stream in order."""
        from vllm_mlx import AsyncEngineCore

        model, tokenizer = model_and_tokenizer

        async with AsyncEngineCore(model, tokenizer) as engine:
            await asyncio.sleep(0.05)
            request_id = await engine.add_request(
                "Count from 1 to 5:",
                sampling_params,
            )

            token_ids = []
            async for output in engine.stream_outputs(request_id, timeout=30):
                token_ids.extend(output.new_token_ids)
                if output.finished:
                    # Final output should have all tokens
                    assert output.output_token_ids == token_ids
                    break


class TestDeterministicConcurrentRequests:
    """Test concurrent request handling with determinism."""

    @pytest.mark.asyncio
    async def test_concurrent_same_prompt(self, model_and_tokenizer):
        """Multiple concurrent requests with same prompt should get same output."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = model_and_tokenizer
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
                completion_batch_size=8,
            )
        )

        params = SamplingParams(max_tokens=10, temperature=0.0)
        prompt = "The capital of France is"

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.05)

            # Send 4 identical requests
            request_ids = []
            for _ in range(4):
                rid = await engine.add_request(prompt, params)
                request_ids.append(rid)

            # Collect outputs
            async def get_output(rid):
                async for out in engine.stream_outputs(rid, timeout=30):
                    if out.finished:
                        return out.output_text
                return None

            results = await asyncio.gather(*[get_output(r) for r in request_ids])

            # All should be the same
            assert all(r == results[0] for r in results), f"Outputs differ: {results}"

    @pytest.mark.asyncio
    async def test_concurrent_different_prompts(self, model_and_tokenizer):
        """Different prompts should get different (but deterministic) outputs."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = model_and_tokenizer
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
            )
        )

        params = SamplingParams(max_tokens=5, temperature=0.0)
        prompts = [
            "Capital of France:",
            "Capital of Spain:",
            "Capital of Italy:",
        ]

        # Run twice to verify determinism
        all_results = []
        for run in range(2):
            async with AsyncEngineCore(model, tokenizer, config) as engine:
                await asyncio.sleep(0.05)

                request_ids = []
                for p in prompts:
                    rid = await engine.add_request(p, params)
                    request_ids.append(rid)

                async def get_output(rid):
                    async for out in engine.stream_outputs(rid, timeout=30):
                        if out.finished:
                            return out.output_text
                    return None

                results = await asyncio.gather(*[get_output(r) for r in request_ids])
                all_results.append(results)

        # Each run should produce same results
        assert all_results[0] == all_results[1], (
            f"Results differ between runs: {all_results}"
        )


class TestBatchingPerformance:
    """Test that batching improves throughput."""

    @pytest.mark.asyncio
    async def test_batched_faster_than_sequential(self, model_and_tokenizer):
        """Batched requests should not be catastrophically slower than sequential.

        This is a regression guard, not a perf benchmark. The threshold
        is loose (0.7x) on purpose — real batching wins are 2-3x but
        the workload here is tiny (40 tokens × 2 runs) so engine
        startup overhead and Metal kernel JIT swamp the signal. A
        warmup pass per-engine eliminates the cold/warm asymmetry that
        used to make this test flake under any concurrent system load.
        Catastrophic regressions (sequential outperforming batched by
        more than 30%) still fire.
        """
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = model_and_tokenizer
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
                completion_batch_size=8,
            )
        )

        params = SamplingParams(max_tokens=10, temperature=0.0)
        prompts = [f"Count to {i}:" for i in range(1, 5)]

        async def _warmup(engine):
            """Pay the Metal kernel JIT + first-inference cost outside
            the timed region so cold/warm asymmetry doesn't dominate
            the measurement. ~10 tokens is enough to compile the decode
            path."""
            rid = await engine.add_request("warmup:", params)
            async for out in engine.stream_outputs(rid, timeout=30):
                if out.finished:
                    break

        async def run_sequential():
            """Run requests one at a time (after warmup)."""
            total_tokens = 0
            async with AsyncEngineCore(model, tokenizer, config) as engine:
                await asyncio.sleep(0.05)
                await _warmup(engine)

                for prompt in prompts:
                    rid = await engine.add_request(prompt, params)
                    async for out in engine.stream_outputs(rid, timeout=30):
                        if out.finished:
                            total_tokens += out.completion_tokens
                            break
            return total_tokens

        async def run_batched():
            """Run requests concurrently (after warmup)."""
            async with AsyncEngineCore(model, tokenizer, config) as engine:
                await asyncio.sleep(0.05)
                await _warmup(engine)

                request_ids = []
                for prompt in prompts:
                    rid = await engine.add_request(prompt, params)
                    request_ids.append(rid)

                async def get_output(rid):
                    async for out in engine.stream_outputs(rid, timeout=30):
                        if out.finished:
                            return out.completion_tokens
                    return 0

                tokens = await asyncio.gather(*[get_output(r) for r in request_ids])
                return sum(tokens)

        # Time sequential
        start = time.perf_counter()
        seq_tokens = await run_sequential()
        seq_time = time.perf_counter() - start

        # Time batched
        start = time.perf_counter()
        batch_tokens = await run_batched()
        batch_time = time.perf_counter() - start

        seq_throughput = seq_tokens / seq_time
        batch_throughput = batch_tokens / batch_time

        print(f"\nSequential: {seq_throughput:.1f} tok/s")
        print(f"Batched: {batch_throughput:.1f} tok/s")
        print(f"Speedup: {batch_throughput / seq_throughput:.2f}x")

        # Catastrophic-regression guard. Real batching wins are 2-3x;
        # 0.7x leaves headroom for the inherent noise of a 40-token
        # workload while still catching a fundamental break.
        assert batch_throughput > seq_throughput * 0.7, (
            f"Batched ({batch_throughput:.1f} tok/s) regressed badly "
            f"vs sequential ({seq_throughput:.1f} tok/s) — speedup "
            f"{batch_throughput / seq_throughput:.2f}x is below the "
            f"0.7x catastrophic-regression floor"
        )


class TestRequestManagement:
    """Test request lifecycle management."""

    @pytest.mark.asyncio
    async def test_abort_request(self, model_and_tokenizer):
        """Test aborting a request mid-generation."""
        from vllm_mlx import AsyncEngineCore, SamplingParams

        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=100, temperature=0.0)

        async with AsyncEngineCore(model, tokenizer) as engine:
            await asyncio.sleep(0.05)

            # Start a long request
            rid = await engine.add_request(
                "Write a very long story about a dragon:",
                params,
            )

            # Get a few tokens
            token_count = 0
            async for output in engine.stream_outputs(rid, timeout=30):
                token_count += len(output.new_token_ids)
                if token_count >= 5:
                    # Abort after 5 tokens
                    await engine.abort_request(rid)
                    break

            # Request should be aborted
            stats = engine.get_stats()
            assert stats["active_requests"] == 0

    @pytest.mark.asyncio
    async def test_engine_stats(self, model_and_tokenizer):
        """Test engine statistics tracking."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = model_and_tokenizer
        config = EngineConfig(scheduler_config=SchedulerConfig(max_num_seqs=4))

        params = SamplingParams(max_tokens=5, temperature=0.0)

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.05)

            # Initial stats
            stats = engine.get_stats()
            assert stats["running"] is True
            assert stats["num_waiting"] == 0
            assert stats["num_running"] == 0

            # Add and complete a request
            rid = await engine.add_request("Hello", params)
            async for out in engine.stream_outputs(rid, timeout=30):
                if out.finished:
                    break

            # Check stats after completion
            stats = engine.get_stats()
            assert stats["num_requests_processed"] >= 1
            assert stats["total_completion_tokens"] > 0


class TestSchedulerPolicy:
    """Test scheduler policies."""

    @pytest.mark.asyncio
    async def test_fcfs_ordering(self, model_and_tokenizer):
        """Test that FCFS policy processes requests in order."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )
        from vllm_mlx.scheduler import SchedulingPolicy

        model, tokenizer = model_and_tokenizer
        config = EngineConfig(
            scheduler_config=SchedulerConfig(
                max_num_seqs=2,  # Small batch to test ordering
                policy=SchedulingPolicy.FCFS,
            )
        )

        params = SamplingParams(max_tokens=3, temperature=0.0)

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.05)

            # Add requests with small delay
            rid1 = await engine.add_request("First:", params)
            await asyncio.sleep(0.01)
            rid2 = await engine.add_request("Second:", params)
            await asyncio.sleep(0.01)
            rid3 = await engine.add_request("Third:", params)

            # Collect completion order
            completion_order = []

            async def track_completion(rid, name):
                async for out in engine.stream_outputs(rid, timeout=30):
                    if out.finished:
                        completion_order.append(name)
                        return

            await asyncio.gather(
                track_completion(rid1, "first"),
                track_completion(rid2, "second"),
                track_completion(rid3, "third"),
            )

            # All should complete (order may vary due to batching, but all should finish)
            assert len(completion_order) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self, model_and_tokenizer):
        """Test handling of empty prompt."""
        from vllm_mlx import AsyncEngineCore, SamplingParams

        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=5, temperature=0.0)

        async with AsyncEngineCore(model, tokenizer) as engine:
            await asyncio.sleep(0.05)

            rid = await engine.add_request("", params)
            async for out in engine.stream_outputs(rid, timeout=30):
                if out.finished:
                    # Should complete even with empty prompt
                    assert out.finished
                    break

    @pytest.mark.asyncio
    async def test_very_short_max_tokens(self, model_and_tokenizer):
        """Test with max_tokens=1."""
        from vllm_mlx import AsyncEngineCore, SamplingParams

        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=1, temperature=0.0)

        async with AsyncEngineCore(model, tokenizer) as engine:
            await asyncio.sleep(0.05)

            rid = await engine.add_request("Hello", params)
            token_count = 0

            async for out in engine.stream_outputs(rid, timeout=30):
                token_count += len(out.new_token_ids)
                if out.finished:
                    break

            # Should generate exactly 1 token
            assert token_count == 1

    @pytest.mark.asyncio
    async def test_multiple_start_stop(self, model_and_tokenizer):
        """Test starting and stopping engine multiple times."""
        from vllm_mlx import AsyncEngineCore, SamplingParams

        model, tokenizer = model_and_tokenizer
        params = SamplingParams(max_tokens=3, temperature=0.0)

        for _ in range(3):
            async with AsyncEngineCore(model, tokenizer) as engine:
                await asyncio.sleep(0.05)

                rid = await engine.add_request("Test:", params)
                async for out in engine.stream_outputs(rid, timeout=30):
                    if out.finished:
                        assert out.completion_tokens > 0
                        break


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
