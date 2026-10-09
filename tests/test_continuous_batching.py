# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching performance.

These tests verify that continuous batching properly handles
multiple concurrent requests with improved throughput.
"""

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


import asyncio
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import Scheduler, SchedulerConfig, _common_prefix_len


class TestContinuousBatchingBasic:
    """Basic tests for continuous batching functionality."""

    def test_scheduler_accepts_multiple_requests(self):
        """Test that scheduler can queue multiple requests."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))

        config = SchedulerConfig(max_num_seqs=32)
        scheduler = Scheduler(model, tokenizer, config)

        # Add multiple requests
        for i in range(5):
            request = Request(
                request_id=f"req-{i}",
                prompt=f"Test prompt {i}",
                sampling_params=SamplingParams(max_tokens=50),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 5
        assert scheduler.has_requests()

    def test_scheduler_config_batching_params(self):
        """Test scheduler config has batching parameters."""
        config = SchedulerConfig(
            max_num_seqs=64,
            prefill_batch_size=16,
            completion_batch_size=32,
        )

        assert config.max_num_seqs == 64
        assert config.prefill_batch_size == 16
        assert config.completion_batch_size == 32


@pytest.mark.asyncio
class TestContinuousBatchingIntegration:
    """Integration tests requiring actual model loading."""

    @pytest.fixture
    def small_model(self):
        """Load a small model for testing."""
        try:
            from mlx_lm import load

            model, tokenizer = load("mlx-community/Qwen3-0.6B-8bit")
            return model, tokenizer
        except Exception:
            pytest.skip("Model not available for testing")

    async def test_single_request(self, small_model):
        """Test single request processing."""
        from rapid_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = small_model
        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            params = SamplingParams(max_tokens=10, temperature=0.0)

            rid = await engine.add_request(prompt, params)

            output = None
            async for out in engine.stream_outputs(rid, timeout=30):
                if out.finished:
                    output = out
                    break

            assert output is not None
            assert output.finished
            assert output.completion_tokens > 0

    async def test_concurrent_requests(self, small_model):
        """Test multiple concurrent requests are batched."""
        from rapid_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = small_model
        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                max_num_seqs=32,
                prefill_batch_size=8,
                completion_batch_size=16,
            ),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            prompts = ["What is 2+2?", "Name a color.", "Hello!"]
            params = SamplingParams(max_tokens=20, temperature=0.0)

            # Send all requests
            request_ids = []
            for p in prompts:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rid = await engine.add_request(formatted, params)
                request_ids.append(rid)

            # Collect results
            async def get_result(rid):
                async for out in engine.stream_outputs(rid, timeout=30):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_result(r) for r in request_ids])

            # All should complete
            assert all(r is not None for r in results)
            assert all(r.finished for r in results)
            assert all(r.completion_tokens > 0 for r in results)

    async def test_batching_improves_throughput(self, small_model):
        """Batched throughput must clearly beat sequential.

        Relative comparison rather than an absolute tok/s threshold —
        absolute numbers vary 6× run-to-run on the same machine
        (382-697 tok/s observed) due to GPU contention from the rest
        of the suite, so a hardcoded floor is inherently flaky. Same
        prompts, same model, same engine: the only thing that changes
        between the two phases is concurrent vs serial dispatch, so
        the speedup directly measures batching's benefit.
        """
        from rapid_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = small_model
        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                max_num_seqs=32,
                prefill_batch_size=8,
                completion_batch_size=16,
            ),
        )

        prompts = [
            "What is 2+2?",
            "Name 3 colors.",
            "What is Python?",
            "Capital of Japan?",
            "Who wrote Hamlet?",
        ]
        params = SamplingParams(max_tokens=30, temperature=0.0)

        def _format(p: str) -> str:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            async def _await_one(rid):
                async for out in engine.stream_outputs(rid, timeout=60):
                    if out.finished:
                        return out.completion_tokens
                return 0

            # Warm both execution shapes. A single-request warmup compiles
            # only the batch-size-1 Metal kernels; measuring the first real
            # multi-request batch then charges its shape-specific JIT cost
            # exclusively to the batched phase and can invert the result.
            warm_rid = await engine.add_request(_format(prompts[0]), params)
            await _await_one(warm_rid)
            warm_batch_ids = [
                await engine.add_request(_format(p), params) for p in prompts
            ]
            warm_batch_results = await asyncio.gather(
                *[_await_one(rid) for rid in warm_batch_ids]
            )
            assert all(t > 0 for t in warm_batch_results), warm_batch_results

            # Sequential — one at a time, await each before sending the next
            seq_start = time.perf_counter()
            seq_results: list[int] = []
            for p in prompts:
                rid = await engine.add_request(_format(p), params)
                seq_results.append(await _await_one(rid))
            seq_time = time.perf_counter() - seq_start
            seq_total = sum(seq_results)
            seq_throughput = seq_total / seq_time

            # Batch — submit all then await concurrently
            batch_start = time.perf_counter()
            request_ids = [
                await engine.add_request(_format(p), params) for p in prompts
            ]
            batch_results = await asyncio.gather(*[_await_one(r) for r in request_ids])
            batch_time = time.perf_counter() - batch_start
            batch_total = sum(batch_results)
            batch_throughput = batch_total / batch_time

            speedup = batch_throughput / seq_throughput
            print(
                f"\nSequential: {seq_total} tok in {seq_time:.2f}s = "
                f"{seq_throughput:.1f} tok/s"
            )
            print(
                f"Batch:      {batch_total} tok in {batch_time:.2f}s = "
                f"{batch_throughput:.1f} tok/s"
            )
            print(f"Speedup:    {speedup:.2f}x")

            # Sanity: every request must produce some output
            assert all(t > 0 for t in seq_results), seq_results
            assert all(t > 0 for t in batch_results), batch_results
            # Batching must still materially beat sequential. The measured
            # windows are sub-second once caches are warm, so leave headroom
            # for suite-wide GPU contention; a broken serial scheduler stays
            # near 1.0x, while 1.2x remains a meaningful regression guard.
            assert speedup > 1.2, (
                f"batch={batch_throughput:.1f} not >1.2x of "
                f"sequential={seq_throughput:.1f} (speedup={speedup:.2f}x)"
            )


if __name__ == "__main__":
    # Quick standalone test
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Continuous batching benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("RAPID_MLX_TEST_MODEL")
        or os.environ.get("VLLM_MLX_TEST_MODEL")  # pre-rename name, still honored
        or "mlx-community/Qwen3-8B-6bit",
        help="Model to benchmark",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model

    async def run_benchmark():
        from mlx_lm import load

        from rapid_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        print("=" * 60)
        print("Continuous Batching Benchmark")
        print("=" * 60)
        print(f"Model: {MODEL_NAME}")

        print("\nLoading model...")
        model, tokenizer = load(MODEL_NAME)

        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                max_num_seqs=256,
                prefill_batch_size=8,
                completion_batch_size=32,  # 32 gives optimal throughput
            ),
        )

        prompts = [
            "What is 2+2?",
            "Name 3 colors.",
            "What is Python?",
            "Capital of Japan?",
            "Who wrote Hamlet?",
        ]
        params = SamplingParams(max_tokens=50, temperature=0.7)

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            print(f"\nSending {len(prompts)} concurrent requests...")
            start = time.perf_counter()

            # Use generate() for optimal throughput (no streaming overhead)
            async def run_one(prompt):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                result = await engine.engine.generate(formatted, params)
                return (
                    prompt,
                    result.output_text[:50],
                    result.prompt_tokens,
                    result.completion_tokens,
                )

            results = await asyncio.gather(*[run_one(p) for p in prompts])

            total_time = time.perf_counter() - start
            prompt_tokens = sum(r[2] for r in results)
            completion_tokens = sum(r[3] for r in results)
            total_tokens = prompt_tokens + completion_tokens

            print("\n" + "-" * 60)
            print("Results:")
            for prompt, output, _, tokens in results:
                clean_output = output.replace("\n", " ")[:40]
                print(f"  [{tokens:3d} tok] {prompt[:20]:20s} -> {clean_output}...")

            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total time:    {total_time:.2f}s")
            print(f"Requests:      {len(prompts)}")
            print(f"Total tokens:  {total_tokens}")
            print(f"Throughput:    {total_tokens / total_time:.1f} tok/s")
            print(f"Requests/sec:  {len(prompts) / total_time:.2f}")
            print("=" * 60)

    asyncio.run(run_benchmark())


# --- A request waits for a running prefill of the same prompt prefix -------


_WAIT_DOC = list(range(1000, 9000))


def _wait_request(name, tokens, *, emitted=0):
    request = Request(
        request_id=name, prompt="", sampling_params=SamplingParams(max_tokens=8)
    )
    request.prompt_token_ids = list(tokens)
    request.remaining_tokens = list(tokens)
    request.output_token_ids = [0] * emitted
    return request


def _wait_scheduler(*running, waiting=(), wait_tokens=1024):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(shared_prefix_wait_tokens=wait_tokens)
    scheduler.running = {r.request_id: r for r in running}
    scheduler.waiting = deque(waiting)
    scheduler._shared_prefix_waits = 0
    scheduler._reclaim_prefix_cache_for_prefill = lambda request: 0
    scheduler.stored = {}

    def fetch(tokens):
        for key, cache in scheduler.stored.items():
            if list(tokens[: len(key)]) == list(key):
                scheduler.memory_aware_cache._last_match_type = "prefix"
                return cache, list(tokens[len(key) :])
        scheduler.memory_aware_cache._last_match_type = "miss"
        return None, list(tokens)

    scheduler.memory_aware_cache = SimpleNamespace(
        fetch=fetch, _last_match_type="miss", _entries={}
    )
    return scheduler


def test_common_prefix_len():
    assert _common_prefix_len([], [1]) == 0
    assert _common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert _common_prefix_len(_WAIT_DOC, _WAIT_DOC + [7]) == len(_WAIT_DOC)
    assert _common_prefix_len([5] + _WAIT_DOC, _WAIT_DOC) == 0


def test_request_waits_for_a_prefill_of_the_same_document():
    leader = _wait_request("leader", _WAIT_DOC + [1, 2])
    follower = _wait_request("follower", _WAIT_DOC + [3, 4, 5])
    other = _wait_request("other", list(range(50)))
    scheduler = _wait_scheduler(leader, waiting=[follower, other])

    # The held request does not block an unrelated one behind it.
    assert scheduler._pop_waiting_for_admission() is other
    assert scheduler._pop_waiting_for_admission() is None
    assert list(scheduler.waiting) == [follower]
    assert scheduler._shared_prefix_waits == 1

    # The leader stored its prompt state and produced its first token.
    scheduler.stored[tuple(_WAIT_DOC)] = ["state"]
    leader.output_token_ids = [0]
    assert scheduler._pop_waiting_for_admission() is follower
    assert follower.prompt_cache == ["state"]
    assert follower.cached_tokens == len(_WAIT_DOC)
    assert follower.remaining_tokens == [3, 4, 5]
    assert scheduler._shared_prefix_waits == 1


def test_request_is_released_when_the_leader_leaves_without_storing():
    leader = _wait_request("leader", _WAIT_DOC + [1])
    follower = _wait_request("follower", _WAIT_DOC + [2])
    scheduler = _wait_scheduler(leader, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is None

    del scheduler.running["leader"]  # aborted mid-prefill
    assert scheduler._pop_waiting_for_admission() is follower
    assert follower.prompt_cache is None
    assert follower.cached_tokens == 0
    assert follower.remaining_tokens == _WAIT_DOC + [2]


def test_request_follows_only_the_request_it_was_held_for():
    first = _wait_request("first", _WAIT_DOC + [1])
    second = _wait_request("second", _WAIT_DOC + [3])
    follower = _wait_request("follower", _WAIT_DOC + [2])
    scheduler = _wait_scheduler(first, second, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is None
    assert follower.prefix_wait_leader == "first"

    # Its leader is done; another prefill of the document is still running.
    first.output_token_ids = [0]
    assert scheduler._pop_waiting_for_admission() is follower
    assert follower.prefix_wait_leader is None


def test_request_waits_only_once():
    first = _wait_request("first", _WAIT_DOC + [1])
    follower = _wait_request("follower", _WAIT_DOC + [2])
    scheduler = _wait_scheduler(first, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is None
    del scheduler.running["first"]
    assert scheduler._pop_waiting_for_admission() is follower

    # Requeued (generator not ready) beside another prefill of the document.
    scheduler.running["second"] = _wait_request("second", _WAIT_DOC + [3])
    scheduler.waiting.appendleft(follower)
    assert scheduler._pop_waiting_for_admission() is follower


@pytest.mark.parametrize(
    ("leader_tokens", "follower_tokens"),
    [
        # Too little shared to be worth a wait.
        (_WAIT_DOC[:1000] + [1] * 3000, _WAIT_DOC[:1000] + [2] * 3000),
        # The leader still has far more prompt to process than is shared.
        (_WAIT_DOC[:2000] + [1] * 30_000, _WAIT_DOC[:2000] + [2] * 100),
    ],
)
def test_request_is_not_held_when_waiting_would_not_pay(leader_tokens, follower_tokens):
    leader = _wait_request("leader", leader_tokens)
    follower = _wait_request("follower", follower_tokens)
    scheduler = _wait_scheduler(leader, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is follower
    assert scheduler._shared_prefix_waits == 0


def test_request_is_not_held_behind_a_decoding_request_or_its_own_cached_prefix():
    decoding = _wait_request("decoding", _WAIT_DOC + [1], emitted=5)
    follower = _wait_request("follower", _WAIT_DOC + [2])
    scheduler = _wait_scheduler(decoding, waiting=[follower])
    assert scheduler._pop_waiting_for_admission() is follower

    leader = _wait_request("leader", _WAIT_DOC + [1])
    cached = _wait_request("cached", _WAIT_DOC + [2])
    cached.remaining_tokens = [2]  # the shared part is already cached
    scheduler = _wait_scheduler(leader, waiting=[cached])
    assert scheduler._pop_waiting_for_admission() is cached


def test_zero_disables_the_wait_and_negative_is_rejected():
    leader = _wait_request("leader", _WAIT_DOC + [1])
    follower = _wait_request("follower", _WAIT_DOC + [2])
    scheduler = _wait_scheduler(leader, waiting=[follower], wait_tokens=0)
    assert scheduler._pop_waiting_for_admission() is follower
    with pytest.raises(ValueError, match="shared_prefix_wait_tokens"):
        SchedulerConfig(shared_prefix_wait_tokens=-1)
    with pytest.raises(ValueError, match="shared_prefix_wait_tokens"):
        SchedulerConfig(shared_prefix_wait_tokens=True)


def test_admission_leaves_a_held_request_in_the_queue():
    leader = _wait_request("leader", _WAIT_DOC + [1])
    follower = _wait_request("follower", _WAIT_DOC + [2])
    scheduler = _wait_scheduler(leader, waiting=[follower])
    scheduler._max_running_sequences = lambda: 8
    assert scheduler._schedule_waiting() == []
    assert list(scheduler.waiting) == [follower]
