# SPDX-License-Identifier: Apache-2.0
"""Scheduler-level integration tests for PFlash (#287).

Verifies the PFlash hook in ``Scheduler.add_request`` and the
prefix-cache namespacing fix. The fork only suppressed the
``prefix_boundary`` boundary save; this adaptation gates every cache
fetch and every cache store, so the tests here cover both the fetch
bypass and the store bypass.

Ported scenarios from @michaelasper's fork
(``tests/test_pflash_scheduler.py`` at commit b6089ce). Extended with a
namespacing regression test that fakes a populated ``MemoryAwarePrefix
Cache`` to confirm the compressed-request path never calls into it.
"""

from unittest.mock import MagicMock

from vllm_mlx.pflash import PFlashConfig
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class DummyTokenizer:
    """Minimal tokenizer for scheduler tests — accepts prompt lists
    directly and treats encode() as identity for char-to-int."""

    eos_token_id = None

    def encode(self, prompt):
        if isinstance(prompt, str):
            return [ord(char) for char in prompt]
        return list(prompt)

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def _make_scheduler(
    pflash_config: PFlashConfig,
    *,
    enable_prefix_cache: bool = False,
) -> Scheduler:
    return Scheduler(
        model=object(),
        tokenizer=DummyTokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=enable_prefix_cache,
            use_memory_aware_cache=False,
            pflash_config=pflash_config,
        ),
    )


def _compressing_config() -> PFlashConfig:
    return PFlashConfig(
        mode="always",
        threshold=1,
        keep_ratio=0.25,
        min_keep_tokens=16,
        sink_tokens=4,
        tail_tokens=4,
        block_size=4,
    )


class TestPFlashSchedulerHook:
    def test_engages_above_threshold_without_tools(self):
        scheduler = _make_scheduler(_compressing_config())
        request = Request(
            "req-engage",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)

        assert request.pflash_metadata is not None
        assert request.pflash_metadata["compressed"] is True
        # Logical prompt length is preserved for client-facing usage
        # accounting; the model-level length is the compressed count.
        assert request.num_prompt_tokens == 128
        assert request.model_prompt_tokens == len(request.prompt_token_ids)
        assert request.model_prompt_tokens < request.num_prompt_tokens
        assert request.original_prompt_token_ids == list(range(128))

    def test_skipped_when_request_has_tools(self):
        scheduler = _make_scheduler(_compressing_config())
        request = Request(
            "req-tools",
            list(range(128)),
            SamplingParams(max_tokens=4),
            has_tools=True,
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        assert request.pflash_metadata["reason"] == "tools"
        assert request.model_prompt_tokens == request.num_prompt_tokens == 128
        assert request.prompt_token_ids == list(range(128))

    def test_skipped_when_requires_prompt_integrity(self):
        scheduler = _make_scheduler(_compressing_config())
        request = Request(
            "req-protected",
            list(range(128)),
            SamplingParams(max_tokens=4),
            requires_prompt_integrity=True,
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        assert request.pflash_metadata["reason"] == "protected_prompt"
        assert request.prompt_token_ids == list(range(128))

    def test_skipped_in_auto_mode_below_threshold(self):
        config = PFlashConfig(mode="auto", threshold=1_000, keep_ratio=0.10)
        scheduler = _make_scheduler(config)
        request = Request(
            "req-short",
            list(range(64)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        assert request.pflash_metadata["reason"] == "threshold"

    def test_pflash_disabled_leaves_request_untouched(self):
        scheduler = _make_scheduler(PFlashConfig(mode="off"))
        request = Request(
            "req-off",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)

        assert request.pflash_metadata is None
        assert request.prompt_token_ids == list(range(128))
        assert request.original_prompt_token_ids is None

    def test_prefix_boundary_forced_to_zero_on_compression(self):
        scheduler = _make_scheduler(_compressing_config())
        request = Request(
            "req-boundary",
            list(range(128)),
            SamplingParams(max_tokens=4),
            prefix_boundary=32,
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is True
        assert request.pflash_metadata["prefix_boundary_disabled"] is True
        assert request.prefix_boundary == 0

    def test_metadata_records_scoring_time(self):
        scheduler = _make_scheduler(_compressing_config())
        request = Request(
            "req-scoring",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )
        scheduler.add_request(request)
        assert request.pflash_metadata is not None
        assert request.pflash_metadata["scoring_seconds"] >= 0.0


class TestPrefixCacheNamespacing:
    """The fork's #1 correctness blocker: compressed token sequences are
    positionally non-faithful, so they must never enter the prefix-cache
    trie. Approach A (chosen): every cache fetch/store is gated behind
    ``_pflash_compressed(request)``.
    """

    def test_compressed_request_does_not_fetch_from_prefix_cache(self):
        scheduler = _make_scheduler(_compressing_config())
        # Inject a fake legacy prefix cache so we can assert it is
        # never consulted for a compressed request. We bypass the
        # constructor's auto-disable because enable_prefix_cache=False
        # and instead attach the mock directly.
        fake_cache = MagicMock()
        fake_cache.fetch_cache.return_value = (None, [])
        scheduler.prefix_cache = fake_cache

        request = Request(
            "req-fetch-bypass",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is True
        assert request.cache_hit_type == "miss"
        # The prefix cache must not have been touched by the compressed
        # request — that is the entire point of the namespacing fix.
        fake_cache.fetch_cache.assert_not_called()

    def test_compressed_request_does_not_fetch_from_memory_aware_cache(self):
        scheduler = _make_scheduler(_compressing_config())
        fake_cache = MagicMock()
        fake_cache.fetch.return_value = (None, [])
        fake_cache._last_match_type = "miss"
        fake_cache._entries = {}
        scheduler.memory_aware_cache = fake_cache

        request = Request(
            "req-mem-bypass",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is True
        fake_cache.fetch.assert_not_called()

    def test_uncompressed_request_still_fetches_prefix_cache(self):
        # Regression guard: PFlash compresses long prompts; SHORT
        # prompts in auto mode must still hit the cache so we do not
        # silently disable prefix caching across the whole server.
        scheduler = _make_scheduler(
            PFlashConfig(mode="auto", threshold=10_000, keep_ratio=0.10)
        )
        fake_cache = MagicMock()
        fake_cache.fetch_cache.return_value = (None, [])
        scheduler.prefix_cache = fake_cache

        request = Request(
            "req-short-prompt",
            list(range(64)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)

        assert request.pflash_metadata["compressed"] is False
        fake_cache.fetch_cache.assert_called_once()

    def test_no_pflash_path_unchanged(self):
        # When PFlash is off entirely the cache fetch must still happen.
        scheduler = _make_scheduler(PFlashConfig(mode="off"))
        fake_cache = MagicMock()
        fake_cache.fetch_cache.return_value = (None, [])
        scheduler.prefix_cache = fake_cache

        request = Request(
            "req-no-pflash",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )

        scheduler.add_request(request)
        fake_cache.fetch_cache.assert_called_once()

    def test_cleanup_finished_skips_store_for_compressed_request(self):
        scheduler = _make_scheduler(_compressing_config())
        fake_cache = MagicMock()
        scheduler.prefix_cache = fake_cache

        request = Request(
            "req-cleanup",
            list(range(128)),
            SamplingParams(max_tokens=4),
        )
        scheduler.add_request(request)
        # Simulate a finished compressed request with an extracted cache;
        # _cleanup_finished must not store it.
        request._extracted_cache = ["fake_kv"]
        scheduler.running[request.request_id] = request

        scheduler._cleanup_finished({request.request_id})

        fake_cache.store_cache.assert_not_called()

    def test_cleanup_finished_stores_for_uncompressed_request(self):
        # Inverse regression: PFlash-off requests must still store.
        scheduler = _make_scheduler(PFlashConfig(mode="off"))
        fake_cache = MagicMock()
        # fetch_cache must return (cache_or_None, remaining_tokens).
        fake_cache.fetch_cache.return_value = (None, [])
        scheduler.prefix_cache = fake_cache

        request = Request(
            "req-cleanup-store",
            list(range(64)),
            SamplingParams(max_tokens=4),
        )
        scheduler.add_request(request)
        request._extracted_cache = ["fake_kv"]
        scheduler.running[request.request_id] = request

        scheduler._cleanup_finished({request.request_id})

        fake_cache.store_cache.assert_called_once()
