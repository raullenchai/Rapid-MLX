"""MLLM-lane sampling parity tests (seed / top_k / min_p).

The media lane historically dropped the extended sampling keys: the OpenAI
route layer forwarded them, but ``BatchedEngine``'s MLLM branches never
passed them to ``MLLMScheduler`` and the batch generator built its samplers
from ``(temperature, top_p)`` alone. A media request with a pinned ``seed``
silently sampled from the global RNG while the identical text request
honored it.

These tests pin the full chain:

* ``MLLMScheduler.add_request`` stores the three keys on ``SamplingParams``
  (with ``None``-folds-to-disabled, but ``seed=0`` preserved);
* ``_schedule_waiting`` copies them onto ``MLLMBatchRequest``;
* the batch generator's sampler construction honors the full
  ``(temperature, top_p, min_p, top_k)`` fingerprint, keeps the shared
  batched-sampler fast path for homogeneous unseeded batches, and gives
  every seeded request its own private ``make_seeded_sampler`` closure
  that is reused across steps so its RNG stream stays continuous.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
)
from rapid_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures (mirroring test_mllm_batch_generator.py's stubs)
# ---------------------------------------------------------------------------


def _make_step_stub_generator(vocab: int = 32):
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._shared_batch_sampler = None

    def _language_model(input_tokens, cache=None):
        B = input_tokens.shape[0]
        return mx.zeros((B, 1, vocab))

    gen.language_model = _language_model
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)
    return gen


def _make_request(uid: int = 0, **overrides) -> MLLMBatchRequest:
    fields = {
        "uid": uid,
        "request_id": f"r{uid}",
        "prompt": "hi",
        "max_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    fields.update(overrides)
    return MLLMBatchRequest(**fields)


def _stub_scheduler() -> MLLMScheduler:
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()
    return MLLMScheduler(mock_model, mock_processor, MLLMSchedulerConfig())


class _CountingSeededSampler:
    """Stand-in for ``make_seeded_sampler`` closures with call accounting."""

    def __init__(self, constructions: list):
        self._constructions = constructions
        self.calls = 0
        constructions.append(self)

    def __call__(self, logprobs):
        self.calls += 1
        return mx.zeros((logprobs.shape[0],), dtype=mx.uint32)


# ---------------------------------------------------------------------------
# Scheduler plumbing
# ---------------------------------------------------------------------------


class TestSchedulerParamForwarding:
    def test_add_request_stores_extended_params(self):
        scheduler = _stub_scheduler()
        req_id = scheduler.add_request(prompt="test", top_k=17, min_p=0.08, seed=42)
        params = scheduler.requests[req_id].sampling_params
        assert params.top_k == 17
        assert params.min_p == 0.08
        assert params.seed == 42

    def test_add_request_defaults_are_disabled(self):
        scheduler = _stub_scheduler()
        req_id = scheduler.add_request(prompt="test")
        params = scheduler.requests[req_id].sampling_params
        assert params.top_k == 0
        assert params.min_p == 0.0
        assert params.seed is None

    def test_add_request_preserves_seed_zero(self):
        """``seed=0`` is a legitimate PRNG seed and must not collapse to
        the disabled ``None`` sentinel."""
        scheduler = _stub_scheduler()
        req_id = scheduler.add_request(prompt="test", seed=0)
        assert scheduler.requests[req_id].sampling_params.seed == 0

    def test_schedule_waiting_copies_extended_params(self):
        scheduler = _stub_scheduler()
        scheduler._ensure_batch_generator()

        captured: dict = {}
        original_insert = scheduler.batch_generator.insert

        def _capture_insert(requests):
            captured["requests"] = list(requests)
            return original_insert(requests)

        scheduler.batch_generator.insert = _capture_insert

        scheduler.add_request(prompt="test", top_k=5, min_p=0.1, seed=7)
        scheduler._schedule_waiting()

        assert len(captured["requests"]) == 1
        batch_req = captured["requests"][0]
        assert batch_req.top_k == 5
        assert batch_req.min_p == 0.1
        assert batch_req.seed == 7


# ---------------------------------------------------------------------------
# Batch-generator sampler selection
# ---------------------------------------------------------------------------


class TestBatchSamplerSelection:
    def test_homogeneous_full_fingerprint_uses_shared_fast_path(self, monkeypatch):
        """Same (temp, top_p, min_p, top_k), unseeded → one shared sampler
        built with the complete fingerprint."""
        make_sampler_calls = []

        def fake_make_sampler(**kwargs):
            make_sampler_calls.append(kwargs)
            return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_sampler", fake_make_sampler
        )
        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_seeded_sampler",
            lambda **_kw: pytest.fail("unseeded batch must not build seeded samplers"),
        )

        gen = _make_step_stub_generator()
        requests = [_make_request(i, min_p=0.05, top_k=20) for i in range(3)]
        sampled, _ = MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2], [3]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )

        assert len(make_sampler_calls) == 1
        assert make_sampler_calls[0] == {
            "temp": 0.7,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 20,
        }
        assert gen._shared_batch_sampler is not None
        assert sampled.shape == (3,)

    def test_mixed_fingerprint_falls_back_to_per_row(self, monkeypatch):
        """One row with a different min_p breaks homogeneity → per-row loop."""
        make_sampler_calls = []

        def fake_make_sampler(**kwargs):
            make_sampler_calls.append(kwargs)
            return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_sampler", fake_make_sampler
        )

        gen = _make_step_stub_generator()
        requests = [
            _make_request(0, min_p=0.05),
            _make_request(1, min_p=0.10),
        ]
        MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )

        assert make_sampler_calls == [
            {"temp": 0.7, "top_p": 0.9, "min_p": 0.05},
            {"temp": 0.7, "top_p": 0.9, "min_p": 0.10},
        ]
        assert gen._shared_batch_sampler is None

    def test_out_of_range_top_k_is_clamped_not_fatal(self):
        """The API layer admits ``top_k`` far above any vocabulary and its
        docs promise the backend clamps ``min(top_k, vocab)`` — the text
        lane honours that. mlx-lm's ``apply_top_k`` RAISES for
        ``top_k >= vocab_size``, so the media lane must normalise before
        building the chain or an API-valid request would fail the whole
        batch. Drives the REAL mlx-lm ``make_sampler`` chain (no stub) so
        the raise would surface here."""
        gen = _make_step_stub_generator(vocab=8)
        requests = [
            _make_request(0, top_k=10**9),
            _make_request(1, top_k=8),
            _make_request(2, top_k=10**9, seed=7),
        ]
        # No exception: both rows sample through the clamped chain.
        sampled, _ = MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2], [3]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )
        assert sampled.shape == (3,)

        # Equivalence at the normalisation layer: ``top_k >= vocab`` is
        # semantically "keep every token" — the disabled value.
        from rapid_mlx.mllm_batch_generator import _effective_top_k

        assert _effective_top_k(10**9, 8) == 0
        assert _effective_top_k(8, 8) == 0
        assert _effective_top_k(7, 8) == 7
        assert _effective_top_k(5, None) == 5

    def test_out_of_range_top_k_homogeneous_fast_path_also_clamps(self, monkeypatch):
        """The homogeneous shared-sampler build applies the same vocab
        clamp, so a batch of API-valid large-``top_k`` rows takes the fast
        path instead of raising inside ``apply_top_k``."""
        make_sampler_calls = []

        def fake_make_sampler(**kwargs):
            make_sampler_calls.append(kwargs)
            return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_sampler", fake_make_sampler
        )
        gen = _make_step_stub_generator(vocab=8)
        requests = [_make_request(i, top_k=10**9) for i in range(3)]
        MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2], [3]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )
        assert make_sampler_calls == [{"temp": 0.7, "top_p": 0.9}]
        assert gen._shared_batch_sampler is not None

    def test_seeded_request_gets_private_samplers_per_row(self, monkeypatch):
        """A seeded request never enters the shared fast path, even when
        every row carries the same seed and fingerprint. Each row receives
        its own closure — same seed does NOT mean a shared RNG stream."""
        constructions: list = []
        seeded_sampler_calls: list[dict] = []
        make_sampler_calls = []

        def fake_make_seeded(**kwargs):
            seeded_sampler_calls.append(kwargs)
            return _CountingSeededSampler(constructions)

        def fake_make_sampler(**kwargs):
            make_sampler_calls.append(kwargs)
            return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_seeded_sampler", fake_make_seeded
        )
        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_sampler", fake_make_sampler
        )

        gen = _make_step_stub_generator()
        requests = [
            _make_request(0, seed=42, temperature=0.5, top_p=0.8, min_p=0.1, top_k=7),
            _make_request(1, seed=42, temperature=0.5, top_p=0.8, min_p=0.1, top_k=7),
        ]
        MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )

        # Two distinct closures, one per request; no unseeded construction.
        assert len(constructions) == 2
        assert constructions[0] is not constructions[1]
        assert seeded_sampler_calls == [
            {"seed": 42, "temperature": 0.5, "top_p": 0.8, "min_p": 0.1, "top_k": 7},
            {"seed": 42, "temperature": 0.5, "top_p": 0.8, "min_p": 0.1, "top_k": 7},
        ]
        assert make_sampler_calls == []
        # The shared interning slot must stay empty for seeded batches.
        assert gen._shared_batch_sampler is None

    def test_seeded_sampler_reused_across_decode_steps(self, monkeypatch):
        """One seeded request → exactly one seeded-sampler construction,
        reused every step so the carried PRNG key advances continuously
        instead of restarting from the initial seed."""
        constructions: list = []

        def fake_make_seeded(**kwargs):
            return _CountingSeededSampler(constructions)

        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_seeded_sampler", fake_make_seeded
        )

        gen = _make_step_stub_generator()
        request = _make_request(0, seed=42)
        for _ in range(4):
            MLLMBatchGenerator._step(
                gen,
                mx.array([[1]], dtype=mx.uint32),
                cache=[],
                requests=[request],
            )

        assert len(constructions) == 1
        assert constructions[0].calls == 4

    def test_prefill_token_zero_shares_decode_rng_stream(self, monkeypatch):
        """The first token sampled during prefill must come from the same
        seeded closure later used by decode — rebuilding a fresh seeded
        sampler at decode start would replay the initial PRNG subkey on a
        different draw."""
        from mlx_lm.models import cache as cache_module

        class _Cache:
            def merge(self, _caches):
                return self

        monkeypatch.setattr(
            cache_module, "make_prompt_cache", lambda _model: [_Cache()]
        )
        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.first_incompatible_mllm_cache_type",
            lambda *_args, **_kwargs: None,
        )

        constructions: list = []

        def fake_make_seeded(**kwargs):
            return _CountingSeededSampler(constructions)

        monkeypatch.setattr(
            "rapid_mlx.mllm_batch_generator.make_seeded_sampler", fake_make_seeded
        )

        class _Model:
            def __call__(self, input_ids, cache=None, **kwargs):
                return mx.zeros((1, 1, 4))

            # ``_step`` dispatches through ``language_model``; decode must
            # produce logits through the same stub.
            language_model = lambda _self, input_tokens, cache=None: mx.zeros(
                (input_tokens.shape[0], 1, 4)
            )  # noqa: E731

        gen = MLLMBatchGenerator(
            model=_Model(),
            processor=object(),
            mm_processor=None,
            enable_vision_cache=False,
        )
        gen._preprocess_request = lambda _request: None
        gen._run_vision_encoding = lambda _request, cache: mx.zeros((1, 1, 4))

        request = MLLMBatchRequest(
            uid=0,
            request_id="r0",
            prompt="hello",
            max_tokens=8,
            seed=42,
        )

        batch = gen._process_prompts([request])
        assert int(batch.y.item()) == 0  # stub sampler returns zeros
        assert len(constructions) == 1

        # Decode reuses the prefill-built closure — no second construction.
        sampled, _ = MLLMBatchGenerator._step(
            gen,
            mx.array([[0]], dtype=mx.uint32),
            cache=[],
            requests=[request],
        )
        assert sampled.shape == (1,)
        assert len(constructions) == 1
        assert constructions[0].calls == 2

    def test_seeded_greedy_request_is_deterministic(self):
        """``temperature=0`` with a seed must return the argmax token on
        every draw (greedy short-circuit inside ``make_seeded_sampler``)."""
        from rapid_mlx._seeded_sampler import make_seeded_sampler

        sampler = make_seeded_sampler(seed=1234, temperature=0.0)
        logprobs = mx.log(mx.array([[0.1, 0.6, 0.3]]))
        for _ in range(3):
            assert int(sampler(logprobs).item()) == 1

    def test_seeded_sampled_path_is_repeatable(self):
        """Same closure construction parameters + same logits sequence →
        the same token stream (within-engine determinism contract)."""

        def run():
            from rapid_mlx._seeded_sampler import make_seeded_sampler

            sampler = make_seeded_sampler(seed=42, temperature=0.7, top_p=0.9, top_k=3)
            logprobs = mx.log(mx.array([[0.4, 0.3, 0.2, 0.1]]))
            return [int(sampler(logprobs).item()) for _ in range(16)]

        assert run() == run()

    def test_seeded_interleaved_execution_matches_isolated(self):
        """Two seeded requests interleaved through one batched generator
        produce the same token streams they produce in isolation — the
        request-owned RNG stream must not observe the other request's
        draws or restart from the initial seed.

        Uses the real ``make_seeded_sampler`` (no stub) over a uniform
        64-token logits surface, so RNG cross-contamination or a stream
        restart would change the sampled tokens.
        """
        import mlx.core as mx

        def run_sequence(*requests, steps: int = 12):
            gen = _make_step_stub_generator()
            streams = {req.request_id: [] for req in requests}
            for _ in range(steps):
                sampled, _ = MLLMBatchGenerator._step(
                    gen,
                    mx.array([[req.uid] for req in requests], dtype=mx.uint32),
                    cache=[],
                    requests=list(requests),
                )
                for req, token in zip(
                    requests,
                    [int(sampled.item())] if len(requests) == 1 else sampled.tolist(),
                ):
                    streams[req.request_id].append(token)
            return streams

        seeded_requests = [_make_request(0, seed=7), _make_request(1, seed=99)]

        # Fresh request objects per run: ``_cached_sampler`` carries the
        # RNG stream across steps, so reusing an object would continue the
        # stream instead of restarting it (that continuity is pinned by
        # test_seeded_sampler_reused_across_decode_steps).
        isolated = run_sequence(_make_request(0, seed=7))["r0"]
        interleaved = run_sequence(_make_request(0, seed=7), _make_request(1, seed=99))[
            "r0"
        ]

        assert isolated == interleaved
        # Sanity: uniform logits + real RNG must not be degenerate, or the
        # interleaving comparison would be vacuous.
        assert len(set(isolated)) > 1

        # Round-2 review: the SECOND row's isolation is part of the contract
        # too — contamination that only corrupts later rows' draw order must
        # not slip past an r0-only assertion.
        isolated_r1 = run_sequence(_make_request(1, seed=99))["r1"]
        interleaved_r1 = run_sequence(
            _make_request(0, seed=7), _make_request(1, seed=99)
        )["r1"]
        assert isolated_r1 == interleaved_r1
        assert len(set(isolated_r1)) > 1
        # Distinct seeds must own distinct streams (not merely equal ones).
        assert isolated_r1 != isolated
