# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLLMBatchGenerator model-call kwargs.

Some mlx-vlm model classes (notably ``Gemma3ForConditionalGeneration``)
declare ``pixel_values`` as a *required* positional kwarg in ``__call__``,
even though the inner ``get_input_embeddings`` already handles ``None`` for
the text-only path. Omitting the kwarg raises ``TypeError`` for every
text-only request to those models, so ``_run_vision_encoding`` must always
pass it through — including when it's ``None``.
"""

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


class _RecordingModel:
    """VLM model stub that captures kwargs from its ``__call__``."""

    def __init__(self):
        self.last_call_kwargs = None
        self.last_input_ids = None
        # Provide a language_model attribute so the generator's
        # is_vlm branch picks it up without warnings.
        self.language_model = object()

    def __call__(self, input_ids, cache=None, **kwargs):
        self.last_input_ids = input_ids
        self.last_call_kwargs = kwargs
        # Return a dummy logits tensor — generator only inspects shape via
        # ``hasattr(output, "logits")``; the value is irrelevant for this test.
        return mx.zeros((1, 1, 8))


def _make_generator(model: _RecordingModel) -> MLLMBatchGenerator:
    """Construct a generator without booting Metal / vision cache plumbing."""
    return MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )


def _make_request(*, pixel_values, extra_kwargs=None) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="hello",
        max_tokens=8,
        input_ids=mx.array([1, 2, 3], dtype=mx.int32),
        pixel_values=pixel_values,
        extra_kwargs=extra_kwargs or {},
    )


def test_run_vision_encoding_passes_pixel_values_none_for_text_only_request():
    """Text-only request still includes pixel_values=None in kwargs.

    Gemma3ForConditionalGeneration's ``__call__`` declares ``pixel_values``
    as a required kwarg, so we must always forward it — even when None.
    """
    model = _RecordingModel()
    gen = _make_generator(model)
    request = _make_request(pixel_values=None)

    gen._run_vision_encoding(request, cache=None)

    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is None


def test_run_vision_encoding_forwards_pixel_values_when_set():
    """Multimodal request keeps forwarding the real pixel tensor."""
    model = _RecordingModel()
    gen = _make_generator(model)
    pixels = mx.zeros((1, 3, 4, 4))
    request = _make_request(pixel_values=pixels)

    gen._run_vision_encoding(request, cache=None)

    assert "pixel_values" in model.last_call_kwargs
    # Must be the same object we put in — generator should not silently copy
    # or downcast pixel_values before the forward pass.
    assert model.last_call_kwargs["pixel_values"] is pixels


def test_run_vision_encoding_preserves_extra_kwargs_alongside_pixel_values():
    """Extra processor kwargs (e.g. token_type_ids) survive alongside pixel_values."""
    model = _RecordingModel()
    gen = _make_generator(model)
    request = _make_request(
        pixel_values=None,
        extra_kwargs={"token_type_ids": mx.array([0, 0, 1])},
    )

    gen._run_vision_encoding(request, cache=None)

    assert "pixel_values" in model.last_call_kwargs
    assert model.last_call_kwargs["pixel_values"] is None
    assert "token_type_ids" in model.last_call_kwargs


# ---------------------------------------------------------------------------
# Shutdown — mx.synchronize must not propagate cross-thread errors
# ---------------------------------------------------------------------------


def test_close_swallows_synchronize_thread_error(monkeypatch):
    """`close()` must not propagate RuntimeError from mx.synchronize.

    mlx-lm 0.31.3+ streams are thread-local. When the engine is torn down
    from a thread that isn't the one that owns MLLMBatchGenerator._stream,
    mx.synchronize raises `There is no Stream(gpu, N) in current thread`.
    Pre-fix this propagated out of the lifespan shutdown and produced a
    scary traceback (Persona E v0.6.51 onboarding finding). The sync is
    best-effort on shutdown; the wired-limit reset is what matters.
    """
    import mlx.core as mx

    # Construct a generator and force the wired-limit branch to execute.
    gen = _make_generator(_RecordingModel())
    gen._old_wired_limit = 1234  # any sentinel triggers the close path

    sync_calls: list[object] = []
    set_limit_calls: list[int] = []

    def _raising_sync(stream):
        sync_calls.append(stream)
        raise RuntimeError("There is no Stream(gpu, 2) in current thread")

    def _record_set_limit(value):
        set_limit_calls.append(value)
        return value

    monkeypatch.setattr(mx, "synchronize", _raising_sync)
    monkeypatch.setattr(mx, "set_wired_limit", _record_set_limit)

    # Must not raise.
    gen.close()

    # Best-effort sync attempted exactly once.
    assert len(sync_calls) == 1
    # Wired limit was still reset to the original value — the important
    # cleanup is not skipped just because the cross-thread sync failed.
    assert set_limit_calls == [1234]
    # State is cleared so __del__ is a no-op afterward.
    assert gen._old_wired_limit is None


def test_close_propagates_non_runtime_errors_from_set_wired_limit(monkeypatch):
    """Errors from set_wired_limit are unrelated to the thread bug — keep
    propagating them so a real OS-level failure isn't silently swallowed.
    """
    import mlx.core as mx

    gen = _make_generator(_RecordingModel())
    gen._old_wired_limit = 999

    monkeypatch.setattr(mx, "synchronize", lambda _s: None)

    def _boom(value):
        raise OSError("metal API call failed")

    monkeypatch.setattr(mx, "set_wired_limit", _boom)

    import pytest

    with pytest.raises(OSError, match="metal API call failed"):
        gen.close()


# ---------------------------------------------------------------------------
# Batched-sampler fast path
# ---------------------------------------------------------------------------
#
# When every request in the batch shares (temperature, top_p), _step calls
# a single batched sampler on [B, vocab] instead of looping B times over
# per-row slices. The mlx-lm sampler chain vectorizes along axis=-1, so one
# call produces [B] tokens via one MLX kernel chain. Profiling on Gemma 3
# 12B 4bit (M3 Ultra) at B=8 showed step time drops from 73ms to 52ms,
# concurrent HTTP throughput from 95 to 119 tok/s (+26%). Heterogeneous
# sampling params fall back to the legacy per-row loop and keep the
# pre-existing per-request _cached_sampler attribute.


def _make_step_stub_generator():
    """Minimal MLLMBatchGenerator that returns a deterministic 1x1xV logit."""
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._shared_batch_sampler = None

    def _language_model(input_tokens, cache=None):
        B = input_tokens.shape[0]
        # Tiny vocab (4) so logit math is cheap; row r prefers token r%4.
        return mx.zeros((B, 1, 4))

    gen.language_model = _language_model
    gen.sampler = lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)
    return gen


def _make_sampling_request(uid: int, temperature: float, top_p: float):
    return MLLMBatchRequest(
        uid=uid,
        request_id=f"r{uid}",
        prompt="hi",
        max_tokens=8,
        temperature=temperature,
        top_p=top_p,
    )


def test_step_homogeneous_requests_call_shared_sampler_once(monkeypatch):
    """All requests share (temp, top_p) → one batched sampler call on [B, vocab]."""
    make_sampler_calls = []
    shared_sampler_invocations = []

    def shared_sampler(logprobs):
        shared_sampler_invocations.append(logprobs.shape)
        return mx.zeros((logprobs.shape[0],), dtype=mx.uint32)

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return shared_sampler

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()
    requests = [
        _make_sampling_request(0, 0.7, 0.95),
        _make_sampling_request(1, 0.7, 0.95),
        _make_sampling_request(2, 0.7, 0.95),
        _make_sampling_request(3, 0.7, 0.95),
    ]

    input_tokens = mx.array([[1], [2], [3], [4]], dtype=mx.uint32)
    sampled, _ = MLLMBatchGenerator._step(
        gen, input_tokens, cache=[], requests=requests
    )

    # Exactly one make_sampler + one sampler invocation on the full batch.
    assert len(make_sampler_calls) == 1
    assert make_sampler_calls[0] == {"temp": 0.7, "top_p": 0.95}
    assert len(shared_sampler_invocations) == 1
    assert shared_sampler_invocations[0] == (4, 4)
    assert sampled.shape == (4,)


def test_step_caches_shared_sampler_across_calls(monkeypatch):
    """Repeated steps with the same (temp, top_p) reuse the cached sampler."""
    make_sampler_calls = []

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()
    requests = [
        _make_sampling_request(0, 0.7, 0.95),
        _make_sampling_request(1, 0.7, 0.95),
    ]

    for _ in range(5):
        MLLMBatchGenerator._step(
            gen,
            mx.array([[1], [2]], dtype=mx.uint32),
            cache=[],
            requests=requests,
        )

    # Cache key is stable, so make_sampler is invoked exactly once across
    # five decode steps — this is the per-token amortization we shipped for.
    assert len(make_sampler_calls) == 1


def test_step_param_change_invalidates_cached_sampler(monkeypatch):
    """When (temp, top_p) flips, _shared_batch_sampler is rebuilt."""
    make_sampler_calls = []

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(0, 0.7, 0.95),
            _make_sampling_request(1, 0.7, 0.95),
        ],
    )
    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(0, 0.3, 0.95),
            _make_sampling_request(1, 0.3, 0.95),
        ],
    )

    assert make_sampler_calls == [
        {"temp": 0.7, "top_p": 0.95},
        {"temp": 0.3, "top_p": 0.95},
    ]


def test_step_heterogeneous_requests_use_per_row_loop(monkeypatch):
    """Mixed (temp, top_p) falls back to the per-row loop; each request's
    sampler is built once and cached on the request via _cached_sampler."""
    make_sampler_calls = []

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()
    req_a = _make_sampling_request(0, 0.7, 0.95)
    req_b = _make_sampling_request(1, 0.3, 0.80)

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[req_a, req_b],
    )
    # Two distinct samplers, one per request.
    assert make_sampler_calls == [
        {"temp": 0.7, "top_p": 0.95},
        {"temp": 0.3, "top_p": 0.80},
    ]
    # Both got their per-request cache populated for future reuse.
    assert req_a._cached_sampler[0] == (0.7, 0.95)
    assert req_b._cached_sampler[0] == (0.3, 0.80)
    # Shared batch sampler must NOT have been populated for the mixed batch
    # (homogeneous fast path is the only writer).
    assert gen._shared_batch_sampler is None


def test_step_b1_homogeneous_still_uses_shared_sampler(monkeypatch):
    """B=1 still routes through the homogeneous fast path. Trivially equal
    to the legacy loop semantically, but proves the perf claim's B=1
    "unchanged" baseline isn't actually a sneaky regression."""
    make_sampler_calls = []

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()
    MLLMBatchGenerator._step(
        gen,
        mx.array([[1]], dtype=mx.uint32),
        cache=[],
        requests=[_make_sampling_request(0, 0.7, 0.95)],
    )

    assert len(make_sampler_calls) == 1
    assert gen._shared_batch_sampler is not None
    assert gen._shared_batch_sampler[0] == (0.7, 0.95)


def test_step_batch_uses_dataclass_defaults(monkeypatch):
    """A batch of requests using only the MLLMBatchRequest dataclass
    defaults (temperature=0.7, top_p=0.9) — the canonical concurrent
    benchmark shape — must hit the fast path."""
    make_sampler_calls = []

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()
    # Build via positional defaults only — never overriding temp/top_p.
    requests = [
        MLLMBatchRequest(uid=i, request_id=f"d{i}", prompt="hi") for i in range(4)
    ]

    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2], [3], [4]], dtype=mx.uint32),
        cache=[],
        requests=requests,
    )

    assert len(make_sampler_calls) == 1
    assert make_sampler_calls[0] == {"temp": 0.7, "top_p": 0.9}


def test_step_heterogeneous_then_homogeneous_populates_shared(monkeypatch):
    """A mixed batch leaves ``_shared_batch_sampler`` at None; the next
    homogeneous batch must then populate it. Guards against a regression
    where the het path could leak state that suppressed the fast path."""
    make_sampler_calls = []

    def fake_make_sampler(**kwargs):
        make_sampler_calls.append(kwargs)
        return lambda x: mx.zeros((x.shape[0],), dtype=mx.uint32)

    monkeypatch.setattr("vllm_mlx.mllm_batch_generator.make_sampler", fake_make_sampler)

    gen = _make_step_stub_generator()

    # First batch: mixed params → legacy loop, shared cache untouched.
    MLLMBatchGenerator._step(
        gen,
        mx.array([[1], [2]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(0, 0.7, 0.95),
            _make_sampling_request(1, 0.3, 0.80),
        ],
    )
    assert gen._shared_batch_sampler is None
    assert len(make_sampler_calls) == 2

    # Second batch: homogeneous → fast path fires + populates cache.
    MLLMBatchGenerator._step(
        gen,
        mx.array([[3], [4]], dtype=mx.uint32),
        cache=[],
        requests=[
            _make_sampling_request(2, 0.5, 0.85),
            _make_sampling_request(3, 0.5, 0.85),
        ],
    )
    assert gen._shared_batch_sampler is not None
    assert gen._shared_batch_sampler[0] == (0.5, 0.85)
    # 3 total: 2 from the het batch + 1 fresh for the new homogeneous key.
    assert len(make_sampler_calls) == 3
