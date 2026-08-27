# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MLLMBatchGenerator model-call kwargs.

Some mlx-vlm model classes (notably ``Gemma3ForConditionalGeneration``)
declare ``pixel_values`` as a *required* positional kwarg in ``__call__``,
even though the inner ``get_input_embeddings`` already handles ``None`` for
the text-only path. Omitting the kwarg raises ``TypeError`` for every
text-only request to those models, so ``_run_vision_encoding`` must always
pass it through — including when it's ``None``.
"""

import base64
import io

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_mlx.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    _model_supports_vision_feature_cache,
)


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
# Chunked text-only prefill — issue #1187, Problem B
# ---------------------------------------------------------------------------
#
# A VLM served on the MLLM path prefills a text-only prompt (e.g. a "test"
# message expanded to ~20k tokens by a large Hermes tool schema) through the
# language model. Doing that in a single forward materializes activations for
# every position AND projects logits over every position
# (``[1, seqlen, vocab]``, vocab 262144) — ~20 GB transient on gemma-4-26b,
# enough to max out a 48 GB M4 Max. The fix prefills the prompt prefix in
# bounded chunks (``min(prefill_step_size, 2048)``), evaluating only the KV
# cache state per chunk (mlx prunes the unused lm_head projection), then runs
# a single last-token forward for the ``[1, 1, vocab]`` logits actually
# sampled. Measured end-to-end on gemma-4-26b: 35.2 GB → 18.4 GB peak, ~2x
# faster, identical sampled token. Images are excluded (pixel features must
# stay aligned with placeholder tokens in one vision-merge forward).


class _ChunkRecordingModel:
    """VLM stub recording every forward's (seqlen, kwargs). Returns
    full-sequence ``LanguageModelOutput``-shaped logits so the generator's
    ``hasattr(output, "logits")`` branch and last-token slice are exercised."""

    def __init__(self, vocab: int = 8):
        self.calls: list[tuple[int, dict]] = []
        self.vocab = vocab
        self.language_model = object()

    def __call__(self, input_ids, cache=None, **kwargs):
        seqlen = input_ids.shape[1]
        self.calls.append((seqlen, kwargs))

        class _Out:
            pass

        out = _Out()
        out.logits = mx.zeros((1, seqlen, self.vocab))
        return out


class _FakeCache:
    """Minimal KV-cache stand-in exposing an evaluable ``.state`` and
    counting how many times the chunk barrier reads it."""

    def __init__(self):
        self.state_reads = 0

    @property
    def state(self):
        self.state_reads += 1
        return mx.zeros((1,))


def _make_bare_generator(prefill_step_size: int, model) -> MLLMBatchGenerator:
    """Construct just enough of a generator for ``_run_vision_encoding``
    (reads ``self.model`` / ``self.prefill_step_size`` only)."""
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.model = model
    gen.language_model = getattr(model, "language_model", model)
    gen.prefill_step_size = prefill_step_size
    return gen


def _make_ids_request(n_tokens: int, *, pixel_values=None, image_grid_thw=None):
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="x",
        max_tokens=8,
        input_ids=mx.arange(n_tokens, dtype=mx.int32),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        extra_kwargs={},
    )


def test_run_vision_encoding_chunks_text_only_prefill():
    """A long text-only prompt is prefilled in ``min(step, 2048)`` chunks
    plus a final single-token forward; nothing is projected over the whole
    prompt."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=22000, model=model)
    cache = [_FakeCache()]

    logits = gen._run_vision_encoding(_make_ids_request(5000), cache=cache)

    prefix_seqlens = [c[0] for c in model.calls[:-1]]
    last_seqlen, last_kwargs = model.calls[-1]
    # prefix = 4999 tokens, chunk = min(22000, 2048) = 2048 → 2048, 2048, 903
    assert prefix_seqlens == [2048, 2048, 903]
    # Every chunk is text-only (pixel_values explicitly None for the strict
    # Gemma signatures) — never the full prompt in one shot.
    assert all(c[1].get("pixel_values", "MISSING") is None for c in model.calls[:-1])
    # Final forward is a single token that carries no image.
    assert last_seqlen == 1
    assert last_kwargs.get("pixel_values", "MISSING") is None
    # Returned logits are the last position only, so callers never touch a
    # ``[1, seqlen, vocab]`` tensor.
    assert logits.shape == (1, 1, model.vocab)
    # The per-chunk barrier read the cache state at least once per chunk.
    assert cache[0].state_reads >= len(prefix_seqlens)


def test_run_vision_encoding_chunk_respects_smaller_prefill_step_size():
    """An operator who set a *smaller* ``--prefill-step-size`` (memory-tight
    box) gets chunks no larger than they asked for."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=512, model=model)
    cache = [_FakeCache()]

    gen._run_vision_encoding(_make_ids_request(1500), cache=cache)

    # prefix = 1499, chunk = min(512, 2048) = 512 → 512, 512, 475
    assert [c[0] for c in model.calls[:-1]] == [512, 512, 475]
    assert model.calls[-1][0] == 1


def test_run_vision_encoding_image_request_is_not_chunked():
    """Image requests keep the single vision-merge forward (pixel features
    must stay aligned with their placeholder tokens)."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=22000, model=model)
    pixels = mx.zeros((1, 3, 4, 4))
    cache = [_FakeCache()]

    gen._run_vision_encoding(_make_ids_request(5000, pixel_values=pixels), cache=cache)

    # Exactly one forward over the whole prompt, pixel_values passed through.
    assert len(model.calls) == 1
    assert model.calls[0][0] == 5000
    assert model.calls[0][1].get("pixel_values") is pixels


def test_run_vision_encoding_no_cache_keeps_single_forward_for_long_text():
    """Without a cache the split is impossible (no KV to carry prefix state),
    so even a long text-only prompt stays a single forward."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=2048, model=model)

    gen._run_vision_encoding(_make_ids_request(5000), cache=None)

    assert len(model.calls) == 1
    assert model.calls[0][0] == 5000


def test_run_vision_encoding_chunks_with_all_valid_attention_mask():
    """A processor-shaped text-only request — the realistic case where
    ``mlx_vlm.prepare_inputs`` returns a one-row, all-valid ``attention_mask``
    — still takes the chunked path. ``attention_mask`` is a *separate* request
    field (``_preprocess_request`` excludes it from ``extra_kwargs``), so it
    does NOT trip the ``no_extra_kwargs`` gate; it is simply dropped from the
    chunked forwards, which is lossless for an all-valid, single-request mask
    (mlx-lm's own text prefill likewise passes no mask and relies on the
    causal mask). Regression guard: without this, a reviewer might assume the
    mask disables the memory fix — it must not."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=22000, model=model)
    req = _make_ids_request(5000)
    # Processor supplies a one-row, all-valid mask (a separate field, not in
    # extra_kwargs). extra_kwargs stays empty, exactly as _preprocess_request
    # builds it for a text-only prompt.
    req.attention_mask = mx.ones((1, 5000), dtype=mx.int32)
    assert req.extra_kwargs == {}
    cache = [_FakeCache()]

    logits = gen._run_vision_encoding(req, cache=cache)

    # Still chunked (prompt > one chunk), NOT a single full-prompt forward.
    prefix_seqlens = [c[0] for c in model.calls[:-1]]
    assert prefix_seqlens == [2048, 2048, 903]
    assert model.calls[-1][0] == 1
    assert logits.shape == (1, 1, model.vocab)
    # The all-valid mask is dropped on the chunked path (no per-chunk mask).
    assert all("attention_mask" not in c[1] for c in model.calls)


def test_chunking_falls_back_to_single_forward_with_partial_attention_mask():
    """A NON-all-valid mask (e.g. left-padding, or a reused cache entry with a
    shorter valid span) must NOT be dropped — dropping it on the chunked path
    would silently change attention semantics and corrupt the logits. Such a
    request keeps the single forward, which passes the mask through intact."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=22000, model=model)
    req = _make_ids_request(5000)
    # First 3 positions masked out (0 = do-not-attend) → carries information.
    mask = mx.concatenate(
        [mx.zeros((1, 3), dtype=mx.int32), mx.ones((1, 4997), dtype=mx.int32)],
        axis=1,
    )
    req.attention_mask = mask
    cache = [_FakeCache()]

    gen._run_vision_encoding(req, cache=cache)

    # One forward over the whole prompt, the partial mask forwarded intact.
    # (``_run_vision_encoding`` nulls ``request.attention_mask`` afterwards, so
    # compare against the captured object, not the reset field.)
    assert len(model.calls) == 1
    assert model.calls[0][0] == 5000
    assert model.calls[0][1].get("attention_mask") is mask


def test_chunking_falls_back_to_single_forward_with_extra_kwargs():
    """If a processor ever emits sequence-aligned extra kwargs (e.g.
    ``token_type_ids``) for a text-only request, we must NOT chunk (we would
    silently drop or mis-slice them) — fall back to the single forward that
    forwards ``kwargs`` intact."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=22000, model=model)
    req = _make_ids_request(5000)
    req.extra_kwargs = {"token_type_ids": mx.zeros((1, 5000), dtype=mx.int32)}
    cache = [_FakeCache()]

    gen._run_vision_encoding(req, cache=cache)

    # One forward over the whole prompt, extra kwargs preserved.
    assert len(model.calls) == 1
    assert model.calls[0][0] == 5000
    assert "token_type_ids" in model.calls[0][1]


def test_run_vision_encoding_single_token_uses_single_forward():
    """A 1-token prompt has no prefix to chunk; it stays on the single
    forward (no empty-prefix forward is ever submitted)."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=2048, model=model)

    gen._run_vision_encoding(_make_ids_request(1), cache=[_FakeCache()])

    assert len(model.calls) == 1
    assert model.calls[0][0] == 1


def test_run_vision_encoding_short_text_prompt_uses_single_forward():
    """A prompt that fits inside one chunk keeps the *original* single
    forward — no second forward, no per-chunk ``mx.eval``/``mx.clear_cache``
    barrier on the hot path. Chunking only engages once the prompt is longer
    than one chunk, where the un-chunked activations + full-sequence logits
    would actually spike memory (#1187 B). This guards the latency of the
    common short-prompt case against the chunking added for long prompts."""
    model = _ChunkRecordingModel()
    gen = _make_bare_generator(prefill_step_size=22000, model=model)
    cache = [_FakeCache()]

    # 100 tokens << chunk = min(22000, 2048) = 2048 → single forward.
    gen._run_vision_encoding(_make_ids_request(100), cache=cache)

    assert len(model.calls) == 1
    assert model.calls[0][0] == 100
    # No barrier ran, so the cache state was never force-evaluated.
    assert cache[0].state_reads == 0


# ---------------------------------------------------------------------------
# Numerical equivalence — chunked prefill must match the single forward.
#
# `_TinyCausalLM` is a real (tiny) causal transformer using mlx-lm's own
# `KVCache` / `RotatingKVCache` + causal-mask helper, so the chunked path
# actually computes and retains prefix K/V. We compare the last-position
# logits, the cache offset, the sampled token, AND a following decode step
# against a single-forward reference — including a `RotatingKVCache` whose
# window is smaller than the prompt, which forces sliding-window rotation
# across chunk boundaries (the case #1187's gemma-4 mix relies on).
# ---------------------------------------------------------------------------


class _TinyCausalLM:
    """Minimal real causal LM (embedding + N attention layers + tied-free
    output) driven by mlx-lm caches, for chunk-vs-single equivalence checks."""

    def __init__(
        self, vocab: int = 48, dim: int = 32, n_heads: int = 4, n_layers: int = 2
    ):
        mx.random.seed(0)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab, dim)
        self.wq = [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        self.wk = [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        self.wv = [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        self.wo = [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dim)
        self.out = nn.Linear(dim, vocab, bias=False)
        self.language_model = self
        for m in [
            self.embed,
            self.norm,
            self.out,
            *self.wq,
            *self.wk,
            *self.wv,
            *self.wo,
        ]:
            mx.eval(m.parameters())

    def __call__(self, input_ids, cache=None, **kwargs):
        from mlx_lm.models.base import create_attention_mask

        B, L = input_ids.shape
        h = self.embed(input_ids)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for i in range(self.n_layers):
            c = cache[i] if cache is not None else None
            q = (
                self.wq[i](h)
                .reshape(B, L, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            k = (
                self.wk[i](h)
                .reshape(B, L, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            v = (
                self.wv[i](h)
                .reshape(B, L, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            if c is not None:
                k, v = c.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )
            o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
            h = h + self.wo[i](o)

        class _Out:
            pass

        out = _Out()
        out.logits = self.out(self.norm(h))
        return out


def _make_caches(kind: str, n_layers: int):
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    if kind == "kv":
        return [KVCache() for _ in range(n_layers)]
    max_size = int(kind.split(":")[1])
    return [RotatingKVCache(max_size=max_size) for _ in range(n_layers)]


@pytest.mark.parametrize(
    "kind",
    [
        "kv",  # plain growing cache
        "rot:64",  # rotating window larger than prompt → no rotation
        "rot:16",  # rotating window < prompt → forces sliding-window rotation
    ],
)
def test_chunked_prefill_matches_single_forward_numerically(kind):
    """Chunked prefill (via the real `_run_vision_encoding`) produces the same
    last-token logits, cache offset, sampled token, and next-decode logits as
    a single forward — for plain and rotating KV caches (#1187 B)."""
    n_layers = 2
    model = _TinyCausalLM(n_layers=n_layers)
    # Small chunk to force multiple prefix chunks (prefix=39 → 8,8,8,8,7).
    gen = _make_bare_generator(prefill_step_size=8, model=model)
    n = 40  # < vocab (48) so every token id is valid
    # Same ids the chunked request uses (``_make_ids_request`` → ``arange(n)``),
    # so both paths run on identical input.
    ids = mx.arange(n, dtype=mx.int32)

    # Single-forward reference. Materialize the logits BEFORE the decode step
    # below mutates the cache in place (rotating caches write K/V in place, so
    # a still-lazy logits graph would otherwise read post-decode state).
    single_cache = _make_caches(kind, n_layers)
    single_last = model(ids[None, :], cache=single_cache).logits[:, -1, :]
    mx.eval(single_last)

    # Chunked prefill through the production method.
    chunked_cache = _make_caches(kind, n_layers)
    chunked_last = gen._run_vision_encoding(_make_ids_request(n), cache=chunked_cache)[
        :, -1, :
    ]
    mx.eval(chunked_last)

    def _offset(c):
        o = c.offset
        return o.item() if hasattr(o, "item") else o

    # Cache filled to the same absolute length by both paths (captured BEFORE
    # the decode step below advances it).
    assert _offset(single_cache[0]) == _offset(chunked_cache[0]) == n

    # One decode step on top of each post-prefill cache.
    next_tok = mx.argmax(single_last, axis=-1).reshape(1, 1)
    dec_single = model(next_tok, cache=single_cache).logits[:, -1, :]
    dec_chunked = model(next_tok, cache=chunked_cache).logits[:, -1, :]
    mx.eval(dec_single, dec_chunked)

    # Last-token logits agree within fp32 attention-reduction noise.
    assert mx.allclose(single_last, chunked_last, atol=1e-4, rtol=1e-4)
    # Sampled token is identical.
    assert mx.argmax(single_last, -1).item() == mx.argmax(chunked_last, -1).item()
    # And decoding continues identically from the chunk-built cache.
    assert mx.allclose(dec_single, dec_chunked, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Memory — the chunked path must not materialize `[1, seqlen, vocab]` logits,
# while still genuinely computing (and retaining) the prefix K/V.
# ---------------------------------------------------------------------------


class _KVWritingProjModel:
    """Big-vocab stub that (a) writes input-dependent K/V into a real cache so
    `mx.eval(cache.state)` forces the prefix computation, and (b) projects
    `[1, seqlen, vocab]` logits so peak memory reflects whether the caller
    projected the whole prompt or just the last token."""

    def __init__(self, hidden: int, vocab: int, n_heads: int = 4):
        self.embed = nn.QuantizedEmbedding(vocab, hidden, group_size=64, bits=4)
        self.wkv = nn.Linear(hidden, hidden, bias=False)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.vocab = vocab
        self.language_model = self
        mx.eval(self.embed.parameters(), self.wkv.parameters())

    def __call__(self, input_ids, cache=None, **kwargs):
        B, L = input_ids.shape
        h = self.embed(input_ids)  # input-dependent hidden
        if cache is not None:
            kv = (
                self.wkv(h)
                .reshape(B, L, self.n_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            for c in cache:
                c.update_and_fetch(kv, kv)  # store input-dependent K/V

        class _Out:
            pass

        out = _Out()
        out.logits = self.embed.as_linear(h)  # [1, seqlen, vocab]
        return out


def test_chunked_prefill_avoids_full_sequence_logits_materialization():
    """The chunked path must not materialize a `[1, seqlen, vocab]` logits
    tensor, while still computing the prefix K/V (a real `KVCache` whose
    `.state` `mx.eval` forces). Peak is compared against the single-forward
    (no-cache) path on the SAME method (#1187 B)."""
    from mlx_lm.models.cache import KVCache

    hidden, vocab, n = 128, 32768, 4096
    model = _KVWritingProjModel(hidden, vocab)
    gen = _make_bare_generator(prefill_step_size=2048, model=model)

    # Chunked (real KVCache) FIRST → prefix K/V computed per chunk, prefix
    # logits pruned, only `[1, 1, vocab]` evaled. Measured first so no residual
    # from the single path pollutes it.
    cache = [KVCache()]
    mx.clear_cache()
    mx.reset_peak_memory()
    l_chunked = gen._run_vision_encoding(_make_ids_request(n), cache=cache)
    chunked_shape = l_chunked.shape
    mx.eval(l_chunked[:, -1, :])
    # The prefix K/V really was materialized (offset advanced to n-1 over the
    # prefix chunks + 1 for the last-token forward = n).
    off = cache[0].offset
    assert (off.item() if hasattr(off, "item") else off) == n
    peak_chunked = mx.get_peak_memory()
    del l_chunked
    mx.clear_cache()

    # Single forward (cache=None) → full `[1, n, vocab]` logits materialized
    # because slicing `[:, -1, :]` does not prune the lm_head matmul.
    mx.reset_peak_memory()
    l_single = gen._run_vision_encoding(_make_ids_request(n), cache=None)
    single_shape = l_single.shape
    mx.eval(l_single[:, -1, :])
    peak_single = mx.get_peak_memory()

    assert single_shape == (1, n, vocab)
    assert chunked_shape == (1, 1, vocab)
    # The single path's transient is dominated by the `[1, n, vocab]` fp32
    # matmul output (~0.5 GB here); the chunked path never allocates it.
    full_logits_bytes = n * vocab * 4
    assert peak_single - peak_chunked > full_logits_bytes * 0.4, (
        f"chunked peak {peak_chunked} not meaningfully below single "
        f"{peak_single} (expected ≥{full_logits_bytes * 0.4:.0f} B lower)"
    )


# ---------------------------------------------------------------------------
# Shutdown — mx.synchronize must not propagate cross-thread errors
# ---------------------------------------------------------------------------


def test_close_swallows_synchronize_thread_error(monkeypatch):
    """`close()` must not propagate RuntimeError from mx.synchronize.

    mlx-lm 0.31.3+ streams are thread-local. When the engine is torn down
    from a thread that isn't the one that owns the generator's stream,
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


# ---------------------------------------------------------------------------
# Per-batch cap regression — issue #682
# ---------------------------------------------------------------------------
#
# A high-resolution image (e.g. a 1920×1080 desktop screenshot) decodes to
# ~2200 vision tokens with Qwen3-VL's preprocessor. The original
# ``MLLMSchedulerConfig.prefill_step_size=1024`` default + the
# ``BatchedEngine._start_mllm`` fallback of 2048 (from SchedulerConfig)
# were both too low for typical VLM workloads. With ``prefill_step_size=
# 2048`` a single-request 2292-token batch failed the cap and the
# MLLMScheduler swallowed the ValueError as a soft truncation — the
# route returned 200 OK with empty content + finish_reason=length and
# Desktop rendered the misleading "Reached max_tokens before any output"
# error.
#
# The fix bumps the MLLM-side prefill_step_size to 8192 in two places:
#   - ``MLLMSchedulerConfig.prefill_step_size`` default (for direct
#     scheduler construction, e.g. programmatic use).
#   - ``BatchedEngine._start_mllm`` reads the SchedulerConfig value and
#     applies ``_resolve_mllm_prefill_step_size`` (a bump-policy, NOT a
#     floor) so a server started with the text-LLM default
#     (--prefill-step-size 2048) gets the VLM-tuned 8192. Explicit
#     operator-set values are honored as-is — including smaller ones
#     for memory-constrained deployments (codex r2 MAJOR contract).
#
# The cap arithmetic itself is unchanged — it still bounds aggregate
# merge-time memory; the bump-policy only raises the per-request budget
# for image-heavy prompts on the default code path.


def _make_cap_request(uid: int, token_count: int) -> MLLMBatchRequest:
    """Build a request whose ``input_ids.size`` is ``token_count``."""
    return MLLMBatchRequest(
        uid=uid,
        request_id=f"r{uid}",
        prompt="x",
        max_tokens=8,
        input_ids=mx.zeros((token_count,), dtype=mx.int32),
    )


def _gen_with_prefill_cap(
    prefill_step_size: int, *, vision_prefill_token_budget: int | None = None
) -> MLLMBatchGenerator:
    """Generator with a tunable cap, no real model/processor needed.

    ``_process_prompts`` only reads ``self.prefill_step_size`` /
    ``self._stats`` / ``self.vision_cache`` before raising the cap error,
    so a bare construction is enough to exercise the check.
    """
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.prefill_step_size = prefill_step_size
    gen.vision_prefill_token_budget = (
        prefill_step_size
        if vision_prefill_token_budget is None
        else vision_prefill_token_budget
    )
    gen.vision_cache = None
    gen.model = object()
    gen.language_model = object()
    gen.processor = object()
    gen.mm_processor = None

    class _Stats:
        prompt_tokens = 0
        prompt_time = 0.0
        num_images_processed = 0
        vision_encoding_time = 0.0

    gen._stats = _Stats()
    return gen


def test_mllm_scheduler_config_default_vision_budget_covers_screenshot():
    """The independent vision budget must cover a typical screenshot.

    Pre-fix the default was 1024 — even an 800×600 image would have
    failed the cap on a direct ``MLLMSchedulerConfig()`` construction.
    Post-fix the default is 8192, comfortably above the ~2200-token
    Qwen3-VL output for 1920×1080.
    """
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

    cfg = MLLMSchedulerConfig()
    # 1920×1080 Qwen3-VL: ~2200 vision tokens + chat-template + text.
    # Default must be high enough that a single such request never
    # trips the cap on its own size (#682).
    assert cfg.vision_prefill_token_budget >= 8192, (
        "MLLMSchedulerConfig.vision_prefill_token_budget default "
        f"({cfg.vision_prefill_token_budget}) "
        f"must be at least 8192 to cover 1920×1080 screenshots without "
        f"tripping the per-batch cap (#682)."
    )


def test_vision_admission_budget_is_independent_from_prefill_chunk():
    """A profile-tuned 512 chunk must still admit a normal screenshot."""
    from vllm_mlx.mllm_batch_generator import _prefill_cap_violation
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

    cfg = MLLMSchedulerConfig(
        prefill_step_size=512,
        vision_prefill_token_budget=8192,
    )
    req = _make_vision_cap_request(uid=0, token_count=2292)

    assert cfg.prefill_step_size == 512
    assert _prefill_cap_violation([req], cfg.vision_prefill_token_budget) is None
    assert _prefill_cap_violation([req], cfg.prefill_step_size) is not None


def test_vision_budget_keeps_safe_floor_and_larger_operator_value():
    from vllm_mlx.engine.batched import _resolve_mllm_vision_prefill_token_budget

    assert (
        _resolve_mllm_vision_prefill_token_budget(
            512, configured=None, mllm_default=8192
        )
        == 8192
    )
    assert (
        _resolve_mllm_vision_prefill_token_budget(
            16384, configured=None, mllm_default=8192
        )
        == 16384
    )
    assert (
        _resolve_mllm_vision_prefill_token_budget(
            512, configured=256, mllm_default=8192
        )
        == 256
    )


def test_resolve_mllm_prefill_step_size_bumps_text_default_to_mllm_default():
    """Pin the MLLM ``prefill_step_size`` bump-policy (#682).

    The CLI ships ``--prefill-step-size 2048`` (text-LLM tuned). Without
    the bump, every Desktop sidecar serving a VLM would inherit 2048
    and trip the per-batch cap on a 1920×1080 screenshot.

    Codex r2 MAJOR: an earlier draft used ``max(value, 8192)`` which
    silently overrode memory-constrained operators who explicitly set
    a smaller value. The fix bumps only when the value matches the
    SchedulerConfig dataclass default — any explicit value is honored.

    Codex r3 NIT: the bump-policy is extracted as
    ``_resolve_mllm_prefill_step_size`` so this test exercises the
    production helper directly (not a copied mirror expression) and
    is robust to refactors of ``_start_mllm``.
    """
    from types import SimpleNamespace

    from vllm_mlx.engine.batched import _resolve_mllm_prefill_step_size
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig
    from vllm_mlx.scheduler import SchedulerConfig

    text_default = SchedulerConfig.__dataclass_fields__["prefill_step_size"].default
    mllm_default = MLLMSchedulerConfig.__dataclass_fields__["prefill_step_size"].default

    # The MLLM default must exceed the text default — otherwise the
    # bump is a no-op — and must cover a typical 1920×1080 screenshot.
    assert mllm_default > text_default, (
        f"MLLM default ({mllm_default}) must exceed text default "
        f"({text_default}); otherwise the #682 bump is inert."
    )
    assert mllm_default >= 8192, (
        f"MLLM default ({mllm_default}) must cover 1920×1080 Qwen3-VL "
        f"(~2200 tokens) with headroom for multi-image messages (#682)."
    )

    def _resolved(user_value):
        return _resolve_mllm_prefill_step_size(
            user_value,
            text_default=text_default,
            mllm_default=mllm_default,
        )

    # Default → bumped (the Desktop sidecar case).
    assert _resolved(text_default) == mllm_default, (
        f"text-LLM default ({text_default}) must bump to MLLM default "
        f"({mllm_default}) — this is the #682 fix for Desktop sidecars."
    )

    # Explicit smaller value → honored. This is the codex r2 MAJOR
    # contract: the engine must NOT silently override a user's
    # explicit smaller choice.
    for explicit_smaller in [256, 512, 1024, 1500]:
        assert _resolved(explicit_smaller) == explicit_smaller, (
            f"explicit prefill_step_size={explicit_smaller} must be "
            f"honored as-is (codex r2 MAJOR); got {_resolved(explicit_smaller)}"
        )

    # Explicit larger value → honored (high-end deployment).
    for explicit_larger in [4096, 8192, 16384, 65536]:
        assert _resolved(explicit_larger) == explicit_larger, (
            f"explicit prefill_step_size={explicit_larger} must be "
            f"honored as-is; got {_resolved(explicit_larger)}"
        )

    # ``None`` covers BOTH the "no scheduler_config" path AND the
    # "config object without the attribute" path — the latter via
    # ``getattr(cfg, "prefill_step_size", None)`` in ``_start_mllm``
    # returning ``None`` when the attribute is missing (codex r3 NIT).
    assert _resolved(None) == mllm_default, (
        "missing attribute / no scheduler_config must default to MLLM-tuned"
    )

    # And the getattr path: an object that genuinely lacks the attribute
    # also resolves to the MLLM default. Pins the "config attribute
    # absent" contract that codex r3 NIT called out as untested.
    cfg_without_attr = SimpleNamespace()  # no prefill_step_size attribute
    resolved_missing = _resolve_mllm_prefill_step_size(
        getattr(cfg_without_attr, "prefill_step_size", None),
        text_default=text_default,
        mllm_default=mllm_default,
    )
    assert resolved_missing == mllm_default

    # Explicit value EXACTLY equal to text_default is treated as
    # "took the default" — documented trade-off, #682 outweighs the
    # rare operator who explicitly wants 2048 on VLM. Pinned here so
    # a future refactor that flips the equality direction is caught.
    assert _resolved(text_default) == mllm_default


def test_per_batch_cap_fires_on_oversized_batch_with_actionable_message(
    monkeypatch,
):
    """The cap is still a real guard — it MUST fire when prompts truly
    exceed the budget, with an actionable error message.

    Codex r1 BLOCKING: an earlier draft made the cap tautological by
    deriving ``per_request_cap`` from the batch's own max. That removed
    the memory guard entirely. This test pins the cap as a real check
    and pins the error message wording so the MLLMScheduler client-error
    classifier and the routes/chat.py 400-mapping continue to match.
    """
    # Tiny cap to force the check to fire with a small request size.
    gen = _gen_with_prefill_cap(prefill_step_size=100)
    monkeypatch.setattr(gen, "_preprocess_request", lambda req: None)

    # 500-token VISION request, cap = 100 × 1 = 100 ⇒ 500 > 100 ⇒ raises.
    # (Text-only requests are exempt from the cap since #1848, so the cap
    #  guard is pinned against a vision-bearing request.)
    request = _make_vision_cap_request(uid=0, token_count=500)

    with pytest.raises(ValueError) as excinfo:
        MLLMBatchGenerator._process_prompts(gen, [request])

    msg = str(excinfo.value)
    # Must keep this exact substring — MLLMScheduler's client-error
    # classifier matches on it (#682). If the phrase drifts the
    # soft-truncation regression comes back.
    assert "exceeds the per-batch cap" in msg, (
        f"cap error must keep the marker substring; got: {msg}"
    )
    # Actionable levers — must call out image-downscale for VLM users.
    assert "downscale the image" in msg, (
        f"cap error must suggest image downscale; got: {msg}"
    )
    assert "--prefill-step-size" in msg, (
        f"cap error must mention --prefill-step-size for the text path; got: {msg}"
    )


def test_per_batch_cap_does_not_fail_at_default_on_typical_screenshot(
    monkeypatch,
):
    """End-to-end pin: with the production MLLM default
    ``prefill_step_size=8192``, a single 2292-token request (Qwen3-VL
    on a 1920×1080 screenshot) must NOT trip the cap.

    Pre-fix with default 2048 this raised ValueError("exceeds the
    per-batch cap") which the scheduler swallowed as
    ``finish_reason="length"`` + empty content (#682).
    """
    gen = _gen_with_prefill_cap(prefill_step_size=8192)
    monkeypatch.setattr(gen, "_preprocess_request", lambda req: None)

    # 2292 tokens — typical Qwen3-VL token count for a 1920×1080 image,
    # carried on a vision-bearing request (this is an image prompt).
    request = _make_vision_cap_request(uid=0, token_count=2292)

    # The function will still raise SOMETHING downstream (we handed it
    # bare ``object()`` for model / language_model so the real prefill
    # path can't run), but it must NOT be the per-batch-cap error.
    with pytest.raises(Exception) as excinfo:  # noqa: BLE001 — see below
        MLLMBatchGenerator._process_prompts(gen, [request])

    err_msg = str(excinfo.value)
    assert "exceeds the per-batch cap" not in err_msg, (
        f"with the production MLLM default (8192), a 2292-token "
        f"single-request batch must pass the cap; got: {err_msg}"
    )


def test_profile_chunk_uses_independent_vision_budget_in_process_prompts(
    monkeypatch,
):
    """Exercise the production admission call site for the Gemma profile."""
    gen = _gen_with_prefill_cap(
        prefill_step_size=512,
        vision_prefill_token_budget=8192,
    )
    monkeypatch.setattr(gen, "_preprocess_request", lambda req: None)
    request = _make_vision_cap_request(uid=0, token_count=2292)

    with pytest.raises(Exception) as excinfo:  # noqa: BLE001 — bare model stub
        MLLMBatchGenerator._process_prompts(gen, [request])

    assert "exceeds the per-batch cap" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Vision-feature cache — issue #1854
# ---------------------------------------------------------------------------
#
# rapid's MLLM continuous-batching prefill re-ran the vision encoder on every
# request, so a *repeated* image (the common multi-turn "ask again about the
# same screenshot" case) paid the full ~0.3-0.4s vision cost each time — making
# vision TTFT ~2.4x stock mlx-vlm, whose server caches projected image features
# across requests. The fix forwards mlx-vlm's own ``vision_cache`` / ``_image_key``
# contract into the model forward so ``get_input_embeddings`` reuses the cached
# ``vision_tower + embed_vision`` output. It is gated to model families that
# actually honour the contract (gemma-4 today) and only engages when the
# request carries images.


class _VisionCacheModel(_RecordingModel):
    """Supported-model stub whose ``get_input_embeddings`` honours the mlx-vlm
    ``vision_cache`` / ``_image_key`` contract (detected by source inspection).
    ``__call__`` is inherited from ``_RecordingModel`` so tests can read the
    kwargs the generator forwarded."""

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        # The literal ``vision_cache`` in the body is the marker
        # ``_model_supports_vision_feature_cache`` inspects for.
        vision_cache = kwargs.get("vision_cache")  # noqa: F841
        image_key = kwargs.get("_image_key")  # noqa: F841
        return None


def _make_vision_request(*, pixel_values, vision_feature_key):
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="describe",
        max_tokens=8,
        input_ids=mx.array([1, 2, 3], dtype=mx.int32),
        pixel_values=pixel_values,
        vision_feature_key=vision_feature_key,
        extra_kwargs={},
    )


def test_model_supports_vision_feature_cache_detects_contract():
    """Detection is by ``get_input_embeddings`` source: a model that reads
    ``vision_cache`` is supported; a plain VLM stub is not."""
    assert _model_supports_vision_feature_cache(_VisionCacheModel()) is True
    assert _model_supports_vision_feature_cache(_RecordingModel()) is False


def test_model_supports_vision_feature_cache_false_without_get_input_embeddings():
    class _NoEmbed:
        def __call__(self, *a, **k):
            return None

    assert _model_supports_vision_feature_cache(_NoEmbed()) is False


def test_mllm_batch_request_vision_feature_key_defaults_none():
    req = MLLMBatchRequest(uid=0, request_id="r", prompt="p")
    assert req.vision_feature_key is None


def test_supported_model_enables_feature_cache():
    """A supported model + ``enable_vision_cache=True`` wires a live feature
    cache; an unsupported model leaves it off (no behaviour change)."""
    pytest.importorskip("mlx_vlm.vision_cache")

    supported = MLLMBatchGenerator(
        model=_VisionCacheModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    assert supported._supports_vision_feature_cache is True
    assert supported._vision_feature_cache is not None

    unsupported = MLLMBatchGenerator(
        model=_RecordingModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    assert unsupported._supports_vision_feature_cache is False
    assert unsupported._vision_feature_cache is None


def test_feature_cache_disabled_when_vision_cache_off():
    """``enable_vision_cache=False`` disables the feature cache even for a
    supported model."""
    gen = MLLMBatchGenerator(
        model=_VisionCacheModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )
    assert gen._supports_vision_feature_cache is False
    assert gen._vision_feature_cache is None


def test_run_vision_encoding_injects_vision_cache_for_image_request():
    """A supported model with an image request forwards ``vision_cache`` +
    ``_image_key`` so ``get_input_embeddings`` can reuse projected features."""
    pytest.importorskip("mlx_vlm.vision_cache")
    model = _VisionCacheModel()
    gen = MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    request = _make_vision_request(
        pixel_values=mx.zeros((1, 3, 4, 4)), vision_feature_key="deadbeef"
    )

    gen._run_vision_encoding(request, cache=None)

    assert model.last_call_kwargs.get("_image_key") == "deadbeef"
    assert model.last_call_kwargs.get("vision_cache") is gen._vision_feature_cache


def test_run_vision_encoding_no_vision_cache_for_text_only():
    """A text-only request (no pixel_values, no key) never carries the cache
    kwargs even on a supported model."""
    pytest.importorskip("mlx_vlm.vision_cache")
    model = _VisionCacheModel()
    gen = MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    request = _make_vision_request(pixel_values=None, vision_feature_key=None)

    gen._run_vision_encoding(request, cache=None)

    assert "vision_cache" not in model.last_call_kwargs
    assert "_image_key" not in model.last_call_kwargs


def test_run_vision_encoding_no_vision_cache_for_unsupported_model():
    """An unsupported model keeps the unchanged full-forward path — the cache
    kwargs are never injected even when the request carries images."""
    model = _RecordingModel()
    gen = MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    request = _make_vision_request(
        pixel_values=mx.zeros((1, 3, 4, 4)), vision_feature_key="deadbeef"
    )

    gen._run_vision_encoding(request, cache=None)

    assert "vision_cache" not in model.last_call_kwargs
    assert "_image_key" not in model.last_call_kwargs


class _CachingBehaviorModel:
    """Model stub that mirrors mlx-vlm's real forward contract: ``__call__``
    binds ``pixel_values`` as a named arg (as rapid passes it via kwargs) and
    delegates to ``get_input_embeddings``, forwarding the remaining kwargs —
    exactly like gemma-4. The cache lookup + "encode" (``encode_count`` bump)
    happens INSIDE ``get_input_embeddings`` on a miss, so the test proves the
    real forwarding path, not a shortcut in ``__call__``."""

    def __init__(self):
        self.encode_count = 0
        self.language_model = object()

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        # Mirror gemma-4: read the quoted contract keys "vision_cache" /
        # "_image_key" from kwargs; on a miss, "encode" (count) and store.
        vision_cache = kwargs.get("vision_cache")
        image_key = kwargs.get("_image_key")
        if pixel_values is None:
            return None
        if vision_cache is not None and image_key is not None:
            if vision_cache.get(image_key) is None:
                self.encode_count += 1
                vision_cache.put(image_key, mx.zeros((1, 4)))
        else:
            # No cache forwarded → the encoder always runs (pre-fix behaviour).
            self.encode_count += 1
        return None

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        # ``pixel_values`` is bound from rapid's kwargs; forward the rest
        # (incl. vision_cache/_image_key) into get_input_embeddings.
        self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return mx.zeros((1, 1, 8))


def test_repeated_image_skips_encoder_distinct_image_reencodes():
    """The wiring dedups the vision encoder: a repeated image reuses cached
    features (no re-encode); a distinct image encodes again."""
    pytest.importorskip("mlx_vlm.vision_cache")
    model = _CachingBehaviorModel()
    gen = MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    assert gen._supports_vision_feature_cache is True

    def run(key):
        gen._run_vision_encoding(
            _make_vision_request(
                pixel_values=mx.zeros((1, 3, 4, 4)), vision_feature_key=key
            ),
            cache=None,
        )

    run("image-A")  # miss → encode
    run("image-A")  # hit  → skip
    assert model.encode_count == 1, "repeated image must not re-run the encoder"

    run("image-B")  # distinct → encode
    assert model.encode_count == 2, "a distinct image must encode again"


class _NoContractCachingModel(_CachingBehaviorModel):
    """Same delegating ``__call__`` as ``_CachingBehaviorModel`` but its
    ``get_input_embeddings`` omits the quoted contract markers, so the gate
    does NOT detect it as cache-capable. It still counts an encode per call
    (there is no cache to consult), which is exactly the pre-fix behaviour the
    negative control asserts."""

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if pixel_values is not None:
            self.encode_count += 1
        return None


def test_unsupported_model_always_encodes_repeated_image():
    """Negative control: a model that does not honour the contract keeps the
    unchanged path — the (stubbed) encoder runs on every request even for a
    repeated image, because no cache kwargs are forwarded."""
    model = _NoContractCachingModel()
    gen = MLLMBatchGenerator(
        model=model,
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    assert gen._supports_vision_feature_cache is False

    for _ in range(3):
        gen._run_vision_encoding(
            _make_vision_request(
                pixel_values=mx.zeros((1, 3, 4, 4)), vision_feature_key="image-A"
            ),
            cache=None,
        )
    assert model.encode_count == 3


def _png_data_uri(color):
    """A tiny deterministic PNG data-URI (16x16, solid ``color``)."""
    from PIL import Image

    im = Image.new("RGB", (16, 16), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_preprocess_request_derives_stable_content_key(monkeypatch):
    """``_preprocess_request`` derives ``vision_feature_key`` from image
    *content* (not the temp-file path): the same image yields the same key
    across requests so the cache hits, and different content yields a
    different key. This exercises the real request path (not a manually
    supplied key)."""
    pytest.importorskip("mlx_vlm.vision_cache")
    import mlx_vlm.utils as _vlm_utils

    # Stub only the heavy processor call — we test key derivation, not
    # tokenization. ``_preprocess_request`` imports it as
    # ``from mlx_vlm.utils import prepare_inputs`` at call time, so patching
    # the module attribute takes effect.
    monkeypatch.setattr(
        _vlm_utils,
        "prepare_inputs",
        lambda *a, **k: {
            "input_ids": mx.array([1, 2, 3]),
            "pixel_values": mx.zeros((1, 3, 4, 4)),
        },
    )

    gen = MLLMBatchGenerator(
        model=_VisionCacheModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    assert gen._supports_vision_feature_cache is True

    def key_for(uri):
        req = MLLMBatchRequest(
            uid=0, request_id="r", prompt="p", images=[uri], max_tokens=8
        )
        gen._preprocess_request(req)
        return req.vision_feature_key

    uri_a = _png_data_uri((10, 20, 30))
    key_a1 = key_for(uri_a)
    key_a2 = key_for(_png_data_uri((10, 20, 30)))  # identical content
    key_b = key_for(_png_data_uri((200, 100, 50)))  # different content

    assert key_a1, "a request with an image must get a vision_feature_key"
    assert key_a1 == key_a2, "same image content must yield the same key"
    assert key_a1 != key_b, "different image content must yield a different key"


def test_preprocess_request_no_key_for_unsupported_model(monkeypatch):
    """An unsupported model leaves ``vision_feature_key`` None (the key is only
    computed when the feature is actually wired in)."""
    import mlx_vlm.utils as _vlm_utils

    monkeypatch.setattr(
        _vlm_utils,
        "prepare_inputs",
        lambda *a, **k: {
            "input_ids": mx.array([1, 2, 3]),
            "pixel_values": mx.zeros((1, 3, 4, 4)),
        },
    )
    gen = MLLMBatchGenerator(
        model=_RecordingModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=True,
    )
    assert gen._supports_vision_feature_cache is False

    req = MLLMBatchRequest(
        uid=0,
        request_id="r",
        prompt="p",
        images=[_png_data_uri((10, 20, 30))],
        max_tokens=8,
    )
    gen._preprocess_request(req)
    assert req.vision_feature_key is None


# --------------------------------------------------------------------------- #
# #1848 — a text-only prompt larger than ``prefill_step_size`` must not trip
# the per-batch cap. The cap exists to bound vision-merge memory (#682);
# text-only prompts are prefilled in chunks on the vision-encoding path and
# contribute no vision tokens, so they are exempt.
# --------------------------------------------------------------------------- #
def _make_vision_cap_request(uid: int, token_count: int) -> MLLMBatchRequest:
    """Like ``_make_cap_request`` but with a non-None ``pixel_values`` so it
    counts as a vision-bearing request for the per-batch cap."""
    req = _make_cap_request(uid=uid, token_count=token_count)
    req.pixel_values = mx.zeros((1, 3, 32, 32), dtype=mx.float32)
    req.image_grid_thw = mx.array([[2, 2, 1]], dtype=mx.int32)
    return req


def _text_request(uid: int, token_count: int) -> MLLMBatchRequest:
    """A plain text-only request (no pixel values / grid)."""
    return _make_cap_request(uid=uid, token_count=token_count)


def test_prefill_cap_exempts_large_text_only_prompt():
    """A >8k text-only prompt must NOT trip the per-batch cap (#1848).

    Pre-fix the inline check used ``prefill_step_size * len(requests)`` and
    rejected any single prompt longer than ``prefill_step_size`` (default
    8192), even though the chunked text-only prefill path could handle it.
    This is the regression the issue's DNF reported.
    """
    from vllm_mlx.mllm_batch_generator import _prefill_cap_violation

    req = _text_request(uid=0, token_count=20000)
    assert _prefill_cap_violation([req], prefill_step_size=8192) is None, (
        "a 20k-token text-only prompt must be exempt from the per-batch cap "
        "(it is prefilled in chunks and contributes no vision tokens)"
    )


def test_prefill_cap_still_fires_on_large_vision_request():
    """The cap must STILL bound vision-merge memory: an image request whose
    (text + vision) token count exceeds ``prefill_step_size × n_vision`` is
    rejected with the actionable message (#682 preserved).
    """
    from vllm_mlx.mllm_batch_generator import _prefill_cap_violation

    req = _make_vision_cap_request(uid=0, token_count=20000)
    msg = _prefill_cap_violation([req], prefill_step_size=8192)
    assert msg is not None, "a 20k-token vision request must trip the cap"
    assert "exceeds the per-batch cap" in msg
    assert "downscale the image" in msg


def test_prefill_cap_counts_only_vision_requests_in_budget():
    """In a mixed batch the cap budget scales with the number of *vision*
    requests; text-only requests are exempt and do not inflate the multiplier
    (#1848).
    """
    from vllm_mlx.mllm_batch_generator import _prefill_cap_violation

    # 1 vision request + 1 huge text-only request, cap = 8192 × 1 vision.
    batch = [
        _make_vision_cap_request(uid=0, token_count=8000),  # under vision cap
        _text_request(uid=1, token_count=9000),  # exempt, > 8192
    ]
    assert _prefill_cap_violation(batch, prefill_step_size=8192) is None, (
        "the text-only request must not push the vision budget over the cap"
    )

    # Same batch but now the vision request itself exceeds the cap.
    batch = [
        _make_vision_cap_request(uid=0, token_count=10000),  # exceeds 8192
        _text_request(uid=1, token_count=9000),
    ]
    assert _prefill_cap_violation(batch, prefill_step_size=8192) is not None, (
        "a vision request over the cap must still trip the cap in a mixed batch"
    )


def test_process_prompts_does_not_cap_large_text_only_prompt(monkeypatch):
    """End-to-end pin of the #1848 DNF through ``_process_prompts``.

    With the production MLLM default ``prefill_step_size=8192``, a single
    20000-token TEXT-ONLY prompt must NOT trip the per-batch cap. Pre-fix
    this raised ValueError("exceeds the per-batch cap") and the request
    DNF'd. It will still fail downstream (bare ``object()`` model cannot
    actually prefill), but that failure must NOT be the cap error.
    """
    gen = _gen_with_prefill_cap(prefill_step_size=8192)
    monkeypatch.setattr(gen, "_preprocess_request", lambda req: None)

    request = _text_request(uid=0, token_count=20000)

    with pytest.raises(Exception) as excinfo:  # noqa: BLE001 — see below
        MLLMBatchGenerator._process_prompts(gen, [request])

    err_msg = str(excinfo.value)
    assert "exceeds the per-batch cap" not in err_msg, (
        f"a 20k-token text-only prompt must pass the cap (then fail on the "
        f"bare-model prefill), but got the cap error: {err_msg}"
    )


def test_prefill_cap_exempts_batch_of_long_text_only_prompts():
    """A batch of MANY long text-only prompts must not trip the cap (#1848).

    Concurrency concern raised in review: with text-only now exempt, several
    >8k text prompts can batch together. Even though each is prefilled in
    chunks, they all still need full KV at cache-merge time, so we must not
    accidentally re-introduce a rejection for multi-long-text batches. The
    cap is vision-only by design; an all-text-only batch (any count, any
    length) must never violate it (#1848 / #682 contract preserved).
    """
    from vllm_mlx.mllm_batch_generator import _prefill_cap_violation

    # 4 concurrent long text-only prompts, all > prefill_step_size.
    batch = [_text_request(uid=i, token_count=20000) for i in range(4)]
    assert _prefill_cap_violation(batch, prefill_step_size=8192) is None, (
        "a batch of multiple 20k-token text-only prompts must stay exempt "
        "from the vision-only per-batch cap (no DNF / no cap rejection)"
    )

    # And when one long VISION request joins them, the cap correctly fires
    # only because of the vision request, not the text ones.
    mixed = [_text_request(uid=i, token_count=20000) for i in range(3)]
    mixed.append(_make_vision_cap_request(uid=3, token_count=10000))
    assert _prefill_cap_violation(mixed, prefill_step_size=8192) is not None, (
        "a 10k-token vision request among long text-only peers must still "
        "trip the cap (#682 preserved)"
    )
