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
import mlx.nn as nn
import pytest

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


def _gen_with_prefill_cap(prefill_step_size: int) -> MLLMBatchGenerator:
    """Generator with a tunable cap, no real model/processor needed.

    ``_process_prompts`` only reads ``self.prefill_step_size`` /
    ``self._stats`` / ``self.vision_cache`` before raising the cap error,
    so a bare construction is enough to exercise the check.
    """
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.prefill_step_size = prefill_step_size
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


def test_mllm_scheduler_config_default_prefill_step_size_covers_screenshot():
    """``MLLMSchedulerConfig.prefill_step_size`` default must cover a
    typical 1920×1080 screenshot's vision-token count.

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
    assert cfg.prefill_step_size >= 8192, (
        f"MLLMSchedulerConfig.prefill_step_size default ({cfg.prefill_step_size}) "
        f"must be at least 8192 to cover 1920×1080 screenshots without "
        f"tripping the per-batch cap (#682)."
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

    # 500-token request, cap = 100 × 1 = 100 ⇒ 500 > 100 ⇒ raises.
    request = _make_cap_request(uid=0, token_count=500)

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

    # 2292 tokens — typical Qwen3-VL token count for a 1920×1080 image.
    request = _make_cap_request(uid=0, token_count=2292)

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
