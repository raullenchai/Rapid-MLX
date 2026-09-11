"""Minimal greedy generation for DeepSeek-V4.1 MLX.

The prefill is chunked (default 512 tokens) with an eval between chunks — a
single prefill graph over a 400+ GB build can exceed Metal's command-buffer
watchdog. Each chunk/step snapshots the cache first (MLX setitem mutates
through aliases, but ``x[:]`` pins the pre-step node, so snapshots are free)
and retries once on the CPU stream if the GPU deadline is hit — the pattern
from qwen38-mlx's quantize_stream.
"""

from __future__ import annotations

import mlx.core as mx

from .model import Model


def _cache_snapshot(cache):
    layers = []
    for lc in cache.layers:
        layers.append(
            (
                lc.win_kv[:],
                lc.comp_kv[:] if lc.comp_kv is not None else None,
                lc.index_k[:] if lc.index_k is not None else None,
                (lc.comp_state.kv_state[:], lc.comp_state.score_state[:])
                if lc.comp_state is not None
                else None,
            )
        )
    eng = cache.engram_ids.copy() if cache.engram_ids is not None else None
    return cache.offset, layers, eng


def _cache_restore(cache, snap):
    offset, layers, eng = snap
    cache.offset = offset
    for lc, (win, comp, idxk, cst) in zip(cache.layers, layers):
        lc.win_kv = win
        if comp is not None:
            lc.comp_kv = comp
        if idxk is not None:
            lc.index_k = idxk
        if cst is not None:
            lc.comp_state.kv_state, lc.comp_state.score_state = cst
    if eng is not None:
        cache.engram_ids[:] = eng


def _forward(model, ids, cache):
    """One forward with eval; on a Metal timeout, roll the cache back and
    recompute the same chunk on the CPU stream (last resort, ~20x slower)."""
    snap = _cache_snapshot(cache)
    try:
        lg = model(ids, cache, last_logit_only=True)
        mx.eval(lg)
        return lg
    except RuntimeError as err:
        # After one Metal timeout the process's further GPU submissions are
        # ignored (SubmissionsIgnored), so the only in-process retry that can
        # work is the CPU stream — slow, a true last resort.
        if "Timeout" not in str(err) and "Ignored" not in str(err):
            raise
        print("(metal timeout — recomputing this chunk on the CPU stream)", flush=True)
        _cache_restore(cache, snap)
        mx.clear_cache()
        with mx.stream(mx.cpu):
            lg = model(ids, cache, last_logit_only=True)
            mx.eval(lg)
        return lg


def greedy_generate(
    model: Model,
    input_ids,
    max_new_tokens: int = 64,
    max_seq_len: int | None = None,
    eos_id: int = 1,
    dtype=mx.float32,
    prefill_chunk: int = 512,
):
    """input_ids: list[int] or [1, n] array. Returns the generated ids."""
    ids = mx.array([input_ids] if not hasattr(input_ids[0], "__len__") else input_ids)
    total = ids.shape[1] + max_new_tokens
    cache = model.make_cache(
        bsz=ids.shape[0], max_seq_len=max_seq_len or (total + 8), dtype=dtype
    )
    logits = None
    for a in range(0, ids.shape[1], prefill_chunk):
        logits = _forward(model, ids[:, a : a + prefill_chunk], cache)
    if logits is None:
        raise ValueError("input_ids must contain at least one token")
    out = []
    tok = mx.argmax(logits[:, -1], axis=-1)
    for _ in range(max_new_tokens):
        t = int(tok[0])
        if t == eos_id:
            break
        out.append(t)
        logits = _forward(model, tok[:, None], cache)
        tok = mx.argmax(logits[:, -1], axis=-1)
    return out


def load_tokenizer(path: str):
    import os

    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))
