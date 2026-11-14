# SPDX-License-Identifier: Apache-2.0
"""Request-local prompt scoring; no vocabulary distributions survive prefill."""

from dataclasses import dataclass, field
from typing import Any

# Bound both the model's vocabulary projection and float32 normalization.
PROMPT_LOGPROBS_CHUNK_SIZE = 128


@dataclass
class PromptLogprobs:
    top_k: int
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float | None] = field(default_factory=list)
    top_logprobs: list[dict[int, float] | None] = field(default_factory=list)


def prefill_prompt_logprobs(model: Any, tokens: list[int], result: PromptLogprobs):
    """Score each token from its predecessor and return private KV for decode.

    The final prompt token remains for the normal generation prefill. Chunk
    boundaries include the following target token, so no score is lost there.
    Evaluation before the next forward keeps lazy graphs bounded as well.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    result.token_ids = list(tokens)
    result.logprobs = [None] if tokens else []
    result.top_logprobs = [None] if tokens else []
    for start in range(0, len(tokens) - 1, PROMPT_LOGPROBS_CHUNK_SIZE):
        end = min(start + PROMPT_LOGPROBS_CHUNK_SIZE, len(tokens) - 1)
        logits = model(mx.array([tokens[start:end]]), cache=cache)[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(tokens[start + 1 : end + 1])
        actual = mx.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
        k = min(result.top_k, logprobs.shape[-1])
        if k:
            indices = mx.argpartition(logprobs, -k, axis=-1)[:, -k:]
            values = mx.take_along_axis(logprobs, indices, axis=-1)
            mx.eval(actual, indices, values, [c.state for c in cache])
            for ids, scores in zip(indices.tolist(), values.tolist()):
                result.top_logprobs.append(
                    dict(sorted(zip(ids, scores), key=lambda item: -item[1]))
                )
        else:
            mx.eval(actual, [c.state for c in cache])
            result.top_logprobs.extend({} for _ in range(end - start))
        result.logprobs.extend(actual.tolist())
        del logits, logprobs, actual
    return cache


def legacy_prompt_logprobs(prompt: str, tokenizer: Any, result: PromptLogprobs):
    """Map scores to original text, including special and split Unicode tokens."""
    ids = result.token_ids
    if not ids:
        return {
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "text_offset": [],
        }
    base = getattr(tokenizer, "_tokenizer", tokenizer)
    starts = None
    try:
        encoded = base(prompt, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == ids:
            starts = [int(pair[0]) for pair in encoded["offset_mapping"]]
    except (TypeError, AttributeError, KeyError, NotImplementedError):
        pass
    if starts is None:
        # Prefix decoding avoids treating a partial UTF-8 byte as a character.
        starts = [0]
        for i in range(1, len(ids)):
            prefix = tokenizer.decode(ids[:i]).rstrip("\ufffd")
            starts.append(min(len(prompt), len(prefix)))
    starts = [min(len(prompt), max(0, x)) for x in starts]
    # Byte fallback tokens may share a character offset. Give that character
    # to the last piece, leaving zero-width entries for preceding byte pieces.
    ends = starts[1:] + [len(prompt)] if ids else []
    return {
        "tokens": [prompt[a:b] for a, b in zip(starts, ends)],
        "token_logprobs": result.logprobs,
        "top_logprobs": [
            None
            if top is None
            else {tokenizer.decode([tid]): value for tid, value in top.items()}
            for top in result.top_logprobs
        ],
        "text_offset": starts,
    }
