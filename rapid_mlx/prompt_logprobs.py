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


@dataclass
class PromptScoringState:
    """Private KV and progress retained only until scoring ends or aborts."""

    cache: Any
    result: PromptLogprobs
    position: int = 0

    @classmethod
    def create(cls, model: Any, tokens: list[int], result: PromptLogprobs):
        from mlx_lm.models.cache import make_prompt_cache

        result.token_ids = list(tokens)
        result.logprobs = [None] if tokens else []
        result.top_logprobs = [None] if tokens else []
        return cls(make_prompt_cache(model), result)

    def advance(self, model: Any, tokens: list[int], token_budget: int) -> int:
        """Score a bounded chunk, leaving the final prompt token for decode.

        Include the following target at chunk boundaries. Materialize scores
        and KV before returning so lazy graphs never span scheduler steps.
        """
        import mlx.core as mx

        start = self.position
        end = min(
            start + min(PROMPT_LOGPROBS_CHUNK_SIZE, token_budget), len(tokens) - 1
        )
        if end <= start:
            return 0
        logits = model(mx.array([tokens[start:end]]), cache=self.cache)[0].astype(
            mx.float32
        )
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(tokens[start + 1 : end + 1])
        actual = mx.take_along_axis(logprobs, targets[:, None], axis=-1)[:, 0]
        k = min(self.result.top_k, logprobs.shape[-1])
        if k:
            indices = mx.argpartition(logprobs, -k, axis=-1)[:, -k:]
            values = mx.take_along_axis(logprobs, indices, axis=-1)
            mx.eval(actual, indices, values, [c.state for c in self.cache])
            for ids, scores in zip(indices.tolist(), values.tolist()):
                self.result.top_logprobs.append(
                    dict(sorted(zip(ids, scores), key=lambda item: -item[1]))
                )
        else:
            mx.eval(actual, [c.state for c in self.cache])
            self.result.top_logprobs.extend({} for _ in range(end - start))
        self.result.logprobs.extend(actual.tolist())
        del logits, logprobs, actual
        self.position = end
        return end - start


def prefill_prompt_logprobs(model: Any, tokens: list[int], result: PromptLogprobs):
    """Score synchronously for callers outside the scheduler."""
    state = PromptScoringState.create(model, tokens, result)
    while state.advance(model, tokens, PROMPT_LOGPROBS_CHUNK_SIZE):
        pass
    return state.cache


def _incremental_ends(prompt, tokenizer, ids, special):
    """Decode incrementally, bounding the naive decoder's retained context."""
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

    # Use TokenizerWrapper's streaming factory; a stored detokenizer may be
    # shared with generation or another prompt-offset build in a worker thread.
    factory = getattr(tokenizer, "_detokenizer_class", NaiveStreamingDetokenizer)
    detok = factory(tokenizer)
    detok.reset()
    naive = isinstance(detok, NaiveStreamingDetokenizer)
    ends, context, pieces = [], [], []
    position = previous = 0
    for tid, is_special in zip(ids, special):
        if not is_special:
            detok.add_token(tid)
            context.append(tid)
            text = detok.text
            # Wait for complete UTF-8 before assigning a character to a token.
            length = len(text.rstrip("\ufffd"))
            piece = text[previous:length]
            pieces.append(piece)
            position += len(piece)
            previous = length
            if (
                naive
                and len(context) >= 32
                and (not text.endswith("\ufffd") or len(context) >= 64)
            ):
                # The generic decoder otherwise re-decodes the entire line.
                # Keep a small overlap for whitespace/word-boundary context;
                # even an arbitrarily long line has bounded decode work.
                context = context[-8:]
                detok.reset()
                for token in context:
                    detok.add_token(token)
                previous = len(detok.text.rstrip("\ufffd"))
        ends.append(position)
    detok.finalize()
    if "".join(pieces) != prompt:
        raise ValueError("Tokenizer cannot align prompt token offsets exactly.")
    return ends


def prompt_token_spans(prompt: str, tokenizer: Any, ids: list[int]):
    """Find faithful spans for scored IDs, or reject unavailable alignment.

    Use the tokenizer's source offsets when available. Without them the
    incremental decoded pieces must reproduce the submitted prompt exactly.
    Inserted special tokens retain zero-width spans at their sequence positions.
    """
    base = getattr(tokenizer, "_tokenizer", tokenizer)
    offsets = None
    special = None
    try:
        encoded = base(
            prompt, return_offsets_mapping=True, return_special_tokens_mask=True
        )
        if list(encoded["input_ids"]) == ids:
            offsets = encoded["offset_mapping"]
            special = encoded.get("special_tokens_mask")
            if len(offsets) != len(ids):
                offsets = None
    except (TypeError, AttributeError, KeyError, NotImplementedError):
        pass
    if special is None:
        # Slow tokenizers may provide the insertion mask without offsets.
        try:
            encoded = base(prompt, return_special_tokens_mask=True)
            if list(encoded["input_ids"]) == ids:
                special = encoded["special_tokens_mask"]
        except (TypeError, AttributeError, KeyError, NotImplementedError):
            pass
    if special is None or len(special) != len(ids):
        special_ids = set(getattr(base, "all_special_ids", []))
        special = [
            tid in special_ids and (offsets is None or tuple(offsets[i]) == (0, 0))
            for i, tid in enumerate(ids)
        ]
    content = [i for i, added in enumerate(special) if not added]
    if offsets is None:
        ends = _incremental_ends(prompt, tokenizer, ids, special)
    else:
        ends = [0] * len(ids)
        for index, i in enumerate(content):
            # Shared starts belong to the last byte piece of a character.
            # Before an inserted special token use the preceding span's END,
            # never its (possibly zero) offset or the special token's (0, 0).
            next_i = content[index + 1] if index + 1 < len(content) else len(ids)
            ends[i] = (
                int(offsets[next_i][0] if next_i == i + 1 else offsets[i][1])
                if next_i < len(ids)
                else len(prompt)
            )
    if content:
        ends[content[-1]] = len(prompt)
    starts = []
    position = 0
    for i, added in enumerate(special):
        starts.append(position)
        position = position if added else min(len(prompt), max(position, ends[i]))
        ends[i] = position
    return starts, ends


def legacy_prompt_logprobs(prompt: str, tokenizer: Any, result: PromptLogprobs):
    """Partition the submitted prompt over the actual scored input IDs.

    The completions scheduler uses encode(prompt) with default special-token
    insertion. Keep those BOS/EOS scores as zero-width entries at their sequence
    positions: echo must reproduce the client's prompt, not decoded BOS/EOS
    spellings. A literal special token in the prompt still occupies its span
    (the tokenizer's insertion mask distinguishes it from added tokens).
    """
    ids = result.token_ids
    if not ids:
        return {
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "text_offset": [],
        }
    starts, ends = prompt_token_spans(prompt, tokenizer, ids)
    return {
        "tokens": [prompt[a:b] for a, b in zip(starts, ends)],
        "token_logprobs": result.logprobs,
        "top_logprobs": [
            None if top is None else _decoded_top_logprobs(tokenizer, top)
            for top in result.top_logprobs
        ],
        "text_offset": starts,
    }


def _decoded_top_logprobs(tokenizer, top):
    decoded = {}
    for tid, value in top.items():
        token = tokenizer.decode([tid])
        decoded[token] = max(decoded.get(token, float("-inf")), value)
    return decoded
