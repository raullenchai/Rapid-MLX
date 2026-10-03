# SPDX-License-Identifier: Apache-2.0
"""DRY ("Don't Repeat Yourself") repetition penalty.

The sampler roleplay front ends (SillyTavern, KoboldCpp, text-generation-
webui) use against verbatim loops. For every candidate next token ``t`` it
finds the longest stretch of context that, ending at the current position,
also occurred earlier and was then followed by ``t``. If that stretch is at
least ``allowed_length`` tokens long, ``t``'s logit is reduced by

    multiplier * base ** (match_length - allowed_length)

so extending a long verbatim repeat becomes exponentially unlikely while
short, natural repeats are untouched. A *sequence breaker* (newline, ``:``,
quotes, ``*`` by default) ends any match, so dialogue scaffolding such as
``Name:`` never counts as repetition.

Semantics follow text-generation-webui's reference implementation:
``multiplier`` 0 disables it, ``penalty_last_n`` 0 means the whole context,
and each breaker string contributes the token it ends with when it follows
other text. The history is the request's full prompt plus its committed
output — never the KV-cache view handed to logits processors, which shrinks
on a prefix-cache hit — so a warm and a cold request are penalized alike.

The core works on plain Python ints so it is testable without MLX.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_BASE = 1.75
DEFAULT_ALLOWED_LENGTH = 2
DEFAULT_SEQUENCE_BREAKERS = ("\n", ":", '"', "*")
MAX_MATCH_LENGTH = 50
# A penalty this large already removes the token from any sampler, so a
# match need not be measured past the length that reaches it. The upstream
# algorithm also caps a match at 50 tokens. We extend that cap only when the
# caller's accepted ``allowed_length`` is higher (bounded at 100 by the API),
# so every valid threshold remains meaningful. Keeping a hard work bound is
# essential here: ``base`` is client controlled and can be arbitrarily close
# to 1, where a saturation-only cap would otherwise make the nested scan
# quadratic in the entire context on a repeated-token prompt.
SATURATING_PENALTY = 1e4
_MAX_LOG_PENALTY = 700.0  # math.exp overflow guard


def _max_useful_length(multiplier: float, base: float, allowed_length: int) -> int:
    if base <= 1.0 or multiplier >= SATURATING_PENALTY:
        return allowed_length
    extra = math.log(SATURATING_PENALTY / multiplier) / math.log(base)
    hard_cap = max(MAX_MATCH_LENGTH, allowed_length)
    return min(hard_cap, allowed_length + max(0, math.ceil(extra)))


def dry_penalties(
    tokens: Sequence[int],
    *,
    multiplier: float,
    base: float,
    allowed_length: int,
    breakers: frozenset[int],
    penalty_last_n: int = 0,
) -> dict[int, float]:
    """``{token_id: logit penalty}`` for the next step of ``tokens``."""
    if multiplier <= 0:
        return {}
    if penalty_last_n > 0:
        tokens = tokens[-penalty_last_n:]
    n = len(tokens)
    if n < 2:
        return {}
    last = tokens[-1]
    if last in breakers:
        return {}
    cap = _max_useful_length(multiplier, base, allowed_length)
    longest: dict[int, int] = {}
    for idx in range(n - 1):
        if tokens[idx] != last:
            continue
        following = tokens[idx + 1]
        if following in breakers:
            continue
        length = 1
        while length < cap:
            earlier = idx - length
            if earlier < 0:
                break
            token = tokens[earlier]
            if token in breakers or token != tokens[n - 1 - length]:
                break
            length += 1
        if length > longest.get(following, 0):
            longest[following] = length
    log_base = math.log(base)
    return {
        token: multiplier
        * math.exp(min((length - allowed_length) * log_base, _MAX_LOG_PENALTY))
        for token, length in longest.items()
        if length >= allowed_length
    }


def breaker_token_ids(tokenizer: Any, breakers: Iterable[str]) -> frozenset[int]:
    """Token ids that end a DRY match, one per breaker string.

    Like text-generation-webui, each breaker is encoded after a letter
    (``"a" + breaker``) and contributes its final token, so a breaker is
    identified by the token it produces in running text rather than by every
    fragment of a standalone encoding.
    """
    ids: set[int] = set()
    for text in breakers:
        encoded = _encode(tokenizer, f"a{text}")
        if len(encoded) < 2:
            encoded = _encode(tokenizer, text)
        if encoded:
            ids.add(int(encoded[-1]))
    return frozenset(ids)


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


class DRYLogitsProcessor:
    """DRY settings for one request; bound to it with :meth:`bind`."""

    def __init__(
        self,
        *,
        multiplier: float,
        base: float,
        allowed_length: int,
        breakers: frozenset[int],
        penalty_last_n: int = 0,
    ) -> None:
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.breakers = breakers
        self.penalty_last_n = penalty_last_n

    def penalties(self, history: Sequence[int]) -> dict[int, float]:
        return dry_penalties(
            history,
            multiplier=self.multiplier,
            base=self.base,
            allowed_length=self.allowed_length,
            breakers=self.breakers,
            penalty_last_n=self.penalty_last_n,
        )

    def bind(self, request: Any) -> BoundDRYProcessor:
        return BoundDRYProcessor(self, request)


class BoundDRYProcessor:
    """mlx-lm logits processor ``(tokens, logits) -> logits`` for one request.

    Reads the request's full prompt and committed output at call time and
    ignores the ``tokens`` argument (the KV-cache view, which a prefix-cache
    hit truncates).
    """

    def __init__(self, settings: DRYLogitsProcessor, request: Any) -> None:
        self.settings = settings
        self.request = request

    def history(self) -> list[int]:
        prompt = getattr(self.request, "prompt_token_ids", None) or []
        output = getattr(self.request, "output_token_ids", None) or []
        window = self.settings.penalty_last_n
        if window > 0 and len(output) >= window:
            return list(output[-window:])
        if window > 0:
            prompt = prompt[-(window - len(output)) :]
        return [*prompt, *output]

    def __call__(self, _tokens: Any, logits: Any) -> Any:
        penalties = self.settings.penalties(self.history())
        if not penalties:
            return logits
        import mlx.core as mx

        ids = mx.array(list(penalties))
        amounts = mx.array(list(penalties.values()), dtype=logits.dtype)
        logits[:, ids] = logits[:, ids] - amounts
        return logits
