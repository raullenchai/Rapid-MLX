# SPDX-License-Identifier: Apache-2.0
"""Default decode strategy — wraps mlx-lm BatchGenerator public API.

This is the standard decode implementation that works with ALL models
supported by mlx-lm. No monkey-patching, no internal API access.

Usage::

    from vllm_mlx.pipeline.decode import StandardDecode

    decode = StandardDecode(model, tokenizer, sampler, prefill_step_size=2048)
    uid = decode.insert(DecodeRequest(uid=0, tokens=[...], max_tokens=100))
    while decode.has_active():
        for result in decode.step():
            print(result.token, result.finish_reason)
    decode.close()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from mlx_lm.generate import BatchGenerator

from .interfaces import DecodeRequest, DecodeStrategy, TokenResult

logger = logging.getLogger(__name__)


class StandardDecode(DecodeStrategy):
    """Decode strategy using mlx-lm's BatchGenerator public API.

    Uses only insert()/next()/remove()/close() — no monkey-patching.
    Works with all model architectures (Transformer, Mamba, Gemma 4, etc.).
    """

    def __init__(
        self,
        model: Any,
        default_sampler: Callable | None = None,
        max_tokens: int = 4096,
        stop_tokens: set[int] | None = None,
        prefill_batch_size: int = 8,
        completion_batch_size: int = 32,
        prefill_step_size: int = 2048,
    ):
        self._model = model
        self._default_sampler = default_sampler or (lambda x: x.argmax(-1))
        self._stop_tokens = stop_tokens

        self._bg = BatchGenerator(
            model=model,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            sampler=self._default_sampler,
            prefill_batch_size=prefill_batch_size,
            completion_batch_size=completion_batch_size,
            prefill_step_size=prefill_step_size,
        )

        # Track active UIDs for has_active()
        self._active_uids: set[int] = set()
        self._uid_counter = 0

    def insert(self, request: DecodeRequest) -> int:
        """Insert a request into the batch generator."""
        uid = request.uid if request.uid >= 0 else self._uid_counter
        self._uid_counter = max(self._uid_counter, uid + 1)

        sampler = request.sampler or self._default_sampler
        self._bg.insert(
            [request.tokens],
            max_tokens=[request.max_tokens],
            caches=[request.cache] if request.cache else None,
            samplers=[sampler],
            logits_processors=(
                [request.logits_processors] if request.logits_processors else None
            ),
        )
        self._active_uids.add(uid)
        return uid

    def step(self) -> list[TokenResult]:
        """Run one generation step via BatchGenerator.next().

        Returns TokenResult for each active sequence that produced a token.
        Prompt-processing responses (prefill progress) are silently consumed.
        """
        if not self._active_uids:
            return []

        raw = self._bg.next()

        # mlx-lm 0.31+ returns (prompt_responses, generation_responses)
        if isinstance(raw, tuple):
            _, gen_responses = raw
        else:
            gen_responses = raw

        results = []
        for r in gen_responses:
            result = TokenResult(
                uid=r.uid,
                token=r.token,
                logprobs=r.logprobs,
                finish_reason=r.finish_reason,
                prompt_cache=r.prompt_cache if r.finish_reason else None,
            )
            results.append(result)

            if r.finish_reason:
                self._active_uids.discard(r.uid)

        return results

    def remove(self, uid: int) -> Any | None:
        """Remove a sequence from the batch generator."""
        self._active_uids.discard(uid)
        try:
            caches = self._bg.remove([uid], return_prompt_caches=True)
            return caches[0] if caches else None
        except Exception:
            try:
                self._bg.remove([uid])
            except Exception:
                pass
            return None

    def has_active(self) -> bool:
        """Whether there are sequences actively being decoded."""
        return bool(self._active_uids)

    def close(self) -> None:
        """Release BatchGenerator resources."""
        self._bg.close()
        self._active_uids.clear()
