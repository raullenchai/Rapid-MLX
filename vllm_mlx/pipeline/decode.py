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

        # Check if BatchGenerator.remove supports return_prompt_caches
        import inspect

        sig = inspect.signature(self._bg.remove)
        self._supports_cache_return = "return_prompt_caches" in sig.parameters

    def insert(self, request: DecodeRequest) -> int:
        """Insert a request into the batch generator.

        Returns the UID assigned by BatchGenerator (use this, not request.uid).
        """
        sampler = request.sampler or self._default_sampler
        uids = self._bg.insert(
            [request.tokens],
            max_tokens=[request.max_tokens],
            caches=[request.cache] if request.cache else None,
            samplers=[sampler],
            logits_processors=(
                [request.logits_processors] if request.logits_processors else None
            ),
        )
        uid = uids[0]
        self._active_uids.add(uid)
        return uid

    def step(self) -> list[TokenResult]:
        """Run one generation step via BatchGenerator.next().

        Returns TokenResult for each active sequence that produced a token.
        Prompt-processing responses (prefill progress) are silently consumed.
        """
        if not self._active_uids:
            return []

        # BatchGenerator.next() returns (prompt_responses, generation_responses).
        # prompt_responses carry prefill progress (silently consumed here).
        # gen_responses carry actual tokens. Empty during prefill.
        _, gen_responses = self._bg.next()

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
        """Remove a sequence from the batch generator.

        Returns extracted prompt cache if the BatchGenerator supports it.
        """
        self._active_uids.discard(uid)
        if self._supports_cache_return:
            caches = self._bg.remove([uid], return_prompt_caches=True)
            return caches[0] if caches else None
        self._bg.remove([uid])
        return None

    def has_active(self) -> bool:
        """Whether there are sequences actively being decoded."""
        return bool(self._active_uids)

    def close(self) -> None:
        """Release BatchGenerator resources."""
        self._bg.close()
        self._active_uids.clear()
