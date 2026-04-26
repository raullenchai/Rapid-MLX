# SPDX-License-Identifier: Apache-2.0
"""
DFlash speculative-decoding engine.

Wraps the dflash MLX runtime (https://github.com/dflash/dflash) so that
`rapid-mlx serve <target> --drafter <drafter>` runs block-based draft+verify
with a separate drafter model that conditions on hidden states from selected
target layers.

This engine is single-request: only one prompt is generated at a time; further
requests wait on an asyncio.Lock. Continuous batching is not supported in this
mode.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..api.utils import clean_output_text
from .base import GenerationOutput
from .batched import BatchedEngine

logger = logging.getLogger(__name__)


def _init_dflash_step_thread() -> None:
    """Mirror engine_core._init_mlx_step_thread for our private executor."""
    import mlx.core as mx

    stream = mx.new_stream(mx.default_device())
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    logger.info("DFlash step thread initialized: stream=%s", stream)


@dataclass
class _ActiveRequest:
    started_at: float
    first_token_at: float | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    speculative_steps: int = 0
    acceptance_ratio: float = 0.0
    block_size: int = 0
    block_history: list[int] = field(default_factory=list)


class DFlashEngine(BatchedEngine):
    """Speculative-decoding engine using a separate drafter model."""

    def __init__(
        self,
        model_name: str,
        drafter_path: str,
        block_size: int | None = None,
        adaptive: bool = True,
        adaptive_min: int = 8,
        adaptive_max: int = 22,
        turboquant_bits: float | None = None,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        super().__init__(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            scheduler_config=None,
            stream_interval=1,
            force_mllm=False,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if self._is_mllm:
            raise ValueError(
                "DFlash mode is text-only; do not pass --mllm or use a multimodal model."
            )

        self._drafter_path = drafter_path
        self._drafter: Any = None
        self._block_size_override = block_size
        self._adaptive_enabled = bool(adaptive)
        self._adaptive_min = int(adaptive_min)
        self._adaptive_max = int(adaptive_max)
        self._turboquant_bits = turboquant_bits

        self._lock = asyncio.Lock()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._active: _ActiveRequest | None = None
        self._inflight = 0

        self._adaptive_cfg: Any = None
        self._current_block_size = 0
        self._observed_block_min = 0
        self._observed_block_max = 0

        self._lifetime_proposed = 0
        self._lifetime_accepted = 0
        self._lifetime_responses = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _start_llm(self) -> None:  # noqa: D401 — overrides BatchedEngine
        # Create the dedicated MLX worker thread BEFORE loading the model.
        # All model + drafter operations (including the initial weight load)
        # must run on this thread so that mlx-lm's per-thread generation
        # stream owns the arrays used during inference (PR 161 invariant).
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dflash-step",
            initializer=_init_dflash_step_thread,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_models_in_thread)
        # mark as not-engine-started (we don't run AsyncEngineCore)
        self._engine_started = False

    def _load_models_in_thread(self) -> None:
        """Runs on the dflash-step worker thread."""
        try:
            from dflash.model_mlx import (
                AdaptiveBlockSizeConfig,
                load_draft,
            )
        except ImportError as exc:
            raise RuntimeError(
                "DFlash mode requires the `dflash[mlx]` package. Install with:\n"
                "  pip install -e /Users/samuelfajreldines/dev/dflash[mlx]\n"
                f"Original error: {exc}"
            ) from exc

        from mlx_lm import load as mlx_load

        logger.info("[DFlash] Loading target model: %s", self._model_name)
        self._model, self._tokenizer = mlx_load(self._model_name)

        logger.info("[DFlash] Loading drafter: %s", self._drafter_path)
        self._drafter = load_draft(
            self._drafter_path,
            turboquant_bits=self._turboquant_bits,
        )
        self._drafter.bind(self._model)

        self._current_block_size = (
            int(self._block_size_override)
            if self._block_size_override is not None
            else int(self._drafter.config.block_size)
        )

        if self._adaptive_enabled:
            self._adaptive_cfg = AdaptiveBlockSizeConfig(
                enabled=True,
                min_block_size=self._adaptive_min,
                max_block_size=self._adaptive_max,
                grow_threshold=0.88,
                shrink_threshold=0.55,
                grow_streak=2,
                shrink_streak=2,
            )
        else:
            self._adaptive_cfg = None

        try:
            import mlx.core as mx

            if mx.metal.is_available():
                info = mx.device_info()
                max_rec = info.get(
                    "max_recommended_working_set_size",
                    info.get("memory_size", 0),
                )
                if max_rec > 0:
                    soft = int(max_rec * self._gpu_memory_utilization)
                    mx.set_memory_limit(soft)
                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)
        except Exception as exc:
            logger.warning("[DFlash] Failed to set Metal memory limits: %s", exc)

    async def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._drafter = None

    # ------------------------------------------------------------------
    # Generation core
    # ------------------------------------------------------------------

    async def _stream_dflash(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """Run dflash.stream_generate on the dflash-step thread, yielding GenerationResponse objects."""
        from dflash.model_mlx import stream_generate as _ds

        loop = asyncio.get_running_loop()
        executor = self._executor
        assert executor is not None, "DFlashEngine not started"

        def _make_gen():
            return _ds(
                self._model,
                self._drafter,
                self._tokenizer,
                prompt,
                block_size=self._block_size_override,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                adaptive_block_size=self._adaptive_cfg,
            )

        gen = await loop.run_in_executor(executor, _make_gen)

        sentinel = object()

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return sentinel

        while True:
            resp = await loop.run_in_executor(executor, _next)
            if resp is sentinel:
                return
            yield resp

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        if images or videos:
            raise ValueError("DFlash mode does not support images or videos.")

        self._inflight += 1
        try:
            async with self._lock:
                self._track_request_start()
                last_resp = None
                full_text_parts: list[str] = []
                try:
                    async for resp in self._stream_dflash(
                        prompt, max_tokens, temperature, top_p
                    ):
                        last_resp = resp
                        if resp.text:
                            full_text_parts.append(resp.text)
                        self._update_active(resp, new_text=resp.text or "")
                finally:
                    self._track_request_end()

                return GenerationOutput(
                    text=clean_output_text("".join(full_text_parts)),
                    prompt_tokens=last_resp.prompt_tokens if last_resp else 0,
                    completion_tokens=last_resp.generation_tokens if last_resp else 0,
                    finished=True,
                    finish_reason=(last_resp.finish_reason if last_resp else None) or "stop",
                )
        finally:
            self._inflight -= 1

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        if images or videos:
            raise ValueError("DFlash mode does not support images or videos.")

        self._inflight += 1
        try:
            async with self._lock:
                self._track_request_start()
                cumulative = ""
                last_resp = None
                try:
                    async for resp in self._stream_dflash(
                        prompt, max_tokens, temperature, top_p
                    ):
                        last_resp = resp
                        new_text = resp.text or ""
                        cumulative += new_text
                        self._update_active(resp, new_text=new_text)

                        yield GenerationOutput(
                            text=clean_output_text(cumulative),
                            new_text=new_text,
                            tokens=list(resp.tokens) if resp.tokens else [],
                            prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.generation_tokens,
                            finished=bool(resp.finish_reason),
                            finish_reason=resp.finish_reason,
                        )
                finally:
                    self._track_request_end()

                # Some clients expect a final yield with finished=True
                if last_resp is not None and not last_resp.finish_reason:
                    yield GenerationOutput(
                        text=clean_output_text(cumulative),
                        new_text="",
                        tokens=[],
                        prompt_tokens=last_resp.prompt_tokens,
                        completion_tokens=last_resp.generation_tokens,
                        finished=True,
                        finish_reason="stop",
                    )
        finally:
            self._inflight -= 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _track_request_start(self) -> None:
        self._active = _ActiveRequest(
            started_at=time.time(), block_size=self._current_block_size
        )

    def _track_request_end(self) -> None:
        if self._active is not None:
            self._lifetime_responses += 1
            self._lifetime_proposed += self._active.proposed_tokens
            self._lifetime_accepted += self._active.accepted_tokens
        self._active = None

    def _update_active(self, resp: Any, new_text: str = "") -> None:
        a = self._active
        if a is None:
            return
        if a.first_token_at is None and (new_text or resp.tokens):
            a.first_token_at = time.time()
        a.prompt_tokens = int(resp.prompt_tokens or 0)
        a.generated_tokens = int(resp.generation_tokens or 0)
        a.proposed_tokens = int(resp.proposed_tokens or 0)
        a.accepted_tokens = int(resp.accepted_tokens or 0)
        a.speculative_steps = int(resp.speculative_steps or 0)
        a.acceptance_ratio = float(resp.avg_acceptance_ratio or 0.0)
        history = list(resp.block_size_history or ())
        if history:
            a.block_history = history
            a.block_size = int(history[-1])
            self._current_block_size = a.block_size
            mn = min(history)
            mx_ = max(history)
            self._observed_block_min = (
                mn if self._observed_block_min == 0 else min(self._observed_block_min, mn)
            )
            self._observed_block_max = max(self._observed_block_max, mx_)

    def get_stats(self) -> dict[str, Any]:
        try:
            import mlx.core as mx

            active_mem_gb = mx.get_active_memory() / 1e9
            peak_mem_gb = mx.get_peak_memory() / 1e9
            cache_mem_gb = mx.get_cache_memory() / 1e9
        except Exception:
            active_mem_gb = peak_mem_gb = cache_mem_gb = 0.0

        running_requests: list[dict[str, Any]] = []
        if self._active is not None:
            now = time.time()
            elapsed = now - self._active.started_at
            ttft = (
                (self._active.first_token_at - self._active.started_at)
                if self._active.first_token_at
                else None
            )
            tps = None
            if (
                self._active.first_token_at
                and self._active.generated_tokens > 0
            ):
                window = now - self._active.first_token_at
                if window > 0.01:
                    tps = self._active.generated_tokens / window
            running_requests.append(
                {
                    "request_id": "dflash-active",
                    "status": "running",
                    "phase": (
                        "generation" if self._active.first_token_at else "prefill"
                    ),
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": self._active.prompt_tokens,
                    "completion_tokens": self._active.generated_tokens,
                    "max_tokens": 0,
                    "tokens_per_second": tps,
                    "ttft_s": ttft,
                    "acceptance_ratio": self._active.acceptance_ratio,
                    "block_size": self._active.block_size,
                    "speculative_steps": self._active.speculative_steps,
                    "accepted_tokens": self._active.accepted_tokens,
                    "proposed_tokens": self._active.proposed_tokens,
                    "cache_hit_type": None,
                    "cached_tokens": 0,
                    "progress": 0.0,
                }
            )

        lifetime_ratio = (
            (self._lifetime_accepted / self._lifetime_proposed)
            if self._lifetime_proposed > 0
            else 0.0
        )

        num_running = 1 if self._active is not None else 0
        num_waiting = max(0, self._inflight - num_running)

        return {
            "engine_type": "dflash",
            "model_name": self._model_name,
            "is_mllm": False,
            "loaded": self._loaded,
            "running": self._active is not None,
            "uptime_seconds": time.time() - self._start_time,
            "num_running": num_running,
            "num_waiting": num_waiting,
            "total_requests_processed": self._lifetime_responses,
            "metal_active_memory_gb": active_mem_gb,
            "metal_peak_memory_gb": peak_mem_gb,
            "metal_cache_memory_gb": cache_mem_gb,
            "requests": running_requests,
            "dflash": {
                "lifetime_acceptance_ratio": lifetime_ratio,
                "current_block_size": self._current_block_size,
                "adaptive_enabled": self._adaptive_enabled,
                "adaptive_min": self._adaptive_min,
                "adaptive_max": self._adaptive_max,
                "observed_block_min": self._observed_block_min,
                "observed_block_max": self._observed_block_max,
            },
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None
