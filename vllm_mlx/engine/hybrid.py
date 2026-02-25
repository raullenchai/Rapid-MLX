# SPDX-License-Identifier: Apache-2.0
"""
Hybrid engine combining speculative decoding with continuous batching.

This engine shares a single model instance between SimpleEngine and BatchedEngine,
saving ~44GB RAM while providing:
- Speculative decoding (80+ tok/s) for single-user scenarios
- Continuous batching (60-70 tok/s) for multi-user scenarios

The engine automatically switches between modes based on concurrent request count.

Architecture:
    HybridEngine
    ├── Shared components (loaded once)
    │   ├── _shared_model (44GB)
    │   ├── _shared_tokenizer
    │   └── _ownership_lock
    │
    ├── SimpleEngine (speculative)
    │   └── + draft_model (350MB)
    │
    └── BatchedEngine (batching)
        └── Scheduler + BatchGenerator

Mode switching logic:
    if active_requests < switch_threshold:
        → SimpleEngine (speculative, 80+ tok/s)
    else:
        → BatchedEngine (batching, 60-70 tok/s)
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from mlx_lm import load

from .base import BaseEngine, GenerationOutput
from .batched import BatchedEngine
from .simple import SimpleEngine
from ..model_registry import get_registry

logger = logging.getLogger(__name__)


class HybridEngine(BaseEngine):
    """
    Hybrid engine: speculative decoding for single-user,
    continuous batching for multi-user. Shares model instance.

    This engine provides the best of both worlds:
    - When serving a single user: uses SimpleEngine with speculative decoding
      for maximum throughput (80+ tok/s with Qwen3-0.6B draft model)
    - When serving multiple concurrent users: switches to BatchedEngine
      for efficient continuous batching

    The key innovation is sharing a single model instance (~44GB for Qwen3-Next-80B)
    between both engines, cutting memory usage in half compared to running
    separate servers.
    """

    def __init__(
        self,
        model_name: str,
        draft_model: str | None = None,
        num_draft_tokens: int = 5,
        scheduler_config: Any | None = None,
        stream_interval: int = 1,
        trust_remote_code: bool = True,
        force_mllm: bool = False,
        switch_threshold: int = 2,
    ):
        """
        Initialize the hybrid engine.

        Args:
            model_name: HuggingFace model name or local path
            draft_model: Draft model for speculative decoding (e.g., Qwen3-0.6B-4bit)
            num_draft_tokens: Number of tokens to generate speculatively (default: 5)
            scheduler_config: Scheduler config for batched mode
            stream_interval: Tokens to batch before streaming (batched mode only)
            trust_remote_code: Whether to trust remote code
            force_mllm: Force loading as MLLM even if not auto-detected
            switch_threshold: Number of concurrent requests to trigger batch mode (default: 2)
        """
        self._model_name = model_name
        self._draft_model_name = draft_model
        self._num_draft_tokens = num_draft_tokens
        self._scheduler_config = scheduler_config
        self._stream_interval = stream_interval
        self._trust_remote_code = trust_remote_code
        self._force_mllm = force_mllm
        self._switch_threshold = switch_threshold

        # Shared resources (loaded once)
        self._shared_model = None
        self._shared_tokenizer = None

        # Engine instances
        self._simple: SimpleEngine | None = None
        self._batched: BatchedEngine | None = None
        self._current_mode: str | None = None  # 'simple' or 'batched'

        # Concurrency tracking
        self._active_requests = 0
        self._lock = asyncio.Lock()
        self._switch_lock = asyncio.Lock()

        # State
        self._loaded = False
        self._is_mllm = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def is_mllm(self) -> bool:
        """Check if this is a multimodal model."""
        return self._is_mllm

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._shared_tokenizer

    async def start(self) -> None:
        """Start the engine (load shared model and initialize sub-engines)."""
        if self._loaded:
            return

        logger.info(f"HybridEngine loading shared model: {self._model_name}")

        # Load model once using mlx-lm
        self._shared_model, self._shared_tokenizer = load(self._model_name)

        # Check if MLLM
        from ..api.utils import is_mllm_model

        self._is_mllm = self._force_mllm or is_mllm_model(self._model_name)

        if self._is_mllm:
            logger.warning(
                "HybridEngine does not support MLLM models yet. "
                "Using BatchedEngine only."
            )
            # For MLLM, just use BatchedEngine (no speculative)
            self._batched = BatchedEngine(
                model_name=self._model_name,
                scheduler_config=self._scheduler_config,
                stream_interval=self._stream_interval,
                force_mllm=True,
            )
            await self._batched.start()
            self._current_mode = "batched"
        else:
            # Create SimpleEngine with draft model support
            self._simple = SimpleEngine(
                model_name=self._model_name,
                trust_remote_code=self._trust_remote_code,
                draft_model=self._draft_model_name,
                num_draft_tokens=self._num_draft_tokens,
            )
            # Inject shared model instead of loading again
            await self._simple._inject_shared_model(
                self._shared_model,
                self._shared_tokenizer,
            )

            # Create BatchedEngine (lazy start - don't start engine loop yet)
            self._batched = BatchedEngine(
                model_name=self._model_name,
                trust_remote_code=self._trust_remote_code,
                scheduler_config=self._scheduler_config,
                stream_interval=self._stream_interval,
            )
            # Inject shared model but DON'T start engine loop yet
            # Engine will be started on first switch to batched mode
            await self._batched._inject_shared_model(
                self._shared_model,
                self._shared_tokenizer,
                start_engine=False,  # Lazy start for HybridEngine
            )
            self._batched_engine_started = False

            # Start in simple mode (speculative decoding)
            self._current_mode = "simple"

        self._loaded = True

        spec_info = ""
        if self._draft_model_name and not self._is_mllm:
            spec_info = f", draft={self._draft_model_name}, k={self._num_draft_tokens}"

        logger.info(
            f"HybridEngine ready: {self._model_name} "
            f"(mode={self._current_mode}, threshold={self._switch_threshold}{spec_info})"
        )

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._simple:
            await self._simple.stop()
            self._simple = None

        if self._batched:
            await self._batched.stop()
            self._batched = None

        self._shared_model = None
        self._shared_tokenizer = None
        self._loaded = False
        self._current_mode = None

        logger.info("HybridEngine stopped")

    def _get_engine_for_request(self) -> BaseEngine:
        """
        Get the appropriate engine for the current request.

        Note: This doesn't switch modes - it returns the engine based on
        current mode. Mode switching happens separately.
        """
        # For MLLM, always use batched
        if self._is_mllm:
            return self._batched

        return self._simple if self._current_mode == "simple" else self._batched

    async def _switch_to_mode(self, target_mode: str) -> None:
        """
        Switch to the specified mode, handling ownership transfer.

        This method ensures proper model ownership transfer between engines
        to prevent KV cache conflicts.

        Args:
            target_mode: 'simple' or 'batched'
        """
        if self._current_mode == target_mode:
            return

        async with self._switch_lock:
            # Double-check after acquiring lock
            if self._current_mode == target_mode:
                return

            old_mode = self._current_mode

            if target_mode == "batched":
                # Switching to batched mode
                if self._batched and self._batched._engine:
                    # Start BatchedEngine's engine loop if not started yet (lazy start)
                    if not getattr(self, "_batched_engine_started", True):
                        logger.info("HybridEngine: starting BatchedEngine (lazy start)")
                        await self._batched._engine.engine.start()
                        self._batched_engine_started = True

                    # BatchedEngine needs model ownership for its BatchGenerator
                    registry = get_registry()
                    try:
                        registry.acquire(
                            model=self._shared_model,
                            engine=self._batched._engine.engine,
                            engine_id=self._batched._engine.engine.engine_id,
                            force=True,  # Force transfer from SimpleEngine
                        )
                        logger.info(
                            "HybridEngine: model ownership transferred to BatchedEngine"
                        )
                    except Exception as e:
                        logger.warning(f"Ownership transfer failed: {e}")
                        # Continue anyway - the transfer may have succeeded partially
            else:
                # Switching to simple mode
                # SimpleEngine doesn't need registry ownership (uses mlx_lm.generate directly)
                # Just reset BatchedEngine's scheduler to clear KV cache state
                if self._batched and self._batched._engine:
                    try:
                        self._batched._engine.engine.scheduler.deep_reset()
                        logger.debug("Reset BatchedEngine scheduler for mode switch")
                    except Exception as e:
                        logger.warning(f"Failed to reset BatchedEngine: {e}")

            self._current_mode = target_mode
            logger.info(f"HybridEngine: mode switched {old_mode} -> {target_mode}")

    async def _decide_and_switch_mode(self, entering: bool = True) -> str:
        """
        Decide which mode to use and switch if necessary.

        Args:
            entering: True when entering a request, False when exiting

        Returns:
            The mode to use for this request ('simple' or 'batched')
        """
        if self._is_mllm:
            return "batched"

        # Decide based on active request count
        # When entering: count includes this request
        # When exiting: count excludes this request
        active = self._active_requests

        if active >= self._switch_threshold:
            target_mode = "batched"
        else:
            target_mode = "simple"

        # Only switch when safe (no active requests in the other engine)
        # This is a simplified heuristic - we switch when crossing the threshold
        if target_mode != self._current_mode:
            # Log the decision
            logger.debug(
                f"HybridEngine: active_requests={active}, "
                f"threshold={self._switch_threshold}, "
                f"current={self._current_mode}, target={target_mode}"
            )

            # Only switch to batched immediately, switch to simple when quiet
            if target_mode == "batched":
                await self._switch_to_mode("batched")
            elif active == 0:
                # Switch back to simple only when completely idle
                await self._switch_to_mode("simple")

        return self._current_mode

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate a complete response (non-streaming)."""
        if not self._loaded:
            await self.start()

        async with self._lock:
            self._active_requests += 1

        try:
            # Decide mode and switch if needed
            mode = await self._decide_and_switch_mode(entering=True)
            engine = self._simple if mode == "simple" else self._batched

            return await engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            )
        finally:
            async with self._lock:
                self._active_requests -= 1
            # Check if we should switch back to simple mode
            await self._decide_and_switch_mode(entering=False)

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream generation token by token."""
        if not self._loaded:
            await self.start()

        async with self._lock:
            self._active_requests += 1

        try:
            # Decide mode and switch if needed
            mode = await self._decide_and_switch_mode(entering=True)
            engine = self._simple if mode == "simple" else self._batched

            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                yield output
        finally:
            async with self._lock:
                self._active_requests -= 1
            # Check if we should switch back to simple mode
            await self._decide_and_switch_mode(entering=False)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Chat completion (non-streaming)."""
        if not self._loaded:
            await self.start()

        async with self._lock:
            self._active_requests += 1

        try:
            # Decide mode and switch if needed
            mode = await self._decide_and_switch_mode(entering=True)
            engine = self._simple if mode == "simple" else self._batched

            return await engine.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            )
        finally:
            async with self._lock:
                self._active_requests -= 1
            # Check if we should switch back to simple mode
            await self._decide_and_switch_mode(entering=False)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[dict] | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream chat completion token by token."""
        if not self._loaded:
            await self.start()

        async with self._lock:
            self._active_requests += 1

        try:
            # Decide mode and switch if needed
            mode = await self._decide_and_switch_mode(entering=True)
            engine = self._simple if mode == "simple" else self._batched

            async for output in engine.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            ):
                yield output
        finally:
            async with self._lock:
                self._active_requests -= 1
            # Check if we should switch back to simple mode
            await self._decide_and_switch_mode(entering=False)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "engine_type": "hybrid",
            "model_name": self._model_name,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "current_mode": self._current_mode,
            "active_requests": self._active_requests,
            "switch_threshold": self._switch_threshold,
            "draft_model": self._draft_model_name,
            "num_draft_tokens": self._num_draft_tokens,
        }

        if self._simple:
            stats["simple_engine"] = self._simple.get_stats()
        if self._batched:
            stats["batched_engine"] = self._batched.get_stats()

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics from active engine."""
        if self._current_mode == "simple" and self._simple:
            return self._simple.get_cache_stats()
        elif self._batched:
            return self._batched.get_cache_stats()
        return None
