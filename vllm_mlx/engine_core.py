# SPDX-License-Identifier: Apache-2.0
"""
Engine Core for vllm-mlx continuous batching.

This module provides the EngineCore class that coordinates:
- Model loading and management
- Request scheduling via Scheduler
- Async request processing
- Output streaming

The design follows vLLM's engine architecture adapted for MLX.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .model_registry import get_registry
from .output_collector import RequestOutputCollector, RequestStreamState
from .request import Request, RequestOutput, SamplingParams
from .scheduler import Scheduler, SchedulerConfig

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: SchedulerConfig | None = None
    step_interval: float = 0.001  # 1ms between steps
    stream_interval: int = 1  # Tokens to batch before streaming (1=every token)
    gpu_memory_utilization: float = 0.90  # Fraction of device memory for allocation
    tool_logits_processor_factory: Any | None = None  # Factory for tool logits bias


class EngineCore:
    """
    Core engine for vllm-mlx inference with continuous batching.

    This engine runs the generation loop and manages request lifecycle.
    It provides both sync and async interfaces for request handling.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineConfig | None = None,
        engine_id: str | None = None,
        force_model_ownership: bool = True,
    ):
        """
        Initialize the engine.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Engine configuration
            engine_id: Optional unique ID for this engine (auto-generated if None)
            force_model_ownership: If True (default), forcibly take model ownership
                                   from any existing engine. If False, raises
                                   ModelOwnershipError if model is in use.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        self._engine_id = engine_id or str(uuid.uuid4())
        self._owns_model = False
        self._closed = False

        # Acquire model ownership
        registry = get_registry()
        registry.acquire(
            model=model,
            engine=self,
            engine_id=self._engine_id,
            force=force_model_ownership,
        )
        self._owns_model = True

        # Create scheduler
        scheduler_config = self.config.scheduler_config or SchedulerConfig()
        self.scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=scheduler_config,
            tool_logits_processor_factory=self.config.tool_logits_processor_factory,
        )

        # Output collectors for low-latency streaming (vLLM pattern)
        self._output_collectors: dict[str, RequestOutputCollector] = {}
        self._stream_states: dict[str, RequestStreamState] = {}
        self._finished_events: dict[str, asyncio.Event] = {}

        # Engine state
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: float | None = None
        self._steps_executed = 0

        # Detect hybrid models (GatedDeltaNet) that need request throttling
        self._hybrid_throttle = False
        self._hybrid_lock: asyncio.Lock | None = None  # lazy-init in event loop
        self._last_request_time = 0.0
        try:
            if hasattr(model, "make_cache"):
                from mlx_lm.models.cache import ArraysCache

                test_cache = model.make_cache()
                if any(isinstance(c, ArraysCache) for c in test_cache):
                    self._hybrid_throttle = True
                    logger.info(
                        "Hybrid model detected (GatedDeltaNet) — "
                        "enabling 200ms request throttle for batch safety"
                    )
        except Exception:
            pass

        logger.debug(f"Engine {self._engine_id} initialized")

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._engine_loop())
        logger.info("Engine started")

    async def stop(self) -> None:
        """Stop the engine loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Engine stopped")

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    async def _engine_loop(self) -> None:
        """Main engine loop - hybrid executor for prefill vs generation.

        Prefill steps (long prompts) are run in a thread executor to keep
        the asyncio event loop responsive.  Generation-only steps (~1-3ms)
        are called directly to avoid ~0.5-2ms context switch overhead,
        giving ~5-10% throughput improvement during sustained generation.
        """
        import concurrent.futures

        # Single-thread executor ensures MLX calls are never concurrent
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-step"
        )
        loop = asyncio.get_running_loop()

        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = stream_interval == 1

        # Emergency memory pressure threshold — dynamic based on gpu_memory_utilization.
        # Uses Metal's max recommended working set when available, falling back to
        # device memory. Applies a 5% gap above the soft limit (capped at 99%).
        _gpu_mem_util = self.config.gpu_memory_utilization
        try:
            _device_info = mx.device_info()
            _max_recommended = _device_info.get(
                "max_recommended_working_set_size",
                _device_info.get("memory_size", 0),
            )
            _device_mem = _max_recommended if _max_recommended > 0 else 200 * 1024 * 1024 * 1024
            _memory_pressure_threshold = int(
                _device_mem * min(_gpu_mem_util + 0.05, 0.99)
            )
        except Exception:
            _memory_pressure_threshold = 200 * 1024 * 1024 * 1024
        _memory_check_interval = 64

        while self._running:
            try:
                if self.scheduler.has_requests():
                    # Hybrid approach: use executor only when prefill is likely.
                    # Prefill happens when there are waiting requests that need
                    # to be inserted into the batch (may block for seconds).
                    # Generation-only steps are fast (<3ms) and can run inline.
                    has_waiting = self.scheduler.get_num_waiting() > 0
                    has_partial = (
                        self.scheduler.batch_generator is not None
                        and getattr(self.scheduler.batch_generator, "_partial", None)
                        is not None
                    )
                    needs_executor = has_waiting or has_partial

                    if needs_executor:
                        output = await loop.run_in_executor(
                            _executor, self.scheduler.step
                        )
                    else:
                        output = self.scheduler.step()
                        # Yield to event loop after inline step
                        await asyncio.sleep(0)
                    self._steps_executed += 1

                    # Emergency memory pressure check
                    if self._steps_executed % _memory_check_interval == 0:
                        try:
                            active_mem = mx.get_active_memory()
                            if active_mem > _memory_pressure_threshold:
                                mx.clear_cache()
                                logger.warning(
                                    f"[Memory pressure] {active_mem / 1e9:.1f}GB > "
                                    f"{_memory_pressure_threshold / 1e9:.0f}GB threshold, "
                                    f"forced cache clear"
                                )
                        except Exception:
                            pass

                    # Fast path: distribute outputs to collectors
                    outputs = output.outputs
                    if outputs:
                        collectors = self._output_collectors
                        states = self._stream_states
                        events = self._finished_events

                        for req_output in outputs:
                            rid = req_output.request_id
                            collector = collectors.get(rid)

                            if collector is not None:
                                # Optimized: skip stream_interval check when interval=1
                                if use_simple_streaming:
                                    collector.put(req_output)
                                else:
                                    state = states.get(rid)
                                    if state and state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished,
                                    ):
                                        collector.put(req_output)
                                        state.mark_sent(req_output.completion_tokens)

                            if req_output.finished:
                                event = events.get(rid)
                                if event:
                                    event.set()

                        # Free Metal buffers after distributing finished outputs
                        if output.finished_request_ids:
                            mx.clear_cache()

                        # Always yield to prevent event loop starvation.
                        # Without this, orphaned requests (client disconnected but
                        # request still in scheduler) block the entire event loop,
                        # making the server unresponsive to all HTTP requests.
                        await asyncio.sleep(0)
                else:
                    # No work, yield control
                    await asyncio.sleep(step_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                logger.error(f"Engine loop error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        prefix_boundary: int = 0,
    ) -> str:
        """
        Add a request for processing.

        Args:
            prompt: Input prompt (string or token IDs)
            sampling_params: Generation parameters
            request_id: Optional custom request ID
            images: Optional images for multimodal
            videos: Optional videos for multimodal
            prefix_boundary: Token count for shared prefix (for cache)

        Returns:
            The request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            images=images,
            videos=videos,
            prefix_boundary=prefix_boundary,
        )

        # Throttle requests for hybrid models (GatedDeltaNet + Transformer).
        # Simultaneous batch formation with ArraysCache causes corruption
        # (all outputs become token 0).  A 200ms gap between inserts lets
        # the BatchGenerator fully absorb each request before the next arrives.
        # The first request needs a longer gap (500ms) to allow BatchGenerator
        # creation and Metal shader compilation to complete.
        if self._hybrid_throttle:
            if self._hybrid_lock is None:
                self._hybrid_lock = asyncio.Lock()
            loop = asyncio.get_running_loop()
            async with self._hybrid_lock:
                now = loop.time()
                elapsed = now - self._last_request_time
                gap = 0.5 if self._last_request_time == 0.0 else 0.2
                if elapsed < gap:
                    await asyncio.sleep(gap - elapsed)
                self._last_request_time = loop.time()

        # Setup output collector AFTER throttle so a cancelled sleep
        # doesn't leak per-request state.
        self._output_collectors[request_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[request_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[request_id] = asyncio.Event()

        # Add to scheduler
        self.scheduler.add_request(request)

        return request_id

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        result = self.scheduler.abort_request(request_id)
        self._cleanup_request(request_id)
        return result

    def _cleanup_request(self, request_id: str) -> None:
        """Clean up request tracking."""
        collector = self._output_collectors.pop(request_id, None)
        if collector:
            collector.clear()
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)
        self.scheduler.remove_finished_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request with low-latency non-blocking pattern.

        Uses the vLLM pattern: get_nowait() or await get()
        This avoids unnecessary task switches when output is available.

        Args:
            request_id: The request ID
            timeout: Optional timeout in seconds

        Yields:
            RequestOutput objects as tokens are generated
        """
        import time as _time

        _t0 = _time.monotonic()
        _token_count = 0

        collector = self._output_collectors.get(request_id)
        if collector is None:
            logger.warning(
                f"[stream_outputs] {request_id[:12]} no collector found, returning immediately"
            )
            return

        logger.info(f"[stream_outputs] {request_id[:12]} START waiting for tokens")

        finished_normally = False
        try:
            while True:
                try:
                    if timeout:
                        output = collector.get_nowait()
                        if output is None:
                            output = await asyncio.wait_for(
                                collector.get(), timeout=timeout
                            )
                    else:
                        output = collector.get_nowait() or await collector.get()

                    _token_count += 1
                    if _token_count == 1:
                        logger.info(
                            f"[stream_outputs] {request_id[:12]} first token after "
                            f"{_time.monotonic() - _t0:.1f}s"
                        )

                    yield output

                    if output.finished:
                        finished_normally = True
                        logger.info(
                            f"[stream_outputs] {request_id[:12]} finished normally, "
                            f"{_token_count} tokens in {_time.monotonic() - _t0:.1f}s"
                        )
                        break

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[stream_outputs] {request_id[:12]} TIMEOUT after "
                        f"{_token_count} tokens, {_time.monotonic() - _t0:.1f}s"
                    )
                    break

        except (GeneratorExit, asyncio.CancelledError) as exc:
            logger.info(
                f"[stream_outputs] {request_id[:12]} {type(exc).__name__} after "
                f"{_token_count} tokens, {_time.monotonic() - _t0:.1f}s"
            )

        finally:
            if not finished_normally:
                logger.info(
                    f"[stream_outputs] {request_id[:12]} ABORTING orphaned request "
                    f"({_token_count} tokens generated in {_time.monotonic() - _t0:.1f}s)"
                )
                aborted = self.scheduler.abort_request(request_id)
                logger.info(
                    f"[stream_outputs] {request_id[:12]} abort_request returned {aborted}"
                )
            self._cleanup_request(request_id)
            logger.info(f"[stream_outputs] {request_id[:12]} cleanup done")

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate a complete response (non-streaming).

        This method is optimized to avoid streaming overhead when
        you only need the final result.

        Args:
            prompt: Input prompt
            sampling_params: Generation parameters
            request_id: Optional request ID

        Returns:
            Final RequestOutput with complete text
        """
        request_id = await self.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

        # Wait for completion using event instead of streaming
        # This avoids the waiting_consumer tracking overhead
        event = self._finished_events.get(request_id)
        if event is None:
            raise RuntimeError(f"No event for request {request_id}")

        try:
            # Wait for the request to finish
            await event.wait()

            # Get the final output from collector
            collector = self._output_collectors.get(request_id)
            if collector is None:
                raise RuntimeError(f"No collector for request {request_id}")

            # Drain all outputs and get the last one
            final_output = None
            while True:
                output = collector.get_nowait()
                if output is None:
                    break
                final_output = output

            if final_output is None:
                raise RuntimeError(f"No output for request {request_id}")

            return final_output

        except (asyncio.CancelledError, GeneratorExit):
            logger.info(f"[generate] {request_id[:12]} CANCELLED, aborting request")
            self.scheduler.abort_request(request_id)
            raise

        finally:
            self._cleanup_request(request_id)

    def generate_batch_sync(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """
        Generate responses synchronously for maximum throughput.

        This bypasses the async engine loop entirely, running the scheduler
        directly for optimal batching performance. Use this when you don't
        need streaming and want maximum throughput.

        Args:
            prompts: List of input prompts
            sampling_params: Generation parameters (same for all)

        Returns:
            List of RequestOutput in same order as prompts
        """
        import uuid as uuid_module

        from .request import Request

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Add all requests to scheduler
        request_ids = []
        for prompt in prompts:
            request_id = str(uuid_module.uuid4())
            request = Request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            self.scheduler.add_request(request)
            request_ids.append(request_id)

        # Process until all done - direct scheduler access, no async overhead
        results: dict[str, RequestOutput] = {}
        while self.scheduler.has_requests():
            output = self.scheduler.step()
            for req_output in output.outputs:
                if req_output.finished:
                    results[req_output.request_id] = req_output

        # Cleanup
        for rid in request_ids:
            self.scheduler.remove_finished_request(rid)

        # Return in original order
        return [results[rid] for rid in request_ids]

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        scheduler_stats = self.scheduler.get_stats()
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "steps_executed": self._steps_executed,
            "active_requests": len(self._output_collectors),
            "stream_interval": self.config.stream_interval,
            "requests": self.scheduler.get_running_requests_info(),
            **scheduler_stats,
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get prefix cache statistics."""
        return self.scheduler.get_cache_stats()

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk."""
        return self.scheduler.save_cache_to_disk(cache_dir)

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk."""
        return self.scheduler.load_cache_from_disk(cache_dir)

    def _release_model(self) -> None:
        """Release model ownership."""
        if self._owns_model and not self._closed:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

    def close(self) -> None:
        """
        Explicitly close the engine and release resources.

        This should be called when done using the engine, especially
        if you plan to create another engine with the same model.
        """
        if self._closed:
            return

        # Release model ownership BEFORE setting _closed
        # (_release_model checks not self._closed)
        if self._owns_model:
            registry = get_registry()
            registry.release(self.model, self._engine_id)
            self._owns_model = False
            logger.debug(f"Engine {self._engine_id} released model ownership")

        self._closed = True

        # Reset scheduler to clear BatchGenerator and all caches
        self.scheduler.deep_reset()

        # Clear output collectors
        for collector in self._output_collectors.values():
            collector.clear()
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()

        logger.debug(f"Engine {self._engine_id} closed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._release_model()
        except Exception:
            # Ignore errors during garbage collection
            pass

    @property
    def engine_id(self) -> str:
        """Get the engine ID."""
        return self._engine_id


class AsyncEngineCore:
    """
    Async context manager wrapper for EngineCore.

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            request_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs(request_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineConfig | None = None,
    ):
        self.engine = EngineCore(model, tokenizer, config)

    async def __aenter__(self) -> "AsyncEngineCore":
        await self.engine.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.engine.stop()

    def start(self) -> None:
        """Start engine (creates task in current loop)."""
        asyncio.create_task(self.engine.start())

    async def stop(self) -> None:
        """Stop the engine."""
        await self.engine.stop()

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> str:
        """Add a request."""
        return await self.engine.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            **kwargs,
        )

    async def abort_request(self, request_id: str) -> bool:
        """Abort a request."""
        return await self.engine.abort_request(request_id)

    async def stream_outputs(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> AsyncIterator[RequestOutput]:
        """Stream outputs."""
        async for output in self.engine.stream_outputs(request_id, timeout):
            yield output

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ) -> RequestOutput:
        """Generate complete response."""
        return await self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            **kwargs,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get engine stats."""
        return self.engine.get_stats()

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get prefix cache statistics."""
        return self.engine.get_cache_stats()

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk."""
        return self.engine.save_cache_to_disk(cache_dir)

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk."""
        return self.engine.load_cache_from_disk(cache_dir)
