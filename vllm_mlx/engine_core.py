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
import concurrent.futures
import logging
import os
import sys
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


def _init_mlx_step_thread() -> None:
    """Create the MLX worker thread's generation stream + default stream.

    Runs once when the executor spawns its single worker. Steps:

    1. Create a new stream on this thread. mlx-lm / mlx-vlm both hold a
       module-level ``generation_stream`` that was created on the import
       thread (the asyncio loop / main thread); operations queued onto
       that stream cannot be ``mx.eval``-ed from this worker thread —
       you get ``RuntimeError: There is no Stream(gpu, N) in current
       thread.`` (#170, follow-on to #161 / #167).
    2. Reassign the module-level ``generation_stream`` in BOTH
       ``mlx_lm.generate`` and ``mlx_vlm.generate`` (Gemma 4 etc. lives
       in the latter, and their ``with mx.stream(generation_stream)``
       blocks otherwise capture the import-thread stream).
    3. ``mx.set_default_stream(stream)`` — anything that allocates an
       ``mx.array`` without an explicit stream context (e.g.
       ``BatchRotatingKVCache.__init__`` doing ``mx.array(left_padding)``
       inside ``BatchGenerator._next()``'s prompt cache merge) also
       lands on a thread-local stream we can eval. ``set_default_stream``
       is process-wide, but the main thread no longer issues MLX work
       after warmup is routed here too (PR #173).
    """
    # mlx-lm 0.31.3+: generation_stream is `mx.new_thread_local_stream` bound
    # to the THREAD that created it. BatchGenerator captures
    # `self._stream = stream or generation_stream` at __init__, and any
    # captured ThreadLocalStream from the import thread fails to eval from
    # this worker (#170: "There is no Stream(gpu, 1) in current thread.").
    #
    # Adopt the worker thread's auto-default stream rather than creating a
    # NEW stream. MLX lazily creates a default stream per-thread on first
    # access — using that stream guarantees ad-hoc `mx.array(...)` calls
    # (which use the default stream when no `with mx.stream(...)` context
    # is active) and our `with mx.stream(stream)` overrides converge on the
    # same stream. Creating a new ThreadLocalStream and reassigning it can
    # leave the worker's TRUE default at a different index, so any path that
    # falls back to the default mid-flight ends up tagging arrays with one
    # stream while BatchGenerator evals against the other.
    stream = mx.default_stream(mx.default_device())

    for mod_name in ("mlx_lm.generate", "mlx_vlm.generate"):
        gen_mod = sys.modules.get(mod_name)
        if gen_mod is not None:
            gen_mod.generation_stream = stream

    logger.info(
        "MLX step thread initialized: stream=%s (worker default = generation_stream)",
        stream,
    )


@dataclass
class EngineConfig:
    """Configuration for the engine."""

    model_name: str = ""
    scheduler_config: SchedulerConfig | None = None
    # Housekeeping cadence when the scheduler is empty. The loop wakes
    # immediately on a new ``add_request`` via an asyncio Event; this
    # interval only bounds idle wake-ups for periodic checks. Bumped from
    # 1 ms (kHz polling) to 5 s (0.2 Hz) — see issue #265.
    step_interval: float = 5.0
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
        # Per-request accumulator for stream_interval > 1: each step's
        # new_text/new_token_ids delta is merged here regardless of
        # should_send(); the buffer is flushed in one shot when should_send()
        # triggers, so no token deltas are dropped between sends.
        self._stream_buffers: dict[str, RequestOutput] = {}

        # Engine state
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: float | None = None
        self._steps_executed = 0

        # Idle-wakeup signal: ``add_request`` sets this so the engine loop
        # can wait on it instead of polling at kHz when the scheduler is
        # empty. Lazy-created in ``_engine_loop`` because asyncio.Event
        # binds to the running loop. See issue #265.
        self._idle_event: asyncio.Event | None = None

        # Single MLX worker thread that owns the per-thread generation_stream
        # (created on demand in start(); see _init_mlx_step_thread). All MLX
        # array operations that touch cached KV state must go through this
        # executor — the asyncio loop thread does not own a stream and will
        # raise "There is no Stream(gpu, N) in current thread."
        self._mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Per-model capability profile. Runtime probe acts as the safety
        # net for unknown hybrid arches; regex-based defaults from
        # ``detect_model_config`` flow in via the engine config when the
        # model name is known.
        from .model_auto_config import (
            detect_model_config,
            enrich_model_config,
            format_profile_summary,
            format_profile_table,
            suffix_decoding_hint,
        )

        model_path = (
            getattr(self.config, "model_name", None)
            or getattr(self.config, "model_path", None)
            or getattr(model, "name_or_path", None)
        )
        # Guard against MagicMock / non-string sentinels coming from test
        # stubs and partially-loaded models.
        if not isinstance(model_path, str) or not model_path:
            model_path = None
        base_cfg = detect_model_config(model_path) if model_path else None
        self.model_config = enrich_model_config(base_cfg, model)
        # Plumb profile into Scheduler so spec-decode installs can consult
        # capability gates (e.g. ``supports_spec_decode``).
        self.scheduler.model_config = self.model_config
        self._hybrid_throttle = self.model_config.is_hybrid
        self._hybrid_lock: asyncio.Lock | None = None  # lazy-init in event loop
        self._last_request_time = 0.0

        # Level 1 — always emit a one-line profile summary on engine init.
        # Level 2 — verbose ASCII capability table when explicitly requested
        # via env var ``RAPID_MLX_PROFILE_VERBOSE=1`` (or set on EngineConfig).
        display_path = model_path or "(unknown)"
        logger.info(format_profile_summary(display_path, self.model_config))
        if os.environ.get("RAPID_MLX_PROFILE_VERBOSE") == "1" or getattr(
            self.config, "verbose_profile", False
        ):
            for line in format_profile_table(
                display_path, self.model_config
            ).splitlines():
                logger.info(line)

        # SuffixDecoding eligibility hint — silent for unknown/neutral tiers
        # so this only fires when the bench has data worth surfacing.
        hint = suffix_decoding_hint(self.model_config)
        if hint:
            logger.info(hint)

        logger.debug(f"Engine {self._engine_id} initialized")

    async def start(
        self, executor: concurrent.futures.ThreadPoolExecutor | None = None
    ) -> None:
        """Start the engine loop.

        Args:
            executor: Optional pre-existing single-thread executor (whose
                worker is the mlx-step thread that already loaded the model).
                Reusing the same worker keeps every MLX op — model weights,
                forward passes, cache state — bound to one thread, which is
                the only configuration that survives mlx-lm 0.31.3+
                ThreadLocalStream tagging (#170). When None, a fresh
                executor is created.
        """
        if self._running:
            return

        if executor is not None:
            self._mlx_executor = executor
        else:
            self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mlx-step",
                initializer=_init_mlx_step_thread,
            )
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
        if self._mlx_executor is not None:
            # Tear down BatchGenerator on the worker thread that owns its
            # generation stream. Otherwise its __del__ runs at process exit
            # in some other thread and calls mx.synchronize on a stream
            # that no longer exists, raising
            # "There is no Stream(gpu, N) in current thread."
            try:
                self._run_on_step_thread(self.scheduler._close_batch_generator)
            except Exception as e:
                logger.debug(f"Error closing BatchGenerator on worker: {e}")
            self._mlx_executor.shutdown(wait=True)
            self._mlx_executor = None
        logger.info("Engine stopped")

    def _run_on_step_thread(self, func, *args, **kwargs):
        """Run `func` on the MLX worker thread and return its result.

        Use this for MLX operations called from the asyncio loop thread
        (or any other thread) that touch arrays whose backing stream lives
        on the worker — e.g. saving the prefix cache to disk on shutdown.

        Falls back to a direct call if the executor isn't available
        (engine not started or already stopped); the caller will see the
        same Stream(gpu, N) error it would have seen pre-fix.
        """
        executor = self._mlx_executor
        if executor is None:
            # No worker thread: callers in test/CLI paths run sync. In
            # production after start() this should never trigger; if it
            # does, the call is about to hit Stream(gpu, N) again — log
            # at debug so a future regression surfaces in diagnostics.
            logger.debug(
                "_run_on_step_thread: no executor, running %s inline",
                getattr(func, "__qualname__", func),
            )
            return func(*args, **kwargs)
        future = executor.submit(func, *args, **kwargs)
        return future.result()

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    async def _engine_loop(self) -> None:
        """Main engine loop."""
        # The single mlx-step worker thread is created in start() so that
        # _run_on_step_thread() can also reach it from non-loop callers
        # (e.g. shutdown cache persistence). mlx-lm generation streams are
        # thread-local, and Qwen3.x RotatingKVCache keeps arrays tagged
        # with that stream across prefill and decode.
        _executor = self._mlx_executor
        loop = asyncio.get_running_loop()

        step_interval = self.config.step_interval
        stream_interval = self.config.stream_interval
        use_simple_streaming = stream_interval == 1

        # Bind the idle-wakeup event to the running loop (asyncio.Event must
        # be created on the loop it's awaited from).
        if self._idle_event is None:
            self._idle_event = asyncio.Event()

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
            _device_mem = (
                _max_recommended if _max_recommended > 0 else 200 * 1024 * 1024 * 1024
            )
            _memory_pressure_threshold = int(
                _device_mem * min(_gpu_mem_util + 0.05, 0.99)
            )
        except Exception:
            _memory_pressure_threshold = 200 * 1024 * 1024 * 1024
        # Check every 16 steps (was 64). Sustained-load Metal errors
        # correlate with creeping memory pressure; catching it 4× sooner
        # gives mx.clear_cache() a chance to defuse before the next
        # large allocation pushes us into a Metal-side OOM, which currently
        # propagates as an uncatchable abort (see GitHub #353 / mlx-lm#1015).
        _memory_check_interval = 16

        while self._running:
            try:
                if self.scheduler.has_requests():
                    output = await loop.run_in_executor(_executor, self.scheduler.step)
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
                                    # Merge this step's delta into the buffer so
                                    # tokens that fall between should_send() hits
                                    # are not silently discarded. new_text,
                                    # new_token_ids, and logprobs are all
                                    # per-step deltas (scheduler emits one
                                    # token's worth per step) and must
                                    # accumulate; cumulative status fields take
                                    # the latest value.
                                    buf = self._stream_buffers.get(rid)
                                    # Wrap per-step logprobs (mx.array or None)
                                    # into a list so subsequent merges concat
                                    # instead of overwriting. The flushed
                                    # RequestOutput's logprobs is therefore a
                                    # list[mx.array] when stream_interval > 1.
                                    new_lp = (
                                        [req_output.logprobs]
                                        if req_output.logprobs is not None
                                        else []
                                    )
                                    prev_text = buf.new_text if buf else ""
                                    prev_tokens = buf.new_token_ids if buf else []
                                    prev_lp = (buf.logprobs if buf else None) or []
                                    self._stream_buffers[rid] = RequestOutput(
                                        request_id=rid,
                                        new_token_ids=prev_tokens
                                        + req_output.new_token_ids,
                                        new_text=prev_text + req_output.new_text,
                                        output_token_ids=req_output.output_token_ids,
                                        output_text=req_output.output_text,
                                        finished=req_output.finished,
                                        finish_reason=req_output.finish_reason,
                                        prompt_tokens=req_output.prompt_tokens,
                                        completion_tokens=req_output.completion_tokens,
                                        logprobs=(prev_lp + new_lp) or None,
                                    )
                                    if state and state.should_send(
                                        req_output.completion_tokens,
                                        req_output.finished,
                                    ):
                                        flushed = self._stream_buffers.pop(rid)
                                        collector.put(flushed)
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
                    # No work — block until ``add_request`` sets the
                    # event, with a long fallback timeout for
                    # housekeeping (memory pressure, scheduler.deep_reset
                    # races, etc.). Drops idle CPU from kHz polling to
                    # essentially zero without adding any first-token
                    # latency for requests that arrive mid-idle.
                    try:
                        await asyncio.wait_for(
                            self._idle_event.wait(), timeout=step_interval
                        )
                    except asyncio.TimeoutError:
                        pass
                    self._idle_event.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback

                # If step() raises (e.g. Metal RuntimeError that did propagate
                # synchronously to Python, an unexpected scheduler bug, etc.),
                # the in-flight requests would otherwise hang forever waiting
                # on finished_events that never get set. Surface the error to
                # every waiting client as a structured RequestOutput and
                # release their events so HTTP handlers can return 503 instead
                # of timing out.
                #
                # NOTE: this does NOT recover from the async Metal abort path
                # described in #353. When mlx::core::gpu::check_error throws
                # from inside Metal's addCompletedHandler block, the C++
                # exception propagates straight to std::terminate -> abort()
                # without ever passing through a Python frame, and the whole
                # process dies. Tracking that path against mlx-lm#1015 /
                # ml-explore/mlx#... — once mlx defers the throw to the next
                # eval, this handler will start catching it.
                msg = str(e)
                is_metal = any(
                    needle in msg
                    for needle in ("Metal", "MTL", "command buffer", "gpu::check_error")
                )
                if is_metal:
                    logger.error(
                        "Metal runtime error caught in engine loop — failing "
                        f"in-flight requests and clearing buffers: {e}"
                    )
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass
                else:
                    logger.error(f"Engine loop error: {e}\n{traceback.format_exc()}")

                # Fail all in-flight requests with a structured error output so
                # awaiting HTTP handlers unblock (#353).
                err_text = (
                    f"Inference aborted: {type(e).__name__}: {e}"
                    if is_metal
                    else f"Engine loop error: {type(e).__name__}: {e}"
                )
                for rid in list(self._finished_events.keys()):
                    collector = self._output_collectors.get(rid)
                    if collector is not None:
                        try:
                            collector.put(
                                RequestOutput(
                                    request_id=rid,
                                    new_token_ids=[],
                                    new_text="",
                                    output_token_ids=[],
                                    output_text="",
                                    finished=True,
                                    finish_reason="error",
                                    prompt_tokens=0,
                                    completion_tokens=0,
                                    error=err_text,
                                )
                            )
                        except Exception:
                            pass
                    ev = self._finished_events.get(rid)
                    if ev is not None:
                        ev.set()

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

        # Dispatch to the mlx-step worker so any MLX arrays allocated during
        # prefix cache lookup (memory_aware_cache.fetch deep-copies cached KV
        # state, paged_cache.reconstruct_cache materializes block tensors,
        # etc.) are tagged with the worker's default stream. Otherwise those
        # arrays carry the asyncio loop thread's stream and the next
        # batch_generator.next() raises "There is no Stream(gpu, N) in
        # current thread" inside `mx.eval([c.state for c in self.prompt_cache])`.
        # Complements the warmup/model-load fix in PR #173 / #174.
        if self._mlx_executor is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._mlx_executor, self.scheduler.add_request, request
            )
        else:
            self.scheduler.add_request(request)

        # Wake the engine loop if it's blocked on the idle event.
        # asyncio.Event.set() is loop-thread-safe when called from coros
        # running on the same loop (which add_request always is).
        if self._idle_event is not None:
            self._idle_event.set()

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
        self._stream_buffers.pop(request_id, None)
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

            # Engine loop sets error= when it aborts the request (e.g. Metal
            # runtime error). Surface as InferenceAbortedError so the HTTP
            # layer maps to 503 (#353).
            if final_output.error:
                from .request import InferenceAbortedError

                raise InferenceAbortedError(final_output.error)

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
        """Save prefix cache to disk.

        Routed through the mlx-step worker thread because the KV-cache
        arrays are tagged with that thread's generation_stream; trying to
        materialize them from the asyncio loop thread raises
        "There is no Stream(gpu, N) in current thread."
        """
        return self._run_on_step_thread(self.scheduler.save_cache_to_disk, cache_dir)

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk.

        Loading also goes through the worker thread so the loaded arrays
        end up tagged with the generation_stream that subsequent fetches
        will run on.
        """
        return self._run_on_step_thread(self.scheduler.load_cache_from_disk, cache_dir)

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
        self._stream_buffers.clear()
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
