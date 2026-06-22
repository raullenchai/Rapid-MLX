# SPDX-License-Identifier: Apache-2.0
"""
MLLM Scheduler for multimodal continuous batching.

This scheduler handles Multimodal Language Model requests with continuous
batching support, following the same architecture as the LLM scheduler.

Key features:
- Batch processing of multiple MLLM requests
- Vision embedding caching for repeated images
- Step-based generation loop (like LLM scheduler)
- Support for both streaming and non-streaming generation

Architecture:
1. Requests arrive via add_request() -> waiting queue
2. Scheduler moves requests from waiting to running (via MLLMBatchGenerator)
3. step() method generates one token for ALL running requests
4. Finished requests are removed and outputs returned
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

# MUST install the MLX hardware-compat shim BEFORE any `from mlx_lm.*` import.
# `mlx_lm/__init__.py` re-exports from `mlx_lm.generate`, which captures
# `mx.new_thread_local_stream(mx.default_device())` at module-import time; on
# M5 single-stream GPUs that stream is unusable (#404). The shim is
# idempotent and a no-op on hardware where the original API works.
from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer  # noqa: E402

from .mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
)
from .mllm_cache import MLLMCacheManager  # noqa: E402
from .multimodal_processor import MultimodalProcessor  # noqa: E402
from .request import RequestOutput, RequestStatus, SamplingParams  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class MLLMSchedulerConfig:
    """Configuration for MLLM scheduler."""

    # Maximum concurrent MLLM requests in the batch
    max_num_seqs: int = 16
    # Prefill batch size (all queued requests are prefilled together)
    prefill_batch_size: int = 16
    # Completion batch size
    completion_batch_size: int = 16
    # Prefill step size — per-request prompt-token budget. Vision tokens
    # balloon the prompt size on VLMs (a 1920×1080 screenshot alone is
    # ~2200 tokens on Qwen3-VL), so the MLLM-side default is 8192 to
    # cover typical desktop screenshots and small multi-image messages
    # out-of-the-box. ``BatchedEngine._start_mllm`` applies a bump-policy
    # (see ``_resolve_mllm_prefill_step_size``) so a SchedulerConfig
    # carrying the text-LLM default (2048) gets bumped to 8192, while
    # any explicit operator-set value is honored as-is (#682, codex r2).
    prefill_step_size: int = 8192
    # Enable vision embedding cache
    enable_vision_cache: bool = True
    # Maximum cache entries
    vision_cache_size: int = 100
    # Default max tokens
    default_max_tokens: int = 256
    # Default video FPS for frame extraction
    default_video_fps: float = 2.0
    # Maximum video frames
    max_video_frames: int = 128
    # Admission control: hard cap on concurrent in-flight MLLM
    # requests (queued + running). Matches the LLM scheduler
    # convention so ``max_concurrent_requests`` is uniform across
    # both engines. Default 256 provides queue depth on top of
    # ``max_num_seqs`` (waiting requests cost only the tokenised
    # prompt, not KV state). Operators who want admission tied to
    # ``max_num_seqs`` pass ``--max-concurrent-requests`` explicitly.
    max_concurrent_requests: int = 256


@dataclass
class MLLMRequest:
    """
    Extended request for MLLM processing.

    Includes all multimodal data needed for generation.
    """

    request_id: str
    prompt: str
    images: list[str] | None = None
    videos: list[str] | None = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    stop: list[str] = field(default_factory=list)  # Text-based stop sequences
    video_fps: float | None = None
    video_max_frames: int | None = None
    arrival_time: float = field(default_factory=time.time)

    # Batch generator UID (assigned when scheduled)
    batch_uid: int | None = None

    # Status tracking
    status: RequestStatus = RequestStatus.WAITING
    output_text: str = ""
    output_tokens: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    stop_tail: str = ""
    stop_text: str = ""
    stop_text_len: int = 0

    # Token counts
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0


@dataclass
class MLLMSchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: list[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: list[RequestOutput] = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False


class MLLMScheduler:
    """
    Scheduler for Vision Language Model requests with continuous batching.

    This scheduler manages the lifecycle of MLLM requests using the
    MLLMBatchGenerator for efficient batch processing:

    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via batch generator)
    3. step() generates one token for ALL running requests simultaneously
    4. Finished requests are removed and outputs returned

    Example:
        >>> scheduler = MLLMScheduler(model, processor, config)
        >>> # Add requests
        >>> request_id = scheduler.add_request(
        ...     prompt="What's in this image?",
        ...     images=["photo.jpg"]
        ... )
        >>> # Run generation loop
        >>> while scheduler.has_requests():
        ...     output = scheduler.step()
        ...     for req_output in output.outputs:
        ...         if req_output.finished:
        ...             print(f"Finished: {req_output.output_text}")

    For async usage with streaming:
        >>> await scheduler.start()
        >>> request_id = await scheduler.add_request_async(...)
        >>> async for output in scheduler.stream_outputs(request_id):
        ...     print(output.new_text, end="")
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: MLLMSchedulerConfig | None = None,
        step_executor: Any | None = None,
    ):
        """
        Initialize MLLM scheduler.

        Args:
            model: The VLM model
            processor: The VLM processor
            config: Scheduler configuration
            step_executor: Optional pre-created single-thread ThreadPoolExecutor
                that owns the ``mllm-step`` worker. The model MUST have been
                loaded on this executor — under mlx-lm 0.31.3+, every later
                ``mx.eval`` against the model weights has to come from the
                same thread that created them. If ``None``, a fresh executor
                is created in ``_process_loop`` (the caller-loaded model will
                then crash with ``Stream(gpu, N) in current thread``).
        """
        self.model = model
        self.processor = processor
        self.config = config or MLLMSchedulerConfig()
        self._injected_step_executor = step_executor

        # Get model config
        self.model_config = getattr(model, "config", None)

        # Multimodal processor for input preparation
        self.mm_processor = MultimodalProcessor(
            model=model,
            processor=processor,
            config=self.model_config,
        )

        # Vision cache for repeated images
        self.vision_cache: MLLMCacheManager | None = None
        if self.config.enable_vision_cache:
            self.vision_cache = MLLMCacheManager(
                max_entries=self.config.vision_cache_size
            )

        # Get stop tokens from tokenizer
        self.stop_tokens = self._get_stop_tokens()

        # Batch generator (created lazily)
        self.batch_generator: MLLMBatchGenerator | None = None

        # Request management - following vLLM's design
        self.waiting: deque[MLLMRequest] = deque()  # Waiting queue (FCFS)
        self.running: dict[str, MLLMRequest] = {}  # Running requests by ID
        self.requests: dict[str, MLLMRequest] = {}  # All requests by ID
        self.finished_req_ids: set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: dict[str, int] = {}
        self.uid_to_request_id: dict[int, str] = {}

        # Per-request streaming detokenizers for UTF-8-safe incremental decode
        self._detokenizer_pool: dict[str, Any] = {}

        # Output queues for async streaming
        self.output_queues: dict[str, asyncio.Queue] = {}

        # Thread-safe set for deferred aborts (event loop → executor thread).
        # CPython GIL guarantees set.add() and set.pop() are atomic.
        self._pending_abort_ids: set[str] = set()
        # M-01 codex r2 BLOCKING #2: lifetime de-dup ledger for the
        # total cancellation counter. Mirror of
        # ``Scheduler._cancelled_request_ids`` — see that comment for
        # the rationale (TL;DR: ``_pending_abort_ids`` drains every
        # step so it's the wrong ledger to dedupe a lifetime counter
        # against).
        self._cancelled_request_ids: set[str] = set()
        # M-01: once-per-request guard for the disconnect-cause
        # sub-counter. Mirrors ``Scheduler._disconnect_abort_ids`` —
        # the helper-layer ``_force_abort_request`` may fire two or
        # three times per disconnect (disconnect branch + GeneratorExit
        # branch + finally belt-and-suspenders) so the sub-counter
        # needs its own de-dup set instead of leaning on
        # ``_pending_abort_ids`` which drains every step.
        self._disconnect_abort_ids: set[str] = set()
        # M-01 codex r1 BLOCKING #4/#5: same atomicity rationale as
        # ``Scheduler._cancel_counter_lock`` — the check-add-increment
        # for both the total and via_disconnect counters must be
        # serialized across threads. MLLM-active engines reach this
        # path from the same disconnect_guard multi-branch fire AND
        # the explicit cancel route, so the concurrency surface is
        # identical.
        self._cancel_counter_lock = threading.Lock()
        # Aborted request IDs that need queue signaling (executor → event loop).
        self._aborted_queue_ids: set[str] = set()

        # Async processing control
        self._running = False
        self._processing_task: asyncio.Task | None = None
        self._step_executor = None  # ThreadPoolExecutor, created in _process_loop

        # Memory management: periodic mx.clear_cache() to free Metal buffer pool
        self._step_count = 0
        self._clear_cache_interval = 32

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # M-01: cancellation observability — mirror of the text-path
        # scheduler counters so /metrics renders a stable series on
        # MLLM-active engines. See ``Scheduler.__init__`` for the full
        # rationale. Observability only — abort semantics unchanged.
        self.num_requests_cancelled = 0
        self.num_requests_cancelled_via_disconnect = 0

    def _get_stop_tokens(self) -> set[int]:
        """Get stop token IDs from tokenizer.

        Mirrors ``Scheduler._get_stop_tokens`` — see that docstring
        for the rationale behind each of the four sources.
        """
        from .utils.tokenizer import RAPID_EXTRA_EOS_ATTR

        stop_tokens: set[int] = set()
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        # Source 1: mlx-lm TokenizerWrapper's curated set.
        wrapper_ids = getattr(tokenizer, "_eos_token_ids", None)
        if wrapper_ids:
            stop_tokens.update(wrapper_ids)

        # Source 2: legacy singular path.
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                stop_tokens.update(tokenizer.eos_token_id)
            else:
                stop_tokens.add(tokenizer.eos_token_id)

        # Source 3: processor-style plural path.
        if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids is not None:
            if isinstance(tokenizer.eos_token_ids, (list, set, tuple)):
                stop_tokens.update(tokenizer.eos_token_ids)
            else:
                stop_tokens.add(tokenizer.eos_token_ids)

        # Source 4: Rapid-MLX extras stash (see RAPID_EXTRA_EOS_ATTR).
        extras = getattr(tokenizer, RAPID_EXTRA_EOS_ATTR, None)
        if extras:
            stop_tokens.update(extras)

        return stop_tokens

    def _ensure_batch_generator(self) -> None:
        """Ensure batch generator exists."""
        if self.batch_generator is None:
            from mlx_lm.sample_utils import make_sampler

            # Default sampler (can be overridden per-request in future)
            sampler = make_sampler(temp=0.7, top_p=0.9)

            self.batch_generator = MLLMBatchGenerator(
                model=self.model,
                processor=self.processor,
                mm_processor=self.mm_processor,
                max_tokens=self.config.default_max_tokens,
                stop_tokens=self.stop_tokens,
                sampler=sampler,
                prefill_batch_size=self.config.prefill_batch_size,
                completion_batch_size=self.config.completion_batch_size,
                prefill_step_size=self.config.prefill_step_size,
            )

    # ========== Sync API (step-based) ==========

    def add_request(
        self,
        prompt: str,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        request_id: str | None = None,
        **kwargs,
    ) -> str:
        """
        Add a multimodal request to the scheduler (sync version).

        Args:
            prompt: Text prompt (should be formatted with chat template)
            images: List of image inputs (paths, URLs, base64)
            videos: List of video inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Text-based stop sequences
            video_fps: FPS for video frame extraction
            video_max_frames: Max frames to extract from video
            request_id: Optional custom request ID
            **kwargs: Additional generation parameters

        Returns:
            Request ID for tracking
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Admission control: same gate as the LLM scheduler so MLLM
        # paths can't bypass the cap by going through this code path.
        cap = getattr(self.config, "max_concurrent_requests", None)
        if cap is not None and cap > 0 and len(self.requests) >= cap:
            from .scheduler import BackpressureError

            raise BackpressureError(
                f"max_concurrent_requests={cap} reached "
                f"(currently {len(self.requests)} in-flight)"
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        request = MLLMRequest(
            request_id=request_id,
            prompt=prompt,
            images=images,
            videos=videos,
            sampling_params=sampling_params,
            stop=stop or [],
            video_fps=video_fps,
            video_max_frames=video_max_frames,
        )

        # D-M01-2X (0.8.2 dogfood, codex r10 BLOCKING follow-up):
        # mirror the text-path ``Scheduler.add_request`` ledger
        # clear, gated on the same critical section as the
        # ``self.requests[...] = request`` commit. Earlier clears
        # would erase the prior lifetime's dedupe even when
        # ``SamplingParams(...)``, ``MLLMRequest(...)``, or any
        # other request-construction step subsequently raised —
        # re-opening the double-count window for the OLD
        # lifetime should a late ``abort_request`` arrive between
        # the failed admit and the next successful one. The
        # ledgers are otherwise lifetime-persistent across the
        # abort+cleanup window; see
        # ``Scheduler.remove_finished_request`` docstring for the
        # multi-branch race repro the persistence plugs.
        with self._cancel_counter_lock:
            self._cancelled_request_ids.discard(request_id)
            self._disconnect_abort_ids.discard(request_id)
            self.requests[request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added MLLM request {request_id}: "
            f"{len(images or [])} images, {len(videos or [])} videos"
        )

        return request_id

    def abort_request(self, request_id: str) -> bool:
        """
        Queue request for abort.  Thread-safe (called from event loop).

        The actual abort is deferred to the executor thread (inside
        ``_step_no_queue``) to avoid racing with in-flight GPU work
        and shared scheduler state mutations.

        Args:
            request_id: The request ID to abort

        Returns:
            True when an active/queued request was enqueued for abort, False
            when ``request_id`` is unknown to this scheduler. F-151
            hardening: previously this method returned True unconditionally,
            so the route layer would respond ``{"cancelled": true}`` for any
            attacker-supplied string. The route uses the False return as the
            404 signal.
        """
        # Match the text scheduler's notion of "known": currently admitted
        # (``requests``), in a live batch (``request_id_to_uid``), currently
        # running, or already pending abort (idempotent double-cancel). We
        # do NOT count ``finished_req_ids`` — the route contract is
        # "404 when already finished".
        # M-01 codex r1 BLOCKING #4 + r2 BLOCKING #2 + r6 BLOCKING #2:
        # membership check AND check-add-increment serialized under
        # the same lock to close the stale-admission race against
        # ``_do_abort_request`` clearing the dedupe ledgers. See
        # ``Scheduler.abort_request`` for the full rationale.
        with self._cancel_counter_lock:
            if not (
                request_id in self.requests
                or request_id in self.request_id_to_uid
                or request_id in self.running
                or request_id in self._pending_abort_ids
            ):
                logger.debug("Rejected abort for unknown MLLM request_id")
                return False
            already_counted = request_id in self._cancelled_request_ids
            self._cancelled_request_ids.add(request_id)
            self._pending_abort_ids.add(request_id)
            if not already_counted:
                self.num_requests_cancelled += 1
        logger.debug(f"Enqueued abort for request {request_id}")
        return True

    def record_disconnect_abort(self, request_id: str) -> None:
        """M-01: attribute a previously-accepted abort to client disconnect.

        Mirrors ``Scheduler.record_disconnect_abort`` so MLLM-active
        engines surface the same ``via_disconnect`` sub-counter on
        /metrics. See that method's docstring for the once-per-request
        semantics, lock-based atomicity contract (codex r1 BLOCKING
        #5), and thread-safety guarantees.

        Codex r7 NIT #3: gate on ``_cancelled_request_ids`` so the
        ``via_disconnect_total <= cancelled_total`` dashboard
        invariant holds by construction even on programmer error.
        """
        try:
            if not request_id:
                return
            with self._cancel_counter_lock:
                if request_id not in self._cancelled_request_ids:
                    return
                if request_id not in self._disconnect_abort_ids:
                    self._disconnect_abort_ids.add(request_id)
                    self.num_requests_cancelled_via_disconnect += 1
        except Exception:  # pragma: no cover - belt-and-suspenders
            pass

    def _process_pending_aborts(self) -> None:
        """Drain and execute pending abort requests.

        Must be called from the executor / step thread only.
        """
        while self._pending_abort_ids:
            request_id = self._pending_abort_ids.pop()
            self._do_abort_request(request_id)

    def _do_abort_request(self, request_id: str) -> None:
        """Actually abort a request.  Must run on the step thread."""
        request = self.requests.get(request_id)

        # Remove from waiting queue
        if request is not None and request.status == RequestStatus.WAITING:
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from batch generator
        if request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Credit in-flight tokens so dashboard metrics stay accurate
        # (without this, aborted requests' tokens vanish from /v1/status).
        # Same ``request is not None`` guard as below — late/duplicate
        # aborts arrive after self.requests.pop() and would otherwise
        # raise AttributeError on the dereference.
        if request is not None and request.num_output_tokens > 0:
            self.total_completion_tokens += request.num_output_tokens
            self.total_prompt_tokens += request.num_prompt_tokens

        # Mark as aborted
        if request is not None:
            request.status = RequestStatus.FINISHED_ABORTED
        self.finished_req_ids.add(request_id)

        # D-M01-2X + D-M01-DEAD (0.8.2 dogfood): do NOT discard the
        # dedupe ledgers — keep them lifetime-persistent. Mirrors
        # the same fix applied to ``Scheduler.remove_finished_request``
        # in the text path. See that docstring for the full repro of
        # the disconnect_guard multi-branch fire race that wiping
        # these ledgers opens up. The MLLM path has the SAME race
        # surface (disconnect_guard fires from up to three branches,
        # each may invoke ``abort_request`` against the MLLM
        # scheduler) — so keeping the ledger persistent here is
        # required for the dashboard invariant
        # ``via_disconnect_total <= cancelled_total`` and the
        # "exactly one tick per abort" contract that three personas
        # independently observed broken on PyPI 0.8.2.
        with self._cancel_counter_lock:
            self.requests.pop(request_id, None)
            self._detokenizer_pool.pop(request_id, None)

        # Do NOT write to output_queues here — this may run on the
        # executor thread where asyncio.Queue is not safe.  Mark for
        # signaling on the event loop thread via _distribute_outputs.
        self._aborted_queue_ids.add(request_id)

        logger.debug(f"Aborted request {request_id}")
        mx.clear_cache()

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _schedule_waiting(self) -> list[MLLMRequest]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        self._ensure_batch_generator()

        scheduled = []
        batch_requests = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Create batch request
            batch_req = MLLMBatchRequest(
                uid=-1,  # Will be assigned by batch generator
                request_id=request.request_id,
                prompt=request.prompt,
                images=request.images,
                videos=request.videos,
                max_tokens=request.sampling_params.max_tokens,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                video_fps=request.video_fps,
                video_max_frames=request.video_max_frames,
            )
            batch_requests.append(batch_req)

            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            scheduled.append(request)

        # Insert into batch generator
        if batch_requests and self.batch_generator is not None:
            uids = self.batch_generator.insert(batch_requests)

            for uid, request in zip(uids, scheduled):
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid

                logger.debug(f"Scheduled request {request.request_id} (uid={uid})")

        return scheduled

    def _process_batch_responses(
        self, responses: list[MLLMBatchResponse]
    ) -> tuple[list[RequestOutput], set[str]]:
        """
        Process responses from batch generator.

        Args:
            responses: List of MLLMBatchResponse objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Stamp prompt-token count on the request once the batch
            # generator has actually preprocessed the prompt (vision
            # encoding includes image-patch token expansion, so the
            # count can only be known AFTER ``_process_prompts``). Pre-
            # fix this field was never assigned, so MLLM responses always
            # reported ``usage.prompt_tokens=0``. The check is "only set
            # once" semantics: the first response carries the real count
            # and we memoise it on the ``MLLMRequest`` so every later
            # streaming chunk + the final response inherit it. ``> 0``
            # filters out the default-zero responses from corner cases
            # (text-only fallback that somehow lands here with no
            # preprocessing) so we don't overwrite a real count with 0.
            if request.num_prompt_tokens == 0:
                resp_pt = getattr(response, "prompt_tokens", 0) or 0
                if resp_pt > 0:
                    request.num_prompt_tokens = resp_pt

            token_is_control_stop_token = bool(
                getattr(response, "token_is_stop_token", False)
            )

            # Append generated content tokens to request state. Backend
            # EOS/control stop ids are scheduler control signals, not user
            # output, so they must not appear in output_token_ids either.
            if not token_is_control_stop_token:
                request.output_tokens.append(response.token)
            request.num_output_tokens = len(request.output_tokens)

            finish_reason = response.finish_reason

            # Decode the new token using streaming detokenizer (UTF-8 safe).
            # Backend EOS/control stop tokens are not decoded. Backend
            # responses that finish with normal text still detokenize so
            # the rolling matcher can keep visible text before a user stop.
            had_detok = request_id in self._detokenizer_pool
            if not had_detok:
                if hasattr(tokenizer, "detokenizer"):
                    detok = tokenizer.detokenizer
                else:
                    detok = NaiveStreamingDetokenizer(tokenizer)
                detok.reset()
                self._detokenizer_pool[request_id] = detok
            detok = self._detokenizer_pool[request_id]
            stop_params = [s for s in request.stop if s] if request.stop else []
            if token_is_control_stop_token:
                new_text = ""
            else:
                detok.add_token(response.token)
                new_text = detok.last_segment
            if token_is_control_stop_token and finish_reason is not None:
                baseline_text = (
                    request.stop_text
                    if stop_params and request.stop_text
                    else request.output_text
                )
                detok.finalize()
                finalized_text = detok.text
                if isinstance(finalized_text, str) and finalized_text.startswith(
                    baseline_text
                ):
                    new_text = finalized_text[len(baseline_text) :]
            if not isinstance(new_text, str):
                # Unit-test mocks may not implement the streaming
                # detokenizer contract. Production detokenizers
                # return str; this fallback keeps legacy tests and
                # defensive adapters on the same stop path.
                new_text = (
                    ""
                    if token_is_control_stop_token
                    else tokenizer.decode([response.token])
                )

            output_new_text = new_text
            output_output_text = ""
            output_finished = False
            output_finish_reason: str | None = None
            output_matched_stop: str | None = None

            stop_trimmed = False
            if (finish_reason != "stop" or new_text) and stop_params:
                if (
                    not had_detok
                    and request.stop_text_len == 0
                    and not request.stop_tail
                    and len(request.output_tokens) > 1
                ):
                    # Direct scheduler tests and defensive adapters can
                    # enter here with historical output_tokens but no
                    # streaming detokenizer state. Seed from a one-time
                    # full decode only for that no-detok legacy shape.
                    # If a detokenizer already exists, empty new_text means
                    # it is buffering partial bytes; offsets must wait for
                    # the detokenizer to emit text.
                    request.stop_text = tokenizer.decode(request.output_tokens[:-1])
                    request.stop_text_len = len(request.stop_text)
                    max_stop_len = max(len(s) for s in stop_params)
                    keep = max(0, max_stop_len - 1)
                    request.stop_tail = request.stop_text[-keep:] if keep else ""
                    request.output_text = request.stop_text
                    output_output_text = request.output_text
                prev_text_len = request.stop_text_len
                if stop_params and new_text:
                    max_stop_len = max(len(s) for s in stop_params)
                    keep = max(0, max_stop_len - 1)
                    previous_seen_len = len(request.stop_text)
                    streamed_so_far = request.stop_text + new_text
                    # Only the prior suffix can participate in a new
                    # cross-token stop match. This keeps the common
                    # per-token path bounded by len(new_text)+max_stop_len.
                    window_base = max(0, previous_seen_len - keep)
                    stop_window = request.stop_text[window_base:] + new_text
                    match: tuple[int, str] | None = None
                    for stop_str in stop_params:
                        search_from = max(
                            0, previous_seen_len - window_base - len(stop_str) + 1
                        )
                        local_idx = stop_window.find(stop_str, search_from)
                        if local_idx != -1:
                            global_idx = window_base + local_idx
                            if match is None or global_idx < match[0]:
                                match = (global_idx, stop_str)
                    if match is not None:
                        idx, stop_str = match
                        finish_reason = "stop"
                        output_finish_reason = finish_reason
                        # H-03: pin WHICH user-supplied stop fired so
                        # the Anthropic adapter can surface
                        # ``stop_reason="stop_sequence"`` +
                        # ``stop_sequence: <str>`` per the public spec.
                        # Mirrors the text-scheduler companion change so
                        # MLLM-backed ``/v1/messages`` traffic gets the
                        # same surface as the text path.
                        output_matched_stop = stop_str
                        # Emit only the valid prefix before the stop marker
                        # in new_text so streaming clients don't lose content.
                        visible_text = streamed_so_far[:idx]
                        output_new_text = visible_text[prev_text_len:]
                        request.output_text = visible_text
                        output_output_text = visible_text
                        request.stop_text = streamed_so_far
                        request.stop_text_len = len(streamed_so_far)
                        request.stop_tail = ""
                        stop_trimmed = True
                    else:
                        request.stop_text = streamed_so_far
                        if finish_reason is not None:
                            safe_upto = len(request.stop_text)
                        else:
                            safe_upto = max(0, len(request.stop_text) - keep)
                        output_new_text = request.stop_text[
                            request.stop_text_len : safe_upto
                        ]
                        request.stop_text_len = safe_upto
                        request.stop_tail = (
                            (
                                ""
                                if finish_reason is not None
                                else request.stop_text[-keep:]
                            )
                            if keep
                            else ""
                        )
                        request.output_text = request.stop_text[:safe_upto]
                        output_output_text = request.output_text
                elif stop_params:
                    # ``new_text`` may be empty while the detokenizer is
                    # holding an incomplete byte sequence. Preserve the
                    # existing tail and wait for a real text segment.
                    pass
            else:
                if finish_reason != "stop" or new_text:
                    request.output_text += new_text
                    output_output_text = request.output_text
                else:
                    output_new_text = ""
                    output_output_text = request.output_text

            # Check if finished
            if finish_reason is not None:
                if (
                    not stop_trimmed
                    and stop_params
                    and request.stop_text
                    and request.stop_text_len < len(request.stop_text)
                ):
                    for stop_str in stop_params:
                        idx = request.stop_text.find(stop_str)
                        if idx != -1:
                            finish_reason = "stop"
                            output_finish_reason = finish_reason
                            visible_text = request.stop_text[:idx]
                            output_new_text = visible_text[
                                request.stop_text_len : len(visible_text)
                            ]
                            request.output_text = visible_text
                            output_output_text = visible_text
                            request.stop_text_len = len(request.stop_text)
                            request.stop_tail = ""
                            output_matched_stop = stop_str
                            stop_trimmed = True
                            break
                if (
                    not stop_trimmed
                    and stop_params
                    and request.stop_text
                    and request.stop_text_len < len(request.stop_text)
                ):
                    held_text = request.stop_text[request.stop_text_len :]
                    request.stop_text_len = len(request.stop_text)
                    output_new_text += held_text
                    request.output_text += held_text
                    output_output_text = request.output_text
                if finish_reason == "stop":
                    request.status = RequestStatus.FINISHED_STOPPED
                elif finish_reason == "length":
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED

                output_finished = True
                output_finish_reason = finish_reason
                finished_ids.add(request_id)

                # Use trimmed output if set by stop-string check, else
                # finalize streaming detokenizer for full output.
                # Use explicit flag instead of string truthiness — empty string
                # is a valid trimmed result (stop at position 0).
                if stop_trimmed or finish_reason == "stop":
                    output_output_text = request.output_text
                else:
                    detok = self._detokenizer_pool.get(request_id)
                    if detok is not None:
                        detok.finalize()
                        output_output_text = detok.text
                    else:
                        output_output_text = tokenizer.decode(request.output_tokens)
                    request.output_text = output_output_text
                request.finish_reason = finish_reason
                self._detokenizer_pool.pop(request_id, None)

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            # output_token_ids is a live reference (not a defensive copy):
            # consumers read it synchronously; the per-decode list() was O(n).
            #
            # ``logprobs`` is wired through from the MLLMBatchResponse so a
            # ``logprobs=true, top_logprobs=K`` chat request gets the same
            # per-token data the text-only AR path produces. Pre-fix the
            # MLLM path silently dropped the field — every chunk reached
            # the route with ``logprobs=None`` and the OpenAI ``choices[0].
            # logprobs`` slot serialised as ``null``. The shape matches the
            # text path's ``RequestOutput.logprobs`` field exactly so the
            # downstream ``_extract_streaming_token_logprobs`` extractor
            # works unmodified for both paths.
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    new_token_ids=[]
                    if token_is_control_stop_token
                    else [response.token],
                    new_text=output_new_text,
                    output_token_ids=request.output_tokens,
                    output_text=output_output_text,
                    finished=output_finished,
                    finish_reason=output_finish_reason,
                    prompt_tokens=request.num_prompt_tokens,
                    completion_tokens=request.num_output_tokens,
                    logprobs=getattr(response, "logprobs", None),
                    matched_stop=output_matched_stop,
                )
            )

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: set[str]) -> None:
        """Clean up finished requests."""
        for request_id in finished_ids:
            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Track as finished
            self.finished_req_ids.add(request_id)
            self.requests.pop(request_id, None)

    def _step_no_queue(self) -> MLLMSchedulerOutput:
        """Execute one scheduling step WITHOUT queue distribution.

        This is the thread-safe core of ``step()``.  It performs all
        GPU/CPU-heavy work (scheduling, vision encoding, generation)
        but does NOT touch ``self.output_queues`` (which are
        ``asyncio.Queue`` instances and not thread-safe).

        Abort requests are deferred from the event loop thread and
        processed here at the start of each step, ensuring all shared
        state mutations happen on a single thread.

        Returns:
            MLLMSchedulerOutput with results of this step.
        """
        # Process deferred aborts FIRST (same thread as all other mutations)
        self._process_pending_aborts()

        output = MLLMSchedulerOutput()

        # Schedule waiting requests
        scheduled = self._schedule_waiting()
        output.scheduled_request_ids = [r.request_id for r in scheduled]
        output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)

        # Run generation step if we have running requests
        if self.batch_generator is not None and self.running:
            try:
                responses = self.batch_generator.next()
            except (ValueError, RuntimeError) as e:
                # Oversized prompt or other unrecoverable error — fail all
                # running requests instead of retrying forever.
                err_msg = str(e)
                logger.error(f"Batch generation failed: {err_msg}")
                error_ids = set(self.running.keys())

                # Remove from batch generator BEFORE scheduler cleanup so
                # stale requests don't poison subsequent batches.
                if self.batch_generator is not None:
                    uids_to_remove = [
                        self.request_id_to_uid[rid]
                        for rid in error_ids
                        if rid in self.request_id_to_uid
                    ]
                    if uids_to_remove:
                        self.batch_generator.remove(uids_to_remove)

                # Differentiate CLIENT errors (image/video fetch failures,
                # oversized prompt) from SERVER errors (runtime crash).
                # Client errors get a non-None ``error`` field so
                # ``stream_outputs`` can raise — letting the route layer
                # convert to HTTP 400 instead of the previous silent
                # 200+empty-content+finish_reason=length pattern that
                # caused #457 (Anthropic SDK clients + curl saw a 200 OK
                # with no signal that the image fetch had failed) and
                # #682 (Desktop users sending 1920×1080 screenshots to a
                # VLM saw an empty assistant message + "Reached max_tokens
                # before any output" rendered by the client).
                #
                # The "per-batch cap" string is the marker raised by
                # ``mllm_batch_generator._process_prompts`` when prompt
                # tokens (vision + text) exceed the configured cap. For
                # VLM the typical trigger is a high-resolution image; the
                # error message already tells the user to downscale or
                # raise --prefill-step-size, so surfacing the message
                # is strictly more informative than the legacy soft
                # truncation.
                is_client_error = (
                    "Failed to process image" in err_msg
                    or "Failed to process video" in err_msg
                    or "exceeds the per-batch cap" in err_msg
                )
                # Create error outputs (queue delivery deferred to caller).
                for request_id in error_ids:
                    output.outputs.append(
                        RequestOutput(
                            request_id=request_id,
                            output_text="",
                            finished=True,
                            error=err_msg if is_client_error else None,
                            finish_reason="error" if is_client_error else "length",
                        )
                    )
                output.finished_request_ids = error_ids
                self._cleanup_finished(error_ids)
                return output

            output.has_work = True

            if responses:
                outputs, finished_ids = self._process_batch_responses(responses)
                output.outputs = outputs
                output.finished_request_ids = finished_ids

                self._cleanup_finished(finished_ids)
                if finished_ids:
                    mx.clear_cache()

        # Adaptive periodic cache clear: scale inversely with concurrency
        # to prevent Metal buffer pool growth during long generations
        active_seqs = len(self.running)
        min_interval = max(4, self._clear_cache_interval // 4)
        effective_interval = max(
            min_interval, self._clear_cache_interval // max(1, active_seqs // 8)
        )

        self._step_count += 1
        if self._step_count % effective_interval == 0:
            mx.clear_cache()

        # Clear finished tracking for next step
        self.finished_req_ids = set()

        return output

    def _distribute_outputs(self, output: MLLMSchedulerOutput) -> None:
        """Push step outputs and abort signals to async queues.

        MUST be called on the event loop thread (asyncio.Queue is not
        thread-safe).
        """
        for req_output in output.outputs:
            queue = self.output_queues.get(req_output.request_id)
            if queue is not None:
                try:
                    queue.put_nowait(req_output)
                    if req_output.finished:
                        queue.put_nowait(None)  # Signal end
                except asyncio.QueueFull:
                    pass

        # Signal queues for requests aborted during this step
        while self._aborted_queue_ids:
            request_id = self._aborted_queue_ids.pop()
            queue = self.output_queues.get(request_id)
            if queue is not None:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    def step(self) -> MLLMSchedulerOutput:
        """
        Execute one scheduling step (includes queue distribution).

        Convenience wrapper that calls ``_step_no_queue`` followed by
        ``_distribute_outputs``.  Safe to call from the event loop
        thread (the original sync API).

        Returns:
            MLLMSchedulerOutput with results of this step
        """
        output = self._step_no_queue()
        self._distribute_outputs(output)
        return output

    def get_request(self, request_id: str) -> MLLMRequest | None:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> MLLMRequest | None:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    # ========== Async API (for streaming) ==========

    async def start(self) -> None:
        """Start the async scheduler processing loop."""
        if self._running:
            return

        self._running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        logger.info(
            f"MLLM Scheduler started with max_num_seqs={self.config.max_num_seqs}"
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self.batch_generator is not None:
            self.batch_generator.close()
            self.batch_generator = None

        # Shut down the step executor to avoid leaking worker threads.
        # Only shut down if we own it — caller-supplied executors stay alive.
        if self._step_executor is not None:
            if getattr(self, "_owns_step_executor", True):
                self._step_executor.shutdown(wait=False)
            self._step_executor = None

        logger.info("MLLM Scheduler stopped")

    async def _process_loop(self) -> None:
        """Main async processing loop.

        Every step (prefill *and* generation) runs on the dedicated
        ``mllm-step`` worker. mlx-lm 0.31.3+ tags every ``mx.array`` with
        the calling thread's default stream, and ``BatchGenerator`` keeps
        KV state across calls — splitting prefill (worker) and decode
        (loop thread) means the next ``batch_generator.next()`` from the
        loop thread crashes with "There is no Stream(gpu, N) in current
        thread". Same bug class as #170 / PR #173 / #174 / #182.

        Queue distribution always happens on the event loop thread to
        avoid thread-safety issues with asyncio.Queue.

        Thread safety note: ``add_request()`` mutates ``self.requests``
        (dict) and ``self.waiting`` (deque) from the event loop thread
        while ``_step_no_queue()`` reads/pops them on the executor
        thread.  Under CPython, ``dict.__setitem__`` and
        ``deque.append``/``deque.popleft`` are atomic (protected by
        the GIL), so these concurrent accesses are safe.  Abort
        requests are fully deferred via ``_pending_abort_ids`` to
        avoid compound mutations across threads.
        """
        import concurrent.futures

        # Reuse the executor that loaded the model (so step calls hit the
        # same thread the model arrays are tagged with). Only fall back to
        # a fresh executor when no caller-supplied executor exists — that
        # path will hit Stream(gpu, N) on the first batch_generator.next()
        # under mlx-lm 0.31.3+, but it preserves the legacy behavior for
        # any sync test/CLI code path that constructs MLLMScheduler directly.
        if self._injected_step_executor is not None:
            self._step_executor = self._injected_step_executor
            self._owns_step_executor = False
        else:
            self._step_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mllm-step"
            )
            self._owns_step_executor = True
        loop = asyncio.get_running_loop()
        # The currently in-flight step ``concurrent.futures.Future``
        # (None when the loop is between steps). Lets the ``finally``
        # block wait on THIS specific future with a bounded timeout
        # instead of issuing a blocking ``shutdown(wait=True)`` second
        # join that no asyncio cancel can unblock (codex r3 BLOCKING #1).
        self._inflight_step_cf: concurrent.futures.Future | None = None

        try:
            while self._running:
                try:
                    if self.has_requests():
                        # MEMORY guideline (knowledge/gotchas.md):
                        # "asyncio Future cancel does NOT stop executor
                        # thread — use ``executor.submit`` +
                        # ``cf.cancelled()`` gate, not ``run_in_executor``".
                        #
                        # The prior shape was
                        # ``await loop.run_in_executor(self._step_executor,
                        # self._step_no_queue)``. On loop cancellation
                        # (``self._running`` flipped or task cancelled
                        # during shutdown) the asyncio-side Future flipped
                        # to CANCELLED immediately but the executor thread
                        # kept running ``_step_no_queue`` against
                        # ``BatchGenerator`` state that the shutdown path
                        # then races to tear down — exactly the
                        # "Aborting orphaned MLLM request" / mllm-step
                        # zombie shape Ana flagged in the C-04 recon
                        # (R3 in /tmp/dogfood-085/c04-recon.md).
                        #
                        # Mirror the proven pattern from
                        # ``engine_core.py:855``: hold the underlying
                        # ``concurrent.futures.Future`` directly, await it
                        # via ``asyncio.wrap_future``, and gate any
                        # post-cancel cleanup on ``cf.cancelled()`` so we
                        # only ever consume an ``output`` that actually
                        # came back from the executor thread.
                        # Codex r8 BLOCKING #1: ``submit()`` itself can
                        # raise synchronously if the executor was
                        # already shut down (e.g. shutdown raced ahead
                        # of this call). Guard the submit so
                        # ``_inflight_step_cf`` is never left dangling
                        # and the scheduler loop breaks cleanly instead
                        # of retrying against a dead executor.
                        try:
                            cf = self._step_executor.submit(self._step_no_queue)
                        except RuntimeError as _submit_exc:
                            logger.warning(
                                "MLLM scheduler executor rejected new work "
                                "(%s); breaking step loop for clean shutdown",
                                _submit_exc,
                            )
                            self._inflight_step_cf = None
                            break
                        # Stash so the ``finally`` block can wait on
                        # THIS specific in-flight cf with a bounded
                        # timeout instead of starting an
                        # uncancellable ``shutdown(wait=True)`` second
                        # join (codex r3 BLOCKING #1). The reference is
                        # cleared on the success path below; the cancel
                        # path preserves it for the outer
                        # ``finally``-block drain; the non-cancel
                        # ``Exception`` path is handled by the explicit
                        # ``except Exception`` arm a few lines down
                        # which clears it before re-raising (codex r6
                        # BLOCKING #2).
                        self._inflight_step_cf = cf
                        try:
                            output = await asyncio.wrap_future(cf, loop=loop)
                        except asyncio.CancelledError:
                            # The asyncio side is cancelled; the executor
                            # may already be running (or completed).
                            # ``asyncio.wrap_future`` will have called
                            # ``cf.cancel()`` — succeeds only if the work
                            # had not started yet. If the work DID start,
                            # let it run to completion silently so it
                            # doesn't race the shutdown teardown of
                            # ``BatchGenerator``/``self._step_executor``.
                            # We deliberately do NOT call
                            # ``cf.result()`` here because the outer
                            # ``finally`` block will wait on
                            # ``self._inflight_step_cf`` with a
                            # bounded timeout (codex r3 BLOCKING #1
                            # follow-up) and then ``shutdown(wait=False,
                            # cancel_futures=True)`` the executor. The
                            # drain done-callback below logs any
                            # executor-side exception under DEBUG.
                            if not cf.cancelled():

                                def _drain_step_result(_future: Any) -> None:
                                    # Surface any executor-side
                                    # exception at DEBUG so silent
                                    # errors during shutdown still
                                    # leave a trail without spamming
                                    # the log on normal cancel.
                                    #
                                    # Codex r5 BLOCKING #1: catch
                                    # ``Exception`` (not
                                    # ``BaseException``). Letting
                                    # ``KeyboardInterrupt`` /
                                    # ``SystemExit`` / ``GeneratorExit``
                                    # propagate is the correct
                                    # behaviour during shutdown — the
                                    # callback runs on the executor
                                    # thread, where those exception
                                    # types signal interpreter-level
                                    # teardown that nothing in this
                                    # path should be swallowing.
                                    try:
                                        _future.result()
                                    except Exception as _exc:
                                        logger.debug(
                                            "MLLM step exception during"
                                            " cancellation drain: %r",
                                            _exc,
                                        )

                                cf.add_done_callback(_drain_step_result)
                            raise
                        except Exception:
                            # Codex r6 BLOCKING #2: a non-cancel
                            # ``Exception`` raised by ``_step_no_queue``
                            # (executor-side) propagates through
                            # ``wrap_future``. Clear ``_inflight_step_cf``
                            # before re-raising so the outer
                            # ``except Exception`` arm at the loop level
                            # (which logs + retries) doesn't leave a
                            # done()-but-still-recorded reference for
                            # the eventual ``finally`` block to chase.
                            # The cf is already done() at this point,
                            # so the outer-finally drain would no-op
                            # against it, but clearing here keeps the
                            # contract clean: ``_inflight_step_cf`` is
                            # non-None only when there's actually
                            # outstanding executor work that the
                            # shutdown path might need to drain.
                            self._inflight_step_cf = None
                            raise

                        # Successful step → clear the inflight reference
                        # so the ``finally`` block sees an empty slot
                        # once the loop has exited the
                        # ``has_requests()`` branch on this iteration.
                        self._inflight_step_cf = None

                        # Distribute outputs to queues ON the event loop thread
                        # (asyncio.Queue is not thread-safe).
                        if output is not None:
                            self._distribute_outputs(output)

                        # Yield to other tasks
                        await asyncio.sleep(0)
                    else:
                        # No work, wait a bit
                        await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in MLLM process loop: {e}")
                    await asyncio.sleep(0.1)
        finally:
            cancel_to_reraise: asyncio.CancelledError | None = None
            if self._step_executor is not None:
                if getattr(self, "_owns_step_executor", True):
                    # Codex r1 BLOCKING #3 + codex r2/r3 BLOCKING follow-ups:
                    # bound the teardown WITHOUT relying on
                    # ``shutdown(wait=True)`` (an uncancellable blocking
                    # join that no asyncio timeout can stop). Instead
                    # wait on THIS step's specific ``cf`` via
                    # ``asyncio.wrap_future`` with a bounded timeout,
                    # then drop the executor reference with
                    # ``shutdown(wait=False)`` regardless of outcome:
                    #
                    #   * Happy path — the in-flight step (if any)
                    #     completes within ``_drain_secs``. ``cf``
                    #     finishes, ``BatchGenerator``/``scheduler``
                    #     state was mutated by the step EXACTLY ONCE,
                    #     and the subsequent ``shutdown(wait=False)``
                    #     only has the (empty) submit queue left to
                    #     drain. The "Aborting orphaned MLLM request"
                    #     race C-04 §3.R3 flagged is closed.
                    #   * Wedged path — the in-flight step is stuck
                    #     (Metal driver hang, etc.). The
                    #     ``asyncio.wait_for`` times out after
                    #     ``_drain_secs``, we log a WARNING, and the
                    #     follow-on ``shutdown(wait=False,
                    #     cancel_futures=True)`` releases the executor
                    #     reference. The worker thread is left to its
                    #     wedged state (we can't unwedge it from
                    #     Python), but lifespan shutdown makes
                    #     progress — exactly the bounded-shutdown
                    #     guarantee codex r3 BLOCKING #1 demanded.
                    #   * No in-flight step — the wait_for is a no-op
                    #     against an already-resolved (or never-set)
                    #     future.
                    _drain_secs = 5.0
                    inflight = self._inflight_step_cf
                    self._inflight_step_cf = None
                    # Codex r5 BLOCKING #2: split exception handling so
                    # ``asyncio.CancelledError`` propagates after we
                    # release the executor reference. A second-cancel
                    # storm (caller invoked ``stop()`` mid-shutdown
                    # and the lifespan task got cancelled again) MUST
                    # surface back to the caller — swallowing it leaves
                    # ``stop()`` blocked on the unfinished
                    # ``_processing_task``. ``cancel_to_reraise`` is
                    # declared at the top of the ``finally`` block so
                    # the re-raise sits outside the
                    # ``_owns_step_executor`` branch.
                    if (
                        inflight is not None
                        and not inflight.done()
                        and not inflight.cancelled()
                    ):
                        # Codex r8 BLOCKING #2: ``asyncio.wait_for``
                        # cancels the awaitable on timeout, and
                        # cancelling an ``asyncio.wrap_future``
                        # propagates ``cancel()`` to the underlying
                        # ``concurrent.futures.Future``. For a
                        # step that's queued-but-not-started, that
                        # cancel succeeds and discards the work —
                        # violating the surrounding comment's
                        # contract ("the step either finishes once
                        # or is abandoned only when wedged"). The
                        # observed silent-data-loss shape would be:
                        # a final scheduled step is queued; lifespan
                        # shutdown fires; the timeout cancels the
                        # CF; the step never runs; an inflight
                        # generation completes with truncated
                        # output because its kv-fetch never
                        # happened. ``asyncio.shield`` prevents the
                        # timeout's cancellation from propagating
                        # to the inner ``wrap_future`` — the
                        # awaitable is cancelled at the wait_for
                        # boundary but the wrapped CF is left
                        # alone. The shutdown still proceeds in
                        # bounded time (drain returns or
                        # TimeoutError fires), and the executor
                        # is torn down with ``wait=False`` below
                        # so the worker thread is released
                        # regardless.
                        wrapped = asyncio.wrap_future(inflight, loop=loop)
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(wrapped),
                                timeout=_drain_secs,
                            )
                        except TimeoutError:
                            # Wedged step — operator-visible WARNING.
                            logger.warning(
                                "MLLM step exceeded %.1fs drain budget"
                                " during shutdown; abandoning the"
                                " worker thread and proceeding with"
                                " non-blocking executor teardown",
                                _drain_secs,
                            )
                        except asyncio.CancelledError as exc:
                            # Re-raise AFTER releasing the executor
                            # reference below (the executor cleanup is
                            # non-blocking, so we can do it on the
                            # exit path without losing the cancel
                            # signal to the caller).
                            cancel_to_reraise = exc
                        except Exception as exc:
                            # Executor-side raise — log only; this
                            # path is best-effort observability and
                            # must not mask the real shutdown trigger.
                            logger.debug("MLLM step in-flight drain ended with %r", exc)
                    try:
                        self._step_executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:  # pragma: no cover — defensive
                        logger.debug(
                            "MLLM step executor shutdown raised", exc_info=True
                        )
                self._step_executor = None
            if cancel_to_reraise is not None:
                # Surface the cancellation to whatever is awaiting
                # ``_processing_task`` (typically
                # ``MLLMScheduler.stop`` -> outer FastAPI lifespan).
                # We've already cleaned up the executor reference, so
                # the re-raise leaves no resource leak.
                raise cancel_to_reraise

    async def add_request_async(
        self,
        prompt: str,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        video_fps: float | None = None,
        video_max_frames: int | None = None,
        **kwargs,
    ) -> str:
        """
        Add a multimodal request (async version with output queue).

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Text-based stop sequences
            video_fps: FPS for video frame extraction
            video_max_frames: Max frames to extract from video
            **kwargs: Additional parameters

        Returns:
            Request ID for tracking
        """
        request_id = self.add_request(
            prompt=prompt,
            images=images,
            videos=videos,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            video_fps=video_fps,
            video_max_frames=video_max_frames,
            **kwargs,
        )

        # Create output queue for streaming
        self.output_queues[request_id] = asyncio.Queue()

        return request_id

    async def stream_outputs(
        self,
        request_id: str,
    ) -> AsyncIterator[RequestOutput]:
        """
        Stream outputs for a request.

        Args:
            request_id: The request ID to stream

        Yields:
            RequestOutput objects as tokens are generated
        """
        output_queue = self.output_queues.get(request_id)
        if output_queue is None:
            return

        finished_normally = False
        try:
            while True:
                output = await output_queue.get()
                if output is None:
                    finished_normally = True
                    break
                if output.error:
                    # Surface scheduler-side client errors (image/video
                    # fetch failures, etc.) as exceptions so the route
                    # layer can map to a meaningful HTTP status (#457).
                    # Mark finished BEFORE raising so the finally block
                    # doesn't double-abort what's already cleaned up.
                    finished_normally = True
                    raise ValueError(output.error)
                yield output
                if output.finished:
                    finished_normally = True
                    break
        finally:
            if not finished_normally:
                logger.info(f"Aborting orphaned MLLM request {request_id}")
                self.abort_request(request_id)
            # Cleanup queue
            if request_id in self.output_queues:
                del self.output_queues[request_id]

    async def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate complete output for a request (non-streaming).

        Args:
            prompt: Text prompt
            images: Image inputs
            videos: Video inputs
            **kwargs: Generation parameters

        Returns:
            Final RequestOutput
        """
        request_id = await self.add_request_async(
            prompt=prompt,
            images=images,
            videos=videos,
            **kwargs,
        )

        # Collect all outputs
        final_output = None
        async for output in self.stream_outputs(request_id):
            final_output = output
            if output.finished:
                break

        if final_output is None:
            # Create empty output on error. finish_reason="length" keeps
            # the response OpenAI-spec-compliant; see scheduler.py rationale.
            final_output = RequestOutput(
                request_id=request_id,
                output_text="",
                finished=True,
                finish_reason="length",
            )

        # Cleanup
        if request_id in self.requests:
            del self.requests[request_id]

        return final_output

    # ========== Stats and utilities ==========

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_finished": len(self.finished_req_ids),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            # M-01: cancellation observability — mirror of the text
            # scheduler stats so the same /metrics renderer surfaces a
            # flat-line zero on MLLM-active engines that never see an
            # abort. See ``Scheduler.get_stats`` for the full rationale.
            "num_requests_cancelled": self.num_requests_cancelled,
            "num_requests_cancelled_via_disconnect": (
                self.num_requests_cancelled_via_disconnect
            ),
        }

        if self.batch_generator is not None:
            batch_stats = self.batch_generator.stats()
            stats["batch_generator"] = batch_stats.to_dict()
            # Add vision embedding cache stats from batch generator
            stats["vision_embedding_cache"] = (
                self.batch_generator.get_vision_cache_stats()
            )

        if self.vision_cache:
            stats["vision_cache"] = self.vision_cache.get_stats()

        # Include Metal memory stats
        try:
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    def reset(self) -> None:
        """Reset the scheduler state."""
        # Abort all requests
        for request_id in list(self.requests.keys()):
            self.abort_request(request_id)

        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._detokenizer_pool.clear()
        # M-01 codex r2/r8: drop the cancellation lifetime ledgers
        # alongside the in-flight state. The Prometheus counters
        # themselves are NOT zeroed (lifetime-cumulative contract).
        # Codex r8 BLOCKING #1 ordering: clear AFTER the abort loop
        # so a concurrent ``record_disconnect_abort`` for an
        # in-flight request can't interleave inconsistently, AND
        # under the lock so the clear is atomic against any
        # concurrent abort-path mutation.
        with self._cancel_counter_lock:
            self._cancelled_request_ids.clear()
            self._disconnect_abort_ids.clear()

        if self.batch_generator is not None:
            self.batch_generator.close()
            self.batch_generator = None

        if self.vision_cache:
            self.vision_cache.clear()
