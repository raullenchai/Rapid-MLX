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
import time
import uuid

import mlx.core as mx
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple

from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

from .mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
)
from .mlx_streams import bind_generation_streams
from .multimodal_processor import MultimodalProcessor
from .request import RequestOutput, RequestStatus, SamplingParams

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
    # Prefill step size for chunked prefill
    prefill_step_size: int = 1024
    # Enable vision embedding cache
    enable_vision_cache: bool = True
    # Maximum cache entries
    vision_cache_size: int = 100
    # Default max tokens
    default_max_tokens: int = 256
    # Default video FPS for frame extraction
    default_video_fps: float = 2.0
    # KV cache memory limit (from --cache-memory-mb)
    cache_memory_mb: Optional[int] = None
    # Maximum video frames
    max_video_frames: int = 128
    # Enable MTP speculative decoding
    enable_mtp: bool = False
    # Number of draft tokens for MTP
    mtp_num_draft_tokens: int = 1
    # Enable KV prefix cache for text-only requests
    enable_prefix_cache: bool = True
    # MLLM supports the memory-aware prefix cache only; if disabled, prefix
    # caching is disabled rather than silently falling back to another cache.
    use_memory_aware_cache: bool = True
    # Memory limit for prefix cache (None = auto-detect)
    prefix_cache_memory_mb: Optional[int] = None
    # KV cache quantization for prefix cache store/fetch
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    # Interleaved prefill/decode budget per step (0 = disabled, blocking prefill)
    chunked_prefill_tokens: int = 0
    # Maximum KV cache size per sequence (0 = unbounded; >0 enables RotatingKVCache)
    max_kv_size: int = 0


@dataclass
class MLLMRequest:
    """
    Extended request for MLLM processing.

    Includes all multimodal data needed for generation.
    """

    request_id: str
    prompt: str
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    audio: Optional[List[str]] = None
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.time)

    # Batch generator UID (assigned when scheduled)
    batch_uid: Optional[int] = None

    # Status tracking
    status: RequestStatus = RequestStatus.WAITING
    output_text: str = ""
    output_tokens: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None

    # Token counts
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    mtp_drafts: int = 0
    mtp_accepted: int = 0

    # Timing
    first_token_time: Optional[float] = None


@dataclass
class MLLMSchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: List[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: Set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: List[RequestOutput] = field(default_factory=list)
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
        config: Optional[MLLMSchedulerConfig] = None,
    ):
        """
        Initialize MLLM scheduler.

        Args:
            model: The VLM model
            processor: The VLM processor
            config: Scheduler configuration
        """
        self.model = model
        self.processor = processor
        self.config = config or MLLMSchedulerConfig()

        # Get model config
        self.model_config = getattr(model, "config", None)

        # Multimodal processor for input preparation
        self.mm_processor = MultimodalProcessor(
            model=model,
            processor=processor,
            config=self.model_config,
        )

        # Get stop tokens from tokenizer
        self.stop_tokens = self._get_stop_tokens()

        # Batch generator (created lazily)
        self.batch_generator: Optional[MLLMBatchGenerator] = None

        # Request management - following vLLM's design
        self.waiting: deque[MLLMRequest] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, MLLMRequest] = {}  # Running requests by ID
        self.requests: Dict[str, MLLMRequest] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # Per-request streaming detokenizers for UTF-8-safe incremental decode
        self._detokenizer_pool: Dict[str, Any] = {}

        # Output queues for async streaming
        self.output_queues: Dict[str, asyncio.Queue] = {}

        # Async processing control
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None

        # Memory management: periodic mx.clear_cache() to free Metal buffer pool
        self._step_count = 0
        self._clear_cache_interval = 32

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Memory management: periodic mx.clear_cache() to free Metal buffers
        self._step_count = 0
        self._clear_cache_interval = 32

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer and generation_config.json."""
        stop_tokens = set()
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )

        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                stop_tokens.update(tokenizer.eos_token_id)
            else:
                stop_tokens.add(tokenizer.eos_token_id)

        if hasattr(tokenizer, "eos_token_ids") and tokenizer.eos_token_ids is not None:
            if isinstance(tokenizer.eos_token_ids, (list, set, tuple)):
                stop_tokens.update(tokenizer.eos_token_ids)
            else:
                stop_tokens.add(tokenizer.eos_token_ids)

        # Also read generation_config.json which may have additional EOS tokens
        # (e.g., Gemma 4 has <turn|>=106, <|tool_response>=50 as EOS)
        model_path = getattr(tokenizer, "name_or_path", None)
        if model_path:
            import json
            from pathlib import Path

            gc_path = Path(model_path) / "generation_config.json"
            if gc_path.exists():
                try:
                    gc = json.loads(gc_path.read_text())
                    gc_eos = gc.get("eos_token_id")
                    if isinstance(gc_eos, list):
                        stop_tokens.update(gc_eos)
                    elif gc_eos is not None:
                        stop_tokens.add(gc_eos)
                except Exception:
                    pass

        return stop_tokens

    def _ensure_batch_generator(self) -> None:
        """Ensure batch generator exists."""
        if self.batch_generator is None:
            from mlx_lm.sample_utils import make_sampler

            from .memory_cache import MemoryCacheConfig

            # Default sampler (can be overridden per-request in future)
            sampler = make_sampler(temp=0.7, top_p=0.9)

            # Configure KV prefix cache for text-only requests
            # KV cache quantization reduces prefix cache memory ~4x (BF16→Q8).
            # Quantization happens on store(), dequantization on fetch() —
            # the model always receives normal KVCache with plain arrays.
            prefix_cache_config = None
            if self.config.enable_prefix_cache and self.config.use_memory_aware_cache:
                prefix_cache_config = MemoryCacheConfig(
                    max_memory_mb=self.config.prefix_cache_memory_mb,
                    kv_quantize=self.config.kv_cache_quantization,
                    kv_bits=self.config.kv_cache_quantization_bits,
                    kv_group_size=self.config.kv_cache_quantization_group_size,
                )

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
                prefix_cache_config=prefix_cache_config,
                max_kv_size=self.config.max_kv_size,
            )

            # Install chunked prefill BEFORE MTP (MTP wraps _next,
            # chunked replaces it — MTP then wraps the chunked version)
            if self.config.chunked_prefill_tokens > 0:
                from .mllm_batch_generator import install_chunked_prefill_mllm

                install_chunked_prefill_mllm(
                    self.batch_generator,
                    budget=self.config.chunked_prefill_tokens,
                )

            # Install MTP if enabled and language model supports it
            if self.config.enable_mtp:
                lm = self.batch_generator.language_model
                if hasattr(lm, "mtp") and lm.mtp is not None:
                    from .mllm_batch_generator import install_mtp_mllm

                    install_mtp_mllm(
                        self.batch_generator,
                        lm,
                        num_draft_tokens=self.config.mtp_num_draft_tokens,
                    )

    # ========== Sync API (step-based) ==========

    def add_request(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Add a multimodal request to the scheduler (sync version).

        Args:
            prompt: Text prompt (should be formatted with chat template)
            images: List of image inputs (paths, URLs, base64)
            videos: List of video inputs
            audio: List of audio inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            request_id: Optional custom request ID
            **kwargs: Additional generation parameters.  ``logits_processors``
                — list of callables ``(tokens, logits) -> logits`` applied
                during sampling (e.g. constrained JSON decoding).

        Returns:
            Request ID for tracking
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=kwargs.pop("top_k", 0),
            min_p=kwargs.pop("min_p", 0.0),
            presence_penalty=kwargs.pop("presence_penalty", 0.0),
            repetition_penalty=kwargs.pop("repetition_penalty", 1.0),
            stop=kwargs.pop("stop", None) or [],
            logits_processors=kwargs.pop("logits_processors", None),
        )

        request = MLLMRequest(
            request_id=request_id,
            prompt=prompt,
            images=images,
            videos=videos,
            audio=audio,
            sampling_params=sampling_params,
        )

        # Estimate prompt token count for monitoring (text tokens only;
        # vision tokens are added during prefill but this gives a useful
        # approximation for the status endpoint).
        tokenizer = (
            self.processor.tokenizer
            if hasattr(self.processor, "tokenizer")
            else self.processor
        )
        try:
            request.num_prompt_tokens = len(tokenizer.encode(prompt))
        except Exception:
            pass

        self.requests[request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added MLLM request {request_id}: "
            f"{len(images or [])} images, {len(videos or [])} videos"
        )

        return request_id

    def abort_request(self, request_id: str) -> bool:
        """
        Abort a request.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted
        """
        request = self.requests.get(request_id)
        if request is None:
            return False

        # Signal batch generator to abort any in-progress prefill for this
        # request.  The prefill loop checks _aborted_request_ids between
        # chunks and raises PrefillAbortedError to exit early.
        if self.batch_generator is not None:
            self.batch_generator.abort_prefill(request_id)

        # Remove from waiting queue
        if request.status == RequestStatus.WAITING:
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from batch generator.
        #
        # IMPORTANT: `abort_request` may be called from the asyncio event
        # loop (e.g. in `stream_outputs`' `finally` block on client
        # disconnect) while `scheduler.step()` — and therefore the
        # batch generator's forward pass — is running on a separate
        # executor thread (see engine_core.py: loop.run_in_executor).
        #
        # Calling `batch_generator.remove([uid])` eagerly here would
        # trigger `active_batch.filter(...)`, which creates an
        # `mx.array` and submits Metal work.  If the scheduler thread
        # has an open Metal encoder mid-forward-pass, two threads
        # submit to the same stream concurrently and Metal asserts
        # with ``encodeSignalEvent:value: with uncommitted encoder``,
        # aborting the process.
        #
        # Instead we defer the removal to the scheduler thread: it
        # will drain the queue at the next safe boundary (start of
        # step(), before any forward pass).
        if request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.schedule_removal([uid])
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Credit in-flight tokens so dashboard metrics stay accurate
        # (without this, aborted requests' tokens vanish from /v1/status).
        if request.num_output_tokens > 0:
            self.total_completion_tokens += request.num_output_tokens
            self.total_prompt_tokens += request.num_prompt_tokens

        # Mark as aborted
        request.status = RequestStatus.FINISHED_ABORTED
        self.finished_req_ids.add(request_id)
        self.requests.pop(request_id, None)

        self._detokenizer_pool.pop(request_id, None)

        # Signal output queue
        if request_id in self.output_queues:
            try:
                self.output_queues[request_id].put_nowait(None)
            except asyncio.QueueFull:
                pass

        logger.debug(f"Aborted request {request_id}")
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _schedule_waiting(self) -> List[MLLMRequest]:
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
                audio=request.audio,
                max_tokens=request.sampling_params.max_tokens,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                top_k=request.sampling_params.top_k,
                min_p=request.sampling_params.min_p,
                presence_penalty=request.sampling_params.presence_penalty,
                repetition_penalty=request.sampling_params.repetition_penalty,
                logits_processors=request.sampling_params.logits_processors,
            )
            batch_requests.append(batch_req)

            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
            scheduled.append(request)

            self.total_prompt_tokens += request.num_prompt_tokens

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
        self, responses: List[MLLMBatchResponse]
    ) -> Tuple[List[RequestOutput], Set[str]]:
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

            # Handle error responses from failed preprocessing
            if response.finish_reason == "error":
                output = RequestOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    new_text="",
                    output_token_ids=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    finished=True,
                    finish_reason="error",
                )
                request.status = RequestStatus.FINISHED_ABORTED
                request.output_text = ""
                request.finish_reason = "error"
                finished_ids.add(request_id)
                self.num_requests_processed += 1
                logger.warning(f"Request {request_id} failed during preprocessing")
                outputs.append(output)
                continue

            # Append token to request
            request.output_tokens.append(response.token)
            request.num_output_tokens = len(request.output_tokens)
            if response.mtp_attempted:
                request.mtp_drafts += response.mtp_attempted_count
            if response.from_draft:
                request.mtp_accepted += 1

            if request.first_token_time is None and request.num_output_tokens > 0:
                request.first_token_time = time.time()

            # Decode the new token using streaming detokenizer (UTF-8 safe).
            # Skip stop tokens — they are not content.
            if response.finish_reason == "stop":
                new_text = ""
            else:
                if request_id not in self._detokenizer_pool:
                    detok = NaiveStreamingDetokenizer(tokenizer)
                    self._detokenizer_pool[request_id] = detok
                detok = self._detokenizer_pool[request_id]
                detok.add_token(response.token)
                new_text = detok.last_segment

            # Enforce SamplingParams.stop on the cumulative output text
            # (issue #469).  Mirrors the LLM scheduler path: maintain a
            # running buffer only when stop strings are present, scan for
            # the earliest match, truncate, and schedule the request's uid
            # for removal from the active batch.
            matched_stop: Optional[str] = None
            if request.sampling_params.stop and response.finish_reason is None:
                if new_text:
                    request.output_text = (request.output_text or "") + new_text
                buf = request.output_text or ""
                if buf:
                    best_idx: Optional[int] = None
                    for cand in request.sampling_params.stop:
                        if not cand:
                            continue
                        idx = buf.find(cand)
                        if idx == -1:
                            continue
                        if best_idx is None or idx < best_idx or (
                            idx == best_idx and len(cand) > len(matched_stop or "")
                        ):
                            best_idx = idx
                            matched_stop = cand
                    if matched_stop is not None and best_idx is not None:
                        request.output_text = buf[:best_idx]
                        if request_id in self.request_id_to_uid:
                            uid = self.request_id_to_uid[request_id]
                            if self.batch_generator is not None:
                                try:
                                    self.batch_generator.schedule_removal([uid])
                                except Exception as exc:  # pragma: no cover
                                    logger.debug(
                                        "[stop_seq][mllm] schedule_removal "
                                        "failed for uid=%s req=%s: %s",
                                        uid,
                                        request_id[:12],
                                        exc,
                                    )
                            self.uid_to_request_id.pop(uid, None)
                            self.request_id_to_uid.pop(request_id, None)
                        try:
                            response.finish_reason = "stop"
                        except (AttributeError, TypeError):  # pragma: no cover
                            pass

            # Create output
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=request.output_tokens,
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                mtp_drafts=request.mtp_drafts,
                mtp_accepted=request.mtp_accepted,
                matched_stop=matched_stop,
            )

            # Check if finished
            if response.finish_reason is not None:
                if response.finish_reason == "stop":
                    request.status = RequestStatus.FINISHED_STOPPED
                elif response.finish_reason == "length":
                    request.status = RequestStatus.FINISHED_LENGTH_CAPPED

                output.finished = True
                output.finish_reason = response.finish_reason
                output.matched_stop = matched_stop
                finished_ids.add(request_id)

                # Finalize streaming detokenizer and get full output.  When
                # a client stop_sequence triggered early termination, keep
                # our pre-truncated buffer instead of the full detokenizer
                # text (which still contains the matched suffix).
                detok = self._detokenizer_pool.pop(request_id, None)
                if matched_stop is not None:
                    output.output_text = request.output_text or ""
                elif detok is not None:
                    detok.finalize()
                    output.output_text = detok.text
                else:
                    output.output_text = tokenizer.decode(request.output_tokens)
                request.output_text = output.output_text
                request.finish_reason = response.finish_reason

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests."""
        for request_id in finished_ids:
            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Drain from requests dict to prevent linear memory growth
            self.requests.pop(request_id, None)

            # Remove UID mappings
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Clean up detokenizer pool (handles abort/timeout cases)
            self._detokenizer_pool.pop(request_id, None)

            # Track as finished
            self.finished_req_ids.add(request_id)
            self.requests.pop(request_id, None)

        # Clear Metal buffer pool after cleanup to release memory
        if finished_ids:
            mx.clear_cache()

    def step(self) -> MLLMSchedulerOutput:
        """
        Execute one scheduling step.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via MLLMBatchGenerator
        3. Processes outputs and handles finished requests

        Returns:
            MLLMSchedulerOutput with results of this step
        """
        output = MLLMSchedulerOutput()

        # Drain any deferred removals queued from other threads (e.g.
        # the asyncio event loop during client-disconnect aborts).
        # This MUST run before any forward pass to avoid the Metal
        # ``encodeSignalEvent: uncommitted encoder`` race.  See
        # `abort_request` and `MLLMBatchGenerator.schedule_removal`.
        if self.batch_generator is not None:
            self.batch_generator.process_pending_removals()

        # Schedule waiting requests
        scheduled = self._schedule_waiting()
        output.scheduled_request_ids = [r.request_id for r in scheduled]
        output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)

        # Run generation step if we have running requests
        if self.batch_generator is not None and self.running:
            responses = self.batch_generator.next()
            output.has_work = True

            if responses:
                outputs, finished_ids = self._process_batch_responses(responses)
                output.outputs = outputs
                output.finished_request_ids = finished_ids

                # Push to async queues
                for req_output in outputs:
                    queue = self.output_queues.get(req_output.request_id)
                    if queue is not None:
                        try:
                            queue.put_nowait(req_output)
                            if req_output.finished:
                                queue.put_nowait(None)  # Signal end
                        except asyncio.QueueFull:
                            pass

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

    def get_request(self, request_id: str) -> Optional[MLLMRequest]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[MLLMRequest]:
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

        logger.info("MLLM Scheduler stopped")

    async def _process_loop(self) -> None:
        """Main async processing loop.

        MLLM models are loaded on the server/event-loop thread, so their MLX
        arrays and cache state must be consumed on that same thread.  Unlike
        the text-only EngineCore path, moving MLLM prefill to a worker crosses
        MLX stream ownership and can fail with "no Stream in current thread".

        Text-only preprocessing (Jinja2 template rendering + tokenization) is
        run BEFORE ``step()`` with ``await asyncio.sleep(0)`` yields between
        each request.  This prevents long preprocessing (10-30+ s for 40K+
        token conversations) from blocking health checks and new connections.
        """
        streams_bound = False

        def _ensure_streams_bound() -> None:
            nonlocal streams_bound
            if not streams_bound:
                bind_generation_streams()
                streams_bound = True

        loop = asyncio.get_running_loop()

        while self._running:
            try:
                # --- Early preprocessing phase ---
                # Run text-only preprocessing (Jinja2 template rendering +
                # tokenization) in a thread-pool executor so the event loop
                # stays responsive for health checks, new connections, and
                # active streaming requests.  Preprocessing is CPU-bound
                # (no MLX GPU work) and HuggingFace tokenizers are
                # thread-safe, so this is safe to offload.
                bg = self.batch_generator
                if bg is not None:
                    for req in list(getattr(bg, "unprocessed_requests", ())):
                        if (
                            req.input_ids is None
                            and not req.images
                            and not req.videos
                            and not req.audio
                        ):
                            try:
                                tic = time.perf_counter()
                                await loop.run_in_executor(
                                    None, bg._preprocess_request, req
                                )
                                elapsed = time.perf_counter() - tic
                                if elapsed > 1.0:
                                    n_tok = (
                                        req.input_ids.size
                                        if req.input_ids is not None
                                        else 0
                                    )
                                    logger.info(
                                        f"Preprocessing {req.request_id[:12]}"
                                        f": {n_tok} tokens in {elapsed:.2f}s"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Early preprocessing failed for "
                                    f"{req.request_id}: {e}"
                                )

                # --- Step phase ---
                if self.has_requests():
                    _ensure_streams_bound()
                    tic = time.perf_counter()
                    self.step()
                    elapsed = time.perf_counter() - tic
                    if elapsed > 2.0:
                        logger.warning(
                            f"Slow MLLM step: {elapsed:.2f}s "
                            f"(waiting={len(self.waiting)}, "
                            f"running={len(self.running)})"
                        )
                    # Yield multiple event-loop cycles so that pending
                    # HTTP health checks can complete.  A single
                    # asyncio.sleep() gives only ONE _run_once() cycle,
                    # but an HTTP request needs ~3 cycles minimum:
                    #   1. accept TCP connection
                    #   2. read HTTP request / parse headers
                    #   3. run handler / write response
                    # Using repeated asyncio.sleep(0) gives many cycles
                    # with negligible wall-clock overhead (<1ms total).
                    n_yields = 10 if elapsed > 1.0 else 5
                    for _ in range(n_yields):
                        await asyncio.sleep(0)
                else:
                    # No work, wait a bit
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in MLLM process loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def add_request_async(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Add a multimodal request (async version with output queue).

        Args:
            prompt: Text prompt
            images: List of image inputs
            videos: List of video inputs
            audio: List of audio inputs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            **kwargs: Additional parameters

        Returns:
            Request ID for tracking
        """
        request_id = self.add_request(
            prompt=prompt,
            images=images,
            videos=videos,
            audio=audio,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
                if output.finished:
                    finished_normally = True
                    yield output
                    break
                yield output
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
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audio: Optional[List[str]] = None,
        **kwargs,
    ) -> RequestOutput:
        """
        Generate complete output for a request (non-streaming).

        Args:
            prompt: Text prompt
            images: Image inputs
            videos: Video inputs
            audio: Audio inputs
            **kwargs: Generation parameters

        Returns:
            Final RequestOutput
        """
        request_id = await self.add_request_async(
            prompt=prompt,
            images=images,
            videos=videos,
            audio=audio,
            **kwargs,
        )

        # Collect all outputs
        final_output = None
        async for output in self.stream_outputs(request_id):
            final_output = output
            if output.finished:
                break

        if final_output is None:
            # Create empty output on error
            final_output = RequestOutput(
                request_id=request_id,
                output_text="",
                finished=True,
                finish_reason="error",
            )

        # Cleanup
        if request_id in self.requests:
            del self.requests[request_id]

        return final_output

    # ========== Stats and utilities ==========

    def get_running_requests_info(self) -> List[Dict[str, Any]]:
        """Per-request details for status endpoint."""
        now = time.time()
        result = []

        # Waiting requests
        for req in self.waiting:
            result.append(
                {
                    "request_id": req.request_id,
                    "status": "waiting",
                    "phase": "queued",
                    "elapsed_s": round(now - req.arrival_time, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": 0,
                    "max_tokens": req.sampling_params.max_tokens,
                    "progress": 0.0,
                    "tokens_per_second": None,
                    "ttft_s": None,
                    "cache_hit_type": None,
                    "cached_tokens": 0,
                }
            )

        # Running requests
        for req in self.running.values():
            n_out = req.num_output_tokens
            elapsed = now - req.arrival_time

            if n_out == 0:
                phase = "prefill"
            else:
                phase = "generation"

            tok_s = None
            ttft = None
            if req.first_token_time is not None:
                ttft = round(req.first_token_time - req.arrival_time, 3)
                gen_elapsed = now - req.first_token_time
                if gen_elapsed > 0 and n_out > 0:
                    tok_s = round(n_out / gen_elapsed, 1)

            max_tokens = req.sampling_params.max_tokens
            if phase == "prefill" and self.batch_generator is not None:
                pp = self.batch_generator.get_prefill_progress(req.request_id)
                if pp is not None:
                    progress = round(pp[0] / pp[1], 3) if pp[1] > 0 else 0.0
                else:
                    progress = 0.0
            else:
                progress = round(n_out / max_tokens, 3) if max_tokens > 0 else 0.0

            result.append(
                {
                    "request_id": req.request_id,
                    "status": "running",
                    "phase": phase,
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": n_out,
                    "max_tokens": max_tokens,
                    "progress": min(progress, 1.0),
                    "tokens_per_second": tok_s,
                    "ttft_s": ttft,
                    "cache_hit_type": None,
                    "cached_tokens": 0,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_finished": len(self.finished_req_ids),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "requests": self.get_running_requests_info(),
        }

        if self.batch_generator is not None:
            batch_stats = self.batch_generator.stats()
            stats["batch_generator"] = batch_stats.to_dict()
            # Vision embedding cache stats from batch generator
            vec_stats = self.batch_generator.get_vision_cache_stats()
            stats["vision_embedding_cache"] = vec_stats
            if hasattr(self.batch_generator, "get_mtp_stats"):
                stats["mtp"] = self.batch_generator.get_mtp_stats()

        # Include Metal memory stats
        try:
            if mx.metal.is_available():
                active_gb = round(mx.get_active_memory() / 1e9, 2)
                peak_gb = round(mx.get_peak_memory() / 1e9, 2)
                cache_gb = round(mx.get_cache_memory() / 1e9, 2)
                stats["metal_active_memory_gb"] = active_gb
                stats["metal_peak_memory_gb"] = peak_gb
                stats["metal_cache_memory_gb"] = cache_gb
        except Exception:
            active_gb = 0
            cache_gb = 0

        # KV prefix cache stats for /v1/status and monitoring UI.
        if self.batch_generator is not None:
            prefix_stats = self.batch_generator.get_prefix_cache_stats()
        else:
            prefix_stats = {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "evictions": 0,
                "tokens_saved": 0,
                "current_memory_mb": 0.0,
                "max_memory_mb": 0.0,
                "memory_utilization": 0.0,
                "entry_count": 0,
            }
        stats["memory_aware_cache"] = prefix_stats

        return stats

    def clear_runtime_caches(self) -> Dict[str, bool]:
        """Clear runtime caches without resetting scheduler/request state."""
        cleared = {
            "vision_cache": False,
            "prefix_cache": False,
        }
        if self.vision_cache:
            self.vision_cache.clear()
            cleared["vision_cache"] = True
        if (
            self.batch_generator is not None
            and self.batch_generator.prefix_cache is not None
        ):
            self.batch_generator.prefix_cache.clear()
            cleared["prefix_cache"] = True
        return cleared

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

        if self.batch_generator is not None:
            self.batch_generator.close()
            self.batch_generator = None

        if self.vision_cache:
            self.vision_cache.clear()
