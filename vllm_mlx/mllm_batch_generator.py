# SPDX-License-Identifier: Apache-2.0
"""
MLLM Batch Generator for multimodal continuous batching.

This module implements continuous batching for Multimodal Language Models (MLLMs)
like Qwen3-VL, following the same architecture as LLM continuous batching but
adapted for vision models.

Key insight: VLM models have a `model.language_model` which is a standard LLM.
After the initial forward pass with vision encoding, text generation uses only
the language model - which CAN be batched using the same BatchKVCache pattern.

Architecture:
1. Vision inputs are processed per-request (not batched)
2. Initial VLM forward pass extracts cross-attention states / encoder outputs
3. Language model generation is batched using BatchKVCache (like LLM batching)
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig, _trim_cache_offset
from .multimodal_processor import MultimodalProcessor
from .vision_embedding_cache import VisionEmbeddingCache

logger = logging.getLogger(__name__)


def _processors_can_retire(processors: Optional[List[Callable]]) -> bool:
    """True when any processor advertises a retire-to-content transition."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        isinstance(getattr(p, "is_retired", None), bool) for p in processors
    )


def _mark_mtp_attempts_on_primary_responses(
    responses: List["MLLMBatchResponse"],
    attempted_drafts_by_uid: Dict[int, int],
) -> None:
    """Mark only responses from steps that actually attempted MTP drafts."""
    for response in responses:
        draft_count = attempted_drafts_by_uid.pop(response.uid, 0)
        if draft_count <= 0 or response.finish_reason is not None:
            continue
        response.mtp_attempted = True
        response.mtp_attempted_count = draft_count
    attempted_drafts_by_uid.clear()


def _drop_retired_processors(
    processors: Optional[List[Callable]],
) -> tuple[Optional[List[Callable]], int]:
    """Drop retire-capable processors that have completed their work."""
    if not processors:
        return processors, 0

    remaining = []
    retired_count = 0
    for processor in processors:
        if getattr(processor, "is_retired", False) is True:
            retired_count += 1
            continue
        remaining.append(processor)
    return (remaining or None), retired_count


class PrefillAbortedError(Exception):
    """Raised when a prefill is aborted due to client disconnect."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        super().__init__(f"Prefill aborted for request {request_id}")


def _cache_eval_tensors(cache: List[Any]) -> List[Any]:
    """Return realized tensors that break lazy cache graphs between chunks."""
    tensors: List[Any] = []
    for c in cache:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        if keys is not None or values is not None:
            if keys is not None:
                tensors.append(keys)
            if values is not None:
                tensors.append(values)
            continue

        try:
            state = getattr(c, "state", None)
        except AttributeError:
            state = None
        if state is None:
            continue
        if isinstance(state, (list, tuple)):
            tensors.extend(s for s in state if s is not None)
        else:
            tensors.append(state)
    return tensors


def _eval_prompt_cache(cache: List[Any]) -> None:
    """Evaluate all cache tensors used by hybrid chunked prefill."""
    tensors = _cache_eval_tensors(cache)
    if tensors:
        mx.eval(*tensors)


@dataclass
class MLLMBatchRequest:
    """
    Request data for MLLM batch processing.

    Contains all information needed to process a multimodal request
    within the batch generator.
    """

    uid: int  # Unique identifier within the batch generator
    request_id: str  # External request ID
    prompt: str  # Text prompt
    images: Optional[List[str]] = None  # Image paths/URLs/base64
    videos: Optional[List[str]] = None  # Video inputs
    audio: Optional[List[str]] = None  # Audio inputs
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    # Extra logits processors (e.g. JSON schema constrained decoding).
    # Merged with built-in repetition/presence penalty processors in
    # ``_prefill_batch``.
    logits_processors: Optional[List[Callable]] = None

    # Processed inputs (set after vision preprocessing)
    input_ids: Optional[mx.array] = None
    pixel_values: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None
    image_grid_thw: Optional[mx.array] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Text-only flag (no images/videos — eligible for prefix cache)
    is_text_only: bool = False

    # Generation state
    num_tokens: int = 0  # Tokens generated so far
    output_tokens: List[int] = field(default_factory=list)

    # Vision state (populated after initial VLM forward pass)
    vision_encoded: bool = False
    cross_attention_states: Optional[Any] = None  # For models that use cross-attention
    encoder_outputs: Optional[Any] = None  # For encoder-decoder models


@dataclass
class MLLMBatchResponse:
    """
    Response from a batch generation step.

    Contains the generated token and metadata for a single request.
    """

    uid: int  # Batch generator UID
    request_id: str  # External request ID
    token: int  # Generated token
    logprobs: mx.array  # Log probabilities
    finish_reason: Optional[str] = None  # "stop", "length", or None
    prompt_cache: Optional[Callable[[], List[Any]]] = None  # Cache extraction function
    from_draft: bool = False  # True when this response is an accepted MTP draft
    mtp_attempted: bool = False  # True when the primary step attempted MTP
    mtp_attempted_count: int = 0  # Number of draft tokens attempted


@dataclass
class MLLMBatch:
    """
    Represents an active batch of MLLM requests.

    Manages the batch state including tokens, caches, and metadata
    for all requests being processed together.
    """

    uids: List[int]
    request_ids: List[str]
    y: mx.array  # Current token(s) for each request [batch_size]
    logprobs: List[mx.array]  # Log probs for each request
    max_tokens: List[int]  # Max tokens per request
    num_tokens: List[int]  # Tokens generated per request
    cache: List[Any]  # BatchKVCache for language model
    requests: List[MLLMBatchRequest]  # Full request data
    logits_processors: Optional[List[Optional[List[Callable]]]] = None
    samplers: Optional[List[Optional[Callable]]] = None

    def __len__(self) -> int:
        return len(self.uids)

    def filter(self, keep_idx: List[int]) -> None:
        """
        Filter batch to keep only requests at specified indices.

        Args:
            keep_idx: Indices of requests to keep
        """
        self.uids = [self.uids[k] for k in keep_idx]
        self.request_ids = [self.request_ids[k] for k in keep_idx]
        self.logprobs = [self.logprobs[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        self.requests = [self.requests[k] for k in keep_idx]
        if self.logits_processors is not None:
            self.logits_processors = [self.logits_processors[k] for k in keep_idx]
        if self.samplers is not None:
            self.samplers = [self.samplers[k] for k in keep_idx]

        keep_idx_array = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx_array]

        # Filter cache entries
        for c in self.cache:
            if hasattr(c, "filter"):
                c.filter(keep_idx_array)

    def extend(self, other: "MLLMBatch") -> None:
        """
        Extend this batch with another batch.

        Args:
            other: Batch to merge into this one
        """
        self.uids.extend(other.uids)
        self.request_ids.extend(other.request_ids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs.extend(other.logprobs)
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        self.requests.extend(other.requests)

        # Extend logits_processors
        if self.logits_processors is not None or other.logits_processors is not None:
            # At this point self.uids already includes other.uids from extend above
            self_len = len(self.uids) - len(other.uids)
            self_lp = self.logits_processors or [None] * self_len
            other_lp = other.logits_processors or [None] * len(other.uids)
            self.logits_processors = list(self_lp) + list(other_lp)

        # Extend samplers
        if self.samplers is not None or other.samplers is not None:
            self_len = len(self.uids) - len(other.uids)
            self_s = self.samplers or [None] * self_len
            other_s = other.samplers or [None] * len(other.uids)
            self.samplers = list(self_s) + list(other_s)

        # Extend cache - handle both BatchKVCache (.keys/.values) and
        # ArraysCache (.cache list) from hybrid models like Qwen3.5. Some
        # cache integrations, such as quantized SDPA caches, expose state only
        # through empty()/extend() and do not publish .keys.
        for c, o in zip(self.cache, other.cache):
            if c is not None and o is not None and hasattr(c, "extend"):
                try:
                    has_kv = hasattr(c, "keys") and c.keys is not None
                    has_arrays = hasattr(c, "cache")
                    has_extendable_state = hasattr(c, "empty") and not c.empty()
                    if has_kv or has_arrays or has_extendable_state:
                        c.extend(o)
                except Exception as e:
                    logger.warning(f"Failed to extend cache: {e}")

    def extract_cache(self, idx: int) -> List[Any]:
        """
        Extract cache for a single request (for prefix caching).

        Handles BatchRotatingKVCache negative left_padding bug:
        during generation with rotation, left_padding becomes negative,
        causing extract() to use Python negative indexing and truncate
        the buffer to only generation tokens instead of the full window.
        """
        from mlx_lm.models.cache import (
            BatchRotatingKVCache,
            RotatingKVCache,
        )

        result = []
        for c in self.cache:
            if not hasattr(c, "extract"):
                result.append(None)
            elif isinstance(c, BatchRotatingKVCache):
                # Custom extraction: clamp left_padding to >= 0
                cache = RotatingKVCache(c.max_size)
                padding = max(0, c.left_padding[idx].item())
                offset = c.offset[idx].item()
                cache.keys = c.keys[idx : idx + 1]
                cache.values = c.values[idx : idx + 1]
                cache._idx = c._idx
                if c.rotated:
                    cache.keys = mx.roll(cache.keys, -c._idx, axis=2)
                    cache.values = mx.roll(cache.values, -c._idx, axis=2)
                    cache._idx = c.max_size
                cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
                cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
                cache.offset = offset
                cache._idx = cache.keys.shape[2]
                cache.step = getattr(c, "step", c.max_size)
                cache.keep = getattr(c, "keep", 0)
                result.append(cache)
            else:
                result.append(c.extract(idx))
        return result


class MLLMBatchStats:
    """Statistics for MLLM batch generation."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.prompt_time: float = 0
        self.generation_tokens: int = 0
        self.generation_time: float = 0
        self.vision_encoding_time: float = 0
        self.num_images_processed: int = 0
        self.peak_memory: float = 0

    @property
    def prompt_tps(self) -> float:
        if self.prompt_time == 0:
            return 0
        return self.prompt_tokens / self.prompt_time

    @property
    def generation_tps(self) -> float:
        if self.generation_time == 0:
            return 0
        return self.generation_tokens / self.generation_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_time": self.prompt_time,
            "prompt_tps": self.prompt_tps,
            "generation_tokens": self.generation_tokens,
            "generation_time": self.generation_time,
            "generation_tps": self.generation_tps,
            "vision_encoding_time": self.vision_encoding_time,
            "num_images_processed": self.num_images_processed,
            "peak_memory": self.peak_memory,
        }


def _left_pad_prompts(
    prompts: List[List[int]], max_length: Optional[int] = None
) -> mx.array:
    """
    Left-pad prompts to uniform length.

    Args:
        prompts: List of token lists
        max_length: Target length (computed if not provided)

    Returns:
        Padded prompts as mx.array [batch_size, seq_len]
    """
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + list(p) for p in prompts])


class MLLMBatchGenerator:
    """
    Batch generator for Vision Language Models.

    This class manages continuous batching for MLLM requests:

    1. Vision Encoding Phase:
       - Process images/videos through vision encoder (per-request)
       - Extract vision features and merge with text embeddings
       - Store cross-attention states for language model

    2. Language Generation Phase:
       - Use language model with BatchKVCache for batched generation
       - Generate tokens for all requests simultaneously
       - Same pattern as LLM BatchGenerator

    Example:
        >>> generator = MLLMBatchGenerator(model, processor)
        >>> uids = generator.insert([request1, request2])
        >>> while responses := generator.next():
        ...     for resp in responses:
        ...         print(f"Request {resp.request_id}: token={resp.token}")
    """

    # Generation stream for async eval
    _stream = None

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        mm_processor: Optional[MultimodalProcessor] = None,
        max_tokens: int = 256,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        prefill_batch_size: int = 4,  # Smaller for MLLM due to vision overhead
        completion_batch_size: int = 16,  # Can be larger for text generation
        prefill_step_size: int = 1024,
        enable_vision_cache: bool = True,
        vision_cache_size: int = 100,
        prefix_cache_config: Optional[MemoryCacheConfig] = None,
        max_kv_size: int = 0,
    ):
        """
        Initialize MLLM batch generator.

        Args:
            model: The VLM model (must have model.language_model)
            processor: The VLM processor for tokenization and image processing
            mm_processor: Optional MultimodalProcessor for input preparation
            max_tokens: Default max tokens per request
            stop_tokens: Set of stop token IDs
            sampler: Sampling function (default: argmax)
            prefill_batch_size: Max requests to prefill together
            completion_batch_size: Max requests for completion batching
            prefill_step_size: Tokens to process per prefill step
            enable_vision_cache: Enable vision embedding caching
            vision_cache_size: Max entries in vision cache
            prefix_cache_config: Config for KV prefix cache (text-only requests)
            max_kv_size: Maximum KV cache size per sequence (0 = unbounded)
        """
        self.model = model
        self.processor = processor
        self.mm_processor = mm_processor
        self.max_kv_size = max_kv_size

        # Get language model for text generation
        self.language_model = getattr(model, "language_model", model)

        # Check if this is actually a VLM with separate language model
        self.is_vlm = hasattr(model, "language_model")
        if self.is_vlm:
            logger.info(
                "MLLMBatchGenerator: Using VLM's language_model for batched generation"
            )
        else:
            logger.warning(
                "MLLMBatchGenerator: Model does not have language_model, using model directly"
            )

        # Patch attention for BatchKVCache compatibility
        from .patches.qwen3_5_mllm import patch_qwen35_attention_for_batching
        from .patches.gemma4_mllm import patch_gemma4_attention_for_batching
        from .patches.glm4v_moe_mllm import patch_glm4v_moe_for_batching

        patch_qwen35_attention_for_batching()
        patch_gemma4_attention_for_batching()
        patch_glm4v_moe_for_batching()

        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.prefill_step_size = prefill_step_size

        # Request management
        self.unprocessed_requests: List[MLLMBatchRequest] = []
        self.active_batch: Optional[MLLMBatch] = None
        self.uid_counter = 0

        # Statistics
        self._stats = MLLMBatchStats()

        # Error responses for requests that failed during preprocessing
        self._pending_error_responses: List[MLLMBatchResponse] = []

        # Per-request prefill progress: request_id → (processed_tokens, total_tokens)
        self._prefill_progress: Dict[str, Tuple[int, int]] = {}

        # Aborted request IDs — checked between prefill chunks to allow
        # early termination when a client disconnects during long prefill.
        # Set operations are GIL-protected, safe across event-loop and
        # executor threads.
        self._aborted_request_ids: set = set()

        # Deferred removal queue — UIDs scheduled for removal from another
        # thread (typically the event loop on client disconnect).  The
        # actual removal, which mutates `active_batch` and touches MLX
        # arrays, must happen on the scheduler thread to avoid a race with
        # an in-flight forward pass.  Metal asserts ("encodeSignalEvent
        # with uncommitted encoder") if two threads submit GPU work on the
        # same stream concurrently.  See `schedule_removal` /
        # `process_pending_removals`.
        self._pending_removal_uids: set = set()
        self._pending_removal_lock = threading.Lock()

        # Vision embedding cache for repeated images
        self.vision_cache = VisionEmbeddingCache(
            max_pixel_entries=vision_cache_size,
            max_encoding_entries=vision_cache_size // 2,
            enabled=enable_vision_cache,
        )
        if enable_vision_cache:
            logger.info(
                f"MLLMBatchGenerator: Vision cache enabled (size={vision_cache_size})"
            )

        # KV prefix cache for text-only requests
        self.prefix_cache: Optional[MemoryAwarePrefixCache] = None
        if prefix_cache_config is not None:
            self.prefix_cache = MemoryAwarePrefixCache(
                model=self.language_model,
                config=prefix_cache_config,
            )
            logger.info("MLLMBatchGenerator: KV prefix cache enabled")

        # Normalize chat template for prefix-cache stability.
        # Qwen3.5 chat template retroactively changes formatting of earlier
        # assistant messages based on last_query_index (position of last
        # non-tool user message).  When a user text message is appended,
        # last_query_index jumps forward, removing <think> blocks from
        # earlier assistant turns — shifting tokens mid-sequence and
        # breaking prefix match.  Fix: always use plain format for
        # historical assistant turns (thinking is still added by the
        # generation prompt at the end).
        self._normalize_chat_template_for_prefix_cache()

        # Compute think-suffix length for prefix cache key stripping.
        # Models with enable_thinking=True add <think>\n to the generation
        # prompt.  This breaks prefix cache (stored key ends with <think>
        # but next request has actual response at that position).
        # Stripping the suffix from cache keys enables clean PREFIX match.
        self._think_suffix_len = self._compute_think_suffix_len()

        # Generation stream
        if MLLMBatchGenerator._stream is None:
            MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())

        # Memory management
        self._old_wired_limit = None
        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )

    def _normalize_chat_template_for_prefix_cache(self) -> None:
        """Patch chat template so historical assistant turns are prefix-stable.

        Qwen3.5's chat template computes ``last_query_index`` — the position
        of the last non-tool-response user message — and conditionally wraps
        assistant turns after that index in ``<think>...\\n</think>\\n\\n``.
        When a new user text message is appended, ``last_query_index`` jumps
        forward, retroactively removing these ``<think>`` wrappers from
        earlier assistant turns.  This shifts tokens mid-sequence and breaks
        prefix cache.

        Fix: replace the conditional with the plain (ELSE) branch so ALL
        historical assistant messages use ``<|im_start|>assistant\\ncontent``
        without any injected ``<think>`` block.  The generation prompt still
        adds ``<think>\\n`` at the very end, so the model generates thinking.
        """
        if self.prefix_cache is None:
            return  # No prefix cache — no need to normalize

        # Find the chat template.  VLM processors (e.g. Qwen3VLProcessor)
        # keep a SEPARATE copy of chat_template from their tokenizer — both
        # must be patched.  The processor's copy is used by
        # BatchedEngine._apply_chat_template() (text rendering), while the
        # tokenizer's copy is used by _compute_think_suffix_len().
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        # Prefer the processor's own template (it's the one used for rendering)
        template = getattr(self.processor, "chat_template", None)
        if not template:
            template = getattr(tokenizer, "chat_template", None)
        if not template or "last_query_index" not in template:
            return  # Not affected

        import re

        # The pattern in Qwen3.5 template:
        #   {%- if loop.index0 > ns.last_query_index %}
        #       {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        #   {%- else %}
        #       {{- '<|im_start|>' + message.role + '\n' + content }}
        #   {%- endif %}
        #
        # Replace with just the ELSE branch (always plain format).
        pattern = (
            r"\{%-\s*if\s+loop\.index0\s*>\s*ns\.last_query_index\s*%\}"
            r".*?"
            r"\{%-\s*else\s*%\}"
            r"\s*(\{\{-.*?content.*?\}\})"
            r"\s*\{%-\s*endif\s*%\}"
        )
        new_template = re.sub(pattern, r"\1", template, flags=re.DOTALL)
        if new_template != template:
            # Patch ALL copies: processor, tokenizer, and any dict variants.
            if hasattr(self.processor, "chat_template"):
                self.processor.chat_template = new_template
            tokenizer.chat_template = new_template
            logger.info(
                "[prefix_cache] Normalized chat template: removed "
                "last_query_index conditional for prefix-stable assistant turns"
            )
        else:
            logger.debug(
                "[prefix_cache] Chat template has last_query_index but "
                "regex did not match — template may use a different pattern"
            )

    def _compute_think_suffix_len(self) -> int:
        """Compute how many extra tokens enable_thinking=True adds at the END.

        Compares the generation prompt suffix with and without
        ``enable_thinking`` to find the think-tag suffix length
        (typically ``<think>\\n`` = 2 tokens for Qwen3/Qwen3.5).

        Returns 0 if the template doesn't support ``enable_thinking``.
        """
        try:
            # Find something with apply_chat_template
            applicator = None
            for candidate in [
                getattr(self.processor, "tokenizer", None),
                self.processor,
            ]:
                if candidate is not None and hasattr(candidate, "apply_chat_template"):
                    applicator = candidate
                    break

            if applicator is None:
                return 0

            dummy = [{"role": "user", "content": "x"}]

            try:
                text_with = applicator.apply_chat_template(
                    dummy,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                text_without = applicator.apply_chat_template(
                    dummy,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return 0

            # Check if enable_thinking adds a known think tag at the end.
            # enable_thinking may also change the system prompt, so we can't
            # simply compare lengths — we look at the ending instead.
            for tag in ["<think>\n", "<think>"]:
                if text_with.endswith(tag) and not text_without.endswith(tag):
                    tokenizer = getattr(self.processor, "tokenizer", self.processor)
                    suffix_tokens = tokenizer.encode(tag)
                    base_tokens = tokenizer.encode("")
                    suffix_len = len(suffix_tokens) - len(base_tokens)
                    if suffix_len > 0:
                        logger.info(
                            f"[think_suffix] Detected think tag "
                            f"'{tag.strip()}' = {suffix_len} token(s)"
                        )
                    return max(0, suffix_len)

            return 0
        except Exception:
            return 0

    def close(self) -> None:
        """Release resources and reset wired limit."""
        if self._old_wired_limit is not None:
            mx.synchronize(MLLMBatchGenerator._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def abort_prefill(self, request_id: str) -> None:
        """Signal that a request's prefill should be aborted.

        Called from the event loop thread when a client disconnects.
        The prefill loop checks this set between chunks and raises
        PrefillAbortedError to exit early.
        """
        self._aborted_request_ids.add(request_id)
        logger.info(f"[abort_prefill] Marked {request_id} for prefill abort")

    def schedule_removal(self, uids: List[int]) -> None:
        """Thread-safe deferred removal of UIDs from the batch.

        Safe to call from any thread (typically the event loop during
        client-disconnect cleanup).  The actual `remove()`, which creates
        ``mx.array`` instances and filters the KV cache, runs on the
        scheduler thread via :meth:`process_pending_removals` at the next
        batch boundary.  This avoids the Metal ``encodeSignalEvent:
        uncommitted encoder`` crash that occurs when two threads submit
        GPU work on the same stream concurrently.
        """
        with self._pending_removal_lock:
            self._pending_removal_uids.update(uids)

    def process_pending_removals(self) -> None:
        """Remove any UIDs enqueued via :meth:`schedule_removal`.

        MUST be called from the scheduler thread only, at a safe point
        (e.g. the start of :meth:`MLLMScheduler.step` before any forward
        pass has been issued).  Safe to call even when the queue is
        empty (no-op).
        """
        # Swap the pending set under a lock so enqueues from other threads
        # cannot be dropped between snapshot and clear.
        with self._pending_removal_lock:
            if not self._pending_removal_uids:
                return
            pending = self._pending_removal_uids
            self._pending_removal_uids = set()

        uids = list(pending)
        self.remove(uids)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def insert(
        self,
        requests: List[MLLMBatchRequest],
    ) -> List[int]:
        """
        Insert requests for batch processing.

        Args:
            requests: List of MLLMBatchRequest to process

        Returns:
            List of UIDs assigned to requests
        """
        uids = []
        for req in requests:
            req.uid = self.uid_counter
            self.uid_counter += 1
            self.unprocessed_requests.append(req)
            uids.append(req.uid)

        # Sort by estimated complexity (no images = simpler)
        self.unprocessed_requests = sorted(
            self.unprocessed_requests,
            key=lambda x: (
                0 if not x.images and not x.videos and not x.audio else 1,
                len(x.images or []) + len(x.videos or []) + len(x.audio or []),
            ),
        )

        logger.debug(f"Inserted {len(requests)} requests, UIDs: {uids}")
        return uids

    def remove(self, uids: List[int]) -> None:
        """
        Remove requests from processing.

        Args:
            uids: List of UIDs to remove
        """
        uid_set = set(uids)

        # Remove from active batch
        if self.active_batch is not None:
            keep_idx = [
                i for i, uid in enumerate(self.active_batch.uids) if uid not in uid_set
            ]
            if keep_idx:
                self.active_batch.filter(keep_idx)
            else:
                self.active_batch = None

        # Remove from unprocessed
        self.unprocessed_requests = [
            r for r in self.unprocessed_requests if r.uid not in uid_set
        ]

    def _preprocess_request(self, request: MLLMBatchRequest) -> None:
        """
        Preprocess a single MLLM request (vision encoding).

        This prepares the inputs by:
        1. Processing images/videos through the processor
        2. Tokenizing the prompt with image tokens
        3. Running vision encoder to get features

        Uses vision cache to skip processing for repeated images.
        Idempotent: if input_ids is already set, returns immediately.

        Args:
            request: Request to preprocess
        """
        # Already preprocessed (e.g. by early executor offloading in
        # _process_loop or chunked prefill interleaving).  Only skip for
        # text-only requests; media requests need pixel/audio cache lookup
        # even if input_ids was set.
        if (
            request.input_ids is not None
            and not request.images
            and not request.videos
            and not request.audio
        ):
            return

        from mlx_vlm.utils import prepare_inputs

        tic = time.perf_counter()

        # Collect all images (including video frames) and audio inputs
        all_images = []
        all_audio = []

        if request.images:
            from .models.mllm import process_image_input

            for img in request.images:
                try:
                    path = process_image_input(img)
                    all_images.append(path)
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")

        if request.videos:
            from .models.mllm import (
                process_video_input,
                extract_video_frames_smart,
                save_frames_to_temp,
                DEFAULT_FPS,
                MAX_FRAMES,
            )

            for video in request.videos:
                try:
                    video_path = process_video_input(video)
                    frames = extract_video_frames_smart(
                        video_path,
                        fps=DEFAULT_FPS,
                        max_frames=MAX_FRAMES,
                    )
                    frame_paths = save_frames_to_temp(frames)
                    all_images.extend(frame_paths)
                except Exception as e:
                    logger.warning(f"Failed to process video: {e}")

        if request.audio:
            from .models.mllm import process_audio_input

            for audio in request.audio:
                try:
                    path = process_audio_input(audio)
                    all_audio.append(path)
                except Exception as e:
                    logger.warning(f"Failed to process audio: {e}")

        # Check pixel cache first
        cached_pixels = None
        if not all_audio:
            cached_pixels = self.vision_cache.get_pixel_cache(
                all_images, request.prompt
            )
        if cached_pixels is not None:
            # Cache hit - use cached pixel values
            request.input_ids = cached_pixels.input_ids
            request.pixel_values = cached_pixels.pixel_values
            request.attention_mask = cached_pixels.attention_mask
            request.image_grid_thw = cached_pixels.image_grid_thw
            request.extra_kwargs = dict(cached_pixels.extra_kwargs)

            logger.debug(
                f"Pixel cache HIT for request {request.request_id}: "
                f"saved {cached_pixels.processing_time:.2f}s"
            )
            return

        # Cache miss - process images
        # Get model config
        model_config = getattr(self.model, "config", None)
        image_token_index = (
            getattr(model_config, "image_token_index", None) if model_config else None
        )

        # Prepare inputs using mlx_vlm
        inputs = prepare_inputs(
            self.processor,
            images=all_images if all_images else None,
            audio=all_audio if all_audio else None,
            prompts=request.prompt,
            image_token_index=image_token_index,
        )

        request.input_ids = inputs.get("input_ids")
        request.pixel_values = inputs.get("pixel_values")
        request.attention_mask = inputs.get("attention_mask")

        # Extract extra kwargs
        request.extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        request.image_grid_thw = request.extra_kwargs.pop("image_grid_thw", None)

        processing_time = time.perf_counter() - tic

        # Store in pixel cache for future reuse
        if all_images and not all_audio and request.pixel_values is not None:
            self.vision_cache.set_pixel_cache(
                images=all_images,
                prompt=request.prompt,
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                image_grid_thw=request.image_grid_thw,
                extra_kwargs=request.extra_kwargs,
                processing_time=processing_time,
            )

        self._stats.num_images_processed += len(all_images)
        self._stats.vision_encoding_time += processing_time

        # Mark text-only requests (eligible for prefix cache)
        request.is_text_only = not bool(all_images or all_audio)

        logger.debug(
            f"Preprocessed request {request.request_id}: "
            f"{len(all_images)} images, {len(all_audio)} audio clips, "
            f"{request.input_ids.size if request.input_ids is not None else 0} tokens "
            f"({processing_time:.2f}s)"
        )

    @staticmethod
    def _copy_prefix_cache(cache_list):
        """Create shallow copies of cache objects to prevent mutation of stored prefix cache.

        MLX arrays are immutable and safe to share, but cache objects have mutable
        Python attributes (offset, _idx) that get modified by update_and_fetch().
        Without copying, the stored prefix cache entry is corrupted after each use.
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        copies = []
        for c in cache_list:
            if isinstance(c, RotatingKVCache):
                new_c = RotatingKVCache(c.max_size, c.keep)
                new_c.step = c.step
                new_c.keys = c.keys
                new_c.values = c.values
                new_c.offset = c.offset
                new_c._idx = c._idx
                copies.append(new_c)
            elif isinstance(c, KVCache):
                new_c = KVCache()
                new_c.step = c.step
                new_c.keys = c.keys
                new_c.values = c.values
                new_c.offset = c.offset
                copies.append(new_c)
            else:
                copies.append(c)
        return copies

    @staticmethod
    def _has_empty_rotating_cache(cache_list):
        """Check if any RotatingKVCache layer has no data (keys=None).

        This happens when prefix cache stores a long response where all
        sliding-window entries were trimmed (entries_to_keep=0).
        Using such a cache produces garbage — fall through to full prefill.
        """
        from mlx_lm.models.cache import RotatingKVCache

        for c in cache_list:
            if isinstance(c, RotatingKVCache) and c.keys is None:
                return True
        return False

    @staticmethod
    def _trim_rotating_caches(cache_list):
        """Trim RotatingKVCache buffers restored from prefix cache.

        Prefix cache stores the full KV state (offset may exceed max_size for
        sliding-window layers).  RotatingKVCache._update_in_place computes
        ``new_size = min(step, max_size - prev)`` which goes negative when
        ``prev > max_size``, crashing with "Negative dimensions not allowed".

        Trimming the buffer to max_size and clamping offset/idx prevents this.
        """
        from mlx_lm.models.cache import RotatingKVCache

        for layer_cache in cache_list:
            if not isinstance(layer_cache, RotatingKVCache):
                continue
            if layer_cache.keys is None:
                layer_cache.offset = 0
                continue
            buf_len = layer_cache.keys.shape[2]
            if buf_len > layer_cache.max_size:
                trim_size = buf_len - layer_cache.max_size
                layer_cache.keys = layer_cache._trim(trim_size, layer_cache.keys)
                layer_cache.values = layer_cache._trim(trim_size, layer_cache.values)
                layer_cache._idx = layer_cache.max_size
            layer_cache.offset = min(layer_cache.offset, layer_cache.max_size)
            # Defensive: ensure size() <= keys.shape[2] to prevent merge crash.
            # Prefix cache trimming can create offset > keys.shape[2] when
            # a supersequence/LCP trim crosses the max_size boundary.
            buf_len = layer_cache.keys.shape[2]
            if min(layer_cache.offset, layer_cache.max_size) > buf_len:
                logger.warning(
                    f"RotatingKVCache offset ({layer_cache.offset}) > "
                    f"buffer ({buf_len}), capping to buffer size"
                )
                layer_cache.offset = buf_len

    def _run_chunked_text_prefill(
        self, request: MLLMBatchRequest, cache: List[Any]
    ) -> mx.array:
        """
        Run prefill in chunks for text-only requests, reporting real progress.

        Processes input_ids in prefill_step_size chunks through the language
        model, updating ``_prefill_progress`` after each chunk so the status
        endpoint can report accurate prefill percentage.

        Returns:
            Logits from the last chunk (same contract as _run_vision_encoding).
        """
        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        total = input_ids.shape[1]
        step = self.prefill_step_size

        # Short prompt — process in one shot (no chunking overhead)
        if total <= step:
            self._prefill_progress[request.request_id] = (total, total)
            output = self.language_model(input_ids, cache=cache)
            request.vision_encoded = True
            # Release preprocessed inputs after encoding (issue #442)
            request.pixel_values = None
            request.attention_mask = None
            request.image_grid_thw = None
            request.extra_kwargs.clear()
            if hasattr(output, "logits"):
                return output.logits
            return output

        logger.info(
            f"[chunked_prefill] Starting {request.request_id[:12]}: "
            f"{total} tokens, step={step}"
        )

        # Process all chunks except the last
        processed = 0
        chunk_count = 0
        while processed + step < total:
            # Check for abort between chunks (client disconnect)
            if request.request_id in self._aborted_request_ids:
                self._aborted_request_ids.discard(request.request_id)
                logger.info(
                    f"[chunked_prefill] Aborted {request.request_id} at "
                    f"{processed}/{total} tokens"
                )
                raise PrefillAbortedError(request.request_id)

            chunk = input_ids[:, processed : processed + step]
            self.language_model(chunk, cache=cache)
            # Eval ALL cache types to break the lazy graph between chunks.
            # ArraysCache (e.g. GatedDeltaNet) has .state; KVCache (full
            # attention) has .keys/.values. Hybrid models like Qwen3.5 use
            # both. Skipping either type lets the computation graph grow
            # across chunks → OOM on long prompts.
            _eval_prompt_cache(cache)
            processed += step
            chunk_count += 1
            self._prefill_progress[request.request_id] = (processed, total)

            # Log progress every 10 chunks so operators can see prefill
            # is progressing (not hanging) during long prompts.
            if chunk_count % 10 == 0:
                logger.info(
                    f"[chunked_prefill] {request.request_id[:12]}: "
                    f"chunk {chunk_count}, {processed}/{total} tokens"
                )

            # Release Metal buffer pool periodically.  Full-attention layers
            # produce attention score buffers that grow each chunk (1024 ×
            # growing_context).  Old smaller buffers can't be reused, so the
            # pool accumulates O(N²) memory without clearing.
            if chunk_count % 4 == 0:
                mx.clear_cache()

        # Last chunk — return logits for sampling
        last_chunk = input_ids[:, processed:]
        output = self.language_model(last_chunk, cache=cache)
        request.vision_encoded = True
        # Release preprocessed inputs after encoding (issue #442)
        request.pixel_values = None
        request.attention_mask = None
        request.image_grid_thw = None
        request.extra_kwargs.clear()
        self._prefill_progress[request.request_id] = (total, total)

        if chunk_count > 0:
            logger.info(
                f"[chunked_prefill] Completed {request.request_id[:12]}: "
                f"{total} tokens in {chunk_count + 1} chunks"
            )

        if hasattr(output, "logits"):
            return output.logits
        return output

    def _run_vision_encoding(
        self, request: MLLMBatchRequest, cache: Optional[List[Any]] = None
    ) -> mx.array:
        """
        Run the initial VLM forward pass to encode vision and get first logits.

        This runs the full VLM model (vision + language) on the prompt,
        which encodes the images and fills the provided KV cache.

        Args:
            request: Preprocessed request with input_ids and pixel_values
            cache: KV cache list for the language model. If provided, the
                   language model writes its KV state directly into this cache
                   during the forward pass.

        Returns:
            Logits from the forward pass
        """
        # Build model call kwargs
        kwargs = dict(request.extra_kwargs)

        if request.pixel_values is not None:
            kwargs["pixel_values"] = request.pixel_values
        if request.attention_mask is not None:
            kwargs["attention_mask"] = request.attention_mask
        if request.image_grid_thw is not None:
            kwargs["image_grid_thw"] = request.image_grid_thw

        # Run full VLM forward pass with cache.
        # The VLM passes cache= through to self.language_model(),
        # so the language model writes KV state directly into our cache.
        input_ids = request.input_ids
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        output = self.model(input_ids, cache=cache, **kwargs)
        request.vision_encoded = True

        # Release preprocessed vision inputs now that they have been encoded
        # into the KV cache.  pixel_values can be hundreds of MB for multi-
        # image requests; holding them pins Metal buffers for the entire
        # generation duration (issue #442).
        request.pixel_values = None
        request.attention_mask = None
        request.image_grid_thw = None
        request.extra_kwargs.clear()

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            return output.logits
        return output

    def _process_prompts(self, requests: List[MLLMBatchRequest]) -> MLLMBatch:
        """
        Process a batch of requests through vision encoding and initial prefill.

        For MLLM, this is more complex than LLM:
        1. Preprocess each request (tokenize, process images)
        2. Run vision encoding per-request with individual KVCache objects
        3. Merge individual caches into a BatchKVCache for generation

        Args:
            requests: Requests to process

        Returns:
            MLLMBatch ready for generation
        """
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        tic = time.perf_counter()

        # Preprocess all requests (per-request error handling)
        failed_requests = []
        for req in requests:
            try:
                self._preprocess_request(req)
            except Exception as e:
                logger.error(
                    f"Failed to preprocess request {req.request_id}: "
                    f"{type(e).__name__}: {e}"
                )
                failed_requests.append(req)

        # Remove failed requests from batch and create error responses
        if failed_requests:
            for req in failed_requests:
                requests.remove(req)
                self._pending_error_responses.append(
                    MLLMBatchResponse(
                        uid=req.uid,
                        request_id=req.request_id,
                        token=0,
                        logprobs=mx.zeros(1),
                        finish_reason="error",
                    )
                )

        if not requests:
            # All requests failed
            return None

        logits_processors_by_request: dict[str, Optional[List[Callable]]] = {}
        samplers_by_request: dict[str, Optional[Callable]] = {}
        for req in requests:
            need_rep = req.repetition_penalty and req.repetition_penalty != 1.0
            need_pres = req.presence_penalty and req.presence_penalty != 0.0
            combined: List[Callable] = []
            if need_rep or need_pres:
                lp_kwargs = {}
                if need_rep:
                    lp_kwargs["repetition_penalty"] = req.repetition_penalty
                if need_pres:
                    lp_kwargs["presence_penalty"] = req.presence_penalty
                combined.extend(make_logits_processors(**lp_kwargs))
                logger.info(
                    f"[sampling] request={req.request_id[:12]} "
                    f"rep_penalty={req.repetition_penalty} "
                    f"pres_penalty={req.presence_penalty}"
                )
            if req.logits_processors:
                combined.extend(req.logits_processors)
                logger.info(
                    f"[sampling] request={req.request_id[:12]} "
                    f"extra_logits_processors={len(req.logits_processors)}"
                )
            logits_processors_by_request[req.request_id] = combined or None

            samplers_by_request[req.request_id] = make_sampler(
                temp=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
            )
            logger.info(
                f"[sampling] request={req.request_id[:12]} "
                f"temp={req.temperature} top_p={req.top_p} "
                f"top_k={req.top_k} min_p={req.min_p}"
            )

        def _sample_first_token(req: MLLMBatchRequest, logits: mx.array):
            sample_logits = logits
            processors = logits_processors_by_request.get(req.request_id)
            if processors:
                empty_tokens = mx.array([], dtype=mx.uint32)
                for processor in processors:
                    sample_logits = processor(empty_tokens, sample_logits)

            logprobs = sample_logits - mx.logsumexp(
                sample_logits, axis=-1, keepdims=True
            )
            sampler = samplers_by_request.get(req.request_id) or self.sampler
            sampled = sampler(logprobs)
            mx.eval(sampled, logprobs)
            return sampled, logprobs

        total_prompt_tokens = sum(
            req.input_ids.size if req.input_ids is not None else 1 for req in requests
        )
        self._stats.prompt_tokens += total_prompt_tokens

        # Log large prompts for monitoring (was previously a hard check that
        # caused infinite retry loops when requests exceeded the limit).
        max_batch_tokens = self.prefill_step_size * len(requests)
        if total_prompt_tokens > max_batch_tokens:
            logger.warning(
                f"Large batch prefill: {total_prompt_tokens} tokens "
                f"(step_size={self.prefill_step_size}, requests={len(requests)}). "
                f"Processing may be slow."
            )

        # Run vision encoding for each request with its own KVCache.
        # Vision encoding cannot be batched because each request may have
        # different images/pixel values. We pass a per-request KVCache to
        # the VLM so the language model writes its KV state directly into it.
        #
        # For text-only requests, we check the prefix cache first. If there's
        # a hit, we skip the full VLM forward and run only the language model
        # on the remaining (uncached) tokens.
        first_tokens = []
        all_logprobs = []
        per_request_caches = []

        aborted_requests = []
        for req in requests:
            try:
                # Check abort before starting prefill
                if req.request_id in self._aborted_request_ids:
                    self._aborted_request_ids.discard(req.request_id)
                    raise PrefillAbortedError(req.request_id)

                # Try prefix cache for all requests (text-only and multimodal).
                # VLM forward writes the same KV state as language model forward
                # for text tokens, so cached KV from a previous VLM run is valid.
                # However, if the remaining (uncached) tokens contain image
                # placeholders, we must fall back to VLM forward instead of
                # running them through the language model alone.
                cached_kv = None
                remaining_ids = None
                if self.prefix_cache is not None and req.input_ids is not None:
                    input_ids_list = req.input_ids.reshape(-1).tolist()
                    # Strip think suffix from lookup key so stored entries
                    # (also stripped) match as clean PREFIX.
                    S = self._think_suffix_len
                    lookup_ids = input_ids_list[:-S] if S > 0 else input_ids_list
                    cached_kv, remaining_ids = self.prefix_cache.fetch(lookup_ids)
                    # Append think suffix back to remaining so the model
                    # sees the full generation prompt (<think>\n).
                    if cached_kv is not None and S > 0:
                        remaining_ids = list(remaining_ids) + input_ids_list[-S:]

                    # If remaining tokens contain image placeholders, the
                    # language-model-only path cannot handle them — clear the
                    # cache hit so we fall through to full VLM forward.
                    if cached_kv is not None and remaining_ids:
                        img_tok = getattr(
                            getattr(self.model, "config", None),
                            "image_token_index",
                            None,
                        )
                        if img_tok is not None and img_tok in remaining_ids:
                            cached_kv = None
                            remaining_ids = None

                # Detect empty RotatingKVCache in cached entry — if any sliding-window
                # layer has keys=None (all entries trimmed), the cache is unusable.
                # Fall through to full prefill instead of producing garbage.
                if cached_kv is not None and self._has_empty_rotating_cache(cached_kv):
                    logger.warning(
                        f"Prefix cache hit for {req.request_id} has empty "
                        f"RotatingKVCache layers — falling through to full prefill"
                    )
                    cached_kv = None
                    remaining_ids = None

                if cached_kv is not None and remaining_ids:
                    # Prefix/LCP match — run language model on remaining tokens.
                    # Copy cache to prevent mutation of stored prefix cache entry.
                    request_cache = self._copy_prefix_cache(cached_kv)
                    self._trim_rotating_caches(request_cache)
                    remaining = mx.array(remaining_ids)[None, :]
                    cached_count = len(input_ids_list) - len(remaining_ids)
                    total_tokens = len(input_ids_list)
                    remaining_count = len(remaining_ids)

                    with mx.stream(MLLMBatchGenerator._stream):
                        step = self.prefill_step_size
                        if remaining_count <= step:
                            # Short remaining — process in one shot
                            self._prefill_progress[req.request_id] = (
                                total_tokens,
                                total_tokens,
                            )
                            logits = self.language_model(remaining, cache=request_cache)
                        else:
                            # Chunked prefill on remaining tokens
                            self._prefill_progress[req.request_id] = (
                                cached_count,
                                total_tokens,
                            )
                            processed = 0
                            chunk_count = 0
                            while processed + step < remaining_count:
                                # Check for abort between chunks
                                if req.request_id in self._aborted_request_ids:
                                    self._aborted_request_ids.discard(req.request_id)
                                    logger.info(
                                        f"[chunked_prefill] Aborted {req.request_id} "
                                        f"at {cached_count + processed}/{total_tokens} tokens"
                                    )
                                    raise PrefillAbortedError(req.request_id)

                                chunk = remaining[:, processed : processed + step]
                                self.language_model(chunk, cache=request_cache)
                                # Eval ALL cache types (see _run_chunked_text_prefill)
                                _eval_prompt_cache(request_cache)
                                processed += step
                                chunk_count += 1
                                self._prefill_progress[req.request_id] = (
                                    cached_count + processed,
                                    total_tokens,
                                )
                                if chunk_count % 4 == 0:
                                    mx.clear_cache()
                            # Last chunk — return logits
                            remaining = remaining[:, processed:]
                            logits = self.language_model(remaining, cache=request_cache)
                            self._prefill_progress[req.request_id] = (
                                total_tokens,
                                total_tokens,
                            )

                        if hasattr(logits, "logits"):
                            logits = logits.logits

                        last_logits = logits[:, -1, :]

                        sampled, logprobs = _sample_first_token(req, last_logits)

                        first_tokens.append(sampled.item())
                        all_logprobs.append(logprobs.squeeze(0))

                    per_request_caches.append(request_cache)
                    req.vision_encoded = True
                    logger.debug(
                        f"Prefix cache hit for {req.request_id}: "
                        f"cached={cached_count}, "
                        f"remaining={remaining_count}"
                    )

                elif cached_kv is not None and not remaining_ids:
                    # Exact/supersequence match — cache has all prompt tokens,
                    # but we still need logits for the last position.
                    # Trim by 1 so re-running the last token produces correct
                    # logits for the next-token prediction.
                    # _trim_cache_offset creates new cache objects (safe for
                    # stored entry).
                    request_cache = _trim_cache_offset(cached_kv, 1)
                    last_token = req.input_ids[:, -1:]
                    total_tokens = len(input_ids_list)
                    self._prefill_progress[req.request_id] = (
                        total_tokens,
                        total_tokens,
                    )

                    with mx.stream(MLLMBatchGenerator._stream):
                        logits = self.language_model(last_token, cache=request_cache)
                        if hasattr(logits, "logits"):
                            logits = logits.logits

                        last_logits = logits[:, -1, :]

                        sampled, logprobs = _sample_first_token(req, last_logits)

                        first_tokens.append(sampled.item())
                        all_logprobs.append(logprobs.squeeze(0))

                    per_request_caches.append(request_cache)
                    req.vision_encoded = True
                    logger.debug(
                        f"Prefix cache exact hit for {req.request_id}: "
                        f"all {total_tokens} tokens cached"
                    )

                else:
                    # Cache miss — full forward pass
                    request_cache = make_prompt_cache(
                        self.language_model,
                        max_kv_size=self.max_kv_size or None,
                    )

                    with mx.stream(MLLMBatchGenerator._stream):
                        # Text-only: chunked prefill with real progress tracking
                        # Multimodal: atomic VLM forward (vision encoder needs full input)
                        if req.is_text_only:
                            logits = self._run_chunked_text_prefill(
                                req, cache=request_cache
                            )
                        else:
                            logits = self._run_vision_encoding(req, cache=request_cache)

                        # Extract last token logits
                        last_logits = logits[:, -1, :]

                        sampled, logprobs = _sample_first_token(req, last_logits)

                        first_tokens.append(sampled.item())
                        all_logprobs.append(logprobs.squeeze(0))

                    per_request_caches.append(request_cache)

            except PrefillAbortedError:
                aborted_requests.append(req)
                self._prefill_progress.pop(req.request_id, None)
                self._pending_error_responses.append(
                    MLLMBatchResponse(
                        uid=req.uid,
                        request_id=req.request_id,
                        token=0,
                        logprobs=mx.zeros(1),
                        finish_reason="abort",
                    )
                )

        # Remove aborted requests — they have no entries in the parallel
        # lists (first_tokens, all_logprobs, per_request_caches)
        if aborted_requests:
            for req in aborted_requests:
                requests.remove(req)
            mx.clear_cache()
            if not requests:
                return None

        # Merge per-request caches into batched caches.
        # Both KVCache.merge() and ArraysCache.merge() produce batch-aware
        # caches that support filter/extend/extract for continuous batching.
        #
        # Fix: RotatingKVCache._update_concat does NOT trim on first call —
        # if prompt length > max_size, the buffer grows beyond max_size.
        # BatchRotatingKVCache.merge() then hits a shape mismatch when
        # copying via _temporal_order (full buffer) into a max_size slice.
        # Trim buffer to max_size before merging.
        from mlx_lm.models.cache import RotatingKVCache

        for rc in per_request_caches:
            self._trim_rotating_caches(rc)
            for layer_cache in rc:
                if isinstance(layer_cache, RotatingKVCache):
                    if layer_cache.keys is not None:
                        # Normalize wrapped rotating cache for merge:
                        # after rotation _idx wraps around but merge()
                        # expects _idx == actual buffer size.
                        # Use keys.shape[2] (actual entries) NOT size()
                        # which can be inconsistent after prefix cache trim
                        # (size() = min(offset, max_size) but buffer may
                        # have fewer entries when trimmed).
                        actual_buf = layer_cache.keys.shape[2]
                        if layer_cache._idx != actual_buf and actual_buf > 0:
                            layer_cache.keys = layer_cache._temporal_order(
                                layer_cache.keys
                            )
                            layer_cache.values = layer_cache._temporal_order(
                                layer_cache.values
                            )
                            layer_cache._idx = actual_buf

        try:
            batch_cache = [
                per_request_caches[0][layer_idx].merge(
                    [c[layer_idx] for c in per_request_caches]
                )
                for layer_idx in range(len(per_request_caches[0]))
            ]
        except Exception as e:
            sample_type = type(per_request_caches[0][0]).__name__
            logger.error(
                f"Failed to merge per-request caches ({sample_type}): "
                f"{type(e).__name__}: {e}"
            )
            raise

        # Create initial y (first generated tokens)
        y = mx.array(first_tokens)

        batch_logits_processors = [
            logits_processors_by_request.get(req.request_id) for req in requests
        ]
        has_any_lp = any(batch_logits_processors)
        batch_samplers = [samplers_by_request.get(req.request_id) for req in requests]
        has_any_sampler = any(batch_samplers)

        self._stats.prompt_time += time.perf_counter() - tic

        # Release preprocessed vision inputs for all requests now that
        # they have been encoded into the batch KV cache.  pixel_values,
        # input_ids, etc. can be hundreds of MB per request; holding them
        # for the entire generation duration pins Metal buffers (issue #442).
        for req in requests:
            req.pixel_values = None
            req.attention_mask = None
            req.image_grid_thw = None
            req.extra_kwargs.clear()

        return MLLMBatch(
            uids=[req.uid for req in requests],
            request_ids=[req.request_id for req in requests],
            y=y,
            logprobs=all_logprobs,
            max_tokens=[req.max_tokens for req in requests],
            num_tokens=[0] * len(requests),
            cache=batch_cache,
            requests=requests,
            logits_processors=batch_logits_processors if has_any_lp else None,
            samplers=batch_samplers if has_any_sampler else None,
        )

    def _step(
        self,
        input_tokens: mx.array,
        cache: List[Any],
        logits_processors: Optional[List[Optional[List[Callable]]]] = None,
        output_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Optional[Callable]]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """
        Run one generation step through the language model.

        Args:
            input_tokens: Input tokens [batch_size, 1] or [batch_size]
            cache: BatchKVCache for the language model
            logits_processors: Per-request logits processors (e.g. repetition penalty)
            output_tokens: Per-request generated tokens so far (needed by processors)
            samplers: Per-request sampler functions (for top_k/min_p)

        Returns:
            Tuple of (sampled tokens, logprobs list)
        """
        # Ensure correct shape
        if input_tokens.ndim == 1:
            input_tokens = input_tokens[:, None]

        # Run language model only (not full VLM)
        output = self.language_model(input_tokens, cache=cache)

        # Handle LanguageModelOutput or plain tensor
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        logits = logits[:, -1, :]

        # Apply per-request logits processors (repetition penalty etc.)
        if logits_processors and output_tokens and any(logits_processors):
            processed_logits = []
            for e in range(logits.shape[0]):
                sample_logits = logits[e : e + 1]
                if logits_processors[e]:
                    # ``output_tokens[e]`` already contains all generated
                    # tokens including the current step's input token (built
                    # by the caller as ``req.output_tokens + [token]``).
                    full_context = output_tokens[e]
                    for processor in logits_processors[e]:
                        sample_logits = processor(mx.array(full_context), sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        # Sample — per-request samplers for top_k/min_p support
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if samplers and any(samplers):
            sampled_list = []
            for e in range(logprobs.shape[0]):
                s = samplers[e] if samplers[e] else self.sampler
                sampled_list.append(s(logprobs[e : e + 1]))
            sampled = mx.concatenate(sampled_list, axis=0)
        else:
            sampled = self.sampler(logprobs)

        return sampled, list(logprobs)

    def _next(self) -> List[MLLMBatchResponse]:
        """
        Internal next() implementation.

        Returns:
            List of MLLMBatchResponse for this step
        """
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0

        # Only start a new batch when there is no active batch generating.
        # Per-request KV caches are created during vision encoding and then
        # merged into a single BatchKVCache. Merging into an active batch
        # mid-generation would cause shape mismatches in attention layers,
        # so queued requests wait until the current batch finishes.
        # Exception: text-only requests can be extended into an active batch
        # via the elif branch below (they skip vision encoding entirely).
        if num_active == 0:
            requests = self.unprocessed_requests[: self.completion_batch_size]

            if len(requests) == 0:
                self.active_batch = None
                return []

            try:
                # Save count before _process_prompts which modifies
                # `requests` in-place via .remove() for failed items.
                num_to_consume = len(requests)
                new_batch = self._process_prompts(requests)
                self.unprocessed_requests = self.unprocessed_requests[num_to_consume:]
                self.active_batch = new_batch
                prompt_processing = True
            except Exception as e:
                logger.error(
                    f"Failed to process batch of {len(requests)} prompts: "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                # Remove failed requests to avoid infinite retry loop
                self.unprocessed_requests = self.unprocessed_requests[len(requests) :]
                for req in requests:
                    self._pending_error_responses.append(
                        MLLMBatchResponse(
                            uid=req.uid,
                            request_id=req.request_id,
                            token=0,
                            logprobs=mx.zeros(1),
                            finish_reason="error",
                        )
                    )

        # Mid-batch extend: text-only requests can join an active batch
        # without vision encoding (no shape mismatch risk).
        elif self.unprocessed_requests:
            text_only = [
                r for r in self.unprocessed_requests if not r.images and not r.videos
            ][: self.completion_batch_size]

            if text_only:
                try:
                    # Capture UIDs before _process_prompts modifies
                    # text_only in-place via .remove() for failed items.
                    all_uids = {r.uid for r in text_only}
                    new_batch = self._process_prompts(text_only)
                    # Remove ALL requested (both successful and failed)
                    self.unprocessed_requests = [
                        r for r in self.unprocessed_requests if r.uid not in all_uids
                    ]
                    if new_batch is not None:
                        batch.extend(new_batch)
                    prompt_processing = True
                except Exception as e:
                    logger.warning(
                        f"Failed to extend batch with text-only requests: "
                        f"{type(e).__name__}: {e}"
                    )
                    # Remove failed requests to avoid infinite retry loop
                    processed_uids = {r.uid for r in text_only}
                    self.unprocessed_requests = [
                        r
                        for r in self.unprocessed_requests
                        if r.uid not in processed_uids
                    ]
                    for req in text_only:
                        self._pending_error_responses.append(
                            MLLMBatchResponse(
                                uid=req.uid,
                                request_id=req.request_id,
                                token=0,
                                logprobs=mx.zeros(1),
                                finish_reason="error",
                            )
                        )

        # Collect any pending error responses (from failed preprocessing)
        error_responses = []
        if self._pending_error_responses:
            error_responses = list(self._pending_error_responses)
            self._pending_error_responses.clear()

        # Generate next token for active batch
        batch = self.active_batch
        if batch is None:
            return error_responses

        y, logprobs = batch.y, batch.logprobs
        output_tokens = None
        if batch.logits_processors:
            y_list = y.tolist()
            output_tokens = [
                list(req.output_tokens) + [token]
                for req, token in zip(batch.requests, y_list)
            ]
        batch.y, batch.logprobs = self._step(
            y[:, None],
            batch.cache,
            batch.logits_processors,
            output_tokens,
            batch.samplers,
        )
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()

        if prompt_processing and num_active == 0:
            # Pure prompt processing (new batch, no prior generation)
            self._stats.prompt_time += toc - tic
        else:
            # Generation step — even if a new request was extended into the
            # batch, the dominant cost is generating for all existing requests.
            self._stats.generation_time += toc - tic

        # Build responses and track finished
        keep_idx = []
        end_idx = []
        responses = []

        for i, (token, uid, request_id, num_tok, max_tok, req) in enumerate(
            zip(
                y,
                batch.uids,
                batch.request_ids,
                batch.num_tokens,
                batch.max_tokens,
                batch.requests,
            )
        ):
            num_tok += 1
            batch.num_tokens[i] = num_tok
            req.num_tokens = num_tok
            req.output_tokens.append(token)

            if batch.logits_processors and _processors_can_retire(
                batch.logits_processors[i]
            ):
                remaining_processors, retired_count = _drop_retired_processors(
                    batch.logits_processors[i]
                )
                if retired_count > 0:
                    # Keep the per-request slot but replace an empty processor
                    # stack with None. The next `_mtp_step` uses any([None]) ==
                    # False, so a fully retired request becomes MTP-eligible
                    # without changing batch alignment.
                    batch.logits_processors[i] = remaining_processors
                    logger.info(
                        "[MTP-MLLM] request=%s retired %d processor(s); "
                        "mtp_eligible_next_step=%s",
                        request_id[:12],
                        retired_count,
                        remaining_processors is None,
                    )

            finish_reason = None
            cache_fn = None

            if token in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(i)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(i)
            else:
                keep_idx.append(i)

            if finish_reason is not None:
                # Extract cache for this request
                cache_fn = lambda idx=i: batch.extract_cache(idx)
                # Cleanup prefill progress tracking
                self._prefill_progress.pop(request_id, None)

            responses.append(
                MLLMBatchResponse(
                    uid=uid,
                    request_id=request_id,
                    token=token,
                    logprobs=logprobs[i],
                    finish_reason=finish_reason,
                    prompt_cache=cache_fn,
                )
            )

        # Store caches for finished text-only requests BEFORE filtering
        self._maybe_store_prefix_cache(batch, end_idx)

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return error_responses + responses

    def next(self) -> List[MLLMBatchResponse]:
        """
        Generate next token for all requests in the batch.

        Returns:
            List of MLLMBatchResponse, one per active request
        """
        with mx.stream(MLLMBatchGenerator._stream):
            return self._next()

    def stats(self) -> MLLMBatchStats:
        """
        Get generation statistics.

        Returns:
            MLLMBatchStats with timing and token counts
        """
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def _maybe_store_prefix_cache(
        self, batch: MLLMBatch, end_indices: List[int]
    ) -> None:
        """Store KV caches for finished text-only requests into prefix cache.

        Must be called BEFORE batch.filter() so that indices are still valid.
        """
        if self.prefix_cache is None or not end_indices:
            return
        for i in end_indices:
            req = batch.requests[i]
            if req.input_ids is not None:
                try:
                    extracted = batch.extract_cache(i)
                    input_ids_list = req.input_ids.reshape(-1).tolist()
                    # Store prompt-only KV: trim generated tokens (+ think
                    # suffix) so the stored offset equals key length exactly.
                    # The exact-match path trims by 1 at fetch time to
                    # re-derive logits for the last prompt token.
                    output_count = batch.num_tokens[i]
                    S = self._think_suffix_len
                    total_trim = output_count + S
                    prompt_cache = _trim_cache_offset(extracted, total_trim)
                    cache_key = input_ids_list[:-S] if S > 0 else input_ids_list
                    self.prefix_cache.store(cache_key, prompt_cache)
                except Exception as e:
                    logger.warning(
                        f"Failed to store prefix cache for {req.request_id}: {type(e).__name__}: {e}"
                    )

    def get_prefill_progress(self, request_id: str) -> Optional[Tuple[int, int]]:
        """Return (processed_tokens, total_tokens) or None."""
        return self._prefill_progress.get(request_id)

    def get_vision_cache_stats(self) -> Dict[str, Any]:
        """Get vision cache statistics."""
        return self.vision_cache.get_stats()

    def get_prefix_cache_stats(self) -> Dict[str, Any]:
        """Get KV prefix cache statistics."""
        if self.prefix_cache is not None:
            return self.prefix_cache.get_stats()
        return {
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

    def has_pending(self) -> bool:
        """Check if there are pending or active requests."""
        return bool(self.unprocessed_requests or self.active_batch)


def install_mtp_mllm(
    batch_gen: "MLLMBatchGenerator",
    language_model: Any,
    num_draft_tokens: int = 1,
) -> None:
    """Install MTP (Multi-Token Prediction) on an MLLMBatchGenerator.

    Adapts the always-advance MTP strategy from scheduler._install_mtp
    for the MLLM batched generation path. Handles hybrid model caches
    (BatchKVCache for attention + ArraysCache for recurrent layers).

    Flow per generation step:
    1. Use skip_state logits/hidden OR run model forward -> sample primary
    2. MTP head drafts one token
    3. Verify [primary, draft] in one model call (always advances cache)
    4. Accept: skip_state from pos 1, defer draft for next step emission
       Reject: trim KV by 2 + restore RNN state + re-advance with primary
    5. Draft is emitted in the NEXT generation step after primary
    """
    from .scheduler import make_sampler

    _orig_step = batch_gen._step
    _draft_sampler = make_sampler(temp=0.0)

    # Skip state: stored logits + hidden from verify pass
    _skip_state: list = [None]

    # Deferred drafts keyed by UID
    _deferred_drafts: Dict[int, dict] = {}
    _attempted_drafts_by_uid: Dict[int, int] = {}

    # MTP stats. These are intentionally exposed through get_mtp_stats() so
    # /v1/status can distinguish "weights injected" from useful draft work.
    _mtp_stats_lock = threading.Lock()
    _mtp_stats = {"attempted": 0, "accepted": 0, "rejected": 0, "errors": 0}
    _bypass_counts = {
        "prefill": 0,
        "no_active_batch": 0,
        "concurrent_batch": 0,
        "non_greedy_sampling": 0,
        "logits_processors": 0,
    }

    def _get_mtp_stats() -> Dict[str, Any]:
        with _mtp_stats_lock:
            attempted = _mtp_stats["attempted"]
            accepted = _mtp_stats["accepted"]
            rejected = _mtp_stats["rejected"]
            errors = _mtp_stats["errors"]
            bypass_counts = dict(_bypass_counts)
        verified = accepted + rejected
        acceptance_rate = accepted / verified if verified > 0 else 0.0
        return {
            "enabled": True,
            "requested_draft_tokens": num_draft_tokens,
            "effective_draft_tokens": 1,
            "mode": "always_advance_verified",
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
            "errors": errors,
            "acceptance_rate": acceptance_rate,
            "bypass_counts": bypass_counts,
            "bypass_counts_semantics": "per_condition_overlapping_not_total_steps",
        }

    batch_gen.get_mtp_stats = _get_mtp_stats

    def _mtp_step(
        input_tokens: mx.array,
        cache: List[Any],
        logits_processors: Optional[List[Optional[List[Callable]]]] = None,
        output_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Optional[Callable]]] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        """Extended _step with MTP always-advance strategy."""
        batch_size = input_tokens.shape[0]
        active_requests = (
            list(batch_gen.active_batch.requests)
            if batch_gen.active_batch is not None
            else []
        )
        has_non_greedy_sampling = any(
            getattr(req, "temperature", 0.0) not in (0, 0.0)
            or getattr(req, "top_p", 1.0) < 1.0
            or getattr(req, "top_k", 0) != 0
            or getattr(req, "min_p", 0.0) != 0.0
            for req in active_requests
        )

        # Prefill guard: skip MTP for multi-token input or when no active batch
        # Also skip MTP when batch has multiple active requests (MTP overhead
        # hurts aggregate throughput in concurrent scenarios). The current
        # verifier is only correctness-safe for greedy decoding with no
        # request-local logits processors. Accepted drafts are emitted directly
        # from the greedy draft/argmax verify path; they do not pass through the
        # request-local sampler. Non-greedy decoding needs a sampler-aware
        # verifier before this guard can be safely relaxed.
        prefill_bypass = input_tokens.shape[1] > 1
        no_active_batch_bypass = batch_gen.active_batch is None
        concurrent_batch_bypass = (
            batch_gen.active_batch is not None and len(batch_gen.active_batch) > 1
        )
        non_greedy_bypass = has_non_greedy_sampling
        logits_processors_bypass = logits_processors is not None and any(
            logits_processors
        )
        if (
            prefill_bypass
            or no_active_batch_bypass
            or concurrent_batch_bypass
            or non_greedy_bypass
            or logits_processors_bypass
        ):
            # Keep the descriptions near the guards so operator-facing
            # telemetry stays dynamic instead of duplicating code predicates:
            # prefill=input_tokens.shape[1] > 1
            # no_active_batch=active_batch is None
            # concurrent_batch=len(active_batch) > 1
            # non_greedy_sampling=request sampler is not greedy
            # logits_processors=request-local processors are active
            with _mtp_stats_lock:
                if prefill_bypass:
                    _bypass_counts["prefill"] += 1
                if no_active_batch_bypass:
                    _bypass_counts["no_active_batch"] += 1
                if concurrent_batch_bypass:
                    _bypass_counts["concurrent_batch"] += 1
                if non_greedy_bypass:
                    _bypass_counts["non_greedy_sampling"] += 1
                if logits_processors_bypass:
                    _bypass_counts["logits_processors"] += 1
            _skip_state[0] = None
            return _orig_step(
                input_tokens, cache, logits_processors, output_tokens, samplers
            )

        # Check skip state
        skip = _skip_state[0]
        if skip is not None and skip["logits"].shape[0] != batch_size:
            skip = None
            _skip_state[0] = None

        if skip is not None:
            logits = skip["logits"]
            hidden_states = skip["hidden"]
            _skip_state[0] = None
        else:
            # Normal forward with return_hidden
            model_output = language_model(input_tokens, cache=cache, return_hidden=True)
            if isinstance(model_output, tuple):
                logits, hidden_states = model_output
            else:
                return _orig_step(
                    input_tokens, cache, logits_processors, output_tokens, samplers
                )
            logits = logits[:, -1, :]

        # Apply logits processors before sampling
        if logits_processors and output_tokens and any(logits_processors):
            processed_logits = []
            for e in range(batch_size):
                sample_logits = logits[e : e + 1]
                if logits_processors[e]:
                    for processor in logits_processors[e]:
                        sample_logits = processor(
                            mx.array(output_tokens[e]), sample_logits
                        )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        # Sample primary (use per-request sampler if available)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if samplers and any(samplers):
            sampled_list = []
            for e in range(logprobs.shape[0]):
                s = samplers[e] if samplers[e] else batch_gen.sampler
                sampled_list.append(s(logprobs[e : e + 1]))
            primary_tokens = mx.concatenate(sampled_list, axis=0)
        else:
            primary_tokens = batch_gen.sampler(logprobs)

        current_uids = list(batch_gen.active_batch.uids)

        # MTP draft + always-advance verify
        try:
            with _mtp_stats_lock:
                _mtp_stats["attempted"] += 1
            draft_logits = language_model.mtp_forward(
                hidden_states[:, -1:, :],
                primary_tokens[:, None],
                mtp_cache=None,
            )
            draft_logits = draft_logits[:, -1, :]
            draft_logprobs = draft_logits - mx.logsumexp(
                draft_logits, axis=-1, keepdims=True
            )
            draft_tokens = _draft_sampler(draft_logprobs)
            for uid in current_uids:
                # Current MLLM MTP drafts one token per primary step. Keep this
                # as a count so future multi-token drafters can report >1.
                _attempted_drafts_by_uid[uid] = 1

            # Snapshot RNN state for hybrid models
            _rnn_snapshots = {}
            for _ci, _c in enumerate(cache):
                if not (hasattr(_c, "is_trimmable") and _c.is_trimmable()):
                    if hasattr(_c, "state"):
                        _rnn_snapshots[_ci] = [
                            mx.array(s) if s is not None else None for s in _c.state
                        ]

            # Verify [primary, draft]
            verify_input = mx.concatenate(
                [primary_tokens[:, None], draft_tokens[:, None]], axis=1
            )
            verify_output = language_model(
                verify_input, cache=cache, return_hidden=True
            )
            if isinstance(verify_output, tuple):
                verify_logits, verify_hidden = verify_output
            else:
                verify_logits = verify_output
                verify_hidden = None

            # Verified mode: check if draft matches verify prediction
            verify_pred = mx.argmax(verify_logits[:, 0, :], axis=-1)
            mx.eval(verify_pred, draft_tokens)
            pred_list = verify_pred.tolist()
            draft_list = draft_tokens.tolist()
            all_accepted = pred_list == draft_list

            if all_accepted and verify_hidden is not None:
                # ACCEPT
                _skip_state[0] = {
                    "logits": verify_logits[:, 1, :],
                    "hidden": verify_hidden[:, -1:, :],
                }
                mx.async_eval(_skip_state[0]["logits"], _skip_state[0]["hidden"])
                verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                    verify_logits[:, 0, :], axis=-1, keepdims=True
                )
                for e in range(batch_size):
                    uid = current_uids[e]
                    _deferred_drafts[uid] = {
                        "token": draft_list[e],
                        "logprobs": verify_lp[e],
                    }
                with _mtp_stats_lock:
                    _mtp_stats["accepted"] += 1

            else:
                # REJECT
                if _rnn_snapshots:
                    # Hybrid model: undo entire verify, re-advance with primary
                    for c in cache:
                        if (
                            hasattr(c, "is_trimmable")
                            and c.is_trimmable()
                            and hasattr(c, "trim")
                        ):
                            c.trim(2)
                    for _ci, _snap in _rnn_snapshots.items():
                        cache[_ci].state = _snap
                    rerun_out = language_model(
                        primary_tokens[:, None],
                        cache=cache,
                        return_hidden=True,
                    )
                    if isinstance(rerun_out, tuple):
                        rerun_logits, rerun_hidden = rerun_out
                    else:
                        rerun_logits = rerun_out
                        rerun_hidden = None
                    if rerun_hidden is not None:
                        _skip_state[0] = {
                            "logits": rerun_logits[:, -1, :],
                            "hidden": rerun_hidden[:, -1:, :],
                        }
                        mx.async_eval(
                            _skip_state[0]["logits"],
                            _skip_state[0]["hidden"],
                        )
                    else:
                        _skip_state[0] = None
                else:
                    # Pure attention model: simple trim
                    for c in cache:
                        if (
                            hasattr(c, "is_trimmable")
                            and c.is_trimmable()
                            and hasattr(c, "trim")
                        ):
                            c.trim(1)
                    if verify_hidden is not None:
                        _skip_state[0] = {
                            "logits": verify_logits[:, 0, :],
                            "hidden": verify_hidden[:, 0:1, :],
                        }
                        mx.async_eval(
                            _skip_state[0]["logits"],
                            _skip_state[0]["hidden"],
                        )
                    else:
                        _skip_state[0] = None
                for uid in current_uids:
                    _deferred_drafts.pop(uid, None)
                with _mtp_stats_lock:
                    _mtp_stats["rejected"] += 1

        except Exception as e:
            logger.warning(f"[MTP-MLLM] draft/verify failed: {e}")
            _skip_state[0] = None
            with _mtp_stats_lock:
                _mtp_stats["errors"] += 1

        # Log MTP stats every 50 steps
        with _mtp_stats_lock:
            acc = _mtp_stats["accepted"]
            rej = _mtp_stats["rejected"]
            err = _mtp_stats["errors"]
        total = acc + rej + err
        if total > 0 and total % 50 == 0:
            rate = acc / (acc + rej) * 100 if (acc + rej) > 0 else 0
            logger.info(
                f"[MTP-MLLM] stats: accepted={acc} rejected={rej} "
                f"errors={err} acceptance={rate:.0f}%"
            )

        return primary_tokens, list(logprobs)

    # Wrap _next to emit deferred MTP drafts
    batch_gen._inner_next = batch_gen._next

    def _mtp_next() -> List[MLLMBatchResponse]:
        """Wrapper around _next that emits deferred MTP draft tokens."""
        if batch_gen.active_batch is None:
            _skip_state[0] = None
            _deferred_drafts.clear()
            _attempted_drafts_by_uid.clear()

        # Save deferred drafts from previous step
        prev_deferred: Dict[int, dict] = {}
        if batch_gen.active_batch is not None:
            for uid in batch_gen.active_batch.uids:
                if uid in _deferred_drafts:
                    prev_deferred[uid] = _deferred_drafts.pop(uid)

        responses = batch_gen._inner_next()

        if responses:
            _mark_mtp_attempts_on_primary_responses(responses, _attempted_drafts_by_uid)

        if not prev_deferred or not responses:
            return responses

        # Augment responses with deferred drafts
        augmented: List[MLLMBatchResponse] = []
        draft_end_uids: set = set()

        for r in responses:
            uid = r.uid
            augmented.append(r)

            if r.finish_reason is not None:
                _deferred_drafts.pop(uid, None)
                prev_deferred.pop(uid, None)
                continue

            if uid in prev_deferred:
                draft_info = prev_deferred.pop(uid)
                draft_t = draft_info["token"]
                draft_lp = draft_info["logprobs"]

                if draft_t in batch_gen.stop_tokens:
                    augmented.append(
                        MLLMBatchResponse(
                            uid=uid,
                            request_id=r.request_id,
                            token=draft_t,
                            logprobs=draft_lp,
                            finish_reason="stop",
                            from_draft=True,
                        )
                    )
                    draft_end_uids.add(uid)
                else:
                    draft_finish = None
                    batch = batch_gen.active_batch
                    if batch is not None:
                        for e, bu in enumerate(batch.uids):
                            if bu == uid:
                                batch.num_tokens[e] += 1
                                batch.requests[e].output_tokens.append(draft_t)
                                if batch.num_tokens[e] >= batch.max_tokens[e]:
                                    draft_finish = "length"
                                    draft_end_uids.add(uid)
                                break

                    augmented.append(
                        MLLMBatchResponse(
                            uid=uid,
                            request_id=r.request_id,
                            token=draft_t,
                            logprobs=draft_lp,
                            finish_reason=draft_finish,
                            from_draft=True,
                        )
                    )

        # Store prefix caches for draft-ended sequences BEFORE filtering
        if draft_end_uids and batch_gen.active_batch is not None:
            end_indices = [
                e
                for e, u in enumerate(batch_gen.active_batch.uids)
                if u in draft_end_uids
            ]
            batch_gen._maybe_store_prefix_cache(batch_gen.active_batch, end_indices)

            keep = [
                e
                for e, u in enumerate(batch_gen.active_batch.uids)
                if u not in draft_end_uids
            ]
            if keep:
                batch_gen.active_batch.filter(keep)
            else:
                batch_gen.active_batch = None

        return augmented

    batch_gen._step = _mtp_step
    batch_gen._next = _mtp_next

    if num_draft_tokens != 1:
        logger.warning(
            "[MTP-MLLM] num_draft_tokens=%d requested, but the current batched "
            "MLLM MTP path drafts exactly one token per verify step",
            num_draft_tokens,
        )
    total = _mtp_stats
    logger.info(
        f"[MTP-MLLM] installed with num_draft_tokens={num_draft_tokens}, "
        f"effective_draft_tokens=1, always-advance verified mode"
    )


def install_chunked_prefill_mllm(
    batch_gen: "MLLMBatchGenerator",
    budget: int = 1024,
) -> None:
    """Install interleaved prefill/decode on an MLLMBatchGenerator.

    When a long text-only request arrives, instead of blocking the entire
    event loop for 20-60+ seconds during prefill, this processes ONE chunk
    of the new request's prefill per ``step()`` call.  Between steps the
    scheduler yields to the event loop (``await asyncio.sleep(0)``), so
    health/status/metrics endpoints remain responsive.

    When an active batch is generating, prefill chunks are interleaved with
    generation steps to keep throughput for existing requests at 30-50 tok/s.

    Args:
        batch_gen: The MLLMBatchGenerator to patch.
        budget: Max tokens to prefill per step (chunk size).
    """
    from mlx_lm.models.cache import make_prompt_cache

    _orig_next = batch_gen._next
    batch_gen._partial = None
    batch_gen._chunked_prefill_budget = budget

    logger.info(
        f"[chunked-prefill-mllm] Installing interleaved prefill/decode "
        f"(budget={budget} tokens/step)"
    )

    def _generation_step() -> List[MLLMBatchResponse]:
        """Run one generation step for the active batch. Returns responses."""
        # Collect pending error responses
        error_responses = list(batch_gen._pending_error_responses)
        batch_gen._pending_error_responses.clear()

        batch = batch_gen.active_batch
        if batch is None:
            return error_responses

        tic = time.perf_counter()
        y, logprobs = batch.y, batch.logprobs
        output_tokens = (
            [req.output_tokens for req in batch.requests]
            if batch.logits_processors
            else None
        )
        batch.y, batch.logprobs = batch_gen._step(
            y[:, None],
            batch.cache,
            batch.logits_processors,
            output_tokens,
            batch.samplers,
        )
        # Synchronous eval — must have results before context switch
        mx.eval(batch.y, batch.logprobs)

        y = y.tolist()
        batch_gen._stats.generation_time += time.perf_counter() - tic

        # Build responses and track finished
        keep_idx = []
        end_idx = []
        responses = []

        for i, (token, uid, request_id, num_tok, max_tok, req) in enumerate(
            zip(
                y,
                batch.uids,
                batch.request_ids,
                batch.num_tokens,
                batch.max_tokens,
                batch.requests,
            )
        ):
            num_tok += 1
            batch.num_tokens[i] = num_tok
            req.num_tokens = num_tok
            req.output_tokens.append(token)

            finish_reason = None

            if token in batch_gen.stop_tokens:
                finish_reason = "stop"
                end_idx.append(i)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(i)
            else:
                keep_idx.append(i)

            if finish_reason is not None:
                batch_gen._prefill_progress.pop(request_id, None)

            responses.append(
                MLLMBatchResponse(
                    uid=uid,
                    request_id=request_id,
                    token=token,
                    logprobs=logprobs[i],
                    finish_reason=finish_reason,
                    prompt_cache=(
                        (lambda idx=i: batch.extract_cache(idx))
                        if finish_reason is not None
                        else None
                    ),
                )
            )

        # Store caches for finished text-only requests BEFORE filtering
        batch_gen._maybe_store_prefix_cache(batch, end_idx)

        # Remove finished requests from batch
        if end_idx:
            if keep_idx:
                batch.filter(keep_idx)
            else:
                batch_gen.active_batch = None

        batch_gen._stats.generation_tokens += len(responses)
        return error_responses + responses

    def _chunked_next() -> List[MLLMBatchResponse]:
        """Interleaved prefill/decode: one prefill chunk + one gen step."""

        # === Phase 1: Continue partial prefill ===
        if batch_gen._partial is not None:
            partial = batch_gen._partial
            req = partial["request"]

            # Abort check
            if req.request_id in batch_gen._aborted_request_ids:
                batch_gen._aborted_request_ids.discard(req.request_id)
                batch_gen._partial = None
                mx.clear_cache()
                batch_gen._prefill_progress.pop(req.request_id, None)
                batch_gen._pending_error_responses.append(
                    MLLMBatchResponse(
                        uid=req.uid,
                        request_id=req.request_id,
                        token=0,
                        logprobs=mx.zeros(1),
                        finish_reason="abort",
                    )
                )
                return _generation_step()

            step = batch_gen._chunked_prefill_budget
            remaining = partial["remaining_ids"]
            remaining_count = remaining.shape[1]

            if remaining_count > step:
                # Process ONE chunk
                tic = time.perf_counter()
                batch_gen.language_model(remaining[:, :step], cache=partial["cache"])
                _eval_prompt_cache(partial["cache"])
                partial["remaining_ids"] = remaining[:, step:]
                partial["processed"] += step
                partial["chunk_count"] += 1
                batch_gen._prefill_progress[req.request_id] = (
                    partial["cached_count"] + partial["processed"],
                    partial["total"],
                )
                batch_gen._stats.prompt_time += time.perf_counter() - tic

                # Periodic memory cleanup
                if partial["chunk_count"] % 4 == 0:
                    mx.clear_cache()

                # Process any short pending requests inline so they
                # don't wait for the entire long prefill to finish.
                # IMPORTANT: Only inline requests whose prompt fits
                # within the chunk budget — longer requests must wait
                # for their own interleaved prefill (Phase 2).
                if batch_gen.unprocessed_requests:
                    _budget = batch_gen._chunked_prefill_budget
                    short_reqs = []
                    for r in batch_gen.unprocessed_requests:
                        if r.images or r.videos:
                            continue
                        if r.input_ids is None:
                            try:
                                batch_gen._preprocess_request(r)
                            except Exception:
                                continue
                        if r.input_ids is not None and r.input_ids.size <= _budget:
                            short_reqs.append(r)
                    if short_reqs:
                        try:
                            new_batch = batch_gen._process_prompts(short_reqs)
                            if new_batch is not None:
                                if batch_gen.active_batch is not None:
                                    batch_gen.active_batch.extend(new_batch)
                                else:
                                    batch_gen.active_batch = new_batch
                        except Exception as e:
                            logger.warning(
                                f"[chunked-prefill-mllm] Failed to process "
                                f"inline short requests: {e}"
                            )

                if batch_gen.active_batch is not None:
                    return _generation_step()
                else:
                    # Idle server — yield to event loop between chunks
                    return []
            else:
                # Last chunk — finalize prefill
                tic = time.perf_counter()
                logits = batch_gen.language_model(remaining, cache=partial["cache"])
                if hasattr(logits, "logits"):
                    logits = logits.logits
                last_logits = logits[:, -1, :]

                # Apply logits processors for first token
                if getattr(req, "logits_processors", None):
                    empty_tokens = mx.array([], dtype=mx.int32)
                    for processor in req.logits_processors:
                        last_logits = processor(empty_tokens, last_logits)

                logprobs = last_logits - mx.logsumexp(
                    last_logits, axis=-1, keepdims=True
                )
                sampled = batch_gen.sampler(logprobs)
                mx.eval(sampled, logprobs)

                batch_gen._prefill_progress[req.request_id] = (
                    partial["total"],
                    partial["total"],
                )
                batch_gen._stats.prompt_time += time.perf_counter() - tic

                # Build single-request batch
                from mlx_lm.sample_utils import make_logits_processors, make_sampler

                req_lp = []
                need_rep = req.repetition_penalty and req.repetition_penalty != 1.0
                need_pres = req.presence_penalty and req.presence_penalty != 0.0
                if need_rep or need_pres:
                    lp_kwargs = {}
                    if need_rep:
                        lp_kwargs["repetition_penalty"] = req.repetition_penalty
                    if need_pres:
                        lp_kwargs["presence_penalty"] = req.presence_penalty
                    req_lp.extend(make_logits_processors(**lp_kwargs))
                if req.logits_processors:
                    req_lp.extend(req.logits_processors)

                req_sampler = None
                if req.top_k != 0 or req.min_p != 0.0:
                    req_sampler = make_sampler(
                        temp=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        min_p=req.min_p,
                    )

                new_batch = MLLMBatch(
                    uids=[req.uid],
                    request_ids=[req.request_id],
                    y=sampled,
                    logprobs=[logprobs.squeeze(0)],
                    max_tokens=[req.max_tokens],
                    num_tokens=[0],
                    cache=partial["cache"],
                    requests=[req],
                    logits_processors=[req_lp] if req_lp else None,
                    samplers=[req_sampler] if req_sampler else None,
                )

                # Extend active batch or set as new
                if batch_gen.active_batch is not None:
                    # Convert per-request cache to batch-compatible format
                    # via merge (same as _process_prompts does)
                    from mlx_lm.models.cache import RotatingKVCache

                    request_cache = partial["cache"]
                    batch_gen._trim_rotating_caches(request_cache)
                    for layer_cache in request_cache:
                        if isinstance(layer_cache, RotatingKVCache):
                            if layer_cache.keys is not None:
                                actual_buf = layer_cache.keys.shape[2]
                                if layer_cache._idx != actual_buf and actual_buf > 0:
                                    layer_cache.keys = layer_cache._temporal_order(
                                        layer_cache.keys
                                    )
                                    layer_cache.values = layer_cache._temporal_order(
                                        layer_cache.values
                                    )
                                    layer_cache._idx = actual_buf

                    # Convert single-request cache to B=1 batch format
                    # so it can be extended into the active batch.
                    merged_cache = [
                        request_cache[layer_idx].merge([request_cache[layer_idx]])
                        for layer_idx in range(len(request_cache))
                    ]
                    new_batch.cache = merged_cache
                    batch_gen.active_batch.extend(new_batch)
                else:
                    # No active batch — convert single-request cache
                    # to B=1 batch format for the new active batch.
                    request_cache = partial["cache"]
                    merged_cache = [
                        request_cache[layer_idx].merge([request_cache[layer_idx]])
                        for layer_idx in range(len(request_cache))
                    ]
                    new_batch.cache = merged_cache
                    batch_gen.active_batch = new_batch

                # Store in prefix cache (prompt-only)
                if batch_gen.prefix_cache is not None and req.input_ids is not None:
                    try:
                        input_ids_list = req.input_ids.reshape(-1).tolist()
                        S = batch_gen._think_suffix_len
                        cache_key = input_ids_list[:-S] if S > 0 else input_ids_list
                        # Trim output: at store time output_count=0, so
                        # trim by S only (matching canonical path's
                        # output_count + S invariant).
                        trim_amount = S
                        store_cache = _trim_cache_offset(partial["cache"], trim_amount)
                        batch_gen.prefix_cache.store(cache_key, store_cache)
                    except Exception as e:
                        logger.warning(
                            f"Failed to store prefix cache after chunked "
                            f"prefill for {req.request_id}: {e}"
                        )

                logger.info(
                    f"[chunked-prefill-mllm] Completed interleaved prefill "
                    f"for {req.request_id[:12]}: "
                    f"{partial['total']} tokens in {partial['chunk_count']} chunks"
                )
                batch_gen._partial = None
                mx.clear_cache()
                return _generation_step()

        # === Phase 2: No partial — check for new requests ===
        batch = batch_gen.active_batch
        num_active = len(batch) if batch else 0

        if batch_gen.unprocessed_requests:
            # Find first text-only request eligible for interleaving
            text_only_req = None
            for r in batch_gen.unprocessed_requests:
                if not r.images and not r.videos:
                    text_only_req = r
                    break

            if text_only_req is not None:
                try:
                    # Preprocess to get input_ids
                    batch_gen._preprocess_request(text_only_req)
                except Exception as e:
                    logger.error(
                        f"Failed to preprocess request "
                        f"{text_only_req.request_id}: {e}"
                    )
                    batch_gen.unprocessed_requests.remove(text_only_req)
                    batch_gen._pending_error_responses.append(
                        MLLMBatchResponse(
                            uid=text_only_req.uid,
                            request_id=text_only_req.request_id,
                            token=0,
                            logprobs=mx.zeros(1),
                            finish_reason="error",
                        )
                    )
                    return _generation_step()

                # Check prefix cache
                input_ids = text_only_req.input_ids
                if input_ids.ndim == 1:
                    input_ids = input_ids[None, :]

                cached_kv = None
                remaining_ids = None
                cached_count = 0
                total_tokens = input_ids.shape[1]

                if batch_gen.prefix_cache is not None:
                    input_ids_list = input_ids.reshape(-1).tolist()
                    S = batch_gen._think_suffix_len
                    lookup_ids = input_ids_list[:-S] if S > 0 else input_ids_list
                    cached_kv, remaining_ids = batch_gen.prefix_cache.fetch(lookup_ids)
                    if cached_kv is not None and S > 0:
                        remaining_ids = list(remaining_ids) + input_ids_list[-S:]

                    # Check for empty rotating cache
                    if cached_kv is not None and batch_gen._has_empty_rotating_cache(
                        cached_kv
                    ):
                        cached_kv = None
                        remaining_ids = None

                if cached_kv is not None and remaining_ids:
                    # Prefix cache hit
                    request_cache = batch_gen._copy_prefix_cache(cached_kv)
                    batch_gen._trim_rotating_caches(request_cache)
                    remaining = mx.array(remaining_ids)[None, :]
                    cached_count = total_tokens - len(remaining_ids)
                    remaining_count = len(remaining_ids)
                elif cached_kv is not None and not remaining_ids:
                    # Exact hit — trim cache by 1 so replaying the last token
                    # produces correct logits (same as _process_prompts path).
                    request_cache = _trim_cache_offset(cached_kv, 1)
                    remaining = input_ids[:, -1:]
                    cached_count = total_tokens - 1
                    remaining_count = 1
                else:
                    # Cache miss — full prefill
                    request_cache = make_prompt_cache(
                        batch_gen.language_model,
                        max_kv_size=batch_gen.max_kv_size or None,
                    )
                    remaining = input_ids
                    cached_count = 0
                    remaining_count = total_tokens

                # Decide: interleave or immediate
                if remaining_count > batch_gen._chunked_prefill_budget:
                    # LONG prompt — start partial (interleaved) prefill
                    logger.info(
                        f"[chunked-prefill-mllm] Starting interleaved prefill "
                        f"for {text_only_req.request_id[:12]}: "
                        f"{remaining_count} remaining tokens "
                        f"(cached={cached_count}, budget={batch_gen._chunked_prefill_budget})"
                    )
                    batch_gen._partial = {
                        "request": text_only_req,
                        "cache": request_cache,
                        "remaining_ids": remaining,
                        "processed": 0,
                        "total": total_tokens,
                        "cached_count": cached_count,
                        "chunk_count": 0,
                    }
                    batch_gen.unprocessed_requests.remove(text_only_req)
                    text_only_req.vision_encoded = True

                    # Process first chunk immediately
                    step = batch_gen._chunked_prefill_budget
                    tic = time.perf_counter()
                    batch_gen.language_model(remaining[:, :step], cache=request_cache)
                    _eval_prompt_cache(request_cache)
                    batch_gen._partial["remaining_ids"] = remaining[:, step:]
                    batch_gen._partial["processed"] = step
                    batch_gen._partial["chunk_count"] = 1
                    batch_gen._prefill_progress[text_only_req.request_id] = (
                        cached_count + step,
                        total_tokens,
                    )
                    batch_gen._stats.prompt_time += time.perf_counter() - tic

                    if num_active > 0:
                        return _generation_step()
                    else:
                        # Idle server — yield to event loop between chunks
                        return []
                # else: SHORT prompt — fall through to _orig_next.
                # _preprocess_request is idempotent for text-only (sets
                # input_ids if not already set); _process_prompts checks
                # input_ids and skips redundant preprocessing.

        # === Phase 3: No partial, no long prompt — original behavior ===
        return _orig_next()

    # Patch remove() to handle partial abort
    _orig_remove = batch_gen.remove

    def _patched_remove(uids: List[int]) -> None:
        if batch_gen._partial is not None:
            if batch_gen._partial["request"].uid in set(uids):
                batch_gen._partial = None
                mx.clear_cache()
        _orig_remove(uids)

    batch_gen.remove = _patched_remove
    batch_gen._next = _chunked_next

    logger.info(f"[chunked-prefill-mllm] Installed (budget={budget} tokens/step)")
