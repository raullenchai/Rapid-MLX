# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for vllm-mlx continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import mlx.core as mx
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig
from .paged_cache import PagedCacheManager
from .ssd_cache import SSDCacheConfig, SSDCacheTier
from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .utils.mamba_cache import ensure_mamba_support

logger = logging.getLogger(__name__)

# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "cache",
    "BatchKVCache",
]


def _normalize_logits_processors(logits_processors):
    """Normalize empty per-sequence processor slots to lists."""
    if logits_processors is None:
        return None
    return [processors or [] for processors in logits_processors]


def _sanitize_batch_generator_logits_processors(batch_generator) -> None:
    """Sanitize stale BatchGenerator processor state before decode."""
    active_batch = getattr(batch_generator, "active_batch", None)
    if active_batch is not None and hasattr(active_batch, "logits_processors"):
        active_batch.logits_processors = _normalize_logits_processors(
            active_batch.logits_processors
        )

    partial = getattr(batch_generator, "_partial", None)
    if isinstance(partial, dict) and "logits_processors" in partial:
        partial["logits_processors"] = _normalize_logits_processors(
            partial["logits_processors"]
        )


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048
    # Optional override for MLLM prefill guard (None = use MLLM default).
    mllm_prefill_step_size: Optional[int] = None

    # Prefix cache settings
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100  # Max cached entries (legacy, ignored if memory-aware)

    # Memory-aware cache settings (recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: Optional[int] = None  # None = auto-detect (20% of available RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting

    # KV cache quantization (reduces prefix cache memory)
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # Paged cache settings (experimental - for memory efficiency)
    use_paged_cache: bool = (
        False  # Use BlockAwarePrefixCache instead of PrefixCacheManager
    )
    paged_cache_block_size: int = 64  # Tokens per block
    max_cache_blocks: int = 1000  # Maximum number of cache blocks

    # Chunked prefill: max tokens to prefill per scheduler step (0 = disabled)
    # When enabled, large prompts are split into chunks so that active
    # generation requests are not starved during long prefills.
    chunked_prefill_tokens: int = 0

    # Mid-prefill cache saving: save intermediate KV cache every N tokens
    # during chunked prefill. If the client disconnects mid-prefill, the
    # saved cache is reused for the next request with the same prefix.
    # 0 = disabled. Only effective when chunked_prefill_tokens > 0.
    mid_prefill_save_interval: int = 8192

    # SSD cache tiering
    ssd_cache_dir: Optional[str] = None  # None = disabled
    ssd_cache_max_gb: float = 10.0

    # Maximum KV cache size per sequence (0 = unbounded; >0 enables RotatingKVCache)
    max_kv_size: int = 0

    # MTP (Multi-Token Prediction) settings
    # Uses the model's built-in MTP head to predict multiple tokens per step
    enable_mtp: bool = False
    mtp_num_draft_tokens: int = 1  # Number of draft tokens from MTP head
    mtp_optimistic: bool = False  # Skip acceptance check for max speed

    def __post_init__(self) -> None:
        if self.mllm_prefill_step_size is not None and self.mllm_prefill_step_size <= 0:
            raise ValueError("mllm_prefill_step_size must be > 0 when provided")


@dataclass
class SchedulerOutput:
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


def _install_prompt_cache_save(batch_gen: "BatchGenerator", prompt_cache_save) -> None:
    """Monkey-patch ``_process_prompts`` to capture prompt-only cache state.

    Can be installed independently of chunked prefill.  If chunked prefill is
    also installed, *it* takes over ``_process_prompts`` and invokes the
    callback itself, so call this **before** ``_install_chunked_prefill``.
    """
    _orig_process_prompts = batch_gen._process_prompts

    try:
        from mlx_lm.generate import Batch as _batch_cls
    except ImportError:
        _batch_cls = None  # extract_cache fallback handled in patched fn

    def _patched_process_prompts(prompts, _self=batch_gen):
        batch = _orig_process_prompts(prompts)
        for e, uid in enumerate(batch.uids):
            if batch.num_tokens[e] == 0:
                try:
                    prompt_cache_save(uid, batch.extract_cache(e))
                except Exception:
                    pass
        return batch

    batch_gen._process_prompts = _patched_process_prompts


def _install_chunked_prefill(
    batch_gen: "BatchGenerator",
    budget: int,
    mid_prefill_save=None,
    prompt_cache_save=None,
    pending_abort_ids: Optional[Set[str]] = None,
    uid_to_request_id: Optional[Dict[int, str]] = None,
    requests: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Monkey-patch a BatchGenerator instance so that large prefills are
    broken into chunks of at most *budget* tokens each.

    Between chunks the generation loop gets a chance to produce one token
    for every active request, preventing starvation during long prefills.

    Args:
        batch_gen: The BatchGenerator to patch.
        budget: Max tokens per prefill chunk.
        mid_prefill_save: Optional callback(uid, processed, prompt_cache)
            called after each chunk to save intermediate KV cache state.
    """
    import time as _time

    from mlx_lm.generate import (
        _left_pad_prompts,
        _make_cache,
        _merge_caches,
        _right_pad_prompts,
    )

    try:
        from mlx_lm.generate import _lazy_extract_cache
    except ImportError:

        def _lazy_extract_cache(cache, idx):
            return (c.extract(idx) for c in cache)

    try:
        from mlx_lm.generate import Batch as _batch_cls
    except ImportError:

        @dataclass
        class _batch_cls:
            uids: List[int]
            y: Any
            logprobs: List[Any]
            max_tokens: List[int]
            num_tokens: List[int]
            cache: List[Any]
            samplers: List[Any]
            logits_processors: List[Any]
            tokens: List[Any]

            def __len__(self):
                return len(self.uids)

            def filter(self, keep_idx: List[int]):
                self.uids = [self.uids[k] for k in keep_idx]
                self.logprobs = [self.logprobs[k] for k in keep_idx]
                self.max_tokens = [self.max_tokens[k] for k in keep_idx]
                self.num_tokens = [self.num_tokens[k] for k in keep_idx]
                self.samplers = [self.samplers[k] for k in keep_idx]
                self.logits_processors = [self.logits_processors[k] for k in keep_idx]
                self.tokens = [self.tokens[k] for k in keep_idx]
                keep_idx_mx = mx.array(keep_idx, mx.int32)
                self.y = self.y[keep_idx_mx]
                for c in self.cache:
                    c.filter(keep_idx_mx)

            def extend(self, other):
                self.uids.extend(other.uids)
                self.y = mx.concatenate([self.y, other.y])
                self.logprobs.extend(other.logprobs)
                self.num_tokens.extend(other.num_tokens)
                self.max_tokens.extend(other.max_tokens)
                self.samplers.extend(other.samplers)
                self.logits_processors.extend(other.logits_processors)
                self.tokens.extend(other.tokens)
                for c, o in zip(self.cache, other.cache):
                    c.extend(o)

            def extract_cache(self, idx):
                return [c.extract(idx) for c in self.cache]

    # Keep references to originals
    _orig_next = batch_gen._next
    _orig_remove = batch_gen.remove
    _orig_process_prompts = batch_gen._process_prompts

    # Partial prefill state (None when no prefill in progress)
    batch_gen._partial = None

    # Monkey-patch _process_prompts to capture prompt-only cache state.
    # At the point where _process_prompts returns, the Batch cache contains
    # the exact prompt-only state: all prompt tokens have been processed
    # through the model, but no output token has been fed back yet.
    # This is the only safe capture point for hybrid Mamba+Transformer
    # models whose MambaCache state is cumulative.
    if prompt_cache_save is not None:

        def _patched_process_prompts(prompts, _self=batch_gen):
            batch = _orig_process_prompts(prompts)
            for e, uid in enumerate(batch.uids):
                if batch.num_tokens[e] == 0:
                    try:
                        prompt_cache_save(uid, batch.extract_cache(e))
                    except Exception:
                        pass
            return batch

        batch_gen._process_prompts = _patched_process_prompts

    def _generation_step(self=batch_gen):
        """Run one generation step on the active batch. Returns responses."""
        batch = self.active_batch
        if batch is None or len(batch) == 0:
            return []

        tic_gen = _time.perf_counter()
        y, logprobs = batch.y, batch.logprobs
        for i, toks in enumerate(batch.tokens):
            batch.tokens[i] = mx.concatenate((toks, y[i : i + 1]))
        batch.y, batch.logprobs = self._step(
            y[:, None],
            batch.cache,
            batch.samplers,
            batch.logits_processors,
            batch.tokens,
        )
        mx.async_eval(batch.y, batch.logprobs)
        # Evaluate accumulated tokens to prevent Metal buffer buildup
        # from lazy mx.concatenate() chains holding AGXAllocation handles
        if batch.tokens:
            mx.async_eval(*batch.tokens)

        y = y.tolist()
        self._stats.generation_time += _time.perf_counter() - tic_gen

        keep_idx = []
        end_idx = []
        responses = []
        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            cache_out = None
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            if finish_reason is not None:
                cache_out = batch.extract_cache(e)
            responses.append(
                self.Response(uid, t, logprobs[e], finish_reason, cache_out)
            )

        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)
        return responses

    def _chunked_next(self=batch_gen):  # noqa: C901
        """
        Replacement for _next() that chunks large prefills.

        Only intercepts when:
        1. A partial prefill is in progress (_partial is not None)
        2. The next prompt batch exceeds the budget

        Everything else delegates to the original _next().
        """
        # ----- Continue a partial prefill -----
        if self._partial is not None:
            # Check for pending aborts BEFORE processing next chunk
            if pending_abort_ids is not None and uid_to_request_id is not None:
                partial_rids = {uid_to_request_id.get(u) for u in self._partial["uids"]}
                aborted_rids = partial_rids & pending_abort_ids
                if aborted_rids:
                    logger.info(
                        f"[chunked_prefill] abort detected mid-prefill, "
                        f"clearing partial for: {aborted_rids}"
                    )
                    self._partial = None
                    mx.clear_cache()
                    return self._generation_step()

            tic = _time.perf_counter()
            partial = self._partial
            inputs = partial["inputs"]
            prompt_cache = partial["cache"]
            remaining = inputs.shape[1]
            prompt_checkpoint = max(1, int(partial.get("prompt_checkpoint", 1)))

            n_to_process = (
                min(budget, remaining - prompt_checkpoint)
                if remaining > prompt_checkpoint
                else 0
            )

            if n_to_process > 0:
                self.model(mx.contiguous(inputs[:, :n_to_process]), cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                inputs = inputs[:, n_to_process:]
                partial["inputs"] = inputs
                partial["processed"] += n_to_process

                self.prompt_progress_callback(
                    [
                        (uid, partial["processed"], partial["total"])
                        for uid in partial["uids"]
                    ]
                )

                # Save intermediate cache for disconnect resilience
                if mid_prefill_save is not None and len(partial["uids"]) == 1:
                    mid_prefill_save(
                        partial["uids"][0], partial["processed"], prompt_cache
                    )

                if partial.get("is_cached"):
                    mx.clear_cache()

            # Check if prefill is done once only the checkpoint tail remains.
            if inputs.shape[1] <= prompt_checkpoint:
                # Finalize
                if partial.get("is_cached"):
                    mx.eval([c.state for c in prompt_cache])
                    inputs = partial["last_inputs"]

                for c in prompt_cache:
                    c.finalize()

                if self.prompt_checkpoint_callback is not None:
                    self.prompt_checkpoint_callback(
                        [
                            (
                                uid,
                                prompt_checkpoint,
                                _lazy_extract_cache(prompt_cache, i),
                            )
                            for i, uid in enumerate(partial["uids"])
                        ]
                    )
                mx.clear_cache()

                # Mirror upstream BatchGenerator semantics: after finalize() and
                # the checkpoint callback, replay the remaining checkpoint tail
                # except for the final token, which _step() consumes.
                if prompt_checkpoint > 1:
                    self.model(
                        mx.contiguous(inputs[:, : prompt_checkpoint - 1]),
                        cache=prompt_cache,
                    )
                    mx.eval([c.state for c in prompt_cache])
                    mx.clear_cache()

                y, logprobs = self._step(
                    inputs,
                    prompt_cache,
                    partial["samplers"],
                    partial["logits_processors"],
                    partial["tokens"],
                )
                mx.async_eval(y, logprobs)

                new_batch = _batch_cls(
                    list(partial["uids"]),
                    y,
                    list(logprobs),
                    list(partial["max_tokens"]),
                    [0] * len(partial["uids"]),
                    prompt_cache,
                    list(partial["samplers"]),
                    list(partial["logits_processors"]),
                    partial["tokens"],
                )

                # Save prompt-only cache BEFORE merging into active batch.
                # This is the chunked-prefill equivalent of the
                # _patched_process_prompts hook — at this point the cache
                # contains the exact prompt-only state (num_tokens == 0).
                if prompt_cache_save is not None and len(partial["uids"]) == 1:
                    uid = partial["uids"][0]
                    try:
                        prompt_cache_save(uid, new_batch.extract_cache(0))
                    except Exception:
                        pass

                if self.active_batch is None:
                    self.active_batch = new_batch
                else:
                    self.active_batch.extend(new_batch)

                self._partial = None
                self._stats.prompt_time += _time.perf_counter() - tic
            else:
                # Not done yet — record prompt time for this chunk
                self._stats.prompt_time += _time.perf_counter() - tic

            # Generation step for active requests between chunks
            return self._generation_step()

        # ----- No partial — check if next prompt batch needs chunking -----
        num_active = len(self.active_batch) if self.active_batch else 0
        num_to_add = self.completion_batch_size - num_active

        if num_to_add >= self.prefill_batch_size and self.unprocessed_prompts:
            batch_prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            if batch_prompts:
                total_tokens = sum(len(p[1]) for p in batch_prompts)

                # Check if any prompt has a prefix_boundary that
                # requires two-phase prefill for cache save at that boundary.
                _needs_boundary_split = False
                if requests is not None and uid_to_request_id is not None:
                    for _uid, _toks, *_ in batch_prompts:
                        _rid = uid_to_request_id.get(_uid)
                        _req = requests.get(_rid) if _rid else None
                        if _req and getattr(_req, "prefix_boundary", 0) > 0:
                            _needs_boundary_split = True
                            break

                if total_tokens > budget or _needs_boundary_split:
                    # Large prompt batch or prefix boundary — start partial prefill
                    tic = _time.perf_counter()

                    # Eval outstanding generation tokens before switching.
                    # Also drain pending async_eval when active_batch is None
                    # (previous request finished) — stale async_eval work on
                    # generation_stream can block subsequent model forwards.
                    if self.active_batch is not None:
                        mx.eval(self.active_batch.y, self.active_batch.logprobs)
                        self._stats.generation_time += _time.perf_counter() - tic
                        tic = _time.perf_counter()
                    else:
                        mx.clear_cache()

                    (
                        uids,
                        inputs_raw,
                        max_tokens_list,
                        caches,
                        samplers,
                        logits_processors,
                        prompt_checkpoints,
                    ) = zip(*batch_prompts)
                    lengths = [len(p) for p in inputs_raw]
                    max_length = max(lengths)
                    padding = [max_length - ln for ln in lengths]
                    tokens = [mx.array(inp) for inp in inputs_raw]
                    # Match mlx-lm's prompt_checkpoint contract: positive values
                    # name the checkpoint token position in the prompt, while
                    # non-positive values already encode an offset from the end.
                    checkpoint_offsets = [
                        (ln - pc if pc > 0 else -pc)
                        for ln, pc in zip(lengths, prompt_checkpoints)
                    ]
                    prompt_checkpoint = max(1, max(checkpoint_offsets))
                    is_cached = not all(c[0].empty() for c in caches)

                    self._stats.prompt_tokens += sum(lengths)

                    if not is_cached:
                        padded = _left_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _make_cache(
                            self.model, padding, self.max_kv_size
                        )
                    else:
                        last_inputs = mx.array(
                            [p[-prompt_checkpoint:] for p in inputs_raw]
                        )
                        padded = _right_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _merge_caches(caches)
                        for c in prompt_cache:
                            c.prepare(
                                lengths=[ln - prompt_checkpoint for ln in lengths],
                                right_padding=padding,
                            )

                    # Remove from unprocessed
                    self.unprocessed_prompts = self.unprocessed_prompts[
                        self.prefill_batch_size :
                    ]

                    # Process first chunk — if prefix_boundary is set,
                    # use it as the first chunk size so that mid_prefill_save
                    # can capture the exact prefix cache state (critical for
                    # hybrid Mamba+Transformer models where trim is unsafe).
                    # When the request already has cached tokens (cache hit),
                    # adjust the boundary relative to the remaining tokens.
                    _first_chunk = budget
                    if _needs_boundary_split and len(batch_prompts) == 1:
                        _uid0 = uids[0]
                        _rid0 = uid_to_request_id.get(_uid0)
                        _req0 = requests.get(_rid0) if _rid0 else None
                        _pb = getattr(_req0, "prefix_boundary", 0) if _req0 else 0
                        _cached = getattr(_req0, "cached_tokens", 0) if _req0 else 0
                        _adjusted_pb = _pb - _cached
                        if 0 < _adjusted_pb < padded.shape[1] - prompt_checkpoint + 1:
                            _first_chunk = _adjusted_pb
                    n_to_process = min(
                        _first_chunk, padded.shape[1] - prompt_checkpoint
                    )
                    if n_to_process > 0:
                        self.model(
                            mx.contiguous(padded[:, :n_to_process]),
                            cache=prompt_cache,
                        )
                        mx.eval([c.state for c in prompt_cache])
                        padded = padded[:, n_to_process:]
                        if is_cached:
                            mx.clear_cache()

                    self._partial = {
                        "uids": list(uids),
                        "inputs": padded,
                        "cache": prompt_cache,
                        "tokens": tokens,
                        "max_tokens": list(max_tokens_list),
                        "samplers": list(samplers),
                        "logits_processors": list(logits_processors),
                        "prompt_checkpoint": prompt_checkpoint,
                        "processed": n_to_process,
                        "total": max_length,
                        "is_cached": is_cached,
                    }
                    if is_cached:
                        self._partial["last_inputs"] = last_inputs

                    self.prompt_progress_callback(
                        [
                            (uid, n_to_process, max_length)
                            for uid in self._partial["uids"]
                        ]
                    )

                    # Save intermediate cache for disconnect resilience
                    if mid_prefill_save is not None and len(uids) == 1:
                        mid_prefill_save(uids[0], n_to_process, prompt_cache)

                    self._stats.prompt_time += _time.perf_counter() - tic

                    # Generation step for active requests
                    return self._generation_step()

                else:
                    # Small prompt batch — process directly without _orig_next.
                    # _orig_next's while loop processes multiple batches per call
                    # which causes batch-dimension mismatches in DeltaRNN conv_state
                    # when mixing prefix-cached and fresh prompts.
                    # Processing one batch per _next call avoids this.
                    tic = _time.perf_counter()

                    # Eval outstanding generation tokens before prefill.
                    # Also drain when active_batch is None to clear stale
                    # async_eval work from the previous request.
                    if self.active_batch is not None:
                        mx.eval(self.active_batch.y, self.active_batch.logprobs)
                        self._stats.generation_time += _time.perf_counter() - tic
                        tic = _time.perf_counter()
                    else:
                        mx.clear_cache()

                    new_batch = self._process_prompts(batch_prompts)
                    self.unprocessed_prompts = self.unprocessed_prompts[
                        self.prefill_batch_size :
                    ]

                    if self.active_batch is None:
                        self.active_batch = new_batch
                    else:
                        self.active_batch.extend(new_batch)

                    self._stats.prompt_time += _time.perf_counter() - tic
                    return self._generation_step()

        # Pure generation or no work — run generation step directly
        return self._generation_step()

    def _patched_remove(uids_to_remove, _self=batch_gen):
        """Clear partial state if aborted request is being prefilled."""
        if _self._partial is not None:
            partial_uids = set(_self._partial["uids"])
            if partial_uids & set(uids_to_remove):
                logger.info(
                    f"[chunked_prefill] clearing partial state for aborted uids: "
                    f"{partial_uids & set(uids_to_remove)}"
                )
                _self._partial = None
                mx.clear_cache()  # flush Metal encoders after dropping partial state
        _orig_remove(uids_to_remove)

    batch_gen._next = _chunked_next
    batch_gen._generation_step = _generation_step
    batch_gen.remove = _patched_remove

    logger.info(f"[chunked_prefill] installed with budget={budget} tokens per step")


def _install_mtp(
    batch_gen: "BatchGenerator",
    model: Any,
    num_draft_tokens: int = 1,
    optimistic: bool = False,
) -> None:
    """
    Monkey-patch a BatchGenerator to use MTP (Multi-Token Prediction)
    with always-advance strategy for hybrid MambaCache + KVCache.

    Flow per generation step:
    1. Use skip_state logits/hidden OR run model forward -> sample primary
    2. MTP head drafts one token after primary
    3. Verify [primary, draft] in one model call (always advances cache)
    4. Accept: skip_state from pos 1, defer draft for next step emission
       Reject: trim KVCache by 1, skip_state from pos 0 (no cold start)
    5. Draft is emitted in the NEXT generation step after primary
    """
    _orig_step = batch_gen._step

    # Greedy sampler for MTP draft tokens
    _draft_sampler = make_sampler(temp=0.0)

    # Skip state: when MTP accepts, the cache already consumed [primary, draft].
    # Next _step call receives primary as input but must NOT re-feed it.
    # Instead, use stored logits from the verify pass.
    # Format: {'logits': (B, V), 'hidden': (B, 1, H)}
    _skip_state = [None]

    # Deferred drafts: draft tokens to emit in the NEXT generation step,
    # keyed by UID for stability across batch changes.
    # Format: {uid: {'token': int, 'logprobs': mx.array}}
    _deferred_drafts = {}

    # MTP stats
    _mtp_stats = {"accepted": 0, "rejected": 0, "errors": 0}

    def _mtp_step(
        input_tokens,
        prompt_cache,
        samplers,
        logits_processors,
        tokens,
    ):
        """
        Extended _step with MTP always-advance strategy.

        Every step (after skip):
        1. Use skip_state logits/hidden OR run model forward
        2. Sample primary token P
        3. MTP head drafts token D
        4. Verify [P, D] in one model call (always advances cache)
        5. Accept: skip_state from position 1 (after D), defer D
           Reject: trim KVCache by 1, skip_state from position 0 (after P)

        No snapshot/restore — eliminates cold starts after rejection.
        MambaCache layers accept minor pollution on reject (exponential decay).

        During prefill (multi-token input), MTP is skipped entirely.
        """
        batch_size = input_tokens.shape[0]

        # --- Prefill guard: skip MTP for multi-token input,
        # during _process_prompts (active_batch not yet set), or when
        # the cache doesn't belong to the active batch (e.g. during
        # _process_prompts in the 2nd+ iteration of _orig_next's loop
        # or during _chunked_next partial prefill finalization).
        if (
            input_tokens.shape[1] > 1
            or batch_gen.active_batch is None
            or prompt_cache is not batch_gen.active_batch.cache
        ):
            _skip_state[0] = None
            return _orig_step(
                input_tokens,
                prompt_cache,
                samplers,
                logits_processors,
                tokens,
            )

        # --- Check skip state from previous MTP step ---
        skip = _skip_state[0]
        if skip is not None:
            if skip["logits"].shape[0] != batch_size:
                # Batch size changed since skip was stored — invalidate
                skip = None
                _skip_state[0] = None

        if skip is not None:
            # Skip mode: model already processed input_tokens during
            # previous verify. Use stored logits + hidden instead.
            logits = skip["logits"]
            hidden_states = skip["hidden"]
            _skip_state[0] = None
        else:
            # Normal model forward
            model_output = model(input_tokens, cache=prompt_cache, return_hidden=True)
            if isinstance(model_output, tuple):
                logits, hidden_states = model_output
            else:
                # Model doesn't support return_hidden — fall back
                return _orig_step(
                    input_tokens,
                    prompt_cache,
                    samplers,
                    logits_processors,
                    tokens,
                )
            logits = logits[:, -1, :]

        # --- Apply logits processors + sample primary ---
        logits_processors = _normalize_logits_processors(logits_processors) or []
        if any(logits_processors):
            logger.debug(
                f"[logits_proc] applying {sum(len(lp) for lp in logits_processors)} "
                f"processors to batch_size={batch_size}"
            )
            processed_logits = []
            for e in range(batch_size):
                sample_logits = logits[e : e + 1]
                for processor in logits_processors[e]:
                    sample_logits = processor(tokens[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if any(samplers):
            all_samples = []
            for e in range(batch_size):
                sample_sampler = samplers[e] or batch_gen.sampler
                sampled = sample_sampler(logprobs[e : e + 1])
                all_samples.append(sampled)
            primary_tokens = mx.concatenate(all_samples, axis=0)
        else:
            primary_tokens = batch_gen.sampler(logprobs)

        # Get current UIDs (guaranteed non-empty: prefill guard above
        # prevents MTP from running when active_batch is None).
        current_uids = list(batch_gen.active_batch.uids)

        # --- MTP draft + always-advance verify ---
        try:
            # Draft: predict token n+2 from hidden states + primary (n+1)
            draft_logits = model.mtp_forward(
                hidden_states[:, -1:, :],
                primary_tokens[:, None],
                mtp_cache=None,
            )
            draft_logits = draft_logits[:, -1, :]
            draft_logprobs = draft_logits - mx.logsumexp(
                draft_logits, axis=-1, keepdims=True
            )
            draft_tokens = _draft_sampler(draft_logprobs)

            # Always-advance: feed [primary, draft] and let cache advance.
            #
            # Hybrid models (e.g. Qwen3-Next) mix attention (KVCache) and
            # recurrent layers (MambaCache/DeltaRNN).  KVCache supports
            # trim(1) to undo the draft token on reject, but recurrent
            # state is irreversible — rejected drafts permanently pollute
            # the RNN state, causing progressive output corruption.
            #
            # For hybrid models we snapshot recurrent state before verify
            # and on reject: trim KV by 2 (remove both P and D), restore
            # RNN snapshot, then re-advance with just P so both cache
            # types end up consistent at [..., P].
            # Skip RNN snapshots in optimistic mode — it never rejects,
            # so the copies are wasted (~147 MB/step of lazy graph nodes
            # that prevent pre-verify Metal buffers from being freed).
            _rnn_snapshots = {}
            if not optimistic:
                for _ci, _c in enumerate(prompt_cache):
                    if not (hasattr(_c, "is_trimmable") and _c.is_trimmable()):
                        if hasattr(_c, "state"):
                            _rnn_snapshots[_ci] = [
                                s.copy() if s is not None else None for s in _c.state
                            ]

            verify_input = mx.concatenate(
                [primary_tokens[:, None], draft_tokens[:, None]], axis=1
            )
            verify_output = model(verify_input, cache=prompt_cache, return_hidden=True)
            if isinstance(verify_output, tuple):
                verify_logits, verify_hidden = verify_output
            else:
                verify_logits = verify_output
                verify_hidden = None

            if optimistic:
                # --- OPTIMISTIC: always accept, zero sync ---
                if verify_hidden is not None:
                    _skip_state[0] = {
                        "logits": verify_logits[:, 1, :],
                        "hidden": verify_hidden[:, -1:, :],
                    }
                    verify_lp = verify_logits[:, 0, :] - mx.logsumexp(
                        verify_logits[:, 0, :], axis=-1, keepdims=True
                    )
                    mx.async_eval(
                        _skip_state[0]["logits"],
                        _skip_state[0]["hidden"],
                        draft_tokens,
                        verify_lp,
                    )
                    for e in range(batch_size):
                        uid = current_uids[e]
                        _deferred_drafts[uid] = {
                            "token_array": draft_tokens[e : e + 1],
                            "logprobs": verify_lp[e],
                        }
                else:
                    _skip_state[0] = None
                _mtp_stats["accepted"] += 1
            else:
                # --- VERIFIED MODE: single eval + Python comparison ---
                verify_pred = mx.argmax(verify_logits[:, 0, :], axis=-1)
                mx.eval(verify_pred, draft_tokens)
                pred_list = verify_pred.tolist()
                draft_list = draft_tokens.tolist()
                all_accepted = pred_list == draft_list

                if all_accepted and verify_hidden is not None:
                    # --- ACCEPT ---
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
                    _mtp_stats["accepted"] += 1

                else:
                    # --- REJECT (always-advance) ---
                    if _rnn_snapshots:
                        # Hybrid model: undo the entire verify pass
                        # (both P and D) for all cache types, then
                        # re-advance with just P for a consistent state.
                        for c in prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(2)
                        for _ci, _snap in _rnn_snapshots.items():
                            prompt_cache[_ci].state = _snap
                        # Re-advance with primary only — both KV and RNN
                        # now advance by exactly 1 (the primary token).
                        rerun_out = model(
                            primary_tokens[:, None],
                            cache=prompt_cache,
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
                        # Pure attention model: simple trim(1) is enough.
                        for c in prompt_cache:
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
                    _mtp_stats["rejected"] += 1

        except Exception as e:
            logger.debug(f"[MTP] draft/verify failed: {e}")
            _skip_state[0] = None
            _mtp_stats["errors"] += 1

        return primary_tokens, list(logprobs)

    # Wrap _next() to emit deferred MTP drafts after each primary token.
    # This works regardless of whether _chunked_next or original _next is
    # the current _next implementation, because it sits at the top level.
    # Store as attribute so it's always the correct reference, even after
    # BatchGenerator recreation.
    batch_gen._inner_next = batch_gen._next

    def _mtp_next(self=batch_gen):
        """Wrapper around _next that emits deferred MTP draft tokens.

        After each primary token, if the previous step's MTP draft was
        accepted, it is emitted as an additional response.
        """
        # Clear stale MTP state when no batch is active.
        # This prevents skip_state/deferred_drafts from a finished request
        # from leaking into the next request and causing stale computation
        # graph references on generation_stream.
        if self.active_batch is None:
            _skip_state[0] = None
            _deferred_drafts.clear()

        # Save deferred drafts from PREVIOUS step before _inner_next
        # runs _mtp_step, which may store NEW deferred drafts.
        prev_deferred = {}
        if self.active_batch is not None:
            for uid in self.active_batch.uids:
                if uid in _deferred_drafts:
                    prev_deferred[uid] = _deferred_drafts.pop(uid)

        # Run the inner _next (original or chunked) — calls _mtp_step
        responses = self._inner_next()

        if not prev_deferred or not responses:
            return responses

        # Augment responses with deferred drafts from the previous step.
        # The Response from _next reports the OLD batch.y (the primary
        # from the *previous* _step call). The deferred draft follows
        # that primary in the token stream, so emit it AFTER the primary.
        augmented = []
        draft_end_uids = set()
        for r in responses:
            uid = r.uid

            # Emit the primary response first
            augmented.append(r)

            if r.finish_reason is not None:
                # Sequence ended with primary — discard any pending draft
                _deferred_drafts.pop(uid, None)
                prev_deferred.pop(uid, None)
                continue

            # Emit deferred draft AFTER its primary
            if uid in prev_deferred:
                draft_info = prev_deferred.pop(uid)
                if "token" in draft_info:
                    draft_t = draft_info["token"]
                else:
                    draft_t = draft_info["token_array"].item()
                draft_lp = draft_info["logprobs"]

                if draft_t in self.stop_tokens:
                    augmented.append(
                        self.Response(uid, draft_t, draft_lp, "stop", None)
                    )
                    draft_end_uids.add(uid)
                else:
                    draft_finish = None
                    batch = self.active_batch
                    if batch is not None:
                        for e, bu in enumerate(batch.uids):
                            if bu == uid:
                                batch.num_tokens[e] += 1
                                batch.tokens[e] = mx.concatenate(
                                    (batch.tokens[e], mx.array([draft_t]))
                                )
                                if batch.num_tokens[e] >= batch.max_tokens[e]:
                                    draft_finish = "length"
                                    draft_end_uids.add(uid)
                                break

                    draft_cache_out = None
                    if draft_finish is not None and batch is not None:
                        for e, bu in enumerate(batch.uids):
                            if bu == uid:
                                draft_cache_out = batch.extract_cache(e)
                                break

                    augmented.append(
                        self.Response(
                            uid, draft_t, draft_lp, draft_finish, draft_cache_out
                        )
                    )

        # Remove sequences that finished due to draft tokens
        if draft_end_uids and self.active_batch is not None:
            keep = [
                e
                for e, u in enumerate(self.active_batch.uids)
                if u not in draft_end_uids
            ]
            if keep:
                self.active_batch.filter(keep)
            else:
                self.active_batch = None

        return augmented

    batch_gen._step = _mtp_step
    batch_gen._next = _mtp_next

    if num_draft_tokens != 1:
        logger.warning(
            "[MTP] num_draft_tokens=%d requested, but the current batched MTP "
            "path drafts exactly one token per verify step",
            num_draft_tokens,
        )
    mode_str = "optimistic (no verify)" if optimistic else "always-advance"
    logger.info(
        f"[MTP] installed with num_draft_tokens={num_draft_tokens}, "
        f"effective_draft_tokens=1, {mode_str} mode"
    )


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SchedulerConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # Per-request streaming detokenizers for UTF-8-safe incremental decode
        self._detokenizer_pool: Dict[str, Any] = {}

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, Request] = {}  # Running requests by ID
        self.requests: Dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: Optional[BatchGenerator] = None
        self._current_sampler_params: Optional[Tuple] = None

        # Prefix cache for KV state reuse
        self.prefix_cache: Optional[PrefixCacheManager] = None
        self.memory_aware_cache: Optional[MemoryAwarePrefixCache] = None
        self.paged_cache_manager: Optional[PagedCacheManager] = None
        self.block_aware_cache: Optional[BlockAwarePrefixCache] = None
        self._ssd_tier: Optional[SSDCacheTier] = None

        if self.config.enable_prefix_cache:
            if self.config.use_paged_cache:
                # Use paged cache for memory efficiency
                self.paged_cache_manager = PagedCacheManager(
                    block_size=self.config.paged_cache_block_size,
                    max_blocks=self.config.max_cache_blocks,
                )
                self.block_aware_cache = BlockAwarePrefixCache(
                    model=model,
                    paged_cache_manager=self.paged_cache_manager,
                )
                logger.info(
                    f"Paged cache enabled: block_size={self.config.paged_cache_block_size}, "
                    f"max_blocks={self.config.max_cache_blocks}"
                )
            elif self.config.use_memory_aware_cache:
                # Use memory-aware cache (recommended for large models)
                cache_config = MemoryCacheConfig(
                    max_memory_mb=self.config.cache_memory_mb,
                    max_memory_percent=self.config.cache_memory_percent,
                    kv_quantize=self.config.kv_cache_quantization,
                    kv_bits=self.config.kv_cache_quantization_bits,
                    kv_group_size=self.config.kv_cache_quantization_group_size,
                    kv_min_quantize_tokens=self.config.kv_cache_min_quantize_tokens,
                )
                self.memory_aware_cache = MemoryAwarePrefixCache(
                    model=model,
                    config=cache_config,
                )
                logger.info(
                    f"Memory-aware cache enabled: "
                    f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB"
                )

                if self.config.ssd_cache_dir is not None:
                    ssd_config = SSDCacheConfig(
                        cache_dir=self.config.ssd_cache_dir,
                        max_size_gb=self.config.ssd_cache_max_gb,
                    )
                    self._ssd_tier = SSDCacheTier(ssd_config)
                    self._ssd_tier.start_writer()
                    self._ssd_tier.reconcile()
                    self.memory_aware_cache.set_ssd_tier(self._ssd_tier)
                    logger.info(
                        f"SSD cache tier enabled: dir={self.config.ssd_cache_dir}, "
                        f"max={self.config.ssd_cache_max_gb}GB"
                    )
            else:
                # Use legacy entry-count based prefix cache
                self.prefix_cache = PrefixCacheManager(
                    model=model,
                    max_entries=self.config.prefix_cache_size,
                )
                logger.info(
                    f"Prefix cache enabled with max_entries={self.config.prefix_cache_size}"
                )

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: Set[str] = set()

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Memory management: periodic mx.clear_cache() to free Metal command buffers
        # Lower interval = less VRAM spike during generation but slight throughput cost
        self._step_count = 0
        self._clear_cache_interval = 32
        self._memory_log_interval = 256

    def _get_actual_tokenizer(self, tokenizer: Any) -> Any:
        """
        Get the actual tokenizer from a processor or tokenizer.

        MLLM models use processors (e.g., Qwen3VLProcessor) which wrap
        the tokenizer. This method extracts the actual tokenizer.
        """
        # If it has encode method, it's already a tokenizer
        if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
            return tokenizer
        # If it's a processor, get the wrapped tokenizer
        if hasattr(tokenizer, "tokenizer"):
            return tokenizer.tokenizer
        # Fallback to the original
        return tokenizer

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text, handling both tokenizers and processors.
        """
        return self._actual_tokenizer.decode(token_ids)

    def _get_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
            self._detokenizer_pool[request_id] = detok
        return self._detokenizer_pool[request_id]

    def _cleanup_detokenizer(self, request_id: str) -> None:
        """Remove the streaming detokenizer for a finished request."""
        self._detokenizer_pool.pop(request_id, None)

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer or processor."""
        stop_tokens = set()
        # Check both the processor/tokenizer and the actual tokenizer
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                if isinstance(tok.eos_token_id, list):
                    stop_tokens.update(tok.eos_token_id)
                else:
                    stop_tokens.add(tok.eos_token_id)
            if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
                if isinstance(tok.eos_token_ids, (list, set, tuple)):
                    stop_tokens.update(tok.eos_token_ids)
                else:
                    # Handle case where eos_token_ids is a single int
                    stop_tokens.add(tok.eos_token_ids)
        return stop_tokens

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
        )

        stop_tokens = self._get_stop_tokens()
        # Add custom stop token IDs
        if sampling_params.stop_token_ids:
            stop_tokens.update(sampling_params.stop_token_ids)

        def _prefill_progress(progress_list):
            """Log prefill progress for each uid chunk."""
            for uid, processed, total in progress_list:
                rid = self.uid_to_request_id.get(uid, "?")
                logger.info(
                    f"[prefill] request={rid[:12] if isinstance(rid, str) else rid} "
                    f"tokens={processed}/{total}"
                )

        bg = BatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
        )
        # Set callback as attribute — used by _install_chunked_prefill
        # monkey-patch. Not a BatchGenerator constructor parameter.
        bg.prompt_progress_callback = _prefill_progress

        # Install chunked prefill only when explicitly configured.
        # memory_aware_cache fetch/store works independently; the mid-prefill
        # save callback is an optimisation, not a requirement.
        # When chunked_prefill_tokens == 0 (the default), honour the user's
        # intent — do NOT silently re-enable chunked prefill just because
        # memory_aware_cache is active (see #178).
        chunked_budget = self.config.chunked_prefill_tokens
        need_chunked = chunked_budget > 0

        # The chunked prefill monkey-patch relies on BatchGenerator internals
        # (_process_prompts, active_batch, _step, etc.) that were refactored
        # in mlx-lm 0.31.x.  Skip gracefully when the required API is absent.
        chunked_compatible = hasattr(bg, "_process_prompts") and hasattr(
            bg, "active_batch"
        )

        prompt_cache_cb = None
        if self.memory_aware_cache is not None:
            prompt_cache_cb = self._make_prompt_cache_save_callback()

        if need_chunked and chunked_compatible:
            # Full chunked prefill with mid-prefill saves and prompt cache
            # save wired through the chunked next() and _process_prompts
            # monkey-patches inside _install_chunked_prefill.
            mid_prefill_cb = None
            save_interval = self.config.mid_prefill_save_interval
            if save_interval > 0 and self.memory_aware_cache is not None:
                mid_prefill_cb = self._make_mid_prefill_save_callback(save_interval)
                logger.info(f"[mid_prefill_cache] enabled, interval={save_interval}")
            _install_chunked_prefill(
                bg,
                chunked_budget,
                mid_prefill_cb,
                prompt_cache_save=prompt_cache_cb,
                pending_abort_ids=self._pending_abort_ids,
                uid_to_request_id=self.uid_to_request_id,
                requests=self.requests,
            )
        elif need_chunked and not chunked_compatible:
            logger.warning(
                "Chunked prefill disabled: mlx-lm BatchGenerator lacks required "
                "internals (_process_prompts, active_batch). Upgrade mlx-lm or "
                "check compatibility."
            )

        # When chunked prefill is off but memory_aware_cache is active,
        # install the lightweight _process_prompts hook so prompt-only
        # cache entries are still captured.  This is the only safe capture
        # point for hybrid Mamba+Transformer models (#178).
        if not need_chunked and prompt_cache_cb is not None:
            if hasattr(bg, "_process_prompts"):
                _install_prompt_cache_save(bg, prompt_cache_cb)

        # Install MTP if the model supports it
        if self.config.enable_mtp:
            if hasattr(self.model, "mtp") and self.model.mtp is not None:
                _install_mtp(
                    bg,
                    model=self.model,
                    num_draft_tokens=self.config.mtp_num_draft_tokens,
                    optimistic=self.config.mtp_optimistic,
                )
            else:
                logger.warning(
                    "[MTP] --enable-mtp is set but model has no MTP head "
                    "(model.mtp is None). MTP will be disabled."
                )

        return bg

    def _make_prompt_cache_save_callback(self):
        """Create a callback that stores prompt-only KV/Mamba cache.

        Called from ``_generation_step`` right before the first output token
        is fed into the model.  At that point ``num_tokens == 0`` and the
        batch cache contains the exact prompt-only state (correct for both
        KVCache and MambaCache/ArraysCache layers).

        The cache is stored with key = prompt_token_ids so that a future
        request with the identical prompt gets an exact hit.
        """
        import time as _time

        def _prompt_cache_save(uid, extracted_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return

            prompt_tokens = list(request.prompt_token_ids)
            # Trim cache by 1 so the stored KV has offset = N-1.
            # On exact fetch the scheduler sends the last prompt token
            # for reprocessing (lines 1872-1877).  Without this trim
            # the last token would be placed at position N instead of N-1.
            from .memory_cache import _trim_cache_offset

            trimmed_cache = _trim_cache_offset(extracted_cache, 1)
            _t0 = _time.monotonic()
            # evict_prefixes=False: keep mid-prefill boundary entries so
            # that future requests with the same prefix but different
            # suffix get a prefix cache hit (critical for agentic multi-turn).
            stored = self.memory_aware_cache.store(
                prompt_tokens, trimmed_cache, evict_prefixes=False
            )
            _dt = _time.monotonic() - _t0
            if stored:
                logger.info(
                    f"[prompt_cache_save] request={request_id[:12]} "
                    f"prompt_tokens={len(prompt_tokens)} "
                    f"store_time={_dt:.3f}s"
                )

        return _prompt_cache_save

    def _make_mid_prefill_save_callback(self, save_interval: int):
        """Create a callback for saving intermediate KV cache during chunked prefill.

        The callback is called after each chunk with (uid, processed_tokens,
        prompt_cache).  It extracts the cache state (immutable MLX array
        snapshots), reconstructs KVCache objects, and stores them in the
        memory-aware prefix cache so that a subsequent request with the same
        prompt prefix can skip the already-computed tokens.
        """
        import time as _time

        def _mid_prefill_save(uid, processed_tokens, prompt_cache):
            request_id = self.uid_to_request_id.get(uid)
            if not request_id:
                return
            request = self.requests.get(request_id)
            if not request or not request.prompt_token_ids:
                return

            total_cached = (request.cached_tokens or 0) + processed_tokens

            # Always save at prefix_boundary (message boundary for cache
            # reuse with different final user messages).
            prefix_boundary = getattr(request, "prefix_boundary", 0)
            at_prefix_boundary = prefix_boundary > 0 and total_cached == prefix_boundary

            # Throttle: only save every save_interval tokens,
            # unless we're at the prefix boundary.
            last_save = getattr(request, "_mid_prefill_last_save", 0)
            if not at_prefix_boundary and total_cached - last_save < save_interval:
                return

            # Extract immutable state snapshots
            extracted = self._extract_cache_states(prompt_cache)
            if not extracted:
                return

            # Reconstruct cache objects (directly usable by BatchGenerator)
            reconstructed = self._reconstruct_cache_from_states(extracted)
            if not reconstructed:
                return

            prefix_tokens = list(request.prompt_token_ids[:total_cached])

            # Remove previous intermediate entry to avoid memory waste
            old_key = getattr(request, "_mid_prefill_cache_key", None)
            if old_key is not None:
                self.memory_aware_cache.remove(list(old_key))

            _t0 = _time.monotonic()
            stored = self.memory_aware_cache.store(prefix_tokens, reconstructed)
            _dt = _time.monotonic() - _t0

            if stored:
                request._mid_prefill_last_save = total_cached
                request._mid_prefill_cache_key = tuple(prefix_tokens)
                logger.info(
                    f"[mid_prefill_cache] request={request_id[:12]} "
                    f"saved {total_cached}/{len(request.prompt_token_ids)} tokens "
                    f"({total_cached * 100 // len(request.prompt_token_ids)}%) "
                    f"store_time={_dt:.3f}s"
                )
            else:
                logger.debug(
                    f"[mid_prefill_cache] request={request_id[:12]} "
                    f"store rejected for {total_cached} tokens"
                )

        return _mid_prefill_save

    def _close_batch_generator(self) -> None:
        """Properly close BatchGenerator to restore wired_limit."""
        if self.batch_generator is not None:
            try:
                if hasattr(self.batch_generator, "close"):
                    self.batch_generator.close()
            except Exception as e:
                logger.debug(f"Error closing BatchGenerator: {e}")
            self.batch_generator = None

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> None:
        """Ensure BatchGenerator exists with compatible settings."""
        sampler_params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
        )

        # Create new generator if needed or if sampling params changed
        if (
            self.batch_generator is None
            or self._current_sampler_params != sampler_params
        ):
            # If we have an existing generator with requests, we need to drain it first
            if self.batch_generator is not None and self.running:
                logger.warning(
                    "Sampling parameters changed with active requests. "
                    "New requests will use new parameters after current batch completes."
                )
                return

            # Keep prefix cache across BatchGenerator recreations.
            # KV cache entries depend only on the input tokens, not on
            # sampling params (temperature, top_p, min_p).  Since the
            # server runs a single model, the cache is always valid.
            if self.batch_generator is not None:
                n_entries = 0
                if self.memory_aware_cache is not None:
                    n_entries = len(self.memory_aware_cache._entries)
                elif self.prefix_cache is not None:
                    n_entries = (
                        len(self.prefix_cache)
                        if hasattr(self.prefix_cache, "__len__")
                        else 0
                    )
                logger.info(
                    f"[batch_generator] recreating (sampler params changed), "
                    f"keeping {n_entries} cache entries"
                )

            self._close_batch_generator()
            self.batch_generator = self._create_batch_generator(sampling_params)
            self._current_sampler_params = sampler_params

    def _validate_cache(self, cache: Any) -> bool:
        """
        Validate that a cache object is usable.

        Checks for None references AND shape compatibility.  Restored
        cache entries must have batch_size == 1 (single sequence) so
        they can be merged into the running batch by _merge_caches.
        A shape mismatch here (e.g. batch=2 from a stale entry) would
        cause a concatenation crash inside _merge_caches.

        Args:
            cache: The cache object to validate

        Returns:
            True if cache is valid and usable
        """
        if cache is None:
            return False

        # Check if it's a list of cache layers
        if isinstance(cache, list):
            if len(cache) == 0:
                return False
            # Check each layer
            for layer_cache in cache:
                if layer_cache is None:
                    return False
                # Check if layer has expected structure
                if hasattr(layer_cache, "keys") and layer_cache.keys is None:
                    return False
                if hasattr(layer_cache, "values") and layer_cache.values is None:
                    return False
                # Validate batch dimension == 1 for KVCache layers
                if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
                    if layer_cache.keys.shape[0] != 1:
                        logger.debug(
                            f"Cache layer invalid: keys batch={layer_cache.keys.shape[0]}, expected 1"
                        )
                        return False
                # Validate batch dimension for MambaCache layers
                if hasattr(layer_cache, "cache") and isinstance(
                    layer_cache.cache, list
                ):
                    for arr in layer_cache.cache:
                        if arr is not None and arr.shape[0] != 1:
                            logger.debug(
                                f"Cache layer invalid: mamba batch={arr.shape[0]}, expected 1"
                            )
                            return False

        # Check BatchKVCache structure
        if hasattr(cache, "caches"):
            if cache.caches is None:
                return False
            for c in cache.caches:
                if c is None:
                    return False

        return True

    def _extract_cache_states(self, raw_cache: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract actual tensor state from each layer cache.

        This extracts the real KV data using mlx-lm's cache.state property,
        allowing the data to be stored and reconstructed later even after
        the BatchGenerator is recreated.

        Args:
            raw_cache: List of KVCache objects from mlx-lm

        Returns:
            List of dicts with {state: (keys, values), meta_state: (offset,), class_name: str}
        """
        if not raw_cache:
            return []

        extracted = []
        for layer_cache in raw_cache:
            try:
                if hasattr(layer_cache, "state") and hasattr(layer_cache, "meta_state"):
                    state = layer_cache.state  # (keys, values) or more for Mamba
                    meta = layer_cache.meta_state  # (offset,) as strings
                    extracted.append(
                        {
                            "state": state,
                            "meta_state": meta,
                            "class_name": type(layer_cache).__name__,
                            "class_ref": type(layer_cache),
                        }
                    )
            except Exception as e:
                logger.debug(f"Failed to extract state from cache layer: {e}")
                continue

        return extracted if len(extracted) == len(raw_cache) else []

    def _reconstruct_cache_from_states(
        self, extracted_states: List[Dict[str, Any]]
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from extracted cache states.

        This is the inverse of _extract_cache_states(). Uses mlx-lm's
        _BaseCache.from_state() to reconstruct any cache type (KVCache,
        MambaCache, etc.) from its state/meta_state.

        Args:
            extracted_states: List of dicts from _extract_cache_states()

        Returns:
            List of cache objects, or None if reconstruction fails
        """
        if not extracted_states:
            return None

        try:
            caches = []
            for layer_state in extracted_states:
                state = layer_state.get("state")
                meta_state = layer_state.get("meta_state")
                cache_cls = layer_state.get("class_ref")
                if state is None:
                    return None

                if cache_cls is not None and hasattr(cache_cls, "from_state"):
                    # BatchKVCache doesn't inherit from KVCache, so
                    # _merge_caches can't handle it. Convert to KVCache
                    # (safe because mid-prefill save is always batch_size=1).
                    from mlx_lm.models.cache import (
                        BatchKVCache as _BatchKVCache,
                        KVCache as _KVCache,
                    )

                    if cache_cls is _BatchKVCache:
                        # BatchKVCache.state = (keys, values, offset, left_padding)
                        keys, values = state[0], state[1]
                        cache = _KVCache()
                        cache.keys = keys
                        cache.values = values
                        cache.offset = keys.shape[2]
                    else:
                        cache = cache_cls.from_state(state, meta_state)
                else:
                    # Fallback: try KVCache manual reconstruction
                    from mlx_lm.models.cache import KVCache

                    if len(state) != 2:
                        return None
                    cache = KVCache()
                    cache.keys, cache.values = state
                    cache.offset = (
                        int(meta_state[0]) if meta_state else cache.keys.shape[2]
                    )

                caches.append(cache)

            return caches

        except Exception as e:
            logger.info(f"[mid_prefill_cache] reconstruct EXCEPTION: {e}")
            return None

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                # Handle both tokenizers and processors (for MLLM models)
                if hasattr(self.tokenizer, "encode"):
                    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
                elif hasattr(self.tokenizer, "tokenizer") and hasattr(
                    self.tokenizer.tokenizer, "encode"
                ):
                    # Processor wraps tokenizer (e.g., Qwen3VLProcessor)
                    request.prompt_token_ids = self.tokenizer.tokenizer.encode(
                        request.prompt
                    )
                else:
                    raise AttributeError(
                        f"Tokenizer {type(self.tokenizer)} has no 'encode' method. "
                        "Continuous batching requires a tokenizer with encode support."
                    )
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Check prefix cache for cached KV state
        if self.block_aware_cache is not None:
            # Use paged cache
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                request.prompt_token_ids,
            )
            if block_table and block_table.num_tokens > 0:
                request.cache_hit_type = "hit"
                # Reconstruct actual KVCache objects from stored tensor data
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                if reconstructed:
                    request.prompt_cache = reconstructed
                    request.block_table = block_table
                    request.cached_tokens = block_table.num_tokens
                    request.shared_prefix_blocks = len(block_table.block_ids)
                    request.remaining_tokens = remaining
                    logger.debug(
                        f"Request {request.request_id}: paged cache hit, "
                        f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks, "
                        f"{len(remaining)} tokens remaining, cache reconstructed"
                    )
                else:
                    # Reconstruction failed, treat as cache miss
                    request.cache_hit_type = "miss"
                    request.remaining_tokens = request.prompt_token_ids
                    logger.debug(
                        f"Request {request.request_id}: paged cache reconstruction failed"
                    )
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
        elif self.memory_aware_cache is not None:
            # Use memory-aware prefix cache
            import time as _time

            _fetch_t0 = _time.monotonic()
            cache, remaining = self.memory_aware_cache.fetch(request.prompt_token_ids)
            _fetch_dt = _time.monotonic() - _fetch_t0
            request.cache_hit_type = self.memory_aware_cache._last_match_type
            if cache:
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                logger.info(
                    f"[cache_fetch] request={request.request_id[:12]} HIT "
                    f"prompt_tokens={len(request.prompt_token_ids)} "
                    f"cached={request.cached_tokens} remaining={len(remaining)} "
                    f"time={_fetch_dt:.3f}s"
                )
            else:
                request.remaining_tokens = request.prompt_token_ids
                logger.info(
                    f"[cache_fetch] request={request.request_id[:12]} MISS "
                    f"prompt_tokens={len(request.prompt_token_ids)} "
                    f"time={_fetch_dt:.3f}s entries={len(self.memory_aware_cache._entries)}"
                )
                # Check SSD tier for cold-tier hit
                if self._ssd_tier is not None:
                    ssd_candidate = self.memory_aware_cache.check_ssd(
                        request.prompt_token_ids
                    )
                    if ssd_candidate is not None:
                        request.cache_hit_type = "ssd_pending"
                        request._ssd_candidate = ssd_candidate
        elif self.prefix_cache is not None:
            # Use legacy prefix cache
            cache, remaining = self.prefix_cache.fetch_cache(request.prompt_token_ids)
            if cache:
                request.cache_hit_type = "hit"
                request.prompt_cache = cache
                request.cached_tokens = len(request.prompt_token_ids) - len(remaining)
                request.remaining_tokens = remaining
                logger.debug(
                    f"Request {request.request_id}: cache hit, "
                    f"{request.cached_tokens} tokens cached, "
                    f"{len(remaining)} tokens remaining"
                )
            else:
                request.cache_hit_type = "miss"
                request.remaining_tokens = request.prompt_token_ids
        else:
            request.cache_hit_type = "miss"
            request.remaining_tokens = request.prompt_token_ids

        # Add to tracking
        self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def abort_request(self, request_id: str) -> bool:
        """
        Queue request for abort. Thread-safe, called from any thread.

        The actual abort is deferred to the executor thread (inside step())
        to avoid race conditions with in-flight Metal GPU operations.

        Args:
            request_id: The request ID to abort

        Returns:
            True (abort is always enqueued)
        """
        self._pending_abort_ids.add(request_id)
        logger.info(f"[abort_request] {request_id[:12]} enqueued for deferred abort")
        return True

    def _process_pending_aborts(self) -> None:
        """Drain and process pending abort requests. Called from executor thread."""
        while self._pending_abort_ids:
            request_id = self._pending_abort_ids.pop()
            self._do_abort_request(request_id)

    def _do_abort_request(self, request_id: str) -> bool:
        """
        Actually abort a request. Must be called from the executor thread.

        Handles the case where the request was already removed from
        self.requests by _cleanup_request() but still lives in the
        BatchGenerator (e.g. in _partial or active_batch).

        Args:
            request_id: The request ID to abort

        Returns:
            True if any cleanup was performed, False otherwise
        """
        request = self.requests.get(request_id)
        was_waiting = False
        was_running = False
        removed_from_batch = False

        # Remove from waiting queue
        if request is not None and request.status == RequestStatus.WAITING:
            was_waiting = True
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from running (BatchGenerator) — do this even if request
        # was already cleaned up from self.requests, because the UID may
        # still be live inside the BatchGenerator (_partial / active_batch).
        if request_id in self.request_id_to_uid:
            was_running = True
            uid = self.request_id_to_uid[request_id]
            if self.batch_generator is not None:
                self.batch_generator.remove([uid])
                removed_from_batch = True
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Credit in-flight tokens so dashboard metrics stay accurate
        # (without this, aborted requests' tokens vanish from /v1/status).
        if request is not None and request.num_output_tokens > 0:
            self.total_completion_tokens += request.num_output_tokens
            self.total_prompt_tokens += request.num_prompt_tokens

        if request is not None:
            request.set_finished(RequestStatus.FINISHED_ABORTED)
            # Release cache references so Metal buffers can be freed
            request.prompt_cache = None
            request._extracted_cache = None
        self.finished_req_ids.add(request_id)
        self._cleanup_detokenizer(request_id)

        # Flush Metal encoders after removing arrays from batch
        mx.clear_cache()

        logger.info(
            f"[abort_request] {request_id[:12]} ABORTED "
            f"was_waiting={was_waiting} was_running={was_running} "
            f"removed_from_batch={removed_from_batch} "
            f"remaining_running={len(self.running)} remaining_waiting={len(self.waiting)}"
        )
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

    def _schedule_waiting(self) -> List[Request]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        # Attempt synchronous SSD promotion for any ssd_pending requests
        # before scheduling. This keeps SSD I/O out of fetch() while
        # avoiding engine modifications.
        if self._ssd_tier is not None:
            self._try_promote_ssd_pending()

        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            # Determine tokens to process and cache to use
            # Note: Don't use `remaining_tokens or prompt_token_ids` because empty list
            # is falsy in Python. For exact cache match, remaining_tokens=[] but we should
            # pass just the last token so BatchGenerator can start generation.
            if (
                request.remaining_tokens is not None
                and len(request.remaining_tokens) == 0
            ):
                # Exact cache match - pass only last token for generation kickoff
                tokens_to_process = request.prompt_token_ids[-1:]
            elif request.remaining_tokens:
                tokens_to_process = request.remaining_tokens
            else:
                tokens_to_process = request.prompt_token_ids
            cache_to_use = request.prompt_cache  # May be None

            # Create bounded cache when max_kv_size is configured and no cache exists
            if cache_to_use is None and self.config.max_kv_size > 0:
                from mlx_lm.models.cache import make_prompt_cache

                cache_to_use = make_prompt_cache(
                    self.model, max_kv_size=self.config.max_kv_size
                )

            # Validate cache before using it
            if cache_to_use is not None and not self._validate_cache(cache_to_use):
                logger.debug(
                    f"Request {request.request_id}: invalid cache detected, "
                    f"proceeding without cache"
                )
                cache_to_use = None
                request.prompt_cache = None
                request.cached_tokens = 0
                request.remaining_tokens = request.prompt_token_ids
                tokens_to_process = request.prompt_token_ids

            # Build per-request logits_processors from repetition_penalty and
            # any caller-supplied extras (e.g. JSON schema constrained
            # decoding).
            rep_penalty = request.sampling_params.repetition_penalty
            extra_lp = request.sampling_params.logits_processors or []
            combined_lp: list = []
            if rep_penalty and rep_penalty != 1.0:
                combined_lp.extend(
                    make_logits_processors(repetition_penalty=rep_penalty)
                )
                logger.info(
                    f"[rep_penalty] request={request.request_id[:12]} "
                    f"penalty={rep_penalty}"
                )
            if extra_lp:
                combined_lp.extend(extra_lp)
                logger.info(
                    f"[logits_proc] request={request.request_id[:12]} "
                    f"extra_processors={len(extra_lp)}"
                )
            lp = combined_lp

            # Insert into BatchGenerator with optional cache.
            # Wrap in try/except: if cache shapes are incompatible
            # (e.g. stale entry after BatchGenerator recreation),
            # fall back to no-cache insert instead of crashing.
            insert_kwargs = {
                "max_tokens": [request.sampling_params.max_tokens],
                "caches": [cache_to_use] if cache_to_use else None,
                # Always pass logits_processors (even empty list) so that
                # mlx_lm BatchGenerator never stores None per-sequence.
                "logits_processors": [lp] if lp else [[]],
            }
            try:
                uids = self.batch_generator.insert(
                    [tokens_to_process],
                    **insert_kwargs,
                )
            except Exception as e:
                if cache_to_use is not None:
                    logger.warning(
                        f"[cache_insert_error] request={request.request_id[:12]} "
                        f"cache insert failed ({e}), retrying without cache"
                    )
                    cache_to_use = None
                    request.prompt_cache = None
                    request.cached_tokens = 0
                    request.remaining_tokens = request.prompt_token_ids
                    tokens_to_process = request.prompt_token_ids
                    insert_kwargs["caches"] = None
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        **insert_kwargs,
                    )
                else:
                    raise

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                # Release the prompt cache reference now that BatchGenerator
                # has its own copy.  Holding this reference prevents MLX from
                # freeing the Metal buffers until the request object is GC'd,
                # which under sustained traffic can accumulate hundreds of GB
                # of wired memory (issue #442).
                request.prompt_cache = None
                self.running[request.request_id] = request
                scheduled.append(request)

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = (
                    f", {request.cached_tokens} cached"
                    if request.cached_tokens > 0
                    else ""
                )
                tokens_to_prefill = len(tokens_to_process)
                rep_info = (
                    f" rep_penalty={rep_penalty}"
                    if rep_penalty and rep_penalty != 1.0
                    else ""
                )
                logger.info(
                    f"[schedule] request={request.request_id[:12]} uid={uid} "
                    f"prompt_tokens={request.num_prompt_tokens} "
                    f"tokens_to_prefill={tokens_to_prefill}{cache_info} "
                    f"max_tokens={request.sampling_params.max_tokens}{rep_info} "
                    f"running={len(self.running)} waiting={len(self.waiting)}"
                )

        return scheduled

    def _process_batch_responses(
        self, responses: List[Any]
    ) -> Tuple[List[RequestOutput], Set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Append token to request
            request.append_output_token(response.token)

            # Record first token time for TTFT metric
            if request.first_token_time is None and request.num_output_tokens > 0:
                import time as _time

                request.first_token_time = _time.time()

            # Decode the new token using streaming detokenizer (UTF-8 safe)
            if response.finish_reason == "stop":
                new_text = ""
            else:
                detok = self._get_detokenizer(request_id)
                detok.add_token(response.token)
                new_text = detok.last_segment

            # Create output
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=request.output_token_ids,
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
            )

            # Check if finished
            if response.finish_reason is not None:
                if response.finish_reason == "stop":
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif response.finish_reason == "length":
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

                output.finished = True
                output.finish_reason = response.finish_reason
                finished_ids.add(request_id)

                # Finalize streaming detokenizer and get full output
                detok = self._detokenizer_pool.get(request_id)
                if detok is not None:
                    detok.finalize()
                    output.output_text = detok.text
                else:
                    output.output_text = self._decode_tokens(request.output_token_ids)
                request.output_text = output.output_text
                self._cleanup_detokenizer(request_id)

                # Extract cache for future reuse (critical for agentic multi-turn)
                if hasattr(response, "prompt_cache"):
                    try:
                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache:
                            # For paged cache, extract actual tensor states
                            # This allows cache to survive BatchGenerator recreation
                            if self.block_aware_cache is not None:
                                extracted_cache = self._extract_cache_states(raw_cache)
                                if extracted_cache:
                                    request._extracted_cache = extracted_cache
                                    logger.debug(
                                        f"Extracted {len(extracted_cache)} layer states "
                                        f"for request {request_id}"
                                    )
                            else:
                                # Standard cache stores object references
                                request._extracted_cache = raw_cache
                    except Exception as e:
                        logger.debug(f"Failed to extract cache for {request_id}: {e}")

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Store cache for future reuse
            if request is not None and request.prompt_token_ids:
                if self.block_aware_cache is not None:
                    # Store in paged cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            self.block_aware_cache.store_cache(
                                request_id,
                                full_token_sequence,
                                request._extracted_cache,
                            )
                            logger.debug(
                                f"Stored paged cache for request {request_id} "
                                f"({len(full_token_sequence)} tokens: {len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output)"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to store paged cache for {request_id}: {e}"
                            )
                    # NOTE: Do NOT call release_cache here - blocks should persist
                    # for future requests to share. The LRU eviction will clean up
                    # unused blocks when under memory pressure.

                elif self.memory_aware_cache is not None:
                    # Keep mid-prefill entry as prefix cache for future
                    # requests that share a common prefix (e.g. same system
                    # prompt + tools but different user message).  LRU
                    # eviction handles memory pressure.

                    # Store in memory-aware prefix cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            import time as _time

                            _store_t0 = _time.monotonic()
                            stored = self.memory_aware_cache.store(
                                full_token_sequence,
                                request._extracted_cache,
                                evict_prefixes=False,
                            )
                            _store_dt = _time.monotonic() - _store_t0
                            # NOTE: We intentionally do NOT store a prompt-only
                            # cache entry.  Hybrid Mamba+Transformer models
                            # (like Qwen3-Coder-Next) have MambaCache layers
                            # whose state is cumulative and cannot be trimmed
                            # back to "prompt only".  Reusing such state causes
                            # the model to immediately produce EOS.
                            # The full prompt+output entry is stored above; a
                            # future request with the same prompt will hit the
                            # supersequence match path in the fetch, which is
                            # now disabled for safety (see memory_cache.py).

                            logger.info(
                                f"[cache_store] request={request_id[:12]} "
                                f"tokens={len(full_token_sequence)} "
                                f"({len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output) "
                                f"stored={stored} time={_store_dt:.3f}s "
                                f"cache_entries={len(self.memory_aware_cache._entries)} "
                                f"cache_mem={self.memory_aware_cache._current_memory / 1e6:.0f}MB"
                            )
                            # Release the original FP16 cache reference so
                            # memory can be reclaimed (the quantized copy
                            # lives inside the prefix cache now).
                            request._extracted_cache = None
                        except Exception as e:
                            logger.debug(
                                f"Failed to store memory-aware cache for {request_id}: {e}"
                            )

                elif self.prefix_cache is not None:
                    # Store in legacy prefix cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    # The next turn's prompt will include the previous response
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
                            self.prefix_cache.store_cache(
                                full_token_sequence,
                                request._extracted_cache,
                            )
                            logger.debug(
                                f"Stored cache for request {request_id} "
                                f"({len(full_token_sequence)} tokens: {len(request.prompt_token_ids)} prompt + {len(request.output_token_ids)} output)"
                            )
                        except Exception as e:
                            logger.debug(f"Failed to store cache for {request_id}: {e}")

            # Evaluate stored cache tensors incrementally (per-layer) to prevent
            # a deferred batch evaluation spike when all lazy ops resolve at once.
            # This spreads the VRAM cost across smaller per-layer evaluations.
            if (
                request is not None
                and hasattr(request, "_extracted_cache")
                and request._extracted_cache
            ):
                for layer in request._extracted_cache:
                    if isinstance(layer, dict) and "state" in layer:
                        keys, values = layer["state"]
                        mx.eval(keys, values)
                    elif hasattr(layer, "keys") and hasattr(layer, "values"):
                        keys_attr = layer.keys
                        values_attr = layer.values
                        if not callable(keys_attr) and not callable(values_attr):
                            mx.eval(keys_attr, values_attr)

            # Release all cache references on the request so Metal buffers
            # can be freed.  The prefix cache (if any) holds its own copy;
            # keeping a second reference here pins the buffers in wired memory
            # until the request object is GC'd (issue #442).
            if request is not None:
                request.prompt_cache = None
                request._extracted_cache = None

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

        # Free Metal command buffers after cleanup (prevents end-of-generation spike)
        if finished_ids:
            mx.clear_cache()

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        error_str = str(error)
        return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)

    def _is_stream_thread_error(self, error: Exception) -> bool:
        """Check if an error indicates MLX stream/thread ownership mismatch."""
        error_str = str(error)
        return "no Stream(" in error_str or "no Stream(gpu" in error_str

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Properly close batch generator (this is the source of the corruption)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        logger.info("Cache recovery completed")

    def _recover_from_generation_error(self) -> Set[str]:
        """Recover from fatal generation error (OOM, Metal crash).

        Aborts all running requests and resets batch state.
        Unlike cache corruption recovery, does NOT reschedule —
        the request that OOMed would just OOM again.

        Returns:
            Set of aborted request IDs.
        """
        # Close batch generator (clears _partial state, active_batch)
        self._close_batch_generator()
        self._current_sampler_params = None

        # Abort all running requests
        aborted_ids: Set[str] = set()
        for request_id in list(self.running):
            request = self.running.get(request_id)
            if request is not None:
                request.set_finished(RequestStatus.FINISHED_ABORTED)
            aborted_ids.add(request_id)
            self.finished_req_ids.add(request_id)
        self.running.clear()
        self._detokenizer_pool.clear()

        # Clear UID mappings (batch generator is gone)
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Release Metal memory
        mx.clear_cache()

        logger.warning(
            f"[generation_error_recovery] aborted {len(aborted_ids)} running requests, "
            f"batch generator closed, Metal cache cleared"
        )
        return aborted_ids

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # Reset request state
            request.status = RequestStatus.WAITING
            request.batch_uid = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def step(self, max_retries: int = 1) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. Automatically recovers from cache corruption errors

        Args:
            max_retries: Number of times to retry on cache errors (default 1)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Process pending aborts FIRST (in executor thread, safe for MLX)
        self._process_pending_aborts()

        for attempt in range(max_retries + 1):
            try:
                # Schedule waiting requests
                scheduled = self._schedule_waiting()
                output.scheduled_request_ids = [r.request_id for r in scheduled]
                output.num_scheduled_tokens = sum(
                    r.num_prompt_tokens for r in scheduled
                )

                # Run generation step if we have running requests
                if self.batch_generator is not None and self.running:
                    _sanitize_batch_generator_logits_processors(self.batch_generator)
                    result = self.batch_generator.next()
                    output.has_work = True

                    # mlx-lm >=0.31.x returns (prompt_responses, generation_responses);
                    # older versions returned a flat list.
                    if isinstance(result, tuple):
                        responses = result[1]  # generation_responses only
                    else:
                        responses = result

                    if responses:
                        outputs, finished_ids = self._process_batch_responses(responses)
                        output.outputs = outputs
                        output.finished_request_ids = finished_ids
                        self._cleanup_finished(finished_ids)

                # Success - break out of retry loop
                break

            except TypeError as e:
                # Catch the NoneType error specifically
                if self._is_cache_corruption_error(e):
                    if attempt < max_retries:
                        logger.warning(
                            f"Cache corruption detected (attempt {attempt + 1}), "
                            f"performing recovery and retry..."
                        )
                        # Deep reset to recover
                        self._recover_from_cache_error()
                        # Re-add any running requests back to waiting
                        self._reschedule_running_requests()
                    else:
                        logger.error(
                            f"Cache corruption not recoverable after "
                            f"{max_retries + 1} attempts"
                        )
                        raise
                else:
                    raise
            except Exception as e:
                if self._is_stream_thread_error(e):
                    raise
                import traceback

                logger.error(
                    f"Error in batch generation step: {e}\n{traceback.format_exc()}"
                )
                # Recover from fatal errors (OOM, Metal crash) instead of
                # re-raising, which would cause infinite loop in engine_core.
                aborted_ids = self._recover_from_generation_error()
                for rid in aborted_ids:
                    output.outputs.append(
                        RequestOutput(
                            request_id=rid,
                            finished=True,
                            finish_reason="error",
                        )
                    )
                output.finished_request_ids = aborted_ids
                break

        # Clear finished tracking for next step
        old_finished = self.finished_req_ids
        self.finished_req_ids = set()

        # Adaptive interval: scale inversely with concurrency to prevent
        # Metal resource handle exhaustion under high-concurrency workloads.
        active_seqs = len(self.running)
        min_interval = max(4, self._clear_cache_interval // 4)
        effective_interval = max(
            min_interval, self._clear_cache_interval // max(1, active_seqs // 8)
        )

        self._step_count += 1
        if self._step_count % effective_interval == 0:
            # Evaluate batch tokens to collapse lazy concatenation chains
            if (
                self.batch_generator is not None
                and hasattr(self.batch_generator, "active_batch")
                and self.batch_generator.active_batch is not None
                and hasattr(self.batch_generator.active_batch, "tokens")
            ):
                tokens = self.batch_generator.active_batch.tokens
                if tokens:
                    mx.eval(*tokens)
            mx.clear_cache()

        # Periodically log memory stats for monitoring
        if self._step_count % self._memory_log_interval == 0:
            try:
                if mx.metal.is_available():
                    active_gb = mx.get_active_memory() / 1e9
                    peak_gb = mx.get_peak_memory() / 1e9
                    cache_gb = mx.get_cache_memory() / 1e9
                    logger.info(
                        f"[Metal memory] active={active_gb:.1f}GB "
                        f"peak={peak_gb:.1f}GB cache={cache_gb:.1f}GB "
                        f"step={self._step_count} "
                        f"running={len(self.running)} waiting={len(self.waiting)}"
                    )
            except Exception:
                pass

        return output

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[Request]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_running_requests_info(self) -> List[Dict[str, Any]]:
        """Per-request details for status endpoint."""
        import time as _time

        now = _time.time()
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
                    "max_tokens": req.max_tokens,
                    "progress": 0.0,
                    "tokens_per_second": None,
                    "ttft_s": None,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        # Running requests
        for req in self.running.values():
            n_out = req.num_output_tokens
            elapsed = now - req.arrival_time

            # Phase detection
            if n_out == 0:
                phase = "prefill"
            else:
                phase = "generation"

            # Tokens per second (generation phase only)
            tok_s = None
            ttft = None
            if req.first_token_time is not None:
                ttft = round(req.first_token_time - req.arrival_time, 3)
                gen_elapsed = now - req.first_token_time
                if gen_elapsed > 0 and n_out > 0:
                    tok_s = round(n_out / gen_elapsed, 1)

            # Progress: completion_tokens / max_tokens
            progress = round(n_out / req.max_tokens, 3) if req.max_tokens > 0 else 0.0

            result.append(
                {
                    "request_id": req.request_id,
                    "status": "running",
                    "phase": phase,
                    "elapsed_s": round(elapsed, 2),
                    "prompt_tokens": req.num_prompt_tokens,
                    "completion_tokens": n_out,
                    "max_tokens": req.max_tokens,
                    "progress": min(progress, 1.0),
                    "tokens_per_second": tok_s,
                    "ttft_s": ttft,
                    "cache_hit_type": req.cache_hit_type,
                    "cached_tokens": req.cached_tokens,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        # Include Metal memory stats
        try:
            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        # Include cache stats
        if self.block_aware_cache is not None:
            stats["paged_cache"] = self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            stats["memory_aware_cache"] = self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            stats["prefix_cache"] = self.prefix_cache.get_stats()
        return stats

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.block_aware_cache is not None:
            return self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            return self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            return self.prefix_cache.get_stats()
        return None

    def clear_runtime_caches(self) -> Dict[str, bool]:
        """Clear prefix-cache state without resetting scheduler/request state."""
        cleared = {
            "paged_cache": False,
            "memory_aware_cache": False,
            "prefix_cache": False,
        }
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
            cleared["paged_cache"] = True
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
            cleared["memory_aware_cache"] = True
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
            cleared["prefix_cache"] = True
        return cleared

    def reset(self) -> None:
        """Reset the scheduler state."""
        # Drain any pending deferred aborts
        self._pending_abort_ids.clear()

        # Abort all requests directly (reset is synchronous)
        for request_id in list(self.requests.keys()):
            self._do_abort_request(request_id)

        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._detokenizer_pool.clear()
        self._close_batch_generator()
        self._current_sampler_params = None

        # Clear caches
        self.clear_runtime_caches()

        # Close SSD tier on reset
        self.close_ssd_tier()

    def deep_reset(self) -> None:
        """
        Deep reset that clears ALL cache state including model-level caches.

        This is more aggressive than reset() and should be used when
        switching engines or recovering from errors.
        """
        # Standard reset first
        self.reset()

        # Clear any model-level cache state
        # MLX models may have internal cache references
        if hasattr(self.model, "cache"):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "cache"):
                    layer.cache = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                    layer.self_attn.cache = None

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")

    # -----------------------------------------------------------------
    # Cache persistence
    # -----------------------------------------------------------------

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        """Save prefix cache to disk for persistence across restarts."""
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.save_to_disk(cache_dir)
        logger.info("[cache_persist] no memory-aware cache to save")
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.load_from_disk(cache_dir)
        logger.info("[cache_persist] no memory-aware cache to load into")
        return 0

    def clear_prefix_cache(self) -> None:
        """Clear the in-memory prefix cache (keeps disk cache untouched)."""
        if self.memory_aware_cache is not None and hasattr(
            self.memory_aware_cache, "clear"
        ):
            self.memory_aware_cache.clear()
            logger.info("[clear_prefix_cache] memory-aware cache cleared")
            return
        if self.prefix_cache is not None and hasattr(self.prefix_cache, "clear"):
            self.prefix_cache.clear()
            logger.info("[clear_prefix_cache] prefix cache cleared")

    def close_ssd_tier(self) -> None:
        """Shut down the SSD cache tier if present."""
        if self._ssd_tier is not None:
            self._ssd_tier.close()
            self._ssd_tier = None
            logger.info("SSD cache tier closed")

    def _try_promote_ssd_pending(self) -> None:
        """Attempt synchronous SSD promotion for waiting requests tagged ssd_pending.

        Called from _schedule_waiting() before requests are moved to running.
        Reads SSD entries synchronously (disk I/O stays out of fetch() per spec).
        """
        for request in self.waiting:
            if getattr(request, "cache_hit_type", None) != "ssd_pending":
                continue

            candidate = getattr(request, "_ssd_candidate", None)
            if candidate is None:
                continue

            memory_bytes = candidate["memory_bytes"]

            # Check RAM budget availability
            if self.memory_aware_cache is None:
                request.cache_hit_type = "miss"
                continue

            if not self.memory_aware_cache.try_reserve_memory(memory_bytes):
                self._ssd_tier._stats.promotion_failures += 1
                request.cache_hit_type = "miss"
                logger.info(
                    f"[ssd_promote] request={request.request_id[:12]} "
                    f"budget denied ({memory_bytes} bytes)"
                )
                continue

            # Use the SSD entry's actual token count for read and store,
            # NOT the full prompt tokens. For prefix hits these differ.
            matched_count = candidate["matched_tokens"]
            matched_tokens = tuple(request.prompt_token_ids[:matched_count])

            try:
                cache_layers = self._ssd_tier._read_entry(
                    matched_tokens, candidate["file_path"]
                )
            except Exception:
                self.memory_aware_cache.release_reserved_memory(memory_bytes)
                self._ssd_tier._stats.promotion_failures += 1
                request.cache_hit_type = "miss"
                logger.exception(
                    f"[ssd_promote] request={request.request_id[:12]} "
                    f"disk read failed"
                )
                continue

            if cache_layers is None:
                self.memory_aware_cache.release_reserved_memory(memory_bytes)
                self._ssd_tier._stats.promotion_failures += 1
                request.cache_hit_type = "miss"
                continue

            # Release tentative budget (store() will account properly)
            self.memory_aware_cache.release_reserved_memory(memory_bytes)

            # Reconstruct and store under the matched prefix tokens
            reconstructed = self._reconstruct_ssd_layers(cache_layers)
            if reconstructed is None:
                request.cache_hit_type = "miss"
                continue

            self.memory_aware_cache.store(
                list(matched_tokens), reconstructed, evict_prefixes=False
            )

            request.prompt_cache = reconstructed
            request.cached_tokens = matched_count
            request.remaining_tokens = request.prompt_token_ids[matched_count:]
            request.cache_hit_type = "ssd_hit"

            self._ssd_tier._stats.ssd_hits += 1
            self._ssd_tier._index.touch(matched_tokens)

            logger.info(
                f"[ssd_promote] request={request.request_id[:12]} "
                f"{candidate['match_type']} promote: {matched_count}/{len(request.prompt_token_ids)} tokens from SSD, "
                f"{len(request.remaining_tokens)} remaining"
            )

    async def promote_from_ssd(self, request) -> bool:
        """Promote a cold-tier cache entry for a request (async version).

        Alternative to _try_promote_ssd_pending() for callers with an
        async event loop. Uses asyncio.to_thread for non-blocking disk I/O.

        Returns True if promotion succeeded and request was updated.
        """
        if self._ssd_tier is None:
            return False

        candidate = getattr(request, "_ssd_candidate", None)
        if candidate is None:
            return False

        def reserve_budget(nbytes: int) -> bool:
            """Tentatively reserve RAM budget for promotion."""
            if self.memory_aware_cache is None:
                return False
            return self.memory_aware_cache.try_reserve_memory(nbytes)

        def release_budget(nbytes: int) -> None:
            """Release tentatively reserved budget on failure."""
            if self.memory_aware_cache is not None:
                self.memory_aware_cache.release_reserved_memory(nbytes)

        # Use matched token count, not full prompt, for prefix hits
        matched_count = candidate.get("matched_tokens", len(request.prompt_token_ids))
        matched_tokens = tuple(request.prompt_token_ids[:matched_count])

        cache_layers = await self._ssd_tier.async_promote(
            matched_tokens, reserve_budget, release_budget
        )

        if cache_layers is None:
            request.cache_hit_type = "miss"
            return False

        # Release tentative budget — store() will account properly
        release_budget(candidate["memory_bytes"])

        # Reconstruct cache objects from deserialized layer dicts
        reconstructed = self._reconstruct_ssd_layers(cache_layers)
        if reconstructed is None:
            request.cache_hit_type = "miss"
            return False

        # Store in RAM cache under the matched prefix tokens
        self.memory_aware_cache.store(
            list(matched_tokens), reconstructed, evict_prefixes=False
        )

        request.prompt_cache = reconstructed
        request.cached_tokens = matched_count
        request.remaining_tokens = request.prompt_token_ids[matched_count:]
        request.cache_hit_type = "ssd_hit"

        logger.info(
            f"[ssd_promote] request={request.request_id[:12]} "
            f"{candidate.get('match_type', 'exact')} promote: "
            f"{matched_count}/{len(request.prompt_token_ids)} tokens from SSD, "
            f"{len(request.remaining_tokens)} remaining"
        )
        return True

    def _reconstruct_ssd_layers(self, layer_dicts: list[dict]) -> list | None:
        """Reconstruct cache objects from deserialized layer dicts.

        Converts numpy arrays back to MLX arrays and creates KVCache objects.
        """
        try:
            from mlx_lm.models.cache import ArraysCache, KVCache

            result = []
            for ld in layer_dicts:
                if "keys" in ld and "values" in ld:
                    kv = KVCache()
                    kv.keys = mx.array(ld["keys"])
                    kv.values = mx.array(ld["values"])
                    kv.offset = ld["offset"]
                    for attr in ("max_size", "keep", "step", "_idx"):
                        if attr in ld:
                            setattr(kv, attr, ld[attr])
                    result.append(kv)
                elif "state" in ld:
                    state_arrays = [mx.array(a) for a in ld["state"]]
                    layer_obj = ArraysCache(len(state_arrays))
                    layer_obj.state = state_arrays
                    result.append(layer_obj)
                else:
                    logger.warning(
                        f"[ssd_promote] unknown layer dict format: {list(ld.keys())}"
                    )
                    return None
            return result
        except Exception as e:
            logger.warning(f"[ssd_promote] reconstruction failed: {e}")
            return None
