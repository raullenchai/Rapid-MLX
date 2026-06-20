# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for rapid-mlx continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import logging
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mlx.core as mx

# MUST install the MLX hardware-compat shim BEFORE importing mlx_lm.generate.
# mlx_lm/generate.py captures `mx.new_thread_local_stream(mx.default_device())`
# at module-import time; on M5 single-stream GPUs that stream is unusable
# (#404). The shim is idempotent and a no-op on hardware where the original
# API works.
from . import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.generate import BatchGenerator  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: E402
from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer  # noqa: E402

from ._sampler_fast_path import (  # noqa: E402
    is_fused_top_p_eligible,
    make_fused_top_p_temp_sampler,
)
from .memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig  # noqa: E402
from .paged_cache import PagedCacheManager
from .pflash import PFlashConfig, compress_request_tokens
from .prefix_cache import BlockAwarePrefixCache, PrefixCacheManager
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .utils.decode import IncrementalDecoder
from .utils.mamba_cache import ensure_mamba_support

logger = logging.getLogger(__name__)


def _pflash_compressed(request: Request) -> bool:
    """Whether PFlash replaced this request's prompt with a compressed
    subsequence. Used to gate every prefix-cache store/fetch site so the
    compressed token sequence — which is positionally non-faithful to
    the original prompt — never enters the shared trie.
    """
    return bool(
        request.pflash_metadata is not None
        and request.pflash_metadata.get("compressed", False)
    )


# Enable MambaCache batching support for models like Nemotron
ensure_mamba_support()

# Error patterns that indicate cache corruption.
# Each pattern must be specific enough to avoid false positives.
# The bare word "cache" was removed because it matched unrelated TypeErrors
# (e.g. "unsupported operand type for cache_size"), masking real bugs and
# triggering unnecessary cache wipes + request reschedules.
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "BatchKVCache",
    "KVCache",
    "cache is not subscriptable",
    "cache has no attribute",
]


def _assemble_stop_tokens(
    sampling_params: SamplingParams, model_stop_tokens: set[int]
) -> set[int]:
    """Build the stop-token set the BatchGenerator should respect for one request.

    Contract (locked by ``tests/test_community_bench.py::test_scheduler_honours_ignore_eos``):

    - ``sampling_params.ignore_eos=True`` → suppress every token in
      ``model_stop_tokens`` (the model's own EOS + chat-template terminators).
      Matches llama.cpp ``llama-bench --no-eos`` and vLLM upstream semantics.
      Used by community-bench's ``tg128`` / ``tg512`` rounds where the
      contract is "decode exactly N tokens", not "decode until the model
      decides to stop".
    - ``sampling_params.stop_token_ids`` is **always** unioned in.
      Those are *caller intent*, not model intent, so they survive
      ``ignore_eos=True``.
    - ``sampling_params.ignore_eos=False`` (default) → return the union
      of model stop tokens and any caller stop ids. Normal serve / chat
      behaviour.

    Why this is a free function: extracted from ``_create_batch_generator``
    so the test exercises the production assembly directly. A local
    stand-in in the test could pass even if this function deleted the
    ``ignore_eos`` branch.
    """
    stop_tokens: set[int] = (
        set() if sampling_params.ignore_eos else set(model_stop_tokens)
    )
    if sampling_params.stop_token_ids:
        stop_tokens.update(sampling_params.stop_token_ids)
    return stop_tokens


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

    # Prefix cache settings
    enable_prefix_cache: bool = True
    prefix_cache_size: int = 100  # Max cached entries (legacy, ignored if memory-aware)

    # Memory-aware cache settings (recommended for large models)
    use_memory_aware_cache: bool = True  # Use memory-based eviction
    cache_memory_mb: int | None = None  # None = auto-detect (20% of available RAM)
    cache_memory_percent: float = 0.20  # Fraction of available RAM if auto-detecting

    # KV cache quantization (reduces prefix cache memory)
    kv_cache_quantization: bool = False
    kv_cache_quantization_bits: int = 8
    kv_cache_quantization_group_size: int = 64
    kv_cache_min_quantize_tokens: int = 256

    # TurboQuant V-only compression (asymmetric: K=FP16, V=3-4bit rotated Lloyd-Max)
    kv_cache_turboquant: bool = False
    kv_cache_turboquant_bits: int | None = None  # None = auto-select by head_dim
    kv_cache_turboquant_group_size: int = 32

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

    # MTP (Multi-Token Prediction) settings
    # Uses the model's built-in MTP head to predict multiple tokens per step
    enable_mtp: bool = False
    mtp_num_draft_tokens: int = 1  # Number of draft tokens from MTP head
    mtp_optimistic: bool = False  # Skip acceptance check for max speed

    # SuffixDecoding — drafter-free speculative decoding using a suffix
    # tree over prompt + generated tokens. Predicts repeated patterns
    # (tool boilerplate, JSON schemas, ReAct loops) at zero drafter
    # cost. Pure-attention only; the architecture allowlist is enforced
    # via ``ModelConfig.supports_spec_decode`` at install time.
    enable_suffix_decoding: bool = False
    suffix_max_draft: int = 8  # Max draft tokens per step (verify cost ∝ this)
    suffix_max_suffix_len: int = 4  # Longest k-gram indexed for matching
    suffix_min_confidence: float = 0.3  # Vote confidence floor before truncating
    # Skip the verify forward when the drafter returned fewer than this
    # many tokens. Single-token drafts are common on free-form chat where
    # the drafter sees a weak match — verify cost dominates the small
    # win. Default 2 keeps chat near regression-floor while still
    # accepting most useful drafts on tool/JSON workloads.
    suffix_min_draft_len: int = 2

    # Admission control: hard cap on concurrent in-flight requests
    # (queued + running). A buggy client (or simple fork bomb) used to
    # be able to OOM the Metal allocator and crash the server for all
    # other clients; ``add_request`` now raises ``BackpressureError``
    # at the cap and routes return 503 with Retry-After. Default 256
    # provides ample queue depth on top of ``max_num_seqs`` — waiting
    # requests only carry their tokenised prompt, not KV cache state,
    # so the memory cost of a queue is small even when ``max_num_seqs``
    # is constrained. Operators who want admission to mirror
    # ``max_num_seqs`` exactly can pass ``--max-concurrent-requests``
    # (codex R7 flagged the gap; the explicit override resolves it
    # without breaking existing tests that intentionally send more
    # requests than ``max_num_seqs`` to exercise the queue).
    max_concurrent_requests: int = 256

    # PFlash long-prompt prefill compression (#287). Disabled by default;
    # see vllm_mlx/pflash.py for the design notes and the prefix-cache
    # bypass on compressed requests.
    pflash_config: PFlashConfig = field(default_factory=PFlashConfig)

    def __post_init__(self) -> None:
        # PFlashConfig is dataclass(frozen=True), so .validate() returns
        # a new instance; reassign so the SchedulerConfig holds the
        # validated copy. Done in __post_init__ to keep callers from
        # threading .validate() through every construction site.
        self.pflash_config = self.pflash_config.validate()


class BackpressureError(Exception):
    """Raised when admission control rejects a new request.

    Caught by route handlers and converted to HTTP 503 with a
    Retry-After header so well-behaved clients back off and retry.
    Distinguished from ``ValueError`` so the scheduler's narrow
    batch-error catch path doesn't swallow it.
    """


@dataclass
class SchedulerOutput:
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


def _install_chunked_prefill(
    batch_gen: "BatchGenerator",
    budget: int,
    mid_prefill_save=None,
    prompt_cache_save=None,
    pending_abort_ids: set[str] | None = None,
    uid_to_request_id: dict[int, str] | None = None,
    requests: dict[str, Any] | None = None,
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

    # mlx-lm 0.31+ renamed Batch → GenerationBatch with different constructor
    try:
        from mlx_lm.generate import Batch as _Batch

        _USE_NEW_BATCH = False
    except ImportError:
        from mlx_lm.generate import GenerationBatch as _Batch

        _USE_NEW_BATCH = True

    def _make_batch(
        model,
        uids,
        y,
        logprobs,
        max_tokens,
        num_tokens,
        prompt_cache,
        samplers,
        logits_processors,
        tokens,
    ):
        if _USE_NEW_BATCH:
            # GenerationBatch(model, uids, inputs, prompt_cache, tokens,
            #   samplers, fallback_sampler, logits_processors, state_machines, max_tokens)
            return _Batch(
                model=model,
                uids=uids,
                inputs=y,
                prompt_cache=prompt_cache,
                tokens=tokens,
                samplers=samplers,
                fallback_sampler=lambda x: x.argmax(-1),
                logits_processors=logits_processors,
                state_machines=[],
                max_tokens=max_tokens,
            )
        else:
            return _Batch(
                uids,
                y,
                logprobs,
                max_tokens,
                num_tokens,
                prompt_cache,
                samplers,
                logits_processors,
                tokens,
            )

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

            n_to_process = min(budget, remaining - 1) if remaining > 1 else 0

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

            # Check if prefill is done (only 1 token left or 0)
            if inputs.shape[1] <= 1:
                # Finalize
                if partial.get("is_cached"):
                    mx.eval([c.state for c in prompt_cache])
                    inputs = partial["last_inputs"]

                for c in prompt_cache:
                    c.finalize()
                mx.clear_cache()

                y, logprobs = self._step(
                    inputs,
                    prompt_cache,
                    partial["samplers"],
                    partial["logits_processors"],
                    partial["tokens"],
                )
                mx.async_eval(y, logprobs)

                new_batch = _make_batch(
                    model=batch_gen.model,
                    uids=list(partial["uids"]),
                    y=y,
                    logprobs=logprobs,
                    max_tokens=list(partial["max_tokens"]),
                    num_tokens=[0] * len(partial["uids"]),
                    prompt_cache=prompt_cache,
                    samplers=list(partial["samplers"]),
                    logits_processors=list(partial["logits_processors"]),
                    tokens=partial["tokens"],
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
                        _prompt_checkpoints,
                    ) = zip(*batch_prompts)
                    lengths = [len(p) for p in inputs_raw]
                    max_length = max(lengths)
                    padding = [max_length - ln for ln in lengths]
                    tokens = [mx.array(inp) for inp in inputs_raw]
                    is_cached = not all(c[0].empty() for c in caches)

                    self._stats.prompt_tokens += sum(lengths)

                    if not is_cached:
                        padded = _left_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _make_cache(
                            self.model, padding, self.max_kv_size
                        )
                    else:
                        last_inputs = mx.array([p[-1:] for p in inputs_raw])
                        padded = _right_pad_prompts(inputs_raw, max_length=max_length)
                        prompt_cache = _merge_caches(caches)
                        for c in prompt_cache:
                            c.prepare(
                                lengths=[ln - 1 for ln in lengths],
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
                        if 0 < _adjusted_pb < padded.shape[1]:
                            _first_chunk = _adjusted_pb
                    n_to_process = min(_first_chunk, padded.shape[1] - 1)
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


def _install_dense_sampler_fastpath(batch_gen: "BatchGenerator") -> None:
    """Swap to mlx-lm's batched sampler fast path when the running batch
    is homogeneous in sampling params.

    mlx-lm's ``GenerationBatch._step`` (``mlx_lm/generate.py:1320``) takes
    a per-row Python loop + ``mx.concatenate`` whenever
    ``any(self.samplers)`` is True. The Scheduler attaches a per-request
    sampler on every ``insert(...)``, so that branch is taken for every
    multi-request batch — bypassing the fast ``fallback_sampler(logprobs)``
    path that runs sampling once on ``[B, vocab]``.

    When every entry in ``self.samplers`` is the same callable instance,
    sampling is mathematically identical to invoking that one callable on
    the full ``[B, vocab]`` matrix (mlx-lm's ``apply_top_p`` /
    ``apply_min_p`` / ``apply_top_k`` / ``categorical_sampling`` all
    operate row-wise along ``axis=-1``). The Scheduler interns samplers
    via ``_get_request_sampler``, so identity-equality of the entries in
    ``self.samplers`` already implies value-equality of the sampling
    params — no separate key check needed.

    Heterogeneous batches (mixed temp/top_p across requests) fall back to
    mlx-lm's original per-row loop — correctness preserved.

    Companion to ``MLLMBatchGenerator._step`` fast path in
    ``mllm_batch_generator.py`` (PR #519). This installs the same shape
    on the dense LLM path that lives inside mlx-lm.
    """
    import types

    gen_batch = getattr(batch_gen, "_generation_batch", None)
    if gen_batch is None or not hasattr(gen_batch, "_step"):
        return

    # ``gen_batch._step`` may already be a bound method (vanilla mlx-lm)
    # OR a plain closure replaced by ``_install_suffix_decoding`` (which
    # writes ``gb._step = _suffix_step`` — see the assignment in that
    # function). Both shapes accept zero args (the closure closes over
    # ``gb``; the bound method already carries ``self``), so calling
    # ``orig_step()`` without args works for either.
    orig_step = gen_batch._step

    def patched_step(self):
        samplers = self.samplers
        if samplers and len(samplers) >= 2:
            first = samplers[0]
            if first is not None and all(s is first for s in samplers[1:]):
                saved_samplers = self.samplers
                saved_fallback = self.fallback_sampler
                self.samplers = [None] * len(samplers)
                self.fallback_sampler = first
                try:
                    return orig_step()
                finally:
                    self.samplers = saved_samplers
                    self.fallback_sampler = saved_fallback
        return orig_step()

    gen_batch._step = types.MethodType(patched_step, gen_batch)
    logger.info("[dense_sampler_fastpath] installed on BatchGenerator")


def _install_mtp(
    batch_gen: "BatchGenerator",
    model: Any,
    num_draft_tokens: int = 1,
    optimistic: bool = False,
) -> bool:
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

    Returns True when patches are installed, False when the BatchGenerator
    is incompatible (e.g. hybrid Gated-DeltaNet generators that route through
    their own step flow and lack ``_step`` / ``_orig_next``). In the
    incompatible case the generator is left untouched so the request still
    completes — MTP is just silently dropped after a clear warning. See #477.
    """
    if not hasattr(batch_gen, "_step"):
        logger.warning(
            "[MTP] BatchGenerator %s has no _step attribute — this model "
            "uses a different generation flow (likely hybrid Gated-DeltaNet, "
            "e.g. Qwen3.6-35B-A3B). MTP install skipped; the request "
            "continues normally without MTP. Other spec-decode paths "
            "(suffix / DFlash) installed via --force-spec-decode are "
            "unaffected. See issue #477.",
            type(batch_gen).__name__,
        )
        return False
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
        if any(logits_processors):
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
            _rnn_snapshots = {}
            for _ci, _c in enumerate(prompt_cache):
                if not (hasattr(_c, "is_trimmable") and _c.is_trimmable()):
                    if hasattr(_c, "state"):
                        _orig_state = _c.state
                        _copied = [
                            s.copy() if s is not None else None for s in _orig_state
                        ]
                        # Preserve original type (tuple vs list) so downstream
                        # code that does `h, c = cache.state` doesn't break.
                        if isinstance(_orig_state, tuple):
                            _copied = tuple(_copied)
                        _rnn_snapshots[_ci] = _copied

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
            # Clear deferred drafts from this failed attempt to prevent
            # stale tokens from being emitted in the next _mtp_next call.
            for uid in current_uids:
                _deferred_drafts.pop(uid, None)
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

    mode_str = "optimistic (no verify)" if optimistic else "always-advance"
    logger.info(
        f"[MTP] installed with num_draft_tokens={num_draft_tokens}, {mode_str} mode"
    )
    return True


def _install_suffix_decoding(
    batch_gen: "BatchGenerator",
    model: Any,
    profile: Any | None,
    max_draft: int,
    max_suffix_len: int,
    min_confidence: float,
    requests: dict[str, Any],
    uid_to_request_id: dict[int, str],
    min_draft_len: int = 2,
) -> None:
    """Monkey-patch BatchGenerator's GenerationBatch to add SuffixDecoding.

    Drafter-free spec-decode: a suffix-tree index over prompt + emitted
    tokens predicts repeated patterns (tool calls, JSON, code edits,
    ReAct loops) at zero drafter cost. Big wins on agent workloads
    (3-5×); ~1× on free-form chat (regression-floor).

    The hot path lives in ``GenerationBatch._step`` (mlx-lm 0.31+):

      1. Drafter builds up to ``max_draft`` candidate tokens.
      2. We run ``model([X, d_0..d_{K-1}])`` of shape (1, K+1).
      3. Greedy compare argmax(logits[i]) vs draft[i]; accept up to
         first mismatch. ``n_accepted ∈ [0, K]``.
      4. Trim trimmable cache layers by ``K - n_accepted``.
      5. Emit ``n_accepted + 1`` new tokens: ``[d_0..d_{n-1}, bonus]``
         where ``bonus = preds[n_accepted]``.

    Wrapped ``GenerationBatch.next()`` augments the single Response
    that ``_step`` returns with ``n_accepted`` extra synthetic Responses
    so the engine sees the full token burst.

    Falls through to ``_orig_step`` when:
      - batch size != 1 (multi-request not handled in v1),
      - sampler is non-greedy (temperature > 0 / top_p < 1 / top_k > 0),
      - logits processors are configured (would need per-position apply),
      - drafter returns empty (low repetition).

    The architecture allowlist is enforced upstream via
    ``ModelConfig.supports_spec_decode``: hybrid linear-attention models
    (Qwen3.5/3.6 GatedDeltaNet, Granite 4 Mamba2) skip install entirely
    because chunked-batched verify isn't numerically equivalent to
    step-update on recurrent layers — see SUFFIX_POC_REPORT.md.
    """
    from .speculative.suffix_decoding import SuffixDecodingDrafter

    if profile is not None and not profile.supports_spec_decode:
        logger.warning(
            "[SuffixDecoding] disabled: model is hybrid (linear-attention/"
            "Mamba). Multi-token verify path is not numerically equivalent "
            "to step-update on recurrent layers. See "
            "evals/results/SUFFIX_POC_REPORT.md."
        )
        return

    # mlx-lm 0.31+ moved the actual generation step from BatchGenerator
    # to GenerationBatch. The _generation_batch instance is created once
    # in BatchGenerator.__init__ and is mutated (extend/filter) in place
    # — so a single instance-level patch persists across all sequences.
    gb = getattr(batch_gen, "_generation_batch", None)
    if gb is None:
        logger.warning(
            "[SuffixDecoding] disabled: BatchGenerator has no _generation_batch "
            "attribute (mlx-lm version mismatch — expected ≥0.31)."
        )
        return

    _orig_step = gb._step
    _orig_next = gb.next

    # Per-uid drafter state. Lazy-init on first encounter (we need the
    # request's prompt_token_ids to seed the suffix index).
    _drafters: dict[int, SuffixDecodingDrafter] = {}
    # When _step does a verify forward, it stashes the extra emitted
    # tokens here (one entry per accepted draft + bonus). The wrapped
    # ``next()`` then drains the queue, producing one synthetic Response
    # per token so the engine surface stays consistent.
    _pending_emits: dict[int, list[tuple[int, mx.array]]] = {}

    _stats = {
        "verify_steps": 0,
        "fallthrough_steps": 0,
        # Total draft TOKENS proposed across all verify steps (i.e., the
        # sum of K over verify_steps), not the count of verify proposals.
        # Mirrors ``DraftStats.total_draft_tokens_proposed`` naming.
        "draft_tokens_proposed": 0,
        "tokens_accepted": 0,
        "errors": 0,
        # Diagnostic breakdown of WHY we fell through. Sum should equal
        # ``fallthrough_steps``. Useful when debugging "no drafts, no
        # speedup" reports — points at the specific guard.
        "ft_batch_size": 0,
        "ft_uids_size": 0,
        "ft_non_greedy": 0,
        "ft_logits_processors": 0,
        "ft_no_draft": 0,
        "ft_cooldown": 0,
        "ft_non_trimmable_cache": 0,
    }

    # Cooldown state: when verify keeps producing 0-acceptance (e.g.,
    # free-form chat where drafter has weak signal), each verify pays
    # K-token forward overhead for ~zero gain. Detect three consecutive
    # zero-accept verifies and skip drafting for the next 10 steps;
    # after that try once. Tool/JSON workloads keep accepting → never
    # triggered. Chat hits ~90% skip → near regression-floor.
    _consecutive_zero_accepts = [0]
    _cooldown_remaining = [0]
    _COOLDOWN_TRIGGER = 3
    _COOLDOWN_LENGTH = 10

    def _is_greedy_for_uid(uid: int) -> bool:
        """Detect whether the request's sampler is effectively greedy.

        With ``temperature == 0`` mlx-lm short-circuits to argmax, so
        top_p / top_k are no-ops in that regime — we only check the
        temperature. (Defaults of top_p=0.9 / top_k=0 are common and
        don't actually change the sampler when temp=0.)

        Greedy verify only matches the user-requested distribution
        when the actual sampler is greedy; otherwise we fall through to
        keep token-stream stochasticity intact.
        """
        req_id = uid_to_request_id.get(uid)
        req = requests.get(req_id) if req_id else None
        if req is None or req.sampling_params is None:
            return True
        sp = req.sampling_params
        if sp.temperature is None or sp.temperature == 0.0:
            return True
        return False

    def _suffix_step():
        """Wrapped GenerationBatch._step.

        Original signature: ``() -> (List[int], List[mx.array])``.
        We preserve that contract — return the **single** primary token
        (= the input that was just fed through the model) plus its
        logprobs. Additional emitted tokens (accepted drafts + bonus)
        are stashed in ``_pending_emits`` for ``_suffix_next`` to drain.
        """
        # Single-request guard. _next_tokens has shape (B,).
        if gb._next_tokens is None or gb._next_tokens.shape[0] != 1:
            _stats["fallthrough_steps"] += 1
            _stats["ft_batch_size"] += 1
            return _orig_step()

        if len(gb.uids) != 1:
            _stats["fallthrough_steps"] += 1
            _stats["ft_uids_size"] += 1
            return _orig_step()

        uid = gb.uids[0]
        if not _is_greedy_for_uid(uid):
            _stats["fallthrough_steps"] += 1
            _stats["ft_non_greedy"] += 1
            return _orig_step()

        # Skip when logits_processors are set — applying them at every
        # speculative position would change the math in a way the
        # standalone PoC didn't validate. Defer to a follow-up.
        # Defensive ``getattr``: GenerationBatch grew this attribute in
        # mlx-lm 0.31; older builds would AttributeError here and silently
        # disable the entire suffix-decoding install.
        _lp = getattr(gb, "logits_processors", None)
        if _lp and any(p for p in _lp if p):
            _stats["fallthrough_steps"] += 1
            _stats["ft_logits_processors"] += 1
            return _orig_step()

        # Lazy-init drafter on first encounter for this uid.
        drafter = _drafters.get(uid)
        if drafter is None:
            req_id = uid_to_request_id.get(uid)
            req = requests.get(req_id) if req_id else None
            prompt_ids = (
                list(req.prompt_token_ids)
                if req is not None and req.prompt_token_ids
                else []
            )
            drafter = SuffixDecodingDrafter(
                max_draft_tokens=max_draft,
                max_suffix_len=max_suffix_len,
                min_confidence=min_confidence,
            )
            drafter.add_prompt_tokens(prompt_ids)
            # Catch up any tokens already in gb.tokens[0] (rare path —
            # only if suffix decoding were enabled mid-stream).
            try:
                for t in gb.tokens[0]:
                    drafter.add_generated_token(int(t))
            except Exception:  # noqa: BLE001
                pass
            _drafters[uid] = drafter

        # The token we're about to feed (= last step's sampled token).
        # Also the one ``_orig_step`` would return as ``inputs.tolist()``.
        inputs = gb._next_tokens
        last_token = int(inputs[0].item())
        drafter.add_generated_token(last_token)

        # Build draft.
        try:
            draft = drafter.get_draft()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[SuffixDecoding] drafter error: {e!r}")
            _stats["errors"] += 1
            _stats["fallthrough_steps"] += 1
            return _orig_step()

        if not draft or len(draft) < min_draft_len:
            # No (or too-short) repetition signal — vanilla step.
            # Short drafts on free-form text would pay verify-forward
            # overhead for almost no acceptance gain (chat regression-
            # floor). Skip them.
            _stats["fallthrough_steps"] += 1
            _stats["ft_no_draft"] += 1
            return _orig_step()

        # Cooldown check: skip verify if we're in a cooldown window
        # following several zero-accept verifies. This stops chat
        # workloads from paying verify overhead they can't recoup.
        if _cooldown_remaining[0] > 0:
            _cooldown_remaining[0] -= 1
            _stats["fallthrough_steps"] += 1
            _stats["ft_cooldown"] += 1
            return _orig_step()

        # Defense-in-depth: even though ``profile.supports_spec_decode``
        # already gates installation on hybrid arches, verify that EVERY
        # cache layer is trimmable before paying the verify-forward cost.
        # If any layer can't trim and we end up needing to roll back, the
        # cache state would silently diverge — better to fall through.
        for c in gb.prompt_cache:
            if not (
                hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "trim")
            ):
                _stats["fallthrough_steps"] += 1
                _stats["ft_non_trimmable_cache"] += 1
                return _orig_step()

        K = len(draft)
        _stats["verify_steps"] += 1
        _stats["draft_tokens_proposed"] += K

        # Verify forward: [last_token, d_0..d_{K-1}] of shape (1, K+1).
        try:
            draft_arr = mx.array([draft], dtype=inputs.dtype)
            verify_input = mx.concatenate([inputs[:, None], draft_arr], axis=1)
            verify_logits = gb.model(verify_input, cache=gb.prompt_cache)
            # logits shape (1, K+1, V); greedy verify.
            preds = mx.argmax(verify_logits, axis=-1)
            mx.eval(preds)
            preds_list = preds.tolist()[0]
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[SuffixDecoding] verify forward failed: {e!r}")
            _stats["errors"] += 1
            # Cache was not advanced because the forward raised; safe to
            # retry via vanilla path below.
            return _orig_step()

        # Accept up to first mismatch (greedy).
        n_accepted = 0
        for i in range(K):
            if preds_list[i] == draft[i]:
                n_accepted += 1
            else:
                break

        # Cooldown bookkeeping: track consecutive zero-accept verifies
        # so workloads with weak drafter signal (e.g., free-form chat)
        # automatically stop paying verify overhead.
        if n_accepted == 0:
            _consecutive_zero_accepts[0] += 1
            if _consecutive_zero_accepts[0] >= _COOLDOWN_TRIGGER:
                _cooldown_remaining[0] = _COOLDOWN_LENGTH
                _consecutive_zero_accepts[0] = 0
        else:
            _consecutive_zero_accepts[0] = 0

        n_rejected = K - n_accepted
        if n_rejected > 0:
            # Pre-checked above — every layer here is trimmable.
            for c in gb.prompt_cache:
                c.trim(n_rejected)

        # Token emission accounting.
        #
        # _orig_step emits one token per call: the ``inputs`` it just
        # fed through the model (= what was previously in
        # ``_next_tokens``). The newly-sampled token is stashed in
        # ``_next_tokens`` for the next step.
        #
        # For spec-decode the verify forward consumed K+1 tokens
        # (last_token + K drafts), so we have committed to the cache
        # ``[..., last_token, d_0..d_{n_accepted-1}]`` after trim.
        # Tokens that NEED to surface on the response stream:
        #
        #   - last_token   ← primary, returned by this _step (1 token)
        #   - d_0..d_{n-1} ← accepted drafts (n tokens, drained by
        #                    _suffix_next as synthetic responses)
        #
        # The bonus (= preds[n_accepted], the correction at the
        # rejection point or the post-K bonus) is **NOT** emitted this
        # step — it gets stashed in _next_tokens and surfaces as the
        # primary of the NEXT _step call. Otherwise it would duplicate
        # (see early bug: every-other-token doubling).
        bonus = preds_list[n_accepted]

        full_logprobs = verify_logits - mx.logsumexp(
            verify_logits, axis=-1, keepdims=True
        )
        # The primary's logprobs come from the PREVIOUS step (saved in
        # gb._next_logprobs). Passing them through preserves the same
        # contract as _orig_step.
        primary_logprobs = (
            gb._next_logprobs[0]
            if gb._next_logprobs is not None and len(gb._next_logprobs) > 0
            else full_logprobs[0, 0, :]
        )
        extra_tokens = list(draft[:n_accepted])
        extra_logprobs: list[mx.array] = []
        for i in range(n_accepted):
            # full_logprobs[0, i, :] is the logprobs row that PRODUCED
            # the token at sequence position N+i+1, i.e. d_i.
            extra_logprobs.append(full_logprobs[0, i, :])
        # logprobs row at position n_accepted is the one that produced
        # the bonus — used for the bonus surfacing in the next step.
        bonus_logprobs = full_logprobs[0, n_accepted, :]

        # Drafter history += newly-committed tokens. We add ONLY the
        # accepted drafts here; ``bonus`` will be added on the next
        # ``_suffix_step`` call (line ~1235 ``drafter.add_generated_token
        # (last_token)`` where ``last_token = bonus`` since we just
        # stashed it in ``_next_tokens``). Adding it here too would
        # double-index it in the suffix tree and skew future drafts.
        for tok in extra_tokens:
            drafter.add_generated_token(tok)
        drafter.record_acceptance(n_accepted)
        _stats["tokens_accepted"] += n_accepted

        # Update gb state for the next _step call. Bonus becomes the
        # next step's primary input. async_eval overlaps device work
        # with engine bookkeeping (matches _orig_step's pattern).
        bonus_arr = mx.array([bonus], dtype=inputs.dtype)
        gb._next_tokens = bonus_arr
        gb._next_logprobs = [bonus_logprobs]
        mx.async_eval(bonus_arr, bonus_logprobs)

        # _step normally appends inputs.tolist()[i] to gb.tokens[i].
        # We do the same for last_token (the primary that we return).
        # The extra tokens get appended in the next() wrapper as each
        # synthetic Response is built, mirroring _orig_step's flow.
        gb.tokens[0].append(last_token)

        # Stash extras for next() to drain.
        _pending_emits[uid] = list(zip(extra_tokens, extra_logprobs))

        return [last_token], [primary_logprobs]

    def _suffix_next():
        """Wrapped GenerationBatch.next.

        Calls ``_orig_next`` (which calls our wrapped ``_step``) for the
        primary Response, then for each pending extra token builds a
        synthetic Response, handling stop-token / max-tokens like the
        original ``next()`` does.
        """
        responses = _orig_next()

        # Drop drafters for finished uids unconditionally — each drafter
        # holds up to ``max_history`` indexed tokens, so a leak here adds
        # up over a long-running server even on workloads that never hit
        # the synthetic-emit path. Run this before the early-return so
        # plain (non-spec-decode) finishes are also reaped.
        if responses:
            for r in responses:
                if r.finish_reason is not None:
                    _pending_emits.pop(r.uid, None)
                    _drafters.pop(r.uid, None)

        if not _pending_emits or not responses:
            return responses

        augmented = list(responses)
        for r in responses:
            uid = r.uid
            if r.finish_reason is not None:
                # Already reaped above — just skip.
                continue

            pending = _pending_emits.pop(uid, None)
            if not pending:
                continue

            # Find this uid's row in gb (post _orig_next, gb may have
            # been filtered if the primary finished — but we already
            # filtered out finished primaries above).
            try:
                row = gb.uids.index(uid)
            except ValueError:
                # Sequence already gone (filtered by _orig_next somehow);
                # bail out for this uid.
                continue

            for emit_idx, (tok, lp) in enumerate(pending):
                # Append to gb.tokens[row] for the synthetic emit; matches
                # the bookkeeping our wrapped _step already does for the
                # primary token (mlx-lm's original _step does NOT append).
                gb.tokens[row].append(tok)
                gb._num_tokens[row] += 1

                # Run the stop-machine on this token to detect stop seqs.
                finish_reason = None
                match_sequence = None
                current_state = None
                try:
                    new_state, match_sequence, current_state = gb.state_machines[
                        row
                    ].match(gb._matcher_states[row], tok)
                    gb._matcher_states[row] = new_state
                    if match_sequence is not None and current_state is None:
                        finish_reason = "stop"
                except Exception:  # noqa: BLE001
                    # If the matcher is in an unexpected state for any
                    # reason, treat the synthetic emit as plain. We'd
                    # rather emit a token than crash the request.
                    pass

                if finish_reason is None and gb._num_tokens[row] >= gb.max_tokens[row]:
                    finish_reason = "length"

                if finish_reason is not None:
                    # Roll back KV cache for any *unconsumed* accepted
                    # drafts. The verify forward in ``_suffix_step``
                    # advanced the cache through ALL ``n_accepted``
                    # drafts; if we stop early at ``emit_idx``, the
                    # remaining ``len(pending) - emit_idx - 1`` drafts
                    # were never surfaced — their KV state must come
                    # back out of the cache or it'll poison prefix-cache
                    # reuse for the next request that hits this prefix.
                    unused = len(pending) - emit_idx - 1
                    if unused > 0:
                        for c in gb.prompt_cache:
                            if (
                                hasattr(c, "is_trimmable")
                                and c.is_trimmable()
                                and hasattr(c, "trim")
                            ):
                                c.trim(unused)
                    augmented.append(
                        gb.Response(
                            uid=uid,
                            token=tok,
                            logprobs=lp,
                            finish_reason=finish_reason,
                            current_state=current_state,
                            match_sequence=match_sequence,
                            prompt_cache=gb.extract_cache(row),
                            all_tokens=gb.tokens[row],
                        )
                    )
                    # Filter the finished sequence out of gb.
                    keep = [i for i in range(len(gb.uids)) if i != row]
                    if keep:
                        gb.filter(keep)
                    else:
                        # Cleared the only sequence; reset the batch.
                        gb.filter([])
                    # Drop the drafter — sequence is done, its history
                    # would otherwise live in _drafters until the
                    # BatchGenerator itself is replaced.
                    _drafters.pop(uid, None)
                    # No more pending to emit for this uid.
                    break

                augmented.append(
                    gb.Response(
                        uid=uid,
                        token=tok,
                        logprobs=lp,
                        finish_reason=None,
                        current_state=current_state,
                        match_sequence=match_sequence,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        return augmented

    gb._step = _suffix_step
    gb.next = _suffix_next
    # Telemetry attached to the BatchGenerator (where the rest of the
    # engine looks for it) and to gb for direct inspection.
    batch_gen._suffix_stats = _stats
    gb._suffix_stats = _stats
    # Expose the per-uid drafter dict for tests to assert lifecycle
    # cleanup. Production code should not mutate this directly.
    gb._suffix_drafters = _drafters

    logger.info(
        "[SuffixDecoding] installed: max_draft=%d, max_suffix_len=%d, "
        "min_confidence=%.2f (single-request fast path; B>1 falls through)",
        max_draft,
        max_suffix_len,
        min_confidence,
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
        config: SchedulerConfig | None = None,
        tool_logits_processor_factory: Any | None = None,
        model_config: Any | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
            tool_logits_processor_factory: Optional callable that creates a
                logits processor for tool call structural token biasing.
                Called with no args, returns a processor or None.
            model_config: Optional ``ModelConfig`` from
                ``vllm_mlx.model_auto_config``. Used as a capability gate for
                spec-decoding installs (SuffixDecoding refuses to enable on
                hybrid linear-attention models).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()
        self._tool_logits_processor_factory = tool_logits_processor_factory
        self.model_config = model_config

        # Detect if tokenizer is a processor (MLLM) and get the actual tokenizer
        self._actual_tokenizer = self._get_actual_tokenizer(tokenizer)

        # Per-request streaming detokenizers for UTF-8-safe incremental decode
        self._detokenizer_pool: dict[str, Any] = {}

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: dict[str, Request] = {}  # Running requests by ID
        self.requests: dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: set[str] = set()  # Recently finished

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: dict[str, int] = {}
        self.uid_to_request_id: dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: BatchGenerator | None = None
        self._current_sampler_params: tuple | None = None

        # Sampler cache: interns ``make_sampler`` results keyed on
        # ``(temp, top_p, min_p, top_k)``. Homogeneous concurrent
        # batches end up sharing one callable, which lets
        # ``_install_dense_sampler_fastpath`` detect them by identity and
        # swap to mlx-lm's batched fast path.
        #
        # Bounded LRU (``OrderedDict``) because the cache key is
        # request-controlled: an adversarial client could otherwise
        # stream many unique float values for ``(temp, top_p, min_p,
        # top_k)`` and grow the cache without bound. Production traffic
        # almost always converges to one or two distinct keys, so a
        # small cap is more than enough; evicting an entry just costs
        # one ``make_sampler`` call the next time that key reappears.
        self._sampler_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._sampler_cache_max = 32

        # Prefix cache for KV state reuse
        self.prefix_cache: PrefixCacheManager | None = None
        self.memory_aware_cache: MemoryAwarePrefixCache | None = None
        self.paged_cache_manager: PagedCacheManager | None = None
        self.block_aware_cache: BlockAwarePrefixCache | None = None

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
                    kv_turboquant=self.config.kv_cache_turboquant,
                    kv_turboquant_bits=self.config.kv_cache_turboquant_bits,
                    kv_turboquant_group_size=self.config.kv_cache_turboquant_group_size,
                )
                self.memory_aware_cache = MemoryAwarePrefixCache(
                    model=model,
                    config=cache_config,
                )
                logger.info(
                    f"Memory-aware cache enabled: "
                    f"limit={self.memory_aware_cache.memory_limit_mb:.1f}MB"
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
        self._pending_abort_ids: set[str] = set()

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # PFlash observability (M-02 reframe). When PFlash compresses a
        # prompt the request bypasses the prefix-cache fetch + store
        # paths entirely (positional-fiction safety; see comment block
        # near ``compress_request_tokens``). That bypass is correct but
        # silences ``rapid_mlx_prefix_cache_*`` on PFlash-always tiers
        # (e.g. verified-tier aliases), making /metrics look frozen at
        # ``hits=0/misses=1``. These two counters let operators see
        # PFlash is doing meaningful work even when the cache series
        # stays flat. Observability only — bypass semantics unchanged.
        self.pflash_bypass_count = 0
        self.pflash_compressed_tokens_dropped = 0

        # Memory management: periodic mx.clear_cache() to free Metal command buffers
        # Lower interval = less VRAM spike during generation but slight throughput cost
        self._step_count = 0
        self._clear_cache_interval = 32
        self._memory_log_interval = 256

        # Prompt-boundary cache snapshot callback for the new mlx-lm 0.31+ API.
        # Built lazily once memory_aware_cache exists and reused per step.
        # Without this hook, hybrid models can't satisfy repeated identical
        # prompts via supersequence fallback (issue #163).
        self._prompt_cache_save_cb = (
            self._make_prompt_cache_save_callback()
            if self.memory_aware_cache is not None
            else None
        )

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

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text, handling both tokenizers and processors.
        """
        return self._actual_tokenizer.decode(token_ids)

    def _get_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for a request."""
        if request_id not in self._detokenizer_pool:
            if hasattr(self.tokenizer, "detokenizer"):
                detok = self.tokenizer.detokenizer
            else:
                detok = NaiveStreamingDetokenizer(self._actual_tokenizer)
            detok.reset()
            self._detokenizer_pool[request_id] = detok
        return self._detokenizer_pool[request_id]

    def _cleanup_detokenizer(self, request_id: str) -> None:
        """Remove the streaming detokenizer for a finished request."""
        self._detokenizer_pool.pop(request_id, None)

    def _get_stop_tokens(self) -> set[int]:
        """Get stop token IDs from tokenizer or processor.

        Resolution order (all sources unioned — set semantics make
        overlap harmless):

        1. ``TokenizerWrapper._eos_token_ids`` — the curated set
           mlx-lm's own ``BatchGenerator`` uses to halt generation.
           Grown at load time by
           ``augment_eos_token_ids_from_generation_config`` to
           include the chat-template terminator (Gemma 3
           ``<end_of_turn>``, Qwen3 ``<|endoftext|>``, Llama 3
           ``<|eot_id|>``, etc.).
        2. ``tok.eos_token_id`` — the underlying HF tokenizer's
           primary id. Required for non-wrapped tokenizers (custom
           fallback paths, mlx-vlm processor objects).
        3. ``tok.eos_token_ids`` — some processors expose the plural
           form natively.
        4. ``tok._rapid_extra_eos_token_ids`` — the union stashed by
           ``augment_eos_token_ids_from_generation_config`` on raw
           HF tokenizers whose ``eos_token_ids`` is a property that
           rejects non-string assignment. This is the surface that
           rescues mlx-vlm processors (Gemma 3 VL etc.).
        """
        from .utils.tokenizer import RAPID_EXTRA_EOS_ATTR

        stop_tokens: set[int] = set()
        # Check both the processor/tokenizer and the actual tokenizer
        for tok in [self.tokenizer, self._actual_tokenizer]:
            if tok is None:
                continue
            # Source 1: mlx-lm TokenizerWrapper's curated set.
            wrapper_ids = getattr(tok, "_eos_token_ids", None)
            if wrapper_ids:
                stop_tokens.update(wrapper_ids)
            # Source 2: legacy singular path.
            if hasattr(tok, "eos_token_id") and tok.eos_token_id is not None:
                if isinstance(tok.eos_token_id, list):
                    stop_tokens.update(tok.eos_token_id)
                else:
                    stop_tokens.add(tok.eos_token_id)
            # Source 3: processor-style plural path.
            if hasattr(tok, "eos_token_ids") and tok.eos_token_ids is not None:
                if isinstance(tok.eos_token_ids, (list, set, tuple)):
                    stop_tokens.update(tok.eos_token_ids)
                else:
                    stop_tokens.add(tok.eos_token_ids)
            # Source 4: Rapid-MLX extras stash (see RAPID_EXTRA_EOS_ATTR).
            extras = getattr(tok, RAPID_EXTRA_EOS_ATTR, None)
            if extras:
                stop_tokens.update(extras)
        return stop_tokens

    def _get_request_sampler(self, sampling_params: SamplingParams) -> Any:
        """Return a cached sampler for these sampling params.

        Interning samplers by ``(temp, top_p, min_p, top_k)`` is what
        lets ``_install_dense_sampler_fastpath`` detect homogeneous
        batches via identity comparison on ``GenerationBatch.samplers``.
        Without this, every request would carry its own
        ``make_sampler`` closure even when the params are identical,
        forcing the slow per-row loop in mlx-lm.

        WARNING: the cache key intentionally covers only the four
        knobs threaded through to ``make_sampler``. If we ever start
        forwarding xtc_probability / xtc_threshold / xtc_special_tokens
        per request, the key MUST grow accordingly — otherwise
        homogeneous-looking batches would silently share an incorrect
        sampler.
        """
        # Codex round-2 BLOCKER #3 fix: read the env-var BEFORE the cache
        # lookup so that flipping ``RAPID_MLX_DISABLE_FUSED_SAMPLER`` in a
        # long-lived process can disable the fast path on the next request
        # without us serving a stale cached fused sampler. The disabled
        # state is folded into the cache key so the two branches don't
        # collide either.
        # Codex round-5 NIT: accept a small set of truthy values so operators
        # who set ``RAPID_MLX_DISABLE_FUSED_SAMPLER=true`` (the more natural
        # form for a boolean knob) actually get the fast path disabled,
        # instead of silently leaving it on.
        _fused_disabled = os.environ.get(
            "RAPID_MLX_DISABLE_FUSED_SAMPLER", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        key = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
            sampling_params.top_k,
            _fused_disabled,
        )
        cached = self._sampler_cache.get(key)
        if cached is not None:
            # LRU bookkeeping — keep the hot key warm.
            self._sampler_cache.move_to_end(key)
            return cached
        # Fast path for the dominant chat config (temp + top_p, with or
        # without top_k). See ``vllm_mlx/_sampler_fast_path.py`` for the
        # math-equivalence argument and the perf data behind it (Qwen 3.6
        # 35B 4-bit B=1: 65 -> 92 tok/s). Falls through to mlx-lm's chain
        # whenever the request enables min_p, xtc, sets temperature == 0
        # (mlx-lm already short-circuits to argmax), is top-k-only with no
        # nucleus cut (mlx-lm uses a cheaper partition primitive there),
        # or whenever the operator sets
        # ``RAPID_MLX_DISABLE_FUSED_SAMPLER=1`` as an escape hatch.
        if not _fused_disabled and is_fused_top_p_eligible(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        ):
            sampler = make_fused_top_p_temp_sampler(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
            )
            if not getattr(self, "_fused_top_p_logged", False):
                logger.info(
                    "[fused_top_p_sampler] engaged for temp=%.3f top_p=%.3f top_k=%d",
                    sampling_params.temperature,
                    sampling_params.top_p,
                    sampling_params.top_k,
                )
                self._fused_top_p_logged = True
        else:
            sampler = make_sampler(
                temp=sampling_params.temperature,
                top_p=sampling_params.top_p,
                min_p=sampling_params.min_p,
                top_k=sampling_params.top_k,
            )
        self._sampler_cache[key] = sampler
        # Evict the least-recently-used entry once we exceed the cap.
        # Identity-sharing only matters for live in-flight batches; a
        # freshly evicted sampler that reappears just costs one
        # ``make_sampler`` call.
        if len(self._sampler_cache) > self._sampler_cache_max:
            self._sampler_cache.popitem(last=False)
        return sampler

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        )

        stop_tokens = _assemble_stop_tokens(sampling_params, self._get_stop_tokens())

        # mlx-lm 0.31.3+: BatchGenerator captures generation_stream at __init__
        # via a thread-local Stream; without an explicit stream= the captured
        # stream is whatever the import-thread had — which on the asyncio loop
        # thread is unreachable from the mlx-step worker that runs .next(),
        # so every request fails with "There is no Stream(gpu, 1) in current
        # thread" (#170 hot path; complements the warmup fix in PR #173).
        # _create_batch_generator runs on the mlx-step thread so default_stream
        # here is the worker's stream (our `_init_mlx_step_thread` sets it).
        bg_kwargs = dict(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
        )
        try:
            import mlx.core as _mx

            bg = BatchGenerator(
                **bg_kwargs, stream=_mx.default_stream(_mx.default_device())
            )
        except TypeError:
            # mlx-lm < 0.31.3 — no `stream` kwarg; fall back to legacy path.
            bg = BatchGenerator(**bg_kwargs)

        # Install chunked prefill when explicitly configured OR when
        # memory-aware cache is active (needed for prefix_boundary saves
        # in agentic multi-turn workloads with hybrid Mamba+Transformer models).
        #
        # NOTE: mlx-lm 0.31+ has native prefill_step_size support in BatchGenerator.
        # Our _install_chunked_prefill monkey-patches the old Batch API which was
        # removed in 0.31+. Skip the monkey-patch if the old API is unavailable.
        chunked_budget = self.config.chunked_prefill_tokens
        need_chunked = chunked_budget > 0 or self.memory_aware_cache is not None
        _has_old_batch_api = hasattr(bg, "_process_prompts")
        if need_chunked and _has_old_batch_api:
            if chunked_budget <= 0:
                # No explicit budget — use a very large value so normal
                # prompts pass through unchanged.  Prefix boundary splits
                # still trigger via _needs_boundary_split.
                chunked_budget = 999_999
            mid_prefill_cb = None
            save_interval = self.config.mid_prefill_save_interval
            if save_interval > 0 and self.memory_aware_cache is not None:
                mid_prefill_cb = self._make_mid_prefill_save_callback(save_interval)
                logger.info(f"[mid_prefill_cache] enabled, interval={save_interval}")
            prompt_cache_cb = None
            if self.memory_aware_cache is not None:
                prompt_cache_cb = self._make_prompt_cache_save_callback()
            _install_chunked_prefill(
                bg,
                chunked_budget,
                mid_prefill_cb,
                prompt_cache_save=prompt_cache_cb,
                pending_abort_ids=self._pending_abort_ids,
                uid_to_request_id=self.uid_to_request_id,
                requests=self.requests,
            )
        elif need_chunked and not _has_old_batch_api:
            # mlx-lm 0.31+ removed _process_prompts, so the full chunked
            # prefill monkey-patch can't run. The prompt-boundary cache
            # snapshot (the part that actually feeds the prefix cache)
            # is wired into Scheduler.step() via end_of_prompt response
            # signals — see _snapshot_promoted_prompts (issue #163).
            # The per-message boundary save is wired via insert_segments
            # + end_of_segment — see _snapshot_boundary_segments
            # (issue #427).
            if chunked_budget > 0:
                logger.info(
                    "[chunked_prefill] mlx-lm 0.31+ removed the legacy "
                    "Batch API; --chunked-prefill-tokens=%d is no-op'd "
                    "and native prefill_step_size=%d is used instead. "
                    "Per-message boundary snapshots ARE supported via "
                    "insert_segments (issue #427).",
                    chunked_budget,
                    self.config.prefill_step_size,
                )

        # Install MTP if the model supports it. SOP §10: gate on
        # ``model_config.supports_spec_decode`` so the --no-spec-decode
        # escape hatch (and any future override) takes effect on MTP
        # the same way it gates SuffixDecoding above.
        if self.config.enable_mtp:
            if (
                getattr(self, "model_config", None) is not None
                and not self.model_config.supports_spec_decode
            ):
                logger.warning(
                    "[MTP] --enable-mtp is set but profile says "
                    "supports_spec_decode=False (possibly via "
                    "--no-spec-decode). MTP will be disabled."
                )
            elif hasattr(self.model, "mtp") and self.model.mtp is not None:
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

        # Install SuffixDecoding (drafter-free spec-decode). Mutually
        # exclusive with --enable-mtp at the CLI layer.
        if self.config.enable_suffix_decoding:
            _install_suffix_decoding(
                bg,
                model=self.model,
                profile=self.model_config,
                max_draft=self.config.suffix_max_draft,
                max_suffix_len=self.config.suffix_max_suffix_len,
                min_confidence=self.config.suffix_min_confidence,
                min_draft_len=self.config.suffix_min_draft_len,
                requests=self.requests,
                uid_to_request_id=self.uid_to_request_id,
            )

        # Install batched-sampler fast path. Must run AFTER MTP /
        # SuffixDecoding since they may replace _step on the
        # GenerationBatch — our wrapper has to sit at the outermost
        # layer so it can short-circuit the per-row loop wherever the
        # final _step ends up. SuffixDecoding/MTP wrappers themselves
        # call into the original ``_step`` and ignore ``self.samplers``,
        # so this layering is safe.
        _install_dense_sampler_fastpath(bg)

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
            # PFlash bypass: see scheduler.add_request — compressed
            # prompt_token_ids are not positionally faithful so storing
            # KV under this key would poison the trie.
            if _pflash_compressed(request):
                return

            prompt_tokens = list(request.prompt_token_ids)
            _t0 = _time.monotonic()
            # evict_prefixes=False: keep mid-prefill boundary entries so
            # that future requests with the same prefix but different
            # suffix get a prefix cache hit (critical for agentic multi-turn).
            stored = self.memory_aware_cache.store(
                prompt_tokens, extracted_cache, evict_prefixes=False
            )
            _dt = _time.monotonic() - _t0
            if stored:
                logger.info(
                    f"[prompt_cache_save] request={request_id[:12]} "
                    f"prompt_tokens={len(prompt_tokens)} "
                    f"store_time={_dt:.3f}s"
                )

        return _prompt_cache_save

    def _snapshot_promoted_prompts(self, prompt_responses) -> None:
        """Snapshot prompt-only cache for sequences just promoted to generation.

        Reads the public ``end_of_prompt`` flag from mlx-lm 0.31+'s prompt
        responses, then uses the public ``BatchGenerator.extract_cache`` API
        to capture the per-uid cache state. Each capture is forwarded to the
        prompt-cache-save callback so a future request with the identical
        prompt finds an exact-match entry in the prefix cache.

        This is the new-API equivalent of the ``_patched_process_prompts``
        hook installed by ``_install_chunked_prefill`` for the legacy Batch
        API. Without it, hybrid models (Mamba/DeltaNet+Transformer) MISS
        the prefix cache forever because their non-trimmable cache layers
        cannot satisfy the supersequence fallback path (issue #163).
        """
        if self._prompt_cache_save_cb is None or not prompt_responses:
            return

        promoted_uids = [
            resp.uid
            for resp in prompt_responses
            if getattr(resp, "end_of_prompt", False)
        ]
        if not promoted_uids:
            return

        try:
            extracted = self.batch_generator.extract_cache(promoted_uids)
        except Exception as exc:
            logger.debug("[prompt_cache_save] extract_cache failed: %s", exc)
            return

        for uid, payload in extracted.items():
            # Promoted sequences (stage == 2) return (cache, tokens). Any
            # other shape means the uid was already removed before the
            # snapshot — skip silently.
            if isinstance(payload, tuple) and len(payload) == 2:
                cache, _tokens = payload
                try:
                    self._prompt_cache_save_cb(uid, cache)
                except Exception as exc:
                    logger.debug(
                        "[prompt_cache_save] callback failed for uid=%s: %s",
                        uid,
                        exc,
                    )

    def _snapshot_boundary_segments(self, prompt_responses) -> None:
        """Snapshot KV/Mamba cache at ``prefix_boundary`` for multi-turn workloads.

        Issue #427: hybrid models (linear-attention/Mamba + Transformer)
        MISS the LCP-based prefix cache on every turn of a growing
        conversation because the prior turn's cached entry has a tail
        that diverges from the new turn (e.g. ``<think>\\n`` template
        sentinel emitted by ``add_generation_prompt=True`` gets replaced
        by actual assistant content on the next turn) and Mamba layers
        are non-trimmable, so the supersequence fallback can't reuse
        the prefix either.

        Fix: when a request arrives with ``prefix_boundary > 0``,
        ``_schedule_waiting`` inserts it via ``insert_segments(
        [[prefix_seg, tail_seg]])`` so BatchGenerator processes the
        prefix segment as its own boundary. When that segment finishes,
        the response carries ``end_of_segment=True`` **without**
        ``end_of_prompt=True`` (the tail still has work to do). That's
        our cue to extract the cache via the public
        ``BatchGenerator.extract_cache`` API and store it under the
        ``prefix_boundary`` token prefix — so the *next* turn's lookup
        finds the boundary entry and skips re-prefilling the shared
        prefix.

        This is the mlx-lm 0.31+ replacement for the boundary-save
        path that was disabled when the legacy ``_install_chunked_prefill``
        monkey-patch could no longer run (the internal Batch API was
        removed in 0.31). The ``_make_mid_prefill_save_callback``
        infrastructure is still present for clients that downgrade to
        the legacy API; this new path coexists rather than replaces it.
        """
        if self.memory_aware_cache is None or not prompt_responses:
            return

        boundary_uids: list[int] = []
        for resp in prompt_responses:
            if not getattr(resp, "end_of_segment", False):
                continue
            # end_of_prompt promotions are handled by
            # _snapshot_promoted_prompts (whole-prompt entry, issue #163).
            # We only want the *inter*-segment boundary here.
            if getattr(resp, "end_of_prompt", False):
                continue
            request_id = self.uid_to_request_id.get(resp.uid)
            if not request_id:
                continue
            request = self.requests.get(request_id)
            if not request or getattr(request, "prefix_boundary", 0) <= 0:
                continue
            # Defense-in-depth: validate progress[0] equals the
            # expected boundary offset. mlx-lm 0.31+ rewrites
            # ``[[prefix, tail]]`` into ``[[prefix, tail[:-1], tail[-1:]]]``
            # when ``len(tail) > 1`` (generate.py:1646-1648), so
            # end_of_segment fires THREE times — once at prefix done,
            # once at tail[:-1] done, and end_of_prompt at tail[-1:].
            # The `_boundary_snapshot_taken` guard below blocks the
            # second fire, but this progress check skips it deterministically.
            progress = getattr(resp, "progress", None)
            expected_offset = request.prefix_boundary - (request.cached_tokens or 0)
            if (
                progress is not None
                and isinstance(progress, tuple)
                and len(progress) >= 1
                and progress[0] != expected_offset
            ):
                continue
            # Once-per-request guard: prevents a future API change that
            # repeats end_of_segment from producing duplicate stores.
            if getattr(request, "_boundary_snapshot_taken", False):
                continue
            boundary_uids.append(resp.uid)

        if not boundary_uids:
            return

        try:
            extracted = self.batch_generator.extract_cache(boundary_uids)
        except Exception as exc:
            logger.debug("[boundary_snapshot] extract_cache failed: %s", exc)
            return

        import time as _time

        for uid, payload in extracted.items():
            # Stage-1 (in-prompt) and stage-2 (promoted) both return
            # ``(cache, tokens)``. Anything else means the uid was
            # already removed before the snapshot — skip silently.
            if not (isinstance(payload, tuple) and len(payload) == 2):
                continue
            cache, _tokens = payload

            request_id = self.uid_to_request_id.get(uid)
            request = self.requests.get(request_id) if request_id else None
            if not request:
                continue
            # PFlash bypass: defensive guard. add_request also zeros
            # prefix_boundary for compressed requests so the next
            # condition would short-circuit anyway, but a future change
            # touching prefix_boundary must not silently start poisoning
            # the trie.
            if _pflash_compressed(request):
                continue
            prefix_boundary = getattr(request, "prefix_boundary", 0)
            if prefix_boundary <= 0:
                continue

            states = self._extract_cache_states(cache)
            if not states:
                continue
            reconstructed = self._reconstruct_cache_from_states(states)
            if not reconstructed:
                continue

            prefix_tokens = list(request.prompt_token_ids[:prefix_boundary])
            _t0 = _time.monotonic()
            stored = False
            try:
                # evict_prefixes=False matches the prompt-cache save
                # path — keep boundary entries so later turns with the
                # same prefix but different suffix still hit.
                stored = self.memory_aware_cache.store(
                    prefix_tokens, reconstructed, evict_prefixes=False
                )
            except Exception as exc:
                logger.debug(
                    "[boundary_snapshot] store failed for uid=%s: %s", uid, exc
                )
            _dt = _time.monotonic() - _t0
            # Mark the guard after the attempt (success OR failure) so a
            # repeated end_of_segment doesn't redo the expensive
            # extract+reconstruct cycle. A failed store usually means
            # the entry already exists (returns False) or the cache is
            # busy — retrying every step would be pure waste. DeepSeek
            # finding #2 on PR #435.
            request._boundary_snapshot_taken = True

            if stored:
                logger.info(
                    f"[boundary_snapshot] request={request_id[:12]} "
                    f"saved {prefix_boundary} tokens at message boundary "
                    f"store_time={_dt:.3f}s"
                )

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
            # PFlash bypass: see scheduler.add_request for the
            # positional-fiction rationale.
            if _pflash_compressed(request):
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

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> bool:
        """Ensure BatchGenerator exists with compatible stop token configuration.

        Returns:
            ``True`` if a compatible generator is ready for the caller
            to insert this request into. ``False`` if the caller MUST
            requeue the request — the current generator's stop_tokens
            differ from what this request requires, and there are active
            requests still draining, so admitting now would silently bind
            this request to the wrong stop-token set. The contract is a hard
            refusal, not advisory; ``_schedule_waiting`` requeues on ``False``
            to preserve stop_token and ignore_eos semantics across overlapping
            batches. Per-request samplers (temperature, top_p, etc.) do NOT
            trigger a new generator — they are passed directly to insert().
        """
        # Per-request samplers (temperature, top_p, min_p, top_k) are passed
        # directly to batch_gen.insert(..., samplers=[request_sampler]), so
        # they do NOT need to match the global BatchGenerator sampler. The
        # only generator-level invariant is stop_tokens, which is computed at
        # init time via _assemble_stop_tokens(sampling_params, model_stop_tokens).
        #
        # Two requests must share the same BatchGenerator iff they produce the
        # same final stop_tokens set. Since model_stop_tokens is invariant
        # (fixed per model/server run), the key is:
        # (frozenset(request.stop_token_ids), request.ignore_eos).
        #
        # Requests with different temperatures but same stop config can now
        # batch together, fixing issue where Claude Code's big agentic request
        # (33K tokens, default temp) blocked smaller concurrent requests with
        # different temps for 115 seconds (Rapid-MLX #611 follow-up).
        sampler_params = (
            frozenset(sampling_params.stop_token_ids or ()),
            bool(sampling_params.ignore_eos),
        )

        # Create new generator if needed or if sampling params changed
        if (
            self.batch_generator is None
            or self._current_sampler_params != sampler_params
        ):
            # If we have an existing generator with requests, the new
            # request's stop_tokens / sampler are incompatible with the
            # live generator. Refuse admission — the caller (
            # ``_schedule_waiting``) requeues and retries on the next
            # step, after the running batch has had a chance to drain.
            # Previously we returned without recreating but left the
            # stale generator in place; ``_schedule_waiting`` would then
            # insert the new request into it, silently inheriting the
            # wrong ignore_eos behavior (codex P2 on PR #612).
            if self.batch_generator is not None and self.running:
                logger.warning(
                    "Stop token configuration changed with active requests. "
                    "Requeuing request — admission deferred until current "
                    "batch drains so stop_tokens remain consistent."
                )
                return False

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

        return True

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
                    if (
                        hasattr(layer_cache.keys, "shape")
                        and layer_cache.keys.shape[0] != 1
                    ):
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

    def _extract_cache_states(self, raw_cache: list[Any]) -> list[dict[str, Any]]:
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
        self, extracted_states: list[dict[str, Any]]
    ) -> list[Any] | None:
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
                    )
                    from mlx_lm.models.cache import (
                        KVCache as _KVCache,
                    )

                    if cache_cls is _BatchKVCache:
                        # BatchKVCache.state = (keys, values, offset, left_padding)
                        keys, values = state[0], state[1]
                        cache = _KVCache()
                        cache.keys = keys
                        cache.values = values
                        cache.offset = keys.shape[2] if hasattr(keys, "shape") else 0
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
                        int(meta_state[0])
                        if meta_state
                        else (
                            cache.keys.shape[2] if hasattr(cache.keys, "shape") else 0
                        )
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

        Raises:
            BackpressureError: If the in-flight request count is at or
                above ``config.max_concurrent_requests``. Routes catch
                this and return 503 with Retry-After.
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Admission control: cap concurrent in-flight requests so a
        # buggy/abusive client can't OOM Metal and crash the server
        # for everyone else. Check BEFORE tokenization so the cost of
        # being over the cap is just a dict lookup.
        cap = self.config.max_concurrent_requests
        if cap is not None and cap > 0 and len(self.requests) >= cap:
            raise BackpressureError(
                f"max_concurrent_requests={cap} reached "
                f"(currently {len(self.requests)} in-flight)"
            )

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

        # Logical-vs-model prompt-length split (#287). num_prompt_tokens
        # is what gets reported to clients / usage tracking; PFlash may
        # shorten prompt_token_ids before prefill, so model_prompt_tokens
        # tracks the post-transform length used by the scheduler.
        if request.prompt_token_ids is not None and request.model_prompt_tokens == 0:
            request.model_prompt_tokens = len(request.prompt_token_ids)

        # PFlash long-prompt compression — must run before any cache
        # lookup. When compression engages, prompt_token_ids is replaced
        # by the kept-token subsequence and the prefix cache is bypassed
        # entirely (both fetch and store, see below) because the
        # compressed token sequence is a positional fiction: position i
        # in compressed land does NOT correspond to position i in the
        # original prompt, so reusing KV computed for the uncompressed
        # prefix would inject position-shifted state into a later
        # uncompressed request that shares the same sink prefix.
        pflash_compressed = False
        if self.config.pflash_config.mode != "off" and request.prompt_token_ids:
            original_tokens = list(request.prompt_token_ids)
            original_prefix_boundary = request.prefix_boundary
            scoring_start = time.monotonic()
            compressed_tokens, metadata = compress_request_tokens(
                original_tokens,
                self.config.pflash_config,
                has_tools=request.has_tools,
                requires_prompt_integrity=request.requires_prompt_integrity,
            )
            metadata["scoring_seconds"] = time.monotonic() - scoring_start
            metadata["logical_prompt_tokens"] = len(original_tokens)
            metadata["model_prompt_tokens"] = len(compressed_tokens)
            metadata["prefix_boundary_original"] = original_prefix_boundary
            metadata["prefix_boundary_disabled"] = False
            request.pflash_metadata = metadata
            if metadata["compressed"]:
                pflash_compressed = True
                # M-02: count every prompt that took the PFlash bypass
                # so /metrics surfaces the work that prefix-cache
                # counters can't (the compressed sequence skips both
                # fetch and store — see the explanation block above).
                # ``tokens_dropped`` = logical prompt length minus kept
                # length, i.e. the saving operators want for capacity
                # planning.
                self.pflash_bypass_count += 1
                self.pflash_compressed_tokens_dropped += max(
                    0, len(original_tokens) - len(compressed_tokens)
                )
                request.original_prompt_token_ids = original_tokens
                request.prompt_token_ids = compressed_tokens
                request.model_prompt_tokens = len(compressed_tokens)
                # prefix_boundary indexes into the ORIGINAL prompt; the
                # compressed sequence is non-prefix so a boundary save
                # would point at meaningless tokens. Force-disable.
                if original_prefix_boundary > 0:
                    request.prefix_boundary = 0
                    metadata["prefix_boundary_disabled"] = True
                logger.info(
                    f"[pflash] request={request.request_id[:12]} "
                    f"compressed {metadata['original_tokens']} -> "
                    f"{metadata['kept_tokens']} tokens "
                    f"ratio={metadata['compression_ratio']:.3f} "
                    f"scoring_ms={metadata['scoring_seconds'] * 1000.0:.2f}"
                )
            else:
                logger.debug(
                    f"[pflash] request={request.request_id[:12]} skipped "
                    f"reason={metadata['reason']} tokens={metadata['original_tokens']}"
                )

        # Check prefix cache for cached KV state. Compressed requests
        # MUST skip the lookup — see PFlash comment above for the
        # positional-fiction explanation.
        if pflash_compressed:
            request.cache_hit_type = "miss"
            request.remaining_tokens = request.prompt_token_ids
        elif self.block_aware_cache is not None:
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
            True when an active/queued request was enqueued for abort, False
            when ``request_id`` is unknown to this scheduler. F-151 hardening:
            previously this method returned True unconditionally — including
            for arbitrary attacker-supplied strings — which let the route
            layer respond ``{"cancelled": true}`` for any id. The route uses
            the False return as the 404 signal.
        """
        # Consider the request "known" if it lives in any of: the canonical
        # ``requests`` dict (admitted but not finished), the BatchGenerator
        # uid map (admitted into a live batch — may already have been popped
        # from ``requests`` by an in-flight ``_cleanup_request``), the
        # ``running`` map (currently scheduled), or ``_pending_abort_ids``
        # (a concurrent abort enqueue made this method idempotent — return
        # True so a double-cancel doesn't 404 the second caller). We do NOT
        # treat ``finished_req_ids`` as "known" because the abort would be
        # a no-op and the route contract is "404 when already finished".
        if (
            request_id in self.requests
            or request_id in self.request_id_to_uid
            or request_id in self.running
            or request_id in self._pending_abort_ids
        ):
            self._pending_abort_ids.add(request_id)
            logger.info(
                f"[abort_request] {request_id[:12]} enqueued for deferred abort"
            )
            return True
        logger.info("[abort_request] unknown request_id (rejected without enqueue)")
        return False

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

        # Remove from waiting queue.
        # When request is not None we can remove by identity; when it's None
        # (already popped by _cleanup_request) we must scan by request_id so
        # the deque entry doesn't survive the abort.
        if request is not None and request.status == RequestStatus.WAITING:
            was_waiting = True
            try:
                self.waiting.remove(request)
            except ValueError:
                pass
        elif request is None:
            # Scan waiting deque by request_id — request object was already
            # removed from self.requests but may still sit in the deque.
            for waiting_req in list(self.waiting):
                if waiting_req.request_id == request_id:
                    was_waiting = True
                    try:
                        self.waiting.remove(waiting_req)
                    except ValueError:
                        pass
                    break

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

    def _schedule_waiting(self) -> list[Request]:
        """
        Move requests from waiting queue to running.

        Returns:
            List of requests that were scheduled
        """
        scheduled = []

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            request = self.waiting.popleft()

            # Ensure we have a batch generator. The False return means
            # the live generator has incompatible stop_tokens / sampler
            # for this request and is still draining — we must NOT admit
            # into the stale generator (Rapid-MLX #611 / codex P2 on
            # PR #612). Requeue and break so the next ``step`` retries
            # once the running batch completes.
            if not self._ensure_batch_generator(request.sampling_params):
                self.waiting.appendleft(request)
                break

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

            # Insert into BatchGenerator with optional cache.
            # Wrap in try/except: if cache shapes are incompatible
            # (e.g. stale entry after BatchGenerator recreation),
            # fall back to no-cache insert instead of crashing.
            # Create per-request logits processors
            request_processors: list = []
            if self._tool_logits_processor_factory:
                processor = self._tool_logits_processor_factory()
                if processor is not None:
                    request_processors.append(processor)
            # Penalty knobs (#355) — only add the processor when at least
            # one penalty is non-default. mlx-lm's make_logits_processors
            # returns an empty list when all knobs are at defaults, but
            # constructing it unconditionally would still allocate the
            # context-tracking arrays for every request.
            #
            # OpenAI-spec penalties (frequency/presence) are defined over
            # the entire generated sequence, not a sliding window. mlx-lm's
            # default context_size of 20 truncates the visibility window so
            # aggressively that callers report the penalty "feels like a
            # no-op" on chat-length outputs (#470). We bump the OpenAI-spec
            # ones to 4096 — enough to cover the vast majority of chat
            # responses without bloating per-request arrays. Repetition
            # penalty stays at mlx-lm's default 20 since it's a rapid-mlx
            # extension (not OpenAI-spec) and is documented as multiplicative
            # over a rolling window.
            sp = request.sampling_params
            if (
                sp.repetition_penalty != 1.0
                or sp.presence_penalty != 0.0
                or sp.frequency_penalty != 0.0
            ):
                request_processors.extend(
                    make_logits_processors(
                        repetition_penalty=(
                            sp.repetition_penalty
                            if sp.repetition_penalty != 1.0
                            else None
                        ),
                        presence_penalty=(
                            sp.presence_penalty if sp.presence_penalty != 0.0 else None
                        ),
                        presence_context_size=4096,
                        frequency_penalty=(
                            sp.frequency_penalty
                            if sp.frequency_penalty != 0.0
                            else None
                        ),
                        frequency_context_size=4096,
                    )
                )
            request_logits_processors = (
                [request_processors] if request_processors else None
            )

            # Per-request sampler (temperature/top_p/top_k/min_p may differ
            # per request). Without this, all requests use the BatchGenerator's
            # default sampler (argmax), ignoring the requested temperature.
            # ``_get_request_sampler`` interns by sampling-param tuple so that
            # homogeneous batches share one callable — required for
            # ``_install_dense_sampler_fastpath`` to detect them by identity.
            request_sampler = self._get_request_sampler(request.sampling_params)

            # Issue #427: split the insert at prefix_boundary so the
            # per-message cache snapshot can fire after the prefix
            # segment prefills (see _snapshot_boundary_segments). Only
            # useful when (a) we have somewhere to save, (b) the request
            # has a multi-turn shared prefix set, and (c) the boundary
            # lies strictly inside the tokens we're about to process —
            # otherwise there's nothing new to capture at the boundary.
            boundary_local_split: int | None = None
            if (
                self.memory_aware_cache is not None
                and getattr(request, "prefix_boundary", 0) > 0
                and len(tokens_to_process) > 1
            ):
                _pb = request.prefix_boundary
                _cached = request.cached_tokens or 0
                _local = _pb - _cached
                if 0 < _local < len(tokens_to_process):
                    boundary_local_split = _local

            try:
                if boundary_local_split is not None:
                    uids = self.batch_generator.insert_segments(
                        [
                            [
                                tokens_to_process[:boundary_local_split],
                                tokens_to_process[boundary_local_split:],
                            ]
                        ],
                        max_tokens=[request.sampling_params.max_tokens],
                        caches=[cache_to_use] if cache_to_use else None,
                        samplers=[request_sampler],
                        logits_processors=request_logits_processors,
                    )
                else:
                    uids = self.batch_generator.insert(
                        [tokens_to_process],
                        max_tokens=[request.sampling_params.max_tokens],
                        caches=[cache_to_use] if cache_to_use else None,
                        samplers=[request_sampler],
                        logits_processors=request_logits_processors,
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
                    # Recompute split against the now-full prompt
                    # (cached_tokens=0 so boundary == split).
                    if (
                        self.memory_aware_cache is not None
                        and getattr(request, "prefix_boundary", 0) > 0
                        and 0 < request.prefix_boundary < len(tokens_to_process)
                    ):
                        uids = self.batch_generator.insert_segments(
                            [
                                [
                                    tokens_to_process[: request.prefix_boundary],
                                    tokens_to_process[request.prefix_boundary :],
                                ]
                            ],
                            max_tokens=[request.sampling_params.max_tokens],
                            caches=None,
                            samplers=[request_sampler],
                            logits_processors=request_logits_processors,
                        )
                    else:
                        uids = self.batch_generator.insert(
                            [tokens_to_process],
                            max_tokens=[request.sampling_params.max_tokens],
                            caches=None,
                            samplers=[request_sampler],
                            logits_processors=request_logits_processors,
                        )
                else:
                    raise

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                # Attach incremental decoder for multi-byte safe streaming
                request._decoder = IncrementalDecoder(self._actual_tokenizer)
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
                logger.info(
                    f"[schedule] request={request.request_id[:12]} uid={uid} "
                    f"prompt_tokens={request.num_prompt_tokens} "
                    f"tokens_to_prefill={tokens_to_prefill}{cache_info} "
                    f"max_tokens={request.sampling_params.max_tokens} "
                    f"running={len(self.running)} waiting={len(self.waiting)}"
                )

        return scheduled

    def _process_batch_responses(
        self, responses: list[Any]
    ) -> tuple[list[RequestOutput], set[str]]:
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

            # Decode the new token using IncrementalDecoder for multi-byte
            # safety (emoji, CJK). Skip stop tokens — they are not content.
            if response.finish_reason == "stop":
                new_text = ""
            else:
                decoder = getattr(request, "_decoder", None)
                if decoder is not None:
                    new_text = decoder.add_token(response.token)
                else:
                    new_text = self._decode_tokens([response.token])

            # output_token_ids is a live reference (not a defensive copy):
            # consumers read it synchronously; the per-decode list() was O(n).
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token],
                new_text=new_text,
                output_token_ids=request.output_token_ids,
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                cached_tokens=request.cached_tokens,
                logprobs=response.logprobs,
            )

            # Check text-based stop sequences. ``SamplingParams.stop`` is a
            # list of user-supplied strings (OpenAI-API contract); mlx-lm's
            # BatchGenerator only honours ``stop_token_ids``, so we have to
            # match-and-truncate on the decoded output here. MLLMScheduler
            # has had the equivalent check since launch; the text scheduler
            # was silently dropping ``request.stop`` until #354 / regression
            # tests 1, 2, 4, 5 surfaced the gap.
            #
            # Surface choice: the IncrementalDecoder's ``get_full_text()`` is
            # the AUTHORITATIVE surface for stop matching — it is what the
            # client sees on the streaming path, byte-for-byte. The previous
            # implementation called ``self._decode_tokens()`` (a fresh
            # ``tokenizer.decode(token_ids)`` with the wrapper's default
            # ``skip_special_tokens=True``), which on tokenizer families
            # whose default decoding strips text the streaming detokenizer
            # preserves (Phi-3.5 / Gemma-3n SentencePiece variants surfaced
            # in the 2026-06-18 fuzz battery) produced a window that did
            # NOT contain the literal stop string the user was looking
            # for — even though the streamed surface DID contain it. Using
            # the incremental decoder closes this skew at the sampler
            # layer without tokenizer-family-specific casing.
            finish_reason = response.finish_reason
            stop_trimmed = False
            stop_params = request.sampling_params.stop or []
            if finish_reason is None and stop_params:
                decoder = getattr(request, "_decoder", None)
                if decoder is not None:
                    decoded_so_far = decoder.get_full_text()
                else:
                    decoded_so_far = self._decode_tokens(request.output_token_ids)
                for stop_str in stop_params:
                    if stop_str and stop_str in decoded_so_far:
                        finish_reason = "stop"
                        # H-03: pin WHICH user-supplied stop fired so
                        # the Anthropic adapter can surface
                        # ``stop_reason="stop_sequence"`` +
                        # ``stop_sequence: <str>`` per the public spec.
                        # OpenAI's ``finish_reason="stop"`` bucket
                        # already lumps EOS and stop-string together so
                        # the OpenAI surface ignores this field.
                        output.matched_stop = stop_str
                        idx = decoded_so_far.index(stop_str)
                        trimmed_total = decoded_so_far[:idx]
                        request.output_text = trimmed_total
                        stop_trimmed = True
                        # Adjust new_text so streaming clients only see the
                        # valid prefix, never the stop marker itself.
                        # Pre-token streaming surface ≡ the decoder's
                        # ``prev_text``  — what the client has seen so
                        # far. Computing it as ``decoded_so_far -
                        # new_text`` is fragile: when the incremental
                        # decoder holds back a U+FFFD-incomplete
                        # sequence ``new_text == ""`` but
                        # ``decoded_so_far`` grew, so the subtraction
                        # math reset to the wrong boundary and could
                        # leak or drop text on multibyte streams
                        # (codex r8 BLOCKING). Falling back to
                        # ``decoded_so_far - new_text`` only when no
                        # decoder is attached (text-only paths that
                        # decode in bulk).
                        if decoder is not None:
                            prev_text = decoder.prev_text
                        else:
                            prev_text = (
                                decoded_so_far[: -len(new_text)]
                                if new_text
                                else decoded_so_far
                            )
                        if len(trimmed_total) > len(prev_text):
                            output.new_text = trimmed_total[len(prev_text) :]
                        else:
                            output.new_text = ""
                        break

            # Check if finished
            if finish_reason is not None:
                response.finish_reason = finish_reason
                if response.finish_reason == "stop":
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif response.finish_reason == "length":
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

                output.finished = True
                output.finish_reason = response.finish_reason
                finished_ids.add(request_id)

                if stop_trimmed:
                    # request.output_text was already truncated to the prefix
                    # before the stop string — using that as the final output
                    # preserves the truncation; re-decoding here would put the
                    # stop marker back in.
                    output.output_text = request.output_text
                    self._cleanup_detokenizer(request_id)
                else:
                    # Decode full output using decoder if available (ensures
                    # any held-back multi-byte chars are flushed)
                    decoder = getattr(request, "_decoder", None)
                    if decoder is not None:
                        output.output_text = decoder.get_full_text()
                    else:
                        output.output_text = self._decode_tokens(
                            request.output_token_ids
                        )
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

    def _cleanup_finished(self, finished_ids: set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        for request_id in finished_ids:
            request = self.running.get(request_id)

            # PFlash bypass: compressed requests skip the prefix-cache
            # store entirely. Their prompt_token_ids holds the
            # compressed subsequence so a stored entry would be keyed by
            # positions that do not match any real prompt prefix.
            pflash_skip_store = request is not None and _pflash_compressed(request)

            # Store cache for future reuse
            if (
                request is not None
                and request.prompt_token_ids
                and not pflash_skip_store
            ):
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

    def _recover_from_generation_error(self) -> set[str]:
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
        aborted_ids: set[str] = set()
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
                # Use model_prompt_tokens — when PFlash engages the
                # prefill workload is the compressed length, not the
                # logical (client-visible) prompt length.
                output.num_scheduled_tokens = sum(
                    r.model_prompt_tokens or r.num_prompt_tokens for r in scheduled
                )

                # Run generation step if we have running requests
                if self.batch_generator is not None and self.running:
                    raw_next = self.batch_generator.next()
                    output.has_work = True

                    # mlx-lm 0.31+ returns (prompt_responses, generation_responses) tuple
                    # older versions return a flat list of responses
                    if isinstance(raw_next, tuple):
                        prompt_responses, responses = raw_next
                        self._snapshot_promoted_prompts(prompt_responses)
                        # issue #427: per-message boundary snapshot for
                        # multi-turn hybrid workloads (segment finished
                        # but prompt still has tail to process).
                        self._snapshot_boundary_segments(prompt_responses)
                    else:
                        responses = raw_next

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
                            # OpenAI ChatCompletion only accepts {stop, length,
                            # tool_calls, content_filter, function_call}. We
                            # report "length" for aborted requests so spec-
                            # validating clients (openai-python, pydantic-ai)
                            # can parse the response; callers reading
                            # ``RequestOutput.error`` still see the abort
                            # details. (#v0.6.63 onboarding sweep)
                            finish_reason="length",
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
            # mlx-lm 0.31+ renamed active_batch to _generation_batch
            _active = None
            if self.batch_generator is not None:
                _active = getattr(
                    self.batch_generator, "active_batch", None
                ) or getattr(self.batch_generator, "_generation_batch", None)
            if _active is not None and hasattr(_active, "tokens"):
                tokens = _active.tokens
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

    def get_request(self, request_id: str) -> Request | None:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Request | None:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_running_requests_info(self) -> list[dict[str, Any]]:
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

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            # M-02: PFlash observability counters. ``bypass_count`` is
            # the number of requests where PFlash compression engaged
            # and the prefix-cache fetch/store was skipped;
            # ``compressed_tokens_dropped`` is the cumulative number of
            # prompt tokens removed by the compressor (logical minus
            # kept). Both default to zero on engines without PFlash so
            # /metrics renders a flat-line 0 instead of an absent
            # series.
            "pflash_bypass_count": self.pflash_bypass_count,
            "pflash_compressed_tokens_dropped": self.pflash_compressed_tokens_dropped,
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

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self.block_aware_cache is not None:
            return self.block_aware_cache.get_stats()
        elif self.memory_aware_cache is not None:
            return self.memory_aware_cache.get_stats()
        elif self.prefix_cache is not None:
            return self.prefix_cache.get_stats()
        return None

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
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        if self.memory_aware_cache is not None:
            self.memory_aware_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

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

    def save_cache_to_disk(self, cache_dir: str, should_abort=None) -> bool:
        """Save prefix cache to disk for persistence across restarts.

        ``should_abort`` is an optional ``Callable[[float], bool]``
        (the ``float`` is the next entry's predicted write duration)
        that signals the lifespan SIGTERM-grace deadline to the per-
        entry loop inside ``MemoryAwarePrefixCache.save_to_disk``.
        Zero-arg callables are accepted via auto-detection for
        backwards compatibility. See that method's docstring for the
        partial-commit guarantee.
        """
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.save_to_disk(
                cache_dir, should_abort=should_abort
            )
        logger.info("[cache_persist] no memory-aware cache to save")
        return False

    def load_cache_from_disk(self, cache_dir: str) -> int:
        """Load prefix cache from disk. Returns number of entries loaded."""
        if self.memory_aware_cache is not None:
            return self.memory_aware_cache.load_from_disk(cache_dir)
        logger.info("[cache_persist] no memory-aware cache to load into")
        return 0
