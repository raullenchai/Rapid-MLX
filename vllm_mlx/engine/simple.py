# SPDX-License-Identifier: Apache-2.0
"""
Simple engine for maximum single-user throughput.

This engine wraps mlx-lm directly with zero overhead for optimal
performance when serving a single user at a time.
"""

import asyncio
import contextvars
import hashlib
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

# Re-entrancy guard for SimpleEngine._track_request_stream so that
# internal fallback paths inside _stream_chat_impl (which call back into
# self.stream_generate) don't double-count a single external request.
# contextvars propagates per-asyncio-task, so concurrent requests still
# each get their own outermost tracking pass.
_in_tracker: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_simple_engine_in_tracker", default=False
)

import mlx.core as mx

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import clean_output_text, has_media_content, is_mllm_model
from .base import (
    BaseEngine,
    EngineBusy,
    GenerationOutput,
    cleanup_startup_cancellation,
    run_blocking_startup_work,
)
from .chat_template_safety import normalize_messages_for_chat_template
from ..mlx_streams import bind_generation_streams

logger = logging.getLogger(__name__)


class _StopStreamGate:
    """Hold-back-tail stop-sequence gate for streaming text deltas.

    Mirrors the scheduler and ``MLXLanguageModel.stream_generate``
    enforcement (issue #469) so wrapper paths that aggregate model
    chunks (text-mode MLLM, spec-prefill) get the same guarantees:

      * the matched stop string itself never reaches the client as a
        streamed text delta;
      * a partial prefix of any stop string is held back until the next
        chunk disambiguates it;
      * the final accumulated buffer is truncated at the match.

    Usage::

        gate = _StopStreamGate(["END", "STOP"])
        for raw_chunk in upstream:
            delta, finished, matched = gate.feed(raw_chunk)
            yield ...delta...
            if finished:
                break
        # Optional flush on length/abort termination:
        delta, _ = gate.flush()
    """

    def __init__(self, stop: list[str] | None):
        self._stops: list[str] = [s for s in (stop or []) if s]
        self._max_len: int = max((len(s) for s in self._stops), default=0)
        self._buf: str = ""
        self._published: int = 0
        self.matched: str | None = None
        self.finished: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self._stops)

    @property
    def buf(self) -> str:
        return self._buf

    def feed(self, chunk: str) -> tuple[str, bool, str | None]:
        """Append ``chunk`` and return ``(delta_to_emit, finished, matched)``.

        ``delta_to_emit`` may be empty when the current tail is wholly
        inside the safety window.  ``finished`` is True only when a
        stop string fully matched; in that case ``matched`` carries it.
        """
        if not self._stops:
            self._buf += chunk
            self._published = len(self._buf)
            return chunk, False, None

        self._buf += chunk
        best_idx: int | None = None
        for cand in self._stops:
            idx = self._buf.find(cand)
            if idx == -1:
                continue
            if best_idx is None or idx < best_idx or (
                idx == best_idx and len(cand) > len(self.matched or "")
            ):
                best_idx = idx
                self.matched = cand
        if self.matched is not None and best_idx is not None:
            self._buf = self._buf[:best_idx]
            delta = self._buf[self._published:]
            self._published = len(self._buf)
            self.finished = True
            return delta, True, self.matched

        # No match — hold back trailing ``max_len - 1`` chars.
        hold_back = self._max_len - 1
        safe_end = max(0, len(self._buf) - hold_back)
        if safe_end > self._published:
            delta = self._buf[self._published:safe_end]
            self._published = safe_end
            return delta, False, None
        return "", False, None

    def flush(self) -> tuple[str, str | None]:
        """Emit any remaining buffered text (called on length/abort)."""
        if self._published >= len(self._buf):
            return "", self.matched
        delta = self._buf[self._published:]
        self._published = len(self._buf)
        return delta, self.matched


def _bind_worker_generation_streams() -> None:
    """Rebind mlx generation streams inside the current worker thread."""
    bind_generation_streams()


def _seed_logits_processors(
    seed_tokens: mx.array | None,
    processors: list[Any] | None,
) -> list[Any] | None:
    """Wrap logits processors so continuation decode sees the full prompt."""
    if not processors:
        return None
    if seed_tokens is None or seed_tokens.size == 0:
        return list(processors)

    def _wrap(processor):
        def _seeded(tokens, logits):
            merged = seed_tokens
            if tokens is not None:
                if not isinstance(tokens, mx.array):
                    tokens_arr = mx.array(tokens, dtype=mx.uint32)
                else:
                    tokens_arr = tokens
                if tokens_arr.size > 0:
                    merged = mx.concatenate([seed_tokens, tokens_arr])
            return processor(merged, logits)

        return _seeded

    return [_wrap(processor) for processor in processors]


def _sample_with_processors(
    tokens: mx.array | None,
    logits: mx.array,
    sampler: Any,
    logits_processors: list[Any] | None,
) -> tuple[mx.array, mx.array]:
    """Sample a token while honoring any active logits processors."""
    if logits_processors:
        is_1d = logits.ndim == 1
        if is_1d:
            logits = logits[None]
        for processor in logits_processors:
            logits = processor(tokens, logits)
        if is_1d:
            logits = logits.squeeze(0)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok = sampler(logprobs)
    return tok, logprobs


def _processors_can_retire(processors: list[Any] | None) -> bool:
    """True when any processor advertises a retire-to-content transition."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        isinstance(getattr(p, "is_retired", None), bool) for p in processors
    )


def _processors_retired(processors: list[Any] | None) -> bool:
    """True when any retire-capable processor has entered its retired state."""
    if os.getenv("VLLM_MLX_ENABLE_THINKING_RETIREMENT_RESUME") != "1":
        return False
    return bool(processors) and any(
        getattr(p, "is_retired", False) is True for p in processors
    )


class _SpecPrefillCancelled(Exception):
    """Cooperative cancellation sentinel for blocking SpecPrefill workers."""


class SimpleEngine(BaseEngine):
    """
    Simple engine for direct model calls.

    This engine provides maximum throughput for single-user scenarios
    by calling mlx-lm/mlx-vlm directly without batching overhead.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        enable_cache: bool = True,
        force_mllm: bool = False,
        mtp: bool = False,
        mtp_num_draft_tokens: int = 1,
        prefill_step_size: int = 2048,
        specprefill_enabled: bool = False,
        specprefill_threshold: int = 8192,
        specprefill_keep_pct: float = 0.3,
        specprefill_backbone_pct: float = 0.0,
        specprefill_draft_model: str | None = None,
        max_kv_size: int = 0,
        mllm_draft_model: str | None = None,
        mllm_draft_kind: str | None = None,
        mllm_draft_block_size: int | None = None,
    ):
        """
        Initialize the simple engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Whether to trust remote code
            enable_cache: Enable VLM cache for multimodal models
            force_mllm: Force loading as MLLM even if not auto-detected
            mtp: Enable native MTP speculative decoding (model must have MTP head)
            mtp_num_draft_tokens: Draft tokens per speculative MTP step
            prefill_step_size: Chunk size for prompt prefill processing (default: 2048)
            specprefill_enabled: Enable SpecPrefill (attention-based sparse prefill)
            specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill
            specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
            specprefill_backbone_pct: Fraction of chunks to reserve for evenly
                spaced coverage (default: 0.0)
            specprefill_draft_model: Path to small draft model for importance scoring
            max_kv_size: Maximum KV cache size per sequence (0 = unbounded)
            mllm_draft_model: Optional MLLM speculative draft/assistant model path
            mllm_draft_kind: Optional mlx-vlm draft kind, for example "mtp"
            mllm_draft_block_size: Optional speculative block size for mlx-vlm
        """
        self._model_name = model_name
        self._created_at = time.time()
        self._trust_remote_code = trust_remote_code
        self._enable_cache = enable_cache
        self._is_mllm = force_mllm or is_mllm_model(model_name)
        self._mtp = mtp
        self._mtp_num_draft_tokens = mtp_num_draft_tokens
        self._prefill_step_size = prefill_step_size

        # Request stats (parity with BatchedEngine for /v1/status monitoring).
        # Without these, monitoring sees zero traffic for SimpleEngine-backed
        # servers (e.g. Gemma 4 31B with --mllm-draft-model + MTP).
        self._total_requests_processed: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._num_running: int = 0
        # Rolling window of (completion_tokens, duration_s) for tps computation.
        self._recent_completions: deque = deque(maxlen=20)
        # Live per-request state, mirroring BatchedEngine's "requests" list
        # in /v1/status (request_id, phase, ttft_s, tokens_per_second, ...).
        self._active_requests: dict[str, dict[str, Any]] = {}

        # SpecPrefill config
        self._specprefill_enabled = specprefill_enabled
        self._specprefill_threshold = specprefill_threshold
        self._specprefill_keep_pct = specprefill_keep_pct
        self._specprefill_backbone_pct = specprefill_backbone_pct
        self._specprefill_draft_model_path = specprefill_draft_model
        self._mllm_draft_model_path = mllm_draft_model
        self._mllm_draft_kind = mllm_draft_kind
        self._mllm_draft_block_size = mllm_draft_block_size

        # KV cache size limit
        self._max_kv_size = max_kv_size

        self._model = None
        self._loaded = False

        # Per-request routing state (MLLM+MTP mode)
        self._text_model = None
        self._text_tokenizer = None

        # SpecPrefill draft model (loaded at start if enabled)
        self._draft_model = None

        # Lock to serialize MLX operations (prevents Metal command buffer conflicts)
        self._generation_lock = asyncio.Lock()
        self._generation_lock_admission = (
            os.environ.get("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "fail_fast")
            .strip()
            .lower()
        )
        if self._generation_lock_admission not in {"fail_fast", "wait"}:
            logger.warning(
                "Invalid VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION=%r; using fail_fast",
                self._generation_lock_admission,
            )
        self._generation_lock_admission = "fail_fast"
        self._generation_waiters = 0
        self._generation_busy_rejections = 0

        # System prompt KV cache (reduces repeated prefill across requests).
        # OrderedDict acts as an LRU keyed by system-prefix hash so that the
        # main agent and any sub-agents with different toolsets can coexist
        # without thrashing a single snapshot slot.
        # Value is (snapshot_list, system_token_count).
        self._system_kv_capacity = max(
            1, int(os.environ.get("VLLM_MLX_SYSTEM_KV_SLOTS", "4"))
        )
        self._system_kv_cache: "OrderedDict[str, tuple[list, int]]" = OrderedDict()
        # Cache-effectiveness counters. Incremented only from inside the
        # serialized worker (single writer) so plain ``+=`` is safe; reads
        # from ``get_stats`` may be slightly stale, which is fine for
        # metrics.
        self._system_kv_cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        }
        # True only when the model's prompt cache can be snapshotted and
        # restored for the manual system-prefix cache branch. Plain KV caches
        # and hybrid ``ArraysCache`` entries are safe when their state
        # containers are copied at snapshot/restore boundaries. Sliding-window
        # cache classes such as ``RotatingKVCache`` remain disabled because
        # their extra cursor metadata is not captured by ``.state`` alone.
        self._supports_system_kv_cache: bool = False

    @staticmethod
    def _clone_cache_state(value: Any) -> Any:
        """Copy cache state containers without duplicating immutable MLX arrays."""
        if isinstance(value, tuple):
            return tuple(SimpleEngine._clone_cache_state(v) for v in value)
        if isinstance(value, list):
            return [SimpleEngine._clone_cache_state(v) for v in value]
        return value

    @classmethod
    def _snapshot_prompt_cache(cls, prompt_cache: list[Any]) -> list[Any]:
        """Capture cache states without aliasing mutable state containers."""
        return [cls._clone_cache_state(c.state) for c in prompt_cache]

    @classmethod
    def _restore_prompt_cache(
        cls, prompt_cache: list[Any], snapshot: list[Any]
    ) -> None:
        """Restore cache states without letting decode mutate the saved snapshot."""
        for i, saved_state in enumerate(snapshot):
            prompt_cache[i].state = cls._clone_cache_state(saved_state)

    @staticmethod
    def _iter_cache_state_arrays(value: Any):
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from SimpleEngine._iter_cache_state_arrays(item)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            yield value

    @classmethod
    def _eval_cache_snapshot(cls, snapshot: list[Any]) -> None:
        arrays = list(cls._iter_cache_state_arrays(snapshot))
        if arrays:
            mx.eval(arrays)

    @staticmethod
    def _cache_class_is_system_snapshot_safe(cache_entry: Any) -> bool:
        try:
            from mlx_lm.models.cache import ArraysCache, KVCache

            return isinstance(cache_entry, (KVCache, ArraysCache))
        except Exception:
            cache_type = type(cache_entry).__name__
            return cache_type in {"KVCache", "ArraysCache"}

    @classmethod
    def _probe_system_kv_cache_support(cls, model: Any, route: str) -> bool:
        try:
            from mlx_lm.models.cache import make_prompt_cache

            probe_cache = make_prompt_cache(model)
            supported = bool(probe_cache) and all(
                cls._cache_class_is_system_snapshot_safe(c) for c in probe_cache
            )
            if not supported:
                cache_types = sorted({type(c).__name__ for c in probe_cache})
                logger.info(
                    "System KV cache snapshot disabled (%s): model returned "
                    "unsupported cache entries (%s); requests will use the "
                    "uncached path",
                    route,
                    cache_types,
                )
            return supported
        except Exception as e:
            logger.debug(
                "System KV cache support probe failed (%s, %s); "
                "disabling snapshot path",
                route,
                e,
            )
            return False

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
        if not self._loaded or self._model is None:
            return None
        if self._is_mllm:
            return getattr(self._model, "processor", None)
        return self._model.tokenizer

    def _generation_lock_holder_summary(self) -> str:
        if not self._active_requests:
            return "none"

        holders = []
        now = time.time()
        for request_id, info in self._active_requests.items():
            elapsed_s = info.get("elapsed_s")
            started_at = info.get("started_at")
            if started_at is not None:
                elapsed_s = round(now - started_at, 1)
            kind = info.get("kind", "unknown")
            status = info.get("status", "unknown")
            holders.append(
                f"{request_id}:{status}:{kind}:"
                f"prompt={info.get('prompt_tokens', 0)}:"
                f"completion={info.get('completion_tokens', 0)}:"
                f"elapsed_s={elapsed_s if elapsed_s is not None else 'unknown'}"
            )
        return ",".join(holders)

    @asynccontextmanager
    async def _acquire_generation_slot(self, request_id: str):
        """Admission control for SimpleEngine's serialized MLX route."""
        if (
            self._generation_lock_admission == "fail_fast"
            and self._generation_lock.locked()
        ):
            self._generation_busy_rejections += 1
            raise EngineBusy(
                "SimpleEngine serialized route is busy; "
                f"request_id={request_id}; "
                f"active={self._generation_lock_holder_summary()}; "
                f"waiters={self._generation_waiters}; "
                "retry later"
            )

        self._generation_waiters += 1
        acquired = False
        try:
            async with self._generation_lock:
                acquired = True
                self._generation_waiters -= 1
                yield
        finally:
            if not acquired and self._generation_waiters > 0:
                self._generation_waiters -= 1

    def prepare_for_start(self) -> None:
        """Load the backing model off the serving event loop."""
        if self._model is not None:
            return

        if self._is_mllm:
            from ..models.mllm import MLXMultimodalLM

            self._model = MLXMultimodalLM(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                enable_cache=self._enable_cache,
                max_kv_size=self._max_kv_size,
                draft_model=self._mllm_draft_model_path,
                draft_kind=self._mllm_draft_kind,
                draft_block_size=self._mllm_draft_block_size,
            )
        else:
            from ..models.llm import MLXLanguageModel

            self._model = MLXLanguageModel(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                mtp=self._mtp,
                mtp_num_draft_tokens=self._mtp_num_draft_tokens,
            )

        self._model.load()

    def _uses_default_prepare_for_start(self) -> bool:
        """Return True when prepare_for_start is the class implementation."""
        method = getattr(self.prepare_for_start, "__func__", None)
        return method is SimpleEngine.prepare_for_start

    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        if self._loaded:
            return
        try:
            if self._model is None:
                if self._uses_default_prepare_for_start():
                    # MLX generation streams are thread-local. Keep model load on
                    # the event-loop thread so default LLM stream_generate() runs
                    # on the same thread that owns model-associated streams.
                    self.prepare_for_start()
                else:
                    # Test doubles and custom overrides may block; preserve the
                    # cancellation-safe threaded startup helper for those cases.
                    await run_blocking_startup_work(self.prepare_for_start)
            self._loaded = True

            if self._mtp and self._mtp_num_draft_tokens != 1:
                logger.warning(
                    "Native mlx_lm MTP currently ignores num_draft_tokens=%d; "
                    "effective speculative draft depth remains 1",
                    self._mtp_num_draft_tokens,
                )

            # Probe whether this model's prompt cache is snapshot-safe for the
            # stream_chat system-prefix cache branch. This is also refreshed
            # below for MLLM text routing after the parallel TextModel exists.
            if not self._is_mllm and self._model is not None:
                backing_model = getattr(self._model, "model", self._model)
                self._supports_system_kv_cache = self._probe_system_kv_cache_support(
                    backing_model,
                    "stream_chat",
                )

            # Build parallel mlx_lm TextModel for text-only routing.
            # Even when MTP is disabled, text-only requests should not be trapped
            # on the slower mlx_vlm multimodal path.
            if self._is_mllm and self._should_route_text_through_text_model():
                try:
                    from ..text_model_from_vlm import build_text_model

                    self._text_model = build_text_model(
                        self._model.model, self._model_name
                    )

                    if self._text_model is not None:
                        self._text_tokenizer = self._model.get_tokenizer()
                        self._supports_system_kv_cache = (
                            self._probe_system_kv_cache_support(
                                self._text_model,
                                "mllm_text",
                            )
                        )

                        # Apply Qwen3.5 eos_token fix (matches MLXLanguageModel.load)
                        if "qwen3" in self._model_name.lower():
                            self._text_tokenizer.eos_token = "<|im_end|>"
                            self._text_tokenizer.eos_token_id = (
                                self._text_tokenizer.convert_tokens_to_ids("<|im_end|>")
                            )

                        # Probe the derived TextModel's prompt cache for snapshot-safety
                        # (same gate stream_chat uses for the pure-LLM path).
                        # _stream_generate_text only enters the system-KV cache branch
                        # when this flag is True, so sliding-window text models won't
                        # desynchronize on restore.
                        #
                        # Probe args must match the runtime constructor in
                        # _stream_generate_text (max_kv_size=self._max_kv_size or None).
                        # Under bounded-KV serving (max_kv_size > 0) make_prompt_cache
                        # returns RotatingKVCache for models without a custom
                        # make_cache; probing with default args would mis-classify that
                        # path as snapshot-safe.
                        try:
                            from mlx_lm.models.cache import KVCache, make_prompt_cache

                            probe_cache = make_prompt_cache(
                                self._text_model, max_kv_size=self._max_kv_size or None
                            )
                            self._supports_system_kv_cache = bool(probe_cache) and all(
                                isinstance(c, KVCache) for c in probe_cache
                            )
                            if not self._supports_system_kv_cache:
                                cache_types = sorted(
                                    {type(c).__name__ for c in probe_cache}
                                )
                                logger.info(
                                    "System KV cache snapshot disabled for MLLM "
                                    "text routing: TextModel returned non-KVCache "
                                    "entries (%s); _stream_generate_text will use "
                                    "the uncached path",
                                    cache_types,
                                )
                        except Exception as e:
                            logger.debug(
                                "MLLM TextModel KV cache support probe failed "
                                "(%s); disabling snapshot path",
                                e,
                            )
                            self._supports_system_kv_cache = False

                        has_mtp = (
                            hasattr(self._text_model, "mtp")
                            and self._text_model.mtp is not None
                        )
                        logger.info(
                            "MLLM text routing: text-only -> mlx_lm TextModel "
                            "(MTP=%s), media -> mlx_vlm",
                            has_mtp and self._mtp,
                        )
                    else:
                        self._text_model = None
                        self._text_tokenizer = None

                except Exception as e:
                    logger.error("MLLM text routing setup failed: %s", e)
                    self._text_model = None
                    self._text_tokenizer = None

            # Load SpecPrefill draft model (small model for importance scoring)
            if self._specprefill_enabled and self._specprefill_draft_model_path:
                try:
                    from mlx_lm import load as mlx_lm_load

                    self._draft_model, _ = mlx_lm_load(
                        self._specprefill_draft_model_path
                    )
                    logger.info(
                        "SpecPrefill: draft model loaded (%s), threshold=%d, keep=%.0f%%",
                        self._specprefill_draft_model_path,
                        self._specprefill_threshold,
                        self._specprefill_keep_pct * 100,
                    )
                except Exception as e:
                    logger.error("SpecPrefill: draft model load failed: %s", e)
                    self._draft_model = None

            # Warn if MTP is enabled without continuous-batching and text routing not available
            if self._mtp and (not self._is_mllm or self._text_model is None):
                logger.warning(
                    "[MTP] --enable-mtp without --continuous-batching: "
                    "speculative decoding via draft tokens will not be active. "
                    "For full MTP support, use: --enable-mtp --continuous-batching"
                )

            mtp_info = ""
            if self._mtp:
                mtp_info = (
                    f", MTP={self._mtp}(configured={self._mtp_num_draft_tokens}, "
                    "effective=1)"
                )
            routing = ", routing=per-request" if self._text_model is not None else ""
            specprefill_info = (
                ", SpecPrefill=active" if self._draft_model is not None else ""
            )
            logger.info(
                f"SimpleEngine loaded: {self._model_name} "
                f"(MLLM={self._is_mllm}{mtp_info}{routing}{specprefill_info})"
            )
        except asyncio.CancelledError:
            await cleanup_startup_cancellation(self.stop)
            raise

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        self._model = None
        self._text_model = None
        self._text_tokenizer = None
        self._draft_model = None
        self._loaded = False
        self._system_kv_cache.clear()
        for k in self._system_kv_cache_stats:
            self._system_kv_cache_stats[k] = 0
        self._supports_system_kv_cache = False
        logger.info("SimpleEngine stopped")

    def _should_route_text_through_text_model(
        self, *, mllm_draft_requested: bool = False
    ) -> bool:
        """Return whether text-only MLLM requests may use mlx_lm TextModel."""
        return not (mllm_draft_requested and self._mllm_draft_model_path is not None)

    async def _run_blocking_serialized(
        self,
        func,
        /,
        *args,
        request_id: str | None = None,
        on_cancel=None,
        **kwargs,
    ):
        """Run a blocking MLX operation under the generation lock.

        Cancellation must not release the async lock before the worker thread
        finishes, or a follow-up request can enter MLX/Metal concurrently and
        corrupt the command-buffer state.
        """
        request_id = request_id or f"simple-{id(func):x}"
        async with self._acquire_generation_slot(request_id):
            started_at = time.time()
            self._active_requests[request_id] = {
                "request_id": request_id,
                "status": "running",
                "kind": "blocking_serialized",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_s": 0.0,
                "started_at": started_at,
            }

            def run_bound():
                _bind_worker_generation_streams()
                return func(*args, **kwargs)

            task = asyncio.create_task(asyncio.to_thread(run_bound))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if on_cancel is not None:
                    try:
                        on_cancel()
                    except Exception:
                        logger.debug(
                            "Blocking worker cancellation callback failed",
                            exc_info=True,
                        )
                try:
                    await task
                except BaseException:
                    pass
                raise
            finally:
                self._active_requests.pop(request_id, None)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Thin accumulator over stream_generate(). stream_generate() is the
        only code path that consumes per-request SpecPrefill overrides
        (`specprefill`, `specprefill_keep_pct`) and routes through
        _stream_generate_specprefill() when engaged. The prior direct
        self._model.generate() path silently dropped those overrides for
        non-streaming /v1/completions callers, so extra_body.specprefill
        was advertised by the server but had no effect on this route.

        By iterating stream_generate() and returning the last
        GenerationOutput, non-streaming clients get the same SpecPrefill
        engagement, accurate prompt_tokens reporting, and per-request
        override support as streaming clients.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional parameters forwarded to stream_generate,
                including per-request `specprefill` / `specprefill_keep_pct`

        Returns:
            GenerationOutput with complete text
        """
        if not self._loaded:
            await self.start()

        last_output: GenerationOutput | None = None
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs,
        ):
            last_output = output

        if last_output is None:
            return GenerationOutput(text="", finish_reason="stop")

        text = clean_output_text(last_output.text)
        return GenerationOutput(
            text=text,
            tokens=list(last_output.tokens),
            prompt_tokens=last_output.prompt_tokens,
            completion_tokens=last_output.completion_tokens,
            finish_reason=last_output.finish_reason,
            finished=True,
            matched_stop=getattr(last_output, "matched_stop", None),
        )

    async def _track_request_stream(
        self,
        source_gen: AsyncIterator[GenerationOutput],
        *,
        max_tokens: int = 0,
    ) -> AsyncIterator[GenerationOutput]:
        """Yield-through wrapper that records per-request live state and
        final ``prompt_tokens``/``completion_tokens`` counters.

        Mirrors the fields BatchedEngine emits per running request
        (``request_id``, ``phase``, ``elapsed_s``, ``ttft_s``,
        ``tokens_per_second``, ``progress``, ...) so dashboards built
        against ``/v1/status`` show individual in-flight requests for
        SimpleEngine-backed services as well (Gemma 4 31B + MTP, etc.).

        Re-entrant calls (e.g. the cache-fallback path inside
        ``_stream_chat_impl`` that delegates to ``self.stream_generate``)
        are detected via the ``_in_tracker`` context variable and pass
        through without a second tracking entry, so each external
        request is counted exactly once.

        Note: we deliberately use ``set(True)``/``set(False)`` rather
        than ``set(token)``/``reset(token)``. FastAPI/uvicorn finalize
        streaming generators from a different async context than the
        one that created them; ``ContextVar.reset(token)`` raises
        ``ValueError`` in that case ("Token was created in a different
        Context"), which surfaces as a terminal-frame streaming error.
        ``set(False)`` works in any context and the contextvar is only
        consumed inside this method, so there is no value to preserve.
        """
        if _in_tracker.get():
            async for output in source_gen:
                yield output
            return
        _in_tracker.set(True)
        request_id = str(uuid.uuid4())
        start = time.time()
        ttft_s: float | None = None
        last_p = 0
        last_c = 0
        entry: dict[str, Any] = {
            "request_id": request_id,
            "status": "running",
            "phase": "prefill",
            "elapsed_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_tokens": max_tokens,
            "progress": 0.0,
            "tokens_per_second": 0.0,
            "ttft_s": None,
            "cache_hit_type": None,
            "cached_tokens": 0,
        }
        self._active_requests[request_id] = entry
        self._num_running += 1
        try:
            async for output in source_gen:
                now = time.time()
                if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                    last_p = output.prompt_tokens
                    entry["prompt_tokens"] = last_p
                if hasattr(output, "completion_tokens") and output.completion_tokens:
                    if ttft_s is None:
                        ttft_s = now - start
                        entry["ttft_s"] = round(ttft_s, 3)
                        entry["phase"] = "generation"
                    last_c = output.completion_tokens
                    entry["completion_tokens"] = last_c
                entry["elapsed_s"] = round(now - start, 2)
                if max_tokens > 0:
                    entry["progress"] = round(min(1.0, last_c / max_tokens), 3)
                if ttft_s is not None and last_c > 0:
                    gen_elapsed = max(1e-3, (now - start) - ttft_s)
                    entry["tokens_per_second"] = round(last_c / gen_elapsed, 1)
                yield output
        finally:
            self._active_requests.pop(request_id, None)
            self._num_running = max(0, self._num_running - 1)
            if last_c > 0:
                duration = time.time() - start
                self._total_requests_processed += 1
                self._total_prompt_tokens += last_p
                self._total_completion_tokens += last_c
                self._recent_completions.append((last_c, duration))
            _in_tracker.set(False)

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Public stream-generate wrapper with request stats tracking."""
        async for output in self._track_request_stream(
            self._stream_generate_impl(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ),
            max_tokens=max_tokens,
        ):
            yield output

    async def _stream_generate_impl(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct_override = kwargs.pop("specprefill_keep_pct", None)
        specprefill_backbone_pct_override = kwargs.pop("specprefill_backbone_pct", None)
        request_id = str(kwargs.pop("request_id", "") or f"simple-{id(prompt):x}")

        # SpecPrefill for non-MLLM models (MLLM+MTP handles it in _stream_generate_text)
        if not self._is_mllm and self._draft_model is not None:
            use_specprefill = True
            if specprefill_override is False:
                use_specprefill = False

            if use_specprefill:
                tokenizer = self._model.tokenizer
                add_special = tokenizer.bos_token is None or not prompt.startswith(
                    tokenizer.bos_token
                )
                tokens_list = tokenizer.encode(prompt, add_special_tokens=add_special)
                n_tokens = len(tokens_list)

                # Threshold check (skip when force-enabled via per-request override)
                if (
                    specprefill_override is not True
                    and n_tokens <= self._specprefill_threshold
                ):
                    use_specprefill = False

                # Upper bound: cap to avoid draft model OOM
                _SPECPREFILL_MAX_TOKENS = 65536
                if use_specprefill and n_tokens > _SPECPREFILL_MAX_TOKENS:
                    logger.warning(
                        "SpecPrefill: prompt %d tokens exceeds max %d, "
                        "falling back to normal path",
                        n_tokens,
                        _SPECPREFILL_MAX_TOKENS,
                    )
                    use_specprefill = False

                if use_specprefill:
                    async for output in self._stream_generate_specprefill(
                        prompt,
                        tokens_list,
                        max_tokens,
                        temperature,
                        top_p,
                        stop=stop,
                        specprefill_keep_pct=specprefill_keep_pct_override,
                        specprefill_backbone_pct=specprefill_backbone_pct_override,
                        **kwargs,
                    ):
                        yield output
                    return

        async with self._acquire_generation_slot(request_id):
            started_at = time.time()
            self._active_requests[request_id] = {
                "request_id": request_id,
                "status": "running",
                "kind": "stream_generate",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_s": 0.0,
                "started_at": started_at,
            }
            # Non-stream chat runs in a worker thread and rebinds generation
            # streams there. Rebind again on the current thread before
            # stream_generate so nonstream->stream mode switches remain valid.
            _bind_worker_generation_streams()

            try:
                accumulated_text = ""
                prompt_tokens = 0
                completion_tokens = 0
                finished = False

                for chunk in self._model.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs,
                ):
                    prompt_tokens = (
                        chunk.prompt_tokens
                        if hasattr(chunk, "prompt_tokens") and chunk.prompt_tokens
                        else prompt_tokens
                    )
                    # Synthetic stop-window drain chunks carry text but
                    # represent zero new samples; don't double-count
                    # them (codex r8 NIT).
                    if not getattr(chunk, "is_drain", False):
                        completion_tokens += 1
                    if request_id in self._active_requests:
                        self._active_requests[request_id].update(
                            {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "elapsed_s": round(time.time() - started_at, 1),
                            }
                        )
                    new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                    accumulated_text += new_text

                    finished = (
                        getattr(chunk, "finished", False)
                        or completion_tokens >= max_tokens
                    )
                    finish_reason = None
                    if finished:
                        finish_reason = getattr(chunk, "finish_reason", "stop")

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=getattr(chunk, "matched_stop", None),
                    )

                    if finished:
                        break

                if not finished:
                    if prompt_tokens == 0:
                        prompt_tokens = len(self._model.tokenizer.encode(prompt))
                    if request_id in self._active_requests:
                        self._active_requests[request_id].update(
                            {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "elapsed_s": round(time.time() - started_at, 1),
                            }
                        )
                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text="",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        finished=True,
                        finish_reason=None,
                    )
            finally:
                self._active_requests.pop(request_id, None)

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
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        if not self._loaded:
            await self.start()

        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})

        async def aggregate_stream_chat() -> GenerationOutput:
            final_output = GenerationOutput(text="")
            async for output in self.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            ):
                final_output = output
            text = clean_output_text(final_output.text)
            return GenerationOutput(
                text=text,
                tokens=list(final_output.tokens),
                prompt_tokens=final_output.prompt_tokens,
                completion_tokens=final_output.completion_tokens,
                finish_reason=final_output.finish_reason,
                mtp_drafts=final_output.mtp_drafts,
                mtp_accepted=final_output.mtp_accepted,
                matched_stop=getattr(final_output, "matched_stop", None),
            )

        # mlx-lm non-streaming chat with tools can stall indefinitely on some
        # local models, while the streaming path completes normally. Reuse the
        # streaming implementation and aggregate its final state so both chat
        # APIs share the same tool-capable execution path.
        if tools and not self._is_mllm:
            return await aggregate_stream_chat()

        # Request-local logits processors (response_format / constrained JSON)
        # need token-boundary progress and cancellation.  The blocking
        # model.chat() call below only returns after the whole completion, so a
        # slow constrained decode can look like a no-progress non-stream wedge
        # and hold the serialized generation lock until max_tokens/timeout.
        if kwargs.get("logits_processors") and not self._is_mllm:
            return await aggregate_stream_chat()

        # Text-only requests on MLLM models should always aggregate the
        # streaming path for non-streaming chat. This keeps one execution seam
        # and avoids mlx_vlm non-stream thread/stream ownership mismatches.
        if self._is_mllm and not has_media_content(messages):
            return await aggregate_stream_chat()

        # Convert tools for template if provided
        template_tools = convert_tools_for_template(tools) if tools else None

        if self._is_mllm:
            if chat_template_kwargs:
                kwargs["chat_template_kwargs"] = chat_template_kwargs
            output = await self._run_blocking_serialized(
                self._model.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=template_tools,
                **kwargs,
            )
            text = clean_output_text(output.text)
            return GenerationOutput(
                text=text,
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                finish_reason=output.finish_reason,
                mtp_drafts=getattr(output, "mtp_drafts", 0),
                mtp_accepted=getattr(output, "mtp_accepted", 0),
            )
        else:
            output = await self._run_blocking_serialized(
                self._model.chat,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=template_tools,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            )
            text = clean_output_text(output.text)
            # Preserve upstream prompt accounting while routing the blocking
            # chat call through the cancellation-safe serialized runner.
            tokenizer = self._model.tokenizer
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
            }
            if template_tools:
                template_kwargs["tools"] = template_tools
            prompt_ids = tokenizer.apply_chat_template(messages, **template_kwargs)
            prompt_token_count = len(prompt_ids)
            return GenerationOutput(
                text=text,
                tokens=output.tokens,
                prompt_tokens=prompt_token_count,
                completion_tokens=len(output.tokens),
                finish_reason=output.finish_reason,
                matched_stop=getattr(output, "matched_stop", None),
            )

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
        """Public stream-chat wrapper with request stats tracking."""
        async for output in self._track_request_stream(
            self._stream_chat_impl(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                images=images,
                videos=videos,
                **kwargs,
            ),
            max_tokens=max_tokens,
        ):
            yield output

    async def _stream_chat_impl(
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
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Optional tool definitions
            images: Optional image URLs/paths
            videos: Optional video URLs/paths
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        if not self._loaded:
            await self.start()

        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        mllm_draft_requested = bool(kwargs.pop("mllm_draft", False))

        # Convert tools for template
        template_tools = convert_tools_for_template(tools) if tools else None

        # Per-request routing: text-only through mlx_lm TextModel
        if (
            self._is_mllm
            and self._text_model is not None
            and self._should_route_text_through_text_model(
                mllm_draft_requested=mllm_draft_requested
            )
            and not has_media_content(messages)
        ):
            has_mtp = (
                hasattr(self._text_model, "mtp") and self._text_model.mtp is not None
            )
            logger.info("Text-only request → LLM path (MTP=%s)", has_mtp and self._mtp)
            if chat_template_kwargs:
                kwargs["chat_template_kwargs"] = chat_template_kwargs
            async for chunk in self._stream_generate_text(
                messages,
                max_tokens,
                temperature,
                top_p,
                tools=template_tools,
                **kwargs,
            ):
                yield chunk
            return

        def mllm_call_kwargs() -> dict:
            local_kwargs = dict(kwargs)
            if chat_template_kwargs:
                local_kwargs["chat_template_kwargs"] = chat_template_kwargs
            if mllm_draft_requested:
                local_kwargs["mllm_draft"] = True
            return local_kwargs

        # Build prompt using tokenizer
        if self._is_mllm:
            if self._text_model is not None:
                logger.info("Media request → MLLM path")
            # For MLLM, use stream_chat which yields tokens incrementally.
            # Must hold _generation_lock to prevent concurrent Metal access
            # (e.g. OpenCode sends title + main request simultaneously).
            accumulated_text = ""
            token_count = 0

            # Stop-sequence enforcement gate for both MLLM paths below
            # (issue #469, codex r8).  ``self._model.stream_chat`` doesn't
            # enforce ``stop`` itself, so we hold-back-buffer and surface
            # ``matched_stop`` here exactly like the LLM/spec-prefill
            # paths do above.
            _mllm_stop = kwargs.get("stop") or []
            _mllm_stop_gate = _StopStreamGate(_mllm_stop)

            # Text-only fallback when no TextModel exists: keep execution on the
            # current thread. Routing through to_thread can break mlx_vlm stream
            # ownership on some models (Stream(gpu, N) mismatch).
            if self._text_model is None and not has_media_content(messages):
                local_kwargs = mllm_call_kwargs()

                async with self._generation_lock:
                    _bind_worker_generation_streams()
                    for chunk in self._model.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=template_tools,
                        **local_kwargs,
                    ):
                        token_count += 1
                        raw_new_text = (
                            chunk.text if hasattr(chunk, "text") else str(chunk)
                        )
                        new_text, stop_hit, matched_here = _mllm_stop_gate.feed(
                            raw_new_text
                        )
                        if _mllm_stop_gate.enabled:
                            accumulated_text = _mllm_stop_gate.buf
                        else:
                            accumulated_text += raw_new_text

                        upstream_finish = chunk.finish_reason
                        finished = stop_hit or upstream_finish is not None
                        if stop_hit:
                            finish_reason = "stop"
                        else:
                            finish_reason = upstream_finish if finished else None

                        # On non-stop-match termination, drain the safety
                        # window so the trailing chars don't disappear.
                        if finished and not stop_hit and _mllm_stop_gate.enabled:
                            tail, _ = _mllm_stop_gate.flush()
                            if tail:
                                new_text = (new_text or "") + tail
                                accumulated_text = _mllm_stop_gate.buf

                        yield GenerationOutput(
                            text=accumulated_text,
                            new_text=new_text,
                            prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                            completion_tokens=token_count,
                            finished=finished,
                            finish_reason=finish_reason,
                            mtp_drafts=getattr(chunk, "mtp_drafts", 0),
                            mtp_accepted=getattr(chunk, "mtp_accepted", 0),
                            matched_stop=matched_here,
                        )

                        if finished:
                            break
                    else:
                        # Upstream stream_chat exhausted without setting
                        # ``finish_reason`` AND we never matched/broke.
                        # Drain the gate so safety-window bytes ship.
                        if _mllm_stop_gate.enabled:
                            tail, _ = _mllm_stop_gate.flush()
                            if tail:
                                accumulated_text = _mllm_stop_gate.buf
                                yield GenerationOutput(
                                    text=accumulated_text,
                                    new_text=tail,
                                    prompt_tokens=0,
                                    completion_tokens=token_count,
                                    finished=True,
                                    finish_reason="stop",
                                )
                return

            # Run stream_chat in thread pool since it's synchronous
            def run_stream():
                local_kwargs = mllm_call_kwargs()
                return list(
                    self._model.stream_chat(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        tools=template_tools,
                        **local_kwargs,
                    )
                )

            chunks = await self._run_blocking_serialized(run_stream)

            broke_on_stop = False
            for chunk in chunks:
                token_count += 1
                raw_new_text = (
                    chunk.text if hasattr(chunk, "text") else str(chunk)
                )
                new_text, stop_hit, matched_here = _mllm_stop_gate.feed(
                    raw_new_text
                )
                if _mllm_stop_gate.enabled:
                    accumulated_text = _mllm_stop_gate.buf
                else:
                    accumulated_text += raw_new_text

                upstream_finish = chunk.finish_reason
                finished = stop_hit or upstream_finish is not None
                if stop_hit:
                    finish_reason = "stop"
                    broke_on_stop = True
                else:
                    finish_reason = upstream_finish if finished else None

                if finished and not stop_hit and _mllm_stop_gate.enabled:
                    tail, _ = _mllm_stop_gate.flush()
                    if tail:
                        new_text = (new_text or "") + tail
                        accumulated_text = _mllm_stop_gate.buf

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                    completion_tokens=token_count,
                    finished=finished,
                    finish_reason=finish_reason,
                    mtp_drafts=getattr(chunk, "mtp_drafts", 0),
                    mtp_accepted=getattr(chunk, "mtp_accepted", 0),
                    matched_stop=matched_here,
                )

                if finished:
                    break

            # If the buffered chunk-list exhausted without any chunk
            # marking finished AND no stop match landed, drain the
            # safety window so trailing chars don't disappear.
            if (
                not broke_on_stop
                and chunks
                and chunks[-1].finish_reason is None
                and _mllm_stop_gate.enabled
            ):
                tail, _ = _mllm_stop_gate.flush()
                if tail:
                    accumulated_text = _mllm_stop_gate.buf
                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=tail,
                        prompt_tokens=0,
                        completion_tokens=token_count,
                        finished=True,
                        finish_reason="stop",
                    )
            return

        # For LLM, apply chat template and stream
        tokenizer = self._model.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            # Per-request enable_thinking override; default: True unless coder model.
            enable_thinking = kwargs.pop("enable_thinking", None)
            if enable_thinking is None:
                enable_thinking = "coder" not in self._model_name.lower()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
            }
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)
            if template_tools:
                template_kwargs["tools"] = template_tools
            safe_messages = normalize_messages_for_chat_template(messages)

            try:
                prompt = tokenizer.apply_chat_template(safe_messages, **template_kwargs)
            except TypeError:
                # Some templates don't support all kwargs
                for key in ["tools", "enable_thinking", *chat_template_kwargs.keys()]:
                    if key in template_kwargs:
                        del template_kwargs[key]
                prompt = tokenizer.apply_chat_template(safe_messages, **template_kwargs)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"

        # --- System-prompt KV caching on the pure-LLM stream_chat path ---
        # Mirrors the cache in _stream_generate_text. Locates the system prefix
        # via probe-divergence (cf. prompt_warmup._build_strict_prefix_string):
        # render the template with two different user contents and take the
        # shared prefix. Works across Qwen/ChatML, Llama, Gemma, and any other
        # chat format -- no per-model marker list. Falls back to the original
        # uncached self.stream_generate() if the system prefix can't be
        # isolated or any step of the cache-aware path raises.
        cache_hit = False
        suffix_tokens = None
        system_tokens = None
        system_token_count = 0
        full_token_count = 0
        system_hash = None
        kv_cache_eligible = False
        # Snapshot reference captured at gate time so a concurrent MISS that
        # mutates ``self._system_kv_cache`` between the gate and the restore
        # (which runs later inside ``_run_blocking_serialized``) can't
        # desynchronize the restored KV from the hash that decided HIT.
        hit_snapshot: Any = None

        # Decode-control gate.
        # The cache branch below drives ``mlx_lm.stream_generate`` directly with only
        # ``prompt``, ``max_tokens``, ``sampler`` (built from temperature+top_p), and
        # ``prompt_cache``.
        # The uncached fallback threads ``**kwargs`` through ``self.stream_generate``,
        # which preserves ``stop``, request-local ``logits_processors`` (parser stop
        # tokens and JSON-constrained decoding attached by server.py per request), and
        # the ``top_k`` / ``min_p`` / ``presence_penalty`` / ``repetition_penalty``
        # sampling controls.
        # If the cache branch ran with any of those active, cache-eligible and uncached
        # requests would silently decode under different constraints.
        # Skip the cache branch in that case so both paths share identical decode
        # semantics.
        # server.py always supplies the no-op defaults (``top_k=0``, ``min_p=0.0``,
        # ``presence_penalty=0.0``, ``repetition_penalty=1.0``); compare against those
        # rather than ``key in kwargs`` so the common path still hits the cache.
        cache_blocking_controls: list[str] = []
        if kwargs.get("stop"):
            cache_blocking_controls.append("stop")
        if kwargs.get("logits_processors"):
            cache_blocking_controls.append("logits_processors")
        if (kwargs.get("top_k") or 0) > 0:
            cache_blocking_controls.append("top_k")
        if (kwargs.get("min_p") or 0.0) > 0.0:
            cache_blocking_controls.append("min_p")
        if (kwargs.get("presence_penalty") or 0.0) != 0.0:
            cache_blocking_controls.append("presence_penalty")
        if (kwargs.get("repetition_penalty") or 1.0) != 1.0:
            cache_blocking_controls.append("repetition_penalty")

        # Engine-feature gate.
        # The cache branch also bypasses engine-level features that
        # ``self.stream_generate`` (and the ``MLXLanguageModel.stream_generate``
        # wrapper underneath it) layer on top of ``mlx_lm.stream_generate``.
        # Same correctness reasoning as the decode-control gate: cache-eligible
        # and uncached requests must decode under identical engine semantics, so
        # skip the cache branch when any of these are active.
        # Specifically:
        #   - ``self._mtp`` injects ``mtp=True`` and ``num_draft_tokens`` into
        #     the mlx-lm call (see ``MLXLanguageModel.stream_generate``).
        #   - A loaded SpecPrefill draft model (``self._draft_model is not None``,
        #     set when ``specprefill_enabled`` + ``specprefill_draft_model`` are
        #     configured at engine init) routes large prompts through
        #     ``_stream_generate_specprefill`` instead of the plain stream path.
        #   - A per-request ``specprefill`` override from ``extra_body`` (popped
        #     by the wrapper from ``kwargs``) can force or suppress SpecPrefill
        #     for a single request.
        #     ``specprefill=False`` is a meaningful suppression signal — gate on
        #     ``is not None`` rather than truthiness so the wrapper sees it.
        #   - ``self._max_kv_size`` (when > 0) caps the prompt cache; the cache
        #     branch builds its cache with ``make_prompt_cache(model)`` and has
        #     no equivalent bound.
        if self._mtp:
            cache_blocking_controls.append("mtp")
        if self._draft_model is not None:
            cache_blocking_controls.append("specprefill_loaded")
        if kwargs.get("specprefill") is not None:
            cache_blocking_controls.append("specprefill_request_override")
        if (self._max_kv_size or 0) > 0:
            cache_blocking_controls.append("max_kv_size")
        # Sliding-window models build their prompt cache from RotatingKVCache
        # entries whose ``.state`` aliases buffers that ``update_and_fetch``
        # mutates in place. Snapshot capture would corrupt the cached prefix
        # on the next decode. Probed once at start; ``False`` if the model
        # exposes any non-KVCache entries or the probe failed.
        if not self._supports_system_kv_cache:
            cache_blocking_controls.append("non_kv_cache_class")

        if cache_blocking_controls:
            logger.info(
                "System KV cache SKIP (stream_chat): request or engine has "
                "controls/features the cache branch cannot honor (%s); using "
                "uncached path",
                cache_blocking_controls,
            )

        # Normalize messages to plain dicts. The public stream_chat signature
        # types messages as list[dict], but internal callers (server.py,
        # tests) sometimes pass Pydantic Message objects directly; those
        # don't expose a dict-style .get() interface.
        def _to_msg_dict(m: Any) -> dict[str, Any]:
            if isinstance(m, dict):
                return m
            if hasattr(m, "model_dump"):
                return m.model_dump()
            if hasattr(m, "dict"):
                return m.dict()
            return {
                "role": getattr(m, "role", None),
                "content": getattr(m, "content", ""),
            }

        messages_for_cache = [_to_msg_dict(m) for m in messages]
        has_system = any(m.get("role") == "system" for m in messages_for_cache)
        if (
            has_system
            and not cache_blocking_controls
            and hasattr(tokenizer, "apply_chat_template")
        ):

            def _with_user(user_content: str) -> list[dict[str, Any]]:
                msgs = [dict(m) for m in messages_for_cache]
                if msgs and msgs[-1].get("role") == "user":
                    msgs[-1] = {**msgs[-1], "content": user_content}
                else:
                    msgs = [*msgs, {"role": "user", "content": user_content}]
                return msgs

            rendered_a: Any = None
            rendered_b: Any = None
            try:
                rendered_a = tokenizer.apply_chat_template(
                    _with_user("Alpha"), **template_kwargs
                )
                rendered_b = tokenizer.apply_chat_template(
                    _with_user("Bravo"), **template_kwargs
                )
            except Exception:
                pass

            if isinstance(rendered_a, str) and isinstance(rendered_b, str):
                boundary = 0
                diverged = False
                for i in range(min(len(rendered_a), len(rendered_b))):
                    if rendered_a[i] != rendered_b[i]:
                        diverged = True
                        break
                    boundary = i + 1

                if diverged and boundary >= 16:
                    system_prefix_text = rendered_a[:boundary]
                    system_hash = hashlib.sha256(
                        system_prefix_text.encode()
                    ).hexdigest()[:16]

                    add_special = tokenizer.bos_token is None or not prompt.startswith(
                        tokenizer.bos_token
                    )
                    full_tokens_list = tokenizer.encode(
                        prompt, add_special_tokens=add_special
                    )
                    system_tokens_list = tokenizer.encode(
                        system_prefix_text, add_special_tokens=add_special
                    )
                    full_token_count = len(full_tokens_list)
                    system_token_count = len(system_tokens_list)

                    if (
                        len(full_tokens_list) > system_token_count
                        and full_tokens_list[:system_token_count] == system_tokens_list
                    ):
                        system_tokens = system_tokens_list
                        suffix_tokens = full_tokens_list[system_token_count:]
                        kv_cache_eligible = True
                        # Read the snapshot reference once. If we promote to
                        # HIT, ``hit_snapshot`` is the exact list the dict
                        # lookup just returned. A later concurrent MISS that
                        # mutates ``self._system_kv_cache`` before our
                        # serialized worker restores it cannot alias what we
                        # captured here — dict.get is atomic under the GIL
                        # and returns a reference to an immutable tuple.
                        candidate = self._system_kv_cache.get(system_hash)
                        if candidate is not None and system_token_count == candidate[1]:
                            cache_hit = True
                            hit_snapshot = candidate[0]
                            logger.info(
                                "System KV cache HIT (stream_chat): reusing %d "
                                "tokens, prefilling %d new (hash=%s)",
                                system_token_count,
                                len(suffix_tokens),
                                system_hash,
                            )
                        else:
                            logger.info(
                                "System KV cache MISS (stream_chat): will "
                                "prefill %d system + %d suffix tokens (hash=%s)",
                                system_token_count,
                                len(suffix_tokens),
                                system_hash,
                            )

        if kv_cache_eligible:
            # Cache-aware path: drive mlx-lm directly with a pre-populated cache.
            # Stream chunks back to the caller via an asyncio.Queue (mirrors
            # _stream_generate_text) so the client sees tokens as they arrive
            # rather than after the full generation finishes.
            loop = asyncio.get_running_loop()
            response_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            abort_event = threading.Event()

            def _emit_response(resp: Any) -> None:
                if abort_event.is_set():
                    return
                loop.call_soon_threadsafe(response_queue.put_nowait, ("resp", resp))

            def _emit_done() -> None:
                loop.call_soon_threadsafe(response_queue.put_nowait, ("done", None))

            def _emit_error(exc: BaseException) -> None:
                loop.call_soon_threadsafe(response_queue.put_nowait, ("error", exc))

            def _run_with_cache() -> None:
                from mlx_lm import stream_generate as mlx_stream_generate
                from mlx_lm.models.cache import make_prompt_cache
                from mlx_lm.sample_utils import make_sampler

                model = self._model.model
                sampler = make_sampler(temp=temperature, top_p=top_p)

                if cache_hit:
                    bc = make_prompt_cache(model)
                    # Restore from the closure-local reference captured at the
                    # gate, never from ``self._system_kv_cache`` directly:
                    # a concurrent MISS could have evicted the entry between
                    # the gate check and this point. Restore clones mutable
                    # state containers so decode cannot mutate the saved LRU
                    # snapshot by reference.
                    self._restore_prompt_cache(bc, hit_snapshot)
                    # Bump LRU position. Safe to mutate here because the
                    # worker is serialized under ``_generation_lock``.
                    if system_hash in self._system_kv_cache:
                        self._system_kv_cache.move_to_end(system_hash)
                    self._system_kv_cache_stats["hits"] += 1
                else:
                    bc = make_prompt_cache(model)
                    sys_arr = mx.array(system_tokens)
                    step = self._prefill_step_size
                    while sys_arr.size > step:
                        model(sys_arr[:step][None], cache=bc)
                        self._eval_cache_snapshot([c.state for c in bc])
                        sys_arr = sys_arr[step:]
                        mx.clear_cache()
                    if sys_arr.size > 0:
                        model(sys_arr[None], cache=bc)
                        self._eval_cache_snapshot([c.state for c in bc])

                    # Free intermediate prefill activations before snapshotting.
                    # Intentionally stricter than the MLLM path, which does not
                    # ``mx.clear_cache()`` between its last prefill chunk and
                    # the snapshot; here we want the snapshot to reflect only
                    # the KV state, not residual activations from prefill.
                    mx.clear_cache()

                    snapshot = self._snapshot_prompt_cache(bc)
                    self._eval_cache_snapshot(snapshot)
                    self._system_kv_cache[system_hash] = (snapshot, system_token_count)
                    self._system_kv_cache.move_to_end(system_hash)
                    evicted_count = 0
                    while len(self._system_kv_cache) > self._system_kv_capacity:
                        evicted_hash, _ = self._system_kv_cache.popitem(last=False)
                        self._system_kv_cache_stats["evictions"] += 1
                        evicted_count += 1
                        logger.info(
                            "System KV cache EVICTED (stream_chat): hash=%s "
                            "(capacity=%d)",
                            evicted_hash,
                            self._system_kv_capacity,
                        )
                    if evicted_count:
                        # Eviction dropped MLX array refs; reclaim Metal heap.
                        # Skip on the common non-eviction path to avoid
                        # flushing the Metal allocator's reuse pool.
                        mx.clear_cache()
                    self._system_kv_cache_stats["misses"] += 1
                    self._system_kv_cache_stats["stores"] += 1
                    try:
                        cache_mb = sum(c.nbytes for c in bc) / 1e6
                    except Exception:
                        cache_mb = -1
                    logger.info(
                        "System KV cache STORED (stream_chat): %d tokens " "(%.1f MB)",
                        system_token_count,
                        cache_mb,
                    )

                prompt_arr = mx.array(suffix_tokens)
                for resp in mlx_stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt_arr,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=bc,
                ):
                    if abort_event.is_set():
                        break
                    _emit_response(resp)

            async def _produce_responses() -> None:
                try:
                    await self._run_blocking_serialized(
                        _run_with_cache,
                        on_cancel=abort_event.set,
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    _emit_error(exc)
                else:
                    _emit_done()

            producer_task = asyncio.create_task(_produce_responses())

            accumulated_text = ""
            token_count = 0
            finished = False
            cache_path_failed_before_first_token = False
            try:
                while True:
                    kind, payload = await response_queue.get()
                    if kind == "done":
                        break
                    if kind == "error":
                        if token_count == 0:
                            logger.warning(
                                "Pure-LLM KV-cache path failed before first "
                                "token (%s); falling back to uncached "
                                "stream_generate",
                                payload,
                            )
                            cache_path_failed_before_first_token = True
                            break
                        # Already streamed partial output; can't cleanly
                        # restart on the uncached path, so surface the error.
                        raise payload
                    resp = payload
                    token_count += 1
                    new_text = resp.text if hasattr(resp, "text") else str(resp)
                    accumulated_text += new_text
                    finish_reason = getattr(resp, "finish_reason", None)
                    finished = finish_reason is not None or token_count >= max_tokens
                    if finish_reason is None and finished:
                        finish_reason = "stop"

                    yield GenerationOutput(
                        text=accumulated_text,
                        new_text=new_text,
                        prompt_tokens=full_token_count,
                        completion_tokens=token_count,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=getattr(resp, "matched_stop", None),
                    )
                    if finished:
                        break
            finally:
                if not producer_task.done():
                    abort_event.set()
                    try:
                        await producer_task
                    except BaseException:
                        pass

            if cache_path_failed_before_first_token:
                # Internal fallback to the public stream_generate. The
                # ``_in_tracker`` context flag prevents double counting
                # in _track_request_stream.
                async for output in self.stream_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                ):
                    yield output
            return

        # Fallback: no system prefix detected -> original uncached path.
        # Re-entrancy guard in _track_request_stream keeps stats single-counted.
        async for output in self.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        ):
            yield output

    async def _stream_generate_specprefill(
        self,
        prompt: str,
        tokens: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
        specprefill_keep_pct: float | None = None,
        specprefill_backbone_pct: float | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """SpecPrefill path for non-MTP models (Nemotron, GPT-OSS, etc).

        Scores token importance with the draft model, sparse-prefills the target
        model, then generates autoregressively. Falls back to normal generation
        on any error.
        """
        from threading import Event

        model = self._model.model
        tokenizer = self._model.tokenizer
        n_tokens = len(tokens)
        cancel_requested = Event()

        def _request_cancel() -> None:
            cancel_requested.set()

        def _cancel_check() -> None:
            if cancel_requested.is_set():
                raise _SpecPrefillCancelled()

        def _run_all():
            try:
                return _run_specprefill()
            except _SpecPrefillCancelled:
                raise
            except Exception as e:
                logger.error("SpecPrefill failed, falling back to normal path: %s", e)
                return _run_normal()

        def _run_specprefill():
            """Score tokens, sparse prefill, generate autoregressively."""
            import time
            from types import SimpleNamespace

            import mlx.core as mx
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.sample_utils import make_sampler

            from ..specprefill import (
                cleanup_rope,
                score_tokens,
                select_chunks,
                sparse_prefill,
            )

            cache = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)

            try:
                # Phase 1: Score with draft model
                t0 = time.monotonic()
                importance = score_tokens(
                    self._draft_model,
                    tokens,
                    prefill_step_size=self._prefill_step_size,
                    cancel_check=_cancel_check,
                )
                t_score = time.monotonic() - t0

                # Phase 2: Select important chunks
                _cancel_check()
                effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
                effective_backbone = (
                    specprefill_backbone_pct
                    if specprefill_backbone_pct is not None
                    else self._specprefill_backbone_pct
                )
                selected = select_chunks(
                    importance,
                    keep_pct=effective_keep,
                    backbone_pct=effective_backbone,
                )
                n_selected = selected.shape[0]

                # Phase 3: Sparse prefill on target model
                t0 = time.monotonic()
                logits = sparse_prefill(
                    model,
                    tokens,
                    selected,
                    cache,
                    step_size=self._prefill_step_size,
                    cancel_check=_cancel_check,
                )
                t_prefill = time.monotonic() - t0

                logger.info(
                    "SpecPrefill: scored %d tokens in %.1fs, "
                    "sparse prefill %d/%d (keep=%.0f%%) in %.1fs",
                    n_tokens,
                    t_score,
                    n_selected,
                    n_tokens,
                    n_selected / n_tokens * 100,
                    t_prefill,
                )

                # Phase 4: Generate via engine's standard pipelined path
                sampler = make_sampler(temp=temperature, top_p=top_p)
                _cancel_check()
                first_token_id = sampler(logits[:, -1, :]).item()
                first_text = tokenizer.decode([first_token_id])
                eos_id = tokenizer.eos_token_id

                results = [
                    SimpleNamespace(
                        text=first_text,
                        finish_reason="stop" if first_token_id == eos_id else None,
                    )
                ]

                if first_token_id != eos_id:
                    for chunk in self._model.stream_generate(
                        prompt=mx.array([first_token_id]),
                        max_tokens=max_tokens - 1,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        prompt_cache=cache,
                    ):
                        _cancel_check()
                        new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                        results.append(
                            SimpleNamespace(
                                text=new_text,
                                finish_reason=getattr(chunk, "finish_reason", None),
                                # Preserve matched_stop from the model
                                # layer so the Anthropic adapter can
                                # report stop_sequence (issue #469).
                                matched_stop=getattr(
                                    chunk, "matched_stop", None
                                ),
                            )
                        )

                return results

            finally:
                cleanup_rope(model)

        def _run_normal():
            """Fallback: normal generation without specprefill."""
            from types import SimpleNamespace

            results = []
            for chunk in self._model.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs,
            ):
                _cancel_check()
                new_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                results.append(
                    SimpleNamespace(
                        text=new_text,
                        finish_reason=getattr(chunk, "finish_reason", None),
                        matched_stop=getattr(chunk, "matched_stop", None),
                    )
                )
            return results

        all_resps = await self._run_blocking_serialized(
            _run_all, on_cancel=_request_cancel
        )

        # Yield results as GenerationOutput.  The seed first-token from
        # the manual sample plus subsequent stream chunks both flow
        # through a single stop-stream gate so a stop string that lands
        # on the seed (or spans seed→continuation) is correctly caught
        # (issue #469).
        accumulated_text = ""
        token_count = 0
        finished = False
        _stop_gate = _StopStreamGate(stop)
        for i, resp in enumerate(all_resps):
            token_count += 1
            raw_new_text = resp.text
            new_text, stop_hit, matched_here = _stop_gate.feed(raw_new_text)
            if _stop_gate.enabled:
                accumulated_text = _stop_gate.buf
            else:
                accumulated_text += raw_new_text

            is_last = i == len(all_resps) - 1
            finished = stop_hit or is_last or token_count >= max_tokens
            if stop_hit:
                finish_reason = "stop"
            else:
                finish_reason = resp.finish_reason or (
                    "stop" if finished else None
                )
            if finished and not stop_hit and _stop_gate.enabled:
                tail, _ = _stop_gate.flush()
                if tail:
                    new_text = (new_text or "") + tail
                    accumulated_text = _stop_gate.buf

            yield GenerationOutput(
                text=accumulated_text,
                new_text=new_text,
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=finished,
                finish_reason=finish_reason,
                matched_stop=matched_here or getattr(resp, "matched_stop", None),
            )

            if finished:
                break

        if not finished:
            # Drain any chars the safety window held back so the client
            # doesn't lose the trailing ``max_stop_len - 1`` bytes on
            # generator-exhaustion / EOS termination (codex r6).
            tail_text = ""
            if _stop_gate.enabled:
                tail, _ = _stop_gate.flush()
                if tail:
                    tail_text = tail
                    accumulated_text = _stop_gate.buf
            yield GenerationOutput(
                text=accumulated_text,
                new_text=tail_text,
                prompt_tokens=n_tokens,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    async def _stream_generate_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """Text-only generation via mlx_lm TextModel.

        Used when text-only MLLM routing is active and the request has no media.
        Runs the full generation in a single thread to maintain Metal safety.

        System prompt KV caching: on the first request, prefills system tokens
        and snapshots backbone KV state. Subsequent requests with the same
        system prompt restore the snapshot and only prefill the suffix tokens.
        """
        import hashlib
        import os

        import mlx.core as mx
        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.models import cache as cache_module
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # Per-request specprefill overrides (from extra_body)
        specprefill_override = kwargs.pop("specprefill", None)
        specprefill_keep_pct = kwargs.pop("specprefill_keep_pct", None)
        specprefill_backbone_pct = kwargs.pop("specprefill_backbone_pct", None)
        chat_template_kwargs = dict(kwargs.pop("chat_template_kwargs", {}) or {})
        top_k = kwargs.pop("top_k", 0)
        min_p = kwargs.pop("min_p", 0.0)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        stop = kwargs.pop("stop", None)
        external_logits_processors = kwargs.pop("logits_processors", None)
        abort_event = threading.Event()

        # Per-request enable_thinking override; fall back to env var / default True.
        enable_thinking = kwargs.pop("enable_thinking", None)
        if enable_thinking is None:
            enable_thinking_env = os.environ.get("VLLM_MLX_ENABLE_THINKING", "true")
            enable_thinking = enable_thinking_env.lower() in ("true", "1", "yes")

        # Apply chat template for full prompt
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        template_kwargs.update(chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools
        safe_messages = normalize_messages_for_chat_template(messages)

        try:
            full_prompt = self._text_tokenizer.apply_chat_template(
                safe_messages, **template_kwargs
            )
        except TypeError:
            # Template doesn't accept tools= or enable_thinking=
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            full_prompt = self._text_tokenizer.apply_chat_template(
                safe_messages, **template_kwargs
            )

        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        penalty_processors = make_logits_processors(
            repetition_penalty=(
                repetition_penalty if repetition_penalty != 1.0 else None
            ),
            presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
        )
        all_processors = (external_logits_processors or []) + (penalty_processors or [])
        custom_logits_active = bool(all_processors)
        max_tokens = max_tokens or 4096

        # --- System prompt KV caching ---
        backbone_cache = None  # Backbone-only cache (no MTP), used by both paths
        prompt_to_send = full_prompt  # Default: send full prompt text
        cache_hit = False
        system_token_count = 0
        full_token_count = 0
        system_hash = None
        system_tokens = None
        suffix_tokens = None
        full_tokens_list = None
        cache_blocking_controls = []
        if not self._supports_system_kv_cache:
            cache_blocking_controls.append("non_kv_cache_class")
        if cache_blocking_controls:
            logger.info(
                "System KV cache SKIP (text route): request or engine has "
                "controls/features the cache branch cannot honor (%s); using "
                "uncached path",
                cache_blocking_controls,
            )

        # Extract system messages for caching
        has_system = any(m.get("role") == "system" for m in messages)

        if has_system and self._text_model is not None and not cache_blocking_controls:
            # Find system prefix boundary in full prompt text.
            # ChatML format: system section ends where first non-system message begins.
            # Works with tools (rendered inside system section by Qwen templates).
            system_prefix_end = -1
            for marker in ("<|im_start|>user\n", "<|im_start|>assistant\n"):
                idx = full_prompt.find(marker)
                if idx > 0:
                    system_prefix_end = idx
                    break

            if system_prefix_end > 0:
                system_prefix_text = full_prompt[:system_prefix_end]
                system_hash = hashlib.sha256(system_prefix_text.encode()).hexdigest()[
                    :16
                ]

                # Tokenize both (matching stream_generate's tokenization logic)
                tokenizer = self._text_tokenizer
                add_special = tokenizer.bos_token is None or not full_prompt.startswith(
                    tokenizer.bos_token
                )
                full_tokens_list = tokenizer.encode(
                    full_prompt, add_special_tokens=add_special
                )
                full_token_count = len(full_tokens_list)

                system_tokens_list = tokenizer.encode(
                    system_prefix_text, add_special_tokens=add_special
                )
                system_token_count = len(system_tokens_list)

                # Verify system tokens are a proper prefix of full tokens
                prefix_valid = (
                    len(full_tokens_list) > system_token_count
                    and full_tokens_list[:system_token_count] == system_tokens_list
                )

                if prefix_valid:
                    system_tokens = system_tokens_list
                    suffix_tokens = full_tokens_list[system_token_count:]

                    hit_candidate = self._system_kv_cache.get(system_hash)
                    if (
                        hit_candidate is not None
                        and system_token_count == hit_candidate[1]
                    ):
                        # Cache HIT — restore KV state into fresh backbone cache
                        def make_cache_with_snapshot(
                            text_model,
                            system_kv_snapshot,
                            _max_kv_size=self._max_kv_size,
                        ):
                            import mlx.core as mx
                            from mlx_lm.models.cache import make_prompt_cache

                            backbone_cache = make_prompt_cache(
                                text_model, max_kv_size=_max_kv_size or None
                            )
                            SimpleEngine._restore_prompt_cache(
                                backbone_cache,
                                system_kv_snapshot,
                            )

                            prompt_to_send = mx.array(suffix_tokens)
                            return backbone_cache, prompt_to_send

                        backbone_cache, prompt_to_send = (
                            await self._run_blocking_serialized(
                                make_cache_with_snapshot,
                                self._text_model,
                                hit_candidate[0],
                            )
                        )
                        # Bump LRU position now that we know we'll use it.
                        if system_hash in self._system_kv_cache:
                            self._system_kv_cache.move_to_end(system_hash)
                        self._system_kv_cache_stats["hits"] += 1
                        cache_hit = True

                        logger.info(
                            "System KV cache HIT: reusing %d cached tokens, "
                            "prefilling %d new tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                    else:
                        # Cache MISS — will prefill system tokens and snapshot
                        logger.info(
                            "System KV cache MISS: will prefill %d system tokens, "
                            "%d suffix tokens (hash=%s)",
                            system_token_count,
                            len(suffix_tokens),
                            system_hash,
                        )
                else:
                    logger.debug(
                        "System KV cache: prefix token validation failed, "
                        "using full prompt (%d tokens)",
                        len(full_tokens_list),
                    )
                    system_token_count = 0

        # Determine if SpecPrefill should be used
        # Per-request boolean override: True = force enable, False = force disable
        if specprefill_override is False:
            use_specprefill = False
        elif specprefill_override is True and self._draft_model is not None:
            use_specprefill = True  # Force enable, skip threshold check
        else:
            use_specprefill = self._draft_model is not None

        # For specprefill, ensure we have token IDs (not just prompt text)
        if use_specprefill and suffix_tokens is None and full_tokens_list is None:
            tokenizer = self._text_tokenizer
            add_special = tokenizer.bos_token is None or not full_prompt.startswith(
                tokenizer.bos_token
            )
            full_tokens_list = tokenizer.encode(
                full_prompt, add_special_tokens=add_special
            )
            full_token_count = len(full_tokens_list)

        # Tokens for specprefill: suffix (if system KV) or full prompt
        specprefill_tokens = (
            suffix_tokens if suffix_tokens is not None else full_tokens_list
        )
        specprefill_offset = system_token_count if suffix_tokens is not None else 0

        # Threshold check: only use specprefill on long prompts
        # (skipped when per-request boolean forces enable)
        if (
            use_specprefill
            and specprefill_override is not True
            and (
                specprefill_tokens is None
                or len(specprefill_tokens) <= self._specprefill_threshold
            )
        ):
            use_specprefill = False

        # Upper bound: cap specprefill to avoid draft model OOM on very long prompts
        # 65536 tokens ~ 2GB draft KV cache on Qwen3.5-4B (32KB/token x 8 attn layers)
        _SPECPREFILL_MAX_TOKENS = 65536
        if (
            use_specprefill
            and specprefill_tokens is not None
            and len(specprefill_tokens) > _SPECPREFILL_MAX_TOKENS
        ):
            logger.warning(
                "SpecPrefill: prompt %d tokens exceeds max %d, "
                "falling back to normal path",
                len(specprefill_tokens),
                _SPECPREFILL_MAX_TOKENS,
            )
            use_specprefill = False

        loop = asyncio.get_running_loop()
        response_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _emit_response(resp: Any) -> None:
            if abort_event.is_set():
                return
            loop.call_soon_threadsafe(response_queue.put_nowait, ("resp", resp))

        def _emit_done() -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("done", None))

        def _emit_error(exc: BaseException) -> None:
            loop.call_soon_threadsafe(response_queue.put_nowait, ("error", exc))

        def _seed_from_last_response(prompt_cache, last_resp):
            last_tok = getattr(last_resp, "token", None)
            if last_tok is not None:
                cache_module.trim_prompt_cache(prompt_cache, 1)
                return mx.array([last_tok], dtype=mx.uint32)
            return mx.array(
                self._text_tokenizer.encode(getattr(last_resp, "text", "")),
                dtype=mx.uint32,
            )

        def _resume_after_processor_retirement(
            model,
            prompt_cache,
            prompt,
            remaining_tokens: int,
        ) -> None:
            resume_kwargs = dict(
                max_tokens=remaining_tokens,
                sampler=sampler,
                prefill_step_size=self._prefill_step_size,
                prompt_cache=prompt_cache,
            )
            if hasattr(model, "make_mtp_cache") and model.mtp is not None:
                # Resume speculative decode from the retained backbone cache with
                # a fresh MTP cache so stale speculative state cannot survive the
                # processor-to-content handoff.
                resume_kwargs["prompt_cache"] = prompt_cache + model.make_mtp_cache()
                resume_kwargs["num_draft_tokens"] = self._mtp_num_draft_tokens
            for resp in mlx_stream_generate(
                model,
                self._text_tokenizer,
                prompt=prompt,
                **resume_kwargs,
            ):
                if abort_event.is_set():
                    logger.info("Text route: abort requested; stopping resume decode")
                    break
                _emit_response(resp)

        # Run all Metal ops in a single serialized thread.
        def _run_all():
            nonlocal backbone_cache, prompt_to_send

            model = self._text_model
            can_retire_processors = _processors_can_retire(all_processors)
            use_mtp = (
                self._mtp
                and not custom_logits_active
                and hasattr(model, "mtp")
                and model.mtp is not None
            )
            if self._mtp and custom_logits_active:
                logger.info(
                    "Text route: disabling MTP for request-local logits processors"
                )

            # Cache MISS with valid prefix: prefill system tokens and snapshot
            if (
                not cache_hit
                and system_token_count > 0
                and system_tokens is not None
                and suffix_tokens is not None
            ):
                mc = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)
                sys_arr = mx.array(system_tokens)

                # Prefill system tokens in chunks (matching generate_step)
                step = self._prefill_step_size
                while sys_arr.size > step:
                    model(sys_arr[:step][None], cache=mc)
                    self._eval_cache_snapshot([c.state for c in mc])
                    sys_arr = sys_arr[step:]
                    mx.clear_cache()
                if sys_arr.size > 0:
                    model(sys_arr[None], cache=mc)
                    self._eval_cache_snapshot([c.state for c in mc])

                # Snapshot backbone cache. Cache arrays are treated as immutable;
                # mutable state containers are copied so hybrid ArraysCache
                # entries cannot alias the saved system-prefix state.
                snapshot = self._snapshot_prompt_cache(mc)
                self._eval_cache_snapshot(snapshot)

                self._system_kv_cache[system_hash] = (snapshot, system_token_count)
                self._system_kv_cache.move_to_end(system_hash)
                evicted_count = 0
                while len(self._system_kv_cache) > self._system_kv_capacity:
                    evicted_hash, _ = self._system_kv_cache.popitem(last=False)
                    self._system_kv_cache_stats["evictions"] += 1
                    evicted_count += 1
                    logger.info(
                        "System KV cache EVICTED: hash=%s (capacity=%d)",
                        evicted_hash,
                        self._system_kv_capacity,
                    )
                if evicted_count:
                    # Eviction dropped MLX array refs; reclaim Metal heap.
                    # Skip on the common non-eviction path to avoid flushing
                    # the Metal allocator's reuse pool.
                    mx.clear_cache()
                self._system_kv_cache_stats["misses"] += 1
                self._system_kv_cache_stats["stores"] += 1

                backbone_cache = mc
                prompt_to_send = mx.array(suffix_tokens)
                logger.info(
                    "System KV cache: stored %d-token snapshot (%.1f MB), "
                    "prefilling %d remaining",
                    system_token_count,
                    sum(c.nbytes for c in mc) / 1e6,
                    len(suffix_tokens),
                )

            # --- SpecPrefill path (with fallback to normal on failure) ---
            if use_specprefill:
                try:
                    _run_specprefill(model, backbone_cache, use_mtp)
                    return
                except Exception as e:
                    logger.error(
                        "SpecPrefill failed, falling back to normal MTP path: %s",
                        e,
                    )
                    # Discard potentially corrupted cache
                    backbone_cache = None
                    prompt_to_send = full_prompt

            # --- Normal path (mlx_lm stream_generate) ---
            prompt_cache = None
            if backbone_cache is not None:
                # Add MTP cache on top of backbone
                if use_mtp and hasattr(model, "make_mtp_cache"):
                    mtp_cache = model.make_mtp_cache()
                    prompt_cache = backbone_cache + mtp_cache
                else:
                    prompt_cache = backbone_cache

            gen_kwargs = dict(
                max_tokens=max_tokens,
                sampler=sampler,
                prefill_step_size=self._prefill_step_size,
            )
            if all_processors:
                gen_kwargs["logits_processors"] = all_processors
            if use_mtp:
                gen_kwargs["num_draft_tokens"] = self._mtp_num_draft_tokens
            if prompt_cache is not None:
                gen_kwargs["prompt_cache"] = prompt_cache
            if can_retire_processors and not use_mtp:
                shared_cache = prompt_cache
                if shared_cache is None:
                    shared_cache = make_prompt_cache(
                        model, max_kv_size=self._max_kv_size or None
                    )
                gen_kwargs["prompt_cache"] = shared_cache

                token_count = 0
                last_resp = None
                retired = False
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                ):
                    if abort_event.is_set():
                        logger.info(
                            "Text route: abort requested; stopping decode after %d tokens",
                            token_count,
                        )
                        break
                    _emit_response(resp)
                    token_count += 1
                    last_resp = resp
                    retired = _processors_retired(all_processors)
                    if retired:
                        logger.info(
                            "Text route: request-local processor retired after %d tokens; "
                            "resuming content phase with MTP=%s",
                            token_count,
                            hasattr(model, "make_mtp_cache") and model.mtp is not None,
                        )
                        break

                if retired and token_count < max_tokens and last_resp is not None:
                    seed = _seed_from_last_response(shared_cache, last_resp)
                    _resume_after_processor_retirement(
                        model,
                        shared_cache,
                        seed,
                        max_tokens - token_count,
                    )
            else:
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=prompt_to_send,
                    **gen_kwargs,
                ):
                    if abort_event.is_set():
                        logger.info("Text route: abort requested; stopping decode")
                        break
                    _emit_response(resp)

        def _run_specprefill(model, bc, use_mtp):
            """Score tokens, sparse prefill, then continue on the standard decode path."""
            from types import SimpleNamespace

            from mlx_lm import stream_generate as mlx_stream_generate
            from mlx_lm.models.cache import make_prompt_cache

            from ..specprefill import (
                cleanup_rope,
                score_tokens,
                select_chunks,
                sparse_prefill,
            )

            # Create backbone cache if not already from system KV
            if bc is None:
                bc = make_prompt_cache(model, max_kv_size=self._max_kv_size or None)

            try:
                # Phase 1: Score with draft model
                import time

                t0 = time.monotonic()
                importance = score_tokens(
                    self._draft_model,
                    specprefill_tokens,
                    prefill_step_size=self._prefill_step_size,
                )
                t_score = time.monotonic() - t0

                # Phase 2: Select important chunks
                effective_keep = specprefill_keep_pct or self._specprefill_keep_pct
                effective_backbone = (
                    specprefill_backbone_pct
                    if specprefill_backbone_pct is not None
                    else self._specprefill_backbone_pct
                )
                selected = select_chunks(
                    importance,
                    keep_pct=effective_keep,
                    backbone_pct=effective_backbone,
                )
                n_selected = selected.shape[0]
                n_total = len(specprefill_tokens)

                # Phase 3: Sparse prefill on target model
                t0 = time.monotonic()
                logits = sparse_prefill(
                    model,
                    specprefill_tokens,
                    selected,
                    bc,
                    step_size=self._prefill_step_size,
                    position_offset=specprefill_offset,
                )
                t_prefill = time.monotonic() - t0

                logger.info(
                    "SpecPrefill: scored %d tokens in %.1fs, "
                    "sparse prefill %d/%d (keep=%.0f%%) in %.1fs "
                    "(offset=%d, effective_keep=%.2f)",
                    n_total,
                    t_score,
                    n_selected,
                    n_total,
                    n_selected / n_total * 100,
                    t_prefill,
                    specprefill_offset,
                    effective_keep,
                )

                # Phase 4: Sample the first token from the prefilled logits, then
                # continue through mlx_lm's normal decode path so MTP and request-
                # local logits processors remain active after sparse prefill.
                eos_id = self._text_tokenizer.eos_token_id
                seed_tokens = (
                    mx.array(full_tokens_list, dtype=mx.uint32)
                    if full_tokens_list is not None
                    else None
                )
                seeded_processors = _seed_logits_processors(seed_tokens, all_processors)
                y, _ = _sample_with_processors(
                    None,
                    logits[:, -1, :].squeeze(0),
                    sampler,
                    seeded_processors,
                )
                mx.eval(y)

                generated_ids = []
                prev_decoded = ""

                tok_id = y.item()
                generated_ids.append(tok_id)

                decoded = self._text_tokenizer.decode(generated_ids)
                new_text = decoded[len(prev_decoded) :]
                prev_decoded = decoded

                is_eos = tok_id == eos_id
                _emit_response(
                    SimpleNamespace(
                        text=new_text,
                        finish_reason="stop" if is_eos else None,
                    )
                )

                if abort_event.is_set():
                    logger.info(
                        "SpecPrefill text route: abort requested after seed token"
                    )
                    return

                if is_eos or max_tokens <= 1:
                    return

                prompt_cache = bc
                if use_mtp and hasattr(model, "make_mtp_cache"):
                    prompt_cache = bc + model.make_mtp_cache()

                continuation_prompt = mx.array([tok_id], dtype=mx.uint32)
                token_count = 1
                if _processors_retired(all_processors) and token_count < max_tokens:
                    logger.info(
                        "SpecPrefill text route: request-local processor retired after seed token; "
                        "resuming content phase with MTP=%s",
                        hasattr(model, "make_mtp_cache") and model.mtp is not None,
                    )
                    _resume_after_processor_retirement(
                        model,
                        bc,
                        continuation_prompt,
                        max_tokens - token_count,
                    )
                    return

                last_resp = None
                retired = False
                for resp in mlx_stream_generate(
                    model,
                    self._text_tokenizer,
                    prompt=continuation_prompt,
                    max_tokens=max_tokens - token_count,
                    sampler=sampler,
                    prefill_step_size=self._prefill_step_size,
                    logits_processors=seeded_processors,
                    prompt_cache=prompt_cache,
                    mtp=use_mtp,
                ):
                    if abort_event.is_set():
                        logger.info(
                            "SpecPrefill text route: abort requested; stopping decode"
                        )
                        break
                    _emit_response(resp)
                    token_count += 1
                    last_resp = resp
                    retired = _processors_retired(all_processors)
                    if retired:
                        logger.info(
                            "SpecPrefill text route: request-local processor retired after %d tokens; "
                            "resuming content phase with MTP=%s",
                            token_count,
                            hasattr(model, "make_mtp_cache") and model.mtp is not None,
                        )
                        break

                if retired and token_count < max_tokens and last_resp is not None:
                    seed = _seed_from_last_response(bc, last_resp)
                    _resume_after_processor_retirement(
                        model,
                        bc,
                        seed,
                        max_tokens - token_count,
                    )

            finally:
                cleanup_rope(model)

        async def _produce_responses() -> None:
            try:
                await self._run_blocking_serialized(
                    _run_all,
                    on_cancel=abort_event.set,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                _emit_error(exc)
            else:
                _emit_done()

        producer_task = asyncio.create_task(_produce_responses())

        # Yield results as GenerationOutput
        accumulated_text = ""
        token_count = 0
        finished = False
        _stop_gate = _StopStreamGate(stop)
        try:
            while True:
                kind, payload = await response_queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise payload
                resp = payload

                token_count += 1
                raw_new_text = resp.text if hasattr(resp, "text") else str(resp)
                # Route the raw chunk through the stop-stream gate so
                # neither the per-token SSE delta nor the cumulative
                # ``text`` ever carries the matched stop string or any
                # of its in-flight prefixes (issue #469).
                new_text, stop_hit, matched_here = _stop_gate.feed(
                    raw_new_text
                )
                if _stop_gate.enabled:
                    accumulated_text = _stop_gate.buf
                else:
                    accumulated_text += raw_new_text
                finished = stop_hit or token_count >= max_tokens
                finish_reason = getattr(resp, "finish_reason", None)
                if stop_hit:
                    finish_reason = "stop"
                elif finish_reason is None and finished:
                    finish_reason = "stop"
                elif finish_reason is not None:
                    finished = True
                # On length/abort termination flush whatever the safety
                # window held back so the client sees the full output.
                if finished and not stop_hit and _stop_gate.enabled:
                    tail, _ = _stop_gate.flush()
                    if tail:
                        new_text = (new_text or "") + tail
                        accumulated_text = _stop_gate.buf

                yield GenerationOutput(
                    text=accumulated_text,
                    new_text=new_text,
                    prompt_tokens=full_token_count or 0,
                    completion_tokens=token_count,
                    finished=finished,
                    finish_reason=finish_reason,
                    matched_stop=matched_here
                    or getattr(resp, "matched_stop", None),
                )

                if finished:
                    break
        finally:
            if not producer_task.done():
                abort_event.set()
            await producer_task

        if not finished:
            # Drain any chars the safety window held back so the client
            # doesn't lose the trailing ``max_stop_len - 1`` bytes on
            # producer-done / cancellation / EOS termination (codex r6).
            tail_text = ""
            if _stop_gate.enabled:
                tail, _ = _stop_gate.flush()
                if tail:
                    tail_text = tail
                    accumulated_text = _stop_gate.buf
            yield GenerationOutput(
                text=accumulated_text,
                new_text=tail_text,
                prompt_tokens=full_token_count or 0,
                completion_tokens=token_count,
                finished=True,
                finish_reason="length",
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        # Compute rolling generation_tps from recent completions.
        gen_tps = 0.0
        if self._recent_completions:
            total_tok = sum(c for c, _ in self._recent_completions)
            total_sec = sum(s for _, s in self._recent_completions)
            if total_sec > 0:
                gen_tps = total_tok / total_sec
        # Snapshot active requests with live elapsed_s refreshed at read time.
        now = time.time()
        requests_snapshot: list[dict[str, Any]] = []
        for entry in self._active_requests.values():
            snap = dict(entry)
            # entry stores last-known elapsed at last yield; refresh here so
            # the snapshot is meaningful even between yields.
            requests_snapshot.append(snap)
        stats = {
            "engine_type": "simple",
            "model_name": self._model_name,
            "uptime_seconds": now - self._created_at,
            "is_mllm": self._is_mllm,
            "loaded": self._loaded,
            "running": self._loaded,
            "num_running": self._num_running,
            "num_waiting": self._generation_waiters,
            "num_requests_processed": self._total_requests_processed,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "batch_generator": {
                "generation_tps": gen_tps,
                "prompt_tps": 0.0,
            },
            "requests": requests_snapshot,
            "generation_lock": {
                "locked": self._generation_lock.locked(),
                "admission": self._generation_lock_admission,
                "busy_rejections": self._generation_busy_rejections,
            },
        }

        # MLLM prefix cache stats, remapped to the shape BatchedEngine emits
        # under "memory_aware_cache" so monitoring dashboards (which key off
        # current_memory_mb / max_memory_mb / memory_utilization /
        # entry_count) render cache utilization for SimpleEngine services.
        if self._is_mllm and self._model is not None:
            try:
                raw_cache = self._model.get_cache_stats()
            except Exception:
                raw_cache = None
            if raw_cache and raw_cache.get("enabled"):
                current_mb = float(raw_cache.get("memory_used_mb", 0) or 0)
                max_mb = float(raw_cache.get("max_memory_mb", 0) or 0)
                stats["memory_aware_cache"] = {
                    "hits": raw_cache.get("hits", 0),
                    "misses": raw_cache.get("misses", 0),
                    "hit_rate": raw_cache.get("hit_rate", 0.0),
                    "evictions": raw_cache.get("evictions", 0),
                    "tokens_saved": raw_cache.get("tokens_saved", 0),
                    "current_memory_mb": round(current_mb, 2),
                    "max_memory_mb": round(max_mb, 2),
                    "memory_utilization": (
                        round(current_mb / max_mb, 4) if max_mb > 0 else 0.0
                    ),
                    "entry_count": raw_cache.get(
                        "cache_entries", raw_cache.get("entries", 0)
                    ),
                }

        # SpecPrefill stats
        if self._draft_model is not None:
            stats["specprefill"] = {
                "enabled": True,
                "draft_model": self._specprefill_draft_model_path,
                "threshold": self._specprefill_threshold,
                "keep_pct": self._specprefill_keep_pct,
                "backbone_pct": self._specprefill_backbone_pct,
            }

        # System KV cache stats (LRU over multiple system prefixes)
        if self._system_kv_cache:
            slots = []
            total_bytes = 0
            for slot_hash, (snapshot, tokens) in self._system_kv_cache.items():
                slot_bytes = 0
                for entry in snapshot:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        slot_bytes += entry[0].nbytes + entry[1].nbytes
                    elif isinstance(entry, list):
                        slot_bytes += sum(a.nbytes for a in entry if a is not None)
                total_bytes += slot_bytes
                slots.append(
                    {
                        "hash": slot_hash,
                        "tokens": tokens,
                        "memory_mb": round(slot_bytes / 1e6, 1),
                    }
                )
            counters = dict(self._system_kv_cache_stats)
            denom = counters["hits"] + counters["misses"]
            counters["hit_ratio"] = (
                round(counters["hits"] / denom, 3) if denom > 0 else None
            )
            stats["system_kv_cache"] = {
                "capacity": self._system_kv_capacity,
                "in_use": len(self._system_kv_cache),
                "total_memory_mb": round(total_bytes / 1e6, 1),
                "slots": slots,
                "counters": counters,
            }

        # Include Metal memory stats
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                stats["metal_active_memory_gb"] = round(mx.get_active_memory() / 1e9, 2)
                stats["metal_peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
                stats["metal_cache_memory_gb"] = round(mx.get_cache_memory() / 1e9, 2)
        except Exception:
            pass

        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics for the system-prompt KV LRU plus, when the
        model is multimodal, the MLLM's own cache stats.
        """
        result: dict[str, Any] = {}
        if self._supports_system_kv_cache:
            counters = dict(self._system_kv_cache_stats)
            denom = counters["hits"] + counters["misses"]
            counters["hit_ratio"] = (
                round(counters["hits"] / denom, 3) if denom > 0 else None
            )
            result["system_kv_cache"] = {
                "capacity": self._system_kv_capacity,
                "in_use": len(self._system_kv_cache),
                "counters": counters,
            }
        if self._is_mllm and self._model is not None:
            result["mllm_cache"] = self._model.get_cache_stats()
        return result or None

    def clear_runtime_caches(self) -> dict[str, Any] | None:
        """Clear engine-managed runtime caches.

        Includes the multi-slot system-prompt KV LRU — each retained snapshot
        is multi-GB on the Metal heap, so DELETE /v1/cache must drop them or
        the operator's reset is silently incomplete. Counters reset alongside
        so /v1/cache/stats reflects the cleared state immediately.

        OrderedDict ops are atomic under the GIL: a concurrent worker that has
        already captured a tuple reference from .get() finishes safely against
        its own copy; any new request after this call hits MISS and repopulates
        from scratch. No need to acquire _generation_lock for the clear itself.
        """
        result: dict[str, Any] = {}

        dropped = len(self._system_kv_cache)
        if dropped or any(self._system_kv_cache_stats.values()):
            self._system_kv_cache.clear()
            for k in self._system_kv_cache_stats:
                self._system_kv_cache_stats[k] = 0
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
            result["system_kv_cache"] = {"dropped_entries": dropped}

        if self._is_mllm and self._model is not None:
            self._model.clear_cache()
            result["model_cache"] = True

        return result or None
