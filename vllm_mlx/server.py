# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for rapid-mlx.

This module provides a FastAPI server that exposes an OpenAI-compatible
API for LLM and MLLM (Multimodal Language Model) inference using MLX on Apple Silicon.

Supports two modes:
- Simple mode (default): Maximum throughput for single-user scenarios
- Batched mode: Continuous batching for multiple concurrent users

Features:
- Text-only LLM inference (mlx-lm)
- Multimodal MLLM inference with images and video (mlx-vlm)
- OpenAI-compatible chat/completions API
- Streaming responses
- MCP (Model Context Protocol) tool integration
- Tool calling (Qwen/Llama formats)

Usage:
    # Start the server
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

The server provides:
    - POST /v1/completions - Text completions
    - POST /v1/chat/completions - Chat completions (with multimodal support)
    - GET /v1/models - List available models
    - GET /health - Health check
    - GET /v1/mcp/tools - List MCP tools
    - GET /v1/mcp/servers - MCP server status
    - POST /v1/mcp/execute - Execute MCP tool
"""

import argparse
import asyncio
import gc
import logging
import os
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Single source of truth for the OpenAI-shaped 400 / 422 / 500 envelopes
# (F-161 / F-162 / F-163 / F-094-class). Defined in ``middleware`` so
# tests can install the same handlers on a stub FastAPI without
# importing the heavy engine stack.
from .middleware.exception_handlers import (  # noqa: E402
    _decode_error_response,  # noqa: F401 — re-exported for back-compat
    install_exception_handlers,  # noqa: F401 — re-exported for tests
)
from .middleware.exception_handlers import (
    _http_error_response as _http_exception_handler_impl,  # noqa: F401
)


# Back-compat shim: ``tests/test_context_length_exceeded.py`` and
# ``tests/test_config_and_middleware.py`` import this symbol from
# ``vllm_mlx.server`` and register it manually on a stub FastAPI.
# The real handler now lives in ``middleware/exception_handlers.py``;
# keep this signature stable so the existing test suites keep working.
async def _http_exception_handler(request, exc):  # noqa: ARG001
    return _http_exception_handler_impl(exc)


# Re-export for backwards compatibility with tests
from .api.anthropic_adapter import (  # noqa: F401
    anthropic_to_openai,
    openai_to_anthropic,
)
from .api.anthropic_models import AnthropicRequest  # noqa: F401
from .api.models import (
    AssistantMessage,  # noqa: F401
    ChatCompletionChoice,  # noqa: F401
    ChatCompletionChunk,  # noqa: F401
    ChatCompletionChunkChoice,  # noqa: F401
    ChatCompletionChunkDelta,  # noqa: F401
    ChatCompletionRequest,  # noqa: F401
    ChatCompletionResponse,  # noqa: F401
    ChoiceLogProbs,  # noqa: F401
    CompletionChoice,  # noqa: F401
    CompletionRequest,  # noqa: F401
    CompletionResponse,  # noqa: F401
    CompletionTokensDetails,  # noqa: F401
    ContentPart,  # noqa: F401
    FunctionCall,  # noqa: F401
    ImageUrl,  # noqa: F401
    MCPServerInfo,  # noqa: F401
    MCPToolInfo,  # noqa: F401
    Message,  # noqa: F401
    ModelInfo,  # noqa: F401
    TokenLogProb,  # noqa: F401
    ToolCall,  # noqa: F401
    TopLogProb,  # noqa: F401
    Usage,  # noqa: F401
    VideoUrl,  # noqa: F401
)
from .api.tool_calling import (
    build_json_system_prompt,  # noqa: F401
    convert_tools_for_template,  # noqa: F401
    extract_json_schema_for_guided,  # noqa: F401
    parse_json_output,  # noqa: F401
    parse_tool_calls,  # noqa: F401
)
from .api.utils import (
    SPECIAL_TOKENS_PATTERN,  # noqa: F401
    StreamingThinkRouter,  # noqa: F401
    StreamingToolCallFilter,  # noqa: F401
    clean_output_text,  # noqa: F401
    extract_json_from_response,  # noqa: F401
    extract_multimodal_content,  # noqa: F401
    is_mllm_model,  # noqa: F401
    resolve_serving_lane,  # noqa: F401
    sanitize_output,  # noqa: F401
    strip_special_tokens,  # noqa: F401
    strip_thinking_tags,  # noqa: F401
)
from .config import get_config
from .engine import (
    BaseEngine,
    BatchedEngine,
)
from .runtime.model_registry import ModelEntry, ModelRegistry
from .runtime.resident_models import ResidentModelManager, estimate_model_bytes
from .service.helpers import (  # noqa: F401 — re-export for backward compat
    _FALLBACK_TEMPERATURE,
    _FALLBACK_TOP_P,
    _TOOL_USE_SYSTEM_SUFFIX,
    _build_usage,
    _cascade,
    _disconnect_guard,
    _extract_token_logprob,
    _inject_json_instruction,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _resolve_frequency_penalty,
    _resolve_max_tokens,
    _resolve_min_p,
    _resolve_model_name,
    _resolve_presence_penalty,
    _resolve_repetition_penalty,
    _resolve_temperature,
    _resolve_top_k,
    _resolve_top_p,
    _validate_model_name,
    _validate_tool_call_params,
    _wait_with_disconnect,
    get_engine,
    get_usage,
)
from .tool_parsers import ToolParserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_log_level(log_level: str) -> str:
    return log_level.upper()


def configure_logging(log_level: str) -> str:
    normalized = normalize_log_level(log_level)
    logging.getLogger().setLevel(getattr(logging, normalized, logging.INFO))
    logger.setLevel(getattr(logging, normalized, logging.INFO))

    # Silence chatty transport-layer loggers unless the user explicitly asked
    # for DEBUG. At INFO level, ``httpx`` emits one line per HF Hub request
    # (config.json, README, every model shard), which floods startup with a
    # screenful of pure noise before the model even loads. ``huggingface_hub``
    # also doubles up on transfer chatter. Pinning them to WARNING leaves
    # genuine errors visible without the per-request play-by-play.
    #
    # On DEBUG we explicitly reset to NOTSET so they inherit the root level,
    # making this idempotent across repeated configure_logging() calls in the
    # same process (test fixtures, in-process restarts).
    chatty_loggers = ("httpx", "httpcore", "urllib3", "huggingface_hub")
    target = logging.NOTSET if normalized == "DEBUG" else logging.WARNING
    for name in chatty_loggers:
        logging.getLogger(name).setLevel(target)

    return normalized.lower()


# Multi-model registry — supports loading 2+ models simultaneously.
# When populated, get_engine() routes by request model name.
# Backward-compatible: single-model mode still uses _engine global as before.
_model_registry = ModelRegistry()
_residency_manager: ResidentModelManager | None = None
_resident_memory_limit_bytes: int = 0
_resident_idle_ttl_seconds: float = 0.0
_resident_gpu_memory_utilization: float = 0.90

# Global engine instance (single-model legacy path, also primary model in multi-model)
_engine: BaseEngine | None = None
# Background prefix-cache load scheduled after the readiness flip (#1350). Held
# at module scope so ``asyncio`` doesn't garbage-collect the task mid-flight and
# so shutdown can await a still-running load to completion before the shutdown
# save and engine teardown.
_prefix_cache_load_task = None  # asyncio.Task | None
_model_name: str | None = None
_model_alias: str | None = None  # Short alias used to start the model (if any)
# Task #292 (Bo R13/R14): operator opt-in for ``/v1/audio/*`` routes on a
# text-only server. Set to True by ``--enable-audio`` (text mode) or by
# :func:`vllm_mlx.cli._serve_audio_mode` (audio mode). The audio-mode
# helper also stamps a registry-known model_alias/model_name so the gate
# in :func:`register_audio_routes_if_enabled` fires from the registry
# branch — the flag is the explicit-opt-in fallback for text-only servers
# that intentionally want the audio routes mounted (e.g. side-car patterns
# where the audio backend lives in a separate process the routes proxy to).
_enable_audio_lane: bool = False
_model_path: str | None = (
    None  # Actual model path (for cache dir, not affected by --served-model-name)
)
_default_max_tokens: int = 4096
_default_max_tokens_is_explicit: bool = False
_thinking_token_budget: int = 2048  # Extra tokens added for thinking models
_default_timeout: float = 1800.0  # Default request timeout in seconds (30 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p
_default_top_k: int | None = None  # Set via --default-top-k
_default_min_p: float | None = None  # Set via --default-min-p
_default_repetition_penalty: float | None = None  # Set via --default-repetition-penalty
_default_presence_penalty: float | None = None  # Set via --default-presence-penalty
_default_frequency_penalty: float | None = None  # Set via --default-frequency-penalty

# Sampling overlays populated from the model's AliasProfile +
# generation_config.json once the path is known (load_model). Both stay
# as None pre-load; the resolve helpers tolerate missing dicts.
_alias_recommended_sampling: dict[str, float | int] | None = None
_generation_config_sampling: dict[str, float | int] | None = None


# Global MCP manager
_mcp_manager = None
_mcp_executor = None
# Issue #1716: MCP is optional and must never be able to fail server boot.
# When init/reload fails these carry the reason (and the path to retry from)
# out to ``/v1/mcp/servers`` so the desktop app can render something the user
# can act on instead of an empty connector list.
_mcp_init_error: str | None = None
_mcp_config_path: str | None = None
# Per-server entries dropped by the tolerant config load, with their reasons.
_mcp_rejected: list = []
# Serializes concurrent ``reload_mcp`` calls: two overlapping reloads both
# tearing down and rebuilding the global manager would corrupt each other's
# state. Created lazily (module import must not touch the event loop).
_mcp_reload_lock: "asyncio.Lock | None" = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used
# Operator embedding-length config (issue #1381). Set once from the CLI and
# reused by route-triggered loads so both the pre-load and lazy paths build
# the engine with the same limits.
_embedding_max_length: int | str = "auto"
_embedding_overflow_policy: str = "truncate"

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False


def _resolve_api_key(argv_value: str | None) -> str | None:
    """Resolve the effective API key with env-var fallback.

    Argv-inline (``--api-key X``) wins for backwards-compat with
    existing scripts; otherwise we fall back to the ``RAPID_MLX_API_KEY``
    env var. The env-var form keeps the bearer key out of ``argv``
    (visible to ``ps -ef`` for any local user) — this is the path
    rapid-desktop's sidecar shim uses to avoid the codex BLOCKER #3
    "bearer-in-shell-history" leak.

    Exposed at module scope (not buried inside ``main()``) so the
    env-fallback contract is directly unit-testable without booting
    a model — a regression here is the bug the dogfood-v0.8.2 finding
    #3 exposed, so a test-via-the-real-code path matters.
    """
    return argv_value or os.environ.get("RAPID_MLX_API_KEY")


# Per-request body size cap (DoS defense). 0 disables. Resolved from
# CLI ``--max-request-bytes`` / ``RAPID_MLX_MAX_REQUEST_BYTES`` and
# pushed into ``ServerConfig.max_request_bytes`` via ``_sync_config``;
# the ASGI middleware (``middleware/body_size.py``) reads it per
# request, so a test fixture mutating the config takes effect immediately.
_max_request_bytes: int = 8 * 1024 * 1024

# SSE keepalive interval (F-070 DoS / proxy idle-timeout defense). 0
# disables. Resolved from ``RAPID_MLX_SSE_KEEPALIVE_SECONDS`` and
# pushed into ``ServerConfig.sse_keepalive_seconds`` via
# ``_sync_config``. ``_disconnect_guard`` reads it at start of each
# stream — see ``vllm_mlx/service/helpers.py``.
_sse_keepalive_seconds: float = 20.0

# Body-receive idle timeout (F-072 slow-DoS defense). 0 disables.
# Resolved from ``RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS`` and pushed
# into ``ServerConfig.body_receive_timeout_seconds`` via
# ``_sync_config``. The ``RequestBodyLimitMiddleware`` wraps each
# ``receive()`` in ``asyncio.wait_for`` until the body is fully on the
# wire, emits HTTP 408 on timeout.
_body_receive_timeout_seconds: float = 15.0

# Reasoning parser (for models like Qwen3, DeepSeek-R1, MiniMax)
_reasoning_parser = None  # ReasoningParser instance when enabled
_reasoning_parser_name: str | None = None  # Parser name (e.g., "minimax")

# Tool calling configuration
_enable_auto_tool_choice: bool = False
_tool_call_parser: str | None = None  # Parser name: auto, mistral, qwen, llama, hermes
_tool_parser_instance = None  # Instantiated parser
_enable_tool_logits_bias: bool = False  # Jump-forward decoding for tool calls

# GC control (Tier 0 optimization)
_gc_control: bool = True  # Disable GC during generation to avoid latency spikes
_no_thinking: bool = (
    False  # --no-thinking: force enable_thinking=False in chat template
)

#: Keep a mid-conversation ``role="system"`` message at its position
#: instead of hoisting it into the leading system block. Set from
#: ``serve --relocate-mid-conversation-system``; OFF by default.
_relocate_mid_conversation_system: bool = False

# Pinned prefix cache (Tier 0 optimization)
_pin_system_prompt: bool = False  # Auto-pin system prompt prefix cache blocks
_pinned_system_prompt_hash: str | None = None  # Hash of pinned system prompt


from .runtime.cache import (
    load_prefix_cache_from_disk as _load_prefix_cache_from_disk,
)
from .runtime.cache import (
    save_prefix_cache_to_disk as _save_prefix_cache_to_disk,
)


async def _shutdown_save_prefix_cache() -> None:
    """Lifespan shutdown step: persist prefix cache off the event loop.

    The synchronous ``_save_prefix_cache_to_disk`` call streams 200-300
    MB per entry through ``save_prompt_cache`` under the GIL. Calling
    it on the asyncio loop thread (which is what the lifespan handler
    runs on) starves any other coroutine waiting on the same loop for
    tens of seconds — ``/healthz`` polls from supervisors that still
    consider us "stopping" hang, graceful-shutdown HTTP responses
    never flush, etc.

    Extracted from inline-in-lifespan so tests can pin the
    ``asyncio.to_thread`` wrapper at its production callsite. Codex
    flagged PR #667 round 1 because the previous test wrapped
    ``to_thread`` itself — a regression that dropped the wrapper from
    the lifespan would not have been caught. The test now drives THIS
    function and watches whether the loop stays responsive during the
    save; if anyone in the future replaces the ``await asyncio.to_thread
    (...)`` line below with a direct call, the regression fires.
    """
    if _engine is None or not hasattr(_engine, "save_cache_to_disk"):
        return
    await asyncio.to_thread(_save_prefix_cache_to_disk)


async def _deferred_load_prefix_cache() -> None:
    """Lifespan startup step: warm the prefix cache off the readiness path.

    Mirror of :func:`_shutdown_save_prefix_cache` for the load side (#1350).
    The synchronous ``_load_prefix_cache_from_disk`` streams hundreds of MB
    off disk under the GIL; running it inline in the lifespan handler — before
    ``_cfg.ready = True`` — kept ``/health/ready`` and ``/v1/models`` at 503
    for the entire load. It is a pure warm-start optimization, so the lifespan
    now flips readiness first and schedules THIS coroutine as a background
    task. Extracted (rather than inlined as a closure) so a regression test can
    pin the ``asyncio.to_thread`` wrapper at its production callsite: if anyone
    later replaces the wrapped call below with a direct one, the loop-starves-
    during-load regression fires. Failures are non-fatal — a cold cache only
    costs a few early prefix recomputes, never a wedged server.
    """
    if _engine is None or not hasattr(_engine, "load_cache_from_disk"):
        return
    try:
        await asyncio.to_thread(_load_prefix_cache_from_disk)
    except Exception as _e:  # noqa: BLE001
        logger.warning(f"[lifespan] deferred prefix-cache load failed: {_e}")


async def _drain_deferred_prefix_cache_load() -> None:
    """Lifespan shutdown step: let the deferred prefix-cache load FINISH (#1350).

    We AWAIT the background load task rather than cancel it. ``Task.cancel()``
    only unblocks us from awaiting the ``asyncio.to_thread`` wrapper — the
    worker thread keeps running ``_load_prefix_cache_from_disk``, which reads
    the on-disk cache and calls into the engine. Running the shutdown save or
    ``_engine.stop()`` underneath a still-live loader would race the cache
    files and the engine's own state. Awaiting is bounded by the load duration
    (the same cost the old synchronous on-startup load always paid) and is only
    ever non-instant in the rare case shutdown arrives mid-load; in the common
    case the task is already done and this returns immediately. The load
    coroutine swallows its own errors, so the guard here is belt-and-suspenders.
    """
    task = _prefix_cache_load_task
    if task is None:
        return
    try:
        await task
    except Exception as _e:  # noqa: BLE001
        logger.debug(f"[lifespan] deferred prefix-cache load cleanup: {_e}")


def _do_tool_grammar_warmup(tokenizer, parser_cls) -> bool:
    """Pre-build the llguidance ``LLTokenizer`` (+ warm the grammar path).

    CPU-ONLY and idempotent, safe to run on a worker thread: it touches only
    llguidance's Rust surface, never an MLX GPU op, so it does not trip the
    per-thread MLX stream gotcha (#170) that forbids GPU evals off the step
    thread. The dominant cost this hoists off the first request is the
    ``LLTokenizer`` build — a ~1s, vocab-scale operation llguidance's own docs
    flag "expensive … should be cached". ``get_lltokenizer`` memoizes it on the
    tokenizer, so building it here means the first real tool-call request hits a
    warm cache instead of paying ~1s inline. Returns True if the tokenizer warmed.
    """
    from .api.tool_grammar import (
        build_tool_grammar,
        get_lltokenizer,
        get_request_matcher,
    )

    lltok = get_lltokenizer(tokenizer)
    if lltok is None:
        return False
    # Also warm the grammar-build + compiled-matcher path (module imports, the
    # Lark->grammar compile, one automaton construction) with a trivial 0-arg
    # tool so the first real request's setup is fully hot. Per-request schemas
    # differ, so we cannot pre-compile a client's specific grammar — the win is
    # the shared LLTokenizer above; this just primes the rest of the code path.
    try:
        parser = parser_cls(tokenizer=tokenizer)
        warm_tools = [
            {
                "name": "_rapid_mlx_warmup",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        ]
        grammar = build_tool_grammar(warm_tools, "required", parser)
        if grammar is not None:
            get_request_matcher(lltok, grammar)
    except Exception:
        # Non-fatal: the LLTokenizer (the expensive part) is already warm. Log
        # the secondary grammar-path priming failure rather than swallowing it
        # silently, so a real defect here is diagnosable (codex #1155 nit).
        logger.warning(
            "Tool-grammar warmup: LLTokenizer built, but grammar-path priming "
            "failed (non-fatal — first real request re-primes it)",
            exc_info=True,
        )
    return True


async def _warmup_tool_grammar(engine) -> None:
    """Startup warmup for grammar-constrained tool calling (#558).

    Gated so it only fires when the server is actually configured for
    grammar-capable tool calling: a ``tool_call_parser`` is set, the parser
    class opts into ``SUPPORTS_GRAMMAR``, and the llguidance stack is importable.
    Any other case returns immediately (a text-only or non-grammar deploy pays
    nothing at boot). The heavy build runs via ``asyncio.to_thread`` so the ~1s
    LLTokenizer construction never blocks the event loop. Fully non-fatal.
    """
    cfg = get_config()
    parser_name = getattr(cfg, "tool_call_parser", None)
    if not parser_name:
        return
    try:
        from .api.tool_grammar import HAS_LL_TOKENIZER, HAS_LLGUIDANCE
    except Exception:
        return
    if not (HAS_LLGUIDANCE and HAS_LL_TOKENIZER):
        return
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        return
    try:
        from .tool_parsers import ToolParserManager

        parser_cls = ToolParserManager.get_tool_parser(parser_name)
    except Exception:
        return
    if not getattr(parser_cls, "SUPPORTS_GRAMMAR", False):
        return
    # Run the warmup on the loop's default thread pool and AWAIT it (the ~1s
    # build runs off the event loop, so the loop stays responsive; startup waits
    # for it the same way it already waits for ``generate_warmup``'s Metal-shader
    # compile above). We deliberately impose NO timeout / background thread: a
    # ``wait_for`` cannot cancel ``to_thread`` (the worker keeps running), and a
    # detached daemon thread is untracked at shutdown (codex #1155). The warmup is
    # a bounded, single-flighted CPU build that cannot realistically hang, so the
    # simplest correct shape is a plain awaited ``to_thread`` — no orphaned worker
    # to manage, and a genuine llguidance defect surfaces as a normal startup
    # error rather than being masked by a timeout.
    warmed = await asyncio.to_thread(_do_tool_grammar_warmup, tokenizer, parser_cls)
    if warmed:
        logger.info(
            "Tool-grammar warmup complete (LLTokenizer pre-built for parser %r)",
            parser_name,
        )


def _detect_hybrid_for_warmup(engine) -> bool:
    """Whether the loaded model is hybrid for warmup-gating purposes.

    Hybrid (GatedDeltaNet/Mamba + Transformer) models must SKIP the bare
    ``generate_warmup`` (it contaminates compiled kernel state that
    interferes with batched inference) and take the full-request warmup
    path instead. Two independent signals, either sufficing:

    1. The engine's fail-closed ``_is_hybrid_model()`` profile probe
       (BatchedEngine). This replaced the old ``_hybrid_throttle`` read,
       which stopped implying is-hybrid when the #115 admission throttle
       default flipped OFF (codex review on the retirement PR).
    2. The pre-existing ``make_cache()``/``ArraysCache`` structural
       detection through the wrapper layers — retained unchanged as the
       fallback for engine wrappers that do not expose the probe.

    MLLM engines are excluded FIRST, before either signal (their warmup
    path is separate). Today the combination cannot arise — hybrid VLMs
    auto-downgrade to the text lane because the MLLM engine cannot build
    a BatchKVCache over an ArraysCache backbone (#352) — but if the MLLM
    lane ever gains hybrid support, warmup must fail closed to the bare
    path rather than silently entering the hybrid one.
    """
    if getattr(engine, "_is_mllm", False):
        return False
    probe = getattr(engine, "_is_hybrid_model", None)
    if callable(probe) and bool(probe()):
        return True
    model = getattr(engine, "_model", None) or getattr(engine, "_shared_model", None)
    if model and hasattr(model, "model") and not hasattr(model, "make_cache"):
        model = model.model
    if model and hasattr(model, "make_cache"):
        try:
            from mlx_lm.models.cache import ArraysCache

            return any(isinstance(c, ArraysCache) for c in model.make_cache())
        except Exception:
            pass
    return False


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager

    # Install process-death observability BEFORE any executor is created.
    # Two complementary mechanisms (codex r3 NIT clarification):
    #
    #   * ``faulthandler.enable()`` installs an async-signal-safe
    #     C-level handler for SIGSEGV / SIGBUS / SIGILL / SIGFPE /
    #     SIGABRT — i.e. all the crash signals MLX / Metal C extensions
    #     can raise. The handler writes a Python traceback to stderr
    #     before the interpreter dies. SIGABRT is owned ONLY by
    #     faulthandler — it is intentionally NOT in our
    #     ``signal.signal`` chain (a Python-level handler there would
    #     call ``logging``, which is not async-signal-safe and would
    #     downgrade the abort-path observability).
    #
    #   * A chained ``signal.signal`` handler for SIGTERM and SIGHUP
    #     only. The handler logs a single WARNING line, dumps
    #     ``faulthandler.dump_traceback(all_threads=True)``, then
    #     chains to uvicorn's prior ``handle_exit`` (so graceful
    #     shutdown still runs) or, when the prior was ``SIG_DFL``
    #     (e.g. SIGHUP under uvicorn, which does not capture SIGHUP),
    #     restores SIG_DFL + ``raise_signal`` so the kernel-level
    #     terminate-by-default fires after the log line lands.
    #
    # Mirrors C-04 recon §3.R1 + §3.R2 (``/tmp/dogfood-085/c04-recon.md``)
    # — three persona logs from the 0.8.5 dogfood ran exclusively in the
    # "process disappeared between two stdout writes" shape with no
    # traceback, no shutdown banner, no crash report. Without these hooks
    # the operator cannot tell SIGKILL (un-catchable) from SIGTERM
    # (catchable but currently invisible) from a Metal segfault. The
    # install is idempotent + safe off the main thread (returns False
    # rather than raising), so re-entry from test harnesses and embedded-
    # uvicorn contexts is tolerated.
    from ._signal_observability import install_signal_observability

    install_signal_observability()

    # GC control: raise thresholds to reduce GC frequency with large models
    if _gc_control:
        gc.set_threshold(100_000, 50, 50)
        logger.info("GC control enabled: thresholds set to (100000, 50, 50)")

    # Startup: Start engine if loaded (needed for BatchedEngine in uvicorn's event loop)
    if _engine is not None and hasattr(_engine, "_loaded") and not _engine._loaded:
        try:
            await _engine.start()
        except Exception as _start_exc:
            # Opt-in telemetry (Phase 2.2 error wiring): serve's real weight
            # load happens HERE in the async lifespan, not in the CLI's
            # ``load_model()`` (which only does config read + MLLM/LLM
            # type-detection). A failure here is THE ``serve`` model-load
            # failure — the CLI-side wiring (PR #1207) cannot see it. Record
            # a bucketed error (allowlisted category + traceback fingerprint
            # only, never the model name / message / path), then re-raise so
            # startup still aborts exactly as before. ``emit.error`` is
            # ``is_enabled()``-gated and ``@_safe`` → a no-op when telemetry
            # is off and can never mask the failure.
            from vllm_mlx.telemetry import emit as _telemetry_emit

            _telemetry_emit.error(
                category="model_load_failure", exc=_start_exc, phase="startup"
            )
            raise

    # Warmup: generate one token to trigger Metal shader compilation.
    # Runs here (not in CLI) so all engine types are fully started first.
    if _engine is not None:
        import time as _time

        logger.info("Warming up (compiling Metal shaders)...")
        _warmup_start = _time.monotonic()
        try:
            _is_hybrid = _detect_hybrid_for_warmup(_engine)
            if not _is_hybrid:
                _engine.generate_warmup()
                # NOTE: do NOT call `mx.eval(mx.zeros(1))` here — that
                # allocates on the main (asyncio loop) thread which lazily
                # creates Stream(gpu, 1), and any subsequent eval of arrays
                # whose graph touches that stream from the mlx-step worker
                # raises "There is no Stream(gpu, 1) in current thread"
                # (#170). `generate_warmup()` already routes its own forward
                # + eval through the step thread, which is what we want.
            else:
                # Hybrid models need a full request warmup to compile
                # Metal shaders and prime the BatchGenerator, preventing
                # corruption on the first concurrent batch.
                logger.info(
                    "Hybrid model: running full request warmup "
                    "(compiling GatedDeltaNet kernels)"
                )
                try:
                    async for _ in _engine.stream_chat(
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=2,
                        temperature=0.0,
                    ):
                        pass
                except Exception as _e:
                    logger.debug(f"Hybrid warmup error (non-fatal): {_e}")
        except Exception as e:
            logger.debug(f"Warmup failed (non-fatal): {e}")
        _warmup_secs = _time.monotonic() - _warmup_start
        logger.info(f"Warmup complete ({_warmup_secs:.1f}s)")

    # Publish the startup engine into the resident-model lifecycle after it is
    # fully started. Legacy routes still expose this engine through cfg.engine,
    # so it is the protected primary; additional engines are manager-owned and
    # eligible for LRU/TTL eviction.
    global _residency_manager
    if _residency_manager is None:
        _residency_manager = configure_model_residency(
            memory_limit_gb=_resident_memory_limit_bytes / 1024**3,
            idle_ttl_seconds=_resident_idle_ttl_seconds,
            gpu_memory_utilization=_resident_gpu_memory_utilization,
        )
    if _engine is not None:
        _primary_entry = next(
            (
                entry
                for entry in _model_registry.list_entries()
                if entry.engine is _engine
            ),
            None,
        )
        if _primary_entry is not None and not _residency_manager.contains(
            _primary_entry.model_name
        ):
            _residency_manager.register_primary(
                _primary_entry,
                estimated_bytes=estimate_model_bytes(
                    _model_alias or _primary_entry.model_name
                ),
            )
    await _residency_manager.start()

    # Tool-grammar warmup (#558): the FIRST grammar-constrained tool call
    # otherwise pays a one-time ~1s llguidance ``LLTokenizer`` build on the
    # request path (measured on gpt-oss-20b: ~1.7s cold first tool-call vs
    # ~0.37s warm — distinct schemas AFTER the first are already warm, so the
    # cost is the shared tokenizer build, not per-schema compile). Pre-build it
    # at startup, off the event loop, so no user request eats the cold-start.
    # Self-gates to grammar-capable tool deployments and is non-fatal.
    if _engine is not None:
        try:
            await _warmup_tool_grammar(_engine)
        except Exception as _e:
            logger.debug(f"Tool-grammar warmup failed (non-fatal): {_e}")

    # Prefix-cache load is deferred OFF the readiness path (#1359 follow-up
    # #1350): a large persisted cache used to block here — between engine
    # start and ``_cfg.ready = True`` — so ``/health/ready`` and ``/v1/models``
    # stayed 503 for the whole (potentially multi-second) disk load. It is a
    # pure warm-start optimization, so it is now scheduled as a background
    # task AFTER the readiness flip below; see ``_prefix_cache_load_task``.

    # Initialize MCP if config provided. VLLM_MLX_MCP_CONFIG is the
    # deprecated pre-rename alias. Prefer the first var that points to an
    # existing file so a stale new var doesn't shadow a working legacy one
    # (mirrors load_mcp_config's existence-aware fallback); fall back to the
    # first that is merely set so a genuinely-missing path still surfaces an
    # error rather than being silently ignored.
    mcp_env_vars = ("RAPID_MLX_MCP_CONFIG", "VLLM_MLX_MCP_CONFIG")
    mcp_candidates = [v for v in (os.environ.get(k) for k in mcp_env_vars) if v]
    mcp_config = next(
        (p for p in mcp_candidates if os.path.isfile(os.path.expanduser(p))),
        mcp_candidates[0] if mcp_candidates else None,
    )
    if mcp_config:
        await init_mcp(mcp_config)

    # F-K-CAPABILITIES-OMIT-AUDIO: run a deep audio-lane dry-run so
    # the per-lane status surfaces on ``/v1/models`` capability tags
    # BEFORE the first user request lands on a degraded backend. The
    # existing shallow probe only checks ``mlx_audio`` importability;
    # a model that loads but can't generate output (F-K-WHISPER-500
    # shape) still passed the shallow probe and 500'd at first use.
    #
    # Off by default to keep cold-start fast for text-only deploys —
    # turn on via ``RAPID_MLX_AUDIO_DEEP_PROBE=1`` when running an
    # audio-serving build. The dry-run is non-fatal: any failure is
    # caught inside ``deep_probe_audio_lane`` and recorded as
    # ``degraded`` / ``missing``; the lifespan completes regardless
    # so a torn audio backend doesn't block server boot.
    #
    # Codex r2 NIT #3: call ``deep_probe_audio_lane`` unconditionally
    # (even when ``mlx_audio`` is missing) so ``/v1/models`` carries
    # ``audio_lanes={"stt":"missing","tts":"missing"}`` on bare
    # installs. The prior branch short-circuited on ``find_spec``
    # and ``audio_lanes`` came back ``null``, hiding the "no audio
    # extra installed" state from operators using the field for
    # health. ``deep_probe_audio_lane`` already runs the shallow
    # presence check internally and records the missing-extra
    # status via the same code path the route's 503 envelope uses.
    # Codex r3 NIT #2: lowercase the env value before comparing so
    # ``RAPID_MLX_AUDIO_DEEP_PROBE=False`` (capital F) and ``NO``
    # (uppercase) are treated as falsy, not truthy. Mirrors the
    # convention used by every other ``RAPID_MLX_*`` boolean knob.
    _audio_deep_probe = os.environ.get("RAPID_MLX_AUDIO_DEEP_PROBE", "").strip().lower()
    if _audio_deep_probe and _audio_deep_probe not in ("0", "false", "no"):
        try:
            from .audio.probe import deep_probe_audio_lane as _deep_probe

            logger.info("Running deep audio probe (STT + TTS dry-run)...")
            _stt_status = _deep_probe("stt")
            _tts_status = _deep_probe("tts")
            logger.info(
                "Audio lane status — stt=%s, tts=%s",
                _stt_status.get("status"),
                _tts_status.get("status"),
            )
        except Exception as _audio_err:  # noqa: BLE001
            logger.warning("Deep audio probe failed (non-fatal): %s", _audio_err)

    # All slow startup work done. Flip the readiness flag so /health/ready
    # starts returning 200. Anything that races a request before this point
    # would otherwise hit a not-yet-warmed engine.
    _cfg = get_config()
    from .routes.video import start_video_jobs

    start_video_jobs()
    _cfg.ready = True

    # Now that readiness is flipped, warm the prefix cache from disk in the
    # background (#1350). The memory-aware cache installs each imported entry
    # as a single atomic bulk-swap under its own lock, so a request that
    # arrives mid-load either misses (recompute — always correct) or hits the
    # fully-installed cache, never a partially-populated one. Worst case a few
    # early requests recompute their prefix; they never see a wedged server.
    if _engine is not None and hasattr(_engine, "load_cache_from_disk"):
        global _prefix_cache_load_task
        _prefix_cache_load_task = asyncio.create_task(_deferred_load_prefix_cache())

    # Render the real "Ready:" / "Connect:" banner now — only here is the
    # port truly accepting connections AND the engine warmed up. The CLI's
    # earlier "Starting server …" line is replaced by this. Output is produced
    # by the connect SSOT (:mod:`vllm_mlx.connect`) so the served banner and
    # ``rapid-mlx connect`` can never disagree about an endpoint. If neither
    # the host/port nor inherited-fd source of truth was stashed (e.g.
    # embedded usage where uvicorn is owned elsewhere), fall back silently.
    from vllm_mlx.connect import endpoints_from_bind, render_banner

    _ep = endpoints_from_bind(
        _cfg.bind_host,
        _cfg.bind_port,
        model=_cfg.model_alias or _cfg.model_name,
        listen_fd=_cfg.bind_listen_fd,
    )
    if _ep.listen_fd is not None or (_cfg.bind_host and _cfg.bind_port):
        print(render_banner(_ep), end="")

    yield

    # Shutdown: stop accepting "ready" before tearing things down.
    # R15 Sven B2 (task #306): also flip ``draining`` so /healthz
    # surfaces 503 to the load balancer / k8s readiness probe. Without
    # this flip the orchestrator keeps routing new traffic into a
    # tearing-down instance until the TCP listener is fully closed,
    # producing tail-end request loss the operator can't distinguish
    # from a crash. In-flight requests continue to completion; only
    # the readiness signal flips here.
    _shutdown_cfg = get_config()
    _shutdown_cfg.ready = False
    _shutdown_cfg.draining = True

    # Shutdown teardown: save cache, close MCP, stop engine.
    #
    # ``_shutdown_save_prefix_cache`` wraps the synchronous save in
    # ``asyncio.to_thread`` — see that function's docstring for the
    # rationale. Extracted so the regression test pins the wrapper at
    # the production callsite rather than wrapping ``to_thread``
    # test-side (codex PR #667 round 1 BLOCKING-3).
    #
    # Opt-in telemetry (Phase 2.2 error wiring): a crash while tearing
    # down is exactly the "process disappeared during shutdown" shape the
    # signal-observability hooks above were installed for. Record a
    # bucketed ``shutdown_traceback`` error (allowlisted category/phase +
    # traceback fingerprint only — no message text or path), then re-raise
    # so the shutdown path behaves identically. ``emit.error`` is
    # ``is_enabled()``-gated and ``@_safe`` → a no-op when telemetry is off
    # and never masks the failure.
    try:
        from .routes.video import shutdown_video_jobs

        await shutdown_video_jobs()

        # Let the deferred prefix-cache load (#1350) finish before we save or
        # tear down the engine — see the helper's docstring for why we await
        # rather than cancel.
        await _drain_deferred_prefix_cache_load()

        await _shutdown_save_prefix_cache()

        # Shutdown: Close MCP connections and stop engine
        if _mcp_manager is not None:
            await _mcp_manager.stop()
            logger.info("MCP manager stopped")
        if _residency_manager is not None:
            await _residency_manager.shutdown()
        if _engine is not None:
            await _engine.stop()
            logger.info("Engine stopped")
    except Exception as _shutdown_exc:
        from vllm_mlx.telemetry import emit as _telemetry_emit

        _telemetry_emit.error(
            category="shutdown_traceback", exc=_shutdown_exc, phase="shutdown"
        )
        raise

    # Round 19 codex review (PR #532): Drive the telemetry session_end
    # path here too. ``atexit`` does NOT fire on SIGTERM (systemd /
    # Docker / Kubernetes graceful stop), so an opted-in user running
    # ``rapid-mlx serve`` under a service manager would otherwise lose
    # the lifecycle end event. uvicorn drives this lifespan shutdown
    # on SIGTERM, so the hook lands. The latch inside the telemetry
    # emit module ensures the event is sent exactly once even if
    # atexit fires later as well.
    try:
        from vllm_mlx.telemetry import emit as _telemetry_emit

        _telemetry_emit.fire_session_end_hook()
    except Exception:
        # Telemetry must never crash the shutdown path. Logged at
        # debug only -- this is best-effort cleanup.
        logger.debug("telemetry session_end hook failed (non-fatal)")


app = FastAPI(
    title="Rapid-MLX API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.6.0",
    lifespan=lifespan,
)

# SECURITY: bound the request body of /v1/audio/transcriptions at the
# ASGI layer so honest-Content-Length DoS attempts are rejected before
# Starlette's multipart parser drains the receive channel and spools
# the body to disk. See vllm_mlx/routes/audio.py for the rationale.
from .routes.audio import install_audio_body_limit_middleware  # noqa: E402

install_audio_body_limit_middleware(app)

# SECURITY: video multipart auth/body cap must run before Starlette spools an
# UploadFile. It also owns the 21 MiB request allowance for the 20 MiB file cap.
from .routes.video import install_video_body_limit_middleware  # noqa: E402

install_video_body_limit_middleware(app)

from .routes.images import install_image_body_limit_middleware  # noqa: E402

install_image_body_limit_middleware(app)

# SECURITY: blanket request-body size cap across all /v1/* routes.
# Defends against the DoS pattern documented in rapid-desktop#273 / #463
# where a 10–100 MB JSON body silently runs full prefill (~60–90 s on a
# 27B alias) before the client times out. See middleware/body_size.py
# for the design rationale, the path-scoping (skips
# ``/v1/audio/transcriptions`` so the multipart-aware 25 MB cap upstream
# is not trampled by the generic 8 MiB JSON cap), and the limit lookup
# (ServerConfig.max_request_bytes, overridable via --max-request-bytes /
# RAPID_MLX_MAX_REQUEST_BYTES).
# SECURITY: blanket request-body JSON nesting-depth cap across all
# /v1/* JSON routes. Defends against the D-DEEP-JSON DoS pattern where
# a ~10 KB body of ``{"a":{"a":…}}`` 1000 levels deep blew the Python
# recursion limit inside Pydantic's body validator and surfaced as
# HTTP 500 on every body-binding route (chat / completions /
# embeddings / messages / responses). See middleware/body_depth.py
# for the design rationale; the cap is read from
# ``RAPID_MLX_MAX_BODY_DEPTH`` per request (default 64) so a test
# fixture mutating the env takes effect immediately. Installed BEFORE
# the size cap so the size cap ends up OUTERMOST at request time
# (Starlette stacks middleware in reverse install order). That way a
# 100 MB body gets bounced for size before this middleware ever sees
# it — the depth gate is only reached for bodies that already pass
# the size cap.
from .middleware.body_depth import install_request_body_depth_middleware  # noqa: E402

install_request_body_depth_middleware(app)

from .middleware.body_size import install_request_body_limit_middleware  # noqa: E402

install_request_body_limit_middleware(app)

# R8-H6: ASGI fast-path for ``GET /healthz`` + ``GET /livez``.
#
# Stack ordering note (codex r3 NIT clarification): Starlette stacks
# user middleware in REVERSE install order — last install runs FIRST
# per request. The fast-path is installed here (after the body-size +
# body-depth + audio middlewares), so it sits OUTSIDE those three at
# request time. ``cli.configure_cors_from_env`` runs LATER (at boot,
# after this module loads) and ALSO uses ``app.add_middleware``, so
# CORS lands OUTSIDE the fast-path when it's enabled. That ordering
# is intentional + acceptable:
#
#   * Starlette's CORSMiddleware short-circuits with a single
#     ``await self.app(scope, receive, send)`` when the request
#     carries no ``Origin`` header — and k8s / supervisord / Docker /
#     systemd probes do not send Origin. So for the probe slice that
#     this fast-path targets, the CORS layer is effectively a
#     1-microsecond pass-through; the fast-path still answers the
#     probe without touching the router, dependency graph, or
#     response serialization.
#   * Browser cross-origin hits (which DO carry Origin) take the
#     fall-through path inside the fast-path (``_has_origin`` returns
#     True), reach CORSMiddleware on the way back out via the inner
#     app's response, and ship with the correct ACAO header.
#
# Among the body-size + body-depth + audio middlewares, the fast-path
# IS outermost — and those middlewares already early-return for GET
# requests anyway, so the probe path was never paying for them.
from .middleware.probe_fastpath import install_probe_fastpath_middleware  # noqa: E402

install_probe_fastpath_middleware(app)

# CORS configuration — configurable via --cors-origins CLI flag and the
# ``RAPID_MLX_CORS_*`` env-var family (F-090 / F-091). The previous default
# registered CORSMiddleware with ``allow_origins=["*"]`` and
# ``allow_methods=["*"]`` (DELETE/GET/HEAD/OPTIONS/PATCH/POST/PUT) for
# every request, which let any browser-side attacker make authenticated
# requests to ``/v1/chat/completions`` if the user had an open tab. The
# new default is **no CORS at all**: operators opt in by setting
# ``RAPID_MLX_CORS_ALLOW_ORIGINS`` (or ``--cors-origins`` for ad-hoc use).
# A wildcard is still permitted for back-compat but logs a startup WARNING
# so the operator notices.

# Default method/header allowlists used when CORS is enabled. ``POST,GET,
# OPTIONS`` matches every route this server actually exposes — DELETE /
# PATCH / PUT are not routed at all (closes F-091's over-broad ACAM).
_DEFAULT_CORS_METHODS: tuple[str, ...] = ("POST", "GET", "OPTIONS")
_DEFAULT_CORS_HEADERS: tuple[str, ...] = (
    "Content-Type",
    "Authorization",
    "X-Rapid-MLX-Internal",
)
_DEFAULT_CORS_MAX_AGE: int = 3600


@dataclass(frozen=True)
class ResolvedCORSPolicy:
    """The fully resolved CORS policy shared by all HTTP server modes."""

    origins: tuple[str, ...]
    methods: tuple[str, ...]
    headers: tuple[str, ...]
    max_age: int
    allow_credentials: bool


_last_resolved_cors_policy: ResolvedCORSPolicy | None = None


def get_resolved_cors_policy() -> ResolvedCORSPolicy | None:
    """Return the policy configured during this process's CLI startup."""
    return _last_resolved_cors_policy


class _SpecAlignedCORSMiddleware(CORSMiddleware):
    """``CORSMiddleware`` whose preflight rejection is spec-aligned (L-02).

    Upstream Starlette returns ``400 Bad Request`` with body
    ``"Disallowed CORS …"`` when a preflight ``OPTIONS`` fails any of the
    origin / method / headers checks. The upstream comment even concedes
    the 400 is an opinionated debugging aid:

        # We don't strictly need to use 400 responses here, since its up
        # to the browser to enforce the CORS policy, but its more
        # informative if we do.

    The 400 is noisy in real-world devtools (the spec only requires that
    the response omit ``Access-Control-Allow-Origin`` — the browser then
    blocks the real request). It also surprises authenticated reverse
    proxies that interpret a 4xx preflight as "the origin is wrong".

    This subclass returns ``200 OK`` with **no** ``Access-Control-Allow-Origin``
    header and ``Vary: Origin`` (so caches that key on Origin don't bleed
    across origins). Browsers still block the request because ACAO is
    absent; devtools shows the missing-header signal that operators
    expect when CORS denies their origin.

    The fail-closed empty-CSV path (#758 ``3da8230``) is unaffected: that
    branch never registers any middleware at all, so the preflight ``OPTIONS``
    still 405s (no allowed method on the route). Only the env-locked /
    explicit-allowlist mismatch surface flips from 400 to 200.
    """

    def preflight_response(self, request_headers):  # type: ignore[override]
        # Re-use upstream's diagnostic logic to compute "is this preflight
        # actually allowed?" — it builds the right headers dict, mutates
        # ``Access-Control-Allow-Origin`` only when the origin is in the
        # allowlist, and accumulates a ``failures`` list. We just trade
        # the 400 envelope on failure for a 200 + ``Vary: Origin``.
        response = super().preflight_response(request_headers)
        if response.status_code == 200:
            return response
        # Strip ``Access-Control-Allow-Origin`` (upstream may have written
        # it on a partial-allow path — defensive; current upstream never
        # writes it when the origin is the failing dimension, but a
        # future patch could broaden the partial-allow path) and pin
        # ``Vary: Origin`` so caches don't reuse this 200 across origins.
        # ``response.headers`` is a starlette ``MutableHeaders`` whose
        # ``items()`` repeats duplicate keys (e.g. two ``vary`` rows).
        # We build a single-row dict so the spec-aligned response doesn't
        # carry ``Vary: Origin, Origin`` (upstream already set ``Vary:
        # Origin`` in ``preflight_headers`` for non-wildcard configs).
        headers: dict[str, str] = {}
        for k, v in response.headers.items():
            lk = k.lower()
            if lk == "access-control-allow-origin":
                continue
            # The upstream 400 body is longer than our constant ``"OK"``.
            # Carrying its Content-Length into PlainTextResponse makes the
            # wire response claim bytes that never arrive: curl exits 18 and
            # strict HTTP clients raise IncompleteRead. Let PlainTextResponse
            # calculate the length of the replacement body instead.
            if lk == "content-length":
                continue
            if lk == "vary":
                continue  # canonicalized below
            headers[k] = v
        headers["Vary"] = "Origin"
        # Body is a constant ``"OK"`` so a curious operator who hits the
        # preflight by hand (``curl -X OPTIONS``) sees a non-empty 200
        # rather than a confusing blank response. The browser never
        # surfaces the preflight body to JS regardless — what makes the
        # browser block the real request is the missing
        # ``Access-Control-Allow-Origin`` header. Codex round-1 NIT
        # flagged the prior "200 with empty body" comment as diverging
        # from this body shape; pinning ``"OK"`` here and in the
        # regression suite keeps code, comment, and tests aligned.
        from starlette.responses import PlainTextResponse

        return PlainTextResponse("OK", status_code=200, headers=headers)


def configure_cors(
    origins: list[str],
    *,
    methods: list[str] | None = None,
    headers: list[str] | None = None,
    max_age: int | None = None,
    allow_credentials: bool | None = None,
) -> None:
    """Register the CORS middleware with the given allowlist.

    Backwards-compatible signature: callers that pass only ``origins``
    (tests, ``share`` CLI, the dflash speculative server) still get the
    *legacy* wide-open ``allow_methods=["*"]`` / ``allow_headers=["*"]``
    behavior — codex round-2 BLOCKING flagged that silently narrowing
    these on the single-arg path would break existing browser clients
    that send headers like ``OpenAI-Organization`` or
    ``X-Requested-With`` (preflight 200 → real-request fails because the
    header isn't on the allowlist). The F-091 narrowing only kicks in
    on the env-aware path (``configure_cors_from_env``) which passes
    explicit ``methods=`` / ``headers=`` lists — new callers see the
    restrictive default, legacy callers stay wide-open.

    When the wildcard ``*`` is present, ``allow_credentials`` is forced to
    False to comply with the Fetch standard — browsers reject responses
    that combine ``Access-Control-Allow-Origin: *`` with
    ``Access-Control-Allow-Credentials: true``, so the previous default
    silently broke any cross-origin client that sent cookies or
    Authorization headers.

    NOTE: this function unconditionally registers the middleware on the
    module-level ``app``. ``configure_cors_from_env`` is the production
    entry-point — it skips registration entirely when no origins are
    configured, which is what closes F-090 at the default-deny layer.
    """
    if not origins:
        # Defensive: callers should not invoke ``configure_cors`` with an
        # empty list (production goes through ``configure_cors_from_env``
        # which short-circuits earlier). Bail rather than register a
        # middleware that would deny everything but still leak the
        # ``Access-Control-*`` header surface.
        return
    wildcard = "*" in origins
    if allow_credentials is None:
        allow_credentials = not wildcard
    elif allow_credentials and wildcard:
        # Operator explicitly asked for credentials AND wildcard. The
        # Fetch spec rejects this combination; force-disable credentials
        # so the response stays valid and log a warning so the operator
        # notices. ``%s`` interpolation (rather than baking the literal
        # ``RAPID_MLX_…`` env-var name into the format string) avoids
        # the ``tests/test_no_out_of_band_routing.py`` constant-scan
        # false-positive — same trick used by the body-receive timeout
        # / SSE keepalive blocks in cli.py.
        logger.warning(
            "%s requested with a wildcard origin is invalid per the "
            "Fetch spec; forcing allow_credentials=False",
            "RAPID_MLX_CORS_ALLOW_CREDENTIALS",
        )
        allow_credentials = False
    # ``methods=None`` and ``headers=None`` mean "back-compat single-arg
    # caller" — preserve the legacy ``["*"]`` so existing clients keep
    # working. ``configure_cors_from_env`` always passes explicit lists,
    # so it gets the F-091 narrowing.
    #
    # L-02: the subclass returns 200 + ``Vary: Origin`` (no
    # ``Access-Control-Allow-Origin``) instead of 400 ``Disallowed CORS
    # …`` on origin/method/headers mismatch. Browsers still block the
    # real request because ACAO is absent; devtools sees the missing
    # header instead of a cryptic 400 envelope.
    app.add_middleware(
        _SpecAlignedCORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=list(methods) if methods is not None else ["*"],
        allow_headers=list(headers) if headers is not None else ["*"],
        max_age=max_age if max_age is not None else _DEFAULT_CORS_MAX_AGE,
    )


def _parse_csv(value: str) -> list[str]:
    """Split a comma-separated env-var value, trimming whitespace and
    dropping empty entries. Used by ``configure_cors_from_env`` so a
    trailing comma or stray space doesn't accidentally register an empty
    origin (which CORSMiddleware would silently never match)."""
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_trusted_hosts(cli_hosts: list[str] | None = None) -> list[str]:
    """OPT-IN Host-header allowlist (DNS-rebinding / Host-header-spoofing
    hardening) via Starlette's ``TrustedHostMiddleware``.

    Resolution:
      1. ``--trusted-hosts`` CLI flag (comma-separated) — takes precedence.
      2. ``RAPID_MLX_TRUSTED_HOSTS`` env var (comma-separated).
      3. Unset / empty → middleware is NOT registered. This is the default:
         restricting the Host header would break ``rapid-mlx share`` (which
         forwards the public-facing Host header into the local server) and
         LAN access via machine hostname, so an operator opts in deliberately.

    ``allowed_hosts`` (Starlette) values cross-match the request ``Host``
    header against glob patterns; ``*`` and ``localhost``/``127.0.0.1`` are
    typical. A request whose Host matches nothing is rejected with 400.
    """
    hosts: list[str] = []
    if cli_hosts is not None:
        # argparse's nargs="+" accepts both ``a b`` and values users commonly
        # write as ``a,b``. Normalize each entry so CLI and env semantics match.
        hosts = [host for entry in cli_hosts for host in _parse_csv(entry)]
    else:
        env_raw = os.environ.get("RAPID_MLX_TRUSTED_HOSTS")
        if env_raw:
            hosts = _parse_csv(env_raw)
    if not hosts:
        return []
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    logger.info(
        "TrustedHostMiddleware enabled (allowed_hosts=%s): requests with a "
        "non-matching Host header are rejected.",
        hosts,
    )
    return hosts


def configure_cors_from_env(
    cli_origins: list[str] | None = None,
) -> list[str]:
    """Resolve CORS configuration from CLI args + env vars and conditionally
    register the middleware.

    Resolution order:
      1. ``cli_origins`` (CLI ``--cors-origins`` flag) — when supplied,
         takes precedence over the env var (matches the pattern used by
         ``--max-request-bytes`` vs ``RAPID_MLX_MAX_REQUEST_BYTES``).
      2. ``RAPID_MLX_CORS_ALLOW_ORIGINS`` env var (comma-separated).
      3. Unset / empty → CORS middleware is NOT registered. Cross-origin
         requests get no ``Access-Control-Allow-Origin`` header and the
         preflight ``OPTIONS`` returns 405 (no leak of allowed methods).
         This is the production-friendly default (F-090).

    Method / header / max-age / credentials overrides come from the rest
    of the ``RAPID_MLX_CORS_*`` family; see ``vllm_mlx/cli.py``.

    Returns the resolved origin list (empty list when CORS is disabled).
    """
    global _last_resolved_cors_policy

    # ``came_from_cli`` discriminates the two compat tiers (codex round-3
    # BLOCKING). The legacy ``--cors-origins`` CLI path used to imply
    # ``allow_headers=["*"]`` / ``allow_methods=["*"]``; existing browser
    # clients send custom headers like ``OpenAI-Organization`` and
    # ``X-Requested-With`` that would now fail preflight if we silently
    # narrowed those defaults. The env-driven path
    # (``RAPID_MLX_CORS_ALLOW_ORIGINS``) is brand-new in this PR — it
    # gets the restrictive default (closes F-091 by default). Operators
    # on either path can still pin the methods/headers explicitly via
    # ``RAPID_MLX_CORS_ALLOW_METHODS`` / ``_HEADERS``.
    origins: list[str] = []
    came_from_cli = False
    came_from_default = False
    env_present_but_empty = False
    if cli_origins:
        origins = list(cli_origins)
        came_from_cli = True
    else:
        # Codex round-2 BLOCKING (#758): distinguish "env var absent" from
        # "env var present but parsed empty". The friendly default-wildcard
        # is for operators who never set the var (single-machine local
        # dev). An operator who DID set the var to a templating-broken
        # value like ``" , ,, "`` clearly intended a real allowlist — the
        # safest interpretation is the literal empty list (no CORS, fail
        # closed) plus a startup WARNING so the typo is visible. Falling
        # through to wildcard would silently fail open for a deployment
        # bug, which is the failure mode codex was flagging.
        env_raw = os.environ.get("RAPID_MLX_CORS_ALLOW_ORIGINS")
        if env_raw is not None:
            parsed = _parse_csv(env_raw)
            if parsed:
                origins = parsed
            else:
                env_present_but_empty = True
                # Format via %s/%r placeholders so the env-var name only
                # appears in the args tuple (test_no_out_of_band_routing
                # treats inline ``RAPID_MLX_<X>=...`` literals as routing
                # references — same shape as the methods/headers warnings
                # below).
                logger.warning(
                    "%s=%r parsed to an empty list (whitespace / "
                    "trailing commas only). Treating as fail-closed "
                    "(no CORS middleware) so a deployment templating "
                    "bug is visible. Unset the env var entirely to use "
                    "the friendly default wildcard, or set it to a real "
                    "comma-separated origin list.",
                    "RAPID_MLX_CORS_ALLOW_ORIGINS",
                    env_raw,
                )

    if not origins and not env_present_but_empty:
        # Default-allow wildcard for friendly single-machine UX. rapid-mlx
        # is primarily run locally — defaulting to deny would break any
        # browser-based frontend ("CORS error" in the console) without an
        # obvious server-side signal. Operators on multi-tenant or
        # production deployments lock down via
        # ``RAPID_MLX_CORS_ALLOW_ORIGINS=https://your.app`` (the existing
        # env-var family still applies).
        origins = ["*"]
        came_from_default = True
        logger.info(
            "CORS allow-origin defaulting to wildcard '*' (no "
            "RAPID_MLX_CORS_ALLOW_ORIGINS set). Set the env var to an "
            "explicit origin list (e.g. "
            "'https://chat.openai.com,https://claude.ai') to lock down "
            "for production / multi-tenant deployments."
        )

    if "*" in origins and not came_from_default:
        logger.warning(
            "CORS allow-origin set to wildcard '*' — any origin can call this "
            "server from a browser. Set RAPID_MLX_CORS_ALLOW_ORIGINS to an "
            "explicit origin list for production deployments."
        )

    # Resolve method / header / max-age / credentials overrides.
    #
    # Codex round-1 BLOCKING: distinguish ``env unset`` from ``env set to
    # an all-whitespace / all-empty CSV``. If we treated both as "use
    # default", an operator typo like ``RAPID_MLX_CORS_ALLOW_METHODS=" , "``
    # would silently broaden the surface to the default POST/GET/OPTIONS
    # instead of narrowing it. The defensive shape is: ``env present but
    # empty after parse`` → log a WARNING and fall back to the default,
    # so the operator sees the typo in the startup log rather than
    # discovering it via a Sentry alert later.
    #
    # Codex round-3 BLOCKING: when ``came_from_cli`` is True and the env
    # override is unset, the default for methods/headers is the legacy
    # wide-open ``["*"]`` — not the restrictive F-091 default — so the
    # documented CLI back-compat path doesn't silently break browser
    # clients that send custom headers (``OpenAI-Organization`` etc.).
    methods_env = os.environ.get("RAPID_MLX_CORS_ALLOW_METHODS")
    if methods_env is None:
        methods = ["*"] if came_from_cli else list(_DEFAULT_CORS_METHODS)
    else:
        methods = _parse_csv(methods_env)
        if not methods:
            fallback_methods = ["*"] if came_from_cli else list(_DEFAULT_CORS_METHODS)
            logger.warning(
                "%s=%r parsed to an empty list (whitespace / trailing "
                "commas only); falling back to %s. Set the env var to a "
                "real comma-separated method list, or unset it entirely "
                "to use the default.",
                "RAPID_MLX_CORS_ALLOW_METHODS",
                methods_env,
                fallback_methods,
            )
            methods = fallback_methods

    headers_env = os.environ.get("RAPID_MLX_CORS_ALLOW_HEADERS")
    if headers_env is None:
        headers = ["*"] if came_from_cli else list(_DEFAULT_CORS_HEADERS)
    else:
        headers = _parse_csv(headers_env)
        if not headers:
            fallback_headers = ["*"] if came_from_cli else list(_DEFAULT_CORS_HEADERS)
            logger.warning(
                "%s=%r parsed to an empty list (whitespace / trailing "
                "commas only); falling back to %s. Set the env var to a "
                "real comma-separated header list, or unset it entirely "
                "to use the default.",
                "RAPID_MLX_CORS_ALLOW_HEADERS",
                headers_env,
                fallback_headers,
            )
            headers = fallback_headers

    max_age_env = os.environ.get("RAPID_MLX_CORS_MAX_AGE", "").strip()
    max_age = _DEFAULT_CORS_MAX_AGE
    if max_age_env:
        try:
            max_age = max(0, int(max_age_env))
        except ValueError:
            logger.warning(
                "%s=%r is not an integer; falling back to the %d s default",
                "RAPID_MLX_CORS_MAX_AGE",
                max_age_env,
                _DEFAULT_CORS_MAX_AGE,
            )

    # Codex round-1 NIT: the documented default is ``False`` (matching
    # the security-correct default-deny stance of the rest of the
    # ``RAPID_MLX_CORS_*`` family), but the legacy ``configure_cors``
    # back-compat path defaults to ``True`` for any non-wildcard origin
    # (Fetch-spec-correct but at odds with the documentation). Resolve by
    # making the default explicit here: env unset → False. Operators who
    # need cookies / Authorization auto-forwarded must opt in by setting
    # ``RAPID_MLX_CORS_ALLOW_CREDENTIALS=true``. Existing
    # ``configure_cors(origins)`` callers in tests / share / dflash still
    # see the legacy behavior — those callers don't go through this
    # resolver.
    creds_env = os.environ.get("RAPID_MLX_CORS_ALLOW_CREDENTIALS", "").strip().lower()
    allow_credentials: bool = False
    if creds_env:
        if creds_env in ("1", "true", "yes", "on"):
            allow_credentials = True
        elif creds_env in ("0", "false", "no", "off"):
            allow_credentials = False
        else:
            logger.warning(
                "%s=%r is not a boolean; falling back to the False default",
                "RAPID_MLX_CORS_ALLOW_CREDENTIALS",
                creds_env,
            )

    # Keep the policy snapshot byte-for-byte equivalent to the middleware
    # actually installed below. In particular, a credentialed wildcard is
    # invalid under Fetch and must never reappear on DFlash via its separate
    # FastAPI application.
    if "*" in origins and allow_credentials:
        logger.warning(
            "%s requested with a wildcard origin is invalid per the "
            "Fetch spec; forcing allow_credentials=False",
            "RAPID_MLX_CORS_ALLOW_CREDENTIALS",
        )
        allow_credentials = False

    # Fail-closed path: ``RAPID_MLX_CORS_ALLOW_ORIGINS`` was set but
    # parsed to an empty list (operator-controlled typo). Don't register
    # CORSMiddleware — that's the visible signal the WARNING above
    # promises. Returning ``[]`` here also keeps the legacy ``configure_cors``
    # back-compat stub callable from tests that monkeypatch a 1-arg
    # lambda — we never call ``configure_cors(...)`` on the fail-closed
    # path, so the stub's signature doesn't matter.
    if not origins:
        _last_resolved_cors_policy = None
        return []

    _last_resolved_cors_policy = ResolvedCORSPolicy(
        origins=tuple(origins),
        methods=tuple(methods),
        headers=tuple(headers),
        max_age=max_age,
        allow_credentials=allow_credentials,
    )
    configure_cors(
        origins,
        methods=methods,
        headers=headers,
        max_age=max_age,
        allow_credentials=allow_credentials,
    )
    return origins


# Auth and rate limiting — moved to middleware/auth.py
from .middleware.auth import (  # noqa: E402
    RateLimiter,  # noqa: F401
    check_rate_limit,  # noqa: F401
    configure_rate_limiter,  # noqa: F401
    verify_api_key,  # noqa: F401
)
from .middleware.auth import (
    rate_limiter as _rate_limiter,  # noqa: F401 — configured in main()
)

# ── Wire the unified exception handlers onto the production app ─────
#
# All handler bodies live in ``vllm_mlx.middleware.exception_handlers``
# (no heavy imports) so isolated route tests can install the identical
# wiring on a stub FastAPI app without dragging the engine stack into
# the fixture. Production delegates here for:
#
# * ``StarletteHTTPException`` → wraps detail in the OpenAI-shaped
#   envelope (and passes structured ``{"error": {...}}`` detail through
#   unchanged — the ``context_length_exceeded`` escape hatch still
#   works). Closes F-013 / F-094 leakage at the envelope layer.
# * ``json.JSONDecodeError`` → 400 with OpenAI-shaped envelope
#   (closes F-161 / F-162 — ``await request.json()`` failures were
#   hitting the generic 500 path before).
# * ``RequestValidationError`` → 400 with sanitized message
#   (strips ``detail[*].input`` echo from F-094/F-104 and the
#   pydantic.dev URL from F-163).
# * ``Exception`` → 500 ``Internal server error`` with no message leak.
install_exception_handlers(app)


def _detect_native_tool_support() -> bool:
    """
    Detect if the active tool parser supports native tool format.

    Native format means role="tool" messages and tool_calls fields
    are preserved instead of being converted to text.

    Returns:
        True if native format should be preserved
    """
    cfg = get_config()
    if not cfg.enable_auto_tool_choice or not cfg.tool_call_parser:
        return False

    try:
        parser_cls = ToolParserManager.get_tool_parser(cfg.tool_call_parser)
        return parser_cls.supports_native_format()
    except KeyError:
        # Parser not found - this is a configuration error, log as error
        logger.error(
            f"Tool parser '{cfg.tool_call_parser}' not found. "
            f"Available parsers: {ToolParserManager.list_registered()}"
        )
        return False
    except Exception as e:
        # Unexpected error during detection
        logger.warning(f"Failed to detect native tool support: {e}")
        return False


def load_embedding_model(
    model_name: str | None,
    *,
    lock: bool = False,
    reuse_existing: bool = True,
    max_length: int | str | None = None,
    overflow_policy: str | None = None,
) -> None:
    """Load or reuse the embedding model engine when configured.

    ``max_length`` / ``overflow_policy`` (issue #1381) are remembered in
    module globals when provided (the CLI passes them once at startup), so
    route-triggered lazy loads — which call this without them — build the
    engine with the same operator-configured limits.
    """
    global _embedding_engine, _embedding_model_locked
    global _embedding_max_length, _embedding_overflow_policy

    if not model_name:
        return

    if max_length is not None:
        _embedding_max_length = max_length
    if overflow_policy is not None:
        _embedding_overflow_policy = overflow_policy

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(
        model_name,
        max_length=_embedding_max_length,
        overflow_policy=_embedding_overflow_policy,
    )
    _embedding_engine.load()

    # Sync into config for route modules
    cfg = get_config()
    cfg.embedding_engine = _embedding_engine
    cfg.embedding_model_locked = _embedding_model_locked


def _ensure_routing_config(model_name: str) -> None:
    """Materialize the checkpoint config on disk before the offline routing
    probes run, and FAIL FAST if it cannot be materialized.

    ``resolve_serving_lane`` reads the checkpoint config from the local cache
    to decide the MLLM-vs-text lane. On a first-time uncached remote startup
    that config does not exist yet, so a hybrid VLM would probe "not hybrid"
    and get routed into the MLLM engine that cannot serve it (#352 dogfood
    P1-②).

    Contract: on a normal return the routing probes have real config evidence.
    - Already materialized (cached repo, a prior prefetch, or a local dir that
      ships a config) -> nothing to do; the probe is reliable. Fully offline
      and cheap, so warm starts and the unit suite never trigger a download.
    - Otherwise prefetch via the same canonical mirror/HF fetch the CLI uses,
      then VERIFY the config actually landed. If it did not, raise an
      actionable error instead of letting the caller route on a guess — a
      silent miss here misroutes a hybrid VLM into the crashing MLLM lane.

    A hard disk-space gate (``SystemExit``) from the prefetch is an intentional
    fail-fast and propagates unchanged. Module-level so tests can substitute
    the prefetch to simulate "config appears only after download".
    """
    from .model_metadata import read_model_metadata

    # Config already readable (warm cache / local checkpoint dir) -> the routing
    # probe has real evidence; skip the prefetch so warm starts and the unit
    # suite never download.
    if read_model_metadata(model_name) is not None:
        return
    # A local path the user pointed us at: trust their files. If a config is
    # genuinely absent the engine's own loader surfaces it with its own
    # message; we must not try to "download" a filesystem path.
    if os.path.exists(model_name):
        return

    _prefetch_exc: Exception | None = None
    try:
        from .cli import _ensure_model_downloaded

        _ensure_model_downloaded(model_name)
    except SystemExit:
        # ``_ensure_model_downloaded`` may exit(1) on a hard disk-space gate —
        # that is an intentional fail-fast; let it propagate.
        raise
    except Exception as _e:  # noqa: BLE001 — preserved below, not swallowed
        _prefetch_exc = _e
        logger.debug("routing-config prefetch raised (will re-verify): %r", _e)

    # VERIFY the prefetch actually put the config on disk. If it did not, the
    # routing probes would fall back to a guess and could misroute a hybrid VLM
    # into the MLLM engine that cannot serve it (#352). Fail fast with an
    # actionable message instead of silently guess-routing — and chain the
    # original prefetch error so its real cause (auth / network / 404) is not
    # lost.
    if read_model_metadata(model_name) is None:
        raise RuntimeError(
            f"Could not materialize the checkpoint config for {model_name!r} "
            "before selecting the serving lane. The MLLM-vs-text routing "
            "decision needs the model's config.json on disk; without it a "
            "hybrid VLM can be misrouted into the multimodal engine that "
            "cannot serve it (GH #352). Check network / HuggingFace access / "
            "disk space and retry, or pass --no-mllm to force the text-only "
            "lane (or --mllm to force the multimodal lane)."
        ) from _prefetch_exc
    if _prefetch_exc is not None:
        # Config landed, so we CAN resolve the lane — but the prefetch still
        # errored (e.g. a partial download: config.json present, weights
        # incomplete, or a late auth/network fault). Don't discard that cause;
        # surface it at WARNING so a later weight-load failure is attributable
        # instead of appearing as an unrelated error downstream.
        logger.warning(
            "routing-config prefetch for %r reported an error even though its "
            "config materialized; the model may be partially downloaded and "
            "fail to load its weights later. Original error: %r",
            model_name,
            _prefetch_exc,
        )


def load_model(
    model_name: str,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int | None = None,
    force_mllm: bool = False,
    gpu_memory_utilization: float = 0.90,
    prefill_step_size: int | None = None,
    *,
    served_model_name: str | None = None,
    mtp: bool = False,
    max_tokens_is_explicit: bool | None = None,
    force_text: bool = False,
    force_hybrid: bool = False,
    no_hybrid: bool = False,
    force_spec_decode: bool = False,
    no_spec_decode: bool = False,
    force_openai_harmony_streaming: bool = False,
    no_openai_harmony_streaming: bool = False,
    enable_disk_stream: bool = False,
    disk_stream_cache_gb: float = 1.0,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        scheduler_config: Scheduler config for BatchedEngine
        stream_interval: Tokens to batch before streaming
        max_tokens: Default max tokens for generation. ``None`` uses the
            programmatic default.
        max_tokens_is_explicit: True when max_tokens came from an explicit
            operator setting such as ``serve --max-tokens``. When omitted,
            programmatic callers that pass ``max_tokens`` are treated as
            explicit while callers that omit it keep the implicit default.
        force_mllm: Force loading as MLLM even if not auto-detected
        gpu_memory_utilization: Fraction of device memory (0.0-1.0, default 0.90)
        prefill_step_size: DEPRECATED — pass via
            ``scheduler_config.prefill_step_size`` instead. Pre-0.6.52 this
            parameter was accepted but silently ignored (the value never
            reached BatchedEngine — root cause of #400). Kept here for
            back-compat with external callers; if provided it is translated
            into ``scheduler_config.prefill_step_size`` and a DeprecationWarning
            is emitted. Will be removed in a future release.
        mtp: DEPRECATED compatibility alias. ``mtp=True`` is translated to
            ``scheduler_config.spec_decode == "mtp"`` so older
            ``load_model(..., mtp=True)`` callers still opt into MTP while the
            public runtime moves to ``--speculative-config`` /
            ``SchedulerConfig(spec_decode="mtp")``.
        force_text: Keyword-only. Force loading as text-only LLM even when
            auto-detection would route as MLLM. Escape hatch for incomplete
            vision-tower checkpoints (#393) and text-only forks of multimodal
            architectures whose config.json still declares vision_config.
            Mutually exclusive with ``force_mllm``. Keyword-only to avoid
            shifting positional args for existing callers.
        force_hybrid / no_hybrid: Keyword-only. SOP §10 escape hatches
            for ``ModelConfig.is_hybrid`` auto-detection. Forwarded to
            ``BatchedEngine`` → ``EngineConfig``. Mutually exclusive.
        force_spec_decode / no_spec_decode: Keyword-only. SOP §10
            escape hatches for ``ModelConfig.supports_spec_decode``
            auto-detection. Mutually exclusive.
        enable_disk_stream / disk_stream_cache_gb: Keyword-only.
            ``--disk-stream`` (PRD-rapid-mlx-integration.md). Forwarded
            to ``BatchedEngine``, which loads lazily and installs
            ``vllm_mlx.disk_stream_patch`` in ``_start_llm`` before the
            model reaches ``AsyncEngineCore``. Default False keeps every
            existing caller's behavior unchanged.
    """
    max_tokens_was_supplied = max_tokens is not None
    if max_tokens is None:
        max_tokens = 32768
    if max_tokens_is_explicit is None:
        max_tokens_is_explicit = max_tokens_was_supplied

    if mtp:
        import warnings

        from .scheduler import SchedulerConfig

        existing_spec_decode = (
            getattr(scheduler_config, "spec_decode", "none")
            if scheduler_config is not None
            else "none"
        )
        if existing_spec_decode not in ("none", "mtp"):
            raise ValueError(
                "load_model(mtp=True) conflicts with "
                f"scheduler_config.spec_decode={existing_spec_decode!r}; "
                "pass only one speculative decoding method."
            )
        if scheduler_config is not None and getattr(
            scheduler_config, "enable_suffix_decoding", False
        ):
            raise ValueError(
                "load_model(mtp=True) conflicts with "
                "scheduler_config.enable_suffix_decoding=True; pass only one "
                "speculative decoding method."
            )
        if (
            scheduler_config is not None
            and (getattr(scheduler_config, "dflash_drafter_path", "") or "").strip()
        ):
            raise ValueError(
                "load_model(mtp=True) conflicts with "
                "scheduler_config.dflash_drafter_path; pass only one "
                "speculative decoding method."
            )
        if scheduler_config is not None and getattr(
            scheduler_config, "mtp_optimistic", False
        ):
            # Unified spec-decode interface (PR #1050): the vendored MTP
            # installer does not honour ``mtp_optimistic``. Direct mutation
            # of scheduler_config below would bypass ``__post_init__``, so
            # enforce the same reject rule here to avoid silent drift.
            raise ValueError(
                "load_model(mtp=True) cannot be combined with "
                "scheduler_config.mtp_optimistic=True — mtp_optimistic "
                "is not supported under the unified spec-decode interface."
            )
        warnings.warn(
            "load_model(mtp=True) is deprecated; pass "
            "SchedulerConfig(spec_decode='mtp') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if scheduler_config is None:
            scheduler_config = SchedulerConfig(enable_mtp=True)
        elif existing_spec_decode == "none":
            scheduler_config.enable_mtp = True
            scheduler_config.spec_decode = "mtp"
            scheduler_config.mtp_max_k = max(
                1, int(getattr(scheduler_config, "mtp_num_draft_tokens", 1))
            )

    if prefill_step_size is not None:
        import warnings

        from .scheduler import SchedulerConfig

        warnings.warn(
            "load_model(prefill_step_size=...) is deprecated; "
            "pass via scheduler_config.prefill_step_size instead. "
            "Pre-0.6.52 this kwarg was silently ignored (#400).",
            DeprecationWarning,
            stacklevel=2,
        )
        if scheduler_config is None:
            scheduler_config = SchedulerConfig(prefill_step_size=prefill_step_size)
        else:
            scheduler_config.prefill_step_size = prefill_step_size

    global \
        _engine, \
        _model_name, \
        _model_path, \
        _default_max_tokens, \
        _default_max_tokens_is_explicit, \
        _tool_parser_instance, \
        _alias_recommended_sampling, \
        _generation_config_sampling

    _default_max_tokens = max_tokens
    _default_max_tokens_is_explicit = max_tokens_is_explicit
    _model_path = model_name
    _model_name = served_model_name or model_name
    _tool_parser_instance = None

    # Populate the sampling overlays now that we know which model we're
    # serving. Both are best-effort — an alias without curated sampling
    # or a model missing generation_config.json simply contributes an
    # empty layer to the cascade in service/helpers.py.
    from .model_aliases import resolve_profile
    from .utils.generation_config import load_generation_config_sampling

    _alias_recommended_sampling = None
    # resolve_profile handles both alias-name and HF-path lookups, so a
    # single call suffices regardless of which form load_model was passed.
    _profile = resolve_profile(_model_alias or model_name)
    if _profile is not None and _profile.recommended_sampling:
        _alias_recommended_sampling = dict(_profile.recommended_sampling)

    # Alias-declared ``is_text_only`` → the registered ``force_text``
    # routing kwarg. When an alias profile pins ``is_text_only=True``
    # (e.g. Ternary-Bonsai-27B: a multimodal-config checkpoint whose
    # vision path our mlx-vlm loader can't drive, but whose text tower is
    # coherent via mlx-lm's qwen3_5), fold that into the effective
    # ``force_text`` so the text-only mlx-lm lane is chosen with no CLI
    # flag. This is NOT a new routing surface: ``is_text_only`` is a
    # state description (parallel to ``is_hybrid`` / ``is_moe``) and it
    # feeds the SAME ``force_text`` kwarg already registered in
    # ``AUTO_ROUTING_FLAG_PAIRS`` (``--mllm`` / ``--no-mllm``, #393).
    #
    # Set it UNCONDITIONALLY (do NOT gate on ``not force_mllm``): an
    # explicit ``--mllm`` on such an alias must then collide with this
    # ``force_text=True`` at the ``force_mllm and force_text``
    # mutual-exclusion check below and raise loudly — an operator who
    # insists on the (broken) MLLM path for a text-only-pinned alias gets
    # a clear error, NOT a silent flip to the garbling MLLM engine.
    # Gating on ``not force_mllm`` here would suppress that guard and
    # silently select the broken path (codex #1116 BLOCKING).
    if _profile is not None and _profile.is_text_only:
        if not force_text:
            logger.info(
                "Alias profile declares is_text_only=True — routing to the "
                "text-only mlx-lm lane (MLLM auto-detection overridden per "
                "alias, #393)"
            )
        force_text = True
        # Fail FAST on the alias-pin ↔ ``--mllm`` conflict — before the
        # generation-config load / guardrail I/O below. The
        # general ``force_mllm and force_text`` guard further down still
        # covers direct ``load_model(force_mllm=True, force_text=True)``
        # callers; this early raise just avoids doing config I/O for an
        # invocation we already know is invalid (codex #1116 nit).
        if force_mllm:
            raise ValueError(
                "force_mllm and force_text are mutually exclusive — "
                "pick at most one to override auto-detection. "
                "(alias pins is_text_only=True but --mllm was also given)"
            )

    # Hybrid/linear-attention VLM checkpoints (e.g. Qwen3.5/3.6/3.8 GatedDeltaNet
    # with a vision tower) auto-route to the MLLM lane on their vision weights.
    # Post-#1798 the MLLM engine CAN serve an ArraysCache backbone, but only in
    # a serialized one-request-at-a-time lane (a BatchKVCache cannot be built
    # over ArraysCache, so concurrent batching stays off — GitHub #352). Left
    # alone, the naive ``rapid-mlx serve <flagship>`` command would boot the
    # whole model into that B=1 lane, capping text throughput for every request.
    # Auto-fall-back to the text-only mlx-lm lane HERE, at the routing layer,
    # with one clear INFO line, so the common text path keeps full batching and
    # --mllm opts into the serialized vision lane. The dense text lane serves the
    # GatedDeltaNet backbone coherently and keeps ``is_hybrid=False`` (avoiding
    # the metal::malloc throttle wedge the 4B/9B/27B dense variants hit under
    # the hybrid scheduler path — see model_auto_config r6-A R6-C1).
    #
    # The fallback is tracked in ``_auto_text_fallback`` — a state DISTINCT
    # from the explicit ``force_text`` / ``--no-mllm`` flag — so diagnostics say
    # "auto-downgraded" and never falsely claim the user passed ``--no-mllm``
    # (codex #2 on #1178). The materialize-then-probe order is load-bearing:
    # ``_ensure_routing_config`` must run BEFORE ``resolve_serving_lane`` so a
    # first-time uncached hybrid VLM has real config evidence and is routed on
    # fact, not on a missing config (codex BLOCKING on #1178). Only fires in
    # auto mode: an explicit ``--mllm`` (force_mllm) is respected so the
    # operator who wants vision gets the serialized MLLM lane (#1798) — for a
    # hybrid backbone that serves vision at B=1; for an arch mlx-vlm cannot
    # drive it still errors — rather than a silent override. #352 dogfood
    # P1-② (0.10.16).
    #
    # The generative-media lanes are exempt. An ``image-gen`` / ``video-gen``
    # alias never reaches ``resolve_serving_lane``'s question at all — it
    # branches to ImageEngine / VideoEngine below — so the MLLM-vs-text
    # preflight has nothing to decide for it. Running it anyway is not merely
    # wasted work: mflux-layout checkpoints keep their weights and configs in
    # ``transformer/`` / ``text_encoder/`` / ``vae/`` subdirectories and ship
    # no ``config.json`` at the checkpoint root, so ``_ensure_routing_config``
    # cannot materialize one and raises. That is how the Images tab's
    # ``z-image-turbo`` alias could never start: a fully-cached 5.5 GB
    # checkpoint refused with an error about hybrid-VLM misrouting, a hazard
    # that does not exist for a diffusion model.
    _auto_text_fallback = False
    _is_generative_media = _profile is not None and _profile.modality in (
        "image-gen",
        "video-gen",
    )
    if not force_text and not force_mllm and not _is_generative_media:
        _ensure_routing_config(model_name)
        _lane_is_mllm, _auto_text_fallback = resolve_serving_lane(
            model_name, force_mllm=force_mllm, force_text=force_text
        )
        if _auto_text_fallback:
            logger.info(
                "Model %r auto-downgraded to the text-only mlx-lm lane for "
                "full batched throughput: it is a multimodal checkpoint whose "
                "language backbone the MLLM continuous-batching engine cannot "
                "batch — either hybrid/linear-attention (GatedDeltaNet: "
                "Qwen3.5/3.6/3.8) or a vision architecture the installed "
                "mlx-vlm cannot drive yet (e.g. muse_glimmer, served via the "
                "vendored text backbone). Pass --mllm to serve vision: a "
                "hybrid backbone runs a serialized one-request-at-a-time lane "
                "(#1798); an unsupported arch errors instead. Pass --no-mllm "
                "to silence this notice.",
                model_name,
            )

    try:
        gen_cfg = load_generation_config_sampling(model_name)
    except Exception as _e:  # pragma: no cover — defensive belt-and-suspenders
        logger.debug(f"generation_config load failed (non-fatal): {_e}")
        gen_cfg = {}
    _generation_config_sampling = gen_cfg or None

    # R15 task #297: warn loudly when the operator's three-tuple matches
    # the MoE + MXFP4 + multi-device throughput cliff (mlx#3402) or the
    # MoE + NVFP4 dynamic-range loss (mlx#2962). Best-effort — wrapped
    # in a try so a guardrail bug can NEVER prevent model load.
    try:
        from ._mxfp4_moe_guardrail import check_from_profile

        check_from_profile(
            model_name=model_name,
            profile=_profile,
            alias=_model_alias,
        )
    except Exception as _e:  # pragma: no cover — defensive belt-and-suspenders
        logger.debug(f"mxfp4/moe guardrail probe failed (non-fatal): {_e}")

    if force_mllm and force_text:
        raise ValueError(
            "force_mllm and force_text are mutually exclusive — "
            "pick at most one to override auto-detection."
        )
    if force_hybrid and no_hybrid:
        raise ValueError(
            "force_hybrid and no_hybrid are mutually exclusive — "
            "pick at most one to override auto-detection."
        )
    if force_spec_decode and no_spec_decode:
        raise ValueError(
            "force_spec_decode and no_spec_decode are mutually exclusive — "
            "pick at most one to override auto-detection."
        )
    if force_openai_harmony_streaming and no_openai_harmony_streaming:
        raise ValueError(
            "force_openai_harmony_streaming and no_openai_harmony_streaming "
            "are mutually exclusive — pick at most one to override the "
            "HarmonyStreamingRouter auto-upgrade gate (#516)."
        )
    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")
    if force_text:
        logger.info(
            "Force text-only mode enabled via --no-mllm flag "
            "(MLLM auto-detection overridden, #393)"
        )

    # The engine picks the text lane for BOTH an explicit ``--no-mllm``
    # (``force_text``) and the automatic hybrid-backbone downgrade
    # (``_auto_text_fallback``). Kept as separate inputs above so the log lines
    # attribute the reason correctly; combined here to select the final lane.
    _effective_force_text = force_text or _auto_text_fallback

    # Modality dispatch: ``text-diffusion`` aliases route to the
    # discrete-text-diffusion engine (mlx-vlm DiffusionGemma path).
    # Default ``text`` keeps the AR BatchedEngine flow that every
    # existing alias has used since #156. The check goes through the
    # already-resolved alias profile so the same call (alias-name or
    # full HF path) lands on the right lane without re-resolving.
    _profile_modality = _profile.modality if _profile is not None else "text"
    if _profile_modality == "video-gen":
        from .runtime.video_lane import VideoEngine

        _video_hf_path = _profile.hf_path if _profile is not None else model_name
        logger.info(
            f"Loading model with VideoEngine (modality=video-gen): {_video_hf_path}"
        )
        _engine = VideoEngine(model_name=_video_hf_path)
        logger.info(f"Video model ready for lazy generation: {model_name}")
    elif _profile_modality == "image-gen":
        from .runtime.image_lane import ImageEngine, require_image_runtime_or_exit

        _image_hf_path = _profile.hf_path if _profile is not None else model_name
        # Preflight the optional image stack (Python ≥3.11 + mflux) BEFORE
        # advertising a ready server, so a missing runtime fails at startup with
        # an actionable diagnostic instead of a generic HTTP 500 on first request.
        require_image_runtime_or_exit(_image_hf_path)
        logger.info(
            f"Loading model with ImageEngine (modality=image-gen): {_image_hf_path}"
        )
        _engine = ImageEngine(model_name=_image_hf_path)
        logger.info(f"Image model ready for lazy generation: {model_name}")
    elif _profile_modality == "text-diffusion":
        from .runtime.diffusion_lane import DiffusionEngine

        # ``python -m vllm_mlx.server --model <alias>`` bypasses
        # cli.py's alias resolution and calls load_model() with the
        # raw alias. BatchedEngine has its own internal resolution,
        # but DiffusionEngine hands the string straight to
        # ``mlx_vlm.utils.load`` which only accepts HF paths — so we
        # must use the profile's resolved hf_path here (codex round
        # 10 [P2]).
        _diffusion_hf_path = _profile.hf_path if _profile is not None else model_name
        logger.info(
            f"Loading model with DiffusionEngine "
            f"(modality=text-diffusion): {_diffusion_hf_path}"
        )
        _engine = DiffusionEngine(
            model_name=_diffusion_hf_path,
            max_tokens=max_tokens,
            scheduler_config=scheduler_config,
        )
        # Eager load — server lifespan's startup_event runs ``await
        # _engine.start()`` like it does for BatchedEngine, but the
        # diffusion engine has additional sanity checks at load time
        # (block-family verification) that we want to surface during
        # the synchronous load_model() call so a misconfigured alias
        # fails BEFORE the lifespan hook hands control to uvicorn.
        _engine._load_blocking()  # noqa: SLF001 — internal helper
        logger.info(f"Model loaded: {model_name}")
    else:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            force_text=_effective_force_text,
            gpu_memory_utilization=gpu_memory_utilization,
            force_hybrid=force_hybrid,
            no_hybrid=no_hybrid,
            force_spec_decode=force_spec_decode,
            no_spec_decode=no_spec_decode,
            force_openai_harmony_streaming=force_openai_harmony_streaming,
            no_openai_harmony_streaming=no_openai_harmony_streaming,
            enable_disk_stream=enable_disk_stream,
            disk_stream_cache_gb=disk_stream_cache_gb,
        )
        logger.info(f"Model loaded: {model_name}")

    # Sync globals into ServerConfig BEFORE _detect_native_tool_support reads
    # them via get_config(). Detection short-circuits when cfg.tool_call_parser
    # is None or cfg.enable_auto_tool_choice is False, so an unsynced cfg
    # silently disables native tool format and forces api/utils.py into the
    # prose-conversion fallback ([Calling tool: ...]) — the model then mimics
    # that format on subsequent turns. See #225.
    _sync_config()

    # Opt-in prompt-deterministic response cache: configure the process
    # singleton's LRU capacity from the resolved SchedulerConfig knob.
    # 0 (default) keeps the cache inert. ``configure_response_cache``
    # atomically sets capacity, clears the store, and bumps the epoch — a
    # stored completion is only valid for the exact model artifact that
    # produced it, but the key spans only the model id, so this (re)load
    # invalidation prevents serving completions from a previously-loaded
    # model after a reload of changed weights under the same id.
    #
    # load_model is boot-only: it is invoked once from serve_command before
    # uvicorn begins accepting requests, and there is no runtime model-swap
    # route. So the order in which the engine is published versus the cache
    # invalidated cannot race with a live request — the epoch-versioned
    # reconfigure is correctness-by-construction today, and defense-in-depth
    # if a runtime reload endpoint is ever added.
    #
    # Best-effort: a cache-config failure must never block model load — but
    # on failure the PREVIOUS cache must NOT stay live under the NEW model
    # (that would serve stale cross-model output). The fail-safe rebinds the
    # singleton to a fresh disabled instance via ``force_disable_response_
    # cache`` rather than calling a method on the possibly-wedged instance
    # that just failed — a fresh capacity-0 object is inert by construction.
    try:
        from .response_cache import configure_response_cache

        configure_response_cache(
            int(getattr(scheduler_config, "response_cache_entries", 0) or 0)
        )
    except Exception as _rc_e:
        logger.warning(
            f"response cache reconfigure failed on model load ({_rc_e}); "
            "forcing the cache disabled + empty so it cannot serve stale "
            "cross-model output"
        )
        try:
            from .response_cache import force_disable_response_cache

            force_disable_response_cache()
        except Exception:  # pragma: no cover — defensive
            pass

    # Set native tool format support on the engine (thread-safe via instance property)
    _engine.preserve_native_tool_format = _detect_native_tool_support()
    if _engine.preserve_native_tool_format:
        logger.info(f"Native tool format enabled for parser: {_tool_call_parser}")

    # Set up tool logits bias processor factory (jump-forward decoding)
    if _enable_tool_logits_bias and _enable_auto_tool_choice and _tool_call_parser:
        try:
            from .api.tool_logits import create_tool_logits_processor

            tokenizer = None
            if hasattr(_engine, "_tokenizer"):
                tokenizer = _engine._tokenizer
            elif hasattr(_engine, "tokenizer"):
                tokenizer = _engine.tokenizer
            if tokenizer is not None:
                # Create factory that produces fresh processors per request
                # Accepts optional tools for parameter value schema constraint
                def _make_factory(parser_name, tok):
                    def factory(tools=None):
                        return create_tool_logits_processor(
                            parser_name, tok, tools=tools
                        )

                    return factory

                factory = _make_factory(_tool_call_parser, tokenizer)
                # Set on BatchedEngine for use during scheduler init
                if hasattr(_engine, "_tool_logits_processor_factory"):
                    _engine._tool_logits_processor_factory = factory
                logger.info(f"Tool logits bias enabled for parser: {_tool_call_parser}")
            else:
                logger.warning("Tool logits bias requested but tokenizer not available")
        except Exception as e:
            logger.warning(f"Failed to set up tool logits bias: {e}")

    logger.info(f"Default max tokens: {_default_max_tokens}")

    # Register in multi-model registry
    aliases = set()
    if _model_alias and _model_alias != _model_name:
        aliases.add(_model_alias)
    entry = ModelEntry(
        engine=_engine,
        model_name=_model_name,
        model_path=_model_path or model_name,
        aliases=aliases,
        tool_call_parser=_tool_call_parser,
        reasoning_parser=_reasoning_parser_name,
        is_mllm=getattr(_engine, "is_mllm", False),
        max_tokens=_default_max_tokens,
    )
    _model_registry.add(entry, is_default=True)

    # Defensive re-sync. `_sync_config()` already ran earlier (before
    # `_detect_native_tool_support()`); under current invariants this call is
    # redundant — `cfg.model_registry` holds a reference to `_model_registry`,
    # every global synced is set before engine construction, and `_engine`
    # mutations propagate via `cfg.engine`. Kept anyway because the bug this
    # PR fixes (#225) was a silent call-ordering failure, and the cost of an
    # idempotent re-sync is trivial against the cost of re-introducing the
    # same failure mode if a future change violates the invariants.
    _sync_config()

    # Task #292: attach ``/v1/audio/*`` routes only when the loaded model
    # actually supports audio OR the operator passed ``--enable-audio``.
    # Pre-fix the router was attached at module import (before any model
    # was loaded), so a text-only ``rapid-mlx serve <text-model>`` boot
    # advertised the audio paths and 500'd on first POST. Calling the
    # helper here — after the model is loaded and ``_model_name`` is
    # stamped — gives FastAPI the chance to return a stock 404 for the
    # audio paths on text-only servers, matching the customer-visible
    # behaviour the Bo R13/R14 fuzz wave asked for.
    register_audio_routes_if_enabled()


async def _load_dynamic_resident_model(
    model_name: str,
    model_path: str | None,
    performance=None,
    image_mode: str | None = None,
) -> ModelEntry:
    """Construct and start one non-primary engine for the residency manager."""

    from .model_aliases import resolve_profile

    profile = resolve_profile(model_name) or (
        resolve_profile(model_path) if model_path else None
    )
    resolved_path = model_path or (
        profile.hf_path if profile is not None else model_name
    )
    modality = profile.modality if profile is not None else "text"

    if modality == "image-gen":
        from .runtime.image_lane import ImageEngine

        engine = ImageEngine(model_name=resolved_path)
        # Dynamic loads are explicit operator/app requests. Materialize the
        # lazy mflux weights before returning so "resident" and budget usage
        # have their literal meanings on the control-plane response.
        await asyncio.to_thread(engine.ensure_resident, mode=image_mode)
    elif modality == "text-diffusion":
        from .runtime.diffusion_lane import DiffusionEngine

        engine = DiffusionEngine(
            model_name=resolved_path,
            max_tokens=_default_max_tokens,
        )
        engine._load_blocking()  # noqa: SLF001
        if hasattr(engine, "_loaded") and not engine._loaded:
            await engine.start()
        engine.generate_warmup()
    elif modality in ("video-gen", "audio"):
        raise RuntimeError(
            f"runtime residency loading is not available for modality {modality!r}"
        )
    else:
        from .runtime.resident_models import resident_scheduler_kwargs
        from .scheduler import SchedulerConfig

        engine = BatchedEngine(
            model_name=resolved_path,
            force_text=bool(profile is not None and profile.is_text_only),
            gpu_memory_utilization=_resident_gpu_memory_utilization,
            scheduler_config=SchedulerConfig(**resident_scheduler_kwargs(performance)),
        )
        await engine.start()
        try:
            engine.generate_warmup()
        except Exception as exc:  # noqa: BLE001 - warmup is an optimization
            logger.debug("Dynamic model warmup failed (non-fatal): %s", exc)

    return ModelEntry(
        engine=engine,
        model_name=model_name,
        model_path=resolved_path,
        aliases=set(),
        tool_call_parser=(profile.tool_call_parser if profile is not None else None),
        reasoning_parser=(profile.reasoning_parser if profile is not None else None),
        is_mllm=getattr(engine, "is_mllm", False),
        max_tokens=_default_max_tokens,
    )


def configure_model_residency(
    *,
    memory_limit_gb: float = 0,
    idle_ttl_seconds: float = 0,
    gpu_memory_utilization: float = 0.90,
) -> ResidentModelManager:
    """Configure the process-wide resident-model manager before startup."""

    global _residency_manager
    global _resident_memory_limit_bytes
    global _resident_idle_ttl_seconds
    global _resident_gpu_memory_utilization

    _resident_memory_limit_bytes = max(0, int(float(memory_limit_gb) * 1024**3))
    _resident_idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
    _resident_gpu_memory_utilization = float(gpu_memory_utilization)
    _residency_manager = ResidentModelManager(
        _model_registry,
        _load_dynamic_resident_model,
        memory_limit_bytes=_resident_memory_limit_bytes,
        idle_ttl_seconds=_resident_idle_ttl_seconds,
        on_primary_changed=_set_resident_primary,
    )
    get_config().residency_manager = _residency_manager
    return _residency_manager


def _set_resident_primary(entry: ModelEntry) -> None:
    """Publish a replacement assistant as the legacy/default engine."""

    global _engine, _model_name, _model_alias, _model_path
    global _enable_auto_tool_choice, _tool_call_parser, _tool_parser_instance
    global _reasoning_parser, _reasoning_parser_name

    _engine = entry.engine
    _model_name = entry.model_name
    _model_alias = entry.model_name
    _model_path = entry.model_path
    _tool_call_parser = entry.tool_call_parser
    _tool_parser_instance = None
    _enable_auto_tool_choice = entry.tool_call_parser is not None
    _reasoning_parser_name = entry.reasoning_parser
    if entry.reasoning_parser is not None:
        from .reasoning import get_parser

        _reasoning_parser = get_parser(entry.reasoning_parser)()
    else:
        _reasoning_parser = None

    cfg = get_config()
    cfg.engine = entry.engine
    cfg.model_name = entry.model_name
    cfg.model_alias = entry.model_name
    cfg.model_path = entry.model_path
    cfg.enable_auto_tool_choice = _enable_auto_tool_choice
    cfg.tool_call_parser = entry.tool_call_parser
    cfg.tool_parser_instance = None
    cfg.reasoning_parser = _reasoning_parser
    cfg.reasoning_parser_name = entry.reasoning_parser


def _sync_config() -> None:
    """Copy server globals into the ServerConfig singleton.

    Called after load_model() and whenever globals change. Bridges the old
    global-variable pattern with the new config object.

    **Must remain idempotent.** load_model() calls this twice (once early
    before _detect_native_tool_support() reads cfg, once again after the
    model registry add as a safety net for future call-site drift). All
    assignments below MUST be straight overwrites — no counters, no
    callback fires, no cache invalidations that depend on prior state.
    See test_sync_config_is_idempotent in tests/test_server_load_model_order.py.
    """
    cfg = get_config()
    cfg.engine = _engine
    cfg.model_name = _model_name
    cfg.model_alias = _model_alias
    cfg.model_path = _model_path
    cfg.default_max_tokens = _default_max_tokens
    cfg.default_max_tokens_is_explicit = _default_max_tokens_is_explicit
    cfg.default_timeout = _default_timeout
    cfg.default_temperature = _default_temperature
    cfg.default_top_p = _default_top_p
    cfg.default_top_k = _default_top_k
    cfg.default_min_p = _default_min_p
    cfg.default_repetition_penalty = _default_repetition_penalty
    cfg.default_presence_penalty = _default_presence_penalty
    cfg.default_frequency_penalty = _default_frequency_penalty
    cfg.alias_recommended_sampling = _alias_recommended_sampling
    cfg.generation_config_sampling = _generation_config_sampling
    cfg.enable_auto_tool_choice = _enable_auto_tool_choice
    cfg.tool_call_parser = _tool_call_parser
    cfg.tool_parser_instance = _tool_parser_instance
    cfg.enable_tool_logits_bias = _enable_tool_logits_bias
    cfg.reasoning_parser = _reasoning_parser
    cfg.reasoning_parser_name = _reasoning_parser_name
    cfg.mcp_manager = _mcp_manager
    cfg.embedding_engine = _embedding_engine
    cfg.embedding_model_locked = _embedding_model_locked
    cfg.api_key = _api_key
    cfg.max_request_bytes = _max_request_bytes
    cfg.sse_keepalive_seconds = _sse_keepalive_seconds
    cfg.body_receive_timeout_seconds = _body_receive_timeout_seconds
    cfg.gc_control = _gc_control
    cfg.no_thinking = _no_thinking
    cfg.relocate_mid_conversation_system = _relocate_mid_conversation_system
    cfg.thinking_token_budget = _thinking_token_budget
    cfg.pin_system_prompt = _pin_system_prompt
    cfg.pinned_system_prompt_hash = _pinned_system_prompt_hash
    cfg.mcp_executor = _mcp_executor
    cfg.mcp_init_error = _mcp_init_error
    cfg.mcp_rejected = _mcp_rejected
    cfg.mcp_config_path = _mcp_config_path
    cfg.model_registry = _model_registry
    cfg.residency_manager = _residency_manager
    cfg.enable_audio_lane = _enable_audio_lane


# Re-export for backward compatibility (test_streaming_pipeline_integration)
from .routes.anthropic import _emit_content_pieces  # noqa: F401, E402

# =============================================================================
# MCP Initialization
# =============================================================================


async def _start_mcp(config_path: str) -> None:
    """Build a manager/executor pair from ``config_path`` and publish them.

    Raises on failure. Callers decide whether that is fatal —
    :func:`init_mcp` (boot) does not, :func:`reload_mcp` (explicit user
    action) reports it back over HTTP.
    """
    global _mcp_manager, _mcp_executor, _mcp_rejected

    from vllm_mlx.mcp import (
        MCPClientManager,
        ToolExecutor,
        ToolSandbox,
        load_mcp_config,
        set_sandbox,
    )

    # Issue #1716: tolerant load. A single entry that fails security
    # validation is dropped and reported through ``/v1/mcp/servers`` rather
    # than taking every other connector down with it.
    config = load_mcp_config(config_path, tolerant=True)
    _mcp_rejected = list(config.rejected)
    for entry in _mcp_rejected:
        logger.warning(f"MCP server '{entry.name}' rejected: {entry.error}")

    # Build locally and publish only after a clean start. ``start()`` connects
    # child processes; if it (or the wiring below) raises with some already up,
    # stopping the local manager first is what keeps a failed init from
    # orphaning subprocesses under a discarded, never-published manager.
    manager = MCPClientManager(config)
    try:
        await manager.start()

        # Wire allowed_high_risk_tools from config into the global sandbox so
        # default-deny on shell/exec/eval tools respects the user's allowlist.
        set_sandbox(
            ToolSandbox(
                allowed_high_risk_tools=set(config.allowed_high_risk_tools),
            )
        )

        executor = ToolExecutor(manager)
    except Exception:
        try:
            await manager.stop()
        except Exception as stop_err:  # pragma: no cover - defensive
            logger.warning(f"Error stopping half-started MCP manager: {stop_err}")
        raise

    _mcp_manager = manager
    _mcp_executor = executor

    logger.info(f"MCP initialized with {len(manager.get_all_tools())} tools")


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file — never fatal.

    Issue #1716: this used to re-raise, and it runs inside the lifespan
    startup, so a missing config file or one unstartable server took the
    WHOLE server down — no chat, no models, no error the desktop app could
    render. MCP is an optional capability; failing to bring it up must
    degrade to "no connectors" rather than "no server". The reason is kept in
    ``_mcp_init_error`` and surfaced on ``/v1/mcp/servers`` so the app can
    show something actionable.
    """
    global _mcp_manager, _mcp_executor, _mcp_init_error, _mcp_config_path

    _mcp_config_path = config_path
    _mcp_init_error = None

    try:
        await _start_mcp(config_path)
    except ImportError:
        _mcp_init_error = "MCP SDK not installed. Install with: pip install mcp"
        logger.error(_mcp_init_error)
        _mcp_manager = None
        _mcp_executor = None
    except Exception as e:
        _mcp_init_error = f"Failed to initialize MCP: {e}"
        logger.error(_mcp_init_error)
        _mcp_manager = None
        _mcp_executor = None

    # Sync whatever we ended up with (including the None/error case) into the
    # ServerConfig singleton so MCP routes see it. Keeping this inside
    # init_mcp() means every code path that initializes MCP also publishes it.
    _sync_config()


async def reload_mcp(config_path: str | None = None) -> str | None:
    """Tear down the running MCP manager and rebuild it from disk.

    Backs ``POST /v1/mcp/reload`` (issue #1716): the desktop app edits
    ``mcp.json`` and needs the change to take effect without restarting the
    model, which would mean a multi-GB reload for a one-line config edit.

    Returns ``None`` on success, or the error string on failure. The old
    manager is stopped either way — a half-torn-down manager whose child
    processes are still alive is worse than no manager.

    A ``/v1/mcp/execute`` call that captured the old manager just before a
    reload can still run against it as it is torn down; that resolves to a
    clean "server not connected" error result (the client reports
    ``is_connected == False``), not a crash. Concurrent reloads, which WOULD
    corrupt the shared globals, are serialized by ``_mcp_reload_lock``.
    """
    global _mcp_manager, _mcp_executor, _mcp_init_error, _mcp_config_path
    global _mcp_reload_lock

    if _mcp_reload_lock is None:
        _mcp_reload_lock = asyncio.Lock()

    async with _mcp_reload_lock:
        path = config_path or _mcp_config_path
        if path is None:
            _mcp_init_error = (
                "No MCP config path known — start the server with --mcp-config"
            )
            _sync_config()
            return _mcp_init_error

        _mcp_config_path = path

        if _mcp_manager is not None:
            try:
                await _mcp_manager.stop()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Error stopping MCP manager during reload: {e}")
        _mcp_manager = None
        _mcp_executor = None

        _mcp_init_error = None
        try:
            await _start_mcp(path)
        except Exception as e:
            _mcp_init_error = f"Failed to reload MCP: {e}"
            logger.error(_mcp_init_error)
            _mcp_manager = None
            _mcp_executor = None

        _sync_config()
        return _mcp_init_error


# =============================================================================
# Route modules — imported after all server globals are defined to avoid
# circular imports (route modules import verify_api_key etc. from this module)
# =============================================================================
from .routes.anthropic import router as _anthropic_router

# Task #292: ``_audio_router`` is no longer registered at import time —
# :func:`register_audio_routes_if_enabled` (called from ``load_model``
# and :func:`vllm_mlx.cli._serve_audio_mode`) imports the router lazily
# through ``vllm_mlx.routes.audio.register_audio_routes``. Removing the
# unused top-level alias keeps the rebound dispatcher hot path short.
from .routes.cache import router as _cache_router
from .routes.chat import router as _chat_router
from .routes.completions import router as _completions_router
from .routes.embeddings import router as _embeddings_router
from .routes.health import admin_router as _health_admin_router
from .routes.health import probe_router as _probe_router
from .routes.health import router as _health_router
from .routes.images import router as _images_router
from .routes.mcp_routes import admin_router as _mcp_admin_router
from .routes.mcp_routes import router as _mcp_router
from .routes.metrics import router as _metrics_router
from .routes.models import router as _models_router
from .routes.residency import router as _residency_router
from .routes.responses import router as _responses_router
from .routes.video import router as _video_router

app.include_router(_probe_router)
app.include_router(_health_router)
# Destructive control-plane routes (F-150 / F-151). Distinct router so the
# ``X-Rapid-MLX-Internal: true`` gate ALSO applies when ``--api-key`` is unset.
app.include_router(_health_admin_router)
app.include_router(_metrics_router)
# Keep literal residency paths ahead of ``/v1/models/{model_id:path}`` so the
# latter cannot consume ``residency`` as an ordinary model id.
app.include_router(_residency_router)
app.include_router(_models_router)
app.include_router(_chat_router)
app.include_router(_completions_router)
app.include_router(_anthropic_router)
app.include_router(_responses_router)
app.include_router(_video_router)
# Image lane is registered unconditionally like video: a text-only server
# answers /v1/images/generations with the 409 "image_model_not_loaded"
# envelope (the router's own gate), never a stray 404.
app.include_router(_images_router)
app.include_router(_embeddings_router)
app.include_router(_mcp_router)
# ``/v1/mcp/reload`` — separate router so the Bearer-OR-x-api-key gate applies
# even when ``--api-key`` is unset (mirrors ``_health_admin_router``).
app.include_router(_mcp_admin_router)
# Task #292: ``_audio_router`` is registered LAZILY (after model load) by
# :func:`register_audio_routes_if_enabled` — text-only servers (Bo R13/R14
# fuzz wave: Qwen3-7B-4bit, etc.) must answer ``/v1/audio/*`` with a
# stock 404 instead of advertising routes that 500 on first call.
app.include_router(_cache_router)


def register_audio_routes_if_enabled() -> bool:
    """Task #292: attach the audio router only when audio is enabled.

    The gate is:

    * The loaded model alias / HF id resolves through the audio
      registry (the audio-mode boot path
      :func:`vllm_mlx.cli._serve_audio_mode` always populates
      ``_model_name`` / ``_model_alias`` with a registry-known id), OR
    * The operator passed ``--enable-audio`` on a text-mode boot
      (``_enable_audio_lane`` is True).

    Returns True when the router was attached on this call, False
    otherwise. Idempotent: called from ``load_model`` (text path),
    :func:`_post_audio_mode_routes_hook` (audio path), and the legacy
    ``python -m vllm_mlx.server`` entrypoint. Doing it here keeps the
    decision close to the boot state that drives it instead of
    threading the model name through three call sites.
    """
    from .routes.audio import audio_routes_should_register, register_audio_routes

    if not audio_routes_should_register(
        model_name=_model_name,
        model_alias=_model_alias,
        enable_audio_lane=_enable_audio_lane,
    ):
        return False
    return register_audio_routes(app)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    if os.environ.get("RAPID_PYSAMPLE"):
        from ._pysample import install as _pysample_install

        _pysample_install()
    parser = argparse.ArgumentParser(
        description="Rapid-MLX OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start the server
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # With MCP tools
    python -m vllm_mlx.server --model mlx-community/Qwen3-4B-4bit --mcp-config mcp.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind to (default: 127.0.0.1, loopback-only). "
            "Pass 0.0.0.0 to expose the server on every interface "
            "(LAN reachable)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--log-level",
        type=normalize_log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn (case-insensitive)",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model). Also disables the automatic text-only fallback: a vision-config checkpoint with no usable vision tower normally auto-degrades to text-only serving (#1187), but with --mllm it hard-fails instead.",
    )
    parser.add_argument(
        "--no-mllm",
        "--text-only",
        dest="no_mllm",
        action="store_true",
        help="Force text-only LLM routing even when auto-detection would route as MLLM (#393 escape hatch). Mutually exclusive with --mllm.",
    )
    # SOP §10 routing-override escape hatches — mirror the unified CLI
    # (vllm_mlx/cli.py) so this standalone entrypoint never becomes a
    # silent gap for any auto-routing decision.
    parser.add_argument(
        "--no-tool-call-parser",
        dest="no_tool_call_parser",
        action="store_true",
        default=False,
        help="Force-disable tool-call parser auto-detection. Mutually exclusive with --tool-call-parser.",
    )
    parser.add_argument(
        "--no-reasoning-parser",
        dest="no_reasoning_parser",
        action="store_true",
        default=False,
        help="Force-disable reasoning parser auto-detection. Mutually exclusive with --reasoning-parser.",
    )
    parser.add_argument(
        "--force-hybrid",
        dest="force_hybrid",
        action="store_true",
        default=False,
        help="Force-treat the model as hybrid (Mamba/linear-attention). Mutually exclusive with --no-hybrid.",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="no_hybrid",
        action="store_true",
        default=False,
        help="Force-treat the model as non-hybrid (full attention). Mutually exclusive with --force-hybrid.",
    )
    parser.add_argument(
        "--force-spec-decode",
        dest="force_spec_decode",
        action="store_true",
        default=False,
        help="Force-enable speculative-decode eligibility. Mutually exclusive with --no-spec-decode.",
    )
    parser.add_argument(
        "--no-spec-decode",
        dest="no_spec_decode",
        action="store_true",
        default=False,
        help="Force-disable speculative-decode eligibility (suffix/MTP/DFlash). Mutually exclusive with --force-spec-decode.",
    )
    # #516 — HarmonyStreamingRouter auto-upgrade escape hatches (SOP G11).
    parser.add_argument(
        "--force-openai-harmony-streaming",
        dest="force_openai_harmony_streaming",
        action="store_true",
        default=False,
        help="Force-on HarmonyStreamingRouter (bypass compat gate). Debug only. Mutually exclusive with --no-openai-harmony-streaming.",
    )
    parser.add_argument(
        "--no-openai-harmony-streaming",
        dest="no_openai_harmony_streaming",
        action="store_true",
        default=False,
        help="Force-off HarmonyStreamingRouter upgrade; use legacy state machine. Mutually exclusive with --force-openai-harmony-streaming.",
    )
    # PFlash long-prompt prefill compression (#287). Off by default. The
    # unified ``rapid-mlx serve`` CLI exposes the same surface; we mirror
    # it here so the standalone ``python -m vllm_mlx.server`` path is
    # not a silent gap (SOP §10).
    from .cli import _add_pflash_args as _add_pflash_args_to_server_parser

    _add_pflash_args_to_server_parser(parser)
    import argparse as _ap

    # TurboQuant flags — MUST match ``rapid-mlx serve`` (cli.py) choice
    # set + defaults so this standalone entry is functionally at parity.
    # Pre-#969, this parser was missing the ``"none"`` off-switch added
    # in #962 (argparse rejected the flag outright) AND the parsed values
    # were never threaded into ``SchedulerConfig`` below (silent drop —
    # same bug class as #400). Both entries now share the
    # ``turboquant_scheduler_kwargs`` helper.
    parser.add_argument(
        "--kv-cache-turboquant",
        nargs="?",
        const="v4",
        default=None,
        choices=["v4", "k8v4", "none"],
        help=_ap.SUPPRESS,
    )
    parser.add_argument(
        "--kv-cache-turboquant-bits", type=int, default=None, help=_ap.SUPPRESS
    )
    parser.add_argument(
        "--kv-cache-turboquant-group-size", type=int, default=32, help=_ap.SUPPRESS
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default max tokens for generation (caps when client sends None)",
    )
    # ``--api-key`` accepts an inline value OR falls back to the
    # ``RAPID_MLX_API_KEY`` env var. The env-var form keeps the bearer
    # key out of ``argv`` (visible to ``ps -ef`` for any local user) —
    # the standalone-shim spawn path that rapid-desktop's sidecar uses
    # would otherwise leak the per-launch bearer token in the process
    # list (codex BLOCKER taxonomy #3, dogfood-v0.8.2 finding #3).
    # Inline value still works for backwards-compat with existing
    # scripts; if both are set, the inline value wins.
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for authentication (if not set, falls back to the "
            "RAPID_MLX_API_KEY env var; if neither, no auth required)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Default request timeout in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    # Tool call parser options
    from .tool_parsers.abstract_tool_parser import ToolParserManager

    tool_parser_choices = ToolParserManager.list_registered()
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        choices=tool_parser_choices,
        help=(
            "Tool call parser to use for structured tool call extraction. "
            f"Options: {', '.join(tool_parser_choices)}. "
            "Automatically enables --enable-auto-tool-choice."
        ),
    )
    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=False,
        help="Enable automatic tool choice (required with --tool-call-parser)",
    )
    parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=False,
        help="Enable jump-forward decoding bias for tool call structural tokens",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Pre-load an embedding model at startup (e.g. "
            "mlx-community/all-MiniLM-L6-v2-4bit). Requires the "
            "[embeddings] extra: pip install 'rapid-mlx[embeddings]'."
        ),
    )
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Default temperature for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Default top_p for generation when not specified in request",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Default top_k for generation when not specified in request",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Tokens to process per prefill chunk (default: 2048). "
        "Larger values may improve TTFT on Apple Silicon with sufficient memory.",
    )
    parser.add_argument(
        "--vision-min-pixels",
        type=int,
        default=0,
        help="Minimum pixels for dynamic-resolution VLM inputs (0: model default).",
    )
    parser.add_argument(
        "--vision-max-pixels",
        type=int,
        default=0,
        help="Maximum pixels for dynamic-resolution VLM inputs (0: model default).",
    )
    # Task #292: mirror the ``rapid-mlx serve`` ``--enable-audio`` flag
    # on the legacy ``python -m vllm_mlx.server`` entrypoint so the same
    # text-mode-with-audio escape hatch is available to operators who
    # boot via the older command (e.g. supervisord units, internal
    # tools pinned to the module-form invocation).
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        default=False,
        help=(
            "Mount the ``/v1/audio/*`` routes even when the loaded model "
            "is text-only. Audio-capable models auto-mount the routes; "
            "this flag is only needed on text-mode boots."
        ),
    )

    args = parser.parse_args()

    # PortSweep pre-flight (codex round-1 MAJOR on PR #848): mirror the
    # ``rapid-mlx serve`` CLI's loopback-shadow probe here so the
    # legacy ``python -m vllm_mlx.server`` entrypoint doesn't silently
    # reopen the v0.8.2 dogfood-finding-#2 bypass. Probes ``args.host``
    # AND ``127.0.0.1`` when ``args.host`` is a wildcard alias
    # (``0.0.0.0`` or ``""``) so a co-resident loopback-only listener
    # is caught before we sink time into model load.
    from .cli import _port_preflight_or_die

    _port_preflight_or_die(args.host, args.port, model=args.model)

    # F-H08-INCOMPLETE: the ``[embeddings]`` extra-required guard MUST
    # fire BEFORE logging configuration and the security/banner side
    # effects below. Pre-fix on this entrypoint the probe ran AFTER the
    # parser-init log lines and the security summary, then the user
    # saw "error: --embedding-model requires the [embeddings] extra"
    # interleaved with banner output and exit-2 — Diego logged this
    # as a warning-and-fall-through. Hoisting the probe puts the
    # error first with nothing else on stderr/stdout before it.
    if getattr(args, "embedding_model", None):
        from .embedding import require_mlx_embeddings_or_exit

        require_mlx_embeddings_or_exit()

    uvicorn_log_level = configure_logging(args.log_level)

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _default_temperature, _default_top_p, _default_top_k
    global _enable_audio_lane
    # Task #292: forward ``--enable-audio`` to the gate that decides
    # whether ``load_model``'s post-load hook attaches the audio router.
    # Codex r2 NIT #2: assign from the parsed value directly so a second
    # in-process ``main()`` call (test harness, embedded usage) without
    # ``--enable-audio`` clears any stale ``True`` from a prior run —
    # without this the gate would silently advertise audio on the next
    # text-only boot.
    _enable_audio_lane = bool(getattr(args, "enable_audio", False))
    # Env-fallback for the bearer key: keep it out of argv where
    # ``ps -ef`` would leak it. ``_resolve_api_key`` is the single
    # SSOT for the policy (inline-wins, env-fallback) — see its
    # docstring for the dogfood-v0.8.2 finding #3 context.
    _api_key = _resolve_api_key(args.api_key)
    _default_timeout = args.timeout
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p
    if args.default_top_k is not None:
        _default_top_k = args.default_top_k

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = configure_rate_limiter(args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        # Don't reveal whether the key came from argv (visible to ps)
        # or env (the recommended form); the user already knows which
        # they used.
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning(
            "  Authentication: DISABLED - Set RAPID_MLX_API_KEY env or "
            "use --api-key to enable"
        )
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["RAPID_MLX_MCP_CONFIG"] = args.mcp_config

    # Auto-detect parser config from model name when not explicitly set.
    # SOP §10: honor --no-tool-call-parser / --no-reasoning-parser opt-
    # outs so this entrypoint matches the unified CLI behavior.
    _opt_out_tool = getattr(args, "no_tool_call_parser", False)
    _opt_out_reasoning = getattr(args, "no_reasoning_parser", False)
    if args.tool_call_parser and _opt_out_tool:
        parser.error(
            "--tool-call-parser and --no-tool-call-parser are mutually exclusive"
        )
    if args.reasoning_parser and _opt_out_reasoning:
        parser.error(
            "--reasoning-parser and --no-reasoning-parser are mutually exclusive"
        )
    if not args.tool_call_parser or not args.reasoning_parser:
        from .model_auto_config import detect_model_config

        auto_config = detect_model_config(args.model)
        if auto_config:
            if (
                not args.tool_call_parser
                and not _opt_out_tool
                and auto_config.tool_call_parser
            ):
                args.tool_call_parser = auto_config.tool_call_parser
                logger.info(
                    f"Auto-configured --tool-call-parser {auto_config.tool_call_parser}"
                )
            if (
                not args.reasoning_parser
                and not _opt_out_reasoning
                and auto_config.reasoning_parser
            ):
                args.reasoning_parser = auto_config.reasoning_parser
                logger.info(
                    f"Auto-configured --reasoning-parser {auto_config.reasoning_parser}"
                )

    # Initialize tool call parser if specified via CLI (or auto-detected)
    if args.tool_call_parser:
        global _enable_auto_tool_choice, _tool_call_parser, _enable_tool_logits_bias
        _tool_call_parser = args.tool_call_parser
        _enable_auto_tool_choice = True  # Implied by --tool-call-parser
        logger.info(f"Tool call parser enabled: {args.tool_call_parser}")
    if args.enable_auto_tool_choice:
        _enable_auto_tool_choice = True
    if args.enable_tool_logits_bias:
        _enable_tool_logits_bias = True

    # Initialize reasoning parser if specified (or auto-detected)
    if args.reasoning_parser:
        global _reasoning_parser, _reasoning_parser_name
        from .reasoning import get_parser

        parser_cls = get_parser(args.reasoning_parser)
        _reasoning_parser = parser_cls()
        _reasoning_parser_name = args.reasoning_parser
        logger.info(f"Reasoning parser enabled: {args.reasoning_parser}")

    # Pre-load embedding model if specified. The H-08 guard already
    # fired at the top of this function (F-H08-INCOMPLETE fix); by the
    # time we reach this point either ``args.embedding_model`` is None
    # or ``mlx_embeddings`` is importable. The shared helper re-probes
    # defensively as belt-and-braces and also performs the D-EMBED-ALIAS
    # alias-resolution + ModelNotFoundError translation so behaviour
    # matches the unified ``rapid-mlx serve`` path exactly. Lazy import
    # to avoid a circular at module-load time (cli imports server in
    # ``serve_command``; server imports cli only inside this branch).
    if args.embedding_model:
        from .cli import _load_embedding_model_or_exit

        _load_embedding_model_or_exit(args, load_embedding_model)

    # Build a SchedulerConfig so user-supplied flags on this standalone entry
    # (`python -m vllm_mlx.server` / `mise run`) reach the engine. Pre-0.6.52
    # this entrypoint forwarded args.prefill_step_size to load_model where it
    # was silently dropped — same bug class as #400. The unified rapid-mlx
    # CLI builds a richer SchedulerConfig in cli.py; the standalone path only
    # exposes a small subset of flags, so we plumb just those.
    from .model_aliases import resolve_profile as _srv_resolve_profile
    from .pflash import resolve_pflash_config as _server_pflash_resolve_config
    from .pflash import validate_model_support as _server_pflash_validate
    from .scheduler import SchedulerConfig

    # Per-alias PFlash default (#287): verified Qwen3.5 / Qwen3.6 aliases
    # switch to ``always`` when the user passes no ``--pflash`` flag; all
    # other aliases keep the conservative ``off``. Explicit overrides win.
    #
    # Resolve the FINAL serving lane once. PFlash defaulting and
    # ``validate_model_support`` must both see the effective lane, NOT the raw
    # multimodal classification: a hybrid VLM that auto-downgrades to the
    # text-only lane is PFlash-capable there, exactly as an explicit
    # ``--text-only`` run would be (#352 dogfood P1-②).
    #
    # Only materialize the checkpoint config when we actually need to
    # AUTO-detect the lane (neither ``--mllm`` nor ``--no-mllm`` given). An
    # explicit lane flag short-circuits ``resolve_serving_lane`` before it reads
    # any config, so materializing there is unnecessary — and running the
    # ``_ensure_routing_config`` fail-fast would DENY the very ``--no-mllm``
    # escape hatch its own error message advertises when the config cannot be
    # fetched. Mirror ``load_model()``'s flag-first skip (codex BLOCKING #1178).
    # Same generative-media exemption as ``load_model`` above: an image-gen /
    # video-gen alias branches to its own engine and never asks the
    # MLLM-vs-text question, while an mflux-layout checkpoint has no
    # root-level config.json for the preflight to materialize.
    _srv_profile = _srv_resolve_profile(args.model)
    _srv_generative_media = _srv_profile is not None and _srv_profile.modality in (
        "image-gen",
        "video-gen",
    )
    _srv_force_mllm = getattr(args, "mllm", False)
    _srv_force_text = getattr(args, "no_mllm", False)
    if not _srv_force_mllm and not _srv_force_text and not _srv_generative_media:
        _ensure_routing_config(args.model)
    _srv_is_mllm, _ = resolve_serving_lane(
        args.model,
        force_mllm=_srv_force_mllm,
        force_text=_srv_force_text,
    )
    # Resolve mode AND per-alias keep_ratio through the single shared helper —
    # the same call ``cli.py`` uses for ``serve``/``bench``. Going through
    # ``resolve_pflash_config`` (rather than ``resolve_pflash_mode_default`` +
    # ``config_from_args`` directly) is what applies a per-alias
    # ``pflash_keep_ratio`` override (#1458): a verified alias pinned at a
    # non-default ratio (e.g. bonsai-27b-2bit @0.50, whose mid-prompt recall
    # collapses to 1/5 at the 0.20 engine default) would otherwise auto-enable
    # PFlash at the lossy 0.20 here while ``rapid-mlx serve`` used 0.50 — the
    # two serving entrypoints must not drift. It mutates ``args.pflash`` and
    # ``args.pflash_keep_ratio`` in place so later readers see resolved values.
    try:
        server_pflash_config = _server_pflash_resolve_config(
            args, model_name=args.model, is_multimodal=_srv_is_mllm
        )
        _server_pflash_validate(
            server_pflash_config,
            model_name=args.model,
            is_mllm=_srv_is_mllm,
        )
    except ValueError as e:
        parser.error(str(e))

    # TurboQuant resolution (#969): mirror ``rapid-mlx serve`` so the
    # standalone entry actually honors ``--kv-cache-turboquant``. The
    # helper collapses the ``"none"`` off-switch sentinel to ``None`` and
    # applies the per-alias ``k8v4_verified`` default. There is no
    # ``--kv-cache-quantization`` flag on this parser, so the mutual-
    # exclusion check in ``cli.py`` is a no-op here (``getattr`` guard on
    # the shared helper covers programmatic callers that inject one).
    from .turboquant import (
        resolve_turboquant_mode_default as _server_turboquant_resolve_default,
    )
    from .turboquant import (
        turboquant_scheduler_kwargs as _server_turboquant_scheduler_kwargs,
    )

    args.kv_cache_turboquant = _server_turboquant_resolve_default(
        args, model_name=args.model
    )

    if args.vision_min_pixels < 0 or args.vision_max_pixels < 0:
        parser.error("vision pixel bounds must be non-negative")
    if (
        args.vision_min_pixels
        and args.vision_max_pixels
        and args.vision_min_pixels > args.vision_max_pixels
    ):
        parser.error("--vision-min-pixels must not exceed --vision-max-pixels")

    scheduler_config = SchedulerConfig(
        prefill_step_size=args.prefill_step_size,
        vision_min_pixels=args.vision_min_pixels,
        vision_max_pixels=args.vision_max_pixels,
        pflash_config=server_pflash_config,
        **_server_turboquant_scheduler_kwargs(args),
    )

    # Load model before starting server
    _max_tokens_is_explicit = args.max_tokens is not None
    if args.max_tokens is None:
        args.max_tokens = 4096

    if args.mllm and args.no_mllm:
        parser.error("--mllm and --no-mllm are mutually exclusive")
    if getattr(args, "force_hybrid", False) and getattr(args, "no_hybrid", False):
        parser.error("--force-hybrid and --no-hybrid are mutually exclusive")
    if getattr(args, "force_spec_decode", False) and getattr(
        args, "no_spec_decode", False
    ):
        parser.error("--force-spec-decode and --no-spec-decode are mutually exclusive")
    if getattr(args, "force_openai_harmony_streaming", False) and getattr(
        args, "no_openai_harmony_streaming", False
    ):
        parser.error(
            "--force-openai-harmony-streaming and "
            "--no-openai-harmony-streaming are mutually exclusive"
        )
    load_model(
        args.model,
        scheduler_config=scheduler_config,
        max_tokens=args.max_tokens,
        max_tokens_is_explicit=_max_tokens_is_explicit,
        force_mllm=args.mllm,
        force_text=args.no_mllm,
        force_hybrid=getattr(args, "force_hybrid", False),
        no_hybrid=getattr(args, "no_hybrid", False),
        force_spec_decode=getattr(args, "force_spec_decode", False),
        no_spec_decode=getattr(args, "no_spec_decode", False),
        force_openai_harmony_streaming=getattr(
            args, "force_openai_harmony_streaming", False
        ),
        no_openai_harmony_streaming=getattr(args, "no_openai_harmony_streaming", False),
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level=uvicorn_log_level)


if __name__ == "__main__":
    main()
