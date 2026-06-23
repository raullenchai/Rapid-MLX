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
    # Simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Batched mode (for multiple concurrent users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

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

# Global engine instance (single-model legacy path, also primary model in multi-model)
_engine: BaseEngine | None = None
_model_name: str | None = None
_model_alias: str | None = None  # Short alias used to start the model (if any)
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

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used

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

# Cloud routing (offload large-context requests to cloud LLM)
_cloud_router = None  # CloudRouter instance when --cloud-model is set

# GC control (Tier 0 optimization)
_gc_control: bool = True  # Disable GC during generation to avoid latency spikes
_no_thinking: bool = (
    False  # --no-thinking: force enable_thinking=False in chat template
)

# Pinned prefix cache (Tier 0 optimization)
_pin_system_prompt: bool = False  # Auto-pin system prompt prefix cache blocks
_pinned_system_prompt_hash: str | None = None  # Hash of pinned system prompt


from .runtime.cache import (  # noqa: E402
    get_cache_dir as _get_cache_dir,  # noqa: F401
)
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
        await _engine.start()

    # Warmup: generate one token to trigger Metal shader compilation.
    # Runs here (not in CLI) so all engine types are fully started first.
    if _engine is not None:
        import time as _time

        logger.info("Warming up (compiling Metal shaders)...")
        _warmup_start = _time.monotonic()
        try:
            # Skip warmup for hybrid models (GatedDeltaNet) to avoid
            # contaminating compiled kernel state that interferes with
            # batched inference.  Check multiple engine wrappers:
            # BatchedEngine sets _hybrid_throttle via EngineCore,
            # Check model for hybrid cache
            _is_hybrid = getattr(_engine, "_hybrid_throttle", False)
            if not _is_hybrid and not getattr(_engine, "_is_mllm", False):
                # Try to find the raw model through wrapper layers
                _model = getattr(_engine, "_model", None) or getattr(
                    _engine, "_shared_model", None
                )
                # Unwrap model wrapper if needed
                if (
                    _model
                    and hasattr(_model, "model")
                    and not hasattr(_model, "make_cache")
                ):
                    _model = _model.model
                if _model and hasattr(_model, "make_cache"):
                    try:
                        from mlx_lm.models.cache import ArraysCache

                        _test_cache = _model.make_cache()
                        _is_hybrid = any(
                            isinstance(c, ArraysCache) for c in _test_cache
                        )
                    except Exception:
                        pass
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

    # Load persisted cache from disk (AFTER engine start — AsyncEngineCore must exist)
    if _engine is not None and hasattr(_engine, "load_cache_from_disk"):
        _load_prefix_cache_from_disk()

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
    _cfg.ready = True

    # Print the real "Ready:" banner now — only here is the port truly
    # accepting connections AND the engine warmed up. The CLI's earlier
    # "Starting server …" line is replaced by this. If neither the
    # host/port nor inherited-fd source of truth was stashed (e.g.
    # embedded usage where uvicorn is owned elsewhere), fall back silently.
    if _cfg.bind_host and _cfg.bind_port:
        print(f"  Ready: http://{_cfg.bind_host}:{_cfg.bind_port}/v1")
        print(f"  Docs:  http://{_cfg.bind_host}:{_cfg.bind_port}/docs")
        print()
    elif _cfg.bind_listen_fd is not None:
        # Socket-activation branch: the supervisor's ``getsockname`` is the
        # source of truth for the bind address (we don't probe it). Print
        # the fd shape so log readers can match it to the supervisor's
        # ``LISTEN_FDS=1`` handoff record.
        print(f"  Ready: inherited fd {_cfg.bind_listen_fd}")
        print()

    yield

    # Shutdown: stop accepting "ready" before tearing things down.
    get_config().ready = False

    # Shutdown: Save cache to disk BEFORE stopping engine.
    #
    # Delegates to ``_shutdown_save_prefix_cache`` which wraps the
    # synchronous save in ``asyncio.to_thread`` — see that function's
    # docstring for the rationale. Extracted so the regression test
    # pins the wrapper at the production callsite rather than wrapping
    # ``to_thread`` test-side (codex PR #667 round 1 BLOCKING-3).
    await _shutdown_save_prefix_cache()

    # Shutdown: Close MCP connections and stop engine
    if _mcp_manager is not None:
        await _mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _engine is not None:
        await _engine.stop()
        logger.info("Engine stopped")

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

    # Fail-closed path: ``RAPID_MLX_CORS_ALLOW_ORIGINS`` was set but
    # parsed to an empty list (operator-controlled typo). Don't register
    # CORSMiddleware — that's the visible signal the WARNING above
    # promises. Returning ``[]`` here also keeps the legacy ``configure_cors``
    # back-compat stub callable from tests that monkeypatch a 1-arg
    # lambda — we never call ``configure_cors(...)`` on the fail-closed
    # path, so the stub's signature doesn't matter.
    if not origins:
        return []

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
) -> None:
    """Load or reuse the embedding model engine when configured."""
    global _embedding_engine, _embedding_model_locked

    if not model_name:
        return

    if lock:
        _embedding_model_locked = model_name

    if (
        reuse_existing
        and _embedding_engine is not None
        and _embedding_engine.model_name == model_name
    ):
        return

    from .embedding import EmbeddingEngine

    _embedding_engine = EmbeddingEngine(model_name)
    _embedding_engine.load()

    # Sync into config for route modules
    cfg = get_config()
    cfg.embedding_engine = _embedding_engine
    cfg.embedding_model_locked = _embedding_model_locked


def load_model(
    model_name: str,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int | None = None,
    force_mllm: bool = False,
    gpu_memory_utilization: float = 0.90,
    prefill_step_size: int | None = None,
    cloud_model: str | None = None,
    cloud_threshold: int = 20000,
    cloud_api_base: str | None = None,
    cloud_api_key: str | None = None,
    served_model_name: str | None = None,
    mtp: bool = False,
    *,
    max_tokens_is_explicit: bool | None = None,
    force_text: bool = False,
    force_hybrid: bool = False,
    no_hybrid: bool = False,
    force_spec_decode: bool = False,
    no_spec_decode: bool = False,
    force_openai_harmony_streaming: bool = False,
    no_openai_harmony_streaming: bool = False,
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
        mtp: Enable native MTP speculative decoding
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
    """
    max_tokens_was_supplied = max_tokens is not None
    if max_tokens is None:
        max_tokens = 32768
    if max_tokens_is_explicit is None:
        max_tokens_is_explicit = max_tokens_was_supplied

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
        _cloud_router, \
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
    try:
        gen_cfg = load_generation_config_sampling(model_name)
    except Exception as _e:  # pragma: no cover — defensive belt-and-suspenders
        logger.debug(f"generation_config load failed (non-fatal): {_e}")
        gen_cfg = {}
    _generation_config_sampling = gen_cfg or None

    # Initialize cloud router if --cloud-model is set
    if cloud_model:
        from .cloud_router import CloudRouter

        _cloud_router = CloudRouter(
            cloud_model=cloud_model,
            threshold=cloud_threshold,
            api_base=cloud_api_base,
            api_key=cloud_api_key,
        )
        logger.info(
            f"Cloud routing enabled: model={cloud_model}, threshold={cloud_threshold} new tokens"
        )
    else:
        _cloud_router = None

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

    # Modality dispatch: ``text-diffusion`` aliases route to the
    # discrete-text-diffusion engine (mlx-vlm DiffusionGemma path).
    # Default ``text`` keeps the AR BatchedEngine flow that every
    # existing alias has used since #156. The check goes through the
    # already-resolved alias profile so the same call (alias-name or
    # full HF path) lands on the right lane without re-resolving.
    _profile_modality = _profile.modality if _profile is not None else "text"
    if _profile_modality == "text-diffusion":
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
            force_text=force_text,
            gpu_memory_utilization=gpu_memory_utilization,
            force_hybrid=force_hybrid,
            no_hybrid=no_hybrid,
            force_spec_decode=force_spec_decode,
            no_spec_decode=no_spec_decode,
            force_openai_harmony_streaming=force_openai_harmony_streaming,
            no_openai_harmony_streaming=no_openai_harmony_streaming,
        )
        logger.info(f"Model loaded: {model_name}")

    # Sync globals into ServerConfig BEFORE _detect_native_tool_support reads
    # them via get_config(). Detection short-circuits when cfg.tool_call_parser
    # is None or cfg.enable_auto_tool_choice is False, so an unsynced cfg
    # silently disables native tool format and forces api/utils.py into the
    # prose-conversion fallback ([Calling tool: ...]) — the model then mimics
    # that format on subsequent turns. See #225.
    _sync_config()

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
    cfg.inference_lock = None  # legacy, unused with BatchedEngine
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
    cfg.cloud_router = _cloud_router
    cfg.gc_control = _gc_control
    cfg.no_thinking = _no_thinking
    cfg.thinking_token_budget = _thinking_token_budget
    cfg.pin_system_prompt = _pin_system_prompt
    cfg.pinned_system_prompt_hash = _pinned_system_prompt_hash
    cfg.mcp_executor = _mcp_executor
    cfg.model_registry = _model_registry


# Re-export for backward compatibility (test_streaming_pipeline_integration)
from .routes.anthropic import _emit_content_pieces  # noqa: F401, E402

# =============================================================================
# MCP Initialization
# =============================================================================


async def init_mcp(config_path: str):
    """Initialize MCP manager from config file."""
    global _mcp_manager, _mcp_executor

    try:
        from vllm_mlx.mcp import (
            MCPClientManager,
            ToolExecutor,
            ToolSandbox,
            load_mcp_config,
            set_sandbox,
        )

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

        # Wire allowed_high_risk_tools from config into the global sandbox so
        # default-deny on shell/exec/eval tools respects the user's allowlist.
        set_sandbox(
            ToolSandbox(
                allowed_high_risk_tools=set(config.allowed_high_risk_tools),
            )
        )

        _mcp_executor = ToolExecutor(_mcp_manager)

        logger.info(f"MCP initialized with {len(_mcp_manager.get_all_tools())} tools")

    except ImportError:
        logger.error("MCP SDK not installed. Install with: pip install mcp")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP: {e}")
        raise


# =============================================================================
# Route modules — imported after all server globals are defined to avoid
# circular imports (route modules import verify_api_key etc. from this module)
# =============================================================================
from .routes.anthropic import router as _anthropic_router
from .routes.audio import router as _audio_router
from .routes.cache import router as _cache_router
from .routes.chat import router as _chat_router
from .routes.completions import router as _completions_router
from .routes.embeddings import router as _embeddings_router
from .routes.health import admin_router as _health_admin_router
from .routes.health import probe_router as _probe_router
from .routes.health import router as _health_router
from .routes.mcp_routes import router as _mcp_router
from .routes.metrics import router as _metrics_router
from .routes.models import router as _models_router
from .routes.responses import router as _responses_router

app.include_router(_probe_router)
app.include_router(_health_router)
# Destructive control-plane routes (F-150 / F-151). Distinct router so the
# ``X-Rapid-MLX-Internal: true`` gate ALSO applies when ``--api-key`` is unset.
app.include_router(_health_admin_router)
app.include_router(_metrics_router)
app.include_router(_models_router)
app.include_router(_chat_router)
app.include_router(_completions_router)
app.include_router(_anthropic_router)
app.include_router(_responses_router)
app.include_router(_embeddings_router)
app.include_router(_mcp_router)
app.include_router(_audio_router)
app.include_router(_cache_router)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(
        description="Rapid-MLX OpenAI-compatible server for LLM and MLLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with simple mode (maximum throughput)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # Start with continuous batching (for multiple users)
    python -m vllm_mlx.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

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
        default="0.0.0.0",
        help="Host to bind to",
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
        help="Force loading as MLLM (multimodal language model)",
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
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help="Enable continuous batching (default: on).",
    )
    # PFlash long-prompt prefill compression (#287). Off by default. The
    # unified ``rapid-mlx serve`` CLI exposes the same surface; we mirror
    # it here so the standalone ``python -m vllm_mlx.server`` path is
    # not a silent gap (SOP §10).
    from .cli import _add_pflash_args as _add_pflash_args_to_server_parser

    _add_pflash_args_to_server_parser(parser)
    # Deprecated flags — accepted silently to avoid breaking user scripts
    import argparse as _ap

    parser.add_argument(
        "--simple-engine", action="store_true", default=False, help=_ap.SUPPRESS
    )
    parser.add_argument(
        "--kv-bits", type=int, default=None, choices=[4, 8], help=_ap.SUPPRESS
    )
    parser.add_argument("--kv-group-size", type=int, default=64, help=_ap.SUPPRESS)
    parser.add_argument("--draft-model", type=str, default=None, help=_ap.SUPPRESS)
    parser.add_argument("--num-draft-tokens", type=int, default=4, help=_ap.SUPPRESS)
    # TurboQuant flags — accepted but only functional via rapid-mlx serve (cli.py)
    parser.add_argument("--kv-cache-turboquant", action="store_true", help=_ap.SUPPRESS)
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
        "--cloud-model",
        type=str,
        default=None,
        help="Cloud model string for litellm (e.g. 'anthropic/claude-sonnet-4-5-20250929'). "
        "When set, large-context requests are routed to the cloud provider.",
    )
    parser.add_argument(
        "--cloud-threshold",
        type=int,
        default=20000,
        help="New token threshold to trigger cloud routing (default: 20000)",
    )
    parser.add_argument(
        "--cloud-api-base",
        type=str,
        default=None,
        help="Custom API base URL for cloud model (for OpenAI-compatible providers like Zhipu).",
    )
    parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="API key for cloud model (overrides environment variable).",
    )

    args = parser.parse_args()

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
    from .pflash import config_from_args as _server_pflash_config_from_args
    from .pflash import resolve_pflash_mode_default as _server_pflash_resolve_default
    from .pflash import validate_model_support as _server_pflash_validate
    from .scheduler import SchedulerConfig

    # Per-alias PFlash default (#287): verified Qwen3.5 / Qwen3.6 aliases
    # switch to ``always`` when the user passes no ``--pflash`` flag; all
    # other aliases keep the conservative ``off``. Explicit overrides win.
    args.pflash = _server_pflash_resolve_default(args, model_name=args.model)
    try:
        server_pflash_config = _server_pflash_config_from_args(args)
        _server_pflash_validate(
            server_pflash_config,
            model_name=args.model,
            is_mllm=getattr(args, "mllm", False) or is_mllm_model(args.model),
        )
    except ValueError as e:
        parser.error(str(e))

    scheduler_config = SchedulerConfig(
        prefill_step_size=args.prefill_step_size,
        pflash_config=server_pflash_config,
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
        cloud_model=args.cloud_model,
        cloud_threshold=args.cloud_threshold,
        cloud_api_base=args.cloud_api_base,
        cloud_api_key=args.cloud_api_key,
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
