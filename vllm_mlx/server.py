# SPDX-License-Identifier: Apache-2.0
"""
Unified OpenAI-compatible API server for vllm-mlx.

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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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
    HybridEngine,
    SimpleEngine,
)
from .runtime.model_registry import ModelEntry, ModelRegistry
from .service.helpers import (  # noqa: F401 — re-export for backward compat
    _FALLBACK_TEMPERATURE,
    _FALLBACK_TOP_P,
    _TOOL_USE_SYSTEM_SUFFIX,
    _build_usage,
    _disconnect_guard,
    _extract_token_logprob,
    _inject_json_instruction,
    _maybe_pin_system_prompt,
    _parse_tool_calls_with_parser,
    _resolve_max_tokens,
    _resolve_model_name,
    _resolve_temperature,
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
    return normalized.lower()


# Multi-model registry — supports loading 2+ models simultaneously.
# When populated, get_engine() routes by request model name.
# Backward-compatible: single-model mode still uses _engine global as before.
_model_registry = ModelRegistry()

# Global engine instance (single-model legacy path, also primary model in multi-model)
_engine: BaseEngine | None = None
# Serialize requests in SimpleEngine to prevent Metal/cache corruption from concurrent use
_inference_lock: asyncio.Lock | None = None
_model_name: str | None = None
_model_alias: str | None = None  # Short alias used to start the model (if any)
_model_path: str | None = (
    None  # Actual model path (for cache dir, not affected by --served-model-name)
)
_default_max_tokens: int = 4096
_thinking_token_budget: int = 2048  # Extra tokens added for thinking models
_default_timeout: float = 300.0  # Default request timeout in seconds (5 minutes)
_default_temperature: float | None = None  # Set via --default-temperature
_default_top_p: float | None = None  # Set via --default-top-p


# Global MCP manager
_mcp_manager = None
_mcp_executor = None

# Global embedding engine (lazy loaded)
_embedding_engine = None
_embedding_model_locked: str | None = None  # Set when --embedding-model is used

# API key authentication
_api_key: str | None = None
_auth_warning_logged: bool = False

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


async def lifespan(app: FastAPI):
    """FastAPI lifespan for startup/shutdown events."""
    global _engine, _mcp_manager

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
            import mlx.core as mx

            # Skip warmup for hybrid models (GatedDeltaNet) to avoid
            # contaminating compiled kernel state that interferes with
            # batched inference.  Check multiple engine wrappers:
            # BatchedEngine sets _hybrid_throttle via EngineCore,
            # HybridEngine wraps model in _shared_model,
            # SimpleEngine wraps in MLXLanguageModel._model.
            _is_hybrid = getattr(_engine, "_hybrid_throttle", False)
            if not _is_hybrid and not getattr(_engine, "_is_mllm", False):
                # Try to find the raw model through wrapper layers
                _model = getattr(_engine, "_model", None) or getattr(
                    _engine, "_shared_model", None
                )
                # Unwrap MLXLanguageModel if needed
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
                mx.eval(mx.zeros(1))  # Force sync
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

    # Initialize MCP if config provided
    mcp_config = os.environ.get("VLLM_MLX_MCP_CONFIG")
    if mcp_config:
        await init_mcp(mcp_config)

    yield

    # Shutdown: Save cache to disk BEFORE stopping engine
    if _engine is not None and hasattr(_engine, "save_cache_to_disk"):
        _save_prefix_cache_to_disk()

    # Shutdown: Close MCP connections and stop engine
    if _mcp_manager is not None:
        await _mcp_manager.stop()
        logger.info("MCP manager stopped")
    if _engine is not None:
        await _engine.stop()
        logger.info("Engine stopped")


app = FastAPI(
    title="Rapid-MLX API",
    description="OpenAI-compatible API for MLX LLM/MLLM inference on Apple Silicon",
    version="0.3.7",
    lifespan=lifespan,
)

# CORS configuration — configurable via --cors-origins CLI flag
_cors_origins: list[str] = ["*"]


def configure_cors(origins: list[str]) -> None:
    """Configure CORS middleware with the given allowed origins."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def serialize_inference(request: Request, call_next):
    """Serialize inference requests in SimpleEngine to prevent Metal crashes.

    SimpleEngine uses a single prompt cache and Metal command buffer — concurrent
    requests cause memory corruption. This middleware queues requests so only one
    runs at a time. BatchedEngine handles concurrency natively via its Scheduler.
    """
    if _inference_lock is None or not request.url.path.startswith("/v1/chat"):
        return await call_next(request)
    async with _inference_lock:
        return await call_next(request)


# Auth and rate limiting — moved to middleware/auth.py
from .middleware.auth import (  # noqa: E402
    RateLimiter,  # noqa: F401
    check_rate_limit,  # noqa: F401
    verify_api_key,  # noqa: F401
)
from .middleware.auth import (
    rate_limiter as _rate_limiter,  # noqa: F401 — configured in main()
)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions so they return JSON 500 instead of killing
    the connection. This keeps the server alive for subsequent requests."""
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    from starlette.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}},
    )


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
    use_batching: bool = False,
    scheduler_config=None,
    stream_interval: int = 1,
    max_tokens: int = 32768,
    force_mllm: bool = False,
    gpu_memory_utilization: float = 0.90,
    draft_model: str | None = None,
    num_draft_tokens: int = 4,
    prefill_step_size: int = 2048,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    cloud_model: str | None = None,
    cloud_threshold: int = 20000,
    cloud_api_base: str | None = None,
    cloud_api_key: str | None = None,
    served_model_name: str | None = None,
    mtp: bool = False,
    specprefill_enabled: bool = False,
    specprefill_threshold: int = 8192,
    specprefill_keep_pct: float = 0.3,
    specprefill_draft_model: str = None,
):
    """
    Load a model (auto-detects MLLM vs LLM).

    Args:
        model_name: HuggingFace model name or local path
        use_batching: Use continuous batching (BatchedEngine) vs simple mode (SimpleEngine)
        scheduler_config: Scheduler config for batched mode
        stream_interval: Tokens to batch before streaming (batched mode only)
        max_tokens: Default max tokens for generation
        force_mllm: Force loading as MLLM even if not auto-detected
        gpu_memory_utilization: Fraction of device memory for Metal allocation
            limit and emergency threshold (0.0-1.0, default 0.90)
        draft_model: Optional draft model for speculative decoding
        num_draft_tokens: Number of tokens to generate speculatively per step
        prefill_step_size: Tokens to process per prefill chunk (default: 2048)
        kv_bits: KV cache quantization bits (None=no quantization, 4 or 8)
        kv_group_size: Group size for KV cache quantization (default: 64)
        mtp: Enable native MTP speculative decoding (SimpleEngine only)
        specprefill_enabled: Enable SpecPrefill (SimpleEngine only)
        specprefill_threshold: Minimum suffix tokens to trigger SpecPrefill (default: 8192)
        specprefill_keep_pct: Fraction of tokens to keep (default: 0.3)
        specprefill_draft_model: Path to small draft model for SpecPrefill scoring
    """
    global \
        _engine, \
        _model_name, \
        _model_path, \
        _default_max_tokens, \
        _tool_parser_instance, \
        _cloud_router, \
        _inference_lock

    # Only serialize requests for SimpleEngine (single prompt cache, no concurrency)
    # BatchedEngine handles concurrency natively via Scheduler
    _inference_lock = None if use_batching else asyncio.Lock()
    _default_max_tokens = max_tokens
    _model_path = model_name
    _model_name = served_model_name or model_name
    # Reset tool parser instance when model is reloaded (tokenizer may change)
    _tool_parser_instance = None

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

    if force_mllm:
        logger.info("Force MLLM mode enabled via --mllm flag")

    if draft_model:
        logger.info(f"Speculative decoding enabled with draft model: {draft_model}")
        logger.info(f"  num_draft_tokens: {num_draft_tokens}")

    if use_batching and draft_model:
        # Hybrid mode: shared model with speculative decoding + continuous batching
        logger.info(f"Loading model with HybridEngine: {model_name}")
        logger.info(
            "  Hybrid mode: speculative decoding for single user, "
            "batching for multiple users"
        )
        _engine = HybridEngine(
            model_name=model_name,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
        )
        # HybridEngine will be started in lifespan (uvicorn's event loop)
        logger.info(f"Model loaded (hybrid mode): {model_name}")
    elif use_batching:
        logger.info(f"Loading model with BatchedEngine: {model_name}")
        _engine = BatchedEngine(
            model_name=model_name,
            scheduler_config=scheduler_config,
            stream_interval=stream_interval,
            force_mllm=force_mllm,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # BatchedEngine will be started in lifespan (uvicorn's event loop)
        # Just log for now
        logger.info(f"Model loaded (batched mode): {model_name}")
    else:
        logger.info(f"Loading model with SimpleEngine: {model_name}")
        _engine = SimpleEngine(
            model_name=model_name,
            force_mllm=force_mllm,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            mtp=mtp,
            specprefill_enabled=specprefill_enabled,
            specprefill_threshold=specprefill_threshold,
            specprefill_keep_pct=specprefill_keep_pct,
            specprefill_draft_model=specprefill_draft_model,
        )
        # Start SimpleEngine synchronously (no background loop)
        # Use new_event_loop() for Python 3.10+ compatibility (get_event_loop() is deprecated)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_engine.start())
        model_type = "MLLM" if _engine.is_mllm else "LLM"
        logger.info(f"{model_type} model loaded (simple mode): {model_name}")

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

    # Sync globals into ServerConfig so route modules can use get_config()
    _sync_config()


def _sync_config() -> None:
    """Copy server globals into the ServerConfig singleton.

    Called after load_model() and whenever globals change.
    Bridges the old global-variable pattern with the new config object.
    """
    cfg = get_config()
    cfg.engine = _engine
    cfg.model_name = _model_name
    cfg.model_alias = _model_alias
    cfg.model_path = _model_path
    cfg.inference_lock = _inference_lock
    cfg.default_max_tokens = _default_max_tokens
    cfg.default_timeout = _default_timeout
    cfg.default_temperature = _default_temperature
    cfg.default_top_p = _default_top_p
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
        from vllm_mlx.mcp import MCPClientManager, ToolExecutor, load_mcp_config

        config = load_mcp_config(config_path)
        _mcp_manager = MCPClientManager(config)
        await _mcp_manager.start()

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
from .routes.chat import router as _chat_router
from .routes.completions import router as _completions_router
from .routes.embeddings import router as _embeddings_router
from .routes.health import router as _health_router
from .routes.mcp_routes import router as _mcp_router
from .routes.models import router as _models_router

app.include_router(_health_router)
app.include_router(_models_router)
app.include_router(_chat_router)
app.include_router(_completions_router)
app.include_router(_anthropic_router)
app.include_router(_embeddings_router)
app.include_router(_mcp_router)
app.include_router(_audio_router)


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
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn",
    )
    parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force loading as MLLM (multimodal language model)",
    )
    parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help="Enable continuous batching (default: on). Use --simple-engine to disable.",
    )
    parser.add_argument(
        "--simple-engine",
        action="store_true",
        help="Use legacy SimpleEngine instead of BatchedEngine (single-user, no batching)",
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
        default=4096,
        help="Default max tokens for generation (caps when client sends None)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (if not set, no auth required)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Default request timeout in seconds (default: 300)",
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
        help="Pre-load an embedding model at startup (e.g. mlx-community/all-MiniLM-L6-v2-4bit)",
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
        "--draft-model",
        type=str,
        default=None,
        help="Draft model for speculative decoding (must use same tokenizer as main model)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help="Number of tokens to generate speculatively per step (default: 4)",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Tokens to process per prefill chunk (default: 2048). "
        "Larger values may improve TTFT on Apple Silicon with sufficient memory.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="KV cache quantization bits (4 or 8). Reduces memory for long contexts.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
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
    uvicorn_log_level = configure_logging(args.log_level)

    # Set global configuration
    global _api_key, _default_timeout, _rate_limiter
    global _default_temperature, _default_top_p
    _api_key = args.api_key
    _default_timeout = args.timeout
    if args.default_temperature is not None:
        _default_temperature = args.default_temperature
    if args.default_top_p is not None:
        _default_top_p = args.default_top_p

    # Configure rate limiter
    if args.rate_limit > 0:
        _rate_limiter = RateLimiter(requests_per_minute=args.rate_limit, enabled=True)
        logger.info(
            f"Rate limiting enabled: {args.rate_limit} requests/minute per client"
        )

    # Security summary at startup
    logger.info("=" * 60)
    logger.info("SECURITY CONFIGURATION")
    logger.info("=" * 60)
    if _api_key:
        logger.info("  Authentication: ENABLED (API key required)")
    else:
        logger.warning("  Authentication: DISABLED - Use --api-key to enable")
    if args.rate_limit > 0:
        logger.info(f"  Rate limiting: ENABLED ({args.rate_limit} req/min)")
    else:
        logger.warning("  Rate limiting: DISABLED - Use --rate-limit to enable")
    logger.info(f"  Request timeout: {args.timeout}s")
    logger.info("=" * 60)

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["VLLM_MLX_MCP_CONFIG"] = args.mcp_config

    # Auto-detect parser config from model name when not explicitly set
    if not args.tool_call_parser or not args.reasoning_parser:
        from .model_auto_config import detect_model_config

        auto_config = detect_model_config(args.model)
        if auto_config:
            if not args.tool_call_parser and auto_config.tool_call_parser:
                args.tool_call_parser = auto_config.tool_call_parser
                logger.info(
                    f"Auto-configured --tool-call-parser {auto_config.tool_call_parser}"
                )
            if not args.reasoning_parser and auto_config.reasoning_parser:
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

    # Pre-load embedding model if specified
    load_embedding_model(args.embedding_model, lock=True)

    # Load model before starting server
    load_model(
        args.model,
        use_batching=args.continuous_batching
        and not getattr(args, "simple_engine", False),
        max_tokens=args.max_tokens,
        force_mllm=args.mllm,
        draft_model=args.draft_model,
        num_draft_tokens=args.num_draft_tokens,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        cloud_model=args.cloud_model,
        cloud_threshold=args.cloud_threshold,
        cloud_api_base=args.cloud_api_base,
        cloud_api_key=args.cloud_api_key,
    )

    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level=uvicorn_log_level)


if __name__ == "__main__":
    main()
