# SPDX-License-Identifier: Apache-2.0
"""
Server configuration — replaces 30+ global variables in server.py.

All server-wide state lives here in a single ServerConfig instance,
accessible from routes and middleware via `get_config()`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..engine.base import BaseEngine


@dataclass
class ServerConfig:
    """All server-wide mutable state in one place.

    Instead of 30+ module-level globals scattered across server.py,
    all state is accessed through this config object. Routes and
    middleware import ``get_config()`` to access it.
    """

    # --- Engine ---
    engine: BaseEngine | None = None
    model_name: str | None = None
    model_alias: str | None = None
    model_path: str | None = None
    inference_lock: asyncio.Lock | None = None
    # True only after lifespan has finished engine.start(), warmup,
    # prefix-cache load_from_disk, and MCP init. Used by /health/ready
    # so callers (e.g. validation pipelines) can wait for a real
    # readiness signal rather than racing the first inference against
    # in-progress warmup. Reset to False during lifespan shutdown.
    ready: bool = False

    # Bind address and port stashed by the CLI before uvicorn.run() so the
    # lifespan hook can print the "Ready:" banner with the real URL only
    # AFTER warmup completes (and the port is actually bound). Without this
    # the banner prints before uvicorn binds the port, and a user who curls
    # immediately gets a connection-refused.
    #
    # In the ``--listen-fd`` socket-activation branch, the supervisor owns
    # the bound address; the CLI populates ``bind_listen_fd`` instead, and
    # the lifespan banner prints the fd form. Mutually exclusive with the
    # host/port pair — see ``cli._run_uvicorn``.
    bind_host: str | None = None
    bind_port: int | None = None
    bind_listen_fd: int | None = None

    # --- Defaults ---
    default_max_tokens: int = 4096
    thinking_token_budget: int = 2048
    # 1800s (30 min) matches vLLM and most OpenAI-compat proxy
    # defaults. The old 300s default silently truncated reasoning
    # generations (DeepSeek-R1, Qwen-thinking) and 30B+ greedy
    # completions that needed >5 min of decode. Override via
    # CLI ``--timeout`` or per-request ``timeout`` field.
    default_timeout: float = 1800.0
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_min_p: float | None = None
    default_repetition_penalty: float | None = None
    default_presence_penalty: float | None = None
    default_frequency_penalty: float | None = None

    # --- Sampling overlay (layers 3 & 4 of the resolve chain) ---
    # Resolve order for every sampling param:
    #   1. request body
    #   2. CLI --default-* flag (``default_temperature``, etc.)
    #   3. AliasProfile.recommended_sampling   (this dict)
    #   4. generation_config.json from model snapshot   (this dict)
    #   5. hard-coded fallback (only temperature + top_p)
    # Both overlays are populated by ``server.load_model()`` once the
    # model path is known. Empty dicts when unset.
    alias_recommended_sampling: dict[str, float | int] | None = None
    generation_config_sampling: dict[str, float | int] | None = None

    # --- Tool calling ---
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    tool_parser_instance: Any = None
    enable_tool_logits_bias: bool = False

    # --- Reasoning ---
    reasoning_parser: Any = None
    reasoning_parser_name: str | None = None

    # --- MCP ---
    mcp_manager: Any = None
    mcp_executor: Any = None

    # --- Embeddings ---
    embedding_engine: Any = None
    embedding_model_locked: str | None = None

    # --- Auth ---
    api_key: str | None = None

    # --- Request-body size cap (DoS defense, #463 / rapid-desktop#273) ---
    # Hard cap on the wire-level request body size (bytes). Enforced at
    # the ASGI layer by ``RequestBodyLimitMiddleware`` BEFORE FastAPI
    # JSON parsing or tokenization runs, so an attacker who sends a
    # multi-MB body (10 MB → ~60 s hang, 100 MB → ~90 s hang were
    # observed pre-fix on a 27B alias) is bounced with 413 in
    # microseconds instead of starving a worker through full prefill.
    #
    # Default 8 MiB comfortably fits a 128k-token Qwen prompt
    # (≈ 500 KB JSON) plus tool schemas and a small inline image_url,
    # while still rejecting the 10–100 MB DoS payloads documented in
    # rapid-desktop#273. Overridable via ``--max-request-bytes`` or
    # ``RAPID_MLX_MAX_REQUEST_BYTES``; ``0`` disables the cap.
    max_request_bytes: int = 8 * 1024 * 1024

    # --- SSE keepalive (DoS / proxy idle-timeout defense, F-070) ---
    # Interval (seconds) at which ``_disconnect_guard`` emits an SSE
    # comment line (``: keepalive\n\n``) when the upstream generator
    # is silent. Prefill on long prompts (e.g. 64k-token Qwen) can
    # produce >60 s of pure TCP silence between the HTTP headers and
    # the first content delta — that silently kills EventSource clients
    # (browser ~45 s), nginx (``proxy_read_timeout 60``), Cloudflare
    # (100 s), and most reverse proxies. Emitting a comment line at a
    # fixed cadence keeps the connection alive without polluting the
    # parsed event stream (SSE comments start with ``:`` and are
    # ignored by every conforming consumer).
    #
    # Default 20 s sits comfortably below the tightest common idle
    # timeout (30 s — some SaaS gateways) while staying invisible to
    # short generations. Set to 0 via ``RAPID_MLX_SSE_KEEPALIVE_SECONDS=0``
    # to disable the heartbeat entirely (escape hatch for operators
    # whose upstream proxies are configured with generous timeouts).
    sse_keepalive_seconds: float = 20.0

    # --- Body-receive idle timeout (slow-DoS defense, F-072) ---
    # Maximum allowed idle time (seconds) between consecutive
    # ``http.request`` ASGI messages for a body-carrying request. When
    # exceeded, ``RequestBodyLimitMiddleware`` emits HTTP 408 and tears
    # down the connection. Defends against the slowloris-class attack
    # documented in F-072: send POST headers advertising
    # ``Content-Length: 1000000`` but then hold the socket open
    # forever without sending body bytes — pre-fix, the worker stayed
    # pinned indefinitely (≥30 s observed). With a fleet of such
    # sockets an attacker can starve the worker pool.
    #
    # Default 15 s is generous enough for honest slow clients on
    # mobile / satellite links (a 1 MB body at 64 kbps takes ~130 s
    # but ships ≥1 KB chunks every ~125 ms, well under the per-chunk
    # deadline) while bouncing connections that ship nothing. Set to 0
    # via ``RAPID_MLX_BODY_RECEIVE_TIMEOUT_SECONDS=0`` to disable.
    body_receive_timeout_seconds: float = 15.0

    # --- Cloud routing ---
    cloud_router: Any = None

    # --- Behavior flags ---
    gc_control: bool = True
    no_thinking: bool = False
    pin_system_prompt: bool = False
    pinned_system_prompt_hash: str | None = None

    # --- Multi-model ---
    model_registry: Any = None


# Singleton instance
_config = ServerConfig()


def get_config() -> ServerConfig:
    """Get the server config singleton."""
    return _config


def reset_config() -> ServerConfig:
    """Reset config to defaults (for testing)."""
    global _config
    _config = ServerConfig()
    return _config
