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

    # --- Defaults ---
    default_max_tokens: int = 4096
    thinking_token_budget: int = 2048
    default_timeout: float = 300.0
    default_temperature: float | None = None
    default_top_p: float | None = None

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
