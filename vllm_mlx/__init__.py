# SPDX-License-Identifier: Apache-2.0
"""
Rapid-MLX: fast local LLM inference for Apple Silicon.

A standalone OpenAI-compatible inference server built on Apple's MLX
framework, mlx-lm for LLMs, and mlx-vlm for vision-language models.

Features:
- Continuous batching scheduler
- OpenAI-compatible API server
- Support for LLM and multimodal models
"""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("rapid-mlx")
except Exception:
    __version__ = "0.0.0"  # fallback for editable installs without metadata

# Rebrand runtime logger names from the legacy ``vllm_mlx.*`` namespace to
# the product-facing ``rapid_mlx.*`` namespace before any submodule has had
# a chance to create a record. The Python package directory keeps the
# ``vllm_mlx/`` name (renaming would touch hundreds of imports and break
# external integrations); only what users see in log output changes. The
# rebrand is a single ``logging.setLogRecordFactory`` call, idempotent and
# scoped to the ``vllm_mlx`` prefix -- uvicorn/fastapi/asyncio/httpx
# namespaces flow through untouched. See ``_log_namespace`` for the
# rationale (handler-attached filters and logger-attached filters were both
# rejected; factory is the only path that catches records from descendant
# loggers without imposing churn on every ``getLogger(__name__)`` site).
from vllm_mlx._log_namespace import install_log_namespace_rebrand

install_log_namespace_rebrand()

# All imports are lazy to allow usage on non-Apple Silicon platforms
# (e.g., CI running on Linux) where mlx_lm is not available. The MLX
# hardware-compat shim (#404 M5 single-stream) lives in `_mlx_compat`
# and is installed at the top of every submodule that imports
# `mlx_lm.generate` — NOT here, so that `import vllm_mlx` stays free of
# mlx.core import (which can SIGABRT on systems with mlx installed but
# Metal unavailable).


def __getattr__(name):
    """Lazy load all components to avoid mlx_lm import on non-Apple platforms."""
    # Request management
    if name in ("Request", "RequestOutput", "RequestStatus", "SamplingParams"):
        from vllm_mlx import request

        return getattr(request, name)

    # Scheduler
    if name in ("Scheduler", "SchedulerConfig", "SchedulerOutput"):
        from vllm_mlx import scheduler

        return getattr(scheduler, name)

    # Engine
    if name in ("EngineCore", "AsyncEngineCore", "EngineConfig"):
        from vllm_mlx import engine_core

        return getattr(engine_core, name)

    # Prefix cache
    if name in ("PrefixCacheManager", "PrefixCacheStats", "BlockAwarePrefixCache"):
        from vllm_mlx import prefix_cache

        return getattr(prefix_cache, name)

    # Paged cache
    if name in ("PagedCacheManager", "CacheBlock", "BlockTable", "CacheStats"):
        from vllm_mlx import paged_cache

        return getattr(paged_cache, name)

    # MLLM cache (with legacy VLM aliases)
    if name in (
        "MLLMCacheManager",
        "MLLMCacheStats",
        "VLMCacheManager",
        "VLMCacheStats",
    ):
        from vllm_mlx import mllm_cache

        # Map legacy VLM names to MLLM
        mllm_name = name.replace("VLM", "MLLM") if name.startswith("VLM") else name
        return getattr(mllm_cache, mllm_name)

    # Model registry
    if name in ("get_registry", "ModelOwnershipError"):
        from vllm_mlx import model_registry

        return getattr(model_registry, name)

    # vLLM integration components (require torch)
    if name == "MLXPlatform":
        from vllm_mlx.vllm_platform import MLXPlatform

        return MLXPlatform
    if name == "MLXWorker":
        from vllm_mlx.worker import MLXWorker

        return MLXWorker
    if name == "MLXModelRunner":
        from vllm_mlx.model_runner import MLXModelRunner

        return MLXModelRunner
    if name == "MLXAttentionBackend":
        from vllm_mlx.attention import MLXAttentionBackend

        return MLXAttentionBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core (lazy loaded, require torch)
    "MLXPlatform",
    "MLXWorker",
    "MLXModelRunner",
    "MLXAttentionBackend",
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (LLM)
    "PrefixCacheManager",
    "PrefixCacheStats",
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "CacheStats",
    # MLLM cache (images/videos)
    "MLLMCacheManager",
    "MLLMCacheStats",
    # Legacy aliases
    "VLMCacheManager",
    "VLMCacheStats",
    # Version
    "__version__",
]
