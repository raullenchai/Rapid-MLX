# SPDX-License-Identifier: Apache-2.0
"""
vllm-mlx: Apple Silicon MLX backend for vLLM

This package provides native Apple Silicon GPU acceleration for vLLM
using Apple's MLX framework, mlx-lm for LLMs, and mlx-vlm for
vision-language models.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Support for LLM and multimodal models
"""

__version__ = "0.2.5"

# All imports are lazy to allow usage on non-Apple Silicon platforms
# (e.g., CI running on Linux) where mlx_lm is not available.


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
