# SPDX-License-Identifier: Apache-2.0
"""Health, status, and cache management endpoints."""

import gc

from fastapi import APIRouter, HTTPException

from ..config import get_config

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint."""
    cfg = get_config()

    mcp_info = None
    if cfg.mcp_manager is not None:
        connected = sum(
            1
            for s in cfg.mcp_manager.get_server_status()
            if s.state.value == "connected"
        )
        total = len(cfg.mcp_manager.get_server_status())
        mcp_info = {
            "enabled": True,
            "servers_connected": connected,
            "servers_total": total,
            "tools_available": len(cfg.mcp_manager.get_all_tools()),
        }

    engine_stats = cfg.engine.get_stats() if cfg.engine else {}

    return {
        "status": "healthy",
        "model_loaded": cfg.engine is not None,
        "model_name": cfg.model_name,
        "model_type": "mllm" if (cfg.engine and cfg.engine.is_mllm) else "llm",
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }


@router.post("/v1/cache/clear")
async def clear_cache():
    """Clear the prompt KV cache."""
    cfg = get_config()
    if cfg.engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    model = getattr(cfg.engine, "_model", None)
    if model is not None and hasattr(model, "_prompt_cache"):
        model._prompt_cache = None
        model._cached_token_ids = []
        gc.collect()
        return {"status": "ok", "message": "Prompt cache cleared"}
    return {"status": "ok", "message": "No prompt cache to clear"}


@router.get("/v1/status")
async def status():
    """Real-time status with per-request details."""
    cfg = get_config()
    if cfg.engine is None:
        return {"status": "not_loaded", "model": None, "requests": []}

    stats = cfg.engine.get_stats()

    return {
        "status": "generating" if stats.get("running") else "idle",
        "model": cfg.model_name,
        "uptime_s": round(stats.get("uptime_seconds", 0), 1),
        "steps_executed": stats.get("steps_executed", 0),
        "num_running": stats.get("num_running", 0),
        "num_waiting": stats.get("num_waiting", 0),
        "total_requests_processed": stats.get("num_requests_processed", 0),
        "total_prompt_tokens": stats.get("total_prompt_tokens", 0),
        "total_completion_tokens": stats.get("total_completion_tokens", 0),
        "metal": {
            "active_memory_gb": stats.get("metal_active_memory_gb"),
            "peak_memory_gb": stats.get("metal_peak_memory_gb"),
            "cache_memory_gb": stats.get("metal_cache_memory_gb"),
        },
        "cache": stats.get("memory_aware_cache")
        or stats.get("paged_cache")
        or stats.get("prefix_cache"),
        "requests": stats.get("requests", []),
    }


@router.get("/v1/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    try:
        from mlx_vlm.utils import (
            get_multimodal_kv_cache_stats,
            get_pil_cache_stats,
            get_pixel_values_cache_stats,
        )

        return {
            "multimodal_kv_cache": get_multimodal_kv_cache_stats(),
            "pixel_values_cache": get_pixel_values_cache_stats(),
            "pil_image_cache": get_pil_cache_stats(),
        }
    except ImportError:
        return {
            "message": "Vision cache stats not available (text-only model loaded). "
            "Prompt cache is managed internally by the engine.",
            "model_type": "llm",
        }


@router.delete("/v1/cache")
async def clear_all_caches():
    """Clear all caches."""
    try:
        from mlx_vlm.utils import (
            clear_multimodal_kv_cache,
            clear_pixel_values_cache,
        )

        clear_multimodal_kv_cache()
        clear_pixel_values_cache()
        return {
            "status": "cleared",
            "caches": ["multimodal_kv", "pixel_values", "pil_image"],
        }
    except ImportError:
        return {"error": "Cache clear not available (mlx_vlm not loaded)"}
