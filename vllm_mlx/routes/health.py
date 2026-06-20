# SPDX-License-Identifier: Apache-2.0
"""Health, status, and cache management endpoints."""

import gc
import logging

from fastapi import APIRouter, Depends, HTTPException

from ..config import get_config
from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

# Probe endpoints (no auth) — exposed for k8s/LB liveness+readiness checks
# which cannot send Authorization headers by default. Without splitting,
# `--api-key X` makes every probe fail and pods get marked unhealthy.
probe_router = APIRouter()

# Management endpoints (auth-gated when --api-key is configured) — read-only
# status / cache stats. ``verify_api_key`` is a no-op when ``--api-key`` is
# unset; that's fine for status reads but NOT for destructive routes — those
# live on ``admin_router`` below.
router = APIRouter(dependencies=[Depends(verify_api_key)])

# Destructive control-plane routes (cache clear, request cancel). Per the
# operator-intent revert of #728, these run on plain ``verify_api_key`` —
# single-machine UX means the prior ``verify_internal_admin`` gate (header
# requirement) was friction for no benefit on the common deployment. The
# cancel envelope sanitization + scheduler ``abort_request`` correctness
# fix from #728 STAY — those are real bugs unrelated to the auth gate.
admin_router = APIRouter(dependencies=[Depends(verify_api_key)])


@probe_router.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Root path — returns a minimal alive response.

    Claude Code (and other Anthropic SDK clients) send ``HEAD /`` as a
    connectivity probe before attempting any API call. Without a handler
    here FastAPI returns 404, which the client interprets as "server
    unreachable" and aborts. This endpoint lives on ``probe_router``
    (no-auth) so the probe succeeds regardless of ``--api-key``.
    """
    return {"status": "ok"}


@probe_router.get("/health")
async def health():
    """Health check endpoint.

    `model_loaded` flips True as soon as the engine object exists
    (mid-lifespan), but warmup + prefix-cache load can still be in
    progress. `ready` flips True only after lifespan finishes all
    startup work — that's the signal callers actually want to gate
    on. Use /health/ready (returns 503 until ready) for poll-until-up
    callers; this endpoint is the human-readable view.
    """
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
        "ready": cfg.ready,
        "model_loaded": cfg.engine is not None,
        "model_name": cfg.model_name,
        "model_type": "mllm" if (cfg.engine and cfg.engine.is_mllm) else "llm",
        "engine_type": engine_stats.get("engine_type", "unknown"),
        "mcp": mcp_info,
    }


@probe_router.get("/health/ready")
async def health_ready():
    """Strict readiness probe — 503 until lifespan startup is fully done.

    Lifespan order is: engine.start() → warmup (Metal kernel JIT) →
    load_from_disk (prefix cache) → MCP init → set ready=True. The
    first inference request would otherwise compete with warmup +
    cache load and look like a hang. Validation pipelines and
    container orchestrators should poll this instead of /v1/models
    (which returns 200 the moment the FastAPI app binds).
    """
    cfg = get_config()
    if not cfg.ready:
        raise HTTPException(status_code=503, detail="model loading")
    return {"ready": True, "model": cfg.model_name}


@probe_router.get("/healthz")
async def healthz():
    """k8s-convention alias for /health. Many orchestrator templates default
    to /healthz; without this alias they 404 and the operator has to override
    every chart."""
    return await health()


@probe_router.get("/readyz")
async def readyz():
    """k8s-convention alias for /health/ready."""
    return await health_ready()


@probe_router.get("/livez")
async def livez():
    """k8s liveness probe — 200 if the process is alive. Does not check
    model readiness; for that use /readyz."""
    return {"status": "alive"}


@admin_router.post("/v1/requests/{request_id}/cancel")
async def cancel_request(request_id: str):
    """Cancel an active or queued request.

    The ``request_id`` is the ``chatcmpl-xxx`` ID returned in the first SSE
    streaming chunk (or in the non-streaming response body). Returns 404 if
    the request is not found, has already finished, or the loaded engine
    does not implement abort.

    F-151 hardening:
    * Schedulers used to return ``True`` for *any* string (the abort was
      enqueued unconditionally), so the route would 200-OK an attacker who
      poked random IDs — both an info leak (confirmed the route exists with
      a real engine behind it) and a validation bypass. The underlying
      schedulers now return False for unknown IDs; this route forwards that
      as 404.
    * The success response previously embedded ``cfg.model_name`` (which is
      the HF repo id when ``--served-model-name`` is not set), so any
      anonymous caller could learn which weights are loaded. We no longer
      echo the model in the cancel envelope — clients that need it can hit
      ``/v1/models``.
    * The 500 fallback used to include ``{exc}`` raw; engine exception
      messages sometimes carry the HF path. We now emit a generic message
      and rely on the server log for diagnosis.
    """
    cfg = get_config()
    if cfg.engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    try:
        aborted = await cfg.engine.abort_request(request_id)
    except Exception:  # pragma: no cover - engine-side errors are rare
        # F-151: don't echo the exception (some engine messages carry the
        # HF repo path / model snapshot location). The full traceback goes
        # to the server log for the operator.
        logger.exception("Failed to cancel request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel request (see server logs)",
        ) from None

    if not aborted:
        # F-151: keep the detail short and avoid echoing model_name. The
        # request_id IS user-supplied so echoing it back is fine; what we
        # must NOT echo is server-side state (model / engine internals).
        raise HTTPException(
            status_code=404,
            detail="Request not found or already finished",
        )

    logger.info("[cancel_request] accepted request_id=%s", request_id)
    # F-151: drop ``model`` field. Anyone who can cancel a request they own
    # already knows which model they targeted; an attacker who pokes random
    # IDs (now 404'd above) must not be able to fingerprint the loaded
    # weights via the success envelope.
    return {
        "object": "request.cancel",
        "id": request_id,
        "cancelled": True,
    }


@admin_router.delete("/v1/requests/{request_id}")
async def delete_request(request_id: str):
    """OpenAI-style alias for cancelling an active or queued request."""
    return await cancel_request(request_id)


@admin_router.post("/v1/cache/clear")
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
    bg = stats.get("batch_generator")
    if not isinstance(bg, dict):
        bg = {}

    # Coerce missing-or-None to a float zero. `or 0` would collapse a
    # legitimate 0.0 value to int 0; dashboards with strict number-type
    # schemas care about the difference.
    def _tps(key: str) -> float:
        v = bg.get(key)
        return 0.0 if v is None else v

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
        "generation_tps": _tps("generation_tps"),
        "prompt_tps": _tps("prompt_tps"),
        "metal": {
            "active_memory_gb": stats.get("metal_active_memory_gb"),
            "peak_memory_gb": stats.get("metal_peak_memory_gb"),
            "cache_memory_gb": stats.get("metal_cache_memory_gb"),
        },
        # Always emit an object (never null) so dashboards with strict
        # number-or-object schemas don't crash when prefix cache is
        # disabled via --disable-prefix-cache.
        "cache": (
            stats.get("memory_aware_cache")
            or stats.get("paged_cache")
            or stats.get("prefix_cache")
            or {"enabled": False}
        ),
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


@admin_router.delete("/v1/cache")
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
