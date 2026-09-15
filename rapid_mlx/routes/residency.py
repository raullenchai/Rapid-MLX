"""Authenticated control plane for loading and evicting resident models."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, StrictBool, model_validator

from ..config import get_config
from ..kv_cache_dtype import KVCacheQuantizationUnsupportedError
from ..middleware.auth import verify_api_key
from ..middleware.exception_handlers import (
    register_request_model,
    register_request_path,
)
from ..model_aliases import resolve_profile
from ..runtime.resident_models import (
    ResidentModelBusyError,
    ResidentModelCapacityError,
    ResidentModelError,
    ResidentPerformanceConfig,
    resolve_resident_performance,
)

router = APIRouter(dependencies=[Depends(verify_api_key)])


class ModelPerformanceRequest(BaseModel):
    kv_cache_dtype: Literal["bf16", "int8", "int4"] | None = None
    kv_cache_turboquant: Literal["v4", "k8v4"] | None = None
    prefix_cache_enabled: bool | None = None
    cache_memory_mb: int | None = Field(default=None, ge=256, le=32768)

    @model_validator(mode="after")
    def one_kv_mode(self):
        if self.kv_cache_dtype is not None and self.kv_cache_turboquant is not None:
            raise ValueError("KV dtype and TurboQuant are mutually exclusive")
        return self

    def runtime_value(self) -> ResidentPerformanceConfig | None:
        value = ResidentPerformanceConfig(**self.model_dump())
        return None if value.is_empty else value


class ModelLoadRequest(BaseModel):
    model: str = Field(..., min_length=1)
    model_path: str | None = None
    estimated_size_gb: float | None = Field(default=None, gt=0, le=1024)
    # ``StrictBool`` requires an actual JSON boolean on the wire. Pydantic
    # v2's lax mode coerces ``"yes"`` / ``"on"`` / ``1`` / ``0`` onto
    # ``bool``, so a string like ``"yes"`` silently became ``True`` and
    # triggered a real resident-model reload (issue #2362). Strict mode
    # keeps ``true``/``false`` identical while rejecting every non-boolean
    # wire form with a 4xx naming the field via the unified validation
    # envelope.
    pin: StrictBool = False
    replace_group: str | None = Field(default=None, pattern="^(assistant)$")
    image_mode: Literal["generation", "editing"] | None = None
    performance: ModelPerformanceRequest | None = None
    reload_if_changed: StrictBool = False
    replace_mode: Literal["reject", "wait", "abort"] = "reject"
    memory_policy: Literal["keep_then_commit", "evict_first_if_needed"] = (
        "evict_first_if_needed"
    )


class ModelPinRequest(BaseModel):
    pinned: StrictBool = True


# Register the load-endpoint request model with the safe error-location
# contract (D-ENVELOPE-FIELD-LEAK) so a validation 400 on a schema-owned
# `performance` / `estimated_size_gb` / `replace_group` setting reports the
# real field path in `error.message` and mirrors it into `error.param`,
# exactly as the chat endpoints do for their request models. Without this,
# every string loc component collapses to the `<field>` placeholder and
# `error.param` stays null (VAL-2361). The plugin-style registration is
# called at import time and stays idempotent; the middleware module itself
# deliberately avoids importing this engine-heavy route module.
register_request_model(ModelLoadRequest)
register_request_path("/v1/models/load", ModelLoadRequest)


def _manager():
    manager = get_config().residency_manager
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Resident model management is not initialized",
        )
    return manager


def _record_payload(record) -> dict:
    return next(
        item
        for item in _manager().snapshot()["models"]
        if item["id"] == record.model_id
    )


@router.get("/v1/models/residency")
async def model_residency():
    """Return resident models and actual process usage against the ceiling."""

    from ..runtime.audio_worker import audio_worker

    snapshot = _manager().snapshot()
    snapshot["audio_lanes"] = audio_worker.snapshot()
    return snapshot


def _resolved_group_for_profile(modality: str) -> str:
    """Map a request-facing profile modality to its lifecycle replacement group.

    Mirrors ``resident_models._replacement_group`` (assistant for text, mllm,
    vision, and text-diffusion): a diffusion-gemma-26b checkpoint runs as a text
    engine whose lifecycle group is "assistant", so the request-facing profile
    modality ("text-diffusion") must resolve to the same group or the
    ``resolved_group != replace_group`` guard raises a spurious 409 when the
    Desktop picks it with a chat model resident.  Kept in one place so both the
    request FIX path here and the engine-derived group stay in parity
    (#0131-routing-groups Fix 2).
    """
    return "assistant" if modality in {"text", "vision", "text-diffusion"} else modality


@router.post("/v1/models/load")
async def load_resident_model(request: ModelLoadRequest):
    """Load a model into the current process, evicting idle LRU entries first."""

    manager = _manager()
    estimated_bytes = (
        int(request.estimated_size_gb * 1024**3)
        if request.estimated_size_gb is not None
        else None
    )
    try:
        performance = (
            request.performance.runtime_value() if request.performance else None
        )
        model_profile = resolve_profile(request.model)
        path_profile = (
            resolve_profile(request.model_path) if request.model_path else None
        )
        # The loader consumes model_path when supplied, so only that
        # checkpoint's metadata may authorize destructive admission. An
        # unknown override falls back to keep-then-commit rather than trusting
        # the request-facing alias for different weights.
        profile = path_profile if request.model_path else model_profile
        resolved_group = None
        if profile is not None:
            resolved_group = _resolved_group_for_profile(profile.modality)
        if (
            performance is not None
            and profile is not None
            and profile.modality
            in {
                "image-gen",
                "text-diffusion",
                "video-gen",
                "audio",
            }
        ):
            raise HTTPException(
                status_code=422,
                detail="Performance overrides are only supported for text models.",
            )
        performance = resolve_resident_performance(
            performance,
            model_name=request.model,
            model_path=request.model_path,
        )
        record = await manager.load(
            request.model,
            model_path=request.model_path,
            estimated_bytes=estimated_bytes,
            pin=request.pin,
            replace_group=request.replace_group,
            image_mode=request.image_mode,
            performance=performance,
            reload_if_changed=request.reload_if_changed,
            replace_mode=request.replace_mode,
            memory_policy=request.memory_policy,
            resolved_group=resolved_group,
        )
    except HTTPException:
        raise
    except KVCacheQuantizationUnsupportedError as exc:
        # Explicit quantized-KV request the model can't serve (#78):
        # reject before load with the actionable reason.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ResidentModelCapacityError as exc:
        projection = exc.replacement_projection
        if projection is None:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        raise HTTPException(
            status_code=507,
            detail={
                "error": {
                    "message": str(exc),
                    "type": "insufficient_capacity_error",
                    "code": "insufficient_capacity_error",
                    "param": "estimated_size_gb",
                },
                "replacement_projection": projection.payload(),
            },
        ) from exc
    except ResidentModelError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load resident model: {type(exc).__name__}",
        ) from exc
    return _record_payload(record)


@router.put("/v1/models/{model_id:path}/pin")
async def pin_resident_model(model_id: str, request: ModelPinRequest):
    try:
        record = await _manager().set_pinned(model_id, request.pinned)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model is not resident") from exc
    except ResidentModelError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _record_payload(record)


@router.delete("/v1/models/{model_id:path}", status_code=204)
async def unload_resident_model(model_id: str):
    try:
        await _manager().unload(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model is not resident") from exc
    except ResidentModelBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ResidentModelError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)
