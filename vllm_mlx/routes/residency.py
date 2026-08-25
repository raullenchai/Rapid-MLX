"""Authenticated control plane for loading and evicting resident models."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, model_validator

from ..config import get_config
from ..middleware.auth import verify_api_key
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
    pin: bool = False
    replace_group: str | None = Field(default=None, pattern="^(assistant)$")
    image_mode: Literal["generation", "editing"] | None = None
    performance: ModelPerformanceRequest | None = None
    reload_if_changed: bool = False
    replace_mode: Literal["reject", "wait", "abort"] = "reject"


class ModelPinRequest(BaseModel):
    pinned: bool = True


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
        profile = resolve_profile(request.model) or (
            resolve_profile(request.model_path) if request.model_path else None
        )
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
        )
    except HTTPException:
        raise
    except ResidentModelCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
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
