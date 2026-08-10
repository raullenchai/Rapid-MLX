"""Authenticated control plane for loading and evicting resident models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from ..config import get_config
from ..middleware.auth import verify_api_key
from ..runtime.resident_models import (
    ResidentModelBusyError,
    ResidentModelCapacityError,
    ResidentModelError,
)

router = APIRouter(dependencies=[Depends(verify_api_key)])


class ModelLoadRequest(BaseModel):
    model: str = Field(..., min_length=1)
    model_path: str | None = None
    estimated_size_gb: float | None = Field(default=None, gt=0, le=1024)
    pin: bool = False
    replace_group: str | None = Field(default=None, pattern="^(assistant)$")


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

    return _manager().snapshot()


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
        record = await manager.load(
            request.model,
            model_path=request.model_path,
            estimated_bytes=estimated_bytes,
            pin=request.pin,
            replace_group=request.replace_group,
        )
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
