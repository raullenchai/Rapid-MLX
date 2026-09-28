# SPDX-License-Identifier: Apache-2.0
"""Read-only effective runtime configuration endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from ..config import get_config
from ..middleware.auth import verify_api_key
from ..runtime.effective_config import EffectiveRuntimeConfig

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/v1/runtime/config")
async def effective_runtime_config() -> dict[str, object]:
    """Return the exact values and provenance used by the active engine."""

    server_config = get_config()
    config = server_config.effective_runtime_config
    model = server_config.effective_runtime_model
    if (
        not server_config.ready
        or not isinstance(config, EffectiveRuntimeConfig)
        or not isinstance(model, str)
        or not model.strip()
    ):
        raise HTTPException(status_code=503, detail="runtime config is not resolved")
    return {"model": model, **config.to_wire()}
