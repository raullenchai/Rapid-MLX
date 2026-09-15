# SPDX-License-Identifier: Apache-2.0
"""Authenticated HTTP surface for bounded local agent runs."""

from __future__ import annotations

import asyncio
from threading import RLock
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..agent_runtime.server import (
    AgentApprovalRequest,
    AgentEventsView,
    AgentRunCapacityError,
    AgentRunConflictError,
    AgentRunCreateRequest,
    AgentRunNotFoundError,
    AgentRunView,
    AgentServerService,
    AgentToolRegistryUnavailableError,
    AgentToolResultRequest,
    AgentToolSelectionError,
)
from ..config import get_config
from ..middleware.auth import check_rate_limit, verify_api_key
from ..model_metadata import read_model_metadata
from ..service.helpers import _validate_model_name

router = APIRouter(
    prefix="/v1/agent",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)

_service: AgentServerService | None = None
_service_lock = RLock()
_service_shutting_down = False
_METADATA_UNSET = object()
_single_model_metadata_cache: tuple[object, str, dict | None] | None = None


def get_agent_service() -> AgentServerService:
    global _service
    with _service_lock:
        if _service_shutting_down:
            raise AgentRunCapacityError("agent runtime is shutting down")
        if _service is None:
            _service = AgentServerService()
        return _service


async def close_agent_service() -> None:
    global _service, _service_shutting_down
    with _service_lock:
        if _service_shutting_down:
            return
        _service_shutting_down = True
        service = _service
    if service is not None:
        await service.close()
    with _service_lock:
        if _service is service:
            _service = None


def start_agent_service_lifecycle() -> None:
    """Explicitly open the singleton gate for a new FastAPI lifespan."""

    global _service_shutting_down, _single_model_metadata_cache
    with _service_lock:
        if _service is not None:
            raise RuntimeError("agent service survived the previous lifespan")
        _service_shutting_down = False
        _single_model_metadata_cache = None


async def _entry_model_config(entry) -> dict | None:
    """Read immutable metadata once for this concrete registry generation."""

    cached = getattr(entry, "_agent_profile_model_config", _METADATA_UNSET)
    if cached is not _METADATA_UNSET:
        return cast(dict | None, cached)
    metadata = await asyncio.to_thread(read_model_metadata, entry.model_path)
    config = metadata.config if metadata is not None else None
    entry._agent_profile_model_config = config
    return config


async def _single_model_config(generation: object, path: str) -> dict | None:
    """Cache profile metadata for one concrete single-engine generation."""

    global _single_model_metadata_cache
    with _service_lock:
        cached = _single_model_metadata_cache
        if cached is not None and cached[0] is generation and cached[1] == path:
            return cached[2]
    metadata = await asyncio.to_thread(read_model_metadata, path)
    config = metadata.config if metadata is not None else None
    with _service_lock:
        _single_model_metadata_cache = (generation, path, config)
    return config


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AgentRunNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AgentRunCapacityError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AgentToolRegistryUnavailableError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AgentToolSelectionError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.post(
    "/runs",
    response_model=AgentRunView,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_agent_run(request: AgentRunCreateRequest) -> AgentRunView:
    cfg = get_config()
    request_model = request.model or cfg.model_alias or cfg.model_name
    if not request_model:
        raise HTTPException(status_code=503, detail="no text model is configured")
    _validate_model_name(request_model)
    profile_model = cfg.model_path or cfg.model_alias or request_model
    profile_model_config = None
    profile_tool_call_parser = cfg.tool_call_parser
    model_generation = cfg.engine
    if cfg.model_registry is not None:
        try:
            entry = cfg.model_registry.get_entry(request_model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        profile_model = entry.model_path or entry.model_name
        try:
            profile_model_config = await _entry_model_config(entry)
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail="model metadata is temporarily unavailable",
            ) from exc
        profile_tool_call_parser = entry.tool_call_parser
        model_generation = entry
    elif cfg.model_path:
        try:
            profile_model_config = await _single_model_config(
                cfg.engine, cfg.model_path
            )
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail="model metadata is temporarily unavailable",
            ) from exc
    try:
        return await get_agent_service().create(
            request,
            model=profile_model,
            request_model=request_model,
            profile_model_config=profile_model_config,
            profile_tool_call_parser=profile_tool_call_parser,
            model_generation=model_generation,
        )
    except (
        AgentRunCapacityError,
        AgentToolSelectionError,
        AgentToolRegistryUnavailableError,
    ) as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}", response_model=AgentRunView)
async def get_agent_run(run_id: str) -> AgentRunView:
    try:
        return await get_agent_service().get(run_id)
    except (AgentRunNotFoundError, AgentRunCapacityError) as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}/events", response_model=AgentEventsView)
async def get_agent_events(
    run_id: str, after: int = Query(default=0, ge=0)
) -> AgentEventsView:
    try:
        return await get_agent_service().events(run_id, after=after)
    except (AgentRunNotFoundError, AgentRunCapacityError) as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/approval", response_model=AgentRunView)
async def approve_agent_action(
    run_id: str, request: AgentApprovalRequest
) -> AgentRunView:
    try:
        return await get_agent_service().approve(run_id, request)
    except (
        AgentRunNotFoundError,
        AgentRunConflictError,
        AgentRunCapacityError,
    ) as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/tool-result", response_model=AgentRunView)
async def submit_agent_tool_result(
    run_id: str, request: AgentToolResultRequest
) -> AgentRunView:
    try:
        return await get_agent_service().submit_result(run_id, request)
    except (
        AgentRunNotFoundError,
        AgentRunConflictError,
        AgentRunCapacityError,
    ) as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/cancel", response_model=AgentRunView)
async def cancel_agent_run(run_id: str) -> AgentRunView:
    try:
        return await get_agent_service().cancel(run_id)
    except (
        AgentRunNotFoundError,
        AgentRunConflictError,
        AgentRunCapacityError,
    ) as exc:
        raise _http_error(exc) from exc
