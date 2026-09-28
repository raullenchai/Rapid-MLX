"""Server-side CUA surface: create/poll/approve/cancel computer-use runs.

Conventions follow ``routes/agents.py``: numbered events with ``after=SEQ``
polling (no SSE), a status machine, explicit approval endpoint, bearer auth +
rate limit on the whole router. The run itself is the productized
``rapid_mlx.cua`` loop; the computer (AX/CGEvent) is the shared resource, so
exactly one run may be active.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..cua import service as cua_service
from ..cua.config import delete_user_preset, load_config, save_user_preset
from ..middleware.auth import check_rate_limit, verify_api_key

router = APIRouter(
    prefix="/v1/cua",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)


class CUAPlannerInfo(BaseModel):
    name: str
    model: str
    url: str
    text_only: bool
    note: str = ""
    has_api_key: bool = False
    user_created: bool = False


class CUAPlannerCreateRequest(BaseModel):
    """User adds a cloud brain from the app settings.

    Providing api_key is the user's explicit consent to send task data to
    this endpoint; remote URLs must be HTTPS (validated on save).
    """

    name: str = Field(min_length=1, max_length=32)
    url: str = Field(min_length=8, max_length=2000)
    model: str = Field(min_length=1, max_length=500)
    api_key: str | None = Field(default=None, max_length=2000)
    reasoning_effort: str | None = Field(default=None, max_length=20)
    text_only: bool = False


class CUARunCreateRequest(BaseModel):
    app: str = Field(min_length=1, max_length=120)
    goal: str = Field(min_length=1, max_length=4000)
    planner: str = Field(default="local-27b", min_length=1, max_length=2000)
    planner_model: str | None = Field(default=None, max_length=500)
    planner_url: str | None = Field(default=None, max_length=2000)
    open_url: str = Field(default="", max_length=2000)
    allowed_domain: str = Field(default="", max_length=200)
    max_steps: int = Field(default=12, ge=1, le=40)
    human_login: bool = False


class CUARunCreated(BaseModel):
    run_id: str
    status: str


class CUARunView(BaseModel):
    run_id: str
    app: str
    goal: str
    status: str
    final_summary: str
    error: str
    planner: str
    events_after_seq: int
    events: list[dict[str, Any]]
    run_dir: str


class CUARunList(BaseModel):
    runs: list[dict[str, Any]]


class CUAApprovalResult(BaseModel):
    run_id: str
    approved: bool


def _service() -> cua_service.CUAService:
    return cua_service.get_cua_service()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, cua_service.CUARunNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, cua_service.CUARunConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.get("/planners", response_model=list[CUAPlannerInfo])
async def list_planners() -> list[CUAPlannerInfo]:
    presets = load_config()["presets"]
    return [
        CUAPlannerInfo(
            name=name,
            model=preset["model"],
            url=preset["url"],
            text_only=bool(preset.get("text_only", False)),
            note=preset.get("note", ""),
            has_api_key=bool(preset.get("api_key")),
            user_created=bool(preset.get("user_created", False)),
        )
        for name, preset in sorted(presets.items())
    ]


@router.post("/planners", response_model=CUAPlannerInfo, status_code=201)
async def create_planner(request: CUAPlannerCreateRequest) -> CUAPlannerInfo:
    try:
        name, preset = save_user_preset(
            request.name,
            request.url,
            request.model,
            api_key=request.api_key or None,
            reasoning_effort=request.reasoning_effort,
            text_only=request.text_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CUAPlannerInfo(
        name=name,
        model=preset["model"],
        url=preset["url"],
        text_only=bool(preset.get("text_only", False)),
        note=preset.get("note", ""),
        has_api_key=bool(preset.get("api_key")),
        user_created=True,
    )


@router.delete("/planners/{name}")
async def delete_planner(name: str) -> dict:
    try:
        delete_user_preset(name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": name}


@router.post("/runs", response_model=CUARunCreated, status_code=202)
async def create_run(request: CUARunCreateRequest) -> CUARunCreated:
    try:
        run = await _service().create(
            app=request.app,
            goal=request.goal,
            planner=request.planner,
            planner_model=request.planner_model,
            planner_url=request.planner_url,
            open_url=request.open_url,
            allowed_domain=request.allowed_domain,
            max_steps=request.max_steps,
            human_login=request.human_login,
        )
    except (ValueError, cua_service.CUARunConflictError) as exc:
        raise _http_error(exc) from exc
    return CUARunCreated(run_id=run.run_id, status=run.status)


@router.get("/runs", response_model=CUARunList)
async def list_runs() -> CUARunList:
    return CUARunList(runs=_service().list_runs())


@router.get("/runs/{run_id}", response_model=CUARunView)
async def get_run(run_id: str, after: int = Query(default=0, ge=0)) -> CUARunView:
    try:
        return CUARunView(**_service().get(run_id).view(events_after=after))
    except cua_service.CUARunNotFoundError as exc:
        raise _http_error(exc) from exc


@router.get("/runs/{run_id}/events", response_model=CUARunView)
async def get_run_events(
    run_id: str, after: int = Query(default=0, ge=0)
) -> CUARunView:
    try:
        return CUARunView(**_service().get(run_id).view(events_after=after))
    except cua_service.CUARunNotFoundError as exc:
        raise _http_error(exc) from exc


@router.post("/runs/{run_id}/approval", response_model=CUAApprovalResult)
async def approve_run(run_id: str) -> CUAApprovalResult:
    try:
        approved = _service().get(run_id).approve()
    except cua_service.CUARunNotFoundError as exc:
        raise _http_error(exc) from exc
    if not approved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="run is not awaiting approval",
        )
    return CUAApprovalResult(run_id=run_id, approved=True)


@router.post("/runs/{run_id}/cancel", response_model=CUARunView)
async def cancel_run(run_id: str) -> CUARunView:
    try:
        run = _service().cancel(run_id)
        return CUARunView(**run.view())
    except cua_service.CUARunNotFoundError as exc:
        raise _http_error(exc) from exc
