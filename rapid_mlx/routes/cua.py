"""Server-side CUA surface: create/poll/approve/cancel computer-use runs.

Conventions follow ``routes/agents.py``: numbered events with ``after=SEQ``
polling (no SSE), a status machine, explicit approval endpoint, bearer auth +
rate limit on the whole router. The run itself is the productized
``rapid_mlx.cua`` loop; the computer (AX/CGEvent) is the shared resource, so
exactly one run may be active.
"""

from __future__ import annotations

import sys
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ..computer_use.errors import ComputerUseError
from ..config import get_config
from ..cua import service as cua_service
from ..cua.config import delete_user_preset, load_config, save_user_preset
from ..middleware.auth import check_rate_limit, security, verify_api_key


async def require_cua_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> bool:
    """Computer control requires an explicitly configured server bearer."""
    if not get_config().api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Computer use API requires a server API key",
        )
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    return bool(await verify_api_key(credentials))


router = APIRouter(
    prefix="/v1/cua",
    dependencies=[Depends(require_cua_auth), Depends(check_rate_limit)],
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


class CUAEvent(BaseModel):
    """Stable event envelope; event-specific fields remain forward compatible."""

    model_config = ConfigDict(extra="allow")

    kind: str
    seq: int
    ts: float
    app: str | None = None
    step: int | None = None
    action: str | None = None
    step_instruction: str | None = None
    element_index: int | None = None
    target_label: str | None = None
    target: str | None = None
    latency_s: float | None = None
    outcome: str | None = None
    tree_changed: bool | None = None
    url_after: str | None = None
    reason: str | None = None
    timeout_s: float | None = None
    approved: bool | None = None
    status: str | None = None
    final_summary: str | None = None
    error: str | None = None


class CUAPendingGate(BaseModel):
    gate_id: str
    kind: Literal["approval"] = "approval"
    reason: str
    requested_at: float | None = None
    expires_at: float | None = None


class CUARunView(BaseModel):
    run_id: str
    app: str
    goal: str
    status: str
    final_summary: str
    error: str
    planner: str
    events_after_seq: int
    events: list[CUAEvent]
    pending_gate: CUAPendingGate | None = None


class CUARunSummary(BaseModel):
    run_id: str
    app: str
    goal: str
    status: str
    created_at: float


class CUARunList(BaseModel):
    runs: list[CUARunSummary]


class CUAApprovalResult(BaseModel):
    run_id: str
    approved: bool


class CUAGateDecision(BaseModel):
    gate_id: str = Field(min_length=1, max_length=64)
    approved: bool = True


class CUACapabilityFeatures(BaseModel):
    app_discovery: bool = True
    window_discovery: bool = True
    window_selection: bool = False
    visual_observation: bool = False
    approval_gate_id: bool = True


class CUACapabilities(BaseModel):
    protocol_version: int = 1
    available: bool
    platform: str
    discovery: list[str]
    run_operations: list[str]
    max_concurrent_runs: int
    features: CUACapabilityFeatures


class CUAPermissions(BaseModel):
    accessibility: bool | None
    screen_recording: bool | None
    hints: list[str] = Field(default_factory=list)


class CUAApp(BaseModel):
    name: str | None
    bundle_id: str | None = Field(alias="bundleId")
    pid: int

    model_config = ConfigDict(populate_by_name=True)


class CUAWindow(BaseModel):
    window_id: str
    index: int
    title: str
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None


def _service() -> cua_service.CUAService:
    return cua_service.get_cua_service()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, cua_service.CUARunNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, cua_service.CUARunConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, cua_service.CUAGateMismatchError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, cua_service.CUAGateDecisionConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, ComputerUseError):
        code_to_status = {
            "app_not_found": status.HTTP_404_NOT_FOUND,
            "window_not_found": status.HTTP_404_NOT_FOUND,
            "unsupported_platform": status.HTTP_501_NOT_IMPLEMENTED,
            "permission_denied": status.HTTP_403_FORBIDDEN,
        }
        return HTTPException(
            status_code=code_to_status.get(exc.code, status.HTTP_400_BAD_REQUEST),
            detail={
                "code": exc.code,
                "message": exc.message,
                "recovery": list(exc.recovery),
            },
        )
    return HTTPException(status_code=500, detail=str(exc))


def _backend():
    # Keep server/router imports safe on Linux and on Macs without PyObjC.
    from ..computer_use import backend

    return backend


def _discovery_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ImportError):
        exc = ComputerUseError(
            "unsupported_platform", "computer use requires macOS with PyObjC"
        )
    return _http_error(exc)


@router.get("/capabilities", response_model=CUACapabilities)
async def get_capabilities() -> CUACapabilities:
    return CUACapabilities(
        available=sys.platform == "darwin",
        platform=sys.platform,
        discovery=["permissions", "apps", "windows"],
        run_operations=["create", "poll", "approve", "deny", "cancel"],
        max_concurrent_runs=cua_service.MAX_CONCURRENT_RUNS,
        features=CUACapabilityFeatures(),
    )


@router.get("/permissions", response_model=CUAPermissions)
async def get_permissions() -> CUAPermissions:
    if sys.platform != "darwin":
        return CUAPermissions(
            accessibility=None,
            screen_recording=None,
            hints=["Computer use requires a macOS host with PyObjC installed."],
        )
    try:
        payload = await run_in_threadpool(_backend().permissions)
        return CUAPermissions(**payload)
    except (ComputerUseError, ImportError) as exc:
        raise _discovery_error(exc) from exc


@router.get("/apps", response_model=list[CUAApp], response_model_by_alias=False)
async def list_apps() -> list[CUAApp]:
    try:
        apps = await run_in_threadpool(_backend().list_apps)
        return [CUAApp(**app) for app in apps]
    except (ComputerUseError, ImportError) as exc:
        raise _discovery_error(exc) from exc


@router.get("/apps/{app}/windows", response_model=list[CUAWindow])
async def list_windows(app: str) -> list[CUAWindow]:
    try:
        windows = await run_in_threadpool(_backend().list_windows, app)
        return [CUAWindow(**window) for window in windows]
    except (ComputerUseError, ImportError) as exc:
        raise _discovery_error(exc) from exc


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
async def approve_run(
    run_id: str, decision: CUAGateDecision | None = None
) -> CUAApprovalResult:
    try:
        requested = True if decision is None else decision.approved
        resolved = (
            _service()
            .get(run_id)
            .resolve_gate(
                requested, gate_id=None if decision is None else decision.gate_id
            )
        )
    except (
        cua_service.CUARunNotFoundError,
        cua_service.CUAGateMismatchError,
        cua_service.CUAGateDecisionConflictError,
    ) as exc:
        raise _http_error(exc) from exc
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="run is not awaiting approval",
        )
    return CUAApprovalResult(run_id=run_id, approved=requested)


@router.post("/runs/{run_id}/cancel", response_model=CUARunView)
async def cancel_run(run_id: str) -> CUARunView:
    try:
        run = _service().cancel(run_id)
        return CUARunView(**run.view())
    except cua_service.CUARunNotFoundError as exc:
        raise _http_error(exc) from exc
