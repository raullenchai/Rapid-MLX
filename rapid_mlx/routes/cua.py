"""Server-side CUA surface: create/poll/approve/cancel computer-use runs.

Conventions follow ``routes/agents.py``: numbered events with ``after=SEQ``
polling (no SSE), a status machine, explicit approval endpoint, bearer auth +
rate limit on the whole router. The run itself is the productized
``rapid_mlx.cua`` loop; the computer (AX/CGEvent) is the shared resource, so
exactly one run may be active.
"""

from __future__ import annotations

import base64
import ipaddress
import os
import sys
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    allow_remote: bool = False


class CUAPlannerCreateRequest(BaseModel):
    """User adds an OpenAI-compatible planner endpoint from app settings."""

    name: str = Field(min_length=1, max_length=32)
    url: str = Field(min_length=8, max_length=2000)
    model: str = Field(min_length=1, max_length=500)
    api_key: str | None = Field(default=None, max_length=2000)
    reasoning_effort: str | None = Field(default=None, max_length=20)
    text_only: bool = False
    allow_remote: bool = False


class CUATarget(BaseModel):
    target_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    app: str = Field(min_length=5, max_length=32, pattern=r"^pid:[1-9][0-9]*$")
    pid: int = Field(gt=0)
    window_id: str = Field(min_length=1, max_length=128)
    allowed_domain: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_pid_selector(self) -> CUATarget:
        if self.app != f"pid:{self.pid}":
            raise ValueError("target app must exactly match pid:<pid>")
        return self


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
    window_id: str | None = Field(default=None, min_length=1, max_length=128)
    client_request_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[^/]+$"
    )
    targets: list[CUATarget] | None = Field(default=None, min_length=2, max_length=3)
    initial_target_id: str | None = Field(default=None, min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_target_set(self) -> CUARunCreateRequest:
        if self.targets is None:
            if self.initial_target_id is not None:
                raise ValueError("initial_target_id requires targets")
            return self
        if self.window_id is not None or self.allowed_domain or self.open_url:
            raise ValueError(
                "targets cannot be combined with window_id, allowed_domain, or open_url"
            )
        ids = [target.target_id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("target_id values must be unique")
        anchors = [(target.pid, target.window_id) for target in self.targets]
        if len(anchors) != len(set(anchors)):
            raise ValueError("each target must bind a distinct process window")
        if self.initial_target_id not in set(ids):
            raise ValueError("initial_target_id must name one submitted target")
        initial = next(t for t in self.targets if t.target_id == self.initial_target_id)
        if self.app != initial.app:
            raise ValueError("app must match the initial target app")
        return self


class CUARunCreated(BaseModel):
    run_id: str
    status: str
    window_id: str | None = None
    client_request_id: str | None = None
    targets: list[CUATarget] = Field(default_factory=list)
    active_target_id: str | None = None


class CUAEvent(BaseModel):
    """Stable event envelope; event-specific fields remain forward compatible."""

    model_config = ConfigDict(extra="allow")

    kind: str
    seq: int
    ts: float
    gate_id: str | None = None
    target_id: str | None = None
    from_target_id: str | None = None
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
    recovery: list[str] = Field(default_factory=list)


class CUAPendingGate(BaseModel):
    gate_id: str
    kind: Literal["approval"] = "approval"
    reason: str
    requested_at: float | None = None
    expires_at: float | None = None
    action: str | None = None
    target: str | None = None
    target_id: str | None = None


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
    window_id: str | None = None
    targets: list[CUATarget] = Field(default_factory=list)
    active_target_id: str | None = None


class CUARunSummary(BaseModel):
    run_id: str
    app: str
    goal: str
    status: str
    created_at: float
    window_id: str | None = None
    active_target_id: str | None = None


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
    window_selection: bool = True
    visual_observation: bool = False
    screenshot_observation: bool = False
    observation_without_activation: bool = False
    approval_gate_id: bool = True
    idempotent_run_create: bool = True
    multi_target_runs: bool = True
    switch_target: bool = True
    permission_request: bool = False


class CUACapabilities(BaseModel):
    protocol_version: int = 2
    available: bool
    platform: str
    discovery: list[str]
    run_operations: list[str]
    max_concurrent_runs: int
    features: CUACapabilityFeatures
    max_run_targets: int = 3


class CUAPermissions(BaseModel):
    accessibility: bool | None
    screen_recording: bool | None
    hints: list[str] = Field(default_factory=list)


class CUAPermissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permission: Literal["accessibility", "screen_recording"]


class CUAPermissionRequestResult(BaseModel):
    permission: Literal["accessibility", "screen_recording"]
    granted: bool
    permissions: CUAPermissions


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


class CUAObservationRequest(BaseModel):
    """Request a fresh observation of one PID-bound, opaque window ID."""

    model_config = ConfigDict(extra="forbid")

    app: str = Field(min_length=1, max_length=120)
    pid: int = Field(gt=0)
    window_id: str = Field(min_length=1, max_length=64)
    screenshot: bool = False


class CUAObservationApp(BaseModel):
    name: str | None
    bundle_id: str | None = Field(alias="bundleId")
    pid: int

    model_config = ConfigDict(populate_by_name=True)


class CUAObservationElement(BaseModel):
    index: int
    role: str
    subrole: str = ""
    label: str
    actions: list[str] = Field(default_factory=list)
    x: int
    y: int
    width: int
    height: int
    center: list[int]


class CUAObservationImage(BaseModel):
    media_type: Literal["image/png"] = "image/png"
    encoding: Literal["base64"] = "base64"
    data: str
    byte_count: int


class CUAObservation(BaseModel):
    snapshot_id: str
    observed_at: float
    app: CUAObservationApp
    window_id: str
    window_index: int
    window: CUAWindow
    coordinate_space: Literal["screen"]
    elements: list[CUAObservationElement]
    element_count: int
    truncated: bool
    screenshot: CUAObservationImage | None = None


MAX_OBSERVATION_PNG_BYTES = 4 * 1024 * 1024


def _screenshots_enabled() -> bool:
    """Screenshots are sensitive and require an explicit server opt-in."""
    return os.getenv("RAPID_MLX_CUA_EXPOSE_SCREENSHOTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _service() -> cua_service.CUAService:
    return cua_service.get_cua_service()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, cua_service.CUARunNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, cua_service.CUARunConflictError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, cua_service.CUARequestIdentityConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "request_identity_conflict",
                "message": str(exc),
                "recovery": [],
            },
        )
    if isinstance(exc, cua_service.CUARequestIdentityNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "request_identity_not_found",
                "message": str(exc),
                "recovery": [],
            },
        )
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
            "permission_request_failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "app_mismatch": status.HTTP_409_CONFLICT,
            "window_stale": status.HTTP_409_CONFLICT,
            "target_drift": status.HTTP_409_CONFLICT,
            "screenshot_too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
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


def _observation_error(exc: Exception) -> HTTPException:
    error = _discovery_error(exc)
    error.headers = {"Cache-Control": "no-store", "Pragma": "no-cache"}
    return error


def _public_observation_element(item: dict) -> CUAObservationElement:
    """Remove secret field contents before an AX element crosses HTTP."""
    public = dict(item)
    if (
        public.get("role") == "AXSecureTextField"
        or public.get("subrole") == "AXSecureTextField"
    ):
        public["label"] = "[secure text redacted]"
    return CUAObservationElement(**public)


@router.get("/capabilities", response_model=CUACapabilities)
async def get_capabilities() -> CUACapabilities:
    native_observation = sys.platform == "darwin"
    accessibility_ready = False
    screen_recording_ready = False
    if native_observation:
        try:
            permission_state = await run_in_threadpool(_backend().permissions)
            accessibility_ready = permission_state.get("accessibility") is True
            screen_recording_ready = permission_state.get("screen_recording") is True
        except (ComputerUseError, ImportError):
            pass
    return CUACapabilities(
        available=native_observation,
        platform=sys.platform,
        discovery=["permissions", "apps", "windows"],
        run_operations=["create", "poll", "approve", "deny", "cancel"],
        max_concurrent_runs=cua_service.MAX_CONCURRENT_RUNS,
        features=CUACapabilityFeatures(
            visual_observation=native_observation and accessibility_ready,
            screenshot_observation=(
                native_observation
                and accessibility_ready
                and screen_recording_ready
                and _screenshots_enabled()
            ),
            observation_without_activation=native_observation and accessibility_ready,
            permission_request=(
                native_observation and get_config().cua_permission_requests_enabled
            ),
        ),
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


@router.post("/permissions/request", response_model=CUAPermissionRequestResult)
async def request_permission(
    request: CUAPermissionRequest,
    http_request: Request,
) -> CUAPermissionRequestResult:
    """Request one macOS grant only after an explicit authenticated POST."""
    if not get_config().cua_permission_requests_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "local_request_required",
                "message": "permission prompts require a direct loopback CUA server",
                "recovery": [],
            },
        )
    client_host = http_request.client.host if http_request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "local_request_required",
                "message": "permission prompts require a loopback client",
                "recovery": [],
            },
        )
    if sys.platform != "darwin":
        raise _discovery_error(
            ComputerUseError(
                "unsupported_platform", "computer use permissions require macOS"
            )
        )
    try:
        payload = await run_in_threadpool(
            _backend().request_permission, request.permission
        )
        return CUAPermissionRequestResult(**payload)
    except (ComputerUseError, ImportError, ValueError) as exc:
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


@router.post("/observations", response_model=CUAObservation)
async def create_observation(
    request: CUAObservationRequest, response: Response
) -> CUAObservation:
    """Observe an exact app window without activating it or consulting cache."""
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    if sys.platform != "darwin":
        raise _observation_error(
            ComputerUseError(
                "unsupported_platform", "visual observation requires macOS with PyObjC"
            )
        )
    if request.screenshot and not _screenshots_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "screenshot_disabled",
                "message": "screenshot exposure is disabled by server policy",
                "recovery": [],
            },
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )

    try:
        permission_state = await run_in_threadpool(_backend().permissions)
        if permission_state.get("accessibility") is not True:
            raise ComputerUseError(
                "permission_denied",
                "Accessibility permission is required to observe UI elements",
            )
        if request.screenshot and permission_state.get("screen_recording") is not True:
            raise ComputerUseError(
                "permission_denied",
                "Screen Recording permission is required for screenshots",
            )

        snapshot = await run_in_threadpool(
            _backend().get_app_state,
            f"pid:{request.pid}",
            screenshot=request.screenshot,
            use_cache=False,
            window_id=request.window_id,
            activate=False,
        )
        app_info = snapshot.get("app") or {}
        canonical_names = {
            str(app_info.get("name") or "").casefold(),
            str(app_info.get("bundleId") or "").casefold(),
        }
        if (
            app_info.get("pid") != request.pid
            or request.app.casefold() not in canonical_names
        ):
            raise ComputerUseError(
                "app_mismatch",
                "the requested app identity does not match the target process",
            )
        if snapshot.get("window_id") != request.window_id:
            raise ComputerUseError(
                "window_stale",
                "the observed window does not match the requested window",
            )

        png = snapshot.get("screenshot_png")
        image = None
        if request.screenshot:
            if not isinstance(png, bytes):
                raise ComputerUseError(
                    "screenshot_failed",
                    "the selected window produced no PNG screenshot",
                )
            if len(png) > MAX_OBSERVATION_PNG_BYTES:
                raise ComputerUseError(
                    "screenshot_too_large",
                    f"PNG exceeds the {MAX_OBSERVATION_PNG_BYTES}-byte observation limit",
                )
            image = CUAObservationImage(
                data=base64.b64encode(png).decode("ascii"), byte_count=len(png)
            )

        return CUAObservation(
            snapshot_id=snapshot["snapshot_id"],
            observed_at=snapshot["observed_at"],
            app=CUAObservationApp(**app_info),
            window_id=snapshot["window_id"],
            window_index=snapshot["window_index"],
            window=CUAWindow(**snapshot["window"]),
            coordinate_space=snapshot["coordinate_space"],
            elements=[
                _public_observation_element(item) for item in snapshot["elements"]
            ],
            element_count=snapshot["element_count"],
            truncated=snapshot["truncated"],
            screenshot=image,
        )
    except (ComputerUseError, ImportError) as exc:
        raise _observation_error(exc) from exc


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
            allow_remote=bool(preset.get("allow_remote", bool(preset.get("api_key")))),
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
            allow_remote=request.allow_remote,
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
        allow_remote=bool(preset.get("allow_remote", False)),
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
            window_id=request.window_id,
            client_request_id=request.client_request_id,
            targets=(
                [target.model_dump() for target in request.targets]
                if request.targets
                else None
            ),
            initial_target_id=request.initial_target_id,
        )
    except (
        ValueError,
        cua_service.CUARunConflictError,
        cua_service.CUARequestIdentityConflictError,
        ComputerUseError,
    ) as exc:
        raise _http_error(exc) from exc
    return CUARunCreated(
        run_id=run.run_id,
        status=run.status,
        window_id=run.window_id,
        client_request_id=run.client_request_id,
        targets=[CUATarget.model_validate(target) for target in run.targets],
        active_target_id=run.active_target_id,
    )


@router.get("/runs", response_model=CUARunList)
async def list_runs() -> CUARunList:
    return CUARunList(
        runs=[CUARunSummary.model_validate(run) for run in _service().list_runs()]
    )


@router.get("/runs/by-request/{client_request_id}", response_model=CUARunCreated)
async def get_run_by_request(
    client_request_id: str = Path(min_length=1, max_length=128, pattern=r"^[^/]+$"),
) -> CUARunCreated:
    try:
        run = _service().get_by_request_id(client_request_id)
    except cua_service.CUARequestIdentityNotFoundError as exc:
        raise _http_error(exc) from exc
    return CUARunCreated(
        run_id=run.run_id,
        status=run.status,
        window_id=run.window_id,
        client_request_id=run.client_request_id,
        targets=[CUATarget.model_validate(target) for target in run.targets],
        active_target_id=run.active_target_id,
    )


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
async def approve_run(run_id: str, decision: CUAGateDecision) -> CUAApprovalResult:
    try:
        requested = decision.approved
        resolved = (
            _service().get(run_id).resolve_gate(requested, gate_id=decision.gate_id)
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
