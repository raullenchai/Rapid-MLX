"""Native macOS backend for the model-agnostic computer-use tool suite.

Wraps the AX probe (tools/gui_verifier_cua_poc/ax_driver.py primitives) into
a snapshot-cache-first surface: caches keyed by app/window with a TTL,
element actions that accept an element index (semantic) or coordinates
(fallback), direct AX value writes with read-back verification before
falling back to synthetic typing, and typed errors with recovery hints.

The layer is deliberately model-free: any agent (cloud, local GLM, local
Qwen) drives the machine through these calls.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse

from .errors import ComputerUseError

_permission_request_lock = threading.Lock()
_finder_rename_binding_lock = threading.Lock()
_finder_rename_bindings: dict[tuple[int, str], tuple[object, object, str, float]] = {}

SNAPSHOT_TTL_S = 120.0
AX_COLLECT_TIMEOUT_S = 20.0
FILL_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
SNAPSHOT_CACHE_MAX = 32
MAX_TRANSIENT_TARGETS = 64
MOUSE_BUTTONS = ("left", "right", "middle")
from . import ax_driver, background_input

KEY_ALIASES: dict[str, int] = {
    "return": 36,
    "enter": 36,
    "escape": 53,
    "esc": 53,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "forwarddelete": 117,
    "arrowdown": 125,
    "down": 125,
    "arrowup": 126,
    "up": 126,
    "arrowleft": 123,
    "left": 123,
    "arrowright": 124,
    "right": 124,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
}
LETTER_RE = re.compile(r"^[a-z0-9=]$")
# kCGEventFlagMask exact values
MODIFIER_FLAGS = {
    "cmd": 1 << 20,
    "command": 1 << 20,
    "ctrl": 1 << 18,
    "control": 1 << 18,
    "alt": 1 << 19,
    "option": 1 << 19,
    "opt": 1 << 19,
    "shift": 1 << 17,
    "fn": 1 << 23,
}


_ELECTRON_BY_PID: dict[tuple[int, object], bool] = {}


def _is_electron(app_info: dict) -> bool:
    """Whether the app embeds Electron (Slack, VS Code, Notion, ...).

    Electron apps carry Chromium's web-content AX and input behavior under
    their own bundle ids, so they are recognized by the framework they ship.
    """
    pid = app_info.get("pid")
    if not isinstance(pid, int):
        return False
    key = (pid, app_info.get("processStartTime"))
    if key not in _ELECTRON_BY_PID:
        found = False
        try:
            running = ax_driver._application_for_pid(pid)
            url = running.bundleURL() if running is not None else None
            if url is not None:
                found = (
                    Path(str(url.path()))
                    / "Contents/Frameworks/Electron Framework.framework"
                ).exists()
        except Exception:  # noqa: BLE001 - unreadable now; ask again next time
            return False
        _ELECTRON_BY_PID[key] = found
    return _ELECTRON_BY_PID[key]


# Chromium builds its accessibility tree right after AXManualAccessibility is
# switched on, but drops AX actions (press, focus) for ~1.5 s more while the
# action routing comes up (measured on Electron: the first AXPress within
# 1 s is a silent no-op that still returns success).
_CHROMIUM_AX_ACTION_READY_S = 2.0


def _await_ax_actions_ready(app_info: dict) -> None:
    if not _needs_web_content_retry(app_info):
        return
    age = ax_driver.exposure_age(int(app_info["pid"]))
    if age is not None and age < _CHROMIUM_AX_ACTION_READY_S:
        time.sleep(_CHROMIUM_AX_ACTION_READY_S - age)


def _needs_web_content_retry(app_info: dict) -> bool:
    """Only Chromium-family AX trees need the lazy web-content retry loop."""

    bundle = str(app_info.get("bundleId") or app_info.get("bundle_id") or "").lower()
    return any(
        marker in bundle for marker in ("chrome", "chromium", "edge")
    ) or _is_electron(app_info)


def _resolved_app_info(running: Any) -> dict:
    """Return app identity, including a private process-incarnation marker."""
    info = {
        "name": (running.localizedName() or "").lower(),
        "bundleId": (running.bundleIdentifier() or "").lower(),
        "pid": int(running.processIdentifier()),
    }
    try:
        launched = running.launchDate()
        started_at = float(launched.timeIntervalSince1970())
    except Exception:  # noqa: BLE001 - Finder/older bridges may omit launchDate
        try:
            import psutil

            started_at = float(psutil.Process(info["pid"]).create_time())
        except Exception:  # noqa: BLE001 - unavailable identity fails closed upstream
            return info
    if started_at > 0:
        # Internal only: HTTP response models intentionally drop this field.
        info["processStartTime"] = started_at
    return info


def _resolve_app(app: str, *, activate: bool = True) -> tuple[object, dict]:
    """Find a running app by name substring, bundle id, or pid:N."""
    try:
        import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817
    except ImportError as exc:
        raise ComputerUseError(
            "unsupported_platform", "computer-use actions require macOS with PyObjC"
        ) from exc

    wanted_pid = None
    if app.startswith("pid:"):
        try:
            wanted_pid = int(app.split(":", 1)[1])
        except ValueError as exc:
            raise ComputerUseError(
                "invalid_argument", f"bad pid spec: {app!r}"
            ) from exc
    lowered = app.lower()
    if wanted_pid is not None:
        running = ax_driver._application_for_pid(wanted_pid, AS)
        if running is not None:
            return _ax_app_element(running, activate=activate), _resolved_app_info(
                running
            )
        raise ComputerUseError("app_not_found", f"no running app matches {app!r}")
    for running in ax_driver._running_applications(AS):
        if (
            not running.isActive()
            and wanted_pid is None
            and running.activationPolicy() != 0
        ):
            continue
        info = _resolved_app_info(running)
        name = str(info["name"])
        bundle = str(info["bundleId"])
        if lowered in (name, bundle) or lowered in name or lowered in bundle:
            return _ax_app_element(running, activate=activate), info
    raise ComputerUseError("app_not_found", f"no running app matches {app!r}")


def _ax_app_element(running_app: Any, activate: bool = True) -> object:
    from ApplicationServices import (
        AXUIElementCreateApplication,
        AXUIElementSetAttributeValue,
    )

    element = AXUIElementCreateApplication(int(running_app.processIdentifier()))
    first = ax_driver.first_exposure(running_app)
    if first:
        AXUIElementSetAttributeValue(element, "AXManualAccessibility", True)
        AXUIElementSetAttributeValue(element, "AXEnhancedUserInterface", True)
    if activate:
        # Synthetic events need the window frontmost; the first click otherwise
        # only raises the window and keystrokes land on the wrong focus.
        try:
            running_app.activateWithOptions_(
                1 << 1
            )  # NSApplicationActivateIgnoringOtherApps
        except Exception:  # noqa: BLE001 - older pyobjc / activation best-effort
            try:
                running_app.unhide()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.6)
    if first:
        time.sleep(0.3)
    return element


class SnapshotCache:
    """TTL cache of AX snapshots keyed by (app spec, window index)."""

    def __init__(self) -> None:
        self._entries: dict[
            tuple[str, int, int | str | None, bool], tuple[float, dict]
        ] = {}

    def get(
        self,
        app: str,
        window_index: int,
        screenshot: bool = False,
        window_id: int | str | None = None,
    ) -> dict | None:
        key = (app, window_index, window_id, screenshot)
        entry = self._entries.get(key)
        if entry is None:
            return None
        created, snapshot = entry
        if time.time() - created > SNAPSHOT_TTL_S:
            self._entries.pop(key, None)
            return None
        return snapshot

    def put(
        self,
        app: str,
        window_index: int,
        snapshot: dict,
        screenshot: bool = False,
        window_id: int | str | None = None,
    ) -> None:
        self._entries[(app, window_index, window_id, screenshot)] = (
            time.time(),
            snapshot,
        )
        if len(self._entries) > SNAPSHOT_CACHE_MAX:
            oldest = min(self._entries, key=lambda key: self._entries[key][0])
            self._entries.pop(oldest, None)

    def clear(self) -> None:
        self._entries.clear()


_CACHE = SnapshotCache()


def _window_records(app_info: dict) -> list[dict]:
    """Return this process's visible layer-zero windows in CG front-to-back order."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    raw = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for window in raw or []:
        if int(window.get("kCGWindowOwnerPID", -1)) != int(app_info["pid"]):
            continue
        if int(window.get("kCGWindowLayer", 99)) != 0:
            continue
        number = window.get("kCGWindowNumber")
        bounds = window.get("kCGWindowBounds") or {}
        if number is None:
            continue
        seen.add(int(number))
        records.append(
            {
                "index": len(records),
                "window_id": f"cg:{int(number)}",
                "title": window.get("kCGWindowName") or "",
                "x": bounds.get("X"),
                "y": bounds.get("Y"),
                "width": bounds.get("Width"),
                "height": bounds.get("Height"),
            }
        )
    if any(not record["title"] for record in records):
        # Without Screen Recording, CG hides window titles; AX still has them.
        titles = _ax_window_titles(int(app_info["pid"]))
        for record in records:
            if not record["title"]:
                record["title"] = titles.get(int(record["window_id"][3:]), "")
    # Windows on other Spaces (e.g. behind the user's full-screen app) are
    # not "on screen" but still take AX + SkyLight input. Keep only CG windows
    # that are real AX windows of this app (same CGWindowID), so hidden helper
    # surfaces never become targets.
    for window in _offscreen_ax_windows(app_info, seen):
        window["index"] = len(records)
        records.append(window)
    return records


def _cg_window_names_visible() -> bool:
    """Whether CG reports window titles (it hides them without Screen Recording)."""
    try:
        import Quartz

        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:  # noqa: BLE001 - older macOS without the API
        return True


def _ax_window_titles(pid: int) -> dict[int, str]:
    """CG window id -> AX title for ``pid``'s windows on the current Space
    (best-effort: empty when Accessibility cannot be read)."""
    if ax_driver.AS is None:
        return {}
    titles: dict[int, str] = {}
    try:
        app_element = ax_driver.AXUIElementCreateApplication(pid)
        for window in ax_driver._as_list(ax_driver._get(app_element, "AXWindows")):
            window_id = background_input.ax_window_id(window)
            title = ax_driver._get(window, "AXTitle")
            if window_id is not None and isinstance(title, str):
                titles[int(window_id)] = title
    except Exception:  # noqa: BLE001 - titles must never break listing
        return {}
    return titles


# CG window ids already searched for by remote token, per pid, tagged with the
# process start time so a recycled pid is searched (and its cached element
# ids dropped) afresh.
_REMOTE_SCANNED: dict[int, tuple[object, set[int]]] = {}


def _offscreen_ax_windows(app_info: dict, seen: set[int]) -> list[dict]:
    """Layer-0 CG windows of ``app_info`` missing from the on-screen list
    that map to one of its AX windows by CGWindowID (best-effort)."""
    if ax_driver.AS is None:
        return []
    try:
        return _offscreen_ax_windows_unchecked(app_info, seen)
    except Exception:  # noqa: BLE001 - discovery must never break listing
        return []


def _offscreen_ax_windows_unchecked(app_info: dict, seen: set[int]) -> list[dict]:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListOptionAll,
    )

    pid = int(app_info["pid"])
    # Validate the process incarnation before any AX read: remote element ids
    # cached for a recycled pid would resolve windows of another process.
    incarnation = app_info.get("processStartTime")
    scanned = _REMOTE_SCANNED.get(pid)
    if scanned is None or scanned[0] != incarnation:
        ax_driver._REMOTE_IDS.pop(pid, None)
        scanned = _REMOTE_SCANNED[pid] = (incarnation, set())
    tried = scanned[1]
    app_element = ax_driver.AXUIElementCreateApplication(pid)

    def ax_titles() -> dict[int, str]:
        frames = {}
        for ax_window in ax_driver._app_windows(app_element):
            if ax_driver._get(ax_window, "AXMinimized"):
                continue
            wid = background_input.ax_window_id(ax_window)
            if wid:
                frames[wid] = str(ax_driver._get(ax_window, "AXTitle") or "")
        return frames

    titles = ax_titles()
    cg_windows = (
        CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID) or []
    )
    names_visible = _cg_window_names_visible()

    def candidate(window: dict) -> bool:
        # Untitled layer-0 surfaces are mostly helpers (Chromium has several);
        # scanning for each would cost the full budget. An untitled off-Space
        # window is still found via AXFocusedWindow/AXMainWindow. Without
        # Screen Recording CG hides every title, so a plausible size stands
        # in for it (helper surfaces are 1x1 or thin strips).
        if names_visible:
            return bool(window.get("kCGWindowName"))
        bounds = window.get("kCGWindowBounds") or {}
        return (
            float(bounds.get("Width", 0)) >= 200
            and float(bounds.get("Height", 0)) >= 150
        )

    unmapped = {
        int(window["kCGWindowNumber"])
        for window in cg_windows
        if int(window.get("kCGWindowOwnerPID", -1)) == pid
        and int(window.get("kCGWindowLayer", 99)) == 0
        and candidate(window)
        and int(window["kCGWindowNumber"]) not in titles
        and int(window["kCGWindowNumber"]) not in seen
    }
    if unmapped - tried:
        # Windows on another Space are invisible to AXWindows; resolve them
        # by remote token (bounded scan, once per unseen window).
        tried |= unmapped
        ax_driver.discover_remote_windows(pid, unmapped)
        titles = ax_titles()
    out = []
    for window in cg_windows:
        number = window.get("kCGWindowNumber")
        if (
            number is None
            or int(number) in seen
            or int(number) not in titles
            or int(window.get("kCGWindowOwnerPID", -1)) != int(app_info["pid"])
            or int(window.get("kCGWindowLayer", 99)) != 0
        ):
            continue
        bounds = window.get("kCGWindowBounds") or {}
        out.append(
            {
                "window_id": f"cg:{int(number)}",
                "title": window.get("kCGWindowName") or titles[int(number)],
                "x": bounds.get("X"),
                "y": bounds.get("Y"),
                "width": bounds.get("Width"),
                "height": bounds.get("Height"),
                "offscreen": True,
            }
        )
    return out


def _select_window(
    app_info: dict, *, window_index: int = 0, window_id: int | str | None = None
) -> dict:
    windows = _window_records(app_info)
    if window_id is not None:
        wanted = _cg_window_id(window_id)
        for window in windows:
            if _cg_window_id(window["window_id"]) == wanted:
                return window
        raise ComputerUseError(
            "window_not_found",
            f"window id {window_id} is not an on-screen window owned by pid "
            f"{app_info['pid']}",
        )
    if window_index < 0 or window_index >= len(windows):
        raise ComputerUseError(
            "window_not_found", f"window index {window_index} is not available"
        )
    return windows[window_index]


def _cg_window_id(window_id: int | str) -> int:
    raw = (
        window_id[3:]
        if isinstance(window_id, str) and window_id.startswith("cg:")
        else window_id
    )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ComputerUseError(
            "invalid_argument", f"invalid CG window id {window_id!r}"
        ) from exc
    if value <= 0:
        raise ComputerUseError(
            "invalid_argument", f"invalid CG window id {window_id!r}"
        )
    return value


def _same_window(lhs: dict, rhs: dict) -> bool:
    return _cg_window_id(lhs["window_id"]) == _cg_window_id(rhs["window_id"]) and all(
        lhs.get(key) == rhs.get(key) for key in ("x", "y", "width", "height")
    )


def _pid_app_element(app_info: dict) -> object:
    try:
        return ax_driver._app_element(
            app_info["name"], expected_pid=int(app_info["pid"])
        )
    except ax_driver.AppNotFoundError as exc:
        raise ComputerUseError("target_drift", "selected app exited") from exc


def _focused_ax_window(app_info: dict) -> object | None:
    app_element = _pid_app_element(app_info)
    app_windows = ax_driver._app_windows(app_element)
    focused = ax_driver._get(app_element, "AXFocusedWindow")
    if focused is None:
        focused = ax_driver._get(app_element, "AXFocusedUIElement")
        if (
            focused is not None
            and ax_driver._get(focused, "AXRole") == "AXWindow"
            and any(focused == window for window in app_windows)
        ):
            return focused
        seen: set[int] = set()
        for _ in range(12):
            if focused is None or id(focused) in seen:
                focused = None
                break
            seen.add(id(focused))
            if ax_driver._get(focused, "AXRole") == "AXWindow":
                break
            focused = ax_driver._get(focused, "AXParent")
        else:
            focused = None
    if (
        focused is None
        or ax_driver._get(focused, "AXRole") != "AXWindow"
        or not any(focused == window for window in app_windows)
    ):
        return None
    return focused


def _focused_ax_element(app_info: dict) -> object | None:
    """Return the exact focused control for the PID-bound app."""

    app_element = _pid_app_element(app_info)
    return ax_driver._get(app_element, "AXFocusedUIElement")


def _ax_cg_frames_match(
    ax_frame: tuple[float, float, float, float],
    cg_frame: tuple[float, float, float, float],
) -> bool:
    ax_x, ax_y, ax_w, ax_h = ax_frame
    cg_x, cg_y, cg_w, cg_h = cg_frame
    return all(
        abs(actual - expected) <= 4.0
        for actual, expected in (
            (ax_x, cg_x),
            (ax_y, cg_y),
            (ax_x + ax_w, cg_x + cg_w),
            (ax_y + ax_h, cg_y + cg_h),
        )
    )


def _focused_transient_window(
    app_info: dict,
    anchor: dict,
    *,
    trusted_window_id: int | str | None = None,
    baseline_window_ids: set[str] | None = None,
) -> dict | None:
    """Return one tightly bounded same-process focused companion window.

    The selected window remains the authorization anchor. A companion is only
    admitted when it contains the app's exact focused AX element, maps
    one-to-one to a same-PID layer-zero CG window, sits in front of and inside
    the anchor, and is either new or already trusted by ID.
    """
    focused = _focused_ax_window(app_info)
    if focused is None and app_info.get("bundleId") == "com.apple.finder":
        focused = _focused_ax_element(app_info)
    frame = ax_driver._point_size(focused) if focused is not None else None
    if frame is None:
        return None
    anchor_frame = (
        float(anchor["x"]),
        float(anchor["y"]),
        float(anchor["width"]),
        float(anchor["height"]),
    )
    if _ax_cg_frames_match(frame, anchor_frame):
        return None
    matches = []
    for candidate in _window_records(app_info):
        candidate_frame = (
            float(candidate["x"]),
            float(candidate["y"]),
            float(candidate["width"]),
            float(candidate["height"]),
        )
        if _ax_cg_frames_match(frame, candidate_frame):
            matches.append(candidate)
    if len(matches) != 1:
        return None
    candidate = matches[0]
    candidate_id = str(candidate["window_id"])
    if trusted_window_id is not None:
        if _cg_window_id(candidate_id) != _cg_window_id(trusted_window_id):
            return None
    elif baseline_window_ids is None or candidate_id in baseline_window_ids:
        return None
    if int(candidate["index"]) >= int(anchor["index"]):
        return None
    ax, ay, aw, ah = anchor_frame
    cx, cy, cw, ch = frame
    tolerance = 2.0
    if (
        cw <= 0
        or ch <= 0
        or aw <= 0
        or ah <= 0
        or cx < ax - tolerance
        or cy < ay - tolerance
        or cx + cw > ax + aw + tolerance
        or cy + ch > ay + ah + tolerance
        or cw * ch > aw * ah * 0.25
    ):
        return None
    return candidate


def _validate_snapshot_window(
    snapshot: dict,
    *,
    point: tuple[float, float] | None = None,
    require_topmost: bool = True,
) -> dict:
    """Fail closed when an observation is old or its exact CGWindow drifted.

    ``require_topmost`` guards input that is hit-tested by the WindowServer
    (global HID events land on whatever window is on top). Input routed to the
    exact pid/window does not care about occlusion, so background delivery
    keeps the in-bounds check but skips the topmost check.
    """
    observed_at = snapshot.get("observed_at")
    if (
        not isinstance(observed_at, (int, float))
        or time.time() - observed_at > SNAPSHOT_TTL_S
    ):
        raise ComputerUseError(
            "stale_observation",
            f"snapshot {snapshot.get('snapshot_id')} is stale; re-observe before acting",
        )
    expected = snapshot.get("window")
    app_info = snapshot.get("app") or {}
    if (
        not isinstance(expected, dict)
        or "window_id" not in expected
        or "pid" not in app_info
    ):
        raise ComputerUseError(
            "stale_observation", "snapshot has no stable window identity; re-observe"
        )
    current = _select_window(app_info, window_id=expected["window_id"])
    if not _same_window(expected, current):
        raise ComputerUseError(
            "target_drift",
            f"window {expected['window_id']} moved or resized since snapshot "
            f"{snapshot.get('snapshot_id')}; re-observe before acting",
        )
    if point is not None:
        x, y = point
        left, top = current.get("x"), current.get("y")
        width, height = current.get("width"), current.get("height")
        if (
            not isinstance(left, (int, float))
            or not isinstance(top, (int, float))
            or not isinstance(width, (int, float))
            or not isinstance(height, (int, float))
        ):
            raise ComputerUseError("target_drift", "selected window has invalid bounds")
        if not (left <= x < left + width and top <= y < top + height):
            raise ComputerUseError(
                "target_drift",
                f"point ({x}, {y}) is outside selected window {expected['window_id']}",
            )
        if require_topmost and _topmost_window_id_at(x, y) != _cg_window_id(
            expected["window_id"]
        ):
            raise ComputerUseError(
                "target_occluded",
                f"selected window {expected['window_id']} is not topmost at ({x}, {y})",
            )
    return current


def _topmost_window_id_at(x: float, y: float) -> int | None:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    for window in windows or []:
        # Normal application windows and their blocking sheets/panels are on
        # the normal window layer. Higher system layers include full-screen,
        # visually transparent surfaces owned by services such as the Dock and
        # Notification Center; treating those bounds as opaque makes every
        # underlying app target look occluded.
        if window.get("kCGWindowLayer") != 0:
            continue
        if float(window.get("kCGWindowAlpha", 1)) <= 0:
            continue
        bounds = window.get("kCGWindowBounds") or {}
        left, top = bounds.get("X"), bounds.get("Y")
        width, height = bounds.get("Width"), bounds.get("Height")
        number = window.get("kCGWindowNumber")
        if (
            number is None
            or not isinstance(left, (int, float))
            or not isinstance(top, (int, float))
            or not isinstance(width, (int, float))
            or not isinstance(height, (int, float))
        ):
            continue
        if left <= x < left + width and top <= y < top + height:
            return int(number)
    return None


# ``get_app_state(activate=...)`` value for internal re-observations: activate
# only when the delivery route needs it (see :func:`observation_activates`).
OBSERVE_BY_ROUTE: Literal["route"] = "route"


# Live AX refs of recent snapshots, by snapshot id, for callers that track
# element identity across observations (perception). Not serialized.
_LIVE_ELEMENTS: dict[str, list[object]] = {}
_LIVE_ELEMENTS_KEEP = 16


def _remember_live_elements(snapshot_id: str, elements: list[object]) -> None:
    _LIVE_ELEMENTS[snapshot_id] = elements
    while len(_LIVE_ELEMENTS) > _LIVE_ELEMENTS_KEEP:
        _LIVE_ELEMENTS.pop(next(iter(_LIVE_ELEMENTS)))


def live_elements(snapshot: dict) -> list[object] | None:
    return _LIVE_ELEMENTS.get(str(snapshot.get("snapshot_id")))


def get_app_state(
    app: str,
    window_index: int = 0,
    screenshot: bool = True,
    use_cache: bool = True,
    window_id: int | str | None = None,
    *,
    activate: bool | Literal["route"] = True,
    trusted_transient_window_id: int | str | None = None,
    transient_baseline_window_ids: set[str] | None = None,
) -> dict:
    """Snapshot one window: elements with indexes, tree text, optional PNG."""
    # Read-only callers can explicitly forbid activation; this is a security
    # boundary because an observation request must not steal focus or expose a
    # different window. The public default still activates.
    # ``activate=OBSERVE_BY_ROUTE`` lets the delivery route decide; internal
    # re-observations (no caller snapshot, post-action state) pass it so that
    # background delivery never hands the user's keyboard to the target.
    if activate == OBSERVE_BY_ROUTE:
        ax_element, app_info = _resolve_app(app, activate=False)
        if observation_activates(app_info):
            ax_element, app_info = _resolve_app(app)
    elif activate:
        ax_element, app_info = _resolve_app(app)
    else:
        ax_element, app_info = _resolve_app(app, activate=False)
    window = _select_window(app_info, window_index=window_index, window_id=window_id)
    if (
        use_cache
        and trusted_transient_window_id is None
        and transient_baseline_window_ids is None
    ):
        cached = _CACHE.get(
            app, window_index, screenshot=screenshot, window_id=window_id
        )
        if (
            cached is not None
            and cached.get("app", {}).get("pid") == app_info["pid"]
            and isinstance(cached.get("window"), dict)
            and _same_window(cached["window"], window)
        ):
            return cached
    resolved_index = window["index"]
    collection_status: dict[str, bool] = {}
    targets = _collect_with_timeout(
        app_info["name"] or app,
        window_index=resolved_index,
        window=window,
        expected_pid=app_info["pid"],
        retry_web_content=_needs_web_content_retry(app_info),
        collection_status=collection_status,
    )
    for target in targets:
        target["source_window_id"] = window["window_id"]
    transient_window = (
        _focused_transient_window(
            app_info,
            window,
            trusted_window_id=trusted_transient_window_id,
            baseline_window_ids=transient_baseline_window_ids,
        )
        if trusted_transient_window_id is not None
        or transient_baseline_window_ids is not None
        else None
    )
    if transient_window is not None:
        transient_targets = _collect_with_timeout(
            app_info["name"] or app,
            window_index=transient_window["index"],
            window=transient_window,
            expected_pid=app_info["pid"],
            collection_status=collection_status,
            window_frame_tolerance=4.0,
        )
        if len(transient_targets) > MAX_TRANSIENT_TARGETS:
            transient_targets = transient_targets[:MAX_TRANSIENT_TARGETS]
            collection_status["partial"] = True
        if len(targets) + len(transient_targets) > ax_driver.MAX_NODES:
            targets = targets[: max(0, ax_driver.MAX_NODES - len(transient_targets))]
            collection_status["partial"] = True
        offset = len(targets)
        for position, target in enumerate(transient_targets):
            target["target_id"] = f"t{offset + position:03d}"
            target["source_window_id"] = transient_window["window_id"]
        targets.extend(transient_targets)
    if not targets and resolved_index:
        raise ComputerUseError(
            "window_not_found", f"window index {resolved_index} is not available"
        )
    # collect() is bound to the single AX window whose frame uniquely matches
    # the selected CGWindow record.
    elements = []
    for target in targets:
        index = int(target["target_id"][1:])
        rect = target.get("rect") or [0, 0, 0, 0]
        elements.append(
            {
                "index": index,
                "role": target["role"],
                "subrole": target.get("subrole") or "",
                "parent_role": target.get("parent_role") or "",
                "label": target["text"],
                "value": target.get("value"),
                "actions": target.get("actions", []),
                "x": round(rect[0]),
                "y": round(rect[1]),
                "width": round(rect[2]),
                "height": round(rect[3]),
                "center": target.get("center")
                or [round(rect[0] + rect[2] / 2), round(rect[1] + rect[3] / 2)],
                "source_window_id": target["source_window_id"],
                "states": target.get("states") or [],
                "placeholder": target.get("placeholder"),
            }
        )
    live_elements = [target.get("element") for target in targets]
    tree_lines = [
        f"[{e['index']}] {e['role']}{'*' if 'AXPress' in e['actions'] else ''} {e['label'][:90]}"
        + (f" = {e['value']!r}" if e.get("value") is not None else "")
        for e in elements
    ]
    snapshot = {
        "snapshot_id": f"{app_info['pid']}:{window['window_id']}:{uuid.uuid4().hex}",
        "observed_at": time.time(),
        "app": app_info,
        "window_index": resolved_index,
        "window_id": window["window_id"],
        "window": window,
        "coordinate_space": "screen",
        "elements": elements,
        "element_count": len(elements),
        "tree_text": "\n".join(tree_lines),
        "truncated": collection_status.get("partial", False)
        or len(elements) >= ax_driver.MAX_NODES,
        "budget_exhausted": bool(collection_status.get("budget_exhausted")),
        "visible_window_ids": [
            str(record["window_id"]) for record in _window_records(app_info)
        ],
    }
    if transient_window is not None:
        snapshot["transient_window"] = transient_window
    _remember_live_elements(snapshot["snapshot_id"], live_elements)
    png = (
        screenshot_window(
            app_info["name"] or app,
            resolved_index,
            window_id=_cg_window_id(window["window_id"]),
            expected_pid=app_info["pid"],
        )
        if screenshot
        else None
    )
    if png is not None:
        snapshot["screenshot_png"] = png
    if transient_window is None:
        _CACHE.put(
            app, window_index, snapshot, screenshot=screenshot, window_id=window_id
        )
    return snapshot


def screenshot_window(
    app_name: str,
    window_index: int = 0,
    *,
    window_id: int | str | None = None,
    expected_pid: int | None = None,
) -> bytes | None:
    """Capture the app's front window via screencapture (needs Screen Recording)."""
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    candidates: list[int] = []
    for window in windows:
        owner = str(window.get("kCGWindowOwnerName", ""))
        if app_name.lower() not in owner.lower():
            continue
        if window.get("kCGWindowLayer", 99) != 0:
            continue
        if (
            expected_pid is not None
            and int(window.get("kCGWindowOwnerPID", -1)) != expected_pid
        ):
            continue
        window_number = window.get("kCGWindowNumber")
        if window_number is not None:
            candidates.append(int(window_number))
    if not candidates:
        raise ComputerUseError(
            "window_not_found", f"no on-screen window for {app_name!r}"
        )
    if window_id is not None:
        numeric_window_id = _cg_window_id(window_id)
        if numeric_window_id not in candidates:
            raise ComputerUseError(
                "window_not_found", f"window id {window_id} is not available"
            )
        window_number = numeric_window_id
    elif window_index >= len(candidates):
        raise ComputerUseError(
            "window_not_found", f"window index {window_index} is not available"
        )
    else:
        window_number = candidates[window_index]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        out_path = handle.name
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-o", f"-l{window_number}", out_path],
            capture_output=True,
            timeout=15,
        )
        png = Path(out_path).read_bytes()
        if result.returncode != 0 or len(png) < 8_000:
            raise ComputerUseError(
                "screenshot_failed",
                "screencapture returned no usable window image",
            )
        return png
    finally:
        Path(out_path).unlink(missing_ok=True)


def _element(snapshot: dict, element_index: int, live: bool = False) -> dict:
    for entry in snapshot["elements"]:
        if entry["index"] == element_index:
            return cast(dict, entry)
    raise ComputerUseError(
        "element_not_found",
        f"element {element_index} is not in the current snapshot "
        f"(snapshot {snapshot.get('snapshot_id')})",
    )


def _collect_with_timeout(
    app_name: str,
    *,
    window_index: int = 0,
    window: dict | None = None,
    expected_pid: int | None = None,
    timeout_s: float | None = None,
    retry_web_content: bool = False,
    collection_status: dict[str, bool] | None = None,
    window_frame_tolerance: float = 0.5,
    budget_s: float | None = None,
) -> list[dict]:
    """ax_driver.collect with a watchdog.

    macOS AX calls have no built-in timeout: when a target app's
    accessibility service wedges (Chrome does this in the field),
    AXUIElementCopyAttributeValue blocks forever and the whole agent run
    freezes mid-step (2026-09-27 dogfood). Run the walk in a daemon worker
    thread and convert a timeout into an honest ComputerUseError; a wedged
    worker leaks as a daemon instead of blocking the run forever.
    """
    import threading

    timeout_s = AX_COLLECT_TIMEOUT_S if timeout_s is None else timeout_s
    outcome: dict[str, Any] = {}
    partial_targets: list[dict] = []

    walk_status: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["value"] = ax_driver.collect(
                app_name,
                keep_elements=True,
                max_windows=1,
                window_index=window_index,
                window_frame=(
                    (
                        float(window["x"]),
                        float(window["y"]),
                        float(window["width"]),
                        float(window["height"]),
                    )
                    if window is not None
                    else None
                ),
                window_frame_tolerance=window_frame_tolerance,
                expected_pid=expected_pid,
                retry_web_content=retry_web_content,
                partial_out=partial_targets,
                budget_s=budget_s,
                walk_status=walk_status,
            )
        except ax_driver.AppNotFoundError as exc:
            outcome["error"] = ComputerUseError("app_not_found", str(exc))
        except Exception as exc:  # noqa: BLE001 - surfaced below
            outcome["error"] = exc

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout_s)
    if worker_thread.is_alive():
        if partial_targets:
            # The daemon may remain blocked in a target app's AX call. Copy
            # fully appended entries so the returned snapshot is immutable
            # while that abandoned worker eventually unwinds.
            completed = [dict(target) for target in partial_targets]
            for target in completed:
                rect = target.get("rect")
                if rect and "center" not in target:
                    target["center"] = [
                        round(rect[0] + rect[2] / 2),
                        round(rect[1] + rect[3] / 2),
                    ]
            if completed:
                if collection_status is not None:
                    collection_status["partial"] = True
                return completed
        raise ComputerUseError(
            "ax_unavailable",
            f"accessibility tree collection for {app_name!r} timed out "
            f"after {timeout_s:.0f}s; the app's AX service is likely wedged",
        )
    if "error" in outcome:
        error = cast(BaseException, outcome["error"])
        if isinstance(error, ComputerUseError):
            raise error
        raise ComputerUseError(
            "ax_unavailable",
            f"accessibility tree collection for {app_name!r} failed: {error}",
        ) from error
    if collection_status is not None:
        if walk_status.get("budget_exhausted") or walk_status.get("depth_cap"):
            collection_status["partial"] = True  # content was left unwalked
        if walk_status.get("budget_exhausted"):
            collection_status["budget_exhausted"] = True
    return cast(list[dict], outcome.get("value", []))


# An action re-walks the tree to find its target; it may take longer than an
# observation, because a target the observation saw must be found again.
ACTION_WALK_BUDGET_S = 4.0


def _live_element(
    snapshot: dict, element_index: int, *, validate_point: bool = True
) -> object:
    """Re-collect and return the live AX ref for an index, if still present."""
    # Every AX action resolves its target here first.
    _await_ax_actions_ready(snapshot["app"])
    expected = _element(snapshot, element_index)
    source_window_id = expected.get("source_window_id", snapshot.get("window_id"))
    is_transient = source_window_id != snapshot.get("window_id")
    center = expected.get("center")
    point = tuple(center) if isinstance(center, list) and len(center) == 2 else None
    anchor = _validate_snapshot_window(snapshot)
    if is_transient:
        observed_transient = snapshot.get("transient_window")
        if not isinstance(observed_transient, dict):
            raise ComputerUseError(
                "target_drift", "transient target has no trusted window identity"
            )
        current_window = _focused_transient_window(
            snapshot["app"],
            anchor,
            trusted_window_id=source_window_id,
        )
        if current_window is None or not _same_window(
            observed_transient, current_window
        ):
            raise ComputerUseError(
                "target_drift", "transient companion changed or lost focus; re-observe"
            )
        if validate_point:
            raise ComputerUseError(
                "synthetic_input_blocked",
                "transient companion targets require an exact Accessibility action",
                (
                    "Use an exact editable or pressable control from the fresh snapshot.",
                ),
            )
    else:
        # AX actions/values reach a covered element directly; only the
        # global-HID route is hit-tested and needs the point to be topmost.
        current_window = _validate_snapshot_window(
            snapshot,
            point=point if validate_point else None,
            require_topmost=not _background_delivery(snapshot),
        )
    fresh = _collect_with_timeout(
        snapshot["app"]["name"],
        window_index=current_window["index"],
        window=current_window,
        expected_pid=int(snapshot["app"]["pid"]),
        retry_web_content=_needs_web_content_retry(snapshot["app"]),
        window_frame_tolerance=4.0 if is_transient else 0.5,
        budget_s=ACTION_WALK_BUDGET_S,
    )
    for target in fresh:
        fresh_index = int(target["target_id"][1:])
        expected_local_index = (
            element_index
            - sum(
                1
                for entry in snapshot.get("elements", [])
                if entry.get("source_window_id") == snapshot.get("window_id")
            )
            if is_transient
            else element_index
        )
        if fresh_index == expected_local_index:
            comparisons = (
                ("role", "role"),
                ("label", "text"),
                ("center", "center"),
            )
            if any(
                expected.get(snapshot_key) is not None
                and expected.get(snapshot_key) != target.get(target_key)
                for snapshot_key, target_key in comparisons
            ):
                raise ComputerUseError(
                    "element_not_found",
                    f"element {element_index} changed since snapshot "
                    f"{snapshot.get('snapshot_id')}; re-observe before acting",
                )
            live = target.get("element")
            _rouse_woken_renderer(snapshot, live)
            return live
    raise ComputerUseError(
        "element_not_found",
        f"element {element_index} no longer present in the fresh AX tree",
    )


def _rouse_woken_renderer(snapshot: dict, live: object) -> None:
    """Make a Chromium renderer woken while hidden accept AX actions again.

    It serves its tree but silently drops AXPress/AXFocused until it handles
    one renderer-side action; AXScrollToVisible on the element about to be
    acted on is such an action and is otherwise harmless (measured on
    Electron: every press dropped before it, none after).
    """
    pid = int(snapshot["app"]["pid"])
    if live is None or not ax_driver.renderer_was_woken(pid):
        return
    try:
        err = ax_driver.AXUIElementPerformAction(live, "AXScrollToVisible")
    except Exception:  # noqa: BLE001 - keep the mark; the next action retries
        return
    if err in (None, 0):
        # Cleared only once the renderer took the action; otherwise the next
        # AX action tries again.
        ax_driver.clear_woken(pid)
        time.sleep(0.2)


def _numeric_request(current: object, value: str) -> float | None:
    """The number to write when the control holds a number (a stepper)."""
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        return None
    try:
        return float(value)
    except ValueError:
        return None


# Chromium applies an accepted AX write asynchronously; reading back at once
# misses it, and the typing fallback then applies the value a second time
# ("2" became "22"). A write is given this long to show.
AX_WRITE_READBACK_S = 0.5


def _await_readback(live: object, value: str, numeric: float | None) -> str | None:
    deadline = time.monotonic() + AX_WRITE_READBACK_S
    while True:
        if numeric is None:
            text = _read_value(live)
        else:
            raw = ax_driver._get(live, "AXValue")
            if (
                isinstance(raw, (int, float))
                and not isinstance(raw, bool)
                and raw == numeric
            ):
                return value
            text = raw if isinstance(raw, str) else None
        if text == value or time.monotonic() > deadline:
            return text
        time.sleep(0.05)


def _read_value(live_element: object) -> str | None:
    value = ax_driver._get(live_element, "AXValue")
    return value if isinstance(value, str) else None


def _normalized_finder_editor_value(live_element: object) -> str | None:
    value = _read_value(live_element)
    return (
        value.replace("\u200b", "").replace("\ufeff", "") if value is not None else None
    )


def is_finder_snapshot(snapshot: dict) -> bool:
    """Identify Finder only from OS-resolved bundle metadata."""
    return (
        str(snapshot.get("app", {}).get("bundleId", "")).casefold()
        == "com.apple.finder"
    )


def _is_selected_finder_row_under_focused_outline(snapshot: dict, live: object) -> bool:
    """Allow Enter only for Finder's exact selected item or inline editor."""
    if not is_finder_snapshot(snapshot):
        return False
    focused = _focused_ax_element(snapshot["app"])
    if focused is None or ax_driver._get(focused, "AXRole") != "AXOutline":
        return False
    role = ax_driver._get(live, "AXRole")
    parent = ax_driver._get(live, "AXParent")
    if role == "AXRow":
        return ax_driver._get(live, "AXSelected") is True and parent == focused
    if role != "AXTextField" or ax_driver._get(live, "AXSelected") is not True:
        return False
    cell = parent
    row = ax_driver._get(cell, "AXParent") if cell is not None else None
    return (
        ax_driver._get(cell, "AXRole") == "AXCell"
        and ax_driver._get(cell, "AXSelected") is True
        and ax_driver._get(row, "AXRole") == "AXRow"
        and ax_driver._get(row, "AXSelected") is True
        and ax_driver._get(row, "AXParent") == focused
    )


def inspect_focused_element(
    snapshot: dict,
    element_index: int,
    *,
    allow_selected_finder_row: bool = False,
) -> dict:
    """Return an indexed target only when it is the exact focused AX element.

    Keyboard activation is routed by focus rather than coordinates. Resolve
    the snapshot entry back to a live Accessibility object so callers do not
    trust planner text or a stale serialized label when deciding whether
    Enter or Space can commit an external effect.
    """
    entry = _element(snapshot, element_index)
    live = _live_element(snapshot, element_index, validate_point=False)
    focused = _focused_ax_element(snapshot["app"])
    exact_focus = focused is not None and live == focused
    finder_row = (
        allow_selected_finder_row
        and _is_selected_finder_row_under_focused_outline(snapshot, live)
    )
    if not exact_focus and not finder_row:
        raise ComputerUseError(
            "target_drift",
            "keyboard target is not the exact focused Accessibility element",
        )
    return dict(entry)


def _finish_action(
    app: str,
    snapshot: dict,
    result: dict,
    *,
    verified: bool | None,
    verification: str,
    include_post_state: bool = False,
) -> dict:
    """Return honest outcome metadata plus a best-effort fresh local state.

    ``route`` names the transport that delivered the action and ``effect`` is
    the strongest claim the evidence supports: ``confirmed`` needs a positive
    read-back, ``suspected_noop`` a negative one; dispatch alone is only ever
    ``unverifiable``.
    """
    result.setdefault("route", _route_for_mode(str(result.get("mode", ""))))
    result.update(
        {
            "attempted": True,
            "verified": verified,
            "verification": verification,
            "effect": (
                "confirmed"
                if verified is True
                else "suspected_noop"
                if verified is False
                else "unverifiable"
            ),
            "window_id": snapshot.get("window_id"),
        }
    )
    if not include_post_state:
        return result
    try:
        result["post_action_state"] = get_app_state(
            app,
            screenshot=False,
            use_cache=False,
            window_id=snapshot.get("window_id"),
            activate=OBSERVE_BY_ROUTE,
        )
    except ComputerUseError as exc:
        result["post_action_state"] = None
        result["post_action_state_error"] = {
            "code": exc.code,
            "message": exc.message,
        }
    return result


def _window_center(snapshot: dict) -> tuple[float, float]:
    window = snapshot["window"]
    return (
        float(window["x"]) + float(window["width"]) / 2,
        float(window["y"]) + float(window["height"]) / 2,
    )


ROUTE_ACCESSIBILITY = "accessibility"
ROUTE_PID = "pid_events"
ROUTE_GLOBAL_HID = "global_hid"


def _route_for_mode(mode: str) -> str:
    if mode.startswith("SkyLight"):
        return ROUTE_PID
    if "CGEvent" in mode or mode == "synthetic-typing":
        return ROUTE_GLOBAL_HID
    return ROUTE_ACCESSIBILITY


def _require_background_if_forced() -> None:
    if (
        background_input.delivery_mode() == "background"
        and not background_input.skylight_available()
    ):
        raise ComputerUseError(
            "action_failed",
            f"{background_input.DELIVERY_ENV}=background but the SkyLight "
            "input route is unavailable; refusing to fall back to global HID",
        )


def _background_delivery(snapshot: dict) -> bool:
    """Whether synthetic input for this snapshot goes to its exact pid/window.

    Finder's inline-rename editor rebinds focus asynchronously and its safety
    checks are built around foreground activation, so it keeps the HID route.
    """
    if is_finder_snapshot(snapshot):
        return False
    _require_background_if_forced()
    return background_input.background_enabled()


def _target_ids(snapshot: dict) -> tuple[int, int]:
    return int(snapshot["app"]["pid"]), _cg_window_id(snapshot["window"]["window_id"])


def _live_front_pid() -> int | None:
    """Current front pid from WindowServer, with a live AppKit fallback."""
    front = background_input.front_pid()
    if front is None:
        try:
            from AppKit import NSWorkspace

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            front = int(app.processIdentifier()) if app is not None else None
        except Exception:  # noqa: BLE001 - no front app to restore to
            front = None
    return front


def _frontmost_window() -> tuple[int, int] | None:
    """(pid, CGWindowID) of the user's key window.

    The key window is the front app's ``AXFocusedWindow`` (CG z-order does
    not identify it and skips floating panels). Falls back to
    ``(front app pid, 0)`` when it cannot be mapped, so focus is handed back
    by re-activating that app rather than to a guessed window.
    """
    front = _live_front_pid()
    if front is None:
        return None
    return front, _key_window_id(front) or 0


def _key_window_id(pid: int) -> int | None:
    """CGWindowID of ``pid``'s focused window, or None when unreadable."""
    if ax_driver.AS is None:
        return None
    app_element = ax_driver.AXUIElementCreateApplication(pid)
    return background_input.ax_window_id(ax_driver._get(app_element, "AXFocusedWindow"))


def _sheet_owner_id(pid: int) -> int | None:
    """CG id of the window whose sheet holds ``pid``'s key focus, if any.

    Making a window key also makes its attached sheet (an alert, a save
    panel) key; that is still the target, not a window the user picked.
    """
    if ax_driver.AS is None:
        return None
    app_element = ax_driver.AXUIElementCreateApplication(pid)
    focused = ax_driver._get(app_element, "AXFocusedWindow")
    if focused is None or ax_driver._get(focused, "AXRole") != "AXSheet":
        return None
    return background_input.ax_window_id(ax_driver._get(focused, "AXParent"))


def _sheet_summary(pid: int) -> str:
    app_element = ax_driver.AXUIElementCreateApplication(pid)
    sheet = ax_driver._get(app_element, "AXFocusedWindow")
    texts, buttons, stack, seen = [], [], [sheet], 0
    while stack and seen < 200:
        node = stack.pop()
        seen += 1
        role = ax_driver._get(node, "AXRole")
        if role == "AXStaticText":
            texts.append(str(ax_driver._get(node, "AXValue") or ""))
        elif role == "AXButton" and ax_driver._get(node, "AXTitle"):
            buttons.append(str(ax_driver._get(node, "AXTitle")))
        stack.extend(reversed(ax_driver._as_list(ax_driver._get(node, "AXChildren"))))
    return f"{' '.join(t for t in texts if t)[:300]} (buttons: {', '.join(buttons)})"


def _restore_user_focus(
    previous: tuple[int, int] | None, pid: int, window_id: int
) -> bool | None:
    """Hand keyboard focus back to the user's window after a background click.

    Returns None when there was nothing to restore (the target already was the
    user's front window), otherwise whether restoration was accepted. A
    different window of the same app is restored too: focus-without-raise made
    the target window key, and the user's typing must not follow it.
    """
    if previous is None or tuple(previous) == (pid, window_id):
        return None
    previous_pid, previous_wid = previous
    current = _live_front_pid()
    if current is None:
        # Restoring a stale capture when the live foreground is unknown can
        # override a user switch that happened during the gesture.
        return False
    if current not in (previous_pid, pid):
        # The user switched to a third app during the gesture: their new
        # choice wins; restoring the stale capture would steal it back.
        return None
    if current == previous_pid and previous_wid:
        key = _key_window_id(previous_pid)
        if (
            key is not None
            and key not in (previous_wid, window_id)
            and not (previous_pid == pid and _sheet_owner_id(pid) == window_id)
        ):
            # The user picked another window of their app mid-gesture.
            return None
    other_app = previous_pid != pid
    if other_app and background_input.front_process_matches(pid, window_id):
        key = _key_window_id(pid)
        if key is not None and key != window_id:
            # The user switched to another window of the target app.
            return None
        # The target activated itself in response to the click (some apps do
        # on mouseDown); re-activate the user's app rather than leave it behind.
        return _activate_app(previous_pid)
    if previous_wid and background_input.restore_focus_after_without_raise(
        previous_pid, previous_wid, pid, window_id
    ):
        return True
    # No window to target, or the record was refused: re-activating the
    # user's app is the remaining way to give their keyboard back.
    return _activate_app(previous_pid) if other_app else False


def _undo_self_activation(previous_pid: int | None, pid: int) -> bool | None:
    """Re-activate the user's app if the target activated itself.

    Used after a gesture that moved no focus itself (the target was already
    key, or a transient held focus). Returns None when the foreground did not
    move to the target, otherwise whether the user's app was re-activated.
    """
    if previous_pid is None or previous_pid == pid:
        return None
    if _live_front_pid() != pid:
        return None
    return _activate_app(previous_pid)


def _settle_and_undo_self_activation(
    previous_pid: int | None, pid: int, state: dict
) -> None:
    try:
        # Let the target consume the stream (and any activation it triggers).
        time.sleep(0.05)
        restored = _undo_self_activation(previous_pid, pid)
    except Exception:  # noqa: BLE001 - surfaced as focus_restored
        restored = False
    if restored is not None:
        state["focus_restored"] = restored


# Private snapshot key set by _prepare_synthetic_action when the keyboard
# target is a validated transient companion window rather than the snapshot
# window itself.
_KEYBOARD_WINDOW = "_keyboard_window"


@contextmanager
def _keyed_target(snapshot: dict, *, force: bool = False):
    """Make the target window key (without raising it) for one keyboard gesture.

    Keys posted to a pid land on that process's key window. When the target
    is another window of the app the user is typing in, or a window of an app
    whose key window is a different one, posting straight to the pid would
    type into the wrong window. The target is made key, verified to hold
    focus, and the user's window gets focus back afterwards, all under the
    gesture lock. Yields a dict that receives ``focus_restored``.
    """
    pid, window_id = _target_ids(snapshot)
    state: dict[str, Any] = {}
    transient = snapshot.get(_KEYBOARD_WINDOW)
    with background_input.GESTURE_LOCK:
        if transient is not None:
            # A validated transient companion (popover, completion list) of
            # the snapshot window already holds focus; making the anchor key
            # would dismiss it. Keys go to it only while it is still exactly
            # the focused window.
            front = _live_front_pid()
            _validate_focused_window(
                snapshot,
                transient,
                require_active_app=False,
                require_exact_window_id=True,
            )
            try:
                yield state
            finally:
                _settle_and_undo_self_activation(front, pid, state)
            return
        previous = _frontmost_window()
        if previous is None:
            raise ComputerUseError(
                "action_failed", "could not capture the user's focused window"
            )
        # ``force``: AX keeps naming an inactive app's last key window, but
        # in-process it resigned key; a menu command that acts on the key
        # window (Electron's zoom roles) needs it really key.
        if (
            not force
            and _key_window_id(pid) == window_id
            and (previous[0] != pid or previous[1] == window_id)
        ):
            # Already the app's key window and the user is either elsewhere
            # or in that very window: keys to the pid reach it without any
            # focus change.
            _validate_focused_window(
                snapshot, require_active_app=False, require_exact_window_id=True
            )
            try:
                yield state
            finally:
                # Nothing was moved, but a key the target handles may make
                # it activate itself; hand the foreground back if so.
                _settle_and_undo_self_activation(previous[0], pid, state)
            return
        try:
            if not _synthesize(
                background_input.activate_without_raise, pid, window_id, previous[1]
            ):
                raise ComputerUseError(
                    "synthetic_input_blocked",
                    "window could not be made key in the background",
                )
            # Key status moves asynchronously; wait for it rather than guess.
            # Within the user's own (active) app, AppKit can hand key status
            # back to the user's window ~0.1 s after the switch (measured in
            # TextEdit), so a bounce-back is answered with one more switch.
            started = last_post = time.monotonic()
            reposts = 0
            validated = False
            while True:
                time.sleep(0.02)
                if _sheet_owner_id(pid) == window_id:
                    break
                if _key_window_id(pid) == window_id:
                    try:
                        _validate_focused_window(
                            snapshot,
                            require_active_app=False,
                            require_exact_window_id=True,
                        )
                        validated = True
                        break
                    except ComputerUseError:
                        pass  # AX's focused window can trail key status
                now = time.monotonic()
                if now - started > 1.5:
                    break
                if reposts < 3 and now - last_post > 0.25:
                    reposts += 1
                    last_post = now
                    _synthesize(
                        background_input.activate_without_raise,
                        pid,
                        window_id,
                        _key_window_id(previous[0]) or previous[1],
                    )
            if _sheet_owner_id(pid) == window_id:
                # The target is blocked by its own sheet; keys would go to it.
                raise ComputerUseError(
                    "synthetic_input_blocked",
                    f"window {snapshot.get('window_id')} is showing a dialog: "
                    f"{_sheet_summary(pid)}",
                    ("Answer the dialog first (press one of its buttons).",),
                )
            if not validated:
                _validate_focused_window(
                    snapshot, require_active_app=False, require_exact_window_id=True
                )
            yield state
            # Let the target consume the stream before focus moves back.
            time.sleep(0.05)
        finally:
            try:
                state["focus_restored"] = _restore_user_focus(previous, pid, window_id)
            except Exception:  # noqa: BLE001 - surfaced as focus_restored
                state["focus_restored"] = False


@contextmanager
def _guard_user_focus(snapshot: dict):
    """Hand the user's key window back if an AX action moved it.

    Accessibility actions need no focus, but some move it as a side effect
    (a context menu or its item makes the clicked window key in AppKit,
    measured in TextEdit); the user's next keystroke would then land in the
    target. Yields a dict that receives ``focus_restored`` when it was moved.
    """
    state: dict[str, Any] = {}
    if not _background_delivery(snapshot):
        yield state
        return
    pid, window_id = _target_ids(snapshot)
    # Held across the action so no other gesture interleaves with the focus
    # capture and the hand-back.
    with background_input.GESTURE_LOCK:
        previous = _frontmost_window()
        try:
            yield state
        finally:
            current = _frontmost_window() if previous is not None else None
            # Only key status landing on the exact target window is the
            # action's doing; any other switch meanwhile is the user's to keep.
            if (
                previous is not None
                and tuple(previous) != (pid, window_id)
                and current is not None
                and tuple(current) == (pid, window_id)
            ):
                try:
                    state["focus_restored"] = _restore_user_focus(
                        previous, pid, window_id
                    )
                except Exception:  # noqa: BLE001 - surfaced as focus_restored
                    state["focus_restored"] = False


def _synthesize(primitive, *args, **kwargs) -> bool:
    """Run a background input primitive; SPI/ctypes errors become typed."""
    try:
        return bool(primitive(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - surfaced as action_failed
        raise ComputerUseError(
            "action_failed", f"background input failed: {exc}"
        ) from exc


def _activate_app(pid: int) -> bool:
    running = ax_driver._application_for_pid(pid)
    if running is None:
        return False
    try:
        return bool(running.activateWithOptions_(1 << 1))
    except Exception:  # noqa: BLE001 - restoration is best-effort
        return False


def _window_origin(window: dict) -> tuple[float, float] | None:
    left, top = window.get("x"), window.get("y")
    if isinstance(left, (int, float)) and isinstance(top, (int, float)):
        return float(left), float(top)
    return None


def _pixel_click(
    snapshot: dict,
    x: float,
    y: float,
    *,
    button: str = "left",
    count: int = 1,
    flags: int = 0,
) -> dict:
    """Click a screen point inside the snapshot window; returns delivery metadata.

    Background delivery posts to the exact pid/window without moving the
    cursor or raising the window, so occlusion is irrelevant. Foreground
    delivery posts global HID events and therefore requires the window to be
    topmost at the point. There is no silent fallback between the two.
    """
    if _background_delivery(snapshot):
        window = _validate_snapshot_window(
            snapshot, point=(x, y), require_topmost=False
        )
        pid, window_id = _target_ids(snapshot)
        # One transaction under the gesture lock: concurrent clicks must not
        # restore focus in the middle of each other's streams, and focus is
        # handed back even when synthesis fails after the target was focused.
        with background_input.GESTURE_LOCK:
            previous = _frontmost_window()
            if previous is None:
                # Focus-without-raise defocuses whatever is front; without a
                # capture there is nothing to hand focus back to.
                raise ComputerUseError(
                    "action_failed",
                    "could not capture the user's focused window before clicking",
                )
            restored = None
            try:
                if not _synthesize(
                    background_input.click,
                    pid,
                    window_id,
                    float(x),
                    float(y),
                    button=button,
                    count=count,
                    flags=flags,
                    window_origin=_window_origin(window),
                    front_wid=previous[1],
                ):
                    raise ComputerUseError(
                        "action_failed", "background click could not be synthesized"
                    )
                # Let the target consume the stream before focus moves back.
                time.sleep(0.05)
            finally:
                # Never let a restoration error mask the click's own outcome
                # (a delivered click reported as failed invites a retry).
                try:
                    restored = _restore_user_focus(previous, pid, window_id)
                except Exception:  # noqa: BLE001 - surfaced as focus_restored
                    restored = False
        result = {
            "mode": "SkyLight-click",
            "route": ROUTE_PID,
            "button": button,
            "click_count": count,
            "focus_restored": restored,
        }
        if restored is False:
            # The click itself was delivered, so this is not an action failure
            # (raising would invite a duplicate retry); surface it instead.
            result["warning"] = (
                "keyboard focus could not be handed back to the user's window"
            )
        return result
    if flags:
        raise ComputerUseError(
            "synthetic_input_blocked",
            "modifier clicks need background delivery; a foreground click "
            "would hold modifiers on the user's keyboard",
        )
    _validate_snapshot_window(snapshot, point=(x, y))
    ax_driver._cg_click(float(x), float(y), clicks=count, button=button)
    return {
        "mode": "CGEvent-click",
        "route": ROUTE_GLOBAL_HID,
        "button": button,
        "click_count": count,
    }


def _modifier_flags(modifiers: list[str] | str | None) -> int:
    if not modifiers:
        return 0
    names = modifiers.split("+") if isinstance(modifiers, str) else list(modifiers)
    flags = 0
    for name in (n.strip().lower() for n in names):
        if not name:
            continue
        if name not in MODIFIER_FLAGS:
            raise ComputerUseError("unsupported_key", f"unknown modifier {name!r}")
        flags |= MODIFIER_FLAGS[name]
    return flags


def drag(
    app: str,
    from_x: int,
    from_y: int,
    to_x: int,
    to_y: int,
    expected_snapshot: dict | None = None,
    window_id: int | str | None = None,
    include_post_state: bool = False,
) -> dict:
    """Left-button drag between two screen points inside the snapshot window."""
    snapshot = expected_snapshot or get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    if not _background_delivery(snapshot):
        raise ComputerUseError(
            "synthetic_input_blocked",
            "drag needs background delivery; foreground drags would move the "
            "user's cursor",
        )
    pid, target_wid = _target_ids(snapshot)
    with background_input.GESTURE_LOCK:
        # Validate under the lock so no other gesture moves the window
        # between the check and the posted stream.
        window = _validate_snapshot_window(
            snapshot, point=(from_x, from_y), require_topmost=False
        )
        _validate_snapshot_window(snapshot, point=(to_x, to_y), require_topmost=False)
        previous = _frontmost_window()
        if previous is None:
            raise ComputerUseError(
                "action_failed",
                "could not capture the user's focused window before dragging",
            )
        restored = None
        try:
            if not _synthesize(
                background_input.drag,
                pid,
                target_wid,
                float(from_x),
                float(from_y),
                float(to_x),
                float(to_y),
                window_origin=_window_origin(window),
                front_wid=previous[1],
            ):
                raise ComputerUseError(
                    "action_failed", "background drag could not be synthesized"
                )
            time.sleep(0.05)
        finally:
            try:
                restored = _restore_user_focus(previous, pid, target_wid)
            except Exception:  # noqa: BLE001 - surfaced as focus_restored
                restored = False
    delivery = {
        "mode": "SkyLight-drag",
        "route": ROUTE_PID,
        "from": [from_x, from_y],
        "to": [to_x, to_y],
        "focus_restored": restored,
    }
    if restored is False:
        delivery["warning"] = (
            "keyboard focus could not be handed back to the user's window"
        )
    return _finish_action(
        app,
        snapshot,
        delivery,
        verified=None,
        verification="synthetic drag emitted; outcome not asserted",
        include_post_state=include_post_state,
    )


def _same_process(expected: dict):
    """The running app for ``expected``'s pid, only if it is still that process.

    A pid can be recycled after the observed process exits, so the full
    identity (bundle, name, launch time) is checked on every lookup.
    """
    if any(expected.get(key) is None for key in _PROCESS_IDENTITY):
        # Without the launch time a recycled pid is indistinguishable.
        raise ComputerUseError("target_drift", "selected app identity is incomplete")
    running = ax_driver._application_for_pid(int(expected["pid"]))
    if running is None:
        raise ComputerUseError("target_drift", "selected app exited")
    info = _resolved_app_info(running)
    for key in (*_PROCESS_IDENTITY, "name"):
        wanted = expected.get(key)
        if wanted is not None and info.get(key) != wanted:
            raise ComputerUseError("target_drift", "selected app identity changed")
    return running


_PROCESS_IDENTITY = ("pid", "bundleId", "processStartTime")


def _borrow_foreground(snapshot: dict) -> None:
    """Activate exactly the snapshot's process for a foreground-only step.

    Background observation leaves the target inactive, so paths that still
    need AppKit activation (Save via the menu bar, TextEdit document binding,
    the Cmd+A synthetic-typing fallback) activate it here, narrowly and only
    after the process identity matches the observation.
    """
    # Decide on the process's actual state, not the delivery policy: the user
    # may have switched away since an activating observation.
    expected = snapshot.get("app") or {}
    running = _same_process(expected)
    if bool(running.isActive()):
        return
    # A stale or closed window must not cost the user their foreground: check
    # the exact observed window before activating (callers re-validate after).
    _validate_snapshot_window(snapshot, require_topmost=False)
    try:
        accepted = bool(running.activateWithOptions_(1 << 1))
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed failure
        raise ComputerUseError("action_failed", "could not activate target") from exc
    if not accepted:
        raise ComputerUseError("action_failed", "target refused activation")
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        # Re-resolve each poll: without an NSRunLoop in this process a held
        # NSRunningApplication never refreshes its isActive property.
        current = _same_process(expected)
        if bool(current.isActive()):
            time.sleep(0.15)  # let AppKit settle key/main window
            return
        time.sleep(0.05)
    raise ComputerUseError("target_drift", "target did not become active")


def observation_activates(app_info: dict | None) -> bool:
    """Whether observing ``app_info`` must activate it first.

    With background delivery every action the agent loop can plan (clicks,
    scrolls, fills, Enter/Tab/Escape/arrows/Space) is routed to the exact
    pid/window, so observation leaves the user's front app alone. Activation
    stays for foreground delivery, Finder (its rename flow is built around
    activation) and an app whose identity is not yet known.
    """
    bundle = str((app_info or {}).get("bundleId") or "").casefold()
    if not bundle or bundle == "com.apple.finder":
        return True
    return not background_input.background_enabled()


def _keyboard_background(snapshot: dict, modifiers: int = 0) -> bool:
    """Whether a key/text dispatch can go to the target pid in the background.

    Command chords are menu key equivalents: AppKit only dispatches them (and
    only enables the menu items) while the app is active, so they keep the
    foreground route.
    """
    return _background_delivery(snapshot) and not modifiers & MODIFIER_FLAGS["cmd"]


def _focus_fields(state: dict) -> dict:
    """``focus_restored`` (plus a warning when it failed) from a
    :func:`_keyed_target` state; None when focus never moved."""
    if state.get("focus_restored") is None:
        return {"focus_restored": None}
    fields: dict[str, Any] = {"focus_restored": state["focus_restored"]}
    if state["focus_restored"] is False:
        # The input itself was delivered, so this is not an action failure
        # (raising would invite a duplicate retry); surface it instead.
        fields["warning"] = (
            "keyboard focus could not be handed back to the user's window"
        )
    return fields


def _send_key(
    snapshot: dict,
    keycode: int,
    modifiers: int,
    background: bool,
    delivery: dict | None = None,
) -> str:
    """Post one key; ``delivery`` (if given) receives the focus outcome."""
    if background:
        pid, _ = _target_ids(snapshot)
        with _keyed_target(snapshot) as state:
            if not _synthesize(background_input.press_key, pid, keycode, modifiers):
                raise ComputerUseError(
                    "action_failed", "background key could not be synthesized"
                )
        if delivery is not None:
            delivery.update(_focus_fields(state))
        return ROUTE_PID
    if modifiers:
        ax_driver._press_key(keycode, modifiers)
    else:
        ax_driver._press_key(keycode)
    return ROUTE_GLOBAL_HID


def _validate_focused_window(
    snapshot: dict,
    expected_window: dict | None = None,
    *,
    require_active_app: bool = True,
    allow_exact_main_window: bool = False,
    require_exact_window_id: bool = False,
) -> None:
    services = ax_driver.AS
    expected_pid = int(snapshot["app"]["pid"])
    application_class = getattr(services, "NSRunningApplication", None)
    resolver = getattr(
        application_class, "runningApplicationWithProcessIdentifier_", None
    )
    if resolver is not None:
        try:
            running = resolver(expected_pid)
            is_active = getattr(running, "isActive", None)
            active = bool(is_active()) if callable(is_active) else False
        except Exception:  # noqa: BLE001 - identity probes fail closed
            active = False
    else:
        # Compatibility fallback for older framework bridges. Production
        # PyObjC exposes NSRunningApplication.isActive. This fallback is used
        # only when an older bridge lacks that resolver API entirely.
        workspace = services.NSWorkspace.sharedWorkspace() if services else None
        frontmost = workspace.frontmostApplication() if workspace is not None else None
        active = (
            frontmost is not None and int(frontmost.processIdentifier()) == expected_pid
        )
    if require_active_app and not active:
        raise ComputerUseError(
            "target_drift",
            f"pid {snapshot['app']['pid']} is no longer frontmost; re-observe",
        )
    focused = _focused_ax_window(snapshot["app"])
    if focused is None and allow_exact_main_window:
        app_element = _pid_app_element(
            {"name": snapshot["app"]["name"], "pid": expected_pid}
        )
        main = ax_driver._get(app_element, "AXMainWindow")
        windows = ax_driver._app_windows(app_element)
        detached = ax_driver._get(app_element, "AXFocusedUIElement")
        main_frame = ax_driver._point_size(main) if main is not None else None
        detached_frame = (
            ax_driver._point_size(detached) if detached is not None else None
        )
        detached_parent = (
            ax_driver._get(detached, "AXParent") if detached is not None else None
        )
        main_x, main_y, main_w, main_h = main_frame or (0.0, 0.0, 0.0, 0.0)
        detached_x, detached_y, detached_w, detached_h = detached_frame or (
            -1.0,
            -1.0,
            0.0,
            0.0,
        )
        detached_is_bound = (
            main_frame is not None
            and detached_frame is not None
            and ax_driver._get(detached, "AXRole") == "AXTextField"
            and ax_driver._get(detached, "AXFocused") is True
            and detached_parent == app_element
            and detached_x >= main_x
            and detached_y >= main_y
            and detached_x + detached_w <= main_x + main_w
            and detached_y + detached_h <= main_y + main_h
        )
        if (
            detached_is_bound
            and main is not None
            and any(main == window for window in windows)
            and any(detached == window for window in windows)
        ):
            focused = main
    focused_frame = ax_driver._point_size(focused) if focused is not None else None
    expected = expected_window or snapshot["window"]
    if require_exact_window_id:
        focused_window_id = background_input.ax_window_id(focused)
        expected_window_id = _cg_window_id(expected["window_id"])
        if focused_window_id != expected_window_id:
            raise ComputerUseError(
                "target_drift",
                f"window {expected['window_id']} is not the focused AX window; re-observe",
            )
    expected_frame = (
        float(expected["x"]),
        float(expected["y"]),
        float(expected["width"]),
        float(expected["height"]),
    )
    anchor_focus = expected.get("window_id") == snapshot.get("window_id")
    frame_matches = focused_frame is not None and (
        all(
            abs(actual - wanted) <= 0.5
            for actual, wanted in zip(focused_frame, expected_frame, strict=True)
        )
        if anchor_focus
        else _ax_cg_frames_match(focused_frame, expected_frame)
    )
    if not frame_matches:
        raise ComputerUseError(
            "target_drift",
            f"window {expected['window_id']} is not the focused AX window; re-observe",
        )


def raise_selected_window(
    app: str, snapshot: dict, *, focus_exact_window: bool = False
) -> dict:
    """Raise only the exact PID-bound selected AX window.

    This is a bounded recovery for an already observed occlusion. It never
    moves, resizes, closes, or chooses another window.
    """

    expected = snapshot.get("window")
    expected_app = snapshot.get("app") or {}
    if not isinstance(expected, dict) or "pid" not in expected_app:
        raise ComputerUseError(
            "stale_observation", "selected window identity is missing"
        )
    _validate_snapshot_window(snapshot)
    app_element, app_info = _resolve_app(
        f"pid:{int(expected_app['pid'])}", activate=False
    )
    for key in ("pid", "bundleId", "name", "processStartTime"):
        wanted = expected_app.get(key)
        if wanted is not None and app_info.get(key) != wanted:
            raise ComputerUseError("target_drift", "selected app identity changed")
    current = _select_window(app_info, window_id=expected["window_id"])
    if not _same_window(expected, current):
        raise ComputerUseError("target_drift", "selected window moved before recovery")
    frame = (
        float(current["x"]),
        float(current["y"]),
        float(current["width"]),
        float(current["height"]),
    )
    matches = [
        window
        for window in ax_driver._app_windows(app_element)
        if (candidate := ax_driver._point_size(window)) is not None
        and _ax_cg_frames_match(candidate, frame)
    ]
    if len(matches) != 1 or "AXRaise" not in ax_driver._action_names(matches[0]):
        raise ComputerUseError(
            "target_occluded",
            "selected window cannot be raised unambiguously; move the covering window",
        )

    # The approval sheet leaves Rapid frontmost. Only after the PID, window ID,
    # frame, and unique AX window have matched the approved observation may we
    # activate that exact process. Activation can change AX/window ordering, so
    # resolve and validate the same identities again before AXRaise.
    app_element, activated_info = _resolve_app(
        f"pid:{int(expected_app['pid'])}", activate=True
    )
    for key in ("pid", "bundleId", "name", "processStartTime"):
        wanted = expected_app.get(key)
        if wanted is not None and activated_info.get(key) != wanted:
            raise ComputerUseError(
                "target_drift", "selected app identity changed during focus recovery"
            )
    activated = _select_window(activated_info, window_id=expected["window_id"])
    if not _same_window(expected, activated):
        raise ComputerUseError(
            "target_drift", "selected window moved during focus recovery"
        )
    matches = [
        window
        for window in ax_driver._app_windows(app_element)
        if (candidate := ax_driver._point_size(window)) is not None
        and _ax_cg_frames_match(candidate, frame)
    ]
    if len(matches) != 1 or "AXRaise" not in ax_driver._action_names(matches[0]):
        raise ComputerUseError(
            "target_occluded",
            "selected window changed while restoring application focus",
        )
    if focus_exact_window:
        focus_err = ax_driver.AXUIElementSetAttributeValue(matches[0], "AXMain", True)
        if focus_err != ax_driver.kAXErrorSuccess:
            raise ComputerUseError(
                "target_occluded", "selected window rejected exact focus restoration"
            )
    err = ax_driver.AXUIElementPerformAction(matches[0], "AXRaise")
    if err != ax_driver.kAXErrorSuccess:
        raise ComputerUseError("target_occluded", "selected window rejected AXRaise")
    time.sleep(0.2)
    after = _select_window(activated_info, window_id=expected["window_id"])
    if not _same_window(expected, after):
        raise ComputerUseError(
            "target_drift", "selected window changed during recovery"
        )
    # Finder rebuilds its AX window/editor relationship asynchronously after
    # activation. Keep the recovery bounded, but allow slow hosts up to the
    # same three-second class as other local AX stabilization waits.
    focus_attempts = 30 if focus_exact_window else 1
    for attempt in range(focus_attempts):
        running = ax_driver._application_for_pid(int(expected_app["pid"]))
        if running is None:
            raise ComputerUseError(
                "target_drift", "selected app exited while focus settled"
            )
        settled_info = _resolved_app_info(running)
        for key in ("pid", "bundleId", "name", "processStartTime"):
            wanted = expected_app.get(key)
            if wanted is not None and settled_info.get(key) != wanted:
                raise ComputerUseError(
                    "target_drift", "selected app identity changed while focus settled"
                )
        current = _select_window(settled_info, window_id=expected["window_id"])
        if not _same_window(expected, current):
            raise ComputerUseError(
                "target_drift", "selected window changed while focus settled"
            )
        try:
            _validate_focused_window(
                snapshot, allow_exact_main_window=focus_exact_window
            )
            break
        except ComputerUseError as exc:
            focus_is_settling = "is not the focused AX window" in exc.message
            if not focus_is_settling or attempt + 1 == focus_attempts:
                raise
            time.sleep(0.1)
    return after


def validate_selected_window_focus(snapshot: dict) -> None:
    """Revalidate an exact observed window's identity and current AX focus."""

    _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot)


SAVE_MENU_MAX_NODES = 128
SAVE_MENU_MAX_DEPTH = 6
TEXTEDIT_SAVE_MAX_BYTES = 1_048_576
TEXTEDIT_VALUE_MAX_NODES = 256
TEXTEDIT_VALUE_MAX_DEPTH = 12


def _save_menu_candidate(snapshot: dict) -> tuple[object, tuple[str, ...], object]:
    """Resolve one exact native Save menu item for the selected document.

    The command metadata, rather than a localized title, distinguishes Save
    from Save As and Duplicate. The returned live reference never leaves this
    process and is re-resolved before dispatch.
    """

    _borrow_foreground(snapshot)
    _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot)
    app_info = snapshot["app"]
    app_element = _pid_app_element(app_info)
    focused = _focused_ax_window(app_info)
    document = ax_driver._get(focused, "AXDocument") if focused is not None else None
    if not isinstance(document, str) or not document.strip():
        raise ComputerUseError(
            "target_drift",
            "selected window has no stable existing document; refusing Save",
        )
    menu_bar = ax_driver._get(app_element, "AXMenuBar")
    if menu_bar is None:
        raise ComputerUseError("element_not_found", "app has no accessible menu bar")
    matches: list[tuple[object, tuple[str, ...]]] = []
    seen = 0

    def walk(element: object, path: tuple[str, ...], depth: int) -> None:
        nonlocal seen
        if depth > SAVE_MENU_MAX_DEPTH or seen >= SAVE_MENU_MAX_NODES:
            return
        seen += 1
        role = ax_driver._get(element, "AXRole")
        title_value = ax_driver._get(element, "AXTitle")
        title = title_value.strip()[:160] if isinstance(title_value, str) else ""
        next_path = (*path, title) if title else path
        command = ax_driver._get(element, "AXMenuItemCmdChar")
        modifiers = ax_driver._get(element, "AXMenuItemCmdModifiers")
        enabled = ax_driver._get(element, "AXEnabled")
        actions = ax_driver._action_names(element)
        if (
            role == "AXMenuItem"
            and isinstance(command, str)
            and command.casefold() == "s"
            and isinstance(modifiers, int)
            and not isinstance(modifiers, bool)
            and modifiers == 0
            and enabled is True
            and "AXPress" in actions
        ):
            identifier = ax_driver._get(element, "AXIdentifier")
            identity = (
                *next_path,
                command.casefold(),
                str(modifiers),
                identifier if isinstance(identifier, str) else "",
            )
            matches.append((element, identity))
        for child in ax_driver._as_list(ax_driver._get(element, "AXChildren")):
            walk(child, next_path, depth + 1)

    walk(menu_bar, (), 0)
    if len(matches) != 1:
        raise ComputerUseError(
            "element_not_found",
            "native Save command is unavailable or ambiguous; no input was sent",
        )
    element, identity = matches[0]
    return element, identity, document


def inspect_save_document(app: str, snapshot: dict) -> dict:
    """Return a private identity token for an approvable native Save action."""

    del app
    _, identity, document = _save_menu_candidate(snapshot)
    return {"save_identity": (document, *identity)}


def inspect_autosaving_document(app: str, snapshot: dict) -> dict:
    """Bind a TextEdit fill to the exact existing file it may autosave."""

    del app
    bundle = str(
        snapshot.get("app", {}).get("bundleId")
        or snapshot.get("app", {}).get("bundle_id")
        or ""
    ).casefold()
    if bundle != "com.apple.textedit":
        raise ComputerUseError(
            "invalid_argument", "autosaving document inspection requires TextEdit"
        )
    _borrow_foreground(snapshot)
    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window)
    focused = _focused_ax_window(snapshot["app"])
    document = ax_driver._get(focused, "AXDocument") if focused is not None else None
    if document is None or document == "":
        return {"autosave_identity": None}
    if not isinstance(document, str):
        raise ComputerUseError(
            "target_drift", "selected TextEdit document identity is invalid"
        )
    try:
        parsed = urlparse(document)
        path = Path(unquote(parsed.path))
    except (TypeError, ValueError) as exc:
        raise ComputerUseError(
            "target_drift", "selected TextEdit document identity is invalid"
        ) from exc
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not path.is_absolute()
    ):
        raise ComputerUseError(
            "target_drift", "selected TextEdit document is not a stable local file"
        )
    app_info = snapshot["app"]
    return {
        "autosave_identity": (
            document,
            int(app_info["pid"]),
            app_info.get("processStartTime"),
            str(snapshot.get("window_id") or ""),
        )
    }


def _unique_textedit_plain_text_value(
    snapshot: dict, focused_window: object
) -> str | None:
    bundle = str(
        snapshot.get("app", {}).get("bundleId")
        or snapshot.get("app", {}).get("bundle_id")
        or ""
    ).casefold()
    if bundle != "com.apple.textedit":
        return None
    values: list[str] = []
    visited = 0

    def walk(element: object, depth: int) -> None:
        nonlocal visited
        if depth > TEXTEDIT_VALUE_MAX_DEPTH or visited >= TEXTEDIT_VALUE_MAX_NODES:
            return
        visited += 1
        role = ax_driver._get(element, "AXRole")
        subrole = ax_driver._get(element, "AXSubrole")
        if role == "AXTextArea" and subrole != "AXSecureTextField":
            value = ax_driver._get(element, "AXValue")
            if isinstance(value, str):
                values.append(value)
        for child in ax_driver._as_list(ax_driver._get(element, "AXChildren")):
            walk(child, depth + 1)

    walk(focused_window, 0)
    if len(values) != 1:
        return None
    try:
        if len(values[0].encode("utf-8")) > TEXTEDIT_SAVE_MAX_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    return values[0]


def _verify_textedit_plain_text_save(
    snapshot: dict, document: object, focused_window: object
) -> bool:
    """Compare one exact TextEdit AX value with a stable local UTF-8 file.

    Contents never leave this function. Any URL, file-identity, encoding,
    size, or AX ambiguity returns unknown rather than a false verification.
    """

    value = _unique_textedit_plain_text_value(snapshot, focused_window)
    if value is None or not isinstance(document, str):
        return False
    try:
        parsed = urlparse(document)
        invalid_url = (
            parsed.scheme != "file"
            or parsed.netloc not in {"", "localhost"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        )
    except ValueError:
        return False
    if invalid_url:
        return False
    try:
        path = Path(unquote(parsed.path))
    except (TypeError, ValueError):
        return False
    if not path.is_absolute() or path.suffix.casefold() != ".txt":
        return False
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            return False
        if before.st_size > TEXTEDIT_SAVE_MAX_BYTES:
            return False
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            opened_identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            )
            if opened_identity != before_identity:
                return False
            chunks = []
            remaining = TEXTEDIT_SAVE_MAX_BYTES + 1
            while remaining > 0:
                chunk = os.read(fd, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after_fd = os.fstat(fd)
        finally:
            os.close(fd)
        after_path = path.lstat()
    except (OSError, ValueError):
        return False
    after_fd_identity = (
        after_fd.st_dev,
        after_fd.st_ino,
        after_fd.st_size,
        after_fd.st_mtime_ns,
    )
    after_path_identity = (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    )
    if (
        len(raw) > TEXTEDIT_SAVE_MAX_BYTES
        or stat.S_ISLNK(after_path.st_mode)
        or not stat.S_ISREG(after_path.st_mode)
        or after_fd_identity != opened_identity
        or after_path_identity != opened_identity
        or after_fd.st_size != len(raw)
    ):
        return False
    try:
        return raw.decode("utf-8") == value
    except UnicodeDecodeError:
        return False


def save_document(
    app: str, snapshot: dict, *, expected_identity: tuple[str, ...]
) -> dict:
    """Press an exact native Save item; never synthesize a shortcut or click."""

    live, identity, document = _save_menu_candidate(snapshot)
    if (document, *identity) != expected_identity:
        raise ComputerUseError(
            "target_drift", "native Save command changed after approval; re-observe"
        )
    focused = _focused_ax_window(snapshot["app"])
    edited_before = ax_driver._get(focused, "AXEdited") if focused is not None else None
    err = ax_driver.AXUIElementPerformAction(live, "AXPress")
    if err != ax_driver.kAXErrorSuccess:
        raise ComputerUseError("action_failed", "native Save command rejected AXPress")
    time.sleep(0.2)
    verified = None
    verification_source = "unverified"
    try:
        _, after_identity, after_document = _save_menu_candidate(snapshot)
        focused_after = _focused_ax_window(snapshot["app"])
        edited_after = (
            ax_driver._get(focused_after, "AXEdited")
            if focused_after is not None
            else None
        )
        same_binding = after_identity == identity and after_document == document
        if same_binding and edited_before is True and edited_after is False:
            verified = True
            verification_source = "ax_edited_same_document"
        elif same_binding and _verify_textedit_plain_text_save(
            snapshot, document, focused_after
        ):
            verified = True
            verification_source = "textedit_plain_text_exact_disk_match"
    except ComputerUseError:
        # AXPress was accepted already. A post-action focus/menu change cannot
        # retroactively become an execution rejection; report it as unverified.
        pass
    verification = (
        "same-document persistence was verified"
        if verified is True
        else "native Save AXPress was accepted; persistence could not be verified"
    )
    return _finish_action(
        app,
        snapshot,
        {
            "ok": True,
            "mode": "AXPress",
            "executed": True,
            "verification_source": verification_source,
        },
        verified=verified,
        verification=verification,
    )


# Roles whose click is itself a commit (pressing, toggling, following, opening
# a menu). Focusing one of these for a key press must not click it.
COMMIT_ON_CLICK_ROLES = {
    "AXButton",
    "AXCheckBox",
    "AXDisclosureTriangle",
    "AXLink",
    "AXMenuBarItem",
    "AXMenuButton",
    "AXMenuItem",
    "AXPopUpButton",
    "AXRadioButton",
    "AXSwitch",
}


_POPUP_MENU_LAYER = 101  # kCGPopUpMenuWindowLevel


def _open_popup_menus(snapshot: dict) -> set[int]:
    """CG ids of the snapshot app's on-screen popup-menu windows (best-effort).

    Empty unless background delivery serves a Chromium-family app, the only
    case :func:`_dismiss_lingering_popup` acts on.
    """
    if not _background_delivery(snapshot) or not _needs_web_content_retry(
        snapshot["app"]
    ):
        return set()
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListOptionOnScreenOnly,
        )

        pid = int(snapshot["app"]["pid"])
        return {
            int(window["kCGWindowNumber"])
            for window in CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, kCGNullWindowID
            )
            or []
            if int(window.get("kCGWindowOwnerPID", -1)) == pid
            and int(window.get("kCGWindowLayer", 0)) == _POPUP_MENU_LAYER
            and window.get("kCGWindowNumber") is not None
        }
    except Exception:  # noqa: BLE001 - best-effort probe
        return set()


def _dismiss_lingering_popup(snapshot: dict, before: set[int]) -> None:
    """Close a native popup menu left open after picking a web <select> option.

    Chrome applies the option on AXPress but keeps its popup menu up; while
    it is open the app's AXWindows lists only the menu, which breaks window
    binding on the next step. Escape keeps the chosen value. Only a popup
    that was already open when the option was picked (``before``) counts; a
    menu the press itself opened is left alone.
    """
    if not before:
        return
    try:
        pid = int(snapshot["app"]["pid"])

        def lingering() -> bool:
            return bool(_open_popup_menus(snapshot) & before)

        # Held from the settle wait to the menu closing: no other gesture can
        # interleave with the Escape or the menu's keyboard tracking.
        with background_input.GESTURE_LOCK:
            time.sleep(0.2)
            # Only while the user is in another app: an Escape to the pid of
            # the app they are using could close a menu they opened.
            if not lingering() or _live_front_pid() in (None, pid):
                return
            if not _synthesize(background_input.press_key, pid, KEY_ALIASES["escape"]):
                return
            deadline = time.monotonic() + 1.5
            while lingering() and time.monotonic() < deadline:
                time.sleep(0.05)
    except Exception:  # noqa: BLE001 - the pick itself already succeeded
        return


def _open_menu_count(pid: int) -> int | None:
    """Menu windows ``pid`` has up, on any Space; None when unreadable.

    An open menu runs a tracking loop that takes the keyboard system-wide:
    while one is up, the user's typing goes to it, even when it was opened
    in a window on another Space and is invisible (measured with Electron).
    On-screen-only window lists miss exactly that case.
    """
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListOptionAll,
        )

        return sum(
            1
            for window in CGWindowListCopyWindowInfo(
                kCGWindowListOptionAll, kCGNullWindowID
            )
            or []
            if int(window.get("kCGWindowOwnerPID", -1)) == pid
            and int(window.get("kCGWindowLayer", 0)) == _POPUP_MENU_LAYER
        )
    except Exception:  # noqa: BLE001 - callers treat unknown as unreadable
        return None


def _menu_opened_despite_error(snapshot: dict, before: int | None) -> bool:
    """Whether an Accessibility action that returned an error still opened a
    menu (an AX call can time out after the action took effect). Raises
    when open menus can no longer be counted, since one may be up."""
    if before is None:
        return False
    pid = int(snapshot["app"]["pid"])
    deadline = time.monotonic() + 0.3
    while True:
        count = _open_menu_count(pid)
        if count is not None and count > before:
            return True
        if time.monotonic() >= deadline:
            if count is None:
                raise ComputerUseError(
                    "action_failed",
                    "the Accessibility action failed and open menus could not "
                    "be counted" + _MENU_LEFT_OPEN,
                )
            return False
        time.sleep(0.05)


def _menus_before(
    snapshot: dict, menu_item: str | None, *, expect_menu: bool = False
) -> int | None:
    """Menu count before an action that may open one, or None to skip
    settling. ``menu_item`` needs the count (the choice is made from the
    menu the action opens), and so does a background action expected to open
    a menu (it must be closed again), so an unreadable count fails before
    acting."""
    background = _background_delivery(snapshot)
    if menu_item is None and not background:
        return None
    before = _open_menu_count(int(snapshot["app"]["pid"]))
    if before is None and (menu_item is not None or (background and expect_menu)):
        raise ComputerUseError(
            "action_failed",
            "open menus could not be counted; a menu this action opens "
            "could not be closed safely",
        )
    return before


def _menu_items_under(element: object | None) -> list[object]:
    """Menu items of the menu ``element`` opened (popup, menu button, context
    menu), searched a few levels down: AppKit hangs an AXMenu under the
    element, Chromium exposes the options under the popup itself."""
    found: list[object] = []
    stack = [(element, 0)] if element is not None else []
    while stack and len(found) < 200:
        node, depth = stack.pop()
        for child in ax_driver._as_list(ax_driver._get(node, "AXChildren")):
            role = ax_driver._get(child, "AXRole")
            if role == "AXMenuItem":
                found.append(child)
            elif role in ("AXMenu", "AXGroup", "AXList") and depth < 3:
                stack.append((child, depth + 1))
    return found


def _cancel_menu(perform: Any, menu_element: object) -> None:
    """AXCancel a menu; a failure only leaves the Escape fallback to close it."""
    try:
        perform(menu_element, "AXCancel")
    except Exception:  # noqa: BLE001 - the caller escapes and confirms closure
        pass


def _close_menus(pid: int, before: int, element: object | None) -> bool:
    """Close menus that appeared since ``before``; True once they are gone."""
    if element is not None:
        from ApplicationServices import AXUIElementPerformAction

        for child in ax_driver._as_list(ax_driver._get(element, "AXChildren")):
            if ax_driver._get(child, "AXRole") == "AXMenu":
                _cancel_menu(AXUIElementPerformAction, child)
    deadline = time.monotonic() + 1.5
    escaped = 0
    while True:
        count = _open_menu_count(pid)
        if count is None:
            return False  # cannot confirm the menu closed
        if count <= before:
            return True
        if time.monotonic() > deadline:
            return False
        if escaped < 2 and time.monotonic() > deadline - 1.2 + 0.4 * escaped:
            # Escape goes to the menu's tracking loop and keeps a value
            # already chosen (Chrome keeps its <select> popup up after a pick).
            _synthesize(background_input.press_key, pid, KEY_ALIASES["escape"])
            escaped += 1
        time.sleep(0.05)


class _NoMenuOpenedError(ComputerUseError):
    """The action asked to choose ``choose`` but opened no menu."""

    def __init__(self, snapshot: dict, element_index: int | None, choose: str):
        super().__init__(
            "element_not_found", f"the click opened no menu to choose {choose!r} from"
        )
        self.snapshot = snapshot
        self.element_index = element_index


def _settle_menus(
    snapshot: dict,
    element_index: int | None,
    before: int | None,
    choose: str | None,
    *,
    expect_menu: bool,
) -> dict | None:
    """Leave no menu open after a background action.

    A menu the action opened is read (its item titles are returned so the
    next step can name one), the item ``choose`` is pressed when asked, and
    every new menu is closed before returning -- an open menu would take
    the user's keystrokes.
    """
    if before is None:
        return None
    pid = int(snapshot["app"]["pid"])
    deadline = time.monotonic() + (0.6 if expect_menu or choose else 0.0)
    count = _open_menu_count(pid)
    while (count is None or count <= before) and time.monotonic() < deadline:
        time.sleep(0.05)
        count = _open_menu_count(pid)
    if count is None:
        # Unknown is not "no menu": one may be up and taking the keys, and
        # Escape cannot be sent blindly (it would reach the content).
        if choose is not None:
            raise ComputerUseError(
                "action_failed",
                f"open menus could not be counted after the action, so "
                f"{choose!r} was not chosen{_MENU_LEFT_OPEN}",
            )
        return {
            "opened": None,
            "closed": False,
            "warning": "open menus could not be counted after the action"
            + _MENU_LEFT_OPEN,
        }
    if count <= before:
        if choose is not None:
            raise _NoMenuOpenedError(snapshot, element_index, choose)
        if not expect_menu:
            return None
        return {
            "opened": False,
            "note": (
                "no menu appeared (Chromium shows no popup for a window on "
                "another Space); pass menu_item=<option> to choose one"
            ),
        }
    try:
        return _read_choose_and_close(snapshot, element_index, before, choose)
    except BaseException as exc:
        # Whatever went wrong (drift, an AX error), the menu must not stay
        # open; error paths that already closed it make this a cheap recount.
        try:
            closed = _close_menus(pid, before, None)
        except Exception:  # noqa: BLE001 - the original error is the one to report
            closed = False
        if closed or not isinstance(exc, Exception):
            raise
        if isinstance(exc, ComputerUseError):
            if _MENU_LEFT_OPEN not in exc.message:
                exc.message += _MENU_LEFT_OPEN
                exc.args = (exc.message,)
            raise
        raise ComputerUseError(
            "action_failed", f"reading the menu failed: {exc}{_MENU_LEFT_OPEN}"
        ) from exc


def _read_choose_and_close(
    snapshot: dict, element_index: int | None, before: int, choose: str | None
) -> dict:
    pid = int(snapshot["app"]["pid"])
    element = (
        _live_element(snapshot, element_index, validate_point=False)
        if element_index is not None
        else None
    )
    items = _menu_items_under(element)
    titles = [_menu_item_title(i) for i in items]
    chosen = None
    if choose is not None:
        wanted = _menu_title_key(choose)
        match = next(
            (i for i, t in zip(items, titles) if _menu_title_key(t) == wanted), None
        )
        if match is None or ax_driver._get(match, "AXEnabled") is False:
            closed = _close_menus(pid, before, element)
            shown = ", ".join(t for t in titles if t)[:400]
            raise ComputerUseError(
                "element_not_found",
                f"menu has no enabled item {choose!r} (items: {shown})"
                + ("" if closed else _MENU_LEFT_OPEN),
            )
        from ApplicationServices import AXUIElementPerformAction

        if AXUIElementPerformAction(match, "AXPress") != 0:
            closed = _close_menus(pid, before, element)
            raise ComputerUseError(
                "accessibility_error",
                f"menu item {choose!r} could not be pressed"
                + ("" if closed else _MENU_LEFT_OPEN),
            )
        chosen = _menu_item_title(match)
        time.sleep(0.15)
    closed = _close_menus(pid, before, element)
    report: dict[str, Any] = {"closed": closed}
    if chosen is not None:
        report["chosen"] = chosen
    else:
        report["items"] = [t for t in titles if t][:60]
        report["note"] = (
            "the menu was read and closed (an open menu would take the user's "
            "keys); repeat the click with menu_item=<title> to choose"
        )
    if not closed:
        report["warning"] = "a menu stayed open; the user's typing may go to it"
    return report


def _focus_without_commit(snapshot: dict, live: object | None) -> str | None:
    """Give ``live`` keyboard focus via AXFocused; the mode name, or None."""
    if live is None:
        return None
    focused = _focused_ax_element(snapshot["app"])
    if focused is not None and live == focused:
        return "AXFocusVerified"
    err = ax_driver.AXUIElementSetAttributeValue(live, "AXFocused", True)
    if err != ax_driver.kAXErrorSuccess:
        return None
    # Chromium applies focus asynchronously; poll briefly.
    deadline = time.monotonic() + 0.3
    while True:
        time.sleep(0.05)
        focused = _focused_ax_element(snapshot["app"])
        if focused is not None and live == focused:
            return "AXFocused"
        if focused is None and ax_driver._get(live, "AXFocused") is True:
            # Electron answers no app-level AXFocusedUIElement; the element's
            # own focus state is the remaining exact evidence.
            return "AXFocused"
        if time.monotonic() > deadline:
            return None


def click(
    app: str,
    element_index: int | None = None,
    x: int | None = None,
    y: int | None = None,
    click_count: int = 1,
    mouse_button: str = "left",
    expected_snapshot: dict | None = None,
    window_id: int | str | None = None,
    include_post_state: bool = False,
    focus_only: bool = False,
    modifiers: list[str] | str | None = None,
    menu_item: str | None = None,
) -> dict:
    """Click an element or point; ``menu_item`` chooses from the menu it opens.

    In background delivery no menu is ever left open: one the click opened
    is read and closed (its items are returned), or with ``menu_item`` the
    item is pressed in the same call.
    """
    try:
        return _click(
            app,
            element_index,
            x,
            y,
            click_count,
            mouse_button,
            expected_snapshot,
            window_id,
            include_post_state,
            focus_only,
            modifiers,
            menu_item,
        )
    except _NoMenuOpenedError as exc:
        # A Chromium popup on another Space opens no menu, but its option is
        # still settable through Accessibility -- on the very element the
        # click resolved (same snapshot, same index).
        snapshot = exc.snapshot
        entry = next(
            (
                e
                for e in snapshot.get("elements", [])
                if e.get("index") == exc.element_index
            ),
            None,
        )
        if (
            menu_item is None
            or exc.element_index is None
            or mouse_button != "left"
            or entry is None
            or entry.get("role") != "AXPopUpButton"
        ):
            raise
        result = set_value(
            app,
            exc.element_index,
            menu_item,
            expected_snapshot=snapshot,
            window_id=window_id,
            include_post_state=include_post_state,
        )
    result["warning"] = "; ".join(
        w
        for w in (
            result.get("warning"),
            "the popup opened no menu; its value was set instead",
        )
        if w
    )
    return result


def _click(
    app,
    element_index,
    x,
    y,
    click_count,
    mouse_button,
    expected_snapshot,
    window_id,
    include_post_state,
    focus_only,
    modifiers,
    menu_item,
) -> dict:
    if mouse_button not in MOUSE_BUTTONS:
        raise ComputerUseError(
            "invalid_argument", f"unsupported mouse button {mouse_button!r}"
        )
    if isinstance(click_count, bool) or click_count not in (1, 2, 3):
        raise ComputerUseError("invalid_argument", "click_count must be 1, 2 or 3")
    click_count = int(click_count)
    flags = _modifier_flags(modifiers)
    if flags and focus_only:
        raise ComputerUseError("invalid_argument", "focus_only takes no modifiers")
    if menu_item is not None and focus_only:
        raise ComputerUseError("invalid_argument", "focus_only takes no menu_item")
    if element_index is not None:
        snapshot = expected_snapshot or get_app_state(
            app,
            screenshot=False,
            use_cache=False,
            window_id=window_id,
            activate=OBSERVE_BY_ROUTE,
        )
        entry = _element(snapshot, element_index)
        opens_menu = mouse_button == "right" or entry.get("role") in (
            "AXPopUpButton",
            "AXMenuButton",
        )
        menus_before = (
            None
            if focus_only
            else _menus_before(snapshot, menu_item, expect_menu=opens_menu)
        )
        is_transient = entry.get(
            "source_window_id", snapshot.get("window_id")
        ) != snapshot.get("window_id")
        center = entry["center"]
        actions = entry["actions"]
        # The semantic equivalent of this gesture, when the element advertises
        # one: AX actions work on occluded/background windows and never touch
        # the cursor. Only advertised actions are attempted -- an unadvertised
        # action fails with kAXErrorActionUnsupported or silently no-ops.
        semantic = None
        if focus_only:
            # Focusing must never commit: an AXPress here would press a button
            # under a plan that only asked for a key (bypassing click consent).
            pass
        elif flags:
            # Accessibility actions carry no modifiers; a Shift/Cmd click is
            # only expressible as a pixel click.
            pass
        elif mouse_button == "left" and click_count == 1 and "AXPress" in actions:
            semantic = "AXPress"
        elif mouse_button == "left" and click_count == 2 and "AXOpen" in actions:
            semantic = "AXOpen"
        elif mouse_button == "right" and click_count == 1 and "AXShowMenu" in actions:
            semantic = "AXShowMenu"
        live = None
        if semantic is not None or is_transient or focus_only:
            # Menus and popovers can be owned by the selected app/window while
            # appearing outside the window's content bounds. Revalidate the
            # exact window and AX target identity, then prefer the semantic
            # action on that live object. Coordinate fallbacks remain bounded.
            live = _live_element(snapshot, element_index, validate_point=False)
        if live is not None and semantic is not None:
            import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
            from ApplicationServices import AXUIElementPerformAction

            popups_before = (
                _open_popup_menus(snapshot)
                if entry.get("role") == "AXMenuItem"
                else set()
            )
            with _guard_user_focus(snapshot) as guard:
                err = AXUIElementPerformAction(live, semantic)
                # An error can arrive after the action took effect; a menu
                # it opened must still be settled (and is not clicked again).
                accepted = err == AS.kAXErrorSuccess or _menu_opened_despite_error(
                    snapshot, menus_before
                )
                menu = (
                    _settle_menus(
                        snapshot,
                        element_index,
                        menus_before,
                        menu_item,
                        expect_menu=opens_menu,
                    )
                    if accepted
                    else None
                )
            if accepted:
                if popups_before:
                    _dismiss_lingering_popup(snapshot, popups_before)
                delivery = {"mode": semantic, "element_index": element_index}
                if err != AS.kAXErrorSuccess:
                    delivery["ax_error"] = err
                if menu is not None:
                    delivery["menu"] = menu
                delivery.update(_focus_fields(guard))
                return _finish_action(
                    app,
                    snapshot,
                    delivery,
                    verified=None,
                    verification="action accepted by Accessibility; outcome not asserted",
                    include_post_state=include_post_state,
                )
        if (
            live is not None
            and focus_only
            and entry.get("role") in FILL_ROLES
            and live == _focused_ax_element(snapshot["app"])
        ):
            expected_window = snapshot.get("transient_window") if is_transient else None
            if _background_delivery(snapshot):
                _validate_focused_window(
                    snapshot, expected_window, require_active_app=False
                )
            else:
                _validate_focused_window(snapshot, expected_window)
            return _finish_action(
                app,
                snapshot,
                {"mode": "AXFocusVerified", "element_index": element_index},
                verified=True,
                verification="exact Accessibility element remained focused",
                include_post_state=include_post_state,
            )
        if focus_only:
            focused = _focus_without_commit(snapshot, live)
            keyed: dict = {}
            if (
                focused is None
                and live is not None
                and not is_transient
                and _background_delivery(snapshot)
            ):
                # AXFocused only takes in the app's key window; make the
                # target key for the focus change. The window keeps its first
                # responder after the user's window gets focus back. (Never
                # for a transient: making its anchor key would dismiss it.)
                with _keyed_target(snapshot) as keyed:
                    focused = _focus_without_commit(snapshot, live)
            if focused is not None:
                return _finish_action(
                    app,
                    snapshot,
                    {
                        "mode": focused,
                        "element_index": element_index,
                        **_focus_fields(keyed),
                    },
                    verified=True,
                    verification="exact Accessibility element holds keyboard focus",
                    include_post_state=include_post_state,
                )
            # A pixel click can never be proven non-committing (a web input
            # may submit on click and still read as a plain AXTextField), so
            # focus without AXFocused is the planner's consent-gated click.
            raise ComputerUseError(
                "synthetic_input_blocked",
                "cannot focus this control without clicking it; "
                "plan a click (consent-gated) instead",
            )
        if is_transient:
            raise ComputerUseError(
                "synthetic_input_blocked",
                "transient companion target is not exactly pressable or focused",
            )
        # The lock spans the click and the menu it opens, so no other
        # gesture lands between opening the menu and closing it.
        with background_input.GESTURE_LOCK:
            delivery = _pixel_click(
                snapshot,
                float(center[0]),
                float(center[1]),
                button=mouse_button,
                count=click_count,
                flags=flags,
            )
            delivery.update({"element_index": element_index, "at": center})
            menu = _settle_menus(
                snapshot,
                element_index,
                menus_before,
                menu_item,
                expect_menu=opens_menu,
            )
        if menu is not None:
            delivery["menu"] = menu
        return _finish_action(
            app,
            snapshot,
            delivery,
            verified=None,
            verification="synthetic click emitted; outcome not asserted",
            include_post_state=include_post_state,
        )
    if focus_only:
        # Focus without commit needs an exact AX element; a bare point can
        # only be clicked.
        raise ComputerUseError(
            "invalid_argument", "focus_only requires an element index, not x/y"
        )
    if x is None or y is None:
        raise ComputerUseError(
            "invalid_argument", "click requires --element-index or both --x and --y"
        )
    if menu_item is not None:
        # The menu's items are read under the clicked element; a bare point
        # has none, so the choice could never be made.
        raise ComputerUseError(
            "invalid_argument", "menu_item requires an element index, not x/y"
        )
    snapshot = expected_snapshot or get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    menus_before = _menus_before(snapshot, None, expect_menu=mouse_button == "right")
    with background_input.GESTURE_LOCK:
        delivery = _pixel_click(
            snapshot,
            float(x),
            float(y),
            button=mouse_button,
            count=click_count,
            flags=flags,
        )
        delivery["at"] = [x, y]
        menu = _settle_menus(
            snapshot,
            None,
            menus_before,
            menu_item,
            expect_menu=mouse_button == "right",
        )
    if menu is not None:
        delivery["menu"] = menu
    return _finish_action(
        app,
        snapshot,
        delivery,
        verified=None,
        verification="synthetic click emitted; outcome not asserted",
        include_post_state=include_post_state,
    )


def set_value(
    app: str,
    element_index: int,
    value: str,
    expected_snapshot: dict | None = None,
    window_id: int | str | None = None,
    include_post_state: bool = False,
) -> dict:
    """Write a value into a settable element; verify by reading it back.

    Falls back to synthetic typing (click, Cmd+A, delete, CGEvent unicode) when
    the element rejects direct AX writes. The returned verification field tells
    the agent whether the value landed exactly.
    """
    snapshot = expected_snapshot or get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    entry = _element(snapshot, element_index)
    is_transient = entry.get(
        "source_window_id", snapshot.get("window_id")
    ) != snapshot.get("window_id")
    live = _live_element(snapshot, element_index, validate_point=not is_transient)
    if live is not None:
        is_finder_item = (
            is_finder_snapshot(snapshot)
            and entry.get("role") == "AXTextField"
            and entry.get("parent_role") == "AXCell"
        )
        finder_file_reference = (
            _finder_file_reference_for_editor(live, snapshot)
            if is_finder_item
            else None
        )
        from ApplicationServices import (  # type: ignore[import-untyped]
            AXUIElementSetAttributeValue,
            kAXValueAttribute,
        )

        if entry.get("role") == "AXPopUpButton" and not _needs_web_content_retry(
            snapshot["app"]
        ):
            # A native popup's value is not settable; choosing works through
            # its own menu, which AppKit opens and accepts in the background.
            with _guard_user_focus(snapshot) as guard:
                readback = _choose_from_ax_menu(
                    live, value, int(snapshot["app"]["pid"])
                )
            delivery = {
                "mode": "AXMenuChoose",
                "element_index": element_index,
                "actual": readback,
                **_focus_fields(guard),
            }
            return _finish_action(
                app,
                snapshot,
                delivery,
                verified=_menu_title_key(readback) == _menu_title_key(value),
                verification="popup value read back after choosing the menu item",
                include_post_state=include_post_state,
            )
        # Only a control whose value is a number (a stepper, a slider) is
        # read first; a text field's value is text, and a secure field's
        # contents are never read.
        numeric = (
            None
            if entry.get("role") in ax_driver.EDITABLE_ROLES
            or entry.get("role") == "AXSecureTextField"
            or entry.get("subrole") == "AXSecureTextField"
            else _numeric_request(ax_driver._get(live, kAXValueAttribute), value)
        )
        err = AXUIElementSetAttributeValue(
            live, kAXValueAttribute, value if numeric is None else numeric
        )
        if err == 0:
            readback = _await_readback(live, value, numeric)
            if readback == value:
                if is_finder_item:
                    actual_path = _finder_file_reference_path(finder_file_reference)
                    requested_basename = value.replace("\u200b", "").replace(
                        "\ufeff", ""
                    )
                    if (
                        actual_path is not None
                        and Path(actual_path).name == requested_basename
                    ):
                        _clear_finder_rename_binding(snapshot)
                        return _finish_action(
                            app,
                            snapshot,
                            {
                                "mode": "AXSetValue",
                                "element_index": element_index,
                                "verification_source": "finder_file_reference_basename",
                                "actual_basename": Path(actual_path).name,
                            },
                            verified=True,
                            verification="Finder file-reference URL resolved to the requested basename",
                            include_post_state=include_post_state,
                        )
                return _finish_action(
                    app,
                    snapshot,
                    {"mode": "AXSetValue", "element_index": element_index},
                    verified=True,
                    verification="exact AX value readback matched requested text",
                    include_post_state=include_post_state,
                )
    if is_transient:
        raise ComputerUseError(
            "synthetic_input_blocked",
            "transient companion rejected exact AXSetValue; synthetic typing is disabled",
            ("Re-observe the control or use a safe exact Accessibility action.",),
        )
    # AX write either failed or did not land; degrade to synthetic typing.
    result = _synthetic_fill(snapshot, element_index, value)
    verified = result.pop("verified", None)
    return _finish_action(
        app,
        snapshot,
        result,
        verified=verified,
        verification=(
            "exact AX value readback matched requested text"
            if verified is True
            else "synthetic typing emitted; exact readback unavailable"
        ),
        include_post_state=include_post_state,
    )


def _background_fill(snapshot: dict, element_index: int, value: str) -> dict:
    """Replace a field's text without activating the app (cua's recipe).

    AXFocused focuses the field, AXSelectedTextRange selects its whole text,
    and the value arrives as SkyLight unicode events to the pid, replacing the
    selection. No global HID click, no Cmd+A, no activation. When the window
    is not key (e.g. on another Space) AXFocused is ignored, so the window is
    first made key with focus-without-raise and focus is handed back after.
    """
    live = _live_element(snapshot, element_index, validate_point=False)
    if live is None:
        raise ComputerUseError(
            "synthetic_input_blocked", "field could not be focused in the background"
        )
    typer = (
        _choose_by_typeahead
        if _element(snapshot, element_index).get("role") == "AXPopUpButton"
        else _type_into_focused
    )
    with _keyed_target(snapshot) as state:
        if _focus_without_commit(snapshot, live) is None:
            raise ComputerUseError(
                "synthetic_input_blocked",
                "field could not be focused in the background",
            )
        result = typer(snapshot, live, element_index, value)
    result.update(_focus_fields(state))
    return result


def _type_into_focused(
    snapshot: dict, live: object, element_index: int, value: str
) -> dict:
    services = ax_driver.AS
    current = _read_value(live)
    if current is not None:
        length = len(current.encode("utf-16-le")) // 2
    else:
        # An unreadable value is not an empty field: without its length the
        # replace could append to existing text.
        count = ax_driver._get(live, "AXNumberOfCharacters")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ComputerUseError(
                "synthetic_input_blocked",
                "field text could not be read to replace it",
            )
        length = count
    selection = services.AXValueCreate(services.kAXValueCFRangeType, (0, length))
    if length and (
        ax_driver.AXUIElementSetAttributeValue(live, "AXSelectedTextRange", selection)
        != ax_driver.kAXErrorSuccess
    ):
        raise ComputerUseError(
            "synthetic_input_blocked", "field text could not be selected for replace"
        )
    _validate_focused_window(
        snapshot, require_active_app=False, require_exact_window_id=True
    )
    pid, _ = _target_ids(snapshot)
    if value:
        sent = _synthesize(background_input.type_text, pid, value)
    else:
        # Typing nothing would leave the selection in place: clearing the
        # field takes an explicit delete of the selected text.
        sent = not length or _synthesize(
            background_input.press_key, pid, KEY_ALIASES["delete"]
        )
    if not sent:
        raise ComputerUseError(
            "action_failed", "background text could not be synthesized"
        )
    # Never re-type on a short readback: off-Space, the AX value can trail the
    # page (the page already holds the whole text), so a "missing" suffix is a
    # perception lag, not a drop. Wait for the value to settle instead.
    deadline = time.monotonic() + 3.0
    while True:
        time.sleep(0.15)
        readback = _read_value(live)
        if readback == value or time.monotonic() > deadline:
            break
    return {
        "mode": "SkyLight-fill",
        "element_index": element_index,
        "verified": True if readback == value else None,
        "actual": readback,
    }


def _open_menu_of(live: object, timeout: float = 1.0) -> object | None:
    deadline = time.monotonic() + timeout
    while True:
        for child in ax_driver._as_list(ax_driver._get(live, "AXChildren")):
            if ax_driver._get(child, "AXRole") == "AXMenu":
                return child
        if time.monotonic() > deadline:
            return None
        time.sleep(0.05)


def _close_menu(live: object, menu_element: object, pid: int) -> bool:
    """Close a popup's menu; a menu left open tracks the keyboard system-wide.

    Returns whether the menu is confirmed gone.
    """
    from ApplicationServices import AXUIElementPerformAction

    _cancel_menu(AXUIElementPerformAction, menu_element)
    escapes = 0
    deadline = time.monotonic() + 0.5
    while _open_menu_of(live, timeout=0) is not None:
        if time.monotonic() > deadline:
            if escapes >= 2:
                return False
            # Escape goes to the menu's tracking loop in the owning process.
            _synthesize(background_input.press_key, pid, KEY_ALIASES["escape"], 0)
            escapes += 1
            deadline = time.monotonic() + 0.5
        time.sleep(0.05)
    return True


_MENU_LEFT_OPEN = "; a menu may still be open and take the user's typing"


def _choose_from_ax_menu(live: object, value: str, pid: int) -> str | None:
    """Open a native popup through Accessibility and press the item ``value``.

    The closed popup exposes no items, so it is opened with AXPress (AppKit
    shows the menu without activating the app), the item is matched by title
    and pressed, and the popup's value is read back. An unknown title cancels
    the menu and reports the available ones.
    """
    from ApplicationServices import AXUIElementPerformAction

    # A menu that was just closed is still torn down for a moment, and a
    # press landing then is swallowed (measured on back-to-back choices);
    # wait for it to go before opening a fresh one.
    deadline = time.monotonic() + 0.75
    while _open_menu_of(live, timeout=0) is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    lingering = _open_menu_of(live, timeout=0)
    if lingering is not None and not _close_menu(live, lingering, pid):
        raise ComputerUseError(
            "action_failed",
            f"the popup's previous menu did not close{_MENU_LEFT_OPEN}",
        )
    time.sleep(0.15)
    if AXUIElementPerformAction(live, "AXPress") != 0:
        # The press may have opened the menu before the AX call failed.
        opened = _open_menu_of(live, timeout=0.3)
        closed = opened is None or _close_menu(live, opened, pid)
        raise ComputerUseError(
            "accessibility_error",
            "popup could not be opened" + ("" if closed else _MENU_LEFT_OPEN),
        )
    menu_element = _open_menu_of(live, timeout=1.5)
    if menu_element is None:
        # A menu arriving just after the timeout would track the keyboard;
        # look once more and close it before giving up.
        late = _open_menu_of(live, timeout=0.3)
        closed = late is None or _close_menu(live, late, pid)
        raise ComputerUseError(
            "action_failed",
            "popup opened no menu" + ("" if closed else _MENU_LEFT_OPEN),
        )
    items = [
        item
        for item in ax_driver._as_list(ax_driver._get(menu_element, "AXChildren"))
        if ax_driver._get(item, "AXRole") == "AXMenuItem"
    ]
    wanted = _menu_title_key(value)
    match = next(
        (i for i in items if _menu_title_key(_menu_item_title(i)) == wanted), None
    )
    if match is None or ax_driver._get(match, "AXEnabled") is False:
        closed = _close_menu(live, menu_element, pid)
        titles = ", ".join(t for t in (_menu_item_title(i) for i in items) if t)[:400]
        raise ComputerUseError(
            "value_not_settable",
            f"popup has no enabled option {value!r} (options: {titles})"
            + ("" if closed else _MENU_LEFT_OPEN),
        )
    if AXUIElementPerformAction(match, "AXPress") != 0:
        closed = _close_menu(live, menu_element, pid)
        raise ComputerUseError(
            "accessibility_error",
            f"option {value!r} could not be pressed"
            + ("" if closed else _MENU_LEFT_OPEN),
        )
    deadline = time.monotonic() + 1.0
    while True:
        readback = _read_value(live)
        if _menu_title_key(readback) == wanted or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    # Pressing an item normally dismisses the menu; never leave one open.
    still_open = _open_menu_of(live, timeout=0)
    if still_open is not None and not _close_menu(live, still_open, pid):
        raise ComputerUseError(
            "action_failed",
            f"chose {value!r} but the popup's menu stayed open{_MENU_LEFT_OPEN}",
        )
    return readback


def _choose_by_typeahead(
    snapshot: dict, live: object, element_index: int, value: str
) -> dict:
    """Pick a popup option by typing its label into the focused, closed popup.

    Opening the menu needs the window on screen (Chrome renders no menu for a
    window on another Space) and leaves a menu tracking the keyboard; typeahead
    on the closed control changes the selection directly and fires ``change``.
    """
    _validate_focused_window(
        snapshot, require_active_app=False, require_exact_window_id=True
    )
    pid, _ = _target_ids(snapshot)
    if not _synthesize(background_input.type_text, pid, value):
        raise ComputerUseError(
            "action_failed", "background text could not be synthesized"
        )
    deadline = time.monotonic() + 2.0
    while True:
        time.sleep(0.15)
        readback = _read_value(live)
        if readback == value or time.monotonic() > deadline:
            break
    return {
        "mode": "SkyLight-typeahead",
        "element_index": element_index,
        "verified": True if readback == value else None,
        "actual": readback,
    }


def _synthetic_fill(snapshot: dict, element_index: int, value: str) -> dict:
    if _keyboard_background(snapshot):
        return _background_fill(snapshot, element_index, value)
    entry = _element(snapshot, element_index)
    center = entry["center"]
    # Cmd+A is a menu key equivalent: this fallback is foreground-only.
    _borrow_foreground(snapshot)
    _validate_snapshot_window(snapshot, point=(float(center[0]), float(center[1])))
    ax_driver._cg_click(float(center[0]), float(center[1]))
    time.sleep(0.3)
    _validate_focused_window(snapshot)
    ax_driver._press_key(ax_driver._keycode_for("a"), modifiers=ax_driver.FLAG_COMMAND)
    time.sleep(0.1)
    ax_driver._press_key(KEY_ALIASES["delete"])
    ax_driver._type_text(value)
    time.sleep(0.4)
    # Verify by VALUE, not index: typing can open suggestion dropdowns and the
    # tree rebuilds with shifted indexes (a runtime element identity would
    # remove this class of staleness; snapshots are refreshed instead).
    try:
        current_window = _select_window(
            snapshot["app"], window_id=snapshot["window_id"]
        )
        fresh = _collect_with_timeout(
            snapshot["app"]["name"],
            window_index=current_window["index"],
            window=current_window,
            expected_pid=int(snapshot["app"]["pid"]),
            retry_web_content=_needs_web_content_retry(snapshot["app"]),
        )
        for target in fresh:
            if target.get("element") is None or target["role"] not in FILL_ROLES:
                continue
            readback = _read_value(target["element"])
            if readback == value:
                return {
                    "mode": "synthetic-typing",
                    "element_index": element_index,
                    "verified": True,
                    "actual": readback,
                }
    except Exception:  # noqa: BLE001 - verification is best-effort
        pass
    return {
        "mode": "synthetic-typing",
        "element_index": element_index,
        "verified": None,
        "recovery": "could not re-read the element to verify; re-run get-app-state",
    }


def _prepare_synthetic_action(
    app: str,
    window_id: int | str | None,
    expected_snapshot: dict | None = None,
    element_index: int | None = None,
    *,
    allow_focused_editable_enter: bool = False,
    allow_finder_main_window: bool = False,
    allow_finder_escape: bool = False,
    modifiers: int = 0,
) -> dict:
    """Validate the keyboard target before a synthetic key/text dispatch.

    Keys always land on the app's key window, so the exact window must be the
    focused AX window either way. Only foreground (global HID) delivery also
    needs the app frontmost and the window uncovered; callers pick the route
    with the same ``_keyboard_background(snapshot, modifiers)`` decision.
    """
    snapshot = expected_snapshot or get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    background = _keyboard_background(snapshot, modifiers)
    # Foreground keeps the historical (HID) validation call shapes exactly;
    # background relaxes only the frontmost/topmost requirements.
    window_kwargs = {"require_topmost": False} if background else {}
    is_finder_transient = False
    is_transient = False
    if element_index is not None:
        entry = _element(snapshot, element_index)
        if entry.get("source_window_id", snapshot.get("window_id")) != snapshot.get(
            "window_id"
        ):
            live = _live_element(snapshot, element_index, validate_point=False)
            focused_element = _focused_ax_element(snapshot["app"])
            if live is None or not (
                ax_driver._get(live, "AXFocused") is True or live == focused_element
            ):
                raise ComputerUseError(
                    "target_drift",
                    "transient target changed or lost focus before key dispatch",
                )
            anchor = _validate_snapshot_window(snapshot)
            transient = snapshot.get("transient_window")
            if not isinstance(transient, dict):
                raise ComputerUseError(
                    "target_drift", "transient window identity is missing"
                )
            current = _focused_transient_window(
                snapshot["app"], anchor, trusted_window_id=entry["source_window_id"]
            )
            if current is None or not _same_window(transient, current):
                raise ComputerUseError(
                    "target_drift", "transient companion changed or lost focus"
                )
            expected_window = current
            is_transient = True
            is_finder_transient = is_finder_snapshot(snapshot)
        else:
            entry_is_focused_editable = (
                allow_focused_editable_enter
                and entry.get("role") in FILL_ROLES
                and (
                    live := _live_element(snapshot, element_index, validate_point=False)
                )
                is not None
                and live == _focused_ax_element(snapshot["app"])
            )
            if entry_is_focused_editable:
                expected_window = _validate_snapshot_window(snapshot)
                center = entry.get("center")
                if (
                    not isinstance(center, (list, tuple))
                    or len(center) != 2
                    or not all(isinstance(value, (int, float)) for value in center)
                ):
                    raise ComputerUseError(
                        "target_drift", "focused editable target has invalid bounds"
                    )
                topmost_id = _topmost_window_id_at(float(center[0]), float(center[1]))
                same_app_ids = {
                    _cg_window_id(record["window_id"])
                    for record in _window_records(snapshot["app"])
                }
                if not background and topmost_id not in same_app_ids:
                    raise ComputerUseError(
                        "target_occluded",
                        "selected window is covered by another process at keyboard dispatch",
                    )
            else:
                expected_window = _validate_snapshot_window(
                    snapshot, point=_window_center(snapshot), **window_kwargs
                )
    else:
        expected_window = _validate_snapshot_window(
            snapshot, point=_window_center(snapshot), **window_kwargs
        )
    if allow_finder_main_window:
        _validate_focused_window(
            snapshot,
            snapshot["window"],
            allow_exact_main_window=True,
        )
    elif allow_finder_escape:
        if not is_finder_snapshot(snapshot):
            raise ComputerUseError(
                "target_drift", "Finder editor recovery requires a Finder snapshot"
            )
        _validate_focused_window(
            snapshot,
            snapshot["window"],
            allow_exact_main_window=True,
        )
    elif is_finder_transient:
        _validate_focused_window(
            snapshot,
            snapshot["window"],
            allow_exact_main_window=True,
        )
    elif not background:
        _validate_focused_window(snapshot, expected_window)
    elif is_transient:
        # A transient companion holds the keyboard: validate it exactly now
        # and have _keyed_target re-check it (without moving key status).
        _validate_focused_window(
            snapshot,
            expected_window,
            require_active_app=False,
            require_exact_window_id=True,
        )
        return {**snapshot, _KEYBOARD_WINDOW: expected_window}
    # Background keyboard dispatch to the snapshot window validates focus
    # inside _keyed_target, after the target window has been made key.
    return snapshot


def type_text(
    app: str,
    text: str,
    window_id: int | str | None = None,
    include_post_state: bool = False,
) -> dict:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ComputerUseError(
            "invalid_argument", "text contains an unpaired surrogate"
        ) from exc
    if len(text) > background_input.MAX_TYPE_TEXT_CHARS:
        raise ComputerUseError(
            "invalid_argument",
            f"text longer than {background_input.MAX_TYPE_TEXT_CHARS} characters; "
            "set the field's value instead of typing it",
        )
    snapshot = _prepare_synthetic_action(app, window_id)
    focus: dict[str, Any] = {}
    if _keyboard_background(snapshot):
        with _keyed_target(snapshot) as state:
            if not _synthesize(
                background_input.type_text, _target_ids(snapshot)[0], text
            ):
                raise ComputerUseError(
                    "action_failed", "background text could not be synthesized"
                )
        focus = _focus_fields(state)
        mode = "SkyLight-unicode"
    else:
        ax_driver._type_text(text)
        mode = "CGEvent-unicode"
    return _finish_action(
        app,
        snapshot,
        {"mode": mode, "characters": len(text), **focus},
        verified=None,
        verification="synthetic text emitted; focused value was not readable",
        include_post_state=include_post_state,
    )


def _finder_file_reference_path(value: object) -> str | None:
    """Resolve Finder's opaque AXURL to the item's current filesystem path."""
    try:
        from Foundation import NSURL  # type: ignore[import-untyped]

        url = (
            value if hasattr(value, "filePathURL") else NSURL.URLWithString_(str(value))
        )
        file_path_url = url.filePathURL() if url is not None else None
        path = file_path_url.path() if file_path_url is not None else None
    except Exception:  # noqa: BLE001 - optional PyObjC/Foundation boundary
        return None
    return str(path) if path else None


def _finder_rename_path_state(binding: dict[str, Any]) -> tuple[str, str]:
    """Classify the exact bound Finder item without trusting a rebuilt AX tree."""
    actual_path = _finder_file_reference_path(binding["file_reference"])
    if actual_path is None:
        raise ComputerUseError(
            "target_drift", "Finder rename file reference is no longer available"
        )
    original_path = str(binding["original_path"])
    requested_path = str(Path(original_path).with_name(binding["requested_basename"]))
    if actual_path == original_path:
        return "unchanged", actual_path
    if actual_path == requested_path:
        return "committed", actual_path
    raise ComputerUseError(
        "target_drift", "Finder rename target moved or changed unexpectedly"
    )


def _finder_transaction_editor(
    binding: dict[str, Any], snapshot: dict, element_index: int, *, expected_value: str
) -> object:
    """Revalidate the exact selected Finder editor captured before approval."""
    if (
        int(snapshot.get("app", {}).get("pid", -1)) != binding["pid"]
        or str(snapshot.get("window_id") or "") != binding["window_id"]
    ):
        raise ComputerUseError(
            "target_drift", "Finder rename app or window changed after approval"
        )
    live = None
    try:
        entry = _element(snapshot, element_index)
        if entry.get("role") == "AXTextField":
            live = _live_element(snapshot, element_index, validate_point=False)
    except ComputerUseError:
        # Finder rebuilds the transient editor tree after focus changes. The
        # serialized index is only a hint; the opaque file reference below is
        # the authority for an exact focused-editor rebind.
        pass
    reference = (
        _finder_transaction_reference(
            live, snapshot, detached_expected_value=expected_value
        )
        if live is not None
        else None
    )
    if reference is None:
        focused = _focused_ax_element(snapshot["app"])
        if focused is not None and focused != live:
            live = focused
            reference = _finder_transaction_reference(
                live, snapshot, detached_expected_value=expected_value
            )
    if live is None or reference is None:
        raise ComputerUseError(
            "target_drift", "Finder rename editor is no longer in the selected row"
        )
    if reference != binding["file_reference"]:
        raise ComputerUseError(
            "target_drift",
            "Finder rename editor no longer identifies the approved item",
        )
    return live


def _resume_finder_transaction_editor(
    binding: dict[str, Any], snapshot: dict, element_index: int
) -> object:
    """Re-enter rename for the same bound Finder row after approval took focus."""
    original_name = Path(str(binding["original_path"])).name
    try:
        return _finder_transaction_editor(
            binding, snapshot, element_index, expected_value=original_name
        )
    except ComputerUseError:
        # The approval sheet can close Finder's transient inline editor. Do not
        # trust a rebuilt index: reopen only from the retained row + opaque file
        # reference after proving it is still the sole selected outline row.
        pass
    _finder_rename_path_state(binding)
    key = _finder_rename_binding_key(snapshot)
    with _finder_rename_binding_lock:
        cached = _finder_rename_bindings.get(key)
    if cached is None:
        raise ComputerUseError(
            "target_drift", "approved Finder rename binding is no longer available"
        )
    row, reference, path, observed_at = cached

    def validate_bound_row() -> None:
        parent = ax_driver._get(row, "AXParent")
        selected_rows = [
            candidate
            for candidate in ax_driver._as_list(ax_driver._get(parent, "AXChildren"))
            if ax_driver._get(candidate, "AXRole") == "AXRow"
            and ax_driver._get(candidate, "AXSelected") is True
        ]
        if (
            time.monotonic() - observed_at > SNAPSHOT_TTL_S
            or reference != binding["file_reference"]
            or path != binding["original_path"]
            or _finder_file_reference_path(reference) != binding["original_path"]
            or ax_driver._get(row, "AXRole") != "AXRow"
            or ax_driver._get(row, "AXSelected") is not True
            or ax_driver._get(parent, "AXRole") != "AXOutline"
            or selected_rows != [row]
        ):
            raise ComputerUseError(
                "target_drift", "approved Finder row changed while restoring rename"
            )

    validate_bound_row()
    rename_item = _finder_rename_menu_item(snapshot)
    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window, allow_exact_main_window=True)
    _finder_rename_path_state(binding)
    validate_bound_row()
    if (
        ax_driver.AXUIElementPerformAction(rename_item, "AXPress")
        != ax_driver.kAXErrorSuccess
    ):
        raise ComputerUseError(
            "action_failed", "Finder native Rename command rejected AXPress"
        )
    time.sleep(0.1)
    _finder_rename_path_state(binding)
    live = _finder_item_editor_for_path(snapshot, str(binding["original_path"]), row)
    if (
        _finder_transaction_reference(
            live, snapshot, detached_expected_value=original_name
        )
        != binding["file_reference"]
    ):
        raise ComputerUseError(
            "target_drift", "restored Finder editor is not the approved item"
        )
    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window, allow_exact_main_window=True)
    return live


def inspect_finder_rename(
    snapshot: dict, element_index: int, requested_basename: str
) -> dict[str, Any]:
    """Bind a proposed inline rename to one selected Finder file reference."""
    if not is_finder_snapshot(snapshot):
        raise ComputerUseError("invalid_argument", "target is not a Finder rename")
    requested = requested_basename.replace("\u200b", "").replace("\ufeff", "")
    if not requested or requested != Path(requested).name:
        raise ComputerUseError(
            "invalid_argument", "Finder rename requires one basename"
        )
    entry = _element(snapshot, element_index)
    if entry.get("role") != "AXTextField":
        raise ComputerUseError(
            "target_drift", "Finder rename target is not an item editor"
        )
    live = _live_element(snapshot, element_index, validate_point=False)
    reference = (
        _finder_transaction_reference(live, snapshot) if live is not None else None
    )
    if reference is None:
        key = _finder_rename_binding_key(snapshot)
        with _finder_rename_binding_lock:
            cached = _finder_rename_bindings.get(key)
        cached_name = Path(cached[2]).name if cached is not None else ""
        indexed_label = (
            str(entry.get("label") or "").replace("\u200b", "").replace("\ufeff", "")
        )
        if indexed_label == cached_name:
            focused = _focused_ax_element(snapshot["app"])
            if focused is not None and focused != live:
                live = focused
                reference = _finder_transaction_reference(live, snapshot)
    if live is None or reference is None:
        raise ComputerUseError(
            "target_drift",
            "Finder rename target is not the selected item under the focused outline",
        )
    original_path = (
        _finder_file_reference_path(reference) if reference is not None else None
    )
    if original_path is None:
        raise ComputerUseError(
            "target_drift", "Finder rename target has no stable file reference"
        )
    binding: dict[str, Any] = {
        "pid": int(snapshot["app"]["pid"]),
        "window_id": str(snapshot.get("window_id") or ""),
        "file_reference": reference,
        "original_path": original_path,
        "requested_basename": requested,
    }
    _finder_rename_path_state(binding)
    return binding


def set_finder_rename_value(
    app: str,
    snapshot: dict,
    element_index: int,
    binding: dict[str, Any],
) -> dict:
    """Write an approved Finder basename only while its exact binding is valid."""
    _finder_rename_path_state(binding)
    # The approval card made Rapid frontmost. Restore only the exact approved
    # Finder window, with the original opaque reference checked on both sides,
    # before reacquiring its focused inline editor.
    raise_selected_window(app, snapshot, focus_exact_window=True)
    _finder_rename_path_state(binding)
    from ApplicationServices import (  # type: ignore[import-untyped]
        AXUIElementSetAttributeValue,
        kAXValueAttribute,
    )

    live = _resume_finder_transaction_editor(binding, snapshot, element_index)
    # Keep the opaque-reference check adjacent to the irreversible AX write.
    # The approval pause and editor lookup must not leave a race window in
    # which a moved item receives the approved basename.
    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window, allow_exact_main_window=True)
    _finder_rename_path_state(binding)
    err = AXUIElementSetAttributeValue(
        live, kAXValueAttribute, binding["requested_basename"]
    )
    if (
        err != 0
        or _normalized_finder_editor_value(live) != binding["requested_basename"]
    ):
        raise ComputerUseError(
            "action_failed", "Finder rejected the approved rename value"
        )
    state, actual_path = _finder_rename_path_state(binding)
    if state == "committed":
        _clear_finder_rename_binding(snapshot)
    return _finish_action(
        app,
        snapshot,
        {
            "mode": "AXSetValue",
            "element_index": element_index,
            "verification_source": (
                "finder_file_reference_basename" if state == "committed" else "pending"
            ),
            "actual_basename": Path(actual_path).name,
        },
        verified=True if state == "committed" else None,
        verification=(
            "Finder file-reference URL resolved to the requested basename"
            if state == "committed"
            else "approved Finder rename value is staged but not committed"
        ),
    )


def commit_finder_rename(
    app: str,
    snapshot: dict,
    element_index: int,
    binding: dict[str, Any],
) -> dict:
    """Commit one previously approved Finder rename, or verify it already landed."""
    state, actual_path = _finder_rename_path_state(binding)
    if state == "committed":
        _clear_finder_rename_binding(snapshot)
        result = _finish_action(
            app,
            snapshot,
            {
                "mode": "FinderRenameAlreadyCommitted",
                "key": "enter",
                "executed": False,
                "verification_source": "finder_file_reference_basename",
                "actual_basename": Path(actual_path).name,
            },
            verified=True,
            verification="Finder file-reference URL already resolved to the requested basename",
        )
        result["attempted"] = False
        return result
    live = _finder_transaction_editor(
        binding,
        snapshot,
        element_index,
        expected_value=str(binding["requested_basename"]),
    )
    if _normalized_finder_editor_value(live) != binding["requested_basename"]:
        raise ComputerUseError(
            "target_drift",
            "Finder rename editor no longer contains the approved basename",
        )
    # The approval card can leave Finder in the background. Restore only the
    # exact already-approved window, with the same opaque file reference
    # checked immediately before and after focus changes. Raising Finder may
    # itself commit the staged rename, in which case no Enter is needed.
    _finder_rename_path_state(binding)
    raise_selected_window(app, snapshot, focus_exact_window=True)
    state, actual_path = _finder_rename_path_state(binding)
    if state == "committed":
        _clear_finder_rename_binding(snapshot)
        result = _finish_action(
            app,
            snapshot,
            {
                "mode": "FinderRenameCommittedDuringFocusRestore",
                "key": "enter",
                "executed": False,
                "verification_source": "finder_file_reference_basename",
                "actual_basename": Path(actual_path).name,
            },
            verified=True,
            verification="Finder file-reference URL resolved to the requested basename",
        )
        result["attempted"] = False
        return result
    live = _finder_transaction_editor(
        binding,
        snapshot,
        element_index,
        expected_value=str(binding["requested_basename"]),
    )
    if _normalized_finder_editor_value(live) != binding["requested_basename"]:
        raise ComputerUseError(
            "target_drift", "Finder rename editor changed during focus restoration"
        )
    # Finder snapshots always take the foreground route (see _background_delivery).
    prepared = _prepare_synthetic_action(
        app,
        None,
        snapshot,
        allow_finder_main_window=True,
    )
    focused = _focused_ax_element(prepared["app"])
    exact_editor_focus = live == focused
    selected_editor_under_focused_outline = (
        _is_selected_finder_row_under_focused_outline(prepared, live)
    )
    if (not exact_editor_focus and not selected_editor_under_focused_outline) or (
        _finder_transaction_reference(
            live,
            prepared,
            detached_expected_value=str(binding["requested_basename"]),
        )
        != binding["file_reference"]
    ):
        raise ComputerUseError(
            "target_drift",
            "Finder rename editor changed before approved Enter dispatch",
        )
    # This is the last operation before the single approved key dispatch.
    _finder_rename_path_state(binding)
    ax_driver._press_key(KEY_ALIASES["enter"])
    actual_path = binding["original_path"]
    for _ in range(10):
        state, actual_path = _finder_rename_path_state(binding)
        if state == "committed":
            break
        time.sleep(0.1)
    verified = state == "committed"
    if verified:
        _clear_finder_rename_binding(snapshot)
    return _finish_action(
        app,
        snapshot,
        {
            "ok": verified,
            "mode": "CGEvent-keycode",
            "key": "enter",
            "executed": True,
            "verification_source": "finder_file_reference_basename",
            "actual_basename": Path(actual_path).name,
        },
        verified=verified,
        verification=(
            "Finder file-reference URL resolved to the requested basename"
            if verified
            else "Finder file-reference URL did not resolve to the requested basename"
        ),
    )


def _finder_rename_binding_key(snapshot: dict) -> tuple[int, str]:
    return int(snapshot["app"]["pid"]), str(snapshot.get("window_id") or "")


def _finder_file_reference_for_editor(
    live: object,
    snapshot: dict,
    *,
    allow_selected_row_rebind: bool = False,
    detached_expected_value: str | None = None,
) -> object | None:
    """Bind Finder's replacement editor to the original selected item."""
    cell = ax_driver._get(live, "AXParent")
    row = ax_driver._get(cell, "AXParent") if cell is not None else None
    reference = ax_driver._get(live, "AXURL")
    key = _finder_rename_binding_key(snapshot)
    if reference is not None:
        path = _finder_file_reference_path(reference)
        if row is not None and path is not None:
            with _finder_rename_binding_lock:
                _finder_rename_bindings[key] = (row, reference, path, time.monotonic())
        return reference
    with _finder_rename_binding_lock:
        binding = _finder_rename_bindings.get(key)
    if binding is None:
        return None
    bound_row, bound_reference, bound_path, observed_at = binding
    if (
        time.monotonic() - observed_at > SNAPSHOT_TTL_S
        or _finder_file_reference_path(bound_reference) != bound_path
    ):
        with _finder_rename_binding_lock:
            _finder_rename_bindings.pop(key, None)
        return None
    if row != bound_row:
        # Finder can recreate the AXRow object when inline editing begins.
        # Retain the original opaque reference, but use it only to classify a
        # generic Enter that is already independently bound to the exact
        # focused editor and selected replacement row. This never authorizes
        # input or relaxes the synthetic-action window/focus guards.
        cell = ax_driver._get(live, "AXParent")
        if not allow_selected_row_rebind:
            return None
        focused = live == _focused_ax_element(snapshot["app"])
        replacement_row = (
            ax_driver._get(cell, "AXRole") == "AXCell"
            and ax_driver._get(row, "AXRole") == "AXRow"
            and ax_driver._get(row, "AXSelected") is True
        )
        bound_parent = ax_driver._get(bound_row, "AXParent")
        selected_rows = [
            candidate
            for candidate in ax_driver._as_list(
                ax_driver._get(bound_parent, "AXChildren")
            )
            if ax_driver._get(candidate, "AXRole") == "AXRow"
            and ax_driver._get(candidate, "AXSelected") is True
        ]
        detached_editor = (
            ax_driver._get(live, "AXRole") == "AXTextField"
            and ax_driver._get(live, "AXURL") is None
            and ax_driver._get(cell, "AXRole") == "AXApplication"
            and ax_driver._get(bound_row, "AXRole") == "AXRow"
            and ax_driver._get(bound_row, "AXSelected") is True
            and ax_driver._get(bound_parent, "AXRole") == "AXOutline"
            and selected_rows == [bound_row]
            and _normalized_finder_editor_value(live)
            == (detached_expected_value or Path(bound_path).name)
        )
        if not (focused and (replacement_row or detached_editor)):
            return None
    return bound_reference


def _finder_transaction_reference(
    live: object, snapshot: dict, *, detached_expected_value: str | None = None
) -> object | None:
    """Resolve the exact selected item behind Finder's inline editor."""
    selected_row_editor = _is_selected_finder_row_under_focused_outline(snapshot, live)
    if selected_row_editor:
        return _finder_file_reference_for_editor(
            live,
            snapshot,
            allow_selected_row_rebind=True,
            detached_expected_value=detached_expected_value,
        )
    # Recent Finder versions expose the active inline editor as a focused,
    # no-URL AXTextField directly under the app. The retained binding still
    # has to identify one selected AXRow under an outline and an unchanged
    # opaque file reference; _finder_file_reference_for_editor proves those
    # facts before returning it.
    parent = ax_driver._get(live, "AXParent")
    if (
        live == _focused_ax_element(snapshot["app"])
        and ax_driver._get(live, "AXRole") == "AXTextField"
        and ax_driver._get(live, "AXURL") is None
        and ax_driver._get(parent, "AXRole") == "AXApplication"
    ):
        return _finder_file_reference_for_editor(
            live,
            snapshot,
            allow_selected_row_rebind=True,
            detached_expected_value=detached_expected_value,
        )
    return None


def _clear_finder_rename_binding(snapshot: dict) -> None:
    with _finder_rename_binding_lock:
        _finder_rename_bindings.pop(_finder_rename_binding_key(snapshot), None)


def _finder_rename_menu_item(snapshot: dict) -> object:
    """Return Finder's one enabled native Rename menu command."""
    app_info = snapshot["app"]
    app_element = _pid_app_element(app_info)
    menu_bar = ax_driver._get(app_element, "AXMenuBar")
    matches: list[object] = []
    seen = 0

    def walk(element: object, depth: int) -> None:
        nonlocal seen
        if depth > SAVE_MENU_MAX_DEPTH or seen >= SAVE_MENU_MAX_NODES:
            return
        seen += 1
        command = ax_driver._get(element, "AXMenuItemCmdChar")
        modifiers = ax_driver._get(element, "AXMenuItemCmdModifiers")
        if (
            ax_driver._get(element, "AXRole") == "AXMenuItem"
            and (
                (command in {"\r", "\n"} and modifiers == 0)
                or ax_driver._get(element, "AXTitle") == "Rename"
            )
            and ax_driver._get(element, "AXEnabled") is True
            and "AXPress" in ax_driver._action_names(element)
        ):
            matches.append(element)
        for child in ax_driver._as_list(ax_driver._get(element, "AXChildren")):
            walk(child, depth + 1)

    if menu_bar is not None:
        walk(menu_bar, 0)
    if len(matches) != 1:
        raise ComputerUseError(
            "element_not_found",
            "Finder native Rename command is unavailable or ambiguous",
        )
    return matches[0]


def _finder_item_editor_for_path(
    snapshot: dict, expected_path: str, expected_row: object
) -> object:
    """Reacquire Finder's replacement inline editor for one file reference."""
    focused_element = _focused_ax_element(snapshot["app"])
    if focused_element is not None:
        focused_cell = ax_driver._get(focused_element, "AXParent")
        focused_row = ax_driver._get(focused_cell, "AXParent")
        focused_reference = ax_driver._get(focused_element, "AXURL")
        if (
            ax_driver._get(focused_element, "AXRole") == "AXTextField"
            and ax_driver._get(focused_cell, "AXRole") == "AXCell"
            and focused_row == expected_row
            and ax_driver._get(focused_row, "AXSelected") is True
            and (
                focused_reference is None
                or _finder_file_reference_path(focused_reference) == expected_path
            )
        ):
            return focused_element
        # Recent Finder versions can detach the active rename editor directly
        # under AXApplication. Accept that shape only when the retained opaque
        # binding still names this exact row/path, the row is the sole selected
        # item in its outline, and the focused value is still the original name.
        key = _finder_rename_binding_key(snapshot)
        with _finder_rename_binding_lock:
            cached = _finder_rename_bindings.get(key)
        if cached is not None:
            bound_row, bound_reference, bound_path, observed_at = cached
            outline = ax_driver._get(expected_row, "AXParent")
            selected_rows = [
                candidate
                for candidate in ax_driver._as_list(
                    ax_driver._get(outline, "AXChildren")
                )
                if ax_driver._get(candidate, "AXRole") == "AXRow"
                and ax_driver._get(candidate, "AXSelected") is True
            ]
            detached_matches = (
                time.monotonic() - observed_at <= SNAPSHOT_TTL_S
                and bound_row == expected_row
                and bound_path == expected_path
                and _finder_file_reference_path(bound_reference) == expected_path
                and ax_driver._get(focused_element, "AXRole") == "AXTextField"
                and ax_driver._get(focused_element, "AXFocused") is True
                and ax_driver._get(focused_element, "AXURL") is None
                and ax_driver._get(focused_cell, "AXRole") == "AXApplication"
                and ax_driver._get(expected_row, "AXRole") == "AXRow"
                and ax_driver._get(expected_row, "AXSelected") is True
                and ax_driver._get(outline, "AXRole") == "AXOutline"
                and selected_rows == [expected_row]
                and _normalized_finder_editor_value(focused_element)
                == Path(expected_path).name
            )
            if detached_matches:
                return focused_element
    window = _focused_ax_window(snapshot["app"])
    matches: list[object] = []
    seen = 0

    def walk(element: object, depth: int) -> None:
        nonlocal seen
        if depth > TEXTEDIT_VALUE_MAX_DEPTH or seen >= TEXTEDIT_VALUE_MAX_NODES:
            return
        seen += 1
        if ax_driver._get(element, "AXRole") == "AXTextField":
            cell = ax_driver._get(element, "AXParent")
            candidate_row = ax_driver._get(cell, "AXParent")
            if (
                ax_driver._get(cell, "AXRole") == "AXCell"
                and candidate_row == expected_row
                and ax_driver._get(candidate_row, "AXSelected") is True
            ):
                reference = ax_driver._get(element, "AXURL")
                if _finder_file_reference_path(reference) == expected_path:
                    matches.append(element)
        for child in ax_driver._as_list(ax_driver._get(element, "AXChildren")):
            walk(child, depth + 1)

    if window is not None:
        walk(window, 0)
    if len(matches) != 1:
        raise ComputerUseError(
            "target_drift",
            "Finder replacement rename editor is unavailable or ambiguous",
        )
    return matches[0]


def _finder_inline_rename(
    app: str,
    snapshot: dict,
    element_index: int,
    *,
    include_post_state: bool = False,
) -> dict | None:
    """Commit and verify one Finder item-cell inline rename.

    Finder can accept ``AXValue`` and display it in the editor without
    committing the directory entry. Bind the commit to the exact AXCell text
    field, retain its opaque file-reference URL, then resolve that same item
    after ``AXConfirm``/Enter and compare the filesystem basename.
    """
    entry = _element(snapshot, element_index)
    if (
        not is_finder_snapshot(snapshot)
        or entry.get("role") != "AXTextField"
        or entry.get("parent_role") != "AXCell"
    ):
        return None
    live = _live_element(snapshot, element_index, validate_point=False)
    if live is None:
        raise ComputerUseError("target_drift", "Finder inline rename editor changed")
    file_reference = ax_driver._get(live, "AXURL")
    requested_name = _read_value(live)
    if file_reference is None or not requested_name:
        return None

    # A selected Finder row exposes the same AXTextField/AXCell shape before
    # inline editing begins. Only treat it as a pending commit once its editor
    # value differs from the file reference's current basename. The first
    # Enter must continue through the generic path so Finder can start rename.
    expected_basename = requested_name.replace("\u200b", "").replace("\ufeff", "")
    original_path = _finder_file_reference_path(file_reference)
    if original_path is None or Path(original_path).name == expected_basename:
        return None

    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window)
    cell = ax_driver._get(live, "AXParent")
    row = ax_driver._get(cell, "AXParent") if cell is not None else None
    if (
        ax_driver._get(cell, "AXRole") != "AXCell"
        or ax_driver._get(row, "AXRole") != "AXRow"
    ):
        raise ComputerUseError(
            "target_drift", "Finder rename target is no longer the bound item row"
        )

    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817

    if ax_driver._get(row, "AXSelected") is not True:
        if (
            AS.AXUIElementSetAttributeValue(row, "AXSelected", True)
            != AS.kAXErrorSuccess
        ):
            raise ComputerUseError(
                "target_drift", "Finder rename target could not be selected"
            )
        time.sleep(0.05)
        if ax_driver._get(row, "AXSelected") is not True:
            raise ComputerUseError(
                "target_drift", "Finder rename target did not accept selection"
            )
        expected_window = _validate_snapshot_window(snapshot)
        _validate_focused_window(snapshot, expected_window)

    rename_item = _finder_rename_menu_item(snapshot)
    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window)
    if (
        ax_driver._get(row, "AXSelected") is not True
        or _finder_file_reference_path(file_reference) != original_path
    ):
        raise ComputerUseError(
            "target_drift", "Finder rename target changed during menu resolution"
        )
    if AS.AXUIElementPerformAction(rename_item, "AXPress") != AS.kAXErrorSuccess:
        raise ComputerUseError(
            "action_failed", "Finder native Rename command rejected AXPress"
        )
    time.sleep(0.1)
    if _finder_file_reference_path(file_reference) != original_path:
        raise ComputerUseError("target_drift", "Finder rename target changed")
    live = _finder_item_editor_for_path(snapshot, original_path, row)
    if not (
        ax_driver._get(live, "AXFocused") is True
        or live == _focused_ax_element(snapshot["app"])
    ):
        raise ComputerUseError(
            "target_drift", "Finder native rename editor is not focused"
        )
    expected_window = _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot, expected_window)
    ax_driver._press_key(ax_driver._keycode_for("a"), modifiers=ax_driver.FLAG_COMMAND)
    ax_driver._type_text(expected_basename)
    ax_driver._press_key(KEY_ALIASES["enter"])

    # AXValue may contain invisible editor sentinels. They are not part of a
    # legal Finder basename and must not make an uncommitted editor look true.
    actual_path = None
    for _ in range(10):
        actual_path = _finder_file_reference_path(file_reference)
        if actual_path is not None and Path(actual_path).name == expected_basename:
            break
        time.sleep(0.1)
    verified = actual_path is not None and Path(actual_path).name == expected_basename
    if verified:
        _clear_finder_rename_binding(snapshot)
    return _finish_action(
        app,
        snapshot,
        {
            "ok": verified,
            "mode": "FinderRenameMenu+CGEvent-text",
            "key": "enter",
            "executed": True,
            "actual_basename": Path(actual_path).name if actual_path else None,
            "verification_source": "finder_file_reference_basename",
        },
        verified=verified,
        verification=(
            "Finder file-reference URL resolved to the requested basename"
            if verified
            else "Finder file-reference URL did not resolve to the requested basename"
        ),
        include_post_state=include_post_state,
    )


def press_key(
    app: str,
    key: str,
    window_id: int | str | None = None,
    include_post_state: bool = False,
    expected_snapshot: dict | None = None,
    element_index: int | None = None,
) -> dict:
    normalized = key.strip().lower()
    finder_enter_binding: tuple[object, str, str] | None = None
    if expected_snapshot is not None and element_index is not None:
        entry = _element(expected_snapshot, element_index)
        is_transient = entry.get(
            "source_window_id", expected_snapshot.get("window_id")
        ) != expected_snapshot.get("window_id")
        finder_escape = normalized in {"escape", "esc"} and is_finder_snapshot(
            expected_snapshot
        )
        if is_transient and normalized != "enter" and not finder_escape:
            raise ComputerUseError(
                "unsupported_key",
                "transient companion targets only allow Enter after exact focus validation",
            )
        if normalized == "enter":
            rename_result = _finder_inline_rename(
                app,
                expected_snapshot,
                element_index,
                include_post_state=include_post_state,
            )
            if rename_result is not None:
                return rename_result
            if (
                is_finder_snapshot(expected_snapshot)
                and entry.get("role") == "AXTextField"
                and entry.get("parent_role") == "AXCell"
            ):
                live = _live_element(
                    expected_snapshot, element_index, validate_point=False
                )
                reference = (
                    _finder_file_reference_for_editor(
                        live,
                        expected_snapshot,
                        allow_selected_row_rebind=True,
                    )
                    if live is not None
                    else None
                )
                requested = _read_value(live) if live is not None else None
                before_path = (
                    _finder_file_reference_path(reference)
                    if reference is not None
                    else None
                )
                if requested and before_path:
                    finder_enter_binding = (
                        reference,
                        before_path,
                        requested.replace("\u200b", "").replace("\ufeff", ""),
                    )
    if normalized in KEY_ALIASES:
        snapshot = _prepare_synthetic_action(
            app,
            window_id,
            expected_snapshot,
            element_index,
            allow_focused_editable_enter=normalized == "enter",
            allow_finder_escape=(
                normalized in {"escape", "esc"}
                and expected_snapshot is not None
                and is_finder_snapshot(expected_snapshot)
            ),
        )
        if (
            normalized in {"enter", "return", "space"}
            and expected_snapshot is not None
            and element_index is not None
        ):
            # Planner-bound keyboard activation follows focus, not the
            # serialized index. Re-bind immediately before dispatch. The
            # user-directed press-key CLI has no indexed planner target.
            inspect_focused_element(
                snapshot,
                element_index,
                allow_selected_finder_row=normalized in {"enter", "return"},
            )
        focus: dict[str, Any] = {}
        route = _send_key(
            snapshot,
            KEY_ALIASES[normalized],
            0,
            _keyboard_background(snapshot),
            focus,
        )
        mode = "SkyLight-keycode" if route == ROUTE_PID else "CGEvent-keycode"
        if finder_enter_binding is not None:
            reference, before_path, requested_basename = finder_enter_binding
            actual_path = before_path
            for _ in range(10):
                actual_path = _finder_file_reference_path(reference) or actual_path
                if (
                    Path(actual_path).name == requested_basename
                    and Path(before_path).name != requested_basename
                ):
                    _clear_finder_rename_binding(snapshot)
                    return _finish_action(
                        app,
                        snapshot,
                        {
                            "mode": mode,
                            "key": normalized,
                            "verification_source": "finder_file_reference_basename",
                            "actual_basename": Path(actual_path).name,
                            **focus,
                        },
                        verified=True,
                        verification="Finder file-reference URL resolved to the requested basename",
                        include_post_state=include_post_state,
                    )
                time.sleep(0.1)
        return _finish_action(
            app,
            snapshot,
            {"mode": mode, "key": normalized, **focus},
            verified=None,
            verification="synthetic key emitted; outcome not asserted",
            include_post_state=include_post_state,
        )
    if normalized in ax_driver.KEYCODE_MAP:
        snapshot = _prepare_synthetic_action(
            app, window_id, expected_snapshot, element_index
        )
        focus = {}
        route = _send_key(
            snapshot,
            ax_driver._keycode_for(normalized),
            0,
            _keyboard_background(snapshot),
            focus,
        )
        mode = "SkyLight-keycode" if route == ROUTE_PID else "CGEvent-keycode"
        return _finish_action(
            app,
            snapshot,
            {"mode": mode, "key": normalized, **focus},
            verified=None,
            verification="synthetic key emitted; outcome not asserted",
            include_post_state=include_post_state,
        )
    raise ComputerUseError("unsupported_key", f"unsupported single key {key!r}")


def hotkey(
    app: str,
    key: str,
    window_id: int | str | None = None,
    include_post_state: bool = False,
) -> dict:
    """Modifier chord like 'Cmd+A', 'Ctrl+Shift+Tab'."""
    parts = [p.strip().lower() for p in key.split("+") if p.strip()]
    if len(parts) < 2:
        raise ComputerUseError(
            "unsupported_key", f"hotkey needs a modifier plus a key: {key!r}"
        )
    modifiers = 0
    for part in parts[:-1]:
        if part not in MODIFIER_FLAGS:
            raise ComputerUseError("unsupported_key", f"unknown modifier {part!r}")
        modifiers |= MODIFIER_FLAGS[part]
    key_part = parts[-1]
    if key_part in KEY_ALIASES:
        keycode = KEY_ALIASES[key_part]
    elif key_part in ax_driver.KEYCODE_MAP:
        keycode = ax_driver._keycode_for(key_part)
    else:
        raise ComputerUseError(
            "unsupported_key", f"unsupported hotkey key {key_part!r}"
        )
    if modifiers & MODIFIER_FLAGS["cmd"] and window_id is not None:
        probe = get_app_state(
            app,
            screenshot=False,
            use_cache=False,
            window_id=window_id,
            activate=OBSERVE_BY_ROUTE,
        )
        if _background_delivery(probe):
            matched, item = _menu_equivalent_lookup(
                probe["app"], key_part, modifiers, keycode
            )
            if not matched:
                # Not a menu command: the chord is for the content (a web
                # app's Cmd+K, an editor binding), which takes it in the
                # background.
                return _background_chord(
                    app, probe, keycode, modifiers, key, include_post_state
                )
            if item is not None:
                # A menu command: AppKit only dispatches key equivalents to
                # the active app, but the menu item itself can be pressed
                # through Accessibility with the target made key -- no raise,
                # no activation, and it cannot hit the user's window.
                return _press_menu_item(
                    app,
                    probe,
                    item,
                    f"{key} ({_menu_item_title(item)})",
                    include_post_state,
                )
            if _process_is_active(probe):
                # Possibly a menu command whose item could not be resolved
                # (unreadable menu tree): the user is in this app, so its
                # menus are live; make the target key for the chord so it
                # cannot hit the user's window.
                return _background_chord(
                    app, probe, keycode, modifiers, key, include_post_state
                )
            # AppKit runs menu commands only for the active app; raising it
            # would take the user's screen, so say so instead of failing on a
            # geometry check.
            raise ComputerUseError(
                "synthetic_input_blocked",
                f"{key} is (or may be) a menu command of "
                f"{probe['app']['name']}; menu "
                "commands only run while the app is in front",
                (
                    "Use an Accessibility action or set-value for the same "
                    "effect, or ask the user before bringing the app forward.",
                ),
            )
    snapshot = _prepare_synthetic_action(app, window_id, modifiers=modifiers)
    focus: dict[str, Any] = {}
    if _keyboard_background(snapshot, modifiers):
        _send_key(snapshot, keycode, modifiers, True, focus)
        mode = "SkyLight-hotkey"
    else:
        import Quartz

        down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
        up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
        Quartz.CGEventSetFlags(down, modifiers)
        Quartz.CGEventSetFlags(up, modifiers)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        time.sleep(0.02)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
        mode = "CGEvent-hotkey"
    return _finish_action(
        app,
        snapshot,
        {"mode": mode, "key": key, **focus},
        verified=None,
        verification="synthetic hotkey emitted; outcome not asserted",
        include_post_state=include_post_state,
    )


# AXMenuItemCmdModifiers: Cmd is implied unless bit 3 (0x8) is set.
_MENU_SHIFT, _MENU_OPTION, _MENU_CONTROL, _MENU_NO_CMD = 0x1, 0x2, 0x4, 0x8
_MENU_SCAN_LIMIT = 3000


# Menu-tree nodes whose children must be readable for a chord to be ruled out.
_MENU_CONTAINER_ROLES = frozenset({"AXMenuBar", "AXMenuBarItem", "AXMenu"})


def _menu_bar(app_info: dict) -> object | None:
    return ax_driver._get(_pid_app_element(app_info), "AXMenuBar")


def _menu_item_title(item: object) -> str:
    # Chromium's <select> options carry their label in AXValue, not AXTitle.
    for attribute in ("AXTitle", "AXValue", "AXDescription"):
        value = ax_driver._get(item, attribute)
        if isinstance(value, str) and value:
            return value
    return ""


def _menu_equivalent_lookup(
    app_info: dict, key_part: str, modifiers: int, keycode: int | None = None
) -> tuple[bool, object | None]:
    """Whether a Cmd chord is one of the app's menu key equivalents, and its
    menu item when it could be resolved.

    Items match by ``AXMenuItemCmdChar`` or, for arrows and function keys,
    ``AXMenuItemCmdVirtualKey``. An unreadable or truncated menu tree (a menu
    bar without readable items, a menu whose children fail to read), or a
    matching key with unreadable modifiers, counts as a match without an item
    (fail closed: it may be a menu command that cannot be pressed through
    Accessibility). A leaf item without ``AXChildren`` is normal.
    """
    want = 0
    if modifiers & MODIFIER_FLAGS["shift"]:
        want |= _MENU_SHIFT
    if modifiers & MODIFIER_FLAGS["option"]:
        want |= _MENU_OPTION
    if modifiers & MODIFIER_FLAGS["ctrl"]:
        want |= _MENU_CONTROL
    bar = _menu_bar(app_info)
    if bar is None:
        # Unreadable menu bar: cannot rule a menu command out, fail closed.
        return True, None
    uncertain = False
    stack, seen = [bar], 0
    while stack:
        if seen >= _MENU_SCAN_LIMIT:
            return True, None  # truncated scan: fail closed as well
        node = stack.pop()
        seen += 1
        char = ax_driver._get(node, "AXMenuItemCmdChar")
        vkey = ax_driver._get(node, "AXMenuItemCmdVirtualKey")
        if (
            isinstance(char, str)
            and char.strip()
            and char.casefold() == key_part.casefold()
        ) or (
            keycode is not None
            and isinstance(vkey, int)
            and not isinstance(vkey, bool)
            and vkey == keycode
        ):
            mods = ax_driver._get(node, "AXMenuItemCmdModifiers")
            if not isinstance(mods, int):
                uncertain = True
            elif not mods & _MENU_NO_CMD and mods == want:
                return True, node
        if node is bar or ax_driver._get(node, "AXRole") in _MENU_CONTAINER_ROLES:
            readable, raw = ax_driver._get_checked(node, "AXChildren")
            children = ax_driver._as_list(raw)
            if node is bar and not children:
                return True, None  # no readable menu items: fail closed
            if not readable:
                uncertain = True  # a menu could not be enumerated: fail closed
        else:
            children = ax_driver._as_list(ax_driver._get(node, "AXChildren"))
        stack.extend(children)
    return uncertain, None


def _is_menu_equivalent(
    app_info: dict, key_part: str, modifiers: int, keycode: int | None = None
) -> bool:
    """Whether a Cmd chord is (or may be) one of the app's menu key
    equivalents; see :func:`_menu_equivalent_lookup`."""
    return _menu_equivalent_lookup(app_info, key_part, modifiers, keycode)[0]


def _menu_title_key(title: object) -> str:
    return str(title or "").strip().rstrip("…").rstrip(".").strip().casefold()


def _menu_item_by_path(app_info: dict, path: list[str]) -> object:
    """Resolve ``["Edit", "Find", "Find…"]`` against the app's menu bar.

    Each step matches a menu title case-insensitively, ignoring a trailing
    ellipsis. Raises with the available titles at the step that failed.
    """
    node = _menu_bar(app_info)
    if node is None:
        raise ComputerUseError("element_not_found", "the app has no menu bar")
    for depth, step in enumerate(path):
        wanted = _menu_title_key(step)
        candidates = []
        for child in ax_driver._as_list(ax_driver._get(node, "AXChildren")):
            if ax_driver._get(child, "AXRole") == "AXMenu":
                candidates.extend(
                    ax_driver._as_list(ax_driver._get(child, "AXChildren"))
                )
            else:
                candidates.append(child)
        titles = [_menu_item_title(c) for c in candidates]
        match = next(
            (c for c, t in zip(candidates, titles) if _menu_title_key(t) == wanted),
            None,
        )
        if match is None:
            shown = ", ".join(t for t in titles if t)[:400]
            raise ComputerUseError(
                "element_not_found",
                f"no menu item {step!r} under {' > '.join(path[:depth]) or 'the menu bar'}"
                f" (available: {shown})",
            )
        node = match
    return node


def _press_menu_item(
    app: str, snapshot: dict, item: object, label: str, include_post_state: bool
) -> dict:
    """Press a menu item for the target window without raising it.

    Menu commands act on the app's key window, so the target is made key
    (verified, under the gesture lock) for the press and the user's window
    gets focus back. A disabled item is refused rather than pressed into a
    silent no-op.
    """
    clipboard_before = _clipboard_change_count()
    background = _background_delivery(snapshot)
    chord = (
        _menu_item_chord(item) if background and _process_is_active(snapshot) else None
    )
    keyed: AbstractContextManager[dict[str, Any]] = (
        _keyed_target(snapshot, force=True) if background else nullcontext({})
    )
    err = 0
    with keyed as state:
        if not background:
            # Foreground: the caller borrowed the foreground; the command acts
            # on the key window, so it must be exactly the target.
            _validate_focused_window(snapshot, require_exact_window_id=True)
        if chord is not None:
            # The user is in this app: its key equivalent is validated
            # against the (now key) target window, whereas AXEnabled is only
            # refreshed when a menu opens and is stale for this window.
            pid, _ = _target_ids(snapshot)
            if not _synthesize(background_input.press_key, pid, chord[0], chord[1]):
                raise ComputerUseError(
                    "action_failed", "menu key equivalent could not be synthesized"
                )
        else:
            if ax_driver._get(item, "AXEnabled") is False:
                raise ComputerUseError(
                    "synthetic_input_blocked",
                    f"menu item {label} is disabled for window "
                    f"{snapshot.get('window_id')}",
                    (
                        "macOS disables editing commands (Copy, Paste, ...) of an app "
                        "that is not in front; read text from the observation and "
                        "write it with set-value instead, or ask the user before "
                        "bringing the app forward.",
                    ),
                )
            from ApplicationServices import AXUIElementPerformAction

            err = AXUIElementPerformAction(item, "AXPress")
            # kAXErrorCannotComplete: the command ran a modal loop (a dialog)
            # and the press outlived the AX timeout; the command did run.
            if err not in (0, -25204):
                raise ComputerUseError(
                    "accessibility_error", f"menu item {label} press failed: {err}"
                )
    delivery = {
        "mode": "SkyLight-menu-chord" if chord is not None else "AXMenuPress",
        "route": ROUTE_PID if chord is not None else "accessibility",
        "menu_item": label,
        **_focus_fields(state),
    }
    if err == -25204:
        delivery["warning"] = (
            "the command is still running (it may have opened a dialog)"
        )
    clipboard_after = _clipboard_change_count()
    if clipboard_before is not None and clipboard_after is not None:
        # Copy/Cut report whether they produced anything (an empty selection
        # copies nothing); other commands show an unexpected clipboard write.
        delivery["clipboard_changed"] = clipboard_after != clipboard_before
    return _finish_action(
        app,
        snapshot,
        delivery,
        verified=None,
        verification=(
            "menu key equivalent sent to the key target; outcome not asserted"
            if chord is not None
            else "menu command pressed through Accessibility; outcome not asserted"
        ),
        include_post_state=include_post_state,
    )


def _process_is_active(snapshot: dict) -> bool:
    running = ax_driver._application_for_pid(int(snapshot["app"]["pid"]))
    try:
        return bool(running is not None and running.isActive())
    except Exception:  # noqa: BLE001 - treat unknown as inactive
        return False


def _menu_item_chord(item: object) -> tuple[int, int] | None:
    """(keycode, flags) of a menu item's Cmd key equivalent, if it has one."""
    char = ax_driver._get(item, "AXMenuItemCmdChar")
    mods = ax_driver._get(item, "AXMenuItemCmdModifiers")
    if not isinstance(char, str) or not char.strip() or not isinstance(mods, int):
        return None
    if mods & _MENU_NO_CMD:
        return None
    try:
        keycode = ax_driver._keycode_for(char.lower())
    except Exception:  # noqa: BLE001 - glyph keys (arrows, F-keys) fall back to AX
        return None
    if keycode is None:
        return None
    flags = MODIFIER_FLAGS["cmd"]
    if mods & _MENU_SHIFT:
        flags |= MODIFIER_FLAGS["shift"]
    if mods & _MENU_OPTION:
        flags |= MODIFIER_FLAGS["option"]
    if mods & _MENU_CONTROL:
        flags |= MODIFIER_FLAGS["ctrl"]
    return keycode, flags


def _clipboard_change_count() -> int | None:
    try:
        from AppKit import NSPasteboard

        return int(NSPasteboard.generalPasteboard().changeCount())
    except Exception:  # noqa: BLE001 - the report is optional
        return None


def menu(
    app: str,
    path: list[str] | str,
    window_id: int | str | None = None,
    include_post_state: bool = False,
    expected_snapshot: dict | None = None,
) -> dict:
    """Run a menu-bar command (``"Edit > Select All"``) for the target window.

    Works while the app is in the background: the item is pressed through
    Accessibility with the target window made key, so neither the app nor
    the window is raised.
    """
    steps = [p.strip() for p in (path.split(">") if isinstance(path, str) else path)]
    steps = [p for p in steps if p]
    if len(steps) < 2:
        raise ComputerUseError(
            "invalid_argument", f"menu path needs a menu and an item: {path!r}"
        )
    snapshot = expected_snapshot or get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    if not _background_delivery(snapshot):
        _borrow_foreground(snapshot)
    item = _menu_item_by_path(snapshot["app"], steps)
    if ax_driver._as_list(ax_driver._get(item, "AXChildren")):
        raise ComputerUseError(
            "invalid_argument", f"{' > '.join(steps)} is a submenu, not a command"
        )
    return _press_menu_item(app, snapshot, item, " > ".join(steps), include_post_state)


def _background_chord(
    app: str,
    snapshot: dict,
    keycode: int,
    modifiers: int,
    key: str,
    include_post_state: bool,
) -> dict:
    pid, _ = _target_ids(snapshot)
    with _keyed_target(snapshot) as state:
        if not _synthesize(background_input.press_key, pid, keycode, modifiers):
            raise ComputerUseError(
                "action_failed", "background chord could not be synthesized"
            )
    return _finish_action(
        app,
        snapshot,
        {
            "mode": "SkyLight-chord",
            "key": key,
            "route": ROUTE_PID,
            **_focus_fields(state),
        },
        verified=None,
        verification="background chord delivered to the content; outcome not asserted",
        include_post_state=include_post_state,
    )


_AX_SCROLL_NODE_LIMIT = 400
_AX_SCROLL_STEPS = 8


def _ax_scroll_target(
    live: object, vertical: bool, forward: bool, reach: float
) -> tuple[float, float, object] | None:
    """Pick the descendant of scroller ``live`` to bring into view.

    Returns (viewport start, viewport end, node). Chromium reports content
    clipped by the scroller as zero-size frames pinned to its edge, so such
    content has no measurable distance: the nearest one in document order is
    taken. Measurable content prefers the furthest still within ``reach``.
    """
    frame = ax_driver._point_size(live)
    if frame is None:
        return None
    x0, y0, w, h = frame
    lo, hi = (y0, y0 + h) if vertical else (x0, x0 + w)
    within: tuple[float, object] | None = None
    beyond: tuple[float, object] | None = None
    clipped: list[object] = []
    queue, seen = ax_driver._as_list(ax_driver._get(live, "AXChildren")), 0
    while queue and seen < _AX_SCROLL_NODE_LIMIT:
        node = queue.pop(0)
        seen += 1
        # Document order: a node's children come before its next sibling.
        queue[:0] = ax_driver._as_list(ax_driver._get(node, "AXChildren"))
        box = ax_driver._point_size(node)
        if box is None:
            continue
        start = box[1] if vertical else box[0]
        length = box[3] if vertical else box[2]
        if length <= 0:
            if (start >= hi - 1) if forward else (start <= lo + 1):
                clipped.append(node)
            continue
        end = start + length
        travel = end - hi if forward else lo - start
        if travel <= 1:
            continue
        if travel <= reach:
            if within is None or travel > within[0]:
                within = (travel, node)
        elif beyond is None or travel < beyond[0]:
            beyond = (travel, node)
    if within is not None:
        return lo, hi, within[1]
    if clipped:
        return lo, hi, clipped[0] if forward else clipped[-1]
    if beyond is not None:
        return lo, hi, beyond[1]
    return None


def _ax_subtree_frames(live: object) -> list[tuple[float, float, float, float]]:
    frames: list[tuple[float, float, float, float]] = []
    queue = ax_driver._as_list(ax_driver._get(live, "AXChildren"))
    while queue and len(frames) < _AX_SCROLL_NODE_LIMIT:
        node = queue.pop(0)
        queue.extend(ax_driver._as_list(ax_driver._get(node, "AXChildren")))
        box = ax_driver._point_size(node)
        if box is not None:
            frames.append(box)
    return frames


def _ax_scroll(
    snapshot: dict, point: tuple[float, float], direction: str, pages: float
) -> dict | None:
    """Scroll the scroller under ``point`` by bringing hidden content into view.

    The smallest snapshot element containing the point that has content past
    its edge in ``direction`` is the scroller. Its descendants are scrolled
    into view (AXScrollToVisible) until about ``pages`` viewports have
    passed. Returns None when nothing moved, so the caller can fall back to a
    wheel event.
    """
    px, py = point
    containers = sorted(
        (
            e
            for e in snapshot.get("elements", [])
            if all(
                isinstance(e.get(k), (int, float))
                for k in ("x", "y", "width", "height")
            )
            and e["width"] > 0
            and e["height"] > 0
            and e["x"] <= px <= e["x"] + e["width"]
            and e["y"] <= py <= e["y"] + e["height"]
        ),
        key=lambda e: e["width"] * e["height"],
    )
    vertical = direction in {"up", "down"}
    forward = direction in {"down", "right"}
    for container in containers[:6]:
        try:
            live = _live_element(
                snapshot, int(container["index"]), validate_point=False
            )
        except ComputerUseError:
            continue
        if live is None:
            continue
        frame = ax_driver._point_size(live)
        if frame is None:
            continue
        reach = (frame[3] if vertical else frame[2]) * max(pages, 0.1)
        moved = 0.0
        for _ in range(_AX_SCROLL_STEPS):
            picked = _ax_scroll_target(live, vertical, forward, reach - moved)
            if picked is not None:
                node = picked[2]
            elif not forward:
                # Content above or left with no nodes of its own: bringing
                # the scroller's first content box into view returns to its
                # start.
                children = ax_driver._as_list(ax_driver._get(live, "AXChildren"))
                if not children:
                    break
                node = children[0]
            else:
                break
            before = _ax_subtree_frames(live)
            ax_driver.AXUIElementPerformAction(node, "AXScrollToVisible")
            time.sleep(0.1)
            after = _ax_subtree_frames(live)
            if before == after:
                break
            # Clipped frames are pinned to the edge, so this underestimates
            # the distance; it still bounds the loop.
            step = max(
                (
                    abs((a[1] - b[1]) if vertical else (a[0] - b[0]))
                    for a, b in zip(after, before)
                ),
                default=1.0,
            )
            step = max(step, 1.0)
            moved += step
            if moved >= reach * 0.9:
                break
        if moved >= 1:
            return {"scroller": int(container["index"]), "points": round(moved)}
    return None


def scroll(
    app: str,
    direction: str,
    pages: float = 1.0,
    x: int | None = None,
    y: int | None = None,
    window_id: int | str | None = None,
    include_post_state: bool = False,
    expected_snapshot: dict | None = None,
) -> dict:
    if direction not in {"up", "down", "left", "right"}:
        raise ComputerUseError(
            "invalid_argument", f"unsupported direction {direction!r}"
        )
    snapshot = expected_snapshot or get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    point = (x, y) if x is not None and y is not None else _window_center(snapshot)
    lines = int(max(1, round(pages * 10)))
    delta = lines if direction in {"up", "left"} else -lines
    if _background_delivery(snapshot):
        # A wheel event routed to the exact window is hit-tested at the point,
        # so neither focus nor occlusion matters (and nested scrollers that
        # never take keyboard focus still scroll).
        window = _validate_snapshot_window(snapshot, point=point, require_topmost=False)
        pid, cg_window_id = _target_ids(snapshot)
        vertical = direction in {"up", "down"}
        if ax_driver.window_is_onscreen(cg_window_id) is False:
            # Chromium drops wheel events for a window the window server is
            # not compositing (another Space), so content is scrolled into
            # view through accessibility instead.
            moved = _ax_scroll(snapshot, point, direction, pages)
            if moved is not None:
                return _finish_action(
                    app,
                    snapshot,
                    {"mode": "AX-scroll", "direction": direction, **moved},
                    verified=True,
                    verification="scrolled content into view; its position changed",
                    include_post_state=include_post_state,
                )
        if not _synthesize(
            background_input.scroll,
            pid,
            cg_window_id,
            float(point[0]),
            float(point[1]),
            lines_y=delta if vertical else 0,
            lines_x=0 if vertical else delta,
            window_origin=_window_origin(window),
        ):
            raise ComputerUseError(
                "action_failed", "background scroll could not be synthesized"
            )
        return _finish_action(
            app,
            snapshot,
            {"mode": "SkyLight-scroll", "direction": direction, "lines": lines},
            verified=None,
            verification="synthetic scroll emitted; resulting position was not asserted",
            include_post_state=include_post_state,
        )
    _validate_snapshot_window(snapshot, point=point)
    _validate_focused_window(snapshot)
    import Quartz

    if direction in {"up", "down"}:
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, 1, delta
        )
    else:
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, 2, 0, delta
        )
    Quartz.CGEventSetLocation(event, point)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    time.sleep(0.05)
    return _finish_action(
        app,
        snapshot,
        {"mode": "CGEvent-scroll", "direction": direction, "lines": lines},
        verified=None,
        verification="synthetic scroll emitted; resulting position was not asserted",
        include_post_state=include_post_state,
    )


def perform_secondary_action(
    app: str,
    element_index: int,
    action: str,
    window_id: int | str | None = None,
    include_post_state: bool = False,
) -> dict:
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
    from ApplicationServices import AXUIElementPerformAction

    snapshot = get_app_state(
        app,
        screenshot=False,
        use_cache=False,
        window_id=window_id,
        activate=OBSERVE_BY_ROUTE,
    )
    entry = _element(snapshot, element_index)
    live = _live_element(snapshot, element_index)
    if live is None or action not in entry["actions"]:
        raise ComputerUseError(
            "value_not_settable",
            f"action {action!r} not advertised on element {element_index} "
            f"(advertised: {entry['actions']})",
        )
    menus_before = _menus_before(snapshot, None, expect_menu=action == "AXShowMenu")
    with _guard_user_focus(snapshot) as guard:
        err = AXUIElementPerformAction(live, action)
        # AXShowMenu and friends open menus; never leave one taking the keys,
        # also when the AX call reported an error after opening it.
        accepted = err == AS.kAXErrorSuccess or _menu_opened_despite_error(
            snapshot, menus_before
        )
        menu = (
            _settle_menus(
                snapshot,
                element_index,
                menus_before,
                None,
                expect_menu=action == "AXShowMenu",
            )
            if accepted
            else None
        )
    if not accepted:
        raise ComputerUseError(
            "accessibility_error", f"AXPerformAction {action} failed: {err}"
        )
    delivery: dict[str, Any] = {
        "mode": "AXPerformAction",
        "action": action,
        "element_index": element_index,
    }
    if err != AS.kAXErrorSuccess:
        delivery["ax_error"] = err
    if menu is not None:
        delivery["menu"] = menu
    delivery.update(_focus_fields(guard))
    return _finish_action(
        app,
        snapshot,
        delivery,
        verified=None,
        verification="action accepted by Accessibility; outcome not asserted",
        include_post_state=include_post_state,
    )


def permissions() -> dict:
    """Report accessibility + screen-recording TCC status."""
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
    import Quartz

    trusted = AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: False})
    preflight: bool | None = True
    try:
        preflight = bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:  # noqa: BLE001 - older macOS without the API
        preflight = None
    return {
        "accessibility": bool(trusted),
        "screen_recording": preflight,
        "hints": [
            "Grant Accessibility for keystroke injection, window control, UI automation.",
            "Grant Screen Recording for screenshots and visual verification.",
        ],
    }


def request_permission(permission: str) -> dict:
    """Request one CUA permission after an explicit authenticated POST."""
    if permission not in {"accessibility", "screen_recording"}:
        raise ValueError(f"unsupported computer use permission: {permission}")

    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817
    import Quartz

    # Permission sheets are process-global. Do not race two authenticated
    # clients into concurrent system prompts.
    with _permission_request_lock:
        try:
            if permission == "accessibility":
                _ = bool(
                    AS.AXIsProcessTrustedWithOptions(
                        {AS.kAXTrustedCheckOptionPrompt: True}
                    )
                )
            else:
                _ = bool(Quartz.CGRequestScreenCaptureAccess())
            current = permissions()
        except Exception as exc:
            raise ComputerUseError(
                "permission_request_failed",
                f"macOS could not request {permission.replace('_', ' ')} permission",
                recovery=(
                    "Open System Settings > Privacy & Security and review the Computer Use helper permission.",
                ),
            ) from exc

    # Request APIs may return before a Settings change is observable. Never
    # promote the grant above a fresh preflight from this same helper process.
    return {
        "permission": permission,
        "granted": current.get(permission) is True,
        "permissions": current,
    }


def list_apps() -> list[dict]:
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional

    apps = []
    for running in ax_driver._running_applications(AS):
        if running.activationPolicy() != 0:  # regular apps only
            continue
        apps.append(
            {
                "name": running.localizedName(),
                "bundleId": running.bundleIdentifier(),
                "pid": int(running.processIdentifier()),
            }
        )
    return apps


def discover_target_windows(*, limit: int = 24) -> list[dict]:
    """Return a bounded, front-to-back catalog of regular app windows.

    This is discovery only. Run creation still resolves every PID and opaque
    window ID again before granting the planner any authority.
    """
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    excluded = {
        "com.rapidmlx.rapid",
        "com.rapidmlx.rapid.computer-use",
        "com.apple.dock",
        "com.apple.controlcenter",
        "com.apple.notificationcenterui",
        "com.apple.systemuiserver",
    }
    apps: dict[int, dict] = {}
    for app in ax_driver._running_applications(AS):
        if (
            app.activationPolicy() != 0
            or str(app.bundleIdentifier() or "").casefold() in excluded
        ):
            continue
        info = _resolved_app_info(app)
        info["name"] = app.localizedName()
        info["bundleId"] = app.bundleIdentifier()
        apps[int(app.processIdentifier())] = info
    raw = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    catalog: list[dict] = []
    for z_order, window in enumerate(raw or []):
        pid = int(window.get("kCGWindowOwnerPID", -1))
        app = apps.get(pid)
        if app is None or int(window.get("kCGWindowLayer", 99)) != 0:
            continue
        number = window.get("kCGWindowNumber")
        bounds = window.get("kCGWindowBounds") or {}
        width = bounds.get("Width")
        height = bounds.get("Height")
        if (
            number is None
            or not isinstance(width, (int, float))
            or not isinstance(height, (int, float))
        ):
            continue
        if width <= 1 or height <= 1:
            continue
        catalog.append(
            {
                "catalog_id": f"w{len(catalog) + 1}",
                "app": app,
                "window": {
                    "window_id": f"cg:{int(number)}",
                    "title": str(window.get("kCGWindowName") or "")[:200],
                    "x": bounds.get("X"),
                    "y": bounds.get("Y"),
                    "width": width,
                    "height": height,
                },
                "z_order": z_order,
            }
        )
        if len(catalog) >= limit:
            break
    return catalog


_AUTOMATION_INITIAL_TIMEOUT_S = 30
_AUTOMATION_STEADY_TIMEOUT_S = 5
_AUTOMATION_READY_BUNDLES: set[str] = set()


def read_url(
    app: str,
    window_id: int | str | None = None,
    *,
    require_permission: bool = False,
    allow_background_app: bool = False,
) -> str:
    """Read the browser's active-tab URL from a trusted application API.

    Accessibility values are page-controlled and must never authorize a domain
    guard. Automation permission or browser support failures return an empty
    URL; callers with an allowed-domain policy fail closed on that value.
    """
    try:
        _, app_info = _resolve_app(app, activate=False)
        window = _select_window(app_info, window_id=window_id)
        if window["index"] != 0:
            return ""
        selected = {
            "app": app_info,
            "window": window,
            "window_id": window["window_id"],
        }
        if allow_background_app:
            # Resolver-only read: Rapid is foreground because Start was
            # clicked. Still require the exact selected item to be this
            # browser process's own front/focused window before asking its
            # trusted Automation API for the front-tab URL.
            _validate_focused_window(selected, require_active_app=False)
        else:
            _validate_focused_window(selected)
        bundle_id = str(app_info.get("bundleId") or "")
        if not re.fullmatch(r"[A-Za-z0-9.-]+", bundle_id):
            return ""
        bundle_key = bundle_id.lower()
        if app.startswith("pid:"):
            if ax_driver.AS is None:
                return ""
            same_bundle_pids = {
                int(running.processIdentifier())
                for running in ax_driver._running_applications()
                if (running.bundleIdentifier() or "").lower() == bundle_key
                and running.activationPolicy() == 0
            }
            if same_bundle_pids != {int(app_info["pid"])}:
                return ""
        if bundle_key in {
            "com.apple.safari",
            "com.apple.safaritechnologypreview",
        }:
            tab_property = "current tab"
        elif bundle_key.startswith(
            ("com.google.chrome", "com.microsoft.edgemac", "org.chromium.chromium")
        ):
            tab_property = "active tab"
        else:
            return ""
        timeout = (
            _AUTOMATION_STEADY_TIMEOUT_S
            if bundle_key in _AUTOMATION_READY_BUNDLES or not require_permission
            else _AUTOMATION_INITIAL_TIMEOUT_S
        )
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'tell application id "{bundle_id}" to get URL of {tab_property} of front window',
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            if require_permission:
                _AUTOMATION_READY_BUNDLES.discard(bundle_key)
                raise ComputerUseError(
                    "automation_permission_required",
                    "browser URL access timed out while waiting for macOS "
                    "Automation authorization; allow Rapid-MLX to control the "
                    "selected browser in System Settings > Privacy & Security "
                    "> Automation, then retry",
                ) from exc
            return ""
        if result.returncode != 0:
            stderr = result.stderr or ""
            permission_denied = "(-1743)" in stderr or (
                "not authorized to send apple events" in stderr.lower()
            )
            if require_permission and permission_denied:
                _AUTOMATION_READY_BUNDLES.discard(bundle_key)
                raise ComputerUseError(
                    "automation_permission_required",
                    "browser URL access is not authorized; allow Rapid-MLX to "
                    "control the selected browser in System Settings > Privacy "
                    "& Security > Automation, then retry",
                )
            return ""
        _AUTOMATION_READY_BUNDLES.add(bundle_key)
        url = result.stdout.strip()
        if url.startswith(("http://", "https://")):
            return url
    except ComputerUseError as exc:
        if exc.code == "automation_permission_required":
            raise
        return ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def list_windows(app: str) -> list[dict]:
    _, app_info = _resolve_app(app, activate=False)
    out = _window_records(app_info)
    if not out:
        raise ComputerUseError(
            "window_not_found", f"{app_info['name']!r} has no on-screen windows"
        )
    return out


def validate_window(app: str, window_id: int | str) -> dict:
    """Resolve an opaque window ID against an app PID without activating it."""
    _, app_info = _resolve_app(app, activate=False)
    window = _select_window(app_info, window_id=window_id)
    return {"app": app_info, "window_id": window["window_id"], "window": window}
