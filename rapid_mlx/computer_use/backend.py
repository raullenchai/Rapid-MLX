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
import time
import uuid
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

from .errors import ComputerUseError

SNAPSHOT_TTL_S = 120.0
AX_COLLECT_TIMEOUT_S = 20.0
FILL_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
SNAPSHOT_CACHE_MAX = 32
MAX_TRANSIENT_TARGETS = 64
from . import ax_driver

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


def _needs_web_content_retry(app_info: dict) -> bool:
    """Only Chromium-family AX trees need the lazy web-content retry loop."""

    bundle = str(app_info.get("bundleId") or app_info.get("bundle_id") or "").lower()
    return any(marker in bundle for marker in ("chrome", "chromium", "edge"))


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
    except Exception:  # noqa: BLE001 - older bridges may omit launchDate
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
    for window in raw or []:
        if int(window.get("kCGWindowOwnerPID", -1)) != int(app_info["pid"]):
            continue
        if int(window.get("kCGWindowLayer", 99)) != 0:
            continue
        number = window.get("kCGWindowNumber")
        bounds = window.get("kCGWindowBounds") or {}
        if number is None:
            continue
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
    return records


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


def _focused_ax_window(app_info: dict) -> object | None:
    app_element = ax_driver._app_element(
        app_info["name"], expected_pid=int(app_info["pid"])
    )
    app_windows = ax_driver._as_list(ax_driver._get(app_element, "AXWindows"))
    focused = ax_driver._get(app_element, "AXFocusedWindow")
    if focused is None:
        focused = ax_driver._get(app_element, "AXFocusedUIElement")
        if focused is not None and any(focused == window for window in app_windows):
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
    if focused is None or not any(focused == window for window in app_windows):
        return None
    return focused


def _focused_ax_element(app_info: dict) -> object | None:
    """Return the exact focused control for the PID-bound app."""

    app_element = ax_driver._app_element(
        app_info["name"], expected_pid=int(app_info["pid"])
    )
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
    frame = ax_driver._point_size(focused) if focused is not None else None
    if frame is None:
        return None
    anchor_frame = tuple(float(anchor[key]) for key in ("x", "y", "width", "height"))
    if _ax_cg_frames_match(frame, anchor_frame):
        return None
    matches = []
    for candidate in _window_records(app_info):
        candidate_frame = tuple(
            float(candidate[key]) for key in ("x", "y", "width", "height")
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
    snapshot: dict, *, point: tuple[float, float] | None = None
) -> dict:
    """Fail closed when an observation is old or its exact CGWindow drifted."""
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
        if _topmost_window_id_at(x, y) != _cg_window_id(expected["window_id"]):
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


def get_app_state(
    app: str,
    window_index: int = 0,
    screenshot: bool = True,
    use_cache: bool = True,
    window_id: int | str | None = None,
    *,
    activate: bool = True,
    trusted_transient_window_id: int | str | None = None,
    transient_baseline_window_ids: set[str] | None = None,
) -> dict:
    """Snapshot one window: elements with indexes, tree text, optional PNG."""
    # Keep the historical action-path behavior by default. Read-only callers
    # can explicitly forbid activation; this is a security boundary because an
    # observation request must not steal focus or expose a different window.
    ax_element, app_info = (
        _resolve_app(app) if activate else _resolve_app(app, activate=False)
    )
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
                "label": target["text"],
                "actions": target.get("actions", []),
                "x": round(rect[0]),
                "y": round(rect[1]),
                "width": round(rect[2]),
                "height": round(rect[3]),
                "center": target.get("center")
                or [round(rect[0] + rect[2] / 2), round(rect[1] + rect[3] / 2)],
                "source_window_id": target["source_window_id"],
            }
        )
    tree_lines = [
        f"[{e['index']}] {e['role']}{'*' if 'AXPress' in e['actions'] else ''} {e['label'][:90]}"
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
        "visible_window_ids": [
            str(record["window_id"]) for record in _window_records(app_info)
        ],
    }
    if transient_window is not None:
        snapshot["transient_window"] = transient_window
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
            )
        except SystemExit as exc:
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
    return cast(list[dict], outcome.get("value", []))


def _live_element(
    snapshot: dict, element_index: int, *, validate_point: bool = True
) -> object:
    """Re-collect and return the live AX ref for an index, if still present."""
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
        current_window = _validate_snapshot_window(
            snapshot, point=point if validate_point else None
        )
    fresh = _collect_with_timeout(
        snapshot["app"]["name"],
        window_index=current_window["index"],
        window=current_window,
        expected_pid=int(snapshot["app"]["pid"]),
        retry_web_content=_needs_web_content_retry(snapshot["app"]),
        window_frame_tolerance=4.0 if is_transient else 0.5,
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
            return target.get("element")
    raise ComputerUseError(
        "element_not_found",
        f"element {element_index} no longer present in the fresh AX tree",
    )


def _read_value(live_element: object) -> str | None:
    value = ax_driver._get(live_element, "AXValue")
    return value if isinstance(value, str) else None


def _finish_action(
    app: str,
    snapshot: dict,
    result: dict,
    *,
    verified: bool | None,
    verification: str,
    include_post_state: bool = False,
) -> dict:
    """Return honest outcome metadata plus a best-effort fresh local state."""
    result.update(
        {
            "attempted": True,
            "verified": verified,
            "verification": verification,
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


def _validate_focused_window(
    snapshot: dict, expected_window: dict | None = None
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
        active = frontmost is not None and int(frontmost.processIdentifier()) == expected_pid
    if not active:
        raise ComputerUseError(
            "target_drift",
            f"pid {snapshot['app']['pid']} is no longer frontmost; re-observe",
        )
    focused = _focused_ax_window(snapshot["app"])
    focused_frame = ax_driver._point_size(focused) if focused is not None else None
    expected = expected_window or snapshot["window"]
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


def raise_selected_window(app: str, snapshot: dict) -> dict:
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
    for key in ("pid", "bundleId", "name"):
        wanted = expected_app.get(key)
        if wanted is not None and app_info.get(key) != wanted:
            raise ComputerUseError("target_drift", "selected app identity changed")
    current = _select_window(app_info, window_id=expected["window_id"])
    if not _same_window(expected, current):
        raise ComputerUseError("target_drift", "selected window moved before recovery")
    frame = tuple(float(current[key]) for key in ("x", "y", "width", "height"))
    matches = [
        window
        for window in ax_driver._as_list(ax_driver._get(app_element, "AXWindows"))
        if (candidate := ax_driver._point_size(window)) is not None
        and _ax_cg_frames_match(candidate, frame)
    ]
    if len(matches) != 1 or "AXRaise" not in ax_driver._action_names(matches[0]):
        raise ComputerUseError(
            "target_occluded",
            "selected window cannot be raised unambiguously; move the covering window",
        )
    err = ax_driver.AXUIElementPerformAction(matches[0], "AXRaise")
    if err != ax_driver.kAXErrorSuccess:
        raise ComputerUseError("target_occluded", "selected window rejected AXRaise")
    time.sleep(0.2)
    after = _select_window(app_info, window_id=expected["window_id"])
    if not _same_window(expected, after):
        raise ComputerUseError(
            "target_drift", "selected window changed during recovery"
        )
    _validate_focused_window(snapshot)
    return after


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

    _validate_snapshot_window(snapshot)
    _validate_focused_window(snapshot)
    app_info = snapshot["app"]
    app_element = ax_driver._app_element(
        str(app_info["name"]), expected_pid=int(app_info["pid"])
    )
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
) -> dict:
    if element_index is not None:
        snapshot = expected_snapshot or get_app_state(
            app, screenshot=False, use_cache=False, window_id=window_id
        )
        entry = _element(snapshot, element_index)
        is_transient = entry.get(
            "source_window_id", snapshot.get("window_id")
        ) != snapshot.get("window_id")
        center = entry["center"]
        live = None
        if (
            "AXPress" in entry["actions"]
            or is_transient
            or (focus_only and entry.get("role") in FILL_ROLES)
        ):
            # Menus and popovers can be owned by the selected app/window while
            # appearing outside the window's content bounds. Revalidate the
            # exact window and AX target identity, then prefer AXPress on that
            # live object. Coordinate fallbacks remain bounded below.
            live = _live_element(snapshot, element_index, validate_point=False)
        if live is not None and "AXPress" in entry["actions"]:
            import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
            from ApplicationServices import AXUIElementPerformAction

            err = AXUIElementPerformAction(live, "AXPress")
            if err == AS.kAXErrorSuccess:
                return _finish_action(
                    app,
                    snapshot,
                    {"mode": "AXPress", "element_index": element_index},
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
            _validate_focused_window(snapshot, expected_window)
            return _finish_action(
                app,
                snapshot,
                {"mode": "AXFocusVerified", "element_index": element_index},
                verified=True,
                verification="exact Accessibility element remained focused",
                include_post_state=include_post_state,
            )
        if is_transient:
            raise ComputerUseError(
                "synthetic_input_blocked",
                "transient companion target is not exactly pressable or focused",
            )
        _validate_snapshot_window(snapshot, point=tuple(center))
        ax_driver._cg_click(float(center[0]), float(center[1]), clicks=click_count)
        return _finish_action(
            app,
            snapshot,
            {"mode": "CGEvent-click", "element_index": element_index, "at": center},
            verified=None,
            verification="synthetic click emitted; outcome not asserted",
            include_post_state=include_post_state,
        )
    if x is None or y is None:
        raise ComputerUseError(
            "invalid_argument", "click requires --element-index or both --x and --y"
        )
    snapshot = expected_snapshot or get_app_state(
        app, screenshot=False, use_cache=False, window_id=window_id
    )
    _validate_snapshot_window(snapshot, point=(x, y))
    ax_driver._cg_click(float(x), float(y), clicks=click_count)
    return _finish_action(
        app,
        snapshot,
        {"mode": "CGEvent-click", "at": [x, y]},
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
        app, screenshot=False, use_cache=False, window_id=window_id
    )
    entry = _element(snapshot, element_index)
    is_transient = entry.get(
        "source_window_id", snapshot.get("window_id")
    ) != snapshot.get("window_id")
    live = _live_element(snapshot, element_index, validate_point=not is_transient)
    if live is not None:
        from ApplicationServices import (  # type: ignore[import-untyped]
            AXUIElementSetAttributeValue,
            kAXValueAttribute,
        )

        err = AXUIElementSetAttributeValue(live, kAXValueAttribute, value)
        if err == 0:
            readback = _read_value(live)
            if readback == value:
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


def _synthetic_fill(snapshot: dict, element_index: int, value: str) -> dict:
    entry = _element(snapshot, element_index)
    center = entry["center"]
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
) -> dict:
    snapshot = expected_snapshot or get_app_state(
        app, screenshot=False, use_cache=False, window_id=window_id
    )
    if element_index is not None:
        entry = _element(snapshot, element_index)
        if entry.get("source_window_id", snapshot.get("window_id")) != snapshot.get(
            "window_id"
        ):
            live = _live_element(snapshot, element_index, validate_point=False)
            focused_element = _focused_ax_window(snapshot["app"])
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
                if topmost_id not in same_app_ids:
                    raise ComputerUseError(
                        "target_occluded",
                        "selected window is covered by another process at keyboard dispatch",
                    )
            else:
                expected_window = _validate_snapshot_window(
                    snapshot, point=_window_center(snapshot)
                )
    else:
        expected_window = _validate_snapshot_window(
            snapshot, point=_window_center(snapshot)
        )
    _validate_focused_window(snapshot, expected_window)
    return snapshot


def type_text(
    app: str,
    text: str,
    window_id: int | str | None = None,
    include_post_state: bool = False,
) -> dict:
    snapshot = _prepare_synthetic_action(app, window_id)
    ax_driver._type_text(text)
    return _finish_action(
        app,
        snapshot,
        {"mode": "CGEvent-unicode", "characters": len(text)},
        verified=None,
        verification="synthetic text emitted; focused value was not readable",
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
    if expected_snapshot is not None and element_index is not None:
        entry = _element(expected_snapshot, element_index)
        is_transient = entry.get(
            "source_window_id", expected_snapshot.get("window_id")
        ) != expected_snapshot.get("window_id")
        if is_transient and normalized != "enter":
            raise ComputerUseError(
                "unsupported_key",
                "transient companion targets only allow Enter after exact focus validation",
            )
    if normalized in KEY_ALIASES:
        snapshot = _prepare_synthetic_action(
            app,
            window_id,
            expected_snapshot,
            element_index,
            allow_focused_editable_enter=normalized == "enter",
        )
        ax_driver._press_key(KEY_ALIASES[normalized])
        return _finish_action(
            app,
            snapshot,
            {"mode": "CGEvent-keycode", "key": normalized},
            verified=None,
            verification="synthetic key emitted; outcome not asserted",
            include_post_state=include_post_state,
        )
    if normalized in ax_driver.KEYCODE_MAP:
        snapshot = _prepare_synthetic_action(
            app, window_id, expected_snapshot, element_index
        )
        ax_driver._press_key(ax_driver._keycode_for(normalized))
        return _finish_action(
            app,
            snapshot,
            {"mode": "CGEvent-keycode", "key": normalized},
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
    snapshot = _prepare_synthetic_action(app, window_id)
    import Quartz

    down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
    up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
    Quartz.CGEventSetFlags(down, modifiers)
    Quartz.CGEventSetFlags(up, modifiers)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    time.sleep(0.02)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
    return _finish_action(
        app,
        snapshot,
        {"mode": "CGEvent-hotkey", "key": key},
        verified=None,
        verification="synthetic hotkey emitted; outcome not asserted",
        include_post_state=include_post_state,
    )


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
        app, screenshot=False, use_cache=False, window_id=window_id
    )
    point = (x, y) if x is not None and y is not None else _window_center(snapshot)
    _validate_snapshot_window(snapshot, point=point)
    _validate_focused_window(snapshot)
    import Quartz

    lines = int(max(1, round(pages * 10)))
    delta = lines if direction in {"up", "left"} else -lines
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
        app, screenshot=False, use_cache=False, window_id=window_id
    )
    entry = _element(snapshot, element_index)
    live = _live_element(snapshot, element_index)
    if live is None or action not in entry["actions"]:
        raise ComputerUseError(
            "value_not_settable",
            f"action {action!r} not advertised on element {element_index} "
            f"(advertised: {entry['actions']})",
        )
    err = AXUIElementPerformAction(live, action)
    if err != AS.kAXErrorSuccess:
        raise ComputerUseError(
            "accessibility_error", f"AXPerformAction {action} failed: {err}"
        )
    return _finish_action(
        app,
        snapshot,
        {"mode": "AXPerformAction", "action": action, "element_index": element_index},
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


def read_url(app: str, window_id: int | str | None = None) -> str:
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
        _validate_focused_window(
            {"app": app_info, "window": window, "window_id": window["window_id"]}
        )
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
        result = subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application id "{bundle_id}" to get URL of {tab_property} of front window',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
        url = result.stdout.strip()
        if url.startswith(("http://", "https://")):
            return url
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
