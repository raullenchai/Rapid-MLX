"""Native macOS backend for the model-agnostic computer-use tool suite.

Wraps the AX probe (tools/gui_verifier_cua_poc/ax_driver.py primitives) into
an Orca-shaped surface: snapshot caches keyed by app/window with a TTL,
element actions that accept an element index (semantic) or coordinates
(fallback), direct AX value writes with read-back verification before
falling back to synthetic typing, and typed errors with recovery hints.

The layer is deliberately model-free: any agent (cloud, local GLM, local
Qwen) drives the machine through these calls.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from .errors import ComputerUseError

SNAPSHOT_TTL_S = 120.0
FILL_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
SNAPSHOT_CACHE_MAX = 32
AXDRIVER_DIR = Path(__file__).resolve().parents[2] / "tools" / "gui_verifier_cua_poc"

import sys  # noqa: E402

if str(AXDRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(AXDRIVER_DIR))

import ax_driver  # noqa: E402

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


def _resolve_app(app: str) -> tuple[object, dict]:
    """Find a running app by name substring, bundle id, or pid:N."""
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional

    workspace = AS.NSWorkspace.sharedWorkspace()
    wanted_pid = None
    if app.startswith("pid:"):
        try:
            wanted_pid = int(app.split(":", 1)[1])
        except ValueError as exc:
            raise ComputerUseError(
                "invalid_argument", f"bad pid spec: {app!r}"
            ) from exc
    lowered = app.lower()
    for running in workspace.runningApplications():
        if (
            not running.isActive()
            and wanted_pid is None
            and running.activationPolicy() != 0
        ):
            continue
        name = (running.localizedName() or "").lower()
        bundle = (running.bundleIdentifier() or "").lower()
        pid = int(running.processIdentifier())
        if wanted_pid is not None:
            if pid == wanted_pid:
                return _ax_app_element(running), {
                    "name": name,
                    "bundleId": bundle,
                    "pid": pid,
                }
            continue
        if lowered in (name, bundle) or lowered in name or lowered in bundle:
            return _ax_app_element(running), {
                "name": name,
                "bundleId": bundle,
                "pid": pid,
            }
    raise ComputerUseError("app_not_found", f"no running app matches {app!r}")


def _ax_app_element(running_app: object, activate: bool = True) -> object:
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
        self._entries: dict[tuple[str, int], tuple[float, dict]] = {}

    def get(self, app: str, window_index: int) -> dict | None:
        entry = self._entries.get((app, window_index))
        if entry is None:
            return None
        created, snapshot = entry
        if time.time() - created > SNAPSHOT_TTL_S:
            self._entries.pop((app, window_index), None)
            return None
        return snapshot

    def put(self, app: str, window_index: int, snapshot: dict) -> None:
        self._entries[(app, window_index)] = (time.time(), snapshot)
        if len(self._entries) > SNAPSHOT_CACHE_MAX:
            oldest = min(self._entries, key=lambda key: self._entries[key][0])
            self._entries.pop(oldest, None)

    def clear(self) -> None:
        self._entries.clear()


_CACHE = SnapshotCache()


def get_app_state(
    app: str, window_index: int = 0, screenshot: bool = True, use_cache: bool = True
) -> dict:
    """Snapshot one window: elements with indexes, tree text, optional PNG."""
    if use_cache:
        cached = _CACHE.get(app, window_index)
        if cached is not None:
            return cached
    ax_element, app_info = _resolve_app(app)
    targets = ax_driver.collect(
        app_info["name"] or app, keep_elements=True, max_windows=1
    )
    # collect() enumerates all windows; emulate per-window slicing cheaply by
    # keeping the first window's worth (POC) and flag truncation.
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
            }
        )
    tree_lines = [
        f"[{e['index']}] {e['role']}{'*' if 'AXPress' in e['actions'] else ''} {e['label'][:90]}"
        for e in elements
    ]
    snapshot = {
        "snapshot_id": f"{app_info['pid']}:{window_index}:{int(time.time())}",
        "app": app_info,
        "window_index": window_index,
        "coordinate_space": "screen",
        "elements": elements,
        "element_count": len(elements),
        "tree_text": "\n".join(tree_lines),
        "truncated": len(elements) >= ax_driver.MAX_NODES,
    }
    png = (
        screenshot_window(app_info["name"] or app, window_index) if screenshot else None
    )
    if png is not None:
        snapshot["screenshot_png"] = png
    _CACHE.put(app, window_index, snapshot)
    return snapshot


def screenshot_window(app_name: str, window_index: int = 0) -> bytes | None:
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
    candidates = []
    for window in windows:
        owner = str(window.get("kCGWindowOwnerName", ""))
        if app_name.lower() not in owner.lower():
            continue
        if window.get("kCGWindowLayer", 99) != 0:
            continue
        bounds = window.get("kCGWindowBounds", {})
        candidates.append(
            (
                bounds.get("Width", 0) * bounds.get("Height", 0),
                window.get("kCGWindowNumber"),
            )
        )
    if not candidates:
        raise ComputerUseError(
            "window_not_found", f"no on-screen window for {app_name!r}"
        )
    candidates.sort(reverse=True)
    window_number = candidates[min(window_index, len(candidates) - 1)][1]
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
            return entry
    raise ComputerUseError(
        "element_not_found",
        f"element {element_index} is not in the current snapshot "
        f"(snapshot {snapshot.get('snapshot_id')})",
    )


def _live_element(snapshot: dict, element_index: int) -> object:
    """Re-collect and return the live AX ref for an index, if still present."""
    fresh = ax_driver.collect(
        snapshot["app"]["name"], keep_elements=True, max_windows=1
    )
    for target in fresh:
        if int(target["target_id"][1:]) == element_index:
            return target.get("element")
    raise ComputerUseError(
        "element_not_found",
        f"element {element_index} no longer present in the fresh AX tree",
    )


def _read_value(live_element: object) -> str | None:
    value = ax_driver._get(live_element, "AXValue")
    return value if isinstance(value, str) else None


def click(
    app: str,
    element_index: int | None = None,
    x: int | None = None,
    y: int | None = None,
    click_count: int = 1,
    mouse_button: str = "left",
) -> dict:
    snapshot = get_app_state(app, use_cache=False)
    if element_index is not None:
        entry = _element(snapshot, element_index)
        center = entry["center"]
        live = None
        try:
            live = _live_element(snapshot, element_index)
        except ComputerUseError:
            live = None
        if live is not None and "AXPress" in entry["actions"]:
            import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
            from ApplicationServices import AXUIElementPerformAction

            err = AXUIElementPerformAction(live, "AXPress")
            if err == AS.kAXErrorSuccess:
                return {"mode": "AXPress", "element_index": element_index}
        ax_driver._cg_click(float(center[0]), float(center[1]), clicks=click_count)
        return {"mode": "CGEvent-click", "element_index": element_index, "at": center}
    if x is None or y is None:
        raise ComputerUseError(
            "invalid_argument", "click requires --element-index or both --x and --y"
        )
    ax_driver._cg_click(float(x), float(y), clicks=click_count)
    return {"mode": "CGEvent-click", "at": [x, y]}


def set_value(app: str, element_index: int, value: str) -> dict:
    """Write a value into a settable element; verify by reading it back.

    Falls back to synthetic typing (click, Cmd+A, delete, CGEvent unicode) when
    the element rejects direct AX writes. The returned verification field tells
    the agent whether the value landed exactly.
    """
    from ApplicationServices import (  # noqa: N813  # camelcase pyobjc module, alias is conventional
        AXUIElementSetAttributeValue,
        kAXValueAttribute,
    )

    snapshot = get_app_state(app, use_cache=False)
    _element(snapshot, element_index)
    live = _live_element(snapshot, element_index)
    if live is not None:
        err = AXUIElementSetAttributeValue(live, kAXValueAttribute, value)
        if err == 0:
            readback = _read_value(live)
            if readback == value:
                return {
                    "mode": "AXSetValue",
                    "element_index": element_index,
                    "verified": True,
                }
    # AX write either failed or did not land; degrade to synthetic typing.
    return _synthetic_fill(snapshot, element_index, value)


def _synthetic_fill(snapshot: dict, element_index: int, value: str) -> dict:
    entry = _element(snapshot, element_index)
    center = entry["center"]
    ax_driver._cg_click(float(center[0]), float(center[1]))
    time.sleep(0.3)
    ax_driver._press_key(ax_driver._keycode_for("a"), modifiers=ax_driver.FLAG_COMMAND)
    time.sleep(0.1)
    ax_driver._press_key(KEY_ALIASES["delete"])
    ax_driver._type_text(value)
    time.sleep(0.4)
    # Verify by VALUE, not index: typing can open suggestion dropdowns and the
    # tree rebuilds with shifted indexes (Orca solves this with runtimeIds).
    try:
        fresh = ax_driver.collect(
            snapshot["app"]["name"], keep_elements=True, max_windows=1
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


def type_text(app: str, text: str) -> dict:
    ax_driver._type_text(text)
    return {"mode": "CGEvent-unicode", "characters": len(text)}


def press_key(app: str, key: str) -> dict:
    normalized = key.strip().lower()
    if normalized in KEY_ALIASES:
        ax_driver._press_key(KEY_ALIASES[normalized])
        return {"mode": "CGEvent-keycode", "key": normalized}
    if normalized in ax_driver.KEYCODE_MAP:
        ax_driver._press_key(ax_driver._keycode_for(normalized))
        return {"mode": "CGEvent-keycode", "key": normalized}
    raise ComputerUseError("unsupported_key", f"unsupported single key {key!r}")


def hotkey(app: str, key: str) -> dict:
    """Modifier chord like 'Cmd+A', 'Ctrl+Shift+Tab'."""
    import Quartz

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
    down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
    up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
    Quartz.CGEventSetFlags(down, modifiers)
    Quartz.CGEventSetFlags(up, modifiers)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    time.sleep(0.02)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
    return {"mode": "CGEvent-hotkey", "key": key}


def scroll(
    app: str,
    direction: str,
    pages: float = 1.0,
    x: int | None = None,
    y: int | None = None,
) -> dict:
    import Quartz

    if direction not in {"up", "down", "left", "right"}:
        raise ComputerUseError(
            "invalid_argument", f"unsupported direction {direction!r}"
        )
    lines = int(max(1, round(pages * 10)))
    sign = -1 if direction in {"up", "left"} else 1
    events = [(0, sign * lines)] if direction in {"up", "down"} else [(1, sign * lines)]
    if x is not None and y is not None:
        ax_driver._cg_click(float(x), float(y))  # position pointer for scroll target
    for axis, delta in events:
        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, axis, delta
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.05)
    return {"mode": "CGEvent-scroll", "direction": direction, "lines": lines}


def perform_secondary_action(app: str, element_index: int, action: str) -> dict:
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
    from ApplicationServices import AXUIElementPerformAction

    snapshot = get_app_state(app, use_cache=False)
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
    return {"mode": "AXPerformAction", "action": action, "element_index": element_index}


def permissions() -> dict:
    """Report accessibility + screen-recording TCC status."""
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N813, N817  # camelcase pyobjc module, alias is conventional
    import Quartz

    trusted = AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: False})
    preflight = True
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

    workspace = AS.NSWorkspace.sharedWorkspace()
    apps = []
    for running in workspace.runningApplications():
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


def list_windows(app: str) -> list[dict]:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    _, app_info = _resolve_app(app)
    windows = CGWindowListCopyWindowInfo(
        kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    out = []
    index = 0
    for window in windows:
        if (
            str(window.get("kCGWindowOwnerName", "")).lower()
            != app_info["name"].lower()
        ):
            continue
        if window.get("kCGWindowLayer", 99) != 0:
            continue
        bounds = window.get("kCGWindowBounds", {})
        out.append(
            {
                "index": index,
                "title": window.get("kCGWindowName") or "",
                "x": bounds.get("X"),
                "y": bounds.get("Y"),
                "width": bounds.get("Width"),
                "height": bounds.get("Height"),
            }
        )
        index += 1
    if not out:
        raise ComputerUseError(
            "window_not_found", f"{app_info['name']!r} has no on-screen windows"
        )
    return out
