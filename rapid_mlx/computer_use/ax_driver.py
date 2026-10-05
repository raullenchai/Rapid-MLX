"""Native macOS Accessibility element probe — DOM-free target enumeration.

Drives apps through the macOS Accessibility (AX) tree instead of injected
page scripts:

* enumerates actionable elements of any app (native OR browser content —
  Chrome exposes its web area through AX once manual accessibility is on),
* emits targets in the same shape the CUA planner protocol already consumes
  (t000..., label text, role, geometry),
* performs semantic actions (AXPress) with a CGEvent click fallback, so a
  planner never needs injected DOM ids.

Usage:
  python ax_driver.py --app "Google Chrome" --dump targets.json
  python ax_driver.py --app "Google Chrome" --press t003
  python ax_driver.py --app "Finder" --dump -            # print to stdout

POC quality: bounded walk, no incremental caching, no AXObserver yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from typing import Any, cast

try:
    import ApplicationServices as AS  # type: ignore[import-untyped]  # noqa: N817
    from ApplicationServices import (  # type: ignore[import-untyped]  # pragma: no cover
        AXUIElementCopyActionNames,
        AXUIElementCopyAttributeValue,
        AXUIElementCreateApplication,
        AXUIElementCreateSystemWide,
        AXUIElementPerformAction,
        AXUIElementSetAttributeValue,
        AXValueGetValue,
        kAXErrorSuccess,
    )
    from Quartz import (  # type: ignore[import-untyped]  # pragma: no cover
        CGEventCreateKeyboardEvent,
        CGEventCreateMouseEvent,
        CGEventKeyboardSetUnicodeString,
        CGEventPost,
        CGEventSetFlags,
        CGEventSetIntegerValueField,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventMouseMoved,
        kCGHIDEventTap,
        kCGMouseEventClickState,
    )
except ImportError:  # Linux CI and non-macOS clients may still import the package.
    AS = None  # type: ignore[assignment]

    def _macos_only(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("computer-use actions require macOS with PyObjC installed")

    AXUIElementCopyActionNames = _macos_only
    AXUIElementCopyAttributeValue = _macos_only
    AXUIElementCreateApplication = _macos_only
    AXUIElementCreateSystemWide = _macos_only
    AXUIElementPerformAction = _macos_only
    AXUIElementSetAttributeValue = _macos_only
    AXValueGetValue = _macos_only
    CGEventCreateKeyboardEvent = _macos_only
    CGEventCreateMouseEvent = _macos_only
    CGEventKeyboardSetUnicodeString = _macos_only
    CGEventPost = _macos_only
    CGEventSetFlags = _macos_only
    CGEventSetIntegerValueField = _macos_only
    kAXErrorSuccess = 0  # noqa: N816
    kCGEventLeftMouseDown = 0  # noqa: N816
    kCGEventLeftMouseUp = 0  # noqa: N816
    kCGEventMouseMoved = 0  # noqa: N816
    kCGHIDEventTap = 0  # noqa: N816
    kCGMouseEventClickState = 0  # noqa: N816

FLAG_COMMAND = 1 << 20  # kCGEventFlagMaskCommand
FLAG_NONE = 0

# macOS HID usage IDs (kVK_ANSI_*)
KEYCODE_MAP = {
    "a": 0x00,
    "s": 0x01,
    "d": 0x02,
    "f": 0x03,
    "h": 0x04,
    "g": 0x05,
    "z": 0x06,
    "x": 0x07,
    "c": 0x08,
    "v": 0x09,
    "b": 0x0B,
    "q": 0x0C,
    "w": 0x0D,
    "e": 0x0E,
    "r": 0x0F,
    "y": 0x10,
    "t": 0x11,
    "1": 0x12,
    "2": 0x13,
    "3": 0x14,
    "4": 0x15,
    "6": 0x16,
    "5": 0x17,
    "=": 0x18,
    "9": 0x19,
    "7": 0x1A,
    "-": 0x1B,
    "8": 0x1C,
    "0": 0x1D,
    "]": 0x1E,
    "o": 0x1F,
    "u": 0x20,
    "[": 0x21,
    "i": 0x22,
    "p": 0x23,
    "l": 0x25,
    "j": 0x26,
    "'": 0x27,
    "k": 0x28,
    ";": 0x29,
    "\\": 0x2A,
    ",": 0x2B,
    "/": 0x2C,
    "n": 0x2D,
    "m": 0x2E,
    ".": 0x2F,
}

# Chrome/Safari only surface web content to AX when an assistive client asks.
_MANUAL_ACCESSIBILITY = "AXManualAccessibility"
_ENHANCED_UI = "AXEnhancedUserInterface"

INTERESTING_ROLES = {
    "AXButton",
    "AXLink",
    "AXTextField",
    "AXTextArea",
    "AXComboBox",
    "AXCheckBox",
    "AXRadioButton",
    "AXPopUpButton",
    "AXMenuButton",
    "AXMenuItem",
    "AXTabGroup",
    "AXSlider",
    "AXSearchField",
}
EDITABLE_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
MAX_NODES = 600
MAX_DEPTH = 22
CLICKABLE_SUBSTRINGS = ("button", "link", "menuitem", "tab", "checkbox", "radio")
PRIORITY_CONTAINER_ROLES = {"AXToolbar", "AXTabGroup", "AXMenuBar"}
PRIORITY_CONTROL_ROLES = {
    "AXButton",
    "AXCheckBox",
    "AXComboBox",
    "AXMenuButton",
    "AXPopUpButton",
    "AXRadioButton",
    "AXSearchField",
    "AXTextField",
}
REPETITIVE_CONTAINER_ROLES = {"AXTable", "AXOutline", "AXList"}
# Sorting every row in a huge virtualized table would add an AX role lookup per
# child before collection. Small sibling groups cover window-level regions and
# toolbars without turning prioritization into an unbounded IPC pre-scan.
PRIORITY_SORT_MAX_CHILDREN = 64


def _get(element: object, attribute: str) -> object:
    err, value = AXUIElementCopyAttributeValue(element, attribute, None)
    return value if err == kAXErrorSuccess else None


# kAXErrorNoValue and kAXErrorAttributeUnsupported: the attribute is simply
# not there, which is not a failed read.
_AX_ABSENT_ERRORS = frozenset({-25212, -25205})


def _get_checked(element: object, attribute: str) -> tuple[bool, object]:
    """``(readable, value)``; ``readable`` is False only for a failed read.

    A missing or unsupported attribute reads as ``(True, None)``.
    """
    err, value = AXUIElementCopyAttributeValue(element, attribute, None)
    if err == kAXErrorSuccess:
        return True, value
    return err in _AX_ABSENT_ERRORS, None


def _as_list(raw: object) -> list[object]:
    """pyobjc returns AX arrays as NSMutableArray — iterable, never a python
    list. isinstance(raw, (list, tuple)) is always False for them and silently
    emptied every snapshot (dogfood 2026-09-27); normalize once here."""
    if raw is None:
        return []
    try:
        return list(cast(Iterable[object], raw))
    except TypeError:
        return []


def _app_windows(app: object) -> list[object]:
    """AXWindows plus the focused/main window and remotely resolved windows.

    For an app whose windows sit on another Space (e.g. behind a full-screen
    app) AXWindows is empty while AXFocusedWindow/AXMainWindow may still
    return one off-Space window. Others are reachable only by remote token;
    those already discovered by :func:`discover_remote_windows` are included.
    """
    from . import background_input

    windows = _as_list(_get(app, "AXWindows"))
    # Dedupe by CGWindowID: the same window reached through different
    # attributes (or by remote token) is not always CFEqual.
    have = {background_input.ax_window_id(w) for w in windows}
    extras = [_get(app, "AXFocusedWindow"), _get(app, "AXMainWindow")]
    pid = _pid_of(app)
    if pid is not None and pid in _REMOTE_IDS:
        extras += _remote_windows(pid)
    for extra in extras:
        if extra is None:
            continue
        wid = background_input.ax_window_id(extra)
        if wid is None:
            if not any(extra == w for w in windows):
                windows.append(extra)
        elif wid not in have:
            have.add(wid)
            windows.append(extra)
    return windows


def _pid_of(element: object) -> int | None:
    if AS is None:
        return None
    try:
        err, pid = AS.AXUIElementGetPid(element, None)
    except Exception:  # noqa: BLE001 - not an AX element
        return None
    return int(pid) if err == 0 else None


# --- remote-token windows (yabai) ---------------------------------------------
# AXWindows lists only windows on the active Space. An AX element can be rebuilt
# from (pid, element id) with the private _AXUIElementCreateWithRemoteToken;
# scanning element ids finds every window of the process. Discovered ids are
# cached per pid; a scan runs only when a CG window cannot be mapped otherwise.

_REMOTE_IDS: dict[int, set[int]] = {}
_REMOTE_MAGIC = 0x636F636F  # 'coco'
_remote_create: Any = None


def _remote_factory() -> Any:
    global _remote_create
    if _remote_create is None:
        import ctypes

        lib = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/"
            "Frameworks/HIServices.framework/HIServices"
        )
        cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        fn = getattr(lib, "_AXUIElementCreateWithRemoteToken", None)
        if fn is None:
            _remote_create = False
            return False
        fn.restype = ctypes.c_void_p
        fn.argtypes = [ctypes.c_void_p]
        cf.CFDataCreate.restype = ctypes.c_void_p
        cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        _remote_create = (fn, cf)
    return _remote_create


def _remote_element(pid: int, element_id: int) -> object | None:
    import struct

    import objc  # type: ignore[import-untyped]

    factory = _remote_factory()
    if not factory:
        return None
    fn, cf = factory
    token = struct.pack("<iiIQ", int(pid), 0, _REMOTE_MAGIC, int(element_id))
    data = cf.CFDataCreate(None, token, len(token))
    if not data:
        return None
    try:
        ref = fn(data)
    finally:
        cf.CFRelease(data)
    if not ref:
        return None
    # The create rule hands us a +1 reference; let the bridge own it.
    element: object = objc.objc_object(c_void_p=ref)
    cf.CFRelease(ref)
    return element


def _remote_windows(pid: int) -> list[object]:
    out = []
    for element_id in sorted(_REMOTE_IDS.get(pid, ())):
        element = _remote_element(pid, element_id)
        if element is not None and _get(element, "AXRole") == "AXWindow":
            out.append(element)
    return out


def discover_remote_windows(
    pid: int, wanted: set[int], *, budget_s: float = 2.0, limit: int = 400_000
) -> set[int]:
    """Scan element ids of ``pid`` for windows; returns the CG ids found.

    Stops once every id in ``wanted`` is found or the time budget runs out.
    """
    from . import background_input

    found: set[int] = set()
    ids = _REMOTE_IDS.setdefault(int(pid), set())
    deadline = time.monotonic() + budget_s
    for element_id in range(limit):
        if element_id % 64 == 0 and time.monotonic() > deadline:
            break
        element = _remote_element(pid, element_id)
        if element is None or _get(element, "AXRole") != "AXWindow":
            continue
        wid = background_input.ax_window_id(element)
        if wid:
            ids.add(element_id)
            found.add(wid)
            if wanted <= found:
                break
    return found


def _action_names(element: object) -> list[str]:
    err, names = AXUIElementCopyActionNames(element, None)
    return list(names) if err == kAXErrorSuccess and names else []


def _priority_children(element: object) -> list[object]:
    """Return stable, bounded region ordering with navigation before tables."""

    children = _as_list(_get(element, "AXChildren"))
    if len(children) < 2 or len(children) > PRIORITY_SORT_MAX_CHILDREN:
        return children

    def priority(child: object) -> int:
        role = _get(child, "AXRole")
        if role in PRIORITY_CONTAINER_ROLES:
            return 0
        if role in PRIORITY_CONTROL_ROLES:
            return 1
        if role in REPETITIVE_CONTAINER_ROLES:
            return 3
        return 2

    # Python's stable sort preserves the original AX order inside each region,
    # so repeated observations of an unchanged tree keep identical target IDs.
    return sorted(children, key=priority)


def _point_size(element: object) -> tuple[float, float, float, float] | None:
    pos = _get(element, "AXPosition")
    size = _get(element, "AXSize")
    if pos is None or size is None:
        return None
    try:
        _, point = AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        _, sz = AXValueGetValue(size, AS.kAXValueCGSizeType, None)
        return float(point.x), float(point.y), float(sz.width), float(sz.height)
    except Exception:  # pragma: no cover - pyobjc variants
        return None


def _label(element: object) -> str:
    for attribute in ("AXDescription", "AXTitle", "AXValue"):
        value = _get(element, attribute)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")[:160]
    return ""


def _walk(
    element: object,
    depth: int,
    out: list[dict],
    counter: list[int],
    roles_seen: set[str] | None = None,
) -> None:
    if depth > MAX_DEPTH or counter[0] >= MAX_NODES:
        return
    raw_role = _get(element, "AXRole")
    role = raw_role if isinstance(raw_role, str) else ""
    if roles_seen is not None:
        roles_seen.add(role)
    raw_subrole = _get(element, "AXSubrole")
    subrole = raw_subrole if isinstance(raw_subrole, str) else ""
    parent = _get(element, "AXParent")
    raw_parent_role = _get(parent, "AXRole") if parent is not None else None
    parent_role = raw_parent_role if isinstance(raw_parent_role, str) else ""
    # Never read AXDescription/AXTitle/AXValue from a secure field. Redacting
    # after _label() would already have copied a credential into process memory,
    # planner context, traces, or an HTTP observation.
    secure_text = role == "AXSecureTextField" or subrole == "AXSecureTextField"
    label = "[secure text redacted]" if secure_text else _label(element)
    actions = _action_names(element)
    geom = _point_size(element)
    actionable = "AXPress" in actions or "AXPick" in actions or "AXIncrement" in actions
    editable = role in EDITABLE_ROLES
    interesting = (
        actionable
        or role in INTERESTING_ROLES
        or (role == "AXStaticText" and label)
        or any(s in role.lower() for s in CLICKABLE_SUBSTRINGS)
    )
    # Empty editable controls still need a stable target and geometry so a
    # planner can fill a blank document or form. Other empty structural nodes
    # remain excluded to preserve the bounded grounding budget.
    if interesting and (label or actionable or editable):
        counter[0] += 1
        value = None
        if editable and not secure_text:
            raw_value = _get(element, "AXValue")
            if isinstance(raw_value, str) and raw_value.strip() != label.strip():
                value = raw_value[:120]
        out.append(
            {
                "target_id": f"t{counter[0] - 1:03d}",
                "role": role,
                "subrole": subrole,
                "parent_role": parent_role,
                "text": label,
                "value": value,
                "actions": actions[:6],
                "rect": geom,
                "element": element,  # live ref, popped before serialization
            }
        )
    for child in _priority_children(element):
        _walk(child, depth + 1, out, counter, roles_seen)
        if counter[0] >= MAX_NODES:
            return


def _application_for_pid(
    pid: int, application_services: Any | None = None
) -> Any | None:
    services = application_services or AS
    if services is None:
        return None
    application_class = getattr(services, "NSRunningApplication", None)
    resolver = getattr(
        application_class, "runningApplicationWithProcessIdentifier_", None
    )
    if resolver is not None:
        app = resolver(pid)
    else:
        app = next(
            (
                candidate
                for candidate in services.NSWorkspace.sharedWorkspace().runningApplications()
                if int(candidate.processIdentifier()) == pid
            ),
            None,
        )
    is_terminated = getattr(app, "isTerminated", None)
    return None if callable(is_terminated) and is_terminated() else app


def _running_applications(application_services: Any | None = None) -> list[Any]:
    """Return live applications without relying on NSWorkspace notifications.

    The model-free sidecar has no AppKit event loop.  A long-lived
    ``NSWorkspace.runningApplications`` collection can therefore miss apps
    launched after the process.  CoreGraphics supplies a current PID set;
    resolving those PIDs through ``NSRunningApplication`` refreshes identity
    while the workspace PIDs preserve windowless-app compatibility.
    """
    services = application_services or AS
    if services is None:
        return []
    workspace_apps = list(services.NSWorkspace.sharedWorkspace().runningApplications())
    cached = {int(app.processIdentifier()): app for app in workspace_apps}
    pids = set(cached)
    try:
        import Quartz  # type: ignore[import-untyped]  # noqa: N813

        records = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
        )
        pids.update(
            int(record.get("kCGWindowOwnerPID", 0))
            for record in records or []
            if int(record.get("kCGWindowOwnerPID", 0)) > 0
        )
    except Exception:  # noqa: BLE001 - optional CG refresh must retain workspace fallback
        pass

    applications = []
    resolver = getattr(
        getattr(services, "NSRunningApplication", None),
        "runningApplicationWithProcessIdentifier_",
        None,
    )
    for pid in sorted(pids):
        app = (
            _application_for_pid(pid, services)
            if resolver is not None
            else cached.get(pid)
        )
        if app is None:
            continue
        is_terminated = getattr(app, "isTerminated", None)
        if callable(is_terminated) and is_terminated():
            continue
        applications.append(app)
    return applications


class AppNotFoundError(LookupError):
    """No running process matches the requested app (it may have exited).

    A ``LookupError`` rather than ``SystemExit``: this module also runs inside
    the long-lived server, where a ``BaseException`` would sail past every
    ``except Exception`` and take down the worker or event loop.
    """


# Processes already initialised, keyed by (kind, pid, launch time) so a
# recycled pid is handled again. "attrs": the AXManualAccessibility /
# AXEnhancedUserInterface poke was sent (it sticks for the process lifetime;
# only the first poke needs the settle wait). "settle": the plain first-touch
# settle wait ran (no attributes set).
_EXPOSED: set[tuple[str, int, float]] = set()


def _launch_time(app: Any) -> float | None:
    try:
        launched = float(app.launchDate().timeIntervalSince1970())
    except Exception:  # noqa: BLE001 - some bridges omit launchDate
        return None
    return launched if launched > 0 else None


def first_exposure(app: Any, kind: str = "attrs") -> bool:
    """True the first time ``app``'s process is seen for ``kind`` (then
    remembered). Without a launch time the process cannot be told apart from
    a recycled pid, so it is never remembered."""
    launched = _launch_time(app)
    if launched is None:
        return True
    key = (kind, int(app.processIdentifier()), launched)
    if key in _EXPOSED:
        return False
    _EXPOSED.add(key)
    return True


def _app_element(app_name: str, expected_pid: int | None = None) -> object:
    if AS is None:
        raise RuntimeError("computer-use actions require macOS with PyObjC installed")
    # Dogfood find (2026-09-27): substring matching picked up system XPC
    # helpers whose localized name merely contains the app name (e.g.
    # "ThemeWidgetControlViewService (Rapid)"), yielding an AX element with
    # no windows and empty snapshots for every app. Prefer exact-name matches
    # over Apple's own XPC processes, then verify the element has windows.
    candidates: list[Any] = []
    exact: list[Any] = []
    wanted = app_name.lower()
    applications = (
        [_application_for_pid(expected_pid)]
        if expected_pid is not None
        else _running_applications()
    )
    for app in applications:
        if app is None:
            continue
        if expected_pid is not None and int(app.processIdentifier()) != expected_pid:
            continue
        name = (app.localizedName() or "").lower()
        if wanted not in name:
            continue
        (exact if name == wanted else candidates).append(app)
    matches = exact + candidates
    for app in matches:
        element = AXUIElementCreateApplication(app.processIdentifier())
        # Chrome builds web-content AX trees lazily; ask it to expose them.
        # Dogfood find (2026-09-27): the same poke on AppKit/SwiftUI apps
        # retriggers an AX tree rebuild whose in-flight state hides deep
        # children (windows look fine, walks come back empty), so restrict
        # the poke to Chrome-family apps.
        bundle = (app.bundleIdentifier() or "").lower()
        if "chrome" in bundle or "chromium" in bundle or "edge" in bundle:
            if first_exposure(app):
                AXUIElementSetAttributeValue(element, _MANUAL_ACCESSIBILITY, True)
                AXUIElementSetAttributeValue(element, _ENHANCED_UI, True)
                time.sleep(0.4)
        elif first_exposure(app, "settle"):
            time.sleep(0.1)
        # Window verification only matters when several processes matched
        # (an XPC helper could shadow the real app); with a single candidate
        # take it as-is — an AX-unresponsive app should still be selectable.
        if len(matches) == 1 or _app_windows(element):
            return element
    suffix = f" with pid {expected_pid}" if expected_pid is not None else ""
    raise AppNotFoundError(f"app not found: {app_name!r}{suffix}")


def collect(
    app_name: str,
    keep_elements: bool = False,
    max_windows: int = 3,
    window_index: int = 0,
    window_frame: tuple[float, float, float, float] | None = None,
    window_frame_tolerance: float = 0.5,
    expected_pid: int | None = None,
    retry_web_content: bool = False,
    partial_out: list[dict] | None = None,
) -> list[dict]:
    if window_index < 0:
        raise ValueError("window_index must be non-negative")
    app = (
        _app_element(app_name, expected_pid=expected_pid)
        if expected_pid is not None
        else _app_element(app_name)
    )
    targets: list[dict] = partial_out if partial_out is not None else []
    counter = [0]
    attempts = 4 if retry_web_content else 1
    for attempt in range(attempts):
        windows = _app_windows(app)
        if window_frame is not None:
            matching = []
            for window in windows:
                frame = _point_size(window)
                if frame is not None:
                    ax_x, ax_y, ax_w, ax_h = frame
                    cg_x, cg_y, cg_w, cg_h = window_frame
                    edges_match = all(
                        abs(actual - expected) <= window_frame_tolerance
                        for actual, expected in (
                            (ax_x, cg_x),
                            (ax_y, cg_y),
                            (ax_x + ax_w, cg_x + cg_w),
                            (ax_y + ax_h, cg_y + cg_h),
                        )
                    )
                    if edges_match:
                        matching.append(window)
            if len(matching) != 1:
                raise RuntimeError(
                    "selected CGWindow does not map to exactly one AX window"
                )
            selected_windows = matching
        else:
            selected_windows = windows[window_index : window_index + max_windows]
        targets.clear()
        counter = [0]
        roles_seen: set[str] = set()
        for window in selected_windows:
            _walk(window, 0, targets, counter, roles_seen)
            if counter[0] >= MAX_NODES:
                break
        # AXWebArea is structural (never a target), so look at what was walked.
        if (
            not retry_web_content
            or "AXWebArea" in roles_seen
            or attempt == attempts - 1
        ):
            break
        # Chrome builds the web-content AX tree lazily; _app_element has
        # already enabled manual accessibility for Chromium apps. Repeating
        # that write here restarts the tree build, and writing it for native
        # apps can temporarily hide their deep children.
        time.sleep(1.5)
    if not targets and window_index == 0 and window_frame is None:  # menu-bar-only apps
        _walk(AXUIElementCreateSystemWide(), 0, targets, counter)
    if not keep_elements:
        for entry in targets:
            entry.pop("element", None)
    for entry in targets:
        rect = entry.get("rect")
        if rect:
            entry["center"] = [
                round(rect[0] + rect[2] / 2),
                round(rect[1] + rect[3] / 2),
            ]
    return targets


# (down, up, CGMouseButton) per button; right/other values are the stable
# kCGEventRightMouseDown/Up and kCGEventOtherMouseDown/Up enum constants.
_HID_BUTTONS = {
    "left": (kCGEventLeftMouseDown, kCGEventLeftMouseUp, 0),
    "right": (3, 4, 1),
    "middle": (25, 26, 2),
}
_K_CG_MOUSE_EVENT_BUTTON_NUMBER = 3


def _cg_click(x: float, y: float, clicks: int = 1, button: str = "left") -> None:
    down_type, up_type, button_number = _HID_BUTTONS[button]
    move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), 0)
    CGEventPost(kCGHIDEventTap, move)
    time.sleep(0.05)
    for click_state in range(1, max(1, clicks) + 1):
        down = CGEventCreateMouseEvent(None, down_type, (x, y), button_number)
        up = CGEventCreateMouseEvent(None, up_type, (x, y), button_number)
        for event in (down, up):
            CGEventSetIntegerValueField(event, kCGMouseEventClickState, click_state)
            CGEventSetIntegerValueField(
                event, _K_CG_MOUSE_EVENT_BUTTON_NUMBER, button_number
            )
        CGEventPost(kCGHIDEventTap, down)
        time.sleep(0.03)
        CGEventPost(kCGHIDEventTap, up)
        if click_state < clicks:
            time.sleep(0.08)


def _press_key(keycode: int, modifiers: int = FLAG_NONE) -> None:
    down = CGEventCreateKeyboardEvent(None, keycode, True)
    up = CGEventCreateKeyboardEvent(None, keycode, False)
    if modifiers:
        CGEventSetFlags(down, modifiers)
        CGEventSetFlags(up, modifiers)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(0.02)
    CGEventPost(kCGHIDEventTap, up)


def _keycode_for(key: str) -> int:
    return KEYCODE_MAP[key]


def _type_text(text: str) -> None:
    """Type text with real HID keycodes (Chromium drops unicode-string events
    on web content); explicit per-event flags prevent sticky modifiers.
    Non-keymap characters (CJK, symbols) fall back to unicode-string events."""
    for ch in text:
        lower = ch.lower()
        if lower in KEYCODE_MAP:
            keycode = KEYCODE_MAP[lower]
            flags = (1 << 17) if ch.isupper() else FLAG_NONE  # kCGEventFlagMaskShift
            down = CGEventCreateKeyboardEvent(None, keycode, True)
            up = CGEventCreateKeyboardEvent(None, keycode, False)
            CGEventSetFlags(down, flags)
            CGEventSetFlags(up, flags)
            CGEventPost(kCGHIDEventTap, down)
            time.sleep(0.02)
            CGEventPost(kCGHIDEventTap, up)
            time.sleep(0.02)
        else:
            down = CGEventCreateKeyboardEvent(None, 0, True)
            up = CGEventCreateKeyboardEvent(None, 0, False)
            try:
                CGEventKeyboardSetUnicodeString(down, len(ch), ch)
                CGEventKeyboardSetUnicodeString(up, len(ch), ch)
            except TypeError:  # older pyobjc: (event, string) form
                CGEventKeyboardSetUnicodeString(down, ch)
                CGEventKeyboardSetUnicodeString(up, ch)
            CGEventPost(kCGHIDEventTap, down)
            time.sleep(0.02)
            CGEventPost(kCGHIDEventTap, up)


def press(targets: list[dict], target_id: str, app_name: str) -> dict:
    for entry in targets:
        if entry["target_id"] != target_id:
            continue
        live = _app_element(app_name)
        # Re-resolve by re-walking (live refs are not serialized with --dump).
        fresh: list[dict] = []
        counter = [0]
        windows = _app_windows(live)
        for window in windows:
            _walk(window, 0, fresh, counter)
            if any(f["target_id"] == target_id for f in fresh):
                break
        match = next((f for f in fresh if f["target_id"] == target_id), None)
        if match is None:
            return {
                "ok": False,
                "error": f"{target_id} no longer present (tree shifted)",
            }
        live_element = match.get("element")
        if live_element is not None and "AXPress" in match["actions"]:
            err = AXUIElementPerformAction(live_element, "AXPress")
            if err == kAXErrorSuccess:
                return {"ok": True, "mode": "AXPress", "target": target_id}
        rect = match.get("rect")
        if not rect or rect[2] <= 0 or rect[3] <= 0:
            return {"ok": False, "error": "no AXPress and no usable geometry"}
        _cg_click(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        return {"ok": True, "mode": "CGEvent-click", "target": target_id}
    return {"ok": False, "error": f"unknown target_id {target_id!r}"}


def main() -> None:
    global MAX_NODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="running app name substring")
    parser.add_argument(
        "--dump", default=None, help="write targets JSON here ('-' prints)"
    )
    parser.add_argument("--press", default=None, help="AXPress/click a tNNN target id")
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    args = parser.parse_args()

    MAX_NODES = args.max_nodes

    try:
        _cli(args)
    except AppNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


def _cli(args: argparse.Namespace) -> None:
    targets = collect(args.app)
    if args.dump:
        payload = [{k: v for k, v in t.items() if k != "element"} for t in targets]
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        if args.dump == "-":
            print(text)
        else:
            with open(args.dump, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"{len(payload)} targets -> {args.dump}", file=sys.stderr)
    if args.press:
        result = press(targets, args.press, args.app)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    if not args.dump and not args.press:
        for entry in targets[:40]:
            print(entry["target_id"], entry["role"], entry["text"][:60])


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
