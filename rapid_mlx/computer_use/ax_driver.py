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
# Shown instead of what the user typed into a field only they may fill
# (kept equal to guards.USER_VALUE).
USER_VALUE = "[entered by the user]"
# Controls whose AXValue is their state (a select's choice, a stepper's number).
VALUE_ROLES = {
    "AXPopUpButton",
    "AXIncrementor",
    "AXSlider",
    "AXProgressIndicator",
    "AXLevelIndicator",
    "AXValueIndicator",
}
# Web pages nest deep: a Walmart result grid sits ~30 levels under the window,
# and a whole results page is ~800 nodes.
MAX_NODES = 4000
MAX_DEPTH = 60
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


def window_is_onscreen(window_id: int) -> bool | None:
    """Whether the window server is compositing ``window_id`` right now.

    False for a window on another Space or minimized; None when unknown.
    """
    try:
        from Quartz import (  # type: ignore[import-untyped]
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionIncludingWindow,
        )

        info = CGWindowListCopyWindowInfo(kCGWindowListOptionIncludingWindow, window_id)
    except Exception:  # pragma: no cover - pyobjc variants
        return None
    if not info:
        return None
    return bool(info[0].get("kCGWindowIsOnscreen"))


def _wake_hidden_renderer(window: object) -> bool:
    """Make Chromium re-send a hidden window's web-content tree.

    A Chromium window that was already off screen (another Space, covered)
    when accessibility was first enabled never builds its web-content tree:
    the hidden renderer does not serialize it. A resize does (measured on
    Electron), so the window is grown by one point and put straight back.
    Only off-screen windows are touched; the user cannot see the change.
    """
    from . import background_input

    window_id = background_input.ax_window_id(window)
    frame = _point_size(window)
    if window_id is None or frame is None or window_is_onscreen(window_id) is not False:
        return False
    _, _, width, height = frame

    def resize(size: tuple[float, float]) -> bool:
        value = AS.AXValueCreate(AS.kAXValueCGSizeType, size)
        return bool(AS.AXUIElementSetAttributeValue(window, "AXSize", value) == 0)

    if not resize((width + 1.0, height)):
        return False
    # Never leave the user's window a point larger: retry the restore.
    for _ in range(3):
        if resize((width, height)):
            return True
        time.sleep(0.05)
    return False


# Batched reads per node (AXUIElementCopyMultipleAttributeValues) instead of
# one IPC round trip per attribute; Finder's file list walked 15.7k single
# reads (8.3 s) before this. Text attributes are a second batch, read only
# once the node is known not to be a secure field: a password's AXValue must
# never be copied into this process.
_WALK_ATTRIBUTES = (
    "AXRole",
    "AXSubrole",
    "AXPosition",
    "AXSize",
    "AXChildren",
    "AXVisibleRows",
    "AXEnabled",
    "AXFocused",
    "AXSelected",
    "AXExpanded",
)
_TEXT_ATTRIBUTES = ("AXDescription", "AXTitle", "AXValue", "AXPlaceholderValue")
# AX messaging timeouts are per element (not inherited), so each node gets one
# before its first read: a wedged node costs this, not the 6 s default.
NODE_MESSAGING_TIMEOUT_S = 0.25
# Whole-walk time budget; a walk that runs out returns what it has, marked
# truncated, instead of blocking the step.
WALK_BUDGET_S = 3.0


def _is_ax_error(value: object) -> bool:
    try:
        from CoreFoundation import CFGetTypeID  # type: ignore[import-untyped]

        return bool(
            CFGetTypeID(value) == AS.AXValueGetTypeID()
            and AS.AXValueGetType(value) == AS.kAXValueAXErrorType
        )
    except Exception:  # pragma: no cover - non-CF values
        return False


def _read_batch(element: object, attributes: tuple[str, ...]) -> dict[str, object]:
    try:
        err, values = AS.AXUIElementCopyMultipleAttributeValues(
            element, list(attributes), 0, None
        )
    except Exception:  # pyobjc variants / test doubles
        err, values = -1, None
    if err != kAXErrorSuccess or values is None or len(values) != len(attributes):
        return {attribute: _get(element, attribute) for attribute in attributes}
    return {
        attribute: (None if value is None or _is_ax_error(value) else value)
        for attribute, value in zip(attributes, values)
    }


# What a secure field may be asked for: its name, the element that labels it
# (a reference, not text) and how many characters it holds; never its value
# or title (some apps mirror the contents there).
_SECURE_NAME_ATTRIBUTES = (
    "AXDescription",
    "AXPlaceholderValue",
    "AXNumberOfCharacters",
    "AXTitleUIElement",
)
# A secure field's label element (an HTML <label>, a native text label) is
# read for its text only once its own role shows it holds no typed input.
_LABEL_KIND_ATTRIBUTES = ("AXRole", "AXSubrole")
_LABEL_TEXT_ATTRIBUTES = ("AXValue", "AXTitle", "AXDescription")
# A static text this long is a paragraph, not the name of the field after it.
MAX_FIELD_LABEL_CHARS = 80
# Web dialogs (role=dialog / alertdialog) and native sheets: the walk keeps
# them as rows even unnamed, so an observation can say one is open.
DIALOG_SUBROLES = frozenset(
    {"AXApplicationDialog", "AXApplicationAlertDialog", "AXDialog", "AXSystemDialog"}
)


def _is_secure_node(node: dict[str, object]) -> bool:
    return "AXSecureTextField" in (node.get("AXRole"), node.get("AXSubrole"))


def _read_node(element: object) -> dict[str, object]:
    """The walk's attributes for one element in two bounded requests."""
    try:
        AS.AXUIElementSetMessagingTimeout(element, NODE_MESSAGING_TIMEOUT_S)
    except Exception:  # pyobjc variants / test doubles
        pass
    node = _read_batch(element, _WALK_ATTRIBUTES)
    node.update(
        _read_batch(
            element,
            _SECURE_NAME_ATTRIBUTES if _is_secure_node(node) else _TEXT_ATTRIBUTES,
        )
    )
    return node


def _clean_name(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().replace("\n", " ")[:160]
    return ""


def _secure_name(element: object, node: dict[str, object]) -> str:
    """A secure field's name, read without touching its value or title.

    Its label element first (Chrome exposes an HTML <label> only there),
    then its description, then its placeholder.
    """
    return (
        _label_element_text(element, node.get("AXTitleUIElement"))
        or _clean_name(node.get("AXDescription"))
        or _clean_name(node.get("AXPlaceholderValue"))
    )


def _label_element_text(field: object, label: object) -> str:
    """The text of the element that labels ``field``.

    The label is a different element: its role is read first, and its text
    only when that role is not one that holds typed input (a field that
    names another field, or the field itself, is never read).
    """
    if label is None or label == field:
        return ""
    try:
        AS.AXUIElementSetMessagingTimeout(label, NODE_MESSAGING_TIMEOUT_S)
    except Exception:  # pyobjc variants / test doubles
        pass
    kind = _read_batch(label, _LABEL_KIND_ATTRIBUTES)
    role = kind.get("AXRole")
    if (
        not isinstance(role, str)
        or not role
        or role in EDITABLE_ROLES
        or _is_secure_node(kind)
    ):
        return ""
    text = _read_batch(label, _LABEL_TEXT_ATTRIBUTES)
    for attribute in _LABEL_TEXT_ATTRIBUTES:
        name = _clean_name(text.get(attribute))
        if name:
            return name
    return ""


def _preceding_label(out: list[dict], path: tuple[int, ...]) -> str:
    """The static text just before a field in its own group ("Password").

    Only when nothing else came between them (a label two controls up
    names something else), and only a short text in the field's parent.
    """
    if not out or out[-1].get("role") != "AXStaticText":
        return ""
    parent = list(path[:-1])
    text = str(out[-1].get("text") or "")
    if out[-1].get("path", [])[: len(parent)] != parent:
        return ""
    return text if 0 < len(text) <= MAX_FIELD_LABEL_CHARS else ""


def _secure_filled(node: dict[str, object]) -> bool | None:
    """Whether a secure field holds anything (None: it could not be read)."""
    count = node.get("AXNumberOfCharacters")
    if isinstance(count, int) and not isinstance(count, bool):
        return count > 0
    return None


def _frame_of(node: dict[str, object]) -> tuple[float, float, float, float] | None:
    pos, size = node.get("AXPosition"), node.get("AXSize")
    if pos is None or size is None:
        return None
    try:
        _, point = AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        _, sz = AXValueGetValue(size, AS.kAXValueCGSizeType, None)
        return float(point.x), float(point.y), float(sz.width), float(sz.height)
    except Exception:  # pragma: no cover - pyobjc variants
        return None


def _label_of(node: dict[str, object]) -> str:
    cap = 300 if node.get("AXRole") == "AXStaticText" else 160
    for attribute in ("AXDescription", "AXTitle", "AXValue"):
        value = node.get(attribute)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")[:cap]
    return ""


def _field_name(node: dict[str, object]) -> str:
    for attribute in ("AXDescription", "AXTitle", "AXPlaceholderValue"):
        value = node.get(attribute)
        if isinstance(value, str) and value.strip():
            return value.strip().replace("\n", " ")[:160]
    return ""


def _states_of(node: dict[str, object], role: str) -> list[str]:
    states = []
    if node.get("AXEnabled") is False:
        states.append("disabled")
    if node.get("AXFocused") is True:
        states.append("focused")
    if node.get("AXSelected") is True:
        states.append("selected")
    if node.get("AXExpanded") is True:
        states.append("expanded")
    if role in {"AXCheckBox", "AXRadioButton"}:
        value = node.get("AXValue")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            states.append(
                "checked" if value == 1 else "mixed" if value == 2 else "unchecked"
            )
    return states


def _child_elements(node: dict[str, object], role: str) -> list[object]:
    """Children to descend into; a list or table contributes only the rows on
    screen (AXVisibleRows), since scrolled-away rows cannot be acted on and a
    long list would otherwise eat the whole budget."""
    if role in REPETITIVE_CONTAINER_ROLES:
        rows = _as_list(node.get("AXVisibleRows"))
        if rows:
            return rows
    return _as_list(node.get("AXChildren"))


def _out_of_time(budget: dict[str, Any] | None) -> bool:
    if budget is None or time.monotonic() <= budget["deadline"]:
        return False
    budget["truncated"] = True
    budget["deadline_hit"] = True
    return True


def _walk(
    element: object,
    depth: int,
    out: list[dict],
    counter: list[int],
    roles_seen: set[str] | None = None,
    *,
    node: dict[str, object] | None = None,
    parent_role: str = "",
    budget: dict[str, Any] | None = None,
    in_web: bool = False,
    path: tuple[int, ...] = (),
) -> None:
    if depth > MAX_DEPTH or counter[0] >= MAX_NODES:
        if budget is not None:
            budget["truncated"] = True  # never drop content silently
            if depth > MAX_DEPTH:
                budget["depth_cap"] = True
        return
    if _out_of_time(budget):
        return
    if node is None:
        node = _read_node(element)
    raw_role = node.get("AXRole")
    role = raw_role if isinstance(raw_role, str) else ""
    if roles_seen is not None:
        roles_seen.add(role)
    raw_subrole = node.get("AXSubrole")
    subrole = raw_subrole if isinstance(raw_subrole, str) else ""
    # Never read AXDescription/AXTitle/AXValue from a secure field. Redacting
    # after _label_of() would already have copied a credential into process memory,
    # planner context, traces, or an HTTP observation.
    secure_text = _is_secure_node(node)
    label = "[secure text redacted]" if secure_text else _label_of(node)
    # A text field without a name of its own is labelled by its value, so a
    # field only the user may fill is named by its description, title or
    # placeholder, and what was typed into it is never shown.
    user_only = False
    if role in EDITABLE_ROLES and not secure_text:
        # Imported here so `python ax_driver.py` keeps working as a script.
        from . import guards

        name = _field_name(node)
        if name and guards.needs_human_input(role, subrole, name):
            label, user_only = name, True
    actions = _action_names(element)
    geom = _frame_of(node)
    actionable = "AXPress" in actions or "AXPick" in actions or "AXIncrement" in actions
    editable = role in EDITABLE_ROLES
    dialog = role == "AXSheet" or (role == "AXGroup" and subrole in DIALOG_SUBROLES)
    interesting = (
        actionable
        or dialog
        or role in INTERESTING_ROLES
        or (role == "AXStaticText" and label)
        or any(s in role.lower() for s in CLICKABLE_SUBSTRINGS)
    )
    # Empty editable controls still need a stable target and geometry so a
    # planner can fill a blank document or form. Other empty structural nodes
    # remain excluded to preserve the bounded grounding budget.
    if interesting and (label or actionable or editable or dialog):
        counter[0] += 1
        value = None
        if (editable or role in VALUE_ROLES) and not secure_text:
            raw_value = node.get("AXValue")
            if user_only:
                value = USER_VALUE if isinstance(raw_value, str) and raw_value else ""
            elif isinstance(raw_value, (int, float)) and not isinstance(
                raw_value, bool
            ):
                value = f"{raw_value:g}"
            elif (
                isinstance(raw_value, str)
                and raw_value.strip() != label.strip()
                and (editable or raw_value.strip())
            ):
                value = raw_value[:120]
        target: dict[str, Any] = {
            "target_id": f"t{counter[0] - 1:03d}",
            "role": role,
            "subrole": subrole,
            "parent_role": parent_role,
            "text": label,
            "value": value,
            "actions": actions[:6],
            "rect": geom,
            "states": _states_of(node, role),
            "placeholder": (
                node.get("AXPlaceholderValue")
                if isinstance(node.get("AXPlaceholderValue"), str)
                else None
            ),
            # Where it sits: child positions from the walk root, and whether
            # it is page content (inside a web area) or the app's own chrome.
            "path": list(path),
            "web": in_web or role == "AXWebArea",
            "element": element,  # live ref, popped before serialization
        }
        if secure_text:
            # Its name and whether it holds anything; never what it holds.
            target["field_name"] = _secure_name(element, node) or (
                _preceding_label(out, path)
            )
            target["filled"] = _secure_filled(node)
        out.append(target)
    children = _child_elements(node, role)
    # A web page is laid out in document order; sorting its controls ahead of
    # their labels and text would scramble what the model reads.
    in_web = in_web or role == "AXWebArea"
    if not in_web and 2 <= len(children) <= PRIORITY_SORT_MAX_CHILDREN:
        # Read small sibling groups up front: the reads are needed anyway and
        # their roles order navigation before tables.
        read: list[tuple[int, object, dict[str, object]]] = []
        for position, child in enumerate(children):
            if _out_of_time(budget):
                return  # wedged siblings must not outlast the walk budget
            read.append((position, child, _read_node(child)))

        def priority(entry: tuple[int, object, dict[str, object]]) -> int:
            child_role = entry[2].get("AXRole")
            if child_role in PRIORITY_CONTAINER_ROLES:
                return 0
            if child_role in PRIORITY_CONTROL_ROLES:
                return 1
            if child_role in REPETITIVE_CONTAINER_ROLES:
                return 3
            return 2

        # Python's stable sort preserves the original AX order inside each
        # region, so repeated observations of an unchanged tree keep
        # identical target IDs.
        ordered: list[tuple[int, object, dict[str, object] | None]] = [
            *sorted(read, key=priority)
        ]
    else:
        ordered = [(position, child, None) for position, child in enumerate(children)]
    for position, child, child_node in ordered:
        _walk(
            child,
            depth + 1,
            out,
            counter,
            roles_seen,
            node=child_node,
            parent_role=role,
            budget=budget,
            in_web=in_web,
            path=(*path, position),
        )
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
# recycled pid is handled again, mapped to the monotonic time it was first
# seen (the readiness clock :func:`exposure_age` reads). "attrs": the
# AXManualAccessibility / AXEnhancedUserInterface poke was sent (it sticks for
# the process lifetime; only the first poke needs the settle wait). "settle":
# the plain first-touch settle wait ran (no attributes set).
_EXPOSED: dict[tuple[str, int, float], float] = {}


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
    _EXPOSED[key] = time.monotonic()
    return True


_WOKEN_PIDS: set[int] = set()


def _mark_woken(app_element: object) -> None:
    pid = _pid_of(app_element)
    if pid is not None:
        _WOKEN_PIDS.add(pid)


def clear_woken(pid: int) -> None:
    _WOKEN_PIDS.discard(pid)


def renderer_was_woken(pid: int) -> bool:
    """True when this process's web content had to be woken while hidden."""
    return pid in _WOKEN_PIDS


def _restart_exposure(app_element: object) -> None:
    """Restart the readiness clock of the process behind ``app_element``."""
    pid = _pid_of(app_element)
    if pid is None:
        return
    for key in [k for k in _EXPOSED if k[1] == pid]:
        _EXPOSED[key] = time.monotonic()


def exposure_age(pid: int) -> float | None:
    """Seconds since this process's accessibility was first switched on here."""
    started = [t for (_, p, _), t in _EXPOSED.items() if p == pid]
    return time.monotonic() - max(started) if started else None


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
    budget_s: float | None = None,
    walk_status: dict[str, Any] | None = None,
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
        budget: dict[str, Any] = {
            "deadline": time.monotonic()
            + (WALK_BUDGET_S if budget_s is None else budget_s),
            "truncated": False,
        }
        for window in selected_windows:
            _walk(window, 0, targets, counter, roles_seen, budget=budget)
            if counter[0] >= MAX_NODES:
                break
        if walk_status is not None:
            walk_status["budget_exhausted"] = bool(budget.get("deadline_hit"))
            walk_status["depth_cap"] = bool(budget.get("depth_cap"))
            walk_status["node_cap"] = counter[0] >= MAX_NODES
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
        woken = attempt == 0 and any(
            [_wake_hidden_renderer(window) for window in selected_windows]
        )
        if woken:
            # The woken renderer serves its tree but drops AX actions until
            # it handles one renderer-side action (see backend).
            _restart_exposure(app)
            _mark_woken(app)
        time.sleep(0.4 if woken else 1.5)
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
