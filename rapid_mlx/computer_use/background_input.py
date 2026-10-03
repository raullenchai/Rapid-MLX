"""Non-intrusive ("background") synthetic input for the computer-use layer.

Semantic Accessibility actions (``AXPress``, AX value writes) already leave the
user's cursor and frontmost app alone. Everything else -- pixel clicks on
content without an AX tree, wheel scrolls, keystrokes -- historically went
through the global HID tap (``CGEventPost(kCGHIDEventTap, ...)``), which warps
the cursor, requires the target to be frontmost and fights the user for input.

This module routes those events to *one window of one process* instead, so the
agent can operate a window while the user keeps working elsewhere:

* mouse events are posted with the private ``SLEventPostToPid`` (which tickles
  the WindowServer activity monitor that the public ``CGEventPostToPid`` skips)
  after stamping the private CGEvent routing fields (target pid, window number)
  and making the target AppKit-active *without raising it*;
* keyboard events carry an ``SLSEventAuthenticationMessage`` (macOS 15+) so
  Chromium/Electron accept them as live input; menu key equivalents are posted
  without it because only the IOHIDPostEvent path reaches ``NSMenu``.

The recipes are ported from trycua/cua (MIT)
``libs/cua-driver/rust/crates/platform-macos/src/input/{skylight,mouse,keyboard}.rs``;
the focus-without-raise record originates in yabai (MIT). SIP may stay
enabled: the SPIs are reached by ``dlopen`` of the system PrivateFramework.
TCC Accessibility grants must be attributed to the hosting signed bundle.

Everything degrades gracefully: if a required symbol does not resolve,
``skylight_available()`` is ``False`` and callers keep the HID path. Event
*plans* are pure functions so they are unit-testable on any platform.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_double,
    c_int32,
    c_int64,
    c_long,
    c_size_t,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass

__all__ = [
    "DELIVERY_ENV",
    "MouseStep",
    "activate_without_raise",
    "background_enabled",
    "click",
    "click_plan",
    "delivery_mode",
    "front_process_matches",
    "keyboard_auth_available",
    "press_key",
    "restore_focus_after_without_raise",
    "scroll",
    "scroll_ticks",
    "skylight_available",
    "type_text",
]

DELIVERY_ENV = "RAPID_MLX_CUA_INPUT_DELIVERY"
_DELIVERY_MODES = ("auto", "background", "foreground")

# --- CGEvent constants -------------------------------------------------------
_K_HID_SYSTEM_STATE = 1  # kCGEventSourceStateHIDSystemState
_MOUSE_MOVED = 5
_EVENT_TYPES = {
    "left": (1, 2),  # kCGEventLeftMouseDown / Up
    "right": (3, 4),  # kCGEventRightMouseDown / Up
    "middle": (25, 26),  # kCGEventOtherMouseDown / Up
}
_BUTTON_NUMBER = {"left": 0, "right": 1, "middle": 2}
_SCROLL_UNIT_LINE = 1
_MAX_LINES_PER_TICK = 10

# Raw CGEvent field indexes (see cua-driver mouse.rs for provenance).
_F_PHASE = 0  # kCGMouseEventNumber, reused as gesture phase marker
_F_CLICK_STATE = 1  # kCGMouseEventClickState
_F_BUTTON = 3  # kCGMouseEventButtonNumber
_F_SUBTYPE = 7  # kCGMouseEventSubtype (3 = NSEventSubtypeTouch)
_F_TARGET_PID = 40  # kCGEventTargetUnixProcessID; Chromium's synthetic filter
_F_WINDOW = 51  # windowNumber
_F_CLICK_GROUP = 58  # gesture coalescing id
_F_WINDOW_UNDER = 91  # kCGMouseEventWindowUnderMousePointer
_F_WINDOW_HANDLER = 92  # ...ThatCanHandleThisEvent

_KEY_GAP_S = 0.008


@dataclass(frozen=True)
class MouseStep:
    """One event of a planned background mouse gesture."""

    event_type: int
    x: float
    y: float
    phase: int
    click_state: int
    button_number: int
    delay_after_s: float


def click_plan(
    x: float, y: float, *, button: str = "left", count: int = 1
) -> list[MouseStep]:
    """Pure event plan for a background click (no OS calls).

    Left clicks use cua's Chromium-safe recipe: a stamped ``mouseMoved``
    primer, an off-screen ``(-1, -1)`` down/up that satisfies Chromium's
    user-activation gate without hitting any DOM node, then the real
    down/up pair(s) with clickState 1..N. Right/middle clicks keep the primer
    but skip the decoy, and stamp the matching button number -- a right-down
    stamped as button 0 is delivered as a left click.
    """
    if button not in _EVENT_TYPES:
        raise ValueError(f"unsupported mouse button {button!r}")
    count = max(1, min(int(count), 3))
    down, up = _EVENT_TYPES[button]
    number = _BUTTON_NUMBER[button]
    steps = [MouseStep(_MOUSE_MOVED, x, y, 2, 0, 0, 0.015)]
    if button == "left":
        steps += [
            MouseStep(down, -1.0, -1.0, 1, 1, 0, 0.001),
            MouseStep(up, -1.0, -1.0, 2, 1, 0, 0.100),
        ]
    for index in range(1, count + 1):
        steps.append(MouseStep(down, x, y, 3, index, number, 0.028))
        steps.append(
            MouseStep(up, x, y, 3, index, number, 0.080 if index < count else 0.0)
        )
    return steps


def scroll_ticks(lines: int) -> list[int]:
    """Split a signed line delta into per-notch wheel deltas (pure).

    Discrete notches let renderers animate instead of coalescing one jump and
    keep each event inside the ±10-line range apps handle reliably.
    """
    lines = int(lines)
    if lines == 0:
        return []
    sign = 1 if lines > 0 else -1
    remaining = abs(lines)
    ticks = []
    while remaining > 0:
        step = min(_MAX_LINES_PER_TICK, remaining)
        ticks.append(sign * step)
        remaining -= step
    return ticks


def delivery_mode() -> str:
    """Configured delivery mode: ``auto`` (default), ``background`` or ``foreground``."""
    mode = os.environ.get(DELIVERY_ENV, "auto").strip().lower()
    return mode if mode in _DELIVERY_MODES else "auto"


def background_enabled() -> bool:
    """True when synthetic input should be routed to the target process."""
    return delivery_mode() != "foreground" and skylight_available()


# --- symbol loading ----------------------------------------------------------


class _CGPoint(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


class _PSN(Structure):  # ProcessSerialNumber: two UInt32
    _fields_ = [("hi", c_uint32), ("lo", c_uint32)]


def _bind(lib, name: str, argtypes: list, restype) -> object | None:
    fn = getattr(lib, name, None)
    if fn is None:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


def _load() -> dict | None:
    """Resolve the CoreGraphics + SkyLight symbols we need; None if unavailable."""
    try:
        cg = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        sky = ctypes.CDLL(
            "/System/Library/PrivateFrameworks/SkyLight.framework/SkyLight"
        )
        cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
    except OSError:
        return None
    try:
        hiservices = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
    except OSError:
        hiservices = None
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    except OSError:
        objc = None

    s: dict = {}
    required = {
        "source_create": _bind(cg, "CGEventSourceCreate", [c_uint32], c_void_p),
        "mouse_event": _bind(
            cg,
            "CGEventCreateMouseEvent",
            [c_void_p, c_uint32, _CGPoint, c_uint32],
            c_void_p,
        ),
        "key_event": _bind(
            cg, "CGEventCreateKeyboardEvent", [c_void_p, c_uint16, c_bool], c_void_p
        ),
        "set_unicode": _bind(
            cg,
            "CGEventKeyboardSetUnicodeString",
            [c_void_p, c_long, POINTER(c_uint16)],
            None,
        ),
        "set_location": _bind(cg, "CGEventSetLocation", [c_void_p, _CGPoint], None),
        "set_flags": _bind(cg, "CGEventSetFlags", [c_void_p, c_uint64], None),
        "public_post": _bind(cg, "CGEventPostToPid", [c_int32, c_void_p], None),
        "release": _bind(cf, "CFRelease", [c_void_p], None),
        "malloc_size": _bind(
            ctypes.CDLL("/usr/lib/libSystem.B.dylib"),
            "malloc_size",
            [c_void_p],
            c_size_t,
        ),
        "sl_post": _bind(sky, "SLEventPostToPid", [c_int32, c_void_p], None),
        "set_field": _bind(
            sky, "SLEventSetIntegerValueField", [c_void_p, c_uint32, c_int64], None
        ),
        "post_record": _bind(
            sky, "SLPSPostEventRecordTo", [c_void_p, c_void_p], c_int32
        ),
        "front_process": _bind(sky, "_SLPSGetFrontProcess", [POINTER(_PSN)], c_int32),
    }
    if any(value is None for value in required.values()):
        return None
    s.update(required)

    # The scroll constructor is variadic; bind the 2-wheel form explicitly.
    scroll_ctor = getattr(cg, "CGEventCreateScrollWheelEvent2", None)
    if scroll_ctor is not None:
        scroll_ctor.argtypes = [c_void_p, c_uint32, c_uint32, c_int32, c_int32, c_int32]
        scroll_ctor.restype = c_void_p
    s["scroll_event"] = scroll_ctor

    # CGEventSetWindowLocation is a private export; it may live in either lib.
    set_win_loc = getattr(cg, "CGEventSetWindowLocation", None) or getattr(
        sky, "CGEventSetWindowLocation", None
    )
    if set_win_loc is None:
        return None
    set_win_loc.argtypes = [c_void_p, c_double, c_double]
    set_win_loc.restype = None
    s["set_window_location"] = set_win_loc

    # Window-owner PSN lookup (modern) with GetProcessForPID fallback.
    s["main_connection"] = _bind(sky, "CGSMainConnectionID", [], c_uint32) or _bind(
        sky, "SLSMainConnectionID", [], c_uint32
    )
    s["window_owner"] = _bind(
        sky, "SLSGetWindowOwner", [c_uint32, c_uint32, POINTER(c_uint32)], c_int32
    )
    s["connection_psn"] = _bind(
        sky, "SLSGetConnectionPSN", [c_uint32, POINTER(_PSN)], c_int32
    )
    s["process_for_pid"] = (
        _bind(hiservices, "GetProcessForPID", [c_int32, POINTER(_PSN)], c_int32)
        if hiservices is not None
        else None
    )
    if s["process_for_pid"] is None and not (
        s["main_connection"] and s["window_owner"] and s["connection_psn"]
    ):
        return None

    # Keyboard auth envelope (optional; absent before macOS 15).
    s["set_auth"] = _bind(
        sky, "SLEventSetAuthenticationMessage", [c_void_p, c_void_p], None
    )
    s["auth_factory"] = None
    if objc is not None and s["set_auth"] is not None:
        get_class = _bind(objc, "objc_getClass", [c_char_p], c_void_p)
        register = _bind(objc, "sel_registerName", [c_char_p], c_void_p)
        responds = _bind(objc, "class_respondsToSelector", [c_void_p, c_void_p], c_bool)
        metaclass_of = _bind(objc, "object_getClass", [c_void_p], c_void_p)
        send = getattr(objc, "objc_msgSend", None)
        if get_class and register and responds and metaclass_of and send is not None:
            cls = get_class(b"SLSEventAuthenticationMessage")
            sel = register(b"messageWithEventRecord:pid:version:")
            # The factory is a CLASS method, so probe the metaclass. Probing the
            # class object (as the upstream port does) checks instance methods
            # and is false on macOS 26, silently dropping the envelope. The
            # selector is absent before macOS 15; then keys post unenveloped.
            if cls and sel and responds(metaclass_of(cls), sel):
                factory = CFUNCTYPE(
                    c_void_p, c_void_p, c_void_p, c_void_p, c_int32, c_uint32
                )(ctypes.cast(send, c_void_p).value)
                s["auth_factory"] = (factory, cls, sel)

    source = s["source_create"](_K_HID_SYSTEM_STATE)
    if not source:
        return None
    s["source"] = source
    return s


_SYMS: dict | None = None
_LOADED = False
_LOAD_LOCK = threading.Lock()
# One gesture at a time: interleaved primers/decoys from two callers would
# corrupt each other's click-state and focus records. Reentrant so a caller
# can hold it across a whole focus transaction (capture -> gesture -> restore).
GESTURE_LOCK = threading.RLock()
_GESTURE_LOCK = GESTURE_LOCK


def _syms() -> dict | None:
    global _SYMS, _LOADED
    if not _LOADED:
        with _LOAD_LOCK:
            if not _LOADED:
                try:
                    _SYMS = _load()
                except Exception:  # noqa: BLE001 - any binding failure means unavailable
                    _SYMS = None
                _LOADED = True
    return _SYMS


def skylight_available() -> bool:
    """True when the private SkyLight post path resolved and can be used."""
    return _syms() is not None


def keyboard_auth_available() -> bool:
    """True when keyboard events can carry the Chromium auth envelope (macOS 15+)."""
    s = _syms()
    return bool(s and s["auth_factory"])


# --- process / focus plumbing ------------------------------------------------


def _psn_for_window(wid: int, pid: int) -> _PSN | None:
    s = _syms()
    if s is None:
        return None
    psn = _PSN()
    if wid and s["main_connection"] and s["window_owner"] and s["connection_psn"]:
        owner = c_uint32(0)
        if (
            s["window_owner"](s["main_connection"](), int(wid), byref(owner)) == 0
            and owner.value
            and s["connection_psn"](owner.value, byref(psn)) == 0
        ):
            return psn
    if (
        s["process_for_pid"] is not None
        and s["process_for_pid"](int(pid), byref(psn)) == 0
    ):
        return psn
    return None


def _front_psn() -> _PSN | None:
    s = _syms()
    if s is None:
        return None
    psn = _PSN()
    return psn if s["front_process"](byref(psn)) == 0 else None


def front_process_matches(pid: int, wid: int) -> bool | None:
    """Whether WindowServer considers ``pid`` (owner of ``wid``) frontmost."""
    front = _front_psn()
    target = _psn_for_window(wid, pid)
    if front is None or target is None:
        return None
    return (front.hi, front.lo) == (target.hi, target.lo)


def _focus_record(wid: int, direction: int) -> ctypes.Array:
    """248-byte focus (0x01) / defocus (0x02) record for ``SLPSPostEventRecordTo``."""
    buf = (c_uint8 * 0xF8)()
    buf[0x04] = 0xF8
    buf[0x08] = 0x0D
    for offset, byte in enumerate(int(wid).to_bytes(4, "little")):
        buf[0x3C + offset] = byte
    buf[0x8A] = direction
    return buf


def _post_record(psn: _PSN, record: ctypes.Array) -> bool:
    s = _syms()
    return (
        s["post_record"](
            ctypes.cast(byref(psn), c_void_p), ctypes.cast(record, c_void_p)
        )
        == 0
    )


def activate_without_raise(target_pid: int, target_wid: int) -> bool:
    """Make ``target_wid`` key for input WITHOUT raising it or moving Spaces.

    Defocuses the current front process, then focuses the target window. The
    user's front app stays frontmost (NSWorkspace still reports it) but its key
    window stops receiving keys until
    :func:`restore_focus_after_without_raise` hands focus back.
    """
    if _syms() is None or not target_wid:
        return False
    front = _front_psn()
    target = _psn_for_window(target_wid, target_pid)
    if front is None or target is None:
        return False
    defocused = _post_record(front, _focus_record(target_wid, 0x02))
    focused = _post_record(target, _focus_record(target_wid, 0x01))
    return defocused and focused


def restore_focus_after_without_raise(
    previous_pid: int, previous_wid: int, target_pid: int, target_wid: int
) -> bool:
    """Reverse :func:`activate_without_raise` once a background gesture is done."""
    if _syms() is None or not previous_wid or not target_wid:
        return False
    previous = _psn_for_window(previous_wid, previous_pid)
    target = _psn_for_window(target_wid, target_pid)
    if previous is None or target is None:
        return False
    defocused = _post_record(target, _focus_record(target_wid, 0x02))
    focused = _post_record(previous, _focus_record(previous_wid, 0x01))
    return defocused and focused


# --- mouse -------------------------------------------------------------------


def _stamp_mouse(
    ev: int,
    pid: int,
    wid: int,
    step: MouseStep,
    group: int,
    window_point: tuple[float, float] | None = None,
) -> None:
    s = _syms()
    set_field = s["set_field"]
    set_field(ev, _F_PHASE, step.phase)
    set_field(ev, _F_CLICK_STATE, step.click_state)
    set_field(ev, _F_BUTTON, step.button_number)
    set_field(ev, _F_SUBTYPE, 3)
    set_field(ev, _F_TARGET_PID, int(pid))
    set_field(ev, _F_WINDOW, int(wid))
    set_field(ev, _F_WINDOW_UNDER, int(wid))
    set_field(ev, _F_WINDOW_HANDLER, int(wid))
    set_field(ev, _F_CLICK_GROUP, group)
    # The left-click recipe stamps the screen point (cua's Chromium route);
    # right/middle/scroll stamp the window-local point, as cua's
    # *_with_window_local primitives do, so the hit-test uses it directly.
    wx, wy = window_point if window_point is not None else (step.x, step.y)
    s["set_window_location"](ev, float(wx), float(wy))


def _window_point(
    x: float, y: float, origin: tuple[float, float] | None
) -> tuple[float, float] | None:
    if origin is None:
        return None
    return float(x) - float(origin[0]), float(y) - float(origin[1])


def click(
    pid: int,
    wid: int,
    x: float,
    y: float,
    *,
    button: str = "left",
    count: int = 1,
    flags: int = 0,
    window_origin: tuple[float, float] | None = None,
) -> bool:
    """Deliver a pixel click at screen point ``(x, y)`` inside window ``wid``.

    The target is made AppKit-active without being raised, the planned event
    stream is posted to the process, and nothing touches the hardware cursor.
    Returns False (posting nothing) when the SPI is unavailable. Focus is NOT
    restored here; callers decide (see ``restore_focus_after_without_raise``)
    and should hold :data:`GESTURE_LOCK` across the whole transaction.
    ``window_origin`` is the window's top-left in screen points; non-left
    buttons stamp the window-local point derived from it.
    """
    s = _syms()
    if s is None or not wid:
        return False
    plan = click_plan(x, y, button=button, count=count)
    with _GESTURE_LOCK:
        if not activate_without_raise(pid, wid):
            # Without key focus the stream could reach a different responder;
            # post nothing (any partial defocus is undone by the caller's
            # focus restoration).
            return False
        time.sleep(0.05)
        group = time.time_ns() & 0x7FFFFFFF
        for step in plan:
            ev = s["mouse_event"](
                s["source"],
                step.event_type,
                _CGPoint(step.x, step.y),
                step.button_number,
            )
            if not ev:
                return False
            try:
                local = (
                    None
                    if button == "left"
                    else _window_point(step.x, step.y, window_origin)
                )
                _stamp_mouse(ev, pid, wid, step, group, local)
                if flags and step.phase == 3:
                    s["set_flags"](ev, int(flags))
                s["sl_post"](int(pid), ev)
                if button != "left":
                    # AppKit right/middle handlers drop SkyLight-only delivery;
                    # cua posts both routes for non-left buttons.
                    s["public_post"](int(pid), ev)
            finally:
                s["release"](ev)
            if step.delay_after_s:
                time.sleep(step.delay_after_s)
    return True


def scroll(
    pid: int,
    wid: int,
    x: float,
    y: float,
    *,
    lines_y: int = 0,
    lines_x: int = 0,
    window_origin: tuple[float, float] | None = None,
) -> bool:
    """Wheel-scroll the element under screen point ``(x, y)`` of window ``wid``.

    A real wheel event is hit-tested at the point, so nested ``overflow:auto``
    regions that never take keyboard focus scroll too. Positive ``lines_y``
    reveals content above; negative reveals content below.
    """
    s = _syms()
    if s is None or not wid or s["scroll_event"] is None:
        return False
    ticks_y = scroll_ticks(lines_y)
    ticks_x = scroll_ticks(lines_x)
    width = max(len(ticks_y), len(ticks_x))
    if width == 0:
        return True
    ticks_y += [0] * (width - len(ticks_y))
    ticks_x += [0] * (width - len(ticks_x))
    local = _window_point(x, y, window_origin)
    wx, wy = local if local is not None else (x, y)
    with _GESTURE_LOCK:
        group = time.time_ns() & 0x7FFFFFFF
        primer = MouseStep(_MOUSE_MOVED, x, y, 2, 0, 0, 0.012)
        ev = s["mouse_event"](s["source"], _MOUSE_MOVED, _CGPoint(x, y), 0)
        if not ev:
            return False
        try:
            _stamp_mouse(ev, pid, wid, primer, group, local)
            s["sl_post"](int(pid), ev)
            s["public_post"](int(pid), ev)
        finally:
            s["release"](ev)
        time.sleep(primer.delay_after_s)
        for dy, dx in zip(ticks_y, ticks_x, strict=True):
            ev = s["scroll_event"](s["source"], _SCROLL_UNIT_LINE, 2, dy, dx, 0)
            if not ev:
                return False
            try:
                s["set_location"](ev, _CGPoint(x, y))
                s["set_window_location"](ev, float(wx), float(wy))
                s["set_field"](ev, _F_TARGET_PID, int(pid))
                s["set_field"](ev, _F_WINDOW, int(wid))
                s["set_field"](ev, _F_WINDOW_UNDER, int(wid))
                s["set_field"](ev, _F_WINDOW_HANDLER, int(wid))
                # SkyLight reaches backgrounded Chromium/Catalyst; the public
                # route lands on AppKit/WKWebView scrollers.
                s["sl_post"](int(pid), ev)
                s["public_post"](int(pid), ev)
            finally:
                s["release"](ev)
            time.sleep(0.03)
    return True


# --- keyboard ----------------------------------------------------------------


# A live SLSEventRecord is a malloc block of 256 bytes on macOS 26; anything
# much smaller is not a record.
_MIN_RECORD_BYTES = 128


def _event_record(ev: int) -> int | None:
    """Pointer to the ``SLSEventRecord`` embedded in a ``CGEventRef``, or None.

    ``__CGEvent`` is ``{CFRuntimeBase(16), uint32, pad, SLSEventRecord *}``.
    cua probes offsets 24, 32 and 16 and takes the first non-null word, which
    is unsafe: on macOS 26 the event object is 32 bytes (offset 32 is out of
    bounds) and offset 16 holds a non-pointer. Only accept a word that lies
    inside the event allocation and points at a live malloc block of record
    size; otherwise the caller posts without the envelope.
    """
    malloc_size = _syms()["malloc_size"]
    if malloc_size is None:
        return None
    event_bytes = malloc_size(int(ev))
    for offset in (24, 32, 16):
        if offset + 8 > event_bytes:
            continue
        value = c_void_p.from_address(int(ev) + offset).value
        if value and malloc_size(value) >= _MIN_RECORD_BYTES:
            return value
    return None


def _post_key_event(pid: int, ev: int, *, authenticated: bool) -> None:
    s = _syms()
    if authenticated and s["auth_factory"] is not None:
        factory, cls, sel = s["auth_factory"]
        record = _event_record(ev)
        if record:
            message = factory(cls, sel, record, int(pid), 0)
            if message:
                s["set_auth"](ev, message)
    s["sl_post"](int(pid), ev)


def press_key(
    pid: int, keycode: int, flags: int = 0, *, menu_shortcut: bool = False
) -> bool:
    """Press and release ``keycode`` with exactly ``flags`` held, to ``pid``.

    Keys reach the process's key window/first responder without activating it.
    ``menu_shortcut=True`` omits the auth envelope: with it, SkyLight takes a
    direct mach path that bypasses ``NSApplication.sendEvent:`` so NSMenu key
    equivalents (Cmd+S, Cmd+N, ...) never fire.
    """
    s = _syms()
    if s is None:
        return False
    with _GESTURE_LOCK:
        for down in (True, False):
            ev = s["key_event"](s["source"], int(keycode), down)
            if not ev:
                return False
            try:
                # HIDSystemState sources inherit physically held modifiers;
                # always overwrite, including with 0.
                s["set_flags"](ev, int(flags))
                _post_key_event(pid, ev, authenticated=not menu_shortcut)
            finally:
                s["release"](ev)
            time.sleep(_KEY_GAP_S)
    return True


def type_text(pid: int, text: str) -> bool:
    """Type ``text`` literally into ``pid``'s focused element.

    One keycode-0 event pair per Unicode scalar carrying the character, so the
    active keyboard layout and IME are bypassed (CJK and emoji arrive
    literally). Flags are forced to zero on every event, otherwise Chromium
    reads an uppercase character as Shift and leaks it into the next one.
    """
    s = _syms()
    if s is None:
        return False
    with _GESTURE_LOCK:
        for ch in text:
            units = ch.encode("utf-16-le")
            buf = (c_uint16 * (len(units) // 2)).from_buffer_copy(units)
            for down in (True, False):
                ev = s["key_event"](s["source"], 0, down)
                if not ev:
                    return False
                try:
                    s["set_unicode"](ev, len(buf), buf)
                    s["set_flags"](ev, 0)
                    _post_key_event(pid, ev, authenticated=True)
                finally:
                    s["release"](ev)
                time.sleep(_KEY_GAP_S)
    return True
