"""SkyLight SPI plumbing of the background hands, driven through fake libraries.

The real symbols only exist on macOS; these doubles stand in for the dylibs so
symbol binding, PSN lookup, focus records and the key/auth post path are
exercised on every platform.
"""

# ruff: noqa: N802 - PyObjC test doubles intentionally mirror Objective-C names.

import ctypes
import sys
import types

import pytest

from rapid_mlx.computer_use import backend, background_input, errors

# --- fake dylibs --------------------------------------------------------------


class _Fn:
    """A bindable stand-in for a ctypes foreign function."""

    def __init__(self, impl=None):
        self.impl = impl or (lambda *a: 0)
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.impl(*args)


class _Lib:
    def __init__(self, **symbols):
        self._symbols = symbols

    def __getattr__(self, name):
        try:
            return self._symbols[name]
        except KeyError:
            raise AttributeError(name) from None


_MSG_SEND = ctypes.CFUNCTYPE(ctypes.c_void_p)(lambda: None)

_PATHS = {
    "CoreGraphics": "cg",
    "SkyLight": "sky",
    "CoreFoundation": "cf",
    "libSystem": "libsystem",
    "ApplicationServices": "hiservices",
    "libobjc": "objc",
}


def _libs(*, responds=True, source=1):
    return {
        "cg": _Lib(
            CGEventSourceCreate=_Fn(lambda kind: source),
            CGEventCreateMouseEvent=_Fn(),
            CGEventCreateKeyboardEvent=_Fn(),
            CGEventKeyboardSetUnicodeString=_Fn(),
            CGEventSetLocation=_Fn(),
            CGEventSetFlags=_Fn(),
            CGEventGetType=_Fn(),
            CGEventCreateScrollWheelEvent2=_Fn(),
            CGEventSetWindowLocation=_Fn(),
        ),
        "sky": _Lib(
            SLEventPostToPid=_Fn(),
            SLEventSetIntegerValueField=_Fn(),
            SLPSPostEventRecordTo=_Fn(),
            _SLPSGetFrontProcess=_Fn(),
            CGSMainConnectionID=_Fn(),
            SLSGetWindowOwner=_Fn(),
            SLSGetConnectionPSN=_Fn(),
            SLEventSetAuthenticationMessage=_Fn(),
        ),
        "cf": _Lib(CFRelease=_Fn()),
        "libsystem": _Lib(malloc_size=_Fn(), malloc_zone_from_ptr=_Fn()),
        "hiservices": _Lib(
            GetProcessForPID=_Fn(),
            GetProcessPID=_Fn(),
            _AXUIElementGetWindow=_Fn(),
        ),
        "objc": _Lib(
            objc_getClass=_Fn(lambda name: 11),
            sel_registerName=_Fn(lambda name: 12),
            class_respondsToSelector=_Fn(lambda cls, sel: responds),
            object_getClass=_Fn(lambda obj: 13),
            objc_msgSend=_MSG_SEND,
        ),
    }


def _install_cdll(monkeypatch, libs, missing=()):
    def cdll(path):
        for marker, key in _PATHS.items():
            if marker in path:
                if key in missing or key not in libs:
                    raise OSError(path)
                return libs[key]
        raise AssertionError(f"unexpected library {path}")

    monkeypatch.setattr(background_input.ctypes, "CDLL", cdll)


def test_load_binds_every_symbol_and_the_auth_factory(monkeypatch):
    _install_cdll(monkeypatch, _libs())
    s = background_input._load()
    assert s is not None
    assert s["source"] == 1
    assert s["scroll_event"].restype is ctypes.c_void_p
    assert s["set_window_location"].argtypes[1] is ctypes.c_double
    factory, cls, sel = s["auth_factory"]
    assert (cls, sel) == (11, 12)
    assert s["sl_post"].argtypes == [ctypes.c_int32, ctypes.c_void_p]


def test_load_without_auth_selector_posts_unenveloped(monkeypatch):
    _install_cdll(monkeypatch, _libs(responds=False))
    assert background_input._load()["auth_factory"] is None


def test_load_survives_missing_optional_libraries(monkeypatch):
    libs = _libs()
    libs["cg"]._symbols.pop("CGEventCreateScrollWheelEvent2")
    # Without libSystem the record checks are unavailable, which is required.
    _install_cdll(monkeypatch, libs, missing=("libsystem",))
    assert background_input._load() is None
    _install_cdll(monkeypatch, libs, missing=("objc", "hiservices"))
    s = background_input._load()
    assert s["scroll_event"] is None
    assert s["auth_factory"] is None
    assert s["process_for_pid"] is None and s["ax_window"] is None


@pytest.mark.parametrize(
    ("lib", "symbol"),
    [("cg", "CGEventSetWindowLocation"), ("sky", "SLPSPostEventRecordTo")],
)
def test_load_requires_core_symbols(monkeypatch, lib, symbol):
    libs = _libs()
    libs[lib]._symbols.pop(symbol)
    _install_cdll(monkeypatch, libs)
    assert background_input._load() is None


def test_load_requires_a_psn_lookup(monkeypatch):
    libs = _libs()
    libs["sky"]._symbols.pop("SLSGetWindowOwner")
    _install_cdll(monkeypatch, libs, missing=("hiservices",))
    assert background_input._load() is None


def test_load_requires_an_event_source(monkeypatch):
    _install_cdll(monkeypatch, _libs(source=0))
    assert background_input._load() is None


def test_binding_error_means_unavailable(monkeypatch):
    def broken():
        raise TypeError("bad argtypes")

    monkeypatch.setattr(background_input, "_load", broken)
    monkeypatch.setattr(background_input, "_LOADED", False)
    monkeypatch.setattr(background_input, "_SYMS", None)
    assert background_input.skylight_available() is False
    assert background_input.keyboard_auth_available() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        background_input._live()


def test_keyboard_auth_reflects_the_factory(monkeypatch):
    monkeypatch.setattr(background_input, "_syms", lambda: {"auth_factory": None})
    assert background_input.keyboard_auth_available() is False
    monkeypatch.setattr(background_input, "_syms", lambda: {"auth_factory": (1,)})
    assert background_input.keyboard_auth_available() is True


# --- PSN / front process --------------------------------------------------------


def _write_psn(ref, hi, lo):
    ref._obj.hi, ref._obj.lo = hi, lo
    return 0


def _psn_syms(monkeypatch, **overrides):
    def owner(conn, wid, out):
        out._obj.value = {101: 77}.get(wid, 0)
        return 0

    syms = {
        "main_connection": lambda: 5,
        "window_owner": owner,
        "connection_psn": lambda conn, ref: _write_psn(ref, 0, conn),
        "process_for_pid": lambda pid, ref: _write_psn(ref, 1, pid),
        "front_process": lambda ref: _write_psn(ref, 0, 77),
        "pid_for_psn": lambda psn, out: setattr(out._obj, "value", 4) or 0,
    }
    syms.update(overrides)
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    return syms


def _key(psn):
    return (psn.hi, psn.lo)


def test_psn_prefers_the_window_owner_then_the_pid(monkeypatch):
    syms = _psn_syms(monkeypatch)
    assert _key(background_input._psn_for_window(101, 4)) == (0, 77)
    # Unowned window: fall back to the pid's process.
    assert _key(background_input._psn_for_window(202, 4)) == (1, 4)
    syms["process_for_pid"] = lambda pid, ref: -600
    assert background_input._psn_for_window(202, 4) is None
    syms["process_for_pid"] = None
    assert background_input._psn_for_window(0, 4) is None
    monkeypatch.setattr(background_input, "_syms", lambda: None)
    assert background_input._psn_for_window(101, 4) is None


def test_front_pid_and_front_process_match(monkeypatch):
    syms = _psn_syms(monkeypatch)
    assert background_input.front_pid() == 4
    assert background_input.front_process_matches(4, 101) is True
    assert background_input.front_process_matches(4, 202) is False
    syms["pid_for_psn"] = lambda psn, out: -1
    assert background_input.front_pid() is None
    syms["pid_for_psn"] = None
    assert background_input.front_pid() is None
    syms["front_process"] = lambda ref: -1
    assert background_input.front_pid() is None
    assert background_input.front_process_matches(4, 101) is None
    monkeypatch.setattr(background_input, "_syms", lambda: None)
    assert background_input.front_pid() is None


def test_activation_needs_both_processes(monkeypatch):
    monkeypatch.setattr(background_input, "_syms", lambda: None)
    assert background_input.activate_without_raise(4, 101) is False
    monkeypatch.setattr(background_input, "_syms", lambda: {"ok": True})
    monkeypatch.setattr(background_input, "_front_psn", lambda: None)
    monkeypatch.setattr(background_input, "_psn_for_window", lambda wid, pid: "t")
    assert background_input.activate_without_raise(4, 101) is False


def test_post_record_reports_acceptance(monkeypatch):
    seen = []
    syms = {"post_record": lambda psn, rec: seen.append((psn, rec)) or 0}
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    record = background_input._focus_record(101, 0x01)
    assert background_input._post_record(background_input._PSN(0, 7), record)
    assert seen[0][1].value == ctypes.addressof(record)
    syms["post_record"] = lambda psn, rec: -1
    assert not background_input._post_record(background_input._PSN(0, 7), record)


def test_restore_focus_posts_defocus_then_focus(monkeypatch):
    posted = []
    monkeypatch.setattr(background_input, "_syms", lambda: {"ok": True})
    monkeypatch.setattr(
        background_input, "_psn_for_window", lambda wid, pid: {555: "user"}.get(wid)
    )
    monkeypatch.setattr(
        background_input,
        "_post_record",
        lambda psn, rec: posted.append((psn, rec[0x8A])) or True,
    )
    # The target window has no PSN: nothing is posted.
    assert background_input.restore_focus_after_without_raise(9, 555, 4, 101) is False
    assert posted == []
    monkeypatch.setattr(background_input, "_psn_for_window", lambda wid, pid: wid)
    assert background_input.restore_focus_after_without_raise(9, 555, 4, 101)
    assert posted == [(101, 0x02), (555, 0x01)]
    assert background_input.restore_focus_after_without_raise(9, 0, 4, 101) is False
    monkeypatch.setattr(background_input, "_syms", lambda: None)
    assert background_input.restore_focus_after_without_raise(9, 555, 4, 101) is False


def test_ax_window_id_maps_bridged_elements(monkeypatch):
    def get_window(ref, out):
        out._obj.value = 4242 if ref.value == 99 else 0
        return 0

    syms = {"ax_window": get_window}
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    fake_objc = types.ModuleType("objc")
    fake_objc.pyobjc_id = lambda element: element
    monkeypatch.setitem(sys.modules, "objc", fake_objc)
    assert background_input.ax_window_id(99) == 4242
    assert background_input.ax_window_id(98) is None
    assert background_input.ax_window_id(None) is None

    def not_bridged(_element):
        raise TypeError("not a CF object")

    fake_objc.pyobjc_id = not_bridged
    assert background_input.ax_window_id(99) is None
    syms["ax_window"] = None
    assert background_input.ax_window_id(99) is None


# --- posting --------------------------------------------------------------------


def _posting_syms(monkeypatch):
    log = []
    made = iter(range(1, 1000))
    syms = {
        "source": 1,
        "mouse_event": lambda *a: next(made),
        "key_event": lambda *a: next(made),
        "scroll_event": lambda *a: next(made),
        "set_field": lambda *a: None,
        "set_flags": lambda ev, flags: log.append(("flags", ev, flags)),
        "set_location": lambda *a: None,
        "set_window_location": lambda *a: None,
        "set_unicode": lambda ev, n, buf: log.append(("unicode", ev, buf[:n])),
        "sl_post": lambda pid, ev: log.append(("post", pid, ev)),
        "release": lambda ev: None,
        "auth_factory": None,
    }
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    monkeypatch.setattr(background_input, "activate_without_raise", lambda *a: True)
    monkeypatch.setattr(background_input.time, "sleep", lambda *_: None)
    return syms, log


def test_click_stamps_modifier_flags_on_button_events(monkeypatch):
    _, log = _posting_syms(monkeypatch)
    shift = 1 << 17
    assert background_input.click(4, 101, 5.0, 5.0, flags=shift)
    stamped = [entry for entry in log if entry[0] == "flags"]
    assert stamped and all(entry[2] == shift for entry in stamped)


def test_scroll_without_constructor_or_distance(monkeypatch):
    syms, log = _posting_syms(monkeypatch)
    assert background_input.scroll(4, 101, 5.0, 5.0) is True
    assert log == []
    syms["scroll_event"] = None
    assert background_input.scroll(4, 101, 5.0, 5.0, lines_y=-3) is False


def test_best_effort_swallows_cleanup_errors():
    def fail():
        raise OSError("cleanup failed")

    background_input._best_effort(fail)


def test_press_key_posts_down_then_up_with_exact_flags(monkeypatch):
    _, log = _posting_syms(monkeypatch)
    assert background_input.press_key(4, 36, 1 << 20)
    assert log == [
        ("flags", 1, 1 << 20),
        ("post", 4, 1),
        ("flags", 2, 1 << 20),
        ("post", 4, 2),
    ]


def test_type_text_posts_one_unicode_pair_per_character(monkeypatch):
    _, log = _posting_syms(monkeypatch)
    assert background_input.type_text(4, "a世")
    posts = [entry for entry in log if entry[0] == "post"]
    assert [entry[2] for entry in posts] == [1, 2, 3, 4]
    assert [entry[2] for entry in log if entry[0] == "unicode"] == [
        [ord("a")],
        [ord("a")],
        [ord("世")],
        [ord("世")],
    ]


def test_key_event_carries_the_auth_envelope(monkeypatch):
    authed = []
    syms = {
        "auth_factory": (lambda cls, sel, rec, pid, ver: rec * 10, 11, 12),
        "set_auth": lambda ev, message: authed.append((ev, message)),
        "sl_post": lambda pid, ev: authed.append(("post", ev)),
    }
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    records = {1: 5, 2: None, 3: 0}
    monkeypatch.setattr(background_input, "_event_record", records.get)
    background_input._post_key_event(4, 1, authenticated=True)
    assert authed == [(1, 50), ("post", 1)]
    authed.clear()
    # No record: post unenveloped. Factory returns nil: post unenveloped.
    background_input._post_key_event(4, 2, authenticated=True)
    syms["auth_factory"] = (lambda *a: 0, 11, 12)
    monkeypatch.setitem(records, 3, 6)
    background_input._post_key_event(4, 3, authenticated=True)
    # Menu shortcuts skip the envelope entirely.
    background_input._post_key_event(4, 1, authenticated=False)
    assert authed == [("post", 2), ("post", 3), ("post", 1)]


def test_record_header_needs_the_event_type_symbol(monkeypatch):
    monkeypatch.setattr(background_input, "_syms", lambda: {"event_type": None})
    assert background_input._record_header_matches(1, 1) is False


# --- backend glue ---------------------------------------------------------------


def test_target_ids_reject_windows_without_cg_id():
    snapshot = {"app": {"pid": 4}, "window": {"window_id": "ax:3"}}
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._target_ids(snapshot)
    assert exc.value.code == "invalid_argument"


def test_frontmost_window_falls_back_to_nsworkspace(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: None)
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: None)
    app = types.SimpleNamespace(processIdentifier=lambda: 31)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: app)
    appkit = types.ModuleType("AppKit")
    appkit.NSWorkspace = types.SimpleNamespace(sharedWorkspace=lambda: workspace)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    assert backend._frontmost_window() == (31, 0)
    workspace.frontmostApplication = lambda: None
    assert backend._frontmost_window() is None

    def broken():
        raise RuntimeError("no window server")

    appkit.NSWorkspace = types.SimpleNamespace(sharedWorkspace=broken)
    assert backend._frontmost_window() is None


def test_key_window_id_reads_ax_focused_window(monkeypatch):
    monkeypatch.setattr(backend.ax_driver, "AS", None)
    assert backend._key_window_id(4) is None
    monkeypatch.setattr(backend.ax_driver, "AS", object())
    monkeypatch.setattr(
        backend.ax_driver, "AXUIElementCreateApplication", lambda pid: ("app", pid)
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: ("win", element, attr),
    )
    monkeypatch.setattr(
        background_input,
        "ax_window_id",
        lambda win: 77 if win == ("win", ("app", 4), "AXFocusedWindow") else None,
    )
    assert backend._key_window_id(4) == 77


def test_activate_app_is_best_effort(monkeypatch):
    monkeypatch.setattr(backend.ax_driver, "_application_for_pid", lambda pid: None)
    assert backend._activate_app(4) is False

    class Running:
        def __init__(self, result):
            self.result = result

        def activateWithOptions_(self, options):
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    for result, expected in ((True, True), (RuntimeError("gone"), False)):
        running = Running(result)
        monkeypatch.setattr(
            backend.ax_driver, "_application_for_pid", lambda pid, r=running: r
        )
        assert backend._activate_app(4) is expected


def test_window_origin_requires_numeric_corner():
    assert backend._window_origin({"x": 1, "y": 2}) == (1.0, 2.0)
    assert backend._window_origin({"x": None, "y": 2}) is None


def test_send_key_routes(monkeypatch):
    snapshot = {"app": {"pid": 4}, "window": {"window_id": "cg:101"}}
    monkeypatch.setattr(background_input, "press_key", lambda *a: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._send_key(snapshot, 36, 0, background=True)
    assert exc.value.code == "action_failed"
    pressed = []
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda *a: pressed.append(a))
    assert backend._send_key(snapshot, 36, 1 << 17, background=False) == (
        backend.ROUTE_GLOBAL_HID
    )
    backend._send_key(snapshot, 36, 0, background=False)
    assert pressed == [(36, 1 << 17), (36,)]
