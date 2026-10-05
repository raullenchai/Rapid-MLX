"""Off-Space windows, drags, key-window switching and dialog guards.

Everything here runs without a screen: the AX server, SkyLight and Quartz are
replaced by fakes, the same way test_computer_use_background.py does it.
"""

# ruff: noqa: N802 - PyObjC test doubles intentionally mirror Objective-C names.

import ctypes
import sys
import time
import types

import pytest

from rapid_mlx.computer_use import ax_driver, backend, background_input, errors


def _snapshot(*, elements=None, window_id="cg:101"):
    window = {
        "index": 0,
        "window_id": window_id,
        "title": "Main",
        "x": 0,
        "y": 0,
        "width": 400,
        "height": 300,
    }
    return {
        "snapshot_id": "s1",
        "observed_at": time.time(),
        "app": {"name": "App", "bundleId": "com.google.chrome", "pid": 4},
        "window_index": 0,
        "window_id": window_id,
        "window": window,
        "elements": elements or [],
    }


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Clock:
    """Stand-in for the ``time`` module: sleeps advance a fake monotonic clock."""

    def __init__(self):
        self.now = 0.0
        self.time = time.time
        self.time_ns = time.time_ns

    def sleep(self, seconds):
        self.now += max(float(seconds), 0.01)

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(backend, "time", fake)
    monkeypatch.setattr(background_input, "time", fake)
    monkeypatch.setattr(ax_driver, "time", fake)
    return fake


@pytest.fixture
def attrs(monkeypatch):
    """Dict-backed AX attributes: ``attrs[element][attribute]``."""
    table: dict = {}
    monkeypatch.setattr(ax_driver, "_get", lambda e, a: table.get(e, {}).get(a))
    monkeypatch.setattr(
        ax_driver, "_get_checked", lambda e, a: (True, table.get(e, {}).get(a))
    )
    return table


@pytest.fixture
def keyboard(monkeypatch, clock):
    """Background delivery with recorded focus moves and validations."""
    calls: list[tuple] = []
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (999, 555))
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 999)
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: calls.append(("reactivate", pid)) or True
    )
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 101)
    monkeypatch.setattr(backend, "_sheet_owner_id", lambda pid: None)
    monkeypatch.setattr(
        background_input,
        "activate_without_raise",
        lambda *a, **k: calls.append(("activate", a)) or True,
    )
    monkeypatch.setattr(
        backend,
        "_restore_user_focus",
        lambda *a: calls.append(("restore", a)) or True,
    )
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda snap, *a, **k: calls.append(("validate", a, k)),
    )
    for name in ("press_key", "type_text"):
        monkeypatch.setattr(
            background_input,
            name,
            lambda *a, _n=name, **k: calls.append((_n, a)) or True,
        )
    return calls


EXACT = {"require_active_app": False, "require_exact_window_id": True}


# --- background_input: drag plan and key-window records ----------------------


def test_drag_plan_presses_moves_and_releases():
    plan = background_input.drag_plan(10, 20, 110, 220, steps=4)
    assert [step.event_type for step in plan] == [5, 1, 2, 1, 6, 6, 6, 6, 2]
    press, *moves, release = plan[3:]
    assert (press.x, press.y) == (10, 20)
    assert [(m.x, m.y) for m in moves] == [
        (35.0, 70.0),
        (60.0, 120.0),
        (85.0, 170.0),
        (110.0, 220.0),
    ]
    assert (release.x, release.y) == (110, 220)
    # The step count is bounded both ways.
    assert len(background_input.drag_plan(0, 0, 1, 1, steps=0)) == 4 + 2 + 1
    assert len(background_input.drag_plan(0, 0, 1, 1, steps=999)) == 4 + 60 + 1


def test_drag_and_click_share_the_posting_path(monkeypatch):
    posted = []
    monkeypatch.setattr(
        background_input,
        "_post_plan",
        lambda pid, wid, plan, **k: posted.append((pid, wid, len(plan), k)) or True,
    )
    assert background_input.drag(4, 101, 0, 0, 10, 10, front_wid=555)
    assert background_input.click(4, 101, 5, 5, front_wid=555)
    assert posted[0] == (
        4,
        101,
        len(background_input.drag_plan(0, 0, 10, 10)),
        {"button": "left", "window_origin": None, "front_wid": 555},
    )
    assert posted[1][3]["button"] == "left"


def test_make_key_record_layout():
    record = background_input._make_key_record(0x01020304, 0x02)
    assert record[0x04] == 0xF8 and record[0x08] == 0x02 and record[0x3A] == 0x10
    assert bytes(record[0x20:0x30]) == b"\xff" * 16
    assert int.from_bytes(bytes(record[0x3C:0x40]), "little") == 0x01020304


def _record_kind(record):
    # 0x0D focus records carry their direction at 0x8A; make-key records
    # carry the mouse down/up kind at 0x08.
    return ("focus", record[0x8A]) if record[0x8A] else ("key", record[0x08])


def test_switch_key_window_posts_defocus_gap_focus_then_make_key(monkeypatch, clock):
    posted = []
    monkeypatch.setattr(
        background_input,
        "_post_record",
        lambda psn, rec: (
            posted.append(
                (int.from_bytes(bytes(rec[0x3C:0x40]), "little"), _record_kind(rec))
            )
            or True
        ),
    )
    psn = background_input._PSN(0, 7)
    assert background_input._switch_key_window(psn, 555, 101)
    assert posted == [
        (555, ("focus", 0x02)),
        (101, ("focus", 0x01)),
        (101, ("key", 0x01)),
        (101, ("key", 0x02)),
    ]
    assert clock.now >= 0.02  # the measured gap before the focus record
    posted.clear()
    monkeypatch.setattr(
        background_input, "_post_record", lambda psn, rec: posted.append(1) and False
    )
    assert background_input._switch_key_window(psn, 555, 101) is False
    assert posted == [1]  # nothing after a refused defocus


def test_same_process_activation_and_restore_switch_key_window(monkeypatch):
    switched = []
    psn = background_input._PSN(3, 9)
    monkeypatch.setattr(background_input, "_syms", lambda: {"ok": True})
    monkeypatch.setattr(background_input, "_front_psn", lambda: psn)
    monkeypatch.setattr(
        background_input,
        "_psn_for_window",
        lambda wid, pid: background_input._PSN(3, 9),
    )
    monkeypatch.setattr(
        background_input,
        "_switch_key_window",
        lambda p, a, b: switched.append((a, b)) or True,
    )
    monkeypatch.setattr(
        background_input,
        "_post_record",
        lambda *a: pytest.fail("same-process switches post no cross-app records"),
    )
    assert background_input.activate_without_raise(4, 101, front_wid=555)
    # Without the user's window id the key status cannot be handed back.
    assert background_input.activate_without_raise(4, 101) is False
    assert background_input.restore_focus_after_without_raise(4, 555, 4, 101)
    assert switched == [(555, 101), (101, 555)]


# --- ax_driver: off-Space windows ----------------------------------------------


def test_app_windows_adds_focused_main_and_remote_windows_once(monkeypatch, attrs):
    attrs["app"] = {
        "AXWindows": ["w1"],
        "AXFocusedWindow": "w1-alias",
        "AXMainWindow": "w2",
    }
    ids = {"w1": 1, "w1-alias": 1, "w2": 2, "r3": 3, "r2": 2}
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: ids.get(w))
    monkeypatch.setattr(ax_driver, "_pid_of", lambda app: 4)
    monkeypatch.setattr(ax_driver, "_REMOTE_IDS", {4: {7}})
    monkeypatch.setattr(ax_driver, "_remote_windows", lambda pid: ["r2", "r3"])
    assert ax_driver._app_windows("app") == ["w1", "w2", "r3"]
    # Unmappable extras are kept unless they equal a listed window.
    ids.clear()
    attrs["app"] = {"AXWindows": ["w1"], "AXFocusedWindow": "w1", "AXMainWindow": "x"}
    monkeypatch.setattr(ax_driver, "_pid_of", lambda app: None)
    assert ax_driver._app_windows("app") == ["w1", "x"]


def test_pid_of(monkeypatch):
    monkeypatch.setattr(ax_driver, "AS", None)
    assert ax_driver._pid_of("el") is None
    services = types.SimpleNamespace(AXUIElementGetPid=lambda el, _: (0, 42))
    monkeypatch.setattr(ax_driver, "AS", services)
    assert ax_driver._pid_of("el") == 42
    services.AXUIElementGetPid = lambda el, _: (-25201, 0)
    assert ax_driver._pid_of("el") is None
    services.AXUIElementGetPid = lambda el, _: (_ for _ in ()).throw(TypeError())
    assert ax_driver._pid_of("el") is None


def test_remote_windows_keep_only_windows(monkeypatch, attrs):
    attrs.update({("e", 1): {"AXRole": "AXWindow"}, ("e", 2): {"AXRole": "AXButton"}})
    monkeypatch.setattr(ax_driver, "_REMOTE_IDS", {4: {2, 1, 3}})
    monkeypatch.setattr(
        ax_driver, "_remote_element", lambda pid, i: None if i == 3 else ("e", i)
    )
    assert ax_driver._remote_windows(4) == [("e", 1)]


def test_discover_remote_windows_stops_when_all_wanted_are_found(
    monkeypatch, attrs, clock
):
    windows = {10: 501, 20: 502, 30: 503}
    for element_id in windows:
        attrs[("e", element_id)] = {"AXRole": "AXWindow"}
    attrs[("e", 5)] = {"AXRole": "AXGroup"}
    probed = []

    def element(pid, element_id):
        probed.append(element_id)
        return ("e", element_id) if element_id in (5, 10, 20, 30) else None

    monkeypatch.setattr(ax_driver, "_remote_element", element)
    monkeypatch.setattr(background_input, "ax_window_id", lambda el: windows.get(el[1]))
    remote: dict = {}
    monkeypatch.setattr(ax_driver, "_REMOTE_IDS", remote)
    assert ax_driver.discover_remote_windows(4, {501, 502}, limit=100) == {501, 502}
    assert max(probed) == 20  # stopped as soon as both were found
    assert remote == {4: {10, 20}}


def test_discover_remote_windows_honors_the_time_budget(monkeypatch, clock):
    monkeypatch.setattr(ax_driver, "_REMOTE_IDS", {})
    probed = []

    def element(pid, element_id):
        probed.append(element_id)
        clock.now += 0.01
        return None

    monkeypatch.setattr(ax_driver, "_remote_element", element)
    assert (
        ax_driver.discover_remote_windows(4, {1}, budget_s=1.0, limit=10_000) == set()
    )
    assert len(probed) < 10_000 and len(probed) % 64 == 0


def test_remote_element_builds_token_and_hands_ownership_to_the_bridge(monkeypatch):
    released, created = [], []

    def create(data):
        created.append(data)
        return 0xBEEF

    cf = types.SimpleNamespace(
        CFDataCreate=lambda alloc, token, n: created.append(bytes(token)) or 0xDA7A,
        CFRelease=released.append,
    )
    _install_module(monkeypatch, "objc", objc_object=lambda c_void_p: ("obj", c_void_p))
    monkeypatch.setattr(ax_driver, "_remote_factory", lambda: (create, cf))
    assert ax_driver._remote_element(4, 9) == ("obj", 0xBEEF)
    token = created[0]
    assert int.from_bytes(token[:4], "little") == 4
    assert int.from_bytes(token[8:12], "little") == ax_driver._REMOTE_MAGIC
    assert int.from_bytes(token[12:20], "little") == 9
    assert released == [0xDA7A, 0xBEEF]
    # Nothing created, nothing returned.
    cf.CFDataCreate = lambda *a: 0
    assert ax_driver._remote_element(4, 9) is None
    cf.CFDataCreate = lambda *a: 0xDA7A
    monkeypatch.setattr(ax_driver, "_remote_factory", lambda: (lambda d: 0, cf))
    assert ax_driver._remote_element(4, 9) is None
    monkeypatch.setattr(ax_driver, "_remote_factory", lambda: False)
    assert ax_driver._remote_element(4, 9) is None


def test_remote_factory_binds_the_spi_once_and_caches_a_missing_symbol(monkeypatch):
    class _Fn:
        restype = argtypes = None

    class _CF:
        def __init__(self):
            self.CFDataCreate, self.CFRelease = _Fn(), _Fn()

    spi, cf = types.SimpleNamespace(_AXUIElementCreateWithRemoteToken=_Fn()), _CF()
    opened = []

    def cdll(path):
        opened.append(path)
        return spi if path.endswith("HIServices") else cf

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setattr(ax_driver, "_remote_create", None)
    fn, bound_cf = ax_driver._remote_factory()
    assert fn is spi._AXUIElementCreateWithRemoteToken and bound_cf is cf
    assert fn.restype is ctypes.c_void_p and fn.argtypes == [ctypes.c_void_p]
    assert cf.CFDataCreate.restype is ctypes.c_void_p
    assert ax_driver._remote_factory() == (fn, cf)
    assert len(opened) == 2  # bound once, then cached
    # An OS without the private symbol is remembered as unavailable.
    monkeypatch.setattr(ax_driver, "_remote_create", None)
    spi = types.SimpleNamespace()
    assert ax_driver._remote_factory() is False
    assert ax_driver._remote_factory() is False
    assert len(opened) == 4


def test_first_exposure_is_per_process_launch(monkeypatch):
    monkeypatch.setattr(ax_driver, "_EXPOSED", {})

    class App:
        def __init__(self, pid, launched):
            self.pid, self.launched = pid, launched

        def processIdentifier(self):
            return self.pid

        def launchDate(self):
            if self.launched is None:
                return None
            return types.SimpleNamespace(timeIntervalSince1970=lambda: self.launched)

    assert ax_driver.first_exposure(App(4, 100.0))
    assert not ax_driver.first_exposure(App(4, 100.0))
    assert ax_driver.first_exposure(App(4, 200.0))  # recycled pid
    # Unknown launch time: never remembered (a recycled pid is poked again).
    assert ax_driver.first_exposure(App(5, None))
    assert ax_driver.first_exposure(App(5, None))
    # The attribute poke and the plain settle wait are tracked separately, so
    # a settle-only first touch never suppresses the attribute poke.
    assert ax_driver.first_exposure(App(6, 1.0), "settle")
    assert ax_driver.first_exposure(App(6, 1.0))
    assert not ax_driver.first_exposure(App(6, 1.0))


def test_walk_reports_editable_values_but_never_secure_ones(monkeypatch, attrs):
    attrs.update(
        {
            "root": {"AXRole": "AXGroup", "AXChildren": ["field", "same", "secret"]},
            "field": {"AXRole": "AXTextField", "AXTitle": "To", "AXValue": "a@b.c"},
            "same": {"AXRole": "AXTextField", "AXValue": "label is value"},
            "secret": {
                "AXRole": "AXTextField",
                "AXSubrole": "AXSecureTextField",
                "AXValue": "hunter2",
            },
        }
    )
    monkeypatch.setattr(ax_driver, "_action_names", lambda el: [])
    monkeypatch.setattr(ax_driver, "_point_size", lambda el: None)
    out, seen = [], set()
    ax_driver._walk("root", 0, out, [0], seen)
    values = {t["text"]: t["value"] for t in out}
    assert values["To"] == "a@b.c"
    assert values["label is value"] is None
    assert values["[secure text redacted]"] is None
    assert "hunter2" not in repr(out)
    assert seen == {"AXGroup", "AXTextField"}


# --- backend: off-Space window records -------------------------------------------


def test_offscreen_windows_are_best_effort(monkeypatch):
    monkeypatch.setattr(ax_driver, "AS", None)
    assert backend._offscreen_ax_windows({"pid": 4}, set()) == []
    monkeypatch.setattr(ax_driver, "AS", object())
    monkeypatch.setattr(
        backend,
        "_offscreen_ax_windows_unchecked",
        lambda *a: (_ for _ in ()).throw(RuntimeError("AX down")),
    )
    assert backend._offscreen_ax_windows({"pid": 4}, set()) == []


def _cg(number, *, pid=4, layer=0, name="Doc", x=0):
    return {
        "kCGWindowNumber": number,
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": layer,
        "kCGWindowName": name,
        "kCGWindowBounds": {"X": x, "Y": 0, "Width": 500, "Height": 400},
    }


def test_offscreen_windows_map_by_window_id_and_scan_remote_once(monkeypatch, attrs):
    cg = [
        _cg(101),  # on screen (already listed)
        _cg(202, name=""),  # off-Space window, AX reachable, CG title empty
        _cg(303),  # off-Space window found only by remote token
        _cg(404, layer=8),  # helper surface: never a target
        _cg(505, pid=9),  # another process
        _cg(606, name="Helper"),  # no AX window: never a target
    ]
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: cg,
        kCGNullWindowID=0,
        kCGWindowListOptionAll=0,
    )
    ax_windows = ["ax101", "ax202", "minimized"]
    attrs.update(
        {
            "ax202": {"AXTitle": "Draft"},
            "ax303": {"AXTitle": "Remote"},
            "minimized": {"AXMinimized": True},
        }
    )
    ids = {"ax101": 101, "ax202": 202, "ax303": 303, "minimized": 707}
    monkeypatch.setattr(ax_driver, "AXUIElementCreateApplication", lambda pid: "app")
    monkeypatch.setattr(ax_driver, "_app_windows", lambda app: list(ax_windows))
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: ids.get(w))
    scans = []

    def discover(pid, wanted):
        scans.append((pid, set(wanted)))
        ax_windows.append("ax303")
        return {303}

    monkeypatch.setattr(ax_driver, "discover_remote_windows", discover)
    monkeypatch.setattr(backend, "_REMOTE_SCANNED", {})
    found = backend._offscreen_ax_windows_unchecked({"pid": 4}, {101})
    assert [(w["window_id"], w["title"], w["offscreen"]) for w in found] == [
        ("cg:202", "Draft", True),
        ("cg:303", "Doc", True),
    ]
    assert scans == [(4, {303, 606})]
    # A window that was already scanned for is not scanned again.
    backend._offscreen_ax_windows_unchecked({"pid": 4}, {101})
    assert len(scans) == 1
    # A recycled pid (new process start time) is scanned afresh and its
    # cached element ids are dropped.
    monkeypatch.setattr(ax_driver, "_REMOTE_IDS", {4: {12}})
    ax_windows.remove("ax303")
    stale_at_read = []
    monkeypatch.setattr(
        ax_driver,
        "_app_windows",
        lambda app: (
            stale_at_read.append(4 in ax_driver._REMOTE_IDS) or list(ax_windows)
        ),
    )
    backend._offscreen_ax_windows_unchecked({"pid": 4, "processStartTime": 2.0}, {101})
    assert len(scans) == 2
    assert 4 not in ax_driver._REMOTE_IDS
    # The old process's element ids are dropped before the first AX read.
    assert stale_at_read and not any(stale_at_read)


def test_window_records_append_offscreen_windows(monkeypatch):
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [_cg(101)],
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )
    seen_args = []
    monkeypatch.setattr(
        backend,
        "_offscreen_ax_windows",
        lambda app, seen: (
            seen_args.append(set(seen))
            or [{"window_id": "cg:202", "title": "Off", "offscreen": True}]
        ),
    )
    records = backend._window_records({"pid": 4})
    assert [(r["index"], r["window_id"]) for r in records] == [
        (0, "cg:101"),
        (1, "cg:202"),
    ]
    assert seen_args == [{101}]


# --- backend: sheets -------------------------------------------------------------


def test_sheet_owner_and_summary(monkeypatch, attrs):
    monkeypatch.setattr(ax_driver, "AS", None)
    assert backend._sheet_owner_id(4) is None
    monkeypatch.setattr(ax_driver, "AS", object())
    monkeypatch.setattr(ax_driver, "AXUIElementCreateApplication", lambda pid: "app")
    monkeypatch.setattr(background_input, "ax_window_id", lambda w: {"doc": 101}.get(w))
    attrs["app"] = {"AXFocusedWindow": "doc"}
    attrs["doc"] = {"AXRole": "AXWindow"}
    assert backend._sheet_owner_id(4) is None
    attrs["app"] = {"AXFocusedWindow": "sheet"}
    attrs["sheet"] = {
        "AXRole": "AXSheet",
        "AXParent": "doc",
        "AXChildren": ["text", "save", "cancel", "untitled"],
    }
    attrs["text"] = {"AXRole": "AXStaticText", "AXValue": "Save changes?"}
    attrs["save"] = {"AXRole": "AXButton", "AXTitle": "Save"}
    attrs["cancel"] = {"AXRole": "AXButton", "AXTitle": "Cancel"}
    attrs["untitled"] = {"AXRole": "AXButton"}
    assert backend._sheet_owner_id(4) == 101
    assert backend._sheet_summary(4) == "Save changes? (buttons: Save, Cancel)"


def test_restore_keeps_target_sheet_from_reading_as_a_user_switch(monkeypatch):
    restored = []
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 4)
    # The target's own sheet is key after the gesture, not a window the user
    # picked: focus still goes back to the user's window of the same app.
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 909)
    monkeypatch.setattr(backend, "_sheet_owner_id", lambda pid: 101)
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: restored.append(a) or True,
    )
    assert backend._restore_user_focus((4, 555), 4, 101) is True
    assert restored == [(4, 555, 4, 101)]
    # Any other key window is the user's choice and wins.
    monkeypatch.setattr(backend, "_sheet_owner_id", lambda pid: None)
    assert backend._restore_user_focus((4, 555), 4, 101) is None


# --- backend: _keyed_target -------------------------------------------------------


def test_keyed_target_fast_path_validates_the_exact_window(keyboard):
    with backend._keyed_target(_snapshot()) as state:
        keyboard.append(("body",))
    assert keyboard == [("validate", (), EXACT), ("body",)]
    assert state == {}


def test_keyed_target_fast_path_undoes_a_self_activation(monkeypatch, keyboard):
    # The key posted to the already-key target made the target app activate
    # itself; the user's app gets the foreground back.
    front = iter([4])
    monkeypatch.setattr(backend, "_live_front_pid", lambda: next(front))
    with backend._keyed_target(_snapshot()) as state:
        keyboard.append(("body",))
    assert keyboard == [("validate", (), EXACT), ("body",), ("reactivate", 999)]
    assert state == {"focus_restored": True}
    assert backend._focus_fields(state) == {"focus_restored": True}


def test_keyed_target_self_activation_restore_error_is_reported(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 4)
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: (_ for _ in ()).throw(OSError("gone"))
    )
    with backend._keyed_target(_snapshot()) as state:
        pass
    assert state == {"focus_restored": False}
    assert "warning" in backend._focus_fields(state)


def test_keyed_target_fast_path_when_user_is_in_the_target_window(
    monkeypatch, keyboard
):
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 101))
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 4)
    with backend._keyed_target(_snapshot()) as state:
        pass
    # The user already was in the target: nothing to hand back.
    assert keyboard == [("validate", (), EXACT)]
    assert state == {}


def test_keyed_target_switches_key_window_inside_the_users_app(monkeypatch, keyboard):
    # The user types in window 555 of the same app; the target 101 is not key.
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    keys = iter([555, 555, 101])
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: next(keys))
    with backend._keyed_target(_snapshot()) as state:
        keyboard.append(("body",))
    assert keyboard == [
        ("activate", (4, 101, 555)),
        ("validate", (), EXACT),
        ("body",),
        ("restore", ((4, 555), 4, 101)),
    ]
    assert state["focus_restored"] is True


def test_keyed_target_fails_closed_when_key_never_moves(monkeypatch, keyboard, clock):
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 555)

    def validate(snap, *a, **k):
        keyboard.append(("validate", a, k))
        raise errors.ComputerUseError("target_drift", "wrong key window")

    monkeypatch.setattr(backend, "_validate_focused_window", validate)
    with (
        pytest.raises(errors.ComputerUseError) as exc,
        backend._keyed_target(_snapshot()),
    ):
        pytest.fail("no keys may be posted to an unverified window")
    assert exc.value.code == "target_drift"
    assert clock.now >= 0.5  # waited for the asynchronous key change
    assert keyboard[-1] == ("restore", ((4, 555), 4, 101))


def test_keyed_target_refused_activation_posts_nothing(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 202)
    monkeypatch.setattr(background_input, "activate_without_raise", lambda *a: False)
    with (
        pytest.raises(errors.ComputerUseError) as exc,
        backend._keyed_target(_snapshot()),
    ):
        pytest.fail("unreachable")
    assert exc.value.code == "synthetic_input_blocked"
    assert keyboard == [("restore", ((999, 555), 4, 101))]


def test_keyed_target_spi_error_is_action_failed(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 202)
    monkeypatch.setattr(
        background_input,
        "activate_without_raise",
        lambda *a: (_ for _ in ()).throw(OSError("SPI")),
    )
    with (
        pytest.raises(errors.ComputerUseError) as exc,
        backend._keyed_target(_snapshot()),
    ):
        pytest.fail("unreachable")
    assert exc.value.code == "action_failed"
    assert keyboard[-1][0] == "restore"


def test_keyed_target_reports_the_targets_own_dialog(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 909)
    monkeypatch.setattr(backend, "_sheet_owner_id", lambda pid: 101)
    monkeypatch.setattr(backend, "_sheet_summary", lambda pid: "Replace? (buttons: OK)")
    with (
        pytest.raises(errors.ComputerUseError) as exc,
        backend._keyed_target(_snapshot()),
    ):
        pytest.fail("keys must not reach the dialog")
    assert exc.value.code == "synthetic_input_blocked"
    assert "Replace? (buttons: OK)" in exc.value.message
    assert keyboard[-1][0] == "restore"


def test_keyed_target_restore_error_is_reported_not_raised(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 202)
    monkeypatch.setattr(
        backend,
        "_restore_user_focus",
        lambda *a: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    monkeypatch.setattr(
        backend, "_key_window_id", lambda pid, keys=iter([202, 101]): next(keys)
    )
    with backend._keyed_target(_snapshot()) as state:
        pass
    assert state["focus_restored"] is False


def test_keyed_target_without_a_captured_window_fails(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_frontmost_window", lambda: None)
    with (
        pytest.raises(errors.ComputerUseError) as exc,
        backend._keyed_target(_snapshot()),
    ):
        pytest.fail("unreachable")
    assert exc.value.code == "action_failed"


def test_keyed_target_keeps_a_transient_companion_key(monkeypatch, keyboard):
    transient = {"window_id": "cg:303", "x": 5, "y": 5, "width": 50, "height": 50}
    snapshot = {**_snapshot(), backend._KEYBOARD_WINDOW: transient}
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 303)
    with backend._keyed_target(snapshot) as state:
        pass
    # No key-window move (it would dismiss the popover), exact validation.
    assert keyboard == [("validate", (transient,), EXACT)]
    assert state == {}
    # A self-activation in response to the keys is still undone.
    front = iter([999, 4])
    monkeypatch.setattr(backend, "_live_front_pid", lambda: next(front))
    with backend._keyed_target(snapshot) as state:
        pass
    assert keyboard[-1] == ("reactivate", 999)
    assert state == {"focus_restored": True}


def test_background_transient_keyboard_target_is_validated_exactly(
    monkeypatch, keyboard
):
    transient = {"window_id": "cg:303", "x": 5, "y": 5, "width": 50, "height": 50}
    snapshot = {
        **_snapshot(
            elements=[
                {
                    "index": 0,
                    "role": "AXTextField",
                    "center": [10, 10],
                    "source_window_id": "cg:303",
                }
            ]
        ),
        "transient_window": transient,
    }
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: "live")
    monkeypatch.setattr(ax_driver, "_get", lambda e, a: None)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(
        backend, "_focused_transient_window", lambda app, anchor, **k: dict(transient)
    )
    prepared = backend._prepare_synthetic_action("App", None, snapshot, 0)
    assert prepared[backend._KEYBOARD_WINDOW] == transient
    assert backend._KEYBOARD_WINDOW not in snapshot
    assert keyboard == [("validate", (transient,), EXACT)]
    assert backend._send_key(prepared, 36, 0, True) == backend.ROUTE_PID
    assert keyboard[1:] == [
        ("validate", (transient,), EXACT),
        ("press_key", (4, 36, 0)),
    ]


# --- backend: keyboard paths through _keyed_target --------------------------------


def test_type_text_and_keys_go_through_the_keyed_target(monkeypatch, keyboard):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    monkeypatch.setattr(
        backend, "_key_window_id", lambda pid, keys=iter([555, 101]): next(keys)
    )
    result = backend.type_text("App", "hi")
    assert result["mode"] == "SkyLight-unicode"
    assert result["focus_restored"] is True and "warning" not in result
    assert [c[0] for c in keyboard] == ["activate", "validate", "type_text", "restore"]


@pytest.mark.parametrize("call", ["press_key", "named_key", "hotkey", "type_text"])
def test_keyboard_results_report_unrestored_focus(monkeypatch, keyboard, call):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 202)
    monkeypatch.setattr(backend, "_restore_user_focus", lambda *a: False)
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    result = {
        "press_key": lambda: backend.press_key("App", "escape"),
        "named_key": lambda: backend.press_key("App", "a"),
        "hotkey": lambda: backend.hotkey("App", "shift+tab"),
        "type_text": lambda: backend.type_text("App", "hi"),
    }[call]()
    assert result["focus_restored"] is False
    assert "could not be handed back" in result["warning"]


# --- backend: drag -----------------------------------------------------------------


@pytest.fixture
def dragging(monkeypatch, clock):
    calls = []
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (999, 555))
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snap, **k: calls.append(("window", k)) or snap["window"],
    )
    monkeypatch.setattr(
        backend, "_restore_user_focus", lambda *a: calls.append(("restore", a)) or True
    )
    monkeypatch.setattr(
        background_input,
        "drag",
        lambda *a, **k: calls.append(("drag", a, k)) or True,
    )
    monkeypatch.setattr(
        backend,
        "_finish_action",
        lambda app, snap, delivery, **k: {**delivery, **k},
    )
    return calls


def test_drag_validates_both_points_and_restores_focus(dragging):
    result = backend.drag("App", 10, 20, 30, 40, expected_snapshot=_snapshot())
    assert dragging == [
        ("window", {"point": (10, 20), "require_topmost": False}),
        ("window", {"point": (30, 40), "require_topmost": False}),
        (
            "drag",
            (4, 101, 10.0, 20.0, 30.0, 40.0),
            {"window_origin": (0.0, 0.0), "front_wid": 555},
        ),
        ("restore", ((999, 555), 4, 101)),
    ]
    assert result["mode"] == "SkyLight-drag"
    assert result["focus_restored"] is True and "warning" not in result
    assert result["verified"] is None


def test_drag_requires_background_delivery(monkeypatch, dragging):
    monkeypatch.setenv(background_input.DELIVERY_ENV, "foreground")
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.drag("App", 1, 1, 2, 2, expected_snapshot=_snapshot())
    assert exc.value.code == "synthetic_input_blocked"
    assert dragging == []


@pytest.mark.parametrize("raises", [True, False])
def test_drag_primitive_failure_is_action_failed_and_restores(
    monkeypatch, dragging, raises
):
    def refuse(*a, **k):
        if raises:
            raise OSError("SPI failed")
        return False

    monkeypatch.setattr(background_input, "drag", refuse)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.drag("App", 1, 1, 2, 2, expected_snapshot=_snapshot())
    assert exc.value.code == "action_failed"
    assert dragging[-1][0] == "restore"


def test_drag_reports_unrestored_focus(monkeypatch, dragging):
    monkeypatch.setattr(
        backend,
        "_restore_user_focus",
        lambda *a: (_ for _ in ()).throw(RuntimeError("gone")),
    )
    result = backend.drag("App", 1, 1, 2, 2, expected_snapshot=_snapshot())
    assert result["focus_restored"] is False and "warning" in result


def test_drag_without_a_captured_window_fails(monkeypatch, dragging):
    monkeypatch.setattr(backend, "_frontmost_window", lambda: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.drag("App", 1, 1, 2, 2, expected_snapshot=_snapshot())
    assert exc.value.code == "action_failed"
    assert not any(call[0] == "drag" for call in dragging)


# --- backend: background fill and popup typeahead ----------------------------------


@pytest.fixture
def filling(monkeypatch, keyboard, attrs):
    values = {"live": "old text"}
    attrs["live"] = values
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(
        backend, "_focus_without_commit", lambda snap, live: "AXFocused"
    )
    monkeypatch.setattr(
        backend, "_borrow_foreground", lambda snap: pytest.fail("no foreground borrow")
    )
    services = types.SimpleNamespace(
        AXValueCreate=lambda kind, rng: ("range", rng), kAXValueCFRangeType=4
    )
    monkeypatch.setattr(ax_driver, "AS", services)
    monkeypatch.setattr(ax_driver, "kAXErrorSuccess", 0, raising=False)
    selections = []
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementSetAttributeValue",
        lambda el, attr, val: selections.append((attr, val)) or 0,
        raising=False,
    )

    def type_text(pid, text):
        keyboard.append(("type_text", (pid, text)))
        values["AXValue"] = text
        return True

    monkeypatch.setattr(background_input, "type_text", type_text)
    values["AXValue"] = "old text"
    return selections


def test_background_fill_selects_all_then_types_without_foreground(filling, keyboard):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    result = backend._synthetic_fill(snapshot, 0, "néw")
    assert filling == [("AXSelectedTextRange", ("range", (0, 8)))]
    assert result == {
        "mode": "SkyLight-fill",
        "element_index": 0,
        "verified": True,
        "actual": "néw",
        "focus_restored": None,  # the target already was key: nothing moved
    }
    assert [c[0] for c in keyboard] == ["validate", "validate", "type_text"]
    assert all(c[2] == EXACT for c in keyboard if c[0] == "validate")


def test_background_fill_with_an_empty_value_deletes_the_selection(
    monkeypatch, filling, keyboard, attrs
):
    def press_key(pid, keycode, *a, **k):
        keyboard.append(("press_key", (pid, keycode)))
        attrs["live"]["AXValue"] = ""
        return True

    monkeypatch.setattr(background_input, "press_key", press_key)
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    result = backend._synthetic_fill(snapshot, 0, "")
    assert filling == [("AXSelectedTextRange", ("range", (0, 8)))]
    assert ("press_key", (4, backend.KEY_ALIASES["delete"])) in keyboard
    assert not any(c[0] == "type_text" for c in keyboard)
    assert result["verified"] is True and result["actual"] == ""


def test_background_fill_empty_value_failed_delete_is_action_failed(
    monkeypatch, filling, keyboard
):
    monkeypatch.setattr(background_input, "press_key", lambda *a, **k: False)
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "")
    assert exc.value.code == "action_failed"


def test_background_fill_reports_unverified_readback(
    monkeypatch, filling, keyboard, attrs
):
    monkeypatch.setattr(
        background_input,
        "type_text",
        lambda pid, text: keyboard.append(("type_text", text)) or True,
    )
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    result = backend._synthetic_fill(snapshot, 0, "new")
    assert result["verified"] is None and result["actual"] == "old text"


def test_background_fill_never_appends_to_an_unreadable_value(
    monkeypatch, filling, keyboard, attrs
):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    attrs["live"]["AXValue"] = None
    attrs["live"]["AXNumberOfCharacters"] = 5
    backend._synthetic_fill(snapshot, 0, "new")
    assert filling == [("AXSelectedTextRange", ("range", (0, 5)))]
    filling.clear()
    keyboard.clear()
    attrs["live"]["AXValue"] = None
    attrs["live"]["AXNumberOfCharacters"] = None
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "new")
    assert exc.value.code == "synthetic_input_blocked"
    assert not any(c[0] == "type_text" for c in keyboard)


def test_background_fill_fails_closed(monkeypatch, filling, keyboard):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    monkeypatch.setattr(backend, "_focus_without_commit", lambda snap, live: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "x")
    assert exc.value.code == "synthetic_input_blocked"
    monkeypatch.setattr(
        backend, "_focus_without_commit", lambda snap, live: "AXFocused"
    )
    monkeypatch.setattr(
        ax_driver, "AXUIElementSetAttributeValue", lambda *a: -25200, raising=False
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "x")
    assert exc.value.code == "synthetic_input_blocked"
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "x")
    assert exc.value.code == "synthetic_input_blocked"
    assert not any(c[0] == "type_text" for c in keyboard)


def test_background_fill_spi_error_is_action_failed(monkeypatch, filling, keyboard):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXTextField", "center": [1, 1]}]
    )
    monkeypatch.setattr(background_input, "type_text", lambda *a: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "x")
    assert exc.value.code == "action_failed"


def test_popup_is_chosen_by_typeahead_without_opening_it(filling, keyboard):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXPopUpButton", "center": [1, 1]}]
    )
    result = backend._synthetic_fill(snapshot, 0, "Canada")
    assert result["mode"] == "SkyLight-typeahead" and result["verified"] is True
    assert filling == []  # no text selection on a popup
    assert ("type_text", (4, "Canada")) in keyboard


def test_popup_typeahead_spi_error_is_action_failed(monkeypatch, filling):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXPopUpButton", "center": [1, 1]}]
    )
    monkeypatch.setattr(background_input, "type_text", lambda *a: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._synthetic_fill(snapshot, 0, "Canada")
    assert exc.value.code == "action_failed"


# --- backend: lingering popup menus -------------------------------------------------


def test_dismiss_lingering_popup_escapes_until_closed(monkeypatch, keyboard):
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 999)
    lists = iter([[_cg(1, layer=101)]] * 3 + [[]])
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: next(lists),
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=2,
    )
    backend._dismiss_lingering_popup(_snapshot(), {1})
    assert keyboard == [("press_key", (4, backend.KEY_ALIASES["escape"]))]


def test_dismiss_lingering_popup_ignores_a_menu_the_press_opened(monkeypatch, keyboard):
    # Popup 2 appeared after the press (e.g. a menu item opening a submenu
    # or a page menu); only popup 1, open before the pick, may be escaped.
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 999)
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [_cg(2, layer=101)],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=2,
    )
    backend._dismiss_lingering_popup(_snapshot(), {1})
    backend._dismiss_lingering_popup(_snapshot(), set())
    assert keyboard == []


@pytest.mark.parametrize("front", [4, None])
def test_dismiss_lingering_popup_leaves_the_users_own_app_alone(
    monkeypatch, keyboard, front
):
    # The user is in the target app (or unknown): the open menu may be theirs.
    monkeypatch.setattr(backend, "_live_front_pid", lambda: front)
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [_cg(1, layer=101)],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=2,
    )
    backend._dismiss_lingering_popup(_snapshot(), {1})
    assert keyboard == []


def test_dismiss_lingering_popup_is_a_no_op_otherwise(monkeypatch, keyboard):
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [_cg(1, pid=9, layer=101), _cg(2)],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=2,
    )
    backend._dismiss_lingering_popup(_snapshot(), {1})
    assert keyboard == []
    # A failing probe never turns the successful pick into an error.
    sys.modules["Quartz"].CGWindowListCopyWindowInfo = lambda *a: 1 / 0
    backend._dismiss_lingering_popup(_snapshot(), {1})
    # Only Chromium keeps a picked <select> popup open; other apps' menus
    # are never escaped.
    sys.modules["Quartz"].CGWindowListCopyWindowInfo = lambda *a: pytest.fail("probe")
    native = _snapshot()
    native["app"] = {**native["app"], "bundleId": "com.apple.textedit"}
    backend._dismiss_lingering_popup(native, {1})
    monkeypatch.setenv(background_input.DELIVERY_ENV, "foreground")
    sys.modules["Quartz"].CGWindowListCopyWindowInfo = lambda *a: pytest.fail("probe")
    backend._dismiss_lingering_popup(_snapshot(), {1})


def test_dismiss_lingering_popup_stops_when_escape_or_probe_fails(
    monkeypatch, keyboard
):
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 999)
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [_cg(1, layer=101)],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=2,
    )
    waits = []
    monkeypatch.setattr(backend.time, "monotonic", lambda: waits.append(1) or 0.0)
    # An Escape that cannot be delivered ends the attempt without waiting.
    monkeypatch.setattr(backend, "_synthesize", lambda *a, **k: False)
    backend._dismiss_lingering_popup(_snapshot(), {1})
    assert waits == []
    # A failing front-app probe never turns the successful pick into an error.
    monkeypatch.setattr(backend, "_live_front_pid", lambda: 1 / 0)
    backend._dismiss_lingering_popup(_snapshot(), {1})


def test_menu_item_press_dismisses_a_lingering_popup(monkeypatch, keyboard):
    dismissed = []
    snapshot = _snapshot(
        elements=[
            {"index": 0, "role": "AXMenuItem", "center": [1, 1], "actions": ["AXPress"]}
        ]
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda el, action: 0,
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(
        backend, "_dismiss_lingering_popup", lambda s, before: dismissed.append(before)
    )
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    open_before = iter([{7}, set()])
    monkeypatch.setattr(backend, "_open_popup_menus", lambda s: next(open_before))
    for _ in range(2):
        assert backend.click("App", element_index=0, expected_snapshot=snapshot) == {
            "mode": "AXPress",
            "element_index": 0,
            "focus_restored": None,
        }
    # Cleanup runs only when a popup was already open before the press.
    assert dismissed == [{7}]


def test_non_menu_press_never_probes_popups(monkeypatch, keyboard):
    snapshot = _snapshot(
        elements=[
            {"index": 0, "role": "AXButton", "center": [1, 1], "actions": ["AXPress"]}
        ]
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda el, action: 0,
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(
        backend, "_open_popup_menus", lambda s: pytest.fail("probed popups")
    )
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    assert backend.click("App", element_index=0, expected_snapshot=snapshot)[
        "mode"
    ] == ("AXPress")


def test_open_popup_menus_is_scoped_and_best_effort(monkeypatch):
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *a: [
            _cg(1, layer=101),
            _cg(2, pid=9, layer=101),
            _cg(3),
        ],
        kCGNullWindowID=0,
        kCGWindowListOptionOnScreenOnly=2,
    )
    assert backend._open_popup_menus(_snapshot()) == {1}
    native = _snapshot()
    native["app"] = {**native["app"], "bundleId": "com.apple.textedit"}
    assert backend._open_popup_menus(native) == set()
    sys.modules["Quartz"].CGWindowListCopyWindowInfo = lambda *a: 1 / 0
    assert backend._open_popup_menus(_snapshot()) == set()


# --- backend: focus_only on a window that is not key --------------------------------


def test_focus_only_makes_the_target_key_and_never_commits(monkeypatch, keyboard):
    snapshot = _snapshot(
        elements=[
            {"index": 0, "role": "AXButton", "center": [1, 1], "actions": ["AXPress"]}
        ]
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    # Another window is key until the switch lands.
    monkeypatch.setattr(
        backend, "_key_window_id", lambda pid, keys=iter([202, 101]): next(keys)
    )
    focused = iter([None, "AXFocused"])
    monkeypatch.setattr(backend, "_focus_without_commit", lambda s, live: next(focused))
    monkeypatch.setattr(
        backend, "_pixel_click", lambda *a, **k: pytest.fail("focus_only clicked")
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda *a: pytest.fail("focus_only pressed"),
    )
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    result = backend.click(
        "App", element_index=0, expected_snapshot=snapshot, focus_only=True
    )
    assert result["mode"] == "AXFocused"
    assert result["focus_restored"] is True
    assert [c[0] for c in keyboard] == ["activate", "validate", "restore"]


def test_focus_only_never_rekeys_for_a_transient_target(monkeypatch, keyboard):
    snapshot = _snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXButton",
                "center": [1, 1],
                "actions": [],
                "source_window_id": "cg:303",
            }
        ]
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(backend, "_focus_without_commit", lambda s, live: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click(
            "App", element_index=0, expected_snapshot=snapshot, focus_only=True
        )
    assert exc.value.code == "synthetic_input_blocked"
    assert keyboard == []


# --- backend: Cmd chords on a bound window -------------------------------------------


_UNREADABLE = object()


@pytest.fixture
def chord(monkeypatch, keyboard, attrs):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_pid_app_element", lambda app: "app")
    monkeypatch.setattr(backend, "_finish_action", lambda app, snap, d, **k: d)
    attrs["app"] = {"AXMenuBar": "bar"}
    attrs["bar"] = {"AXRole": "AXMenuBar", "AXChildren": ["file"]}
    attrs["file"] = {"AXRole": "AXMenu", "AXChildren": ["save", "save-as", "no-cmd"]}

    def checked(element, attribute):
        value = attrs.get(element, {}).get(attribute)
        return (False, None) if value is _UNREADABLE else (True, value)

    monkeypatch.setattr(ax_driver, "_get_checked", checked)
    attrs["save"] = {"AXMenuItemCmdChar": "S", "AXMenuItemCmdModifiers": 0}
    attrs["save-as"] = {"AXMenuItemCmdChar": "S", "AXMenuItemCmdModifiers": 0x1}
    attrs["no-cmd"] = {"AXMenuItemCmdChar": "K", "AXMenuItemCmdModifiers": 0x8}
    attrs["file"]["AXChildren"].append("back")
    attrs["back"] = {"AXMenuItemCmdVirtualKey": 123, "AXMenuItemCmdModifiers": 0}
    return snapshot


def test_is_menu_equivalent_matches_char_and_modifiers(chord):
    app = chord["app"]
    cmd, shift = backend.MODIFIER_FLAGS["cmd"], backend.MODIFIER_FLAGS["shift"]
    assert backend._is_menu_equivalent(app, "s", cmd)
    assert backend._is_menu_equivalent(app, "s", cmd | shift)
    assert not backend._is_menu_equivalent(
        app, "s", cmd | backend.MODIFIER_FLAGS["option"]
    )
    assert not backend._is_menu_equivalent(app, "k", cmd)  # not a Cmd item
    # Arrow/function-key equivalents carry a virtual key instead of a char.
    assert backend._is_menu_equivalent(app, "left", cmd, 123)
    assert not backend._is_menu_equivalent(app, "right", cmd, 124)


def test_is_menu_equivalent_fails_closed_on_unreadable_or_truncated_menus(
    monkeypatch, chord, attrs
):
    cmd = backend.MODIFIER_FLAGS["cmd"]
    monkeypatch.setattr(backend, "_MENU_SCAN_LIMIT", 2)
    assert backend._is_menu_equivalent(chord["app"], "k", cmd)
    attrs["app"] = {}
    assert backend._is_menu_equivalent(chord["app"], "k", cmd)
    monkeypatch.setattr(backend, "_MENU_SCAN_LIMIT", 3000)
    attrs["app"] = {"AXMenuBar": "bar"}
    attrs["save"]["AXMenuItemCmdModifiers"] = None  # unreadable modifiers
    assert backend._is_menu_equivalent(chord["app"], "s", cmd | 0x20000)


def test_is_menu_equivalent_maps_control_and_unreadable_modifiers(chord, attrs):
    app, cmd = chord["app"], backend.MODIFIER_FLAGS["cmd"]
    ctrl = backend.MODIFIER_FLAGS["ctrl"]
    attrs["file"]["AXChildren"].append("ctrl-k")
    attrs["ctrl-k"] = {"AXMenuItemCmdChar": "J", "AXMenuItemCmdModifiers": 0x4}
    assert backend._is_menu_equivalent(app, "j", cmd | ctrl)
    assert not backend._is_menu_equivalent(app, "j", cmd)
    # A matching key whose modifiers cannot be read could be the command.
    attrs["ctrl-k"]["AXMenuItemCmdModifiers"] = None
    assert backend._is_menu_equivalent(app, "j", cmd)


def test_is_menu_equivalent_fails_closed_when_menu_children_are_unreadable(
    chord, attrs
):
    app, cmd = chord["app"], backend.MODIFIER_FLAGS["cmd"]
    assert not backend._is_menu_equivalent(app, "k", cmd)
    # A menu whose items cannot be read could hide Cmd+K.
    attrs["file"]["AXChildren"] = _UNREADABLE
    assert backend._is_menu_equivalent(app, "k", cmd)
    # So could a menu bar that lists no menus at all, or fails to read.
    attrs["bar"]["AXChildren"] = []
    assert backend._is_menu_equivalent(app, "k", cmd)
    attrs["bar"]["AXChildren"] = _UNREADABLE
    assert backend._is_menu_equivalent(app, "k", cmd)


def test_is_menu_equivalent_accepts_leaves_and_empty_menus(chord, attrs):
    app, cmd = chord["app"], backend.MODIFIER_FLAGS["cmd"]
    # Leaf items have no AXChildren (the fixture's items), and a menu that
    # genuinely has no items reads as empty: neither is a failure.
    attrs["bar"]["AXChildren"] = ["file", "empty"]
    attrs["empty"] = {"AXRole": "AXMenuBarItem", "AXChildren": []}
    assert not backend._is_menu_equivalent(app, "k", cmd)


def test_get_checked_separates_absent_from_failed_reads(monkeypatch):
    results = {"AXChildren": (0, ["x"]), "AXNone": (-25212, None)}
    results["AXUnsupported"] = (-25205, None)
    results["AXFailed"] = (-25204, None)  # kAXErrorCannotComplete
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementCopyAttributeValue",
        lambda element, attribute, _: results[attribute],
    )
    monkeypatch.setattr(ax_driver, "kAXErrorSuccess", 0, raising=False)
    assert ax_driver._get_checked("e", "AXChildren") == (True, ["x"])
    assert ax_driver._get_checked("e", "AXNone") == (True, None)
    assert ax_driver._get_checked("e", "AXUnsupported") == (True, None)
    assert ax_driver._get_checked("e", "AXFailed") == (False, None)


def test_content_chord_goes_to_the_bound_window_in_the_background(chord, keyboard):
    result = backend.hotkey("App", "cmd+k", window_id="cg:101")
    assert result["mode"] == "SkyLight-chord" and result["route"] == "pid_events"
    assert keyboard == [
        ("validate", (), EXACT),
        ("press_key", (4, 40, backend.MODIFIER_FLAGS["cmd"])),
    ]


def test_menu_chord_of_an_inactive_app_is_refused(monkeypatch, chord, keyboard, attrs):
    monkeypatch.setattr(
        ax_driver,
        "_application_for_pid",
        lambda pid: types.SimpleNamespace(isActive=lambda: False),
    )
    # The chord may be a menu command but its item cannot be resolved
    # (unreadable modifiers), so it cannot be pressed through Accessibility.
    attrs["save"]["AXMenuItemCmdModifiers"] = None
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.hotkey("App", "cmd+s", window_id="cg:101")
    assert exc.value.code == "synthetic_input_blocked"
    assert keyboard == []


def test_menu_chord_of_the_users_app_targets_the_bound_window(
    monkeypatch, chord, keyboard
):
    monkeypatch.setattr(
        ax_driver,
        "_application_for_pid",
        lambda pid: types.SimpleNamespace(isActive=lambda: True),
    )
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    monkeypatch.setattr(
        backend, "_key_window_id", lambda pid, keys=iter([555, 101]): next(keys)
    )
    result = backend.hotkey("App", "cmd+s", window_id="cg:101")
    assert result["focus_restored"] is True
    assert [c[0] for c in keyboard] == ["activate", "validate", "press_key", "restore"]


def test_chord_spi_error_is_action_failed(monkeypatch, chord):
    monkeypatch.setattr(background_input, "press_key", lambda *a: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.hotkey("App", "cmd+k", window_id="cg:101")
    assert exc.value.code == "action_failed"


def test_process_is_active_treats_errors_as_inactive(monkeypatch):
    def boom():
        raise RuntimeError("gone")

    monkeypatch.setattr(
        ax_driver,
        "_application_for_pid",
        lambda pid: types.SimpleNamespace(isActive=boom),
    )
    assert backend._process_is_active(_snapshot()) is False
    monkeypatch.setattr(ax_driver, "_application_for_pid", lambda pid: None)
    assert backend._process_is_active(_snapshot()) is False


# --- backend: observation does not activate with background delivery ---------------


class _StopError(Exception):
    pass


@pytest.mark.parametrize(
    ("delivery", "expected"),
    [("background", [False]), ("foreground", [False, True])],
)
def test_route_observation_activates_only_for_foreground_delivery(
    monkeypatch, delivery, expected
):
    monkeypatch.setenv(background_input.DELIVERY_ENV, delivery)
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    resolved = []

    def resolve(app, activate=True):
        resolved.append(activate)
        return object(), {"name": "App", "bundleId": "com.example.app", "pid": 4}

    monkeypatch.setattr(backend, "_resolve_app", resolve)
    monkeypatch.setattr(
        backend, "_select_window", lambda *a, **k: (_ for _ in ()).throw(_StopError())
    )
    with pytest.raises(_StopError):
        backend.get_app_state(
            "App",
            screenshot=False,
            use_cache=False,
            activate=backend.OBSERVE_BY_ROUTE,
        )
    assert resolved == expected
    resolved.clear()
    # The public default is unchanged: it activates on either route.
    with pytest.raises(_StopError):
        backend.get_app_state("App", screenshot=False, use_cache=False)
    assert resolved == [True]
    resolved.clear()
    with pytest.raises(_StopError):
        backend.get_app_state("App", screenshot=False, use_cache=False, activate=False)
    assert resolved == [False]
    resolved.clear()
    with pytest.raises(_StopError):
        backend.get_app_state("App", screenshot=False, use_cache=False, activate=True)
    assert resolved == [True]


def test_internal_reobservations_observe_by_route(monkeypatch, keyboard):
    # With no caller snapshot the action re-observes the window, and the
    # post-action state is observed too; both let the route decide, so
    # background delivery never activates the target.
    seen = []

    def observe(app, **kwargs):
        seen.append(kwargs.get("activate", "default"))
        return _snapshot(
            elements=[
                {
                    "index": 0,
                    "role": "AXButton",
                    "center": [1, 1],
                    "actions": ["AXPress"],
                }
            ]
        )

    monkeypatch.setattr(backend, "get_app_state", observe)
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda el, action: 0,
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    backend.click("App", element_index=0, include_post_state=True)
    assert seen and set(seen) == {backend.OBSERVE_BY_ROUTE}


def test_ax_app_element_exposes_once_per_process(monkeypatch, clock):
    sets = []
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementCreateApplication=lambda pid: ("element", pid),
        AXUIElementSetAttributeValue=lambda *a: sets.append(a[1]),
    )
    monkeypatch.setattr(ax_driver, "_EXPOSED", {})

    class App:
        def processIdentifier(self):
            return 4

        def launchDate(self):
            return types.SimpleNamespace(timeIntervalSince1970=lambda: 50.0)

    backend._ax_app_element(App(), activate=False)
    backend._ax_app_element(App(), activate=False)
    assert sets == ["AXManualAccessibility", "AXEnhancedUserInterface"]


# --- cua: stale-guard telemetry and window binding -----------------------------------


def test_tree_signature_ignores_tab_memory_usage():
    from rapid_mlx.cua import loop

    a = {"tree_text": "[0] AXTab Inbox - Memory usage - 120 MB"}
    b = {"tree_text": "[0] AXTab Inbox - Memory usage - 1,204.5 MB"}
    c = {"tree_text": "[0] AXTab Outbox - Memory usage - 120 MB"}
    assert loop._tree_signature(a) == loop._tree_signature(b)
    assert loop._tree_signature(a) != loop._tree_signature(c)
    # Only a trailing suffix is telemetry; the same text inside content is not.
    d = {"tree_text": "[0] AXStaticText x - Memory usage - 1 MB here"}
    e = {"tree_text": "[0] AXStaticText x - Memory usage - 2 MB here"}
    assert loop._tree_signature(d) != loop._tree_signature(e)
    # A trailing suffix on a non-tab line is page content, not telemetry.
    f = {"tree_text": "[0] AXStaticText Usage - Memory usage - 1 MB"}
    g = {"tree_text": "[0] AXStaticText Usage - Memory usage - 2 MB"}
    assert loop._tree_signature(f) != loop._tree_signature(g)
    # Chrome's tab strip uses AXRadioButton (selected tabs carry a star).
    h = {"tree_text": "[3] AXRadioButton* Inbox - Memory usage - 1 MB"}
    i = {"tree_text": "[3] AXRadioButton* Inbox - Memory usage - 9 MB"}
    assert loop._tree_signature(h) == loop._tree_signature(i)


def test_observed_change_lists_appeared_and_disappeared_labels():
    from rapid_mlx.cua import loop

    before = {"tree_text": "[0] AXButton Send\n[1] AXTextField To\nnoise"}
    after = {"tree_text": "[0] AXTextField To\n[1] AXStaticText Sent!"}
    assert loop._observed_change(before, after) == {
        "appeared": ["AXStaticText Sent!"],
        "disappeared": ["AXButton Send"],
    }
    twice = {"tree_text": "[0] AXButton OK\n[1] AXButton OK"}
    once = {"tree_text": "[0] AXButton OK"}
    assert loop._observed_change(twice, once)["disappeared"] == ["AXButton OK"]


def test_cli_run_binds_to_a_validated_window(monkeypatch, tmp_path):
    import rapid_mlx.cua.cli as cli_mod
    import rapid_mlx.cua.loop as loop_mod
    from rapid_mlx.cua import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cua-config.json")
    monkeypatch.setattr(config_mod, "RUNS_DIR", tmp_path / "runs")
    seen = {}

    async def fake_run(config, app, goal, **kwargs):
        seen.update(kwargs)
        return {"status": "done", "final_summary": "ok"}

    monkeypatch.setattr(loop_mod, "run", fake_run)
    monkeypatch.setattr(
        backend,
        "validate_window",
        lambda app, wid: {
            "app": {"name": "App", "pid": 4, "bundleId": "b"},
            "window_id": "cg:303",
        },
    )
    rc = cli_mod.main(["run", "--app", "App", "--goal", "g", "--window-id", "cg:303"])
    assert rc == 0
    assert seen["window_id"] == "cg:303"
    assert seen["backend_app"] == "pid:4"
    assert seen["expected_app"]["pid"] == 4
