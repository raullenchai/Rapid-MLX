"""Background (pid/window-routed) input delivery for the computer-use hands.

Pure event-plan and routing tests run everywhere. The live test drives a real
Calculator window and is opt-in: ``RAPID_MLX_LIVE_GUI=1`` on a Mac whose test
host holds Accessibility permission.
"""

# ruff: noqa: N802 - PyObjC test doubles intentionally mirror Objective-C names.

import os
import subprocess
import sys
import time
import types

import pytest

from rapid_mlx.computer_use import backend, background_input, errors


def _snapshot(*, bundle_id="com.example.app", elements=None):
    window = {
        "index": 0,
        "window_id": "cg:101",
        "title": "Main",
        "x": 0,
        "y": 0,
        "width": 400,
        "height": 300,
    }
    return {
        "snapshot_id": "s1",
        "observed_at": time.time(),
        "app": {"name": "App", "bundleId": bundle_id, "pid": 4},
        "window_index": 0,
        "window_id": window["window_id"],
        "window": window,
        "elements": elements or [],
    }


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def background(monkeypatch):
    """Route through background delivery and record every primitive call."""
    calls: list[tuple] = []
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    # The user's captured front process remains current unless a test changes it.
    monkeypatch.setattr(background_input, "front_pid", lambda: 999)
    monkeypatch.setattr(
        background_input,
        "click",
        lambda *a, **k: calls.append(("click", a, k)) or True,
    )
    monkeypatch.setattr(
        background_input,
        "scroll",
        lambda *a, **k: calls.append(("scroll", a, k)) or True,
    )
    monkeypatch.setattr(
        background_input,
        "press_key",
        lambda *a, **k: calls.append(("press_key", a, k)) or True,
    )
    monkeypatch.setattr(
        background_input,
        "type_text",
        lambda *a, **k: calls.append(("type_text", a, k)) or True,
    )
    monkeypatch.setattr(background_input, "front_process_matches", lambda *a: False)
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: calls.append(("restore", a, {})) or True,
    )
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (999, 555))
    # The target window (cg:101) is already its app's key window, so a
    # keyboard gesture needs no key-window switch unless a test says so.
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 101)
    monkeypatch.setattr(backend, "_sheet_owner_id", lambda pid: None)
    monkeypatch.setattr(
        background_input,
        "activate_without_raise",
        lambda *a, **k: calls.append(("activate", a, k)) or True,
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_cg_click",
        lambda *a, **k: pytest.fail("background route must not post HID clicks"),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_press_key",
        lambda *a, **k: pytest.fail("background route must not post HID keys"),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_type_text",
        lambda *a, **k: pytest.fail("background route must not post HID text"),
    )
    return calls


# --- pure plans ---------------------------------------------------------------


def test_left_click_plan_primes_then_decoys_off_screen():
    plan = background_input.click_plan(10.0, 20.0)
    assert [(s.event_type, s.x, s.y, s.phase) for s in plan] == [
        (5, 10.0, 20.0, 2),  # mouseMoved primer at the target
        (1, -1.0, -1.0, 1),  # off-screen decoy down
        (2, -1.0, -1.0, 2),  # off-screen decoy up
        (1, 10.0, 20.0, 3),
        (2, 10.0, 20.0, 3),
    ]
    assert {s.button_number for s in plan} == {0}


def test_right_click_plan_has_no_decoy_and_stamps_right_button():
    plan = background_input.click_plan(1.0, 2.0, button="right")
    assert [s.event_type for s in plan] == [5, 3, 4]
    # A right-down stamped as button 0 is delivered as a left click.
    assert [s.button_number for s in plan[1:]] == [1, 1]
    middle = background_input.click_plan(1.0, 2.0, button="middle")
    assert [(s.event_type, s.button_number) for s in middle[1:]] == [(25, 2), (26, 2)]


def test_double_click_plan_increments_click_state():
    plan = background_input.click_plan(1.0, 2.0, count=2)
    targets = [s for s in plan if s.phase == 3]
    assert [s.click_state for s in targets] == [1, 1, 2, 2]
    assert targets[-1].delay_after_s == 0.0


def test_click_plan_rejects_unknown_button():
    with pytest.raises(ValueError):
        background_input.click_plan(0, 0, button="side")


def test_scroll_ticks_split_into_bounded_notches():
    assert background_input.scroll_ticks(0) == []
    assert background_input.scroll_ticks(-25) == [-10, -10, -5]
    assert background_input.scroll_ticks(7) == [7]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("BACKGROUND", "background"),
        ("foreground", "foreground"),
        ("forground", "foreground"),  # a typo never opts into the SPI
        ("", "auto"),
    ],
)
def test_delivery_mode_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(background_input.DELIVERY_ENV, raising=False)
    else:
        monkeypatch.setenv(background_input.DELIVERY_ENV, value)
    assert background_input.delivery_mode() == expected


def test_foreground_mode_disables_background_even_with_skylight(monkeypatch):
    monkeypatch.setattr(background_input, "skylight_available", lambda: True)
    monkeypatch.setenv(background_input.DELIVERY_ENV, "foreground")
    assert background_input.background_enabled() is False
    monkeypatch.setenv(background_input.DELIVERY_ENV, "auto")
    assert background_input.background_enabled() is True


def test_missing_frameworks_mean_unavailable_without_raising(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("no such framework")

    monkeypatch.setattr(background_input.ctypes, "CDLL", refuse)
    monkeypatch.setattr(background_input, "_LOADED", False)
    monkeypatch.setattr(background_input, "_SYMS", None)
    assert background_input.skylight_available() is False
    assert background_input.click(1, 2, 3.0, 4.0) is False
    assert background_input.type_text(1, "x") is False
    assert background_input.press_key(1, 36) is False


# --- routing in the backend ---------------------------------------------------


def test_coordinate_click_routes_to_pid_without_topmost_check(monkeypatch, background):
    snapshot = _snapshot()
    validations = []
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snap, **k: validations.append(k) or snap["window"],
    )
    result = backend.click("App", x=50, y=60, expected_snapshot=snapshot)
    assert validations == [{"point": (50.0, 60.0), "require_topmost": False}]
    assert background[0] == (
        "click",
        (4, 101, 50.0, 60.0),
        {"button": "left", "count": 1, "window_origin": (0.0, 0.0), "front_wid": 555},
    )
    # The user's front window (pid 999) gets keyboard focus back.
    assert background[1] == ("restore", (999, 555, 4, 101), {})
    assert result["mode"] == "SkyLight-click"
    assert result["route"] == "pid_events"
    assert result["effect"] == "unverifiable"
    assert result["focus_restored"] is True


def test_no_focus_restore_when_target_is_already_front(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 101))
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    result = backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert [call[0] for call in background] == ["click"]
    assert result["focus_restored"] is None


def test_focus_restored_to_other_window_of_same_app(monkeypatch, background):
    # The user edits window 555 of the same app while the agent clicks 101:
    # focus-without-raise made 101 key, so 555 must get focus back.
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "_frontmost_window", lambda: (4, 555))
    monkeypatch.setattr(background_input, "front_pid", lambda: 4)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    result = backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert background[1] == ("restore", (4, 555, 4, 101), {})
    assert result["focus_restored"] is True


def test_focus_restored_even_when_click_synthesis_fails(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )

    def explode(*_a, **_k):
        raise OSError("SPI failed after focusing the target")

    monkeypatch.setattr(background_input, "click", explode)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert exc.value.code == "action_failed"
    assert isinstance(exc.value.__cause__, OSError)
    assert background == [("restore", (999, 555, 4, 101), {})]


def test_focus_restore_runs_inside_the_gesture_transaction(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    owned = []
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: owned.append(background_input.GESTURE_LOCK._is_owned()) or True,
    )
    backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert owned == [True]


def _fake_syms(monkeypatch):
    log = []
    syms = {
        "source": 1,
        "mouse_event": lambda *a: 7,
        "scroll_event": lambda *a: 8,
        "set_field": lambda *a: None,
        "set_flags": lambda *a: None,
        "set_location": lambda *a: None,
        "set_window_location": lambda ev, x, y: log.append((ev, x, y)),
        "sl_post": lambda *a: None,
        "release": lambda *a: None,
    }
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    monkeypatch.setattr(background_input, "activate_without_raise", lambda *a: True)
    monkeypatch.setattr(background_input.time, "sleep", lambda *_: None)
    return log


def test_click_posts_nothing_when_focus_without_raise_fails(monkeypatch):
    log = _fake_syms(monkeypatch)
    posted = []
    background_input._syms()["sl_post"] = lambda pid, ev: posted.append(ev)
    monkeypatch.setattr(background_input, "activate_without_raise", lambda *a: False)
    assert background_input.click(4, 101, 5.0, 5.0) is False
    assert log == []
    assert posted == []


def test_scroll_fails_when_primer_cannot_be_created(monkeypatch):
    _fake_syms(monkeypatch)
    background_input._syms()["mouse_event"] = lambda *a: 0
    assert background_input.scroll(4, 101, 5.0, 5.0, lines_y=-3) is False


def test_event_record_rejects_out_of_bounds_and_non_heap_words(monkeypatch):
    import ctypes

    record = (ctypes.c_uint8 * 256)()
    record[4] = 0xF8  # record length
    record[8] = 10  # kCGEventKeyDown
    rec = ctypes.addressof(record)
    event = (ctypes.c_uint64 * 4)()  # a 32-byte event: offsets 0..24 only
    event[2] = 0xADE49453  # offset 16: junk, not a malloc block
    event[3] = rec  # offset 24: candidate record pointer
    sizes = {ctypes.addressof(event): 32, rec: 256}

    def malloc_size(ptr):
        assert ptr in sizes, "malloc_size must only see zone-owned pointers"
        return sizes[ptr]

    syms = {
        "malloc_size": malloc_size,
        "malloc_zone_from_ptr": lambda ptr: 1 if ptr in sizes else None,
        "event_type": lambda ev: 10,
    }
    monkeypatch.setattr(background_input, "_syms", lambda: syms)
    assert background_input._event_record(ctypes.addressof(event)) == rec
    syms["event_type"] = lambda ev: 11  # header type is not this event's
    assert background_input._event_record(ctypes.addressof(event)) is None
    syms["event_type"] = lambda ev: 10
    record[4] = 0x10  # header length is not a record's
    assert background_input._event_record(ctypes.addressof(event)) is None
    record[4] = 0xF8
    sizes[rec] = 16  # too small to be a record
    assert background_input._event_record(ctypes.addressof(event)) is None
    syms["malloc_zone_from_ptr"] = None
    assert background_input._event_record(ctypes.addressof(event)) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="needs CoreGraphics")
def test_event_record_found_on_real_events():
    if not background_input.skylight_available():
        pytest.skip("SkyLight SPI unavailable on this host")
    s = background_input._syms()
    for ev in (
        s["key_event"](s["source"], 36, True),
        s["key_event"](s["source"], 36, False),
        s["mouse_event"](s["source"], 1, background_input._CGPoint(1, 1), 0),
    ):
        try:
            assert background_input._event_record(ev) is not None
        finally:
            s["release"](ev)


def test_restore_skips_when_user_moved_to_third_app(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: 31)
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: pytest.fail("must not steal focus back from the user's new app"),
    )
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: pytest.fail("no re-activation")
    )
    assert backend._restore_user_focus((999, 555), 4, 101) is None


def test_gesture_events_released_when_posting_raises(monkeypatch):
    _fake_syms(monkeypatch)
    syms = background_input._syms()
    released, made = [], iter(range(1, 100))
    syms["release"] = lambda ev: released.append(ev)
    syms["mouse_event"] = lambda *a: next(made)

    def boom(*_a):
        raise OSError("post failed")

    syms["sl_post"] = boom
    with pytest.raises(OSError):
        background_input.click(4, 101, 5.0, 5.0)
    assert sorted(released) == [1, 2, 3, 4, 5]


def test_restore_falls_back_to_app_activation(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: None)
    previous = types.SimpleNamespace(processIdentifier=lambda: 999)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: previous)
    _install_module(
        monkeypatch,
        "AppKit",
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
    )
    activated = []
    monkeypatch.setattr(background_input, "front_process_matches", lambda *a: False)
    monkeypatch.setattr(
        background_input, "restore_focus_after_without_raise", lambda *a: False
    )
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: activated.append(pid) or True
    )
    # Focus record refused -> re-activate the user's app.
    assert backend._restore_user_focus((999, 555), 4, 101) is True
    # No user window on screen (pid, 0) -> app activation, no record.
    assert backend._restore_user_focus((999, 0), 4, 101) is True
    assert activated == [999, 999]
    # Same app, record refused: activation cannot pick a window.
    workspace.frontmostApplication = lambda: types.SimpleNamespace(
        processIdentifier=lambda: 4
    )
    assert backend._restore_user_focus((4, 555), 4, 101) is False


def test_restore_skips_when_workspace_detects_mid_gesture_switch(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: None)
    replacement = types.SimpleNamespace(processIdentifier=lambda: 31)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: replacement)
    _install_module(
        monkeypatch,
        "AppKit",
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
    )
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: pytest.fail("must not restore stale focus over the user's new app"),
    )
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: pytest.fail("must not re-activate")
    )
    assert backend._restore_user_focus((999, 555), 4, 101) is None


def test_restore_fails_closed_when_live_foreground_is_unknown(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: None)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: None)
    _install_module(
        monkeypatch,
        "AppKit",
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
    )
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: pytest.fail("unknown foreground must not receive a stale restore"),
    )
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: pytest.fail("must not re-activate")
    )
    assert backend._restore_user_focus((999, 555), 4, 101) is False


def test_click_fails_closed_without_a_focus_capture(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_frontmost_window", lambda: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert exc.value.code == "action_failed"
    assert background == []


def test_restore_error_does_not_mask_delivered_click(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )

    def explode(*_a):
        raise RuntimeError("activation raised")

    monkeypatch.setattr(backend, "_restore_user_focus", explode)
    result = backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert result["focus_restored"] is False
    assert "warning" in result


def test_type_text_rejects_unpaired_surrogate(monkeypatch, background):
    monkeypatch.setattr(
        backend,
        "_prepare_synthetic_action",
        lambda *a, **k: pytest.fail("invalid text must be rejected first"),
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.type_text("App", "a\ud800b")
    assert exc.value.code == "invalid_argument"


def test_unrestored_focus_is_reported_as_warning(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_restore_user_focus", lambda *a: False)
    result = backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert result["focus_restored"] is False
    assert "focus" in result["warning"]


def test_right_click_and_scroll_stamp_window_local_point(monkeypatch):
    log = _fake_syms(monkeypatch)
    assert background_input.click(
        4, 101, 550.0, 300.0, button="right", window_origin=(500.0, 200.0)
    )
    assert {(x, y) for _, x, y in log} == {(50.0, 100.0)}
    log.clear()
    assert background_input.scroll(
        4, 101, 550.0, 300.0, lines_y=-3, window_origin=(500.0, 200.0)
    )
    assert {(x, y) for _, x, y in log} == {(50.0, 100.0)}


def test_left_click_keeps_screen_point_recipe(monkeypatch):
    log = _fake_syms(monkeypatch)
    assert background_input.click(4, 101, 550.0, 300.0, window_origin=(500.0, 200.0))
    assert {(x, y) for _, x, y in log} == {(550.0, 300.0), (-1.0, -1.0)}


def test_foreground_mode_keeps_global_hid_and_topmost_check(monkeypatch):
    monkeypatch.setenv(background_input.DELIVERY_ENV, "foreground")
    snapshot = _snapshot()
    validations, clicks = [], []
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snap, **k: validations.append(k) or snap["window"],
    )
    monkeypatch.setattr(
        backend.ax_driver, "_cg_click", lambda *a, **k: clicks.append((a, k))
    )
    result = backend.click(
        "App", x=5, y=6, expected_snapshot=snapshot, mouse_button="right"
    )
    assert validations == [{"point": (5.0, 6.0)}]
    assert clicks == [((5.0, 6.0), {"clicks": 1, "button": "right"})]
    assert result["route"] == "global_hid"


def test_failed_background_click_raises_instead_of_falling_back(
    monkeypatch, background
):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(background_input, "click", lambda *a, **k: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", x=5, y=5, expected_snapshot=snapshot)
    assert exc.value.code == "action_failed"


@pytest.mark.parametrize(
    ("button", "count", "actions", "expected"),
    [
        ("left", 1, ["AXPress", "AXShowMenu"], "AXPress"),
        ("left", 2, ["AXPress", "AXOpen"], "AXOpen"),
        ("right", 1, ["AXPress", "AXShowMenu"], "AXShowMenu"),
    ],
)
def test_element_click_prefers_advertised_semantic_action(
    monkeypatch, background, button, count, actions, expected
):
    snapshot = _snapshot(
        elements=[{"index": 0, "role": "AXCell", "center": [9, 9], "actions": actions}]
    )
    performed = []
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda el, action: performed.append(action) or 0,
    )
    result = backend.click(
        "App",
        element_index=0,
        expected_snapshot=snapshot,
        mouse_button=button,
        click_count=count,
    )
    assert performed == [expected]
    assert result["mode"] == expected
    assert result["route"] == "accessibility"
    assert background == []


def test_element_without_semantic_action_gets_routed_pixel_gesture(
    monkeypatch, background
):
    snapshot = _snapshot(
        elements=[
            {"index": 0, "role": "AXGroup", "center": [9, 8], "actions": ["AXPress"]}
        ]
    )
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    result = backend.click(
        "App", element_index=0, expected_snapshot=snapshot, mouse_button="right"
    )
    assert background[0] == (
        "click",
        (4, 101, 9.0, 8.0),
        {"button": "right", "count": 1, "window_origin": (0.0, 0.0), "front_wid": 555},
    )
    assert result["button"] == "right"
    assert result["element_index"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"mouse_button": "back"}, "mouse button"), ({"click_count": 4}, "click_count")],
)
def test_click_rejects_unsupported_gestures(kwargs, message):
    with pytest.raises(errors.ComputerUseError, match=message) as exc:
        backend.click("App", x=1, y=1, expected_snapshot=_snapshot(), **kwargs)
    assert exc.value.code == "invalid_argument"


def test_type_text_goes_to_pid_without_requiring_frontmost(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    window_checks, focus_checks = [], []
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snap, **k: window_checks.append(k) or snap["window"],
    )
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda *a, **k: focus_checks.append(k),
    )
    result = backend.type_text("App", "héllo 世界")
    assert window_checks == [{"point": (200.0, 150.0), "require_topmost": False}]
    # The exact window must still be the app's focused AX window.
    assert focus_checks == [
        {"require_active_app": False, "require_exact_window_id": True}
    ]
    assert background == [("type_text", (4, "héllo 世界"), {})]
    assert result["mode"] == "SkyLight-unicode"
    assert result["route"] == "pid_events"


@pytest.mark.parametrize("focused_window_id", [202, None])
def test_background_text_rejects_identical_frame_wrong_or_unmapped_window(
    monkeypatch, background, focused_window_id
):
    snapshot = _snapshot()
    focused = object()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snap, **k: snap["window"],
    )
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: focused)
    monkeypatch.setattr(
        backend.ax_driver,
        "_point_size",
        lambda window: (0.0, 0.0, 400.0, 300.0),
    )
    monkeypatch.setattr(
        background_input, "ax_window_id", lambda window: focused_window_id
    )

    with pytest.raises(errors.ComputerUseError) as exc:
        backend.type_text("App", "must not reach another document")
    assert exc.value.code == "target_drift"
    assert background == []


def test_plain_and_modified_keys_go_to_pid(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    assert backend.press_key("App", "escape")["mode"] == "SkyLight-keycode"
    assert backend.hotkey("App", "Shift+Tab")["mode"] == "SkyLight-hotkey"
    assert background == [
        ("press_key", (4, 53, 0), {}),
        ("press_key", (4, 48, backend.MODIFIER_FLAGS["shift"]), {}),
    ]


def test_command_chords_stay_foreground(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    focus_checks = []
    monkeypatch.setattr(
        backend, "_validate_focused_window", lambda *a, **k: focus_checks.append(k)
    )
    posted = []
    _install_module(
        monkeypatch,
        "Quartz",
        CGEventCreateKeyboardEvent=lambda *a: object(),
        CGEventSetFlags=lambda *a: None,
        CGEventPost=lambda *a: posted.append(a),
        kCGHIDEventTap=0,
    )
    result = backend.hotkey("App", "Cmd+S")
    # Menu key equivalents only fire for the active app: keep the frontmost check.
    assert focus_checks == [{}]
    assert len(posted) == 2
    assert result["route"] == "global_hid"
    assert background == []


def test_finder_keeps_foreground_route(monkeypatch, background):
    snapshot = _snapshot(bundle_id="com.apple.finder")
    assert backend._background_delivery(snapshot) is False
    assert backend._keyboard_background(snapshot) is False


def test_scroll_routes_wheel_to_pid(monkeypatch, background):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda *a, **k: pytest.fail("wheel delivery does not need focus"),
    )
    result = backend.scroll("App", "down", expected_snapshot=snapshot)
    assert background == [
        (
            "scroll",
            (4, 101, 200.0, 150.0),
            {
                "lines_y": -10,
                "lines_x": 0,
                "window_origin": (0.0, 0.0),
            },
        )
    ]
    assert result["mode"] == "SkyLight-scroll"
    backend.scroll("App", "left", pages=0.5, expected_snapshot=snapshot)
    assert background[-1][2]["lines_x"] == 5
    assert background[-1][2]["lines_y"] == 0


@pytest.mark.parametrize(
    ("verified", "effect"),
    [(True, "confirmed"), (False, "suspected_noop"), (None, "unverifiable")],
)
def test_finish_action_reports_effect_and_route(verified, effect):
    result = backend._finish_action(
        "App", _snapshot(), {"mode": "AXSetValue"}, verified=verified, verification="v"
    )
    assert result["effect"] == effect
    assert result["route"] == "accessibility"


# --- live -------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("RAPID_MLX_LIVE_GUI") != "1",
    reason="live GUI test; set RAPID_MLX_LIVE_GUI=1 on a Mac with Accessibility granted",
)
def test_live_background_click_types_into_calculator_without_stealing_focus():
    import AppKit
    import ApplicationServices as AS  # noqa: N817  # conventional pyobjc alias
    import Quartz

    if not background_input.skylight_available():
        pytest.skip("SkyLight SPI unavailable on this host")
    workspace = AppKit.NSWorkspace.sharedWorkspace()

    def value(element, attribute):
        return AS.AXUIElementCopyAttributeValue(element, attribute, None)[1]

    def find(element, predicate, depth=0):
        if depth > 12:
            return None
        if predicate(element):
            return element
        for child in value(element, "AXChildren") or []:
            hit = find(child, predicate, depth + 1)
            if hit is not None:
                return hit
        return None

    def center(element):
        _, origin = AS.AXValueGetValue(
            value(element, "AXPosition"), AS.kAXValueCGPointType, None
        )
        _, size = AS.AXValueGetValue(
            value(element, "AXSize"), AS.kAXValueCGSizeType, None
        )
        return origin.x + size.width / 2, origin.y + size.height / 2

    def cursor():
        return Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))

    if (
        subprocess.run(["pgrep", "-x", "Calculator"], capture_output=True).returncode
        == 0
    ):
        pytest.skip("Calculator is already running; refusing to disturb it")
    front_before, cursor_before = (
        workspace.frontmostApplication().processIdentifier(),
        cursor(),
    )
    previous = backend._frontmost_window()
    subprocess.run(["open", "-g", "-a", "Calculator"], check=True)
    pid = None
    try:
        for _ in range(40):
            out = subprocess.run(
                ["pgrep", "-x", "Calculator"], capture_output=True, text=True
            )
            if out.stdout.split():
                pid = int(out.stdout.split()[0])
                break
            time.sleep(0.25)
        assert pid is not None
        time.sleep(2.0)
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        wid = next(
            int(w["kCGWindowNumber"])
            for w in windows
            if int(w.get("kCGWindowOwnerPID", -1)) == pid
            and w.get("kCGWindowLayer") == 0
        )
        app = AS.AXUIElementCreateApplication(pid)

        def button(label):
            return find(
                app,
                lambda e: (
                    value(e, "AXRole") == "AXButton"
                    and (value(e, "AXTitle") or value(e, "AXDescription")) == label
                ),
            )

        AS.AXUIElementPerformAction(button("All Clear"), "AXPress")
        time.sleep(0.3)
        for digit in "73":
            # One production transaction per click: focus is handed back
            # before the next click, and checked every time.
            with background_input.GESTURE_LOCK:
                try:
                    assert background_input.click(
                        pid, wid, *center(button(digit)), front_wid=previous[1]
                    )
                    time.sleep(0.05)
                finally:
                    restored = backend._restore_user_focus(previous, pid, wid)
            assert restored is not False
            time.sleep(0.25)
            assert workspace.frontmostApplication().processIdentifier() == front_before
            if previous[1]:
                # Keyboard focus is back on the user's window, not Calculator.
                assert backend._key_window_id(previous[0]) == previous[1]
        time.sleep(0.4)
        shown = []
        find(
            app,
            lambda e: (
                isinstance(value(e, "AXValue"), str)
                and shown.append(value(e, "AXValue"))
                and False
            ),
        )
        assert any("73" in text for text in shown), shown
        assert workspace.frontmostApplication().processIdentifier() == front_before
        after = cursor()
        assert (round(after.x), round(after.y)) == (
            round(cursor_before.x),
            round(cursor_before.y),
        )
    finally:
        # Quit only the instance this test launched, never one opened since.
        if pid is not None:
            running = (
                AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(
                    pid
                )
            )
            if running is not None and running.bundleIdentifier() == (
                "com.apple.calculator"
            ):
                running.terminate()


# --- pr_validate codex round 5 -------------------------------------------------


def test_forced_background_without_skylight_refuses_hid(monkeypatch):
    monkeypatch.setenv(background_input.DELIVERY_ENV, "background")
    monkeypatch.setattr(background_input, "skylight_available", lambda: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._background_delivery(_snapshot())
    assert exc.value.code == "action_failed"
    # Finder is foreground by design, even when background is forced.
    assert backend._background_delivery(_snapshot(bundle_id="com.apple.finder")) is (
        False
    )


def test_click_allocation_failure_posts_nothing(monkeypatch):
    _fake_syms(monkeypatch)
    posted, released, made = [], [], iter(range(1, 100))
    syms = background_input._syms()
    syms["sl_post"] = lambda *a: posted.append(a)
    syms["release"] = lambda ev: released.append(ev)
    # The fourth event of the plan cannot be created.
    syms["mouse_event"] = lambda *a: (lambda n: 0 if n == 4 else n)(next(made))
    activated = []
    monkeypatch.setattr(
        background_input, "activate_without_raise", lambda *a: activated.append(a)
    )
    assert background_input.click(4, 101, 5.0, 5.0) is False
    assert posted == [] and activated == []
    assert released == [1, 2, 3]


def test_key_and_text_allocation_failure_posts_nothing(monkeypatch):
    _fake_syms(monkeypatch)
    syms = background_input._syms()
    posted = []
    syms["sl_post"] = lambda *a: posted.append(a)
    syms["auth_factory"] = None
    made = iter(range(1, 100))
    syms["key_event"] = lambda *a: (lambda n: 0 if n == 2 else n)(next(made))
    assert background_input.press_key(4, 51) is False
    made = iter(range(1, 100))
    syms["key_event"] = lambda *a: (lambda n: 0 if n == 5 else n)(next(made))
    assert background_input.type_text(4, "abc") is False
    assert posted == []


def test_frontmost_window_is_front_apps_ax_key_window(monkeypatch):
    # The key window comes from AXFocusedWindow, not CG z-order (which skips
    # floating panels and does not identify the key window).
    monkeypatch.setattr(background_input, "front_pid", lambda: 77)
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: {77: 9}.get(pid))
    assert backend._frontmost_window() == (77, 9)
    monkeypatch.setattr(background_input, "front_pid", lambda: 88)
    # Unmappable key window: restore by activating that app, never a guess.
    assert backend._frontmost_window() == (88, 0)


def test_restore_skips_when_user_picked_another_window_of_their_app(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: 999)
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 777)
    monkeypatch.setattr(
        background_input,
        "restore_focus_after_without_raise",
        lambda *a: pytest.fail("must not refocus the stale window"),
    )
    assert backend._restore_user_focus((999, 555), 4, 101) is None


def test_restore_skips_when_user_switched_to_another_window_of_target(monkeypatch):
    # The target app is front, but its key window is not the one the agent
    # clicked: the user chose that window, so their app is not re-activated.
    monkeypatch.setattr(background_input, "front_pid", lambda: 4)
    monkeypatch.setattr(background_input, "front_process_matches", lambda *a: True)
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 202)
    monkeypatch.setattr(backend, "_activate_app", lambda pid: pytest.fail("steal"))
    assert backend._restore_user_focus((999, 555), 4, 101) is None


def test_restore_reactivates_user_app_when_target_self_activated(monkeypatch):
    monkeypatch.setattr(background_input, "front_pid", lambda: 4)
    monkeypatch.setattr(background_input, "front_process_matches", lambda *a: True)
    monkeypatch.setattr(backend, "_key_window_id", lambda pid: 101)
    activated = []
    monkeypatch.setattr(
        backend, "_activate_app", lambda pid: activated.append(pid) or True
    )
    assert backend._restore_user_focus((999, 555), 4, 101) is True
    assert activated == [999]


def test_defocus_record_names_the_front_window(monkeypatch):
    posted = []
    monkeypatch.setattr(background_input, "_syms", lambda: {"ok": True})
    front, target = background_input._PSN(0, 1), background_input._PSN(0, 2)
    names = {id(front): "front", id(target): "target"}
    monkeypatch.setattr(background_input, "_front_psn", lambda: front)
    monkeypatch.setattr(background_input, "_psn_for_window", lambda wid, pid: target)
    monkeypatch.setattr(
        background_input,
        "_post_record",
        lambda psn, rec: (
            posted.append(
                (
                    names[id(psn)],
                    int.from_bytes(bytes(rec[0x3C:0x40]), "little"),
                    rec[0x8A],
                )
            )
            or True
        ),
    )
    assert background_input.activate_without_raise(4, 101, front_wid=555)
    assert posted == [("front", 555, 0x02), ("target", 101, 0x01)]
    posted.clear()
    # Unknown front window: cua's recipe (target id in the defocus record).
    assert background_input.activate_without_raise(4, 101)
    assert posted[0] == ("front", 101, 0x02)


@pytest.mark.parametrize("count", [0, 4, 1.9, "2", True, None])
def test_click_count_rejects_non_integral_values(count):
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", x=1, y=1, click_count=count)
    assert exc.value.code == "invalid_argument"


def test_focus_failure_rolls_back_the_defocus(monkeypatch):
    posted = []
    monkeypatch.setattr(background_input, "_syms", lambda: {"ok": True})
    front, target = background_input._PSN(0, 1), background_input._PSN(0, 2)
    names = {id(front): "front", id(target): "target"}
    monkeypatch.setattr(background_input, "_front_psn", lambda: front)
    monkeypatch.setattr(background_input, "_psn_for_window", lambda wid, pid: target)

    def post(psn, rec):
        posted.append((names[id(psn)], rec[0x8A]))
        return psn is front

    monkeypatch.setattr(background_input, "_post_record", post)
    assert background_input.activate_without_raise(4, 101, front_wid=555) is False
    assert posted == [("front", 0x02), ("target", 0x01), ("front", 0x01)]


def test_defocus_failure_posts_no_focus(monkeypatch):
    posted = []
    monkeypatch.setattr(background_input, "_syms", lambda: {"ok": True})
    front, target = background_input._PSN(0, 1), background_input._PSN(0, 2)
    names = {id(front): "front", id(target): "target"}
    monkeypatch.setattr(background_input, "_front_psn", lambda: front)
    monkeypatch.setattr(background_input, "_psn_for_window", lambda wid, pid: target)
    monkeypatch.setattr(
        background_input,
        "_post_record",
        lambda psn, rec: posted.append(names[id(psn)]) and False,
    )
    assert background_input.activate_without_raise(4, 101, front_wid=555) is False
    assert posted == ["front"]


def _interrupt_after(n):
    """sl_post stub raising KeyboardInterrupt on the n-th post (1-based)."""
    posted = []

    def post(pid, ev):
        posted.append(ev)
        if len(posted) == n:
            raise KeyboardInterrupt

    return posted, post


@pytest.mark.parametrize("button", ["left", "right"])
def test_interrupted_click_still_posts_the_up(monkeypatch, button):
    _fake_syms(monkeypatch)
    syms = background_input._syms()
    made = iter(range(1, 100))
    syms["mouse_event"] = lambda *a: next(made)
    plan = background_input.click_plan(5.0, 5.0, button=button)
    down = next(
        i
        for i, st in enumerate(plan)
        if st.event_type == background_input._EVENT_TYPES[button][0]
    )
    posted, syms["sl_post"] = _interrupt_after(down + 1)
    with pytest.raises(KeyboardInterrupt):
        background_input.click(4, 101, 5.0, 5.0, button=button)
    up_type = background_input._EVENT_TYPES[button][1]
    up = next(j for j in range(down + 1, len(plan)) if plan[j].event_type == up_type)
    assert posted[-1] == up + 1  # events are numbered from 1


def test_interrupted_key_press_still_posts_the_up(monkeypatch):
    _fake_syms(monkeypatch)
    syms = background_input._syms()
    made = iter(range(1, 100))
    syms["key_event"] = lambda *a: next(made)
    monkeypatch.setattr(
        background_input,
        "_post_key_event",
        lambda pid, ev, **k: syms["sl_post"](pid, ev),
    )
    posted, syms["sl_post"] = _interrupt_after(1)
    monkeypatch.setattr(background_input.time, "sleep", lambda *_: None)
    with pytest.raises(KeyboardInterrupt):
        background_input.press_key(4, 36)
    assert posted == [1, 2]


def test_interrupted_typing_releases_the_held_character(monkeypatch):
    _fake_syms(monkeypatch)
    syms = background_input._syms()
    made = iter(range(1, 100))
    syms["key_event"] = lambda *a: next(made)
    syms["set_unicode"] = lambda *a: None
    monkeypatch.setattr(
        background_input,
        "_post_key_event",
        lambda pid, ev, **k: syms["sl_post"](pid, ev),
    )
    posted, syms["sl_post"] = _interrupt_after(3)  # down of the 2nd character
    with pytest.raises(KeyboardInterrupt):
        background_input.type_text(4, "abc")
    assert posted == [1, 2, 3, 4]


@pytest.mark.parametrize("raises", [True, False])
@pytest.mark.parametrize("primitive", ["press_key", "type_text", "scroll"])
def test_primitive_errors_surface_as_action_failed(
    monkeypatch, background, primitive, raises
):
    snapshot = _snapshot()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **k: snap["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)

    def explode(*_a, **_k):
        if not raises:
            return False  # the primitive refused (allocation, no focus, ...)
        raise OSError("SPI failed")

    monkeypatch.setattr(background_input, primitive, explode)
    with pytest.raises(errors.ComputerUseError) as exc:
        if primitive == "press_key":
            backend.press_key("App", "Tab", expected_snapshot=snapshot)
        elif primitive == "type_text":
            monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
            backend.type_text("App", "hi")
        else:
            backend.scroll("App", "down", expected_snapshot=snapshot)
    assert exc.value.code == "action_failed"


def test_allocation_exception_releases_created_events(monkeypatch):
    _fake_syms(monkeypatch)
    released = []
    background_input._syms()["release"] = released.append

    def create(index):
        if index == 2:
            raise OSError("create failed")
        return index + 1

    with pytest.raises(OSError):
        background_input._allocate(4, create)
    assert released == [1, 2]


def test_type_text_bounds_length(monkeypatch, background):
    too_long = "x" * (background_input.MAX_TYPE_TEXT_CHARS + 1)
    monkeypatch.setattr(
        backend, "_prepare_synthetic_action", lambda *a: pytest.fail("no dispatch")
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.type_text("App", too_long)
    assert exc.value.code == "invalid_argument"


def test_type_text_primitive_refuses_oversized_text_before_allocating(monkeypatch):
    _fake_syms(monkeypatch)
    background_input._syms()["key_event"] = lambda *a: pytest.fail("no allocation")
    too_long = "x" * (background_input.MAX_TYPE_TEXT_CHARS + 1)
    assert background_input.type_text(4, too_long) is False
