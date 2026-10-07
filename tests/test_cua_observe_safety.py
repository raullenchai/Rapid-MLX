"""Observation without focus theft, plus the hands safety fixes that ride with it."""

import sys
import time
import types

import pytest

from rapid_mlx.computer_use import ax_driver, backend, background_input, errors
from rapid_mlx.cua import gates
from rapid_mlx.cua.loop import CUARun


def _snapshot(element: dict) -> dict:
    window = {
        "index": 0,
        "window_id": "cg:101",
        "x": 0,
        "y": 0,
        "width": 400,
        "height": 300,
    }
    return {
        "snapshot_id": "s1",
        "observed_at": time.time(),
        "app": {"name": "App", "bundleId": "com.example.app", "pid": 4},
        "window_index": 0,
        "window_id": "cg:101",
        "window": window,
        "elements": [{"index": 0, "center": [50, 60], **element}],
    }


def _fake_application_services(monkeypatch, perform):
    """Stand-in for the PyObjC module so these tests also run off macOS."""
    module = types.ModuleType("ApplicationServices")
    module.AXUIElementPerformAction = perform
    module.kAXErrorSuccess = 0
    monkeypatch.setitem(sys.modules, "ApplicationServices", module)
    return module


@pytest.fixture
def no_input(monkeypatch):
    """Fail on any delivered press or pixel click."""
    monkeypatch.setattr(
        backend,
        "_pixel_click",
        lambda *a, **k: pytest.fail("focus_only must not click this control"),
    )
    _fake_application_services(
        monkeypatch,
        lambda *a: pytest.fail("focus_only must never perform an AX action"),
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")


# --- focus_only never commits ----------------------------------------------------


def test_focus_only_focuses_button_via_axfocused_without_pressing(
    monkeypatch, no_input
):
    snapshot = _snapshot(
        {"role": "AXButton", "label": "Delete", "actions": ["AXPress"]}
    )
    state = {"focused": None}
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: state["focused"])

    def set_attr(element, attr, value):
        assert (element, attr, value) == ("live", "AXFocused", True)
        state["focused"] = element
        return ax_driver.kAXErrorSuccess

    monkeypatch.setattr(backend.ax_driver, "AXUIElementSetAttributeValue", set_attr)
    result = backend.click("App", 0, expected_snapshot=snapshot, focus_only=True)
    assert result["mode"] == "AXFocused"
    assert result["verified"] is True


def test_focus_only_refuses_commit_control_it_cannot_focus(monkeypatch, no_input):
    snapshot = _snapshot(
        {"role": "AXButton", "label": "Delete", "actions": ["AXPress"]}
    )
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: None)
    monkeypatch.setattr(
        backend.ax_driver, "AXUIElementSetAttributeValue", lambda *a: -25205
    )
    # Foreground delivery: the background key-window retry would reach this
    # Mac's real window server.
    monkeypatch.setattr(backend, "_background_delivery", lambda snap: False)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", 0, expected_snapshot=snapshot, focus_only=True)
    assert exc.value.code == "synthetic_input_blocked"


@pytest.mark.parametrize(
    "role", ["AXTextField", "AXTextArea", "AXComboBox", "AXRow", "AXStaticText"]
)
def test_focus_only_never_pixel_clicks_when_axfocused_fails(monkeypatch, role):
    # AX cannot prove a click is non-committing (a web input may submit on
    # click and still read as AXTextField), so no role gets a pixel fallback.
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: None)
    monkeypatch.setattr(
        backend.ax_driver, "AXUIElementSetAttributeValue", lambda *a: -25205
    )
    # Foreground delivery: the background key-window retry would reach this
    # Mac's real window server.
    monkeypatch.setattr(backend, "_background_delivery", lambda snap: False)
    monkeypatch.setattr(
        backend, "_pixel_click", lambda *a, **k: pytest.fail("must not click")
    )
    snapshot = _snapshot({"role": role, "label": "x", "actions": []})
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", 0, expected_snapshot=snapshot, focus_only=True)
    assert exc.value.code == "synthetic_input_blocked"


def test_focus_only_rejects_coordinate_click(monkeypatch):
    monkeypatch.setattr(
        backend, "_pixel_click", lambda *a, **k: pytest.fail("must not click")
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click(
            "App", x=10, y=20, expected_snapshot=_snapshot({}), focus_only=True
        )
    assert exc.value.code == "invalid_argument"


def test_plain_click_keeps_semantic_press(monkeypatch):
    snapshot = _snapshot({"role": "AXButton", "label": "OK", "actions": ["AXPress"]})
    pressed = []
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    _fake_application_services(
        monkeypatch, lambda el, action: pressed.append(action) or 0
    )
    result = backend.click("App", 0, expected_snapshot=snapshot)
    assert pressed == ["AXPress"]
    assert result["mode"] == "AXPress"


# --- secure fields and identity ---------------------------------------------------


def test_sign_in_detects_secure_subrole():
    snapshot = {
        "elements": [
            {"label": "Sign in", "role": "AXButton"},
            {"label": "", "role": "AXTextField", "subrole": "AXSecureTextField"},
        ]
    }
    assert gates.looks_like_sign_in(snapshot)
    snapshot["elements"][1]["subrole"] = ""
    assert not gates.looks_like_sign_in(snapshot)


def test_window_identity_includes_bundle():
    snap = {"app": {"pid": 4, "bundleId": "com.a"}, "window_index": 0}
    other = {"app": {"pid": 4, "bundleId": "com.b"}, "window_index": 0}
    assert CUARun._window_identity(snap)[1] == "com.a"
    assert CUARun._window_identity(snap) != CUARun._window_identity(other)


def test_app_exit_is_typed_not_system_exit(monkeypatch):
    def gone(*_a, **_k):
        raise ax_driver.AppNotFoundError("app not found")

    monkeypatch.setattr(backend.ax_driver, "_app_element", gone)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._focused_ax_element({"name": "App", "pid": 4})
    assert exc.value.code == "target_drift"
    assert not issubclass(ax_driver.AppNotFoundError, SystemExit)


# --- observation without activation ---------------------------------------------


@pytest.mark.parametrize(
    ("enabled", "bundle", "activates"),
    [
        (True, "com.example.app", False),
        (True, "com.apple.finder", True),
        (True, "", True),
        (False, "com.example.app", True),
    ],
)
def test_observation_activates_matrix(monkeypatch, enabled, bundle, activates):
    monkeypatch.setattr(background_input, "background_enabled", lambda: enabled)
    assert backend.observation_activates({"bundleId": bundle}) is activates


def test_observation_activates_without_identity(monkeypatch):
    monkeypatch.setattr(background_input, "background_enabled", lambda: True)
    assert backend.observation_activates(None) is True


@pytest.mark.parametrize("enabled", [True, False])
def test_loop_observes_without_activation_on_background_route(
    monkeypatch, tmp_path, enabled
):
    monkeypatch.setattr(background_input, "background_enabled", lambda: enabled)
    seen = []
    snapshot = _snapshot({"role": "AXButton"})
    monkeypatch.setattr(
        backend, "get_app_state", lambda app, **k: seen.append(k) or snapshot
    )
    loop = CUARun.__new__(CUARun)
    loop.backend_app = "pid:4"
    loop.window_id = "cg:101"
    loop._trusted_transient_window_id = None
    loop.expected_app = {"pid": 4, "bundleId": "com.example.app"}
    loop._get_app_state(screenshot=False)
    assert ("activate" in seen[0]) is enabled
    if enabled:
        assert seen[0]["activate"] is False


# --- codex round 1 on PR 2 ------------------------------------------------------


class _Running:
    def __init__(self, active=False, start=1.0):
        self.active = active
        self.start = start
        self.activations = 0

    def isActive(self):  # noqa: N802 - mirrors NSRunningApplication
        return self.active

    def activateWithOptions_(self, _options):  # noqa: N802
        self.activations += 1
        self.active = True
        return True


def _borrow_setup(monkeypatch, running, start=1.0):
    monkeypatch.setattr(background_input, "background_enabled", lambda: True)
    monkeypatch.setattr(backend.ax_driver, "_application_for_pid", lambda pid: running)
    monkeypatch.setattr(
        backend,
        "_resolved_app_info",
        lambda r: {
            "pid": 4,
            "bundleId": "com.example.app",
            "name": "App",
            "processStartTime": start,
        },
    )
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **kw: snap["window"]
    )
    snapshot = _snapshot({"role": "AXTextField"})
    snapshot["app"]["processStartTime"] = 1.0
    return snapshot


def test_borrow_foreground_activates_exact_process(monkeypatch):
    running = _Running()
    snapshot = _borrow_setup(monkeypatch, running)
    backend._borrow_foreground(snapshot)
    assert running.activations == 1


def test_borrow_foreground_never_activates_for_a_stale_window(monkeypatch):
    running = _Running()
    snapshot = _borrow_setup(monkeypatch, running)

    def stale(snap, **kw):
        assert kw == {"require_topmost": False}
        raise errors.ComputerUseError("target_drift", "window closed")

    monkeypatch.setattr(backend, "_validate_snapshot_window", stale)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "target_drift"
    assert running.activations == 0


def test_borrow_foreground_refuses_recycled_pid(monkeypatch):
    running = _Running()
    snapshot = _borrow_setup(monkeypatch, running, start=2.0)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "target_drift"
    assert running.activations == 0


def test_borrow_foreground_reactivates_on_foreground_route(monkeypatch):
    # An activating observation does not prove the app is still frontmost.
    running = _Running()
    snapshot = _borrow_setup(monkeypatch, running)
    monkeypatch.setattr(background_input, "background_enabled", lambda: False)
    backend._borrow_foreground(snapshot)
    assert running.activations == 1


def test_save_and_synthetic_fill_borrow_foreground_first(monkeypatch):
    # The foreground (global HID, Cmd+A) fill path; background delivery fills
    # without taking the foreground (test_computer_use_offspace.py).
    monkeypatch.setattr(background_input, "background_enabled", lambda: False)
    order = []
    monkeypatch.setattr(
        backend, "_borrow_foreground", lambda snap: order.append("borrow")
    )

    def stop(*_a, **_k):
        order.append("validate")
        raise errors.ComputerUseError("target_drift", "stop")

    monkeypatch.setattr(backend, "_validate_snapshot_window", stop)
    snapshot = _snapshot({"role": "AXTextField"})
    for call in (
        lambda: backend._save_menu_candidate(snapshot),
        lambda: backend._synthetic_fill(snapshot, 0, "x"),
    ):
        order.clear()
        with pytest.raises(errors.ComputerUseError):
            call()
        assert order == ["borrow", "validate"]


def test_ax_driver_cli_maps_late_app_exit(monkeypatch):
    import sys

    monkeypatch.setattr(ax_driver, "collect", lambda app: [{"target_id": "t000"}])

    def gone(*_a):
        raise ax_driver.AppNotFoundError("app not found: 'A'")

    monkeypatch.setattr(ax_driver, "press", gone)
    monkeypatch.setattr(sys, "argv", ["ax_driver", "--app", "A", "--press", "t000"])
    with pytest.raises(SystemExit, match="app not found"):
        ax_driver.main()


def test_borrow_foreground_refuses_recycled_pid_already_active(monkeypatch):
    running = _Running(active=True)
    snapshot = _borrow_setup(monkeypatch, running, start=2.0)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "target_drift"


def test_borrow_foreground_revalidates_identity_while_polling(monkeypatch):
    running = _Running()
    snapshot = _borrow_setup(monkeypatch, running)
    starts = iter([1.0, 2.0])
    monkeypatch.setattr(
        backend,
        "_resolved_app_info",
        lambda r: {
            "pid": 4,
            "bundleId": "com.example.app",
            "name": "App",
            "processStartTime": next(starts),
        },
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "target_drift"


# --- coverage of the identity and focus edges -------------------------------------


def test_same_process_refuses_an_exited_app(monkeypatch):
    monkeypatch.setattr(backend.ax_driver, "_application_for_pid", lambda pid: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._same_process(
            {"pid": 4, "bundleId": "com.example.app", "processStartTime": 1.0}
        )
    assert exc.value.code == "target_drift"


def test_borrow_foreground_skips_an_already_active_target(monkeypatch):
    running = _Running(active=True)
    snapshot = _borrow_setup(monkeypatch, running)
    backend._borrow_foreground(snapshot)
    assert running.activations == 0


def test_borrow_foreground_types_activation_errors(monkeypatch):
    running = _Running()

    def refuse(_options):
        raise RuntimeError("activation refused")

    running.activateWithOptions_ = refuse
    snapshot = _borrow_setup(monkeypatch, running)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "action_failed"


def test_same_process_requires_launch_time(monkeypatch):
    monkeypatch.setattr(
        backend.ax_driver,
        "_application_for_pid",
        lambda pid: pytest.fail("incomplete identity must not be resolved"),
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._same_process({"pid": 4, "bundleId": "com.example.app"})
    assert exc.value.code == "target_drift"


def test_borrow_foreground_fails_fast_when_activation_refused(monkeypatch):
    running = _Running()
    running.activateWithOptions_ = lambda _options: False
    snapshot = _borrow_setup(monkeypatch, running)
    monkeypatch.setattr(backend.time, "sleep", lambda *_: pytest.fail("no polling"))
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "action_failed"


def test_borrow_foreground_times_out_when_target_stays_inactive(monkeypatch):
    running = _Running()
    running.activateWithOptions_ = lambda _options: True  # never becomes active
    snapshot = _borrow_setup(monkeypatch, running)
    clock = iter(range(0, 100))
    monkeypatch.setattr(backend.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(backend.time, "sleep", lambda *_: None)
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._borrow_foreground(snapshot)
    assert exc.value.code == "target_drift"


def test_focus_without_commit_edges(monkeypatch):
    snapshot = _snapshot({"role": "AXButton"})
    assert backend._focus_without_commit(snapshot, None) is None
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: "live")
    assert backend._focus_without_commit(snapshot, "live") == "AXFocusVerified"


def test_focus_only_on_already_focused_field_skips_activation(monkeypatch, no_input):
    snapshot = _snapshot({"role": "AXTextField", "label": "Name", "actions": []})
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: "live")
    monkeypatch.setattr(backend, "_background_delivery", lambda snap: True)
    checks = []
    monkeypatch.setattr(
        backend, "_validate_focused_window", lambda *a, **k: checks.append(k)
    )
    result = backend.click("App", 0, expected_snapshot=snapshot, focus_only=True)
    assert result["mode"] == "AXFocusVerified"
    assert checks == [{"require_active_app": False}]
