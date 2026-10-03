"""Observation without focus theft, plus the hands safety fixes that ride with it."""

import time

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


@pytest.fixture
def no_input(monkeypatch):
    """Fail on any delivered press or pixel click."""
    monkeypatch.setattr(
        backend,
        "_pixel_click",
        lambda *a, **k: pytest.fail("focus_only must not click this control"),
    )
    import ApplicationServices as AS  # noqa: N813, N817

    monkeypatch.setattr(
        AS,
        "AXUIElementPerformAction",
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
    with pytest.raises(errors.ComputerUseError) as exc:
        backend.click("App", 0, expected_snapshot=snapshot, focus_only=True)
    assert exc.value.code == "synthetic_input_blocked"


def test_focus_only_still_clicks_non_committing_targets(monkeypatch):
    snapshot = _snapshot({"role": "AXRow", "label": "item", "actions": []})
    clicks = []
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: None)
    monkeypatch.setattr(
        backend.ax_driver, "AXUIElementSetAttributeValue", lambda *a: -25205
    )
    monkeypatch.setattr(
        backend,
        "_pixel_click",
        lambda snap, x, y, **k: clicks.append((x, y)) or {"mode": "click"},
    )
    backend.click("App", 0, expected_snapshot=snapshot, focus_only=True)
    assert clicks == [(50.0, 60.0)]


def test_plain_click_keeps_semantic_press(monkeypatch):
    snapshot = _snapshot({"role": "AXButton", "label": "OK", "actions": ["AXPress"]})
    pressed = []
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    import ApplicationServices as AS  # noqa: N813, N817

    monkeypatch.setattr(
        AS,
        "AXUIElementPerformAction",
        lambda el, action: pressed.append(action) or AS.kAXErrorSuccess,
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
