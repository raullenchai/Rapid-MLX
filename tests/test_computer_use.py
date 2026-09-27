"""Unit tests for the model-agnostic computer-use layer (no AX calls)."""

# ruff: noqa: N802 - PyObjC test doubles intentionally mirror Objective-C names.

import json
import sys
import types

import pytest

from rapid_mlx.computer_use import ax_driver, backend, errors


def test_error_payload_carries_recovery_hints():
    exc = errors.ComputerUseError("element_not_found", "element 5 vanished")
    payload = exc.to_payload()
    assert payload["ok"] is False
    assert payload["error"]["code"] == "element_not_found"
    assert any("get-app-state" in hint for hint in payload["error"]["recovery"])


def test_unknown_code_gets_empty_recovery():
    exc = errors.ComputerUseError("weird_code", "mystery")
    assert exc.recovery == ()


def test_hotkey_requires_modifier_and_key():
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.hotkey("Google Chrome", "A")
    assert excinfo.value.code == "unsupported_key"
    with pytest.raises(errors.ComputerUseError):
        backend.hotkey("Google Chrome", "Cmd+NotAKey")


def test_press_key_rejects_multi_key():
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("Google Chrome", "Cmd+C")
    assert excinfo.value.code == "unsupported_key"


def test_keycode_map_matches_ax_driver():
    # backend and driver must agree on the HID mapping
    for key, code in ax_driver.KEYCODE_MAP.items():
        assert backend.ax_driver.KEYCODE_MAP[key] == code
    assert backend.ax_driver.KEYCODE_MAP["a"] == 0x00
    assert backend.ax_driver.KEYCODE_MAP["m"] == 0x2E


def test_click_requires_element_index_or_coordinates():
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.click("Google Chrome")
    assert excinfo.value.code == "invalid_argument"


def test_snapshot_cache_expires(monkeypatch):
    cache = backend.SnapshotCache()
    now = 1000.0
    monkeypatch.setattr(backend.time, "time", lambda: now)
    cache.put("app", 0, {"snapshot_id": "a"})
    assert cache.get("app", 0)["snapshot_id"] == "a"
    monkeypatch.setattr(backend.time, "time", lambda: now + backend.SNAPSHOT_TTL_S + 1)
    assert cache.get("app", 0) is None


def test_snapshot_cache_evicts_oldest(monkeypatch):
    cache = backend.SnapshotCache()
    for i in range(backend.SNAPSHOT_CACHE_MAX + 4):
        monkeypatch.setattr(backend.time, "time", lambda i=i: 1000.0 + i)
        cache.put(f"app{i}", 0, {"snapshot_id": i})
    keys = {key[0] for key in cache._entries}
    assert "app3" not in keys  # oldest evicted
    assert len(cache._entries) <= backend.SNAPSHOT_CACHE_MAX


def test_snapshot_cache_separates_pixel_and_ax_only_entries():
    cache = backend.SnapshotCache()
    cache.put("app", 0, {"kind": "ax"}, screenshot=False)
    cache.put("app", 0, {"kind": "pixels"}, screenshot=True)
    assert cache.get("app", 0, screenshot=False) == {"kind": "ax"}
    assert cache.get("app", 0, screenshot=True) == {"kind": "pixels"}


def test_resolve_app_unknown_name():
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._resolve_app("Definitely Not Running App XYZ")
    assert excinfo.value.code in {"app_not_found", "unsupported_platform"}


# ---------------------------------------------------------------- cache / cli / read_url


def test_snapshot_cache_ttl_and_eviction(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(backend.time, "time", lambda: now["t"])
    cache = backend.SnapshotCache()
    cache.put("App", 0, {"snap": 1})
    assert cache.get("App", 0) == {"snap": 1}
    now["t"] += backend.SNAPSHOT_TTL_S + 1
    assert cache.get("App", 0) is None  # expired
    for i in range(backend.SNAPSHOT_CACHE_MAX + 2):
        cache.put(f"App{i}", 0, {"i": i})
    assert len(cache._entries) <= backend.SNAPSHOT_CACHE_MAX


def test_read_url_uses_axvalue_and_escapes_app_name(monkeypatch):
    import types

    class _FakeElement:
        pass

    state = {"axvalue": "https://example.com/x"}

    def fake_get(_element, attr):
        return state["axvalue"] if attr == "AXValue" else None

    def fake_collect(app, keep_elements=True, max_windows=1):
        return [{"element": _FakeElement()}]

    monkeypatch.setattr(backend.ax_driver, "_get", fake_get)
    monkeypatch.setattr(backend.ax_driver, "collect", fake_collect)
    assert backend.read_url("Google Chrome") == "https://example.com/x"

    # no AXValue anywhere -> AppleScript fallback with escaped app name
    state["axvalue"] = None
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(stdout="", stderr="", returncode=1)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    assert backend.read_url('Weird "App"') == ""
    script = captured["cmd"][2]
    assert 'Weird \\"App\\"' in script  # quotes escaped, no injection
    assert captured["cmd"][0] == "osascript"


def test_read_url_empty_value_without_osascript_result(monkeypatch):
    import types

    class _FakeElement:
        pass

    monkeypatch.setattr(backend.ax_driver, "_get", lambda e, a: None)
    monkeypatch.setattr(
        backend.ax_driver, "collect", lambda *a, **k: [{"element": _FakeElement()}]
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda cmd, **k: types.SimpleNamespace(
            stdout="https://ok.example", stderr="", returncode=0
        ),
    )
    assert backend.read_url("Safari") == "https://ok.example"


def test_coordinate_click_activates_target_without_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "_resolve_app", lambda app: calls.append(app))
    monkeypatch.setattr(
        backend,
        "get_app_state",
        lambda *args, **kwargs: pytest.fail(
            "coordinate click must not take a snapshot"
        ),
    )
    monkeypatch.setattr(backend.ax_driver, "_cg_click", lambda *args, **kwargs: None)
    assert backend.click("Target App", x=4, y=9)["at"] == [4, 9]
    assert calls == ["Target App"]


def test_synthetic_keyboard_actions_activate_target_app(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "_resolve_app", lambda app: calls.append(app))
    monkeypatch.setattr(backend.ax_driver, "_type_text", lambda text: None)
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda *args, **kwargs: None)

    backend.type_text("Target App", "secret")
    backend.press_key("Target App", "return")
    assert calls == ["Target App", "Target App"]


def test_hotkey_and_scroll_activate_target_before_posting(monkeypatch):
    calls = []
    scroll_events = []
    monkeypatch.setattr(backend, "_resolve_app", lambda app: calls.append(app))
    fake_quartz = types.SimpleNamespace(
        CGEventCreateKeyboardEvent=lambda *_args: object(),
        CGEventSetFlags=lambda *_args: None,
        CGEventPost=lambda *_args: None,
        CGEventCreateScrollWheelEvent=lambda *_args: (
            scroll_events.append(_args) or object()
        ),
        kCGHIDEventTap=0,
        kCGScrollEventUnitLine=0,
    )
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    backend.hotkey("Target App", "Cmd+A")
    backend.scroll("Target App", "down")
    assert calls == ["Target App", "Target App"]
    assert scroll_events == [(None, 0, 1, -10)]

    scroll_events.clear()
    backend.scroll("Target App", "left", pages=0.5)
    assert scroll_events == [(None, 0, 2, 0, 5)]


def test_get_app_state_uses_requested_window_without_screenshot(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app: (object(), {"name": "Target App", "bundleId": "x", "pid": 7}),
    )

    def fake_collect(app, **kwargs):
        captured.update(kwargs)
        return [
            {
                "target_id": "t000",
                "role": "AXButton",
                "text": "OK",
                "actions": ["AXPress"],
                "rect": [0, 0, 10, 10],
            }
        ]

    monkeypatch.setattr(backend.ax_driver, "collect", fake_collect)
    state = backend.get_app_state(
        "Target App", window_index=2, screenshot=False, use_cache=False
    )
    assert state["window_index"] == 2
    assert captured["window_index"] == 2


def test_cli_capabilities_and_error_envelope(capsys):
    from rapid_mlx.computer_use import cli

    assert cli.main(["capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "get-app-state" in payload["observation"]

    assert (
        cli.main(["set-value", "--app", "X", "--element-index", "1", "--text", "v"])
        != 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] in {
        "app_not_found",
        "element_not_found",
        "ax_set_failed",
        "unsupported_platform",
    }


# ---------------------------------------------------------------- full mocked provider contracts


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _RunningApp:
    def __init__(
        self, name="Target App", bundle="com.test.app", pid=42, *, active=True, policy=0
    ):
        self._name = name
        self._bundle = bundle
        self._pid = pid
        self._active = active
        self._policy = policy
        self.calls = []

    def localizedName(self):
        return self._name

    def bundleIdentifier(self):
        return self._bundle

    def processIdentifier(self):
        return self._pid

    def isActive(self):
        return self._active

    def activationPolicy(self):
        return self._policy

    def activateWithOptions_(self, value):
        self.calls.append(("activate", value))

    def unhide(self):
        self.calls.append(("unhide",))


class _Workspace:
    def __init__(self, apps):
        self._apps = apps

    def runningApplications(self):
        return self._apps


class _NSWorkspace:
    workspace = _Workspace([])

    @classmethod
    def sharedWorkspace(cls):
        return cls.workspace


def test_cli_dispatches_every_command(monkeypatch, capsys, tmp_path):
    from rapid_mlx.computer_use import cli

    calls = []
    monkeypatch.setattr(backend, "permissions", lambda: {"accessibility": True})
    monkeypatch.setattr(backend, "list_apps", lambda: [{"name": "A"}])
    monkeypatch.setattr(backend, "list_windows", lambda app: [{"title": app}])
    monkeypatch.setattr(
        backend,
        "get_app_state",
        lambda *a, **k: {"elements": [], "screenshot_png": b"png"},
    )
    for name in (
        "click",
        "set_value",
        "type_text",
        "press_key",
        "hotkey",
        "scroll",
        "perform_secondary_action",
    ):
        monkeypatch.setattr(
            backend,
            name,
            lambda *a, _name=name, **k: calls.append((_name, a, k)) or {"mode": _name},
        )

    commands = [
        ["permissions"],
        ["list-apps"],
        ["list-windows", "--app", "A"],
        [
            "get-app-state",
            "--app",
            "A",
            "--no-screenshot",
            "--refresh",
            "--png-out",
            str(tmp_path / "x.png"),
        ],
        ["click", "--app", "A", "--x", "1", "--y", "2", "--click-count", "2"],
        ["set-value", "--app", "A", "--element-index", "2", "--text", "v"],
        ["type-text", "--app", "A", "--text", "v"],
        ["press-key", "--app", "A", "--key", "return"],
        ["hotkey", "--app", "A", "--key", "cmd+a"],
        [
            "scroll",
            "--app",
            "A",
            "--direction",
            "right",
            "--pages",
            ".5",
            "--x",
            "1",
            "--y",
            "2",
        ],
        [
            "perform-secondary-action",
            "--app",
            "A",
            "--element-index",
            "3",
            "--action",
            "AXShowMenu",
        ],
    ]
    for command in commands:
        assert cli.main(command) == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True
    assert (tmp_path / "x.png").read_bytes() == b"png"
    assert {item[0] for item in calls} == {
        "click",
        "set_value",
        "type_text",
        "press_key",
        "hotkey",
        "scroll",
        "perform_secondary_action",
    }


def test_cli_stdin_empty_and_mouse_button_guards(monkeypatch, capsys):
    from rapid_mlx.computer_use import cli

    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: "stdin-value"))
    monkeypatch.setattr(backend, "set_value", lambda *a: {"mode": "set"})
    monkeypatch.setattr(backend, "type_text", lambda *a: {"mode": "type"})
    assert (
        cli.main(["set-value", "--app", "A", "--element-index", "1", "--text-stdin"])
        == 0
    )
    capsys.readouterr()
    assert cli.main(["type-text", "--app", "A", "--text-stdin"]) == 0
    capsys.readouterr()
    for command in (
        ["set-value", "--app", "A", "--element-index", "1"],
        ["type-text", "--app", "A"],
        ["click", "--app", "A", "--x", "1", "--y", "2", "--mouse-button", "right"],
    ):
        assert cli.main(command) == 1
        assert (
            json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_argument"
        )


def test_snapshot_payload_encodes_png():
    from rapid_mlx.computer_use import cli

    payload = cli._snapshot_payload({"x": 1, "screenshot_png": b"abc"}, True)
    assert payload == {"x": 1, "screenshot_png_base64": "YWJj"}
    assert cli._snapshot_payload({"x": 1, "screenshot_png": b"abc"}, False) == {"x": 1}


def test_resolve_app_by_name_bundle_and_pid(monkeypatch):
    regular = _RunningApp(active=False)
    hidden = _RunningApp("Hidden", "com.test.hidden", 7, active=False, policy=1)
    _NSWorkspace.workspace = _Workspace([hidden, regular])
    fake_as = _install_module(
        monkeypatch, "ApplicationServices", NSWorkspace=_NSWorkspace
    )
    monkeypatch.setattr(
        backend, "_ax_app_element", lambda app: ("ax", app.processIdentifier())
    )

    assert backend._resolve_app("target")[1]["pid"] == 42
    assert backend._resolve_app("com.test.app")[0] == ("ax", 42)
    assert backend._resolve_app("pid:7")[1]["pid"] == 7
    with pytest.raises(errors.ComputerUseError, match="bad pid"):
        backend._resolve_app("pid:nope")
    with pytest.raises(errors.ComputerUseError, match="no running app"):
        backend._resolve_app("missing")
    assert fake_as.NSWorkspace is _NSWorkspace


def test_ax_app_element_activation_and_fallback(monkeypatch):
    calls = []
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementCreateApplication=lambda pid: ("element", pid),
        AXUIElementSetAttributeValue=lambda *args: calls.append(args),
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    app = _RunningApp()
    assert backend._ax_app_element(app) == ("element", 42)
    assert app.calls == [("activate", 2)]

    app.activateWithOptions_ = lambda _: (_ for _ in ()).throw(RuntimeError("old"))
    backend._ax_app_element(app)
    assert app.calls[-1] == ("unhide",)
    app.unhide = lambda: (_ for _ in ()).throw(RuntimeError("hidden"))
    backend._ax_app_element(app)


def _target(index=0, *, role="AXButton", actions=None, rect=None, element=object()):
    return {
        "target_id": f"t{index:03d}",
        "role": role,
        "subrole": None,
        "text": "Label",
        "actions": ["AXPress"] if actions is None else actions,
        "rect": [1.2, 2.3, 10.0, 20.0] if rect is None else rect,
        "center": [6, 12],
        "element": element,
    }


def test_get_app_state_cache_snapshot_and_window_errors(monkeypatch):
    backend._CACHE.clear()
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app: (object(), {"name": "A", "bundleId": "b", "pid": 4}),
    )
    monkeypatch.setattr(backend.ax_driver, "collect", lambda *a, **k: [_target()])
    monkeypatch.setattr(backend, "screenshot_window", lambda *a: b"png-data")
    monkeypatch.setattr(backend.time, "time", lambda: 10.0)
    state = backend.get_app_state("A", screenshot=True, use_cache=False)
    assert state["element_count"] == 1
    assert state["tree_text"] == "[0] AXButton* Label"
    assert state["screenshot_png"] == b"png-data"
    monkeypatch.setattr(backend, "_resolve_app", lambda _: pytest.fail("cache missed"))
    assert backend.get_app_state("A", screenshot=True) is state

    backend._CACHE.clear()
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app: (object(), {"name": "A", "bundleId": "b", "pid": 4}),
    )
    monkeypatch.setattr(backend.ax_driver, "collect", lambda *a, **k: [])
    with pytest.raises(errors.ComputerUseError, match="window index"):
        backend.get_app_state("A", window_index=2, screenshot=False, use_cache=False)


def test_screenshot_window_success_and_failures(monkeypatch):
    windows = [
        {"kCGWindowOwnerName": "Other", "kCGWindowLayer": 0, "kCGWindowNumber": 1},
        {"kCGWindowOwnerName": "Target App", "kCGWindowLayer": 2, "kCGWindowNumber": 2},
        {"kCGWindowOwnerName": "Target App", "kCGWindowLayer": 0, "kCGWindowNumber": 3},
    ]
    quartz = _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *_: windows,
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )

    def run_ok(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[-1]).write_bytes(b"x" * 9000)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend.subprocess, "run", run_ok)
    assert len(backend.screenshot_window("Target")) == 9000
    with pytest.raises(errors.ComputerUseError, match="index 1"):
        backend.screenshot_window("Target", 1)
    quartz.CGWindowListCopyWindowInfo = lambda *_: []
    with pytest.raises(errors.ComputerUseError, match="no on-screen"):
        backend.screenshot_window("Target")
    quartz.CGWindowListCopyWindowInfo = lambda *_: windows
    monkeypatch.setattr(
        backend.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1)
    )
    with pytest.raises(errors.ComputerUseError, match="no usable"):
        backend.screenshot_window("Target")


def test_element_live_and_read_helpers(monkeypatch):
    snapshot = {"snapshot_id": "s", "app": {"name": "A"}, "elements": [{"index": 2}]}
    assert backend._element(snapshot, 2)["index"] == 2
    with pytest.raises(errors.ComputerUseError, match="element 3"):
        backend._element(snapshot, 3)
    monkeypatch.setattr(
        backend.ax_driver, "collect", lambda *a, **k: [_target(2, element="live")]
    )
    assert backend._live_element(snapshot, 2) == "live"
    with pytest.raises(errors.ComputerUseError, match="no longer"):
        backend._live_element(snapshot, 3)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: "value")
    assert backend._read_value("live") == "value"
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: 3)
    assert backend._read_value("live") is None


def test_element_click_ax_and_fallback(monkeypatch):
    snapshot = {"elements": [{"index": 0, "center": [6, 12], "actions": ["AXPress"]}]}
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a: "live")
    fake_as = _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda *a: 0,
    )
    assert backend.click("A", element_index=0)["mode"] == "AXPress"
    fake_as.AXUIElementPerformAction = lambda *a: 1
    clicks = []
    monkeypatch.setattr(
        backend.ax_driver, "_cg_click", lambda *a, **k: clicks.append((a, k))
    )
    assert backend.click("A", element_index=0, click_count=2)["mode"] == "CGEvent-click"
    monkeypatch.setattr(
        backend,
        "_live_element",
        lambda *a: (_ for _ in ()).throw(
            errors.ComputerUseError("element_not_found", "gone")
        ),
    )
    backend.click("A", element_index=0)
    assert len(clicks) == 2


def test_set_value_and_synthetic_fill_paths(monkeypatch):
    synthetic_fill = backend._synthetic_fill
    snapshot = {
        "app": {"name": "A"},
        "window_index": 0,
        "elements": [{"index": 0, "center": [1, 2]}],
    }
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a: "live")
    module = _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXValueAttribute="AXValue",
        AXUIElementSetAttributeValue=lambda *a: 0,
    )
    monkeypatch.setattr(backend, "_read_value", lambda _: "wanted")
    assert backend.set_value("A", 0, "wanted")["verified"] is True
    module.AXUIElementSetAttributeValue = lambda *a: 1
    monkeypatch.setattr(backend, "_synthetic_fill", lambda *a: {"mode": "fallback"})
    assert backend.set_value("A", 0, "wanted")["mode"] == "fallback"
    monkeypatch.setattr(backend, "_synthetic_fill", synthetic_fill)

    events = []
    monkeypatch.setattr(
        backend.ax_driver, "_cg_click", lambda *a, **k: events.append("click")
    )
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda *a, **k: events.append("key")
    )
    monkeypatch.setattr(
        backend.ax_driver, "_type_text", lambda *a: events.append("type")
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: [_target(role="AXTextField", element="fresh")],
    )
    monkeypatch.setattr(backend, "_read_value", lambda _: "wanted")
    result = backend._synthetic_fill(snapshot, 0, "wanted")
    assert result["verified"] is True and events == ["click", "key", "key", "type"]
    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("AX")),
    )
    assert backend._synthetic_fill(snapshot, 0, "wanted")["verified"] is None


def test_press_hotkey_scroll_and_secondary_paths(monkeypatch):
    calls = []
    app_services = _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda *a: 0,
    )
    _install_module(
        monkeypatch,
        "Quartz",
        CGEventCreateKeyboardEvent=lambda *a: object(),
        CGEventSetFlags=lambda *a: None,
        CGEventPost=lambda *a: None,
        CGEventCreateScrollWheelEvent=lambda *a: object(),
        kCGHIDEventTap=0,
        kCGScrollEventUnitLine=0,
    )
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app: calls.append(("resolve", app))
    )
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda *a, **k: calls.append(("key", a, k))
    )
    monkeypatch.setattr(
        backend.ax_driver, "_cg_click", lambda *a, **k: calls.append(("click", a, k))
    )
    assert backend.press_key("A", "a")["key"] == "a"
    assert backend.hotkey("A", "Ctrl+Shift+Tab")["mode"] == "CGEvent-hotkey"
    with pytest.raises(errors.ComputerUseError, match="unknown modifier"):
        backend.hotkey("A", "Meta+A")
    with pytest.raises(errors.ComputerUseError, match="unsupported hotkey"):
        backend.hotkey("A", "Cmd+?")
    with pytest.raises(errors.ComputerUseError, match="unsupported direction"):
        backend.scroll("A", "around")
    backend.scroll("A", "up", x=3, y=4)

    snapshot = {"elements": [{"index": 0, "actions": ["AXShowMenu"]}]}
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a: "live")
    assert (
        backend.perform_secondary_action("A", 0, "AXShowMenu")["mode"]
        == "AXPerformAction"
    )
    with pytest.raises(errors.ComputerUseError, match="not advertised"):
        backend.perform_secondary_action("A", 0, "AXPick")
    app_services.AXUIElementPerformAction = lambda *a: 5
    with pytest.raises(errors.ComputerUseError, match="failed"):
        backend.perform_secondary_action("A", 0, "AXShowMenu")


def test_permissions_apps_windows_and_read_url_failures(monkeypatch):
    regular = _RunningApp()
    accessory = _RunningApp("Accessory", policy=1)
    _NSWorkspace.workspace = _Workspace([accessory, regular])
    app_services = _install_module(
        monkeypatch,
        "ApplicationServices",
        NSWorkspace=_NSWorkspace,
        AXIsProcessTrustedWithOptions=lambda _: True,
        kAXTrustedCheckOptionPrompt="prompt",
    )
    quartz = _install_module(
        monkeypatch,
        "Quartz",
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )
    quartz.CGPreflightScreenCaptureAccess = lambda: False
    assert backend.permissions()["accessibility"] is True
    assert backend.permissions()["screen_recording"] is False
    quartz.CGPreflightScreenCaptureAccess = lambda: (_ for _ in ()).throw(
        RuntimeError()
    )
    assert backend.permissions()["screen_recording"] is None
    assert backend.list_apps() == [
        {"name": "Target App", "bundleId": "com.test.app", "pid": 42}
    ]

    monkeypatch.setattr(
        backend, "_resolve_app", lambda app: (object(), {"name": "target app"})
    )
    windows = [
        {"kCGWindowOwnerName": "other", "kCGWindowLayer": 0},
        {"kCGWindowOwnerName": "target app", "kCGWindowLayer": 1},
        {
            "kCGWindowOwnerName": "target app",
            "kCGWindowLayer": 0,
            "kCGWindowName": "Main",
            "kCGWindowBounds": {"X": 1, "Y": 2, "Width": 3, "Height": 4},
        },
    ]
    quartz.CGWindowListCopyWindowInfo = lambda *_: windows
    assert backend.list_windows("A")[0]["title"] == "Main"
    quartz.CGWindowListCopyWindowInfo = lambda *_: []
    with pytest.raises(errors.ComputerUseError, match="no on-screen"):
        backend.list_windows("A")

    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()),
    )
    monkeypatch.setattr(
        backend.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    assert backend.read_url("A") == ""
    assert app_services.NSWorkspace is _NSWorkspace


def test_ax_driver_tree_collect_and_events(monkeypatch):
    attrs = {
        "root": {"AXRole": "AXWindow", "AXChildren": ["button"]},
        "button": {
            "AXRole": "AXButton",
            "AXTitle": "Go\nNow",
            "AXActions": ["AXPress"],
            "AXSubrole": "",
            "AXPosition": "p",
            "AXSize": "s",
            "AXChildren": [],
        },
    }
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementCopyAttributeValue",
        lambda e, a, _: (0, attrs.get(e, {}).get(a)),
    )
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementCopyActionNames",
        lambda e, _: (0, attrs.get(e, {}).get("AXActions")),
    )
    point = types.SimpleNamespace(x=1, y=2)
    size = types.SimpleNamespace(width=10, height=20)
    monkeypatch.setattr(
        ax_driver,
        "AS",
        types.SimpleNamespace(kAXValueCGPointType=1, kAXValueCGSizeType=2),
    )
    monkeypatch.setattr(
        ax_driver,
        "AXValueGetValue",
        lambda value, *_: (0, point if value == "p" else size),
    )
    out = []
    ax_driver._walk("root", 0, out, [0])
    assert out[0]["text"] == "Go Now"
    assert out[0]["rect"] == (1.0, 2.0, 10.0, 20.0)
    assert ax_driver._get("missing", "x") is None
    assert ax_driver._action_names("missing") == []
    assert ax_driver._point_size("missing") is None

    monkeypatch.setattr(ax_driver, "_app_element", lambda _: "app")
    monkeypatch.setattr(
        ax_driver, "_get", lambda e, a: ["root"] if a == "AXWindows" else None
    )
    monkeypatch.setattr(ax_driver, "AXUIElementSetAttributeValue", lambda *a: None)
    monkeypatch.setattr(
        ax_driver, "_walk", lambda e, d, out, c: out.append(dict(_target(element=e)))
    )
    assert ax_driver.collect("A", keep_elements=False)[0]["center"] == [6, 12]
    assert "element" not in ax_driver.collect("A", keep_elements=False)[0]
    with pytest.raises(ValueError):
        ax_driver.collect("A", window_index=-1)

    events = []
    monkeypatch.setattr(ax_driver.time, "sleep", lambda _: None)
    monkeypatch.setattr(ax_driver, "CGEventCreateMouseEvent", lambda *a: ("mouse", a))
    monkeypatch.setattr(ax_driver, "CGEventCreateKeyboardEvent", lambda *a: ("key", a))
    monkeypatch.setattr(ax_driver, "CGEventPost", lambda *a: events.append(("post", a)))
    monkeypatch.setattr(
        ax_driver,
        "CGEventSetIntegerValueField",
        lambda *a: events.append(("clicks", a)),
    )
    monkeypatch.setattr(
        ax_driver, "CGEventSetFlags", lambda *a: events.append(("flags", a))
    )
    monkeypatch.setattr(
        ax_driver,
        "CGEventKeyboardSetUnicodeString",
        lambda *a: events.append(("unicode", a)),
    )
    ax_driver._cg_click(1, 2, 2)
    ax_driver._press_key(4, ax_driver.FLAG_COMMAND)
    ax_driver._type_text("A!")
    assert any(e[0] == "unicode" for e in events)
    assert ax_driver._keycode_for("a") == 0


def test_ax_driver_app_collect_retries_and_press(monkeypatch):
    app = _RunningApp()
    ax_element = object()
    fake_as = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: _Workspace([app]))
    )
    monkeypatch.setattr(ax_driver, "AS", fake_as)
    monkeypatch.setattr(
        ax_driver, "AXUIElementCreateApplication", lambda pid: ax_element
    )
    sets = []
    monkeypatch.setattr(
        ax_driver, "AXUIElementSetAttributeValue", lambda *a: sets.append(a)
    )
    monkeypatch.setattr(ax_driver.time, "sleep", lambda _: None)
    assert ax_driver._app_element("target") is ax_element
    with pytest.raises(SystemExit, match="not found"):
        ax_driver._app_element("missing")
    monkeypatch.setattr(ax_driver, "AS", None)
    with pytest.raises(RuntimeError, match="macOS"):
        ax_driver._app_element("target")

    monkeypatch.setattr(ax_driver, "_app_element", lambda _: "app")
    attempts = {"n": 0}
    monkeypatch.setattr(
        ax_driver, "_get", lambda e, a: ["window"] if a == "AXWindows" else None
    )

    def walk(e, d, out, c):
        attempts["n"] += 1
        out.append(
            {
                **_target(element=e),
                "role": "AXWebArea" if attempts["n"] > 1 else "AXButton",
            }
        )
        c[0] += 1

    monkeypatch.setattr(ax_driver, "_walk", walk)
    collected = ax_driver.collect("A", keep_elements=True)
    assert collected and attempts["n"] == 2 and sets

    monkeypatch.setattr(ax_driver, "_app_element", lambda _: "app")
    monkeypatch.setattr(
        ax_driver, "_get", lambda e, a: ["window"] if a == "AXWindows" else None
    )

    def press_walk(e, d, out, c):
        out.append(_target(element="live"))

    monkeypatch.setattr(ax_driver, "_walk", press_walk)
    monkeypatch.setattr(ax_driver, "AXUIElementPerformAction", lambda *a: 0)
    assert ax_driver.press([_target()], "t000", "A")["mode"] == "AXPress"
    monkeypatch.setattr(ax_driver, "AXUIElementPerformAction", lambda *a: 1)
    monkeypatch.setattr(ax_driver, "_cg_click", lambda *a: None)
    assert ax_driver.press([_target()], "t000", "A")["mode"] == "CGEvent-click"
    assert ax_driver.press([_target()], "t999", "A")["ok"] is False
    monkeypatch.setattr(ax_driver, "_walk", lambda *a: None)
    assert ax_driver.press([_target()], "t000", "A")["ok"] is False


def test_ax_driver_press_rejects_missing_geometry(monkeypatch):
    monkeypatch.setattr(ax_driver, "_app_element", lambda _: "app")
    monkeypatch.setattr(
        ax_driver, "_get", lambda e, a: ["window"] if a == "AXWindows" else None
    )
    monkeypatch.setattr(
        ax_driver,
        "_walk",
        lambda e, d, out, c: out.append(_target(rect=None, actions=[], element="live")),
    )
    # Override helper's default rect explicitly after construction.
    original = [_target()]

    def no_geom(e, d, out, c):
        item = _target(actions=[], element="live")
        item["rect"] = None
        out.append(item)

    monkeypatch.setattr(ax_driver, "_walk", no_geom)
    assert ax_driver.press(original, "t000", "A")["ok"] is False


def test_remaining_small_backend_branches(monkeypatch):
    cache = backend.SnapshotCache()
    assert cache.get("missing", 0) is None

    hidden = _RunningApp("Hidden", pid=7, active=False, policy=1)
    _NSWorkspace.workspace = _Workspace([hidden])
    _install_module(monkeypatch, "ApplicationServices", NSWorkspace=_NSWorkspace)
    monkeypatch.setattr(backend, "_ax_app_element", lambda app: object())
    with pytest.raises(errors.ComputerUseError, match="no running app"):
        backend._resolve_app("pid:8")

    snapshot = {"app": {"name": "A"}, "elements": [{"index": 0, "center": [1, 2]}]}
    monkeypatch.setattr(backend.ax_driver, "_cg_click", lambda *a, **k: None)
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda *a, **k: None)
    monkeypatch.setattr(backend.ax_driver, "_type_text", lambda *a: None)
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: [
            _target(role="AXButton", element="skip"),
            _target(role="AXTextField", element=None),
        ],
    )
    assert backend._synthetic_fill(snapshot, 0, "wanted")["verified"] is None

    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: [{"element": None}, {"element": "live"}],
    )
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: None)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout="", returncode=1),
    )
    assert backend.read_url("A") == ""


def test_cli_unreachable_fallback_and_module_entry(monkeypatch, capsys):
    import runpy

    from rapid_mlx.computer_use import cli

    original_build_parser = cli.build_parser
    parser = types.SimpleNamespace(
        parse_args=lambda argv: types.SimpleNamespace(subcommand="bogus")
    )
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    assert cli.main([]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_argument"

    monkeypatch.setattr(cli, "build_parser", original_build_parser)
    monkeypatch.setattr(sys, "argv", ["rapid_mlx.computer_use", "capabilities"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("rapid_mlx.computer_use.__main__", run_name="__main__")
    assert excinfo.value.code == 0
    capsys.readouterr()


def test_ax_driver_limits_menu_bar_and_old_unicode(monkeypatch):
    ax_driver._walk("ignored", ax_driver.MAX_DEPTH + 1, [], [0])

    attrs = {
        "root": {"AXRole": "AXWindow", "AXChildren": ["child"]},
        "child": {
            "AXRole": "AXButton",
            "AXTitle": "Child",
            "AXActions": ["AXPress"],
            "AXChildren": [],
        },
    }
    monkeypatch.setattr(ax_driver, "MAX_NODES", 1)
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementCopyAttributeValue",
        lambda e, a, _: (0, attrs.get(e, {}).get(a)),
    )
    monkeypatch.setattr(
        ax_driver,
        "AXUIElementCopyActionNames",
        lambda e, _: (0, attrs.get(e, {}).get("AXActions")),
    )
    monkeypatch.setattr(ax_driver, "_point_size", lambda e: None)
    out = []
    ax_driver._walk("root", 0, out, [0])
    assert len(out) == 1

    monkeypatch.setattr(ax_driver, "_app_element", lambda _: "app")
    monkeypatch.setattr(ax_driver, "_get", lambda *a: [])
    monkeypatch.setattr(ax_driver, "AXUIElementCreateSystemWide", lambda: "system")
    monkeypatch.setattr(ax_driver, "AXUIElementSetAttributeValue", lambda *a: None)
    monkeypatch.setattr(
        ax_driver, "_walk", lambda e, d, out, c: out.append(_target(element=e))
    )
    assert ax_driver.collect("A", keep_elements=True)[0]["element"] == "system"

    monkeypatch.setattr(ax_driver, "MAX_NODES", 1)
    monkeypatch.setattr(
        ax_driver, "_get", lambda e, a: ["one", "two"] if a == "AXWindows" else None
    )
    monkeypatch.setattr(
        ax_driver,
        "_walk",
        lambda e, d, out, c: (
            out.append({**_target(element=e), "role": "AXWebArea"}),
            c.__setitem__(0, 1),
        ),
    )
    assert len(ax_driver.collect("A", keep_elements=True)) == 1

    calls = []
    monkeypatch.setattr(ax_driver, "CGEventCreateKeyboardEvent", lambda *a: object())

    def old_unicode(*args):
        if len(args) == 3:
            raise TypeError("old signature")
        calls.append(args)

    monkeypatch.setattr(ax_driver, "CGEventKeyboardSetUnicodeString", old_unicode)
    monkeypatch.setattr(ax_driver, "CGEventPost", lambda *a: None)
    monkeypatch.setattr(ax_driver.time, "sleep", lambda _: None)
    ax_driver._type_text("!")
    assert len(calls) == 2


def test_ax_driver_cli_main_modes(monkeypatch, capsys, tmp_path):
    target = {k: v for k, v in _target().items() if k != "element"}
    monkeypatch.setattr(ax_driver, "collect", lambda app: [target])

    monkeypatch.setattr(sys, "argv", ["ax_driver", "--app", "A", "--dump", "-"])
    ax_driver.main()
    assert "t000" in capsys.readouterr().out

    out = tmp_path / "targets.json"
    monkeypatch.setattr(
        sys, "argv", ["ax_driver", "--app", "A", "--dump", str(out), "--max-nodes", "7"]
    )
    ax_driver.main()
    assert json.loads(out.read_text())[0]["target_id"] == "t000"
    assert "1 targets" in capsys.readouterr().err

    monkeypatch.setattr(ax_driver, "press", lambda *a: {"ok": True})
    monkeypatch.setattr(sys, "argv", ["ax_driver", "--app", "A", "--press", "t000"])
    with pytest.raises(SystemExit) as excinfo:
        ax_driver.main()
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    monkeypatch.setattr(sys, "argv", ["ax_driver", "--app", "A"])
    ax_driver.main()
    assert "t000 AXButton Label" in capsys.readouterr().out


def test_ax_driver_non_macos_stub():
    if hasattr(ax_driver, "_macos_only"):
        with pytest.raises(RuntimeError, match="macOS"):
            ax_driver._macos_only()
