"""Unit tests for the model-agnostic computer-use layer (no AX calls)."""

import json

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


def test_resolve_app_unknown_name():
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._resolve_app("Definitely Not Running App XYZ")
    assert excinfo.value.code == "app_not_found"


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
    }
