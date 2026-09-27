"""Unit tests for the model-agnostic computer-use layer (no AX calls)."""

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
