"""Unit tests for the model-agnostic computer-use layer (no AX calls)."""

# ruff: noqa: N802 - PyObjC test doubles intentionally mirror Objective-C names.

import json
import sys
import threading
import types

import pytest

from rapid_mlx.computer_use import ax_driver, backend, errors


def _window(window_id=101, index=0, x=0, y=0, width=100, height=100):
    opaque_id = window_id if isinstance(window_id, str) else f"cg:{window_id}"
    return {
        "index": index,
        "window_id": opaque_id,
        "title": "Main",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def _stable_snapshot(*, window_id=101, elements=None, observed_at=100.0):
    window = _window(window_id=window_id)
    return {
        "snapshot_id": "planned",
        "observed_at": observed_at,
        "app": {"name": "A", "bundleId": "b", "pid": 4},
        "window_index": 0,
        "window_id": window["window_id"],
        "window": window,
        "elements": elements or [],
    }


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


def _install_save_menu(monkeypatch, items, *, edited_values=(None, None)):
    app_element = {"AXMenuBar": {"AXRole": "AXMenuBar", "AXChildren": items}}
    focused = {
        "AXDocument": "file:///tmp/document.txt",
        "AXEdited": edited_values[0],
    }
    pressed = []
    monkeypatch.setattr(backend, "_validate_snapshot_window", lambda snapshot: None)
    monkeypatch.setattr(backend, "_validate_focused_window", lambda snapshot: None)
    monkeypatch.setattr(backend.ax_driver, "_app_element", lambda *a, **k: app_element)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: focused)
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_action_names",
        lambda element: list(element.get("actions", [])),
    )

    def perform(element, action):
        pressed.append((element, action))
        focused["AXEdited"] = edited_values[1]
        return backend.ax_driver.kAXErrorSuccess

    monkeypatch.setattr(backend.ax_driver, "AXUIElementPerformAction", perform)
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    return pressed


def _save_item(title="Save", modifiers=0, enabled=True):
    return {
        "AXRole": "AXMenuItem",
        "AXTitle": title,
        "AXMenuItemCmdChar": "S",
        "AXMenuItemCmdModifiers": modifiers,
        "AXEnabled": enabled,
        "actions": ["AXPress"],
        "AXChildren": [],
    }


def test_native_save_uses_unique_command_and_stays_unverified_without_axedited(
    monkeypatch,
):
    save = _save_item()
    pressed = _install_save_menu(
        monkeypatch,
        [save, _save_item("Save As", 3), _save_item("Duplicate", 1)],
    )
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    binding = backend.inspect_save_document("pid:4", snapshot)
    result = backend.save_document(
        "pid:4", snapshot, expected_identity=tuple(binding["save_identity"])
    )
    assert pressed == [(save, "AXPress")]
    assert result["executed"] is True
    assert result["verified"] is None
    assert result["verification_source"] == "unverified"
    assert "could not be verified" in result["verification"]


def test_native_save_verifies_only_exact_edited_transition(monkeypatch):
    save = _save_item()
    _install_save_menu(monkeypatch, [save], edited_values=(True, False))
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    binding = backend.inspect_save_document("pid:4", snapshot)
    result = backend.save_document(
        "pid:4", snapshot, expected_identity=tuple(binding["save_identity"])
    )
    assert result["verified"] is True
    assert result["verification_source"] == "ax_edited_same_document"


@pytest.mark.parametrize("failure", ["document", "menu_bar"])
def test_native_save_requires_stable_document_and_accessible_menu(monkeypatch, failure):
    _install_save_menu(monkeypatch, [_save_item()])
    if failure == "document":
        monkeypatch.setattr(backend, "_focused_ax_window", lambda app: {})
    else:
        monkeypatch.setattr(
            backend.ax_driver, "_app_element", lambda *a, **k: {"AXMenuBar": None}
        )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.inspect_save_document(
            "pid:4", _stable_snapshot(observed_at=backend.time.time())
        )
    assert excinfo.value.code in {"target_drift", "element_not_found"}


def test_native_save_menu_walk_is_depth_bounded(monkeypatch):
    item = _save_item()
    for _ in range(backend.SAVE_MENU_MAX_DEPTH + 2):
        item = {"AXRole": "AXMenu", "AXChildren": [item]}
    _install_save_menu(monkeypatch, [item])

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.inspect_save_document(
            "pid:4", _stable_snapshot(observed_at=backend.time.time())
        )
    assert excinfo.value.code == "element_not_found"


def test_native_save_reports_rejected_axpress(monkeypatch):
    _install_save_menu(monkeypatch, [_save_item()])
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    binding = backend.inspect_save_document("pid:4", snapshot)
    monkeypatch.setattr(backend.ax_driver, "AXUIElementPerformAction", lambda *a: 1)

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.save_document(
            "pid:4", snapshot, expected_identity=tuple(binding["save_identity"])
        )
    assert excinfo.value.code == "action_failed"


def _install_autosave_document(monkeypatch, document):
    monkeypatch.setattr(backend, "_validate_snapshot_window", lambda snapshot: {})
    monkeypatch.setattr(
        backend, "_validate_focused_window", lambda snapshot, window: None
    )
    focused = object()
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: focused)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attribute: document if attribute == "AXDocument" else None,
    )


def test_autosaving_document_binds_exact_existing_textedit_file(monkeypatch):
    _install_autosave_document(monkeypatch, "file:///tmp/My%20Notes.txt")
    snapshot = _textedit_snapshot()
    snapshot["app"]["processStartTime"] = 123.0
    snapshot["window_id"] = "cg:55"

    result = backend.inspect_autosaving_document("TextEdit", snapshot)

    assert result == {
        "autosave_identity": ("file:///tmp/My%20Notes.txt", 4, 123.0, "cg:55")
    }


def test_autosaving_document_allows_untitled_textedit_document(monkeypatch):
    _install_autosave_document(monkeypatch, None)

    assert backend.inspect_autosaving_document("TextEdit", _textedit_snapshot()) == {
        "autosave_identity": None
    }


@pytest.mark.parametrize(
    ("bundle_id", "document"),
    [
        ("com.example.Editor", "file:///tmp/notes.txt"),
        ("com.apple.TextEdit", object()),
        ("com.apple.TextEdit", "https://example.com/notes.txt"),
        ("com.apple.TextEdit", "http://["),
        ("com.apple.TextEdit", "file://remote.example/tmp/notes.txt"),
        ("com.apple.TextEdit", "file:relative.txt"),
    ],
)
def test_autosaving_document_rejects_untrusted_identity(
    monkeypatch, bundle_id, document
):
    _install_autosave_document(monkeypatch, document)
    snapshot = _textedit_snapshot()
    snapshot["app"]["bundleId"] = bundle_id

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.inspect_autosaving_document("TextEdit", snapshot)

    assert excinfo.value.code in {"invalid_argument", "target_drift"}


def test_validate_selected_window_focus_runs_both_authority_checks(monkeypatch):
    snapshot = _stable_snapshot()
    calls = []
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda value: calls.append(("snapshot", value)),
    )
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda value: calls.append(("focus", value)),
    )

    backend.validate_selected_window_focus(snapshot)

    assert calls == [("snapshot", snapshot), ("focus", snapshot)]


def _textedit_snapshot():
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    snapshot["app"]["bundleId"] = "com.apple.TextEdit"
    return snapshot


def _ax_text_window(value, *, duplicate=False):
    children = [
        {
            "AXRole": "AXTextArea",
            "AXSubrole": None,
            "AXValue": value,
            "AXChildren": [],
        }
    ]
    if duplicate:
        children.append(dict(children[0]))
    return {"AXRole": "AXWindow", "AXChildren": children}


def test_textedit_plain_text_save_verifies_exact_stable_utf8_match(
    monkeypatch, tmp_path
):
    path = tmp_path / "notes.txt"
    path.write_text("saved line\n", encoding="utf-8")
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    assert backend._verify_textedit_plain_text_save(
        _textedit_snapshot(), path.as_uri(), _ax_text_window("saved line\n")
    )


@pytest.mark.parametrize("failure", ["mismatch", "duplicate", "wrong_bundle", "binary"])
def test_textedit_plain_text_save_rejects_ambiguous_or_inexact_evidence(
    monkeypatch, tmp_path, failure
):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"saved line\n" if failure != "binary" else b"\xff\xfe")
    snapshot = _textedit_snapshot()
    if failure == "wrong_bundle":
        snapshot["app"]["bundleId"] = "com.example.Editor"
    window = _ax_text_window(
        "different" if failure == "mismatch" else "saved line\n",
        duplicate=failure == "duplicate",
    )
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    assert not backend._verify_textedit_plain_text_save(snapshot, path.as_uri(), window)


def test_textedit_plain_text_save_rejects_symlink_non_txt_and_oversize(
    monkeypatch, tmp_path
):
    target = tmp_path / "target.txt"
    target.write_text("same", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    markdown = tmp_path / "notes.md"
    markdown.write_text("same", encoding="utf-8")
    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * (backend.TEXTEDIT_SAVE_MAX_BYTES + 1))
    window = _ax_text_window("same")
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    for path in (link, markdown, large):
        assert not backend._verify_textedit_plain_text_save(
            _textedit_snapshot(), path.as_uri(), window
        )


@pytest.mark.parametrize(
    "value",
    ["x" * (backend.TEXTEDIT_SAVE_MAX_BYTES + 1), "bad-surrogate-\ud800"],
)
def test_textedit_ax_value_rejects_oversize_or_non_utf8(monkeypatch, value):
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    assert (
        backend._unique_textedit_plain_text_value(
            _textedit_snapshot(), _ax_text_window(value)
        )
        is None
    )


def test_textedit_ax_value_bounds_tree_depth(monkeypatch):
    leaf = _ax_text_window("hidden")["AXChildren"][0]
    root = leaf
    for _ in range(backend.TEXTEDIT_VALUE_MAX_DEPTH + 2):
        root = {"AXRole": "AXGroup", "AXChildren": [root]}
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    assert backend._unique_textedit_plain_text_value(_textedit_snapshot(), root) is None


def test_textedit_save_rejects_path_parse_and_io_failures(monkeypatch, tmp_path):
    window = _ax_text_window("same")
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    monkeypatch.setattr(
        backend, "Path", lambda value: (_ for _ in ()).throw(ValueError("bad path"))
    )
    assert not backend._verify_textedit_plain_text_save(
        _textedit_snapshot(), "file:///tmp/notes.txt", window
    )

    monkeypatch.undo()
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    path = tmp_path / "notes.txt"
    path.write_text("same", encoding="utf-8")
    monkeypatch.setattr(
        backend.os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("denied"))
    )
    assert not backend._verify_textedit_plain_text_save(
        _textedit_snapshot(), path.as_uri(), window
    )


def test_textedit_save_rejects_open_file_identity_mismatch(monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("same", encoding="utf-8")
    window = _ax_text_window("same")
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    original_fstat = backend.os.fstat

    def changed_fstat(fd):
        current = original_fstat(fd)
        return types.SimpleNamespace(
            st_dev=current.st_dev,
            st_ino=current.st_ino + 1,
            st_size=current.st_size,
            st_mtime_ns=current.st_mtime_ns,
        )

    monkeypatch.setattr(backend.os, "fstat", changed_fstat)
    assert not backend._verify_textedit_plain_text_save(
        _textedit_snapshot(), path.as_uri(), window
    )


@pytest.mark.parametrize(
    "document",
    [
        "https://example.com/notes.txt",
        "file://remote.example/notes.txt",
        "file://user@localhost/notes.txt",
        "file://localhost:123/notes.txt",
        "file:///tmp/notes.txt?version=1",
        "file:///tmp/notes.txt#fragment",
        "file://[invalid/notes.txt",
    ],
)
def test_textedit_plain_text_save_rejects_nonlocal_or_ambiguous_urls(
    monkeypatch, document
):
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    assert not backend._verify_textedit_plain_text_save(
        _textedit_snapshot(), document, _ax_text_window("same")
    )


def test_textedit_plain_text_save_rejects_path_replaced_by_same_inode_symlink(
    monkeypatch, tmp_path
):
    path = tmp_path / "notes.txt"
    alias = tmp_path / "same-inode.txt"
    path.write_text("same", encoding="utf-8")
    backend.os.link(path, alias)
    window = _ax_text_window("same")
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    original_close = backend.os.close
    replaced = False

    def close_and_replace(fd):
        nonlocal replaced
        original_close(fd)
        path.unlink()
        path.symlink_to(alias)
        replaced = True

    monkeypatch.setattr(backend.os, "close", close_and_replace)
    assert not backend._verify_textedit_plain_text_save(
        _textedit_snapshot(), path.as_uri(), window
    )
    assert replaced


def test_native_save_uses_exact_textedit_disk_match_when_axedited_is_absent(
    monkeypatch,
):
    live = object()
    focused = {}
    monkeypatch.setattr(
        backend,
        "_save_menu_candidate",
        lambda snapshot: (
            live,
            ("File", "Save", "s", "0", ""),
            "file:///tmp/a.txt",
        ),
    )
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: focused)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: None)
    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        lambda *a: backend.ax_driver.kAXErrorSuccess,
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(backend, "_verify_textedit_plain_text_save", lambda *a: True)
    expected = ("file:///tmp/a.txt", "File", "Save", "s", "0", "")
    result = backend.save_document(
        "pid:4",
        _textedit_snapshot(),
        expected_identity=expected,
    )
    assert result["verified"] is True
    assert result["verification_source"] == "textedit_plain_text_exact_disk_match"
    assert result["verification"] == "same-document persistence was verified"


@pytest.mark.parametrize(
    "items",
    [
        [],
        [_save_item(enabled=False)],
        [_save_item(), _save_item("Other Save")],
        [_save_item("Save As", 3)],
    ],
)
def test_native_save_rejects_missing_disabled_ambiguous_or_modified_command(
    monkeypatch, items
):
    pressed = _install_save_menu(monkeypatch, items)
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.inspect_save_document(
            "pid:4", _stable_snapshot(observed_at=backend.time.time())
        )
    assert excinfo.value.code == "element_not_found"
    assert pressed == []


def test_native_save_revalidates_identity_before_press(monkeypatch):
    save = _save_item()
    pressed = _install_save_menu(monkeypatch, [save])
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.save_document("pid:4", snapshot, expected_identity=("different",))
    assert excinfo.value.code == "target_drift"
    assert pressed == []


def test_native_save_post_dispatch_drift_remains_executed_but_unverified(
    monkeypatch,
):
    live = object()
    calls = 0

    def candidate(_snapshot):
        nonlocal calls
        calls += 1
        if calls == 1:
            return live, ("File", "Save", "s", "0", ""), "file:///tmp/a.txt"
        raise errors.ComputerUseError("target_drift", "focus changed after dispatch")

    monkeypatch.setattr(backend, "_save_menu_candidate", candidate)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: {})
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: None)
    pressed = []
    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        lambda element, action: (
            pressed.append((element, action)) or backend.ax_driver.kAXErrorSuccess
        ),
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    expected = ("file:///tmp/a.txt", "File", "Save", "s", "0", "")
    result = backend.save_document("pid:4", snapshot, expected_identity=expected)
    assert pressed == [(live, "AXPress")]
    assert result["executed"] is True
    assert result["verified"] is None


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


def test_read_url_ignores_page_controlled_axvalue_spoof(monkeypatch):
    window = _window()
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app, **kwargs: (
            object(),
            {"name": "browser", "bundleId": "com.google.Chrome", "pid": 4},
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: pytest.fail("domain guard must not inspect AX values"),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda *a: "https://allowed.example/spoofed-by-page",
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: window)
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda cmd, **k: types.SimpleNamespace(
            stdout="https://evil.example/actual-tab", stderr="", returncode=0
        ),
    )
    assert (
        backend.read_url("Browser", window_id=window["window_id"])
        == "https://evil.example/actual-tab"
    )


def test_read_url_uses_trusted_active_tab_and_fails_closed(monkeypatch):
    captured = {}
    monkeypatch.setattr(backend, "_AUTOMATION_READY_BUNDLES", set())
    app_info = {"name": "safari", "bundleId": "com.apple.Safari", "pid": 4}
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)

    def trusted_url(cmd, **kwargs):
        captured["cmd"] = cmd
        return types.SimpleNamespace(
            stdout="https://ok.example", stderr="", returncode=0
        )

    monkeypatch.setattr(backend.subprocess, "run", trusted_url)
    assert backend.read_url("Safari") == "https://ok.example"
    assert captured["cmd"] == [
        "osascript",
        "-e",
        'tell application id "com.apple.Safari" to get URL of current tab of front window',
    ]

    # _resolve_app normalizes bundle IDs to lowercase; supported-browser
    # matching must use the same normalization.
    app_info["bundleId"] = "com.apple.safari"
    assert backend.read_url("Safari") == "https://ok.example"
    assert captured["cmd"] == [
        "osascript",
        "-e",
        'tell application id "com.apple.safari" to get URL of current tab of front window',
    ]

    app_info["bundleId"] = "com.google.chrome"
    assert backend.read_url("Chrome") == "https://ok.example"
    assert captured["cmd"] == [
        "osascript",
        "-e",
        'tell application id "com.google.chrome" to get URL of active tab of front window',
    ]

    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda cmd, **k: types.SimpleNamespace(
            stdout="",
            stderr="Not authorized to send Apple events to Safari. (-1743)",
            returncode=1,
        ),
    )
    assert backend.read_url("Safari") == ""
    with pytest.raises(backend.ComputerUseError) as permission_error:
        backend.read_url("Safari", require_permission=True)
    assert permission_error.value.code == "automation_permission_required"
    assert "System Settings > Privacy & Security > Automation" in str(
        permission_error.value
    )

    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda cmd, **k: types.SimpleNamespace(
            stdout="", stderr="browser has no front window (-1728)", returncode=1
        ),
    )
    assert backend.read_url("Safari", require_permission=True) == ""

    app_info["bundleId"] = "com.example.unsupported"
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: pytest.fail("unsupported browser must fail before osascript"),
    )
    assert backend.read_url("Unsupported") == ""


def test_read_url_resolver_mode_allows_background_app_but_keeps_window_binding(
    monkeypatch,
):
    app_info = {"name": "safari", "bundleId": "com.apple.Safari", "pid": 4}
    window = _window()
    validated = []
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: window)
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda snapshot, **kwargs: validated.append((snapshot, kwargs)),
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            stdout="https://example.com", stderr="", returncode=0
        ),
    )

    assert (
        backend.read_url(
            "Safari", window_id=window["window_id"], allow_background_app=True
        )
        == "https://example.com"
    )
    assert validated == [
        (
            {"app": app_info, "window": window, "window_id": window["window_id"]},
            {"require_active_app": False},
        )
    ]


def test_read_url_waits_for_initial_automation_prompt_then_uses_steady_timeout(
    monkeypatch,
):
    app_info = {"name": "safari", "bundleId": "com.apple.Safari", "pid": 4}
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(backend, "_AUTOMATION_READY_BUNDLES", set())
    timeouts = []

    def trusted_url(cmd, **kwargs):
        timeouts.append(kwargs["timeout"])
        return types.SimpleNamespace(
            stdout="https://example.com", stderr="", returncode=0
        )

    monkeypatch.setattr(backend.subprocess, "run", trusted_url)
    assert backend.read_url("Safari", require_permission=True) == "https://example.com"
    assert backend.read_url("Safari", require_permission=True) == "https://example.com"
    assert timeouts == [
        backend._AUTOMATION_INITIAL_TIMEOUT_S,
        backend._AUTOMATION_STEADY_TIMEOUT_S,
    ]


def test_read_url_timeout_revokes_ready_bundle(monkeypatch):
    app_info = {"name": "safari", "bundleId": "com.apple.Safari", "pid": 4}
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(backend, "_AUTOMATION_READY_BUNDLES", {"com.apple.safari"})
    timeouts = []

    def timeout(cmd, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise backend.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(backend.subprocess, "run", timeout)
    for _ in range(2):
        with pytest.raises(backend.ComputerUseError) as permission_error:
            backend.read_url("Safari", require_permission=True)
        assert permission_error.value.code == "automation_permission_required"
    assert timeouts == [
        backend._AUTOMATION_STEADY_TIMEOUT_S,
        backend._AUTOMATION_INITIAL_TIMEOUT_S,
    ]


def test_read_url_denial_revokes_ready_bundle(monkeypatch):
    app_info = {"name": "safari", "bundleId": "com.apple.Safari", "pid": 4}
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(backend, "_AUTOMATION_READY_BUNDLES", {"com.apple.safari"})
    timeouts = []

    def denied(cmd, **kwargs):
        timeouts.append(kwargs["timeout"])
        return types.SimpleNamespace(
            stdout="",
            stderr="Not authorized to send Apple events to Safari. (-1743)",
            returncode=1,
        )

    monkeypatch.setattr(backend.subprocess, "run", denied)
    for _ in range(2):
        with pytest.raises(backend.ComputerUseError) as permission_error:
            backend.read_url("Safari", require_permission=True)
        assert permission_error.value.code == "automation_permission_required"
    assert timeouts == [
        backend._AUTOMATION_STEADY_TIMEOUT_S,
        backend._AUTOMATION_INITIAL_TIMEOUT_S,
    ]


def test_read_url_surfaces_initial_automation_timeout(monkeypatch):
    app_info = {"name": "safari", "bundleId": "com.apple.Safari", "pid": 4}
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(backend, "_AUTOMATION_READY_BUNDLES", set())
    timeouts = []

    def timeout(cmd, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise backend.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(backend.subprocess, "run", timeout)
    with pytest.raises(backend.ComputerUseError) as permission_error:
        backend.read_url("Safari", require_permission=True)
    assert permission_error.value.code == "automation_permission_required"
    assert "timed out while waiting for macOS Automation authorization" in str(
        permission_error.value
    )

    assert backend.read_url("Safari", require_permission=False) == ""
    assert timeouts == [
        backend._AUTOMATION_INITIAL_TIMEOUT_S,
        backend._AUTOMATION_STEADY_TIMEOUT_S,
    ]


def test_read_url_rejects_active_tab_from_different_selected_window(monkeypatch):
    app_info = {"name": "browser", "bundleId": "com.example.browser", "pid": 4}
    background_window = _window(window_id=202, index=1)
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: background_window)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: pytest.fail(
            "front-tab URL must not authorize a different selected window"
        ),
    )
    assert backend.read_url("Browser", window_id="cg:202") == ""

    app_info["bundleId"] = 'com.bad"\nscript'
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: pytest.fail("invalid bundle id must fail before osascript"),
    )
    assert backend.read_url("Safari") == ""


def test_read_url_pid_target_rejects_duplicate_browser_bundle(monkeypatch):
    app_info = {"name": "browser", "bundleId": "com.google.Chrome", "pid": 42}
    monkeypatch.setattr(
        backend, "_resolve_app", lambda app, **kwargs: (object(), app_info)
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)

    def running(pid):
        return types.SimpleNamespace(
            processIdentifier=lambda: pid,
            bundleIdentifier=lambda: "com.google.Chrome",
            activationPolicy=lambda: 0,
        )

    workspace = types.SimpleNamespace(
        runningApplications=lambda: [running(99), running(42)]
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "AS",
        types.SimpleNamespace(
            NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace)
        ),
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: pytest.fail("ambiguous bundle must fail before osascript"),
    )
    assert backend.read_url("pid:42", window_id="cg:101") == ""


def test_coordinate_click_binds_to_selected_window(monkeypatch):
    snapshot = _stable_snapshot(elements=[])
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(backend.ax_driver, "_cg_click", lambda *args, **kwargs: None)
    result = backend.click("Target App", x=4, y=9, window_id=101)
    assert result["at"] == [4, 9]
    assert result["window_id"] == "cg:101"
    assert result["verified"] is None


def test_synthetic_keyboard_actions_activate_target_app(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXButton",
                "label": "Continue",
                "center": [50, 50],
                "actions": [],
                "source_window_id": "cg:101",
            }
        ]
    )
    calls = []
    monkeypatch.setattr(
        backend,
        "_prepare_synthetic_action",
        lambda app, window_id, *a, **k: calls.append((app, window_id)) or snapshot,
    )
    monkeypatch.setattr(backend.ax_driver, "_type_text", lambda text: None)
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend, "inspect_focused_element", lambda *a, **k: {"index": 0}
    )

    backend.type_text("Target App", "secret")
    backend.press_key(
        "Target App", "return", expected_snapshot=snapshot, element_index=0
    )
    assert calls == [("Target App", None), ("Target App", None)]


def test_direct_press_key_without_planner_target_preserves_cli_action(monkeypatch):
    snapshot = _stable_snapshot()
    dispatched = []
    monkeypatch.setattr(backend, "_prepare_synthetic_action", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend,
        "inspect_focused_element",
        lambda *a, **k: pytest.fail("direct CLI key has no planner target"),
    )
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda key: dispatched.append(key)
    )

    result = backend.press_key("Target App", "return")

    assert result["key"] == "return"
    assert dispatched == [backend.KEY_ALIASES["return"]]


def test_hotkey_and_scroll_activate_target_before_posting(monkeypatch):
    snapshot = _stable_snapshot()
    calls = []
    scroll_events = []
    monkeypatch.setattr(
        backend,
        "_prepare_synthetic_action",
        lambda app, window_id: calls.append(app) or snapshot,
    )
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    fake_quartz = types.SimpleNamespace(
        CGEventCreateKeyboardEvent=lambda *_args: object(),
        CGEventSetFlags=lambda *_args: None,
        CGEventPost=lambda *_args: None,
        CGEventCreateScrollWheelEvent=lambda *_args: (
            scroll_events.append(_args) or object()
        ),
        CGEventSetLocation=lambda *_args: None,
        kCGHIDEventTap=0,
        kCGScrollEventUnitLine=0,
    )
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)

    backend.hotkey("Target App", "Cmd+A")
    backend.scroll("Target App", "down")
    assert calls == ["Target App"]
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
    monkeypatch.setattr(
        backend,
        "_select_window",
        lambda *a, **k: _window(window_id=303, index=2, x=4, y=5),
    )
    monkeypatch.setattr(
        backend,
        "_window_records",
        lambda app: [_window(window_id=303, index=2, x=4, y=5)],
    )
    state = backend.get_app_state(
        "Target App", window_index=2, screenshot=False, use_cache=False
    )
    assert state["window_index"] == 2
    assert state["window_id"] == "cg:303"
    assert captured["window_index"] == 2
    assert captured["window_frame"] == (4.0, 5.0, 100.0, 100.0)
    assert captured["expected_pid"] == 7


def test_get_app_state_merges_only_verified_transient_targets(monkeypatch):
    app_info = {"name": "Finder", "bundleId": "com.apple.finder", "pid": 716}
    anchor = _window(window_id=1647, index=1, x=986, y=538, width=920, height=436)
    popup = _window(window_id=1803, index=0, x=1288, y=926, width=88, height=21)
    monkeypatch.setattr(backend, "_resolve_app", lambda app: (object(), app_info))
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: anchor)
    monkeypatch.setattr(backend, "_window_records", lambda app: [popup, anchor])
    monkeypatch.setattr(backend, "_focused_transient_window", lambda *a, **k: popup)

    def collect(app, **kwargs):
        if kwargs["window"]["window_id"] == "cg:1803":
            return [
                {
                    "target_id": "t000",
                    "role": "AXTextField",
                    "text": "untitled folder",
                    "actions": [],
                    "rect": [1291, 929, 82, 15],
                }
            ]
        return [
            {
                "target_id": "t000",
                "role": "AXButton",
                "text": "Action",
                "actions": ["AXPress"],
                "rect": [1000, 550, 20, 20],
            }
        ]

    monkeypatch.setattr(backend, "_collect_with_timeout", collect)
    state = backend.get_app_state(
        "pid:716",
        screenshot=False,
        use_cache=False,
        window_id="cg:1647",
        transient_baseline_window_ids={"cg:1647"},
    )

    assert state["window_id"] == "cg:1647"
    assert state["transient_window"]["window_id"] == "cg:1803"
    assert [(e["index"], e["source_window_id"]) for e in state["elements"]] == [
        (0, "cg:1647"),
        (1, "cg:1803"),
    ]
    assert state["elements"][1]["label"] == "untitled folder"


def test_get_app_state_bounds_transient_and_total_target_counts(monkeypatch):
    app_info = {"name": "Finder", "bundleId": "com.apple.finder", "pid": 716}
    anchor = _window(window_id=1647, index=1)
    popup = _window(window_id=1803, index=0, x=10, y=10, width=20, height=20)
    monkeypatch.setattr(backend, "_resolve_app", lambda app: (object(), app_info))
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: anchor)
    monkeypatch.setattr(backend, "_window_records", lambda app: [popup, anchor])
    monkeypatch.setattr(backend, "_focused_transient_window", lambda *a, **k: popup)
    monkeypatch.setattr(backend, "MAX_TRANSIENT_TARGETS", 2)
    monkeypatch.setattr(backend.ax_driver, "MAX_NODES", 3)

    def targets(count):
        return [
            {
                "target_id": f"t{index:03d}",
                "role": "AXButton",
                "text": f"item {index}",
                "actions": ["AXPress"],
                "rect": [0, 0, 10, 10],
            }
            for index in range(count)
        ]

    monkeypatch.setattr(
        backend,
        "_collect_with_timeout",
        lambda app, **kwargs: targets(
            4 if kwargs["window"]["window_id"] == "cg:1803" else 2
        ),
    )
    state = backend.get_app_state(
        "pid:716",
        screenshot=False,
        use_cache=False,
        window_id="cg:1647",
        transient_baseline_window_ids={"cg:1647"},
    )

    assert len(state["elements"]) == 3
    assert state["truncated"] is True
    assert [item["source_window_id"] for item in state["elements"]] == [
        "cg:1647",
        "cg:1803",
        "cg:1803",
    ]


def test_finder_file_reference_resolution_fails_closed_on_bridge_error(monkeypatch):
    class BrokenURL:
        @staticmethod
        def URLWithString_(value):
            raise RuntimeError("Foundation bridge unavailable")

    _install_module(monkeypatch, "Foundation", NSURL=BrokenURL)
    assert backend._finder_file_reference_path("file:///.file/id=1") is None


def test_finder_editor_reference_records_exact_row_binding(monkeypatch):
    snapshot = _stable_snapshot()
    live, cell, row, reference = (object() for _ in range(4))
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {"AXParent": cell, "AXURL": reference},
            cell: {"AXParent": row},
        }.get(element, {}).get(attr),
    )
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Original"
    )
    monkeypatch.setattr(backend.time, "monotonic", lambda: 42.0)
    backend._finder_rename_bindings.clear()

    assert backend._finder_file_reference_for_editor(live, snapshot) is reference
    assert backend._finder_rename_bindings[(4, "cg:101")][:3] == (
        row,
        reference,
        "/tmp/Original",
    )


def test_pid_bound_url_read_requires_accessibility_runtime(monkeypatch):
    app_info = {"name": "Browser", "bundleId": "com.example.browser", "pid": 4}
    window = _window()
    monkeypatch.setattr(backend, "_resolve_app", lambda *a, **k: (object(), app_info))
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: window)
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    monkeypatch.setattr(backend.ax_driver, "AS", None)

    assert backend.read_url("pid:4") == ""


def test_select_window_id_is_bound_to_resolved_app_pid(monkeypatch):
    monkeypatch.setattr(
        backend,
        "_window_records",
        lambda app_info: [_window(window_id=303)] if app_info["pid"] == 7 else [],
    )
    assert (
        backend._select_window({"pid": 7}, window_id="cg:303")["window_id"] == "cg:303"
    )
    with pytest.raises(errors.ComputerUseError) as exc:
        backend._select_window({"pid": 8}, window_id="cg:303")
    assert exc.value.code == "window_not_found"
    assert "owned by pid 8" in exc.value.message


def test_validate_window_is_read_only_and_returns_canonical_id(monkeypatch):
    resolved: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app, activate=True: (
            resolved.append((app, activate))
            or (object(), {"name": app, "bundleId": "b", "pid": 7})
        ),
    )
    monkeypatch.setattr(
        backend,
        "_select_window",
        lambda app_info, **kwargs: _window(window_id=303),
    )
    selection = backend.validate_window("Target App", "opaque")
    assert resolved == [("Target App", False)]
    assert selection["window_id"] == "cg:303"
    assert selection["app"]["pid"] == 7


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
    monkeypatch.setattr(backend, "set_value", lambda *a, **k: {"mode": "set"})
    monkeypatch.setattr(backend, "type_text", lambda *a, **k: {"mode": "type"})
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
        backend,
        "_ax_app_element",
        lambda app, **kwargs: ("ax", app.processIdentifier()),
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
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_window_records", lambda app: [_window()])
    monkeypatch.setattr(backend, "screenshot_window", lambda *a, **k: b"png-data")
    monkeypatch.setattr(backend.time, "time", lambda: 10.0)
    state = backend.get_app_state("A", screenshot=True, use_cache=False)
    assert state["element_count"] == 1
    assert state["tree_text"] == "[0] AXButton* Label"
    assert state["screenshot_png"] == b"png-data"
    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: pytest.fail("valid cache should skip AX collection"),
    )
    assert backend.get_app_state("A", screenshot=True) is state

    backend._CACHE.clear()
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app: (object(), {"name": "A", "bundleId": "b", "pid": 4}),
    )
    monkeypatch.setattr(
        backend,
        "_select_window",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError(
                "window_not_found", "window index 2 is not available"
            )
        ),
    )
    monkeypatch.setattr(backend.ax_driver, "collect", lambda *a, **k: [])
    with pytest.raises(errors.ComputerUseError, match="window index"):
        backend.get_app_state("A", window_index=2, screenshot=False, use_cache=False)


def test_get_app_state_cache_revalidates_reorder_and_closed_window(monkeypatch):
    backend._CACHE.clear()
    app_info = {"name": "A", "bundleId": "b", "pid": 4}
    monkeypatch.setattr(backend, "_resolve_app", lambda app: (object(), app_info))
    selected = {"window": _window(window_id=101, index=0)}
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: selected["window"])
    monkeypatch.setattr(backend, "_window_records", lambda app: [selected["window"]])
    monkeypatch.setattr(backend.ax_driver, "collect", lambda *a, **k: [_target()])
    first = backend.get_app_state("A", screenshot=False, use_cache=False)
    assert first["window_id"] == "cg:101"

    selected["window"] = _window(window_id=202, index=0)
    reordered = backend.get_app_state("A", screenshot=False, use_cache=True)
    assert reordered["window_id"] == "cg:202"
    assert reordered["snapshot_id"] != first["snapshot_id"]

    backend._CACHE.put("A", 0, reordered, screenshot=False, window_id=202)
    monkeypatch.setattr(
        backend,
        "_select_window",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError("window_not_found", "window id 202 closed")
        ),
    )
    with pytest.raises(errors.ComputerUseError, match="closed"):
        backend.get_app_state("A", screenshot=False, window_id=202)


def test_screenshot_window_success_and_failures(monkeypatch):
    windows = [
        {"kCGWindowOwnerName": "Other", "kCGWindowLayer": 0, "kCGWindowNumber": 1},
        {"kCGWindowOwnerName": "Target App", "kCGWindowLayer": 2, "kCGWindowNumber": 2},
        {
            "kCGWindowOwnerName": "Target App",
            "kCGWindowOwnerPID": 41,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 3,
        },
        {
            "kCGWindowOwnerName": "Target App",
            "kCGWindowOwnerPID": 42,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 4,
        },
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
    assert (
        len(backend.screenshot_window("Target", window_id="cg:4", expected_pid=42))
        == 9000
    )
    with pytest.raises(errors.ComputerUseError, match="window id cg:3"):
        backend.screenshot_window("Target", window_id="cg:3", expected_pid=42)
    with pytest.raises(errors.ComputerUseError, match="index 2"):
        backend.screenshot_window("Target", 2)
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
    snapshot = _stable_snapshot(elements=[{"index": 2}])
    assert backend._element(snapshot, 2)["index"] == 2
    with pytest.raises(errors.ComputerUseError, match="element 3"):
        backend._element(snapshot, 3)
    monkeypatch.setattr(
        backend.ax_driver, "collect", lambda *a, **k: [_target(2, element="live")]
    )
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    assert backend._live_element(snapshot, 2) == "live"
    with pytest.raises(errors.ComputerUseError, match="not in the current snapshot"):
        backend._live_element(snapshot, 3)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: "value")
    assert backend._read_value("live") == "value"
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: 3)
    assert backend._read_value("live") is None


def test_live_element_rejects_snapshot_index_drift(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 2,
                "role": "AXButton",
                "label": "Safe target",
                "center": [10, 20],
            }
        ]
    )
    drifted = _target(2, role="AXButton", element="wrong-live-element")
    drifted["text"] = "Different target"
    monkeypatch.setattr(backend, "_collect_with_timeout", lambda *a, **k: [drifted])
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    with pytest.raises(errors.ComputerUseError, match="changed since snapshot"):
        backend._live_element(snapshot, 2)


@pytest.mark.parametrize("failure", ["missing_identity", "lost_focus", "synthetic"])
def test_live_transient_element_requires_trusted_exact_action(monkeypatch, failure):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 2,
                "role": "AXButton",
                "label": "Confirm",
                "center": [10, 20],
                "source_window_id": "cg:202",
            }
        ]
    )
    transient = _window(window_id=202, x=5, y=5, width=20, height=20)
    if failure != "missing_identity":
        snapshot["transient_window"] = transient
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(
        backend,
        "_focused_transient_window",
        lambda *a, **k: None if failure == "lost_focus" else transient,
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._live_element(snapshot, 2, validate_point=failure == "synthetic")
    assert excinfo.value.code in {"target_drift", "synthetic_input_blocked"}


def test_window_id_survives_window_reorder(monkeypatch):
    app = {"name": "A", "pid": 4}
    first = [_window(window_id=101, index=0), _window(window_id=202, index=1)]
    reordered = [_window(window_id=202, index=0), _window(window_id=101, index=1)]
    states = iter((first, reordered))
    monkeypatch.setattr(backend, "_window_records", lambda _: next(states))

    assert backend._select_window(app, window_id=202)["index"] == 1
    assert backend._select_window(app, window_id=202)["index"] == 0


def test_window_selector_and_snapshot_validation_fail_closed(monkeypatch):
    app = {"name": "A", "pid": 4}
    windows = [_window(window_id=101, index=0), _window(window_id=202, index=1)]
    monkeypatch.setattr(backend, "_window_records", lambda _: windows)
    assert backend._select_window(app, window_index=1)["window_id"] == "cg:202"
    with pytest.raises(errors.ComputerUseError, match="not an on-screen window"):
        backend._select_window(app, window_id="cg:303")
    for invalid in ("not-a-window", "cg:0"):
        with pytest.raises(errors.ComputerUseError) as excinfo:
            backend._cg_window_id(invalid)
        assert excinfo.value.code == "invalid_argument"

    monkeypatch.setattr(backend.time, "time", lambda: 100.0)
    incomplete = _stable_snapshot(observed_at=100.0)
    incomplete.pop("window")
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_snapshot_window(incomplete)
    assert excinfo.value.code == "stale_observation"

    snapshot = _stable_snapshot(observed_at=100.0)
    moved = {**snapshot["window"], "x": 1}
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: moved)
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_snapshot_window(snapshot)
    assert excinfo.value.code == "target_drift"

    invalid_bounds = {**snapshot["window"], "width": None}
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: invalid_bounds)
    snapshot["window"] = invalid_bounds
    with pytest.raises(errors.ComputerUseError, match="invalid bounds"):
        backend._validate_snapshot_window(snapshot, point=(5, 5))

    snapshot = _stable_snapshot(observed_at=100.0)
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: snapshot["window"])
    monkeypatch.setattr(backend, "_topmost_window_id_at", lambda *a: 999)
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_snapshot_window(snapshot, point=(5, 5))
    assert excinfo.value.code == "target_occluded"
    monkeypatch.setattr(backend, "_topmost_window_id_at", lambda *a: 101)
    assert (
        backend._validate_snapshot_window(snapshot, point=(5, 5)) == snapshot["window"]
    )


def test_window_records_and_topmost_probe_skip_invalid_entries(monkeypatch):
    raw = [
        {"kCGWindowOwnerPID": 4, "kCGWindowLayer": 0},
        {
            "kCGWindowOwnerPID": 4,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 101,
            "kCGWindowAlpha": 0,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 100, "Height": 100},
        },
        {
            "kCGWindowOwnerPID": 4,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 202,
            "kCGWindowBounds": {"X": None, "Y": 0, "Width": 100, "Height": 100},
        },
        {
            "kCGWindowOwnerPID": 4,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 303,
            "kCGWindowBounds": {"X": 200, "Y": 200, "Width": 50, "Height": 50},
        },
        {
            "kCGWindowOwnerPID": 4,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 404,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 100, "Height": 100},
        },
    ]
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *_: raw,
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )
    records = backend._window_records({"pid": 4})
    assert [record["window_id"] for record in records] == [
        "cg:101",
        "cg:202",
        "cg:303",
        "cg:404",
    ]
    assert backend._topmost_window_id_at(5, 5) == 404
    assert backend._topmost_window_id_at(150, 150) is None


def test_topmost_probe_ignores_system_layers_but_keeps_normal_window_occlusion(
    monkeypatch,
):
    raw = [
        {
            "kCGWindowOwnerName": "Notification Center",
            "kCGWindowLayer": 21,
            "kCGWindowNumber": 900,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1920, "Height": 1080},
        },
        {
            "kCGWindowOwnerName": "Blocking Panel",
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 202,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 100, "Height": 100},
        },
        {
            "kCGWindowOwnerName": "Target App",
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 101,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 500, "Height": 500},
        },
    ]
    _install_module(
        monkeypatch,
        "Quartz",
        CGWindowListCopyWindowInfo=lambda *_: raw,
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )

    assert backend._topmost_window_id_at(50, 50) == 202
    assert backend._topmost_window_id_at(250, 250) == 101


def test_stale_snapshot_rejected_before_input(monkeypatch):
    snapshot = _stable_snapshot(observed_at=10.0)
    monkeypatch.setattr(
        backend.time,
        "time",
        lambda: 10.0 + backend.SNAPSHOT_TTL_S + 0.1,
    )
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_snapshot_window(snapshot)
    assert excinfo.value.code == "stale_observation"


def test_coordinate_action_rejects_point_outside_selected_window(monkeypatch):
    snapshot = _stable_snapshot(observed_at=100.0)
    monkeypatch.setattr(backend.time, "time", lambda: 100.0)
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: snapshot["window"])
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_snapshot_window(snapshot, point=(101, 50))
    assert excinfo.value.code == "target_drift"


def test_synthetic_input_reports_attempted_but_unverified(monkeypatch):
    snapshot = _stable_snapshot()
    monkeypatch.setattr(backend, "_prepare_synthetic_action", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend.ax_driver, "_type_text", lambda text: None)
    result = backend.type_text("A", "hello", window_id=101)
    assert result["attempted"] is True
    assert result["verified"] is None
    assert "synthetic text emitted" in result["verification"]


def test_raise_selected_window_uses_unique_pid_bound_ax_window(monkeypatch):
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    selected_ax = object()
    other_ax = object()
    app_element = object()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda value: value["window"]
    )
    resolutions = []
    foreground = {"finder": False}  # Rapid approval sheet is frontmost.

    def resolve(app, *, activate=True):
        resolutions.append((app, activate))
        if activate:
            foreground["finder"] = True
        return app_element, dict(snapshot["app"])

    monkeypatch.setattr(backend, "_resolve_app", resolve)
    monkeypatch.setattr(
        backend, "_select_window", lambda *a, **k: dict(snapshot["window"])
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: (
            [other_ax, selected_ax]
            if element is app_element and attr == "AXWindows"
            else None
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_point_size",
        lambda element: (
            (50.0, 50.0, 20.0, 20.0)
            if element is other_ax
            else (0.0, 0.0, 100.0, 100.0)
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_action_names",
        lambda element: ["AXRaise"] if element is selected_ax else [],
    )
    actions = []
    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        lambda element, action: (
            actions.append((element, action)) or backend.ax_driver.kAXErrorSuccess
        ),
    )
    focused = []
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda value: (
            focused.append(value)
            if foreground["finder"]
            else pytest.fail("approved action resumed before Finder was frontmost")
        ),
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)

    assert backend.raise_selected_window("pid:4", snapshot) == snapshot["window"]
    assert actions == [(selected_ax, "AXRaise")]
    assert focused == [snapshot]
    assert resolutions == [("pid:4", False), ("pid:4", True)]


def test_raise_selected_window_rejects_ambiguous_ax_match(monkeypatch):
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    app_element = object()
    windows = [object(), object()]
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda value: value["window"]
    )
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda app, *, activate=True: (app_element, dict(snapshot["app"])),
    )
    monkeypatch.setattr(
        backend, "_select_window", lambda *a, **k: dict(snapshot["window"])
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: (
            windows if element is app_element and attr == "AXWindows" else None
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda _: (0.0, 0.0, 100.0, 100.0)
    )
    monkeypatch.setattr(backend.ax_driver, "_action_names", lambda _: ["AXRaise"])
    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        lambda *a: pytest.fail("ambiguous window must not be raised"),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.raise_selected_window("pid:4", snapshot)
    assert excinfo.value.code == "target_occluded"


def test_raise_selected_window_requires_snapshot_identity():
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.raise_selected_window("Finder", {"window": {}})
    assert excinfo.value.code == "stale_observation"


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("app_drift", "target_drift"),
        ("activated_app_drift", "target_drift"),
        ("process_start_drift", "target_drift"),
        ("activated_process_start_drift", "target_drift"),
        ("window_drift", "target_drift"),
        ("post_activate_ax_drift", "target_occluded"),
        ("raise_rejected", "target_occluded"),
        ("post_raise_drift", "target_drift"),
    ],
)
def test_raise_selected_window_fails_closed_across_identity_boundaries(
    monkeypatch, failure, code
):
    snapshot = _stable_snapshot(observed_at=backend.time.time())
    snapshot["app"]["processStartTime"] = 100.0
    selected_ax = object()
    app_element = object()
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda value: value["window"]
    )
    activated = {"value": False}
    resolutions = []

    def resolve(app, *, activate=True):
        resolutions.append(activate)
        activated["value"] = activate
        app_info = dict(snapshot["app"])
        if failure == "app_drift" or (failure == "activated_app_drift" and activate):
            app_info["name"] = "Different"
        if failure == "process_start_drift" or (
            failure == "activated_process_start_drift" and activate
        ):
            app_info["processStartTime"] = 200.0
        return app_element, app_info

    monkeypatch.setattr(backend, "_resolve_app", resolve)
    selections = 0

    def select(*args, **kwargs):
        nonlocal selections
        selections += 1
        current = dict(snapshot["window"])
        if failure == "window_drift" or (
            failure == "post_raise_drift" and selections > 2
        ):
            current["x"] += 20
        return current

    monkeypatch.setattr(backend, "_select_window", select)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: (
            [selected_ax, object()]
            if failure == "post_activate_ax_drift"
            and activated["value"]
            and element is app_element
            and attr == "AXWindows"
            else [selected_ax]
            if element is app_element and attr == "AXWindows"
            else None
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda _: (0.0, 0.0, 100.0, 100.0)
    )
    monkeypatch.setattr(backend.ax_driver, "_action_names", lambda _: ["AXRaise"])
    actions = []

    def perform_action(*args):
        actions.append(args)
        return 1 if failure == "raise_rejected" else backend.ax_driver.kAXErrorSuccess

    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        perform_action,
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(backend, "_validate_focused_window", lambda value: None)

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.raise_selected_window("pid:4", snapshot)
    assert excinfo.value.code == code
    if failure == "process_start_drift":
        assert resolutions == [False]
        assert actions == []
    elif failure == "activated_process_start_drift":
        assert resolutions == [False, True]
        assert actions == []


def test_synthetic_input_rejects_background_process(monkeypatch):
    snapshot = _stable_snapshot()
    background = types.SimpleNamespace(processIdentifier=lambda: 99)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: background)
    monkeypatch.setattr(
        backend.ax_driver,
        "AS",
        types.SimpleNamespace(
            NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace)
        ),
    )
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_focused_window(snapshot)
    assert excinfo.value.code == "target_drift"


def test_focused_window_uses_fresh_running_application_activity(monkeypatch):
    snapshot = _stable_snapshot()
    stale_frontmost = types.SimpleNamespace(processIdentifier=lambda: 99)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: stale_frontmost)
    fresh = types.SimpleNamespace(isActive=lambda: True)
    services = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: fresh
        ),
    )
    monkeypatch.setattr(backend.ax_driver, "AS", services)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: "focused")
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda *a: (0.0, 0.0, 100.0, 100.0)
    )

    backend._validate_focused_window(snapshot)


def test_focused_window_rejects_inactive_fresh_running_application(monkeypatch):
    snapshot = _stable_snapshot()
    stale_frontmost = types.SimpleNamespace(processIdentifier=lambda: 4)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: stale_frontmost)
    fresh = types.SimpleNamespace(isActive=lambda: False)
    services = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: fresh
        ),
    )
    monkeypatch.setattr(backend.ax_driver, "AS", services)

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_focused_window(snapshot)
    assert excinfo.value.code == "target_drift"


def test_background_url_validation_still_requires_exact_internal_focused_window(
    monkeypatch,
):
    snapshot = _stable_snapshot()
    fresh = types.SimpleNamespace(isActive=lambda: False)
    services = types.SimpleNamespace(
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: fresh
        )
    )
    monkeypatch.setattr(backend.ax_driver, "AS", services)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: "focused")
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda *a: (0.0, 0.0, 100.0, 100.0)
    )

    backend._validate_focused_window(snapshot, require_active_app=False)

    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda *a: (10.0, 0.0, 100.0, 100.0)
    )
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_focused_window(snapshot, require_active_app=False)
    assert excinfo.value.code == "target_drift"


@pytest.mark.parametrize("resolver_result", [None, RuntimeError("bridge failed")])
def test_focused_window_fails_closed_when_fresh_resolution_fails(
    monkeypatch, resolver_result
):
    snapshot = _stable_snapshot()
    stale_frontmost = types.SimpleNamespace(processIdentifier=lambda: 4)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: stale_frontmost)

    def resolve(pid):
        if isinstance(resolver_result, Exception):
            raise resolver_result
        return resolver_result

    services = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=resolve
        ),
    )
    monkeypatch.setattr(backend.ax_driver, "AS", services)

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._validate_focused_window(snapshot)
    assert excinfo.value.code == "target_drift"


def test_focused_window_and_post_action_state_contract(monkeypatch):
    snapshot = _stable_snapshot()
    frontmost = types.SimpleNamespace(processIdentifier=lambda: 4)
    workspace = types.SimpleNamespace(frontmostApplication=lambda: frontmost)
    monkeypatch.setattr(
        backend.ax_driver,
        "AS",
        types.SimpleNamespace(
            NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace)
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver, "_app_element", lambda app, **kwargs: "application"
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attribute: (
            ["focused"] if attribute == "AXWindows" else "focused"
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda *a: (0.0, 0.0, 100.0, 100.0)
    )
    backend._validate_focused_window(snapshot)
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda *a: (1.0, 0.0, 100.0, 100.0)
    )
    with pytest.raises(errors.ComputerUseError, match="not the focused AX window"):
        backend._validate_focused_window(snapshot)

    fresh = {"snapshot_id": "fresh"}
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: fresh)
    result = backend._finish_action(
        "A",
        snapshot,
        {"mode": "test"},
        verified=None,
        verification="test",
        include_post_state=True,
    )
    assert result["post_action_state"] is fresh

    def failed_state(*args, **kwargs):
        raise errors.ComputerUseError("window_not_found", "closed")

    monkeypatch.setattr(backend, "get_app_state", failed_state)
    result = backend._finish_action(
        "A",
        snapshot,
        {"mode": "test"},
        verified=None,
        verification="test",
        include_post_state=True,
    )
    assert result["post_action_state"] is None
    assert result["post_action_state_error"]["code"] == "window_not_found"


def test_prepare_synthetic_action_observes_and_validates(monkeypatch):
    snapshot = _stable_snapshot()
    calls = []
    monkeypatch.setattr(
        backend,
        "get_app_state",
        lambda *a, **k: calls.append((a, k)) or snapshot,
    )
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda *a, **k: calls.append("window") or snapshot["window"],
    )
    monkeypatch.setattr(
        backend, "_validate_focused_window", lambda *a: calls.append("focus")
    )
    assert backend._prepare_synthetic_action("A", "cg:101") is snapshot
    assert calls[1:] == ["window", "focus"]


def test_enter_allows_same_pid_overlay_for_exact_focused_anchor_editable(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "Address and search",
                "center": [50, 20],
                "actions": [],
                "source_window_id": "cg:101",
            }
        ]
    )
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    validations = []
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snap, **kwargs: validations.append(kwargs) or snap["window"],
    )
    monkeypatch.setattr(
        backend,
        "_window_records",
        lambda app: [snapshot["window"], {"window_id": "cg:202"}],
    )
    topmost_points = []
    monkeypatch.setattr(
        backend,
        "_topmost_window_id_at",
        lambda *point: topmost_points.append(point) or 202,
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    pressed = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda key: pressed.append(key)
    )

    result = backend.press_key(
        "pid:4", "Enter", expected_snapshot=snapshot, element_index=0
    )

    assert result["mode"] == "CGEvent-keycode"
    assert validations == [{}]
    assert topmost_points == [(50.0, 20.0)]
    assert pressed == [backend.KEY_ALIASES["enter"]]


def test_enter_rejects_foreign_overlay_despite_exact_focused_editable(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "Address and search",
                "center": [50, 20],
                "actions": [],
                "source_window_id": "cg:101",
            }
        ]
    )
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snap, **kwargs: snap["window"]
    )
    monkeypatch.setattr(backend, "_window_records", lambda app: [snapshot["window"]])
    window_center = backend._window_center(snapshot)
    monkeypatch.setattr(
        backend,
        "_topmost_window_id_at",
        lambda x, y: 303 if (x, y) == (50.0, 20.0) else 101,
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_press_key",
        lambda *a: pytest.fail("foreign overlay must block keyboard dispatch"),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("pid:4", "Enter", expected_snapshot=snapshot, element_index=0)

    assert excinfo.value.code == "target_occluded"
    assert window_center != (50.0, 20.0)


def test_collect_watchdog_preserves_structured_errors(monkeypatch):
    expected = errors.ComputerUseError("element_not_found", "window vanished")

    def failed_collect(*args, **kwargs):
        raise expected

    monkeypatch.setattr(backend.ax_driver, "collect", failed_collect)
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._collect_with_timeout("A", timeout_s=1)
    assert excinfo.value is expected


def test_collect_watchdog_translates_missing_pid_to_typed_error(monkeypatch):
    monkeypatch.setattr(
        backend.ax_driver,
        "collect",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("pid missing")),
    )
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._collect_with_timeout("A", expected_pid=42, timeout_s=1)
    assert excinfo.value.code == "app_not_found"
    assert "pid missing" in excinfo.value.message


def test_collect_watchdog_returns_completed_priority_prefix_when_tail_blocks(
    monkeypatch,
):
    release = threading.Event()
    priority = _target(element="cpu-tab", role="AXRadioButton")
    priority["text"] = "CPU"
    priority.pop("center")

    def slow_tail(*args, partial_out, **kwargs):
        partial_out.append(priority)
        release.wait(1)
        return partial_out

    monkeypatch.setattr(backend.ax_driver, "collect", slow_tail)
    status = {}
    try:
        collected = backend._collect_with_timeout(
            "Activity Monitor", timeout_s=0.01, collection_status=status
        )
    finally:
        release.set()

    assert len(collected) == 1
    assert collected[0]["text"] == "CPU"
    assert collected[0] is not priority
    assert collected[0]["element"] == "cpu-tab"
    assert collected[0]["center"] == [6, 12]
    assert status == {"partial": True}


@pytest.mark.parametrize(
    ("bundle", "expected"),
    [
        ("com.google.Chrome", True),
        ("org.chromium.Chromium", True),
        ("com.microsoft.edgemac", True),
        ("com.apple.ActivityMonitor", False),
        ("com.apple.TextEdit", False),
        ("com.apple.Safari", False),
    ],
)
def test_web_content_retry_is_explicitly_limited_to_chromium(bundle, expected):
    assert backend._needs_web_content_retry({"bundleId": bundle}) is expected


def test_element_click_ax_and_fallback(monkeypatch):
    snapshot = {"elements": [{"index": 0, "center": [6, 12], "actions": ["AXPress"]}]}
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
    monkeypatch.setattr(backend, "_validate_snapshot_window", lambda *a, **k: {})
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
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError("element_not_found", "gone")
        ),
    )
    with pytest.raises(errors.ComputerUseError, match="gone"):
        backend.click("A", element_index=0)
    assert len(clicks) == 1


def test_click_recollects_ax_element_from_snapshot_pid(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXButton",
                "label": "Safe",
                "center": [6, 12],
                "actions": ["AXPress"],
            }
        ]
    )
    captured: dict = {}

    def collect(app_name, **kwargs):
        captured.update({"app_name": app_name, **kwargs})
        target = _target(0, role="AXButton", element="pid-4-element")
        target.update({"text": "Safe", "center": [6, 12]})
        return [target]

    monkeypatch.setattr(backend, "_collect_with_timeout", collect)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    pressed: list[object] = []
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda element, action: (
            pressed.append((element, action)) or 0
        ),
    )
    result = backend.click("pid:4", element_index=0, expected_snapshot=snapshot)
    assert result["mode"] == "AXPress"
    assert captured["app_name"] == "A"
    assert captured["expected_pid"] == 4
    assert pressed == [("pid-4-element", "AXPress")]


def test_click_allows_exact_axpress_menu_outside_selected_window(monkeypatch):
    target = {
        "index": 0,
        "role": "AXMenuItem",
        "label": "New Folder",
        "center": [1928, 602],
        "actions": ["AXPress"],
    }
    snapshot = _stable_snapshot(elements=[target])
    fresh = _target(0, role="AXMenuItem", element="menu-item")
    fresh.update({"text": "New Folder", "center": [1928, 602]})
    monkeypatch.setattr(backend.time, "time", lambda: 100.0)
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: snapshot["window"])
    monkeypatch.setattr(backend, "_collect_with_timeout", lambda *a, **k: [fresh])
    monkeypatch.setattr(
        backend.ax_driver,
        "_cg_click",
        lambda *a, **k: pytest.fail("exact AXPress must not synthesize a click"),
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda element, action: 0,
    )

    result = backend.click("pid:4", element_index=0, expected_snapshot=snapshot)

    assert result["mode"] == "AXPress"
    assert result["element_index"] == 0


def test_click_keeps_outside_window_coordinate_fallback_fail_closed(monkeypatch):
    target = {
        "index": 0,
        "role": "AXMenuItem",
        "label": "New Folder",
        "center": [1928, 602],
        "actions": ["AXPress"],
    }
    snapshot = _stable_snapshot(elements=[target])
    fresh = _target(0, role="AXMenuItem", element="menu-item")
    fresh.update({"text": "New Folder", "center": [1928, 602]})
    monkeypatch.setattr(backend.time, "time", lambda: 100.0)
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: snapshot["window"])
    monkeypatch.setattr(backend, "_collect_with_timeout", lambda *a, **k: [fresh])
    monkeypatch.setattr(
        backend.ax_driver,
        "_cg_click",
        lambda *a, **k: pytest.fail("outside-window fallback must not click"),
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        kAXErrorSuccess=0,
        AXUIElementPerformAction=lambda element, action: 1,
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.click("pid:4", element_index=0, expected_snapshot=snapshot)

    assert excinfo.value.code == "target_drift"
    assert "outside selected window" in excinfo.value.message


def test_finder_inline_editor_is_admitted_as_new_contained_focused_companion(
    monkeypatch,
):
    anchor = {
        "index": 1,
        "window_id": "cg:1647",
        "x": 986,
        "y": 538,
        "width": 920,
        "height": 436,
    }
    popup = {
        "index": 0,
        "window_id": "cg:1803",
        "x": 1288,
        "y": 926,
        "width": 88,
        "height": 21,
    }
    focused = object()
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: focused)
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda element: (1291, 929, 82, 15)
    )
    monkeypatch.setattr(backend, "_window_records", lambda app: [popup, anchor])

    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 716},
            anchor,
            baseline_window_ids={"cg:1647"},
        )
        == popup
    )
    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 716},
            anchor,
            baseline_window_ids={"cg:1803", "cg:1647"},
        )
        is None
    )


def test_focused_ax_window_accepts_exact_focused_ui_element_listed_as_window(
    monkeypatch,
):
    application = object()
    inline_editor = object()
    monkeypatch.setattr(backend.ax_driver, "_app_element", lambda *a, **k: application)

    def get(element, attribute):
        if attribute == "AXFocusedWindow":
            return None
        if attribute == "AXFocusedUIElement":
            return inline_editor
        if attribute == "AXWindows":
            return [inline_editor, object()]
        return None

    monkeypatch.setattr(backend.ax_driver, "_get", get)

    assert backend._focused_ax_window({"name": "Finder", "pid": 716}) is inline_editor


def test_focused_ax_window_walks_from_focused_control_to_listed_window(monkeypatch):
    application = object()
    window = object()
    parent = object()
    control = object()
    monkeypatch.setattr(backend.ax_driver, "_app_element", lambda *a, **k: application)

    def get(element, attribute):
        if element is application:
            return {
                "AXWindows": [window],
                "AXFocusedWindow": None,
                "AXFocusedUIElement": control,
            }.get(attribute)
        if attribute == "AXRole":
            return "AXWindow" if element is window else "AXGroup"
        if attribute == "AXParent":
            return {control: parent, parent: window}.get(element)
        return None

    monkeypatch.setattr(backend.ax_driver, "_get", get)

    assert backend._focused_ax_window({"name": "Finder", "pid": 716}) is window


@pytest.mark.parametrize("parent_mode", ["cycle", "too_deep"])
def test_focused_ax_window_rejects_unbounded_parent_chains(monkeypatch, parent_mode):
    application = object()
    control = object()
    window = object()
    chain = [object() for _ in range(13)]
    monkeypatch.setattr(backend.ax_driver, "_app_element", lambda *a, **k: application)

    def get(element, attribute):
        if element is application:
            return {
                "AXWindows": [window],
                "AXFocusedWindow": None,
                "AXFocusedUIElement": control,
            }.get(attribute)
        if attribute == "AXRole":
            return "AXGroup"
        if attribute == "AXParent":
            if parent_mode == "cycle":
                return control
            if element is control:
                return chain[0]
            index = chain.index(element)
            return chain[index + 1] if index + 1 < len(chain) else None
        return None

    monkeypatch.setattr(backend.ax_driver, "_get", get)

    assert backend._focused_ax_window({"name": "Finder", "pid": 716}) is None


def test_focused_ax_element_is_pid_bound(monkeypatch):
    application = object()
    control = object()
    calls = []
    monkeypatch.setattr(
        backend.ax_driver,
        "_app_element",
        lambda name, expected_pid: calls.append((name, expected_pid)) or application,
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attribute: (
            control if attribute == "AXFocusedUIElement" else None
        ),
    )

    assert backend._focused_ax_element({"name": "Finder", "pid": "716"}) is control
    assert calls == [("Finder", 716)]


def test_transient_companion_rejects_missing_frame_or_anchor_sized_window(monkeypatch):
    anchor = {
        "index": 1,
        "window_id": "cg:1",
        "x": 100,
        "y": 100,
        "width": 500,
        "height": 400,
    }
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: object())
    monkeypatch.setattr(backend.ax_driver, "_point_size", lambda element: None)
    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 4}, anchor, baseline_window_ids={"cg:1"}
        )
        is None
    )

    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda element: (100, 100, 500, 400)
    )
    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 4}, anchor, baseline_window_ids={"cg:1"}
        )
        is None
    )


def test_transient_companion_requires_unique_record_and_trusted_identity(monkeypatch):
    anchor = {
        "index": 2,
        "window_id": "cg:1",
        "x": 100,
        "y": 100,
        "width": 500,
        "height": 400,
    }
    candidate = {
        "index": 0,
        "window_id": "cg:2",
        "x": 120,
        "y": 120,
        "width": 100,
        "height": 40,
    }
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: object())
    monkeypatch.setattr(
        backend.ax_driver, "_point_size", lambda element: (120, 120, 100, 40)
    )
    monkeypatch.setattr(
        backend, "_window_records", lambda app: [candidate, dict(candidate)]
    )
    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 4}, anchor, trusted_window_id="cg:2"
        )
        is None
    )

    monkeypatch.setattr(backend, "_window_records", lambda app: [candidate])
    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 4}, anchor, trusted_window_id="cg:99"
        )
        is None
    )


@pytest.mark.parametrize(
    "popup",
    [
        {
            "index": 0,
            "window_id": "cg:2",
            "x": 610,
            "y": 210,
            "width": 88,
            "height": 21,
        },
        {
            "index": 2,
            "window_id": "cg:2",
            "x": 110,
            "y": 110,
            "width": 88,
            "height": 21,
        },
        {
            "index": 0,
            "window_id": "cg:2",
            "x": 100,
            "y": 100,
            "width": 600,
            "height": 400,
        },
    ],
)
def test_transient_companion_rejects_outside_behind_or_oversized(monkeypatch, popup):
    anchor = {
        "index": 1,
        "window_id": "cg:1",
        "x": 100,
        "y": 100,
        "width": 500,
        "height": 400,
    }
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: object())
    monkeypatch.setattr(
        backend.ax_driver,
        "_point_size",
        lambda element: (
            popup["x"],
            popup["y"],
            popup["width"],
            popup["height"],
        ),
    )
    monkeypatch.setattr(backend, "_window_records", lambda app: [popup, anchor])

    assert (
        backend._focused_transient_window(
            {"name": "Finder", "pid": 4},
            anchor,
            baseline_window_ids={"cg:1"},
        )
        is None
    )


def test_transient_exact_set_value_never_falls_back_to_synthetic(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "untitled folder",
                "center": [1332, 936],
                "actions": [],
                "source_window_id": "cg:1803",
            }
        ]
    )
    snapshot["transient_window"] = {
        "index": 0,
        "window_id": "cg:1803",
        "x": 1288,
        "y": 926,
        "width": 88,
        "height": 21,
    }
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend,
        "_synthetic_fill",
        lambda *a, **k: pytest.fail("transient target must not synthesize typing"),
    )
    module = _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *a: 1,
        kAXValueAttribute="AXValue",
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.set_value("pid:4", 0, "named folder", expected_snapshot=snapshot)

    assert module is not None
    assert excinfo.value.code == "synthetic_input_blocked"


def test_transient_exact_set_value_succeeds_with_readback(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "untitled folder",
                "center": [1332, 936],
                "actions": [],
                "source_window_id": "cg:1803",
            }
        ]
    )
    snapshot["transient_window"] = {
        "index": 0,
        "window_id": "cg:1803",
        "x": 1288,
        "y": 926,
        "width": 88,
        "height": 21,
    }
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_read_value", lambda element: "named folder")
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *a: 0,
        kAXValueAttribute="AXValue",
    )

    result = backend.set_value("pid:4", 0, "named folder", expected_snapshot=snapshot)

    assert result["mode"] == "AXSetValue"
    assert result["verified"] is True


class _FinderFileReference:
    def __init__(self, current_path):
        self.current_path = current_path

    def filePathURL(self):
        return self

    def path(self):
        return self.current_path()


def _finder_rename_snapshot():
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "parent_role": "AXCell",
                "label": "Verified CUA Folder",
                "center": [50, 50],
                "actions": [],
                "source_window_id": "cg:101",
            }
        ]
    )
    snapshot["app"] = {"name": "Finder", "bundleId": "com.apple.finder", "pid": 4}
    return snapshot


def test_finder_identity_requires_os_bundle_metadata():
    assert backend.is_finder_snapshot(
        {"app": {"name": "pid:93136", "bundleId": "com.apple.finder"}}
    )
    assert not backend.is_finder_snapshot(
        {"app": {"name": "Finder", "bundleId": "com.example.lookalike"}}
    )


def test_finder_set_value_reports_exact_disk_persistence(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, reference = object(), object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            "AXValue": "Verified CUA Folder",
            "AXURL": reference,
        }.get(attr),
    )
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Verified CUA Folder",
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *a: 0,
        kAXValueAttribute="AXValue",
    )

    result = backend.set_value(
        "pid:4", 0, "Verified CUA Folder", expected_snapshot=snapshot
    )

    assert result["verified"] is True
    assert result["verification_source"] == "finder_file_reference_basename"
    assert result["actual_basename"] == "Verified CUA Folder"


def _install_finder_rename_tree(
    monkeypatch, live, reference, value, selected_state=None
):
    selected_state = selected_state or [True]
    cell, row, menu_bar, rename_item, app_element = (object() for _ in range(5))

    def get(element, attr):
        values = {
            live: {
                "AXURL": reference,
                "AXValue": value,
                "AXParent": cell,
                "AXFocused": True,
            },
            cell: {"AXRole": "AXCell", "AXParent": row},
            row: {"AXRole": "AXRow", "AXSelected": selected_state[0]},
            app_element: {"AXMenuBar": menu_bar},
            menu_bar: {"AXChildren": [rename_item]},
            rename_item: {
                "AXRole": "AXMenuItem",
                "AXTitle": "Rename",
                "AXEnabled": True,
                "AXChildren": [],
            },
        }
        return values.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(backend.ax_driver, "_app_element", lambda *a, **k: app_element)
    monkeypatch.setattr(
        backend.ax_driver,
        "_action_names",
        lambda element: ["AXPress"] if element is rename_item else [],
    )
    return rename_item


def _install_selected_finder_editor(monkeypatch, snapshot, path_state):
    live, cell, row, outline, reference = (object() for _ in range(5))

    def get(element, attr):
        values = {
            live: {
                "AXRole": "AXTextField",
                "AXParent": cell,
                "AXSelected": True,
                "AXURL": reference,
                "AXValue": "After",
            },
            cell: {"AXRole": "AXCell", "AXParent": row, "AXSelected": True},
            row: {
                "AXRole": "AXRow",
                "AXParent": outline,
                "AXSelected": True,
            },
            outline: {"AXRole": "AXOutline"},
        }
        return values.get(element, {}).get(attr)

    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "raise_selected_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: outline)
    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: path_state[0] if value is reference else None,
    )
    return live, reference


def test_finder_rename_transaction_binds_exact_selected_item(monkeypatch):
    snapshot = _finder_rename_snapshot()
    snapshot["elements"][0]["parent_role"] = ""
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )

    binding = backend.inspect_finder_rename(snapshot, 0, "After")

    assert binding["file_reference"] is reference
    assert binding["original_path"] == "/tmp/Before"
    assert binding["requested_basename"] == "After"


def test_finder_rename_transaction_binds_detached_focused_editor(monkeypatch):
    snapshot = _finder_rename_snapshot()
    snapshot["elements"][0]["parent_role"] = ""
    snapshot["elements"][0]["label"] = "Before"
    live, indexed, app_root, bound_row, outline, reference, other_reference = (
        object() for _ in range(7)
    )
    key = backend._finder_rename_binding_key(snapshot)
    backend._finder_rename_bindings[key] = (
        bound_row,
        reference,
        "/tmp/Before",
        backend.time.monotonic(),
    )
    selected = {"value": True}

    def get(element, attr):
        return {
            live: {
                "AXRole": "AXTextField",
                "AXParent": app_root,
                "AXURL": None,
                "AXValue": "Before\u200b\u200b",
            },
            indexed: {
                "AXRole": "AXTextField",
                "AXURL": other_reference,
                "AXParent": object(),
            },
            app_root: {"AXRole": "AXApplication"},
            bound_row: {
                "AXRole": "AXRow",
                "AXParent": outline,
                "AXSelected": selected["value"],
            },
            outline: {"AXRole": "AXOutline", "AXChildren": [bound_row]},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: indexed)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Before" if value is reference else "/tmp/Other",
    )

    binding = backend.inspect_finder_rename(snapshot, 0, "After")
    assert binding["file_reference"] is reference

    selected["value"] = False
    with pytest.raises(errors.ComputerUseError, match="selected item"):
        backend.inspect_finder_rename(snapshot, 0, "After")


def test_finder_detached_editor_rejects_wrong_item_and_multiselection(monkeypatch):
    snapshot = _finder_rename_snapshot()
    snapshot["elements"][0].update({"parent_role": "", "label": "Before"})
    live, app_root, bound_row, other_row, outline, reference = (
        object() for _ in range(6)
    )
    key = backend._finder_rename_binding_key(snapshot)
    backend._finder_rename_bindings[key] = (
        bound_row,
        reference,
        "/tmp/Before",
        backend.time.monotonic(),
    )
    editor_value = {"value": "Other"}
    selected_rows = {"value": [bound_row]}

    def get(element, attr):
        return {
            live: {
                "AXRole": "AXTextField",
                "AXParent": app_root,
                "AXURL": None,
                "AXValue": editor_value["value"],
            },
            app_root: {"AXRole": "AXApplication"},
            bound_row: {"AXRole": "AXRow", "AXParent": outline, "AXSelected": True},
            other_row: {"AXRole": "AXRow", "AXParent": outline, "AXSelected": True},
            outline: {"AXRole": "AXOutline", "AXChildren": selected_rows["value"]},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Before"
    )

    with pytest.raises(errors.ComputerUseError, match="selected item"):
        backend.inspect_finder_rename(snapshot, 0, "After")

    editor_value["value"] = "Before"
    selected_rows["value"] = [bound_row, other_row]
    with pytest.raises(errors.ComputerUseError, match="selected item"):
        backend.inspect_finder_rename(snapshot, 0, "After")


def test_finder_rename_transaction_rejects_dialog_field(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a, **k: None)

    with pytest.raises(errors.ComputerUseError, match="selected item"):
        backend.inspect_finder_rename(snapshot, 0, "After")


def test_finder_rename_transaction_skips_enter_if_reference_already_committed(
    monkeypatch,
):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/After"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    presses = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda *a, **k: presses.append(a)
    )

    result = backend.commit_finder_rename("pid:4", snapshot, 0, binding)

    assert result["verified"] is True
    assert result["executed"] is False
    assert result["verification_source"] == "finder_file_reference_basename"
    assert presses == []


def test_finder_rename_transaction_stages_then_commits_exact_editor(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    writes = []
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *args: writes.append(args) or 0,
        kAXValueAttribute="AXValue",
    )

    staged = backend.set_finder_rename_value("pid:4", snapshot, 0, binding)

    assert staged["verified"] is None
    assert staged["verification_source"] == "pending"
    assert len(writes) == 1
    raises = []
    foreground = {"finder": False}  # Approval card left Rapid frontmost.

    def restore_finder(*args):
        assert foreground["finder"] is False
        raises.append(args)
        foreground["finder"] = True
        return snapshot["window"]

    monkeypatch.setattr(backend, "raise_selected_window", restore_finder)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda *a, **k: (
            None
            if foreground["finder"]
            else pytest.fail("Enter prepared before Finder focus was restored")
        ),
    )
    presses = []

    def press(key):
        presses.append(key)
        path_state[0] = "/tmp/After"

    monkeypatch.setattr(backend.ax_driver, "_press_key", press)
    committed = backend.commit_finder_rename("pid:4", snapshot, 0, binding)

    assert committed["verified"] is True
    assert committed["actual_basename"] == "After"
    assert raises == [("pid:4", snapshot)]
    assert presses == [backend.KEY_ALIASES["enter"]]


def test_finder_rename_resume_reopens_same_bound_row_after_approval(monkeypatch):
    snapshot = _finder_rename_snapshot()
    row, outline, reference, rename_item, restored_editor = (object() for _ in range(5))
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    backend._finder_rename_bindings[backend._finder_rename_binding_key(snapshot)] = (
        row,
        reference,
        "/tmp/Before",
        backend.time.monotonic(),
    )
    monkeypatch.setattr(
        backend,
        "_finder_transaction_editor",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError(
                "target_drift", "transient companion changed or lost focus"
            )
        ),
    )
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Before" if value is reference else None,
    )

    def get(element, attr):
        return {
            row: {"AXParent": outline, "AXRole": "AXRow", "AXSelected": True},
            outline: {"AXRole": "AXOutline", "AXChildren": [row]},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda value: value["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(backend, "_finder_rename_menu_item", lambda _: rename_item)
    actions = []
    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        lambda element, action: (
            actions.append((element, action)) or backend.ax_driver.kAXErrorSuccess
        ),
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        backend,
        "_finder_item_editor_for_path",
        lambda value, path, expected_row: (
            restored_editor
            if value is snapshot and path == "/tmp/Before" and expected_row is row
            else pytest.fail("rename editor rebound outside the approved row")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_finder_transaction_reference",
        lambda live, *a, **k: reference if live is restored_editor else None,
    )

    assert (
        backend._resume_finder_transaction_editor(binding, snapshot, 0)
        is restored_editor
    )
    assert actions == [(rename_item, "AXPress")]


def test_finder_rename_resume_accepts_exact_detached_focused_editor(monkeypatch):
    snapshot = _finder_rename_snapshot()
    editor, app_root, row, outline, reference = (object() for _ in range(5))
    backend._finder_rename_bindings[backend._finder_rename_binding_key(snapshot)] = (
        row,
        reference,
        "/tmp/Before",
        backend.time.monotonic(),
    )

    def get(element, attr):
        return {
            editor: {
                "AXRole": "AXTextField",
                "AXParent": app_root,
                "AXURL": None,
                "AXFocused": True,
                "AXValue": "Before\u200b",
            },
            app_root: {"AXRole": "AXApplication"},
            row: {"AXRole": "AXRow", "AXParent": outline, "AXSelected": True},
            outline: {"AXRole": "AXOutline", "AXChildren": [row]},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a: editor)
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Before" if value is reference else None,
    )
    monkeypatch.setattr(
        backend,
        "_focused_ax_window",
        lambda *a: pytest.fail("exact detached editor must not use tree fallback"),
    )

    assert backend._finder_item_editor_for_path(snapshot, "/tmp/Before", row) is editor


@pytest.mark.parametrize("mutation", ["value", "selection", "reference"])
def test_finder_rename_resume_rejects_unbound_detached_editor(monkeypatch, mutation):
    snapshot = _finder_rename_snapshot()
    editor, app_root, row, other_row, outline, reference = (object() for _ in range(6))
    backend._finder_rename_bindings[backend._finder_rename_binding_key(snapshot)] = (
        row,
        reference,
        "/tmp/Before",
        backend.time.monotonic(),
    )

    def get(element, attr):
        selected = [row, other_row] if mutation == "selection" else [row]
        return {
            editor: {
                "AXRole": "AXTextField",
                "AXParent": app_root,
                "AXURL": None,
                "AXFocused": True,
                "AXValue": "Other" if mutation == "value" else "Before",
            },
            app_root: {"AXRole": "AXApplication"},
            row: {"AXRole": "AXRow", "AXParent": outline, "AXSelected": True},
            other_row: {"AXRole": "AXRow", "AXSelected": True},
            outline: {"AXRole": "AXOutline", "AXChildren": selected},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a: editor)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda *a: None)
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: (
            "/tmp/Other"
            if mutation == "reference" and value is reference
            else "/tmp/Before"
            if value is reference
            else None
        ),
    )

    with pytest.raises(errors.ComputerUseError, match="unavailable or ambiguous"):
        backend._finder_item_editor_for_path(snapshot, "/tmp/Before", row)


def test_finder_rename_resume_rejects_different_selected_row(monkeypatch):
    snapshot = _finder_rename_snapshot()
    row, other_row, outline, reference = (object() for _ in range(4))
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    backend._finder_rename_bindings[backend._finder_rename_binding_key(snapshot)] = (
        row,
        reference,
        "/tmp/Before",
        backend.time.monotonic(),
    )
    monkeypatch.setattr(
        backend,
        "_finder_transaction_editor",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError("target_drift", "editor closed")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Before" if value is reference else None,
    )

    def get(element, attr):
        return {
            row: {"AXParent": outline, "AXRole": "AXRow", "AXSelected": False},
            other_row: {"AXRole": "AXRow", "AXSelected": True},
            outline: {"AXRole": "AXOutline", "AXChildren": [row, other_row]},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(
        backend,
        "_finder_rename_menu_item",
        lambda *a: pytest.fail("different selection must not consume approval"),
    )

    with pytest.raises(errors.ComputerUseError, match="approved Finder row changed"):
        backend._resume_finder_transaction_editor(binding, snapshot, 0)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_binding", "binding is no longer available"),
        ("menu_selection_drift", "approved Finder row changed"),
        ("menu_window_drift", "window changed during menu resolution"),
        ("rename_rejected", "Rename command rejected"),
        ("restored_reference", "not the approved item"),
    ],
)
def test_finder_rename_resume_fails_closed_at_each_reentry_boundary(
    monkeypatch, failure, message
):
    snapshot = _finder_rename_snapshot()
    row, outline, reference, rename_item, restored_editor = (object() for _ in range(5))
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    binding_key = backend._finder_rename_binding_key(snapshot)
    if failure == "missing_binding":
        backend._finder_rename_bindings.pop(binding_key, None)
    else:
        backend._finder_rename_bindings[binding_key] = (
            row,
            reference,
            "/tmp/Before",
            backend.time.monotonic(),
        )
    monkeypatch.setattr(
        backend,
        "_finder_transaction_editor",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError("target_drift", "editor closed")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Before" if value is reference else None,
    )

    selected = {"value": True}

    def get(element, attr):
        return {
            row: {
                "AXParent": outline,
                "AXRole": "AXRow",
                "AXSelected": selected["value"],
            },
            outline: {"AXRole": "AXOutline", "AXChildren": [row]},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)

    def resolve_menu(_snapshot):
        if failure == "menu_selection_drift":
            selected["value"] = False
        return rename_item

    monkeypatch.setattr(backend, "_finder_rename_menu_item", resolve_menu)
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda value: (
            (_ for _ in ()).throw(
                errors.ComputerUseError(
                    "target_drift", "window changed during menu resolution"
                )
            )
            if failure == "menu_window_drift"
            else value["window"]
        ),
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    actions = []
    monkeypatch.setattr(
        backend.ax_driver,
        "AXUIElementPerformAction",
        lambda *a: (
            actions.append(a)
            or (
                1 if failure == "rename_rejected" else backend.ax_driver.kAXErrorSuccess
            )
        ),
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        backend, "_finder_item_editor_for_path", lambda *a: restored_editor
    )
    monkeypatch.setattr(
        backend,
        "_finder_transaction_reference",
        lambda *a, **k: object() if failure == "restored_reference" else reference,
    )

    with pytest.raises(errors.ComputerUseError, match=message):
        backend._resume_finder_transaction_editor(binding, snapshot, 0)
    if failure in {"menu_selection_drift", "menu_window_drift"}:
        assert actions == []


def test_finder_rename_transaction_fails_closed_on_reference_drift(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Unexpected"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    presses = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda *a, **k: presses.append(a)
    )

    with pytest.raises(errors.ComputerUseError, match="moved or changed"):
        backend.commit_finder_rename("pid:4", snapshot, 0, binding)
    assert presses == []


def test_finder_rename_path_state_rejects_missing_reference(monkeypatch):
    monkeypatch.setattr(backend, "_finder_file_reference_path", lambda _value: None)
    with pytest.raises(errors.ComputerUseError, match="no longer available"):
        backend._finder_rename_path_state(
            {
                "file_reference": object(),
                "original_path": "/tmp/Before",
                "requested_basename": "After",
            }
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("window", "app or window"),
        ("shape", "editor shape"),
        ("selection", "selected row"),
        ("reference", "approved item"),
    ],
)
def test_finder_rename_transaction_revalidates_every_identity_boundary(
    monkeypatch, mutation, message
):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    live, reference = _install_selected_finder_editor(monkeypatch, snapshot, path_state)
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    if mutation == "window":
        snapshot["window_id"] = "cg:other"
    elif mutation == "shape":
        snapshot["elements"][0]["role"] = "AXButton"
    elif mutation == "selection":
        monkeypatch.setattr(
            backend, "_is_selected_finder_row_under_focused_outline", lambda *a: False
        )
    else:
        monkeypatch.setattr(
            backend, "_finder_file_reference_for_editor", lambda *a, **k: object()
        )
    with pytest.raises(errors.ComputerUseError, match=message):
        backend._finder_transaction_editor(binding, snapshot, 0, expected_value="After")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("app", "not a Finder"),
        ("basename", "one basename"),
        ("selection", "selected item"),
        ("shape", "item editor"),
        ("reference", "selected item"),
        ("path", "no stable file reference"),
    ],
)
def test_inspect_finder_rename_rejects_untrusted_bindings(
    monkeypatch, mutation, message
):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _install_selected_finder_editor(monkeypatch, snapshot, path_state)
    requested = "After"
    if mutation == "app":
        snapshot["app"]["bundleId"] = "com.apple.TextEdit"
    elif mutation == "basename":
        requested = "dir/After"
    elif mutation == "selection":
        monkeypatch.setattr(
            backend, "_is_selected_finder_row_under_focused_outline", lambda *a: False
        )
    elif mutation == "shape":
        snapshot["elements"][0]["role"] = "AXButton"
    elif mutation == "reference":
        monkeypatch.setattr(
            backend, "_finder_file_reference_for_editor", lambda *a, **k: None
        )
    elif mutation == "path":
        monkeypatch.setattr(backend, "_finder_file_reference_path", lambda _ref: None)
    with pytest.raises(errors.ComputerUseError, match=message):
        backend.inspect_finder_rename(snapshot, 0, requested)


def test_finder_rename_set_reports_immediate_disk_commit(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }

    def set_value(*_args):
        path_state[0] = "/tmp/After"
        return 0

    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=set_value,
        kAXValueAttribute="AXValue",
    )
    result = backend.set_finder_rename_value("pid:4", snapshot, 0, binding)
    assert result["verified"] is True
    assert result["verification_source"] == "finder_file_reference_basename"


def test_finder_rename_set_rejects_failed_ax_write(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *_args: 1,
        kAXValueAttribute="AXValue",
    )
    with pytest.raises(errors.ComputerUseError, match="rejected"):
        backend.set_finder_rename_value("pid:4", snapshot, 0, binding)


def test_finder_rename_set_rechecks_path_immediately_before_ax_write(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    checks = []

    def check(_binding):
        checks.append("path")
        if len(checks) == 2:
            raise errors.ComputerUseError("target_drift", "moved before write")
        return "unchanged", "/tmp/Before"

    monkeypatch.setattr(backend, "_finder_rename_path_state", check)
    writes = []
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *args: writes.append(args) or 0,
        kAXValueAttribute="AXValue",
    )

    with pytest.raises(errors.ComputerUseError, match="moved before write"):
        backend.set_finder_rename_value("pid:4", snapshot, 0, binding)
    assert checks == ["path", "path"]
    assert writes == []


def test_finder_rename_commit_rejects_changed_editor_and_unverified_dispatch(
    monkeypatch,
):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    live, reference = _install_selected_finder_editor(monkeypatch, snapshot, path_state)
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    original_get = backend.ax_driver._get
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: (
            "Different"
            if element is live and attr == "AXValue"
            else original_get(element, attr)
        ),
    )
    with pytest.raises(errors.ComputerUseError, match="approved basename"):
        backend.commit_finder_rename("pid:4", snapshot, 0, binding)
    monkeypatch.setattr(backend.ax_driver, "_get", original_get)
    monkeypatch.setattr(
        backend, "raise_selected_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda _key: None)
    monkeypatch.setattr(backend.time, "sleep", lambda _delay: None)
    result = backend.commit_finder_rename("pid:4", snapshot, 0, binding)
    assert result["verified"] is False
    assert result["executed"] is True


def test_finder_rename_focus_restore_commit_skips_duplicate_enter(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    raises = []

    def raise_window(*args):
        raises.append(args)
        path_state[0] = "/tmp/After"
        return snapshot["window"]

    monkeypatch.setattr(backend, "raise_selected_window", raise_window)
    presses = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda key: presses.append(key)
    )

    result = backend.commit_finder_rename("pid:4", snapshot, 0, binding)

    assert result["verified"] is True
    assert result["executed"] is False
    assert result["verification_source"] == "finder_file_reference_basename"
    assert raises == [("pid:4", snapshot)]
    assert presses == []


def test_finder_rename_drift_during_focus_restore_dispatches_no_enter(monkeypatch):
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    _live, reference = _install_selected_finder_editor(
        monkeypatch, snapshot, path_state
    )
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }

    def raise_window(*_args):
        path_state[0] = "/tmp/Unexpected"
        return snapshot["window"]

    monkeypatch.setattr(backend, "raise_selected_window", raise_window)
    presses = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda key: presses.append(key)
    )

    with pytest.raises(errors.ComputerUseError, match="moved or changed"):
        backend.commit_finder_rename("pid:4", snapshot, 0, binding)
    assert presses == []


def test_finder_inline_rename_verifies_committed_file_reference(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live = object()
    paths = iter(
        [
            "/tmp/untitled folder",
            "/tmp/untitled folder",
            "/tmp/untitled folder",
            "/tmp/Verified CUA Folder",
        ]
    )
    last_path = ["/tmp/untitled folder"]

    def current_path():
        try:
            last_path[0] = next(paths)
        except StopIteration:
            pass
        return last_path[0]

    reference = _FinderFileReference(current_path)
    selected_state = [False]
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snapshot: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    rename_item = _install_finder_rename_tree(
        monkeypatch,
        live,
        reference,
        "Verified CUA Folder\u200b\u200b",
        selected_state,
    )
    monkeypatch.setattr(backend, "_finder_item_editor_for_path", lambda *a: live)
    keys = []
    typed = []
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementPerformAction=lambda element, action: int(
            element is not rename_item
        ),
        AXUIElementSetAttributeValue=lambda *args: (
            selected_state.__setitem__(0, True) or 0
        ),
        kAXErrorSuccess=0,
    )
    _install_module(monkeypatch, "Foundation", NSURL=object())
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        backend.ax_driver,
        "_press_key",
        lambda key, modifiers=0: keys.append((key, modifiers)),
    )
    monkeypatch.setattr(backend.ax_driver, "_type_text", typed.append)

    result = backend.press_key(
        "pid:4", "enter", expected_snapshot=snapshot, element_index=0
    )

    assert keys == [
        (backend.ax_driver._keycode_for("a"), backend.ax_driver.FLAG_COMMAND),
        (backend.KEY_ALIASES["enter"], 0),
    ]
    assert typed == ["Verified CUA Folder"]
    assert result["ok"] is True
    assert result["verified"] is True
    assert result["actual_basename"] == "Verified CUA Folder"
    assert result["verification_source"] == "finder_file_reference_basename"


def test_finder_inline_rename_rejects_uncommitted_ax_value(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live = object()
    reference = _FinderFileReference(lambda: "/tmp/untitled folder")
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snapshot: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    rename_item = _install_finder_rename_tree(
        monkeypatch, live, reference, "Verified CUA Folder"
    )
    monkeypatch.setattr(backend, "_finder_item_editor_for_path", lambda *a: live)
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementPerformAction=lambda element, action: int(
            element is not rename_item
        ),
        kAXErrorSuccess=0,
    )
    _install_module(monkeypatch, "Foundation", NSURL=object())
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda *a, **k: None)
    monkeypatch.setattr(backend.ax_driver, "_type_text", lambda text: None)

    result = backend.press_key(
        "pid:4", "enter", expected_snapshot=snapshot, element_index=0
    )

    assert result["ok"] is False
    assert result["verified"] is False
    assert result["actual_basename"] == "untitled folder"


def test_finder_inline_rename_rejects_unselectable_exact_row(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, cell, row = object(), object(), object()
    reference = _FinderFileReference(lambda: "/tmp/Original Folder")
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snapshot: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {
                "AXURL": reference,
                "AXValue": "Verified CUA Folder",
                "AXParent": cell,
            },
            cell: {"AXRole": "AXCell", "AXParent": row},
            row: {"AXRole": "AXRow", "AXSelected": False},
        }.get(element, {}).get(attr),
    )
    _install_module(monkeypatch, "Foundation", NSURL=object())
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=lambda *a: 1,
        kAXErrorSuccess=0,
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("pid:4", "enter", expected_snapshot=snapshot, element_index=0)

    assert excinfo.value.code == "target_drift"
    assert "could not be selected" in excinfo.value.message


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("live_none", "editor changed"),
        ("bad_hierarchy", "bound item row"),
        ("selection_not_sticky", "did not accept selection"),
        ("rename_rejected", "rejected AXPress"),
        ("path_changed", "rename target changed"),
        ("editor_unfocused", "editor is not focused"),
    ],
)
def test_finder_inline_rename_fails_closed_at_each_native_boundary(
    monkeypatch, failure, message
):
    snapshot = _finder_rename_snapshot()
    live = object()
    if failure == "live_none":
        monkeypatch.setattr(backend, "_live_element", lambda *a, **k: None)
        with pytest.raises(errors.ComputerUseError, match=message):
            backend._finder_inline_rename("pid:4", snapshot, 0)
        return

    reference = _FinderFileReference(lambda: "/tmp/Original")
    selected = [failure not in {"selection_not_sticky"}]
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda value: value["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    rename_item = _install_finder_rename_tree(
        monkeypatch, live, reference, "Renamed", selected
    )
    original_get = backend.ax_driver._get
    if failure == "bad_hierarchy":
        monkeypatch.setattr(
            backend.ax_driver,
            "_get",
            lambda element, attr: (
                "AXGroup"
                if attr == "AXRole" and original_get(element, "AXRole") == "AXCell"
                else original_get(element, attr)
            ),
        )
    if failure == "editor_unfocused":
        monkeypatch.setattr(
            backend.ax_driver,
            "_get",
            lambda element, attr: (
                False
                if element is live and attr == "AXFocused"
                else original_get(element, attr)
            ),
        )
        monkeypatch.setattr(backend, "_focused_ax_element", lambda app: object())
    monkeypatch.setattr(backend, "_finder_item_editor_for_path", lambda *a: live)
    path_calls = 0

    def path(value):
        nonlocal path_calls
        path_calls += 1
        return (
            "/tmp/Changed"
            if failure == "path_changed" and path_calls >= 3
            else "/tmp/Original"
        )

    monkeypatch.setattr(backend, "_finder_file_reference_path", path)

    def set_attribute(*args):
        if failure != "selection_not_sticky":
            selected[0] = True
        return 0

    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementSetAttributeValue=set_attribute,
        AXUIElementPerformAction=lambda element, action: int(
            failure == "rename_rejected" and element is rename_item
        ),
        kAXErrorSuccess=0,
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)

    with pytest.raises(errors.ComputerUseError, match=message):
        backend._finder_inline_rename("pid:4", snapshot, 0)


def test_finder_inline_rename_without_reference_uses_generic_path(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: None)
    assert backend._finder_inline_rename("pid:4", snapshot, 0) is None


def test_finder_selected_row_enter_is_not_misclassified_as_commit(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live = object()
    reference = _FinderFileReference(lambda: "/tmp/Verified CUA Folder")
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            "AXURL": reference,
            "AXValue": "Verified CUA Folder",
        }.get(attr),
    )
    _install_module(monkeypatch, "Foundation", NSURL=object())

    assert backend._finder_inline_rename("pid:4", snapshot, 0) is None


def test_finder_selected_row_under_focused_outline_is_valid_keyboard_target(
    monkeypatch,
):
    snapshot = _finder_rename_snapshot()
    snapshot["elements"][0].update(
        {"role": "AXRow", "parent_role": "AXOutline", "label": "Selected item"}
    )
    row, outline = object(), object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: row)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: outline)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            row: {"AXRole": "AXRow", "AXSelected": True, "AXParent": outline},
            outline: {"AXRole": "AXOutline"},
        }.get(element, {}).get(attr),
    )

    assert (
        backend.inspect_focused_element(snapshot, 0, allow_selected_finder_row=True)
        == snapshot["elements"][0]
    )


def test_finder_selected_inline_editor_under_focused_outline_is_valid(monkeypatch):
    snapshot = _finder_rename_snapshot()
    editor, cell, row, outline = (object() for _ in range(4))
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: editor)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: outline)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            editor: {
                "AXRole": "AXTextField",
                "AXSelected": True,
                "AXParent": cell,
            },
            cell: {"AXRole": "AXCell", "AXSelected": True, "AXParent": row},
            row: {"AXRole": "AXRow", "AXSelected": True, "AXParent": outline},
            outline: {"AXRole": "AXOutline"},
        }.get(element, {}).get(attr),
    )

    assert (
        backend.inspect_focused_element(snapshot, 0, allow_selected_finder_row=True)
        == snapshot["elements"][0]
    )


def test_finder_dialog_text_field_is_not_valid_rename_target(monkeypatch):
    snapshot = _finder_rename_snapshot()
    editor, group, outline = object(), object(), object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: editor)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: outline)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            editor: {
                "AXRole": "AXTextField",
                "AXSelected": True,
                "AXParent": group,
            },
            group: {"AXRole": "AXGroup", "AXSelected": True},
            outline: {"AXRole": "AXOutline"},
        }.get(element, {}).get(attr),
    )

    with pytest.raises(errors.ComputerUseError, match="exact focused"):
        backend.inspect_focused_element(snapshot, 0, allow_selected_finder_row=True)


@pytest.mark.parametrize("selected", [False, None])
def test_finder_unselected_row_under_focused_outline_is_rejected(monkeypatch, selected):
    snapshot = _finder_rename_snapshot()
    snapshot["elements"][0].update(
        {"role": "AXRow", "parent_role": "AXOutline", "label": "Selected item"}
    )
    row, outline = object(), object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: row)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: outline)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            row: {"AXRole": "AXRow", "AXSelected": selected, "AXParent": outline},
            outline: {"AXRole": "AXOutline"},
        }.get(element, {}).get(attr),
    )

    with pytest.raises(errors.ComputerUseError, match="exact focused"):
        backend.inspect_focused_element(snapshot, 0, allow_selected_finder_row=True)


def test_activation_key_rechecks_exact_focus_immediately_before_dispatch(monkeypatch):
    snapshot = _finder_rename_snapshot()
    snapshot["app"] = {
        "name": "TextEdit",
        "bundleId": "com.apple.TextEdit",
        "pid": 4,
    }
    calls = []
    monkeypatch.setattr(backend, "_finder_inline_rename", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_prepare_synthetic_action", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend,
        "inspect_focused_element",
        lambda *a, **k: calls.append((a, k)) or snapshot["elements"][0],
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_press_key",
        lambda key: calls.append(("dispatch", key)),
    )

    backend.press_key("pid:4", "enter", expected_snapshot=snapshot, element_index=0)

    assert calls[-2][0][1] == 0
    assert calls[-2][1] == {"allow_selected_finder_row": True}
    assert calls[-1][0] == "dispatch"


def test_finder_generic_enter_verifies_same_file_reference_commit(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, reference = object(), object()
    paths = iter(["/tmp/Original", "/tmp/Original", "/tmp/Verified CUA Folder"])
    monkeypatch.setattr(backend, "_finder_inline_rename", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            "AXURL": reference,
            "AXValue": "Verified CUA Folder",
        }.get(attr),
    )
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: next(paths)
    )
    monkeypatch.setattr(backend, "_prepare_synthetic_action", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "inspect_focused_element", lambda *a, **k: snapshot["elements"][0]
    )
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda key: None)

    result = backend.press_key(
        "pid:4", "enter", expected_snapshot=snapshot, element_index=0
    )

    assert result["verified"] is True
    assert result["verification_source"] == "finder_file_reference_basename"
    assert result["actual_basename"] == "Verified CUA Folder"


def test_finder_generic_enter_uses_bound_reference_for_replacement_editor(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, cell, row, original_row, reference = (object() for _ in range(5))
    paths = iter(["/tmp/Original", "/tmp/Original", "/tmp/Verified CUA Folder"])
    key = backend._finder_rename_binding_key(snapshot)
    backend._finder_rename_bindings[key] = (
        original_row,
        reference,
        "/tmp/Original",
        backend.time.monotonic(),
    )
    monkeypatch.setattr(backend, "_finder_inline_rename", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {
                "AXURL": None,
                "AXValue": "Verified CUA Folder",
                "AXParent": cell,
            },
            cell: {"AXParent": row, "AXRole": "AXCell"},
            row: {"AXRole": "AXRow", "AXSelected": True},
        }.get(element, {}).get(attr),
    )
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: live)
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: next(paths)
    )
    monkeypatch.setattr(backend, "_prepare_synthetic_action", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "inspect_focused_element", lambda *a, **k: snapshot["elements"][0]
    )
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda key: None)

    result = backend.press_key(
        "pid:4", "enter", expected_snapshot=snapshot, element_index=0
    )

    assert result["verified"] is True
    assert result["verification_source"] == "finder_file_reference_basename"
    assert key not in backend._finder_rename_bindings


def test_finder_replacement_binding_fails_closed_on_missing_or_drift(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, cell, row, other_row, reference = (object() for _ in range(5))
    key = backend._finder_rename_binding_key(snapshot)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {"AXParent": cell, "AXURL": None},
            cell: {"AXParent": row, "AXRole": "AXCell"},
            row: {"AXRole": "AXRow", "AXSelected": True},
        }.get(element, {}).get(attr),
    )
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Original"
    )
    backend._finder_rename_bindings.pop(key, None)
    assert backend._finder_file_reference_for_editor(live, snapshot) is None

    backend._finder_rename_bindings[key] = (
        other_row,
        reference,
        "/tmp/Original",
        backend.time.monotonic(),
    )
    assert backend._finder_file_reference_for_editor(live, snapshot) is None
    assert key in backend._finder_rename_bindings
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: object())
    assert (
        backend._finder_file_reference_for_editor(
            live, snapshot, allow_selected_row_rebind=True
        )
        is None
    )


def test_finder_replacement_binding_expires_without_authorizing_editor(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, cell, row, reference = (object() for _ in range(4))
    key = backend._finder_rename_binding_key(snapshot)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {"AXParent": cell, "AXURL": None},
            cell: {"AXParent": row},
        }.get(element, {}).get(attr),
    )
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Original"
    )
    backend._finder_rename_bindings[key] = (
        row,
        reference,
        "/tmp/Original",
        backend.time.monotonic() - backend.SNAPSHOT_TTL_S - 1,
    )

    assert backend._finder_file_reference_for_editor(live, snapshot) is None
    assert key not in backend._finder_rename_bindings


def test_finder_rename_menu_requires_one_enabled_native_command(monkeypatch):
    snapshot = _finder_rename_snapshot()
    app_element, menu_bar, first, second = (object() for _ in range(4))
    children = [first, second]
    monkeypatch.setattr(backend.ax_driver, "_app_element", lambda *a, **k: app_element)

    def get(element, attr):
        if element is app_element and attr == "AXMenuBar":
            return menu_bar
        if element is menu_bar and attr == "AXChildren":
            return children
        if element in {first, second}:
            return {
                "AXRole": "AXMenuItem",
                "AXMenuItemCmdChar": "\r",
                "AXMenuItemCmdModifiers": 0,
                "AXEnabled": True,
                "AXChildren": [],
            }.get(attr)
        return None

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(backend.ax_driver, "_action_names", lambda item: ["AXPress"])

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._finder_rename_menu_item(snapshot)
    assert excinfo.value.code == "element_not_found"

    children.pop()
    assert backend._finder_rename_menu_item(snapshot) is first


def test_finder_replacement_editor_traversal_requires_unique_bound_path(monkeypatch):
    snapshot = _finder_rename_snapshot()
    expected_row, window, editor, cell, reference = (object() for _ in range(5))
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: None)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: window)

    def get(element, attr):
        return {
            window: {"AXChildren": [editor]},
            editor: {
                "AXRole": "AXTextField",
                "AXParent": cell,
                "AXURL": reference,
                "AXChildren": [],
            },
            cell: {"AXRole": "AXCell", "AXParent": expected_row},
            expected_row: {"AXSelected": True},
        }.get(element, {}).get(attr)

    monkeypatch.setattr(backend.ax_driver, "_get", get)
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Original"
    )

    assert (
        backend._finder_item_editor_for_path(snapshot, "/tmp/Original", expected_row)
        is editor
    )


def test_finder_inline_rename_rejects_focus_drift_before_typing(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, cell, row, rename_item = object(), object(), object(), object()
    reference = object()
    validations = []
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda snapshot: (
            validations.append(True),
            snapshot["window"]
            if len(validations) == 1
            else (_ for _ in ()).throw(
                errors.ComputerUseError("target_drift", "focus changed")
            ),
        )[1],
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Original"
    )
    monkeypatch.setattr(
        backend, "_finder_rename_menu_item", lambda snapshot: rename_item
    )
    monkeypatch.setattr(backend, "_finder_item_editor_for_path", lambda *a: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: live)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {
                "AXURL": reference,
                "AXValue": "Renamed",
                "AXParent": cell,
                "AXFocused": True,
            },
            cell: {"AXRole": "AXCell", "AXParent": row},
            row: {"AXRole": "AXRow", "AXSelected": True},
        }.get(element, {}).get(attr),
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementPerformAction=lambda *a: 0,
        kAXErrorSuccess=0,
    )
    monkeypatch.setattr(backend.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        backend.ax_driver,
        "_press_key",
        lambda *a, **k: pytest.fail("focus drift must not emit a key"),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_type_text",
        lambda *a: pytest.fail("focus drift must not emit text"),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("pid:4", "enter", expected_snapshot=snapshot, element_index=0)

    assert excinfo.value.code == "target_drift"
    assert len(validations) == 2


def test_finder_replacement_editor_rejects_a_different_selected_row(monkeypatch):
    snapshot = _finder_rename_snapshot()
    editor, cell, other_row, expected_row, window = (object() for _ in range(5))
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: editor)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: window)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            editor: {"AXRole": "AXTextField", "AXParent": cell, "AXURL": None},
            cell: {"AXRole": "AXCell", "AXParent": other_row},
            other_row: {"AXSelected": True},
            window: {"AXChildren": []},
        }.get(element, {}).get(attr),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._finder_item_editor_for_path(snapshot, "/tmp/Original", expected_row)

    assert excinfo.value.code == "target_drift"


def test_finder_replacement_editor_accepts_focused_exact_row(monkeypatch):
    snapshot = _finder_rename_snapshot()
    editor, cell, row = object(), object(), object()
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: editor)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            editor: {"AXRole": "AXTextField", "AXParent": cell, "AXURL": None},
            cell: {"AXRole": "AXCell", "AXParent": row},
            row: {"AXSelected": True},
        }.get(element, {}).get(attr),
    )

    assert (
        backend._finder_item_editor_for_path(snapshot, "/tmp/Original", row) is editor
    )


def test_finder_replacement_editor_traversal_is_depth_bounded(monkeypatch):
    snapshot = _finder_rename_snapshot()
    row = object()
    window = {"AXRole": "AXWindow", "AXChildren": []}
    cursor = window
    for _ in range(backend.TEXTEDIT_VALUE_MAX_DEPTH + 2):
        child = {"AXRole": "AXGroup", "AXChildren": []}
        cursor["AXChildren"] = [child]
        cursor = child
    monkeypatch.setattr(backend, "_focused_ax_element", lambda app: None)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda app: window)
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._finder_item_editor_for_path(snapshot, "/tmp/Original", row)
    assert excinfo.value.code == "target_drift"


def test_finder_generic_enter_retries_file_reference_before_unverified(
    monkeypatch,
):
    snapshot = _finder_rename_snapshot()
    live, reference = object(), object()
    monkeypatch.setattr(backend, "_finder_inline_rename", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            "AXURL": reference,
            "AXValue": "Verified CUA Folder",
        }.get(attr),
    )
    monkeypatch.setattr(
        backend,
        "_finder_file_reference_path",
        lambda value: "/tmp/Original",
    )
    monkeypatch.setattr(backend, "_prepare_synthetic_action", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        backend, "inspect_focused_element", lambda *a, **k: snapshot["elements"][0]
    )
    monkeypatch.setattr(backend.ax_driver, "_press_key", lambda key: None)
    sleeps = []
    monkeypatch.setattr(backend.time, "sleep", sleeps.append)

    result = backend.press_key(
        "pid:4", "enter", expected_snapshot=snapshot, element_index=0
    )

    assert result["verified"] is None
    assert sleeps == [0.1] * 10


def test_finder_inline_rename_rejects_drift_during_menu_resolution(monkeypatch):
    snapshot = _finder_rename_snapshot()
    live, cell, row, rename_item = object(), object(), object(), object()
    reference = object()
    selected = [True]
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda snapshot: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(
        backend, "_finder_file_reference_path", lambda value: "/tmp/Original"
    )
    monkeypatch.setattr(
        backend,
        "_finder_rename_menu_item",
        lambda snapshot: (selected.__setitem__(0, False), rename_item)[1],
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            live: {
                "AXURL": reference,
                "AXValue": "Renamed",
                "AXParent": cell,
            },
            cell: {"AXRole": "AXCell", "AXParent": row},
            row: {"AXRole": "AXRow", "AXSelected": selected[0]},
        }.get(element, {}).get(attr),
    )
    _install_module(
        monkeypatch,
        "ApplicationServices",
        AXUIElementPerformAction=lambda *a: pytest.fail(
            "selection drift must prevent Rename AXPress"
        ),
        kAXErrorSuccess=0,
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("pid:4", "enter", expected_snapshot=snapshot, element_index=0)

    assert excinfo.value.code == "target_drift"
    assert "menu resolution" in excinfo.value.message


def test_transient_enter_requires_same_focused_companion(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "named folder",
                "center": [1332, 936],
                "actions": [],
                "source_window_id": "cg:1803",
            }
        ]
    )
    transient = {
        "index": 0,
        "window_id": "cg:1803",
        "x": 1288,
        "y": 926,
        "width": 88,
        "height": 21,
    }
    snapshot["transient_window"] = transient
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_focused_transient_window", lambda *a, **k: transient)
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a, **k: False)
    monkeypatch.setattr(
        backend, "_validate_focused_window", lambda snap, expected: None
    )
    pressed = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda key: pressed.append(key)
    )

    result = backend.press_key(
        "pid:4", "Enter", expected_snapshot=snapshot, element_index=0
    )

    assert result["mode"] == "CGEvent-keycode"
    assert pressed == [backend.KEY_ALIASES["enter"]]

    monkeypatch.setattr(backend, "_focused_ax_window", lambda *a, **k: object())
    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("pid:4", "Enter", expected_snapshot=snapshot, element_index=0)
    assert excinfo.value.code == "target_drift"
    assert pressed == [backend.KEY_ALIASES["enter"]]


@pytest.mark.parametrize("failure", ["missing_identity", "companion_drift"])
def test_synthetic_transient_dispatch_revalidates_window_identity(monkeypatch, failure):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "name",
                "center": [10, 10],
                "source_window_id": "cg:1803",
            }
        ]
    )
    transient = _window(window_id=1803, x=5, y=5, width=20, height=20)
    if failure != "missing_identity":
        snapshot["transient_window"] = transient
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_window", lambda *a: live)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a: True)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )
    monkeypatch.setattr(backend, "_focused_transient_window", lambda *a, **k: None)

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._prepare_synthetic_action(
            "pid:4", None, snapshot, 0, allow_focused_editable_enter=True
        )
    assert excinfo.value.code == "target_drift"


def test_synthetic_focused_editable_requires_numeric_center(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "name",
                "center": ["left", 10],
            }
        ]
    )
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a: live)
    monkeypatch.setattr(
        backend, "_validate_snapshot_window", lambda *a, **k: snapshot["window"]
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._prepare_synthetic_action(
            "pid:4", None, snapshot, 0, allow_focused_editable_enter=True
        )
    assert excinfo.value.code == "target_drift"


def test_synthetic_noneditable_target_uses_window_center_validation(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[{"index": 0, "role": "AXButton", "label": "Open", "center": [5, 5]}]
    )
    validated = []
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda value, **kwargs: (
            validated.append(kwargs.get("point")) or snapshot["window"]
        ),
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)

    assert backend._prepare_synthetic_action("pid:4", None, snapshot, 0) is snapshot
    assert validated == [(50.0, 50.0)]


def test_finder_rename_menu_walk_is_depth_bounded(monkeypatch):
    leaf = {
        "AXRole": "AXMenuItem",
        "AXTitle": "Rename",
        "AXEnabled": True,
        "actions": ["AXPress"],
        "AXChildren": [],
    }
    menu = leaf
    for _ in range(backend.SAVE_MENU_MAX_DEPTH + 2):
        menu = {"AXRole": "AXMenu", "AXChildren": [menu]}
    monkeypatch.setattr(
        backend.ax_driver, "_app_element", lambda *a, **k: {"AXMenuBar": menu}
    )
    monkeypatch.setattr(
        backend.ax_driver, "_get", lambda element, attr: element.get(attr)
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_action_names",
        lambda element: list(element.get("actions", [])),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend._finder_rename_menu_item(_finder_rename_snapshot())
    assert excinfo.value.code == "element_not_found"


def test_transient_companion_rejects_non_enter_key_before_dispatch(monkeypatch):
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "named folder",
                "center": [1332, 936],
                "actions": [],
                "source_window_id": "cg:1803",
            }
        ]
    )
    snapshot["transient_window"] = {"window_id": "cg:1803"}
    pressed = []
    monkeypatch.setattr(
        backend.ax_driver, "_press_key", lambda key: pressed.append(key)
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.press_key("pid:4", "a", expected_snapshot=snapshot, element_index=0)

    assert excinfo.value.code == "unsupported_key"
    assert pressed == []


def test_transient_focused_editable_is_only_successful_for_press_focus_phase(
    monkeypatch,
):
    live = object()
    snapshot = _stable_snapshot(
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "named folder",
                "center": [1332, 936],
                "actions": [],
                "source_window_id": "cg:1803",
            }
        ]
    )
    snapshot["transient_window"] = {"window_id": "cg:1803"}
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a, **k: False)
    monkeypatch.setattr(
        backend.ax_driver,
        "_cg_click",
        lambda *a, **k: pytest.fail("transient target must not use coordinates"),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.click("pid:4", 0, expected_snapshot=snapshot)
    assert excinfo.value.code == "synthetic_input_blocked"

    result = backend.click("pid:4", 0, expected_snapshot=snapshot, focus_only=True)
    assert result["mode"] == "AXFocusVerified"


def test_anchor_focused_editable_skips_occluded_center_for_press_focus_phase(
    monkeypatch,
):
    snapshot = _stable_snapshot(
        window_id=22,
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "Address and search",
                "center": [500, 90],
                "actions": [],
                "source_window_id": "cg:22",
            }
        ],
    )
    live = object()
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: live)
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: live)
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a, **k: False)
    focused_windows = []
    monkeypatch.setattr(
        backend,
        "_validate_focused_window",
        lambda snap, expected=None: focused_windows.append(expected),
    )
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda *a, **k: pytest.fail(
            "focused AX target must not probe its covered center"
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_cg_click",
        lambda *a, **k: pytest.fail("focused AX target must not receive a click"),
    )

    result = backend.click("pid:4", 0, expected_snapshot=snapshot, focus_only=True)

    assert result["mode"] == "AXFocusVerified"
    assert focused_windows == [None]


def test_anchor_editable_focus_drift_keeps_occlusion_fail_closed(monkeypatch):
    snapshot = _stable_snapshot(
        window_id=22,
        elements=[
            {
                "index": 0,
                "role": "AXTextField",
                "label": "Address and search",
                "center": [500, 90],
                "actions": [],
                "source_window_id": "cg:22",
            }
        ],
    )
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: object())
    monkeypatch.setattr(backend, "_focused_ax_element", lambda *a, **k: object())
    monkeypatch.setattr(backend.ax_driver, "_get", lambda *a, **k: False)
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda *a, **k: (_ for _ in ()).throw(
            errors.ComputerUseError("target_occluded", "covered")
        ),
    )
    monkeypatch.setattr(
        backend.ax_driver,
        "_cg_click",
        lambda *a, **k: pytest.fail("focus drift must not dispatch a click"),
    )

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.click("pid:4", 0, expected_snapshot=snapshot, focus_only=True)

    assert excinfo.value.code == "target_occluded"


def test_set_value_and_synthetic_fill_paths(monkeypatch):
    synthetic_fill = backend._synthetic_fill
    snapshot = _stable_snapshot(elements=[{"index": 0, "center": [1, 2]}])
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
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
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_snapshot_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
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
        CGEventSetLocation=lambda *a: None,
        kCGHIDEventTap=0,
        kCGScrollEventUnitLine=0,
    )
    snapshot_for_input = _stable_snapshot(
        elements=[{"index": 0, "actions": ["AXShowMenu"], "center": [6, 12]}]
    )
    monkeypatch.setattr(
        backend,
        "_prepare_synthetic_action",
        lambda app, window_id, *a, **k: (
            calls.append(("resolve", app)) or snapshot_for_input
        ),
    )
    monkeypatch.setattr(
        backend,
        "_validate_snapshot_window",
        lambda *a, **k: snapshot_for_input["window"],
    )
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
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
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot_for_input)
    backend.scroll("A", "up", x=3, y=4)

    snapshot = snapshot_for_input
    monkeypatch.setattr(backend, "get_app_state", lambda *a, **k: snapshot)
    monkeypatch.setattr(backend, "_live_element", lambda *a, **k: "live")
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
        backend,
        "_resolve_app",
        lambda app, **kwargs: (object(), {"name": "target app", "pid": 42}),
    )
    windows = [
        {"kCGWindowOwnerName": "other", "kCGWindowOwnerPID": 9, "kCGWindowLayer": 0},
        {
            "kCGWindowOwnerName": "target app",
            "kCGWindowOwnerPID": 42,
            "kCGWindowLayer": 1,
        },
        {
            "kCGWindowOwnerName": "target app",
            "kCGWindowOwnerPID": 42,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 77,
            "kCGWindowName": "Main",
            "kCGWindowBounds": {"X": 1, "Y": 2, "Width": 3, "Height": 4},
        },
    ]
    quartz.CGWindowListCopyWindowInfo = lambda *_: windows
    listed = backend.list_windows("A")[0]
    assert listed["title"] == "Main"
    assert listed["window_id"] == "cg:77"
    assert (
        backend._select_window(
            {"name": "target app", "pid": 42}, window_id=listed["window_id"]
        )["window_id"]
        == "cg:77"
    )
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


def test_live_app_discovery_refreshes_stale_workspace_by_pid(monkeypatch):
    cached = _RunningApp("Target App", "com.test.app", 42)
    terminated = _RunningApp("Old App", "com.test.old", 50)
    terminated.isTerminated = lambda: True
    safari = _RunningApp("Safari", "com.apple.Safari", 23429)
    resolved = {42: cached, 50: terminated, 23429: safari}
    fake_as = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(
            sharedWorkspace=lambda: _Workspace([cached, terminated])
        ),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: resolved.get(pid)
        ),
    )
    monkeypatch.setattr(ax_driver, "AS", fake_as)
    _install_module(
        monkeypatch,
        "ApplicationServices",
        NSWorkspace=fake_as.NSWorkspace,
        NSRunningApplication=fake_as.NSRunningApplication,
    )
    _install_module(
        monkeypatch,
        "Quartz",
        kCGWindowListOptionAll=1,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_: [
            {"kCGWindowOwnerPID": 23429},
        ],
    )

    assert [app.processIdentifier() for app in ax_driver._running_applications()] == [
        42,
        23429,
    ]
    assert backend.list_apps() == [
        {"name": "Target App", "bundleId": "com.test.app", "pid": 42},
        {"name": "Safari", "bundleId": "com.apple.Safari", "pid": 23429},
    ]

    monkeypatch.setattr(
        backend,
        "_ax_app_element",
        lambda app, activate: ("ax", app.processIdentifier()),
    )
    element, info = backend._resolve_app("pid:23429", activate=False)
    assert element == ("ax", 23429)
    assert info == {"name": "safari", "bundleId": "com.apple.safari", "pid": 23429}

    monkeypatch.setattr(
        ax_driver, "AXUIElementCreateApplication", lambda pid: ("ax", pid)
    )
    monkeypatch.setattr(ax_driver.time, "sleep", lambda _: None)
    assert ax_driver._app_element("safari", expected_pid=23429) == ("ax", 23429)


def test_target_window_discovery_is_bounded_ordered_and_excludes_rapid(monkeypatch):
    rapid = _RunningApp("Rapid", "com.rapidmlx.rapid", 10)
    finder = _RunningApp("Finder", "com.apple.finder", 42)
    textedit = _RunningApp("TextEdit", "com.apple.TextEdit", 43)
    workspace = _Workspace([rapid, finder, textedit])
    _install_module(
        monkeypatch,
        "ApplicationServices",
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: workspace),
    )
    quartz = _install_module(
        monkeypatch,
        "Quartz",
        kCGNullWindowID=0,
        kCGWindowListExcludeDesktopElements=1,
        kCGWindowListOptionOnScreenOnly=2,
    )
    quartz.CGWindowListCopyWindowInfo = lambda *_: [
        {
            "kCGWindowOwnerPID": 10,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 1,
            "kCGWindowBounds": {"Width": 800, "Height": 600},
        },
        {
            "kCGWindowOwnerPID": 43,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 2,
            "kCGWindowName": "Notes",
            "kCGWindowBounds": {"X": 1, "Y": 2, "Width": 800, "Height": 600},
        },
        {
            "kCGWindowOwnerPID": 42,
            "kCGWindowLayer": 0,
            "kCGWindowNumber": 3,
            "kCGWindowName": "Documents",
            "kCGWindowBounds": {"X": 3, "Y": 4, "Width": 900, "Height": 700},
        },
    ]

    catalog = backend.discover_target_windows(limit=1)

    assert len(catalog) == 1
    assert catalog[0]["catalog_id"] == "w1"
    assert catalog[0]["app"]["name"] == "TextEdit"
    assert catalog[0]["window"]["window_id"] == "cg:2"


def test_resolved_app_info_falls_back_to_process_create_time(monkeypatch):
    running = types.SimpleNamespace(
        localizedName=lambda: "Finder",
        bundleIdentifier=lambda: "com.apple.finder",
        processIdentifier=lambda: 42,
        launchDate=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        types.SimpleNamespace(
            Process=lambda pid: types.SimpleNamespace(create_time=lambda: 1234.5)
        ),
    )

    assert backend._resolved_app_info(running)["processStartTime"] == 1234.5


def test_resolved_app_info_includes_process_incarnation_marker():
    launched = types.SimpleNamespace(timeIntervalSince1970=lambda: 1234.5)
    running = types.SimpleNamespace(
        localizedName=lambda: "TextEdit",
        bundleIdentifier=lambda: "com.apple.TextEdit",
        processIdentifier=lambda: 42,
        launchDate=lambda: launched,
    )
    assert backend._resolved_app_info(running) == {
        "name": "textedit",
        "bundleId": "com.apple.textedit",
        "pid": 42,
        "processStartTime": 1234.5,
    }


def test_live_duplicate_browser_pid_keeps_url_guard_fail_closed(monkeypatch):
    selected = _RunningApp("Safari", "com.apple.Safari", 23429)
    duplicate = _RunningApp("Safari", "com.apple.Safari", 23430)
    resolved = {23429: selected, 23430: duplicate}
    fake_as = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(
            sharedWorkspace=lambda: _Workspace([selected])
        ),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: resolved.get(pid)
        ),
    )
    monkeypatch.setattr(ax_driver, "AS", fake_as)
    _install_module(
        monkeypatch,
        "Quartz",
        kCGWindowListOptionAll=1,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_: [
            {"kCGWindowOwnerPID": 23429},
            {"kCGWindowOwnerPID": 23430},
        ],
    )
    monkeypatch.setattr(
        backend,
        "_resolve_app",
        lambda *a, **k: (
            object(),
            {"name": "safari", "bundleId": "com.apple.safari", "pid": 23429},
        ),
    )
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: {"index": 0})
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a, **k: None)
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: pytest.fail("ambiguous browser must not reach AppleScript"),
    )

    assert backend.read_url("pid:23429", window_id="cg:1") == ""


def test_live_app_discovery_falls_back_when_core_graphics_fails(monkeypatch):
    cached = _RunningApp("Target App", "com.test.app", 42)
    fake_as = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(sharedWorkspace=lambda: _Workspace([cached])),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: (
                cached if pid == 42 else None
            )
        ),
    )
    monkeypatch.setattr(ax_driver, "AS", fake_as)
    _install_module(
        monkeypatch,
        "Quartz",
        kCGWindowListOptionAll=1,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_: (_ for _ in ()).throw(
            RuntimeError("CG service unavailable")
        ),
    )

    assert ax_driver._running_applications() == [cached]


def test_secure_ax_subrole_is_redacted_before_snapshot_and_tree(monkeypatch):
    secret = "hunter2-private"
    accessed = []

    def get_attribute(_element, attribute):
        accessed.append(attribute)
        return {
            "AXRole": "AXTextField",
            "AXSubrole": "AXSecureTextField",
            "AXValue": secret,
            "AXChildren": [],
        }.get(attribute)

    monkeypatch.setattr(ax_driver, "_get", get_attribute)
    monkeypatch.setattr(ax_driver, "_action_names", lambda _element: [])
    monkeypatch.setattr(ax_driver, "_point_size", lambda _element: (1, 2, 3, 4))
    targets = []
    ax_driver._walk("secure", 0, targets, [0])

    assert targets[0]["role"] == "AXTextField"
    assert targets[0]["subrole"] == "AXSecureTextField"
    assert targets[0]["text"] == "[secure text redacted]"
    assert "AXValue" not in accessed
    assert secret not in repr(targets)

    app_info = {"name": "A", "bundleId": "a.test", "pid": 42}
    window = {
        "window_id": "cg:123",
        "index": 0,
        "title": "Window",
        "x": 0,
        "y": 0,
        "width": 100,
        "height": 100,
    }
    monkeypatch.setattr(backend, "_resolve_app", lambda *a, **k: (object(), app_info))
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: window)
    monkeypatch.setattr(backend, "_window_records", lambda app: [window])
    monkeypatch.setattr(backend, "_collect_with_timeout", lambda *a, **k: targets)
    snapshot = backend.get_app_state("pid:42", screenshot=False, use_cache=False)
    assert secret not in repr(snapshot)
    assert secret not in snapshot["tree_text"]
    assert "[secure text redacted]" in snapshot["tree_text"]


def test_ax_driver_tree_collect_and_events(monkeypatch):
    class ObjCArray:
        def __init__(self, values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

    attrs = {
        "root": {"AXRole": "AXWindow", "AXChildren": ObjCArray(["button"])},
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
    assert ax_driver._as_list(None) == []
    assert ax_driver._as_list(object()) == []

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

    monkeypatch.setattr(
        ax_driver,
        "_get",
        lambda e, a: ["front", "selected"] if a == "AXWindows" else None,
    )
    monkeypatch.setattr(
        ax_driver,
        "_point_size",
        lambda window: {
            "front": (0.0, 0.0, 50.0, 50.0),
            "selected": (100.0, 100.0, 80.0, 60.0),
        }[window],
    )
    selected = ax_driver.collect(
        "A",
        keep_elements=True,
        max_windows=1,
        window_frame=(100.0, 100.0, 80.0, 60.0),
    )
    assert selected[0]["element"] == "selected"
    monkeypatch.setattr(
        ax_driver,
        "_point_size",
        lambda window: {
            "front": (1291.0, 929.0, 82.0, 15.0),
            "selected": (100.0, 100.0, 80.0, 60.0),
        }[window],
    )
    inset = ax_driver.collect(
        "A",
        keep_elements=True,
        max_windows=1,
        window_frame=(1288.0, 926.0, 88.0, 21.0),
        window_frame_tolerance=4.0,
    )
    assert inset[0]["element"] == "front"
    with pytest.raises(RuntimeError, match="exactly one AX window"):
        ax_driver.collect("A", window_frame=(10.0, 10.0, 10.0, 10.0))

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
    helper = _RunningApp("ThemeWidgetControlViewService (Target App)", pid=41)
    app = _RunningApp()
    fake_as = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(
            sharedWorkspace=lambda: _Workspace([helper, app])
        )
    )
    monkeypatch.setattr(ax_driver, "AS", fake_as)
    monkeypatch.setattr(
        ax_driver, "AXUIElementCreateApplication", lambda pid: ("element", pid)
    )
    monkeypatch.setattr(
        ax_driver,
        "_get",
        lambda element, attr: ["window"] if attr == "AXWindows" else None,
    )
    sets = []
    monkeypatch.setattr(
        ax_driver, "AXUIElementSetAttributeValue", lambda *a: sets.append(a)
    )
    monkeypatch.setattr(ax_driver.time, "sleep", lambda _: None)
    assert ax_driver._app_element("Target App") == ("element", 42)
    assert sets == []  # native apps must not be forced into manual AX mode

    same_name_other_pid = _RunningApp("Target App", pid=99)
    fake_as.NSWorkspace = types.SimpleNamespace(
        sharedWorkspace=lambda: _Workspace([same_name_other_pid, app])
    )
    assert ax_driver._app_element("Target App", expected_pid=42) == ("element", 42)

    chrome = _RunningApp("Chrome", "com.google.Chrome", pid=43)
    fake_as.NSWorkspace = types.SimpleNamespace(
        sharedWorkspace=lambda: _Workspace([chrome])
    )
    assert ax_driver._app_element("Chrome") == ("element", 43)
    assert [call[1] for call in sets] == [
        ax_driver._MANUAL_ACCESSIBILITY,
        ax_driver._ENHANCED_UI,
    ]
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
    sets_before_collect = list(sets)
    collected = ax_driver.collect("A", keep_elements=True)
    assert collected and attempts["n"] == 1
    assert collected[0]["role"] == "AXButton"
    collected = ax_driver.collect("A", keep_elements=True, retry_web_content=True)
    assert collected and attempts["n"] == 2
    assert collected[0]["role"] == "AXWebArea"
    assert sets == sets_before_collect

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
    monkeypatch.setattr(backend, "_ax_app_element", lambda app, **kwargs: object())
    with pytest.raises(errors.ComputerUseError, match="no running app"):
        backend._resolve_app("pid:8")

    snapshot = _stable_snapshot(elements=[{"index": 0, "center": [1, 2]}])
    monkeypatch.setattr(backend, "_validate_snapshot_window", lambda *a, **k: _window())
    monkeypatch.setattr(backend, "_validate_focused_window", lambda *a: None)
    monkeypatch.setattr(backend, "_select_window", lambda *a, **k: _window())
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


def test_ax_walk_keeps_blank_editable_control_without_adding_empty_structure(
    monkeypatch,
):
    attributes = {
        "root": {"AXRole": "AXWindow", "AXChildren": ["editor", "empty-label"]},
        "editor": {
            "AXRole": "AXTextArea",
            "AXValue": "",
            "AXChildren": [],
        },
        "empty-label": {"AXRole": "AXStaticText", "AXChildren": []},
    }
    monkeypatch.setattr(
        ax_driver,
        "_get",
        lambda element, attribute: attributes.get(element, {}).get(attribute),
    )
    monkeypatch.setattr(ax_driver, "_action_names", lambda _element: [])
    monkeypatch.setattr(
        ax_driver,
        "_point_size",
        lambda element: (10, 20, 300, 200) if element == "editor" else None,
    )

    targets = []
    ax_driver._walk("root", 0, targets, [0])

    assert len(targets) == 1
    assert targets[0]["role"] == "AXTextArea"
    assert targets[0]["text"] == ""
    assert targets[0]["actions"] == []
    assert targets[0]["rect"] == (10, 20, 300, 200)


def test_ax_walk_prioritizes_top_controls_before_long_table(monkeypatch):
    cells = [f"cell-{index}" for index in range(20)]
    attributes = {
        "root": {"AXRole": "AXWindow", "AXChildren": ["table", "toolbar"]},
        "table": {"AXRole": "AXTable", "AXChildren": cells},
        "toolbar": {"AXRole": "AXToolbar", "AXChildren": ["cpu", "memory"]},
        "cpu": {"AXRole": "AXRadioButton", "AXTitle": "CPU", "AXChildren": []},
        "memory": {
            "AXRole": "AXRadioButton",
            "AXTitle": "Memory",
            "AXChildren": [],
        },
        **{
            cell: {
                "AXRole": "AXStaticText",
                "AXValue": f"process {index}",
                "AXChildren": [],
            }
            for index, cell in enumerate(cells)
        },
    }
    monkeypatch.setattr(ax_driver, "MAX_NODES", 4)
    monkeypatch.setattr(
        ax_driver,
        "_get",
        lambda element, attribute: attributes.get(element, {}).get(attribute),
    )
    monkeypatch.setattr(ax_driver, "_action_names", lambda _element: [])
    monkeypatch.setattr(ax_driver, "_point_size", lambda _element: None)

    targets = []
    ax_driver._walk("root", 0, targets, [0])

    assert [target["text"] for target in targets] == [
        "CPU",
        "Memory",
        "process 0",
        "process 1",
    ]
    assert [target["target_id"] for target in targets] == [
        "t000",
        "t001",
        "t002",
        "t003",
    ]


def test_ax_walk_does_not_prescan_large_sibling_collections(monkeypatch):
    children = [f"cell-{index}" for index in range(65)]
    role_reads = []

    def get_attribute(element, attribute):
        if element == "root":
            return (
                "AXWindow"
                if attribute == "AXRole"
                else (children if attribute == "AXChildren" else None)
            )
        if attribute == "AXRole":
            role_reads.append(element)
            return "AXStaticText"
        if attribute == "AXValue":
            return element
        return [] if attribute == "AXChildren" else None

    monkeypatch.setattr(ax_driver, "MAX_NODES", 2)
    monkeypatch.setattr(ax_driver, "_get", get_attribute)
    monkeypatch.setattr(ax_driver, "_action_names", lambda _element: [])
    monkeypatch.setattr(ax_driver, "_point_size", lambda _element: None)

    targets = []
    ax_driver._walk("root", 0, targets, [0])

    assert [target["text"] for target in targets] == ["cell-0", "cell-1"]
    assert set(role_reads) == {"cell-0", "cell-1"}


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


def test_permission_request_prompts_exact_grant_and_returns_fresh_preflight(
    monkeypatch,
):
    calls = []
    application_services = types.SimpleNamespace(
        kAXTrustedCheckOptionPrompt="prompt",
        AXIsProcessTrustedWithOptions=lambda options: (
            calls.append(("ax", options)) or True
        ),
    )
    quartz = types.SimpleNamespace(
        CGRequestScreenCaptureAccess=lambda: calls.append(("screen", None)) or True,
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", application_services)
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setattr(
        backend,
        "permissions",
        lambda: {
            "accessibility": False,
            "screen_recording": True,
            "hints": [],
        },
    )

    accessibility = backend.request_permission("accessibility")
    screen = backend.request_permission("screen_recording")

    assert calls == [("ax", {"prompt": True}), ("screen", None)]
    # A request result never promotes a permission above fresh preflight.
    assert accessibility["granted"] is False
    assert screen["granted"] is True
    assert accessibility["permissions"]["accessibility"] is False


def test_permission_request_failure_is_typed_and_unknown_value_never_prompts(
    monkeypatch,
):
    calls = []
    application_services = types.SimpleNamespace(
        kAXTrustedCheckOptionPrompt="prompt",
        AXIsProcessTrustedWithOptions=lambda options: (
            calls.append(options) or (_ for _ in ()).throw(RuntimeError("denied"))
        ),
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", application_services)
    monkeypatch.setitem(sys.modules, "Quartz", types.SimpleNamespace())

    with pytest.raises(errors.ComputerUseError) as excinfo:
        backend.request_permission("accessibility")
    assert excinfo.value.code == "permission_request_failed"
    assert "denied" not in excinfo.value.message
    assert calls == [{"prompt": True}]

    with pytest.raises(ValueError, match="unsupported computer use permission"):
        backend.request_permission("automation")
    assert calls == [{"prompt": True}]


def test_ax_discovery_handles_unavailable_services_and_stale_processes(monkeypatch):
    monkeypatch.setattr(ax_driver, "AS", None)
    assert ax_driver._application_for_pid(42) is None
    assert ax_driver._running_applications() == []

    terminated = _RunningApp("Old App", pid=42)
    terminated.isTerminated = lambda: True
    services = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(
            sharedWorkspace=lambda: _Workspace([terminated])
        ),
        NSRunningApplication=types.SimpleNamespace(
            runningApplicationWithProcessIdentifier_=lambda pid: terminated
        ),
    )
    monkeypatch.setattr(ax_driver, "AS", services)
    _install_module(
        monkeypatch,
        "Quartz",
        kCGWindowListOptionAll=1,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_: [],
    )
    assert ax_driver._running_applications() == []


def test_ax_selector_rejects_stale_pid_and_unrelated_name(monkeypatch):
    mismatched_pid = _RunningApp("Target App", pid=99)
    unrelated_name = _RunningApp("Other App", pid=42)
    monkeypatch.setattr(ax_driver, "AS", object())
    monkeypatch.setattr(ax_driver, "_application_for_pid", lambda pid: mismatched_pid)
    with pytest.raises(SystemExit, match="not found"):
        ax_driver._app_element("Target App", expected_pid=42)

    monkeypatch.setattr(ax_driver, "_running_applications", lambda: [unrelated_name])
    with pytest.raises(SystemExit, match="not found"):
        ax_driver._app_element("Target App")


def test_ax_running_apps_filters_terminated_cached_entry(monkeypatch):
    terminated = _RunningApp("Old App", pid=42)
    terminated.isTerminated = lambda: True
    services = types.SimpleNamespace(
        NSWorkspace=types.SimpleNamespace(
            sharedWorkspace=lambda: _Workspace([terminated])
        ),
        NSRunningApplication=object(),
    )
    monkeypatch.setattr(ax_driver, "AS", services)
    _install_module(
        monkeypatch,
        "Quartz",
        kCGWindowListOptionAll=1,
        kCGNullWindowID=0,
        CGWindowListCopyWindowInfo=lambda *_: [],
    )
    assert ax_driver._running_applications() == []


def test_ax_selector_ignores_unresolved_expected_pid(monkeypatch):
    monkeypatch.setattr(ax_driver, "AS", object())
    monkeypatch.setattr(ax_driver, "_application_for_pid", lambda pid: None)
    with pytest.raises(SystemExit, match="not found"):
        ax_driver._app_element("Target App", expected_pid=42)


def test_finder_unselected_inline_editor_cannot_bind_keyboard_activation(monkeypatch):
    """An unselected editor must never inherit the selected-row Enter exception."""
    snapshot = _finder_rename_snapshot()
    editor, outline = object(), object()
    monkeypatch.setattr(backend, "_focused_ax_element", lambda _app: outline)
    monkeypatch.setattr(
        backend.ax_driver,
        "_get",
        lambda element, attr: {
            editor: {
                "AXRole": "AXTextField",
                "AXSelected": False,
                "AXParent": outline,
            },
            outline: {"AXRole": "AXOutline"},
        }.get(element, {}).get(attr),
    )

    assert (
        backend._is_selected_finder_row_under_focused_outline(snapshot, editor) is False
    )


def test_finder_rename_commit_rejects_editor_drift_during_focus_restore(monkeypatch):
    """A focus change must not send Enter if the approved editor value changed."""
    snapshot = _finder_rename_snapshot()
    path_state = ["/tmp/Before"]
    live, reference = _install_selected_finder_editor(monkeypatch, snapshot, path_state)
    binding = {
        "pid": 4,
        "window_id": "cg:101",
        "file_reference": reference,
        "original_path": "/tmp/Before",
        "requested_basename": "After",
    }
    editor_value = {"value": "After"}
    monkeypatch.setattr(backend, "_finder_transaction_editor", lambda *a, **k: live)
    monkeypatch.setattr(
        backend, "_normalized_finder_editor_value", lambda _live: editor_value["value"]
    )

    def restore(_app, _snapshot):
        editor_value["value"] = "Different"
        return snapshot["window"]

    monkeypatch.setattr(backend, "raise_selected_window", restore)
    monkeypatch.setattr(
        backend.ax_driver,
        "_press_key",
        lambda _key: pytest.fail("drifted editor must not receive Enter"),
    )

    with pytest.raises(
        errors.ComputerUseError, match="changed during focus restoration"
    ):
        backend.commit_finder_rename("pid:4", snapshot, 0, binding)
    assert path_state == ["/tmp/Before"]
