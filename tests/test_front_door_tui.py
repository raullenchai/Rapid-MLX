"""Offline coverage for launcher rendering and input."""

import os
import pty
import termios

import pytest

from rapid_mlx import first_run, front_door
from rapid_mlx import front_door_tui as tui


def test_render_screens():
    assert "Press Enter" in tui.welcome()
    items = [("chat", "Chat in this terminal", "Talk here")]
    assert "▸ Chat" in tui.launcher("Rapid-MLX 1", "M3 · 16 GB", items, 0)
    assert "Server running" in tui.launcher(
        "Rapid-MLX 1", "M3", items, 0, "Server running"
    )
    assert all(
        len(line) <= 16
        for line in tui.launcher("Rapid-MLX 1", "M3", items, 0, width=16).splitlines()
    )
    assert "qwen" in tui.picker("Codex", [("qwen", "3 GB (not downloaded)")], 0, "q")
    assert "No matching models" in tui.picker("Codex", [], 0, "z")
    assert "▸ Yes" in tui.confirm("qwen")
    assert "▸ No" in tui.confirm("qwen", False)
    assert "50%" in tui.progress("model", 50, 100, 10)
    assert (
        f"rapid-mlx pull {front_door.QUICK_START_MODEL}"
        in front_door.render_non_tty("1", 0, [])
    )


def test_terminal_restore_on_exception(monkeypatch):
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    stream = os.fdopen(os.dup(slave), "r")
    monkeypatch.setattr(tui.sys, "stdin", stream)
    monkeypatch.setattr(
        tui.os, "read", lambda *args: (_ for _ in ()).throw(RuntimeError("failed"))
    )
    try:
        with pytest.raises(RuntimeError):
            tui.read_key()
        assert termios.tcgetattr(slave)[3] & (termios.ICANON | termios.ECHO) == before[
            3
        ] & (termios.ICANON | termios.ECHO)
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_launcher_launch_and_saved_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("rapid_mlx.run.cli._fits_host", lambda *args: True)
    monkeypatch.setattr(
        "rapid_mlx.model_auto_config.detect_model_config",
        lambda _: type("Config", (), {"tool_call_parser": "yes"})(),
    )
    state = front_door.FrontDoorState(
        "1",
        "M3",
        16,
        [{"alias": "qwen3.5-4b-4bit", "role": "smart", "cached": True}],
        cached=["qwen3.5-4b-4bit"],
        selected="qwen3.5-4b-4bit",
    )
    keys = iter(["enter", "enter", "enter"])
    plan = front_door.run_interactive(
        state, key_reader=lambda: next(keys), out=lambda _: None
    )
    assert plan.steps == [["launch", "claude-code", "--model", "qwen3.5-4b-4bit"]]
    assert first_run.front_door_preferences()["selection"] == "claude-code"


def _session(monkeypatch, keys, cached=True):
    monkeypatch.setattr(
        front_door, "picker_entries", lambda state, **kwargs: [("example", "Example")]
    )
    monkeypatch.setattr(first_run, "front_door_preferences", lambda: {"welcome": True})
    monkeypatch.setattr(first_run, "save_front_door_preferences", lambda _: None)
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("rapid_mlx.run.cli._fits_host", lambda *_: True)
    monkeypatch.setattr(
        "rapid_mlx.model_auto_config.detect_model_config", lambda _: None
    )
    state = front_door.FrontDoorState(
        "1",
        "M3",
        16,
        [
            {
                "alias": "example",
                "role": "smart",
                "cached": cached,
                "download_size_gb": 3,
            }
        ],
        cached=["example"] if cached else [],
        selected="example",
    )
    output = []
    iterator = iter(keys)
    plan = front_door.run_interactive(
        state,
        key_reader=lambda: next(iterator),
        out=output.append,
        free_port=lambda: 8001,
    )
    return plan, output


def test_more_server_and_help(monkeypatch):
    plan, output = _session(monkeypatch, ["up", "enter", "enter", "enter"])
    assert plan.steps == [["serve", "example", "--port", "8001"]]
    assert any("Start an API server" in line for line in output)
    plan, _ = _session(monkeypatch, ["up", "enter", "up", "up", "enter"])
    assert plan.steps == [["--help"]]


def test_other_tools_and_back(monkeypatch):
    plan, _ = _session(
        monkeypatch, ["up", "enter", "down", "enter", "down", "enter", "enter"]
    )
    assert plan.steps == [["launch", "aider", "--model", "example"]]
    plan, output = _session(
        monkeypatch, ["up", "enter", "down", "enter", "left", "left", "quit"]
    )
    assert plan is None
    assert sum("More..." in line for line in output) >= 2


def test_filter_configure_and_confirmation(monkeypatch):
    plan, output = _session(
        monkeypatch,
        [
            "right",
            "e",
            "backspace",
            "enter",
            "enter",
            "enter",
            "right",
            "enter",
            "quit",
        ],
        cached=False,
    )
    assert plan is None
    assert any("Download example?" in line for line in output)
    plan, _ = _session(monkeypatch, ["enter", "enter", "enter"], cached=False)
    assert plan.steps == [
        ["pull", "example"],
        ["launch", "claude-code", "--model", "example"],
    ]


def test_scroll_and_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    items = [(str(i), f"Item {i}", "description") for i in range(10)]
    text = tui.launcher("Title", "hardware", items, 9, width=30, height=8, color=True)
    assert "Item 9" in text and "Item 0" not in text
    assert "\x1b[1m" in text and "\x1b[2m" in text
    assert len(text.splitlines()) <= 8


def test_per_file_progress_hook(monkeypatch):
    import io

    from rapid_mlx import _mirror

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    monkeypatch.setattr(_mirror.sys, "stdout", stream)
    monkeypatch.setenv("RAPID_MLX_FRONT_DOOR_PULL", "1")
    tracker = _mirror._ProgressTracker(total=100)
    tracker.file_progress("weights.safetensors", 50, 100, _mirror.time.monotonic() - 1)
    assert "pulling weights.safetensors:" in stream.getvalue()
    assert "50%" in stream.getvalue()
    cls = _mirror._tracking_hf_tqdm_class(
        suppress_progress=True, filename="hf.bin", tracker=tracker
    )
    bar = cls(total=0, unit="B")
    bar.total = 100
    bar.update(25)
    bar.close()
    assert "pulling hf.bin:" in stream.getvalue()


def test_hf_file_count_is_not_rendered(monkeypatch):
    from rapid_mlx import _mirror

    calls = []
    tracker = type(
        "Tracker", (), {"file_progress": lambda self, *args: calls.append(args)}
    )()
    cls = _mirror._tracking_hf_tqdm_class(suppress_progress=True, tracker=tracker)
    bar = cls(total=3, unit="files")
    bar.update(1)
    bar.close()
    assert not calls


def test_list_spacing_and_styles(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    text = tui.launcher(
        "Title",
        "Mac",
        [("a", "First", "Detail"), ("b", "Second", "Other")],
        0,
        height=30,
        color=True,
    )
    assert (
        "\x1b[1;97m▸ First\x1b[0m\n\x1b[2m    Detail\x1b[0m\n\n\x1b[1m  Second" in text
    )
    assert "Other\x1b[0m\n\n\x1b[2m↑" in text


def test_picker_excludes_unknown_size(monkeypatch):
    from rapid_mlx.model_aliases import ModelProfile

    monkeypatch.setattr(
        "rapid_mlx.model_aliases.list_profiles",
        lambda: {
            "unknown-9b-4bit": ModelProfile(hf_path="example/unknown"),
        },
    )
    monkeypatch.setattr("rapid_mlx.model_sizes.size_bytes", lambda _: None)
    state = front_door.FrontDoorState("1", "M4", 48, [])
    assert front_door.picker_entries(state) == []


def test_picker_excludes_draft_only_checkpoint():
    state = front_door.FrontDoorState(
        "1",
        "M4",
        48,
        [{"alias": "qwen3.5-4b-4bit", "role": "smart"}],
        cached=["qwen3.6-35b-mtp-4bit"],
    )
    assert "qwen3.6-35b-mtp-4bit" not in dict(front_door.picker_entries(state))
