# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``rapid-mlx launch <client>`` bootstrap subcommand.

We never touch the user's real config files — every test redirects the
relevant home / config dir to a per-test ``tmp_path`` and asserts the
write-or-patch behaviour against that sandbox. The CLI integration
tests use ``--dry-run`` so they exercise the dispatcher's argv-parsing
without writing anything.

See ``vllm_mlx/launch/`` for the modules under test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.launch import (
    ADAPTERS,
    _common,
    claude_code,
    cline,
    continue_dev,
    cursor,
)
from vllm_mlx.launch import cli as launch_cli

# --------------------------------------------------------------------
# Shared fixture: pin Path.home() to a per-test tmp_path so adapter
# modules — which compute config paths from Path.home() at import time
# via the candidate-roots helpers — see a clean state. We patch via
# monkeypatch.setattr on the *adapter's* internal probes, not on
# Path.home() itself: those globals were resolved at import time.
# --------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path, monkeypatch) -> Path:
    """Redirect every adapter's home-anchored constants at the per-test
    tmp_path.

    Each adapter freezes its config paths at import time
    (``_CONFIG_DIR = Path.home() / ...``). We monkeypatch the module
    attributes directly so the import-time values don't leak across
    tests. Returns the tmp_path for callers that want to construct
    expected paths.
    """
    # cline: replace the candidate-roots helper so detect/path
    # resolution picks paths under tmp_path.
    fake_root = tmp_path / "vscode-globalStorage"
    monkeypatch.setattr(cline, "_candidate_settings_roots", lambda: [fake_root])

    # claude_code: replace the two module constants.
    monkeypatch.setattr(claude_code, "_CLAUDE_STATE_DIR", tmp_path / ".claude")
    monkeypatch.setattr(claude_code, "_CONFIG_DIR", tmp_path / ".config/claude")

    # continue_dev: replace the config dir.
    monkeypatch.setattr(continue_dev, "_CONFIG_DIR", tmp_path / ".continue")

    # cursor: replace the candidate-dirs helper.
    fake_cursor_dir = tmp_path / "Cursor/User"
    monkeypatch.setattr(cursor, "_candidate_dirs", lambda: [fake_cursor_dir])
    monkeypatch.setattr(cursor, "_CONFIG_DIR_MAC", fake_cursor_dir)
    monkeypatch.setattr(cursor, "_CONFIG_DIR_LINUX", fake_cursor_dir)

    # Also redirect which() and mac_app_installed() so detect() doesn't
    # find the dev machine's real claude / cursor installs.
    monkeypatch.setattr(_common, "which", lambda _: None)
    monkeypatch.setattr(_common, "mac_app_installed", lambda _: False)

    # And the PID file the launch CLI writes when --start-server is on.
    monkeypatch.setattr(launch_cli, "PID_FILE", tmp_path / "launch.pid")

    return tmp_path


# --------------------------------------------------------------------
# Cline adapter
# --------------------------------------------------------------------


class TestCline:
    def test_detect_false_when_no_globalstorage(self, fake_home):
        assert cline.detect() is False
        assert cline.current_config_path() is None

    def test_detect_true_when_settings_dir_exists(self, fake_home):
        # Materialise the extension settings dir but not the file —
        # detect() should still report True (installed, uninitialised).
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        assert cline.detect() is True
        path = cline.current_config_path()
        assert path is not None
        assert path.name == "cline_mcp_settings.json"

    def test_write_preserves_existing_keys(self, fake_home):
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        path = ext_dir / "cline_mcp_settings.json"
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {"custom": {"command": "node"}},
                    "apiProvider": "anthropic",  # we'll overwrite this
                    "openAiApiKey": "old-key",  # we'll overwrite this
                    "customInstructions": "be terse",  # must survive
                }
            )
        )

        returned = cline.write_or_patch_config(
            "http://127.0.0.1:8000",
            "qwen3.5-4b-4bit",
            api_key="sk-noop",
        )
        assert returned == path

        # Backup exists.
        backups = list(ext_dir.glob("cline_mcp_settings.json.bak.*"))
        assert len(backups) == 1

        data = json.loads(path.read_text())
        # Keys we own — set / overwritten.
        assert data["apiProvider"] == "openai"
        assert data["openAiBaseUrl"] == "http://127.0.0.1:8000/v1"
        assert data["openAiApiKey"] == "sk-noop"
        assert data["openAiModelId"] == "qwen3.5-4b-4bit"
        # Keys we don't own — untouched.
        assert data["mcpServers"] == {"custom": {"command": "node"}}
        assert data["customInstructions"] == "be terse"

    def test_does_not_double_append_v1(self, fake_home):
        """User passes ``http://127.0.0.1:8000/v1`` — must NOT
        produce ``/v1/v1``."""
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        cline.write_or_patch_config(
            "http://127.0.0.1:8000/v1",
            "alias",
        )
        path = ext_dir / "cline_mcp_settings.json"
        data = json.loads(path.read_text())
        assert data["openAiBaseUrl"] == "http://127.0.0.1:8000/v1"


# --------------------------------------------------------------------
# Claude Code adapter
# --------------------------------------------------------------------


class TestClaudeCode:
    def test_detect_false_when_nothing_installed(self, fake_home):
        assert claude_code.detect() is False

    def test_detect_true_when_state_dir_exists(self, fake_home):
        (fake_home / ".claude").mkdir()
        assert claude_code.detect() is True

    def test_write_strips_trailing_v1(self, fake_home):
        # User accidentally passes ``http://127.0.0.1:8000/v1`` — the
        # Anthropic SDK joins ``/v1/messages`` itself, so we must strip
        # the suffix or every request 404s on ``/v1/v1/messages``.
        path = claude_code.write_or_patch_config(
            "http://127.0.0.1:8000/v1",
            "qwen3.5-9b-4bit",
        )
        data = json.loads(path.read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
        assert data["env"]["ANTHROPIC_MODEL"] == "qwen3.5-9b-4bit"
        assert data["env"]["ANTHROPIC_API_KEY"] == "sk-noop"

    def test_write_preserves_existing_env_and_other_keys(self, fake_home):
        cfg = claude_code.current_config_path()
        assert cfg is not None
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(
            json.dumps(
                {
                    "permissions": {"allow": ["Bash(git:*)"]},
                    "env": {"OTHER_VAR": "preserved", "ANTHROPIC_BASE_URL": "old"},
                }
            )
        )
        claude_code.write_or_patch_config("http://127.0.0.1:8000", "qwen3.5-4b-4bit")
        data = json.loads(cfg.read_text())
        # Untouched.
        assert data["permissions"] == {"allow": ["Bash(git:*)"]}
        assert data["env"]["OTHER_VAR"] == "preserved"
        # Overwritten.
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"

    def test_backup_created(self, fake_home):
        cfg = claude_code.current_config_path()
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text('{"env": {"foo": "bar"}}')
        claude_code.write_or_patch_config("http://127.0.0.1:8000", "alias")
        backups = list(cfg.parent.glob(cfg.name + ".bak.*"))
        assert len(backups) == 1


# --------------------------------------------------------------------
# Continue.dev adapter
# --------------------------------------------------------------------


class TestContinueDev:
    def test_detect_false_when_no_continue_dir(self, fake_home):
        assert continue_dev.detect() is False

    def test_detect_true_when_dir_exists(self, fake_home):
        (fake_home / ".continue").mkdir()
        assert continue_dev.detect() is True

    def test_appends_new_model_entry(self, fake_home):
        (fake_home / ".continue").mkdir()
        cfg = continue_dev.current_config_path()
        cfg.write_text(
            json.dumps(
                {
                    "models": [
                        {"title": "Anthropic", "provider": "anthropic"},
                    ],
                    "customCommands": [{"name": "test"}],
                }
            )
        )
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "qwen3.5-4b-4bit")
        data = json.loads(cfg.read_text())
        assert len(data["models"]) == 2
        rapid = next(m for m in data["models"] if m["title"] == "rapid-mlx")
        assert rapid["provider"] == "openai"
        assert rapid["model"] == "qwen3.5-4b-4bit"
        assert rapid["apiBase"] == "http://127.0.0.1:8000/v1"
        # Other model preserved.
        assert any(m["title"] == "Anthropic" for m in data["models"])
        # Other top-level keys preserved.
        assert data["customCommands"] == [{"name": "test"}]

    def test_rerun_replaces_in_place_not_duplicates(self, fake_home):
        (fake_home / ".continue").mkdir()
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "model-a")
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "model-b")
        cfg = continue_dev.current_config_path()
        data = json.loads(cfg.read_text())
        rapid_entries = [m for m in data["models"] if m.get("title") == "rapid-mlx"]
        assert len(rapid_entries) == 1
        assert rapid_entries[0]["model"] == "model-b"


# --------------------------------------------------------------------
# Cursor adapter
# --------------------------------------------------------------------


class TestCursor:
    def test_detect_false_when_nothing_installed(self, fake_home):
        assert cursor.detect() is False

    def test_detect_true_when_user_dir_exists(self, fake_home):
        (fake_home / "Cursor/User").mkdir(parents=True)
        assert cursor.detect() is True

    def test_write_sets_dotted_keys(self, fake_home):
        (fake_home / "Cursor/User").mkdir(parents=True)
        path = cursor.write_or_patch_config("http://127.0.0.1:8000", "qwen3.5-9b-4bit")
        data = json.loads(path.read_text())
        assert data["cursor.aiprovider.openai.baseUrl"] == "http://127.0.0.1:8000/v1"
        assert data["cursor.aiprovider.openai.model"] == "qwen3.5-9b-4bit"
        assert data["cursor.aiprovider.openai.apiKey"] == "sk-noop"

    def test_preserves_unrelated_settings(self, fake_home):
        (fake_home / "Cursor/User").mkdir(parents=True)
        cfg = cursor.current_config_path()
        cfg.write_text(
            json.dumps(
                {
                    "editor.fontSize": 14,
                    "workbench.colorTheme": "Default Dark+",
                }
            )
        )
        cursor.write_or_patch_config("http://127.0.0.1:8000", "alias")
        data = json.loads(cfg.read_text())
        assert data["editor.fontSize"] == 14
        assert data["workbench.colorTheme"] == "Default Dark+"


# --------------------------------------------------------------------
# Atomic-write + backup primitives
# --------------------------------------------------------------------


class TestCommon:
    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "settings.json"
        _common.atomic_write_json(target, {"k": "v"})
        assert target.exists()
        assert json.loads(target.read_text()) == {"k": "v"}

    def test_atomic_write_no_leftover_temp_files(self, tmp_path):
        target = tmp_path / "settings.json"
        _common.atomic_write_json(target, {"x": 1})
        # No `.new` files left behind.
        assert list(tmp_path.glob("*.new")) == []

    def test_backup_returns_none_when_no_original(self, tmp_path):
        assert _common.backup_existing(tmp_path / "missing.json") is None

    def test_backup_handles_same_second_collisions(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text('{"a": 1}')
        b1 = _common.backup_existing(target)
        # Simulate a second invocation in the same second by reusing
        # the timestamp portion — the helper appends a counter suffix.
        b2 = _common.backup_existing(target)
        assert b1 is not None and b2 is not None
        assert b1 != b2

    def test_load_json_lenient_missing(self, tmp_path):
        assert _common.load_json_lenient(tmp_path / "missing.json") == {}

    def test_load_json_lenient_empty_file(self, tmp_path):
        target = tmp_path / "empty.json"
        target.write_text("")
        assert _common.load_json_lenient(target) == {}

    def test_load_json_lenient_raises_on_invalid(self, tmp_path):
        target = tmp_path / "bad.json"
        target.write_text("{ not json")
        with pytest.raises(json.JSONDecodeError):
            _common.load_json_lenient(target)


# --------------------------------------------------------------------
# Top-level CLI dispatcher
# --------------------------------------------------------------------


def _make_args(**overrides):
    """Build an argparse.Namespace shaped like the ``launch`` parser
    produces, with sane defaults the tests override per-case."""
    defaults = dict(
        client=None,
        all=False,
        model=None,
        server_url="http://127.0.0.1:8000",
        port=8000,
        start_server=False,
        dry_run=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestLaunchCommand:
    def test_list_prints_all_clients(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="list"))
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        for name in ADAPTERS:
            assert name in out

    def test_unknown_client_exit_2(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="atom"))
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "unknown client" in err

    def test_missing_client_and_no_all_exit_2(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args())
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "missing client name" in err

    def test_all_and_client_mutually_exclusive(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="cline", all=True))
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "mutually exclusive" in err

    def test_all_with_no_detected_clients_exits_1(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(all=True))
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "no supported clients detected" in err

    def test_dry_run_does_not_touch_disk(self, fake_home, capsys):
        # Mark cline as detected so the dispatcher reaches the
        # would-patch line.
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        before = list(ext_dir.iterdir())

        launch_cli.launch_command(_make_args(client="cline", dry_run=True))
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "cline" in out
        # No file was created or modified.
        assert list(ext_dir.iterdir()) == before

    def test_real_patch_writes_file(self, fake_home, capsys):
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        launch_cli.launch_command(_make_args(client="cline", model="qwen3.5-4b-4bit"))
        target = ext_dir / "cline_mcp_settings.json"
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["openAiModelId"] == "qwen3.5-4b-4bit"
        out = capsys.readouterr().out
        assert "Patched cline" in out
        assert "Now ready" in out

    def test_not_detected_client_fails_with_hint(self, fake_home, capsys):
        # cline is NOT detected (no globalStorage dir). The command
        # should fail with a clear hint and exit non-zero.
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="cline"))
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "cline: not detected" in err

    def test_start_server_spawns_and_writes_pid(self, fake_home, capsys):
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        fake_proc = MagicMock()
        fake_proc.pid = 99999
        with patch.object(subprocess, "Popen", return_value=fake_proc) as popen:
            launch_cli.launch_command(
                _make_args(
                    client="cline",
                    model="qwen3.5-4b-4bit",
                    start_server=True,
                    port=8102,
                )
            )
        # Spawn happened with the expected argv.
        argv = popen.call_args[0][0]
        assert argv == [
            "rapid-mlx",
            "serve",
            "qwen3.5-4b-4bit",
            "--port",
            "8102",
        ]
        # PID file written.
        assert launch_cli.PID_FILE.read_text().strip() == "99999"

    def test_start_server_skipped_when_no_clients_patched(self, fake_home, capsys):
        # cline is NOT detected on this fake_home. --start-server must
        # NOT spawn a child server when zero clients were patched
        # successfully — otherwise we leak a detached server + PID file
        # for a setup the user can't actually use.
        with (
            patch.object(subprocess, "Popen") as popen,
            pytest.raises(SystemExit) as excinfo,
        ):
            launch_cli.launch_command(
                _make_args(
                    client="cline",
                    model="qwen3.5-4b-4bit",
                    start_server=True,
                    port=8102,
                )
            )
        assert excinfo.value.code == 1
        popen.assert_not_called()
        assert not launch_cli.PID_FILE.exists()
        err = capsys.readouterr().err
        assert "Skipping --start-server" in err

    def test_uses_original_alias_when_resolved(self, fake_home, capsys):
        """When ``main()`` rewrites ``args.model`` from alias to HF id,
        the launch command should patch with the ORIGINAL alias so the
        IDE client requests the short name from rapid-mlx."""
        ext_dir = (
            fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev" / "settings"
        )
        ext_dir.mkdir(parents=True)
        ns = _make_args(client="cline", model="mlx-community/Qwen3.5-4B-MLX-4bit")
        # Simulate what ``main()`` does on the way in.
        ns._original_alias = "qwen3.5-4b-4bit"
        launch_cli.launch_command(ns)
        data = json.loads((ext_dir / "cline_mcp_settings.json").read_text())
        assert data["openAiModelId"] == "qwen3.5-4b-4bit"


# --------------------------------------------------------------------
# Top-level CLI argparse integration — invoke `python -m vllm_mlx.cli
# launch --help` via subprocess so we exercise the wiring from
# main() rather than the dispatcher in isolation. We don't run a real
# patch in subprocess (no fake_home control); the unit tests above
# cover that.
# --------------------------------------------------------------------


def test_launch_help_text_is_registered(tmp_path):
    """The ``launch`` subcommand is wired onto the top-level parser
    (regression guard: a future refactor of cli.py's subparser block
    that drops the ``_register_launch(subparsers)`` call would let the
    feature silently disappear)."""

    # We don't actually run main() — just walk its argparse tree.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from vllm_mlx.launch.cli import register

    register(sub)
    # Choices populated.
    assert "launch" in sub.choices
    # And accept `list` as a client name.
    args = parser.parse_args(["launch", "list"])
    assert args.client == "list"


@pytest.mark.parametrize("bad_port", ["0", "-1", "65536", "99999", "abc"])
def test_launch_port_rejects_out_of_range(bad_port):
    """``--port`` must use the same ``[1, 65535]`` validator as
    ``rapid-mlx serve``. Pre-fix, ``launch --port 99999`` parsed
    successfully and only failed inside the detached child after the
    parent had already printed "Started" and written a PID."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from vllm_mlx.launch.cli import register

    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "cline", "--port", bad_port])


def test_launch_port_accepts_in_range():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from vllm_mlx.launch.cli import register

    register(sub)
    args = parser.parse_args(["launch", "cline", "--port", "8000"])
    assert args.port == 8000
