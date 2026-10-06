# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``rapid-mlx launch <client>`` bootstrap subcommand.

We never touch the user's real config files — every test redirects the
relevant home / config dir to a per-test ``tmp_path`` and asserts the
write-or-patch behaviour against that sandbox. The CLI integration
tests use ``--dry-run`` so they exercise the dispatcher's argv-parsing
without writing anything.

See ``rapid_mlx/launch/`` for the modules under test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rapid_mlx.launch import (
    ADAPTERS,
    _common,
    claude_code,
    cline,
    continue_dev,
    cursor,
)
from rapid_mlx.launch import cli as launch_cli

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
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    # The clients' own relocation variables must not leak in from the host.
    for name in ("CLINE_DIR", "CLINE_DATA_DIR", "CONTINUE_GLOBAL_DIR"):
        monkeypatch.delenv(name, raising=False)

    # cline: Cline's data dir and the VS Code globalStorage roots both
    # resolve under tmp_path.
    fake_root = tmp_path / "vscode-globalStorage"
    monkeypatch.setattr(cline, "_candidate_settings_roots", lambda: [fake_root])
    monkeypatch.setattr(cline, "_cline_data_dir", lambda: tmp_path / ".cline" / "data")

    # claude_code: replace the two module constants.
    monkeypatch.setattr(claude_code, "_CLAUDE_STATE_DIR", tmp_path / ".claude")
    monkeypatch.setattr(claude_code, "_CONFIG_DIR", tmp_path / ".claude")

    # continue_dev: replace the config dir.
    monkeypatch.setattr(continue_dev, "_CONFIG_DIR", tmp_path / ".continue")

    fake_cursor_dir = tmp_path / "Cursor/User"
    monkeypatch.setattr(cursor, "_candidate_dirs", lambda: [fake_cursor_dir])
    monkeypatch.setattr(cursor, "_CONFIG_DIR_MAC", fake_cursor_dir)
    monkeypatch.setattr(cursor, "_CONFIG_DIR_LINUX", fake_cursor_dir)

    # Also redirect which() and mac_app_installed() so detect() doesn't
    # find the dev machine's real client installs.
    monkeypatch.setattr(_common, "which", lambda _: None)
    monkeypatch.setattr(_common, "mac_app_installed", lambda _: False)

    # And the PID file the launch CLI writes when --start-server is on.
    monkeypatch.setattr(launch_cli, "PID_FILE", tmp_path / "launch.pid")

    return tmp_path


def _install_cline(home: Path) -> Path:
    """Materialise Cline's settings dir (what a first Cline run creates)."""
    settings = home / ".cline" / "data" / "settings"
    settings.mkdir(parents=True)
    return settings


def _cline_settings(data: dict) -> dict:
    assert data["lastUsedProvider"] == "openai-compatible"
    return data["providers"]["openai-compatible"]["settings"]


# --------------------------------------------------------------------
# Cline adapter
# --------------------------------------------------------------------


FIXTURES = Path(__file__).parent / "fixtures" / "agent_configs"


class TestCline:
    def test_detect_false_when_nothing_installed(self, fake_home):
        assert cline.detect() is False
        # The path is well-defined even before Cline ever ran.
        assert cline.current_config_path() == (
            fake_home / ".cline/data/settings/providers.json"
        )

    def test_detect_true_for_cli_on_path(self, fake_home, monkeypatch):
        monkeypatch.setattr(
            _common,
            "which",
            lambda cmd: "/usr/local/bin/cline" if cmd == "cline" else None,
        )
        assert cline.detect() is True

    def test_detect_true_for_cline_data_dir(self, fake_home):
        (fake_home / ".cline" / "data").mkdir(parents=True)
        assert cline.detect() is True

    def test_detect_true_for_vscode_extension(self, fake_home):
        (fake_home / "vscode-globalStorage" / "saoudrizwan.claude-dev").mkdir(
            parents=True
        )
        assert cline.detect() is True

    def test_data_dir_follows_cline_environment(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLINE_DATA_DIR", raising=False)
        monkeypatch.setenv("CLINE_DIR", str(tmp_path / "cline-home"))
        assert cline._cline_data_dir() == tmp_path / "cline-home" / "data"
        monkeypatch.setenv("CLINE_DATA_DIR", str(tmp_path / "explicit"))
        assert cline._cline_data_dir() == tmp_path / "explicit"
        monkeypatch.delenv("CLINE_DATA_DIR")
        monkeypatch.delenv("CLINE_DIR")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert cline._cline_data_dir() == tmp_path / "home" / ".cline" / "data"

    def test_new_file_matches_what_cline_auth_writes(self, fake_home, monkeypatch):
        """Golden: byte-for-byte what ``cline auth -p openai -b … -k sk-noop
        -m qwen3.5-4b-4bit`` (Cline CLI 3.0.68) wrote into a fresh HOME."""
        monkeypatch.setattr(cline, "_now_iso", lambda: "2026-10-06T18:08:54.038Z")
        path = cline.write_or_patch_config(
            "http://127.0.0.1:8123", "qwen3.5-4b-4bit", api_key="sk-noop"
        )
        assert path == fake_home / ".cline/data/settings/providers.json"
        golden = FIXTURES / "cline_auth_3.0.68_providers.json"
        assert path.read_text() == golden.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
        assert list(path.parent.glob("*.bak.*")) == []

    def test_preserves_other_providers_and_backs_up(self, fake_home):
        settings_dir = _install_cline(fake_home)
        path = settings_dir / "providers.json"
        existing = {
            "version": 1,
            "lastUsedProvider": "anthropic",
            "modes": {"voiceInput": {"enabled": True}},
            "providers": {
                "anthropic": {
                    "settings": {"provider": "anthropic", "apiKey": "sk-ant-x"},
                    "updatedAt": "2026-01-01T00:00:00.000Z",
                    "tokenSource": "manual",
                },
                "openai-compatible": {
                    "settings": {
                        "provider": "openai-compatible",
                        "apiKey": "old",
                        "model": "old-model",
                        "baseUrl": "https://example.invalid/v1",
                        "headers": {"X-Team": "a"},
                    },
                    "updatedAt": "2026-01-01T00:00:00.000Z",
                    "tokenSource": "migration",
                },
            },
        }
        path.write_text(json.dumps(existing))

        cline.write_or_patch_config("http://127.0.0.1:8000/v1", "qwen3.5-4b-4bit")

        data = json.loads(path.read_text())
        assert data["providers"]["anthropic"] == existing["providers"]["anthropic"]
        assert data["modes"] == existing["modes"]
        entry = data["providers"]["openai-compatible"]
        assert entry["settings"] == {
            "provider": "openai-compatible",
            "apiKey": "sk-noop",
            "model": "qwen3.5-4b-4bit",
            "baseUrl": "http://127.0.0.1:8000/v1",
            "headers": {"X-Team": "a"},
        }
        assert entry["tokenSource"] == "migration"
        assert entry["updatedAt"].endswith("Z")
        assert data["lastUsedProvider"] == "openai-compatible"
        backups = list(settings_dir.glob("providers.json.bak.*"))
        assert len(backups) == 1
        assert json.loads(backups[0].read_text()) == existing

    def test_rerun_is_a_no_op(self, fake_home):
        path = cline.write_or_patch_config("http://127.0.0.1:8000", "m")
        before = path.read_bytes()
        cline.write_or_patch_config("http://127.0.0.1:8000/", "m")
        assert path.read_bytes() == before
        assert list(path.parent.glob("*.bak.*")) == []

    @pytest.mark.parametrize(
        "content",
        [
            "not json",
            "[]",
            '{"version": 2, "providers": {}}',
            '{"version": 1, "providers": []}',
        ],
    )
    def test_refuses_to_rewrite_a_file_cline_would_reject(self, fake_home, content):
        settings_dir = _install_cline(fake_home)
        path = settings_dir / "providers.json"
        path.write_text(content)
        with pytest.raises(ValueError):
            cline.write_or_patch_config("http://127.0.0.1:8000", "m")
        assert path.read_text() == content
        assert list(settings_dir.glob("*.bak.*")) == []

    def test_never_touches_mcp_settings(self, fake_home):
        mcp = (
            fake_home
            / "vscode-globalStorage/saoudrizwan.claude-dev/settings"
            / "cline_mcp_settings.json"
        )
        mcp.parent.mkdir(parents=True)
        mcp.write_text('{"mcpServers": {}}')
        cline.write_or_patch_config("http://127.0.0.1:8000", "m")
        assert mcp.read_text() == '{"mcpServers": {}}'
        assert list(mcp.parent.iterdir()) == [mcp]

    def test_preview_redacts_keys_and_writes_nothing(self, fake_home):
        settings_dir = _install_cline(fake_home)
        path = settings_dir / "providers.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {
                        "anthropic": {
                            "settings": {"provider": "anthropic", "apiKey": "real"},
                            "updatedAt": "2026-01-01T00:00:00.000Z",
                            "tokenSource": "manual",
                        }
                    },
                }
            )
        )
        before = path.read_text()
        shown, diff, notes = cline.preview("http://127.0.0.1:8000", "m", "secret")
        assert shown == path
        assert notes == ()
        assert "real" not in diff and "secret" not in diff
        assert '+  "lastUsedProvider": "openai-compatible"' in diff
        assert path.read_text() == before
        cline.write_or_patch_config("http://127.0.0.1:8000", "m", api_key="secret")
        assert cline.preview("http://127.0.0.1:8000", "m", "secret")[1] == ""

    def test_post_setup_notes_give_exact_extension_steps(self):
        lines = cline.post_setup_notes("http://127.0.0.1:8000", "m", None)
        text = "\n".join(lines)
        assert "API Provider: OpenAI Compatible" in text
        assert "Base URL:     http://127.0.0.1:8000/v1" in text
        assert "Model ID:     m" in text
        assert "sk-noop" in text
        keyed = "\n".join(cline.post_setup_notes("http://h/v1", "m", "s3cret"))
        assert "s3cret" not in keyed
        assert "RAPID_MLX_API_KEY" in keyed


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

    def test_launch_writes_live_claude_context(self, fake_home, monkeypatch):
        from rapid_mlx.agents import adapter

        (fake_home / ".claude").mkdir()
        monkeypatch.setattr(adapter, "fetch_context_window", lambda *_args: 131072)
        monkeypatch.setattr(
            "rapid_mlx.run.cli._cached_context_window", lambda _model: None
        )

        launch_cli.launch_command(_make_args(client="claude-code", model="local-model"))

        data = json.loads(claude_code.current_config_path().read_text())
        assert data["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"

    def test_write_preserves_existing_env_and_other_keys(self, fake_home):
        cfg = claude_code.current_config_path()
        assert cfg is not None
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(
            json.dumps(
                {
                    "permissions": {"allow": ["Bash(git:*)"]},
                    "env": {
                        "OTHER_VAR": "preserved",
                        "ANTHROPIC_BASE_URL": "old",
                        "ANTHROPIC_AUTH_TOKEN": "proxy-token",
                    },
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
        assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == ""

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

    def test_targets_config_yaml(self, fake_home):
        assert continue_dev.current_config_path() == fake_home / ".continue/config.yaml"

    def test_continue_global_dir_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CONTINUE_GLOBAL_DIR", str(tmp_path / "cfg"))
        assert continue_dev.current_config_path() == tmp_path / "cfg" / "config.yaml"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CONTINUE_GLOBAL_DIR", "rel")
        assert continue_dev.current_config_path() == tmp_path / "rel" / "config.yaml"

    def test_new_config_golden(self, fake_home):
        (fake_home / ".continue").mkdir()
        path = continue_dev.write_or_patch_config(
            "http://127.0.0.1:8000", "qwen3.5-4b-4bit"
        )
        golden = FIXTURES / "continue_new_config.yaml"
        assert path.read_text() == golden.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
        assert not (fake_home / ".continue/config.json").exists()

    def test_converter_matches_continue_upstream(self):
        """Golden: Continue's own ``convertJsonToYamlConfig``
        (@continuedev/config-yaml 1.42.0) on the same input. We differ only
        by the v1 header Continue writes for new files and by dropping our
        own legacy entry (it is replaced, not duplicated)."""
        legacy = json.loads((FIXTURES / "continue_legacy_config.json").read_text())
        upstream = json.loads(
            (FIXTURES / "continue_legacy_converted_upstream.json").read_text()
        )
        upstream.update(name="Local Config", version="1.0.0", schema="v1")
        upstream["models"] = [m for m in upstream["models"] if m["name"] != "rapid-mlx"]
        assert continue_dev.convert_legacy_config(legacy) == upstream

    def test_migrates_legacy_json_without_touching_it(self, fake_home, capsys):
        cont = fake_home / ".continue"
        cont.mkdir()
        legacy = cont / "config.json"
        legacy_bytes = (FIXTURES / "continue_legacy_config.json").read_bytes()
        legacy.write_bytes(legacy_bytes)

        path = continue_dev.write_or_patch_config(
            "http://127.0.0.1:8000", "qwen3.5-4b-4bit"
        )

        assert path == cont / "config.yaml"
        golden = FIXTURES / "continue_migrated_config.yaml"
        assert path.read_text() == golden.read_text()
        assert legacy.read_bytes() == legacy_bytes
        assert list(cont.glob("*.bak.*")) == []
        assert "config.json is left unchanged" in capsys.readouterr().err

    def test_patches_existing_yaml_in_place(self, fake_home):
        import yaml

        cont = fake_home / ".continue"
        cont.mkdir()
        path = cont / "config.yaml"
        path.write_text(
            "name: Mine\nversion: 0.1.0\nschema: v1\n"
            "models:\n"
            "- uses: anthropic/claude-sonnet\n"
            "- name: rapid-mlx\n  provider: openai\n  model: old\n"
            "  apiBase: http://127.0.0.1:9000/v1\n  apiKey: sk-noop\n"
            "  defaultCompletionOptions:\n    temperature: 0.1\n"
            "rules:\n- keep it short\n"
        )
        # A legacy json beside an existing yaml is ignored by Continue: it
        # must not be migrated again.
        (cont / "config.json").write_text('{"models": [{"title": "Old"}]}')

        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "new")

        data = yaml.safe_load(path.read_text())
        assert data["name"] == "Mine" and data["version"] == "0.1.0"
        assert data["rules"] == ["keep it short"]
        assert data["models"][0] == {"uses": "anthropic/claude-sonnet"}
        assert data["models"][1] == {
            "name": "rapid-mlx",
            "provider": "openai",
            "model": "new",
            "apiBase": "http://127.0.0.1:8000/v1",
            "apiKey": "sk-noop",
            "defaultCompletionOptions": {"temperature": 0.1},
        }
        assert len(data["models"]) == 2
        assert len(list(cont.glob("config.yaml.bak.*"))) == 1

    def test_rerun_is_a_no_op(self, fake_home):
        (fake_home / ".continue").mkdir()
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "model-a")
        path = continue_dev.current_config_path()
        before = path.read_bytes()
        continue_dev.write_or_patch_config("http://127.0.0.1:8000/v1", "model-a")
        assert path.read_bytes() == before
        assert list(path.parent.glob("*.bak.*")) == []

    def test_rerun_replaces_in_place_not_duplicates(self, fake_home):
        import yaml

        (fake_home / ".continue").mkdir()
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "model-a")
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "model-b")
        data = yaml.safe_load(continue_dev.current_config_path().read_text())
        rapid = [m for m in data["models"] if m.get("name") == "rapid-mlx"]
        assert [m["model"] for m in rapid] == ["model-b"]

    @pytest.mark.parametrize(
        ("name", "content"),
        [
            ("config.yaml", "models: [unclosed\n"),
            ("config.yaml", "- a list\n"),
            ("config.json", "{not json"),
            ("config.json", "[1]"),
        ],
    )
    def test_refuses_unparsable_configs(self, fake_home, name, content):
        cont = fake_home / ".continue"
        cont.mkdir()
        (cont / name).write_text(content)
        with pytest.raises(ValueError):
            continue_dev.write_or_patch_config("http://127.0.0.1:8000", "m")
        assert sorted(p.name for p in cont.iterdir()) == [name]

    def test_blank_yaml_is_treated_as_new(self, fake_home):
        cont = fake_home / ".continue"
        cont.mkdir()
        (cont / "config.yaml").write_text("\n")
        path = continue_dev.write_or_patch_config("http://127.0.0.1:8000", "m")
        assert "name: rapid-mlx" in path.read_text()

    def test_preview_redacts_migrated_keys(self, fake_home):
        cont = fake_home / ".continue"
        cont.mkdir()
        (cont / "config.json").write_bytes(
            (FIXTURES / "continue_legacy_config.json").read_bytes()
        )
        path, diff, notes = continue_dev.preview(
            "http://127.0.0.1:8000", "qwen3.5-4b-4bit"
        )
        assert path == cont / "config.yaml"
        assert not path.exists()
        for secret in ("sk-ant-example", "voyage-example", "jira-example"):
            assert secret not in diff
        assert "+  apiKey: sk-noop" in diff
        assert len(notes) == 1 and "config.json" in notes[0]


# --------------------------------------------------------------------
# Cursor adapter (public HTTPS endpoints only)
# --------------------------------------------------------------------


class TestCursor:
    def test_detect_false_when_nothing_installed(self, fake_home):
        assert cursor.detect() is False

    def test_detect_true_when_user_dir_exists(self, fake_home):
        (fake_home / "Cursor/User").mkdir(parents=True)
        assert cursor.detect() is True

    def test_write_sets_dotted_keys(self, fake_home):
        (fake_home / "Cursor/User").mkdir(parents=True)
        path = cursor.write_or_patch_config(
            "https://rapid.example.com",
            "qwen3.5-9b-4bit",
            api_key="cursor-secret",
        )
        data = json.loads(path.read_text())
        assert data["cursor.aiprovider.openai.baseUrl"] == (
            "https://rapid.example.com/v1"
        )
        assert data["cursor.aiprovider.openai.model"] == "qwen3.5-9b-4bit"
        assert data["cursor.aiprovider.openai.apiKey"] == "cursor-secret"

    def test_preserves_unrelated_settings(self, fake_home):
        (fake_home / "Cursor/User").mkdir(parents=True)
        cfg = cursor.current_config_path()
        cfg.write_text(json.dumps({"editor.fontSize": 14}))
        cursor.write_or_patch_config(
            "https://rapid.example.com", "alias", api_key="cursor-secret"
        )
        assert json.loads(cfg.read_text())["editor.fontSize"] == 14

    @pytest.mark.parametrize(
        "server_url",
        [
            "http://rapid.example.com",
            "https://127.0.0.1:8000",
            "https://rapid.local:8000",
        ],
    )
    def test_direct_write_rejects_non_public_endpoint(self, fake_home, server_url):
        config_path = fake_home / "Cursor/User/settings.json"
        with pytest.raises(ValueError):
            cursor.write_or_patch_config(
                server_url,
                "alias",
                api_key="cursor-secret",
                config_path=config_path,
            )
        assert not config_path.exists()

    def test_direct_write_requires_api_key(self, fake_home):
        config_path = fake_home / "Cursor/User/settings.json"
        with pytest.raises(ValueError, match="RAPID_MLX_API_KEY"):
            cursor.write_or_patch_config(
                "https://rapid.example.com", "alias", config_path=config_path
            )
        assert not config_path.exists()

    @pytest.mark.parametrize(
        "server_url",
        [
            "https://rapid.example.com?token=value",
            "https://rapid.example.com#settings",
        ],
    )
    def test_rejects_query_and_fragment(self, fake_home, server_url):
        (fake_home / "Cursor/User").mkdir(parents=True)
        with pytest.raises(ValueError, match="query string or fragment"):
            cursor.write_or_patch_config(server_url, "alias")


# --------------------------------------------------------------------
# Atomic-write + backup primitives
# --------------------------------------------------------------------


class TestCommon:
    def test_redact_secrets_hides_credentials_only(self):
        data = {
            "apiKey": "real",
            "placeholder": {"apiKey": "sk-noop"},
            "nested": [{"accessToken": "t", "tokenSource": "manual"}],
            "maxTokens": 10,
            "GITHUB_TOKEN": "g",
            "password": "",
            "model": "m",
        }
        assert _common.redact_secrets(data) == {
            "apiKey": "<redacted>",
            "placeholder": {"apiKey": "sk-noop"},
            "nested": [{"accessToken": "<redacted>", "tokenSource": "manual"}],
            "maxTokens": 10,
            "GITHUB_TOKEN": "<redacted>",
            "password": "",
            "model": "m",
        }
        assert data["apiKey"] == "real"

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

    def test_backup_is_never_more_permissive_than_its_source(self, tmp_path):
        """A backup must not widen access to what it copies.

        ``atomic_write_json`` writes the config itself through ``mkstemp``,
        so the live file is 0600 — and ``launch`` puts ``RAPID_MLX_API_KEY``
        into it. The backup used to be a plain ``write_bytes``, i.e.
        ``0666 & ~umask`` — 0644 on a default install — so every second
        ``rapid-mlx launch`` dropped the live bearer token into a
        world-readable file beside the protected one.

        Mutation check: restore ``bak.write_bytes(path.read_bytes())`` and
        this fails with 0644 (or whatever the ambient umask yields).
        """
        target = tmp_path / "config.json"
        _common.atomic_write_json(target, {"apiKey": "sk-secret"})
        source_mode = target.stat().st_mode & 0o777
        assert source_mode == 0o600, (
            "precondition: the config itself is written restrictively — if "
            "this changed, the backup expectation below must change with it"
        )

        bak = _common.backup_existing(target)

        assert bak is not None
        assert bak.read_bytes() == target.read_bytes()
        assert bak.stat().st_mode & 0o777 == source_mode

    def test_backup_matches_an_open_source_only_when_acls_are_readable(self, tmp_path):
        """Mirror a deliberately-open source — but only where we can prove it.

        A user who chmod'd their own config group-readable did so on purpose,
        and silently tightening the backup makes the recovery copy behave
        differently from the thing it recovers. That reasoning holds only
        while the mode bits tell the whole story. An ACL can *deny* a
        principal the bits would otherwise admit, and a freshly created file
        carries none — so reproducing 0644 from a 0644-plus-deny-ACL source
        hands the file to exactly the account it shut out.

        On Linux the ACL shows up as a ``system.posix_acl_*`` xattr, so
        absence is proof and the mode is reproduced. macOS has no
        ``os.listxattr`` at all and its ACLs are not xattrs anyway, so
        equivalence can never be established there and the backup stays
        owner-only. Tighter than the source still restores; wider does not
        un-leak.
        """
        target = tmp_path / "config.json"
        target.write_text('{"a": 1}')
        target.chmod(0o644)

        bak = _common.backup_existing(target)

        assert bak is not None
        if hasattr(os, "listxattr"):
            assert bak.stat().st_mode & 0o777 == 0o644
        else:
            assert bak.stat().st_mode & 0o777 == 0o600, (
                "without an ACL API we cannot vouch for group/other access"
            )

    def test_backup_drops_group_bits_when_the_group_cannot_be_matched(
        self, tmp_path, monkeypatch
    ):
        """Mode bits are numbers; what matters is who they authorize.

        A new file takes the *directory's* group, not the source's. Copying
        0640 from an ``alice:secrets`` config onto an ``alice:staff`` backup
        keeps the number and changes the audience — every member of staff can
        then read the API key. When the group cannot be adopted, the backup
        stays owner-only: tighter than the source still restores.
        """
        target = tmp_path / "config.json"
        target.write_text('{"apiKey": "sk-secret"}')
        target.chmod(0o640)

        def _refuse(*args, **kwargs):
            raise PermissionError("not a member of that group")

        monkeypatch.setattr(_common.os, "fchown", _refuse)

        bak = _common.backup_existing(target)

        assert bak is not None
        assert bak.read_bytes() == target.read_bytes()
        assert bak.stat().st_mode & 0o077 == 0, (
            "backup kept group/other access it could not vouch for"
        )

    def test_backup_never_touches_the_destination_by_name(self, tmp_path, monkeypatch):
        """Ownership and mode go through the descriptor, never the path.

        Anyone who can write the config's *directory* can unlink our backup
        and leave a symlink where it was. A pathname-based chown/chmod would
        then follow that symlink and re-permission someone else's file. The
        O_EXCL create is what makes the name ours; addressing the descriptor
        from then on is what keeps it ours.

        A 0640 source is what forces the interesting path: the mode-narrowing
        block only runs when the source has group/other bits, so a 0600 source
        would pass this test without ever reaching a chown or a chmod.
        """
        target = tmp_path / "config.json"
        target.write_text('{"apiKey": "sk-secret"}')
        target.chmod(0o640)

        def _boom(*args, **kwargs):
            raise AssertionError("backup_existing addressed the backup by pathname")

        monkeypatch.setattr(_common.os, "chown", _boom)
        monkeypatch.setattr(_common.os, "chmod", _boom)

        bak = _common.backup_existing(target)

        assert bak is not None
        assert bak.read_bytes() == target.read_bytes()
        # The mode still has to have been applied — through the descriptor,
        # since the pathname calls above would have raised. How wide it lands
        # is the ACL policy's business (see the test above), so assert only
        # that it is a mode this function could legitimately have chosen.
        assert bak.stat().st_mode & 0o777 in (0o600, 0o640)

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
        json=False,
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

    def test_list_json_is_complete_deduplicated_registry(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="list", json=True))
        assert excinfo.value.code == 0
        targets = json.loads(capsys.readouterr().out)
        ids = [target["id"] for target in targets]
        assert len(ids) == len(set(ids)) == 16
        assert "deepseek-harness" in ids
        # The launch registry's four config writers lead in display order;
        # pi is the highest-starred agents profile not already covered.
        assert ids[:5] == ["cline", "claude-code", "continue-dev", "cursor", "pi"]
        assert {target["kind"] for target in targets} == {
            "config_writer",
            "adapter_profile",
        }
        writer = next(t for t in targets if t["id"] == "claude-code")
        assert writer["config_path"].startswith("~/")
        assert next(t for t in targets if t["id"] == "codex")["config_path"] is None

    def test_unknown_client_exit_2(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="atom"))
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "unknown client" in err

    def test_cursor_explains_localhost_is_unsupported(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="cursor"))
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "publicly reachable HTTPS" in err
        assert "Cursor's servers" in err

    @pytest.mark.parametrize(
        "server_url",
        [
            "https://127.0.0.1:8000",
            "https://192.168.1.20:8000",
            "https://rapid.local:8000",
        ],
    )
    def test_cursor_rejects_private_https_endpoints(
        self, fake_home, capsys, server_url
    ):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url=server_url)
            )
        assert excinfo.value.code == 2
        assert "cannot reach" in capsys.readouterr().err

    def test_cursor_accepts_public_https_endpoint(self, fake_home, capsys, monkeypatch):
        (fake_home / "Cursor/User").mkdir(parents=True)
        monkeypatch.setenv("RAPID_MLX_API_KEY", "cursor-secret")
        launch_cli.launch_command(
            _make_args(
                client="cursor",
                server_url="https://rapid.example.com",
                dry_run=True,
            )
        )
        out = capsys.readouterr().out
        assert "[dry-run] cursor: detected=True" in out

    def test_cursor_does_not_infer_backend_routability_from_local_dns(
        self, fake_home, capsys, monkeypatch
    ):
        (fake_home / "Cursor/User").mkdir(parents=True)
        monkeypatch.setenv("RAPID_MLX_API_KEY", "cursor-secret")
        with patch("socket.getaddrinfo") as resolver:
            launch_cli.launch_command(
                _make_args(
                    client="cursor",
                    server_url="https://rapid.example.com",
                    dry_run=True,
                )
            )
        resolver.assert_not_called()

    def test_cursor_rejects_shorthand_loopback_address(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url="https://127.1")
            )
        assert excinfo.value.code == 2
        assert "cannot reach" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("server_url", "error_text"),
        [
            ("https://rapid.example.com:0", "non-zero HTTPS port"),
            ("https://rapid.example.com?token=value", "query string or fragment"),
            ("https://rapid.example.com#settings", "query string or fragment"),
        ],
    )
    def test_cursor_rejects_malformed_public_url_components(
        self, fake_home, capsys, server_url, error_text
    ):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url=server_url)
            )
        assert excinfo.value.code == 2
        assert error_text in capsys.readouterr().err

    @pytest.mark.parametrize(
        "server_url",
        [
            "https://[",
            "https://127.0.0.1\\@example.com",
            "https://user:password@example.com",
            "https://%31%32%37.0.0.1",
            "https://例子.测试",
        ],
    )
    def test_cursor_rejects_ambiguous_authority_without_traceback(
        self, fake_home, capsys, server_url
    ):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url=server_url)
            )
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "launch:" in err
        assert "Traceback" not in err

    def test_cursor_canonicalizes_validated_url(self, fake_home, capsys, monkeypatch):
        (fake_home / "Cursor/User").mkdir(parents=True)
        monkeypatch.setenv("RAPID_MLX_API_KEY", "cursor-secret")
        launch_cli.launch_command(
            _make_args(
                client="cursor",
                server_url="https://EXAMPLE.COM.:443/api/",
            )
        )
        config = json.loads(cursor.current_config_path().read_text())
        assert config["cursor.aiprovider.openai.baseUrl"] == (
            "https://example.com:443/api/v1"
        )
        assert config["cursor.aiprovider.openai.apiKey"] == "cursor-secret"

    def test_cursor_public_endpoint_requires_api_key(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url="https://example.com")
            )
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "require RAPID_MLX_API_KEY" in err
        assert "unauthenticated" in err

    def test_cursor_uses_api_key_from_environment(self, fake_home, capsys, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_API_KEY", "cursor-env-secret")
        (fake_home / "Cursor/User").mkdir(parents=True)

        launch_cli.launch_command(
            _make_args(client="cursor", server_url="https://rapid.example.com")
        )

        config = json.loads(cursor.current_config_path().read_text())
        assert config["cursor.aiprovider.openai.apiKey"] == "cursor-env-secret"

    def test_cursor_rejects_multicast_address(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url="https://224.0.0.1")
            )
        assert excinfo.value.code == 2
        assert "cannot reach" in capsys.readouterr().err

    def test_cursor_rejects_ipv6_site_local_address(self, fake_home, capsys):
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cursor", server_url="https://[fec0::1]")
            )
        assert excinfo.value.code == 2
        assert "cannot reach" in capsys.readouterr().err

    def test_all_skips_cursor_for_default_local_endpoint(self, fake_home, capsys):
        (fake_home / "Cursor/User").mkdir(parents=True)
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(all=True))
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "cursor: skipped" in err
        assert "publicly reachable HTTPS" in err
        assert "no supported clients detected" in err

    def test_all_reports_cursor_skipped_without_api_key(self, fake_home, capsys):
        (fake_home / "Cursor/User").mkdir(parents=True)
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(all=True, server_url="https://rapid.example.com")
            )
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "cursor: skipped" in err
        assert "require RAPID_MLX_API_KEY" in err

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
        ext_dir = _install_cline(fake_home)
        before = list(ext_dir.iterdir())

        launch_cli.launch_command(_make_args(client="cline", dry_run=True))
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "cline" in out
        # No file was created or modified.
        assert list(ext_dir.iterdir()) == before

    def test_real_patch_writes_file(self, fake_home, capsys):
        ext_dir = _install_cline(fake_home)
        launch_cli.launch_command(_make_args(client="cline", model="qwen3.5-4b-4bit"))
        target = ext_dir / "providers.json"
        assert target.exists()
        data = json.loads(target.read_text())
        assert _cline_settings(data)["model"] == "qwen3.5-4b-4bit"
        out = capsys.readouterr().out
        assert "Patched cline" in out
        assert "Now ready" in out

    def test_not_detected_client_fails_with_hint(self, fake_home, capsys):
        # cline is NOT detected (no CLI, data dir or extension). The command
        # should fail with a clear hint and exit non-zero.
        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(_make_args(client="cline"))
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "cline: not detected" in err

    def test_start_server_spawns_and_writes_pid(self, fake_home, capsys):
        _install_cline(fake_home)
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

    def test_claude_start_server_uses_cached_context_before_boot(
        self, fake_home, monkeypatch
    ):
        from rapid_mlx.agents import adapter

        (fake_home / ".claude").mkdir()
        monkeypatch.setattr(
            adapter,
            "fetch_context_window",
            lambda *_args: pytest.fail("server has not started yet"),
        )
        monkeypatch.setattr(
            "rapid_mlx.run.cli._cached_context_window", lambda _model: 131072
        )
        fake_proc = MagicMock()
        fake_proc.pid = 99997
        with patch.object(subprocess, "Popen", return_value=fake_proc):
            launch_cli.launch_command(
                _make_args(client="claude-code", model="local-model", start_server=True)
            )

        data = json.loads(claude_code.current_config_path().read_text())
        assert data["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"

    def test_api_key_is_passed_to_client_and_started_server(
        self, fake_home, capsys, monkeypatch
    ):
        ext_dir = _install_cline(fake_home)
        fake_proc = MagicMock()
        fake_proc.pid = 99998
        monkeypatch.setenv("RAPID_MLX_API_KEY", "shared-secret")
        with patch.object(subprocess, "Popen", return_value=fake_proc) as popen:
            launch_cli.launch_command(_make_args(client="cline", start_server=True))
        config = json.loads((ext_dir / "providers.json").read_text())
        assert _cline_settings(config)["apiKey"] == "shared-secret"
        assert popen.call_args.kwargs["env"]["RAPID_MLX_API_KEY"] == "shared-secret"

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

    def test_dry_run_previews_redacted_diff_for_cline(
        self, fake_home, capsys, monkeypatch
    ):
        _install_cline(fake_home)
        monkeypatch.setenv("RAPID_MLX_API_KEY", "shared-secret")
        launch_cli.launch_command(_make_args(client="cline", dry_run=True))
        out = capsys.readouterr().out
        assert "providers.json (proposed)" in out
        assert '"baseUrl": "http://127.0.0.1:8000/v1"' in out
        assert "shared-secret" not in out
        assert not (fake_home / ".cline/data/settings/providers.json").exists()

    def test_dry_run_reports_already_configured(self, fake_home, capsys):
        (fake_home / ".continue").mkdir()
        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "m")
        launch_cli.launch_command(
            _make_args(client="continue-dev", model="m", dry_run=True)
        )
        out = capsys.readouterr().out
        assert "continue-dev: already configured; no changes" in out

    def test_dry_run_shows_migration_note_and_unreadable_config(
        self, fake_home, capsys
    ):
        cont = fake_home / ".continue"
        cont.mkdir()
        (cont / "config.json").write_text('{"models": []}')
        launch_cli.launch_command(_make_args(client="continue-dev", dry_run=True))
        out = capsys.readouterr().out
        assert "continue-dev: note: Continue now reads config.yaml" in out
        assert "config.yaml (proposed)" in out

        (cont / "config.json").write_text("{broken")
        launch_cli.launch_command(_make_args(client="continue-dev", dry_run=True))
        out = capsys.readouterr().out
        assert "continue-dev: cannot preview" in out
        assert sorted(p.name for p in cont.iterdir()) == ["config.json"]

    def test_real_cline_patch_prints_extension_steps(self, fake_home, capsys):
        _install_cline(fake_home)
        launch_cli.launch_command(_make_args(client="cline", model="m"))
        out = capsys.readouterr().out
        assert "API Provider: OpenAI Compatible" in out
        assert "Model ID:     m" in out

    def test_continue_dev_launch_writes_yaml(self, fake_home, capsys):
        (fake_home / ".continue").mkdir()
        launch_cli.launch_command(_make_args(client="continue-dev", model="m"))
        out = capsys.readouterr().out
        assert "Patched continue-dev config at" in out
        assert "config.yaml" in out
        assert not (fake_home / ".continue/config.json").exists()

    def test_continue_is_an_alias_for_continue_dev(self, fake_home, capsys):
        """``launch continue`` must resolve exactly like ``launch
        continue-dev`` (#2082): ``rapid-mlx agents`` calls the same
        product ``continue``, so both slugs are accepted here. The
        canonical registry id stays ``continue-dev``."""
        (fake_home / ".continue").mkdir()
        launch_cli.launch_command(_make_args(client="continue-dev", dry_run=True))
        canonical_out = capsys.readouterr().out
        launch_cli.launch_command(_make_args(client="continue", dry_run=True))
        alias_out = capsys.readouterr().out
        assert alias_out == canonical_out
        assert "continue-dev: detected=True" in alias_out

    def test_uses_original_alias_when_resolved(self, fake_home, capsys):
        """When ``main()`` rewrites ``args.model`` from alias to HF id,
        the launch command should patch with the ORIGINAL alias so the
        IDE client requests the short name from rapid-mlx."""
        ext_dir = _install_cline(fake_home)
        ns = _make_args(client="cline", model="mlx-community/Qwen3.5-4B-MLX-4bit")
        # Simulate what ``main()`` does on the way in.
        ns._original_alias = "qwen3.5-4b-4bit"
        launch_cli.launch_command(ns)
        data = json.loads((ext_dir / "providers.json").read_text())
        assert _cline_settings(data)["model"] == "qwen3.5-4b-4bit"


# --------------------------------------------------------------------
# Top-level CLI argparse integration — invoke `python -m rapid_mlx.cli
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
    from rapid_mlx.launch.cli import register

    register(sub)
    # Choices populated.
    assert "launch" in sub.choices
    # And accept `list` as a client name.
    args = parser.parse_args(["launch", "list"])
    assert args.client == "list"


def test_launch_help_lists_canonical_name_with_alias_noted():
    """The client help keeps ``continue-dev`` as the canonical name and
    notes the ``continue`` alias (#2082)."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from rapid_mlx.launch.cli import register

    register(sub)
    # argparse wraps help text at arbitrary columns; normalize whitespace
    # before matching so the assertion doesn't depend on wrap width.
    help_text = " ".join(sub.choices["launch"].format_help().split())
    assert "continue-dev" in help_text
    assert "Aliases: continue (for continue-dev)" in help_text


@pytest.mark.parametrize("bad_port", ["0", "-1", "65536", "99999", "abc"])
def test_launch_port_rejects_out_of_range(bad_port):
    """``--port`` must use the same ``[1, 65535]`` validator as
    ``rapid-mlx serve``. Pre-fix, ``launch --port 99999`` parsed
    successfully and only failed inside the detached child after the
    parent had already printed "Started" and written a PID."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from rapid_mlx.launch.cli import register

    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "cline", "--port", bad_port])


def test_launch_port_accepts_in_range():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from rapid_mlx.launch.cli import register

    register(sub)
    args = parser.parse_args(["launch", "cline", "--port", "8000"])
    assert args.port == 8000


def test_launch_rejects_api_key_on_command_line():
    """Secrets for launch must travel via RAPID_MLX_API_KEY, never argv."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from rapid_mlx.launch.cli import register

    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "cursor", "--api-key", "leaked-secret"])
