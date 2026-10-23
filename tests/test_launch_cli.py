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
    for name in (
        "CLINE_DIR",
        "CLINE_DATA_DIR",
        "CLINE_PROVIDER_SETTINGS_PATH",
        "CONTINUE_GLOBAL_DIR",
    ):
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

    def test_provider_settings_path_override(self, tmp_path, monkeypatch):
        path = tmp_path / "custom" / "providers.json"
        monkeypatch.setenv("CLINE_PROVIDER_SETTINGS_PATH", str(path))
        assert cline.current_config_path() == path

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
                    "note": "kept",
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
        assert entry["note"] == "kept"
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
            '{"version": true, "providers": {}}',
            '{"version": 1, "modes": null, "providers": {}}',
            '{"version": 1, "lastUsedProvider": "", "providers": {}}',
            '{"version": 1, "providers": {"x": []}}',
            '{"version": 1, "providers": {"x": {"updatedAt": "t"}}}',
            '{"version": 1, "providers": {"x": {"settings": {"provider": "x"}}}}',
            '{"version": 1, "providers": {"x": {"settings": {"provider": "x"},'
            ' "updatedAt": "2026-01-01T00:00:00+02:00"}}}',
            '{"version": 1, "providers": {"x": {"settings": {"provider": "x"},'
            ' "updatedAt": "2026-01-01T00:00:00Z", "tokenSource": "stolen"}}}',
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

    def test_writes_through_a_symlinked_providers_file(self, fake_home):
        settings_dir = _install_cline(fake_home)
        real = fake_home / "dotfiles" / "providers.json"
        real.parent.mkdir()
        real.write_text('{"version": 1, "providers": {}}')
        link = settings_dir / "providers.json"
        link.symlink_to(real)

        cline.write_or_patch_config("http://127.0.0.1:8000", "m")

        assert link.is_symlink()
        assert _cline_settings(json.loads(real.read_text()))["model"] == "m"
        assert len(list(real.parent.glob("providers.json.bak.*"))) == 1

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
        # A header value is a credential whatever the header is called.
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {
                        "openai-compatible": {
                            "settings": {
                                "provider": "openai-compatible",
                                "headers": {"X-Custom": "hdr-secret"},
                            },
                            "updatedAt": "2026-01-01T00:00:00.000Z",
                        }
                    },
                }
            )
        )
        before = path.read_text()
        diff = cline.preview("http://127.0.0.1:8000", "m", "sk-noop")[1]
        assert "hdr-secret" not in diff and '"X-Custom": "<redacted>"' in diff
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
        assert data["env"]["ANTHROPIC_API_KEY"] == ""
        assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-noop"

    def test_launch_writes_live_claude_context(self, fake_home, monkeypatch):
        from rapid_mlx.agents import adapter

        (fake_home / ".claude").mkdir()
        monkeypatch.setattr(adapter, "fetch_context_window", lambda *_args: 131072)
        monkeypatch.setattr(
            "rapid_mlx.run.cli._cached_context_window", lambda _model: None
        )

        launch_cli.launch_command(
            _make_args(client="claude-code", model="local-model", legacy_config=True)
        )

        data = json.loads(claude_code.current_config_path().read_text())
        assert data["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"

    @pytest.mark.parametrize(
        "server_url", ["https://example.invalid", "http://127.0.0.1:8000"]
    )
    def test_explicit_server_url_keeps_existing_config_writer(
        self, fake_home, server_url
    ):
        (fake_home / ".claude").mkdir()
        launch_cli.launch_command(
            _make_args(
                client="claude-code",
                model="local-model",
                server_url=server_url,
            )
        )
        data = json.loads(claude_code.current_config_path().read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == server_url

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
        assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-noop"

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
        is the v1 header Continue writes for new files."""
        legacy = json.loads((FIXTURES / "continue_legacy_config.json").read_text())
        upstream = json.loads(
            (FIXTURES / "continue_legacy_converted_upstream.json").read_text()
        )
        upstream.update(name="Local Config", version="1.0.0", schema="v1")
        assert continue_dev.convert_legacy_config(legacy) == upstream

    def test_converter_passes_through_shapes_it_does_not_map(self):
        legacy = {
            "models": ["bare-model-id"],
            "tabAutocompleteModel": [
                {"title": "a1", "provider": "ollama", "model": "x"},
                "bare",
            ],
            "contextProviders": ["raw-context"],
            "customCommands": ["raw-command"],
            "docs": ["raw-doc"],
            "experimental": {"modelContextProtocolServers": ["raw-server"]},
        }
        converted = continue_dev.convert_legacy_config(legacy)
        assert converted["models"] == [
            "bare-model-id",
            {
                "name": "a1",
                "provider": "ollama",
                "model": "x",
                "roles": ["autocomplete"],
            },
            "bare",
        ]
        assert converted["context"] == ["raw-context"]
        assert converted["prompts"] == ["raw-command"]
        assert converted["docs"] == ["raw-doc"]
        assert converted["mcpServers"] == ["raw-server"]

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

    def test_existing_entry_without_chat_role_gains_it(self):
        existing = {
            "models": [
                {"name": "rapid-mlx", "provider": "openai", "roles": ["autocomplete"]}
            ]
        }
        after = continue_dev.patched_config(existing, "http://h", "m")
        assert after["models"][0]["roles"] == ["chat", "autocomplete"]
        assert existing["models"][0]["roles"] == ["autocomplete"]

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

    @pytest.mark.parametrize("content", ["", "{}\n"])
    def test_existing_empty_yaml_wins_over_legacy_json(self, fake_home, content):
        import yaml

        cont = fake_home / ".continue"
        cont.mkdir()
        (cont / "config.yaml").write_text(content)
        # Continue ignores config.json once config.yaml exists, so even a
        # broken legacy file must not be read or migrated.
        (cont / "config.json").write_text("{broken")
        path = continue_dev.write_or_patch_config("http://127.0.0.1:8000", "m")
        data = yaml.safe_load(path.read_text())
        assert [m["name"] for m in data["models"]] == ["rapid-mlx"]
        assert (cont / "config.json").read_text() == "{broken"

    def test_blank_yaml_is_treated_as_new(self, fake_home):
        cont = fake_home / ".continue"
        cont.mkdir()
        (cont / "config.yaml").write_text("\n")
        path = continue_dev.write_or_patch_config("http://127.0.0.1:8000", "m")
        assert "name: rapid-mlx" in path.read_text()

    def test_writes_through_a_symlinked_config_yaml(self, fake_home):
        cont = fake_home / ".continue"
        cont.mkdir()
        real = fake_home / "dotfiles" / "config.yaml"
        real.parent.mkdir()
        real.write_text("name: Mine\nversion: 1.0.0\nmodels: []\n")
        (cont / "config.yaml").symlink_to(real)

        continue_dev.write_or_patch_config("http://127.0.0.1:8000", "m")

        assert (cont / "config.yaml").is_symlink()
        assert "name: rapid-mlx" in real.read_text()
        assert len(list(real.parent.glob("config.yaml.bak.*"))) == 1

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
        (cont / "config.json").write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "title": "Proxy",
                            "provider": "openai",
                            "requestOptions": {
                                "headers": {"Authorization": "Bearer hdr-secret"}
                            },
                        }
                    ],
                    "experimental": {
                        "modelContextProtocolServers": [
                            {
                                "transport": {
                                    "command": "x",
                                    "env": {"DATABASE_URL": "postgres://u:pw@h"},
                                }
                            }
                        ]
                    },
                }
            )
        )
        diff = continue_dev.preview("http://127.0.0.1:8000", "m")[1]
        assert "hdr-secret" not in diff and "pw@h" not in diff
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
        server_url=None,
        port=8000,
        start_server=False,
        dry_run=False,
        json=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestLaunchCommand:
    def test_claude_desktop_prints_in_app_values_without_touching_client(
        self, fake_home, monkeypatch, capsys
    ):
        from rapid_mlx.agents import isolated
        from rapid_mlx.launch import claude_desktop

        calls = []
        monkeypatch.setattr(
            claude_desktop,
            "configure",
            lambda model, **options: (
                calls.append((model, options)) or ("http://127.0.0.1:7796", "test-key")
            ),
        )
        monkeypatch.setattr(isolated, "connection_status", lambda *_: "key-mismatch")
        launch_cli.launch_command(
            _make_args(client="claude-desktop", model="local-model", port=7796)
        )
        assert calls == [
            (
                "local-model",
                {"opus": None, "sonnet": None, "haiku": None, "port": 7796},
            )
        ]
        output = capsys.readouterr().out
        assert "Gateway URL: http://127.0.0.1:7796" in output
        assert "API key: test-key" in output
        assert "Restart the local server" in output
        assert "Apply Changes" in output
        assert not (
            fake_home / "Library" / "Application Support" / "Claude-3p"
        ).exists()

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            ("model-mismatch", "Start local-model on this port before Test"),
            ("stopped", "Start the local server before Test"),
        ],
    )
    def test_claude_desktop_reports_gateway_state(
        self, fake_home, monkeypatch, capsys, status, message
    ):
        from rapid_mlx.agents import isolated
        from rapid_mlx.launch import claude_desktop

        monkeypatch.setattr(
            claude_desktop,
            "configure",
            lambda *_args, **_kwargs: ("http://127.0.0.1:7796", "test-key"),
        )
        monkeypatch.setattr(isolated, "connection_status", lambda *_: status)
        launch_cli.launch_command(
            _make_args(client="claude-desktop", model="local-model", port=7796)
        )
        assert message in capsys.readouterr().out

    def test_claude_desktop_dry_run_and_remove_do_not_touch_client(
        self, fake_home, monkeypatch, capsys
    ):
        from rapid_mlx.launch import claude_desktop

        mapping = fake_home / "claude-desktop.json"
        mapping.write_text("saved", encoding="utf-8")
        monkeypatch.setattr(claude_desktop, "mapping_path", lambda: mapping)
        launch_cli.launch_command(_make_args(client="claude-desktop", dry_run=True))
        assert mapping.read_text(encoding="utf-8") == "saved"
        assert "Claude Desktop is untouched" in capsys.readouterr().out
        launch_cli.launch_command(_make_args(client="claude-desktop", remove=True))
        assert not mapping.exists()
        assert "mapping disabled" in capsys.readouterr().out

    def test_claude_desktop_requires_suitable_local_model(
        self, fake_home, monkeypatch, capsys
    ):
        from rapid_mlx.run import cli as run_cli

        monkeypatch.setattr(run_cli, "_select_model", lambda **_kwargs: None)
        with pytest.raises(SystemExit, match="1"):
            launch_cli.launch_command(_make_args(client="claude-desktop"))
        assert "no tool-capable model fits" in capsys.readouterr().err

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
        assert len(ids) == len(set(ids)) == 13
        assert "deepseek-harness" not in ids
        # Claude Desktop uses the in-app gateway flow and never writes its config.
        assert ids[:6] == [
            "claude-desktop",
            "cline",
            "claude-code",
            "continue-dev",
            "cursor",
            "pi",
        ]
        assert {target["kind"] for target in targets} == {
            "config_writer",
            "adapter_profile",
        }
        writer = next(t for t in targets if t["id"] == "claude-code")
        assert writer["config_path"].startswith("~/")
        assert next(t for t in targets if t["id"] == "codex")["config_path"] is None

    def test_discovery_capabilities_match_dispatch(
        self, fake_home, capsys, monkeypatch
    ):
        from rapid_mlx.agents.isolated import SUPPORTED
        from rapid_mlx.integrations import integration_targets_json

        # Exercise both detection labels without depending on installed clients.
        monkeypatch.setattr(cline, "detect", lambda: True)
        monkeypatch.setattr(cursor, "detect", lambda: False)
        assert launch_cli._print_list(as_json=True) == 0
        targets = json.loads(capsys.readouterr().out)
        capabilities = {t["id"]: t["capability"] for t in targets}
        assert {n for n, c in capabilities.items() if c == "launchable"} == SUPPORTED
        assert {n for n, c in capabilities.items() if c == "config-only"} == {
            "cline",
            "continue-dev",
            "cursor",
        }
        assert {n for n, c in capabilities.items() if c == "manual"} == {
            "claude-desktop"
        }
        assert launch_cli._print_list() == 0
        output = capsys.readouterr().out
        rows = [line.split() for line in output.splitlines() if line.startswith("  ")]
        assert [(r[0], r[1]) for r in rows] == list(capabilities.items())
        assert "config-only — detected" in output
        assert "config-only — not detected" in output
        assert "public HTTPS" in output
        assert "start the server and client separately" in output
        # Unsupported SDK/test profiles remain discoverable outside launch.
        excluded = {t["id"] for t in integration_targets_json()} - capabilities.keys()
        assert excluded == {
            "langchain",
            "openhands",
            "kilo-code",
            "pydanticai",
            "smolagents",
            "deepseek-harness",
        }
        for name in ["unknown-client", *sorted(excluded)]:
            with pytest.raises(SystemExit) as excinfo:
                launch_cli.launch_command(_make_args(client=name))
            assert excinfo.value.code == 2
            error = capsys.readouterr().err
            supported = error.split("Supported: ", 1)[1]
            assert supported.strip().removesuffix(".") == ", ".join(
                f"{n} ({c})" for n, c in capabilities.items()
            )

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

    def test_session_launch_keeps_default_port_when_unspecified(
        self, fake_home, capsys
    ):
        """The existing one-session launch still targets port 8000 by default."""
        _install_cline(fake_home)
        args = _make_args(client="cline", port=None, dry_run=True)

        launch_cli.launch_command(args)

        assert args.port == 8000
        assert "[dry-run]" in capsys.readouterr().out

    def test_real_patch_writes_file(self, fake_home, capsys):
        ext_dir = _install_cline(fake_home)
        launch_cli.launch_command(_make_args(client="cline", model="qwen3.5-4b-4bit"))
        target = ext_dir / "providers.json"
        assert target.exists()
        data = json.loads(target.read_text())
        assert _cline_settings(data)["model"] == "qwen3.5-4b-4bit"
        out = capsys.readouterr().out
        assert "Patched cline" in out
        assert "Configured: cline." in out
        assert "Now ready" not in out

    @pytest.mark.parametrize("client", ["cline", "continue-dev"])
    @pytest.mark.parametrize("start_server", [False, True])
    def test_legacy_configuration_does_not_claim_readiness(
        self, fake_home, capsys, client, start_server
    ):
        _install_cline(fake_home)
        (fake_home / ".continue").mkdir()
        with patch.object(
            launch_cli, "_start_server_background", return_value=99999
        ) as start:
            launch_cli.launch_command(
                _make_args(
                    client=client,
                    model="qwen3.5-4b-4bit",
                    server_url="http://127.0.0.1:18409",
                    port=18409,
                    start_server=start_server,
                )
            )
        out = capsys.readouterr().out
        assert f"Configured: {client}. No client was opened." in out
        assert "Now ready" not in out
        assert "then open the client" in out
        assert "configured endpoint" in out
        if start_server:
            start.assert_called_once()
            assert "Wait for the server to finish loading" in out
            assert "No server was started" not in out
        else:
            start.assert_not_called()
            assert "No server was started. Start Rapid-MLX" in out

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
                _make_args(
                    client="claude-code",
                    model="local-model",
                    start_server=True,
                    legacy_config=True,
                )
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
    assert '"continue" is an alias for "continue-dev"' in help_text
    for agent in ("Codex", "Claude Code", "OpenCode", "Pi", "Hermes", "Crush", "Goose"):
        assert agent in help_text


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


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_adapter_remove_preserves_user_config(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.current_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if client == "continue-dev":
        path.write_text(
            "models:\n  - name: personal\n    model: user-model\ncustom: keep\n"
        )
    else:
        path.write_text(
            json.dumps({"editor.fontSize": 15}) if client == "cursor" else "{}"
        )
    adapter.write_or_patch_config("https://example.com", "rapid-model", "secret-key")
    if client == "cline":
        data = cline.load_providers(path)
        data["providers"]["openai-compatible"]["settings"]["customHeaders"] = {
            "X-Test": "keep"
        }
        data["providers"]["personal"] = {
            "settings": {"provider": "personal", "apiKey": "user-key"},
            "updatedAt": "2026-01-01T00:00:00Z",
        }
        _common.atomic_write_json(path, data)
    before = path.read_bytes()
    launch_cli.launch_command(_make_args(client=client, remove=True, dry_run=True))
    assert path.read_bytes() == before
    launch_cli.launch_command(_make_args(client=client, remove=True))
    if client == "continue-dev":
        data = continue_dev.load_yaml_config(path)
        assert data["models"] == [{"name": "personal", "model": "user-model"}]
        assert data["custom"] == "keep"
    else:
        data = json.loads(path.read_text())
        if client == "cursor":
            assert data == {"editor.fontSize": 15}
        else:
            assert "lastUsedProvider" not in data
            assert data["providers"]["personal"]["settings"]["apiKey"] == "user-key"
            assert data["providers"]["openai-compatible"]["settings"] == {
                "provider": "openai-compatible",
                "customHeaders": {"X-Test": "keep"},
            }
    assert any(p.read_bytes() == before for p in path.parent.glob(path.name + ".bak.*"))
    after = path.read_bytes()
    launch_cli.launch_command(_make_args(client=client, remove=True))
    assert path.read_bytes() == after


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_adapter_remove_missing_and_malformed(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.current_config_path()
    assert adapter.remove_config() is False
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[invalid")
    with pytest.raises(SystemExit) as exc:
        launch_cli.launch_command(_make_args(client=client, remove=True))
    assert exc.value.code == 1
    assert path.read_text() == "[invalid"


@pytest.mark.parametrize("client", ["cline", "cursor"])
def test_shared_adapter_remove_preserves_changed_and_unmarked(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config(
        "https://example.com", "rapid-model", "secret-key"
    )
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    assert "secret-key" not in receipt.read_text()
    assert receipt.stat().st_mode & 0o777 == 0o600
    data = json.loads(path.read_text())
    fields = (
        data
        if client == "cursor"
        else data["providers"]["openai-compatible"]["settings"]
    )
    key = "cursor.aiprovider.openai.model" if client == "cursor" else "model"
    fields[key] = "user-edit"
    _common.atomic_write_json(path, data)
    assert adapter.remove_config()
    result = json.loads(path.read_text())
    fields = (
        result
        if client == "cursor"
        else result["providers"]["openai-compatible"]["settings"]
    )
    assert fields[key] == "user-edit"
    adapter.write_or_patch_config("https://example.com", "rapid-model", "secret-key")
    receipt.unlink()
    before = path.read_bytes()
    assert not adapter.remove_config()
    assert path.read_bytes() == before
    receipt.write_text("[]")
    with pytest.raises(ValueError, match="receipt"):
        adapter.remove_config()
    assert path.read_bytes() == before


def test_remove_rejects_unexpected_shapes(fake_home):
    path = cursor.current_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        cursor.remove_config()
    path = continue_dev.current_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("models: invalid\n")
    with pytest.raises(ValueError, match="list"):
        continue_dev.remove_config()


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_adapter_remove_writes_through_symlink(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.current_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    target = fake_home / "real-settings"
    path.symlink_to(target)
    adapter.write_or_patch_config("https://example.com", "rapid-model", "secret-key")
    assert adapter.remove_config()
    assert path.is_symlink()
    assert target.exists()


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_adapter_remove_retires_receipt(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "rapid-model", "key")
    configured = path.read_bytes()
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    assert adapter.remove_config()
    assert not receipt.exists()
    # Recreating the same settings manually must not inherit stale ownership.
    path.write_bytes(configured)
    assert not adapter.remove_config()
    assert path.read_bytes() == configured


@pytest.mark.parametrize("change", ["model", "custom", "unmarked"])
def test_continue_remove_preserves_customized_entry(fake_home, change):
    path = continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    data = continue_dev.load_yaml_config(path)
    if change == "unmarked":
        path.with_name(path.name + ".rapid-mlx-owned.json").unlink()
    else:
        data["models"][0][change] = "user-value"
        path.write_text(continue_dev.dump_yaml(data))
    before = path.read_bytes()
    assert not continue_dev.remove_config()
    assert path.read_bytes() == before


@pytest.mark.parametrize("client", ["cline", "continue-dev"])
def test_setup_plan_records_removable_ownership(fake_home, client):
    from rapid_mlx.agents.setup import apply_setup_plan, build_setup_plan

    plan = build_setup_plan(client, "https://example.com", "rapid-model")
    apply_setup_plan(plan)
    assert ADAPTERS[client].remove_config()


def test_continue_remove_preserves_legacy_json_and_hub_models(fake_home):
    path = continue_dev.current_config_path()
    path.parent.mkdir(parents=True)
    legacy = path.with_name("config.json")
    legacy.write_text('{"models": [{"title": "personal", "model": "user-model"}]}')
    original = legacy.read_bytes()
    continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    data = continue_dev.load_yaml_config(path)
    data["models"].extend([{"uses": "user/model"}, "user/reference"])
    path.write_text(continue_dev.dump_yaml(data))
    assert continue_dev.remove_config()
    assert legacy.read_bytes() == original
    assert continue_dev.load_yaml_config(path)["models"] == data["models"][1:]


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_adapter_remove_write_failure_preserves_config_and_receipt(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "rapid-model", "key")
    before = path.read_bytes()
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    ownership = receipt.read_bytes()
    with (
        patch.object(_common, "atomic_write_text", side_effect=OSError("read-only")),
        pytest.raises(SystemExit) as exc,
    ):
        launch_cli.launch_command(_make_args(client=client, remove=True))
    assert exc.value.code == 1
    assert path.read_bytes() == before
    assert receipt.read_bytes() == ownership


@pytest.mark.parametrize("extra", ["updated: 2026-10-22", "custom: {1: a, text: b}"])
def test_continue_remove_preserves_yaml_specific_values(fake_home, extra):
    path = continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    with path.open("a") as stream:
        stream.write(f"- name: personal\n  {extra}\n")
    data = continue_dev.load_yaml_config(path)
    data["models"].reverse()
    path.write_text(continue_dev.dump_yaml(data))
    assert continue_dev.remove_config()
    assert continue_dev.load_yaml_config(path)["models"] == data["models"][:-1]


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
@pytest.mark.parametrize("marked", [True, False])
def test_matching_config_does_not_create_or_refresh_ownership(
    fake_home, client, marked
):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "rapid-model", "key")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    if not marked:
        receipt.unlink()
    before = path.read_bytes()
    with patch.object(_common, "atomic_write_json") as write_json:
        adapter.write_or_patch_config("https://example.com", "rapid-model", "key")
    write_json.assert_not_called()
    assert path.read_bytes() == before
    assert receipt.exists() == marked
    assert adapter.remove_config() == marked


@pytest.mark.parametrize("client", ["cline", "continue-dev"])
def test_unchanged_setup_plan_does_not_claim_user_settings(fake_home, client):
    from rapid_mlx.agents.setup import apply_setup_plan, build_setup_plan

    plan = build_setup_plan(client, "https://example.com", "rapid-model")
    apply_setup_plan(plan)
    receipt = plan.path.with_name(plan.path.name + ".rapid-mlx-owned.json")
    receipt.unlink()
    before = plan.path.read_bytes()
    plan = build_setup_plan(client, "https://example.com", "rapid-model")
    assert not plan.changed
    apply_setup_plan(plan)
    assert not receipt.exists()
    assert not ADAPTERS[client].remove_config()
    assert plan.path.read_bytes() == before


@pytest.mark.parametrize(
    "client,setup",
    [
        ("cline", False),
        ("continue-dev", False),
        ("cursor", False),
        ("cline", True),
        ("continue-dev", True),
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_receipt_failure_rolls_back_config(fake_home, client, existing, setup):
    from rapid_mlx.agents.setup import apply_setup_plan, build_setup_plan

    adapter = ADAPTERS[client]
    path = adapter.current_config_path().resolve()
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    if existing:
        adapter.write_or_patch_config("https://example.com", "old-model", "key")
    before = path.read_bytes() if existing else None
    old_receipt = receipt.read_bytes() if existing else None
    original = _common.atomic_write_json

    def fail_receipt(destination, data):
        if destination == receipt:
            raise OSError("receipt publication failed")
        original(destination, data)

    with (
        patch.object(_common, "atomic_write_json", side_effect=fail_receipt),
        pytest.raises(OSError, match="receipt publication"),
    ):
        if setup:
            apply_setup_plan(
                build_setup_plan(client, "https://example.com", "new-model")
            )
        else:
            adapter.write_or_patch_config("https://example.com", "new-model", "key")
    assert (path.read_bytes() if path.exists() else None) == before
    assert (receipt.read_bytes() if receipt.exists() else None) == old_receipt


@pytest.mark.parametrize("clear", [False, True])
def test_remove_owned_selection_when_provider_settings_are_user_owned(fake_home, clear):
    path = cline.write_or_patch_config("https://example.com", "rapid-model", "key")
    data = cline.load_providers(path)
    settings = data["providers"]["openai-compatible"]["settings"]
    for key in ("apiKey", "model", "baseUrl"):
        if clear:
            del settings[key]
        else:
            settings[key] = "user-edit"
    _common.atomic_write_json(path, data)
    assert cline.remove_config()
    result = cline.load_providers(path)
    assert "lastUsedProvider" not in result
    assert result["providers"] == data["providers"]


def test_remove_only_one_owned_model_entry(fake_home):
    path = continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    data = continue_dev.load_yaml_config(path)
    data["models"].append(dict(data["models"][0]))
    path.write_text(continue_dev.dump_yaml(data))
    assert continue_dev.remove_config()
    assert continue_dev.load_yaml_config(path)["models"] == data["models"][1:]
    assert not continue_dev.remove_config()


@pytest.mark.parametrize("client", ["cline", "continue-dev"])
def test_setup_noop_guard_still_applies_independent_credentials(fake_home, client):
    from dataclasses import replace

    import yaml

    from rapid_mlx.agents.setup import apply_setup_plan, build_setup_plan

    apply_setup_plan(build_setup_plan(client, "https://example.com", "rapid-model"))
    plan = build_setup_plan(client, "https://example.com", "rapid-model")
    assert not plan.changed
    receipt = plan.path.with_name(plan.path.name + ".rapid-mlx-owned.json")
    receipt.unlink()
    before = plan.path.read_bytes()
    credentials = fake_home / "credential-overrides.yaml"
    plan = replace(
        plan,
        credentials_path=credentials,
        credentials_before={},
        credentials_after={"test_key": "updated-value"},
    )
    assert plan.before == plan.after
    assert plan.changed
    apply_setup_plan(plan)
    assert yaml.safe_load(credentials.read_text()) == {"test_key": "updated-value"}
    assert plan.path.read_bytes() == before
    assert not receipt.exists()


def test_customized_owned_model_does_not_authorize_deleting_duplicate(fake_home):
    path = continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    data = continue_dev.load_yaml_config(path)
    data["models"].append(dict(data["models"][0]))
    data["models"][0]["custom"] = "user-value"
    path.write_text(continue_dev.dump_yaml(data))
    before = path.read_bytes()
    assert not continue_dev.remove_config()
    assert path.read_bytes() == before


@pytest.mark.parametrize("extra", ["updated: 2026-10-22", "custom: {1: a, text: b}"])
def test_remove_preserves_owned_model_with_yaml_customization(fake_home, extra):
    import yaml

    path = continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    data = continue_dev.load_yaml_config(path)
    data["models"][0].update(yaml.safe_load(extra))
    path.write_text(continue_dev.dump_yaml(data))
    before = path.read_bytes()
    assert not continue_dev.remove_config()
    assert path.read_bytes() == before


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_receipt_retirement_failure_rolls_back_removal(fake_home, client):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "rapid-model", "key")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    before = path.read_bytes()
    ownership = receipt.read_bytes()
    with (
        patch(
            "rapid_mlx.launch._ownership.forget",
            side_effect=OSError("cannot retire receipt"),
        ),
        pytest.raises(SystemExit) as exc,
    ):
        launch_cli.launch_command(_make_args(client=client, remove=True))
    assert exc.value.code == 1
    assert path.read_bytes() == before
    assert receipt.read_bytes() == ownership
    assert adapter.remove_config()
    assert not receipt.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_receipt_replacement_failure_restores_both_files(fake_home, existing):
    from rapid_mlx.launch import _ownership

    path = fake_home / "transaction.json"
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    if existing:
        _ownership.write(
            path,
            {"model": "old"},
            lambda: _common.atomic_write_json(path, {"model": "old"}),
            before_fields={},
        )
    before = path.read_bytes() if existing else None
    prior_receipt = receipt.read_bytes() if existing else None
    original = _common.atomic_write_json

    def publish_then_fail(destination, data):
        original(destination, data)
        if destination == receipt:
            raise OSError("failed after receipt replacement")

    with (
        patch.object(_common, "atomic_write_json", side_effect=publish_then_fail),
        pytest.raises(OSError, match="after receipt replacement"),
    ):
        _ownership.write(
            path,
            {"model": "new"},
            lambda: _common.atomic_write_json(path, {"model": "new"}),
            before_fields={},
        )
    assert (path.read_bytes() if path.exists() else None) == before
    assert (receipt.read_bytes() if receipt.exists() else None) == prior_receipt


def test_retirement_interruption_restores_receipt_and_config(fake_home):
    from rapid_mlx.launch import _ownership

    path = cline.write_or_patch_config("https://example.com", "rapid-model", "key")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    before, prior_receipt = path.read_bytes(), receipt.read_bytes()
    original = _ownership.forget

    def retire_then_interrupt(target):
        original(target)
        raise KeyboardInterrupt

    with (
        patch.object(_ownership, "forget", side_effect=retire_then_interrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        cline.remove_config()
    assert path.read_bytes() == before
    assert receipt.read_bytes() == prior_receipt


def test_overlapping_configuration_failure_cannot_undo_success(fake_home):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from rapid_mlx.launch import _ownership

    path = fake_home / "transaction.json"
    _common.atomic_write_json(path, {"model": "original"})
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    first_written, release_first = Event(), Event()
    second_started, second_finished = Event(), Event()
    original = _common.atomic_write_json
    first_thread = []

    def fail_first_receipt(destination, data):
        if destination == receipt and current_thread() is first_thread[0]:
            raise OSError("first receipt failed")
        original(destination, data)

    def first_write():
        first_thread.append(current_thread())

        def config():
            original(path, {"model": "first"})
            first_written.set()
            assert release_first.wait(5)

        _ownership.write(path, {"model": "first"}, config, before_fields={})

    def second_write():
        second_started.set()
        _ownership.write(
            path,
            {"model": "second"},
            lambda: original(path, {"model": "second"}),
            before_fields={},
        )
        second_finished.set()

    with (
        patch.object(_common, "atomic_write_json", side_effect=fail_first_receipt),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(first_write)
        try:
            assert first_written.wait(5)
            second = pool.submit(second_write)
            assert second_started.wait(5)
            # Give the second writer a chance to finish if the lock is absent.
            # Its successful write must remain after the first writer rolls back.
            second_finished.wait(0.2)
        finally:
            release_first.set()
        with pytest.raises(OSError, match="first receipt failed"):
            first.result(timeout=5)
        second.result(timeout=5)
    data = json.loads(path.read_text())
    assert data == {"model": "second"}
    assert _ownership.remove_fields(path, data, ("model",))


@pytest.mark.parametrize("client", ["cline", "cursor"])
@pytest.mark.parametrize("already_owned", [False, True])
def test_partial_update_preserves_unchanged_user_fields(
    fake_home, client, already_owned
):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "old-model", "key")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    if not already_owned:
        receipt.unlink()
    adapter.write_or_patch_config("https://example.com", "new-model", "key")
    assert adapter.remove_config()
    data = json.loads(path.read_text())
    if client == "cursor":
        fields = data
        assert "cursor.aiprovider.openai.model" not in fields
        preserved = {
            "cursor.aiprovider.openai.baseUrl": "https://example.com/v1",
            "cursor.aiprovider.openai.apiKey": "key",
        }
    else:
        fields = data["providers"]["openai-compatible"]["settings"]
        assert "model" not in fields
        preserved = {"baseUrl": "https://example.com/v1", "apiKey": "key"}
        assert ("lastUsedProvider" in data) != already_owned
    for key, value in preserved.items():
        if already_owned:
            assert key not in fields
        else:
            assert fields[key] == value


@pytest.mark.parametrize("custom", [False, "extra", "model"])
def test_reconfigured_user_model_entry_is_not_claimed(fake_home, custom):
    import yaml

    path = continue_dev.write_or_patch_config("https://example.com", "old-model")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    if custom:
        data = continue_dev.load_yaml_config(path)
        if custom == "model":
            data["models"][0]["model"] = yaml.safe_load("2026-10-22")
        else:
            data["models"][0].update(yaml.safe_load("updated: 2026-10-22"))
        path.write_text(continue_dev.dump_yaml(data))
    else:
        receipt.unlink()
    continue_dev.write_or_patch_config("https://example.com", "new-model")
    before = path.read_bytes()
    assert not continue_dev.remove_config()
    assert path.read_bytes() == before


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
@pytest.mark.parametrize("no_op", [False, True])
def test_concurrent_removal_preserves_later_configuration(fake_home, client, no_op):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from rapid_mlx.launch import _ownership

    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "old-model", "key")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    if no_op:
        if client == "continue-dev":
            data = continue_dev.load_yaml_config(path)
            data["models"][0]["custom"] = "user-owned"
            path.write_text(continue_dev.dump_yaml(data))
        else:
            data = json.loads(path.read_text())
            fields = (
                data
                if client == "cursor"
                else data["providers"]["openai-compatible"]["settings"]
            )
            for key in adapter._OWNED_SETTINGS:
                fields[key] = "user-owned"
            if client == "cline":
                data["lastUsedProvider"] = "user-provider"
            _common.atomic_write_json(path, data)
    paused, release, writer_started, writer_finished = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    removal_thread = []
    target = "forget" if no_op else "remove_fields"
    original = getattr(_ownership, target)

    def pause_removal(*args, **kwargs):
        if current_thread() is removal_thread[0] and not paused.is_set():
            paused.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    def remove():
        removal_thread.append(current_thread())
        return adapter.remove_config()

    def write():
        writer_started.set()
        adapter.write_or_patch_config("https://example.com", "new-model", "new-key")
        writer_finished.set()

    with (
        patch.object(_ownership, target, side_effect=pause_removal),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        removing = pool.submit(remove)
        try:
            assert paused.wait(5)
            writing = pool.submit(write)
            assert writer_started.wait(5)
            writer_finished.wait(0.2)
        finally:
            release.set()
        assert removing.result(timeout=5) != no_op
        writing.result(timeout=5)
    assert receipt.exists()
    if client == "continue-dev":
        assert continue_dev.load_yaml_config(path)["models"][0]["model"] == "new-model"
    else:
        data = json.loads(path.read_text())
        fields = (
            data
            if client == "cursor"
            else data["providers"]["openai-compatible"]["settings"]
        )
        key = "cursor.aiprovider.openai.model" if client == "cursor" else "model"
        assert fields[key] == "new-model"


@pytest.mark.parametrize("strip_custom", [False, True])
def test_migrated_user_model_entry_keeps_its_custom_fields(fake_home, strip_custom):
    path = continue_dev.current_config_path()
    path.parent.mkdir(parents=True)
    legacy = path.with_name("config.json")
    legacy.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "title": "rapid-mlx",
                        "provider": "openai",
                        "model": "user-model",
                        "completionOptions": {"temperature": 0.2},
                    }
                ]
            }
        )
    )
    continue_dev.write_or_patch_config("https://example.com", "rapid-model")
    if strip_custom:
        data = continue_dev.load_yaml_config(path)
        data["models"][0].pop("roles")
        data["models"][0].pop("defaultCompletionOptions")
        path.write_text(continue_dev.dump_yaml(data))
    before = path.read_bytes()
    assert not continue_dev.remove_config()
    assert path.read_bytes() == before
    if not strip_custom:
        assert continue_dev.load_yaml_config(path)["models"][0][
            "defaultCompletionOptions"
        ] == {"temperature": 0.2}


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_explicit_isolated_removal_never_changes_shared_settings(
    fake_home, client, dry_run, capsys
):
    adapter = ADAPTERS[client]
    path = adapter.write_or_patch_config("https://example.com", "rapid-model", "key")
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    before, owned = path.read_bytes(), receipt.read_bytes()
    with patch(
        "rapid_mlx.agents.isolated.remove", return_value=False
    ) as isolated_remove:
        launch_cli.launch_command(
            _make_args(client=client, isolated=True, remove=True, dry_run=dry_run)
        )
    if dry_run:
        isolated_remove.assert_not_called()
        assert "isolated" in capsys.readouterr().out
    else:
        isolated_remove.assert_called_once_with(client)
    assert path.read_bytes() == before
    assert receipt.read_bytes() == owned


@pytest.mark.parametrize("client", ["cline", "continue-dev", "cursor"])
def test_all_removal_rejects_ambiguous_scope_without_shared_writes(fake_home, client):
    path = ADAPTERS[client].write_or_patch_config(
        "https://example.com", "rapid-model", "key"
    )
    receipt = path.with_name(path.name + ".rapid-mlx-owned.json")
    before, owned = path.read_bytes(), receipt.read_bytes()
    with pytest.raises(SystemExit) as exc:
        launch_cli.launch_command(_make_args(client=client, all=True, remove=True))
    assert exc.value.code == 2
    assert path.read_bytes() == before
    assert receipt.read_bytes() == owned


@pytest.mark.parametrize("client", ["cline", "continue-dev"])
@pytest.mark.parametrize("direct_plan", [False, True])
def test_setup_owned_receipt_and_removal_use_symlink_target(
    fake_home, client, direct_plan
):
    from dataclasses import replace

    from rapid_mlx.agents.setup import apply_setup_plan, build_setup_plan

    adapter = ADAPTERS[client]
    link = adapter.current_config_path()
    link.parent.mkdir(parents=True)
    target = fake_home / "real-settings"
    link.symlink_to(target)
    plan = build_setup_plan(client, "https://example.com", "rapid-model")
    assert plan.path == target.resolve()
    if direct_plan:
        plan = replace(plan, path=link)
    assert apply_setup_plan(plan) == target.resolve()
    assert link.is_symlink()
    assert target.with_name(target.name + ".rapid-mlx-owned.json").exists()
    assert not link.with_name(link.name + ".rapid-mlx-owned.json").exists()
    assert adapter.remove_config()
    assert link.is_symlink()
