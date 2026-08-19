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
import io
import json
import os
import re
import subprocess
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.launch import (
    ADAPTERS,
    _common,
    claude_code,
    cline,
    openhands,
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
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)

    # cline: pin the data root at tmp_path via the same env var the
    # extension itself honours, and blank the VS Code extension probe
    # so the dev machine's real Cline install can't make detect() true.
    monkeypatch.setenv("CLINE_DATA_DIR", str(tmp_path / "cline-data"))
    monkeypatch.delenv("CLINE_DIR", raising=False)
    monkeypatch.setattr(cline, "_candidate_extension_dirs", lambda: [])

    # claude_code: replace the two module constants.
    monkeypatch.setattr(claude_code, "_CLAUDE_STATE_DIR", tmp_path / ".claude")
    monkeypatch.setattr(claude_code, "_CONFIG_DIR", tmp_path / ".claude")

    # openhands: pin the data root via the same env var the adapter
    # honours, so a real ~/.openhands on the dev machine can't make
    # detect() true. OPENHANDS_URL is cleared for the same reason a
    # locally running agent-canvas must not receive test PATCHes.
    monkeypatch.setenv("OPENHANDS_DIR", str(tmp_path / ".openhands"))
    monkeypatch.delenv("OPENHANDS_URL", raising=False)

    # Also redirect which() and mac_app_installed() so detect() doesn't
    # find the dev machine's real client installs.
    monkeypatch.setattr(_common, "which", lambda _: None)
    monkeypatch.setattr(_common, "mac_app_installed", lambda _: False)

    # And the PID file the launch CLI writes when --start-server is on.
    monkeypatch.setattr(launch_cli, "PID_FILE", tmp_path / "launch.pid")
    monkeypatch.setattr(launch_cli, "_start_port_available", lambda _port: True)

    return tmp_path


# --------------------------------------------------------------------
# Cline adapter
# --------------------------------------------------------------------


class TestCline:
    def test_detect_false_when_nothing_installed(self, fake_home):
        assert cline.detect() is False
        assert cline.current_config_path() is None

    def test_detect_true_when_data_dir_exists(self, fake_home):
        # Cline has run at least once — the data tree is the store we
        # patch, so its mere existence is proof enough.
        (fake_home / "cline-data").mkdir(parents=True)
        assert cline.detect() is True
        path = cline.current_config_path()
        assert path is not None
        assert path.name == "globalState.json"

    def test_detect_true_when_only_extension_installed(self, fake_home, monkeypatch):
        # Installed from the marketplace but never opened: no data dir,
        # but the versioned extension directory is there. We create the
        # data tree ourselves on write.
        ext_root = fake_home / "vscode-extensions"
        (ext_root / "saoudrizwan.claude-dev-4.1.10").mkdir(parents=True)
        monkeypatch.setattr(cline, "_candidate_extension_dirs", lambda: [ext_root])
        assert cline.detect() is True

    def test_write_preserves_existing_keys(self, fake_home):
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        state_path = data_dir / "globalState.json"
        state_path.write_text(
            json.dumps(
                {
                    "planModeApiProvider": "anthropic",  # we'll overwrite
                    "actModeApiProvider": "anthropic",  # we'll overwrite
                    "taskHistory": [{"id": "abc"}],  # must survive
                    "preferredLanguage": "English",  # must survive
                }
            )
        )
        secrets_path = data_dir / "secrets.json"
        secrets_path.write_text(json.dumps({"openRouterApiKey": "keep-me"}))

        returned = cline.write_or_patch_config(
            "http://127.0.0.1:8000",
            "qwen3.5-4b-4bit",
            api_key="sk-noop",
        )
        assert returned == state_path

        # Backup exists for each file we rewrote.
        assert len(list(data_dir.glob("globalState.json.bak.*"))) == 1
        assert len(list(data_dir.glob("secrets.json.bak.*"))) == 1

        state = json.loads(state_path.read_text())
        # Keys we own — set / overwritten. Plan and Act are separate
        # provider selections in Cline; both have to point at us.
        assert state["planModeApiProvider"] == "openai"
        assert state["actModeApiProvider"] == "openai"
        assert state["openAiBaseUrl"] == "http://127.0.0.1:8000/v1"
        assert state["planModeOpenAiModelId"] == "qwen3.5-4b-4bit"
        assert state["actModeOpenAiModelId"] == "qwen3.5-4b-4bit"
        assert state["welcomeViewCompleted"] is True
        # Keys we don't own — untouched.
        assert state["taskHistory"] == [{"id": "abc"}]
        assert state["preferredLanguage"] == "English"

        # The API key is a secret, and lives in its own file.
        secrets = json.loads(secrets_path.read_text())
        assert secrets["openAiApiKey"] == "sk-noop"
        assert secrets["openRouterApiKey"] == "keep-me"
        assert "openAiApiKey" not in state

    def test_write_populates_next_bundle_providers_json(self, fake_home):
        """The extension can be flipped between its ``legacy`` and
        ``next`` bundles by a remote rollout, so we must configure both
        or a server-side flip silently unconfigures the user."""
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        (data_dir / "settings").mkdir()
        (data_dir / "settings" / "providers.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "lastUsedProvider": "cline",
                    "providers": {
                        "cline": {
                            "settings": {"provider": "cline", "model": "some-model"},
                            "updatedAt": "2026-08-19T02:42:26Z",
                            "tokenSource": "oauth",
                        }
                    },
                }
            )
        )

        cline.write_or_patch_config(
            "http://127.0.0.1:8000",
            "qwen3.5-4b-4bit",
            api_key="sk-noop",
        )

        doc = json.loads((data_dir / "settings" / "providers.json").read_text())
        assert doc["version"] == 1
        assert doc["lastUsedProvider"] == "openai-compatible"
        entry = doc["providers"]["openai-compatible"]
        assert entry["settings"] == {
            "provider": "openai-compatible",
            "apiKey": "sk-noop",
            "model": "qwen3.5-4b-4bit",
            "baseUrl": "http://127.0.0.1:8000/v1",
        }
        assert entry["tokenSource"] == "manual"
        # zod's ``.datetime()`` rejects Python's default microsecond
        # precision — the timestamp has to be second-granular UTC.
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["updatedAt"])
        # The user's other provider (and its OAuth token) survives.
        assert doc["providers"]["cline"]["tokenSource"] == "oauth"

    def test_creates_tree_when_cline_never_opened(self, fake_home, monkeypatch):
        ext_root = fake_home / "vscode-extensions"
        (ext_root / "saoudrizwan.claude-dev-4.1.10").mkdir(parents=True)
        monkeypatch.setattr(cline, "_candidate_extension_dirs", lambda: [ext_root])

        cline.write_or_patch_config("http://127.0.0.1:8000", "alias")

        data_dir = fake_home / "cline-data"
        assert (
            json.loads((data_dir / "globalState.json").read_text())[
                "actModeApiProvider"
            ]
            == "openai"
        )
        assert (data_dir / "secrets.json").exists()
        assert (data_dir / "settings" / "providers.json").exists()

    def test_does_not_double_append_v1(self, fake_home):
        """User passes ``http://127.0.0.1:8000/v1`` — must NOT
        produce ``/v1/v1``."""
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        cline.write_or_patch_config(
            "http://127.0.0.1:8000/v1",
            "alias",
        )
        state = json.loads((data_dir / "globalState.json").read_text())
        assert state["openAiBaseUrl"] == "http://127.0.0.1:8000/v1"

    def test_cline_dir_env_relocates_the_tree(self, fake_home, monkeypatch):
        """A user who moved Cline off a synced home dir must get their
        REAL config patched, not the default path."""
        monkeypatch.delenv("CLINE_DATA_DIR")
        relocated = fake_home / "elsewhere"
        monkeypatch.setenv("CLINE_DIR", str(relocated))
        (relocated / "data").mkdir(parents=True)

        assert cline.current_config_path() == relocated / "data" / "globalState.json"


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


class TestOpenHands:
    """OpenHands is the one adapter that cannot write its own config.

    ``~/.openhands/settings.json`` stores ``api_key`` Fernet-encrypted
    with a key we don't hold, so the only supported write is a PATCH to
    the running agent-server, which encrypts on our behalf. These tests
    capture the request instead of the file.
    """

    @staticmethod
    def _install(fake_home, key: str = "session-key") -> None:
        """Lay down the two on-disk markers a real install leaves."""
        canvas = fake_home / ".openhands" / "agent-canvas"
        canvas.mkdir(parents=True, exist_ok=True)
        (canvas / "api-key.txt").write_text(key + "\n")

    @staticmethod
    def _capture(monkeypatch) -> list:
        """Intercept urlopen and record the PATCH requests sent.

        The port sweep issues GET probes through the same urlopen, so
        only the writes are recorded — otherwise ``sent[0]`` would be a
        probe rather than the settings update under test. Probes are
        answered 200 so the first candidate port wins and the sweep
        stops.
        """
        sent = []

        def fake_urlopen(request, timeout=None):
            if request.get_method() == "PATCH":
                sent.append(request)
            response = MagicMock(status=200)
            response.__enter__ = lambda s: s
            response.__exit__ = lambda *a: False
            return response

        monkeypatch.setattr(openhands.urllib.request, "urlopen", fake_urlopen)
        return sent

    def test_detect_false_when_nothing_installed(self, fake_home):
        assert openhands.detect() is False

    def test_detect_true_when_data_dir_exists(self, fake_home):
        self._install(fake_home)
        assert openhands.detect() is True

    def test_detect_does_not_require_a_running_server(self, fake_home):
        # A user with OpenHands installed but closed must be told to
        # start it, not told it isn't installed.
        self._install(fake_home)
        assert openhands.detect() is True
        assert openhands.current_config_path() is not None

    def test_model_carries_the_litellm_provider_prefix(self, fake_home, monkeypatch):
        # OpenHands routes completions through LiteLLM, which cannot
        # resolve a bare non-catalog name and errors before any request
        # reaches us.
        self._install(fake_home)
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8001", "qwen3.5-4b-4bit")
        llm = json.loads(sent[0].data)["agent_settings_diff"]["llm"]
        assert llm["model"] == "openai/qwen3.5-4b-4bit"

    def test_bearer_is_forwarded_verbatim(self, fake_home, monkeypatch):
        # The desktop app's per-launch bearer must arrive unmodified or
        # every completion 401s.
        self._install(fake_home)
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config(
            "http://127.0.0.1:8001", "model", api_key="deadbeef"
        )
        llm = json.loads(sent[0].data)["agent_settings_diff"]["llm"]
        assert llm["api_key"] == "deadbeef"

    def test_session_key_authenticates_the_patch(self, fake_home, monkeypatch):
        self._install(fake_home, key="s3cret")
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8001", "model")
        assert sent[0].get_header("X-session-api-key") == "s3cret"
        assert sent[0].get_method() == "PATCH"

    def test_base_url_gains_v1_but_is_not_doubled(self, fake_home, monkeypatch):
        self._install(fake_home)
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8001", "model")
        openhands.write_or_patch_config("http://127.0.0.1:8001/v1", "model")
        urls = [
            json.loads(r.data)["agent_settings_diff"]["llm"]["base_url"] for r in sent
        ]
        assert urls == ["http://127.0.0.1:8001/v1"] * 2

    def test_diff_carries_only_the_three_keys_we_own(self, fake_home, monkeypatch):
        # The user's retries, reasoning effort, condenser and tool config
        # live in the same LLM block and must survive.
        self._install(fake_home)
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8001", "model")
        body = json.loads(sent[0].data)
        assert set(body) == {"agent_settings_diff"}
        assert set(body["agent_settings_diff"]["llm"]) == {
            "model",
            "base_url",
            "api_key",
        }

    def test_not_installed_raises_actionable_error(self, fake_home):
        with pytest.raises(FileNotFoundError, match="does not appear to be installed"):
            openhands.write_or_patch_config("http://127.0.0.1:8001", "model")

    def test_missing_session_key_names_the_fix(self, fake_home):
        # Installed but never opened: the directory exists, the key file
        # does not.
        (fake_home / ".openhands" / "agent-canvas").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="session key not found"):
            openhands.write_or_patch_config("http://127.0.0.1:8001", "model")

    def test_unreachable_server_explains_why_a_file_write_is_not_enough(
        self, fake_home, monkeypatch
    ):
        self._install(fake_home)

        def refuse(request, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(openhands.urllib.request, "urlopen", refuse)
        with pytest.raises(RuntimeError, match="start OpenHands and re-run"):
            openhands.write_or_patch_config("http://127.0.0.1:8001", "model")

    def test_openhands_url_env_redirects_the_patch(self, fake_home, monkeypatch):
        self._install(fake_home)
        monkeypatch.setenv("OPENHANDS_URL", "http://127.0.0.1:9999/")
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8001", "model")
        assert sent[0].full_url == "http://127.0.0.1:9999/api/settings"

    def test_explicit_url_is_never_probed(self, fake_home, monkeypatch):
        # An address the user typed must not be silently second-guessed.
        self._install(fake_home)
        monkeypatch.setenv("OPENHANDS_URL", "http://127.0.0.1:9999")
        monkeypatch.setattr(
            openhands,
            "_is_openhands",
            lambda *a: pytest.fail("explicit OPENHANDS_URL was probed"),
        )
        self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8001", "model")

    def test_sweep_skips_a_port_that_is_not_openhands(self, fake_home, monkeypatch):
        # The real case this exists for: rapid-mlx holds :8000 (its own
        # default) and agent-canvas was moved to :8010 to get out of the
        # way. The pasted command carries no port, so we must find it.
        self._install(fake_home)
        monkeypatch.setattr(
            openhands,
            "_is_openhands",
            lambda url, key: url == "http://127.0.0.1:8010",
        )
        sent = self._capture(monkeypatch)
        openhands.write_or_patch_config("http://127.0.0.1:8000", "model")
        assert sent[0].full_url == "http://127.0.0.1:8010/api/settings"

    def test_sweep_authenticates_rather_than_just_pinging(self, fake_home, monkeypatch):
        # Probing "is anything listening" would latch onto rapid-mlx on
        # :8000. The predicate must carry the session key.
        self._install(fake_home, key="s3cret")
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_header("X-session-api-key")))
            raise OSError("refused")

        monkeypatch.setattr(openhands.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError):
            openhands.write_or_patch_config("http://127.0.0.1:8000", "model")
        assert seen, "no probe was attempted"
        assert all(key == "s3cret" for _, key in seen)

    def test_sweep_failure_falls_back_to_the_default_port(
        self, fake_home, monkeypatch
    ):
        # Nothing answered anywhere: the error should still name a
        # plausible address rather than an empty string.
        self._install(fake_home)
        monkeypatch.setattr(openhands, "_is_openhands", lambda *a: False)

        def refuse(request, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(openhands.urllib.request, "urlopen", refuse)
        with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:8000"):
            openhands.write_or_patch_config("http://127.0.0.1:8001", "model")

    def test_404_blames_the_port_not_openhands(self, fake_home, monkeypatch):
        # :8000 is both agent-canvas' ingress and rapid-mlx's own default
        # serve port. When rapid-mlx answers instead, the user must be
        # pointed at the port clash rather than sent hunting through
        # OpenHands for a fault that isn't there.
        self._install(fake_home)

        def not_found(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {}, io.BytesIO(b'{"error":{}}')
            )

        monkeypatch.setattr(openhands.urllib.request, "urlopen", not_found)
        with pytest.raises(RuntimeError, match="not as OpenHands"):
            openhands.write_or_patch_config("http://127.0.0.1:8001", "model")

    def test_other_http_errors_surface_the_server_detail(self, fake_home, monkeypatch):
        self._install(fake_home)

        def unauthorized(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(b'{"detail":"Unauthorized"}'),
            )

        monkeypatch.setattr(openhands.urllib.request, "urlopen", unauthorized)
        with pytest.raises(RuntimeError, match="HTTP 401"):
            openhands.write_or_patch_config("http://127.0.0.1:8001", "model")



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
        port=None,
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
        assert len(ids) == len(set(ids)) == 14
        assert "deepseek-harness" in ids
        assert ids[:2] == ["cline", "claude-code"]
        # Cursor is present, but as a guide-only profile: its provider
        # settings are not a config file we could write. See
        # ``agents/profiles/cursor.yaml``.
        cursor_target = next(t for t in targets if t["id"] == "cursor")
        assert cursor_target["kind"] == "adapter_profile"
        assert cursor_target["config_path"] is None
        assert "continue" not in ids and "continue-dev" not in ids
        assert {target["kind"] for target in targets} == {
            "config_writer",
            "adapter_profile",
        }
        writer = next(t for t in targets if t["id"] == "claude-code")
        assert writer["config_path"].startswith("~/")
        assert next(t for t in targets if t["id"] == "codex")["config_path"] is None

    def test_mac_fake_registry_matches_production_registry(self):
        """GUI golden flows must exercise the same kinds and order as Rapid."""
        from vllm_mlx.integrations import integration_targets_json

        root = Path(__file__).resolve().parents[1]
        fake = root / "apps" / "rapid-mac" / "scripts" / "fake-rapid-mlx.sh"
        output = subprocess.check_output(
            [str(fake), "launch", "list", "--json"], text=True
        )

        assert json.loads(output) == integration_targets_json()

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
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        before = list(data_dir.iterdir())

        launch_cli.launch_command(_make_args(client="cline", dry_run=True))
        out = capsys.readouterr().out
        assert "[dry-run]" in out
        assert "cline" in out
        # No file was created or modified.
        assert list(data_dir.iterdir()) == before

    def test_real_patch_writes_file(self, fake_home, capsys):
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        launch_cli.launch_command(_make_args(client="cline", model="qwen3.5-4b-4bit"))
        target = data_dir / "globalState.json"
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["actModeOpenAiModelId"] == "qwen3.5-4b-4bit"
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
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
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
        state = json.loads((data_dir / "globalState.json").read_text())
        assert state["openAiBaseUrl"] == "http://127.0.0.1:8102/v1"
        # PID file written.
        assert launch_cli.PID_FILE.read_text().strip() == "99999"

    def test_openhands_start_server_avoids_its_default_ingress(
        self, fake_home, monkeypatch, capsys
    ):
        TestOpenHands._install(fake_home)
        sent = TestOpenHands._capture(monkeypatch)
        fake_proc = MagicMock(pid=99997)

        with patch.object(subprocess, "Popen", return_value=fake_proc) as popen:
            launch_cli.launch_command(
                _make_args(
                    client="openhands",
                    model="qwen3.5-4b-4bit",
                    start_server=True,
                )
            )

        llm = json.loads(sent[0].data)["agent_settings_diff"]["llm"]
        assert llm["base_url"] == "http://127.0.0.1:8001/v1"
        assert popen.call_args[0][0][-2:] == ["--port", "8001"]

    def test_start_server_rejects_an_occupied_port_before_writing(
        self, fake_home, monkeypatch, capsys
    ):
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        monkeypatch.setattr(launch_cli, "_start_port_available", lambda _port: False)

        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(client="cline", start_server=True, port=8123)
            )

        assert excinfo.value.code == 1
        assert not (data_dir / "globalState.json").exists()
        assert "already in use" in capsys.readouterr().err

    def test_start_server_rejects_conflicting_url_and_port(self, fake_home, capsys):
        (fake_home / "cline-data").mkdir(parents=True)

        with pytest.raises(SystemExit) as excinfo:
            launch_cli.launch_command(
                _make_args(
                    client="cline",
                    start_server=True,
                    port=8123,
                    server_url="http://127.0.0.1:8124",
                )
            )

        assert excinfo.value.code == 2
        assert "different ports" in capsys.readouterr().err

    def test_api_key_is_passed_to_client_and_started_server(
        self, fake_home, capsys, monkeypatch
    ):
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        fake_proc = MagicMock()
        fake_proc.pid = 99998
        monkeypatch.setenv("RAPID_MLX_API_KEY", "shared-secret")
        with patch.object(subprocess, "Popen", return_value=fake_proc) as popen:
            launch_cli.launch_command(_make_args(client="cline", start_server=True))
        secrets = json.loads((data_dir / "secrets.json").read_text())
        assert secrets["openAiApiKey"] == "shared-secret"
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

    def test_uses_original_alias_when_resolved(self, fake_home, capsys):
        """When ``main()`` rewrites ``args.model`` from alias to HF id,
        the launch command should patch with the ORIGINAL alias so the
        IDE client requests the short name from rapid-mlx."""
        data_dir = fake_home / "cline-data"
        data_dir.mkdir(parents=True)
        ns = _make_args(client="cline", model="mlx-community/Qwen3.5-4B-MLX-4bit")
        # Simulate what ``main()`` does on the way in.
        ns._original_alias = "qwen3.5-4b-4bit"
        launch_cli.launch_command(ns)
        data = json.loads((data_dir / "globalState.json").read_text())
        assert data["actModeOpenAiModelId"] == "qwen3.5-4b-4bit"


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


def test_launch_help_lists_supported_clients():
    """The client help lists exactly the registry ids. Cursor and
    Continue.dev were removed from ``launch`` (they are not writable
    config files), so they must not reappear in the help text."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from vllm_mlx.launch.cli import register

    register(sub)
    # argparse wraps help text at arbitrary columns; normalize whitespace
    # before matching so the assertion doesn't depend on wrap width.
    help_text = " ".join(sub.choices["launch"].format_help().split())
    assert "Supported: cline, claude-code." in help_text
    assert "continue" not in help_text


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


def test_launch_rejects_api_key_on_command_line():
    """Secrets for launch must travel via RAPID_MLX_API_KEY, never argv."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    from vllm_mlx.launch.cli import register

    register(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["launch", "cline", "--api-key", "leaked-secret"])
