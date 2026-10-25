"""Day-two configurations stay private and never rewrite existing agent state."""

import hashlib
import http.client
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import urllib.error
from argparse import ArgumentParser, Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from rapid_mlx.agents import isolated


def _launch_child(returncode=0):
    child = MagicMock()
    child.__enter__.return_value = child
    child.wait.return_value = returncode
    return child


@pytest.fixture
def private_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def agent_clis_installed(monkeypatch):
    """These tests cover profile and server flows, not CLI installation.

    The missing-CLI preflight has its own tests; here every agent CLI is
    treated as installed so results do not depend on the host's PATH.
    """
    monkeypatch.setattr(
        "rapid_mlx.agents.cli_preflight.agent_cli_found",
        lambda binary: True,
    )


def test_stable_key_is_reused_and_owner_only(private_home):
    first = isolated.stable_key()
    assert len(first) == 64
    assert isolated.stable_key() == first
    assert stat.S_IMODE(isolated.key_path().stat().st_mode) == 0o600
    assert stat.S_IMODE(isolated.connection_dir().stat().st_mode) == 0o700


def test_profiles_are_isolated_idempotent_and_removable(private_home):
    existing = private_home / ".codex" / "config.toml"
    existing.parent.mkdir()
    existing.write_text('model = "my-remote-model"\n')
    for name in sorted(isolated.SUPPORTED):
        command = isolated.configure(name, "qwen3.5-9b-4bit")
        assert name.split("-")[0] in command
        root = isolated.connection_dir() / "agents" / name
        before = {
            p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
        assert isolated.configure(name, "qwen3.5-9b-4bit") == command
        assert before == {
            p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
        }
        assert all(
            stat.S_IMODE(p.stat().st_mode)
            == (
                0o700
                if p.name in ("run", "git", "crush-launch", "aider-launch")
                or p.parent.name == "shell"
                else 0o600
            )
            for p in root.rglob("*")
            if p.is_file()
        )
        assert isolated.remove(name)
        assert not isolated.remove(name)
    assert existing.read_text() == 'model = "my-remote-model"\n'


def test_codex_profile_and_claude_settings(private_home):
    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    root = isolated.connection_dir() / "agents"
    codex = tomllib.loads((root / "codex" / "rapid.config.toml").read_text())
    assert codex["model_provider"] == "rapid-mlx"
    assert codex["model"] == "qwen3.5-9b-4bit"
    assert codex["features"] == {
        "apps": False,
        "plugins": False,
        "remote_plugin": False,
        "memories": False,
    }
    assert codex["memories"] == {"use_memories": False, "generate_memories": False}
    assert codex["skills"]["max_context_tokens"] == 256
    assert (
        codex["model_providers"]["rapid-mlx"]["base_url"] == "http://127.0.0.1:7660/v1"
    )
    isolated.configure("claude-code", "qwen3.5-9b-4bit", port=7660)
    claude = json.loads((root / "claude-code" / "settings.json").read_text())
    assert claude["env"]["ANTHROPIC_API_KEY"] == ""
    assert claude["env"]["ANTHROPIC_AUTH_TOKEN"] == isolated.stable_key()
    assert claude["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:7660"


def test_claude_launch_overrides_stored_subscription_without_touching_it(private_home):
    # A saved login must not be read or rewritten by launch. The private
    # settings give the CLI a Bearer credential that takes precedence.
    state = private_home / ".claude.json"
    saved = '{"oauthAccount":{"accountUuid":"simulated-login"}}\n'
    state.write_text(saved)
    command = isolated.configure("claude-code", "local-model", port=7660)
    settings = json.loads(
        (isolated.connection_dir() / "agents/claude-code/settings.json").read_text()
    )
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == isolated.stable_key()
    assert settings["env"]["ANTHROPIC_API_KEY"] == ""
    runner = (isolated.connection_dir() / "agents/claude-code/run").read_text()
    assert "--settings" in runner
    assert "--tools Bash,Edit,Read,Write,Glob,Grep" in runner
    assert (
        f"CLAUDE_CONFIG_DIR={isolated.connection_dir() / 'agents/claude-code/config'}"
        in runner
    )
    assert command.endswith("/claude-code/run")
    assert state.read_text() == saved
    assert not (private_home / ".claude").exists()


def test_claude_launch_rejects_symlinked_private_config(private_home):
    target = private_home / ".claude"
    target.mkdir()
    root = isolated.connection_dir() / "agents/claude-code"
    root.mkdir(parents=True)
    (root / "config").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="refusing symlinked config path"):
        isolated.configure("claude-code", "local-model", port=7660)

    assert not list(target.iterdir())


def test_codex_launcher_pins_the_local_model(private_home):
    # A saved TUI model can override the profile value. The executable flags
    # must still choose the exact local model advertised by Rapid-MLX.
    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    root = isolated.connection_dir() / "agents" / "codex"
    config = tomllib.loads((root / "rapid.config.toml").read_text())
    runner = (root / "run").read_text()
    assert config["model"] == "qwen3.5-9b-4bit"
    assert "codex --profile rapid --model qwen3.5-9b-4bit" in runner


def test_saved_port_is_reused_by_later_profiles(private_home):
    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    assert isolated.stable_port() == 7660
    command = isolated.configure("qwen-code", "qwen3.5-9b-4bit")
    assert command.endswith("/qwen-code/run")
    qwen_run = (isolated.connection_dir() / "agents" / "qwen-code" / "run").read_text()
    assert f"$(cat {isolated.port_path()})" in qwen_run
    hermes = isolated.configure("hermes", "qwen3.5-9b-4bit")
    assert hermes.endswith("/hermes/run")
    assert (
        "-p rapid"
        in (isolated.connection_dir() / "agents" / "hermes" / "run").read_text()
    )
    assert (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).is_file()
    assert (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / ".rapid-bundled-skills-cleaned"
    ).is_file()


def test_every_runner_refreshes_private_endpoint_after_port_change(private_home):
    for name in sorted(isolated.SUPPORTED):
        isolated.configure(name, "qwen3.5-9b-4bit", port=7660)
        runner = (isolated.connection_dir() / "agents" / name / "run").read_text()
        assert f"launch {name} --isolated --check" in runner
        if name == "hermes":
            assert runner.count("launch hermes --isolated") == 1
        else:
            assert f"launch {name} --isolated --model qwen3.5-9b-4bit" in runner


@pytest.mark.parametrize(
    "model,port", [("qwen3.5-4b-4bit", 8724), ("local/model", 7660)]
)
def test_hermes_provider_is_selected_from_private_config(private_home, model, port):
    isolated.configure("hermes", model, port=port)
    root = isolated.connection_dir() / "agents" / "hermes"
    config = yaml.safe_load((root / "profiles" / "rapid" / "config.yaml").read_text())
    assert config["model"]["provider"] == "custom"
    assert config["model"]["default"] == model
    assert config["model"]["base_url"] == f"http://127.0.0.1:{port}/v1"
    assert config["model"]["api_key"] == isolated.stable_key()
    assert config["model"]["context_length"] >= 64000
    command = shlex.split((root / "run").read_text().splitlines()[-1])
    assert f"HERMES_HOME={root}" in command
    # The CLI's provider choices omit custom even though its YAML accepts it.
    # Omit the override so the private config wins over ambient provider settings.
    assert command[command.index("hermes") :] == [
        "hermes",
        "-p",
        "rapid",
        "chat",
        "--model",
        model,
    ]


def test_old_opencode_shape_and_hermes_auth(private_home):
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="1.0.0"
    ):
        isolated.configure("opencode", "qwen3.5-9b-4bit")
    config = json.loads(
        (
            isolated.connection_dir() / "agents" / "opencode" / "opencode.json"
        ).read_text()
    )
    assert "providers" not in config
    assert config["model"] == "rapid-mlx/qwen3.5-9b-4bit"
    runner = (isolated.connection_dir() / "agents" / "opencode" / "command").read_text()
    assert "OPENCODE_CONFIG=" in runner
    assert (
        f"HOME={isolated.connection_dir() / 'agents' / 'opencode' / 'home'}" in runner
    )
    assert "XDG_CONFIG_HOME=" in runner
    assert "XDG_DATA_HOME=" in runner
    assert "env -u OPENCODE_CONFIG_DIR" in runner
    assert "OPENCODE_CONFIG_CONTENT=" in runner
    assert runner.index("OPENCODE_CONFIG_CONTENT=") < runner.index("env -u")
    assert "opencode --model" not in runner
    isolated.configure("hermes", "qwen3.5-9b-4bit")
    hermes = (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).read_text()
    assert f'api_key: "{isolated.stable_key()}"' in hermes
    assert yaml.safe_load(hermes)["platform_toolsets"]["cli"] == ["terminal", "file"]
    assert (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / ".no-bundled-skills"
    ).is_file()
    assert not (isolated.connection_dir() / "agents" / "hermes" / "auth.json").exists()
    goose = isolated.configure("goose", "qwen3.5-9b-4bit")
    assert "goose session --provider" in Path(goose).read_text()


def test_opencode_commands_see_real_home_while_config_stays_private(
    tmp_path, monkeypatch
):
    home = tmp_path / "Real User Home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    user_xdg = home / "my config"
    (user_xdg / "gh").mkdir(parents=True)
    (user_xdg / "gh" / "hosts.yml").write_text("login: real\n")
    (home / ".gitconfig").write_text(
        "[user]\n  name = Real Developer\n  email = real@example.invalid\n"
    )
    project = tmp_path / "project dir"
    project.mkdir()
    monkeypatch.chdir(project)
    commands = tmp_path / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    agent_env = tmp_path / "agent-env"
    child_env = tmp_path / "child-env"
    # Stand-in for OpenCode: record its own environment, then start a command
    # through $SHELL with -c, as its shell tool does.
    probe = commands / "opencode"
    probe.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" '
        f'"$OPENCODE_CONFIG" "$SHELL" > "{agent_env}"\n'
        'exec "$SHELL" -c \'printf "%s\\n" "$HOME" "${XDG_CONFIG_HOME-unset}" '
        '"${XDG_DATA_HOME-unset}" "$SHELL" "$(git config --global user.email)" '
        f'"$(cat "$XDG_CONFIG_HOME/gh/hosts.yml")" > "{child_env}"\'\n'
    )
    probe.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(user_xdg))
    for variable in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        monkeypatch.delenv(variable, raising=False)
    for variable in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_COUNT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    with (
        patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)),
        patch(
            "rapid_mlx.agents.opencode_version.installed_version",
            return_value="2.1.2",
        ),
    ):
        isolated.configure("opencode", "qwen3.5-9b-4bit", port=7660)
    root = isolated.connection_dir() / "agents" / "opencode"
    wrapper = root / "shell" / "sh"
    assert stat.S_IMODE(wrapper.stat().st_mode) == 0o700
    subprocess.run(
        [shutil.which("zsh") or "/bin/bash", str(root / "run")],
        cwd=project,
        check=True,
    )
    assert agent_env.read_text().splitlines() == [
        str(root / "home"),
        str(root / "xdg-config"),
        str(root / "xdg-data"),
        str(root / "opencode.json"),
        str(wrapper),
    ]
    assert child_env.read_text().splitlines() == [
        str(home),
        str(user_xdg),
        "unset",
        "/bin/sh",
        "real@example.invalid",
        "login: real",
    ]
    assert not (home / ".config").exists()
    assert not (user_xdg / "opencode").exists()


def test_opencode_shell_falls_back_from_unsupported_login_shell(
    private_home, monkeypatch
):
    monkeypatch.setenv("SHELL", "/usr/local/bin/fish")
    monkeypatch.setenv("XDG_STATE_HOME", str(private_home / "state dir"))
    monkeypatch.setattr(isolated.shutil, "which", lambda name: None)
    isolated.configure("opencode", "qwen3.5-9b-4bit")
    root = isolated.connection_dir() / "agents" / "opencode"
    wrapper = (root / "shell" / "sh").read_text()
    assert (
        f"XDG_STATE_HOME='{private_home / 'state dir'}'; export XDG_STATE_HOME"
        in wrapper
    )
    assert "SHELL=/bin/sh; export SHELL" in wrapper
    assert wrapper.endswith('exec /bin/sh "$@"\n')
    assert f"SHELL={root / 'shell' / 'sh'} " in (root / "command").read_text()


def test_opencode_runner_home_exists_and_survives_refresh(private_home):
    isolated.configure("opencode", "qwen3.5-9b-4bit")
    home = isolated.connection_dir() / "agents" / "opencode" / "home"
    assert home.is_dir()
    assert home.stat().st_mode & 0o777 == 0o700
    marker = home / "session-marker"
    marker.write_text("keep")

    isolated.configure("opencode", "qwen3.5-9b-4bit")
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("cleanup_outcome", ["clean", "timeout", "nonzero", "leftover"])
def test_existing_private_hermes_skills_are_opted_out(private_home, cleanup_outcome):
    root = isolated.connection_dir() / "agents" / "hermes"
    skills = root / "profiles" / "rapid" / "skills"
    skills.mkdir(parents=True)
    bundled_skill = skills / "SKILL.md"
    bundled_skill.write_text("bundled skill")
    with (
        patch("rapid_mlx.agents.isolated.shutil.which", return_value="hermes"),
        patch("rapid_mlx.agents.isolated.subprocess.run") as cleanup,
    ):
        if cleanup_outcome == "timeout":
            cleanup.side_effect = subprocess.TimeoutExpired("hermes", 120)
        else:
            cleanup.return_value.returncode = 1 if cleanup_outcome == "nonzero" else 0
            if cleanup_outcome == "clean":
                cleanup.side_effect = lambda *_args, **_kwargs: (
                    bundled_skill.unlink() or cleanup.return_value
                )
        for _ in range(2):
            if cleanup_outcome in {"timeout", "nonzero"}:
                with pytest.raises(ValueError, match="skill cleanup did not complete"):
                    isolated.configure("hermes", "qwen3.5-9b-4bit")
            else:
                isolated.configure("hermes", "qwen3.5-9b-4bit")
    assert cleanup.call_count == (2 if cleanup_outcome in {"timeout", "nonzero"} else 1)
    assert cleanup.call_args.kwargs["env"]["HERMES_HOME"] == str(root)
    assert (root / "profiles" / "rapid" / ".no-bundled-skills").is_file() == (
        cleanup_outcome not in {"timeout", "nonzero"}
    )
    assert (
        root / "profiles" / "rapid" / ".rapid-bundled-skills-cleaned"
    ).is_file() == (cleanup_outcome not in {"timeout", "nonzero"})


def test_existing_private_hermes_skills_require_cli(private_home):
    skills = (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "skills"
    )
    skills.mkdir(parents=True)
    with (
        patch("rapid_mlx.agents.isolated.shutil.which", return_value=None),
        pytest.raises(ValueError, match="needs the Hermes CLI"),
    ):
        isolated.configure("hermes", "qwen3.5-9b-4bit")


def test_hermes_uses_served_context_ceiling_before_opening(private_home):
    from rapid_mlx.launch.cli import launch_command

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {
                            "id": "qwen3.5-9b-4bit",
                            "context_window": 262144,
                            "max_model_len": 154524,
                        }
                    ]
                }
            ).encode()

    with (
        patch("urllib.request.urlopen", return_value=Response()),
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=0),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()) as opened,
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(_launch_args("hermes", "--model", "qwen3.5-9b-4bit"))
    config = (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).read_text()
    assert yaml.safe_load(config)["model"]["context_length"] == 131072
    opened.assert_called_once()


def test_codex_private_config_and_catalog_use_live_served_window(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"qwen3.5-9b-4bit","context_window":262144,"max_model_len":98304}]}'

    with patch("urllib.request.urlopen", return_value=Response()):
        isolated.configure(
            "codex", "qwen3.5-9b-4bit", port=7660, require_live_context=True
        )
    root = isolated.connection_dir() / "agents" / "codex"
    config = tomllib.loads((root / "rapid.config.toml").read_text())
    catalog = json.loads((root / "rapid-mlx-model-catalog.json").read_text())
    assert config["model_context_window"] == 98304
    assert catalog["models"][0]["context_window"] == 98304


def test_codex_live_preflight_requires_served_limit(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"qwen3.5-9b-4bit","context_window":262144}]}'

    with (
        patch("urllib.request.urlopen", return_value=Response()),
        pytest.raises(ValueError, match="ready local server"),
    ):
        isolated.configure(
            "codex", "qwen3.5-9b-4bit", port=7660, require_live_context=True
        )


def test_codex_offline_profile_uses_provisional_window(private_home):
    with patch("urllib.request.urlopen", side_effect=OSError):
        isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    root = isolated.connection_dir() / "agents" / "codex"
    config = tomllib.loads((root / "rapid.config.toml").read_text())
    catalog = json.loads((root / "rapid-mlx-model-catalog.json").read_text())
    assert config["model_context_window"] == 32768
    assert catalog["models"][0]["context_window"] == 32768


def test_codex_saved_runner_check_stops_without_live_limit(private_home):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.agents.isolated.served_context_length", return_value=None),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args("codex", "--isolated", "--check", "--model", "qwen3.5-9b-4bit")
        )


def test_hermes_rejects_served_window_below_minimum(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"small","context_window":32768,"max_model_len":32768}]}'

    with (
        patch("urllib.request.urlopen", return_value=Response()),
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=0),
        patch("rapid_mlx.launch.cli.Popen") as opened,
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("hermes", "--model", "small"))
    assert "at least 64,000" in capsys.readouterr().err
    opened.assert_not_called()


def test_hermes_requires_live_limit_at_launch(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch(
            "rapid_mlx.agents.isolated.hermes_context_length",
            side_effect=[64_000, ValueError("model card has no context limit")],
        ),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=0),
        patch("rapid_mlx.launch.cli.Popen") as opened,
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("hermes", "--model", "qwen3.5-9b-4bit"))
    assert "model card has no context limit" in capsys.readouterr().err
    opened.assert_not_called()


def test_hermes_uses_cached_window_only_while_server_is_stopped(private_home):
    with (
        patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()),
        patch("rapid_mlx.run.cli._cached_context_window", return_value=262144),
    ):
        assert isolated.hermes_context_length(7660, "a" * 64, "model") == 131072
        with pytest.raises(ValueError, match="ready local server"):
            isolated.hermes_context_length(7660, "a" * 64, "model", require_live=True)


def test_hermes_ignores_unrelated_single_model_listing(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"another-model","context_window":32768,"max_model_len":32768}]}'

    with (
        patch("urllib.request.urlopen", return_value=Response()),
        patch("rapid_mlx.run.cli._cached_context_window", return_value=262144),
    ):
        assert isolated.hermes_context_length(7660, "a" * 64, "wanted-model") == 131072
        with pytest.raises(ValueError, match="ready local server"):
            isolated.hermes_context_length(
                7660, "a" * 64, "wanted-model", require_live=True
            )


def test_hermes_failed_preflight_stops_only_new_server(private_home):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch(
            "rapid_mlx.agents.isolated.hermes_context_length",
            side_effect=[64_000, ValueError("served limit too short")],
        ),
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "ready"],
        ),
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=0),
        patch("rapid_mlx.run.cli._confirm_download", return_value=True),
        patch("rapid_mlx.cli._ensure_model_downloaded"),
        patch("rapid_mlx.launch.cli._start_server_background", return_value=12345),
        patch("rapid_mlx.launch.cli._stop_failed_start") as stopped,
        patch("rapid_mlx.launch.cli.Popen") as opened,
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args(
                "hermes",
                "--isolated",
                "--run",
                "--start-server",
                "--model",
                "qwen3.5-9b-4bit",
            )
        )
    stopped.assert_called_once_with(12345)
    opened.assert_not_called()


def test_hermes_check_refreshes_only_from_live_served_limit(private_home):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch(
            "rapid_mlx.agents.isolated.hermes_context_length", return_value=98304
        ) as probe,
    ):
        launch_command(
            _launch_args(
                "hermes", "--isolated", "--check", "--model", "qwen3.5-9b-4bit"
            )
        )
    assert probe.call_args.kwargs["require_live"] is True
    config = (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).read_text()
    assert yaml.safe_load(config)["model"]["context_length"] == 98304


def test_hermes_live_configuration_probes_once(private_home):
    with patch(
        "rapid_mlx.agents.isolated.hermes_context_length", return_value=98304
    ) as probe:
        isolated.configure("hermes", "model", port=7660, require_live_context=True)
    probe.assert_called_once()
    assert probe.call_args.kwargs["require_live"] is True
    config = (
        isolated.connection_dir()
        / "agents"
        / "hermes"
        / "profiles"
        / "rapid"
        / "config.yaml"
    ).read_text()
    assert yaml.safe_load(config)["model"]["context_length"] == 98304


def test_hermes_live_check_rejects_missing_served_ceiling(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"model","context_window":262144}]}'

    with (
        patch("urllib.request.urlopen", return_value=Response()),
        pytest.raises(ValueError, match="ready local server"),
    ):
        isolated.hermes_context_length(7660, "a" * 64, "model", require_live=True)


def test_hermes_saved_runner_check_stops_on_missing_limit(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch(
            "rapid_mlx.agents.isolated.hermes_context_length",
            side_effect=ValueError("served limit unavailable"),
        ),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args("hermes", "--isolated", "--check", "--model", "model")
        )
    assert "served limit unavailable" in capsys.readouterr().err


def test_new_opencode_model_comes_from_private_config(private_home):
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="2.0.26"
    ):
        isolated.configure("opencode", "qwen3.5-9b-4bit")
    root = isolated.connection_dir() / "agents" / "opencode"
    config = json.loads((root / "opencode.json").read_text())
    assert config["model"] == "rapid-mlx/qwen3.5-9b-4bit"
    assert "rapid-mlx" in config["providers"]
    runner = (root / "command").read_text()
    assert "OPENCODE_CONFIG=" in runner
    assert "OPENCODE_CONFIG_CONTENT=" in runner
    assert "opencode --standalone" in runner
    assert "opencode --model" not in runner
    smoke = isolated._command(
        "opencode", root, "qwen3.5-9b-4bit", 7660, version="2.1.2", opencode_run=True
    )
    assert smoke.endswith("opencode run --standalone")


def test_old_opencode_runner_keeps_v1_cli_flags(private_home):
    with patch(
        "rapid_mlx.agents.opencode_version.installed_version", return_value="1.0.0"
    ):
        isolated.configure("opencode", "qwen3.5-9b-4bit")
    runner = (isolated.connection_dir() / "agents" / "opencode" / "command").read_text()
    assert "opencode --standalone" not in runner
    smoke = isolated._command(
        "opencode",
        isolated.connection_dir() / "agents" / "opencode",
        "qwen3.5-9b-4bit",
        7660,
        version="1.0.0",
        opencode_run=True,
    )
    assert smoke.endswith("opencode run")


def test_isolated_dry_run_preserves_saved_profile(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    isolated.configure("codex", "qwen3.5-9b-4bit")
    args = Namespace(
        client="codex", isolated=False, remove=True, all=False, dry_run=True, run=False
    )
    launch_command(args)
    assert (
        isolated.connection_dir() / "agents" / "codex" / "rapid.config.toml"
    ).exists()
    assert "[dry-run]" in capsys.readouterr().out


def test_connection_requires_exact_advertised_model(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":[{"id":"qwen3.5-9b-4bit-WRONG"}]}'

    with patch("urllib.request.urlopen", return_value=Response()) as request:
        assert (
            isolated.connection_status(7659, "a" * 64, "qwen3.5-9b-4bit")
            == "model-mismatch"
        )
        assert (
            request.call_args.args[0].get_header("Authorization")
            == "Bearer " + "a" * 64
        )


def test_malformed_model_list_returns_one_server_error(private_home):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b'{"data":null}'

    with patch("urllib.request.urlopen", return_value=Response()):
        assert isolated.connection_status(7659, "a" * 64, "model") == "server-error"


def test_connection_status_shares_local_refusal_classification(private_home):
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
        assert isolated.connection_status(7659, "a" * 64, "model") == "stopped"
    with patch("urllib.request.urlopen", side_effect=TimeoutError()):
        assert isolated.connection_status(7659, "a" * 64, "model") == "server-error"


def test_launch_waits_through_startup_http_error(private_home):
    from rapid_mlx.launch.cli import launch_command, register

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    register(subparsers)
    args = parser.parse_args(
        [
            "launch",
            "codex",
            "--isolated",
            "--model",
            "qwen3.5-9b-4bit",
            "--run",
            "--start-server",
        ]
    )
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "server-error", "ready"],
        ),
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=0),
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.cli._ensure_model_downloaded"),
        patch("rapid_mlx.agents.isolated.served_context_length", return_value=98304),
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()) as agent,
        patch("time.sleep"),
        pytest.raises(SystemExit) as exit_info,
    ):
        launch_command(args)
    assert exit_info.value.code == 0
    start.assert_called_once()
    agent.assert_called_once()


def test_private_storage_rejects_links_and_unsafe_files(private_home):
    target = private_home / "other"
    target.mkdir()
    directory = isolated.connection_dir()
    directory.parent.mkdir()
    directory.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated._private_dir(directory)
    with pytest.raises(ValueError, match="symlink"):
        isolated.stable_key()
    directory.unlink()
    directory.mkdir()
    (directory / "api-key").symlink_to(target / "key")
    with pytest.raises(ValueError, match="symlink"):
        isolated.stable_key()
    (directory / "api-key").unlink()
    key = directory / "api-key"
    key.write_text("bad\n")
    key.chmod(0o600)
    with pytest.raises(ValueError, match="invalid"):
        isolated.stable_key()
    key.write_text("a" * 64)
    key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        isolated.stable_key()
    key.unlink()
    assert isolated.stable_key(create=False) is None
    (directory / "port").write_text("0\n")
    with pytest.raises(ValueError, match="1..65535"):
        isolated.stable_port()
    with pytest.raises(ValueError, match="1..65535"):
        isolated.save_port(0)
    (directory / "port").unlink()
    (directory / "port").symlink_to(target / "missing-port")
    with pytest.raises(ValueError, match="symlink"):
        isolated.stable_port()


def test_private_write_rejects_file_link_and_closes_on_failure(private_home):
    destination = isolated.connection_dir() / "test"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(private_home / "other")
    with pytest.raises(ValueError, match="symlinked config file"):
        isolated._private_write(destination, "secret")
    destination.unlink()
    with (
        patch("rapid_mlx.agents.isolated.os.fchmod", side_effect=OSError("denied")),
        pytest.raises(OSError, match="denied"),
    ):
        isolated._private_write(destination, "secret")


def test_key_creation_race_reuses_winner(private_home):
    original_open = os.open

    def raced_open(path, flags, mode):
        if path == isolated.key_path():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("a" * 64 + "\n")
            path.chmod(0o600)
            raise FileExistsError(path)
        return original_open(path, flags, mode)

    with patch("rapid_mlx.agents.isolated.os.open", side_effect=raced_open):
        assert isolated.stable_key() == "a" * 64


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.HTTPError("url", 401, "no", {}, None), "key-mismatch"),
        (urllib.error.HTTPError("url", 503, "loading", {}, None), "server-error"),
        (urllib.error.URLError(ConnectionRefusedError()), "stopped"),
        (ValueError("bad JSON"), "server-error"),
    ],
)
def test_connection_status_categorizes_failures(private_home, error, expected):
    with patch("urllib.request.urlopen", side_effect=error):
        assert isolated.connection_status(7659, "a" * 64, "model") == expected


def test_configure_and_remove_refuse_unowned_paths(private_home, monkeypatch):
    with pytest.raises(ValueError, match="unsupported"):
        isolated.configure("not-an-agent", "model")
    with pytest.raises(ValueError, match="unsupported"):
        isolated.remove("not-an-agent")
    root = isolated.connection_dir() / "agents" / "codex"
    root.parent.mkdir(parents=True)
    root.symlink_to(private_home)
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated.remove("codex")
    root.unlink()
    root.mkdir()
    (root / "config.toml").write_text("old")
    isolated.configure("codex", "qwen3.5-9b-4bit")
    assert not (root / "config.toml").exists()
    (root / "config.toml").symlink_to(private_home / "other")
    with pytest.raises(ValueError, match="symlinked config file"):
        isolated.configure("codex", "qwen3.5-9b-4bit")
    monkeypatch.setenv("RAPID_MLX_API_KEY", "original")
    (root / "config.toml").unlink()
    isolated.configure("codex", "qwen3.5-9b-4bit")
    assert os.environ["RAPID_MLX_API_KEY"] == "original"


def test_runner_command_rejects_unknown_tool_and_uses_current_binary(
    private_home, monkeypatch
):
    with pytest.raises(ValueError, match="unsupported"):
        isolated._command("unknown", private_home, "model", 7659)
    binary = private_home / "rapid-mlx"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setattr("sys.argv", [str(binary)])
    assert isolated._cli_invocation() == str(binary)
    monkeypatch.setattr(shutil, "which", lambda _: str(binary))
    assert isolated._display_cli_invocation() == "rapid-mlx"
    monkeypatch.setattr(shutil, "which", lambda _: "/different/rapid-mlx")
    assert isolated._display_cli_invocation() == str(binary)


def _launch_args(*argv):
    from rapid_mlx.launch.cli import register

    parser = ArgumentParser()
    register(parser.add_subparsers())
    return parser.parse_args(["launch", *argv])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["cline", "--run"], "require --isolated"),
        (["--isolated"], "name one agent"),
        (["codex", "--isolated", "--check"], "requires --model"),
        (["codex", "--isolated", "--check", "--model", "m"], "key is missing"),
        (["unknown", "--isolated", "--model", "m"], "unknown agent"),
    ],
)
def test_isolated_cli_rejects_incomplete_requests(private_home, capsys, argv, message):
    from rapid_mlx.launch.cli import launch_command

    with pytest.raises(SystemExit):
        launch_command(_launch_args(*argv))
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ready", None),
        ("stopped", "Start it with"),
        ("model-mismatch", "does not match"),
    ],
)
def test_isolated_cli_checks_saved_endpoint(private_home, capsys, status, expected):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value=status),
        patch("rapid_mlx.agents.isolated.served_context_length", return_value=32768),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
    ):
        if expected:
            with pytest.raises(SystemExit, match="1"):
                launch_command(
                    _launch_args("codex", "--isolated", "--check", "--model", "m")
                )
            assert expected in capsys.readouterr().err
        else:
            launch_command(
                _launch_args("codex", "--isolated", "--check", "--model", "m")
            )


def test_isolated_cli_saves_warns_and_removes(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.recommendations.physical_ram_gb", return_value=18),
        patch("rapid_mlx.run.cli._fits_host", return_value=False),
    ):
        launch_command(
            _launch_args(
                "codex", "--isolated", "--model", "gemma-4-12b-it", "--port", "7659"
            )
        )
    output = capsys.readouterr()
    assert "Saved an isolated codex profile" in output.out
    assert "not recommended for this Mac's memory" in output.err
    assert "known to stall" in output.err
    launch_command(_launch_args("codex", "--remove"))
    assert "Removed codex" in capsys.readouterr().out
    launch_command(_launch_args("codex", "--remove"))
    assert "No saved profile" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "message"),
    [
        ("stopped", "Local server is stopped"),
        ("model-mismatch", "does not match"),
        ("server-error", "does not match"),
    ],
)
def test_isolated_run_reports_unavailable_server(private_home, capsys, status, message):
    from rapid_mlx.launch.cli import launch_command

    args = ["codex", "--isolated", "--run", "--model", "qwen3.5-9b-4bit"]
    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value=status),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args(*args))
    output = capsys.readouterr()
    assert message in output.out + output.err


def test_isolated_run_executes_when_ready(private_home):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.agents.isolated.served_context_length", return_value=98304),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()) as call,
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(
            _launch_args("codex", "--isolated", "--run", "--model", "qwen3.5-9b-4bit")
        )
    assert call.call_args.args[0][:2] == ["/bin/zsh", "-lc"]
    config = (
        isolated.connection_dir() / "agents" / "codex" / "rapid.config.toml"
    ).read_text()
    assert tomllib.loads(config)["model_context_window"] == 98304


def test_claude_launch_explains_folder_trust_without_pretrusting(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()),
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(
            _launch_args(
                "claude-code", "--isolated", "--run", "--model", "qwen3.5-9b-4bit"
            )
        )
    assert "Review the folder before choosing Yes" in capsys.readouterr().out
    settings = json.loads(
        (
            isolated.connection_dir() / "agents" / "claude-code" / "settings.json"
        ).read_text()
    )
    assert "trustedDirectories" not in settings


def test_running_model_reuses_single_checkpoint_with_two_advertised_ids():
    class Reply:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return self.payload

    with patch(
        "rapid_mlx.agents.isolated.urllib.request.urlopen",
        return_value=Reply(
            b'{"data":[null,{"id":"mlx-community/Qwen3.5-9B-4bit"},{"id":"qwen3.5-9b-4bit"},{"id":"audio","capabilities":["audio"]},{"id":"tier","anthropic_family_tier":"opus"}]}'
        ),
    ) as get:
        assert isolated.running_model(7660, "secret") == "qwen3.5-9b-4bit"
    assert get.call_args.args[0].get_header("Authorization") == "Bearer secret"
    with (
        patch(
            "rapid_mlx.agents.isolated.urllib.request.urlopen",
            return_value=Reply(
                b'{"data":[{"id":"qwen3.5-9b-4bit"},{"id":"qwen3.5-4b-4bit"}]}'
            ),
        ),
        pytest.raises(ValueError, match="multiple text checkpoints"),
    ):
        isolated.running_model(7660, "secret")
    with (
        patch("rapid_mlx.agents.isolated.urllib.request.urlopen", side_effect=OSError),
        pytest.raises(ValueError, match="could not verify"),
    ):
        isolated.running_model(7660, "secret")
    with (
        patch(
            "rapid_mlx.agents.isolated.urllib.request.urlopen",
            side_effect=http.client.IncompleteRead(b""),
        ),
        pytest.raises(ValueError, match="could not verify"),
    ):
        isolated.running_model(7660, "secret")
    with patch(
        "rapid_mlx.agents.isolated.urllib.request.urlopen",
        side_effect=urllib.error.URLError(ConnectionRefusedError()),
    ):
        assert isolated.running_model(7660, "secret") is None
    with (
        patch(
            "rapid_mlx.agents.isolated.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("url", 401, "denied", {}, None),
        ),
        pytest.raises(ValueError, match="could not verify"),
    ):
        isolated.running_model(7660, "secret")
    for payload, message in (
        (b"{}", "invalid model list"),
        (b'{"data":[]}', "text model"),
    ):
        with (
            patch(
                "rapid_mlx.agents.isolated.urllib.request.urlopen",
                return_value=Reply(payload),
            ),
            pytest.raises(ValueError, match=message),
        ):
            isolated.running_model(7660, "secret")
    with (
        patch(
            "rapid_mlx.agents.isolated.urllib.request.urlopen",
            return_value=Reply(b'{"data":[{"id":"custom"}]}'),
        ),
        patch("rapid_mlx.model_aliases.resolve_model", side_effect=ValueError),
        pytest.raises(ValueError, match="could not resolve"),
    ):
        isolated.running_model(7660, "secret")


def test_running_model_capability_and_checkpoint_floor():
    from types import SimpleNamespace

    from rapid_mlx.agents import get_profile
    from rapid_mlx.launch import cli as launch_cli

    codex = get_profile("codex")
    assert codex is not None
    with patch("rapid_mlx.model_auto_config.detect_model_config", return_value=None):
        assert not launch_cli._usable_running_model("unknown", codex)
    with patch(
        "rapid_mlx.model_auto_config.detect_model_config",
        return_value=SimpleNamespace(tool_call_parser=None, hf_path="repo/Model-9B"),
    ):
        assert not launch_cli._usable_running_model("model-9b-4bit", codex)
    with patch(
        "rapid_mlx.model_auto_config.detect_model_config",
        return_value=SimpleNamespace(
            tool_call_parser="hermes", hf_path="repo/Model-9B"
        ),
    ):
        assert not launch_cli._usable_running_model("model-4b-4bit", codex)
        assert launch_cli._usable_running_model("model-9b-4bit", codex)
        assert launch_cli._usable_running_model("repo/Model-9B", codex)
        assert launch_cli._usable_running_model("model-14B_Instruct", codex)
        assert not launch_cli._usable_running_model("gemma-4-12b-it-4bit", codex)
    with patch("rapid_mlx.model_aliases.resolve_model", side_effect=ValueError):
        assert launch_cli._same_checkpoint("custom", "custom")
        assert not launch_cli._same_checkpoint("custom", "other")


def test_short_launch_reuses_compatible_running_model(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch(
            "rapid_mlx.agents.isolated.running_model", return_value="qwen3.5-9b-4bit"
        ),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._select_model") as select,
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()),
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(_launch_args("goose"))
    select.assert_not_called()
    assert isolated.saved_model("goose") == "qwen3.5-9b-4bit"
    assert "Using running model" in capsys.readouterr().out


@pytest.mark.parametrize("explicit", [False, True])
def test_short_launch_preserves_saved_quantization(private_home, capsys, explicit):
    from rapid_mlx.launch.cli import launch_command

    isolated.configure("goose", "qwen3.5-9b-8bit")
    profile_dir = isolated.connection_dir() / "agents" / "goose"
    before = {p: p.read_bytes() for p in profile_dir.rglob("*") if p.is_file()}
    with (
        patch(
            "rapid_mlx.agents.isolated.running_model", return_value="qwen3.5-9b-4bit"
        ),
        patch("rapid_mlx.agents.isolated.connection_status", return_value="ready"),
        patch("rapid_mlx.run.cli._select_model") as select,
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()) as call,
        pytest.raises(SystemExit) as exit_info,
    ):
        launch_command(
            _launch_args("goose", *(["--model", "qwen3.5-9b-4bit"] if explicit else []))
        )
    select.assert_not_called()
    if explicit:
        assert exit_info.value.code == 0
        assert isolated.saved_model("goose") == "qwen3.5-9b-4bit"
        call.assert_called_once()
    else:
        assert exit_info.value.code == 1
        assert isolated.saved_model("goose") == "qwen3.5-9b-8bit"
        assert {
            p: p.read_bytes() for p in profile_dir.rglob("*") if p.is_file()
        } == before
        call.assert_not_called()
        assert "stop it before launching qwen3.5-9b-8bit" in capsys.readouterr().err


def test_short_launch_refuses_explicit_different_running_model(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch(
            "rapid_mlx.agents.isolated.running_model", return_value="qwen3.5-9b-4bit"
        ),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("goose", "--model", "qwen3.5-4b-4bit"))
    assert isolated.saved_model("goose") is None
    assert "stop it before launching" in capsys.readouterr().err


def test_short_launch_keeps_profile_untouched_when_server_cannot_be_verified(
    private_home, capsys
):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch(
            "rapid_mlx.agents.isolated.running_model",
            side_effect=ValueError("could not verify the running local server"),
        ),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("goose"))
    assert isolated.saved_model("goose") is None
    assert "could not verify" in capsys.readouterr().err


@pytest.mark.parametrize("explicit", [False, True])
def test_short_launch_rejects_unusable_saved_running_model(
    private_home, capsys, explicit
):
    from rapid_mlx.launch.cli import launch_command

    isolated.configure("codex", "qwen3.5-4b-4bit")
    with (
        patch(
            "rapid_mlx.agents.isolated.running_model", return_value="qwen3.5-4b-4bit"
        ),
        patch("rapid_mlx.model_auto_config.detect_model_config", return_value=None),
        patch("rapid_mlx.launch.cli.Popen") as call,
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args("codex", *(["--model", "qwen3.5-4b-4bit"] if explicit else []))
        )
    call.assert_not_called()
    assert "does not meet" in capsys.readouterr().err


@pytest.mark.parametrize("repeat_explicit", [False, True])
@pytest.mark.parametrize("active", ["qwen3.5-4b-8bit", "mlx-community/Qwen3.5-4B-8bit"])
def test_small_selected_model_first_launch_then_repeat(
    private_home, capsys, repeat_explicit, active
):
    from rapid_mlx.launch.cli import launch_command

    model = "qwen3.5-4b-8bit"
    with (
        patch("rapid_mlx.agents.isolated.running_model", return_value=active),
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "ready", "ready"],
        ),
        patch("rapid_mlx.agents.isolated.served_context_length", return_value=98304),
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.cli._ensure_model_downloaded") as download,
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()) as call,
        patch("rapid_mlx.run.cli._confirm_download", return_value=True),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
    ):
        # No key/server on first launch; configure persists the user's choice.
        with pytest.raises(SystemExit) as first:
            launch_command(_launch_args("codex", "--model", model))
        assert first.value.code == 0
        assert isolated.saved_model("codex") == model
        assert "recommended size" in capsys.readouterr().err
        with pytest.raises(SystemExit) as repeat:
            launch_command(
                _launch_args("codex", *(["--model", model] if repeat_explicit else []))
            )
        assert repeat.value.code == 0
    start.assert_called_once()
    download.assert_called_once()
    assert call.call_count == 2
    assert "recommended size" in capsys.readouterr().err


def test_small_running_model_without_user_choice_is_not_automatically_reused(
    private_home, capsys
):
    from rapid_mlx.launch.cli import launch_command

    isolated.stable_key()
    with (
        patch(
            "rapid_mlx.agents.isolated.running_model", return_value="qwen3.5-4b-8bit"
        ),
        patch(
            "rapid_mlx.run.cli._select_model", return_value="qwen3.5-9b-4bit"
        ) as select,
        patch("rapid_mlx.launch.cli.Popen") as call,
        pytest.raises(SystemExit) as result,
    ):
        launch_command(_launch_args("codex"))
    assert result.value.code == 1
    assert select.call_args.kwargs["explicit"] is None
    call.assert_not_called()
    assert "stop it before launching" in capsys.readouterr().err


def test_short_launch_reuses_saved_model_and_starts_stopped_server(
    private_home, capsys
):
    from rapid_mlx.launch.cli import launch_command

    isolated.configure("codex", "qwen3.5-9b-4bit", port=7660)
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "ready"],
        ),
        patch("rapid_mlx.agents.isolated.served_context_length", return_value=98304),
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.cli._ensure_model_downloaded") as download,
        patch("rapid_mlx.launch.cli.Popen", return_value=_launch_child()) as call,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="0"),
    ):
        launch_command(_launch_args("codex"))
    start.assert_called_once_with(
        "qwen3.5-9b-4bit", 7660, api_key=isolated.stable_key()
    )
    download.assert_called_once_with("mlx-community/Qwen3.5-9B-4bit")
    assert call.call_args.args[0][:2] == ["/bin/zsh", "-lc"]
    output = capsys.readouterr().out
    assert "launch codex" in output
    assert "Preparing qwen3.5-9b-4bit" in output
    assert "Loading qwen3.5-9b-4bit" in output


def test_short_launch_requires_download_consent_before_spawning(private_home):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.agents.isolated.connection_status", return_value="stopped"),
        patch("rapid_mlx.run.cli._confirm_download", return_value=False) as consent,
        patch("rapid_mlx.launch.cli._start_server_background") as start,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("codex", "--model", "qwen3.5-9b-4bit"))
    consent.assert_called_once_with("qwen3.5-9b-4bit", no_download=False, yes=False)
    start.assert_not_called()


def test_cancelled_model_start_terminates_only_spawned_server(
    private_home, monkeypatch
):
    from rapid_mlx.launch import cli as launch_cli

    pid_file = private_home / "launch.pid"
    pid_file.write_text("12345\n")
    monkeypatch.setattr(launch_cli, "PID_FILE", pid_file)
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", KeyboardInterrupt],
        ),
        patch("rapid_mlx.run.cli._confirm_download", return_value=True),
        patch("rapid_mlx.launch.cli._start_server_background", return_value=12345),
        patch("rapid_mlx.cli._ensure_model_downloaded"),
        patch("rapid_mlx.launch.cli.os.kill") as kill,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(KeyboardInterrupt),
    ):
        launch_cli.launch_command(_launch_args("codex", "--model", "qwen3.5-9b-4bit"))
    kill.assert_called_once_with(12345, signal.SIGTERM)
    assert not pid_file.exists()


def test_failed_start_cleanup_tolerates_exited_server_and_missing_pid_file(
    private_home, monkeypatch
):
    from rapid_mlx.launch import cli as launch_cli

    monkeypatch.setattr(launch_cli, "PID_FILE", private_home / "missing.pid")
    with patch("rapid_mlx.launch.cli.os.kill", side_effect=ProcessLookupError) as kill:
        launch_cli._stop_failed_start(12345)
    kill.assert_called_once_with(12345, signal.SIGTERM)


def test_started_server_with_wrong_key_is_stopped(private_home, monkeypatch, capsys):
    from rapid_mlx.launch import cli as launch_cli

    pid_file = private_home / "launch.pid"
    pid_file.write_text("12345\n")
    monkeypatch.setattr(launch_cli, "PID_FILE", pid_file)
    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", "key-mismatch"],
        ),
        patch("rapid_mlx.run.cli._confirm_download", return_value=True),
        patch("rapid_mlx.launch.cli._start_server_background", return_value=12345),
        patch("rapid_mlx.cli._ensure_model_downloaded"),
        patch("rapid_mlx.launch.cli.os.kill") as kill,
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_cli.launch_command(_launch_args("codex", "--model", "qwen3.5-9b-4bit"))
    kill.assert_called_once_with(12345, signal.SIGTERM)
    assert not pid_file.exists()
    assert "does not match this profile" in capsys.readouterr().err


def test_isolated_cli_reports_no_suitable_model(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch("rapid_mlx.run.cli._select_model", return_value=None),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(_launch_args("codex", "--isolated"))
    assert "no model fits this Mac" in capsys.readouterr().err


def test_isolated_run_times_out_during_startup(private_home, capsys):
    from rapid_mlx.launch.cli import launch_command

    with (
        patch(
            "rapid_mlx.agents.isolated.connection_status",
            side_effect=["stopped", *["server-error"] * 600],
        ),
        patch("rapid_mlx.launch.cli._start_server_background"),
        patch("rapid_mlx.cli._ensure_model_downloaded"),
        patch("rapid_mlx.run.cli._fits_host", return_value=True),
        patch("time.sleep"),
        pytest.raises(SystemExit, match="1"),
    ):
        launch_command(
            _launch_args(
                "codex",
                "--isolated",
                "--run",
                "--start-server",
                "--model",
                "qwen3.5-9b-4bit",
            )
        )
    assert "did not become ready" in capsys.readouterr().out


def test_no_setup_server_reuses_saved_key_without_overriding_explicit_env(
    private_home, monkeypatch
):
    from rapid_mlx.run.cli import _spawn_foreground_serve

    key = isolated.stable_key()
    captured = []
    args = Namespace(host="127.0.0.1", port=7659, no_setup=True, no_download=False)
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)

    def spawn(_command, **kwargs):
        captured.append(kwargs["env"].get("RAPID_MLX_API_KEY"))
        return object()

    with patch("rapid_mlx.run.cli.subprocess.Popen", side_effect=spawn):
        _spawn_foreground_serve("qwen3.5-9b-4bit", args)
        monkeypatch.setenv("RAPID_MLX_API_KEY", "explicit")
        _spawn_foreground_serve("qwen3.5-9b-4bit", args)
        monkeypatch.delenv("RAPID_MLX_API_KEY")
        args.no_setup = False
        _spawn_foreground_serve("qwen3.5-9b-4bit", args)
    assert captured == [key, "explicit", None]


@pytest.mark.parametrize("ancestor", ["connect", "agents"])
def test_profile_operations_reject_symlinked_ancestor(private_home, ancestor):
    external = private_home / "external"
    external.mkdir()
    marker = external / "do-not-delete"
    marker.write_text("keep")
    connect = isolated.connection_dir()
    agents = connect / "agents"
    if ancestor == "connect":
        connect.parent.mkdir()
        connect.symlink_to(external)
    else:
        connect.mkdir(parents=True)
        agents.symlink_to(external)
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated.configure("codex", "qwen3.5-9b-4bit")
    with pytest.raises(ValueError, match="symlinked config path"):
        isolated.remove("codex")
    assert marker.read_text() == "keep"
    assert not (external / "codex").exists()


def test_qwen_runner_reads_current_port_after_saved_port_moves(
    private_home, monkeypatch
):
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    qwen = commands / "qwen"
    qwen.write_text('#!/bin/sh\nprintf "%s" "$OPENAI_BASE_URL" > "$TEST_CAPTURE"\n')
    qwen.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    capture = private_home / "qwen-endpoint"
    monkeypatch.setenv("TEST_CAPTURE", str(capture))
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("qwen-code", "qwen3.5-9b-4bit", port=7660)
    isolated.save_port(7661)
    runner = isolated.connection_dir() / "agents" / "qwen-code" / "run"
    subprocess.run([shutil.which("zsh") or "/bin/bash", str(runner)], check=True)
    assert capture.read_text() == "http://127.0.0.1:7661/v1"


@pytest.mark.parametrize("model", ("qwen3.5-4b-8bit", "qwen3.5-9b-4bit"))
def test_qwen_runner_overrides_saved_cloud_auth_without_editing_it(
    private_home, monkeypatch, model
):
    settings = private_home / ".qwen/settings.json"
    settings.parent.mkdir()
    original = (
        b'{"security":{"auth":{"selectedType":"qwen-oauth"}},'
        b'"model":{"name":"cloud"},'
        b'"modelProviders":{"openai":[{"id":"qwen3.5-9b-4bit",'
        b'"baseUrl":"https://cloud.example/v1","envKey":"CLOUD_KEY"}]}}\n'
    )
    settings.write_bytes(original)
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    qwen = commands / "qwen"
    qwen.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_CAPTURE"\n'
        'printf "%s" "$QWEN_HOME" > "$TEST_QWEN_HOME_CAPTURE"\n'
        'printf "%s" "$QWEN_RUNTIME_DIR" > "$TEST_QWEN_RUNTIME_CAPTURE"\n'
    )
    qwen.chmod(0o700)
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    capture = private_home / "qwen-args"
    home_capture = private_home / "qwen-home"
    runtime_capture = private_home / "qwen-runtime"
    monkeypatch.setenv("QWEN_RUNTIME_DIR", str(private_home / "existing-runtime"))
    monkeypatch.setenv("TEST_CAPTURE", str(capture))
    monkeypatch.setenv("TEST_QWEN_HOME_CAPTURE", str(home_capture))
    monkeypatch.setenv("TEST_QWEN_RUNTIME_CAPTURE", str(runtime_capture))
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("qwen-code", model, port=7660)
    runner = isolated.connection_dir() / "agents" / "qwen-code" / "run"
    subprocess.run([shutil.which("zsh") or "/bin/bash", str(runner)], check=True)
    args = capture.read_text().splitlines()
    assert "--safe-mode" in args
    assert "--bare" not in args
    assert "--yolo" not in args
    assert "--approval-mode" not in args
    assert "--allowed-tools" not in args
    assert args[args.index("--auth-type") + 1] == "openai"
    assert args[args.index("--model") + 1] == model
    assert args[args.index("--openai-base-url") + 1] == "http://127.0.0.1:7660/v1"
    assert home_capture.read_text() == str(
        isolated.connection_dir() / "agents" / "qwen-code"
    )
    assert runtime_capture.read_text() == str(
        isolated.connection_dir() / "agents" / "qwen-code" / "runtime"
    )
    assert settings.read_bytes() == original


def test_crush_runner_uses_private_config_directory(private_home, monkeypatch):
    user_config = private_home / ".config/crush/crush.json"
    user_config.parent.mkdir(parents=True)
    original = b'{"models":{"large":{"provider":"cloud","model":"cloud"}}}\n'
    user_config.write_bytes(original)
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    crush = commands / "crush"
    crush.write_text(
        '#!/bin/sh\ntest -f "$CRUSH_GLOBAL_CONFIG/crush.json" || exit 8\n'
        'printf "%s" "$CRUSH_GLOBAL_CONFIG" > "$TEST_CAPTURE"\n'
        'printf "%s" "$CRUSH_GLOBAL_DATA" > "$TEST_DATA_CAPTURE"\n'
        'printf "%s\\n" "$@" > "$TEST_ARGS_CAPTURE"\n'
        'printf "%s" "${GIT_DIR:-}" > "$TEST_GIT_CAPTURE"\n'
        'printf "%s" "${GIT_AUTHOR_NAME:-}" > "$TEST_AUTHOR_CAPTURE"\n'
    )
    crush.chmod(0o700)
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    capture = private_home / "crush-config-dir"
    args_capture = private_home / "crush-args"
    data_capture = private_home / "crush-global-data"
    git_capture = private_home / "crush-git-scope"
    author_capture = private_home / "crush-author"
    monkeypatch.setenv("TEST_CAPTURE", str(capture))
    monkeypatch.setenv("TEST_ARGS_CAPTURE", str(args_capture))
    monkeypatch.setenv("TEST_DATA_CAPTURE", str(data_capture))
    monkeypatch.setenv("TEST_GIT_CAPTURE", str(git_capture))
    monkeypatch.setenv("TEST_AUTHOR_CAPTURE", str(author_capture))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-cloud-key")
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    config = json.loads(
        (isolated.connection_dir() / "agents/crush/crush.json").read_text()
    )
    assert config["models"]["large"]["provider"] == "rapid-mlx"
    assert config["models"]["small"]["provider"] == "rapid-mlx"
    assert config["models"]["small"]["model"] == "qwen3.5-9b-4bit"
    assert config["options"]["disable_default_providers"] is True
    runner = isolated.connection_dir() / "agents" / "crush" / "run"
    assert (
        str(isolated.connection_dir() / "agents/crush/bin/crush-launch")
        in runner.read_text()
    )
    monkeypatch.setenv("GIT_DIR", str(private_home / "wrong-repo"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Current Author")
    subprocess.run(["/bin/sh", str(runner)], check=True)
    assert git_capture.read_text() == ""
    assert author_capture.read_text() == "Current Author"
    assert capture.read_text() == str(isolated.connection_dir() / "agents/crush")
    assert data_capture.read_text() == str(
        isolated.connection_dir() / "agents/crush/global-data"
    )
    assert stat.S_IMODE(Path(data_capture.read_text()).stat().st_mode) == 0o700
    args = args_capture.read_text().splitlines()
    project_id = hashlib.sha256(str(Path.cwd().resolve()).encode()).hexdigest()[:24]
    assert args[args.index("--data-dir") + 1] == str(
        isolated.connection_dir() / "agents/crush/data" / project_id
    )
    assert (
        stat.S_IMODE(Path(args[args.index("--data-dir") + 1]).stat().st_mode) == 0o700
    )
    assert user_config.read_bytes() == original


def test_crush_isolated_launch_rejects_project_config_override(
    private_home, monkeypatch
):
    project = private_home / "project" / "nested"
    project.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(project.parent)], check=True)
    cloud_config = project.parent / ".crush.json"
    cloud_config.write_text('{"models":{"large":{"provider":"cloud","model":"cloud"}}}')
    monkeypatch.chdir(project)
    monkeypatch.setenv("GIT_DIR", str(private_home / "wrong-repo"))
    monkeypatch.setenv("GIT_WORK_TREE", str(private_home / "wrong-tree"))
    with pytest.raises(ValueError, match="project configuration"):
        isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    assert cloud_config.read_text().startswith('{"models"')


def test_crush_ignores_config_above_project_and_resets_saved_model(
    private_home, monkeypatch
):
    project = private_home / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    (private_home / "crush.json").write_text('{"models":{"large":{"model":"cloud"}}}')
    monkeypatch.chdir(project)
    isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    root = isolated.connection_dir() / "agents" / "crush"
    project_id = hashlib.sha256(str(project.resolve()).encode()).hexdigest()[:24]
    data = root / "data" / project_id
    assert stat.S_IMODE(data.stat().st_mode) == 0o700
    stale = {
        "models": {
            "large": {"model": "cloud", "provider": "cloud", "temperature": 0.4},
            "small": {"model": "cloud-small", "provider": "cloud", "max_tokens": 2048},
        },
        "providers": {"cloud": {"api_key": "fake"}},
        "options": {
            "data_directory": "/tmp/remote",
            "disable_default_providers": False,
            "compact_mode": True,
        },
        "theme": "dark",
    }
    (data / "crush.json").write_text(json.dumps(stale))
    (data / "crush.db").write_text("history")
    global_data = root / "global-data"
    global_data.mkdir(exist_ok=True)
    (global_data / "crush.json").write_text(json.dumps(stale))
    isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    for saved in (data / "crush.json", global_data / "crush.json"):
        assert json.loads(saved.read_text()) == {
            "models": {
                "large": {
                    "provider": "rapid-mlx",
                    "model": "qwen3.5-9b-4bit",
                    "temperature": 0.4,
                },
                "small": {
                    "provider": "rapid-mlx",
                    "model": "qwen3.5-9b-4bit",
                    "max_tokens": 2048,
                },
            },
            "options": {"compact_mode": True},
            "theme": "dark",
        }
    assert (data / "crush.db").read_text() == "history"


def test_crush_rejects_invalid_saved_config(private_home, monkeypatch):
    project = private_home / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    saved = (
        isolated.connection_dir() / "agents" / "crush" / "global-data" / "crush.json"
    )
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text('{"models": ["cloud"], "theme": "dark"}')
    isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    pinned = {
        slot: {"provider": "rapid-mlx", "model": "qwen3.5-9b-4bit"}
        for slot in ("large", "small")
    }
    assert json.loads(saved.read_text()) == {"models": pinned, "theme": "dark"}
    saved.write_text('{"models": {"large": {}}, "theme": "dark"}')
    isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    assert json.loads(saved.read_text()) == {"models": pinned, "theme": "dark"}
    saved.write_text("[]")
    with pytest.raises(ValueError, match="saved configuration"):
        isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    saved.write_text("{")
    with pytest.raises(ValueError, match="saved configuration.*crush.json"):
        isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)


def test_crush_data_is_separate_for_each_project(private_home, monkeypatch):
    projects = [private_home / "a", private_home / "b"]
    for project in projects:
        project.mkdir()
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    crush = commands / "crush"
    crush.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_ARGS_CAPTURE"\n')
    crush.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    monkeypatch.chdir(projects[0])
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("crush", "qwen3.5-9b-4bit", port=7660)
    runner = isolated.connection_dir() / "agents/crush/run"
    marker = private_home / "imported-project-module"
    (projects[1] / "hashlib.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
    )
    captured = []
    for project in projects:
        args_capture = private_home / f"{project.name}-args"
        monkeypatch.setenv("TEST_ARGS_CAPTURE", str(args_capture))
        subprocess.run(["/bin/sh", str(runner)], cwd=project, check=True)
        args = args_capture.read_text().splitlines()
        captured.append(args[args.index("--data-dir") + 1])
    assert captured[0] != captured[1]
    assert not marker.exists()
    for project, data_dir in zip(projects, captured):
        project_id = hashlib.sha256(str(project.resolve()).encode()).hexdigest()[:24]
        assert data_dir == str(
            isolated.connection_dir() / "agents/crush/data" / project_id
        )


@pytest.mark.parametrize("agent", ("crush", "aider"))
def test_private_helper_handles_spaced_interpreter_path(
    private_home, monkeypatch, agent
):
    project = private_home / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    capture = private_home / "launched"
    client = commands / agent
    client.write_text(f'#!/bin/sh\nprintf launched > "{capture}"\n')
    client.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    interpreter_dir = private_home / "Application Support"
    interpreter_dir.mkdir()
    interpreter = interpreter_dir / "python3"
    interpreter.symlink_to(sys.executable)
    with (
        patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)),
        patch("rapid_mlx.agents.isolated.sys.executable", str(interpreter)),
    ):
        isolated.configure(agent, "qwen3.5-9b-4bit", port=7660)
    runner = isolated.connection_dir() / "agents" / agent / "run"
    subprocess.run(["/bin/sh", str(runner)], check=True)
    assert capture.read_text() == "launched"


@pytest.mark.parametrize("model", ("qwen3.5-9b-4bit", "flint-40b-a2.5b-4bit"))
def test_aider_runner_ignores_existing_cloud_config(private_home, monkeypatch, model):
    user_config = private_home / ".aider.conf.yml"
    original = b"model: openai/cloud\nopenai-api-key: fake-cloud-key\n"
    user_config.write_bytes(original)
    (private_home / ".gitconfig").write_text("[user]\n  name = Real Developer\n")
    # Run from a project outside any repository, so a checkout's own
    # user.name cannot shadow the global identity this test asserts.
    project = private_home / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    args_capture = private_home / "aider-args"
    env_capture = private_home / "aider-env"
    git_capture = private_home / "aider-git-name"
    aider = commands / "aider"
    aider.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > "{args_capture}"\n'
        f'printf "%s\\n%s\\n%s\\n%s\\n" "$AIDER_OPENAI_API_KEY" "$AIDER_OPENAI_API_BASE" "$HOME" "$AIDER_MODEL_SETTINGS_FILE" > "{env_capture}"\n'
        f'git config user.name > "{git_capture}"\n'
    )
    aider.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    monkeypatch.setenv("AIDER_OPENAI_API_KEY", "fake-cloud-key")
    monkeypatch.setenv("AIDER_OPENAI_API_BASE", "https://cloud.invalid/v1")
    monkeypatch.setenv("AIDER_MODEL_SETTINGS_FILE", "cloud-models.yml")
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("aider", model, port=7660)
    runner = isolated.connection_dir() / "agents/aider/run"
    subprocess.run([shutil.which("zsh") or "/bin/bash", str(runner)], check=True)
    args = args_capture.read_text().splitlines()
    assert args[args.index("--config") + 1] == str(
        isolated.connection_dir() / "agents/aider/aider.conf.yml"
    )
    assert args[args.index("--openai-api-base") + 1] == "http://127.0.0.1:7660/v1"
    assert args[args.index("--weak-model") + 1] == f"openai/{model}"
    assert args[args.index("--editor-model") + 1] == f"openai/{model}"
    assert env_capture.read_text().splitlines() == [
        isolated.key_path().read_text().strip(),
        "http://127.0.0.1:7660/v1",
        str(isolated.connection_dir() / "agents/aider/home"),
        "",
    ]
    assert git_capture.read_text().strip() == "Real Developer"
    root = isolated.connection_dir() / "agents/aider"
    config = yaml.safe_load((root / "aider.conf.yml").read_text())
    assert config == {
        "model-settings-file": str(root / "aider.model.settings.yml"),
        "edit-format": "diff",
        "editor-edit-format": "editor-diff",
        "suggest-shell-commands": True,
        "auto-test": False,
    }
    assert "--yes" not in args and "--yes-always" not in args
    assert yaml.safe_load(Path(config["model-settings-file"]).read_text()) == [
        {
            "name": f"openai/{model}",
            "edit_format": "diff",
            "editor_edit_format": "editor-diff",
            "use_repo_map": True,
            "use_temperature": False,
        }
    ]
    private_env = (isolated.connection_dir() / "agents/aider/aider.env").read_text()
    assert (
        f"AIDER_OPENAI_API_KEY={isolated.key_path().read_text().strip()}" in private_env
    )
    assert "AIDER_OPENAI_API_BASE=http://127.0.0.1:7660/v1" in private_env
    assert user_config.read_bytes() == original


def test_aider_rejects_project_configuration_before_writing_state(
    private_home, monkeypatch
):
    project = private_home / "project"
    project.mkdir()
    config = project / ".aider.conf.yml"
    config.write_text("yes-always: true\n")
    monkeypatch.chdir(project)
    with pytest.raises(ValueError, match="project configuration"):
        isolated.configure("aider", "qwen3.5-9b-4bit", port=7660)
    assert config.read_text() == "yes-always: true\n"
    config.unlink()
    (project / ".env").write_text("AIDER_MODEL_SETTINGS_FILE=cloud-models.yml\n")
    with pytest.raises(ValueError, match="project configuration"):
        isolated.configure("aider", "qwen3.5-9b-4bit", port=7660)


def test_aider_git_shim_preserves_environment_only_commit_identity(
    private_home, monkeypatch
):
    project = private_home / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text("#!/bin/sh\nexit 0\n")
    rapid.chmod(0o700)
    aider = commands / "aider"
    aider.write_text("#!/bin/sh\ngit commit --allow-empty -m local-edit\n")
    aider.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Environment Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "author@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Environment Committer")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "committer@example.invalid")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "rapidmlxtest.empty")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "")
    monkeypatch.setenv("GIT_DIR", str(private_home / "wrong-repo"))
    monkeypatch.setenv("GIT_WORK_TREE", str(private_home / "wrong-tree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(private_home / "wrong-index"))
    monkeypatch.setenv("GIT_BAD-NAME", "should-not-enter-a-shell-script")
    monkeypatch.chdir(project)
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure("aider", "qwen3.5-9b-4bit", port=7660)
    runner = isolated.connection_dir() / "agents/aider/run"
    assert (
        "should-not-enter-a-shell-script"
        not in (isolated.connection_dir() / "agents/aider/bin/git").read_text()
    )
    assert (
        "Environment Author"
        not in (isolated.connection_dir() / "agents/aider/bin/git").read_text()
    )
    assert (
        "GIT_DIR="
        not in (isolated.connection_dir() / "agents/aider/bin/git").read_text()
    )
    assert (
        "GIT_WORK_TREE="
        not in (isolated.connection_dir() / "agents/aider/bin/git").read_text()
    )
    assert (
        "GIT_INDEX_FILE="
        not in (isolated.connection_dir() / "agents/aider/bin/git").read_text()
    )
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Current Author")
    subprocess.run([shutil.which("zsh") or "/bin/bash", str(runner)], check=True)
    for variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(variable)
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%an <%ae>|%cn <%ce>"], cwd=project, text=True
    ).strip()
    assert commit == (
        "Current Author <author@example.invalid>|"
        "Environment Committer <committer@example.invalid>"
    )


def test_aider_rejects_existing_home_config_from_home_directory(
    private_home, monkeypatch
):
    config = private_home / ".aider.conf.yml"
    config.write_text("model: cloud\n")
    (private_home / ".aider.model.settings.yml").write_text("- name: cloud\n")
    (private_home / ".env").write_text("OPENAI_API_BASE=https://cloud.invalid\n")
    monkeypatch.chdir(private_home)
    with pytest.raises(ValueError, match="project configuration"):
        isolated.configure("aider", "qwen3.5-9b-4bit", port=7660)
    assert config.read_text() == "model: cloud\n"
    assert not isolated.connection_dir().exists()


def test_aider_rejects_home_git_root_config_from_subdirectory(
    private_home, monkeypatch
):
    subprocess.run(["git", "init", "-q", str(private_home)], check=True)
    (private_home / ".aider.model.settings.yml").write_text("- name: cloud\n")
    project = private_home / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("GIT_DIR", str(private_home / "wrong-repo"))
    monkeypatch.setenv("GIT_WORK_TREE", str(private_home / "wrong-tree"))
    with pytest.raises(ValueError, match="project configuration"):
        isolated.configure("aider", "qwen3.5-9b-4bit", port=7660)


def test_aider_requires_git_for_private_home_shim(private_home, monkeypatch):
    monkeypatch.setattr("rapid_mlx.agents.isolated.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="requires git"):
        isolated.configure("aider", "qwen3.5-9b-4bit", port=7660)
    assert not isolated.connection_dir().exists()


@pytest.mark.parametrize(
    ("agent", "project_config"),
    [("crush", ".crush.json"), ("aider", ".aider.conf.yml")],
)
def test_saved_runner_checks_new_working_directory_before_agent_starts(
    private_home, monkeypatch, agent, project_config
):
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    commands = private_home / "commands"
    commands.mkdir()
    rapid = commands / "rapid-mlx"
    rapid.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if '--check' not in sys.argv:\n"
        "    from rapid_mlx.agents import isolated\n"
        "    isolated.configure(sys.argv[2], 'qwen3.5-9b-4bit', port=7660)\n"
    )
    rapid.chmod(0o700)
    marker = private_home / "agent-started"
    client = commands / agent
    client.write_text(f'#!/bin/sh\ntouch "{marker}"\n')
    client.chmod(0o700)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    first = private_home / "first"
    second = private_home / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    with patch("rapid_mlx.agents.isolated._cli_invocation", return_value=str(rapid)):
        isolated.configure(agent, "qwen3.5-9b-4bit", port=7660)
    runner = isolated.connection_dir() / "agents" / agent / "run"
    unsafe = second / project_config
    unsafe.write_text("cloud override\n")
    result = subprocess.run(
        [shutil.which("zsh") or "/bin/bash", str(runner)],
        cwd=second,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not marker.exists()
    assert unsafe.read_text() == "cloud override\n"
    if agent == "crush":
        unsafe.unlink()
        project_id = hashlib.sha256(str(second.resolve()).encode()).hexdigest()[:24]
        saved = (
            isolated.connection_dir()
            / "agents"
            / "crush"
            / "data"
            / project_id
            / "crush.json"
        )
        saved.parent.mkdir(parents=True)
        saved.write_text('{"models":{"large":{"model":"cloud","provider":"cloud"}}}')
        result = subprocess.run(
            [shutil.which("zsh") or "/bin/bash", str(runner)],
            cwd=second,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert marker.exists()
        assert json.loads(saved.read_text())["models"]["large"] == {
            "provider": "rapid-mlx",
            "model": "qwen3.5-9b-4bit",
        }


@pytest.mark.parametrize("probe_failure", ["unknown", "timeout"])
def test_opencode_unknown_version_launches_v1_without_unsupported_flags(
    private_home, monkeypatch, capsys, probe_failure
):
    from rapid_mlx.agents import opencode_version

    binary = private_home / ".opencode" / "bin" / "opencode"
    binary.parent.mkdir(parents=True)
    # Model a 1.x executable that rejects the 2.x flag, even when its version
    # cannot be identified. Its installation directory is absent from PATH.
    binary.write_text('#!/bin/sh\n[ "$#" -eq 0 ] || exit 1\n')
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(isolated, "_cli_invocation", lambda: "/usr/bin/true")
    if probe_failure == "timeout":
        result = subprocess.TimeoutExpired([str(binary), "--version"], 5)
        probe = patch.object(opencode_version.subprocess, "run", side_effect=result)
    else:
        probe = patch.object(
            opencode_version.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "unknown"),
        )
    with probe as run:
        runner = isolated.configure("opencode", "model", port=7660)
    assert run.call_args.args[0] == [str(binary), "--version"]
    assert run.call_args.kwargs["timeout"] == 5
    assert "version unknown" in capsys.readouterr().err
    root = isolated.connection_dir() / "agents" / "opencode"
    command = (root / "command").read_text()
    assert command.rstrip().endswith(shlex.quote(str(binary)))
    assert "--standalone" not in command
    subprocess.run(
        [shutil.which("zsh") or "/bin/bash", shlex.split(runner)[0]],
        check=True,
        timeout=10,
    )


def test_opencode_refresh_uses_new_executable_and_argv_without_rewriting_runner(
    private_home, monkeypatch
):
    # The first configuration runs without the per-user install on PATH.
    first = private_home / ".opencode" / "bin" / "opencode"
    first.parent.mkdir(parents=True)
    first.write_text('#!/bin/sh\nprintf "2.0.26\\n"\n')
    first.chmod(0o700)
    second = private_home / "new bin" / "opencode"
    second.parent.mkdir()
    record = private_home / "executed"
    second.write_text(
        '#!/bin/sh\nif [ "$1" = "--version" ]; then echo 1.19.4; exit 0; fi\n'
        '[ "$#" -eq 0 ] || exit 1\n'
        f'printf "%s" "$0" > {shlex.quote(str(record))}\n'
    )
    second.chmod(0o700)
    # A real subprocess refreshes configuration, just as the CLI does after
    # checking the endpoint. The fake endpoint check requires no live server.
    refresh = private_home / "refresh.py"
    cli = f"{shlex.quote(sys.executable)} {shlex.quote(str(refresh))}"
    refresh.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from rapid_mlx.agents import isolated\n"
        "if '--check' not in sys.argv:\n"
        f"    isolated._cli_invocation = lambda: {cli!r}\n"
        "    isolated.configure('opencode', 'model', port=7660)\n"
    )
    monkeypatch.setenv("HOME", str(private_home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(isolated, "_cli_invocation", lambda: cli)
    runner = Path(shlex.split(isolated.configure("opencode", "model", port=7660))[0])
    command = runner.parent / "command"
    assert f"{first} --standalone" in command.read_text()
    before = runner.stat().st_mtime_ns
    monkeypatch.setenv("PATH", f"{second.parent}:/usr/bin:/bin")
    subprocess.run(
        [shutil.which("zsh") or "/bin/bash", str(runner)], check=True, timeout=10
    )
    assert record.read_text() == str(second)
    assert "--standalone" not in command.read_text()
    assert runner.stat().st_mtime_ns == before
    config = json.loads((runner.parent / "opencode.json").read_text())
    assert "provider" in config and "providers" not in config


@pytest.mark.parametrize("version", [None, "", "1.19.4", "3.0.0"])
def test_opencode_only_known_supported_major_gets_standalone(private_home, version):
    command = isolated._command(
        "opencode", private_home, "model", 7660, version=version
    )
    assert "--standalone" not in command
