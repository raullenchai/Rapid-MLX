# SPDX-License-Identifier: Apache-2.0
"""Private, removable agent configurations for the local Desktop endpoint."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.codex_catalog import build_codex_model_info
from rapid_mlx.agents.server_hint import is_connection_refused

SUPPORTED = frozenset(
    {
        "claude-code",
        "codex",
        "pi",
        "opencode",
        "hermes",
        "goose",
        "qwen-code",
        "crush",
        "aider",
    }
)
_API_KEY_ENV = "RAPID_MLX_API_KEY"
_GIT_PASSTHROUGH = frozenset(
    {
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_DATE",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_COUNT",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_SSH_VARIANT",
        "GIT_ASKPASS",
        "GIT_TERMINAL_PROMPT",
    }
)
# Shells OpenCode accepts for its shell tool on every supported major version.
_OPENCODE_SHELLS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})
_XDG_DIRS = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
_HERMES_MIN_CONTEXT = 64_000
_HERMES_MAX_CONTEXT = 131_072


def connection_dir() -> Path:
    return Path.home() / ".rapid-mlx" / "connect"


def key_path() -> Path:
    return connection_dir() / "api-key"


def port_path() -> Path:
    return connection_dir() / "port"


def saved_model(name: str) -> str | None:
    """Read the model selected when this private agent profile was saved."""
    path = connection_dir() / "agents" / name / "model"
    _reject_symlinked_ancestors(path)
    if not path.is_file():
        return None
    model = path.read_text(encoding="utf-8").strip()
    return model or None


def _reject_symlinked_ancestors(path: Path) -> None:
    current = path
    while current != Path.home():
        if current.is_symlink():
            raise ValueError(f"refusing symlinked config path: {current}")
        current = current.parent


def _private_dir(path: Path) -> None:
    _reject_symlinked_ancestors(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _private_write(path: Path, content: str) -> None:
    _private_dir(path.parent)
    if path.is_symlink():
        raise ValueError(f"refusing symlinked config file: {path}")
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
    except BaseException:
        os.close(fd)
        raise


def _private_python_launcher(path: Path, source: str) -> None:
    script = path.with_suffix(".py")
    _private_write(script, source)
    _private_write(
        path,
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -I "
        f'{shlex.quote(str(script))} "$@"\n',
    )
    path.chmod(0o700)


def stable_key(*, create: bool = True) -> str | None:
    path = key_path()
    _reject_symlinked_ancestors(path)
    if path.exists():
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("local API key must be a regular 0600 file")
        key = path.read_text(encoding="ascii").strip()
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("local API key is invalid; rotate it in Desktop settings")
        return key
    if not create:
        return None
    _private_dir(path.parent)
    key = secrets.token_hex(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return stable_key(create=False)
    with os.fdopen(fd, "w", encoding="ascii") as file:
        file.write(key + "\n")
    return key


def stable_port() -> int:
    path = port_path()
    _reject_symlinked_ancestors(path)
    if path.exists():
        port = int(path.read_text(encoding="ascii").strip())
        if not 1 <= port <= 65535:
            raise ValueError("saved local port is outside 1..65535")
        return port
    return 7659


def save_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("local port is outside 1..65535")
    _private_write(port_path(), f"{port}\n")


def connection_status(port: int, key: str, model: str) -> str:
    """Return ready, stopped, or a specific local configuration error."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return "key-mismatch" if exc.code in {401, 403} else "server-error"
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return "stopped" if is_connection_refused(exc) else "server-error"
    except (ValueError, TypeError):
        return "server-error"
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return "server-error"
    advertised = {item.get("id") for item in models if isinstance(item, dict)}
    return "ready" if model in advertised else "model-mismatch"


def served_context_length(
    port: int, key: str, model: str, *, require_served: bool = False
) -> int | None:
    """Read the active KV limit, bounded by the model's native window."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        models = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(models, list):
            entry = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict) and item.get("id") == model
                ),
                None,
            )
            if entry is not None:
                from rapid_mlx.agents.adapter import _valid_context_window

                native = _valid_context_window(entry.get("context_window"))
                served = _valid_context_window(entry.get("max_model_len"))
                if served is not None or (native is not None and not require_served):
                    return min(value for value in (served, native) if value is not None)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, TypeError):
        pass
    return None


def hermes_context_length(
    port: int, key: str, model: str, *, require_live: bool = False
) -> int:
    """Use the served KV budget for Hermes, bounded by the native window.

    A stopped server can be configured provisionally from cached model metadata.
    The launch path calls this again after readiness, before starting Hermes.
    """
    context = served_context_length(port, key, model, require_served=require_live)
    if context is not None:
        context = min(context, _HERMES_MAX_CONTEXT)
    if context is None and not require_live:
        from rapid_mlx.run.cli import _cached_context_window

        context = _cached_context_window(model)
        if context is not None:
            context = min(context, _HERMES_MAX_CONTEXT)
    if context is None:
        if require_live:
            raise ValueError(
                f"Hermes needs a context limit from the ready local server for {model}; "
                "check /v1/models and retry."
            )
        # A server started by launch is probed again before the agent opens.
        return _HERMES_MIN_CONTEXT
    if context < _HERMES_MIN_CONTEXT:
        raise ValueError(
            f"Hermes needs at least {_HERMES_MIN_CONTEXT:,} context tokens, "
            f"but {model} can serve only {context:,}. Choose a 64K-capable model "
            "such as qwen3.5-9b-4bit or raise the server context limit."
        )
    return context


def running_model(port: int, key: str) -> str | None:
    """Find a single text checkpoint on the authenticated local server."""
    from http.client import HTTPException

    from rapid_mlx.model_aliases import resolve_model

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return None
        raise ValueError("could not verify the running local server") from exc
    except (OSError, TimeoutError, ValueError, TypeError, HTTPException) as exc:
        raise ValueError("could not verify the running local server") from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("local server returned an invalid model list")
    names: list[str] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("id")
        if (
            isinstance(name, str)
            and name
            and not entry.get("anthropic_family_tier")
            and (
                not isinstance(entry.get("capabilities"), list)
                or "text" in entry["capabilities"]
            )
        ):
            names.append(name)
    if not names:
        raise ValueError("local server did not advertise a text model")
    try:
        checkpoints = {resolve_model(name) for name in names}
    except ValueError as exc:
        raise ValueError("could not resolve the running model") from exc
    if len(checkpoints) != 1:
        raise ValueError("local server has multiple text checkpoints")
    return min(names, key=lambda name: ("/" in name, len(name)))


def _user_shell() -> str:
    """Return the POSIX shell OpenCode would otherwise use for commands."""
    shell = os.environ.get("SHELL", "")
    if (
        os.path.isabs(shell)
        and Path(shell).name in _OPENCODE_SHELLS
        and os.path.isfile(shell)
        and os.access(shell, os.X_OK)
    ):
        return shell
    # OpenCode rejects shells such as fish and falls back to zsh on macOS.
    return shutil.which("zsh") or "/bin/sh"


def _opencode_shell(root: Path) -> Path:
    # OpenCode derives shell arguments from the basename, so keep it.
    return root / "shell" / Path(_user_shell()).name


def _write_opencode_shell(root: Path) -> None:
    """Give commands started by OpenCode the user's own HOME and XDG dirs.

    OpenCode reads its global config, credentials, and session data only from
    HOME and the XDG base directories, so the OpenCode process itself needs
    private values. Its shell and terminal tools start every command through
    $SHELL, so this wrapper restores the user's environment before the real
    shell starts. Git identity, CLI logins, and toolchains then behave as in
    the user's own terminal.
    """
    quote = shlex.quote
    real = _user_shell()
    lines = ["#!/bin/sh"]
    for variable, value in (
        ("HOME", str(Path.home())),
        ("SHELL", real),
        *((variable, os.environ.get(variable)) for variable in _XDG_DIRS),
    ):
        lines.append(
            f"unset {variable}"
            if value is None
            else f"{variable}={quote(value)}; export {variable}"
        )
    lines.append(f'exec {quote(real)} "$@"')
    wrapper = _opencode_shell(root)
    _private_write(wrapper, "\n".join(lines) + "\n")
    wrapper.chmod(0o700)


def _cli_invocation() -> str:
    current = Path(sys.argv[0])
    if current.name == "rapid-mlx" and current.is_file():
        return shlex.quote(str(current.resolve()))
    binary = shutil.which("rapid-mlx")
    return shlex.quote(binary) if binary else "rapid-mlx"


def _display_cli_invocation() -> str:
    """Prefer the command users can type when PATH finds this installation."""
    invocation = _cli_invocation()
    found = shutil.which("rapid-mlx")
    if found and Path(found).resolve() == Path(shlex.split(invocation)[0]).resolve():
        return "rapid-mlx"
    return invocation


def _command(
    name: str,
    root: Path,
    model: str,
    port: int,
    *,
    version: str | None = None,
    opencode_run: bool = False,
    executable: str | None = None,
) -> str:
    quote = shlex.quote
    key = quote(str(key_path()))
    # The secret is read at invocation time, so rotating it does not stale a
    # saved profile. It is never placed in the printed command or process argv.
    credential = f'{_API_KEY_ENV}="$(cat {key})"'
    if name == "codex":
        return (
            f"{credential} CODEX_HOME={quote(str(root))} "
            f"codex --profile rapid --model {quote(model)}"
        )
    if name == "pi":
        return f"{credential} PI_CODING_AGENT_DIR={quote(str(root))} pi --provider rapid-mlx --model {quote(model)}"
    if name == "opencode":
        # OpenCode 2.x removed the top-level --model flag. Both supported
        # config shapes already select the model with the top-level `model` key.
        # Its config loader also reads and may rewrite the default XDG file.
        # Give the process a private home. Inline config has precedence over
        # project config, which would otherwise replace our selected model.
        # SHELL restores the user's HOME and XDG dirs for agent commands.
        config = quote(str(root / "opencode.json"))
        # OpenCode 2.x otherwise joins its shared background service on a
        # fixed port. That service may belong to the user's normal OpenCode
        # session and does not inherit this private profile or Rapid key.
        standalone = " --standalone" if (version or "").startswith("2.") else ""
        invocation = quote(executable or "opencode") + (" run" if opencode_run else "")
        return (
            f'{credential} OPENCODE_CONFIG_CONTENT="$(cat {config})" '
            "env -u OPENCODE_CONFIG_DIR "
            f"HOME={quote(str(root / 'home'))} "
            f"XDG_CONFIG_HOME={quote(str(root / 'xdg-config'))} "
            f"XDG_DATA_HOME={quote(str(root / 'xdg-data'))} "
            f"XDG_CACHE_HOME={quote(str(root / 'xdg-cache'))} "
            f"XDG_STATE_HOME={quote(str(root / 'xdg-state'))} "
            f"SHELL={quote(str(_opencode_shell(root)))} "
            f"OPENCODE_CONFIG={config} {invocation}{standalone}"
        )
    if name == "hermes":
        return f"{credential} HERMES_HOME={quote(str(root))} hermes -p rapid chat --provider custom --model {quote(model)}"
    if name == "goose":
        return f"{credential} XDG_CONFIG_HOME={quote(str(root))} GOOSE_MODEL={quote(model)} goose session --provider custom_rapid_mlx"
    if name == "crush":
        return (
            f"{credential} CRUSH_GLOBAL_CONFIG={quote(str(root))} "
            f"CRUSH_GLOBAL_DATA={quote(str(root / 'global-data'))} "
            f"{quote(str(root / 'bin' / 'crush-launch'))}"
        )
    if name == "aider":
        return quote(str(root / "bin" / "aider-launch"))
    if name == "qwen-code":
        saved_port = quote(str(port_path()))
        base = f'"http://127.0.0.1:$(cat {saved_port})/v1"'
        return (
            f"QWEN_HOME={quote(str(root))} QWEN_RUNTIME_DIR={quote(str(root / 'runtime'))} "
            f'OPENAI_API_KEY="$(cat {key})" OPENAI_BASE_URL={base} '
            f"OPENAI_MODEL={quote(model)} qwen --bare --auth-type openai "
            f"--model {quote(model)} --openai-base-url {base}"
        )
    if name == "claude-code":
        # The CLI can rewrite ~/.claude.json even with --settings. Its
        # documented config-dir override keeps that state under our profile.
        # Limit the private local profile to core coding tools: Claude sends
        # each enabled tool schema in the first request to the local model.
        tools = "Bash,Edit,Read,Write,Glob,Grep"
        return (
            f"CLAUDE_CONFIG_DIR={quote(str(root / 'config'))} "
            f"claude --settings {quote(str(root / 'settings.json'))} "
            f"--tools {tools}"
        )
    raise ValueError(f"unsupported isolated agent: {name}")


def configure(
    name: str,
    model: str,
    *,
    port: int | None = None,
    require_live_context: bool = False,
) -> str:
    """Render an existing registry profile under Rapid's private directory."""
    profile = get_profile(name)
    if profile is None or profile.name not in SUPPORTED:
        raise ValueError(f"unsupported isolated agent: {name}")
    name = profile.name
    if name in ("aider", "crush"):
        git_discovery_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
    if name == "aider":
        cwd = Path.cwd()
        git = shutil.which("git")
        if git is None:
            raise ValueError("isolated aider launch requires git")
        git_root = (
            subprocess.run(
                [git, "rev-parse", "--show-toplevel"],
                cwd=cwd,
                env=git_discovery_env,
                capture_output=True,
                text=True,
                check=False,
            )
            if git
            else None
        )
        dirs = {cwd}
        if git_root and git_root.returncode == 0:
            dirs.add(Path(git_root.stdout.strip()).resolve())
        if any(
            (directory / filename).exists() or (directory / filename).is_symlink()
            for directory in dirs
            for filename in (".aider.conf.yml", ".aider.model.settings.yml", ".env")
        ):
            raise ValueError(
                "isolated aider launch cannot override project configuration"
            )
    if name == "crush":
        # Project config outranks CRUSH_GLOBAL_CONFIG and may select a cloud
        # endpoint. Crush stops discovery at the Git root (or cwd outside Git).
        # Refuse before creating any private launch state.
        cwd = Path.cwd()
        git = shutil.which("git")
        git_root = (
            subprocess.run(
                [git, "rev-parse", "--show-toplevel"],
                cwd=cwd,
                env=git_discovery_env,
                capture_output=True,
                text=True,
                check=False,
            )
            if git
            else None
        )
        boundary = (
            Path(git_root.stdout.strip()).resolve()
            if git_root and git_root.returncode == 0
            else cwd
        )
        for directory in (cwd, *cwd.parents):
            if any(
                (directory / filename).exists() or (directory / filename).is_symlink()
                for filename in (".crushrc", "crushrc", ".crush.json", "crush.json")
            ):
                raise ValueError(
                    "isolated crush launch cannot override project configuration"
                )
            if directory == boundary:
                break
    port = port or stable_port()
    save_port(port)
    key = stable_key()
    assert key is not None
    root = connection_dir() / "agents" / name
    _private_dir(root)
    if name == "opencode":
        _private_dir(root / "home")
        _write_opencode_shell(root)
    if name == "qwen-code":
        _private_dir(root / "runtime")
    if name == "aider":
        _private_dir(root / "home")
        assert git is not None
        git_shim = root / "bin" / "git"
        _private_write(
            git_shim,
            f"#!/bin/sh\nHOME={shlex.quote(str(Path.home()))} "
            f'exec {shlex.quote(git)} "$@"\n',
        )
        git_shim.chmod(0o700)
    if name == "crush":
        # UI model selections in either higher-precedence layer must not
        # override the model just selected in Rapid. Keep session databases.
        project_id = hashlib.sha256(str(Path.cwd().resolve()).encode()).hexdigest()[:24]
        _private_dir(root / "global-data")
        _private_dir(root / "data" / project_id)
        for saved_config in (
            root / "global-data" / "crush.json",
            root / "data" / project_id / "crush.json",
        ):
            _reject_symlinked_ancestors(saved_config)
            if saved_config.is_file():
                try:
                    saved = json.loads(saved_config.read_text())
                except (json.JSONDecodeError, UnicodeError) as exc:
                    raise ValueError(
                        f"invalid isolated crush saved configuration: {saved_config}"
                    ) from exc
                if not isinstance(saved, dict):
                    raise ValueError("invalid isolated crush saved configuration")
                saved.pop("providers", None)
                models = saved.get("models")
                models = models if isinstance(models, dict) else {}
                tuning = {
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                }
                pinned = {}
                for slot in ("large", "small"):
                    pinned[slot] = {"provider": "rapid-mlx", "model": model}
                    selected = models.get(slot)
                    if isinstance(selected, dict):
                        pinned[slot].update(
                            {
                                field: value
                                for field, value in selected.items()
                                if field in tuning
                            }
                        )
                saved["models"] = pinned
                options = saved.get("options")
                if isinstance(options, dict):
                    options.pop("data_directory", None)
                    options.pop("disable_default_providers", None)
                _private_write(saved_config, json.dumps(saved, indent=2) + "\n")
    _private_write(root / "model", model + "\n")
    base = f"http://127.0.0.1:{port}/v1"
    previous = os.environ.get("RAPID_MLX_API_KEY")
    os.environ["RAPID_MLX_API_KEY"] = key
    try:
        version = None
        executable = None
        context_length = None
        if name == "opencode":
            from rapid_mlx.agents.opencode_version import (
                installed_version,
                resolve_executable,
            )

            executable = resolve_executable()
            version = installed_version(binary=executable)
            if version is None:
                print(
                    "OpenCode version unknown; omitting version-specific flags.",
                    file=sys.stderr,
                )
        if name == "hermes":
            context_length = hermes_context_length(
                port, key, model, require_live=require_live_context
            )
        if name == "codex":
            context_length = served_context_length(
                port, key, model, require_served=require_live_context
            )
            if context_length is None:
                if require_live_context:
                    raise ValueError(
                        f"Codex needs a context limit from the ready local server for {model}; "
                        "check /v1/models and retry."
                    )
                context_length = 32_768
        rendered = profile.render_config(
            base, model, version, context_length=context_length
        )
    finally:
        if previous is None:
            os.environ.pop("RAPID_MLX_API_KEY", None)
        else:
            os.environ["RAPID_MLX_API_KEY"] = previous
    if name == "codex":
        assert isinstance(rendered, str)
        catalog = root / "rapid-mlx-model-catalog.json"
        rendered = rendered.replace("{model_catalog_path}", str(catalog))
        # Keep the local model's first turn free of account integrations and
        # memory context. These are Codex config keys, scoped to this private
        # profile rather than the user's normal CODEX_HOME.
        rendered += (
            "\n[features]\n"
            "apps = false\n"
            "plugins = false\n"
            "remote_plugin = false\n"
            "memories = false\n"
            "\n[memories]\n"
            "use_memories = false\n"
            "generate_memories = false\n"
            "\n[skills]\n"
            "max_context_tokens = 256\n"
        )
        legacy = root / "config.toml"
        if legacy.is_symlink():
            raise ValueError(f"refusing symlinked config file: {legacy}")
        if legacy.exists():
            legacy.unlink()
        _private_write(root / "rapid.config.toml", rendered)
        _private_write(
            catalog,
            json.dumps({"models": [build_codex_model_info(model, context_length)]})
            + "\n",
        )
    elif name == "pi":
        _private_write(root / "models.json", str(rendered))
    elif name == "opencode":
        _private_write(root / "opencode.json", str(rendered))
    elif name == "hermes":
        import yaml

        # Keep this private profile lean without changing the user's Hermes home.
        profile_root = root / "profiles" / "rapid"
        marker = profile_root / ".no-bundled-skills"
        cleaned = profile_root / ".rapid-bundled-skills-cleaned"
        _reject_symlinked_ancestors(marker)
        _reject_symlinked_ancestors(cleaned)
        if not cleaned.is_file():
            skills_dir = profile_root / "skills"
            _reject_symlinked_ancestors(skills_dir)
            if skills_dir.is_dir():
                if not shutil.which("hermes"):
                    raise ValueError(
                        "Hermes private skill cleanup needs the Hermes CLI; install it and retry launch."
                    )
                # Hermes removes only its own unmodified bundled skills.
                try:
                    result = subprocess.run(
                        [
                            "hermes",
                            "-p",
                            "rapid",
                            "skills",
                            "opt-out",
                            "--remove",
                            "--yes",
                        ],
                        env={**os.environ, "HERMES_HOME": str(root)},
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=120,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    result = None
                if result is not None and result.returncode == 0:
                    _private_write(
                        cleaned,
                        "Bundled skills opt-out completed for private profile.\n",
                    )
                else:
                    raise ValueError(
                        "Hermes private skill cleanup did not complete; retry launch."
                    )
            elif not skills_dir.is_dir():
                _private_write(
                    cleaned, "Private profile started without bundled skills.\n"
                )
        if not marker.is_file():
            _private_write(marker, "Private local profile: no bundled skills.\n")
        hermes_config = yaml.safe_load(str(rendered))
        hermes_config["platform_toolsets"] = {"cli": ["terminal", "file"]}
        rendered = yaml.safe_dump(hermes_config, sort_keys=False).replace(
            "  base_url: ", f'  api_key: "{key}"\n  base_url: ', 1
        )
        _private_write(root / "profiles" / "rapid" / "config.yaml", str(rendered))
    elif name == "goose":
        _private_write(
            root / "goose" / "custom_providers" / "custom_rapid_mlx.json", str(rendered)
        )
    elif name == "crush":
        _private_write(root / "crush.json", str(rendered))
    elif name == "claude-code":
        _private_dir(root / "config")
        settings = {
            "env": {
                "ANTHROPIC_BASE_URL": base.removesuffix("/v1"),
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": key,
                "ANTHROPIC_MODEL": model,
            }
        }
        _private_write(root / "settings.json", json.dumps(settings, indent=2) + "\n")
    elif name == "aider":
        _private_write(
            root / "aider.env",
            f"AIDER_OPENAI_API_KEY={key}\nOPENAI_API_KEY={key}\n"
            f"AIDER_OPENAI_API_BASE={base}\nOPENAI_API_BASE={base}\n",
        )
        # Unknown OpenAI-compatible models otherwise fall back to whole-file
        # edits, which can mistake fenced command output for another file.
        model_settings = root / "aider.model.settings.yml"
        _private_write(
            model_settings,
            json.dumps(
                [
                    {
                        "name": f"openai/{model}",
                        "edit_format": "diff",
                        "editor_edit_format": "editor-diff",
                        "use_repo_map": True,
                        "use_temperature": False,
                    }
                ],
                indent=2,
            )
            + "\n",
        )
        _private_write(
            root / "aider.conf.yml",
            json.dumps(
                {
                    "model-settings-file": str(model_settings),
                    "edit-format": "diff",
                    "editor-edit-format": "editor-diff",
                    "suggest-shell-commands": True,
                    "auto-test": False,
                },
                indent=2,
            )
            + "\n",
        )
        helper = root / "bin" / "aider-launch"
        _private_python_launcher(
            helper,
            "import os\nfrom pathlib import Path\n"
            f"root = Path({str(root)!r})\n"
            f"key_path = Path({str(key_path())!r})\n"
            f"port_path = Path({str(port_path())!r})\n"
            f"model = {model!r}\n"
            f"allowed_git = {tuple(sorted(_GIT_PASSTHROUGH))!r}\n"
            "source = os.environ\n"
            "environment = {'PATH': str(root / 'bin') + os.pathsep + source.get('PATH', ''), "
            "'HOME': str(root / 'home'), 'TERM': source.get('TERM', 'xterm-256color')}\n"
            "for name in ('SSH_AUTH_SOCK', 'SSH_ASKPASS', 'GPG_TTY', 'GNUPGHOME', "
            "'EMAIL', 'LANG', 'LC_ALL'):\n"
            "    if name in source:\n        environment[name] = source[name]\n"
            "for name, value in source.items():\n"
            "    indexed = name.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')) "
            "and name.rsplit('_', 1)[-1].isdigit()\n"
            "    valid = name.isascii() and all(c.isupper() or c.isdigit() or c == '_' "
            "for c in name)\n"
            "    if valid and (name in allowed_git or indexed):\n"
            "        environment[name] = value\n"
            "local_key = key_path.read_text().strip()\n"
            "base = 'http://127.0.0.1:' + port_path.read_text().strip() + '/v1'\n"
            "environment.update({'AIDER_OPENAI_API_KEY': local_key, "
            "'OPENAI_API_KEY': local_key, 'AIDER_OPENAI_API_BASE': base, "
            "'OPENAI_API_BASE': base})\n"
            "selected_model = 'openai/' + model\n"
            "args = ['aider', '--config', str(root / 'aider.conf.yml'), "
            "'--env-file', str(root / 'aider.env'), '--openai-api-base', base, "
            "'--model', selected_model, '--weak-model', selected_model, "
            "'--editor-model', selected_model]\n"
            "os.execvpe('aider', args, environment)\n",
        )
    elif name == "qwen-code":
        _private_write(
            root / "qwen.env", f"OPENAI_BASE_URL={base}\nOPENAI_MODEL={model}\n"
        )
    if name == "crush":
        helper = root / "bin" / "crush-launch"
        _private_python_launcher(
            helper,
            "import hashlib\nimport os\nfrom pathlib import Path\n"
            f"data_root = Path({str(root / 'data')!r})\n"
            "project_id = hashlib.sha256(str(Path.cwd().resolve()).encode()).hexdigest()[:24]\n"
            f"allowed_git = {tuple(sorted(_GIT_PASSTHROUGH))!r}\n"
            "source = os.environ\n"
            "environment = {key: value for key, value in source.items() "
            "if not key.startswith('GIT_')}\n"
            "for name, value in source.items():\n"
            "    indexed = name.startswith(('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')) "
            "and name.rsplit('_', 1)[-1].isdigit()\n"
            "    valid = name.isascii() and all(c.isupper() or c.isdigit() or c == '_' "
            "for c in name)\n"
            "    if valid and (name in allowed_git or indexed):\n"
            "        environment[name] = value\n"
            "os.execvpe('crush', ['crush', '--data-dir', str(data_root / project_id)], environment)\n",
        )
    command = _command(name, root, model, port, version=version, executable=executable)
    if name == "opencode":
        # Refresh can change the executable and flags. Keep them out of the
        # running shell script, which may already have buffered its contents.
        launch = root / "command"
        _private_write(launch, command + "\n")
        command = f"exec /bin/sh {shlex.quote(str(launch))}"
    cli = _cli_invocation()
    runner = root / "run"
    # Reconfigure in the runner's execution directory before starting the
    # agent. This repeats project-config guards and sanitizes that project's
    # saved state, including when a saved runner is reused in another project.
    # A different Desktop session may also have chosen a new port or key.
    refresh = (
        ""
        if name == "hermes"
        else (
            f"{cli} launch {shlex.quote(name)} --isolated "
            f"--model {shlex.quote(model)} >/dev/null || exit 1\n"
        )
    )
    _private_write(
        runner,
        "#!/bin/zsh\n"
        + f"{cli} launch {shlex.quote(name)} --isolated --check --model {shlex.quote(model)} || exit 1\n"
        + refresh
        + command
        + "\n",
    )
    runner.chmod(0o700)
    return shlex.quote(str(runner))


def remove(name: str) -> bool:
    profile = get_profile(name)
    if profile is None or profile.name not in SUPPORTED:
        raise ValueError(f"unsupported isolated agent: {name}")
    root = connection_dir() / "agents" / profile.name
    _reject_symlinked_ancestors(root)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True
