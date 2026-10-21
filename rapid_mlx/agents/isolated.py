# SPDX-License-Identifier: Apache-2.0
"""Private, removable agent configurations for the local Desktop endpoint."""

from __future__ import annotations

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


def hermes_context_length(
    port: int, key: str, model: str, *, require_live: bool = False
) -> int:
    """Use the served KV budget for Hermes, bounded by the native window.

    A stopped server can be configured provisionally from cached model metadata.
    The launch path calls this again after readiness, before starting Hermes.
    """
    context = None
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
                if served is not None or (native is not None and not require_live):
                    context = min(
                        *(
                            value
                            for value in (served, native, _HERMES_MAX_CONTEXT)
                            if value is not None
                        )
                    )
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, TypeError):
        pass
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


def _cli_invocation() -> str:
    current = Path(sys.argv[0])
    if current.name == "rapid-mlx" and current.is_file():
        return shlex.quote(str(current.resolve()))
    binary = shutil.which("rapid-mlx")
    return shlex.quote(binary) if binary else "rapid-mlx"


def _command(name: str, root: Path, model: str, port: int) -> str:
    quote = shlex.quote
    key = quote(str(key_path()))
    # The secret is read at invocation time, so rotating it does not stale a
    # saved profile. It is never placed in the printed command or process argv.
    credential = f'{_API_KEY_ENV}="$(cat {key})"'
    if name == "codex":
        return f"{credential} CODEX_HOME={quote(str(root))} codex --profile rapid"
    if name == "pi":
        return f"{credential} PI_CODING_AGENT_DIR={quote(str(root))} pi --provider rapid-mlx --model {quote(model)}"
    if name == "opencode":
        return f"{credential} OPENCODE_CONFIG={quote(str(root / 'opencode.json'))} opencode --model rapid-mlx/{quote(model)}"
    if name == "hermes":
        return f"{credential} HERMES_HOME={quote(str(root))} hermes -p rapid chat --provider custom --model {quote(model)}"
    if name == "goose":
        return f"{credential} XDG_CONFIG_HOME={quote(str(root))} GOOSE_MODEL={quote(model)} goose session --provider custom_rapid_mlx"
    if name == "crush":
        return (
            f"{credential} CRUSH_GLOBAL_CONFIG={quote(str(root / 'crush.json'))} crush"
        )
    if name == "aider":
        return f'OPENAI_API_KEY="$(cat {key})" aider --env-file {quote(str(root / "aider.env"))} --model openai/{quote(model)}'
    if name == "qwen-code":
        saved_port = quote(str(port_path()))
        return f'OPENAI_API_KEY="$(cat {key})" OPENAI_BASE_URL="http://127.0.0.1:$(cat {saved_port})/v1" OPENAI_MODEL={quote(model)} qwen'
    if name == "claude-code":
        return f"claude --settings {quote(str(root / 'settings.json'))}"
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
    port = port or stable_port()
    save_port(port)
    key = stable_key()
    assert key is not None
    root = connection_dir() / "agents" / name
    _private_dir(root)
    _private_write(root / "model", model + "\n")
    base = f"http://127.0.0.1:{port}/v1"
    previous = os.environ.get("RAPID_MLX_API_KEY")
    os.environ["RAPID_MLX_API_KEY"] = key
    try:
        version = None
        context_length = None
        if name == "opencode":
            from rapid_mlx.agents.opencode_version import installed_version

            version = installed_version()
        if name == "hermes":
            context_length = hermes_context_length(
                port, key, model, require_live=require_live_context
            )
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
        legacy = root / "config.toml"
        if legacy.is_symlink():
            raise ValueError(f"refusing symlinked config file: {legacy}")
        if legacy.exists():
            legacy.unlink()
        _private_write(root / "rapid.config.toml", rendered)
        _private_write(
            catalog,
            json.dumps({"models": [build_codex_model_info(model, None)]}) + "\n",
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
        settings = {
            "env": {
                "ANTHROPIC_BASE_URL": base.removesuffix("/v1"),
                "ANTHROPIC_API_KEY": key,
                "ANTHROPIC_AUTH_TOKEN": "",
                "ANTHROPIC_MODEL": model,
            }
        }
        _private_write(root / "settings.json", json.dumps(settings, indent=2) + "\n")
    elif name == "aider":
        _private_write(root / "aider.env", f"OPENAI_API_BASE={base}\n")
    elif name == "qwen-code":
        _private_write(
            root / "qwen.env", f"OPENAI_BASE_URL={base}\nOPENAI_MODEL={model}\n"
        )
    command = _command(name, root, model, port)
    cli = _cli_invocation()
    runner = root / "run"
    # A different Desktop session may have chosen a new free port or rotated
    # the key. Refresh the private config only after the shared route validates.
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
