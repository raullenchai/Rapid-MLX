# SPDX-License-Identifier: Apache-2.0
"""Private, removable agent configurations for the local Desktop endpoint."""

from __future__ import annotations

import json
import os
import secrets
import shlex
import shutil
import stat
from pathlib import Path

from rapid_mlx.agents import get_profile
from rapid_mlx.agents.codex_catalog import build_codex_model_info

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


def connection_dir() -> Path:
    return Path.home() / ".rapid-mlx" / "connect"


def key_path() -> Path:
    return connection_dir() / "api-key"


def port_path() -> Path:
    return connection_dir() / "port"


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlinked config directory: {path}")
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
    if path.exists():
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
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


def _command(name: str, root: Path, model: str, port: int) -> str:
    quote = shlex.quote
    key = quote(str(key_path()))
    # The secret is read at invocation time, so rotating it does not stale a
    # saved profile. It is never placed in the printed command or process argv.
    credential = f'RAPID_MLX_API_KEY="$(cat {key})"'
    if name == "codex":
        return f"{credential} CODEX_HOME={quote(str(root))} codex --profile rapid"
    if name == "pi":
        return f"{credential} PI_CODING_AGENT_DIR={quote(str(root))} pi --provider rapid-mlx --model {quote(model)}"
    if name == "opencode":
        return f"{credential} OPENCODE_CONFIG={quote(str(root / 'opencode.json'))} opencode --model rapid-mlx/{quote(model)}"
    if name == "hermes":
        return f"{credential} HERMES_HOME={quote(str(root))} hermes -p rapid chat --provider custom --model {quote(model)}"
    if name == "goose":
        return f"{credential} XDG_CONFIG_HOME={quote(str(root))} GOOSE_MODEL={quote(model)} goose session start --provider custom_rapid_mlx"
    if name == "crush":
        return (
            f"{credential} CRUSH_GLOBAL_CONFIG={quote(str(root / 'crush.json'))} crush"
        )
    if name == "aider":
        return f'OPENAI_API_KEY="$(cat {key})" aider --env-file {quote(str(root / "aider.env"))} --model openai/{quote(model)}'
    if name == "qwen-code":
        return f'OPENAI_API_KEY="$(cat {key})" OPENAI_BASE_URL=http://127.0.0.1:{port}/v1 OPENAI_MODEL={quote(model)} qwen'
    if name == "claude-code":
        return f"claude --settings {quote(str(root / 'settings.json'))}"
    raise ValueError(f"unsupported isolated agent: {name}")


def configure(name: str, model: str, *, port: int | None = None) -> str:
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
    base = f"http://127.0.0.1:{port}/v1"
    previous = os.environ.get("RAPID_MLX_API_KEY")
    os.environ["RAPID_MLX_API_KEY"] = key
    try:
        rendered = profile.render_config(base, model)
    finally:
        if previous is None:
            os.environ.pop("RAPID_MLX_API_KEY", None)
        else:
            os.environ["RAPID_MLX_API_KEY"] = previous
    if name == "codex":
        assert isinstance(rendered, str)
        catalog = root / "rapid-mlx-model-catalog.json"
        rendered = rendered.replace("{model_catalog_path}", str(catalog))
        rendered += (
            '\n[profiles.rapid]\nmodel_provider = "rapid-mlx"\nmodel = '
            + json.dumps(model)
            + "\n"
        )
        _private_write(root / "config.toml", rendered)
        _private_write(
            catalog,
            json.dumps({"models": [build_codex_model_info(model, None)]}) + "\n",
        )
    elif name == "pi":
        _private_write(root / "models.json", str(rendered))
    elif name == "opencode":
        _private_write(root / "opencode.json", str(rendered))
    elif name == "hermes":
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
    start = f"rapid-mlx start {shlex.quote(name)} --model {shlex.quote(model)} --no-setup --port {port}"
    message = f"Local server is stopped. Start it with: {start}"
    runner = root / "run"
    _private_write(
        runner,
        "#!/bin/zsh\n"
        + f"if ! /usr/bin/curl -fsS -o /dev/null http://127.0.0.1:{port}/health/ready 2>/dev/null; then\n"
        + f"  /bin/echo {shlex.quote(message)} >&2\n"
        + "  exit 1\nfi\n"
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
    if not root.exists():
        return False
    if root.is_symlink():
        raise ValueError(f"refusing symlinked config directory: {root}")
    shutil.rmtree(root)
    return True
