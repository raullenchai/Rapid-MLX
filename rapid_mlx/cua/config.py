"""CUA configuration: planner presets (slow thinking) are user-selectable.

Fast thinking is always local (laya via System One /v1/rank). Slow thinking is
an OpenAI-compatible chat endpoint the user picks: a cloud model, a local
Rapid-MLX server, or any custom URL. Presets live in
~/.rapid-mlx/cua-config.json and can be overridden per run with flags.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path.home() / ".rapid-mlx" / "cua-config.json"
RUNS_DIR = Path.home() / ".rapid-mlx" / "cua-runs"

DEFAULT_PRESETS: dict[str, dict] = {
    # Built-in defaults are on-device brains only. Cloud brains are
    # user-added from the app settings (name + endpoint + API key); shipping
    # a default cloud preset would point at a URL no user can reach.
    "local-27b": {
        "url": "http://127.0.0.1:18701/v1/chat/completions",
        "model": "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
        "reasoning_effort": None,
        "text_only": True,
        "note": "on-device Qwen3.8-27B (text-only, guided JSON)",
    },
    "local-9b": {
        "url": "http://127.0.0.1:18702/v1/chat/completions",
        "model": "mlx-community/Qwen3.5-9B-8bit",
        "reasoning_effort": None,
        "text_only": True,
        "note": "on-device Qwen3.5-9B (fastest; short flows)",
    },
}

DEFAULT_FAST_RANKER_URL = "http://127.0.0.1:18700/v1/rank"


def is_loopback_url(url: str) -> bool:
    """Return whether an HTTP(S) endpoint is unambiguously on this Mac."""
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class PlannerConfig:
    preset: str
    url: str
    model: str
    reasoning_effort: str | None = None
    text_only: bool = False
    timeout: float = 180.0
    api_key: str | None = None
    allow_remote: bool = False

    def describe(self) -> str:
        kind = "local" if is_loopback_url(self.url) else "remote"
        return f"{self.preset} [{kind}] {self.model}"


@dataclass
class CUAConfig:
    planner: PlannerConfig
    fast_ranker_url: str = DEFAULT_FAST_RANKER_URL
    max_steps: int = 12
    allowed_domain: str = ""
    confirm_actions: bool = False
    human_login: bool = False
    pause_timeout: float = 600.0
    extra: dict = field(default_factory=dict)


def _read_stored() -> dict:
    """Raw stored config (presets + fast_ranker_url), defaults when absent."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            loaded: dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return loaded
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_stored(stored: dict) -> None:
    """Persist raw config atomically; 0600 because presets may carry keys."""
    import fcntl
    import os
    import tempfile

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = CONFIG_PATH.with_suffix(".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fd, tmp_name = tempfile.mkstemp(
            dir=CONFIG_PATH.parent, prefix=".cua-config-", suffix=".tmp"
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(stored, ensure_ascii=False, indent=2))
            os.replace(tmp_name, CONFIG_PATH)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def load_config() -> dict:
    """Load config, creating defaults on first use."""
    stored = _read_stored()
    presets = {**DEFAULT_PRESETS, **stored.get("presets", {})}
    config = {
        "presets": presets,
        "fast_ranker_url": stored.get("fast_ranker_url", DEFAULT_FAST_RANKER_URL),
    }
    if not CONFIG_PATH.exists():
        _write_stored(config)
    return config


def save_config(config: dict) -> None:
    _write_stored(config)


def resolve_planner(
    spec: str,
    url_override: str | None = None,
    model_override: str | None = None,
    text_only_override: bool | None = None,
) -> PlannerConfig:
    """Resolve a planner from a preset name or a full URL."""
    config = load_config()
    presets = config["presets"]
    if spec in presets:
        preset = presets[spec]
        api_key = preset.get("api_key")
        # Legacy presets had no explicit flag; the old effective permission
        # was tied to a key. Keyless remote presets must be re-saved after the
        # user reviews the current data disclosure.
        allow_remote = bool(preset.get("allow_remote", bool(api_key)))
        if url_override and (api_key or allow_remote):
            # A URL override would send the preset's credential to a
            # different endpoint than the one the user consented to.
            raise ValueError(
                "planner URL override is not allowed for a model saved with "
                "credentials or remote-data consent; create a separate model instead"
            )
        return PlannerConfig(
            preset=spec,
            url=url_override or preset["url"],
            model=model_override or preset["model"],
            reasoning_effort=preset.get("reasoning_effort"),
            text_only=bool(preset.get("text_only", False)),
            api_key=api_key,
            allow_remote=allow_remote,
        )
    if spec.startswith(("http://", "https://")):
        if not model_override:
            raise ValueError(
                f"--planner-model is required when --planner is a URL: {spec}"
            )
        return PlannerConfig(
            preset="custom",
            url=spec,
            model=model_override,
            text_only=bool(text_only_override),
        )
    known = ", ".join(sorted(presets))
    raise ValueError(f"unknown planner preset {spec!r} (known: {known})")


PRESET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def _chat_completions_url(url: str) -> str:
    """Accept an OpenAI-compatible base URL or an explicit chat endpoint."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if parts.path.rstrip("/").endswith("/chat/completions"):
        return url
    base_path = parts.path.rstrip("/") or "/v1"
    path = base_path + "/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def save_user_preset(
    name: str,
    url: str,
    model: str,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    text_only: bool = False,
    allow_remote: bool = False,
) -> tuple[str, dict]:
    """Create or update a user-defined brain preset.

    Product behavior (not POC): users add an OpenAI-compatible brain from app
    settings. Consent to send task data to a non-loopback endpoint is stored
    independently from an optional API key. Remote endpoints require HTTPS.
    """
    name = re.sub(r"\s+", "-", (name or "").strip()).lower()
    if not PRESET_NAME_RE.match(name):
        raise ValueError(
            "preset name must be 1-32 chars: lowercase letters, digits, dashes"
        )
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("model URL must start with http:// or https://")
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError("model URL must include a hostname")
    remote = not is_loopback_url(url)
    if remote and not allow_remote:
        raise ValueError(
            "remote model requires explicit consent to send the task goal, "
            "Accessibility snapshot, and optional screenshot"
        )
    if remote and url.startswith("http://"):
        raise ValueError(
            "remote model URL must be HTTPS (task data leaves the machine)"
        )
    if not (model or "").strip():
        raise ValueError("model name is required")
    url = _chat_completions_url(url)
    if name in DEFAULT_PRESETS:
        raise ValueError(f"{name!r} is a built-in model; choose a different name")
    stored = _read_stored()
    presets = stored.setdefault("presets", {})
    if name in presets and not presets[name].get("user_created"):
        raise ValueError(f"preset name {name!r} is reserved")
    if name in presets:
        raise ValueError(
            f"model {name!r} already exists; delete it first to replace it"
        )
    presets[name] = {
        "url": url,
        "model": model.strip(),
        "reasoning_effort": reasoning_effort,
        "text_only": bool(text_only),
        "allow_remote": bool(remote and allow_remote),
        **({"api_key": api_key} if api_key else {}),
        "user_created": True,
    }
    _write_stored(stored)
    created: dict = presets[name]
    return name, created


def delete_user_preset(name: str) -> None:
    """Remove a user-defined preset. Built-in defaults cannot be deleted."""
    name = name.strip().lower()
    if name in DEFAULT_PRESETS:
        raise ValueError(f"built-in preset {name!r} cannot be deleted")
    stored = _read_stored()
    presets = stored.get("presets", {})
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}")
    del presets[name]
    _write_stored(stored)
