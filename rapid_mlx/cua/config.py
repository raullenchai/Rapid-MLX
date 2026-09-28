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
        kind = "remote" if self.allow_remote or ":18888" in self.url else "local"
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
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_stored(stored: dict) -> None:
    """Persist raw config; 0600 because presets may carry API keys."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        CONFIG_PATH.chmod(0o600)
    except OSError:  # pragma: no cover - filesystems without posix perms
        pass


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
        return PlannerConfig(
            preset=spec,
            url=url_override or preset["url"],
            model=model_override or preset["model"],
            reasoning_effort=preset.get("reasoning_effort"),
            text_only=bool(preset.get("text_only", False)),
            api_key=api_key,
            # A user-created preset with credentials is an explicit decision
            # to send task data to that endpoint; that consent unlocks
            # non-loopback planner URLs (https only, enforced in Planner).
            allow_remote=bool(api_key),
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


def save_user_preset(
    name: str,
    url: str,
    model: str,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    text_only: bool = False,
) -> dict:
    """Create or update a user-defined brain preset.

    Product behavior (not POC): users add their own cloud brain from the app
    settings. The endpoint URL and key live in the local config file (chmod
    0600); a preset carrying an api_key is the user's explicit consent to
    send task data to that endpoint, which is what allows non-loopback
    planner URLs (https enforced by Planner).
    """
    name = re.sub(r"\s+", "-", (name or "").strip()).lower()
    if not PRESET_NAME_RE.match(name):
        raise ValueError(
            "preset name must be 1-32 chars: lowercase letters, digits, dashes"
        )
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("brain URL must start with http:// or https://")
    from urllib.parse import urlparse

    remote = urlparse(url).hostname not in ("127.0.0.1", "::1", "localhost")
    if remote and api_key and url.startswith("http://"):
        raise ValueError(
            "remote brain URL must be HTTPS (task data leaves the machine)"
        )
    if not (model or "").strip():
        raise ValueError("brain model is required")
    stored = _read_stored()
    presets = stored.setdefault("presets", {})
    presets[name] = {
        "url": url,
        "model": model.strip(),
        "reasoning_effort": reasoning_effort,
        "text_only": bool(text_only),
        **({"api_key": api_key} if api_key else {}),
        "user_created": True,
    }
    _write_stored(stored)
    return name, presets[name]


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
