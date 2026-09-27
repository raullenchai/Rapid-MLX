"""CUA configuration: planner presets (slow thinking) are user-selectable.

Fast thinking is always local (laya via System One /v1/rank). Slow thinking is
an OpenAI-compatible chat endpoint the user picks: a cloud model, a local
Rapid-MLX server, or any custom URL. Presets live in
~/.rapid-mlx/cua-config.json and can be overridden per run with flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path.home() / ".rapid-mlx" / "cua-config.json"
RUNS_DIR = Path.home() / ".rapid-mlx" / "cua-runs"

DEFAULT_PRESETS: dict[str, dict] = {
    "cloud-glm": {
        "url": "http://127.0.0.1:18888/v1/chat/completions",
        "model": "GLM-5.3-Flash-EXL3",
        "reasoning_effort": "low",
        "text_only": False,
        "note": "GLM-5.3-Flash via ssh tunnel (vision-capable)",
    },
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

    def describe(self) -> str:
        kind = (
            "local"
            if "127.0.0.1" in self.url and ":18888" not in self.url
            else "remote"
        )
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


def load_config() -> dict:
    """Load config, creating defaults on first use."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
    else:
        stored = {}
    presets = {**DEFAULT_PRESETS, **stored.get("presets", {})}
    config = {
        "presets": presets,
        "fast_ranker_url": stored.get("fast_ranker_url", DEFAULT_FAST_RANKER_URL),
    }
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return config


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
        return PlannerConfig(
            preset=spec,
            url=url_override or preset["url"],
            model=model_override or preset["model"],
            reasoning_effort=preset.get("reasoning_effort"),
            text_only=bool(preset.get("text_only", False)),
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
