"""Agent Profile registry — load, query, and manage agent integration profiles.

Usage:
    from vllm_mlx.agents import get_profile, list_profiles

    profile = get_profile("hermes")
    config = profile.render_config("http://localhost:8000/v1", "qwen3.5-9b")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from .base import (
    AgentConfigSpec,
    AgentProfile,
    AgentStreamingSpec,
    AgentTestingSpec,
    AgentVersionSpec,
)

logger = logging.getLogger(__name__)

_PROFILES: dict[str, AgentProfile] = {}
_LOADED = False

PROFILES_DIR = Path(__file__).parent / "profiles"


def _parse_config(data: dict) -> AgentConfigSpec:
    """Parse config section from YAML."""
    cfg = data.get("config", {})
    return AgentConfigSpec(
        type=cfg.get("type", "env"),
        path=cfg.get("path"),
        template=cfg.get("template"),
        env_vars=cfg.get("env_vars"),
    )


def _parse_streaming(data: dict) -> AgentStreamingSpec:
    """Parse streaming section from YAML."""
    s = data.get("streaming", {})
    tags = s.get("extra_tool_tags", [])
    # Convert list-of-lists to list-of-tuples
    extra_tags = [tuple(t) for t in tags] if tags else []
    return AgentStreamingSpec(
        extra_tool_tags=extra_tags,
        suppress_patterns=s.get("suppress_patterns", []),
        max_tools=s.get("max_tools"),
    )


def _parse_testing(data: dict) -> AgentTestingSpec:
    """Parse testing section from YAML."""
    t = data.get("testing", {})
    return AgentTestingSpec(
        binary=t.get("binary"),
        query_cmd=t.get("query_cmd"),
        query_timeout=t.get("query_timeout", 120),
        install_cmd=t.get("install_cmd"),
        specific_tests=t.get("specific_tests"),
    )


def _parse_versions(data: dict) -> list[AgentVersionSpec]:
    """Parse version overrides from YAML."""
    versions = []
    for v in data.get("versions", []):
        vs = AgentVersionSpec(
            version_range=v["range"],
            notes=v.get("notes"),
        )
        if "config" in v:
            vs.config = _parse_config(v)
        if "streaming" in v:
            vs.streaming = _parse_streaming(v)
        if "testing" in v:
            vs.testing = _parse_testing(v)
        versions.append(vs)
    return versions


def _load_profile_from_yaml(path: Path) -> AgentProfile:
    """Load a single profile from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models = data.get("models", {})
    capabilities = data.get("capabilities", {})

    return AgentProfile(
        name=data["name"],
        display_name=data.get("display_name", data["name"]),
        repo=data.get("repo"),
        stars=data.get("stars"),
        config=_parse_config(data),
        recommended_models=models.get("recommended", []),
        parser_override=models.get("parser_override"),
        streaming=_parse_streaming(data),
        testing=_parse_testing(data),
        versions=_parse_versions(data),
        known_issues=data.get("known_issues", []),
        needs_function_calling=capabilities.get("function_calling", True),
        needs_streaming=capabilities.get("streaming", True),
        needs_vision=capabilities.get("vision", False),
        needs_reasoning=capabilities.get("reasoning", False),
    )


def load_profiles(profiles_dir: Path | str | None = None):
    """Load all agent profiles from YAML files.

    Args:
        profiles_dir: Directory containing .yaml profile files.
                      Defaults to vllm_mlx/agents/profiles/
    """
    global _LOADED
    _PROFILES.clear()

    search_dir = Path(profiles_dir) if profiles_dir else PROFILES_DIR
    if not search_dir.exists():
        logger.warning(f"Agent profiles directory not found: {search_dir}")
        _LOADED = True
        return

    for yaml_file in sorted(search_dir.glob("*.yaml")):
        try:
            profile = _load_profile_from_yaml(yaml_file)
            _PROFILES[profile.name] = profile
            logger.debug(f"Loaded agent profile: {profile.name} ({yaml_file.name})")
        except Exception as e:
            logger.warning(f"Failed to load profile {yaml_file.name}: {e}")

    # Also load from user's custom profiles directory
    user_profiles = Path(os.path.expanduser("~/.rapid-mlx/agents"))
    if user_profiles.exists() and user_profiles != search_dir:
        for yaml_file in sorted(user_profiles.glob("*.yaml")):
            try:
                profile = _load_profile_from_yaml(yaml_file)
                _PROFILES[profile.name] = profile
                logger.debug(f"Loaded user agent profile: {profile.name}")
            except Exception as e:
                logger.warning(f"Failed to load user profile {yaml_file.name}: {e}")

    _LOADED = True
    logger.info(f"Loaded {len(_PROFILES)} agent profiles")


def _ensure_loaded():
    """Lazy-load profiles on first access."""
    if not _LOADED:
        load_profiles()


def get_profile(name: str) -> AgentProfile | None:
    """Get an agent profile by name. Returns None if not found."""
    _ensure_loaded()
    return _PROFILES.get(name)


def get_profile_or_generic(name: str) -> AgentProfile:
    """Get an agent profile by name, falling back to 'generic'."""
    _ensure_loaded()
    profile = _PROFILES.get(name)
    if profile:
        return profile
    generic = _PROFILES.get("generic")
    if generic:
        return generic
    # Absolute fallback — return a minimal profile
    return AgentProfile(
        name="generic",
        display_name="Generic Agent",
        config=AgentConfigSpec(
            type="env",
            env_vars={
                "OPENAI_BASE_URL": "{base_url}",
                "OPENAI_API_KEY": "not-needed",
            },
        ),
    )


def list_profiles() -> list[AgentProfile]:
    """List all loaded agent profiles, sorted by stars (descending)."""
    _ensure_loaded()
    return sorted(
        _PROFILES.values(),
        key=lambda p: p.stars or 0,
        reverse=True,
    )


__all__ = [
    "AgentProfile",
    "AgentConfigSpec",
    "AgentStreamingSpec",
    "AgentTestingSpec",
    "AgentVersionSpec",
    "get_profile",
    "get_profile_or_generic",
    "list_profiles",
    "load_profiles",
]
