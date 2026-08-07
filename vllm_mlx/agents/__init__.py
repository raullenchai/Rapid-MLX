"""Agent Profile registry — load, query, and manage agent integration profiles.

Usage:
    from vllm_mlx.agents import get_profile, list_profiles

    profile = get_profile("hermes")
    config = profile.render_config("http://localhost:8000/v1", "qwen3.5-9b-4bit")
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace as dc_replace
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
        home_env=cfg.get("home_env"),
    )


def _parse_streaming(data: dict) -> AgentStreamingSpec:
    """Parse streaming section from YAML."""
    s = data.get("streaming", {})
    tags = s.get("extra_tool_tags", [])
    # Convert list-of-lists to list-of-tuples
    extra_tags = [tuple(t) for t in tags] if tags else []
    return AgentStreamingSpec(
        extra_tool_tags=extra_tags,
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
        streaming=_parse_streaming(data),
        testing=_parse_testing(data),
        versions=_parse_versions(data),
        known_issues=data.get("known_issues", []),
        needs_function_calling=capabilities.get("function_calling", True),
        needs_streaming=capabilities.get("streaming", True),
    )


def _keep_home_env(
    profile: AgentProfile, shadowed: AgentProfile | None
) -> AgentProfile:
    """Carry ``home_env`` across a user profile that shadows a built-in.

    A file in ``~/.rapid-mlx/agents/`` replaces a built-in profile *wholesale*,
    so a hand-written ``codex.yaml`` that never mentions ``home_env`` silently
    removes the redirect and sends ``agents --setup`` back to the operator's
    real ``~/.codex/config.toml`` — the exact bug the redirect exists to stop,
    reintroduced by a file whose author was thinking about models, not safety.

    Overriding the *variable* is still allowed; only dropping it is refused.
    """
    if shadowed is None or not shadowed.config.home_env:
        return profile
    if profile.config.home_env:
        return profile
    logger.warning(
        "User profile %r omits config.home_env; inheriting %r from the built-in "
        "so `agents --setup` cannot write the real config.",
        profile.name,
        shadowed.config.home_env,
    )
    return dc_replace(
        profile,
        config=dc_replace(profile.config, home_env=shadowed.config.home_env),
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
                profile = _keep_home_env(profile, _PROFILES.get(profile.name))
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
    """Get an agent profile by name, or a minimal OPENAI_BASE_URL fallback.

    ``generic.yaml`` was removed in 0.10.2 — the Tier-1 profile set no
    longer contains a "generic" placeholder that would rank into
    ``list_profiles`` output. This helper still exists as a safety net
    for future callers that want a "look-up or default" ergonomic; it
    returns a hardcoded minimal env-var profile when the name is not
    known.
    """
    _ensure_loaded()
    profile = _PROFILES.get(name)
    if profile:
        return profile
    # Hardcoded fallback — mirrors what the removed generic.yaml offered.
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
