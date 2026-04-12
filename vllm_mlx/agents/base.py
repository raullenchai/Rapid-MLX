"""Agent Profile — declarative description of what an agent needs from the inference server.

Each profile captures an agent's configuration format, model preferences,
streaming quirks, and known issues. Adding a new agent = adding a YAML file.

Profiles are versioned to handle agent evolution — when an agent changes its
config format or API in a new version, add a new version block rather than
breaking the old one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AgentConfigSpec:
    """How to configure the agent to point at our API."""

    type: str  # "yaml" | "json" | "env" | "toml"
    path: str | None = None  # config file path (for yaml/json/toml)
    template: str | None = None  # config file template with {base_url}, {model_id}
    env_vars: dict[str, str] | None = None  # env vars for "env" type


@dataclass
class AgentStreamingSpec:
    """Agent-specific streaming behavior."""

    extra_tool_tags: list[tuple[str, str]] = field(default_factory=list)
    suppress_patterns: list[str] = field(default_factory=list)
    max_tools: int | None = None  # how many tools the agent injects per request


@dataclass
class AgentTestingSpec:
    """Integration testing configuration."""

    binary: str | None = None  # path to agent binary
    query_cmd: str | None = None  # command template: "hermes chat -q '{query}'"
    query_timeout: int = 120
    install_cmd: str | None = None  # how to install the agent
    # Framework-specific test module path (relative to tests/integrations/)
    # e.g. "test_pydantic_ai_full.py" — will be imported and run after base tests
    specific_tests: str | None = None


@dataclass
class AgentVersionSpec:
    """Version-specific overrides.

    Agents evolve fast — config formats change, tools get added/removed,
    APIs shift. Instead of breaking old configs, we version them.

    The profile loader picks the best matching version:
    1. Exact match (e.g., "1.2.3")
    2. Semver range match (e.g., ">=1.0,<2.0")
    3. Fallback to base profile (no version constraint)
    """

    version_range: str  # semver range: ">=0.8,<1.0" or ">=1.0"
    config: AgentConfigSpec | None = None  # override config for this version
    streaming: AgentStreamingSpec | None = None  # override streaming
    testing: AgentTestingSpec | None = None  # override testing
    notes: str | None = None  # migration notes


@dataclass
class AgentProfile:
    """Complete agent profile — everything needed to integrate an agent."""

    # Identity
    name: str  # "hermes", "goose", "claude-code"
    display_name: str  # "Hermes Agent"
    repo: str | None = None  # "NousResearch/hermes-agent"
    stars: int | None = None  # for prioritization

    # Base configuration (used when no version match)
    config: AgentConfigSpec = field(default_factory=lambda: AgentConfigSpec("env"))

    # Model compatibility
    recommended_models: list[str] = field(default_factory=list)
    parser_override: str | None = None  # force a specific tool parser

    # Streaming
    streaming: AgentStreamingSpec = field(default_factory=AgentStreamingSpec)

    # Testing
    testing: AgentTestingSpec = field(default_factory=AgentTestingSpec)

    # Version-specific overrides (newest first)
    versions: list[AgentVersionSpec] = field(default_factory=list)

    # Known issues (human-readable, for docs and --info output)
    known_issues: list[str] = field(default_factory=list)

    # Capabilities the agent expects from the server
    needs_function_calling: bool = True
    needs_streaming: bool = True
    needs_vision: bool = False
    needs_reasoning: bool = False

    def get_config_for_version(self, agent_version: str | None = None) -> AgentConfigSpec:
        """Get the config spec, optionally matching an agent version."""
        if agent_version and self.versions:
            for vs in self.versions:
                if _version_matches(agent_version, vs.version_range):
                    if vs.config:
                        return vs.config
        return self.config

    def get_streaming_for_version(self, agent_version: str | None = None) -> AgentStreamingSpec:
        """Get streaming spec, optionally matching an agent version."""
        if agent_version and self.versions:
            for vs in self.versions:
                if _version_matches(agent_version, vs.version_range):
                    if vs.streaming:
                        return vs.streaming
        return self.streaming

    def get_testing_for_version(self, agent_version: str | None = None) -> AgentTestingSpec:
        """Get testing spec, optionally matching an agent version."""
        if agent_version and self.versions:
            for vs in self.versions:
                if _version_matches(agent_version, vs.version_range):
                    if vs.testing:
                        return vs.testing
        return self.testing

    def render_config(self, base_url: str, model_id: str,
                      agent_version: str | None = None) -> str | dict[str, str]:
        """Render the config file content or env vars for this agent.

        Returns:
            str for file-based configs (yaml/json/toml)
            dict for env-based configs
        """
        cfg = self.get_config_for_version(agent_version)
        base_url_no_v1 = base_url.rstrip("/").removesuffix("/v1")

        if cfg.type == "env":
            if not cfg.env_vars:
                return {}
            rendered = {}
            for key, val in cfg.env_vars.items():
                rendered[key] = (
                    val.replace("{base_url}", base_url)
                    .replace("{model_id}", model_id)
                    .replace("{base_url_no_v1}", base_url_no_v1)
                )
            return rendered

        if cfg.template:
            return (
                cfg.template.replace("{base_url}", base_url)
                .replace("{model_id}", model_id)
                .replace("{base_url_no_v1}", base_url_no_v1)
            )
        return ""


def _version_matches(version: str, version_range: str) -> bool:
    """Check if a version string matches a range spec.

    Supports: ">=1.0", "<2.0", ">=1.0,<2.0", "1.2.3" (exact).
    """
    parts = _parse_version(version)
    if not parts:
        return False

    for constraint in version_range.split(","):
        constraint = constraint.strip()
        if constraint.startswith(">="):
            target = _parse_version(constraint[2:])
            if target and parts < target:
                return False
        elif constraint.startswith(">"):
            target = _parse_version(constraint[1:])
            if target and parts <= target:
                return False
        elif constraint.startswith("<="):
            target = _parse_version(constraint[2:])
            if target and parts > target:
                return False
        elif constraint.startswith("<"):
            target = _parse_version(constraint[1:])
            if target and parts >= target:
                return False
        elif constraint.startswith("==") or constraint.startswith("="):
            target = _parse_version(constraint.lstrip("="))
            if target and parts != target:
                return False
        else:
            # Bare version = exact match; unparseable = no match
            target = _parse_version(constraint)
            if target is None:
                return False
            if parts != target:
                return False
    return True


def _parse_version(v: str) -> tuple[int, ...] | None:
    """Parse "1.2.3" into (1, 2, 3)."""
    m = re.match(r"(\d+(?:\.\d+)*)", v.strip())
    if m:
        return tuple(int(x) for x in m.group(1).split("."))
    return None
