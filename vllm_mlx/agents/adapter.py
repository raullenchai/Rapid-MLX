"""Agent adapter — apply an agent profile to the runtime.

Bridges between the declarative AgentProfile and the server's runtime
components (streaming filters, config files, test generation).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import AgentProfile

logger = logging.getLogger(__name__)


def setup_agent_config(
    profile: AgentProfile,
    base_url: str = "http://localhost:8000/v1",
    model_id: str = "default",
    agent_version: str | None = None,
) -> str:
    """Write the agent's config file or print env vars to set up the integration.

    Returns a human-readable summary of what was done.
    """
    rendered = profile.render_config(base_url, model_id, agent_version)
    cfg = profile.get_config_for_version(agent_version)

    if cfg.type == "env":
        lines = []
        for key, val in rendered.items():
            lines.append(f"  export {key}={val}")
        summary = (
            "Run these commands in your shell:\n"
            + "\n".join(lines)
            + "\n\n  (env vars are not persistent — add to your .zshrc/.bashrc for permanent setup)"
        )
        return summary

    if cfg.path:
        config_path = Path(os.path.expanduser(cfg.path))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(rendered)
        summary = f"Wrote config to {config_path}"
        return summary

    return "No config to write (template not specified)"


def get_setup_instructions(
    profile: AgentProfile,
    base_url: str = "http://localhost:8000/v1",
    model_id: str = "default",
    agent_version: str | None = None,
) -> str:
    """Get human-readable setup instructions for an agent."""
    cfg = profile.get_config_for_version(agent_version)
    rendered = profile.render_config(base_url, model_id, agent_version)
    testing = profile.get_testing_for_version(agent_version)

    lines = [
        f"# {profile.display_name} + Rapid-MLX Setup",
        "",
        "## 1. Start Rapid-MLX",
        "",
    ]

    if profile.recommended_models:
        model = profile.recommended_models[0]
        lines.append("```bash")
        lines.append(f"rapid-mlx serve {model}")
        lines.append("```")
    else:
        lines.append("```bash")
        lines.append("rapid-mlx serve <MODEL>")
        lines.append("```")

    lines.append("")
    lines.append(f"## 2. Configure {profile.display_name}")
    lines.append("")

    if cfg.type == "env":
        lines.append("```bash")
        for key, val in rendered.items():
            lines.append(f"export {key}={val}")
        lines.append("```")
    elif cfg.path:
        ext = Path(cfg.path).suffix.lstrip(".")
        lines.append(f"Write to `{cfg.path}`:")
        lines.append(f"```{ext}")
        lines.append(rendered.rstrip())
        lines.append("```")

    if testing and testing.install_cmd:
        lines.append("")
        lines.append(f"## 3. Install {profile.display_name}")
        lines.append("")
        lines.append("```bash")
        lines.append(testing.install_cmd)
        lines.append("```")

    if profile.known_issues:
        lines.append("")
        lines.append("## Known Issues")
        lines.append("")
        for issue in profile.known_issues:
            lines.append(f"- {issue}")

    return "\n".join(lines)


def apply_streaming_config(profile: AgentProfile,
                           agent_version: str | None = None):
    """Inject agent-specific streaming filter tags into the global registry.

    This is called at server startup or when an agent profile is activated,
    to extend the streaming filter with agent-specific patterns.

    Uses the register_tool_call_tag() API from api/utils.py rather than
    directly mutating the list — ensures proper dedup and future extensibility.

    Args:
        profile: The agent profile to apply
        agent_version: Optional version to match version-specific config
    """
    from vllm_mlx.api.utils import register_tool_call_tag

    streaming = profile.get_streaming_for_version(agent_version)
    if not streaming.extra_tool_tags:
        return

    added = 0
    for tag_pair in streaming.extra_tool_tags:
        if register_tool_call_tag(tag_pair[0], tag_pair[1]):
            added += 1

    if added:
        logger.info(
            f"Applied {added} extra streaming filter tags from "
            f"agent profile '{profile.name}'"
        )


def get_extra_tags_for_profile(profile: AgentProfile,
                               agent_version: str | None = None,
                               ) -> list[tuple[str, str]]:
    """Get extra streaming tags from a profile (for per-request filter creation).

    Instead of mutating global state, this returns the tags so they can be
    passed to StreamingToolCallFilter(extra_tags=...) at request time.
    """
    streaming = profile.get_streaming_for_version(agent_version)
    return list(streaming.extra_tool_tags)
