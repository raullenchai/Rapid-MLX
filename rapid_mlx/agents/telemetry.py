# SPDX-License-Identifier: Apache-2.0
"""Best-effort telemetry shared by both agent configuration paths."""

from __future__ import annotations


def _telemetry_agent(agent: str) -> str:
    """Return the registry-approved agent label, falling back to ``other``."""
    from rapid_mlx.telemetry.registry import load_registry

    values = load_registry()["enums"]["agent"]["values"]
    return agent if agent in values and agent != "other" else "other"


def track_agent_configured(agent: str) -> None:
    """Record one completed agent configuration."""
    try:
        from rapid_mlx.telemetry.track import track

        track("agent_configured", {"agent": _telemetry_agent(agent)})
    except Exception:
        return


def track_agent_configure_failed(error_class: str, agent: str | None = None) -> None:
    """Record one failed agent configuration with closed registry labels."""
    try:
        from rapid_mlx.telemetry.track import track

        props = {"error_class": error_class}
        if agent is not None:
            props["agent"] = _telemetry_agent(agent)
        track("agent_configure_failed", props)
    except Exception:
        return
