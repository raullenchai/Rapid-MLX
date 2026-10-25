"""Best-effort, local first-serve attribution; no prompts or network probes."""

from __future__ import annotations

import time

from rapid_mlx.first_run import _state_dir


def record_serve_success() -> None:
    """Remember the first readiness, including quiet and piped CLI sessions."""
    try:
        marker = _state_dir() / "first_serve_success"
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x"):
            pass
    except OSError:
        pass


def track_next_step() -> None:
    try:
        from rapid_mlx.telemetry.track import track

        track("onboarding_next_step_shown", {})
    except Exception:
        pass


def track_first_integration(agent: str) -> None:
    """Claim once, only with consent and within 24 hours of first readiness."""
    try:
        from rapid_mlx.agents.telemetry import _telemetry_agent
        from rapid_mlx.telemetry.track import _enqueue_accepted, would_accept

        age = time.time() - (_state_dir() / "first_serve_success").stat().st_mtime
        if not 0 <= age <= 24 * 60 * 60:
            return
        accepted = would_accept("first_integration", {"agent": _telemetry_agent(agent)})
        if accepted is None:
            return
        marker = _state_dir() / "first_integration_seen"
        with marker.open("x"):
            pass
        try:
            queued = _enqueue_accepted(accepted)
        except Exception:
            marker.unlink(missing_ok=True)
            return
        if not queued:
            marker.unlink(missing_ok=True)
    except Exception:
        pass


def detected_launch_agents() -> list[str]:
    """Use launch capabilities and profile binaries without starting a shell."""
    try:
        import shutil

        from rapid_mlx.agents import get_profile
        from rapid_mlx.integrations import list_launch_targets

        agents = []
        for target in list_launch_targets():
            if target.capability != "launchable":
                continue
            profile = get_profile(target.id)
            if (
                profile
                and profile.testing.binary
                and shutil.which(profile.testing.binary)
            ):
                agents.append(target.id)
        return agents
    except Exception:
        return []
