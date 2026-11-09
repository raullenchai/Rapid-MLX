# SPDX-License-Identifier: Apache-2.0
"""Model-free Desktop lifecycle bridge; Swift owns launch/day deduplication."""

from rapid_mlx.telemetry import consent_runtime, posthog_sender
from rapid_mlx.telemetry.track import set_surface_for_role, track


def report(trigger: str) -> None:
    """Use the normal consent, build, registry and PostHog gates, without inference.

    This entry never discloses or writes consent. Only Desktop-owned helpers
    may report; a regular CLI invocation is inert.
    """
    if not consent_runtime.is_desktop_sidecar():
        return
    set_surface_for_role()
    track("app_opened", {"trigger": trigger})
    posthog_sender.get_sender().flush(timeout=2.0)
