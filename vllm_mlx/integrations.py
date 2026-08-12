# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for desktop/CLI integration discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from vllm_mlx.agents import list_profiles
from vllm_mlx.launch import ADAPTERS


@dataclass(frozen=True)
class IntegrationTarget:
    id: str
    name: str
    kind: str
    config_path: str | None


_LAUNCH_NAMES = {
    "cline": "Cline",
    "claude-code": "Claude Code",
    "continue-dev": "Continue.dev",
    "cursor": "Cursor",
}
_PROFILE_TO_LAUNCH = {"continue": "continue-dev"}
_CONFIG_DESTINATIONS = {
    "cline": "Cline's VS Code settings",
    "claude-code": "~/.claude/settings.json",
    "continue-dev": "~/.continue/config.json",
    "cursor": "Cursor's VS Code settings",
}


def list_integration_targets() -> list[IntegrationTarget]:
    """Return every supported target, merging duplicate setup/profile entries.

    Config-writing targets win when the same product is also represented by an
    adapter profile.  The launch registry supplies display order; remaining
    profiles retain the agents registry's order.
    """
    targets: list[IntegrationTarget] = []
    seen: set[str] = set()
    for target_id, adapter in ADAPTERS.items():
        targets.append(
            IntegrationTarget(
                id=target_id,
                name=_LAUNCH_NAMES.get(target_id, target_id),
                kind="config_writer",
                config_path=_CONFIG_DESTINATIONS[target_id],
            )
        )
        seen.add(target_id)

    for profile in list_profiles():
        target_id = _PROFILE_TO_LAUNCH.get(profile.name, profile.name)
        if target_id in seen:
            continue
        targets.append(
            IntegrationTarget(
                id=target_id,
                name=profile.display_name,
                kind="adapter_profile",
                config_path=None,
            )
        )
        seen.add(target_id)
    return targets


def integration_targets_json() -> list[dict[str, str | None]]:
    return [asdict(target) for target in list_integration_targets()]
