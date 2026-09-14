# SPDX-License-Identifier: Apache-2.0
"""Model-specific budgets for the shared agent runtime."""

from __future__ import annotations

from .models import AgentProfile

MINICPM5_2B_PROFILE = AgentProfile(
    name="minicpm5-2b",
    max_visible_tools=6,
    max_tool_rounds=8,
    repeated_call_limit=2,
)

DEFAULT_PROFILE = AgentProfile(
    name="default",
    max_visible_tools=12,
    max_tool_rounds=12,
    repeated_call_limit=2,
)


def resolve_agent_profile(model: str) -> AgentProfile:
    """Return budgets by model identity, never by client/UI identity."""

    normalized = model.casefold().replace("_", "-")
    if "minicpm5-2b" in normalized:
        return MINICPM5_2B_PROFILE
    return DEFAULT_PROFILE
