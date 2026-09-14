# SPDX-License-Identifier: Apache-2.0
"""Model-specific budgets for the shared agent runtime."""

from __future__ import annotations

import re
from typing import Any

from .models import AgentProfile

MINICPM5_2B_PROFILE = AgentProfile(
    name="minicpm5-2b",
    # Preserve the original six-connector budget and reserve two additional
    # slots for Rapid's bounded host helpers. They are not substitutes for
    # tools the operator deliberately configured.
    max_visible_tools=8,
    # Physical Desktop dogfood showed a two-file + exact-calculation task
    # exhausting eight rounds before synthesis. Keep the six-connector
    # bound, repeat guard, and output ceiling, but allow the same twelve
    # bounded rounds as the default profile so ordinary organizing work can
    # finish instead of surfacing an internal budget code.
    max_tool_rounds=12,
    repeated_call_limit=2,
    max_output_tokens=900,
)

DEFAULT_PROFILE = AgentProfile(
    name="default",
    max_visible_tools=14,
    max_tool_rounds=12,
    repeated_call_limit=2,
)

CONSERVATIVE_LOCAL_PROFILE = AgentProfile(
    name="default-conservative",
    max_visible_tools=8,
    max_tool_rounds=8,
    repeated_call_limit=2,
    max_output_tokens=900,
)

_MINICPM5_2B_CATALOG_IDENTITIES = frozenset(
    {
        "minicpm5-2b-4bit",
        "openbmb/minicpm5-2b",
        "openbmb/minicpm5-2b-mlx",
        "mlx-community/minicpm5-2b-8bit",
    }
)
_MINICPM5_2B_TRUSTED_REPO = re.compile(
    r"^(?:openbmb|mlx-community)/minicpm5-2b(?:-(?:mlx|4bit|8bit|bf16))?$"
)


def _is_minicpm5_2b_config(config: dict[str, Any] | None) -> bool:
    """Recognize the released dense 2B checkpoint from loaded metadata.

    MiniCPM5 deliberately uses the generic ``LlamaForCausalLM`` architecture,
    so a local directory has no family-bearing architecture name. Match the
    complete stable shape instead of trusting directory or served-alias text.
    """

    if config is None:
        return False
    return all(
        config.get(key) == value
        for key, value in {
            "model_type": "llama",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 42,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 130560,
        }.items()
    )


def resolve_agent_profile(
    model: str,
    *,
    model_config: dict[str, Any] | None = None,
    tool_call_parser: str | None = None,
) -> AgentProfile:
    """Return budgets from catalog identity or exact loaded-model metadata."""

    normalized = model.casefold().replace("_", "-")
    catalog_match = (
        normalized in _MINICPM5_2B_CATALOG_IDENTITIES
        or _MINICPM5_2B_TRUSTED_REPO.fullmatch(normalized) is not None
    )
    if catalog_match or (
        tool_call_parser == "minicpm" and _is_minicpm5_2b_config(model_config)
    ):
        return MINICPM5_2B_PROFILE
    if "minicpm5-2b" in normalized:
        # Preserve the old conservative limits for local names without
        # granting them a verified MiniCPM identity.
        return CONSERVATIVE_LOCAL_PROFILE
    return DEFAULT_PROFILE
