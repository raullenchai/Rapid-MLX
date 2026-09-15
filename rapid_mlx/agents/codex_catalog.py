# SPDX-License-Identifier: Apache-2.0
"""Static model metadata used by Codex for Rapid-MLX local models."""

from __future__ import annotations


def build_codex_model_info(model_id: str, context_window: int | None) -> dict:
    """Return a complete Codex model-catalog entry.

    The same descriptor is served dynamically by ``/v1/models`` and written
    by ``rapid-mlx agents codex --setup``.  Keeping one builder prevents the
    static first-turn catalog from drifting from later remote refreshes.
    """
    window = context_window or 262_144
    return {
        "slug": model_id,
        "display_name": model_id,
        "description": "Local model served by Rapid-MLX",
        "default_reasoning_level": "none",
        "supported_reasoning_levels": [
            {"effort": effort, "description": f"{effort} reasoning"}
            for effort in ("none", "low", "medium", "high")
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": (
            "You are Codex, a coding agent working directly in the user's "
            "repository. Use the available shell tools immediately to "
            "inspect, edit, and test. Do not merely announce an action and "
            "stop. Make repository changes with apply_patch instead of "
            "printing a proposed patch as prose. When exec_command is the "
            "selected tool, invoke the apply_patch executable through the "
            "shell. Gather evidence proportional to the task: inspect every "
            "implementation and test path needed for a grounded change, without "
            "using a fixed command limit. Before editing, identify every explicit acceptance "
            "constraint in the user's request; after editing, compare the diff "
            "against those constraints before declaring success. Never repeat an "
            "unchanged search, file read, or failing test command. Use the "
            "repository's supported interpreter and test toolchain, not an "
            "incompatible system default. If the user supplies an interpreter "
            "or working test command, preserve it for subsequent test runs. Once "
            "enough context is known, act "
            "immediately. Continue until the requested change and relevant tests "
            "are complete. Preserve unrelated user changes."
        ),
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "tokens", "limit": 10_000},
        "supports_parallel_tool_calls": True,
        "context_window": window,
        "max_context_window": window,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "tool_mode": "direct",
    }
