# SPDX-License-Identifier: Apache-2.0
"""One served context budget for agent configuration, online and offline."""

from __future__ import annotations

# Claude Code subtracts its own output/autocompact buffer from the window:
# https://code.claude.com/docs/en/model-config#context-window-and-auto-compaction
# Codex applies effective_context_window_percent and its auto-compact limit:
# https://github.com/openai/codex/blob/main/codex-rs/protocol/src/openai_models.rs
# OpenCode reserves limit.output inside limit.context before compaction:
# https://opencode.ai/docs/models/#limits
# Hermes retains the supported 64K minimum and 128K cap:
# https://hermes-agent.nousresearch.com/docs/user-guide/configuration/
# Other templates (including pi, DSH and Qwen Code) conservatively reserve 10%;
# their compaction semantics are not assumed here.
_HERMES_MIN_CONTEXT = 64_000
_HERMES_MAX_CONTEXT = 131_072
_FALLBACK_WINDOWS = {"hermes-isolated": _HERMES_MIN_CONTEXT, "codex-catalog": 262_144}
_WINDOW_FRACTIONS = {"claude": 1.0, "claude-code": 1.0, "codex": 1.0, "hermes": 1.0}


def context_fallback(agent: str, default: int | None = 32768) -> int | None:
    """Preserve each writer's historical unknown-model fallback centrally."""
    return _FALLBACK_WINDOWS.get(agent, default)


def context_output_budget(context: int) -> int:
    """Reserve output inside OpenCode's total context window.

    OpenCode compacts once input exceeds ``context - output``, so on small
    memory-fitted windows the reservation must stay a fraction of the window
    or every turn would trigger compaction.
    """
    return min(8192, max(1, context // 4))


def _entry_ceiling(entry: dict) -> int | None:
    from .adapter import _valid_context_window

    values = [
        value
        for field in ("max_model_len", "context_window")
        if (value := _valid_context_window(entry.get(field))) is not None
    ]
    return min(values) if values else None


def served_context_ceiling(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    require_served: bool = False,
) -> int | None:
    """Read the served ceiling; multi-model servers require an exact match."""
    from .adapter import _fetch_models, fetch_context_window

    # Keep the existing model lookup shared with setup callers. Authenticated
    # and runtime-only probes additionally need the full model entry below.
    if not api_key and not require_served:
        return fetch_context_window(base_url, model)

    models = (
        _fetch_models(base_url, api_key=api_key) if api_key else _fetch_models(base_url)
    )
    for entry in models:
        if entry.get("id") == model:
            return (
                None
                if require_served and not entry.get("max_model_len")
                else _entry_ceiling(entry)
            )
    if len(models) == 1 and (not require_served or models[0].get("max_model_len")):
        return _entry_ceiling(models[0])
    return None


def offline_context_ceiling(model: str) -> int | None:
    """Estimate auto-mode memory fit using cached metadata, never loading weights."""
    from rapid_mlx.byom.preflight import (
        mlx_recommended_working_set_bytes,
        physical_ram_bytes,
    )
    from rapid_mlx.kv_estimation import footprint_from_config, memory_fit_context
    from rapid_mlx.memory_budget import plan_metal_limit
    from rapid_mlx.model_metadata import read_model_metadata
    from rapid_mlx.recommendations import recommendation_footprint_gb
    from rapid_mlx.run.cli import _hf_id

    try:
        metadata = read_model_metadata(_hf_id(model))
        config = metadata.config if metadata else None
        if not isinstance(config, dict):
            return None
        from .adapter import _valid_context_window

        text = config.get("text_config")
        text = text if isinstance(text, dict) else config
        native = _valid_context_window(
            text.get("max_position_embeddings")
        ) or _valid_context_window(config.get("max_position_embeddings"))
        estimate = footprint_from_config(config)
        weights = 0
        snapshot = getattr(metadata, "snapshot_dir", None)
        if snapshot is not None:
            weights = sum(
                path.stat().st_size
                for pattern in ("*.safetensors", "*.npz")
                for path in snapshot.glob(pattern)
                if path.is_file()
            )
        if not weights:
            footprint = recommendation_footprint_gb(model)
            weights = int(footprint * 1024**3) if footprint else 0
        if estimate is None or not weights:
            return native
        device = mlx_recommended_working_set_bytes() or physical_ram_bytes()
        if not device:
            return native
        plan = plan_metal_limit(weights_bytes=weights, device_budget_bytes=device)
        return memory_fit_context(estimate, plan.limit_bytes - weights, native)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def agent_context_budget(
    agent: str,
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    fallback: int | None = None,
    offline: bool = False,
) -> int | None:
    """Resolve the model ceiling and apply the client's window semantics once."""
    ceiling = (
        None if offline else served_context_ceiling(base_url, model, api_key=api_key)
    )
    if ceiling is None:
        ceiling = offline_context_ceiling(model)
    return context_budget_for_ceiling(agent, ceiling, model=model, fallback=fallback)


def context_budget_for_ceiling(
    agent: str,
    ceiling: int | None,
    *,
    model: str = "model",
    fallback: int | None = None,
) -> int | None:
    """Apply the client policy to an already-resolved ceiling."""
    if ceiling is None:
        return context_fallback(agent, fallback)
    window = max(1, int(ceiling * _WINDOW_FRACTIONS.get(agent, 0.90)))
    if agent == "hermes":
        window = min(window, _HERMES_MAX_CONTEXT)
        if window < _HERMES_MIN_CONTEXT:
            raise ValueError(
                f"Hermes needs at least 64,000 context tokens, but {model} can serve "
                f"only {ceiling:,}. Choose a 64K-capable model or raise the server context limit."
            )
    return window
