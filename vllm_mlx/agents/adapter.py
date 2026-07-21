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


class _MergeParseError(Exception):
    """Raised when an existing config file cannot be parsed for merging."""


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    - Dict values are merged recursively (existing keys in *base* that
      are absent from *override* are preserved).
    - All other types in *override* win unconditionally.

    Returns a new dict — neither input is mutated.
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            # Lists and scalars: template value wins unconditionally.
            # This ensures template-defined toolsets are authoritative
            # (user customizations at the dict-key level are preserved,
            # but list contents come from the template).
            merged[key] = val
    return merged


def _atomic_write(target: Path, content: str) -> None:
    """Write *content* to *target* atomically, preserving symlinks and mode.

    When *target* already exists, its mode bits are copied to the
    replacement file.  When it does not exist, a simple ``write_text``
    is used so no metadata is lost (there's nothing to preserve).
    Symlinks are resolved before writing so dotfile-managed configs
    stay connected to their real target.
    """
    import stat
    import tempfile

    resolved = target.resolve()

    # If the file doesn't exist yet, a plain write is safe and
    # avoids the metadata-preservation question entirely.
    if not resolved.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return

    mode = stat.S_IMODE(resolved.stat().st_mode)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=".rapid-mlx-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, str(resolved))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def setup_agent_config(
    profile: AgentProfile,
    base_url: str = "http://localhost:8000/v1",
    model_id: str = "default",
    agent_version: str | None = None,
    *,
    context_length: int | None = None,
) -> str:
    """Write the agent's config file or print env vars to set up the integration.

    For file-based configs (YAML/JSON), if the config file already exists
    it is *merged* rather than overwritten — user customizations are
    preserved while connection details are updated.

    Returns a human-readable summary of what was done.
    """
    rendered = profile.render_config(
        base_url, model_id, agent_version, context_length=context_length
    )
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

        try:
            merged_text = _merge_file_config(config_path, rendered, cfg.type)
        except OSError as exc:
            return (
                f"Cannot read existing config at {config_path} ({exc}). "
                "Remove or fix it manually, then re-run --setup."
            )
        except _MergeParseError as exc:
            return (
                f"Cannot parse existing config at {config_path} ({exc}). "
                "Fix or remove it manually, then re-run --setup."
            )

        try:
            _atomic_write(config_path, merged_text)
        except OSError as exc:
            return (
                f"Cannot write config to {config_path} ({exc}). Check file permissions."
            )

        if merged_text == rendered:
            summary = f"Wrote config to {config_path}"
        else:
            summary = f"Merged config into {config_path} (custom keys preserved)"
        return summary

    return "No config to write (template not specified)"


def _merge_file_config(existing_path: Path, rendered: str, config_type: str) -> str:
    """Merge *rendered* template into an existing config file.

    Returns *rendered* unchanged when the file does not exist (fresh
    write).  Raises ``OSError`` when the file exists but cannot be read
    (caller should NOT overwrite in that case).
    """
    if not existing_path.exists():
        return rendered

    # toml / unknown — overwrite without reading (no merge support)
    if config_type not in ("yaml", "json"):
        return rendered

    # Let OSError propagate — caller must not silently overwrite an
    # unreadable file.
    existing_text = existing_path.read_text(encoding="utf-8")

    if config_type == "yaml":
        return _merge_yaml(existing_text, rendered)
    return _merge_json(existing_text, rendered)


def _merge_yaml(existing_text: str, rendered: str) -> str:
    """Parse both YAML strings, deep-merge, and re-serialize.

    Raises ``_MergeParseError`` when the existing content is malformed
    or not a mapping — the caller decides how to report the failure.
    Empty existing files are treated as a fresh write (no error).
    """
    import yaml

    if not existing_text.strip():
        return rendered
    try:
        existing = yaml.safe_load(existing_text)
    except Exception as exc:
        raise _MergeParseError(f"invalid YAML: {exc}") from exc
    if existing is None:
        return rendered
    if not isinstance(existing, dict):
        raise _MergeParseError("existing config is not a YAML mapping")
    try:
        template = yaml.safe_load(rendered)
    except Exception as exc:
        raise _MergeParseError(f"rendered template is not valid YAML: {exc}") from exc
    if not isinstance(template, dict):
        raise _MergeParseError("rendered template is not a YAML mapping")
    merged = _deep_merge(existing, template)
    return yaml.dump(merged, default_flow_style=False, sort_keys=False)


def _merge_json(existing_text: str, rendered: str) -> str:
    """Parse both JSON strings, deep-merge, and re-serialize.

    Same error semantics as ``_merge_yaml``.
    """
    import json

    if not existing_text.strip():
        return rendered
    try:
        existing = json.loads(existing_text)
    except Exception as exc:
        raise _MergeParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(existing, dict):
        raise _MergeParseError("existing config is not a JSON object")
    try:
        template = json.loads(rendered)
    except Exception as exc:
        raise _MergeParseError(f"rendered template is not valid JSON: {exc}") from exc
    if not isinstance(template, dict):
        raise _MergeParseError("rendered template is not a JSON object")
    merged = _deep_merge(existing, template)
    return json.dumps(merged, indent=2, ensure_ascii=False) + "\n"


def _valid_context_window(value) -> int | None:
    """Return *value* only when it is a positive, non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _fetch_models(base_url: str) -> list[dict]:
    """Fetch the ``/models`` listing from a running server, or ``[]``.

    Validates that the response ``data`` field is a list of mappings;
    malformed responses collapse to ``[]``.
    """
    import json
    import urllib.request

    try:
        url = base_url.rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=2) as resp:
            body = json.loads(resp.read())
            entries = body.get("data") if isinstance(body, dict) else None
            if not isinstance(entries, list):
                return []
            return [e for e in entries if isinstance(e, dict)]
    except Exception:
        return []


def _detect_running_model(base_url: str) -> tuple[str | None, int | None]:
    """Try to detect the model and its context window from the server.

    Returns ``(model_id, context_window)`` — either or both may be
    ``None`` when the server is unreachable or doesn't report the field.
    """
    models = _fetch_models(base_url)
    chosen = None
    # Prefer short alias over full HF path
    for m in models:
        mid = m.get("id")
        if not isinstance(mid, str):
            continue
        if "/" not in mid and mid != "default":
            chosen = m
            break
    if chosen is None and models:
        chosen = models[0]
    if chosen is not None:
        model_id = chosen.get("id")
        if not isinstance(model_id, str) or not model_id:
            model_id = "default"
        ctx = _valid_context_window(chosen.get("context_window"))
        return model_id, ctx
    return None, None


def fetch_context_window(base_url: str, model_id: str) -> int | None:
    """Fetch ``context_window`` for a specific *model_id* from the server.

    Iterates the ``/v1/models`` listing and returns the context window
    for the entry whose ``id`` matches *model_id*.  Only falls back to
    the first entry when exactly one model is served (single-model
    serve); multi-model servers require an exact match to avoid
    advertising the wrong context window.
    """
    models = _fetch_models(base_url)
    # Exact match first
    for m in models:
        if m.get("id") == model_id:
            return _valid_context_window(m.get("context_window"))
    # Fallback only for single-model serve
    if len(models) == 1:
        return _valid_context_window(models[0].get("context_window"))
    return None


def get_setup_instructions(
    profile: AgentProfile,
    base_url: str = "http://localhost:8000/v1",
    model_id: str = "default",
    agent_version: str | None = None,
    *,
    context_length: int | None = None,
) -> str:
    """Get human-readable setup instructions for an agent."""
    # Auto-detect running model if not explicitly set
    if model_id == "default":
        detected_model, detected_ctx = _detect_running_model(base_url)
        if detected_model:
            model_id = detected_model
        if context_length is None and detected_ctx is not None:
            context_length = detected_ctx

    cfg = profile.get_config_for_version(agent_version)
    rendered = profile.render_config(
        base_url, model_id, agent_version, context_length=context_length
    )
    testing = profile.get_testing_for_version(agent_version)

    lines = [
        f"# {profile.display_name} + Rapid-MLX Setup",
        "",
        "## 1. Start Rapid-MLX",
        "",
    ]

    serve_model = (
        model_id
        if model_id != "default"
        else (
            profile.recommended_models[0] if profile.recommended_models else "<MODEL>"
        )
    )
    if profile.recommended_models:
        lines.append("```bash")
        cmd = f"rapid-mlx serve {serve_model}"
        if len(profile.recommended_models) > 1:
            cmd += "  # or any model below"
        lines.append(cmd)
        lines.append("```")
        if len(profile.recommended_models) > 1:
            lines.append("")
            lines.append("Recommended models:")
            for m in profile.recommended_models:
                lines.append(f"- `{m}`")
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


def apply_streaming_config(profile: AgentProfile, agent_version: str | None = None):
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


def get_extra_tags_for_profile(
    profile: AgentProfile,
    agent_version: str | None = None,
) -> list[tuple[str, str]]:
    """Get extra streaming tags from a profile (for per-request filter creation).

    Instead of mutating global state, this returns the tags so they can be
    passed to StreamingToolCallFilter(extra_tags=...) at request time.
    """
    streaming = profile.get_streaming_for_version(agent_version)
    return list(streaming.extra_tool_tags)
