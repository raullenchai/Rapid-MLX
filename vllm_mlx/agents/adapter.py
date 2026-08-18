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


def _resolve_config_path(cfg) -> Path:
    """Where ``--setup`` should write this agent's config.

    Normally the profile's own path (``~/.codex/config.toml``). But every
    agent CLI we drive lets the user relocate its config directory with an
    environment variable — ``CODEX_HOME``, ``HERMES_HOME`` — and when the
    profile names that variable and it is set, we write *there* instead.

    That is what lets a test harness point ``--setup`` at a throwaway
    directory. The alternative it replaces — back up the operator's config,
    overwrite it, restore it afterwards — fails in two ways that are hard to
    notice: the restore is skipped entirely on SIGKILL, and once a config has
    been clobbered by any run, every subsequent run faithfully backs up and
    restores the *damaged* file, so the breakage looks freshly caused each
    time while actually dating from the first incident.

    Only the file name is taken from the profile path, so a relocated home
    never inherits the real one's directory layout.
    """
    assert cfg.path
    default = Path(os.path.expanduser(cfg.path))
    home_env = getattr(cfg, "home_env", None)
    if not home_env:
        return default
    override = os.environ.get(home_env, "").strip()
    if not override:
        return default
    return Path(os.path.expanduser(override)) / default.name


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
    dry_run: bool = False,
) -> str:
    """Write the agent's config file or print env vars to set up the integration.

    For file-based configs (YAML/JSON/TOML), if the config file already
    exists it is *merged* rather than overwritten — user customizations
    are preserved while connection details are updated.

    When *dry_run* is set, the merge is computed and described but nothing is
    written. ``--dry-run`` used to be accepted and then ignored on this path,
    so asking for a preview silently rewrote the operator's real config —
    the opposite of what the flag promises.

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
        config_path = _resolve_config_path(cfg)
        if not dry_run:
            config_path.parent.mkdir(parents=True, exist_ok=True)

        if profile.name == "codex" and isinstance(rendered, str):
            import json

            from .codex_catalog import build_codex_model_info

            catalog_path = config_path.parent / "rapid-mlx-model-catalog.json"
            rendered = rendered.replace(
                "{model_catalog_path}", str(catalog_path.resolve())
            )
            catalog = {"models": [build_codex_model_info(model_id, context_length)]}
            if not dry_run:
                try:
                    _atomic_write(
                        catalog_path,
                        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                    )
                except OSError as exc:
                    return (
                        f"Cannot write Codex model catalog to {catalog_path} "
                        f"({exc}). Check file permissions."
                    )

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

        if dry_run:
            if merged_text == rendered:
                return f"Would write config to {config_path} (nothing written)"
            return (
                f"Would merge config into {config_path}, preserving custom keys "
                "(nothing written)"
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
            if cfg.type == "toml":
                # Say it plainly: the round trip through the parser keeps
                # every key but drops comments, and a user who hand-wrote
                # notes into ~/.codex/config.toml should hear that from us
                # rather than discover it.
                summary += "; comments were not"
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

    # Unknown type — overwrite without reading (no merge support).
    if config_type not in ("yaml", "json", "toml"):
        return rendered

    # Let OSError propagate — caller must not silently overwrite an
    # unreadable file.
    existing_text = existing_path.read_text(encoding="utf-8")

    if config_type == "yaml":
        return _merge_yaml(existing_text, rendered)
    if config_type == "toml":
        return _merge_toml(existing_text, rendered)
    return _merge_json(existing_text, rendered)


def _merge_toml(existing_text: str, rendered: str) -> str:
    """Parse both TOML strings, deep-merge, and re-serialize.

    Same error semantics as ``_merge_yaml``.

    Until this existed, ``toml`` fell into the "unknown type" branch above
    and ``--setup`` replaced the file wholesale. Codex is the only TOML
    profile, so one ``rapid-mlx agents codex --setup`` deleted the user's
    ``approval_policy``, every ``[mcp_servers.*]`` block and every
    ``[projects.*]`` trust entry — with no backup, and a summary line that
    said a neutral "Wrote config to …" while the code already knew it had
    overwritten rather than merged (issue #1532).

    Comments do not survive the round trip, which is the same trade the
    JSON and YAML paths have always made. Losing the formatting of a
    hand-tuned config is an annoyance; losing its contents is data loss.
    """
    import tomli_w

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover — Python 3.10 only
        # ``tomllib`` is stdlib from 3.11; ``pyproject`` still supports 3.10,
        # where a bare import would abort --setup with ModuleNotFoundError on
        # the merge path only — every fresh-write test would stay green.
        # ``tomli`` is the same parser under its pre-stdlib name and is now a
        # runtime dependency under that marker, not just a [test] one.
        import tomli as tomllib

    if not existing_text.strip():
        return rendered
    try:
        existing = tomllib.loads(existing_text)
    except Exception as exc:
        raise _MergeParseError(f"invalid TOML: {exc}") from exc
    try:
        template = tomllib.loads(rendered)
    except Exception as exc:
        raise _MergeParseError(f"rendered template is not valid TOML: {exc}") from exc
    merged = _deep_merge(existing, template)
    return tomli_w.dumps(merged)


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


def fetch_reasoning_support(base_url: str, model_id: str) -> bool | None:
    """Whether *model_id* can emit reasoning, as a THREE-state answer.

    ``True``  — the served model advertises a reasoning parser.
    ``False`` — the entry was found and explicitly advertises none.
    ``None``  — unknown: server unreachable, model not listed, or an
                older rapid-mlx whose ``/v1/models`` predates the field.

    The three states are not decoration. A client that is told a model
    supports graded reasoning shows the user a control for it, so
    guessing in either direction has a cost: claim support a model
    doesn't have and the control is a lie, deny support it does have and
    a working feature disappears. Callers are expected to leave their
    current behaviour alone on ``None`` and only act on ``False``.

    Distinguishing ``False`` from ``None`` relies on ``/v1/models``
    serializing nulls — ``routes/models.py`` deliberately does not set
    ``exclude_none`` "so the shape is stable", so a reasoning-less model
    sends ``"reasoning_parser": null`` while a server too old to know
    about the field omits the key entirely. If that ever changes, this
    collapses to ``None`` (unknown) rather than to a wrong answer.
    """

    def _from_entry(entry: dict) -> bool | None:
        if "reasoning_parser" not in entry:
            return None
        parser = entry.get("reasoning_parser")
        return bool(isinstance(parser, str) and parser.strip())

    models = _fetch_models(base_url)
    for m in models:
        if m.get("id") == model_id:
            return _from_entry(m)
    # Same single-model fallback rule as fetch_context_window: a
    # multi-model serve must match exactly rather than describe some
    # other model's capabilities.
    if len(models) == 1:
        return _from_entry(models[0])
    return None


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
