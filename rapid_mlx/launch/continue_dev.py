# SPDX-License-Identifier: Apache-2.0
"""Continue (VS Code / JetBrains extension and the ``cn`` CLI) launch adapter.

Continue reads its local configuration from ``~/.continue/config.yaml``
(``schema: v1``). The ``cn`` CLI reads only that file, and the IDE
extensions treat it as the primary config: when ``config.yaml`` exists the
legacy ``config.json`` is ignored, and ``config.json`` is used only while no
``config.yaml`` exists (Continue's ``getPrimaryConfigFilePath``).
``CONTINUE_GLOBAL_DIR`` relocates the directory, exactly as Continue does.

We add (or update in place) one model named ``rapid-mlx`` and keep every
other model and key. A new entry goes first in ``models`` so it is
Continue's default chat model; an existing entry keeps its position.

Migration from ``config.json``
------------------------------
Creating ``config.yaml`` next to a populated ``config.json`` would make
Continue stop reading the user's existing JSON models. So when only
``config.json`` exists, the new ``config.yaml`` carries the user's JSON
configuration over with the same mapping Continue's own "Convert to
config.yaml" command uses (``packages/config-yaml/src/converter.ts``), plus
the ``rapid-mlx`` model. ``config.json`` itself is never modified: deleting
or renaming ``config.yaml`` restores the previous behaviour, which is also
what Continue tells users after its own conversion.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _common

# Continue stores its config in ``~/.continue/`` on every platform.
_CONFIG_DIR = Path.home() / ".continue"
_CONFIG_FILENAME = "config.yaml"
_LEGACY_CONFIG_FILENAME = "config.json"

# Name of the model entry we own. A re-run updates it in place.
_MODEL_ENTRY_NAME = "rapid-mlx"

# What Continue writes into a brand-new ``config.yaml`` (core/config/default.ts)
# — required top-level keys of the v1 schema.
_DEFAULT_HEADER: dict[str, Any] = {
    "name": "Local Config",
    "version": "1.0.0",
    "schema": "v1",
}


def _config_dir() -> Path:
    """``$CONTINUE_GLOBAL_DIR`` when set (relative paths resolve against the
    working directory, as Continue does), otherwise ``~/.continue``."""
    configured = os.environ.get("CONTINUE_GLOBAL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().absolute()
    return _CONFIG_DIR


def detect() -> bool:
    """Return True when the Continue config dir exists.

    Deliberately permissive: Continue creates the directory on first
    activation (IDE) or first run (``cn``), before any config file exists,
    and ``rapid-mlx launch continue-dev`` should still succeed by creating
    ``config.yaml``.
    """
    return _config_dir().exists()


def current_config_path() -> Path | None:
    """Return the ``config.yaml`` path we write (always well-defined)."""
    return _config_dir() / _CONFIG_FILENAME


def legacy_config_path(config_path: Path | None = None) -> Path:
    """Return the legacy ``config.json`` beside ``config_path``."""
    path = config_path or current_config_path()
    assert path is not None
    return path.with_name(_LEGACY_CONFIG_FILENAME)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Read ``config.yaml``; ``{}`` when missing or blank.

    Invalid YAML (or a non-mapping document) raises ``ValueError`` so callers
    refuse to overwrite a file we cannot round-trip instead of silently
    replacing the user's edits.
    """
    import yaml

    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def load_legacy_config(path: Path) -> dict[str, Any]:
    """Read the legacy ``config.json``; ``{}`` when missing or blank."""
    import json

    try:
        value = _common.load_json_lenient(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def dump_yaml(data: dict[str, Any]) -> str:
    import yaml

    return str(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    # The upstream converter leaves ``undefined`` fields, which its YAML
    # serializer omits; ``None`` is our equivalent.
    return {k: v for k, v in value.items() if v is not None}


def _convert_model(entry: Any, roles: list[str]) -> Any:
    if not isinstance(entry, dict):
        return entry
    return _drop_none(
        {
            "name": entry.get("title"),
            "provider": entry.get("provider"),
            "model": entry.get("model"),
            "apiKey": entry.get("apiKey"),
            "apiBase": entry.get("apiBase"),
            "roles": roles,
            "requestOptions": entry.get("requestOptions"),
            "defaultCompletionOptions": entry.get("completionOptions"),
        }
    )


def _convert_context(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    name = entry.get("name")
    params = entry.get("params")
    if name in {"web", "debugger", "issue", "database", "google", "http"}:
        return _drop_none({"provider": name, "params": params})
    return _drop_none(
        {
            "uses": f"continuedev/{'open-files' if name == 'open' else name}-context",
            "with": params,
        }
    )


def convert_legacy_config(legacy: dict[str, Any]) -> dict[str, Any]:
    """Map a ``config.json`` onto the ``config.yaml`` schema.

    A port of Continue's own ``convertJsonToYamlConfig`` (Apache-2.0,
    continuedev/continue ``packages/config-yaml/src/converter.ts``) so the
    result matches what the "Convert to config.yaml" command in the IDE
    produces, with the v1 schema header Continue writes for new files.
    """
    # A legacy ``rapid-mlx`` entry is converted like any other; the patch
    # then updates it in place, keeping fields we do not own.
    models: list[Any] = [
        _convert_model(m, ["chat"]) for m in legacy.get("models") or []
    ]
    autocomplete = legacy.get("tabAutocompleteModel")
    if isinstance(autocomplete, list):
        models.extend(_convert_model(m, ["autocomplete"]) for m in autocomplete)
    elif autocomplete:
        models.append(_convert_model(autocomplete, ["autocomplete"]))
    embeddings = legacy.get("embeddingsProvider")
    if isinstance(embeddings, dict):
        models.append(
            _drop_none(
                {
                    "name": "Embeddings Model",
                    "provider": embeddings.get("provider"),
                    "model": embeddings.get("model") or "",
                    "apiKey": embeddings.get("apiKey"),
                    "apiBase": embeddings.get("apiBase"),
                    "roles": ["embed"],
                }
            )
        )
    reranker = legacy.get("reranker")
    if isinstance(reranker, dict):
        params = reranker.get("params") or {}
        models.append(
            _drop_none(
                {
                    "name": "Reranker",
                    "provider": reranker.get("name"),
                    "model": params.get("model") or "",
                    "apiKey": params.get("apiKey"),
                    "apiBase": params.get("apiBase"),
                    "roles": ["rerank"],
                }
            )
        )

    converted: dict[str, Any] = {**_DEFAULT_HEADER, "models": models}
    converted["context"] = [
        _convert_context(c) for c in legacy.get("contextProviders") or []
    ]
    if legacy.get("systemMessage"):
        converted["rules"] = [legacy["systemMessage"]]
    commands = legacy.get("customCommands")
    if commands:
        converted["prompts"] = [
            _drop_none(
                {
                    "name": c.get("name"),
                    "description": c.get("description"),
                    "prompt": c.get("prompt"),
                }
            )
            if isinstance(c, dict)
            else c
            for c in commands
        ]
    experimental = legacy.get("experimental")
    servers = (
        experimental.get("modelContextProtocolServers")
        if isinstance(experimental, dict)
        else None
    )
    if servers:
        converted["mcpServers"] = [
            _drop_none(
                {
                    "command": (s.get("transport") or {}).get("command"),
                    "args": (s.get("transport") or {}).get("args"),
                    "env": (s.get("transport") or {}).get("env"),
                    "name": (s.get("transport") or {}).get("server_name")
                    or "MCP Server",
                }
            )
            if isinstance(s, dict)
            else s
            for s in servers
        ]
    docs = legacy.get("docs")
    if docs:
        converted["docs"] = [
            _drop_none(
                {
                    "name": d.get("title"),
                    "startUrl": d.get("startUrl"),
                    "rootUrl": d.get("rootUrl"),
                    "faviconUrl": d.get("faviconUrl"),
                }
            )
            if isinstance(d, dict)
            else d
            for d in docs
        ]
    if legacy.get("requestOptions") is not None:
        converted["requestOptions"] = legacy["requestOptions"]
    return converted


def patched_config(
    existing: dict[str, Any],
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
) -> dict[str, Any]:
    """Return the ``config.yaml`` mapping we would write, without touching disk.

    An existing ``rapid-mlx`` entry is updated in place (fields we do not
    own, e.g. a user's ``defaultCompletionOptions``, survive); otherwise the
    entry is inserted first so it is the default chat model.
    """
    result = dict(existing)
    for key, value in _DEFAULT_HEADER.items():
        result.setdefault(key, value)

    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    ours = {
        "name": _MODEL_ENTRY_NAME,
        "provider": "openai",
        "model": model,
        "apiBase": base_url,
        "apiKey": api_key,
    }

    models = list(result.get("models") or [])
    for i, entry in enumerate(models):
        # Entries can also be hub references (``uses: ...``); only a mapping
        # carrying our name is ours to update.
        if isinstance(entry, dict) and entry.get("name") == _MODEL_ENTRY_NAME:
            updated = {**entry, **ours}
            roles = entry.get("roles")
            # No ``roles`` means Continue's defaults (chat included); an
            # explicit list must offer chat, or setup would leave the model
            # out of the chat picker.
            if isinstance(roles, list) and "chat" not in roles:
                updated["roles"] = ["chat", *roles]
            models[i] = updated
            break
    else:
        models.insert(0, ours)
    result["models"] = models
    return result


@dataclass(frozen=True)
class ContinuePlan:
    """What a Continue setup would change, computed without side effects."""

    path: Path
    before: dict[str, Any]
    after: dict[str, Any]
    # Set when ``config.yaml`` is created from an existing ``config.json``.
    migrated_from: Path | None = None
    migrated_from_before: dict[str, Any] | None = None
    notes: tuple[str, ...] = field(default=())

    @property
    def changed(self) -> bool:
        return self.before != self.after


def build_plan(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> ContinuePlan:
    logical = config_path or current_config_path()
    assert logical is not None
    # Backup and the atomic replace target the real file, so a config.yaml
    # symlinked into a dotfiles repo keeps its link.
    path = logical.resolve()
    before = load_yaml_config(path)
    if path.exists():
        # Any existing config.yaml (even blank, which Continue refills with
        # its default) is what Continue reads; config.json is then ignored
        # and must not be migrated.
        return ContinuePlan(
            path, before, patched_config(before, server_url, model, api_key)
        )

    legacy_path = legacy_config_path(logical)
    legacy = load_legacy_config(legacy_path)
    if not legacy:
        return ContinuePlan(
            path, before, patched_config({}, server_url, model, api_key)
        )
    base = convert_legacy_config(legacy)
    notes = (
        f"Continue now reads {path.name} instead of {legacy_path}; your "
        f"{legacy_path.name} settings are copied into it with Continue's own "
        "conversion. "
        f"{legacy_path.name} is left unchanged — delete or rename {path.name} "
        "to go back to it.",
    )
    return ContinuePlan(
        path,
        before,
        patched_config(base, server_url, model, api_key),
        migrated_from=legacy_path,
        migrated_from_before=legacy,
        notes=notes,
    )


def write_or_patch_config(
    server_url: str,
    model: str,
    api_key: str = "sk-noop",
    config_path: Path | None = None,
) -> Path:
    """Add or update the ``rapid-mlx`` model in Continue's ``config.yaml``.

    Idempotent: an already-matching config is left byte-for-byte untouched
    (no backup, no rewrite). Otherwise an existing ``config.yaml`` is backed
    up to ``config.yaml.bak.<ts>`` and the new file is written atomically
    with owner-only permissions (it can carry ``RAPID_MLX_API_KEY``).
    ``config.json`` is never written.
    """
    plan = build_plan(server_url, model, api_key, config_path)
    for note in plan.notes:
        print(f"  note: {note}", file=sys.stderr)
    if not plan.changed and plan.path.exists():
        return plan.path
    _common.backup_existing(plan.path)
    _common.atomic_write_text(plan.path, dump_yaml(plan.after))
    return plan.path


def plan_diff(plan: ContinuePlan) -> str:
    """Unified diff of a plan with credentials hidden (preview only)."""
    before = dump_yaml(_common.redact_secrets(plan.before)) if plan.before else ""
    after = dump_yaml(_common.redact_secrets(plan.after))
    return _common.unified_diff(before, after, plan.path)


def preview(
    server_url: str, model: str, api_key: str = "sk-noop"
) -> tuple[Path, str, tuple[str, ...]]:
    """``(path, redacted diff, notes)`` for ``launch --dry-run``; no writes."""
    plan = build_plan(server_url, model, api_key)
    return plan.path, plan_diff(plan) if plan.changed else "", plan.notes
