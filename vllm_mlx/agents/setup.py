# SPDX-License-Identifier: Apache-2.0
"""Safe plan/apply flow for first-class agent integrations."""

from __future__ import annotations

import difflib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_mlx.launch import _common as launch_common
from vllm_mlx.launch import claude_code, continue_dev


@dataclass(frozen=True)
class SetupPlan:
    agent: str
    display_name: str
    path: Path
    before: dict[str, Any]
    after: dict[str, Any]
    base_url: str
    model: str
    format: str = "json"
    credentials_path: Path | None = None
    credentials_before: dict[str, Any] | None = None
    credentials_after: dict[str, Any] | None = None

    @property
    def changed(self) -> bool:
        return self.before != self.after or (
            self.credentials_path is not None
            and self.credentials_before != self.credentials_after
        )

    def diff(self) -> str:
        before = _serialize(self.before, self.format)
        after = _serialize(self.after, self.format)
        primary = "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=str(self.path),
                tofile=str(self.path) + " (proposed)",
                lineterm="",
            )
        )
        if self.credentials_path is None or self.credentials_after is None:
            return primary
        # DSH's credential store may contain real keys for unrelated remote
        # providers.  A setup preview must never echo those secrets.  Show
        # only the presence/absence of the one Rapid-managed key.
        managed_key = "RAPID_MLX_API_KEY"
        credentials_before = (
            {managed_key: "<configured; preserved>"}
            if managed_key in (self.credentials_before or {})
            else {}
        )
        credentials_after = (
            {managed_key: "<configured; preserved>"}
            if managed_key in (self.credentials_before or {})
            else {managed_key: "not-needed"}
        )
        credentials = "\n".join(
            difflib.unified_diff(
                _serialize(credentials_before, "yaml").splitlines(),
                _serialize(credentials_after, "yaml").splitlines(),
                fromfile=str(self.credentials_path),
                tofile=str(self.credentials_path) + " (proposed)",
                lineterm="",
            )
        )
        return "\n".join(part for part in (primary, credentials) if part)


def _serialize(data: dict[str, Any], format: str) -> str:
    if format == "yaml":
        import yaml

        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dsh_settings_path() -> Path:
    configured = os.environ.get("DSH_HOME", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".dsh"
    return root / "settings.yaml"


def _atomic_write_secure_text(path: Path, text: str) -> None:
    """Atomically publish owner-only text beside a potentially secret config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".new", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_setup_plan(
    agent: str,
    base_url: str,
    model: str,
    context_length: int | None = None,
    supports_reasoning: bool | None = None,
) -> SetupPlan:
    """Build a side-effect-free setup plan for a supported client."""
    if agent in {"claude", "claude-code"}:
        path = claude_code.current_config_path()
        assert path is not None
        before = launch_common.load_json_lenient(path)
        after = claude_code.patched_config(before, base_url, model)
        return SetupPlan(
            "claude-code", "Claude Code", path, before, after, base_url, model
        )
    if agent in {"continue", "continue-dev"}:
        path = continue_dev.current_config_path()
        assert path is not None
        before = launch_common.load_json_lenient(path)
        after = continue_dev.patched_config(before, base_url, model)
        return SetupPlan(
            "continue", "Continue.dev", path, before, after, base_url, model
        )
    if agent in {"deepseek-harness", "dsh"}:
        path = _dsh_settings_path()
        before = _load_yaml_mapping(path)
        credentials_path = path.parent / ".credentials.yaml"
        credentials_before = _load_yaml_mapping(credentials_path)
        context = context_length if context_length and context_length > 0 else 32768
        # Report the model's ACTUAL reasoning capability rather than
        # asserting the graded ladder for everything we serve.
        #
        # Harness renders a reasoning-effort control from this block, so a
        # model with no reasoning parser used to get an off/low/medium/high
        # selector that changed nothing the user could observe. pi-ai
        # accepts ``reasoningEfforts: false`` for exactly this case
        # ("set false for a non-reasoning model").
        #
        # Only a definite ``False`` downgrades. ``None`` means we could not
        # find out (server unreachable, model not listed, or a rapid-mlx too
        # old to report the field), and silently deleting a working control
        # on a guess is worse than the cosmetic over-claim this fixes — see
        # ``fetch_reasoning_support`` for why the three states are kept apart.
        reasoning_capable = supports_reasoning is not False
        model_entry: dict[str, Any] = {
            "id": model,
            "name": f"{model} (Rapid-MLX)",
            "contextWindow": context,
            "maxTokens": 8192,
            "reasoningEfforts": (
                {
                    "off": "none",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                }
                if reasoning_capable
                else False
            ),
        }
        patch = {
            "llm-pi-ai": {
                "providers": {
                    "rapid-mlx": {
                        "displayName": "Rapid-MLX Local",
                        "apiKeyEnv": "RAPID_MLX_API_KEY",
                        "api": "openai-completions",
                        "baseURL": base_url.rstrip("/"),
                        "defaultContextWindow": context,
                        "defaultMaxTokens": 8192,
                        "compat": {"supportsReasoningEffort": reasoning_capable},
                        "models": [model_entry],
                    }
                }
            },
            "agent-default-model": {"provider": "rapid-mlx", "model": model},
        }
        credentials_after = dict(credentials_before)
        credentials_after.setdefault("RAPID_MLX_API_KEY", "not-needed")
        return SetupPlan(
            "deepseek-harness",
            "DeepSeek Harness",
            path,
            before,
            _merge_dict(before, patch),
            base_url,
            model,
            "yaml",
            credentials_path,
            credentials_before,
            credentials_after,
        )
    raise ValueError(f"{agent} does not have a first-class safe setup flow")


def apply_setup_plan(plan: SetupPlan) -> Path:
    """Back up the existing config and atomically apply an unchanged plan."""
    # Re-read to prevent overwriting an edit made between preview and consent.
    current = (
        _load_yaml_mapping(plan.path)
        if plan.format == "yaml"
        else launch_common.load_json_lenient(plan.path)
    )
    if current != plan.before:
        raise RuntimeError(f"{plan.path} changed after preview; re-run --setup")
    if plan.credentials_path is not None:
        credentials_current = _load_yaml_mapping(plan.credentials_path)
        if credentials_current != (plan.credentials_before or {}):
            raise RuntimeError(
                f"{plan.credentials_path} changed after preview; re-run --setup"
            )
    launch_common.backup_existing(plan.path)
    if plan.credentials_path is not None and plan.credentials_after is not None:
        launch_common.backup_existing(plan.credentials_path)
        # Publish the harmless loopback sentinel first.  If the later settings
        # write fails, DSH retains its prior provider selection; the reverse
        # order would leave a newly-selected Rapid route unable to authenticate.
        _atomic_write_secure_text(
            plan.credentials_path,
            _serialize(plan.credentials_after, "yaml") + "\n",
        )
    if plan.format == "yaml":
        _atomic_write_secure_text(plan.path, _serialize(plan.after, "yaml") + "\n")
    else:
        launch_common.atomic_write_json(plan.path, plan.after)
    return plan.path


def verify_server(base_url: str, expected_model: str, timeout: float = 2.0) -> str:
    """Verify health and model discovery without performing inference."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"health returned HTTP {response.status}")
        with urllib.request.urlopen(f"{root}/v1/models", timeout=timeout) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"server is not ready at {root}: {exc}") from exc
    models = payload.get("data", []) if isinstance(payload, dict) else []
    ids = [item.get("id") for item in models if isinstance(item, dict)]
    if not ids:
        raise RuntimeError(f"server at {root} reported no models")
    if expected_model != "default" and expected_model not in ids and len(ids) != 1:
        raise RuntimeError(
            f"server does not advertise model {expected_model!r} (found: {', '.join(ids)})"
        )
    return ids[0]


def confirm_plan(plan: SetupPlan) -> bool:
    """Ask before writing; non-interactive callers must use --yes."""
    if not sys.stdin.isatty():
        return False
    try:
        return input("Apply this configuration? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False
