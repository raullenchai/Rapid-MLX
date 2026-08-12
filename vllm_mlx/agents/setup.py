# SPDX-License-Identifier: Apache-2.0
"""Safe plan/apply flow for first-class agent integrations."""

from __future__ import annotations

import difflib
import json
import sys
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

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def diff(self) -> str:
        before = json.dumps(self.before, indent=2, ensure_ascii=False, sort_keys=True)
        after = json.dumps(self.after, indent=2, ensure_ascii=False, sort_keys=True)
        return "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=str(self.path),
                tofile=str(self.path) + " (proposed)",
                lineterm="",
            )
        )


def build_setup_plan(agent: str, base_url: str, model: str) -> SetupPlan:
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
    raise ValueError(f"{agent} does not have a first-class safe setup flow")


def apply_setup_plan(plan: SetupPlan) -> Path:
    """Back up the existing config and atomically apply an unchanged plan."""
    # Re-read to prevent overwriting an edit made between preview and consent.
    current = launch_common.load_json_lenient(plan.path)
    if current != plan.before:
        raise RuntimeError(f"{plan.path} changed after preview; re-run --setup")
    launch_common.backup_existing(plan.path)
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
