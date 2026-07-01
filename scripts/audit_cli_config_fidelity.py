#!/usr/bin/env python3
"""
CLI ↔ Config fidelity audit.

Catches "silent flag drop" bugs where a CLI argparse flag is defined and
plumbed *somewhere*, but never actually reaches the engine config it was
supposed to set.

Motivation: #400 — `--prefill-step-size 32768` was defined in argparse,
passed to `load_model(prefill_step_size=...)` (where it was discarded),
but never added to the `SchedulerConfig(...)` construction kwargs.
`SchedulerConfig.prefill_step_size` silently kept its 2048 default,
which the MLLM batch path then used as a hard cap → user-visible bug.

This audit is structural and runs in <1s. Add to release gate.

Exit code: 0 if clean, 1 if drift detected.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEDULER_PATH = REPO_ROOT / "vllm_mlx" / "scheduler.py"

# Files that parse argparse args AND construct (or should construct) an
# engine config. Each one is a user-facing entrypoint with the same
# silent-flag-drop risk as #400. Order: primary CLI first, then secondary
# entries (python -m vllm_mlx.server, mise run).
CLI_ENTRY_PATHS = [
    REPO_ROOT / "vllm_mlx" / "cli.py",
    REPO_ROOT / "vllm_mlx" / "server.py",
]

# Helpers that RETURN a dict of ``SchedulerConfig`` kwargs and are
# star-unpacked at construction sites via ``**helper(args)``. The audit
# expands the starred unpacking into the field set the helper injects
# so the drift check doesn't false-positive on the shared entrypoint
# mirror pattern (#969 — ``cli.py`` and ``server.py`` both call
# ``turboquant_scheduler_kwargs`` to guarantee lock-step behaviour).
#
# Key: helper function name (matches ``Call.func.id``); may include
# local aliases used by ``from ... import ... as _local``.
# Value: the set of ``SchedulerConfig`` field names the helper returns.
# Keep the field set in sync with the helper body — a mismatch here
# either (a) hides a real drift (values in helper, missing here) or
# (b) creates a false-clean audit (values here, not in helper).
KWARG_INJECTOR_HELPERS: dict[str, set[str]] = {
    "turboquant_scheduler_kwargs": {
        "kv_cache_turboquant",
        "kv_cache_turboquant_bits",
        "kv_cache_turboquant_group_size",
        "kv_cache_turboquant_mode",
    },
    # Import-time aliases used inside the two entrypoints.
    "_turboquant_scheduler_kwargs": {
        "kv_cache_turboquant",
        "kv_cache_turboquant_bits",
        "kv_cache_turboquant_group_size",
        "kv_cache_turboquant_mode",
    },
    "_server_turboquant_scheduler_kwargs": {
        "kv_cache_turboquant",
        "kv_cache_turboquant_bits",
        "kv_cache_turboquant_group_size",
        "kv_cache_turboquant_mode",
    },
}


def _dataclass_field_names(source_path: Path, class_name: str) -> set[str]:
    """Extract field names of a @dataclass via AST — no runtime import.

    Reading the source directly avoids pulling mlx.core (and the rest of
    the engine) into the audit, so this script runs on plain Linux CI
    without the apple-silicon stack.
    """
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        names: set[str] = set()
        for stmt in node.body:
            # `name: type = default` or `name: type`
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names.add(stmt.target.id)
        return names
    raise RuntimeError(f"class {class_name} not found in {source_path}")


def _function_args_refs(func_node: ast.FunctionDef) -> set[str]:
    """Every `args.<X>` attribute access inside the function body."""
    refs: set[str] = set()
    for sub in ast.walk(func_node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "args"
        ):
            refs.add(sub.attr)
    return refs


def audit(cli_path: Path, config_source: Path, config_cls_name: str) -> list[str]:
    """Return a list of drift lines. Empty list = clean.

    Drift definition: a function reads `args.X`, AND calls
    `<config_cls>(...)` without passing `X=`, AND `X` is a field on
    `<config_cls>`. The 3-way intersection means: the user *can* set this
    flag in the subparser this command owns, the config *has* a slot for
    it, but the construction site silently drops the user's value.

    This is narrower than a global flag-vs-field check: it scopes to the
    command function, so benchmark-only flags don't false-positive against
    serve-only construction sites.
    """
    source = cli_path.read_text()
    tree = ast.parse(source)

    field_names = _dataclass_field_names(config_source, config_cls_name)
    issues: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        arg_refs = _function_args_refs(node)
        if not arg_refs:
            continue

        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not (
                isinstance(call.func, ast.Name) and call.func.id == config_cls_name
            ):
                continue

            kwargs: set[str] = set()
            for kw in call.keywords:
                if kw.arg is not None:
                    kwargs.add(kw.arg)
                elif (
                    isinstance(kw.value, ast.Call)
                    and isinstance(kw.value.func, ast.Name)
                    and kw.value.func.id in KWARG_INJECTOR_HELPERS
                ):
                    # ``**helper(args)`` star-unpacking — pick up the
                    # injected fields so the shared entrypoint helper
                    # pattern (#969) doesn't false-positive as drift.
                    kwargs |= KWARG_INJECTOR_HELPERS[kw.value.func.id]
            # Fields that this function CAN supply (args.X is referenced) but
            # this construction site does NOT pass.
            dropped = (field_names & arg_refs) - kwargs
            for field_name in sorted(dropped):
                flag = "--" + field_name.replace("_", "-")
                issues.append(
                    f"{cli_path.name}:{call.lineno} (in {node.name})  "
                    f"{config_cls_name}({flag} dropped) — function reads "
                    f"args.{field_name} and {config_cls_name} has the "
                    f"field, but the kwarg is not passed here."
                )

    return issues


def main() -> int:
    # (source_file, dataclass_name). Add more configs as their CLI surface grows.
    targets = [
        (SCHEDULER_PATH, "SchedulerConfig"),
    ]

    all_issues: list[str] = []
    for entry_path in CLI_ENTRY_PATHS:
        for source, cls_name in targets:
            all_issues.extend(audit(entry_path, source, cls_name))

    if not all_issues:
        print("CLI ↔ Config fidelity: OK")
        return 0

    print("CLI ↔ Config fidelity: DRIFT DETECTED")
    print()
    for line in all_issues:
        print(f"  {line}")
    print()
    print(
        "Each line above is a user-visible silent-failure bug: the user can "
        "type the flag, argparse will accept it, but the engine will never "
        "see the value. Add the kwarg at the construction site."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
