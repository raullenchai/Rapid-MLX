# SPDX-License-Identifier: Apache-2.0
"""Run test files that hosted Linux and the Apple roster do not fully execute."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from ..base import Step, StepResult
from ..context import Context
from .full_unit import (
    _FULL_UNIT_SERIAL_TIMEOUT_S,
    _FULL_UNIT_TIMEOUT_S,
    _extract_failed_lines,
    _last_summary_line,
    _run_group_bounded,
    _timeout_dump,
    pytest_parallel_args,
)


def select_roster_files(repo_root: Path) -> list[str]:
    workflow = repo_root / ".github/workflows/ci.yml"
    roster = set()
    if workflow.exists():
        block = re.search(
            r"^  test-apple-silicon:\s*$([\s\S]*?)(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
            workflow.read_text(),
            re.M,
        )
        if block:
            roster = set(
                re.findall(r"^\s+(tests/[\w./-]+\.py)\s*\\?\s*$", block.group(1), re.M)
            )
    return sorted(path for path in roster if (repo_root / path).is_file())


_MLX_MODULES = {"mlx", "mlx_lm", "mlx_vlm", "mlx_audio", "mlx_embeddings"}


def _has_mlx_availability_guard(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name == "find_spec" and node.args:
                module = node.args[0]
                if isinstance(module, ast.Constant) and isinstance(module.value, str):
                    if module.value.split(".")[0] in _MLX_MODULES:
                        return True
        if isinstance(node, ast.Try):
            exceptions = {
                part.id
                for handler in node.handlers
                if handler.type is not None
                for part in ast.walk(handler.type)
                if isinstance(part, ast.Name)
            }
            if not exceptions & {"ImportError", "ModuleNotFoundError"}:
                continue
            for statement in node.body:
                for imported in ast.walk(statement):
                    if isinstance(imported, ast.Import):
                        modules = [alias.name for alias in imported.names]
                    elif isinstance(imported, ast.ImportFrom):
                        modules = [imported.module or ""]
                    else:
                        continue
                    if any(module.split(".")[0] in _MLX_MODULES for module in modules):
                        return True
    return False


def select_mac_only_files(repo_root: Path) -> list[str]:
    roster = set(select_roster_files(repo_root))
    marker = re.compile(
        r"requires_mlx|real_hf_cache|requires_network|importorskip|darwin|sys\.platform|platform\.(machine|system)"
    )
    # Linux deselects requires_mlx, real_hf_cache and requires_network and
    # skips MLX/macOS-only imports; Apple runs only its explicit roster.
    # At introduction, 130 files / ~2,700 tests ran locally but in no CI job.
    selected = []
    for path in (repo_root / "tests").rglob("test_*.py"):
        relative = path.relative_to(repo_root).as_posix()
        if (
            relative in roster
            or relative == "tests/test_event_loop.py"
            or relative.startswith(("tests/integrations/", "tests/headless_mlx/"))
        ):
            continue
        source = path.read_text()
        if marker.search(source) or _has_mlx_availability_guard(source):
            selected.append(relative)
    return sorted(selected)


class MacOnlyTestsStep(Step):
    name = "mac_only_tests"
    description = "local-only tests no hosted CI job executes (MLX/macOS-bound files outside the Apple CI roster)"

    def should_run(self, ctx: Context) -> bool:
        return ctx.blast_radius != "low"

    def run(self, ctx: Context) -> StepResult:
        files = select_mac_only_files(ctx.repo_root)
        roster = select_roster_files(ctx.repo_root)
        parallel_args, note = pytest_parallel_args()
        ctx.run_log(note)
        raw = os.environ.get("PR_VALIDATE_MAC_ONLY_TIMEOUT_S")
        timeout = (
            int(raw)
            if raw is not None
            else (
                _FULL_UNIT_TIMEOUT_S if parallel_args else _FULL_UNIT_SERIAL_TIMEOUT_S
            )
        )
        if timeout <= 0:
            raise ValueError(
                "PR_VALIDATE_MAC_ONLY_TIMEOUT_S must be a positive integer"
            )

        def run_group(paths: list[str], log_name: str, marker: bool = False):
            if not paths:
                return "skip", "no test files", "", []
            log_path = ctx.artifact_path(log_name)
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                *paths,
                "-q",
                "--no-header",
                "-rfE",
                "-o",
                "faulthandler_timeout=600",
                *parallel_args,
            ]
            if marker:
                cmd.extend(["-m", "real_hf_cache or requires_network"])
            try:
                proc = _run_group_bounded(cmd, cwd=str(ctx.repo_root), timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                log_path.write_text(
                    note + "\n" + _timeout_dump("mac_only_tests pytest", cmd, exc)
                )
                return "fail", f"exceeded {timeout}s", "", [str(log_path)]
            log_path.write_text(note + "\n" + (proc.stdout or "") + (proc.stderr or ""))
            summary = (
                _last_summary_line(proc.stdout) or f"pytest exited {proc.returncode}"
            )
            status = (
                "pass"
                if proc.returncode == 0
                else "skip"
                if proc.returncode == 5
                else "fail"
            )
            details = ""
            if status == "fail":
                failed = _extract_failed_lines(proc.stdout)
                details = (
                    "\n**Failed tests:**\n```\n" + "\n".join(failed[:30]) + "\n```"
                )
                if len(failed) > 30:
                    details += f"\n…and {len(failed) - 30} more — see {log_path}"
            return status, summary, details, [str(log_path)]

        a = run_group(files, "mac-only-tests.log")
        b = run_group(roster, "mac-only-roster-gated.log", marker=True)
        statuses = {a[0], b[0]}
        status = (
            "fail" if "fail" in statuses else "pass" if "pass" in statuses else "skip"
        )
        return StepResult(
            self.name,
            status,
            f"{a[1]} ({len(files)} files); roster gated: {b[1]}; {note}",
            details=note + a[2] + b[2],
            artifacts=a[3] + b[3],
        )
