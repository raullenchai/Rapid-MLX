# SPDX-License-Identifier: Apache-2.0
"""Run the hosted mac-only selection early during local pre-push validation."""

import argparse
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

# These tests still run in the local pr_validate mac-only stage.
# The prefix cache tests need a locally cached model; telemetry needs
# multi-process timing the hosted runner cannot meet; the GUI guard
# needs OS permissions the hosted sandbox denies.
HOSTED_DESELECT = (
    "tests/test_mid_conversation_system_prefix_cache.py::TestRenderedTokenPrefixIsStable::test_leading_tokens_survive_an_injected_nudge",
    "tests/test_mid_conversation_system_prefix_cache.py::TestRenderedTokenPrefixIsStable::test_hoisting_would_have_failed_this_test",
    "tests/test_telemetry_track.py::test_simultaneous_missing_extra_serve_failures_emit_once",
    "tests/test_telemetry_track.py::test_server_start_ready_wire_order_and_opt_out",
    "tests/test_agent_test_gui_guard.py::test_launch_attempt_remains_failure_if_agent_times_out",
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


def select_mac_only_files(repo_root: Path) -> list[str]:
    roster = set(select_roster_files(repo_root))
    marker = re.compile(
        r"requires_mlx|real_hf_cache|requires_network|importorskip|darwin|sys\.platform|platform\.(machine|system)"
    )
    # Linux deselects requires_mlx, real_hf_cache and requires_network and
    # skips MLX/macOS-only imports; the original Apple job runs its roster.
    # The hosted mac-only job and this local step share this selection.
    selected = []
    for path in (repo_root / "tests").rglob("test_*.py"):
        relative = path.relative_to(repo_root).as_posix()
        if (
            relative in roster
            or relative == "tests/test_event_loop.py"
            or relative.startswith(("tests/integrations/", "tests/headless_mlx/"))
        ):
            continue
        if marker.search(path.read_text()):
            selected.append(relative)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="List the mac-only test selection")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--deselect-args", action="store_true")
    parser.add_argument("--shard", metavar="I/N")
    args = parser.parse_args()
    if args.deselect_args:
        print("\n".join(f"--deselect={node}" for node in HOSTED_DESELECT))
        return
    files = select_mac_only_files(Path.cwd())
    if args.shard is not None:
        if not re.fullmatch(r"[0-9]+/[0-9]+", args.shard):
            parser.error("--shard must be I/N with 1 <= I <= N")
        index, total = map(int, args.shard.split("/"))
        if not 1 <= index <= total:
            parser.error("--shard must be I/N with 1 <= I <= N")
        files = sorted(files)[index - 1 :: total]
    print("\n".join(files), end="\n" if files else "")


class MacOnlyTestsStep(Step):
    name = "mac_only_tests"
    description = "early pre-push signal for the hosted Apple mac-only selection"

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


if __name__ == "__main__":
    main()
