"""Record pytest collection/execution and reject incomplete mapped CPU proof.

Load with ``-p scripts.ci_mapped_execution --mapped-execution-manifest PATH``.
The same trusted plugin collects the base suites and records the current run.
The CLI validates both manifests; it never qualifies full candidate evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "rapid-mlx/mapped-execution/v1"


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--mapped-execution-manifest", type=Path)


def pytest_configure(config: Any) -> None:
    if config.getoption("mapped_execution_manifest"):
        config.pluginmanager.register(_Recorder(config), "mapped-execution-recorder")


class _Recorder:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.nodes: list[str] = []
        self.reports: list[dict[str, Any]] = []
        self.deselected = 0
        self.collection_errors = 0

    def pytest_collection_finish(self, session: Any) -> None:
        self.nodes = [item.nodeid for item in session.items]

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected += len(items)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed or report.skipped:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report: Any) -> None:
        self.reports.append(
            {
                "node": report.nodeid,
                "stage": report.when,
                "outcome": report.outcome,
                "xfail": hasattr(report, "wasxfail"),
            }
        )

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self.config.getoption("mapped_execution_manifest").write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "collect_only": bool(self.config.option.collectonly),
                    "exitstatus": int(exitstatus),
                    "nodes": self.nodes,
                    "deselected": self.deselected,
                    "collection_errors": self.collection_errors,
                    "reports": self.reports,
                },
                indent=2,
            )
            + "\n"
        )


def _nodes(manifest: dict[str, Any], tests: list[str], collect_only: bool) -> set[str]:
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("collect_only") is not collect_only
        or manifest.get("exitstatus") != 0
        or manifest.get("deselected") != 0
        or manifest.get("collection_errors") != 0
    ):
        raise ValueError("mapped collection failed, skipped, or deselected tests")
    nodes = manifest.get("nodes")
    if (
        not isinstance(nodes, list)
        or not nodes
        or not all(isinstance(node, str) and "::" in node for node in nodes)
    ):
        raise ValueError("mapped collection must contain test nodes")
    if len(set(nodes)) != len(nodes):
        raise ValueError("duplicate mapped test nodes")
    if {node.split("::", 1)[0] for node in nodes} != set(tests):
        raise ValueError("every selected file must collect tests, with no extra files")
    return set(nodes)


def validate_manifests(
    base: dict[str, Any], executed: dict[str, Any], tests: list[str]
) -> None:
    if not tests or len(set(tests)) != len(tests):
        raise ValueError("selected files must be nonempty and unique")
    baseline_nodes = _nodes(base, tests, True)
    current_nodes = _nodes(executed, tests, False)
    if not baseline_nodes <= current_nodes:
        raise ValueError(
            "mapped tests were removed or renamed; require full validation"
        )
    reports = executed.get("reports")
    if not isinstance(reports, list) or not all(isinstance(r, dict) for r in reports):
        raise ValueError("missing mapped execution reports")
    actual: Counter[tuple[str, str]] = Counter()
    for report in reports:
        node, stage = report.get("node"), report.get("stage")
        if (
            node not in current_nodes
            or stage not in {"setup", "call", "teardown"}
            or report.get("outcome") != "passed"
            or report.get("xfail") is not False
        ):
            raise ValueError(
                "mapped test skipped, failed, xfailed, or reported unknown execution"
            )
        actual[node, stage] += 1
    expected = Counter(
        (node, stage)
        for node in current_nodes
        for stage in ("setup", "call", "teardown")
    )
    if actual != expected:
        raise ValueError(
            "each collected node must pass setup, call, and teardown exactly once"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", required=True)
    args = parser.parse_args()
    validate_manifests(
        json.loads(args.baseline.read_text()),
        json.loads(args.executed.read_text()),
        args.tests,
    )
    print(
        "All mapped tests executed; base collection retained; no skipped or deselected proof"
    )


if __name__ == "__main__":
    main()
