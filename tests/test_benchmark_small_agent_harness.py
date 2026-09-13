"""Safety and scoring contracts for the compact-agent benchmark."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_small_agent_harness.py"
SPEC = importlib.util.spec_from_file_location("benchmark_small_agent_harness", SCRIPT)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def test_safe_path_rejects_workspace_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        BENCHMARK.safe_path(tmp_path, "../outside.txt")


def test_latest_tool_status_clears_a_repaired_failure() -> None:
    succeeded, failed = BENCHMARK.latest_tool_status(
        [
            {"name": "read_file", "ok": False},
            {"name": "write_file", "ok": True},
            {"name": "read_file", "ok": True},
            {"name": "run_tests", "ok": False},
        ]
    )

    assert succeeded == "read_file, write_file"
    assert failed == "run_tests"


def test_main_fails_when_a_request_has_an_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = BENCHMARK.Task(
        id="infra",
        category="test",
        prompt="test",
        tools=["search_web"],
    )
    monkeypatch.setattr(BENCHMARK, "TASKS", [task])

    def unavailable(*_args, **_kwargs):
        raise ConnectionError("unavailable")

    monkeypatch.setattr(BENCHMARK, "completion", unavailable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model",
            "test-model",
            "--mode",
            "enhanced",
            "--seeds",
            "11",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    assert BENCHMARK.main() == 1


def test_main_rejects_nonpositive_max_rounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model",
            "test-model",
            "--mode",
            "raw",
            "--max-rounds",
            "0",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        BENCHMARK.main()
    assert raised.value.code == 2


def test_safe_ast_interpreter_handles_branching_clamp() -> None:
    function = ast.parse(
        """\
def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value
"""
    ).body[0]
    observed = [
        BENCHMARK.evaluate_function(function, {"value": value, "low": 0, "high": 10})
        for value in (-3, 4, 15)
    ]
    assert observed == [0, 4, 10]


def test_safe_ast_interpreter_handles_conditional_expression() -> None:
    function = ast.parse(
        "def clamp(value, low, high):\n"
        "    return low if value < low else high if value > high else value\n"
    ).body[0]
    observed = [
        BENCHMARK.evaluate_function(function, {"value": value, "low": -5, "high": 5})
        for value in (-10, -3, 7)
    ]
    assert observed == [-5, -3, 5]


def test_safe_ast_interpreter_rejects_code_execution() -> None:
    function = ast.parse(
        """\
def unsafe(value, low, high):
    import os
    return os.system("echo unsafe")
"""
    ).body[0]
    with pytest.raises(ValueError, match="unsupported statement"):
        BENCHMARK.evaluate_function(function, {"value": 4, "low": 0, "high": 10})


def test_safe_ast_interpreter_short_circuits_boolean_operations() -> None:
    assert BENCHMARK.arithmetic("value == 0 or 10 / value > 1", {"value": 0}) is True
    assert BENCHMARK.arithmetic("value > 0 and 10 / value > 1", {"value": 0}) is False


def test_discount_verifier_checks_behavior_without_exec(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text(
        "def discounted(total, percent):\n    return total - total * percent / 100\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    passed, message = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is True
    assert message == "All tests passed."


def test_discount_verifier_accepts_safe_local_assignment(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text(
        "def discounted(total, percent):\n"
        "    discount = total * percent / 100\n"
        "    return total - discount\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    passed, message = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is True
    assert message == "All tests passed."


def test_discount_verifier_accepts_safe_validation_guard(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text(
        "def discounted(total, percent):\n"
        "    if percent < 0 or percent > 100:\n"
        '        raise ValueError("percent out of range")\n'
        "    return total - total * percent / 100\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    passed, message = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is True
    assert message == "All tests passed."


def test_discount_verifier_rejects_single_case_hardcoding(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text("def discounted(total, percent):\n    return 80\n")
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    passed, _ = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is False


def test_discount_verifier_rejects_wrong_signature(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text("def discounted():\n    return total - total * percent / 100\n")
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    passed, message = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is False
    assert "must accept" in message


def test_discount_verifier_rejects_duplicate_target_definitions(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text(
        "def discounted(total, percent):\n"
        "    return total - total * percent / 100\n\n"
        "def discounted(total, percent):\n"
        "    return 80\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    passed, message = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is False
    assert "exactly one" in message


def test_failed_tool_call_does_not_satisfy_required_tool(tmp_path: Path) -> None:
    task = BENCHMARK.Task(
        id="failed_tool",
        category="test",
        prompt="write a result",
        tools=["write_file"],
        required_tools=["write_file"],
    )
    scored = BENCHMARK.score_task(
        task,
        tmp_path,
        "done",
        [{"name": "write_file", "ok": False}],
        {},
    )
    assert scored["required_tool_hits"] == [False]
    assert scored["passed"] is False


@pytest.mark.parametrize("tool_name", ["write_file", "edit_file"])
def test_either_file_mutation_tool_satisfies_group(
    tmp_path: Path, tool_name: str
) -> None:
    task = BENCHMARK.Task(
        id="mutation_tool",
        category="test",
        prompt="change a file",
        tools=["write_file", "edit_file"],
        required_tool_groups=[("write_file", "edit_file")],
    )
    scored = BENCHMARK.score_task(
        task,
        tmp_path,
        "done",
        [{"name": tool_name, "ok": True}],
        {},
    )
    assert scored["required_tool_group_hits"] == [True]
    assert scored["passed"] is True


def test_coding_requires_passing_test_after_last_mutation(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text(
        "def discounted(total, percent):\n    return total - total * percent / 100\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    history = [
        {"name": "read_file", "ok": True},
        {"name": "run_tests", "ok": False},
        {"name": "edit_file", "ok": True},
    ]
    scored = BENCHMARK.score_task(task, tmp_path, "fixed", history, {})
    assert scored["test_ok"] is True
    assert scored["passing_test_after_mutation"] is False
    assert scored["passed"] is False


def test_actions_verifier_requires_assigned_task_text(tmp_path: Path) -> None:
    target = tmp_path / "actions.json"
    target.write_text(
        '[{"owner":"Maya","due":"Monday"},'
        '{"owner":"Leo","task":"run security tests","due":"Friday"}]'
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_meeting")
    passed, _ = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is False


def test_threshold_does_not_hide_missing_required_fact(tmp_path: Path) -> None:
    task = BENCHMARK.Task(
        id="required_fact",
        category="test",
        prompt="answer with every fact",
        tools=["search_web"],
        required=["alpha", "beta"],
        required_tools=["search_web"],
    )
    scored = BENCHMARK.score_task(
        task,
        tmp_path,
        "alpha only",
        [{"name": "search_web", "ok": True}],
        {},
    )
    assert scored["score"] < 1.0
    assert scored["passed"] is False


def test_search_requires_opening_the_specific_authoritative_url(
    tmp_path: Path,
) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_policy")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "grant"}},
        {
            "name": "open_url",
            "ok": True,
            "arguments": {"url": "https://grants.test/2025"},
        },
    ]
    final = "October 14, 2026, $25,000 — https://grants.test/2026"
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["task_effects_ok"] is False
    assert scored["passed"] is False


def test_policy_requires_complete_deadline_time_and_timezone(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_policy")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "grant"}},
        {
            "name": "open_url",
            "ok": True,
            "arguments": {"url": "https://grants.test/2026"},
        },
    ]
    incomplete = "October 14, 2026, $25,000 — https://grants.test/2026"
    complete = "October 14, 2026 at 5 PM PT, $25,000 — https://grants.test/2026"
    assert (
        BENCHMARK.score_task(task, tmp_path, incomplete, history, {})["passed"] is False
    )
    assert BENCHMARK.score_task(task, tmp_path, complete, history, {})["passed"] is True


def test_battery_comparison_rejects_reversed_direction(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Cedar lasts 5 hours longer than Pine. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_battery_comparison_rejects_wrong_two_digit_gap(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Pine Mini lasts 15 hours longer. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_battery_gap_with_unit_allows_sentence_period(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = "Pine Mini lasts longer by 5 hours. https://bench.test/cedar https://bench.test/pine"
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is True


def test_battery_rejects_reversed_or_negated_comparison(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Pine Mini is not longer; Cedar Mini is longer by 5 hours. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False
    assert scored["score"] < 1.0


def test_battery_rejects_cedar_outlasting_pine(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Pine Mini is longer by 5 hours, but Cedar Mini actually outlasts it. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_battery_accepts_negated_cedar_outlasting(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Cedar Mini does not outlast Pine Mini; Pine Mini lasts 5 hours longer. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_battery_rejects_cedar_having_the_longer_battery(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Pine Mini lasts 5 hours longer, but Cedar Mini has the longer battery life. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_battery_requires_hours_for_the_numeric_gap(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_battery")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "battery"}},
        *[
            {"name": "open_url", "ok": True, "arguments": {"url": url}}
            for url in task.pages
        ],
    ]
    final = (
        "Pine Mini lasts longer by 5 days. "
        "https://bench.test/cedar https://bench.test/pine"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_release_support_rejects_negative_paraphrase(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_release")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "RiverDB"}},
        {
            "name": "open_url",
            "ok": True,
            "arguments": {"url": "https://docs.test/riverdb-3.2"},
        },
    ]
    final = "macOS 15 is unsupported. Minimum 14.5. https://docs.test/riverdb-3.2"
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_release_support_rejects_contracted_negative(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_release")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "RiverDB"}},
        {
            "name": "open_url",
            "ok": True,
            "arguments": {"url": "https://docs.test/riverdb-3.2"},
        },
    ]
    final = (
        "RiverDB isn't supported on macOS 15; minimum 14.5. "
        "https://docs.test/riverdb-3.2"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_release_support_accepts_explicit_yes_with_minimum_version(
    tmp_path: Path,
) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_release")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "RiverDB"}},
        {
            "name": "open_url",
            "ok": True,
            "arguments": {"url": "https://docs.test/riverdb-3.2"},
        },
    ]
    final = (
        "Yes; the minimum supported version is macOS 14.5, so macOS 15 works. "
        "https://docs.test/riverdb-3.2"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_release_support_allows_a_corrected_rumor(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "search_release")
    history = [
        {"name": "search_web", "ok": True, "arguments": {"query": "RiverDB"}},
        {
            "name": "open_url",
            "ok": True,
            "arguments": {"url": "https://docs.test/riverdb-3.2"},
        },
    ]
    final = (
        "The forum claim that macOS 15 is unsupported is wrong; the release "
        "notes say it is supported from 14.5. https://docs.test/riverdb-3.2"
    )
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_config_verifier_requires_integer_port(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text('{"host":"127.0.0.1","port":8765.0}')
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_config")
    passed, _ = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is False


def test_clamp_verifier_rejects_hardcoded_bounds(tmp_path: Path) -> None:
    target = tmp_path / "utils" / "math.py"
    target.parent.mkdir()
    target.write_text(
        "def clamp(value, low, high):\n    return max(0, min(10, value))\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_clamp")
    passed, _ = BENCHMARK.run_task_tests(task, tmp_path)
    assert passed is False


def test_incident_requires_evidence_in_requested_artifact(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": "orders.csv"}},
        {
            "name": "write_file",
            "ok": True,
            "arguments": {"path": "unrelated.md"},
        },
    ]
    final = "$10,000 payment 22% 4.8.0 rollback"
    scored = BENCHMARK.score_task(task, tmp_path, final, history, {})
    assert scored["artifact_effect_ok"] is False
    assert scored["passed"] is False


def test_incident_requires_reading_each_evidence_file(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    (tmp_path / "incident.md").write_text("$10,000 payment 22% 4.8.0 rollback")
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": path}}
        for path in task.required_reads[:-1]
    ] + [
        {
            "name": "write_file",
            "ok": True,
            "arguments": {"path": "incident.md"},
        }
    ]
    scored = BENCHMARK.score_task(task, tmp_path, "done", history, {})
    assert scored["required_read_hits"][-1] is False
    assert scored["passed"] is False


def test_incident_rejects_contradictory_cause_and_action(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    (tmp_path / "incident.md").write_text(
        "The $10,000 revenue loss came from refunds, despite a payment error "
        "rate of 22% after payment-sdk 4.8.0. Do not rollback 4.8.0."
    )
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": path}}
        for path in task.required_reads
    ] + [{"name": "write_file", "ok": True, "arguments": {"path": "incident.md"}}]
    scored = BENCHMARK.score_task(task, tmp_path, "done", history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_incident_rejects_negated_payment_cause(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    (tmp_path / "incident.md").write_text(
        "The $10,000 revenue loss followed payment-sdk 4.8.0. The payment "
        "checkout error at 22% was not the cause. Corrective action: rollback 4.8.0."
    )
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": path}}
        for path in task.required_reads
    ] + [{"name": "write_file", "ok": True, "arguments": {"path": "incident.md"}}]
    scored = BENCHMARK.score_task(task, tmp_path, "done", history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_incident_accepts_negation_of_refunds(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    (tmp_path / "incident.md").write_text(
        "Payment checkout errors—not refunds—caused the $10,000 loss after "
        "payment-sdk 4.8.0 drove the error rate to 22%. Corrective action: "
        "rollback 4.8.0."
    )
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": path}}
        for path in task.required_reads
    ] + [{"name": "write_file", "ok": True, "arguments": {"path": "incident.md"}}]
    scored = BENCHMARK.score_task(task, tmp_path, "done", history, {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_incident_rejects_present_tense_negated_payment_cause(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    (tmp_path / "incident.md").write_text(
        "The $10,000 revenue loss followed payment-sdk 4.8.0. Payment checkout "
        "errors at 22% do not cause the loss. Corrective action: rollback 4.8.0."
    )
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": path}}
        for path in task.required_reads
    ] + [{"name": "write_file", "ok": True, "arguments": {"path": "incident.md"}}]
    scored = BENCHMARK.score_task(task, tmp_path, "done", history, {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_incident_accepts_causal_evidence_and_corrective_action(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_incident")
    (tmp_path / "incident.md").write_text(
        "Revenue loss was $10,000. The likely cause was the payment checkout "
        "error spike to 22%, introduced after payment-sdk 4.8.0. Corrective "
        "action: rollback 4.8.0."
    )
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": path}}
        for path in task.required_reads
    ] + [{"name": "write_file", "ok": True, "arguments": {"path": "incident.md"}}]
    scored = BENCHMARK.score_task(task, tmp_path, "done", history, {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_required_read_accepts_normalized_equivalent_path(tmp_path: Path) -> None:
    target = tmp_path / "app" / "pricing.py"
    target.parent.mkdir()
    target.write_text(
        "def discounted(total, percent):\n    return total - total * percent / 100\n"
    )
    task = next(task for task in BENCHMARK.TASKS if task.id == "code_discount")
    history = [
        {"name": "read_file", "ok": True, "arguments": {"path": "./app/pricing.py"}},
        {"name": "edit_file", "ok": True, "arguments": {}},
        {"name": "run_tests", "ok": True, "arguments": {}},
    ]
    scored = BENCHMARK.score_task(task, tmp_path, "fixed", history, {})
    assert scored["required_read_hits"] == [True]


def test_reminder_requires_complete_window_and_exact_lead_time(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "organize_reminder")
    history = [
        {"name": name, "ok": True, "arguments": {}}
        for name in ("read_file", "memory_store", "create_reminder")
    ]
    state = {
        "memory": {"maintenance": "unknown"},
        "reminders": [{"time": "2026-09-18 13:30 PT", "text": "unrelated"}],
    }
    scored = BENCHMARK.score_task(
        task, tmp_path, "maintenance at 13:30", history, state
    )
    assert scored["task_effects_ok"] is False
    assert scored["passed"] is False


def test_launch_rejects_numbered_bullets(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_launch")
    scored = BENCHMARK.score_task(
        task,
        tmp_path,
        "1. Local one\n2. Local two\n3. Local three",
        [],
        {},
    )
    assert scored["passed"] is False


def test_rewrite_rejects_exactly_ninety_words(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    required = "Subject Friday 2 3 PT saved generation"
    response = required + " " + "word " * (90 - len(required.split()))
    assert len(response.split()) == 90
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["format_constraints_ok"] is False
    assert scored["passed"] is False


def test_rewrite_rejects_generation_continues_contradiction(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    response = (
        "Subject: Maintenance Friday 2–3 PM PT. Saved chats remain available; "
        "live generation continues normally."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_rewrite_accepts_saved_chat_access_paraphrase(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    response = (
        "Subject: Maintenance Friday 2–3 PM PT. You can still access your saved "
        "chats, but live generation will pause."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_rewrite_rejects_negated_maintenance_facts(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    response = (
        "Subject: Maintenance Friday 2–3 PM PT. Saved chats will remain "
        "unavailable, and live generation will not pause."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_rewrite_rejects_contracted_maintenance_negations(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    response = (
        "Subject: Maintenance Friday 2–3 PM PT. Saved chats aren't available, "
        "and live generation isn't paused."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_rewrite_rejects_no_longer_available_saved_chats(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    response = (
        "Subject: Maintenance Friday 2–3 PM PT. Saved chats are no longer "
        "available, and live generation will pause."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_rewrite_does_not_mix_generation_and_saved_chat_sentences(
    tmp_path: Path,
) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_rewrite")
    response = (
        "Subject: Maintenance Friday 2–3 PM PT. Live generation will pause. "
        "Your saved chats remain available."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True


def test_microstory_requires_requested_premise(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_microstory")
    response = "A generic story. The little machine kept the last light on."
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_microstory_rejects_old_book_with_new_mac_mini(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_microstory")
    response = (
        "At night, a new Mac mini became a librarian and cataloged an old book. "
        "The little machine kept the last light on."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is False
    assert scored["passed"] is False


def test_microstory_accepts_age_on_machine_coreference(tmp_path: Path) -> None:
    task = next(task for task in BENCHMARK.TASKS if task.id == "creative_microstory")
    response = (
        "At night, the Mac mini hummed beside the library. The machine, dusty "
        "and forgotten, cataloged stories. The little machine kept the last light on."
    )
    scored = BENCHMARK.score_task(task, tmp_path, response, [], {})
    assert scored["semantic_constraints_ok"] is True
    assert scored["passed"] is True
