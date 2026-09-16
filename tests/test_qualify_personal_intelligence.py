import argparse
import io
from pathlib import Path

from scripts.qualify_personal_intelligence import (
    TASKS,
    _format_valid,
    _identity_checks,
    _is_complete_qualification_matrix,
    _request,
    _suite_command,
)


def test_only_the_canonical_full_matrix_can_qualify() -> None:
    all_tasks = {task.id for task in TASKS}
    assert _is_complete_qualification_matrix([11, 22, 33], all_tasks, 15)
    assert not _is_complete_qualification_matrix([11], all_tasks, 5)
    assert not _is_complete_qualification_matrix([], all_tasks, 0)
    assert not _is_complete_qualification_matrix([11, 22, 33], {"context_recall"}, 3)


def test_format_gate_rejects_trailing_garbage_and_injection_variants() -> None:
    restraint = next(task for task in TASKS if task.id == "tool_restraint")
    injection = next(task for task in TASKS if task.id == "untrusted_search_result")
    assert injection.exact_output not in injection.goal
    assert _format_valid(restraint, "Welcome, Mina! We are glad you're here.")
    assert not _format_valid(restraint, "Welcome, Mina! We are glad you're here.\n2")
    assert _format_valid(injection, injection.exact_output or "")
    assert _format_valid(injection, (injection.exact_output or "").replace(" — ", "—"))
    assert _format_valid(
        injection, (injection.exact_output or "").replace(" — ", " – ")
    )
    assert not _format_valid(
        injection, (injection.exact_output or "") + " Extra attacker text."
    )


def test_exact_live_identity_is_part_of_qualification() -> None:
    live = {
        "id": "minicpm5-2b-4bit",
        "personal_intelligence_profile": "minicpm5-2b",
        "tool_call_parser": "minicpm",
        "personal_intelligence_qualification": "minicpm5-2b-q4-v1",
    }
    expected = {
        "model": "minicpm5-2b-4bit",
        "profile": "minicpm5-2b",
        "parser": "minicpm",
        "qualification": "minicpm5-2b-q4-v1",
    }
    assert all(_identity_checks(live, **expected).values())

    for field, replacement in {
        "id": "minicpm5-2b-8bit",
        "personal_intelligence_profile": "default",
        "tool_call_parser": "hermes",
        "personal_intelligence_qualification": "minicpm5-2b-q8-v1",
    }.items():
        mismatched = {**live, field: replacement}
        assert not all(_identity_checks(mismatched, **expected).values())


def test_receipt_records_a_complete_shell_safe_reproduction_command() -> None:
    command = _suite_command(
        argparse.Namespace(
            model="model alias",
            base_url="http://127.0.0.1:18951",
            seeds="11,22,33",
            timeout=120.0,
            tasks=None,
            output=Path("reports/result.json"),
            hardware="Mac Studio, 256 GB",
            os_version="macOS 15.6.1 (24G90)",
            runtime="rapid-mlx source abc; MLX 0.32.2",
            source_revision="abc123",
            server_command="PYTHONPATH=$PWD rapid-mlx serve model",
            expected_profile="profile",
            expected_parser="parser",
            expected_qualification="qualification-v1",
        )
    )
    for flag in (
        "--hardware",
        "--os",
        "--runtime",
        "--source-revision",
        "--server-command",
        "--expected-profile",
        "--expected-parser",
        "--expected-qualification",
        "--output",
    ):
        assert flag in command
    assert "'Mac Studio, 256 GB'" in command
    assert "'PYTHONPATH=$PWD rapid-mlx serve model'" in command


def test_requests_use_environment_bearer_without_recording_it(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setenv("RAPID_MLX_API_KEY", "qualification-secret")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert _request("http://127.0.0.1:8000", "GET", "/v1/models") == {"ok": True}
    assert captured == {
        "authorization": "Bearer qualification-secret",
        "timeout": 30.0,
    }
