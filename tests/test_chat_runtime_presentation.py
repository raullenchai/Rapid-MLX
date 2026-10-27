"""Desktop and terminal consume the same bounded owner tool presentation."""

import json

from rapid_mlx.chat_runtime.events import Envelope
from rapid_mlx.chat_runtime.presentation import ToolPresentation, result_presentation


def test_started_and_result_keep_concurrent_call_identity():
    presenter = ToolPresentation()
    presenter.event(
        {
            "type": "tool.started",
            "call_id": "read",
            "name": "read_file",
            "args": {"path": "calc.py"},
        }
    )
    presenter.event(
        {
            "type": "tool.started",
            "call_id": "search",
            "name": "web_search",
            "args": {"query": "weather"},
        }
    )
    search = presenter.event(
        {
            "type": "tool.result",
            "call_id": "search",
            "result": "[1] One\nhttps://example.com\n\n[2] Two\nhttps://example.org",
        }
    )
    read = presenter.event(
        {
            "type": "tool.result",
            "call_id": "read",
            "result": json.dumps(
                {
                    "content": "1|def add(a, b):\n2|    return a-b",
                    "total_lines": 2,
                    "truncated": False,
                }
            ),
        }
    )
    assert (search["label"], search["target"], search["summary"]) == (
        "Search",
        "weather",
        "2 results",
    )
    assert (read["label"], read["target"], read["summary"]) == (
        "Read",
        "calc.py",
        "2 lines",
    )
    assert "return a-b" not in read["summary"]
    Envelope(turn_id="turn", seq=1, event=read)


def test_action_operation_supplies_relative_target_and_real_diff():
    presenter = ToolPresentation("/workspace")
    preview = "--- /workspace/calc.py\n+++ /workspace/calc.py\n@@ -1,2 +1,2 @@\n def add(a,b):\n-    return a-b\n+    return a+b"
    started = presenter.event(
        {
            "type": "tool.started",
            "call_id": "edit",
            "name": "patch",
            "args": {"op_id": "op"},
        },
        {"summary": "Edited /workspace/calc.py (+1 −1)", "preview": preview},
    )
    assert started["label"] == "Edit"
    assert started["target"] == "calc.py"
    result = presenter.event(
        {
            "type": "tool.result",
            "call_id": "edit",
            "result": "Edited /workspace/calc.py (+1 −1)",
        }
    )
    assert result["summary"] == "(+1 −1)"
    assert result["details"] == preview.splitlines()[2:]
    assert result["details_kind"] == "diff"
    denied = presenter.event(
        {
            "type": "tool.result",
            "call_id": "edit",
            "result": "Action denied.",
            "outcome": "error",
        }
    )
    assert denied["details"] == []
    assert denied["summary"] == "Action denied."


def test_diff_bounded_and_secret_preview_stays_redacted():
    preview = "@@ -1 +1 @@\n" + "\n".join(f"+line {n}" for n in range(20))
    result = result_presentation("write_file", "Edited file (+20 −0)", preview=preview)
    assert len(result["details"]) == 8
    assert result["omitted_lines"] == 13
    hidden = result_presentation(
        "write_file",
        "Edited .env (+1 −1)",
        preview="[Secrets-like file contents hidden; 42 new bytes]",
    )
    assert hidden["details"] == ["[Secrets-like file contents hidden; 42 new bytes]"]


def test_command_first_last_lines_exit_and_time():
    result = result_presentation(
        "execute_code", "Ran `python3 code` (exit 0, 0.1s)\nfirst\nmiddle\nlast\n"
    )
    assert result["summary"] == "first … last (exit 0, 0.1s)"
    assert result["details"] == ["first", "last"]
    assert result["omitted_lines"] == 1
    assert result["details_kind"] == "output"
    failed = result_presentation(
        "terminal", "Ran `pytest` (exit 1, 0.2s)\nFAILED test_calc", outcome="error"
    )
    assert failed["summary"] == "FAILED test_calc (exit 1, 0.2s)"


def test_weather_and_fetch_are_concise():
    weather = "Current weather for San Francisco:\n  Conditions: Partly cloudy\n  Temperature: 12°C\n  2026-10-26 Mon (today): Cloudy, 10–16°C, precip 5%, 0 mm"
    assert (
        result_presentation("weather", weather)["summary"]
        == "12°C · Partly cloudy · H 16° L 10°"
    )
    fetch = result_presentation(
        "web_fetch",
        "Fetched page:\n\n[2] API guide\nhttps://example.com\nLots of page text",
    )
    assert fetch["summary"] == "API guide"


def test_json_errors_and_unknown_payloads_never_render_raw_json():
    error = result_presentation(
        "read_file",
        '{"error":{"code":"context_limit","message":"Please narrow the read."}}',
        outcome="error",
    )
    assert error["summary"] == "Please narrow the read."
    for raw in ('{"token":"secret"}', '["secret"]'):
        result = result_presentation("unknown", raw)
        assert result["summary"] == "Completed"
        assert result["details"] == []


def test_permission_title_and_language_are_owner_fields():
    result = ToolPresentation().event(
        {"type": "permission.request", "tool": "execute_code", "preview": "print(1)"}
    )
    assert result["title"] == "Run Python code?"
    assert result["preview_language"] == "python"
