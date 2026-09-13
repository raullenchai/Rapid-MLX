# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import httpx
import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_glm53_real_tasks.py"
    spec = importlib.util.spec_from_file_location("benchmark_glm53_real_tasks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _module()


def test_exact_json_grader_rejects_prose_and_extra_keys() -> None:
    grader = bench._grade_exact_json({"answer": 42})
    assert grader('{"answer": 42}').passed
    assert not grader('Result: {"answer": 42}').passed
    assert not grader('{"answer": 42, "comment": "ok"}').passed


@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="successful code execution requires the macOS sandbox",
)
def test_coding_grader_executes_hidden_cases() -> None:
    good = """```python
def coalesce_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result
```"""
    assert bench._grade_interval_code(good).passed
    assert not bench._grade_interval_code(
        "def coalesce_intervals(intervals):\n    return intervals"
    ).passed


def test_coding_grader_rejects_unsafe_or_hanging_code() -> None:
    assert (
        "forbidden"
        in bench._grade_interval_code(
            "import os\ndef coalesce_intervals(intervals):\n    return []"
        ).detail
    )
    assert (
        "forbidden"
        in bench._grade_interval_code(
            "def coalesce_intervals(intervals):\n    return open('/tmp/x')"
        ).detail
    )
    assert (
        "forbidden top-level"
        in bench._grade_interval_code(
            "print('side effect')\ndef coalesce_intervals(intervals):\n    return []"
        ).detail
    )
    assert (
        "oversized"
        in bench._grade_interval_code(
            "def coalesce_intervals(intervals):\n    return [0] * 1000000000"
        ).detail
    )


def test_coding_grader_fails_closed_without_apple_sandbox(monkeypatch) -> None:
    monkeypatch.setattr(bench.shutil, "which", lambda _name: None)
    result = bench._grade_interval_code(
        "def coalesce_intervals(intervals):\n    return []"
    )
    assert not result.passed
    assert "sandbox-exec" in result.detail


def test_creative_grader_enforces_hard_constraints() -> None:
    body = " ".join(["You crossed the quiet lobby"] * 22)
    text = (
        f"{body} and the elevator remembered every version of your name. "
        "The doors opened onto Tuesday."
    )
    result = bench._grade_creative(text)
    assert result.passed
    assert result.manual_review_required
    assert not bench._grade_creative(text.replace("Tuesday.", "Monday.")).passed
    assert not bench._grade_creative(text.replace("quiet", "dreaming", 1)).passed


def test_instruction_grader_enforces_raw_json_and_key_order() -> None:
    raw = '{"alpha": "desserts", "beta": 6, "gamma": [23, 29, 31]}'
    assert bench._grade_instruction(raw).passed
    assert not bench._grade_instruction(f"```json\n{raw}\n```").passed
    reordered = '{"beta": 6, "alpha": "desserts", "gamma": [23, 29, 31]}'
    assert not bench._grade_instruction(reordered).passed


def test_compare_requires_exact_output_and_no_quality_regression(
    tmp_path: Path,
) -> None:
    def artifact(path: Path, output: str, passed: bool, tps: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_id": "knowledge.x",
                            "category": "knowledge",
                            "max_tokens": 10,
                            "thinking_budget": None,
                            "prompt_tokens": 4,
                            "output": output,
                            "client_completion_tps": tps,
                            "grade": {"passed": passed, "score": float(passed)},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    artifact(baseline, "same", True, 10.0)
    artifact(candidate, "same", True, 12.0)
    assert bench.compare_artifacts([str(baseline), str(candidate)]) == 0
    artifact(candidate, "different", True, 12.0)
    assert bench.compare_artifacts([str(baseline), str(candidate)]) == 3


@pytest.mark.parametrize("use_budget", [False, True])
def test_post_task_records_request_and_server_metrics(use_budget: bool) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/metrics":
            return httpx.Response(
                200,
                json={"latest": {"decode_tok_s": 12.5, "peak_memory_gb": 3.0}},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"answer": 42}', "reasoning": "r"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5},
            },
        )

    task = bench.Task(
        "test",
        "knowledge",
        "prompt",
        20,
        7,
        bench._grade_exact_json({"answer": 42}),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = bench._post_task(
            client,
            base_url="http://example.test/v1",
            model="model",
            task=task,
            use_thinking_budget=use_budget,
        )

    payload = json.loads(requests[0].content)
    assert payload.get("thinking_budget") == (7 if use_budget else None)
    assert result["thinking_budget"] == (7 if use_budget else None)
    assert result["server_metrics"]["decode_tok_s"] == 12.5
    assert result["reasoning"] == "r"
