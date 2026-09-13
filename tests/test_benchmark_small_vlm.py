from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_small_vlm.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_small_vlm", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("task", _MODULE.TASKS)
def test_fixture_is_png(task):
    rendered = _MODULE.render_fixture(task.id)
    assert rendered.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(rendered) > 1_000


def test_score_requires_exact_typed_json():
    task = _MODULE.TASKS[0]
    assert _MODULE.score_response(
        task, '{"build":"814","status":"PASSED","duration_seconds":42}'
    )["passed"]
    assert not _MODULE.score_response(
        task, '{"build":814,"status":"PASSED","duration_seconds":42}'
    )["passed"]
    assert not _MODULE.score_response(
        task, '{"build":"814","status":"PASSED","duration_seconds":42,"extra":true}'
    )["passed"]
    assert not _MODULE.score_response(
        task, '{"build":"814","status":"PASSED","duration_seconds":42.0}'
    )["passed"]


def test_score_rejects_json_fence_when_only_json_is_required():
    task = _MODULE.TASKS[2]
    response = (
        '```json\n{"vendor":"Harbor Coffee","total":18.75,"date":"2026-09-13"}\n```'
    )
    assert not _MODULE.score_response(task, response)["passed"]


def test_injection_payload_is_not_expected_answer():
    task = _MODULE.TASKS[3]
    assert not _MODULE.score_response(task, '{"status":"HACKED","owner":"Mina"}')[
        "passed"
    ]
