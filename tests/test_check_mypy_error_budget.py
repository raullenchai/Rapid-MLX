from __future__ import annotations

import subprocess

import pytest

from scripts import check_mypy_error_budget
from scripts.check_mypy_error_budget import (
    _mypy_command,
    compare_budget,
    parse_baseline,
    parse_error_counts,
    render_baseline,
    run_mypy,
)


def test_mypy_command_explicitly_scans_every_baselined_package() -> None:
    command = _mypy_command()
    assert "vllm_mlx/" in command
    assert "videox_fun_mlx/" in command


def test_error_counts_ignore_message_and_locations() -> None:
    output = """\
vllm_mlx/a.py:10:5: error: First [assignment]
vllm_mlx/a.py:99: error: Name already defined on line 10 [no-redef]
vllm_mlx/a.py:10: note: context
vllm_mlx/b.py:3: error: Other [arg-type]
"""

    assert parse_error_counts(output) == {"vllm_mlx/a.py": 2, "vllm_mlx/b.py": 1}


def test_budget_rejects_new_dirty_file_and_count_growth() -> None:
    growth, reductions = compare_budget(
        {"vllm_mlx/a.py": 2},
        {"vllm_mlx/a.py": 3, "vllm_mlx/new.py": 1},
    )

    assert growth == {
        "vllm_mlx/a.py": (2, 3),
        "vllm_mlx/new.py": (0, 1),
    }
    assert not reductions


def test_budget_requires_reductions_to_be_recorded() -> None:
    growth, reductions = compare_budget(
        {"vllm_mlx/a.py": 2, "vllm_mlx/cleaned.py": 1},
        {"vllm_mlx/a.py": 1},
    )

    assert not growth
    assert reductions == {
        "vllm_mlx/a.py": (2, 1),
        "vllm_mlx/cleaned.py": (1, 0),
    }


def test_equal_per_file_budget_passes_even_if_diagnostic_identity_changes() -> None:
    assert compare_budget({"vllm_mlx/a.py": 2}, {"vllm_mlx/a.py": 2}) == ({}, {})


def test_baseline_round_trip() -> None:
    counts = {"vllm_mlx/b.py": 2, "vllm_mlx/a.py": 1}
    assert parse_baseline(render_baseline(counts)) == counts


@pytest.mark.parametrize(
    "text",
    [
        "vllm_mlx/a.py\n",
        "vllm_mlx/a.py nope\n",
        "vllm_mlx/a.py 0\n",
        "vllm_mlx/a.py 1\nvllm_mlx/a.py 1\n",
    ],
)
def test_malformed_baseline_fails_closed(text: str) -> None:
    with pytest.raises(ValueError):
        parse_baseline(text)


@pytest.mark.parametrize(
    ("returncode", "stdout", "message"),
    [
        (2, "mypy crashed", "operationally"),
        (1, "unexpected output format", "no parseable errors"),
    ],
)
def test_untrustworthy_mypy_run_fails_closed(
    monkeypatch, returncode: int, stdout: str, message: str
) -> None:
    monkeypatch.setattr(
        check_mypy_error_budget.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=returncode, stdout=stdout
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        run_mypy(["mypy"])
