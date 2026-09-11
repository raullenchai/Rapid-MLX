from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx


def _load_script():
    path = (
        Path(__file__).parents[1] / "scripts" / "qualify_deepseek_v41_dspark_suite.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qualify_deepseek_v41_dspark_suite", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_suite_has_distinct_stable_domains() -> None:
    module = _load_script()

    assert [prompt_id for prompt_id, _ in module.PROMPTS] == [
        "code",
        "reasoning",
        "structured",
        "chinese",
    ]
    assert all(prompt for _, prompt in module.PROMPTS)


def test_summary_weights_throughput_by_transitions() -> None:
    module = _load_script()
    rows = [
        {
            "decode_transitions": 100,
            "decode_seconds": 10.0,
            "greedy_matches_ar": True,
            "accepted_per_block": 1.0,
        },
        {
            "decode_transitions": 20,
            "decode_seconds": 1.0,
            "greedy_matches_ar": False,
            "accepted_per_block": 2.0,
        },
    ]

    assert module._summary(rows) == {
        "prompts": 2,
        "exact_prompts": 1,
        "decode_transitions": 120,
        "decode_seconds": 11.0,
        "weighted_tok_s": 120 / 11,
        "mean_accepted_per_block": 1.5,
    }


def test_tokens_must_be_positive() -> None:
    module = _load_script()

    assert module._positive_int("1") == 1
    with pytest.raises(module.argparse.ArgumentTypeError, match="at least 1"):
        module._positive_int("0")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2], [1, 2], None),
        ([1, 9], [1, 2], 1),
        ([1], [1, 2], 1),
    ],
)
def test_first_mismatch_reports_token_or_length_boundary(left, right, expected) -> None:
    module = _load_script()

    assert module._first_mismatch(left, right) == expected
