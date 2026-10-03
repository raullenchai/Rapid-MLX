"""Unit tests for the argparse ``type=`` helpers in ``rapid_mlx.cli_parser``.

These validators reject bad values at parse time, before any model download
or load. ``cli_parser`` imports only light modules, so this runs on the
no-MLX Linux lane too.
"""

from __future__ import annotations

import argparse
import math

import pytest

from rapid_mlx import cli_parser


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("7", 7), ("+3", 3)])
def test_non_negative_int_accepts_zero_and_positive(raw: str, expected: int) -> None:
    assert cli_parser.non_negative_int(raw) == expected


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("-1", "non-negative integer, got -1"),
        ("1.5", "non-negative integer, got '1.5'"),
        ("abc", "non-negative integer, got 'abc'"),
    ],
)
def test_non_negative_int_rejects_negative_and_non_integer(
    raw: str, match: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=match):
        cli_parser.non_negative_int(raw)


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("2048", 2048)])
def test_positive_int_accepts_positive(raw: str, expected: int) -> None:
    assert cli_parser.positive_int(raw) == expected


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("0", "positive integer, got 0"),
        ("-4", "positive integer, got -4"),
        ("x", "positive integer, got 'x'"),
    ],
)
def test_positive_int_rejects_non_positive_and_non_integer(
    raw: str, match: str
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=match):
        cli_parser.positive_int(raw)


@pytest.mark.parametrize(("raw", "expected"), [("0.5", 0.5), ("12", 12.0)])
def test_positive_finite_float_accepts_positive_finite(
    raw: str, expected: float
) -> None:
    assert math.isclose(cli_parser.positive_finite_float(raw), expected)


@pytest.mark.parametrize("raw", ["0", "-1.5", "inf", "-inf", "nan", "lots"])
def test_positive_finite_float_rejects_non_positive_non_finite_and_garbage(
    raw: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive finite number"):
        cli_parser.positive_finite_float(raw)


def test_resolve_cli_version_falls_back_to_dev_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("rapid-mlx")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert cli_parser._resolve_cli_version() == "dev"


def test_resolve_cli_version_reports_installed_distribution() -> None:
    import importlib.metadata

    assert cli_parser._resolve_cli_version() == importlib.metadata.version("rapid-mlx")
