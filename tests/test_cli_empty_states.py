"""Focused regression coverage for actionable top-level CLI empty states."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from vllm_mlx import cli


def test_serve_without_model_is_concise_and_points_to_recipe(capsys):
    with (
        mock.patch.object(sys, "argv", ["rapid-mlx", "serve"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "a model is required" in captured.err
    assert "rapid-mlx recipe" in captured.err
    assert "rapid-mlx serve qwen3.5-4b-4bit" in captured.err
    assert "--kv-cache" not in captured.err
    assert captured.err.count("\n") == 3


def test_serve_help_still_shows_full_reference(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["serve", "--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--kv-cache-quantization" in captured.out
    assert "rapid-mlx recipe" not in captured.err
