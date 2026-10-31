"""Tests for CLI stdout buffering."""

import io
import sys

import pytest

from rapid_mlx import cli


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_non_tty_reconfigure_failure_does_not_stop_main(error, monkeypatch):
    class FailingStdout(io.StringIO):
        def reconfigure(self, *, line_buffering):
            assert line_buffering is True
            raise error("reconfigure failed")

    out = FailingStdout()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "version"])

    cli.main()

    assert out.getvalue().startswith("rapid-mlx ")
