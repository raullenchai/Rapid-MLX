"""Normal pull cancellation preserves the original retry command, without a traceback."""

import os
import signal
import subprocess
import sys

import pytest

from rapid_mlx import cli


@pytest.mark.parametrize(
    "arguments",
    [
        ["whisper-tiny"],
        ["owner/model", "--bits", "4"],
        ["owner/model", "--format", "mxfp4"],
    ],
)
def test_pull_ctrl_c_exits_cleanly(monkeypatch, capsys, arguments):
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    monkeypatch.setenv("RAPID_MLX_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "pull", *arguments])

    def interrupted_transfer(_args):
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(cli, "pull_command", interrupted_transfer)
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 130
    output = capsys.readouterr()
    assert "Download cancelled. Completed files stay cached." in output.err
    assert "Retry: rapid-mlx pull " + " ".join(arguments) in output.err
    assert "Traceback" not in output.err


def test_pull_interrupt_process_exit():
    """Exercise the CLI entry point with a real SIGINT and capture process output."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import signal; from rapid_mlx import cli; "
            "cli.pull_command = lambda args: signal.raise_signal(signal.SIGINT); "
            "cli.main()",
            "pull",
            "whisper-tiny",
        ],
        env={
            **os.environ,
            "RAPID_MLX_TELEMETRY": "0",
            "RAPID_MLX_AUTO_PULL": "1",
            "RAPID_MLX_DISABLE_VERSION_CHECK": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 130
    assert "Retry: rapid-mlx pull whisper-tiny" in result.stderr
    assert "Traceback" not in result.stderr
    assert "KeyboardInterrupt" not in result.stderr
