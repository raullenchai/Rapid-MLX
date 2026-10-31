"""Regression coverage for user-facing benchmark and share messages."""

import ast
import json
from argparse import Namespace
from pathlib import Path

import pytest

from rapid_mlx.bench import _server
from rapid_mlx.cli_parser import build_parser
from rapid_mlx.community_bench.cli import _print_failure
from rapid_mlx.community_bench.local_runner import LocalBenchmarkError
from rapid_mlx.first_run import FIRST_RUN_MODEL, FIRST_RUN_MODEL_SIZE


def test_arrays_cache_configuration_logs_at_info():
    source = Path("rapid_mlx/engine/batched.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and "uses ArraysCache" in node.args[0].value
    ]
    assert len(calls) == 1
    assert isinstance(calls[0].func, ast.Attribute)
    assert isinstance(calls[0].func.value, ast.Name)
    assert calls[0].func.value.id == "logger"
    assert calls[0].func.attr == "info"


@pytest.mark.parametrize(
    ("rc", "message"),
    [
        (
            -9,
            "The local model server was killed by SIGKILL (signal 9) while loading the model. On macOS this usually means the Mac ran out of memory. Try a smaller model (for example a lower-bit quantization), or close other memory-heavy apps and retry.",
        ),
        (
            -15,
            "The local model server was killed by SIGTERM (signal 15) while loading the model.",
        ),
        (
            -999,
            "The local model server was killed by signal 999 while loading the model.",
        ),
        (3, "The local model server exited with code 3 while loading the model."),
    ],
)
def test_server_start_exit_message(rc, message):
    class Proc:
        def poll(self):
            return rc

    with pytest.raises(_server.ServerStartFailed) as error:
        _server._wait_for_health("http://localhost/health", Proc(), 1)
    assert str(error.value) == message


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize(
    "run,saved", [(None, False), ({"run_id": "abc"}, False), ({"run_id": "abc"}, True)]
)
def test_failure_removes_one_error_prefix(capsys, json_output, run, saved):
    error = LocalBenchmarkError("error: error: invalid input", run, saved=saved)
    assert _print_failure(Namespace(json=json_output), error) == 1
    output = capsys.readouterr().err
    if json_output:
        assert json.loads(output)["error"] == "error: invalid input"
    else:
        assert output.rstrip().endswith(": error: invalid input")


def test_help_copy(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "run", "--help"])
    help_text = capsys.readouterr().out
    assert "DSpark" not in help_text
    assert "DDTree" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["share", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert (
        "Extra serve flags passed after -- are refused (they can carry credentials)"
        in help_text
    )


def test_chat_help_describes_local_default(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["chat", "--help"])
    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "rapid_mlx." not in help_text
    assert "already on this Mac" in help_text
    assert FIRST_RUN_MODEL in help_text
    assert FIRST_RUN_MODEL_SIZE in help_text
