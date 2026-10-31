import logging
from types import SimpleNamespace

import pytest

from rapid_mlx.cli import build_parser
from rapid_mlx.community_bench import cli


def test_benchmark_run_parser_accepts_log_level():
    args = build_parser().parse_args(["benchmark", "run", "X", "--log-level", "debug"])
    assert args.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("env", "flag", "expected"),
    [
        (None, None, logging.WARNING),
        ("INFO", None, logging.INFO),
        ("INFO", "ERROR", logging.ERROR),
    ],
)
def test_benchmark_run_configures_logging(monkeypatch, capsys, env, flag, expected):
    root = logging.getLogger()
    original_level, original_handlers = root.level, root.handlers[:]
    try:
        root.handlers = [logging.StreamHandler()]
        root.setLevel(logging.DEBUG)
        if env is None:
            monkeypatch.delenv("RAPID_MLX_LOG_LEVEL", raising=False)
        else:
            monkeypatch.setenv("RAPID_MLX_LOG_LEVEL", env)
        monkeypatch.setattr(cli.LocalRunArchive, "default", lambda: object())

        def run_local(*args, **kwargs):
            # Simulate the server's import-time configuration.
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("rapid_mlx.engine_core").info("engine info marker")
            logging.getLogger("rapid_mlx.engine_core").warning("engine warning marker")
            return {"run_id": "test", "measurements": []}

        monkeypatch.setattr(cli, "run_local", run_local)
        args = SimpleNamespace(
            benchmark_action="run", benchmark_model="X", json=True, log_level=flag
        )
        assert cli.benchmark_command(args) == 0
        assert root.level == expected
        stderr = capsys.readouterr().err
        assert ("engine info marker" in stderr) == (expected <= logging.INFO)
        assert ("engine warning marker" in stderr) == (expected <= logging.WARNING)
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
