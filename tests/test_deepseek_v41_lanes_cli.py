"""Registered lane selection and retired environment migration contracts."""

from types import SimpleNamespace

import pytest

from rapid_mlx.models.deepseek_v41_native.lanes import (
    LANES_ENV,
    request_lanes,
    serve_lanes,
)
from tests.test_deepseek_v41_artifacts import _product_serve_args


@pytest.fixture(autouse=True)
def clear_retired_env(monkeypatch):
    monkeypatch.delenv(LANES_ENV, raising=False)


@pytest.mark.parametrize("value", [1, 2, 4, 8])
def test_lane_choices(value):
    assert request_lanes(value) == value
    args = _product_serve_args("--dsv41-lanes", str(value))
    assert args.dsv41_lanes == value
    assert serve_lanes(value, args.model) == value


@pytest.mark.parametrize("flags", [(), ("--no-dsv41-lanes",), ("--dsv41-lanes", "1")])
def test_serial_default_and_override(flags):
    assert _product_serve_args(*flags).dsv41_lanes == 1
    assert request_lanes() == 1
    assert serve_lanes(1, None) == 1


@pytest.mark.parametrize("value", [0, 3, -1, 16, "2", None, True])
def test_invalid_programmatic_lanes(value):
    with pytest.raises(ValueError, match="--dsv41-lanes must be"):
        request_lanes(value)


@pytest.mark.parametrize("value", ["0", "3", "two", ""])
def test_invalid_parser_lanes(value):
    with pytest.raises(SystemExit) as exc:
        _product_serve_args("--dsv41-lanes", value)
    assert exc.value.code == 2


def test_conflicting_flags():
    with pytest.raises(SystemExit) as exc:
        _product_serve_args("--dsv41-lanes", "4", "--no-dsv41-lanes")
    assert exc.value.code == 2


@pytest.mark.parametrize("value", ["", "1", "2", "4", "8", "invalid"])
def test_retired_environment_errors_even_with_cli(monkeypatch, capsys, value):
    from rapid_mlx import cli

    monkeypatch.setenv(LANES_ENV, value)
    with pytest.raises(ValueError, match="retired; unset it and use --dsv41-lanes"):
        request_lanes(4)
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(SimpleNamespace(dsv41_lanes=4, model="unused"))
    assert exc.value.code == 2
    assert "retired; unset it and use --dsv41-lanes" in capsys.readouterr().err


@pytest.mark.parametrize("model", [None, "unrelated-model"])
def test_other_routes_reject_batched_lanes(model):
    with pytest.raises(ValueError, match="requires a native DeepSeek V4.1 model"):
        serve_lanes(2, model)


def test_retired_env_is_not_allowed_and_only_migration_boundary_may_reference_it():
    from tests.test_no_out_of_band_routing import (
        ALLOWED_RAPID_MLX_ENV_VARS,
        _check_env_constant,
    )

    assert LANES_ENV not in ALLOWED_RAPID_MLX_ENV_VARS
    offenders = []
    _check_env_constant(LANES_ENV, "models/deepseek_v41_native/lanes.py", 1, offenders)
    assert not offenders
    _check_env_constant(LANES_ENV, "cli.py", 1, offenders)
    assert len(offenders) == 1 and "retired" in offenders[0]
