import sys
import types

import numpy as np
import pytest

from rapid_mlx import cli
from rapid_mlx.cli_parser import build_parser
from scripts.audit_cli_config_fidelity import KWARG_INJECTOR_HELPERS


@pytest.fixture(autouse=True)
def probes(monkeypatch):
    existing_turboquant = sys.modules.get("rapid_mlx.turboquant")
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array = np.array
    core.float16 = np.float16
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    import rapid_mlx.api.utils as utils
    import rapid_mlx.model_auto_config as auto
    import rapid_mlx.spec_decode.mtp as mtp

    monkeypatch.setattr(
        mtp,
        "detect_mtp_eligibility",
        lambda *a, **kw: next(
            x for x in mtp.MTPEligibility if x != mtp.MTPEligibility.NONE
        ),
    )
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda sidecar: sidecar)

    monkeypatch.setattr(cli, "_auto_config_lookup_key", lambda x: x)
    monkeypatch.setattr(cli, "_gather_kv_cache_dtype_inputs", lambda *a: ({}, {}))
    monkeypatch.setattr(cli, "_native_mtp_runtime_ready", lambda *a: True)
    monkeypatch.setattr(auto, "detect_model_config", lambda *a: None)
    monkeypatch.setattr(utils, "resolve_serving_lane", lambda *a, **kw: (False, None))
    monkeypatch.setattr(cli, "_preflight_tensorfold_qwen27_or_exit", lambda *a: None)
    yield
    if existing_turboquant is None:
        sys.modules.pop("rapid_mlx.turboquant", None)


def plan(alias, *flags):
    args = build_parser().parse_args(["serve", alias, *flags])
    args._original_alias = alias
    return cli.resolve_serve_acceleration(args)


def test_plain_and_fields():
    result = plan("lfm2.5-1.2b-instruct-4bit")
    assert result.lane == "engine"
    assert result.scheduler_kwargs["spec_decode"] == "none"
    assert (
        set(result.scheduler_kwargs)
        == KWARG_INJECTOR_HELPERS["serve_acceleration_scheduler_kwargs"]
    )


def test_mtp_default_and_opt_out():
    result = plan("qwen3.6-27b-4bit")
    assert result.lane == "engine"
    assert result.scheduler_kwargs["spec_decode"] == "mtp"
    assert result.scheduler_kwargs["mtp_max_k"] == 2
    assert (
        plan("qwen3.6-27b-4bit", "--no-spec-decode").scheduler_kwargs["spec_decode"]
        == "none"
    )


@pytest.mark.parametrize(
    "alias,lane",
    [
        ("qwen3.8-flash-next-tensorfold", "tensorfold_mtp"),
        ("qwen3.8-27b-tensorfold", "tensorfold_dflash"),
    ],
)
def test_tensorfold(alias, lane):
    assert plan(alias).lane == lane


def test_dflash(monkeypatch):
    import rapid_mlx.speculative.dflash.eligibility as eligibility

    monkeypatch.setattr(eligibility, "have_runtime", lambda: True)
    assert (
        plan("qwen3.5-27b-8bit", "--speculative-config", '{"method":"dflash"}').lane
        == "dflash"
    )


def test_kv_flags(capsys):
    result = plan(
        "lfm2.5-1.2b-instruct-4bit",
        "--kv-cache-quantization",
        "--kv-cache-quantization-bits",
        "8",
    )
    assert result.scheduler_kwargs["kv_cache_quantization"]
    assert result.scheduler_kwargs["kv_cache_quantization_bits"] == 8
    with pytest.raises(SystemExit) as exc:
        plan(
            "lfm2.5-1.2b-instruct-4bit",
            "--kv-cache-turboquant",
            "v4",
            "--kv-cache-quantization",
        )
    assert exc.value.code == 1
    assert "mutually exclusive" in capsys.readouterr().out


def test_pflash():
    assert (
        plan("qwen3.6-27b-4bit", "--pflash", "always")
        .scheduler_kwargs["pflash_config"]
        .mode
        == "always"
    )


def test_prefill():
    args = build_parser().parse_args(
        ["serve", "lfm2.5-1.2b-instruct-4bit", "--prefill-step-size", "1024"]
    )
    args._original_alias = args.model
    args._prefill_step_size_explicit = True
    assert cli.resolve_serve_acceleration(args).prefill_step_size == 1024


@pytest.mark.parametrize(
    "backend,lane", [("native", "native_mtp"), ("tensorfold", "tensorfold_mtp")]
)
def test_runtime_lane_mtp_backends(backend, lane):
    assert cli.serve_runtime_lane(types.SimpleNamespace(mtp_backend=backend)) == lane


def test_invalid_pflash_exits(monkeypatch, capsys):
    import rapid_mlx.pflash as pflash

    monkeypatch.setattr(
        pflash,
        "resolve_pflash_config",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("invalid pflash")),
    )
    args = build_parser().parse_args(["serve", "model"])
    with pytest.raises(SystemExit) as exc:
        cli.resolve_serve_pflash_config(args, resolve_auto_config=lambda **kw: None)
    assert exc.value.code == 1
    assert "invalid pflash" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--kv-cache-quantization", "--kv-cache-quantization-bits", "8", "--reasoning"],
    ],
)
def test_kv_cache_config_observability_failure(monkeypatch, flags):
    import rapid_mlx.config as config

    def fail():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(config, "get_config", fail)
    result = plan("lfm2.5-1.2b-instruct-4bit", *flags)
    assert result.kv_cache_decision is not None
    if flags:
        assert "reasoning profile" in result.kv_cache_decision.reason


def test_preflight_ddtree_and_missing_dflash(monkeypatch, capsys):
    import rapid_mlx.speculative.dflash.eligibility as eligibility

    for name in (
        "_normalize_speculative_config_or_exit",
        "_require_tensorfold_family_lane_or_exit",
        "_reject_tensorfold_lanes_off_lane_or_exit",
        "_preflight_native_mtp_or_exit",
        "_preflight_companion_dspark_or_exit",
        "_preflight_dflash_mutexes_or_exit",
    ):
        monkeypatch.setattr(cli, name, lambda *a: None)
    seen = []
    monkeypatch.setattr(
        cli, "_preflight_ddtree_or_exit", lambda args: seen.append(args)
    )
    monkeypatch.setattr(eligibility, "have_runtime", lambda: False)
    args = types.SimpleNamespace(enable_ddtree=True, enable_dflash=True)
    with pytest.raises(SystemExit) as exc:
        cli.serve_acceleration_preflights(args)
    assert seen == [args]
    assert exc.value.code == 1
    assert "DFlash speculative decoding" in capsys.readouterr().out


def test_auto_config_import_failure_and_cache(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapid_mlx.model_auto_config", None)
    seen = []

    def pflash(args, *, resolve_auto_config):
        seen.extend(
            [resolve_auto_config(non_fatal=False), resolve_auto_config(non_fatal=False)]
        )
        return None, False

    monkeypatch.setattr(cli, "resolve_serve_pflash_config", pflash)
    plan("lfm2.5-1.2b-instruct-4bit")
    assert seen == [None, None]


@pytest.mark.parametrize("non_fatal", [True, False])
def test_auto_config_detection_failure(monkeypatch, non_fatal):
    import rapid_mlx.model_auto_config as auto

    def fail(*args):
        raise RuntimeError("detection failed")

    monkeypatch.setattr(auto, "detect_model_config", fail)

    def pflash(args, *, resolve_auto_config):
        assert resolve_auto_config(non_fatal=non_fatal) is None
        return None, False

    monkeypatch.setattr(cli, "resolve_serve_pflash_config", pflash)
    if non_fatal:
        # Avoid a second, fatal detection from the KV resolution path.
        monkeypatch.setattr(
            cli, "resolve_serve_kv_cache", lambda *a, **kw: (None, False)
        )
        plan("lfm2.5-1.2b-instruct-4bit")
    else:
        with pytest.raises(RuntimeError, match="detection failed"):
            plan("lfm2.5-1.2b-instruct-4bit")
