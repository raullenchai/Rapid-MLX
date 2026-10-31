# SPDX-License-Identifier: Apache-2.0
"""Execution schema compatibility for acceleration metadata."""

import json
from pathlib import Path

import jsonschema
import pytest

from rapid_mlx.community_bench.run_builder import execution_config

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "proto/model-runtime/v1/execution-config.schema.json"
    ).read_text()
)


def test_language_override_and_legacy_validate():
    legacy = execution_config("text_generation", context_length=4096)
    jsonschema.validate(legacy, SCHEMA)
    language = legacy["task"]["language"]
    language["kv_cache"] = {
        "mode": "quantized",
        "dtype": "q8",
        "bits_x2": 16,
        "group_size": 64,
        "scheme": "affine",
    }
    language["pflash"] = {
        "mode": "auto",
        "threshold": 1024,
        "keep_ratio_millionths": 200000,
    }
    execution = execution_config("text_generation", language=language)
    jsonschema.validate(execution, SCHEMA)
    language["pflash"]["threshold"] = 0
    assert execution["task"]["language"]["pflash"]["threshold"] == 1024
    jsonschema.validate(execution_config("text_generation", language=language), SCHEMA)


def test_scheme_is_for_quantized_cache_only():
    execution = execution_config("text_generation")
    execution["task"]["language"]["kv_cache"]["scheme"] = "affine"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(execution, SCHEMA)


def test_flag_copies_and_combined_overrides():
    import argparse

    from rapid_mlx.cli_parser import BENCHMARK_ACCELERATION_FLAGS, build_parser
    from rapid_mlx.community_bench.acceleration import request_from_args

    parser = build_parser()
    root = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    serve = root.choices["serve"]
    community = root.choices["benchmark"]
    actions = next(
        a for a in community._actions if isinstance(a, argparse._SubParsersAction)
    )
    run = actions.choices["run"]
    for flag in BENCHMARK_ACCELERATION_FLAGS:
        a, b = serve._option_string_actions[flag], run._option_string_actions[flag]
        assert (a.dest, a.type, a.choices) == (b.dest, b.type, b.choices)
        assert a is not b and b.default == argparse.SUPPRESS
    args = parser.parse_args(
        [
            "benchmark",
            "run",
            "model",
            "--no-spec-decode",
            "--kv-cache-quantization",
            "--kv-cache-quantization-bits",
            "8",
            "--prefill-step-size",
            "1024",
        ]
    )
    request = request_from_args(args)
    assert request.overrides == {
        "no_spec_decode": True,
        "kv_cache_quantization": True,
        "kv_cache_quantization_bits": 8,
        "prefill_step_size": 1024,
    }


def test_request_validation():
    from rapid_mlx.community_bench.acceleration import (
        AccelerationConfigError,
        AccelerationRequest,
        validate_request,
    )

    validate_request(
        AccelerationRequest(baseline=True), "text_generation", is_fixed_runtime=False
    )
    flag = AccelerationRequest(
        overrides={"no_spec_decode": True}, flags=("--no-spec-decode",)
    )
    with pytest.raises(AccelerationConfigError, match="--no-spec-decode"):
        validate_request(
            AccelerationRequest(True, flag.overrides, flag.flags),
            "text_generation",
            is_fixed_runtime=False,
        )
    for task in ("image_generation", "video_generation"):
        with pytest.raises(AccelerationConfigError, match="text-generation"):
            validate_request(flag, task, is_fixed_runtime=False)
    with pytest.raises(AccelerationConfigError, match="qualified runtime"):
        validate_request(flag, "text_generation", is_fixed_runtime=True)


def test_effective_metadata_and_summary():
    from types import SimpleNamespace

    from rapid_mlx.community_bench.acceleration import (
        AccelerationConfigError,
        acceleration_summary,
        effective_language,
    )

    config = SimpleNamespace(
        mtp_max_k=3,
        mtp_sidecar="draft",
        kv_cache_turboquant=False,
        kv_cache_quantization=False,
        enable_prefix_cache=False,
        prefill_step_size=2048,
        pflash_config=None,
    )

    def language(method=None, digest=None, live=None):
        result = effective_language(
            config,
            runtime_method=method,
            draft_identity_digest=digest,
            context_length=4096,
            live_kv_quant=live,
        )
        jsonschema.validate(
            execution_config("text_generation", language=result), SCHEMA
        )
        return result

    with pytest.raises(AccelerationConfigError, match="digest"):
        language("mtp")
    embedded = effective_language(
        config,
        runtime_method="mtp",
        draft_identity_digest=None,
        mtp_head_in_target_repo=True,
        context_length=4096,
        live_kv_quant=None,
    )
    assert embedded["speculative_decoding"] == {
        "method": "mtp",
        "max_draft_tokens": 3,
        "assistant_source": "embedded_head",
    }
    assert acceleration_summary(embedded) == "MTP k=3 · prefill chunk 2048"
    jsonschema.validate(execution_config("text_generation", language=embedded), SCHEMA)
    result = language("mtp", "sha256:" + "a" * 64)
    assert result["speculative_decoding"]["assistant_source"] == "external_model"
    assert language()["speculative_decoding"] == {"method": "none"}
    for method in ("dspark", "suffix", ""):
        with pytest.raises(
            AccelerationConfigError, match="benchmark result cannot record"
        ):
            language(method)
    assert acceleration_summary(language()) == "no accelerations · prefill chunk 2048"
    config.kv_cache_quantization = True
    config.kv_cache_quantization_bits = 8
    config.kv_cache_quantization_group_size = 64
    assert language()["kv_cache"] == {"mode": "unknown", "dtype": "unknown"}
    assert language(live=(64, 8))["kv_cache"] == {
        "mode": "quantized",
        "dtype": "q8",
        "bits_x2": 16,
        "group_size": 64,
        "scheme": "affine",
    }
    config.pflash_config = SimpleNamespace(mode="auto", threshold=1024, keep_ratio=0.2)
    assert (
        acceleration_summary(language("mtp", "sha256:" + "a" * 64, (64, 8)))
        == "MTP k=3 (external draft) · KV q8 (affine) · prefill chunk 2048 · PFlash auto"
    )
    config.kv_cache_turboquant = True
    config.kv_cache_turboquant_bits = 4
    config.kv_cache_turboquant_group_size = 32
    config.kv_cache_turboquant_mode = "k8v4"
    assert language()["kv_cache"] == {"mode": "unknown", "dtype": "unknown"}
    config.mtp_sidecar = None
    assert (
        language("mtp")["speculative_decoding"]["assistant_source"] == "embedded_head"
    )


@pytest.fixture
def acceleration_probes(monkeypatch):
    import sys
    import types

    import numpy as np

    from rapid_mlx import cli

    existing = sys.modules.get("rapid_mlx.turboquant")
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array, core.float16 = np.array, np.float16
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    import rapid_mlx.api.utils as utils
    import rapid_mlx.model_auto_config as auto
    import rapid_mlx.spec_decode.mtp as mtp
    import rapid_mlx.speculative.dflash.eligibility as dflash

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
    monkeypatch.setattr(dflash, "have_runtime", lambda: True)
    yield
    if existing is None:
        sys.modules.pop("rapid_mlx.turboquant", None)


@pytest.mark.parametrize(
    "alias,flags",
    [
        ("qwen3.6-27b-4bit", []),
        ("qwen3.6-27b-4bit", ["--pflash", "always"]),
        ("lfm2.5-1.2b-instruct-4bit", []),
        (
            "lfm2.5-1.2b-instruct-4bit",
            [
                "--no-spec-decode",
                "--kv-cache-quantization",
                "--kv-cache-quantization-bits",
                "8",
                "--prefill-step-size",
                "1024",
            ],
        ),
        ("qwen3.8-flash-next-tensorfold", []),
        ("qwen3.8-27b-tensorfold", []),
        ("qwen3.5-27b-8bit", ["--speculative-config", '{"method":"dflash"}']),
    ],
)
def test_serve_parity(acceleration_probes, alias, flags):
    from rapid_mlx import cli
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench.acceleration import (
        AccelerationConfigError,
        build_serve_args,
        request_from_args,
        resolve_benchmark_acceleration,
    )

    parser = build_parser()
    request = request_from_args(parser.parse_args(["benchmark", "run", alias, *flags]))
    args = build_serve_args(alias, request)
    direct = parser.parse_args(["serve", alias, *flags])
    direct.model = args.model
    if hasattr(args, "_original_alias"):
        direct._original_alias = args._original_alias
    direct._prefill_step_size_explicit = "--prefill-step-size" in flags
    expected = cli.resolve_serve_acceleration(direct)
    if expected.lane == "ddtree":
        with pytest.raises(AccelerationConfigError, match="--baseline"):
            resolve_benchmark_acceleration(alias, request)
    else:
        actual, _ = resolve_benchmark_acceleration(alias, request)
        assert (actual.lane, actual.scheduler_kwargs) == (
            expected.lane,
            expected.scheduler_kwargs,
        )
        if "--prefill-step-size" in flags:
            assert actual.scheduler_kwargs["spec_decode"] == "none"
            assert actual.scheduler_kwargs["kv_cache_quantization_bits"] == 8
            assert actual.scheduler_kwargs["prefill_step_size"] == 1024


def test_serve_error_preserved(acceleration_probes):
    from rapid_mlx.community_bench.acceleration import (
        AccelerationConfigError,
        AccelerationRequest,
        resolve_benchmark_acceleration,
    )

    request = AccelerationRequest(
        overrides={"kv_cache_turboquant": "v4", "kv_cache_quantization": True}
    )
    with pytest.raises(AccelerationConfigError, match="mutually exclusive"):
        resolve_benchmark_acceleration("lfm2.5-1.2b-instruct-4bit", request)


def _fake_live_serving_scheduler(
    monkeypatch, tmp_path, runtime_method, *, quantized=False
):
    import contextvars
    import sys
    from types import SimpleNamespace as Namespace

    from rapid_mlx.community_bench import local_runner as lr
    from rapid_mlx.community_bench import runner

    revision = "a" * 40
    snapshot = tmp_path / "models--org--draft" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    configs = []

    def config(**kwargs):
        defaults = dict(
            mtp_sidecar=str(snapshot),
            mtp_max_k=3,
            kv_cache_turboquant=False,
            kv_cache_quantization=False,
            enable_prefix_cache=False,
            prefill_step_size=2048,
            pflash_config=None,
        )
        if quantized:
            defaults.update(
                kv_cache_quantization=True,
                kv_cache_quantization_bits=8,
                kv_cache_quantization_group_size=64,
                prefill_step_size=1024,
                pflash_config=Namespace(mode="always", threshold=1024, keep_ratio=0.2),
            )
        defaults.update(kwargs)
        obj = Namespace(**defaults)
        configs.append(obj)
        return obj

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.engine.batched",
        Namespace(
            BatchedEngine=object,
            _admission_engine_context=contextvars.ContextVar("engine", default=None),
            _admission_token_context=contextvars.ContextVar("token", default=None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.engine_core",
        Namespace(
            AsyncEngineCore=object,
            EngineConfig=lambda **kw: None,
            _init_mlx_step_thread=lambda: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "rapid_mlx.scheduler", Namespace(SchedulerConfig=config)
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.service.helpers",
        Namespace(get_model_max_context=lambda engine: 4096),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.utils.tokenizer",
        Namespace(load_model_with_fallback=lambda *a, **k: None),
    )
    monkeypatch.setattr(lr, "_is_deepseek_v41_benchmark_target", lambda repo: False)
    monkeypatch.setattr(lr, "_uses_serving_benchmark_engine", lambda target: False)
    monkeypatch.setattr(lr, "build_serve_args", lambda *a: Namespace(model="org/model"))
    monkeypatch.setattr(lr, "_ensure_model_downloaded", lambda model: None)
    monkeypatch.setattr(lr, "preflight_benchmark_acceleration", lambda *a: None)
    monkeypatch.setattr(
        lr,
        "resolve_benchmark_acceleration",
        lambda *a: (
            Namespace(
                lane="engine", scheduler_kwargs={"spec_decode": "mtp", "mtp_max_k": 3}
            ),
            Namespace(model="org/model"),
        ),
    )
    monkeypatch.setattr(
        "rapid_mlx.community_bench.workspace.benchmark_runtime_readiness",
        lambda *a: {"status": "available"},
    )
    engine = Namespace(tokenizer=object())

    async def stop():
        pass

    engine.stop = stop

    async def load(model, scheduler_config, args):
        assert model == "org/model"
        assert scheduler_config.spec_decode == "mtp"
        assert scheduler_config.mtp_max_k == 3
        assert scheduler_config.enable_prefix_cache is False
        engine._test_config = scheduler_config
        return engine

    monkeypatch.setattr(lr, "_load_serving_engine", load)
    monkeypatch.setattr(
        lr,
        "_engine_scheduler",
        lambda engine: Namespace(
            config=configs[0],
            spec_decode_runtime_method=runtime_method,
            _live_kv_quant=(64, 8) if quantized else None,
        ),
    )
    round_result = Namespace(
        decode_tps=100, ttft_ms=10, prompt_tokens=None, output_tokens=None
    )

    async def bench(*args, **kwargs):
        return Namespace(
            short=Namespace(rounds_raw=[round_result] * 5),
            long=Namespace(rounds_raw=[round_result] * 5),
            peak_ram_mb=1,
        )

    monkeypatch.setattr(runner, "run_standardized_bench", bench)
    return lr


@pytest.mark.parametrize("runtime_method", ["mtp", None])
def test_text_measurements_uses_live_serving_scheduler(
    monkeypatch, tmp_path, runtime_method
):
    import asyncio

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, runtime_method)
    capture = {}
    token = lr._LOADED_IDENTITY.set(capture)
    try:
        _, _, identity, language = asyncio.run(lr._text_measurements("org/model"))
    finally:
        lr._LOADED_IDENTITY.reset(token)
    assert capture["identity"] == identity
    spec = language["speculative_decoding"]
    assert spec["method"] == (runtime_method or "none")
    if runtime_method:
        assert spec["assistant_source"] == "external_model"
        assert spec["draft_model_identity_digest"].startswith("sha256:")
        assert identity["components"][1]["source"]["repo_id"] == "org/draft"


@pytest.mark.parametrize("live", [True, False])
def test_run_local_archives_quantized_live_scheduler(monkeypatch, tmp_path, live):
    from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator
    from rapid_mlx.community_bench.hardware import Hardware, Software
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, None, quantized=True)
    if not live:
        monkeypatch.setattr(
            lr,
            "_engine_scheduler",
            lambda engine: __import__("types").SimpleNamespace(
                config=engine._test_config,
                spec_decode_runtime_method=None,
                _live_kv_quant=None,
            ),
        )
    monkeypatch.setattr(
        lr,
        "plan_for_alias",
        lambda alias: {
            "model": {
                "alias": alias,
                "repo_id": "org/model",
                "task_type": "text_generation",
            }
        },
    )
    monkeypatch.setattr(
        lr,
        "collect",
        lambda: (
            Hardware("Apple M4 Pro", 24, 12, 16),
            Software("15.6", "0.13.2", "0.32.1", "3.12.1"),
        ),
    )
    archive = LocalRunArchive(tmp_path / "archive")
    run = lr.run_local("org/model", archive=archive)
    archived = archive.get(run["run_id"])
    BenchmarkRunValidator().validate(archived)
    assert archived["outcome"]["status"] == "completed"
    language = archived["execution"]["task"]["language"]
    if live:
        assert language["kv_cache"]["mode"] == "quantized"
        assert language["kv_cache"]["bits_x2"] == 16
        assert language["kv_cache"]["group_size"] == 64
    else:
        assert language["kv_cache"] == {"mode": "unknown", "dtype": "unknown"}
    assert language["prefill_chunk_size"] == 1024
    assert language["pflash"]["mode"] == "always"


def test_unrecordable_live_method_is_not_archived(monkeypatch, tmp_path):
    from rapid_mlx.community_bench.hardware import Hardware, Software
    from rapid_mlx.community_bench.workspace import LocalRunArchive

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, "dspark")
    monkeypatch.setattr(
        lr,
        "plan_for_alias",
        lambda alias: {
            "model": {
                "alias": alias,
                "repo_id": "org/model",
                "task_type": "text_generation",
            }
        },
    )
    monkeypatch.setattr(
        lr,
        "collect",
        lambda: (
            Hardware("Apple M4 Pro", 24, 12, 16),
            Software("15.6", "0.13.2", "0.32.1", "3.12.1"),
        ),
    )
    archive = LocalRunArchive(tmp_path / "archive")
    with pytest.raises(
        lr.LocalBenchmarkError, match="the engine installed dspark"
    ) as exc:
        lr.run_local("org/model", archive=archive)
    assert exc.value.saved is False
    assert archive.list() == []


def test_baseline_serving_engine_records_live_scheduler(monkeypatch, tmp_path):
    import asyncio
    import sys
    from types import SimpleNamespace

    from rapid_mlx.community_bench.acceleration import AccelerationRequest

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, None)
    monkeypatch.setattr(lr, "_uses_serving_benchmark_engine", lambda target: True)

    class Engine:
        def __init__(self, target, **kwargs):
            self.tokenizer = object()
            self.scheduler = SimpleNamespace(
                config=SimpleNamespace(
                    enable_prefix_cache=False,
                    prefill_step_size=4096,
                    pflash_config=None,
                ),
                projected_memory_max_context=lambda: 4096,
            )

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(
        sys.modules["rapid_mlx.engine.batched"], "BatchedEngine", Engine
    )
    monkeypatch.setattr(lr, "_engine_scheduler", lambda engine: engine.scheduler)
    _, _, _, language = asyncio.run(
        lr._text_measurements(
            "org/model", acceleration=AccelerationRequest(baseline=True)
        )
    )
    assert language["prefill_chunk_size"] == 4096
    assert language["speculative_decoding"] == {"method": "none"}
    assert language["kv_cache"] == {"mode": "unknown", "dtype": "unknown"}


def test_configuration_refusal_is_not_archived(monkeypatch):
    from rapid_mlx.community_bench import local_runner as lr
    from rapid_mlx.community_bench.acceleration import (
        AccelerationRequest,
    )

    monkeypatch.setattr(
        lr,
        "plan_for_alias",
        lambda alias: {
            "model": {"repo_id": "org/model", "task_type": "text_generation"}
        },
    )
    monkeypatch.setattr(lr, "_is_deepseek_v41_benchmark_target", lambda repo: True)
    with pytest.raises(lr.LocalBenchmarkError) as exc:
        lr.run_local("x", acceleration=AccelerationRequest(baseline=True))
    assert exc.value.saved is False


@pytest.mark.parametrize(
    "extra", ["disable_auto_k", "continuous_batching", "allow_dynamic_membership"]
)
def test_unrecorded_speculative_config_refused_before_measurement(monkeypatch, extra):
    from rapid_mlx.community_bench import local_runner as lr
    from rapid_mlx.community_bench.acceleration import AccelerationRequest

    monkeypatch.setattr(
        lr,
        "plan_for_alias",
        lambda alias: {
            "model": {"repo_id": "org/model", "task_type": "text_generation"}
        },
    )
    monkeypatch.setattr(lr, "_is_deepseek_v41_benchmark_target", lambda repo: False)
    monkeypatch.setattr(
        lr,
        "_text_measurements",
        lambda *args, **kwargs: pytest.fail("measurement started"),
    )
    config = {"method": "mtp", "num_speculative_tokens": 3, extra: True}
    with pytest.raises(lr.LocalBenchmarkError, match=extra) as exc:
        lr.run_local(
            "x",
            acceleration=AccelerationRequest(
                overrides={"speculative_config": json.dumps(config)}
            ),
        )
    assert exc.value.saved is False


@pytest.mark.parametrize(
    "config", ['{"method":"mtp","num_speculative_tokens":3}', "invalid JSON", "[]"]
)
def test_speculative_config_accepted_or_left_for_serve(config):
    from rapid_mlx.community_bench.acceleration import (
        AccelerationRequest,
        validate_request,
    )

    validate_request(
        AccelerationRequest(overrides={"speculative_config": config}),
        "text_generation",
        is_fixed_runtime=False,
    )


def test_cli_acceleration_summary_and_json(monkeypatch, capsys):
    from types import SimpleNamespace

    from rapid_mlx.community_bench import cli
    from rapid_mlx.community_bench.acceleration import AccelerationRequest

    language = execution_config("text_generation", context_length=4096)["task"][
        "language"
    ]
    run = {
        "run_id": "run-1",
        "workload": {"task_type": "text_generation"},
        "execution": {"task": {"language": language}},
        "measurements": [],
    }
    monkeypatch.setattr(cli.LocalRunArchive, "default", lambda: object())
    monkeypatch.setattr(cli, "run_local", lambda alias, **kwargs: run)
    monkeypatch.setattr(
        cli,
        "plan_for_alias",
        lambda alias: {
            "model": {"task_type": "text_generation", "repo_id": "test/model"}
        },
    )
    monkeypatch.setattr(cli, "preflight_benchmark_acceleration", lambda *args: None)
    monkeypatch.setattr(cli, "request_from_args", lambda args: AccelerationRequest())
    monkeypatch.setattr(cli, "summarize_measurements", lambda run: [])
    args = SimpleNamespace(benchmark_action="run", benchmark_model="x", json=False)
    assert cli.benchmark_command(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("Acceleration: no accelerations")
    args.json = True
    assert cli.benchmark_command(args) == 0
    assert json.loads(capsys.readouterr().out) == run


def test_resolution_error_is_never_archived(monkeypatch):
    from rapid_mlx.community_bench import local_runner as lr
    from rapid_mlx.community_bench.acceleration import AccelerationConfigError

    monkeypatch.setattr(
        lr,
        "plan_for_alias",
        lambda alias: {
            "model": {"repo_id": "org/model", "task_type": "text_generation"},
            "workload": {"cases": []},
        },
    )
    monkeypatch.setattr(lr, "_is_deepseek_v41_benchmark_target", lambda repo: False)
    monkeypatch.setattr(lr, "resolve_provenance", lambda: {})
    monkeypatch.setattr(lr, "_announce_plan", lambda *a: None)
    monkeypatch.setattr(lr, "collect", lambda: ({}, {}))
    monkeypatch.setattr(lr, "run_conditions", lambda: {})

    async def refuse(*args, **kwargs):
        raise AccelerationConfigError("invalid acceleration")

    monkeypatch.setattr(lr, "_text_measurements", refuse)

    class Archive:
        def save(self, *args):
            pytest.fail("configuration refusal must not be archived")

    with pytest.raises(lr.LocalBenchmarkError, match="invalid acceleration") as exc:
        lr.run_local("x", archive=Archive())
    assert exc.value.saved is False


def test_removed_benchmark_flags_rejected():
    from rapid_mlx.cli_parser import build_parser

    removed = [
        "--mtp-disable-auto-k",
        "--kv-cache-min-quantize-tokens",
        "--kv-cache-turboquant",
        "--kv-cache-turboquant-bits",
        "--kv-cache-turboquant-group-size",
        "--pflash-min-keep-tokens",
        "--pflash-sink-tokens",
        "--pflash-tail-tokens",
        "--pflash-block-size",
        "--pflash-query-window",
        "--pflash-stride-blocks",
        "--pflash-include-tools",
        "--pflash-threshold-basis",
    ]
    parser = build_parser()
    for flag in removed:
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark", "run", "model", flag, "1"])


def test_sidecar_snapshot_subfolders_have_distinct_identities(tmp_path):
    from rapid_mlx.community_bench.local_runner import (
        _sidecar_identity,
        _with_sidecar_identity,
    )
    from rapid_mlx.community_bench.run_builder import unresolved_model_identity

    snapshot = tmp_path / "models--org--draft" / "snapshots" / ("a" * 40)
    for name in ("one/nested", "two/nested"):
        (snapshot / name).mkdir(parents=True)
    first = _sidecar_identity(str(snapshot / "one/nested"))
    second = _sidecar_identity(str(snapshot / "two/nested"))
    root = _sidecar_identity(str(snapshot))
    assert first[:3] == ("org/draft", str(snapshot), "one/nested")
    assert second[:3] == ("org/draft", str(snapshot), "two/nested")
    assert first[3] != second[3]
    assert root[:3] == ("org/draft", str(snapshot), None)
    primary = unresolved_model_identity("org/primary", "text_generation")
    for repo, path, subfolder, _ in (first, second, root):
        combined = _with_sidecar_identity(primary, repo, path, subfolder)
        sidecar = next(
            c for c in combined["components"] if c["component_id"] == "sidecar"
        )
        assert sidecar["source"].get("subfolder") == subfolder


@pytest.mark.parametrize(
    "repo,revision,expected",
    [
        ("ORG/model", "a" * 40, True),
        ("org/other", "a" * 40, False),
        ("org/model", "b" * 40, False),
        ("org/model", None, False),
    ],
)
def test_sidecar_in_target_repo(monkeypatch, repo, revision, expected):
    from rapid_mlx.community_bench import local_runner as lr

    target = {
        "components": [
            {
                "role": "primary",
                "source": {
                    "repo_id": "org/model",
                    "resolved_revision": "a" * 40,
                },
            }
        ]
    }
    monkeypatch.setattr(
        lr,
        "unresolved_model_identity",
        lambda *a, **kw: {"components": [{"source": {"resolved_revision": revision}}]},
    )
    assert lr._sidecar_in_target_repo(target, repo, None) is expected


@pytest.mark.parametrize("same_repo", [True, False])
def test_serving_sidecar_repository_classification(monkeypatch, tmp_path, same_repo):
    import asyncio

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, "mtp")
    original_identity = lr.unresolved_model_identity

    def identity(repo, *args, **kwargs):
        result = original_identity(repo, *args, **kwargs)
        if repo.lower() == "org/model":
            result["components"][0]["source"]["resolved_revision"] = "a" * 40
        return result

    monkeypatch.setattr(lr, "unresolved_model_identity", identity)
    if same_repo:
        snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
        (snapshot / "mtp").mkdir(parents=True)
        (snapshot / "mtp" / "config.json").write_text("{}")
        original_scheduler = lr._engine_scheduler

        def scheduler(engine):
            result = original_scheduler(engine)
            result.config.mtp_sidecar = str(snapshot / "mtp")
            return result

        monkeypatch.setattr(lr, "_engine_scheduler", scheduler)
    _, _, loaded, language = asyncio.run(lr._text_measurements("org/model"))
    spec = language["speculative_decoding"]
    assert spec["assistant_source"] == (
        "embedded_head" if same_repo else "external_model"
    )
    assert ("draft_model_identity_digest" in spec) is not same_repo
    sidecar = next(c for c in loaded["components"] if c["component_id"] == "sidecar")
    assert sidecar["source"]["repo_id"] == ("org/model" if same_repo else "org/draft")
    if same_repo:
        assert sidecar["source"]["subfolder"] == "mtp"


def test_suffix_refused_and_prefix_summary():
    from rapid_mlx.community_bench.acceleration import (
        AccelerationConfigError,
        _refuse_unsupported,
        acceleration_summary,
    )

    with pytest.raises(
        AccelerationConfigError, match="Suffix decoding cannot be benchmarked"
    ):
        _refuse_unsupported("model", "engine", True)
    language = {
        "speculative_decoding": {"method": "none"},
        "kv_cache": {"mode": "none"},
        "prefix_cache_enabled": True,
        "prefill_chunk_size": 2048,
    }
    assert acceleration_summary(language) == "prefix cache · prefill chunk 2048"


@pytest.mark.parametrize("fails", [False, True])
def test_load_serving_engine_forwarding_and_cleanup(monkeypatch, fails):
    import asyncio
    import types

    import rapid_mlx
    from rapid_mlx.community_bench import local_runner as lr

    calls = []

    class Engine:
        async def start(self):
            calls.append("start")
            if fails:
                raise RuntimeError("start failed")

        async def stop(self):
            calls.append("stop")

    engine = Engine()

    def load(model, **kwargs):
        calls.append((model, kwargs))

    server = types.SimpleNamespace(load_model=load, _engine=engine)
    monkeypatch.setattr(rapid_mlx, "server", server, raising=False)
    args = types.SimpleNamespace(
        mllm=True,
        no_mllm=False,
        force_hybrid=True,
        no_hybrid=False,
        force_spec_decode=True,
        no_spec_decode=False,
        force_openai_harmony_streaming=True,
        no_openai_harmony_streaming=False,
    )
    if fails:
        with pytest.raises(RuntimeError, match="start failed"):
            asyncio.run(lr._load_serving_engine("model", "scheduler", args))
        assert calls[-1] == "stop"
    else:
        assert (
            asyncio.run(lr._load_serving_engine("model", "scheduler", args)) is engine
        )
    assert calls[0] == (
        "model",
        {
            "scheduler_config": "scheduler",
            "force_mllm": True,
            "force_text": False,
            "force_hybrid": True,
            "no_hybrid": False,
            "force_spec_decode": True,
            "no_spec_decode": False,
            "force_openai_harmony_streaming": True,
            "no_openai_harmony_streaming": False,
        },
    )
    assert calls[1] == "start"


def test_download_wrapper_delegates(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.community_bench import local_runner as lr

    monkeypatch.setattr(
        cli, "_ensure_model_downloaded", lambda model: ("downloaded", model)
    )
    assert lr._ensure_model_downloaded("org/model") == ("downloaded", "org/model")


@pytest.mark.parametrize(
    "path", ["/cache/bad/snapshots/rev", "/cache/models--org--draft/snapshots"]
)
def test_sidecar_invalid_snapshot(path):
    from rapid_mlx.community_bench import local_runner as lr

    with pytest.raises(lr.AccelerationConfigError, match="Cannot identify"):
        lr._sidecar_identity(path)


def test_sidecar_bare_repo_revision(monkeypatch):
    from rapid_mlx.community_bench import local_runner as lr
    from rapid_mlx.community_bench.run_builder import unresolved_model_identity

    def identity(repo, *args, **kwargs):
        result = unresolved_model_identity(repo, *args, **kwargs)
        result["components"][0]["source"]["resolved_revision"] = revision
        return result

    monkeypatch.setattr(lr, "unresolved_model_identity", identity)
    revision = None
    with pytest.raises(
        lr.AccelerationConfigError,
        match="Cannot resolve speculative sidecar snapshot revision",
    ):
        lr._sidecar_identity("org/draft")
    revision = "a" * 40
    repo, snapshot, subfolder, digest = lr._sidecar_identity("org/draft")
    assert (repo, snapshot, subfolder) == ("org/draft", None, None)
    assert digest.startswith("sha256:")


def test_text_measurements_missing_live_scheduler(monkeypatch, tmp_path):
    import asyncio

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, None)
    monkeypatch.setattr(lr, "_engine_scheduler", lambda engine: None)
    with pytest.raises(RuntimeError, match="Serving engine has no live scheduler"):
        asyncio.run(lr._text_measurements("org/model"))


def test_dedicated_runtime_schema_is_additive():
    root = Path(__file__).resolve().parents[1]
    assert (
        root / "proto/model-runtime/v1/execution-config.schema.json"
    ).read_bytes() == (
        root / "rapid_mlx/catalog/schemas/execution-config.schema.json"
    ).read_bytes()
    execution = execution_config("text_generation")
    jsonschema.validate(execution, SCHEMA)
    language = execution["task"]["language"]
    language["runtime"] = {"kind": "native_mtp"}
    jsonschema.validate(execution, SCHEMA)
    language["runtime"]["unexpected"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(execution, SCHEMA)


def test_model_compute_dtype(monkeypatch):
    import sys
    from types import SimpleNamespace

    mx = SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32", uint32="u32")
    mx.zeros = lambda shape, dtype: SimpleNamespace(size=shape[0], dtype=dtype)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(
        sys.modules,
        "mlx.utils",
        SimpleNamespace(tree_flatten=lambda p: list(p.items())),
    )

    from rapid_mlx.community_bench.acceleration import model_compute_dtype

    def model(*arrays):
        return SimpleNamespace(
            parameters=lambda: {str(i): a for i, a in enumerate(arrays)}
        )

    bf = mx.zeros((100,), dtype=mx.bfloat16)
    fp = mx.zeros((100,), dtype=mx.float16)
    assert model_compute_dtype(model(bf)) == "bfloat16"
    assert model_compute_dtype(model(fp)) == "float16"
    assert model_compute_dtype(model(bf, fp)) == "mixed"
    assert (
        model_compute_dtype(model(bf, mx.zeros((200,), dtype=mx.uint32))) == "bfloat16"
    )
    assert model_compute_dtype(model(mx.zeros((2,), dtype=mx.uint32))) == "unknown"
    assert model_compute_dtype(object()) == "unknown"
    assert (
        model_compute_dtype(model(bf, mx.zeros((1,), dtype=mx.float32))) == "bfloat16"
    )


def test_effective_language_compute_dtype_and_digest():
    from types import SimpleNamespace

    from rapid_mlx.community_bench.acceleration import effective_language

    config = SimpleNamespace(
        enable_prefix_cache=False, prefill_step_size=2048, pflash_config=None
    )

    def language(dtype, live=None):
        return effective_language(
            config,
            runtime_method=None,
            draft_identity_digest=None,
            context_length=4096,
            live_kv_quant=live,
            compute_dtype=dtype,
        )

    assert language("bfloat16")["kv_cache"] == {
        "mode": "full_precision",
        "dtype": "bfloat16",
    }
    assert language("mixed")["kv_cache"] == {"mode": "unknown", "dtype": "unknown"}
    assert language("float16", (64, 8))["kv_cache"]["mode"] == "quantized"
    a = execution_config("text_generation")
    b = execution_config("text_generation", compute_dtype="bfloat16")
    assert b["resources"]["compute_dtype"] == "bfloat16"
    assert a["config_digest"] != b["config_digest"]


@pytest.mark.parametrize("fastpath", [False, True])
@pytest.mark.parametrize("baseline", [False, True])
def test_mllm_serving_engine_run_records_mllm_lane_config(
    monkeypatch, tmp_path, fastpath, baseline
):
    import asyncio
    import subprocess
    import sys
    from types import SimpleNamespace

    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True
    )
    if probe.returncode:
        pytest.skip("MLX Metal device unavailable")
    import mlx.core as mx
    import mlx.nn as nn

    from rapid_mlx.community_bench.acceleration import AccelerationRequest
    from rapid_mlx.mllm_scheduler import MLLMSchedulerConfig

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, None)
    config = MLLMSchedulerConfig(
        prefill_step_size=4096,
        enable_prefix_cache=False,
        max_num_seqs=1,
        prefill_batch_size=1,
        completion_batch_size=1,
    )
    model = nn.Linear(2, 2)
    model.set_dtype(mx.bfloat16)

    async def load(*args, **kwargs):
        async def stop():
            pass

        return SimpleNamespace(
            tokenizer=object(),
            _is_mllm=True,
            _engine=object() if fastpath else None,
            _mllm_dense_singleton_fastpath=fastpath,
            _mllm_scheduler=SimpleNamespace(config=config),
            _model=model,
            stop=stop,
        )

    monkeypatch.setattr(lr, "_load_serving_engine", load)
    if baseline:
        monkeypatch.setattr(lr, "_uses_serving_benchmark_engine", lambda target: True)

        class FakeBatched:
            def __init__(self, *args, **kwargs):
                pass

            async def start(self):
                self.__dict__.update((await load()).__dict__)

        monkeypatch.setattr(
            sys.modules["rapid_mlx.engine.batched"], "BatchedEngine", FakeBatched
        )
    capture = {}
    token = lr._LOADED_IDENTITY.set(capture)
    try:
        _, _, _, language = asyncio.run(
            lr._text_measurements(
                "org/model", acceleration=AccelerationRequest(baseline=baseline)
            )
        )
    finally:
        lr._LOADED_IDENTITY.reset(token)
    assert language["prefill_chunk_size"] == 2048
    assert language["speculative_decoding"] == {"method": "none"}
    assert language["kv_cache"] == {"mode": "full_precision", "dtype": "bfloat16"}
    assert capture["compute_dtype"] == "bfloat16"


@pytest.mark.parametrize(
    "step_size, expected", [(8192, 2048), (4096, 2048), (1024, 1024)]
)
def test_mllm_language_records_effective_prefill_chunk(step_size, expected):
    from types import SimpleNamespace

    from rapid_mlx.community_bench.acceleration import mllm_language

    config = SimpleNamespace(prefill_step_size=step_size, enable_prefix_cache=False)
    language = mllm_language(config, context_length=4096, compute_dtype="bfloat16")
    assert language["prefill_chunk_size"] == expected


@pytest.mark.parametrize("fastpath", [False, True])
def test_mllm_lane_routing_with_headless_fake(monkeypatch, tmp_path, fastpath):
    import asyncio
    from types import SimpleNamespace

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, None)
    config = SimpleNamespace(prefill_step_size=4096, enable_prefix_cache=False)
    model = object()

    async def stop():
        pass

    async def load(*args):
        return SimpleNamespace(
            tokenizer=object(),
            _is_mllm=True,
            _engine=object() if fastpath else None,
            _mllm_dense_singleton_fastpath=fastpath,
            _mllm_scheduler=SimpleNamespace(config=config),
            _model=model,
            stop=stop,
        )

    monkeypatch.setattr(lr, "_load_serving_engine", load)
    monkeypatch.setattr(lr, "_engine_scheduler", lambda engine: None)
    monkeypatch.setattr(
        lr,
        "model_compute_dtype",
        lambda loaded: "bfloat16" if loaded is model else "unknown",
    )
    capture = {}
    token = lr._LOADED_IDENTITY.set(capture)
    try:
        _, _, _, language = asyncio.run(lr._text_measurements("org/model"))
    finally:
        lr._LOADED_IDENTITY.reset(token)
    assert language["prefill_chunk_size"] == 2048
    assert language["speculative_decoding"] == {"method": "none"}
    assert language["kv_cache"] == {"mode": "full_precision", "dtype": "bfloat16"}
    assert capture["compute_dtype"] == "bfloat16"


def test_cli_default_run_skips_early_acceleration_preflight(monkeypatch, capsys):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli

    def unexpected_preflight(*args, **kwargs):
        pytest.fail("default run must not perform early acceleration preflight")

    def runner(*args, **kwargs):
        raise cli.LocalBenchmarkError("runner reached", None, saved=False)

    monkeypatch.setattr(cli, "preflight_benchmark_acceleration", unexpected_preflight)
    monkeypatch.setattr(cli, "run_local", runner)
    args = build_parser().parse_args(["benchmark", "run", "qwen3.8-27b-4bit"])
    assert cli.benchmark_command(args) == 1
    assert "runner reached" in capsys.readouterr().err


@pytest.mark.parametrize("error_type", ["local", "key", "value"])
def test_cli_plan_failure_defers_to_runner(monkeypatch, capsys, error_type):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli

    errors = {
        "local": cli.LocalBenchmarkError("plan failed", None, saved=False),
        "key": KeyError("plan failed"),
        "value": ValueError("plan failed"),
    }

    def failed_plan(*args, **kwargs):
        raise errors[error_type]

    def runner(*args, **kwargs):
        raise cli.LocalBenchmarkError("runner reached", None, saved=False)

    monkeypatch.setattr(cli, "plan_for_alias", failed_plan)
    monkeypatch.setattr(cli, "run_local", runner)
    args = build_parser().parse_args(
        ["benchmark", "run", "unknown", "--speculative-config", '{"method":"suffix"}']
    )
    assert cli.benchmark_command(args) == 1
    assert "runner reached" in capsys.readouterr().err


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_suffix_refused_before_runner(monkeypatch, capsys, json_output):
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli

    def unexpected_runner(*args, **kwargs):
        raise AssertionError("runner must not be reached")

    monkeypatch.setattr(cli, "run_local", unexpected_runner)
    args = build_parser().parse_args(
        [
            "benchmark",
            "run",
            "qwen3.8-27b-4bit",
            "--speculative-config",
            '{"method":"suffix"}',
        ]
        + (["--json"] if json_output else [])
    )
    assert cli.benchmark_command(args) == 1
    output = capsys.readouterr()
    assert "Suffix decoding cannot be benchmarked" in output.err
    assert "Benchmarking" not in output.out + output.err
    assert "Loading" not in output.out + output.err
    if json_output:
        assert json.loads(output.err)["saved"] is False
    else:
        assert "Benchmark command failed:" in output.err


def test_default_acceleration_uses_pinned_serving_snapshot_identity(
    monkeypatch, tmp_path
):
    import asyncio

    import huggingface_hub
    from huggingface_hub import constants

    from rapid_mlx import model_aliases, model_metadata
    from rapid_mlx.community_bench.acceleration import AccelerationRequest

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, "mtp")
    pinned_revision = "a" * 40
    pinned = tmp_path / "models--org--model" / "snapshots" / pinned_revision
    (pinned / "mtp").mkdir(parents=True)
    (pinned / "config.json").write_text("{}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: pinned_revision)
    monkeypatch.setattr(model_aliases, "resolve_hub_model", lambda name: "org/model")
    monkeypatch.setattr(
        model_metadata,
        "_complete_cached_checkpoint",
        lambda root, snapshot, *a, **kw: snapshot if snapshot.is_dir() else None,
    )
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None
    )
    original_scheduler = lr._engine_scheduler

    def scheduler(engine):
        result = original_scheduler(engine)
        result.config.mtp_sidecar = str(pinned / "mtp")
        return result

    monkeypatch.setattr(lr, "_engine_scheduler", scheduler)
    _, _, identity, language = asyncio.run(
        lr._text_measurements("org/model", acceleration=AccelerationRequest())
    )
    primary = next(c for c in identity["components"] if c["role"] == "primary")
    assert primary["source"]["resolved_revision"] == pinned_revision
    speculative = language["speculative_decoding"]
    assert speculative["assistant_source"] == "embedded_head"
    assert "draft_model_identity_digest" not in speculative


@pytest.mark.parametrize("fallback", [False, True])
def test_pinned_serving_snapshot_identity_beats_main(monkeypatch, tmp_path, fallback):
    import asyncio
    import sys

    import huggingface_hub
    from huggingface_hub import constants

    from rapid_mlx import model_aliases, model_metadata
    from rapid_mlx.community_bench.acceleration import AccelerationRequest

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, "mtp")
    pinned_revision, main_revision = "a" * 40, "b" * 40
    root = tmp_path / "models--org--model"
    pinned = root / "snapshots" / pinned_revision
    main = root / "snapshots" / main_revision
    for snapshot in (pinned, main):
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
    (pinned / "mtp").mkdir()
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(main_revision)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: pinned_revision)
    monkeypatch.setattr(model_aliases, "resolve_hub_model", lambda name: "org/model")
    monkeypatch.setattr(
        model_metadata,
        "_complete_cached_checkpoint",
        lambda root, snapshot, *a, **kw: snapshot if snapshot.is_dir() else None,
    )
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *a, **kw: str(main / "config.json"),
    )
    original_scheduler = lr._engine_scheduler

    def scheduler(engine):
        result = original_scheduler(engine)
        result.config.mtp_sidecar = str(pinned / "mtp")
        return result

    monkeypatch.setattr(lr, "_engine_scheduler", scheduler)

    class Engine:
        def __init__(self, *args, **kwargs):
            self.tokenizer = object()

        async def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(
        sys.modules["rapid_mlx.engine.batched"], "BatchedEngine", Engine
    )
    monkeypatch.setattr(
        lr, "_uses_serving_benchmark_engine", lambda target: not fallback
    )
    if fallback:

        def fail(*args, **kwargs):
            raise ValueError("Model type unsupported_arch not supported.")

        monkeypatch.setattr(
            sys.modules["rapid_mlx.utils.tokenizer"], "load_model_with_fallback", fail
        )
    request = AccelerationRequest(baseline=True)
    _, _, identity, language = asyncio.run(
        lr._text_measurements("org/model", acceleration=request)
    )
    primary = next(c for c in identity["components"] if c["role"] == "primary")
    assert primary["source"]["resolved_revision"] == pinned_revision


def test_text_measurements_prefers_loaded_engine_checkpoint(monkeypatch, tmp_path):
    import asyncio

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, None)
    load = lr._load_serving_engine
    checkpoint = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")

    async def load_with_path(*args):
        engine = await load(*args)
        engine._model_name = str(checkpoint)
        return engine

    monkeypatch.setattr(lr, "_load_serving_engine", load_with_path)
    _, _, identity, _ = asyncio.run(lr._text_measurements("org/model"))
    assert identity["components"][0]["source"]["resolved_revision"] == "a" * 40


def test_pinned_sidecar_identity_without_refs_or_config(monkeypatch, tmp_path):
    from huggingface_hub import constants

    from rapid_mlx import model_aliases
    from rapid_mlx.community_bench.local_runner import _sidecar_identity

    repo = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
    revision = model_aliases.resolve_revision(repo)
    assert revision
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "mtp.safetensors").touch()
    found_repo, path, subfolder, digest = _sidecar_identity(repo)
    assert (found_repo, path, subfolder) == (repo, str(snapshot), None)
    assert revision in path and digest.startswith("sha256:")


@pytest.mark.parametrize("cached", [False, True])
def test_serving_sidecar_identity_before_benchmark(monkeypatch, tmp_path, cached):
    import asyncio

    from huggingface_hub import constants

    from rapid_mlx import model_aliases
    from rapid_mlx.community_bench import runner

    lr = _fake_live_serving_scheduler(monkeypatch, tmp_path, "mtp")
    repo = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
    revision = model_aliases.resolve_revision(repo)
    cache = tmp_path / "cache"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    if cached:
        snapshot = (
            cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "mtp.safetensors").touch()
    original_scheduler = lr._engine_scheduler
    live = None

    def scheduler(engine):
        nonlocal live
        if live is None:
            live = original_scheduler(engine)
            live.config.mtp_sidecar = repo
            live.spec_decode_runtime_method = None
        return live

    monkeypatch.setattr(lr, "_engine_scheduler", scheduler)
    original_bench = runner.run_standardized_bench
    calls = []

    async def bench(*args, **kwargs):
        calls.append(True)
        assert live is not None
        live.spec_decode_runtime_method = "mtp"
        return await original_bench(*args, **kwargs)

    monkeypatch.setattr(runner, "run_standardized_bench", bench)
    if not cached:
        with pytest.raises(
            lr.AccelerationConfigError, match="sidecar snapshot revision"
        ):
            asyncio.run(lr._text_measurements("org/model"))
        assert not calls
    else:
        _, _, identity, _ = asyncio.run(lr._text_measurements("org/model"))
        assert calls and any(
            c["source"].get("repo_id") == repo for c in identity["components"]
        )


@pytest.mark.parametrize(
    "function", ["preflight_benchmark_acceleration", "resolve_benchmark_acceleration"]
)
@pytest.mark.parametrize("explicit", [False, True])
def test_tensorfold_suffix_refused_before_loading(monkeypatch, function, explicit):
    from rapid_mlx import cli
    from rapid_mlx.community_bench import acceleration, local_runner

    def forbidden(*args, **kwargs):
        pytest.fail("runtime download or preparation was invoked")

    monkeypatch.setattr(local_runner, "_ensure_model_downloaded", forbidden)
    monkeypatch.setattr(cli, "prepare_dedicated_runtime", forbidden)
    from rapid_mlx.speculative import tensorfold_families, tensorfold_qwen27

    monkeypatch.setattr(tensorfold_qwen27, "require_runtime", lambda: None)
    monkeypatch.setattr(tensorfold_qwen27, "require_environment", lambda: None)
    monkeypatch.setattr(tensorfold_families, "require_memory", lambda profile: None)
    request = acceleration.AccelerationRequest(
        overrides={"speculative_config": '{"method":"suffix","backend":"tensorfold"}'}
        if explicit
        else {}
    )
    with pytest.raises(acceleration.AccelerationConfigError, match="Suffix decoding"):
        getattr(acceleration, function)("gemma-4-26b-tensorfold", request)
