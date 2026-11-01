# SPDX-License-Identifier: Apache-2.0
"""Contract tests for dedicated serial benchmark runtimes, without MLX."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rapid_mlx.community_bench.acceleration import (
    AccelerationConfigError,
    _refuse_engine_method,
    acceleration_summary,
    dedicated_language,
)
from rapid_mlx.community_bench.local_runner import _DedicatedRuntimeBenchmarkAdapter
from rapid_mlx.speculative.serial_benchmark import TokenChunk


def test_dedicated_metadata_and_summary():
    handle = SimpleNamespace(
        lane="dflash",
        speculative={
            "method": "dflash",
            "max_draft_tokens": 15,
            "algorithm": "dflash2",
        },
    )
    language = dedicated_language(
        handle,
        context_length=8192,
        prefill_chunk_size=2048,
        draft_identity_digest="sha256:" + "a" * 64,
    )
    assert language["runtime"] == {"kind": "dflash", "algorithm": "dflash2"}
    assert "DFlash dflash2 k=15 (dedicated runtime)" in acceleration_summary(language)


@pytest.mark.parametrize(
    ("digest", "suffix"),
    [("sha256:" + "a" * 64, ", external draft"), (None, "")],
)
def test_dedicated_mtp_summary_draft_source(digest, suffix):
    handle = SimpleNamespace(
        lane="native_mtp", speculative={"method": "mtp", "max_draft_tokens": 3}
    )
    language = dedicated_language(
        handle,
        context_length=8192,
        prefill_chunk_size=2048,
        draft_identity_digest=digest,
    )
    assert acceleration_summary(language).startswith(
        f"native MTP k=3 (dedicated runtime{suffix})"
    )


@pytest.mark.parametrize("method", ["dspark", "draft_model"])
def test_engine_methods_refused(method):
    with pytest.raises(AccelerationConfigError, match=method):
        _refuse_engine_method({"spec_decode": method})


def test_adapter_counts_and_closes():
    closed = []

    def stream(prompt, maximum):
        try:
            for i in range(maximum):
                yield TokenChunk(i, len(prompt))
        finally:
            closed.append(True)

    async def run(executor):
        adapter = _DedicatedRuntimeBenchmarkAdapter(
            SimpleNamespace(executor=executor, token_stream=stream)
        )
        sampling = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop=None,
            stop_token_ids=None,
            ignore_eos=True,
            max_tokens=3,
        )
        request = await adapter.add_request([1, 2], sampling)
        outputs = [out async for out in adapter.stream_outputs(request)]
        assert outputs[-1].completion_tokens == 3
        assert outputs[-1].prompt_tokens == 2
        assert outputs[-1].finished

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))
    assert closed == [True]


def test_tensorfold_handle_embedded_head(monkeypatch):
    import sys
    from types import ModuleType

    from rapid_mlx.speculative import serial_benchmark

    events = []
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.clear_cache = lambda: events.append("cache")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(serial_benchmark.gc, "collect", lambda: events.append("gc"))
    from rapid_mlx.speculative.serial_benchmark import tensorfold_runtime_handle

    closed = []

    class Provider:
        def _outputs(self, prompt, **kwargs):
            assert kwargs["prompt_token_ids"] == [1, 2]
            assert kwargs["ignore_eos"]
            yield SimpleNamespace(new_token_ids=[])
            yield SimpleNamespace(new_token_ids=[3, 4])

    backend = SimpleNamespace(
        _app=SimpleNamespace(tokenizer=object()), close=lambda: closed.append(True)
    )
    handle = tensorfold_runtime_handle(
        backend,
        Provider(),
        lane="tensorfold_mtp",
        method="mtp",
        algorithm="mtp",
        draft_repo=None,
        draft_revision=None,
        draft_path=None,
        target_path="/target",
        max_draft_tokens=3,
        lanes=1,
    )
    assert [chunk.token for chunk in handle.token_stream([1, 2], 2)] == [3, 4]
    language = dedicated_language(
        handle, context_length=8192, prefill_chunk_size=None, draft_identity_digest=None
    )
    assert language["runtime"] == {"kind": "tensorfold_mtp", "lanes": 1}
    assert language["speculative_decoding"]["assistant_source"] == "embedded_head"
    assert "draft_model_identity_digest" not in language["speculative_decoding"]
    handle.close()
    handle.close()
    assert closed == [True]
    assert handle.tokenizer is None and handle.token_stream is None
    assert events == ["gc", "cache"]


def test_runtime_capture_delegates_stream_attributes():
    import sys

    from rapid_mlx.community_bench.acceleration import _capture_runtime_exit

    original = sys.stderr
    with _capture_runtime_exit():
        assert sys.stdout.isatty() == original.isatty()
        assert sys.stdout.fileno() == original.fileno()
        assert sys.stdout.encoding == original.encoding


@pytest.mark.parametrize(
    ("cap", "model", "expected"),
    [
        (
            32,
            SimpleNamespace(
                head_drafts=SimpleNamespace(drafter=SimpleNamespace(block_size=16)),
                drafts=15,
            ),
            15,
        ),
        (
            8,
            SimpleNamespace(
                head_drafts=SimpleNamespace(drafter=SimpleNamespace(block_size=16)),
                drafts=15,
            ),
            8,
        ),
        (15, SimpleNamespace(drafts=3), 3),
        (2, SimpleNamespace(drafts=3), 2),
        (7, SimpleNamespace(), 7),
    ],
)
def test_tensorfold_live_draft_cap(cap, model, expected):
    from rapid_mlx.speculative.serial_benchmark import tensorfold_live_max_draft

    backend = SimpleNamespace(
        _app=SimpleNamespace(engine=SimpleNamespace(max_draft=cap, model=model))
    )
    assert tensorfold_live_max_draft(backend) == expected


def test_tensorfold_missing_live_cap_refused():
    from rapid_mlx.speculative.serial_benchmark import tensorfold_live_max_draft

    with pytest.raises(RuntimeError, match="live max_draft"):
        tensorfold_live_max_draft(SimpleNamespace())


def test_ddtree_preflight_refuses_without_download(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.community_bench import local_runner
    from rapid_mlx.community_bench.acceleration import _refuse_unsupported

    def download(*args, **kwargs):
        pytest.fail("unexpected download")

    monkeypatch.setattr(local_runner, "_ensure_model_downloaded", download)
    args = SimpleNamespace(enable_ddtree=True)
    assert cli.serve_runtime_lane(args) == "ddtree"
    with pytest.raises(AccelerationConfigError, match="non-streaming"):
        _refuse_unsupported("model", "ddtree", False, args)


def test_runtime_exit_preserves_message(capsys):
    from rapid_mlx.community_bench.acceleration import _capture_runtime_exit

    with (
        pytest.raises(AccelerationConfigError, match="load refused"),
        _capture_runtime_exit(),
    ):
        print("load refused")
        raise SystemExit(1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "load refused" in captured.err


@pytest.fixture
def fake_dedicated(monkeypatch, tmp_path):
    import sys

    from rapid_mlx import cli
    from rapid_mlx.community_bench import local_runner as lr
    from rapid_mlx.community_bench import runner, workspace
    from rapid_mlx.community_bench.hardware import Hardware, Software
    from rapid_mlx.speculative.serial_benchmark import SerialRuntimeHandle

    for name, attrs in {
        "rapid_mlx.engine_core": dict(
            AsyncEngineCore=object,
            EngineConfig=object,
            _init_mlx_step_thread=lambda: None,
        ),
        "rapid_mlx.scheduler": dict(SchedulerConfig=object),
        "rapid_mlx.service.helpers": dict(
            get_model_max_context=lambda *a: 8192,
            _read_local_config_max_context=lambda *a: 8192,
        ),
        "rapid_mlx.utils.tokenizer": dict(
            load_model_with_fallback=lambda *a, **k: None
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attrs))
    target = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    draft = tmp_path / "models--org--draft" / "snapshots" / ("b" * 40)
    target.mkdir(parents=True)
    draft.mkdir(parents=True)
    closed = []
    state = SimpleNamespace(fail=False, exit_at=None)
    executor = ThreadPoolExecutor(max_workers=1)

    def stream(prompt, maximum):
        for i in range(maximum):
            if state.fail and i == 1:
                raise RuntimeError("stream broke")
            yield TokenChunk(i, len(prompt))

    handle = SerialRuntimeHandle(
        lane="dflash",
        tokenizer=SimpleNamespace(vocab_size=1000, all_special_ids=[]),
        executor=executor,
        token_stream=stream,
        speculative={
            "method": "dflash",
            "max_draft_tokens": 15,
            "draft_path": str(draft),
            "algorithm": "dflash2",
        },
        target_path=str(target),
        context_length=8192,
        close=lambda *, wait=True: closed.append(wait),
        prefill_chunk_size=2048,
    )

    def prepare(args):
        if state.exit_at == "prepare":
            print("prepare refused")
            raise SystemExit(1)

        def load():
            print("runtime progress")
            if state.exit_at == "load":
                print("load refused")
                raise SystemExit(1)
            return handle

        return cli.DedicatedRuntimeLaunch("dflash", load, None)

    async def bench(adapter, tokenizer, **kwargs):
        buckets = []
        for prompt_count, output_count in [(512, 128), (2048, 512)]:
            sampling = SimpleNamespace(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop=None,
                stop_token_ids=None,
                ignore_eos=True,
                max_tokens=output_count,
            )
            request = await adapter.add_request([1] * prompt_count, sampling)
            outputs = [out async for out in adapter.stream_outputs(request)]
            assert outputs[-1].completion_tokens == output_count
            result = runner.RoundResult(
                100,
                100,
                10,
                prompt_tokens=outputs[-1].prompt_tokens,
                output_tokens=outputs[-1].completion_tokens,
            )
            buckets.append(runner.BucketResult([result] * 5))
        return runner.BenchResult(*buckets, 1, "fake", "greedy")

    monkeypatch.setattr(runner, "run_standardized_bench", bench)
    monkeypatch.setattr(cli, "prepare_dedicated_runtime", prepare)
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda args: "dflash")
    monkeypatch.setattr(lr, "preflight_benchmark_acceleration", lambda *a: None)
    monkeypatch.setattr(
        lr, "build_serve_args", lambda *a: SimpleNamespace(model="org/model")
    )
    monkeypatch.setattr(
        lr,
        "resolve_benchmark_acceleration",
        lambda *a: (SimpleNamespace(lane="dflash"), SimpleNamespace(model="org/model")),
    )
    monkeypatch.setattr(lr, "_ensure_model_downloaded", lambda *a: None)
    monkeypatch.setattr(lr, "_uses_serving_benchmark_engine", lambda *a: False)
    monkeypatch.setattr(lr, "_is_deepseek_v41_benchmark_target", lambda *a: False)
    monkeypatch.setattr(
        workspace, "benchmark_runtime_readiness", lambda *a: {"status": "available"}
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
    archive = workspace.LocalRunArchive(tmp_path / "archive")
    yield SimpleNamespace(
        lr=lr, archive=archive, closed=closed, state=state, handle=handle
    )
    executor.shutdown(wait=True)


def test_dedicated_run_archives_valid_metadata(fake_dedicated):
    from rapid_mlx.community_bench.benchmark_contracts import BenchmarkRunValidator

    f = fake_dedicated
    run = f.lr.run_local("example", archive=f.archive)
    archived = f.archive.get(run["run_id"])
    BenchmarkRunValidator().validate(archived)
    language = archived["execution"]["task"]["language"]
    assert language["runtime"]["kind"] == "dflash"
    assert language["prefill_chunk_size"] == 2048
    spec = language["speculative_decoding"]
    assert spec["method"] == "dflash" and spec["max_draft_tokens"] == 15
    assert spec["draft_model_identity_digest"].startswith("sha256:")
    assert any(
        c["source"].get("repo_id") == "org/draft"
        for c in archived["model"]["components"]
    )
    assert f.closed == [True]


@pytest.mark.parametrize("embedded", [True, False])
def test_dedicated_mtp_sidecar_identity(fake_dedicated, embedded):
    from pathlib import Path

    f = fake_dedicated
    (Path(f.handle.target_path) / "config.json").write_text("{}")
    draft = f.handle.target_path if embedded else f.handle.speculative["draft_path"]
    f.handle.speculative = {
        "method": "mtp",
        "max_draft_tokens": 15,
        "draft_path": draft,
    }
    workload = {
        "cases": [
            {
                "case_id": "short",
                "target_prompt_tokens": 512,
                "target_output_tokens": 128,
            },
            {
                "case_id": "long",
                "target_prompt_tokens": 2048,
                "target_output_tokens": 512,
            },
        ]
    }
    _, _, identity, language = asyncio.run(
        f.lr._measure_dedicated(f.handle, "org/model", "example", workload, None)
    )
    spec = language["speculative_decoding"]
    assert spec["assistant_source"] == (
        "embedded_head" if embedded else "external_model"
    )
    if embedded:
        assert "draft_model_identity_digest" not in spec
    else:
        assert spec["draft_model_identity_digest"].startswith("sha256:")
    assert any(c["role"] == "other" for c in identity["components"])


def test_dedicated_stream_failure_closes_and_records(fake_dedicated):
    f = fake_dedicated
    f.state.fail = True
    with pytest.raises(f.lr.LocalBenchmarkError, match="stream broke") as exc:
        f.lr.run_local("example", archive=f.archive)
    assert exc.value.saved
    assert f.archive.list()[0]["outcome"]["status"] == "failed"
    assert f.closed == [True]


def test_dedicated_missing_context_refused_and_closed(fake_dedicated, monkeypatch):
    import sys

    f = fake_dedicated
    f.handle.context_length = None
    monkeypatch.setattr(
        sys.modules["rapid_mlx.service.helpers"],
        "_read_local_config_max_context",
        lambda *args: None,
    )
    with pytest.raises(
        f.lr.LocalBenchmarkError,
        match="Cannot determine the dedicated runtime context length",
    ):
        f.lr.run_local("example", archive=f.archive)
    assert f.closed == [True]


@pytest.mark.parametrize("phase", ["prepare", "load"])
def test_runtime_exit_is_unarchived_refusal(fake_dedicated, phase):
    f = fake_dedicated
    f.state.exit_at = phase
    with pytest.raises(f.lr.LocalBenchmarkError, match=f"{phase} refused") as exc:
        f.lr.run_local("example", archive=f.archive)
    assert not exc.value.saved
    assert f.archive.list() == []


def test_cli_json_dedicated_run(fake_dedicated, monkeypatch, capsys):
    import json

    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench import cli

    monkeypatch.setattr(cli.LocalRunArchive, "default", lambda: fake_dedicated.archive)
    args = build_parser().parse_args(["benchmark", "run", "example", "--json"])
    assert cli.benchmark_command(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["outcome"]["status"] == "completed"


@pytest.mark.parametrize(
    ("alias", "flags", "expected"),
    [
        ("qwen3.8-27b-tensorfold", [], "tensorfold_dflash"),
        ("glm5.3-flash-4bit", [], "native_mtp"),
        (
            "qwen3.8-27b-4bit",
            [
                "--speculative-config",
                '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2"}',
            ],
            "dflash",
        ),
        ("llama-3.2-1b-4bit", [], "engine"),
        ("glm5.3-flash-4bit", ["--no-spec-decode"], "engine"),
        ("qwen3.8-27b-4bit", ["--no-spec-decode"], "engine"),
    ],
)
def test_parser_lane_parity(monkeypatch, alias, flags, expected):
    import sys
    import types

    import numpy as np
    import psutil

    from rapid_mlx import cli
    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench.acceleration import (
        build_serve_args,
        request_from_args,
        resolve_benchmark_acceleration,
    )

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
    import rapid_mlx.speculative.native_mtp.runtime as native

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024**3, available=64 * 1024**3),
    )
    monkeypatch.setattr(
        mtp,
        "detect_mtp_eligibility",
        lambda *a, **k: next(
            x for x in mtp.MTPEligibility if x != mtp.MTPEligibility.NONE
        ),
    )
    monkeypatch.setattr(cli, "_cached_mtp_sidecar", lambda sidecar: sidecar)
    monkeypatch.setattr(cli, "_auto_config_lookup_key", lambda x: x)
    monkeypatch.setattr(cli, "_gather_kv_cache_dtype_inputs", lambda *a: ({}, {}))
    monkeypatch.setattr(cli, "_native_mtp_runtime_ready", lambda *a: True)
    monkeypatch.setattr(cli, "_preflight_tensorfold_qwen27_or_exit", lambda *a: None)
    monkeypatch.setattr(auto, "detect_model_config", lambda *a: None)
    monkeypatch.setattr(utils, "resolve_serving_lane", lambda *a, **k: (False, None))
    monkeypatch.setattr(dflash, "have_runtime", lambda: True)
    monkeypatch.setattr(native, "have_runtime", lambda: True)
    monkeypatch.setattr(native, "have_glm_cache_runtime", lambda: True)
    parser = build_parser()
    request = request_from_args(parser.parse_args(["benchmark", "run", alias, *flags]))
    resolved = build_serve_args(alias, request)
    direct = parser.parse_args(["serve", alias, *flags])
    direct.model = resolved.model
    if hasattr(resolved, "_original_alias"):
        direct._original_alias = resolved._original_alias
    direct._prefill_step_size_explicit = False
    plan = cli.resolve_serve_acceleration(direct)
    actual, _ = resolve_benchmark_acceleration(alias, request)
    assert actual.lane == cli.serve_runtime_lane(direct) == plan.lane == expected


def test_baseline_uses_plain_loader(fake_dedicated, monkeypatch):
    import sys

    from rapid_mlx.cli_parser import build_parser
    from rapid_mlx.community_bench.acceleration import request_from_args

    def baseline_loader(*args, **kwargs):
        raise RuntimeError("baseline loader reached")

    monkeypatch.setattr(
        sys.modules["rapid_mlx.utils.tokenizer"],
        "load_model_with_fallback",
        baseline_loader,
    )
    request = request_from_args(
        build_parser().parse_args(["benchmark", "run", "example", "--baseline"])
    )
    assert request.baseline
    with pytest.raises(
        fake_dedicated.lr.LocalBenchmarkError, match="baseline loader reached"
    ):
        fake_dedicated.lr.run_local(
            "example", archive=fake_dedicated.archive, acceleration=request
        )
    assert fake_dedicated.closed == []


@pytest.mark.parametrize("env, expected", [("4", 4), (None, 1)])
def test_tensorfold_loader_lanes(monkeypatch, env, expected):
    from rapid_mlx import cli
    from rapid_mlx.speculative import serial_benchmark, tensorfold_qwen27_server
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    if env is None:
        monkeypatch.delenv(LANES_ENV, raising=False)
    else:
        monkeypatch.setenv(LANES_ENV, env)
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda args: "tensorfold_dflash")
    monkeypatch.setattr(cli, "_preflight_tensorfold_qwen27_or_exit", lambda *a: None)
    monkeypatch.setattr(cli, "_download_tensorfold_artifacts", lambda *a: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        tensorfold_qwen27_server,
        "load_tensorfold_backend",
        lambda **kwargs: calls.append(kwargs) or (object(), object()),
    )
    monkeypatch.setattr(
        serial_benchmark, "tensorfold_live_max_draft", lambda backend: 1
    )
    monkeypatch.setattr(
        serial_benchmark,
        "tensorfold_runtime_handle",
        lambda *a, **k: SimpleNamespace(
            lane=k["lane"],
            speculative={
                "method": k["method"],
                "max_draft_tokens": k["max_draft_tokens"],
                "lanes": k["lanes"],
            },
        ),
    )
    launch = cli.prepare_dedicated_runtime(
        SimpleNamespace(model="qwen3.8-27b-tensorfold")
    )
    handle = launch.load()
    assert calls[0]["tensorfold_lanes"] == expected
    assert (
        dedicated_language(
            handle, context_length=1, prefill_chunk_size=1, draft_identity_digest=None
        )["runtime"]["lanes"]
        == expected
    )


def test_tensorfold_unqualified_lanes_refused_before_load(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    monkeypatch.setenv(LANES_ENV, "4")
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda args: "tensorfold_mtp")
    with pytest.raises(AccelerationConfigError, match="qualified"):
        cli.prepare_dedicated_runtime(SimpleNamespace(model="glm5.3-flash-tensorfold"))


def test_tensorfold_loader_live_cap_failure(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.speculative import serial_benchmark, tensorfold_qwen27_server

    closed = []
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda args: "tensorfold_dflash")
    monkeypatch.setattr(cli, "_preflight_tensorfold_qwen27_or_exit", lambda *a: None)
    monkeypatch.setattr(cli, "_download_tensorfold_artifacts", lambda *a: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    backend = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(
        tensorfold_qwen27_server, "load_tensorfold_backend", lambda **k: (backend, None)
    )

    def fail(backend):
        raise RuntimeError("invalid cap")

    monkeypatch.setattr(serial_benchmark, "tensorfold_live_max_draft", fail)
    with pytest.raises(AccelerationConfigError, match="invalid cap"):
        cli.prepare_dedicated_runtime(SimpleNamespace(model="model")).load()
    assert closed == [True]


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(
            head_drafts=SimpleNamespace(drafter=SimpleNamespace(block_size=1))
        ),
        SimpleNamespace(drafts=0),
    ],
)
def test_tensorfold_invalid_model_cap(model):
    from rapid_mlx.speculative.serial_benchmark import tensorfold_live_max_draft

    backend = SimpleNamespace(
        _app=SimpleNamespace(engine=SimpleNamespace(max_draft=4, model=model))
    )
    with pytest.raises(RuntimeError):
        tensorfold_live_max_draft(backend)


def test_capture_flush_and_engine_preflight(monkeypatch):
    import sys

    from rapid_mlx import cli
    from rapid_mlx.community_bench.acceleration import (
        AccelerationRequest,
        _capture_runtime_exit,
        preflight_benchmark_acceleration,
    )

    with _capture_runtime_exit():
        sys.stdout.flush()
    monkeypatch.setattr(cli, "serve_acceleration_preflights", lambda args: None)
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda args: "engine")
    with pytest.raises(AccelerationConfigError, match="dspark"):
        preflight_benchmark_acceleration(
            "model", AccelerationRequest(overrides={"spec_decode": "dspark"})
        )


def test_adapter_exhaustion_and_timeout():
    async def run(executor):
        adapter = _DedicatedRuntimeBenchmarkAdapter(
            SimpleNamespace(executor=executor, token_stream=lambda *a: (x for x in ()))
        )
        sampling = SimpleNamespace(max_tokens=2)
        adapter._requests["empty"] = ([1], sampling)
        assert [x async for x in adapter.stream_outputs("empty")] == []
        adapter._requests["timeout"] = ([1], sampling)
        with pytest.raises(asyncio.TimeoutError):
            _ = [x async for x in adapter.stream_outputs("timeout", timeout=0)]
        assert not adapter._active

    with ThreadPoolExecutor(max_workers=1) as executor:
        asyncio.run(run(executor))


@pytest.mark.parametrize("case", ["fallback", "mismatch", "draft", "missing"])
def test_dedicated_identity_refusals(fake_dedicated, monkeypatch, case):
    f = fake_dedicated
    lr = f.lr
    if case == "missing":
        f.handle.target_path = None
        with pytest.raises(AccelerationConfigError, match="target snapshot"):
            asyncio.run(
                lr._measure_dedicated(f.handle, "org/model", "example", {}, None)
            )
    elif case == "fallback":
        f.handle.target_path = "org/model"
        monkeypatch.setattr(
            lr,
            "_sidecar_identity",
            lambda path: (_ for _ in ()).throw(ValueError("bad")),
        )
        # The target fallback succeeds; the draft lookup fails separately.
        with pytest.raises(ValueError, match="bad"):
            asyncio.run(
                lr._measure_dedicated(f.handle, "org/model", "example", {}, None)
            )
    else:
        if case == "mismatch":
            f.handle.target_path = "other/model"
            monkeypatch.setattr(
                lr,
                "_sidecar_identity",
                lambda path: (_ for _ in ()).throw(ValueError("bad")),
            )
            match = "catalog checkpoint"
        else:
            f.handle.speculative.pop("draft_path")
            match = "draft snapshot"
        with pytest.raises(AccelerationConfigError, match=match):
            asyncio.run(
                lr._measure_dedicated(f.handle, "org/model", "example", {}, None)
            )


def test_dedicated_launch_refusal(fake_dedicated, monkeypatch):
    from rapid_mlx import cli

    f = fake_dedicated
    monkeypatch.setattr(
        cli,
        "prepare_dedicated_runtime",
        lambda args: cli.DedicatedRuntimeLaunch("dflash", None, "refused"),
    )
    with pytest.raises(f.lr.LocalBenchmarkError, match="refused"):
        f.lr.run_local("example", archive=f.archive)


def test_timeout_submits_close_without_waiting():
    import threading
    import time

    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def stream(*args):
        try:
            started.set()
            release.wait()
            yield TokenChunk(1, 1)
        finally:
            closed.set()

    async def run(executor):
        adapter = _DedicatedRuntimeBenchmarkAdapter(
            SimpleNamespace(executor=executor, token_stream=stream)
        )
        adapter._requests["blocked"] = ([1], SimpleNamespace(max_tokens=1))
        start = time.monotonic()
        with pytest.raises(asyncio.TimeoutError):
            _ = [x async for x in adapter.stream_outputs("blocked", timeout=0.1)]
        assert started.is_set()
        assert time.monotonic() - start < 1
        assert not closed.is_set()
        release.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            asyncio.run(run(executor))
        finally:
            release.set()
    assert closed.is_set()


@pytest.mark.parametrize(
    "alias,lane,profile_id",
    [
        ("glm5.3-flash-tensorfold", "tensorfold_mtp", "glm5.3-flash-tensorfold"),
        ("bonsai2-27b-tensorfold", "tensorfold_dflash", "bonsai2-27b-tensorfold"),
    ],
)
def test_tensorfold_profile_lane_resolution(monkeypatch, alias, lane, profile_id):
    from rapid_mlx import cli
    from rapid_mlx.speculative import tensorfold_concurrency

    seen = []
    monkeypatch.setattr(
        tensorfold_concurrency,
        "request_lanes",
        lambda profile, value: seen.append((profile, value)) or 1,
    )
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda args: lane)
    monkeypatch.setattr(cli, "_preflight_tensorfold_qwen27_or_exit", lambda *a: None)
    monkeypatch.setattr(cli, "_download_tensorfold_artifacts", lambda *a: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    cli.prepare_dedicated_runtime(SimpleNamespace(model=alias, tensorfold_lanes=1))
    assert seen == [(profile_id, 1)]


def test_measure_timeout_does_not_wait_for_blocked_step(fake_dedicated, monkeypatch):
    import threading

    from rapid_mlx.community_bench import runner

    f = fake_dedicated
    started = threading.Event()
    release = threading.Event()
    closed = []

    def stream(*args):
        started.set()
        release.wait()
        yield TokenChunk(1, 1)

    def close(*, wait=True):
        closed.append(wait)
        future = f.handle.executor.submit(lambda: None)
        if wait:
            future.result()

    async def bench(adapter, tokenizer, **kwargs):
        adapter._requests["blocked"] = ([1], SimpleNamespace(max_tokens=1))
        return [x async for x in adapter.stream_outputs("blocked", timeout=0.1)]

    f.handle.token_stream = stream
    f.handle.close = close
    monkeypatch.setattr(runner, "run_standardized_bench", bench)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(
            lambda: asyncio.run(
                f.lr._measure_dedicated(f.handle, "org/model", "example", {}, None)
            )
        )
        try:
            with pytest.raises(asyncio.TimeoutError):
                future.result(timeout=2)
            assert started.is_set() and not release.is_set()
            assert closed == [False]
        finally:
            release.set()


@pytest.mark.parametrize(
    "lane,expected_download",
    [
        ("companion_dspark", False),
        ("tensorfold_mtp", False),
        ("engine", True),
        ("native_mtp", True),
        ("dflash", True),
    ],
)
def test_text_download_follows_resolved_lane(
    fake_dedicated, monkeypatch, lane, expected_download
):
    from rapid_mlx import cli
    from rapid_mlx.community_bench.acceleration import AccelerationRequest
    from rapid_mlx.spec_decode.dspark import eligibility, runtime

    lr = fake_dedicated.lr
    args = SimpleNamespace(model="org/model")
    pair = object()
    calls = []
    monkeypatch.setattr(lr, "build_serve_args", lambda *a: args)
    monkeypatch.setattr(
        cli,
        "_normalize_speculative_config_or_exit",
        lambda a: setattr(
            a,
            "_speculative_config",
            SimpleNamespace(
                method="dspark" if lane == "companion_dspark" else "none", model="draft"
            ),
        ),
    )
    monkeypatch.setattr(eligibility, "resolve_companion_dspark_pair", lambda **kw: pair)
    monkeypatch.setattr(runtime, "have_runtime", lambda: True)
    monkeypatch.setattr(
        cli,
        "serve_runtime_lane",
        lambda a: "companion_dspark" if a._companion_dspark_pair is pair else lane,
    )
    monkeypatch.setattr(lr, "_ensure_model_downloaded", calls.append)

    def stop(*args):
        raise RuntimeError("download decision complete")

    monkeypatch.setattr(lr, "resolve_benchmark_acceleration", stop)
    with pytest.raises(RuntimeError, match="download decision complete"):
        asyncio.run(
            lr._text_measurements("example", acceleration=AccelerationRequest())
        )
    assert calls == (["org/model"] if expected_download else [])


def test_partial_pinned_target_snapshot_dedicated(
    fake_dedicated, monkeypatch, tmp_path
):
    import huggingface_hub
    from huggingface_hub import constants, errors

    from rapid_mlx import model_downloads
    from rapid_mlx.speculative.serial_benchmark import cached_snapshot_path

    def unavailable(*args, **kwargs):
        raise errors.LocalEntryNotFoundError("partial cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unavailable)
    monkeypatch.setattr(model_downloads, "snapshot_download", unavailable)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    revision = "a" * 40
    snapshot = tmp_path / "models--org--model" / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}")
    path = cached_snapshot_path("org/model", revision)
    assert path == str(snapshot)
    fake_dedicated.handle.target_path = path
    workload = {
        "cases": [
            {
                "case_id": "short",
                "target_prompt_tokens": 512,
                "target_output_tokens": 128,
            },
            {
                "case_id": "long",
                "target_prompt_tokens": 2048,
                "target_output_tokens": 512,
            },
        ]
    }
    _, _, identity, _ = asyncio.run(
        fake_dedicated.lr._measure_dedicated(
            fake_dedicated.handle, "org/model", "example", workload, None
        )
    )
    assert identity["components"][0]["source"]["resolved_revision"] == revision
