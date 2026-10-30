# SPDX-License-Identifier: Apache-2.0
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.speculative.serial_benchmark import TokenChunk, mlx_vlm_token_stream


@pytest.fixture(autouse=True)
def fake_mlx(monkeypatch):
    mlx = ModuleType("mlx")
    mlx.__spec__ = ModuleSpec("mlx", loader=None)
    core = ModuleType("mlx.core")
    core.array = lambda values: SimpleNamespace(tolist=lambda: values)
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_stream_uses_exact_ids_ignores_eos_and_restores():
    original = object()
    tokenizer = SimpleNamespace(
        stopping_criteria=original, decode=lambda ids: "decoded"
    )
    observed = {}

    def generate(model, processor, prompt, *, input_ids, **kwargs):
        observed.update(ids=input_ids.tolist(), prompt=prompt, kwargs=kwargs)
        for token in [None, 1, 2, 3, 4]:
            if token is not None and tokenizer.stopping_criteria(token):
                return
            yield SimpleNamespace(token=token, prompt_tokens=2)

    stream = mlx_vlm_token_stream(
        None,
        tokenizer,
        stream_generate_fn=generate,
        generation_kwargs=lambda n: {"max_tokens": n},
    )
    assert list(stream([5, 6], 3)) == [TokenChunk(i, 2) for i in (1, 2, 3)]
    assert observed == {
        "ids": [[5, 6]],
        "prompt": "decoded",
        "kwargs": {"max_tokens": 3},
    }
    assert tokenizer.stopping_criteria is original


def test_stream_restores_after_failure():
    original = object()
    tokenizer = SimpleNamespace(stopping_criteria=original, decode=lambda ids: "")

    def generate(*args, **kwargs):
        raise ValueError("failure")
        yield

    stream = mlx_vlm_token_stream(
        None, tokenizer, stream_generate_fn=generate, generation_kwargs=lambda n: {}
    )
    with pytest.raises(ValueError):
        list(stream([1], 1))
    assert tokenizer.stopping_criteria is original


def test_runtime_handles_read_live_facts(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from rapid_mlx.speculative import serial_benchmark as serial

    server = ModuleType("rapid_mlx.speculative.dflash.server")
    executor = ThreadPoolExecutor(max_workers=1)
    server._dflash_executor = executor
    server.dflash_generation_kwargs = lambda runtime, **kwargs: kwargs
    generate_common = ModuleType("mlx_vlm.generate.common")
    generate_common.DEFAULT_PREFILL_STEP_SIZE = 2048
    monkeypatch.setitem(sys.modules, generate_common.__name__, generate_common)
    common = ModuleType("mlx_vlm.speculative.common")
    common._dflash_block_total = lambda drafter, override: drafter.config.block_size
    monkeypatch.setitem(sys.modules, server.__name__, server)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setattr(serial, "cached_snapshot_path", lambda repo, revision: "/draft")
    events = []
    monkeypatch.setattr(serial.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(
        sys.modules["mlx.core"],
        "clear_cache",
        lambda: events.append("cache"),
        raising=False,
    )
    runtime = SimpleNamespace(
        drafter=SimpleNamespace(config=SimpleNamespace(block_size=8)),
        drafter_repo="repo",
        drafter_revision="rev",
        algorithm="dflash2",
        block_size=3,
        num_speculative_tokens=7,
    )
    processor = SimpleNamespace(tokenizer=object())
    artifacts = SimpleNamespace(target_path="/target", drafter_path="/sidecar")
    handles = [
        serial.dflash_runtime_handle(
            None,
            processor,
            runtime,
            drafter_repo="repo",
            drafter_revision="rev",
            target_path="/target",
        ),
        serial.native_mtp_runtime_handle(
            None, processor, runtime, lambda **kw: kw, pair=None, target_path="/target"
        ),
        serial.companion_dspark_runtime_handle(
            None, processor, runtime, lambda **kw: kw, pair=None, artifacts=artifacts
        ),
    ]
    assert [h.speculative["max_draft_tokens"] for h in handles] == [7, 2, 7]
    assert [h.speculative["method"] for h in handles] == ["dflash", "mtp", "dspark"]
    assert handles[0].speculative["algorithm"] == "dflash2"
    assert handles[2].speculative["draft_path"] == "/sidecar"
    for handle in handles:
        handle.close()
        handle.close()
        assert handle.tokenizer is None and handle.token_stream is None
    assert events == ["gc", "cache"] * 3
    executor.shutdown()


def test_dedicated_launch_refusal_and_placeholder(monkeypatch):
    from rapid_mlx import cli

    args = SimpleNamespace(enable_ddtree=True)
    launch = cli.prepare_dedicated_runtime(args)
    assert launch.lane == "ddtree"
    assert "non-streaming" in launch.refusal
    assert launch.load is None
    with pytest.raises(ValueError):
        cli.prepare_dedicated_runtime(SimpleNamespace())


def test_servers_delegate_loading(monkeypatch):
    from rapid_mlx import _uvicorn
    from rapid_mlx.spec_decode.dspark import server as dspark
    from rapid_mlx.speculative.dflash import server as dflash
    from rapid_mlx.speculative.native_mtp import server as mtp

    calls = []
    monkeypatch.setattr(dflash, "_build_app", lambda **kwargs: object())
    monkeypatch.setattr(_uvicorn, "run_uvicorn", lambda *args, **kwargs: None)
    monkeypatch.setattr(mtp, "run_uvicorn", lambda *args, **kwargs: None)
    monkeypatch.setattr(dspark, "_build_companion_model_info", lambda **kwargs: None)
    monkeypatch.setattr(
        dflash,
        "load_dflash_runtime",
        lambda **kwargs: calls.append(kwargs) or (None, None, None),
    )
    monkeypatch.setattr(
        mtp,
        "load_native_mtp_runtime",
        lambda pair, **kwargs: calls.append((pair, kwargs)) or (None, None, None, None),
    )
    monkeypatch.setattr(
        dspark,
        "load_companion_dspark_runtime",
        lambda pair, artifacts: (
            calls.append((pair, artifacts)) or (None, None, None, None)
        ),
    )
    common = dict(
        host="localhost",
        port=8000,
        served_model_name="model",
        default_max_tokens=10,
        cors_origins=[],
        uvicorn_log_level="error",
    )
    dflash.run_dflash_server(
        main_model_repo="target",
        main_model_revision="t",
        drafter_repo="draft",
        drafter_revision="d",
        experimental_opt_in=True,
        expected_algorithm="dflash2",
        **common,
    )
    pair = SimpleNamespace(target_repo="target")
    mtp.run_native_mtp_server(pair=pair, prefill_step_size=42, **common)
    dspark.run_companion_dspark_server(pair=pair, artifacts="artifacts", **common)
    assert calls == [
        dict(
            main_model_repo="target",
            main_model_revision="t",
            drafter_repo="draft",
            drafter_revision="d",
            experimental_opt_in=True,
            expected_algorithm="dflash2",
        ),
        (pair, {"prefill_step_size": 42}),
        (pair, "artifacts"),
    ]


def test_prepare_dflash_uses_serve_load_kwargs(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.community_bench.acceleration import (
        AccelerationRequest,
        build_serve_args,
    )
    from rapid_mlx.speculative import serial_benchmark
    from rapid_mlx.speculative.dflash import eligibility, server

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(eligibility, "have_runtime", lambda: True)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    args = build_serve_args(
        "qwen3.8-27b-4bit",
        AccelerationRequest(
            overrides={
                "speculative_config": '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2"}'
            }
        ),
    )
    cli.serve_acceleration_preflights(args)
    launch = cli.prepare_dedicated_runtime(args)
    expected = cli._dflash_load_kwargs(args)
    calls = []
    monkeypatch.setattr(
        server,
        "load_dflash_runtime",
        lambda **kw: calls.append(kw) or (None, None, None),
    )
    monkeypatch.setattr(serial_benchmark, "cached_snapshot_path", lambda *a: None)
    monkeypatch.setattr(
        serial_benchmark, "dflash_runtime_handle", lambda *a, **k: "handle"
    )
    assert launch.load() == "handle"
    assert calls == [expected]


def test_prepare_native_and_companion(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.community_bench.acceleration import (
        AccelerationRequest,
        build_serve_args,
    )
    from rapid_mlx.speculative.native_mtp import runtime as mtp_runtime

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(mtp_runtime, "have_runtime", lambda: True)
    monkeypatch.setattr(mtp_runtime, "have_glm_cache_runtime", lambda: True)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    args = build_serve_args(
        "glm5.3-flash-4bit", AccelerationRequest(overrides={"mtp_backend": "native"})
    )
    cli.serve_acceleration_preflights(args)
    assert cli.prepare_dedicated_runtime(args).lane == "native_mtp"
    pair = SimpleNamespace(target_repo="target", drafter_repo="draft")
    monkeypatch.setattr(
        cli,
        "_preflight_companion_dspark_or_exit",
        lambda args: setattr(args, "_companion_dspark_pair", pair),
    )
    monkeypatch.setattr(
        cli, "_download_companion_dspark_artifacts", lambda args, pair: "artifacts"
    )
    args = build_serve_args("glm5.3-flash-4bit", AccelerationRequest())
    cli.serve_acceleration_preflights(args)
    launch = cli.prepare_dedicated_runtime(args)
    assert launch.lane == "companion_dspark" and launch.load is not None
