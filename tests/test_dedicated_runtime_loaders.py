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
            None,
            processor,
            runtime,
            lambda runtime, **kw: kw,
            pair=None,
            target_path="/target",
        ),
        serial.companion_dspark_runtime_handle(
            None,
            processor,
            runtime,
            lambda runtime, **kw: kw,
            pair=None,
            artifacts=artifacts,
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
    apps = []
    monkeypatch.setattr(
        dflash, "_build_app", lambda **kwargs: apps.append(kwargs) or object()
    )
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
        lambda pair, **kwargs: (
            calls.append((pair, kwargs))
            or (None, None, "mtp-runtime", lambda runtime, **kw: runtime)
        ),
    )
    monkeypatch.setattr(
        dspark,
        "load_companion_dspark_runtime",
        lambda pair, artifacts: (
            calls.append((pair, artifacts))
            or (None, None, "dspark-runtime", lambda runtime, **kw: runtime)
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
    assert apps[1]["generation_kwargs_fn"]() == "mtp-runtime"
    assert apps[2]["generation_kwargs_fn"]() == "dspark-runtime"
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


def test_stream_default_import_zero_and_criteria_methods(monkeypatch):
    criteria = object()
    tokenizer = SimpleNamespace(stopping_criteria=criteria, decode=lambda ids: "")

    def generate(*args, **kwargs):
        tokenizer.stopping_criteria.reset()
        tokenizer.stopping_criteria.add_eos_token_ids([1])
        yield SimpleNamespace(token=7, prompt_tokens=1)

    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(stream_generate=generate)
    )
    stream = mlx_vlm_token_stream(
        None, tokenizer, stream_generate_fn=None, generation_kwargs=lambda n: {}
    )
    assert list(stream([1], 0)) == []
    assert list(stream([1], 1)) == [TokenChunk(7, 1)]
    assert tokenizer.stopping_criteria is criteria


def test_cached_snapshot_paths(monkeypatch, tmp_path):
    from huggingface_hub import constants

    from rapid_mlx.speculative.serial_benchmark import cached_snapshot_path

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    assert cached_snapshot_path(str(tmp_path), None) == str(tmp_path.resolve())
    assert cached_snapshot_path("org/model", "rev") is None
    snapshot = tmp_path / "models--org--model" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    assert cached_snapshot_path("org/model", "rev") == str(snapshot)


def test_cached_snapshot_resolves_named_revision(monkeypatch, tmp_path):
    from huggingface_hub import constants

    from rapid_mlx.speculative.serial_benchmark import cached_snapshot_path

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    cache = tmp_path / "models--org--model"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("abc123\n")
    snapshot = cache / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert cached_snapshot_path("org/model", "main") == str(snapshot)


def test_dflash_unknown_profile_refusal(monkeypatch):
    from rapid_mlx import cli, model_aliases
    from rapid_mlx.speculative import dflash
    from rapid_mlx.speculative.dflash import eligibility

    monkeypatch.setattr(model_aliases, "resolve_profile", lambda name: None)
    monkeypatch.setattr(cli, "_resolve_dflash_drafter_repo", lambda *a: "draft")
    monkeypatch.setattr(
        eligibility, "report", lambda *a, **k: SimpleNamespace(warnings=[])
    )

    def refuse(*args, **kwargs):
        raise dflash.DFlashUnavailable("refused")

    monkeypatch.setattr(dflash, "check", refuse)
    with pytest.raises(SystemExit):
        cli._resolve_dflash_launch_or_exit(SimpleNamespace(model="unknown"))


def test_deferred_native_and_companion_loads(monkeypatch):
    from rapid_mlx import cli
    from rapid_mlx.spec_decode.dspark import server as companion
    from rapid_mlx.speculative import serial_benchmark as serial
    from rapid_mlx.speculative.native_mtp import server as native

    pair = SimpleNamespace(target_repo="target", target_revision="rev")
    monkeypatch.setattr(cli, "_prepare_native_mtp_checks", lambda args: pair)
    monkeypatch.setattr(cli, "_native_mtp_prefill", lambda *a: 42)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **k: None)
    monkeypatch.setattr(
        cli, "_download_companion_dspark_artifacts", lambda *a: "artifacts"
    )
    monkeypatch.setattr(native, "load_native_mtp_runtime", lambda *a, **k: (1, 2, 3, 4))
    monkeypatch.setattr(
        companion, "load_companion_dspark_runtime", lambda *a: (1, 2, 3, 4)
    )
    monkeypatch.setattr(serial, "cached_snapshot_path", lambda *a: "/target")
    monkeypatch.setattr(serial, "native_mtp_runtime_handle", lambda *a, **k: (a, k))
    monkeypatch.setattr(
        serial, "companion_dspark_runtime_handle", lambda *a, **k: (a, k)
    )
    args = SimpleNamespace(model="target", _companion_dspark_pair=pair)
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda a: "native_mtp")
    assert cli.prepare_dedicated_runtime(args).load()[1]["target_path"] == "/target"
    monkeypatch.setattr(cli, "serve_runtime_lane", lambda a: "companion_dspark")
    assert cli.prepare_dedicated_runtime(args).load()[1]["artifacts"] == "artifacts"


def test_runtime_handle_releases_model(monkeypatch):
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    from rapid_mlx.speculative import serial_benchmark as serial

    server = ModuleType("rapid_mlx.speculative.dflash.server")
    server._dflash_executor = ThreadPoolExecutor(max_workers=1)
    server.dflash_generation_kwargs = lambda runtime, **kw: {"prefill_step_size": 1}
    monkeypatch.setitem(sys.modules, server.__name__, server)
    common = ModuleType("mlx_vlm.speculative.common")
    common._dflash_block_total = lambda drafter, override: 2
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setattr(serial, "cached_snapshot_path", lambda *args: None)
    monkeypatch.setattr(
        sys.modules["mlx.core"], "clear_cache", lambda: None, raising=False
    )

    class Model:
        pass

    class Runtime:
        drafter = object()
        drafter_repo = "draft"
        drafter_revision = "rev"
        algorithm = "dflash"

    model, runtime = Model(), Runtime()
    model_ref, runtime_ref = weakref.ref(model), weakref.ref(runtime)
    handle = serial.dflash_runtime_handle(
        model,
        SimpleNamespace(tokenizer=object()),
        runtime,
        drafter_repo="draft",
        drafter_revision="rev",
        target_path="/target",
    )

    def fake_stream(model, processor, *, stream_generate_fn, generation_kwargs):
        assert model is model_ref()
        assert generation_kwargs(2)["prefill_step_size"] == 1
        return lambda prompt, maximum: iter([maximum])

    monkeypatch.setattr(serial, "mlx_vlm_token_stream", fake_stream)
    assert list(handle.token_stream([1], 2)) == [2]
    del model, runtime
    handle.close()
    assert model_ref() is None
    assert runtime_ref() is None
    server._dflash_executor.shutdown()


@pytest.mark.parametrize("kind", ["runtime", "tensorfold"])
def test_handle_close_without_wait_queues_cleanup(monkeypatch, kind):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from rapid_mlx.speculative import serial_benchmark as serial

    monkeypatch.setattr(
        sys.modules["mlx.core"], "clear_cache", lambda: None, raising=False
    )
    started = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()
    if kind == "runtime":
        executor = ThreadPoolExecutor(max_workers=1)
        server = ModuleType("rapid_mlx.speculative.dflash.server")
        server._dflash_executor = executor
        monkeypatch.setitem(sys.modules, server.__name__, server)
        handle = serial._runtime_handle(
            "dflash",
            None,
            object(),
            None,
            lambda runtime, **kwargs: {"prefill_step_size": 1},
            {},
            "/target",
        )
        monkeypatch.setattr(serial.gc, "collect", lambda: cleaned.set())
    else:
        backend = SimpleNamespace(
            _app=SimpleNamespace(tokenizer=object()), close=cleaned.set
        )
        handle = serial.tensorfold_runtime_handle(
            backend,
            object(),
            lane="tensorfold_mtp",
            method="mtp",
            algorithm="mtp",
            draft_repo=None,
            draft_revision=None,
            draft_path=None,
            target_path="/target",
            max_draft_tokens=1,
            lanes=1,
        )
        executor = handle.executor

    def block():
        started.set()
        release.wait()

    executor.submit(block)
    assert started.wait(2)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            try:
                caller.submit(handle.close, wait=False).result(timeout=2)
                handle.close(wait=False)
                assert not cleaned.is_set()
            finally:
                release.set()
        executor.shutdown(wait=True)
        assert cleaned.is_set()
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_companion_loader_generation_receives_runtime(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from rapid_mlx.spec_decode.dspark import server as companion
    from rapid_mlx.speculative.dflash import server as dflash

    runtime = SimpleNamespace(drafter=object(), kind="dspark", draft_block_size=8)
    monkeypatch.setattr(
        companion, "load_runtime", lambda pair, artifacts: (1, 2, runtime)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(dflash, "_dflash_executor", executor)
        _, _, loaded, generation = companion.load_companion_dspark_runtime(
            None, "artifacts"
        )
        kwargs = generation(loaded, max_tokens=5, temperature=0.0, top_p=1.0)
    assert kwargs["draft_model"] is runtime.drafter
    assert kwargs["draft_block_size"] == 8
    with pytest.raises(RuntimeError, match="greedy"):
        generation(loaded, max_tokens=5, temperature=1.0, top_p=1.0)
