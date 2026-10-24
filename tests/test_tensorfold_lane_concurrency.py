"""Concurrent request boundaries, without loading a model."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from rapid_mlx.request import RequestOutput
from rapid_mlx.speculative.tensorfold_qwen27_server import TensorFoldRequestProvider


@pytest.mark.parametrize("value", ["2", "4", "8", "3", "", " 1", "2.0"])
@pytest.mark.parametrize("backend", [None, "native"])
def test_cli_refuses_tensorfold_lanes_off_lane(monkeypatch, capsys, value, backend):
    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import (
        LANES_ENV,
        QUALIFIED_PROFILES,
    )

    monkeypatch.setenv(LANES_ENV, value)
    args = SimpleNamespace(
        model="mlx-community/Qwen3.5-4B-4bit",
        _original_alias="qwen3.5-4b-4bit",
        _speculative_config=SimpleNamespace(backend=backend) if backend else None,
    )
    with pytest.raises(SystemExit) as exc:
        cli._reject_tensorfold_lanes_off_lane_or_exit(args)
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert output.err.startswith("error: ")
    assert LANES_ENV in output.err
    assert "only on qualified TensorFold profiles" in output.err
    assert ", ".join(sorted(QUALIFIED_PROFILES)) in output.err
    assert args._original_alias in output.err
    profiles = ", ".join(sorted(QUALIFIED_PROFILES))
    if value in {"2", "4", "8"}:
        assert output.err == (
            f"error: {LANES_ENV}={value} enables concurrent decoding only on "
            f"qualified TensorFold profiles ({profiles}); {args._original_alias!r} "
            f"is not served through one. Unset {LANES_ENV} or set it to 1.\n"
        )
    else:
        assert output.err == (
            f"error: {LANES_ENV} must be 1, 2, 4 or 8 (got {value!r}). "
            f"It enables concurrent decoding only on qualified TensorFold profiles "
            f"({profiles}); unset it or set it to 1 for {args._original_alias!r}.\n"
        )


@pytest.mark.parametrize("value", [None, "1"])
def test_cli_serial_lanes_are_a_noop(monkeypatch, capsys, value):
    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    monkeypatch.delenv(LANES_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(LANES_ENV, value)
    cli._reject_tensorfold_lanes_off_lane_or_exit(SimpleNamespace(model="ordinary"))
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("value", ["4", "3", ""])
def test_cli_tensorfold_lane_delegates_validation(monkeypatch, capsys, value):
    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    monkeypatch.setenv(LANES_ENV, value)
    cli._reject_tensorfold_lanes_off_lane_or_exit(
        SimpleNamespace(_speculative_config=SimpleNamespace(backend="tensorfold"))
    )
    assert capsys.readouterr() == ("", "")


def test_cli_qualified_spelling_off_lane_still_refuses(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import (
        LANES_ENV,
        QUALIFIED_PROFILES,
    )

    model = sorted(QUALIFIED_PROFILES)[0]
    monkeypatch.setenv(LANES_ENV, "4")
    with pytest.raises(SystemExit) as exc:
        cli._reject_tensorfold_lanes_off_lane_or_exit(SimpleNamespace(model=model))
    assert exc.value.code == 2
    assert model in capsys.readouterr().err


@pytest.mark.parametrize("value", ["4", "3"])
def test_serve_refuses_tensorfold_lanes_before_download(monkeypatch, capsys, value):
    from unittest.mock import Mock

    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    monkeypatch.setenv(LANES_ENV, value)
    download = Mock(side_effect=AssertionError("must refuse before downloading"))
    next_preflight = Mock(
        side_effect=AssertionError("must refuse after lane resolution")
    )
    monkeypatch.setattr(cli, "_ensure_model_downloaded", download)
    monkeypatch.setattr(cli, "_reject_unsupported_listen_fd_lane", next_preflight)
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(SimpleNamespace(model="qwen3.5-4b-4bit"))
    assert exc.value.code == 2
    download.assert_not_called()
    next_preflight.assert_not_called()
    output = capsys.readouterr()
    assert not output.out
    assert LANES_ENV in output.err


@pytest.mark.parametrize(
    ("flag", "served"),
    [
        ("embedding_model", "mlx-community/all-MiniLM-L6-v2-4bit"),
        ("rerank_model", "mlx-community/Qwen3-Reranker-0.6B-mxfp8"),
    ],
)
@pytest.mark.parametrize("has_model", [False, True])
@pytest.mark.parametrize("value", ["4", None])
def test_retrieval_only_serve_tensorfold_lanes(
    monkeypatch, capsys, has_model, value, flag, served
):
    from argparse import Namespace
    from unittest.mock import Mock

    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    monkeypatch.delenv(LANES_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(LANES_ENV, value)
    args = Namespace(**{flag: served})
    if has_model:
        args.model = None
    embed_only = Mock()
    resolve_port = Mock(return_value=cli.DEFAULT_SERVE_PORT)
    monkeypatch.setattr(cli, "_serve_retrieval_only_mode", embed_only)
    monkeypatch.setattr(cli, "_resolve_serve_port", resolve_port)

    if value is not None:
        with pytest.raises(SystemExit) as exc:
            cli.serve_command(args)
        assert exc.value.code == 2
        embed_only.assert_not_called()
        resolve_port.assert_not_called()
        output = capsys.readouterr()
        assert not output.out
        assert f"{LANES_ENV}=4" in output.err
        assert repr(served) in output.err
    else:
        cli.serve_command(args)
        resolve_port.assert_called_once()
        embed_only.assert_called_once_with(args)
        assert args.port == cli.DEFAULT_SERVE_PORT
        assert capsys.readouterr() == ("", "")


def test_one_provider_keeps_nonstream_token_results_per_request(monkeypatch):
    provider = TensorFoldRequestProvider(SimpleNamespace())
    first_drained, second_drained = Event(), Event()

    def outputs(prompt, **_kwargs):
        tokens = [1, 2] if prompt == "first" else [3, 4, 5]
        if prompt == "second":
            assert first_drained.wait(3)
        yield RequestOutput(
            request_id=prompt,
            output_token_ids=tokens,
            output_text=prompt,
            prompt_tokens=7,
        )
        provider.last_token_ids = tokens
        if prompt == "first":
            first_drained.set()
            assert second_drained.wait(3)
        else:
            second_drained.set()

    monkeypatch.setattr(provider, "_outputs", outputs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(provider.generate, None, None, "first")
        second = pool.submit(provider.generate, None, None, "second")
        a, b = first.result(timeout=5), second.result(timeout=5)
    assert a.tokens == [1, 2]
    assert b.tokens == [3, 4, 5]
    assert (a.generation_tokens, b.generation_tokens) == (2, 3)


import asyncio
import threading

import httpx

from rapid_mlx.speculative.dflash import server
from rapid_mlx.speculative.tensorfold_qwen27_server import ProviderChunk, ProviderResult


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_execution_policy_rejects_invalid_capacity(capacity):
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(ValueError, match="capacity must be a positive integer"),
    ):
        server.BackendExecutionPolicy(capacity, executor)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("concurrent", [0, 2, 4])
async def test_http_execution_policy_preserves_default_and_capacity(
    monkeypatch, stream, concurrent
):
    monkeypatch.setattr(server, "_dflash_lock", asyncio.Lock())
    capacity = concurrent or 2
    entered, two_entered, release = Event(), Event(), Event()
    mutex = threading.Lock()
    active = peak = 0
    observed = []
    worker_names = []

    def begin():
        nonlocal active, peak
        with mutex:
            active += 1
            peak = max(peak, active)
            observed.append(server._dflash_lock.locked())
            worker_names.append(threading.current_thread().name)
            entered.set()
            if active == capacity:
                two_entered.set()
        assert release.wait(5)

    def end():
        nonlocal active
        with mutex:
            active -= 1

    def generate(*_args, **_kwargs):
        begin()
        try:
            return ProviderResult("ok", [7], 1, 2)
        finally:
            end()

    def chunks(*_args, **_kwargs):
        begin()
        try:
            yield ProviderChunk("ok", 7, 1, 2)
        finally:
            end()

    with ThreadPoolExecutor(max_workers=capacity) as executor:
        app = server._build_app(
            model=None,
            processor=SimpleNamespace(eos_token_id=99),
            runtime=SimpleNamespace(
                algorithm="mtp",
                drafter_repo=None,
                target_revision="a" * 40,
                drafter_revision=None,
            ),
            served_model_name="pilot",
            default_max_tokens=8,
            cors_origins=[],
            max_concurrent_requests=99 if concurrent else 2,
            execution_policy=server.BackendExecutionPolicy(capacity, executor)
            if concurrent
            else None,
            generate_fn=generate,
            stream_generate_fn=chunks,
            render_prompt_fn=lambda *_args, **_kwargs: "prompt",
            generation_kwargs_fn=lambda **kw: {"max_tokens": kw["max_tokens"]},
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            body = {
                "model": "pilot",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            }
            first = asyncio.create_task(client.post("/v1/chat/completions", json=body))
            assert await asyncio.to_thread(entered.wait, 3)
            remaining = [
                asyncio.create_task(client.post("/v1/chat/completions", json=body))
                for _ in range(capacity - 1)
            ]
            try:
                if concurrent:
                    assert await asyncio.to_thread(two_entered.wait, 3)
                else:
                    await asyncio.sleep(0.05)
                    assert not two_entered.is_set()
                overflow = await client.post("/v1/chat/completions", json=body)
                assert overflow.status_code == 503
                assert overflow.headers["Retry-After"] == "1"
                assert overflow.json()["error"]["code"] == "at_capacity"
            finally:
                release.set()
            responses = await asyncio.gather(first, *remaining)
            assert all(r.status_code == 200 for r in responses)
            if stream:
                assert all("data: [DONE]" in r.text for r in responses)
    assert peak == (capacity if concurrent else 1)
    assert observed == [not concurrent] * capacity
    if not concurrent:
        assert all(name.startswith("dflash-worker") for name in worker_names)
    assert not server._dflash_lock.locked()


@pytest.mark.parametrize("value", [True, 0, -1, 3, 9, "auto", "2.0"])
def test_invalid_lane_opt_in_is_refused(value):
    from rapid_mlx.speculative.tensorfold_concurrency import request_lanes

    with pytest.raises(ValueError, match="must be 1, 2, 4 or 8"):
        request_lanes("nemotron-3.5-lightning-tensorfold", value)


@pytest.mark.parametrize(
    "profile",
    [
        "bonsai2-27b-tensorfold",
        "gemma-4-26b-tensorfold",
        "qwen3.8-flash-next-tensorfold",
        "deepseek-v4-flash-tensorfold",
        "glm-5.3-flash-tensorfold",
    ],
)
def test_unmeasured_profile_refuses_lane_opt_in(profile):
    from rapid_mlx.speculative.tensorfold_concurrency import request_lanes

    assert request_lanes(profile, 1) == 1
    with pytest.raises(ValueError, match="has not qualified"):
        request_lanes(profile, 2)


def test_lane_opt_in_is_explicit_and_defaults_serial(monkeypatch):
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV, request_lanes

    monkeypatch.delenv(LANES_ENV, raising=False)
    assert request_lanes("nemotron-3.5-lightning-tensorfold") == 1
    monkeypatch.setenv(LANES_ENV, "4")
    assert request_lanes("nemotron-3.5-lightning-tensorfold") == 4


async def test_concurrent_timeout_retains_capacity_without_global_lock(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(server, "_dflash_lock", asyncio.Lock())
    release, started = Event(), Event()
    admission = server._DFlashAdmission(1)
    reservation = admission.reserve()
    reservation.claim()

    def generate(*_args, **_kwargs):
        started.set()
        assert release.wait(5)
        return ProviderResult("ok", [7], 1, 2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with pytest.raises(HTTPException) as exc:
                await server._non_stream_completion(
                    prompt="prompt",
                    request=SimpleNamespace(),
                    served_model_name="pilot",
                    gen_kwargs={},
                    model=None,
                    processor=SimpleNamespace(),
                    timeout=0.05,
                    admission_reservation=reservation,
                    generate_fn=generate,
                    execution_policy=server.BackendExecutionPolicy(1, executor),
                )
            assert exc.value.status_code == 504
            assert started.is_set()
            assert admission._reservations == 1
            assert not server._dflash_lock.locked()
        finally:
            release.set()
        for _ in range(100):
            if admission._reservations == 0:
                break
            await asyncio.sleep(0.01)
        assert admission._reservations == 0


@pytest.mark.parametrize("concurrent", [False, True])
async def test_cancelled_nonstream_retains_resources_until_worker_finishes(
    monkeypatch, concurrent
):
    from fastapi import HTTPException

    monkeypatch.setattr(server, "_dflash_lock", asyncio.Lock())
    release, started, finished = Event(), Event(), Event()
    admission = server._DFlashAdmission(1)
    reservation = admission.reserve()
    reservation.claim()

    def generate(*_args, **_kwargs):
        started.set()
        try:
            assert release.wait(5)
            return ProviderResult("ok", [7], 1, 2)
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(server, "_dflash_executor", executor)
        task = asyncio.create_task(
            server._non_stream_completion(
                prompt="prompt",
                request=SimpleNamespace(),
                served_model_name="pilot",
                gen_kwargs={},
                model=None,
                processor=SimpleNamespace(),
                timeout=0,
                admission_reservation=reservation,
                generate_fn=generate,
                execution_policy=server.BackendExecutionPolicy(1, executor)
                if concurrent
                else None,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            reservation.release()  # Request teardown must respect deferred ownership.
            assert not finished.is_set()
            assert admission._reservations == 1
            assert server._dflash_lock.locked() is (not concurrent)
            with pytest.raises(HTTPException) as exc:
                admission.reserve()
            assert exc.value.status_code == 503
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for _ in range(100):
            if admission._reservations == 0:
                break
            await asyncio.sleep(0.01)
        assert finished.is_set()
        assert admission._reservations == 0
        assert not server._dflash_lock.locked()
        replacement = admission.reserve()
        replacement.release()
        assert admission._reservations == 0


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("explicit", [False, True])
def test_tensorfold_bootstrap_aligns_load_http_capacity_and_capability(
    monkeypatch, lanes, explicit
):
    import rapid_mlx._uvicorn as uvicorn
    from rapid_mlx.speculative import tensorfold_qwen27_server as adapter
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    captured = {}

    class App:
        def on_event(self, _event):
            return lambda fn: fn

    monkeypatch.setattr(server, "_build_app", lambda **kw: captured.update(kw) or App())
    monkeypatch.setattr(uvicorn, "run_uvicorn", lambda *_args, **_kw: None)
    monkeypatch.setenv(LANES_ENV, str(lanes))
    with ThreadPoolExecutor(max_workers=lanes) as executor:
        backend = SimpleNamespace(
            _app=SimpleNamespace(tokenizer=object()),
            _executor=executor,
            close=lambda: None,
        )
        loader = SimpleNamespace(
            load=lambda *args, **kw: captured.update(load=kw) or backend
        )
        adapter.run_tensorfold_qwen27_server(
            main_model_repo="target",
            main_model_revision=None,
            drafter_repo="",
            drafter_revision=None,
            host="127.0.0.1",
            port=1,
            port_explicit=True,
            served_model_name="pilot",
            default_max_tokens=256,
            cors_origins=[],
            uvicorn_log_level="warning",
            backend_class=loader,
            profile_id="nemotron-3.5-lightning-tensorfold",
            tensorfold_lanes=lanes if explicit else None,
        )
    assert captured["load"].get("lanes", 1) == lanes
    assert captured["max_concurrent_requests"] == lanes
    assert captured["generate_fn_accepts_cancel_event"] is True
    policy = captured["execution_policy"]
    assert (policy.capacity if policy else 1) == lanes
    if lanes == 1:
        assert policy is None
        assert "lanes" not in captured["load"]
    assert (
        captured["runtime_status_extra"]["profile"]["capabilities"]["max_concurrency"]
        == lanes
    )


def test_execution_policy_cannot_unlock_default_gpu_callbacks():
    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(ValueError, match="both generation callbacks"),
    ):
        server._build_app(
            model=None,
            processor=None,
            runtime=None,
            served_model_name="pilot",
            default_max_tokens=1,
            cors_origins=[],
            execution_policy=server.BackendExecutionPolicy(2, executor),
        )


@pytest.fixture
def stalled_provider(monkeypatch, request):
    """A scheduler stop leaves jobs without terminal chunks, as in 0.6.6."""
    import queue
    import sys
    from types import ModuleType

    from rapid_mlx.speculative import tensorfold_qwen27_server as adapter
    from rapid_mlx.speculative.tensorfold_qwen27 import TensorFoldQwen27Backend

    class Cancellation:
        def __init__(self):
            self.event = Event()

        def cancel(self):
            self.event.set()

        def check(self):
            if self.event.is_set():
                raise RuntimeError("cancelled")

    class ChatJob:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.chunks = queue.Queue()
            self.cached_tokens = 0
            self.error = self.stream = None

    class StopPolicy:
        def __init__(self, *_args):
            self.ignore_eos = False
            self.strings = ()
            self.eos_ids = set()

        def visible(self, text, partial=False):
            return text

    for name, attrs in {
        "tensorfold.engine.lane_engine": {"SuffixLookupProposer": lambda **_: None},
        "tensorfold.server.cancellation": {"Cancellation": Cancellation},
        "tensorfold.server.scheduler": {"ChatJob": ChatJob},
        "tensorfold.server.stopping": {"StopPolicy": StopPolicy},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(adapter, "_StableIncrementalText", lambda *_: None)
    submitted, stopped = Event(), Event()
    jobs = []

    def submit(job):
        jobs.append(job)
        submitted.set()

    app = SimpleNamespace(
        max_batch_size=2,
        tokenizer=SimpleNamespace(encode=lambda _: [1]),
        tokenizer_lock=threading.Lock(),
        stop_ids=set(),
        min_match=1,
        _resolve_sampling=lambda *_: None,
        scheduler=SimpleNamespace(submit=submit, stop=stopped.set),
    )
    executor = (
        ThreadPoolExecutor(max_workers=2) if getattr(request, "param", False) else None
    )
    backend = TensorFoldQwen27Backend(app, executor=executor)
    provider = TensorFoldRequestProvider(backend)
    yield backend, provider, submitted, stopped, jobs
    for job in jobs:
        job.cancellation.cancel()
    backend._executor.shutdown(wait=True)


@pytest.mark.parametrize("stream", [False, True])
def test_close_terminates_provider_without_scheduler_terminal(stalled_provider, stream):
    backend, provider, submitted, stopped, jobs = stalled_provider
    operation = (
        (lambda: next(provider._outputs("prompt")))
        if stream
        else (lambda: provider.generate(None, None, "prompt"))
    )
    future = backend._executor.submit(operation)
    assert submitted.wait(2)
    closed = Event()

    def close():
        backend.close()
        closed.set()

    thread = threading.Thread(target=close, daemon=True)
    thread.start()
    try:
        assert closed.wait(0.5), (
            "close hung after scheduler stopped without terminal chunks"
        )
        assert stopped.is_set()
        assert future.done()
        assert not backend._active
    finally:
        # Even on the unfixed implementation, release the worker and bounded test.
        for job in jobs:
            job.cancellation.cancel()
        thread.join(2)


async def test_close_releases_deferred_nonstream_reservation(stalled_provider):
    from fastapi import HTTPException

    backend, provider, submitted, _, _ = stalled_provider
    admission = server._DFlashAdmission(2)
    reservation = admission.reserve()
    reservation.claim()
    with pytest.raises(HTTPException) as exc:
        await server._non_stream_completion(
            prompt="prompt",
            request=SimpleNamespace(),
            served_model_name="pilot",
            gen_kwargs={},
            model=None,
            processor=SimpleNamespace(),
            timeout=0.05,
            admission_reservation=reservation,
            generate_fn=provider.generate,
            execution_policy=server.BackendExecutionPolicy(2, backend._executor),
        )
    assert exc.value.status_code == 504
    assert submitted.is_set()
    assert admission._reservations == 1
    await asyncio.wait_for(asyncio.to_thread(backend.close), 1)
    for _ in range(100):
        if not admission._reservations:
            break
        await asyncio.sleep(0.01)
    assert admission._reservations == 0


def test_close_terminates_scheduler_and_executor_queued_jobs(stalled_provider):
    backend, provider, submitted, _, jobs = stalled_provider
    futures = [
        backend._executor.submit(provider.generate, None, None, "prompt")
        for _ in range(4)
    ]
    assert submitted.wait(2)
    deadline = __import__("time").monotonic() + 2
    while len(jobs) < 2 and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    assert len(jobs) == 2  # Two scheduler jobs; two executor tasks still queued.
    thread = threading.Thread(target=backend.close, daemon=True)
    thread.start()
    try:
        thread.join(1)
        assert not thread.is_alive()
        assert all(f.done() for f in futures)
        assert all(f.exception() is not None for f in futures)
        assert len(jobs) == 2
        assert not backend._active
    finally:
        for job in jobs:
            job.cancellation.cancel()
        thread.join(2)


def test_provider_unregisters_after_scheduler_submission_failure(stalled_provider):
    backend, provider, _, _, _ = stalled_provider

    def refuse(_job):
        raise RuntimeError("submission refused")

    backend._app.scheduler.submit = refuse
    with pytest.raises(RuntimeError, match="submission refused"):
        provider.generate(None, None, "prompt")
    assert not backend._active
    backend.close()


def test_close_cancels_job_registered_before_scheduler_submission(stalled_provider):
    backend, provider, _, stopped, jobs = stalled_provider
    submit_entered, resume_submit = Event(), Event()

    def submit(job):
        jobs.append(job)
        submit_entered.set()
        assert resume_submit.wait(2), "submit was not released after close returned"

    backend._app.scheduler.submit = submit
    executor = ThreadPoolExecutor(max_workers=1)
    close_errors = []
    close_thread = None

    def close():
        try:
            backend.close()
        except Exception as exc:
            close_errors.append(exc)

    try:
        future = executor.submit(provider.generate, None, None, "prompt")
        assert submit_entered.wait(1)
        close_thread = threading.Thread(target=close, daemon=True)
        close_thread.start()
        close_thread.join(1)
        assert not close_thread.is_alive(), "close waited for scheduler submission"
        assert not close_errors
        assert stopped.is_set()

        resume_submit.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            future.result(timeout=1)
        assert not backend._active
        assert jobs[0].cancellation.event.is_set()
    finally:
        resume_submit.set()
        for job in jobs:
            job.cancellation.cancel()
            job.chunks.put(None)
        if close_thread is not None:
            close_thread.join(2)
        executor.shutdown(wait=True, cancel_futures=True)


def test_serial_provider_close_releases_registered_worker(stalled_provider):
    backend, provider, submitted, stopped, jobs = stalled_provider
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(provider.generate, None, None, "prompt")
        assert submitted.wait(2)
        assert len(backend._active) == 1
        backend.close()
        assert stopped.is_set()
        with pytest.raises(RuntimeError, match="cancelled"):
            future.result(timeout=1)
        assert jobs[0].cancellation.event.is_set()
        assert not backend._active


async def test_close_releases_deferred_stream_reservation(stalled_provider):
    backend, provider, submitted, _, _ = stalled_provider
    admission = server._DFlashAdmission(2)
    reservation = admission.reserve()
    reservation.claim()
    request = server.ChatCompletionRequest(
        model="pilot", messages=[{"role": "user", "content": "hi"}]
    )
    chunks = [
        chunk
        async for chunk in server._stream_completion(
            prompt="prompt",
            request=request,
            served_model_name="pilot",
            gen_kwargs={},
            model=None,
            processor=SimpleNamespace(),
            timeout=0.05,
            admission_reservation=reservation,
            stream_generate_fn=provider.stream_generate,
            execution_policy=server.BackendExecutionPolicy(2, backend._executor),
        )
    ]
    assert b"timed out" in b"".join(chunks)
    assert submitted.is_set()
    assert admission._reservations == 1
    # Match the synchronous shutdown hook: callbacks run after the owned
    # executor has joined its cancelled workers and stopped accepting tasks.
    backend.close()
    for _ in range(100):
        if not admission._reservations:
            break
        await asyncio.sleep(0.01)
    assert admission._reservations == 0


@pytest.mark.parametrize("started", [False, True])
async def test_concurrent_closed_or_unstarted_generator_cleanup_after_executor_join(
    started,
):
    def chunks():
        yield "chunk"

    generator = chunks()
    if started:
        assert next(generator) == "chunk"
        with pytest.raises(StopIteration):
            next(generator)
    executor = ThreadPoolExecutor(max_workers=1)
    policy = server.BackendExecutionPolicy(1, executor)
    executor.shutdown(wait=True)
    admission = server._DFlashAdmission(1)
    reservation = admission.reserve()
    reservation.claim()
    lease = server._DFlashStreamLease(
        asyncio.get_running_loop(), reservation, None, policy
    )
    lease.set_generator(generator)
    await lease._close_generator_and_release()
    assert admission._reservations == 0


def test_close_unstarted_tracked_iterator_releases_registry(stalled_provider):
    backend, provider, _, _, _ = stalled_provider

    class Inner:
        close_count = 0

        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("unstarted iterator must not be pulled")

        def close(self):
            self.close_count += 1

    inner = Inner()
    iterator = provider._track_iterator(inner)
    assert backend._provider_iterators == {iterator}
    iterator.close()
    assert not backend._provider_iterators
    assert inner.close_count == 1
    iterator.close()
    assert inner.close_count == 1
    with pytest.raises(StopIteration):
        next(iterator)


def test_close_and_provider_registration_are_atomic(stalled_provider):
    backend, provider, _, stopped, jobs = stalled_provider
    preparing, resume = Event(), Event()

    def encode(_prompt):
        preparing.set()
        assert resume.wait(2)
        return [1]

    backend._app.tokenizer.encode = encode
    future = backend._executor.submit(provider.generate, None, None, "prompt")
    assert preparing.wait(2)
    thread = threading.Thread(target=backend.close, daemon=True)
    thread.start()
    try:
        assert stopped.wait(1)
        resume.set()
        thread.join(1)
        assert not thread.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            future.result(timeout=1)
        assert not jobs
        assert not backend._active
    finally:
        resume.set()
        thread.join(2)


@pytest.mark.parametrize(
    "stalled_provider", [False, True], indirect=True, ids=["owned", "injected"]
)
def test_close_finalizes_provider_suspended_between_stream_pulls(
    stalled_provider, monkeypatch
):
    backend, provider, submitted, _, jobs = stalled_provider
    from rapid_mlx.speculative import tensorfold_qwen27_server as adapter

    class Decoder:
        def __init__(self, *_):
            pass

        def extend(self, _):
            return ""

    monkeypatch.setattr(adapter, "_StableIncrementalText", Decoder)
    iterator = provider.stream_generate(None, None, "prompt")
    future = backend._executor.submit(next, iterator)
    assert submitted.wait(2)
    jobs[0].chunks.put([7])
    assert future.result(timeout=1).token == 7
    backend.close()
    try:
        assert not backend._active, "suspended provider keeps its job/model after close"
        assert not backend._provider_iterators
        assert backend._app is None
        if not backend._owns_executor:
            assert (
                backend._executor.submit(lambda: "usable").result(timeout=1) == "usable"
            )
        import inspect

        assert inspect.getgeneratorstate(iterator) == inspect.GEN_CLOSED
    finally:
        iterator.close()


def test_close_continues_after_tracked_iterator_cleanup_failure(
    stalled_provider, caplog
):
    backend, provider, _, stopped, _ = stalled_provider

    class Inner:
        fail = False
        closed = False

        def close(self):
            self.closed = True
            if self.fail:
                raise OSError("audit write failed")

    inners = [Inner() for _ in range(3)]
    iterators = {provider._track_iterator(inner): inner for inner in inners}
    # Fail the first close in the registry's iteration order, so continuing
    # cleanup is required to finalize either of the remaining iterators.
    iterators[next(iter(backend._provider_iterators))].fail = True
    try:
        backend.close()
        assert all(inner.closed for inner in inners)
        assert not backend._provider_iterators
        assert backend._app is None
        assert stopped.is_set()
        assert "audit write failed" in caplog.text
    finally:
        for iterator in iterators:
            iterator.close()


@pytest.mark.parametrize("stalled_provider", [True], indirect=True)
def test_close_with_injected_executor_leaves_running_iterator_cleanup_to_pull(
    stalled_provider,
):
    import inspect
    import queue

    backend, provider, submitted, stopped, jobs = stalled_provider
    entered, release = Event(), Event()

    def get(**_kwargs):
        entered.set()
        assert release.wait(3)
        raise queue.Empty

    def submit(job):
        job.chunks.get = get
        jobs.append(job)
        submitted.set()

    backend._app.scheduler.submit = submit
    iterator = provider._outputs("prompt")
    future = backend._executor.submit(next, iterator)
    try:
        assert entered.wait(2)
        backend.close()
        assert stopped.is_set()
        assert jobs[0].cancellation.event.is_set()
        assert backend._app is None
        assert not backend._provider_iterators
        assert not future.done()
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="cancelled"):
            future.result(timeout=1)
    assert inspect.getgeneratorstate(iterator) == inspect.GEN_CLOSED
    assert not backend._active
    assert not backend._provider_iterators
    with pytest.raises(RuntimeError, match="closed"):
        provider._outputs("after close")
    assert not backend._provider_iterators
    assert backend._executor.submit(lambda: "usable").result(timeout=1) == "usable"


def test_client_stream_close_finalizes_inner_tracked_job(stalled_provider, monkeypatch):
    backend, provider, submitted, _, jobs = stalled_provider
    from rapid_mlx.speculative import tensorfold_qwen27_server as adapter

    class Decoder:
        def __init__(self, *_):
            pass

        def extend(self, _):
            return ""

    monkeypatch.setattr(adapter, "_StableIncrementalText", Decoder)
    iterator = provider.stream_generate(None, None, "prompt")
    future = backend._executor.submit(next, iterator)
    assert submitted.wait(2)
    jobs[0].chunks.put([7])
    assert future.result(timeout=1).token == 7
    iterator.close()
    try:
        assert not backend._active
        assert not backend._provider_iterators
        assert jobs[0].cancellation.event.is_set()
    finally:
        backend.close()


async def test_serial_timeout_then_close_releases_worker(stalled_provider, monkeypatch):
    from fastapi import HTTPException

    backend, provider, submitted, stopped, jobs = stalled_provider
    lock = asyncio.Lock()
    monkeypatch.setattr(server, "_dflash_lock", lock)

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self):
            super().__init__(max_workers=1)
            self.futures = []

        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            self.futures.append(future)
            return future

    serial_executor = RecordingExecutor()
    monkeypatch.setattr(server, "_dflash_executor", serial_executor)
    admission = server._DFlashAdmission(1)
    reservation = admission.reserve()
    reservation.claim()
    try:
        with pytest.raises(HTTPException) as exc:
            await server._non_stream_completion(
                prompt="prompt",
                request=SimpleNamespace(),
                served_model_name="pilot",
                gen_kwargs={},
                model=None,
                processor=SimpleNamespace(),
                timeout=0.05,
                admission_reservation=reservation,
                generate_fn=provider.generate,
            )
        assert exc.value.status_code == 504
        assert submitted.is_set()
        assert len(backend._active) == 1
        assert len(serial_executor.futures) == 1

        await asyncio.wait_for(asyncio.to_thread(backend.close), 1)
        assert stopped.is_set()
        for _ in range(100):
            if not admission._reservations and not lock.locked():
                break
            await asyncio.sleep(0.01)
        assert not lock.locked(), "serial lock still held after timeout + close"
        assert admission._reservations == 0, "reservation still deferred"
        assert serial_executor.futures[0].done()
        assert not backend._active
    finally:
        for job in jobs:
            job.cancellation.cancel()
        await asyncio.to_thread(serial_executor.shutdown, wait=True)


@pytest.mark.parametrize("concurrent", [False, True])
async def test_nonstream_timeout_cancel_event_releases_worker(
    stalled_provider, monkeypatch, concurrent
):
    from fastapi import HTTPException

    backend, provider, submitted, _, jobs = stalled_provider
    lock = asyncio.Lock()
    monkeypatch.setattr(server, "_dflash_lock", lock)
    serial_executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(server, "_dflash_executor", serial_executor)
    admission = server._DFlashAdmission(1)
    reservation = admission.reserve()
    reservation.claim()
    policy = server.BackendExecutionPolicy(1, backend._executor) if concurrent else None
    gen_kwargs = {"max_tokens": 8}
    try:
        with pytest.raises(HTTPException) as exc:
            await server._non_stream_completion(
                prompt="prompt",
                request=SimpleNamespace(),
                served_model_name="pilot",
                gen_kwargs=gen_kwargs,
                model=None,
                processor=SimpleNamespace(),
                timeout=0.05,
                admission_reservation=reservation,
                generate_fn=provider.generate,
                generate_fn_accepts_cancel_event=True,
                execution_policy=policy,
            )
        assert exc.value.status_code == 504
        assert submitted.is_set()
        for _ in range(100):
            if not admission._reservations and not backend._active:
                break
            await asyncio.sleep(0.01)
        assert gen_kwargs == {"max_tokens": 8}
        assert jobs[0].cancellation.event.is_set()
        assert not backend._active
        assert admission._reservations == 0
        assert not lock.locked()
        if not concurrent:
            response = await asyncio.wait_for(
                server._non_stream_completion(
                    prompt="follow-up",
                    request=server.ChatCompletionRequest(
                        model="pilot",
                        messages=[{"role": "user", "content": "hi"}],
                    ),
                    served_model_name="pilot",
                    gen_kwargs={},
                    model=None,
                    processor=SimpleNamespace(),
                    timeout=1,
                    generate_fn=lambda *_args: ProviderResult("ok", [7], 1, 2),
                ),
                1,
            )
            assert response.choices[0].message.content == "ok"
    finally:
        for job in jobs:
            job.cancellation.cancel()
        backend.close()
        await asyncio.to_thread(serial_executor.shutdown, wait=True)


async def test_cancelled_nonstream_sets_tensorfold_cancel_event(
    stalled_provider, monkeypatch
):
    backend, provider, submitted, _, jobs = stalled_provider
    lock = asyncio.Lock()
    monkeypatch.setattr(server, "_dflash_lock", lock)
    serial_executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(server, "_dflash_executor", serial_executor)
    admission = server._DFlashAdmission(1)
    reservation = admission.reserve()
    reservation.claim()
    task = asyncio.create_task(
        server._non_stream_completion(
            prompt="prompt",
            request=SimpleNamespace(),
            served_model_name="pilot",
            gen_kwargs={},
            model=None,
            processor=SimpleNamespace(),
            timeout=0,
            admission_reservation=reservation,
            generate_fn=provider.generate,
            generate_fn_accepts_cancel_event=True,
        )
    )
    try:
        assert await asyncio.to_thread(submitted.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):
            if not admission._reservations and not backend._active:
                break
            await asyncio.sleep(0.01)
        assert jobs[0].cancellation.event.is_set()
        assert admission._reservations == 0
        assert not backend._active
        assert not lock.locked()
    finally:
        for job in jobs:
            job.cancellation.cancel()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        backend.close()
        await asyncio.to_thread(serial_executor.shutdown, wait=True)


async def test_nonstream_default_does_not_pass_cancel_event(monkeypatch):
    monkeypatch.setattr(server, "_dflash_lock", asyncio.Lock())

    def generate(_model, _processor, _prompt):
        return ProviderResult("ok", [7], 1, 2)

    response = await server._non_stream_completion(
        prompt="prompt",
        request=server.ChatCompletionRequest(
            model="pilot", messages=[{"role": "user", "content": "hi"}]
        ),
        served_model_name="pilot",
        gen_kwargs={},
        model=None,
        processor=SimpleNamespace(),
        timeout=1,
        generate_fn=generate,
    )
    assert response.choices[0].message.content == "ok"


@pytest.mark.parametrize(
    "model",
    [
        "qwen3.8-27b-tensorfold",
        "nemotron-3.5-lightning-tensorfold",
        "Vontra/Qwen3.8-27B-MLX-4bit",
    ],
)
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_cli_explicit_lanes_override_legacy_env(monkeypatch, model, lanes):
    from rapid_mlx import cli
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    monkeypatch.setenv(LANES_ENV, "invalid")
    args = SimpleNamespace(
        model=model,
        tensorfold_lanes=lanes,
        _speculative_config=SimpleNamespace(backend="tensorfold"),
    )
    cli._reject_tensorfold_lanes_off_lane_or_exit(args)
    assert args.tensorfold_lanes == lanes


@pytest.mark.parametrize(
    "model, backend, lanes",
    [
        ("ordinary", None, 4),
        (None, None, 1),
        ("gemma-4-26b-tensorfold", "tensorfold", 1),
        ("qwen3.8-27b-tensorfold", "native", 4),
        ("qwen3.8-27b-tensorfold", "tensorfold", 3),
    ],
)
def test_cli_explicit_lanes_refuse_unqualified_or_invalid(
    capsys, model, backend, lanes
):
    from rapid_mlx import cli

    args = SimpleNamespace(
        model=model,
        tensorfold_lanes=lanes,
        _speculative_config=SimpleNamespace(backend=backend),
    )
    with pytest.raises(SystemExit) as exc:
        cli._reject_tensorfold_lanes_off_lane_or_exit(args)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_legacy_lanes_warn_and_explicit_lanes_win(monkeypatch):
    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV, request_lanes

    monkeypatch.setenv(LANES_ENV, "4")
    with pytest.warns(UserWarning, match="deprecated; use --tensorfold-lanes"):
        assert request_lanes("qwen3.8-27b-tensorfold") == 4
    assert request_lanes("qwen3.8-27b-tensorfold", 1) == 1


def test_legacy_lane_notice_visible_with_default_library_warning_filters():
    import os
    import subprocess
    import sys
    from pathlib import Path

    from rapid_mlx.speculative.tensorfold_concurrency import LANES_ENV

    environment = os.environ.copy()
    environment.pop("PYTHONWARNINGS", None)
    environment[LANES_ENV] = "4"
    library_call = (
        "from rapid_mlx.speculative.tensorfold_concurrency import request_lanes\n"
        "print(request_lanes('qwen3.8-27b-tensorfold'))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"exec({library_call!r}, {{'__name__': 'operator_warning_probe'}})",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "4"
    assert (
        "UserWarning: RAPID_MLX_TENSORFOLD_LANES is deprecated; "
        "use --tensorfold-lanes instead"
    ) in result.stderr
