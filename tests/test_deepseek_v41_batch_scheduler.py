"""CPU-capable scheduler and server contracts using synthetic weights only."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.models.deepseek_v41_native.batch_scheduler import (
    BatchBackend,
    request_lanes,
)
from rapid_mlx.models.deepseek_v41_native.cache import BatchModelCache, ModelCache
from scripts.bench_deepseek_v41_batched_decode import make_tiny_model


class Tokenizer:
    eos_token_id = -1
    clean_up_tokenization_spaces = False

    def encode(self, text, **kwargs):
        return [int(word) for word in text.split()]

    def decode(self, tokens):
        return "".join(chr(65 + token) for token in tokens)


class ObservedModel:
    def __init__(self, model):
        self.model = model
        self.calls = []
        self.hook = lambda ids, cache: None
        self.threads = set()

    def make_cache(self, **kwargs):
        self.threads.add(threading.get_ident())
        return self.model.make_cache(**kwargs)

    def __call__(self, ids, cache, **kwargs):
        self.threads.add(threading.get_ident())
        self.calls.append((ids.shape, isinstance(cache, BatchModelCache)))
        self.hook(ids, cache)
        return self.model(ids, cache, **kwargs)


@pytest.fixture
def model():
    return make_tiny_model()


def alone(model, prompt, count):
    cache = model.make_cache(max_seq_len=len(prompt) + count + 8)
    logits = model(mx.array([prompt]), cache, last_logit_only=True)[:, -1]
    tokens = []
    for _ in range(count):
        token = int(mx.argmax(logits).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache, last_logit_only=True)[:, -1]
    return tokens


def collect(stream):
    try:
        return list(stream)
    finally:
        stream.close()


def wait(event):
    assert event.wait(10), "scheduler did not reach the expected boundary"


@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_continuous_join_leave_exact_greedy_and_owner_thread(model, lanes, monkeypatch):
    prompts = [
        [(row * 3 + token) % 32 for token in range(2 + row)] for row in range(lanes + 1)
    ]
    limits = [16] + [2 + row % 5 for row in range(lanes)]
    expected = [alone(model, prompt, count) for prompt, count in zip(prompts, limits)]
    prefetched = []
    for layer in model.layers:
        if layer.engram is not None:
            monkeypatch.setattr(
                layer.engram.embed,
                "prefetch",
                lambda indices: prefetched.append(indices.copy()),
                raising=False,
            )
    observed = ObservedModel(model)
    entered, release = threading.Event(), threading.Event()
    joined = False

    def hook(ids, cache):
        nonlocal joined
        if isinstance(cache, BatchModelCache) and not joined:
            joined = True
            entered.set()
            wait(release)

    observed.hook = hook
    backend = BatchBackend(observed, Tokenizer(), lanes=lanes)
    try:
        first = backend.submit(prompts[0], max_tokens=limits[0])
        wait(entered)
        streams = [first] + [
            backend.submit(prompt, max_tokens=count)
            for prompt, count in zip(prompts[1:], limits[1:])
        ]
        release.set()
        with ThreadPoolExecutor(lanes + 1) as pool:
            futures = [pool.submit(collect, stream) for stream in streams]
            chunks = [future.result(timeout=20) for future in futures]
        assert [[chunk.token for chunk in stream] for stream in chunks] == expected
        assert [stream[-1].generation_tokens for stream in chunks] == limits
        assert any(shape[0] > 1 and batch for shape, batch in observed.calls)
        assert all(shape[0] <= lanes for shape, batch in observed.calls if batch)
        assert any(indices.shape[0] > 1 for indices in prefetched)
        assert len(observed.threads) == 1
        assert threading.get_ident() not in observed.threads
    finally:
        release.set()
        backend.close()


class CountingModel:
    """Cheap cache-real model for long prefill, EOS and failure injection."""

    def __init__(self):
        self.args = replace(make_tiny_model().args, n_layers=0, max_seq_len=16384)
        self.calls = []
        self.hook = lambda ids, cache: None
        self.next_token = 3

    def make_cache(self, **kwargs):
        return ModelCache(self.args, **kwargs)

    def __call__(self, ids, cache, **kwargs):
        self.calls.append((ids.shape, isinstance(cache, BatchModelCache)))
        self.hook(ids, cache)
        if isinstance(cache, BatchModelCache):
            cache.advance()
            shape = (ids.shape[0], 32)
        else:
            cache.offset += ids.shape[1]
            shape = (ids.shape[0], 1, 32)
        logits = mx.zeros(shape)
        logits[..., self.next_token] = 1
        return logits


@pytest.mark.parametrize("phase", ["prefill", "decode", "unread"])
def test_cancel_frees_lane(phase):
    model = CountingModel()
    entered, release = threading.Event(), threading.Event()
    cancel = threading.Event()
    blocked = False

    def hook(ids, cache):
        nonlocal blocked
        if not blocked and (isinstance(cache, BatchModelCache) == (phase == "decode")):
            blocked = True
            entered.set()
            wait(release)

    model.hook = hook
    backend = BatchBackend(model, Tokenizer(), lanes=1)
    try:
        stream = backend.submit([1] * 1026, max_tokens=4096, cancel_event=cancel)
        wait(entered)
        following = backend.submit([1, 2], max_tokens=3)
        if phase == "unread":
            stream.close()
        else:
            cancel.set()
        release.set()
        with ThreadPoolExecutor() as pool:
            result = pool.submit(collect, following).result(timeout=10)
            if phase != "unread":
                cancelled = pool.submit(collect, stream).result(timeout=10)
                # cancelled mid-step: the step's token is never delivered
                # (only the first token from the finished prefill, if any)
                emitted = [
                    chunk.token for chunk in cancelled if chunk.token is not None
                ]
                assert len(emitted) == (1 if phase == "decode" else 0)
        assert [chunk.token for chunk in result] == [3] * 3
        # A cancelled prefill never runs its second chunk.
        if phase != "decode":
            assert [shape[1] for shape, batch in model.calls if not batch] == [512, 2]
    finally:
        release.set()
        backend.close()


def test_cancel_during_final_prefill_chunk_emits_no_token():
    model = CountingModel()
    entered, release = threading.Event(), threading.Event()
    cancel = threading.Event()
    blocked = False

    def hook(ids, cache):
        nonlocal blocked
        if not blocked and not isinstance(cache, BatchModelCache):
            blocked = True
            entered.set()
            wait(release)

    model.hook = hook
    backend = BatchBackend(model, Tokenizer(), lanes=1)
    try:
        # one-chunk prompt: the cancel lands while its only prefill step runs
        stream = backend.submit([1, 2], max_tokens=8, cancel_event=cancel)
        wait(entered)
        cancel.set()
        release.set()
        cancelled = collect(stream)
        assert [chunk.token for chunk in cancelled if chunk.token is not None] == []
        # the lane is free and no decode step ran for the cancelled row
        assert not any(batch for shape, batch in model.calls)
        assert [
            chunk.token for chunk in collect(backend.submit([1, 2], max_tokens=2))
        ] == [3, 3]
    finally:
        release.set()
        backend.close()


@pytest.mark.parametrize("prefill_failure", [False, True])
def test_step_error_is_delivered_and_backend_recovers(prefill_failure, caplog):
    model = CountingModel()
    entered, release = threading.Event(), threading.Event()
    first_step = True
    failed = False

    def hook(ids, cache):
        nonlocal first_step, failed
        if first_step:
            first_step = False
            entered.set()
            wait(release)
        batched = isinstance(cache, BatchModelCache)
        if not failed and (
            (not batched and prefill_failure) or (batched and len(cache) == 2)
        ):
            failed = True
            raise RuntimeError("injected step failure")

    model.hook = hook
    backend = BatchBackend(model, Tokenizer(), lanes=2)
    try:
        streams = [backend.submit([1, 2], max_tokens=20)]
        wait(entered)
        streams.append(backend.submit([3, 4], max_tokens=20))
        release.set()
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(collect, stream) for stream in streams]
            with pytest.raises(RuntimeError, match="injected step"):
                futures[0].result(timeout=10)
            if prefill_failure:
                assert len(futures[1].result(timeout=10)) == 20
            else:
                with pytest.raises(RuntimeError, match="injected step"):
                    futures[1].result(timeout=10)
            subsequent = backend.submit([5, 6], max_tokens=3)
            assert len(pool.submit(collect, subsequent).result(timeout=10)) == 3
        assert (
            len([record for record in caplog.records if "failed" in record.message])
            == 1
        )
    finally:
        release.set()
        backend.close()


def test_emission_error_fails_only_its_own_row():
    model = CountingModel()
    backend = BatchBackend(model, Tokenizer(), lanes=2)
    prepare = backend._prepare

    def poisoned_prepare(request):
        prepare(request)
        if request.ids[0] == 3:
            add_token = request.detokenizer.add_token

            def add(token):
                if request.generated == 3:
                    raise RuntimeError("injected detokenizer failure")
                add_token(token)

            request.detokenizer.add_token = add

    backend._prepare = poisoned_prepare
    try:
        streams = [
            backend.submit([1, 2], max_tokens=20),
            backend.submit([3, 4], max_tokens=20),
        ]
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(collect, stream) for stream in streams]
            with pytest.raises(RuntimeError, match="injected detokenizer"):
                futures[1].result(timeout=10)
            assert len(futures[0].result(timeout=10)) == 20
    finally:
        backend.close()


def test_failed_construction_releases_both_executors(monkeypatch):
    created = []
    real = ThreadPoolExecutor

    class Recording(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def submit(self, fn, *args, **kwargs):
            if getattr(fn, "__name__", "") == "_run":
                raise RuntimeError("injected start failure")
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(
        "rapid_mlx.models.deepseek_v41_native.batch_scheduler.ThreadPoolExecutor",
        Recording,
    )
    with pytest.raises(RuntimeError, match="injected start"):
        BatchBackend(CountingModel(), Tokenizer(), lanes=2)
    assert len(created) == 2
    assert all(executor._shutdown for executor in created)


def test_chunked_prefill_and_load_share_model_thread():
    threads = []
    model = CountingModel()
    model.hook = lambda *_: threads.append(threading.get_ident())

    def load():
        threads.append(threading.get_ident())
        return model, Tokenizer(), SimpleNamespace()

    backend = BatchBackend(lanes=2, load_fn=load)
    try:
        with ThreadPoolExecutor() as pool:
            result = pool.submit(
                collect, backend.submit([1] * 1026, max_tokens=3)
            ).result(timeout=10)
        assert [shape[1] for shape, batch in model.calls if not batch] == [512, 512, 2]
        assert len(result) == 3
        assert len(set(threads)) == 1
    finally:
        backend.close()


@pytest.mark.parametrize(
    "eos,ignore_eos,count", [(-1, False, 5), (3, False, 1), (3, True, 5)]
)
def test_stream_nonstream_text_tokens_and_eos(eos, ignore_eos, count):
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = eos
    backend = BatchBackend(CountingModel(), tokenizer)
    try:
        chunks = collect(
            backend.stream_generate(
                None, None, "1 2", max_tokens=5, ignore_eos=ignore_eos
            )
        )
        result = backend.generate(
            None, None, [1, 2], max_tokens=5, ignore_eos=ignore_eos
        )
        assert [chunk.token for chunk in chunks] == [3] * count
        assert (
            result.text
            == "".join(chunk.text for chunk in chunks)
            == ("D" * count if eos != 3 else "")
        )
        assert result.generation_tokens == count
        assert result.prompt_tokens == 2
    finally:
        backend.close()


@pytest.mark.parametrize("limit", [0, -1, 4097])
def test_invalid_generation_limit(limit):
    backend = BatchBackend(CountingModel(), Tokenizer())
    try:
        with pytest.raises(HTTPException, match="max_tokens"):
            backend.submit([1, 2], max_tokens=limit)
    finally:
        backend.close()


@pytest.mark.parametrize("length", [1, 8193])
def test_prompt_bounds(length):
    backend = BatchBackend(CountingModel(), Tokenizer())
    try:
        with ThreadPoolExecutor() as pool:
            future = pool.submit(collect, backend.submit([1] * length, max_tokens=1))
            with pytest.raises((ValueError, HTTPException)):
                future.result(timeout=10)
    finally:
        backend.close()


def test_close_fails_pending_active_and_partial_prefill():
    model = CountingModel()
    entered, release = threading.Event(), threading.Event()

    def hook(*_):
        entered.set()
        wait(release)

    model.hook = hook
    backend = BatchBackend(model, Tokenizer(), lanes=1)
    streams = [backend.submit([1] * 1026, max_tokens=4096)]
    wait(entered)
    streams.append(backend.submit([1, 2], max_tokens=3))
    with ThreadPoolExecutor(3) as pool:
        closing = pool.submit(backend.close)
        release.set()
        closing.result(timeout=10)
        for stream in streams:
            with pytest.raises(RuntimeError, match="closed"):
                pool.submit(collect, stream).result(timeout=10)
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.submit([1, 2], max_tokens=3)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 1),
        ("1", 1),
        ("2", 2),
        ("4", 4),
        ("8", 8),
        ("0", None),
        ("3", None),
        ("", None),
        ("two", None),
    ],
)
def test_lanes_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("RAPID_MLX_DSV41_LANES", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_DSV41_LANES", value)
    if expected is None:
        with pytest.raises(ValueError, match="RAPID_MLX_DSV41_LANES"):
            request_lanes()
    else:
        assert request_lanes() == expected


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_server_selects_serial_or_batch_path(monkeypatch, capsys, lanes):
    from pathlib import Path

    from fastapi.testclient import TestClient

    from rapid_mlx.models.deepseek_v41_native import batch_scheduler, server
    from rapid_mlx.speculative.dflash import server as serial

    monkeypatch.setenv("RAPID_MLX_DSV41_LANES", str(lanes))
    monkeypatch.setattr(server, "require_product_memory", lambda: None)
    monkeypatch.setattr(server, "download_target_snapshot", lambda: Path("target"))
    monkeypatch.setattr(server, "download_mtp_snapshot", lambda: Path("mtp"))
    calls = {}
    runtime = SimpleNamespace(
        target_revision="target-revision",
        drafter_revision="draft-revision",
        algorithm="dspark-k4",
        drafter_repo="draft",
        drafter=object(),
        kind="dspark",
    )
    tokenizer = Tokenizer()

    def loader(*args, **kwargs):
        calls.setdefault("loads", []).append(threading.get_ident())
        return CountingModel(), tokenizer, runtime

    monkeypatch.setattr(server, "load_product_runtime", loader)
    original_backend = batch_scheduler.BatchBackend

    def build_backend(**kwargs):
        assert lanes > 1, "serial path must not construct a batch backend"
        backend = original_backend(**kwargs)
        calls["backend"] = backend
        return backend

    monkeypatch.setattr(batch_scheduler, "BatchBackend", build_backend)
    build_app = serial._build_app

    def capture_app(**kwargs):
        calls["app_kwargs"] = kwargs
        return build_app(**kwargs)

    monkeypatch.setattr(serial, "_build_app", capture_app)

    def run(app, **kwargs):
        kwargs["on_server_accepting"]()
        with TestClient(app, base_url="http://127.0.0.1") as client:
            calls["models"] = client.get("/v1/models").json()
            calls["health"] = client.get("/healthz").json()
            if lanes > 1:
                # Rendering is separately covered by the product tests. Exercise
                # both HTTP generation adapters with an exact rendered prompt.
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 3,
                    },
                )
                calls["response"] = response
                calls["stream_response"] = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 3,
                        "stream": True,
                    },
                )

    if lanes > 1:

        def capture_batch_app(**kwargs):
            kwargs["render_prompt_fn"] = lambda *_args, **_kwargs: "1 2"
            return capture_app(**kwargs)

        monkeypatch.setattr(serial, "_build_app", capture_batch_app)
    monkeypatch.setattr("rapid_mlx._uvicorn.run_uvicorn", run)
    server.run_server(
        host="127.0.0.1",
        port=8000,
        served_model_name="test",
        default_max_tokens=3,
        cors_origins=[],
        uvicorn_log_level="warning",
    )
    assert len(calls["loads"]) == 1
    options = calls["app_kwargs"]
    if lanes == 1:
        assert "execution_policy" not in options
        assert "generate_fn_accepts_cancel_event" not in options
        assert options["runtime"] is runtime
        assert options["max_concurrent_requests"] == 256
        assert "DSpark K4 serial mode" in capsys.readouterr().out
    else:
        backend = calls["backend"]
        assert options["execution_policy"].executor is backend.executor
        assert options["execution_policy"].capacity == lanes
        assert options["max_concurrent_requests"] == lanes
        assert options["generate_fn_accepts_cancel_event"] is True
        assert calls["models"]["data"][0]["speculative_decoding"]["configured"] is False
        assert calls["health"]["drafter"] is None
        assert calls["health"]["mode"] == "batched"
        assert calls["response"].status_code == 200, calls["response"].text
        assert calls["response"].json()["usage"]["completion_tokens"] == 3
        assert calls["stream_response"].status_code == 200
        assert "data: [DONE]" in calls["stream_response"].text
        assert backend._closed
        assert f"batched mode, {lanes} lanes" in capsys.readouterr().out


def test_finished_batch_cache_is_released_while_idle():
    import weakref

    model = CountingModel()
    freed = threading.Event()
    references = []

    def hook(ids, cache):
        if isinstance(cache, BatchModelCache):
            references.append(weakref.ref(cache, lambda _: freed.set()))

    model.hook = hook
    backend = BatchBackend(model, Tokenizer())
    try:
        assert len(collect(backend.submit([1, 2], max_tokens=2))) == 2
        wait(freed)
        assert all(reference() is None for reference in references)
    finally:
        backend.close()


def test_upper_prompt_and_generation_bounds_are_accepted():
    backend = BatchBackend(CountingModel(), Tokenizer())
    try:
        with ThreadPoolExecutor() as pool:
            result = pool.submit(
                collect, backend.submit([1] * 8192, max_tokens=4096)
            ).result(timeout=20)
        assert result[-1].prompt_tokens == 8192
        assert result[-1].generation_tokens == 4096
    finally:
        backend.close()


@pytest.mark.parametrize("lanes", [3, True])
def test_backend_rejects_unsupported_lane_counts(lanes):
    with pytest.raises(ValueError, match="lanes must be one of"):
        BatchBackend(CountingModel(), Tokenizer(), lanes=lanes)


def test_ended_stream_stays_ended():
    backend = BatchBackend(CountingModel(), Tokenizer(), lanes=2)
    try:
        stream = backend.submit([1, 2], max_tokens=2)
        assert len(collect(stream)) == 2
        with pytest.raises(StopIteration):
            next(stream)
    finally:
        backend.close()


def test_request_cancelled_while_queued_is_never_admitted():
    model = CountingModel()
    entered, release = threading.Event(), threading.Event()
    blocked = False

    def hook(ids, cache):
        nonlocal blocked
        if not blocked:
            blocked = True
            entered.set()
            wait(release)

    model.hook = hook
    backend = BatchBackend(model, Tokenizer(), lanes=1)
    try:
        first = backend.submit([1, 2], max_tokens=3)
        wait(entered)
        cancel = threading.Event()
        queued = backend.submit([5, 6, 7], max_tokens=3, cancel_event=cancel)
        cancel.set()
        release.set()
        with ThreadPoolExecutor(2) as pool:
            assert len(pool.submit(collect, first).result(timeout=10)) == 3
            assert pool.submit(collect, queued).result(timeout=10) == []
        assert all(shape != (1, 3) for shape, _ in model.calls)
    finally:
        release.set()
        backend.close()


def test_decode_failure_spares_the_partial_prefill():
    model = CountingModel()
    failed = False

    def hook(ids, cache):
        nonlocal failed
        partial = any(
            shape == (1, 512) and not batch for shape, batch in model.calls[:-1]
        )
        if isinstance(cache, BatchModelCache) and partial and not failed:
            failed = True
            raise RuntimeError("injected decode failure")

    model.hook = hook
    backend = BatchBackend(model, Tokenizer(), lanes=2)
    try:
        short = backend.submit([1, 2], max_tokens=4096)
        long = backend.submit([1] * 1100, max_tokens=4)
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(collect, stream) for stream in (short, long)]
            # the decode row ends with the shared step's failure ...
            with pytest.raises(RuntimeError, match="injected decode"):
                futures[0].result(timeout=10)
            # ... while the prefill that was mid-prompt owns its cache and
            # finishes normally
            assert len(futures[1].result(timeout=10)) == 4
        assert failed
        after = backend.submit([3, 4], max_tokens=2)
        assert len(collect(after)) == 2
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ["app", "policy"])
def test_server_closes_batch_backend_when_app_construction_fails(monkeypatch, failure):
    from pathlib import Path

    from rapid_mlx.models.deepseek_v41_native import batch_scheduler, server
    from rapid_mlx.speculative.dflash import server as serial

    monkeypatch.setenv("RAPID_MLX_DSV41_LANES", "2")
    monkeypatch.setattr(server, "require_product_memory", lambda: None)
    monkeypatch.setattr(server, "download_target_snapshot", lambda: Path("target"))
    monkeypatch.setattr(server, "download_mtp_snapshot", lambda: Path("mtp"))
    runtime = SimpleNamespace(target_revision="target-revision")
    monkeypatch.setattr(
        server,
        "load_product_runtime",
        lambda *args, **kwargs: (CountingModel(), Tokenizer(), runtime),
    )
    built = []
    original_backend = batch_scheduler.BatchBackend

    def build_backend(**kwargs):
        built.append(original_backend(**kwargs))
        return built[-1]

    monkeypatch.setattr(batch_scheduler, "BatchBackend", build_backend)

    def broken_app(*args, **kwargs):
        raise RuntimeError("injected app failure")

    if failure == "app":
        monkeypatch.setattr(serial, "_build_app", broken_app)
    else:
        # setup between backend construction and app construction
        monkeypatch.setattr(serial, "BackendExecutionPolicy", broken_app)
    with pytest.raises(RuntimeError, match="injected app failure"):
        server.run_server(
            host="127.0.0.1",
            port=8000,
            served_model_name="test",
            default_max_tokens=3,
            cors_origins=[],
            uvicorn_log_level="warning",
        )
    assert len(built) == 1 and built[0]._closed


def test_shared_failure_gives_each_request_its_own_exception():
    from rapid_mlx.models.deepseek_v41_native.batch_scheduler import _own_copy

    error = RuntimeError("shared")
    copied = _own_copy(error)
    assert copied is not error and str(copied) == "shared"
    assert copied.__cause__ is None
    http = HTTPException(status_code=400, detail="bad")
    own_http = _own_copy(http)
    assert own_http is not http and isinstance(own_http, HTTPException)
    assert (own_http.status_code, own_http.detail) == (400, "bad")

    class UnreplayableError(Exception):
        def __init__(self, a, b):
            super().__init__(f"{a}/{b}")

        def __reduce__(self):
            raise TypeError("no replay")

    odd = UnreplayableError(1, 2)
    own_odd = _own_copy(odd)
    assert own_odd is not odd and own_odd.__cause__ is None
    assert "UnreplayableError" in str(own_odd) and "1/2" in str(own_odd)
