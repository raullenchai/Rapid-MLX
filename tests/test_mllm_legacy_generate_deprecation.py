"""Deprecation contract for MLXMultimodalLM's legacy generation surface.

The serving path (BatchedEngine → MLLMScheduler → MLLMBatchGenerator) loads
models through MLXMultimodalLM but generates exclusively on the native
serialized lane. The legacy ``generate``/``stream_generate``/``chat``/
``stream_chat`` methods still ride mlx-vlm's generation runtime and are
deprecated with a full minor release of notice — these tests pin the
warning contract so the eventual removal cannot land silently.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep

import pytest

from rapid_mlx.models.mllm import MLXMultimodalLM

_LEGACY_METHODS = [
    ("generate", {"prompt": "hi"}),
    ("stream_generate", {"prompt": "hi"}),
    ("chat", {"messages": [{"role": "user", "content": "hi"}]}),
    ("stream_chat", {"messages": [{"role": "user", "content": "hi"}]}),
]


def _raise_on_deprecation(call):
    """Run ``call`` with DeprecationWarning promoted to an error.

    Generator methods only execute their body on first iteration, so the
    caller wraps generator factories and consumes one item.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return call()


@pytest.mark.parametrize("method,kwargs", _LEGACY_METHODS)
def test_legacy_generation_methods_warn(method, kwargs):
    model = MLXMultimodalLM("test-model")
    assert not model._loaded

    if method.startswith("stream_"):
        # A generator method runs its body lazily — creating the generator
        # warns nothing; the warning fires on the first consumed chunk,
        # before any model load or mlx-vlm import.
        produced = getattr(model, method)(**kwargs)
        with pytest.raises(DeprecationWarning, match="legacy generation"):
            _raise_on_deprecation(lambda: next(produced))
    else:
        with pytest.raises(DeprecationWarning, match="legacy generation"):
            _raise_on_deprecation(lambda: getattr(model, method)(**kwargs))
    # The warning must fire before the lazy load: the model stays unloaded.
    assert not model._loaded


def test_deprecation_message_names_the_native_lane():
    model = MLXMultimodalLM("test-model")
    # Skip the lazy load: the warning must fire first, and whatever comes
    # after (an ImportError without mlx-vlm, or an AttributeError on the
    # unloaded stub) is not this test's concern.
    model._loaded = True
    # Post-warning the call fails deterministically on the unloaded stub:
    # ImportError when mlx-vlm is absent, AttributeError on the None
    # processor/model when it is present. Anything else is a regression.
    with (
        pytest.warns(DeprecationWarning, match="BatchedEngine"),
        pytest.raises((ImportError, AttributeError)),
    ):
        model.generate(prompt="hi")


def test_convenience_wrappers_inherit_the_warning():
    # describe_image / answer_about_image / describe_video delegate to
    # generate(), so they are deprecated by transitivity — one assertion
    # pins that delegation does not bypass the warning. Each wrapper
    # warns at its own frame first (per-instance dedupe), hence a fresh
    # model per call.
    for call in (
        lambda: MLXMultimodalLM("test-model").describe_image("nonexistent.png"),
        lambda: MLXMultimodalLM("test-model").answer_about_image(
            "nonexistent.png", "what?"
        ),
        lambda: MLXMultimodalLM("test-model").describe_video("nonexistent.mp4"),
    ):
        with pytest.raises(DeprecationWarning, match="legacy generation"):
            _raise_on_deprecation(call)


def test_legacy_warning_fires_once_per_instance():
    # The "on first use" contract: the legacy surface warns once per
    # model instance — not on every call — and the convenience wrappers
    # rely on that dedupe so delegation does not warn twice.
    from rapid_mlx.models.mllm import _warn_legacy_generation

    model = MLXMultimodalLM("test-model")
    with pytest.warns(DeprecationWarning, match="legacy generation"):
        _warn_legacy_generation(model, "generate")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_legacy_generation(model, "stream_generate")
    assert caught == []
    # A fresh instance warns again.
    with pytest.warns(DeprecationWarning, match="legacy generation"):
        _warn_legacy_generation(MLXMultimodalLM("test-model"), "generate")

    # A warnings-as-errors caller aborts at the warning.  That interrupted
    # attempt must not consume the per-instance notice: retry warns again.
    retry_model = MLXMultimodalLM("test-model")
    for _ in range(2):
        with pytest.raises(DeprecationWarning, match="legacy generation"):
            _raise_on_deprecation(
                lambda: _warn_legacy_generation(retry_model, "generate")
            )
        assert not retry_model._legacy_generation_warned


def test_legacy_warning_is_once_per_instance_under_concurrency(monkeypatch):
    from rapid_mlx.models import mllm as mllm_module

    model = MLXMultimodalLM("test-model")
    start = Barrier(3)
    calls = []

    def _warn(*args, **kwargs):
        calls.append((args, kwargs))
        # Release the GIL long enough for the competing caller to reach the
        # warning-state critical section. Without the instance lock, both call.
        sleep(0.05)

    def _call(method):
        start.wait()
        mllm_module._warn_legacy_generation(model, method)

    monkeypatch.setattr(mllm_module.warnings, "warn", _warn)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_call, method) for method in ("generate", "chat")]
        start.wait()
        for future in futures:
            future.result(timeout=2)

    assert len(calls) == 1


class _FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "".join(f"<{t}>" for t in tokens)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


def test_benchmark_loader_uses_the_production_wrapper(monkeypatch):
    """Benchmarks must inherit the same load-time patches as serving."""
    from rapid_mlx import benchmark as bench
    from rapid_mlx.models import mllm as mllm_module

    calls = []

    class _FakeWrapper:
        def __init__(self, model_name):
            calls.append(("init", model_name))
            self.model = object()
            self.processor = object()
            self.config = {"model_type": "fake"}

        def load(self):
            calls.append(("load",))

    monkeypatch.setattr(mllm_module, "MLXMultimodalLM", _FakeWrapper)
    model, processor, config = bench._load_benchmark_mllm("publisher/model")
    assert calls == [("init", "publisher/model"), ("load",)]
    assert model is not None
    assert processor is not None
    assert config == {"model_type": "fake"}


@pytest.mark.requires_mlx
def test_benchmark_entrypoints_use_loaded_native_lane_for_warmup_and_runs(
    monkeypatch, tmp_path
):
    """Both public benchmark loops reuse one production-loaded generator."""
    from rapid_mlx import benchmark as bench
    from rapid_mlx import optimizations

    class _Hardware:
        chip_name = "Test Chip"
        total_memory_gb = 32

    model = object()
    processor = object()
    config = {"model_type": "fake"}
    generator = object()
    load_calls = []
    build_calls = []
    image_calls = []
    video_calls = []

    def _load(model_name):
        load_calls.append(model_name)
        return model, processor, config

    def _build(got_model, got_processor, max_tokens):
        build_calls.append((got_model, got_processor, max_tokens))
        return generator

    def _image_run(*args, **kwargs):
        image_calls.append((args, kwargs))
        width, height = args[4], args[5]
        return bench.MLLMBenchmarkResult(
            resolution=f"{width}x{height}",
            width=width,
            height=height,
            pixels=width * height,
            time_seconds=1.0,
            tokens_generated=1,
            tokens_per_second=1.0,
            response_preview="ok",
        )

    def _video_run(*args, **kwargs):
        video_calls.append((args, kwargs))
        return bench.VideoBenchmarkResult(
            config_name=args[6],
            fps=args[4],
            max_frames=args[5],
            frames_extracted=1,
            video_duration=1.0,
            time_seconds=1.0,
            prompt_tokens=1,
            completion_tokens=1,
            tokens_per_second=1.0,
            response_preview="ok",
        )

    monkeypatch.setattr(optimizations, "detect_hardware", lambda: _Hardware())
    monkeypatch.setattr(bench, "_load_benchmark_mllm", _load)
    monkeypatch.setattr(bench, "build_bench_generator", _build)
    monkeypatch.setattr(bench, "download_test_image", lambda _url: _FakeImage())
    monkeypatch.setattr(bench, "benchmark_mllm_resolution_native", _image_run)
    monkeypatch.setattr(bench, "benchmark_video_config_native", _video_run)
    monkeypatch.setattr(
        bench,
        "get_video_info",
        lambda _path: {
            "duration": 1.0,
            "total_frames": 4,
            "width": 64,
            "height": 64,
            "fps": 4.0,
        },
    )

    image_results = bench.run_mllm_benchmark(
        "publisher/model", quick=True, max_tokens=7, warmup_runs=1
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fixture")
    video_results = bench.run_video_benchmark(
        "publisher/model",
        video_path=str(video_path),
        quick=True,
        max_tokens=9,
        warmup_runs=1,
    )

    assert load_calls == ["publisher/model", "publisher/model"]
    assert build_calls == [(model, processor, 7), (model, processor, 9)]
    assert len(image_results) == 4
    assert len(image_calls) == 5  # one warmup + four quick configurations
    assert all(args[:3] == (generator, processor, config) for args, _ in image_calls)
    assert image_calls[0][1] == {"warmup": True}
    assert len(video_results) == 3
    assert len(video_calls) == 4  # one warmup + three quick configurations
    assert all(args[:3] == (generator, processor, config) for args, _ in video_calls)
    assert video_calls[0][1] == {"warmup": True}


class _FakeImage:
    size = (1200, 800)


class _FakeLegacyModel:
    """The pre-native-lane first argument: a loaded wrapper model."""

    def __init__(self):
        self.model = object()  # Unused: build_bench_generator is stubbed
        self.processor = _FakeProcessor()
        self.config = {}


def _bench_mllm(monkeypatch):
    """A real MLXMultimodalLM with load() stubbed for wrapper tests."""
    monkeypatch.setattr(
        MLXMultimodalLM, "load", lambda self: setattr(self, "_loaded", True)
    )
    return MLXMultimodalLM("test-model")


class _FakeGenerator:
    """Minimal serialized-lane generator: insert() → next() batches."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.processor = _FakeProcessor()
        self.inserted = None
        self.removed = []

    def insert(self, requests):
        self.inserted = requests[0]
        for batch in self.batches:
            for response in batch:
                if response.request_id is None:
                    response.request_id = self.inserted.request_id
        return [7]

    def next(self):
        if not self.batches:
            return []
        return self.batches.pop(0)

    def remove(self, uids):
        self.removed.extend(uids)


def _response(
    token,
    finish_reason=None,
    prompt_tokens=0,
    token_is_stop_token=False,
    uid=7,
    request_id=None,
):
    from rapid_mlx.mllm_batch_generator import MLLMBatchResponse

    return MLLMBatchResponse(
        uid=uid,
        request_id=request_id,
        token=token,
        logprobs=None,
        finish_reason=finish_reason,
        token_is_stop_token=token_is_stop_token,
        prompt_tokens=prompt_tokens,
    )


@pytest.mark.requires_mlx
def test_native_request_helper_drains_to_finish():
    from rapid_mlx.benchmark import _run_native_mllm_request

    generator = _FakeGenerator(
        [
            [_response(5, prompt_tokens=42)],
            [
                _response(6),
                _response(7, finish_reason="stop", token_is_stop_token=True),
            ],
            # Never reached: the helper stops at the finish_reason above.
            [_response(9)],
        ]
    )
    text, completion, prompt_tokens = _run_native_mllm_request(
        generator,
        "formatted<image>prompt",
        images=["/tmp/x.jpg"],
        videos=["/tmp/v.mp4"],
        video_fps=2.0,
        video_max_frames=8,
        max_tokens=16,
        temperature=0.7,
    )
    # The terminal stop token is a control sentinel, not generated text:
    # it is decoded neither into the text nor into the token count.
    assert text == "<5><6>"
    assert (completion, prompt_tokens) == (2, 42)
    request = generator.inserted
    assert request.uid == -1  # Generator assigns the real uid on insert
    assert request.prompt == "formatted<image>prompt"
    assert request.images == ["/tmp/x.jpg"]
    assert request.videos == ["/tmp/v.mp4"]
    assert (request.video_fps, request.video_max_frames) == (2.0, 8)
    assert (request.max_tokens, request.temperature) == (16, 0.7)


@pytest.mark.requires_mlx
def test_bench_generator_uses_the_complete_scheduler_stop_union(monkeypatch):
    from rapid_mlx import benchmark as bench
    from rapid_mlx import mllm_batch_generator as batch_module
    from rapid_mlx.utils.tokenizer import RAPID_EXTRA_EOS_ATTR

    captured = {}

    class _CapturingGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _Tokenizer:
        _eos_token_ids = {1, 2}
        eos_token_id = 3
        eos_token_ids = (4, 5)

    setattr(_Tokenizer, RAPID_EXTRA_EOS_ATTR, {6})

    class _Processor:
        tokenizer = _Tokenizer()

    class _Model:
        config = {"eos_token_id": 7, "text_config": {"eos_token_id": [8, 9]}}

    monkeypatch.setattr(batch_module, "MLLMBatchGenerator", _CapturingGenerator)
    bench.build_bench_generator(_Model(), _Processor(), 32)
    assert captured["stop_tokens"] == set(range(1, 10))

    # Pin the alternate tokenizer shapes too: the singular field can be a
    # list, while the plural field can itself be a single integer.
    from rapid_mlx.mllm_scheduler import collect_mllm_stop_tokens

    class _AlternateTokenizer:
        eos_token_id = [10, 11]
        eos_token_ids = 12

    assert collect_mllm_stop_tokens(
        _AlternateTokenizer(),
        {"eos_token_id": True, "text_config": {"eos_token_id": False}},
    ) == {10, 11, 12}


@pytest.mark.requires_mlx
def test_native_request_helper_raises_when_the_lane_goes_idle():
    # An empty next() batch without a terminal finish_reason means the lane
    # stopped making progress: returning here would silently truncate the
    # benchmark run and leave the request active in the reused generator.
    # The helper must fail loud instead.
    from rapid_mlx.benchmark import _run_native_mllm_request

    generator = _FakeGenerator([[_response(5)]])
    with pytest.raises(RuntimeError, match="idle"):
        _run_native_mllm_request(generator, "p", max_tokens=4)
    assert generator.batches == []  # The idle batch is what stopped it
    # The stale request must be removed from the reused generator before
    # the helper raises — otherwise every later config inherits it.
    assert generator.removed == [7]


@pytest.mark.requires_mlx
def test_native_request_helper_counts_a_length_terminal_token():
    # A finish_reason="length" cutoff's final token is a real emitted
    # token (token_is_stop_token=False): it stays in the text and count —
    # only the stop sentinel is excluded.
    from rapid_mlx.benchmark import _run_native_mllm_request

    generator = _FakeGenerator([[_response(5), _response(6, finish_reason="length")]])
    text, completion, _ = _run_native_mllm_request(generator, "p", max_tokens=2)
    assert text == "<5><6>"
    assert completion == 2


@pytest.mark.requires_mlx
def test_native_request_sampling_comes_from_request_fields():
    # Sampling is configured per request inside the generator (each
    # MLLMBatchRequest's temperature/top_p builds its sampler via
    # _request_sampler and the homogeneous-batch fast path) — there is no
    # baked generator-level sampler to fall back on. The request must
    # carry the requested temperature and the lane-default top_p.
    from rapid_mlx.benchmark import _run_native_mllm_request

    generator = _FakeGenerator(
        [[_response(1, finish_reason="stop", token_is_stop_token=True)]]
    )
    _run_native_mllm_request(generator, "p", max_tokens=1, temperature=0.0)
    request = generator.inserted
    assert (request.temperature, request.top_p) == (0.0, 0.9)


@pytest.mark.requires_mlx
def test_native_request_helper_ignores_foreign_uids_and_makes_unique_ids():
    from rapid_mlx.benchmark import _run_native_mllm_request

    # The generator is reused across configs: a stale response from an
    # earlier request (different uid) must not be counted toward — or
    # terminate — this one.
    generator = _FakeGenerator(
        [
            [
                _response(9, uid=99),
                _response(8, uid=7, request_id="stale-prior-request"),
                _response(5, prompt_tokens=1),
            ],
            [_response(6, finish_reason="stop", token_is_stop_token=True)],
        ]
    )
    text, completion, _ = _run_native_mllm_request(generator, "p", max_tokens=4)
    # Both the foreign UID and the prior request that reused uid=7 are ignored.
    assert text == "<5>"
    assert completion == 1

    # Each request also carries a unique id so stale responses cannot
    # alias across reuse, even if a uid collided.
    generator_b = _FakeGenerator(
        [[_response(1, finish_reason="stop", token_is_stop_token=True)]]
    )
    _run_native_mllm_request(generator_b, "p", max_tokens=1)
    assert generator_b.inserted.request_id != generator.inserted.request_id


@pytest.mark.requires_mlx
def test_legacy_signature_wrappers_warn_and_delegate(monkeypatch):
    # The pre-native-lane signatures keep working through a deprecated
    # wrapper: it builds the serialized-lane generator internally and
    # never touches mlx-vlm's generation runtime.
    # Templating is out of scope here (the wrappers delegate to it); the
    # fakes carry a bare config that real templating would reject, and
    # templating failures now propagate after the blanket fallback's
    # removal.
    import mlx_vlm.prompt_utils as prompt_utils

    from rapid_mlx import benchmark as bench

    structured = [{"role": "user", "content": "formatted"}]
    rendered = []
    monkeypatch.setattr(
        prompt_utils, "apply_chat_template", lambda *args, **kwargs: structured
    )
    monkeypatch.setattr(
        prompt_utils,
        "get_chat_template",
        lambda processor, messages, add_generation_prompt: (
            rendered.append((processor, messages, add_generation_prompt)) or "rendered"
        ),
    )

    generators = []
    built = []

    def _fake_build(model, processor, max_tokens):
        built.append((type(model).__name__, max_tokens))
        # One fresh generator per build — the real benchmark builds one per
        # run and reuses it across configs that each drain to completion.
        generator = _FakeGenerator(
            [
                [
                    _response(1, prompt_tokens=3),
                    _response(2, finish_reason="stop", token_is_stop_token=True),
                ]
            ]
        )
        generators.append(generator)
        return generator

    monkeypatch.setattr(bench, "build_bench_generator", _fake_build)

    PILImage = pytest.importorskip("PIL.Image")
    image = PILImage.new("RGB", (224, 224))

    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = bench.benchmark_mllm_resolution(
            object(), _FakeProcessor(), {}, image, 224, 224, max_tokens=8
        )
    assert result.tokens_generated == 1  # Stop sentinel excluded
    assert built == [("object", 8)]

    with pytest.warns(DeprecationWarning, match="deprecated"):
        video_result = bench.benchmark_video_config(
            _bench_mllm(monkeypatch),
            "/tmp/nonexistent.mp4",
            1.0,
            4,
            "cfg",
            {"duration": 1.0, "total_frames": 4, "width": 64, "height": 64, "fps": 2.0},
            max_tokens=8,
        )
    assert video_result.completion_tokens == 1
    assert len(built) == 2
    # The video wrapper preserves the exact legacy positional shape:
    # (model, video_path, fps, max_frames, config_name, video_info, ...).
    with pytest.warns(DeprecationWarning, match="deprecated"):
        positional = bench.benchmark_video_config(
            _bench_mllm(monkeypatch),
            "/tmp/v.mp4",
            2.0,
            8,
            "cfg",
            {"duration": 2.0, "total_frames": 8},
        )
    assert positional.completion_tokens == 1
    assert len(rendered) == 3
    assert all(messages is structured for _, messages, _ in rendered)
    assert all(
        add_generation_prompt is True for _, _, add_generation_prompt in rendered
    )


@pytest.mark.requires_mlx
def test_video_wrapper_lazily_loads_an_unloaded_model(monkeypatch):
    # The legacy path lazily loaded an unloaded MLXMultimodalLM on first
    # use; the compatibility wrapper must preserve that contract, or
    # previously valid callers crash on None components.
    # Templating is out of scope here — stub it before the wrapper call.
    import mlx_vlm.prompt_utils as prompt_utils

    from rapid_mlx import benchmark as bench

    monkeypatch.setattr(
        prompt_utils, "apply_chat_template", lambda *args, **kwargs: "formatted"
    )

    load_calls = []

    def _fake_load(self):
        self._loaded = True
        load_calls.append(1)

    monkeypatch.setattr(MLXMultimodalLM, "load", _fake_load)
    model = MLXMultimodalLM("test-model")
    monkeypatch.setattr(
        bench,
        "build_bench_generator",
        lambda m, p, max_tokens: _FakeGenerator(
            [
                [
                    _response(1, prompt_tokens=1),
                    _response(2, finish_reason="stop", token_is_stop_token=True),
                ]
            ]
        ),
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        bench.benchmark_video_config(
            model, "/tmp/v.mp4", 1.0, 4, "cfg", {"duration": 1.0, "total_frames": 4}
        )
    assert len(load_calls) == 1

    # An already-loaded wrapper is not loaded twice.
    with pytest.warns(DeprecationWarning, match="deprecated"):
        bench.benchmark_video_config(
            model, "/tmp/v.mp4", 1.0, 4, "cfg", {"duration": 1.0, "total_frames": 4}
        )
    assert len(load_calls) == 1


def test_video_wrapper_routes_duck_typed_models_through_generate():
    # The pre-native contract accepted any object exposing
    # generate(prompt=..., videos=...); during the deprecation window
    # those keep their exact previous behavior — the warning is the
    # migration notice — instead of being cut off early.
    from rapid_mlx import benchmark as bench

    class _LegacyOutput:
        prompt_tokens = 11
        completion_tokens = 7
        text = "a preview"

    class _DuckTyped:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _LegacyOutput()

    duck = _DuckTyped()
    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = bench.benchmark_video_config(
            duck, "/tmp/v.mp4", 2.0, 8, "cfg", {"duration": 2.0, "total_frames": 16}
        )
    assert duck.calls == [
        {
            "prompt": "Describe what happens in this video. What do you see?",
            "videos": ["/tmp/v.mp4"],
            "video_fps": 2.0,
            "video_max_frames": 8,
            "max_tokens": 150,
            "temperature": 0.7,
        }
    ]
    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)
    assert result.response_preview == "a preview"
    # min(duration * fps, max_frames, total_frames) = min(4, 8, 16)
    assert result.frames_extracted == 4


def test_video_wrapper_prefers_generate_for_foreign_models_with_attributes():
    # A foreign object that happens to carry .model/.processor but is not
    # an MLXMultimodalLM keeps its legacy generate() path — attribute
    # sniffing would silently reroute it onto the native lane, which can
    # only drive an MLXMultimodalLM.
    from rapid_mlx import benchmark as bench

    class _LegacyOutput:
        prompt_tokens = 1
        completion_tokens = 2
        text = "x"

    class _ForeignModel:
        model = object()
        processor = _FakeProcessor()

        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return _LegacyOutput()

    foreign = _ForeignModel()
    with pytest.warns(DeprecationWarning, match="deprecated"):
        result = bench.benchmark_video_config(
            foreign, "/tmp/v.mp4", 1.0, 4, "cfg", {"duration": 1.0, "total_frames": 4}
        )
    assert foreign.calls  # generate() ran, not the native lane
    assert result.completion_tokens == 2


def test_video_wrapper_rejects_objects_with_no_supported_shape():
    # Neither an MLXMultimodalLM (.model/.processor) nor a legacy
    # duck-typed generate(): nothing the wrapper can drive — fail loud
    # with the migration path instead of a bare AttributeError.
    from rapid_mlx import benchmark as bench

    with (
        pytest.warns(DeprecationWarning, match="deprecated"),
        pytest.raises(TypeError, match="Migrate to the native lane"),
    ):
        bench.benchmark_video_config(object(), "/tmp/v.mp4", 1.0, 4, "cfg", {})
