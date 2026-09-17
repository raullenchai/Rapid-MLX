"""Deprecation contract for MLXMultimodalLM's legacy generation surface.

The serving path (BatchedEngine → MLLMScheduler → MLLMBatchGenerator) loads
models through MLXMultimodalLM but generates exclusively on the native
serialized lane. The legacy ``generate``/``stream_generate``/``chat``/
``stream_chat`` methods still ride mlx-vlm's generation runtime and are
deprecated with a full minor release of notice — these tests pin the
warning contract so the eventual removal cannot land silently.
"""

import warnings

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
    # describe_image / answer_about_image delegate to generate(), so they
    # are deprecated by transitivity — one assertion pins that delegation
    # does not bypass the warning.
    model = MLXMultimodalLM("test-model")
    for call in (
        lambda: model.describe_image("nonexistent.png"),
        lambda: model.answer_about_image("nonexistent.png", "what?"),
    ):
        with pytest.raises(DeprecationWarning, match="legacy generation"):
            _raise_on_deprecation(call)


class _FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "".join(f"<{t}>" for t in tokens)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


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
):
    from rapid_mlx.mllm_batch_generator import MLLMBatchResponse

    return MLLMBatchResponse(
        uid=uid,
        request_id="rapid-mlx-bench",
        token=token,
        logprobs=None,
        finish_reason=finish_reason,
        token_is_stop_token=token_is_stop_token,
        prompt_tokens=prompt_tokens,
    )


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


def test_native_request_helper_counts_a_length_terminal_token():
    # A finish_reason="length" cutoff's final token is a real emitted
    # token (token_is_stop_token=False): it stays in the text and count —
    # only the stop sentinel is excluded.
    from rapid_mlx.benchmark import _run_native_mllm_request

    generator = _FakeGenerator([[_response(5), _response(6, finish_reason="length")]])
    text, completion, _ = _run_native_mllm_request(generator, "p", max_tokens=2)
    assert text == "<5><6>"
    assert completion == 2


def test_native_request_helper_ignores_foreign_uids_and_makes_unique_ids():
    from rapid_mlx.benchmark import _run_native_mllm_request

    # The generator is reused across configs: a stale response from an
    # earlier request (different uid) must not be counted toward — or
    # terminate — this one.
    generator = _FakeGenerator(
        [
            [
                _response(9, uid=99),
                _response(5, prompt_tokens=1),
            ],
            [_response(6, finish_reason="stop", token_is_stop_token=True)],
        ]
    )
    text, completion, _ = _run_native_mllm_request(generator, "p", max_tokens=4)
    assert text == "<5>"  # the uid=99 token is ignored
    assert completion == 1

    # Each request also carries a unique id so stale responses cannot
    # alias across reuse, even if a uid collided.
    generator_b = _FakeGenerator(
        [[_response(1, finish_reason="stop", token_is_stop_token=True)]]
    )
    _run_native_mllm_request(generator_b, "p", max_tokens=1)
    assert generator_b.inserted.request_id != generator.inserted.request_id


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

    monkeypatch.setattr(
        prompt_utils, "apply_chat_template", lambda *args, **kwargs: "formatted"
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
