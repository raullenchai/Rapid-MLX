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
    with (
        pytest.warns(DeprecationWarning, match="BatchedEngine"),
        pytest.raises(Exception),
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


class _FakeGenerator:
    """Minimal serialized-lane generator: insert() → next() batches."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.processor = _FakeProcessor()
        self.inserted = None

    def insert(self, requests):
        self.inserted = requests[0]
        return [7]

    def next(self):
        if not self.batches:
            return []
        return self.batches.pop(0)


def _response(token, finish_reason=None, prompt_tokens=0):
    from rapid_mlx.mllm_batch_generator import MLLMBatchResponse

    return MLLMBatchResponse(
        uid=7,
        request_id="rapid-mlx-bench",
        token=token,
        logprobs=None,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
    )


def test_native_request_helper_drains_to_finish():
    from rapid_mlx.benchmark import _run_native_mllm_request

    generator = _FakeGenerator(
        [
            [_response(5, prompt_tokens=42)],
            [_response(6), _response(7, finish_reason="stop")],
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
    assert text == "<5><6><7>"
    assert (completion, prompt_tokens) == (3, 42)
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
