"""Ordinary Gemma prefill keeps one output row without losing prefix state."""

from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

import rapid_mlx.mllm_batch_generator as batch_generator
from rapid_mlx.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    _model_supports_gemma4_last_position,
)


@pytest.fixture(autouse=True)
def qualified_distribution(monkeypatch):
    # Recording doubles exercise the supported contract independently of the
    # optional installed distribution. The real-model test restores metadata.
    monkeypatch.setattr(batch_generator, "distribution_version", lambda _: "0.7.1")


class _Trunk:
    def __call__(self, *, logits_to_keep=None):
        pass


class _LegacyTrunk:
    def __call__(self, **kwargs):
        pass


class _LanguageModel:
    def __init__(self):
        self.model = _Trunk()

    def __call__(self, ids, cache=None, **kwargs):
        if cache is not None:
            cache[0].tokens.extend(ids.reshape(-1).tolist())
        logits = mx.stack((ids, ids + 10), axis=-1).astype(mx.float32)
        keep = kwargs.pop("logits_to_keep", None)
        return SimpleNamespace(logits=logits[:, -keep:, :] if keep else logits)


class _Wrapper:
    def __init__(self, language_model=None):
        self.language_model = language_model or _LanguageModel()
        self.calls = []
        self.fail = False

    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        self.calls.append((ids.shape[1], dict(kwargs)))
        forwarded = dict(kwargs)
        if "logits_to_keep" in kwargs:
            forwarded["logits_to_keep"] = kwargs["logits_to_keep"]
        output = self.language_model(ids, cache=cache, **forwarded)
        if self.fail:
            raise TypeError("model execution failed after cache update")
        return output


class _LegacyWrapper(_Wrapper):
    def __call__(self, ids, cache=None, pixel_values=None, **kwargs):
        self.calls.append((ids.shape[1], dict(kwargs)))
        return self.language_model(ids, cache=cache)


def _model(wrapper=_Wrapper, module="mlx_vlm.models.gemma4.gemma4", **kwargs):
    # Model identity selects the qualified family; behaviour is implemented by
    # independent recording doubles, without importing optional vision code.
    return type("Model", (wrapper,), {"__module__": module})(**kwargs)


def _generator(model, step=1024):
    return MLLMBatchGenerator(
        model,
        processor=object(),
        enable_prefix_cache=False,
        enable_vision_cache=False,
        prefill_step_size=step,
    )


class _Cache:
    def __init__(self):
        self.tokens = []

    @property
    def state(self):
        return mx.array(self.tokens)


def _request(*, pixels=False, extra=None, videos=None, n=5):
    return MLLMBatchRequest(
        uid=1,
        request_id="gemma-prefill",
        prompt="test",
        input_ids=mx.arange(n)[None, :],
        pixel_values=mx.zeros((1, 2, 2)) if pixels else None,
        videos=videos,
        extra_kwargs=dict(extra or {}),
    )


@pytest.mark.parametrize("pixels", [False, True])
def test_ordinary_prefill_keeps_last_logits_and_complete_prefix(pixels):
    model = _model()
    cache = [_Cache()]
    output = _generator(model)._run_vision_encoding(_request(pixels=pixels), cache)
    assert output.shape == (1, 1, 2)
    assert output.tolist() == [[[4, 14]]]
    assert cache[0].tokens == [0, 1, 2, 3, 4]
    assert model.calls == [(5, {"logits_to_keep": 1})]


@pytest.mark.parametrize(
    "model",
    [
        _model(module="mlx_vlm.models.gemma4_unified.gemma4_unified"),
        _model(module="mlx_vlm.models.qwen3_vl.qwen3_vl"),
        _model(module="rapid_mlx.models.gemma4_vendored.gemma4"),
    ],
)
def test_unknown_or_old_wrapper_keeps_full_forward(model):
    output = _generator(model)._run_vision_encoding(_request(), [_Cache()])
    assert output.shape == (1, 5, 2)
    assert model.calls == [(5, {})]


@pytest.mark.parametrize("installed", ["0.7.1", "0.7.2"])
def test_kwargs_only_trunk_is_not_a_capability(monkeypatch, installed):
    monkeypatch.setattr(batch_generator, "distribution_version", lambda _: installed)
    model = _model()
    model.language_model.model = _LegacyTrunk()
    output = _generator(model)._run_vision_encoding(_request(), [_Cache()])
    assert output.shape == (1, 5, 2)
    assert "logits_to_keep" not in model.calls[0][1]


@pytest.mark.parametrize("hop", ["wrapper", "language_model"])
def test_qualified_version_still_requires_wrapper_kwargs(hop):
    def call(self, ids, cache=None):
        return SimpleNamespace(logits=mx.zeros((1, ids.shape[1], 2)))

    model = _model()
    if hop == "wrapper":
        model = type(
            "Model",
            (_Wrapper,),
            {"__module__": "mlx_vlm.models.gemma4.gemma4", "__call__": call},
        )()
    else:
        model.language_model = type(
            "LanguageModel", (_LanguageModel,), {"__call__": call}
        )()
    assert not _model_supports_gemma4_last_position(model)


@pytest.mark.parametrize(
    "extra",
    [
        {"logits_to_keep": 2},
        {"logits_to_keep": 0},
        {"capture_layer_ids": [1]},
        {"hidden_sink": []},
        {"shared_kv_sink": {}},
        {"return_hidden": True},
        {"return_shared_kv": True},
        {"unknown_metadata": True},
    ],
)
def test_explicit_output_and_capture_metadata_are_preserved(extra):
    model = _model()
    _generator(model)._run_vision_encoding(_request(extra=extra), [_Cache()])
    assert model.calls == [(5, extra)]


@pytest.mark.parametrize(
    "key", ["image_position_ids", "token_type_ids", "mm_token_type_ids"]
)
def test_supported_processor_metadata_is_retained(key):
    model = _model()
    value = mx.zeros((1, 5))
    output = _generator(model)._run_vision_encoding(
        _request(pixels=True, extra={key: value}), [_Cache()]
    )
    assert output.shape == (1, 1, 2)
    assert model.calls[0][1][key] is value
    assert model.calls[0][1]["logits_to_keep"] == 1


@pytest.mark.parametrize("cache,videos", [(None, None), ([_Cache()], ["clip"])])
def test_unqualified_cache_or_video_keeps_full_forward(cache, videos):
    model = _model()
    output = _generator(model)._run_vision_encoding(
        _request(pixels=bool(videos), videos=videos), cache
    )
    assert output.shape == (1, 5, 2)
    assert model.calls == [(5, {})]


def test_chunked_prefill_retains_existing_forwards():
    model = _model()
    cache = [_Cache()]
    output = _generator(model, step=4)._run_vision_encoding(_request(n=9), cache)
    assert output.shape == (1, 1, 2)
    assert model.calls == [(4, {}), (4, {}), (1, {})]
    assert cache[0].tokens == list(range(9))


def test_forward_error_is_not_retried_with_a_mutated_cache():
    model = _model()
    model.fail = True
    cache = [_Cache()]
    with pytest.raises(TypeError, match="after cache update"):
        _generator(model)._run_vision_encoding(_request(), cache)
    assert len(model.calls) == 1
    assert cache[0].tokens == list(range(5))


@pytest.mark.parametrize(
    "installed,expected",
    [
        ("0.7.0", False),
        ("0.7.1rc1", False),
        ("0.7.1", True),
        ("0.7.2", True),
        ("0.8.0", True),
        ("unknown", False),
        ("", False),
        (None, False),
    ],
)
def test_distribution_version_gate(monkeypatch, installed, expected):
    monkeypatch.setattr(batch_generator, "distribution_version", lambda _: installed)
    model = _model()
    assert _model_supports_gemma4_last_position(model) is expected
    output = _generator(model)._run_vision_encoding(_request(), [_Cache()])
    assert output.shape == (1, 1 if expected else 5, 2)
    assert model.calls == [(5, {"logits_to_keep": 1} if expected else {})]


def test_missing_distribution_keeps_full_forward(monkeypatch):
    def missing(name):
        assert name == "mlx-vlm"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(batch_generator, "distribution_version", missing)
    model = _model()
    assert _generator(model)._run_vision_encoding(_request(), [_Cache()]).shape == (
        1,
        5,
        2,
    )
    assert model.calls == [(5, {})]


def test_old_distribution_keeps_ignoring_wrapper_ineligible(monkeypatch):
    monkeypatch.setattr(batch_generator, "distribution_version", lambda _: "0.7.0")
    model = _model(wrapper=_LegacyWrapper)
    assert not _model_supports_gemma4_last_position(model)
    assert _generator(model)._run_vision_encoding(_request(), [_Cache()]).shape == (
        1,
        5,
        2,
    )


def test_capability_is_checked_once_without_source_inspection(monkeypatch):
    import inspect

    def unavailable_source(_):
        raise OSError("source unavailable")

    monkeypatch.setattr(inspect, "getsource", unavailable_source)
    model = _model()
    assert _model_supports_gemma4_last_position(model)
    gen = _generator(model)
    monkeypatch.setattr(batch_generator, "distribution_version", lambda _: "unknown")
    assert not _model_supports_gemma4_last_position(model)
    assert gen._run_vision_encoding(_request(), [_Cache()]).shape == (1, 1, 2)


@pytest.mark.parametrize("error", [TypeError, ValueError])
@pytest.mark.parametrize("hop", ["wrapper", "language_model", "trunk"])
def test_uninspectable_signature_keeps_full_forward(monkeypatch, error, hop):
    import inspect

    model = _model()
    target = {
        "wrapper": type(model).__call__,
        "language_model": type(model.language_model).__call__,
        "trunk": type(model.language_model.model).__call__,
    }[hop]
    original = inspect.signature

    def signature(fn):
        if fn is target:
            raise error("signature unavailable")
        return original(fn)

    monkeypatch.setattr(inspect, "signature", signature)
    output = _generator(model)._run_vision_encoding(_request(), [_Cache()])
    assert output.shape == (1, 5, 2)
    assert model.calls == [(5, {})]


@pytest.fixture
def preserved_random_state():
    state = mx.random.state
    try:
        yield
    finally:
        mx.random.state = state


def test_real_language_model_retains_cache_and_next_decode(
    preserved_random_state, monkeypatch
):
    monkeypatch.setattr(batch_generator, "distribution_version", version)
    config_module = pytest.importorskip("mlx_vlm.models.gemma4.config")
    language = pytest.importorskip("mlx_vlm.models.gemma4.language")
    gemma = pytest.importorskip("mlx_vlm.models.gemma4.gemma4")
    config = config_module.TextConfig.from_dict(
        dict(
            hidden_size=16,
            num_hidden_layers=3,
            num_kv_shared_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            vocab_size=32,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=0,
            use_double_wide_mlp=False,
            layer_types=["full_attention"] * 3,
        )
    )
    mx.random.seed(11)
    lm = language.LanguageModel(config)
    # Qualify the actual installed outer wrapper without constructing its
    # vision/audio towers or downloading weights.
    wrapper = gemma.Model.__new__(gemma.Model)
    wrapper.language_model = lm
    assert _model_supports_gemma4_last_position(wrapper), (
        "installed mlx-vlm must support the declared last-position contract"
    )
    request = _request()
    full_cache, kept_cache = lm.make_cache(), lm.make_cache()
    baseline = lm(request.input_ids, cache=full_cache).logits
    kept = _generator(_model(language_model=lm))._run_vision_encoding(
        request, kept_cache
    )
    assert kept.shape == (1, 1, 32)
    # Different query lengths can select different GPU reduction kernels.
    # M5 showed ~1e-3 shape-dependent float32 error; M3 was below 4e-7.
    # Bound every retained logit as well as the sampled token and exact cache.
    assert bool(mx.allclose(kept, baseline[:, -1:, :], atol=2e-3, rtol=1e-4))
    assert bool(mx.array_equal(mx.argmax(kept, -1), mx.argmax(baseline[:, -1:, :], -1)))
    for before, after in zip(full_cache, kept_cache, strict=True):
        assert before.offset == after.offset == 5
        assert all(
            bool(mx.array_equal(a, b))
            for a, b in zip(before.state, after.state, strict=True)
        )
    token = mx.array([[7]])
    full_next = lm(token, cache=full_cache).logits
    kept_next = lm(token, cache=kept_cache).logits
    assert bool(mx.allclose(full_next, kept_next, atol=1e-5))
