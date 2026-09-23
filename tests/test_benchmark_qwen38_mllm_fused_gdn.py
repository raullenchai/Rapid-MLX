"""Contracts for the benchmark-only Qwen3.8 serialized-MLLM fused-GDN A/B."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import benchmark_qwen38_mllm_fused_gdn as bench


class FakeArray:
    def __init__(self, shape: tuple[int, ...], dtype: str):
        self.shape = shape
        self.dtype = dtype

    def __mul__(self, _value):
        return FakeArray(self.shape, self.dtype)

    def astype(self, dtype):
        return FakeArray(self.shape, dtype)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeRandom:
    @staticmethod
    def key(value):
        return value

    @staticmethod
    def normal(shape, key=None):
        assert key is not None
        return FakeArray(shape, FakeMx.bfloat16)


class FakeMx:
    bfloat16 = "bf16"
    float32 = "fp32"
    array = FakeArray
    random = FakeRandom()

    @staticmethod
    def concatenate(values, axis=0):
        assert axis == 1
        left, right = values
        return FakeArray(
            (left.shape[0], left.shape[1] + right.shape[1], left.shape[2]),
            left.dtype,
        )

    @staticmethod
    def zeros(shape, dtype):
        return FakeArray(shape, dtype)

    @staticmethod
    def eval(*_values):
        return None

    @staticmethod
    def array_equal(left, right):
        return FakeScalar(left.shape == right.shape and left.dtype == right.dtype)


class FakeCache:
    def __init__(self, size=2):
        self.cache = [None] * size
        self._left_padding = None
        self._left_padding_advance = 0
        self._lengths = None
        self._lengths_advance = 0
        self._speculation = None
        self._speculation_generation = 0
        self.metadata_revision = 0
        self.window_updates = 0
        self.recurrent_updates = 0
        self.advances = 0
        self.fail_recurrent = False
        self.raise_speculation = False
        self.raise_history = False

    def __getitem__(self, index):
        return self.cache[index]

    def __setitem__(self, index, value):
        self.cache[index] = value

    @property
    def left_padding(self):
        return self._left_padding

    @property
    def lengths(self):
        return self._lengths

    @property
    def is_speculating(self):
        if self.raise_speculation:
            raise RuntimeError("is_speculating unavailable")
        return self._speculation is not None

    @property
    def history_capacity(self):
        if self.raise_history:
            raise RuntimeError("history_capacity unavailable")
        return 0

    def update_window(self, index, source, width, *, lengths=None):
        assert lengths is None
        self.window_updates += 1
        self.cache[index] = FakeArray((1, width, source.shape[2]), source.dtype)
        return self.cache[index]

    def update_recurrent(self, index, length, update):
        self.recurrent_updates += 1
        if self.fail_recurrent:
            raise RuntimeError("post-commit failure")
        output, state = update(self.cache[index], None)
        self.cache[index] = state
        return output, state

    def advance(self, amount):
        self.advances += amount


class FakeGdn:
    stock_calls = 0

    def __init__(self):
        self.hidden_size = bench.HIDDEN_SIZE
        self.training = False
        self.conv1d = SimpleNamespace(
            weight=FakeArray((bench.CONV_DIM, bench.CONV_KERNEL, 1), FakeMx.bfloat16)
        )
        self.A_log = FakeArray((bench.NUM_VALUE_HEADS,), FakeMx.float32)
        self.dt_bias = FakeArray((bench.NUM_VALUE_HEADS,), FakeMx.bfloat16)
        self.norm = SimpleNamespace(
            weight=FakeArray((bench.VALUE_HEAD_DIM,), FakeMx.bfloat16), eps=1e-6
        )
        self.out_proj_raises = False

    def __call__(self, inputs, mask=None, cache=None):
        type(self).stock_calls += 1
        return "stock"

    def in_proj_qkv(self, _inputs):
        return FakeArray((1, 1, bench.CONV_DIM), FakeMx.bfloat16)

    def in_proj_z(self, _inputs):
        return FakeArray((1, 1, bench.VALUE_DIM), FakeMx.bfloat16)

    def _project_gates(self, _inputs):
        shape = (1, 1, bench.NUM_VALUE_HEADS)
        return FakeArray(shape, FakeMx.bfloat16), FakeArray(shape, FakeMx.bfloat16)

    def out_proj(self, output):
        if self.out_proj_raises:
            raise RuntimeError("out projection failed")
        assert output.shape == (1, 1, bench.VALUE_DIM)
        return FakeArray((1, 1, bench.HIDDEN_SIZE), FakeMx.bfloat16)


def _kernel(*_args, **kwargs):
    assert kwargs["num_key_heads"] == bench.NUM_KEY_HEADS
    assert kwargs["num_value_heads"] == bench.NUM_VALUE_HEADS
    assert kwargs["qwen35_semantics"] is True
    return (
        FakeArray((1, 1, bench.VALUE_DIM), FakeMx.bfloat16),
        FakeArray((1, bench.CONV_KERNEL - 1, bench.CONV_DIM), FakeMx.bfloat16),
        FakeArray(
            (
                1,
                bench.NUM_VALUE_HEADS,
                bench.VALUE_HEAD_DIM,
                bench.KEY_HEAD_DIM,
            ),
            FakeMx.float32,
        ),
    )


def _patch(kernel=_kernel):
    layers = [FakeGdn() for _ in range(bench.EXPECTED_GDN_LAYERS)]
    patch = bench.FusedGdnPatch(
        FakeGdn,
        FakeCache,
        layers,
        FakeMx,
        kernel,
        lambda _cache, _steps: None,
        lambda _cache, _steps: None,
    )
    patch.threadgroup_y = 32
    patch.qualified = True
    patch.install()
    return patch, layers


def _cache():
    cache = FakeCache()
    cache[0] = FakeArray((1, bench.CONV_KERNEL - 1, bench.CONV_DIM), FakeMx.bfloat16)
    cache[1] = FakeArray(
        (
            1,
            bench.NUM_VALUE_HEADS,
            bench.VALUE_HEAD_DIM,
            bench.KEY_HEAD_DIM,
        ),
        FakeMx.float32,
    )
    return cache


def _input():
    return FakeArray((1, 1, bench.HIDDEN_SIZE), FakeMx.bfloat16)


def test_baseline_restores_exact_original_class_method():
    patch, _ = _patch()
    try:
        patch.set_candidate(False)
        assert FakeGdn.__call__ is patch.original
        patch.set_candidate(True)
        assert FakeGdn.__call__ is patch.wrapped
        patch.set_candidate(False)
        assert FakeGdn.__call__ is patch.original
    finally:
        patch.close()


def test_candidate_commits_through_exact_cache_apis():
    patch, layers = _patch()
    cache = _cache()
    try:
        output = patch._candidate_call(layers[0], _input(), None, cache)
        assert output.shape == (1, 1, bench.HIDDEN_SIZE)
        assert cache.window_updates == 1
        assert cache.recurrent_updates == 1
        assert cache.advances == 1
        assert patch.hits == 1
        assert patch.layer_hits == [1] + [0] * 47
    finally:
        patch.close()


def test_precommit_out_projection_failure_falls_back_without_cache_touch():
    patch, layers = _patch()
    cache = _cache()
    layers[0].out_proj_raises = True
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
        assert cache.recurrent_updates == 0
        assert cache.advances == 0
        assert patch.hits == 0
    finally:
        patch.close()


def test_precommit_kernel_failure_falls_back_without_cache_touch():
    def fail(*_args, **_kwargs):
        raise RuntimeError("kernel construction failed")

    patch, layers = _patch(fail)
    cache = _cache()
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
        assert cache.recurrent_updates == 0
        assert cache.advances == 0
    finally:
        patch.close()


def test_postcommit_failure_propagates_without_stock_replay():
    patch, layers = _patch()
    cache = _cache()
    cache.fail_recurrent = True
    FakeGdn.stock_calls = 0
    try:
        with pytest.raises(RuntimeError, match="post-commit"):
            patch._candidate_call(layers[0], _input(), None, cache)
        assert cache.window_updates == 1
        assert cache.recurrent_updates == 1
        assert cache.advances == 0
        assert FakeGdn.stock_calls == 0
    finally:
        patch.close()


@pytest.mark.parametrize(
    ("mutation", "stock_calls"),
    [
        (lambda cache, layer, value: setattr(value, "shape", (2, 1, 5120)), 1),
        (lambda cache, layer, value: setattr(cache, "_speculation", {}), 1),
        (lambda cache, layer, value: cache.cache.append(None), 1),
        (lambda cache, layer, value: setattr(cache, "_lengths", object()), 1),
        (lambda cache, layer, value: setattr(cache, "_left_padding", object()), 1),
        (lambda cache, layer, value: setattr(layer, "training", True), 1),
    ],
)
def test_ineligible_runtime_shape_is_untouched_stock(mutation, stock_calls):
    patch, layers = _patch()
    cache = _cache()
    value = _input()
    mutation(cache, layers[0], value)
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], value, None, cache) == "stock"
        assert FakeGdn.stock_calls == stock_calls
        assert cache.window_updates == 0
        assert cache.recurrent_updates == 0
    finally:
        patch.close()


@pytest.mark.parametrize("field", ["raise_speculation", "raise_history"])
def test_raising_cache_metadata_marker_falls_back_stock(field):
    patch, layers = _patch()
    cache = _cache()
    setattr(cache, field, True)
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
    finally:
        patch.close()


def test_missing_cache_metadata_marker_falls_back_stock():
    patch, layers = _patch()
    cache = _cache()
    del cache._speculation
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
    finally:
        patch.close()


def test_cache_subclass_falls_back_stock():
    class CacheSubclass(FakeCache):
        pass

    patch, layers = _patch()
    cache = CacheSubclass()
    cache.cache = _cache().cache
    FakeGdn.stock_calls = 0
    try:
        assert patch._candidate_call(layers[0], _input(), None, cache) == "stock"
        assert FakeGdn.stock_calls == 1
        assert cache.window_updates == 0
    finally:
        patch.close()


class ProbeGdn(FakeGdn):
    def __call__(self, inputs, mask=None, cache=None):
        assert inputs.shape == (1, 1, bench.HIDDEN_SIZE)
        cache[0] = FakeArray(
            (1, bench.CONV_KERNEL - 1, bench.CONV_DIM), FakeMx.bfloat16
        )
        cache[1] = FakeArray(
            (
                1,
                bench.NUM_VALUE_HEADS,
                bench.VALUE_HEAD_DIM,
                bench.KEY_HEAD_DIM,
            ),
            FakeMx.float32,
        )
        cache.advance(1)
        return FakeArray((1, 1, bench.HIDDEN_SIZE), FakeMx.bfloat16)


def test_parity_probe_continues_after_one_threadgroup_exception():
    layers = [ProbeGdn() for _ in range(bench.EXPECTED_GDN_LAYERS)]

    def kernel(*_args, **kwargs):
        if kwargs["threadgroup_y"] == 32:
            raise RuntimeError("unsupported threadgroup")
        return _kernel(*_args, **kwargs)

    patch = bench.FusedGdnPatch(
        ProbeGdn,
        FakeCache,
        layers,
        FakeMx,
        kernel,
        lambda _cache, _steps: None,
        lambda _cache, _steps: None,
    )
    result = bench.run_real_weight_parity_probe(patch, steps=2)
    assert result["pass"] is True
    assert result["threadgroup_y"] == 16
    assert result["candidate_hits"] == 2
    assert patch.qualified is True


def _sample(mode: str, tps: float, token_hash: str = "same") -> dict[str, Any]:
    candidate = mode == "candidate"
    per_layer = bench.EXPECTED_TOKENS + 1 if candidate else 0
    return {
        "mode": mode,
        "completion_tokens": bench.EXPECTED_TOKENS,
        "token_sha256": token_hash,
        "text_sha256": "text",
        "ttft_s": 1.0 if not candidate else 1.05,
        "elapsed_s": 10.0 if not candidate else 9.5,
        "decode_tps": tps,
        "singleton_batch_delta": 1,
        "fused_hits": bench.EXPECTED_GDN_LAYERS * per_layer,
        "fused_layer_hits": [per_layer] * bench.EXPECTED_GDN_LAYERS,
        "memory": {"active_bytes": 10, "peak_bytes": 20},
    }


def _passing_receipt() -> dict[str, Any]:
    pairs = []
    for index in range(bench.EXPECTED_STRATA):
        pairs.append(
            {
                "baseline": [_sample("baseline", 100), _sample("baseline", 100)],
                "candidate": [
                    _sample("candidate", 104),
                    _sample("candidate", 104),
                ],
            }
        )
    return {
        "pairs": pairs,
        "real_weight_parity": {"pass": True, "steps": 32},
        "identity": {"same_model": True, "same_executor": True},
        "configuration": {
            "prefix_cache": False,
            "temperature": 0.0,
            "thinking": False,
            "measured_ignore_eos": True,
        },
        "artifact": {"verified": True},
        "source": {"dirty": False, "source_tree_match": True},
        "media_recovery": {
            "pass": True,
            "stock_before": {"singleton_batch_delta": 1},
            "candidate_image": {"singleton_batch_delta": 1},
            "candidate_text": {"singleton_batch_delta": 1},
            "stock_after": {"singleton_batch_delta": 1},
        },
        "errors": [],
    }


def test_all_decision_gates_pass_on_complete_exact_receipt():
    gates = bench.evaluate_gates(_passing_receipt())
    assert gates["pass"] is True
    assert all(gates["checks"].values())


@pytest.mark.parametrize(
    ("mutate", "failed_gate"),
    [
        (
            lambda receipt: receipt["pairs"][0]["candidate"][0][
                "fused_layer_hits"
            ].__setitem__(0, 256),
            "exact_candidate_hits_per_layer_and_zero_baseline",
        ),
        (
            lambda receipt: receipt["pairs"][0]["candidate"][0].__setitem__(
                "completion_tokens", 255
            ),
            "all_twenty_four_samples_are_256_tokens",
        ),
        (
            lambda receipt: receipt["real_weight_parity"].__setitem__("pass", False),
            "real_weight_32_step_bit_exact",
        ),
        (
            lambda receipt: receipt["source"].__setitem__("dirty", True),
            "source_clean",
        ),
        (
            lambda receipt: receipt["media_recovery"].__setitem__("pass", False),
            "vlm_candidate_and_stock_recovery",
        ),
        (
            lambda receipt: receipt["pairs"][0]["baseline"][0].__setitem__(
                "singleton_batch_delta", 0
            ),
            "singleton_fastpath_engaged",
        ),
        (
            lambda receipt: receipt["media_recovery"]["candidate_image"].__setitem__(
                "singleton_batch_delta", 0
            ),
            "singleton_fastpath_engaged",
        ),
        (
            lambda receipt: [
                sample.__setitem__("elapsed_s", 11.0)
                for pair in receipt["pairs"]
                for sample in pair["candidate"]
            ],
            "median_wall_speedup_gte_1_03",
        ),
        (
            lambda receipt: receipt["pairs"][0]["candidate"][0].__setitem__(
                "elapsed_s", 5.0
            ),
            "paired_wall_ratio_cv_lte_0_05",
        ),
    ],
)
def test_decision_gates_fail_closed(mutate, failed_gate):
    receipt = _passing_receipt()
    mutate(receipt)
    gates = bench.evaluate_gates(receipt)
    assert gates["pass"] is False
    assert gates["checks"][failed_gate] is False


def _media_sample(mode: str, *, completion_tokens=41):
    sample = _sample(mode, 10.0)
    sample.update(
        {
            "completion_tokens": completion_tokens,
            "token_sha256": "image-token-hash",
            "text_sha256": "image-text-hash",
            "text": "A cheetah is visible.",
        }
    )
    if mode == "candidate":
        per_layer = completion_tokens + 1
        sample["fused_layer_hits"] = [per_layer] * bench.EXPECTED_GDN_LAYERS
        sample["fused_hits"] = per_layer * bench.EXPECTED_GDN_LAYERS
    return sample


def _passing_media():
    return {
        "expected": "cheetah",
        "candidate_text_expected": "100",
        "stock_before": _media_sample("baseline"),
        "candidate_image": _media_sample("candidate"),
        "candidate_text": {
            **_media_sample("candidate", completion_tokens=64),
            "text": "The original price is 100.",
        },
        "stock_after": _media_sample("baseline"),
    }


def test_media_sequence_requires_exact_candidate_and_reversible_stock():
    assert bench.media_sequence_pass(_passing_media()) is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda media: media["candidate_image"]["fused_layer_hits"].__setitem__(0, 1),
        lambda media: media["candidate_text"]["fused_layer_hits"].__setitem__(0, 1),
        lambda media: media["stock_after"].__setitem__("fused_hits", 1),
        lambda media: media["stock_after"].__setitem__("token_sha256", "different"),
        lambda media: media.__setitem__("expected", "   "),
        lambda media: media["candidate_text"].__setitem__(
            "text", "The arithmetic answer is 80."
        ),
    ],
)
def test_media_sequence_fails_closed(mutation):
    media = _passing_media()
    mutation(media)
    assert bench.media_sequence_pass(media) is False


def test_pinned_installed_gdn_and_cache_abi_provenance():
    metadata = pytest.importorskip("importlib.metadata")
    if metadata.version("mlx-vlm") != bench.EXPECTED_MLX_VLM_VERSION:
        pytest.skip("requires the pinned benchmark mlx-vlm environment")
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    provenance = bench._stock_method_provenance(Qwen3_5GatedDeltaNet)
    assert provenance["source_sha256"] == bench.EXPECTED_LANGUAGE_SHA256
    cache_provenance = bench._cache_class_provenance(ArraysCache)
    assert cache_provenance["source_sha256"] == bench.EXPECTED_CACHE_SHA256
    assert ArraysCache.__module__ == bench.EXPECTED_CACHE_MODULE
    assert len(ArraysCache(size=2).cache) == 2


def test_validate_args_rejects_blank_image_expect(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    args = argparse.Namespace(
        model=model,
        image_path=image,
        image_expect="  ",
        max_memory_delta_mib=64,
    )
    with pytest.raises(SystemExit):
        bench._validate_args(bench.build_parser(), args)
