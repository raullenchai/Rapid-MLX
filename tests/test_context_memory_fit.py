"""Shared offline/runtime context inversion and weights-only accounting."""

from types import SimpleNamespace

import pytest

from rapid_mlx.kv_estimation import (
    KVFootprintEstimate,
    config_kv_dims,
    footprint_from_config,
    kv_dtype_bytes,
    memory_fit_context,
    rotating_cache_slots,
)
from rapid_mlx.memory_budget import model_weights_bytes


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"dtype": "torch.bfloat16"}, 2),
        ({"torch_dtype": "float64"}, 8),
        ({"dtype": "fp32"}, 4),
        ({"dtype": "int8"}, 1),
        ({"dtype": "unknown", "text_config": {"dtype": "half"}}, 2),
        ({}, 4),
    ],
)
def test_dtype_config_and_nested_fallback(config, expected):
    assert kv_dtype_bytes(config) == expected


def test_config_footprint_preserves_dense_and_nested_geometry():
    dense = {"num_hidden_layers": 4, "num_attention_heads": 2, "hidden_size": 16}
    assert footprint_from_config(dense) == KVFootprintEstimate(512, 0, 0, 0)
    assert footprint_from_config(dense, dtype_bytes=2).per_token_growth_bytes == 256
    assert config_kv_dims({"num_hidden_layers": True}) is None
    assert footprint_from_config({}) is None
    nested = {"dtype": "float16", "text_config": dense}
    assert footprint_from_config(nested).per_token_growth_bytes == 256
    assert config_kv_dims({**dense, "text_config": {**dense, "head_dim": 4}})[:3] == (
        4,
        2,
        4,
    )


def test_hybrid_structural_tower_and_estimator_failure(monkeypatch):
    text = {
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 256,
    }
    config = {"dtype": "float16", "num_hidden_layers": 40, "text_config": text}
    estimate = footprint_from_config(config)
    assert estimate == KVFootprintEstimate(16, 0, 16, 256)
    monkeypatch.setattr(
        "rapid_mlx.kv_estimation.estimate_kv_footprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unfamiliar config")
        ),
    )
    assert footprint_from_config(config) == KVFootprintEstimate(32, 0, 0, 0)


@pytest.mark.parametrize("native", [None, 0, -1, 10])
def test_dense_inverse_and_native_limit(native):
    result = memory_fit_context(KVFootprintEstimate(10, 20, 0, 0), 121, native)
    assert result == 10


@pytest.mark.parametrize(
    "estimate,available,expected",
    [
        (KVFootprintEstimate(0, 0, 0, 0), 100, None),
        (KVFootprintEstimate(1, 0, 0, 0), 0, None),
        (KVFootprintEstimate(1, 100, 0, 0), 100, None),
        (KVFootprintEstimate(0, 100, 0, 0), 101, 1000),
        (KVFootprintEstimate(10, 20, 0, 0), 120, 9),
        (KVFootprintEstimate(2, 10, 1, 512), 522, 127),
    ],
)
def test_fit_boundary_and_fixed_sliding_terms(estimate, available, expected):
    assert memory_fit_context(estimate, available, 1000) == expected


def test_sliding_inverse_is_maximum_fit():
    estimate = KVFootprintEstimate(3, 100, 2, 1024)
    result = memory_fit_context(estimate, 3000, 4000)
    assert result is not None
    footprint = lambda tokens: 100 + 3 * tokens + 2 * rotating_cache_slots(1024, tokens)
    assert footprint(result) < 3000 <= footprint(result + 1)


def test_weights_measurement_ignores_process_residency_and_deduplicates():
    array = SimpleNamespace(nbytes=100)
    model = SimpleNamespace(
        parameters=lambda: {"a": [array], "b": (array, SimpleNamespace(nbytes=20))}
    )
    assert model_weights_bytes(model) == 120


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        {},
        {"a": object()},
        {"a": SimpleNamespace(nbytes=True)},
        {"a": SimpleNamespace(nbytes=-1)},
    ],
)
def test_unknown_weights_are_not_a_fabricated_zero(parameters):
    assert model_weights_bytes(SimpleNamespace(parameters=lambda: parameters)) is None
    assert model_weights_bytes(None) is None
