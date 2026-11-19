"""Optional loaded-model config facts must not break admission profiling."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.lattice import ModelProfile, model_profile
from rapid_mlx.lattice.scheduling import scheduling_value


@pytest.fixture
def cpu_scheduling(monkeypatch):
    mx = ModuleType("mlx.core")
    mx.device_info = lambda: {}
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(
        "rapid_mlx.lattice.overlay_profile", lambda: SimpleNamespace(cells=())
    )
    monkeypatch.setattr("rapid_mlx.lattice.local_calibration", lambda _: {})
    for name in (
        "RAPID_MLX_PREFIX_BOUNDARY_MIN_TOKENS",
        "RAPID_MLX_ADMISSION_WINDOW_MS",
        "RAPID_MLX_BENCHMARK_KNOBS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_legacy_lfm2_admission(cpu_scheduling):
    # Legacy LFM2 args retain layer_types=None when full_attn_idxs is supplied.
    config = dict(
        model_type="lfm2",
        hidden_size=2048,
        layer_types=None,
        full_attn_idxs=[2, 5, 8],
    )
    assert model_profile(config) == ModelProfile(family="lfm2", hidden_size=2048)
    model = SimpleNamespace(args=SimpleNamespace(**config))
    assert scheduling_value("admission_window_ms", model) == 8.0
    assert scheduling_value("prefix_boundary_min_tokens", model) == 0


OPTIONAL_FIELDS = (
    "text_config",
    "language_config",
    "model_type",
    "hidden_size",
    "d_model",
    "moe_intermediate_size",
    "intermediate_size",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "top_k",
    "layer_types",
    "quantization",
    "quantization_config",
)


@pytest.mark.parametrize("field", [*OPTIONAL_FIELDS, "all"])
def test_none_config_fields(cpu_scheduling, field):
    config = dict.fromkeys(OPTIONAL_FIELDS if field == "all" else [field])
    assert model_profile(config) == ModelProfile()
    model = SimpleNamespace(args=SimpleNamespace(**config))
    assert scheduling_value("admission_window_ms", model) == 8.0


@pytest.mark.parametrize("field", ["bits", "group_size", "mode", "all"])
@pytest.mark.parametrize("container", ["quantization", "quantization_config"])
def test_none_quantization_fields(container, field):
    quant = dict.fromkeys(["bits", "group_size", "mode"] if field == "all" else [field])
    assert model_profile({container: quant}) == ModelProfile()


@pytest.mark.parametrize(
    "value", [None, 1, True, "full_attention", {"full_attention": 1}]
)
def test_unsupported_layer_types_are_empty(value):
    assert model_profile({"layer_types": value}).attention_kinds == ()


def test_layer_types_keep_recognized_strings():
    for layers in (list, tuple):
        assert model_profile(
            {
                "layer_types": layers(
                    [
                        "linear_attention",
                        "full_attention",
                        "full_attention",
                        None,
                        [],
                        {},
                    ]
                )
            }
        ).attention_kinds == ("full_attention", "linear_attention")


@pytest.mark.parametrize(
    "field", ["text_config", "language_config", "quantization", "quantization_config"]
)
@pytest.mark.parametrize("value", [1, "invalid", [1]])
def test_unsupported_config_containers(field, value):
    assert model_profile({field: value}) == ModelProfile()


@pytest.mark.parametrize("mode", [[], {}])
def test_unsupported_quantization_mode(mode):
    assert model_profile({"quantization": {"mode": mode}}) == ModelProfile()


def test_none_primary_fields_preserve_fallback_facts():
    assert model_profile(
        {
            "text_config": None,
            "language_config": {
                "model_type": "lfm2_moe",
                "num_experts": None,
                "num_local_experts": 32,
                "num_experts_per_tok": None,
                "top_k": 4,
                "hidden_size": None,
                "d_model": 2048,
                "quantization": None,
            },
            "quantization": None,
            "quantization_config": {"bits": 4, "group_size": 64},
        }
    ) == ModelProfile(
        family="lfm2_moe",
        hidden_size=2048,
        num_experts=32,
        top_k=4,
        bits=4,
        group_size=64,
        quant_mode="affine",
    )
