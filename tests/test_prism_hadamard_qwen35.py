# SPDX-License-Identifier: Apache-2.0
"""Bonsai 2 must use its activation transform, not ordinary affine loading."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.requires_mlx


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_mllm_dispatches_bonsai2_to_owned_loader(monkeypatch, trust_remote_code):
    import mlx_vlm
    import mlx_vlm.utils

    from rapid_mlx.models.mllm import MLXMultimodalLM

    config = {"model_type": "prism_hadamard_qwen35"}
    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    custom_load = Mock(return_value=(model, processor))
    ordinary_load = Mock(
        side_effect=ValueError("Model type prism_hadamard_qwen35 not supported")
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.models.prism_hadamard_qwen35",
        SimpleNamespace(load=custom_load),
    )
    monkeypatch.setattr(mlx_vlm, "load", ordinary_load)
    monkeypatch.setattr(mlx_vlm.utils, "load_config", lambda *a, **kw: config)
    monkeypatch.setattr(
        "rapid_mlx.utils.tokenizer.augment_eos_token_ids_from_generation_config",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "rapid_mlx.utils.tokenizer.repair_byte_level_decoder", lambda *a, **kw: None
    )
    instance = MLXMultimodalLM("org/bonsai2", trust_remote_code=trust_remote_code)
    instance.load()

    ordinary_load.assert_not_called()
    custom_load.assert_called_once_with("org/bonsai2", config)
    assert instance.model is model
    assert instance.processor is processor
    assert instance.config["model_type"] == "qwen3_5"
    assert config["model_type"] == "prism_hadamard_qwen35"
    assert instance._loaded


@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("block", [0, 512, 1024])
def test_packed_matches_dense_hadamard_reference(embedding, block):
    import mlx.core as mx
    import numpy as np

    from rapid_mlx.models.prism_hadamard_qwen35 import Packed

    rng = np.random.default_rng(7)
    width = 1024
    weights = mx.array(rng.normal(size=(4, width)), dtype=mx.float16)
    arrays = mx.quantize(weights, group_size=128, bits=2)
    dense = np.array(mx.dequantize(*arrays, group_size=128, bits=2))
    signs = rng.choice([-1.0, 1.0], width).astype(np.float32)
    layer = Packed(arrays, block, mx.array(signs) if block else None, embedding)

    # Independent Sylvester matrix, rather than calling the runtime transform.
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < block:
        h = np.block([[h, h], [h, -h]])
    if block:
        h /= np.sqrt(block)
    if embedding:
        x = mx.array([[3, 0], [1, 3]])
        expected = dense[np.array(x)]
        if block:
            expected = (expected.reshape(-1, block) @ h).reshape(2, 2, width)
            expected *= signs
    else:
        inputs = rng.normal(size=(2, 1, width)).astype(np.float16)
        x = mx.array(inputs)
        if block:
            inputs = ((inputs * signs).reshape(-1, block) @ h).reshape(2, 1, width)
            inputs = inputs.astype(np.float16)
        expected = inputs.astype(np.float32) @ dense.T.astype(np.float32)
    np.testing.assert_allclose(np.array(layer(x)), expected, atol=0.08, rtol=0.003)


@pytest.mark.parametrize(
    "corruption",
    [
        "empty",
        "duplicate",
        "dtype",
        "shape",
        "nan",
        "block",
        "signs",
        "unexpected_signs",
        "missing_signs",
        "not_dict",
        "missing_path",
        "nonstring_path",
        "missing_tensor",
        "missing_block",
    ],
)
def test_packed_manifest_rejects_invalid_weights(corruption):
    import mlx.core as mx
    from mlx import nn

    from rapid_mlx.models.prism_hadamard_qwen35 import _install_packed

    model = nn.Sequential(nn.Linear(512, 4, bias=False))
    arrays = mx.quantize(
        model.layers[0].weight.astype(mx.float16), group_size=128, bits=2
    )
    prefix = "language_model.layers.0."
    weights = dict(zip((prefix + s for s in ("weight", "scales", "biases")), arrays))
    weights[prefix + "signs"] = mx.ones((512,))
    record = {"path": "layers.0", "block": 512, "embedding": False, "dtype": "float16"}
    records = [record]
    if corruption == "empty":
        records.clear()
    elif corruption == "duplicate":
        records.append(record.copy())
    elif corruption == "dtype":
        record["dtype"] = "bfloat16"
    elif corruption == "shape":
        weights[prefix + "weight"] = arrays[0][:1]
    elif corruption == "nan":
        weights[prefix + "scales"] = mx.full(arrays[1].shape, float("nan"))
    elif corruption == "block":
        record["block"] = 256
    elif corruption == "signs":
        weights[prefix + "signs"] = mx.zeros((512,))
    elif corruption == "unexpected_signs":
        record["block"] = 0
    elif corruption == "not_dict":
        records[0] = ["layers.0"]
    elif corruption == "missing_path":
        del record["path"]
    elif corruption == "nonstring_path":
        record["path"] = 123
    elif corruption == "missing_tensor":
        del weights[prefix + "weight"]
    elif corruption == "missing_block":
        del record["block"]
    else:
        del weights[prefix + "signs"]
    with pytest.raises(ValueError, match="Bonsai 2"):
        _install_packed(model, {"modules": records}, weights)
    assert isinstance(model.layers[0], nn.Linear)


def test_packed_install_and_strict_reload():
    import mlx.core as mx
    from mlx import nn

    from rapid_mlx.models.prism_hadamard_qwen35 import Packed, _install_packed

    model = nn.Sequential(nn.Linear(512, 4, bias=False))
    arrays = mx.quantize(
        model.layers[0].weight.astype(mx.float16), group_size=128, bits=2
    )
    weights = dict(
        zip(
            ("language_model.layers.0." + s for s in ("weight", "scales", "biases")),
            arrays,
        )
    )
    weights["language_model.layers.0.signs"] = mx.ones((512,))
    config = {
        "modules": [
            {"path": "layers.0", "block": 512, "embedding": False, "dtype": "float16"}
        ]
    }
    _install_packed(model, config, weights)
    model.load_weights(
        [(k.removeprefix("language_model."), v) for k, v in weights.items()],
        strict=True,
    )
    assert isinstance(model.layers[0], Packed)
    assert model(mx.ones((1, 512), dtype=mx.float16)).shape == (1, 4)


@pytest.mark.parametrize("schema_version", [1, 2])
def test_loader_uses_only_data_and_validates_schema(
    monkeypatch, tmp_path, schema_version
):
    import mlx.core as mx
    import mlx_vlm.models.qwen3_5 as qwen
    import mlx_vlm.utils
    from mlx import nn

    from rapid_mlx.models import prism_hadamard_qwen35 as prism

    model = nn.Module()
    model.language_model = nn.Sequential(nn.Linear(512, 4, bias=False))
    arrays = mx.quantize(
        model.language_model.layers[0].weight.astype(mx.float16),
        group_size=128,
        bits=2,
    )
    weights = dict(
        zip(
            ("language_model.layers.0." + s for s in ("weight", "scales", "biases")),
            arrays,
        )
    )
    weights["language_model.layers.0.signs"] = mx.ones((512,))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "untrusted.py").write_text(
        "raise AssertionError('snapshot code executed')"
    )
    config = {
        "schema_version": schema_version,
        "model_type": "prism_hadamard_qwen35",
        "base_model_type": "qwen3_5",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "gdn_activation_layout": "grouped",
        "components": {"text": True, "vision": True, "mtp": False},
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model_file": "untrusted.py",
        "requires_runtime": "untrusted.py",
        "modules": [
            {"path": "layers.0", "block": 512, "embedding": False, "dtype": "float16"}
        ],
    }
    constructor = Mock(return_value=model)
    monkeypatch.setattr(qwen, "Model", constructor)
    monkeypatch.setattr(qwen.ModelConfig, "from_dict", lambda c: c)
    monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda _: tmp_path)
    processor = object()
    monkeypatch.setattr(prism, "_build_processor", lambda _: processor)
    if schema_version != 2:
        with pytest.raises(ValueError, match="schema-v2"):
            prism.load(str(tmp_path), config)
        constructor.assert_not_called()
    else:
        loaded, loaded_processor = prism.load(str(tmp_path), config)
        assert loaded is model and loaded_processor is processor
        assert isinstance(model.language_model.layers[0], prism.Packed)
        assert constructor.call_args.args[0]["model_type"] == "qwen3_5"


def test_processor_uses_builtin_classes_without_snapshot_code(monkeypatch, tmp_path):
    import mlx_vlm.models.qwen3_5 as qwen
    import mlx_vlm.tokenizer_utils
    import mlx_vlm.utils
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    from rapid_mlx.models.prism_hadamard_qwen35 import _build_processor

    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    tokenizer = SimpleNamespace(eos_token_id=7)
    tokenizer_load = Mock(return_value=tokenizer)
    image_load = Mock(return_value=object())
    processor = SimpleNamespace(tokenizer=tokenizer)
    processor_class = Mock(return_value=processor)
    detokenizer = Mock(return_value=object())
    stopping = Mock(return_value=object())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer_load)
    monkeypatch.setattr(Qwen2VLImageProcessorPil, "from_pretrained", image_load)
    monkeypatch.setattr(qwen, "Qwen3VLProcessor", processor_class)
    monkeypatch.setattr(
        mlx_vlm.tokenizer_utils, "load_tokenizer", lambda *a, **kw: detokenizer
    )
    monkeypatch.setattr(mlx_vlm.utils, "StoppingCriteria", stopping)

    assert _build_processor(tmp_path) is processor
    tokenizer_load.assert_called_once_with(str(tmp_path), trust_remote_code=False)
    image_load.assert_called_once_with(str(tmp_path))
    processor_class.assert_called_once_with(
        image_processor=image_load.return_value,
        tokenizer=tokenizer,
        video_processor=None,
        chat_template="{{ messages }}",
    )
    detokenizer.assert_called_once_with(tokenizer)
    stopping.assert_called_once_with(7, tokenizer)
    assert processor.detokenizer is detokenizer.return_value
    assert (
        processor.stopping_criteria
        is tokenizer.stopping_criteria
        is stopping.return_value
    )


def _install_metadata_stub(monkeypatch, config):
    """Point ``read_model_metadata`` at a fixed config for the routing guard and
    neutralize the cold-start config prefetch (no network in unit tests)."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata",
        lambda _path: SimpleNamespace(config=config, snapshot_dir=None),
    )
    monkeypatch.setattr(
        "rapid_mlx.server._prefetch_config_for_text_lane_guard",
        lambda *_a, **_k: None,
    )


def test_text_lane_guard_rejects_prism_pack(monkeypatch):
    # A prism_hadamard_qwen35 pack routed to the text lane (base wheel with the
    # vision runtime present, or an explicit --no-mllm) must be rejected BEFORE
    # weights download, not left to crash deep in mlx-lm on an unknown arch.
    import rapid_mlx.server as server

    _install_metadata_stub(monkeypatch, {"model_type": "prism_hadamard_qwen35"})
    # Vision runtime is present (full install + --no-mllm): the guard passes
    # _require_mlx_vlm and then rejects on the flag.
    monkeypatch.setattr(
        "rapid_mlx.models.mllm._require_mlx_vlm", lambda *_a, **_k: None
    )

    with pytest.raises(ValueError, match="multimodal lane"):
        server._reject_text_lane_only_mllm_pack("bonsai2-27b-2bit", "/snap/bonsai2")


def test_text_lane_guard_surfaces_missing_vision_runtime_first(monkeypatch):
    # On a base wheel the pack reaches the text lane precisely because mlx-vlm
    # is missing; the guard must surface that actionable hint, not the flag msg.
    import rapid_mlx.server as server

    _install_metadata_stub(monkeypatch, {"model_type": "prism_hadamard_qwen35"})

    def _raise(*_a, **_k):
        raise ImportError("install 'rapid-mlx[vision]'")

    monkeypatch.setattr("rapid_mlx.models.mllm._require_mlx_vlm", _raise)

    with pytest.raises(ImportError, match=r"rapid-mlx\[vision\]"):
        server._reject_text_lane_only_mllm_pack("bonsai2-27b-2bit", "/snap/bonsai2")


def test_text_lane_guard_cold_start_prefetches_config(monkeypatch):
    # Cold --no-mllm skips routing-config materialization, so the FIRST metadata
    # read is empty. The guard must prefetch config.json and re-read so it still
    # rejects the pack before BatchedEngine pulls the 8.6 GB weights.
    from types import SimpleNamespace

    import rapid_mlx.server as server

    reads = iter(
        [
            SimpleNamespace(config=None, snapshot_dir=None),  # cold: nothing cached
            SimpleNamespace(  # after prefetch: config.json landed
                config={"model_type": "prism_hadamard_qwen35"}, snapshot_dir=None
            ),
        ]
    )
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata", lambda _path: next(reads)
    )
    prefetched = []
    monkeypatch.setattr(
        "rapid_mlx.server._prefetch_config_for_text_lane_guard",
        lambda ref: prefetched.append(ref),
    )
    monkeypatch.setattr(
        "rapid_mlx.models.mllm._require_mlx_vlm", lambda *_a, **_k: None
    )

    with pytest.raises(ValueError, match="multimodal lane"):
        server._reject_text_lane_only_mllm_pack(
            "bonsai2-27b-2bit", "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
        )
    assert prefetched == ["prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"]


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_5"},  # ordinary hybrid VLM — text lane is valid
        {"model_type": "qwen3"},
        None,  # cold start where even the prefetch found nothing
        {},
    ],
)
def test_text_lane_guard_passes_through_other_models(monkeypatch, config):
    import rapid_mlx.server as server

    _install_metadata_stub(monkeypatch, config)
    # Must never be consulted for a non-prism pack.
    monkeypatch.setattr(
        "rapid_mlx.models.mllm._require_mlx_vlm",
        lambda *_a, **_k: pytest.fail("_require_mlx_vlm called for non-prism pack"),
    )

    # No raise, no vision-runtime probe: an ordinary text-lane model is untouched.
    server._reject_text_lane_only_mllm_pack("some-model", "/snap/other")


def test_preflight_requires_vision_for_prism_pack_before_download(monkeypatch):
    # Cold default start (no --no-mllm): the config-only preflight must require
    # the vision runtime for a prism_hadamard_qwen35 pack BEFORE
    # _ensure_routing_config pulls the whole 8.6 GB checkpoint. is_mllm_model()
    # is False for the header-less snapshot, so this relies on exact model_type
    # recognition, not vision-weight/identity heuristics.
    from types import SimpleNamespace

    import rapid_mlx.server as server

    monkeypatch.setattr(server, "_prefetch_routing_metadata", lambda _n: "/snap/pack")
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata",
        lambda _p: SimpleNamespace(
            config={"model_type": "prism_hadamard_qwen35", "vision_config": {}},
            snapshot_dir=None,
        ),
    )
    calls = []

    def _raise(path):
        calls.append(path)
        raise ImportError("install 'rapid-mlx[vision]'")

    monkeypatch.setattr("rapid_mlx.models.mllm._require_mlx_vlm", _raise)

    with pytest.raises(ImportError, match=r"rapid-mlx\[vision\]"):
        server._preflight_vision_runtime("bonsai2-27b-2bit")
    assert calls == ["/snap/pack"]


def test_preflight_accepts_prism_pack_when_vision_runtime_is_available(monkeypatch):
    """The successful preflight path stops after validating the MLLM runtime."""
    from types import SimpleNamespace

    import rapid_mlx.server as server

    monkeypatch.setattr(server, "_prefetch_routing_metadata", lambda _n: "/snap/pack")
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata",
        lambda _p: SimpleNamespace(
            config={"model_type": "prism_hadamard_qwen35"}, snapshot_dir=None
        ),
    )
    calls = []
    monkeypatch.setattr(
        "rapid_mlx.models.mllm._require_mlx_vlm", lambda path: calls.append(path)
    )

    server._preflight_vision_runtime("bonsai2-27b-2bit")
    assert calls == ["/snap/pack"]


def test_text_lane_guard_config_prefetch_is_best_effort(monkeypatch, tmp_path):
    """Local/offline refs do not hit the Hub; remote errors remain non-fatal."""
    import rapid_mlx.server as server

    downloads = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda *args: downloads.append(args),
    )

    server._prefetch_config_for_text_lane_guard(str(tmp_path))
    assert downloads == []

    monkeypatch.setattr(
        "rapid_mlx.model_metadata.hub_offline_mode_active", lambda: True
    )
    server._prefetch_config_for_text_lane_guard("org/model")
    assert downloads == []

    monkeypatch.setattr(
        "rapid_mlx.model_metadata.hub_offline_mode_active", lambda: False
    )
    server._prefetch_config_for_text_lane_guard("org/model")
    assert downloads == [("org/model", "config.json")]

    def _fail(*_args):
        raise OSError("network unavailable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fail)
    server._prefetch_config_for_text_lane_guard("org/model")


def test_preflight_ignores_ordinary_text_checkpoint(monkeypatch):
    # A plain text checkpoint (no vision config, ordinary model_type) must not
    # be forced onto the vision runtime by the new prism recognition.
    from types import SimpleNamespace

    import rapid_mlx.server as server

    monkeypatch.setattr(server, "_prefetch_routing_metadata", lambda _n: "/snap/text")
    monkeypatch.setattr(
        "rapid_mlx.model_metadata.read_model_metadata",
        lambda _p: SimpleNamespace(config={"model_type": "qwen3"}, snapshot_dir=None),
    )
    monkeypatch.setattr(
        "rapid_mlx.models.mllm._require_mlx_vlm",
        lambda *_a, **_k: pytest.fail("_require_mlx_vlm called for a text checkpoint"),
    )

    # Returns cleanly (text lane), no vision requirement.
    server._preflight_vision_runtime("some-text-model")


def test_dynamic_residency_rejects_prism_pack_before_download(monkeypatch):
    """Runtime residency (the Desktop control plane's second model-load entry
    point) must apply the same pre-weight text-lane guard as primary startup:
    a Bonsai 2 pack misrouted to the text lane is rejected before its 8.6 GB
    download, not left to crash deep in mlx-lm on an unknown architecture.

    Drives the real ``_load_dynamic_resident_model`` down its text-lane branch
    with a non-MLLM serving checkpoint, so the ``else`` arm invokes the guard;
    a stubbed prism ``config.json`` makes the real guard raise before any
    engine construction.
    """
    import asyncio

    import rapid_mlx.server as server

    # No profile → the residency loader takes the default text modality with
    # minimal setup; a fixed resolver keeps the checkpoint identity stable.
    monkeypatch.setattr("rapid_mlx.model_aliases.resolve_profile", lambda _n: None)
    monkeypatch.setattr(
        "rapid_mlx.model_aliases.resolve_model", lambda _ref: "/snap/bonsai2"
    )
    monkeypatch.setattr(
        server,
        "_resolve_serving_checkpoint",
        lambda *_a, **_k: server._ServingCheckpoint(
            model_path="/snap/bonsai2",
            load_path="/snap/bonsai2",
            auto_text_fallback=False,
            lane_reason="text_lane_verified",
            is_mllm=False,
        ),
    )
    # Real guard, prism config in place → surfaces before engine construction.
    _install_metadata_stub(monkeypatch, {"model_type": "prism_hadamard_qwen35"})
    monkeypatch.setattr(
        "rapid_mlx.models.mllm._require_mlx_vlm", lambda *_a, **_k: None
    )
    # A BatchedEngine construction here would mean the guard failed to fire.
    monkeypatch.setattr(
        server,
        "BatchedEngine",
        lambda *_a, **_k: pytest.fail("residency built an engine for a prism pack"),
    )

    with pytest.raises(ValueError, match="multimodal lane"):
        asyncio.run(server._load_dynamic_resident_model("bonsai2-27b-2bit", None))
