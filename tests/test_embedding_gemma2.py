# SPDX-License-Identifier: Apache-2.0
"""Native text encoder contracts without downloading any model."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.requires_mlx
mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from rapid_mlx.embedding import (
    EmbeddingEngine,
    EmbeddingInputTooLongError,
    EmbeddingUnsupportedMediaError,
    require_mlx_embeddings_or_exit,
)
from rapid_mlx.embedding_backend import is_native_embedding_model


@pytest.mark.parametrize(
    "name",
    [
        "google/embeddinggemma-2",
        "embeddinggemma-2-bf16",
        "embeddinggemma-2-4bit",
        "mlx-community/embeddinggemma-2-bf16",
        "mlx-community/embeddinggemma-2-4bit",
    ],
)
def test_native_guard_never_requires_legacy_extra(monkeypatch, name):
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda key: object() if key == "mlx_vlm" else None
    )
    monkeypatch.setattr(
        "rapid_mlx.embedding.mlx_embeddings_available",
        lambda: pytest.fail("legacy probe"),
    )
    require_mlx_embeddings_or_exit(name)


def test_native_missing_runtime_exits_with_own_hint(monkeypatch, capsys):
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    with pytest.raises(SystemExit) as error:
        require_mlx_embeddings_or_exit("embeddinggemma-2-4bit")
    assert error.value.code == 2
    assert "rapid-mlx==" in capsys.readouterr().err


def test_legacy_loader_dispatch_is_preserved(monkeypatch):
    import sys

    from rapid_mlx.models.embedding_gemma2 import loader

    calls = []
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    tokenizer = Tokenizer()
    monkeypatch.setitem(
        sys.modules,
        "mlx_embeddings",
        SimpleNamespace(load=lambda name: (calls.append(name) or model, tokenizer)),
    )
    monkeypatch.setattr(loader, "load", lambda _: pytest.fail("native loader"))
    engine = EmbeddingEngine("legacy-embedding-model")
    engine.load()
    assert calls == ["legacy-embedding-model"]
    assert engine.is_loaded
    assert engine.effective_max_length == 512


def test_local_dispatch_checks_actual_architecture(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "embedding_gemma2"}))
    assert is_native_embedding_model(str(tmp_path))
    path.write_text(json.dumps({"model_type": "gemma3"}))
    assert not is_native_embedding_model(str(tmp_path))
    assert not is_native_embedding_model("mlx-community/embeddinggemma-300m-6bit")
    assert not is_native_embedding_model("mlx-community/embeddinggemma-2-mxfp4")


@pytest.mark.parametrize(
    "setting,expected", [("auto", 8192), (1024, 1024), (262144, 8192)]
)
def test_native_supported_window_is_not_rotary_range(setting, expected):
    engine = EmbeddingEngine("embeddinggemma-2-4bit", max_length=setting)
    engine._model = SimpleNamespace(
        model_type="embedding_gemma2",
        config=SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=262144)
        ),
    )
    engine._resolve_effective_max_length()
    assert engine.effective_max_length == expected


def tiny_config():
    return {
        "model_type": "embedding_gemma2",
        "dtype": "bfloat16",
        "text_config": {
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "hidden_size_per_layer_input": 0,
            "embedding_dim": 768,
            "per_layer_config": {"00": {"head_dim": 32, "num_key_value_heads": 1}},
        },
        "vision_config": None,
        "audio_config": None,
    }


class Tokenizer:
    # A raw private backend is not callable, unlike the outer HF tokenizer.
    _tokenizer = object()
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        ids = [[int(token) for token in text.split()] for text in texts]
        if not kwargs.get("padding"):
            return {"input_ids": ids}
        ids = [seq[: kwargs["max_length"]] for seq in ids]
        width = max(map(len, ids))
        return {
            "input_ids": np.array([seq + [0] * (width - len(seq)) for seq in ids]),
            "attention_mask": np.array(
                [[1] * len(seq) + [0] * (width - len(seq)) for seq in ids]
            ),
        }

    def encode(self, text):
        return [int(token) for token in text.split()]


@pytest.fixture
def native_engine():
    from rapid_mlx.models.embedding_gemma2.config import ModelConfig
    from rapid_mlx.models.embedding_gemma2.embedding_gemma2 import Model

    engine = EmbeddingEngine("embeddinggemma-2-4bit")
    engine._model = Model(ModelConfig.from_dict(tiny_config()))
    engine._tokenizer = Tokenizer()
    engine._resolve_effective_max_length()
    return engine


def test_native_text_token_padding_pooling_and_cleanup(native_engine, monkeypatch):
    calls = []
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append(True))
    text = np.array(native_engine.embed(["2 3 4", "2 5"]))
    tokens = np.array(native_engine.embed_tokens([[2, 3, 4], [2, 5]]))
    assert text.shape == (2, 768)
    assert np.isfinite(text).all()
    assert np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-6)
    assert np.allclose(text, tokens)
    single = np.array(native_engine.embed_tokens([[2, 5]]))[0]
    assert np.allclose(tokens[1], single, atol=1e-6)
    assert native_engine.count_tokens(["2 3 4", "2 5"]) == 5
    assert len(calls) == 3


@pytest.mark.parametrize(
    "token", [258880, 258881, 258884, 255999, 258882, 256000, 258883]
)
@pytest.mark.parametrize("mode", ["text", "tokens"])
def test_media_tokens_fail_before_truncation(native_engine, token, mode):
    native_engine.effective_max_length = 1
    with pytest.raises(EmbeddingUnsupportedMediaError, match="text/code only"):
        if mode == "text":
            native_engine.embed([f"2 {token}"])
        else:
            native_engine.embed_tokens([[2, token]])
    assert native_engine.num_truncations == 0


def test_native_overflow_policies_are_preserved(native_engine):
    native_engine.effective_max_length = 2
    native_engine.overflow_policy = "error"
    with pytest.raises(EmbeddingInputTooLongError) as error:
        native_engine.embed(["2 3 4"])
    assert error.value.observed_tokens == 3
    assert error.value.allowed_tokens == 2
    native_engine.overflow_policy = "truncate"
    assert len(native_engine.embed_tokens([[2, 3, 4]])[0]) == 768
    assert native_engine.num_truncations == 1
    assert native_engine.count_tokens(["2 3 4"]) == 2


@pytest.mark.parametrize("quantized", [False, True])
def test_strict_native_loader_and_engine_dispatch(tmp_path, monkeypatch, quantized):
    from rapid_mlx.models.embedding_gemma2 import loader
    from rapid_mlx.models.embedding_gemma2.config import ModelConfig
    from rapid_mlx.models.embedding_gemma2.embedding_gemma2 import Model

    config = tiny_config()
    model = Model(ModelConfig.from_dict(config))
    if quantized:
        nn.quantize(model, group_size=64, bits=4)
        config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
        config["quantization"]["language_model.embedding_projection"] = {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
        }
    model.save_weights(str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        loader.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    engine = EmbeddingEngine(str(tmp_path))
    engine.load()
    assert engine.is_loaded
    assert engine.effective_max_length == 8192
    assert len(engine.embed("2 3")[0]) == 768
    weights = mx.load(str(tmp_path / "model.safetensors"))
    weights.pop(next(iter(weights)))
    # Do not overwrite a lazily mapped source while reading it for the write.
    mx.save_safetensors(str(tmp_path / "incomplete.safetensors"), weights)
    (tmp_path / "incomplete.safetensors").replace(tmp_path / "model.safetensors")
    with pytest.raises(ValueError):
        loader.load(str(tmp_path))


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "gemma4"},
        {"dtype": "float16"},
        {"quantization": {"bits": 4, "group_size": 64, "mode": "mxfp4"}},
        {"quantization": {"bits": 8, "group_size": 64, "mode": "affine"}},
        {
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "language_model.embedding_projection": {
                    "bits": 8,
                    "group_size": 64,
                    "mode": "affine",
                },
            }
        },
    ],
)
def test_native_loader_rejects_other_architecture_precision_and_quantization(
    tmp_path, change
):
    from rapid_mlx.models.embedding_gemma2.loader import load

    (tmp_path / "config.json").write_text(json.dumps({**tiny_config(), **change}))
    with pytest.raises(ValueError):
        load(str(tmp_path))


@pytest.mark.parametrize("shard", ["../weights.safetensors", "/weights.safetensors"])
def test_native_loader_rejects_external_index_shards(tmp_path, shard):
    from rapid_mlx.models.embedding_gemma2.loader import load

    (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"language_model.norm.weight": shard}})
    )
    with pytest.raises(ValueError, match="model directory"):
        load(str(tmp_path))


@pytest.mark.parametrize("indexed", [False, True])
def test_native_loader_rejects_missing_weights(tmp_path, indexed):
    from rapid_mlx.models.embedding_gemma2.loader import load

    (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"language_model.norm.weight": "missing.safetensors"}}
            )
        )
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path))


def test_native_loader_rejects_actual_fp16_weight(tmp_path):
    from rapid_mlx.models.embedding_gemma2.loader import load

    (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"language_model.norm.weight": mx.ones(64, dtype=mx.float16)},
    )
    with pytest.raises(ValueError, match="never FP16"):
        load(str(tmp_path))


def test_native_text_subset_filters_only_unused_weights_and_rejects_media(
    native_engine,
):
    model = native_engine._model
    weights = {
        "model.language_model.norm.weight": mx.ones(64),
        "model.vision_tower.unused": mx.ones(1),
        "model.audio_tower.unused": mx.ones(1),
        "model.embed_vision.unused": mx.ones(1),
        "model.embed_audio.unused": mx.ones(1),
        "model.language_model.rotary_emb.inv_freq": mx.ones(1),
        "model.language_model.unknown": mx.ones(1),
    }
    assert set(model.sanitize(weights)) == {
        "language_model.norm.weight",
        "language_model.unknown",
    }
    assert model.layers is model.language_model.layers
    assert not hasattr(model, "vision_tower")
    assert not hasattr(model, "audio_tower")
    ids = mx.array([[2, 3]])
    assert np.allclose(
        np.array(model(ids, mask=mx.ones_like(ids)).text_embeds),
        np.array(model(ids).text_embeds),
    )
    for name in ("pixel_values", "pixel_values_videos", "input_features"):
        with pytest.raises(ValueError, match="text/code only"):
            model(ids, **{name: mx.ones(1)})


def test_native_per_layer_projection_and_bidirectional_masks():
    from rapid_mlx.models.embedding_gemma2.config import ModelConfig, TextConfig
    from rapid_mlx.models.embedding_gemma2.embedding_gemma2 import Model

    with pytest.raises(ValueError, match="one entry"):
        TextConfig(num_hidden_layers=2, layer_types=["full_attention"])
    config = tiny_config()
    config["text_config"].update(
        num_hidden_layers=2,
        hidden_size_per_layer_input=16,
        layer_types=["sliding_attention", "full_attention"],
        per_layer_config=None,
    )
    model = Model(ModelConfig.from_dict(config))
    ids = mx.array([[2, 3, 0]])
    padded = model(ids, attention_mask=mx.array([[1, 1, 0]])).text_embeds
    unpadded = model(mx.array([[2, 3]])).text_embeds
    assert np.allclose(np.array(padded), np.array(unpadded), atol=1e-6)
    assert np.isfinite(np.array(padded)).all()
    with pytest.raises(ValueError, match="same shape"):
        model(ids, attention_mask=mx.array([[1, 1]]))


@pytest.mark.parametrize("alias", ["embeddinggemma-2-bf16", "embeddinggemma-2-4bit"])
def test_native_alias_task_and_backend_are_not_chat(alias):
    from rapid_mlx.catalog.legacy import build_legacy_catalog_snapshot
    from rapid_mlx.model_aliases import resolve_profile

    profile = resolve_profile(alias)
    assert profile.modality == "embedding"
    assert not profile.supports_image_input
    entries = {a["alias"]: a for a in build_legacy_catalog_snapshot()["aliases"]}
    caps = entries[alias]["capabilities"]
    assert caps["task_types"] == ["embedding"]
    assert caps["operation_modes"] == ["embed"]
    assert caps["runtime_adapter"] == "rapid_mlx/embedding_gemma2"
    assert caps["is_text_only"]


def test_native_cli_boot_helper_resolves_alias_without_legacy_extra(monkeypatch):
    from rapid_mlx import cli

    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name == "mlx_vlm" else None
    )
    monkeypatch.setattr(
        "rapid_mlx.embedding.mlx_embeddings_available",
        lambda: pytest.fail("legacy probe"),
    )
    calls = []
    args = SimpleNamespace(embedding_model="embeddinggemma-2-4bit")
    cli._load_embedding_model_or_exit(args, lambda name, **kw: calls.append((name, kw)))
    assert calls[0][0] == "mlx-community/embeddinggemma-2-4bit"
    assert calls[0][1] == {
        "lock": True,
        "max_length": "auto",
        "overflow_policy": "truncate",
        "max_inputs": 2048,
    }


@pytest.mark.parametrize("available", [False, True])
def test_legacy_cli_boot_helper_keeps_zero_argument_optional_guard(
    monkeypatch, available
):
    from rapid_mlx import cli, embedding

    calls = []
    guard_calls = []
    original_guard = embedding.require_mlx_embeddings_or_exit

    def guard(*args):
        guard_calls.append(args)
        return original_guard(*args)

    monkeypatch.setattr(embedding, "require_mlx_embeddings_or_exit", guard)
    monkeypatch.setattr(embedding, "mlx_embeddings_available", lambda: available)
    args = SimpleNamespace(embedding_model="legacy-embedding-model")
    load = lambda name, **kwargs: calls.append((name, kwargs))
    if available:
        cli._load_embedding_model_or_exit(args, load)
        assert calls == [
            (
                "legacy-embedding-model",
                {
                    "lock": True,
                    "max_length": "auto",
                    "overflow_policy": "truncate",
                    "max_inputs": 2048,
                },
            )
        ]
    else:
        with pytest.raises(SystemExit) as error:
            cli._load_embedding_model_or_exit(args, load)
        assert error.value.code == 2
        assert calls == []
    assert guard_calls == [()]


@pytest.mark.parametrize("model", [None, SimpleNamespace(model_type="legacy")])
def test_media_token_guard_is_native_only(model):
    engine = EmbeddingEngine("legacy-embedding-model")
    engine._model = model
    # Legacy models and unloaded engines have no native media-token config.
    assert engine._reject_media_tokens([[258880, 258881]]) is None


@pytest.mark.parametrize("entrypoint", ["serve", "server"])
@pytest.mark.parametrize("name", ["embeddinggemma-2-4bit", "legacy-model"])
def test_rendered_early_boot_guard_selects_only_its_backend(
    monkeypatch, entrypoint, name
):
    import ast
    import inspect

    from rapid_mlx import cli, server

    function = cli.serve_command if entrypoint == "serve" else server.main
    body = ast.parse(inspect.getsource(function)).body[0].body
    guard = next(
        node
        for node in body
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "embedding"
            and any(
                alias.name == "require_mlx_embeddings_or_exit" for alias in child.names
            )
            for child in node.body
        )
    )
    probes = []
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda module: probes.append(module) or object()
    )
    monkeypatch.setattr(
        "rapid_mlx.embedding.mlx_embeddings_available",
        lambda: probes.append("mlx_embeddings") or True,
    )
    rendered = ast.Module(body=[guard], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(rendered), "actual-boot-guard", "exec"),
        {
            "__package__": "rapid_mlx",
            "args": SimpleNamespace(embedding_model=name),
        },
    )
    assert probes == (
        ["mlx_vlm"] if name.startswith("embeddinggemma-2") else ["mlx_embeddings"]
    )


@pytest.mark.parametrize("entrypoint", ["serve", "server"])
@pytest.mark.parametrize("name", ["embeddinggemma-2-4bit", "legacy-model"])
def test_actual_entrypoint_reaches_architecture_guard_before_loading(
    monkeypatch, entrypoint, name
):
    from rapid_mlx import cli, server
    from rapid_mlx.telemetry import consent_runtime, server_start, track

    class ReachedGuardError(Exception):
        pass

    calls = []

    def stop_at_guard(*args):
        calls.append(args)
        raise ReachedGuardError

    monkeypatch.setattr(
        "rapid_mlx.embedding.require_mlx_embeddings_or_exit", stop_at_guard
    )
    monkeypatch.setattr("rapid_mlx.routes.video.configure_video_jobs", lambda _: None)
    monkeypatch.setattr(
        "rapid_mlx._parent_watchdog.install_parent_watchdog", lambda _: None
    )
    monkeypatch.setattr(cli, "_resolve_serve_port", lambda *args, **kw: 8000)
    monkeypatch.setattr(consent_runtime, "startup", lambda **kw: None)
    monkeypatch.setattr(track, "set_surface_for_role", lambda _: True)
    monkeypatch.setattr(server_start, "attempted", lambda *args, **kw: None)
    monkeypatch.setattr(server_start, "set_failure_stage", lambda _: None)
    with pytest.raises(ReachedGuardError):
        if entrypoint == "serve":
            cli.serve_command(
                SimpleNamespace(model="unregistered-test-model", embedding_model=name)
            )
        else:
            monkeypatch.setattr(
                "sys.argv",
                [
                    "server",
                    "--model",
                    "unregistered-test-model",
                    "--embedding-model",
                    name,
                ],
            )
            server.main.__wrapped__()
    assert calls == ([(name,)] if name.startswith("embeddinggemma-2") else [()])


def test_native_media_error_has_explicit_wire_contract(monkeypatch):
    from tests.test_embeddings_route import _build_embed_app

    engine = SimpleNamespace(effective_max_length=8192, count_tokens=lambda _: 1)

    def reject(_):
        raise EmbeddingUnsupportedMediaError("text/code only")

    engine.embed = reject
    client, restore = _build_embed_app(
        monkeypatch,
        engine,
        embedding_model_locked="mlx-community/embeddinggemma-2-4bit",
    )
    try:
        with client:
            response = client.post(
                "/v1/embeddings", json={"model": "default", "input": "<|image|>"}
            )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "unsupported_embedding_modality"
            response = client.post(
                "/v1/embeddings",
                json={"model": "default", "input": {"image": "unused.jpg"}},
            )
            # Rapid's existing OpenAI exception handler maps schema errors to 400.
            assert response.status_code == 400
            assert response.json()["error"]["code"] != "unsupported_embedding_modality"
    finally:
        restore()


def tiny_vision_config():
    config = tiny_config()
    config.update(
        image_token_id=20,
        boi_token_id=21,
        eoi_token_id=22,
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "default_output_length": 4,
            "position_embedding_size": 16,
        },
    )
    return config


class ImageTokenizer(Tokenizer):
    def decode(self, tokens):
        return " ".join(map(str, tokens)) + " "

    def __call__(self, texts, **kwargs):
        ids = [self.encode(text) for text in texts]
        width = max(map(len, ids))
        return {
            "input_ids": np.array([row + [0] * (width - len(row)) for row in ids]),
            "attention_mask": np.array(
                [[1] * len(row) + [0] * (width - len(row)) for row in ids]
            ),
        }


@pytest.fixture
def vision_engine(tmp_path, monkeypatch):
    from rapid_mlx.models.embedding_gemma2.config import ModelConfig
    from rapid_mlx.models.embedding_gemma2.embedding_gemma2 import Model

    engine = EmbeddingEngine("embeddinggemma-2-4bit", vision=True)
    engine._model = Model(ModelConfig.from_dict(tiny_vision_config()), vision=True)
    engine._tokenizer = ImageTokenizer()
    engine._resolve_effective_max_length()
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "image_processor": {
                    "do_resize": False,
                    "patch_size": 2,
                    "pooling_kernel_size": 2,
                    "max_soft_tokens": 4,
                }
            }
        )
    )
    monkeypatch.setattr("mlx_vlm.utils.get_model_path", lambda _: tmp_path)
    return engine


@pytest.mark.parametrize("variable_size", [False, True])
def test_native_vision_images_paths_padding_and_cached_processor(
    vision_engine, tmp_path, monkeypatch, variable_size
):
    from PIL import Image

    cleanup = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleanup.append(True))
    first = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    second = Image.new("L", (8, 4) if variable_size else (4, 4), 128)
    path = tmp_path / "image.png"
    second.save(path)
    embeddings = np.array(vision_engine.embed_images([first, path]))
    processor = vision_engine._image_processor
    assert embeddings.shape == (2, 768)
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
    single = np.array(vision_engine.embed_images([first]))[0]
    assert np.allclose(embeddings[0], single, atol=1e-6)
    assert vision_engine._image_processor is processor
    assert cleanup == [True, True]
    # Opt-in image processing preserves the text path and media rejection.
    assert len(vision_engine.embed("2 3")[0]) == 768
    with pytest.raises(EmbeddingUnsupportedMediaError, match="text/code only"):
        vision_engine.embed_tokens([[2, 20]])
    weights = {
        "model.vision_tower.keep": mx.ones(1),
        "model.audio_tower.drop": mx.ones(1),
    }
    assert set(vision_engine._model.sanitize(weights)) == {"vision_tower.keep"}


def test_native_vision_support_guards_and_empty_input(native_engine, vision_engine):
    from rapid_mlx.models.embedding_gemma2.config import ModelConfig
    from rapid_mlx.models.embedding_gemma2.embedding_gemma2 import Model

    with pytest.raises(EmbeddingUnsupportedMediaError, match="vision=True"):
        native_engine.embed_images([])
    native_engine.vision = True
    with pytest.raises(EmbeddingUnsupportedMediaError, match="does not support images"):
        native_engine.embed_images([])
    assert vision_engine.embed_images([]) == []
    assert vision_engine._image_processor is None
    with pytest.raises(ValueError, match="does not support images"):
        Model(ModelConfig.from_dict(tiny_config()), vision=True)


def test_native_vision_placeholder_validation_and_overflow(vision_engine, monkeypatch):
    from PIL import Image

    image = Image.new("RGB", (4, 4), "red")
    vision_engine.effective_max_length = 2
    vision_engine.overflow_policy = "truncate"
    with pytest.raises(EmbeddingInputTooLongError) as error:
        vision_engine.embed_images([image])
    assert error.value.observed_tokens == 3
    assert error.value.allowed_tokens == 2
    assert vision_engine.num_truncations == 0
    vision_engine.effective_max_length = 8192
    monkeypatch.setattr(vision_engine._tokenizer, "decode", lambda _: "2 ")
    with pytest.raises(ValueError, match="expand image placeholders"):
        vision_engine.embed_images([image])


def test_native_vision_scatter_preserves_order_and_checks_count():
    from rapid_mlx.models.embedding_gemma2.embedding_gemma2 import Model

    original = mx.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ids = mx.array([[20, 2], [3, 20]])
    features = mx.array([[10.0, 11.0], [12.0, 13.0]])
    scattered = Model._scatter(original, ids, 20, features)
    assert np.array_equal(np.array(scattered), [[[10, 11], [3, 4]], [[5, 6], [12, 13]]])
    with pytest.raises(ValueError, match="feature count"):
        Model._scatter(original, ids, 20, features[:1])
    assert Model._scatter(original, ids, 99, mx.zeros((0, 2))) is original


def test_native_vision_forward_rejects_other_media_and_clears_failed_pass(
    vision_engine, monkeypatch
):
    from PIL import Image

    model = vision_engine._model
    for name in ("pixel_values_videos", "input_features"):
        with pytest.raises(ValueError, match="text/code only"):
            model(mx.array([[2, 3]]), **{name: mx.ones(1)})
    cleanup = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleanup.append(True))

    def fail(*args, **kwargs):
        raise RuntimeError("vision forward failed")

    monkeypatch.setattr(type(model), "__call__", fail)
    with pytest.raises(RuntimeError, match="vision forward failed"):
        vision_engine.embed_images([Image.new("RGB", (4, 4), "red")])
    assert cleanup == [True]


def test_native_vision_loader_dispatch_and_legacy_rejection(vision_engine, monkeypatch):
    import sys

    from rapid_mlx.models.embedding_gemma2 import loader

    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return vision_engine._model, vision_engine._tokenizer

    monkeypatch.setattr(loader, "load", load)
    engine = EmbeddingEngine("embeddinggemma-2-4bit", vision=True)
    engine.load()
    assert engine.is_loaded
    assert engine.effective_max_length == 8192
    assert calls == [("embeddinggemma-2-4bit", {"vision": True})]
    monkeypatch.setitem(
        sys.modules,
        "mlx_embeddings",
        SimpleNamespace(load=lambda _: pytest.fail("legacy weights loaded")),
    )
    legacy = EmbeddingEngine("legacy-model", vision=True)
    with pytest.raises(ValueError, match="requires EmbeddingGemma 2"):
        legacy.load()
