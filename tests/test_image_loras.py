"""User image LoRA library and request contracts; no model downloads."""

import asyncio
import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rapid_mlx.api.models import ImageGenerationRequest
from rapid_mlx.image.engine import ImageGenerationEngine, ImageRequestError
from rapid_mlx.image.loras import (
    LORA_ADAPTER,
    ImageLora,
    _library,
    list_loras,
    resolve_loras,
)
from rapid_mlx.runtime.image_lane import ImageEngine


def test_library_missing_and_name_only(tmp_path, monkeypatch):
    from rapid_mlx.image.loras import list_loras, resolve_loras

    monkeypatch.setenv("RAPID_MLX_LORA_DIR", str(tmp_path / "missing"))
    assert list_loras() == []
    with pytest.raises(ValueError, match="name"):
        resolve_loras([{"name": "../escape"}])


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_LORA_DIR", str(tmp_path))
    (tmp_path / "style.safetensors").write_bytes(b"adapter")
    return tmp_path


def test_default_library(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_LORA_DIR", raising=False)
    from pathlib import Path

    assert _library() == (Path.home() / ".rapid-mlx/loras").resolve()


@pytest.mark.parametrize("name", ["", "../x", "x..y", "x/y", "x\\y", "x\0y", "/tmp/x"])
def test_invalid_names(library, name):
    with pytest.raises(ValueError):
        resolve_loras([{"name": name}])


def test_listing_and_symlink_boundaries(library, tmp_path):
    (library / "alpha.safetensors").write_bytes(b"ab")
    (library / "folder.safetensors").mkdir()
    (library / "nested").mkdir()
    (library / "nested" / "hidden.safetensors").touch()
    (library / "ignored.txt").touch()
    (library / "bad..name.safetensors").touch()
    (library / "inside.safetensors").symlink_to(library / "style.safetensors")
    (library / "broken.safetensors").symlink_to(library / "missing")
    outside = tmp_path.parent / (tmp_path.name + "-outside.safetensors")
    outside.write_bytes(b"secret")
    (library / "escape.safetensors").symlink_to(outside)
    (library / "loop.safetensors").symlink_to(library / "loop.safetensors")
    assert list_loras() == [
        {"id": "alpha", "object": "image.lora", "bytes": 2},
        {"id": "inside", "object": "image.lora", "bytes": 7},
        {"id": "style", "object": "image.lora", "bytes": 7},
    ]
    for name in ("escape", "folder", "broken", "loop", "missing"):
        with pytest.raises(ValueError, match="Available LoRAs"):
            resolve_loras([{"name": name}])
    assert resolve_loras([{"name": "inside"}]) == resolve_loras([{"name": "style"}])


def test_available_names_truncated(library):
    for i in range(25):
        (library / f"a{i:02}.safetensors").touch()
    with pytest.raises(ValueError) as exc:
        resolve_loras([{"name": "missing"}])
    assert "a19" in str(exc.value)
    assert "a20" not in str(exc.value)


def test_scale_defaults_boundaries_and_order(library):
    (library / "other.safetensors").touch()
    assert ImageLora(name="style").scale == 1.0
    assert resolve_loras(None) == resolve_loras([]) == ()
    key = resolve_loras([{"name": "style", "scale": -2}, {"name": "other", "scale": 2}])
    assert key == (
        (
            str(library / "style.safetensors"),
            (library / "style.safetensors").stat().st_mtime_ns,
            -2.0,
        ),
        (
            str(library / "other.safetensors"),
            (library / "other.safetensors").stat().st_mtime_ns,
            2.0,
        ),
    )


_BAD_REQUESTS = [
    [{"name": "style"}] * 5,
    [{"name": "style"}] * 2,
    [{"name": "../style"}],
    [{"name": "style", "scale": 2.1}],
    [{"name": "style", "scale": -2.1}],
    [{"name": "style", "scale": "NaN"}],
    [{"name": "style", "scale": "Infinity"}],
    [{"name": "style", "scale": "-Infinity"}],
    [{"name": "style", "scale": "bad"}],
    [{}],
    {"name": "style"},
]


@pytest.mark.parametrize("loras", _BAD_REQUESTS)
def test_schema_rejections(loras):
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="x", loras=loras)


@pytest.fixture
def client():
    from rapid_mlx.server import app

    return TestClient(app)


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    return buf.getvalue()


def _route_engine(monkeypatch, name="flux2-klein"):
    engine = ImageEngine(name)
    monkeypatch.setattr("rapid_mlx.routes.images._image_engine", lambda *a, **k: engine)
    monkeypatch.setattr(engine, "_emit_model_served", lambda: None)
    return engine


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("loras", _BAD_REQUESTS + [[{"name": "unknown"}]])
def test_route_validation_400(client, library, monkeypatch, edit, loras):
    engine = _route_engine(monkeypatch)
    load = Mock(side_effect=AssertionError("invalid requests must not load weights"))
    monkeypatch.setattr(engine._engine, "_ensure_loaded", load)
    if edit:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "x", "loras": json.dumps(loras)},
            files={"image": ("x.png", _png(), "image/png")},
        )
    else:
        response = client.post(
            "/v1/images/generations", json={"prompt": "x", "loras": loras}
        )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    load.assert_not_called()


@pytest.mark.parametrize("payload", ["{", "null", "42", '"style"'])
def test_edit_invalid_json(client, library, monkeypatch, payload):
    _route_engine(monkeypatch)
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "x", "loras": payload},
        files={"image": ("x.png", _png())},
    )
    assert response.status_code == 400
    assert "loras" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "name", ["sdxl-base", "sd35-large", "hidream-o1-dev", "bonsai-image"]
)
@pytest.mark.parametrize("edit", [False, True])
def test_unsupported_family_400(client, library, monkeypatch, name, edit):
    engine = _route_engine(monkeypatch, name)
    if edit:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "x", "loras": '[{"name":"style"}]'},
            files={"image": ("x.png", _png())},
        )
    else:
        response = client.post(
            "/v1/images/generations", json={"prompt": "x", "loras": [{"name": "style"}]}
        )
    assert response.status_code == 400
    assert name in response.json()["error"]["message"]
    assert engine._engine._model is None
    assert engine._engine._resolve_loras([]) == ()
    with pytest.raises(ImageRequestError, match="does not support image LoRAs"):
        engine._engine.generate(prompt="x", loras=[{"name": "style"}])


def test_list_endpoint(client, library):
    assert client.get("/v1/images/loras").json() == {
        "object": "list",
        "data": [{"id": "style", "object": "image.lora", "bytes": 7}],
    }


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("loras", [None, [], [{"name": "style", "scale": 0.75}]])
def test_routes_forward_to_worker(client, library, monkeypatch, edit, loras):
    engine = _route_engine(monkeypatch)
    seen = []

    def generate(**kwargs):
        engine._engine.validate_request(**kwargs)
        seen.append(kwargs)
        return _png()

    monkeypatch.setattr(engine._engine, "generate", generate)
    monkeypatch.setattr(engine._engine, "_ensure_loaded", lambda **k: None)
    payload = {"prompt": "x"}
    if loras is not None:
        payload["loras"] = json.dumps(loras) if edit else loras
    if edit:
        response = client.post(
            "/v1/images/edits", data=payload, files={"image": ("x.png", _png())}
        )
    else:
        response = client.post("/v1/images/generations", json=payload)
    assert response.status_code == 200, response.text
    assert seen[0].get("loras") == (
        LORA_ADAPTER.validate_python(loras) if loras else None
    )
    assert bool(seen[0].get("image_paths")) == edit


def _mock_load(monkeypatch, engine):
    monkeypatch.setattr(engine, "_ensure_runtime_assets", lambda: None)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    monkeypatch.setattr("rapid_mlx.model_downloads.require_local", lambda name: None)


def test_cache_reuse_rebuild_and_clear(library, monkeypatch):
    import weakref

    from rapid_mlx.image import engine as module

    engine = ImageGenerationEngine("flux2-klein")
    _mock_load(monkeypatch, engine)
    builds = []
    released = []

    class Model:
        pass

    def build():
        assert all(ref() is None for ref in builds)
        model = Model()
        builds.append(weakref.ref(model))
        return model

    monkeypatch.setattr(engine, "_build_model", build)
    monkeypatch.setattr(engine, "_build_edit_model", build)
    monkeypatch.setattr(
        module, "_release_allocator_cache", lambda: released.append(True)
    )
    loras = [{"name": "style"}]
    engine._ensure_loaded(loras=loras)
    engine._ensure_loaded(loras=loras)
    assert len(builds) == 1
    engine._ensure_loaded(loras=[{"name": "style", "scale": 0.5}])
    assert len(builds) == 2
    path = library / "style.safetensors"
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1))
    engine._ensure_loaded(loras=[{"name": "style", "scale": 0.5}])
    assert len(builds) == 3
    engine._ensure_loaded(loras=loras, for_edit=True)
    engine._ensure_loaded(loras=loras, for_edit=True)
    assert len(builds) == 4
    engine._ensure_loaded()
    assert len(builds) == 5
    assert engine._lora_key == ()
    assert len(released) == 4
    engine._ensure_loaded(loras=[])
    assert len(builds) == 5


def test_offline_generate_and_lane_stop(library, monkeypatch):
    from contextlib import nullcontext

    from rapid_mlx.image import engine as module

    monkeypatch.setattr(module, "_bounded_generation_cache", nullcontext)
    adapter = ImageEngine("flux2-klein")
    engine = adapter._engine
    _mock_load(monkeypatch, engine)
    model = SimpleNamespace(
        generate_image=lambda **k: SimpleNamespace(image=Image.new("RGB", (8, 8)))
    )
    keys = []

    def build():
        keys.append(engine._lora_key)
        return model

    monkeypatch.setattr(engine, "_build_model", build)
    monkeypatch.setattr(adapter, "_emit_model_served", lambda: None)
    loras = [{"name": "style"}]
    assert engine.generate(prompt="x", loras=loras).startswith(b"\x89PNG")
    assert keys == [resolve_loras(loras)]
    assert adapter.generate(prompt="x", loras=loras).startswith(b"\x89PNG")
    assert len(keys) == 1
    asyncio.run(adapter.stop())
    assert engine._model is None
    assert engine._lora_key == ()
    assert adapter.generate_with_performance(prompt="x", loras=loras)[0].startswith(
        b"\x89PNG"
    )
    assert keys == [resolve_loras(loras)] * 2


@pytest.mark.parametrize(
    "name,module_path,class_name,edit,variant",
    [
        (
            "flux2-klein",
            "flux2.variants.txt2img.flux2_klein",
            "Flux2Klein",
            False,
            "base",
        ),
        (
            "flux2-klein",
            "flux2.variants.edit.flux2_klein_edit",
            "Flux2KleinEdit",
            True,
            "base",
        ),
        (
            "flux3-klein",
            "flux3.variants.txt2img.flux3_klein",
            "Flux3Klein",
            False,
            "base",
        ),
        (
            "flux3-klein",
            "flux3.variants.edit.flux3_klein_edit",
            "Flux3KleinEdit",
            True,
            "base",
        ),
        ("z-image", "z_image.variants.z_image", "ZImage", False, "base"),
        ("qwen-image", "qwen.variants.txt2img.qwen_image", "QwenImage", False, "base"),
        (
            "qwen-image-edit",
            "qwen.variants.edit.qwen_image_edit",
            "QwenImageEdit",
            True,
            "base",
        ),
        (
            "qwen-image-2.1",
            "qwen21.variants.txt2img.qwen_image_21",
            "QwenImage21",
            False,
            "base",
        ),
        (
            "qwen-image-2.1",
            "qwen21.variants.txt2img.qwen_image_21",
            "QwenImage21",
            True,
            "base",
        ),
        (
            "qwen-image-2.1",
            "qwen21.variants.edit.qwen_image_21_edit",
            "QwenImage21Edit",
            True,
            "references",
        ),
        (
            "qwen-image-2.1",
            "qwen21.variants.txt2img.qwen_image_21",
            "QwenImage21",
            False,
            "full_q4",
        ),
        (
            "qwen-image-2.1",
            "qwen21.variants.txt2img.qwen_image_21",
            "QwenImage21",
            False,
            "turbo",
        ),
        ("flux-schnell", "flux.variants.txt2img.flux", "Flux1", False, "base"),
        ("flux.1-dev", "flux.variants.txt2img.flux", "Flux1", False, "base"),
    ],
)
def test_all_mflux_builders_apply_loras(
    library, monkeypatch, name, module_path, class_name, edit, variant
):
    from types import ModuleType

    def install(name, **attrs):
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix not in sys.modules:
                module = ModuleType(prefix)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, prefix, module)
        for key, value in attrs.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)

    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            vae=SimpleNamespace(),
            transformer=SimpleNamespace(
                norm_out=SimpleNamespace(linear=SimpleNamespace())
            ),
        )

    configs = {
        key: lambda: "config"
        for key in (
            "flux2_klein_4b",
            "flux3_image_klein",
            "qwen_image_21",
            "z_image_turbo",
            "schnell",
            "dev",
        )
    }
    install(
        "mflux.models.common.config.model_config",
        ModelConfig=SimpleNamespace(**configs),
    )
    install("mflux.models." + module_path, **{class_name: build})
    if name == "flux3-klein":
        install(
            "mflux.models.flux3.variants.edit.flux3_klein_edit", Flux3KleinEdit=build
        )
        install("mflux.models.flux3.variants.txt2img.flux3_klein", Flux3Klein=build)
    if variant == "references":
        install("mflux.models.qwen21.variants.txt2img.qwen_image_21", QwenImage21=build)

    class Definition:
        @staticmethod
        def get_components():
            return [
                SimpleNamespace(
                    name="text_encoder",
                    skip_quantization=True,
                    loading_mode="multi_json",
                )
            ]

    definition = Definition
    original = definition.get_components
    install(
        "mflux.models.qwen21.weights.qwen21_weight_definition",
        Qwen21WeightDefinition=definition,
    )
    engine = ImageGenerationEngine(name)
    engine._qwen21_full_q4 = variant == "full_q4"
    engine._qwen21_turbo = variant == "turbo"
    engine._qwen21_references = variant == "references"
    (library / "character.safetensors").touch()
    engine._lora_key = resolve_loras(
        [{"name": "style", "scale": 0.6}, {"name": "character", "scale": -0.2}]
    )
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/snapshot")
    if edit:
        engine._build_edit_model()
    else:
        engine._build_model()
    assert len(calls) == 1
    assert calls[0]["lora_paths"] == [
        str(library / "style.safetensors"),
        str(library / "character.safetensors"),
    ]
    assert calls[0]["lora_scales"] == [0.6, -0.2]
    assert calls[0]["bake_lora"] is False
    assert definition.get_components is original


def test_list_sorts_by_stem_and_missing_endpoint(client, library, monkeypatch):
    (library / "style-more.safetensors").touch()
    assert [item["id"] for item in list_loras()] == ["style", "style-more"]
    monkeypatch.setenv("RAPID_MLX_LORA_DIR", str(library / "missing"))
    assert client.get("/v1/images/loras").json() == {"object": "list", "data": []}
