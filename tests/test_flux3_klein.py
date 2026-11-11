"""FLUX 3 defaults, runtime selection and reference limits without weights."""

import sys
import types

import pytest

from rapid_mlx.image import engine as image_engine
from rapid_mlx.image.engine import ImageGenerationEngine, ImageRequestError
from rapid_mlx.model_aliases import resolve_model


@pytest.mark.parametrize(
    "alias,steps,guidance", [("flux3-klein", 16, None), ("flux3-klein-base", 50, 3.5)]
)
def test_alias_defaults(alias, steps, guidance):
    repo = resolve_model(alias)
    assert repo == "black-forest-labs/" + alias.replace("flux3", "flux-3-image")
    for name in (alias, repo):
        engine = ImageGenerationEngine(name)
        assert engine.family == "flux3-klein"
        assert engine.default_steps == engine.default_edit_steps == steps
        assert engine.default_edit_guidance == guidance
        assert engine._quantize == 4
        assert engine.max_reference_images == 10
        assert engine.supports_generation and engine.supports_editing
        assert not engine.supports_negative_prompt
        kwargs = engine._gen_kwargs(0, "cat", steps, None, None)
        assert kwargs.get("guidance") == guidance
        assert ("guidance" in kwargs) == (guidance is not None)


@pytest.mark.parametrize("dimensions", [(513, 512), (512, 513), (0, 512), (-16, 512)])
def test_dimensions_validate_before_loading(dimensions):
    engine = ImageGenerationEngine("flux3-klein")
    with pytest.raises(ImageRequestError, match="multiples of 16"):
        engine.generate(prompt="cat", width=dimensions[0], height=dimensions[1])
    assert engine._model is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"guidance": 1.0}, "guidance"),
        ({"negative_prompt": "bad"}, "negative_prompt"),
        ({"image_paths": ["x"] * 11}, "10 reference"),
    ],
)
def test_distilled_rejects_unsupported_options(kwargs, match):
    with pytest.raises(ImageRequestError, match=match):
        ImageGenerationEngine("flux3-klein").validate_request(prompt="cat", **kwargs)


@pytest.mark.parametrize("alias", ["flux2-klein-4b", "qwen-image-edit"])
def test_existing_reference_limits(alias):
    engine = ImageGenerationEngine(resolve_model(alias))
    assert engine.max_reference_images == 1
    with pytest.raises(ImageRequestError, match="1 reference"):
        engine.validate_request(prompt="cat", image_paths=["a", "b"])


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("base", [False, True])
def test_flux3_runtime_construction(monkeypatch, editing, base):
    calls = []

    def fake_model(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(vae=types.SimpleNamespace())

    modules = {
        "mflux.models.common.config.model_config": types.SimpleNamespace(
            ModelConfig=types.SimpleNamespace(
                flux3_image_klein=lambda: "distilled",
                flux3_image_klein_base=lambda: "base",
            )
        ),
        "mflux.models.flux3.variants.txt2img.flux3_klein": types.SimpleNamespace(
            Flux3Klein=fake_model
        ),
        "mflux.models.flux3.variants.edit.flux3_klein_edit": types.SimpleNamespace(
            Flux3KleinEdit=fake_model
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    engine = ImageGenerationEngine("flux3-klein" + ("-base" if base else ""))
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: None)
    monkeypatch.setattr(image_engine, "_preserve_vae_decode", lambda _: None)
    (engine._build_edit_model if editing else engine._build_model)()
    assert calls == [
        {
            "quantize": 4,
            "model_path": None,
            "model_config": "base" if base else "distilled",
        }
    ]


def test_old_runtime_error(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mflux.models.flux3.variants.txt2img.flux3_klein", None
    )
    with pytest.raises(image_engine.ImageRuntimeError, match=r"mflux 0\.23"):
        ImageGenerationEngine("flux3-klein")._build_model()


def test_qwen_reference_limit_from_installed_constant(monkeypatch, tmp_path):
    from importlib import metadata

    source = tmp_path / "edit.py"
    source.write_text("MAX_REFERENCES = 7\n")
    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda _: types.SimpleNamespace(locate_file=lambda _: source),
    )
    engine = ImageGenerationEngine("Qwen/Qwen-Image-2.1")
    assert engine.max_reference_images == 7
    engine.validate_request(prompt="edit", image_paths=["a"] * 7)
    with pytest.raises(ImageRequestError, match="7 reference"):
        engine.validate_request(prompt="edit", image_paths=["a"] * 8)
    source.unlink()
    assert image_engine._qwen21_max_references() == 1


def test_catalog_contract():
    from rapid_mlx.cli import _available_models_json_payload
    from rapid_mlx.routes.models import _build_model_info

    rows = {row["alias"]: row for row in _available_models_json_payload()["image"]}
    for alias in ("flux3-klein", "flux3-klein-base"):
        assert rows[alias]["max_reference_images"] == 10
        assert rows[alias]["license"] == "FLUX Kommunity License v1.0"
        assert rows[alias]["requires_hf_login"] is True
        assert "US$5M" in rows[alias]["license_notice"]
        assert _build_model_info(alias).max_reference_images == 10


def test_gated_download_explains_terms_and_login():
    import httpx
    from huggingface_hub.errors import GatedRepoError

    from rapid_mlx.cli import render_hub_error

    text = render_hub_error(
        GatedRepoError(
            "access denied",
            response=httpx.Response(
                403,
                request=httpx.Request(
                    "GET", "https://huggingface.co/black-forest-labs/flux-3-image-klein"
                ),
            ),
        ),
        resolve_model("flux3-klein"),
    )
    assert "FLUX Kommunity License v1.0" in text
    assert "US$5M" in text
    assert "hf auth login" in text
    assert "https://huggingface.co/black-forest-labs/flux-3-image-klein" in text


def test_multi_qwen_dispatch_and_mode_switch(monkeypatch):
    from PIL import Image

    engine = ImageGenerationEngine("Qwen/Qwen-Image-2.1")
    calls = []
    model = types.SimpleNamespace(
        generate_image=lambda **kwargs: (
            calls.append(kwargs)
            or types.SimpleNamespace(image=Image.new("RGB", (8, 8)))
        )
    )
    monkeypatch.setattr(engine, "_ensure_runtime_assets", lambda: None)
    monkeypatch.setattr(engine, "_verify_weights_complete", lambda: None)
    monkeypatch.setattr(image_engine, "_release_allocator_cache", lambda: None)
    monkeypatch.setattr(engine, "_build_edit_model", lambda: model)
    generation = types.SimpleNamespace()
    monkeypatch.setattr(engine, "_build_model", lambda: generation)
    monkeypatch.setattr("rapid_mlx.model_downloads.require_local", lambda _: None)
    monkeypatch.setattr(engine, "max_reference_images", 10)
    engine.generate(
        prompt="combine",
        image_paths=["a", "b", "c"],
        width=None,
        height=None,
        num_inference_steps=40,
    )
    assert engine._loaded_mode == "references"
    assert calls[0]["image_paths"] == ["a", "b", "c"]
    assert "image_strength" not in calls[0]
    assert "image_path" not in calls[0]
    assert engine._ensure_loaded(for_edit=True, reference_count=3) is model
    assert engine._ensure_loaded(for_edit=False) is generation
    assert engine._loaded_mode == "generation"


def test_flux3_checkpoint_layout(monkeypatch, tmp_path):
    import json

    from rapid_mlx import _download_gate as gate

    repo = resolve_model("flux3-klein")
    snapshot = tmp_path / "snapshot"
    for directory in ("text_encoder", "transformer", "vae"):
        (snapshot / directory).mkdir(parents=True, exist_ok=True)
    for relative in (
        "text_encoder/tokenizer.json",
        "text_encoder/model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        (snapshot / relative).write_bytes(b"data")
    for directory, index, shard in (
        ("text_encoder", "model.safetensors.index.json", "model.safetensors"),
        (
            "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors",
        ),
    ):
        (snapshot / directory / index).write_text(
            json.dumps({"weight_map": {"weight": shard}})
        )
    monkeypatch.setattr(
        gate, "_mflux_snapshot_dir", lambda _: (str(tmp_path), str(snapshot))
    )
    assert gate.mflux_missing_weights(repo) == []
    (snapshot / "transformer/diffusion_pytorch_model.safetensors").unlink()
    assert gate.mflux_missing_weights(repo) == [
        "transformer/diffusion_pytorch_model.safetensors"
    ]
