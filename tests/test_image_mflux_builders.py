"""Exercise mflux constructor contracts without importing mflux or MLX."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from rapid_mlx.image.engine import ImageGenerationEngine, _preserve_vae_decode


@pytest.mark.parametrize("decode_api", ["constant", "fallback", "absent"])
def test_image_preserve_vae_decode(decode_api):
    vae = SimpleNamespace()
    if decode_api != "absent":
        vae.decode_mode = "tiled"
    if decode_api == "constant":
        vae.DECODE_LEGACY = object()
    before = vars(vae).copy()

    _preserve_vae_decode(vae)

    if decode_api == "constant":
        assert vae.decode_mode is vae.DECODE_LEGACY
    elif decode_api == "fallback":
        assert vae.decode_mode == "legacy"
    else:
        assert vars(vae) == before


@pytest.mark.parametrize(
    "family,module_name,class_name,config_name,edit",
    [
        (
            "flux2-klein",
            "mflux.models.flux2.variants.txt2img.flux2_klein",
            "Flux2Klein",
            "flux2_klein_4b",
            False,
        ),
        (
            "z-image",
            "mflux.models.z_image.variants.z_image",
            "ZImage",
            "z_image_turbo",
            False,
        ),
        (
            "qwen-image-edit",
            "mflux.models.qwen.variants.edit.qwen_image_edit",
            "QwenImageEdit",
            None,
            False,
        ),
        (
            "qwen-image",
            "mflux.models.qwen.variants.txt2img.qwen_image",
            "QwenImage",
            None,
            False,
        ),
        (
            "flux-schnell",
            "mflux.models.flux.variants.txt2img.flux",
            "Flux1",
            "schnell",
            False,
        ),
        (
            "flux-dev",
            "mflux.models.flux.variants.txt2img.flux",
            "Flux1",
            "dev",
            False,
        ),
        (
            "flux2-klein",
            "mflux.models.flux2.variants.edit.flux2_klein_edit",
            "Flux2KleinEdit",
            "flux2_klein_4b",
            True,
        ),
    ],
)
@pytest.mark.parametrize("quantize,model_path", [(4, None), (None, "/local/snapshot")])
def test_image_mflux_builder_kwargs_and_compatibility(
    monkeypatch,
    family,
    module_name,
    class_name,
    config_name,
    edit,
    quantize,
    model_path,
):
    calls = []
    config = object()
    legacy = object()
    original_bias = object()

    class FakeModel:
        def __init__(self, **kwargs):
            calls.append((self, kwargs))
            self.vae = SimpleNamespace(decode_mode="tiled", DECODE_LEGACY=legacy)
            self.transformer = SimpleNamespace(
                norm_out=SimpleNamespace(linear=SimpleNamespace(bias=original_bias))
            )

    # Install complete fake package chains so these imports also work on Linux
    # where neither mflux nor its optional native dependencies are installed.
    def stub_module(name):
        for index in range(1, len(name.split(".")) + 1):
            prefix = ".".join(name.split(".")[:index])
            if prefix not in fake_modules:
                module = ModuleType(prefix)
                module.__path__ = []
                fake_modules[prefix] = module
                monkeypatch.setitem(sys.modules, prefix, module)
        return fake_modules[name]

    fake_modules = {}
    model_config = SimpleNamespace()
    if config_name is not None:
        setattr(model_config, config_name, lambda: config)
    stub_module("mflux.models.common.config.model_config").ModelConfig = model_config
    setattr(stub_module(module_name), class_name, FakeModel)

    engine = ImageGenerationEngine("flux-schnell", quantize=quantize)
    engine.family = family
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: model_path)

    model = engine._build_edit_model() if edit else engine._build_model()

    expected = {"quantize": quantize, "model_path": model_path}
    if config_name is not None:
        expected["model_config"] = config
    if family == "z-image":
        expected["float32"] = True
    assert calls == [(model, expected)]
    if family in {"flux2-klein", "z-image"}:
        assert model.vae.decode_mode is legacy
        assert model.transformer.norm_out.linear.bias is original_bias
    else:
        assert model.transformer.norm_out.linear.bias is None
