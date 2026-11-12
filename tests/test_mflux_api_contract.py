"""Bind real adapter calls to the installed mflux API without loading weights.

Source-only signatures let this run on hosts where importing MLX aborts. On
MLX hosts the same calls bind to inspect.signature of the actual methods.
"""

from __future__ import annotations

import ast
import copy
import importlib.metadata
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging.version import Version
from PIL import Image

from rapid_mlx.image.engine import ImageGenerationEngine, _preserve_qwen21_precision

MODELS = [
    (
        "Runpod/FLUX.2-klein-4B-mflux-4bit",
        "flux2.variants.txt2img.flux2_klein",
        "Flux2Klein",
        False,
    ),
    (
        "Runpod/FLUX.2-klein-4B-mflux-4bit",
        "flux2.variants.edit.flux2_klein_edit",
        "Flux2KleinEdit",
        True,
    ),
    (
        "filipstrand/Z-Image-Turbo-mflux-4bit",
        "z_image.variants.z_image",
        "ZImage",
        False,
    ),
    (
        "dhairyashil/FLUX.1-schnell-mflux-4bit",
        "flux.variants.txt2img.flux",
        "Flux1",
        False,
    ),
    (
        "mflux-community/qwen-image-mflux-q6",
        "qwen.variants.txt2img.qwen_image",
        "QwenImage",
        False,
    ),
    (
        "OsaurusAI/Qwen-Image-Edit-mflux-q8",
        "qwen.variants.edit.qwen_image_edit",
        "QwenImageEdit",
        True,
    ),
    (
        "mlx-community/Qwen-Image-2.1-mflux-q4",
        "qwen21.variants.txt2img.qwen_image_21",
        "QwenImage21",
        False,
    ),
    (
        "mlx-community/Qwen-Image-2.1-mflux-q4",
        "qwen21.variants.txt2img.qwen_image_21",
        "QwenImage21",
        True,
    ),
    (
        "Qwen/Qwen-Image-2.1-Turbo",
        "qwen21.variants.txt2img.qwen_image_21",
        "QwenImage21",
        False,
    ),
]


@pytest.fixture
def mflux_source():
    try:
        dist = importlib.metadata.distribution("mflux")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("mflux is not installed")
    if Version(dist.version) < Version("0.23"):
        pytest.skip("requires the upgraded mflux API")
    return Path(dist.locate_file("mflux"))


# Cover both text-to-image and the existing one-image img2img on full precision.
MODELS.extend(
    [
        (
            "Qwen/Qwen-Image-2.1",
            "qwen21.variants.txt2img.qwen_image_21",
            "QwenImage21",
            False,
        ),
        (
            "Qwen/Qwen-Image-2.1",
            "qwen21.variants.txt2img.qwen_image_21",
            "QwenImage21",
            True,
        ),
        (
            "mflux-community/flux2-klein-4b-mflux-bf16",
            "flux2.variants.txt2img.flux2_klein",
            "Flux2Klein",
            False,
        ),
    ]
)


def _signature(root, module, class_name, method):
    # Do not import optional native libraries merely to discover a signature.
    # A host with usable MLX exercises the actual object as well.
    loaded = sys.modules.get(module)
    if loaded is not None and hasattr(loaded, class_name):
        return inspect.signature(getattr(getattr(loaded, class_name), method))
    path = root.parent.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    func = copy.deepcopy(
        next(n for n in cls.body if getattr(n, "name", None) == method)
    )
    func.decorator_list = []
    func.returns = None
    for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs):
        arg.annotation = None
    # Default values can reference native objects; only optionality matters.
    func.args.defaults = [ast.Constant(None) for _ in func.args.defaults]
    func.args.kw_defaults = [
        None if d is None else ast.Constant(None) for d in func.args.kw_defaults
    ]
    func.body = [ast.Pass()]
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return inspect.signature(namespace[method])


def _module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "decode_selector",
    [False, True],
    ids=["without_decode_selector", "with_decode_selector"],
)
@pytest.mark.parametrize("repo,module_name,class_name,editing", MODELS)
def test_image_adapter_binds_installed_mflux_signatures(
    monkeypatch, mflux_source, repo, module_name, class_name, editing, decode_selector
):
    module_name = "mflux.models." + module_name
    constructor = _signature(mflux_source, module_name, class_name, "__init__")
    generate = _signature(mflux_source, module_name, class_name, "generate_image")
    calls = []

    class Model:
        def __init__(self, **kwargs):
            constructor.bind(self, **kwargs)
            self.constructor_kwargs = kwargs
            self.transformer = SimpleNamespace(
                norm_out=SimpleNamespace(linear=SimpleNamespace(bias=object()))
            )
            self.vae = SimpleNamespace()
            if decode_selector:
                self.vae.decode_mode = "low_memory"
                if class_name != "ZImage":
                    self.vae.DECODE_LEGACY = "legacy"

        def generate_image(self, **kwargs):
            generate.bind(self, **kwargs)
            calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (8, 8)))

    _module(monkeypatch, module_name, **{class_name: Model})
    config = SimpleNamespace(
        **{
            name: lambda: object()
            for name in (
                "flux2_klein_4b",
                "z_image_turbo",
                "schnell",
                "dev",
                "qwen_image_21",
            )
        }
    )
    _module(monkeypatch, "mflux.models.common.config.model_config", ModelConfig=config)
    _module(
        monkeypatch,
        "mflux.models.qwen21.weights.qwen21_weight_definition",
        Qwen21WeightDefinition=type(
            "Definition", (), {"get_components": staticmethod(lambda: [])}
        ),
    )
    engine = ImageGenerationEngine(repo)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/local/checkpoint")
    model = engine._build_edit_model() if editing else engine._build_model()
    if engine.family in {"flux2-klein", "z-image"}:
        if decode_selector:
            assert model.vae.decode_mode == "legacy"
        else:
            assert not hasattr(model.vae, "decode_mode")
    if engine.family in {"flux-schnell", "qwen-image", "qwen-image-edit"}:
        assert model.transformer.norm_out.linear.bias is None
    if engine.family == "z-image":
        assert model.constructor_kwargs["float32"] is True
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_: model)
    engine.generate(
        prompt="A red bicycle",
        seed=42,
        num_inference_steps=8,
        width=512,
        height=512,
        image_paths=["input.png"] if editing else None,
    )
    assert calls[0]["seed"] == 42
    assert calls[0]["num_inference_steps"] == 8


def test_qwen21_keeps_composed_attention_and_separate_projections():
    attention = SimpleNamespace(use_fused_prologue=True)
    transformer = SimpleNamespace(
        use_packed_projections=True,
        transformer_blocks=[SimpleNamespace(attn=attention)],
    )
    _preserve_qwen21_precision(SimpleNamespace(transformer=transformer))
    assert transformer.use_packed_projections is False
    assert attention.use_fused_prologue is False


def test_mflux_imports_still_exist(mflux_source):
    root = Path(__file__).parents[1]
    for directory in ("rapid_mlx", "scripts", "apps/rapid-mac/scripts"):
        for path in (root / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "mflux."
                ):
                    target = mflux_source.parent.joinpath(*node.module.split("."))
                    assert (
                        target.with_suffix(".py").is_file()
                        or (target / "__init__.py").is_file()
                    ), (path, node.module)


@pytest.mark.parametrize(
    "decode_selector",
    [False, True],
    ids=["without_decode_selector", "with_decode_selector"],
)
def test_bonsai_initializes_inherited_generation_policy(mflux_source, decode_selector):
    """Bonsai bypasses Flux2Initializer; inherited metadata needs its policy."""
    source = Path(__file__).parents[1] / "rapid_mlx/image/bonsai_runtime/runtime.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BonsaiImage"
    )
    init = copy.deepcopy(
        next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    )
    vae = SimpleNamespace()
    if decode_selector:
        vae.DECODE_LEGACY = "legacy"
    policy = SimpleNamespace(generation_parameters=lambda: {})
    precision_constructor = _signature(
        mflux_source,
        "mflux.models.common.compute_precision.compute_precision",
        "ComputePrecision",
        "__init__",
    )

    def compute_precision(dtype):
        precision_constructor.bind(None, dtype)
        return policy

    namespace = {
        "Path": Path,
        "nn": SimpleNamespace(Module=SimpleNamespace(__init__=lambda _: None)),
        "_validate_checkpoint": lambda _: None,
        "ModelConfig": SimpleNamespace(
            flux2_klein_4b=lambda: SimpleNamespace(text_encoder_overrides={})
        ),
        "ComputePrecision": compute_precision,
        "CallbackRegistry": lambda: object(),
        "_TiledFlux2VAE": type(
            "VAE",
            (),
            {"_bonsai_tiling": object(), "__new__": staticmethod(lambda cls: vae)},
        ),
        "TokenizerLoader": SimpleNamespace(load_all=lambda **_: {}),
        "_tokenizer_definition": lambda: object(),
        "_load_text_encoder": lambda *args: object(),
        "WeightLoader": SimpleNamespace(load=lambda **_: {}),
        "_VAEWeightDefinition": object(),
        "WeightApplier": SimpleNamespace(apply_and_quantize=lambda **_: None),
        "_load_transformer": lambda _: object(),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    model = SimpleNamespace()
    namespace["__init__"](model, "/local/bonsai")
    # Exercise the upstream method that caused the regression, without MLX.
    upstream = mflux_source / "models/flux2/variants/txt2img/flux2_klein.py"
    tree = ast.parse(upstream.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Flux2Klein"
    )
    method = next(
        (n for n in cls.body if getattr(n, "name", None) == "_generation_parameters"),
        None,
    )
    assert model.compute_precision is policy
    if decode_selector:
        assert model.vae.decode_mode == "legacy"
    else:
        assert not hasattr(model.vae, "decode_mode")
    if method is None:
        # 0.23 records ComputePrecision directly in generate_image.
        generation = next(
            n for n in cls.body if getattr(n, "name", None) == "generate_image"
        )
        expression = next(
            n.value
            for n in ast.walk(generation)
            if isinstance(n, ast.keyword) and n.arg == "generation_parameters"
        )
        assert (
            eval(
                compile(ast.Expression(expression), str(upstream), "eval"),
                {"self": model},
            )
            == {}
        )
        return
    namespace["VAEDecodeOptions"] = SimpleNamespace(generation_parameters=lambda _: {})
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(upstream), "exec"),
        namespace,
    )
    model.small_text_encoder = False
    model.small_decoder = False
    assert namespace["_generation_parameters"](model) == {}


@pytest.mark.requires_mlx
@pytest.mark.parametrize("repo,module_name,class_name,editing", MODELS)
def test_image_kwargs_bind_native_mflux_methods(repo, module_name, class_name, editing):
    """On a usable MLX host, inspect the installed classes themselves too."""
    import importlib

    cls = getattr(importlib.import_module("mflux.models." + module_name), class_name)
    engine = ImageGenerationEngine(repo)
    inspect.signature(cls).bind(
        quantize=engine._quantize, model_path="/local/checkpoint"
    )
    kwargs = engine._gen_kwargs(42, "A red bicycle", 8, None, None)
    kwargs.update(width=512, height=512)
    if editing:
        if engine.family == "qwen-image-2.1":
            kwargs.update(image_path="input.png", image_strength=0.4)
        else:
            kwargs["image_paths"] = ["input.png"]
    inspect.signature(cls.generate_image).bind(None, **kwargs)


@pytest.mark.parametrize("fail", [False, True])
def test_mflux_reference_loader_preserves_alpha_behavior_and_restores(
    monkeypatch, fail
):
    from rapid_mlx.image.engine import _generate_backend_image

    class ImageUtil:
        @staticmethod
        def load_image(image, composite_alpha=True):
            return (
                image.convert("RGB")
                if not composite_alpha
                else Image.new("RGB", image.size, "white")
            )

    original = ImageUtil.load_image
    _module(monkeypatch, "mflux.utils.image_util", ImageUtil=ImageUtil)
    source = Image.new("RGBA", (2, 2), (255, 0, 0, 0))

    class Model:
        def generate_image(self, **kwargs):
            assert ImageUtil.load_image(source).getpixel((0, 0)) == (255, 0, 0)
            if fail:
                raise RuntimeError("cancelled")
            return "rendered"

    Model.__module__ = "mflux.models.fake"
    if fail:
        with pytest.raises(RuntimeError, match="cancelled"):
            _generate_backend_image(Model(), image_paths=["input.png"])
    else:
        assert _generate_backend_image(Model(), image_paths=["input.png"]) == "rendered"
    assert ImageUtil.load_image is original


def test_qwen21_precision_ignores_unavailable_optimizations():
    attention = SimpleNamespace()
    transformer = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=attention)])
    _preserve_qwen21_precision(SimpleNamespace(transformer=transformer))
    assert not hasattr(transformer, "use_packed_projections")
    assert not hasattr(attention, "use_fused_prologue")
    _preserve_qwen21_precision(SimpleNamespace())


def test_reference_loader_without_composite_option_is_left_alone(monkeypatch):
    from rapid_mlx.image.engine import _generate_backend_image

    class ImageUtil:
        @staticmethod
        def load_image(image):
            return image.convert("RGB")

    original = ImageUtil.load_image
    _module(monkeypatch, "mflux.utils.image_util", ImageUtil=ImageUtil)

    class Model:
        def generate_image(self, **kwargs):
            assert ImageUtil.load_image is original
            return "rendered"

    Model.__module__ = "mflux.models.fake"
    assert _generate_backend_image(Model(), image_path="input.png") == "rendered"


@pytest.mark.parametrize(
    "alias,editing",
    [
        ("flux3-klein", False),
        ("flux3-klein", True),
        ("flux3-klein-base", False),
        ("flux3-klein-base", True),
        ("Qwen/Qwen-Image-2.1", True),
    ],
)
def test_new_reference_backends_bind_installed_signatures(
    monkeypatch, mflux_source, alias, editing
):
    qwen = "Qwen" in alias
    if qwen:
        module_name = "mflux.models.qwen21.variants.edit.qwen_image_21_edit"
        constructor = _signature(
            mflux_source, module_name, "QwenImage21Edit", "__init__"
        )
        generate = _signature(
            mflux_source, module_name, "QwenImage21Edit", "generate_image"
        )
    else:
        constructor = _signature(
            mflux_source,
            "mflux.models.flux3.variants.flux3_klein_core",
            "_Flux3KleinCore",
            "__init__",
        )
        module_name = "mflux.models.flux3.variants." + (
            "edit.flux3_klein_edit" if editing else "txt2img.flux3_klein"
        )
        generate = _signature(
            mflux_source,
            module_name,
            "Flux3KleinEdit" if editing else "Flux3Klein",
            "generate_image",
        )
    calls = []

    class Model:
        def __init__(self, **kwargs):
            constructor.bind(self, **kwargs)
            self.vae = SimpleNamespace()

        def generate_image(self, **kwargs):
            generate.bind(self, **kwargs)
            calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (8, 8)))

    _module(
        monkeypatch,
        "mflux.models.common.config.model_config",
        ModelConfig=SimpleNamespace(
            flux3_image_klein=lambda: object(),
            flux3_image_klein_base=lambda: object(),
            qwen_image_21=lambda: object(),
        ),
    )
    for module, class_name in (
        ("mflux.models.flux3.variants.txt2img.flux3_klein", "Flux3Klein"),
        ("mflux.models.flux3.variants.edit.flux3_klein_edit", "Flux3KleinEdit"),
        ("mflux.models.qwen21.variants.txt2img.qwen_image_21", "QwenImage21"),
        ("mflux.models.qwen21.variants.edit.qwen_image_21_edit", "QwenImage21Edit"),
    ):
        _module(monkeypatch, module, **{class_name: Model})
    engine = ImageGenerationEngine(alias)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/local/checkpoint")
    engine._qwen21_references = qwen
    model = engine._build_edit_model() if editing else engine._build_model()
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_: model)
    engine.generate(
        prompt="combine",
        width=512,
        height=512,
        num_inference_steps=engine.default_steps,
        image_paths=["a.png", "b.png", "c.png"] if editing else None,
    )
    assert len(calls) == 1
    if editing:
        assert calls[0]["image_paths"] == ["a.png", "b.png", "c.png"]
