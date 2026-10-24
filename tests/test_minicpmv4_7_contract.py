# SPDX-License-Identifier: Apache-2.0
"""Offline routing, template and fail-closed provenance contracts for 4.7."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jinja2.sandbox import SandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "rapid_mlx/models/mlx_vlm_vendored/models/minicpmv4_7"
FIXTURES = ROOT / "tests/fixtures/minicpmv4_7"
COMMIT = "b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0"


def test_vendored_provenance():
    inventory = json.loads((VENDOR / "provenance.json").read_text())
    assert inventory["upstream_commit"] == COMMIT
    assert set(inventory["files"]) == {p.name for p in VENDOR.glob("*.py")}
    assert "Permission is hereby granted" in (VENDOR / "LICENSE").read_text()
    for name, entry in inventory["files"].items():
        source = (VENDOR / name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == entry["vendored_sha256"]
        assert COMMIT in source
        body = "\n".join(source.split("\n")[3:])
        if name == "_vision_compat.py":
            region = body[body.index("    def get_vision_embedding(") :]
            assert hashlib.sha256(region.encode()).hexdigest() == entry["region_sha256"]
            assert "VENDOR-DEVIATION(subset)" in body
            continue
        if name == "_vision.py":
            region = body[body.index("class VisionModel(nn.Module):") :]
            assert hashlib.sha256(region.encode()).hexdigest() == entry["region_sha256"]
            assert "VENDOR-DEVIATION(subset)" in body
            continue
        if body.startswith("# VENDOR-DEVIATION"):
            assert "import-redirect" in body
            while body.startswith("#"):
                body = body.split("\n", 1)[1]
        if name == "processing_minicpmv4_7.py":
            # Reverse only the request-local copy hunk to reconstruct upstream.
            replacement = (
                "        # VENDOR-DEVIATION(thread-safety): isolate differing request modes without\n"
                "        # mutating shared processors; matching configured modes need no copies.\n"
                "        from copy import copy\n"
                "\n"
                "        request_processor = self\n"
                "        if any(\n"
                "            component.downsample_mode != mode\n"
                "            for component in (self.image_processor, self.video_processor)\n"
                "        ):\n"
                "            request_processor = copy(self)\n"
                "            request_processor.image_processor = copy(self.image_processor)\n"
                "            request_processor.video_processor = copy(self.video_processor)\n"
                "            request_processor.image_processor.downsample_mode = mode\n"
                "            request_processor.video_processor.downsample_mode = mode\n"
                "        result = super(MiniCPMV4_7Processor, request_processor).__call__(\n"
                "            text=texts, images=images, videos=videos, **kwargs\n"
                "        )\n"
            )
            upstream = (
                "        old_modes = (\n"
                "            self.image_processor.downsample_mode,\n"
                "            self.video_processor.downsample_mode,\n"
                "        )\n"
                "        self.image_processor.downsample_mode = self.video_processor.downsample_mode = (\n"
                "            mode\n"
                "        )\n"
                "        try:\n"
                "            result = super().__call__(\n"
                "                text=texts, images=images, videos=videos, **kwargs\n"
                "            )\n"
                "        finally:\n"
                "            (\n"
                "                self.image_processor.downsample_mode,\n"
                "                self.video_processor.downsample_mode,\n"
                "            ) = old_modes\n"
            )
            assert body.count(replacement) == 1
            body = body.replace(replacement, upstream)
        for new, old in (
            ("from mlx_vlm.models.base import", "from ..base import"),
            ("from ._config import", "from ..minicpmv4_6.config import"),
            (
                "from ._vision import VisionModel",
                "from ..minicpmv4_6.vision import VisionModel",
            ),
            (
                "from ._vision_compat import Model as MiniCPMVModel",
                "from ..minicpmv4_6.minicpmv4_6 import Model as MiniCPMVModel",
            ),
            ("from mlx_vlm.models.minicpmv4_6.", "from ..minicpmv4_6."),
            ("from mlx_vlm.models.qwen3_5_moe.", "from ..qwen3_5_moe."),
        ):
            body = body.replace(new, old)
        assert hashlib.sha256(body.encode()).hexdigest() == entry["upstream_sha256"], (
            name
        )


@pytest.mark.parametrize("size,moe", [("1B", False), ("35B-A3B", True)])
def test_routing_and_profiles(size, moe, tmp_path):
    from rapid_mlx.api.utils import is_mllm_model
    from rapid_mlx.byom import preflight
    from rapid_mlx.byom._vlm_model_types import MLX_VLM_MODEL_TYPES
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.model_auto_config import detect_model_config

    quant = "mxfp4" if moe else "8bit"
    alias = f"minicpm-v4.7-{size.lower()}-{quant}"
    repo = f"mlx-community/MiniCPM-V-4.7-{size}-{quant}"
    assert resolve_profile(alias).hf_path == repo
    assert resolve_profile(alias).supports_image_input
    assert "minicpmv4_7" in MLX_VLM_MODEL_TYPES
    assert "minicpmv4_7" in preflight.RAPID_MODEL_TYPES
    # Arbitrary local names exercise config-driven detection independently.
    (tmp_path / "config.json").write_bytes(
        (FIXTURES / f"{size}-config.json").read_bytes()
    )
    for name in (alias, repo, str(tmp_path)):
        assert is_mllm_model(name)
        config = detect_model_config(name)
        assert config.tool_call_parser == "qwen3_coder_xml"
        assert config.reasoning_parser == "qwen3"
        assert config.is_moe is moe
        assert config.is_hybrid is moe
        assert config.is_hybrid_explicit
        assert not config.supports_spec_decode


def test_release_probe_recognizes_minicpm_vision():
    from scripts.release_check_m3_random import _is_multimodal

    assert _is_multimodal("minicpm-v4.7-1b-8bit")
    assert _is_multimodal("custom-name", "mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4")
    assert not _is_multimodal("minicpm3-4b")


@pytest.fixture
def isolated_loader_modules(monkeypatch):
    """Replace optional architecture dependencies, never importing MLX."""
    import sys
    from types import ModuleType

    modules = {}
    for name in (
        "mlx_vlm",
        "mlx_vlm.utils",
        "mlx_vlm.models",
        "mlx_vlm.prompt_utils",
        "rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7",
        "rapid_mlx.patches.glm5_next_runtime",
        "rapid_mlx.patches.glm5_next_processor",
        "rapid_mlx.patches.glm5_next_forget_gate_quant",
    ):
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    for module, installer in (
        ("glm5_next_runtime", "install_glm5_next_runtime_fix"),
        ("glm5_next_processor", "install_glm5_next_processor_patch"),
        ("glm5_next_forget_gate_quant", "install_glm5_next_forget_gate_quant_fix"),
    ):
        setattr(modules[f"rapid_mlx.patches.{module}"], installer, lambda: None)
    modules["mlx_vlm.prompt_utils"].MODEL_CONFIG = {}
    from enum import Enum

    class MessageFormat(Enum):
        LIST_WITH_IMAGE_FIRST = "image-first"

    modules["mlx_vlm.prompt_utils"].MessageFormat = MessageFormat
    # Restore an existing binding, if any, after checking the real installer.
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.minicpmv4_7", None)
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.minicpmv4_7")
    return modules


def test_loader_binding_is_idempotent_without_mlx(isolated_loader_modules):
    import sys

    from rapid_mlx.models.minicpmv4_7 import install_minicpmv4_7

    modules = isolated_loader_modules
    package = modules["rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7"]
    for _ in range(2):
        install_minicpmv4_7()
        assert sys.modules["mlx_vlm.models.minicpmv4_7"] is package
        assert modules["mlx_vlm.models"].minicpmv4_7 is package
        prompt = modules["mlx_vlm.prompt_utils"]
        assert prompt.MODEL_CONFIG["minicpmv4_7"] is (
            prompt.MessageFormat.LIST_WITH_IMAGE_FIRST
        )


@pytest.mark.parametrize("kind", ["minicpmv4_7", "other", "none", "object"])
@pytest.mark.parametrize("trust_remote_code", [True, False])
def test_mllm_installs_binding_before_load(
    isolated_loader_modules, monkeypatch, tmp_path, kind, trust_remote_code
):
    import sys
    from types import SimpleNamespace

    from rapid_mlx.models import mllm
    from rapid_mlx.utils import tokenizer

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
    monkeypatch.setattr(
        tokenizer, "augment_eos_token_ids_from_generation_config", lambda *a: None
    )
    monkeypatch.setattr(tokenizer, "repair_byte_level_decoder", lambda *a: None)
    modules = isolated_loader_modules
    config = (
        None
        if kind == "none"
        else SimpleNamespace()
        if kind == "object"
        else {"model_type": kind}
    )
    calls = []

    def load_config(name, **kwargs):
        calls.append(("config", name, kwargs))
        return config

    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=object())

    def load(name, **kwargs):
        calls.append(("model", name, kwargs))
        if kind == "minicpmv4_7":
            assert (
                sys.modules["mlx_vlm.models.minicpmv4_7"]
                is modules["rapid_mlx.models.mlx_vlm_vendored.models.minicpmv4_7"]
            )
            assert "minicpmv4_7" in modules["mlx_vlm.prompt_utils"].MODEL_CONFIG
        else:
            assert "mlx_vlm.models.minicpmv4_7" not in sys.modules
            assert not modules["mlx_vlm.prompt_utils"].MODEL_CONFIG
        return model, processor

    modules["mlx_vlm.utils"].get_model_path = lambda name, **kwargs: tmp_path
    modules["mlx_vlm.utils"].load_config = load_config
    modules["mlx_vlm"].load = load
    wrapper = mllm.MLXMultimodalLM(
        "local-checkpoint", trust_remote_code=trust_remote_code, enable_cache=False
    )
    if kind in {"none", "object"}:
        with pytest.raises(ValueError, match="must contain an object"):
            wrapper.load()
        assert [call[0] for call in calls] == ["config"]
        return
    wrapper.load()
    wrapper.load()
    options = {"trust_remote_code": trust_remote_code}
    assert calls == [
        ("config", str(tmp_path), options),
        ("model", str(tmp_path), options),
    ]
    assert wrapper._loaded
    assert wrapper.model is model
    assert wrapper.processor is processor
    assert wrapper.config is config
    assert "4.7 dense and MoE" in wrapper.list_supported_model_families()["MiniCPM-V"]


@pytest.mark.parametrize("size", ["1B", "35B-A3B"])
def test_checkpoint_template_and_existing_parser(size):
    from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    template = SandboxedEnvironment().from_string(
        (FIXTURES / f"{size}-chat_template.jinja").read_text()
    )
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "weather", "arguments": {"city": "Paris"}}}
            ],
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True)
    call = prompt[prompt.index("<tool_call>", prompt.index("<|im_start|>assistant")) :]
    call = call[: call.index("</tool_call>") + len("</tool_call>")]
    result = Qwen3CoderToolParser().extract_tool_calls(call, request={"tools": tools})
    assert result.tools_called
    assert result.tool_calls[0]["name"] == "weather"
    assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}
    # Raw Jinja leaves the flag undefined; mlx_vlm supplies False when omitted.
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=False,
    ).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_canvas_rope_text_and_image():
    # rope.py has only NumPy dependencies; load it without the MLX package init.
    from types import SimpleNamespace

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "minicpm_canvas_test", VENDOR / "rope.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(**json.loads((FIXTURES / "1B-config.json").read_text()))
    rope = module.CanvasRoPE(config)
    tokens = np.array(
        [[1, config.image_start_id, config.image_token_id, config.image_end_id, 2]]
    )
    positions, delta = rope.get_rope_index(
        tokens,
        mm_token_type_ids=(tokens == config.image_token_id).astype(np.int32),
        target_sizes=np.array([[4, 4]]),
    )
    assert positions.shape == (3, 1, 5)
    assert delta.shape == (1, 1)
    assert positions[:, 0, -1].tolist() == [3, 3, 3]


@pytest.fixture
def isolated_processor_class(monkeypatch):
    """Exercise the vendored processor without importing the MLX package init."""
    import sys
    from types import ModuleType, SimpleNamespace

    class ParentProcessor:
        def __init__(self):
            self.image_processor = SimpleNamespace(downsample_mode="16x")
            self.video_processor = SimpleNamespace(downsample_mode="4x")
            self.image_token_id = 1
            self.video_token_id = 2

        def _normalize_text(self, text):
            return [text]

        def _normalize_images(self, inputs, *args):
            return [[]]

        _normalize_videos = _normalize_images

        def _count_image_markers(self, text):
            return 0

        _count_video_markers = _count_image_markers

    base = ModuleType("mlx_vlm.models.base")
    base.install_auto_processor_patch = lambda *args: None
    base.load_chat_template = lambda *args: None
    parent = ModuleType("mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6")
    parent.MiniCPMVProcessor = ParentProcessor
    for name in (
        "MiniCPMVImageProcessor",
        "MiniCPMVVideoProcessor",
        "_extract_hf_from_pretrained_kwargs",
        "_load_local_qwen2_tokenizer",
    ):
        setattr(parent, name, None)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    spec = importlib.util.spec_from_file_location(
        "minicpm_processor_thread_test", VENDOR / "processing_minicpmv4_7.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniCPMV4_7Processor, ParentProcessor


@pytest.mark.parametrize("second_mode", ["16x", None])
def test_processor_concurrent_downsample_modes(isolated_processor_class, second_mode):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import numpy as np

    processor_class, parent = isolated_processor_class
    processor = processor_class()
    barrier = Barrier(2, timeout=5)
    shared = (processor.image_processor, processor.video_processor)

    def parent_call(self, **kwargs):
        before = (
            self.image_processor.downsample_mode,
            self.video_processor.downsample_mode,
        )
        barrier.wait()
        assert (shared[0].downsample_mode, shared[1].downsample_mode) == ("16x", "4x")
        assert before == (
            self.image_processor.downsample_mode,
            self.video_processor.downsample_mode,
        )
        return {"input_ids": np.array([[0, 1, 2]]), "observed_modes": before}

    parent.__call__ = parent_call
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(processor, text="first", downsample_mode="4x")
        kwargs = {} if second_mode is None else {"downsample_mode": second_mode}
        second = pool.submit(processor, text="second", **kwargs)
        first_result, second_result = (
            first.result(timeout=10),
            second.result(timeout=10),
        )
    assert first_result["observed_modes"] == ("4x", "4x")
    assert second_result["observed_modes"] == ("16x", "16x")
    assert first_result["downsample_mode"] == "4x"
    assert second_result["downsample_mode"] == "16x"
    assert processor.image_processor is shared[0]
    assert processor.video_processor is shared[1]
    assert (shared[0].downsample_mode, shared[1].downsample_mode) == ("16x", "4x")
    assert first_result["mm_token_type_ids"].tolist() == [[0, 1, 2]]


@pytest.mark.parametrize("fail", [False, True])
def test_processor_never_mutates_shared_modes(isolated_processor_class, fail):
    from types import SimpleNamespace

    import numpy as np

    processor_class, parent = isolated_processor_class
    processor = processor_class()

    class ReadOnlyMode(SimpleNamespace):
        def __setattr__(self, name, value):
            raise AssertionError("shared processor mutated")

        def __copy__(self):
            return SimpleNamespace(**vars(self))

    processor.image_processor = ReadOnlyMode(downsample_mode="16x")
    processor.video_processor = ReadOnlyMode(downsample_mode="16x")
    shared = (processor.image_processor, processor.video_processor)

    def parent_call(self, **kwargs):
        assert (shared[0].downsample_mode, shared[1].downsample_mode) == ("16x", "16x")
        assert self.image_processor is not shared[0]
        assert self.video_processor is not shared[1]
        assert (
            self.image_processor.downsample_mode
            == self.video_processor.downsample_mode
            == "4x"
        )
        if fail:
            raise RuntimeError("preprocessing failed")
        return {"input_ids": np.array([[0]])}

    parent.__call__ = parent_call
    if fail:
        with pytest.raises(RuntimeError, match="preprocessing failed"):
            processor(text="hello", downsample_mode="4x")
    else:
        assert processor(text="hello", downsample_mode="4x")["downsample_mode"] == "4x"
    assert processor.image_processor is shared[0]
    assert processor.video_processor is shared[1]
    assert (shared[0].downsample_mode, shared[1].downsample_mode) == ("16x", "16x")


@pytest.mark.parametrize("mode", ["4x", "16x"])
def test_processor_matching_modes_call_through(isolated_processor_class, mode):
    import numpy as np

    processor_class, parent = isolated_processor_class
    processor = processor_class()
    processor.image_processor.downsample_mode = mode
    processor.video_processor.downsample_mode = mode

    def parent_call(self, **kwargs):
        assert self is processor
        return {"input_ids": np.array([[0]])}

    parent.__call__ = parent_call
    assert processor(text="hello")["downsample_mode"] == mode
