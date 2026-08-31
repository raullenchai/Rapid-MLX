# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the GLM-5 Next processor compatibility layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from PIL import Image

from vllm_mlx.patches import glm5_next_processor as processor_patch
from vllm_mlx.patches.glm5_next_processor import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize,
)


class _TokenizerStub:
    model_input_names = ["input_ids", "attention_mask"]

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return {"<|image|>": 120, "<|video|>": 121}[token]

    @staticmethod
    def __call__(texts, **kwargs):
        del kwargs
        rows = [[1] + [120] * text.count("<|image|>") + [2] for text in texts]
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }


def test_token_budget_keeps_native_448_tile() -> None:
    assert smart_resize(2, 448, 448) == (448, 448)
    processor = Glm5NextImageProcessor()
    assert processor.get_number_of_image_patches(448, 448) == 1024


@pytest.mark.parametrize(
    ("num_frames", "height", "width"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_smart_resize_rejects_empty_media(
    num_frames: int, height: int, width: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        smart_resize(num_frames, height, width)


def test_processor_expands_exact_image_placeholder_count() -> None:
    image_processor = Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=1,
        max_image_tokens=4,
    )
    processor = Glm5NextProcessor(
        image_processor=image_processor,
        tokenizer=_TokenizerStub(),
    )

    inputs = processor(
        images=[Image.new("RGB", (8, 4), "blue")],
        text=["<|begin_of_image|><|image|><|end_of_image|>"],
    )

    image_tokens = int(mx.sum(inputs["input_ids"] == 120).item())
    expected_tokens = int(inputs["image_grid_thw"][0].prod().item()) // 4
    assert image_tokens == expected_tokens == 2
    assert inputs["pixel_values"].shape == (8, 24)
    assert inputs["mm_token_type_ids"].shape == inputs["input_ids"].shape


def test_processor_rejects_image_placeholder_count_mismatch() -> None:
    processor = Glm5NextProcessor(tokenizer=_TokenizerStub())
    with pytest.raises(ValueError, match="More images were provided"):
        processor(
            images=[Image.new("RGB", (28, 28), "blue")],
            text=["no image marker"],
        )


def test_local_optional_metadata_miss_never_falls_through_to_hub(
    monkeypatch, tmp_path: Path
) -> None:
    import huggingface_hub

    def fail(*_args, **_kwargs):
        raise AssertionError("local checkpoint must not make a Hub request")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    assert processor_patch._load_json(tmp_path, "processor_config.json") is None


def test_install_registers_auto_processor_and_prompt_shape_in_clean_process(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"glm5_next"}')
    script = """
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
from vllm_mlx.patches import glm5_next_processor as patch
patch.Glm5NextProcessor.from_pretrained = classmethod(
    lambda cls, path, **kwargs: ("glm5-next", str(path))
)
assert patch.install_glm5_next_processor_patch() is True
assert patch.install_glm5_next_processor_patch() is False
assert patch.is_installed() is True

from transformers import AutoProcessor
assert AutoProcessor.from_pretrained(model_path) == ("glm5-next", str(model_path))
from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat
assert MODEL_CONFIG["glm5_next"] is MessageFormat.LIST_WITH_IMAGE_FIRST
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mllm_load_installs_processor_before_runtime_load() -> None:
    script = """
from vllm_mlx.models import mllm
from vllm_mlx.patches import glm5_next_forget_gate_quant as quant_patch
from vllm_mlx.patches import glm5_next_processor as patch
from vllm_mlx.patches import glm5_next_runtime as runtime_patch

events = []
patch.install_glm5_next_processor_patch = lambda: events.append("processor")
quant_patch.install_glm5_next_forget_gate_quant_fix = lambda: events.append("quant")
runtime_patch.install_glm5_next_runtime_fix = lambda: events.append("runtime")
mllm._require_mlx_vlm = lambda: None

import mlx_vlm
mlx_vlm.load = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop"))

try:
    mllm.MLXMultimodalLM("local/model").load()
except RuntimeError as exc:
    assert str(exc) == "stop"
else:
    raise AssertionError("stubbed load must stop")
assert events == ["runtime", "processor", "quant"]
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
