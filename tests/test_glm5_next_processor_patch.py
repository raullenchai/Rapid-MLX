# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the GLM-5 Next processor compatibility layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core", reason="requires Apple MLX")

from vllm_mlx.patches import glm5_next_processor as processor_patch
from vllm_mlx.patches.glm5_next_processor import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize,
)


class _TokenizerStub:
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        self.calls = []

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return {"<|image|>": 120, "<|video|>": 121}[token]

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        rows = [[1] + [120] * text.count("<|image|>") + [2] for text in texts]
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    @staticmethod
    def batch_decode(*args, **kwargs):
        return (args, kwargs)

    @staticmethod
    def decode(*args, **kwargs):
        return (args, kwargs)

    @staticmethod
    def apply_chat_template(*args, **kwargs):
        return (args, kwargs)


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


def test_smart_resize_downscales_and_rejects_impossible_budget() -> None:
    height, width = smart_resize(
        2,
        1000,
        500,
        factor=28,
        min_image_tokens=1,
        max_image_tokens=64,
    )
    assert height % 28 == width % 28 == 0
    assert 2 * height * width <= 64 * 2 * 28**2
    with pytest.raises(ValueError, match="too small"):
        smart_resize(2, 100, 100, factor=28, max_image_tokens=0)
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(
            2,
            1,
            10_000,
            factor=28,
            min_image_tokens=1,
            max_image_tokens=1,
        )


def test_image_shape_conversion_and_resize_edges(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (4, 3), 127).save(path)
    from_path = processor_patch._to_channel_first(path, True)
    assert from_path.shape == (3, 3, 4)
    assert processor_patch._to_channel_first(
        np.zeros((3, 5), dtype=np.uint8), True
    ).shape == (3, 3, 5)

    alpha = np.zeros((4, 5, 4), dtype=np.uint8)
    assert processor_patch._to_channel_first(
        alpha, True, input_data_format="channels_last"
    ).shape == (3, 4, 5)
    channel_first_gray = np.zeros((1, 3, 5), dtype=np.uint8)
    assert processor_patch._to_channel_first(channel_first_gray, True).shape == (
        3,
        3,
        5,
    )
    with pytest.raises(ValueError, match="3D image"):
        processor_patch._to_channel_first(np.zeros((4,)), True)
    with pytest.raises(ValueError, match="infer channel dimension"):
        processor_patch._to_channel_first(np.zeros((2, 3, 5)), False)
    with pytest.raises(ValueError, match="RGB image"):
        processor_patch._to_channel_first(
            np.zeros((2, 3, 5)), False, input_data_format="channels_first"
        )

    float_image = np.full((3, 2, 2), 0.5, dtype=np.float32)
    resized = processor_patch._resize_channel_first(float_image, 4, 4)
    assert resized.shape == (3, 4, 4)
    assert resized.dtype == np.float32
    assert np.allclose(resized, 0.5, atol=1 / 255)


def test_path_image_is_materialized_before_file_closes(monkeypatch) -> None:
    image = Image.new("RGB", (5, 4), "red")
    state = {"closed": False}

    class TrackedImage:
        def __enter__(self):
            return image

        def __exit__(self, *_args):
            state["closed"] = True
            image.close()

    monkeypatch.setattr(processor_patch.Image, "open", lambda _path: TrackedImage())
    result = processor_patch._to_channel_first("image.png", True)
    assert state["closed"] is True
    assert result.shape == (3, 4, 5)


def test_ambiguous_channel_shape_preserves_pixels_with_explicit_format() -> None:
    channel_first = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
    inferred = processor_patch._to_channel_first(channel_first, True)
    assert np.array_equal(inferred, channel_first)

    channel_last = np.transpose(channel_first, (1, 2, 0))
    explicit = processor_patch._to_channel_first(
        channel_last,
        True,
        input_data_format="channels_last",
    )
    assert np.array_equal(explicit, channel_first)


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


def test_processor_rejects_more_tokens_than_images() -> None:
    processor = Glm5NextProcessor(tokenizer=_TokenizerStub())
    with pytest.raises(ValueError, match="More image tokens"):
        processor(
            images=[Image.new("RGB", (28, 28), "blue")],
            text=["<|image|><|image|>"],
        )


@pytest.mark.parametrize("images", [None, []])
def test_processor_rejects_image_marker_without_media(images) -> None:
    processor = Glm5NextProcessor(tokenizer=_TokenizerStub())
    with pytest.raises(ValueError, match="without images"):
        processor(images=images, text=["inspect <|image|>"])


def test_processor_uses_request_merge_size_for_placeholder_count() -> None:
    image_processor = Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=1,
        max_image_tokens=16,
    )
    processor = Glm5NextProcessor(
        image_processor=image_processor,
        tokenizer=_TokenizerStub(),
    )
    inputs = processor(
        images=Image.new("RGB", (16, 16), "blue"),
        text="<|image|>",
        merge_size=4,
    )
    image_tokens = int(mx.sum(inputs["input_ids"] == 120).item())
    expected = int(inputs["image_grid_thw"][0].prod().item()) // 4**2
    assert image_tokens == expected


def test_image_processor_scalar_paths_and_optional_transforms() -> None:
    image_processor = Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=1,
        max_image_tokens=4,
    )
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    assert len(image_processor.fetch_images(image)) == 1
    output = image_processor(
        images=image,
        return_tensors=None,
        do_rescale=False,
        do_normalize=False,
    )
    assert output["pixel_values"].shape == (8, 24)
    with pytest.raises(ValueError, match="must not be None"):
        image_processor.preprocess()


def test_text_only_defaults_delegation_and_model_inputs() -> None:
    tokenizer = _TokenizerStub()
    processor = Glm5NextProcessor(tokenizer=tokenizer)
    empty = processor(text=None, return_tensors=None)
    assert empty["input_ids"] == [[1, 2]]
    single = processor(
        text="hello",
        padding_side="left",
        return_mm_token_type_ids=False,
        return_tensors=None,
    )
    assert single["input_ids"] == [[1, 2]]
    assert tokenizer.calls[-1][1]["padding_side"] == "left"
    assert processor.batch_decode([1], skip_special_tokens=True) == (
        ([1],),
        {"skip_special_tokens": True},
    )
    assert processor.decode([1]) == (([1],), {})
    assert processor.apply_chat_template([{"role": "user"}]) == (
        ([{"role": "user"}],),
        {},
    )
    assert processor.model_input_names == [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    ]


def test_local_optional_metadata_miss_never_falls_through_to_hub(
    monkeypatch, tmp_path: Path
) -> None:
    import huggingface_hub

    def fail(*_args, **_kwargs):
        raise AssertionError("local checkpoint must not make a Hub request")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    assert processor_patch._load_json(tmp_path, "processor_config.json") is None


def test_json_metadata_local_remote_and_invalid_shapes(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"vision_config":{"patch_size":16}}')
    assert processor_patch._load_json(tmp_path, "config.json") == {
        "vision_config": {"patch_size": 16}
    }
    config.write_text("[]")
    assert processor_patch._load_json(tmp_path, "config.json") is None

    downloaded = tmp_path / "downloaded.json"
    downloaded.write_text('{"remote":true}')
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *_args, **_kwargs: downloaded
    )
    assert processor_patch._load_json("org/model", "config.json") == {"remote": True}
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LocalEntryNotFoundError("offline")
        ),
    )
    assert processor_patch._load_json("org/model", "config.json") is None

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *_args, **_kwargs: malformed
    )
    with pytest.raises(ValueError):
        processor_patch._load_json("org/model", "config.json")


def test_local_metadata_honors_subfolder(tmp_path: Path) -> None:
    subfolder = tmp_path / "tested-revision"
    subfolder.mkdir()
    (subfolder / "config.json").write_text('{"revision":"pinned"}')
    assert processor_patch._load_json(
        tmp_path,
        "config.json",
        hub_kwargs={"subfolder": "tested-revision", "local_files_only": True},
    ) == {"revision": "pinned"}


def test_image_processor_kwargs_merge_checkpoint_metadata(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"vision_config":{"patch_size":16,"temporal_patch_size":3,'
        '"spatial_merge_size":4}}'
    )
    (tmp_path / "processor_config.json").write_text(
        '{"image_processor":{"patch_size":14,"max_image_tokens":99}}'
    )
    assert processor_patch._image_processor_kwargs(tmp_path) == {
        "patch_size": 14,
        "max_image_tokens": 99,
        "temporal_patch_size": 3,
        "merge_size": 4,
    }


def test_from_pretrained_builds_local_processor(monkeypatch, tmp_path: Path) -> None:
    tokenizer = _TokenizerStub()
    tokenizer.chat_template = "tokenizer template"
    (tmp_path / "config.json").write_text(
        '{"vision_config":{"patch_size":16,"spatial_merge_size":2}}'
    )
    (tmp_path / "processor_config.json").write_text(
        '{"chat_template":"checkpoint template"}'
    )
    loaded = []
    monkeypatch.setattr(
        processor_patch.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        processor_patch,
        "load_chat_template",
        lambda instance, path: loaded.append((instance, path)),
    )
    processor = Glm5NextProcessor.from_pretrained(tmp_path, return_tensors="mlx")
    assert processor.chat_template == "checkpoint template"
    assert processor.image_processor.patch_size == 16
    assert loaded == [(tokenizer, tmp_path)]


def test_from_pretrained_forwards_hub_identity_to_all_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    import huggingface_hub

    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    hub_calls = []
    tokenizer_calls = []

    def fake_download(repo_id, filename, **kwargs):
        hub_calls.append((repo_id, filename, kwargs))
        return metadata

    tokenizer = _TokenizerStub()
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        processor_patch.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: tokenizer_calls.append((path, kwargs)) or tokenizer,
    )

    options = {
        "revision": "tested-commit",
        "cache_dir": tmp_path / "hub-cache",
        "local_files_only": True,
        "force_download": False,
        "token": "test-token",
        "subfolder": "processor",
    }
    Glm5NextProcessor.from_pretrained("org/model", **options)

    assert tokenizer_calls == [("org/model", options)]
    assert [filename for _, filename, _ in hub_calls] == [
        "processor_config.json",
        "config.json",
        "processor_config.json",
    ]
    assert all(repo_id == "org/model" for repo_id, _, _ in hub_calls)
    assert all(kwargs == options for _, _, kwargs in hub_calls)


def test_installer_registers_once_without_replacing_existing_prompt_shape(
    monkeypatch,
) -> None:
    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

    calls = []
    original = MODEL_CONFIG.get("glm5_next")
    existed = "glm5_next" in MODEL_CONFIG
    MODEL_CONFIG.pop("glm5_next", None)
    processor_patch._INSTALLED = False
    monkeypatch.setattr(
        processor_patch,
        "install_auto_processor_patch",
        lambda model_type, processor: calls.append((model_type, processor)),
    )
    try:
        assert processor_patch.install_glm5_next_processor_patch() is True
        assert processor_patch.install_glm5_next_processor_patch() is False
        assert processor_patch.is_installed() is True
        assert calls == [("glm5_next", Glm5NextProcessor)]
        assert MODEL_CONFIG["glm5_next"] is MessageFormat.LIST_WITH_IMAGE_FIRST
    finally:
        if existed:
            MODEL_CONFIG["glm5_next"] = original
        else:
            MODEL_CONFIG.pop("glm5_next", None)
        processor_patch._INSTALLED = False


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
