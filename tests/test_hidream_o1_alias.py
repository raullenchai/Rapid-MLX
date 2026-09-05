# SPDX-License-Identifier: Apache-2.0
"""Contracts for the HiDream-O1 Dev MLX image backend and product alias."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vllm_mlx import _download_gate
from vllm_mlx.catalog import build_catalog_bundle
from vllm_mlx.image.engine import (
    ImageGenerationEngine,
    ImageRuntimeError,
    _detect_family,
)
from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.model_sizes import size_bytes
from vllm_mlx.runtime.resident_models import estimate_model_bytes

REPO = "mlx-community/HiDream-O1-Image-Dev-mlx-bf16"
REVISION = "33c7a00bce8e3410304f83ec408a15a1eb6782df"
SIZE = 17_649_873_024


def test_alias_routes_to_generation_only_image_backend() -> None:
    profile = resolve_profile("hidream-o1-dev")
    assert profile is not None
    assert profile.hf_path == REPO
    assert profile.modality == "image-gen"
    assert profile.min_memory_gb == 32
    engine = ImageGenerationEngine(REPO)
    assert engine.family == "hidream-o1-dev"
    assert engine.supports_generation is True
    assert engine.supports_editing is False
    assert engine.default_steps == 28
    assert engine._prequantized is True  # noqa: SLF001


def test_alias_pins_download_size_revision_and_resident_budget() -> None:
    assert _download_gate.IMAGE_MODEL_REVISIONS[REPO] == REVISION
    assert size_bytes(REPO) == SIZE
    for name in ("hidream-o1-dev", REPO, "/models/hidream_o1_dev"):
        assert estimate_model_bytes(name) == int(18.0 * 1024**3)


def test_atomic_catalog_names_the_new_backend() -> None:
    bundle = build_catalog_bundle()
    record = next(
        alias
        for alias in bundle["snapshot"]["aliases"]
        if alias["alias"] == "hidream-o1-dev"
    )
    capabilities = record["capabilities"]
    assert capabilities["runtime_adapter"] == "rapid_mlx/hidream_o1"
    assert capabilities["operation_modes"] == ["text_to_image"]


@pytest.mark.parametrize(
    ("name", "family"),
    [
        (REPO, "hidream-o1-dev"),
        ("/models/hidream_o1_dev", "hidream-o1-dev"),
    ],
)
def test_family_detection(name: str, family: str) -> None:
    assert _detect_family(name) == family


def test_patch_round_trip_and_published_schedule() -> None:
    pytest.importorskip("mlx")
    from vllm_mlx.image.hidream_runtime.runtime import (
        DEFAULT_TIMESTEPS,
        FlashFlowMatchScheduler,
        _build_sample,
        _patchify,
        _unpatchify,
    )

    image = np.arange(3 * 64 * 96, dtype=np.float32).reshape(3, 64, 96)
    patches = _patchify(image)
    assert patches.shape == (6, 3 * 32 * 32)
    np.testing.assert_array_equal(_unpatchify(patches, 64, 96), image)
    scheduler = FlashFlowMatchScheduler()
    assert tuple(int(value) for value in scheduler.timesteps) == DEFAULT_TIMESTEPS
    assert len(scheduler.sigmas) == 29
    assert float(scheduler.sigmas[-1]) == 0.0

    class OversizedProcessor:
        tokenizer = None
        boi_token = "<boi>"
        tms_token = "<tms>"

        def __init__(self) -> None:
            self.tokenizer = self

        def apply_chat_template(self, *_args, **_kwargs) -> str:
            return "caption"

        def encode(self, *_args, **_kwargs) -> list[int]:
            return list(range(1025))

    config = SimpleNamespace(
        image_token_id=1, video_token_id=2, vision_start_token_id=3
    )
    with pytest.raises(ValueError, match="1024 tokens"):
        _build_sample("prompt", 1024, 1024, OversizedProcessor(), config)


def test_hidream_runtime_numeric_helpers_and_masks() -> None:
    pytest.importorskip("mlx")
    import mlx.core as mx

    from vllm_mlx.image.hidream_runtime import HiDreamO1
    from vllm_mlx.image.hidream_runtime.runtime import (
        BottleneckPatchEmbed,
        FinalLayer,
        FlashFlowMatchScheduler,
        HiDreamConfig,
        TimestepEmbedder,
        _attention_mask,
        _patchify,
        _rope_positions,
        _validate_custom_head_weights,
    )

    assert HiDreamO1.__name__ == "HiDreamO1"
    assert HiDreamConfig().hidden_size == 4096

    sample = mx.ones((1, 2), dtype=mx.bfloat16)
    scheduler = FlashFlowMatchScheduler()
    stepped = scheduler.step(mx.zeros_like(sample), sample, seed=7)
    mx.eval(stepped)
    assert stepped.shape == sample.shape
    assert scheduler.step_index == 1

    assert TimestepEmbedder.embedding(mx.array([0.5]), 4).shape == (1, 4)
    assert TimestepEmbedder.embedding(mx.array([0.5]), 5).shape == (1, 5)
    time_embedder = TimestepEmbedder(hidden_size=4, frequency_embedding_size=4)
    assert time_embedder(mx.array([0.5])).shape == (1, 4)

    patch_embedder = BottleneckPatchEmbed(hidden_size=4)
    assert patch_embedder(mx.zeros((1, 1, 3 * 32 * 32))).shape == (1, 1, 4)
    final = FinalLayer(hidden_size=4)
    assert final(mx.zeros((1, 1, 4))).shape == (1, 1, 3 * 32 * 32)

    expected = {"weight": SimpleNamespace(shape=(2, 3))}
    _validate_custom_head_weights(expected, dict(expected))
    with pytest.raises(ValueError, match="missing=.*weight"):
        _validate_custom_head_weights(expected, {})

    with pytest.raises(ValueError, match="multiples of 32"):
        _patchify(np.zeros((3, 33, 32), dtype=np.float32))

    plain = _rope_positions(
        input_ids=np.asarray([[10, 11]], dtype=np.int64),
        image_grid=np.empty((0, 3), dtype=np.int64),
        image_token_id=20,
        video_token_id=21,
        vision_start_token_id=22,
    )
    np.testing.assert_array_equal(plain[:, 0], [[0, 1], [0, 1], [0, 1]])

    image = _rope_positions(
        input_ids=np.asarray([[10, 22, 20]], dtype=np.int64),
        image_grid=np.asarray([[1, 1, 2]], dtype=np.int64),
        image_token_id=20,
        video_token_id=21,
        vision_start_token_id=22,
    )
    assert image.shape == (3, 1, 3)
    with pytest.raises(ValueError, match="video tokens"):
        _rope_positions(
            input_ids=np.asarray([[22, 21]], dtype=np.int64),
            image_grid=np.empty((0, 3), dtype=np.int64),
            image_token_id=20,
            video_token_id=21,
            vision_start_token_id=22,
        )

    mask = _attention_mask(np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.int64))
    assert mask.shape == (2, 1, 3, 3)
    assert np.all(mask[0, 0, 1] == 0)
    assert mask[1, 0, 0, 1] < 0


def test_build_sample_supports_processor_and_tokenizer_shapes() -> None:
    pytest.importorskip("mlx")
    from vllm_mlx.image.hidream_runtime.runtime import _build_sample

    class Tokenizer:
        def encode(self, _caption, *, add_special_tokens):
            assert add_special_tokens is False
            return [7, 8]

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, *_args, **_kwargs):
            return "caption"

    config = SimpleNamespace(
        image_token_id=20, video_token_id=21, vision_start_token_id=22
    )
    processor = Processor()
    sample = _build_sample("fox", 64, 32, processor, config)
    assert processor.tokenizer.boi_token == "<|boi_token|>"
    assert processor.tokenizer.tms_token == "<|tms_token|>"
    assert sample["input_ids"].shape == (1, 2)
    assert sample["position_ids"].shape == (3, 1, 4)
    assert sample["token_types"].shape == (1, 4)
    assert int(sample["vinput_mask"].sum()) == 2

    class DirectTokenizer(Tokenizer):
        boi_token = "<boi>"
        tms_token = "<tms>"

        def apply_chat_template(self, *_args, **_kwargs):
            return "caption"

    assert _build_sample("fox", 64, 32, DirectTokenizer(), config)[
        "input_ids"
    ].shape == (1, 2)


def test_hidream_constructor_loads_only_validated_custom_heads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("mlx")
    import mlx.core as mx
    import mlx.nn as nn

    from vllm_mlx.image.hidream_runtime import runtime

    class TinyHead(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = mx.zeros((1,))

    class TinyLanguageModel:
        model = SimpleNamespace(embed_tokens=lambda value: value)

    backbone = SimpleNamespace(
        config=SimpleNamespace(),
        vision_tower=object(),
        language_model=TinyLanguageModel(),
    )
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.load = lambda model_path: (backbone, f"processor:{model_path}")
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(runtime, "TimestepEmbedder", TinyHead)
    monkeypatch.setattr(runtime, "BottleneckPatchEmbed", TinyHead)
    monkeypatch.setattr(runtime, "FinalLayer", TinyHead)
    monkeypatch.setattr(
        runtime.mx,
        "load",
        lambda _path: {
            "t_embedder1.weight": mx.zeros((1,)),
            "x_embedder.weight": mx.zeros((1,)),
            "final_layer2.weight": mx.zeros((1,)),
        },
    )
    loaded = []
    monkeypatch.setattr(
        runtime.HiDreamO1,
        "load_weights",
        lambda _self, weights, *, strict: loaded.append((weights, strict)),
    )

    with pytest.raises(FileNotFoundError, match="custom heads"):
        runtime.HiDreamO1(str(tmp_path))

    custom = tmp_path / "extras" / "custom_heads.safetensors"
    custom.parent.mkdir()
    custom.write_bytes(b"weights")
    model = runtime.HiDreamO1(str(tmp_path), on_step=lambda *_args: None)
    assert model.processor == f"processor:{tmp_path}"
    assert model.visual is backbone.vision_tower
    assert loaded and loaded[0][1] is False


def test_hidream_forward_and_tiny_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mlx")
    import mlx.core as mx

    from vllm_mlx.image.hidream_runtime import runtime

    class TinyLanguageCore:
        def embed_tokens(self, input_ids):
            return mx.zeros((*input_ids.shape, 4), dtype=mx.float32)

        def __call__(self, _ids, *, inputs_embeds, **_kwargs):
            return inputs_embeds

    core = TinyLanguageCore()
    forward_self = SimpleNamespace(
        config=SimpleNamespace(tms_token_id=9),
        t_embedder1=lambda _value: mx.ones((1, 4)),
        x_embedder=lambda value: mx.zeros((*value.shape[:2], 4)),
        language_model=SimpleNamespace(model=core),
        final_layer2=lambda value: value,
    )
    input_ids = mx.array([[9, 2]])
    embeddings = runtime.HiDreamO1._text_embeddings(forward_self, input_ids)
    forwarded = runtime.HiDreamO1._forward(
        forward_self,
        embeddings,
        input_ids,
        mx.zeros((3, 1, 3), dtype=mx.int32),
        mx.zeros((1, 1, 3 * 32 * 32)),
        mx.array([0.5]),
        mx.zeros((1, 1, 3, 3)),
    )
    mx.eval(forwarded)
    assert forwarded.shape == (1, 3, 4)

    monkeypatch.setattr(
        runtime,
        "_build_sample",
        lambda *_args, **_kwargs: {
            "input_ids": np.asarray([[9]], dtype=np.int64),
            "position_ids": np.zeros((3, 1, 1), dtype=np.int64),
            "token_types": np.asarray([[1]], dtype=np.int64),
            "vinput_mask": np.asarray([[True]]),
        },
    )
    progress = []
    generate_self = SimpleNamespace(
        processor=object(),
        backbone_config=object(),
        on_step=lambda step, total: progress.append((step, total)),
        _text_embeddings=lambda ids: mx.zeros((*ids.shape, 4)),
        _forward=lambda _emb, _ids, _pos, patches, _time, _mask: patches,
    )
    result = runtime.HiDreamO1.generate_image(
        generate_self, seed=3, prompt="fox", height=32, width=32
    )
    assert result.image.size == (32, 32)
    assert result.image.mode == "RGB"
    assert progress[0] == (0, 28)
    assert progress[-1] == (28, 28)

    with pytest.raises(ValueError, match="28-step"):
        runtime.HiDreamO1.generate_image(
            generate_self,
            seed=3,
            prompt="fox",
            num_inference_steps=27,
            height=32,
            width=32,
        )
    with pytest.raises(ValueError, match="multiples of 32"):
        runtime.HiDreamO1.generate_image(
            generate_self, seed=3, prompt="fox", height=33, width=32
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_inference_steps": 27}, "28-step"),
        ({"prompt": "x" * 4097}, "4096 characters"),
        ({"width": 0}, "between 256 and 2048"),
        ({"height": 4096}, "between 256 and 2048"),
        ({"width": 1008}, "multiples of 32"),
        ({"guidance": 4.0}, "omit the guidance"),
        ({"negative_prompt": "bad"}, "does not support negative_prompt"),
        ({"image_paths": ["source.png"]}, "text-to-image only"),
    ],
)
def test_unsupported_requests_fail_before_loading(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, message: str
) -> None:
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        engine,
        "_ensure_loaded",
        lambda **_kwargs: pytest.fail("invalid request reached the 17 GB loader"),
    )
    request = {
        "prompt": "test",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "seed": 1,
        "guidance": None,
        "negative_prompt": None,
        "image_paths": None,
    }
    request.update(kwargs)
    with pytest.raises(ImageRuntimeError, match=message):
        engine.generate(**request)


@pytest.mark.requires_mlx
def test_token_dense_prompt_fails_before_the_17gb_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    class DenseTokenizer:
        boi_token = None
        tms_token = None

        def apply_chat_template(self, *_args, **_kwargs):
            return "caption"

        def encode(self, *_args, **_kwargs):
            return list(range(1025))

    tokenizer_calls = []
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda source, **kwargs: (
            tokenizer_calls.append((source, kwargs)) or DenseTokenizer()
        ),
    )
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda _name: None
    )
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        engine,
        "_ensure_loaded",
        lambda **_kwargs: pytest.fail("token-dense prompt reached the 17 GB loader"),
    )

    with pytest.raises(ImageRuntimeError, match="1024 tokens"):
        engine.generate(
            prompt="x",
            width=1024,
            height=1024,
            num_inference_steps=28,
        )

    assert tokenizer_calls == [
        (
            REPO,
            {"trust_remote_code": False, "revision": REVISION},
        )
    ]


@pytest.mark.requires_mlx
def test_prompt_tokenizer_failures_are_clean_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda _name: None
    )
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(ImageRuntimeError, match="load.*tokenizer.*offline"):
        engine._validate_hidream_prompt_tokens("fox")

    class BrokenTokenizer:
        boi_token = "<boi>"
        tms_token = "<tms>"

        def apply_chat_template(self, *_args, **_kwargs):
            raise ValueError("bad template")

    engine._prompt_tokenizer = BrokenTokenizer()
    with pytest.raises(ImageRuntimeError, match="tokenize.*bad template"):
        engine._validate_hidream_prompt_tokens("fox")


@pytest.mark.requires_mlx
def test_cancel_during_prompt_tokenizer_init_never_reaches_model_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import AutoTokenizer

    class Tokenizer:
        boi_token = "<boi>"
        tms_token = "<tms>"

        def apply_chat_template(self, *_args, **_kwargs):
            return "caption"

        def encode(self, *_args, **_kwargs):
            return [1]

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda _name: None
    )

    def load_then_cancel(*_args, **_kwargs):
        engine.request_cancel()
        return Tokenizer()

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_then_cancel)
    monkeypatch.setattr(
        engine,
        "_ensure_loaded",
        lambda **_kwargs: pytest.fail("cancelled preflight reached the model loader"),
    )

    from vllm_mlx.image.engine import ImageGenerationCancelled

    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        engine.generate(prompt="fox", width=1024, height=1024, num_inference_steps=28)
    assert engine._active_seq == 0


def test_hidream_cold_snapshot_is_pinned_allowlisted_and_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(
        "vllm_mlx._download_gate.mflux_local_snapshot", lambda _name: None
    )
    verified = []
    monkeypatch.setattr(
        engine, "_verify_weights_complete", lambda: verified.append(True)
    )
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda model, **kwargs: calls.append((model, kwargs)) or "/pinned/snapshot",
    )

    assert engine._model_path_for_mflux() == "/pinned/snapshot"
    assert calls == [
        (
            REPO,
            {
                "revision": REVISION,
                "allow_patterns": list(_download_gate.HIDREAM_O1_DATA_FILES),
            },
        )
    ]
    assert verified == [True]


@pytest.mark.requires_mlx
def test_hidream_engine_build_progress_cancel_and_generate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.image import hidream_runtime
    from vllm_mlx.image.engine import ImageGenerationCancelled

    built = []

    class TinyHiDream:
        def __init__(self, path, *, on_step):
            built.append((path, on_step))
            self.calls = []

        def generate_image(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(image=Image.new("RGB", (8, 8)))

    monkeypatch.setattr(hidream_runtime, "HiDreamO1", TinyHiDream)
    engine = ImageGenerationEngine(REPO)
    monkeypatch.setattr(engine, "_model_path_for_mflux", lambda: "/snapshot")
    model = engine._build_model()
    assert built == [("/snapshot", engine._report_hidream_step)]

    engine._report_hidream_step(4, 28)
    assert engine._progress["running"] is True
    assert engine._progress["step"] == 4
    assert engine._progress["total"] == 28
    monkeypatch.setattr(engine, "_is_cancelled", lambda: True)
    with pytest.raises(ImageGenerationCancelled, match="cancelled"):
        engine._report_hidream_step(5, 28)

    monkeypatch.setattr(engine, "_is_cancelled", lambda: False)
    monkeypatch.setattr(engine, "_ensure_loaded", lambda **_kwargs: model)
    monkeypatch.setattr(engine, "_validate_hidream_prompt_tokens", lambda _prompt: None)
    png = engine.generate(
        prompt="fox", width=256, height=256, num_inference_steps=28, seed=11
    )
    assert png.startswith(b"\x89PNG")
    assert model.calls == [
        {
            "height": 256,
            "width": 256,
            "seed": 11,
            "prompt": "fox",
            "num_inference_steps": 28,
        }
    ]

    missing = ImageGenerationEngine(REPO)
    monkeypatch.setattr(missing, "_model_path_for_mflux", lambda: None)
    with pytest.raises(ImageRuntimeError, match="local model snapshot"):
        missing._build_model()


@pytest.mark.requires_mlx
def test_hidream_sidecar_requires_mlx_vlm_not_mflux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx.runtime import image_lane

    probes = []
    monkeypatch.setattr(
        image_lane.importlib.util,
        "find_spec",
        lambda module: probes.append(module) or object(),
    )
    image_lane.require_image_runtime_or_exit(REPO)
    assert probes == ["mlx_vlm"]


def _seed_snapshot(root: Path, *, omit: str | None = None) -> None:
    for relative in _download_gate.HIDREAM_O1_DATA_FILES:
        if relative == omit:
            continue
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")


@pytest.mark.parametrize(
    "missing_file",
    ["extras/custom_heads.safetensors", "generation_config.json"],
)
def test_complete_hidream_snapshot_requires_every_runtime_data_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_file: str
) -> None:
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    snapshot = (
        tmp_path
        / "models--mlx-community--HiDream-O1-Image-Dev-mlx-bf16"
        / "snapshots"
        / REVISION
    )
    _seed_snapshot(snapshot)
    assert _download_gate.mflux_missing_weights(REPO) == []
    (snapshot / missing_file).unlink()
    assert _download_gate.mflux_missing_weights(REPO) == [missing_file]


def test_custom_heads_reject_missing_extra_and_shape_mismatch() -> None:
    pytest.importorskip("mlx")
    from vllm_mlx.image.hidream_runtime.runtime import (
        _validate_custom_head_weights,
    )

    expected = {
        "t_embedder1.fc1.weight": SimpleNamespace(shape=(4, 2)),
        "x_embedder.proj1.weight": SimpleNamespace(shape=(8, 4)),
    }
    with pytest.raises(ValueError, match="missing=.*x_embedder"):
        _validate_custom_head_weights(
            expected,
            {"t_embedder1.fc1.weight": SimpleNamespace(shape=(4, 2))},
        )
    with pytest.raises(ValueError, match="unexpected=.*rogue"):
        _validate_custom_head_weights(
            expected,
            {
                **expected,
                "rogue.weight": SimpleNamespace(shape=(1,)),
            },
        )
    with pytest.raises(ValueError, match="shape_mismatch=.*t_embedder1"):
        _validate_custom_head_weights(
            expected,
            {
                **expected,
                "t_embedder1.fc1.weight": SimpleNamespace(shape=(4, 3)),
            },
        )


@pytest.mark.parametrize("requested", [REPO, "hidream-o1-dev"])
def test_pull_uses_exact_revision_and_data_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **kwargs: calls.append((args.model, kwargs)),
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    args = SimpleNamespace(model=requested)

    cli.pull_command(args)

    assert calls == [
        (
            REPO,
            {
                "allow_patterns_override": list(_download_gate.HIDREAM_O1_DATA_FILES),
                "revision_override": REVISION,
            },
        )
    ]


def test_non_hidream_pull_keeps_the_generic_download_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_mlx import cli
    from vllm_mlx.audio import registry

    calls = []
    monkeypatch.setattr(
        cli, "_pull_repository", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)
    monkeypatch.setattr(registry, "runtime_assets_for", lambda _repo: ())
    monkeypatch.setattr(registry, "runtime_requirements_for", lambda _repo: ())
    args = SimpleNamespace(model="mlx-community/plain-model")

    cli.pull_command(args)

    assert calls == [((args,), {})]


def test_pinned_pull_bypasses_mirror_and_pins_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lower-level pull path must enforce, not merely receive, the pin."""
    from vllm_mlx import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "_try_mirror_prefetch",
        lambda *_args, **_kwargs: pytest.fail("pinned pulls cannot use mutable main"),
    )
    monkeypatch.setattr(cli, "_blob_identifier", lambda _root: ())
    monkeypatch.setattr(cli, "_print_pull_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _download_gate, "reap_orphan_incomplete_blobs", lambda _repo: (0, 0)
    )

    def fake_snapshot_download(repo_id: str, **kwargs) -> str:
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    cli._pull_repository(
        SimpleNamespace(model=REPO, bits=None, format=None, json=True),
        allow_patterns_override=list(_download_gate.HIDREAM_O1_DATA_FILES),
        revision_override=REVISION,
    )

    assert calls == [
        (
            REPO,
            {
                "allow_patterns": list(_download_gate.HIDREAM_O1_DATA_FILES),
                "revision": REVISION,
            },
        )
    ]
