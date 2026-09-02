# SPDX-License-Identifier: Apache-2.0
"""Integration coverage against the actually installed Wan MLX runtime.

The fake-based contracts in ``tests/test_wan_diffusers.py`` run everywhere,
including the Linux no-MLX leg, but they cannot notice when the pinned
``mlx-video-with-audio`` distribution itself drifts: a changed
``generate_video`` signature, a renamed loader global, or a different
``mx.load``/reshape behavior would pass every fake and fail only on a real
generation. These tests close that gap by exercising the production adapter
against the real ``mlx_video.generate_wan.generate_video`` code object and
real safetensors IO through ``mlx.core`` — without downloading checkpoints,
touching the network, or allocating model-sized tensors.
"""

from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_mlx

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_video.generate_wan")

import mlx.core as mx
import mlx.nn as nn
import mlx_video.generate_wan as wan_generator

import vllm_mlx.video.wan_diffusers as wan_diffusers
from tests.test_wan_diffusers import _layout
from vllm_mlx.video.wan import WanVideoEngine
from vllm_mlx.video.wan_diffusers import (
    _load_sharded,
    _scoped_generate_function,
    _transformer_key,
)

_WAN_ENV_KEYS = (
    "RAPID_MLX_WAN_MODEL_DIR",
    "RAPID_MLX_WAN_STEPS",
    "RAPID_MLX_WAN_SCHEDULER",
    "RAPID_MLX_WAN_TILING",
    "RAPID_MLX_WAN_LORA",
    "RAPID_MLX_WAN_LORA_HIGH",
    "RAPID_MLX_WAN_LORA_LOW",
)


class _FirstLoaderSeamReachedError(Exception):
    """Sentinel proving the real pipeline reached the injected loader seam."""


def test_pinned_runtime_distribution_version_matches_contract() -> None:
    # The adapter's cloned-globals technique is validated against exactly this
    # release; a silent bump must fail here before it fails on a user machine.
    assert importlib.metadata.version("mlx-video-with-audio") == "0.1.36"


def test_real_generate_video_reaches_first_loader_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the production adapter into the REAL upstream ``generate_video``.

    The request must survive the runtime's own config parsing, kwarg
    normalization, frame/alignment validation and seed setup, then stop at the
    first injected loader seam (``load_t5_encoder``) — before any tokenizer
    download or model-sized allocation.
    """
    root = _layout(tmp_path / "checkpoint")
    for key in _WAN_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    # The scoped clone must wrap the genuine upstream code object, and cloning
    # must not change the public signature the adapter calls.
    scoped = _scoped_generate_function(root, wan_generator)
    assert scoped.__code__ is wan_generator.generate_video.__code__
    # FunctionType clones do not carry __annotations__, so compare the
    # parameters the adapter actually relies on — name, order, kind and
    # default — instead of full Signature equality.
    scoped_params = inspect.signature(scoped).parameters
    real_params = inspect.signature(wan_generator.generate_video).parameters
    assert [(p.name, p.kind, p.default) for p in scoped_params.values()] == [
        (p.name, p.kind, p.default) for p in real_params.values()
    ]

    seam_calls: list[tuple[Path, object]] = []

    def stop_at_t5(seam_root: Path, config: object) -> None:
        seam_calls.append((seam_root, config))
        raise _FirstLoaderSeamReachedError

    monkeypatch.setattr(wan_diffusers, "_load_t5", stop_at_t5)

    engine = WanVideoEngine(str(root))
    output_path = tmp_path / "out.mp4"
    with pytest.raises(_FirstLoaderSeamReachedError):
        engine.generate(
            prompt="integration smoke",
            output_path=output_path,
            width=64,
            height=64,
            num_frames=5,
            seed=7,
            image=None,
        )

    assert not output_path.exists()
    assert len(seam_calls) == 1
    seam_root, config = seam_calls[0]
    # Loaders must be bound to the original checkpoint, not the temp view.
    assert seam_root == engine.model_path
    assert seam_root == root.resolve()
    # The synthesized view config must materialize the pinned 1.3B preset in
    # the runtime's own WanModelConfig, with the T5 fields _load_t5 consumes.
    assert (config.dim, config.ffn_dim, config.num_heads, config.num_layers) == (
        1536,
        8960,
        12,
        30,
    )
    assert config.model_version == "2.1"
    assert config.dual_model is False
    assert config.patch_size == (1, 2, 2)
    assert config.vae_stride == (4, 8, 8)
    assert (config.t5_dim, config.t5_num_layers, config.t5_vocab_size) == (
        4096,
        24,
        256384,
    )


class _TinyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.modulation = mx.zeros((2, 3))
        self.head = nn.Linear(2, 3)


class _TinyPatchModel(nn.Module):
    """Smallest parameter tree matching real ``_transformer_key`` targets."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding_proj = nn.Linear(12, 2)
        self.head = _TinyHead()


def test_load_sharded_real_safetensors_preserves_patch_layout(
    tmp_path: Path,
) -> None:
    """Round-trip real ``mx.save_safetensors``/``mx.load`` through the loader.

    Position-distinct values prove the patch reshape flattens each kernel in
    [C, pt, ph, pw] order exactly as WanModel's patchifier expects — a
    channels-last transpose would produce the right shape but wrong numbers.
    """
    directory = tmp_path / "transformer"
    directory.mkdir()
    patch_weight = mx.arange(24, dtype=mx.float32).reshape(2, 3, 1, 2, 2)
    patch_bias = mx.array([5.0, 7.0])
    modulation = mx.arange(100, 106, dtype=mx.float32).reshape(2, 3)
    head_weight = mx.arange(200, 206, dtype=mx.float32).reshape(3, 2)
    head_bias = mx.array([300.0, 301.0, 302.0])
    mx.save_safetensors(
        str(directory / "shard-a.safetensors"),
        {"patch_embedding.weight": patch_weight, "patch_embedding.bias": patch_bias},
    )
    mx.save_safetensors(
        str(directory / "shard-b.safetensors"),
        {
            "scale_shift_table": modulation,
            "proj_out.weight": head_weight,
            "proj_out.bias": head_bias,
        },
    )
    (directory / "weights.index.json").write_text(
        '{"weight_map": {'
        '"patch_embedding.weight": "shard-a.safetensors", '
        '"patch_embedding.bias": "shard-a.safetensors", '
        '"scale_shift_table": "shard-b.safetensors", '
        '"proj_out.weight": "shard-b.safetensors", '
        '"proj_out.bias": "shard-b.safetensors"}}'
    )

    model = _TinyPatchModel()
    _load_sharded(
        model,
        tmp_path,
        "transformer/weights.index.json",
        5,
        _transformer_key,
        reshape_patch=True,
    )
    mx.eval(model.parameters())

    loaded_patch = model.patch_embedding_proj.weight
    assert loaded_patch.shape == (2, 12)
    assert loaded_patch.dtype == mx.float32
    assert mx.array_equal(loaded_patch, mx.arange(24, dtype=mx.float32).reshape(2, 12))
    assert mx.array_equal(model.patch_embedding_proj.bias, patch_bias)
    assert mx.array_equal(model.head.modulation, modulation)
    assert mx.array_equal(model.head.head.weight, head_weight)
    assert mx.array_equal(model.head.head.bias, head_bias)
