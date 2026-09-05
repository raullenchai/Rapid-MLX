"""Rapid-owned adapter for the fixed Bonsai Image MLX checkpoint."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer, TokenizerLoader
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import (
    ComponentDefinition,
    TokenizerDefinition,
)
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import (
    Qwen3TextEncoder,
)
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mlx.utils import tree_unflatten

from ._vendor import (
    Flux2KleinFastTransformer,
    Flux2KleinMegakernelSpec,
    load_klein_fast_packed_weights_from_disk,
)

BONSAI_IMAGE_REPO = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
BONSAI_IMAGE_REVISION = "2c24c81b934a658ba5590cf39088ba929985b4a8"
BONSAI_IMAGE_STEPS = 4
BONSAI_IMAGE_GUIDANCE = 1.0
BONSAI_IMAGE_MAX_PROMPT_TOKENS = 512

_PACKED_TRANSFORMER = "transformer-packed-mflux"
_TEXT_ENCODER = "text_encoder-mlx-4bit"
_TEXT_ENCODER_BITS = 4
_TEXT_ENCODER_GROUP_SIZE = 64


class BonsaiCheckpointError(RuntimeError):
    """The pinned snapshot is absent, incomplete, or structurally invalid."""


class _VAEWeightDefinition:
    @staticmethod
    def get_components() -> list[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_vae_mapping,
            )
        ]

    @staticmethod
    def get_download_patterns() -> list[str]:
        return ["vae/*.safetensors", "vae/*.json"]

    @staticmethod
    def quantization_predicate(path: str, module: object) -> bool:
        return hasattr(module, "to_quantized")


def _tokenizer_definition() -> TokenizerDefinition:
    return TokenizerDefinition(
        name="qwen3",
        hf_subdir="tokenizer",
        tokenizer_class="Qwen2TokenizerFast",
        encoder_class=LanguageTokenizer,
        max_length=BONSAI_IMAGE_MAX_PROMPT_TOKENS,
        use_chat_template=True,
        chat_template_kwargs={"enable_thinking": False},
        download_patterns=["tokenizer/**"],
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BonsaiCheckpointError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise BonsaiCheckpointError(f"{path.name} must contain a JSON object.")
    return value


def _load_safetensors(path: Path) -> dict[str, mx.array]:
    loaded = mx.load(str(path))
    if not isinstance(loaded, dict):
        raise BonsaiCheckpointError(f"{path.name} did not contain named tensors.")
    return cast(dict[str, mx.array], loaded)


def _validate_checkpoint(root: Path) -> None:
    required = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        f"{_TEXT_ENCODER}/config.json",
        f"{_TEXT_ENCODER}/model.safetensors",
        f"{_PACKED_TRANSFORMER}/config.json",
        f"{_PACKED_TRANSFORMER}/quantization_config.json",
        f"{_PACKED_TRANSFORMER}/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    missing = [
        name
        for name in required
        if not (root / name).is_file() or (root / name).stat().st_size <= 0
    ]
    if missing:
        raise BonsaiCheckpointError(
            "Bonsai Image snapshot is incomplete; missing or empty: "
            + ", ".join(missing)
        )

    model_index = _read_json_object(root / "model_index.json")
    if model_index.get("_class_name") != "Flux2KleinPipeline":
        raise BonsaiCheckpointError(
            "Bonsai Image model_index.json is not a FLUX.2 Klein pipeline."
        )

    quant = _read_json_object(root / _PACKED_TRANSFORMER / "quantization_config.json")
    bits = quant.get("bits")
    group_size = quant.get("group_size")
    if (
        isinstance(bits, bool)
        or bits != 2
        or isinstance(group_size, bool)
        or group_size != 128
    ):
        raise BonsaiCheckpointError(
            "Bonsai Image requires the published MLX 2-bit/group-128 transformer."
        )

    text_config = _read_json_object(root / _TEXT_ENCODER / "config.json")
    quantization = text_config.get("quantization")
    if not isinstance(quantization, dict):
        raise BonsaiCheckpointError(
            "Bonsai Image text encoder has no quantization contract."
        )
    if quantization.get("bits") != 4 or quantization.get("group_size") != 64:
        raise BonsaiCheckpointError(
            "Bonsai Image requires the published MLX 4-bit/group-64 text encoder."
        )


def _load_text_encoder(root: Path, overrides: dict[str, Any]) -> Qwen3TextEncoder:
    weights_path = root / _TEXT_ENCODER / "model.safetensors"
    raw = _load_safetensors(weights_path)
    stripped = {
        key.removeprefix("model."): value
        for key, value in raw.items()
        if key.startswith("model.")
    }
    if not stripped:
        raise BonsaiCheckpointError(
            "Bonsai Image text encoder contains no model weights."
        )
    nested = tree_unflatten(list(stripped.items()))
    encoder = Qwen3TextEncoder(**overrides)
    nn.quantize(
        encoder,
        class_predicate=lambda _path, module: hasattr(module, "to_quantized"),
        bits=_TEXT_ENCODER_BITS,
        group_size=_TEXT_ENCODER_GROUP_SIZE,
    )
    encoder.update(nested)
    return encoder


def _load_transformer(root: Path) -> Flux2KleinFastTransformer:
    spec = Flux2KleinMegakernelSpec()
    packed_dir = root / _PACKED_TRANSFORMER
    weights = load_klein_fast_packed_weights_from_disk(
        packed_dir,
        spec,
        dtype=mx.bfloat16,
    )
    transformer = Flux2KleinFastTransformer(
        weights=weights,
        precision="2bit",
        patch_size=1,
        in_channels=spec.in_channels,
        out_channels=spec.in_channels,
        num_layers=spec.num_double_blocks,
        num_single_layers=spec.num_single_blocks,
        attention_head_dim=spec.head_dim,
        num_attention_heads=spec.num_heads,
        joint_attention_dim=spec.context_dim,
        timestep_guidance_channels=256,
        mlp_ratio=spec.mlp_ratio,
        axes_dims_rope=spec.axes_dims_rope,
        rope_theta=spec.rope_theta,
        guidance_embeds=False,
        layer_norm_eps=spec.layer_norm_eps,
        rms_norm_eps=spec.rms_norm_eps,
    )
    raw = _load_safetensors(packed_dir / "diffusion_pytorch_model.safetensors")
    try:
        transformer.time_guidance_embed.linear_1.weight = raw[
            "time_guidance_embed.timestep_embedder.linear_1.weight"
        ].astype(mx.bfloat16)
        transformer.time_guidance_embed.linear_2.weight = raw[
            "time_guidance_embed.timestep_embedder.linear_2.weight"
        ].astype(mx.bfloat16)
    except KeyError as exc:
        raise BonsaiCheckpointError(
            f"Bonsai Image transformer is missing required timestep weights: {exc}"
        ) from exc
    return transformer


class _TiledFlux2VAE(Flux2VAE):
    """Use the upstream Bonsai 128px tiled decode without changing mflux globally."""

    _bonsai_tiling = TilingConfig(vae_decode_tile_size=128, vae_decode_overlap=8)

    def decode_packed_latents(
        self,
        packed_latents: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        return cast(
            mx.array,
            super().decode_packed_latents(
                packed_latents,
                tiling_config=tiling_config or self._bonsai_tiling,
            ),
        )


class BonsaiImage(Flux2Klein):
    """FLUX.2 Klein pipeline backed by Bonsai's fixed prepacked MLX weights."""

    def __init__(self, model_path: str | Path):
        nn.Module.__init__(self)
        self._root = Path(model_path).expanduser().resolve()
        _validate_checkpoint(self._root)
        self.model_config = ModelConfig.flux2_klein_4b()
        self.callbacks = CallbackRegistry()
        self.prompt_cache: dict[str, tuple[mx.array, mx.array]] = {}
        self.tiling_config = _TiledFlux2VAE._bonsai_tiling
        self.tokenizers = TokenizerLoader.load_all(
            definitions=[_tokenizer_definition()],
            model_path=str(self._root),
        )
        self.text_encoder = _load_text_encoder(
            self._root,
            self.model_config.text_encoder_overrides,
        )
        self.vae = _TiledFlux2VAE()
        vae_weights = WeightLoader.load(
            weight_definition=_VAEWeightDefinition,
            model_path=str(self._root),
        )
        WeightApplier.apply_and_quantize(
            weights=vae_weights,
            quantize_arg=None,
            weight_definition=_VAEWeightDefinition,
            models={"vae": self.vae},
        )
        self.transformer = _load_transformer(self._root)
        self.bits = 2
        self.lora_paths = None
        self.lora_scales = None

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        cached = self.prompt_cache.get(prompt)
        if cached is None:
            if self.text_encoder is None:
                self.text_encoder = _load_text_encoder(
                    self._root,
                    self.model_config.text_encoder_overrides,
                )
            prompt_embeds, text_ids, _, _ = super()._encode_prompt_pair(
                prompt=prompt,
                negative_prompt=None,
                guidance=BONSAI_IMAGE_GUIDANCE,
            )
            mx.eval(prompt_embeds, text_ids)
            self.prompt_cache.clear()
            self.prompt_cache[prompt] = (prompt_embeds, text_ids)
            cached = (prompt_embeds, text_ids)
            self.text_encoder = None
            gc.collect()
            mx.clear_cache()
        return cached[0], cached[1], None, None


__all__ = [
    "BONSAI_IMAGE_GUIDANCE",
    "BONSAI_IMAGE_MAX_PROMPT_TOKENS",
    "BONSAI_IMAGE_REPO",
    "BONSAI_IMAGE_REVISION",
    "BONSAI_IMAGE_STEPS",
    "BonsaiCheckpointError",
    "BonsaiImage",
]
