# reference: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

# fmt: off

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from transformers import T5Config

from .clip import CLIPTextModel
from .config import (
    FLUX_SCHNELL,
    AutoencoderConfig,
    CLIPTextModelConfig,
    SD3_2b,
    SD3_8b,
    VAEDecoderConfig,
    VAEEncoderConfig,
)
from .mmdit import MMDiT
from .t5 import SD3T5Encoder
from .tokenizer import T5Tokenizer, Tokenizer
from .vae import Autoencoder, VAEDecoder, VAEEncoder

# import argmaxtools.mlx.utils as axu


RANK = 32
_DEFAULT_MMDIT = "argmaxinc/mlx-stable-diffusion-3-medium"
_MMDIT = {
    "argmaxinc/mlx-stable-diffusion-3-medium": {
        "argmaxinc/mlx-stable-diffusion-3-medium": "sd3_medium.safetensors",
        "vae": "sd3_medium.safetensors",
    },
    "argmaxinc/mlx-FLUX.1-schnell": {
        "argmaxinc/mlx-FLUX.1-schnell": "flux-schnell.safetensors",
        "vae": "ae.safetensors",
    },
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": {
        "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": "flux-schnell-4bit-quantized.safetensors",
        "vae": "ae.safetensors",
    },
    "argmaxinc/mlx-FLUX.1-dev": {
        "argmaxinc/mlx-FLUX.1-dev": "flux1-dev.safetensors",
        "vae": "ae.safetensors",
    },
    "argmaxinc/mlx-stable-diffusion-3.5-large": {
        "argmaxinc/mlx-stable-diffusion-3.5-large": "sd3.5_large.safetensors",
        "vae": "sd3.5_large.safetensors",
    },
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": {
        "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": "sd3.5_large_4bit_quantized.safetensors",
        "vae": "sd3.5_large_4bit_quantized.safetensors",
    },
}
_DEFAULT_MODEL = "argmaxinc/stable-diffusion"
_MODELS = {
    "argmaxinc/stable-diffusion": {
        "clip_l_config": "clip_l/config.json",
        "clip_l": "clip_l/model.fp16.safetensors",
        "clip_g_config": "clip_g/config.json",
        "clip_g": "clip_g/model.fp16.safetensors",
        "tokenizer_l_vocab": "tokenizer_l/vocab.json",
        "tokenizer_l_merges": "tokenizer_l/merges.txt",
        "tokenizer_g_vocab": "tokenizer_g/vocab.json",
        "tokenizer_g_merges": "tokenizer_g/merges.txt",
        "t5": "t5/t5xxl.safetensors",
    },
}

_PREFIX = {
    "argmaxinc/mlx-stable-diffusion-3-medium": {
        "vae_encoder": "first_stage_model.encoder.",
        "vae_decoder": "first_stage_model.decoder.",
    },
    "argmaxinc/mlx-FLUX.1-schnell": {
        "vae_encoder": "encoder.",
        "vae_decoder": "decoder.",
    },
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": {
        "vae_encoder": "encoder.",
        "vae_decoder": "decoder.",
    },
    "argmaxinc/mlx-FLUX.1-dev": {
        "vae_encoder": "encoder.",
        "vae_decoder": "decoder.",
    },
    "argmaxinc/mlx-stable-diffusion-3.5-large": {
        "vae_encoder": "first_stage_model.encoder.",
        "vae_decoder": "first_stage_model.decoder.",
    },
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": {
        "vae_encoder": "first_stage_model.encoder.",
        "vae_decoder": "first_stage_model.decoder.",
    },
}

_CONFIG = {
    "argmaxinc/mlx-stable-diffusion-3-medium": SD3_2b,
    "argmaxinc/mlx-FLUX.1-schnell": FLUX_SCHNELL,
    "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": FLUX_SCHNELL,
    "argmaxinc/mlx-FLUX.1-dev": FLUX_SCHNELL,
    "argmaxinc/mlx-stable-diffusion-3.5-large": SD3_8b,
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": SD3_8b,
}

_FLOAT16 = mx.bfloat16

DEPTH = {
    "argmaxinc/mlx-stable-diffusion-3-medium": 24,
    "argmaxinc/mlx-stable-diffusion-3.5-large": 38,
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": 38,
}
MAX_LATENT_RESOLUTION = {
    "argmaxinc/mlx-stable-diffusion-3-medium": 96,
    "argmaxinc/mlx-stable-diffusion-3.5-large": 192,
    "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": 192,
}

LOCAl_SD3_CKPT = None
_ASSET_ROOTS: dict[str, Path] = {}
_T5_TOKENIZER_ROOT: Path | None = None


def configure_asset_roots(
    roots: dict[str, str | Path], *, t5_tokenizer_root: str | Path
) -> None:
    global _ASSET_ROOTS, _T5_TOKENIZER_ROOT
    _ASSET_ROOTS = {repo_id: Path(root).resolve() for repo_id, root in roots.items()}
    _T5_TOKENIZER_ROOT = Path(t5_tokenizer_root).resolve()


def _asset_path(repo_id: str, filename: str) -> str:
    root = _ASSET_ROOTS.get(repo_id)
    if root is None:
        raise ValueError(f"No reviewed local asset root configured for {repo_id!r}.")
    candidate = root.joinpath(*filename.split("/"))
    allowed_root = root.parents[1].resolve() if root.parent.name == "snapshots" else root
    try:
        path = candidate.resolve(strict=True)
    except OSError:
        raise FileNotFoundError(
            f"Required SD3.5 asset is missing: {filename}"
        ) from None
    if (
        not candidate.is_relative_to(root)
        or not path.is_relative_to(allowed_root)
        or not path.is_file()
    ):
        raise FileNotFoundError(f"Required SD3.5 asset is missing: {filename}")
    return str(path)


def flux_state_dict_adjustments(state_dict, prefix="", hidden_size=3072, mlp_ratio=4):
    state_dict = {
        k.replace("double_blocks", "multimodal_transformer_blocks"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("single_blocks", "unified_transformer_blocks"): v
        for k, v in state_dict.items()
    }

    # Split qkv proj and rename:
    # *transformer_block.attn.qkv.{weigth/bias}  -> transformer_block.attn.{q/k/v}_proj.{weigth/bias}
    # *transformer_block.attn.proj.{weigth/bias} -> transformer_block.attn.o_proj.{weight/bias}
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if "attn.qkv" in k:
            keys_to_pop.append(k)
            for name, weight in zip(["q", "k", "v"], mx.split(state_dict[k], 3)):
                state_dict_update[k.replace("attn.qkv", f"attn.{name}_proj")] = (
                    weight if "weight" in k else weight
                )

    [state_dict.pop(k) for k in keys_to_pop]
    state_dict.update(state_dict_update)

    state_dict = {
        k.replace("txt_attn", "text_transformer_block.attn"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("img_attn", "image_transformer_block.attn"): v
        for k, v in state_dict.items()
    }

    state_dict = {
        k.replace("txt_mlp.0", "text_transformer_block.mlp.fc1"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("txt_mlp.2", "text_transformer_block.mlp.fc2"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("img_mlp.0", "image_transformer_block.mlp.fc1"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("img_mlp.2", "image_transformer_block.mlp.fc2"): v
        for k, v in state_dict.items()
    }

    state_dict = {
        k.replace("img_mod.lin", "image_transformer_block.adaLN_modulation.layers.1"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("txt_mod.lin", "text_transformer_block.adaLN_modulation.layers.1"): v
        for k, v in state_dict.items()
    }

    state_dict = {k.replace(".proj", ".o_proj"): v for k, v in state_dict.items()}

    state_dict = {
        k.replace(".attn.norm.key_norm.scale", ".qk_norm.k_norm.weight"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".attn.norm.query_norm.scale", ".qk_norm.q_norm.weight"): v
        for k, v in state_dict.items()
    }

    state_dict = {
        k.replace(".modulation.lin", ".transformer_block.adaLN_modulation.layers.1"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".norm.key_norm.scale", ".transformer_block.qk_norm.k_norm.weight"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(
            ".norm.query_norm.scale", ".transformer_block.qk_norm.q_norm.weight"
        ): v
        for k, v in state_dict.items()
    }

    # Split qkv proj and mlp in unified transformer block and rename:
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if ".linear1" in k:
            keys_to_pop.append(k)
            for name, weight in zip(
                ["attn.q", "attn.k", "attn.v", "mlp.fc1"],
                mx.split(
                    state_dict[k],
                    [
                        hidden_size,
                        2 * hidden_size,
                        3 * hidden_size,
                        (3 + mlp_ratio) * hidden_size,
                    ],
                ),
            ):
                if name == "mlp.fc1":
                    state_dict_update[
                        k.replace(".linear1", f".transformer_block.{name}")
                    ] = (weight if "weight" in k else weight)
                else:
                    state_dict_update[
                        k.replace(".linear1", f".transformer_block.{name}_proj")
                    ] = (weight if "weight" in k else weight)

    [state_dict.pop(k) for k in keys_to_pop]
    state_dict.update(state_dict_update)

    # Split o_proj and mlp in unified transformer block and rename:
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if ".linear2" in k:
            keys_to_pop.append(k)
            if "bias" in k:
                state_dict_update[
                    k.replace(".linear2", ".transformer_block.attn.o_proj")
                ] = state_dict[k]
                state_dict_update[
                    k.replace(".linear2", ".transformer_block.mlp.fc2")
                ] = state_dict[k]
            else:
                for name, weight in zip(
                    ["attn.o", "mlp.fc2"],
                    mx.split(
                        state_dict[k],
                        [hidden_size, (1 + mlp_ratio) * hidden_size],
                        axis=1,
                    ),
                ):
                    if name == "mlp.fc2":
                        state_dict_update[
                            k.replace(".linear2", f".transformer_block.{name}")
                        ] = (weight if "weight" in k else weight)
                    else:
                        state_dict_update[
                            k.replace(".linear2", f".transformer_block.{name}_proj")
                        ] = (weight if "weight" in k else weight)

    [state_dict.pop(k) for k in keys_to_pop]
    state_dict.update(state_dict_update)

    state_dict = {
        k.replace("img_in.", "x_embedder.proj."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("txt_in.", "context_embedder."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("time_in.", "t_embedder."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("vector_in.", "y_embedder."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".in_layer.", ".mlp.layers.0."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".out_layer.", ".mlp.layers.2."): v for k, v in state_dict.items()
    }

    state_dict = {
        k.replace(
            "final_layer.adaLN_modulation.1", "final_layer.adaLN_modulation.layers.1"
        ): v
        for k, v in state_dict.items()
    }

    state_dict["x_embedder.proj.weight"] = mx.expand_dims(
        mx.expand_dims(state_dict["x_embedder.proj.weight"], axis=1), axis=1
    )

    return state_dict


def mmdit_state_dict_adjustments(state_dict, prefix=""):
    # Remove prefix
    state_dict = {k.lstrip(prefix): v for k, v in state_dict.items()}

    state_dict = {
        k.replace("y_embedder.mlp", "y_embedder.mlp.layers"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("t_embedder.mlp", "t_embedder.mlp.layers"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("adaLN_modulation", "adaLN_modulation.layers"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("al_layer", "final_layer"): v for k, v in state_dict.items()
    }

    # Rename joint_blocks -> multimodal_transformer_blocks
    state_dict = {
        k.replace("joint_blocks", "multimodal_transformer_blocks"): v
        for k, v in state_dict.items()
    }

    # Remap context_block -> text_block
    state_dict = {
        k.replace("context_block", "text_transformer_block"): v
        for k, v in state_dict.items()
    }

    # Remap x_block -> image_block
    state_dict = {
        k.replace("x_block", "image_transformer_block"): v
        for k, v in state_dict.items()
    }

    # Remap qk_norm
    state_dict = {
        k.replace(".attn.ln_q.", ".qk_norm.q_norm."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".attn.ln_k.", ".qk_norm.k_norm."): v for k, v in state_dict.items()
    }

    # Split qkv proj and rename:
    # *transformer_block.attn.qkv.{weigth/bias}  -> transformer_block.attn.{q/k/v}_proj.{weigth/bias}
    # *transformer_block.attn.proj.{weigth/bias} -> transformer_block.attn.o_proj.{weight/bias}
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if "attn.qkv" in k:
            keys_to_pop.append(k)
            for name, weight in zip(["q", "k", "v"], mx.split(state_dict[k], 3)):
                state_dict_update[k.replace("attn.qkv", f"attn.{name}_proj")] = (
                    weight if "weight" in k else weight
                )

    [state_dict.pop(k) for k in keys_to_pop]
    state_dict.update(state_dict_update)

    state_dict = {
        k.replace("attn.proj", "attn.o_proj"): (
            v if "attn.proj" in k and "weight" in k else v
        )
        for k, v in state_dict.items()
    }

    # Filter out VAE Decoder related tensors
    state_dict = {k: v for k, v in state_dict.items() if "decoder." not in k}

    # Filter out VAE Encoder related tensors
    state_dict = {k: v for k, v in state_dict.items() if "encoder." not in k}

    # Filter out k_proj.bias related tensors
    state_dict = {k: v for k, v in state_dict.items() if "k_proj.bias" not in k}

    # Filter out teacher_model related tensors
    state_dict = {k: v for k, v in state_dict.items() if "teacher_model." not in k}

    # Remap pos_embed buffer -> nn.Embedding
    state_dict = {
        k.replace("pos_embed", "x_pos_embedder.pos_embed.weight"): (
            v[0] if "pos_embed" in k else v
        )
        for k, v in state_dict.items()
    }

    # Transpose x_embedder.proj.weight
    state_dict["x_embedder.proj.weight"] = state_dict[
        "x_embedder.proj.weight"
    ].transpose(0, 2, 3, 1)

    return state_dict


def vae_decoder_state_dict_adjustments(state_dict, prefix="decoder."):
    # Keep only the keys that have the prefix
    state_dict = {k: v for k, v in state_dict.items() if prefix in k}
    state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}

    # Filter out MMDIT related tensors
    state_dict = {k: v for k, v in state_dict.items() if "diffusion_model." not in k}

    state_dict = {k.replace("up", "up_blocks"): v for k, v in state_dict.items()}
    state_dict = {k.replace("mid", "mid_blocks"): v for k, v in state_dict.items()}

    state_dict = {
        k.replace("mid_blocks.block_1", "mid_blocks.0"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("mid_blocks.block_2", "mid_blocks.2"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("mid_blocks.attn_1", "mid_blocks.1"): v for k, v in state_dict.items()
    }

    state_dict = {k.replace(".norm.", ".group_norm."): v for k, v in state_dict.items()}

    state_dict = {k.replace(".q", ".query_proj"): v for k, v in state_dict.items()}
    state_dict = {k.replace(".k", ".key_proj"): v for k, v in state_dict.items()}
    state_dict = {k.replace(".v", ".value_proj"): v for k, v in state_dict.items()}
    state_dict = {k.replace(".proj_out", ".out_proj"): v for k, v in state_dict.items()}

    state_dict = {k.replace(".block.", ".resnets."): v for k, v in state_dict.items()}
    state_dict = {
        k.replace(".nin_shortcut.", ".conv_shortcut."): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".up_blockssample.conv.", ".upsample."): v
        for k, v in state_dict.items()
    }

    state_dict = {
        k.replace("norm_out", "conv_norm_out"): v for k, v in state_dict.items()
    }

    # reshape weights

    state_dict = {
        k: v.transpose(0, 2, 3, 1) if "upsample" in k and "weight" in k else v
        for k, v in state_dict.items()
    }
    state_dict = {
        k: (
            v.transpose(0, 2, 3, 1)
            if "resnets" in k and "conv" in k and "weight" in k
            else v
        )
        for k, v in state_dict.items()
    }
    state_dict = {
        k: (
            v.transpose(0, 2, 3, 1)
            if "mid_blocks" in k and "conv" in k and "weight" in k
            else v
        )
        for k, v in state_dict.items()
    }
    state_dict = {
        k: v[:, 0, 0, :] if "conv_shortcut.weight" in k else v
        for k, v in state_dict.items()
    }
    state_dict = {
        k: v[:, :, 0, 0] if "proj.weight" in k else v for k, v in state_dict.items()
    }
    state_dict["conv_in.weight"] = state_dict["conv_in.weight"].transpose(0, 2, 3, 1)
    state_dict["conv_out.weight"] = state_dict["conv_out.weight"].transpose(0, 2, 3, 1)

    return state_dict


def vae_encoder_state_dict_adjustments(state_dict, prefix="encoder."):
    # Keep only the keys that have the prefix
    state_dict = {k: v for k, v in state_dict.items() if prefix in k}
    state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}

    # Filter out MMDIT related tensors
    state_dict = {k: v for k, v in state_dict.items() if "diffusion_model." not in k}

    state_dict = {k.replace("down.", "down_blocks."): v for k, v in state_dict.items()}
    state_dict = {
        k.replace(".downsample.conv.", ".downsample."): v for k, v in state_dict.items()
    }
    state_dict = {k.replace(".block.", ".resnets."): v for k, v in state_dict.items()}
    state_dict = {
        k.replace(".nin_shortcut.", ".conv_shortcut."): v for k, v in state_dict.items()
    }

    state_dict = {k.replace(".q", ".query_proj"): v for k, v in state_dict.items()}
    state_dict = {k.replace(".k", ".key_proj"): v for k, v in state_dict.items()}
    state_dict = {k.replace(".v", ".value_proj"): v for k, v in state_dict.items()}
    state_dict = {k.replace(".proj_out", ".out_proj"): v for k, v in state_dict.items()}

    state_dict = {k.replace("mid", "mid_blocks"): v for k, v in state_dict.items()}

    state_dict = {
        k.replace("mid_blocks.block_1", "mid_blocks.0"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("mid_blocks.block_2", "mid_blocks.2"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("mid_blocks.attn_1", "mid_blocks.1"): v for k, v in state_dict.items()
    }

    state_dict = {k.replace(".norm.", ".group_norm."): v for k, v in state_dict.items()}
    state_dict = {
        k.replace("norm_out", "conv_norm_out"): v for k, v in state_dict.items()
    }

    # reshape weights

    state_dict = {
        k: v.transpose(0, 2, 3, 1) if "downsample" in k and "weight" in k else v
        for k, v in state_dict.items()
    }
    state_dict = {
        k: (
            v.transpose(0, 2, 3, 1)
            if "resnets" in k and "conv" in k and "weight" in k
            else v
        )
        for k, v in state_dict.items()
    }
    state_dict = {
        k: (
            v.transpose(0, 2, 3, 1)
            if "mid_blocks" in k and "conv" in k and "weight" in k
            else v
        )
        for k, v in state_dict.items()
    }
    state_dict = {
        k: v[:, 0, 0, :] if "conv_shortcut.weight" in k else v
        for k, v in state_dict.items()
    }
    state_dict = {
        k: v[:, :, 0, 0] if "proj.weight" in k else v for k, v in state_dict.items()
    }
    state_dict["conv_in.weight"] = state_dict["conv_in.weight"].transpose(0, 2, 3, 1)
    state_dict["conv_out.weight"] = state_dict["conv_out.weight"].transpose(0, 2, 3, 1)

    return state_dict


def t5_encoder_state_dict_adjustments(state_dict, prefix=""):
    state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items()}

    for i in range(2):
        state_dict = {
            k.replace(f"layer.{i}.layer_norm", f"ln{i+1}"): v
            for k, v in state_dict.items()
        }
    for i in range(2):
        state_dict = {k.replace(f"layer.{i}.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("block", "layers"): v for k, v in state_dict.items()}
    state_dict = {
        k.replace("SelfAttention.q", "attention.query_proj"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("SelfAttention.k", "attention.key_proj"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("SelfAttention.v", "attention.value_proj"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("SelfAttention.o", "attention.out_proj"): v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("DenseReluDense", "dense"): v for k, v in state_dict.items()
    }

    state_dict["encoder.relative_attention_bias.embeddings.weight"] = state_dict[
        "encoder.layers.0.SelfAttention.relative_attention_bias.weight"
    ]
    del state_dict["encoder.layers.0.SelfAttention.relative_attention_bias.weight"]

    state_dict["wte.weight"] = state_dict["encoder.embed_tokens.weight"]
    del state_dict["encoder.embed_tokens.weight"]
    del state_dict["shared.weight"]

    state_dict["encoder.ln.weight"] = state_dict["encoder.final_layer_norm.weight"]
    del state_dict["encoder.final_layer_norm.weight"]

    return state_dict


def map_clip_text_encoder_weights(key, value):
    # Remove prefixes
    if key.startswith("text_model."):
        key = key[11:]
    if key.startswith("embeddings."):
        key = key[11:]
    if key.startswith("encoder."):
        key = key[8:]

    # Map attention layers
    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")

    # Map ffn layers
    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")

    return [(key, value)]


def map_vae_weights(key, value):
    # Map up/downsampling
    if "downsamplers" in key:
        key = key.replace("downsamplers.0.conv", "downsample")
    if "upsamplers" in key:
        key = key.replace("upsamplers.0.conv", "upsample")

    # Map attention layers
    if "to_k" in key:
        key = key.replace("to_k", "key_proj")
    if "to_out.0" in key:
        key = key.replace("to_out.0", "out_proj")
    if "to_q" in key:
        key = key.replace("to_q", "query_proj")
    if "to_v" in key:
        key = key.replace("to_v", "value_proj")

    # Map the mid block
    if "mid_block.resnets.0" in key:
        key = key.replace("mid_block.resnets.0", "mid_blocks.0")
    if "mid_block.attentions.0" in key:
        key = key.replace("mid_block.attentions.0", "mid_blocks.1")
    if "mid_block.resnets.1" in key:
        key = key.replace("mid_block.resnets.1", "mid_blocks.2")

    # Map the quant/post_quant layers
    if "quant_conv" in key:
        key = key.replace("quant_conv", "quant_proj")
        value = value.squeeze()

    # Map the conv_shortcut to linear
    if "conv_shortcut.weight" in key:
        value = value.squeeze()

    if len(value.shape) == 4:
        value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape)

    return [(key, value)]


""" Code obtained from
https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/model_io.py
"""


def _flatten(params):
    return [(k, v) for p in params for (k, v) in p]


""" Code obtained from
https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/model_io.py
"""


def _load_safetensor_weights(mapper, model, weight_file, float16: bool = False):
    dtype = _FLOAT16 if float16 else mx.float32
    weights = mx.load(weight_file)
    weights = _flatten([mapper(k, v.astype(dtype)) for k, v in weights.items()])
    model.update(tree_unflatten(weights))


def _check_key(key: str, part: str):
    if key not in _MODELS:
        raise ValueError(
            f"[{part}] '{key}' model not found, choose one of {{{','.join(_MODELS.keys())}}}"
        )


def load_mmdit(
    key: str = _DEFAULT_MMDIT,
    float16: bool = False,
    model_key: str = "mmdit_2b",
    low_memory_mode: bool = True,
    only_modulation_dict: bool = False,
):
    """Load the MM-DiT model from the checkpoint file."""
    """only_modulation_dict: Only returns the modulation dictionary"""
    dtype = _FLOAT16 if float16 else mx.float32
    config = _CONFIG[key]
    config.low_memory_mode = low_memory_mode
    model = MMDiT(config)

    mmdit_weights = _MMDIT[key][model_key]
    mmdit_weights_ckpt = LOCAl_SD3_CKPT or _asset_path(key, mmdit_weights)
    _asset_path(key, "config.json")
    weights = mx.load(mmdit_weights_ckpt)
    prefix = "model.diffusion_model."

    if key != "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized":
        weights = mmdit_state_dict_adjustments(weights, prefix=prefix)
    else:
        nn.quantize(
            model, class_predicate=lambda _, module: isinstance(module, nn.Linear)
        )
        weights = {k.replace(prefix, ""): v for k, v in weights.items() if prefix in k}

    weights = {
        k: v.astype(dtype) if v.dtype != mx.uint32 else v for k, v in weights.items()
    }
    if only_modulation_dict:
        weights = {k: v for k, v in weights.items() if "adaLN" in k}
        return tree_flatten(weights)
    model.load_weights(list(weights.items()))

    return model


def load_flux(
    key: str = "argmaxinc/mlx-FLUX.1-schnell",
    float16: bool = False,
    model_key: str = "argmaxinc/mlx-FLUX.1-schnell",
    low_memory_mode: bool = True,
    only_modulation_dict: bool = False,
):
    """Load the MM-DiT Flux model from the checkpoint file."""
    dtype = _FLOAT16 if float16 else mx.float32
    config = FLUX_SCHNELL
    config.low_memory_mode = low_memory_mode
    model = MMDiT(config)

    flux_weights = _MMDIT[key][model_key]
    flux_weights_ckpt = LOCAl_SD3_CKPT or _asset_path(key, flux_weights)
    _asset_path(key, "config.json")
    weights = mx.load(flux_weights_ckpt)

    if model_key in ["argmaxinc/mlx-FLUX.1-schnell", "argmaxinc/mlx-FLUX.1-dev"]:
        weights = flux_state_dict_adjustments(
            weights,
            prefix="",
            hidden_size=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
        )
    elif (
        model_key == "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized"
    ):  # 4-bit ckpt already adjusted
        nn.quantize(model)

    weights = {
        k: v.astype(dtype) if v.dtype != mx.uint32 else v for k, v in weights.items()
    }
    if only_modulation_dict:
        weights = {k: v for k, v in weights.items() if "adaLN" in k}
        return tree_flatten(weights)
    model.update(tree_unflatten(tree_flatten(weights)))

    return model


def load_text_encoder(
    key: str = _DEFAULT_MODEL,
    float16: bool = False,
    model_key: str = "text_encoder",
    config_key: Optional[str] = None,
):
    """Load the stable diffusion text encoder from Hugging Face Hub."""
    _check_key(key, "load_text_encoder")

    config_key = config_key or (model_key + "_config")

    # Download the config and create the model
    text_encoder_config = _MODELS[key][config_key]
    with open(_asset_path(key, text_encoder_config)) as f:
        config = json.load(f)

    with_projection = "WithProjection" in config["architectures"][0]

    model = CLIPTextModel(
        CLIPTextModelConfig(
            num_layers=config["num_hidden_layers"],
            model_dims=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_length=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
            projection_dim=config["projection_dim"] if with_projection else None,
            hidden_act=config.get("hidden_act", "quick_gelu"),
        )
    )

    # Download the weights and map them into the model
    text_encoder_weights = _MODELS[key][model_key]
    weight_file = _asset_path(key, text_encoder_weights)
    _load_safetensor_weights(map_clip_text_encoder_weights, model, weight_file, float16)

    return model


def load_autoencoder(key: str = _DEFAULT_MODEL, float16: bool = False):
    """Load the stable diffusion autoencoder from Hugging Face Hub."""
    _check_key(key, "load_autoencoder")

    # Download the config and create the model
    vae_config = _MODELS[key]["vae_config"]
    with open(_asset_path(key, vae_config)) as f:
        config = json.load(f)

    config["latent_channels"] = 16

    model = Autoencoder(
        AutoencoderConfig(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            latent_channels_out=2 * config["latent_channels"],
            latent_channels_in=config["latent_channels"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            scaling_factor=config.get("scaling_factor", 0.18215),
        )
    )

    # Download the weights and map them into the model
    vae_weights = _MODELS[key]["vae"]
    weight_file = _asset_path(key, vae_weights)
    _load_safetensor_weights(map_vae_weights, model, weight_file, float16)

    return model


def load_vae_decoder(
    key: str = _DEFAULT_MMDIT,
    float16: bool = False,
    model_key: str = "vae",
):
    """Load the SD3 VAE Decoder model from the checkpoint file."""
    config = VAEDecoderConfig()
    model = VAEDecoder(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        resnet_groups=config.resnet_groups,
    )

    dtype = _FLOAT16 if float16 else mx.float32
    vae_weights = _MMDIT[key][model_key]
    vae_weights_ckpt = LOCAl_SD3_CKPT or _asset_path(key, vae_weights)
    weights = mx.load(vae_weights_ckpt)
    prefix = _PREFIX[key]["vae_decoder"]

    if key != "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized":
        weights = vae_decoder_state_dict_adjustments(weights, prefix=prefix)
    else:
        weights = {k.replace(prefix, ""): v for k, v in weights.items() if prefix in k}

    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()))

    return model


def load_vae_encoder(
    key: str = _DEFAULT_MMDIT,
    float16: bool = False,
    model_key: str = "vae",
):
    """Load the SD3 VAE Encoder model from the checkpoint file."""
    config = VAEEncoderConfig()
    model = VAEEncoder(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        resnet_groups=config.resnet_groups,
    )

    dtype = _FLOAT16 if float16 else mx.float32
    vae_weights = _MMDIT[key][model_key]
    vae_weights_ckpt = LOCAl_SD3_CKPT or _asset_path(key, vae_weights)
    weights = mx.load(vae_weights_ckpt)
    prefix = _PREFIX[key]["vae_encoder"]

    if key != "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized":
        weights = vae_encoder_state_dict_adjustments(weights, prefix=prefix)
    else:
        weights = {k.replace(prefix, ""): v for k, v in weights.items() if prefix in k}

    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()))

    return model


def load_t5_encoder(
    key: str = _DEFAULT_MODEL,
    float16: bool = False,
    model_key: str = "t5",
    low_memory_mode: bool = True,
):
    if _T5_TOKENIZER_ROOT is None:
        raise ValueError("No reviewed local T5 tokenizer root configured.")
    config = T5Config.from_pretrained(_T5_TOKENIZER_ROOT, local_files_only=True)
    model = SD3T5Encoder(config, low_memory_mode=low_memory_mode)

    dtype = _FLOAT16 if float16 else mx.float32
    t5_weights = _MODELS[key][model_key]
    weights = mx.load(_asset_path(key, t5_weights))
    weights = t5_encoder_state_dict_adjustments(weights, prefix="")
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.update(tree_unflatten(tree_flatten(weights)))

    return model


def load_tokenizer(
    key: str = _DEFAULT_MODEL,
    vocab_key: str = "tokenizer_vocab",
    merges_key: str = "tokenizer_merges",
    pad_with_eos: bool = False,
):
    _check_key(key, "load_tokenizer")

    vocab_file = _asset_path(key, _MODELS[key][vocab_key])
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)

    merges_file = _asset_path(key, _MODELS[key][merges_key])
    with open(merges_file, encoding="utf-8") as f:
        bpe_merges = f.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
    bpe_merges = [tuple(m.split()) for m in bpe_merges]
    bpe_ranks = dict(map(reversed, enumerate(bpe_merges)))

    return Tokenizer(bpe_ranks, vocab, pad_with_eos)


def load_t5_tokenizer(max_context_length: int = 256):
    if _T5_TOKENIZER_ROOT is None:
        raise ValueError("No reviewed local T5 tokenizer root configured.")
    config = T5Config.from_pretrained(_T5_TOKENIZER_ROOT, local_files_only=True)
    return T5Tokenizer(config, max_context_length, _T5_TOKENIZER_ROOT)

# fmt: on
