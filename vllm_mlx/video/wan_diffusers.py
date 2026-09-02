# SPDX-License-Identifier: Apache-2.0
"""Read the official Wan 2.1 Diffusers layout directly with MLX.

The Desktop catalog downloads one pinned repository.  This adapter maps its
sharded safetensors into the existing ``mlx-video-with-audio`` model classes
without PyTorch and without materializing a second converted checkpoint.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType, ModuleType

from .wan import WanBackendError

_TRANSFORMER_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
_T5_INDEX = "text_encoder/model.safetensors.index.json"
_VAE_FILE = "vae/diffusion_pytorch_model.safetensors"
_EXPECTED_TRANSFORMER_KEYS = 825
_EXPECTED_T5_KEYS = 242
_EXPECTED_VAE_DECODER_KEYS = 108

_WAN21_COMPONENT_CONTRACTS: dict[str, dict[str, object]] = {
    "model_index.json": {
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    },
    "transformer/config.json": {
        "_class_name": "WanTransformer3DModel",
        "patch_size": [1, 2, 2],
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "num_layers": 30,
        "ffn_dim": 8960,
        "text_dim": 4096,
    },
    "text_encoder/config.json": {
        "model_type": "umt5",
        "vocab_size": 256384,
        "d_model": 4096,
        "d_ff": 10240,
        "num_heads": 64,
        "num_layers": 24,
        "relative_attention_num_buckets": 32,
    },
    "vae/config.json": {
        "_class_name": "AutoencoderKLWan",
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
    },
}


def _index_shards(root: Path, relative: str, expected: int) -> set[Path] | None:
    """Return the shard paths an index references, or None when malformed."""
    try:
        payload = json.loads((root / relative).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != expected:
        return None
    directory = (root / relative).parent
    shards: set[Path] = set()
    for source, filename in weight_map.items():
        if (
            not isinstance(source, str)
            or not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            return None
        shards.add(directory / filename)
    return shards


_TOKENIZER_FILES = (
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def _is_regular_nonempty_file(path: Path) -> bool:
    """True only for an existing regular file holding at least one byte."""
    return path.is_file() and path.stat().st_size > 0


def is_diffusers_wan21_layout(root: Path) -> bool:
    """Decide routing for the pinned Desktop layout, rejecting malformed trees.

    Routing must fail closed: a missing or empty tokenizer artifact, a
    malformed or wrong-cardinality safetensors index, an unsafe shard name, or
    a missing, irregular, or zero-byte weight file (the VAE or any referenced
    shard) all mean this is not the audited layout. The tokenizer artifacts
    mirror the download gate's pin for Wan-AI/Wan2.1-T2V-1.3B-Diffusers.
    """
    try:
        if not _is_regular_nonempty_file(root / _VAE_FILE):
            return False
        for relative in _TOKENIZER_FILES:
            if not _is_regular_nonempty_file(root / relative):
                return False
        for relative, expected in (
            (_TRANSFORMER_INDEX, _EXPECTED_TRANSFORMER_KEYS),
            (_T5_INDEX, _EXPECTED_T5_KEYS),
        ):
            shards = _index_shards(root, relative, expected)
            if shards is None or not all(
                _is_regular_nonempty_file(shard) for shard in shards
            ):
                return False
        for relative, expected_contract in _WAN21_COMPONENT_CONTRACTS.items():
            payload = json.loads((root / relative).read_text())
            if not isinstance(payload, dict) or any(
                payload.get(key) != value for key, value in expected_contract.items()
            ):
                return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return True


# Wan-specific pipeline/component classes pinned by the download gate; the
# generic UMT5 text encoder is deliberately excluded so a plain T5 checkpoint
# is never mistaken for the Wan 2.1 layout.
_WAN21_IDENTITY_MARKERS: tuple[tuple[str, object], ...] = tuple(
    (relative, contract["_class_name"])
    for relative, contract in _WAN21_COMPONENT_CONTRACTS.items()
    if "_class_name" in contract
)


def _identifies_as_diffusers_wan21(root: Path) -> bool:
    """Recognize a tree that claims to be the pinned Wan 2.1 Diffusers layout.

    Deliberately independent of ``is_diffusers_wan21_layout``: a damaged copy
    of the audited checkpoint must be recognized so generation raises instead
    of reaching the incompatible preconverted-generator path. Identity requires
    a positively readable pinned component class; a marker that is missing,
    malformed, or lost to a concurrent delete simply does not match, so
    arbitrary directories and true preconverted checkpoints keep their
    existing routing.
    """
    for relative, class_name in _WAN21_IDENTITY_MARKERS:
        try:
            payload = json.loads((root / relative).read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("_class_name") == class_name:
            return True
    return False


def desktop_wan21_config() -> dict[str, object]:
    """Config fields consumed by both Rapid's guards and the MLX runtime."""
    return {
        "model_type": "t2v",
        "model_version": "2.1",
        "dim": 1536,
        "ffn_dim": 8960,
        "num_heads": 12,
        "num_layers": 30,
        "dual_model": False,
        "boundary": 0.0,
        "sample_shift": 5.0,
        "sample_steps": 50,
        "sample_guide_scale": 5.0,
        "sample_fps": 16,
        "max_area": 704 * 1280,
    }


def _transformer_key(key: str) -> str:
    exact = {
        "patch_embedding.weight": "patch_embedding_proj.weight",
        "patch_embedding.bias": "patch_embedding_proj.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding_0.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding_0.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding_1.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding_1.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding_0.weight",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding_0.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding_1.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding_1.bias",
        "condition_embedder.time_proj.weight": "time_projection.weight",
        "condition_embedder.time_proj.bias": "time_projection.bias",
        "scale_shift_table": "head.modulation",
        "proj_out.weight": "head.head.weight",
        "proj_out.bias": "head.head.bias",
    }
    if key in exact:
        return exact[key]
    match = re.fullmatch(r"blocks\.(\d+)\.(.+)", key)
    if match is None:
        raise WanBackendError(f"unsupported Wan 2.1 transformer tensor {key!r}")
    block, tail = match.groups()
    replacements = (
        ("attn1.to_out.0.", "self_attn.o."),
        ("attn1.to_q.", "self_attn.q."),
        ("attn1.to_k.", "self_attn.k."),
        ("attn1.to_v.", "self_attn.v."),
        ("attn1.norm_q.", "self_attn.norm_q."),
        ("attn1.norm_k.", "self_attn.norm_k."),
        ("attn2.to_out.0.", "cross_attn.o."),
        ("attn2.to_q.", "cross_attn.q."),
        ("attn2.to_k.", "cross_attn.k."),
        ("attn2.to_v.", "cross_attn.v."),
        ("attn2.norm_q.", "cross_attn.norm_q."),
        ("attn2.norm_k.", "cross_attn.norm_k."),
        ("ffn.net.0.proj.", "ffn.fc1."),
        ("ffn.net.2.", "ffn.fc2."),
        ("norm2.", "norm3."),
    )
    if tail == "scale_shift_table":
        tail = "modulation"
    else:
        for source, target in replacements:
            if tail.startswith(source):
                tail = target + tail[len(source) :]
                break
        else:
            raise WanBackendError(f"unsupported Wan 2.1 transformer tensor {key!r}")
    return f"blocks.{block}.{tail}"


def _t5_key(key: str) -> str:
    if key == "shared.weight":
        return "token_embedding.weight"
    if key == "encoder.final_layer_norm.weight":
        return "norm.weight"
    match = re.fullmatch(r"encoder\.block\.(\d+)\.layer\.([01])\.(.+)", key)
    if match is None:
        raise WanBackendError(f"unsupported Wan 2.1 text encoder tensor {key!r}")
    block, layer, tail = match.groups()
    mapping = {
        ("0", "layer_norm.weight"): "norm1.weight",
        ("0", "SelfAttention.q.weight"): "attn.q.weight",
        ("0", "SelfAttention.k.weight"): "attn.k.weight",
        ("0", "SelfAttention.v.weight"): "attn.v.weight",
        ("0", "SelfAttention.o.weight"): "attn.o.weight",
        (
            "0",
            "SelfAttention.relative_attention_bias.weight",
        ): "pos_embedding.embedding.weight",
        ("1", "layer_norm.weight"): "norm2.weight",
        ("1", "DenseReluDense.wi_0.weight"): "ffn.gate_proj.weight",
        ("1", "DenseReluDense.wi_1.weight"): "ffn.fc1.weight",
        ("1", "DenseReluDense.wo.weight"): "ffn.fc2.weight",
    }
    mapped = mapping.get((layer, tail))
    if mapped is None:
        raise WanBackendError(f"unsupported Wan 2.1 text encoder tensor {key!r}")
    return f"blocks.{block}.{mapped}"


def _vae_decoder_key(key: str) -> str | None:
    exact = {
        "post_quant_conv.weight": "conv2.weight",
        "post_quant_conv.bias": "conv2.bias",
        "decoder.conv_in.weight": "decoder.conv1.weight",
        "decoder.conv_in.bias": "decoder.conv1.bias",
        "decoder.norm_out.gamma": "decoder.head.0.gamma",
        "decoder.conv_out.weight": "decoder.head.2.weight",
        "decoder.conv_out.bias": "decoder.head.2.bias",
    }
    if key in exact:
        return exact[key]
    if not key.startswith("decoder."):
        return None
    match = re.fullmatch(r"decoder\.mid_block\.resnets\.(\d+)\.(.+)", key)
    if match:
        block, tail = match.groups()
        residual = {
            "norm1.gamma": "residual.0.gamma",
            "conv1.weight": "residual.2.weight",
            "conv1.bias": "residual.2.bias",
            "norm2.gamma": "residual.3.gamma",
            "conv2.weight": "residual.6.weight",
            "conv2.bias": "residual.6.bias",
        }.get(tail)
        if residual:
            return f"decoder.middle.{int(block) * 2}.{residual}"
    match = re.fullmatch(r"decoder\.mid_block\.attentions\.0\.(.+)", key)
    if match:
        return "decoder.middle.1." + match.group(1)
    match = re.fullmatch(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", key)
    if match:
        block, resnet, tail = match.groups()
        old_block = int(block) * 4 + int(resnet)
        residual = {
            "norm1.gamma": "residual.0.gamma",
            "conv1.weight": "residual.2.weight",
            "conv1.bias": "residual.2.bias",
            "norm2.gamma": "residual.3.gamma",
            "conv2.weight": "residual.6.weight",
            "conv2.bias": "residual.6.bias",
            "conv_shortcut.weight": "shortcut.weight",
            "conv_shortcut.bias": "shortcut.bias",
        }.get(tail)
        if residual:
            return f"decoder.upsamples.{old_block}.{residual}"
    match = re.fullmatch(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)", key)
    if match and int(match.group(1)) < 3:
        return f"decoder.upsamples.{int(match.group(1)) * 4 + 3}.{match.group(2)}"
    raise WanBackendError(f"unsupported Wan 2.1 VAE decoder tensor {key!r}")


def _read_index(
    root: Path, relative: str, expected: int, mapper: Callable[[str], str]
) -> dict[str, list[tuple[str, str]]]:
    try:
        payload = json.loads((root / relative).read_text())
        weight_map = payload["weight_map"]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as exc:
        raise WanBackendError("the Wan 2.1 safetensors index is unreadable") from exc
    if not isinstance(weight_map, dict) or len(weight_map) != expected:
        raise WanBackendError(
            "the Wan 2.1 safetensors index has an unexpected tensor set"
        )
    grouped: dict[str, list[tuple[str, str]]] = {}
    mapped_names: set[str] = set()
    for source, filename in weight_map.items():
        if (
            not isinstance(source, str)
            or not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            raise WanBackendError(
                "the Wan 2.1 safetensors index contains an unsafe entry"
            )
        target = mapper(source)
        if target in mapped_names:
            raise WanBackendError(
                "the Wan 2.1 safetensors index maps duplicate tensors"
            )
        mapped_names.add(target)
        grouped.setdefault(filename, []).append((source, target))
    return grouped


def _safetensors_tensor_names(path: Path) -> frozenset[str]:
    """Tensor names from a safetensors container, parsing metadata only.

    ``safe_open`` validates the header without materializing tensor payloads,
    so the probe stays bounded even for multi-GB shards. A malformed header,
    an unreadable file, or a non-safetensors byte stream fails closed.
    """
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(path), framework="numpy") as container:
            return frozenset(container.keys())
    except (OSError, SafetensorError) as exc:
        raise WanBackendError(
            f"the Wan 2.1 checkpoint file {path.name!r} is not a readable "
            "safetensors container"
        ) from exc


def validate_wan21_checkpoint_artifacts(root: Path) -> bool:
    """Metadata-only proof that the pinned checkpoint artifacts are loadable.

    The layout probe (:func:`is_diffusers_wan21_layout`) checks component
    manifests, index cardinality and file presence, but accepts arbitrary
    source keys and raw non-safetensors shard bytes; the download gate would
    then skip repair and generation would fail on :func:`_read_index` or
    ``mx.load``. This validator closes that gap while staying bounded: every
    index source key must map through the production mappers (duplicate
    targets rejected inside :func:`_read_index`), every indexed tensor must
    live in the shard its index points at, every referenced shard and the VAE
    file must parse as safetensors containers, and the VAE header must carry
    exactly the pinned uniquely-mapped decoder tensor set. No model is
    instantiated and no tensor payload is ever read.
    """
    try:
        for relative, expected, mapper in (
            (_TRANSFORMER_INDEX, _EXPECTED_TRANSFORMER_KEYS, _transformer_key),
            (_T5_INDEX, _EXPECTED_T5_KEYS, _t5_key),
        ):
            directory = (root / relative).parent
            grouped = _read_index(root, relative, expected, mapper)
            for filename, names in grouped.items():
                present = _safetensors_tensor_names(directory / filename)
                if any(source not in present for source, _ in names):
                    return False
        decoder_targets: set[str] = set()
        for source in _safetensors_tensor_names(root / _VAE_FILE):
            target = _vae_decoder_key(source)
            if target is None:
                continue
            if target in decoder_targets:
                return False
            decoder_targets.add(target)
        return len(decoder_targets) == _EXPECTED_VAE_DECODER_KEYS
    except WanBackendError:
        return False


def _validate_target_parameters(
    model, mapped_names: set[str], *, ignored: frozenset[str] = frozenset()
) -> None:
    """Fail before loading if the pinned mapping and runtime model diverge."""
    from mlx.utils import tree_flatten

    model_names = {name for name, _ in tree_flatten(model.parameters())} - ignored
    if mapped_names != model_names:
        missing = sorted(model_names - mapped_names)
        unexpected = sorted(mapped_names - model_names)
        raise WanBackendError(
            "the Wan 2.1 tensor mapping does not match the bundled MLX model "
            f"(missing={missing[:3]!r}, unexpected={unexpected[:3]!r})"
        )


def _load_sharded(
    model,
    root: Path,
    relative: str,
    expected: int,
    mapper: Callable[[str], str],
    *,
    dtype=None,
    reshape_patch: bool = False,
    ignored_model_parameters: frozenset[str] = frozenset(),
) -> None:
    import mlx.core as mx

    directory = (root / relative).parent
    grouped = _read_index(root, relative, expected, mapper)
    _validate_target_parameters(
        model,
        {target for names in grouped.values() for _, target in names},
        ignored=ignored_model_parameters,
    )
    for filename, names in grouped.items():
        path = directory / filename
        if not path.is_file():
            raise WanBackendError(f"the Wan 2.1 checkpoint is missing {filename!r}")
        source_weights = mx.load(str(path), return_metadata=False)
        weights = []
        for source, target in names:
            if source not in source_weights:
                raise WanBackendError(f"the Wan 2.1 shard is missing tensor {source!r}")
            value = source_weights[source]
            if reshape_patch and source == "patch_embedding.weight":
                value = value.reshape(value.shape[0], -1)
            if dtype is not None:
                value = value.astype(dtype)
            weights.append((target, value))
        model.load_weights(weights, strict=False)
        del source_weights, weights


def _load_transformer(root: Path, config, quantization=None, loras=None):
    if quantization or loras:
        raise WanBackendError(
            "Wan 2.1 Desktop weights do not support quantization or LoRA overlays"
        )
    import mlx.core as mx
    from mlx_video.models.wan.model import WanModel

    model = WanModel(config)
    _load_sharded(
        model,
        root,
        _TRANSFORMER_INDEX,
        _EXPECTED_TRANSFORMER_KEYS,
        _transformer_key,
        reshape_patch=True,
        ignored_model_parameters=frozenset({"freqs"}),
    )
    mx.eval(model.parameters())
    return model


def _load_t5(root: Path, config):
    import mlx.core as mx
    from mlx_video.models.wan.text_encoder import T5Encoder

    encoder = T5Encoder(
        vocab_size=config.t5_vocab_size,
        dim=config.t5_dim,
        dim_attn=config.t5_dim_attn,
        dim_ffn=config.t5_dim_ffn,
        num_heads=config.t5_num_heads,
        num_layers=config.t5_num_layers,
        num_buckets=config.t5_num_buckets,
        shared_pos=False,
    )
    _load_sharded(
        encoder, root, _T5_INDEX, _EXPECTED_T5_KEYS, _t5_key, dtype=mx.float32
    )
    mx.eval(encoder.parameters())
    return encoder


def _load_vae(root: Path, config=None):
    import mlx.core as mx
    from mlx_video.models.wan.vae import WanVAE

    source = mx.load(str(root / _VAE_FILE), return_metadata=False)
    weights = []
    for key, value in source.items():
        target = _vae_decoder_key(key)
        if target is None:
            continue
        if "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        elif "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))
        weights.append((target, value.astype(mx.float32)))
    if len(weights) != _EXPECTED_VAE_DECODER_KEYS or len(
        {key for key, _ in weights}
    ) != len(weights):
        raise WanBackendError("the Wan 2.1 VAE has an unexpected decoder tensor set")
    vae = WanVAE(z_dim=16)
    _validate_target_parameters(
        vae,
        {key for key, _ in weights},
        ignored=frozenset({"mean", "std", "inv_std"}),
    )
    vae.load_weights(weights, strict=False)
    mx.eval(vae.parameters())
    return vae


def _scoped_generate_function(root: Path, generator) -> Callable:
    """Clone the generator with request-local loaders and tokenizer imports."""
    if not is_diffusers_wan21_layout(root):
        raise WanBackendError("the Wan 2.1 checkpoint layout is incomplete")
    original = generator.generate_video
    original_import = original.__builtins__["__import__"]

    class LocalTokenizer:
        @classmethod
        def from_pretrained(cls, _model_name, *args, **kwargs):
            from transformers import AutoTokenizer

            kwargs["local_files_only"] = True
            return AutoTokenizer.from_pretrained(root / "tokenizer", *args, **kwargs)

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported = original_import(name, globals, locals, fromlist, level)
        if name == "transformers" and "AutoTokenizer" in fromlist:
            proxy = ModuleType("transformers")
            # Module attributes live in __dict__; mypy's lvalue lookup does
            # not consult ModuleType.__getattr__, so write there directly.
            proxy.__dict__["AutoTokenizer"] = LocalTokenizer
            return proxy
        return imported

    builtins = dict(original.__builtins__)
    builtins["__import__"] = scoped_import
    namespace = dict(original.__globals__)
    namespace.update(
        {
            "__builtins__": builtins,
            "load_wan_model": lambda _path, config, quantization=None, loras=None: (
                _load_transformer(root, config, quantization, loras)
            ),
            "load_t5_encoder": lambda _path, config: _load_t5(root, config),
            "load_vae_decoder": lambda _path, config=None: _load_vae(root, config),
        }
    )
    scoped = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    scoped.__kwdefaults__ = original.__kwdefaults__
    return scoped


@contextmanager
def diffusers_runtime(root: Path, generator) -> Iterator[tuple[Path, Callable]]:
    """Create a temporary converted-layout view and isolated generator."""
    scoped_generate = _scoped_generate_function(root, generator)
    temporary = tempfile.TemporaryDirectory(prefix="rapidmlx-wan21-view-")
    view = Path(temporary.name)
    (view / "config.json").write_text(json.dumps(desktop_wan21_config()))
    try:
        yield view, scoped_generate
    finally:
        temporary.cleanup()


def generate_with_runtime(root: Path, generator, generation_kwargs: dict) -> None:
    """Generate through request-local loaders for the official layout."""
    if is_diffusers_wan21_layout(root):
        with diffusers_runtime(root, generator) as (model_view, scoped_generate):
            # The temporary view must not leak into the caller's mapping:
            # replace model_dir on a copy only.
            scoped_generate(**{**generation_kwargs, "model_dir": str(model_view)})
        return
    if _identifies_as_diffusers_wan21(root):
        raise WanBackendError("the Wan 2.1 checkpoint layout is incomplete")
    generator.generate_video(**generation_kwargs)
