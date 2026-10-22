"""Module-level orchestration helpers shared across pipelines.

These functions own the multi-component flows that don't belong to a
single :mod:`utils.blocks` block — loading transformers (with LoRA
fusion + optional block streaming), the audio + video decode-and-mux
sequence, and waveform-to-WAV serialization.

Every function takes its dependencies as arguments rather than
``self``, so pipelines can call them via composition (no
:class:`BasePipeline` inheritance required).
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

if TYPE_CHECKING:
    from ltx_pipelines_mlx.utils.blocks import AudioDecoder, VideoDecoder


def resolve_model_dir(model_dir: str | Path) -> Path:
    """Resolve a model dir — return local path or download from HuggingFace."""
    path = Path(model_dir)
    if path.exists():
        return path
    return Path(snapshot_download(str(model_dir)))


def detect_model_version(model_dir: str | Path) -> str:
    """Detect the LTX checkpoint version from a model dir's transformer config.

    Reads ``embedded_config.json`` (preferred) or ``config.json`` in the model
    dir and returns:

    - ``"2.5.0"`` when the config sets ``use_keyframes_abs_pos_embedding: true``
      (the LTX-2.5 DiT marker tensor) or an explicit ``model_version`` field
      starting with ``2.5``;
    - ``"2.3"`` otherwise (the local port's default).

    The 2.5 transformer weights are otherwise 1:1 compatible with the 2.3
    loader (same architecture / key names), so this only gates the text
    encoder path and optional components.

    Args:
        model_dir: Local model directory (already resolved).

    Returns:
        ``"2.5.0"`` or ``"2.3"``.
    """
    import json

    model_dir = Path(model_dir)
    for name in ("embedded_config.json", "config.json"):
        path = model_dir / name
        if not path.exists():
            continue
        try:
            config = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        transformer = config.get("transformer", config)
        model_version = transformer.get("model_version")
        if isinstance(model_version, str) and model_version.startswith("2.5"):
            return "2.5.0"
        if transformer.get("use_keyframes_abs_pos_embedding"):
            return "2.5.0"
        return "2.3"
    return "2.3"


def resolve_lora_path(path: str) -> str:
    """Resolve a LoRA path — return local path or download from HuggingFace.

    Supports local ``.safetensors`` files and HuggingFace repo IDs
    (e.g. ``"some-user/my-lora"``).  When a repo is downloaded its
    ``.safetensors`` files are collected; exactly one must be present.
    """
    local = Path(path)
    if local.exists():
        return str(local)

    print(f"Downloading LoRA from HuggingFace: {path}", file=sys.stderr)
    repo_dir = Path(snapshot_download(path))
    safetensors_files = list(repo_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {repo_dir}")
    if len(safetensors_files) > 1:
        names = sorted(f.name for f in safetensors_files)
        raise ValueError(
            f"Ambiguous: {len(safetensors_files)} .safetensors files in repo '{path}' "
            f"({', '.join(names)}); pass the full local path to the specific file instead."
        )
    return str(safetensors_files[0])


def fuse_pending_loras(
    transformer_weights: dict[str, mx.array],
    lora_paths: list[tuple[str, float]],
) -> dict[str, mx.array]:
    """Fuse LoRA deltas into transformer weights before model loading."""
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core_mlx.loader.sft_loader import SafetensorsStateDictLoader

    model_sd = StateDict(sd=transformer_weights, size=0, dtype=set())
    loader = SafetensorsStateDictLoader()

    lora_sds = []
    for lora_path, strength in lora_paths:
        resolved = resolve_lora_path(lora_path)
        lora_sd = loader.load(resolved, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
        print(f"  Fusing LoRA: {lora_path} (strength={strength:.2f})", file=sys.stderr)

    fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)
    return fused_sd.sd


def load_transformer(
    transformer_path: Path,
    *,
    low_ram_streaming: bool = False,
) -> LTXModel:
    """Build an LTXModel from safetensors with optional block streaming.

    When ``low_ram_streaming`` is ``True``, drops blocks 1..47 before
    quantization (so apply_quantization only materializes block 0),
    loads non-block weights, and wraps the model in StreamingLTXModel
    for per-forward block streaming.

    The model config is read from the checkpoint directory
    (``embedded_config.json`` / ``config.json``) so hyperparameters such as
    ``av_ca_timestep_scale_multiplier`` track the checkpoint rather than the
    hardcoded dataclass defaults (issue #37).
    """
    config = LTXModelConfig.from_checkpoint_dir(transformer_path.parent)
    dit = LTXModel(config)
    weights = load_split_safetensors(transformer_path, prefix="transformer.")
    if low_ram_streaming:
        from ltx_core_mlx.loader.block_streaming import BlockStreamer, StreamingLTXModel

        dit.transformer_blocks = [dit.transformer_blocks[0]]
        apply_quantization(dit, weights)
        non_block = [(k, v) for k, v in weights.items() if not k.startswith("transformer_blocks.")]
        dit.load_weights(non_block, strict=False)
        streamer = BlockStreamer(transformer_path, block_prefix="transformer.transformer_blocks.")
        dit = StreamingLTXModel(dit, streamer)
    else:
        apply_quantization(dit, weights)
        dit.load_weights(list(weights.items()))
    mx.eval(dit.parameters())
    aggressive_cleanup()
    return dit


def load_dev_transformer(
    model_dir: Path,
    transformer_name: str,
    *,
    low_ram_streaming: bool = False,
) -> LTXModel:
    """Load the dev (non-distilled) transformer; raises if missing."""
    dev_path = model_dir / transformer_name
    if not dev_path.exists():
        raise FileNotFoundError(
            f"Dev transformer not found: {dev_path}\n"
            "This pipeline requires the dev model for CFG guidance.\n"
            "Use: --model dgrauet/ltx-2.3-mlx-q8"
        )
    return load_transformer(dev_path, low_ram_streaming=low_ram_streaming)


def save_waveform(waveform: mx.array, path: str, sample_rate: int = 48000) -> None:
    """Save a ``(B, C, T)`` or ``(B, T)`` waveform to a 16-bit PCM WAV file."""
    wav = waveform[0]
    if wav.ndim == 2:
        num_channels = wav.shape[0]
        wav = wav.T
    else:
        num_channels = 1
        wav = wav[:, None]

    wav_np = np.array(wav.astype(mx.float32), dtype=np.float32)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    wav_int16 = (wav_np * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wav_int16.tobytes())


def decode_and_save_video(
    video_decoder: VideoDecoder,
    audio_decoder: AudioDecoder,
    video_latent: mx.array,
    audio_latent: mx.array,
    output_path: str,
    *,
    frame_rate: float,
    low_memory: bool = True,
) -> str:
    """Decode audio+video latents and mux to mp4 via ffmpeg.

    Args:
        video_decoder: :class:`VideoDecoder` block (loads vae_decoder lazily).
        audio_decoder: :class:`AudioDecoder` block (audio VAE + vocoder).
        video_latent: Encoded video latent.
        audio_latent: Encoded audio latent.
        output_path: Destination mp4 path.
        frame_rate: Output frame rate.
        low_memory: When ``True`` aggressively releases intermediate
            buffers between audio and video decode.
    """
    import tempfile

    waveform = audio_decoder(audio_latent)
    if low_memory:
        aggressive_cleanup()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
        audio_path = _tmp.name
    save_waveform(waveform, audio_path, sample_rate=48000)

    video_decoder.decode_and_stream(video_latent, output_path, frame_rate=frame_rate, audio_path=audio_path)

    Path(audio_path).unlink(missing_ok=True)
    aggressive_cleanup()

    return output_path


def combined_image_conditionings(
    images,
    *,
    enc_h: int,
    enc_w: int,
    spatial_dims: tuple[int, int, int],
    video_encoder,
    frame_rate: float,
    model_dir: str | Path | None = None,
):
    """Build a list of conditioning items from a list of input images.

    Mirrors upstream ``ltx_pipelines.utils.helpers.combined_image_conditionings``:

    - First image with ``frame_idx == 0`` becomes a
      :class:`VideoConditionByLatentIndex` (replaces latent[0]).
    - Other images become :class:`VideoConditionByKeyframeIndex`
      entries appended at their respective frame indices.

    Args:
        images: List of :class:`ImageConditioningInput`.
        enc_h: Encoder spatial height (must be divisible by 32).
        enc_w: Encoder spatial width.
        spatial_dims: ``(F, H, W)`` latent shape of the target video.
        video_encoder: VAE encoder instance (must expose ``encode``).
        frame_rate: Frame rate for keyframe positions.
        model_dir: Optional model dir. When given, the image CRF default is
            resolved per model version (LTX-2.5 → 18, 2.3 → 33); an explicit
            ``crf`` on the conditioning item always wins, except an explicit
            value equal to the 2.3 default (33) is treated as unset and
            re-resolved for the active version.

    Returns:
        List of conditioning items ready to feed into
        :func:`create_noised_state`.
    """
    from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core_mlx.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_pipelines_mlx.utils.media_io import (
        DEFAULT_IMAGE_CRF,
        default_image_crf,
        load_image_and_preprocess,
    )

    # Per-version I2V image CRF default: the 2.3 port inherits upstream's 33;
    # the official 2.5 pipelines preprocess at 18 (fix #4).
    version_crf = default_image_crf(detect_model_version(model_dir)) if model_dir is not None else DEFAULT_IMAGE_CRF

    conditionings: list = []
    for img in images:
        # Forward the per-image CRF (upstream-iso). Falls back to the
        # version-aware default (33 for 2.3, 18 for 2.5) when the image item
        # doesn't expose a CRF field or still carries the 2.3 default.
        img_crf = getattr(img, "crf", None)
        if img_crf is None or img_crf == DEFAULT_IMAGE_CRF:
            img_crf = version_crf
        img_tensor = load_image_and_preprocess(img.path, enc_h, enc_w, crf=img_crf)
        img_tensor = img_tensor[:, :, None, :, :]  # add F=1 dim
        ref_latent = video_encoder.encode(img_tensor)  # (1, 128, 1, H', W')
        ref_tokens = ref_latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)

        if img.frame_idx == 0:
            conditionings.append(
                VideoConditionByLatentIndex(
                    frame_indices=[0],
                    clean_latent=ref_tokens,
                    strength=img.strength,
                )
            )
        else:
            conditionings.append(
                VideoConditionByKeyframeIndex(
                    frame_idx=img.frame_idx,
                    keyframe_latent=ref_tokens,
                    spatial_dims=spatial_dims,
                    frame_rate=frame_rate,
                    strength=img.strength,
                )
            )
    return conditionings


__all__ = [
    "combined_image_conditionings",
    "decode_and_save_video",
    "detect_model_version",
    "fuse_pending_loras",
    "load_dev_transformer",
    "load_transformer",
    "resolve_lora_path",
    "resolve_model_dir",
    "save_waveform",
]
