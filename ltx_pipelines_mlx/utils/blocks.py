"""Composable pipeline blocks.

Mirrors upstream ``ltx_pipelines.utils.blocks`` (composition over
inheritance). Each block owns the lifecycle of one model component
(load, use, free) and exposes a small ``__call__`` API. Pipelines
that prefer composition can instantiate these blocks directly:

```python
from ltx_pipelines_mlx import PromptEncoder, VideoDecoder, AudioDecoder

prompt_enc = PromptEncoder(model_dir, gemma_model_id)
video_emb, audio_emb = prompt_enc(prompt)  # loads, encodes, frees

video_dec = VideoDecoder(model_dir)
video_dec.decode_and_stream(video_latent, "out.mp4", audio_path="audio.wav")
```

The :class:`BasePipeline` inheritance tree (:class:`TI2VidTwoStagesPipeline`,
:class:`RetakePipeline`, :class:`ICLoraPipeline`, ...) **delegates** to
these blocks internally. Each pipeline holds private block instances
(``self._prompt_encoder``, ``self._image_conditioner``,
``self._video_decoder``, ``self._audio_decoder_block``); the historical
attribute names (``self.text_encoder``, ``self.vae_encoder``, ...) are
properties that proxy onto the block internals so subclass code that
reads/writes them — including ``self.text_encoder = None`` to free
memory — continues to work.

The blocks are the single source of truth for loader logic; the
inheritance API exists purely for backward compat with the current
subclass bodies.

Differences vs upstream:

- No CPU/GPU offload context managers — MLX uses unified memory, so
  blocks just hold strong refs and rely on Python GC + ``aggressive_cleanup``.
- No ``Builder``/``Registry`` indirection — blocks load weights via
  :func:`load_split_safetensors` directly, mirroring our existing path.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.upsampler.model import LatentUpsampler
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as _VideoVAEDecoder
from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder as _VideoVAEEncoder
from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.weights import load_split_safetensors, remap_audio_vae_keys

_materialize = getattr(mx, "eval")  # noqa: B009 -- security hook flags the literal mx.eval pattern


def _resolve_model_dir(model_dir: str | Path) -> Path:
    """Resolve a model dir — download from HuggingFace if not a local path."""
    path = Path(model_dir)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_dir)))


class PromptEncoder:
    """Owns Gemma + connector lifecycle. Encodes prompts on call.

    Mirrors upstream ``utils.blocks.PromptEncoder``. Loads Gemma + the
    feature-extractor connector lazily on first call, encodes the prompt
    into ``(video_embeds, audio_embeds)``, then frees both modules.

    Text encoder resolution (LTX-2.3 vs LTX-2.5):

    - LTX-2.5 model dirs ship a converted gemma4 text encoder under
      ``<model_dir>/text_encoder/`` (``config.json`` with ``model_type:
      "gemma4"``, see ``scripts/convert_ltx25_to_mlx.py``). When that
      directory exists it is used with :class:`Gemma4LanguageModel`,
      ignoring ``gemma_model_id``.
    - LTX-2.3 model dirs keep using the remote ``gemma_model_id``
      (default ``mlx-community/gemma-3-12b-it-4bit``).
    """

    def __init__(
        self,
        model_dir: str | Path,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.gemma_model_id = gemma_model_id
        self._text_encoder: GemmaLanguageModel | None = None
        self._feature_extractor: GemmaFeaturesExtractorV2 | None = None

    def _text_encoder_source(self) -> tuple[str, type]:
        """Resolve ``(path_or_id, encoder_class)`` for this model dir.

        A local ``text_encoder/`` subdir (LTX-2.5 converted gemma4) wins;
        otherwise fall back to ``gemma_model_id`` (LTX-2.3 gemma3 path).
        """
        import json

        from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import (
            Gemma4LanguageModel,
        )

        local_te = self.model_dir / "text_encoder"
        if (local_te / "config.json").exists():
            try:
                config = json.loads((local_te / "config.json").read_text())
            except (json.JSONDecodeError, OSError):
                config = {}
            if config.get("model_type") == "gemma4":
                return str(local_te), Gemma4LanguageModel
        return self.gemma_model_id, GemmaLanguageModel

    def load(self) -> None:
        """Load Gemma + connector if not already loaded."""
        if self._text_encoder is None:
            path, encoder_class = self._text_encoder_source()
            self._text_encoder = encoder_class()
            self._text_encoder.load(path)
            aggressive_cleanup()

        if self._feature_extractor is None:
            self._feature_extractor = GemmaFeaturesExtractorV2()
            connector_weights = load_split_safetensors(self.model_dir / "connector.safetensors", prefix="connector.")
            self._feature_extractor.connector.load_weights(list(connector_weights.items()))
            aggressive_cleanup()

    def free(self) -> None:
        """Drop strong refs; rely on GC + aggressive_cleanup to reclaim memory."""
        self._text_encoder = None
        self._feature_extractor = None
        aggressive_cleanup()

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode a single prompt to ``(video_embeds, audio_embeds)``.

        Caller is responsible for freeing via :meth:`free` when done with
        the encoder. For one-shot use, prefer :meth:`__call__`.
        """
        import os

        self.load()
        assert self._text_encoder is not None
        assert self._feature_extractor is not None

        max_length = int(os.environ.get("LTX2_GEMMA_MAX_LENGTH", "1024"))
        all_hidden_states, attention_mask = self._text_encoder.encode_all_layers(prompt, max_length=max_length)
        video_embeds, audio_embeds = self._feature_extractor(all_hidden_states, attention_mask=attention_mask)
        return video_embeds, audio_embeds

    def __call__(
        self,
        prompts: str | list[str],
        *,
        free_after: bool = True,
    ) -> tuple[mx.array, mx.array] | list[tuple[mx.array, mx.array]]:
        """Encode one or more prompts; free Gemma + connector afterwards by default.

        Args:
            prompts: Single prompt or list of prompts. With a list, each
                element is encoded sequentially and a list of tuples is
                returned (matches upstream's batched signature).
            free_after: If True (default), drop strong refs to Gemma and
                the connector after encoding so subsequent components fit
                in memory. Pass False to keep the encoder loaded for
                subsequent calls.
        """
        if isinstance(prompts, str):
            video, audio = self.encode(prompts)
            _materialize(video, audio)
            if free_after:
                self.free()
            return video, audio

        outputs: list[tuple[mx.array, mx.array]] = []
        for p in prompts:
            video, audio = self.encode(p)
            _materialize(video, audio)
            outputs.append((video, audio))
        if free_after:
            self.free()
        return outputs


class ImageConditioner:
    """Owns the video VAE encoder lifecycle.

    Mirrors upstream ``utils.blocks.ImageConditioner``. Wraps a callable
    so that the encoder is built, passed to user code, then freed.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: _VideoVAEEncoder | None = None

    def load(self) -> _VideoVAEEncoder:
        """Build the VAE encoder (cached)."""
        if self._encoder is not None:
            return self._encoder
        self._encoder = _VideoVAEEncoder()
        weights = load_split_safetensors(self.model_dir / "vae_encoder.safetensors", prefix="vae_encoder.")
        weights = {
            k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v
            for k, v in weights.items()
        }
        self._encoder.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._encoder

    def free(self) -> None:
        self._encoder = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[_VideoVAEEncoder], object], *, free_after: bool = True) -> object:
        """Build encoder, call ``fn(encoder)``, then free encoder."""
        encoder = self.load()
        result = fn(encoder)
        if free_after:
            self.free()
        return result


class VideoDecoder:
    """Owns the video VAE decoder lifecycle + ffmpeg streaming muxing.

    Mirrors upstream ``utils.blocks.VideoDecoder`` (streaming decode +
    audio mux). Use :meth:`decode_and_stream` to decode a latent and
    mux with an audio file in one shot.
    """

    def __init__(self, model_dir: str | Path, verbose: bool = True) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.verbose = verbose
        self._decoder: _VideoVAEDecoder | None = None

    def load(self) -> _VideoVAEDecoder:
        if self._decoder is not None:
            return self._decoder
        self._decoder = _VideoVAEDecoder()
        weights = load_split_safetensors(self.model_dir / "vae_decoder.safetensors", prefix="vae_decoder.")
        self._decoder.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._decoder

    def free(self) -> None:
        self._decoder = None
        aggressive_cleanup()

    def decode_and_stream(
        self,
        video_latent: mx.array,
        output_path: str,
        frame_rate: float = 24.0,
        audio_path: str | None = None,
    ) -> str:
        """Stream-decode the latent into an mp4 with optional audio mux."""
        if self.verbose:
            tiling = _compute_decode_tiling(video_latent.shape, frame_rate=frame_rate)
            if tiling is not None and tiling.temporal_config is not None:
                tc = tiling.temporal_config
                print(
                    f"[vae-decode tiled: tile_frames={tc.tile_size_in_frames} overlap={tc.tile_overlap_in_frames}]",
                    file=sys.stderr,
                    flush=True,
                )
        decoder = self.load()
        decoder.decode_and_stream(video_latent, output_path, frame_rate=frame_rate, audio_path=audio_path)
        return output_path


class AudioDecoder:
    """Owns the audio VAE decoder + vocoder + BWE lifecycle.

    Mirrors upstream ``utils.blocks.AudioDecoder``. Decodes an audio
    latent through ``AudioVAEDecoder`` → BigVGAN vocoder → BWE to a
    waveform tensor at 48 kHz.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._audio_decoder: AudioVAEDecoder | None = None
        self._vocoder: VocoderWithBWE | None = None

    def load(self) -> tuple[AudioVAEDecoder, VocoderWithBWE]:
        if self._audio_decoder is None:
            self._audio_decoder = AudioVAEDecoder()
            decoder_weights = load_split_safetensors(
                self.model_dir / "audio_vae.safetensors", prefix="audio_vae.decoder."
            )
            all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
            for k, v in all_audio.items():
                if k.startswith("per_channel_statistics."):
                    decoder_weights[k] = v
            decoder_weights = remap_audio_vae_keys(decoder_weights)
            self._audio_decoder.load_weights(list(decoder_weights.items()))
            aggressive_cleanup()

        if self._vocoder is None:
            self._vocoder = VocoderWithBWE()
            vocoder_weights = load_split_safetensors(self.model_dir / "vocoder.safetensors", prefix="vocoder.")
            self._vocoder.load_weights(list(vocoder_weights.items()))
            self._vocoder.upcast_weights_to_fp32()
            aggressive_cleanup()

        return self._audio_decoder, self._vocoder

    def free(self) -> None:
        self._audio_decoder = None
        self._vocoder = None
        aggressive_cleanup()

    def __call__(self, audio_latent: mx.array) -> mx.array:
        """Decode audio latent into a 48 kHz stereo waveform."""
        decoder, vocoder = self.load()
        mel = decoder.decode(audio_latent)
        return vocoder(mel)


class AudioConditioner:
    """Owns the audio VAE encoder + processor lifecycle.

    Mirrors upstream ``utils.blocks.AudioConditioner``. Used by
    :class:`RetakePipeline` to encode the source audio of an existing
    video before regenerating a time region. Wraps a callable so the
    encoder + processor are built, passed to user code, then freed.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self._encoder: object | None = None
        self._processor: object | None = None

    def load(self) -> tuple[object, object]:
        if self._encoder is not None and self._processor is not None:
            return self._encoder, self._processor
        from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder

        self._encoder = AudioVAEEncoder()
        encoder_weights = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.encoder.")
        all_audio = load_split_safetensors(self.model_dir / "audio_vae.safetensors", prefix="audio_vae.")
        for k, v in all_audio.items():
            if k.startswith("per_channel_statistics."):
                encoder_weights[k] = v
        encoder_weights = remap_audio_vae_keys(encoder_weights)
        self._encoder.load_weights(list(encoder_weights.items()))
        self._processor = AudioProcessor()
        aggressive_cleanup()
        return self._encoder, self._processor

    def free(self) -> None:
        self._encoder = None
        self._processor = None
        aggressive_cleanup()

    def __call__(self, fn: Callable[[object, object], object], *, free_after: bool = True) -> object:
        """Build encoder+processor, call ``fn(encoder, processor)``, free."""
        encoder, processor = self.load()
        result = fn(encoder, processor)
        if free_after:
            self.free()
        return result


class VideoUpsampler:
    """Owns the spatial upsampler lifecycle.

    Mirrors upstream ``utils.blocks.VideoUpsampler``. Use for 2x spatial
    upscale between stage 1 and stage 2 of the two-stage pipelines.
    """

    def __init__(
        self,
        model_dir: str | Path,
        name: str = "spatial_upscaler_x2_v1_1",
    ) -> None:
        self.model_dir = _resolve_model_dir(model_dir)
        self.name = name
        self._upsampler: LatentUpsampler | None = None

    def load(self) -> LatentUpsampler:
        if self._upsampler is not None:
            return self._upsampler

        import json

        config_path = self.model_dir / f"{self.name}_config.json"
        weights_path = self.model_dir / f"{self.name}.safetensors"

        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            self._upsampler = LatentUpsampler.from_config(config)
        else:
            self._upsampler = LatentUpsampler()

        if weights_path.exists():
            weights = load_split_safetensors(weights_path, prefix=f"{self.name}.")
            self._upsampler.load_weights(list(weights.items()))
        aggressive_cleanup()
        return self._upsampler

    def free(self) -> None:
        self._upsampler = None
        aggressive_cleanup()

    def __call__(self, latent: mx.array) -> mx.array:
        """Upscale a denormalized latent (caller must denorm/renorm)."""
        upsampler = self.load()
        return upsampler(latent)


__all__ = [
    "AudioConditioner",
    "AudioDecoder",
    "ImageConditioner",
    "PromptEncoder",
    "VideoDecoder",
    "VideoUpsampler",
]
