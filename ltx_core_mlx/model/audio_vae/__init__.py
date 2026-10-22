"""Audio VAE — decoder, encoder, vocoder, BWE, and audio processing."""

import mlx.core as mx

from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
from ltx_core_mlx.model.audio_vae.encoder import AudioVAEEncoder
from ltx_core_mlx.model.audio_vae.processor import AudioProcessor
from ltx_core_mlx.model.audio_vae.vocoder import BigVGANVocoder

__all__ = [
    "AudioProcessor",
    "AudioVAEDecoder",
    "AudioVAEEncoder",
    "BigVGANVocoder",
    "VocoderWithBWE",
    "encode_audio",
]


def encode_audio(
    waveform: mx.array,
    sample_rate: int,
    encoder: AudioVAEEncoder,
    processor: AudioProcessor | None = None,
) -> mx.array:
    """Encode audio waveform to latent representation.

    Args:
        waveform: (1, channels, samples) audio waveform.
        sample_rate: Sample rate of the waveform.
        encoder: Audio VAE encoder model.
        processor: Audio processor (created from defaults if None).

    Returns:
        Audio latent (1, 8, T, 16).
    """
    if processor is None:
        processor = AudioProcessor(sample_rate=sample_rate)

    mel = processor.waveform_to_mel(waveform)  # (1, channels, T', n_mels)
    return encoder.encode(mel)
