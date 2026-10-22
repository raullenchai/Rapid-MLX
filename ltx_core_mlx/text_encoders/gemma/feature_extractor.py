"""GemmaFeaturesExtractorV2 — multi-layer hidden state projection + dual connectors.

Ported from ltx-core/src/ltx_core/text_encoders/gemma/feature_extractor.py

Weight keys (under ``connector.``):
    ``text_embedding_projection.video_aggregate_embed.{weight,bias}``
    ``text_embedding_projection.audio_aggregate_embed.{weight,bias}``
    ``video_embeddings_connector.learnable_registers``
    ``video_embeddings_connector.transformer_1d_blocks.{0..7}.*``
    ``audio_embeddings_connector.learnable_registers``
    ``audio_embeddings_connector.transformer_1d_blocks.{0..7}.*``
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.text_encoders.gemma.embeddings_connector import Embeddings1DConnector

# Threshold below which we force intermediate graph materialization to
# avoid Metal GPU watchdog timeouts ("Impacting Interactivity").
# Machines with more memory can handle the full graph in one pass.
_LOW_MEMORY_THRESHOLD = 48 * 1024**3  # 48 GB


def _materialize(*arrays: mx.array) -> None:
    """Force MLX compute graph materialization on low-memory devices.

    On devices with <= 48GB, breaks large compute graphs into smaller
    Metal command buffers. On larger devices, this is a no-op.
    """
    if mx.device_info()["memory_size"] <= _LOW_MEMORY_THRESHOLD:
        # NOTE: mx.eval is MLX graph evaluation, NOT Python eval()
        mx.eval(*arrays)


class TextEmbeddingProjection(nn.Module):
    """Projects concatenated multi-layer Gemma hidden states to video/audio dims.

    Takes the concatenation of all 49 Gemma layers' hidden states
    (49 * 3840 = 188160) and projects to separate video and audio dimensions.

    Weight keys:
        ``video_aggregate_embed.{weight,bias}``: (video_dim, input_dim)
        ``audio_aggregate_embed.{weight,bias}``: (audio_dim, input_dim)

    Args:
        input_dim: Concatenated hidden state dimension (num_layers * hidden_dim).
        video_dim: Output dimension for video embeddings.
        audio_dim: Output dimension for audio embeddings.
    """

    def __init__(
        self,
        input_dim: int = 188160,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        embedding_dim: int = 3840,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.video_aggregate_embed = nn.Linear(input_dim, video_dim, bias=True)
        self.audio_aggregate_embed = nn.Linear(input_dim, audio_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        """Project multi-layer hidden states with rescaling.

        Reference: FeatureExtractorV2 applies _rescale_norm(normed, target_dim, embedding_dim)
        before projection: x * sqrt(target_dim / embedding_dim).

        Args:
            hidden_states: (B, seq_len, input_dim) concatenated layer outputs.

        Returns:
            Tuple of (video_embeds, audio_embeds).
        """
        v_dim = self.video_aggregate_embed.weight.shape[0]
        v_scale = math.sqrt(v_dim / self.embedding_dim)
        video_embeds = self.video_aggregate_embed(hidden_states * v_scale)
        _materialize(video_embeds)

        a_dim = self.audio_aggregate_embed.weight.shape[0]
        a_scale = math.sqrt(a_dim / self.embedding_dim)
        audio_embeds = self.audio_aggregate_embed(hidden_states * a_scale)
        _materialize(audio_embeds)

        return video_embeds, audio_embeds


class TextEncoderConnector(nn.Module):
    """Top-level connector module matching connector.safetensors structure.

    Combines the text embedding projection with separate video and audio
    transformer connectors.

    Weight keys (all under ``connector.`` prefix in safetensors):
        ``text_embedding_projection.*``
        ``video_embeddings_connector.*``
        ``audio_embeddings_connector.*``

    Args:
        caption_channels: Gemma hidden dimension (3840 for Gemma 3 12B).
        num_gemma_layers: Number of Gemma layers including embedding (49).
        video_dim: Video embedding dimension.
        audio_dim: Audio embedding dimension.
        num_heads: Number of attention heads in connectors.
        video_head_dim: Head dimension for video connector.
        audio_head_dim: Head dimension for audio connector.
        num_layers: Number of transformer blocks per connector.
        num_registers: Number of learnable register tokens.
        ff_mult: Feed-forward multiplier.
        max_pos: Maximum position for RoPE.
        norm_output: Whether to normalize connector output.
    """

    def __init__(
        self,
        caption_channels: int = 3840,
        num_gemma_layers: int = 49,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        num_heads: int = 32,
        video_head_dim: int = 128,
        audio_head_dim: int = 64,
        num_layers: int = 8,
        num_registers: int = 128,
        ff_mult: float = 4.0,
        max_pos: int = 4096,
        norm_output: bool = True,
    ):
        super().__init__()

        input_dim = num_gemma_layers * caption_channels  # 49 * 3840 = 188160

        self.text_embedding_projection = TextEmbeddingProjection(
            input_dim=input_dim,
            video_dim=video_dim,
            audio_dim=audio_dim,
        )

        self.video_embeddings_connector = Embeddings1DConnector(
            dim=video_dim,
            num_heads=num_heads,
            head_dim=video_head_dim,
            num_layers=num_layers,
            num_registers=num_registers,
            ff_mult=ff_mult,
            max_pos=max_pos,
            norm_output=norm_output,
        )

        self.audio_embeddings_connector = Embeddings1DConnector(
            dim=audio_dim,
            num_heads=num_heads,
            head_dim=audio_head_dim,
            num_layers=num_layers,
            num_registers=num_registers,
            ff_mult=ff_mult,
            max_pos=max_pos,
            norm_output=norm_output,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Project and refine multi-layer hidden states.

        Args:
            hidden_states: (B, seq_len, num_layers * caption_channels)
                concatenated hidden states from all Gemma layers.
            attention_mask: (B, seq_len) boolean mask.

        Returns:
            Tuple of (video_embeds, audio_embeds).
        """
        # Project to separate video/audio dimensions. The TextEmbeddingProjection's
        # built-in _materialize splits the giant 188160→4096 matmul into
        # video and audio dispatches on <=48 GB Macs.
        video_embeds, audio_embeds = self.text_embedding_projection(hidden_states)

        # Refine through transformer connectors. Embeddings1DConnector
        # materializes per-block on <=48 GB Macs (see embeddings_connector.py).
        video_embeds = self.video_embeddings_connector(video_embeds, attention_mask=attention_mask)
        audio_embeds = self.audio_embeddings_connector(audio_embeds, attention_mask=attention_mask)

        return video_embeds, audio_embeds


class GemmaFeaturesExtractorV2(nn.Module):
    """Complete text encoding pipeline.

    Pipeline:
        1. Extract hidden states from all 49 Gemma layers
        2. Stack as (B, seq_len, num_layers * hidden_dim)
        3. Apply per-token RMS normalization + rescale
        4. Project via TextEmbeddingProjection (separate video/audio)
        5. Refine through transformer connectors

    Args:
        caption_channels: Gemma hidden dimension.
        num_gemma_layers: Number of Gemma layers.
        video_dim: Video embedding dimension.
        audio_dim: Audio embedding dimension.
        num_heads: Connector attention heads.
        video_head_dim: Video connector head dimension.
        audio_head_dim: Audio connector head dimension.
        num_connector_layers: Connector transformer layers.
        num_registers: Number of learnable registers.
        norm_type: Normalization type for hidden states.
    """

    def __init__(
        self,
        caption_channels: int = 3840,
        num_gemma_layers: int = 49,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        num_heads: int = 32,
        video_head_dim: int = 128,
        audio_head_dim: int = 64,
        num_connector_layers: int = 8,
        num_registers: int = 128,
        norm_type: str = "per_token_rms",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.num_gemma_layers = num_gemma_layers
        self.caption_channels = caption_channels

        self.connector = TextEncoderConnector(
            caption_channels=caption_channels,
            num_gemma_layers=num_gemma_layers,
            video_dim=video_dim,
            audio_dim=audio_dim,
            num_heads=num_heads,
            video_head_dim=video_head_dim,
            audio_head_dim=audio_head_dim,
            num_layers=num_connector_layers,
            num_registers=num_registers,
        )

    def __call__(
        self,
        all_hidden_states: list[mx.array],
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Extract features from all Gemma hidden states.

        Args:
            all_hidden_states: List of (B, seq_len, hidden_dim) tensors,
                one per Gemma layer (49 total).
            attention_mask: (B, seq_len) boolean mask.

        Returns:
            Tuple of (video_embeds, audio_embeds).
        """
        # Per-token RMS normalization, then stack and reshape.
        # Reference: torch.stack(hidden_states, dim=-1) → [B, T, D, L]
        # then RMSNorm over dim=2 (D), then reshape to [B, T, D*L].
        # This gives D-interleaved order: [d0_l0, d0_l1, ..., d1_l0, ...]
        # which the projection weights were trained with.
        if self.norm_type == "per_token_rms":
            # Stack on last axis: [B, T, D, L]
            encoded = mx.stack(all_hidden_states, axis=-1)
            # RMSNorm over D dimension (axis=2)
            variance = mx.mean(encoded * encoded, axis=2, keepdims=True)
            normed = encoded * mx.rsqrt(variance + 1e-6)
            # Reshape to [B, T, D*L] — D-interleaved order
            B, T, D, L = normed.shape
            stacked = normed.reshape(B, T, D * L)
            # Zero out padding positions
            if attention_mask is not None:
                mask_3d = attention_mask[:, :, None].astype(stacked.dtype)
                stacked = stacked * mask_3d
        else:
            stacked = mx.stack(all_hidden_states, axis=-1)
            B, T, D, L = stacked.shape
            stacked = stacked.reshape(B, T, D * L)

        # Materialize before projection to limit Metal command buffer size
        _materialize(stacked)

        # Project and refine through connectors
        return self.connector(stacked, attention_mask=attention_mask)
