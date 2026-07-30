"""CogVideoX Transformer3D model, ported to MLX.

Port of videox_fun/models/cogvideox_transformer3d.py.
All sequence operations use (B, L, D) format.
"""

import json
import math
import os
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .embeddings import get_3d_sincos_pos_embed, apply_rotary_emb


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------


class CogVideoXLayerNormZero(nn.Module):
    """Modulated LayerNorm producing shift/scale/gate for video and text streams."""

    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, affine=elementwise_affine)
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)

    def __call__(self, hidden_states, encoder_hidden_states, temb):
        mod = nn.silu(temb)
        mod = self.linear(mod)
        if mod.ndim == 2:
            mod = mx.expand_dims(mod, 1)
        shift_h, scale_h, gate_h, shift_e, scale_e, gate_e = mx.split(mod, 6, axis=-1)
        h = self.norm(hidden_states) * (1 + scale_h) + shift_h
        e = self.norm(encoder_hidden_states) * (1 + scale_e) + shift_e
        return h, e, gate_h, gate_e


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm used for the final output block."""

    def __init__(self, embedding_dim: int, output_dim: int, elementwise_affine=True, eps=1e-5, chunk_dim=1):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, eps=eps, affine=elementwise_affine)

    def __call__(self, x, temb=None):
        emb = self.linear(self.silu(temb))
        # CogVideoX uses chunk_dim=1: shift first, then scale
        shift, scale = mx.split(emb, 2, axis=-1)
        if shift.ndim == 2:
            shift = mx.expand_dims(shift, 1)
            scale = mx.expand_dims(scale, 1)
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class _GELUApprox(nn.Module):
    """GELU activation with linear projection (diffusers GELU wrapper)."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def __call__(self, x):
        return nn.gelu_approx(self.proj(x))


class _GEGLU(nn.Module):
    """GEGLU activation with linear projection."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def __call__(self, x):
        h = self.proj(x)
        gate, value = mx.split(h, 2, axis=-1)
        return nn.gelu_approx(gate) * value


class FeedForward(nn.Module):
    """Feed-forward network matching diffusers naming convention.

    Weight keys: net.0.proj.weight, net.2.weight.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int = 0,
        dropout: float = 0.0,
        bias: bool = True,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()
        inner_dim = inner_dim or 4 * dim

        if activation_fn == "geglu":
            act = _GEGLU(dim, inner_dim, bias=bias)
        else:
            # "gelu-approximate" or default
            act = _GELUApprox(dim, inner_dim, bias=bias)

        self.net = [
            act,  # net.0 (with .proj)
            nn.Dropout(p=dropout),  # net.1
            nn.Linear(inner_dim, dim, bias=bias),  # net.2
        ]

    def __call__(self, x):
        x = self.net[0](x)
        x = self.net[2](x)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head attention with optional QK normalization."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        out_bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        # Diffusers uses nn.ModuleList([Linear, Dropout]) -> keys: to_out.0.weight
        self.to_out = [nn.Linear(inner_dim, query_dim, bias=out_bias)]

        self.norm_q = nn.LayerNorm(dim_head, eps=eps) if qk_norm else None
        self.norm_k = nn.LayerNorm(dim_head, eps=eps) if qk_norm else None

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Concatenated self-attention over text + video tokens.

        Args:
            hidden_states: (B, L_video, D) video features.
            encoder_hidden_states: (B, L_text, D) text features.
            image_rotary_emb: Optional (cos, sin) for RoPE.

        Returns:
            (attn_video, attn_text) both updated.
        """
        text_seq_length = encoder_hidden_states.shape[1]

        # Concatenate text + video
        x = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        B, L, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape to (B, H, L, D)
        q = q.reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        # Apply RoPE to video portion only
        if image_rotary_emb is not None:
            q_text = q[:, :, :text_seq_length]
            q_video = q[:, :, text_seq_length:]
            q_video = apply_rotary_emb(q_video, image_rotary_emb)
            q = mx.concatenate([q_text, q_video], axis=2)

            k_text = k[:, :, :text_seq_length]
            k_video = k[:, :, text_seq_length:]
            k_video = apply_rotary_emb(k_video, image_rotary_emb)
            k = mx.concatenate([k_text, k_video], axis=2)

        # SDPA
        scale = self.dim_head**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

        # Reshape back to (B, L, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.to_out[0](out)

        # Split back
        enc_out = out[:, :text_seq_length]
        vid_out = out[:, text_seq_length:]
        return vid_out, enc_out


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class CogVideoXPatchEmbed(nn.Module):
    """Patch embedding for CogVideoX."""

    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        self.post_patch_height = sample_height // patch_size
        self.post_patch_width = sample_width // patch_size
        self.post_time_compression_frames = (sample_frames - 1) // temporal_compression_ratio + 1

        if patch_size_t is None:
            # CogVideoX 1.0: Conv2d patchification
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                bias=bias,
            )
        else:
            # CogVideoX 1.5: Linear after manual reshape
            self.proj = nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)

        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.pos_embedding = pos_embedding

    def _get_positional_embeddings(self, sample_height, sample_width, sample_frames):
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embed = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
        )
        # Flatten (T, H*W, D) -> (T*H*W, D)
        pos_embed_flat = pos_embed.reshape(-1, self.embed_dim)

        joint = mx.zeros((1, self.max_text_seq_length + num_patches, self.embed_dim))
        # Place video pos embeddings after text positions
        # We'll construct by concatenation
        text_zeros = mx.zeros((1, self.max_text_seq_length, self.embed_dim))
        video_pos = mx.expand_dims(pos_embed_flat[:num_patches], 0)
        joint = mx.concatenate([text_zeros, video_pos], axis=1)
        return joint

    def __call__(self, text_embeds: mx.array, image_embeds: mx.array):
        """
        Args:
            text_embeds: (B, text_seq_len, text_dim)
            image_embeds: (B, num_frames, channels, height, width) — NOTE: channels-first per-frame
        """
        text_embeds = self.text_proj(text_embeds)
        text_seq_length = text_embeds.shape[1]

        B, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            # Conv2d path: reshape to (B*F, H, W, C) for MLX Conv2d
            img = image_embeds.reshape(B * num_frames, channels, height, width)
            img = img.transpose(0, 2, 3, 1)  # NCHW -> NHWC for MLX Conv2d
            img = self.proj(img)  # (B*F, H', W', embed_dim)
            BF, H2, W2, D = img.shape
            img = img.reshape(B, num_frames, H2 * W2, D)
            image_embeds = img.reshape(B, num_frames * H2 * W2, D)
        else:
            # Linear path: manual patchify
            p = self.patch_size
            p_t = self.patch_size_t
            # (B, F, C, H, W) -> (B, F, H, W, C)
            img = image_embeds.transpose(0, 1, 3, 4, 2)
            img = img.reshape(B, num_frames // p_t, p_t, height // p, p, width // p, p, channels)
            img = img.transpose(0, 1, 3, 5, 7, 2, 4, 6)
            img = img.reshape(B, -1, channels * p_t * p * p)
            image_embeds = self.proj(img)

        embeds = mx.concatenate([text_embeds, image_embeds], axis=1)

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            seq_length = height * width * num_frames // (self.patch_size**2)
            pos_embeds = self.pos_embedding

            # Interpolate pos embeddings for variable resolution via nearest
            emb_size = embeds.shape[-1]
            pos_video = pos_embeds[:, text_seq_length:]
            pT = self.post_time_compression_frames
            pH = self.post_patch_height
            pW = self.post_patch_width
            pos_video = pos_video.reshape(1, pT, pH, pW, emb_size)

            target_H = height // self.patch_size
            target_W = width // self.patch_size

            if target_H != pH or target_W != pW:
                from mlx_arsenal.spatial import interpolate_nearest

                pos_video = interpolate_nearest(pos_video, size=(pT, target_H, target_W))

            pos_video = pos_video.reshape(1, -1, emb_size)
            pos_embeds = mx.concatenate([pos_embeds[:, :text_seq_length], pos_video], axis=1)
            pos_embeds = pos_embeds[:, : text_seq_length + seq_length]
            embeds = embeds + pos_embeds

        return embeds


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class CogVideoXBlock(nn.Module):
    """Transformer block for CogVideoX."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            eps=1e-6,
        )
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = FeedForward(dim, inner_dim=ff_inner_dim or 0, dropout=dropout, bias=ff_bias)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, mx.array]:
        text_seq_length = encoder_hidden_states.shape[1]

        # Norm + modulate
        norm_h, norm_e, gate_msa, enc_gate_msa = self.norm1(hidden_states, encoder_hidden_states, temb)

        # Attention
        attn_h, attn_e = self.attn1(
            hidden_states=norm_h,
            encoder_hidden_states=norm_e,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_h
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_e

        # Norm + modulate
        norm_h, norm_e, gate_ff, enc_gate_ff = self.norm2(hidden_states, encoder_hidden_states, temb)

        # Feed-forward on concatenated
        ff_input = mx.concatenate([norm_e, norm_h], axis=1)
        ff_output = self.ff(ff_input)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, dim: int, flip_sin_to_cos: bool = True, freq_shift: int = 0):
        super().__init__()
        self.dim = dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift

    def __call__(self, timesteps: mx.array) -> mx.array:
        half = self.dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        args = timesteps.astype(mx.float32).reshape(-1, 1) * freqs.reshape(1, -1) + self.freq_shift
        if self.flip_sin_to_cos:
            return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


class TimestepEmbedding(nn.Module):
    """Timestep embedding: Linear -> activation -> Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x, cond=None):
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


# ---------------------------------------------------------------------------
# Full Transformer Model
# ---------------------------------------------------------------------------


class CogVideoXTransformer3DModel(nn.Module):
    """CogVideoX Transformer for video generation/inpainting."""

    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        add_noise_in_inpaint_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self._config = {
            "patch_size": patch_size,
            "patch_size_t": patch_size_t,
            "use_rotary_positional_embeddings": use_rotary_positional_embeddings,
            "out_channels": out_channels,
        }

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. Transformer blocks
        self.transformer_blocks: List[CogVideoXBlock] = []
        for _ in range(num_layers):
            self.transformer_blocks.append(
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
            )

        self.norm_final = nn.LayerNorm(inner_dim, eps=norm_eps, affine=norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            elementwise_affine=norm_elementwise_affine,
            eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            output_dim = patch_size * patch_size * (out_channels or in_channels)
        else:
            output_dim = patch_size * patch_size * patch_size_t * (out_channels or in_channels)

        self.proj_out = nn.Linear(inner_dim, output_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        timestep_cond: Optional[mx.array] = None,
        inpaint_latents: Optional[mx.array] = None,
        control_latents: Optional[mx.array] = None,
        image_rotary_emb: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: (B, F, C, H, W) video latents (channels-first per frame).
            encoder_hidden_states: (B, text_len, text_dim) text embeddings.
            timestep: (B,) or scalar timestep.
            inpaint_latents: Optional (B, F, C_mask, H, W) mask+masked_video.
            image_rotary_emb: Optional (cos, sin) for RoPE.

        Returns:
            (B, F, C_out, H, W) predicted noise/velocity.
        """
        batch_size, num_frames, channels, height, width = hidden_states.shape
        p = self._config["patch_size"]
        p_t = self._config["patch_size_t"]

        local_num_frames = num_frames
        if num_frames == 1 and p_t is not None:
            hidden_states = mx.concatenate([hidden_states, mx.zeros_like(hidden_states)], axis=1)
            if inpaint_latents is not None:
                inpaint_latents = mx.concatenate([inpaint_latents, mx.zeros_like(inpaint_latents)], axis=1)
            local_num_frames = num_frames + 1

        # 1. Time embedding
        t_emb = self.time_proj(timestep)
        # Mirror reference :597 — time_proj computes fp32 by design; cast to
        # the hidden dtype before the embedding MLP or the fp32 temb
        # contaminates every block's modulation path.
        t_emb = t_emb.astype(hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        if inpaint_latents is not None:
            hidden_states = mx.concatenate([hidden_states, inpaint_latents], axis=2)
        if control_latents is not None:
            hidden_states = mx.concatenate([hidden_states, control_latents], axis=2)

        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        if not self._config["use_rotary_positional_embeddings"]:
            hidden_states = self.norm_final(hidden_states)
        else:
            hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        if p_t is None:
            out_channels = self._config["out_channels"] or channels
            output = hidden_states.reshape(batch_size, local_num_frames, height // p, width // p, out_channels, p, p)
            output = output.transpose(0, 1, 4, 2, 5, 3, 6)
            output = output.reshape(batch_size, local_num_frames, out_channels, height, width)
        else:
            out_channels = self._config["out_channels"] or channels
            output = hidden_states.reshape(
                batch_size, (local_num_frames + p_t - 1) // p_t, height // p, width // p, out_channels, p_t, p, p
            )
            output = output.transpose(0, 1, 5, 4, 2, 6, 3, 7)
            output = output.reshape(batch_size, -1, out_channels, height, width)

        if num_frames == 1:
            output = output[:, :num_frames]

        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        subfolder: str = None,
        transformer_additional_kwargs: dict = None,
    ):
        """Load from mlx-forge converted dir or PyTorch HuggingFace dir."""
        from videox_fun_mlx.utils import load_config, load_mlx_weights

        transformer_additional_kwargs = transformer_additional_kwargs or {}

        if subfolder:
            config_path = os.path.join(pretrained_model_path, subfolder)
        else:
            config_path = pretrained_model_path

        # Priority: transformer_config.json > config.json["transformer"] > config.json
        tf_config_file = os.path.join(pretrained_model_path, "transformer_config.json")
        if os.path.exists(tf_config_file):
            with open(tf_config_file) as f:
                config = json.load(f)
        else:
            config = load_config(config_path)
            # If nested config (mlx-forge root config.json), extract transformer section
            if "transformer" in config and "num_layers" not in config:
                config = config["transformer"]

        init_keys = {
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "flip_sin_to_cos",
            "freq_shift",
            "time_embed_dim",
            "text_embed_dim",
            "num_layers",
            "dropout",
            "attention_bias",
            "sample_width",
            "sample_height",
            "sample_frames",
            "patch_size",
            "patch_size_t",
            "temporal_compression_ratio",
            "max_text_seq_length",
            "activation_fn",
            "timestep_activation_fn",
            "norm_elementwise_affine",
            "norm_eps",
            "spatial_interpolation_scale",
            "temporal_interpolation_scale",
            "use_rotary_positional_embeddings",
            "use_learned_positional_embeddings",
            "patch_bias",
            "add_noise_in_inpaint_model",
        }
        filtered_config = {k: v for k, v in config.items() if k in init_keys}
        filtered_config.update(transformer_additional_kwargs)

        model = cls(**filtered_config)
        weights = load_mlx_weights(pretrained_model_path, "transformer")
        from videox_fun_mlx.utils import quantize_model_from_weights

        quantize_model_from_weights(model, weights, pretrained_model_path, "transformer")
        if "in_channels" in transformer_additional_kwargs:
            # When in_channels is overridden (e.g. VOID uses 48 vs base 33),
            # skip patch_embed.proj weights that have the wrong shape
            model_in_ch = transformer_additional_kwargs["in_channels"]
            p = filtered_config.get("patch_size", 2)
            p_t = filtered_config.get("patch_size_t", 2)
            expected_dim = model_in_ch * p * p * (p_t or 1)
            filtered_weights = {
                k: v
                for k, v in weights.items()
                if not (k == "patch_embed.proj.weight" and v.shape[-1] != expected_dim)
                and not (
                    k == "patch_embed.proj.bias"
                    and "patch_embed.proj.weight" in weights
                    and weights["patch_embed.proj.weight"].shape[-1] != expected_dim
                )
            }
            model.load_weights(list(filtered_weights.items()), strict=False)
        else:
            model.load_weights(list(weights.items()))

        leaves = nn.utils.tree_flatten(model.trainable_parameters())
        param_count = sum(v.size for _, v in leaves)
        print(f"Loaded transformer: {param_count / 1e6:.1f}M parameters")

        return model
