"""BasicAVTransformerBlock -- joint audio+video transformer block.

Ported from ltx-core/src/ltx_core/model/transformer/transformer.py

Per-block weight keys (under ``transformer_blocks.N``):
    attn1, audio_attn1               -- self-attention (video / audio)
    attn2, audio_attn2               -- text cross-attention (video / audio)
    audio_to_video_attn              -- A->V cross-modal attention
    video_to_audio_attn              -- V->A cross-modal attention
    ff, audio_ff                     -- feed-forward (video / audio)
    scale_shift_table                -- (9, video_dim)  video self-attn AdaLN
    audio_scale_shift_table          -- (9, audio_dim)  audio self-attn AdaLN
    prompt_scale_shift_table         -- (2, video_dim)  video text cross-attn
    audio_prompt_scale_shift_table   -- (2, audio_dim)  audio text cross-attn
    scale_shift_table_a2v_ca_video   -- (5, video_dim)  AV cross-attn video side
    scale_shift_table_a2v_ca_audio   -- (5, audio_dim)  AV cross-attn audio side
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core_mlx.model.transformer.attention import Attention
from ltx_core_mlx.model.transformer.feed_forward import FeedForward


class BasicAVTransformerBlock(nn.Module):
    """Joint audio+video transformer block with adaptive layer norm.

    Each block processes both video and audio tokens through:
      1. Self-attention (video & audio, with AdaLN from per-block tables + timestep)
      2. Text cross-attention (video & audio)
      3. Audio-video cross-modal attention (bidirectional)
      4. Feed-forward (video & audio)

    Args:
        video_dim: Video hidden dimension (default 4096).
        audio_dim: Audio hidden dimension (default 2048).
        video_num_heads: Number of attention heads for video (default 32).
        audio_num_heads: Number of attention heads for audio (default 32).
        video_head_dim: Per-head dimension for video (default 128).
        audio_head_dim: Per-head dimension for audio (default 64).
        av_cross_num_heads: Heads for cross-modal attention (default 32).
        av_cross_head_dim: Per-head dim for cross-modal attention (default 64).
        ff_mult: Feed-forward expansion factor.
        norm_eps: Epsilon for layer norms.
    """

    def __init__(
        self,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        video_num_heads: int = 32,
        audio_num_heads: int = 32,
        video_head_dim: int = 128,
        audio_head_dim: int = 64,
        av_cross_num_heads: int = 32,
        av_cross_head_dim: int = 64,
        ff_mult: float = 4.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        # ---- Video self-attention ----
        self.attn1 = Attention(
            query_dim=video_dim,
            num_heads=video_num_heads,
            head_dim=video_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )

        # ---- Audio self-attention ----
        self.audio_attn1 = Attention(
            query_dim=audio_dim,
            num_heads=audio_num_heads,
            head_dim=audio_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )

        # ---- Video text cross-attention ----
        self.attn2 = Attention(
            query_dim=video_dim,
            num_heads=video_num_heads,
            head_dim=video_head_dim,
            use_rope=False,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )

        # ---- Audio text cross-attention ----
        self.audio_attn2 = Attention(
            query_dim=audio_dim,
            num_heads=audio_num_heads,
            head_dim=audio_head_dim,
            use_rope=False,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )

        # ---- Audio-to-Video cross-modal attention ----
        # Q from video (query_dim=video_dim), K/V from audio (kv_dim=audio_dim)
        # Uses av_cross inner dim, output goes to video_dim
        # Reference passes cross-modal RoPE (1D temporal positions)
        self.audio_to_video_attn = Attention(
            query_dim=video_dim,
            kv_dim=audio_dim,
            out_dim=video_dim,
            num_heads=av_cross_num_heads,
            head_dim=av_cross_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )

        # ---- Video-to-Audio cross-modal attention ----
        # Q from audio (query_dim=audio_dim), K/V from video (kv_dim=video_dim)
        # Uses av_cross inner dim, output goes to audio_dim
        self.video_to_audio_attn = Attention(
            query_dim=audio_dim,
            kv_dim=video_dim,
            out_dim=audio_dim,
            num_heads=av_cross_num_heads,
            head_dim=av_cross_head_dim,
            use_rope=True,
            norm_eps=norm_eps,
            apply_gated_attention=True,
        )

        # ---- Video feed-forward ----
        self.ff = FeedForward(video_dim, dim_out=video_dim, mult=ff_mult)

        # ---- Audio feed-forward ----
        self.audio_ff = FeedForward(audio_dim, dim_out=audio_dim, mult=ff_mult)

        # ---- Scale-shift tables (raw parameters, added to timestep-computed params) ----
        # Video self-attn: 9 params (shift, scale, gate) x 3 (self-attn, text-xattn, ff)
        self.scale_shift_table = mx.zeros((9, video_dim))
        # Audio self-attn: same structure
        self.audio_scale_shift_table = mx.zeros((9, audio_dim))
        # Text cross-attention: 2 params (shift, scale) per modality
        self.prompt_scale_shift_table = mx.zeros((2, video_dim))
        self.audio_prompt_scale_shift_table = mx.zeros((2, audio_dim))
        # AV cross-attention: 5 params per side (shift_q, scale_q, shift_kv, scale_kv, gate)
        self.scale_shift_table_a2v_ca_video = mx.zeros((5, video_dim))
        self.scale_shift_table_a2v_ca_audio = mx.zeros((5, audio_dim))

        self._norm_eps = norm_eps

    @staticmethod
    def _unpack_adaln(params: mx.array, table: mx.array, num_params: int, dim: int) -> list[mx.array]:
        """Unpack AdaLN parameters and add scale_shift_table.

        Handles both scalar (B, P*dim) and per-token (B, N, P*dim) inputs.

        Returns:
            List of P arrays, each broadcastable with (B, N, dim):
            - Scalar mode: (B, 1, dim)
            - Per-token mode: (B, N, dim)
        """
        if params.ndim == 2:
            # Scalar: (B, P*dim) → (B, P, dim)
            p = params.reshape(-1, num_params, dim)
            p = p + table[None, :num_params, :]
            return [p[:, i, :][:, None, :] for i in range(num_params)]
        else:
            # Per-token: (B, N, P*dim) → (B, N, P, dim)
            B, N, _ = params.shape
            p = params.reshape(B, N, num_params, dim)
            p = p + table[None, None, :num_params, :]
            return [p[:, :, i, :] for i in range(num_params)]

    def _rms_norm(self, x: mx.array) -> mx.array:
        """Affine-free RMS norm (no mean subtraction, matches reference rms_norm)."""
        return mx.fast.rms_norm(x, weight=None, eps=self._norm_eps)

    def compute_video_normed_sa(
        self,
        video_hidden: mx.array,
        video_adaln_params: mx.array,
    ) -> mx.array:
        """Modulated video input to self-attention — the TeaCache gate signal.

        Mirrors the first two ops of ``__call__`` (the line that computes
        ``video_normed = rms(x) * (1 + scale_sa) + shift_sa``) without running
        attention. Used for cheap probe / cache-decision computations.

        Args:
            video_hidden: (B, Nv, video_dim) post-patchify hidden states.
            video_adaln_params: (B, 9*video_dim) or (B, Nv, 9*video_dim) AdaLN
                parameters for self-attn / ff / text-xattn (indices 0..2 are
                self-attn shift/scale/gate).

        Returns:
            Modulated input ``rms(video_hidden) * (1 + scale_sa) + shift_sa``,
            shape ``(B, Nv, video_dim)``.
        """
        vdim = video_hidden.shape[-1]
        v_shift_sa, v_scale_sa, *_ = self._unpack_adaln(video_adaln_params, self.scale_shift_table, 9, vdim)
        return self._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa

    def __call__(
        self,
        video_hidden: mx.array,
        audio_hidden: mx.array,
        video_adaln_params: mx.array,
        audio_adaln_params: mx.array,
        video_prompt_adaln_params: mx.array,
        audio_prompt_adaln_params: mx.array,
        av_ca_video_params: mx.array,
        av_ca_audio_params: mx.array,
        av_ca_a2v_gate_params: mx.array,
        av_ca_v2a_gate_params: mx.array,
        video_text_embeds: mx.array | None = None,
        audio_text_embeds: mx.array | None = None,
        video_rope_freqs: mx.array | None = None,
        audio_rope_freqs: mx.array | None = None,
        video_cross_rope_freqs: mx.array | None = None,
        audio_cross_rope_freqs: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
        block_idx: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass for joint audio+video block.

        Args:
            video_hidden: (B, Nv, video_dim).
            audio_hidden: (B, Na, audio_dim).
            video_adaln_params: (B, 9 * video_dim) from top-level adaln_single.
            audio_adaln_params: (B, 9 * audio_dim) from top-level audio_adaln_single.
            video_prompt_adaln_params: (B, 2 * video_dim) for text cross-attn.
            audio_prompt_adaln_params: (B, 2 * audio_dim) for text cross-attn.
            av_ca_video_params: (B, 4 * video_dim) for AV cross-attn video side.
            av_ca_audio_params: (B, 4 * audio_dim) for AV cross-attn audio side.
            av_ca_a2v_gate_params: (B, video_dim) gate for A->V.
            av_ca_v2a_gate_params: (B, audio_dim) gate for V->A.
            video_text_embeds: (B, Nt, video_dim) text embeddings for video.
            audio_text_embeds: (B, Nt, audio_dim) text embeddings for audio.
            video_rope_freqs: RoPE frequencies for video.
            audio_rope_freqs: RoPE frequencies for audio.
            video_attention_mask: Attention mask for video self-attention.
            audio_attention_mask: Attention mask for audio self-attention.
            perturbations: Optional perturbation config for STG guidance.
                When provided, generates masks that zero out attention outputs
                for perturbed samples in the batch.
            block_idx: Index of this block in the transformer stack, used for
                per-block perturbation lookup.

        Returns:
            Tuple of (video_hidden, audio_hidden).
        """
        # --- Compute block-level modulation by ADDING tables to timestep params ---
        # 9 params: [0-2]=self-attn, [3-5]=ff, [6-8]=text-xattn (reference ordering)
        vdim = video_hidden.shape[-1]
        adim = audio_hidden.shape[-1]

        (v_shift_sa, v_scale_sa, v_gate_sa, v_shift_ff, v_scale_ff, v_gate_ff, v_shift_ca, v_scale_ca, v_gate_ca) = (
            self._unpack_adaln(video_adaln_params, self.scale_shift_table, 9, vdim)
        )

        (a_shift_sa, a_scale_sa, a_gate_sa, a_shift_ff, a_scale_ff, a_gate_ff, a_shift_ca, a_scale_ca, a_gate_ca) = (
            self._unpack_adaln(audio_adaln_params, self.audio_scale_shift_table, 9, adim)
        )

        # AV cross-attn tables: 4 scale/shift params + 1 gate per side
        # Reference ordering: [0]=scale_a2v, [1]=shift_a2v, [2]=scale_v2a, [3]=shift_v2a, [4]=gate
        (av_v_scale_a2v, av_v_shift_a2v, av_v_scale_v2a, av_v_shift_v2a) = self._unpack_adaln(
            av_ca_video_params, self.scale_shift_table_a2v_ca_video, 4, vdim
        )
        # AV gate: scalar (B, dim) or per-token (B, N, dim)
        if av_ca_a2v_gate_params.ndim == 2:
            av_v_gate_a2v = (av_ca_a2v_gate_params + self.scale_shift_table_a2v_ca_video[4, :])[:, None, :]
        else:
            av_v_gate_a2v = av_ca_a2v_gate_params + self.scale_shift_table_a2v_ca_video[None, None, 4, :]

        (av_a_scale_a2v, av_a_shift_a2v, av_a_scale_v2a, av_a_shift_v2a) = self._unpack_adaln(
            av_ca_audio_params, self.scale_shift_table_a2v_ca_audio, 4, adim
        )
        if av_ca_v2a_gate_params.ndim == 2:
            av_a_gate_v2a = (av_ca_v2a_gate_params + self.scale_shift_table_a2v_ca_audio[4, :])[:, None, :]
        else:
            av_a_gate_v2a = av_ca_v2a_gate_params + self.scale_shift_table_a2v_ca_audio[None, None, 4, :]

        # --- 1. Video self-attention ---
        video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_sa) + v_shift_sa
        # STG: pass perturbation mask into attention so it blends with values
        v_ptb_mask = None
        if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, block_idx):
            v_ptb_mask = perturbations.mask_like(
                PerturbationType.SKIP_VIDEO_SELF_ATTN, block_idx, video_hidden[:, :1, :1, None]
            )  # (B, 1, 1, 1)
        video_sa_out = self.attn1(
            video_normed,
            rope_freqs=video_rope_freqs,
            attention_mask=video_attention_mask,
            perturbation_mask=v_ptb_mask,
        )
        video_hidden = video_hidden + video_sa_out * v_gate_sa

        # --- 2. Audio self-attention ---
        audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_sa) + a_shift_sa
        a_ptb_mask = None
        if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, block_idx):
            a_ptb_mask = perturbations.mask_like(
                PerturbationType.SKIP_AUDIO_SELF_ATTN, block_idx, audio_hidden[:, :1, :1, None]
            )
        audio_sa_out = self.audio_attn1(
            audio_normed,
            rope_freqs=audio_rope_freqs,
            attention_mask=audio_attention_mask,
            perturbation_mask=a_ptb_mask,
        )
        audio_hidden = audio_hidden + audio_sa_out * a_gate_sa

        # --- 3. Video text cross-attention (AdaLN indices 6-8 + prompt table for KV) ---
        if video_text_embeds is not None:
            video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_ca) + v_shift_ca
            vp_shift, vp_scale = self._unpack_adaln(video_prompt_adaln_params, self.prompt_scale_shift_table, 2, vdim)
            text_scaled = video_text_embeds * (1.0 + vp_scale) + vp_shift
            video_hidden = video_hidden + self.attn2(video_normed, encoder_hidden_states=text_scaled) * v_gate_ca

        # --- 4. Audio text cross-attention ---
        if audio_text_embeds is not None:
            audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_ca) + a_shift_ca
            ap_shift, ap_scale = self._unpack_adaln(
                audio_prompt_adaln_params, self.audio_prompt_scale_shift_table, 2, adim
            )
            text_scaled = audio_text_embeds * (1.0 + ap_scale) + ap_shift
            audio_hidden = audio_hidden + self.audio_attn2(audio_normed, encoder_hidden_states=text_scaled) * a_gate_ca

        # --- 5-6. Audio-Video cross-modal attention ---
        # Reference: norm both modalities ONCE, shared by both A2V and V2A
        video_norm3 = self._rms_norm(video_hidden)
        audio_norm3 = self._rms_norm(audio_hidden)

        # A2V: Q from video, KV from audio
        # Reference applies perturbation mask OUTSIDE attention (attn * gate * mask),
        # not inside (unlike self-attention which blends with values).
        video_q_a2v = video_norm3 * (1.0 + av_v_scale_a2v) + av_v_shift_a2v
        audio_kv_a2v = audio_norm3 * (1.0 + av_a_scale_a2v) + av_a_shift_a2v
        a2v_out = (
            self.audio_to_video_attn(
                video_q_a2v,
                encoder_hidden_states=audio_kv_a2v,
                rope_freqs=video_cross_rope_freqs,
                rope_freqs_k=audio_cross_rope_freqs,
            )
            * av_v_gate_a2v
        )
        if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, block_idx):
            a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, block_idx, video_hidden)
            a2v_out = a2v_out * a2v_mask
        video_hidden = video_hidden + a2v_out

        # V2A: Q from audio, KV from video (using pre-A2V norms)
        audio_q_v2a = audio_norm3 * (1.0 + av_a_scale_v2a) + av_a_shift_v2a
        video_kv_v2a = video_norm3 * (1.0 + av_v_scale_v2a) + av_v_shift_v2a
        v2a_out = (
            self.video_to_audio_attn(
                audio_q_v2a,
                encoder_hidden_states=video_kv_v2a,
                rope_freqs=audio_cross_rope_freqs,
                rope_freqs_k=video_cross_rope_freqs,
            )
            * av_a_gate_v2a
        )
        if perturbations is not None and perturbations.any_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, block_idx):
            v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, block_idx, audio_hidden)
            v2a_out = v2a_out * v2a_mask
        audio_hidden = audio_hidden + v2a_out

        # --- 7. Video feed-forward (uses indices 3-5) ---
        video_normed = self._rms_norm(video_hidden) * (1.0 + v_scale_ff) + v_shift_ff
        video_hidden = video_hidden + self.ff(video_normed) * v_gate_ff

        # --- 8. Audio feed-forward (uses indices 3-5) ---
        audio_normed = self._rms_norm(audio_hidden) * (1.0 + a_scale_ff) + a_shift_ff
        audio_hidden = audio_hidden + self.audio_ff(audio_normed) * a_gate_ff

        return video_hidden, audio_hidden
