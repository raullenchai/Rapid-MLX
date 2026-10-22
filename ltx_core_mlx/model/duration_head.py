"""DurationHead — LTX-2.5 shot-duration predictor (port of ComfyUI duration_head.py).

Predicts the natural shot duration (in seconds) from the caption-connector
token outputs, without running the diffusion pipeline. Optional component:
``duration_head.safetensors`` (16 tensors) is loaded only when present.

Weight keys (after stripping ``duration_head.`` / ``model.diffusion_model.duration_head.``):
    video_input_proj.{weight,bias}        -- [256, 4096] video connector -> pooler dim
    audio_input_proj.{weight,bias}        -- [256, 2048] audio connector -> pooler dim
    video_modality_emb / audio_modality_emb -- [256] per-modality bias
    attention_pooler.query_tokens         -- [1, 256]
    attention_pooler.cross_attn.in_proj_weight [768, 256] (fused Q/K/V, torch MHA layout)
    attention_pooler.cross_attn.in_proj_bias   [768]
    attention_pooler.cross_attn.out_proj.{weight,bias} [256, 256]
    mlp_hidden.{weight,bias}              -- [256, 256]
    mlp_out.{weight,bias}                 -- [1, 256]

Output: ``exp()`` of the MLP head → seconds. ``seconds_to_num_frames`` snaps
to the causal VAE's ``8k + 1`` temporal grid.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class _FusedMultiheadAttention(nn.Module):
    """Torch ``nn.MultiheadAttention``-compatible fused cross-attention.

    Weight keys (under ``cross_attn.``): ``in_proj_weight`` [3*hidden, hidden]
    (rows q|k|v), ``in_proj_bias`` [3*hidden], ``out_proj.{weight,bias}``
    [hidden, hidden] — matching the LTX-2.5 duration-head checkpoint 1:1.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.in_proj_weight = mx.zeros((3 * hidden_dim, hidden_dim))
        self.in_proj_bias = mx.zeros((3 * hidden_dim,))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def __call__(self, queries: mx.array, tokens: mx.array) -> mx.array:
        """Cross-attend ``queries`` (B, Nq, H) against ``tokens`` (B, T, H)."""
        B = queries.shape[0]
        head_dim = self.hidden_dim // self.num_heads

        def _project(x: mx.array, w: mx.array, b: mx.array) -> list[mx.array]:
            proj = x @ w.T + b  # (B, T, 3*hidden)
            q, k, v = mx.split(proj, 3, axis=-1)
            return [
                t.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
                for t in (q, k, v)
            ]

        q = _project(queries, self.in_proj_weight, self.in_proj_bias)[0]
        _, k, v = _project(tokens, self.in_proj_weight, self.in_proj_bias)
        scores = (q * (head_dim**-0.5)) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(scores, axis=-1)
        pooled = attn @ v  # (B, heads, Nq, head_dim)
        pooled = pooled.transpose(0, 2, 1, 3).reshape(B, self.num_heads * head_dim)
        pooled = pooled.reshape(B, -1, self.hidden_dim)
        return self.out_proj(pooled)


class AttentionPooler(nn.Module):
    """Cross-attend ``num_queries`` learnable tokens against ``tokens``.

    Weight keys (under ``attention_pooler.``): ``query_tokens`` [Nq, hidden]
    and ``cross_attn.*`` (fused torch-MHA layout) — the official 2.5 layout.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_tokens = mx.zeros((num_queries, hidden_dim))
        self.cross_attn = _FusedMultiheadAttention(hidden_dim, num_heads)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Pool ``tokens`` (B, T, hidden) → (B, num_queries, hidden)."""
        B = tokens.shape[0]
        queries = mx.broadcast_to(self.query_tokens[None, :, :], (B, self.num_queries, self.hidden_dim))
        return self.cross_attn(queries, tokens)


class DurationHead(nn.Module):
    """Predict duration in seconds from one or both connector outputs.

    Args:
        video_cross_attention_dim: Video connector dim (4096 for LTX-2.5).
        audio_cross_attention_dim: Audio connector dim (2048 for LTX-2.5).
        pooler_hidden_dim: Pooler / MLP hidden dim (256).
        num_queries: Number of pooler queries (1).
        num_pooler_heads: Pooler attention heads (4).
        mlp_hidden: MLP hidden dim (256).
    """

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = mx.zeros((pooler_hidden_dim,))
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = mx.zeros((pooler_hidden_dim,))
        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim, num_queries=num_queries, num_heads=num_pooler_heads
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def __call__(
        self,
        video_tokens: mx.array | None = None,
        audio_tokens: mx.array | None = None,
    ) -> mx.array:
        """Predict duration.

        Args:
            video_tokens: (B, T_v, video_cross_attention_dim) connector output.
            audio_tokens: (B, T_a, audio_cross_attention_dim) connector output.
                At least one required.

        Returns:
            Duration in seconds, shape (B,).
        """
        token_groups = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        if not token_groups:
            raise ValueError("DurationHead requires at least one of video_tokens / audio_tokens")

        pooled = self.attention_pooler(mx.concatenate(token_groups, axis=1))
        pooled = pooled.reshape(pooled.shape[0], -1)
        hidden = nn.gelu_approx(self.mlp_hidden(pooled))
        return mx.exp(self.mlp_out(hidden).squeeze(-1))


def seconds_to_num_frames(
    seconds,
    frame_rate: float,
    min_seconds: float,
    max_seconds: float,
    time_scale: int = 8,
) -> int:
    """Convert seconds to a frame count on the VAE's ``8k + 1`` causal grid.

    Clamps to ``[min_seconds, max_seconds]`` and snaps (floor) to the
    ``8k + 1`` temporal grid; snapping that undershoots the minimum bumps up
    to the next grid point instead. Port of ComfyUI ``seconds_to_num_frames``.

    Args:
        seconds: Predicted duration in seconds (float or array scalar).
        frame_rate: Video frame rate (24 for LTX-2.5).
        min_seconds: Minimum allowed duration.
        max_seconds: Maximum allowed duration.
        time_scale: Causal VAE temporal grid step (8).

    Returns:
        Number of video frames.
    """
    seconds = float(seconds)
    min_frames = max(1, round(min_seconds * frame_rate))
    max_frames = round(max_seconds * frame_rate)
    raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
    frames = (raw_frames - 1) // time_scale * time_scale + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


def load_duration_head(
    model_dir,
    filename: str = "duration_head.safetensors",
    prefix: str = "duration_head.",
) -> DurationHead | None:
    """Load the optional duration head from a model dir; ``None`` if absent.

    Accepts either of the official key layouts (``duration_head.*`` or
    ``model.diffusion_model.duration_head.*``, auto-stripped) and the MLX
    split layout (``duration_head.`` prefix).

    Args:
        model_dir: Model directory containing ``duration_head.safetensors``.
        filename: Weights filename.
        prefix: Prefix to strip from the safetensors keys.

    Returns:
        A :class:`DurationHead` with weights loaded, or ``None`` when the
        weights file is not present (duration prediction is optional).
    """
    from pathlib import Path

    from ltx_core_mlx.utils.weights import load_split_safetensors

    path = Path(model_dir) / filename
    if not path.exists():
        return None

    head = DurationHead()
    raw = load_split_safetensors(path)
    weights = {}
    for key, value in raw.items():
        if key.startswith("model.diffusion_model.duration_head."):
            stripped = key[len("model.diffusion_model.duration_head.") :]
        elif key.startswith(prefix):
            stripped = key[len(prefix) :]
        elif key == "__metadata__":
            continue
        else:
            stripped = key
        weights[stripped] = value
    head.load_weights(list(weights.items()))
    return head
