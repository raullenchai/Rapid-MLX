"""``Modality`` — input bundle for one of (video, audio) in the LTX-2 DiT.

Ported from ``ltx_core.model.transformer.modality``. Bundles all the
per-token state needed for one modality of a transformer forward:
latent, sigma, per-token timesteps, positions, text context, and the
optional masks that conditioning items might attach.

In the upstream pipeline this dataclass is the canonical interface for
passing data into and out of helpers like
:class:`~ltx_core_mlx.components.modality_tiling.VideoModalityTiler`.
Our pipelines currently call :meth:`LTXModel.__call__` with these
fields as separate kwargs; the helper layer wraps/unwraps Modality at
its boundaries so the math matches upstream verbatim.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Modality:
    """Input data for a single modality (video or audio) in the transformer.

    Attributes:
        latent: Patchified latent tokens, shape ``(B, T, D)`` where *B*
            is the batch size, *T* is the total number of tokens
            (generated + appended conditioning), and *D* is the input
            dimension.
        sigma: ``(B,)``. Current sigma value, used for cross-attention
            timestep calculations.
        timesteps: Per-token timestep embeddings, shape ``(B, T)``.
        positions: Positional coordinates, shape ``(B, 3, T)`` for
            video (time, height, width) or ``(B, 1, T)`` for audio.
        context: Text conditioning embeddings from the prompt encoder.
        enabled: Whether this modality is active in the current
            forward pass.
        context_mask: Optional mask for the text context tokens.
        attention_mask: Optional ``(B, T, T)`` self-attention mask in
            ``[0, 1]``. ``None`` means full attention between all
            tokens. Built incrementally by conditioning items.
    """

    latent: mx.array
    sigma: mx.array
    timesteps: mx.array
    positions: mx.array
    context: mx.array
    enabled: bool = True
    context_mask: mx.array | None = None
    attention_mask: mx.array | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes.

        Mirrors upstream's ``Modality.split``. Used by guidance code
        that batches multiple guidance variants (cond / neg / ptb /
        mod) and needs to break them apart again.
        """
        offsets: list[int] = []
        acc = 0
        for s in sizes:
            offsets.append(acc)
            acc += s

        split_fields: dict[str, list[mx.array | bool | None]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, mx.array):
                split_fields[f.name] = [value[off : off + sz] for off, sz in zip(offsets, sizes, strict=True)]
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * len(sizes)
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(len(sizes))]
