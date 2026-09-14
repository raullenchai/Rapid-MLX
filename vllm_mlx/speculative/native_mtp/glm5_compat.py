# SPDX-License-Identifier: Apache-2.0
"""Narrow GLM-5.3 compatibility seam for released mlx-vlm runtimes.

mlx-vlm 0.7.1 contains the corrected GLM target, temporal cache transactions,
and the MTP checkpoint loader.  Its GLM drafter still owns mutable generation
state, however, while Rapid's transaction deliberately keeps all request state
inside :class:`SpeculativeCache`.  This module replaces only that drafter call
contract.  A future upstream stateless implementation is detected
structurally and left untouched.

The stateless forward contract is adapted from the MIT-licensed implementation
in Blaizzy/mlx-vlm#2206.  The surrounding transaction and product policy remain
Rapid-owned.
"""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_INSTALLED = False
_MARKER = "_RAPID_STATELESS_GLM_MTP"


def _is_stateless_drafter(drafter_type: Any) -> bool:
    """Recognize Rapid's adapter or an upstream equivalent by call shape."""
    if getattr(drafter_type, _MARKER, False):
        return True
    try:
        import inspect

        parameters = inspect.signature(drafter_type.__call__).parameters
    except (TypeError, ValueError):
        return False
    return all(
        name in parameters
        for name in ("tokens", "hidden", "cache", "position", "target_model")
    )


def install_glm5_mtp_compatibility() -> bool:
    """Install the stateless GLM drafter adapter when upstream still needs it.

    Returns ``True`` when this call changed the runtime and ``False`` when the
    upstream implementation was already compatible or the adapter was already
    installed.  Missing or structurally different optional dependencies raise;
    the caller's capability probe converts those failures into a safe AR
    fallback.
    """
    global _INSTALLED

    with _LOCK:
        from mlx_vlm.models.linear import linear
        from mlx_vlm.speculative.drafters import glm5_next_mtp as package
        from mlx_vlm.speculative.drafters.glm5_next_mtp import (
            glm5_next_mtp as implementation,
        )

        released_type: Any = package.Glm5NextMTPDraftModel
        if _is_stateless_drafter(released_type):
            _INSTALLED = True
            return False

        class RapidGlm5NextMTPDraftModel(released_type):
            """Released checkpoint layout with request-owned forward state."""

            _RAPID_STATELESS_GLM_MTP = True

            def bind(self, target_model):
                self.validate_target_compatibility(target_model)
                return self

            def __call__(
                self,
                tokens,
                hidden,
                cache,
                position,
                target_model,
                lengths=None,
            ):
                import mlx.core as mx

                if hidden.ndim != 3 or hidden.shape[-1] != self.args.hidden_size:
                    raise ValueError(
                        "MTP hidden states must have shape "
                        "[batch, tokens, hidden_size]."
                    )
                positions = mx.array(position, dtype=mx.int32).reshape(-1, 1)
                positions = positions + mx.arange(tokens.shape[1], dtype=mx.int32)[None]
                embeddings = target_model.model.embed_tokens(tokens)
                embeddings = mx.where(positions[..., None] == 0, 0, embeddings)
                projected = linear(
                    self.eh_proj,
                    mx.concatenate(
                        [self.enorm(embeddings), self.hnorm(hidden)], axis=-1
                    ),
                )

                block = self.mtp_block
                residual = projected
                attention, _ = block.self_attn(
                    block.input_layernorm(projected),
                    cache=cache[0],
                    padding_mask=positions >= 0,
                )
                projected = residual + attention
                projected = projected + block.mlp(
                    block.post_attention_layernorm(projected)
                )
                projected = self.shared_head_norm(projected)

                head = (
                    target_model.model.embed_tokens.as_linear
                    if target_model.args.tie_word_embeddings
                    else target_model.lm_head
                )
                last = (
                    projected[:, -1:]
                    if lengths is None
                    else mx.take_along_axis(
                        projected,
                        mx.maximum(mx.array(lengths), 1)[:, None, None] - 1,
                        axis=1,
                    )
                )
                return linear(head, last), projected

        RapidGlm5NextMTPDraftModel.__name__ = "Glm5NextMTPDraftModel"
        RapidGlm5NextMTPDraftModel.__qualname__ = "Glm5NextMTPDraftModel"
        RapidGlm5NextMTPDraftModel.__module__ = implementation.__name__

        implementation.Glm5NextMTPDraftModel = RapidGlm5NextMTPDraftModel
        package.Glm5NextMTPDraftModel = RapidGlm5NextMTPDraftModel
        package.Model = RapidGlm5NextMTPDraftModel

        _INSTALLED = True
        return True


def is_installed() -> bool:
    return _INSTALLED


__all__ = [
    "install_glm5_mtp_compatibility",
    "is_installed",
]
