# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for Qwen3.5 / Qwen3.6 models (vendor PR #990).

mlx-lm 0.31.3 does not yet ship PR #990, so its
``mlx_lm.models.qwen3_5.TextModel.__call__`` does not accept
``return_hidden`` or ``n_confirmed`` and the class has no
``mtp_forward`` / ``make_mtp_cache`` methods. Without those four
surfaces, :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
can't drive the model.

This module mirrors the pattern from
:mod:`vllm_mlx.patches.qwen3_next_mtp` (the existing Qwen3-Next
runtime injection used by ``--enable-mtp``):

1. Construct the MTP module that PR #990 adds to ``TextModel`` —
   delegated to :func:`vllm_mlx.spec_decode.mtp.head.build_mtp_module`.
2. Quantize the MTP module to match the base model's quantization (so
   the weight tensors land in the right shape for ``load_weights``).
3. Load the ``mtp.*`` weights from ``model.safetensors`` (or
   ``model-mtp.safetensors`` when the converter split them out).
4. Monkey-patch the ``TextModel`` instance's ``__class__`` to a
   subclass that adds the four MTP surfaces.

Coverage scope
--------------

In-scope: the dense ``TextModel`` (``mlx_lm.models.qwen3_5.TextModel``)
and its MoE subclass (``mlx_lm.models.qwen3_5_moe.Model``). Both
expose ``self.model`` (the underlying ``Qwen3_5TextModel``),
``self.args``, ``self.lm_head`` (or tied embeddings), and ``self.layers``
— enough for the patch to wire up identically.

Out-of-scope: VLM wrappers (no Qwen3.5/3.6 VLM has shipped MTP weights
yet), pipeline-parallel splits (PR #990 supports them upstream but
adapting the patch is a follow-up — the operator can boot single-
device for MTP in the meantime).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_inner_text_model(model: Any) -> Any:
    """Return the ``TextModel`` instance the patch must monkey-patch.

    The dense Qwen3.5 / 3.6 model is itself the ``TextModel``. The MoE
    model subclasses ``Qwen3_5Model`` (which IS the dense TextModel)
    so this is just ``model``. A VLM wrapper would expose
    ``model.language_model`` — we don't support that path yet but the
    shape check is here so the failure mode is a clear log line
    rather than an AttributeError mid-forward.
    """
    if hasattr(model, "model") and hasattr(model, "args"):
        return model

    inner = getattr(model, "language_model", None)
    if inner is not None and hasattr(inner, "args"):
        logger.warning(
            "[mtp.inject] Qwen3.5/3.6 + VLM wrapper not yet supported. "
            "MTP is only wired for the text-only model class."
        )
        return None

    return None


def inject_mtp_support(model: Any) -> bool:
    """Inject MTP support into a loaded Qwen3.5 / Qwen3.6 model.

    Args:
        model: A model loaded via ``mlx_lm.load()``. Must be a
            ``mlx_lm.models.qwen3_5.TextModel`` or its MoE subclass
            from ``qwen3_5_moe.Model``. Other architectures raise
            no-op early (returns ``False``).

    Returns:
        ``True`` when the patch landed and the model now exposes
        ``mtp_forward``, ``make_mtp_cache``, ``return_hidden``, and
        ``n_confirmed`` — the four contract surfaces
        :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
        depends on. ``False`` when the model is not Qwen3.5 / 3.6 or
        the MTP weights are not present in the checkpoint (operator
        must re-convert from HF with PR #990's ``sanitize()`` path
        that preserves ``mtp.*`` keys).

    Notes:
        The function does NOT touch the ``ArraysCache.rollback_state``
        slot — that patch is installed once at module import time by
        :func:`vllm_mlx.spec_decode.mtp.cache_patch.patch_arrays_cache_rollback_state`,
        and is independent of which model instance is patched here.
    """
    import mlx.core as mx

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject] model %s is not a Qwen3.5/3.6 TextModel; "
            "skipping MTP injection.",
            type(model).__name__,
        )
        return False

    args = inner.args
    num_mtp_layers = getattr(args, "mtp_num_hidden_layers", 0) or 0
    if num_mtp_layers < 1:
        logger.info(
            "[mtp.inject] config has mtp_num_hidden_layers=%d; "
            "skipping MTP injection (re-convert from HF with PR #990 "
            "sanitize() path to preserve mtp.* weights).",
            num_mtp_layers,
        )
        return False

    # --- Step 1: Build the MTP module from the head vendored module ---
    from .head import build_mtp_module

    mtp = build_mtp_module(args, num_mtp_layers)
    logger.info(
        "[mtp.inject] Built MTP module (%d layer(s), hidden_size=%d).",
        num_mtp_layers,
        getattr(args, "hidden_size", -1),
    )

    # --- Step 2: Attach to the inner model so weight load can resolve
    # ``mtp.*`` keys against ``model.model.mtp.*`` paths the converter
    # writes out. ``model.load_weights`` then routes them naturally.
    inner.mtp = mtp

    # --- Step 3: Monkey-patch the model class to add the four MTP
    # surfaces. We subclass ``type(model)`` so the patch composes with
    # any other runtime patches (e.g. quantization wrappers).
    original_class = type(inner)

    class _Qwen3_5WithMTP(original_class):  # type: ignore[valid-type, misc]
        """``TextModel`` + MTP surfaces injected by R15 #302 (vendor PR #990)."""

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            # Delegate to the inner ``Qwen3_5TextModel`` so the
            # backbone runs unchanged, but route ``n_confirmed``
            # through every layer (so GatedDeltaNet's
            # ``_process_chunk`` split fires at the right boundary).
            hidden = self.model(
                inputs,
                cache=cache,
                input_embeddings=input_embeddings,
                n_confirmed=n_confirmed,
            )
            normed = self.model.norm(hidden)
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)
            if return_hidden:
                return out, hidden
            return out

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache,
        ):
            """Run the MTP head and project through the shared lm_head."""
            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(mtp_out)
            return self.lm_head(mtp_out)

        def make_mtp_cache(self):
            """Return fresh ``KVCache`` entries — one per MTP layer."""
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _Qwen3_5WithMTP
    mx.eval(mtp.parameters())
    logger.info(
        "[mtp.inject] Patched %s with MTP surfaces "
        "(return_hidden, n_confirmed, mtp_forward, make_mtp_cache).",
        original_class.__name__,
    )
    return True


def validate_mtp_support(model: Any) -> bool:
    """Verify that ``inject_mtp_support`` succeeded on ``model``.

    Used by the CLI's boot-time MTP wiring: the operator gets a
    clear warning if the injection silently dropped MTP rather than
    discovering it mid-generation when the first ``mtp_forward`` call
    raises ``AttributeError``.

    Checks:

    1. Model has ``mtp`` attribute (or ``model.mtp`` for the dense
       variant).
    2. ``mtp_forward`` is callable.
    3. ``make_mtp_cache`` is callable.
    4. ``__call__`` accepts ``return_hidden`` and ``n_confirmed``.
    """
    import inspect

    inner = _resolve_inner_text_model(model)
    if inner is None:
        return False

    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning("[mtp.validate] model.__call__ does not accept return_hidden.")
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning("[mtp.validate] model.__call__ does not accept n_confirmed.")
        return False
    return True
