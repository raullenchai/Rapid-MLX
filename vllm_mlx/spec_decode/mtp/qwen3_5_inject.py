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
3. Load the MTP weights from a separate ``mtp_sidecar`` checkpoint —
   ``mlx-community/Qwen3.5-9B-MTP-4bit`` ships the head as a 131 MB
   standalone safetensors file with top-level keys (``fc.*``,
   ``layers.0.*``, ``norm.weight``, ``pre_fc_norm_{hidden,embedding}.weight``).
4. Monkey-patch the ``TextModel`` instance's ``__class__`` to a
   subclass that adds the four MTP surfaces (``__call__`` with
   ``return_hidden``/``n_confirmed``, ``mtp_forward``,
   ``make_mtp_cache``).

Coverage scope
--------------

In-scope: the dense ``TextModel`` (``mlx_lm.models.qwen3_5.TextModel``),
its MoE subclass (``mlx_lm.models.qwen3_5_moe.Model``), and the VLM
wrapper (``mlx_lm.models.qwen3_5.Model``) where the text model is
nested under ``model.language_model``. The patch always targets the
inner ``TextModel`` — never the outer VLM wrapper (whose ``__call__``
just delegates).

Out-of-scope: ``n_confirmed`` is accepted on ``__call__`` for ABI
parity with PR #990 but is currently a no-op below the wrapper —
proper rollback of ``GatedDeltaNet`` SSM/conv state at the confirmed
boundary requires patching the layer's forward (tracked separately).
Linear-attention cache state may drift on draft rejection; this is a
known limitation that affects only the LOSSLESS contract on the
hybrid (GatedDeltaNet) attention layers, not the full-attention
layers or the throughput measurement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_inner_text_model(model: Any) -> Any:
    """Return the ``TextModel`` instance the patch must monkey-patch.

    For mlx-lm 0.31.3's Qwen3.5 architecture, ``mlx_lm.load(...)``
    returns the VLM-style ``Model`` wrapper whose ``language_model``
    field is the actual ``TextModel`` (carrying ``embed_tokens``,
    ``lm_head``, the ``model.layers`` backbone, and ``args``). The
    wrapper itself only has ``args = ModelArgs(model_type,
    text_config)`` and a delegating ``__call__`` — patching it would
    leave ``self.model.embed_tokens`` undefined for the injected
    ``mtp_forward``.

    Three shapes are accepted:

    * The outer VLM-style ``Model`` with ``model.language_model`` (real
      runtime path).
    * The inner ``TextModel`` itself (the test path constructs this
      directly to avoid the heavy VLM init).
    * A custom shell that exposes ``args`` + ``model`` and where
      ``args`` has either ``hidden_size`` (the inner-TextModel-like
      shape) or ``mtp_num_hidden_layers`` (the explicit-test shape).
      Used by ``test_inject_mtp_support_rejects_*`` paths.
    """
    # Case 1: VLM wrapper — text model lives under ``language_model``.
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm

    # Case 2: Already the inner TextModel (or a test shell). The inner
    # TextModel exposes both ``model`` (the backbone) and ``args``.
    if hasattr(model, "model") and hasattr(model, "args"):
        return model

    return None


def _detect_base_quantization(inner: Any) -> dict | None:
    """Detect the quantization params used by the base model.

    Walks the inner ``TextModel`` looking for a ``QuantizedLinear``
    instance and reads its ``bits`` / ``group_size`` / ``mode``. The
    MTP module must be quantized with the same params so its weight
    shapes match the sidecar's safetensors layout (4-bit / group_size
    64 / affine for ``mlx-community/Qwen3.5-9B-MTP-4bit``).

    Returns ``None`` for FP base models — the caller skips quantize
    in that case.
    """
    try:
        from mlx.nn import QuantizedEmbedding, QuantizedLinear
    except ImportError:  # pragma: no cover — mlx.nn always available
        return None

    backbone = getattr(inner, "model", None)
    if backbone is None:
        return None

    # Try a full-attention layer's q_proj first (always present + quantized).
    for layer in getattr(backbone, "layers", []):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            qp = layer.self_attn.q_proj
            if isinstance(qp, QuantizedLinear):
                return {
                    "bits": int(qp.bits),
                    "group_size": int(qp.group_size),
                    "mode": getattr(qp, "mode", "affine"),
                }

    # Fall back: embed_tokens (QuantizedEmbedding has bits/group_size too).
    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {
            "bits": int(embed.bits),
            "group_size": int(embed.group_size),
            "mode": getattr(embed, "mode", "affine"),
        }

    return None


def _resolve_sidecar_dir(mtp_sidecar: str | Path) -> Path | None:
    """Resolve a sidecar reference to a local directory.

    Accepts either:

    * An absolute / relative path to a directory or safetensors file
      (used by tests and operators with pre-downloaded weights).
    * An HF Hub repo name like ``mlx-community/Qwen3.5-9B-MTP-4bit``
      (downloaded via ``snapshot_download`` to the HF cache).

    Returns ``None`` if the reference cannot be resolved — caller
    treats this as a soft failure and logs.
    """
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.exists():
        return path if path.is_dir() else path.parent

    # Treat as HF repo id.
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return Path(local)
    except Exception as exc:  # pragma: no cover — network failure path
        logger.warning(
            "[mtp.inject] could not resolve sidecar %r: %s",
            mtp_sidecar,
            exc,
        )
        return None


def _find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    """Pick the safetensors file inside ``sidecar_dir`` that holds the MTP head.

    The mlx-community ``Qwen3.5-9B-MTP-4bit`` repo ships
    ``model.safetensors`` (single shard, 131 MB, 31 keys, no ``mtp.``
    prefix). Other vendors may ship ``model-mtp.safetensors`` (the
    Qwen3-Next convention used by ``add_mtp_weights.py``). Try both.
    """
    candidates = (
        sidecar_dir / "model-mtp.safetensors",
        sidecar_dir / "model.safetensors",
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def inject_mtp_support(model: Any, mtp_sidecar: str | Path | None = None) -> bool:
    """Inject MTP support into a loaded Qwen3.5 / Qwen3.6 model.

    Args:
        model: A model loaded via ``mlx_lm.load()``. Either the VLM
            wrapper ``Model`` (with ``model.language_model``) or the
            inner ``TextModel`` directly (tests pass this shape).
        mtp_sidecar: Optional reference to a separate checkpoint
            holding the MTP head's safetensors. Accepts an HF Hub
            repo id (``mlx-community/Qwen3.5-9B-MTP-4bit``) or a local
            directory path. When ``None``, the MTP module is built
            and quantized but RETAINS RANDOM INIT weights — the
            patched ``mtp_forward`` will produce useless drafts
            (accept rate ~0%). Used only by unit tests that pin
            wiring + surfaces without paying the 131 MB download
            cost. **Production callers (bench, server boot) MUST
            pass a sidecar** to get real speedup.

    Returns:
        ``True`` when the patch landed and the model now exposes
        ``mtp_forward``, ``make_mtp_cache``, ``return_hidden``, and
        ``n_confirmed`` — the four contract surfaces
        :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
        depends on. ``False`` when the model is not Qwen3.5 / 3.6,
        the config lacks ``mtp_num_hidden_layers``, or the sidecar
        cannot be resolved.

    Notes:
        ``n_confirmed`` is accepted on the patched ``__call__`` for
        ABI parity with PR #990 but does NOT thread through to the
        ``GatedDeltaNet`` SSM rollback path (that requires patching
        the layer's forward; tracked separately). This affects
        lossless on draft rejection through linear-attention layers
        only.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Install the GatedDeltaNet chunk-split patch so the verify forward
    # snapshots ``(conv_state, ssm_state)`` at ``n_confirmed`` and the
    # generator's ``_rollback_draft`` can restore them on draft
    # rejection. Without this the linear-attention layers' SSM state
    # drifts on every reject and the LOSSLESS contract fails within
    # ~10 tokens at temp=0. Idempotent — safe to call multiple times.
    from .cache_patch import (
        patch_arrays_cache_rollback_state,
        patch_gated_delta_net_for_mtp,
    )

    patch_arrays_cache_rollback_state()
    patch_gated_delta_net_for_mtp()

    inner = _resolve_inner_text_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject] model %s has neither model.language_model nor "
            "(model + args); skipping MTP injection.",
            type(model).__name__,
        )
        return False

    args = inner.args

    # 1. Resolve num_mtp_layers. Prefer the dataclass attr (which
    # tests set via object.__setattr__); fall back to the outer
    # wrapper's text_config dict (the real runtime path — mlx-lm
    # 0.31.3's TextModelArgs lacks ``mtp_num_hidden_layers`` so the
    # field gets dropped during ``BaseModelArgs.from_dict``).
    num_mtp_layers = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
    if num_mtp_layers < 1:
        outer_args = getattr(model, "args", None)
        text_config = getattr(outer_args, "text_config", None) or {}
        if isinstance(text_config, dict):
            num_mtp_layers = int(text_config.get("mtp_num_hidden_layers", 0) or 0)
        if num_mtp_layers >= 1:
            # Surface it on the dataclass so downstream code (incl.
            # validate_mtp_support, accept_counter labels) can read it
            # off ``args.mtp_num_hidden_layers`` uniformly.
            try:
                object.__setattr__(args, "mtp_num_hidden_layers", num_mtp_layers)
            except (TypeError, AttributeError):  # pragma: no cover — frozen
                pass

    if num_mtp_layers < 1:
        logger.info(
            "[mtp.inject] config has no mtp_num_hidden_layers; "
            "skipping MTP injection."
        )
        return False

    # --- Step 1: Build the MTP module from the vendored head ---
    from .head import build_mtp_module

    mtp = build_mtp_module(args, num_mtp_layers)
    logger.info(
        "[mtp.inject] Built MTP module (%d layer(s), hidden_size=%d).",
        num_mtp_layers,
        getattr(args, "hidden_size", -1),
    )

    # --- Step 2: Quantize MTP to match the base model's quantization ---
    quant_info = _detect_base_quantization(inner)
    if quant_info is not None:
        nn.quantize(
            mtp,
            group_size=quant_info["group_size"],
            bits=quant_info["bits"],
        )
        logger.info(
            "[mtp.inject] Quantized MTP: %d-bit, group_size=%d",
            quant_info["bits"],
            quant_info["group_size"],
        )

    # --- Step 3: Load MTP weights from sidecar safetensors ---
    if mtp_sidecar is not None:
        sidecar_dir = _resolve_sidecar_dir(mtp_sidecar)
        if sidecar_dir is None:
            logger.warning(
                "[mtp.inject] sidecar %r could not be resolved; "
                "skipping MTP injection.",
                mtp_sidecar,
            )
            return False
        weights_file = _find_mtp_weights_file(sidecar_dir)
        if weights_file is None:
            logger.warning(
                "[mtp.inject] no model.safetensors / model-mtp.safetensors "
                "found in %s; skipping MTP injection.",
                sidecar_dir,
            )
            return False
        raw = mx.load(str(weights_file))
        # Some sidecars (Qwen3-Next ``add_mtp_weights.py`` output) prefix
        # every key with ``mtp.``; others (mlx-community/Qwen3.5-9B-MTP-4bit)
        # store at top-level. Strip the prefix if present so both shapes
        # land on the MTP module's parameter tree.
        mtp_weights = {
            (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
            for k, v in raw.items()
        }
        # ``strict=False`` so the load tolerates unrelated keys (rare,
        # but defensive — converters occasionally bundle metadata).
        mtp.load_weights(list(mtp_weights.items()), strict=False)
        mx.eval(mtp.parameters())
        logger.info(
            "[mtp.inject] Loaded %d MTP weight tensors from %s",
            len(mtp_weights),
            weights_file.name,
        )
    else:
        # No sidecar: leave the MTP module at random init. This is the
        # test-only path — production callers MUST pass a sidecar.
        mx.eval(mtp.parameters())
        logger.warning(
            "[mtp.inject] inject_mtp_support called without mtp_sidecar — "
            "MTP head retains RANDOM init weights (accept rate ~0%%). "
            "Pass mtp_sidecar='mlx-community/Qwen3.5-9B-MTP-4bit' (or "
            "equivalent) to load real weights."
        )

    # --- Step 4: Attach + monkey-patch ``TextModel`` class ---
    inner.mtp = mtp
    original_class = type(inner)

    class _Qwen3_5WithMTP(original_class):  # type: ignore[valid-type, misc]
        """``TextModel`` + MTP surfaces injected by R15 #302 (vendor PR #990).

        The forward is inlined from
        ``mlx_lm.models.qwen3_5.Qwen3_5TextModel.__call__`` so that:

        * ``return_hidden=True`` can return the pre-norm hidden state
          the MTP head consumes (the upstream forward returns only the
          post-norm output).
        * ``n_confirmed`` is accepted on the signature for ABI parity
          with PR #990 (the generator passes ``n_confirmed=1`` during
          verify forwards). It is currently a no-op below this layer
          — the GatedDeltaNet rollback patch is tracked separately.
        """

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            from mlx_lm.models.base import create_attention_mask, create_ssm_mask

            inner_m = self.model
            if input_embeddings is not None:
                hidden_states = input_embeddings
            else:
                hidden_states = inner_m.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner_m.layers)

            # Tag each ArraysCache (linear-attention) with the
            # confirmed boundary so the patched GatedDeltaNet splits
            # ``gated_delta_update`` into two chunks and writes
            # ``(conv_snap, ssm_snap)`` to ``cache.rollback_state``.
            # KVCache slots ignore the tag — their rollback is the
            # existing ``c.trim(1)`` path. Tagged values are cleared
            # in the ``finally`` block so a later non-MTP forward
            # (mtp_forward, prefill, etc.) on the same cache list
            # doesn't accidentally re-trigger a split.
            if n_confirmed > 0:
                for c in cache:
                    if c is not None and hasattr(c, "rollback_state"):
                        c.n_confirmed_for_mtp = n_confirmed

            try:
                fa_mask = create_attention_mask(
                    hidden_states, cache[inner_m.fa_idx]
                )
                ssm_mask = create_ssm_mask(
                    hidden_states, cache[inner_m.ssm_idx]
                )
                for layer, c in zip(inner_m.layers, cache):
                    mask = ssm_mask if layer.is_linear else fa_mask
                    hidden_states = layer(hidden_states, mask=mask, cache=c)
            finally:
                if n_confirmed > 0:
                    for c in cache:
                        if c is not None and hasattr(c, "n_confirmed_for_mtp"):
                            c.n_confirmed_for_mtp = 0

            # Return PRE-norm hidden so MTP can apply its own
            # ``pre_fc_norm_hidden`` — matches PR #990's contract that
            # ``mtp_forward(hidden, ...)`` consumes pre-norm hidden.
            normed = inner_m.norm(hidden_states)
            if self.args.tie_word_embeddings:
                out = inner_m.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)

            if return_hidden:
                return out, hidden_states
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
