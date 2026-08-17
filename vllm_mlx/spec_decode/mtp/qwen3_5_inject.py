# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for Qwen3.5 / Qwen3.6 models (vendor PR #990).

mlx-lm 0.31.3 does not yet ship PR #990, so its
``mlx_lm.models.qwen3_5.TextModel.__call__`` does not accept
``return_hidden`` or ``n_confirmed`` and the class has no
``mtp_forward`` / ``make_mtp_cache`` methods. Without those four
surfaces, :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
can't drive the model.

This module mirrors the pattern from
:mod:`vllm_mlx.patches.qwen3_next_mtp` (the Qwen3-Next runtime injection):

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

``n_confirmed`` rollback: implemented as of this PR. ``__call__``
accepts ``n_confirmed`` and threads it through to each
``ArraysCache`` via ``n_confirmed_for_mtp`` before the forward, so
the patched ``GatedDeltaNet.__call__`` (installed by
``patch_gated_delta_net_for_mtp``) can snapshot ``(conv_state,
ssm_state)`` AT the confirmed-token boundary. On draft rejection the
generator's ``_rollback_draft`` restores the snapshot per cache
instance. Lossless contract confirmed byte-equal × 3 profiles on
mlx-community/Qwen3.5-9B-4bit + mlx-community/Qwen3.5-9B-MTP-4bit
(see ``tests/test_mtp_real_weights.py``).
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
    instance and reads its ``bits`` / ``group_size``. The MTP module
    must be quantized with the same params so its weight shapes match
    the sidecar's safetensors layout (4-bit / group_size 64 / affine
    for ``mlx-community/Qwen3.5-9B-MTP-4bit``).

    Returns ``None`` for FP base models — the caller skips quantize
    in that case.

    NOTE: only ``bits`` + ``group_size`` are returned. ``nn.quantize``
    in the mlx-lm versions we target does not accept a ``mode`` arg —
    it always applies the affine mode. The mlx-community sidecars
    similarly assume affine. Returning ``mode`` would be dead data
    that callers cannot pass through, so it's dropped. If/when
    ``nn.quantize`` grows mode support, extend the dict here and
    pipe it through at the inject call-site.
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
                }

    # Fall back: embed_tokens (QuantizedEmbedding has bits/group_size too).
    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {
            "bits": int(embed.bits),
            "group_size": int(embed.group_size),
        }

    return None


def _warn_if_mtp_quantization_mismatch(
    base_quant: dict | None, sidecar_quant: dict | None
) -> None:
    """Recommend matched MTP precision when both quantizations are known."""
    if base_quant is None or sidecar_quant is None or base_quant == sidecar_quant:
        return

    logger.warning(
        "[mtp.inject] MTP sidecar quantization (%d-bit, group_size=%d) "
        "differs from base model (%d-bit, group_size=%d): pairing effects "
        "are model-dependent: slower than no speculation on Qwen3.6 "
        "(#1258), faster on Qwen3.8-27B. Benchmark your pairing.",
        sidecar_quant["bits"],
        sidecar_quant["group_size"],
        base_quant["bits"],
        base_quant["group_size"],
    )


# MLX affine quantization's practical value set. Every mlx-community
# checkpoint (base + MTP sidecar alike) uses one of these — the shipped
# Qwen3.6 pairing is 4-/8-bit at group_size 64. A sidecar whose tensors
# imply a ``(bits, group_size)`` outside this set is anomalous (a
# hand-corrupted or unknown packing); we refuse it rather than feed an
# odd width into ``nn.quantize``. Kept deliberately narrow: over-inclusion
# re-opens the exact mismatch this fix closes, while a genuinely-new width
# simply needs one entry added here.
_MLX_AFFINE_BITS = frozenset({2, 3, 4, 5, 6, 8})
_MLX_AFFINE_GROUP_SIZES = frozenset({32, 64, 128})


def _infer_sidecar_fc_quantization(
    mtp_weights: dict, fc_out_dims: int, fc_in_dims: int
) -> dict | None:
    """Infer the sidecar fc layer's quantization from its LOADED tensors.

    The sidecar tensors are the ground truth for how the checkpoint is
    packed — strictly more reliable than any ``config.json`` proxy, which
    a checkpoint may omit, mis-declare, or leave incomplete. The MTP
    head's ``fc`` is an ``nn.Linear(2H, H)``; once MLX affine-quantizes
    it, the packed ``fc.weight`` has shape ``(out, in * bits // 32)`` and
    ``fc.scales`` has shape ``(out, in // group_size)``. Given the known
    full-precision dims ``(out, in)`` read off the freshly-built module,
    inverting those two shapes recovers ``(bits, group_size)`` exactly.

    The MTP module must be quantized to match the sidecar it loads: a
    QuantizedLinear's packed ``weight`` / ``scales`` shapes are a
    function of ``(bits, group_size)``, so quantizing the module to a
    DIFFERENT width than the sidecar tensors leaves a packing the layer's
    ``bits`` attr disagrees with, and ``mx.quantized_matmul`` raises
    "weight and scales incompatible" at the first MTP draft step —
    surfacing to the client as an intermittent EMPTY response
    (``prompt_tokens=0``). Reading the packing from the tensors fixes it
    at the true source.

    Returns:
      * ``{"bits", "group_size"}`` — ``fc.scales`` is present, so the fc
        is quantized; the affine params derived from the packed shapes
        (validated against MLX's supported set + shape self-consistency).
      * ``None`` — no ``fc.scales`` tensor, so the fc is full-precision;
        the caller keeps the MTP module FP.

    Raises:
      ``ValueError`` — the sidecar's fc packing cannot be trusted: either
      ``fc.scales`` is present but the shapes are inconsistent / the
      derived ``(bits, group_size)`` are outside MLX affine's supported
      set, OR ``fc.scales`` is absent yet a present ``fc.weight`` does
      NOT carry the full-precision ``(out, in)`` shape (a truncated
      quantized sidecar). The caller refuses injection rather than
      mis-pack the module.
    """
    scales = mtp_weights.get("fc.scales")
    if scales is None:
        # No per-group scales → the fc is full-precision. Guard against a
        # TRUNCATED quantized sidecar that shipped a packed ``fc.weight``
        # but lost its ``fc.scales``: if ``fc.weight`` is present it MUST
        # carry the exact full-precision ``(out, in)`` shape, else a
        # packed weight would be loaded into an ``nn.Linear`` and crash
        # at inference. A *missing* ``fc.weight`` is left to the
        # downstream coverage check (the module stays FP and the check
        # refuses the partial head).
        fp_weight = mtp_weights.get("fc.weight")
        if fp_weight is not None:
            fp_shape = tuple(int(d) for d in fp_weight.shape)
            if fp_shape != (fc_out_dims, fc_in_dims):
                raise ValueError(
                    f"fc has no scales but fc.weight shape {fp_shape} != "
                    f"full-precision ({fc_out_dims}, {fc_in_dims}) — a "
                    f"truncated quantized sidecar (packed weight, missing "
                    f"scales)"
                )
        return None
    weight = mtp_weights.get("fc.weight")
    if weight is None:
        raise ValueError("fc.scales present but fc.weight missing")
    w_shape = tuple(int(d) for d in weight.shape)
    s_shape = tuple(int(d) for d in scales.shape)
    if len(w_shape) != 2 or len(s_shape) != 2:
        raise ValueError(f"unexpected fc tensor ranks: weight{w_shape} scales{s_shape}")
    if w_shape[0] != fc_out_dims or s_shape[0] != fc_out_dims:
        raise ValueError(
            f"fc out-dim mismatch: weight{w_shape} scales{s_shape} "
            f"vs module out={fc_out_dims}"
        )
    packed_cols, scale_cols = w_shape[1], s_shape[1]
    if scale_cols <= 0 or packed_cols <= 0:
        raise ValueError(
            f"non-positive fc packing cols: weight{w_shape} scales{s_shape}"
        )
    if fc_in_dims % scale_cols != 0:
        raise ValueError(
            f"in-dim {fc_in_dims} not divisible by scales cols {scale_cols}"
        )
    group_size = fc_in_dims // scale_cols
    if (32 * packed_cols) % fc_in_dims != 0:
        raise ValueError(
            f"packed weight cols {packed_cols} inconsistent with in-dim {fc_in_dims}"
        )
    bits = (32 * packed_cols) // fc_in_dims
    if bits not in _MLX_AFFINE_BITS or group_size not in _MLX_AFFINE_GROUP_SIZES:
        raise ValueError(
            f"derived bits={bits} group_size={group_size} outside MLX affine set"
        )
    # Cross-check: the packing must be exactly reproducible from the
    # derived params (guards against a coincidental divisibility match).
    if packed_cols != fc_in_dims * bits // 32 or scale_cols != fc_in_dims // group_size:
        raise ValueError(
            f"inconsistent packing: weight{w_shape} scales{s_shape} "
            f"do not reproduce from bits={bits} group_size={group_size}"
        )
    return {"bits": bits, "group_size": group_size}


def _resolve_sidecar_file(mtp_sidecar: str | Path) -> Path | None:
    """Resolve a sidecar reference to a concrete safetensors file path.

    Accepts:

    * An absolute / relative path to a directory containing a
      ``model.safetensors``, ``model-mtp.safetensors``, or
      ``mtp/model.safetensors`` file
      (operators with a pre-downloaded HF snapshot).
    * An absolute / relative path to a ``*.safetensors`` file
      directly (operators with a hand-assembled sidecar; the
      filename does NOT have to be one of the two well-known
      names).
    * An HF Hub repo name like ``mlx-community/Qwen3.5-9B-MTP-4bit``
      (downloaded via ``snapshot_download`` to the HF cache, then
      probed for the layouts above). The nested ``mtp/`` layout lets a
      single repository contain target and drafter weights without
      ``mlx_lm.load`` mistaking the drafter for a target-model shard.

    Returns ``None`` if the reference cannot be resolved — caller
    treats this as a soft failure and logs.
    """
    if mtp_sidecar is None:
        return None

    path = Path(mtp_sidecar)
    if path.is_file():
        # Explicit file path — use it verbatim. Supports operator
        # workflows where the sidecar lives at a custom filename
        # (``mtp-q4-g64.safetensors``, ``qwen3_5_mtp_head.safetensors``,
        # …). Skipping the well-known-name probe avoids the silent
        # "file is at a non-default name → fall back to None" trap
        # codex flagged on PR #954 review.
        return path
    if path.is_dir():
        return _find_mtp_weights_file(path)

    # Treat as HF repo id.
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar))
        return _find_mtp_weights_file(Path(local))
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
        sidecar_dir / "mtp" / "model.safetensors",
        sidecar_dir / "model.safetensors",
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def inject_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    """Inject MTP support into a loaded Qwen3.5 / Qwen3.6 model.

    Args:
        model: A model loaded via ``mlx_lm.load()``. Either the VLM
            wrapper ``Model`` (with ``model.language_model``) or the
            inner ``TextModel`` directly (tests pass this shape).
        mtp_sidecar: Optional reference to a separate checkpoint
            holding the MTP head's safetensors. Accepts an HF Hub
            repo id (``mlx-community/Qwen3.5-9B-MTP-4bit``), a local
            directory path, or a direct path to a ``.safetensors``
            file.
        allow_random_init: When ``True``, permit ``mtp_sidecar=None``
            and ship the MTP head with its RANDOM INIT weights (the
            patched ``mtp_forward`` produces useless drafts, accept
            rate ~0%). Test-only. Codex flagged on PR #954 that
            allowing this by default lets production callers silently
            enable a useless/slow draft model, so the default is
            ``False`` — a missing sidecar in production now returns
            ``False`` from this function and the model is left
            unmodified. The bench, server boot, and the rapid-mlx
            spec_decode pipeline MUST pass a sidecar.

    Returns:
        ``True`` when the patch landed and the model now exposes
        ``mtp_forward``, ``make_mtp_cache``, ``return_hidden``, and
        ``n_confirmed`` — the four contract surfaces
        :func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step`
        depends on. ``False`` when the model is not Qwen3.5 / 3.6,
        the config lacks ``mtp_num_hidden_layers``, the sidecar
        cannot be resolved, or ``mtp_sidecar`` is ``None`` and
        ``allow_random_init`` is ``False``.

    Notes:
        This function is NEW in this PR (Qwen3.5 native MTP). It is
        NOT the legacy ``vllm_mlx.patches.qwen3_next_mtp.inject_mtp_support``
        used by the scheduler (different signature, different model
        family, different load path). The only production caller of
        this function is ``bench/bench_spec_decode_mtp.py`` (which
        already passes ``mtp_sidecar``). There are no pre-existing
        bare ``inject_mtp_support(model)`` call-sites to break with
        the new ``allow_random_init=False`` default.

        ``n_confirmed`` rollback is implemented as of this PR: it
        threads through to each ``ArraysCache`` via
        ``n_confirmed_for_mtp`` before forward, so the patched
        ``GatedDeltaNet.__call__`` (installed by
        ``patch_gated_delta_net_for_mtp``) can snapshot
        ``(conv_state, ssm_state)`` AT the confirmed-token boundary.
    """
    import mlx.core as mx
    import mlx.nn as nn

    # NOTE: the global ``ArraysCache`` rollback_state class-default and
    # the ``GatedDeltaNet.__call__`` chunk-split patches are deferred
    # until AFTER every can-fail validation completes (see ``# --- Step
    # 5`` below). Codex flagged on PR #954 that installing these
    # monkey-patches up-front meant a failed sidecar load left
    # process-global behavior mutated even though inject_mtp_support
    # returned False. The patches are now strictly post-validation.

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
            "[mtp.inject] config has no mtp_num_hidden_layers; skipping MTP injection."
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

    # --- Step 2: Resolve the sidecar file up-front ---
    # Resolved before quantization because the MTP module must be
    # quantized to match the SIDECAR checkpoint it loads (see Step 3),
    # whose packing is read off the sidecar's own tensors. Resolving here
    # lets Step 3 load the tensors once and Step 4 reuse them without a
    # second ``snapshot_download`` / ``mx.load``.
    weights_file: Path | None = None
    if mtp_sidecar is not None:
        weights_file = _resolve_sidecar_file(mtp_sidecar)
        if weights_file is None:
            logger.warning(
                "[mtp.inject] sidecar %r could not be resolved to a "
                "safetensors file; skipping MTP injection. "
                "Pass either a repo id (mlx-community/Qwen3.5-9B-MTP-4bit), "
                "a directory containing model-mtp.safetensors, "
                "mtp/model.safetensors, or model.safetensors, or the "
                "file path directly.",
                mtp_sidecar,
            )
            return False

    # --- Step 3: Match the MTP module's quantization to the SIDECAR ---
    # The packed weight/scales shapes of a ``QuantizedLinear`` are a
    # function of ``(bits, group_size)``. Loading a sidecar quantized
    # at a DIFFERENT bit-width than the module was quantized to leaves
    # a weight whose packing disagrees with the layer's ``bits`` attr,
    # so ``mx.quantized_matmul`` raises "weight and scales incompatible"
    # at the first MTP draft step. Historically we matched the BASE
    # model's quantization, which is only safe when base bits == sidecar
    # bits (the mlx-community 4bit+4bit pairing). A mixed pairing — an
    # 8-bit base with the only-published 4-bit MTP head
    # (Qwen3.6-27B-MLX-8bit + Qwen3.6-27B-MTP-4bit) — broke it.
    #
    # For an EXPLICIT sidecar we read the packing from the sidecar's own
    # TENSORS (ground truth) — never the base model, and never a
    # ``config.json`` proxy a checkpoint may omit or mis-declare. The
    # presence of ``fc.scales`` distinguishes quantized from
    # full-precision, and the packed ``fc.weight`` / ``fc.scales`` shapes
    # recover ``(bits, group_size)`` exactly. The base-model fallback is
    # legitimate ONLY on the no-sidecar path, where the MTP head is the
    # base checkpoint's own and its bits match by construction.
    mtp_weights: dict | None = None
    if weights_file is not None:
        # Load the sidecar tensors up-front so their real layout drives
        # quantization; Step 4 reuses this dict for the coverage check.
        # A truncated/unreadable/corrupt safetensors file makes ``mx.load``
        # raise — that must hit the same return-False fail-safe as every
        # other bad-sidecar path below, not escape as an uncaught exception
        # and abort the request mid-generation.
        try:
            raw = mx.load(str(weights_file))
            # Some sidecars (Qwen3-Next ``add_mtp_weights.py`` output) prefix
            # every key with ``mtp.``; others (mlx-community/Qwen3.5-9B-MTP-4bit)
            # store at top-level. Strip the prefix if present so both shapes
            # land on the MTP module's parameter tree.
            mtp_weights = {
                (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
                for k, v in raw.items()
            }
        except Exception as exc:
            logger.error(
                "[mtp.inject] sidecar %r could not be read/parsed (%s); "
                "refusing MTP injection rather than raise mid-request. Omit "
                "--speculative-config to run the plain base path.",
                mtp_sidecar,
                exc,
            )
            return False
        # Read the fc's full-precision (out, in) dims off the freshly-built
        # module, then invert the sidecar's packed fc shapes to recover its
        # quantization.
        #
        # Scope (intentional): the recovered ``(bits, group_size)`` is applied
        # UNIFORMLY to every quantizable leaf. Every shipped MTP sidecar
        # (``mlx-community/Qwen3.5-9B-MTP-*``) is uniformly affine-packed, so
        # fc is a faithful oracle for the whole head. A hypothetical mixed-bit
        # sidecar (e.g. bf16 fc + 4-bit decoder, or 4-bit fc + 8-bit MoE gate)
        # is NOT mis-packed here: the Step 4 shape/dtype verification below
        # catches the resulting per-leaf disagreement and REFUSES the inject
        # (returns False → clean non-MTP fallback), never shipping a head that
        # aborts requests mid-generation. Per-leaf quantization inference would
        # be a feature to *accept* such layouts, not a fix — the fail-safe
        # refusal already holds.
        fc_out_dims, fc_in_dims = (int(d) for d in mtp.fc.weight.shape)
        try:
            sidecar_quant = _infer_sidecar_fc_quantization(
                mtp_weights, fc_out_dims, fc_in_dims
            )
        except ValueError as exc:
            logger.error(
                "[mtp.inject] sidecar %r has a quantized fc whose tensor "
                "layout is inconsistent or an unsupported packing (%s); "
                "refusing MTP injection rather than mis-pack the module and "
                "abort requests at the first draft step. Provide a sidecar "
                "packed with MLX affine quantization, or omit "
                "--speculative-config to run the plain base path.",
                mtp_sidecar,
                exc,
            )
            return False
        if sidecar_quant is None:
            logger.info(
                "[mtp.inject] Sidecar fc is full-precision; "
                "leaving MTP module full-precision."
            )
        else:
            # ``nn.quantize`` applies the fc-derived ``(bits, group_size)``
            # to EVERY quantizable leaf and can itself RAISE — e.g. an fc
            # wide enough for group_size=128 whose sibling projections are
            # narrower than 128 (``last dimension needs to be divisible by
            # group size``). That failure lands here, BEFORE Step 4's safe
            # shape/dtype refusal can run, so it must be caught: an uncaught
            # exception out of inject_mtp_support aborts the request at the
            # first draft step — the very empty-response class this fix
            # closes. Refuse (return False → clean non-MTP fallback) instead.
            try:
                nn.quantize(
                    mtp,
                    group_size=sidecar_quant["group_size"],
                    bits=sidecar_quant["bits"],
                )
            except Exception as exc:
                logger.error(
                    "[mtp.inject] sidecar %r infers %d-bit / group_size=%d "
                    "from its fc, but quantizing the MTP module at that "
                    "packing failed (%s); refusing MTP injection rather than "
                    "raise mid-request. The sidecar's fc packing is "
                    "incompatible with its own narrower leaves — provide a "
                    "uniformly-packed sidecar, or omit --speculative-config "
                    "to run the plain base path.",
                    mtp_sidecar,
                    sidecar_quant["bits"],
                    sidecar_quant["group_size"],
                    exc,
                )
                return False
            logger.info(
                "[mtp.inject] Quantized MTP: %d-bit, group_size=%d "
                "(from sidecar tensors)",
                sidecar_quant["bits"],
                sidecar_quant["group_size"],
            )
    else:
        # No explicit sidecar: the MTP head is the base checkpoint's own,
        # so the base model's quantization is the correct match.
        base_quant = _detect_base_quantization(inner)
        if base_quant is not None:
            # Same fail-safe as the sidecar path: never let a quantize
            # failure escape as an uncaught exception (→ empty response).
            try:
                nn.quantize(
                    mtp,
                    group_size=base_quant["group_size"],
                    bits=base_quant["bits"],
                )
            except Exception as exc:
                logger.error(
                    "[mtp.inject] base model is %d-bit / group_size=%d but "
                    "quantizing the MTP module at that packing failed (%s); "
                    "refusing MTP injection rather than raise mid-request.",
                    base_quant["bits"],
                    base_quant["group_size"],
                    exc,
                )
                return False
            logger.info(
                "[mtp.inject] Quantized MTP: %d-bit, group_size=%d (from base)",
                base_quant["bits"],
                base_quant["group_size"],
            )

    # --- Step 4: Load MTP weights from sidecar safetensors ---
    if mtp_sidecar is not None:
        # ``mtp_weights`` was loaded + prefix-normalised in Step 3
        # (``mtp_sidecar is not None`` implies ``weights_file is not None``).
        assert mtp_weights is not None
        # Pre-load coverage check: codex flagged on PR #954 that
        # ``strict=False`` lets the load silently succeed even when
        # sidecar tensors are missing or misspelled — leaving part of
        # the MTP head random-init while inject_mtp_support still
        # returns True. Compute the expected parameter map off
        # ``mtp.parameters()`` (post-quantize, so ``weight`` /
        # ``scales`` / ``biases`` for QuantizedLinear layers) and
        # refuse the inject if any required tensor is missing.
        from mlx.utils import tree_flatten

        expected = dict(tree_flatten(mtp.parameters()))
        expected_keys = set(expected)
        loaded_keys = set(mtp_weights.keys())
        missing = expected_keys - loaded_keys
        if missing:
            logger.warning(
                "[mtp.inject] sidecar %s is missing %d required MTP "
                "tensor(s); refusing to ship a partially-random-init head. "
                "Missing keys (first 8): %s. "
                "Either grab a correctly-converted sidecar (e.g. "
                "mlx-community/Qwen3.5-9B-MTP-4bit) or regenerate via "
                "the add_mtp_weights.py converter.",
                weights_file.name,
                len(missing),
                sorted(missing)[:8],
            )
            return False
        # Shape verification: the quantization we applied above is inferred
        # from the fc layer and applied UNIFORMLY across the module. Verify
        # that EVERY sidecar tensor matches the corresponding post-quantize
        # parameter shape EXACTLY — a mismatch means the sidecar's packing
        # disagrees with the module's (a mixed-bit, differently-grouped, or
        # corrupted non-fc layer), which is precisely what makes
        # ``mx.quantized_matmul`` raise "weight and scales incompatible" at
        # the first draft step (the empty-response bug). Refuse rather than
        # ship a module that crashes mid-generation.
        shape_mismatches = {
            k: (
                tuple(int(d) for d in mtp_weights[k].shape),
                tuple(int(d) for d in v.shape),
            )
            for k, v in expected.items()
            if tuple(int(d) for d in mtp_weights[k].shape)
            != tuple(int(d) for d in v.shape)
        }
        if shape_mismatches:
            sample = sorted(shape_mismatches.items())[:8]
            logger.warning(
                "[mtp.inject] sidecar %s has %d tensor(s) whose shape "
                "disagrees with the module's quantization (got vs expected); "
                "refusing rather than ship a head that aborts requests at the "
                "first draft step. Mismatches (first 8): %s. This usually "
                "means the sidecar is packed at a different bit-width / "
                "group_size than its fc layer, or is corrupted.",
                weights_file.name,
                len(shape_mismatches),
                sample,
            )
            return False
        # Dtype verification (by ROLE, not exact match): a shape-correct
        # sidecar can still smuggle in a wrong-*dtype* tensor that
        # ``load_weights(strict=False)`` installs without casting, then
        # ``mx.quantized_matmul`` / ``mx.gather_qmm`` rejects. MLX packs
        # quantized ``weight`` as unsigned 32-bit; an ``int32``/``float32``
        # packed weight (or an integer ``scales``/``biases``) has the right
        # shape but blows up at the first draft step — the same empty-response
        # class this fix closes. Enforce by role: an integer-typed expected
        # parameter (the packed ``weight``) must match its dtype EXACTLY,
        # while a floating-typed expected parameter (``scales`` / ``biases`` /
        # any FP weight) need only be *some* floating dtype — a real sidecar
        # legitimately ships bf16 scales against the freshly-quantized fp32
        # module and ``load_weights`` casts them, so an exact float check
        # would false-reject valid checkpoints.
        dtype_mismatches = {}
        for k, v in expected.items():
            got_dt = mtp_weights[k].dtype
            exp_dt = v.dtype
            if mx.issubdtype(exp_dt, mx.integer):
                ok = got_dt == exp_dt
            else:
                ok = mx.issubdtype(got_dt, mx.floating)
            if not ok:
                dtype_mismatches[k] = (str(got_dt), str(exp_dt))
        if dtype_mismatches:
            sample = sorted(dtype_mismatches.items())[:8]
            logger.warning(
                "[mtp.inject] sidecar %s has %d tensor(s) whose dtype is "
                "incompatible with the module's quantization (got vs expected); "
                "refusing rather than ship a head that aborts requests at the "
                "first draft step. Mismatches (first 8): %s. A packed quantized "
                "weight must be unsigned 32-bit and scales/biases must be "
                "floating; a mismatch means the sidecar is corrupted or was "
                "converted with an incompatible packer.",
                weights_file.name,
                len(dtype_mismatches),
                sample,
            )
            return False
        # ``strict=False`` still — we deliberately tolerate EXTRA
        # keys (metadata blobs some converters bundle), but the
        # coverage + shape + dtype checks above prove every required
        # tensor is present and fits its target parameter exactly.
        #
        # ``mx.load`` above (Step 3) is LAZY — it only reads the
        # safetensors header, not tensor DATA. A truncated/unreadable
        # sidecar with a valid header sails through every check above
        # and only RAISES here, at ``mx.eval(mtp.parameters())``
        # materialization. That must hit the same return-False
        # fail-safe as every other bad-sidecar path in this function,
        # not escape as an uncaught exception and abort the request
        # mid-generation.
        try:
            mtp.load_weights(list(mtp_weights.items()), strict=False)
            mx.eval(mtp.parameters())
        except Exception as exc:
            logger.error(
                "[mtp.inject] sidecar %r raised while materializing MTP "
                "weights (%s); refusing MTP injection rather than abort "
                "the request mid-generation. Omit --speculative-config "
                "to run the plain base path.",
                mtp_sidecar,
                exc,
            )
            return False
        extra = loaded_keys - expected_keys
        logger.info(
            "[mtp.inject] Loaded %d/%d expected MTP weight tensors from %s%s",
            len(expected_keys),
            len(expected_keys),
            weights_file.name,
            f" (+{len(extra)} extra sidecar key(s) ignored)" if extra else "",
        )
        _warn_if_mtp_quantization_mismatch(
            _detect_base_quantization(inner), sidecar_quant
        )
    else:
        # No sidecar.
        if not allow_random_init:
            # Codex round-5 BLOCKING fix: default is fail-closed. A
            # missing sidecar in production silently enabled a draft
            # model with random init weights (~0% accept rate) —
            # invisible regression that LOOKS like spec-decode is
            # running but emits zero speedup. Refuse the inject.
            logger.warning(
                "[mtp.inject] inject_mtp_support called without "
                "mtp_sidecar and allow_random_init=False; refusing to "
                "ship a random-init MTP head. Pass "
                "mtp_sidecar='mlx-community/Qwen3.5-9B-MTP-4bit' (or "
                "equivalent) for production use, or set "
                "allow_random_init=True for unit-test wiring probes."
            )
            return False
        # Test-only path — explicit opt-in to random-init weights for
        # wiring tests that pin the surfaces without paying the
        # 131 MB sidecar download cost.
        mx.eval(mtp.parameters())
        logger.warning(
            "[mtp.inject] inject_mtp_support called with "
            "allow_random_init=True — MTP head retains RANDOM init "
            "weights (accept rate ~0%%). This is the test-only path; "
            "do not use in production."
        )

    # --- Step 5: Install global ArraysCache + GatedDeltaNet patches ---
    # Deferred from the top of this function so a failed validation /
    # sidecar load (above) leaves the process global state untouched.
    # Both patches are idempotent + transparent at n_confirmed=0, so
    # a successful inject_mtp_support that runs after a failed one
    # still lands cleanly.
    from .cache_patch import (
        patch_arrays_cache_rollback_state,
        patch_gated_delta_net_for_mtp,
    )

    patch_arrays_cache_rollback_state()
    patch_gated_delta_net_for_mtp()

    # --- Step 6: Attach + monkey-patch ``TextModel`` class ---
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

        # The generic generator only enables prompt-copy speculation for
        # backends whose MTP cache-history synchronization has been audited.
        # This injector covers the Qwen 3.5/3.6/3.8 family.
        mtp_prompt_lookup_supported = True

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
                fa_mask = create_attention_mask(hidden_states, cache[inner_m.fa_idx])
                ssm_mask = create_ssm_mask(hidden_states, cache[inner_m.ssm_idx])
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
            return_hidden: bool = False,
        ):
            """Run the MTP head and project through the shared lm_head."""
            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            if self.args.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(mtp_out)
            else:
                logits = self.lm_head(mtp_out)
            return (logits, mtp_out) if return_hidden else logits

        def mtp_greedy(self, hidden_states, next_token_ids, mtp_cache):
            """Return greedy MTP ids without materializing full-vocab logits."""
            if self.args.tie_word_embeddings:
                return None
            from .quantized_argmax import quantized_argmax

            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            token = quantized_argmax(self.lm_head, mtp_out)
            if token is None:
                return None
            return token, mtp_out

        def make_mtp_cache(self):
            """Return fresh ``KVCache`` entries — one per MTP layer.

            All MTP layers are full-attention by design (PR #990's
            ``MTPDecoderLayer`` is hard-coded to ``self_attn =
            Attention(...)`` — see ``vllm_mlx/spec_decode/mtp/head.py``
            line 89-115). The MTP head deliberately does NOT include
            ``GatedDeltaNet`` linear-attention layers (the backbone's
            hybrid layout via ``args.full_attention_interval`` does not
            apply here). So ``KVCache`` is correct for every MTP layer
            — there are no ``ArraysCache`` slots to maintain on this
            side of the rollback. The ``ArraysCache.rollback_state``
            machinery installed by this PR exists to handle the
            BACKBONE's linear-attention layers (where the GatedDeltaNet
            patch lives), not the MTP head.
            """
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _Qwen3_5WithMTP
    # Some callers retain the VLM-style outer wrapper and pass it onward.
    # Mirror the capability there so the gate follows the injected model
    # regardless of which supported shape reaches the generator.
    if model is not inner:
        model.mtp_prompt_lookup_supported = True
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
