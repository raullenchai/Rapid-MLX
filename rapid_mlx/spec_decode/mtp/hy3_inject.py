# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP injection for HY3 (Tencent Hunyuan 3, ``model_type=hy_v3``).

Forked from :mod:`rapid_mlx.spec_decode.mtp.qwen3_5_inject`. HY3's native MTP head
is DeepSeek-V3-shaped (``enorm``/``hnorm``/``eh_proj``), not Qwen3.5-shaped
(``pre_fc_norm_*``/``fc``), so we build the head from
:func:`rapid_mlx.spec_decode.mtp.hy3_head.build_hy3_mtp_module` and inject the same
four contract surfaces the generator needs:

* ``__call__(inputs, cache=None, input_embeddings=None, return_hidden=False,
  n_confirmed=0)`` — inlines ``hy_v3.HYV3Model``'s single-node backbone loop and
  returns ``(logits, pre_final_norm_hidden)`` when ``return_hidden=True``. HY3 is
  pure-attention (no GatedDeltaNet), so ``n_confirmed`` is a no-op.
* ``mtp_forward(hidden, next_token_ids, mtp_cache)`` -> logits, via the shared
  ``lm_head`` (HY3 has ``tie_word_embeddings=false``).
* ``make_mtp_cache()`` -> ``[KVCache()]`` (1 full-attention MTP layer).

The loaded HY3 model is ``rapid_mlx.models.hy_v3.Model`` (no VLM wrapper): it holds
``.model`` (the ``HYV3Model`` backbone with ``embed_tokens`` / ``layers`` /
``norm``), ``.lm_head``, and ``.args`` (a ``hy_v3.ModelArgs``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rapid_mlx.spec_decode.mtp.detect import _safe_int

logger = logging.getLogger(__name__)

# The 4-bit HY3 MLX checkpoint (mlx-community/Hy3-preview-4bit) STRIPS the
# native MTP head at convert time (it keeps only backbone layers 0..79). The
# head lives in this published sidecar, extracted from the full-precision
# tencent/Hy3-preview layer 80 (see scripts/extract_hy3_mtp.py). Unlike
# Qwen3.5 — whose MTP head ships inside the base checkpoint — HY3 must resolve
# this sidecar by default so a bare
# ``--speculative-config '{"method":"mtp"}'`` boot works with no extra flag.
DEFAULT_HY3_MTP_SIDECAR = "mlx-community/Hy3-preview-MTP-4bit"
# Pin an immutable commit so a silently re-pushed sidecar can never land
# weights into a production boot (codex BLOCKING #1). Bump deliberately
# alongside a re-extract. Only the DEFAULT repo is pinned; an operator-
# supplied ``mtp_sidecar`` (path or repo) is used verbatim.
DEFAULT_HY3_MTP_SIDECAR_REVISION = "dfc35f7d4d1facdfb9bf607908fca569dcb1ab87"
# The default sidecar is extracted at exactly this quantization. Auto-
# resolving it onto a base with a different quant would load packed
# tensors with incompatible shapes (codex BLOCKING #3), so the default
# path is gated on the base matching this spec. An explicit sidecar
# bypasses the gate (the operator vouches for the pairing).
DEFAULT_HY3_MTP_SIDECAR_BASE_QUANT = {"bits": 4, "group_size": 64}


def _resolve_inner_model(model: Any) -> Any:
    """Return the HY3 ``Model`` instance to patch (holds ``.model`` + ``.args``)."""
    # HY3 has no VLM wrapper — the loaded object already exposes .model + .args.
    if hasattr(model, "model") and hasattr(model, "args"):
        return model
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "args") and hasattr(lm, "model"):
        return lm
    return None


def _detect_base_quantization(inner: Any) -> dict | None:
    """Detect ``bits`` / ``group_size`` from a backbone QuantizedLinear."""
    try:
        from mlx.nn import QuantizedEmbedding, QuantizedLinear
    except ImportError:  # pragma: no cover
        return None

    backbone = getattr(inner, "model", None)
    if backbone is None:
        return None
    for layer in getattr(backbone, "layers", []):
        if layer is None:
            continue
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            qp = layer.self_attn.q_proj
            if isinstance(qp, QuantizedLinear):
                return {"bits": int(qp.bits), "group_size": int(qp.group_size)}
    embed = getattr(backbone, "embed_tokens", None)
    if isinstance(embed, QuantizedEmbedding):
        return {"bits": int(embed.bits), "group_size": int(embed.group_size)}
    return None


def _find_mtp_weights_file(sidecar_dir: Path) -> Path | None:
    for c in (sidecar_dir / "model-mtp.safetensors", sidecar_dir / "model.safetensors"):
        if c.exists():
            return c
    return None


def _resolve_sidecar_file(
    mtp_sidecar: str | Path, revision: str | None = None
) -> Path | None:
    if mtp_sidecar is None:
        return None
    # Integrity guard (codex R5 BLOCKING #1): when a revision is pinned (the
    # default-sidecar path), resolve the identifier EXCLUSIVELY as an HF repo at
    # that immutable commit. Otherwise a local directory in the server's cwd
    # that happens to be named like the repo id (e.g.
    # ``mlx-community/Hy3-preview-MTP-4bit/``) would shadow the pinned repo and
    # silently defeat the revision integrity guarantee. An explicit operator-
    # supplied sidecar carries no revision, so local paths still resolve there.
    if revision is not None:
        try:
            from huggingface_hub import snapshot_download

            local = snapshot_download(repo_id=str(mtp_sidecar), revision=revision)
            return _find_mtp_weights_file(Path(local))
        except Exception as exc:  # pragma: no cover — network failure path
            logger.warning(
                "[mtp.inject.hy3] could not resolve pinned sidecar %r@%s: %s",
                mtp_sidecar,
                revision,
                exc,
            )
            return None
    path = Path(mtp_sidecar)
    if path.is_file():
        return path
    if path.is_dir():
        return _find_mtp_weights_file(path)
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(repo_id=str(mtp_sidecar), revision=revision)
        return _find_mtp_weights_file(Path(local))
    except Exception as exc:  # pragma: no cover — network failure path
        logger.warning(
            "[mtp.inject.hy3] could not resolve sidecar %r: %s", mtp_sidecar, exc
        )
        return None


# The two config keys that carry HY3's next-n layer count, most-specific
# first. Must stay in lock-step with ``detect._mtp_num_hidden_layers``'s
# per-family key list for ``hy_v3`` — otherwise a config detection deems
# eligible could be refused here (codex BLOCKING #2). HY3 ships
# ``num_nextn_predict_layers`` (DeepSeek-V3 convention); the
# ``mtp_num_hidden_layers`` alias is accepted as a hand-converted fallback.
_HY3_MTP_LAYER_KEYS = ("num_nextn_predict_layers", "mtp_num_hidden_layers")


def _resolve_num_mtp_layers(inner: Any, model: Any) -> int:
    """Resolve HY3's next-n layer count from the model args / wrapper config.

    Reads the same key set as ``detect._mtp_num_hidden_layers`` for
    ``model_type == "hy_v3"`` so detection and injection never disagree on
    eligibility.
    """
    args = inner.args
    for key in _HY3_MTP_LAYER_KEYS:
        # Guarded parse (codex R9): a hand-edited config shipping a
        # non-numeric ``num_nextn_predict_layers`` (e.g. ``"unknown"``)
        # must degrade to "no MTP" rather than raise and abort server
        # boot. ``detect._safe_int`` is the same fail-closed coercion the
        # detection side uses, so inject and detect never disagree.
        n = _safe_int(getattr(args, key, None), 0)
        if n >= 1:
            return n
    # Fall back to a wrapper text_config dict (defensive; HY3 has no wrapper).
    outer_args = getattr(model, "args", None)
    text_config = getattr(outer_args, "text_config", None) or {}
    if isinstance(text_config, dict):
        for key in _HY3_MTP_LAYER_KEYS:
            n = _safe_int(text_config.get(key), 0)
            if n >= 1:
                return n
    return 0


def inject_hy3_mtp_support(
    model: Any,
    mtp_sidecar: str | Path | None = None,
    *,
    allow_random_init: bool = False,
) -> bool:
    """Inject the four MTP contract surfaces onto a loaded HY3 model.

    Args:
        model: Loaded ``rapid_mlx.models.hy_v3.Model`` (no VLM wrapper).
        mtp_sidecar: Path / dir / HF repo id holding the extracted MTP head.
            When ``None`` (the common ``--speculative-config '{"method":"mtp"}'``
            path with no explicit ``--mtp-sidecar``), it defaults to
            :data:`DEFAULT_HY3_MTP_SIDECAR` and is downloaded from the Hub on
            first use — HY3's 4-bit backbone ships with the head stripped, so a
            sidecar is always required (there is no baked-in head to fall back
            to). Pass ``allow_random_init=True`` (test-only) to skip resolution.
        allow_random_init: Test-only escape hatch — permits a random-init head
            with no sidecar. Never enable in production (~0% accept rate).

    Returns:
        ``True`` when the four MTP surfaces are attached; ``False`` on any
        refusal (unresolvable sidecar, missing tensors, wrong arch).
    """
    import mlx.core as mx
    import mlx.nn as nn

    inner = _resolve_inner_model(model)
    if inner is None:
        logger.warning(
            "[mtp.inject.hy3] model %s lacks (model + args); skipping.",
            type(model).__name__,
        )
        return False

    args = inner.args
    num_mtp_layers = _resolve_num_mtp_layers(inner, model)
    if num_mtp_layers != 1:
        # HY3 ships a single-layer (DeepSeek-V3-style) MTP head. detect may class
        # num_nextn_predict_layers >= 2 as the (unimplemented) TREE variant, but
        # build_hy3_mtp_module raises for != 1, so a num != 1 config would crash
        # boot. Fail-closed to False here (codex R4 BLOCKING #1) — the runtime
        # simply runs without MTP rather than aborting.
        logger.info(
            "[mtp.inject.hy3] num_nextn_predict_layers=%d (need exactly 1); "
            "HY3 MTP unsupported for this config, skipping.",
            num_mtp_layers,
        )
        return False

    # Pipeline-parallel guard (codex R2 BLOCKING #3). The injected ``__call__``
    # below inlines the single-node backbone loop over ``backbone.layers`` and
    # has no distributed send/recv. Under pipeline parallelism the backbone
    # holds only its ``pipeline_layers`` shard and iterating ``layers`` would
    # call absent/non-local layers. Refuse rather than silently mis-run. HY3
    # serves single-node in rapid-mlx today, so this only fail-closes a config
    # the runtime does not support.
    backbone = getattr(inner, "model", None)
    if int(getattr(backbone, "pipeline_size", 1) or 1) > 1:
        logger.warning(
            "[mtp.inject.hy3] pipeline_size=%s > 1; MTP injection does not "
            "support pipeline-parallel HY3. Refusing.",
            getattr(backbone, "pipeline_size", None),
        )
        return False

    # Steps 1-3 (build head, quantize, load sidecar) run under one guard:
    # a truncated / unreadable / malformed sidecar — or any build error —
    # must honor this function's False-on-refusal contract rather than let
    # an exception abort server boot (codex R4 BLOCKING #2).
    try:
        # --- Step 1: Build the HY3 MTP head. ---
        from .hy3_head import build_hy3_mtp_module

        mtp = build_hy3_mtp_module(args, num_mtp_layers)
        logger.info(
            "[mtp.inject.hy3] Built HY3 MTP head (%d layer(s), hidden_size=%d).",
            num_mtp_layers,
            getattr(args, "hidden_size", -1),
        )

        # --- Step 2: Quantize to match base (4-bit Linear; 8-bit router.gate). ---
        quant_info = _detect_base_quantization(inner)

        # Resolve the DEFAULT sidecar BEFORE quantizing the head: HY3's base has no
        # baked-in head, so a missing sidecar defaults to the published repo — but
        # ONLY when the base quantization matches what that sidecar was extracted
        # at (codex BLOCKING #3; a mismatch would load shape-incompatible tensors),
        # and at a pinned revision (codex BLOCKING #1). An explicit sidecar bypasses
        # both gates (the operator vouches for the pairing).
        sidecar_revision = None
        if mtp_sidecar is None and not allow_random_init:
            if quant_info != DEFAULT_HY3_MTP_SIDECAR_BASE_QUANT:
                logger.warning(
                    "[mtp.inject.hy3] base quantization %s != default sidecar's %s; "
                    "the auto-resolved %s would load shape-incompatible tensors. "
                    "Pass an explicit mtp_sidecar extracted for this base.",
                    quant_info,
                    DEFAULT_HY3_MTP_SIDECAR_BASE_QUANT,
                    DEFAULT_HY3_MTP_SIDECAR,
                )
                return False
            mtp_sidecar = DEFAULT_HY3_MTP_SIDECAR
            sidecar_revision = DEFAULT_HY3_MTP_SIDECAR_REVISION
            logger.info(
                "[mtp.inject.hy3] no explicit sidecar; defaulting to %s@%s",
                DEFAULT_HY3_MTP_SIDECAR,
                sidecar_revision[:12],
            )
        if quant_info is not None:

            def _class_predicate(path: str, module) -> Any:
                # Only touch modules that actually support quantization (Linear /
                # Embedding / SwitchLinear expose ``to_quantized``); never norms.
                # Router gate keeps its own bits (8) but the SAME group_size as the
                # base — extract_hy3_mtp.py quantizes the router with the base's
                # group_size, so hard-coding 64 here would reject an explicit
                # sidecar built for a non-gs64 base (codex R2 BLOCKING #2). Mirrors
                # hy_v3.Model.quant_predicate + the default nn.quantize gate.
                if not hasattr(module, "to_quantized"):
                    return False
                if path.endswith("mlp.router.gate"):
                    return {"group_size": quant_info["group_size"], "bits": 8}
                return True

            nn.quantize(
                mtp,
                group_size=quant_info["group_size"],
                bits=quant_info["bits"],
                class_predicate=_class_predicate,
            )
            logger.info(
                "[mtp.inject.hy3] Quantized MTP head: %d-bit gs=%d (router.gate 8-bit)",
                quant_info["bits"],
                quant_info["group_size"],
            )

        # --- Step 3: Load sidecar weights with strict coverage check. ---
        if mtp_sidecar is not None:
            weights_file = _resolve_sidecar_file(mtp_sidecar, revision=sidecar_revision)
            if weights_file is None:
                logger.warning(
                    "[mtp.inject.hy3] sidecar %r could not be resolved; skipping.",
                    mtp_sidecar,
                )
                return False
            raw = mx.load(str(weights_file))
            mtp_weights = {
                (k.removeprefix("mtp.") if k.startswith("mtp.") else k): v
                for k, v in raw.items()
            }
            from mlx.utils import tree_flatten

            expected_keys = {k for k, _ in tree_flatten(mtp.parameters())}
            loaded_keys = set(mtp_weights.keys())
            missing = expected_keys - loaded_keys
            if missing:
                logger.warning(
                    "[mtp.inject.hy3] sidecar %s missing %d required tensor(s); "
                    "refusing partial-random head. Missing (first 8): %s",
                    weights_file.name,
                    len(missing),
                    sorted(missing)[:8],
                )
                return False

            # Shape / dtype-category guard (codex BLOCKING): the key-name coverage
            # above does NOT catch a sidecar quantized for a different base (e.g.
            # 8-bit packed tensors landing on a 4-bit head), which would either
            # error opaquely in load_weights or corrupt inference. A quant mismatch
            # shows up as (a) a packed-shape mismatch and/or (b) a floating-vs-
            # integer dtype-category flip (packed quant weights are uint32; the
            # freshly-initialised template's unquantized tensors are floating).
            #
            # We deliberately do NOT require exact dtype equality WHEN BOTH SIDES
            # ARE FLOATING: the real sidecar carries bf16 unquantized tensors
            # (RMSNorm weights, biases) while the template initialises them in
            # fp32, and ``load_weights`` casts float->float losslessly-in-kind.
            # Requiring the raw dtype (as an earlier revision did) rejected a
            # valid bf16 sidecar. But we must NOT wave through a non-float dtype
            # for a float slot, nor a differing integer dtype for a packed uint32
            # slot (codex R8: a same-shape bool / int8 / int32 must still be
            # refused). So: shapes must match, and dtypes must be EXACTLY equal
            # unless BOTH are floating.
            def _dtype_compatible(got: Any, want: Any) -> bool:
                if got == want:
                    return True
                return bool(mx.issubdtype(got, mx.floating)) and bool(
                    mx.issubdtype(want, mx.floating)
                )

            expected = {
                k: (v.shape, v.dtype) for k, v in tree_flatten(mtp.parameters())
            }
            bad = [
                (
                    k,
                    (tuple(expected[k][0]), str(expected[k][1])),
                    (tuple(mtp_weights[k].shape), str(mtp_weights[k].dtype)),
                )
                for k in expected_keys
                if tuple(mtp_weights[k].shape) != tuple(expected[k][0])
                or not _dtype_compatible(mtp_weights[k].dtype, expected[k][1])
            ]
            if bad:
                logger.warning(
                    "[mtp.inject.hy3] sidecar %s has %d shape/dtype-mismatched "
                    "tensor(s) (likely a quantization mismatch vs the base). "
                    "First (key, expected(shape,dtype), got(shape,dtype)): %s. "
                    "Refusing to load. Use a sidecar extracted for this base quant.",
                    weights_file.name,
                    len(bad),
                    bad[0],
                )
                return False
            mtp.load_weights(list(mtp_weights.items()), strict=False)
            mx.eval(mtp.parameters())
            extra = loaded_keys - expected_keys
            logger.info(
                "[mtp.inject.hy3] Loaded %d/%d MTP tensors from %s%s",
                len(expected_keys),
                len(expected_keys),
                weights_file.name,
                f" (+{len(extra)} extra ignored)" if extra else "",
            )
        else:
            if not allow_random_init:
                logger.warning(
                    "[mtp.inject.hy3] no mtp_sidecar and allow_random_init=False; "
                    "refusing random-init head."
                )
                return False
            mx.eval(mtp.parameters())
            logger.warning(
                "[mtp.inject.hy3] allow_random_init=True — RANDOM init head (test-only)."
            )

    except Exception as exc:  # noqa: BLE001 — fail-closed on ANY head/load error
        logger.warning(
            "[mtp.inject.hy3] head build / sidecar load failed (%s: %s); "
            "refusing MTP injection (model left unmodified).",
            type(exc).__name__,
            exc,
        )
        return False

    # --- Step 4: Attach + monkey-patch the HY3 Model class. ---
    inner.mtp = mtp
    original_class = type(inner)

    class _HY3WithMTP(original_class):  # type: ignore[valid-type, misc]
        """HY3 ``Model`` + the four MTP surfaces the generator drives."""

        def __call__(  # type: ignore[override]
            self,
            inputs,
            cache=None,
            input_embeddings=None,
            return_hidden: bool = False,
            n_confirmed: int = 0,
        ):
            from mlx_lm.models.base import create_attention_mask

            backbone = self.model
            if input_embeddings is not None:
                h = input_embeddings
            else:
                h = backbone.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(backbone.layers)

            # Single-node inline of hy_v3.HYV3Model.__call__ (pipeline_size=1,
            # rank=0 => no distributed branches). n_confirmed is a no-op:
            # HY3 is pure-attention, no SSM/conv state to snapshot.
            #
            # NOTE: pass cache[0] (a single KVCache), NOT the list. mlx-lm's
            # create_attention_mask does `if cache and hasattr(cache, "make_mask")`
            # and calls that single cache's offset-aware make_mask(N). Passing the
            # LIST would fail that hasattr (lists have no make_mask), silently drop
            # to a plain causal mask, and LOSE the warm-cache offset. This mirrors
            # upstream hy_v3.HYV3Model.__call__ exactly (models/hy_v3.py:309) and
            # is covered by test_inject_forward_warm_cache_offset.
            mask = create_attention_mask(h, cache[0])
            for layer, c in zip(backbone.layers, cache):
                h = layer(h, mask, cache=c)

            pre_norm_hidden = h
            normed = backbone.norm(h)
            if self.args.enable_lm_head_fp32:
                normed = normed.astype(mx.float32)
            if self.args.tie_word_embeddings:
                out = backbone.embed_tokens.as_linear(normed)
            else:
                out = self.lm_head(normed)

            if return_hidden:
                return out, pre_norm_hidden
            return out

        def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
            mtp_out = self.mtp(
                hidden_states,
                next_token_ids,
                self.model.embed_tokens,
                mtp_cache,
            )
            if self.args.enable_lm_head_fp32:
                mtp_out = mtp_out.astype(mx.float32)
            if self.args.tie_word_embeddings:
                return self.model.embed_tokens.as_linear(mtp_out)
            return self.lm_head(mtp_out)

        def make_mtp_cache(self):
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp.layers]

    inner.__class__ = _HY3WithMTP
    logger.info(
        "[mtp.inject.hy3] Patched %s with MTP surfaces "
        "(return_hidden, n_confirmed, mtp_forward, make_mtp_cache).",
        original_class.__name__,
    )
    return True


def validate_hy3_mtp_support(model: Any) -> bool:
    """Verify inject_hy3_mtp_support attached the four surfaces."""
    import inspect

    inner = _resolve_inner_model(model)
    if inner is None:
        return False
    if getattr(inner, "mtp", None) is None:
        logger.warning("[mtp.validate.hy3] model.mtp is missing.")
        return False
    if not callable(getattr(inner, "mtp_forward", None)):
        logger.warning("[mtp.validate.hy3] model.mtp_forward is missing.")
        return False
    if not callable(getattr(inner, "make_mtp_cache", None)):
        logger.warning("[mtp.validate.hy3] model.make_mtp_cache is missing.")
        return False
    sig = inspect.signature(type(inner).__call__)
    if "return_hidden" not in sig.parameters:
        logger.warning("[mtp.validate.hy3] __call__ lacks return_hidden.")
        return False
    if "n_confirmed" not in sig.parameters:
        logger.warning("[mtp.validate.hy3] __call__ lacks n_confirmed.")
        return False
    return True
