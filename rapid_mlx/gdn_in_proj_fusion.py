# SPDX-License-Identifier: Apache-2.0
"""Fuse the four GatedDeltaNet input projections into one quantized matmul.

At decode, mlx-lm's ``qwen3_5.GatedDeltaNet`` issues four projection
launches per GDN layer per step: ``in_proj_qkv`` (N=2*key_dim+value_dim),
``in_proj_z`` (N=value_dim), and the two 48-row ``in_proj_b``/``in_proj_a``
slivers that are pure launch overhead at single-token widths. Affine
quantization packs each output row independently, so concatenating all
four weight matrices along the output axis and issuing ONE
``quantized_matmul`` is bit-identical to the four separate calls — but
only while the matmul stays on the narrow-batch kernel family.

Width gating (the correctness core): on mlx 0.31 / Apple Silicon the
fused concat is byte-identical to the split calls for ``M = B*S <= 11``
and diverges from ``M = 12`` up, where the wide kernel accumulates the
short z/b/a row blocks in a different order. The fused path is therefore
gated to ``M <= _FUSED_MAX_ROWS`` (8), and eligibility is decided at
install time by byte-parity probes (below). Wider inputs — prefill and
large batched steps — run per-projection matmuls on ROW-SLICE VIEWS of
the fused arrays: identical inputs to the stock path, so byte-exact by
construction, with no duplicate weight memory.

Two install-time probes guard the rewrite; if either fails, the model is
left stock (no partial fusion):

* Projection parity on the FIRST eligible layer's real weights: fused
  concat vs the four stock projections, byte-compared for every
  ``M in 1.._FUSED_MAX_ROWS`` — pins the kernel-dispatch boundary on the
  actual hardware, mlx build, and weight geometry.
* Whole-layer parity on a small synthetic ``GatedDeltaNet``: stock
  ``__call__`` vs the fused reimplementation, byte-compared with and
  without cache — catches drift if a future mlx-lm changes the layer
  body our fused call mirrors.

Verified on Qwen3.8-27B-4bit (M3 Ultra + M2 Pro): decode +2.0% / +0.9%,
6-prompt greedy byte-identical, 1041-token prefill logits byte-identical.
Applies to every ``qwen3_5``-family checkpoint (Qwen3.5/3.6/3.8 dense and
MoE — ``qwen3_5_moe`` reuses the same layer class).

Set ``RAPID_MLX_GDN_IN_PROJ_FUSION=0`` to disable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Fused single-matmul path is used only when B*S <= this. Must stay at or
# below the byte-parity boundary of the narrow-batch quantized kernel
# (empirically 11 on mlx 0.31); the install-time probe re-verifies every
# row count in range on the running configuration.
_FUSED_MAX_ROWS = 8


def _gdn_imports():
    """Import lazily so a changed mlx-lm degrades to no-fusion, never a crash."""
    try:
        import mlx.nn as nn
        from mlx_lm.models.gated_delta import gated_delta_update
        from mlx_lm.models.qwen3_5 import GatedDeltaNet
    except ImportError:
        return None
    return nn, gated_delta_update, GatedDeltaNet


def _proj_modules(gdn: Any):
    return [gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a]


def _can_fuse(gdn: Any) -> bool:
    """Structural gate: only fuse a projection quartet the concat is exact for."""
    if hasattr(gdn, "in_proj_fused"):
        return False
    if getattr(gdn, "sharding_group", None) is not None:
        return False
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        if not hasattr(gdn, name):
            return False
    try:
        import mlx.nn as nn
    except ImportError:
        return False
    parts = _proj_modules(gdn)
    base = parts[0]
    if not all(type(p) is nn.QuantizedLinear for p in parts):
        return False
    # The sliced wide path calls mx.quantized_matmul without a mode
    # argument, i.e. with the affine interpretation; any other mode must
    # stay stock.
    if getattr(base, "mode", "affine") != "affine":
        return False
    for p in parts:
        if (p.group_size, p.bits, getattr(p, "mode", "affine")) != (
            base.group_size,
            base.bits,
            getattr(base, "mode", "affine"),
        ):
            return False
        if "bias" in p:
            return False
        if p.get("biases") is None or p["weight"].shape[1] != base["weight"].shape[1]:
            return False
        if p["weight"].dtype != base["weight"].dtype:
            return False
        # Scale/bias precision decides kernel numerics; a quartet mixing
        # them cannot share one fused matmul.
        if p["scales"].dtype != base["scales"].dtype:
            return False
        if p["biases"].dtype != base["biases"].dtype:
            return False
    return True


def _concat_params(parts):
    weight = mx.concatenate([p["weight"] for p in parts], axis=0)
    scales = mx.concatenate([p["scales"] for p in parts], axis=0)
    biases = mx.concatenate([p["biases"] for p in parts], axis=0)
    return weight, scales, biases


def _fused_projections(gdn: Any, inputs: mx.array):
    """The four projection outputs, from the fused arrays.

    Narrow inputs take one fused matmul + split; wide inputs take four
    matmuls on row-slice views (byte-identical to the stock projections).
    """
    fused = gdn.in_proj_fused
    bounds = gdn._rapid_gdn_bounds
    if (
        inputs.shape[0] * inputs.shape[1] <= _FUSED_MAX_ROWS
        and inputs.dtype in gdn._rapid_gdn_dtypes
    ):
        out = fused(inputs)
        return mx.split(out, bounds[:-1], axis=-1)
    gs, bits = fused.group_size, fused.bits
    outs = []
    lo = 0
    for hi in bounds:
        outs.append(
            mx.quantized_matmul(
                inputs,
                fused.weight[lo:hi],
                fused.scales[lo:hi],
                fused.biases[lo:hi],
                transpose=True,
                group_size=gs,
                bits=bits,
            )
        )
        lo = hi
    return outs


def _make_fused_call(orig_call, nn, gated_delta_update):
    def fused_call(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        if not hasattr(self, "in_proj_fused"):
            return orig_call(self, inputs, mask, cache)

        # Mirrors qwen3_5.GatedDeltaNet.__call__ (mlx-lm 0.31) with the
        # four projections replaced; the install-time whole-layer parity
        # probe pins this copy byte-for-byte against the stock body.
        B, S, _ = inputs.shape
        qkv, z, b, a = _fused_projections(self, inputs)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        return out

    return fused_call


def _ensure_call_patch(gdn_cls, nn, gated_delta_update) -> None:
    # The per-class marker is the sole authority: a reloaded/replaced
    # GatedDeltaNet class must be re-patched even though this module
    # already patched an earlier incarnation (a module-global flag here
    # would skip it and leave fused instances dispatching to a stock
    # __call__ whose projections were deleted).
    if getattr(gdn_cls, "_rapid_gdn_in_proj_fused_call", False):
        return
    orig = gdn_cls.__call__
    gdn_cls.__call__ = _make_fused_call(orig, nn, gated_delta_update)
    gdn_cls._rapid_gdn_in_proj_fused_call = True
    gdn_cls._rapid_gdn_in_proj_original_call = orig


_UINT_OF_SIZE = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}


def _bytes_of(arr: mx.array) -> bytes:
    """Raw bit pattern of ``arr`` — no dtype conversion, so signed zeros
    and NaN payloads cannot be normalized away."""
    import numpy as np

    u = getattr(mx, _UINT_OF_SIZE[arr.dtype.size])
    return np.array(mx.view(arr, u), copy=True).tobytes()


# Activation dtypes the fused path may serve. Fusion for a dtype is only
# enabled after the projection probe passes for it; anything else takes
# the (always byte-exact) sliced path at runtime.
_PROBE_DTYPES = ("bfloat16", "float16")


def _probe_projection_parity(gdn: Any) -> frozenset:
    """Fused concat vs stock projections on this quartet's real weights,
    byte-compared per candidate dtype for every fused row count
    (``(1, M)`` for M=1.._FUSED_MAX_ROWS plus a multi-batch ``(2, 4)``
    factorization). Returns the set of dtypes that passed — pins the
    kernel-dispatch boundary on the running hardware/mlx build."""
    parts = _proj_modules(gdn)
    weight, scales, biases = _concat_params(parts)
    gs, bits = parts[0].group_size, parts[0].bits
    bounds = []
    total = 0
    for p in parts:
        total += p["weight"].shape[0]
        bounds.append(total)
    hidden = parts[0]["scales"].shape[1] * gs
    shapes = [(1, rows) for rows in range(1, _FUSED_MAX_ROWS + 1)]
    if _FUSED_MAX_ROWS >= 8:
        shapes.append((2, 4))
    passed = set()
    for dtype_name in _PROBE_DTYPES:
        dtype = getattr(mx, dtype_name)
        try:
            ok = True
            for b_dim, rows in shapes:
                x = (
                    mx.random.normal(
                        (b_dim, rows, hidden), key=mx.random.key(b_dim * 100 + rows)
                    )
                    * 0.3
                ).astype(dtype)
                fused = mx.quantized_matmul(
                    x, weight, scales, biases, transpose=True, group_size=gs, bits=bits
                )
                fused_parts = mx.split(fused, bounds[:-1], axis=-1)
                stock_parts = [p(x) for p in parts]
                mx.eval(fused_parts, stock_parts)
                for sp, fp in zip(stock_parts, fused_parts):
                    if _bytes_of(sp) != _bytes_of(fp):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                passed.add(dtype)
        except Exception:  # noqa: BLE001 — kernel/runtime failure: dtype stays stock
            continue
    return frozenset(passed)


def _probe_whole_layer_parity(gdn_cls, nn, gated_delta_update) -> bool:
    """Stock ``__call__`` vs the fused reimplementation on a small synthetic
    layer, byte-compared without cache and across a prefill+decode cache
    sequence (catches mlx-lm body drift)."""
    try:
        from mlx_lm.models.cache import ArraysCache
        from mlx_lm.models.qwen3_5 import TextModelArgs

        args = TextModelArgs(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            vocab_size=128,
        )
        stock = gdn_cls(args)
        fusedm = gdn_cls(args)
        fusedm.update(stock.parameters())
        for m in (stock, fusedm):
            nn.quantize(m, group_size=32, bits=4)
            m.eval()  # serving mode: gated_delta_update takes the kernel path
        fusedm.update(stock.parameters())
        mx.eval(stock.parameters(), fusedm.parameters())

        if not _can_fuse(fusedm):
            return False
        _fuse_one(fusedm, frozenset({mx.bfloat16}))

        for rows in (1, 2, 8, 16):
            x = (
                mx.random.normal(
                    (1, rows, args.hidden_size), key=mx.random.key(100 + rows)
                )
                * 0.3
            ).astype(mx.bfloat16)
            y_stock = stock(x, None, None)
            y_fused = fusedm(x, None, None)
            mx.eval(y_stock, y_fused)
            if _bytes_of(y_stock) != _bytes_of(y_fused):
                return False

        # Prefill (wide, sliced path) then decode (narrow, fused path)
        # through a cache — masked and with ragged ``cache.lengths`` set,
        # covering the branches the batched engine exercises. Outputs AND
        # the carried conv/recurrent state must match bit-for-bit.
        xp = (
            mx.random.normal((1, 16, args.hidden_size), key=mx.random.key(7)) * 0.3
        ).astype(mx.bfloat16)
        xd = (
            mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(8)) * 0.3
        ).astype(mx.bfloat16)
        mask = mx.arange(16)[None, :] < 12
        outs = []
        for layer in (stock, fusedm):
            cache = ArraysCache(size=2)
            cache.lengths = mx.array([12])
            y1 = layer(xp, mask, cache)
            cache.lengths = None
            y2 = layer(xd, None, cache)
            mx.eval(y1, y2, cache[0], cache[1])
            outs.append((y1, y2, cache[0], cache[1]))
        for a, b in zip(outs[0], outs[1]):
            if _bytes_of(a) != _bytes_of(b):
                return False
        return True
    except Exception:  # noqa: BLE001 — synthetic probe failure: stay stock
        return False


def _fuse_one(gdn: Any, dtypes: frozenset = frozenset({mx.bfloat16})) -> None:
    """Rewrite one layer, failure-atomically.

    Everything that can realistically fail (array concat + eval) happens
    before any attribute on ``gdn`` changes. The commits are then plain
    Python attribute writes ordered so that every intermediate state is
    behaviorally correct: the patched ``__call__`` dispatches on
    ``in_proj_fused``, which is set only after the fused container and
    its bounds/dtype gates are complete; the stock projections are
    deleted last (an interruption there merely leaks their buffers until
    the next ``mx.clear_cache``).
    """
    import copy

    parts = _proj_modules(gdn)
    weight, scales, biases = _concat_params(parts)
    mx.eval(weight, scales, biases)
    bounds = []
    total = 0
    for p in parts:
        total += p["weight"].shape[0]
        bounds.append(total)

    # Fresh container (not the live in_proj_qkv) so the stock path is
    # never observed with enlarged geometry; quantization params carry
    # over via the copy, the copied arrays are replaced immediately.
    fused = copy.deepcopy(parts[0])
    fused.weight = weight
    fused.scales = scales
    fused.biases = biases

    gdn._rapid_gdn_bounds = bounds
    gdn._rapid_gdn_dtypes = dtypes
    gdn.in_proj_fused = fused
    del gdn.in_proj_qkv
    del gdn.in_proj_z
    del gdn.in_proj_b
    del gdn.in_proj_a


def fuse_gdn_in_proj(model: Any) -> int:
    """Fuse the GDN input projections on a loaded model, in place.

    Returns the number of fused ``GatedDeltaNet`` instances (0 when
    disabled via ``RAPID_MLX_GDN_IN_PROJ_FUSION=0``, mlx-lm is
    unavailable, nothing is eligible, or the install-time parity probes
    fail). Non-GDN models are a cheap no-op. Idempotent: fused instances
    carry ``in_proj_fused`` and are skipped by the structural gate.
    Never raises — an optional optimization must not be able to fail a
    model load, so any unexpected error before the commit phase degrades
    to "model left stock" (return 0), and an error during the commit
    phase degrades to a partially-fused model whose fused layers are
    individually parity-proven and fully functional — the return value
    is the number of layers actually committed in every case.
    """
    if os.environ.get("RAPID_MLX_GDN_IN_PROJ_FUSION", "1") == "0":
        logger.info("[gdn_fusion] disabled via RAPID_MLX_GDN_IN_PROJ_FUSION=0")
        return 0
    try:
        return _install(model)
    except Exception:  # noqa: BLE001 — never let the optimization fail a load
        logger.warning("[gdn_fusion] install failed; model left as-is", exc_info=True)
        return 0


def _install(model: Any) -> int:
    imports = _gdn_imports()
    if imports is None:
        return 0
    nn, gated_delta_update, gdn_cls = imports

    try:
        modules = [m for _, m in model.named_modules()]
    except Exception:  # noqa: BLE001 — unusual model containers: no fusion
        return 0
    # Exact type only: a subclass may override __call__ and read the
    # split projections directly.
    targets = [m for m in modules if type(m) is gdn_cls and _can_fuse(m)]
    if not targets:
        return 0

    # Phase 1 (no mutation): byte-parity-probe EVERY eligible layer on
    # its own real weights. The probes are single-row-to-8-row matmuls,
    # so the whole pass costs well under a second even at 48 layers, and
    # no layer is ever fused on the strength of another layer's parity.
    plans = []
    for gdn in targets:
        dtypes = _probe_projection_parity(gdn)
        if dtypes:
            plans.append((gdn, dtypes))
    if not plans:
        logger.warning(
            "[gdn_fusion] projection byte-parity probe failed for every "
            "eligible layer on this mlx/hardware configuration; "
            "leaving model stock"
        )
        return 0

    _ensure_call_patch(gdn_cls, nn, gated_delta_update)
    if not _probe_whole_layer_parity(gdn_cls, nn, gated_delta_update):
        logger.warning(
            "[gdn_fusion] whole-layer byte-parity probe failed (mlx-lm "
            "body drift?); leaving model stock"
        )
        return 0

    # Phase 2 (commit): plain in-place attribute rewrites per layer. A
    # failure here leaves earlier layers fused (each is independently
    # parity-proven and functional); report the true committed count.
    committed = 0
    try:
        for gdn, dtypes in plans:
            _fuse_one(gdn, dtypes)
            committed += 1
            # Freed projection buffers land in the MLX pool; drain per
            # layer so the load transient stays bounded to one layer's
            # worth.
            mx.clear_cache()
    except Exception:  # noqa: BLE001 — keep the partially-fused model serving
        # A crash inside _fuse_one after ``in_proj_fused`` was installed
        # (i.e. during the trailing deletes) still leaves that layer on
        # the fused path — count it by its observable state, not by
        # whether _fuse_one returned.
        committed = sum(
            1 for gdn, _ in plans if getattr(gdn, "in_proj_fused", None) is not None
        )
        logger.warning(
            "[gdn_fusion] commit failed after %d/%d layers; the fused "
            "layers remain active and parity-proven",
            committed,
            len(plans),
            exc_info=True,
        )
        return committed
    logger.info("[gdn_fusion] in-projection fusion applied: %d GDN layers", committed)
    return committed
