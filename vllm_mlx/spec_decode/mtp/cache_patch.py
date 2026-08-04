# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm's ``ArraysCache`` to carry a ``rollback_state`` slot.

mlx-lm PR #990 adds a ``rollback_state: Optional[tuple] = None`` class
attribute to ``mlx_lm.models.cache.ArraysCache``. It is set by the
GatedDeltaNet layer's ``_process_chunk`` split (saves the
``(conv_state, ssm_state)`` snapshot at position ``n_confirmed``) and
read by the MTP generator's ``_rollback_draft`` (restores the snapshot
on draft rejection). Both writers and readers run under
``mx.stream(generation_stream)`` so the lock-free attribute access is
safe.

Until upstream merges, our installed ``mlx_lm 0.31.3`` does not have
the attribute. Setting it on a per-instance basis from the patched
model's ``_process_chunk`` would work, but Python's attribute lookup
falls back to the class only after the instance miss, so the FIRST
write succeeds — but the ``hasattr(cache, "rollback_state")`` guard in
the generator's ``_clear_rollback`` runs against the CLASS first and
would return ``False`` on a fresh cache, skipping the clear. That's
fine in isolation (nothing to clear) but the same guard is used to
gate ``rollback_state is not None`` checks; without the class slot we
have to fall back to ``getattr(c, "rollback_state", None)`` everywhere
which is fragile.

Patching the class once at import time is the simple fix. The patch is:

* Idempotent — calling :func:`patch_arrays_cache_rollback_state` twice
  is a no-op.
* Reversible only via process restart — the patch is intentionally
  one-way. There is no test path that needs to un-patch (mlx-lm's
  ``ArraysCache`` is a behaviorally-pure attribute slot; adding it
  doesn't change anything for callers that don't touch it).
* Safe under future mlx-lm versions that add the slot themselves —
  the guard checks ``"rollback_state" in cls.__dict__`` before
  patching, so once upstream lands the change this becomes a no-op.

The patch is applied automatically the first time
:func:`vllm_mlx.spec_decode.mtp.generator.mtp_generate_step` is
imported (the import in the generator module forces the side-effect).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# Module-level guard so concurrent threads importing the generator
# don't race on the class attribute install. Without the lock, two
# threads could both see ``"rollback_state" not in cls.__dict__`` and
# both setattr — harmless for an attribute set to ``None`` (the writes
# are identical) but conceptually racy. The lock keeps the install
# atomic.
_install_lock = threading.Lock()
_patched = False
_gated_delta_patched = False
_orig_gated_delta_call = None


def patch_arrays_cache_rollback_state() -> bool:
    """Install ``rollback_state = None`` on ``mlx_lm.models.cache.ArraysCache``.

    Returns ``True`` if the patch was applied, ``False`` if the slot
    was already present (either from a previous call or from a future
    mlx-lm version that lands the change upstream).

    Raises:
        ImportError: If ``mlx_lm.models.cache`` cannot be imported.
            The MTP path is fundamentally unusable without mlx-lm so
            we let the import error propagate rather than silently
            falling back.
    """
    global _patched

    with _install_lock:
        if _patched:
            return False

        # Defer the import so a static analyzer can't trip on the
        # mlx_lm dependency before the package is installed (the
        # generator module itself imports mlx_lm at the top, so by the
        # time this patch fires, the import must already work — but
        # we still keep it lazy for symmetry with the rest of the MTP
        # package).
        from mlx_lm.models.cache import ArraysCache

        # ``cls.__dict__`` check (not ``hasattr``) so a future mlx-lm
        # that ships the slot wins over our patch — we don't want to
        # shadow an upstream rename or type change.
        if "rollback_state" in ArraysCache.__dict__:
            _patched = True
            logger.debug(
                "[mtp.cache_patch] ArraysCache.rollback_state already present "
                "(upstream version or prior patch); skipping install."
            )
            return False

        # The class attribute default is ``None``; instance writes
        # shadow it transparently. This mirrors the upstream PR #990
        # patch verbatim (``ArraysCache`` is a ``_BaseCache`` subclass
        # built via ``__new__``, so class-level defaults are the right
        # shape — there is no ``__init__`` that would otherwise
        # initialize the slot).
        ArraysCache.rollback_state = None  # type: ignore[attr-defined]
        _patched = True
        logger.info(
            "[mtp.cache_patch] Installed rollback_state slot on "
            "ArraysCache (vendored from mlx-lm PR #990)."
        )
        return True


def patch_gated_delta_net_for_mtp() -> bool:
    """Wrap ``GatedDeltaNet.__call__`` with a chunk-split version that
    snapshots SSM/conv state at a confirmed boundary.

    PR #990 adds an ``n_confirmed`` parameter to ``GatedDeltaNet`` so
    that during the MTP verify forward (which processes
    ``[main_tok, draft_tok]`` as a 2-token batch with
    ``n_confirmed=1``) the layer splits its
    :func:`mlx_lm.models.gated_delta.gated_delta_update` call into two
    chunks and writes ``(conv_state_at_boundary, ssm_state_at_boundary)``
    to ``cache.rollback_state``. On draft rejection the
    :func:`vllm_mlx.spec_decode.mtp.generator._rollback_draft` path
    restores those snapshots so the linear-attention state matches
    "after main_tok, before draft_tok" — the position the next
    generator iteration's input ``[verify_tok_id, new_draft]`` expects
    to attend from.

    Without this patch the verify forward advances the SSM by 2 steps
    and there is no way to roll back to position 1 on rejection — the
    LOSSLESS contract breaks on the linear-attention layers (only;
    full-attention's ``KVCache.trim(1)`` already handles its rollback).
    Output diverges from the non-spec-decode baseline within ~10
    tokens at 90% accept rate.

    The patch:

    * Is idempotent — calling twice is a no-op.
    * Is transparent — when ``cache.n_confirmed_for_mtp`` is 0 (the
      class default), the wrapped call falls through to the original
      ``__call__`` unchanged. Production non-MTP code paths are
      unaffected.
    * Reads the chunk boundary from ``cache.n_confirmed_for_mtp``,
      which the MTP-wrapped ``TextModel.__call__`` sets before each
      ``layer.linear_attn`` invocation. Threading via a cache attr
      avoids changing the layer's call signature (and so avoids
      touching ``DecoderLayer.__call__`` / ``Qwen3_5TextModel.__call__``
      upstream).

    Returns ``True`` when the patch was applied (or already in place),
    ``False`` if mlx-lm cannot be imported.
    """
    global _gated_delta_patched, _orig_gated_delta_call

    with _install_lock:
        if _gated_delta_patched:
            return True

        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm.models.cache import ArraysCache
            from mlx_lm.models.gated_delta import gated_delta_update
            from mlx_lm.models.qwen3_5 import GatedDeltaNet
        except ImportError:  # pragma: no cover — mlx_lm always available
            logger.warning(
                "[mtp.cache_patch] Could not import GatedDeltaNet; "
                "skipping rollback-state install."
            )
            return False

        # Add class-default MTP slots to ArraysCache so the layer can
        # read them without ``AttributeError`` on untagged caches.
        # ``snapshot_offsets``: list of token-counts at which to snapshot
        # the (conv, ssm) state during a verify forward (multi-slot for
        # chain-of-K). ``rollback_states``: dict {n_from_end: (conv, ssm)}
        # the generator's ``_rollback_draft(n)`` restores from.
        # ``n_confirmed_for_mtp`` kept for backward-compat detection.
        if "n_confirmed_for_mtp" not in ArraysCache.__dict__:
            ArraysCache.n_confirmed_for_mtp = 0  # type: ignore[attr-defined]
        if "snapshot_offsets" not in ArraysCache.__dict__:
            ArraysCache.snapshot_offsets = None  # type: ignore[attr-defined]
        if "rollback_states" not in ArraysCache.__dict__:
            ArraysCache.rollback_states = None  # type: ignore[attr-defined]
        # ``rollback_recompute``: a per-layer closure the single-pass verify
        # forward stashes so ``_rollback_draft(n)`` can recompute the
        # accepted-prefix (conv, ssm) state on the rare draft rejection —
        # replaces the old materialized per-boundary snapshot dict.
        if "rollback_recompute" not in ArraysCache.__dict__:
            ArraysCache.rollback_recompute = None  # type: ignore[attr-defined]

        _orig_gated_delta_call = GatedDeltaNet.__call__

        def _patched_call(self, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            offsets = None
            if cache is not None:
                raw = getattr(cache, "snapshot_offsets", None)
                if raw:
                    offsets = sorted({int(o) for o in raw if 0 < int(o) < S})

            # Fast path — no MTP snapshot requested, S<2, no cache, or
            # tensor-parallel (verify runs single-device). Byte-equal to
            # the original forward; snapshot state left untouched.
            if cache is None or not offsets or S < 2 or self.sharding_group is not None:
                return _orig_gated_delta_call(self, inputs, mask=mask, cache=cache)

            # --- Single-pass forward + lazy recompute-on-reject ---
            # ``cache`` is the PER-LAYER ArraysCache for this one
            # GatedDeltaNet instance; writes below affect only it.
            #
            # The verify window (S = K+1 tokens) is run as ONE fused
            # ``gated_delta_update`` — byte-equal to the unsplit forward.
            # The previous implementation split the scan into K+1 segments
            # to materialize a (conv, ssm) snapshot at every draft boundary
            # (K+1 kernel launches per GDN layer per verify round — measured
            # as the dominant MTP cost, ~30% of throughput at K=2). Because
            # ~90% of rounds accept every draft and never roll back, that
            # snapshot work is wasted almost every round. Instead we run one
            # fused scan and stash a cheap rollback CLOSURE; only on the rare
            # rejection does ``_rollback_draft`` call it to recompute the
            # accepted-prefix state with ONE short scan from the pre-window
            # state. Since ``gated_delta_update`` is a pure sequential scan,
            # rescanning ``[0:keep]`` from that anchor is byte-exact with the
            # true keep-token state the old snapshot stored.
            qkv = self.in_proj_qkv(inputs)
            z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
            b = self.in_proj_b(inputs)
            a = self.in_proj_a(inputs)

            if cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            n_keep = self.conv_kernel_size - 1
            # Conv state after all S tokens (last n_keep of conv_input).
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

            # Pre-window SSM state (before this verify window) — the anchor
            # every rollback recomputes from. ``cache[1]`` is overwritten
            # with the post-scan state below, so keep a reference here.
            pre_ssm = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            # Single fused scan over the whole verify window.
            out, st = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                pre_ssm,
                mask,
                use_kernel=not self.training,
            )

            # Rollback closure: recompute (conv, ssm) after keeping the
            # first ``S - n_to_drop`` positions, from the pre-window state.
            # Captures the per-position tensors by default-arg so each GDN
            # layer's closure is independent and evaluates lazily (q/k/v are
            # already materialized by the verify sync, so this is just the
            # short scan, no re-projection).
            _use_kernel = not self.training

            def _recompute_boundary(
                n_to_drop,
                *,
                _pre=pre_ssm,
                _q=q,
                _k=k,
                _v=v,
                _a=a,
                _b=b,
                _conv=conv_input,
                _mask=mask,
                _slen=S,
                _nk=n_keep,
                _alog=self.A_log,
                _dt=self.dt_bias,
                _uk=_use_kernel,
            ):
                keep = _slen - n_to_drop
                ms = _mask[:, :keep] if _mask is not None else None
                _, st_keep = gated_delta_update(
                    _q[:, :keep],
                    _k[:, :keep],
                    _v[:, :keep],
                    _a[:, :keep],
                    _b[:, :keep],
                    _alog,
                    _dt,
                    _pre,
                    ms,
                    use_kernel=_uk,
                )
                conv_keep = mx.contiguous(_conv[:, keep : keep + _nk, :])
                return conv_keep, st_keep

            cache.rollback_recompute = _recompute_boundary

            cache[1] = st
            # Advance by the FULL S — mirrors upstream cache.advance(S).
            cache.advance(S)

            out = self.norm(out, z)
            out = self.out_proj(out.reshape(B, S, -1))
            return out

        GatedDeltaNet.__call__ = _patched_call  # type: ignore[assignment]
        _gated_delta_patched = True
        logger.info(
            "[mtp.cache_patch] Installed GatedDeltaNet chunk-split for MTP "
            "rollback (snapshot at cache.n_confirmed_for_mtp boundary)."
        )
        return True


def _is_patched_for_tests() -> bool:
    """Test-only — inspect the install flag."""
    return _patched


def _unpatch_for_tests() -> None:
    """Test-only — clear the install flag and remove the class attr.

    Allows tests to verify the install side-effect by toggling the
    install state. Never called from production.
    """
    global _patched, _gated_delta_patched, _orig_gated_delta_call

    with _install_lock:
        try:
            from mlx_lm.models.cache import ArraysCache

            if "rollback_state" in ArraysCache.__dict__:
                delattr(ArraysCache, "rollback_state")
            if "n_confirmed_for_mtp" in ArraysCache.__dict__:
                delattr(ArraysCache, "n_confirmed_for_mtp")
        except ImportError:
            pass
        if _gated_delta_patched and _orig_gated_delta_call is not None:
            try:
                from mlx_lm.models.qwen3_5 import GatedDeltaNet

                GatedDeltaNet.__call__ = _orig_gated_delta_call  # type: ignore[assignment]
            except ImportError:
                pass
        _patched = False
        _gated_delta_patched = False
        _orig_gated_delta_call = None
