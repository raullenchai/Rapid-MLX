# SPDX-License-Identifier: Apache-2.0
"""Single-pass GatedDeltaNet rollback for chain-of-K MTP.

Locks the ``cache_patch`` contract that the verify forward runs as ONE
fused ``gated_delta_update`` (no per-boundary segment splitting) and
stashes a rollback CLOSURE on the cache. The closure recomputes the
``(conv, ssm)`` state after keeping the first ``S - n_to_drop`` verify
positions — byte-exact with the true ``(S - n_to_drop)``-token recurrent
state — so ``_rollback_draft(n)`` can restore to any accepted depth on a
partial chain-of-K accept without leaving rejected draft state in the
cache.

Regression for the single-pass rewrite (PR perf/mtp-single-pass-gdn-
rollback): the previous implementation split the verify into ``K+1``
separate ``gated_delta_update`` calls to materialize a snapshot at every
boundary — ~K+1 GDN kernel launches per layer per round, measured as the
dominant MTP verify cost. The single-pass path pays one fused scan and
only recomputes on the rare rejection.
"""

import mlx.core as mx
import pytest


def _build_gated_delta_net():
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
    )
    layer = GatedDeltaNet(args)
    # Inference mode: ``gated_delta_update(use_kernel=not self.training)`` — so
    # eval() exercises the SAME Metal-kernel segmented path production uses
    # (training mode would silently fall to the pure-ops reference instead).
    layer.eval()
    mx.eval(layer.parameters())
    return layer, args


@pytest.mark.parametrize("K", [1, 2, 3])
def test_rollback_closure_recomputes_true_boundary(K):  # noqa: N803  (K = draft width)
    """The stashed rollback closure recomputes the exact ``(S - n)``-token
    recurrent state for every ``n_to_drop`` in 1..K."""
    mx.random.seed(0)
    from mlx_lm.models.cache import ArraysCache

    from vllm_mlx.spec_decode.mtp import cache_patch

    # Install the single-pass patch; keep a handle to the unpatched forward
    # for the independent reference states.
    cache_patch.patch_gated_delta_net_for_mtp()
    orig_call = cache_patch._orig_gated_delta_call
    assert orig_call is not None

    layer, args = _build_gated_delta_net()

    # A FIXED prefix + verify window so every cache starts from the same state.
    prefix = mx.random.normal((1, 3, args.hidden_size))
    verify = mx.random.normal((1, K + 1, args.hidden_size))
    mx.eval(prefix, verify)

    def fresh_cache():
        c = ArraysCache(size=2)
        orig_call(layer, prefix, mask=None, cache=c)
        mx.eval(c[0], c[1])
        return c

    # Single-pass patched run over the (K+1)-token verify window.
    c_pat = fresh_cache()
    c_pat.snapshot_offsets = list(range(1, K + 1))
    layer(verify, mask=None, cache=c_pat)
    mx.eval(c_pat[0], c_pat[1])

    assert c_pat.rollback_recompute is not None, (
        "single-pass verify must stash a rollback closure"
    )

    # For each n_to_drop, the closure must reproduce the state after
    # processing the first (S - n) verify tokens, computed independently via
    # the original forward. The closure rescans ``[0:keep]`` from the same
    # pre-window anchor with the same kernel, so a tight float tolerance —
    # not exact equality — is the correct bar (observed divergence ~1e-7).
    S = K + 1
    for n in range(1, K + 1):
        c_ref = fresh_cache()
        orig_call(layer, verify[:, : S - n], mask=None, cache=c_ref)
        mx.eval(c_ref[0], c_ref[1])
        conv_keep, ssm_keep = c_pat.rollback_recompute(n)
        mx.eval(conv_keep, ssm_keep)
        assert float(mx.max(mx.abs(conv_keep - c_ref[0]))) < 1e-5, (
            f"conv rollback at n_to_drop={n} diverges from the true {S - n}-token state"
        )
        assert float(mx.max(mx.abs(ssm_keep - c_ref[1]))) < 1e-5, (
            f"ssm rollback at n_to_drop={n} diverges from the true {S - n}-token state"
        )


def test_patched_forward_output_matches_unsplit():
    """The single-pass verify forward is byte-equal to the unpatched forward
    for the confirmed tokens (it must not change the output or the final
    cache state — it only additionally stashes the rollback closure)."""
    mx.random.seed(1)
    from mlx_lm.models.cache import ArraysCache

    from vllm_mlx.spec_decode.mtp import cache_patch

    cache_patch.patch_gated_delta_net_for_mtp()
    orig_call = cache_patch._orig_gated_delta_call
    layer, args = _build_gated_delta_net()

    prefix = mx.random.normal((1, 3, args.hidden_size))
    verify = mx.random.normal((1, 3, args.hidden_size))  # K=2 verify window
    mx.eval(prefix, verify)

    def fresh_cache():
        c = ArraysCache(size=2)
        orig_call(layer, prefix, mask=None, cache=c)
        mx.eval(c[0], c[1])
        return c

    c_orig = fresh_cache()
    out_orig = orig_call(layer, verify, mask=None, cache=c_orig)
    mx.eval(out_orig, c_orig[0], c_orig[1])

    c_pat = fresh_cache()
    c_pat.snapshot_offsets = [1, 2]
    out_pat = layer(verify, mask=None, cache=c_pat)
    mx.eval(out_pat, c_pat[0], c_pat[1])

    # EXACT equality — the single-pass path runs the SAME one fused
    # ``gated_delta_update`` over the same tokens as the unpatched forward,
    # so per-token logits and the final (conv, ssm) state are bit-identical.
    # A tolerance here would let a logit-changing regression slip through and
    # quietly break the lossless speculative-decoding contract.
    assert bool(mx.array_equal(out_orig, out_pat)), "single-pass changed the output"
    assert bool(mx.array_equal(c_orig[0], c_pat[0])), "single-pass changed conv state"
    assert bool(mx.array_equal(c_orig[1], c_pat[1])), "single-pass changed ssm state"


def test_fast_path_leaves_no_rollback_closure():
    """A forward that requests NO snapshot (no ``snapshot_offsets``, ``S < 2``,
    or a tensor-parallel bail) must NOT leave a ``rollback_recompute`` closure
    on the cache.

    This is the safety precondition behind the generator's ``_rollback_draft``
    early-abort guard: if a reject — or an early abort that re-enters the
    rewind path — tries to roll back a cache that never snapshotted, the
    ``None`` sentinel makes ``_rollback_draft`` raise LOUDLY rather than
    silently reuse a stale closure and mis-rewind the recurrent state. A
    fast-path forward that silently left a stale closure behind would let an
    abort rewind to the wrong boundary and corrupt the SSM state undetected.
    """
    mx.random.seed(2)
    from mlx_lm.models.cache import ArraysCache

    from vllm_mlx.spec_decode.mtp import cache_patch

    cache_patch.patch_gated_delta_net_for_mtp()
    orig_call = cache_patch._orig_gated_delta_call
    layer, args = _build_gated_delta_net()

    prefix = mx.random.normal((1, 3, args.hidden_size))
    verify = mx.random.normal((1, 3, args.hidden_size))
    mx.eval(prefix, verify)

    c = ArraysCache(size=2)
    orig_call(layer, prefix, mask=None, cache=c)
    mx.eval(c[0], c[1])
    assert c.rollback_recompute is None  # nothing stashed yet

    # No snapshot requested -> the patched call takes the fast single-pass
    # branch and must leave ``rollback_recompute`` untouched (class default).
    assert getattr(c, "snapshot_offsets", None) is None
    layer(verify, mask=None, cache=c)
    mx.eval(c[0], c[1])
    assert c.rollback_recompute is None, (
        "a fast-path (no-snapshot) forward must not stash a rollback closure — "
        "else an abort/reject could silently reuse a stale rewind"
    )
