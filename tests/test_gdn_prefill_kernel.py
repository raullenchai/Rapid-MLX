# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the blocked-seq GDN prefill kernel.

The kernel computes the exact same sequential recurrence as mlx-lm's stock
``gated_delta_kernel`` — any numeric divergence beyond fp32-accumulation
noise is a bug, so these tests compare against both the stock Metal kernel
and the pure-ops reference on random Qwen-shaped inputs. Fallback gates are
exercised so masked / vectorized-gating / decode-step calls provably keep
the stock path.
"""

from __future__ import annotations

import importlib

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.requires_mlx

if not mx.metal.is_available():  # pragma: no cover - CI runners have Metal
    pytest.skip("Metal GPU required for GDN kernel tests", allow_module_level=True)

from mlx_lm.models import gated_delta as gd

from vllm_mlx import gdn_prefill


def _stock_kernel():
    """The unwrapped mlx-lm kernel, regardless of whether install() ran.

    Another test module importing ``vllm_mlx.scheduler`` installs the
    wrapper at collection time; comparing against the module attribute
    would then be fast-path-versus-fast-path. The wrapper carries the
    true stock implementation on ``._stock``.
    """
    fn = gd.gated_delta_kernel
    return getattr(fn, "_stock", fn)


# Qwen3.8-27B GDN geometry (scaled-down head counts, same head dims).
B, HK, HV, DK, DV = 1, 2, 6, 128, 128


def _inputs(t_len, dtype=mx.bfloat16, seed=7, g_ndim=3):
    mx.random.seed(seed)
    q = (mx.random.normal((B, t_len, HK, DK)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, t_len, HK, DK)) * 0.1).astype(dtype)
    v = (mx.random.normal((B, t_len, HV, DV)) * 0.1).astype(dtype)
    if g_ndim == 3:
        g = mx.sigmoid(mx.random.normal((B, t_len, HV)).astype(mx.float32))
    else:
        g = mx.sigmoid(mx.random.normal((B, t_len, HV, DK)).astype(mx.float32))
    beta = mx.sigmoid(mx.random.normal((B, t_len, HV)).astype(mx.float32))
    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def _max_rel_err(a: mx.array, b: mx.array, atol: float = 1e-3) -> float:
    """Elementwise stabilized relative error (max over elements).

    Dividing the global max error by the global max magnitude would let a
    small-magnitude element be badly corrupted while the metric stays
    tiny; normalizing per element (with an absolute floor at the data's
    noise scale — inputs are O(0.1..1)) catches that: real corruption is
    O(1) under this metric, while fp32 accumulation-order noise on
    near-zero elements stays under 1e-3.
    """
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    rel = mx.abs(a32 - b32) / (mx.abs(a32) + atol)
    return float(rel.max())


class TestBlockedSeqNumerics:
    @pytest.mark.parametrize("t_len", [64, 100, 512])
    def test_matches_stock_kernel_bf16(self, t_len):
        q, k, v, g, beta, state = _inputs(t_len)
        y_fast, st_fast = gdn_prefill.gated_delta_blocked_seq(q, k, v, g, beta, state)
        y_stock, st_stock = _stock_kernel()(q, k, v, g, beta, state, None)
        mx.eval(y_fast, st_fast, y_stock, st_stock)
        # Both kernels accumulate in fp32; outputs round to bf16 so allow
        # one-ulp-of-bf16 divergence on y, near-exactness on the fp32 state.
        assert _max_rel_err(y_stock, y_fast) < 2e-2
        assert _max_rel_err(st_stock, st_fast) < 1e-3

    def test_matches_ops_reference_fp32(self):
        # fp32 end-to-end pins the algorithm itself (no rounding slack).
        q, k, v, g, beta, state = _inputs(96, dtype=mx.float32)
        y_fast, st_fast = gdn_prefill.gated_delta_blocked_seq(q, k, v, g, beta, state)
        y_ref, st_ref = gd.gated_delta_ops(q, k, v, g, beta, state, None)
        mx.eval(y_fast, st_fast, y_ref, st_ref)
        assert _max_rel_err(y_ref, y_fast) < 1e-3
        assert _max_rel_err(st_ref, st_fast) < 1e-3

    def test_state_chains_across_calls(self):
        # Split a 128-token prompt into two 64-token chunks: chained fast
        # calls must equal one full stock call (chunked-prefill invariant).
        q, k, v, g, beta, state = _inputs(128)
        y_full, st_full = _stock_kernel()(q, k, v, g, beta, state, None)
        y1, st1 = gdn_prefill.gated_delta_blocked_seq(
            q[:, :64], k[:, :64], v[:, :64], g[:, :64], beta[:, :64], state
        )
        y2, st2 = gdn_prefill.gated_delta_blocked_seq(
            q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], beta[:, 64:], st1
        )
        y_chain = mx.concatenate([y1, y2], axis=1)
        mx.eval(y_full, st_full, y_chain, st2)
        assert _max_rel_err(y_full, y_chain) < 2e-2
        assert _max_rel_err(st_full, st2) < 1e-3


class TestBlockedSeqValidation:
    def test_rejects_wrong_dk(self):
        q, k, v, g, beta, state = _inputs(64)
        q_bad = mx.concatenate([q, q], axis=-1)  # Dk=256
        k_bad = mx.concatenate([k, k], axis=-1)
        with pytest.raises(ValueError, match="Dk == 128"):
            gdn_prefill.gated_delta_blocked_seq(q_bad, k_bad, v, g, beta, None)

    def test_rejects_unaligned_dv(self):
        q, k, v, g, beta, state = _inputs(64)
        v_bad = v[..., :100]  # Dv=100, not a multiple of 32
        with pytest.raises(ValueError, match="Dv % 32"):
            gdn_prefill.gated_delta_blocked_seq(q, k, v_bad, g, beta, None)

    def test_rejects_wrong_state_layout(self):
        q, k, v, g, beta, state = _inputs(64)
        bad = mx.zeros((B, HV, DV // 2, DK), dtype=mx.float32)
        with pytest.raises(ValueError, match="fp32 state of shape"):
            gdn_prefill.gated_delta_blocked_seq(q, k, v, g, beta, bad)

    def test_state_none_initializes_and_matches_stock(self):
        # First prefill chunk arrives with state=None: the fast path must
        # zero-init fp32 state exactly like the stock update path.
        q, k, v, g, beta, state = _inputs(64)
        y_fast, st_fast = gdn_prefill.gated_delta_blocked_seq(q, k, v, g, beta, None)
        y_stock, st_stock = _stock_kernel()(q, k, v, g, beta, state, None)
        mx.eval(y_fast, st_fast, y_stock, st_stock)
        assert _max_rel_err(y_stock, y_fast) < 2e-2
        assert _max_rel_err(st_stock, st_fast) < 1e-3


class TestEligibilityGate:
    def test_fast_path_shapes_pass(self):
        q, k, v, g, beta, state = _inputs(64)
        assert gdn_prefill._eligible(q, k, v, g, beta, state, None)

    def test_state_none_is_eligible(self):
        # state=None must not crash the gate (reading .dtype off None would
        # raise before the fast path could zero-init) and is a valid
        # fast-path shape.
        q, k, v, g, beta, _ = _inputs(64)
        assert gdn_prefill._eligible(q, k, v, g, beta, None, None)

    @pytest.mark.parametrize(
        "mutation",
        [
            "masked",
            "vector_gating",
            "decode_step",
            "wrong_dk",
            "bf16_state",
            "hv_not_multiple_of_hk",
            "state_shape_mismatch",
        ],
    )
    def test_fallback_shapes_fail(self, mutation):
        q, k, v, g, beta, state = _inputs(64)
        mask = None
        if mutation == "masked":
            mask = mx.ones((B, 64), dtype=mx.bool_)
        elif mutation == "vector_gating":
            _, _, _, g, _, _ = _inputs(64, g_ndim=4)
        elif mutation == "decode_step":
            # Truncate EVERY tensor to one token so only the minimum-token
            # gate (not a [B, T] mismatch) determines the fallback.
            q, k, v, g, beta = (x[:, :1] for x in (q, k, v, g, beta))
        elif mutation == "wrong_dk":
            q = mx.concatenate([q, q], axis=-1)  # Dk=256
            k = mx.concatenate([k, k], axis=-1)
        elif mutation == "bf16_state":
            state = state.astype(mx.bfloat16)
        elif mutation == "hv_not_multiple_of_hk":
            # Hv=5 with Hk=2: the kernel's hk = hv / (Hv / Hk) head map
            # would be wrong (and Hv < Hk would divide by zero).
            v = v[:, :, :5]
            g = g[:, :, :5]
            beta = beta[:, :, :5]
            state = state[:, :5]
        elif mutation == "state_shape_mismatch":
            # fp32 but the wrong layout: pointer arithmetic would read/write
            # out of bounds, so the gate must reject it.
            state = mx.zeros((B, HV, DV // 2, DK), dtype=mx.float32)
        assert not gdn_prefill._eligible(q, k, v, g, beta, state, mask)


class TestInstall:
    """Each test restores the TRUE stock kernel first: another test module
    importing ``vllm_mlx.scheduler`` may have installed the wrapper at
    collection time, and reload+install against an already-wrapped module
    attribute would otherwise test the wrong object.
    """

    @pytest.fixture(autouse=True)
    def _restore_module_state(self):
        # Snapshot and restore BOTH the patched mlx-lm function and the
        # gdn_prefill module globals (_installed / _original_kernel):
        # restoring only the function would leave a reloaded module claiming
        # "installed" with nothing bound, making later tests order-dependent.
        fn_before = gd.gated_delta_kernel
        installed_before = gdn_prefill._installed
        original_before = gdn_prefill._original_kernel
        yield
        gd.gated_delta_kernel = fn_before
        gdn_prefill._installed = installed_before
        gdn_prefill._original_kernel = original_before

    def _fresh(self):
        stock = _stock_kernel()
        gd.gated_delta_kernel = stock
        mod = importlib.reload(gdn_prefill)
        return mod, stock

    def test_install_wraps_and_is_idempotent(self, monkeypatch):
        mod, stock = self._fresh()
        try:
            assert mod.install() is True
            wrapped = gd.gated_delta_kernel
            assert wrapped is not stock
            assert wrapped._stock is stock
            # second install is a no-op
            assert mod.install() is True
            assert gd.gated_delta_kernel is wrapped
        finally:
            gd.gated_delta_kernel = stock

    def test_install_never_wraps_the_wrapper(self, monkeypatch):
        # A reloaded module copy must detect an already-bound wrapper and
        # leave it alone — double-wrapping would make the stock fallback
        # recurse back into the fast path.
        mod, stock = self._fresh()
        try:
            assert mod.install() is True
            wrapped = gd.gated_delta_kernel
            mod2 = importlib.reload(gdn_prefill)  # resets module globals
            assert mod2.install() is True
            assert gd.gated_delta_kernel is wrapped  # not re-wrapped
            assert gd.gated_delta_kernel._stock is stock
        finally:
            gd.gated_delta_kernel = stock

    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_GDN_PREFILL", "0")
        mod, stock = self._fresh()
        try:
            assert mod.install() is False
            assert gd.gated_delta_kernel is stock
        finally:
            gd.gated_delta_kernel = stock

    def test_fast_path_failure_degrades_to_stock_permanently(self, monkeypatch):
        # A fast-path exception (e.g. Metal JIT rejection on an exotic GPU)
        # must fall back to the stock kernel and stay there — it can never
        # crash a request.
        mod, stock = self._fresh()
        calls = {"stock": 0}

        def counting_stock(*args, **kwargs):
            calls["stock"] += 1
            return stock(*args, **kwargs)

        gd.gated_delta_kernel = counting_stock
        assert mod.install() is True

        real_blocked_seq = mod.gated_delta_blocked_seq

        def boom(*args, **kwargs):
            raise RuntimeError("simulated Metal JIT failure")

        monkeypatch.setattr(mod, "gated_delta_blocked_seq", boom)
        q, k, v, g, beta, state = _inputs(64)
        y1, st1 = gd.gated_delta_kernel(q, k, v, g, beta, state, None)
        mx.eval(y1, st1)
        assert calls["stock"] == 1  # degraded, not crashed
        # Restore the REAL fast path behind a counter: the wrapper must not
        # invoke it again — proving fast_path_dead persisted, not merely
        # that a second exception also fell through to stock.
        calls_fast = {"n": 0}

        def counting_fast(*args, **kwargs):
            calls_fast["n"] += 1
            return real_blocked_seq(*args, **kwargs)

        monkeypatch.setattr(mod, "gated_delta_blocked_seq", counting_fast)
        y2, st2 = gd.gated_delta_kernel(q, k, v, g, beta, state, None)
        mx.eval(y2, st2)
        assert calls["stock"] == 2
        assert calls_fast["n"] == 0

    def test_wrapper_falls_back_for_masked_input(self, monkeypatch):
        mod, stock = self._fresh()
        calls = {"stock": 0}

        def counting_stock(*args, **kwargs):
            calls["stock"] += 1
            return stock(*args, **kwargs)

        gd.gated_delta_kernel = counting_stock
        try:
            assert mod.install() is True
            q, k, v, g, beta, state = _inputs(64)
            mask = mx.ones((B, 64), dtype=mx.bool_)
            y, st = gd.gated_delta_kernel(q, k, v, g, beta, state, mask)
            mx.eval(y, st)
            assert calls["stock"] == 1
        finally:
            gd.gated_delta_kernel = stock
