# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the architecture-aware KV footprint estimator.

These pin the contract that the D-METAL-CAP admission gate depends on
(``scheduler._resolve_kv_bytes_per_token`` → ``estimate_kv_footprint`` /
``rotating_cache_slots``):

* Dense / unknown configs are BYTE-IDENTICAL to the uniform formula: the
  estimate is ``(uniform, 0, 0, 0)`` — no baseline, no sliding term.
* Recognized hybrids get the accurate per-request footprint split across three
  terms: only full-attention layers grow unbounded per token
  (``per_token_growth_bytes``); sliding-window layers are a request-dependent
  slot term (``sliding_slot_bytes`` × ``rotating_cache_slots(window, T)``);
  recurrent layers reserve a token-independent fixed state
  (``fixed_baseline_bytes``).
* The estimate is over-count-safe throughout — it never under-counts the counted
  layers. A sliding layer with no readable window (or an unknown layer type) is
  folded into full unbounded growth; a sliding layer WITH a window contributes a
  slot term whose per-request slot count is an upper bound on the real
  ``RotatingKVCache`` allocation at every token count, capped at the full window.
* KV-sharing zeroes a borrower ONLY when the exact per-index producer map
  (mirrored from ``models/gemma4_vendored/language.py``) gives it a same-type
  producer below the split; an orphan borrower is charged, never zeroed.

Hermetic: pure config dataclasses (``SimpleNamespace``) and dicts — no model
load, no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapid_mlx.kv_estimation import (
    _ROTATING_CACHE_KEEP,
    _ROTATING_CACHE_STEP,
    KVFootprintEstimate,
    estimate_kv_footprint,
    rotating_cache_slots,
)

# Rotating-cache granularity — mirrors kv_estimation's ``_ROTATING_CACHE_STEP`` /
# ``_ROTATING_CACHE_KEEP`` (sourced from mlx_lm's ``RotatingKVCache.step = 256``;
# ``KEEP = 4`` is the largest ``keep`` any shipped construction uses, so the slot
# count is an upper bound for every path).
_STEP = 256
_KEEP = 4


def _slots(window: int, tokens: int) -> int:
    """Reference re-implementation of ``rotating_cache_slots`` for assertions."""
    if window <= 0 or tokens <= 0:
        return 0
    effective = min(tokens, window) + _KEEP
    return ((effective + _STEP - 1) // _STEP) * _STEP


def _uniform(num_layers: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    return 2 * num_layers * kv_heads * head_dim * dtype_bytes


def _est(cfg, *, num_layers, kv_heads, head_dim, dtype_bytes=2) -> KVFootprintEstimate:
    return estimate_kv_footprint(
        cfg,
        dtype_bytes=dtype_bytes,
        uniform_per_token_bytes=_uniform(num_layers, kv_heads, head_dim, dtype_bytes),
        base_num_layers=num_layers,
        base_kv_heads=kv_heads,
        base_head_dim=head_dim,
    )


class TestRotatingCacheSlots:
    """The per-request slot count: an over-count-safe, sublinear upper bound."""

    def test_reference_agreement(self):
        for window, tokens in [(4096, 1), (128, 10_000), (500, 9_999), (256, 300)]:
            assert rotating_cache_slots(window, tokens) == _slots(window, tokens)

    def test_zero_window_or_tokens_is_zero(self):
        assert rotating_cache_slots(0, 100) == 0
        assert rotating_cache_slots(128, 0) == 0
        assert rotating_cache_slots(0, 0) == 0

    def test_short_request_far_below_full_window(self):
        # T=1 against a 4096 window rounds up (1 + keep) to ONE step block, not
        # the whole window — the codex-flagged over-count of short requests.
        assert rotating_cache_slots(4096, 1) == 256
        full = rotating_cache_slots(4096, 4096)
        assert full > 256  # short request charged far less than the full window

    def test_caps_at_full_window_for_long_request(self):
        window = 128
        full = rotating_cache_slots(window, window)  # ceil((128+4)/256)*256 = 256
        assert full == 256
        for tokens in (window, window + 1, 10_000, 1_000_000):
            assert rotating_cache_slots(window, tokens) == full  # flat past window

    def test_monotonic_non_decreasing_in_tokens(self):
        window = 4096
        prev = -1
        for tokens in range(1, 6000, 37):
            slots = rotating_cache_slots(window, tokens)
            assert slots >= prev
            prev = slots

    def test_is_upper_bound_of_real_buffer(self):
        # The real RotatingKVCache buffer after t tokens is
        # ``min(ceil(t/step)*step, max_size)`` (decode path). Our slot count adds
        # ``keep`` before rounding, so it is >= that real buffer for every t.
        window = 1000
        for tokens in (1, 100, 256, 257, 999, 1000, 1001, 5000):
            real = min(((tokens + _STEP - 1) // _STEP) * _STEP, window)
            assert rotating_cache_slots(window, tokens) >= real

    def test_step_aligned_multiples(self):
        # ceil((min(T,window)+keep)/step)*step is always a multiple of step and
        # >= min(T,window)+keep.
        for window, tokens in [(500, 5000), (513, 5000), (128, 50), (256, 256)]:
            slots = rotating_cache_slots(window, tokens)
            assert slots % _STEP == 0
            assert slots >= min(tokens, window) + _KEEP


class TestDenseFallback:
    """Anything not a recognized hybrid stays byte-identical to uniform."""

    def _assert_byte_identical(self, est, num_layers, kv, hd, dtype_bytes=2):
        assert est.per_token_growth_bytes == _uniform(num_layers, kv, hd, dtype_bytes)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0

    def test_dense_config_is_byte_identical(self):
        cfg = SimpleNamespace(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
        self._assert_byte_identical(
            _est(cfg, num_layers=32, kv_heads=8, head_dim=128), 32, 8, 128
        )

    def test_sliding_window_without_layer_types_falls_back(self):
        # Mistral-like: a global ``sliding_window`` field but NO per-layer
        # ``layer_types``. Ambiguous → uniform, no sliding term.
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            sliding_window=4096,
        )
        self._assert_byte_identical(
            _est(cfg, num_layers=32, kv_heads=8, head_dim=128), 32, 8, 128
        )

    def test_wrong_length_layer_types_falls_back(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=["full_attention"] * 30,  # len != num_layers
        )
        self._assert_byte_identical(
            _est(cfg, num_layers=32, kv_heads=8, head_dim=128), 32, 8, 128
        )

    def test_non_string_layer_types_falls_back(self):
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=[0, 1, 2, 3],  # not strings
        )
        self._assert_byte_identical(
            _est(cfg, num_layers=4, kv_heads=8, head_dim=128), 4, 8, 128
        )

    def test_zero_base_dims_returns_uniform(self):
        cfg = SimpleNamespace(num_hidden_layers=0)
        est = estimate_kv_footprint(
            cfg,
            dtype_bytes=2,
            uniform_per_token_bytes=0,
            base_num_layers=0,
            base_kv_heads=0,
            base_head_dim=0,
        )
        assert est.per_token_growth_bytes == 0
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0


class TestKVSharing:
    """Gemma-4 / Gemma-3n ``num_kv_shared_layers`` borrowers = 0 KV, validated
    against the EXACT per-index producer map (codex round 11 BLOCKING #2)."""

    def test_shared_layers_reduce_growth_exactly(self):
        # 35 layers, all full-attention, last 20 borrow → 15 producers grow.
        n, n_shared, kv, hd = 35, 20, 1, 128
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["full_attention"] * n,
            num_kv_shared_layers=n_shared,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * (n - n_shared) * kv * hd * 2
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0

    def test_shared_layers_without_layer_types_falls_back(self):
        # num_kv_shared_layers alone (no per-layer map) can't verify the borrower
        # map, so we stay on uniform (codex round 1 BLOCKING #3).
        n, n_shared, kv, hd = 40, 10, 4, 64
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            num_kv_shared_layers=n_shared,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0

    def test_codex_example_all_borrowers_validly_mapped_are_zeroed(self):
        # Codex round 11's counterexample: producers ``[full, sliding, full]``,
        # borrowers ``[full, full, sliding]``. Under the SHIPPED type-keyed map
        # (``models/gemma4_vendored/language.py``: each borrower reuses the LAST
        # same-type producer below the split) EVERY borrower here has a same-type
        # producer — full→idx2, full→idx2, sliding→idx1 — so zeroing all three is
        # CORRECT, not a mis-sourced mapping. Remaining charged: the 3 producers
        # (2 full grow per token, 1 sliding contributes a slot term).
        n, kv, hd, window = 6, 1, 64, 128
        layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",  # producers 0..2
            "full_attention",
            "full_attention",
            "sliding_attention",  # borrowers 3..5
        ]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            num_kv_shared_layers=3,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * 2 * kv * hd * 2  # idx0, idx2 full
        assert est.sliding_slot_bytes == 1 * 2 * kv * hd * 2  # idx1 sliding producer
        assert est.sliding_window == window
        assert est.fixed_baseline_bytes == 0

    def test_orphan_borrower_type_is_charged_not_zeroed(self):
        # The per-index map DIVERGES from a plain set check here: producers are
        # all ``full`` (idx0,1), borrowers are ``full`` (idx2) and ``sliding``
        # (idx3). The full borrower maps to a same-type producer → zeroed; the
        # sliding borrower has NO sliding producer below the split → it is NOT
        # zeroed but charged its own (sliding) footprint. A set-subset test would
        # instead refuse to zero ANYTHING; the exact per-index map correctly
        # zeroes the mappable borrower and charges only the orphan.
        n, kv, hd, window = 4, 1, 64, 128
        layer_types = [
            "full_attention",
            "full_attention",  # producers 0..1
            "full_attention",  # borrower 2 → maps to a full producer → zeroed
            "sliding_attention",  # borrower 3 → orphan → charged as sliding
        ]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            num_kv_shared_layers=2,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * 2 * kv * hd * 2  # idx0, idx1 full
        assert est.sliding_slot_bytes == 1 * 2 * kv * hd * 2  # idx3 orphan charged
        assert est.sliding_window == window

    def test_recurrent_layer_in_borrower_range_is_charged_not_zeroed(self):
        # A recurrent (linear_attention) layer that sits in the last-N borrower
        # positions and whose type matches an earlier recurrent producer must NOT
        # be zeroed: KV sharing shares ATTENTION K/V, never recurrent state. It is
        # charged its own fixed recurrent state (codex round 13 BLOCKING #1). The
        # full borrower (idx2) still maps to a full producer and is zeroed.
        n, kv, hd = 4, 2, 128
        gdn = dict(
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        layer_types = [
            "linear_attention",  # 0 producer (recurrent)
            "full_attention",  # 1 producer (full)
            "full_attention",  # 2 borrower → maps to full producer 1 → zeroed
            "linear_attention",  # 3 borrower, recurrent → charged, NOT zeroed
        ]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            num_kv_shared_layers=2,
            **gdn,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        key_dim = 16 * 128
        value_dim = 32 * 128
        state = 4 * (32 * 128 * 128 + (4 - 1) * (2 * key_dim + value_dim))
        # idx0 producer + idx3 recurrent borrower (NOT zeroed) → both charged.
        assert est.fixed_baseline_bytes == 2 * state
        # idx1 full producer grows; idx2 full borrower is zeroed.
        assert est.per_token_growth_bytes == 2 * kv * hd * 2
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0

    def test_zero_shared_is_not_a_hybrid_signal(self):
        cfg = SimpleNamespace(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            num_kv_shared_layers=0,
        )
        est = _est(cfg, num_layers=32, kv_heads=8, head_dim=128)
        assert est.per_token_growth_bytes == _uniform(32, 8, 128, 2)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0


class TestSlidingWindow:
    """GPT-OSS ~50% sliding layers → a request-dependent slot term, not growth."""

    def test_half_sliding_halves_growth_and_exposes_slot_term(self):
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        n_full = n // 2
        n_sliding = n // 2
        # Only the full layers grow unbounded per token — halved vs uniform.
        assert est.per_token_growth_bytes == 2 * n_full * kv * hd * 2
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2) // 2
        # Sliding layers contribute a per-SLOT term (one slot's bytes across all
        # sliding layers) plus the shared window — NOT a fixed whole-window
        # baseline (codex round 11 BLOCKING #1).
        assert est.sliding_slot_bytes == 2 * n_sliding * kv * hd * 2
        assert est.sliding_window == window
        assert est.fixed_baseline_bytes == 0

    def test_sliding_without_window_folds_into_full_growth(self):
        # A sliding layer whose window is unreadable cannot be bounded → folded
        # into full per-token growth; no sliding term is advertised.
        n, kv, hd = 24, 8, 64
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            # sliding_window intentionally absent
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0
        assert est.fixed_baseline_bytes == 0

    def test_all_sliding_has_zero_per_token_growth(self):
        n, kv, hd, window = 8, 4, 64, 256
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["sliding_attention"] * n,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 0
        assert est.sliding_slot_bytes == 2 * n * kv * hd * 2
        assert est.sliding_window == window

    def test_per_layer_window_list_folds_into_full_growth(self):
        # The per-request slot model charges every sliding layer with ONE window,
        # so it is only sound for a single positive SCALAR window. A per-layer
        # window LIST is never collapsed to one scalar (scalar-only policy) — it
        # folds every sliding layer into full per-token growth (over-count-safe;
        # codex round 13 BLOCKING #3). Differing entries here would obviously
        # under-count if reduced to the min/first.
        n, kv, hd = 24, 8, 64
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=[128, 4096] * (n // 2),  # differing per-layer windows
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # A list window → every layer grows unbounded (uniform result).
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0
        assert est.fixed_baseline_bytes == 0

    def test_partial_window_list_does_not_borrow_for_unknown_sliding_layer(self):
        # codex counterexample: a MISALIGNED per-layer window list where one
        # sliding layer has a positive window and ANOTHER sliding layer's entry is
        # null (unknown). Borrowing the known 128 for the unknown-window sliding
        # layer would under-reserve if its real window were larger, so ANY list
        # shape — even one with a single distinct positive value — folds every
        # sliding layer into full per-token growth. The estimate must NOT expose a
        # slot term or a window here.
        n, kv, hd = 4, 8, 64
        layer_types = [
            "sliding_attention",  # 0 — window 128
            "sliding_attention",  # 1 — window UNKNOWN (null)
            "full_attention",
            "full_attention",
        ]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=[128, None, None, None],  # only one distinct positive
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)  # all full growth
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0
        assert est.fixed_baseline_bytes == 0


class TestRecurrent:
    """Qwen3.5/3.6 GatedDeltaNet layers have zero PER-TOKEN growth; their fixed
    state lands in the token-independent baseline (never the sliding term)."""

    _GDN_FIELDS = dict(
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

    @staticmethod
    def _gdn_state_bytes():
        num_v, num_k, hk, hv, conv = 32, 16, 128, 128, 4
        key_dim = num_k * hk
        value_dim = num_v * hv
        recurrent = num_v * hk * hv
        conv_dim = 2 * key_dim + value_dim
        conv_state = (conv - 1) * conv_dim
        return 4 * (recurrent + conv_state)

    def test_gateddeltanet_state_reserved_in_baseline_zero_growth(self):
        n, kv, hd = 48, 2, 128
        pattern = ["linear_attention"] * 3 + ["full_attention"]
        layer_types = pattern * (n // 4)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            **self._GDN_FIELDS,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        n_full = n // 4
        n_recurrent = n - n_full
        assert est.per_token_growth_bytes == 2 * n_full * kv * hd * 2
        assert est.fixed_baseline_bytes == n_recurrent * self._gdn_state_bytes()
        assert est.sliding_slot_bytes == 0
        assert est.sliding_window == 0

    @pytest.mark.parametrize(
        "gdn_type",
        ["linear_attention", "gated_delta_net", "gated_deltanet"],
    )
    def test_gateddeltanet_aliases_all_sized(self, gdn_type):
        n, kv, hd = 8, 4, 64
        layer_types = [gdn_type] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            **self._GDN_FIELDS,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * 1 * kv * hd * 2
        assert est.fixed_baseline_bytes == (n - 1) * self._gdn_state_bytes()
        assert est.sliding_slot_bytes == 0

    def test_unsizeable_recurrent_falls_back_to_full_growth(self):
        n, kv, hd = 8, 4, 64
        layer_types = ["linear_attention"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0

    def test_mamba_not_sized_falls_back_to_full_growth(self):
        n, kv, hd = 8, 4, 64
        layer_types = ["mamba"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_n_heads=24,
            mamba_d_head=64,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0

    def test_gateddeltanet_missing_conv_kernel_falls_back(self):
        n, kv, hd = 8, 4, 64
        layer_types = ["linear_attention"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            # linear_conv_kernel_dim intentionally absent
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0

    def test_full_attention_name_containing_linear_is_not_recurrent(self):
        n, kv, hd = 6, 4, 64
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["gated_full_attention", "linear_global_attention"] * (n // 2),
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0

    def test_generic_recurrent_family_not_mis_sized(self):
        n, kv, hd = 8, 4, 64
        layer_types = ["rwkv"] * (n - 1) + ["full_attention"]
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            state_size=128,
            intermediate_size=1536,
            conv_kernel_size=4,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.fixed_baseline_bytes == 0
        assert est.sliding_slot_bytes == 0


class TestHybridDimsAndNesting:
    def test_global_head_dim_used_for_full_layers(self):
        # Realistic Gemma-4 text: 35 layers, 4:1 sliding:full, last 20 borrow,
        # full layers use the wider global_head_dim/global kv heads.
        n, n_shared, kv, hd = 35, 20, 1, 256
        global_hd, global_kv, window = 512, 1, 512
        layer_types = (["sliding_attention"] * 4 + ["full_attention"]) * (n // 5)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            global_head_dim=global_hd,
            num_global_key_value_heads=global_kv,
            num_kv_shared_layers=n_shared,
            sliding_window=window,
            layer_types=layer_types,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        # Producers = first 15 layers. Full at indices 4, 9, 14 → 3 full,
        # 12 sliding. Borrowers (last 20) contribute nothing.
        assert est.per_token_growth_bytes == 2 * 3 * global_kv * global_hd * 2
        # Sliding layers use local dims for the per-slot term.
        assert est.sliding_slot_bytes == 12 * 2 * kv * hd * 2
        assert est.sliding_window == window
        assert est.fixed_baseline_bytes == 0

    def test_layer_structure_read_from_text_config(self):
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(
                num_key_value_heads=kv,
                head_dim=hd,
                layer_types=layer_types,
                sliding_window=window,
            )
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * (n // 2) * kv * hd * 2
        assert est.sliding_slot_bytes == 2 * (n // 2) * kv * hd * 2
        assert est.sliding_window == window

    def test_dict_config_supported(self):
        # The offline hybrid probe reads parsed config.json dicts.
        n, kv, hd, window = 24, 8, 64, 128
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = {
            "num_hidden_layers": n,
            "num_key_value_heads": kv,
            "head_dim": hd,
            "layer_types": layer_types,
            "sliding_window": window,
        }
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == 2 * (n // 2) * kv * hd * 2
        assert est.sliding_slot_bytes == 2 * (n // 2) * kv * hd * 2
        assert est.sliding_window == window

    def test_unknown_layer_type_charged_as_full_growth(self):
        n, kv, hd = 4, 8, 128
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["mystery_attention"] * n,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, 2)
        assert est.sliding_slot_bytes == 0
        assert est.fixed_baseline_bytes == 0


class TestNeverUnderCounts:
    """The per-request projection is always >= the true counted-layer footprint,
    at every token count — and never over-counts a short request past the full
    window."""

    def _project(self, est, tokens):
        return (
            est.fixed_baseline_bytes
            + est.per_token_growth_bytes * tokens
            + est.sliding_slot_bytes * rotating_cache_slots(est.sliding_window, tokens)
        )

    def test_projection_upper_bounds_true_footprint(self):
        # GPT-OSS-like hybrid. For ANY token count the projection must be >= the
        # true footprint of the counted layers:
        #   full layers: exact tokens × per-layer
        #   sliding layers: min(tokens, window) × per-layer  (real buffer ≤ this,
        #                   rounded up to step blocks — bounded by our slot count)
        n, kv, hd, window, db = 24, 8, 64, 128, 2
        layer_types = ["sliding_attention", "full_attention"] * (n // 2)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        n_full = n // 2
        n_sliding = n // 2
        per_layer = 2 * kv * hd * db
        for tokens in (1, 100, window, window + 1, 4096, 32768):
            projected = self._project(est, tokens)
            true_full = n_full * tokens * per_layer
            true_sliding = n_sliding * min(tokens, window) * per_layer
            assert projected >= true_full + true_sliding

    def test_short_request_not_over_charged_full_window(self):
        # The codex round 11 win: a SHORT request (T=1) against a big window is
        # NOT charged the whole window. Its sliding term is one step block, far
        # below what a fixed whole-window baseline would have charged.
        n, kv, hd, window, db = 8, 4, 64, 4096, 2
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["sliding_attention"] * n,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        short = self._project(est, 1)
        full_window = est.sliding_slot_bytes * rotating_cache_slots(window, window)
        assert short == est.sliding_slot_bytes * 256  # one step block
        assert short < full_window  # strictly less than the whole-window charge

    def test_monotonic_and_capped_projection(self):
        n, kv, hd, window, db = 8, 4, 64, 512, 2
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["sliding_attention", "full_attention"] * (n // 2),
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        # The sliding CONTRIBUTION alone is monotonic non-decreasing and caps at
        # the full-window buffer.
        cap = est.sliding_slot_bytes * rotating_cache_slots(window, window)
        prev = -1
        for tokens in (1, 200, 511, 512, 513, 5000):
            slide = est.sliding_slot_bytes * rotating_cache_slots(window, tokens)
            assert slide >= prev
            assert slide <= cap
            prev = slide

    def test_full_layer_growth_never_below_uniform_base(self):
        n, kv, hd, db = 8, 8, 128, 2
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=["full_attention"] * n,
            global_head_dim=16,  # << base head_dim
            num_global_key_value_heads=1,  # << base kv heads
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        assert est.per_token_growth_bytes == _uniform(n, kv, hd, db)
        assert est.sliding_slot_bytes == 0

    def test_sliding_slot_rate_never_below_uniform_base(self):
        # A nested/malformed config whose local (sliding) dims read SMALLER than
        # the base dims must not shrink the per-slot rate below the uniform
        # per-layer charge (codex round 7 BLOCKING #2).
        n, kv, hd, db, window = 8, 8, 128, 2, 256
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=8,  # << base head_dim (128) — malformed / non-authoritative
            layer_types=["sliding_attention"] * n,
            sliding_window=window,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        # Per-slot term at the clamped base per-layer rate, not the tiny local
        # dims (which would under-count).
        assert est.sliding_slot_bytes == n * 2 * kv * hd * db
        assert est.per_token_growth_bytes == 0

    def test_projection_covers_recurrent_fixed_state(self):
        n, kv, hd, db = 16, 2, 128, 2
        gdn = dict(
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        layer_types = (["linear_attention"] * 3 + ["full_attention"]) * (n // 4)
        cfg = SimpleNamespace(
            num_hidden_layers=n,
            num_key_value_heads=kv,
            head_dim=hd,
            layer_types=layer_types,
            **gdn,
        )
        est = _est(cfg, num_layers=n, kv_heads=kv, head_dim=hd, dtype_bytes=db)
        n_full = n // 4
        n_recurrent = n - n_full
        key_dim = 16 * 128
        value_dim = 32 * 128
        state = 4 * (32 * 128 * 128 + (4 - 1) * (2 * key_dim + value_dim))
        assert est.fixed_baseline_bytes == n_recurrent * state
        full_per_layer = 2 * kv * hd * db
        for tokens in (1, 100, 4096, 32768):
            projected = self._project(est, tokens)
            assert projected >= n_full * tokens * full_per_layer + n_recurrent * state


class TestRotatingCacheConstantsMatchInstalledMlxLm:
    """Drift guard: our sliding-slot estimate is grounded in two ``RotatingKVCache``
    internals from the INSTALLED ``mlx_lm`` — the ``step`` block granularity and
    the ``keep`` sink count that ``make_prompt_cache`` reserves. If a future
    ``mlx_lm`` upgrade changes either, ``rotating_cache_slots`` would silently
    UNDER-reserve KV on the D-METAL-CAP admission path (an OOM risk — the
    load-bearing failure). These tests import the real module and FAIL LOUDLY on
    any change so the constant gets re-grounded rather than drifting unnoticed.

    Hermetic: imports an installed pure-Python module and constructs a rotating
    cache from a fake model (a ``SimpleNamespace`` with ``.layers`` and no
    ``make_cache``) — no weights, no network, no model load.
    """

    def test_step_matches_our_constant(self):
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        installed_step = cache_mod.RotatingKVCache.step
        assert installed_step == _ROTATING_CACHE_STEP, (
            "mlx_lm RotatingKVCache.step changed from the value kv_estimation "
            f"assumes ({_ROTATING_CACHE_STEP} -> installed {installed_step}); "
            "rotating_cache_slots would round to the wrong block size. If step "
            "GREW this under-reserves KV at admission (OOM risk). Re-ground "
            "_ROTATING_CACHE_STEP against the new impl."
        )

    def test_make_prompt_cache_keep_matches_our_bound(self):
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        # Fake model: exposes .layers, no ``make_cache`` -> make_prompt_cache takes
        # the RotatingKVCache branch when a max_kv_size (window) is supplied.
        fake_model = SimpleNamespace(layers=[object(), object()])
        assert not hasattr(fake_model, "make_cache")
        caches = cache_mod.make_prompt_cache(fake_model, max_kv_size=512)
        assert caches, "make_prompt_cache returned no layers"
        assert all(isinstance(c, cache_mod.RotatingKVCache) for c in caches), (
            "make_prompt_cache(max_kv_size=...) did not build RotatingKVCache layers"
        )
        installed_keeps = {c.keep for c in caches}
        assert installed_keeps == {_ROTATING_CACHE_KEEP}, (
            "mlx_lm make_prompt_cache builds rotating layers with keep="
            f"{installed_keeps}, but kv_estimation assumes keep="
            f"{_ROTATING_CACHE_KEEP}. If the installed keep is LARGER than our "
            "constant, rotating_cache_slots under-reserves KV at admission (OOM "
            "risk). Re-ground _ROTATING_CACHE_KEEP (it must be >= every keep any "
            "shipped construction uses)."
        )
        # Restate the safety invariant explicitly: our KEEP must remain an upper
        # bound over the installed construction's keep.
        assert max(installed_keeps) <= _ROTATING_CACHE_KEEP

    @staticmethod
    def _static_keep_upper_bound(node):
        """Static upper bound of a ``keep=`` expression, or ``None`` if unbounded.

        Recognizes exactly the two provably-safe forms the engine uses:

        * a constant int literal → its own value;
        * the re-wrap idiom ``getattr(<src>, "keep", <int literal>)`` → the literal
          default. Such a construction inherits ``<src>``'s keep, and every source
          ``RotatingKVCache`` in the tree is itself bounded by THIS scan, so the
          inherited value is bounded too; the fallback is the literal default.

        Any other expression (a config attribute, a variable, a call we don't
        recognize) is NOT statically bounded → ``None``, and the caller FAILS: a
        keep that can exceed the bound at runtime would silently under-reserve KV.
        """
        import ast

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return node.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "keep"
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, int)
            and not isinstance(node.args[2].value, bool)
        ):
            return node.args[2].value
        return None

    def test_no_shipped_rotating_cache_construction_exceeds_our_keep(self):
        # ``make_prompt_cache`` (tested above) is the GENERIC construction path,
        # but the engine ALSO builds ``RotatingKVCache`` DIRECTLY in vendored
        # stacks (Gemma-4, DeepSeek-V4). If any shipped construction set ``keep``
        # ABOVE our ``_ROTATING_CACHE_KEEP`` bound, ``rotating_cache_slots`` would
        # under-reserve KV at admission (OOM risk). Statically scan EVERY exact
        # ``RotatingKVCache(...)`` call in the package and, for each, prove its
        # ``keep`` is bounded — the "validate the max keep across all constructors"
        # guarantee, hermetic (AST parse, no imports run, no model load).
        #
        # This does NOT silently ignore non-literal keeps: a construction whose
        # ``keep`` cannot be statically bounded (anything but a literal int or the
        # ``getattr(src, "keep", <int>)`` re-wrap idiom) FAILS the test, forcing a
        # human to prove the new form safe or extend the recognizer. A missing
        # ``keep`` kwarg is the ``RotatingKVCache`` default (0), which is safe.
        # ``BatchRotatingKVCache`` is a distinct batch-decode re-pack (own
        # signature) and is intentionally excluded by the exact-name match.
        import ast
        import pathlib

        import rapid_mlx

        pkg_root = pathlib.Path(rapid_mlx.__file__).parent
        total_constructions = 0
        keeps_checked = 0
        for py in pkg_root.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                else:
                    continue
                if name != "RotatingKVCache":
                    continue
                total_constructions += 1
                keep_kw = next((kw for kw in node.keywords if kw.arg == "keep"), None)
                if keep_kw is None:
                    continue  # default keep=0 → safe
                bound = self._static_keep_upper_bound(keep_kw.value)
                assert bound is not None, (
                    f"{py.name}:{node.lineno} constructs RotatingKVCache with a keep= "
                    "expression this OOM-safety scan cannot statically bound "
                    f"({ast.dump(keep_kw.value)}). A keep that can exceed "
                    f"_ROTATING_CACHE_KEEP={_ROTATING_CACHE_KEEP} at runtime would make "
                    "rotating_cache_slots under-reserve KV at admission. Use a literal "
                    'keep <= the bound, the getattr(src, "keep", <int>) re-wrap idiom, '
                    "or extend _static_keep_upper_bound to prove the new form safe."
                )
                assert bound <= _ROTATING_CACHE_KEEP, (
                    f"{py.name}:{node.lineno} constructs RotatingKVCache(keep={bound}), "
                    f"which EXCEEDS _ROTATING_CACHE_KEEP={_ROTATING_CACHE_KEEP}; "
                    "rotating_cache_slots would under-reserve KV at admission (OOM "
                    "risk). Raise _ROTATING_CACHE_KEEP to bound every shipped keep."
                )
                keeps_checked += 1
        # The vendored Gemma-4 / DeepSeek-V4 stacks are in-tree, so the scan must
        # find real constructions AND validate at least one keep — a zero count
        # means the scan silently missed them (e.g. a rename) → false assurance.
        assert total_constructions > 0, (
            "no RotatingKVCache(...) constructions found in rapid_mlx — the keep "
            "drift scan matched nothing and cannot guard the bound"
        )
        assert keeps_checked > 0, (
            "no explicit RotatingKVCache(keep=...) constructions were validated"
        )
