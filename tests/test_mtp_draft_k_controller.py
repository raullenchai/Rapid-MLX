# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the universal MTP draft-``k`` auto-tune controller (0.9.11 PR-1).

The controller lives in
:mod:`vllm_mlx.spec_decode.mtp.draft_k_controller` and is intentionally
pure Python (no MLX imports). These tests exercise its state machine
without spawning a server or importing MLX — they can run under any
Python 3.10+ interpreter.

Coverage:

* Ramp-up: feeding all-accepts must ramp ``k`` up to ``k_max``, and
  each adjustment must respect the ``cooldown`` gate.
* Ramp-down: feeding all-rejects must ramp ``k`` down to ``k_min``.
* Hold: an accept rate strictly between the two thresholds must
  produce zero adjustments no matter how many attempts are recorded.
* Constructor validation: NaN, ``+inf``, ``-inf`` on either
  threshold must raise; inverted thresholds must raise; degenerate
  ``k_min == k_max == 1`` must construct without error and never
  change ``k``.
* Regression guard for the generator: with auto-tune off (the
  default), the vendored ``mtp_generate_step`` must emit exactly the
  same tokens it did before PR-1 landed. We exercise this via the
  same all-accept / all-reject scripts the lossless test uses.
"""

from __future__ import annotations

import math

import pytest

from vllm_mlx.spec_decode.mtp.draft_k_controller import (
    DraftKController,
    clear_global_controller,
    get_global_controller,
    install_global_controller,
)


# ---------------------------------------------------------------------------
# Fixture — isolate the module-level singleton between cases.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_controller_singleton():
    """Ensure ``get_global_controller()`` returns ``None`` at case start.

    Tests that flip on the global for integration coverage must not
    leak into subsequent cases.
    """
    clear_global_controller()
    yield
    clear_global_controller()


# ---------------------------------------------------------------------------
# 1. Ramp-up: all-accepts should bump k to k_max, respecting cooldown.
# ---------------------------------------------------------------------------


def test_all_accepts_ramps_up_to_k_max_respecting_cooldown():
    """All-accept feed with cooldown=W (fully populated window) should
    produce exactly ``k_max - k_start`` adjustments spaced ``cooldown``
    attempts apart.

    With W=cooldown=8, k_start=1, k_max=4, and every attempt an
    accept:

    * attempts 1..7: window fills, no adjustment (window not full yet).
    * attempt 8: window fills (rate=1.0), first evaluation fires,
      k=2. Steps_since_eval resets to 0.
    * attempts 9..15: cooldown counting up, no re-eval.
    * attempt 16: cooldown elapsed, rate=1.0 → k=3.
    * attempt 24: rate=1.0 → k=4 (=k_max, capped).
    * attempt 32: rate=1.0, k is at k_max already, no change.

    So the adjustment schedule is deterministic: (attempt 8: 1→2),
    (attempt 16: 2→3), (attempt 24: 3→4), then no further changes.
    """
    ctrl = DraftKController(
        k_min=1,
        k_max=4,
        k_start=1,
        window=8,
        upshift_threshold=0.80,
        downshift_threshold=0.40,
        cooldown=8,
    )

    # Attempts 1..7: window is filling; no eval fires.
    for _ in range(7):
        ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 1
    assert ctrl.snapshot()["adjust_count"] == 0

    # Attempt 8: first eval, k should bump 1 → 2.
    ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 2
    assert ctrl.snapshot()["adjust_count"] == 1

    # Attempts 9..15: cooldown counter re-populating.
    for _ in range(7):
        ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 2

    # Attempt 16: second eval, k should bump 2 → 3.
    ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 3

    # Attempt 24: third eval, k should bump 3 → 4.
    for _ in range(8):
        ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 4
    assert ctrl.snapshot()["adjust_count"] == 3

    # Attempt 32+: k is at k_max, no further changes even with 100
    # more all-accept attempts.
    for _ in range(100):
        ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 4
    assert ctrl.snapshot()["adjust_count"] == 3


# ---------------------------------------------------------------------------
# 2. Ramp-down: all-rejects should bump k to k_min.
# ---------------------------------------------------------------------------


def test_all_rejects_ramps_down_to_k_min():
    """All-reject feed with k_start at k_max should ratchet down to
    k_min. Same deterministic cadence as the all-accept case.
    """
    ctrl = DraftKController(
        k_min=1,
        k_max=4,
        k_start=4,
        window=8,
        upshift_threshold=0.80,
        downshift_threshold=0.40,
        cooldown=8,
    )

    for _ in range(8):
        ctrl.record_attempt(accepted=False)
    assert ctrl.current_k() == 3

    for _ in range(8):
        ctrl.record_attempt(accepted=False)
    assert ctrl.current_k() == 2

    for _ in range(8):
        ctrl.record_attempt(accepted=False)
    assert ctrl.current_k() == 1

    # Floor reached: further rejects must NOT go below k_min.
    for _ in range(100):
        ctrl.record_attempt(accepted=False)
    assert ctrl.current_k() == 1


# ---------------------------------------------------------------------------
# 3. In-band rate should hold (no oscillation).
# ---------------------------------------------------------------------------


def test_in_band_accept_rate_holds():
    """Alternating accept/reject → rate=0.50 which is in the (0.40,
    0.80) hold band. No adjustments should fire, no matter how many
    attempts we record.
    """
    ctrl = DraftKController(
        k_min=1,
        k_max=4,
        k_start=2,
        window=8,
        upshift_threshold=0.80,
        downshift_threshold=0.40,
        cooldown=8,
    )

    for i in range(1000):
        ctrl.record_attempt(accepted=(i % 2 == 0))
    assert ctrl.current_k() == 2
    assert ctrl.snapshot()["adjust_count"] == 0


def test_no_oscillation_within_same_window():
    """A single window flip from mostly-accepts to mostly-rejects
    must NOT produce an up-then-down oscillation within the SAME
    cooldown period.

    The cooldown gate is the invariant that guarantees this — the
    controller cannot re-evaluate until at least ``cooldown`` new
    attempts have been recorded. As long as ``cooldown`` is set
    (>= 1), the invariant holds by construction.
    """
    ctrl = DraftKController(
        k_min=1,
        k_max=4,
        k_start=2,
        window=8,
        upshift_threshold=0.80,
        downshift_threshold=0.40,
        cooldown=8,
    )

    # Fill the window with 100% accepts. This should adjust ONCE
    # (k 2 → 3) at the 8th attempt.
    for _ in range(8):
        ctrl.record_attempt(accepted=True)
    assert ctrl.current_k() == 3

    # Immediately flip to rejects. Cooldown says we can't re-eval
    # until 8 more attempts. Rejects during the cooldown should not
    # trigger a downshift even though the accept-rate over the
    # window is dropping.
    for _ in range(7):
        ctrl.record_attempt(accepted=False)
    assert ctrl.current_k() == 3, (
        "Downshift must not fire within the cooldown window "
        "even when the rate has dropped."
    )


# ---------------------------------------------------------------------------
# 4. Constructor validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_ctor_rejects_nan_and_infinity_on_upshift(bad):
    """NaN and infinities on upshift_threshold must raise. Pydantic
    ``Field(ge=, le=)`` accepts NaN silently (see the memory gotcha
    ``pydantic-field-does-not-reject-nan``), so we validate here.
    """
    with pytest.raises(ValueError, match="upshift_threshold"):
        DraftKController(upshift_threshold=bad, downshift_threshold=0.40)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_ctor_rejects_nan_and_infinity_on_downshift(bad):
    """Same gate for downshift_threshold."""
    with pytest.raises(ValueError, match="downshift_threshold"):
        DraftKController(upshift_threshold=0.80, downshift_threshold=bad)


def test_ctor_rejects_upshift_lteq_downshift():
    """Inverted thresholds (upshift <= downshift) must raise. A
    controller with them inverted would ratchet ``k`` down on high
    accept rates — a silent inversion that would be very hard to
    diagnose in production.
    """
    with pytest.raises(ValueError, match="strictly greater"):
        DraftKController(upshift_threshold=0.40, downshift_threshold=0.80)
    with pytest.raises(ValueError, match="strictly greater"):
        DraftKController(upshift_threshold=0.50, downshift_threshold=0.50)


def test_ctor_rejects_thresholds_out_of_range():
    """Thresholds must be in (0, 1] and [0, 1) respectively."""
    with pytest.raises(ValueError, match="upshift_threshold"):
        DraftKController(upshift_threshold=0.0, downshift_threshold=0.0)
    with pytest.raises(ValueError, match="upshift_threshold"):
        DraftKController(upshift_threshold=1.5, downshift_threshold=0.40)
    with pytest.raises(ValueError, match="downshift_threshold"):
        DraftKController(upshift_threshold=0.80, downshift_threshold=-0.1)
    with pytest.raises(ValueError, match="downshift_threshold"):
        DraftKController(upshift_threshold=0.80, downshift_threshold=1.0)


def test_ctor_rejects_bad_integer_bounds():
    """k_min, k_max, window, cooldown must be positive ints."""
    with pytest.raises(ValueError, match="k_min"):
        DraftKController(k_min=0, k_max=4)
    with pytest.raises(ValueError, match="k_max"):
        DraftKController(k_min=3, k_max=2)
    with pytest.raises(ValueError, match="window"):
        DraftKController(window=0)
    with pytest.raises(ValueError, match="cooldown"):
        DraftKController(cooldown=0)


def test_ctor_rejects_bool_where_int_expected():
    """``bool`` is a subclass of ``int`` in Python, but a ``True``/
    ``False`` mistake in a caller should fail loud rather than
    silently mis-configuring the controller.
    """
    with pytest.raises(TypeError, match="k_min"):
        DraftKController(k_min=True, k_max=4)


def test_ctor_rejects_start_out_of_bounds():
    """k_start must satisfy k_min <= start <= k_max."""
    with pytest.raises(ValueError, match="k_start"):
        DraftKController(k_min=2, k_max=4, k_start=1)
    with pytest.raises(ValueError, match="k_start"):
        DraftKController(k_min=2, k_max=4, k_start=5)


# ---------------------------------------------------------------------------
# 5. Degenerate k_min == k_max == 1 must not crash and must never change k.
# ---------------------------------------------------------------------------


def test_degenerate_bounds_never_change_k():
    """k_min == k_max == 1 → controller is a no-op ``k``-wise but
    still records attempts and updates its window (so /metrics keeps
    getting fresh accept-rate numbers).
    """
    ctrl = DraftKController(
        k_min=1,
        k_max=1,
        window=8,
        upshift_threshold=0.80,
        downshift_threshold=0.40,
        cooldown=8,
    )

    for i in range(1000):
        ctrl.record_attempt(accepted=(i % 3 != 0))  # ~66% accepts
    assert ctrl.current_k() == 1
    # Adjustments would be attempted at cooldown boundaries but
    # capped by the bounds — the adjust counter must stay at 0.
    assert ctrl.snapshot()["adjust_count"] == 0


# ---------------------------------------------------------------------------
# 6. Snapshot shape + module-global installer contract.
# ---------------------------------------------------------------------------


def test_snapshot_returns_expected_keys():
    ctrl = DraftKController(k_min=1, k_max=4, window=4, cooldown=4)
    snap = ctrl.snapshot()
    expected_keys = {
        "current_k",
        "k_min",
        "k_max",
        "window_size",
        "window_fill",
        "window_accept_rate",
        "cooldown_steps",
        "steps_since_eval",
        "adjust_count",
        "upshift_threshold",
        "downshift_threshold",
    }
    assert set(snap) == expected_keys
    assert snap["current_k"] == 1
    assert snap["window_fill"] == 0
    assert snap["window_accept_rate"] == 0.0
    assert math.isfinite(snap["upshift_threshold"])
    assert math.isfinite(snap["downshift_threshold"])


def test_global_controller_starts_uninstalled():
    """The module-global controller must default to ``None`` so the
    static-``k`` fast path in generator.py stays engaged when the
    operator hasn't opted in.
    """
    assert get_global_controller() is None


def test_install_and_clear_global_controller_roundtrip():
    ctrl = DraftKController(k_min=1, k_max=2, window=2, cooldown=2)
    install_global_controller(ctrl)
    assert get_global_controller() is ctrl
    clear_global_controller()
    assert get_global_controller() is None


def test_install_rejects_non_controller():
    with pytest.raises(TypeError, match="DraftKController"):
        install_global_controller("not a controller")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. Regression guard for generator.py — auto-tune off must be a no-op.
# ---------------------------------------------------------------------------


def test_generator_static_k_path_unchanged_without_controller():
    """When ``draft_k_controller`` is not installed globally AND not
    passed explicitly, the vendored ``mtp_generate_step`` must emit
    the same tokens it did before PR-1 landed.

    We exercise this by reusing the mocked-model plumbing from
    :mod:`tests.test_mtp_spec_decode` and asserting the emitted
    token sequence in the all-accept scenario byte-matches the
    reference. If the ``if draft_k_controller is not None`` guard
    in generator.py ever regressed to a call on a None controller,
    an AttributeError would surface here.
    """
    mx = pytest.importorskip("mlx.core")
    from tests.test_mtp_spec_decode import _MockedQwen35Model  # noqa: E402
    from vllm_mlx.spec_decode.mtp.accept_counter import MTPAcceptCounter
    from vllm_mlx.spec_decode.mtp.cache_patch import _unpatch_for_tests
    from vllm_mlx.spec_decode.mtp.generator import mtp_generate_step

    # Ensure the ArraysCache patch is a clean install for this
    # case (matches the test_mtp_lossless fixture behaviour).
    _unpatch_for_tests()

    # Re-bind mlx_lm.generate.generation_stream to this thread's
    # default stream (see test_mtp_spec_decode fixture docstring).
    import sys

    import mlx_lm.generate  # noqa: F401

    sys.modules["mlx_lm.generate"].generation_stream = mx.default_stream(
        mx.default_device()
    )

    # Global controller must be uninstalled (autouse fixture guarantees
    # this) — sanity-check the invariant.
    assert get_global_controller() is None

    # Same all-accept script as test_mtp_lossless: primary=7, three
    # accept pairs.
    backbone_script = [7, 11, 13, 15, 17, 19, 21]
    mtp_script = [11, 0, 15, 0, 19]
    model = _MockedQwen35Model(backbone_script, mtp_script)
    counter = MTPAcceptCounter()

    tokens = [
        tok
        for tok, _lp, _fd in mtp_generate_step(
            mx.array([1], mx.uint32),
            model,
            max_tokens=7,
            accept_counter=counter,
        )
    ]

    # The regression guard: static-k output matches the reference
    # sequence. Any change to generator.py that broke the
    # controller-off path would flip this.
    assert tokens == [7, 11, 13, 15, 17, 19, 21]
    _unpatch_for_tests()
