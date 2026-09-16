# SPDX-License-Identifier: Apache-2.0
"""Shared Hypothesis strategies for the hermetic property-based suite.

Every strategy here builds a *small in-memory value* — no model load, no
booted server. That is a hard constraint, not an accident: property
fuzzing multiplies the work by ``max_examples``, so it can only stay fast
enough to run on every commit if each example is cheap. Model-requiring
metamorphic properties (streaming == non-streaming, temp=0 determinism)
are deliberately out of scope and belong in the integration suite.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from hypothesis import strategies as st

# The live quantized KV cache only ever uses these (group_size, bits)
# pairs — see ``rapid_mlx/quantized_batch_cache.py``. ``mx.quantize``
# requires the quantized (last) dim to be an exact multiple of group_size.
QUANT_BITS: tuple[int, ...] = (4, 8)
QUANT_GROUP_SIZES: tuple[int, ...] = (32, 64, 128)

# Per-tensor magnitude scales so the round-trip invariants are exercised
# across both tiny (~1e-2) and large (~5e1) activations. The affine
# quantization step scales with the data range, so any correct error
# bound has to be magnitude-invariant — hence the spread.
_MAGNITUDE_SCALES: tuple[float, ...] = (1e-2, 1.0, 7.0, 5e1)


@st.composite
def mlx_kv_tensors(draw, *, max_rows: int = 6, max_groups: int = 4):
    """Draw ``(x, group_size, bits)`` for the KV round-trip invariants.

    ``x`` is a finite ``mx.array`` of shape ``(rows, head_dim)`` where
    ``head_dim = group_size * n_groups`` — the exact-divisibility
    ``mx.quantize`` needs along its last (head) axis. Values span
    negative + positive and small + large magnitudes; NaN/inf are out of
    scope (they never reach the KV cache — attention/SDPA would already
    have produced NaN upstream). ``bits`` covers {4, 8} and
    ``group_size`` covers {32, 64, 128}, always with a divisible head_dim.

    Values are generated from a Hypothesis-drawn seed via NumPy rather
    than element-by-element: shrinking a 3072-float list adds no signal
    for these whole-tensor numeric invariants, and the seed keeps every
    failing example perfectly reproducible while staying fast.

    ``dist`` covers, per group of ``group_size`` consecutive elements:

    * ``normal`` / ``uniform`` — general spread,
    * ``bimodal`` — mass at the group extrema (worst case for affine
      min/max quantization),
    * ``constant`` — every element in a group equal, at a nonzero
      per-group offset (zero quantization step — the divide-by-step
      edge),
    * ``narrow`` — a tiny variation around a large nonzero per-group
      offset (the affine-precision / catastrophic-cancellation edge).
    """
    bits = draw(st.sampled_from(QUANT_BITS))
    group_size = draw(st.sampled_from(QUANT_GROUP_SIZES))
    n_groups = draw(st.integers(min_value=1, max_value=max_groups))
    rows = draw(st.integers(min_value=1, max_value=max_rows))
    scale = draw(st.sampled_from(_MAGNITUDE_SCALES))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    dist = draw(st.sampled_from(("normal", "uniform", "bimodal", "constant", "narrow")))

    head_dim = group_size * n_groups
    rng = np.random.default_rng(seed)
    if dist == "normal":
        base = rng.standard_normal((rows, head_dim))
    elif dist == "uniform":
        base = rng.uniform(-1.0, 1.0, size=(rows, head_dim))
    elif dist == "bimodal":  # mass at the group extrema — worst case for
        # affine min/max quantization.
        base = rng.choice((-1.0, 1.0), size=(rows, head_dim))
        base = base + 0.05 * rng.standard_normal((rows, head_dim))
    elif dist == "constant":  # every element in a group identical, at a
        # nonzero per-group offset -> (max - min) == 0, i.e. step == 0.
        offsets = rng.standard_normal((rows, n_groups, 1))
        base = np.repeat(offsets, group_size, axis=2).reshape(rows, head_dim)
    else:  # "narrow" — a tiny variation around a LARGE per-group offset,
        # stressing affine precision (the reconstruction floor is set by
        # the offset magnitude, not the group range).
        offsets = rng.standard_normal((rows, n_groups, 1)) * 10.0
        jitter = 1e-3 * rng.standard_normal((rows, n_groups, group_size))
        base = (np.repeat(offsets, group_size, axis=2) + jitter).reshape(rows, head_dim)
    x = mx.array((base * scale).astype(np.float32))
    return x, group_size, bits


# ---- sampling-parameter float strategies -------------------------------


def nonfinite_floats() -> st.SearchStrategy:
    """NaN, +inf, -inf — the non-finite forms every sampling-param
    validator must reject (the H-10 fix)."""
    return st.sampled_from((float("nan"), float("inf"), float("-inf")))


def in_range_floats(
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> st.SearchStrategy:
    """Finite floats inside ``[lo, hi]``, honoring per-bound inclusivity."""
    return st.floats(
        min_value=lo,
        max_value=hi,
        exclude_min=not lo_inclusive,
        exclude_max=not hi_inclusive,
        allow_nan=False,
        allow_infinity=False,
    )


def out_of_range_finite_floats(
    lo: float,
    hi: float,
    *,
    lo_inclusive: bool = True,
    hi_inclusive: bool = True,
) -> st.SearchStrategy:
    """Finite floats that are INVALID for the range with the given
    inclusivity — the values a correct validator must reject.

    Spans the ENTIRE finite range on each side (no artificial outer cap):
    strictly ``< lo`` reaches down to ``-1.8e308`` and strictly ``> hi``
    up to ``+1.8e308``, so large magnitudes like ``1e308`` are exercised —
    a validator that only bounded, say, ``|x| < 1e6`` would be caught.

    Bound-inclusivity aware (the Fix-1 gap): when a bound is *exclusive*
    the endpoint itself is invalid, so it is included in the generated
    set. For ``top_p``'s ``(0, 1]`` this makes ``0.0`` reachable — without
    it a regression that started accepting ``top_p == 0.0`` would stay
    green.

    * always: strictly below ``lo`` and strictly above ``hi`` (full finite
      range),
    * when ``lo`` is exclusive: the exact endpoint ``lo`` (e.g. ``0.0``),
    * when ``hi`` is exclusive: the exact endpoint ``hi``.

    Hypothesis realizes ``exclude_min`` / ``exclude_max`` with
    ``math.nextafter``, so every strictly-outside value is a representable
    float *distinct* from the bound — no float-rounding value can silently
    land ON an accepted boundary and turn a rejection property flaky.
    """
    below = st.floats(
        max_value=lo,
        exclude_max=True,
        allow_nan=False,
        allow_infinity=False,
    )
    above = st.floats(
        min_value=hi,
        exclude_min=True,
        allow_nan=False,
        allow_infinity=False,
    )
    branches = [below, above]
    # An EXCLUSIVE bound means the endpoint value is itself invalid and
    # must appear in the invalid set.
    if not lo_inclusive:
        branches.append(st.just(float(lo)))
    if not hi_inclusive:
        branches.append(st.just(float(hi)))
    return st.one_of(branches)
