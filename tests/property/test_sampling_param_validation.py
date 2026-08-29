# SPDX-License-Identifier: Apache-2.0
"""Property-based generalization of the H-10 sampling-param fix.

H-10 closed a silent-burn class: a NaN / out-of-range ``temperature`` or
``top_p`` HTTP-200'd straight into a Metal kernel and crashed the server.
The regression tests pin single example values; these properties pin the
invariant *over the entire float space*:

* every non-finite float is rejected,
* every finite value strictly outside the documented range is rejected,
* every finite value inside the range is accepted AND stored unchanged.

Each request model documents its own range, so we assert only what that
model actually enforces:

* OpenAI ``ChatCompletionRequest`` / ``CompletionRequest``:
  ``temperature`` in ``[0, 2]``, ``top_p`` in ``(0, 1]``.
* Anthropic ``AnthropicRequest``: ``temperature`` in ``[0, 1]``,
  ``top_p`` in ``(0, 1]``.

Pure Pydantic construction — no server, fully hermetic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")


import math

from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from vllm_mlx.api.anthropic_models import AnthropicRequest
from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest

from .strategies import (
    in_range_floats,
    nonfinite_floats,
    out_of_range_finite_floats,
)

pytestmark = [pytest.mark.property, pytest.mark.requires_mlx]


# --- minimal valid constructors -----------------------------------------
# Each builder supplies only the required fields for that model so the
# sampling param under test is the sole variable.


def _build_chat(**kw) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **kw)


def _build_completion(**kw) -> CompletionRequest:
    return CompletionRequest(prompt="hi", **kw)


def _build_anthropic(**kw) -> AnthropicRequest:
    return AnthropicRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        **kw,
    )


# Plain spec data: (label, builder, {field: (lo, hi, lo_inclusive, hi_inclusive)}).
# Ranges are read straight off the Field(...) bounds + field validators in
# api/models.py and api/anthropic_models.py — assert only what the code
# guarantees.
_SPECS = [
    (
        "chat",
        _build_chat,
        {"temperature": (0.0, 2.0, True, True), "top_p": (0.0, 1.0, False, True)},
    ),
    (
        "completion",
        _build_completion,
        {"temperature": (0.0, 2.0, True, True), "top_p": (0.0, 1.0, False, True)},
    ),
    (
        "anthropic",
        _build_anthropic,
        {"temperature": (0.0, 1.0, True, True), "top_p": (0.0, 1.0, False, True)},
    ),
]

# Per-model params (the whole field dict) for the explicit endpoint sweep.
_MODEL_SPECS = [
    pytest.param(build, fields, id=label) for label, build, fields in _SPECS
]

# Per-(model, field) params so Hypothesis is REQUIRED to exercise EVERY
# field/model combination. Drawing the field inside the property (the old
# ``sampled_from``) left coverage probabilistic: a validator broken on a
# field that happened not to be drawn could stay green. Parametrizing the
# field makes each combination its own guaranteed test case.
_FIELD_SPECS = [
    pytest.param(build, field, bounds, id=f"{label}-{field}")
    for label, build, fields in _SPECS
    for field, bounds in sorted(fields.items())
]


@pytest.mark.parametrize("build,field,bounds", _FIELD_SPECS)
@given(data=st.data())
def test_nonfinite_always_rejected(build, field, bounds, data):
    """Constructing any of these models with a non-finite ``temperature``
    or ``top_p`` raises ``ValidationError`` (never a silent 200 → kernel).
    """
    bad = data.draw(nonfinite_floats())
    with pytest.raises(ValidationError):
        build(**{field: bad})


@pytest.mark.parametrize("build,field,bounds", _FIELD_SPECS)
@given(data=st.data())
def test_out_of_range_finite_rejected(build, field, bounds, data):
    """Finite values outside the documented range raise ``ValidationError``.

    Bound-inclusivity aware: for an exclusive bound (e.g. ``top_p``'s
    ``gt=0.0``) the endpoint itself is invalid and is drawn too, so a
    regression that started accepting ``top_p == 0.0`` fails here.
    """
    lo, hi, lo_incl, hi_incl = bounds
    bad = data.draw(
        out_of_range_finite_floats(lo, hi, lo_inclusive=lo_incl, hi_inclusive=hi_incl)
    )
    # Guard the strategy's own contract: the drawn value really is invalid
    # for this range + inclusivity.
    assert math.isfinite(bad)
    is_invalid = (
        bad < lo
        or bad > hi
        or (bad == lo and not lo_incl)
        or (bad == hi and not hi_incl)
    )
    assert is_invalid, f"strategy produced an in-range value {bad!r}"
    with pytest.raises(ValidationError):
        build(**{field: bad})


@pytest.mark.parametrize("build,fields", _MODEL_SPECS)
def test_excluded_endpoints_rejected(build, fields):
    """Every EXCLUSIVE range endpoint is rejected — explicitly, not just
    via fuzzing. In practice this pins ``top_p == 0.0`` (range ``(0, 1]``)
    on every model that enforces it, so a future loosening to ``ge=0.0``
    can't slip through green.
    """
    checked_any = False
    for field, (lo, hi, lo_incl, hi_incl) in fields.items():
        if not lo_incl:
            checked_any = True
            with pytest.raises(ValidationError):
                build(**{field: float(lo)})
        if not hi_incl:
            checked_any = True
            with pytest.raises(ValidationError):
                build(**{field: float(hi)})
    # Sanity: the spec table must contain at least one exclusive bound
    # (top_p), otherwise this test would silently assert nothing.
    assert checked_any


@pytest.mark.parametrize("build,field,bounds", _FIELD_SPECS)
@given(data=st.data())
def test_in_range_accepted_and_preserved(build, field, bounds, data):
    """Finite values inside the range construct OK and are stored under
    numeric equality — no silent clamping or coercion.

    Scope note: signed-zero normalization (``-0.0`` stored as ``0.0``) is
    deliberately out of scope — the two are numerically equal and
    behaviourally inert for sampling, so this asserts ``==`` (numeric),
    not IEEE-754 byte identity.
    """
    lo, hi, lo_incl, hi_incl = bounds
    value = data.draw(
        in_range_floats(lo, hi, lo_inclusive=lo_incl, hi_inclusive=hi_incl)
    )
    req = build(**{field: value})
    assert getattr(req, field) == value
