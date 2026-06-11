# SPDX-License-Identifier: Apache-2.0
"""Schema tests for AliasProfile diffusion knobs.

Three knobs added with the in-house diffusion loop (see
``runtime/diffusion_loop.py``):

- ``diffusion_backend``: ``"rapid"`` (default, in-house) | ``"mlx-vlm"``
  (escape hatch). Routing decision — typo silently swaps generation
  loops, so the loader must reject unknown values.
- ``diffusion_fixed_steps`` (default ``8``): number of denoising
  passes per canvas in the rapid backend. ``8`` is the empirical
  optimum on DiffusionGemma 26B-A4B 4-bit — 1.76x speedup on long-
  form vs mlx-vlm at byte-comparable quality (valid JSON, working
  code, coherent prose). Setting ``null`` in JSON opts into the
  ``_stable_and_confident`` adaptive-stop predicate for bit-for-bit
  step-count parity with upstream (slightly slower, useful for
  spec-decode draft-budget accounting). Any int value must be a
  positive JSON int — silently truncating ``8.5`` would hide a
  hand-edit typo.
- ``diffusion_sc_every`` (default 1): self-conditioning matmul cadence.
  Empirical: only ``1`` keeps quality on DiffusionGemma 26B-A4B 4-bit;
  ``>=2`` collapses output to ``"the the the…"`` (see quality eval
  ``research/diffusion-gemma/quality/eval-20260611-1001.md``). Kept as
  a knob for future diffusion families.

These tests pin the contract so a future schema edit can't silently
flip routing.
"""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import (
    AliasProfile,
    _coerce,
    list_profiles,
    resolve_profile,
)


def test_defaults_match_diffusion_gemma_empirical_optimum():
    """Code-level defaults must match what we found is optimal for the
    only shipped diffusion alias today. If a future change moves the
    default away from these, the diff is review-visible."""
    prof = AliasProfile(hf_path="x/y")
    assert prof.diffusion_backend == "rapid"
    assert prof.diffusion_fixed_steps == 8  # empirical optimum
    assert prof.diffusion_sc_every == 1


def test_text_alias_inherits_defaults_silently():
    """An existing text-modality alias (no diffusion_* fields in JSON)
    must keep loading — the new fields default to no-ops for AR."""
    qwen = resolve_profile("qwen3.5-4b-4bit")
    assert qwen is not None
    assert qwen.modality == "text"
    assert qwen.diffusion_backend == "rapid"
    assert qwen.diffusion_fixed_steps == 8
    assert qwen.diffusion_sc_every == 1


def test_diffusion_gemma_alias_picks_up_defaults():
    """The shipped DiffusionGemma alias today does not set the fields
    explicitly — they must come from the AliasProfile defaults."""
    diff = resolve_profile("diffusion-gemma-26b")
    assert diff is not None
    assert diff.modality == "text-diffusion"
    assert diff.diffusion_backend == "rapid"
    assert diff.diffusion_fixed_steps == 8  # empirical optimum
    assert diff.diffusion_sc_every == 1


def test_fixed_steps_explicit_null_opts_into_adaptive_mode():
    """Operators can opt into the vendored ``_stable_and_confident``
    adaptive-stop predicate by setting ``"diffusion_fixed_steps": null``
    in the alias JSON. The loader must propagate ``None`` to the
    AliasProfile (which the lane translates into "omit the kwarg" so
    the rapid loop's signature default enables adaptive)."""
    prof = _coerce(
        "x",
        {
            "hf_path": "a/b",
            "modality": "text-diffusion",
            "supports_spec_decode": False,
            "diffusion_fixed_steps": None,
        },
    )
    assert prof.diffusion_fixed_steps is None


def test_fixed_steps_explicit_int_overrides_default():
    """Explicit ints override the ``8`` default. ``16`` is plausible
    if a future diffusion model converges slower than DiffusionGemma."""
    prof = _coerce(
        "x",
        {
            "hf_path": "a/b",
            "modality": "text-diffusion",
            "supports_spec_decode": False,
            "diffusion_fixed_steps": 16,
        },
    )
    assert prof.diffusion_fixed_steps == 16


# =============================================================================
# Validation — typos must fail loud at load
# =============================================================================


def test_backend_typo_fails_loud():
    """Routing decision — typo silently swaps loops. Fail at load."""
    with pytest.raises(ValueError, match="diffusion_backend must be one of"):
        _coerce(
            "x",
            {
                "hf_path": "a/b",
                "modality": "text-diffusion",
                "supports_spec_decode": False,
                "diffusion_backend": "rappid",
            },
        )


def test_fixed_steps_must_be_positive_int():
    with pytest.raises(
        ValueError, match="diffusion_fixed_steps must be a JSON integer"
    ):
        _coerce(
            "x",
            {
                "hf_path": "a/b",
                "modality": "text-diffusion",
                "supports_spec_decode": False,
                "diffusion_fixed_steps": 8.5,
            },
        )
    with pytest.raises(ValueError, match="diffusion_fixed_steps must be >= 1"):
        _coerce(
            "x",
            {
                "hf_path": "a/b",
                "modality": "text-diffusion",
                "supports_spec_decode": False,
                "diffusion_fixed_steps": 0,
            },
        )


def test_sc_every_must_be_positive_int():
    with pytest.raises(ValueError, match="diffusion_sc_every must be a JSON integer"):
        _coerce(
            "x",
            {
                "hf_path": "a/b",
                "modality": "text-diffusion",
                "supports_spec_decode": False,
                "diffusion_sc_every": True,  # bool subclass of int — rejected
            },
        )
    with pytest.raises(ValueError, match="diffusion_sc_every must be >= 1"):
        _coerce(
            "x",
            {
                "hf_path": "a/b",
                "modality": "text-diffusion",
                "supports_spec_decode": False,
                "diffusion_sc_every": -1,
            },
        )


# =============================================================================
# Modality gate — diffusion_* on a text alias is a configuration error
# =============================================================================


def test_diffusion_field_on_text_alias_fails_loud():
    """Setting any diffusion knob on a text-modality alias is a typo
    (the AR runtime ignores it). Mirror the DFlash/spec-decode gate
    pattern — fail at load, not at request time."""
    for field, val in [
        ("diffusion_backend", "rapid"),
        ("diffusion_fixed_steps", 8),
        ("diffusion_sc_every", 1),
    ]:
        with pytest.raises(ValueError, match=f"{field} is only meaningful when"):
            _coerce(
                "x",
                {
                    "hf_path": "a/b",
                    "modality": "text",
                    **{field: val},
                },
            )


def test_diffusion_field_accepted_on_text_diffusion_alias():
    """Same fields, modality=text-diffusion → must accept."""
    prof = _coerce(
        "x",
        {
            "hf_path": "a/b",
            "modality": "text-diffusion",
            "supports_spec_decode": False,
            "diffusion_backend": "mlx-vlm",
            "diffusion_fixed_steps": 16,
            "diffusion_sc_every": 1,
        },
    )
    assert prof.diffusion_backend == "mlx-vlm"
    assert prof.diffusion_fixed_steps == 16
    assert prof.diffusion_sc_every == 1


# =============================================================================
# Closed-key schema — unknown key under diffusion_* still rejected
# =============================================================================


def test_unknown_diffusion_key_still_rejected():
    """The closed-key gate at _ALLOWED_PROFILE_KEYS protects against a
    contributor typoing a NEW field. Verify a plausible typo is caught."""
    with pytest.raises(ValueError, match="unknown key"):
        _coerce(
            "x",
            {
                "hf_path": "a/b",
                "modality": "text-diffusion",
                "supports_spec_decode": False,
                "diffusion_step_count": 8,  # typo — should be diffusion_fixed_steps
            },
        )


# =============================================================================
# Every alias still loads — drift sentinel
# =============================================================================


def test_all_existing_aliases_still_load():
    """A schema edit that crashes existing alias loads is the worst
    regression class. Pin it here."""
    profs = list_profiles()
    assert len(profs) >= 50  # rough sanity bound; real count is 73 at 2026-06-11
    for name, prof in profs.items():
        assert prof.diffusion_backend in ("rapid", "mlx-vlm"), name
        # fixed_steps is either None (adaptive) or a positive int.
        if prof.diffusion_fixed_steps is not None:
            assert prof.diffusion_fixed_steps >= 1, name
        assert prof.diffusion_sc_every >= 1, name
