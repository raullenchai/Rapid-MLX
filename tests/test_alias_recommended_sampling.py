# SPDX-License-Identifier: Apache-2.0
"""Tests for ``AliasProfile.recommended_sampling`` coercion."""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import AliasProfile, _coerce


def test_none_default():
    profile = _coerce("fake", {"hf_path": "fake/Model"})
    assert profile.recommended_sampling is None


def test_dict_form_coerced_to_sorted_tuple():
    profile = _coerce(
        "fake",
        {
            "hf_path": "fake/Model",
            "recommended_sampling": {
                "top_p": 0.95,
                "temperature": 0.6,
                "top_k": 20,
            },
        },
    )
    # Order is canonicalized to sorted keys so identical configs hash
    # the same regardless of JSON insertion order.
    assert profile.recommended_sampling == (
        ("temperature", 0.6),
        ("top_k", 20.0),
        ("top_p", 0.95),
    )


def test_all_supported_keys_accepted():
    profile = _coerce(
        "fake",
        {
            "hf_path": "fake/Model",
            "recommended_sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "min_p": 0.05,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.3,
            },
        },
    )
    assert profile.recommended_sampling is not None
    assert len(profile.recommended_sampling) == 7


def test_empty_dict_collapses_to_none():
    """Empty dict in JSON shouldn't leave the alias half-populated."""
    profile = _coerce(
        "fake",
        {"hf_path": "fake/Model", "recommended_sampling": {}},
    )
    assert profile.recommended_sampling is None


def test_rejects_unknown_key():
    with pytest.raises(ValueError, match="unsupported key"):
        _coerce(
            "fake",
            {
                "hf_path": "fake/Model",
                "recommended_sampling": {"typical_p": 0.9},
            },
        )


def test_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="must be a number"):
        _coerce(
            "fake",
            {
                "hf_path": "fake/Model",
                "recommended_sampling": {"temperature": "0.7"},
            },
        )


def test_rejects_bool_value():
    """JSON ``true`` is also ``int`` in Python; must not slip through."""
    with pytest.raises(ValueError, match="must be a number"):
        _coerce(
            "fake",
            {
                "hf_path": "fake/Model",
                "recommended_sampling": {"temperature": True},
            },
        )


def test_rejects_non_dict_payload():
    with pytest.raises(ValueError, match="must be an object"):
        _coerce(
            "fake",
            {
                "hf_path": "fake/Model",
                "recommended_sampling": [("temperature", 0.7)],
            },
        )


def test_frozen_dataclass_with_recommended_sampling_is_hashable():
    """Tuple-of-tuples default keeps AliasProfile hashable / sharable."""
    a = AliasProfile(
        hf_path="x/y",
        recommended_sampling=(("temperature", 0.7),),
    )
    b = AliasProfile(
        hf_path="x/y",
        recommended_sampling=(("temperature", 0.7),),
    )
    # frozen dataclasses are hashable when all fields are hashable
    assert hash(a) == hash(b)
    assert a == b
