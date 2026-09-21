# SPDX-License-Identifier: Apache-2.0
"""``chip_token`` — the one mapping behind the closed ``chip`` enum.

Three contracts, in the order the transport cares about them:

1. A known brand string lands on exactly its enum token ("Apple M3
   Ultra" -> "m3-ultra"), case- and whitespace-insensitively, a
   "(Virtual)" suffix and all.
2. An Apple generation past the closed list degrades to "apple-other";
   an Intel brand string to "intel"; garbage and non-strings to
   "other". The input is never echoed and the call never raises.
3. No drift against events.json: every value the function can return is
   a registry value, and every ``m<N>[-variant]`` registry value
   round-trips from its natural brand string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rapid_mlx.telemetry.chip import _CHIP_VALUES, chip_token

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_REGISTRY = REPO_ROOT / "rapid_mlx" / "telemetry" / "events.json"

# The registry's ``m<N>[-variant]`` value shape, which the module's
# M-half mirrors token for token.
_M_VALUE_RE = re.compile(r"^m(\d+)(?:-(pro|max|ultra))?$")


def _registry_chip_enum() -> set[str]:
    """The ``chip`` enum values, loaded straight from events.json."""
    registry = json.loads(ENGINE_REGISTRY.read_text(encoding="utf-8"))
    return set(registry["enums"]["chip"]["values"])


# (brand, expected) — the free-form shapes redact._read_chip_brand() and
# platform.processor() can produce around the M-series.
_BRAND_CASES = [
    # The finding's three examples.
    ("Apple M3 Ultra", "m3-ultra"),
    ("Apple M2", "m2"),
    ("Apple M4 Pro", "m4-pro"),
    # One row per remaining shape; the drift test covers all 24 tokens.
    ("Apple M1", "m1"),
    ("Apple M1 Max", "m1-max"),
    ("Apple M5", "m5"),
    ("Apple M6 Ultra", "m6-ultra"),
    # Case-insensitive.
    ("apple m3 ultra", "m3-ultra"),
    ("APPLE M2 PRO", "m2-pro"),
    # Extra whitespace.
    ("  Apple   M3   Ultra  ", "m3-ultra"),
    # A VM/parenthesised suffix must not break the parse.
    ("Apple M3 Ultra (Virtual)", "m3-ultra"),
    ("Apple M2 (Virtual)", "m2"),
    # The bare HARDWARE_PROFILES key form classify_chip_tier accepts.
    ("M4 Pro", "m4-pro"),
    ("M1", "m1"),
    # The FIRST variant token wins, mirroring chip_tier.
    ("Apple M4 Pro Max", "m4-pro"),
]


@pytest.mark.parametrize(
    ("brand", "expected"), _BRAND_CASES, ids=[repr(c[0]) for c in _BRAND_CASES]
)
def test_a_known_brand_string_maps_to_its_enum_token(brand, expected):
    assert chip_token(brand) == expected


@pytest.mark.parametrize(
    ("brand", "expected"),
    [
        # The finding's example: a real future generation.
        ("Apple M9 Max", "apple-other"),
        ("Apple M9", "apple-other"),
        ("Apple M99 Ultra", "apple-other"),
    ],
)
def test_an_apple_generation_outside_the_enum_is_apple_other(brand, expected):
    """Still Apple silicon, so the dedicated fallback — never a
    fabricated "m9" and never the input text."""

    assert chip_token(brand) == expected


@pytest.mark.parametrize(
    ("brand", "expected"),
    [
        # Real machdep.cpu.brand_string form on an Intel Mac.
        ("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz", "intel"),
        ("Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz", "intel"),
        ("Intel(R) Xeon(R) W-2140B CPU @ 3.20GHz", "intel"),
        # Bare prose form.
        ("Intel Core i9", "intel"),
        ("intel", "intel"),
    ],
)
def test_an_intel_brand_string_maps_to_intel(brand, expected):
    assert chip_token(brand) == expected


@pytest.mark.parametrize(
    "brand",
    [
        # The redact fallbacks, verbatim.
        "",
        "   ",
        "Unknown",
        "unknown",
        # "apple" without an M<gen> token is not Apple silicon per
        # chip_tier, so it is "other", not "apple-other".
        "Apple Silicon",
        # platform.processor() on Linux.
        "x86_64",
        # An incidental M<n> in an unrelated string — chip_tier's
        # documented non-Apple rejection.
        "BMW M3",
        # Round 1, P2-3: the tightened intel matcher must not claim
        # unrelated words that merely share the prefix.
        "Intelligence Core",
        # The virtualized brand string: "apple" is not a standalone
        # token, so the reused parser (deliberately) does not claim it.
        "VirtualApple @ 2.50GHz processor with 2 cores",
        "hello world",
    ],
)
def test_unrecognized_strings_map_to_other(brand):
    assert chip_token(brand) == "other"


@pytest.mark.parametrize(
    "brand",
    [
        # Round 1, P2-1: a generation token longer than int()'s 4300-digit
        # conversion limit made classify_chip_tier raise ValueError
        # before the guard existed. Both the Apple-prefixed and the bare
        # (profile-key) form reach the int() conversion.
        "Apple M" + "9" * 5000,
        "M" + "9" * 5000,
    ],
    ids=["apple-prefixed", "bare"],
)
def test_an_absurdly_long_generation_token_never_raises(brand):
    assert chip_token(brand) == "other"


@pytest.mark.parametrize(
    "brand",
    [
        None,
        123,
        3.14,
        True,
        ["Apple M3"],
        {"chip": "Apple M3"},
        b"Apple M3",
    ],
)
def test_a_non_string_never_raises_and_maps_to_other(brand):
    assert chip_token(brand) == "other"


def test_the_result_is_always_a_registry_value_and_never_echoes_the_input():
    """The wire guarantee, walked over a corpus of every shape above."""

    enum = _registry_chip_enum()
    corpus = [case[0] for case in _BRAND_CASES] + [
        "Apple M9 Max",
        "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
        "",
        "   ",
        "Apple Silicon",
        "secret-internal-brand-string",
        None,
        123,
    ]
    for brand in corpus:
        token = chip_token(brand)
        assert token in enum, (brand, token)
        # An output equal to its input is allowed only when the input is
        # itself an enum value (e.g. the bare "intel" string above);
        # otherwise the mapper would be echoing caller text.
        assert token != brand or brand in enum


def test_every_value_chip_token_can_return_is_a_registry_value():
    """Drift (a): ``_CHIP_VALUES`` is exactly the function's return
    universe — every ``return`` hands back one of its members — so the
    module's closed set staying a subset of the events.json enum is what
    keeps free-form text off the wire if the registry ever renames or
    drops a value."""

    assert _registry_chip_enum().issuperset(_CHIP_VALUES)


def _registry_m_values() -> list[str]:
    """The enum's ``m<N>[-variant]`` values, with a loud failure instead
    of a vacuous pass if the pattern ever stops matching the registry's
    spelling."""

    values = _registry_chip_enum()
    m_values = sorted(value for value in values if _M_VALUE_RE.match(value))
    skipped = sorted(
        value
        for value in values
        if value.startswith("m") and not _M_VALUE_RE.match(value)
    )
    assert m_values, "chip enum no longer carries any m<N>[-variant] value"
    assert not skipped, f"m-values the round-trip pattern missed: {skipped}"
    return m_values


@pytest.mark.parametrize("value", _registry_m_values())
def test_every_registry_m_value_round_trips_from_its_brand_string(value):
    """Drift (b): for every ``m<N>[-variant]`` value in events.json, the
    natural brand string "Apple M<N> <Variant>" maps back to EXACTLY that
    value. The expected value comes from the JSON file and the brand
    string is built independently of the module's table, so a registry
    edit the mapper does not follow turns this red instead of silently
    degrading to "apple-other"."""

    match = _M_VALUE_RE.match(value)
    assert match is not None
    generation, variant = match.group(1), match.group(2)
    brand = (
        f"Apple M{generation} {variant.capitalize()}"
        if variant
        else f"Apple M{generation}"
    )
    assert chip_token(brand) == value
