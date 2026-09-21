# SPDX-License-Identifier: Apache-2.0
"""``chip`` — map the raw brand string onto the closed ``chip`` enum.

Today ``redact._read_chip_brand()`` reports the raw
``machdep.cpu.brand_string`` (e.g. ``"Apple M3 Ultra"``), which is
free-form text, and telemetry v2 forbids free-form strings on the wire.
The ``chip`` enum in ``events.json`` is the closed vocabulary
:func:`rapid_mlx.chip_tier.classify_chip_tier` already parses — an
``M<generation>`` token crossed with the base/Pro/Max/Ultra variants —
plus three fallbacks: ``apple-other`` for an Apple generation past the
closed list, ``intel`` for Intel Macs, ``other`` for non-Darwin
fallbacks and anything unrecognized.

This module is that mapping, and the only one: the v2 transport routes
every ``chip`` value through :func:`chip_token` the way emitters route
quantization through ``quant.quant_token()``. Nothing calls it yet; it
lands first so the mapping can be reviewed and drift-pinned against the
registry in isolation (``tests/test_telemetry_chip.py``).

Known limit: a translated (Rosetta) Python reports ``VirtualApple @
2.50GHz processor`` — no standalone ``apple`` token, no ``M<n>`` — so
such Macs report ``other`` and undercount Apple silicon.

The contract, in the order it matters:

* the return value is ALWAYS one of ``events.json`` ->
  ``enums.chip.values`` — never a slice of the input, so a brand string
  can never ride the wire;
* it never raises: a non-string, an empty string, or unparseable
  garbage — even a brand string whose generation token is thousands of
  digits long, which would overflow ``int()`` inside the classifier —
  collapses to ``other``;
* generation/variant parsing is delegated to
  :func:`rapid_mlx.chip_tier.classify_chip_tier` (case-insensitive,
  whitespace-tolerant, accepts the bare ``"M4 Pro"`` profile-key form,
  ignores parenthesised suffixes like ``"(Virtual)"``), so this module
  owns only the enum-close step and the two non-Apple fallbacks.
"""

from __future__ import annotations

from rapid_mlx.chip_tier import VARIANT_BASE, classify_chip_tier

#: The closed ``chip`` enum, verbatim from ``events.json`` ->
#: ``enums.chip.values``. A literal, like quant's token table, so it can
#: be eyeballed against the registry — and the gate every composed
#: ``m<generation>[-<variant>]`` token must pass before it is returned,
#: so the set is load-bearing in production, not just in the drift test.
#: That test pins it in BOTH directions: a new registry generation fails
#: the round-trip test until it is added here, and a stale token here
#: fails the subset test.
_CHIP_VALUES: frozenset[str] = frozenset(
    {
        "m1",
        "m1-pro",
        "m1-max",
        "m1-ultra",
        "m2",
        "m2-pro",
        "m2-max",
        "m2-ultra",
        "m3",
        "m3-pro",
        "m3-max",
        "m3-ultra",
        "m4",
        "m4-pro",
        "m4-max",
        "m4-ultra",
        "m5",
        "m5-pro",
        "m5-max",
        "m5-ultra",
        "m6",
        "m6-pro",
        "m6-max",
        "m6-ultra",
        "apple-other",
        "intel",
        "other",
    }
)


def chip_token(brand: str | None) -> str:
    """Closed ``chip`` enum value for a raw chip brand string.

    ``"Apple M3 Ultra"`` -> ``"m3-ultra"``, ``"Apple M2"`` -> ``"m2"``,
    ``"Apple M4 Pro"`` -> ``"m4-pro"``. Matching is case-insensitive and
    tolerant of extra whitespace and suffixes like ``"(Virtual)"`` —
    exactly what :func:`~rapid_mlx.chip_tier.classify_chip_tier`
    accepts.

    An Apple chip whose generation is not in the enum (``"Apple M9
    Max"``) collapses to ``apple-other``; an Intel brand string
    (``"Intel(R) Core(TM) i7 ..."``) to ``intel``; anything else — a
    non-string, an empty string, a non-Apple non-Intel string — to
    ``other``. The return value is always one of ``_CHIP_VALUES`` and
    the input is never echoed; the function never raises.
    """
    if not isinstance(brand, str):
        # The current caller (redact._read_chip_brand()) returns str, but
        # the closed-enum contract holds even against a future caller
        # bug: anything that is not a brand string is "other", never an
        # exception and never input-shaped text.
        return "other"

    try:
        tier = classify_chip_tier(brand)
    except Exception:
        # classify_chip_tier is not contractually total: a pathological
        # brand string — a generation token longer than int()'s 4300-digit
        # conversion limit — raises ValueError from the regex handler
        # (round 1, P2-1). The closed-enum contract outranks it: degrade
        # to "other", never propagate.
        return "other"

    if tier.is_apple_silicon:
        # ChipTier guarantees generation is an int whenever
        # is_apple_silicon is True.
        if tier.variant == VARIANT_BASE:
            token = f"m{tier.generation}"
        else:
            token = f"m{tier.generation}-{tier.variant.lower()}"
        if token in _CHIP_VALUES:
            return token
        # Apple silicon, but a generation the closed list does not carry
        # (a composed token can never equal a fallback value): the
        # dedicated fallback rather than a fabricated "m9".
        return "apple-other"

    # Not Apple M-series. classify_chip_tier lumps Intel in with every
    # other unknown, so the one distinction the enum still carries is
    # recovered from the brand string itself: real sysctl output on an
    # Intel Mac reads "Intel(R) Core(TM) i7-9750H ...", i.e. a token that
    # is exactly "intel" or starts with "intel(" (the parenthesised
    # form). A bare prefix match would claim unrelated words —
    # "Intelligence Core" — so it is deliberately tighter.
    if any(
        part == "intel" or part.startswith("intel(") for part in brand.lower().split()
    ):
        return "intel"
    # "", "   ", "Apple Silicon", "Unknown", "x86_64", "BMW M3", ... —
    # the redact fallbacks and unrecognized strings, exactly "other".
    return "other"
