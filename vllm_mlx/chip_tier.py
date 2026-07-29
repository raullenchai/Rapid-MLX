# SPDX-License-Identifier: Apache-2.0
"""Structured Apple-silicon chip-tier classifier.

Chip *detection* already exists in the engine:

* :func:`vllm_mlx.optimizations.detect_hardware` returns a
  :class:`~vllm_mlx.optimizations.HardwareInfo` whose ``chip_name`` is a short
  profile key (e.g. ``"M3 Ultra"``).
* :func:`detect_chip_tier` (below) reads the raw ``machdep.cpu.brand_string``
  via ``sysctl`` (e.g. ``"Apple M3 Ultra"``).

What was MISSING is a *structured* tier — a small, pure parse of that free-form
chip string into fields code can branch on without substring-matching at every
call site. This module adds exactly that and nothing more.

The classifier is a pure function over a string: it performs no I/O, imports no
heavy deps, and never raises on unexpected input (unknown / non-Apple strings
return an explicit "unknown" tier). That makes it hermetically unit-testable and
safe to call on any platform.

Current consumer: the KV-quant differential quality gate
(``scripts/kv_quant_quality_gate.py`` / :mod:`vllm_mlx.kv_quant_gate`) records
the chip tier in its report and RAM/compute-gates the optional long-context
(NIAH) metric on it — a low-RAM M1/M2 skips the expensive retrieval pass.

Future consumer (NOT wired here): MTP speculative decoding currently uses a
single global ``DEFAULT_MAX_K`` (``vllm_mlx/spec_decode/mtp/draft_k_controller_v2.py``).
The natural next use of this classifier is to tier ``max_k`` by chip generation
/ variant (a bandwidth-rich M3/M4 Ultra can profitably run a deeper draft tree
than an M1). That wiring is intentionally left to a follow-up PR — this module
only provides the classification primitive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical variant labels. Order is significant only for display; the parser
# matches case-insensitively against the exact whitespace-delimited token.
VARIANT_BASE = "base"
VARIANT_PRO = "Pro"
VARIANT_MAX = "Max"
VARIANT_ULTRA = "Ultra"
VARIANT_UNKNOWN = "unknown"

# Suffix token (lower-cased) -> canonical variant label. A chip string with an
# ``M<gen>`` token but none of these suffixes is the base variant.
_VARIANT_TOKENS: dict[str, str] = {
    "pro": VARIANT_PRO,
    "max": VARIANT_MAX,
    "ultra": VARIANT_ULTRA,
}

# ``M<generation>`` token, e.g. ``m1`` / ``m3`` / ``m4``. Anchored so that a
# stray ``m`` inside another word (``Maximum``) can't match — we test it against
# already-split whitespace tokens.
_M_TOKEN_RE = re.compile(r"^m(\d+)$")


@dataclass(frozen=True)
class ChipTier:
    """Structured view of an Apple-silicon chip string.

    Attributes:
        raw: The original chip string, unmodified (for logging / reporting).
        is_apple_silicon: True when the string parsed as an ``M<n>`` Apple chip.
            False for Intel Macs, empty strings, the ``"Apple Silicon"`` /
            ``"Unknown"`` fallbacks, and any non-Apple platform.
        generation: The integer generation (1, 2, 3, 4, ...) when known, else
            ``None``. ``None`` whenever ``is_apple_silicon`` is False.
        variant: One of :data:`VARIANT_BASE` / :data:`VARIANT_PRO` /
            :data:`VARIANT_MAX` / :data:`VARIANT_ULTRA` for a recognized Apple
            chip, or :data:`VARIANT_UNKNOWN` when the chip is unknown/non-Apple.
        is_m3_or_newer: Convenience gate — True iff this is an Apple chip whose
            generation is >= 3. Consumers use it as a cheap "modern, bandwidth-
            rich chip" signal (e.g. to enable the optional NIAH gate metric).
    """

    raw: str
    is_apple_silicon: bool
    generation: int | None
    variant: str
    is_m3_or_newer: bool


def classify_chip_tier(chip_name: str | None) -> ChipTier:
    """Parse a free-form chip string into a structured :class:`ChipTier`.

    Handles every shape the engine's existing detectors can produce:

    * ``machdep.cpu.brand_string`` form — ``"Apple M3 Ultra"``, ``"Apple M2 Pro"``,
      ``"Apple M1"``.
    * ``HARDWARE_PROFILES`` key form — ``"M4 Max"``, ``"M1"``.
    * Fallbacks / non-Apple — ``"Apple Silicon"``, ``"Unknown"``, ``""``,
      ``"Intel(R) Core(TM) i7"``, ``None`` — all classified as the explicit
      unknown tier (never raises).

    Matching is case-insensitive and token-based (whitespace-delimited), so a
    substring like ``"Max"`` inside ``"Maximum"`` cannot spuriously set the
    variant. Only the *first* ``M<n>`` token drives the generation, and it is
    accepted as an Apple chip only when the string is Apple-branded OR that token
    leads the string (the bare ``HARDWARE_PROFILES`` key form) — so an incidental
    ``M<n>`` (``"BMW M3"``) is NOT misread as Apple silicon.

    Args:
        chip_name: The chip string from ``detect_hardware().chip_name`` or the
            raw ``machdep.cpu.brand_string`` (see :func:`detect_chip_tier`).
            ``None`` is tolerated.

    Returns:
        A :class:`ChipTier`. For an unrecognized string:
        ``ChipTier(raw, is_apple_silicon=False, generation=None,
        variant="unknown", is_m3_or_newer=False)``.
    """
    raw = chip_name or ""
    tokens = raw.lower().split()
    has_apple = "apple" in tokens

    generation: int | None = None
    gen_index: int | None = None
    for idx, token in enumerate(tokens):
        match = _M_TOKEN_RE.match(token)
        if match:
            generation = int(match.group(1))
            gen_index = idx
            break

    # Accept the ``M<n>`` token as an Apple chip ONLY when the string is
    # Apple-branded (``"Apple M3 Ultra"``) OR the token LEADS the string (the
    # bare ``HARDWARE_PROFILES`` key form ``"M3 Ultra"`` / ``"M1"``). This rejects
    # an incidental ``M<n>`` in an unrelated string — ``"BMW M3"`` (gen token not
    # first, no "apple") stays the unknown tier, honoring the documented
    # non-Apple fallback (same lenient-substring lesson as the variant tokens).
    if generation is None or not (has_apple or gen_index == 0):
        # Not a recognizable Apple ``M<n>`` chip — Intel, empty, a fallback
        # string, or an incidental M-token. Explicit unknown tier; never crash.
        return ChipTier(
            raw=raw,
            is_apple_silicon=False,
            generation=None,
            variant=VARIANT_UNKNOWN,
            is_m3_or_newer=False,
        )

    variant = VARIANT_BASE
    for token in tokens:
        mapped = _VARIANT_TOKENS.get(token)
        if mapped is not None:
            variant = mapped
            break

    return ChipTier(
        raw=raw,
        is_apple_silicon=True,
        generation=generation,
        variant=variant,
        is_m3_or_newer=generation >= 3,
    )


def detect_chip_tier() -> ChipTier:
    """Classify the *current* machine's chip.

    Reads the raw ``machdep.cpu.brand_string`` (e.g. ``"Apple M3 Ultra"``) via
    ``sysctl`` and runs it through :func:`classify_chip_tier`. Any failure — a
    non-macOS host, a missing ``sysctl``, a timeout — degrades to an empty
    string and hence the unknown tier, so this never raises.
    """
    chip_name = ""
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        chip_name = result.stdout.strip()
    except Exception:
        chip_name = ""
    return classify_chip_tier(chip_name)
