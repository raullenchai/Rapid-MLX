# SPDX-License-Identifier: Apache-2.0
"""``quant`` — one canonical quantization token per model name.

The ``quant`` enum in ``events.json`` is a *canonical* vocabulary: one
token per quantization concept. The catalog spells the same concept two
ways — ``granite-4.2-30b-4bit`` and its ``ibm-granite/granite-4.2-30b-q4-mlx``
checkpoint, ``cogvideox-fun-5b-q4``, ``ltx-2.5-mlx-q8`` — so the registry
would need ``q4`` *and* ``4bit`` if it took alias spellings verbatim. Two
enum values for one concept split every chart and leave a future
classifier free to pick either, so instead the ``q<n>`` spelling is
**normalized here** to ``<n>bit`` and the enum stays canonical.

This module is the normalization, and it is the only one: block 5's
emitters call :func:`quant_token`, and
``tests/test_telemetry_registry_drift.py`` walks every catalog alias and
``hf_path`` through it, so a new alias spelling that this table does not
recognize fails CI instead of quietly riding out as ``unknown``.

**Nothing here transmits**, and the return value is always one of the
``quant`` enum's values — never free text taken from the input.
"""

from __future__ import annotations

import re

#: Split a name into alphanumeric runs: ``ibm-granite/granite-4.2-30b-q4-mlx``
#: -> ``ibm granite granite 4 2 30b q4 mlx``.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Raw name token -> canonical ``quant`` enum value. The ``q<n>`` rows are
#: the normalization the module exists for; the rest are identities.
_CANONICAL_BY_TOKEN: dict[str, str] = {
    "2bit": "2bit",
    "3bit": "3bit",
    "4bit": "4bit",
    "6bit": "6bit",
    "8bit": "8bit",
    "q2": "2bit",
    "q3": "3bit",
    "q4": "4bit",
    "q6": "6bit",
    "q8": "8bit",
    "bf16": "bf16",
    "fp16": "fp16",
    "mxfp4": "mxfp4",
    "nvfp4": "nvfp4",
    "dwq": "dwq",
    # ``mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8``: a DQ3 dynamic quant that
    # also carries ``q8``. Without this row the name reports ``8bit`` for a
    # 3-bit-dominant checkpoint, because ``q8`` is the only token either
    # regex can see.
    "dq3": "3bit",
}

#: A token that *looks* like a quantization or precision marker but is not in
#: the table above resolves to ``other`` rather than ``unknown``: we did
#: recognize a quantization, we just have no canonical name for it yet. The
#: float widths are ``[a-z]{0,2}fp\d+`` / ``bf\d+`` rather than only the
#: two-letter form, so ``-fp8`` and ``-fp32`` land on ``other`` like ``bf16``
#: and ``fp16`` land on themselves -- this enum already treats precision as a
#: quant value, and reporting an FP8 checkpoint as "no marker at all" would be
#: a lie in the one direction the enum cannot correct later.
_QUANT_SHAPED_RE = re.compile(
    r"^(?:q\d+|dq\d+|\d+bit|\d+bpw|int\d+|[a-z]{0,2}fp\d+|bf\d+|nf\d+|awq|gptq)$"
)

#: Precedence when a name carries more than one marker, most specific
#: first.
#:
#: A quantization SCHEME outranks a bit width: ``gpt-oss-20b-mxfp4-q8`` is
#: an MXFP4 checkpoint that happens to keep 8-bit companions, and
#: ``...-4bit-dwq`` is a DWQ checkpoint. Without that, ``dwq`` and
#: ``mxfp4`` would be unreachable enum values, since those names always
#: carry a width too.
#:
#: A PRECISION does NOT, and this is the opposite call on purpose.
#: ``qwen3.8-27b-4bit-fp16`` (``rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX``)
#: is a 4-bit checkpoint with an fp16 MTP head; ranking ``fp16`` first put
#: a 4-bit model in the "not quantized" bucket, which is the one error the
#: enum cannot correct later. ``bf16`` / ``fp16`` still win when the name
#: carries no width at all (``north-mini-code-bf16``), so they stay
#: reachable. Review round 1, P1.
_PRECEDENCE: tuple[str, ...] = (
    "dwq",
    "mxfp4",
    "nvfp4",
    "2bit",
    "3bit",
    "4bit",
    "6bit",
    "8bit",
    "bf16",
    "fp16",
    "other",
)
_RANK: dict[str, int] = {value: index for index, value in enumerate(_PRECEDENCE)}
# Every value the table can produce needs a rank, or the ``min`` below
# raises KeyError on the first name that hits the unranked value.
# ``tests/test_telemetry_quant.py`` pins this in both directions.

#: No marker at all: an unquantized repo, or a name that simply does not say.
UNKNOWN = "unknown"


def known_quant_tokens() -> frozenset[str]:
    """The raw name tokens this module recognizes.

    The drift check compares the catalog's quant-shaped tokens against
    this set, so a new spelling has to be mapped deliberately.
    """

    return frozenset(_CANONICAL_BY_TOKEN)


def canonical_quant_values() -> frozenset[str]:
    """Every value :func:`quant_token` can return."""

    return frozenset(_CANONICAL_BY_TOKEN.values()) | {"other", UNKNOWN}


def quant_token(alias_or_path: str) -> str:
    """Canonical ``quant`` enum value for a catalog alias or ``hf_path``.

    Returns ``"unknown"`` when the name carries no quantization marker and
    ``"other"`` when it carries one we have no canonical name for. It
    never returns a substring of the input.

    **Emitters must pass the resolved ``hf_path``, not the alias.** A
    catalog alias is free to omit the quantization it was built at —
    ``gpt-oss-20b`` alone reports ``unknown`` while its checkpoint
    ``mlx-community/gpt-oss-20b-MXFP4-Q8`` reports ``mxfp4`` — and 34 of
    the 215 catalog aliases are in that position today. The ``hf_path``
    is also what ``telemetry_model_id`` already resolves, so it costs the
    caller nothing.
    """

    found: list[str] = []
    for raw in _TOKEN_RE.findall(alias_or_path.lower()):
        canonical = _CANONICAL_BY_TOKEN.get(raw)
        if canonical is not None:
            found.append(canonical)
        elif _QUANT_SHAPED_RE.match(raw):
            found.append("other")
    if not found:
        return UNKNOWN
    return min(found, key=lambda value: _RANK[value])
