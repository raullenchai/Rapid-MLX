# SPDX-License-Identifier: Apache-2.0
"""``quant_token`` — the one normalization behind the ``quant`` enum.

The contract, in the order the fix cares about it:

1. ``q<n>`` spellings normalize to ``<n>bit``; the enum has no ``q4``.
2. A name with a scheme *and* a width reports the scheme.
3. The return value is always an enum value, never a slice of the input.
"""

from __future__ import annotations

import pytest

from rapid_mlx.telemetry.quant import (
    canonical_quant_values,
    known_quant_tokens,
    quant_token,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # The finding's two examples, alias spelling and checkpoint path.
        ("ltx-2.5-mlx-q8", "8bit"),
        ("cogvideox-fun-5b-q4", "4bit"),
        ("dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q4", "4bit"),
        # The same concept, spelled the other way, must land on the same value.
        ("granite-4.2-30b-4bit", "4bit"),
        ("ibm-granite/granite-4.2-30b-q4-mlx", "4bit"),
        ("granite-4.2-30b-8bit", "8bit"),
        ("ibm-granite/granite-4.2-30b-q8-mlx", "8bit"),
        ("mflux-community/qwen-image-mflux-q6", "6bit"),
        # Identity rows.
        ("qwen3.5-4b-2bit", "2bit"),
        ("some-model-3bit", "3bit"),
        ("north-mini-code-bf16", "bf16"),
        ("some-model-fp16", "fp16"),
        ("some-model-nvfp4", "nvfp4"),
    ],
)
def test_spellings_normalize_to_one_canonical_token(name, expected):
    assert quant_token(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Scheme beats width: an MXFP4 checkpoint with 8-bit companions is
        # an MXFP4 checkpoint, and without this ``mxfp4`` / ``dwq`` would be
        # unreachable enum values.
        ("gpt-oss-20b-mxfp4-q8", "mxfp4"),
        ("mlx-community/gpt-oss-120b-MXFP4-Q4", "mxfp4"),
        ("some-model-4bit-dwq", "dwq"),
        ("some-model-8bit-nvfp4", "nvfp4"),
    ],
)
def test_the_scheme_wins_over_the_bit_width(name, expected):
    assert quant_token(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Review round 1, P1: the real catalog entry this got wrong. A
        # 4-bit checkpoint with an fp16 MTP head must not report fp16,
        # i.e. must not land in the "not quantized" bucket.
        ("qwen3.8-27b-4bit-fp16", "4bit"),
        ("rapid-mlx/Qwen3.8-27B-4bit-MTP-fp16-MLX", "4bit"),
        ("some-model-bf16-4bit", "4bit"),
        # Round 2, P2: the same rule in the ``other`` lane. An INT4
        # checkpoint with bf16 activations is quantized, whatever we call
        # the scheme.
        ("acme/model-int4-bf16", "other"),
        ("acme/model-awq-fp16", "other"),
    ],
)
def test_a_bit_width_wins_over_a_precision(name, expected):
    assert quant_token(name) == expected


def test_a_precision_still_wins_when_there_is_no_width():
    """The other half of the rule: bf16 / fp16 stay reachable."""

    assert quant_token("north-mini-code-bf16") == "bf16"
    assert quant_token("mlx-community/North-Mini-Code-1.0-bf16") == "bf16"
    assert quant_token("some-model-fp16") == "fp16"


def test_a_dynamic_quant_reports_its_own_width_not_its_companion():
    """Review round 1, P2: ``DQ3_K_M-q8`` is a 3-bit-dominant checkpoint.
    Before ``dq3`` was mapped, ``q8`` was the only token either regex could
    see and the name reported ``8bit``."""

    assert quant_token("mlx-community/Kimi-K2.6-mlx-DQ3_K_M-q8") == "3bit"
    # Round 2, P2: the DQ family mirrors the Q family rather than being a
    # one-row special case. ``DQ3_K_M`` is exactly as mixed as the
    # ``Q4_K_M`` this table already normalizes to ``4bit``.
    assert quant_token("acme/model-DQ4_K_M") == "4bit"
    assert quant_token("acme/model-DQ8_0") == "8bit"
    # A DQ width with no enum value falls through, like ``q5`` does: a
    # recognized quantization without a canonical name, not "no marker".
    assert quant_token("acme/model-DQ5_K_M") == "other"
    assert quant_token("acme/model-q5") == "other"


@pytest.mark.parametrize(
    "name",
    [
        # Quantized, but by a scheme we have no canonical name for.
        "qwen3.8-27b-mixed-3.5bpw",
        "some-model-int4",
        "some-model-awq",
        "some-model-gptq",
        "some-model-5bit",
        # Float widths the enum has no name for. This enum already treats
        # precision as a quant value (bf16 / fp16 ARE values), so reporting
        # an FP8 checkpoint as "no marker at all" would be the one lie the
        # enum cannot correct later.
        "acme/model-fp8",
        "acme/model-fp32",
        "acme/model-bf8",
        "acme/model-nf4",
    ],
)
def test_an_unnamed_quantization_is_other_not_free_text(name):
    assert quant_token(name) == "other"


@pytest.mark.parametrize(
    "name",
    [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "qwen3.5-4b",
        "",
        "acme-internal-support-bot",
    ],
)
def test_a_name_with_no_marker_is_unknown(name):
    assert quant_token(name) == "unknown"


def test_a_bit_width_is_not_confused_with_a_parameter_count():
    """``4b`` is four billion parameters; ``4bit`` is the quantization."""

    assert quant_token("qwen3.5-4b") == "unknown"
    assert quant_token("gpt-oss-120b") == "unknown"


def test_the_result_is_never_a_slice_of_the_input():
    """The guarantee that keeps caller-controlled text off the wire."""

    values = canonical_quant_values()
    for name in ("secret-project-q4", "/home/someone/models/my-tune-8bit", "q99"):
        assert quant_token(name) in values


def test_known_tokens_all_resolve_to_themselves_or_their_canonical_form():
    values = canonical_quant_values()
    for raw in known_quant_tokens():
        token = quant_token(f"model-{raw}")
        assert token in values
        assert token != "unknown"


@pytest.mark.parametrize(
    "name",
    [
        # A fiscal quarter, not a 4-bit checkpoint: the token is ``2026q4``,
        # and the marker patterns are anchored so it cannot match.
        "acme/model-2026q4",
        # Version and parameter-count tokens next to nothing quant-shaped.
        "qwen3-vl-30b",
        "deepseek-v3.2-exp",
        "mlx-community/Qwen3-VL-8B-Instruct",
    ],
)
def test_version_and_size_tokens_are_not_read_as_quantization(name):
    """False positives are worse than 'unknown': they would report a
    quantization the checkpoint does not have."""

    assert quant_token(name) == "unknown"


def test_every_canonical_value_has_a_precedence_rank():
    """``quant_token`` ranks its matches; a table value with no rank would
    raise KeyError on the first name that hits it, in a telemetry path whose
    whole contract is that it cannot take ``serve`` down."""

    from rapid_mlx.telemetry.quant import _RANK

    assert canonical_quant_values() - {"unknown"} == set(_RANK)
