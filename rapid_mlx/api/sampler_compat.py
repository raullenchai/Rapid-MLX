# SPDX-License-Identifier: Apache-2.0
"""Roleplay-front-end sampler fields: accept what we run, refuse the rest.

SillyTavern, KoboldCpp-style clients and text-generation-webui send a wide
sampler payload on every request, most of it at "off" values. Pydantic would
silently drop any field the request models do not declare, so a user who
turns on, say, XTC would get plain sampling with no hint. This module makes
that explicit:

* Fields Rapid-MLX implements are declared on the request models (DRY,
  ``repetition_penalty_range``) and documented below.
* Fields it does NOT implement are accepted only at their neutral ("off")
  value — so a stock SillyTavern payload works — and rejected with a 400 that
  names the field as soon as the user actually enables one.
* KoboldCpp's ``rep_pen_range`` spelling is folded into
  ``repetition_penalty_range``.
"""

from __future__ import annotations

import json
import math
from typing import Any

# Sampler settings Rapid-MLX does not implement, with the values that mean
# "off" in the clients that send them. ``None`` (or absence) is always off.
UNSUPPORTED_SAMPLERS: dict[str, tuple[Any, ...]] = {
    "typical_p": (1.0,),
    "typical": (1.0,),
    "tfs": (1.0,),
    "tfs_z": (1.0,),
    "top_a": (0.0,),
    "mirostat": (0,),
    "mirostat_mode": (0,),
    "xtc_probability": (0.0,),
    "smoothing_factor": (0.0,),
    "dynamic_temperature": (False,),
    "dynatemp_range": (0.0,),
    "epsilon_cutoff": (0.0,),
    "eta_cutoff": (0.0,),
    "penalty_alpha": (0.0,),
    "no_repeat_ngram_size": (0,),
    "encoder_repetition_penalty": (1.0,),
    "guidance_scale": (1.0,),
    "nsigma": (0.0,),
    "top_n_sigma": (0.0, -1.0),
    "skew": (0.0,),
    # These controls are present in generic SillyTavern/Kobold payloads too.
    # Accept their stock identities, but never let an enabled value disappear
    # through Pydantic's extra-field ignore policy.
    "rep_pen_slope": (1.0,),
    "sampler_order": ([6, 0, 1, 3, 4, 2, 5],),
    "temperature_last": (False,),
    "custom_token_bans": ("", []),
    "banned_strings": ([],),
}

SUPPORTED_SUMMARY = (
    "temperature, top_p, top_k, min_p, repetition_penalty "
    "(+ repetition_penalty_range), presence_penalty, frequency_penalty, "
    "DRY (dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, "
    "dry_sequence_breakers)"
)

_RANGE_ALIASES = ("rep_pen_range",)
MAX_SEQUENCE_BREAKERS = 64
MAX_SEQUENCE_BREAKER_CHARS = 32


def _is_neutral(value: Any, neutral: tuple[Any, ...]) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or any(isinstance(option, bool) for option in neutral):
        return isinstance(value, bool) and any(
            isinstance(option, bool) and value is option for option in neutral
        )
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return False
        return any(
            isinstance(option, (int, float))
            and not isinstance(option, bool)
            and float(value) == float(option)
            for option in neutral
        )
    return any(value == option for option in neutral)


def apply_sampler_compat(data: Any) -> Any:
    """``model_validator(mode="before")`` hook for the OpenAI request models."""
    if not isinstance(data, dict):
        return data
    for key, neutral in UNSUPPORTED_SAMPLERS.items():
        if key in data and not _is_neutral(data[key], neutral):
            raise ValueError(
                f"sampler setting '{key}' is not supported by Rapid-MLX; set it "
                f"to {neutral[0]!r} or remove it. Supported samplers: "
                f"{SUPPORTED_SUMMARY}."
            )
    present = [alias for alias in _RANGE_ALIASES if alias in data]
    if present:
        data = dict(data)
        for alias in present:
            value = data.pop(alias)
            data.setdefault("repetition_penalty_range", value)
    return data


def parse_sequence_breakers(value: Any) -> list[str] | None:
    """Accept a list of strings or the JSON-encoded list SillyTavern sends."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(
                "dry_sequence_breakers must be a list of strings or a JSON "
                "array of strings"
            ) from None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("dry_sequence_breakers must be a list of strings")
    if len(value) > MAX_SEQUENCE_BREAKERS or any(
        not item or len(item) > MAX_SEQUENCE_BREAKER_CHARS for item in value
    ):
        raise ValueError(
            f"dry_sequence_breakers allows at most {MAX_SEQUENCE_BREAKERS} "
            f"non-empty strings of up to {MAX_SEQUENCE_BREAKER_CHARS} characters"
        )
    return value
