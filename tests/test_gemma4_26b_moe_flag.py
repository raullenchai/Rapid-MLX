# SPDX-License-Identifier: Apache-2.0
"""``is_moe`` must be consistent across the Gemma 4 26B-A4B family.

Before this test the three quantizations of the *same* checkpoint disagreed:
``gemma-4-26b-4bit`` had no ``is_moe`` key at all (so it resolved to the
``False`` default), ``gemma-4-26b-qat-4bit`` said ``false``, and
``gemma-4-26b-8bit`` said ``true``. The value had also been copied onto the
assistant drafter, which is the one entry in the family that genuinely is
dense.

The ground truth is the checkpoint's own ``config.json``:

* ``mlx-community/gemma-4-26b-a4b-it-4bit`` — ``text_config.num_experts =
  128``, ``top_k_experts = 8``, ``moe_intermediate_size = 704``. All three
  quantizations are conversions of that one checkpoint, so they cannot
  disagree.
* ``mlx-community/gemma-4-26B-A4B-it-assistant-bf16`` — ``model_type =
  gemma4_assistant``, ``text_config.num_experts = None``, four dense
  layers. It is the 0.4B speculative drafter, not a copy of the target.

``is_moe`` is not cosmetic: :mod:`vllm_mlx.speculative.dflash.eligibility`
rejects MoE aliases outright (drafter hidden-state fusion misfires on
expert-routing churn) and :mod:`vllm_mlx._mxfp4_moe_guardrail` keys off it,
so a dense-tagged MoE alias can pass a gate that exists to stop it. Today
the 4-bit entries are also blocked by the ``precision >= 8-bit`` criterion,
which is exactly why the mis-tag survived review — the wrong answer was
masked by an unrelated gate rather than being harmless.
"""

from __future__ import annotations

import pytest

from vllm_mlx.model_aliases import list_profiles

# Every alias below resolves to a conversion of the same 26B-A4B checkpoint.
_MOE_TARGETS = (
    "gemma-4-26b-4bit",
    "gemma-4-26b-qat-4bit",
    "gemma-4-26b-8bit",
)

# The ``-assistant`` entries are Google's 4-layer speculative drafters, one
# per base size. None of them is MoE, including the one paired with the MoE
# target.
_DENSE_ASSISTANTS = (
    "gemma-4-e2b-assistant",
    "gemma-4-e4b-assistant",
    "gemma-4-12b-assistant",
    "gemma-4-26b-assistant",
    "gemma-4-31b-assistant",
    "gemma-4-12b-qat-assistant-4bit",
)


@pytest.mark.parametrize("alias", _MOE_TARGETS)
def test_26b_targets_are_tagged_moe(alias: str) -> None:
    profiles = list_profiles()
    assert alias in profiles, f"{alias}: missing from aliases.json"
    assert profiles[alias].is_moe is True, (
        f"{alias}: is_moe={profiles[alias].is_moe!r}, expected True. "
        f"gemma-4-26B-A4B is a 128-expert / top-8 MoE (25.2B total, 3.8B "
        f"active); every quantization of it must agree."
    )


@pytest.mark.parametrize("alias", _DENSE_ASSISTANTS)
def test_assistant_drafters_are_tagged_dense(alias: str) -> None:
    profiles = list_profiles()
    assert alias in profiles, f"{alias}: missing from aliases.json"
    assert profiles[alias].is_moe is False, (
        f"{alias}: is_moe={profiles[alias].is_moe!r}, expected False. The "
        f"gemma-4-*-assistant checkpoints are 4-layer dense drafters "
        f"(``num_experts: null``); the flag belongs on the target they "
        f"draft for, not on the drafter."
    )


def test_26b_family_agrees_with_itself() -> None:
    """The failure mode this file exists for: same checkpoint, three
    quantizations, three different answers. Assert agreement directly so a
    future entry that copies the wrong neighbour is caught even if the
    expected value above is ever revised.
    """
    profiles = list_profiles()
    flags = {a: profiles[a].is_moe for a in _MOE_TARGETS if a in profiles}
    assert len(set(flags.values())) == 1, (
        f"quantizations of one checkpoint disagree on is_moe: {flags}"
    )
