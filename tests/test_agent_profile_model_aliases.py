# SPDX-License-Identifier: Apache-2.0
"""Every model an agent profile recommends must be a real alias.

``rapid-mlx agents <id>`` prints a setup guide, and the desktop Launch page
prints the same guide. The guide's "Recommended models" list is the one part a
reader is meant to copy straight into a ``pull`` command, so an alias that does
not exist fails at the first thing the guide asks the user to do.

That is what ``hermes.yaml`` shipped: it recommended ``qwen3.5-35b-a3b``, taking
the name from the upstream model card (``Qwen3.5-35B-A3B``) rather than from
``aliases.json``, where the same weights are registered as
``qwen3.5-35b-4bit``. Nothing failed at build time — the profile is data, and
nothing cross-checked it against the alias registry until a user ran it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "vllm_mlx"
_PROFILES = sorted((_ROOT / "agents" / "profiles").glob("*.yaml"))


def _valid_aliases() -> set[str]:
    return set(json.loads((_ROOT / "aliases.json").read_text()).keys())


def _recommended(profile: Path) -> list[str]:
    """The ``models.recommended`` list, read without a YAML dependency.

    Deliberately a narrow regex rather than a YAML parse: the block is a flat
    list of quoted scalars in every profile, and the test exists to be run in
    the minimal environment that `tests/test_adapters_import_without_mlx.py`
    also targets.
    """
    text = profile.read_text()
    block = re.search(r"^\s*recommended:\n((?:\s*- \"[^\"]+\"\n)+)", text, re.M)
    if block is None:
        return []
    return re.findall(r'- "([^"]+)"', block.group(1))


@pytest.mark.parametrize("profile", _PROFILES, ids=lambda p: p.stem)
def test_recommended_models_resolve(profile: Path) -> None:
    valid = _valid_aliases()
    unknown = [
        model
        for model in _recommended(profile)
        # A full HuggingFace path needs no alias entry.
        if "/" not in model and model not in valid
    ]
    assert not unknown, (
        f"{profile.name} recommends {unknown}, which `rapid-mlx pull` cannot "
        f"resolve. Check aliases.json for the registered name — the alias "
        f"often differs from the upstream model card."
    )


def test_at_least_one_profile_recommends_something() -> None:
    """Guards the regex above.

    If the profile format changes, ``_recommended`` starts returning ``[]`` for
    every file and the parametrized test passes while checking nothing.
    """
    assert any(_recommended(profile) for profile in _PROFILES)
