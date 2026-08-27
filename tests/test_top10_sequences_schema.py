# SPDX-License-Identifier: Apache-2.0
"""Schema validation for ``tests/integrations/top10_sequences.yaml``.

Issue #2496 checked in a declarative spec of the exact `/v1/models/load`
sequences the Tier-1 gate must drive so the "second load after a primary"
path — the class that shipped the 0.13.1 MTP release blocker #2438 —
is exercised every gate run. This test keeps that spec well-formed so a
future maintainer can extend it without silently breaking the gate driver.

This test is pure-CPU / no MLX. It reads the YAML file and, when available,
cross-checks every alias against ``vllm_mlx/aliases.json``. ``aliases.json``
is plain data (no mlx import at module load), so the whole module stays out
of the no-MLX Linux test-matrix constraint.

What it pins:
  * ``top_10_aliases`` is a non-empty ordered list of strings.
  * Every sequence has a unique name, an ordered, non-empty ``steps`` list,
    and the required scalar fields with the documented allowed values.
  * ``mtp_first: true`` means the first step's model IS the declared primary
    and is one of the top-10 aliases (so the gate can reproduce the MTP lane).
  * A sequence whose name is ``*-aba`` loads the SAME model first and last,
    and a DIFFERENT model in between (the A->B->A invariant).
  * Every step ``model`` is either a top-10 alias or a local absolute path,
    and every alias referenced anywhere resolves in the repo's
    ``vllm_mlx/aliases.json`` (checked against the live file the loader uses).
  * Per-step ``expected_status`` present; ``replace_group`` constrained to
    ``"assistant"`` or ``null``; ``replace_mode`` constrained to the loader's
    accepted set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEQUENCES_FILE = _REPO_ROOT / "tests" / "integrations" / "top10_sequences.yaml"
_ALIASES_FILE = _REPO_ROOT / "vllm_mlx" / "aliases.json"

VALID_REPLACE_MODES = frozenset({"reject", "wait", "abort"})
VALID_ACTIONS = frozenset({"load"})


@pytest.fixture(scope="module")
def spec() -> dict:
    assert _SEQUENCES_FILE.is_file(), f"missing {_SEQUENCES_FILE}"
    with open(_SEQUENCES_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _aliases() -> dict:
    return json.loads(_ALIASES_FILE.read_text(encoding="utf-8"))


def _canonical_name(alias: str) -> str:
    """A step may reference an alias string; use it as-is (aliases are the
    keys of ``aliases.json``). Local snapshot paths start with ``/``."""
    return alias


def test_yaml_is_a_mapping(spec) -> None:
    assert isinstance(spec, dict), "top-level YAML must be a mapping"
    for key in (
        "base_schema_version",
        "top_10_aliases",
        "sequences",
    ):
        assert key in spec, f"missing top-level key {key!r}"


def test_schema_version_is_int(spec) -> None:
    assert isinstance(spec["base_schema_version"], int)
    assert spec["base_schema_version"] >= 1


def test_top_10_aliases_are_strings_and_unique(spec) -> None:
    aliases = spec["top_10_aliases"]
    assert isinstance(aliases, list) and aliases, "top_10_aliases must be non-empty"
    assert all(isinstance(a, str) and a for a in aliases), (
        "aliases must be non-empty strings"
    )
    assert len(set(aliases)) == len(aliases), "top_10_aliases entries must be unique"


def test_all_top_10_aliases_resolve_in_aliases_json(spec) -> None:
    keys = set(_aliases())
    for alias in spec["top_10_aliases"]:
        assert alias in keys, f"top-10 alias {alias!r} is not in vllm_mlx/aliases.json"


def _assert_step_shape(step, seq_name, idx) -> None:
    assert isinstance(step, dict), f"{seq_name} step {idx} must be a mapping"
    assert step.get("action") in VALID_ACTIONS, (
        f"{seq_name} step {idx}: action must be one of {sorted(VALID_ACTIONS)}"
    )
    assert step.get("model") and isinstance(step["model"], str), (
        f"{seq_name} step {idx}: requires non-empty 'model'"
    )
    assert "expected_status" in step and isinstance(step["expected_status"], int), (
        f"{seq_name} step {idx}: requires int 'expected_status'"
    )
    rg = step.get("replace_group")
    assert rg in (None, "assistant"), (
        f"{seq_name} step {idx}: replace_group must be 'assistant' or null, got {rg!r}"
    )
    rm = step.get("replace_mode", "reject")
    assert rm in VALID_REPLACE_MODES, (
        f"{seq_name} step {idx}: replace_mode must be one of "
        f"{sorted(VALID_REPLACE_MODES)}, got {rm!r}"
    )
    if "estimated_size_gb" in step and step["estimated_size_gb"] is not None:
        assert isinstance(step["estimated_size_gb"], (int, float)) and (
            step["estimated_size_gb"] > 0
        ), f"{seq_name} step {idx}: estimated_size_gb must be > 0 if present"


def _is_alias_ref(model, top_aliases) -> bool:
    return isinstance(model, str) and (model in top_aliases or model.startswith("/"))


def test_every_sequence_is_well_formed(spec) -> None:
    top_aliases = set(spec["top_10_aliases"])
    aliases_keys = set(_aliases())
    seen_names = set()
    for seq in spec["sequences"]:
        assert isinstance(seq, dict), "each sequence must be a mapping"
        name = seq.get("name")
        assert name and name not in seen_names, (
            "sequence names must be unique, non-empty"
        )
        seen_names.add(name)
        assert isinstance(seq.get("mtp_first"), bool), f"{name}: mtp_first must be bool"
        assert seq.get("primary_alias") in top_aliases, (
            f"{name}: primary_alias must be in top_10_aliases"
        )
        steps = seq.get("steps")
        assert isinstance(steps, list) and len(steps) >= 2, (
            f"{name}: a meaningful sequence needs >= 2 ordered steps"
        )
        if "serve_extra" in seq:
            assert isinstance(seq["serve_extra"], list), (
                f"{name}: serve_extra must be a list"
            )
        # Every referenced alias must resolve in the live aliases table.
        for step in steps:
            model = step["model"]
            if model in top_aliases:
                assert model in aliases_keys, (
                    f"{name}: model {model!r} is a top-10 alias but missing from aliases.json"
                )
            else:
                assert model.startswith("/"), (
                    f"{name}: model {model!r} must be a top-10 alias or an absolute local path"
                )
        # Ordered steps each well-formed.
        for idx, step in enumerate(steps):
            _assert_step_shape(step, name, idx)


def test_mtp_first_loads_mtp_primary_before_a_second_model(spec) -> None:
    """mtp_first: true => step 1 is the declared primary (an MTP-capable top-10
    alias), and a second, DIFFERENT model follows — that ordering is what
    reaches the #2438 'second load after an MTP primary' path."""
    top_aliases = set(spec["top_10_aliases"])
    for seq in spec["sequences"]:
        if not seq["mtp_first"]:
            continue
        first_model = seq["steps"][0]["model"]
        assert first_model == seq["primary_alias"], (
            f"{seq['name']}: with mtp_first=true, step 1 must load the declared "
            f"primary_alias {seq['primary_alias']!r}, got {first_model!r}"
        )
        assert first_model in top_aliases, (
            f"{seq['name']}: mtp_first primary must be a top-10 alias"
        )
        second_model = seq["steps"][1]["model"]
        assert second_model != first_model, (
            f"{seq['name']}: step 2 must be a DIFFERENT model to reach the "
            f"load-after-a-primary path"
        )


def test_mtp_first_sequences_declare_serve_speculative_config(spec) -> None:
    """An mtp_first sequence must describe how the gate enables MTP on the
    primary (serve_extra with --speculative-config), otherwise there is no
    Stream(gpu,3) to tear down and the sequence silently loses its value."""
    for seq in spec["sequences"]:
        if not seq["mtp_first"]:
            continue
        extra = seq.get("serve_extra") or []
        assert "--speculative-config" in extra, (
            f"{seq['name']}: mtp_first=true requires serve_extra to include "
            f"--speculative-config so the gate boots an MTP primary"
        )


def test_aba_sequences_load_same_model_first_and_last(spec) -> None:
    """An ``*-aba`` sequence keeps the A->B->A invariant: first and last step
    load the SAME model; the middle step loads a DIFFERENT one."""
    for seq in spec["sequences"]:
        if not seq["name"].endswith("-aba"):
            continue
        steps = seq["steps"]
        assert steps[0]["model"] == steps[-1]["model"], (
            f"{seq['name']}: A->B->A requires first and last step to load the same model"
        )
        middle_models = [s["model"] for s in steps[1:-1]]
        assert middle_models, f"{seq['name']}: A->B->A requires an in-between (B) step"
        assert all(m != steps[0]["model"] for m in middle_models), (
            f"{seq['name']}: the in-between (B) step must load a DIFFERENT model"
        )


def test_second_load_after_primary_requests_replacement(spec) -> None:
    """The teardown path is only reached when the second load names the
    text group (`replace_group: assistant`). A bare secondary load would never
    exercise #2438 — pin that every 3+ step sequence replaces on its second
    load."""
    for seq in spec["sequences"]:
        steps = seq["steps"]
        if len(steps) < 3:
            continue
        second: dict = steps[1]
        assert second.get("replace_group") == "assistant", (
            f"{seq['name']}: step 2 must use replace_group='assistant' or it "
            f"won't tear down the primary (no #2438 coverage)"
        )
