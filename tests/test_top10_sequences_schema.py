# SPDX-License-Identifier: Apache-2.0
"""Schema validation for ``tests/integrations/top10_sequences.yaml``.

Issue #2496 checked in a declarative spec of the residency-load / MTP-serve
sequences the Tier-1 gate must drive so the "second load after a primary"
path — the class that shipped the 0.13.0 / 0.13.1 MTP release blocker #2438 —
is exercised every gate run. This test keeps that spec well-formed so a
future maintainer can extend it without silently breaking the gate driver.

This test is pure-CPU / no MLX. It reads the YAML file and, when available,
cross-checks every alias against ``vllm_mlx/aliases.json``. ``aliases.json``
is plain data (no mlx import at module load), so the whole module stays out
of the no-MLX Linux test-matrix constraint.

What it pins (schema v4):
  * ``top_10_aliases`` is a non-empty ordered list of unique strings.
  * Every sequence has a unique name, an ordered, non-empty ``steps`` list,
    and the required scalar fields with the documented allowed values.
  * ``mtp`` is one of ``"none" / "first" / "second"``. Any non-"none" sequence
    must declare a ``serve_alias`` (in top_10_aliases), a ``serve_extra`` that
    enables spec-decode, and a non-empty ``metrics_expected`` — the MTP serve
    is the only way a Stream(gpu,3) / MTP accept counters exist.
  * ``mtp: first`` loads its serve_alias on step 1; ``mtp: second`` must NOT
    load serve_alias on step 1 (the MTP primary arrives at a later step).
  * A sequence whose name is ``*-aba`` loads the SAME model first and last,
    and a DIFFERENT model in between (the A->B->A invariant).
  * Every step ``model`` is a top-10 alias (v4 dropped the absolute-path form)
    and resolves in the repo's ``vllm_mlx/aliases.json``.
  * Every step carries ``expected_status``; ``replace_group`` constrained to
    ``"assistant"`` or ``null``; ``replace_mode`` constrained to the loader's
    accepted set; optional positive ``timeout_seconds`` on steps and sequences.
  * ``metrics_expected`` entries are well-formed (metric is a string, method /
    description optional strings, require_nonzero a bool, on_absent in
    {"fail","pass"}, min/max numeric when present).
  * Coverage: gemma-4-26b-4bit and bonsai-27b-2bit are actually LOADED by some
    sequence step (v3 listed them but never loaded them), and both MTP-first
    and MTP-second orderings exist.
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
VALID_MTP = frozenset({"none", "first", "second"})
VALID_ON_ABSENT = frozenset({"fail", "pass"})
# Aliases that, per the v3->v4 coverage mandate, must actually be LOADED by
# some sequence step (they were listed in top_10_aliases but never loaded).
MUST_LOAD_ALIASES = ("gemma-4-26b-4bit", "bonsai-27b-2bit")


@pytest.fixture(scope="module")
def spec() -> dict:
    assert _SEQUENCES_FILE.is_file(), f"missing {_SEQUENCES_FILE}"
    with open(_SEQUENCES_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _aliases() -> dict:
    return json.loads(_ALIASES_FILE.read_text(encoding="utf-8"))


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


def _assert_metric_expectation_shape(mx, seq_name, idx) -> None:
    assert isinstance(mx, dict), f"{seq_name} metrics_expected[{idx}] must be a mapping"
    metric = mx.get("metric")
    assert isinstance(metric, str) and metric, (
        f"{seq_name} metrics_expected[{idx}]: requires a non-empty string 'metric'"
    )
    if "method" in mx:
        assert isinstance(mx["method"], str), (
            f"{seq_name} metrics_expected[{idx}]: 'method' must be a string"
        )
    if "require_nonzero" in mx:
        assert isinstance(mx["require_nonzero"], bool), (
            f"{seq_name} metrics_expected[{idx}]: 'require_nonzero' must be bool"
        )
    for bound in ("min", "max"):
        if bound in mx and mx[bound] is not None:
            assert isinstance(mx[bound], (int, float)), (
                f"{seq_name} metrics_expected[{idx}]: '{bound}' must be numeric"
            )
    if "on_absent" in mx:
        assert mx["on_absent"] in VALID_ON_ABSENT, (
            f"{seq_name} metrics_expected[{idx}]: on_absent must be one of "
            f"{sorted(VALID_ON_ABSENT)}"
        )
    if "description" in mx:
        assert isinstance(mx["description"], str), (
            f"{seq_name} metrics_expected[{idx}]: 'description' must be a string"
        )


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
    if step.get("timeout_seconds") is not None:
        assert isinstance(step["timeout_seconds"], (int, float)) and (
            step["timeout_seconds"] > 0
        ), f"{seq_name} step {idx}: timeout_seconds must be > 0 if present"


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

        mtp = seq.get("mtp")
        assert mtp in VALID_MTP, (
            f"{name}: mtp must be one of {sorted(VALID_MTP)}, got {mtp!r}"
        )
        serve_alias = seq.get("serve_alias")
        if mtp != "none":
            assert serve_alias in top_aliases, (
                f"{name}: mtp={mtp!r} requires serve_alias in top_10_aliases"
            )
            extra = seq.get("serve_extra") or []
            assert isinstance(extra, list) and "--speculative-config" in extra, (
                f"{name}: mtp={mtp!r} requires serve_extra to include "
                f"--speculative-config so the gate boots a spec-decode MTP serve"
            )
            assert seq.get("metrics_expected"), (
                f"{name}: mtp={mtp!r} requires a non-empty metrics_expected "
                f"(the #2421 accept-rate / attempts surface)"
            )
        else:
            assert "metrics_expected" in seq and seq["metrics_expected"] == [], (
                f"{name}: non-MTP sequences must set an empty metrics_expected list"
            )

        if seq.get("timeout_seconds") is not None:
            assert isinstance(seq["timeout_seconds"], (int, float)) and (
                seq["timeout_seconds"] > 0
            ), f"{name}: timeout_seconds must be > 0 if present"

        steps = seq.get("steps")
        assert isinstance(steps, list) and len(steps) >= 2, (
            f"{name}: a meaningful sequence needs >= 2 ordered steps"
        )

        # Every referenced model must be a top-10 alias and resolve live.
        for step in steps:
            model = step["model"]
            assert model in top_aliases, (
                f"{name}: model {model!r} must be a top_10_aliases entry (v4 "
                f"dropped the absolute-path / local form)"
            )
            assert model in aliases_keys, (
                f"{name}: model {model!r} is a top-10 alias but missing from aliases.json"
            )

        for idx, step in enumerate(steps):
            _assert_step_shape(step, name, idx)
        for idx, mx in enumerate(seq.get("metrics_expected") or []):
            _assert_metric_expectation_shape(mx, name, idx)


def test_mtp_first_loads_mtp_primary_on_step1(spec) -> None:
    """mtp: first => step 1 loads the serve_alias (the MTP-served primary),
    and a second, DIFFERENT model follows — that ordering reaches the #2438
    'second load after an MTP primary' path with the MTP primary first."""
    top_aliases = set(spec["top_10_aliases"])
    for seq in spec["sequences"]:
        if seq.get("mtp") != "first":
            continue
        serve_alias = seq["serve_alias"]
        assert seq["serve_alias"] in top_aliases
        assert seq["steps"][0]["model"] == serve_alias, (
            f"{seq['name']}: with mtp=first, step 1 must load serve_alias "
            f"{serve_alias!r}, got {seq['steps'][0]['model']!r}"
        )
        second_model = seq["steps"][1]["model"]
        assert second_model != serve_alias, (
            f"{seq['name']}: step 2 must be a DIFFERENT model to reach the "
            f"load-after-a-primary path"
        )


def test_mtp_second_does_not_load_primary_on_step1(spec) -> None:
    """mtp: second => step 1 must be a DIFFERENT model; the MTP-served primary
    (serve_alias) arrives on a LATER step. Covers the opposite load order that
    a first-only sequence never visits."""
    for seq in spec["sequences"]:
        if seq.get("mtp") != "second":
            continue
        serve_alias = seq["serve_alias"]
        assert seq["steps"][0]["model"] != serve_alias, (
            f"{seq['name']}: with mtp=second, step 1 must NOT load serve_alias "
            f"{serve_alias!r}; the MTP primary arrives at a later step"
        )
        later_models = [s["model"] for s in seq["steps"][1:]]
        assert serve_alias in later_models, (
            f"{seq['name']}: with mtp=second, a LATER step must load "
            f"serve_alias {serve_alias!r}"
        )


def test_mtp_sequences_assert_attempts_and_accept_floor(spec) -> None:
    """Every MTP sequence's metrics_expected must cover BOTH the raw attempts
    counter (proves spec-decode actually ran -> there WAS a Stream(gpu,3)) and
    the accept-ratio floor (the #2421 assertion)."""
    for seq in spec["sequences"]:
        if seq.get("mtp") == "none":
            continue
        metrics = {m.get("metric") for m in seq.get("metrics_expected") or []}
        assert "rapid_mlx_spec_decode_attempts_total" in metrics, (
            f"{seq['name']}: MTP sequence must assert the attempts counter "
            f"(proves MTP ran / Stream(gpu,3) existed)"
        )
        assert "rapid_mlx_spec_decode_accept_ratio" in metrics, (
            f"{seq['name']}: MTP sequence must assert the #2421 accept_ratio floor"
        )


def test_mtp_second_exists_in_addition_to_mtp_first(spec) -> None:
    """#2438/#2496 want BOTH orderings: an MTP primary served first AND served
    second. A repo that only encodes first-forgets the opposite choreography."""
    mtp_modes = {seq.get("mtp") for seq in spec["sequences"]}
    assert "first" in mtp_modes, "at least one mtp: first sequence required"
    assert "second" in mtp_modes, "at least one mtp: second sequence required"


def test_gemma_and_bonsai_actually_load(spec) -> None:
    """v3 listed gemma-4-26b and bonsai-27b aliases but never loaded them.
    Every MUST_LOAD alias must appear as a step 'model' in some sequence."""
    loaded = {
        model
        for seq in spec["sequences"]
        for step in seq.get("steps", [])
        for model in (step["model"],)
    }
    for alias in MUST_LOAD_ALIASES:
        assert alias in loaded, (
            f"{alias} is listed in top_10_aliases but no sequence step loads it"
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
