# SPDX-License-Identifier: Apache-2.0
"""Tests for the Marvin's Garden contrastive decision pipeline.

These tests run WITHOUT mlx: the generator, validator, renderer, and SFT
converter are pure stdlib (+jsonschema, skipped cleanly when absent). The
label-readout evaluator imports mlx lazily and is only smoke-checked for
module integrity here.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MG_DIR = REPO_ROOT / "bench" / "marvins_garden"
DATA_DIR = MG_DIR / "data"

if str(MG_DIR) not in sys.path:
    sys.path.insert(0, str(MG_DIR))

render = importlib.import_module("render")
gen = importlib.import_module("generate_contrastive")
validate = importlib.import_module("validate")
sft = importlib.import_module("to_chat_sft")

TRAIN = DATA_DIR / "pairs_train.jsonl"
HELDOUT = DATA_DIR / "pairs_heldout.jsonl"


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Generator determinism
# ---------------------------------------------------------------------------


def test_generator_is_deterministic():
    train_a, held_a, manifest_a = gen.generate_dataset(seed=1234, n_groups=12, heldout_groups=4)
    train_b, held_b, manifest_b = gen.generate_dataset(seed=1234, n_groups=12, heldout_groups=4)
    assert json.dumps(train_a, sort_keys=True) == json.dumps(train_b, sort_keys=True)
    assert json.dumps(held_a, sort_keys=True) == json.dumps(held_b, sort_keys=True)
    assert manifest_a == manifest_b


def test_generator_seed_changes_scenarios():
    train_a, _, _ = gen.generate_dataset(seed=1234, n_groups=12, heldout_groups=4)
    train_b, _, _ = gen.generate_dataset(seed=5678, n_groups=12, heldout_groups=4)
    assert [r["input"] for r in train_a] != [r["input"] for r in train_b]


# ---------------------------------------------------------------------------
# Committed dataset integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [TRAIN, HELDOUT])
def test_committed_data_contrast_integrity(path):
    rows = _load_rows(path)
    assert rows, f"{path} is empty"
    groups: dict[str, dict[str, str]] = {}
    for row in rows:
        members = groups.setdefault(row["contrast_group"], {})
        members[row["pair_id"].rsplit("-", 1)[-1]] = row
        assert row["label"] in row["candidates"]
        assert row["flip_key"] in validate.KNOWN_FAMILIES[row["family"]]
    for group, members in groups.items():
        assert sorted(members) == ["a", "b"], group
        labels = {m["label"] for m in members.values()}
        assert len(labels) == 2, f"{group}: contrastive labels must flip"


def test_committed_data_labels_are_derivable_from_scenario():
    """The audit copy (meta.scenario) must re-derive the label via the policy rule."""
    for path in (TRAIN, HELDOUT):
        for row in _load_rows(path):
            pick = gen._FAMILIES[row["family"]]["pick"](row["meta"]["scenario"])
            assert pick == row["label"], f"{row['pair_id']}: rule says {pick!r}, label is {row['label']!r}"


def test_committed_data_matches_seed_regen():
    """The committed files must equal a fresh run with the committed seed."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    train, heldout, fresh_manifest = gen.generate_dataset(
        seed=manifest["seed"],
        n_groups=manifest["groups_total"],
        heldout_groups=manifest["groups_heldout"],
    )
    assert fresh_manifest == manifest
    assert json.dumps(train, sort_keys=True) == json.dumps(_load_rows(TRAIN), sort_keys=True)
    assert json.dumps(heldout, sort_keys=True) == json.dumps(_load_rows(HELDOUT), sort_keys=True)


# ---------------------------------------------------------------------------
# Routing menu internal consistency (prompt must state the operative facts)
# ---------------------------------------------------------------------------


def test_route_specs_state_the_rule_facts():
    for alias, spec in gen.ROUTE_SPECS.items():
        facts = gen.ROUTE_FACTS[alias]
        assert f"{facts['min_host_ram_gb']} GB" in spec, alias
        assert f"{facts['max_ctx']} tok" in spec, alias
        assert ("vision-capable" in spec) == facts["vision"], alias
    # Option lines shown to the model carry the alias name AND the spec.
    for alias, line in zip(gen.ROUTE_MENU, gen._FAMILIES["model_routing"]["option_lines"]):
        assert alias in line and gen.ROUTE_SPECS[alias] in line


def test_route_flip_keys_flip_the_pick():
    """Every flip key must be able to change the routing decision."""
    train, _, _ = gen.generate_dataset(seed=99, n_groups=48, heldout_groups=8)
    seen_keys = {row["flip_key"] for row in train if row["family"] == "model_routing"}
    assert seen_keys == set(gen.ROUTE_FLIP_KEYS)


# ---------------------------------------------------------------------------
# Rendering / letter mapping
# ---------------------------------------------------------------------------


def test_letter_mapping_covers_menu_and_rejects_overflow():
    assert render.letter_for(0) == "A"
    assert render.letter_for(7) == "H"
    with pytest.raises(ValueError):
        render.letter_for(8)
    with pytest.raises(ValueError):
        render.label_letter(["x", "y"], "z")


def test_all_styles_render_with_same_options():
    for family, spec in gen._FAMILIES.items():
        rows = [r for r in _load_rows(HELDOUT) if r["family"] == family][:1]
        assert rows, family
        row = rows[0]
        for style in render.STYLES:
            if style == "base":
                prompt = row["input"]
            else:
                prompt = render.render_prompt(
                    family,
                    spec["display_fields"](row["meta"]["scenario"]),
                    row["candidates"],
                    spec["option_lines"],
                    style=style,
                )
            for letter in ("A.", "B."):
                assert letter in prompt, (family, style, letter)
            assert (
                "Answer with a single letter" in prompt
                or "One letter only" in prompt
                or "Respond with exactly one letter" in prompt
            ), (family, style)


# ---------------------------------------------------------------------------
# SFT conversion
# ---------------------------------------------------------------------------


def test_sft_conversion_emits_single_letter_completion():
    row = _load_rows(HELDOUT)[0]
    chat = sft.pair_to_chat(row)
    assert chat["messages"][0]["role"] == "user"
    assert chat["messages"][0]["content"] == row["input"]
    completion = chat["messages"][1]["content"]
    assert completion in render.LETTERS
    assert completion == render.label_letter(row["candidates"], row["label"])


# ---------------------------------------------------------------------------
# Validator behavior
# ---------------------------------------------------------------------------


def test_validator_fails_closed_without_jsonschema(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    target = tmp_path / "whatever.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    assert validate.main([str(target)]) == 1


@pytest.mark.skipif(importlib.util.find_spec("jsonschema") is None, reason="jsonschema not installed")
def test_validator_passes_committed_data():
    assert validate.validate_file(TRAIN, validate._load_schema()) == []
    assert validate.validate_file(HELDOUT, validate._load_schema()) == []


@pytest.mark.skipif(importlib.util.find_spec("jsonschema") is None, reason="jsonschema not installed")
def test_validator_rejects_label_outside_candidates(tmp_path):
    rows = _load_rows(HELDOUT)[:1]
    rows[0]["label"] = "not-a-candidate"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    problems = validate.validate_file(bad, validate._load_schema())
    assert any("not in candidates" in p for p in problems)
