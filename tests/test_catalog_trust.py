# SPDX-License-Identifier: Apache-2.0
"""Differential catalog identity checks use only local JSON, never remote HEAD."""

import json

import pytest

from scripts.catalog_trust import catalog_changes, is_trust_control, parse_catalog

PATH = "rapid_mlx/aliases.json"
PIN = "a" * 40


@pytest.mark.parametrize(
    "entry",
    [
        "org/model",
        {"hf_path": "org/model"},
        {"hf_path": "org/model", "revision": "main"},
    ],
)
def test_new_mutable_catalog_source_is_rejected(entry):
    changed, errors = catalog_changes({}, {"new": entry}, PATH)
    assert changed == ["new"]
    assert errors


def test_reviewed_new_identity_and_semantic_weight_change_require_review():
    original = {"alias": {"hf_path": "org/model", "revision": PIN}}
    assert catalog_changes({}, original, PATH) == (["alias"], [])
    updated = {"alias": {**original["alias"], "is_text_only": True}}
    assert catalog_changes(original, updated, PATH) == (["alias"], [])


def test_cosmetic_edit_does_not_force_unreviewed_mass_pinning():
    before = {"old": {"hf_path": "org/model", "notes": "old"}}
    after = {"old": {"hf_path": "org/model", "notes": "new"}}
    assert catalog_changes(before, after, PATH) == ([], [])
    after["old"]["hf_path"] = "attacker/model"
    assert catalog_changes(before, after, PATH)[1]


def test_removing_pin_or_changing_subfolder_rejected():
    assert catalog_changes(
        {"x": {"hf_path": "org/model", "revision": PIN}},
        {"x": {"hf_path": "org/model"}},
        PATH,
    )[1]
    assert catalog_changes(
        {"x": {"hf_path": "org/model"}},
        {"x": {"hf_path": "org/model", "subfolder": "other"}},
        PATH,
    )[1]


def test_audio_runtime_package_changes_are_not_cosmetic():
    before = {"_runtime_requirements": {"kokoro": []}}
    after = {"_runtime_requirements": {"kokoro": ["attacker"]}}
    assert catalog_changes(before, after, "rapid_mlx/audio/aliases.json") == (
        ["_runtime_requirements"],
        [],
    )


@pytest.mark.parametrize(
    "path",
    [
        "scripts/catalog_trust.py",
        "scripts/merge_on_green.py",
        "scripts/pr_validate/steps/supply_chain.py",
        ".github/workflows/validate-community-submission.yml",
        "rapid_mlx/audio/registry.py",
        "rapid_mlx/audio/runtime_requirements.py",
        "rapid_mlx/audio/tts.py",
        "rapid_mlx/audio/processor.py",
        "rapid_mlx/audio/sam_security.py",
        "rapid_mlx/audio/sa3/models/defs/checkpoint_security.py",
        "rapid_mlx/utils/torch_checkpoint.py",
        "rapid_mlx/catalog/schemas/model-alias.schema.json",
    ],
)
def test_gate_edits_require_human_review(path):
    assert is_trust_control(path)


def test_duplicate_json_keys_fail_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_catalog('{"x": {"hf_path": "a", "hf_path": "b"}}')


def test_supply_chain_blocks_catalog_only_unpinned_pr(tmp_path, monkeypatch):
    import scripts.catalog_trust as policy
    from scripts.pr_validate.context import Context
    from scripts.pr_validate.steps.supply_chain import SupplyChainStep

    root = tmp_path / "repo"
    (root / "rapid_mlx").mkdir(parents=True)
    (root / PATH).write_text(json.dumps({"evil": {"hf_path": "attacker/model"}}))
    diff = tmp_path / "diff"
    diff.write_text("")
    ctx = Context(pr_number=1)
    ctx.repo_root = root
    ctx.base_sha = "b" * 40
    ctx.files_changed = [PATH]
    ctx.diff_path = str(diff)
    ctx.work_dir = tmp_path / "artifacts"
    monkeypatch.setattr(policy, "read_git_catalog", lambda *_args: {})
    result = SupplyChainStep().run(ctx)
    assert result.status == "fail"
    assert any("reviewed full 40-hex" in finding for finding in result.findings)


def test_alias_reordering_requires_review_of_reverse_lookup_winner():
    before = {
        "one": {"hf_path": "org/model", "is_text_only": True},
        "two": {"hf_path": "org/model", "is_text_only": False},
    }
    assert catalog_changes(before, dict(reversed(list(before.items()))), PATH) == (
        ["<alias-order>"],
        [],
    )


def test_controller_package_initializer_is_a_trust_control():
    assert is_trust_control("scripts/__init__.py")
