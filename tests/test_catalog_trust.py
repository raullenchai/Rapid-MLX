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
        "scripts/vendor_ltx25.py",
        "ltx_core_mlx/model/transformer/model.py",
        "ltx_pipelines_mlx/cli.py",
        "rapid_mlx/kernels/affine8_attention.py",
        "rapid_mlx/kernels/affine8_attention.metal",
        "scripts/merge_on_green.py",
        "scripts/pr_validate/steps/supply_chain.py",
        ".github/workflows/validate-community-submission.yml",
        "rapid_mlx/audio/registry.py",
        "rapid_mlx/audio/runtime_requirements.py",
        "rapid_mlx/audio/stt.py",
        "rapid_mlx/audio/vad_artifacts.py",
        "rapid_mlx/model_catalog_lock.json",
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


@pytest.mark.parametrize("value", [None, [], 1, {"revision": PIN}])
def test_malformed_new_identity_fails_closed(value):
    assert catalog_changes({}, {"new": value}, PATH)[1]


@pytest.mark.parametrize(
    "assets",
    [
        None,
        [],
        {"family": None},
        {"family": [None]},
        {"family": [{"repo_id": "org/asset"}]},
    ],
)
def test_malformed_or_unpinned_runtime_assets_fail_closed(assets):
    assert catalog_changes({}, {"_runtime_assets": assets}, PATH)[1]


def test_runtime_asset_positive_and_grandfathered_identity():
    existing = {"repo_id": "org/old"}
    added = {"repo_id": "org/new", "revision": PIN}
    before = {"_runtime_assets": {"family": [existing]}}
    after = {"_runtime_assets": {"family": [existing, added]}}
    assert catalog_changes(before, after, PATH) == (["_runtime_assets"], [])


@pytest.mark.parametrize("previous", [{"repo_id": "org/model"}, "repo_id", None])
def test_malformed_prior_runtime_assets_cannot_grandfather_new_values(previous):
    before = {"_runtime_assets": {"family": previous}}
    after = {"_runtime_assets": {"family": ["repo_id"]}}
    assert catalog_changes(before, after, PATH)[1]
    after["_runtime_assets"]["family"] = [{"repo_id": "org/model"}]
    assert catalog_changes(before, after, PATH)[1]
    after["_runtime_assets"]["family"][0]["revision"] = PIN
    assert catalog_changes(before, after, PATH) == (["_runtime_assets"], [])


def test_edited_runtime_asset_list_revalidates_existing_asset_shapes():
    before = {"_runtime_assets": {"family": ["repo_id"]}}
    after = {
        "_runtime_assets": {
            "family": ["repo_id", {"repo_id": "org/new", "revision": PIN}]
        }
    }
    assert catalog_changes(before, after, PATH)[1]


def test_audio_and_removal_identity_rules():
    path = "rapid_mlx/audio/aliases.json"
    assert catalog_changes(
        {}, {"new": {"hf_id": "org/model", "revision": PIN}}, path
    ) == (["new"], [])
    assert catalog_changes({"old": "org/model"}, {}, PATH) == (["old"], [])
    assert catalog_changes({"old": "org/model"}, {"renamed": "org/model"}, PATH)[1]


def test_existing_catalog_is_unchanged_not_certified():
    from pathlib import Path

    from scripts.catalog_trust import CATALOGS

    root = Path(__file__).resolve().parents[1]
    for path in CATALOGS:
        data = parse_catalog((root / path).read_text())
        assert catalog_changes(data, data, path) == ([], [])


def test_nonobject_catalog_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        parse_catalog("[]")


def test_git_catalog_and_worktree_errors(tmp_path):
    import subprocess

    from scripts.catalog_trust import audit_worktree, read_git_catalog

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    target = tmp_path / PATH
    target.parent.mkdir()
    target.write_text('{"old": "org/model"}')
    git("add", ".")
    git("commit", "-qm", "fixture")
    base = git("rev-parse", "HEAD")
    assert read_git_catalog(tmp_path, base, PATH) == {"old": "org/model"}
    assert audit_worktree(tmp_path, base, [PATH]) == ([], [])
    target.unlink()
    assert audit_worktree(tmp_path, base, [PATH]) == ([PATH + ":old"], [])
    target.symlink_to("missing")
    assert "symlink" in audit_worktree(tmp_path, base, [PATH])[1][0]
    target.unlink()
    target.write_text("[]")
    assert audit_worktree(tmp_path, base, [PATH])[1]
    with pytest.raises(ValueError, match="exact"):
        read_git_catalog(tmp_path, "main", PATH)


def test_cli_report_and_comparison(tmp_path, monkeypatch, capsys):
    import runpy
    import sys
    from pathlib import Path

    import scripts.catalog_trust as policy

    target = tmp_path / PATH
    target.parent.mkdir()
    target.write_text(
        json.dumps(
            {"old": "org/model", "pinned": {"hf_path": "org/pinned", "revision": PIN}}
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["catalog_trust", "--report-unpinned"])
    assert policy.main() == 0
    assert json.loads(capsys.readouterr().out) == {"old": "org/model"}
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(Path(policy.__file__)), run_name="__main__")
    assert result.value.code == 0
    monkeypatch.setattr(sys, "argv", ["catalog_trust"])
    with pytest.raises(SystemExit):
        policy.main()
    monkeypatch.setattr(sys, "argv", ["catalog_trust", "--base", PIN])
    assert policy.main() == 1
