#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline mocked-response contracts for PF-3 environment read-back."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_release_environment.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_release_environment", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _env(
    *,
    reviewers=("raullenchai",),
    prevent_self_review=False,
    can_admins_bypass=False,
    name="rapid-mac-tag",
    deployment_mode=None,
):
    body = {
        "name": name,
        "protection_rules": [
            {
                "id": 1,
                "type": "required_reviewers",
                "prevent_self_review": prevent_self_review,
                "reviewers": [
                    {"type": "User", "id": 1000 + i, "reviewer": {"login": login}}
                    for i, login in enumerate(reviewers)
                ],
            }
        ],
        "deployment_branch_policy": (
            {"custom_branch_policies": True, "protected_branches": False}
            if deployment_mode is None
            else deployment_mode
        ),
        "can_admins_bypass": can_admins_bypass,
    }
    return body


def _policy(*, branches=(("main", "branch"),), total_count=None):
    return {
        "total_count": len(branches) if total_count is None else total_count,
        "branch_policies": [{"name": name, "type": ptype} for name, ptype in branches],
    }


def _write(tmp_path, obj, name):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_healthy_environment_reads_back(checker, tmp_path):
    env = _write(tmp_path, _env(), "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    evidence = checker.read_back(env_json=env, policy_json=pol)
    joined = "\n".join(evidence)
    assert "raullenchai" in joined
    assert "main" in joined
    assert "can_admins_bypass=false" in joined and "FORBIDDEN" not in joined


def test_wrong_reviewer_fails(checker, tmp_path):
    env = _write(tmp_path, _env(reviewers=("someone-else",)), "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="raullenchai"):
        checker.read_back(env_json=env, policy_json=pol)


def test_multiple_reviewers_fails(checker, tmp_path):
    env = _write(tmp_path, _env(reviewers=("raullenchai", "bot")), "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="EXACTLY one reviewer"):
        checker.read_back(env_json=env, policy_json=pol)


def test_extra_team_reviewer_fails(checker, tmp_path):
    # Expected User PLUS a Team reviewer must fail: reviewers must be exactly
    # one entry, no extra member riding along.
    reviewers = [
        {"type": "User", "id": 1000, "reviewer": {"login": "raullenchai"}},
        {"type": "Team", "id": 2000, "reviewer": {"login": "release-eng"}},
    ]
    env = {
        "name": "rapid-mac-tag",
        "protection_rules": [
            {
                "id": 1,
                "type": "required_reviewers",
                "prevent_self_review": False,
                "reviewers": reviewers,
            }
        ],
        "deployment_branch_policy": {
            "custom_branch_policies": True,
            "protected_branches": False,
        },
        "can_admins_bypass": True,
    }
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="EXACTLY one reviewer"):
        checker.read_back(env_json=env, policy_json=pol)


def test_malformed_reviewer_entry_fails(checker, tmp_path):
    # A sole reviewer that is not a User (e.g. type Team) must fail.
    reviewers = [{"type": "Team", "id": 2000, "reviewer": {"login": "release-eng"}}]
    env = {
        "name": "rapid-mac-tag",
        "protection_rules": [
            {
                "id": 1,
                "type": "required_reviewers",
                "prevent_self_review": False,
                "reviewers": reviewers,
            }
        ],
        "deployment_branch_policy": {
            "custom_branch_policies": True,
            "protected_branches": False,
        },
        "can_admins_bypass": True,
    }
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="must be a User"):
        checker.read_back(env_json=env, policy_json=pol)


def test_missing_deployment_mode_fails(checker, tmp_path):
    env = _env()
    del env["deployment_branch_policy"]
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="deployment_branch_policy"):
        checker.read_back(env_json=env, policy_json=pol)


def test_wrong_deployment_mode_fails(checker, tmp_path):
    # protected-branches mode active (not custom branch policies) is a NO-GO.
    env = _write(
        tmp_path,
        _env(
            deployment_mode={
                "custom_branch_policies": False,
                "protected_branches": True,
            }
        ),
        "env.json",
    )
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(
        checker.EnvironmentGateError, match="custom_branch_policies=true"
    ):
        checker.read_back(env_json=env, policy_json=pol)


def test_missing_can_admins_bypass_fails(checker, tmp_path):
    env = _env()
    del env["can_admins_bypass"]
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="can_admins_bypass"):
        checker.read_back(env_json=env, policy_json=pol)


def test_nonboolean_can_admins_bypass_fails(checker, tmp_path):
    env = _env(can_admins_bypass="yes")
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="can_admins_bypass"):
        checker.read_back(env_json=env, policy_json=pol)


def test_can_admins_bypass_false_is_required(checker, tmp_path):
    # can_admins_bypass must be exactly false: admin bypass disabled is the only
    # acceptable state (normal required-reviewer approval).
    env = _env(can_admins_bypass=False)
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    evidence = checker.read_back(env_json=env, policy_json=pol)
    joined = "\n".join(evidence)
    assert "can_admins_bypass=false" in joined
    assert "FORBIDDEN" not in joined


def test_can_admins_bypass_true_fails_closed(checker, tmp_path):
    # Admin bypass enabled means an admin could approve without the required
    # reviewer flow — the RC claim gate must fail closed, not merely warn.
    env = _env(can_admins_bypass=True)
    env = _write(tmp_path, env, "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="can_admins_bypass"):
        checker.read_back(env_json=env, policy_json=pol)


def test_no_required_reviewers_rule_fails(checker, tmp_path):
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"name": "rapid-mac-tag", "protection_rules": []}))
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="required_reviewers"):
        checker.read_back(env_json=env, policy_json=pol)


def test_prevent_self_review_true_fails(checker, tmp_path):
    env = _write(tmp_path, _env(prevent_self_review=True), "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="prevent_self_review"):
        checker.read_back(env_json=env, policy_json=pol)


def test_policy_total_count_not_one_fails(checker, tmp_path):
    env = _write(tmp_path, _env(), "env.json")
    pol = _write(
        tmp_path,
        _policy(branches=(("main", "branch"), ("dev", "branch"))),
        "policy.json",
    )
    with pytest.raises(checker.EnvironmentGateError, match="total_count"):
        checker.read_back(env_json=env, policy_json=pol)


def test_policy_wrong_branch_name_fails(checker, tmp_path):
    env = _write(tmp_path, _env(), "env.json")
    pol = _write(tmp_path, _policy(branches=(("trunk", "branch"),)), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="main"):
        checker.read_back(env_json=env, policy_json=pol)


def test_policy_wrong_type_fails(checker, tmp_path):
    env = _write(tmp_path, _env(), "env.json")
    pol = _write(tmp_path, _policy(branches=(("main", "tag"),)), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="must be exactly"):
        checker.read_back(env_json=env, policy_json=pol)


def test_missing_environment_json_fails(checker, tmp_path):
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="cannot read"):
        checker.read_back(env_json=tmp_path / "missing.json", policy_json=pol)


def test_wrong_env_name_fails(checker, tmp_path):
    env = _write(tmp_path, _env(name="production-tag"), "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="rapid-mac-tag"):
        checker.read_back(env_json=env, policy_json=pol)
