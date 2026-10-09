#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline mocked-response contracts for PF-3 environment read-back."""

from __future__ import annotations

import json

import pytest

from scripts import check_release_environment


@pytest.fixture(scope="module")
def checker():
    return check_release_environment


def test_parser_accepts_explicit_branch_allowlist(checker):
    args = checker._parser().parse_args(
        [
            "--environment-json",
            "env.json",
            "--policy-json",
            "policies.json",
            "--expected-branch",
            "main",
            "--expected-branch",
            "release/0.16.0",
        ]
    )
    assert args.expected_branches == ["main", "release/0.16.0"]


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


def test_main_and_frozen_release_branches_pass(checker, tmp_path):
    env = _write(tmp_path, _env(), "env.json")
    pol = _write(
        tmp_path,
        _policy(branches=(("main", "branch"), ("release/0.16.0", "branch"))),
        "policy.json",
    )
    evidence = checker.read_back(
        env_json=env, policy_json=pol, expected_branches=("main", "release/0.16.0")
    )
    assert "release/0.16.0" in "\n".join(evidence)


def test_frozen_policy_missing_fails(checker, tmp_path):
    env = _write(tmp_path, _env(), "env.json")
    pol = _write(tmp_path, _policy(), "policy.json")
    with pytest.raises(checker.EnvironmentGateError, match="total_count"):
        checker.read_back(
            env_json=env, policy_json=pol, expected_branches=("main", "release/0.16.0")
        )


def _run_workflow_environment_check(
    tmp_path, workflow, branch, *, environment=None, policies=None
):
    """Execute the actual caller's rendered Bash plus real structured checker."""
    import os
    import shlex
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    root = Path(__file__).parents[1]
    document = yaml.safe_load((root / ".github/workflows" / workflow).read_text())
    snippets = [
        step["run"]
        for job in document["jobs"].values()
        for step in job.get("steps", [])
        if "EXPECTED_BRANCH_ARGS=" in step.get("run", "")
    ]
    assert len(snippets) == 1
    command = snippets[0][snippets[0].index("EXPECTED_BRANCH_ARGS=") :]
    end = (
        'echo "PF3_OK=1"'
        if workflow == "release-preflight.yml"
        else 'echo "::notice::live rapid-mac-tag'
    )
    if workflow != "post-dmg-engine-recovery.yml":
        command = command[: command.index(end)]
    # Only interpreter selection and evidence output location differ from CI.
    command = command.replace(
        "python3 scripts/", shlex.quote(sys.executable) + " scripts/"
    )
    command = command.replace(
        "/tmp/release-env-evidence.txt", '"$RUNNER_TEMP/release-env-evidence.txt"'
    )
    environment = _env() if environment is None else environment
    policies = (
        _policy(branches=(("main", "branch"), ("release/0.16.0", "branch")))
        if policies is None
        else policies
    )
    for name in ("pf3-env.json", "re-env.json"):
        _write(tmp_path, environment, name)
    for name in ("pf3-policy.json", "re-policy.json"):
        _write(tmp_path, policies, name)
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", command],
        cwd=root,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "ENV_NAME": "rapid-mac-tag",
            "SOURCE_BRANCH": branch,
            "TARGET_BRANCH": branch,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    "workflow",
    ["auto-release.yml", "release-preflight.yml", "post-dmg-engine-recovery.yml"],
)
@pytest.mark.parametrize("branch", ["main", "release/0.16.0"])
def test_workflow_environment_inventory_is_not_selected_by_active_ref(
    tmp_path, workflow, branch
):
    result = _run_workflow_environment_check(tmp_path, workflow, branch)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "['main', 'release/0.16.0']" in result.stdout


@pytest.mark.parametrize(
    "workflow",
    ["auto-release.yml", "release-preflight.yml", "post-dmg-engine-recovery.yml"],
)
@pytest.mark.parametrize(
    "branches",
    [
        (
            ("main", "branch"),
            ("release/0.16.0", "branch"),
            ("release/unreviewed", "branch"),
        ),
        (("main", "branch"), ("release/*", "branch")),
        (("main", "branch"), ("release/0.16.0", "tag")),
    ],
)
def test_rendered_workflow_still_rejects_policy_drift(tmp_path, workflow, branches):
    result = _run_workflow_environment_check(
        tmp_path, workflow, "main", policies=_policy(branches=branches)
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "workflow",
    ["auto-release.yml", "release-preflight.yml", "post-dmg-engine-recovery.yml"],
)
@pytest.mark.parametrize(
    "environment", [_env(can_admins_bypass=True), _env(reviewers=("bot",))]
)
def test_rendered_workflow_still_requires_existing_reviewer_rules(
    tmp_path, workflow, environment
):
    result = _run_workflow_environment_check(
        tmp_path, workflow, "main", environment=environment
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "workflow",
    ["auto-release.yml", "release-preflight.yml", "post-dmg-engine-recovery.yml"],
)
def test_main_workflow_remains_usable_after_frozen_policy_cleanup(tmp_path, workflow):
    result = _run_workflow_environment_check(
        tmp_path, workflow, "main", policies=_policy()
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exactly ['main']" in result.stdout


@pytest.mark.parametrize("workflow", ["auto-release.yml", "release-preflight.yml"])
def test_frozen_workflow_requires_its_policy_even_with_main_compatibility(
    tmp_path, workflow
):
    result = _run_workflow_environment_check(
        tmp_path, workflow, "release/0.16.0", policies=_policy()
    )
    assert result.returncode != 0


def test_retained_policy_flag_cannot_widen_another_expected_inventory(
    checker, tmp_path
):
    with pytest.raises(checker.EnvironmentGateError, match="requires main only"):
        checker.read_back(
            env_json=_write(tmp_path, _env(), "env.json"),
            policy_json=_write(tmp_path, _policy(), "policy.json"),
            expected_branches=("other",),
            allow_retained_0160_policy=True,
        )


@pytest.mark.parametrize(
    "branches,expected",
    [
        ((("main", "branch"),), 0),
        ((("main", "branch"), ("release/0.16.0", "branch")), 0),
        ((("main", "branch"), ("release/unapproved", "branch")), 1),
        ((("release/0.16.0", "branch"),), 1),
    ],
)
def test_cli_retained_policy_compatibility_is_bounded(
    checker, tmp_path, capsys, branches, expected
):
    env = _write(tmp_path, _env(), "env.json")
    policy = _write(tmp_path, _policy(branches=branches), "policy.json")
    assert (
        checker.main(
            [
                "--environment-json",
                str(env),
                "--policy-json",
                str(policy),
                "--allow-retained-0160-policy",
            ]
        )
        == expected
    )
    captured = capsys.readouterr()
    assert ("can_admins_bypass=false" in captured.out) == (expected == 0)
