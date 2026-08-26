#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PF-3: read back and verify the production tag environment (rapid-mac-tag).

A workflow reference to an environment silently auto-creates an UNPROTECTED one,
so code comments or a YAML ``environment:`` line are not, by themselves, an
approval gate. This checker reads back the LIVE environment protection rules and
deployment-branch policy from the GitHub REST API and fails closed if the tag
claim would not be human-authorized:

  * the environment must exist with exactly one ``required_reviewers`` rule;
  * its reviewers must be EXACTLY one user, the expected reviewer login;
  * ``prevent_self_review`` must be false (a truthful owner-approval contract);
  * the deployment-branch policy must be exactly one branch policy ``{name:
    main, type: branch}`` (deployments may only be requested from main);
  * ``can_admins_bypass`` must be exactly false (GitHub supports disabling admin
    bypass); missing or true fails closed so a claim never rides an admin who
    could approve without a required-reviewer approval.

Inputs are the raw JSON responses (environment + deployment-branch-policies) read
from the REST API by the caller; this script only evaluates structured JSON, never
text or regex, so it is fully testable offline with mocked responses.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

EXPECTED_REVIEWER = "raullenchai"
EXPECTED_ENV_NAME = "rapid-mac-tag"
EXPECTED_BRANCH = "main"


class EnvironmentGateError(Exception):
    """Raised on any fail-closed environment condition."""


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentGateError(
            f"cannot read environment JSON {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise EnvironmentGateError(f"environment JSON {path} is not an object")
    return data


def read_back(
    *,
    env_json: Path,
    policy_json: Path,
    expected_reviewer: str = EXPECTED_REVIEWER,
    expected_env_name: str = EXPECTED_ENV_NAME,
    expected_branch: str = EXPECTED_BRANCH,
) -> list[str]:
    """Verify the environment; return evidence lines or raise EnvironmentGateError."""

    env = _load(env_json)
    if env.get("name") != expected_env_name:
        raise EnvironmentGateError(
            f"read-back environment name {env.get('name')!r} != expected {expected_env_name!r}"
        )

    evidence: list[str] = []

    # The reviewer contract: EXACTLY one User reviewer, whose login is the expected
    # reviewer. Any extra entry (User/Team/anything), a non-User type, or a
    # malformed reviewer record fails closed — an extra read-access user or a
    # Team reviewer would otherwise silently ride along.
    rules = [
        r
        for r in env.get("protection_rules", [])
        if isinstance(r, dict) and r.get("type") == "required_reviewers"
    ]
    if len(rules) != 1:
        raise EnvironmentGateError(
            f"environment must have exactly one required_reviewers rule, got {len(rules)}"
        )
    rule = rules[0]
    if rule.get("prevent_self_review") is not False:
        raise EnvironmentGateError(
            "prevent_self_review must be false for a truthful owner-approval "
            f"contract, got {rule.get('prevent_self_review')!r}"
        )
    reviewers = rule.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 1:
        raise EnvironmentGateError(
            f"required_reviewers must list EXACTLY one reviewer, got {reviewers!r}"
        )
    entry = reviewers[0]
    if not isinstance(entry, dict) or entry.get("type") != "User":
        raise EnvironmentGateError(
            f"the sole required reviewer must be a User, got {entry!r}"
        )
    reviewer = entry.get("reviewer")
    login = reviewer.get("login") if isinstance(reviewer, dict) else None
    if login != expected_reviewer:
        raise EnvironmentGateError(
            f"the sole required reviewer must be exactly '{expected_reviewer}', got {login!r}"
        )
    evidence.append(
        f"REQUIRED_REVIEWERS = [{expected_reviewer}] (one User, prevent_self_review=False)"
    )

    # Human-approval mode: deployment_branch_policy must structurally be
    # custom-branch-policies ACTIVE and protected-branches OFF. The plural
    # deployment-branch-policies endpoint alone proves a policy is listed, not
    # that this mode is what gates deployments — so require the mode field too.
    dbp = env.get("deployment_branch_policy")
    if not isinstance(dbp, dict):
        raise EnvironmentGateError(
            f"environment deployment_branch_policy must be an object, got {dbp!r}"
        )
    if (
        dbp.get("custom_branch_policies") is not True
        or dbp.get("protected_branches") is not False
    ):
        raise EnvironmentGateError(
            "deployment mode must be custom_branch_policies=true and "
            f"protected_branches=false, got {dbp!r}"
        )

    # Branch policy list: exactly one branch policy {name: main, type: branch}.
    policies = _load(policy_json)
    if policies.get("total_count") != 1:
        raise EnvironmentGateError(
            f"deployment-branch-policy total_count must be 1, got {policies.get('total_count')!r}"
        )
    branches = policies.get("branch_policies")
    if not isinstance(branches, list) or len(branches) != 1:
        raise EnvironmentGateError(
            f"deployment-branch-policy must list exactly one branch policy, got {branches!r}"
        )
    policy = branches[0]
    if (
        not isinstance(policy, dict)
        or policy.get("name") != expected_branch
        or policy.get("type") != "branch"
    ):
        raise EnvironmentGateError(
            f"deployment policy must be exactly {{name: {expected_branch!r}, type: branch}}, got {policy!r}"
        )
    evidence.append(
        f"deployment branch policy: exactly one branch policy {expected_branch!r} (type branch); "
        "custom_branch_policies=true, protected_branches=false"
    )

    # can_admins_bypass must be present and EXACTLY false. GitHub supports
    # disabling admin bypass for an environment, so an RC tag claim must require
    # the normal required-reviewer flow — an admin who could approve without a
    # review (or a drifted/missing field we cannot assert on) fails closed.
    # prevent_self_review=false is intentionally UNCHANGED: the sole owning
    # reviewer still needs the ordinary required-reviewer approval.
    admin_bypass = env.get("can_admins_bypass")
    if admin_bypass is not False:
        raise EnvironmentGateError(
            "can_admins_bypass must be exactly false (admin bypass must be "
            "disabled); got "
            f"{'missing' if admin_bypass is None else admin_bypass!r}"
        )
    evidence.append(
        "can_admins_bypass=false (admin bypass disabled; approval requires a reviewer)"
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--expected-reviewer", default=EXPECTED_REVIEWER)
    parser.add_argument("--expected-env-name", default=EXPECTED_ENV_NAME)
    parser.add_argument("--expected-branch", default=EXPECTED_BRANCH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = read_back(
            env_json=args.environment_json,
            policy_json=args.policy_json,
            expected_reviewer=args.expected_reviewer,
            expected_env_name=args.expected_env_name,
            expected_branch=args.expected_branch,
        )
    except EnvironmentGateError as exc:
        print(f"release environment: {exc}", file=sys.stderr)
        return 1
    print("\n".join(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
