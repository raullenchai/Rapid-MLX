# SPDX-License-Identifier: Apache-2.0
"""DeepSeek Harness is a Tier-1 agent — pin the wiring that makes that true.

"Tier-1" in this repo is not a label in a README, it is two concrete
mechanisms:

1. ``tests/integrations/agent_smoke.sh`` drives the REAL ``dsh`` binary
   through a real bug-fix task on every release; the release job ``needs``
   that gate, so a regression cannot tag or publish.
2. ``bench --tier harness`` / release_check_m3.sh G7b smoke the
   chat-completions wire for every first-class harness.

Both are shell/CI surfaces that no unit test would otherwise touch, so the
Tier-1 claim could rot silently — the profile would keep loading, the docs
would keep saying "Tier-1", and nothing would fail. These tests assert the
wiring itself.

Each test below is written so that REMOVING the thing it guards makes it
red, which is the only property that makes a guard worth having.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "tests" / "integrations" / "agent_smoke.sh"


@pytest.fixture(scope="module")
def smoke_src() -> str:
    return SMOKE.read_text(encoding="utf-8")


def test_dsh_is_a_first_class_harness_profile() -> None:
    """``bench --tier harness`` must actually sweep deepseek-harness."""
    from vllm_mlx.bench.tier_runner import HARNESS_PROFILES

    assert "deepseek-harness" in HARNESS_PROFILES


def test_harness_profiles_mirror_in_release_check_is_in_lockstep() -> None:
    """release_check_m3_random.py hardcodes a mirror of HARNESS_PROFILES.

    It cannot import the real one (importing the package pulls mlx_lm at
    module load), so the mirror is copy-paste by design — and copy-paste
    is exactly what drifts. Compare the two directly.
    """
    from vllm_mlx.bench.tier_runner import HARNESS_PROFILES

    mirror_src = (REPO_ROOT / "scripts" / "release_check_m3_random.py").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r"^HARNESS_PROFILES = (\([^)]*\))", mirror_src, re.MULTILINE | re.DOTALL
    )
    assert match, "release_check_m3_random.py no longer defines HARNESS_PROFILES"
    # literal_eval, not exec: this only ever needs to read a tuple of string
    # literals, and it should stay unable to do anything else even if that
    # file grows something executable next to the assignment.
    mirrored = ast.literal_eval(match.group(1))
    assert mirrored == HARNESS_PROFILES


def test_bench_submission_schema_accepts_every_harness_profile() -> None:
    """``harness_result`` uses ``additionalProperties: false``.

    A profile added to HARNESS_PROFILES but not to the schema makes every
    ``--tier harness --submit`` run fail validation at submission time,
    which is a long way from the change that caused it.
    """
    import json

    from vllm_mlx.bench.tier_runner import HARNESS_PROFILES

    schema = json.loads(
        (REPO_ROOT / "community-benchmarks" / "schema.json").read_text(encoding="utf-8")
    )
    allowed = set(schema["properties"]["harness_result"]["properties"])
    missing = set(HARNESS_PROFILES) - allowed
    assert not missing, (
        f"harness_result schema rejects {sorted(missing)}; add them to "
        f"community-benchmarks/schema.json properties (required stays as-is — "
        f"see that field's description for the back-compat reason)"
    )


def test_release_gate_runs_dsh_and_counts_it(smoke_src: str) -> None:
    """The gate must define, invoke, and grade dsh — all three."""
    assert "run_dsh()" in smoke_src, "no run_dsh function"
    assert re.search(r"^run_dsh$", smoke_src, re.MULTILINE), (
        "run_dsh is defined but never invoked — a Tier-1 agent that never runs"
    )
    assert '"$R_DSH"' in smoke_src, "dsh result is never graded in the pass loop"


def test_release_gate_grades_dsh_in_the_blocking_loop(smoke_src: str) -> None:
    """dsh's result must be in the loop that exits non-zero, not just printed.

    Printing a FAIL and still exiting 0 is the failure mode that makes a
    gate decorative.
    """
    loop = re.search(r"^for r in (.+); do$", smoke_src, re.MULTILINE)
    assert loop, "the Tier-1 pass/fail loop is gone"
    assert "$R_DSH" in loop.group(1), (
        f"dsh missing from the blocking loop: {loop.group(1)}"
    )


def test_release_gate_redirects_and_guards_dsh_home(smoke_src: str) -> None:
    """DSH_HOME must be redirected AND validated like CODEX_HOME/HERMES_HOME.

    ``agents dsh --setup`` writes a credential file, so an un-redirected or
    blank DSH_HOME does more damage than a stray provider block.
    """
    assert "export DSH_HOME=" in smoke_src
    guard = re.search(r"^for _home_var in (.+); do$", smoke_src, re.MULTILINE)
    assert guard, "the throwaway-home guard loop is gone"
    assert "DSH_HOME" in guard.group(1), (
        f"DSH_HOME skips the blank/real-config guard: {guard.group(1)}"
    )
    assert re.search(r"DSH_HOME\)\s+_home_real=", smoke_src), (
        "DSH_HOME has no ~/.dsh case in the real-config refusal, so the guard "
        "silently falls through to the empty default and never fires"
    )


def test_release_gate_fingerprints_the_real_dsh_credential_store(
    smoke_src: str,
) -> None:
    """Both files ``--setup`` can write must be fingerprinted, not just one."""
    fingerprint = re.search(r"_real_fingerprint\(\) \{.*?\n\}", smoke_src, re.DOTALL)
    assert fingerprint, "_real_fingerprint is gone"
    body = fingerprint.group(0)
    assert ".dsh/settings.yaml" in body
    assert ".dsh/.credentials.yaml" in body, (
        "the credential file --setup writes is not fingerprinted, so a "
        "redirect failure could rewrite it unnoticed"
    )


def test_release_gate_does_not_trust_dsh_exit_code(smoke_src: str) -> None:
    """dsh exits 0 on hard failure, so grading must come from ``verify``.

    Measured on 0.1.0-rc.7: a settings.yaml naming an unregistered provider
    prints ``NO_ADAPTER: ...`` and still returns 0. If someone "tidies" the
    runner into an exit-code check, every dsh run passes forever.
    """
    run_dsh = re.search(r"run_dsh\(\) \{.*?\n\}", smoke_src, re.DOTALL)
    assert run_dsh, "run_dsh is gone"
    assert '[ "$(verify dsh)" = PASS ]' in run_dsh.group(0), (
        "run_dsh no longer grades on verify(); dsh's exit status is not a "
        "pass signal (it returns 0 even when it fails outright)"
    )


def test_dsh_profile_warns_that_exit_code_is_not_a_pass_signal() -> None:
    """The trap above must be documented where an integrator will hit it."""
    from vllm_mlx.agents import get_profile, load_profiles

    load_profiles()
    profile = get_profile("dsh")
    assert profile is not None
    assert any("exits 0" in issue for issue in profile.known_issues), (
        "the exit-code trap is not in known_issues"
    )
