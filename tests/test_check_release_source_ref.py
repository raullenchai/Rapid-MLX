from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

from scripts import check_release_source_ref as mod

ROOT = Path(__file__).resolve().parent.parent
A = "a" * 40
B = "b" * 40


def test_main_compatibility():
    assert (
        "main"
        in mod.check_source(
            source_ref="refs/heads/main",
            live_sha=A,
            accepted_sha=A,
            release_sha=A,
            version="9.9.9",
        )[0]
    )


@pytest.mark.parametrize(
    "ref,version",
    [
        ("refs/heads/release/0.16.1", "0.16.0"),
        (mod.FROZEN_REF, "0.16.1"),
        ("refs/tags/v0.16.0", "0.16.0"),
    ],
)
def test_frozen_wrong_ref_or_version_fails(ref, version):
    with pytest.raises(mod.ReleaseSourceError):
        mod.check_source(
            source_ref=ref, live_sha=A, accepted_sha=A, release_sha=A, version=version
        )


def test_stale_head_fails_before_git():
    with pytest.raises(mod.ReleaseSourceError, match="no longer"):
        mod.check_source(
            source_ref=mod.FROZEN_REF,
            live_sha=A,
            accepted_sha="b" * 40,
            release_sha=A,
            version=mod.FROZEN_VERSION,
        )


def test_malformed_sha_and_cli_result(capsys):
    assert (
        mod.main(
            [
                "--source-ref",
                "refs/heads/main",
                "--live-sha",
                A,
                "--accepted-sha",
                A,
                "--release-sha",
                A,
                "--version",
                "1.2.3",
            ]
        )
        == 0
    )
    assert "accepted == release" in capsys.readouterr().out
    assert (
        mod.main(
            [
                "--source-ref",
                "refs/heads/main",
                "--live-sha",
                "SHORT",
                "--accepted-sha",
                A,
                "--release-sha",
                A,
                "--version",
                "1.2.3",
            ]
        )
        == 1
    )
    assert "lowercase full SHA" in capsys.readouterr().err


@pytest.mark.parametrize(
    "stage,match",
    [
        ("ancestor", "not an ancestor"),
        ("merge", "linear"),
        ("aggregate", "non-policy"),
        ("commit", "touched disallowed"),
    ],
)
def test_frozen_history_fail_closed(monkeypatch, stage, match):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "R", (), {"returncode": int(stage == "ancestor")}
        )(),
    )

    def output(command, **kwargs):
        joined = " ".join(command)
        if "--min-parents=2" in joined:
            return A + "\n" if stage == "merge" else ""
        if "diff --name-only" in joined:
            return (
                "rapid_mlx/server.py\n" if stage == "aggregate" else "pyproject.toml\n"
            )
        if "rev-list" in joined:
            return A + "\n"
        if "diff-tree" in joined:
            return "rapid_mlx/server.py\n" if stage == "commit" else "pyproject.toml\n"
        raise AssertionError(command)

    monkeypatch.setattr(mod.subprocess, "check_output", output)
    with pytest.raises(mod.ReleaseSourceError, match=match):
        mod.check_source(
            source_ref=mod.FROZEN_REF,
            live_sha=A,
            accepted_sha=A,
            release_sha=A,
            version=mod.FROZEN_VERSION,
        )


def test_module_entrypoint_rejects_bad_sha(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            str(ROOT / "scripts/check_release_source_ref.py"),
            "--source-ref",
            "refs/heads/main",
            "--live-sha",
            "bad",
            "--accepted-sha",
            A,
            "--release-sha",
            A,
            "--version",
            "1.2.3",
        ],
    )
    with pytest.raises(SystemExit, match="1"):
        runpy.run_path(
            str(ROOT / "scripts/check_release_source_ref.py"), run_name="__main__"
        )


def test_frozen_bump_rejects_release_authority_change(monkeypatch):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )

    def output(command, **kwargs):
        joined = " ".join(command)
        if "--min-parents=2" in joined:
            return ""
        if f"{mod.FROZEN_PRODUCT_SHA}..{A}" in joined and "diff --name-only" in joined:
            return ".github/workflows/auto-release.yml\n"
        if f"{B}..{A}" in joined:
            return ".github/workflows/auto-release.yml\n"
        if "rev-list" in joined:
            return A + "\n"
        if "diff-tree" in joined:
            return ".github/workflows/auto-release.yml\n"
        raise AssertionError(command)

    monkeypatch.setattr(mod.subprocess, "check_output", output)
    with pytest.raises(mod.ReleaseSourceError, match="metadata"):
        mod.check_source(
            source_ref=mod.FROZEN_REF,
            live_sha=A,
            accepted_sha=A,
            release_sha=A,
            version=mod.FROZEN_VERSION,
            bump_base_sha=B,
        )


def test_frozen_current_policy_tree_passes():
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    evidence = mod.check_source(
        source_ref=mod.FROZEN_REF,
        live_sha=head,
        accepted_sha=head,
        release_sha=head,
        version=mod.FROZEN_VERSION,
        repo=str(ROOT),
    )
    assert mod.FROZEN_PRODUCT_SHA in "\n".join(evidence)


def test_workflows_pin_only_exact_frozen_route():
    auto = (ROOT / ".github/workflows/auto-release.yml").read_text()
    pre = (ROOT / ".github/workflows/release-preflight.yml").read_text()
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    desktop = (ROOT / ".github/workflows/rapid-mac-ci.yml").read_text()
    assert "refs/heads/release/0.16.0" in auto
    assert 'FROZEN_RELEASE_VERSION" != "0.16.0"' in auto
    assert auto.count("check_release_source_ref.py") >= 3
    assert "EXPECTED_BRANCH_ARGS+=(--expected-branch release/0.16.0)" in auto
    assert 'TARGET_BRANCH" != "release/0.16.0"' in pre
    assert 'PR_COMMITS" != "1"' in pre
    assert "git/ref/heads/release/0.16.0" in pre
    assert '--bump-base-sha "$BASE_SHA"' in pre
    assert "check_release_source_ref.py" in pre
    assert 'TARGET_BRANCH" = "release/0.16.0"' in pre
    assert "--cov=scripts.check_release_source_ref" in ci
    assert "--cov=scripts.check_release_environment" in ci
    assert "branches: [main, release/0.16.0]" in ci
    assert "branches: [main, release/0.16.0]" in desktop
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in ci
    assert "needs.changes.outputs.reuse_ci != 'true'" in ci
    assert "  tests:" in ci
    assert "  desktop-tests:" in desktop
