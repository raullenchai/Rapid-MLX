from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest
import yaml

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


@pytest.mark.parametrize(
    "changed_path",
    [".github/workflows/auto-release.yml", "scripts/upload_release_r2.py"],
)
def test_frozen_bump_rejects_release_authority_change(monkeypatch, changed_path):
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
            return changed_path + "\n"
        if f"{B}..{A}" in joined:
            return changed_path + "\n"
        if "rev-list" in joined:
            return A + "\n"
        if "diff-tree" in joined:
            return changed_path + "\n"
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


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/rapid-mac-release.yml",
        "scripts/upload_release_r2.py",
        "tests/test_upload_release_r2.py",
        "apps/rapid-mac/Tests/RapidTests/ReleaseManifestWorkflowTests.swift",
    ],
)
def test_exact_multipart_repair_paths_are_valid_policy_history(monkeypatch, path):
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *_a, **_k: type("R", (), {"returncode": 0})()
    )

    def output(command, **kwargs):
        joined = " ".join(command)
        if "--min-parents=2" in joined:
            return ""
        if "diff --name-only" in joined or "diff-tree" in joined:
            return path + "\n"
        if "rev-list" in joined:
            return A + "\n"
        raise AssertionError(command)

    monkeypatch.setattr(mod.subprocess, "check_output", output)
    assert mod.check_source(
        source_ref=mod.FROZEN_REF,
        live_sha=A,
        accepted_sha=A,
        release_sha=A,
        version=mod.FROZEN_VERSION,
    )


def test_frozen_policy_history_passes_in_hermetic_git_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Release Test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "release@example.invalid"],
        check=True,
    )
    (repo / "README.md").write_text("frozen product\n")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "frozen product"], check=True
    )
    frozen = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    policy = repo / "docs" / "development" / "releasing.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("frozen release policy\n")
    subprocess.run(
        ["git", "-C", str(repo), "add", str(policy.relative_to(repo))], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "release policy"], check=True
    )
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(mod, "FROZEN_PRODUCT_SHA", frozen)
    evidence = mod.check_source(
        source_ref=mod.FROZEN_REF,
        live_sha=head,
        accepted_sha=head,
        release_sha=head,
        version=mod.FROZEN_VERSION,
        repo=str(repo),
    )
    assert frozen in "\n".join(evidence)

    # Reproduce the hosted depth-one checkout: the frozen ancestor is absent.
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth=1", repo.as_uri(), str(shallow)], check=True
    )
    with pytest.raises(mod.ReleaseSourceError, match="not an ancestor"):
        mod.check_source(
            source_ref=mod.FROZEN_REF,
            live_sha=head,
            accepted_sha=head,
            release_sha=head,
            version=mod.FROZEN_VERSION,
            repo=str(shallow),
        )
    workflow = yaml.safe_load((ROOT / ".github/workflows/auto-release.yml").read_text())
    checkout = workflow["jobs"]["detect"]["steps"][0]
    depth = checkout.get("with", {}).get("fetch-depth", 1)
    configured = tmp_path / "configured"
    args = ["git", "clone", "-q"]
    if depth:
        args.append(f"--depth={depth}")
    subprocess.run([*args, repo.as_uri(), str(configured)], check=True)
    assert mod.check_source(
        source_ref=mod.FROZEN_REF,
        live_sha=head,
        accepted_sha=head,
        release_sha=head,
        version=mod.FROZEN_VERSION,
        repo=str(configured),
    )


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
