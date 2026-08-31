# SPDX-License-Identifier: Apache-2.0
"""Behavioral Git-lineage tests for trusted Mergify mlx attestations."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "check_mergify_mlx_attestation",
    ROOT / "scripts/check_mergify_mlx_attestation.py",
)
attestation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = attestation
SPEC.loader.exec_module(attestation)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _candidate_graph(tmp_path: Path) -> dict[str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "CI Test")
    _git(repo, "config", "user.email", "ci@example.invalid")
    (repo / "pyproject.toml").write_text('[project]\ndependencies=["mlx-vlm==1"]\n')
    base = _commit(repo, "base")

    _git(repo, "switch", "-c", "source-a")
    (repo / "pyproject.toml").write_text('[project]\ndependencies=["mlx-vlm==2"]\n')
    source_a = _commit(repo, "change bound")
    _git(repo, "switch", "-c", "candidate", base)
    _git(repo, "merge", "--no-ff", "source-a", "-m", "Merge of #10")
    ahead = _git(repo, "rev-parse", "HEAD")

    _git(repo, "switch", "-c", "source-b")
    (repo / "README.md").write_text("current batch\n")
    source_b = _commit(repo, "docs")
    _git(repo, "switch", "candidate")
    _git(repo, "merge", "--no-ff", "source-b", "-m", "Merge of #20")
    candidate = _git(repo, "rev-parse", "HEAD")
    return {
        "repo": str(repo),
        "base": base,
        "ahead": ahead,
        "candidate": candidate,
        "source_a": source_a,
        "source_b": source_b,
    }


def test_real_queue_info_fixture_parses_supported_metadata():
    raw = (ROOT / "tests/fixtures/mergify-queue-info-batch.json").read_text()
    metadata = attestation.parse_queue_metadata(raw)
    assert metadata.checking_base_sha == "68817d9a40db187b53a1a0d76888c0283d03e116"
    assert metadata.pull_numbers == (2812, 2810, 2821, 2826)


def test_ahead_bound_change_is_verified_even_when_current_batch_is_docs(tmp_path):
    graph = _candidate_graph(tmp_path)
    metadata = attestation.parse_queue_metadata(
        json.dumps(
            {
                "checking_base_sha": graph["ahead"],
                "pull_requests": [{"number": 20, "scopes": []}],
            }
        )
    )
    integrations = attestation.collect_integrations(
        metadata,
        real_base_sha=graph["base"],
        candidate_sha=graph["candidate"],
        cwd=graph["repo"],
    )

    by_number = {item.number: item for item in integrations}
    assert set(by_number) == {10, 20}
    assert by_number[10].changes_pyproject is True
    assert by_number[10].changes_mlx_bounds is True
    assert by_number[10].exact_source_sha == graph["source_a"]
    assert by_number[20].changes_pyproject is False
    assert by_number[20].changes_mlx_bounds is False
    assert by_number[20].exact_source_sha == graph["source_b"]


def test_unrelated_pyproject_change_does_not_require_mlx_attestation(tmp_path):
    graph = _candidate_graph(tmp_path)
    repo = Path(graph["repo"])
    _git(repo, "switch", "-c", "source-c")
    (repo / "pyproject.toml").write_text(
        '[project]\ndependencies=["mlx-vlm==2", "httpx==1"]\n'
    )
    source_c = _commit(repo, "change unrelated dependency")
    _git(repo, "switch", "candidate")
    _git(repo, "merge", "--no-ff", "source-c", "-m", "Merge of #30")
    candidate = _git(repo, "rev-parse", "HEAD")
    metadata = attestation.parse_queue_metadata(
        json.dumps(
            {
                "checking_base_sha": graph["candidate"],
                "pull_requests": [{"number": 30, "scopes": []}],
            }
        )
    )

    integrations = attestation.collect_integrations(
        metadata,
        real_base_sha=graph["base"],
        candidate_sha=candidate,
        cwd=repo,
    )

    by_number = {item.number: item for item in integrations}
    assert by_number[30].exact_source_sha == source_c
    assert by_number[30].changes_pyproject is True
    assert by_number[30].changes_mlx_bounds is False


def test_unexpected_first_parent_commit_fails_closed(tmp_path):
    graph = _candidate_graph(tmp_path)
    repo = Path(graph["repo"])
    (repo / "pyproject.toml").write_text('[project]\ndependencies=["mlx-vlm==3"]\n')
    malformed = _commit(repo, "untrusted direct candidate edit")
    metadata = attestation.parse_queue_metadata(
        json.dumps(
            {
                "checking_base_sha": graph["ahead"],
                "pull_requests": [{"number": 20, "scopes": []}],
            }
        )
    )

    with pytest.raises(attestation.AttestationError, match="unexpected first-parent"):
        attestation.collect_integrations(
            metadata,
            real_base_sha=graph["base"],
            candidate_sha=malformed,
            cwd=repo,
        )


def _mock_source_pr(monkeypatch, *, changes: dict | None = None):
    source = json.loads(
        (ROOT / "tests/fixtures/github-pull-2792-attestation.json").read_text()
    )
    source.update(changes or {})
    payload = json.dumps(source).encode()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def urlopen(request, timeout):
        assert request.full_url.endswith("/pulls/2792")
        return Response(payload)

    monkeypatch.setattr(
        attestation.urllib.request,
        "urlopen",
        urlopen,
    )


def test_exact_source_sha_accepts_the_live_pull_request_attestation(monkeypatch):
    _mock_source_pr(monkeypatch)
    assert attestation.source_pr_attested(
        "raullenchai/Rapid-MLX",
        2792,
        "fd384c93e2a8182a0547e24f098e3063bd4c9b2e",
        "fixture-token",
    )


def test_check_for_a_different_sha_cannot_authorize_source(monkeypatch):
    _mock_source_pr(monkeypatch)
    assert not attestation.source_pr_attested(
        "raullenchai/Rapid-MLX",
        2792,
        "0000000000000000000000000000000000000000",
        "fixture-token",
    )


def test_source_from_a_fork_cannot_authorize_candidate(monkeypatch):
    _mock_source_pr(
        monkeypatch,
        changes={
            "head": {
                "sha": "fd384c93e2a8182a0547e24f098e3063bd4c9b2e",
                "repo": {"full_name": "someone/fork"},
            }
        },
    )
    assert not attestation.source_pr_attested(
        "raullenchai/Rapid-MLX",
        2792,
        "fd384c93e2a8182a0547e24f098e3063bd4c9b2e",
        "fixture-token",
    )


def test_missing_attestation_cannot_authorize_source(monkeypatch):
    _mock_source_pr(monkeypatch, changes={"body": "No proof.", "labels": []})
    assert not attestation.source_pr_attested(
        "raullenchai/Rapid-MLX",
        2792,
        "fd384c93e2a8182a0547e24f098e3063bd4c9b2e",
        "fixture-token",
    )
