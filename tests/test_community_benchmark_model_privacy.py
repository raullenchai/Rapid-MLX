# SPDX-License-Identifier: Apache-2.0
"""A benchmark submission may name only anonymously public model repos."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.community_bench import atomic_upload, cli
from rapid_mlx.community_bench.publication import (
    PublicationRefused,
    ensure_public_model_sources,
)
from rapid_mlx.community_bench.workspace import LocalRunArchive
from rapid_mlx.telemetry import model_id


def _run(repo="privacy-test/model"):
    run = json.loads(
        (Path(__file__).parent / "fixtures/community_bench_cached_run.json").read_text()
    )
    run["model"]["components"][0]["source"]["repo_id"] = repo
    return run


@pytest.fixture(autouse=True)
def isolated_proof(monkeypatch):
    model_id._reset_for_tests()
    monkeypatch.setattr(model_id, "_marker_path", lambda repo: None)
    monkeypatch.setattr(model_id, "_hub_connection_can_prove_public", lambda: True)
    monkeypatch.setattr(model_id, "hf_auth_state", lambda: False)
    yield
    model_id._reset_for_tests()


@pytest.fixture
def metadata(monkeypatch):
    call = Mock(return_value=SimpleNamespace(private=False, gated=False))
    monkeypatch.setattr("huggingface_hub.model_info", call)
    return call


@pytest.mark.parametrize(
    "fields",
    [
        {"private": True, "gated": False},
        {"private": False, "gated": "auto"},
        {"private": False, "gated": "manual"},
        {"private": False},
    ],
)
def test_private_gated_and_unknown_metadata_refused(fields, metadata, monkeypatch):
    metadata.return_value = SimpleNamespace(**fields)
    post = Mock()
    monkeypatch.setattr(atomic_upload, "post_submission", post)
    with pytest.raises(PublicationRefused, match="private, gated, or unproven"):
        atomic_upload.upload_run(_run(), assume_yes=True)
    post.assert_not_called()
    assert not model_id.is_proven_public("privacy-test/model")


def test_public_repo_is_rechecked_even_with_a_telemetry_proof(metadata):
    model_id.note_hub_fetch("privacy-test/model")
    preview = atomic_upload.preview_run(_run(), install_id="0123456789ab")
    assert (
        preview["payload"]["model"]["components"][0]["source"]["repo_id"]
        == "privacy-test/model"
    )
    metadata.assert_called_once_with("privacy-test/model", token=False, timeout=5)


def test_catalog_repo_allowed_offline(metadata):
    atomic_upload.preview_run(
        _run("mlx-community/LFM2.5-1.2B-Instruct-4bit"), install_id="0123456789ab"
    )
    metadata.assert_not_called()


def test_public_probe_is_anonymous_even_with_token(metadata, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-user-token")
    monkeypatch.setattr(model_id, "hf_auth_state", lambda: True)
    preview = atomic_upload.preview_run(_run(), install_id="0123456789ab")
    metadata.assert_called_once_with("privacy-test/model", token=False, timeout=5)
    assert (
        preview["payload"]["model"]["components"][0]["source"]["repo_id"]
        == "privacy-test/model"
    )
    # Ambient auth must not mint a persistent telemetry proof.
    assert not model_id.is_proven_public("privacy-test/model")


def test_network_failure_refused(metadata):
    metadata.side_effect = TimeoutError("a URL or credential must not be echoed")
    with pytest.raises(
        PublicationRefused, match="anonymous public access could not be verified"
    ) as error:
        atomic_upload.preview_run(_run())
    assert "credential" not in str(error.value)


def test_successful_probe_records_proof(metadata):
    atomic_upload.preview_run(_run(), install_id="0123456789ab")
    assert metadata.call_count == 1
    assert model_id.is_proven_public("privacy-test/model")


def test_telemetry_proof_does_not_bypass_gated_check(metadata, monkeypatch):
    # The telemetry store proves "anonymously readable", which a gated repo
    # satisfies; publication must still read private/gated afresh.
    import time

    monkeypatch.setitem(model_id._proven_public, "privacy-test/model", time.time())
    metadata.return_value = SimpleNamespace(private=False, gated="auto")
    with pytest.raises(PublicationRefused, match="gated"):
        atomic_upload.preview_run(_run())
    metadata.assert_called_once()
    assert not model_id.is_proven_public("privacy-test/model")


def test_slow_metadata_response_hits_wall_clock_deadline(metadata, monkeypatch):
    import threading

    from rapid_mlx.community_bench import publication

    release = threading.Event()
    metadata.side_effect = lambda *a, **k: release.wait(5)
    monkeypatch.setattr(publication, "_PUBLIC_PROOF_DEADLINE_S", 0.05)
    try:
        with pytest.raises(PublicationRefused, match="could not be verified"):
            atomic_upload.preview_run(_run())
    finally:
        release.set()


def test_expired_proof_cannot_authorize_publication(metadata, monkeypatch):
    monkeypatch.setitem(model_id._proven_public, "privacy-test/model", 1)
    metadata.side_effect = TimeoutError()
    with pytest.raises(PublicationRefused):
        atomic_upload.preview_run(_run())
    metadata.assert_called_once()


def test_drafter_component_checked_too(metadata):
    run = _run("mlx-community/LFM2.5-1.2B-Instruct-4bit")
    drafter = copy.deepcopy(run["model"]["components"][0])
    drafter["component_id"] = "z-draft"
    drafter["role"] = "other"
    drafter["source"]["repo_id"] = "private-org/draft"
    run["model"]["components"].append(drafter)
    metadata.return_value = SimpleNamespace(private=False, gated="auto")
    with pytest.raises(PublicationRefused, match=r"components\[1\]"):
        atomic_upload.preview_run(run)
    metadata.assert_called_once_with("private-org/draft", token=False, timeout=5)


@pytest.mark.parametrize("repo", ["/private/models/secret", "./secret", "~/secret"])
def test_local_paths_never_published(repo, metadata):
    with pytest.raises(PublicationRefused, match="local-path"):
        ensure_public_model_sources(_run(repo))
    metadata.assert_not_called()


def test_existing_relative_directory_cannot_masquerade_as_public_repo(
    tmp_path, monkeypatch, metadata
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mlx-community" / "LFM2.5-1.2B-Instruct-4bit").mkdir(parents=True)
    with pytest.raises(PublicationRefused, match="local-path"):
        ensure_public_model_sources(_run("mlx-community/LFM2.5-1.2B-Instruct-4bit"))
    metadata.assert_not_called()


def test_untrusted_hub_endpoint_refused(metadata, monkeypatch):
    monkeypatch.setattr(model_id, "_hub_connection_can_prove_public", lambda: False)
    with pytest.raises(PublicationRefused, match="anonymous public access"):
        atomic_upload.preview_run(_run())
    metadata.assert_not_called()


@pytest.mark.parametrize(
    "preview,json_output", [(True, True), (False, True), (True, False)]
)
def test_cli_exposes_refusal_to_consent_preview_and_desktop(
    tmp_path, monkeypatch, capsys, metadata, preview, json_output
):
    run = _run()
    archive = LocalRunArchive(tmp_path)
    archive.save(run)
    monkeypatch.setattr(
        cli.LocalRunArchive, "default", classmethod(lambda cls: archive)
    )
    metadata.return_value = SimpleNamespace(private=True, gated=False)
    args = SimpleNamespace(
        benchmark_action="share",
        run_id=run["run_id"],
        json=json_output,
        yes=True,
        preview=preview,
    )
    assert cli.benchmark_command(args) != 0
    captured = capsys.readouterr()
    assert not captured.out
    if json_output:
        document = json.loads(captured.err)
        assert document["refused"] is True
        assert "private, gated, or unproven" in document["error"]
        assert "payload" not in document
    else:
        assert "private, gated, or unproven" in captured.err


def test_real_hub_client_never_sends_hf_token(monkeypatch):
    import httpx
    from huggingface_hub import hf_api

    monkeypatch.setenv("HF_TOKEN", "secret-user-token")
    monkeypatch.setattr(model_id, "hf_auth_state", lambda: True)
    seen = []

    def respond(request):
        seen.append(request)
        assert "authorization" not in request.headers
        assert "secret-user-token" not in str(request.headers)
        return httpx.Response(
            200, json={"id": "privacy-test/model", "private": False, "gated": False}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(hf_api, "get_session", lambda: client)
        atomic_upload.preview_run(_run(), install_id="0123456789ab")
    assert len(seen) == 1
    assert str(seen[0].url) == "https://huggingface.co/api/models/privacy-test/model"


def test_local_source_refused_before_consent_or_upload(metadata, monkeypatch):
    run = _run()
    run["model"]["components"][0]["source"] = {"kind": "local"}
    consent = Mock()
    post = Mock()
    monkeypatch.setattr(atomic_upload, "_ask_consent", consent)
    monkeypatch.setattr(atomic_upload, "post_submission", post)
    with pytest.raises(PublicationRefused, match="local-path"):
        atomic_upload.upload_run(run)
    consent.assert_not_called()
    post.assert_not_called()
    metadata.assert_not_called()


def test_repo_id_outside_hub_rules_is_refused(metadata):
    # Schema-shaped (org/name) but longer than the Hub allows, so it cannot
    # be proven public and is refused before any network call.
    from rapid_mlx.community_bench.publication import ensure_public_model_sources

    with pytest.raises(PublicationRefused, match="no public Hugging Face"):
        ensure_public_model_sources(_run("org/" + "a" * 120))
    metadata.assert_not_called()


def test_repeated_component_repo_is_checked_once(metadata):
    run = _run()
    twin = copy.deepcopy(run["model"]["components"][0])
    twin["component_id"] = "z-twin"
    twin["role"] = "other"
    run["model"]["components"].append(twin)
    from rapid_mlx.community_bench.publication import ensure_public_model_sources

    ensure_public_model_sources(run)
    metadata.assert_called_once_with("privacy-test/model", token=False, timeout=5)


def _paired_run(repo, depth):
    run = _run(repo)
    for _ in range(depth):
        outer = _run("mlx-community/LFM2.5-1.2B-Instruct-4bit")
        outer["paired_runs"] = [run]
        run = outer
    return run


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(
    "fields",
    [
        {"private": True, "gated": False},
        {"private": False, "gated": "auto"},
        {"private": False, "gated": "manual"},
    ],
)
def test_paired_private_or_gated_component_refused(depth, fields, metadata):
    paired = _run("mlx-community/LFM2.5-1.2B-Instruct-4bit")
    drafter = copy.deepcopy(paired["model"]["components"][0])
    drafter.update(component_id="z-draft", role="other")
    drafter["source"]["repo_id"] = "private-org/draft"
    paired["model"]["components"].append(drafter)
    run = paired
    for _ in range(depth):
        outer = _run("mlx-community/LFM2.5-1.2B-Instruct-4bit")
        outer["paired_runs"] = [run]
        run = outer
    metadata.return_value = SimpleNamespace(**fields)

    with pytest.raises(PublicationRefused) as error:
        ensure_public_model_sources(run)

    assert "paired_runs[0]." * depth + "model.components[1]" in str(error.value)
    metadata.assert_called_once_with("private-org/draft", token=False, timeout=5)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize(
    "source",
    [
        {"kind": "local"},
        {"kind": "huggingface", "repo_id": "/private/models/secret"},
        {"kind": "huggingface", "repo_id": "./secret"},
    ],
)
def test_paired_local_source_refused(depth, source, metadata):
    run = _paired_run("privacy-test/model", depth)
    paired = run
    for _ in range(depth):
        paired = paired["paired_runs"][0]
    paired["model"]["components"][0]["source"] = source

    with pytest.raises(PublicationRefused) as error:
        ensure_public_model_sources(run)

    assert "paired_runs[0]." * depth + "model.components[0]" in str(error.value)
    assert "local-path" in str(error.value)
    metadata.assert_not_called()


@pytest.mark.parametrize(
    "repo", ["mlx-community/LFM2.5-1.2B-Instruct-4bit", "privacy-test/model"]
)
def test_paired_catalog_or_public_repo_accepted(repo, metadata):
    run = _paired_run(repo, 1)
    preview = atomic_upload.preview_run(run, install_id="0123456789ab")
    source = preview["payload"]["paired_runs"][0]["model"]["components"][0]["source"]
    assert source["repo_id"] == repo
    if repo == "privacy-test/model":
        metadata.assert_called_once_with(repo, token=False, timeout=5)
    else:
        metadata.assert_not_called()


def test_repo_shared_by_outer_and_nested_runs_checked_once(metadata):
    run = _run()
    run["paired_runs"] = [_paired_run("privacy-test/model", 2), _run()]
    ensure_public_model_sources(run)
    metadata.assert_called_once_with("privacy-test/model", token=False, timeout=5)


@pytest.mark.parametrize("unreachable", [False, True])
def test_paired_refusal_precedes_payload_digest_and_consent(
    unreachable, metadata, monkeypatch
):
    if unreachable:
        metadata.side_effect = TimeoutError()
    else:
        metadata.return_value = SimpleNamespace(private=True, gated=False)
    stages = [
        "project_run_for_publication",
        "atomic_run_digest",
        "_ask_consent",
        "post_submission",
    ]
    mocks = [Mock() for _ in stages]
    for name, mock in zip(stages, mocks):
        monkeypatch.setattr(atomic_upload, name, mock)

    with pytest.raises(PublicationRefused, match=r"paired_runs\[0\]\.model"):
        atomic_upload.upload_run(_paired_run("privacy-test/model", 1))

    for mock in mocks:
        mock.assert_not_called()
