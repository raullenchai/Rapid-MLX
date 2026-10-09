import hashlib
import io
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import upload_release_r2 as target


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    payload = tmp_path / "release.zip"
    payload.write_bytes(b"signed-release-bytes")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "private-fixture-token")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "stale-inherited-token")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "stale-legacy-token")
    response = {"success": True, "result": {"id": "b" * 32, "status": "active"}}
    monkeypatch.setattr(
        target.urllib.request,
        "urlopen",
        lambda *_a, **_k: io.StringIO(json.dumps(response)),
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ContentLength": payload.stat().st_size,
                    "Metadata": {
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()
                    },
                }
            ),
        )

    monkeypatch.setattr(target.subprocess, "run", run)
    return payload, response, calls


def invoke(payload):
    target.upload(
        payload, "rapid-desktop-dist", "rapid-mac/0.16.0/archive.zip", "application/zip"
    )


def test_uses_existing_s3_transfer_and_keeps_credentials_out_of_arguments(transfer):
    payload, _, calls = transfer
    invoke(payload)
    command, kwargs = calls[0]
    assert command[:3] == ["aws", "s3", "cp"]
    assert "--content-type" in command and "application/zip" in command
    assert "private-fixture-token" not in " ".join(command)
    assert kwargs["env"]["AWS_ACCESS_KEY_ID"] == "b" * 32
    assert (
        kwargs["env"]["AWS_SECRET_ACCESS_KEY"]
        == hashlib.sha256(b"private-fixture-token").hexdigest()
    )
    assert "AWS_SESSION_TOKEN" not in kwargs["env"]
    assert "AWS_SECURITY_TOKEN" not in kwargs["env"]
    assert "CLOUDFLARE_API_TOKEN" not in kwargs["env"]
    assert calls[1][0][1:3] == ["s3api", "head-object"]


def test_large_payload_is_delegated_intact_to_s3_transfer_manager(transfer):
    payload, _, calls = transfer
    with payload.open("wb") as stream:
        stream.truncate(350 * 1024 * 1024)
    invoke(payload)
    assert calls[0][0][3] == str(payload)
    assert payload.stat().st_size == 350 * 1024 * 1024


@pytest.mark.parametrize(
    "field,value",
    [("success", False), ("status", "disabled"), ("id", "wrong"), ("id", 17)],
)
def test_unverified_token_never_uploads(transfer, field, value):
    payload, response, calls = transfer
    if field == "success":
        response[field] = value
    else:
        response["result"][field] = value
    with pytest.raises(ValueError, match="verified active"):
        invoke(payload)
    assert calls == []


@pytest.mark.parametrize("account,token", [("invalid", "present"), ("a" * 32, "")])
def test_bad_credentials_reject(transfer, monkeypatch, account, token):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", account)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", token)
    with pytest.raises(ValueError, match="credentials"):
        invoke(transfer[0])


@pytest.mark.parametrize(
    "bucket,key", [("other", "rapid-mac/x"), ("rapid-desktop-dist", "models/x")]
)
def test_wrong_namespace_rejects(transfer, bucket, key):
    with pytest.raises(ValueError, match="distribution bucket"):
        target.upload(transfer[0], bucket, key, "application/zip")


def test_missing_payload_rejects(transfer):
    with pytest.raises(ValueError, match="not a file"):
        invoke(transfer[0].with_name("absent"))


@pytest.mark.parametrize(
    "remote",
    [{"ContentLength": 0}, {"ContentLength": 20, "Metadata": {"sha256": "wrong"}}],
)
def test_remote_mismatch_is_not_success(transfer, monkeypatch, remote):
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda command, **_kw: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(remote)
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        invoke(transfer[0])


def test_cli_entrypoint(transfer, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_release_r2.py",
            str(transfer[0]),
            "rapid-desktop-dist",
            "rapid-mac/0.16.0/archive.zip",
            "application/zip",
        ],
    )
    runpy.run_path(target.__file__, run_name="__main__")
    assert len(transfer[2]) == 2


def test_upload_failure_propagates_without_head_success(transfer, monkeypatch):
    def fail(command, **_kw):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(target.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        invoke(transfer[0])


def test_release_workflow_enrolls_multipart_upload_and_hosted_coverage():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/rapid-mac-release.yml").read_text()
    )
    steps = workflow["jobs"]["mirror-dist"]["steps"]
    assert any(step.get("uses", "").startswith("actions/checkout@") for step in steps)
    shell = next(
        step["run"]
        for step in steps
        if step.get("name") == "Mirror update archives and compose updater fallback"
    )
    assert shell.count("python3 scripts/upload_release_r2.py") == 2
    assert "wrangler" not in shell
    assert '"$DMG" "$R2_BUCKET" "$VERSIONED_KEY"' in shell
    assert '"$SPARKLE_ZIP" "$R2_BUCKET" "$SPARKLE_KEY"' in shell
    assert (
        "--cov=scripts.upload_release_r2"
        in (root / ".github/workflows/ci.yml").read_text()
    )
