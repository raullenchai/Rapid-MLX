"""Unit tests for ``scripts/mirror_to_r2.py``.

Three cases, all offline (no network, no boto3 calls):

1. CLI argparse smoke — the parser accepts the documented flag set and
   surfaces defaults.
2. Content-type router shape — ``.json`` / ``.md`` / everything-else.
3. Skip-if-exists head-check logic — with a mocked boto3-shaped client,
   an R2 object whose size matches HF is SKIPPED (no ``upload_file``
   call); a size mismatch or 404 forces an upload.

Codex round-2 BLOCKING #4: ``boto3`` / ``botocore`` are only pulled by
the ``[mirror]`` optional extra. Default ``pip install rapid-mlx[dev]``
does not include them, and CI that runs ``pytest tests/`` without the
mirror extra should collect + skip cleanly rather than fail at import.
``pytest.importorskip("botocore")`` at module load enforces that.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Codex round-2 BLOCKING #4: skip the whole module if botocore isn't
# installed (default dev env without the [mirror] extra). Load-time
# check so the mirror_to_r2 module import below — which does not
# eagerly import boto3 itself, but many of the tests DO — doesn't
# collect into a false failure.
pytest.importorskip("botocore")

# Load the CLI module from ``scripts/`` with its package identity so its
# package-relative imports follow the same path as normal module execution.
_SCRIPT = Path(__file__).parent.parent / "scripts" / "mirror_to_r2.py"
_SPEC = importlib.util.spec_from_file_location("scripts.mirror_to_r2", _SCRIPT)
assert _SPEC and _SPEC.loader
mirror_to_r2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mirror_to_r2
_SPEC.loader.exec_module(mirror_to_r2)


# --------- 1. CLI argparse smoke ---------


def test_cli_parser_accepts_repo_id_and_defaults() -> None:
    """The parser accepts a bare repo_id and surfaces all documented defaults."""
    p = mirror_to_r2._build_parser()
    args = p.parse_args(["mlx-community/Qwen3-0.6B-4bit"])
    assert args.repo_id == "mlx-community/Qwen3-0.6B-4bit"
    assert args.bucket == mirror_to_r2.DEFAULT_BUCKET
    assert args.profile == mirror_to_r2.DEFAULT_PROFILE
    assert args.endpoint_url == mirror_to_r2.DEFAULT_ENDPOINT_URL
    assert args.public_base == mirror_to_r2.DEFAULT_PUBLIC_BASE
    assert args.dry_run is False
    assert args.verify_only is False
    assert args.force_unmirrored is False
    assert args.tmp_dir is None


def test_cli_parser_accepts_all_flags() -> None:
    """Every documented flag can be overridden from the CLI."""
    p = mirror_to_r2._build_parser()
    args = p.parse_args(
        [
            "some/repo",
            "--endpoint-url",
            "https://elsewhere.example",
            "--bucket",
            "other-bucket",
            "--profile",
            "other-profile",
            "--public-base",
            "https://elsewhere.example",
            "--dry-run",
            "--verify-only",
            "--force-unmirrored",
            "--tmp-dir",
            "/mnt/scratch",
        ]
    )
    assert args.endpoint_url == "https://elsewhere.example"
    assert args.bucket == "other-bucket"
    assert args.profile == "other-profile"
    assert args.public_base == "https://elsewhere.example"
    assert args.dry_run is True
    # Codex round-1 NIT: --verify-only was not part of the smoke matrix
    # before. One documented flag must always be assertable.
    assert args.verify_only is True
    assert args.force_unmirrored is True
    assert args.tmp_dir == "/mnt/scratch"


def test_cli_parser_rejects_missing_repo_id() -> None:
    """Bare invocation is an error (repo_id is positional-required)."""
    p = mirror_to_r2._build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_unmirrored_repo_refuses_without_force_and_allows_override(
    monkeypatch, capsys
) -> None:
    entry = type(
        "Entry",
        (),
        {"reason": "unused: no pulls", "since": "2026-09-24"},
    )()
    monkeypatch.setattr(
        mirror_to_r2,
        "load_unmirrored",
        lambda *_args: {"org/repo": entry},
    )
    hf_calls = []
    monkeypatch.setattr(
        mirror_to_r2, "_hf_files", lambda repo: hf_calls.append(repo) or []
    )
    monkeypatch.setattr(mirror_to_r2, "_r2_client", lambda *_args: object())

    assert mirror_to_r2.mirror_repo("org/repo") == 2
    assert hf_calls == []
    refusal = capsys.readouterr().err
    assert refusal.count("SKIP intentionally unmirrored") == 1
    assert "unused: no pulls" in refusal
    assert "2026-09-24" in refusal
    assert "--force-unmirrored" in refusal

    assert mirror_to_r2.mirror_repo("org/repo", force_unmirrored=True) == 0
    assert hf_calls == ["org/repo"]


def test_redirect_preserves_head_method() -> None:
    handler = mirror_to_r2._FinalUrlRedirectHandler()
    request = urllib.request.Request("https://models.example/file", method="HEAD")
    redirected = handler.redirect_request(
        request, io.BytesIO(), 302, "Found", Message(), "https://dl.example/file"
    )
    assert redirected is not None
    assert redirected.get_method() == "HEAD"


# --------- 2. Content-type router shape ---------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("config.json", "application/json"),
        ("tokenizer_config.json", "application/json"),
        ("nested/dir/config.json", "application/json"),
        ("README.md", "text/markdown"),
        ("subdir/NOTES.md", "text/markdown"),
        ("model.safetensors", "application/octet-stream"),
        ("model-00001-of-00013.safetensors", "application/octet-stream"),
        ("tokenizer.model", "application/octet-stream"),
        ("special_tokens_map", "application/octet-stream"),
        ("weights.bin", "application/octet-stream"),
        (".gitattributes", "application/octet-stream"),
    ],
)
def test_content_type_for(filename: str, expected: str) -> None:
    assert mirror_to_r2.content_type_for(filename) == expected


def test_content_type_for_is_case_insensitive_on_extension() -> None:
    """Uppercase extensions map to the same MIME type as lowercase."""
    assert mirror_to_r2.content_type_for("README.MD") == "text/markdown"
    assert mirror_to_r2.content_type_for("Config.JSON") == "application/json"


# --------- 3. Skip-if-exists head-check logic ---------


def test_should_skip_when_r2_size_matches_hf_size() -> None:
    """R2 already has the object at the expected size → SKIP (no upload)."""
    assert mirror_to_r2.should_skip(existing_size=937, expected_size=937) is True


def test_should_skip_false_when_r2_missing() -> None:
    """R2 HEAD → 404 (``None``) means we must upload."""
    assert mirror_to_r2.should_skip(existing_size=None, expected_size=937) is False


def test_should_skip_false_on_size_mismatch() -> None:
    """R2 has an object of the wrong size (partial upload) → re-upload."""
    assert mirror_to_r2.should_skip(existing_size=500, expected_size=937) is False


def test_should_skip_false_when_hf_size_unknown() -> None:
    """If HF metadata didn't expose a size (``expected_size is None``),
    refuse to skip.

    Prevents accepting a truncated leftover R2 object from a previous
    run where we couldn't validate the true length.

    Codex round-2 BLOCKING #2: this must NOT collapse with the
    "expected_size == 0 (real empty file)" branch below.
    """
    assert mirror_to_r2.should_skip(existing_size=500, expected_size=None) is False


def test_should_skip_true_for_legitimate_empty_file() -> None:
    """A real 0-byte HF file already at 0 bytes on R2 → skip.

    Codex round-2 BLOCKING #2: without distinguishing ``None`` (unknown)
    from ``0`` (empty), every empty file gets re-uploaded on every run.
    """
    assert mirror_to_r2.should_skip(existing_size=0, expected_size=0) is True


def test_should_skip_false_when_r2_has_content_but_expected_empty() -> None:
    """R2 has non-empty bytes but HF says empty → re-upload (size mismatch)."""
    assert mirror_to_r2.should_skip(existing_size=42, expected_size=0) is False


# ---- codex round-1 BLOCKING #1: size-only isn't proof of identity for LFS


def test_should_skip_true_when_size_and_sha_match() -> None:
    """Both size AND sha match → skip."""
    sha = "a" * 64
    assert (
        mirror_to_r2.should_skip(
            existing_size=1000,
            expected_size=1000,
            existing_sha256=sha,
            expected_sha256=sha,
        )
        is True
    )


def test_should_skip_false_when_size_matches_but_sha_missing_on_lfs() -> None:
    """Size match + HF has LFS sha + R2 has no sha metadata → re-upload.

    Guards against stale pre-metadata uploads: an object uploaded before
    we started tagging ``x-amz-meta-hf-sha256`` still shows the right
    size but we cannot prove it's the current bytes.
    """
    sha = "a" * 64
    assert (
        mirror_to_r2.should_skip(
            existing_size=1000,
            expected_size=1000,
            existing_sha256=None,
            expected_sha256=sha,
        )
        is False
    )


def test_should_skip_false_when_sha_mismatch() -> None:
    """Size match + shas differ → same-length-but-different-bytes: re-upload."""
    assert (
        mirror_to_r2.should_skip(
            existing_size=1000,
            expected_size=1000,
            existing_sha256="a" * 64,
            expected_sha256="b" * 64,
        )
        is False
    )


def test_should_skip_true_size_only_for_non_lfs_files() -> None:
    """Non-LFS files (no HF sha256) fall back to size-only skip.

    Rationale in ``should_skip`` docstring: small text files without
    LFS hashes shouldn't be forced back through the network on every
    resume pass.
    """
    assert (
        mirror_to_r2.should_skip(
            existing_size=100,
            expected_size=100,
            existing_sha256="ignored" * 8,  # not 64 chars but that's fine here
            expected_sha256=None,  # HF didn't give us a sha
        )
        is True
    )


def test_r2_head_size_returns_content_length() -> None:
    """A mocked boto3 client returns the size on ``head_object``."""
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 12345}
    got = mirror_to_r2._r2_head_size(client, "bucket", "key")
    assert got == 12345
    client.head_object.assert_called_once_with(Bucket="bucket", Key="key")


def test_r2_head_size_returns_none_on_404() -> None:
    """A boto3 ``ClientError`` with ``Code=404`` maps to ``None``."""
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    assert mirror_to_r2._r2_head_size(client, "bucket", "key") is None


def test_r2_head_size_raises_on_permission_error() -> None:
    """Non-404 errors bubble up (fail-fast on real problems)."""
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "HeadObject"
    )
    with pytest.raises(ClientError):
        mirror_to_r2._r2_head_size(client, "bucket", "key")


# ---- codex round-1 BLOCKING #2: percent-encode segments in the public URL


@pytest.mark.parametrize(
    ("key", "expected_url"),
    [
        (
            "mlx-community/simple/config.json",
            "https://models.rapidmlx.com/mlx-community/simple/config.json",
        ),
        (
            "mlx-community/repo name/model card.md",
            "https://models.rapidmlx.com/mlx-community/repo%20name/model%20card.md",
        ),
        (
            # ``#`` and ``?`` would otherwise start a fragment / query.
            "mlx-community/repo#v2/config?.json",
            "https://models.rapidmlx.com/mlx-community/repo%23v2/config%3F.json",
        ),
        (
            "/leading-slash/repo/file.json",
            "https://models.rapidmlx.com/leading-slash/repo/file.json",
        ),
    ],
)
def test_public_url_percent_encodes_segments(key: str, expected_url: str) -> None:
    assert mirror_to_r2._public_url("https://models.rapidmlx.com", key) == expected_url


# ---- upload adds hf-sha256 metadata for LFS files, omits for non-LFS


def test_upload_one_tags_hf_sha_for_lfs_files(tmp_path) -> None:
    """LFS weight shards get x-amz-meta-hf-sha256 on upload."""
    client = MagicMock()
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"stub")
    sha = "c" * 64
    mirror_to_r2._upload_one(
        client,
        f,
        "bucket",
        "org/repo/model.safetensors",
        "application/octet-stream",
        lfs_sha256=sha,
    )
    call = client.upload_file.call_args
    assert call.kwargs["ExtraArgs"]["ContentType"] == "application/octet-stream"
    assert call.kwargs["ExtraArgs"]["Metadata"] == {"hf-sha256": sha}


def test_upload_one_omits_metadata_when_no_sha(tmp_path) -> None:
    """Non-LFS files (no HF sha256) upload without a Metadata block."""
    client = MagicMock()
    f = tmp_path / "config.json"
    f.write_bytes(b"{}")
    mirror_to_r2._upload_one(
        client,
        f,
        "bucket",
        "org/repo/config.json",
        "application/json",
        lfs_sha256=None,
    )
    extra = client.upload_file.call_args.kwargs["ExtraArgs"]
    assert extra["ContentType"] == "application/json"
    assert "Metadata" not in extra


# --------- non-LFS files: git blob id decides skip and verify ---------

_CONFIG = b'{"a": 1}\n'
_STALE = b'{"a": 2}\n'  # same size, older content


def _blob_oid(data: bytes) -> str:
    import hashlib

    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data, usedforsecurity=False
    ).hexdigest()


class _BlobClient:
    """boto3-shaped stand-in holding one object per key."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.uploads: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def head_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
        return {"ContentLength": len(self.objects[Key]), "Metadata": {}}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs):  # noqa: N803
        self.uploads.append(Key)
        self.objects[Key] = Path(Filename).read_bytes()


def test_should_skip_non_lfs_requires_matching_blob_id() -> None:
    oid = _blob_oid(_CONFIG)
    assert mirror_to_r2.should_skip(9, 9, existing_git_oid=oid, expected_git_oid=oid)
    assert not mirror_to_r2.should_skip(
        9, 9, existing_git_oid=_blob_oid(_STALE), expected_git_oid=oid
    )
    assert not mirror_to_r2.should_skip(9, 9, expected_git_oid=oid)


def test_r2_git_oid_hashes_the_object_body() -> None:
    client = _BlobClient({"k": _CONFIG})
    assert mirror_to_r2._r2_git_oid(client, "b", "k") == _blob_oid(_CONFIG)


def test_r2_git_oid_refuses_a_short_body() -> None:
    client = MagicMock()
    client.get_object.return_value = {
        "Body": io.BytesIO(b"abc"),
        "ContentLength": 10,
    }
    with pytest.raises(OSError, match="short read"):
        mirror_to_r2._r2_git_oid(client, "b", "k")


def test_hf_files_keeps_blob_id_for_non_lfs_files_only(monkeypatch) -> None:
    import huggingface_hub

    oid = _blob_oid(_CONFIG)
    siblings = [
        MagicMock(rfilename="config.json", size=9, lfs=None, blob_id=oid.upper()),
        MagicMock(
            rfilename="model.safetensors",
            size=64,
            lfs=MagicMock(sha256="a" * 64),
            blob_id="b" * 40,
        ),
        MagicMock(rfilename="odd.txt", size=1, lfs=None, blob_id="short"),
        MagicMock(rfilename="nonhex.txt", size=1, lfs=None, blob_id="z" * 40),
    ]

    class Api:
        def model_info(self, repo, **kwargs):
            return MagicMock(siblings=siblings)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    files = {f.relpath: f for f in mirror_to_r2._hf_files("org/repo")}
    assert files["config.json"].git_oid == oid
    assert files["model.safetensors"].git_oid is None
    assert files["odd.txt"].git_oid is None
    assert files["nonhex.txt"].git_oid is None


def _run_blob_repo(monkeypatch, tmp_path, r2_body: bytes, **kwargs):
    item = mirror_to_r2.FileMeta(
        "config.json", len(_CONFIG), "org/repo/config.json", None, _blob_oid(_CONFIG)
    )
    client = _BlobClient({item.key: r2_body})
    local = tmp_path / "hf-config.json"
    monkeypatch.setattr(mirror_to_r2, "load_unmirrored", lambda *_a: {})
    monkeypatch.setattr(mirror_to_r2, "_hf_files", lambda *_a: [item])
    monkeypatch.setattr(mirror_to_r2, "_r2_client", lambda *_a: client)
    monkeypatch.setattr(
        mirror_to_r2,
        "_download_one_hf",
        lambda *_a: (local.write_bytes(_CONFIG), local)[1],
    )
    monkeypatch.setattr(mirror_to_r2, "_http_range_get_status", lambda _url: 200)
    rc = mirror_to_r2.mirror_repo("org/repo", tmp_dir=tmp_path, **kwargs)
    return rc, client


def test_same_size_stale_config_is_reuploaded(monkeypatch, tmp_path, capsys) -> None:
    rc, client = _run_blob_repo(monkeypatch, tmp_path, _STALE)
    assert rc == 0
    assert client.uploads == ["org/repo/config.json"]
    assert client.objects["org/repo/config.json"] == _CONFIG
    assert "1 uploaded, 0 skipped" in capsys.readouterr().out


def test_current_config_is_skipped_by_blob_id(monkeypatch, tmp_path, capsys) -> None:
    rc, client = _run_blob_repo(monkeypatch, tmp_path, _CONFIG)
    assert rc == 0
    assert client.uploads == []
    assert "blob-id+size" in capsys.readouterr().out


def test_verify_fails_on_a_stale_config(monkeypatch, tmp_path, capsys) -> None:
    rc, client = _run_blob_repo(monkeypatch, tmp_path, _STALE, verify_only=True)
    assert rc == 3
    assert client.uploads == []
    assert "r2-blob-id:" in capsys.readouterr().err
