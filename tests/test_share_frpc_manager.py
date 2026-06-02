# SPDX-License-Identifier: Apache-2.0
"""Unit tests for vllm_mlx.share.frpc_manager.

We mock the network + filesystem so tests don't actually touch the
GitHub release CDN. The integrity check is the load-bearing assertion:
a corrupted download must raise instead of silently caching a bad
binary.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx.share import frpc_manager
from vllm_mlx.share._constants import FRPC_VERSION


def _fake_tarball(frpc_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=f"frp_{FRPC_VERSION}_test/frpc")
        info.size = len(frpc_bytes)
        tf.addfile(info, io.BytesIO(frpc_bytes))
    return buf.getvalue()


def test_platform_tag_recognized_on_supported_hosts():
    with patch("platform.system", return_value="Darwin"), patch(
        "platform.machine", return_value="arm64"
    ):
        assert frpc_manager._platform_tag() == "darwin_arm64"

    with patch("platform.system", return_value="Linux"), patch(
        "platform.machine", return_value="x86_64"
    ):
        assert frpc_manager._platform_tag() == "linux_amd64"


def test_platform_tag_rejects_unsupported_os():
    with patch("platform.system", return_value="Windows"), pytest.raises(
        RuntimeError, match="only supported on macOS and Linux"
    ):
        frpc_manager._platform_tag()


def test_platform_tag_rejects_unsupported_arch():
    with patch("platform.system", return_value="Linux"), patch(
        "platform.machine", return_value="riscv64"
    ), pytest.raises(RuntimeError, match="riscv64"):
        frpc_manager._platform_tag()


def test_ensure_downloads_verifies_and_extracts(tmp_path: Path):
    fake_binary = b"#!/fake/frpc\n" + b"x" * 1024
    tarball = _fake_tarball(fake_binary)
    expected_sha = hashlib.sha256(tarball).hexdigest()

    def fake_urlretrieve(url, dest):
        Path(dest).write_bytes(tarball)

    with patch.object(frpc_manager, "_cache_dir", return_value=tmp_path), patch.dict(
        frpc_manager.FRPC_SHA256, {"darwin_arm64": expected_sha}
    ), patch("platform.system", return_value="Darwin"), patch(
        "platform.machine", return_value="arm64"
    ), patch.object(frpc_manager, "_download", side_effect=fake_urlretrieve):
        binp = frpc_manager.ensure()

    assert binp.exists()
    assert binp.read_bytes() == fake_binary
    assert binp.stat().st_mode & 0o100  # executable bit set


def test_ensure_rejects_corrupted_download(tmp_path: Path):
    tarball = _fake_tarball(b"frpc")
    wrong_sha = "0" * 64

    with patch.object(frpc_manager, "_cache_dir", return_value=tmp_path), patch.dict(
        frpc_manager.FRPC_SHA256, {"darwin_arm64": wrong_sha}
    ), patch("platform.system", return_value="Darwin"), patch(
        "platform.machine", return_value="arm64"
    ), patch.object(
        frpc_manager,
        "_download",
        side_effect=lambda url, dest: Path(dest).write_bytes(tarball),
    ), pytest.raises(RuntimeError, match="sha256 mismatch"):
        frpc_manager.ensure()

    # Mismatch path must not leave a stale binary or tmp file behind that
    # a subsequent call would silently reuse.
    assert not (tmp_path / f"frpc-{FRPC_VERSION}").exists()
    assert not any(tmp_path.glob("*.tar.gz"))


def test_ensure_reuses_cached_binary(tmp_path: Path):
    cached = tmp_path / f"frpc-{FRPC_VERSION}"
    cached.write_bytes(b"already there")
    cached.chmod(0o755)

    with patch.object(frpc_manager, "_cache_dir", return_value=tmp_path), patch.object(
        frpc_manager, "_download"
    ) as mock_download:
        binp = frpc_manager.ensure()

    assert binp == cached
    mock_download.assert_not_called()


def test_render_config_emits_valid_toml_shape():
    cfg = frpc_manager.render_config(
        server_addr="tunnel.rapidmlx.com",
        server_port=7000,
        auth_token="t0ken",
        subdomain="abc123",
        local_port=8765,
    )
    assert 'serverAddr = "tunnel.rapidmlx.com"' in cfg
    assert "serverPort = 7000" in cfg
    assert 'auth.token = "t0ken"' in cfg
    assert 'subdomain = "abc123"' in cfg
    assert "localPort = 8765" in cfg
    # Proxy block present and HTTP-typed (not TCP — required for subdomain routing).
    assert "[[proxies]]" in cfg
    assert 'type = "http"' in cfg
