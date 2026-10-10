# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
import ssl

import pytest

from rapid_mlx.share.privacy import _known_namespaces, clear_legacy_prompt_cache
from rapid_mlx.share.tls import client_context
from rapid_mlx.share.ws_tunnel import TunnelClient


def test_cleanup_only_selected_model_and_all_interrupted_snapshots(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    target = "model"
    fingerprint = sorted(_known_namespaces(target, target))[0]
    names = [
        fingerprint,
        fingerprint + ".new",
        fingerprint + ".old",
        target + "-other",
        "unrelated",
    ]
    for name in names:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "tokens.bin").write_bytes(b"private")
    assert clear_legacy_prompt_cache(target) == 3
    assert (root / (target + "-other") / "tokens.bin").exists()
    assert (root / "unrelated" / "tokens.bin").exists()
    assert clear_legacy_prompt_cache(target) == 0


def test_cleanup_fails_closed_under_contention(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    fingerprint = sorted(_known_namespaces("model", "model"))[0]
    (root / fingerprint).mkdir(parents=True)
    with (root / (fingerprint + ".txlock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(OSError, match="in use"):
            clear_legacy_prompt_cache("model")
    assert (root / fingerprint).exists()


def test_cleanup_refuses_symlink_without_touching_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    root.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "private").write_bytes(b"keep")
    fingerprint = sorted(_known_namespaces("model", "model"))[0]
    (root / fingerprint).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(OSError, match="symlink"):
        clear_legacy_prompt_cache("model")
    assert (elsewhere / "private").read_bytes() == b"keep"


def test_tls_verification_and_relay_scheme():
    context = client_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    client = TunnelClient(local_port=18765, tunnel_id="test")
    assert client._verified_connect_kwargs("wss://relay.test/up", None)[
        "ssl"
    ].check_hostname
    assert "ssl" not in client._verified_connect_kwargs("ws://localhost/up", None)


@pytest.mark.parametrize("model", ["model.new", "model.old"])
def test_model_transaction_suffix_is_not_stripped(tmp_path, monkeypatch, model):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    fingerprint = sorted(_known_namespaces(model, model))[0]
    for suffix in ("", ".new", ".old"):
        (root / (fingerprint + suffix)).mkdir(parents=True)
    assert clear_legacy_prompt_cache(model) == 3


def test_colliding_model_names_fail_closed_without_deleting_either(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    selected = sorted(_known_namespaces("a/b", "a--b"))[0]
    other = sorted(_known_namespaces("a--b", "a--b"))[0]
    for name in (selected, other):
        (root / name).mkdir(parents=True)
        (root / name / "tokens.bin").write_bytes(b"keep")
    with pytest.raises(OSError, match="ambiguous"):
        clear_legacy_prompt_cache("a/b")
    assert (root / selected / "tokens.bin").exists()
    assert (root / other / "tokens.bin").exists()


def test_ambiguous_legacy_name_ending_in_suffix_stops_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".cache/rapid-mlx/prefix_cache"
    (root / "org--foo.new").mkdir(parents=True)
    with pytest.raises(OSError, match="ambiguous"):
        clear_legacy_prompt_cache("org/foo.new")


def test_radix_artifact_is_private_before_publication(tmp_path, monkeypatch):
    import os
    import stat

    from rapid_mlx.runtime.radix_index import RadixPrefixIndex

    radix = RadixPrefixIndex()
    radix.insert((100, 200, 300))
    path = tmp_path / "radix.index"
    staging = tmp_path / "radix.index.tmp"
    staging.write_text("old")
    staging.chmod(0o666)
    real_dump = __import__("json").dump

    def checked_dump(payload, stream):
        assert stat.S_IMODE(os.fstat(stream.fileno()).st_mode) == 0o600
        return real_dump(payload, stream)

    monkeypatch.setattr("rapid_mlx.runtime.radix_index.json.dump", checked_dump)
    old_umask = os.umask(0)
    try:
        radix.save(str(path))
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "selected,legacy", [("model", "model"), (".model", "model"), ("model", "model.new")]
)
def test_all_unhashed_legacy_candidates_are_ambiguous(
    tmp_path, monkeypatch, selected, legacy
):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".cache/rapid-mlx/prefix_cache" / legacy
    path.mkdir(parents=True)
    (path / "tokens.bin").write_bytes(b"keep")
    with pytest.raises(OSError, match="ambiguous"):
        clear_legacy_prompt_cache(selected)
    assert (path / "tokens.bin").read_bytes() == b"keep"


@pytest.mark.parametrize("component", [".cache", ".cache/rapid-mlx"])
def test_cleanup_refuses_symlinked_ancestor(tmp_path, monkeypatch, component):
    monkeypatch.setenv("HOME", str(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    marker = elsewhere / "keep"
    marker.write_text("unchanged")
    link = tmp_path / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(OSError, match="symlink"):
        clear_legacy_prompt_cache("model")
    assert marker.read_text() == "unchanged"


def test_broken_trust_bundle_uses_terminal_certificate_error(tmp_path, monkeypatch):
    import certifi

    from rapid_mlx.share.tls import CERTIFICATE_HINT

    broken = tmp_path / "broken.pem"
    broken.write_text("not a certificate")
    monkeypatch.setattr(certifi, "where", lambda: str(broken))
    with pytest.raises(ssl.SSLCertVerificationError, match="SSL_CERT_FILE") as error:
        client_context()
    assert CERTIFICATE_HINT in str(error.value)
