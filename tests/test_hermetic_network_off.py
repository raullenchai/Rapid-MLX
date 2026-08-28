# SPDX-License-Identifier: Apache-2.0
"""Regression for #2518 — the hermetic unit fixture is network-off by default.

``tests/conftest.py::_hermetic_hf_and_config_dirs`` used to redirect the HF
cache to an empty per-test directory but left network access open, so any
unit test that reached ``server.load_model`` with a real repo id quietly
downloaded ``mlx-community/Qwen3.5-9B-4bit`` into ``tmp_path`` — once per
test — and kept the hosted no-MLX lane busy for ~20 minutes (CI run
33084172753, ``test_server_load_model_order.py``). These tests pin the guard
itself: the hub is offline, an uncached ``snapshot_download`` fails
immediately without touching the Hub, a non-loopback socket connect is
refused with the offending test named, loopback keeps working, and the exact
prefetch that leaked can no longer download anything.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest

FAKE_REPO = "rapid-mlx-tests/does-not-exist-2518"
LEAKED_REPO = "mlx-community/Qwen3.5-9B-4bit"
# TEST-NET-1 (RFC 5737) is never routable, so a leak could not succeed here
# anyway — but the guard must refuse BEFORE any syscall, which the timing
# assertions pin.
OFF_BOX = ("192.0.2.1", 443)


def _guard_armed() -> bool:
    return bool(getattr(socket.socket.connect, "_hermetic_network_guard", False))


def _hub_cache_entry(repo_id: str) -> Path:
    return Path(os.environ["HF_HUB_CACHE"]) / f"models--{repo_id.replace('/', '--')}"


def test_default_fixture_pins_hub_offline_and_arms_socket_guard():
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert hf_constants.is_offline_mode() is True
    assert _guard_armed()
    assert getattr(socket.socket.connect_ex, "_hermetic_network_guard", False)


def test_snapshot_download_of_uncached_repo_fails_fast_offline():
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import LocalEntryNotFoundError

    started = time.monotonic()
    with pytest.raises(LocalEntryNotFoundError) as excinfo:
        hub.snapshot_download(FAKE_REPO)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, (
        f"offline refusal took {elapsed:.1f}s — did it hit the network?"
    )
    assert "outgoing traffic has been disabled" in str(excinfo.value)
    assert not _hub_cache_entry(FAKE_REPO).exists()


def test_non_loopback_connect_is_refused_immediately(request):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        started = time.monotonic()
        with pytest.raises(RuntimeError, match=r"192\.0\.2\.1:443") as excinfo:
            sock.connect(OFF_BOX)
        elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert type(excinfo.value).__name__ == "HermeticNetworkAccessError"
    message = str(excinfo.value)
    # Names the offending test and the reviewable opt-in.
    assert request.node.nodeid in message
    assert "requires_network" in message
    # Not an OSError: HTTP clients must not retry/translate it into a slow
    # ConnectionError.
    assert not isinstance(excinfo.value, OSError)


def test_connect_ex_is_guarded_too():
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match=r"192\.0\.2\.1:443"),
    ):
        sock.connect_ex(OFF_BOX)


def test_hostname_destinations_are_refused_before_dns():
    """A hostname that is not ``localhost`` is off-box: refuse without resolving."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match=r"huggingface\.co:443"),
    ):
        sock.connect(("huggingface.co", 443))


def test_loopback_sockets_still_work():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            conn, _ = server.accept()
            conn.close()
            assert client.getpeername()[0] == "127.0.0.1"


def test_leaked_prefetch_cannot_download_under_default_fixture(capsys):
    """The exact call that leaked: ``server._ensure_routing_config`` on an
    uncached repo id. It must fail fast (the product's own offline refusal or
    the hub's offline error, whichever fires first) and leave no cache entry —
    no ``config.json``, no shards."""
    server = pytest.importorskip("vllm_mlx.server")

    started = time.monotonic()
    with pytest.raises((SystemExit, RuntimeError)):
        server._ensure_routing_config(LEAKED_REPO)
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, f"prefetch took {elapsed:.1f}s — did it start a download?"
    assert not _hub_cache_entry(LEAKED_REPO).exists()
    out = capsys.readouterr()
    assert "offline" in (out.out + out.err).lower()


@pytest.mark.real_hf_cache
def test_real_hf_cache_only_opts_out_of_hf_cache_redirection(tmp_path):
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert hf_constants.is_offline_mode() is True
    assert _guard_armed()
    for var in (
        "RAPID_MLX_STATE_DIR",
        "RAPID_MLX_HOME",
        "RAPID_MLX_DDTREE_PATCH_CACHE",
        "RAPID_MLX_CONFIG_HOME",
    ):
        assert Path(os.environ[var]).is_relative_to(tmp_path)


@pytest.mark.requires_network
def test_requires_network_marker_disarms_the_guard():
    assert not _guard_armed()
    assert not getattr(socket.socket.connect_ex, "_hermetic_network_guard", False)
    # The hub constant mirrors the host's real environment again (a developer
    # may legitimately run the suite under HF_HUB_OFFLINE=1, so only equality
    # with the env is asserted, never "online").
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    raw = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    expected = raw is not None and raw.lower() in {"1", "on", "yes", "true"}
    assert hf_constants.is_offline_mode() is expected
