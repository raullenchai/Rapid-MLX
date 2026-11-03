# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import psutil
import pytest

from rapid_mlx.chat_runtime import SCHEMA_VERSION
from rapid_mlx.chat_runtime.owner import (
    CAPABILITIES,
    FORMAT_VERSION,
    WIRE_VERSION,
    ClientLeases,
    discover,
    discover_or_start,
    profile_directory,
)
from rapid_mlx.headless_service.config import atomic_write


def test_desktop_discovery_versions_match_owner():
    swift = (
        Path(__file__).parents[1]
        / "apps/rapid-mac/Sources/Rapid/Chat/ChatOwnerConnection.swift"
    ).read_text(encoding="utf-8")
    discovery = swift[swift.index("struct ChatOwnerDiscovery") :]
    versions = {
        name: int(re.search(rf"static let {name} = (\d+)", discovery).group(1))
        for name in ("formatVersion", "wireVersion", "schemaVersion")
    }
    assert versions == {
        "formatVersion": FORMAT_VERSION,
        "wireVersion": WIRE_VERSION,
        "schemaVersion": SCHEMA_VERSION,
    }


def stop_owned(owner):
    process = psutil.Process(owner["pid"])
    assert abs(process.create_time() - owner["process_start_time"]) < 0.01
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_two_clients_one_owner_stale_discovery_and_permissions(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setenv("RAPID_MLX_CHAT_IDLE_SECONDS", "60")
    profile = "race"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        owners = list(pool.map(lambda _: discover_or_start(profile), range(2)))
    first = owners[0]
    try:
        from rapid_mlx.chat_runtime.store import Store

        assert len({owner["pid"] for owner in owners}) == 1
        assert first["schema_version"] == Store.schema_version == 5
        run = profile_directory(profile) / "run"
        assert stat.S_IMODE(run.stat().st_mode) == 0o700
        assert stat.S_IMODE((run / "discovery.json").stat().st_mode) == 0o600
        credential = run / first["credential_file"]
        assert stat.S_IMODE(credential.stat().st_mode) == 0o600
        assert "token" not in json.loads((run / "discovery.json").read_text())
        with httpx.Client(trust_env=False) as client:
            response = client.get(first["endpoint"] + "/v1/rapid/chat/capabilities")
            assert response.status_code == 401
            response = client.get(
                first["endpoint"] + "/v1/rapid/chat/capabilities",
                headers={"Authorization": "Bearer " + first["token"]},
            )
            assert response.json()["schema_version"] == first["schema_version"]
            assert "application/pdf" in response.json()["capabilities"]["attachments"]
        credential.chmod(0o644)
        with pytest.raises(RuntimeError, match="private"):
            discover(profile)
        credential.chmod(0o600)
        assert discover(profile)["owner_epoch"] == first["owner_epoch"]
    finally:
        stop_owned(first)
    # A stale record points to THIS process. Discovery must never signal it.
    stale = {key: value for key, value in first.items() if key != "token"}
    stale.update(pid=os.getpid(), process_start_time=psutil.Process().create_time())
    atomic_write(run / stale["credential_file"], b"stale-test-credential")
    atomic_write(run / "discovery.json", json.dumps(stale).encode())
    second = discover_or_start(profile)
    try:
        assert second["owner_epoch"] != first["owner_epoch"]
        assert bool(second["token"] != first["token"])
        assert psutil.pid_exists(os.getpid())
        assert second["store_uuid"] == first["store_uuid"]
    finally:
        stop_owned(second)


def test_owner_idle_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setenv("RAPID_MLX_CHAT_IDLE_SECONDS", "0.7")
    owner = discover_or_start("idle")
    process = psutil.Process(owner["pid"])
    headers = {"Authorization": "Bearer " + owner["token"]}
    try:
        with httpx.Client(
            base_url=owner["endpoint"], headers=headers, trust_env=False
        ) as client:
            assert (
                client.post(
                    "/v1/rapid/chat/clients", json={"client_id": "test"}
                ).status_code
                == 200
            )
            time.sleep(0.9)
            assert process.is_running()
            # An open invalidation feed must not pin an expired/detached owner
            # in Uvicorn's graceful request drain forever.
            with client.stream("GET", "/v1/rapid/chat/changes") as feed:
                assert next(feed.iter_lines()).startswith(":")
                assert client.delete("/v1/rapid/chat/clients/test").status_code == 204
                process.wait(timeout=6)
        assert not (profile_directory("idle") / "run/discovery.json").exists()
    finally:
        if process.is_running():
            stop_owned(owner)


def test_ensure_bootstrap_is_credential_silent(tmp_path):
    environment = {
        **os.environ,
        "RAPID_MLX_CHAT_HOME": str(tmp_path),
        "RAPID_MLX_CHAT_IDLE_SECONDS": "60",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.chat_runtime.owner",
            "--profile",
            "ensure-test",
            "--ensure",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    run = tmp_path / "ensure-test" / "run"
    record = json.loads((run / "discovery.json").read_text())
    token = (run / record["credential_file"]).read_text().strip()
    assert token and token not in result.stderr
    try:
        again = subprocess.run(
            result.args, env=environment, capture_output=True, text=True, timeout=25
        )
        assert again.returncode == 0, again.stderr
        assert again.stdout == "" and token not in again.stderr
        assert (
            json.loads((run / "discovery.json").read_text())["owner_epoch"]
            == record["owner_epoch"]
        )
    finally:
        stop_owned({**record, "token": token})


def test_ensure_failure_is_silent_and_nonzero(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.chat_runtime.owner",
            "--profile",
            "../invalid",
            "--ensure",
        ],
        env={**os.environ, "RAPID_MLX_CHAT_HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip() == "Chat owner unavailable: ValueError"


def test_lease_expiry_active_work_and_detach(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    leases = ClientLeases(ttl=10)
    leases.attach("client")
    clock[0] = 11
    assert not leases.idle(5, False)
    clock[0] = 17
    assert leases.idle(5, False)
    assert not leases.idle(5, True)
    leases.attach("again")
    leases.detach("again")
    clock[0] = 23
    assert leases.idle(5, False)
    assert CAPABILITIES["side_effect_tools"] is True


def test_idle_clock_starts_when_owner_is_published(monkeypatch):
    from rapid_mlx.chat_runtime import owner

    clock = [0.0]
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    leases = ClientLeases()
    clock[0] = 1.0  # Recovery/publication took longer than the idle timeout.
    assert leases.idle(0.3, False)
    leases.touch()
    assert not leases.idle(0.3, False)
    assert not leases.clients
    clock[0] = 1.4
    assert leases.idle(0.3, False)


async def test_owner_resets_idle_after_recovery_before_serving(tmp_path, monkeypatch):
    import uvicorn

    from rapid_mlx.chat_runtime import owner, runtime
    from rapid_mlx.chat_runtime.store import Store

    clock = [0.0]
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    # Construct on the same fake clock used by recovery and publication.
    leases = ClientLeases()
    monkeypatch.setattr(owner, "ClientLeases", lambda: leases)
    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setenv("RAPID_MLX_CHAT_IDLE_SECONDS", "0.3")

    async def recover():
        clock[0] = 1.0

    async def close():
        pass

    async def open_store(_directory):
        return SimpleNamespace(
            store_uuid="test-store", recover_interrupted=recover, close=close
        )

    monkeypatch.setattr(Store, "open", open_store)
    monkeypatch.setattr(
        runtime, "Runtime", lambda store: SimpleNamespace(active_turns={}, close=close)
    )
    served = []

    class Server:
        should_exit = False

        def __init__(self, config):
            pass

        async def serve(self, *, sockets):
            assert (tmp_path / "slow-start/run/discovery.json").exists()
            assert clock[0] == 1.0
            assert not leases.idle(0.3, False)
            served.append(True)

    monkeypatch.setattr(uvicorn, "Server", Server)
    await owner.run_owner("slow-start")
    assert served == [True]
    assert not (tmp_path / "slow-start/run/discovery.json").exists()


@pytest.mark.parametrize(
    "authorization,status,remains_idle",
    [("Bearer test-token", 200, False), ("", 401, True), ("Bearer wrong", 401, True)],
)
def test_only_authenticated_requests_reset_idle(
    monkeypatch, authorization, status, remains_idle
):
    from fastapi.testclient import TestClient

    from rapid_mlx.chat_runtime import owner
    from rapid_mlx.chat_runtime.api import PREFIX, create_app

    clock = [0.0]
    # Patch only the leases' clock, leaving TestClient's event loop clock alone.
    monkeypatch.setattr(owner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    leases = ClientLeases()
    app = create_app(
        SimpleNamespace(schema_version=SCHEMA_VERSION, store_uuid="test-store"),
        SimpleNamespace(),
        "test-token",
        leases=leases,
    )
    with TestClient(app) as client:
        clock[0] = 1.0
        assert leases.idle(0.3, False)
        response = client.get(
            PREFIX + "/capabilities", headers={"Authorization": authorization}
        )
        assert response.status_code == status
        assert leases.idle(0.3, False) is remains_idle
        assert not leases.clients  # Discovery counts even before POST /clients.
        clock[0] = 1.4
        assert leases.idle(0.3, False)


@pytest.mark.parametrize("profile", ["../other", "a/b", "", ".hidden", "a" * 65])
def test_profile_rejects_path_traversal(profile):
    with pytest.raises(ValueError):
        profile_directory(profile)
