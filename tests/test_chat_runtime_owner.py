# SPDX-License-Identifier: Apache-2.0
import concurrent.futures
import json
import os
import signal
import stat
import time

import httpx
import psutil
import pytest

from rapid_mlx.chat_runtime.owner import (
    CAPABILITIES,
    ClientLeases,
    discover,
    discover_or_start,
    profile_directory,
)
from rapid_mlx.headless_service.config import atomic_write


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
        assert first["schema_version"] == Store.schema_version == 4
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


@pytest.mark.parametrize("profile", ["../other", "a/b", "", ".hidden", "a" * 65])
def test_profile_rejects_path_traversal(profile):
    with pytest.raises(ValueError):
        profile_directory(profile)
