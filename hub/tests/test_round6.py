"""Round 6: startup must fail without clobbering an active Hub; fresh login."""

import asyncio
import json
import sys

import httpx
import pytest
from rapid_hub import __main__ as cli
from rapid_hub.app import Hub
from rapid_hub.supervisor import Supervisor
from test_hub import ROOT, client, make_hub
from test_services import reported


def test_port_validation_preserves_credentials_jobs_and_ownership(
    tmp_path, monkeypatch
):
    (tmp_path / "admin-token").write_text("original")
    jobs = json.dumps(dict(run=dict(state="running")))
    (tmp_path / "jobs.json").write_text(jobs)

    def occupied(self, ident, port):
        raise ValueError("occupied plugin port")

    monkeypatch.setattr(Supervisor, "check_port", occupied)
    with pytest.raises(ValueError, match="occupied"):
        Hub({}, tmp_path, ROOT / "apps", plugin_ports={"research-loop": 54321})
    assert (tmp_path / "admin-token").read_text() == "original"
    assert (tmp_path / "jobs.json").read_text() == jobs
    assert not (tmp_path / ".owner.lock").exists()
    with pytest.raises(ValueError, match="Unknown plugin"):
        Hub({}, tmp_path, ROOT / "apps", plugin_ports={"missing": 54321})
    assert (tmp_path / "admin-token").read_text() == "original"


def test_data_directory_exclusive_ownership_and_release(tmp_path):
    async def run():
        hub = make_hub(tmp_path, reported)
        original = (tmp_path / "admin-token").read_bytes()
        jobs = (tmp_path / "jobs.json").read_bytes()
        try:
            with pytest.raises(ValueError, match="already owned"):
                make_hub(tmp_path, reported)
            assert (tmp_path / "admin-token").read_bytes() == original
            assert (tmp_path / "jobs.json").read_bytes() == jobs
        finally:
            await hub.client.aclose()
            hub.release_ownership()
        next_hub = make_hub(tmp_path, reported)
        assert next_hub.admin != hub.admin
        await next_hub.client.aclose()
        next_hub.release_ownership()

    asyncio.run(run())


def test_login_nonce_admin_only_single_use_and_rotates(tmp_path):
    async def run():
        hub = make_hub(tmp_path, reported)
        async with client(hub) as plugin:
            assert (await plugin.post("/hub/v1/login-nonce")).status_code == 403
            assert (
                await plugin.post("/hub/v1/login-nonce", headers={"Authorization": ""})
            ).status_code == 401
        async with client(hub, hub.admin) as admin:
            old = hub.login
            response = await admin.post("/hub/v1/login-nonce")
            url = response.json()["url"]
            assert response.status_code == 200 and url.startswith(
                hub.dashboard_url + "/login?nonce="
            )
            assert hub.admin not in url and hub.login != old
            assert (await admin.get("/login?nonce=" + old)).status_code == 401
            assert (await admin.get(url)).status_code == 303
            assert (await admin.get(url)).status_code == 401
            again = (await admin.post("/hub/v1/login-nonce")).json()["url"]
            assert again != url and (await admin.get(again)).status_code == 303
        await hub.client.aclose()
        hub.release_ownership()

    asyncio.run(run())


def test_login_cli_token_file_and_instance_check(tmp_path, monkeypatch, capsys):
    token = tmp_path / "admin-token"
    token.write_text("admin-secret")
    token.chmod(0o600)
    (tmp_path / "connection.json").write_text(
        json.dumps(dict(url="http://127.0.0.1:54321", instance_id="live"))
    )
    calls = []

    def get(url, **kwargs):
        assert url.endswith("/hub/v1/instance")
        return httpx.Response(200, json=dict(instance_id="live"))

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200,
            json=dict(url="http://hub.localhost:54321/login?nonce=fresh"),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(cli.httpx, "get", get)
    monkeypatch.setattr(cli.httpx, "post", post)
    monkeypatch.setattr(
        sys, "argv", ["rapid-hub", "login", "--data-dir", str(tmp_path)]
    )
    cli.main()
    assert (
        capsys.readouterr().out.strip()
        == "http://hub.localhost:54321/login?nonce=fresh"
    )
    assert calls[0][1]["headers"]["Authorization"] == "Bearer admin-secret"
    assert "admin-secret" not in calls[0][0]
    token.chmod(0o644)
    with pytest.raises(SystemExit):
        cli.main()
    assert len(calls) == 1
    token.chmod(0o600)
    (tmp_path / "connection.json").write_text(
        json.dumps(dict(url="http://127.0.0.1:54321", instance_id="stale"))
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert len(calls) == 1
