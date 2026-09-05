# SPDX-License-Identifier: Apache-2.0
"""The ``/api/connectors`` surface.

The engine's MCP routes are faked: these assert what this package composes
and refuses, not what the engine answers.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient
from test_app import AUTH, JSON_CT, TOKEN, FakeCatalog, FakeEngine

from rmlx_web import app as app_module
from rmlx_web.app import WebConfig, create_app
from rmlx_web.connectors import ConnectorStore
from rmlx_web.supervisor import ChildState


@pytest.fixture
def store(tmp_path):
    return ConnectorStore(
        config_path=tmp_path / "mcp.json",
        settings_path=tmp_path / "rmlx-web.json",
    )


def build(store, engine=None):
    return TestClient(
        create_app(
            WebConfig(
                token=TOKEN,
                engine=engine or FakeEngine(),
                catalog=FakeCatalog(),
                connectors=store,
            )
        )
    )


def test_unauthenticated_non_browser_cannot_mutate_connector_state(store):
    """No Origin is normal for scripts, so the bearer is the security boundary."""
    with build(store) as client:
        add = client.post(
            "/api/connectors/servers",
            headers=JSON_CT,
            json=stdio_payload(command="npx", args=["-y", "attacker-package"]),
        )
        enable = client.post(
            "/api/connectors/settings",
            headers=JSON_CT,
            json={"enabled": True},
        )

    assert add.status_code == 401
    assert enable.status_code == 401
    assert store.servers == []
    assert store.is_enabled is False


def fake_engine_mcp(
    monkeypatch, *, servers=None, tools=None, configured=True, error=None
):
    """Answer the engine's MCP reads. ``None`` means the route 404s."""

    async def fake_get(client, *, base_url, path, api_key, params=None, timeout=10.0):
        request = httpx.Request("GET", f"{base_url}{path}")
        if path == "/v1/mcp/servers":
            return httpx.Response(
                200,
                json={
                    "servers": servers if servers is not None else [],
                    "error": error,
                    "configured": configured,
                },
                request=request,
            )
        if path == "/v1/mcp/tools":
            return httpx.Response(
                200,
                json={"tools": tools if tools is not None else []},
                request=request,
            )
        return httpx.Response(404, json={}, request=request)

    monkeypatch.setattr(app_module.proxy, "proxy_get", fake_get)


def stdio_payload(name="fs", command="npx", **extra):
    return {"server": {"name": name, "command": command, **extra}}


class TestSnapshot:
    def test_it_answers_without_an_engine(self, store, monkeypatch):
        # The config is renderable whether or not a model is running — most
        # of a model switch is an unreachable engine.
        with build(store, FakeEngine(state=ChildState.STOPPED)) as client:
            body = client.get("/api/connectors", headers=AUTH).json()

        assert body["enabled"] is False
        assert body["servers"] == []
        assert body["engine_running"] is False

    def test_the_engine_is_not_polled_while_connectors_are_off(
        self, store, monkeypatch
    ):
        called = []

        async def fake_get(client, **kwargs):
            called.append(kwargs["path"])
            raise AssertionError("should not be reached")

        monkeypatch.setattr(app_module.proxy, "proxy_get", fake_get)

        with build(store) as client:
            client.get("/api/connectors", headers=AUTH)

        assert called == []

    def test_server_rows_carry_the_engines_status(self, store, monkeypatch):
        store.set_enabled(True)
        store.upsert_payload = None
        fake_engine_mcp(
            monkeypatch,
            servers=[
                {
                    "name": "fs",
                    "state": "connected",
                    "transport": "stdio",
                    "tools_count": 3,
                    "error": None,
                }
            ],
        )
        with build(store) as client:
            client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json=stdio_payload(),
            )
            body = client.get("/api/connectors", headers=AUTH).json()

        assert body["engine_servers"][0]["state"] == "connected"
        assert body["engine_reachable"] is True

    def test_an_illegal_tool_name_is_never_advertised(self, store, monkeypatch):
        # A connector names its own tools, so nothing bounds the composite.
        # A name the model cannot emit reads as "that tool does nothing".
        store.set_enabled(True)
        fake_engine_mcp(
            monkeypatch,
            tools=[
                {"name": "fs__read", "description": "d", "server": "fs"},
                {"name": "bad name__read", "description": "d", "server": "bad name"},
                {"name": "x" * 65, "description": "d", "server": "fs"},
            ],
        )
        with build(store) as client:
            body = client.get("/api/connectors", headers=AUTH).json()

        assert [tool["name"] for tool in body["tools"]] == ["fs__read"]


class TestRestartBanner:
    def _armed(self, store, monkeypatch, *, configured):
        store.set_enabled(True)
        fake_engine_mcp(monkeypatch, configured=configured)
        with build(store) as client:
            client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json=stdio_payload(),
            )
            return client.get("/api/connectors", headers=AUTH).json()

    def test_a_child_with_no_config_needs_a_restart(self, store, monkeypatch):
        # `--mcp-config` is read once at spawn, so a child started before the
        # master switch cannot pick connectors up.
        assert self._armed(store, monkeypatch, configured=False)["needs_restart"]

    def test_a_configured_child_does_not(self, store, monkeypatch):
        assert not self._armed(store, monkeypatch, configured=True)["needs_restart"]

    def test_it_is_derived_not_recorded(self, store, monkeypatch):
        # It was `@State` on the Mac and that was wrong: switching tabs reset
        # it while the condition it described still held.
        self._armed(store, monkeypatch, configured=False)
        fake_engine_mcp(monkeypatch, configured=False)
        with build(store) as client:
            again = client.get("/api/connectors", headers=AUTH).json()
        assert again["needs_restart"] is True

    def test_nothing_enabled_never_raises_it(self, store, monkeypatch):
        # `launch_config_path` intentionally stays None with no enabled
        # server, so a restart could not clear the banner it raised.
        store.set_enabled(True)
        fake_engine_mcp(monkeypatch, configured=False)
        with build(store) as client:
            client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json=stdio_payload(enabled=False),
            )
            body = client.get("/api/connectors", headers=AUTH).json()

        assert body["needs_restart"] is False


class TestWrites:
    def test_adding_a_connector_reloads_the_engine(self, store, monkeypatch):
        reloaded = []

        async def fake_post_query(client, *, base_url, path, api_key, **kwargs):
            reloaded.append(path)
            return httpx.Response(
                200,
                json={"servers": [], "configured": True},
                request=httpx.Request("POST", f"{base_url}{path}"),
            )

        monkeypatch.setattr(app_module.proxy, "proxy_post_query", fake_post_query)
        fake_engine_mcp(monkeypatch)
        store.set_enabled(True)

        with build(store) as client:
            response = client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json=stdio_payload(),
            )

        # A reload is what makes an edit apply without a model restart.
        assert reloaded == ["/v1/mcp/reload"]
        assert [s["name"] for s in response.json()["servers"]] == ["fs"]

    def test_an_invalid_connector_is_refused_with_its_reason(self, store):
        with build(store) as client:
            response = client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json={"server": {"name": "my server", "command": "npx"}},
            )

        assert response.status_code == 400
        assert "letters, numbers" in response.json()["error"]["message"]

    def test_a_reconfiguration_revokes_that_servers_grants(self, store, monkeypatch):
        fake_engine_mcp(monkeypatch)
        with build(store) as client:
            client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json=stdio_payload(),
            )
            client.post(
                "/api/connectors/settings",
                headers={**AUTH, **JSON_CT},
                json={"tool": "fs__read", "grant": True},
            )
            client.post(
                "/api/connectors/settings",
                headers={**AUTH, **JSON_CT},
                json={"tool": "time__now", "grant": True},
            )

            body = client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json={**stdio_payload(command="uvx"), "replacing": "fs"},
            ).json()

        # Pointing `fs` at a different program must not inherit the consent
        # given to the old one — but `time` is untouched.
        assert body["granted_tools"] == ["time__now"]

    def test_removing_a_connector_revokes_its_grants(self, store, monkeypatch):
        fake_engine_mcp(monkeypatch)
        with build(store) as client:
            client.post(
                "/api/connectors/servers",
                headers={**AUTH, **JSON_CT},
                json=stdio_payload(),
            )
            client.post(
                "/api/connectors/settings",
                headers={**AUTH, **JSON_CT},
                json={"tool": "fs__read", "grant": True},
            )

            body = client.post(
                "/api/connectors/servers/remove",
                headers={**AUTH, **JSON_CT},
                json={"name": "fs"},
            ).json()

        assert body["servers"] == []
        assert body["granted_tools"] == []

    def test_removing_an_unknown_connector_is_a_404(self, store):
        with build(store) as client:
            response = client.post(
                "/api/connectors/servers/remove",
                headers={**AUTH, **JSON_CT},
                json={"name": "ghost"},
            )
        assert response.status_code == 404

    def test_the_master_switch_persists(self, store, monkeypatch):
        fake_engine_mcp(monkeypatch)
        with build(store) as client:
            body = client.post(
                "/api/connectors/settings",
                headers={**AUTH, **JSON_CT},
                json={"enabled": True},
            ).json()

        assert body["enabled"] is True
        assert store.is_enabled is True

    def test_a_per_tool_switch_persists(self, store, monkeypatch):
        fake_engine_mcp(monkeypatch)
        with build(store) as client:
            body = client.post(
                "/api/connectors/settings",
                headers={**AUTH, **JSON_CT},
                json={"tool": "fs__read", "tool_enabled": False},
            ).json()

        assert body["disabled_tools"] == ["fs__read"]


class TestRestart:
    def test_it_respawns_the_loaded_model(self, store):
        engine = FakeEngine(model="qwen3.5-9b-4bit")
        with build(store, engine) as client:
            response = client.post(
                "/api/connectors/restart", headers={**AUTH, **JSON_CT}, json={}
            )

        assert response.status_code == 200
        assert response.json()["model"] == "qwen3.5-9b-4bit"

    def test_it_refuses_with_no_model(self, store):
        engine = FakeEngine(state=ChildState.STOPPED, model=None)
        with build(store, engine) as client:
            response = client.post(
                "/api/connectors/restart", headers={**AUTH, **JSON_CT}, json={}
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "no_model"

    def test_it_refuses_in_attach_mode(self, store):
        # A restart kills a child this process does not own.
        engine = FakeEngine(can_switch=False)
        with build(store, engine) as client:
            response = client.post(
                "/api/connectors/restart", headers={**AUTH, **JSON_CT}, json={}
            )

        assert response.status_code == 409
        assert response.json()["error"]["type"] == "switching_disabled"

    def test_double_request_is_single_flight_and_shutdown_cancels_it(self, store):
        class BlockingEngine(FakeEngine):
            def __init__(self):
                super().__init__(model="qwen3.5-9b-4bit")
                self.entered = threading.Event()
                self.cancelled = threading.Event()

            async def start(self, model, *, modality="text"):
                self.started.append(model)
                self.entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

        engine = BlockingEngine()
        with build(store, engine) as client:
            first = client.post(
                "/api/connectors/restart", headers={**AUTH, **JSON_CT}, json={}
            )
            assert engine.entered.wait(timeout=10)
            duplicate = client.post(
                "/api/connectors/restart", headers={**AUTH, **JSON_CT}, json={}
            )
            status = client.get("/api/status", headers=AUTH)

            assert first.status_code == 200
            assert duplicate.status_code == 200
            assert engine.started == ["qwen3.5-9b-4bit"]
            assert status.json()["state"] == "starting"

        assert engine.cancelled.is_set()


class TestExecute:
    def _run(self, store, monkeypatch, upstream, body=None):
        async def fake_post(self, url, **kwargs):
            return upstream(url, kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        with build(store) as client:
            return client.post(
                "/api/connectors/execute",
                headers={**AUTH, **JSON_CT},
                json=body or {"name": "fs__read", "arguments": '{"path":"/tmp"}'},
            )

    def test_a_result_is_flattened_for_the_model(self, store, monkeypatch):
        store.set_enabled(True)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: httpx.Response(
                200,
                json={"tool_name": "fs__read", "content": "hello", "is_error": False},
                request=httpx.Request("POST", url),
            ),
        )
        assert response.json() == {"content": "hello", "is_error": False}

    def test_an_empty_result_says_so(self, store, monkeypatch):
        # Handing the model "" invites it to invent the answer.
        store.set_enabled(True)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: httpx.Response(
                200,
                json={"tool_name": "fs__read", "content": "", "is_error": False},
                request=httpx.Request("POST", url),
            ),
        )
        assert "no content" in response.json()["content"]

    def test_a_structured_result_is_serialised(self, store, monkeypatch):
        store.set_enabled(True)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: httpx.Response(
                200,
                json={"tool_name": "t", "content": {"b": 2, "a": 1}, "is_error": False},
                request=httpx.Request("POST", url),
            ),
        )
        assert response.json()["content"] == '{"a": 1, "b": 2}'

    def test_a_disabled_tool_is_refused_here_too(self, store, monkeypatch):
        # Defence in depth: the page filters the advertised list, but that
        # does not stop a malformed model emitting the name anyway.
        store.set_enabled(True)
        store.set_tool_enabled("fs__read", False)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: pytest.fail("the engine must not be reached"),
        )
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "tool_disabled"

    def test_nothing_runs_while_connectors_are_off(self, store, monkeypatch):
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: pytest.fail("the engine must not be reached"),
        )
        assert response.status_code == 409
        assert response.json()["error"]["type"] == "connectors_disabled"

    def test_the_engines_operator_language_is_replaced(self, store, monkeypatch):
        # "Start server with --mcp-config" names a flag a phone user has no
        # way to pass.
        store.set_enabled(True)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: httpx.Response(
                503,
                json={"detail": "MCP not configured. Start server with --mcp-config"},
                request=httpx.Request("POST", url),
            ),
        )
        assert "--mcp-config" not in response.json()["error"]["message"]
        assert "restart" in response.json()["error"]["message"].lower()

    def test_a_sandbox_refusal_is_passed_through(self, store, monkeypatch):
        # It names the pattern that blocked the tool, which nothing composed
        # here would know.
        store.set_enabled(True)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: httpx.Response(
                400,
                json={"detail": "Tool 'shell' matches high-risk pattern 'shell'"},
                request=httpx.Request("POST", url),
            ),
        )
        assert "high-risk pattern" in response.json()["error"]["message"]

    def test_arguments_reach_the_engine_as_an_object(self, store, monkeypatch):
        store.set_enabled(True)
        captured = {}

        def upstream(url, kwargs):
            captured.update(kwargs.get("json") or {})
            return httpx.Response(
                200,
                json={"tool_name": "t", "content": "ok", "is_error": False},
                request=httpx.Request("POST", url),
            )

        self._run(store, monkeypatch, upstream)
        assert captured["arguments"] == {"path": "/tmp"}
        assert captured["tool_name"] == "fs__read"

    def test_empty_arguments_mean_a_no_arg_tool(self, store, monkeypatch):
        store.set_enabled(True)
        captured = {}

        def upstream(url, kwargs):
            captured.update(kwargs.get("json") or {})
            return httpx.Response(
                200,
                json={"tool_name": "t", "content": "ok", "is_error": False},
                request=httpx.Request("POST", url),
            )

        self._run(
            store, monkeypatch, upstream, body={"name": "time__now", "arguments": ""}
        )
        assert captured["arguments"] == {}

    def test_non_object_arguments_come_back_as_a_tool_error(self, store, monkeypatch):
        # The model wrote them, so this answers the model rather than
        # failing the request.
        store.set_enabled(True)
        response = self._run(
            store,
            monkeypatch,
            lambda url, kwargs: pytest.fail("the engine must not be reached"),
            body={"name": "fs__read", "arguments": "[1, 2]"},
        )
        assert response.status_code == 200
        assert response.json()["is_error"] is True
