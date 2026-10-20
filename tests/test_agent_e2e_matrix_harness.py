"""The release harness must not silently invent request or config evidence."""

import asyncio
import json

from scripts.agent_e2e.instrument.sitecustomize import _Tap
from scripts.agent_e2e.run import request_evidence, snapshot
from scripts.agent_e2e.seed_home import FILES


def test_snapshot_only_tracks_preexisting_user_config(tmp_path):
    path = tmp_path / ".codex/config.toml"
    path.parent.mkdir()
    path.write_text(FILES[".codex/config.toml"])
    before = snapshot(tmp_path, "codex")
    (path.parent / "session.json").write_text("{}")
    (tmp_path / ".rapid-mlx").mkdir()
    assert snapshot(tmp_path, "codex") == before
    path.write_text(path.read_text() + "# changed\n")
    assert snapshot(tmp_path, "codex") != before


def test_request_evidence_never_substitutes_config_model(tmp_path):
    assert request_evidence(None, "pi", 0, 100)["model_requested"] is None
    log = tmp_path / "requests.jsonl"
    log.write_text(
        json.dumps(
            {
                "agent": "pi",
                "time": 10,
                "path": "/v1/chat/completions",
                "model": "actual-request",
                "status": 200,
                "prompt_tokens": 1234,
            }
        )
        + "\n"
    )
    assert request_evidence(log, "pi", 0, 100) == {
        "model_requested": "actual-request",
        "auth_accepted": True,
        "first_prompt_tokens": 1234,
    }
    log.write_text(
        json.dumps(
            {
                "agent": "codex",
                "time": 10,
                "path": "/v1/responses",
                "model": "actual-request",
                "status": 200,
                "prompt_tokens": 88,
            }
        )
        + "\n"
    )
    assert request_evidence(log, "codex", 0, 100)["model_requested"] == "actual-request"


def test_metadata_tap_forwards_request_and_response_unchanged(tmp_path, monkeypatch):
    log = tmp_path / "requests.jsonl"
    current = tmp_path / "agent"
    current.write_text("pi\n")
    monkeypatch.setenv("AGENT_E2E_REQUEST_LOG", str(log))
    monkeypatch.setenv("AGENT_E2E_CURRENT_AGENT", str(current))
    original = b'{"model":"actual-request","messages":[{"content":"private prompt"}]}'
    response = b'{"usage":{"prompt_tokens":456},"choices":[]}'
    seen = []
    sent = []

    async def app(scope, receive, send):
        seen.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": response})

    async def receive():
        return {"type": "http.request", "body": original, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(
        _Tap(app)({"type": "http", "path": "/v1/chat/completions"}, receive, send)
    )
    assert seen[0]["body"] == original
    assert sent[-1]["body"] == response
    logged = log.read_text()
    assert "private prompt" not in logged
    event = json.loads(logged)
    assert event["agent"] == "pi"
    assert event["model"] == "actual-request"
    assert event["prompt_tokens"] == 456
    assert event["status"] == 200
