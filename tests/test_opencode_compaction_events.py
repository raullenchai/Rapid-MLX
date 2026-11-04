# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the real-model OpenCode gate's event assertions."""

import json

import pytest

from tests.integrations.test_opencode_memory_compaction import (
    _assert_compaction_recovered,
)


def _events(*events):
    return "\n".join(json.dumps(event) for event in events)


def _finish(reason="stop"):
    return {"type": "step_finish", "part": {"type": "step-finish", "reason": reason}}


def _overflow(name="ContextOverflowError"):
    return {
        "type": "error",
        "error": {"name": name, "data": {"message": "Internal error during streaming"}},
    }


COMPACTION = {"type": "part", "part": {"type": "compaction"}}


@pytest.mark.parametrize(
    "recovery",
    [
        [COMPACTION],
        [_overflow(), _finish("tool-calls")],
        [_overflow()],
    ],
)
def test_recovery_ends_normally(recovery):
    _assert_compaction_recovered(
        _events({"type": "text", "part": {"text": "answer"}}, *recovery, _finish())
    )


@pytest.mark.parametrize("name", ["UnknownError", "APIError", "InternalError", None])
def test_non_context_errors_fail_even_after_compaction(name):
    with pytest.raises(AssertionError):
        _assert_compaction_recovered(_events(COMPACTION, _overflow(name), _finish()))


@pytest.mark.parametrize(
    "events",
    [
        [],
        [_finish()],
        [{"type": "text", "part": {"text": "compaction"}}, _finish()],
        [_finish(), _overflow()],
        [COMPACTION],
        [COMPACTION, _finish("tool-calls")],
        [COMPACTION, _finish("error")],
        [COMPACTION, _finish(), _finish("error")],
        [COMPACTION, _finish(), _overflow()],
        [_overflow(), _finish("error")],
    ],
)
def test_missing_recovery_or_normal_finish_fails(events):
    with pytest.raises(AssertionError):
        _assert_compaction_recovered(_events(*events))


def test_invalid_json_fails():
    with pytest.raises(json.JSONDecodeError):
        _assert_compaction_recovered("not json")


@pytest.mark.parametrize(
    "statuses",
    [
        [200],
        [400, 200],
        [400, 400, 400, 200],
        [503, 200],
        [400, 400, 400, 400, 200],
        [400],
    ],
)
def test_e2e_gate_with_synthetic_process(tmp_path, monkeypatch, statuses):
    from types import SimpleNamespace

    from tests.integrations import test_opencode_memory_compaction as gate

    server = tmp_path / "server.log"
    server.write_text("")
    config = {
        "provider": {
            "local": {
                "models": {gate.MODEL: {"limit": {"context": 900, "output": 100}}}
            }
        }
    }

    def run(command, **kwargs):
        if command[0] != "opencode":
            path = (
                tmp_path / "home" / ".rapid-mlx" / "connect" / "local" / "opencode.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(config))
        else:
            assert command[1:4] == ["run", "--format", "json"]
            server.write_text(
                "\n".join(
                    f'"POST /v1/chat/completions" {status}' for status in statuses
                )
            )
        return SimpleNamespace(
            stdout=_events(COMPACTION, _finish()), stderr="", returncode=0
        )

    monkeypatch.setattr(gate.subprocess, "run", run)
    monkeypatch.setattr(
        gate,
        "_models",
        lambda: {
            "data": [{"id": gate.MODEL, "max_model_len": 1000, "context_window": 2000}]
        },
    )
    if 503 in statuses or statuses.count(400) > 3 or statuses[-1] != 200:
        with pytest.raises(AssertionError):
            gate.test_opencode_compacts_instead_of_looping(server, tmp_path)
    else:
        gate.test_opencode_compacts_instead_of_looping(server, tmp_path)
