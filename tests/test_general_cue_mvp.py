import json
from types import SimpleNamespace

import pytest

from tools.general_cue_mvp.rapid_general_cue import (
    JevConfig,
    JevDecision,
    JevGuardBackend,
    _make_rapid_compatible,
    _parse_arguments,
    build_metacua_arguments,
)


def test_forwards_bounded_local_muse_session(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    args = build_metacua_arguments(
        [
            "--goal",
            "Open Calculator and compute 17 times 23",
            "--api-key",
            "local-secret",
        ]
    )

    assert args[:2] == ["metacua", "agent"]
    assert args[args.index("--model") + 1] == "muse-glimmer-30b-4bit"
    assert args[args.index("--base-url") + 1] == "http://127.0.0.1:8000/v1"
    assert args[args.index("--max-steps") + 1] == "20"
    assert args[args.index("--max-images") + 1] == "5"
    assert args[args.index("--allow-bash") + 1] == "false"
    assert "--overlay" in args


def test_jev_arguments_stay_out_of_upstream_argv():
    args, config = _parse_arguments(
        [
            "--goal",
            "Open Calculator",
            "--api-key",
            "local-secret",
            "--jev-mode",
            "shadow",
            "--jev-api-key",
            "decision-secret",
            "--jev-execute-threshold",
            "0.7",
        ]
    )

    assert config == JevConfig(
        mode="shadow",
        url="http://127.0.0.1:8700/v1/systemone",
        api_key="decision-secret",
        execute_threshold=0.7,
        planner_max_output_tokens=768,
    )
    assert all(not item.startswith("--jev") for item in args)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8000/v1",
        "http://example.com:8000/v1",
        "http://localhost:8000/v1",
        "http://user:pass@127.0.0.1:8000/v1",
    ],
)
def test_rejects_non_loopback_or_credentialed_endpoints(url):
    with pytest.raises(SystemExit):
        build_metacua_arguments(
            ["--goal", "safe task", "--api-key", "local-secret", "--base-url", url]
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8700/v1/systemone",
        "http://example.com:8700/v1/systemone",
        "http://localhost:8700/v1/systemone",
        "http://user:pass@127.0.0.1:8700/v1/systemone",
    ],
)
def test_rejects_non_loopback_jev_endpoints(url):
    with pytest.raises(SystemExit):
        build_metacua_arguments(
            [
                "--goal",
                "safe task",
                "--api-key",
                "local-secret",
                "--jev-url",
                url,
            ]
        )


@pytest.mark.parametrize(
    "flag,value",
    [("--max-steps", "0"), ("--max-steps", "41"), ("--max-images", "6")],
)
def test_rejects_unbounded_history_and_steps(flag, value):
    with pytest.raises(SystemExit):
        build_metacua_arguments(
            ["--goal", "safe task", "--api-key", "local-secret", flag, value]
        )


class _FakeBackend:
    label = "fake muse"

    def initial_conversation(self, goal, screenshot):
        return [{"goal": goal}]

    def send(self, system, conversation):
        del system, conversation
        item = {
            "type": "function_call",
            "call_id": "call-1",
            "name": "computer.computer",
            "arguments": json.dumps({"action": "left_click", "coordinate": [400, 300]}),
        }
        return SimpleNamespace(
            assistant_items=[item],
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    name="computer.computer",
                    input={"action": "left_click", "coordinate": [400, 300]},
                )
            ],
            text="",
            thinking="inspect button",
            finish="tool_use",
            raw_response={"output": [item]},
        )

    def tool_result_items(self, runs, screenshot, notes=None):
        return [runs, screenshot, notes]


class _FakeGuard:
    def __init__(self, execute_probability):
        self.execute_probability = execute_probability
        self.state = None

    def assess(self, state):
        self.state = state
        return JevDecision(
            execute_probability=self.execute_probability,
            replan_probability=1 - self.execute_probability,
            latency_ms=12.5,
        )


class _FailingGuard:
    def assess(self, state):
        del state
        raise RuntimeError("offline")


def _wrapped(mode, probability, threshold=0.5):
    guard = _FakeGuard(probability)
    backend = JevGuardBackend(
        _FakeBackend(),
        guard,
        JevConfig(
            mode=mode,
            url="http://127.0.0.1:8700/v1/systemone",
            api_key=None,
            execute_threshold=threshold,
            planner_max_output_tokens=768,
        ),
    )
    backend.initial_conversation("click the correct button", SimpleNamespace())
    return backend, guard


def test_shadow_scores_without_changing_action():
    backend, guard = _wrapped("shadow", 0.2)
    result = backend.send("system", [{"image_url": "data:image/png;base64,secret"}])

    assert result.tool_calls[0].input["action"] == "left_click"
    assert result.raw_response["_rapid_jev"]["rejected"] is True
    assert guard.state["goal"] == "click the correct button"
    assert "secret" not in json.dumps(guard.state)


def test_guard_replaces_rejected_call_and_replay_item():
    backend, _ = _wrapped("guard", 0.2)
    result = backend.send("system", [])

    assert result.tool_calls[0].input == {"action": "screenshot"}
    assert json.loads(result.assistant_items[0]["arguments"]) == {
        "action": "screenshot"
    }


def test_guard_preserves_accepted_action():
    backend, _ = _wrapped("guard", 0.8)
    result = backend.send("system", [])

    assert result.tool_calls[0].input["action"] == "left_click"


def test_guard_failure_fails_open_and_records_error():
    config = JevConfig(
        mode="guard",
        url="http://127.0.0.1:8700/v1/systemone",
        api_key=None,
        execute_threshold=0.5,
        planner_max_output_tokens=768,
    )
    backend = JevGuardBackend(_FakeBackend(), _FailingGuard(), config)
    backend.initial_conversation("click the correct button", SimpleNamespace())

    result = backend.send("system", [])

    assert result.tool_calls[0].input["action"] == "left_click"
    assert result.raw_response["_rapid_jev"]["error"].endswith("offline")


def test_meta_tool_names_are_rewritten_for_openai_contract():
    class Backend:
        def _tools(self):
            return [
                {"type": "function", "name": "computer.computer"},
                {"type": "function", "name": "computer.stop"},
            ]

    backend = _make_rapid_compatible(Backend())

    assert [tool["name"] for tool in backend._tools()] == [
        "computer_computer",
        "computer_stop",
    ]


def test_screenshot_is_separate_from_function_output():
    class Backend:
        def _image_block(self, screenshot):
            return {"type": "input_image", "image_url": screenshot}

    run = SimpleNamespace(
        call_id="call-1",
        output="clicked",
        is_error=False,
    )
    backend = _make_rapid_compatible(Backend())

    items = backend.tool_result_items([run], "data:image/png;base64,image")

    assert items[0] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "clicked",
    }
    assert items[1]["type"] == "message"
    assert items[1]["content"][1]["type"] == "input_image"


def test_tool_turn_replay_drops_muse_channel_message():
    class Backend:
        def send(self, system, conversation):
            del system, conversation
            return SimpleNamespace(
                tool_calls=[SimpleNamespace(id="call-1")],
                assistant_items=[
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "to=self...assistant to=computer_computer",
                            }
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "computer_computer",
                    },
                ],
            )

    result = _make_rapid_compatible(Backend()).send("system", [])

    assert result.assistant_items == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "computer_computer",
        }
    ]
