"""The evidence collector must not hide duplicate or corrupt deltas."""

from .harness import norm


def test_messages_duplicate_start_is_not_overwritten():
    packet = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "name": "read", "input": {"x": 1}},
    }
    result = norm([packet, packet], "messages", True)
    assert result["calls"] == [["readread", '{"x": 1}{"x": 1}']]


def test_responses_uses_actual_argument_deltas():
    packets = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "read", "arguments": ""},
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "broken",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "read", "arguments": "{}"},
        },
    ]
    assert norm(packets, "responses", True)["calls"] == [["read", "broken"]]


def test_chat_duplicate_start_is_not_overwritten():
    packet = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "function": {"name": "read", "arguments": "{}"}}
                    ]
                }
            }
        ]
    }
    assert norm([packet, packet], "chat", True)["calls"] == [["readread", "{}{}"]]


def test_responses_terminal_control_matches_stream():
    from .harness import route

    raw = '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
    result = route(None, "hermes", raw, "responses", True, capture=True)
    terminal = [
        p["response"] for p in result["packets"] if p["type"] == "response.completed"
    ]
    assert len(terminal) == 1
    projection = norm(terminal, "responses", False)
    assert projection == {key: result[key] for key in ("content", "reasoning", "calls")}
    done = [
        p["item"] for p in result["packets"] if p["type"] == "response.output_item.done"
    ]
    assert norm([{"output": done}], "responses", False) == projection
