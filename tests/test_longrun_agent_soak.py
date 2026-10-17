"""Response diagnostics for the live-server soak without loading a model."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts.longrun_agent_soak import Soak, diagnostic_repr, response_details


def make_soak(tmp_path):
    args = SimpleNamespace(
        output=tmp_path,
        duration=1,
        seed=1,
        timeout=5,
        concurrency=1,
        model="test",
        url="http://localhost",
    )
    return Soak(args)


async def close_soak(soak):
    await soak.client.aclose()
    soak.errors.close()
    soak.csv_file.close()


def read_error(tmp_path):
    return json.loads((tmp_path / "errors.jsonl").read_text().strip())


def test_blank_response_records_bounded_details(tmp_path):
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "content": " " * 600,
                    "reasoning_content": "r" * 2100,
                },
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 80},
    }

    async def run():
        soak = make_soak(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)
            )
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)
        return soak

    soak = asyncio.run(run())
    error = read_error(tmp_path)
    assert soak.totals["errors"] == 1
    assert error["error"] == "empty answer"
    assert error["finish_reason"] == "length"
    assert error["usage"] == body["usage"]
    assert error["content_length"] == 600
    assert error["reasoning_content"] == "r" * 2000
    assert error["content_repr"] == repr(" " * 600)[:500]
    assert error["request_message_count"] == 2


def test_no_choices_records_missing_response_fields(tmp_path):
    async def run():
        soak = make_soak(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert error["error"] == "no choices"
    assert error["finish_reason"] is None
    assert error["usage"] is None
    assert error["content_length"] == 0
    assert error["reasoning_content"] == ""
    assert error["content_repr"] == "''"
    assert error["request_message_count"] == 2


@pytest.mark.parametrize(
    ("followup_body", "expected_error"),
    [
        (
            {
                "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                "usage": {"completion_tokens": 1},
            },
            "empty tool follow-up answer",
        ),
        ({"choices": []}, "tool follow-up has no choices"),
        ({"choices": [{"message": {"content": "answer"}}]}, None),
    ],
)
def test_tool_followup_records_actual_request_message_count(
    tmp_path, followup_body, expected_error
):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "lookup_symbol",
                                            "arguments": '{"name":"parse_request"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(200, json=followup_body)

    async def run():
        soak = make_soak(tmp_path)
        soak.sequence = 4
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    if expected_error is None:
        assert calls == 2
        assert (tmp_path / "errors.jsonl").read_text() == ""
        return
    error = read_error(tmp_path)
    assert calls == 2
    assert error["error"] == expected_error
    assert error["stage"] == "tool_followup"
    assert error["finish_reason"] == (
        followup_body["choices"][0]["finish_reason"]
        if followup_body["choices"]
        else None
    )
    assert error["usage"] == followup_body.get("usage")
    assert error["request_message_count"] == 4


def test_incomplete_stream_records_received_content(tmp_path):
    async def run():
        soak = make_soak(tmp_path)
        soak.sequence = 1
        chunk = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "delta": {"content": "partial", "reasoning_content": "thinking"},
                }
            ],
            "usage": {"completion_tokens": 2},
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, text=f"data: {json.dumps(chunk)}\n\n"
                )
            )
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert error["error"] == "stream missing completion marker"
    assert error["finish_reason"] == "stop"
    assert error["content_length"] == 7
    assert error["content_repr"] == "'partial'"
    assert error["reasoning_content"] == "thinking"
    assert error["usage"] == {"completion_tokens": 2}


@pytest.mark.parametrize("failure", ["timeout", "http", "other"])
def test_request_exceptions_do_not_add_response_details(tmp_path, failure):
    async def run():
        soak = make_soak(tmp_path)

        async def fail(*args):
            if failure == "timeout":
                raise TimeoutError
            if failure == "http":
                response = httpx.Response(
                    503, request=httpx.Request("POST", soak.args.url)
                )
                raise httpx.HTTPStatusError(
                    "failure", request=response.request, response=response
                )
            raise RuntimeError("failure")

        soak.send = fail
        await soak.request(0)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert "finish_reason" not in error


@pytest.mark.parametrize("reasoning", [["r" * 3000], {"text": "r" * 3000}, 123, 0])
def test_nonstring_reasoning_preserves_empty_answer(tmp_path, reasoning):
    body = {"choices": [{"message": {"content": "", "reasoning_content": reasoning}}]}

    async def run():
        soak = make_soak(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)
            )
        ) as client:
            await soak.request(0, client=client)
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert error["error"] == "empty answer"
    assert error["reasoning_content"] == diagnostic_repr(reasoning, 2000)
    assert error["content_length"] == 0
    assert error["content_repr"] == "''"
    assert error["request_message_count"] == 2


@pytest.mark.parametrize(
    "finish_reason", ["f" * 3000, ["f" * 3000], 123], ids=["string", "list", "int"]
)
@pytest.mark.parametrize(
    ("usage", "expected_usage"),
    [
        (
            {
                "prompt_tokens": 31,
                "completion_tokens": 80,
                "total_tokens": 111,
                "prompt_tokens_details": {"payload": "u" * 10000},
                "arbitrary_tokens": 123,
                "input_tokens": True,
                "output_tokens": "u" * 10000,
                "cached_tokens": 2**10000,
            },
            {"prompt_tokens": 31, "completion_tokens": 80, "total_tokens": 111},
        ),
        (["u" * 10000], diagnostic_repr(["u" * 10000], 500)),
        ("u" * 10000, diagnostic_repr("u" * 10000, 500)),
        (0, "0"),
    ],
)
def test_oversized_response_metadata_is_bounded(finish_reason, usage, expected_usage):
    details = response_details(
        {"choices": [{"finish_reason": finish_reason}], "usage": usage}, 2
    )
    expected_reason = (
        finish_reason
        if isinstance(finish_reason, str)
        else diagnostic_repr(finish_reason, 200)
    )[:200]
    assert details["finish_reason"] == expected_reason
    assert details["usage"] == expected_usage
    assert len(json.dumps(details)) < 1000


@pytest.mark.parametrize(
    "value", [None, False, 0, 123, "invalid", [], [123], {"0": 123}]
)
def test_response_details_handles_malformed_containers(value):
    for body in (value, {"choices": value}, {"choices": [value]}):
        details = response_details(body, 2)
        assert details["content_length"] == 0
        assert details["content_repr"] == "''"
    details = response_details({"choices": [{"message": value}]}, 2)
    assert details["content_length"] == 0
    assert details["reasoning_content"] == ""


@pytest.mark.parametrize("content", [["c" * 3000], {"text": "c" * 3000}, 123, 0, False])
def test_response_details_serializes_nonstring_content(content):
    details = response_details({"choices": [{"message": {"content": content}}]}, 2)
    assert details["content_length"] == len(diagnostic_repr(content, 500))
    assert details["content_repr"] == diagnostic_repr(content, 500)


def test_response_details_handles_unrepresentable_json_values():
    value = 10**5000
    details = response_details(
        {
            "choices": [
                {
                    "finish_reason": value,
                    "message": {"content": value, "reasoning_content": value},
                }
            ],
            "usage": value,
        },
        2,
    )
    assert details["finish_reason"] == "<int>"
    assert details["usage"] == "<int>"
    assert details["content_length"] == 5
    assert details["content_repr"] == "<int>"
    assert details["reasoning_content"] == "<int>"


@pytest.mark.parametrize("shape", ["wide", "deep"])
def test_response_details_bounds_large_nonstring_values(tmp_path, shape):
    value = list(range(100_000))
    if shape == "deep":
        value = 0
        for _ in range(1000):
            value = {"nested": value}
        value = [value]
    details = response_details(
        {
            "choices": [
                {
                    "finish_reason": value,
                    "message": {"content": value, "reasoning_content": value},
                }
            ],
            "usage": value,
        },
        2,
    )

    async def run():
        soak = make_soak(tmp_path)
        soak.record(
            "nonstream", soak.start, 0, 0, "initial", "malformed", details=details
        )
        await close_soak(soak)

    asyncio.run(run())
    error = read_error(tmp_path)
    assert 0 < error["content_length"] == len(error["content_repr"]) <= 500
    for field, limit in (
        ("finish_reason", 200),
        ("usage", 500),
        ("content_repr", 500),
        ("reasoning_content", 2000),
    ):
        assert len(error[field]) <= limit
        assert "..." in error[field]
    assert len(json.dumps(error)) < 4000


@pytest.mark.parametrize(
    ("kind", "status", "chunks", "done", "expected_error", "disconnected"),
    [
        ("stream", 503, 0, False, "ValueError: HTTP 503: unavailable", False),
        ("stream", 200, None, False, "ValueError: stream chunk missing choices", False),
        ("stream", 200, 1, True, None, False),
        ("disconnect", 200, 3, False, None, True),
        (
            "disconnect",
            200,
            1,
            False,
            "disconnect stream ended before 3 chunks",
            False,
        ),
    ],
)
def test_stream_completion_and_failure_diagnostics(
    tmp_path, kind, status, chunks, done, expected_error, disconnected
):
    chunk = {"choices": [{"finish_reason": "stop", "delta": {"content": "answer"}}]}
    text = (
        "data: {}\n\n" if chunks is None else f"data: {json.dumps(chunk)}\n\n" * chunks
    )
    if done:
        text += "data: [DONE]\n\n"
    if status != 200:
        text = "unavailable"

    async def run():
        soak = make_soak(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, text=text)
            )
        ) as client:
            result = await soak.send(kind, {"stream": True, "messages": []}, client)
        await close_soak(soak)
        return result

    result = asyncio.run(run())
    assert result[0] is disconnected
    assert result[3] == expected_error
    if kind == "disconnect" and expected_error:
        assert result[5]["content_repr"] == "'answer'"
    elif expected_error:
        assert result[5] is None
    elif kind == "stream":
        assert result[4] == {"role": "assistant", "content": "answer"}
