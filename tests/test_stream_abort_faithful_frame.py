# SPDX-License-Identifier: MIT
"""#3564: a generation-time engine abort that escapes AFTER the streaming
response has committed its headers must still reach the client as a terminal
SSE error frame carrying the SAME stable ``error.code`` the pre-commit path
returns as an HTTP 503 — so the Desktop GUI maps it to a curated failure card
instead of a silent truncation or the generic "Internal error during
streaming" mask.

These exercise ``_disconnect_guard`` directly (no engine, no mlx): feed it a
generator that yields one committed chunk and then raises
``InferenceAbortedError`` with a classified ``error_kind``, and assert the
frames it emits. Runs on the Linux lane (pure-Python).
"""

import json

import pytest

from rapid_mlx.request import InferenceAbortedError
from rapid_mlx.service.helpers import _disconnect_guard

# A raw exception message that MUST NOT leak into any client-facing frame.
_SECRET = "/Users/secret/prompt-fragment-and-internal-path.safetensors"


class _ConnectedRequest:
    """A request that never reports a client disconnect."""

    async def is_disconnected(self) -> bool:
        return False


def _collect_frames(error_kind, raw_message: str) -> list[str]:
    async def _run() -> list[str]:
        async def _gen():
            # The role delta commits the SSE response (headers on the wire).
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            # The engine aborts mid-generation AFTER commit.
            raise InferenceAbortedError(raw_message, error_kind=error_kind)

        frames: list[str] = []
        # keepalive_seconds=0 disables heartbeats so only the real frames show.
        async for frame in _disconnect_guard(
            _gen(),
            _ConnectedRequest(),
            poll_interval=0.01,
            keepalive_seconds=0,
        ):
            frames.append(frame)
        return frames

    import asyncio

    return asyncio.run(_run())


def _terminal_error(frames: list[str]) -> dict:
    """Extract the JSON ``error`` object from the terminal data frame."""
    # The last non-[DONE] data frame carries the error envelope.
    data_frames = [f for f in frames if f.startswith("data: ") and "[DONE]" not in f]
    payload = data_frames[-1][len("data: ") :].strip()
    return json.loads(payload)["error"]


def test_mid_stream_oom_abort_emits_faithful_insufficient_memory_frame():
    frames = _collect_frames(
        "insufficient_memory",
        f"kIOGPUCommandBufferCallbackErrorOutOfMemory {_SECRET}",
    )
    # The committed role chunk is delivered first.
    assert frames[0].startswith('data: {"choices"')
    # Terminal frame carries the stable code + curated message.
    error = _terminal_error(frames)
    assert error["code"] == "insufficient_memory"
    assert error["type"] == "server_error"
    assert "out of memory" in error["message"].lower()
    # And the stream is properly terminated for OpenAI-compatible clients.
    assert frames[-1] == "data: [DONE]\n\n"


def test_mid_stream_generic_abort_emits_faithful_engine_aborted_frame():
    frames = _collect_frames(
        "engine_aborted",
        f"RuntimeError: scheduler stepped on a landmine {_SECRET}",
    )
    error = _terminal_error(frames)
    assert error["code"] == "engine_aborted"
    assert error["type"] == "server_error"
    assert "try again" in error["message"].lower()
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.parametrize("error_kind", ["insufficient_memory", "engine_aborted"])
def test_mid_stream_abort_frame_never_leaks_raw_exception_text(error_kind):
    frames = _collect_frames(error_kind, f"boom {_SECRET}")
    blob = "".join(frames)
    # The raw path / prompt fragment must never cross the sanitisation
    # boundary onto the wire — only the fixed per-code message is allowed.
    assert _SECRET not in blob
    assert "boom" not in blob


def test_mid_stream_text_lane_abort_reclassifies_oom_from_message():
    """The text lane (engine_core) leaves ``error_kind`` None on an engine
    abort; the frame builder must re-derive the category from the message,
    exactly as the non-streaming route does — so a text-model generation-time
    OOM is ALSO faithful mid-stream, not just the pre-classified MLLM lane."""
    frames = _collect_frames(
        None,
        f"Metal out of memory while allocating {_SECRET}",
    )
    error = _terminal_error(frames)
    assert error["code"] == "insufficient_memory"
    assert frames[-1] == "data: [DONE]\n\n"


def test_mid_stream_text_lane_unclassified_abort_falls_back_to_engine_aborted():
    frames = _collect_frames(None, f"some non-memory engine fault {_SECRET}")
    error = _terminal_error(frames)
    assert error["code"] == "engine_aborted"
    assert _SECRET not in "".join(frames)


@pytest.mark.parametrize(
    "error_kind, raw_message, expected_code",
    [
        # A model-replacement cancellation: NOT an engine fault, so both the
        # pre-commit HTTP 503 and the post-commit SSE frame must say
        # ``model_replacement`` (codex MAJOR — timing changed this before).
        ("lifecycle", "primary model replaced mid-flight", "model_replacement"),
        # A pre-classified generation-time OOM (MLLM lane).
        (
            "insufficient_memory",
            "kIOGPUCommandBufferCallbackErrorOutOfMemory",
            "insufficient_memory",
        ),
        # A pre-classified transient engine crash.
        ("engine_aborted", "scheduler stepped on a landmine", "engine_aborted"),
    ],
)
def test_precommit_http_and_postcommit_sse_envelopes_agree(
    error_kind, raw_message, expected_code
):
    """The client must see the SAME category code + curated message whether an
    abort escapes BEFORE the streaming response commits (mapped to an HTTP 503)
    or AFTER (a terminal SSE frame). Timing must never change what a client
    sees (#3564)."""
    from rapid_mlx.request import InferenceAbortedError
    from rapid_mlx.routes.chat import _inference_aborted_http_exception

    sse_error = _terminal_error(_collect_frames(error_kind, raw_message))
    http_error = _inference_aborted_http_exception(
        InferenceAbortedError(raw_message, error_kind=error_kind)
    ).detail["error"]

    assert sse_error["code"] == expected_code
    # The HTTP body additionally carries ``param``; the SSE frame omits it —
    # neither is a client-facing category signal.
    assert http_error["code"] == sse_error["code"]
    assert http_error["message"] == sse_error["message"]
