"""Current-turn image projection and capability admission contracts."""

import base64
import json

import httpx
import pytest
from pydantic_ai.messages import BinaryContent, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from rapid_mlx.chat_runtime.history import _project_messages
from rapid_mlx.chat_runtime.runtime import ImageCapabilityError, Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store
from rapid_mlx.chat_runtime.token_budget import (
    IMAGE_TOKEN_ESTIMATE,
    TokenCounter,
    count_payload,
)


async def test_image_turn_and_replay(tmp_path):
    store = await Store.open(tmp_path)
    conversation = await store.create_conversation()
    image = b"private-image-bytes"
    attachment = await store.put_attachment(
        conversation["id"], "x.png", image, "image/png"
    )
    seen = []

    async def stream(messages, info):
        seen.extend(messages)
        yield "done"

    runtime = Runtime(
        store, model=FunctionModel(stream_function=stream), web_enabled=False
    )
    runtime.clients._http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"data": [{"id": "local", "capabilities": ["vision"]}]}
            )
        )
    )
    request = TurnRequest(
        turn_id="one",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="look",
        attachment_ids=[attachment["id"]],
        inference_url="http://localhost:1234",
        model="local",
        context_tokens=8192,
        max_tokens=256,
    )
    try:
        await runtime.submit(request)
        events = [e async for e in runtime.events("one") if e]
        prompt = next(
            p
            for m in seen
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, list)
        )
        assert prompt.content[0] == "look"
        assert isinstance(prompt.content[1], BinaryContent)
        assert (
            prompt.content[1].data == image
            and prompt.content[1].media_type == "image/png"
        )
        assert (await TokenCounter().measure(seen))[
            "used_tokens"
        ] >= IMAGE_TOKEN_ESTIMATE
        assert base64.b64encode(image).decode() not in json.dumps(count_payload(seen))
        rows = await store.branch_messages(conversation["branch_id"])
        replay = _project_messages(
            rows, attachments={attachment["id"]: ("image/png", image)}
        )
        assert any(
            "[Earlier image: x.png; not included]" in p.content
            for m in replay
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        assert base64.b64encode(image).decode() not in json.dumps(rows) + str(events)
        manager = await runtime._context_manager(request)
        assert attachment["id"] not in manager.deps.attachment_ids
        branch = await store.regenerate(conversation["branch_id"], "regen")
        request.turn_id, request.branch_id, request.expected_revision, request.mode = (
            "two",
            branch["id"],
            branch["revision"],
            "regenerate",
        )
        seen.clear()
        from rapid_mlx.chat_runtime.tools.web import WebTools

        runtime.web_tools = WebTools()
        await runtime.submit(request)
        _ = [e async for e in runtime.events("two") if e]
        assert any(
            isinstance(item, BinaryContent) and item.data == image
            for m in seen
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, list)
            for item in p.content
        )
    finally:
        await runtime.close()
        await store.close()


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"data": [{"id": "local", "capabilities": []}]}, "image_model_unsupported"),
        ({"data": []}, "image_capability_unknown"),
    ],
)
async def test_reject_before_mutation(tmp_path, payload, code):
    store = await Store.open(tmp_path)
    conversation = await store.create_conversation()
    attachment = await store.put_attachment(
        conversation["id"], "x.png", b"image", "image/png"
    )
    runtime = Runtime(store, web_enabled=False)
    runtime.clients._http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    )
    request = TurnRequest(
        turn_id="one",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="look",
        attachment_ids=[attachment["id"]],
        inference_url="http://localhost:1234",
        model="local",
    )
    try:
        with pytest.raises(ImageCapabilityError) as error:
            await runtime.submit(request)
        assert error.value.code == code
        assert await store.branch_messages(conversation["branch_id"]) == []
    finally:
        await runtime.close()
        await store.close()


async def test_mixed_projection_and_synthesis():
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.models import ModelRequestParameters

    from rapid_mlx.chat_runtime.context import fit_synthesis
    from rapid_mlx.chat_runtime.token_budget import text_value

    data = b"pixels"
    row = {
        "id": "a",
        "created_at": "2024-01-01T00:00:00+00:00",
        "role": "user",
        "parts": [
            {"type": "text", "text": "hello"},
            {"type": "attachment", "name": "doc", "attachment_id": "d"},
            {"type": "attachment", "name": "image", "attachment_id": "i"},
        ],
    }
    history = _project_messages(
        [row],
        current_id="a",
        attachments={"d": ("text/plain", b"doc"), "i": ("image/png", data)},
    )
    part = history[0].parts[0]
    assert part.content[1].startswith("Attached UTF-8 document")
    assert text_value(part.content[2]) == ""
    fitted, _, _ = await fit_synthesis(
        history + [ModelRequest(parts=[UserPromptPart("notice")])],
        ModelRequestParameters(),
        TokenCounter(),
        10000,
    )
    assert part in fitted[0].parts


async def test_invalid_capability_and_store_validation(tmp_path):
    from rapid_mlx.chat_runtime.model_client import ModelClients

    clients = ModelClients(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, json={"data": [{"id": "m", "capabilities": None}]}
                )
            )
        )
    )
    assert await clients.vision_capability("http://localhost:1234", "m") is None
    await clients.close()
    store = await Store.open(tmp_path)
    try:
        c = await store.create_conversation()
        for data in (b"x" * (1024 * 1024 + 1), b"\x00"):
            with pytest.raises(ValueError):
                await store.put_attachment(c["id"], "x", data)
    finally:
        await store.close()


async def test_capability_error_api(tmp_path):
    from rapid_mlx.chat_runtime.api import PREFIX, create_app

    store = await Store.open(tmp_path)
    c = await store.create_conversation()
    attachment = await store.put_attachment(c["id"], "x.png", b"x", "image/png")
    runtime = Runtime(store, web_enabled=False)
    runtime.clients._http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(503))
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(store, runtime, "secret")),
            base_url="http://test",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            response = await client.post(
                PREFIX + "/turns",
                json={
                    "turn_id": "t",
                    "conversation_id": c["id"],
                    "branch_id": c["branch_id"],
                    "expected_revision": 0,
                    "attachment_ids": [attachment["id"]],
                    "model": "m",
                    "inference_url": "http://localhost:1234",
                },
            )
            assert response.json()["detail"]["code"] == "image_capability_unknown"
    finally:
        await runtime.close()
        await store.close()


def test_extraction_worker_main(monkeypatch, capsys):
    import io
    import runpy
    import sys

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"hello")))
    monkeypatch.setattr(sys, "argv", ["worker", "a.txt"])
    runpy.run_module("rapid_mlx.chat_runtime.file_extraction", run_name="__main__")
    assert json.loads(capsys.readouterr().out)["text"] == "hello"


async def test_image_overflow_never_drops_pixels():
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.models import ModelRequestParameters

    from rapid_mlx.chat_runtime.context import (
        ContextSynthesisUnavailableError,
        fit_synthesis,
    )

    prompt = UserPromptPart(
        ["describe", BinaryContent(b"pixels", media_type="image/png")]
    )
    with pytest.raises(ContextSynthesisUnavailableError):
        await fit_synthesis(
            [ModelRequest(parts=[prompt])],
            ModelRequestParameters(),
            TokenCounter(),
            500,
        )
