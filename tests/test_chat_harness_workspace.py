# SPDX-License-Identifier: Apache-2.0
"""Client-supplied workspace authority and document upload integration."""

import io
import zipfile
from types import SimpleNamespace

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel

from rapid_mlx.chat_client import ChatClient
from rapid_mlx.chat_harness_render import HarnessRenderer, tool_progress
from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Conflict, Store
from rapid_mlx.chat_state import TurnState


def turn(conversation, **kwargs):
    return TurnRequest(
        turn_id="first",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="Hello",
        model="fake",
        inference_url="http://127.0.0.1:18800",
        **kwargs,
    )


async def test_workspace_survives_reopen_and_never_defaults_to_owner_cwd(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    store = await Store.open(tmp_path / "store")
    try:
        old = await store.create_conversation()
        assert old["workspace"] is None
        conversation = await store.create_conversation(workspace=str(alias))
        assert conversation["workspace"] == str(workspace.resolve())
        with pytest.raises(ValueError, match="absolute"):
            await store.create_conversation(workspace="relative")
        with pytest.raises(ValueError, match="accessible"):
            await store.create_conversation(workspace=str(tmp_path / "missing"))
        await store.begin_turn(turn(old, workspace=str(workspace)).model_dump())
        await store.finish_turn("first", "cancelled")
        assert (await store.get_conversation(old["id"]))["workspace"] == str(
            workspace.resolve()
        )
        with pytest.raises(Conflict, match="workspace"):
            await store.begin_turn(
                turn(conversation, workspace=str(tmp_path))
                .model_copy(update={"turn_id": "other"})
                .model_dump()
            )
    finally:
        await store.close()
    store = await Store.open(tmp_path / "store")
    try:
        assert (await store.get_conversation(conversation["id"]))["workspace"] == str(
            workspace.resolve()
        )
    finally:
        await store.close()


async def test_api_workspace_and_explicit_office_attachment(tmp_path):
    instructions = []

    async def answer(messages, info):
        instructions.append(info.instructions)
        yield "Hello"

    store = await Store.open(tmp_path / "store")
    runtime = Runtime(store, model=FunctionModel(stream_function=answer))
    app = create_app(store, runtime, "token")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://owner",
            headers={"Authorization": "Bearer token"},
        ) as client:
            response = await client.post(
                PREFIX + "/conversations", json={"workspace": str(tmp_path)}
            )
            assert response.status_code == 201
            conversation = response.json()
            assert conversation["workspace"] == str(tmp_path.resolve())
            response = await client.post(
                PREFIX + "/turns", json=turn(conversation).model_dump()
            )
            assert response.status_code == 200
            assert '"outcome":"completed"' in response.text
            assert str(tmp_path.resolve()) in instructions[0]
            assert (
                "file paths may be absolute, relative to this workspace, or ~/path"
                in instructions[0]
            )
            document = io.BytesIO()
            with zipfile.ZipFile(document, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body><w:p><w:r><w:t>Quarterly report</w:t></w:r></w:p></w:body></w:document>",
                )
            uploaded = await client.post(
                PREFIX + f"/conversations/{conversation['id']}/attachments",
                params={"name": "report.docx"},
                content=document.getvalue(),
            )
            assert uploaded.status_code == 201, uploaded.text
            text = await store.read_attachment(
                conversation["id"], uploaded.json()["id"]
            )
            assert "Quarterly report" in text
            # Explicit upload is the authority even for a secrets-like name.
            uploaded = await client.post(
                PREFIX + f"/conversations/{conversation['id']}/attachments",
                params={"name": ".env"},
                content=b"EXPLICIT_TEST_VALUE=fixture",
            )
            assert uploaded.status_code == 201
            image = await client.post(
                PREFIX + f"/conversations/{conversation['id']}/attachments",
                params={"name": "picture.png"},
                content=b"not-an-image",
            )
            assert image.status_code == 415
            assert image.json()["detail"]["code"] == "image_unsupported"
    finally:
        await runtime.close()
        await store.close()


def test_client_captures_launch_workspace_and_uploads_office_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    client = ChatClient(SimpleNamespace(), "http://127.0.0.1:18800", "fake")
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/conversations":
            return {"id": "conversation", "branch_id": "branch"}
        return {"id": "attachment"}

    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "refresh", lambda: None)
    monkeypatch.setattr(client, "show_chat_status", lambda: None)
    try:
        client.new()
        assert calls[0][2]["json"]["workspace"] == str(tmp_path.resolve())
        document = tmp_path / "office.docx"
        document.write_bytes(b"PK\x03\x04\xff")
        client.command(f"/attach {document}")
        assert calls[-1][2]["content"] == document.read_bytes()
        assert client.attachments == ["attachment"]
    finally:
        client.client.close()


def test_tool_progress_uses_arguments_and_sanitizes_terminal_controls():
    assert (
        tool_progress({"name": "read_file", "args": {"path": "report.pdf"}})
        == "Reading report.pdf…"
    )
    assert tool_progress(
        {"name": "grep", "args": {"pattern": "foo", "path": "./src"}}
    ) == ("Searching for 'foo' in ./src…")
    state = TurnState("turn")
    stream = io.StringIO()
    renderer = HarnessRenderer(state, stream=stream)
    event = {
        "wire_version": 1,
        "turn_id": "turn",
        "seq": 1,
        "event": {
            "type": "tool.started",
            "call_id": "call",
            "name": "glob",
            "args": {"pattern": "*.md\x1b[31m", "path": "."},
        },
    }
    state.apply(event)
    renderer.update(event)
    assert "Finding '*.md" in stream.getvalue()
    assert "' in .…" in stream.getvalue()
    assert "\x1b" not in stream.getvalue()
