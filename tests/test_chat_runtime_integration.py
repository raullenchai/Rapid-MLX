# SPDX-License-Identifier: Apache-2.0
"""Contracts for the combined file/action owner, without a CLI client."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from rapid_mlx.chat_runtime import file_operations
from rapid_mlx.chat_runtime.api import PREFIX, create_app
from rapid_mlx.chat_runtime.path_policy import FileError
from rapid_mlx.chat_runtime.runtime import Runtime, chat_instructions
from rapid_mlx.chat_runtime.store import Store
from rapid_mlx.chat_runtime.tools import REGISTRY, TurnDeps
from rapid_mlx.chat_runtime.tools.action_policy import execution_policy


async def test_current_store_reopens_with_history_and_ledger(tmp_path):
    store = await Store.open(tmp_path)
    conversation = await store.create_conversation(workspace=str(tmp_path))
    identity, revision = store.store_uuid, await store.catalog_revision()
    await store.close()
    store = await Store.open(tmp_path)
    try:
        assert store.store_uuid == identity
        assert await store.get_conversation(conversation["id"]) == {
            **conversation,
            "uncertain_operations": [],
        }
        assert await store.catalog_revision() == revision
        with sqlite3.connect(store.path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 4
            assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        await store.close()


@pytest.mark.parametrize(
    "path",
    [
        ".docker/notes.txt",
        "browser-profile/data",
        "cookies.sqlite",
        ".config/gh/hosts.yml",
        "huggingface/token",
        "x.oauth.json",
        "service-account-prod.json",
        "passwords.txt",
        "private.kdbx",
        "session-secret.txt",
        "hosts.yml",
        "id_ecdsa",
        "shadow",
        "passwd",
        "gcloud/config",
        "keychain/item",
        "session.token",
    ],
)
async def test_both_file_adapters_deny_union_before_reading(
    tmp_path, monkeypatch, path
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("PRIVATE_PREIMAGE")
    store = await Store.open(tmp_path / "profile")
    runtime = Runtime(store)
    deps = TurnDeps(store, "conversation", frozenset(), None, workspace=str(workspace))
    try:
        result = file_operations.dispatch(
            {
                "operation": "read_file",
                "workspace": str(workspace),
                "args": {"path": path},
            }
        )
        assert result["error"]["code"] == "sensitive_path"
        assert "PRIVATE_PREIMAGE" not in json.dumps(result)

        def forbidden_read(*args):
            pytest.fail("denied path was inspected before policy enforcement")

        monkeypatch.setattr(
            "rapid_mlx.chat_runtime.tools.action_tools.read_image", forbidden_read
        )
        with pytest.raises(FileError, match="Secrets-like"):
            runtime.actions.prepare(
                "write_file", {"path": path, "content": "new"}, deps
            )
        assert execution_policy(f"cat '{path}'", str(workspace)).prompt
        assert execution_policy(
            f"print(open({path!r}).read())", str(workspace), python=True
        ).prompt
        assert target.read_text() == "PRIVATE_PREIMAGE"
    finally:
        await runtime.close()
        await store.close()


def test_single_inventory_and_deduplicated_instructions():
    assert len(REGISTRY) == len({spec.name for spec in REGISTRY}) == 12
    deps = TurnDeps(
        None, "c", frozenset({"a"}), None, workspace="/workspace", web=object()
    )
    instructions = chat_instructions(SimpleNamespace(deps=deps))
    assert instructions.count("Workspace:") == 1
    assert instructions.count("untrusted reference material") == 1
    assert instructions.count("visible answer") == 1
    assert instructions.count("use `patch`, `write_file` to make the") == 1


async def test_owner_api_combined_journey_approval_undo_and_replay(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "facts.txt").write_text("The project code is bluebird.")
    (workspace / "build").mkdir()
    link = tmp_path / "launch-link"
    link.symlink_to(workspace, target_is_directory=True)
    attachment_id = ""
    seen_tools = []

    async def model(messages, info):
        seen_tools.append({tool.name for tool in info.function_tools})
        calls = [
            ("read_attachment", {"attachment_id": attachment_id}),
            ("read_file", {"path": "facts.txt"}),
            ("write_file", {"path": "answer.py", "content": "print(5)\n"}),
            ("execute_code", {"language": "python", "code": "print(2 + 3)"}),
            ("terminal", {"command": "rm -rf build"}),
        ]
        count = sum(p.part_kind == "tool-return" for m in messages for p in m.parts)
        if count < len(calls):
            name, args = calls[count]
            yield {
                0: DeltaToolCall(
                    name=name, json_args=json.dumps(args), tool_call_id=f"call-{count}"
                )
            }
        else:
            yield "Read both sources, saved answer.py and calculated 5. Deletion was denied."

    store = await Store.open(tmp_path / "profile")
    runtime = Runtime(store, model=FunctionModel(stream_function=model))
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(create_app(store, runtime, "token")),
        base_url="http://owner",
        headers={"Authorization": "Bearer token"},
    )
    task = None
    try:
        conversation = (
            await client.post(PREFIX + "/conversations", json={"workspace": str(link)})
        ).json()
        assert conversation["workspace"] == str(workspace)
        cid = conversation["id"]
        uploaded = await client.post(
            PREFIX + f"/conversations/{cid}/attachments",
            params={"name": "reference.txt"},
            content=b"An explicit snapshot.",
        )
        attachment_id = uploaded.json()["id"]
        body = {
            "turn_id": "combined",
            "conversation_id": cid,
            "branch_id": conversation["branch_id"],
            "expected_revision": 0,
            "text": "Read, write, run and delete",
            "attachment_ids": [attachment_id],
            "model": "fake",
            "inference_url": "http://127.0.0.1:18800",
            "context_tokens": 32768,
        }
        task = asyncio.create_task(client.post(PREFIX + "/turns", json=body))
        async with asyncio.timeout(20):
            while True:
                operations = (
                    await client.get(PREFIX + f"/conversations/{cid}/operations")
                ).json()
                pending = next(
                    (op for op in operations if op["state"] == "intent"), None
                )
                if pending:
                    break
                if task.done():
                    pytest.fail(
                        f"Turn ended without expected permission: {(await task).text}"
                    )
                await asyncio.sleep(0.01)
            assert pending["tool"] == "terminal"
            assert (workspace / "build").is_dir()
            denied = await client.post(
                PREFIX + "/permission.response",
                json={
                    "op_id": pending["id"],
                    "revision": pending["revision"],
                    "decision": "deny",
                    "reason": "Keep the build directory",
                },
            )
            assert denied.status_code == 200
            response = await task
        assert response.status_code == 200
        trace = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert trace[-1]["event"]["outcome"] == "completed"
        assert all(tools == {spec.name for spec in REGISTRY} for tools in seen_tools)
        assert any(e["event"].get("summary") == "Reading facts.txt…" for e in trace)
        assert any(
            e["event"]["type"] == "tool.output" and "5" in e["event"]["text"]
            for e in trace
        )
        assert (workspace / "answer.py").read_text() == "print(5)\n"
        assert (workspace / "build").is_dir()
        replay = await client.get(PREFIX + "/turns/combined/events")
        assert replay.text == response.text
        undo = await client.post(
            PREFIX + f"/conversations/{cid}/undo", json={"operation_id": "restore"}
        )
        assert undo.json()["state"] == "succeeded"
        assert not (workspace / "answer.py").exists()
        repeated = await client.post(
            PREFIX + f"/conversations/{cid}/undo", json={"operation_id": "restore"}
        )
        assert repeated.json() == undo.json()
        snapshot = (await client.get(PREFIX + f"/conversations/{cid}")).json()
        assert snapshot["workspace"] == str(workspace)
        body.update(
            turn_id="different-workspace",
            workspace=str(tmp_path),
            expected_revision=snapshot["branches"][0]["revision"],
        )
        assert (await client.post(PREFIX + "/turns", json=body)).status_code == 409
    finally:
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await client.aclose()
        await runtime.close()
        await store.close()


def test_execution_inspection_uses_shared_paths_after_shell_tokenization(tmp_path):
    (tmp_path / ".env").write_text("PRIVATE_VALUE")
    (tmp_path / "alias").symlink_to(tmp_path / ".env")
    for command in (
        "cat auth.json; true",
        "cat .docker/config.json|wc -c",
        "cat alias",
    ):
        policy = execution_policy(command, str(tmp_path))
        assert policy.prompt and not policy.allow_conversation
    assert execution_policy(
        "print(open('alias').read())", str(tmp_path), python=True
    ).prompt
