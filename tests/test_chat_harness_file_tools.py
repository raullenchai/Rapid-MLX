# SPDX-License-Identifier: Apache-2.0
"""Workspace authority, bounded discovery, and durable read-only tool contracts."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from rapid_mlx.chat_runtime import file_operations as files
from rapid_mlx.chat_runtime import file_worker
from rapid_mlx.chat_runtime.runtime import Runtime, TurnRequest
from rapid_mlx.chat_runtime.store import Store
from rapid_mlx.chat_runtime.tools import TurnDeps
from rapid_mlx.chat_runtime.tools import files as adapters


def request(workspace, operation="read_file", **args):
    return files.dispatch(
        {"operation": operation, "workspace": str(workspace), "args": args}
    )


def write(root, path, content="needle\n"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def test_workspace_authority_and_symlink_escape(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write(workspace, "fact.txt", "allowed")
    outside = write(tmp_path, "outside.txt", "must not leak")
    (workspace / "escape.txt").symlink_to(outside)
    (workspace / "inside.txt").symlink_to(workspace / "fact.txt")
    monkeypatch.chdir(tmp_path)
    assert "allowed" in request(workspace, path="fact.txt")["content"]
    assert "allowed" in request(workspace, path="inside.txt")["content"]
    for path in (str(outside), "../outside.txt", "escape.txt"):
        result = request(workspace, path=path)
        assert result["error"]["code"] == "outside_workspace"
        assert "/attach" in result["error"]["message"]
        assert "must not leak" not in json.dumps(result)
    assert (
        files.dispatch({"operation": "read_file", "args": {"path": "outside.txt"}})[
            "error"
        ]["code"]
        == "workspace_unavailable"
    )


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "sub/.env.example",
        ".ssh/config",
        ".aws/credentials",
        "secrets.json",
        "credentials.yaml",
        "private.pem",
        "auth.json",
        "Library/Keychains/login.keychain-db",
        "Library/Browser/Cookies.sqlite",
        ".config/gcloud/application_default_credentials.json",
        ".git-credentials",
        ".cache/huggingface/token",
        "cache/bws_cache.json",
        ".machinecontrol.token",
    ],
)
def test_private_paths_are_denied_inside_workspace(tmp_path, path):
    write(tmp_path, path, "confidential")
    assert request(tmp_path, path=path)["error"]["code"] == "sensitive_path"
    for operation, pattern in (("grep", "confidential"), ("glob", "**/*")):
        result = request(tmp_path, operation, pattern=pattern)
        assert not result["matches"]


def test_alias_to_secret_and_secret_alias_to_public_are_denied(tmp_path):
    secret = write(tmp_path, ".env", "confidential")
    (tmp_path / "public.txt").symlink_to(secret)
    public = write(tmp_path, "allowed.txt", "public")
    (tmp_path / "credentials.json").symlink_to(public)
    for path in ("public.txt", "credentials.json"):
        assert request(tmp_path, path=path)["error"]["code"] == "sensitive_path"


def test_numbered_paging_and_character_cap_preserve_unicode(tmp_path):
    original = "🌍" * 9000 + "END\n" + "\n".join(f"row {i}" for i in range(500))
    write(tmp_path, "large.txt", original)
    offset, recovered, pages = 1, [], 0
    while True:
        page = request(tmp_path, path="large.txt", offset=offset, limit=2000)
        assert len(json.dumps(page, ensure_ascii=False).encode()) <= files.MAX_OUTPUT
        for line in page["content"].splitlines():
            if "|" in line and line.split("|", 1)[0].isdigit():
                recovered.append(line.split("|", 1)[1])
        pages += 1
        if not page["truncated"]:
            break
        assert page["next_offset"] > offset
        assert "use read_file" in page["content"]
        offset = page["next_offset"]
        assert pages < 30
    assert "".join(recovered) == "".join(original.splitlines())
    assert pages > 1
    page = request(tmp_path, path="large.txt", offset=6, limit=2)
    assert page["content"].startswith("6|row 0\n7|row 1")
    assert (
        request(tmp_path, path="large.txt", offset=0)["error"]["code"]
        == "invalid_arguments"
    )


def test_low_context_cap_must_make_progress_or_return_typed_error(tmp_path):
    write(tmp_path, "long.txt", "🌍" * 2000)
    result = files.dispatch(
        {
            "operation": "read_file",
            "workspace": str(tmp_path),
            "args": {"path": "long.txt"},
            "output_cap": 1024,
        }
    )
    assert (
        "error" in result or result["next_offset"] is None or result["next_offset"] > 1
    )


def test_search_ignore_binary_venv_and_backend(tmp_path):
    write(tmp_path, ".gitignore", "/root.txt\n*.log\n!keep.log\nignored/\n")
    write(tmp_path, "src/.gitignore", "/local.txt\n!rescue.log\n")
    for path in (
        "root.txt",
        "drop.log",
        "ignored/file.txt",
        "src/local.txt",
        "src/drop.log",
        ".git/config",
        "node_modules/a.js",
        ".venv/a.py",
        "custom_env/pyvenv.cfg",
        "custom_env/a.py",
    ):
        write(tmp_path, path)
    for path in (
        "keep.log",
        "src/root.txt",
        "src/rescue.log",
        "src/code.py",
        "src/nested/local.txt",
    ):
        write(tmp_path, path, "Needle here\n")
    (tmp_path / "binary.txt").write_bytes(b"needle\0binary")
    (tmp_path / "invalid.txt").write_bytes(b"needle\xff")
    result = request(tmp_path, "grep", pattern="needle", case=False)
    assert result["backend"] == "python"
    assert {item["path"] for item in result["matches"]} == {
        "keep.log",
        "src/root.txt",
        "src/rescue.log",
        "src/code.py",
        "src/nested/local.txt",
    }
    assert all(item["line"] == 1 for item in result["matches"])
    assert not request(tmp_path, "grep", pattern="needle", case=True)["matches"]
    filtered = request(tmp_path, "grep", pattern="Needle", path="src", glob="*.py")
    assert [item["path"] for item in filtered["matches"]] == ["src/code.py"]
    assert not request(tmp_path, "grep", pattern="needle", path="ignored")["matches"]
    assert request(tmp_path, "grep", pattern="[")["error"]["code"] == "invalid_pattern"


def test_glob_order_recursive_pattern_and_results_cap(tmp_path):
    for index, path in enumerate(("old.md", "src/middle.md", "src/nested/new.md"), 1):
        target = write(tmp_path, path)
        os.utime(target, (index, index))
    write(tmp_path, "src/code.py")
    result = request(tmp_path, "glob", pattern="**/*.md")
    assert result["matches"] == ["src/nested/new.md", "src/middle.md", "old.md"]
    limited = request(tmp_path, "glob", pattern="*.md", max_results=2)
    assert len(limited["matches"]) == 2
    assert limited["truncated"]
    assert "narrow" in limited["note"]
    assert request(tmp_path, "glob", pattern="src/*.md")["matches"] == ["src/middle.md"]


def test_grep_result_cap_and_long_lines(tmp_path):
    write(tmp_path, "matches.txt", "\n".join(["needle" + "x" * 2000] * 10))
    result = request(tmp_path, "grep", pattern="needle", max_results=3)
    assert len(result["matches"]) == 3
    assert result["truncated"]
    assert all("Line truncated" in item["text"] for item in result["matches"])


def test_grep_python_regex_semantics(tmp_path):
    write(tmp_path, "matches.txt", "foobar\nbar\nfoobar again\n")
    result = request(tmp_path, "grep", pattern=r"(?<=foo)bar")
    assert result["matches"] == [
        {"path": "matches.txt", "line": 1, "text": "foobar"},
        {"path": "matches.txt", "line": 3, "text": "foobar again"},
    ]
    assert result["backend"] == "python"
    assert not result["truncated"]
    assert request(tmp_path, "grep", pattern=r"\w+", case=True)["backend"] == "python"
    write(tmp_path, "cr.txt", "needle\rrest\n")
    assert (
        request(tmp_path, "grep", pattern="rest", path="cr.txt")["matches"][0]["line"]
        == 2
    )
    lookahead = request(tmp_path, "grep", pattern=r"foo(?=bar)")
    assert lookahead["matches"] == result["matches"]
    assert lookahead["backend"] == "python"


@pytest.mark.parametrize("cancel", [False, True])
async def test_worker_timeout_and_cancel_reap_owned_process(
    tmp_path, monkeypatch, cancel
):
    real_spawn = asyncio.create_subprocess_exec
    started = asyncio.Event()
    children = []
    events = []

    async def spawn(*args, **kwargs):
        process = await real_spawn(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        children.append(process)
        started.set()
        return process

    async def emit(event):
        events.append(event)
        return event

    monkeypatch.setattr(file_worker.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(adapters, "TOOL_TIMEOUT", 0.1 if not cancel else 60)
    deps = TurnDeps(None, "conversation", frozenset(), emit, workspace=str(tmp_path))
    task = asyncio.create_task(
        adapters.read_file(SimpleNamespace(deps=deps, tool_call_id="read"), "file.txt")
    )
    await started.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = json.loads(await task)
        assert result["error"]["code"] == "timeout"
        assert events[-1]["outcome"] == "error"
    assert children[0].returncode is not None
    assert events[0]["type"] == "tool.started"


async def test_runtime_file_events_persistence_and_regenerate_rereads(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = write(workspace, "fact.txt", "original fact")
    calls = 0
    seen_results = []

    async def stream(messages, info):
        nonlocal calls
        calls += 1
        assert {"read_file", "grep", "glob"} <= {
            tool.name for tool in info.function_tools
        }
        if calls % 2:
            yield {
                0: DeltaToolCall(
                    name="read_file",
                    json_args='{"path":"fact.txt"}',
                    tool_call_id=f"read-{calls}",
                )
            }
        else:
            returned = [
                part.content
                for message in messages
                for part in message.parts
                if part.part_kind == "tool-return"
            ]
            seen_results.append(json.loads(returned[-1]))
            yield "The file has been read."

    store = await Store.open(tmp_path / "store")
    conversation = await store.create_conversation(workspace=str(workspace))
    runtime = Runtime(store, model=FunctionModel(stream_function=stream))
    first = TurnRequest(
        turn_id="first",
        conversation_id=conversation["id"],
        branch_id=conversation["branch_id"],
        expected_revision=0,
        text="Read fact.txt",
        inference_url="http://localhost:18800",
        model="fake",
        context_tokens=16384,
    )
    try:
        await runtime.submit(first)
        events = [event async for event in runtime.events(first.turn_id) if event]
        assert events[-1]["event"]["outcome"] == "completed"
        kinds = [event["event"]["type"] for event in events]
        assert (
            kinds.index("tool.started")
            < kinds.index("tool.result")
            < kinds.index("turn.terminal")
        )
        rows = await store.branch_messages(first.branch_id)
        assert [row["role"] for row in rows] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert "original fact" in json.dumps(rows)
        target.write_text("changed fact")
        fork = await store.regenerate(first.branch_id, "regenerate")
        second = first.model_copy(
            update={
                "turn_id": "second",
                "branch_id": fork["id"],
                "expected_revision": fork["revision"],
                "mode": "regenerate",
                "text": "",
            }
        )
        await runtime.submit(second)
        second_events = [
            event async for event in runtime.events(second.turn_id) if event
        ]
        assert second_events[-1]["event"]["outcome"] == "completed"
        assert "original fact" in seen_results[0]["content"]
        assert "changed fact" in seen_results[1]["content"]
        assert "changed fact" in json.dumps(
            await store.branch_messages(second.branch_id)
        )
        assert "original fact" in json.dumps(
            await store.branch_messages(first.branch_id)
        )
        assert (
            len(
                [
                    event
                    for event in second_events
                    if event["event"]["type"] == "tool.started"
                ]
            )
            == 1
        )
    finally:
        await runtime.close()
        await store.close()
    reopened = await Store.open(tmp_path / "store")
    try:
        assert (await reopened.get_conversation(conversation["id"]))[
            "workspace"
        ] == str(workspace)
        assert "changed fact" in json.dumps(await reopened.replay_events("second"))
    finally:
        await reopened.close()


def test_output_cap_accounts_for_json_escaping(tmp_path):
    write(tmp_path, "escaped.txt", ('"\\' * 1000 + "\n") * 20)
    result = request(tmp_path, path="escaped.txt", limit=2000)
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= files.MAX_OUTPUT


def test_search_cannot_enter_venv_via_explicit_descendant(tmp_path):
    write(tmp_path, "custom/pyvenv.cfg", "home = python")
    write(tmp_path, "custom/deep/module.py", "needle")
    for path in ("custom/deep", "custom/deep/module.py"):
        assert not request(tmp_path, "grep", pattern="needle", path=path)["matches"]
        assert not request(tmp_path, "glob", pattern="*.py", path=path)["matches"]


def test_replaced_workspace_and_lexical_secret_alias(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside, "fact.txt", "outside fact")
    workspace.rmdir()
    workspace.symlink_to(outside, target_is_directory=True)
    assert request(workspace, path="fact.txt")["error"]["code"] == "workspace_changed"
    (outside / ".env.alias").symlink_to(outside / "fact.txt")
    assert request(outside, path=".env.alias")["error"]["code"] == "sensitive_path"


def test_ignore_contents_negation_and_large_skips(tmp_path):
    write(tmp_path, ".gitignore", "foo/**\n!foo/keep.txt\n")
    write(tmp_path, "foo/keep.txt")
    write(tmp_path, "foo/drop.txt")
    result = request(tmp_path, "grep", pattern="needle")
    assert [hit["path"] for hit in result["matches"]] == ["foo/keep.txt"]
    write(tmp_path, "large.py", "needle\n" + "a" * (1024 * 1024))
    result = request(tmp_path, "grep", pattern="needle", path="large.py")
    assert result["truncated"] and result["skipped_files"] == 1
    assert "read_file" in result["note"]


async def test_workers_cannot_import_from_untrusted_cwd(tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    write(
        tmp_path,
        "rapid_mlx/__init__.py",
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    write(tmp_path, "fact.txt", "trusted read")
    result = await file_worker.run_worker(
        "rapid_mlx.chat_runtime.file_operations",
        json.dumps(
            {
                "operation": "read_file",
                "workspace": str(tmp_path),
                "args": {"path": "fact.txt"},
            }
        ).encode(),
    )
    assert "trusted read" in result["content"]
    result = await file_worker.run_worker(
        "rapid_mlx.chat_runtime.file_extraction",
        b"plain attachment",
        arguments=("note.txt",),
    )
    assert result["text"] == "plain attachment"
    assert not marker.exists()


async def test_parallel_reads_share_context_budget(tmp_path):
    write(tmp_path, "note.txt", "x" * 1800)
    events = []

    async def emit(event):
        events.append(event)
        return event

    deps = TurnDeps(
        None,
        "conversation",
        frozenset(),
        emit,
        read_budget_bytes=5000,
        workspace=str(tmp_path),
    )
    results = await asyncio.gather(
        *[
            adapters.read_file(
                SimpleNamespace(deps=deps, tool_call_id=str(i)), "note.txt"
            )
            for i in range(3)
        ]
    )
    assert 1 <= sum("error" not in json.loads(result) for result in results) < 3
    assert json.loads(results[-1])["error"]["code"] == "context_limit"
    assert sum(len(result.encode()) for result in results) <= 5000


def test_grep_uses_python_even_when_rg_is_available(tmp_path, monkeypatch):
    import shutil
    import subprocess

    write(tmp_path, "cr.txt", "needle\rrest\n")
    monkeypatch.setattr(shutil, "which", lambda _: "rg")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=b"1:needle\rrest\n", stderr=b""
        ),
    )
    result = request(tmp_path, "grep", pattern="rest")
    assert result["backend"] == "python"
    assert result["matches"] == [{"path": "cr.txt", "line": 2, "text": "rest"}]


@pytest.mark.parametrize("code", ["search_limit", "workspace_changed"])
def test_search_preserves_scan_limit_and_other_errors(tmp_path, monkeypatch, code):
    def interrupted(*args):
        raise files.FileError(code, "scan stopped")
        yield

    monkeypatch.setattr(files, "walk_files", interrupted)
    result = request(tmp_path, "grep", pattern="needle")
    if code == "search_limit":
        assert result["truncated"]
    else:
        assert result["error"]["code"] == code
