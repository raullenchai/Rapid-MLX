# SPDX-License-Identifier: Apache-2.0
"""CPU-only local-search pilot contracts; no MLX/model/network required."""

import asyncio
import errno
import io
import os
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from rapid_mlx.local_search.chunking import CHUNKING_VERSION, Chunk, chunk_units
from rapid_mlx.local_search.extraction import (
    MAX_EMBEDDED_CHUNKS_PER_FILE,
    SkippedError,
    Unit,
    extract,
    read_safe,
    scan,
    validate_folder,
)
from rapid_mlx.local_search.search import fuse, snippet
from rapid_mlx.local_search.service import LocalSearch, ModelLoadError, local_model_path
from rapid_mlx.local_search.store import Store
from rapid_mlx.local_search_runtime import validate_exposure
from rapid_mlx.middleware.exception_handlers import install_exception_handlers

FIRST_ROUTES = [
    ("GET", "/status", None),
    ("POST", "/folders", "folder"),
    ("DELETE", "/folders/unknown", None),
    ("POST", "/reindex", {}),
    ("POST", "/reindex", {"folder_id": "unknown"}),
    ("POST", "/search", {"query": "apple"}),
    ("POST", "/search", {"query": "apple", "folder_ids": ["unknown"]}),
]


@pytest.mark.parametrize("host", ["127.0.0.1", "127.23.4.5", "::1", " 127.0.0.1 "])
def test_local_search_allows_unauthenticated_loopback(host):
    validate_exposure(enabled=True, host=host, api_key=None)


@pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "", "   ", "ip6-localhost"])
def test_local_search_requires_authentication_for_host_names(host, monkeypatch):
    # Names are re-resolved at bind time, so they are never trusted as
    # loopback, even when they currently resolve to it.
    def fail(*args, **kwargs):
        raise AssertionError("validation must not resolve host names")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(ValueError, match="requires API-key authentication"):
        validate_exposure(enabled=True, host=host, api_key=None)
    validate_exposure(enabled=True, host=host, api_key="secret")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "server.local"])
def test_local_search_requires_authentication_off_loopback(host):
    with pytest.raises(ValueError, match="requires API-key authentication"):
        validate_exposure(enabled=True, host=host, api_key=None)
    validate_exposure(enabled=True, host=host, api_key="secret")
    validate_exposure(enabled=False, host=host, api_key=None)


def test_local_search_requires_authentication_for_inherited_listener():
    with pytest.raises(ValueError, match="--listen-fd requires API-key"):
        validate_exposure(
            enabled=True,
            host="127.0.0.1",
            api_key=None,
            inherited_listener=True,
        )
    validate_exposure(
        enabled=True,
        host="127.0.0.1",
        api_key="secret",
        inherited_listener=True,
    )


def test_unified_cli_rejects_remote_local_search_before_model_resolution():
    env = os.environ.copy()
    env.pop("RAPID_MLX_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.cli",
            "serve",
            "qwen3.5-4b-4bit",
            "--host",
            "0.0.0.0",
            "--enable-local-search",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "non-loopback --host requires API-key authentication" in result.stderr
    assert "Alias:" not in result.stdout
    assert "Resolving" not in result.stderr


def test_unified_cli_exposure_helper_reports_actionable_error(monkeypatch, capsys):
    from rapid_mlx import cli

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    args = SimpleNamespace(
        enable_local_search=True,
        host="0.0.0.0",
        api_key=None,
        listen_fd=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._validate_local_search_exposure_or_exit(args)

    assert exc_info.value.code == 2
    assert "requires API-key authentication" in capsys.readouterr().err


def test_unified_cli_runs_exposure_gate_immediately_after_parse(monkeypatch):
    from rapid_mlx import cli

    class ExposureGateReachedError(Exception):
        pass

    observed = []

    def stop_after_gate(args):
        observed.append((args.command, args.enable_local_search, args.host))
        raise ExposureGateReachedError

    monkeypatch.setattr(cli, "_validate_local_search_exposure_or_exit", stop_after_gate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "qwen3.5-4b-4bit",
            "--host",
            "0.0.0.0",
            "--enable-local-search",
        ],
    )

    with pytest.raises(ExposureGateReachedError):
        cli.main.__wrapped__()

    assert observed == [("serve", True, "0.0.0.0")]


def test_standalone_server_parser_rejects_inherited_listener(capsys):
    from rapid_mlx import server

    with pytest.raises(SystemExit) as exc_info:
        server._build_parser().parse_args(["--model", "test", "--listen-fd", "3"])

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --listen-fd 3" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [[], ["--allow-unauthenticated-public"]])
def test_standalone_server_rejects_remote_local_search_after_parse(
    monkeypatch, capsys, extra
):
    from rapid_mlx import server

    monkeypatch.delenv("RAPID_MLX_API_KEY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid_mlx.server",
            "--model",
            "qwen3.5-4b-4bit",
            "--host",
            "0.0.0.0",
            "--enable-local-search",
            *extra,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        server.main.__wrapped__()

    assert exc_info.value.code == 2
    assert "requires API-key authentication" in capsys.readouterr().err


@pytest.mark.parametrize("existing", [False, True])
def test_index_owner_only_under_permissive_umask(tmp_path, monkeypatch, existing):
    path = tmp_path / "public-parent" / "index" / "index.sqlite3"
    path.parent.parent.mkdir(mode=0o1777)
    path.parent.parent.chmod(0o1777)
    previous_umask = os.umask(0)
    first = second = None
    try:
        if existing:
            first = Store(path, "test")
            path.parent.chmod(0o755)
            for suffix in ("", "-wal", "-shm"):
                Path(str(path) + suffix).chmod(0o644)
            # A cold, non-hot rollback journal also needs repair before recovery.
            journal = Path(str(path) + "-journal")
            journal.write_bytes(b"\0" * 512)
            journal.chmod(0o644)
        connect = sqlite3.connect

        def protected_connect(database, *args, **kwargs):
            # Assert protection before SQLite's first opportunity to write.
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
            if existing:
                assert stat.S_IMODE(journal.stat().st_mode) == 0o600
            return connect(database, *args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", protected_connect)
        second = Store(path, "test")
        assert second.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        with second.db:
            second.db.execute("INSERT INTO meta VALUES ('private_text', 'secret')")
        for suffix in ("", "-wal", "-shm"):
            assert stat.S_IMODE(Path(str(path) + suffix).stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o1777
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()
        os.umask(previous_umask)


def test_index_rollback_journal_inherits_private_mode(tmp_path):
    previous_umask = os.umask(0)
    store = None
    try:
        path = tmp_path / "index" / "index.sqlite3"
        store = Store(path, "test")
        store.db.execute("PRAGMA journal_mode=DELETE")
        store.db.execute("INSERT INTO meta VALUES ('private_text', 'secret')")
        assert stat.S_IMODE(Path(str(path) + "-journal").stat().st_mode) == 0o600
        store.db.rollback()
    finally:
        if store is not None:
            store.close()
        os.umask(previous_umask)


def test_private_index_directory_with_unrelated_files_needs_no_chmod(
    tmp_path, monkeypatch
):
    tmp_path.chmod(0o700)
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    unrelated.chmod(0o644)
    monkeypatch.setattr(os, "fchmod", lambda *args: pytest.fail("unexpected chmod"))
    previous_umask = os.umask(0)
    try:
        store = Store(tmp_path / "index.sqlite3", "test")
        store.close()
        assert stat.S_IMODE(unrelated.stat().st_mode) == 0o644
    finally:
        os.umask(previous_umask)


@pytest.mark.parametrize(
    "unsafe, message",
    [
        ("mixed", "dedicated index directory"),
        ("directory_entry", "dedicated index directory"),
        ("symlink_entry", "dedicated index directory"),
        ("foreign_directory", "owned by the current user"),
        ("foreign_file", "owned by the current user"),
        ("chmod_denied", "permission repair denied"),
        ("chmod_ignored", "owner-only"),
        ("symlink_directory", "index"),
        ("symlink_database", "index.sqlite3"),
        ("symlink_wal", "index.sqlite3-wal"),
        ("symlink_shm", "index.sqlite3-shm"),
        ("symlink_journal", "index.sqlite3-journal"),
        ("hardlink", "without hard links"),
        ("fifo", "regular files"),
        ("acl_denied", "ACL repair denied"),
        ("acl_ignored", "Could not remove"),
        ("acl_probe_denied", "ACL probe denied"),
    ],
)
@pytest.mark.parametrize("method,route,body", FIRST_ROUTES)
def test_unsafe_index_refused_with_readable_status(
    tmp_path, monkeypatch, unsafe, message, method, route, body
):
    from rapid_mlx.routes.local_search import router

    directory = tmp_path / "index"
    directory.mkdir(mode=0o700)
    path = directory / "index.sqlite3"
    outside = tmp_path / "outside"
    outside.write_text("unrelated content")
    outside.chmod(0o644)
    if unsafe == "mixed":
        (directory / "unrelated.txt").write_text("keep")
        directory.chmod(0o755)
    elif unsafe == "directory_entry":
        path.mkdir()
        directory.chmod(0o755)
    elif unsafe == "symlink_entry":
        path.symlink_to(outside)
        directory.chmod(0o755)
    elif unsafe.startswith("foreign_"):
        if unsafe == "foreign_file":
            path.write_bytes(b"")
            path.chmod(0o644)
        target = (directory if unsafe == "foreign_directory" else path).stat().st_ino
        fstat = os.fstat

        def foreign_owner(fd):
            info = fstat(fd)
            if info.st_ino == target:
                values = list(info)
                values[4] = os.geteuid() + 1
                return os.stat_result(values)
            return info

        monkeypatch.setattr(os, "fstat", foreign_owner)
    elif unsafe.startswith("chmod_"):
        directory.chmod(0o755)

        def cannot_repair(fd, mode):
            if unsafe == "chmod_denied":
                raise PermissionError("permission repair denied")

        monkeypatch.setattr(os, "fchmod", cannot_repair)
    elif unsafe == "symlink_directory":
        from rapid_mlx.local_search import store as module

        prepare = module._prepare_index

        def swap_after_resolution(resolved):
            directory.rmdir()
            directory.symlink_to(tmp_path, target_is_directory=True)
            prepare(resolved)

        monkeypatch.setattr(module, "_prepare_index", swap_after_resolution)
    elif unsafe.startswith("symlink_"):
        suffix = {"database": "", "wal": "-wal", "shm": "-shm", "journal": "-journal"}[
            unsafe.removeprefix("symlink_")
        ]
        Path(str(path) + suffix).symlink_to(outside)
    elif unsafe.startswith("acl_"):
        from rapid_mlx.local_search import store as module

        monkeypatch.setattr(module.sys, "platform", "darwin")
        monkeypatch.setattr(module, "_ancestor_acl_dangerous", lambda _: False)
        target = directory.stat().st_ino

        def probe(fd):
            if os.fstat(fd).st_ino != target:
                return False
            if unsafe == "acl_probe_denied":
                raise PermissionError("ACL probe denied")
            return True

        def strip(fd):
            if os.fstat(fd).st_ino == target and unsafe == "acl_denied":
                raise PermissionError("ACL repair denied")

        monkeypatch.setattr(module, "_acl_present", probe)
        monkeypatch.setattr(module, "_remove_acl", strip)
    elif unsafe == "hardlink":
        os.link(outside, path)
    else:
        os.mkfifo(path)
    monkeypatch.setattr(sqlite3, "connect", lambda *args: pytest.fail("SQLite opened"))

    async def scenario():
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(router)
        service = LocalSearch(directory, resolve_model=lambda _: None)
        app.state.local_search = service
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://localhost"
            ) as client:
                payload = {"path": str(tmp_path)} if body == "folder" else body
                response = await asyncio.wait_for(
                    client.request(method, "/v1/local-search" + route, json=payload),
                    timeout=5,
                )
            assert response.status_code == 503
            error = response.json()["error"]
            assert error["code"] == "store_open_failed"
            assert error["param"] == "local_search_dir"
            assert message in error["message"]
            assert stat.S_IMODE(outside.stat().st_mode) == 0o644
            assert outside.read_text() == "unrelated content"
            if unsafe in {"mixed", "directory_entry", "symlink_entry"}:
                assert stat.S_IMODE(directory.stat().st_mode) == 0o755
        finally:
            await service.close()

    previous_umask = os.umask(0)
    try:
        asyncio.run(scenario())
    finally:
        os.umask(previous_umask)


@pytest.fixture(autouse=True)
def allow_test_scratch(monkeypatch):
    # macOS temp trees resolve under /private; never traverse the actual root.
    from rapid_mlx.local_search import extraction

    monkeypatch.setattr(
        extraction,
        "SKIP_PREFIXES",
        tuple(p for p in extraction.SKIP_PREFIXES if p != Path("/private")),
    )
    from concurrent.futures import ThreadPoolExecutor

    from rapid_mlx.config import get_config

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-model-worker")

    async def model_worker(func, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: func(*args, **kwargs)
        )

    from types import SimpleNamespace

    monkeypatch.setattr(
        get_config(), "engine", SimpleNamespace(execute_on_model_worker=model_worker)
    )
    walk = os.walk

    def temp_tree_only(root, *args, **kwargs):
        assert Path(root).resolve() != Path("/"), "Tests must never walk the real root"
        yield from walk(root, *args, **kwargs)

    monkeypatch.setattr(os, "walk", temp_tree_only)
    yield
    executor.shutdown()


class FakeEmbedder:
    def __init__(self):
        self.vision = True
        self.image_calls = []
        self.calls = []
        self.threads = []
        self.is_loaded = False

    def embed(self, texts):
        self.calls.append(texts)
        self.threads.append(threading.get_ident())
        self.is_loaded = True
        return [
            [1.0, 1.0 if "apple" in t else 0.0, 1.0 if "banana" in t else 0.0]
            for t in texts
        ]

    def embed_images(self, images):
        self.image_calls.append(len(images))
        self.threads.append(threading.get_ident())
        self.is_loaded = True
        return [[0.2, 1.0, 0.0] for _ in images]


# Deadlock guards only: correctness is established by events, never elapsed time.
SYNC_TIMEOUT = 30


class BlockingCall:
    """A worker gate that fails immediately if invoked on the event loop."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.loop_thread = threading.get_ident()
        self.entered = asyncio.Event()
        self.release = threading.Event()

    def __call__(self):
        assert threading.get_ident() != self.loop_thread, (
            "blocking call ran on event loop"
        )
        self.loop.call_soon_threadsafe(self.entered.set)
        assert self.release.wait(SYNC_TIMEOUT), "worker gate was never released"


class ObservedLock(asyncio.Lock):
    """Signal when a competing coroutine actually reaches a held lock."""

    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.waiting.set()
        return await super().acquire()


async def loop_checkpoint():
    # A queued callback acknowledges a loop turn after task start/cancellation.
    reached = asyncio.Event()
    asyncio.get_running_loop().call_soon(reached.set)
    await reached.wait()


def observe_chat_pause(search):
    paused = asyncio.Event()
    wait = search._wait_for_generation

    class Progress(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "paused_reason" and value == "chat":
                paused.set()

    async def observed():
        search.progress = Progress(search.progress)
        await wait()

    search._wait_for_generation = observed
    search.chat_paused = paused


async def wait_chat_pause(search):
    await asyncio.wait_for(search.chat_paused.wait(), SYNC_TIMEOUT)
    assert search.progress.get("paused_reason") == "chat"


def pause_after_embedding(search, *, batch=1, images=False):
    """Hold a completed batch before its caller can publish or start another."""
    entered, release = asyncio.Event(), asyncio.Event()
    embed = search._embed_async
    calls = 0

    async def observed(inputs, **kwargs):
        nonlocal calls
        result = await embed(inputs, **kwargs)
        if kwargs.get("images", False) == images:
            calls += 1
            if calls == batch:
                entered.set()
                await release.wait()
        return result

    search._embed_async = observed
    return entered, release


@pytest.mark.parametrize(
    "activity", ["admitted_requests", "running_requests", "queued_requests"]
)
@pytest.mark.parametrize("media", [False, True])
def test_indexing_yields_to_generation_and_search_stays_interactive(
    tmp_path, monkeypatch, activity, media
):
    from rapid_mlx.config import get_config

    signal = {"admitted_requests": 0, "running_requests": 0, "queued_requests": 0}
    signal[activity] = 1
    monkeypatch.setattr(
        get_config().engine,
        "lifecycle_status",
        lambda: {**signal, "active_requests": sum(signal.values())},
        raising=False,
    )

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        observe_chat_pause(search)
        root = folder(tmp_path)
        if media:
            from PIL import Image

            Image.new("RGB", (16, 16)).save(root / "apple.png")
        else:
            (root / "apple.txt").write_text(
                "An apple orchard produces delicious fruit."
            )
        try:
            (await search.add_folder(str(root)))
            await wait_chat_pause(search)
            assert not fake.calls and not fake.image_calls
            assert (await search.status_async())["indexing"]["paused_reason"] == "chat"
            result = await asyncio.wait_for(
                search.search("apple", 10, None), SYNC_TIMEOUT
            )
            assert result["mode"] == "hybrid"
            assert len(fake.calls) == 1 and "query: apple" in fake.calls[0][0]
            assert search.progress["paused_reason"] == "chat"
            signal[activity] = 0
            await asyncio.wait_for(search.wait_idle(), SYNC_TIMEOUT)
            if media:
                assert fake.image_calls == [1]
            else:
                assert len(fake.calls) == 2
            assert "paused_reason" not in (await search.status_async())["indexing"]
            assert search.status()["folders"][0]["state"] == "idle"
        finally:
            await search.close()

    asyncio.run(scenario())


def test_cancel_indexing_while_waiting_for_chat(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config

    monkeypatch.setattr(
        get_config().engine,
        "lifecycle_status",
        lambda: {"active_requests": 1},
        raising=False,
    )

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        observe_chat_pause(search)
        root = folder(tmp_path)
        (root / "apple.txt").write_text("An apple orchard produces delicious fruit.")
        try:
            registered, _ = await search.add_folder(str(root))
            await wait_chat_pause(search)
            await asyncio.wait_for(search.delete_folder(registered["id"]), SYNC_TIMEOUT)
            assert not fake.calls
            assert search.task is None
            assert "paused_reason" not in search.status()["indexing"]
            assert not search.status()["folders"]
        finally:
            await search.close()

    asyncio.run(scenario())


def test_chat_wait_backoff_is_bounded_and_clears_pause_on_resume(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config

    signal = {"active_requests": 1}
    monkeypatch.setattr(
        get_config().engine, "lifecycle_status", lambda: signal, raising=False
    )

    async def scenario():
        search = engine(tmp_path)
        delays = []

        async def sleep(delay):
            assert search.status()["indexing"]["paused_reason"] == "chat"
            delays.append(delay)
            if len(delays) == 6:
                signal["active_requests"] = 0

        try:
            with monkeypatch.context() as patch:
                patch.setattr(asyncio, "sleep", sleep)
                await search._wait_for_generation()
            assert delays == [0.01, 0.02, 0.04, 0.08, 0.1, 0.1]
            assert "paused_reason" not in search.status()["indexing"]
            assert search.engine is None
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("images", [False, True])
def test_chat_arriving_between_batches_defers_next_batch(tmp_path, monkeypatch, images):
    from rapid_mlx.config import get_config

    signal = {"active_requests": 0}
    monkeypatch.setattr(
        get_config().engine, "lifecycle_status", lambda: signal, raising=False
    )

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        observe_chat_pause(search)
        inputs = [object()] * 4 if images else ["apple"] * 4
        try:
            await search._embed_async(inputs, images=images, background=True)
            signal["active_requests"] = 1
            batch = asyncio.create_task(
                search._embed_async(inputs, images=images, background=True)
            )
            try:
                await wait_chat_pause(search)
                calls = fake.image_calls if images else fake.calls
                assert len(calls) == 1
                signal["active_requests"] = 0
                await asyncio.wait_for(batch, SYNC_TIMEOUT)
                assert len(calls) == 2
                assert "paused_reason" not in search.progress
            finally:
                if not batch.done():
                    batch.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await batch
        finally:
            await search.close()

    asyncio.run(scenario())


def test_background_rechecks_chat_after_query_lock(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config

    signal = {"active_requests": 0}
    checked = asyncio.Event()

    def status():
        checked.set()
        return signal

    monkeypatch.setattr(get_config().engine, "lifecycle_status", status, raising=False)

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        observe_chat_pause(search)
        try:
            async with search._model_gate:
                batch = asyncio.create_task(
                    search._embed_async(["apple"], background=True)
                )
                await asyncio.wait_for(checked.wait(), SYNC_TIMEOUT)
                signal["active_requests"] = 1
            try:
                await wait_chat_pause(search)
                assert not fake.calls
                signal["active_requests"] = 0
                await asyncio.wait_for(batch, SYNC_TIMEOUT)
                assert len(fake.calls) == 1
            finally:
                if not batch.done():
                    batch.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await batch
        finally:
            await search.close()

    asyncio.run(scenario())


def engine(tmp_path, fake=None, **kwargs):
    fake = fake or FakeEmbedder()
    search = LocalSearch(
        tmp_path / "index",
        resolve_model=lambda _: "local-checkpoint",
        embedder_factory=lambda _: fake,
        yield_seconds=0,
        **kwargs,
    )
    return search


def folder(tmp_path, name="source"):
    root = tmp_path / name
    root.mkdir()
    return root


def test_user_files_beside_the_index_are_indexed(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        try:
            await search.store.call("folders")  # the index opens lazily
            beside = search.store.path.parent / "note.txt"
            beside.write_text("harbour pilot schedule")
            fid = (await search.add_folder(str(tmp_path)))[0]["id"]
            await search.wait_idle()
            names = {Path(p).name for p in search.store.files(fid)}
            assert "note.txt" in names
            assert not {n for n in names if n.startswith(search.store.path.name)}
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("suffix", [".txt", ".md", ".py", ".json", ".yaml", ".csv"])
def test_text_extract(suffix):
    assert (
        extract(Path("file" + suffix), "中文 text\nsecond".encode())[0].text
        == "中文 text\nsecond"
    )


def test_html_docx_pdf():
    html = extract(
        Path("a.html"), b"<h1>Title</h1><p>A &amp; B</p><script>secret()</script>"
    )[0].text
    assert "A & B" in html and "secret" not in html and "<" not in html
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t>World</w:t></w:r></w:p></w:body></w:document>',
        )
    assert extract(Path("a.docx"), data.getvalue())[0].text == "Hello\nWorld"
    units = extract(Path("a.pdf"), small_pdf())
    assert [(u.text, u.page) for u in units] == [("First", 1), ("Second", 2)]
    assert [c.page for c in chunk_units(units)] == [1, 2]


def small_pdf():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for label in ("First", "Second"):
        page = writer.add_blank_page(300, 300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 30 30 Td ({label}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


def test_pdf_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(SkippedError, match="unsupported"):
        extract(Path("a.pdf"), b"pdf")


@pytest.mark.parametrize(
    "path,data", [("a.txt", b"\x00abc"), ("a.txt", b"\xff"), ("a.mp4", b"movie")]
)
def test_binary_and_unsupported(path, data):
    with pytest.raises(SkippedError):
        extract(Path(path), data)


def test_scan_and_safe_read(tmp_path):
    root = folder(tmp_path)
    (root / "good.md").write_text("hello")
    for name in (
        ".env.production",
        "token.pem",
        "private.key",
        "id_rsa_backup",
        "login.keychain-db",
        ".hidden",
    ):
        (root / name).write_text("secret")
    for name in (
        ".git",
        ".ssh",
        ".aws",
        ".gnupg",
        "node_modules",
        "build",
        "tool.app",
        "venv",
        "Dist",
        "DerivedData",
        "CACHES",
    ):
        (root / name).mkdir()
        (root / name / "bad.txt").write_text("secret")
    (root / "outside.txt").symlink_to(tmp_path / "external.txt")
    files, skipped = scan(root)
    assert files == [root / "good.md"] and skipped == 18
    with pytest.raises(SkippedError, match="size cap"):
        read_safe(root / "good.md", root, 3)
    with pytest.raises(OSError):
        read_safe(root / "outside.txt", root)
    outside = folder(tmp_path, "outside")
    (outside / "bad.txt").write_text("secret")
    (root / "swapped").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        read_safe(root / "swapped" / "bad.txt", root)


def test_chunk_boundaries_and_lines():
    text = "a" * 960 + "\n# Heading\n" + "b" * 1300 + "\n\n" + "c" * 200
    chunks = chunk_units([Unit(text)])
    assert chunks[0].text == "a" * 960
    assert all(len(c.text) <= 1200 for c in chunks)
    assert chunks[1].text.startswith("# Heading")
    assert chunks[1].line == 2
    assert chunks[0].line == 1 and chunks[-1].line >= 3
    assert chunks[-1].text.endswith("c" * 200)
    assert chunk_units([Unit(" \n\n hello\nworld")])[0].line == 3
    assert chunk_units([Unit("  ")]) == []
    with pytest.raises(ValueError):
        chunk_units([Unit(text)], 100, 60)


def test_latin_chunks_never_split_words_and_preserve_source_lines():
    words = [f"word{i:03d}" for i in range(80)]
    text = "  \n\n" + "\n".join(" ".join(words[i : i + 13]) for i in range(0, 80, 13))
    chunks = chunk_units([Unit(text)], target=80, overlap=17)
    assert len(chunks) > 5
    assert {word for chunk in chunks for word in chunk.text.split()} == set(words)
    for chunk in chunks:
        offset = text.index(chunk.text)
        assert chunk.line == text.count("\n", 0, offset) + 1
        assert len(chunk.text) <= 80
    # A separator before the preferred final quarter still beats cutting a word.
    fallback = "short " + "x" * 79 + " tail words"
    assert chunk_units([Unit(fallback)], 80, 10)[0].text == "short"
    exact = chunk_units([Unit("a" * 80 + " next")], 80, 10)
    assert [c.text for c in exact] == ["a" * 80, "next"]


@pytest.mark.parametrize("punctuation", list("。！？；，"))
def test_cjk_chunks_snap_end_and_overlap_to_punctuation(punctuation):
    sentences = [chr(0x4E00 + i) * 13 + punctuation for i in range(20)]
    chunks = chunk_units([Unit("".join(sentences))], target=50, overlap=15)
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 and c.line == 1 for c in chunks)
    for chunk in chunks:
        assert chunk.text in "".join(sentences)
        assert len(chunk.text) % 14 == 0
        assert chunk.text.endswith(punctuation)
    assert all(any(sentence in c.text for c in chunks) for sentence in sentences)


def test_long_unbroken_tokens_and_unspaced_cjk_allow_hard_cuts():
    for token in ("x" * 301, "汉" * 301):
        chunks = chunk_units([Unit(token)], target=80, overlap=15)
        assert [len(c.text) for c in chunks] == [80, 80, 80, 80, 41]
        assert all(c.line == 1 for c in chunks)


@pytest.mark.parametrize("separator", ["\n", ". ", "。", "！", "？", "；", "，"])
def test_snippets_start_at_line_or_sentence_and_prefer_highlight_window(separator):
    text = ("Earlier readable text" + separator) * 30
    text += "Target apple and banana" + separator + "Following text " * 50
    excerpt, highlights = snippet(text, ["apple", "banana"], size=100)
    offset = text.index(excerpt)
    assert offset == 0 or text[:offset].endswith(separator)
    assert len(excerpt) <= 100
    assert [excerpt[a:b] for a, b in highlights] == ["apple", "banana"]


def test_snippets_do_not_cut_into_a_long_single_sentence():
    text = "Beginning " + "readable words " * 60 + "apple"
    excerpt, _ = snippet(text, ["apple"])
    assert excerpt.startswith("Beginning ")


def test_snippets_prefer_more_highlighted_terms():
    text = "Apple alone. " * 50 + "Both apple and banana appear here."
    excerpt, highlights = snippet(text, ["apple", "banana"], size=80)
    assert {excerpt[a:b].lower() for a, b in highlights} == {"apple", "banana"}


def test_semantic_only_snippet_skips_fences_and_punctuation_lines():
    text = "```python\n~~~\n---\n... ! _ \\\n\n  Readable source text\n```"
    excerpt, highlights = snippet(text, ["absent"])
    assert excerpt == "Readable source text\n```"
    assert highlights == []
    # A partial query term in a fence language label must not win the window.
    assert snippet(text, ["python", "absent"])[0] == excerpt
    assert snippet("```\n---", [])[0] == "```\n---"


@pytest.mark.parametrize("old_version", [None, "1"])
@pytest.mark.parametrize("available", [False, True])
def test_chunking_version_invalidates_and_rebuilds_unchanged_files(
    tmp_path, old_version, available
):
    async def scenario():
        root = folder(tmp_path)
        path = root / "a.txt"
        path.write_text("Readable apple text " * 140)
        search = engine(tmp_path)
        if not available:
            search.resolve_model = lambda _: None
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        original_file = search.store.files(fid)[str(path)]
        # Stand in for chunks created by the old raw-offset chunker.
        search.store.replace(
            fid,
            path,
            path.stat(),
            original_file["content_hash"],
            [Chunk("yte count, rate,", 1, None)],
            [[1, 0, 0]] if available else None,
            "indexed" if available else "keyword",
        )
        old_ids = set(search.store.rows())
        with nullcontext():
            if old_version is None:
                search.store.sql("DELETE FROM meta WHERE key='chunking_version'")
            else:
                search.store.sql(
                    "UPDATE meta SET value=? WHERE key='chunking_version'",
                    (old_version,),
                )
        await search.close()
        resumed = engine(tmp_path)
        if not available:
            resumed.resolve_model = lambda _: None
        assert not resumed.store.vectors
        assert resumed.store.files(fid)[str(path)]["status"] == "pending"
        assert (
            resumed.store.sql("SELECT value FROM meta WHERE key='chunking_version'")[0][
                0
            ]
            == CHUNKING_VERSION
        )
        resumed.start()
        await resumed.wait_idle()
        assert old_ids.isdisjoint(resumed.store.rows())
        assert [r["text"] for r in resumed.store.rows().values()] == [
            c.text for c in chunk_units([Unit(path.read_text())])
        ]
        rebuilt = resumed.store.files(fid)[str(path)]
        assert rebuilt["content_hash"] == original_file["content_hash"]
        assert rebuilt["status"] == ("indexed" if available else "keyword")
        await resumed.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "value",
    ["relative", "/tmp/../tmp", "/tmp/.ssh", "/does-not-exist"],
)
def test_invalid_folder(value):
    with pytest.raises((ValueError, OSError)):
        validate_folder(value)


@pytest.mark.parametrize(
    "name", [".hidden", ".ssh", ".aws", ".gnupg", "id_rsa", "keys.pem"]
)
@pytest.mark.parametrize("nested", [False, True])
def test_sensitive_folder_roots_rejected_at_add_time(tmp_path, name, nested):
    sensitive = folder(tmp_path, name)
    root = folder(sensitive, "documents") if nested else sensitive
    search = engine(tmp_path)
    try:
        with pytest.raises(ValueError, match="hidden, sensitive, or an excluded"):
            asyncio.run(search.add_folder(str(root)))
        assert search.store.folders() == []
        assert scan(root) == ([], 1)
    finally:
        asyncio.run(search.close())


def test_resolved_folder(tmp_path):
    root = folder(tmp_path)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    assert validate_folder(str(link)) == root.resolve()
    assert validate_folder(str(root)) == root.resolve()


def test_virtual_environment_folder_stays_excluded(tmp_path):
    root = folder(tmp_path)
    (root / "pyvenv.cfg").touch()
    with pytest.raises(
        ValueError, match="^path is hidden, sensitive, or an excluded directory$"
    ):
        validate_folder(str(root))


@pytest.mark.parametrize("root_kind", ["home", "ancestor", "outside"])
def test_home_and_package_exclusions_for_any_root(tmp_path, monkeypatch, root_kind):
    from rapid_mlx.local_search.extraction import PACKAGE_SUFFIXES

    home = folder(tmp_path, "home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = {"home": home, "ancestor": tmp_path, "outside": folder(tmp_path, "outside")}[
        root_kind
    ]
    library = home / "Library"
    (library / "Application Support").mkdir(parents=True)
    (library / "Application Support" / "private.txt").write_text("secret")
    (root / "good.txt").write_text("apple")
    # Other folders called Library are ordinary user-selected content.
    ordinary_library = root / "Documents" / "Library"
    ordinary_library.mkdir(parents=True)
    (ordinary_library / "good.txt").write_text("banana")
    for suffix in PACKAGE_SUFFIXES:
        package = root / ("Example" + suffix.upper())
        package.mkdir()
        (package / "private.txt").write_text("secret")
        with pytest.raises(ValueError, match="excluded"):
            validate_folder(str(package))
        assert scan(package) == ([], 1)
    for selected_root in (library, library / "Application Support"):
        assert validate_folder(str(selected_root)) == selected_root.resolve()
        assert scan(selected_root) == (
            [library / "Application Support" / "private.txt"],
            0,
        )
    assert validate_folder(str(root)) == root.resolve()
    files, skipped = scan(root)
    assert set(files) == {root / "good.txt", ordinary_library / "good.txt"}
    assert skipped == len(PACKAGE_SUFFIXES) + (root_kind != "outside")

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        assert search.status()["totals"]["files"] == 2
        assert len(fake.calls) == 2
        assert {row["path"] for row in search.store.rows().values()} == {
            str(p) for p in files
        }
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_first", [False, True])
def test_overlapping_folders_reuse_index_without_reembedding(tmp_path, parent_first):
    async def scenario():
        parent = folder(tmp_path)
        child = folder(parent, "docs")
        sibling = folder(parent, "downloads")
        (child / "a.txt").write_text("apple")
        (sibling / "b.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            first, second = (parent, child) if parent_first else (child, parent)
            registered, created = await search.add_folder(str(first))
            assert created
            if not parent_first:
                (await search.add_folder(str(sibling)))
            await search.wait_idle()
            before = search.store.rows()
            vectors = dict(search.store.vectors)
            assert len(fake.calls) == 2
            covering, created = await search.add_folder(str(second))
            assert created == (not parent_first)
            assert (covering["id"] == registered["id"]) == parent_first
            assert len(search.store.folders()) == 1
            assert set(search.store.rows()) == set(before)
            assert all(
                r["folder_id"] == covering["id"] for r in search.store.rows().values()
            )
            await search.wait_idle()
            assert len(fake.calls) == 2  # Absorption preserves both fingerprints.
            assert set(search.store.vectors) == set(vectors)
            alias = tmp_path / "alias"
            alias.symlink_to(child, target_is_directory=True)
            repeated, created = await search.add_folder(str(alias))
            assert repeated["id"] == covering["id"] and not created
            await search.wait_idle()
            assert len(fake.calls) == 2
            for query in ("apple", "banana"):
                hits = (await search.search(query, 20, [covering["id"]]))["results"]
                assert hits and all(hit["folder_id"] == covering["id"] for hit in hits)
            await search.delete_folder(covering["id"])
            assert not search.store.rows() and not search.store.vectors
            assert search.store.totals()["files"] == 0
            assert not (await search.search("apple", 20, None))["results"]
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("active", [False, True])
def test_absorbing_queued_or_active_folder_resumes_under_new_root(
    tmp_path, monkeypatch, active
):
    from rapid_mlx.local_search import service

    async def scenario():
        parent = folder(tmp_path)
        child = folder(parent, "docs")
        (child / "a.txt").write_text("apple")
        later = folder(child, "nested")
        (later / "b.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        paused = asyncio.Event()
        release = asyncio.Event()
        sleep = service.asyncio.sleep

        async def pause(seconds):
            if search.progress["files_done"] == 1 and not paused.is_set():
                paused.set()
                await release.wait()
            await sleep(seconds)

        monkeypatch.setattr(service.asyncio, "sleep", pause)
        start_worker = search._start_worker
        if not active:
            # Keep this case queued until absorption finishes; add_folder's
            # awaits otherwise let it drift into arbitrary active-file work.
            monkeypatch.setattr(search, "_start_worker", lambda: None)
        try:
            old, _ = await search.add_folder(str(child))
            if active:
                await asyncio.wait_for(paused.wait(), SYNC_TIMEOUT)
                assert len(fake.calls) == 1
            new, created = await search.add_folder(str(parent))
            assert created and new["id"] != old["id"]
            assert not search.store.folder(old["id"])
            assert old["id"] not in search.pending
            if not active:
                monkeypatch.setattr(search, "_start_worker", start_worker)
                start_worker()
            release.set()
            await search.wait_idle()
            assert len(fake.calls) == 2
            assert search.store.folder(new["id"])["state"] == "idle"
            assert search.status()["totals"]["files"] == 2
            assert all(
                row["folder_id"] == new["id"] for row in search.store.rows().values()
            )
        finally:
            release.set()
            await search.close()

    asyncio.run(scenario())


def test_pdf_index_and_search_preserve_page_locator(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        path = root / "pages.pdf"
        path.write_bytes(small_pdf())
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == 201
            await app.state.local_search.wait_idle()
            assert fake.calls == [
                ["title: pages | text: First", "title: pages | text: Second"]
            ]
            for query, page in (("First", 1), ("Second", 2)):
                response = await client.post(
                    "/v1/local-search/search", json={"query": query}
                )
                assert response.status_code == 200
                hit = response.json()["results"][0]
                assert "keyword" in hit["matched"]
                assert (
                    hit["path"] == str(path)
                    and hit["page"] == page
                    and hit["line"] is None
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_first", [False, True])
def test_folder_routes_accept_home_and_absorb_overlap(
    tmp_path, monkeypatch, parent_first
):
    home = folder(tmp_path, "home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    docs = folder(home, "Documents")
    (docs / "a.txt").write_text("apple")

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            first, second = (home, docs) if parent_first else (docs, home)
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(first)}
            )
            assert response.status_code == 201
            first_id = response.json()["id"]
            await app.state.local_search.wait_idle()
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(second)}
            )
            assert response.status_code == (200 if parent_first else 201)
            assert (response.json()["id"] == first_id) == parent_first
            assert response.json()["path"] == str(home)
            await app.state.local_search.wait_idle()
            # Accept Whole Mac without walking the machine in a CPU test.
            monkeypatch.setattr(app.state.local_search, "queue", AsyncMock())
            response = await client.post("/v1/local-search/folders", json={"path": "/"})
            assert response.status_code == 201
            assert response.json()["path"] == "/"
            assert len(app.state.local_search.store.folders()) == 1
            assert len(fake.calls) == 1
            assert app.state.local_search.status()["totals"]["files"] == 1

    asyncio.run(scenario())


def test_store_transactions_matrix_model_change(tmp_path):
    root = folder(tmp_path)
    path = root / "a.txt"
    path.write_text("apple 中文检索")
    db = tmp_path / "index.sqlite3"
    store = Store(db, "model-a")
    fid, created = store.add_folder(root)
    assert created and store.add_folder(root) == (fid, False)
    store.replace(
        fid,
        path,
        path.stat(),
        "hash",
        [Chunk(path.read_text(), 1, None)],
        [[1, 0, 0]],
        "indexed",
    )
    row_id = next(iter(store.rows()))
    assert store.semantic([1, 0, 0], {row_id}) == [row_id]
    assert np.shares_memory(store.vectors[row_id], store._matrix)
    assert store.vectors[row_id].dtype == np.float32
    assert store.keyword(["中文检索"], {row_id}) == [row_id]
    assert store.keyword(["中文"], {row_id}) == [row_id]
    assert store.rows([]) == {}
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert len(store.db.execute("SELECT vector FROM chunks").fetchone()[0]) == 6
    store.close()
    store = Store(db, "model-b")
    assert not store.vectors
    assert store.db.execute("SELECT vector FROM chunks").fetchone()[0] is None
    assert next(iter(store.files(fid).values()))["status"] == "pending"
    assert store.keyword(["apple"], set(store.rows()))
    store.delete_folder(fid)
    assert store.totals()["chunks"] == 0
    assert store.db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0
    store.close()


def test_incremental_add_modify_delete_hash_and_restart(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        path = root / "first.txt"
        path.write_text("apple")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.status()["totals"]["files"] == 1
        assert fake.calls == [["title: first | text: apple"]]
        await search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 1
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        await search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 1  # identical hash despite new mtime
        path.write_text("banana")
        (root / "second.md").write_text("apple banana")
        await search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 3 and search.status()["totals"]["files"] == 2
        path.unlink()
        await search.queue(fid)
        await search.wait_idle()
        assert (
            search.status()["totals"]["files"] == 1 and len(search.store.vectors) == 1
        )
        assert all(t != threading.get_ident() for t in fake.threads)
        assert len(set(fake.threads)) == 1
        search.store.set_folder(fid, "indexing")
        await search.close()
        restarted = engine(tmp_path, fake, model_id="changed-model")
        assert not restarted.store.vectors
        restarted.start()
        await restarted.wait_idle()
        assert len(fake.calls) == 4 and len(restarted.store.vectors) == 1
        await restarted.close()

    asyncio.run(scenario())


def test_refresh_pass_selects_only_stale_idle_folders(tmp_path, monkeypatch):
    async def scenario():
        search = engine(tmp_path)
        monkeypatch.setattr(
            "rapid_mlx.local_search.service.time", SimpleNamespace(time=lambda: 10000)
        )
        search.folders = AsyncMock(
            return_value=[
                {"id": fid, "state": state, "last_indexed_at": timestamp}
                for fid, state, timestamp in [
                    ("stale", "idle", 8199),
                    ("never", "idle", None),
                    ("fresh", "idle", 8201),
                    ("boundary", "idle", 8200),
                    ("indexing", "indexing", 0),
                    ("queued", "queued", 0),
                    ("pending", "idle", 0),
                ]
            ]
        )
        assert search.refresh_interval == 1800
        search.pending = ["pending"]
        search.queue = AsyncMock()
        try:
            await search._refresh_stale()
            assert [call.args[0] for call in search.queue.await_args_list] == [
                "stale",
                "never",
            ]
            search.queue.reset_mock()
            search.progress["active"] = True
            await search._refresh_stale()
            search.queue.assert_not_awaited()
            search.progress["active"] = False
            await search.close()
            await search._refresh_stale()
            search.queue.assert_not_awaited()
            assert search.folders.await_count == 1
        finally:
            if not search.closed:
                await search.close()

    asyncio.run(scenario())


def test_refresh_stale_with_real_queue_does_not_deadlock(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        (root / "first.txt").write_text("apple")
        search = engine(tmp_path)
        release = asyncio.Event()

        async def blocked_index(_folder):
            await release.wait()

        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            indexed_at = (await search.folder(fid))["last_indexed_at"]
            monkeypatch.setattr(
                "rapid_mlx.local_search.service.time",
                SimpleNamespace(time=lambda: indexed_at + search.refresh_interval + 1),
            )
            monkeypatch.setattr(search, "_index", blocked_index)

            await asyncio.wait_for(search._refresh_stale(), timeout=5)

            assert fid in search.pending or (
                search.task is not None and not search.task.done()
            )
            assert (await search.folder(fid))["state"] in {"queued", "indexing"}
        finally:
            await search.close()

    asyncio.run(scenario())


def test_refresh_loop_queues_folder_that_becomes_stale_between_ticks(
    tmp_path, monkeypatch
):
    async def scenario():
        search = engine(tmp_path, refresh_interval=1800)
        now = 1800.0
        tick = 60.0
        sleeps = []
        search.folders = AsyncMock(
            return_value=[{"id": "folder", "state": "idle", "last_indexed_at": 61}]
        )

        async def advance_clock(delay):
            nonlocal now
            sleeps.append(delay)
            assert delay == tick
            assert len(sleeps) <= 2
            if len(sleeps) == 2:
                # The first tick was at age 1799, just before the stale threshold.
                search.queue.assert_not_awaited()
                assert search.folders.await_count == 1
            now += delay

        async def queued(_folder_id):
            search.closed = True

        search.queue = AsyncMock(side_effect=queued)
        monkeypatch.setattr("rapid_mlx.local_search.service.REFRESH_TICK_SECONDS", tick)
        monkeypatch.setattr(
            "rapid_mlx.local_search.service.time", SimpleNamespace(time=lambda: now)
        )
        monkeypatch.setattr(
            "rapid_mlx.local_search.service.asyncio",
            SimpleNamespace(sleep=advance_clock, to_thread=asyncio.to_thread),
        )
        try:
            await search._refresh_loop()
            assert sleeps == [tick, tick]
            assert now - 61 == 1859 < search.refresh_interval + tick
            search.queue.assert_awaited_once_with("folder")
        finally:
            search.closed = False
            await search.close()

    asyncio.run(scenario())


def test_refresh_loop_recovers_after_store_error(tmp_path, caplog):
    async def scenario():
        search = engine(tmp_path, refresh_interval=0.01)
        search.folders = AsyncMock(
            side_effect=[
                RuntimeError("store unavailable"),
                [{"id": "stale", "state": "idle", "last_indexed_at": 0}],
            ]
        )
        queued = asyncio.Event()
        search.queue = AsyncMock(side_effect=lambda _: queued.set())
        task = asyncio.create_task(search._refresh_loop())
        search.refresh_task = task
        try:
            await asyncio.wait_for(queued.wait(), SYNC_TIMEOUT)
            assert search.folders.await_count == 2
            search.queue.assert_awaited_once_with("stale")
            assert not task.done()
            assert (
                "Local-search automatic refresh failed: store unavailable"
                in caplog.text
            )
            assert any(
                record.name == "rapid_mlx.local_search.service"
                and record.levelname == "WARNING"
                for record in caplog.records
            )
        finally:
            await search.close()
        assert task.cancelled()

    asyncio.run(scenario())


def test_refresh_discovers_new_files_without_reembedding_or_rewriting_unchanged(
    tmp_path, monkeypatch
):
    async def scenario():
        root = folder(tmp_path)
        (root / "first.txt").write_text("apple")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            indexed_at = (await search.folder(fid))["last_indexed_at"]
            monkeypatch.setattr(
                "rapid_mlx.local_search.service.time",
                SimpleNamespace(time=lambda: indexed_at + search.refresh_interval + 1),
            )
            store_call = search.store.call
            replacements = []

            async def observe_replace(name, *args, **kwargs):
                if name == "replace":
                    replacements.append(args[1])
                return await store_call(name, *args, **kwargs)

            monkeypatch.setattr(search.store, "call", observe_replace)
            await search._refresh_stale()
            await search.wait_idle()
            assert len(fake.calls) == 1
            assert replacements == []
            # Age the next pass too; no manual queue/reindex is used.
            monkeypatch.setattr(
                "rapid_mlx.local_search.service.time",
                SimpleNamespace(
                    time=lambda: indexed_at + 2 * (search.refresh_interval + 1),
                    perf_counter=time.perf_counter,
                ),
            )
            new_file = root / "today-memo.txt"
            new_file.write_text("banana memorandum")
            await search._refresh_stale()
            await search.wait_idle()
            assert fake.calls == [
                ["title: first | text: apple"],
                ["title: today-memo | text: banana memorandum"],
            ]
            assert replacements == [new_file]
            results = await search.search("memorandum", 10, None)
            assert results["results"][0]["path"] == str(new_file)
        finally:
            await search.close()

    asyncio.run(scenario())


def test_refresh_loop_sleeps_when_empty_and_stops_on_close(tmp_path, monkeypatch):
    async def scenario():
        search = engine(tmp_path, refresh_interval=0.01)
        passed = asyncio.Event()
        original = search._refresh_stale

        async def observed_pass():
            await original()
            passed.set()

        monkeypatch.setattr(search, "_refresh_stale", observed_pass)
        search.start()
        try:
            await search.wait_idle()
            await asyncio.wait_for(passed.wait(), SYNC_TIMEOUT)
            assert not search.pending
            assert not search.progress["active"]
            task = search.refresh_task
        finally:
            await search.close()
        assert task.cancelled()
        # Also exercise a loop entered after shutdown: no sleep or store work.
        await search._refresh_loop()

    asyncio.run(scenario())


def test_text_embedding_budget_keyword_tail_and_small_semantic(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        path = root / "run.txt"
        lines = [
            ("tailneedle" if i == 900 else "apple banana").ljust(1190, " ")
            for i in range(1000)
        ]
        path.write_text("\n".join(lines))
        small = root / "small.txt"
        small.write_text("apple")

        class OrchardEmbedder(FakeEmbedder):
            def embed(self, texts):
                vectors = super().embed(texts)
                # Make orchard semantically closer to apple than to a generic
                # shared bias, rather than relying on a fitted cosine floor.
                return [
                    [0.5, 1, 0] if "query: orchard" in text else vector
                    for text, vector in zip(texts, vectors)
                ]

        fake = OrchardEmbedder()
        search = engine(tmp_path, fake)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        large_batches = [
            batch for batch in fake.calls if batch[0].startswith("title: run |")
        ]
        assert len(large_batches) == (MAX_EMBEDDED_CHUNKS_PER_FILE + 3) // 4
        assert sum(map(len, large_batches)) == MAX_EMBEDDED_CHUNKS_PER_FILE
        assert not any(
            "tailneedle" in text for batch in large_batches for text in batch
        )
        totals = search.status()["totals"]
        assert totals["chunks"] == 1001
        assert totals["embedded_chunks"] == MAX_EMBEDDED_CHUNKS_PER_FILE + 1
        assert totals["keyword_only_chunks"] == 1000 - MAX_EMBEDDED_CHUNKS_PER_FILE
        results = (await search.search("tailneedle", 20, None))["results"]
        hit = next(r for r in results if r["path"] == str(path))
        assert "keyword" in hit["matched"] and hit["line"] == 901
        assert "tailneedle" in hit["snippet"] and hit["page"] is None
        assert sum(r["path"] == str(path) for r in results) == 1
        results = (await search.search("orchard", 20, None))["results"]
        assert [r["path"] for r in results] == [str(small), str(path)]
        assert results[0]["matched"] == ["semantic"]
        assert search.store._matrix.dtype == np.float32
        assert len(search.store.vectors) == MAX_EMBEDDED_CHUNKS_PER_FILE + 1
        rows = set(search.store.rows())
        calls = len(fake.calls)
        await search.queue(fid)
        await search.wait_idle()
        # Absorption also reuses mixed vector/keyword files without embedding.
        parent_id = (await search.add_folder(str(tmp_path)))[0]["id"]
        await search.wait_idle()
        assert set(search.store.rows()) == rows and len(fake.calls) == calls
        await search.close()
        resumed = engine(tmp_path, fake, model_id="new-model")
        assert not resumed.store.vectors
        assert resumed.store.keyword(["tailneedle"], set(resumed.store.rows()))
        resumed.start()
        await resumed.wait_idle()
        assert (
            sum(len(b) for b in fake.calls[calls:]) == MAX_EMBEDDED_CHUNKS_PER_FILE + 1
        )
        assert resumed.store.folder(parent_id)["chunks"] == 1001
        await resumed.close()

    asyncio.run(scenario())


def test_existing_index_trims_text_vectors_without_reembedding(tmp_path):
    root = folder(tmp_path)
    path = root / "old.pdf"
    path.touch()
    db = tmp_path / "index.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    count = MAX_EMBEDDED_CHUNKS_PER_FILE + 5
    store.replace(
        fid,
        path,
        path.stat(),
        "hash",
        [
            Chunk(f"page {i} tailneedle" if i == count else f"page {i}", None, i)
            for i in range(1, count + 1)
        ],
        [[1.0, 0.0]] * count,
        "indexed",
    )
    media = root / "video.mp4"
    media.touch()
    store.replace(
        fid,
        media,
        media.stat(),
        None,
        [Chunk("", None, None, "video", float(i)) for i in range(count)],
        [[0.0, 1.0]] * count,
        "indexed",
    )
    ids = set(store.rows())
    store.close()
    store = Store(db, "model")
    assert set(store.rows()) == ids
    assert len(store.vectors) == MAX_EMBEDDED_CHUNKS_PER_FILE + count
    assert store.files(fid)[str(path)]["status"] == "indexed"
    tail = store.keyword(["tailneedle"], ids)[0]
    assert tail not in store.vectors
    assert f'"page": {count}' in store.rows()[tail]["locator"]
    keyword = root / "keyword.txt"
    keyword.touch()
    store.replace(
        fid,
        keyword,
        keyword.stat(),
        "hash",
        [Chunk("keywordonly", 1, None)],
        None,
        "keyword",
    )
    keyword_ids = set(store.keyword(["keywordonly"], set(store.rows())))
    store.close()
    store = Store(db, "new-model")
    assert not store.vectors
    assert store.files(fid)[str(keyword)]["status"] == "keyword"
    assert set(store.keyword(["keywordonly"], set(store.rows()))) == keyword_ids
    assert tail in store.rows() and tail not in store.vectors
    store.close()


def test_single_file_yields_and_cancels_between_embedding_batches(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "large.txt").write_text("apple\n" * 20000)
        fake = FakeEmbedder()
        first_batch = asyncio.Event()
        loop = asyncio.get_running_loop()
        embed = fake.embed

        def observed(texts):
            vectors = embed(texts)
            loop.call_soon_threadsafe(first_batch.set)
            return vectors

        fake.embed = observed
        search = engine(tmp_path, fake)

        async def chat_and_cancel():
            await first_batch.wait()
            # A competing task runs before batch two of the same file.
            assert len(fake.calls) == 1
            search.task.cancel()

        chat = asyncio.create_task(chat_and_cancel())
        fid = (await search.add_folder(str(root)))[0]["id"]
        with pytest.raises(asyncio.CancelledError):
            await search.wait_idle()
        await chat
        assert len(fake.calls) == 1 and search.store.folder(fid)["chunks"] == 0
        await search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == 1 + MAX_EMBEDDED_CHUNKS_PER_FILE // search.batch_size
        await search.close()

    asyncio.run(scenario())


def test_missing_model_keyword_and_resume(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("中文检索 apple")
        available = [False]
        fake = FakeEmbedder()
        search = LocalSearch(
            tmp_path / "index",
            resolve_model=lambda _: "local" if available[0] else None,
            embedder_factory=lambda _: fake,
            yield_seconds=0,
        )
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        status = search.status()
        assert status["folders"][0]["state"] == "waiting_for_model"
        assert status["model"]["downloaded"] is False and not fake.calls
        settled_task = search.task
        for _ in range(10):
            await search.status_async()
            await search.wait_idle()
            assert search.task is settled_task
        result = await search.search("中文", 20, None)
        assert result["mode"] == "keyword" and result["results"][0]["matched"] == [
            "keyword"
        ]
        available[0] = True
        resumed = await search.status_async()
        assert resumed["model"]["downloaded"] is True
        assert resumed["folders"][0]["state"] in {"queued", "indexing"}
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "idle" and fake.calls
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("media", [False, True], ids=["text", "image"])
@pytest.mark.parametrize(
    "failure",
    [
        "input",
        "too_long",
        "placeholder",
        "invalid_output",
        "zero_output",
        "lazy_load",
        "factory",
        "resolve",
    ],
)
def test_embedding_failure_scope(tmp_path, monkeypatch, media, failure):
    from unittest.mock import Mock

    from PIL import Image

    EmbeddingInputTooLongError = cpu_embedding_module(
        monkeypatch
    ).EmbeddingInputTooLongError
    from rapid_mlx.local_search import service as module

    async def scenario():
        root = folder(tmp_path)
        paths = [root / (f"{name}.png" if media else f"{name}.txt") for name in "abc"]
        for path in paths:
            if media:
                Image.new("RGB", (4, 4)).save(path)
            else:
                path.write_text("An apple orchard grows delicious fruit.")
        # Explicit discovery order makes the continuation contract deterministic.
        monkeypatch.setattr(
            module, "scan_batches", lambda *_: (batch for batch in [(paths, 0)])
        )
        fake = FakeEmbedder()
        original = fake.embed_images if media else fake.embed
        calls = []
        loop_thread = threading.get_ident()

        def failing_input(inputs):
            assert threading.get_ident() != loop_thread
            calls.append(search.progress["current_path"])
            if len(calls) == 1:
                if failure == "invalid_output":
                    return [[float("nan"), 1.0] for _ in inputs]
                if failure == "zero_output":
                    return [[0.0, 0.0] for _ in inputs]
                if failure == "too_long":
                    raise EmbeddingInputTooLongError(
                        observed_tokens=100, allowed_tokens=10, index=0
                    )
                if failure == "placeholder":
                    raise ValueError("Tokenizer failed to expand image placeholders")
                if failure == "input":
                    from rapid_mlx.local_search.service import EmbeddingInputError

                    raise EmbeddingInputError("invalid embedding input")
                raise ValueError("model execution failed")
            return original(inputs)

        monkeypatch.setattr(fake, "embed_images" if media else "embed", failing_input)
        search = engine(tmp_path, fake)
        factory = Mock(return_value=fake)
        if failure == "factory":
            factory.side_effect = ValueError("device unavailable")
        search.embedder_factory = factory
        if failure == "resolve":
            resolve = search.resolve_model

            def failing_resolve(model_id):
                if threading.current_thread().name.startswith("test-model-worker"):
                    raise ValueError("device unavailable")
                return resolve(model_id)

            monkeypatch.setattr(search, "resolve_model", failing_resolve)
        release = Mock(wraps=search._release_embedder)
        monkeypatch.setattr(search, "_release_embedder", release)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            files = search.store.files(fid)
            state = search.store.folder(fid)
            assert files[str(paths[0])]["status"] == "keyword"
            assert files[str(paths[0])]["embedded"] == 0
            if failure in {"factory", "resolve", "lazy_load"}:
                assert calls == (
                    [] if failure in {"factory", "resolve"} else [str(paths[0])]
                )
                assert all(row["embedded"] == 0 for row in files.values())
                assert state["state"] == "waiting_for_model"
                assert state["error"] == (
                    "model execution failed"
                    if failure == "lazy_load"
                    else "device unavailable"
                )
                assert search.engine is None
                assert fid in search._attempted_checkpoints
                assert release.called
            else:
                assert calls == [str(path) for path in paths]
                assert all(files[str(path)]["embedded"] > 0 for path in paths[1:])
                assert state["state"] == "idle"
                assert state["error"] is None
                assert search.worker_error is None
                assert search.engine is fake
                assert factory.call_count == 1
                assert fid not in search._attempted_checkpoints
                release.assert_not_called()
            result = await search.search(paths[0].name, 20, None)
            assert any(hit["path"] == str(paths[0]) for hit in result["results"])
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_missing_worker_is_model_failure(tmp_path, monkeypatch, resident, background):
    from types import SimpleNamespace

    from rapid_mlx.config import get_config
    from rapid_mlx.local_search.service import ModelLoadError

    async def scenario():
        search = engine(tmp_path, FakeEmbedder())
        if resident:
            await search._embed_async(["warmup"])
        monkeypatch.setattr(get_config(), "engine", SimpleNamespace())
        try:
            with pytest.raises(ModelLoadError) as caught:
                await search._embed_async(["apple"], background=background)
            assert not caught.value.load_failed
            assert search.engine is None
            assert search.worker_error == str(caught.value)
            assert not search._attempted_checkpoints
        finally:
            await search.close()

    asyncio.run(scenario())


def test_loaded_embedder_recovers_after_worker_replacement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx.config import get_config

    async def scenario():
        root = folder(tmp_path)
        first = root / "first.txt"
        first.write_text("An apple orchard grows delicious fruit.")
        model = folder(tmp_path, "model")
        (model / "model.safetensors").write_bytes(b"unchanged checkpoint")
        cfg = get_config()
        original_worker = cfg.engine.execute_on_model_worker
        calls = []

        async def worker(name, func):
            calls.append(name)
            return await original_worker(func)

        async def worker_a(func):
            return await worker("a", func)

        async def worker_b(func):
            return await worker("b", func)

        monkeypatch.setattr(
            cfg, "engine", SimpleNamespace(execute_on_model_worker=worker_a)
        )
        fake = FakeEmbedder()
        search = LocalSearch(
            tmp_path / "index",
            resolve_model=lambda _: str(model),
            embedder_factory=lambda _: fake,
            yield_seconds=0,
        )
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            assert search.engine is fake
            second = root / "second.txt"
            second.write_text("A banana orchard grows delicious fruit.")
            await search.store.call("set_folder", fid, "waiting_for_model")
            monkeypatch.setattr(
                cfg, "engine", SimpleNamespace(execute_on_model_worker=worker_b)
            )

            status = await search.status_async()
            assert status["folders"][0]["state"] in {"queued", "indexing"}
            await search.wait_idle()
            assert search.store.folder(fid)["state"] == "idle"
            assert search.store.files(fid)[str(second)]["embedded"] > 0
            assert "a" in calls and calls[-1] == "b"
            assert search._owner_worker == worker_b
        finally:
            await search.close()

    asyncio.run(scenario())


def test_worker_refresh_defensive_paths(tmp_path, monkeypatch):
    async def old_worker(work):
        return work()

    async def new_worker(work):
        return work()

    async def scenario():
        search = engine(tmp_path)
        try:
            search.engine = FakeEmbedder()
            search._owner_worker = old_worker
            workers = iter((new_worker, old_worker))
            monkeypatch.setattr(search, "_worker", lambda: next(workers))
            assert await search._refresh_worker() is old_worker

            search._owner_worker = None
            monkeypatch.setattr(search, "_worker", lambda: new_worker)
            assert await search._refresh_worker() is None
            assert search.worker_error == "Model worker changed before teardown"

            async def broken_worker(_):
                raise RuntimeError("dispatch unavailable")

            search._owner_worker = broken_worker
            assert await search._refresh_worker() is None
            assert search.worker_error == "dispatch unavailable"
        finally:
            search.engine = None
            search._owner_worker = None
            await search.close()

    asyncio.run(scenario())


def test_missing_worker_releases_resident_engine_on_owner(tmp_path, monkeypatch):
    owner_calls = []

    async def owner(work):
        owner_calls.append(work)
        return work()

    async def scenario():
        search = engine(tmp_path)
        search.engine = FakeEmbedder()
        search._owner_worker = owner
        monkeypatch.setattr(search, "_worker", lambda: None)
        try:
            with pytest.raises(ModelLoadError, match="no sanctioned model worker"):
                await search._model_call_owned(lambda: None)
            assert search.engine is None
            assert len(owner_calls) == 1
        finally:
            await search.close()

    asyncio.run(scenario())


def test_release_after_model_error_is_best_effort(tmp_path, monkeypatch):
    async def scenario():
        search = engine(tmp_path)
        try:
            monkeypatch.setattr(
                search,
                "_model_call",
                AsyncMock(side_effect=ModelLoadError("teardown unavailable")),
            )
            await search._release_after_model_error()
            search.engine = FakeEmbedder()
            await search._release_after_model_error()
            assert search.engine is not None
            assert search._model_call.await_count == 1
        finally:
            search.engine = None
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change",
    ["path", "directory_mtime", "file_mtime", "replacement", "blob", "absent_present"],
)
def test_failed_checkpoint_polls_wait_for_change(tmp_path, monkeypatch, change):
    from rapid_mlx.local_search import service as module

    async def scenario():
        root = folder(tmp_path)
        (root / "apple.txt").write_text("An apple orchard grows delicious fruit.")
        model = folder(tmp_path, "model")
        weights = model / "model.safetensors"
        blob = tmp_path / "blob"
        blob.write_bytes(b"broken checkpoint")
        if change == "blob":
            weights.symlink_to(blob)
        else:
            weights.write_bytes(b"broken checkpoint")
        model_path = [str(model)]
        broken = [True]
        fake = FakeEmbedder()

        def factory(_):
            if broken[0]:
                raise ValueError("incompatible checkpoint")
            return fake

        search = LocalSearch(
            tmp_path / "index",
            resolve_model=lambda _: model_path[0],
            embedder_factory=factory,
            yield_seconds=0,
        )
        index = AsyncMock(wraps=search._index)
        monkeypatch.setattr(search, "_index", index)
        walks = []
        scan = module.scan_batches

        def counted_scan(*args):
            walks.append(args[0])
            return scan(*args)

        monkeypatch.setattr(module, "scan_batches", counted_scan)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            for _ in range(10):
                status = await search.status_async()
                await search.wait_idle()
                assert status["folders"][0]["state"] == "waiting_for_model"
                assert status["folders"][0]["error"] == "incompatible checkpoint"
                assert (
                    status["semantic_unavailable_reason"] == "incompatible checkpoint"
                )
                assert index.await_count == len(walks) == 1

            if change == "path":
                model_path[0] = str(folder(tmp_path, "new-model"))
            elif change == "absent_present":
                model_path[0] = None
                status = await search.status_async()
                await search.wait_idle()
                assert status["folders"][0]["error"] == "incompatible checkpoint"
                assert index.await_count == len(walks) == 1
                model_path[0] = str(model)
            elif change in {"directory_mtime", "file_mtime", "blob"}:
                target = (
                    model
                    if change == "directory_mtime"
                    else blob
                    if change == "blob"
                    else weights
                )
                info = target.stat()
                os.utime(
                    target, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000)
                )
            else:
                replacement = tmp_path / "replacement"
                replacement.write_bytes(weights.read_bytes())
                replacement.replace(weights)
            broken[0] = False
            await search.status_async()
            await search.wait_idle()
            assert search.store.folder(fid)["state"] == "idle"
            assert search.store.folder(fid)["error"] is None
            assert fake.calls
            for _ in range(10):
                status = await search.status_async()
                await search.wait_idle()
                assert status["semantic_unavailable_reason"] is None
                assert index.await_count == len(walks) == 2
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["reindex", "add"])
def test_failed_checkpoint_explicit_action_retries(tmp_path, monkeypatch, action):
    from rapid_mlx.routes.local_search import router

    async def scenario():
        root = folder(tmp_path)
        (root / "apple.txt").write_text("An apple orchard grows delicious fruit.")
        search = engine(tmp_path)

        def fail(_):
            raise ValueError("corrupt checkpoint")

        search.embedder_factory = fail
        index = AsyncMock(wraps=search._index)
        monkeypatch.setattr(search, "_index", index)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            await search.status_async()
            await search.wait_idle()
            assert index.await_count == 1
            if action == "add":
                registered, created = await search.add_folder(str(root))
                assert registered["id"] == fid and not created
            else:
                app = FastAPI()
                app.include_router(router)
                app.state.local_search = search
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://localhost"
                ) as client:
                    response = await client.post(
                        "/v1/local-search/reindex", json={"folder_id": fid}
                    )
                assert response.status_code == 202
            await search.wait_idle()
            assert index.await_count == 2
            status = await search.status_async()
            await search.wait_idle()
            assert index.await_count == 2
            assert status["folders"][0]["state"] == "waiting_for_model"
            assert status["folders"][0]["error"] == "corrupt checkpoint"
        finally:
            await search.close()

    asyncio.run(scenario())


def test_batches_yield_cancel_and_resume(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "long.txt").write_text("apple\n" * 4000)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake, batch_size=2)
        entered, release = pause_after_embedding(search)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await asyncio.wait_for(entered.wait(), SYNC_TIMEOUT)
        await search.close()
        release.set()
        assert search.store.path.exists()
        resumed = engine(tmp_path, fake, batch_size=2)
        resumed.start()
        await resumed.wait_idle()
        assert resumed.store.folder(fid)["chunks"] > 10
        assert all(len(batch) <= 2 for batch in fake.calls)
        assert all(
            len(batch[i]) <= len(batch[i + 1])
            for batch in fake.calls
            for i in range(len(batch) - 1)
        )
        await resumed.close()

    asyncio.run(scenario())


def test_fusion_exact_bonus_and_highlights():
    ranking = fuse([1, 2, 3], [3, 4])
    assert ranking[0][0] == 3 and ranking[0][2] == ["semantic", "keyword"]
    assert next(s for i, s, _ in ranking if i == 1) > next(
        s for i, s, _ in ranking if i == 4
    )
    assert {i: score for i, score, _ in fuse([1, 2, 3], [4, 5, 6])} == {
        i: score for i, score, _ in fuse([4, 5, 6], [1, 2, 3])
    }
    assert all(0 <= s <= 1 for _, s, _ in ranking)
    text, offsets = snippet(
        "Earlier sentence. " * 45 + "Apple 中文检索 " + "Later sentence. " * 45,
        ["apple", "中文"],
    )
    assert len(text) <= 500 and [text[a:b] for a, b in offsets] == ["Apple", "中文"]
    text, offsets = snippet("Straße apple", ["apple"])
    assert [text[a:b] for a, b in offsets] == ["apple"]


def test_search_collapses_filters_fallback(tmp_path):
    async def scenario():
        roots = [folder(tmp_path, "one"), folder(tmp_path, "two")]
        for root in roots:
            (root / "a.txt").write_text("apple\n" * 600)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        ids = [(await search.add_folder(str(root)))[0]["id"] for root in roots]
        await search.wait_idle()
        result = await search.search("apple", 20, None)
        assert result["mode"] == "hybrid" and len(result["results"]) == 2
        assert fake.calls[-1] == ["task: search result | query: apple"]
        result = await search.search("apple", 1, [ids[1]])
        assert (
            len(result["results"]) == 1 and result["results"][0]["folder_id"] == ids[1]
        )
        assert (await search.search("apple", 20, []))["results"] == []
        fake.embed = lambda _: (_ for _ in ()).throw(RuntimeError("unavailable"))
        assert (await search.search("apple", 20, None))["mode"] == "keyword"
        await search.close()

    asyncio.run(scenario())


def test_skipped_missing_and_bad_documents(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "bad.docx").write_bytes(b"not-a-zip")
        (root / "bad.txt").write_bytes(b"\0binary")
        (root / "unknown.bin").write_bytes(b"unknown")
        (root / "good.txt").write_text("apple")
        search = engine(tmp_path)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["files_skipped"] == 3
        assert search.store.folder(fid)["files_indexed"] == 1
        await search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["files_skipped"] == 3
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
        await search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "missing"
        assert not (await search.search("apple", 20, None))["results"]
        await search.close()

    asyncio.run(scenario())


def runnable_text_snapshot(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"embedding_gemma2"}')
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    return path


@pytest.mark.parametrize("downloads_disabled", [False, True])
def test_local_availability_never_downloads(tmp_path, monkeypatch, downloads_disabled):
    import huggingface_hub

    from rapid_mlx import cli, model_downloads

    monkeypatch.setattr(model_downloads, "disabled", lambda: downloads_disabled)

    monkeypatch.setattr(cli, "_cache_runnability", lambda _: False)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: pytest.fail("missing model must not resolve/download"),
    )
    assert local_model_path("missing") is None
    monkeypatch.setattr(cli, "_cache_runnability", lambda _: True)
    calls = []
    model_path = runnable_text_snapshot(tmp_path / "model")

    def snapshot(model, **kwargs):
        calls.append(kwargs)
        return str(model_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert local_model_path("cached") == str(model_path)
    assert calls == [{"local_files_only": True}]


@pytest.mark.parametrize("downloads_disabled", [False, True])
def test_local_availability_cache_miss_fails_closed(monkeypatch, downloads_disabled):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    from rapid_mlx import cli, model_downloads, model_metadata

    monkeypatch.setattr(model_downloads, "disabled", lambda: downloads_disabled)
    monkeypatch.setattr(cli, "_cache_runnability", lambda _: True)
    monkeypatch.setattr(
        model_metadata, "resolve_unreferenced_cached_snapshot", lambda _: None
    )
    monkeypatch.setattr(
        model_metadata, "resolve_offline_cached_snapshot", lambda _: None
    )
    monkeypatch.setattr(model_metadata, "_cached_file", lambda *args: None)
    calls = []

    def missing(model, **kwargs):
        calls.append((model, kwargs))
        raise LocalEntryNotFoundError("cache entry disappeared")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    assert local_model_path("test/missing") is None
    assert calls == [("test/missing", {"local_files_only": True})]


@asynccontextmanager
async def client_app(tmp_path, monkeypatch, enabled=True):
    import rapid_mlx.local_search.service as module
    from rapid_mlx.local_search import runtime

    fake = FakeEmbedder()
    monkeypatch.setattr(module, "LocalSearch", lambda directory: engine(tmp_path, fake))
    runtime.configure(enabled, str(tmp_path / "index"))
    app = FastAPI()
    install_exception_handlers(app)
    await runtime.startup(app)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            yield client, app, fake
    finally:
        await runtime.shutdown(app)
        runtime.configure(False)


def test_routes_and_frozen_shapes(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple banana")
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            status = (await client.get("/v1/local-search/status")).json()
            assert set(status) == {
                "enabled",
                "model",
                "folders",
                "indexing",
                "totals",
                "semantic_unavailable_reason",
            }
            assert status["model"]["loaded"] is False and not fake.calls
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == 201
            folder_data = response.json()
            assert set(folder_data) == {
                "id",
                "path",
                "state",
                "files_indexed",
                "files_skipped",
                "skip_reasons",
                "chunks",
                "last_indexed_at",
                "error",
            }
            fid = folder_data["id"]
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == 200 and response.json()["id"] == fid
            await app.state.local_search.wait_idle()
            response = await client.post(
                "/v1/local-search/search", json={"query": "apple"}
            )
            assert response.status_code == 200
            body = response.json()
            assert set(body) == {"results", "took_ms", "mode"}
            result = body["results"][0]
            assert set(result) == {
                "kind",
                "timestamp",
                "content_indexed",
                "skip_reason",
                "path",
                "folder_id",
                "title",
                "snippet",
                "highlights",
                "score",
                "matched",
                "page",
                "line",
                "modified_at",
                "size",
            }
            assert result["page"] is None and result["line"] == 1
            assert result["snippet"][slice(*result["highlights"][0])] == "apple"
            for payload in ({"folder_id": fid}, {"folder_id": None}, {}):
                response = await client.post("/v1/local-search/reindex", json=payload)
                assert (
                    response.status_code == 202 and response.json()["enabled"] is True
                )
                await app.state.local_search.wait_idle()
            response = await client.delete(f"/v1/local-search/folders/{fid}")
            assert response.status_code == 204 and not response.content
            assert (await client.get("/v1/local-search/status")).json()["totals"][
                "chunks"
            ] == 0
            response = await client.delete(f"/v1/local-search/folders/{fid}")
            assert (
                response.status_code == 404
                and response.json()["error"]["code"] == "folder_not_found"
            )

    asyncio.run(scenario())


def test_flag_gating_and_cli(tmp_path, monkeypatch):
    from rapid_mlx.cli import build_parser

    args = build_parser().parse_args(["serve", "chat-model"])
    assert args.enable_local_search is False and args.local_search_dir is None
    args = build_parser().parse_args(
        ["serve", "chat-model", "--enable-local-search", "--local-search-dir", "/index"]
    )
    assert args.enable_local_search and args.local_search_dir == "/index"

    async def scenario():
        async with client_app(tmp_path, monkeypatch, enabled=False) as (client, app, _):
            for method, url, body in (
                ("GET", "status", None),
                ("POST", "folders", {"path": "/"}),
                ("DELETE", "folders/no-id", None),
                ("POST", "reindex", {}),
                ("POST", "search", {"query": "apple"}),
            ):
                response = await client.request(
                    method, "/v1/local-search/" + url, json=body
                )
                assert response.status_code == 404
            assert not any(
                "/v1/local-search" in getattr(r, "path", "") for r in app.routes
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "endpoint,body",
    [
        ("folders", {"path": ".ssh"}),
        ("folders", {"path": "relative"}),
        ("search", {"query": ""}),
        ("search", {"query": "   "}),
        ("search", {"query": "a", "limit": 0}),
        ("search", {"query": "a", "limit": 101}),
        ("search", {"query": "apple", "folder_ids": ["unknown"]}),
        ("reindex", {"folder_id": "unknown"}),
    ],
)
def test_route_errors(tmp_path, monkeypatch, endpoint, body):
    if body.get("path") == ".ssh":
        body = {"path": str(folder(tmp_path, ".ssh"))}

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, _, _):
            response = await client.post("/v1/local-search/" + endpoint, json=body)
            assert response.status_code in (400, 404)
            assert set(response.json()["error"]) == {"message", "type", "code", "param"}

    asyncio.run(scenario())


def test_routes_share_auth_rate_dependencies(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config
    from rapid_mlx.middleware import auth

    cfg = get_config()
    monkeypatch.setattr(cfg, "api_key", "test-key")
    monkeypatch.setattr(auth, "rate_limiter", auth.RateLimiter(1, enabled=True))

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, _, _):
            response = await client.get("/v1/local-search/status")
            assert response.status_code == 401
            response = await client.get(
                "/v1/local-search/status", headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code == 200
            response = await client.post(
                "/v1/local-search/search",
                json={"query": "apple"},
                headers={"Authorization": "Bearer test-key"},
            )
            assert response.status_code == 429

    asyncio.run(scenario())


def test_default_state_directory(tmp_path, monkeypatch):
    from rapid_mlx.local_search.runtime import default_directory

    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    assert default_directory() == tmp_path / "state" / "local-search"


def test_failed_replacement_preserves_index(tmp_path):
    root = folder(tmp_path)
    path = root / "a.txt"
    path.write_text("apple")
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    chunks = [Chunk("apple", 1, None)]
    store.replace(fid, path, path.stat(), "before", chunks, [[1.0, 0.0]], "indexed")
    before = set(store.rows())
    with pytest.raises(IndexError):
        store.replace(
            fid, path, path.stat(), "after", chunks * 2, [[0.0, 1.0]], "indexed"
        )
    assert set(store.rows()) == before and set(store.vectors) == before
    assert store.keyword(["apple"], before)
    store.close()


@pytest.mark.parametrize("already_pending", [False, True])
def test_add_folder_cancels_and_requeues_active_once(
    tmp_path, monkeypatch, already_pending
):
    async def scenario():
        one = folder(tmp_path, "one")
        two = folder(tmp_path, "two")
        (one / "apple.txt").write_text("apple orchard")
        (one / "banana.txt").write_text("banana grove")
        (two / "cherry.txt").write_text("cherry harvest")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake, batch_size=1)
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        embed = search._embed_async
        index = search._index
        start_worker = search._start_worker
        attempts, queued = [], []
        calls = 0

        async def paused_embed(inputs, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # The first file is committed; the second has not been embedded.
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                    raise
            return await embed(inputs, **kwargs)

        async def observed_index(registered):
            attempts.append(registered["id"])
            await index(registered)

        def observed_start_worker():
            queued.append(list(search.pending))
            start_worker()

        monkeypatch.setattr(search, "_embed_async", paused_embed)
        monkeypatch.setattr(search, "_index", observed_index)
        adding = None
        try:
            first = (await search.add_folder(str(one)))[0]["id"]
            await asyncio.wait_for(entered.wait(), SYNC_TIMEOUT)
            worker = search.task
            assert not worker.done()
            assert search.progress["folder_id"] == first
            assert search.store.folder(first)["files_indexed"] == 1
            assert len(fake.calls) == 1
            assert search.pending == []
            if already_pending:
                # Public queue() can enqueue an active root for another pass.
                await search.queue(first)
            monkeypatch.setattr(search, "_start_worker", observed_start_worker)
            adding = asyncio.create_task(search.add_folder(str(two)))
            await asyncio.wait_for(cancelled.wait(), SYNC_TIMEOUT)
            # Task.cancelling() needs Python 3.11; the event proves cancellation.
            assert not worker.done()
            assert not adding.done()
            assert len(fake.calls) == 1
            release.set()
            second = (await adding)[0]["id"]
            assert worker.cancelled()
            assert queued == [[first, second]]
            await search.wait_idle()
            assert attempts == [first, first, second]
            assert search.pending == []
            for fid, count in ((first, 2), (second, 1)):
                registered = search.store.folder(fid)
                assert registered["state"] == "idle"
                assert registered["files_indexed"] == count
                assert all(
                    row["status"] == "indexed"
                    for row in search.store.files(fid).values()
                )
            assert len(search.store.vectors) == 3
            assert sorted(text for batch in fake.calls for text in batch) == [
                "title: apple | text: apple orchard",
                "title: banana | text: banana grove",
                "title: cherry | text: cherry harvest",
            ]
        finally:
            release.set()
            if adding is not None:
                await adding
            await search.close()

    asyncio.run(scenario())


def test_active_delete_keeps_next_folder_job(tmp_path):
    async def scenario():
        one = folder(tmp_path, "one")
        two = folder(tmp_path, "two")
        (one / "long.txt").write_text("apple\n" * 4000)
        (two / "short.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake, batch_size=1)
        entered, release = pause_after_embedding(search)
        first = (await search.add_folder(str(one)))[0]["id"]
        second = (await search.add_folder(str(two)))[0]["id"]
        await asyncio.wait_for(entered.wait(), SYNC_TIMEOUT)
        await search.delete_folder(first)
        release.set()
        await search.wait_idle()
        assert search.store.folder(first) is None
        assert search.store.folder(second)["files_indexed"] == 1
        assert len(search.store.vectors) == 1
        await search.close()

    asyncio.run(scenario())


def test_replaced_folder_symlink_is_rejected(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        search = engine(tmp_path)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        saved = tmp_path / "saved"
        root.rename(saved)
        root.symlink_to(saved, target_is_directory=True)
        assert (await search.search("apple", 20, None))["results"] == []
        await search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["state"] in {"error", "missing"}
        await search.close()

    asyncio.run(scenario())


def test_invalid_output_keeps_keyword_text(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        fake = FakeEmbedder()
        fake.embed = lambda texts: [[float("nan")]] * len(texts)
        search = engine(tmp_path, fake)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "idle"
        assert search.store.folder(fid)["error"] is None
        assert search.engine is fake
        assert not search.store.vectors
        result = await search.search("apple", 20, None)
        assert result["mode"] == "keyword" and len(result["results"]) == 1
        await search.close()

    asyncio.run(scenario())


def test_routes_keyword_without_model(tmp_path, monkeypatch):
    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("苹果 搜索")
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            app.state.local_search.resolve_model = lambda _: None
            await client.post("/v1/local-search/folders", json={"path": str(root)})
            await app.state.local_search.wait_idle()
            status = (await client.get("/v1/local-search/status")).json()
            assert status["model"] == {
                "id": app.state.local_search.model_id,
                "downloaded": False,
                "loaded": False,
                "vision": False,
            }
            assert status["folders"][0]["state"] == "waiting_for_model"
            result = (
                await client.post("/v1/local-search/search", json={"query": "苹果"})
            ).json()
            assert result["mode"] == "keyword" and len(result["results"]) == 1
            assert not fake.calls

    asyncio.run(scenario())


def test_keyword_limits_do_not_starve_other_files(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "long.txt").write_text("apple\n" * 22000)
        (root / "short.txt").write_text("apple")
        search = LocalSearch(
            tmp_path / "index", resolve_model=lambda _: None, yield_seconds=0
        )
        (await search.add_folder(str(root)))
        await search.wait_idle()
        result = await search.search("apple", 20, None)
        assert {Path(r["path"]).name for r in result["results"]} == {
            "long.txt",
            "short.txt",
        }
        await search.close()

    asyncio.run(scenario())


def test_unreferenced_available_snapshot_uses_existing_resolver(tmp_path, monkeypatch):
    import huggingface_hub

    from rapid_mlx import cli, model_metadata

    model_path = runnable_text_snapshot(tmp_path / "model")
    monkeypatch.setattr(cli, "_cache_runnability", lambda _: True)
    monkeypatch.setattr(
        model_metadata,
        "resolve_unreferenced_cached_snapshot",
        lambda _: model_path,
    )
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **kw: pytest.fail("already resolved locally"),
    )
    assert local_model_path("test/model") == str(model_path)


@pytest.mark.parametrize("identifier", [False, True])
def test_precise_hybrid_ranking_and_highlights(tmp_path, identifier):
    class RankedEmbedder(FakeEmbedder):
        def embed(self, texts):
            self.is_loaded = True
            vectors = []
            for text in texts:
                if text.startswith("task:"):
                    vectors.append([1.0, 0.0])
                else:
                    name = text.split(" | ")[0].removeprefix("title: ")
                    rank = int(name[-1])
                    vectors.append([1.0, rank / 5.0])
            return vectors

    async def scenario():
        root = folder(tmp_path)
        common = "how do I roll back a bad "
        target = 2 if identifier else 0
        for i in range(6):
            content = common + "configuration"
            if i == target:
                content += " RAPID_MLX_HOME" if identifier else " release"
            (root / f"guide{i}.md").write_text(content)
        search = engine(tmp_path, RankedEmbedder())
        (await search.add_folder(str(root)))
        await search.wait_idle()
        query = "how do I RAPID_MLX_HOME" if identifier else common + "release"
        rows = search.store.rows()
        vector = search._embed(["task: query"])[0]
        pure = search.store.semantic(vector, set(rows))
        assert Path(rows[pure[target]]["path"]).name == f"guide{target}.md"
        result = await search.search(query, 4, None)
        assert Path(result["results"][0]["path"]).name == f"guide{target}.md"
        first = result["results"][0]
        assert first["matched"] == ["semantic", "keyword"]
        assert [first["snippet"][a:b] for a, b in first["highlights"]] == [
            "RAPID_MLX_HOME" if identifier else "release"
        ]
        await search.close()

    asyncio.run(scenario())


def test_no_distinctive_terms_require_literal_phrase(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for i in range(4):
        path = root / f"guide{i}.txt"
        path.write_text("how to do a rollback")
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk(path.read_text(), 1, None)],
            None,
            "keyword",
        )
    allowed = set(store.rows())
    assert store.distinctive_terms(["how", "do", "a", "rollback"]) == []
    assert (
        store.keyword(
            ["how", "do", "a", "rollback"], allowed, query="how do a rollback"
        )
        == []
    )
    assert store.keyword(["how", "to"], allowed, query="how to")
    assert store.keyword(["how", "to"], allowed, query="how-to") == []
    store.close()


def test_distinctive_all_terms_cjk_and_filename(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for name, text in [
        ("RAPID_MLX_HOME.txt", "苹果 错误 code42"),
        ("other.txt", "香蕉 code42"),
        ("third.txt", "普通内容"),
    ]:
        path = root / name
        path.write_text(text)
        store.replace(
            fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword"
        )
    rows = store.rows()
    allowed = set(rows)
    expected = [i for i, row in rows.items() if "HOME" in row["path"]]
    assert store.keyword(["苹果", "错误"], allowed) == expected
    assert store.keyword(["苹果", "香蕉"], allowed) == []
    assert store.keyword(["错误", "RAPID_MLX_HOME"], allowed) == expected
    assert store.keyword(["RAPID_MLX_HOME"], allowed) == expected
    assert store.keyword(["missing", "RAPID_MLX_HOME"], allowed) == []
    store.close()


def test_fusion_combines_different_chunks_of_same_file():
    rows = {1: {"path": "right"}, 2: {"path": "other"}, 3: {"path": "right"}}
    ranking = fuse([1, 2], [3], rows=rows)
    assert ranking[0][0] == 3
    assert ranking[0][1] == pytest.approx(1.0)
    assert ranking[0][2] == ["semantic", "keyword"]
    assert len(ranking) == 2
    assert fuse([1], [2])[0][1] == fuse([2], [1])[0][1]
    assert fuse([1], [1])[0][1] > fuse([2], [1])[0][1]


def write_video(path, colors=(0, 0, 255), fps=2, seconds=12):
    cv2 = pytest.importorskip("cv2")

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert writer.isOpened(), "test codec must be available"
    try:
        for frame in range(fps * seconds):
            color = colors[min(len(colors) - 1, int(frame / fps / 5))]
            writer.write(np.full((32, 32, 3), color, dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.parametrize(
    "suffix", ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]
)
def test_image_extraction_first_frame(tmp_path, suffix):
    from PIL import Image

    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / ("picture." + suffix)
    red = Image.new("RGB", (32, 24), "red")
    if suffix == "gif":
        red.save(
            path, save_all=True, append_images=[Image.new("RGB", red.size, "blue")]
        )
    else:
        red.save(path)
    frames = list(media_frames(path, root))
    assert len(frames) == 1 and frames[0].timestamp is None
    assert frames[0].image.mode == "RGB"
    pixel = frames[0].image.getpixel((0, 0))
    assert pixel[0] >= 250 and pixel[2] <= 5


def test_video_sampling_timestamp_dedupe_and_cap(tmp_path):
    from rapid_mlx.local_search.extraction import media_frames, sample_timestamps

    root = folder(tmp_path)
    path = root / "clip.mp4"
    write_video(path)
    frames = list(media_frames(path, root))
    # Consecutive black frames at 0 and 5 seconds collapse; white at 10 survives.
    assert [f.timestamp for f in frames] == [0.0, 10.0]
    assert frames[0].image.getpixel((0, 0))[0] < 5
    assert frames[1].image.getpixel((0, 0))[0] > 245
    assert sample_timestamps(1200, 2) == [float(i) for i in range(0, 600, 5)]
    long = sample_timestamps(7200, 2)
    assert len(long) == 120 and long[0] == 0 and long[-1] == 3599.5
    assert np.diff(long) == pytest.approx([3599.5 / 119] * 119)
    for count, fps in [(0, 1), (100, 0), (100, float("nan"))]:
        with pytest.raises(SkippedError):
            sample_timestamps(count, fps)


def test_video_six_second_scenes_survive_five_second_sampling(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "scenes.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    assert writer.isOpened()
    try:
        for color in (0, 80, 160, 240):
            for _ in range(60):
                writer.write(np.full((32, 32, 3), color, np.uint8))
    finally:
        writer.release()
    frames = list(media_frames(path, root))
    assert [f.timestamp for f in frames] == [0.0, 10.0, 15.0, 20.0]
    for index, frame in enumerate(frames):
        assert index * 6 <= frame.timestamp < (index + 1) * 6
        assert abs(frame.image.getpixel((0, 0))[0] - index * 80) < 6


def test_video_long_decode_is_bounded(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")

    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "long.mkv"
    path.write_bytes(b"fake container")
    seeks = []

    class Capture:
        def isOpened(self):  # noqa: N802 - OpenCV interface
            return True

        def get(self, key):
            if key in {cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT}:
                return 8
            return 2 if key == cv2.CAP_PROP_FPS else 7200

        def set(self, key, value):
            seeks.append(value)
            return True

        def read(self):
            return True, np.full((8, 8, 3), len(seeks) % 2 * 255, np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = list(media_frames(path, root))
    assert len(frames) == len(seeks) == 120
    assert frames[-1].timestamp == 3599.5 and capture.released


@pytest.mark.parametrize("suffix", ["png", "mp4", "heic"])
def test_bad_media_skipped_and_size_caps(tmp_path, suffix):
    from rapid_mlx.local_search.extraction import (
        IMAGE_MAX_BYTES,
        VIDEO_MAX_BYTES,
        media_frames,
    )

    root = folder(tmp_path)
    path = root / ("bad." + suffix)
    path.write_bytes(b"not media")
    with pytest.raises(SkippedError):
        list(media_frames(path, root))
    with path.open("wb") as stream:
        stream.truncate((VIDEO_MAX_BYTES if suffix == "mp4" else IMAGE_MAX_BYTES) + 1)
    with pytest.raises(SkippedError, match="size cap"):
        list(media_frames(path, root))


def test_media_index_routes_kinds_and_keyword_fallback(tmp_path, monkeypatch):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        (root / "z-document.txt").write_text("apple")
        Image.new("RGB", (24, 24), "red").save(root / "a-apple.png")
        write_video(root / "b-apple.mp4")
        (root / "bad.png").write_bytes(b"corrupt")
        async with client_app(tmp_path, monkeypatch) as (client, app, fake):
            # Media cannot start until the folder's documents are searchable.
            original = fake.embed_images

            def check_order(images):
                assert any(
                    row["kind"] == "text"
                    for row in app.state.local_search.store.rows().values()
                )
                return original(images)

            fake.embed_images = check_order
            await client.post("/v1/local-search/folders", json={"path": str(root)})
            await app.state.local_search.wait_idle()
            status = (await client.get("/v1/local-search/status")).json()
            assert status["totals"]["files_by_kind"] == {
                "text": 1,
                "image": 1,
                "video": 1,
            }
            assert status["model"]["vision"] is True
            assert status["folders"][0]["files_skipped"] == 1
            assert fake.image_calls == [1, 2]
            assert len(set(fake.threads)) == 1
            assert threading.get_ident() not in fake.threads
            for kinds in (None, ["text"], ["image"], ["video"], [], ["image", "video"]):
                response = await client.post(
                    "/v1/local-search/search", json={"query": "apple", "kinds": kinds}
                )
                assert response.status_code == 200
                hits = response.json()["results"]
                assert {hit["kind"] for hit in hits} == set(
                    kinds if kinds is not None else ["text", "image", "video"]
                )
                for hit in hits:
                    if hit["kind"] != "text":
                        assert hit["snippet"] == "" and hit["highlights"] == []
                        assert hit["page"] is hit["line"] is None
                        # High-cosine vectors pass the available background gate.
                        assert hit["matched"] == ["semantic", "keyword"]
                    assert (hit["timestamp"] is not None) == (hit["kind"] == "video")
                grouped = (
                    await client.post(
                        "/v1/local-search/search",
                        json={
                            "query": "apple",
                            "kinds": kinds,
                            "group_limit": 1,
                        },
                    )
                ).json()
                assert grouped["results"] == hits
                if kinds is None or len(set(kinds)) > 1:
                    assert {g["kind"] for g in grouped["groups"]} == {
                        hit["kind"] for hit in hits
                    }
                    assert all(
                        g["total"] == len(g["results"]) == 1 for g in grouped["groups"]
                    )
                    for group in grouped["groups"]:
                        if group["kind"] == "video":
                            assert group["results"][0]["timestamp"] is not None
                else:
                    assert "groups" not in grouped
            for invalid in (0, 51, -1, "invalid"):
                assert (
                    await client.post(
                        "/v1/local-search/search",
                        json={
                            "query": "apple",
                            "group_limit": invalid,
                        },
                    )
                ).status_code == 400
            response = await client.post(
                "/v1/local-search/search", json={"query": "apple", "kinds": ["audio"]}
            )
            assert response.status_code == 400
            # Unchanged media reuses its vectors; failed query model still allows filename hits.
            before = list(fake.image_calls)
            await app.state.local_search.queue(status["folders"][0]["id"])
            await app.state.local_search.wait_idle()
            assert fake.image_calls == before
            fake.embed = lambda _: (_ for _ in ()).throw(RuntimeError("offline"))
            result = await app.state.local_search.search(
                "apple", 20, None, ["image", "video"]
            )
            assert result["mode"] == "keyword"
            assert all(r["matched"] == ["keyword"] for r in result["results"])
            grouped = await app.state.local_search.search(
                "apple", 20, None, ["image", "video"], 1
            )
            assert grouped["results"] == result["results"]
            assert {g["kind"] for g in grouped["groups"]} == {"image", "video"}
            assert all(g["total"] == 1 for g in grouped["groups"])
            assert (
                await app.state.local_search.search(
                    "not present", 20, None, group_limit=5
                )
            )["groups"] == []
            rows = app.state.local_search.store.sql(
                "SELECT text,filename FROM chunks_fts WHERE text=''"
            )
            assert len(rows) == 3 and all(
                r["filename"].startswith(("a-apple", "b-apple")) for r in rows
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [1, 19, 20, 29, 30, 120])
def test_semantic_gate_uses_index_background_and_rank_cap(count):
    from rapid_mlx.local_search.search import semantic_ranking

    rows = {i: {"path": str(i)} for i in range(count)}
    scores = [(i, 0.7) for i in range(count)]
    assert semantic_ranking(scores, rows, 0.6) == list(range(min(count, 3)))
    assert semantic_ranking(scores, rows, 0.8) == list(range(min(count, 3)))
    assert semantic_ranking(scores, rows, 0.7) == list(range(min(count, 3)))
    assert semantic_ranking([(0, 0.0)], rows, 0.0) == []
    assert semantic_ranking([], {}, 0.0) == []


def test_query_outlier_rescue_and_repeated_chunks_do_not_change_null():
    from rapid_mlx.local_search.search import semantic_ranking

    scores = [(0, 0.6)] + [(i, 0.2) for i in range(1, 101)]
    rows = {i: {"path": str(i)} for i in range(101)}
    assert semantic_ranking(scores, rows, 0.9) == [0]
    # Additional frames/chunks of the same file cannot create extra observations.
    repeated = [(101 + i, 0.6) for i in range(1000)]
    rows.update({i: {"path": "0"} for i, _ in repeated})
    assert semantic_ranking([scores[0]] + repeated + scores[1:], rows, 0.9) == [0]
    # A smooth, modest high score is not a significant rare outlier.
    assert semantic_ranking([(i, 0.2 + i / 1000) for i in range(101)], rows, 0.9) == []
    # Summation roundoff must not manufacture variance in a uniform pool.
    assert semantic_ranking([(i, 0.2) for i in range(101)], rows, 0.9) == [0, 1, 2]


def test_background_matches_cross_file_pairs_and_invalidates(tmp_path):
    search = engine(tmp_path)
    root = folder(tmp_path)
    fid = search.store.add_folder(root)[0]
    for name, kind, vectors in [
        ("a", "text", [[1, 0, 0], [0, 1, 0]]),
        ("b", "text", [[1, 0, 0]]),
        ("c", "image", [[0, 0, 1]]),
    ]:
        path = root / name
        path.write_text("fixture")
        search.store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk("", None, None, kind)] * len(vectors),
            vectors,
            "indexed",
        )
    rows = search.store.search_rows()
    # No query scoring is needed to initialise the background. The two
    # cross-file text pairs are 1 and 0; all within-file pairs are excluded.
    bars = search.store.semantic_background(rows)
    assert bars == {"text": 1 / 6, "image": 0.0, "video": 0.0}
    assert search.store.semantic_background(rows) is bars
    # Equal-file pooled pairs are 0.5, 0, 0: mean 1/6, capping the
    # narrower text-only estimate. Kind filtering retains the same pool.
    assert (
        search.store.semantic_background(search.store.search_rows(kinds=["text"]))[
            "text"
        ]
        == 1 / 6
    )
    # Repeating the same frames/chunks leaves the file centroid unchanged.
    path = root / "a"
    search.store.replace(
        fid,
        path,
        path.stat(),
        None,
        [Chunk("", None, None)] * 20,
        [[1, 0, 0], [0, 1, 0]] * 10,
        "indexed",
    )
    assert search.store.semantic_background(search.store.search_rows())["text"] == 1 / 6
    ids = [
        r["id"]
        for r in search.store.sql(
            "SELECT id FROM files WHERE path=?", (str(root / "b"),)
        )
    ]
    search.store.remove_file(ids[0])
    assert search.store.semantic_background(search.store.search_rows())["text"] == 0.0
    search.store.close()


def test_semantic_window_reserves_heads_without_reordering_stronger_hits():
    rows = {
        i: {"path": str(i), "kind": "text" if i < 15 else "image"} for i in range(17)
    }
    scores = {i: 0.95 - i / 100 for i in range(15)}
    scores.update({15: 0.7, 16: 0.69})
    kwargs = dict(
        rows=rows, semantic_legs=[list(range(15)), [15, 16], []], semantic_scores=scores
    )
    original = fuse([], [], **kwargs)
    assert [i for i, _, _ in original[:10]] == list(range(10))
    reserved = fuse([], [], limit=10, **kwargs)
    assert [i for i, _, _ in reserved[:10]] == list(range(8)) + [15, 16]
    assert [score for _, score, _ in reserved[:10]] == sorted(
        [score for _, score, _ in reserved[:10]], reverse=True
    )
    literal = fuse([], [14], keyword_evidence={14: 2}, limit=10, **kwargs)
    assert literal[0][0] == 14 and {15, 16} <= {i for i, _, _ in literal[:10]}


def test_semantic_gate_float32_boundary_and_one_file():
    from rapid_mlx.local_search.search import semantic_ranking

    rows = {0: {"path": "one"}}
    assert semantic_ranking([(0, float(np.float32(0.9)))], rows, 0.9) == [0]
    assert semantic_ranking([(0, 0.899)], rows, 0.9) == [0]
    assert semantic_ranking([(0, 0.1)], rows, 0.0) == [0]


@pytest.mark.parametrize("strong_kind", ["image", "text"])
def test_standardized_semantic_order_with_fake_embedder(
    tmp_path, monkeypatch, strong_kind
):
    class QueryEmbedder(FakeEmbedder):
        def embed(self, texts):
            super().embed(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

    async def scenario():
        fake = QueryEmbedder()
        search = engine(tmp_path, fake)
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        cosines = {}
        for kind in ("text", "image"):
            # Higher raw text cosine is ordinary when image is the outlier;
            # reverse the evidence without granting media an unconditional boost.
            values = (
                [0.65] + [0.2 + n / 1000 for n in range(20)]
                if kind == "image"
                else [0.8, 0.79] + [0.65 + n / 128 for n in range(20)]
            )
            if strong_kind == "text":
                values = (
                    [0.65] + [0.55 + n / 250 for n in range(20)]
                    if kind == "image"
                    else [0.85] + [0.2 + n / 1000 for n in range(20)]
                )
            for n, value in enumerate(values):
                path = root / f"{kind}_{n}.txt"
                path.write_text("fixture")
                cosines[str(path)] = value
                search.store.replace(
                    fid,
                    path,
                    path.stat(),
                    None,
                    [Chunk("fixture", None, None, kind)],
                    [[value, np.sqrt(1 - value**2), 0]],
                    "indexed",
                )
        # Isolate ordering from admission; all candidates have positive cosine.
        monkeypatch.setattr(
            search.store,
            "semantic_background",
            lambda _: dict.fromkeys(("text", "image", "video"), 0.0),
        )
        hits = (await search.search("meaning", 10, None))["results"]
        assert hits[0]["path"] == str(root / f"{strong_kind}_0.txt")
        assert hits[0]["matched"] == ["semantic"]
        for kind in ("text", "image"):
            values = [cosines[hit["path"]] for hit in hits if hit["kind"] == kind]
            assert values == sorted(values, reverse=True)
        grouped = await search.search("meaning", 10, None, group_limit=5)
        assert grouped["results"] == hits
        assert grouped["groups"][0]["kind"] == strong_kind
        assert {g["kind"]: g["total"] for g in grouped["groups"]} == {
            "text": 21 if strong_kind == "text" else 22,
            "image": 21,
        }
        for group in grouped["groups"]:
            assert len(group["results"]) == 5
            values = [cosines[hit["path"]] for hit in group["results"]]
            assert values == sorted(values, reverse=True)
        # Full filename and exact phrase evidence keep the existing positions.
        before = (await search.search("text_0", 10, None))["results"]
        assert before[0]["path"] == str(root / "text_0.txt")
        lexical = await search.search("text_0", 1, None, group_limit=1)
        assert lexical["results"] == before[:1]
        assert lexical["groups"][0]["kind"] == "text"
        assert lexical["groups"][0]["results"][0]["path"] == str(root / "text_0.txt")
        await search.close()
        assert fake.calls

    asyncio.run(scenario())


def test_standardized_near_ties_interleave_without_moving_lexical_hits():
    rows = {i: {"path": str(i), "kind": "text" if i < 6 else "image"} for i in range(9)}
    kwargs = dict(
        rows=rows,
        semantic_legs=[[0, 1, 2, 3, 4, 5], [6, 7, 8]],
        semantic_scores={i: 0.9 - i / 100 for i in rows},
        keyword_evidence={0: 3, 1: 2, 2: 1},
    )
    baseline = fuse([], [0, 1, 2], **kwargs)
    ranked = fuse(
        [], [0, 1, 2], semantic_standardized=dict.fromkeys(rows, 4.0), **kwargs
    )
    assert ranked[:3] == baseline[:3]
    assert [i for i, _, _ in ranked[3:]] == [3, 6, 4, 7, 5, 8]


def test_grouped_rankings_lexical_tiers_and_best_file_head_statistics():
    from rapid_mlx.local_search.search import grouped_rankings

    rows = {i: {"path": str(i), "kind": "text" if i < 4 else "image"} for i in range(6)}
    # The document's lexical chunk differs from its best semantic chunk.
    rows[6] = {"path": "0", "kind": "text"}
    groups = grouped_rankings(
        [0, 1, 2],
        rows=rows,
        semantic_legs=[[6, 3, 2, 1], [4, 5]],
        semantic_scores={6: 0.9, 3: 0.8, 2: 0.7, 1: 0.6, 4: 0.95, 5: 0.9},
        semantic_standardized={6: 1, 3: 0.5, 2: 0.2, 1: 0, 4: 8, 5: 6},
        keyword_evidence={0: 3, 1: 2, 2: 1},
    )
    assert [kind for kind, _ in groups] == ["text", "image"]
    assert [[i for i, _, _ in hits] for _, hits in groups] == [[0, 1, 2, 3], [4, 5]]
    # With no lexical evidence the stronger standardized head wins; ties have
    # deterministic kind order, without near-tie interleaving between sections.
    for strength, order in [(8, ["image", "text"]), (1, ["text", "image"])]:
        groups = grouped_rankings(
            [],
            rows=rows,
            semantic_legs=[[6], [4]],
            semantic_scores={6: 0.9, 4: 0.7},
            semantic_standardized={6: 1, 4: strength},
            keyword_evidence={},
        )
        assert [kind for kind, _ in groups] == order


def test_group_totals_exclude_missing_files_and_respect_scope(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        for kind in ("text", "image", "video"):
            for n in range(3):
                path = root / f"{kind}-{n}.txt"
                path.write_text("fixture")
                search.store.replace(
                    fid,
                    path,
                    path.stat(),
                    None,
                    [Chunk("fixture" if kind == "text" else "", None, None, kind)],
                    [[1, 0, 0]],
                    "indexed",
                )
        (root / "text-1.txt").unlink()
        search._embed = lambda _: [[1, 0, 0]]
        response = await search.search("meaning", 1, None, group_limit=1)
        assert {g["kind"]: g["total"] for g in response["groups"]} == {
            "text": 2,
            "image": 3,
            "video": 3,
        }
        assert len(response["results"]) == 1
        assert all(len(g["results"]) == 1 for g in response["groups"])
        assert (await search.search("meaning", 1, [], group_limit=50))["groups"] == []
        assert "groups" not in await search.search(
            "meaning", 1, None, ["text", "text"], 50
        )
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [1, 30])
def test_group_totals_are_bounded_by_keyword_candidate_window(tmp_path, limit):
    async def scenario():
        search = LocalSearch(tmp_path / "index", resolve_model=lambda _: None)
        try:
            root = folder(tmp_path)
            fid = search.store.add_folder(root)[0]
            for n in range(160):
                path = root / f"note-{n}.txt"
                path.write_text("apple orchard")
                search.store.replace(
                    fid,
                    path,
                    path.stat(),
                    None,
                    [Chunk("apple orchard", None, None, "text")],
                    None,
                    "keyword",
                )
            response = await search.search("apple", limit, None, group_limit=1)
            assert response["mode"] == "keyword"
            assert len(response["results"]) == limit
            assert len(response["groups"]) == 1
            group = response["groups"][0]
            assert group["kind"] == "text"
            assert len(group["results"]) == 1
            assert group["total"] == max(100, limit * 5) < 160
        finally:
            await search.close()

    asyncio.run(scenario())


def test_adaptive_gate_and_keyword_bypass_with_fake_embedder(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        (root / "fruit.md").write_text("banana")
        (root / "other.md").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        # Identical indexed vectors set their background to one. The query's
        # shared bias dimension alone cannot admit unrelated uniform matches.
        assert len((await search.search("unrelated", 20, None))["results"]) == 2
        hits = (await search.search("banana", 20, None))["results"]
        assert len(hits) == 2 and all("keyword" in r["matched"] for r in hits)
        # Keyword evidence bypasses rejection even with an orthogonal query.
        search._embed = lambda _: [[0, 0, 1]]
        hits = (await search.search("banana", 20, None))["results"]
        assert len(hits) == 2 and all("keyword" in r["matched"] for r in hits)
        assert fake.calls
        await search.close()

    asyncio.run(scenario())


def test_query_outlier_rescue_with_fake_embedder(tmp_path):
    class OutlierEmbedder(FakeEmbedder):
        def embed(self, texts):
            super().embed(texts)
            vectors = []
            for text in texts:
                if text.startswith("task: search result"):
                    vectors.append([0, 0, 1] if "absent" in text else [1, 0, 0])
                else:
                    cosine = 0.6 if "needle" in text else 0.2
                    vectors.append([cosine, np.sqrt(1 - cosine**2), 0])
            return vectors

    async def scenario():
        root = folder(tmp_path)
        for n in range(101):
            (root / f"file_{n}.txt").write_text("needle" if n == 0 else "bulk")
        search = engine(tmp_path, OutlierEmbedder())
        (await search.add_folder(str(root)))
        await search.wait_idle()
        hits = (await search.search("meaning", 10, None))["results"]
        assert len(hits) == 1 and hits[0]["path"] == str(root / "file_0.txt")
        assert hits[0]["matched"] == ["semantic"]
        assert (
            search.store.semantic_background(search.store.search_rows())["text"] > 0.6
        )
        assert not (await search.search("absent", 10, None))["results"]
        await search.close()

    asyncio.run(scenario())


def test_singleton_kinds_interleave_and_keep_best_video_frame(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        search = engine(tmp_path)
        fid = search.store.add_folder(root)[0]
        for kind, value in [("text", 0.99), ("image", 0.76), ("video", 0.50)]:
            path = root / (
                "hit." + {"text": "txt", "image": "png", "video": "mp4"}[kind]
            )
            path.write_bytes(b"fixture")
            chunks = [
                Chunk(
                    "document" if kind == "text" else "",
                    1 if kind == "text" else None,
                    None,
                    kind,
                    0.0 if kind == "video" else None,
                )
            ]
            vectors = [[value, np.sqrt(1 - value * value), 0]]
            if kind == "video":
                chunks.append(Chunk("", None, None, kind, 10.0))
                vectors.append([0.80, np.sqrt(1 - 0.80 * 0.80), 0])
            search.store.replace(
                fid, path, path.stat(), None, chunks, vectors, "indexed"
            )
        search._embed = lambda _: np.array([[1.0, 0.0, 0.0]])
        result = await search.search("unrelated", 20, None)
        assert len(result["results"]) == 3
        assert [r["kind"] for r in result["results"]] == ["text", "image", "video"]
        scores = [r["score"] for r in result["results"]]
        assert scores[0] > scores[1] > scores[2]
        video = next(r for r in result["results"] if r["kind"] == "video")
        assert video["timestamp"] == 10.0
        result = await search.search("hit", 20, None, ["video"])
        assert result["results"][0]["timestamp"] == 10.0
        assert result["results"][0]["matched"] == ["semantic", "keyword"]
        await search.close()

    asyncio.run(scenario())


def test_adaptive_background_independent_of_other_kinds_and_scoped_folders(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        search = engine(tmp_path)
        fid = search.store.add_folder(root)[0]
        # Each kind has its own background; dense stills cannot suppress a video.
        for n, (kind, values) in enumerate(
            [("image", [0.63])]
            + [("image", [0.7])] * 28
            + [("video", [0.76, 0.69]), ("text", [0.0])]
        ):
            path = root / f"entry_{n}.{kind}"
            path.write_bytes(b"fixture")
            chunks = [
                Chunk("", None, None, kind, float(j * 5) if kind == "video" else None)
                for j in range(len(values))
            ]
            vectors = [[v, np.sqrt(1 - v * v), 0] for v in values]
            search.store.replace(
                fid, path, path.stat(), None, chunks, vectors, "indexed"
            )
        search._embed = lambda _: np.array([[1.0, 0.0, 0.0]])
        for kinds in (None, ["video"], ["image", "video"]):
            results = (await search.search("unrelated", 100, None, kinds))["results"]
            media = [r for r in results if r["kind"] != "text"]
            assert len(media) == 1
            video = next(r for r in media if r["kind"] == "video")
            assert video["timestamp"] == 0.0
        # A file rejected by its index background still qualifies by filename.
        results = (await search.search("entry_0", 100, None, ["image"]))["results"]
        hit = next(r for r in results if r["path"] == str(root / "entry_0.image"))
        assert hit["matched"] == ["keyword"]
        # Folder selection still bounds the pool: an unrelated folder's high
        # cosines must not raise the bar for these selected media.
        other = folder(tmp_path, "other")
        other_fid = search.store.add_folder(other)[0]
        path = other / "other.image"
        path.write_bytes(b"fixture")
        search.store.replace(
            other_fid,
            path,
            path.stat(),
            None,
            [Chunk("", None, None, "image")] * 40,
            [[1, 0, 0]] * 40,
            "indexed",
        )
        results = (await search.search("unrelated", 100, [fid], ["video"]))["results"]
        assert len(results) == 1 and results[0]["timestamp"] == 0.0
        await search.close()

    asyncio.run(scenario())


def test_media_schema_and_chunking_rebuild_previously_skipped_files(tmp_path):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        path = root / "photo.png"
        Image.new("RGB", (24, 24), "red").save(path)
        search = engine(tmp_path)
        fid = search.store.add_folder(root)[0]
        search.store.replace(
            fid, path, path.stat(), None, [], None, "skipped: unsupported"
        )
        search.store.sql("UPDATE meta SET value='1' WHERE key='schema_version'")
        search.store.sql("UPDATE meta SET value='2' WHERE key='chunking_version'")

        await search.close()
        search = engine(tmp_path)
        assert search.store.files(fid)[str(path)]["status"] == "pending"
        search.start()
        await search.wait_idle()
        assert len(search.store.vectors) == 1
        assert search.store.rows()[next(iter(search.store.vectors))]["kind"] == "image"
        assert search.status()["totals"]["files_by_kind"]["image"] == 1
        await search.close()

    asyncio.run(scenario())


def test_media_cancellation_preserves_per_file_commit_and_resumes(tmp_path):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        Image.new("RGB", (24, 24), "red").save(root / "a.png")
        write_video(root / "b.mp4", colors=(0, 255, 0, 255, 0, 255), fps=2, seconds=30)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        entered, release = pause_after_embedding(search, batch=2, images=True)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await asyncio.wait_for(entered.wait(), SYNC_TIMEOUT)
        await search._cancel_worker()
        release.set()
        assert len(search.store.files(fid)) == 1
        assert all(row["kind"] == "image" for row in search.store.rows().values())
        await search.queue(fid)
        await search.wait_idle()
        assert search.store.folder(fid)["files_indexed"] == 2
        assert max(fake.image_calls) <= 4
        await search.close()

    asyncio.run(scenario())


def cpu_embedding_module(monkeypatch):
    """Execute engine policy code with a stub array module; never import Metal."""
    import importlib.util
    from types import ModuleType

    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    spec = importlib.util.spec_from_file_location(
        "rapid_mlx.embedding", Path(__file__).parents[1] / "rapid_mlx" / "embedding.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "rapid_mlx.embedding", module)
    spec.loader.exec_module(module)
    return module


def test_vision_flag_default_and_text_media_rejection_on_http(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx.config import get_config
    from rapid_mlx.routes import embeddings

    module = cpu_embedding_module(monkeypatch)
    engine = module.EmbeddingEngine("mlx-community/embeddinggemma-2-4bit")
    assert engine.vision is False
    with pytest.raises(module.EmbeddingUnsupportedMediaError, match="vision=True"):
        engine.embed_images([])
    ids = dict(
        image_token_id=258880,
        audio_token_id=258881,
        video_token_id=258884,
        boi_token_id=255999,
        eoi_token_id=258882,
        boa_token_id=256000,
        eoa_token_index=258883,
    )
    engine._model = SimpleNamespace(
        model_type="embedding_gemma2", config=SimpleNamespace(**ids)
    )
    engine.effective_max_length = 8192
    engine.count_tokens = lambda _: 1
    engine._tokenizer = lambda texts, **kwargs: {"input_ids": [[2, 258880]]}
    # Even a vision-enabled local engine cannot bypass the text API guard.
    cfg = get_config()
    monkeypatch.setattr(cfg, "embedding_model_locked", engine.model_name)
    monkeypatch.setattr(cfg, "embedding_engine", engine)
    monkeypatch.setattr(cfg, "api_key", None)
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.server",
        SimpleNamespace(load_embedding_model=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.service.helpers",
        SimpleNamespace(_resolve_request_alias_or_default=lambda name, locked: locked),
    )
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(embeddings.router)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            for vision in (False, True):
                engine.vision = vision
                for payload in ("<|image|>", [258880], [[258881]]):
                    result = await client.post(
                        "/v1/embeddings", json={"model": "default", "input": payload}
                    )
                    assert result.status_code == 400
                    assert (
                        result.json()["error"]["code"]
                        == "unsupported_embedding_modality"
                    )
                result = await client.post(
                    "/v1/embeddings",
                    json={"model": "default", "input": {"image": "file.png"}},
                )
                assert result.status_code == 400

    asyncio.run(scenario())


def test_vision_image_engine_uses_pilot_sequence_and_mixed_sizes(monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    module = cpu_embedding_module(monkeypatch)
    module.mx.array = np.asarray
    module.mx.float32 = np.float32
    cleared = []
    module.mx.clear_cache = lambda: cleared.append(True)
    engine = module.EmbeddingEngine("mlx-community/embeddinggemma-2-4bit", vision=True)
    config = SimpleNamespace(boi_token_id=10, image_token_id=11, eoi_token_id=12)
    captured = []

    class Model:
        vision_enabled = True

        def __call__(self, ids, **kwargs):
            captured.append((ids, kwargs))
            return SimpleNamespace(text_embeds=np.array([[1.0, 0.0], [0.0, 1.0]]))

    engine._model = Model()
    engine._model.config = config
    engine.effective_max_length = 8192

    class Tokenizer:
        def decode(self, ids):
            return str(ids[0]) + " "

        def __call__(self, pieces, **kwargs):
            assert pieces == ["10 11 11 12 ", "10 11 11 11 12 "]
            return {
                "input_ids": np.array(
                    [[2, 10, 11, 11, 12, 3, 0], [2, 10, 11, 11, 11, 12, 3]]
                ),
                "attention_mask": np.array(
                    [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]
                ),
            }

    engine._tokenizer = Tokenizer()

    def prepare(images):
        assert [im.size for im in images] == [(8, 8), (16, 8)]
        return {"pixel_values": [np.zeros((3, 8, 8)), np.zeros((3, 8, 16))]}, [2, 3]

    engine._image_processor = prepare
    result = engine.embed_images([Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))])
    assert result == [[1.0, 0.0], [0.0, 1.0]] and len(cleared) == 1
    assert [p.shape for p in captured[0][1]["pixel_values"]] == [(3, 8, 8), (3, 8, 16)]
    assert captured[0][1]["attention_mask"][0, -1] == 0
    assert engine.embed_images([]) == []
    engine.effective_max_length = 2
    with pytest.raises(module.EmbeddingInputTooLongError):
        engine.embed_images([Image.new("RGB", (8, 8)), Image.new("RGB", (16, 8))])
    assert len(captured) == 1


def test_status_vision_available_checkpoint_without_loading(tmp_path):
    import json

    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text(
        json.dumps(
            {"model_type": "embedding_gemma2", "vision_config": {"hidden_size": 8}}
        )
    )
    search = LocalSearch(tmp_path / "index", resolve_model=lambda _: str(model))
    assert search.status()["model"]["vision"] is True
    assert search.engine is None
    config.write_text(json.dumps({"model_type": "embedding_gemma2"}))
    assert search.status()["model"]["vision"] is False
    asyncio.run(search.close())


def test_engine_native_loader_vision_opt_in_only(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx import model_downloads

    module = cpu_embedding_module(monkeypatch)
    monkeypatch.setattr(model_downloads, "require_local", lambda _: None)
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(model_type="embedding_gemma2"), object()

    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.models.embedding_gemma2.loader",
        SimpleNamespace(load=load),
    )
    name = "mlx-community/embeddinggemma-2-4bit"
    module.EmbeddingEngine(name).load()
    module.EmbeddingEngine(name, vision=True).load()
    assert calls == [(name, {}), (name, {"vision": True})]


def test_media_without_model_filename_index_resumes(tmp_path):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        Image.new("RGB", (24, 24), "red").save(root / "sunset.png")
        write_video(root / "sunset.mp4")
        search = engine(tmp_path)
        search.resolve_model = lambda _: None
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "waiting_for_model"
        assert search.status()["totals"]["files_by_kind"] == {
            "text": 0,
            "image": 1,
            "video": 1,
        }
        result = await search.search("sunset", 20, None)
        assert len(result["results"]) == 2 and result["mode"] == "keyword"
        assert not search.store.vectors
        search.resolve_model = lambda _: "local-checkpoint"
        await search.status_async()
        await search.wait_idle()
        assert len(search.store.vectors) == 3
        assert search.store.folder(fid)["state"] == "idle"
        await search.close()

    asyncio.run(scenario())


def test_unexpected_decoder_failure_does_not_abort_folder(tmp_path, monkeypatch):
    from PIL import Image

    async def scenario():
        root = folder(tmp_path)
        (root / "good.txt").write_text("apple")
        (root / "bad.tiff").write_bytes(b"corrupt")
        monkeypatch.setattr(
            Image,
            "open",
            lambda _: (_ for _ in ()).throw(SyntaxError("invalid TIFF header")),
        )
        search = engine(tmp_path)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["state"] == "idle"
        assert search.store.folder(fid)["files_skipped"] == 1
        assert search.store.folder(fid)["files_indexed"] == 1
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "identifier", ["DEC-0054", "src/lib/i18n.ts", "RAPID_MLX_HOME", "foo.bar"]
)
@pytest.mark.parametrize("in_filename", [False, True])
def test_keyword_compounds_are_literal(tmp_path, identifier, in_filename):
    from rapid_mlx.local_search.search import query_terms

    assert query_terms(f"Find ({identifier}). 中文") == ["Find", identifier, "中文"]
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    split = " unrelated ".join(__import__("re").split(r"[-/._]", identifier))
    # Paths containing slashes must be tested in content, since only basenames
    # participate in filename keyword lookup.
    name = identifier if in_filename and "/" not in identifier else "exact"
    entries = [
        (name + ".txt", "needle " + ("" if name == identifier else identifier)),
        ("split.txt", "needle " + split),
        (
            "separator.txt",
            "needle "
            + identifier.replace(".", " ")
            .replace("-", " ")
            .replace("_", " ")
            .replace("/", " "),
        ),
    ]
    for name, text in entries:
        path = root / name
        path.write_text(text)
        store.replace(
            fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword"
        )
    rows = store.rows()
    expected = [i for i, row in rows.items() if Path(row["path"]).name == entries[0][0]]
    assert (
        store.keyword(query_terms(identifier), set(rows), query=identifier) == expected
    )
    # A common identifier must not be discarded for another distinctive term.
    assert (
        store.keyword(
            query_terms("needle " + identifier), set(rows), distinctive=["needle"]
        )
        == expected
    )
    store.close()


def test_video_dedupe_accumulates_slow_drift(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "drift.mp4"
    path.write_bytes(b"fixture")

    class Capture:
        value = -1
        released = False

        def isOpened(self):  # noqa: N802 - decoder interface
            return True

        def get(self, key):
            return 1 if key == cv2.CAP_PROP_FPS else 26

        def set(self, key, value):
            self.value += 1
            return True

        def read(self):
            return True, np.full((32, 32, 3), self.value, np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    frames = list(media_frames(path, root))
    assert [f.timestamp for f in frames] == [0.0, 10.0, 20.0]
    assert capture.released


def test_whole_mac_validation_and_overlap_without_walking(tmp_path):
    assert validate_folder("/") == Path("/")
    for first, second in ((Path("/"), tmp_path), (tmp_path, Path("/"))):
        store = Store(
            tmp_path
            / ("root-first.sqlite" if first == Path("/") else "root-last.sqlite"),
            "fake",
        )
        try:
            fid, _ = store.add_folder(first)
            assert store.add_folder(first) == (fid, False)
            covering_id, created = store.add_folder(second)
            assert created == (second == Path("/"))
            assert (covering_id == fid) == (first == Path("/"))
            assert len(store.folders()) == 1
            assert store.folders()[0]["path"] == "/"
        finally:
            store.close()


@pytest.mark.parametrize(
    "chosen",
    [
        "whole",
        "home",
        "documents",
        "volume",
        "library",
        "icloud",
        "other-library",
        "system",
        "scratch",
    ],
)
def test_absolute_os_and_all_user_library_skips(tmp_path, monkeypatch, chosen):
    from rapid_mlx.local_search import extraction

    os_names = (
        "System",
        "Library",
        "usr",
        "bin",
        "sbin",
        "private",
        "dev",
        "cores",
        "opt",
        "Network",
        "Volumes",
    )
    prefixes = tuple(tmp_path / n for n in os_names)
    monkeypatch.setattr(extraction, "SKIP_PREFIXES", prefixes)
    monkeypatch.setattr(extraction, "USERS_DIRECTORY", tmp_path / "Users")
    home = tmp_path / "Users" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for prefix in prefixes:
        prefix.mkdir()
        (prefix / "excluded.txt").write_text("secret")
    for user in ("alice", "bob"):
        library = tmp_path / "Users" / user / "Library" / "nested"
        library.mkdir(parents=True)
        (library / "excluded.txt").write_text("secret")
    documents = home / "Documents"
    documents.mkdir()
    (documents / "good.txt").write_text("apple")
    volume = tmp_path / "Volumes" / "External"
    volume.mkdir()
    (volume / "good.txt").write_text("banana")
    icloud = home / "Library" / "Mobile Documents"
    icloud.mkdir()
    (icloud / "good.txt").write_text("apple")
    scratch = tmp_path / "private" / "tmp"
    scratch.mkdir()
    (scratch / "good.txt").write_text("banana")
    # The temp tree stands in for "/"; never walk the real filesystem root.
    root = {
        "whole": tmp_path,
        "home": home,
        "documents": documents,
        "volume": volume,
        "library": home / "Library",
        "icloud": icloud,
        "other-library": tmp_path / "Users" / "bob" / "Library",
        "system": tmp_path / "System",
        "scratch": scratch,
    }[chosen]
    expected = {
        "whole": [documents / "good.txt"],
        "home": [documents / "good.txt"],
        "documents": [documents / "good.txt"],
        "volume": [volume / "good.txt"],
        "library": [home / "Library" / "nested" / "excluded.txt", icloud / "good.txt"],
        "icloud": [icloud / "good.txt"],
        "other-library": [root / "nested" / "excluded.txt"],
        "system": [root / "excluded.txt"],
        "scratch": [scratch / "good.txt"],
    }[chosen]
    assert validate_folder(str(root)) == root
    files, skipped = scan(root)
    assert set(files) == set(expected)
    assert skipped == {"whole": len(prefixes) + 2, "home": 1}.get(chosen, 0)
    for selected in (
        *prefixes,
        home / "Library",
        tmp_path / "Users" / "bob" / "Library",
    ):
        assert validate_folder(str(selected)) == selected
        selected_files, selected_skipped = scan(selected)
        assert selected_files
        assert selected_skipped == 0
    assert validate_folder(str(volume)) == volume
    # Boundary matching: similarly named user directories are allowed.
    ordinary = tmp_path / "System Notes"
    ordinary.mkdir()
    assert validate_folder(str(ordinary)) == ordinary

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            (await search.add_folder(str(root)))
            await search.wait_idle()
            assert {row["path"] for row in search.store.rows().values()} == {
                str(path) for path in files
            }
            assert len(fake.calls) == len(expected)
            assert search.status()["folders"][0]["files_skipped"] == skipped
        finally:
            await search.close()

    asyncio.run(scenario())


def test_unreadable_directory_is_skipped_and_indexing_continues(tmp_path, monkeypatch):
    root = folder(tmp_path)
    denied = root / "denied"
    denied.mkdir()
    (root / "good.txt").write_text("apple")
    original = os.scandir

    def scandir(path):
        if isinstance(path, int):
            denied_path = os.fstat(path).st_ino == denied.stat().st_ino
        else:
            denied_path = Path(path) == denied
        if denied_path:
            raise PermissionError("TCC denied")
        return original(path)

    monkeypatch.setattr(os, "scandir", scandir)

    async def scenario():
        search = engine(tmp_path)
        try:
            (await search.add_folder(str(root)))
            await search.wait_idle()
            status = search.status()
            assert status["folders"][0]["state"] == "idle"
            assert status["folders"][0]["files_skipped"] == 1
            assert status["totals"]["files"] == 1
        finally:
            await search.close()

    asyncio.run(scenario())


def test_files_searchable_before_discovery_finishes(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service

    root = folder(tmp_path)
    first = root / "first.txt"
    first.write_text("apple")
    nested = root / "nested"
    nested.mkdir()
    (nested / "later.txt").write_text("banana")

    finished = False
    original_scan = service.scan_batches

    def scan_in_progress(root, root_fd=None):
        nonlocal finished
        yield from original_scan(root)
        finished = True

    monkeypatch.setattr(service, "scan_batches", scan_in_progress)

    async def scenario():
        search = engine(tmp_path)
        continued = asyncio.Event()
        release = asyncio.Event()

        async def pause():
            assert search.progress["active"]
            assert search.progress["files_total"] == 0
            assert not finished
            assert search.progress["files_done"] == 1
            results = (await search.search("apple", 20, None))["results"]
            assert any(r["path"] == str(first) for r in results)
            continued.set()
            await release.wait()

        # Pause indexing at the first file's cooperative yield, before requesting
        # the next discovery batch, so this test never depends on timing.
        paused = False
        async_sleep = service.asyncio.sleep

        async def sleep(seconds):
            nonlocal paused
            if not paused and search.progress["files_done"] == 1:
                paused = True
                await pause()
            await async_sleep(seconds)

        monkeypatch.setattr(service.asyncio, "sleep", sleep)
        try:
            (await search.add_folder(str(root)))
            await asyncio.wait_for(continued.wait(), SYNC_TIMEOUT)
            release.set()
            await search.wait_idle()
            assert search.status()["totals"]["files"] == 2
        finally:
            release.set()
            await search.close()

    asyncio.run(scenario())


def test_whole_mac_api_acceptance_does_not_walk_root(tmp_path, monkeypatch):
    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            monkeypatch.setattr(app.state.local_search, "queue", AsyncMock())
            response = await client.post("/v1/local-search/folders", json={"path": "/"})
            assert response.status_code == 201
            assert response.json()["path"] == "/"
            again = await client.post("/v1/local-search/folders", json={"path": "/"})
            assert again.status_code == 200
            assert again.json()["id"] == response.json()["id"]

    asyncio.run(scenario())


def test_full_phrase_beats_two_weak_votes_and_keeps_best_chunk():
    from rapid_mlx.local_search.search import lexical_tier

    rows = {
        1: {"path": "near.md", "kind": "text"},
        2: {"path": "target.md", "kind": "text"},
        3: {"path": "target.md", "kind": "text"},
        4: {"path": "frame.mp4", "kind": "video"},
    }
    query = "orbital resonance certificate"
    assert (
        lexical_tier(
            query, query.split(), "orbital\nresonance certificate", "target.md"
        )
        == 2
    )
    assert (
        lexical_tier(query, query.split(), "orbital certificate resonance", "near.md")
        == 0
    )
    # A keyword-only literal chunk defeats a semantic+partial keyword neighbour,
    # and repeated weak chunks from that same file cannot replace its snippet.
    ranked = fuse(
        [],
        [2, 1, 3],
        rows=rows,
        semantic_legs=[[1], [], [4]],
        semantic_scores={1: 0.9, 4: 0.8},
        keyword_evidence={2: 2, 1: 0, 3: 0},
    )
    assert ranked[0][0] == 2
    assert ranked[0][2] == ["keyword"]
    assert len(ranked) == 3
    assert ranked[0][1] > ranked[1][1]


def test_literal_lookup_preserves_keyword_order_among_full_phrases():
    rows = {1: {"path": "first.md"}, 2: {"path": "second.md"}}
    # Extra semantic support cannot move a weaker lexical match above the
    # strongest phrase hit; no corpus-specific multiplier is involved.
    ranked = fuse([2], [1, 2], rows=rows, keyword_evidence={1: 2, 2: 2})
    assert [i for i, _, _ in ranked] == [1, 2]
    assert ranked[0][1] > ranked[1][1]


def test_content_copies_are_adjacent_without_hiding_paths():
    rows = {
        1: {"path": "original.md", "kind": "text", "content_hash": "same"},
        2: {"path": "unrelated.md", "kind": "text", "content_hash": "other"},
        3: {"path": "copy.md", "kind": "text", "content_hash": "same"},
    }
    ranked = fuse([1, 2, 3], [], rows=rows)
    assert [i for i, _, _ in ranked] == [1, 3, 2]
    assert len({rows[i]["path"] for i, _, _ in ranked}) == 3
    assert ranked[0][1] == ranked[1][1] > ranked[2][1]
    # A missing hash must never group unrelated files.
    rows[1]["content_hash"] = rows[3]["content_hash"] = None
    assert [i for i, _, _ in fuse([1, 2, 3], [], rows=rows)] == [1, 2, 3]


def test_distinctive_frequency_is_invariant_to_keyword_only_tails(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for name, text in [
        ("a.txt", "needle common"),
        ("b.txt", "common"),
        ("data.txt", "data"),
    ]:
        path = root / name
        path.write_text(text)
        store.replace(
            fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword"
        )
    before = store.distinctive_terms(["needle", "common"])
    data = root / "data.txt"
    # A generated file may dwarf the rest of the corpus, without making words
    # occurring in several other files 'distinctive'.
    store.replace(
        fid,
        data,
        data.stat(),
        None,
        [Chunk("data", n, None) for n in range(200)],
        None,
        "keyword",
    )
    assert store.distinctive_terms(["needle", "common"]) == before == ["needle"]
    # Replacing a file must invalidate frequency results, including removals.
    target = root / "b.txt"
    store.replace(
        fid,
        target,
        target.stat(),
        None,
        [Chunk("needle common", 1, None)],
        None,
        "keyword",
    )
    assert store.distinctive_terms(["needle"]) == []
    store.close()


@pytest.mark.parametrize(
    "query,name,expected",
    [
        ("memo", "memo.pdf", (True, 1 / 2)),
        ("memo", "plain_memo.pdf", (False, 1 / 3)),
        ("memo", "a_very_long_memo_name_v2.pdf", (False, 1 / 8)),
        ("memo", "memory_notes.md", (False, 0)),
        ("memo notes", "memoNotes.md", (True, 2 / 3)),
        ("memo", "XMLMemo2Notes.PDF", (False, 1 / 5)),
        ("memo 2", "memo2.pdf", (True, 2 / 3)),
        ("strasse", "Straße.pdf", (True, 1 / 2)),
        ("école", "ÉcoleMémo.pdf", (False, 1 / 3)),
        ("中文", "年度中文记录.pdf", (False, 1 / 2)),
        ("東京", "東京の記録.pdf", (False, 1 / 2)),
        ("보고", "연간보고서.pdf", (False, 1 / 2)),
        ("memo pdf", "plain_memo.pdf", (False, 2 / 3)),
        ("scanned_deposition", "scanned_deposition.pdf", (True, 2 / 3)),
        ("notes memo", "memo_notes.md", (False, 2 / 3)),
        ("memo missing", "plain_memo.pdf", (False, 0)),
        ("", "memo.pdf", (False, 0)),
    ],
)
def test_filename_token_evidence(query, name, expected):
    from rapid_mlx.local_search.search import filename_match, lexical_tier, query_terms

    terms = query_terms(query)
    assert filename_match(query, terms, name) == expected
    assert lexical_tier(query, terms, "", name) == (
        3
        if expected[1]
        else 1
        if terms and all(term.casefold() in name.casefold() for term in terms)
        else 0
    )
    if query:
        assert lexical_tier(query, terms, query + " unrelated body", "other.txt") == 2


@pytest.mark.parametrize(
    "query,normal,named,distractor",
    [
        ("memo", "plain_memo.pdf", "locked_memo.pdf", "memory buffers"),
        (
            "deposition",
            "plain_deposition.pdf",
            "scanned_deposition.pdf",
            "deposition record",
        ),
        ("中文记录", "年度中文记录.pdf", "锁定中文记录.pdf", "中文记录内容"),
    ],
)
def test_filename_words_outrank_corpus_content_and_name_only(
    tmp_path, monkeypatch, query, normal, named, distractor
):
    from rapid_mlx.local_search import store as storage
    from rapid_mlx.local_search.search import query_terms

    # Force body candidate truncation: filename retrieval must rescue the hits.
    monkeypatch.setattr(storage, "KEYWORD_CANDIDATE_LIMIT", 12)

    async def scenario():
        root = folder(tmp_path)
        for i in range(40):
            (root / f"record_{i}.json").write_text(distractor * (i % 4 + 1) + f" {i}")
        (root / normal).write_bytes(small_pdf() + b"\n% normal fixture")
        (root / named).write_bytes(pdf_fixture(encrypted=query == "memo"))
        (root / f"{query}.pdf").write_bytes(small_pdf() + b"\n% exact fixture")
        long_name = f"a_very_long_{query}_name_v2.pdf"
        (root / long_name).write_bytes(small_pdf() + b"\n% long fixture")
        if query == "memo":
            (root / "memory_notes.md").write_text(distractor)
        search = LocalSearch(tmp_path / "index", resolve_model=lambda _: None)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            rows = search.store.search_rows(None, None)
            evidence = {}
            keyword = search.store.keyword(
                query_terms(query), rows, query=query, evidence=evidence
            )
            content_order = [rows[i]["path"] for i in keyword if evidence[i] < 3]
            assert len(content_order) > 4
            response = await search.search(query, 100, None, group_limit=50)
            hits = response["results"]
            expected = [
                str(root / name) for name in [f"{query}.pdf", normal, named, long_name]
            ]
            # Equal coverage retains the existing candidate order: indexed then name-only.
            assert [hit["path"] for hit in hits[:4]] == expected
            assert [hit["path"] for hit in hits[4:]] == content_order
            assert hits[1]["content_indexed"] is True
            assert hits[2]["content_indexed"] is False
            assert all(hit["matched"] == ["keyword"] for hit in hits)
            assert response["groups"] == [
                {"kind": "text", "results": hits, "total": len(hits)}
            ]
            if query == "memo":
                assert all(
                    evidence[i] < 3
                    for i in keyword
                    if Path(rows[i]["path"]).name == "memory_notes.md"
                )
            # Existing multi-term/full-stem lookups still use this same tier.
            for phrase in [normal.removesuffix(".pdf"), query + " pdf"]:
                assert (await search.search(phrase, 100, None))["results"][0][
                    "content_indexed"
                ] is True
        finally:
            await search.close()

    asyncio.run(scenario())


def test_filename_group_head_has_lexical_evidence():
    from rapid_mlx.local_search.search import grouped_rankings, lexical_tier

    rows = {
        0: {"path": "scanned_deposition.png", "kind": "image"},
        1: {"path": "record.txt", "kind": "text"},
    }
    groups = grouped_rankings(
        [0],
        rows=rows,
        semantic_legs=[[1]],
        semantic_scores={1: 0.99},
        semantic_standardized={1: 10},
        keyword_evidence={
            0: lexical_tier("deposition", ["deposition"], "", rows[0]["path"])
        },
    )
    assert [kind for kind, _ in groups] == ["image", "text"]
    assert groups[0][1][0][0] == 0


def test_common_filename_terms_still_match_and_exact_basename_wins(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        search = LocalSearch(tmp_path / "index", resolve_model=lambda _: None)
        fid = search.store.add_folder(root)[0]
        for name, text in [
            ("orbit-map.txt", "unrelated"),
            ("b.txt", "orbit and map"),
            ("c.txt", "orbit map"),
        ]:
            path = root / name
            path.write_text(text)
            search.store.replace(
                fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword"
            )
        assert search.store.distinctive_terms(["orbit", "map"]) == []
        hits = (await search.search("orbit map", 10, None))["results"]
        assert Path(hits[0]["path"]).name == "orbit-map.txt"
        assert hits[0]["matched"] == ["keyword"]
        assert all(0 <= hit["score"] <= 1 for hit in hits)
        await search.close()

    asyncio.run(scenario())


def test_metadata_search_cache_tracks_replacements_deletion_scope_and_touch(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    path = root / "a.txt"
    path.write_text("first")
    store.replace(
        fid, path, path.stat(), "old", [Chunk("first", 1, None)], None, "keyword"
    )
    rows = store.search_rows()
    first_id = next(iter(rows))
    assert "text" not in rows[first_id]
    assert store.chunk_text(first_id) == "first"
    assert store.search_rows([], None) == {}
    assert store.search_rows(None, ["image"]) == {}
    assert store.search_rows(None, ["text"])[first_id]["content_hash"] == "old"
    path.write_text("replacement")
    store.replace(
        fid, path, path.stat(), "new", [Chunk("replacement", 1, None)], None, "keyword"
    )
    rows = store.search_rows()
    replacement_id = next(iter(rows))
    assert first_id not in rows
    assert rows[replacement_id]["content_hash"] == "new"
    assert store.chunk_text(replacement_id) == "replacement"
    path.write_text("longer replacement")
    store.touch(rows[replacement_id]["file_id"], path.stat())
    assert store.search_rows()[replacement_id]["size"] == path.stat().st_size
    store.remove_file(rows[replacement_id]["file_id"])
    assert store.search_rows() == {}
    store.close()


def test_video_keyword_evidence_survives_best_frame_selection():
    rows = {
        1: {"path": "unique.mp4", "kind": "video"},
        2: {"path": "unique.mp4", "kind": "video"},
        3: {"path": "other.md", "kind": "text"},
    }
    ranked = fuse(
        [],
        [1],
        rows=rows,
        semantic_legs=[[3], [], [2]],
        semantic_scores={2: 0.7, 3: 0.9},
        keyword_evidence={1: 3},
    )
    assert ranked[0][0] == 2
    assert ranked[0][2] == ["semantic", "keyword"]


def test_semantic_collapse_keeps_first_chunk_and_counts_one_vote():
    rows = {
        1: {"path": "frames.mp4", "kind": "video"},
        2: {"path": "other.png", "kind": "image"},
        3: {"path": "frames.mp4", "kind": "video"},
    }
    ranked = fuse([1, 2, 3], [], rows=rows)
    assert [i for i, _, _ in ranked] == [1, 2]
    assert ranked[0][1] == fuse([1], [], rows=rows)[0][1]


def office_zip(parts):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in parts.items():
            archive.writestr(name, text)
    return data.getvalue()


def office_fixture(suffix):
    word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    drawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    sheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    def relationships(entries):
        return (
            f'<Relationships xmlns="{package}">'
            + "".join(
                f'<Relationship Id="{rid}" Type="{rel}/{kind}" Target="{target}"/>'
                for rid, kind, target in entries
            )
            + "</Relationships>"
        )

    def paragraph(ns, label):
        return f'<a:p xmlns:a="{ns}"><a:r><a:t>{label}</a:t></a:r></a:p>'

    if suffix == ".docx":
        parts = {
            "word/document.xml": f'<w:document xmlns:w="{word}"><w:body>'
            + paragraph(word, "apple body")
            + "<w:tbl><w:tr><w:tc>"
            + paragraph(word, "banana table")
            + "</w:tc></w:tr></w:tbl></w:body></w:document>"
        }
        for name in ("header1", "footer1", "footnotes", "endnotes", "comments"):
            parts[f"word/{name}.xml"] = paragraph(word, name + " legal clause")
        return office_zip(parts), [None]
    if suffix == ".pptx":
        presentation = "http://schemas.openxmlformats.org/presentationml/2006/main"
        # Relationship order, not filenames, determines the user-visible slide.
        parts = {
            "ppt/presentation.xml": f'<p:presentation xmlns:p="{presentation}" xmlns:r="{rel}"><p:sldIdLst><p:sldId r:id="s2"/><p:sldId r:id="s1"/></p:sldIdLst></p:presentation>',
            "ppt/_rels/presentation.xml.rels": relationships(
                [
                    ("s1", "slide", "slides/slide1.xml"),
                    ("s2", "slide", "slides/slide2.xml"),
                ]
            ),
            "ppt/slides/slide1.xml": paragraph(drawing, "banana second slide"),
            "ppt/slides/slide2.xml": paragraph(drawing, "apple first slide"),
            "ppt/slides/_rels/slide2.xml.rels": relationships(
                [("note", "notesSlide", "../notesSlides/notesSlide7.xml")]
            ),
            "ppt/notesSlides/notesSlide7.xml": paragraph(
                drawing, "speaker patent notes"
            ),
        }
        return office_zip(parts), [1, 2]
    parts = {
        "xl/workbook.xml": f'<workbook xmlns="{sheet}" xmlns:r="{rel}"><sheets><sheet name="First" r:id="s2"/><sheet name="Second" r:id="s1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": relationships(
            [
                ("s1", "worksheet", "worksheets/sheet1.xml"),
                ("s2", "worksheet", "worksheets/sheet2.xml"),
            ]
        ),
        "xl/sharedStrings.xml": f'<sst xmlns="{sheet}"><si><r><t>apple </t></r><r><t>shared string</t></r></si></sst>',
        "xl/worksheets/sheet2.xml": f'<worksheet xmlns="{sheet}"><sheetData><row><c t="s"><v>0</v></c><c t="inlineStr"><is><t>banana inline string</t></is></c><c><v>42</v></c></row></sheetData></worksheet>',
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{sheet}"><sheetData><row><c t="inlineStr"><is><t>second sheet</t></is></c></row></sheetData></worksheet>',
    }
    return office_zip(parts), [1, 2]


@pytest.mark.parametrize("suffix", [".docx", ".pptx", ".xlsx"])
def test_office_text_and_persisted_locators(tmp_path, suffix):
    data, pages = office_fixture(suffix)
    units = extract(Path("file" + suffix), data)
    assert [u.page for u in units] == pages
    if suffix == ".docx":
        assert all(
            label in units[0].text
            for label in (
                "apple body",
                "banana table",
                "header1",
                "footer1",
                "footnotes",
                "endnotes",
                "comments",
            )
        )
    elif suffix == ".pptx":
        assert (
            "apple first slide" in units[0].text
            and "speaker patent notes" in units[0].text
        )
        assert "banana second slide" in units[1].text
    else:
        assert units[0].text == "apple shared string\nbanana inline string\n42"
        assert units[1].text == "second sheet"

    async def scenario():
        import json

        root = folder(tmp_path)
        (root / ("file" + suffix)).write_bytes(data)
        search = engine(tmp_path)
        try:
            (await search.add_folder(str(root)))
            await search.wait_idle()
            rows = list(search.store.rows().values())
            assert [json.loads(row["locator"])["page"] for row in rows] == pages
            assert all(row["id"] in search.store.vectors for row in rows)
            result = (await search.search("apple", 20, None))["results"][0]
            assert result["page"] == pages[0]
        finally:
            await search.close()

    asyncio.run(scenario())


def pdf_fixture(*, encrypted=False, mixed=False, near_empty=False):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(300, 300)
    if near_empty:
        # Reuse valid resources but retain only a page-number glyph.
        page = writer.add_page(PdfReader(io.BytesIO(small_pdf())).pages[0])
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 30 30 Td (1) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    if mixed:
        writer.add_page(PdfReader(io.BytesIO(small_pdf())).pages[0])
    if encrypted:
        writer.encrypt("password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({}, "Scanned PDF without text"),
        ({"near_empty": True}, "Scanned PDF without text"),
        ({"encrypted": True}, "Password-protected"),
    ],
)
def test_pdf_skip_classification(kwargs, reason):
    with pytest.raises(SkippedError, match=reason):
        extract(Path("file.pdf"), pdf_fixture(**kwargs))


def test_mixed_pdf_indexes_text_page(tmp_path):
    data = pdf_fixture(mixed=True)
    assert extract(Path("mixed.pdf"), data) == [Unit("First", 2)]

    async def scenario():
        root = folder(tmp_path)
        (root / "mixed.pdf").write_bytes(data)
        search = engine(tmp_path)
        try:
            (await search.add_folder(str(root)))
            await search.wait_idle()
            assert (await search.search("First", 20, None))["results"][0]["page"] == 2
            assert search.status()["folders"][0]["skip_reasons"] == {}
        finally:
            await search.close()

    asyncio.run(scenario())


def test_skip_reasons_incremental_restart_and_changes(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service

    expected = {
        "Unsupported file type": 6,
        "Too large": 1,
        "No readable text": 1,
        "Scanned PDF without text": 1,
        "Password-protected": 1,
        "Could not be read": 2,
        "Private or system file": 1,
        "Audio is not supported yet": 1,
    }
    root = folder(tmp_path)
    for suffix in (".doc", ".xls", ".ppt", ".rtf", ".pages", ".unknown"):
        (root / ("old" + suffix)).write_bytes(b"legacy")
    (root / "empty.txt").write_text(" \n\t")
    (root / "bad.docx").write_bytes(b"corrupt")
    (root / "scanned.pdf").write_bytes(pdf_fixture())
    (root / "encrypted.pdf").write_bytes(pdf_fixture(encrypted=True))
    (root / "sound.mp3").write_bytes(b"audio")
    (root / ".env").write_text("secret")
    (root / "big.txt").write_bytes(b"a" * 100)
    (root / "denied.txt").write_text("unreadable")
    (root / "good.txt").write_text("apple")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.bin").write_bytes(b"excluded subtree")
    original_read = service.read_safe
    original_parse = service.parse_document

    def limited_read(path, root, root_fd=None):
        if path.name == "denied.txt":
            raise PermissionError("denied")
        return original_read(path, root, 80, root_fd=root_fd)

    monkeypatch.setattr(service, "read_safe", limited_read)

    # PDF bytes exceed the test-only text cap; let those use the normal cap.
    def read(path, root, root_fd=None):
        return (
            original_read(path, root, root_fd=root_fd)
            if path.suffix == ".pdf"
            else limited_read(path, root, root_fd=root_fd)
        )

    monkeypatch.setattr(service, "read_safe", read)

    async def scenario():
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.status()["folders"][0]["skip_reasons"] == expected
        assert search.store.folder(fid)["files_skipped"] == sum(expected.values()) + 1
        await search.close()
        search = engine(tmp_path, fake)
        try:
            assert search.status()["folders"][0]["skip_reasons"] == expected

            def no_read(*args):
                pytest.fail("unchanged skipped files must not be read or extracted")

            monkeypatch.setattr(service, "read_safe", no_read)
            monkeypatch.setattr(
                service, "parse_document", AsyncMock(side_effect=no_read)
            )
            await search.queue(fid)
            await search.wait_idle()
            assert search.status()["folders"][0]["skip_reasons"] == expected
            assert len(fake.calls) == 1
            monkeypatch.setattr(service, "read_safe", read)
            monkeypatch.setattr(service, "parse_document", original_parse)
            (root / "empty.txt").write_text("banana now readable")
            (root / "scanned.pdf").unlink()
            await search.queue(fid)
            await search.wait_idle()
            changed = dict(expected)
            del changed["No readable text"], changed["Scanned PDF without text"]
            assert search.status()["folders"][0]["skip_reasons"] == changed
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("suffix", [".docx", ".pptx", ".xlsx"])
def test_office_zip_bomb_bound(suffix, monkeypatch):
    # Several individually small members exceed the aggregate expansion cap.
    data = office_zip({f"part{i}.xml": "x" * 800 for i in range(4)})
    reads = []
    original = zipfile.ZipFile.open

    def opened(self, *args, **kwargs):
        reads.append(args[0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", opened)
    with pytest.raises(SkippedError, match="size cap"):
        extract(Path("bomb" + suffix), data, max_bytes=2000)
    assert not reads  # Refuse before any inflation/XML parsing.


def test_xlsx_cell_cap_and_shared_string_expansion(monkeypatch):
    from rapid_mlx.local_search import extraction

    sheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    data, _ = office_fixture(".xlsx")
    monkeypatch.setattr(extraction, "XLSX_MAX_CELLS_PER_SHEET", 2)
    assert (
        extract(Path("a.xlsx"), data)[0].text
        == "apple shared string\nbanana inline string"
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/sharedStrings.xml"] = (
        f'<sst xmlns="{sheet}"><si><t>{"a" * 1000}</t></si></sst>'
    )
    parts["xl/worksheets/sheet2.xml"] = (
        f'<worksheet xmlns="{sheet}">' + '<c t="s"><v>0</v></c>' * 20 + "</worksheet>"
    )
    monkeypatch.setattr(extraction, "XLSX_MAX_CELLS_PER_SHEET", 100)
    with pytest.raises(SkippedError, match="size cap"):
        extract(Path("a.xlsx"), office_zip(parts), max_bytes=6000)


def test_existing_index_upgrade_without_full_rebuild(tmp_path):
    from rapid_mlx.local_search import MODEL_ID

    root = folder(tmp_path)
    store = Store(tmp_path / "index" / "index.sqlite3", MODEL_ID)
    fid = store.add_folder(root)[0]
    good = root / "good.txt"
    good.write_text("apple unchanged")
    store.replace(
        fid,
        good,
        good.stat(),
        None,
        [Chunk("apple unchanged", 1, None)],
        [[1.0, 0.0, 0.0]],
        "indexed",
    )
    good_id = next(iter(store.vectors))
    for suffix in (".docx", ".pptx", ".xlsx"):
        path = root / ("office" + suffix)
        path.write_bytes(office_fixture(suffix)[0])
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [],
            None,
            "indexed" if suffix == ".docx" else "skipped: unsupported",
        )
    blank = root / "scan.pdf"
    blank.write_bytes(pdf_fixture())
    store.replace(fid, blank, blank.stat(), None, [], None, "indexed")
    unsupported = root / "legacy.doc"
    unsupported.write_bytes(b"old")
    store.replace(
        fid, unsupported, unsupported.stat(), None, [], None, "skipped: unsupported"
    )
    # Simulate a database written before the additive extraction upgrade.
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key='extraction_version'")
    store.close()

    async def scenario():
        search = engine(tmp_path)
        try:
            assert search.store.folder(fid)["skip_reasons"] == {
                "Unsupported file type": 1
            }
            assert good_id in search.store.vectors
            search.start()
            await search.wait_idle()
            assert search.store.folder(fid)["skip_reasons"] == {
                "Unsupported file type": 1,
                "Scanned PDF without text": 1,
            }
            assert search.store.folder(fid)["files_indexed"] == 4
            assert good_id in search.store.vectors
            assert search.store.chunk_text(good_id) == "apple unchanged"
            assert not any(
                "unchanged" in text for batch in search.engine.calls for text in batch
            )
        finally:
            await search.close()

    asyncio.run(scenario())


def test_xlsx_reuses_per_file_embedding_budget(tmp_path):
    sheet = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    data, _ = office_fixture(".xlsx")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/worksheets/sheet2.xml"] = (
        f'<worksheet xmlns="{sheet}">'
        + ('<c t="inlineStr"><is><t>apple ' + "legal clause " * 20 + "</t></is></c>")
        * 400
        + "</worksheet>"
    )

    async def scenario():
        root = folder(tmp_path)
        (root / "large.xlsx").write_bytes(office_zip(parts))
        search = engine(tmp_path)
        try:
            (await search.add_folder(str(root)))
            await search.wait_idle()
            assert len(search.store.rows()) > MAX_EMBEDDED_CHUNKS_PER_FILE
            assert len(search.store.vectors) == MAX_EMBEDDED_CHUNKS_PER_FILE
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text",
    [
        "a4" * 150,
        ("QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 3 + "\n") * 5,
        "QWxhZGRpbjpvcGVuIHNlc2FtZQ" * 15,
        '{"payload":[' + ",".join('"value"' for _ in range(60)) + "]}",
        " ".join(chr(97 + i // 26) + chr(97 + i % 26) for i in range(50)),
    ],
)
def test_low_information_shapes(text):
    from rapid_mlx.local_search.eligibility import low_information

    assert low_information(text)


@pytest.mark.parametrize(
    "text",
    [
        "A short note",
        "apple\n" * 50,
        "This report explains how to benchmark throughput and compare the results. "
        * 10,
        "如何运行基准测试并比较每秒处理的令牌数量。" * 30,
        "ベンチマークの結果を比較してグラフを表示します。" * 30,
        "벤치마크 결과와 초당 토큰 처리량을 비교합니다. " * 30,
        "def decode(tokens):\n    return [token.value for token in tokens]\n" * 10,
        "const chart=results.map(row=>({label:row.name,value:row.tokens/row.seconds}));"
        * 10,
        "The values are "
        + "0123456789abcdef" * 5
        + "; keep these notes for the run. " * 20,
    ],
)
def test_language_and_source_remain_semantic(text):
    from rapid_mlx.local_search.eligibility import low_information

    assert not low_information(text)


def test_semantic_format_policy():
    from rapid_mlx.local_search.eligibility import (
        KEYWORD_ONLY_SUFFIXES,
        semantic_format,
    )

    for suffix in KEYWORD_ONLY_SUFFIXES:
        assert not semantic_format(Path("data" + suffix.upper()))
    for suffix in (
        ".md",
        ".txt",
        ".rst",
        ".tex",
        ".html",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".py",
        ".swift",
    ):
        assert semantic_format(Path("document" + suffix))


def test_machine_formats_and_noise_index_without_embedding(tmp_path):
    async def scenario():
        root = folder(tmp_path)
        from rapid_mlx.local_search.eligibility import KEYWORD_ONLY_SUFFIXES

        for suffix in KEYWORD_ONLY_SUFFIXES:
            (root / ("data" + suffix)).write_text("first line\nunique_literal_needle")
        (root / "noise.txt").write_text("ab" * 200)
        (root / "notes.md").write_text("apple benchmark throughput")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert len(fake.calls) == 1 and len(fake.calls[0]) == 1
        assert len(search.store.vectors) == 1
        hits = (await search.search("unique_literal_needle", 100, None))["results"]
        data_hits = [h for h in hits if Path(h["path"]).suffix in KEYWORD_ONLY_SUFFIXES]
        assert len(data_hits) == len(KEYWORD_ONLY_SUFFIXES)
        assert all(h["matched"] == ["keyword"] and h["line"] == 1 for h in data_hits)
        calls = len(fake.calls)
        await search.queue(fid)
        await search.wait_idle()
        assert len(fake.calls) == calls
        await search.close()

    asyncio.run(scenario())


def test_eligibility_migration_keeps_keyword_locators_and_prose(tmp_path):
    root = folder(tmp_path)
    db = tmp_path / "migration.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    for name, text, status, vectors in [
        ("data.jsonl", "needle", "indexed", [[1.0, 0.0]]),
        ("noise.txt", "ab" * 200, "indexed", [[1.0, 0.0]]),
        ("note.md", "readable prose needle", "indexed", [[0.0, 1.0]]),
        ("offline.txt", "readable but no model yet", "keyword", None),
        ("offline.log", "literal data", "keyword", None),
    ]:
        path = root / name
        path.write_text(text)
        store.replace(
            fid, path, path.stat(), name, [Chunk(text, 42, None)], vectors, status
        )
    before = store.rows()
    prose_id = next(i for i, r in before.items() if r["path"].endswith("note.md"))
    prose_vector = store.vectors[prose_id].copy()
    with store.db:
        store.db.execute("DELETE FROM meta WHERE key='eligibility_version'")
    store.close()
    store = Store(db, "model")
    assert store.rows() == before
    assert list(store.vectors) == [prose_id]
    np.testing.assert_array_equal(store.vectors[prose_id], prose_vector)
    assert len(store.keyword(["needle"], set(before))) == 2
    assert store.files(fid)[str(root / "offline.txt")]["status"] == "keyword"
    assert store.files(fid)[str(root / "offline.log")]["status"] == "indexed"
    store.close()
    # Opening twice has no further migration or model-dependent changes.
    store = Store(db, "model")
    assert store.rows() == before and list(store.vectors) == [prose_id]
    store.close()


def test_model_worker_lifecycle_and_streaming_progress(tmp_path):
    """Generation and lazy materialisation share the same one-thread boundary."""
    from concurrent.futures import ThreadPoolExecutor

    async def scenario():
        loop_thread = threading.get_ident()
        events, streamed = [], []
        gate = BlockingCall()
        streaming_done = threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)

        async def worker(func, *args, **kwargs):
            return await asyncio.get_running_loop().run_in_executor(
                executor, lambda: func(*args, **kwargs)
            )

        class LazyVectors:
            def __array__(self, dtype=None, copy=None):
                events.append(("materialize", threading.get_ident()))
                return np.array([[1.0, 0.0, 0.0]], dtype=dtype)

        class Embedder:
            is_loaded = True
            vision = False

            def __init__(self, path):
                events.append(("load", threading.get_ident()))

            def embed(self, texts):
                events.append(("embed", threading.get_ident()))
                gate()
                assert streaming_done.is_set()
                return LazyVectors()

            def __del__(self):
                events.append(("teardown", threading.get_ident()))

        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        search = LocalSearch(
            tmp_path / "index",
            resolve_model=lambda _: "local",
            embedder_factory=Embedder,
            model_worker=worker,
        )

        async def stream():
            await asyncio.wait_for(gate.entered.wait(), SYNC_TIMEOUT)

            async def fragment(n):
                assert not gate.release.is_set()
                streamed.append(n)

            await asyncio.gather(*(fragment(n) for n in range(25)))
            streaming_done.set()
            gate.release.set()

        async def generation():
            for _ in range(25):
                await worker(lambda: events.append(("generate", threading.get_ident())))

        try:
            (await search.add_folder(str(root)))
            await asyncio.gather(search.wait_idle(), stream(), generation())
            await search.search("apple", 20, None)
            await search.close()
            assert {name for name, _ in events} == {
                "load",
                "embed",
                "materialize",
                "teardown",
                "generate",
            }
            assert len({thread for _, thread in events}) == 1
            assert all(thread != loop_thread for _, thread in events)
            assert streamed == list(range(25))
        finally:
            gate.release.set()
            if not search.closed:
                await search.close()
            executor.shutdown()

    asyncio.run(scenario())


def test_missing_model_worker_fails_closed(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config

    monkeypatch.setattr(get_config(), "engine", object())

    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        assert not fake.calls
        assert (
            "sanctioned model worker" in search.status()["semantic_unavailable_reason"]
        )
        assert (await search.search("apple", 20, None))["mode"] == "keyword"
        await search.close()

    asyncio.run(scenario())


def test_ancestor_symlink_and_pinned_pass(tmp_path, monkeypatch):
    from rapid_mlx.local_search import extraction, service

    ancestor = folder(tmp_path, "work")
    root = ancestor / "docs"
    root.mkdir()
    path = root / "a.txt"
    path.write_text("apple inside selected root")
    outside = folder(tmp_path, "outside")
    (outside / "docs").mkdir()
    (outside / "docs" / "a.txt").write_text("apple outside confidential content")
    saved = tmp_path / "saved"
    original_read = service.read_safe

    def swap_read(path, root, **kwargs):
        ancestor.rename(saved)
        ancestor.symlink_to(outside, target_is_directory=True)
        try:
            with pytest.raises(OSError), extraction.open_root(root):
                pass
            data, info = original_read(path, root, **kwargs)
            assert b"outside confidential" not in data
            return data, info
        finally:
            ancestor.unlink()
            saved.rename(ancestor)

    monkeypatch.setattr(service, "read_safe", swap_read)

    async def scenario():
        search = engine(tmp_path)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        hit = (await search.search("apple", 20, None))["results"][0]
        assert "inside selected" in hit["snippet"]
        ancestor.rename(saved)
        ancestor.symlink_to(outside, target_is_directory=True)
        try:
            assert not (await search.search("apple", 20, None))["results"]
        finally:
            ancestor.unlink()
            saved.rename(ancestor)
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "name",
    [
        "credentials.json",
        "secrets.yaml",
        "passwords.txt",
        "vault.kdbx",
        ".netrc",
        "backup_rsa",
        "service-account-prod.json",
        "service_account.json",
    ],
)
def test_secret_filename_skip_reason(tmp_path, name):
    async def scenario():
        root = folder(tmp_path)
        path = root / name
        path.write_text("supersecret")
        search = engine(tmp_path)
        fid = (await search.add_folder(str(root)))[0]["id"]
        await search.wait_idle()
        assert search.store.folder(fid)["skip_reasons"] == {"Private or system file": 1}
        assert search.store.totals()["chunks"] == 0
        assert not (await search.search("supersecret", 20, None))["results"]
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["timeout", "cancel", "memory"])
def test_parser_killed_and_reaped(monkeypatch, stop):
    import psutil

    from rapid_mlx.local_search import parsing

    children = []
    spawn = asyncio.create_subprocess_exec
    spawned = asyncio.Event()

    async def observed(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        children.append(process)
        spawned.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observed)
    if stop == "timeout":
        wait_for = asyncio.wait_for

        async def injected_timeout(awaitable, timeout):
            # Keep the synchronization guard independent of the injected deadline.
            task = asyncio.create_task(awaitable)
            await wait_for(spawned.wait(), SYNC_TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", injected_timeout)
    if stop == "memory":
        monkeypatch.setattr(parsing, "PARSER_MEMORY_BYTES", 1)

    async def scenario():
        task = asyncio.create_task(
            parsing.run_parser(
                [sys.executable, "-c", "import signal; signal.pause()"], b""
            )
        )
        if stop == "cancel":
            await asyncio.wait_for(spawned.wait(), SYNC_TIMEOUT)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(SkippedError, match="Could not be read"):
                await task
        assert children[0].returncode is not None
        assert not psutil.pid_exists(children[0].pid)

    asyncio.run(scenario())


def test_chunk_line_count_work_is_linear():
    class CountedText(str):
        counted = 0

        def count(self, substring, start=0, end=None):
            end = len(self) if end is None else end
            self.counted += end - start
            return super().count(substring, start, end)

    counts = []
    for size in (1024 * 1024, 4 * 1024 * 1024, 20 * 1024 * 1024):
        text = CountedText(("line words here\n" * (size // 16 + 1))[:size])
        chunks = chunk_units([Unit(text)])
        assert chunks[-1].line > chunks[0].line
        assert text.counted <= len(text)
        counts.append(text.counted)
    assert counts[1] <= 4 * (counts[0] + 1200)
    assert counts[2] <= 20 * (counts[0] + 1200)


def test_corpus_budget_blocks_and_keyword_limits(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as storage

    monkeypatch.setattr(storage, "MAX_CORPUS_EMBEDDED_CHUNKS", 4)
    monkeypatch.setattr(storage, "KEYWORD_CANDIDATE_LIMIT", 3)

    async def scenario():
        root = folder(tmp_path)
        for n in range(6):
            (root / f"{n}.txt").write_text("apple")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        totals = search.status()["totals"]
        assert totals["semantic_index_full"] and totals["semantic_chunk_budget"] == 4
        assert totals["embedded_chunks"] == 4 and totals["keyword_only_chunks"] == 2
        assert len(fake.calls) == 4
        shapes = []
        stack = np.stack

        def observed(arrays, *args, **kwargs):
            shapes.append(len(arrays))
            return stack(arrays, *args, **kwargs)

        monkeypatch.setattr(np, "stack", observed)
        rows = search.store.search_rows()
        search.store.semantic_scores([1, 0, 0], set(rows))
        assert shapes == []
        assert search.store._matrix.shape == (4, 3)
        assert all(
            np.shares_memory(v, search.store._matrix)
            for v in search.store.vectors.values()
        )
        assert len(search.store.keyword(["apple"], set(rows))) <= 3
        # Scope is applied before candidate limits, so the last eligible ID survives.
        last = max(rows)
        assert search.store.keyword(["apple"], {last}) == [last]
        first = next(iter(search.store.files(search.store.folders()[0]["id"]).values()))
        search.store.remove_file(first["id"])
        assert not search.status()["totals"]["semantic_index_full"]
        await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("count", [1, 2, 3, 12])
def test_degenerate_semantic_neighbours(count):
    from rapid_mlx.local_search.search import SEMANTIC_FALLBACK_FILES, semantic_ranking

    rows = {n: {"path": f"file-{n}"} for n in range(count)}
    assert semantic_ranking([(n, 0.8) for n in range(count)], rows, 1.0) == list(
        range(min(count, SEMANTIC_FALLBACK_FILES))
    )
    if count < 4:
        assert semantic_ranking(
            [(n, 0.8 - 0.01 * n) for n in range(count)], rows, 1.0
        ) == list(range(count))


def test_flag_off_server_start_import_isolation():
    """Fresh interpreter: stub only GPU boot, retain actual server import graph."""
    import subprocess

    script = """
import asyncio, importlib.machinery, sys, types
from unittest.mock import MagicMock
from fastapi import FastAPI
core = types.ModuleType("mlx.core")
core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
mlx = types.ModuleType("mlx")
mlx.__path__ = []
mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None)
mlx.core = core
sys.modules["mlx"] = mlx
sys.modules["mlx.core"] = core
import rapid_mlx.engine
rapid_mlx.engine.BatchedEngine = MagicMock()
import rapid_mlx.server
from rapid_mlx.local_search_runtime import configure, startup, shutdown
configure(False)
app = FastAPI()
asyncio.run(startup(app))
asyncio.run(shutdown(app))
assert not any(name == "rapid_mlx.local_search" or name.startswith(("rapid_mlx.local_search.", "cv2", "pypdf")) for name in sys.modules)
assert not hasattr(app.state, "local_search")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("count", [1, 2, 3])
def test_identical_file_embeddings_return_paraphrase_neighbours(tmp_path, count):
    async def scenario():
        root = folder(tmp_path)
        for n in range(count):
            (root / f"{n}.txt").write_text("apple")
        search = engine(tmp_path)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        search._embed = lambda _: np.array([[0.8, 0.6, 0.0]])
        hits = (await search.search("paraphrase", 20, None))["results"]
        assert len(hits) == count
        assert all(hit["matched"] == ["semantic"] for hit in hits)
        await search.close()

    asyncio.run(scenario())


def test_async_status_keeps_http_loop_responsive(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        gate = BlockingCall()

        def expensive():
            gate()
            return {}

        search.store.totals = expensive
        status = asyncio.create_task(search.status_async())
        try:
            await asyncio.wait_for(gate.entered.wait(), SYNC_TIMEOUT)

            async def other_request():
                assert not gate.release.is_set()
                assert not status.done()
                return "responsive"

            assert await asyncio.create_task(other_request()) == "responsive"
        finally:
            gate.release.set()
            try:
                await status
            finally:
                await search.close()

    asyncio.run(scenario())


def test_corpus_budget_survives_restart_and_recovers(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as storage

    root = folder(tmp_path)
    db = tmp_path / "index.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    for n in range(4):
        path = root / f"{n}.txt"
        path.write_text("apple")
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk("apple", 1, None)],
            [[1.0, 0.0, 0.0]],
            "indexed",
        )
    store.close()
    monkeypatch.setattr(storage, "MAX_CORPUS_EMBEDDED_CHUNKS", 2)
    store = Store(db, "model")
    assert len(store.vectors) == 2 and store.totals()["semantic_index_full"]
    files = store.files(fid)
    assert sum(row["status"] == "keyword" for row in files.values()) == 2
    store.remove_file(files[str(root / "0.txt")]["id"])
    path = root / "2.txt"
    assert store.embedding_room(path) == 1
    store.replace(
        fid,
        path,
        path.stat(),
        None,
        [Chunk("apple", 1, None)],
        [[1.0, 0.0, 0.0]],
        "indexed",
    )
    assert len(store.vectors) == 2 and store.totals()["semantic_index_full"]
    store.close()


def test_model_worker_removal_releases_current_engine(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config

    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("apple")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        (await search.add_folder(str(root)))
        await search.wait_idle()
        calls = len(fake.calls)
        monkeypatch.setattr(get_config(), "engine", object())
        result = await search.search("apple", 20, None)
        assert result["mode"] == "keyword" and len(fake.calls) == calls
        assert search.engine is None
        assert (
            "no sanctioned model worker"
            in search.status()["semantic_unavailable_reason"]
        )
        await search.close()

    asyncio.run(scenario())


def test_warm_search_no_corpus_iteration_or_vector_select(tmp_path, monkeypatch):
    """Guard the shape of latency, independently of wall-clock runner speed."""

    async def scenario():
        search = engine(tmp_path, FakeEmbedder())
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        for n in range(12):
            path = root / f"document-{n}.txt"
            path.write_text("apple chart")
            search.store.replace(
                fid,
                path,
                path.stat(),
                None,
                [Chunk("apple chart", 1, None)] * 64,
                [[1, n / 20, 0.1]] * 64,
                "indexed",
            )

        class GuardedRows(dict):
            checking = False

            def __iter__(self):
                assert not self.checking, "warm search iterated corpus IDs"
                return super().__iter__()

            def items(self):
                assert not self.checking, "warm search iterated corpus metadata"
                return super().items()

            def values(self):
                assert not self.checking, "warm search iterated corpus metadata"
                return super().values()

        owned = search.store._store
        rows = GuardedRows(search.store.search_rows())
        key = next(iter(owned._search_cache))
        owned._search_cache[key] = rows
        owned._view_scope = (rows, *owned._view_scope[1:])
        await search.search("apple chart", 10, None, group_limit=5)
        rows.checking = True

        class NoIteration(dict):
            def __iter__(self):
                raise AssertionError("warm search iterated corpus vectors")

            def items(self):
                raise AssertionError("warm search iterated corpus vectors")

            def values(self):
                raise AssertionError("warm search iterated corpus vectors")

        owned = search.store._store
        owned.vectors = NoIteration(owned.vectors)
        statements = []
        # Install on the connection's owning worker, never from the test thread.
        search.store._executor.submit(
            owned.db.set_trace_callback, statements.append
        ).result()
        await search.search("apple chart", 10, None, group_limit=5)
        assert not any(
            "vector" in sql.lower() and sql.lstrip().upper().startswith("SELECT")
            for sql in statements
        )
        assert not any("INSERT INTO local_allowed" in sql for sql in statements)
        plans = search.store.sql(
            "EXPLAIN QUERY PLAN SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "AND EXISTS (SELECT 1 FROM local_allowed WHERE id=chunks_fts.rowid)",
            ('"apple"',),
        )
        assert any("VIRTUAL TABLE INDEX" in r[3] and "=M" not in r[3] for r in plans)
        await search.close()

    asyncio.run(scenario())


def test_media_admission_matches_scalar_file_scores(tmp_path):
    from rapid_mlx.local_search.search import query_statistics, semantic_ranking

    async def scenario():
        search = engine(tmp_path, FakeEmbedder())
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        for kind in ("text", "image", "video"):
            for n, cosine in enumerate((0.95, 0.8, 0.65, 0.1, -0.2, -0.7)):
                path = (
                    root
                    / f"{kind}-{n}.{'txt' if kind == 'text' else 'png' if kind == 'image' else 'mp4'}"
                )
                path.write_text("unrelated fixture")
                vector = [cosine, np.sqrt(1 - cosine**2), 0]
                search.store.replace(
                    fid,
                    path,
                    path.stat(),
                    None,
                    [Chunk("unrelated fixture", 1, None, kind)] * (n + 1),
                    [vector] * (n + 1),
                    "indexed",
                )
        rows = search.store.search_rows()
        scalar = sorted(
            [
                (i, float(v @ np.array([1, 0, 0], dtype=np.float32)))
                for i, v in search.store.vectors.items()
            ],
            key=lambda p: (-p[1], p[0]),
        )
        accelerated = search.store.semantic_scores([1, 0, 0], rows)
        background = search.store.semantic_background(rows)
        for kind in ("text", "image", "video"):
            old = [(i, v) for i, v in scalar if rows[i]["kind"] == kind]
            new = [(i, v) for i, v in accelerated if rows[i]["kind"] == kind]
            assert query_statistics(new, rows) == query_statistics(old, rows)
            assert semantic_ranking(new, rows, background[kind]) == semantic_ranking(
                old, rows, background[kind]
            )
        search._embed = lambda _: [[1, 0, 0]]
        response = await search.search("unseen visual query", 10, None, group_limit=5)
        assert {g["kind"] for g in response["groups"]} == {"text", "image", "video"}
        image = next(g for g in response["groups"] if g["kind"] == "image")
        assert image["results"][0]["path"].endswith("image-0.png")
        await search.close()

    asyncio.run(scenario())


def test_query_waits_for_only_current_index_embedding_batch(tmp_path):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def worker(func):
            if not calls:
                entered.set()
                await release.wait()
            return func()

        search = LocalSearch(
            tmp_path / "priority",
            model_id="test",
            resolve_model=lambda _: None,
            model_worker=worker,
        )
        search._model_gate = ObservedLock()
        search._embed = lambda inputs: calls.append(inputs[0]) or [[1, 0, 0]]

        async def indexing():
            await search._embed_async(["batch-1"] * 4)
            await search._embed_async(["batch-2"] * 4)

        index = asyncio.create_task(indexing())
        await entered.wait()
        query = asyncio.create_task(search._embed_async(["query"]))
        await asyncio.wait_for(search._model_gate.waiting.wait(), SYNC_TIMEOUT)
        release.set()
        await asyncio.gather(index, query)
        assert calls == ["batch-1", "query", "batch-2"]
        await search.close()

    asyncio.run(scenario())


def test_warm_search_does_not_queue_behind_index_transaction(tmp_path):
    async def scenario():
        search = engine(tmp_path, FakeEmbedder())
        root = folder(tmp_path)
        fid = search.store.add_folder(root)[0]
        first = root / "first.txt"
        first.write_text("apple chart")
        search.store.replace(
            fid,
            first,
            first.stat(),
            None,
            [Chunk("apple chart", 1, None)],
            [[1, 0, 0]],
            "indexed",
        )
        await search.search("apple chart", 10, None)
        entered, release = threading.Event(), threading.Event()

        class PausedChunks(list):
            def __iter__(self):
                for n, chunk in enumerate(super().__iter__()):
                    if n == 1:
                        entered.set()  # The writer already has an open transaction.
                        assert release.wait(SYNC_TIMEOUT)
                    yield chunk

        second = root / "second.txt"
        second.write_text("other document")
        indexing = asyncio.create_task(
            search.store.call(
                "replace",
                fid,
                second,
                second.stat(),
                None,
                PausedChunks([Chunk("other document", 1, None)] * 2),
                [[0, 1, 0]] * 2,
                "indexed",
            )
        )
        query = None
        try:
            assert await asyncio.to_thread(entered.wait, SYNC_TIMEOUT)
            query = asyncio.create_task(search.search("apple chart", 10, None))
            result = await asyncio.wait_for(asyncio.shield(query), SYNC_TIMEOUT)
            assert not release.is_set() and not indexing.done()
            assert result["results"][0]["path"] == str(first)
        finally:
            release.set()
            await indexing
            if query is not None:
                await query
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation",
    ["add", "delete", "reindex", "reindex_all", "search", "status", "resume"],
)
def test_routes_keep_heartbeat_during_busy_db(tmp_path, monkeypatch, operation):
    from rapid_mlx.local_search import service as module

    main_thread = threading.get_ident()
    validate = module.validate_folder

    def off_loop_validate(value):
        assert threading.get_ident() != main_thread
        return validate(value)

    monkeypatch.setattr(module, "validate_folder", off_loop_validate)

    async def scenario():
        root = folder(tmp_path)
        (root / "apple.txt").write_text("apple evidence")
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            registered = (
                await client.post("/v1/local-search/folders", json={"path": str(root)})
            ).json()
            await app.state.local_search.wait_idle()
            search = app.state.local_search
            gate = BlockingCall()
            waiting = asyncio.Event()
            call = search.store.call

            async def observed_call(*args, **kwargs):
                waiting.set()
                return await call(*args, **kwargs)

            monkeypatch.setattr(search.store, "call", observed_call)
            busy = search.store._executor.submit(gate)
            request = None

            async def operation_request():
                fid = registered["id"]
                if operation == "add":
                    response = await client.post(
                        "/v1/local-search/folders", json={"path": str(root)}
                    )
                elif operation == "delete":
                    response = await client.delete(f"/v1/local-search/folders/{fid}")
                elif operation in {"reindex", "reindex_all"}:
                    response = await client.post(
                        "/v1/local-search/reindex",
                        json={"folder_id": fid if operation == "reindex" else None},
                    )
                elif operation == "search":
                    response = await client.post(
                        "/v1/local-search/search",
                        json={"query": "apple", "folder_ids": [fid]},
                    )
                elif operation == "status":
                    response = await client.get("/v1/local-search/status")
                else:
                    search.start()
                    await search.wait_idle()
                    response = await client.get("/v1/local-search/status")
                assert response.status_code in {200, 202, 204}
                return response

            try:
                await asyncio.wait_for(gate.entered.wait(), SYNC_TIMEOUT)
                request = asyncio.create_task(operation_request())
                await asyncio.wait_for(waiting.wait(), SYNC_TIMEOUT)

                async def heartbeat():
                    assert not gate.release.is_set()
                    assert not busy.done()
                    assert not request.done()
                    return "responsive"

                assert await asyncio.create_task(heartbeat()) == "responsive"
                gate.release.set()
                await asyncio.wait_for(asyncio.shield(request), SYNC_TIMEOUT)
                await search.wait_idle()
            finally:
                gate.release.set()
                await asyncio.wrap_future(busy)
                if request is not None:
                    await request

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_first", [False, True])
@pytest.mark.parametrize("boundary", ["Library", "private", "external"])
def test_explicit_pruned_roots_remain_registered(
    tmp_path, monkeypatch, parent_first, boundary
):
    from rapid_mlx.local_search import extraction

    parent = folder(tmp_path)
    pruned = folder(parent, boundary)
    child = folder(pruned, "Documents")
    (child / "deposition.txt").write_text("apple evidence")
    if boundary == "Library":
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: parent))
    else:
        monkeypatch.setattr(
            extraction, "SKIP_PREFIXES", (*extraction.SKIP_PREFIXES, pruned)
        )

    async def scenario():
        search = engine(tmp_path)
        try:
            first, second = (parent, child) if parent_first else (child, parent)
            await search.add_folder(str(first))
            await search.wait_idle()
            await search.add_folder(str(second))
            await search.wait_idle()
            assert len(search.store.folders()) == 2
            assert search.store.totals()["files"] == 1
            results = (await search.search("deposition", 20, None))["results"]
            assert [r["path"] for r in results] == [str(child / "deposition.txt")]
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("deleted", [False, True])
def test_publication_uses_committed_owner_after_absorption(tmp_path, deleted):
    parent = folder(tmp_path)
    child = folder(parent, "child")
    path = child / "apple.txt"
    path.write_text("apple")
    store = Store(tmp_path / "index.sqlite3", "test")
    try:
        fid = store.add_folder(child)[0]
        change = store._write_replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk("apple", 1, None)],
            [[1, 0, 0]],
            "indexed",
            1,
        )
        parent_id = store.add_folder(parent)[0]
        if deleted:
            store.remove_file(change[1])
        store._publish_replace(change)
        if deleted:
            assert (
                not store.vectors and not store._file_stats and not store._folder_stats
            )
        else:
            assert store._file_stats[change[1]][0] == parent_id
            assert set(store._folder_stats) == {parent_id}
            before = store.semantic_background(store.search_rows([parent_id]))
            store.close()
            store = Store(tmp_path / "index.sqlite3", "test")
            assert store.semantic_background(store.search_rows([parent_id])) == before
            assert store._folder_stats[parent_id][2] == 1
    finally:
        store.close()


def test_budget_truncated_file_upgrades_only_when_capacity_freed(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service as module
    from rapid_mlx.local_search import store as storage

    monkeypatch.setattr(storage, "MAX_CORPUS_EMBEDDED_CHUNKS", 2)
    real_chunks = module.chunk_units
    monkeypatch.setattr(
        module,
        "chunk_units",
        lambda units: (
            [Chunk("apple paragraph", 1, None), Chunk("banana paragraph", 2, None)]
            if units[0].text.startswith("two")
            else real_chunks(units)
        ),
    )

    async def scenario():
        root = folder(tmp_path)
        (root / "a.txt").write_text("one apple paragraph")
        (root / "b.txt").write_text("two paragraphs")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            assert search.store.files(fid)[str(root / "b.txt")]["status"] == "budget"
            calls = len(fake.calls)
            await search.queue(fid)
            await search.wait_idle()
            assert len(fake.calls) == calls
            (root / "a.txt").unlink()
            # The first walk frees capacity at cleanup; the next upgrades b.
            await search.queue(fid)
            await search.wait_idle()
            await search.queue(fid)
            await search.wait_idle()
            assert len(fake.calls) == calls + 1
            assert search.store.files(fid)[str(root / "b.txt")]["status"] == "indexed"
            assert len(search.store.vectors) == 2
            await search.queue(fid)
            await search.wait_idle()
            assert len(fake.calls) == calls + 1
        finally:
            await search.close()

    asyncio.run(scenario())


def test_large_query_and_sql_error_envelope(tmp_path, monkeypatch):
    import sqlite3

    from rapid_mlx.local_search.store import MAX_KEYWORD_TERMS, bounded_terms

    query = " ".join(chr(0x4E00 + i) for i in range(1100))
    assert len(query) == 2199
    assert len(bounded_terms(query.split())) == MAX_KEYWORD_TERMS
    assert bounded_terms(["aa", "bbbb", "ccc", *query.split()], ["aa"])[0:3] == [
        "aa",
        "bbbb",
        "ccc",
    ]

    async def scenario():
        root = folder(tmp_path)
        (root / "document.txt").write_text("apple evidence")
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            await client.post("/v1/local-search/folders", json={"path": str(root)})
            await app.state.local_search.wait_idle()
            for text in [query, '" OR * ( ) : NEAR / - % _', "x" * 8192]:
                response = await client.post(
                    "/v1/local-search/search", json={"query": text}
                )
                assert response.status_code == 200

            async def failed(*args):
                raise sqlite3.OperationalError("invalid expression")

            monkeypatch.setattr(app.state.local_search, "search", failed)
            response = await client.post(
                "/v1/local-search/search", json={"query": query}
            )
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "invalid_query"
            assert response.json()["error"]["param"] == "query"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "name,data,reason,kind",
    [
        ("scanned_deposition.pdf", pdf_fixture(), "Scanned PDF without text", "text"),
        (
            "encrypted_deposition.pdf",
            pdf_fixture(encrypted=True),
            "Password-protected",
            "text",
        ),
        ("unsupported_deposition.xyz", b"contentword", "Unsupported file type", "text"),
        ("empty_deposition.txt", b"", "No readable text", "text"),
        ("broken_deposition.pdf", b"contentword", "Could not be read", "text"),
        ("audio_deposition.mp3", b"contentword", "Audio is not supported yet", "text"),
        ("image_deposition.png", b"contentword", "Could not be read", "image"),
        ("video_deposition.mp4", b"contentword", "Could not be read", "video"),
    ],
)
def test_name_only_result_contract_and_groups(tmp_path, name, data, reason, kind):
    async def scenario():
        root = folder(tmp_path)
        path = root / name
        path.write_bytes(data)
        (root / "secret.txt").write_text("deposition contentword")
        search = engine(tmp_path)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            body = await search.search("deposition", 20, None, group_limit=5)
            assert len(body["results"]) == 1
            result = body["results"][0]
            assert result["path"] == str(path)
            assert (
                result["content_indexed"] is False and result["skip_reason"] == reason
            )
            assert result["snippet"] == "" and result["highlights"] == []
            assert result["page"] is None and result["line"] is None
            assert result["matched"] == ["keyword"] and result["kind"] == kind
            assert body["groups"] == [{"kind": kind, "results": [result], "total": 1}]
            for query in ["contentword", "secret", "unrelated"]:
                assert not (await search.search(query, 20, None))["results"]
            assert not (await search.search("deposition", 20, []))["results"]
            assert not (await search.search("deposition", 20, None, []))["results"]
            totals = search.store.totals()
            assert totals["name_only_files"] == 1
            assert totals["files"] == totals["chunks"] == totals["embedded_chunks"] == 0
            assert totals["files_by_kind"] == {"text": 0, "image": 0, "video": 0}
            metadata = search.store.folder(fid)
            assert metadata["files_skipped"] == 2 and metadata["files_indexed"] == 0
            assert metadata["skip_reasons"] == {reason: 1, "Private or system file": 1}
        finally:
            await search.close()

    asyncio.run(scenario())


def test_name_only_lifecycle_cap_and_legacy_entries(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service as module
    from rapid_mlx.local_search import store as storage

    monkeypatch.setattr(storage, "MAX_NAME_ONLY_FILES_PER_FOLDER", 2)
    root = folder(tmp_path)
    pdf = root / "deposition.pdf"
    pdf.write_bytes(pdf_fixture())
    for name in ["second.xyz", "third.xyz"]:
        (root / name).write_text("unreadbody")
    # Persist precisely the old skipped-row shape, without a parallel table.
    old = Store(tmp_path / "index" / "index.sqlite3", "test")
    fid = old.add_folder(root)[0]
    old.replace(
        fid, pdf, pdf.stat(), None, [], None, "skipped: Scanned PDF without text"
    )
    old.close()
    read = module.read_safe

    def no_unsupported_read(path, *args, **kwargs):
        assert path.suffix != ".xyz"
        return read(path, *args, **kwargs)

    monkeypatch.setattr(module, "read_safe", no_unsupported_read)

    async def scenario():
        search = engine(tmp_path)
        try:
            search.start()
            await search.wait_idle()
            assert search.store.totals()["name_only_files"] == 2
            assert search.store.folder(fid)["files_skipped"] == 3
            assert (await search.search("deposition", 20, None))["results"]
            assert not (await search.search("third", 20, None))["results"]
            pdf.rename(root / "renamed.pdf")
            await search.queue(fid)
            await search.wait_idle()
            assert not (await search.search("deposition", 20, None))["results"]
            (root / "second.xyz").unlink()
            await search.queue(fid)
            await search.wait_idle()
            assert (await search.search("renamed", 20, None))["results"][0][
                "content_indexed"
            ] is False
            (root / "renamed.pdf").write_bytes(small_pdf())
            await search.queue(fid)
            await search.wait_idle()
            hit = (await search.search("renamed", 20, None))["results"][0]
            assert (
                hit["content_indexed"] is True
                and hit["skip_reason"] is None
                and hit["snippet"]
            )
            assert search.store.totals()["name_only_files"] == 1
            parent_id = (await search.add_folder(str(tmp_path)))[0]["id"]
            await search.wait_idle()
            hits = (await search.search("third", 20, None))["results"]
            named = [hit for hit in hits if not hit["content_indexed"]]
            assert len(named) == 1 and named[0]["folder_id"] == parent_id
            await search.delete_folder(parent_id)
            assert search.store.totals()["name_only_files"] == 0
            assert not (await search.search("third", 20, None))["results"]
        finally:
            await search.close()

    asyncio.run(scenario())


def test_name_only_size_rejection_without_read(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service as module

    monkeypatch.setattr(module, "MAX_BYTES", 4)
    monkeypatch.setattr(
        module,
        "read_safe",
        lambda *args, **kwargs: pytest.fail("oversized file must not be read"),
    )

    async def scenario():
        root = folder(tmp_path)
        (root / "large-deposition.txt").write_text("unreadbody")
        search = engine(tmp_path)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            result = (await search.search("deposition", 20, None))["results"][0]
            assert (
                result["skip_reason"] == "Too large"
                and result["content_indexed"] is False
            )
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "missing",
    [None, "tokenizer.json", "tokenizer_config.json", "processor_config.json"],
)
@pytest.mark.parametrize("weights", [False, True])
def test_cached_model_main_ref_ignores_missing_tree_readme(
    tmp_path, monkeypatch, weights, missing
):
    import json

    import huggingface_hub
    from huggingface_hub import constants

    from rapid_mlx import model_metadata

    cache = tmp_path / "cache"
    repo = cache / "models--test--embedding"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        '{"model_type":"embedding_gemma2","vision_config":{"hidden_size":8}}'
    )
    for name in ["tokenizer.json", "tokenizer_config.json", "processor_config.json"]:
        if name != missing:
            (snapshot / name).write_text('{"image_processor":{}}')
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"w" * 4096)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(snapshot.name)
    (repo / "trees").mkdir()
    (repo / "trees" / (snapshot.name + ".json")).write_text(
        json.dumps(
            [
                {"type": "file", "path": name, "size": 4096}
                for name in ["config.json", "model.safetensors", "README.md"]
            ]
        )
    )
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: pytest.fail(
            "must resolve the runnable snapshot without a download call"
        ),
    )
    assert model_metadata.resolve_unreferenced_cached_snapshot("test/embedding") is None
    assert model_metadata.resolve_offline_cached_snapshot("test/embedding") is None
    assert local_model_path("test/embedding") == (
        str(snapshot) if weights and missing is None else None
    )


def test_name_only_metadata_validation_no_reads_and_stale_paths(tmp_path, monkeypatch):
    from rapid_mlx.local_search import extraction
    from rapid_mlx.local_search import service as module

    root = folder(tmp_path)
    child = folder(root, "nested")
    path = child / "Über_deposition.xyz"
    path.write_text("not readable content")

    async def scenario():
        search = engine(tmp_path)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            monkeypatch.setattr(
                module,
                "open_safe",
                lambda *a, **k: pytest.fail(
                    "name-only validation must not open content"
                ),
            )
            leaf = extraction.os.open

            def directories_only(name, flags, *args, **kwargs):
                assert flags & os.O_DIRECTORY, (
                    "name-only validation must open only directories"
                )
                return leaf(name, flags, *args, **kwargs)

            monkeypatch.setattr(extraction.os, "open", directories_only)
            result = await search.search("über", 20, None)
            assert [r["path"] for r in result["results"]] == [str(path)]
            path.write_text("changed source fingerprint")
            assert not (await search.search("über", 20, None))["results"]
            path.unlink()
            path.symlink_to(root / "missing")
            assert not (await search.search("über", 20, None))["results"]
            path.unlink()
            child.rename(root / "moved")
            child.symlink_to(root / "moved", target_is_directory=True)
            assert not (await search.search("über", 20, None))["results"]
        finally:
            await search.close()

    asyncio.run(scenario())


def test_cancel_during_root_open_drains_and_closes(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service as module

    root = folder(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    opening = module.open_root

    class DelayedRoot:
        def __init__(self, value):
            self.context = opening(value)

        def __enter__(self):
            entered.set()
            assert release.wait(SYNC_TIMEOUT)
            return self.context.__enter__()

        def __exit__(self, *args):
            self.context.__exit__(*args)
            finished.set()

    monkeypatch.setattr(module, "open_root", DelayedRoot)

    async def scenario():
        search = engine(tmp_path)
        try:
            task = asyncio.create_task(search._index({"path": str(root)}))
            await asyncio.to_thread(entered.wait)
            task.cancel()
            await loop_checkpoint()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            release.set()
            await search.close()

    asyncio.run(scenario())


def test_safe_source_rejects_invalid_roots_and_nonregular_files(tmp_path):
    from rapid_mlx.local_search.extraction import open_root, open_safe, stat_safe

    root = folder(tmp_path)
    file = root / "plain.txt"
    file.touch()
    with pytest.raises(ValueError, match="directory"):
        validate_folder(str(file))
    with pytest.raises(ValueError, match="absolute"), open_root(Path("relative")):
        pass
    nested = root / "nested"
    nested.mkdir()
    with pytest.raises(SkippedError, match="unsupported"):
        stat_safe(nested, root)
    fifo = root / "pipe.txt"
    os.mkfifo(fifo)
    with pytest.raises(SkippedError, match="unsupported"), open_safe(fifo, root, 100):
        pass
    with pytest.raises(OSError, match="changed"), open_safe(file, root, 100):
        file.write_text("changed while pinned")


@pytest.mark.parametrize("failure", ["scandir", "child", "directory_stat", "file_stat"])
def test_discovery_filesystem_races_are_counted(tmp_path, monkeypatch, failure):
    from rapid_mlx.local_search import extraction

    root = folder(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    file = root / "plain.txt"
    file.touch()
    original_stat, original_open = os.stat, os.open
    if failure == "scandir":
        monkeypatch.setattr(
            os, "scandir", lambda fd: (_ for _ in ()).throw(PermissionError())
        )
    elif failure == "child":
        calls = 0

        def opening(path, *args, **kwargs):
            nonlocal calls
            if path == "nested":
                calls += 1
                if calls == 2:
                    raise PermissionError()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", opening)
    else:

        def statting(path, *args, **kwargs):
            if kwargs.get("dir_fd") is not None and path == (
                "nested" if failure == "directory_stat" else "plain.txt"
            ):
                raise FileNotFoundError()
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", statting)
    batches = list(extraction.scan_batches(root))
    if failure != "child":
        assert sum(count for _, count in batches) == 1
    discovered = [p for files, _ in batches for p in files]
    assert discovered == ([] if failure in {"scandir", "file_stat"} else [file])


def test_read_safe_detects_growth_before_read(tmp_path, monkeypatch):
    from rapid_mlx.local_search import extraction

    root = folder(tmp_path)
    path = root / "growing.txt"
    path.write_text("small")
    original = extraction.open_safe

    from contextlib import contextmanager

    @contextmanager
    def growing(*args, **kwargs):
        with original(*args, **kwargs) as source:
            path.write_text("too much data")
            yield source

    monkeypatch.setattr(extraction, "open_safe", growing)
    with pytest.raises(SkippedError, match="size cap"):
        read_safe(path, root, 5)


@pytest.mark.parametrize("mode", ["seek", "read", "decoder"])
def test_video_decoder_failures_release_capture(tmp_path, monkeypatch, mode):
    cv2 = pytest.importorskip("cv2")

    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "bad.mp4"
    path.write_bytes(b"container")

    class Capture:
        released = False

        def isOpened(self):  # noqa: N802
            return True

        def get(self, key):
            return 1 if key == cv2.CAP_PROP_FPS else 10

        def set(self, key, value):
            return mode != "seek"

        def read(self):
            if mode == "decoder":
                raise cv2.error("decoder failure")
            return False, None

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(SkippedError, match="undecodable video"):
        list(media_frames(path, root))
    assert capture.released


def test_document_expansion_and_entity_bounds(monkeypatch):
    from types import SimpleNamespace

    import pypdf

    from rapid_mlx.local_search.extraction import _OfficeXML

    monkeypatch.setattr(
        pypdf,
        "PdfReader",
        lambda _: SimpleNamespace(
            is_encrypted=False,
            pages=[SimpleNamespace(extract_text=lambda: "long text")],
        ),
    )
    with pytest.raises(SkippedError, match="expanded size cap"):
        extract(Path("large.pdf"), b"", max_bytes=3)
    with pytest.raises(SkippedError, match="Audio is not supported yet"):
        extract(Path("sound.wav"), b"")

    class Archive:
        def infolist(self):
            return []

        def open(self, name):
            return io.BytesIO(
                b"<!DOCTYPE document><document/>"
                if name == "entities"
                else b"<document/>"
            )

    with pytest.raises(SkippedError, match="expanded size cap"):
        _OfficeXML(Archive(), 3).read("ordinary")
    with pytest.raises(SkippedError, match="XML entities"):
        _OfficeXML(Archive(), 100).read("entities")


@pytest.mark.parametrize(
    "platform,suffix,data",
    [
        ("linux", ".txt", b"hello"),
        ("darwin", ".wav", b""),
        ("linux", ".txt", b"oversize"),
        ("darwin", ".docx", b"bad"),
    ],
)
def test_parser_child_entrypoint_in_process(monkeypatch, platform, suffix, data):
    import json
    import resource
    from types import SimpleNamespace

    from rapid_mlx.local_search import parsing

    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", ["parser", suffix])
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(parsing, "MAX_BYTES", 5)
    if data == b"hello":
        import runpy

        monkeypatch.delitem(sys.modules, "rapid_mlx.local_search.parsing")
        runpy.run_module("rapid_mlx.local_search.parsing", run_name="__main__")
    else:
        parsing.main()
    message = json.loads(output.getvalue())
    if data == b"hello":
        assert message == {"units": [{"text": "hello", "page": None}]}
    else:
        assert message == {
            "error": "Too large"
            if data == b"oversize"
            else "Audio is not supported yet"
            if suffix == ".wav"
            else "Could not be read"
        }
    assert len(limits) == (3 if platform == "linux" else 1)


def test_store_schema_upgrade_rejects_unknown_and_normalizes_audio(tmp_path):
    from rapid_mlx.local_search.extraction import SKIP_REASONS

    root = folder(tmp_path)
    path = root / "sound.wav"
    path.touch()
    db = tmp_path / "index.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    store.replace(fid, path, path.stat(), "hash", [], None, "skipped: unsupported")
    store.db.execute("DELETE FROM meta WHERE key='extraction_version'")
    store.db.commit()
    store.close()
    store = Store(db, "model")
    assert store.files(fid)[str(path)]["status"] == "skipped: " + SKIP_REASONS["audio"]
    store.db.execute("UPDATE meta SET value='future' WHERE key='schema_version'")
    store.db.commit()
    store.close()
    with pytest.raises(ValueError, match="schema version"):
        Store(db, "model")


def test_store_dimension_budget_scope_and_unicode_keywords(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as storage

    monkeypatch.setattr(storage, "MAX_CORPUS_EMBEDDED_CHUNKS", 1)
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    try:
        fid = store.add_folder(root)[0]
        for n in range(2):
            path = root / f"{n}.txt"
            path.write_text("ß")
            store.replace(
                fid,
                path,
                path.stat(),
                str(n),
                [Chunk("ß", 1, None)],
                [[1, 0]],
                "indexed",
            )
        rows = store.rows([fid])
        assert len(rows) == 2 and len(store.vectors) == 1
        with pytest.raises(ValueError, match="dimensions"):
            store._put_vector(999, np.array([1, 0, 0]))
        assert store.semantic_background(dict(rows)) == store.semantic_background(
            dict(rows)
        )
        assert store.distinctive_terms(["ß"]) == []
        assert store.keyword(["ß"], set(rows), distinctive=["ß"]) == list(rows)
        assert store.keyword([], set(rows), query="", distinctive=[]) == []
    finally:
        store.close()


def test_fusion_limit_fills_remaining_unreserved_results():
    rows = {i: {"path": f"/{i}.txt", "kind": "text"} for i in range(4)}
    result = fuse([], list(rows), rows=rows, limit=2, keyword_evidence={0: 2, 1: 2})
    assert [r[0] for r in result] == list(rows)


@pytest.mark.parametrize("os_errno", [errno.EPERM, errno.EACCES])
def test_folder_route_scandir_permission_denied(tmp_path, monkeypatch, os_errno):
    root = folder(tmp_path).resolve(strict=True)
    assert root.is_dir()
    scandir = os.scandir
    probed = []

    def denied(path):
        if path == root:
            probed.append(path)
            raise PermissionError(os_errno, "access denied", str(path))
        return scandir(path)

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            monkeypatch.setattr(os, "scandir", denied)
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert probed == [root]
            assert response.status_code == 403
            assert response.json()["error"]["code"] == "folder_permission_denied"
            status = (await client.get("/v1/local-search/status")).json()
            assert status["folders"] == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", [PermissionError(), OSError(errno.EPERM, "storage denied")]
)
def test_folder_route_storage_permission_error_stays_generic(
    tmp_path, monkeypatch, failure
):
    root = folder(tmp_path)

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            validated = []

            def denied(path):
                validated.append(path)
                raise failure

            monkeypatch.setattr(app.state.local_search.store, "add_folder", denied)
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert validated == [root.resolve()]
            assert response.status_code == 400
            assert response.json()["error"] == {
                "message": "path must be an existing absolute directory without '..', outside excluded directories, other than /",
                "type": "invalid_request_error",
                "code": "invalid_folder_path",
                "param": "path",
            }
            assert app.state.local_search.store.folders() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("nested", [False, True])
def test_folder_route_excluded_unreadable_path_stays_invalid(
    tmp_path, monkeypatch, nested
):
    sensitive = folder(tmp_path, ".hidden")
    root = folder(sensitive, "documents") if nested else sensitive
    scandir = os.scandir
    probed = []

    def denied(path):
        if path == root.resolve():
            probed.append(path)
            raise PermissionError("folder denied")
        return scandir(path)

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            monkeypatch.setattr(os, "scandir", denied)
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == 400
            assert response.json()["error"] == {
                "message": "path is hidden, sensitive, or an excluded directory",
                "type": "invalid_request_error",
                "code": "invalid_folder_path",
                "param": "path",
            }
            assert probed == []
            assert app.state.local_search.store.folders() == []

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("entrypoint", ["cli", "server"])
def test_actual_local_search_flag_and_lifespan_wiring(
    tmp_path, monkeypatch, enabled, entrypoint
):
    """Execute production wiring in isolation from the inference runtime."""
    import ast
    from types import SimpleNamespace

    from rapid_mlx import local_search_runtime as runtime

    monkeypatch.setattr(runtime, "_enabled", False)
    monkeypatch.setattr(runtime, "_directory", None)
    source_dir = Path(__file__).resolve().parents[1] / "rapid_mlx"
    path = source_dir / f"{entrypoint}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    function_name = "serve_command" if entrypoint == "cli" else "main"
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    wiring = []
    for index, node in enumerate(function.body):
        if isinstance(node, ast.ImportFrom) and any(
            alias.asname == "configure_local_search" for alias in node.names
        ):
            wiring = function.body[index : index + 2]
            break
    assert wiring
    namespace = {
        "__package__": "rapid_mlx",
        "args": SimpleNamespace(
            enable_local_search=enabled, local_search_dir=str(tmp_path / "index")
        ),
    }
    exec(
        compile(ast.Module(body=wiring, type_ignores=[]), str(path), "exec"), namespace
    )
    assert runtime._enabled is enabled
    assert runtime._directory == str(tmp_path / "index")

    server_path = source_dir / "server.py"
    server_tree = ast.parse(server_path.read_text(), filename=str(server_path))
    lifespan = next(
        node
        for node in server_tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    calls = []
    for index, node in enumerate(lifespan.body):
        if isinstance(node, ast.ImportFrom) and node.module == "local_search_runtime":
            calls.extend(lifespan.body[index : index + 2])
    assert len(calls) == 4
    # Retain the actual source locations so coverage records the production lines.
    function = ast.AsyncFunctionDef(
        name="local_lifecycle",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="app")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=calls[:2] + [ast.Expr(value=ast.Yield(value=None))] + calls[2:],
        decorator_list=[],
    )
    rendered = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"__package__": "rapid_mlx"}
    exec(compile(rendered, str(server_path), "exec"), namespace)

    async def scenario():
        app = FastAPI()
        lifecycle = namespace["local_lifecycle"](app)
        await anext(lifecycle)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/local-search/status")
            assert response.status_code == (200 if enabled else 404)
        assert hasattr(app.state, "local_search") is enabled
        if enabled:
            assert (
                app.state.local_search.store.path
                == tmp_path / "index" / "index.sqlite3"
            )
        with pytest.raises(StopAsyncIteration):
            await anext(lifecycle)
        assert not hasattr(app.state, "local_search")

    asyncio.run(scenario())


@pytest.mark.parametrize("cached", [False, True])
def test_local_checkpoint_resolution_from_cached_config(tmp_path, monkeypatch, cached):
    from rapid_mlx import cli, model_metadata

    runnable_text_snapshot(tmp_path)

    monkeypatch.setattr(cli, "_cache_runnability", lambda _: True)
    monkeypatch.setattr(
        model_metadata, "resolve_unreferenced_cached_snapshot", lambda _: None
    )
    monkeypatch.setattr(
        model_metadata, "resolve_offline_cached_snapshot", lambda _: None
    )
    monkeypatch.setattr(
        model_metadata, "_cached_file", lambda *args: tmp_path / "config.json"
    )
    monkeypatch.setattr(
        model_metadata,
        "_complete_cached_checkpoint",
        lambda *args: tmp_path if cached else None,
    )
    assert local_model_path("test/model") == (str(tmp_path) if cached else None)


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_local_checkpoint_probe_failures_are_unavailable(monkeypatch, error):
    from rapid_mlx import cli

    def fail(_):
        raise error("cache unavailable")

    monkeypatch.setattr(cli, "_cache_runnability", fail)
    assert local_model_path("test/model") is None


def test_embedder_factory_enables_vision_without_loading(monkeypatch):
    import types

    from rapid_mlx.local_search import service as module

    calls = []
    fake = object()
    monkeypatch.setitem(
        sys.modules,
        "rapid_mlx.embedding",
        types.SimpleNamespace(
            EmbeddingEngine=lambda *args, **kwargs: calls.append((args, kwargs)) or fake
        ),
    )
    assert module.make_embedder("checkpoint") is fake
    assert calls == [
        (
            ("checkpoint",),
            {"max_length": "auto", "overflow_policy": "error", "vision": True},
        )
    ]


def test_resume_queues_registered_folders_and_ignores_stale_ids(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        (root / "apple.txt").write_text("An apple orchard grows delicious fruit.")
        registered, _ = await search.store.call("add_folder", root)
        try:
            search.pending.append("deleted-id")
            search.start()
            await search.wait_idle()
            assert search.store.folder(registered)["state"] == "idle"
            assert search.store.totals()["files"] == 1
            await search.queue("deleted-id")
            assert not search.pending
        finally:
            await search.close()

    asyncio.run(scenario())


def test_status_requeues_waiting_folder_when_model_returns(tmp_path, monkeypatch):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid, _ = await search.store.call("add_folder", root)
        await search.store.call("set_folder", fid, "waiting_for_model")
        try:
            await search.status_async()
            await search.wait_idle()
            assert search.store.folder(fid)["state"] == "idle"
        finally:
            await search.close()

    asyncio.run(scenario())


def test_generation_busy_falls_back_to_scheduler_statistics(tmp_path, monkeypatch):
    from rapid_mlx.config import get_config

    monkeypatch.setattr(
        get_config().engine, "get_stats", lambda: {"num_waiting": 1}, raising=False
    )
    search = engine(tmp_path)
    assert search._generation_busy()
    asyncio.run(search.close())


@pytest.mark.parametrize(
    "media,outcome",
    [
        (False, "none"),
        (False, "error"),
        (True, "none"),
        (True, "error"),
        (True, "changed"),
    ],
)
def test_index_handles_embedding_disappearance_and_media_changes(
    tmp_path, monkeypatch, media, outcome
):

    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        source = root / ("apple.png" if media else "apple.txt")
        if media:
            from PIL import Image

            Image.new("RGB", (4, 4)).save(source)
        else:
            source.write_text("An apple orchard grows delicious fruit.")

        async def embed(*args, **kwargs):
            if outcome == "error":
                search._owner_worker = search._worker()
                search.engine = FakeEmbedder()
                from rapid_mlx.local_search.service import ModelLoadError

                raise ModelLoadError("device unavailable")
            return None

        monkeypatch.setattr(search, "_embed_async", embed)
        if outcome == "changed":
            original = search._index_media

            async def changed(*args):
                result = await original(*args)
                source.write_bytes(b"changed")
                return result

            monkeypatch.setattr(search, "_index_media", changed)
        try:
            registered, _ = await search.add_folder(str(root))
            await search.wait_idle()
            row = search.store.files(registered["id"])[str(source)]
            if outcome == "changed":
                assert row["status"].startswith("skipped")
            else:
                assert row["status"] == "keyword"
                assert search.engine is None
                state = search.store.folder(registered["id"])
                assert state["state"] == "waiting_for_model"
                if outcome == "error":
                    assert state["error"] == "device unavailable"
        finally:
            await search.close()

    asyncio.run(scenario())


def test_document_parser_crash_marks_file_skipped(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service as module

    async def fail(*args):
        raise RuntimeError("parser failed")

    monkeypatch.setattr(module, "parse_document", fail)

    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        source = root / "apple.txt"
        source.write_text("An apple orchard grows delicious fruit.")
        try:
            registered, _ = await search.add_folder(str(root))
            await search.wait_idle()
            assert (
                search.store.files(registered["id"])[str(source)]["status"]
                == "skipped: " + module.SKIP_REASONS["read"]
            )
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["missing", "skipped", "unreadable"])
def test_disappearing_previously_indexed_file_is_removed(
    tmp_path, monkeypatch, failure
):
    from rapid_mlx.local_search import service as module

    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        source = root / "apple.txt"
        source.write_text("An apple orchard grows delicious fruit.")
        try:
            registered, _ = await search.add_folder(str(root))
            await search.wait_idle()
            source.write_text("An apple orchard grows delicious fruit again.")
            if failure == "skipped":

                async def parse(*args):
                    source.unlink()
                    raise SkippedError("extraction failed")

                monkeypatch.setattr(module, "parse_document", parse)
            else:

                def read(*args, **kwargs):
                    source.unlink()
                    raise (
                        FileNotFoundError("gone")
                        if failure == "missing"
                        else OSError("unreadable")
                    )

                monkeypatch.setattr(module, "read_safe", read)
            await search.queue(registered["id"])
            await search.wait_idle()
            assert not search.store.files(registered["id"])
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("absorbed", [False, True])
def test_worker_cancellation_marks_queued_or_continues_absorbed_root(
    tmp_path, monkeypatch, absorbed
):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid, _ = await search.store.call("add_folder", root)

        async def cancelled(registered):
            if absorbed:
                await search.store.call("delete_folder", fid)
            raise asyncio.CancelledError

        monkeypatch.setattr(search, "_index", cancelled)
        search.pending = [fid]
        try:
            if absorbed:
                await search._run()
                assert search.store.folder(fid) is None
            else:
                with pytest.raises(asyncio.CancelledError):
                    await search._run()
                assert search.store.folder(fid)["state"] == "queued"
            assert not search.progress["active"]
        finally:
            await search.close()

    asyncio.run(scenario())


def test_worker_records_unexpected_indexing_failure(tmp_path, monkeypatch):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid, _ = await search.store.call("add_folder", root)

        async def fail(_):
            raise RuntimeError("scan failed")

        monkeypatch.setattr(search, "_index", fail)
        search.pending = [fid]
        try:
            await search._run()
            assert search.store.folder(fid)["state"] == "error"
            assert search.store.folder(fid)["error"] == "scan failed"
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("media", [False, True])
def test_cancelled_cpu_discovery_drains_before_closing(tmp_path, monkeypatch, media):
    from rapid_mlx.local_search import service as module

    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Iterator:
        def __iter__(self):
            return self

        def __next__(self):
            started.set()
            assert release.wait(SYNC_TIMEOUT)
            if media:
                raise OSError("decoder failed during cancellation")
            return [], 0

        def close(self):
            assert release.is_set()
            closed.set()

    monkeypatch.setattr(
        module, "media_frames" if media else "scan_batches", lambda *args: Iterator()
    )

    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid, _ = await search.store.call("add_folder", root)
        try:
            task = asyncio.create_task(
                search._index_media(root / "apple.png", root, True, None)
                if media
                else search._index_pinned({"id": fid}, root, None)
            )
            assert await asyncio.to_thread(started.wait, SYNC_TIMEOUT)
            task.cancel()
            await loop_checkpoint()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
        finally:
            release.set()
            await search.close()

    asyncio.run(scenario())


def test_status_reports_disappeared_registered_folder(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        fid, _ = await search.store.call("add_folder", root)
        root.rmdir()
        try:
            status = await search.status_async()
            assert status["folders"][0]["id"] == fid
            assert status["folders"][0]["state"] == "missing"
        finally:
            await search.close()

    asyncio.run(scenario())


def test_zero_embedding_is_rejected_and_preserves_model(tmp_path):
    class ZeroEmbedder(FakeEmbedder):
        def embed(self, texts):
            return [[0.0, 0.0] for _ in texts]

    async def scenario():
        from rapid_mlx.local_search.service import EmbeddingInputError

        fake = ZeroEmbedder()
        search = engine(tmp_path, fake)
        search.worker_error = "previous load failure"
        try:
            with pytest.raises(EmbeddingInputError, match="zero embedding output"):
                await search._embed_async(["apple"])
            assert search.engine is fake
            assert search.worker_error is None
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode",
    [
        "spawn_cancel",
        "probe_gone",
        "probe_init_gone",
        "probe_init_failed",
        "invalid_output",
        "kill_race",
        "stdin_closed",
        "stdin_reset",
        "blocked_sender",
    ],
)
def test_parser_process_races_reap_child(monkeypatch, mode):
    import psutil

    from rapid_mlx.local_search import parsing

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        communicating = asyncio.Event()
        probed = asyncio.Event()

        class Process:
            pid = 123
            returncode = None
            killed = False
            reaped = False

            def write(self, data):
                if mode == "stdin_closed":
                    raise BrokenPipeError()

            async def drain(self):
                if mode == "stdin_reset":
                    raise ConnectionResetError()
                if mode == "blocked_sender":
                    await asyncio.Event().wait()

            def close(self):
                pass

            def abort(self):
                pass

            async def wait_closed(self):
                if mode == "stdin_closed":
                    raise BrokenPipeError()

            async def read(self, size):
                communicating.set()
                if self.returncode is not None:
                    return b""
                if mode == "probe_gone":
                    await probed.wait()
                if mode in {"kill_race", "blocked_sender"}:
                    await asyncio.Event().wait()
                self.returncode = 1 if mode == "probe_init_failed" else 0
                return b"invalid json" if mode == "invalid_output" else b'{"units":[]}'

            def kill(self):
                self.killed = True
                self.returncode = -9
                if mode == "kill_race":
                    raise ProcessLookupError()

            async def wait(self):
                self.reaped = True
                return self.returncode

        process = Process()
        process.stdin = process.stdout = process.transport = process

        async def spawn(*args, **kwargs):
            started.set()
            if mode == "spawn_cancel":
                await release.wait()
            return process

        class Probe:
            def memory_info(self):
                if mode in {"probe_gone", "invalid_output"}:
                    probed.set()
                    raise psutil.NoSuchProcess(process.pid)
                from types import SimpleNamespace

                return SimpleNamespace(rss=0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

        def make_probe(pid):
            if mode in {"probe_init_gone", "probe_init_failed"}:
                raise psutil.NoSuchProcess(pid)
            return Probe()

        monkeypatch.setattr(psutil, "Process", make_probe)
        task = asyncio.create_task(parsing._run_parser(["parser"], b""))
        await started.wait()
        if mode in {"spawn_cancel", "kill_race", "blocked_sender"}:
            if mode in {"kill_race", "blocked_sender"}:
                await communicating.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert process.killed
        elif mode in {"invalid_output", "probe_init_failed"}:
            with pytest.raises(SkippedError, match="Could not be read"):
                await task
        else:
            assert await task == []
        assert process.reaped

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [False, True])
def test_standalone_local_search_parser_flags_without_runtime(enabled):
    import argparse
    import ast

    path = Path(__file__).resolve().parents[1] / "rapid_mlx" / "server.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_parser"
    )
    additions = [
        node
        for node in function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value in ("--enable-local-search", "--local-search-dir")
    ]
    assert len(additions) == 2
    parser = argparse.ArgumentParser()
    exec(
        compile(ast.Module(body=additions, type_ignores=[]), str(path), "exec"),
        {"parser": parser},
    )
    args = parser.parse_args(
        ["--enable-local-search", "--local-search-dir", "/index"] if enabled else []
    )
    assert args.enable_local_search is enabled
    assert args.local_search_dir == ("/index" if enabled else None)


def test_scan_prunes_nested_virtual_environment(tmp_path):
    root = folder(tmp_path)
    nested = root / "tools"
    nested.mkdir()
    (nested / "pyvenv.cfg").touch()
    (nested / "ignored.txt").write_text("not indexed")
    assert scan(root) == ([], 1)


def test_slide_references_cannot_expand_cached_xml_without_bound():
    drawing = "http://schemas.openxmlformats.org/drawingml/2006/main"
    presentation = "http://schemas.openxmlformats.org/presentationml/2006/main"
    relationship = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    data = office_zip(
        {
            "ppt/presentation.xml": f'<p:presentation xmlns:p="{presentation}" xmlns:r="{relationship}"><p:sldIdLst>'
            + '<p:sldId r:id="s1"/>' * 100
            + "</p:sldIdLst></p:presentation>",
            "ppt/_rels/presentation.xml.rels": '<Relationships><Relationship Id="s1" Type="slide" Target="slides/slide1.xml"/></Relationships>',
            "ppt/slides/slide1.xml": f'<a:root xmlns:a="{drawing}"><a:p><a:t>'
            + "x" * 100
            + "</a:t></a:p></a:root>",
        }
    )
    with pytest.raises(SkippedError, match="expanded size cap"):
        extract(Path("repeated.pptx"), data, max_bytes=4000)


def test_unicode_short_keyword_discards_nonmatching_candidates(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    try:
        fid = store.add_folder(root)[0]
        for n, text in enumerate(["ß", "unrelated"]):
            path = root / f"{n}.txt"
            path.write_text(text)
            store.replace(
                fid, path, path.stat(), str(n), [Chunk(text, 1, None)], None, "keyword"
            )
        rows = store.rows()
        assert store.keyword(["ß"], set(rows), distinctive=["ß"]) == [next(iter(rows))]
    finally:
        store.close()


def test_nested_document_paragraphs_cannot_expand_text_without_bound():
    word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    data = office_zip(
        {
            "word/document.xml": f'<w:document xmlns:w="{word}">'
            + "<w:p>" * 20
            + "<w:t>"
            + "x" * 200
            + "</w:t>"
            + "</w:p>" * 20
            + "</w:document>"
        }
    )
    with pytest.raises(SkippedError, match="expanded size cap"):
        extract(Path("nested.docx"), data, max_bytes=1000)


@pytest.mark.parametrize("mode", ["failed_exit", "oversized_output"])
def test_parser_rejects_failure_or_unbounded_output(monkeypatch, mode):
    from rapid_mlx.local_search import parsing

    monkeypatch.setattr(parsing, "MAX_BYTES", 1)
    command = "raise SystemExit(1)" if mode == "failed_exit" else "print('x' * 9)"
    with pytest.raises(SkippedError, match="Could not be read"):
        asyncio.run(parsing.run_parser([sys.executable, "-c", command], b""))


@pytest.mark.parametrize("unsafe", ["removed_folder", "private_file"])
def test_search_omits_results_that_lose_folder_or_privacy_admission(
    tmp_path, monkeypatch, unsafe
):
    from rapid_mlx.local_search import service as module

    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        source = root / "apple.txt"
        source.write_text("An apple orchard grows delicious fruit.")
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            assert (await search.search("apple", 10, None))["results"]
            if unsafe == "private_file":
                monkeypatch.setattr(module, "excluded", lambda path: path == source)
            else:
                call = search.store.call

                async def current_folders(method, *args, **kwargs):
                    if method == "folders":
                        return []
                    return await call(method, *args, **kwargs)

                monkeypatch.setattr(search.store, "call", current_folders)
            assert not (await search.search("apple", 10, None))["results"]
        finally:
            await search.close()

    asyncio.run(scenario())


def test_cancelled_cpu_read_drains_pinned_work_before_returning():
    from rapid_mlx.local_search.service import cpu_call

    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def read():
        started.set()
        assert release.wait(SYNC_TIMEOUT)
        finished.set()
        return b"document"

    async def scenario():
        task = asyncio.create_task(cpu_call(read))
        try:
            assert await asyncio.to_thread(started.wait, SYNC_TIMEOUT)
            task.cancel()
            await loop_checkpoint()
            assert not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
        finally:
            release.set()

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["FROM chunks JOIN files", "row_number()"])
def test_search_metadata_uses_one_snapshot_during_index_commit(tmp_path, boundary):
    root = folder(tmp_path)
    path = root / "new.txt"
    path.write_text("new searchable content")
    reader = Store(tmp_path / "index.sqlite3", "test")
    fid = reader.add_folder(root)[0]
    writer = Store(reader.path, "test")
    committed = []

    def commit_between_reads(sql):
        if boundary in sql and not committed:
            committed.append(True)
            writer.replace(
                fid,
                path,
                path.stat(),
                None,
                [Chunk("new searchable content", "text", {})],
                None,
                "keyword",
            )
            writer.replace(
                fid,
                root / "skip.xyz",
                path.stat(),
                None,
                [],
                None,
                "skipped: Unsupported file type",
            )

    reader.db.set_trace_callback(commit_between_reads)
    try:
        assert reader.search_rows() == {}
        assert committed == [True]
        assert not reader.db.in_transaction
        reader._invalidate_search()
        assert len(reader.search_rows()) == 2
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize(
    "reason",
    ["Could not be read", "Password-protected", "Unsupported file type", "Too large"],
)
def test_reindex_retries_only_transient_read_skips(tmp_path, monkeypatch, reason):
    from rapid_mlx.local_search import service as module

    root = folder(tmp_path)
    path = root / "restored.txt"
    path.write_text("restoredreadability")
    original = path.stat()
    old = Store(tmp_path / "index" / "index.sqlite3", "test")
    fid = old.add_folder(root)[0]
    old.replace(fid, path, original, None, [], None, "skipped: " + reason)
    old.close()
    path.chmod(0o600)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (
        original.st_size,
        original.st_mtime_ns,
    )
    reads = []
    read = module.read_safe

    def recording_read(path, *args, **kwargs):
        reads.append(path)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(module, "read_safe", recording_read)

    async def scenario():
        search = LocalSearch(
            tmp_path / "index",
            model_id="test",
            resolve_model=lambda _: None,
            yield_seconds=0,
        )
        try:
            await search.queue(fid)
            await search.wait_idle()
            assert reads == ([path] if reason == "Could not be read" else [])
            hits = (await search.search("restoredreadability", 20, None))["results"]
            assert bool(hits) == (reason == "Could not be read")
            await search.queue(fid)
            await search.wait_idle()
            assert len(reads) == (1 if reason == "Could not be read" else 0)
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "name,query",
    [
        ("Straße.pdf", "strasse"),
        ("STRASSE.pdf", "straße"),
        ("Straße_100%.pdf", "strasse_100%"),
    ],
)
def test_indexed_unicode_filename_candidates_are_casefolded(tmp_path, name, query):
    root = folder(tmp_path)
    path = root / name
    path.write_text("unrelated content")
    store = Store(tmp_path / "index.sqlite3", "test")
    try:
        fid = store.add_folder(root)[0]
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk("unrelated content", "text", {})],
            None,
            "keyword",
        )
        rows = store.search_rows()
        allowed = set(rows)
        from rapid_mlx.local_search.search import query_terms

        assert store.keyword(query_terms(query), allowed, query=query) == list(rows)
        assert store.keyword(["missing"], allowed, query="missing") == []
    finally:
        store.close()


def test_absorption_waits_for_dequeued_root_claim(tmp_path, monkeypatch, caplog):
    async def scenario():
        parent = folder(tmp_path)
        child = folder(parent, "docs")
        (child / "a.txt").write_text("apple")
        (child / "b.txt").write_text("banana")
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        claimed, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        call = search.store.call

        class ObservedLock(asyncio.Lock):
            async def acquire(self):
                if asyncio.current_task().get_name() == "absorb":
                    waiting.set()
                return await super().acquire()

        search._coordination = ObservedLock()

        async def paused_call(name, *args, **kwargs):
            value = await call(name, *args, **kwargs)
            if (
                name == "folder"
                and asyncio.current_task() is search.task
                and not claimed.is_set()
            ):
                claimed.set()
                await release.wait()
            return value

        monkeypatch.setattr(search.store, "call", paused_call)
        absorbing = None
        try:
            old, _ = await call("add_folder", child)
            search.pending.append(old)
            search._start_worker()
            await asyncio.wait_for(claimed.wait(), SYNC_TIMEOUT)
            assert old not in search.pending
            assert search.progress["folder_id"] is None
            assert search._coordination.locked()
            absorbing = asyncio.create_task(
                search.add_folder(str(parent)), name="absorb"
            )
            await asyncio.wait_for(waiting.wait(), SYNC_TIMEOUT)
            assert not absorbing.done()
            release.set()
            new, created = await absorbing
            assert created
            await search.wait_idle()
            assert len(fake.calls) == 2
            assert not search.store.folder(old)
            assert all(
                row["folder_id"] == new["id"] for row in search.store.rows().values()
            )
            assert search.store.folder(new["id"])["state"] == "idle"
            assert "Local-search indexing failed" not in caplog.text
        finally:
            release.set()
            if absorbing is not None:
                await absorbing
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("grouped", [False, True])
def test_search_validates_candidates_in_one_off_loop_batch(
    tmp_path, monkeypatch, grouped
):
    from rapid_mlx.local_search import service

    async def scenario():
        root = folder(tmp_path)
        (root / "apple.txt").write_text("apple")
        (root / "apple.bin").write_bytes(b"unsupported")
        search = engine(tmp_path)
        entered, release = threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        checks, hops = [], []
        query = None
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            to_thread = asyncio.to_thread

            async def tracked_hop(func, *args, **kwargs):
                if func.__name__ == "<lambda>":
                    hops.append(func)
                return await to_thread(func, *args, **kwargs)

            def validate(original, name):
                def checked(*args, **kwargs):
                    assert threading.get_ident() != loop_thread
                    checks.append((name, threading.get_ident()))
                    entered.set()
                    assert release.wait(SYNC_TIMEOUT)
                    return original(*args, **kwargs)

                return checked

            monkeypatch.setattr(
                service, "open_safe", validate(service.open_safe, "open")
            )
            monkeypatch.setattr(
                service, "stat_safe", validate(service.stat_safe, "stat")
            )
            monkeypatch.setattr(service.asyncio, "to_thread", tracked_hop)
            query = asyncio.create_task(
                search.search("apple", 20, None, group_limit=5 if grouped else None)
            )
            assert await to_thread(entered.wait, SYNC_TIMEOUT)
            # The loop can run this coroutine while filesystem validation waits.
            assert not query.done()
            release.set()
            response = await query
            assert {hit["path"] for hit in response["results"]} == {
                str(root / "apple.txt"),
                str(root / "apple.bin"),
            }
            assert {name for name, _ in checks} == {"open", "stat"}
            assert len({thread for _, thread in checks}) == 1
            assert len(hops) == 1
            if grouped:
                assert response["groups"]
        finally:
            release.set()
            if query is not None:
                await query
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("remaining", [0, 3, 32])
def test_semantic_scoring_compacts_after_large_delete(tmp_path, monkeypatch, remaining):
    from rapid_mlx.local_search import store as storage

    monkeypatch.setattr(storage, "MAX_CORPUS_EMBEDDED_CHUNKS", 64)
    root = folder(tmp_path)
    path = root / "large.txt"
    path.write_text("apple")
    store = Store(tmp_path / "compact.sqlite3", "model")
    try:
        fid, _ = store.add_folder(root)
        # Keep survivors at the end so resetting only trailing slots cannot pass.
        store.replace(
            fid,
            path,
            path.stat(),
            "hash",
            [Chunk("apple", i, None) for i in range(64 - remaining)],
            [[1, 0, 0]] * (64 - remaining),
            "indexed",
        )
        for i in range(remaining):
            other = root / f"keep-{i}.txt"
            other.write_text("apple")
            store.replace(
                fid,
                other,
                other.stat(),
                "hash",
                [Chunk("apple", 1, None)],
                [[1, i % 2, 0]],
                "indexed",
            )
        before = store.semantic_scores([1, 0, 0], store.search_rows())
        file_id = store.files(fid)[str(path)]["id"]
        deleted = {i for i, row in store.rows().items() if row["file_id"] == file_id}
        vectors = {i: v.copy() for i, v in store.vectors.items() if i not in deleted}
        store.remove_file(file_id)
        assert store._used_slots <= 2 * remaining
        assert all(np.array_equal(store.vectors[i], v) for i, v in vectors.items())
        assert all(np.shares_memory(v, store._matrix) for v in store.vectors.values())
        shapes = []
        asarray = storage.np.asarray

        class MeasuredMatrix(np.ndarray):
            def __matmul__(self, other):
                shapes.append(self.shape)
                return super().__matmul__(other)

        def measured(value, *args, **kwargs):
            result = asarray(value, *args, **kwargs)
            return result.view(MeasuredMatrix) if value is store._matrix else result

        monkeypatch.setattr(storage.np, "asarray", measured)
        assert store.semantic_scores([1, 0, 0], store.search_rows()) == [
            hit for hit in before if hit[0] not in deleted
        ]
        assert shapes == ([] if remaining == 0 else [(store._used_slots, 3)])
        # Reuse the compacted arena and verify live scores remain stable.
        store.replace(
            fid,
            path,
            path.stat(),
            "hash",
            [Chunk("apple", 1, None)],
            [[1, 0, 0]],
            "indexed",
        )
        assert (
            len(store.semantic_scores([1, 0, 0], store.search_rows())) == remaining + 1
        )
    finally:
        store.close()


@pytest.mark.parametrize("migration", ["schema_version", "chunking_version"])
def test_schema_chunking_migration_preserves_unchanged_skips(tmp_path, migration):
    from rapid_mlx.local_search.extraction import SKIP_REASONS

    root = folder(tmp_path)
    db = tmp_path / "skips.sqlite3"
    store = Store(db, "model")
    fid, _ = store.add_folder(root)
    statuses = {}
    for name, reason in (
        ("unsupported.bin", "unsupported"),
        ("oversized.txt", "size"),
        ("encrypted.pdf", "password"),
    ):
        path = root / name
        path.write_bytes(b"unchanged")
        status = "skipped: " + SKIP_REASONS[reason]
        statuses[str(path)] = status
        store.replace(fid, path, path.stat(), None, [], None, status)
    store.sql("UPDATE meta SET value='1' WHERE key=?", (migration,))
    store.close()
    store = Store(db, "model")
    try:
        assert {
            path: row["status"] for path, row in store.files(fid).items()
        } == statuses
        assert store.totals()["name_only_files"] == 3
    finally:
        store.close()


def test_worker_queue_removed_while_waiting_for_coordination(tmp_path):
    async def scenario():
        search = engine(tmp_path)
        task = None
        try:
            search._coordination = ObservedLock()
            search.pending.append("removed-root")
            await search._coordination.acquire()
            task = asyncio.create_task(search._run())
            await asyncio.wait_for(search._coordination.waiting.wait(), SYNC_TIMEOUT)
            assert not task.done()
            search.pending.clear()
            search._coordination.release()
            await task
            assert search.progress == search._idle()
        finally:
            if search._coordination.locked():
                search._coordination.release()
            if task is not None:
                await task
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "size", [(8193, 8192), (16_385, 1), (0, 1), (float("nan"), 1), (1, float("inf"))]
)
def test_media_dimension_budget(size):
    from rapid_mlx.local_search import extraction

    with pytest.raises(SkippedError, match="size cap") as caught:
        extraction._check_media_dimensions(*size)
    assert extraction.skip_reason(str(caught.value)) == "Too large"
    extraction._check_media_dimensions(8192, 8192)


@pytest.mark.parametrize("failure", ["pixels", "dimension", "warning", "error", "seek"])
def test_image_resource_limits_precede_pixel_decode(tmp_path, monkeypatch, failure):
    import warnings

    from PIL import Image, ImageOps

    from rapid_mlx.local_search import extraction

    root = folder(tmp_path)
    path = root / "oversized.png"
    Image.new("RGB", (2, 2)).save(path)
    convert = []
    monkeypatch.setattr(ImageOps, "exif_transpose", lambda image: convert.append(image))
    if failure in {"warning", "error"}:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 3 if failure == "warning" else 1)
    elif failure == "pixels":
        monkeypatch.setattr(extraction, "MAX_MEDIA_PIXELS", 3)
    elif failure == "dimension":
        monkeypatch.setattr(extraction, "MAX_MEDIA_DIMENSION", 1)
    else:
        from PIL import PngImagePlugin

        original = PngImagePlugin.PngImageFile.seek

        def seek(image, frame):
            original(image, frame)
            image._size = (8193, 8192)

        monkeypatch.setattr(PngImagePlugin.PngImageFile, "seek", seek)
    with pytest.raises(SkippedError, match="size cap"):
        list(extraction.media_frames(path, root))
    assert not convert
    # The scoped filter must not leak into later image operations.
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        warnings.warn("outside decoder", Image.DecompressionBombWarning, stacklevel=2)
    assert len(observed) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "metadata_pixels",
        "metadata_dimension",
        "actual_pixels",
        "shape",
        "dtype",
        "bytes",
        "valid",
    ],
)
def test_video_resource_limits_with_fake_decoder(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    from rapid_mlx.local_search import extraction

    root = folder(tmp_path)
    path = root / "oversized.mp4"
    path.write_bytes(b"tiny synthetic container")
    calls = []
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    if mode == "shape":
        pixels = np.zeros((2, 2), dtype=np.uint8)
    elif mode == "dtype":
        pixels = pixels.astype(np.float32)
    elif mode == "bytes":
        pixels = SimpleNamespace(
            ndim=3, shape=(2, 2, 3), dtype=np.uint8, nbytes=48 * 1024 * 1024 + 1
        )
    monkeypatch.setattr(
        extraction, "MAX_MEDIA_PIXELS", 3 if mode == "actual_pixels" else 4
    )
    monkeypatch.setattr(extraction, "MAX_MEDIA_DIMENSION", 4)

    class Capture:
        def isOpened(self):  # noqa: N802
            return True

        def get(self, key):
            return {
                1: 3
                if mode == "metadata_pixels"
                else 5
                if mode == "metadata_dimension"
                else 2,
                2: 1 if mode == "actual_pixels" else 2,
                3: 1,
                4: 1,
            }[key]

        def set(self, *args):
            return True

        def read(self):
            calls.append("read")
            return True, pixels

        def release(self):
            calls.append("release")

    def convert(frame, code):
        calls.append("convert")
        return frame

    decoder = SimpleNamespace(
        VideoCapture=lambda _: Capture(),
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        CAP_PROP_FPS=3,
        CAP_PROP_FRAME_COUNT=4,
        CAP_PROP_POS_FRAMES=5,
        INTER_AREA=0,
        COLOR_BGR2RGB=0,
        error=RuntimeError,
        resize=lambda *args, **kwargs: pixels,
        cvtColor=convert,
    )
    monkeypatch.setitem(sys.modules, "cv2", decoder)
    if mode == "valid":
        frames = list(extraction.media_frames(path, root))
        assert len(frames) == 1 and frames[0].image.size == (2, 2)
    else:
        with pytest.raises(
            SkippedError,
            match="undecodable video" if mode in {"shape", "dtype"} else "size cap",
        ):
            list(extraction.media_frames(path, root))
        assert "convert" not in calls
        assert ("read" in calls) == (not mode.startswith("metadata"))
    assert calls[-1] == "release"


def test_nonfinite_video_duration_is_bounded():
    from rapid_mlx.local_search.extraction import sample_timestamps

    with pytest.raises(SkippedError, match="size cap"):
        sample_timestamps(100, 1e-320)


@pytest.mark.parametrize(
    "limit",
    [
        "MAX_ARCHIVE_ENTRIES",
        "MAX_COMPRESSION_RATIO",
        "MAX_XML_DEPTH",
        "MAX_XML_ELEMENTS",
    ],
)
def test_office_structure_budgets(monkeypatch, limit):
    from rapid_mlx.local_search import extraction

    word = extraction._WORD[1:-1]
    parts = {
        "word/document.xml": f'<w:document xmlns:w="{word}"><w:p><w:t>apple</w:t></w:p></w:document>',
        "unused.xml": "x" * 256,
    }
    monkeypatch.setattr(extraction, limit, 1)
    with pytest.raises(SkippedError, match="expanded size cap"):
        extract(Path("bounded.docx"), office_zip(parts))


@pytest.mark.parametrize("suffix", [".pptx", ".xlsx"])
def test_document_page_budget(monkeypatch, suffix):
    from rapid_mlx.local_search import extraction

    monkeypatch.setattr(extraction, "MAX_DOCUMENT_PAGES", 1)
    data = office_fixture(suffix)[0]
    with pytest.raises(SkippedError, match="expanded size cap"):
        extract(Path("bounded" + suffix), data)


def test_total_text_and_paragraph_separator_budget(monkeypatch):
    import xml.etree.ElementTree as ET

    from rapid_mlx.local_search import extraction

    with pytest.raises(SkippedError, match="size cap"):
        extract(Path("large.txt"), b"four", max_bytes=3)
    # A named HTML reference can expand to multiple Unicode codepoints.
    with pytest.raises(SkippedError, match="expanded size cap"):
        extract(Path("large.html"), b"&nGt;", max_bytes=5)
    root = ET.fromstring(
        '<root xmlns:w="' + extraction._WORD[1:-1] + '"><w:p/><w:p/></root>'
    )
    with pytest.raises(SkippedError, match="expanded size cap"):
        extraction._paragraphs(root, extraction._WORD, 1)
    with pytest.raises(SkippedError, match="expanded size cap"):
        extraction._join_bounded(["a", "b"], "\n", 2)


def test_chunk_budget_across_units_and_long_lines(monkeypatch):
    from rapid_mlx.local_search import chunking

    monkeypatch.setattr(chunking, "MAX_CHUNKS_PER_FILE", 2)
    assert len(chunk_units([Unit("x" * 16)], target=8, overlap=0)) == 2
    with pytest.raises(SkippedError, match="size cap"):
        chunk_units([Unit("x" * 24)], target=8, overlap=0)
    with pytest.raises(SkippedError, match="size cap"):
        chunk_units([Unit("a"), Unit("b"), Unit("c")])


def test_directory_depth_budget(tmp_path, monkeypatch):
    from rapid_mlx.local_search import extraction

    root = folder(tmp_path)
    nested = root / "nested"
    nested.mkdir()
    (nested / "file.txt").touch()
    (root / "root.txt").touch()
    monkeypatch.setattr(extraction, "MAX_DIRECTORY_DEPTH", 0)
    paths, skipped = scan(root)
    assert paths == [root / "root.txt"] and skipped == 1


@pytest.mark.parametrize("suffix", [".html", ".htm", ".pdf", ".docx", ".txt"])
def test_document_parser_isolation_selection(monkeypatch, suffix):
    from rapid_mlx.local_search import parsing

    parser = AsyncMock(return_value=[Unit("isolated")])
    monkeypatch.setattr(parsing, "run_parser", parser)
    result = asyncio.run(parsing.parse_document(Path("file" + suffix), b"plain"))
    assert result == [Unit("isolated" if suffix in {".pdf", ".docx"} else "plain")]
    assert parser.await_count == (1 if suffix in {".pdf", ".docx"} else 0)


def test_parser_output_cap_kills_child_without_waiting_for_exit(monkeypatch):
    import psutil

    from rapid_mlx.local_search import parsing

    children = []
    spawn = asyncio.create_subprocess_exec

    async def observed(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observed)
    monkeypatch.setattr(parsing, "MAX_BYTES", 8)
    command = "import sys,signal; sys.stdout.write('x'*65); sys.stdout.flush(); signal.pause()"
    with pytest.raises(SkippedError, match="Could not be read"):
        asyncio.run(
            asyncio.wait_for(
                parsing.run_parser([sys.executable, "-c", command], b"x" * 131_072),
                SYNC_TIMEOUT,
            )
        )
    assert children[0].returncode is not None
    assert not psutil.pid_exists(children[0].pid)


def test_oversized_decoded_image_remains_findable_by_name(tmp_path, monkeypatch):
    from PIL import Image

    from rapid_mlx.local_search import extraction

    monkeypatch.setattr(extraction, "MAX_MEDIA_PIXELS", 3)

    async def scenario():
        root = folder(tmp_path)
        path = root / "oversized_apple.png"
        Image.new("RGB", (2, 2)).save(path)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            hits = (await search.search("oversized apple", 10, None))["results"]
            assert len(hits) == 1 and hits[0]["path"] == str(path)
            assert hits[0]["content_indexed"] is False
            assert hits[0]["skip_reason"] == "Too large"
            assert not fake.image_calls
            assert search.status()["folders"][0]["state"] == "idle"
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "size", [(6000, 4000), (9600, 6400), (8192, 8192), (8193, 8192)]
)
def test_camera_image_pixel_budget_indexes_content(tmp_path, size):
    from PIL import Image

    from rapid_mlx.local_search.extraction import MAX_MEDIA_PIXELS

    assert Image.MAX_IMAGE_PIXELS > MAX_MEDIA_PIXELS
    accepted = size[0] * size[1] <= MAX_MEDIA_PIXELS

    async def scenario():
        root = folder(tmp_path)
        path = root / "camera_apple.png"
        with Image.new("RGB", size, "red") as image:
            image.save(path)
        fake = FakeEmbedder()
        search = engine(tmp_path, fake)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            hit = (await search.search("camera apple", 10, None))["results"][0]
            assert hit["content_indexed"] is accepted
            assert fake.image_calls == ([1] if accepted else [])
            if not accepted:
                assert hit["skip_reason"] == "Too large"
        finally:
            await search.close()

    asyncio.run(scenario())


def test_pdf_over_1000_pages_retains_text_and_page_locators():
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    page = PdfReader(io.BytesIO(small_pdf())).pages[0]
    for _ in range(1501):
        writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    from rapid_mlx.local_search.parsing import parse_document

    units = asyncio.run(parse_document(Path("large.pdf"), output.getvalue()))
    assert len(units) == 1501
    assert units[0] == Unit("First", 1)
    assert units[-1] == Unit("First", 1501)


@pytest.mark.parametrize(
    "size,seconds",
    [
        (0, 10),
        (1024, 10 + 1 / 256),
        (1024**2, 14),
        (20 * 1024**2, 90),
        (30 * 1024**2, 120),
    ],
)
def test_parser_size_deadline_computation(size, seconds):
    from rapid_mlx.local_search.parsing import parser_timeout_seconds

    assert parser_timeout_seconds(size) == seconds


def test_parser_wall_and_cpu_limits_share_size_deadline(monkeypatch):
    import json
    import resource
    from types import SimpleNamespace

    from rapid_mlx.local_search import parsing

    data = b" " * (1024 * 1024)
    sizes = []
    limits = []
    deadlines = []

    def budget(size):
        sizes.append(size)
        return 14.5

    async def wait_for(awaitable, timeout):
        deadlines.append(timeout)
        return await awaitable

    monkeypatch.setattr(parsing, "parser_timeout_seconds", budget)
    monkeypatch.setattr(parsing, "_run_parser", AsyncMock(return_value=[Unit("done")]))
    monkeypatch.setattr(asyncio, "wait_for", wait_for)
    assert asyncio.run(parsing.run_parser([], data)) == [Unit("done")]
    monkeypatch.setattr(resource, "setrlimit", lambda *args: limits.append(args))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    monkeypatch.setattr(sys, "argv", ["parser", ".txt"])
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    parsing.main()
    assert sizes == [len(data), len(data)]
    assert deadlines == [14.5]
    assert (resource.RLIMIT_CPU, (15, 15)) in limits
    assert "units" in json.loads(output.getvalue())


@pytest.mark.parametrize("size,accepted", [((3840, 2160), True), ((7680, 4320), False)])
def test_video_decoded_byte_budget_accepts_4k_skips_8k(
    tmp_path, monkeypatch, size, accepted
):
    from types import SimpleNamespace

    from rapid_mlx.local_search.extraction import media_frames

    root = folder(tmp_path)
    path = root / "camera.mp4"
    path.write_bytes(b"synthetic")
    pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    calls = []

    class Capture:
        def isOpened(self):  # noqa: N802
            return True

        def get(self, key):
            return {1: size[0], 2: size[1], 3: 1, 4: 1}[key]

        def set(self, *args):
            return True

        def read(self):
            return True, pixels

        def release(self):
            calls.append("release")

    def convert(frame, code):
        calls.append("convert")
        return frame

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda _: Capture(),
            CAP_PROP_FRAME_WIDTH=1,
            CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FPS=3,
            CAP_PROP_FRAME_COUNT=4,
            CAP_PROP_POS_FRAMES=5,
            INTER_AREA=0,
            COLOR_BGR2RGB=0,
            error=RuntimeError,
            resize=lambda *args, **kwargs: np.zeros((32, 32, 3), dtype=np.uint8),
            cvtColor=convert,
        ),
    )
    if accepted:
        frames = list(media_frames(path, root))
        assert frames[0].image.size == size
        assert calls == ["convert", "release"]
    else:
        with pytest.raises(SkippedError, match="size cap"):
            list(media_frames(path, root))
        assert calls == ["release"]


@pytest.mark.parametrize("precreated", [False, True])
def test_index_acl_repair_precedes_file_creation(tmp_path, monkeypatch, precreated):
    from rapid_mlx.local_search import store as module

    path = tmp_path / "index" / "index.sqlite3"
    path.parent.mkdir(mode=0o700)
    files = [Path(str(path) + suffix) for suffix in ("", "-wal", "-shm", "-journal")]
    if precreated:
        for file in files:
            file.touch(mode=0o600)
    acls = {path.parent, *files}
    probes, stripped = [], []
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_ancestor_acl_dangerous", lambda _: False)

    def target_for(fd):
        info = os.fstat(fd)
        return next(
            (
                target
                for target in (path.parent, *files)
                if target.exists()
                and (target.stat().st_dev, target.stat().st_ino)
                == (info.st_dev, info.st_ino)
            ),
            None,
        )

    def probe(fd):
        target = target_for(fd)
        probes.append(target)
        if target is not None and target != path.parent:
            assert path.parent not in acls, (
                "directory inheritance must be removed first"
            )
        return target in acls

    def strip(fd):
        target = target_for(fd)
        if target not in acls:
            return
        stripped.append(target)
        acls.remove(target)

    monkeypatch.setattr(module, "_acl_present", probe)
    monkeypatch.setattr(module, "_remove_acl", strip)
    connect = sqlite3.connect

    def protected_connect(*args, **kwargs):
        assert path.parent not in acls and path not in acls
        if precreated:
            assert not acls
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", protected_connect)
    store = Store(path, "test")
    try:
        assert stripped[0] == path.parent
        assert path in probes
        expected = [path.parent, *sorted(files)] if precreated else [path.parent, path]
        assert stripped == expected
    finally:
        store.close()


@pytest.mark.parametrize("target", ["directory", "file"])
def test_index_acl_foreign_owner_is_not_modified(tmp_path, monkeypatch, target):
    from rapid_mlx.local_search import store as module

    path = tmp_path / "index" / "index.sqlite3"
    path.parent.mkdir(mode=0o700)
    path.touch(mode=0o600)
    foreign = (path.parent if target == "directory" else path).stat().st_ino
    fstat = os.fstat

    def owner(fd):
        info = fstat(fd)
        if info.st_ino == foreign:
            values = list(info)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_ancestor_acl_dangerous", lambda _: False)
    monkeypatch.setattr(os, "fstat", owner)
    monkeypatch.setattr(module, "_acl_present", lambda _: False)

    def strip(fd):
        if fstat(fd).st_ino == foreign:
            pytest.fail("foreign ACL modified")

    monkeypatch.setattr(module, "_remove_acl", strip)
    with pytest.raises(PermissionError, match="owned by the current user"):
        Store(path, "test")


def test_index_refuses_nonsticky_replaceable_parent(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    with pytest.raises(PermissionError, match="replaced by another account"):
        Store(parent / "index" / "index.sqlite3", "test")


def test_index_acl_non_darwin_unchanged(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module, "_acl_present", lambda _: pytest.fail("ACL probe on Linux")
    )
    store = Store(tmp_path / "index.sqlite3", "test")
    store.close()


def test_index_acl_probe_is_disabled_off_darwin(monkeypatch):
    from rapid_mlx.local_search import store as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    assert not module._acl_present(-1)


class _FakeCFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


@pytest.mark.parametrize(
    "pointer,error,free_result,expected",
    [
        (None, "missing", 0, False),
        (None, "denied", 0, PermissionError),
        (1, None, 0, True),
        (1, None, 1, PermissionError),
    ],
)
def test_index_acl_fd_probe_outcomes(
    tmp_path, monkeypatch, pointer, error, free_result, expected
):
    import ctypes
    import errno

    from rapid_mlx.local_search import store as module

    def get_acl(_):
        ctypes.set_errno(errno.ENOENT if error == "missing" else errno.EACCES)
        return pointer

    class LibC:
        acl_get_fd = _FakeCFunction(get_acl)
        acl_free = _FakeCFunction(lambda _: free_result)

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: LibC())
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected, match="inspect local-search index ACL"):
                module._acl_present(fd)
        else:
            assert module._acl_present(fd) is expected
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "init_result,set_result,still_present,message",
    [
        (None, 0, False, "secure local-search index ACL"),
        (1, 1, False, "secure local-search index ACL"),
        (1, 0, True, "remove local-search index ACL"),
        (1, 0, False, None),
    ],
)
def test_index_acl_fd_removal_outcomes(
    tmp_path, monkeypatch, init_result, set_result, still_present, message
):
    import ctypes

    from rapid_mlx.local_search import store as module

    probes = iter((True, still_present))
    freed = []

    class LibC:
        acl_init = _FakeCFunction(lambda _: init_result)
        acl_set_fd = _FakeCFunction(lambda *_: set_result)
        acl_free = _FakeCFunction(lambda acl: freed.append(acl) or 0)

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_acl_present", lambda _: next(probes))
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: LibC())
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        if message:
            with pytest.raises(PermissionError, match=message):
                module._remove_acl(fd)
        else:
            module._remove_acl(fd)
        assert freed == ([] if init_result is None else [init_result])
    finally:
        os.close(fd)


def test_index_acl_absent_needs_no_removal(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_acl_present", lambda _: False)
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        module._remove_acl(fd)
    finally:
        os.close(fd)


@pytest.mark.skipif(sys.platform != "darwin", reason="real macOS extended ACLs")
@pytest.mark.parametrize(
    "entry,dangerous",
    [("everyone deny delete", False), ("everyone allow add_subdirectory", True)],
)
def test_index_real_darwin_ancestor_acl(tmp_path, entry, dangerous):
    ancestor = tmp_path / "home"
    ancestor.mkdir(mode=0o700)
    subprocess.run(["/bin/chmod", "+a", entry, str(ancestor)], check=True)
    try:
        path = ancestor / "index" / "index.sqlite3"
        if dangerous:
            with pytest.raises(
                PermissionError, match="replaced by another account"
            ) as error:
                Store(path, "test")
            assert str(ancestor) in str(error.value)
            assert "ACL" in str(error.value)
        else:
            store = Store(path, "test")
            try:
                assert (
                    store.db.execute(
                        "SELECT value FROM meta WHERE key='model_id'"
                    ).fetchone()[0]
                    == "test"
                )
            finally:
                store.close()
    finally:
        subprocess.run(["/bin/chmod", "-N", str(ancestor)], check=True)


def test_index_symlink_target_through_file_is_not_a_directory(tmp_path):
    base = tmp_path / "base"
    base.mkdir(mode=0o700)
    base.chmod(0o700)
    (base / "regular-file").write_text("x")
    link = base / "link"
    link.symlink_to("regular-file/..")
    with pytest.raises(NotADirectoryError, match="regular-file is not a directory"):
        Store(link / "index" / "index.sqlite3", "test")
    assert not (base / "index").exists()


def test_index_resolves_symlinked_component(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    path = link / "index" / "index.sqlite3"
    store = Store(path, "test")
    try:
        assert store.path == path.resolve()
        assert (real / "index" / "index.sqlite3").is_file()
        assert store.db.execute("PRAGMA database_list").fetchone()[2] == str(
            path.resolve()
        )
    finally:
        store.close()


@pytest.mark.parametrize("change", ["vanished", "symlink", "file"])
def test_index_stable_component_changed_after_resolution(tmp_path, monkeypatch, change):
    from rapid_mlx.local_search import store as module

    ancestor = tmp_path / "ancestor"
    directory = ancestor / "index"
    directory.mkdir(parents=True, mode=0o700)
    saved = tmp_path / "saved"
    original_open = os.open
    changed = False

    def swap(path, flags, *args, **kwargs):
        nonlocal changed
        if path == "ancestor" and kwargs.get("dir_fd") is not None and not changed:
            changed = True
            ancestor.rename(saved)
            if change == "symlink":
                ancestor.symlink_to(saved, target_is_directory=True)
            elif change == "file":
                ancestor.touch()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", swap)
    if change == "vanished":
        with pytest.raises(FileNotFoundError) as error:
            Store(directory / "index.sqlite3", "test")
        assert error.value.errno == errno.ENOENT
        return
    with pytest.raises(
        PermissionError, match="ancestor.*changed or is a symlink"
    ) as error:
        Store(directory / "index.sqlite3", "test")
    assert "[Errno" not in str(error.value)


@pytest.mark.parametrize(
    "error_number",
    [errno.EACCES, errno.EPERM, errno.ENOENT, errno.ENOTDIR, errno.ELOOP],
)
def test_index_stable_component_open_errors(tmp_path, monkeypatch, error_number):
    from rapid_mlx.local_search import store as module

    directory = tmp_path / "ancestor" / "index"
    directory.mkdir(parents=True, mode=0o700)
    original_open = module.os.open
    original_error = OSError(error_number, os.strerror(error_number), "ancestor")

    def fail_component(path, flags, *args, **kwargs):
        if path == "ancestor" and kwargs.get("dir_fd") is not None:
            raise original_error
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", fail_component)
    if error_number in {errno.ENOTDIR, errno.ELOOP}:
        with pytest.raises(
            PermissionError, match="ancestor.*changed or is a symlink"
        ) as error:
            Store(directory / "index.sqlite3", "test")
        assert error.value.__cause__ is original_error
    else:
        with pytest.raises(OSError) as error:
            Store(directory / "index.sqlite3", "test")
        assert error.value is original_error


def test_index_ancestor_acl_probe_disabled_off_darwin(monkeypatch):
    import ctypes

    from rapid_mlx.local_search import store as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        ctypes, "CDLL", lambda *a, **k: pytest.fail("ACL probe off darwin")
    )
    assert not module._ancestor_acl_dangerous(-1)


@pytest.mark.parametrize(
    "tags,failure,expected",
    [
        ([2, 2], None, False),
        ([1], None, True),
        ([2, 1], None, True),
        ([99], None, True),
        ([], None, False),
        ([], "missing", False),
        ([], "get_fd", PermissionError),
        ([2], "get_entry", PermissionError),
        ([2], "get_tag", PermissionError),
        ([2], "free", PermissionError),
    ],
)
def test_index_ancestor_acl_classifier(monkeypatch, tags, failure, expected):
    import ctypes
    import errno

    from rapid_mlx.local_search import store as module

    positions, freed = [], []

    def get_acl(fd):
        assert fd == 42
        if failure in {"missing", "get_fd"}:
            ctypes.set_errno(errno.ENOENT if failure == "missing" else errno.EACCES)
            return None
        return 123

    def get_entry(acl, position, entry):
        assert acl == 123
        assert position == (0 if not positions else -1)
        index = len(positions)
        positions.append(position)
        if failure == "get_entry" or index == len(tags):
            ctypes.set_errno(errno.EACCES if failure == "get_entry" else errno.EINVAL)
            return -1
        entry._obj.value = index + 1
        return 0

    def get_tag(entry, tag):
        if failure == "get_tag":
            return -1
        tag._obj.value = tags[entry.value - 1]
        return 0

    class LibC:
        acl_get_fd = _FakeCFunction(get_acl)
        acl_get_entry = _FakeCFunction(get_entry)
        acl_get_tag_type = _FakeCFunction(get_tag)
        acl_free = _FakeCFunction(
            lambda acl: freed.append(acl) or (1 if failure == "free" else 0)
        )

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: LibC())
    if expected is PermissionError:
        with pytest.raises(PermissionError, match="inspect local-search ancestor ACL"):
            module._ancestor_acl_dangerous(42)
    else:
        assert module._ancestor_acl_dangerous(42) is expected
    assert freed == ([] if failure in {"missing", "get_fd"} else [123])


@pytest.mark.parametrize(
    "unsafe", ["foreign_owner", "writable", "sticky_foreign_child"]
)
def test_index_ancestor_error_names_path_and_reason(tmp_path, monkeypatch, unsafe):
    from rapid_mlx.local_search import store as module

    ancestor = tmp_path / "ancestor"
    directory = ancestor / "index"
    directory.mkdir(parents=True, mode=0o700)
    # Set every mode explicitly: a umask-dependent group-writable default
    # would otherwise report the writable reason instead of the owner one.
    ancestor.chmod(
        {"foreign_owner": 0o755, "writable": 0o777, "sticky_foreign_child": 0o1777}[
            unsafe
        ]
    )
    target = (directory if unsafe == "sticky_foreign_child" else ancestor).stat().st_ino
    original_fstat = os.fstat

    def foreign_owner(fd):
        info = original_fstat(fd)
        if unsafe != "writable" and info.st_ino == target:
            values = list(info)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(module.os, "fstat", foreign_owner)
    with pytest.raises(PermissionError, match="replaced by another account") as error:
        Store(directory / "index.sqlite3", "test")
    assert str(ancestor) in str(error.value)
    reason = (
        "foreign owner"
        if unsafe == "foreign_owner"
        else "group- or world-writable without sticky ownership"
    )
    assert reason in str(error.value)


def test_index_worker_writer_uses_resolved_path(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as module

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    worker = module.StoreWorker(link / "index" / "index.sqlite3", "test")
    try:
        resolved = worker._store.path
        link.unlink()
        connect = sqlite3.connect
        opened = []

        def writer_connect(path):
            opened.append(path)
            return connect(path)

        monkeypatch.setattr(module.sqlite3, "connect", writer_connect)
        monkeypatch.setattr(module.Store, "_write_replace", lambda *a, **k: "written")
        assert worker._write_file((), {}, 0) == "written"
        assert opened == [resolved]
    finally:
        worker.close()


def test_index_refuses_acl_on_ancestor(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_ancestor_acl_dangerous", lambda _: True)
    with pytest.raises(PermissionError, match="replaced by another account"):
        Store(tmp_path / "index" / "index.sqlite3", "test")


def test_index_refuses_acl_remaining_on_file(tmp_path, monkeypatch):
    from rapid_mlx.local_search import store as module

    path = tmp_path / "index" / "index.sqlite3"
    path.parent.mkdir(mode=0o700)
    path.touch(mode=0o600)
    target = path.stat().st_ino
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module, "_ancestor_acl_dangerous", lambda _: False)
    monkeypatch.setattr(
        module, "_acl_present", lambda fd: os.fstat(fd).st_ino == target
    )
    monkeypatch.setattr(module, "_remove_acl", lambda _: None)
    with pytest.raises(PermissionError, match="Could not remove"):
        Store(path, "test")


@pytest.mark.skipif(sys.platform != "darwin", reason="real macOS extended ACLs")
@pytest.mark.parametrize("inherited", [False, True])
def test_index_removes_real_darwin_acls(tmp_path, monkeypatch, inherited):
    from rapid_mlx.local_search import store as module

    path = tmp_path / "index" / "index.sqlite3"
    path.parent.mkdir(mode=0o700)
    if inherited:
        # The index directory itself inherits from its parent. Its inheritable
        # grant would propagate to both O_CREAT and SQLite's new WAL/SHM files.
        path.parent.rmdir()
        subprocess.run(
            [
                "/bin/chmod",
                "+a",
                "everyone allow read,file_inherit,directory_inherit",
                str(tmp_path),
            ],
            check=True,
        )
        path.parent.mkdir(mode=0o700)
    else:
        path.touch(mode=0o600)
        for target in (path.parent, path):
            subprocess.run(
                ["/bin/chmod", "+a", "everyone allow read", str(target)], check=True
            )

    def acl_present(target):
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            return module._acl_present(fd)
        finally:
            os.close(fd)

    if inherited:
        with pytest.raises(PermissionError, match="replaced by another account"):
            Store(path, "test")
        subprocess.run(["/bin/chmod", "-N", str(tmp_path)], check=True)
        return
    assert acl_present(path)
    assert acl_present(path.parent)
    connect = sqlite3.connect

    def protected_connect(*args, **kwargs):
        assert not acl_present(path.parent)
        assert not acl_present(path)
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", protected_connect)
    store = Store(path, "test")
    try:
        for suffix in ("", "-wal", "-shm"):
            file = Path(str(path) + suffix)
            assert not acl_present(file)
            assert stat.S_IMODE(file.stat().st_mode) == 0o600
        store.db.execute("PRAGMA journal_mode=DELETE")
        store.db.execute("INSERT INTO meta VALUES ('private_text', 'secret')")
        assert not acl_present(Path(str(path) + "-journal"))
        store.db.rollback()
    finally:
        store.close()
        subprocess.run(["/bin/chmod", "-N", str(tmp_path)], check=True)


@pytest.mark.parametrize("method,route,body", FIRST_ROUTES)
def test_first_route_refuses_newer_schema(tmp_path, method, route, body):
    from rapid_mlx.routes.local_search import router

    directory = tmp_path / "index"
    store = Store(directory / "index.sqlite3", "test")
    with store.db:
        store.db.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    store.close()

    async def scenario():
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(router)
        search = LocalSearch(directory, resolve_model=lambda _: None)
        app.state.local_search = search
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://localhost"
            ) as client:
                payload = {"path": str(tmp_path)} if body == "folder" else body
                response = await asyncio.wait_for(
                    client.request(method, "/v1/local-search" + route, json=payload),
                    timeout=5,
                )
            assert response.status_code == 503
            assert response.json() == {
                "error": {
                    "message": (
                        "Local Search index schema version 999 is unsupported "
                        "(this application supports 3). Update Rapid-MLX "
                        "to open this index, or choose a new index directory."
                    ),
                    "type": "invalid_request_error",
                    "code": "unsupported_index_schema",
                    "param": "local_search_dir",
                }
            }
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "outage",
    [
        "missing",
        "dispatch_error",
        "changed_after_failure",
        "temporarily_missing_after_failure",
    ],
)
@pytest.mark.parametrize("media", [False, True])
def test_model_recovers_when_worker_returns(tmp_path, monkeypatch, outage, media):
    from types import SimpleNamespace

    from rapid_mlx.config import get_config

    async def scenario():
        root = folder(tmp_path)
        if media:
            from PIL import Image

            Image.new("RGB", (4, 4)).save(root / "apple.png")
        else:
            (root / "apple.txt").write_text("An apple orchard grows delicious fruit.")
        model = folder(tmp_path, "model")
        (model / "model.safetensors").write_bytes(b"unchanged checkpoint")
        cfg = get_config()
        original_worker = cfg.engine.execute_on_model_worker
        available = [False]
        fake = FakeEmbedder()

        def factory(_):
            if not available[0]:
                raise ValueError("persistent model failure")
            return fake

        async def dispatch_error(*args):
            if not available[0]:
                raise RuntimeError("model worker offline")
            return await original_worker(*args)

        if outage == "missing":
            monkeypatch.setattr(cfg, "engine", SimpleNamespace())
        elif outage == "dispatch_error":
            monkeypatch.setattr(
                cfg, "engine", SimpleNamespace(execute_on_model_worker=dispatch_error)
            )
        search = LocalSearch(
            tmp_path / "index",
            resolve_model=lambda _: str(model),
            embedder_factory=factory,
            yield_seconds=0,
        )
        index = AsyncMock(wraps=search._index)
        monkeypatch.setattr(search, "_index", index)
        try:
            fid = (await search.add_folder(str(root)))[0]["id"]
            await search.wait_idle()
            assert search.store.folder(fid)["state"] == "waiting_for_model"
            if outage in {"missing", "dispatch_error"}:
                assert fid not in search._attempted_checkpoints
            else:
                assert fid in search._attempted_checkpoints
                for _ in range(3):
                    await search.status_async()
                    await search.wait_idle()
                assert index.await_count == 1
                if outage == "temporarily_missing_after_failure":
                    monkeypatch.setattr(cfg, "engine", SimpleNamespace())
                    await search.status_async()
                    assert fid not in search._attempted_checkpoints
            available[0] = True

            async def replacement_worker(*args, **kwargs):
                return await original_worker(*args, **kwargs)

            worker = (
                replacement_worker
                if outage == "changed_after_failure"
                else dispatch_error
                if outage == "dispatch_error"
                else original_worker
            )
            monkeypatch.setattr(
                cfg, "engine", SimpleNamespace(execute_on_model_worker=worker)
            )
            await search.status_async()
            await search.wait_idle()
            assert index.await_count == 2
            assert search.store.folder(fid)["state"] == "idle"
            assert search.store.folder(fid)["error"] is None
            assert search.store.totals()["embedded_chunks"] > 0
            assert not search._attempted_checkpoints
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure,os_errno,status,code,message",
    [
        (
            failure,
            os_errno,
            403,
            "folder_permission_denied",
            "Rapid-MLX isn't allowed to read this folder. Allow access in System Settings › Privacy & Security, then try again.",
        )
        for failure, os_errno in (
            (PermissionError("denied"), None),
            (OSError("operation not permitted"), errno.EPERM),
            (OSError("access denied"), errno.EACCES),
        )
    ]
    + [
        (
            OSError("unavailable"),
            errno.EIO,
            400,
            "invalid_folder_path",
            "path must be an existing absolute directory without '..', outside excluded directories, other than /",
        )
    ],
)
def test_add_folder_os_errors(
    tmp_path, monkeypatch, failure, os_errno, status, code, message
):
    # OSError(errno, ...) normally constructs a PermissionError automatically.
    # Set errno separately to cover raw OSError from other filesystem APIs too.
    failure.errno = os_errno
    root = folder(tmp_path).resolve()
    scandir = os.scandir

    def denied(path):
        if path == root:
            raise failure
        return scandir(path)

    async def scenario():
        async with client_app(tmp_path, monkeypatch) as (client, app, _):
            monkeypatch.setattr(os, "scandir", denied)
            response = await client.post(
                "/v1/local-search/folders", json={"path": str(root)}
            )
            assert response.status_code == status
            assert response.json()["error"] == {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
                "param": "path",
            }
            assert app.state.local_search.store.folders() == []

    asyncio.run(scenario())


def brute_file_evidence(bodies, terms):
    import math

    wanted = {term.casefold() for term in terms}
    coverage = {
        file_id: {
            term for term in wanted if any(term in text.casefold() for text in texts)
        }
        for file_id, texts in bodies.items()
    }
    weights = {
        term: math.log(
            (len(bodies) + 1) / (1 + sum(term in c for c in coverage.values()))
        )
        for term in wanted
    }
    total = math.fsum(weights.values())
    return {
        file_id: math.fsum(weights[term] for term in covered) / total if total else 0.0
        for file_id, covered in coverage.items()
    }


def test_whole_file_evidence_discounts_repetition_and_absent_terms():
    bodies = {
        1: ["Project status", "Schedule in Zürich", "Schedule in Zürich"],
        2: ["Project status", "Schedule in Lima"],
        3: ["Project status", "Budget in Lima"],
    }
    support = brute_file_evidence(bodies, ["PROJECT", "Zürich", "schedule", "missing"])
    assert 0 < support[2] < support[1] < 1
    assert support[3] == 0
    assert brute_file_evidence({1: ["Project"], 2: ["Project"]}, ["Project"]) == {
        1: 0,
        2: 0,
    }
    assert brute_file_evidence({}, ["schedule"]) == {}
    assert brute_file_evidence(bodies, []) == dict.fromkeys(bodies, 0)
    bodies[1].append("Schedule in Zürich")
    assert (
        brute_file_evidence(bodies, ["PROJECT", "Zürich", "schedule", "missing"])
        == support
    )


def test_candidate_file_evidence_reads_other_passages_but_respects_scope(tmp_path):
    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    for name, texts in [
        ("first.txt", ["Shared opening", "Rare specification"]),
        ("second.txt", ["Shared opening", "General specification"]),
        ("excluded.txt", ["Rare specification"]),
    ]:
        path = root / name
        path.write_text("\n".join(texts))
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk(text, i + 1, None) for i, text in enumerate(texts)],
            None,
            "keyword",
        )
    rows = store.search_rows()
    candidates = [
        i
        for i, row in rows.items()
        if row["path"].endswith(("first.txt", "second.txt"))
        and store.chunk_text(i) == "Shared opening"
    ]
    support = store.query_evidence(candidates, rows, ["rare", "specification"])
    assert len(support) == 2
    assert support[candidates[0]] == 1
    assert support[candidates[1]] == 0
    assert store.query_evidence([], rows, ["rare"]) == {}
    media = {-1: {"file_id": 99, "kind": "image"}}
    assert store.query_evidence([-1], media, ["rare"]) == {-1: 0}
    store.close()


def test_file_support_reorders_semantic_hits_and_group_previews_only():
    from rapid_mlx.local_search.search import grouped_rankings

    rows = {i: {"path": str(i), "kind": "text"} for i in range(3)}
    kwargs = dict(
        rows=rows,
        semantic_legs=[[0, 1, 2]],
        semantic_scores={0: 0.84, 1: 0.80, 2: 0.78},
        semantic_standardized={0: 4, 1: 3, 2: 2},
        keyword_evidence={},
    )
    original = fuse([], [], **kwargs)
    support = {0: 0, 1: 1, 2: 0}
    ranked = fuse([], [], query_evidence=support, **kwargs)
    assert [i for i, _, _ in original] == [0, 1, 2]
    assert [i for i, _, _ in ranked] == [1, 0, 2]
    assert {i for i, _, _ in ranked} == {i for i, _, _ in original}
    assert [s for _, s, _ in ranked] == [s for _, s, _ in original]
    groups = grouped_rankings([], query_evidence=support, **kwargs)
    assert groups == [("text", ranked)]
    kwargs["keyword_evidence"] = {2: 2}
    literal = fuse([], [2], query_evidence=support, **kwargs)
    assert [i for i, _, _ in literal] == [2, 1, 0]
    assert fuse([], [], query_evidence=support, rows={}, semantic_standardized={}) == []


@pytest.mark.parametrize("variable_limit", [None, 4])
def test_indexed_query_evidence_matches_brute_force(tmp_path, variable_limit):
    import sqlite3

    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    texts = [
        ["SHARED Zürich 中文检索", 'quoted "term" a_%\\ ß Ä'],
        ["shared 中文检索", "xy rare"],
        ["shared unrelated", "nothing"],
        ["shared Zürich", "extra"],
        ["shared", "last"],
    ]
    for n, passages in enumerate(texts):
        path = root / f"rare-{n}.txt"
        path.write_text("\n".join(passages))
        store.replace(
            fid,
            path,
            path.stat(),
            None,
            [Chunk(text, i + 1, None) for i, text in enumerate(passages)],
            None,
            "keyword",
        )
    rows = store.search_rows()
    candidates = list(rows)
    bodies = {}
    for row in store.db.execute("SELECT file_id,text FROM chunks"):
        bodies.setdefault(row[0], []).append(row[1])
    if variable_limit and hasattr(store.db, "setlimit"):
        store.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, variable_limit)
    for terms in [
        ["ZÜRICH", "中文检索", "missing", "SHARED"],
        ["中文", "xy", "ß", "Ä", "%", "_", "\\"],
        ["quoted", '"term"', "a_%\\"],
        ["absent"],
        ["shared"],
        ["rare"],
    ]:
        expected = brute_file_evidence(bodies, terms)
        assert store.query_evidence(candidates, rows, terms) == {
            i: expected[rows[i]["file_id"]] for i in candidates
        }
    statements = []
    store.db.set_trace_callback(statements.append)
    assert store.query_evidence(candidates, rows, []) == {}
    assert statements == []
    store.close()


def test_file_signature_bound_deduplication_and_overflow(tmp_path, monkeypatch):
    import zlib

    from rapid_mlx.local_search import store as module

    root = folder(tmp_path)
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    monkeypatch.setattr(module, "MAX_FILE_SIGNATURE_BYTES", 8)
    for name, text in [
        ("repeat.txt", "aa bb " * 100),
        ("large.txt", "aa bb cc"),
        ("empty.txt", ""),
    ]:
        path = root / name
        path.write_text(text)
        store.replace(
            fid, path, path.stat(), None, [Chunk(text, 1, None)], None, "keyword"
        )
    rows = store.search_rows()
    signatures = dict(
        store.db.execute("SELECT file_id,vocabulary FROM file_signatures")
    )
    assert sum(blob is None for blob in signatures.values()) == 1
    assert {
        zlib.decompress(blob) for blob in signatures.values() if blob is not None
    } == {b"", b"aa\nbb"}
    support = store.query_evidence(list(rows), rows, ["aa"])
    assert support == {
        i: float(row["path"].endswith("repeat.txt")) for i, row in rows.items()
    }
    path = root / "repeat.txt"
    store.replace(
        fid, path, path.stat(), None, [Chunk("aa bb " * 100, 1, None)], None, "keyword"
    )
    assert (
        zlib.decompress(
            store.db.execute(
                "SELECT vocabulary FROM file_signatures JOIN files ON file_id=files.id WHERE path=?",
                (str(path),),
            ).fetchone()[0]
        )
        == b"aa\nbb"
    )
    store.delete_folder(fid)
    assert store.db.execute("SELECT count(*) FROM file_signatures").fetchone()[0] == 0
    store.close()


@pytest.mark.parametrize(
    "query,semantic,lexical",
    [
        ("rare", [], []),
        ("rare", [1], []),
        ("rare", [2], [3]),
        ("", [2], []),
        ("rare", [2], []),
    ],
)
def test_search_query_evidence_skip_path(
    tmp_path, monkeypatch, query, semantic, lexical
):
    from rapid_mlx.local_search import service as module

    async def scenario():
        search = engine(tmp_path)
        rows = {
            1: {"path": "image", "file_id": 1, "kind": "image", "folder_id": "missing"},
            2: {"path": "text", "file_id": 2, "kind": "text", "folder_id": "missing"},
            3: {"path": "text", "file_id": 2, "kind": "text", "folder_id": "missing"},
        }
        calls = []

        async def call(name, *args, **kwargs):
            calls.append(name)
            if name == "search_rows":
                return rows
            if name == "distinctive_terms":
                return args[0]
            if name == "keyword":
                return lexical
            if name == "semantic_scores":
                return [(i, 0.8) for i in semantic]
            if name == "semantic_background":
                return dict.fromkeys(module.KINDS, 0)
            if name == "query_evidence":
                return {}
            if name == "folders":
                return []
            raise AssertionError(name)

        try:
            await search.store.call("folders")
            monkeypatch.setattr(search.store, "call", call)
            monkeypatch.setattr(
                search,
                "_embed_async",
                AsyncMock(return_value=np.array([[1, 0]], dtype=np.float32)),
            )
            monkeypatch.setattr(
                module,
                "semantic_ranking",
                lambda scores, *args, **kwargs: [i for i, _ in scores],
            )
            monkeypatch.setattr(module, "fuse", lambda *args, **kwargs: [])
            await search.search(query, 10, None)
            assert ("query_evidence" in calls) == (
                bool(query) and semantic == [2] and not lexical
            )
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", ["1", "2", None])
def test_file_signature_migration_requires_version_and_preserves_stored_text(
    tmp_path, version
):
    from rapid_mlx.local_search.store import SCHEMA_VERSION, SchemaVersionError

    root = folder(tmp_path)
    path = root / "old.txt"
    path.write_text("early late 中文检索")
    db = tmp_path / "old.sqlite3"
    store = Store(db, "model")
    fid = store.add_folder(root)[0]
    store.replace(
        fid,
        path,
        path.stat(),
        "hash",
        [Chunk("early", 1, None), Chunk("late 中文检索", 2, 8)],
        [[1, 0, 0], [0, 1, 0]],
        "indexed",
    )
    before = [tuple(r) for r in store.db.execute("SELECT * FROM chunks")]
    files = [tuple(r) for r in store.db.execute("SELECT * FROM files")]
    store.db.execute("DROP TABLE file_signatures")
    if version:
        store.db.execute(
            "UPDATE meta SET value=? WHERE key='schema_version'", (version,)
        )
    else:
        store.db.execute("DELETE FROM meta WHERE key='schema_version'")
    store.db.commit()
    store.close()
    path.unlink()  # Signatures must be derived from stored text, not sources.
    if version is None:
        before_bytes = db.read_bytes()
        with pytest.raises(SchemaVersionError, match="newer or incompatible version"):
            Store(db, "model")
        assert db.read_bytes() == before_bytes
        return
    store = Store(db, "model")
    if version == "1":
        # Keep main's existing media/chunking upgrade for legacy schema 1.
        before = [(*row[:-1], None) for row in before]
        files = [(*row[:-1], "pending") for row in files]
        assert not store.vectors
    else:
        assert len(store.vectors) == 2
    assert [tuple(r) for r in store.db.execute("SELECT * FROM chunks")] == before
    assert [tuple(r) for r in store.db.execute("SELECT * FROM files")] == files
    assert (
        store.db.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
        == SCHEMA_VERSION
    )
    import zlib

    assert (
        zlib.decompress(
            store.db.execute("SELECT vocabulary FROM file_signatures").fetchone()[0]
        )
        == "early\nlate\n中文检索".encode()
    )
    rows = store.search_rows()
    statements = []
    store.db.set_trace_callback(statements.append)
    store.query_evidence(list(rows), rows, ["late", "中文"])
    assert len(statements) == 1  # one primary-key read per FILE, not per chunk/term
    assert "file_signatures WHERE file_id=" in statements[0]
    store.close()


@pytest.mark.parametrize("layout", ["future", "different_columns", "missing_meta"])
def test_future_schema_is_structured_error_on_all_routes(tmp_path, monkeypatch, layout):
    import sqlite3

    from rapid_mlx.routes.local_search import router

    async def scenario():
        directory = tmp_path / "future"
        directory.mkdir()
        db = directory / "index.sqlite3"
        connection = sqlite3.connect(db)
        if layout == "future":
            connection.execute(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO meta VALUES ('schema_version','999')")
        elif layout == "different_columns":
            connection.execute("CREATE TABLE meta (name TEXT, version TEXT)")
        else:
            connection.execute("CREATE TABLE future_metadata (version TEXT)")
        connection.commit()
        before = db.read_bytes()
        connection.close()
        search = LocalSearch(directory, resolve_model=lambda _: None)
        app = FastAPI()
        install_exception_handlers(app)
        app.include_router(router)
        app.state.local_search = search
        from rapid_mlx.middleware.auth import check_rate_limit, verify_api_key

        app.dependency_overrides[verify_api_key] = lambda: None
        app.dependency_overrides[check_rate_limit] = lambda: None
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                requests = [
                    ("GET", "/status", None),
                    ("POST", "/folders", {"path": str(folder(tmp_path))}),
                    ("DELETE", "/folders/missing", None),
                    ("POST", "/reindex", {}),
                    ("POST", "/search", {"query": "apple"}),
                ]
                for method, route, body in requests:
                    response = await client.request(
                        method, "/v1/local-search" + route, json=body
                    )
                    assert response.status_code == 503
                    error = response.json()["error"]
                    assert error["code"] == "unsupported_index_schema"
                    assert "Update Rapid-MLX" in error["message"]
            assert db.read_bytes() == before
        finally:
            await search.close()

    asyncio.run(scenario())


def test_failed_replacement_preserves_file_signature(tmp_path):
    root = folder(tmp_path)
    path = root / "rollback.txt"
    path.write_text("apple")
    store = Store(tmp_path / "index.sqlite3", "model")
    fid = store.add_folder(root)[0]
    try:
        store.replace(
            fid,
            path,
            path.stat(),
            "before",
            [Chunk("apple", 1, None)],
            [[1, 0]],
            "indexed",
        )
        before = [
            tuple(row) for row in store.db.execute("SELECT * FROM file_signatures")
        ]
        with pytest.raises(IndexError):
            store.replace(
                fid,
                path,
                path.stat(),
                "after",
                [Chunk("banana", 1, None)] * 2,
                [[0, 1]],
                "indexed",
            )
        assert [
            tuple(row) for row in store.db.execute("SELECT * FROM file_signatures")
        ] == before
        assert store.keyword(["apple"], set(store.rows()))
    finally:
        store.close()


@pytest.mark.parametrize("layout", ["different_columns", "missing_meta"])
def test_store_rejects_incompatible_metadata(tmp_path, layout):
    from rapid_mlx.local_search.store import SchemaVersionError

    path = tmp_path / "index.sqlite3"
    with sqlite3.connect(path) as connection:
        table = "meta" if layout == "different_columns" else "future_metadata"
        connection.execute(f"CREATE TABLE {table} (version TEXT)")
    before = path.read_bytes()
    with pytest.raises(SchemaVersionError, match="newer or incompatible version"):
        Store(path, "model")
    assert path.read_bytes() == before


def test_store_initializes_empty_database(tmp_path):
    from rapid_mlx.local_search.store import SCHEMA_VERSION

    path = tmp_path / "index.sqlite3"
    sqlite3.connect(path).close()
    store = Store(path, "model")
    try:
        assert (
            store.db.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
            == SCHEMA_VERSION
        )
    finally:
        store.close()


@pytest.mark.parametrize("table", ["folders", "files", "chunks"])
def test_store_rejects_missing_schema_version_with_data(tmp_path, table):
    from rapid_mlx.local_search.store import SchemaVersionError

    path = tmp_path / "index.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
        connection.execute(f"INSERT INTO {table} VALUES (1)")
    connection.close()
    before = path.read_bytes()
    with pytest.raises(SchemaVersionError) as error:
        Store(path, "model")
    assert str(error.value) == (
        "Local Search index was created by a newer or incompatible "
        "version. Update Rapid-MLX to open this index, or choose a "
        "new index directory."
    )
    assert path.read_bytes() == before
    assert not Path(str(path) + "-wal").exists()


@pytest.mark.parametrize("last_table", ["meta", "folders", "files", "chunks"])
def test_store_recovers_interrupted_first_initialization(tmp_path, last_table):
    from rapid_mlx.local_search.store import SCHEMA_VERSION

    # Reproduce the DDL prefix a crash can leave before the metadata transaction.
    reference = Store(tmp_path / "reference.sqlite3", "model")
    try:
        schemas = {
            row["name"]: row["sql"]
            for row in reference.db.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        reference.close()
    path = tmp_path / "interrupted.sqlite3"
    with sqlite3.connect(path) as connection:
        for table in ("meta", "folders", "files", "chunks"):
            connection.execute(schemas[table])
            if table == last_table:
                break
    connection.close()
    store = Store(path, "model")
    try:
        assert (
            store.db.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
            == SCHEMA_VERSION
        )
        for table in ("folders", "files", "chunks"):
            assert store.db.execute(f"SELECT 1 FROM {table}").fetchone() is None
    finally:
        store.close()


def test_store_preserves_database_corruption_error(tmp_path):
    path = tmp_path / "index.sqlite3"
    path.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
        Store(path, "model")


def test_store_preserves_metadata_probe_io_error(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE meta (key TEXT, value TEXT)")

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "SELECT value FROM meta WHERE key='schema_version'":
                raise sqlite3.OperationalError("disk I/O error")
            return super().execute(sql, *args, **kwargs)

    connect = sqlite3.connect
    connections = []

    def failing_connect(*args, **kwargs):
        connection = connect(*args, factory=FailingConnection, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            Store(path, "model")
        assert len(connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connections[0].execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "overflow",
        "corrupt",
        "over_cap",
        "truncated",
        "invalid_utf8",
        "trailing_data",
    ],
)
def test_search_damaged_file_signature_is_neutral(tmp_path, monkeypatch, damage):
    import zlib

    from rapid_mlx.local_search import service as service_module
    from rapid_mlx.local_search.store import MAX_FILE_SIGNATURE_BYTES

    blobs = {
        "overflow": None,
        "corrupt": b"invalid zlib data",
        "over_cap": zlib.compress(b"apple " * MAX_FILE_SIGNATURE_BYTES),
        "truncated": zlib.compress(b"apple")[:-1],
        "invalid_utf8": zlib.compress(b"apple\xff"),
        "trailing_data": zlib.compress(b"apple") + b"junk",
    }
    # Ensure semantic candidates request evidence even when keyword matches exist.
    monkeypatch.setattr(Store, "keyword", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        service_module,
        "semantic_ranking",
        lambda scores, *args, **kwargs: [i for i, _ in scores],
    )

    async def scenario():
        search = engine(tmp_path)
        root = folder(tmp_path)
        for name, text in (
            ("damaged.txt", "apple orchard"),
            ("valid.txt", "apple orchard"),
            ("other.txt", "orchard pear"),
        ):
            (root / name).write_text(text)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            rows = await search.store.call("search_rows")
            damaged_id = next(
                row["file_id"]
                for row in rows.values()
                if row["path"].endswith("damaged.txt")
            )
            if damage == "missing":
                await search.store.call(
                    "sql", "DELETE FROM file_signatures WHERE file_id=?", (damaged_id,)
                )
            else:
                await search.store.call(
                    "sql",
                    "UPDATE file_signatures SET vocabulary=? WHERE file_id=?",
                    (blobs[damage], damaged_id),
                )
            valid = [i for i, row in rows.items() if row["file_id"] != damaged_id]
            terms = ["apple", "orchard", "pear"]
            baseline = await search.store.call("query_evidence", valid, rows, terms)
            assert baseline == dict.fromkeys(valid, 0.5)
            support = await search.store.call("query_evidence", list(rows), rows, terms)
            assert {i: support[i] for i in valid} == baseline
            assert all(support[i] == 0.0 for i in rows if i not in valid)
            result = await search.search("apple", 10, None)
            assert {Path(hit["path"]).name for hit in result["results"]} == {
                "damaged.txt",
                "valid.txt",
                "other.txt",
            }
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("group_limit", [None, 1, 3])
def test_closest_only_after_empty_normal_results(tmp_path, monkeypatch, group_limit):
    async def scenario():
        root = folder(tmp_path)
        for n in range(8):
            (root / f"file{n}.txt").write_text(f"apple contract section {n}")
        search = engine(tmp_path)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()

            # A diffuse query: positive similarities, all below the index
            # background, with no query-specific outlier. Reverse input order
            # to prove fallback orders by similarity, rather than store order.
            def scores(_, rows):
                return [(i, 0.1 + 0.01 * n) for n, i in enumerate(rows)]

            monkeypatch.setattr(search.store._store, "semantic_scores", scores)
            monkeypatch.setattr(
                search.store._store,
                "semantic_background",
                lambda _: dict.fromkeys(("text", "image", "video"), 1.0),
            )
            rows = search.store.search_rows()
            expected = sorted(scores(None, rows), key=lambda item: -item[1])[:3]
            response = await search.search(
                "departing staff customer lists", 20, None, group_limit=group_limit
            )
            assert response["mode"] == "hybrid"
            assert [h["path"] for h in response["results"]] == [
                rows[i]["path"] for i, _ in expected
            ]
            assert [h["score"] for h in response["results"]] == [s for _, s in expected]
            assert all(h["matched"] == ["closest"] for h in response["results"])
            if group_limit is not None:
                assert response["groups"] == [
                    {
                        "kind": "text",
                        "results": response["results"][:group_limit],
                        "total": 3,
                    }
                ]
            limited = await search.search("departing staff customer lists", 1, None)
            assert limited["results"] == response["results"][:1]
            # All eight files contain the literal phrase: lexical tier 2,
            # ordered by keyword rank, with score (2 + 61 / (60 + rank)) / 4.
            new = await search.search("apple", 20, None, group_limit=group_limit)
            assert new["mode"] == "hybrid"
            assert [h["path"] for h in new["results"]] == [
                str(root / f"file{n}.txt") for n in range(8)
            ]
            assert [h["score"] for h in new["results"]] == pytest.approx(
                [(2 + 61 / (61 + n)) / 4 for n in range(8)]
            )
            assert all(h["matched"] == ["keyword"] for h in new["results"])
            assert new["results"] and all(
                "closest" not in h["matched"] for h in new["results"]
            )
            assert [h["snippet"] for h in new["results"]] == [
                f"apple contract section {n}" for n in range(8)
            ]
            assert all(h["highlights"] == [[0, 5]] for h in new["results"])
            if group_limit is not None:
                assert new["groups"] == [
                    {
                        "kind": "text",
                        "results": new["results"][:group_limit],
                        "total": 8,
                    }
                ]
            else:
                assert "groups" not in new
            # Lower background admits every semantic hit; rank follows the
            # descending fixture similarities (file7 through file0). A single
            # semantic vote is normalized by 2 * 1.05 / 61, then divided by 4.
            monkeypatch.setattr(
                search.store._store,
                "semantic_background",
                lambda _: dict.fromkeys(("text", "image", "video"), 0.01),
            )
            new = await search.search("paraphrase", 20, None, group_limit=group_limit)
            assert new["mode"] == "hybrid"
            assert [h["path"] for h in new["results"]] == [
                str(root / f"file{n}.txt") for n in reversed(range(8))
            ]
            assert [h["score"] for h in new["results"]] == pytest.approx(
                [
                    0.125,
                    (61 / 62 + 0.02) / 8.4,
                    (61 / 63 + 0.02) / 8.4,
                    *[61 / (60 + rank) / 8.4 for rank in range(4, 9)],
                ]
            )
            assert all(h["matched"] == ["semantic"] for h in new["results"])
            assert all("closest" not in h["matched"] for h in new["results"])
            assert [h["snippet"] for h in new["results"]] == [
                f"apple contract section {n}" for n in reversed(range(8))
            ]
            assert all(h["highlights"] == [] for h in new["results"])
            if group_limit is not None:
                assert new["groups"] == [
                    {
                        "kind": "text",
                        "results": new["results"][:group_limit],
                        "total": 8,
                    }
                ]
            else:
                assert "groups" not in new
        finally:
            await search.close()

    asyncio.run(scenario())


def test_closest_sorts_off_event_loop(tmp_path, monkeypatch):
    from rapid_mlx.local_search import service

    async def scenario():
        root = folder(tmp_path)
        (root / "file.txt").write_text("apple contract")
        search = engine(tmp_path)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            monkeypatch.setattr(service, "semantic_ranking", lambda *a, **kw: [])
            loop_thread = threading.current_thread()
            sort_threads = []

            def checked_sorted(iterable, *, key):
                def checked_key(item):
                    sort_threads.append(threading.current_thread())
                    assert threading.current_thread() is not loop_thread
                    return key(item)

                return sorted(iterable, key=checked_key)

            monkeypatch.setattr(service, "sorted", checked_sorted, raising=False)
            response = await search.search("paraphrase", 20, None)
            assert sort_threads
            assert [hit["matched"] for hit in response["results"]] == [["closest"]]
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("content_indexed", [True, False])
@pytest.mark.parametrize("limit", [1, 20])
def test_closest_stops_safety_checks_after_admitted_limit(
    tmp_path, monkeypatch, content_indexed, limit
):
    from rapid_mlx.local_search import service

    async def scenario():
        root = folder(tmp_path)
        for n in range(20):
            (root / f"file{n}.txt").write_text("apple contract")
        search = engine(tmp_path)
        try:
            await search.add_folder(str(root))
            await search.wait_idle()
            monkeypatch.setattr(service, "semantic_ranking", lambda *a, **kw: [])
            monkeypatch.setattr(
                search.store._store,
                "semantic_scores",
                lambda _, rows: [
                    (i, 1.0 - int(Path(row["path"]).stem.removeprefix("file")) * 0.01)
                    for i, row in rows.items()
                ],
            )
            search_rows = search.store._store.search_rows

            def rows(*args):
                return {
                    i: {**row, "content_indexed": content_indexed}
                    for i, row in search_rows(*args).items()
                }

            monkeypatch.setattr(search.store._store, "search_rows", rows)
            calls = {"open_safe": [], "stat_safe": []}

            def counted(name, original):
                def check(path, *args):
                    calls[name].append(path.name)
                    return original(path, *args)

                return check

            for name in calls:
                monkeypatch.setattr(
                    service, name, counted(name, getattr(service, name))
                )
            (root / "file0.txt").unlink()
            response = await search.search("paraphrase", limit, None, group_limit=1)
            count = min(limit, 3)
            assert [Path(hit["path"]).name for hit in response["results"]] == [
                f"file{n}.txt" for n in range(1, count + 1)
            ]
            assert all(hit["matched"] == ["closest"] for hit in response["results"])
            checked = "open_safe" if content_indexed else "stat_safe"
            assert calls[checked] == [f"file{n}.txt" for n in range(count + 1)]
            assert calls["stat_safe" if content_indexed else "open_safe"] == []
        finally:
            await search.close()

    asyncio.run(scenario())


def test_closest_respects_scope_safety_grouping_and_positive_scores(
    tmp_path, monkeypatch
):
    async def scenario():
        root = folder(tmp_path)
        other = folder(tmp_path, "other")
        for n in range(8):
            (root / f"file{n}.txt").write_text("apple contract\n" * 400)
        (other / "outside.txt").write_text("apple")
        search = engine(tmp_path)
        try:
            f, _ = await search.add_folder(str(root))
            await search.add_folder(str(other))
            await search.wait_idle()
            # Force admission empty to isolate downstream fallback boundaries,
            # including repeated chunks per file and stale normal candidates.
            monkeypatch.setattr(
                "rapid_mlx.local_search.service.semantic_ranking", lambda *a, **kw: []
            )

            def scores(_, rows):
                return [
                    (i, 0.8 - int(Path(row["path"]).stem.removeprefix("file")) * 0.1)
                    for i, row in rows.items()
                ]

            monkeypatch.setattr(search.store._store, "semantic_scores", scores)
            (root / "file0.txt").unlink()
            response = await search.search(
                "paraphrase", 20, [f["id"]], ["text"], group_limit=1
            )
            assert [Path(h["path"]).name for h in response["results"]] == [
                "file1.txt",
                "file2.txt",
                "file3.txt",
            ]
            assert "groups" not in response
            assert all(
                h["folder_id"] == f["id"] and h["kind"] == "text"
                for h in response["results"]
            )
            # Keyword ranking is nonempty, but only stale file0 matches.
            stale = await search.search("file0", 20, [f["id"]], ["text"])
            assert all(h["matched"] == ["closest"] for h in stale["results"])
            assert (
                await search.search(
                    "paraphrase", 20, [f["id"]], ["image"], group_limit=1
                )
            )["results"] == []
            assert (await search.search("paraphrase", 20, ["missing"]))["results"] == []
            monkeypatch.setattr(
                search.store._store,
                "semantic_scores",
                lambda _, rows: [(i, 0.0 if i % 2 else -0.1) for i in rows],
            )
            assert (await search.search("paraphrase", 20, [f["id"]], group_limit=1))[
                "results"
            ] == []
            search._embed = lambda _: None
            assert (await search.search("paraphrase", 20, [f["id"]]))["results"] == []
        finally:
            await search.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mode,owner,allowed",
    [
        (0o700, "self", True),
        (0o777, "self", False),
        (0o1777, "foreign", False),
        (0o1777, "self", True),
        # Root-owned links in sticky directories cannot be swapped by others.
        (0o1777, "root", True),
        (0o700, "foreign", False),
    ],
)
def test_index_symlink_provenance(tmp_path, monkeypatch, mode, owner, allowed):
    from rapid_mlx.local_search import store as module

    tmp_path.chmod(0o700)
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o700)
    parent.chmod(mode)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    real.chmod(0o700)
    link = parent / "link"
    link.symlink_to(real, target_is_directory=True)
    lstat = module.os.lstat

    def link_owner(path, *args, **kwargs):
        info = lstat(path, *args, **kwargs)
        if owner != "self" and Path(path) == link:
            values = list(info)
            values[4] = 0 if owner == "root" else os.geteuid() + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(module.os, "lstat", link_owner)
    path = link / "index" / "index.sqlite3"
    if allowed:
        store = Store(path, "test")
        try:
            assert store.path == real / "index" / "index.sqlite3"
        finally:
            store.close()
    else:
        with pytest.raises(
            PermissionError, match="replaced by another account"
        ) as error:
            Store(path, "test")
        assert str(link) in str(error.value)
        assert ("foreign symlink owner" if owner == "foreign" else "writable") in str(
            error.value
        )
        assert not (real / "index").exists()


@pytest.mark.parametrize("target_kind", ["relative", "absolute", "chain", "dotdot"])
def test_index_symlink_targets(tmp_path, monkeypatch, target_kind):
    tmp_path.chmod(0o700)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    real.chmod(0o700)
    link = tmp_path / "link"
    target = real if target_kind == "absolute" else Path("real")
    if target_kind == "chain":
        second = tmp_path / "second"
        second.symlink_to(target, target_is_directory=True)
        target = Path("second")
    elif target_kind == "dotdot":
        target = Path("real/../real")
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    store = Store(Path("link/index/index.sqlite3"), "test")
    try:
        assert store.path == real / "index" / "index.sqlite3"
        assert store.db.execute("PRAGMA database_list").fetchone()[2] == str(store.path)
    finally:
        store.close()


@pytest.mark.parametrize("loop", [True, False])
def test_index_symlink_hop_limit(tmp_path, loop):
    tmp_path.chmod(0o700)
    for number in range(41):
        target = f"link{number + 1}" if number < 40 or not loop else "link0"
        (tmp_path / f"link{number}").symlink_to(target, target_is_directory=True)
    with pytest.raises(PermissionError, match="symlink hop limit"):
        Store(tmp_path / "link0" / "index.sqlite3", "test")


@pytest.mark.parametrize("unsafe", ["ancestor", "parent_owner", "acl"])
def test_index_symlink_parent_safety(tmp_path, monkeypatch, unsafe):
    from rapid_mlx.local_search import store as module

    tmp_path.chmod(0o700)
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir(mode=0o700)
    ancestor.chmod(0o777 if unsafe == "ancestor" else 0o700)
    parent = ancestor / "parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    real.chmod(0o700)
    link = parent / "link"
    link.symlink_to(real, target_is_directory=True)
    inode = parent.stat().st_ino
    fstat = module.os.fstat

    def owner(fd):
        info = fstat(fd)
        if unsafe == "parent_owner" and info.st_ino == inode:
            values = list(info)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(module.os, "fstat", owner)
    if unsafe == "acl":
        monkeypatch.setattr(module.sys, "platform", "darwin")
        monkeypatch.setattr(
            module, "_ancestor_acl_dangerous", lambda fd: fstat(fd).st_ino == inode
        )
    with pytest.raises(PermissionError, match="replaced by another account") as error:
        Store(link / "index.sqlite3", "test")
    assert str(link) in str(error.value)
    assert {"ancestor": "writable", "parent_owner": "foreign owner", "acl": "ACL"}[
        unsafe
    ] in str(error.value)


@pytest.mark.skipif(sys.platform != "darwin", reason="stock macOS /tmp symlink")
def test_index_real_darwin_tmp_symlink():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rapid-mlx-link-", dir="/tmp") as directory:
        parent = Path(directory)
        parent.chmod(0o700)
        store = Store(parent / "index" / "index.sqlite3", "test")
        try:
            assert store.path == (parent / "index" / "index.sqlite3").resolve()
            assert store.path.is_file()
        finally:
            store.close()
