# SPDX-License-Identifier: Apache-2.0
"""Tests for the engine supervisor.

Real ``rapid-mlx serve`` is never spawned here; the child is a short
Python script. That keeps the tests honest about process mechanics
(signals, pipe draining, exit codes) without needing MLX or a model.
"""

from __future__ import annotations

import sys

import httpx
import pytest

from rmlx_web import supervisor
from rmlx_web.supervisor import (
    AttachedEngine,
    ChildState,
    EngineSupervisor,
    SupervisorError,
    find_rapid_mlx_binary,
    pick_free_port,
)


class TestBinaryResolution:
    def test_explicit_path_wins(self):
        assert find_rapid_mlx_binary("/usr/bin/true") == "/usr/bin/true"

    def test_env_override_is_used(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_BIN", "/opt/custom/rapid-mlx")
        assert find_rapid_mlx_binary() == "/opt/custom/rapid-mlx"

    def test_missing_binary_raises_with_an_actionable_message(self, monkeypatch):
        monkeypatch.delenv("RAPID_MLX_BIN", raising=False)
        monkeypatch.setattr(supervisor.shutil, "which", lambda _: None)

        with pytest.raises(SupervisorError) as excinfo:
            find_rapid_mlx_binary()

        message = str(excinfo.value)
        assert "pip install rapid-mlx" in message
        assert "--rapid-mlx-bin" in message

    def test_non_executable_explicit_path_is_rejected(self, tmp_path):
        path = tmp_path / "not-executable"
        path.write_text("#!/bin/sh\n")
        path.chmod(0o644)

        with pytest.raises(SupervisorError):
            find_rapid_mlx_binary(str(path))


class TestPortAllocation:
    def test_returns_a_usable_port(self):
        port = pick_free_port()
        assert 1024 < port <= 65535

    def test_successive_calls_differ(self):
        # Not a guarantee the OS makes, but a same-port result twice in a
        # row would mean the socket is not actually being released.
        assert pick_free_port() != pick_free_port()


class TestSupervisorLifecycle:
    @pytest.mark.asyncio
    async def test_the_child_gets_enable_audio(self, monkeypatch):
        """Every child mounts ``/v1/audio/*``.

        The engine's gate short-circuits on this flag BEFORE it looks at the
        model, so a text server gets the lane too — which is what lets speech
        run beside the chat model instead of needing a switch to an audio
        alias (there is no such thing: audio is a lane, not a served model).
        The STT/TTS engines stay lazy, so the flag costs nothing until used.
        """
        captured: list[str] = []

        async def fake_exec(*argv, **kwargs):
            captured.extend(argv)
            raise OSError("stop here — argv is all this test needs")

        monkeypatch.setattr(supervisor.asyncio, "create_subprocess_exec", fake_exec)

        engine = EngineSupervisor(binary="rapid-mlx", api_key="k")
        with pytest.raises(SupervisorError):
            await engine.start("qwen3-4b")

        assert "--enable-audio" in captured
        # Before the caller's own --serve-arg values, so an operator can
        # still override anything that follows.
        assert captured.index("--enable-audio") > captured.index("qwen3-4b")

    @pytest.mark.asyncio
    async def test_the_child_gets_a_resident_memory_ceiling(self, monkeypatch):
        """Without it the engine reports ``memory_limit_bytes: 0``.

        That is the engine's spelling for "no ceiling", which leaves the
        sidebar's memory panel with a numerator and no denominator.
        """
        captured: list[str] = []

        async def fake_exec(*argv, **kwargs):
            captured.extend(argv)
            raise OSError("stop here — argv is all this test needs")

        monkeypatch.setattr(supervisor.asyncio, "create_subprocess_exec", fake_exec)

        engine = EngineSupervisor(binary="rapid-mlx", api_key="k")
        with pytest.raises(SupervisorError):
            await engine.start("qwen3-4b")

        limit = captured[captured.index("--resident-memory-limit-gb") + 1]
        assert int(limit) >= 4
        assert "--resident-model-idle-ttl" in captured

    @pytest.mark.asyncio
    async def test_the_mcp_config_path_is_read_at_every_spawn(self, monkeypatch):
        """``--mcp-config`` is read once by the engine, at spawn.

        So the value that matters is the one true at the MOMENT of the spawn:
        a snapshot taken when the supervisor was built would arm connectors
        the user switched off half an hour ago, and miss ones they just added.
        """
        captured: list[str] = []
        path: str | None = None

        async def fake_exec(*argv, **kwargs):
            captured.clear()
            captured.extend(argv)
            raise OSError("stop here — argv is all this test needs")

        monkeypatch.setattr(supervisor.asyncio, "create_subprocess_exec", fake_exec)

        engine = EngineSupervisor(
            binary="rapid-mlx", api_key="k", mcp_config_path=lambda: path
        )

        with pytest.raises(SupervisorError):
            await engine.start("qwen3-4b")
        # Off: the child gets no MCP subsystem at all, not merely zero
        # servers.
        assert "--mcp-config" not in captured

        path = "/tmp/mcp.json"
        with pytest.raises(SupervisorError):
            await engine.start("qwen3-4b")
        assert captured[captured.index("--mcp-config") + 1] == "/tmp/mcp.json"

    @pytest.mark.asyncio
    async def test_child_that_exits_immediately_reports_failure(self):
        # A child that never binds a port must be noticed by its exit,
        # not by waiting out the full readiness timeout — otherwise a bad
        # alias would look like a 15-minute hang.
        engine = EngineSupervisor(
            binary=sys.executable,
            api_key="k",
            ready_timeout_s=30.0,
        )

        with pytest.raises(SupervisorError) as excinfo:
            await _start_with_argv(
                engine,
                [sys.executable, "-c", "import sys; sys.exit(3)"],
            )

        assert "exited during startup" in str(excinfo.value)
        assert engine.status().state is ChildState.FAILED

    @pytest.mark.asyncio
    async def test_the_engines_own_error_line_becomes_the_detail(
        self,
    ):  # `start` runs as a detached task, so the exception is swallowed by
        # the caller and `/api/status` is the only thing the page sees. A
        # null detail there leaves it with nothing to show but "unknown
        # error" — while the child printed exactly what to do.
        engine = EngineSupervisor(
            binary=sys.executable, api_key="k", ready_timeout_s=30.0
        )

        script = (
            "import sys;"
            "print('  Alias: flux2-klein-4b');"
            "print('  Error: image generation requires the "
            "`rapid-mlx[image]` Python extra.', file=sys.stderr);"
            "print('  Model: Runpod/FLUX.2-klein-4B');"
            "sys.exit(2)"
        )
        with pytest.raises(SupervisorError):
            await _start_with_argv(engine, [sys.executable, "-c", script])

        detail = engine.status().detail
        assert detail is not None
        # The `Error:` line specifically, not the last few lines: the CLI
        # prints a page of banner text after it, and reporting the tail
        # buries the one line that says what to do.
        assert detail.startswith("image generation requires")
        assert "Model:" not in detail

    @pytest.mark.asyncio
    async def test_a_child_with_no_error_line_still_reports_something(self):
        engine = EngineSupervisor(
            binary=sys.executable, api_key="k", ready_timeout_s=30.0
        )

        with pytest.raises(SupervisorError):
            await _start_with_argv(
                engine, [sys.executable, "-c", "import sys; sys.exit(3)"]
            )

        # Falls back to the supervisor's own account rather than to None.
        assert "exited during startup" in (engine.status().detail or "")

    @pytest.mark.asyncio
    async def test_ready_timeout_is_reported_and_the_child_is_cleaned_up(self):
        engine = EngineSupervisor(
            binary=sys.executable,
            api_key="k",
            # Short deliberately: this test measures the timeout path, not
            # startup speed, and the child never becomes ready by design.
            ready_timeout_s=2.0,
        )

        with pytest.raises(SupervisorError) as excinfo:
            await _start_with_argv(
                engine,
                [sys.executable, "-c", "import time; time.sleep(60)"],
            )

        assert "did not become ready" in str(excinfo.value)
        assert engine.status().state is ChildState.FAILED
        # A half-started child still holds GPU memory; leaving it running
        # would make the next start contend for the device.
        assert engine._process is None

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_nothing_was_started(self):
        engine = EngineSupervisor(binary=sys.executable, api_key="k")
        await engine.stop()
        assert engine.status().state is ChildState.STOPPED

    @pytest.mark.asyncio
    async def test_output_tail_is_bounded(self):
        engine = EngineSupervisor(
            binary=sys.executable,
            api_key="k",
            ready_timeout_s=2.0,
        )

        with pytest.raises(SupervisorError):
            await _start_with_argv(
                engine,
                [
                    sys.executable,
                    "-c",
                    "import sys\n"
                    "for i in range(1000): print('line', i, flush=True)\n"
                    "import time; time.sleep(30)",
                ],
            )

        # A long-running server logs every request; an unbounded tail
        # would grow without limit for the life of the process.
        assert len(engine.status().recent_output) <= 200

    @pytest.mark.asyncio
    async def test_base_url_is_none_until_ready(self):
        engine = EngineSupervisor(binary=sys.executable, api_key="k")
        assert engine.base_url is None


async def _start_with_argv(engine: EngineSupervisor, argv: list[str]) -> None:
    """Drive ``_start_locked`` with a substitute command line.

    The supervisor builds its own argv from the alias; these tests need a
    child that is not `rapid-mlx serve`, so the binary and args are
    swapped for a Python one-liner. Everything after the spawn — the
    readiness poll, the output drain, the failure teardown — is the real
    code path.
    """
    engine._binary = argv[0]
    engine._serve_args = argv[1:]

    original_exec = supervisor.asyncio.create_subprocess_exec

    async def patched(*_ignored_argv, **kwargs):
        return await original_exec(*argv, **kwargs)

    supervisor.asyncio.create_subprocess_exec = patched
    try:
        async with engine._lock:
            await engine._start_locked("fake-alias")
    finally:
        supervisor.asyncio.create_subprocess_exec = original_exec


class TestAttachedEngine:
    def test_reports_ready_and_refuses_switching(self):
        engine = AttachedEngine("http://127.0.0.1:8000/", api_key="k")

        assert engine.base_url == "http://127.0.0.1:8000"
        assert engine.status().state is ChildState.READY
        # Owning nothing means switching is structurally impossible, not
        # merely unimplemented — callers check the flag rather than
        # discovering it from an exception.
        assert engine.can_switch is False

    @pytest.mark.asyncio
    async def test_start_raises(self):
        engine = AttachedEngine("http://127.0.0.1:8000")
        with pytest.raises(SupervisorError):
            await engine.start("anything")


class TestResidentMemoryCeiling:
    def test_is_eighty_percent_of_physical_ram(self, monkeypatch):
        # 16384 * 2097152 = 32 GiB -> floor(32 * 0.8) = 25
        monkeypatch.setattr(
            supervisor.os,
            "sysconf",
            lambda name: 16384 if name == "SC_PAGE_SIZE" else 2097152,
        )
        assert supervisor.resident_memory_ceiling_gb() == 25

    def test_floors_at_four_gib(self, monkeypatch):
        monkeypatch.setattr(supervisor.os, "sysconf", lambda name: 1024)
        assert supervisor.resident_memory_ceiling_gb() == 4

    def test_an_unreadable_probe_falls_back_rather_than_raising(self, monkeypatch):
        # A raise here would take down every model start.
        def boom(name):
            raise ValueError("no such configuration parameter")

        monkeypatch.setattr(supervisor.os, "sysconf", boom)
        assert supervisor.resident_memory_ceiling_gb() == 4


class TestResidencyLoad:
    """Hot ``POST /v1/models/load`` — what lets two models be usable at once."""

    def _engine(self, *, ready=True):
        engine = EngineSupervisor(binary="rapid-mlx", api_key="k")
        if ready:
            engine._state = ChildState.READY
            engine._port = 9999
            engine._model = "chat-model"
            engine._resident = ["chat-model"]
            engine._modalities = {"chat-model": "text"}
        return engine

    @pytest.mark.asyncio
    async def test_a_stopped_engine_cannot_be_hot_loaded(self):
        # There is no process to load into; the caller must respawn.
        engine = self._engine(ready=False)
        outcome, _ = await engine.residency_load("m", modality="text")
        assert outcome is supervisor.ResidencyOutcome.UNSUPPORTED

    @pytest.mark.asyncio
    async def test_an_image_model_joins_the_chat_model(self, monkeypatch):
        """The whole point: different groups coexist.

        Verified live 2026-08-28 — flux2-klein-4b rendered while
        qwen3.5-4b-4bit stayed resident and answered chat.
        """
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        outcome, _ = await engine.residency_load(
            "flux2-klein-4b",
            modality="image-gen",
            size_bytes=4_600_000_000,
            image_mode="generation",
        )

        assert outcome is supervisor.ResidencyOutcome.LOADED
        assert captured["url"].endswith("/v1/models/load")
        assert captured["json"]["image_mode"] == "generation"
        # No `replace_group`: only "assistant" is accepted on the wire, and
        # media groups are single-slot by derivation anyway.
        assert "replace_group" not in captured["json"]
        # Both usable, and the chat model is still the primary.
        assert engine.status().resident == ["chat-model", "flux2-klein-4b"]
        assert engine.status().model == "chat-model"

    @pytest.mark.asyncio
    async def test_a_second_text_model_replaces_the_first(self, monkeypatch):
        # text/vision share one single-slot `assistant` group, so this one
        # evicts rather than joins — and becomes the new primary.
        async def fake_post(self, url, **kwargs):
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        engine._resident = ["chat-model", "flux2-klein-4b"]
        engine._modalities = {"chat-model": "text", "flux2-klein-4b": "image-gen"}

        await engine.residency_load("other-chat", modality="text")

        # The image model is untouched; only the assistant slot changed.
        assert engine.status().resident == ["flux2-klein-4b", "other-chat"]
        assert engine.status().model == "other-chat"

    @pytest.mark.asyncio
    async def test_the_catalog_size_is_sent_rather_than_a_guess(self, monkeypatch):
        """The engine's own fallback regexes a parameter count out of the
        alias and sizes `embeddinggemma-300m-6bit` to zero. Passing a
        measured size is what stops a correct load being refused."""
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured.update(kwargs.get("json") or {})
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        await engine.residency_load("m", modality="text", size_bytes=2 * 1024**3)
        assert captured["estimated_size_gb"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_an_unknown_size_is_omitted_rather_than_sent_as_zero(
        self, monkeypatch
    ):
        # Zero would read as a real reservation of nothing.
        captured = {}

        async def fake_post(self, url, **kwargs):
            captured.update(kwargs.get("json") or {})
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        await engine.residency_load("m", modality="text", size_bytes=None)
        assert "estimated_size_gb" not in captured

    @pytest.mark.asyncio
    async def test_a_capacity_refusal_is_reported_with_the_engines_reason(
        self, monkeypatch
    ):
        # The 507 body names the models it would have had to evict — no
        # message composed here knows that.
        async def fake_post(self, url, **kwargs):
            return httpx.Response(
                507,
                json={
                    "detail": {
                        "error": {"message": "would exceed the 25 GiB ceiling"},
                        "replacement_projection": {},
                    }
                },
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        outcome, reason = await engine.residency_load("huge", modality="text")

        assert outcome is supervisor.ResidencyOutcome.REJECTED
        assert "25 GiB ceiling" in reason
        # Nothing was loaded, so the bookkeeping must not claim otherwise.
        assert engine.status().resident == ["chat-model"]

    @pytest.mark.asyncio
    async def test_an_older_engine_without_the_route_is_unsupported(self, monkeypatch):
        # The ordinary path on an older install, not a failure to report.
        async def fake_post(self, url, **kwargs):
            return httpx.Response(404, json={}, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        outcome, reason = await engine.residency_load("m", modality="text")
        assert outcome is supervisor.ResidencyOutcome.UNSUPPORTED
        assert reason is None

    @pytest.mark.asyncio
    async def test_a_transport_failure_falls_back_rather_than_raising(
        self, monkeypatch
    ):
        async def fake_post(self, url, **kwargs):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        engine = self._engine()
        outcome, _ = await engine.residency_load("m", modality="text")
        assert outcome is supervisor.ResidencyOutcome.UNSUPPORTED

    @pytest.mark.asyncio
    async def test_a_respawn_forgets_everything_that_was_resident(self, monkeypatch):
        # A new process holds only what it was spawned for. Carrying the old
        # list over would tell the page a model is ready that is not loaded.
        async def fake_exec(*argv, **kwargs):
            raise OSError("stop here")

        monkeypatch.setattr(supervisor.asyncio, "create_subprocess_exec", fake_exec)

        engine = self._engine()
        engine._resident = ["chat-model", "flux2-klein-4b"]
        with pytest.raises(SupervisorError):
            await engine.start("fresh-model")

        assert engine.status().resident == []
