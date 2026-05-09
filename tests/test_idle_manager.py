# SPDX-License-Identifier: Apache-2.0
"""Tests for IdleManager: unload-on-idle and reload-on-demand."""

from __future__ import annotations

import asyncio

import pytest

from vllm_mlx.runtime.idle_manager import IdleManager


@pytest.mark.asyncio
async def test_disabled_when_timeout_zero():
    mgr = IdleManager()
    state = {"loaded": True}

    async def unload():
        state["loaded"] = False

    async def reload():
        state["loaded"] = True

    mgr.configure(
        timeout=0.0,
        reload_fn=reload,
        unload_fn=unload,
        is_loaded_fn=lambda: state["loaded"],
    )
    assert mgr.enabled is False
    mgr.start()
    assert mgr._loop_task is None


@pytest.mark.asyncio
async def test_unload_after_idle_then_reload():
    state = {"loaded": True, "unloads": 0, "reloads": 0}

    async def unload():
        state["unloads"] += 1
        state["loaded"] = False

    async def reload():
        state["reloads"] += 1
        state["loaded"] = True

    mgr = IdleManager()
    mgr.configure(
        timeout=0.5,
        reload_fn=reload,
        unload_fn=unload,
        is_loaded_fn=lambda: state["loaded"],
    )
    mgr.start()
    try:
        # Wait past timeout — background loop should call unload.
        await asyncio.sleep(1.5)
        assert state["loaded"] is False
        assert state["unloads"] == 1

        # Next request triggers reload.
        await mgr.ensure_loaded()
        assert state["loaded"] is True
        assert state["reloads"] == 1
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_inflight_blocks_unload():
    state = {"loaded": True, "unloads": 0}

    async def unload():
        state["unloads"] += 1
        state["loaded"] = False

    async def reload():
        state["loaded"] = True

    mgr = IdleManager()
    mgr.configure(
        timeout=0.3,
        reload_fn=reload,
        unload_fn=unload,
        is_loaded_fn=lambda: state["loaded"],
    )
    mgr.request_start()  # simulate active request
    mgr.start()
    try:
        await asyncio.sleep(1.0)
        assert state["unloads"] == 0  # blocked by inflight
        mgr.request_end()
        await asyncio.sleep(2.0)
        assert state["unloads"] == 1
    finally:
        await mgr.stop()


@pytest.mark.asyncio
async def test_ensure_loaded_noop_when_loaded():
    state = {"loaded": True, "reloads": 0}

    async def unload():
        state["loaded"] = False

    async def reload():
        state["reloads"] += 1
        state["loaded"] = True

    mgr = IdleManager()
    mgr.configure(
        timeout=60.0,
        reload_fn=reload,
        unload_fn=unload,
        is_loaded_fn=lambda: state["loaded"],
    )
    await mgr.ensure_loaded()
    assert state["reloads"] == 0


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_single_reload():
    state = {"loaded": False, "reloads": 0}

    async def unload():
        state["loaded"] = False

    async def reload():
        state["reloads"] += 1
        await asyncio.sleep(0.1)
        state["loaded"] = True

    mgr = IdleManager()
    mgr.configure(
        timeout=60.0,
        reload_fn=reload,
        unload_fn=unload,
        is_loaded_fn=lambda: state["loaded"],
    )
    await asyncio.gather(*(mgr.ensure_loaded() for _ in range(5)))
    assert state["reloads"] == 1
