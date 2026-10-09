# SPDX-License-Identifier: Apache-2.0
"""Characterization snapshot of ``rapid-mlx serve`` argument resolution.

``serve_command`` is a ~2,100-line function that turns parsed CLI flags into
process state: ``rapid_mlx.server`` module globals, the ``ServerConfig``
singleton, the ``SchedulerConfig`` and keyword arguments handed to
``load_model``, and the uvicorn launch. This test drives it end to end for a
matrix of representative flag sets, with model download, memory/disk probes,
model loading and uvicorn stubbed, and pins everything it resolves in a golden
file.

It exists so ``serve_command`` can be split into smaller functions safely: a
pure refactor must leave the snapshot byte-identical. A deliberate behavior
change regenerates it with ``UPDATE_SERVE_SNAPSHOT=1`` and the fixture diff
shows exactly what moved.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import io
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rapid_mlx import cli

SNAPSHOT = Path(__file__).parent / "fixtures" / "serve_command_snapshot.json"

MODEL = "qwen3.5-4b-8bit"

# name -> extra argv after ``serve <MODEL>``.
SCENARIOS: dict[str, list[str]] = {
    "defaults": [],
    "tool_and_reasoning_parsers": [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--reasoning-parser",
        "qwen3",
    ],
    "no_parsers": ["--no-tool-call-parser", "--no-reasoning-parser"],
    "request_limits": [
        "--max-tokens",
        "512",
        "--timeout",
        "45",
        "--max-request-bytes",
        "1048576",
        "--api-key",
        "characterization-test-key",
    ],
    "cache_tuning": [
        "--disable-prefix-cache",
        "--prefill-step-size",
        "1024",
        "--max-num-seqs",
        "4",
    ],
    "memory_and_lifecycle": [
        "--gpu-memory-utilization",
        "0.8",
        "--resident-memory-limit-gb",
        "12",
        "--lazy-load",
        "--idle-unload-seconds",
        "90",
    ],
    "chat_behavior": [
        "--no-thinking",
        "--pin-system-prompt",
        "--served-model-name",
        "my-model",
    ],
    "bind": ["--host", "0.0.0.0", "--port", "9123", "--log-level", "DEBUG"],
}

# name -> extra argv that ``serve_command`` must refuse before loading a model.
REJECTIONS: dict[str, list[str]] = {
    "gpu_memory_utilization_out_of_range": ["--gpu-memory-utilization", "1.5"],
    "negative_resident_memory_limit": ["--resident-memory-limit-gb", "-1"],
    "negative_resident_idle_ttl": ["--resident-model-idle-ttl", "-1"],
    "force_and_no_hybrid": ["--force-hybrid", "--no-hybrid"],
    "force_and_no_spec_decode": ["--force-spec-decode", "--no-spec-decode"],
    "force_and_no_harmony_streaming": [
        "--force-openai-harmony-streaming",
        "--no-openai-harmony-streaming",
    ],
}

_SIMPLE = (str, int, float, bool, type(None))

# Attributes that sample the host at runtime rather than reflect a flag.
_VOLATILE_ATTRS = frozenset({"_baseline_memory_bytes"})

# Environment fallbacks serve_command reads when the matching flag is unset.
_SERVE_ENV_FALLBACKS = (
    "RAPID_MLX_MAX_REQUEST_BYTES",
    "RAPID_MLX_SSE_KEEPALIVE_SECONDS",
    "RAPID_PYSAMPLE",
)


def _jsonable(value: Any, depth: int = 0) -> Any:
    if isinstance(value, _SIMPLE):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, Path):
        return str(value)
    if depth > 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(v, depth + 1) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _jsonable(getattr(value, f.name), depth + 1)
            for f in dataclasses.fields(value)
        }
    # Rapid-MLX objects (and the conftest ``SchedulerConfig`` stand-in) are
    # rendered by their attributes; anything else by type name only.
    if hasattr(value, "__dict__") and type(value).__module__.split(".")[0] in (
        "rapid_mlx",
        "conftest",
        "tests",
    ):
        return {
            "__type__": type(value).__name__,
            **{
                k: "<volatile>" if k in _VOLATILE_ATTRS else _jsonable(v, depth + 1)
                for k, v in sorted(vars(value).items())
                if not k.startswith("__")
            },
        }
    return f"<{type(value).__name__}>"


def _scalar_globals(module) -> dict[str, Any]:
    return {
        k: v
        for k, v in vars(module).items()
        if not k.startswith("__") and isinstance(v, _SIMPLE)
    }


def _container_reprs(module) -> dict[str, str]:
    # Only scalars are reset between scenarios, so record any in-place
    # mutation of a container global; it would also outlive this test.
    return {
        k: repr(v)
        for k, v in vars(module).items()
        if not k.startswith("__") and isinstance(v, (list, dict, set))
    }


def _container_snapshots(module) -> dict[str, tuple[Any, Any]]:
    """Keep each container's identity and shallow contents for restoration."""
    snapshots: dict[str, tuple[Any, Any]] = {}
    for key, value in vars(module).items():
        if key.startswith("__"):
            continue
        if isinstance(value, list):
            snapshots[key] = (value, list(value))
        elif isinstance(value, dict):
            snapshots[key] = (value, dict(value))
        elif isinstance(value, set):
            snapshots[key] = (value, set(value))
    return snapshots


def _restore_container(original: Any, snapshot: Any) -> None:
    original.clear()
    if isinstance(original, list):
        original.extend(snapshot)
    else:
        original.update(snapshot)


def _rapid_environment() -> dict[str, str]:
    """Return product-owned environment state without serializing host values."""
    return {key: value for key, value in os.environ.items() if key.startswith("RAPID_")}


# Earlier tests in the same process may leave ``rapid_mlx.server`` globals
# modified, so the import-time defaults come from a fresh interpreter. The
# sentinel keeps any import-time stdout from being parsed as the payload.
_DEFAULTS_SENTINEL = "SERVE_IMPORT_DEFAULTS="
_IMPORT_DEFAULTS = (
    "import json, rapid_mlx.server as s\n"
    f"print({_DEFAULTS_SENTINEL!r} + json.dumps({{'file': s.__file__, 'globals': "
    "{k: v for k, v in vars(s).items() if not k.startswith('__')"
    " and isinstance(v, (str, int, float, bool, type(None)))}}))"
)


@functools.cache
def _server_import_defaults() -> dict[str, Any]:
    from rapid_mlx import server

    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_DEFAULTS],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={k: v for k, v in os.environ.items() if k not in _SERVE_ENV_FALLBACKS},
    )
    line = next(
        line for line in proc.stdout.splitlines() if line.startswith(_DEFAULTS_SENTINEL)
    )
    payload = json.loads(line[len(_DEFAULTS_SENTINEL) :])
    assert payload["file"] == server.__file__, "subprocess imported another copy"
    return payload["globals"]


@contextlib.contextmanager
def _pristine_server_state() -> Iterator[None]:
    """Run from import-time ``rapid_mlx.server`` globals; restore afterwards.

    ``serve_command`` writes process globals. Starting every scenario from the
    import-time defaults keeps the snapshot independent of scenario and test
    order; restoring afterwards keeps this test from leaking into others.
    """
    from rapid_mlx import server
    from rapid_mlx.config import server_config as config_mod
    from rapid_mlx.runtime.model_registry import ModelRegistry

    saved = dict(vars(server))
    saved_containers = _container_snapshots(server)
    saved_config = config_mod._config
    saved_environ = dict(os.environ)
    for key, value in _server_import_defaults().items():
        setattr(server, key, value)
    # Not a scalar, so the defaults above miss it, and other tests register
    # models into it; the residency manager reads it into the config.
    server._model_registry = ModelRegistry()
    config_mod.reset_config()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)
        for original, snapshot in saved_containers.values():
            _restore_container(original, snapshot)
        for key in list(vars(server)):
            if key not in saved:
                delattr(server, key)
        for key, value in saved.items():
            setattr(server, key, value)
        config_mod._config = saved_config


@pytest.fixture
def stub_heavy_serve_deps(monkeypatch):
    """Stub the steps that would download, probe the host, or bind.

    Same stubs as ``tests/test_serve_listen_fd.py``: model download, disk and
    memory probes, the upgrade prompt, and the middleware installers that
    fail once another test has started ``rapid_mlx.server.app``.
    """
    from rapid_mlx import _version_check
    from rapid_mlx import server as server_mod
    from rapid_mlx.middleware import auth as auth_mod
    from rapid_mlx.middleware import request_logging as reqlog_mod
    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", lambda: False)
    monkeypatch.setattr(
        _version_check, "print_staleness_warning_if_any", lambda **_kwargs: None
    )
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda model: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_alias_min_memory", lambda *a, **kw: None)
    # Pin flag resolution, not whether this host has a matching optional
    # vision package installed.  An absent, matching, or stale mlx-vlm must
    # produce the same characterization snapshot.
    monkeypatch.setattr(mllm_mod, "require_mlx_vlm_or_exit", lambda *a, **kw: None)
    # Port resolution probes the host; pin it so a busy port can't leak in.
    monkeypatch.setattr(cli, "_port_collision_host", lambda host, port: None)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *a, **kw: None)
    # Identity, so the snapshot records the level serve_command resolved.
    monkeypatch.setattr(server_mod, "configure_logging", lambda level: level)
    monkeypatch.setattr(server_mod, "configure_cors", lambda *a, **kw: None)
    monkeypatch.setattr(auth_mod, "configure_rate_limiter", lambda *a, **kw: None)
    monkeypatch.setattr(
        reqlog_mod, "install_request_logging_middleware", lambda *a: None
    )
    # serve_command reads these as fallbacks for unset flags; a developer
    # shell or runner that sets one must not change the snapshot.
    for var in _SERVE_ENV_FALLBACKS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _parse(extra: list[str]):
    captured: list = []
    with (
        patch.object(sys, "argv", ["rapid-mlx", "serve", MODEL, *extra]),
        patch.object(cli, "serve_command", side_effect=captured.append),
    ):
        cli.main()
    return captured[0]


def _run_scenario(extra: list[str], monkeypatch) -> dict:
    from rapid_mlx import server
    from rapid_mlx.config import get_config

    args = _parse(extra)
    load_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        server, "load_model", lambda *a, **kw: load_calls.append((a, kw))
    )
    uvicorn_calls: list[dict] = []
    monkeypatch.setattr(
        cli,
        "_run_uvicorn",
        lambda app, args, level: uvicorn_calls.append(
            {"host": args.host, "port": args.port, "log_level": level}
        ),
    )
    monkeypatch.setattr(cli, "_hard_exit_after_serve", lambda: None)

    before = _scalar_globals(server)
    containers_before = _container_reprs(server)
    environ_before = _rapid_environment()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            cli.serve_command(args)
    except SystemExit as exc:
        assert not load_calls, "rejected before loading a model"
        # Only the final line: the refusal itself, not any startup banner.
        lines = [line for line in output.getvalue().splitlines() if line.strip()]
        return {"exit_code": exc.code, "message": lines[-1] if lines else ""}
    after = _scalar_globals(server)
    environ_after = _rapid_environment()

    assert len(load_calls) == 1
    (load_args, load_kwargs) = load_calls[0]
    return {
        "load_model": {"args": _jsonable(load_args), "kwargs": _jsonable(load_kwargs)},
        "server_globals_changed": {
            k: v for k, v in sorted(after.items()) if before.get(k, object()) != v
        },
        "server_config": _jsonable(get_config()),
        "server_containers_mutated": sorted(
            k
            for k, v in _container_reprs(server).items()
            if containers_before.get(k) != v
        ),
        "environment_keys_mutated": sorted(
            key
            for key in set(environ_before) | set(environ_after)
            if environ_before.get(key) != environ_after.get(key)
        ),
        "uvicorn": uvicorn_calls,
    }


def _snapshot(
    monkeypatch,
    *,
    accepted: dict[str, list[str]] = SCENARIOS,
    rejected: dict[str, list[str]] = REJECTIONS,
) -> str:
    result: dict[str, dict] = {"accepted": {}, "rejected": {}}
    for section, table in (("accepted", accepted), ("rejected", rejected)):
        for name, extra in table.items():
            with monkeypatch.context() as scenario_patch, _pristine_server_state():
                result[section][name] = _run_scenario(extra, scenario_patch)
    return json.dumps(result, indent=1, sort_keys=True) + "\n"


def test_serve_command_resolution_matches_snapshot(
    monkeypatch, stub_heavy_serve_deps, scheduler_config_stub
) -> None:
    actual = _snapshot(monkeypatch)
    reordered = _snapshot(
        monkeypatch,
        accepted=dict(reversed(SCENARIOS.items())),
        rejected=dict(reversed(REJECTIONS.items())),
    )
    assert reordered == actual, "snapshot depends on scenario execution order"
    if os.environ.get("UPDATE_SERVE_SNAPSHOT") == "1":
        SNAPSHOT.write_text(actual)
    assert actual == SNAPSHOT.read_text(), (
        "serve_command resolution changed. If intentional, regenerate with "
        "UPDATE_SERVE_SNAPSHOT=1 and review the fixture diff."
    )


def test_pristine_server_state_restores_process_state(monkeypatch) -> None:
    from rapid_mlx import server
    from rapid_mlx.config import ServerConfig, get_config
    from rapid_mlx.config import server_config as config_mod

    original_rejected = list(server._mcp_rejected)
    original_config = get_config()
    env_key = "RAPID_MLX_CHARACTERIZATION_STATE_TEST"
    monkeypatch.setenv(env_key, "before")

    with _pristine_server_state():
        server._mcp_rejected.append("temporary")
        os.environ[env_key] = "during"
        config_mod._config = ServerConfig(model_name="temporary")

    assert server._mcp_rejected == original_rejected
    assert os.environ[env_key] == "before"
    assert get_config() is original_config
