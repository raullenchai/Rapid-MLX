# SPDX-License-Identifier: Apache-2.0
"""Ticket 04 — ``--disk-stream`` / ``--disk-stream-cache-gb`` CLI wiring.

Ticket: ``.scratch/rapid-mlx-disk-stream/issues/04-cli-flags-wiring-test.md``.

Four things are pinned here, none of which need a model download:

1. ``--disk-stream`` / ``--disk-stream-cache-gb`` are registered on
   ``serve`` (opt-in, default off / 1.0) and reach ``server.load_model``
   as ``enable_disk_stream`` / ``disk_stream_cache_gb`` kwargs.
2. ``_check_alias_min_memory`` still fires when ``--disk-stream`` is set
   — the flag does not suppress it.
3. **Wiring-regression test** (AC #4): mirrors
   ``test_deepseek_v32_indexer_gate.py::test_install_fires_on_real_serve_import_path``'s
   subprocess pattern. Drives the REAL ``cli.main()`` -> ``bench_command``
   -> ``vllm_mlx.engine.batched._load_lazy_and_install_disk_stream`` ->
   ``vllm_mlx.utils.tokenizer.load_model_with_fallback(lazy=True)`` chain,
   with only ``mlx_lm.load`` and ``disk_stream_patch.install`` stubbed (no
   real weights), and asserts ``install`` was actually reached with the
   right arguments. Guards against the PR #967 bug class: an installer
   wired to a module the real boot path never touches.

   The two post-load fixups that hit the network by design
   (``_try_inject_mtp_post_load``'s ``mlx_lm.utils._download`` re-resolve,
   ``_neutralize_unbundled_template_types``'s ``tokenizer_config.json``
   probe) are additionally stubbed here, purely so this subprocess's fake
   ``does-not-exist/...`` model name doesn't have to round-trip the real
   Hub — real end-to-end coverage of those fixups against actual
   checkpoints lives in the ``@pytest.mark.slow`` tests in
   ``test_disk_stream_patch.py``, not here.
4. ``--disk-stream`` against an unregistered ``model_type`` fails at load
   time with ``disk_stream_patch``'s own clear
   ``UnsupportedModelTypeError`` message (not some unrelated downstream
   crash) — same subprocess harness, ``disk_stream_patch.install`` left
   UN-mocked so the real registry-miss path runs.
5. **Lazy-path post-load fixup parity** (blocking finding in
   review-notes.md's cli.py/server.py/engine.batched.py/utils.tokenizer.py
   review): ``load_model_with_fallback(..., lazy=True)`` used to skip
   every architecture-independent post-load tokenizer/generation-config
   fixup the non-lazy path always runs — most importantly
   ``augment_eos_token_ids_from_generation_config``, whose own docstring
   names Qwen3/Qwen2.5 as the exact scenario it exists to fix, and
   ``qwen2_moe`` is one of the two registered ``--disk-stream``
   architectures. A fast, in-process test monkeypatches all five fixups
   to record whether/how they fire and asserts they do, without a
   checkpoint download.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# 1. Flags reach server.load_model as enable_disk_stream/disk_stream_cache_gb
# ---------------------------------------------------------------------------


def _drive_serve_capturing_load_model_kwargs(argv_extra, monkeypatch):
    """Drive the REAL ``cli.main()`` for a ``serve`` invocation, stubbing
    every heavy/network boundary, and return the kwargs ``server.load_model``
    was called with plus the number of times ``_check_alias_min_memory``
    ran. Mirrors ``test_audio_route_registration_gate.py``'s proven
    ``cli.main()``-driving pattern (no bespoke argparse re-implementation).
    """
    from vllm_mlx import cli, server

    captured: dict = {}
    min_memory_calls: list = []

    def _stub_load_model(*_a, **kwargs):
        captured.update(kwargs)

    real_check_alias_min_memory = cli._check_alias_min_memory

    def _spy_check_alias_min_memory(user_typed):
        min_memory_calls.append(user_typed)
        return real_check_alias_min_memory(user_typed)

    monkeypatch.setattr(server, "load_model", _stub_load_model)
    monkeypatch.setattr(cli, "_check_alias_min_memory", _spy_check_alias_min_memory)
    monkeypatch.setattr(cli, "_run_uvicorn", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_port_preflight_or_die", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_resolve_audio_model_for_serve", lambda _n: None)
    monkeypatch.setattr("vllm_mlx.api.utils.is_mllm_model", lambda _n: False)
    monkeypatch.setattr("vllm_mlx.audio.probe.is_audio_model_alias", lambda _n: False)
    monkeypatch.setattr(
        "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(
        "vllm_mlx._version_check.print_staleness_warning_if_any", lambda: None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "mlx-community/Qwen3-7B-4bit",
            "--port",
            "0",
            *argv_extra,
        ],
    )
    try:
        cli.main()
    except SystemExit:
        # main() may exit at various points downstream of load_model — we
        # only care that load_model (our observation point) was reached.
        pass
    return captured, min_memory_calls


def test_disk_stream_flag_default_off_reaches_load_model_unset(monkeypatch):
    """Without --disk-stream, every existing invocation is unaffected:
    load_model sees enable_disk_stream=False, disk_stream_cache_gb=1.0."""
    captured, _ = _drive_serve_capturing_load_model_kwargs([], monkeypatch)
    assert captured, "server.load_model was never reached"
    assert captured.get("enable_disk_stream") is False
    assert captured.get("disk_stream_cache_gb") == 1.0


def test_disk_stream_flags_parsed_and_reach_load_model(monkeypatch):
    """--disk-stream --disk-stream-cache-gb 2.5 reaches load_model exactly."""
    captured, min_memory_calls = _drive_serve_capturing_load_model_kwargs(
        ["--disk-stream", "--disk-stream-cache-gb", "2.5"], monkeypatch
    )
    assert captured, "server.load_model was never reached"
    assert captured.get("enable_disk_stream") is True
    assert captured.get("disk_stream_cache_gb") == 2.5

    # AC #3: _check_alias_min_memory still ran even with --disk-stream set
    # (it is called unconditionally in serve_command, before any
    # disk-stream branching — not suppressed).
    assert min_memory_calls == ["mlx-community/Qwen3-7B-4bit"]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_disk_stream_cache_budget_rejects_non_positive_or_non_finite(value):
    from vllm_mlx.cli import positive_finite_float

    with pytest.raises(Exception, match="positive finite"):
        positive_finite_float(value)


# ---------------------------------------------------------------------------
# 2. _check_alias_min_memory's warning still fires (direct-call unit test,
#    per the ticket: "doesn't need --disk-stream wired end-to-end").
# ---------------------------------------------------------------------------


def test_check_alias_min_memory_warns_regardless_of_disk_stream(monkeypatch, capsys):
    """_check_alias_min_memory has no disk-stream awareness at all — it is
    called unconditionally in serve_command (see test above) and its warn
    logic is untouched by this ticket. Pin the warning still fires for an
    alias whose declared floor exceeds this (mocked) machine's RAM."""
    import types

    from vllm_mlx import cli

    fake_profile = types.SimpleNamespace(min_memory_gb=192.0)
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile", lambda _name: fake_profile
    )

    class _FakeVirtualMemory:
        total = 64 * (1024**3)  # 64 GB — well under the 192 GB floor

    monkeypatch.setattr("psutil.virtual_memory", lambda: _FakeVirtualMemory())

    cli._check_alias_min_memory("hy3-preview-4bit")

    out = capsys.readouterr().out
    assert "192" in out
    assert "64.0" in out or "64" in out


# ---------------------------------------------------------------------------
# 3 & 4. Subprocess harness — real bench_command load path, no model
#    download. Shared preamble; each test overrides model_type / whether
#    disk_stream_patch.install is mocked.
# ---------------------------------------------------------------------------

_SUBPROCESS_PREAMBLE = """
import sys
from unittest import mock

# Import the real serve-path modules FIRST — never `disk_stream_patch`
# directly. Unlike the DeepseekV32 indexer gate this patch has no
# import-time side effect, but importing it here only to monkeypatch
# `install` (never calling it ourselves) keeps this test faithful to the
# same discipline: the assertion must be about what the REAL boot path
# (cli -> bench_command -> engine.batched._load_lazy_and_install_disk_stream
# -> utils.tokenizer.load_model_with_fallback -> mlx_lm.load) does, not
# about what this test script calls directly.
from pathlib import Path

import vllm_mlx.utils.tokenizer  # noqa: F401
import vllm_mlx.cli as cli
import vllm_mlx.disk_stream_patch as disk_stream_patch
import mlx_lm


class _Stop(Exception):
    pass


calls = []
_ORIG_INSTALL = disk_stream_patch.install


def _fake_mlx_lm_load(path_or_hf_repo, tokenizer_config=None, model_config=None,
                       adapter_path=None, lazy=False, return_config=False,
                       revision=None):
    if not lazy:
        print("FAIL: mlx_lm.load called with lazy=False on --disk-stream path")
        sys.exit(1)
    model = object()
    tokenizer = object()
    config = {"model_type": %(model_type)r}
    if return_config:
        return model, tokenizer, config
    return model, tokenizer


mlx_lm.load = _fake_mlx_lm_load

%(install_patch)s

with (
    mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
    mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
    mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
    mock.patch(
        "vllm_mlx.utils.tokenizer._resolve_model_path",
        # A real Path, not a bare str: _apply_chat_template_sidecar (now
        # reached from the lazy branch too) does `model_path / "..."`,
        # which raises TypeError for a str left-operand.
        lambda name: Path("/fake/disk-stream-checkpoint"),
    ),
    # The lazy branch now also runs _try_inject_mtp_post_load (which
    # re-resolves the checkpoint via mlx_lm.utils._download, a real Hub
    # call) and _neutralize_unbundled_template_types (which probes
    # tokenizer_config.json, also a real Hub call for an uncached repo).
    # Both are safe no-ops for a real model (the checkpoint is already
    # Hub-cached by the preceding mlx_lm.load) but this subprocess's fake
    # ``does-not-exist/...`` name has no cache entry, so left un-mocked
    # they would 404 against the real Hub before disk_stream_patch.install
    # is ever reached. Stub them out -- this test's job is proving the
    # install() wiring, not re-exercising these fixups (that is covered by
    # test_lazy_load_runs_generic_post_load_tokenizer_fixups below and the
    # real-checkpoint slow tests in test_disk_stream_patch.py).
    mock.patch(
        "vllm_mlx.utils.tokenizer._try_inject_mtp_post_load",
        lambda model, model_name: None,
    ),
    mock.patch(
        "vllm_mlx.utils.tokenizer._neutralize_unbundled_template_types",
        lambda model_name, tokenizer_config: tokenizer_config,
    ),
    mock.patch.object(
        sys,
        "argv",
        [
            "rapid-mlx",
            "bench",
            "does-not-exist/definitely-not-a-real-model",
            "--disk-stream",
            "--num-prompts",
            "1",
            "--max-tokens",
            "1",
        ],
    ),
    mock.patch.object(sys.stdin, "isatty", return_value=False),
):
    import io
    import contextlib

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cli.main()
    except (_Stop, SystemExit):
        pass
    stdout_text = buf.getvalue()
"""


def _run_subprocess(
    script_tail: str, model_type: str, install_patch: str
) -> subprocess.CompletedProcess:
    script = (
        _SUBPROCESS_PREAMBLE
        % {"model_type": model_type, "install_patch": install_patch}
    ) + textwrap.dedent(script_tail)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_disk_stream_install_fires_on_real_bench_load_path():
    """AC #4 — wiring-regression test. Real cli.main() -> bench_command ->
    engine.batched._load_lazy_and_install_disk_stream ->
    utils.tokenizer.load_model_with_fallback(lazy=True) -> mlx_lm.load
    chain must actually reach disk_stream_patch.install(). Only mlx_lm.load
    and disk_stream_patch.install are stubbed (no real weights); everything
    in between is the real production wiring.

    A future refactor that wires --disk-stream's install call to a module
    the real bench/serve boot path never touches (the PR #967 bug class)
    makes this subprocess assertion fail.
    """
    install_patch = textwrap.dedent(
        """
        def _fake_install(model, model_type, checkpoint_path, cache_budget_gb=1.0):
            calls.append((model_type, str(checkpoint_path), cache_budget_gb))
            raise _Stop
        disk_stream_patch.install = _fake_install
        """
    )
    tail = """
    if not calls:
        print("FAIL: disk_stream_patch.install was never called")
        print(stdout_text)
        sys.exit(1)
    model_type, checkpoint_path, cache_gb = calls[0]
    if model_type != "lfm2_moe":
        print(f"FAIL: unexpected model_type {model_type!r}")
        sys.exit(1)
    if checkpoint_path != "/fake/disk-stream-checkpoint":
        print(f"FAIL: unexpected checkpoint_path {checkpoint_path!r}")
        sys.exit(1)
    if cache_gb != 1.0:
        print(f"FAIL: unexpected cache_budget_gb {cache_gb!r}")
        sys.exit(1)
    print("OK")
    """
    result = _run_subprocess(tail, "lfm2_moe", install_patch)

    assert result.returncode == 0, (
        f"subprocess wiring check failed (exit={result.returncode}).\\n"
        f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}\\n"
        "--disk-stream's install() was not reached through the real "
        "cli -> bench_command -> engine.batched -> utils.tokenizer chain "
        "(PR #967-class wiring regression)."
    )
    assert "OK" in result.stdout, (
        f"subprocess did not print OK marker. stdout:\\n{result.stdout}\\n"
        f"stderr:\\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 3b. Same wiring-regression guarantee, but for the ``serve`` path — the
#     PRD's primary user story (``rapid-mlx serve --disk-stream``) is what
#     an operator actually runs; ``bench`` above only proves the flag isn't
#     dead on the benchmark harness. Named analogously to both the bench
#     test above and this repo's own precedent,
#     ``test_deepseek_v32_indexer_gate.py::test_install_fires_on_real_serve_import_path``.
# ---------------------------------------------------------------------------

_SERVE_SUBPROCESS_SCRIPT = """
import asyncio
import contextlib
import io
import sys
from pathlib import Path
from unittest import mock

# Import the real serve-path modules FIRST — never `disk_stream_patch`
# directly. Mirrors the bench test's discipline: the assertion must be
# about what the REAL boot path
#   cli.serve_command -> server.load_model -> BatchedEngine(...)
#   -> BatchedEngine._start_llm
#   -> engine.batched._load_lazy_and_install_disk_stream
#   -> utils.tokenizer.load_model_with_fallback(lazy=True) -> mlx_lm.load
# does, not about what this test script calls directly.
import vllm_mlx.utils.tokenizer  # noqa: F401
import vllm_mlx.cli as cli
import vllm_mlx.server as server
import vllm_mlx.disk_stream_patch as disk_stream_patch
import mlx_lm


class _Stop(Exception):
    pass


calls = []


def _fake_mlx_lm_load(path_or_hf_repo, tokenizer_config=None, model_config=None,
                       adapter_path=None, lazy=False, return_config=False,
                       revision=None):
    if not lazy:
        print("FAIL: mlx_lm.load called with lazy=False on --disk-stream path")
        sys.exit(1)
    model = object()
    tokenizer = object()
    config = {"model_type": "qwen2_moe"}
    if return_config:
        return model, tokenizer, config
    return model, tokenizer


def _fake_install(model, model_type, checkpoint_path, cache_budget_gb=1.0):
    calls.append((model_type, str(checkpoint_path), cache_budget_gb))
    raise _Stop


mlx_lm.load = _fake_mlx_lm_load
disk_stream_patch.install = _fake_install


def _fake_run_uvicorn(app, args, log_level):
    # ``serve`` never loads real weights inside ``server.load_model`` —
    # unlike bench, load_model() only *constructs* the BatchedEngine (see
    # server.py: "server lifespan's startup_event runs ``await
    # _engine.start()`` like it does for BatchedEngine"). The real weight
    # load — and with it, ``_start_llm`` -> ``_load_lazy_and_install_
    # disk_stream`` -> ``disk_stream_patch.install`` — only fires inside
    # the FastAPI ``lifespan`` async generator's startup half, which is
    # normally driven by uvicorn.Server.startup() as part of the ASGI
    # lifespan protocol, right before uvicorn starts accept()-ing
    # connections on the bound socket.
    #
    # Binding a real socket and running a full uvicorn.Server just to
    # reach that one startup step would be slow, flaky in CI, and
    # wouldn't prove anything a lighter probe can't. Starlette's own
    # lifespan wrapper for a bare async-generator ``lifespan`` function
    # does nothing more than ``await agen.__anext__()`` to drive it up to
    # its ``yield`` — that IS the whole mechanism uvicorn relies on. So
    # instead of stubbing this out to a no-op (which would prove nothing
    # about disk-stream wiring), drive that exact same real ``server.
    # lifespan`` generator here. This exercises the literal production
    # startup hook with no server ever bound to a port.
    async def _drive_lifespan_startup():
        agen = server.lifespan(app)
        await agen.__anext__()

    asyncio.run(_drive_lifespan_startup())


with (
    mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
    mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
    mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
    mock.patch.object(cli, "_port_preflight_or_die", lambda *a, **k: None),
    mock.patch.object(cli, "_resolve_audio_model_for_serve", lambda _n: None),
    mock.patch.object(cli, "_run_uvicorn", _fake_run_uvicorn),
    # server.load_model()'s own MLLM-vs-text routing safety gate (#352):
    # it prefetches + verifies the checkpoint config exists on disk before
    # ``resolve_serving_lane`` reads it, and hard-fails if that verification
    # doesn't find one. Unrelated to disk-stream wiring (it's a routing
    # pre-check, not part of the install() chain this test pins) — stub it
    # out so this fake ``does-not-exist/...`` repo doesn't trip the
    # "could not materialize checkpoint config" fail-fast.
    mock.patch.object(server, "_ensure_routing_config", lambda *a, **k: None),
    mock.patch("vllm_mlx.api.utils.is_mllm_model", lambda _n: False),
    mock.patch("vllm_mlx.audio.probe.is_audio_model_alias", lambda _n: False),
    mock.patch(
        "vllm_mlx._version_check.prompt_upgrade_if_available", lambda: False
    ),
    mock.patch(
        "vllm_mlx._version_check.print_staleness_warning_if_any", lambda: None
    ),
    mock.patch(
        "vllm_mlx.utils.tokenizer._resolve_model_path",
        # A real Path, not a bare str — see the bench test's identical
        # comment: _apply_chat_template_sidecar does `model_path / "..."`.
        lambda name: Path("/fake/disk-stream-checkpoint"),
    ),
    # Same as the bench test: stub the two lazy-path post-load fixups
    # that hit the network by design, so this fake ``does-not-exist/...``
    # model name doesn't have to round-trip the real Hub.
    mock.patch(
        "vllm_mlx.utils.tokenizer._try_inject_mtp_post_load",
        lambda model, model_name: None,
    ),
    mock.patch(
        "vllm_mlx.utils.tokenizer._neutralize_unbundled_template_types",
        lambda model_name, tokenizer_config: tokenizer_config,
    ),
    mock.patch.object(
        sys,
        "argv",
        [
            "rapid-mlx",
            "serve",
            "does-not-exist/definitely-not-a-real-model",
            "--port",
            "0",
            "--disk-stream",
        ],
    ),
    # Non-interactive stdin short-circuits cli.main()'s auto-pull confirm
    # gate (which would otherwise probe the real HF Hub for this fake
    # repo's size) — same reasoning as the bench test's identical stub.
    mock.patch.object(sys.stdin, "isatty", return_value=False),
):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            cli.main()
    except (_Stop, SystemExit):
        pass
    stdout_text = buf.getvalue()

if not calls:
    print("FAIL: disk_stream_patch.install was never called")
    print(stdout_text)
    sys.exit(1)
model_type, checkpoint_path, cache_gb = calls[0]
if model_type != "qwen2_moe":
    print(f"FAIL: unexpected model_type {model_type!r}")
    sys.exit(1)
if checkpoint_path != "/fake/disk-stream-checkpoint":
    print(f"FAIL: unexpected checkpoint_path {checkpoint_path!r}")
    sys.exit(1)
if cache_gb != 1.0:
    print(f"FAIL: unexpected cache_budget_gb {cache_gb!r}")
    sys.exit(1)
print("OK")
"""


def test_disk_stream_install_fires_on_real_serve_load_path():
    """Serve-path counterpart of ``test_disk_stream_install_fires_on_real_
    bench_load_path`` above — the wiring-regression test that one leaves
    uncovered. The PRD's primary user story is ``rapid-mlx serve
    --disk-stream``, not ``bench --disk-stream``; a future refactor could
    wire the installer to a helper ``bench`` happens to share while
    ``serve`` quietly stops calling it (exactly the PR #967 bug class: an
    installer wired to a code path the real production boot doesn't
    actually use) and the bench test alone would never catch it.

    Drives the REAL ``cli.main()`` -> ``serve_command`` ->
    ``server.load_model`` -> ``BatchedEngine(...)`` chain in a subprocess
    (same isolation rationale as the bench test and
    ``test_deepseek_v32_indexer_gate.py``'s precedent: a pristine
    interpreter per run, no in-process module-patching pollution across
    the test session). Only ``mlx_lm.load`` and ``disk_stream_patch.
    install`` are stubbed (no real weights); everything in between,
    including the real ``BatchedEngine`` construction, is production
    wiring.

    ``serve`` (unlike ``bench``) does not load weights synchronously
    inside ``cli.main()`` — ``server.load_model()`` only constructs the
    ``BatchedEngine``; the actual weight load happens inside FastAPI's
    ``lifespan`` startup hook, which real ``uvicorn.run()`` drives as
    part of the ASGI lifespan protocol immediately before it starts
    accepting connections. Rather than binding a real socket and running
    a full ``uvicorn.Server`` just to reach that hook, ``cli._run_uvicorn``
    is stubbed to instead drive the real ``server.lifespan`` async
    generator directly up to its startup ``yield`` — the exact mechanism
    Starlette itself uses to run a bare async-generator lifespan
    function, and so the same thing uvicorn would have done. No server
    ever binds a port; the real startup code path still runs in full.

    A future refactor that wires --disk-stream's install call to a
    module the real serve boot path never touches makes this subprocess
    assertion fail.
    """
    result = subprocess.run(
        [sys.executable, "-c", _SERVE_SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"subprocess wiring check failed (exit={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
        "--disk-stream's install() was not reached through the real "
        "cli -> serve_command -> server.load_model -> BatchedEngine -> "
        "_start_llm chain (PR #967-class wiring regression)."
    )
    assert "OK" in result.stdout, (
        f"subprocess did not print OK marker. stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_disk_stream_unregistered_model_type_fails_clearly_at_load_time():
    """AC #5 — an unregistered model_type must fail at load time with
    disk_stream_patch's own clear UnsupportedModelTypeError message, not a
    downstream crash (e.g. an AttributeError from touching the never-real
    `object()` stand-in model). disk_stream_patch.install is left
    UN-mocked here so the real registry-miss path runs.
    """
    # No install_patch — the real disk_stream_patch.install runs and must
    # raise UnsupportedModelTypeError before ever touching `model`.
    tail = """
    if "no StreamingAdapter is registered" not in stdout_text:
        print("FAIL: expected UnsupportedModelTypeError message not found")
        print(stdout_text)
        sys.exit(1)
    if "totally_unregistered_moe_arch_xyz" not in stdout_text:
        print("FAIL: error message did not name the unregistered model_type")
        print(stdout_text)
        sys.exit(1)
    if "AttributeError" in stdout_text or "Traceback" in stdout_text:
        print("FAIL: a downstream crash leaked through instead of a clean error")
        print(stdout_text)
        sys.exit(1)
    print("OK")
    """
    result = _run_subprocess(tail, "totally_unregistered_moe_arch_xyz", "")

    assert result.returncode == 0, (
        f"subprocess unregistered-model-type check failed "
        f"(exit={result.returncode}).\\n"
        f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"subprocess did not print OK marker. stdout:\\n{result.stdout}\\n"
        f"stderr:\\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 5. Lazy-path post-load fixup parity — regression test for the blocking
#    finding in review-notes.md. load_model_with_fallback(lazy=True) must
#    still run every architecture-independent post-load tokenizer fixup
#    the non-lazy path runs, not just the branch-selection logic it's
#    legitimately allowed to skip. Fast, in-process, no checkpoint needed.
# ---------------------------------------------------------------------------


def test_lazy_load_runs_generic_post_load_tokenizer_fixups(monkeypatch):
    """Pins the fix for review-notes.md's blocking finding: the lazy=True
    short-circuit in load_model_with_fallback used to `return` immediately
    after the raw `mlx_lm.load(..., lazy=True)` call, skipping
    `_neutralize_unbundled_template_types`, `_try_inject_mtp_post_load`,
    `_apply_chat_template_sidecar`, `augment_eos_token_ids_from_
    generation_config`, and `repair_byte_level_decoder` — all of which the
    non-lazy path runs unconditionally after every successful load,
    regardless of architecture.

    Concretely dangerous omission: `augment_eos_token_ids_from_
    generation_config`'s own docstring names Qwen3/Qwen2.5 (151645/151643)
    as the scenario it exists to fix, and `qwen2_moe` is one of the two
    registered --disk-stream architectures — without this call, a
    qwen2_moe checkpoint loaded with --disk-stream would silently run to
    max_tokens instead of stopping at its chat-template terminator, the
    same checkpoint stopping correctly without the flag.

    Monkeypatches all five fixups (module-level, so the lazy branch's own
    unqualified calls pick up the stubs) to record whether/how they fire,
    fakes `mlx_lm.load` so no checkpoint download is needed, and asserts
    every one of them ran with the model/tokenizer/model_name the real
    call would pass — plus that `_neutralize_unbundled_template_types`
    (the #1420 guard) ran BEFORE `mlx_lm.load`, since it mutates
    tokenizer_config which mlx_lm.load's own tokenizer loading consumes.
    """
    import mlx_lm

    from vllm_mlx.utils import tokenizer as tok

    order: list[str] = []
    fake_model = object()
    fake_tokenizer = SimpleNamespace(chat_template=None)
    model_name = "mlx-community/fake-qwen2-moe-checkpoint-4bit"

    def _fake_mlx_lm_load(path_or_hf_repo, tokenizer_config=None, **kwargs):
        order.append("mlx_lm.load")
        assert kwargs.get("lazy") is True
        assert tokenizer_config == {"neutralized": True}, (
            "mlx_lm.load must see the tokenizer_config AFTER "
            "_neutralize_unbundled_template_types mutated it, not the "
            "caller's original dict"
        )
        if kwargs.get("return_config"):
            return fake_model, fake_tokenizer, {"model_type": "qwen2_moe"}
        return fake_model, fake_tokenizer

    def _fake_neutralize(name, tokenizer_config):
        order.append("_neutralize_unbundled_template_types")
        assert name == model_name
        return {"neutralized": True}

    def _fake_inject_mtp(model, name):
        order.append("_try_inject_mtp_post_load")
        assert model is fake_model
        assert name == model_name

    def _fake_sidecar(model_path, tokenizer):
        order.append("_apply_chat_template_sidecar")
        assert tokenizer is fake_tokenizer

    def _fake_augment_eos(tokenizer, name):
        order.append("augment_eos_token_ids_from_generation_config")
        assert tokenizer is fake_tokenizer
        assert name == model_name

    def _fake_repair_decoder(tokenizer):
        order.append("repair_byte_level_decoder")
        assert tokenizer is fake_tokenizer

    monkeypatch.setattr(mlx_lm, "load", _fake_mlx_lm_load)
    monkeypatch.setattr(tok, "_neutralize_unbundled_template_types", _fake_neutralize)
    monkeypatch.setattr(tok, "_try_inject_mtp_post_load", _fake_inject_mtp)
    monkeypatch.setattr(tok, "_apply_chat_template_sidecar", _fake_sidecar)
    monkeypatch.setattr(
        tok, "_resolve_model_path", lambda name: "/fake/disk-stream-checkpoint"
    )
    monkeypatch.setattr(
        tok, "augment_eos_token_ids_from_generation_config", _fake_augment_eos
    )
    monkeypatch.setattr(tok, "repair_byte_level_decoder", _fake_repair_decoder)
    monkeypatch.setattr(tok, "_post_load_ubc_evict", lambda name: None)

    model, tokenizer, config = tok.load_model_with_fallback(
        model_name, lazy=True, return_config=True
    )

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    assert config == {"model_type": "qwen2_moe"}
    assert order == [
        "_neutralize_unbundled_template_types",
        "mlx_lm.load",
        "_try_inject_mtp_post_load",
        "_apply_chat_template_sidecar",
        "augment_eos_token_ids_from_generation_config",
        "repair_byte_level_decoder",
    ], f"post-load fixups did not all run, or ran out of order: {order}"
