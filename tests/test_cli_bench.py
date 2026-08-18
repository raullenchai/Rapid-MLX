# SPDX-License-Identifier: Apache-2.0
"""Tests for ``rapid-mlx bench`` R2-mirror routing (regression: #650 shape).

Before this fix, ``bench_command`` (freeform path) and ``_run_submit_flow``
(``--submit`` community-bench path) both delegated the initial weight pull
to ``mlx_lm.load`` → ``huggingface_hub.snapshot_download`` DIRECTLY,
bypassing the R2 mirror at ``models.rapidmlx.com`` that every other
weight-materializing command (``serve``, ``chat``, ``pull``)
prefetches through. Result: users burned bandwidth on HF and tripped rate
limits even though a warm mirror shard was 50 ms away.

These tests lock the fix: both bench paths call ``_ensure_model_downloaded``
(the same helper ``serve`` uses, which handles the R2-first + HF-fallback
contract) BEFORE ``mlx_lm.load`` runs. They also confirm the graceful
degradation contract: when the mirror is unreachable, prefetch returns
without raising and bench proceeds to ``mlx_lm.load`` (which then pulls
from HF directly, matching pre-fix behavior for the fallback case).

The bench-path prefetch pattern was lifted from the (now-removed)
``rapid-mlx jlens`` command's downloading tests (PR #1045).
"""

from __future__ import annotations

import argparse
import importlib

import pytest


# ---------- shared helpers ---------------------------------------------------
def _patch_mlx_lm_load(monkeypatch, fake_load) -> None:
    """Patch the loader ``bench`` actually calls.

    ``bench_command`` and ``_run_submit_flow`` bind
    ``from .utils.tokenizer import load_model_with_fallback as load`` at
    call time — that's
    ``getattr(sys.modules['vllm_mlx.utils.tokenizer'],
    'load_model_with_fallback')`` resolved fresh each call. They used to
    bind ``mlx_lm.load`` directly, but that loader has no
    ``gemma4_unified`` architecture, so every ``gemma-4-12b-*`` alias
    failed to load; the router in ``utils.tokenizer`` is what makes them
    work. Patch that attribute rather than the module object so the other
    imports (``mlx_lm.generate``, ``mlx_lm.utils``…) the engine triggers
    during setup stay intact.
    """
    from vllm_mlx.utils import tokenizer as _tokenizer_mod

    monkeypatch.setattr(
        _tokenizer_mod, "load_model_with_fallback", fake_load, raising=True
    )


def _make_freeform_bench_args(model: str) -> argparse.Namespace:
    """Minimal Namespace to reach ``load(args.model)`` in the freeform path.

    Every attribute below is one that ``bench_command`` (or the shared
    pflash / prefix-cache setup it drives) reads before the load call.
    Setting ``pflash="off"`` short-circuits the alias-lookup branch in
    ``resolve_pflash_mode_default`` and lets ``config_from_args`` build a
    real ``PFlashConfig`` without any HF metadata calls.
    """
    return argparse.Namespace(
        model=model,
        tier=None,
        submit=False,
        force_disk_check=False,
        # PFlash knobs — all required by ``config_from_args``. Values
        # match the CLI defaults (see ``vllm_mlx/args.py``).
        pflash="off",
        pflash_threshold=1024,
        pflash_keep_ratio=0.20,
        pflash_min_keep_tokens=256,
        pflash_sink_tokens=64,
        pflash_tail_tokens=64,
        pflash_block_size=16,
        pflash_query_window=128,
        pflash_stride_blocks=0,
        pflash_include_tools=False,
        mllm=False,
        # Prefix cache toggle read BEFORE the async runner.
        enable_prefix_cache=True,
        disable_prefix_cache=False,
    )


# ---------- freeform bench path (bench_command) -----------------------------
def test_bench_command_prefetches_via_mirror_before_hf_load(monkeypatch) -> None:
    """Freeform ``rapid-mlx bench <alias>`` must call
    ``_ensure_model_downloaded(args.model)`` BEFORE ``mlx_lm.load(args.model)``.

    Regression contract: without the prefetch, ``load`` triggers
    ``huggingface_hub.snapshot_download`` directly and skips the R2
    mirror at ``models.rapidmlx.com`` — the exact bypass that also
    hit ``jlens`` before PR #1045 and ``serve`` before #651.
    """
    cli = importlib.import_module("vllm_mlx.cli")

    order: list[str] = []

    def _fake_prefetch(name: str) -> None:
        order.append(f"prefetch:{name}")

    def _fake_load(name: str):
        order.append(f"load:{name}")
        # Abort inside the async runner: ``ValueError`` hits the
        # bench's own 404-friendly except branch which calls
        # ``sys.exit(1)`` — we swallow that below with ``pytest.raises``.
        raise ValueError("test-abort — bench should reach load AFTER prefetch")

    # Silence the network-heavy pre-checks. ``_ensure_model_downloaded``
    # is the piece under test; ``_check_disk_space`` /
    # ``_check_memory_capacity`` are covered by their own tests.
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", _fake_prefetch)

    # Bypass the pflash alias-tier lookup (would touch aliases.json /
    # detect_model_config). The order test doesn't depend on it.
    monkeypatch.setattr(
        "vllm_mlx.pflash.resolve_pflash_mode_default",
        lambda args, *, model_name, is_multimodal=False, **_kw: "off",
    )
    # Route the ``from mlx_lm import load`` inside bench_command to
    # our mock. Patching ``mlx_lm.load`` on the module object makes
    # the deferred ``from`` import bind to our recorder.
    _patch_mlx_lm_load(monkeypatch, _fake_load)

    args = _make_freeform_bench_args("mlx-community/Qwen3-1.7B-4bit")

    with pytest.raises(SystemExit) as exc:
        cli.bench_command(args)
    # bench_command's own 404-branch translates ValueError → sys.exit(1).
    assert exc.value.code == 1

    # 1) prefetch was consulted, 2) BEFORE mlx_lm.load, 3) with the
    # SAME model id the user typed — no accidental re-resolution.
    assert order == [
        "prefetch:mlx-community/Qwen3-1.7B-4bit",
        "load:mlx-community/Qwen3-1.7B-4bit",
    ], f"order violation: {order}"


def test_bench_command_proceeds_when_mirror_prefetch_is_silent_noop(
    monkeypatch,
) -> None:
    """Graceful degradation contract: when ``_ensure_model_downloaded``
    returns None (the normal contract for cached / local / mirror-miss
    paths — the helper swallows mirror errors internally), the freeform
    bench proceeds to ``mlx_lm.load`` which falls through to HF. Bench
    MUST NOT raise from the prefetch call site.
    """
    cli = importlib.import_module("vllm_mlx.cli")

    load_called: list[str] = []

    def _silent_prefetch(name: str) -> None:
        # Simulates the real helper's happy path (cached repo / warm
        # mirror / silent mirror miss). Return None, no raise.
        return None

    def _fake_load(name: str):
        load_called.append(name)
        raise ValueError("test-abort")

    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", _silent_prefetch)
    monkeypatch.setattr(
        "vllm_mlx.pflash.resolve_pflash_mode_default",
        lambda args, *, model_name, is_multimodal=False, **_kw: "off",
    )
    _patch_mlx_lm_load(monkeypatch, _fake_load)

    args = _make_freeform_bench_args("mlx-community/Qwen3-1.7B-4bit")

    with pytest.raises(SystemExit):
        cli.bench_command(args)

    # Bench proceeded to load — the graceful-fallback contract holds.
    assert load_called == ["mlx-community/Qwen3-1.7B-4bit"]


def test_bench_command_loads_weights_on_mlx_step_worker(monkeypatch) -> None:
    """Freeform ``rapid-mlx bench <alias>`` must load the weights ON the
    mlx-step worker thread (#170), NOT the main / asyncio thread.

    mlx-lm 0.31.3+ binds the module-level generation stream (and each
    thread's auto-default stream) to whichever thread first touches MLX.
    Loading the weights on the main thread and then generating on the
    engine's worker raises "There is no Stream(gpu, N) in current thread"
    on the very first batch step, so every request aborts and the run
    silently reports ``0.00 tok/s`` (the app's "Speed on this Mac" card
    then shows a confident, wrong zero). Pinning the load onto the
    ``mlx-step`` worker — the executor AsyncEngineCore then reuses — is
    the only configuration that survives ThreadLocalStream tagging, and
    it mirrors the ``--submit`` community path + why ``serve`` works.
    """
    import threading

    cli = importlib.import_module("vllm_mlx.cli")

    load_thread: dict[str, str] = {}

    def _fake_load(name: str):
        load_thread["name"] = threading.current_thread().name
        # Abort before the engine spins up — this test only pins WHERE the
        # weights are loaded, not a full generation run.
        raise ValueError("test-abort — captured load thread")

    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda name: None)
    monkeypatch.setattr(
        "vllm_mlx.pflash.resolve_pflash_mode_default",
        lambda args, *, model_name, is_multimodal=False, **_kw: "off",
    )
    _patch_mlx_lm_load(monkeypatch, _fake_load)

    args = _make_freeform_bench_args("mlx-community/Qwen3-1.7B-4bit")

    with pytest.raises(SystemExit) as exc:
        cli.bench_command(args)
    assert exc.value.code == 1

    loaded_on = load_thread.get("name", "")
    assert loaded_on.startswith("mlx-step"), (
        f"weights loaded on {loaded_on!r}, not an 'mlx-step' worker — "
        "freeform bench would hit the cross-thread GPU-stream bug and "
        "report 0 tok/s"
    )


def _capture_bench_lane_signals(monkeypatch, cli):
    """Wire the bench PFlash default + validation seams to record the
    ``is_multimodal`` / ``is_mllm`` they receive, and abort before the heavy
    engine boot. Returns the ``seen`` dict."""
    seen: dict[str, object] = {}

    def _capture_default(args, *, model_name, is_multimodal=False, **_kw):
        seen["default_is_multimodal"] = is_multimodal
        return "off"

    def _capture_validate(config, *, model_name, is_mllm):
        seen["validate_is_mllm"] = is_mllm

    monkeypatch.setattr("vllm_mlx.pflash.resolve_pflash_mode_default", _capture_default)
    monkeypatch.setattr("vllm_mlx.pflash.validate_model_support", _capture_validate)
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_ensure_model_downloaded", lambda name: None)

    def _fake_load(name: str):
        # Stop before the (heavy) real engine boot; bench's own except
        # branch turns this into sys.exit(1).
        raise ValueError("test-abort after lane resolution")

    _patch_mlx_lm_load(monkeypatch, _fake_load)
    return seen


def test_bench_command_hybrid_vlm_benches_on_text_lane(monkeypatch, capsys) -> None:
    """BLOCKING (#1178 codex r4): a hybrid VLM alias must bench on the TEXT
    lane. ``bench`` has no MLLM/continuous-batching engine — it always loads
    the text model via ``mlx_lm.load`` — so PFlash defaulting/validation run on
    the text lane (``is_mllm=False``) and the auto-downgrade is surfaced to the
    user for lane attribution (#352).
    """
    cli = importlib.import_module("vllm_mlx.cli")
    from vllm_mlx.api import utils as api_utils

    # Hybrid VLM: resolver returns (is_mllm_lane=False, auto_text_fallback=True).
    monkeypatch.setattr(
        api_utils, "resolve_serving_lane", lambda name, **kw: (False, True)
    )
    seen = _capture_bench_lane_signals(monkeypatch, cli)

    args = _make_freeform_bench_args("some/hybrid-vlm-4bit")
    with pytest.raises(SystemExit):
        cli.bench_command(args)

    # PFlash defaulting AND validation ran on the text lane.
    assert seen.get("default_is_multimodal") is False
    assert seen.get("validate_is_mllm") is False
    # And the auto-downgrade is attributed to the right lane for the user.
    out = capsys.readouterr().out
    assert "hybrid VLM" in out and "text-only" in out


def test_bench_command_genuine_vlm_validates_on_text_lane(monkeypatch, capsys) -> None:
    """NIT (#1178 codex r5): an ordinary (non-hybrid) VLM resolves to
    ``is_mllm=True`` on the serve path, but ``bench`` is text-only-by-
    construction — it has no MLLM lane for which PFlash is unavailable. So
    PFlash defaulting AND validation must use ``is_mllm=False`` UNCONDITIONALLY,
    never rejecting a PFlash option as if an MLLM lane were in use. No
    auto-downgrade note fires for a genuine VLM (it is not a #352 downgrade).
    """
    cli = importlib.import_module("vllm_mlx.cli")
    from vllm_mlx.api import utils as api_utils

    # Genuine VLM: resolver returns (is_mllm_lane=True, auto_text_fallback=False).
    monkeypatch.setattr(
        api_utils, "resolve_serving_lane", lambda name, **kw: (True, False)
    )
    seen = _capture_bench_lane_signals(monkeypatch, cli)

    args = _make_freeform_bench_args("some/genuine-vlm-4bit")
    with pytest.raises(SystemExit):
        cli.bench_command(args)

    # Despite resolve saying is_mllm=True, bench validates against the text lane.
    assert seen.get("default_is_multimodal") is False
    assert seen.get("validate_is_mllm") is False
    # No hybrid auto-downgrade note for a genuine VLM.
    out = capsys.readouterr().out
    assert "hybrid VLM" not in out


# ---------- --submit community-bench path (_run_submit_flow) ----------------
def _install_submit_flow_stubs(monkeypatch, cli, *, alias: str, hf_path: str):
    """Common patches for the ``--submit`` path: bypass the Apple-Silicon
    gate, the whitelist lookup, and the disk / memory pre-checks so the
    test can reach the executor without touching real hardware or HF."""

    # Community bench is Apple-Silicon only; CI runs on Linux workers,
    # so the ``is_apple_silicon`` gate would exit 2 before our mocks
    # get to observe anything.
    monkeypatch.setattr(
        "vllm_mlx.community_bench.hardware.is_apple_silicon", lambda: True
    )

    # Whitelist lookup: return a stub profile with our controlled hf_path
    # so the alias-key guard passes without touching aliases.json.
    class _Profile:
        def __init__(self, hf_path):
            self.hf_path = hf_path

    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile",
        lambda name: _Profile(hf_path),
    )

    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "_check_memory_capacity", lambda *a, **kw: None)


def test_run_submit_flow_prefetches_via_mirror_before_hf_load(
    monkeypatch,
) -> None:
    """``--submit`` community-bench path MUST call
    ``_ensure_model_downloaded(hf_path)`` BEFORE the thread executor spins
    up ``mlx_lm.load(hf_path)``.

    Running in the main thread means the mirror's per-file progress lines
    print on the contributor's terminal; deferring to the executor would
    hide them behind the ``Loading model …`` line above.
    """
    cli = importlib.import_module("vllm_mlx.cli")

    order: list[str] = []

    def _fake_prefetch(name: str) -> None:
        order.append(f"prefetch:{name}")

    def _fake_load(name: str):
        order.append(f"load:{name}")
        # Return 2 branch: ``ValueError`` inside the executor is caught
        # at cli.py:3554 which returns 2 from ``_run`` (NOT ``sys.exit``).
        raise ValueError("test-abort — submit should reach load AFTER prefetch")

    _install_submit_flow_stubs(
        monkeypatch,
        cli,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
    )
    monkeypatch.setattr(cli, "_ensure_model_downloaded", _fake_prefetch)
    _patch_mlx_lm_load(monkeypatch, _fake_load)

    args = argparse.Namespace(
        model="qwen3.5-9b-4bit",
        submit=True,
        sampled=False,
        notes=None,
        force_disk_check=False,
        # ``_original_alias`` is unset on the direct-alias path — the
        # flow falls back to ``args.model`` (matches production).
    )

    rc = cli._run_submit_flow(args)
    # Load raised ValueError → caught at cli.py:3554 → return 2.
    assert rc == 2

    assert order == [
        "prefetch:mlx-community/Qwen3.5-9B-4bit",
        "load:mlx-community/Qwen3.5-9B-4bit",
    ], f"order violation: {order}"


def test_run_submit_flow_proceeds_when_mirror_prefetch_is_silent_noop(
    monkeypatch,
) -> None:
    """Mirror-fallback contract for ``--submit``: when
    ``_ensure_model_downloaded`` returns None (cached / mirror miss /
    disabled mirror), ``_run_submit_flow`` MUST proceed to
    ``mlx_lm.load``, which then completes the pull via HF."""
    cli = importlib.import_module("vllm_mlx.cli")

    load_called: list[str] = []

    def _silent_prefetch(name: str) -> None:
        return None

    def _fake_load(name: str):
        load_called.append(name)
        raise ValueError("test-abort")

    _install_submit_flow_stubs(
        monkeypatch,
        cli,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
    )
    monkeypatch.setattr(cli, "_ensure_model_downloaded", _silent_prefetch)
    _patch_mlx_lm_load(monkeypatch, _fake_load)

    args = argparse.Namespace(
        model="qwen3.5-9b-4bit",
        submit=True,
        sampled=False,
        notes=None,
        force_disk_check=False,
    )

    rc = cli._run_submit_flow(args)
    assert rc == 2  # ValueError → load-branch → return 2

    # Submit proceeded to load — the graceful-fallback contract holds.
    assert load_called == ["mlx-community/Qwen3.5-9B-4bit"]
