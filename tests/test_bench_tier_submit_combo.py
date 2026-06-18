# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx bench <model> --tier <T> --submit`` — PR #5 unification.

PR #2 landed ``--tier`` and PR #3 landed schema v2 with optional
``smoke_result`` / ``harness_result`` sub-objects, but the CLI kept
the two flags mutually-exclusive (a stub exit-2 guard with a comment
literally saying "PR #3 will unify them"). PR #5 removes the guard
and wires the combo end-to-end.

These tests pin the wiring contract:

- ``run_tier(..., return_results=True)`` returns
  ``(int, {smoke_result, harness_result})`` carrying the schema-v2
  sub-objects (with ``None`` for tiers that didn't run).
- ``run_tier`` without ``return_results=True`` keeps its legacy
  ``int`` return type — no caller-visible drift for code that just
  wants the exit code.
- ``build_submission_payload(tier=..., smoke_result=...,
  harness_result=...)`` accepts the dicts the dispatcher emits and
  produces a payload that validates against ``schema.json`` for every
  tier value (smoke / harness / all). Speed/buckets are always
  present because the schema requires them.
- The previously-mutual-exclusive guard is gone: ``bench --tier all
  --submit`` parses cleanly and dispatches into the new combo flow.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx.bench.tier_runner import TierResult, run_tier

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "community-benchmarks" / "schema.json"


# --------------------------------------------------------------------- #
# Stub inputs                                                            #
# --------------------------------------------------------------------- #


def _stub_bench_inputs():
    """Build the hardware / software / BenchResult triple that
    ``build_submission_payload`` consumes. Mirrors the fixtures in
    ``tests/test_payload_builder_v2_full.py`` so behavior is
    cross-consistent across the schema-v2 test suite.
    """
    from vllm_mlx.community_bench.hardware import Hardware, Software
    from vllm_mlx.community_bench.runner import (
        BenchResult,
        BucketResult,
        RoundResult,
    )

    hw = Hardware(chip="Apple M4 Pro", ram_gb=24, cpu_cores=12, gpu_cores=20)
    sw = Software(macos="26.5.1", rapid_mlx="0.7.25", mlx="0.31.2", python="3.12.13")
    rounds = [
        RoundResult(decode_tps=42.0, prefill_tps=500.0, ttft_ms=120.0) for _ in range(5)
    ]
    bench = BenchResult(
        short=BucketResult(rounds_raw=rounds),
        long=BucketResult(rounds_raw=rounds),
        peak_ram_mb=8192,
        prompt_hash="deadbeefcafebabe",
        sampling="greedy",
    )
    return hw, sw, bench


_SMOKE_PAYLOAD = {
    "boot_time_ms": 1234.5,
    "first_prompt_ok": True,
    "first_token_latency_ms": 87.0,
    "response_excerpt": "2+2 equals 4.",
}

_HARNESS_PAYLOAD = {
    "codex": {"passed": True, "duration_s": 11.4, "error_excerpt": None},
    "opencode": {"passed": True, "duration_s": 9.8, "error_excerpt": None},
    "hermes": {"passed": True, "duration_s": 7.1, "error_excerpt": None},
    "aider": {"passed": True, "duration_s": 14.2, "error_excerpt": None},
    "langchain": {"passed": True, "duration_s": 18.5, "error_excerpt": None},
}


# --------------------------------------------------------------------- #
# Server-boot context manager stub                                       #
# --------------------------------------------------------------------- #


import contextlib


@contextlib.contextmanager
def _fake_serve(model, port=None, **kwargs):
    yield {
        "base_url": f"http://127.0.0.1:{port}/v1",
        "port": port,
        "boot_time_ms": 1234.5,
    }


def _patch_serve_boot():
    """Bundle the server-boot + free-port patches for tier_runner tests."""

    def _free_port(lo, hi):
        return 8500

    return [
        patch(
            "vllm_mlx.bench.tier_runner._find_free_port_in_range",
            side_effect=_free_port,
        ),
        patch("vllm_mlx.bench._server.serve", _fake_serve),
    ]


# --------------------------------------------------------------------- #
# run_tier: structured return contract                                   #
# --------------------------------------------------------------------- #


def test_run_tier_returns_payload_when_requested():
    """``return_results=True`` flips the return type to (int, dict).

    The dict carries the schema-v2-shaped ``smoke_result`` and
    ``harness_result`` for tiers that ran (or ``None`` otherwise),
    so the ``--submit`` wiring can feed it straight to
    ``build_submission_payload`` without re-running tier work.
    """

    def _smoke_stub(model, base_url, boot_time_ms=None):
        return TierResult(
            name="smoke",
            passed=True,
            duration_s=0.5,
            detail="PASS",
            payload=dict(_SMOKE_PAYLOAD),
        )

    def _harness_stub(model, base_url, **kwargs):
        return TierResult(
            name="harness",
            passed=True,
            duration_s=30.0,
            detail="5/5 pass",
            payload=dict(_HARNESS_PAYLOAD),
        )

    with contextlib.ExitStack() as stack:
        for p in _patch_serve_boot():
            stack.enter_context(p)
        stack.enter_context(patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub))
        stack.enter_context(
            patch("vllm_mlx.bench.tier_runner._run_harness", _harness_stub)
        )

        result = run_tier(
            model="qwen3.5-4b-4bit",
            tier="all",
            return_results=True,
            skip_speed=True,
        )

    assert isinstance(result, tuple), (
        "return_results=True must change the return type to a tuple"
    )
    rc, payload = result
    assert rc == 0, f"happy path should exit 0; got {rc}"
    assert payload["smoke_result"] == _SMOKE_PAYLOAD
    assert payload["harness_result"] == _HARNESS_PAYLOAD


def test_run_tier_default_signature_unchanged():
    """Without ``return_results=True``, ``run_tier`` still returns ``int``.

    Locks the back-compat contract: PR #2's callers (e.g. release_check_m3.sh
    G7b) MUST not see a behavior change just because the kwarg exists.
    """

    def _smoke_stub(model, base_url, boot_time_ms=None):
        return TierResult(name="smoke", passed=True, duration_s=0.1)

    with contextlib.ExitStack() as stack:
        for p in _patch_serve_boot():
            stack.enter_context(p)
        stack.enter_context(patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub))

        result = run_tier(model="qwen3.5-4b-4bit", tier="smoke")

    assert isinstance(result, int), (
        "default ``run_tier`` MUST return int, not tuple — back-compat"
    )
    assert result == 0


def test_run_tier_returns_payload_with_none_for_missing_tiers():
    """When a tier didn't run, its slot in the payload dict is ``None``.

    For ``tier='smoke'`` the dispatcher only runs smoke, so
    ``harness_result`` MUST be ``None`` in the returned dict — the
    caller can pattern-match on that rather than checking key presence.
    """

    def _smoke_stub(model, base_url, boot_time_ms=None):
        return TierResult(
            name="smoke",
            passed=True,
            duration_s=0.5,
            payload=dict(_SMOKE_PAYLOAD),
        )

    with contextlib.ExitStack() as stack:
        for p in _patch_serve_boot():
            stack.enter_context(p)
        stack.enter_context(patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub))

        rc, payload = run_tier(
            model="qwen3.5-4b-4bit",
            tier="smoke",
            return_results=True,
        )

    assert rc == 0
    assert payload["smoke_result"] == _SMOKE_PAYLOAD
    assert payload["harness_result"] is None, (
        "harness didn't run for tier=smoke; harness_result MUST be None"
    )


def test_run_tier_skip_speed_avoids_lightweight_probe():
    """``skip_speed=True`` on tier='all' MUST NOT invoke ``_run_speed``.

    The --submit combo path uses ``run_standardized_bench`` directly
    for the speed bucket because the locked B=1 numbers are what
    community-benchmarks compares against. Running the lightweight
    HTTP probe alongside it would waste cycles AND produce non-
    comparable numbers next to the submitted ones (confusing).
    """
    calls: list[str] = []

    def _smoke_stub(model, base_url, boot_time_ms=None):
        calls.append("smoke")
        return TierResult(
            name="smoke",
            passed=True,
            duration_s=0.1,
            payload=dict(_SMOKE_PAYLOAD),
        )

    def _speed_stub(model, base_url, sampled=False):
        calls.append("speed")
        return TierResult(name="speed", passed=True, duration_s=0.1)

    def _harness_stub(model, base_url, **kwargs):
        calls.append("harness")
        return TierResult(
            name="harness",
            passed=True,
            duration_s=0.1,
            payload=dict(_HARNESS_PAYLOAD),
        )

    with contextlib.ExitStack() as stack:
        for p in _patch_serve_boot():
            stack.enter_context(p)
        stack.enter_context(patch("vllm_mlx.bench.tier_runner._run_smoke", _smoke_stub))
        stack.enter_context(patch("vllm_mlx.bench.tier_runner._run_speed", _speed_stub))
        stack.enter_context(
            patch("vllm_mlx.bench.tier_runner._run_harness", _harness_stub)
        )

        run_tier(
            model="qwen3.5-4b-4bit",
            tier="all",
            return_results=True,
            skip_speed=True,
        )

    assert "speed" not in calls, (
        f"skip_speed=True must skip the lightweight speed probe; calls={calls}"
    )
    assert calls == ["smoke", "harness"], (
        f"--tier all skip_speed must run smoke → harness; got {calls}"
    )


def test_run_tier_skip_speed_ignored_for_non_all_tier():
    """``skip_speed`` is a tier=='all'-only knob.

    For tier='speed' alone, skip_speed makes no sense — the user
    asked for ``--tier speed``, the dispatcher honours it. Adding
    skip_speed=True to a tier='speed' call MUST still run the speed
    probe (otherwise we'd silently no-op).
    """
    calls: list[str] = []

    def _speed_stub(model, base_url, sampled=False):
        calls.append("speed")
        return TierResult(name="speed", passed=True, duration_s=0.1)

    with contextlib.ExitStack() as stack:
        for p in _patch_serve_boot():
            stack.enter_context(p)
        stack.enter_context(patch("vllm_mlx.bench.tier_runner._run_speed", _speed_stub))

        run_tier(
            model="qwen3.5-4b-4bit",
            tier="speed",
            return_results=True,
            skip_speed=True,
        )

    assert calls == ["speed"], f"skip_speed must be tier='all' only; got calls={calls}"


# --------------------------------------------------------------------- #
# Payload shape: tier='all' --submit                                     #
# --------------------------------------------------------------------- #


def test_tier_all_submit_payload_shape():
    """``tier='all'`` + smoke/harness inputs build a v2-valid payload.

    Locked invariants:

    - ``schema_version == 2``
    - ``tier == 'all'``
    - ``smoke_result``, ``harness_result``, AND ``buckets`` all present
      (the schema requires ``buckets`` unconditionally — the speed
      bucket of a tier=all submission still comes from
      ``run_standardized_bench``).
    - Payload validates against ``community-benchmarks/schema.json``.
    """
    jsonschema = pytest.importorskip("jsonschema")
    from vllm_mlx.community_bench.submission import build_submission_payload

    hw, sw, bench = _stub_bench_inputs()
    payload = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes="PR5 tier=all combo",
        now=datetime(2026, 6, 17, 10, 30, 0, tzinfo=timezone.utc),
        tier="all",
        smoke_result=_SMOKE_PAYLOAD,
        harness_result=_HARNESS_PAYLOAD,
    )

    assert payload["schema_version"] == 2
    assert payload["tier"] == "all"
    assert payload["smoke_result"] == _SMOKE_PAYLOAD
    assert payload["harness_result"] == _HARNESS_PAYLOAD
    # The schema requires ``buckets`` for every submission (including
    # tier=smoke / harness / all). The speed numbers carry through
    # from the BenchResult we ran in --submit's phase 2.
    assert "buckets" in payload
    assert {"short", "long"} <= payload["buckets"].keys()

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=payload, schema=schema)


def test_tier_smoke_submit_populates_smoke_result_only():
    """``tier='smoke'`` + smoke_result yields a v2-valid payload with
    NO harness_result.

    The schema's ``allOf`` block requires ``smoke_result`` for
    ``tier='smoke'`` AND forbids passing ``harness_result`` when the
    tier doesn't include the harness bucket. The builder enforces
    that coupling locally so callers get a clean Python-side error
    rather than a downstream schema-validation failure two layers up.

    Note: the schema also requires the top-level ``buckets`` field
    on every submission (PR #3's design: speed numbers always
    accompany a community-bench row). So a tier=smoke submission
    DOES still carry the standardized-bench buckets — the smoke
    result is additive, not replacement. The "no buckets" reading
    of the spec would contradict the locked schema.
    """
    jsonschema = pytest.importorskip("jsonschema")
    from vllm_mlx.community_bench.submission import build_submission_payload

    hw, sw, bench = _stub_bench_inputs()
    payload = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes=None,
        now=datetime(2026, 6, 17, tzinfo=timezone.utc),
        tier="smoke",
        smoke_result=_SMOKE_PAYLOAD,
    )

    assert payload["tier"] == "smoke"
    assert payload["smoke_result"] == _SMOKE_PAYLOAD
    assert "harness_result" not in payload, (
        "tier=smoke MUST NOT emit harness_result (schema forbids it)"
    )

    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=payload, schema=schema)


def test_tier_harness_submit_populates_harness_result_only():
    """``tier='harness'`` + harness_result yields a v2-valid payload with
    NO smoke_result.
    """
    jsonschema = pytest.importorskip("jsonschema")
    from vllm_mlx.community_bench.submission import build_submission_payload

    hw, sw, bench = _stub_bench_inputs()
    payload = build_submission_payload(
        hardware=hw,
        software=sw,
        alias="qwen3.5-9b-4bit",
        hf_path="mlx-community/Qwen3.5-9B-4bit",
        bench=bench,
        notes=None,
        now=datetime(2026, 6, 17, tzinfo=timezone.utc),
        tier="harness",
        harness_result=_HARNESS_PAYLOAD,
    )

    assert payload["tier"] == "harness"
    assert payload["harness_result"] == _HARNESS_PAYLOAD
    assert "smoke_result" not in payload
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=payload, schema=schema)


# --------------------------------------------------------------------- #
# CLI surface: mutual-exclusive guard removed                            #
# --------------------------------------------------------------------- #


def test_mutual_exclusive_guard_removed():
    """``bench --tier all --submit`` MUST NOT hard-fail at parse time.

    PR #2 left an exit-2 guard in ``bench_command`` with a comment
    saying "PR #3 will unify them". PR #5 removes it. The guarantee
    we're locking is that the dispatcher routes both flags into the
    combined flow without the old error message ever printing.

    We exercise this via ``main()`` (the real entry point) with a
    stub ``bench_command`` so the parser path is unmodified — that's
    the only way to catch a regression where someone re-introduces
    the guard inside an argparse hook.
    """
    import sys as _sys

    from vllm_mlx import cli as _cli

    captured = {}

    def _fake_bench_command(args):
        captured["called"] = True
        captured["tier"] = getattr(args, "tier", None)
        captured["submit"] = getattr(args, "submit", False)

    old_bench = _cli.bench_command
    old_argv = _sys.argv
    _cli.bench_command = _fake_bench_command
    _sys.argv = [
        "rapid-mlx",
        "bench",
        "qwen3.5-9b-4bit",
        "--tier",
        "all",
        "--submit",
        "--notes",
        "x",
    ]
    try:
        # main() may sys.exit; capture it.
        try:
            _cli.main()
        except SystemExit as e:
            assert e.code in (None, 0), (
                f"main() exited with non-success code {e.code} — "
                "the mutual-exclusive guard appears to still be in place"
            )
    finally:
        _cli.bench_command = old_bench
        _sys.argv = old_argv

    assert captured.get("called") is True, (
        "main() must dispatch to bench_command when --tier and --submit "
        "are both set (guard removal regression)"
    )
    assert captured["tier"] == "all"
    assert captured["submit"] is True


def test_tier_submit_refuses_base_url(capsys):
    """``--base-url`` MUST be rejected for the --submit combo.

    Regression coverage for codex PR #623 BLOCKING-1: when the user
    attaches to a pre-existing server we never measured boot, AND the
    in-process standardized bench would run against a separately-
    loaded engine — so the submitted payload would mislabel both
    smoke and speed numbers. The right call is to fail fast with a
    clear error pointing them at the supported workflow (drop
    --base-url and let --submit boot the server itself).
    """
    import argparse

    from vllm_mlx.cli import _run_tier_submit_flow

    args = argparse.Namespace(
        model="qwen3.5-9b-4bit",
        tier="all",
        submit=True,
        base_url="http://127.0.0.1:8000",
        sampled=False,
        force_disk_check=False,
        notes=None,
        repo_root=None,
    )

    rc = _run_tier_submit_flow(args)
    assert rc == 2, "--base-url with --submit MUST exit 2 (setup error)"

    captured = capsys.readouterr()
    assert "--base-url is incompatible with --submit" in captured.err, (
        "error message must explain WHY --base-url is incompatible "
        "with --submit (boot_time_ms + standardized bench correctness)"
    )


def test_smoke_payload_is_none_when_boot_time_unknown(capsys):
    """``_run_smoke`` with ``boot_time_ms=None`` MUST set payload=None.

    Regression coverage for codex PR #623 BLOCKING-1's defensive
    second layer: even if a future caller bypasses the
    ``--base-url`` + ``--submit`` guard in cli.py, ``_run_smoke``
    itself refuses to invent a ``0.0`` boot-time placeholder. The
    schema's ``boot_time_ms: number, minimum: 0`` would happily
    accept ``0.0`` but the aggregator can't distinguish that from
    "machine boots this model in zero ms" — so the producer fails
    closed and lets the caller decide what to do with the missing
    metric.
    """
    from vllm_mlx.bench.tier_runner import _run_smoke

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "test-model"}]}

    class _FakeStreamResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"4"}}]}',
                    "data: [DONE]",
                ]
            )

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _FakeResp()

        def stream(self, method, url, json=None):
            return _FakeStreamResp()

    with patch("httpx.Client", lambda *a, **k: _FakeClient()):
        r = _run_smoke(
            model="qwen3.5-4b-4bit",
            base_url="http://127.0.0.1:8500/v1",
            boot_time_ms=None,
        )

    assert r.passed is True, "smoke should still PASS — '4' is in the response"
    assert r.payload is None, (
        "boot_time_ms=None MUST yield payload=None (defensive coverage of "
        "PR #623 BLOCKING-1; never invent a 0.0 boot-time)"
    )


def test_smoke_passes_when_only_reasoning_content_streams():
    """``_run_smoke`` MUST recognize ``delta.reasoning_content``.

    Reasoning-parser models (qwen3, gemma4, deepseek_r1) emit their
    ``<think>...</think>`` block as ``delta.reasoning_content`` instead
    of ``delta.content`` — at small max_tokens budgets they can finish
    the reasoning block while still inside it, never producing a
    content delta at all. Pre-fix behavior on v0.7.26 was: smoke
    FAIL on every reasoning-parser model with empty response, because
    the probe only inspected ``delta.content``. qwen3-0.6b-4bit and
    qwen3-8b-4bit both surfaced this during release dogfood.

    This test pins the new behavior: smoke PASSES when the answer ("4")
    appears in ``reasoning_content``, the payload's ``response_excerpt``
    is the reasoning text tagged with ``[reasoning]``, and TTFT is
    measured from the first reasoning chunk.
    """
    from vllm_mlx.bench.tier_runner import _run_smoke

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "qwen3-0.6b-4bit"}]}

    class _FakeStreamResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            # Pure reasoning stream, no content deltas at all — what
            # qwen3-0.6b-4bit actually emits for "what is 2+2" at
            # max_tokens=64. We split the answer "4" across two chunks
            # so the test also catches accidental "first chunk only"
            # regressions.
            return iter(
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"Let me think. "}}]}',
                    'data: {"choices":[{"delta":{"reasoning_content":"2+2 = 4."}}]}',
                    "data: [DONE]",
                ]
            )

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _FakeResp()

        def stream(self, method, url, json=None):
            return _FakeStreamResp()

    with patch("httpx.Client", lambda *a, **k: _FakeClient()):
        r = _run_smoke(
            model="qwen3-0.6b-4bit",
            base_url="http://127.0.0.1:8500/v1",
            boot_time_ms=2500.0,
        )

    assert r.passed is True, (
        "smoke MUST pass when '4' is in reasoning_content — reasoning-parser "
        "models legitimately answer inside <think>...</think>"
    )
    assert r.payload is not None
    assert r.payload["first_prompt_ok"] is True
    assert r.payload["first_token_latency_ms"] > 0, (
        "TTFT must be measured from the first reasoning chunk, not 0.0"
    )
    assert "[reasoning]" in r.payload["response_excerpt"], (
        "When only reasoning content streamed, response_excerpt MUST be "
        "tagged so the corpus / log reader knows it came from "
        "delta.reasoning_content, not delta.content"
    )
    assert "4" in r.payload["response_excerpt"]


def test_tier_submit_routes_through_unified_flow(monkeypatch):
    """``bench_command`` MUST route --tier+--submit to ``_run_tier_submit_flow``.

    We monkeypatch the combined flow to capture the call, then build a
    minimal Namespace and invoke ``bench_command`` directly. Proves
    the bench dispatcher takes the new combo branch BEFORE either the
    bare ``_run_submit_flow`` or bare ``run_tier`` branches — order
    matters because both single-flag paths would otherwise grab
    half the work each and leave the user with a half-built payload.
    """
    import argparse
    import importlib

    cli = importlib.import_module("vllm_mlx.cli")

    captured = {}

    def _fake_combo(args):
        captured["called"] = True
        captured["tier"] = args.tier
        captured["submit"] = args.submit
        return 0

    monkeypatch.setattr(cli, "_run_tier_submit_flow", _fake_combo)

    args = argparse.Namespace(
        model="qwen3.5-9b-4bit",
        tier="all",
        submit=True,
        base_url=None,
        sampled=False,
        force_disk_check=False,
        notes=None,
        repo_root=None,
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.bench_command(args)
    assert excinfo.value.code == 0
    assert captured.get("called") is True, (
        "bench_command MUST route --tier+--submit to the combined flow"
    )
    assert captured["tier"] == "all"
    assert captured["submit"] is True
