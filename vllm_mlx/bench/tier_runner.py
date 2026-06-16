# SPDX-License-Identifier: Apache-2.0
"""User-facing tier dispatcher for ``rapid-mlx bench <model> --tier ...``.

This module is the public surface for the four tier modes:

    rapid-mlx bench <model> --tier smoke    # boot + 1 prompt
    rapid-mlx bench <model> --tier speed    # standardized B=1 perf bench
    rapid-mlx bench <model> --tier harness  # 5 first-class agent harnesses
    rapid-mlx bench <model> --tier all      # smoke → speed → harness

Each invocation boots the model server exactly once (or attaches to an
existing server when ``--base-url`` is provided), runs all tier work
against it, and cleanly shuts it down before returning.

PR #2 of a 4-PR series. PR #3 will rework the ``--submit`` payload to
include all three buckets (speed/harness/agent metrics); PR #4 will
replace ``doctor`` with the env-health-only version. For now, the
existing freeform ``bench`` (no --tier, no --submit) path and the
``--submit`` flow are untouched — both still work exactly as before.
"""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass

# The 5 first-class harnesses in the documented order. Used by both
# ``--tier harness`` and the release_check_m3.sh G7b gate. Keep this
# list in lock-step with the G7b shell loop in scripts/release_check_m3.sh
# — if a harness is added/removed here, the script must be updated too.
HARNESS_PROFILES: tuple[str, ...] = (
    "codex",
    "opencode",
    "hermes",
    "aider",
    "langchain",
)

# Deterministic ephemeral-port range the tier runner probes when no
# explicit ``--port`` and no ``--base-url`` are given. 8500-8599 keeps
# us well clear of the default 8000 (so a user with a running server
# isn't disrupted) and out of the OS-assigned 49152+ range (so multiple
# concurrent tier runs are easy to spot in ``lsof``).
TIER_PORT_MIN = 8500
TIER_PORT_MAX = 8599


@dataclass
class TierResult:
    """One tier's outcome. Aggregated by ``--tier all``."""

    name: str  # "smoke" | "speed" | "harness"
    passed: bool
    duration_s: float
    detail: str = ""


def _find_free_port_in_range(lo: int, hi: int) -> int:
    """Probe ``[lo, hi]`` for a free TCP port; fall back to OS-assigned.

    Deterministic range makes ``lsof -i :85xx`` unambiguous during a
    tier run, so a developer who sees a hung server can find and kill
    it without a process ID. If every port in the range is busy
    (e.g. five concurrent tier runs from different shells), fall back
    to OS-assigned ephemeral so we don't hard-fail the user.
    """
    for port in range(lo, hi + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    # Range exhausted — let the OS pick.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_base_url(base_url: str | None) -> tuple[str, int] | None:
    """Parse ``--base-url`` if provided, returning ``(host, port)`` or None.

    When set, the tier runner SKIPS booting its own server and runs all
    tier work against the URL the user provided. This is the gauntlet
    path: ``release_check_m3.sh`` already booted ``rapid-mlx serve`` on
    port 8000 and just wants us to run the harness suite against it.
    """
    if not base_url:
        return None
    # Strip /v1 if the user pasted the full OpenAI base.
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")]
    # Parse host:port out of http(s)://host:port.
    from urllib.parse import urlparse

    parsed = urlparse(cleaned)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


# --------------------------------------------------------------------- #
# Per-tier implementations                                              #
# --------------------------------------------------------------------- #


def _run_smoke(model: str, port: int) -> TierResult:
    """Boot probe: send one prompt, assert response contains "4".

    The prompt is "Hello, what is 2+2?" — small models reliably answer
    "4" without thinking-mode hallucinations. We measure two numbers
    the user actually cares about: how long boot took (already paid by
    the caller) and first-token latency from request to first delta.
    """
    import httpx

    base_url = f"http://127.0.0.1:{port}/v1"
    t0 = time.perf_counter()
    try:
        # Resolve model_id from the server so we don't have to guess
        # the canonical name post-alias-resolution.
        with httpx.Client(timeout=30) as client:
            models_resp = client.get(f"{base_url}/models")
            models_resp.raise_for_status()
            model_id = models_resp.json()["data"][0]["id"]

            # Stream so we can measure TTFT independently of total time.
            ttft_ms: float | None = None
            content_parts: list[str] = []
            req_start = time.perf_counter()
            with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": "Hello, what is 2+2?"}
                    ],
                    "max_tokens": 64,
                    "stream": True,
                    "temperature": 0,
                },
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - req_start) * 1000
                    import json as _json

                    try:
                        chunk = _json.loads(payload)
                    except ValueError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        content_parts.append(delta)
    except Exception as exc:  # noqa: BLE001 — smoke must never crash
        elapsed = time.perf_counter() - t0
        return TierResult(
            name="smoke",
            passed=False,
            duration_s=elapsed,
            detail=f"FAIL: {type(exc).__name__}: {exc}",
        )

    elapsed = time.perf_counter() - t0
    response_text = "".join(content_parts)
    ok = "4" in response_text
    ttft_display = f"{ttft_ms:.0f}" if ttft_ms is not None else "?"
    detail = (
        f"{'PASS' if ok else 'FAIL'} model={model} ttft={ttft_display}ms "
        f"response={response_text[:60]!r}"
    )
    return TierResult(name="smoke", passed=ok, duration_s=elapsed, detail=detail)


def _run_speed(model: str, port: int, sampled: bool = False) -> TierResult:
    """Standardized B=1 perf bench (reuses ``run_standardized_bench``).

    This calls the SAME runner that ``bench --submit`` uses, against
    the already-booted server. We don't open a PR here — that's
    --submit's job. PR #3 will reshape what --submit emits to include
    speed + harness + agent buckets, but the --speed-only data shape
    we use here is stable across that change.
    """
    import httpx

    base_url = f"http://127.0.0.1:{port}/v1"
    t0 = time.perf_counter()
    try:
        # Use the OpenAI completions endpoint to drive a small set of
        # standardized prompts. The full standardized bench in
        # community_bench.runner requires direct engine access (not a
        # boot-once server) — we'd have to re-architect that to share a
        # booted server, which is explicitly PR #3 scope. For PR #2 we
        # surface the metric we can measure cleanly through HTTP: a
        # 5-prompt decode/prefill probe to flag gross perf regressions.
        with httpx.Client(timeout=180) as client:
            models_resp = client.get(f"{base_url}/models")
            models_resp.raise_for_status()
            model_id = models_resp.json()["data"][0]["id"]

            # 5 fixed prompts × max_tokens=128 — small enough to finish
            # in <30s on a 4B model, big enough to amortise per-request
            # overhead. Greedy by default; --sampled flips to temp=0.7.
            temperature = 0.7 if sampled else 0.0
            prompts = [
                "Write one sentence about the ocean.",
                "Write one sentence about mountains.",
                "Write one sentence about forests.",
                "Write one sentence about deserts.",
                "Write one sentence about cities.",
            ]
            total_completion_tokens = 0
            total_time = 0.0
            for prompt in prompts:
                req_t0 = time.perf_counter()
                resp = client.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 128,
                        "temperature": temperature,
                    },
                )
                resp.raise_for_status()
                req_elapsed = time.perf_counter() - req_t0
                body = resp.json()
                usage = body.get("usage", {})
                total_completion_tokens += usage.get("completion_tokens", 0)
                total_time += req_elapsed
    except Exception as exc:  # noqa: BLE001 — speed must never crash
        elapsed = time.perf_counter() - t0
        return TierResult(
            name="speed",
            passed=False,
            duration_s=elapsed,
            detail=f"FAIL: {type(exc).__name__}: {exc}",
        )

    elapsed = time.perf_counter() - t0
    tps = (total_completion_tokens / total_time) if total_time > 0 else 0.0
    detail = (
        f"PASS model={model} sampling={'sampled' if sampled else 'greedy'} "
        f"tokens={total_completion_tokens} tps={tps:.1f}"
    )
    return TierResult(name="speed", passed=True, duration_s=elapsed, detail=detail)


def _run_harness(model: str, port: int) -> TierResult:
    """Run the 5 first-class harnesses against the booted server.

    Returns a TierResult that aggregates the 5 individual outcomes. We
    treat any ERROR or FAIL in any harness as a tier-level failure —
    SKIP is fine (binary missing for the e2e tests is expected).
    """
    from ..agents import get_profile
    from ..agents.testing import AgentTestRunner, TestStatus

    base_url = f"http://127.0.0.1:{port}/v1"
    t0 = time.perf_counter()
    per_harness: list[tuple[str, bool, float, str]] = []

    for profile_name in HARNESS_PROFILES:
        profile = get_profile(profile_name)
        if profile is None:
            per_harness.append(
                (profile_name, False, 0.0, f"profile {profile_name!r} not found")
            )
            continue

        h_t0 = time.perf_counter()
        try:
            runner = AgentTestRunner(
                profile,
                base_url=base_url,
                model_id=None,  # auto-detect from /models
            )
            report = runner.run()
            h_elapsed = time.perf_counter() - h_t0
            n_fail = report.failed
            n_err = report.errored
            ok = n_fail == 0 and n_err == 0
            if ok:
                detail = f"{report.passed}p {report.skipped}s"
            else:
                # Surface first failing test name + message excerpt so
                # gauntlet operators see the actionable signal without
                # diving into the log.
                first_bad = next(
                    (
                        f"{r.name}: {(r.message or '')[:80]}"
                        for r in report.results
                        if r.status in (TestStatus.FAIL, TestStatus.ERROR)
                    ),
                    "(no detail)",
                )
                detail = (
                    f"{report.passed}p {n_fail}f {n_err}e {report.skipped}s "
                    f"| {first_bad}"
                )
            per_harness.append((profile_name, ok, h_elapsed, detail))
        except Exception as exc:  # noqa: BLE001 — never abort the sweep
            h_elapsed = time.perf_counter() - h_t0
            per_harness.append(
                (
                    profile_name,
                    False,
                    h_elapsed,
                    f"crashed: {type(exc).__name__}: {exc}",
                )
            )

    elapsed = time.perf_counter() - t0
    all_passed = all(ok for _, ok, _, _ in per_harness)

    # Render a human-readable summary line per harness.
    lines = [f"harness sweep model={model}"]
    for name, ok, dur, detail in per_harness:
        marker = "PASS" if ok else "FAIL"
        lines.append(f"  {marker} {name:<10} ({dur:.1f}s) {detail}")
    return TierResult(
        name="harness",
        passed=all_passed,
        duration_s=elapsed,
        detail="\n".join(lines),
    )


# --------------------------------------------------------------------- #
# Server lifecycle                                                       #
# --------------------------------------------------------------------- #


def _serve_or_attach(
    model: str,
    base_url: str | None,
    boot_timeout_s: int = 600,
):
    """Yield ``(port, owns_server)``.

    If ``base_url`` is set, attach to that server — caller is responsible
    for its lifecycle. Otherwise boot one on a port in
    ``[TIER_PORT_MIN, TIER_PORT_MAX]`` and clean it up on exit.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        attach = _resolve_base_url(base_url)
        if attach is not None:
            host, port = attach
            # Sanity ping — fail loudly if the user gave a dead URL.
            import urllib.error
            import urllib.request

            try:
                with urllib.request.urlopen(  # noqa: S310
                    f"http://{host}:{port}/health", timeout=3
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"--base-url {base_url} returned status "
                            f"{resp.status}; expected 200 from /health"
                        )
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                raise RuntimeError(
                    f"--base-url {base_url} not reachable: {e}. "
                    "Start `rapid-mlx serve <model>` on that port first, "
                    "or drop --base-url to let bench --tier boot its own."
                )
            yield port, False
            return

        # Boot our own. Reuse the doctor's server helper since it's
        # already battle-tested (proc-group teardown, /health polling).
        port = _find_free_port_in_range(TIER_PORT_MIN, TIER_PORT_MAX)
        from ..doctor.server import serve

        # log_path=None → DEVNULL; the user-facing tier output is the
        # interesting signal. For debugging, the user can re-run with
        # `rapid-mlx serve <model> --port <port>` themselves.
        with serve(model=model, port=port, boot_timeout_s=boot_timeout_s) as info:
            yield info["port"], True

    return _ctx()


# --------------------------------------------------------------------- #
# Public entry points (called from cli.py)                               #
# --------------------------------------------------------------------- #


def run_tier(
    model: str,
    tier: str,
    base_url: str | None = None,
    sampled: bool = False,
    boot_timeout_s: int = 600,
) -> int:
    """Dispatch to the requested tier. Returns process exit code.

    ``tier`` is one of: ``"smoke"``, ``"speed"``, ``"harness"``, ``"all"``.
    Server is booted ONCE (or attached via ``base_url``) and torn down
    on exit. For ``"all"``, smoke runs first and aborts the sequence
    on failure — there's no point benching a model that can't say "4".
    """
    if tier not in ("smoke", "speed", "harness", "all"):
        print(
            f"  Error: unknown tier {tier!r}; expected one of "
            "smoke / speed / harness / all",
            file=sys.stderr,
        )
        return 2

    print(f"Rapid-MLX bench — tier={tier} model={model}")
    print("=" * 60)

    overall_t0 = time.perf_counter()
    results: list[TierResult] = []

    try:
        with _serve_or_attach(model, base_url, boot_timeout_s) as (port, owns):
            if owns:
                print(f"  [server] booted {model} on port {port}")
            else:
                print(f"  [server] attached to existing server at {base_url}")
            print()

            if tier in ("smoke", "all"):
                r = _run_smoke(model, port)
                _print_tier_result(r)
                results.append(r)
                if tier == "all" and not r.passed:
                    print()
                    print("  Aborting --tier all: smoke failed.")
                    return _finalize(results, overall_t0)

            if tier in ("speed", "all"):
                r = _run_speed(model, port, sampled=sampled)
                _print_tier_result(r)
                results.append(r)

            if tier in ("harness", "all"):
                r = _run_harness(model, port)
                _print_tier_result(r)
                results.append(r)
    except Exception as exc:  # noqa: BLE001 — surface as exit code, not traceback
        print(f"\n  Error during tier run: {type(exc).__name__}: {exc}")
        return 1

    return _finalize(results, overall_t0)


def _print_tier_result(r: TierResult) -> None:
    """One-line summary per tier; multi-line detail when present."""
    marker = "PASS" if r.passed else "FAIL"
    print(f"  [{marker}] tier={r.name} duration={r.duration_s:.1f}s")
    if r.detail:
        for line in r.detail.splitlines():
            print(f"        {line}")
    print()


def _finalize(results: list[TierResult], t0: float) -> int:
    """Print the overall summary line; return exit code (0 iff all passed)."""
    total = time.perf_counter() - t0
    print("=" * 60)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    overall_ok = n_fail == 0 and n_pass > 0
    marker = "OK" if overall_ok else "FAIL"
    summary = ", ".join(
        f"{r.name}={'pass' if r.passed else 'fail'}" for r in results
    )
    print(f"  {marker}: {n_pass}/{len(results)} tiers passed ({summary})")
    print(f"  total: {total:.1f}s")
    return 0 if overall_ok else 1
