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

import os
import socket
import sys
import time
import urllib.error
import urllib.request
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


def _resolve_harness_profile_timeout() -> int:
    """Read the per-profile harness timeout from the environment.

    300s default is wide enough that a healthy harness on a 30B+ model
    still finishes (release-check observed ~120s p95 for codex on
    qwen3.5-9b) but tight enough that a true hang gets caught in well
    under the gauntlet's overall 10-min/model budget.

    Reading at MODULE LOAD honors the documented ``HARNESS_PROFILE_TIMEOUT_S``
    override without needing every caller to thread a kwarg through.
    Invalid / non-positive values fall back to 300 with a stderr warning
    rather than silently using the user's broken value (e.g. ``-1`` would
    cause every harness to "time out" on the very first thread.join()).
    """
    raw = os.environ.get("HARNESS_PROFILE_TIMEOUT_S")
    if raw is None:
        return 300
    try:
        val = int(raw)
        if val <= 0:
            raise ValueError(f"must be positive, got {val}")
        return val
    except ValueError as exc:
        print(
            f"  Warning: ignoring invalid HARNESS_PROFILE_TIMEOUT_S={raw!r} "
            f"({exc}); using 300s default",
            file=sys.stderr,
        )
        return 300


def _resolve_harness_profiles_filter() -> tuple[str, ...] | None:
    """Read an optional ``RAPID_MLX_HARNESS_PROFILES_FILTER`` env var.

    Comma-separated list of profile names to scope the harness sweep to
    (e.g. ``"codex,aider"`` runs only those two and skips
    opencode/hermes/langchain). Used by ``scripts/release_check_m3_random.py``
    (G12) to randomly subset the sweep per (model × harness) pick
    without ballooning gauntlet wall-clock.

    Returns ``None`` (no filter) when the env var is absent. When set:
    * Unknown profile names → stderr warning + dropped from the run.
    * All names unknown → stderr warning + None (no filter — abort the
      filter rather than silently sweep nothing).
    * Empty value or whitespace-only → stderr warning + None.

    Reading at MODULE LOAD mirrors the timeout env above so callers
    don't have to thread a kwarg through ``--tier harness``.
    """
    raw = os.environ.get("RAPID_MLX_HARNESS_PROFILES_FILTER")
    if raw is None:
        return None
    requested = tuple(name.strip() for name in raw.split(",") if name.strip())
    if not requested:
        print(
            "  Warning: RAPID_MLX_HARNESS_PROFILES_FILTER is empty/whitespace; "
            "ignoring filter and running all profiles.",
            file=sys.stderr,
        )
        return None
    known = set(HARNESS_PROFILES)
    valid = tuple(name for name in requested if name in known)
    invalid = tuple(name for name in requested if name not in known)
    if invalid:
        print(
            f"  Warning: RAPID_MLX_HARNESS_PROFILES_FILTER includes "
            f"unknown profile(s) {invalid!r}; valid profiles are "
            f"{HARNESS_PROFILES}.",
            file=sys.stderr,
        )
    if not valid:
        print(
            "  Warning: RAPID_MLX_HARNESS_PROFILES_FILTER matched zero "
            "valid profiles; ignoring filter and running all profiles.",
            file=sys.stderr,
        )
        return None
    return valid


# Per-profile wall-clock cap for the harness sweep. One bad harness
# (e.g. a codex e2e_file_read hang at 156s on a slow model) was taking
# down the in-process server and cascading FAIL to every later profile
# with ECONNREFUSED / "Rapid-MLX server not running". Override via
# ``HARNESS_PROFILE_TIMEOUT_S`` env var for slow boxes / future bigger
# models. Resolved at module-load via ``_resolve_harness_profile_timeout``.
HARNESS_PROFILE_TIMEOUT_S = _resolve_harness_profile_timeout()

# Optional comma-separated subset of ``HARNESS_PROFILES`` to scope a
# sweep to. None = no filter (sweep all 5). Resolved at module-load via
# ``_resolve_harness_profiles_filter``. See that helper for env-var
# semantics and edge-case handling.
HARNESS_PROFILES_FILTER: tuple[str, ...] | None = _resolve_harness_profiles_filter()

# Server health-probe timeout used between harness profiles. The probe
# is a single GET /health — we don't want it to itself hang and look
# like a profile failure, so keep it short.
_HEALTH_PROBE_TIMEOUT_S = 3


@dataclass
class TierResult:
    """One tier's outcome. Aggregated by ``--tier all``.

    ``payload`` is the schema-v2-shaped sub-object the
    community-bench submission would attach for this tier:

    - smoke → matches the schema's ``smoke_result`` object
      (boot_time_ms / first_prompt_ok / first_token_latency_ms /
      response_excerpt).
    - harness → matches the schema's ``harness_result`` object (the
      per-adapter mapping with passed/duration_s/error_excerpt).
    - speed → ``None`` here; the schema-shaped speed buckets come
      from ``run_standardized_bench`` (PR #5's --submit path), not
      from this HTTP-driven dispatcher. The lightweight
      tier-speed probe is intentionally NOT submitted to
      community-benchmarks because its numbers aren't comparable to
      the locked B=1 protocol.

    Populated unconditionally so ``--tier ... --submit`` can read the
    same dict that the non-submit human-readable path already prints,
    with no extra round-trip. Default ``None`` preserves the legacy
    callsite signature for any caller that doesn't care about the
    submission payload.
    """

    name: str  # "smoke" | "speed" | "harness"
    passed: bool
    duration_s: float
    detail: str = ""
    payload: dict | None = None


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

    Strips trailing ``/v1`` if the user pasted the full OpenAI base;
    accepts both ``http://h:p`` and ``http://h:p/v1`` forms.
    """
    if not base_url:
        return None
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")]
    from urllib.parse import urlparse

    parsed = urlparse(cleaned)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _normalize_openai_base(base_url: str | None, port: int) -> str:
    """Return a fully-qualified ``http(s)://host:port/v1`` to hand to tiers.

    When the user passed ``--base-url`` we honor scheme/host from their
    URL (codex review #621 BLOCKING: previous revision only forwarded
    ``port`` so a non-localhost host or https scheme was silently
    discarded). When no ``--base-url`` was given, we constructed a
    local-loopback URL against ``port`` (the boot port).
    """
    if base_url:
        from urllib.parse import urlparse

        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/v1"):
            cleaned = cleaned[: -len("/v1")]
        parsed = urlparse(cleaned)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        # If the URL had no explicit port, fall back to the one we
        # resolved (HTTP default 80, HTTPS default 443).
        resolved_port = parsed.port or port
        return f"{scheme}://{host}:{resolved_port}/v1"
    return f"http://127.0.0.1:{port}/v1"


# --------------------------------------------------------------------- #
# Per-tier implementations                                              #
# --------------------------------------------------------------------- #


def _run_smoke(
    model: str,
    base_url: str,
    boot_time_ms: float | None = None,
) -> TierResult:
    """Boot probe: send one prompt, assert response contains "4".

    The prompt is "Hello, what is 2+2?" — small models reliably answer
    "4" without thinking-mode hallucinations. We measure two numbers
    the user actually cares about: how long boot took (already paid by
    the caller) and first-token latency from request to first delta.

    ``base_url`` is the normalized OpenAI base (e.g. ``http://host:port/v1``).
    The full URL is honored end-to-end — earlier revisions only forwarded
    the port to each tier, which meant ``--base-url`` against a non-localhost
    host (or a https scheme) was silently ignored (codex review #621
    BLOCKING).

    ``boot_time_ms`` is threaded through from ``_serve_or_attach`` so
    the schema v2 ``smoke_result.boot_time_ms`` field carries the real
    spawn-to-healthy wall-clock. When ``None`` (the user attached via
    ``--base-url``, we never measured the boot they paid for) the
    returned ``TierResult.payload`` is itself ``None`` — fails closed
    rather than invent a ``0.0`` placeholder that the corpus can't
    tell apart from "machine boots this model in zero ms" (Codex
    PR #623 BLOCKING-1). The CLI's ``_run_tier_submit_flow`` refuses
    ``--base-url`` for ``--submit`` so this null-payload case can
    only surface via the non-submit ``--tier smoke --base-url``
    path where the payload is never consumed.
    """
    import httpx

    t0 = time.perf_counter()
    try:
        # Resolve model_id from the server so we don't have to guess
        # the canonical name post-alias-resolution.
        with httpx.Client(timeout=30) as client:
            models_resp = client.get(f"{base_url}/models")
            models_resp.raise_for_status()
            model_id = models_resp.json()["data"][0]["id"]

            # Stream so we can measure TTFT independently of total time.
            # We measure TTFT on the first chunk that carries non-empty
            # CONTENT, not the first SSE line — many providers emit a
            # role-only initial chunk (``{"delta": {"role": "assistant"}}``)
            # that would otherwise understate TTFT (codex review #621 NIT).
            ttft_ms: float | None = None
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            req_start = time.perf_counter()
            # max_tokens=256 (was 64): qwen3 / gemma4 / deepseek_r1 reasoning
            # parsers route the first chunk of tokens into ``<think>...</think>``
            # and emit them as ``delta.reasoning_content`` rather than
            # ``delta.content``. A 64-token budget could be entirely consumed
            # by the reasoning prefix on very small models (qwen3-0.6b-4bit
            # hit this — smoke failed with response='' even on a healthy
            # boot). 256 gives reasoning models room to finish thinking and
            # produce the actual answer without making the probe slow on
            # non-reasoning models (they stop at ~5-10 content tokens via
            # finish_reason=stop).
            with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
                    "max_tokens": 256,
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
                    import json as _json

                    try:
                        chunk = _json.loads(payload)
                    except ValueError:
                        continue
                    delta_obj = chunk.get("choices", [{}])[0].get("delta", {})
                    content_delta = delta_obj.get("content") or ""
                    # Reasoning-parser models (qwen3, gemma4, deepseek_r1)
                    # stream ``<think>...</think>`` into reasoning_content
                    # instead of content. We track both so the smoke probe
                    # passes for reasoning models AND measures TTFT from
                    # whichever stream fires first.
                    reasoning_delta = delta_obj.get("reasoning_content") or ""
                    if content_delta or reasoning_delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - req_start) * 1000
                        if content_delta:
                            content_parts.append(content_delta)
                        if reasoning_delta:
                            reasoning_parts.append(reasoning_delta)
    except Exception as exc:  # noqa: BLE001 — smoke must never crash
        elapsed = time.perf_counter() - t0
        return TierResult(
            name="smoke",
            passed=False,
            duration_s=elapsed,
            detail=f"FAIL: {type(exc).__name__}: {exc}",
            # Schema-shaped failure payload — diagnostic only under
            # current policy. ``_run_tier_submit_flow`` aborts BEFORE
            # consuming this payload when smoke fails (no point
            # benching a model that can't say "4"), so the populated
            # failure payload here is for the human-readable tier
            # output and for any future caller that chooses to
            # surface a failure-row to the corpus. Emitted ONLY when
            # we actually measured the boot ourselves; when
            # ``boot_time_ms`` is ``None`` (--base-url attach path)
            # we set ``payload=None`` instead of inventing a ``0.0``
            # placeholder that the corpus can't tell apart from
            # "machine boots in zero ms" (Codex PR #623 BLOCKING-1,
            # review-round-2 NIT-3 clarified the policy).
            payload=(
                {
                    "boot_time_ms": float(boot_time_ms),
                    "first_prompt_ok": False,
                    "first_token_latency_ms": 0.0,
                    "response_excerpt": (f"[error] {type(exc).__name__}: {exc}"[:200]),
                }
                if boot_time_ms is not None
                else None
            ),
        )

    elapsed = time.perf_counter() - t0
    response_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    # Pass if either stream produced "4" — reasoning models may answer
    # entirely inside ``<think>`` for short prompts, and that's still a
    # healthy boot.
    ok = "4" in response_text or "4" in reasoning_text
    # If only reasoning content streamed (no content delta at all), use
    # it as the displayed excerpt so the user can see what the model
    # actually said. Prefix with ``[reasoning]`` so the source is
    # obvious in the corpus / logs.
    if not response_text and reasoning_text:
        display_text = f"[reasoning] {reasoning_text}"
    else:
        display_text = response_text
    ttft_display = f"{ttft_ms:.0f}" if ttft_ms is not None else "?"
    detail = (
        f"{'PASS' if ok else 'FAIL'} model={model} ttft={ttft_display}ms "
        f"response={display_text[:60]!r}"
    )
    return TierResult(
        name="smoke",
        passed=ok,
        duration_s=elapsed,
        detail=detail,
        # Schema-v2 ``smoke_result``. ``response_excerpt`` is capped at
        # 200 chars (schema maxLength); the truncation happens here so
        # the displayed-vs-submitted bytes match exactly. TTFT defaults
        # to 0.0 when the stream emitted no content delta (first_prompt
        # was not ok in that case). When ``boot_time_ms`` is ``None``
        # (--base-url attach path) we OMIT the payload entirely — see
        # the matching comment in the failure branch for the rationale.
        payload=(
            {
                "boot_time_ms": float(boot_time_ms),
                "first_prompt_ok": ok,
                "first_token_latency_ms": (
                    float(ttft_ms) if ttft_ms is not None else 0.0
                ),
                "response_excerpt": display_text[:200],
            }
            if boot_time_ms is not None
            else None
        ),
    )


def _run_speed(model: str, base_url: str, sampled: bool = False) -> TierResult:
    """Standardized B=1 perf bench (reuses ``run_standardized_bench``).

    This calls the SAME runner that ``bench --submit`` uses, against
    the already-booted server. We don't open a PR here — that's
    --submit's job. PR #3 will reshape what --submit emits to include
    speed + harness + agent buckets, but the --speed-only data shape
    we use here is stable across that change.

    ``base_url`` is the normalized OpenAI base (e.g. ``http://host:port/v1``).
    """
    import httpx

    t0 = time.perf_counter()
    total_completion_tokens = 0
    total_content_chars = 0
    total_time = 0.0
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
                # Track content length too — some servers omit usage but
                # still return tokens, and an empty-content response is a
                # silent regression we MUST flag (codex review #621
                # BLOCKING: HTTP 200 with zero output is not a pass).
                content = (
                    body.get("choices", [{}])[0].get("message", {}).get("content") or ""
                )
                total_content_chars += len(content)
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
    # The server is unhealthy if it returned 200s but emitted nothing.
    # Either total_completion_tokens > 0 (usage path) OR total content
    # chars > 0 (content-only fallback) must hold; otherwise this is
    # the "silent empty response" regression class.
    if total_completion_tokens == 0 and total_content_chars == 0:
        detail = (
            f"FAIL model={model} sampling={'sampled' if sampled else 'greedy'} "
            "5/5 prompts returned no completion tokens AND empty content "
            "(server unhealthy or response shape broken)"
        )
        return TierResult(name="speed", passed=False, duration_s=elapsed, detail=detail)

    tps = (total_completion_tokens / total_time) if total_time > 0 else 0.0
    detail = (
        f"PASS model={model} sampling={'sampled' if sampled else 'greedy'} "
        f"tokens={total_completion_tokens} chars={total_content_chars} "
        f"tps={tps:.1f}"
    )
    return TierResult(name="speed", passed=True, duration_s=elapsed, detail=detail)


def _health_check(base_url: str, timeout_s: int = _HEALTH_PROBE_TIMEOUT_S) -> bool:
    """Single GET /health probe. Returns True iff the server answered 200.

    ``base_url`` is the normalized OpenAI base (e.g. ``http://host:port/v1``).
    The bench server's health route lives at ``/health`` (not under
    ``/v1``) — we derive it by stripping a trailing ``/v1``. Any error
    (connection refused, timeout, non-200) returns False — the caller
    interprets that as "server is dead, reboot it if we can".
    """
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")]
    health_url = f"{cleaned}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_s) as resp:  # noqa: S310
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


class _HarnessServerSession:
    """Per-harness server lifecycle for the harness sweep.

    Wraps either (a) a server we own and CAN restart between profiles
    when it dies, or (b) a user-attached server we can only health-check
    against. Exposes a single method ``ensure_healthy()`` that the
    harness loop calls before each profile.

    Why this exists: the harness sweep used to share one ``serve()``
    context manager with the rest of ``run_tier``. If one profile's e2e
    subtest hung long enough to OOM / crash the in-process server (real
    incident: codex e2e_file_read on gemma3-4-12b, qwen3.5-9b — the
    server died, and every later profile then failed with
    ``Rapid-MLX server not running``), nothing brought the server back
    up. Cascade fail. Now a per-profile health check catches the dead
    server and the session reboots before the next profile runs.
    """

    def __init__(
        self,
        model: str,
        boot_timeout_s: int,
        initial_port: int,
        initial_owns: bool,
        initial_base_url: str,
        release_slot,
    ):
        self._model = model
        self._boot_timeout_s = boot_timeout_s
        self._port = initial_port
        self._owns = initial_owns
        self._base_url = initial_base_url
        # ``_release_slot`` is the single-key dict from
        # ``_serve_or_attach``; ``_release_slot["current"]`` always
        # holds the zero-arg release for the LIVE server. On every
        # restart, ``_restart`` reads the slot to kill the current
        # server then writes the replacement's release back into the
        # slot — so the outer ``_serve_or_attach`` finally always
        # targets the live server. ``None`` in attach mode.
        self._release_slot = release_slot
        self._restarts = 0
        # When ``True``, the session refuses all further reboots — set
        # after a single teardown failure to guarantee at most one model
        # server alive at any time (Codex review-3 BLOCKING).
        self._reboot_disabled = False

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def owns(self) -> bool:
        return self._owns

    @property
    def restarts(self) -> int:
        return self._restarts

    def ensure_healthy(self) -> tuple[bool, str | None]:
        """Probe /health; restart if dead and we own the server.

        Returns ``(healthy_now, note)``:

        - ``healthy_now`` — True iff the server is responsive (after a
          possible restart). False if the server is dead AND we can't
          restart it (e.g. user-attached via ``--base-url``).
        - ``note`` — human-readable string for tier output describing
          what we did (restart count, "attached so could not restart",
          etc.) or ``None`` when no action was needed.

        The caller treats ``healthy_now=False`` as "skip this profile
        and mark it failed" — running a profile against a dead server
        just generates a noisy ECONNREFUSED report.
        """
        if _health_check(self._base_url):
            return True, None

        if not self._owns:
            return (
                False,
                "server unreachable and --base-url attached "
                "(cannot restart attached servers)",
            )

        # We own it — tear down and reboot.
        return self._restart()

    def force_restart_after_timeout(self) -> tuple[bool, str | None]:
        """Reboot the server unconditionally after a per-profile timeout.

        The orphaned daemon thread from a timed-out profile may still be
        mid-request against the current server (an httpx call that hasn't
        returned, a subprocess that hasn't finished). If we let the NEXT
        profile share that same server, the orphan's late response would
        race against the new profile's measurements / failure modes.
        Force a fresh server so the next profile starts on a clean slate.

        No-op (returns ``(True, None)``) when we don't own the server —
        we can't restart what the user is hosting, so the next profile
        will simply contend with whatever ghost requests the orphan
        emits. Surface a clear note in that case so the gauntlet
        operator knows the next profile's numbers are suspect.
        """
        if not self._owns:
            return (
                True,
                "profile timed out and --base-url is attached — "
                "subsequent profile measurements may be polluted by the "
                "orphaned worker",
            )
        return self._restart()

    def _restart(self) -> tuple[bool, str | None]:
        # If a previous teardown failed, the OLD server may still be
        # alive — we have no reliable way to tell. The safest stance is
        # to refuse further reboots for the rest of the sweep so we
        # never accidentally run two model servers on the GPU at once.
        # ``ensure_healthy`` will then surface server-not-healthy FAILs
        # for the remaining profiles instead of trying to recover into
        # an unsafe state (Codex review-3 BLOCKING).
        if self._reboot_disabled:
            return (
                False,
                "reboot disabled after a prior teardown failure — "
                "cannot guarantee single-server invariant",
            )

        # Tear down the currently-live owned server BEFORE booting the
        # replacement. The slot's ``current`` entry was either populated
        # by ``_serve_or_attach`` (first restart) or by this method's
        # own swap on a prior restart. Either way, calling it kills
        # exactly the live server — no coexistence window (Codex
        # review-2 BLOCKING).
        old_port = self._port
        if self._release_slot is not None:
            cb = self._release_slot.get("current")
            if cb is not None:
                try:
                    cb()
                except Exception as exc:  # noqa: BLE001
                    # Codex review-5 BLOCKING-A: don't clear the slot
                    # BEFORE the teardown attempt — if it fails, the
                    # outer ``_serve_or_attach`` finalizer must still
                    # have a callback to try. Leave ``cb`` registered
                    # so the final ``_release_current_server`` swallow
                    # path retries the teardown (subsequent SIGTERMs
                    # are cheap; the proc-group teardown is idempotent
                    # via ``ProcessLookupError`` swallowing). Disable
                    # future reboots in THIS sweep so we don't stack a
                    # second engine on whatever zombies the failure
                    # left.
                    self._reboot_disabled = True
                    note = (
                        f"refused to reboot from port {old_port}: "
                        f"teardown of old server raised "
                        f"{type(exc).__name__}: {exc}. Skipping reboot "
                        f"to avoid running two model servers concurrently. "
                        f"Outer finalizer will retry teardown on tier exit"
                    )
                    return False, note
                # Teardown succeeded — now safe to clear the slot.
                self._release_slot["current"] = None

        # New port — old one may still be in TIME_WAIT, and a fresh
        # port keeps lsof unambiguous if the user is watching.
        from ._server import serve

        new_port = _find_free_port_in_range(TIER_PORT_MIN, TIER_PORT_MAX)
        ctx = serve(
            model=self._model, port=new_port, boot_timeout_s=self._boot_timeout_s
        )
        try:
            info = ctx.__enter__()
        except Exception as exc:  # noqa: BLE001 — failed reboot must surface
            note = f"reboot from port {old_port} failed: {type(exc).__name__}: {exc}"
            return False, note

        released = {"done": False}

        def _release_replacement() -> None:
            # As with ``_release_initial``, don't swallow — the next
            # restart needs the signal to refuse a follow-up reboot if
            # this teardown fails.
            if released["done"]:
                return
            released["done"] = True
            ctx.__exit__(None, None, None)

        # Hand the new release back to the slot so the outer
        # ``_serve_or_attach`` finally tears down the REPLACEMENT, not
        # the dead original (Codex review-3 BLOCKING).
        if self._release_slot is not None:
            self._release_slot["current"] = _release_replacement
        self._port = info["port"]
        self._base_url = _normalize_openai_base(None, self._port)
        self._restarts += 1
        note = (
            f"server was unhealthy — rebooted on port {self._port} "
            f"(restart #{self._restarts})"
        )
        return True, note


def _run_single_profile(
    profile_name: str,
    profile,
    base_url: str,
    timeout_s: int,
) -> tuple[bool, float, str, bool]:
    """Run ONE harness profile with a wall-clock cap. Never raises.

    Returns ``(ok, elapsed_s, detail, timed_out)`` — the trailing flag
    lets the caller force a server restart after a timeout so the
    orphaned daemon thread can't pollute the next profile's
    measurements with late-arriving requests.

    The runner is dispatched on a worker thread and joined with a
    ``timeout_s`` deadline. On timeout we abandon the thread (Python
    threads can't be force-killed) and surface a tier-level FAIL with
    a "timed out" detail.

    KNOWN LIMITATION — codex review-5 BLOCKING-B (acknowledged as a
    followup beyond this PR's scope): an abandoned daemon worker can
    keep running its in-flight subprocess / HTTP call after we move on.
    Bounded mitigations are already in play:

    - All subprocesses launched via ``_agent_query`` carry their own
      ``subprocess.run(..., timeout=...)`` (the harness profile's
      ``testing.query_timeout``, default 120s). So the worst-case
      lifetime of an orphan subprocess is bounded.
    - Every ``httpx`` call in ``AgentTestRunner`` carries an explicit
      timeout (30s for API checks; 60-180s for streaming).
    - The forced server restart after a timeout cuts the network
      connection the orphan was using — most HTTP calls error out
      within seconds of that.
    - The worker thread itself is ``daemon=True`` so process exit
      reaps it regardless.

    A complete fix requires running each profile in a child Python
    process with OS-level signal-killing — substantial refactor. Filed
    as a followup; the current per-profile budget + forced restart
    combo handles the production case (codex e2e_file_read hang on
    qwen3.5-9b) cleanly.
    """
    from ..agents.testing import AgentTestRunner, TestStatus

    h_t0 = time.perf_counter()

    def _run() -> tuple[bool, str]:
        runner = AgentTestRunner(
            profile,
            base_url=base_url,
            model_id=None,  # auto-detect from /models
        )
        report = runner.run()
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
                f"{report.passed}p {n_fail}f {n_err}e {report.skipped}s | {first_bad}"
            )
        return ok, detail

    # Use a daemon ``threading.Thread`` directly rather than
    # ``ThreadPoolExecutor`` — the latter's worker threads are
    # NON-daemon by default, so an orphaned hung worker would block the
    # bench CLI from exiting cleanly when ``run_tier`` returns. We
    # really do want the worker to be killable-by-interpreter-exit if
    # it's hung on an HTTP call that never returns; that's the only
    # safe escape hatch on top of the per-profile deadline.
    import threading

    result_container: dict[str, object] = {}

    def _runner_target() -> None:
        # Catch ``Exception`` (NOT ``BaseException``): leaking
        # ``KeyboardInterrupt`` / ``SystemExit`` here would let process
        # termination signals propagate through the worker thread up to
        # the bench CLI's signal handler, instead of being silently
        # demoted to a benchmark FAIL row. Worker threads can't actually
        # receive those signals (Python delivers them to the main
        # thread), but the symmetric exception handling is a Codex
        # review-2 NIT worth honoring.
        try:
            result_container["ok"], result_container["detail"] = _run()
        except Exception as exc:  # noqa: BLE001 — captured for the joiner
            result_container["exc"] = exc

    worker = threading.Thread(
        target=_runner_target,
        name=f"harness-{profile_name}",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        # Hang — abandon the daemon thread to die when (if ever) its
        # blocking call returns. Process exit will reap it because of
        # ``daemon=True``. Return ``timed_out=True`` so the caller
        # forces a server restart before the next profile, isolating
        # the orphan's late requests from the next profile's measurements.
        h_elapsed = time.perf_counter() - h_t0
        return (
            False,
            h_elapsed,
            f"timed out after {timeout_s}s (per-profile cap)",
            True,
        )

    h_elapsed = time.perf_counter() - h_t0
    if "exc" in result_container:
        exc = result_container["exc"]
        return (
            False,
            h_elapsed,
            f"crashed: {type(exc).__name__}: {exc}",
            False,
        )
    return (
        bool(result_container["ok"]),
        h_elapsed,
        str(result_container["detail"]),
        False,
    )


def _run_harness(
    model: str,
    base_url: str,
    session: _HarnessServerSession | None = None,
    per_profile_timeout_s: int | None = None,
) -> TierResult:
    """Run the 5 first-class harnesses against the booted server.

    Returns a TierResult that aggregates the 5 individual outcomes. We
    treat any ERROR or FAIL in any harness as a tier-level failure —
    SKIP is fine (binary missing for the e2e tests is expected).

    ``base_url`` is the normalized OpenAI base used as the FALLBACK when
    no ``session`` is supplied (kept for back-compat with existing unit
    tests and direct ``_run_harness`` callers). When ``session`` is
    provided, its ``base_url`` is consulted before each profile and the
    session decides whether to restart the server if /health is dead —
    that's the path the public ``run_tier`` takes.

    ``per_profile_timeout_s`` caps each individual harness's wall-clock
    time. A hung harness (real-world cause: codex e2e_file_read hanging
    156s on a slow model and taking the in-process server down) is now
    recorded as a per-profile FAIL with a clear timeout marker, and the
    next profile starts immediately against the rebooted server instead
    of cascading ``ECONNREFUSED`` failures. ``None`` (the default)
    resolves to the module-level ``HARNESS_PROFILE_TIMEOUT_S`` at CALL
    time — important so the test suite can monkeypatch the constant for
    fast unit tests (a default-arg sentinel would freeze the value at
    function-definition time).
    """
    from ..agents import get_profile

    timeout_s = (
        per_profile_timeout_s
        if per_profile_timeout_s is not None
        else HARNESS_PROFILE_TIMEOUT_S
    )

    t0 = time.perf_counter()
    per_harness: list[tuple[str, bool, float, str]] = []

    # Optional subset filter — see ``_resolve_harness_profiles_filter``
    # for env-var semantics. Iterating over the filtered list (not
    # HARNESS_PROFILES) preserves the documented order while skipping
    # profiles not in the active sweep. G12 (random-coverage) uses this
    # to scope to e.g. 2 of the 5 harnesses per (model × round) pick.
    profiles_to_run = HARNESS_PROFILES_FILTER or HARNESS_PROFILES
    for profile_name in profiles_to_run:
        profile = get_profile(profile_name)
        if profile is None:
            per_harness.append(
                (profile_name, False, 0.0, f"profile {profile_name!r} not found")
            )
            continue

        # Resolve the URL to hit for this profile + decide if the
        # server is up. With no session, we trust the caller (legacy
        # path / unit tests). With a session, we probe + may reboot.
        if session is None:
            profile_base_url = base_url
        else:
            healthy, note = session.ensure_healthy()
            profile_base_url = session.base_url
            if note:
                print(f"  [server] {note}")
            if not healthy:
                # Can't run this profile — record a FAIL with the
                # reason and move on to the next one (which will also
                # try to ensure health if it was a transient dead-but-
                # rebootable situation we couldn't recover from).
                per_harness.append(
                    (
                        profile_name,
                        False,
                        0.0,
                        f"server not healthy before profile: {note or 'unknown'}",
                    )
                )
                continue

        ok, h_elapsed, detail, timed_out = _run_single_profile(
            profile_name,
            profile,
            profile_base_url,
            timeout_s,
        )
        per_harness.append((profile_name, ok, h_elapsed, detail))

        # Per-profile timeout → the orphaned daemon thread may still be
        # mid-request against this server. If we re-use the server for
        # the next profile, the orphan's response could land DURING the
        # next profile's measurements and corrupt the supposedly
        # independent result. Force a fresh server so each profile
        # starts on a clean slate. Codex review-2 BLOCKING-2.
        if timed_out and session is not None:
            ok_restart, note = session.force_restart_after_timeout()
            if note:
                print(f"  [server] {note}")
            if not ok_restart:
                # Codex review-4 BLOCKING: a failed forced restart means
                # the orphaned daemon thread from the timed-out profile
                # may still be issuing requests against whatever server
                # state survived, AND we couldn't boot a clean
                # replacement (most likely because the teardown raised
                # and the session is now in ``_reboot_disabled`` state).
                # Surface this immediately as part of the timing-out
                # profile's own detail rather than waiting for the next
                # ``ensure_healthy`` probe to repeat the bad news — the
                # operator sees one consolidated FAIL row instead of
                # chasing a stale "server not healthy" message into the
                # NEXT profile's row. Mutate in place because the row
                # was already appended above.
                if per_harness:
                    name, _ok, dur, base_detail = per_harness[-1]
                    suffix = note or "force restart failed"
                    per_harness[-1] = (
                        name,
                        False,
                        dur,
                        f"{base_detail} | server isolation FAILED: {suffix}",
                    )

    elapsed = time.perf_counter() - t0
    all_passed = all(ok for _, ok, _, _ in per_harness)

    # Schema-v2 ``harness_result`` shape: one entry per HARNESS_PROFILES
    # adapter with {passed, duration_s, error_excerpt}. On pass,
    # error_excerpt is ``null`` (schema enforces ``["string", "null"]``);
    # on fail, the first 200 chars of the existing detail line — same
    # truncation cap as schema. The dict is keyed by adapter name; the
    # schema's ``additionalProperties: false`` plus ``required`` of all
    # 5 keys means this stays in lock-step with HARNESS_PROFILES (a
    # mismatch is a schema-validation failure at submission time).
    #
    # When ``HARNESS_PROFILES_FILTER`` is active, ``per_harness`` only
    # contains entries for the filtered profiles — the resulting payload
    # is NOT submission-safe (missing keys). The submit flow rejects
    # ``--base-url`` (the only path that combines tier + submit) AND the
    # filter env is only set by G12, which never passes ``--submit``.
    # Adding a hard refusal in ``_run_tier_submit_flow`` would be belt-
    # and-braces — see release_check_m3_random.py for the only caller.
    payload = {
        name: {
            "passed": ok,
            "duration_s": float(dur),
            "error_excerpt": None if ok else (detail or "")[:200],
        }
        for name, ok, dur, detail in per_harness
    }

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
        payload=payload,
    )


# --------------------------------------------------------------------- #
# Server lifecycle                                                       #
# --------------------------------------------------------------------- #


def _serve_or_attach(
    model: str,
    base_url: str | None,
    boot_timeout_s: int = 600,
):
    """Yield ``(port, owns_server, boot_time_ms, release_slot)``.

    PRIVATE API CHANGE — PR #684 (cascade-fail fix): pre-fix this
    helper yielded a 3-tuple ``(port, owns_server, boot_time_ms)``.
    The 4th element is the mutable release-slot the harness session
    uses to hand replacement-server cleanup back to the outer ``with``.
    Only ``run_tier`` calls this helper today (verified via
    ``grep -rn _serve_or_attach``); the underscore prefix already
    declared it private, so the shape break is intentional and not
    deprecation-worthy.

    If ``base_url`` is set, attach to that server — caller is responsible
    for its lifecycle and ``boot_time_ms`` is ``None`` (the user
    already paid the boot cost, we didn't measure it). ``release_slot``
    is ``None`` in the attach case because we can't tear down a server
    we don't own.

    Otherwise boot one on a port in ``[TIER_PORT_MIN, TIER_PORT_MAX]``
    and clean it up on exit; ``boot_time_ms`` is the wall-clock from
    spawn to first healthy /health response, which feeds the schema v2
    ``smoke_result.boot_time_ms`` field.

    ``release_slot`` is a single-key dict ``{"current": callable}``
    that the harness session mutates on each restart so the outer
    ``with``'s teardown always targets the LIVE server. Sequence:

    1. On enter, ``slot["current"]`` is set to a zero-arg release for
       the initial server.
    2. ``_HarnessServerSession._restart`` reads ``slot["current"]``,
       calls it to kill the old server, boots a fresh one, then writes
       a NEW release closure into ``slot["current"]`` tied to the
       replacement server.
    3. On the outer ``with`` exit, we call ``slot["current"]`` once.
       It targets either the original (no restart) or the most recent
       reboot (restart happened) — never a stale handle to a server
       already torn down.

    This indirection closes both Codex review-2 BLOCKING (the
    coexistence window during a forced restart) and Codex review-3
    BLOCKING (cleanup boundary stays at the outer ``with`` so a tier
    added after harness in ``tier="all"`` sees a live server, not one
    that the harness session prematurely closed).
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
            yield port, False, None, None
            return

        # Boot our own. Helper lives in the bench/ package now (PR #5
        # — PR #622 deleted the original ``doctor/server.py`` without
        # repointing this import, breaking every tier invocation; the
        # bench/ package is the only consumer left so we relocated it
        # alongside the dispatcher). Battle-tested code path: proc-group
        # teardown + /health polling, plus a boot_time_ms readout we
        # need for ``smoke_result.boot_time_ms`` in the v2 schema.
        port = _find_free_port_in_range(TIER_PORT_MIN, TIER_PORT_MAX)
        from ._server import serve

        # log_path=None → DEVNULL; the user-facing tier output is the
        # interesting signal. For debugging, the user can re-run with
        # `rapid-mlx serve <model> --port <port>` themselves. We
        # manually drive the ctx manager (not ``with``) so we can hand
        # the active-server release into the harness session and let it
        # swap in a fresh release on each restart. The outer ``finally``
        # then releases WHATEVER server is currently live (the original
        # if no restart, or the latest replacement if any), so cleanup
        # always targets the live server — not a stale handle the
        # session already tore down.
        ctx = serve(model=model, port=port, boot_timeout_s=boot_timeout_s)
        released = {"done": False}

        def _release_initial() -> None:
            # NOTE: do NOT swallow exceptions here. The harness session
            # treats a raising release as "teardown failed, refuse to
            # boot a replacement" (Codex review-3 BLOCKING). If we
            # silently ate the failure, the session would happily start
            # a second model server while the first was still alive.
            # The outer ``_release_current_server`` does swallow on the
            # FINAL teardown path (it has no reboot to refuse), but the
            # session's restart path needs the raw signal.
            if released["done"]:
                return
            released["done"] = True
            ctx.__exit__(None, None, None)

        # The ``current`` slot starts pointing at the initial server's
        # release. ``_HarnessServerSession._restart`` swaps it on each
        # reboot so this finally always targets the LIVE server. Codex
        # review-3 BLOCKING: this keeps the cleanup boundary at the
        # outer ``with`` so a tier added after harness sees the right
        # server-or-no-server state.
        release_slot: dict[str, object] = {"current": _release_initial}

        def _release_current_server() -> None:
            # Final teardown at outer ``with`` exit — nothing to refuse,
            # so swallow any failure (logging would be nice but the
            # bench is mid-shutdown and the log_path is gone).
            cb = release_slot["current"]
            if cb is None:
                return
            release_slot["current"] = None
            try:
                cb()  # type: ignore[operator]
            except Exception:  # noqa: BLE001
                pass

        # ``__enter__`` lives INSIDE the try so any future code inserted
        # between enter and finally still releases the server cleanly
        # (Codex review-3 NIT-1). If ``__enter__`` itself raises, the
        # finally still runs but ``_release_initial`` is a no-op against
        # a never-entered context — ``ctx.__exit__(None, None, None)``
        # on a generator-based contextmanager that never reached its
        # yield is well-defined: ``contextlib`` swallows the
        # ``StopIteration``.
        try:
            info = ctx.__enter__()
            yield (
                info["port"],
                True,
                info.get("boot_time_ms"),
                release_slot,
            )
        finally:
            _release_current_server()

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
    return_results: bool = False,
    skip_speed: bool = False,
):
    """Dispatch to the requested tier. Returns process exit code.

    ``tier`` is one of: ``"smoke"``, ``"speed"``, ``"harness"``, ``"all"``.
    Server is booted ONCE (or attached via ``base_url``) and torn down
    on exit. For ``"all"``, smoke runs first and aborts the sequence
    on failure — there's no point benching a model that can't say "4".

    Two PR #5 additions (default-off for back-compat):

    - ``return_results=True`` flips the return type to
      ``tuple[int, dict]`` where the dict carries schema-v2-shaped
      ``smoke_result`` / ``harness_result`` sub-objects (None when the
      corresponding tier didn't run, e.g. ``tier=speed``). The
      ``--tier ... --submit`` wiring needs this so it can feed
      ``build_submission_payload`` without re-running the tier work.
    - ``skip_speed=True`` is honored only when ``tier == "all"`` and
      tells the dispatcher to skip the lightweight HTTP-based speed
      probe. The caller (``--tier all --submit``) is going to run the
      locked B=1 ``run_standardized_bench`` against the same model
      separately — running the lightweight probe too would just
      double-cost the bench (and produce numbers that aren't comparable
      to the submitted ones, which is misleading).

    Without either kwarg, the signature and return type are byte-for-
    byte identical to the PR #2 surface — existing callers continue to
    receive ``int``.
    """
    if tier not in ("smoke", "speed", "harness", "all"):
        print(
            f"  Error: unknown tier {tier!r}; expected one of "
            "smoke / speed / harness / all",
            file=sys.stderr,
        )
        if return_results:
            return 2, {"smoke_result": None, "harness_result": None}
        return 2

    print(f"Rapid-MLX bench — tier={tier} model={model}")
    print("=" * 60)

    overall_t0 = time.perf_counter()
    results: list[TierResult] = []

    try:
        with _serve_or_attach(model, base_url, boot_timeout_s) as (
            port,
            owns,
            boot_time_ms,
            release_slot,
        ):
            # Build the fully-qualified base URL ONCE. Each tier gets
            # the same string, so --base-url's scheme + host are honored
            # end-to-end (codex review #621 BLOCKING).
            openai_base = _normalize_openai_base(base_url, port)
            if owns:
                print(f"  [server] booted {model} on port {port}")
            else:
                print(f"  [server] attached to existing server at {openai_base}")
            print()

            if tier in ("smoke", "all"):
                r = _run_smoke(model, openai_base, boot_time_ms=boot_time_ms)
                _print_tier_result(r)
                results.append(r)
                if tier == "all" and not r.passed:
                    print()
                    print("  Aborting --tier all: smoke failed.")
                    return _finalize_with_results(results, overall_t0, return_results)

            # PR #5: --submit code path sets skip_speed=True for tier='all'
            # because it runs the locked B=1 standardized bench against the
            # same model right after. The lightweight tier-speed probe is
            # explicitly NOT comparable to the standardized numbers, so
            # running both would just waste cycles AND mislead anyone
            # eyeballing the two outputs.
            if tier in ("speed", "all") and not (tier == "all" and skip_speed):
                r = _run_speed(model, openai_base, sampled=sampled)
                _print_tier_result(r)
                results.append(r)

            if tier in ("harness", "all"):
                # Hand the harness sweep a session that can health-check
                # /health between profiles and reboot the server when a
                # bad profile (codex hang on slow model → in-process OOM)
                # has taken it down. Without this the cascade fail
                # documented in #682 (codex hangs → opencode/hermes/aider/
                # langchain all FAIL with ECONNREFUSED) re-occurs every
                # time a model trips the timeout.
                #
                # Pass through ``release_slot`` so the session can both
                # tear down the LIVE server before booting a replacement
                # (Codex review-2 BLOCKING) AND hand the replacement's
                # release back into the slot so the outer ``with`` exit
                # cleans up the right server (Codex review-3 BLOCKING).
                session = _build_harness_session(
                    model=model,
                    boot_timeout_s=boot_timeout_s,
                    current_port=port,
                    current_owns=owns,
                    current_base_url=openai_base,
                    release_slot=release_slot,
                )
                r = _run_harness(model, openai_base, session=session)
                # Per Codex review-3 BLOCKING: defer ALL server cleanup
                # to the outer ``_serve_or_attach`` finally block. We
                # DON'T close the session here even though harness is
                # currently the last tier in ``tier="all"``:
                #   - If the session never restarted, the outer release
                #     still points at the live server and the outer
                #     finally tears it down correctly.
                #   - If the session DID restart, ``_restart`` already
                #     swapped ``_serve_or_attach``'s mutable
                #     ``current_server_release`` slot to point at the
                #     latest live server (see ``_restart`` for the
                #     handoff). The outer finally then tears down the
                #     CURRENT server, not the dead original.
                # This keeps cleanup at the outer ``with`` boundary so
                # adding a tier AFTER harness in the future won't see
                # a surprise-dead server.
                _print_tier_result(r)
                results.append(r)
    except Exception as exc:  # noqa: BLE001 — surface as exit code, not traceback
        print(f"\n  Error during tier run: {type(exc).__name__}: {exc}")
        if return_results:
            return 1, _collect_payload(results)
        return 1

    return _finalize_with_results(results, overall_t0, return_results)


def _build_harness_session(
    model: str,
    boot_timeout_s: int,
    current_port: int,
    current_owns: bool,
    current_base_url: str,
    release_slot,
) -> _HarnessServerSession:
    """Construct a ``_HarnessServerSession`` for the harness sweep.

    ``release_slot`` is the mutable ``{"current": callable}`` dict
    yielded by ``_serve_or_attach`` (or ``None`` in attach mode). The
    session reads/writes ``slot["current"]`` on each restart:
    - Read: get the live server's release, call it to kill the old one
      (closes Codex review-2 BLOCKING coexistence window).
    - Write: install the replacement's release so the outer
      ``_serve_or_attach`` finally targets the live server (closes
      Codex review-3 BLOCKING — cleanup boundary stays at the outer
      ``with`` block).
    """
    return _HarnessServerSession(
        model=model,
        boot_timeout_s=boot_timeout_s,
        initial_port=current_port,
        initial_owns=current_owns,
        initial_base_url=current_base_url,
        release_slot=release_slot,
    )


def _collect_payload(results: list[TierResult]) -> dict:
    """Pull the schema-v2 sub-objects out of the per-tier results.

    Always returns the same shape so the caller can pattern-match
    without ``in`` checks:

    - ``smoke_result``: the smoke TierResult's payload, or ``None`` if
      smoke didn't run.
    - ``harness_result``: same for harness.

    Speed is intentionally absent — the speed bucket of a v2
    submission comes from ``run_standardized_bench``, not from the
    tier dispatcher's lightweight HTTP probe (whose numbers aren't
    comparable to the locked B=1 protocol).
    """
    out: dict = {"smoke_result": None, "harness_result": None}
    for r in results:
        if r.name == "smoke":
            out["smoke_result"] = r.payload
        elif r.name == "harness":
            out["harness_result"] = r.payload
    return out


def _finalize_with_results(
    results: list[TierResult],
    t0: float,
    return_results: bool,
):
    """Print summary, then return either ``int`` or ``(int, dict)``.

    Kept as a separate helper so every early-exit branch in
    ``run_tier`` shapes its return value consistently — the int-vs-
    tuple decision lives in exactly one place.
    """
    rc = _finalize(results, t0)
    if return_results:
        return rc, _collect_payload(results)
    return rc


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
    summary = ", ".join(f"{r.name}={'pass' if r.passed else 'fail'}" for r in results)
    print(f"  {marker}: {n_pass}/{len(results)} tiers passed ({summary})")
    print(f"  total: {total:.1f}s")
    return 0 if overall_ok else 1
