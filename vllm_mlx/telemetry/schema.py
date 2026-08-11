# SPDX-License-Identifier: Apache-2.0
"""Telemetry payload schema v1 — wire shape only.

Phase 1 ships the dataclasses + a sample-payload builder used by
``rapid-mlx telemetry preview``. **No event sites populate these in
Phase 1** — they exist so reviewers can audit exactly what could ever
go on the wire, and so Phase 2 can wire events without re-debating the
shape.

Bump ``SCHEMA_VERSION`` whenever a backwards-incompatible field changes
(rename / drop / type-change). Adding optional fields does not require
a bump.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from vllm_mlx.telemetry.redact import (
    bucket_memory_gb,
    bucket_tokens,
    bucket_tps,
    bucket_ttft_ms,
    normalize_caller_agent,
    platform_info,
)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlatformInfo:
    os: str
    os_version: str
    arch: str
    chip: str
    memory_gb: int
    python_version: str


@dataclass(frozen=True)
class SessionPayload:
    subcommand: str  # "serve" | "agents" | "bench" | "chat" | "doctor" | "models"
    duration_seconds: int | None = None  # session_end only; None on session_start
    models_loaded: tuple[str, ...] = ()  # HF repo IDs only (normalized)
    # Schema v1 back-compat slot. Round 4 removed runtime emission of
    # ``engine`` from the emit helpers (it was a free-form ``str`` slot
    # with no information content while ``BatchedEngine`` is the only
    # engine), but the dataclass keeps the optional field so external
    # callers constructing ``SessionPayload(engine=...)`` against
    # ``SCHEMA_VERSION == 1`` don't break. Round 7 codex caught that
    # the field MUST stay in its original positional slot too —
    # positional ``SessionPayload("serve", 10, models, engine, flags)``
    # would otherwise silently mis-bind once ``engine`` moved past
    # ``flag_names``. Re-add to runtime emission only via an enum if a
    # second engine ever lands; bump SCHEMA_VERSION at the same time.
    engine: str = ""
    flag_names: tuple[str, ...] = ()  # names only, sorted, no values
    # Activation-funnel fields (#1272). Both are session METADATA booleans
    # -- derived from session context, NOT from any prompt or generated
    # output -- appended after ``flag_names`` with ``False`` defaults so the
    # positional-slot back-compat rule documented on ``engine`` above holds.
    # No SCHEMA_VERSION bump: additive optional fields are backwards-
    # compatible (old consumers ignore unknown keys), and the schema only
    # bumps on backwards-INCOMPATIBLE changes.
    first_session: bool = False  # first session we RECORD from this client (marker)
    auto_selected: bool = False  # ``chat`` fell back to the starter (no alias given)


@dataclass(frozen=True)
class RequestPayload:
    endpoint: str  # "/v1/chat/completions" etc.
    model_alias: str
    stream: bool
    tool_call_used: bool
    prompt_tokens_bucket: str
    completion_tokens_bucket: str
    ttft_ms_bucket: str
    tps_bucket: str
    status: int
    # v2 addition. Inbound HTTP User-Agent bucketed to a fixed allowlist
    # (redact.normalize_caller_agent) — "which agent is calling", never the
    # raw UA. Optional (default "unknown") so v1 external callers that build
    # RequestPayload positionally don't shift/break. request events ship
    # dark until the call sites land, so this widens no live wire contract.
    caller_agent: str = "unknown"
    # v2.2 addition (#1250). A single derived boolean: did the completion
    # look degenerate (repetition / single-token collapse) per the
    # CLIENT-SIDE ``vllm_mlx.coherence.looks_like_garbage`` heuristic? The
    # detector runs on the caller's machine and ONLY this bool leaves it —
    # never the prompt or the completion text, and the bool is not
    # reversible into either. It is the post-release canary for the #1234
    # class (normal-length but garbage output), which the token-count
    # buckets and the error events cannot see. Appended last + optional so
    # positional ``RequestPayload`` construction stays stable; empty
    # completions are reported ``False`` here (they already show up as the
    # zero completion-token bucket) so this stays a clean "non-empty content
    # looks like garbage" signal.
    output_degenerate: bool = False
    # v2.3 additions (#1250). Exact token counts remain local; these two
    # booleans make empty and abnormally-short completions observable without
    # narrowing the coarse token bucket (which would increase fingerprinting
    # risk). ``completion_abnormally_short`` excludes empty completions so the
    # collector can alert on the two failure modes independently.
    completion_empty: bool = False
    completion_abnormally_short: bool = False


@dataclass(frozen=True)
class ErrorPayload:
    category: str  # "model_load_failure" | "oom" | "tool_parse" | "shutdown_traceback"
    fingerprint: str  # 16-hex from redact.fingerprint_traceback()
    phase: str  # "startup" | "request" | "shutdown"


@dataclass(frozen=True)
class TelemetryPayload:
    """The complete on-the-wire envelope.

    Exactly one of ``session`` / ``request`` / ``error`` is populated
    per payload — the discriminator is the ``event`` field.
    """

    schema_version: int
    client_id: str
    session_id: str
    rapid_mlx_version: str
    platform: PlatformInfo
    event: str  # "session_start" | "session_end" | "request" | "error"
    timestamp: str  # ISO-8601 UTC, "Z" suffix
    session: SessionPayload | None = None
    request: RequestPayload | None = None
    error: ErrorPayload | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the envelope as a JSON-ready dict.

        ``None`` event-payload fields are dropped so the payload doesn't
        carry empty placeholders for the two events it isn't.
        """
        d = asdict(self)
        for key in ("session", "request", "error"):
            if d.get(key) is None:
                d.pop(key, None)
        return d


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_preview_payload(
    *,
    client_id: str,
    rapid_mlx_version: str,
) -> TelemetryPayload:
    """A representative payload for ``rapid-mlx telemetry preview``.

    Built from real platform info + made-up session fields so users can
    see exactly what would leave their machine without having to start
    a server. The session_id is a fixed dummy because previews shouldn't
    burn a real per-process id.
    """
    info = platform_info()
    return TelemetryPayload(
        schema_version=SCHEMA_VERSION,
        client_id=client_id,
        session_id="preview-0000000000000000",
        rapid_mlx_version=rapid_mlx_version,
        platform=PlatformInfo(
            os=info["os"],
            os_version=info["os_version"],
            arch=info["arch"],
            chip=info["chip"],
            memory_gb=info["memory_gb"],
            python_version=info["python_version"],
        ),
        event="session_start",
        timestamp=_utc_now_iso(),
        session=SessionPayload(
            subcommand="serve",
            models_loaded=("mlx-community/Qwen3.5-9B-4bit",),
            flag_names=("port", "host"),
        ),
    )


def sample_request_preview_payload(
    *,
    client_id: str,
    rapid_mlx_version: str,
) -> TelemetryPayload:
    """A representative ``request`` event for ``rapid-mlx telemetry preview``.

    Request events are the highest-volume stream, so previewing one lets users
    see the bucketed, content-free shape real traffic sends — every number is a
    coarse bucket, and the #1250 ``output_degenerate`` flag is a bare boolean.
    Neither the prompt nor the completion text is ever present. Bucket labels
    are produced by the real ``redact`` helpers so the preview matches the wire.
    """
    info = platform_info()
    return TelemetryPayload(
        schema_version=SCHEMA_VERSION,
        client_id=client_id,
        session_id="preview-0000000000000000",
        rapid_mlx_version=rapid_mlx_version,
        platform=PlatformInfo(
            os=info["os"],
            os_version=info["os_version"],
            arch=info["arch"],
            chip=info["chip"],
            memory_gb=info["memory_gb"],
            python_version=info["python_version"],
        ),
        event="request",
        timestamp=_utc_now_iso(),
        request=RequestPayload(
            endpoint="/v1/chat/completions",
            model_alias="mlx-community/Qwen3.5-9B-4bit",
            stream=True,
            tool_call_used=True,
            prompt_tokens_bucket=bucket_tokens(420),
            completion_tokens_bucket=bucket_tokens(180),
            ttft_ms_bucket=bucket_ttft_ms(310.0),
            tps_bucket=bucket_tps(58.0),
            status=200,
            caller_agent=normalize_caller_agent("claude-code/1.0"),
            output_degenerate=False,
            completion_empty=False,
            completion_abnormally_short=False,
        ),
    )


__all__ = [
    "ErrorPayload",
    "PlatformInfo",
    "RequestPayload",
    "SCHEMA_VERSION",
    "SessionPayload",
    "TelemetryPayload",
    "bucket_memory_gb",
    "sample_preview_payload",
    "sample_request_preview_payload",
]
