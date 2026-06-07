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


__all__ = [
    "ErrorPayload",
    "PlatformInfo",
    "RequestPayload",
    "SCHEMA_VERSION",
    "SessionPayload",
    "TelemetryPayload",
    "bucket_memory_gb",
    "sample_preview_payload",
]
