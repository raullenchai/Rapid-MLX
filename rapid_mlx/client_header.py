# SPDX-License-Identifier: Apache-2.0
"""``X-Rapid-Client`` — how a Rapid-owned client identifies itself.

Every HTTP client we ship that talks to a Rapid server sends this header
with one of the labels below. The server buckets the caller from the
header first and falls back to the User-Agent allow-list
(``telemetry.redact.normalize_caller_agent``) for third-party callers.

Two rules make this safe to put on a telemetry payload:

1. **Closed set.** The value is only ever one of ``RAPID_CLIENT_LABELS``.
   Anything else is ignored server-side and never echoed — the header is
   caller-controlled input like any other, so it may only ever *select*
   one of our own labels, never introduce a new string.
2. **No versions, no free text.** The app version already rides on the
   telemetry envelope; repeating it here would only add a fingerprint.

Kept in its own import-light module (no telemetry import) because the
senders are CLI / bench / agent code paths and the receiver is the
telemetry redaction layer — both need the constants, neither should pull
in the other.
"""

from __future__ import annotations

RAPID_CLIENT_HEADER = "X-Rapid-Client"

#: The macOS desktop app talking to its bundled sidecar.
RAPID_CLIENT_DESKTOP = "rapid-desktop"
#: ``rapid-mlx chat`` (and the server it spawns / attaches to).
RAPID_CLIENT_CLI_CHAT = "rapid-cli-chat"
#: The agent runtime / agent test harness.
RAPID_CLIENT_AGENTS = "rapid-agents"
#: ``rapid-mlx bench`` and the community-benchmark runners.
RAPID_CLIENT_BENCH = "rapid-bench"
#: The bundled Gradio chat UI (``rapid-mlx-chat``, ``gradio_app.py``). A
#: separate label from ``rapid-cli-chat`` because it is a separate surface:
#: a browser UI pointed at a server the user started themselves.
RAPID_CLIENT_GRADIO = "rapid-gradio"

RAPID_CLIENT_LABELS: frozenset[str] = frozenset(
    {
        RAPID_CLIENT_DESKTOP,
        RAPID_CLIENT_CLI_CHAT,
        RAPID_CLIENT_AGENTS,
        RAPID_CLIENT_BENCH,
        RAPID_CLIENT_GRADIO,
    }
)


def rapid_client_headers(
    label: str, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Return request headers carrying ``X-Rapid-Client: <label>``.

    ``extra`` is merged first so the client header always wins — a caller
    cannot accidentally overwrite its own identity with a stale dict.
    Raises ``ValueError`` for an off-list label: adding a client is a
    deliberate edit to this file, which is what keeps the set closed.
    """
    if label not in RAPID_CLIENT_LABELS:
        raise ValueError(
            f"unknown Rapid client label {label!r}; "
            f"expected one of {sorted(RAPID_CLIENT_LABELS)}"
        )
    headers = dict(extra or {})
    headers[RAPID_CLIENT_HEADER] = label
    return headers
