# SPDX-License-Identifier: Apache-2.0
"""Anonymous, metadata-only product telemetry.

The v2 pipeline validates events against a closed registry and sends
privacy-hardened batches to PostHog only after the build and consent gates
allow an upload. Legacy v1 transport, queue, schemas, and prompt code have
been removed.

See ``docs/telemetry.md`` (or the README "Telemetry" section) for the
full schema, what we do/do not collect, and how to disable.

Public API:

- ``is_enabled`` — the single decision point. Returns False unless the
  user opted in AND no kill switch is active.
- ``get_consent_state`` — full record (consent bool, when prompted,
  which version prompted them) for ``rapid-mlx telemetry status``.
- ``record_consent`` — persist a yes/no answer.
- ``reset_state`` — delete the preference and rotate the client id.
"""

from rapid_mlx.telemetry.state import (
    ConsentState,
    consent_source,
    get_consent_state,
    get_or_create_client_id,
    is_enabled,
    read_client_id,
    record_consent,
    reset_state,
)

__all__ = [
    "ConsentState",
    "consent_source",
    "get_consent_state",
    "get_or_create_client_id",
    "is_enabled",
    "record_consent",
    "read_client_id",
    "reset_state",
]
