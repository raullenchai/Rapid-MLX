# SPDX-License-Identifier: Apache-2.0
"""Anonymous opt-in usage telemetry — Phase 1 (consent + plumbing only).

This package owns the consent state, redaction primitives, and event
schema. **No network code lives here in Phase 1** — the transport,
collector, and Cloudflare Worker arrive in a follow-up PR. This split
exists so the privacy mechanism (kill switch, consent file, redaction
contracts, CLI subcommand) can land and be reviewed in isolation,
before any byte ever leaves the machine.

See ``docs/telemetry.md`` (or the README "Telemetry" section) for the
full schema, what we do/do not collect, and how to disable.

Public API:

- ``is_enabled`` — the single decision point. Returns False unless the
  user opted in AND no kill switch is active.
- ``get_consent_state`` — full record (consent bool, when prompted,
  which version prompted them) for ``rapid-mlx telemetry status``.
- ``record_consent`` — persist a yes/no answer.
- ``reset_state`` — wipe both consent + client-id files.
- ``maybe_prompt_for_consent`` — first-run interactive prompt; safe to
  call from every subcommand (it skips itself when not appropriate).
"""

from vllm_mlx.telemetry.consent import maybe_prompt_for_consent
from vllm_mlx.telemetry.state import (
    ConsentState,
    consent_source,
    get_consent_state,
    get_or_create_client_id,
    is_enabled,
    record_consent,
    reset_state,
)

__all__ = [
    "ConsentState",
    "consent_source",
    "get_consent_state",
    "get_or_create_client_id",
    "is_enabled",
    "maybe_prompt_for_consent",
    "record_consent",
    "reset_state",
]
