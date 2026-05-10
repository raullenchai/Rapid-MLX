# SPDX-License-Identifier: Apache-2.0
"""Consent + client-id state for opt-in telemetry.

Two files live under ``~/.rapid-mlx/``:

- ``telemetry-client-id`` — a UUID4 string. Stable across runs so we can
  count distinct opted-in machines without identifying them. The user
  can ``rm`` it to reset, or replace its contents with the all-zero UUID
  to anonymize their machine while still contributing aggregate counts.

- ``telemetry-consent.yaml`` — records the user's yes/no answer plus
  metadata (when prompted, which version asked) so we never re-prompt
  the same user and can later detect schema-version bumps that should
  re-prompt.

Two files rather than one because: ``rm telemetry-consent.yaml``
re-triggers the first-run prompt without losing the client_id; ``rm
telemetry-client-id`` rotates identity without re-prompting. Bundling
them would force the user into all-or-nothing.

The ``is_enabled`` decision precedence (highest first):

1. ``--no-telemetry`` CLI flag → forced OFF for this run.
2. ``RAPID_MLX_TELEMETRY=0`` env → forced OFF.
3. Stored consent file → whatever the user answered.
4. Default → OFF. (Anonymous data collection without explicit opt-in is
   a non-starter.)

There is intentionally no env-var equivalent for forcing ON. CI agents
silently opting in via ``RAPID_MLX_TELEMETRY=1`` would skew the data
toward synthetic workloads. Users who want to opt in run
``rapid-mlx telemetry enable`` once.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

ENV_VAR = "RAPID_MLX_TELEMETRY"

# Bump when the on-disk consent file format changes incompatibly. A
# stored record with a smaller schema_version is treated as "never
# prompted" so the user gets re-asked under the new disclosure copy.
CURRENT_CONSENT_SCHEMA_VERSION = 1


def _default_telemetry_dir() -> Path:
    """Resolved at call time so ``HOME`` overrides in tests take effect."""
    return Path.home() / ".rapid-mlx"


def client_id_path() -> Path:
    return _default_telemetry_dir() / "telemetry-client-id"


def consent_path() -> Path:
    return _default_telemetry_dir() / "telemetry-consent.yaml"


@dataclass(frozen=True)
class ConsentState:
    """The on-disk record of the user's opt-in answer.

    ``consent=False`` is meaningfully different from "no file exists":
    the former means the user was prompted and said no (don't re-prompt),
    the latter means we still owe them the first-run disclosure.
    """

    consent: bool
    prompted_at: str  # ISO-8601 UTC, "Z" suffix
    prompted_version: str  # rapid-mlx version that showed the prompt
    schema_version: int = 1


def get_consent_state() -> ConsentState | None:
    """Return the stored consent record, or ``None`` if never prompted.

    ``None`` is a signal to the caller that the first-run prompt should
    fire (subject to the other guards in ``consent.maybe_prompt_for_consent``).
    Malformed files also return ``None`` rather than raising — a corrupt
    consent record should re-prompt, not crash the CLI.
    """
    path = consent_path()
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    consent = data.get("consent")
    prompted_at = data.get("prompted_at")
    prompted_version = data.get("prompted_version")
    if not isinstance(consent, bool) or not isinstance(prompted_at, str):
        return None
    if not isinstance(prompted_version, str):
        return None
    schema_version = data.get("schema_version", 1)
    if not isinstance(schema_version, int):
        schema_version = 1
    # Treat unknown / older schema versions as "never prompted" so a
    # disclosure-copy bump in a future release re-asks every user under
    # the new wording. Forward-compat (newer file from a downgraded
    # rapid-mlx) hits the same path — safer to re-prompt than to honor
    # a record we don't fully understand.
    if schema_version != CURRENT_CONSENT_SCHEMA_VERSION:
        return None
    return ConsentState(
        consent=consent,
        prompted_at=prompted_at,
        prompted_version=prompted_version,
        schema_version=schema_version,
    )


def record_consent(consent: bool, *, rapid_mlx_version: str) -> ConsentState:
    """Persist the user's answer.

    Writes the file with mode 0600 — the directory itself stays at the
    user's umask default, which is fine because the only sensitive bytes
    are inside this file (the client_id UUID is random and the consent
    answer is binary).
    """
    state = ConsentState(
        consent=consent,
        prompted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        prompted_version=rapid_mlx_version,
    )
    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "consent": state.consent,
        "prompted_at": state.prompted_at,
        "prompted_version": state.prompted_version,
        "schema_version": state.schema_version,
    }
    # write-then-rename so a SIGINT mid-write can't leave a half-file
    # that get_consent_state() would silently treat as "never prompted"
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Clean up any leftover .tmp from a previous interrupted write so
    # we never start out with a partial file under our chosen name.
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    tmp.write_text(yaml.safe_dump(payload, sort_keys=True))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    return state


def get_or_create_client_id() -> str:
    """Return the persistent UUID, creating it on first call.

    Idempotent: subsequent calls read the existing file. A user-edited
    file containing the all-zero UUID is preserved as-is (documented
    way to anonymize while still contributing aggregate counts).
    """
    path = client_id_path()
    if path.exists():
        existing = path.read_text().strip()
        if existing:
            return existing
    new_id = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_id + "\n")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    return new_id


def reset_state() -> None:
    """Remove both consent + client-id files. Next run re-prompts."""
    for path in (consent_path(), client_id_path()):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _env_kill_switch_active() -> bool:
    """``RAPID_MLX_TELEMETRY=0`` (or any falsy value) wins.

    Truthy values are intentionally ignored — see module docstring for
    why there's no env-var force-on.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() in ("0", "false", "no", "off", "")


def is_enabled(*, cli_no_telemetry: bool = False) -> bool:
    """Single decision point used by every event-emit site.

    Phase 1 has no event sites — the function exists so Phase 2 can call
    it with no further design work, and so tests can pin the precedence
    contract today.
    """
    if cli_no_telemetry:
        return False
    if _env_kill_switch_active():
        return False
    state = get_consent_state()
    if state is None:
        return False
    return state.consent


def consent_source(*, cli_no_telemetry: bool = False) -> str:
    """Human-readable source of the current is_enabled() answer.

    Used by ``rapid-mlx telemetry status`` so users can debug why
    telemetry is (or isn't) enabled without reading our code.
    """
    if cli_no_telemetry:
        return "cli-flag (--no-telemetry)"
    if _env_kill_switch_active():
        return f"env-var ({ENV_VAR}={os.environ.get(ENV_VAR, '')!r})"
    state = get_consent_state()
    if state is None:
        return "default (no consent recorded)"
    return f"consent-file ({consent_path()})"
