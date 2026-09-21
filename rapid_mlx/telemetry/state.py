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
2. Environment kill switches → forced OFF: ``RAPID_MLX_TELEMETRY=0``
   (any falsy value), the cross-tool ``DO_NOT_TRACK=1`` convention, or a
   CI marker (``CI``, ``GITHUB_ACTIONS``, ``GITLAB_CI``, ``CIRCLECI``,
   ``TRAVIS``, ``BUILDKITE``, ``JENKINS_URL``, ``TEAMCITY_VERSION``) set to
   a non-empty value — build machines are never users.
3. Stored consent file → whatever the user answered.
4. Default → OFF. (Anonymous data collection without explicit opt-in is
   a non-starter.)

There is intentionally no env-var equivalent for forcing ON. CI agents
silently opting in via ``RAPID_MLX_TELEMETRY=1`` would skew the data
toward synthetic workloads. Users who want to opt in run
``rapid-mlx telemetry enable`` once.
"""

from __future__ import annotations

import errno
import fcntl
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ENV_VAR = "RAPID_MLX_TELEMETRY"
#: Cross-tool opt-out convention (https://consoledonottrack.com): ``1`` or
#: ``true`` disables telemetry; any other value is ignored, like Orca does.
DO_NOT_TRACK_ENV = "DO_NOT_TRACK"
#: Presence (non-empty) of any of these means "this process runs on a build
#: machine". CI providers set them for every job; nothing here is a user.
CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "BUILDKITE",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
)

# Bump when the on-disk consent file format OR the disclosure copy changes in
# a way that materially alters what we collect. A stored record whose
# schema_version != this is treated as "never prompted" so the user is re-asked
# under the new disclosure.
#
# v1 -> v2: added the unsampled ``activation`` event (docs/telemetry-activation.md).
# Prior opt-ins consented to a disclosure that never mentioned activation, so
# they must re-see and re-accept before any activation event is emitted —
# adding a new data type silently under old consent would be a consent breach.
CURRENT_CONSENT_SCHEMA_VERSION = 2

_LOCK_RETRY_SECONDS = 1.5
_LOCK_RETRY_INTERVAL_SECONDS = 0.05
_lock_retry_clock = time.monotonic
_lock_retry_sleep = time.sleep


def _default_telemetry_dir() -> Path:
    """Resolved at call time so ``HOME`` overrides in tests take effect."""
    return Path.home() / ".rapid-mlx"


def client_id_path() -> Path:
    return _default_telemetry_dir() / "telemetry-client-id"


def consent_path() -> Path:
    return _default_telemetry_dir() / "telemetry-consent.yaml"


def _read_consent_mapping(path: Path) -> dict[str, Any] | None:
    """Return an absent record as ``{}`` and an unreadable record as None."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_consent(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace the consent path with a mode-0600 YAML mapping."""
    payload = yaml.safe_dump(data, sort_keys=True).encode()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("consent write made no progress")
                offset += written
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _locked_merge_consent(
    merge: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    unreadable_replacement: dict[str, Any] | None = None,
) -> bool:
    """Read, merge, and atomically replace the consent mapping under flock.

    The sibling lock is permanent so waiters cannot split across different
    lock-file inodes. If that permanent lock cannot be opened or acquired
    (for example, because an earlier sudo run left it root-owned), the write
    falls back to an unlocked read-merge-atomic-replace: an explicit user
    choice must not be blocked by stale lock ownership. A present but
    unreadable record is preserved unless ``unreadable_replacement`` is
    supplied by the explicit-consent writer. Replacing the consent path also
    deliberately replaces a symlink with a regular file, matching v1.
    """
    path = consent_path()
    lock_path = path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_merged() -> bool:
        data = _read_consent_mapping(path)
        if data is None:
            if unreadable_replacement is None:
                return False
            merged = dict(unreadable_replacement)
        else:
            merged = merge(data)
        _atomic_write_consent(path, merged)
        return True

    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return write_merged()
    try:
        try:
            deadline = _lock_retry_clock() + _LOCK_RETRY_SECONDS
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EINTR):
                        raise
                    remaining = deadline - _lock_retry_clock()
                    if remaining <= 0:
                        raise
                    _lock_retry_sleep(min(_LOCK_RETRY_INTERVAL_SECONDS, remaining))
        except OSError:
            os.close(lock_fd)
            lock_fd = -1
            return write_merged()
        try:
            return write_merged()
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)


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
    # Defaults to the OLDEST schema (1), not the current one: a ConsentState
    # built without an explicit version is treated as the most conservative
    # (re-promptable) record. Only ``record_consent`` — the one place that
    # actually captures a fresh opt-in — stamps ``CURRENT_CONSENT_SCHEMA_VERSION``
    # explicitly. Defaulting to CURRENT here would let a partially-constructed
    # or legacy record read as "already consented under the newest disclosure"
    # and silently bypass the re-consent gate.
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
        # Stamp the CURRENT schema explicitly — this is the sole place a fresh
        # opt-in is captured, so it is the sole place that should mint a
        # newest-version record.
        schema_version=CURRENT_CONSENT_SCHEMA_VERSION,
    )
    payload = {
        "consent": state.consent,
        "prompted_at": state.prompted_at,
        "prompted_version": state.prompted_version,
        "schema_version": state.schema_version,
    }
    unreadable_replacement = dict(payload)
    # If this post-cutoff process actually delivered the v2 disclosure, an
    # explicit answer replacing an unreadable record may safely carry the
    # current marker. Without actual delivery the marker must stay absent.
    from rapid_mlx.telemetry import consent_runtime
    from rapid_mlx.telemetry.consent_decision import DISCLOSURE_REVISION

    if consent_runtime.notice_was_delivered():
        unreadable_replacement["notice_revision_seen"] = DISCLOSURE_REVISION
    path = consent_path()
    # Retain cleanup of the fixed-name temporary used by the original v1
    # writer. New writes use the shared randomized atomic writer below.
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    try:
        _locked_merge_consent(
            lambda existing: {**existing, **payload},
            unreadable_replacement=unreadable_replacement,
        )
    except OSError as exc:
        raise OSError(f"cannot write {path}: {exc}") from exc
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


def _validate_activation_kind(kind: str) -> None:
    """Reject any ``kind`` not on the fixed allowlist BEFORE it reaches a path.

    Defense in depth: ``kind`` is interpolated into a filename, so an
    unvalidated value with ``../`` (or a separator) could escape
    ``~/.rapid-mlx`` and have ``claim_activation_marker`` create dirs/files
    outside it. Callers in ``emit.activation`` already gate on
    ``ACTIVATION_KINDS``; validating here makes the state layer safe on its own
    regardless of caller. Imported lazily to avoid an import cycle
    (``activation_spec`` is constants-only, but keep state import-light).
    """
    from rapid_mlx.telemetry.activation_spec import ACTIVATION_KINDS

    if kind not in ACTIVATION_KINDS:
        raise ValueError(f"unknown activation kind: {kind!r}")


def activation_marker_path(kind: str) -> Path:
    """Path to the once-per-install marker for an activation ``kind``.

    One file per kind so ``first_inference`` / ``model_pull`` /
    ``agent_setup`` are claimed independently. Kept beside the consent /
    client-id files under ``~/.rapid-mlx/``. ``kind`` is validated against the
    allowlist so no caller-controlled string can be interpolated into a path
    that escapes the telemetry dir.
    """
    _validate_activation_kind(kind)
    return _default_telemetry_dir() / f"activation_seen_{kind}"


def claim_activation_marker(kind: str) -> bool:
    """Atomically claim the once-per-install activation marker for ``kind``.

    Returns ``True`` for exactly ONE call per machine per kind — the first
    process to create the marker via exclusive ``O_CREAT | O_EXCL``
    creation. Every later call (and any concurrent racer that loses) gets
    ``False``. Same mechanism as ``first_run.mark_first_session``.

    Fail-safe toward ``False`` on any error (invalid kind, unwritable state
    dir, etc.): under-reporting an activation is conservative — it never
    inflates the funnel — and never crashes the caller. An out-of-allowlist
    ``kind`` raises inside ``activation_marker_path`` and is caught here, so no
    filesystem operation ever runs for it. The marker is a local empty file;
    only the derived enum ever leaves the machine, and only when telemetry is
    enabled.
    """
    try:
        marker = activation_marker_path(kind)
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Mode "x" == O_CREAT | O_EXCL: the OS guarantees only one concurrent
        # creator succeeds. The context manager closes the fd on every path.
        with open(marker, "x"):
            pass
        return True
    except FileExistsError:
        return False
    except Exception:
        return False


def reset_state() -> None:
    """Remove consent + client-id + activation markers. Next run re-prompts.

    Activation dedup is keyed on the install identity: wiping the ``client_id``
    while leaving the once-per-install ``activation_seen_*`` markers behind
    would leave the freshly generated identity permanently unable to emit any
    milestone the old identity already claimed. Reset clears both so a new
    identity can re-earn its funnel from scratch.

    Also clears the in-process activation latch in ``emit`` so a reset followed
    by re-enabling telemetry *within the same process* can re-earn milestones:
    the persistent markers are gone, and a stale in-memory latch would
    otherwise keep suppressing emission. Late import breaks the state<->emit
    cycle.

    Attempts EVERY path even when one fails (a single unremovable file must not
    skip the rest or the latch clear), then — after clearing the latch — raises
    an aggregated ``OSError`` if any file could not be removed. That last step
    matters: a caller like ``telemetry reset`` must never print "removed" while
    consent or client-id state actually survives on disk (which would leave
    telemetry silently enabled). A file that was already absent is not a
    failure.
    """
    paths = [consent_path(), client_id_path()]
    try:
        paths.extend(_default_telemetry_dir().glob("activation_seen_*"))
    except OSError as exc:
        # Couldn't even enumerate the markers — record it and press on with the
        # consent/client-id removals we already know about.
        failures = [f"{_default_telemetry_dir()}/activation_seen_*: {exc}"]
    else:
        failures = []
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass  # already gone == the desired post-condition
        except OSError as exc:
            # Permission error, a path that is unexpectedly a directory, etc. on
            # ONE entry must not abort the whole cleanup. Record and continue so
            # the remaining unlinks and the latch clear still run; the collected
            # failures are surfaced at the end.
            failures.append(f"{path}: {exc}")
    # The sibling flock is intentionally permanent during normal writes, but
    # reset is an explicit request to remove telemetry state. Its deletion is
    # best-effort: a stale root-owned lock must not turn an otherwise complete
    # reset into a false failure.
    lock_path = consent_path().with_name(consent_path().name + ".lock")
    try:
        lock_path.unlink()
    except OSError:
        pass
    # Clear the latch regardless of file-removal outcome, so a same-process
    # re-enable can re-earn milestones even if some on-disk cleanup failed.
    try:
        from rapid_mlx.telemetry import emit

        emit._reset_activation_latch()
    except Exception:
        pass
    if failures:
        raise OSError("telemetry reset could not remove: " + "; ".join(failures))


def _env_kill_switch_reason() -> str | None:
    """Why the environment forces telemetry OFF, or ``None`` if it doesn't.

    Three switches, checked in this order:

    * ``RAPID_MLX_TELEMETRY`` falsy (``0`` / ``false`` / ``no`` / ``off`` /
      empty). Truthy values are intentionally ignored — see the module
      docstring for why there is no env-var force-on.
    * ``DO_NOT_TRACK`` truthy (``1`` / ``true``, case-insensitive) — the
      cross-tool convention. Other values are ignored rather than guessed.
    * Any CI marker in ``CI_ENV_VARS`` set to a non-empty value.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is not None and raw.strip().lower() in ("0", "false", "no", "off", ""):
        return f"env-var ({ENV_VAR}={raw!r})"
    dnt = os.environ.get(DO_NOT_TRACK_ENV)
    if dnt is not None and dnt.strip().lower() in ("1", "true"):
        return f"env-var ({DO_NOT_TRACK_ENV}={dnt!r})"
    for name in CI_ENV_VARS:
        if os.environ.get(name, "") != "":
            return f"ci ({name} is set)"
    return None


def _env_kill_switch_active() -> bool:
    """True when any environment kill switch is engaged."""
    return _env_kill_switch_reason() is not None


# Process-level kill switch set by ``cli.py`` when ``--no-telemetry`` is
# passed. The ``cli_no_telemetry=`` keyword on ``is_enabled`` only helps
# the few call sites that explicitly thread it through (``status`` /
# ``preview`` UX); the four event-emit helpers in ``emit.py`` do NOT
# accept the kwarg (it would balloon every emit-site signature), so they
# read this flag instead. Set once, after argparse, before any emit.
_cli_kill_switch_active = False


def set_cli_kill_switch(active: bool) -> None:
    """Mark the current process as ``--no-telemetry``.

    Idempotent and global within the process. ``emit.session_start`` is
    the first caller affected; it reads the flag via ``is_enabled()``.
    """
    global _cli_kill_switch_active
    _cli_kill_switch_active = bool(active)


def is_enabled(*, cli_no_telemetry: bool = False) -> bool:
    """Single decision point used by every event-emit site.

    Phase 1 has no event sites — the function exists so Phase 2 can call
    it with no further design work, and so tests can pin the precedence
    contract today.
    """
    if cli_no_telemetry or _cli_kill_switch_active:
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
    if cli_no_telemetry or _cli_kill_switch_active:
        return "cli-flag (--no-telemetry)"
    reason = _env_kill_switch_reason()
    if reason is not None:
        return reason
    state = get_consent_state()
    if state is None:
        return "default (no consent recorded)"
    return f"consent-file ({consent_path()})"
