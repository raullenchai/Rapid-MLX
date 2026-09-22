# SPDX-License-Identifier: Apache-2.0
"""Pin the consent precedence contract.

The four-layer precedence (CLI flag > env-var kill switch > stored
consent > default off) is the *single decision* every event-emit site
in Phase 2+ will rely on. Breaking it silently re-enables (or
silently disables) telemetry for the whole user base, which is the
exact failure mode this issue exists to avoid. Test it directly.
"""

from __future__ import annotations

import errno
import importlib
import subprocess
import sys

import pytest

from rapid_mlx.telemetry import state

_PROCESS_ROLE_ENV_VARS = (
    "RAPID_MLX_PROCESS_ROLE",
    "RAPID_MLX_WATCHDOG_PPID",
)


@pytest.fixture(autouse=True)
def _clean_telemetry_env(monkeypatch):
    for name in (
        state.ENV_VAR,
        state.DO_NOT_TRACK_ENV,
        *state.CI_ENV_VARS,
        *_PROCESS_ROLE_ENV_VARS,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Reroute ``Path.home()`` so consent files land under tmp.

    The telemetry state module reads ``Path.home()`` at call time on
    purpose (so tests can do exactly this). Setting ``HOME`` is the
    documented way to override on POSIX.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    # Force-reload the state module so any cached path objects (there
    # shouldn't be any, but defence in depth) get rebuilt under the new
    # HOME.
    importlib.reload(state)
    return tmp_path


def test_default_is_off(fake_home):
    from rapid_mlx.telemetry.state import is_enabled

    assert is_enabled() is False


def test_consent_round_trip(fake_home):
    from rapid_mlx.telemetry.state import (
        get_consent_state,
        is_enabled,
        record_consent,
    )

    assert get_consent_state() is None
    record_consent(True, rapid_mlx_version="0.6.33")
    state = get_consent_state()
    assert state is not None
    assert state.consent is True
    assert state.prompted_version == "0.6.33"
    from rapid_mlx.telemetry.state import CURRENT_CONSENT_SCHEMA_VERSION

    assert state.schema_version == CURRENT_CONSENT_SCHEMA_VERSION
    assert state.prompted_at.endswith("Z")
    assert is_enabled() is True


def test_prior_v1_consent_is_reprompted_after_activation_added(fake_home):
    """v1 opt-ins predate the ``activation`` disclosure. Bumping the consent
    schema to 2 must make a stored v1 record read as "never prompted" so the
    user re-sees and re-accepts before any activation event can be emitted —
    adding a new collected data type under stale consent would be a breach."""
    import yaml

    from rapid_mlx.telemetry.state import (
        consent_path,
        get_consent_state,
        is_enabled,
    )

    # Hand-write a legacy v1 consent record (consent=yes under old copy).
    p = consent_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(
            {
                "consent": True,
                "prompted_at": "2026-01-01T00:00:00Z",
                "prompted_version": "0.11.0",
                "schema_version": 1,
            }
        )
    )
    # Treated as unprompted -> None -> telemetry stays OFF until re-accepted.
    assert get_consent_state() is None
    assert is_enabled() is False


def test_reset_state_is_best_effort_when_a_path_cannot_be_removed(fake_home):
    from rapid_mlx.telemetry import state

    # A normally-removable consent file...
    state.record_consent(True, rapid_mlx_version="0.0.0+test")
    state.get_or_create_client_id()
    assert state.consent_path().exists()
    # ...and a marker path that is a NON-EMPTY DIRECTORY, so unlink() raises
    # OSError (IsADirectoryError) — the glob picks it up like any marker.
    stuck = state.activation_marker_path("first_inference")
    stuck.mkdir(parents=True, exist_ok=True)
    (stuck / "child").write_text("x")
    result = state.reset_state()

    # Removable state was still removed.
    assert not state.consent_path().exists()
    assert not state.client_id_path().exists()
    assert result.client_id.succeeded is True
    assert len(result.activation_markers) == 1
    assert result.activation_markers[0].succeeded is False


def test_session_id_is_process_stable_and_resettable(fake_home, monkeypatch):
    monkeypatch.setattr(state, "_session_id", None)
    first = state.session_id()
    assert state.session_id() == first
    assert len(first) == 36


def test_rotate_client_id_preserves_consent_and_clears_markers(fake_home):
    state.record_consent(True, rapid_mlx_version="0.15.0")
    original = state.get_or_create_client_id()
    marker = state.activation_marker_path("first_inference")
    marker.touch()

    rotated = state.rotate_client_id()

    assert rotated != original
    assert not marker.exists()
    assert state.get_consent_state() is not None
    assert state.get_consent_state().consent is True


def test_rotate_client_id_creates_identity_when_old_one_is_absent(fake_home):
    assert not state.client_id_path().exists()
    rotated = state.rotate_client_id()
    assert state.client_id_path().read_text().strip() == rotated


def test_rotate_client_id_reports_unremovable_marker(fake_home):
    stuck = state.activation_marker_path("first_inference")
    stuck.mkdir(parents=True)
    (stuck / "child").touch()
    with pytest.raises(OSError, match="identity rotation could not remove"):
        state.rotate_client_id()


def test_rotate_client_id_reports_marker_enumeration_failure(fake_home, monkeypatch):
    class ExplodingTelemetryDir:
        def __truediv__(self, name):
            return fake_home / ".rapid-mlx" / name

        def glob(self, _pattern):
            raise OSError("cannot scan")

    monkeypatch.setattr(
        state, "_default_telemetry_dir", lambda: ExplodingTelemetryDir()
    )
    with pytest.raises(OSError, match="cannot enumerate activation markers"):
        state.rotate_client_id()


def test_claim_activation_marker_is_one_shot_and_rejects_invalid_kind(fake_home):
    assert state.claim_activation_marker("first_inference") is True
    assert state.claim_activation_marker("first_inference") is False
    assert state.claim_activation_marker("../invalid") is False


def test_env_kill_switch_wins_over_consent(fake_home, monkeypatch):
    """Stored consent=True must NOT override RAPID_MLX_TELEMETRY=0.

    Critical contract: the env var is the documented "scripts can force
    off without touching the file" escape hatch. If consent could
    override it, CI runs would silently leak data the user thought they
    had disabled.
    """
    from rapid_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    assert is_enabled() is True
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert is_enabled() is False


def test_cli_flag_wins_over_consent(fake_home):
    """Even with consent=True and no env var, --no-telemetry forces off."""
    from rapid_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    assert is_enabled() is True
    assert is_enabled(cli_no_telemetry=True) is False


def test_env_force_on_is_ignored(fake_home, monkeypatch):
    """RAPID_MLX_TELEMETRY=1 must NOT silently opt the user in.

    Documented as kill-switch only — anything else means a CI agent or
    mistyped env var could enable telemetry without the user ever
    consenting. Default-off when no consent file exists is the contract.
    """
    from rapid_mlx.telemetry.state import is_enabled

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "1")
    assert is_enabled() is False  # still off — no stored consent
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "true")
    assert is_enabled() is False


@pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "off", "  0  ", ""])
def test_env_falsy_values_all_disable(fake_home, monkeypatch, falsy):
    from rapid_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", falsy)
    assert is_enabled() is False, f"falsy value {falsy!r} should kill-switch"


def test_client_id_idempotent(fake_home):
    from rapid_mlx.telemetry.state import get_or_create_client_id

    first = get_or_create_client_id()
    assert first
    assert len(first) == 36  # uuid4 string form
    assert get_or_create_client_id() == first


def test_client_id_creation_ignores_chmod_failure(fake_home, monkeypatch):
    from rapid_mlx.telemetry import state

    monkeypatch.setattr(
        state.os,
        "chmod",
        lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
    )

    created = state.get_or_create_client_id()

    assert state.client_id_path().read_text().strip() == created


def test_read_client_id_never_creates_state(fake_home):
    from rapid_mlx.telemetry.state import client_id_path, read_client_id

    assert read_client_id() is None
    assert not client_id_path().parent.exists()
    client_id_path().parent.mkdir(parents=True)
    client_id_path().write_text("stored-id\n")
    assert read_client_id() == "stored-id"


def test_client_id_user_zeroed_uuid_preserved(fake_home):
    """User can replace client_id with all-zeros to anonymize.

    Documented escape hatch: ``echo 00000000-... > telemetry-client-id``
    keeps the file present (so we don't regenerate) but contributes
    only to anonymous aggregate counts. If we silently overwrote, we'd
    break the documented user contract.
    """
    from rapid_mlx.telemetry.state import client_id_path, get_or_create_client_id

    zero = "00000000-0000-0000-0000-000000000000"
    path = client_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(zero + "\n")
    assert get_or_create_client_id() == zero


def test_reset_state_removes_preference_and_rotates_identity(fake_home):
    from rapid_mlx.telemetry.state import (
        client_id_path,
        consent_path,
        get_or_create_client_id,
        record_consent,
        reset_state,
    )

    record_consent(True, rapid_mlx_version="0.6.33")
    original = get_or_create_client_id()
    assert consent_path().exists()
    assert client_id_path().exists()
    lock = consent_path().with_name("telemetry-consent.yaml.lock")
    lock_inode = lock.stat().st_ino
    result = reset_state()
    assert result.consent_file.succeeded is True
    assert result.consent_lock.succeeded is True
    assert result.client_id.succeeded is True
    assert not consent_path().exists()
    assert lock.exists()
    assert lock.stat().st_ino == lock_inode
    rotated = client_id_path().read_text().strip()
    assert rotated != original
    # Idempotent — a second reset rotates the still-present client ID.
    second = reset_state()
    assert second.client_id.succeeded is True
    assert lock.stat().st_ino == lock_inode
    assert client_id_path().read_text().strip() != rotated


def test_reset_state_empty_home_creates_nothing(fake_home):
    from rapid_mlx.telemetry import state

    before = list(fake_home.rglob("*"))
    result = state.reset_state()

    assert list(fake_home.rglob("*")) == before == []
    assert result.consent_file.existed is False
    assert result.consent_lock.existed is False
    assert result.client_id.existed is False


def test_reset_state_reports_client_id_unlink_error(fake_home, monkeypatch):
    from rapid_mlx.telemetry import state

    identity = state.client_id_path()
    real_unlink = type(identity).unlink

    def fail_identity_unlink(path, *args, **kwargs):
        if path == identity:
            raise PermissionError("denied")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(identity), "unlink", fail_identity_unlink)

    result = state.reset_state()

    assert result.client_id.existed is True
    assert result.client_id.succeeded is False
    assert result.client_id.error_types == ("PermissionError",)


def test_reset_state_reports_client_id_rotation_error(fake_home, monkeypatch):
    state.get_or_create_client_id()
    monkeypatch.setattr(
        state,
        "get_or_create_client_id",
        lambda: (_ for _ in ()).throw(PermissionError("denied")),
    )

    result = state.reset_state()

    assert result.client_id.succeeded is True
    assert result.client_id_rotation_errors == ("PermissionError",)


def test_reset_state_preserves_sibling_lock_inode(fake_home):
    from rapid_mlx.telemetry import state

    state.record_consent(True, rapid_mlx_version="0.6.33")
    lock_path = state.consent_path().with_name(state.consent_path().name + ".lock")
    assert lock_path.exists()
    original_inode = lock_path.stat().st_ino
    state.reset_state()
    assert not state.consent_path().exists()
    assert lock_path.stat().st_ino == original_inode


def test_reset_does_not_delete_consent_when_sibling_lock_is_busy(
    fake_home, monkeypatch
):
    from rapid_mlx.telemetry import state

    state.record_consent(False, rapid_mlx_version="0.14.4")
    original = state.consent_path().read_bytes()
    moments = iter((0.0, 2.0))
    monkeypatch.setattr(state, "_lock_retry_clock", lambda: next(moments))
    monkeypatch.setattr(
        state.fcntl,
        "flock",
        lambda *_args: (_ for _ in ()).throw(
            BlockingIOError(state.errno.EAGAIN, "busy")
        ),
    )

    result = state.reset_state()

    assert result.incomplete
    assert result.consent_lock.error_types == ("BlockingIOError",)
    assert state.consent_path().read_bytes() == original


def test_reset_does_not_delete_consent_when_lock_cannot_open(fake_home, monkeypatch):
    from rapid_mlx.telemetry import state

    state.record_consent(False, rapid_mlx_version="0.14.4")
    original = state.consent_path().read_bytes()
    lock = state.consent_path().with_name("telemetry-consent.yaml.lock")
    real_open = state.os.open

    def deny_lock(path, flags, mode=0o777):
        if path == lock:
            raise PermissionError("lock denied")
        return real_open(path, flags, mode)

    monkeypatch.setattr(state.os, "open", deny_lock)
    result = state.reset_state()

    assert result.incomplete
    assert result.consent_lock.error_types == ("PermissionError",)
    assert state.consent_path().read_bytes() == original


def test_reset_does_not_delete_consent_on_nonretryable_lock_error(
    fake_home, monkeypatch
):
    from rapid_mlx.telemetry import state

    state.record_consent(False, rapid_mlx_version="0.14.4")
    original = state.consent_path().read_bytes()

    def fail_lock(_fd, operation):
        if operation == state.fcntl.LOCK_EX | state.fcntl.LOCK_NB:
            raise OSError(state.errno.EIO, "lock unavailable")

    monkeypatch.setattr(state.fcntl, "flock", fail_lock)
    result = state.reset_state()

    assert result.incomplete
    assert result.consent_lock.error_types == ("OSError",)
    assert state.consent_path().read_bytes() == original


def test_reset_retries_busy_lock_then_uses_it(fake_home, monkeypatch):
    from rapid_mlx.telemetry import state

    state.record_consent(False, rapid_mlx_version="0.14.4")
    real_flock = state.fcntl.flock
    attempts = 0
    sleeps = []

    def busy_once(fd, operation):
        nonlocal attempts
        if operation == state.fcntl.LOCK_EX | state.fcntl.LOCK_NB:
            attempts += 1
            if attempts == 1:
                raise BlockingIOError(state.errno.EAGAIN, "busy")
        return real_flock(fd, operation)

    monkeypatch.setattr(state.fcntl, "flock", busy_once)
    monkeypatch.setattr(state, "_lock_retry_sleep", sleeps.append)
    result = state.reset_state()

    assert result.consent_file.succeeded
    assert attempts == 2
    assert len(sleeps) == 1


def test_reset_state_ignores_marker_enumeration_error(fake_home, monkeypatch):
    from rapid_mlx.telemetry import state

    state.record_consent(True, rapid_mlx_version="0.6.33")
    telemetry_dir = state.consent_path().parent
    real_glob = type(telemetry_dir).glob

    def fail_marker_glob(path, pattern):
        if path == telemetry_dir and pattern == "activation_seen_*":
            raise OSError("marker directory denied")
        return real_glob(path, pattern)

    monkeypatch.setattr(type(telemetry_dir), "glob", fail_marker_glob)
    state.reset_state()

    assert not state.consent_path().exists()


def test_consent_source_reports_origin(fake_home, monkeypatch):
    """The status command shows users *why* telemetry is in its current
    state — verify each source string is correctly reported."""
    from rapid_mlx.telemetry.state import consent_source, record_consent

    assert "default" in consent_source()
    record_consent(True, rapid_mlx_version="0.6.33")
    assert "consent-file" in consent_source()
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert "env-var" in consent_source()
    monkeypatch.delenv("RAPID_MLX_TELEMETRY")
    assert "cli-flag" in consent_source(cli_no_telemetry=True)


def test_corrupt_consent_file_treated_as_unprompted(fake_home):
    """A garbage consent file must NOT crash the CLI — we treat it as
    'never prompted' so the next interactive run re-asks the user.
    Crashing here would block every serve invocation."""
    from rapid_mlx.telemetry.state import consent_path, get_consent_state

    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(":\n  not valid yaml :: at all")
    assert get_consent_state() is None


def test_consent_file_atomic_write(fake_home):
    """``record_consent`` writes via temp + rename so a SIGINT mid-write
    can't leave a half-file. The .tmp file should NOT be present after
    a successful write."""
    from rapid_mlx.telemetry.state import consent_path, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    leftover = consent_path().with_suffix(consent_path().suffix + ".tmp")
    assert not leftover.exists()


def test_record_consent_cleans_up_stale_tmp(fake_home):
    """Simulate an interrupted previous write by pre-planting a .tmp.
    record_consent must overwrite it cleanly and leave nothing behind."""
    import yaml

    from rapid_mlx.telemetry.state import (
        consent_path,
        get_consent_state,
        record_consent,
    )

    cpath = consent_path()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    stale = cpath.with_suffix(cpath.suffix + ".tmp")
    stale.write_text("partial: junk\nthis is not valid")
    assert stale.exists()

    record_consent(True, rapid_mlx_version="0.6.33")
    assert not stale.exists(), "stale .tmp should be cleaned up"
    state = get_consent_state()
    assert state is not None
    assert state.consent is True
    # And the real consent file is well-formed YAML.
    parsed = yaml.safe_load(cpath.read_text())
    assert parsed["consent"] is True


def test_record_consent_preserves_v2_and_desktop_fields(fake_home):
    import yaml

    from rapid_mlx.telemetry import state

    path = state.consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "consent: true\n"
        "desktop_consent: false\n"
        "notice_revision_seen: 1\n"
        "future_key: keepme\n"
    )

    state.record_consent(False, rapid_mlx_version="0.15.1")

    data = yaml.safe_load(path.read_text())
    assert data["consent"] is False
    assert data["desktop_consent"] is False
    assert data["notice_revision_seen"] == 1
    assert data["future_key"] == "keepme"


def test_record_consent_falls_back_after_bounded_lock_retry(fake_home, monkeypatch):
    import yaml

    from rapid_mlx.telemetry import state

    path = state.consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("future_key: keepme\n")
    lock_path = path.with_name(path.name + ".lock")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, os, sys\n"
                "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "sys.stdin.buffer.read(1)\n"
            ),
            str(lock_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "locked\n"

    now = [0.0]
    sleeps = []

    def fake_clock():
        return now[0]

    def fake_sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(state, "_lock_retry_clock", fake_clock)
    monkeypatch.setattr(state, "_lock_retry_sleep", fake_sleep)
    try:
        state.record_consent(False, rapid_mlx_version="0.15.1")
    finally:
        assert holder.stdin is not None
        holder.stdin.write("x")
        holder.stdin.close()
        holder.wait(timeout=5)

    data = yaml.safe_load(path.read_text())
    assert data["consent"] is False
    assert data["future_key"] == "keepme"
    assert sleeps
    assert sum(sleeps) <= 1.5


def test_record_consent_reports_write_path_and_os_error(fake_home, monkeypatch):
    from rapid_mlx.telemetry import state

    path = state.consent_path()

    def no_space(_path, _data):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(state, "_atomic_write_consent", no_space)
    with pytest.raises(OSError) as raised:
        state.record_consent(False, rapid_mlx_version="0.15.1")

    message = str(raised.value)
    assert message.startswith(f"cannot write {path}: ")
    assert "No space left on device" in message
    assert "unreadable" not in message


def test_record_consent_falls_back_with_chmod_000_lock(fake_home):
    import yaml

    from rapid_mlx.telemetry import state

    path = state.consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("future_key: keepme\n")
    lock_path = path.with_name(path.name + ".lock")
    lock_path.write_text("")
    lock_path.chmod(0)
    try:
        state.record_consent(False, rapid_mlx_version="0.15.1")
    finally:
        lock_path.chmod(0o600)
    data = yaml.safe_load(path.read_text())
    assert data["consent"] is False
    assert data["future_key"] == "keepme"


def test_record_consent_replaces_unreadable_record_without_notice(fake_home):
    import yaml

    from rapid_mlx.telemetry import state

    path = state.consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unknown_key: must_be_replaced\n")
    path.chmod(0)
    state.record_consent(False, rapid_mlx_version="0.15.1")
    data = yaml.safe_load(path.read_text())
    assert set(data) == {
        "consent",
        "prompted_at",
        "prompted_version",
        "schema_version",
    }
    assert data["consent"] is False


def test_record_consent_replaces_unreadable_record_and_marks_delivered_notice(
    fake_home, monkeypatch
):
    import yaml

    import rapid_mlx
    from rapid_mlx.telemetry import consent_runtime, state
    from rapid_mlx.telemetry.consent_decision import (
        DISCLOSURE_REVISION,
        ProcessRole,
    )

    monkeypatch.setattr(rapid_mlx, "__version__", "0.15.1")
    consent_runtime._reset_runtime_state_for_tests()
    consent_runtime.startup(role=ProcessRole.INTERACTIVE_CLI)
    path = state.consent_path()
    path.write_text("unknown_key: must_be_replaced\n")
    path.chmod(0)
    state.record_consent(False, rapid_mlx_version="0.15.1")
    data = yaml.safe_load(path.read_text())
    assert set(data) == {
        "consent",
        "notice_revision_seen",
        "prompted_at",
        "prompted_version",
        "schema_version",
    }
    assert data["notice_revision_seen"] == DISCLOSURE_REVISION


def test_record_consent_file_mode_is_0600(fake_home):
    from rapid_mlx.telemetry import state

    state.record_consent(True, rapid_mlx_version="0.15.1")
    assert state.consent_path().stat().st_mode & 0o777 == 0o600


def test_record_consent_preserves_unreadable_directory(fake_home):
    from rapid_mlx.telemetry import state

    path = state.consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    with pytest.raises(OSError, match=r"cannot write .*telemetry-consent.yaml"):
        state.record_consent(False, rapid_mlx_version="0.15.1")

    assert path.is_dir()


def test_schema_version_mismatch_treated_as_unprompted(fake_home):
    """A consent file with a schema_version we don't recognize must
    be treated as 'never prompted' so the user gets re-asked under
    whatever the current disclosure copy is. Forward-compat for
    Phase 2+."""
    import yaml

    from rapid_mlx.telemetry.state import consent_path, get_consent_state

    cpath = consent_path()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(
        yaml.safe_dump(
            {
                "consent": True,
                "prompted_at": "2026-05-10T00:00:00Z",
                "prompted_version": "0.6.33",
                "schema_version": 99,  # future-version we don't know
            }
        )
    )
    assert get_consent_state() is None


def test_do_not_track_disables_like_orca(fake_home, monkeypatch):
    """``DO_NOT_TRACK=1`` / ``true`` win over stored consent; other values
    are ignored rather than guessed."""
    import rapid_mlx.telemetry.state as state

    state.record_consent(True, rapid_mlx_version="0.0.0")
    assert state.is_enabled()
    for value in ("1", "true", " TRUE "):
        monkeypatch.setenv("DO_NOT_TRACK", value)
        assert not state.is_enabled(), value
        assert "DO_NOT_TRACK" in state.consent_source()
    for value in ("0", "false", "", "yes"):
        monkeypatch.setenv("DO_NOT_TRACK", value)
        assert state.is_enabled(), value


def test_ci_markers_disable_telemetry(fake_home, monkeypatch):
    """A build machine is never a user: any CI marker present forces OFF."""
    import rapid_mlx.telemetry.state as state

    state.record_consent(True, rapid_mlx_version="0.0.0")
    assert state.is_enabled()
    for name in state.CI_ENV_VARS:
        monkeypatch.setenv(name, "true")
        assert not state.is_enabled(), name
        assert state.consent_source() == f"ci ({name} is set)"
        monkeypatch.delenv(name)
    # Present but empty is "unset" (CircleCI-style ``CI=`` clears).
    monkeypatch.setenv("CI", "")
    assert state.is_enabled()


def test_kill_switch_reason_precedence(fake_home, monkeypatch):
    import rapid_mlx.telemetry.state as state

    state.record_consent(True, rapid_mlx_version="0.0.0")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert state.consent_source().startswith("env-var (RAPID_MLX_TELEMETRY")
    monkeypatch.delenv("RAPID_MLX_TELEMETRY")
    assert state.consent_source().startswith("env-var (DO_NOT_TRACK")
    monkeypatch.delenv("DO_NOT_TRACK")
    assert state.consent_source() == "ci (GITHUB_ACTIONS is set)"
