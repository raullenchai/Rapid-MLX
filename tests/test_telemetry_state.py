# SPDX-License-Identifier: Apache-2.0
"""Pin the consent precedence contract.

The four-layer precedence (CLI flag > env-var kill switch > stored
consent > default off) is the *single decision* every event-emit site
in Phase 2+ will rely on. Breaking it silently re-enables (or
silently disables) telemetry for the whole user base, which is the
exact failure mode this issue exists to avoid. Test it directly.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Reroute ``Path.home()`` so consent files land under tmp.

    The telemetry state module reads ``Path.home()`` at call time on
    purpose (so tests can do exactly this). Setting ``HOME`` is the
    documented way to override on POSIX.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    # Force-reload the state module so any cached path objects (there
    # shouldn't be any, but defence in depth) get rebuilt under the new
    # HOME.
    import vllm_mlx.telemetry.state as state

    importlib.reload(state)
    return tmp_path


def test_default_is_off(fake_home):
    from vllm_mlx.telemetry.state import is_enabled

    assert is_enabled() is False


def test_consent_round_trip(fake_home):
    from vllm_mlx.telemetry.state import (
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
    assert state.schema_version == 1
    assert state.prompted_at.endswith("Z")
    assert is_enabled() is True


def test_env_kill_switch_wins_over_consent(fake_home, monkeypatch):
    """Stored consent=True must NOT override RAPID_MLX_TELEMETRY=0.

    Critical contract: the env var is the documented "scripts can force
    off without touching the file" escape hatch. If consent could
    override it, CI runs would silently leak data the user thought they
    had disabled.
    """
    from vllm_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    assert is_enabled() is True
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert is_enabled() is False


def test_cli_flag_wins_over_consent(fake_home):
    """Even with consent=True and no env var, --no-telemetry forces off."""
    from vllm_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    assert is_enabled() is True
    assert is_enabled(cli_no_telemetry=True) is False


def test_env_force_on_is_ignored(fake_home, monkeypatch):
    """RAPID_MLX_TELEMETRY=1 must NOT silently opt the user in.

    Documented as kill-switch only — anything else means a CI agent or
    mistyped env var could enable telemetry without the user ever
    consenting. Default-off when no consent file exists is the contract.
    """
    from vllm_mlx.telemetry.state import is_enabled

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "1")
    assert is_enabled() is False  # still off — no stored consent
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "true")
    assert is_enabled() is False


@pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "off", "  0  ", ""])
def test_env_falsy_values_all_disable(fake_home, monkeypatch, falsy):
    from vllm_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", falsy)
    assert is_enabled() is False, f"falsy value {falsy!r} should kill-switch"


def test_client_id_idempotent(fake_home):
    from vllm_mlx.telemetry.state import get_or_create_client_id

    first = get_or_create_client_id()
    assert first
    assert len(first) == 36  # uuid4 string form
    assert get_or_create_client_id() == first


def test_client_id_user_zeroed_uuid_preserved(fake_home):
    """User can replace client_id with all-zeros to anonymize.

    Documented escape hatch: ``echo 00000000-... > telemetry-client-id``
    keeps the file present (so we don't regenerate) but contributes
    only to anonymous aggregate counts. If we silently overwrote, we'd
    break the documented user contract.
    """
    from vllm_mlx.telemetry.state import client_id_path, get_or_create_client_id

    zero = "00000000-0000-0000-0000-000000000000"
    path = client_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(zero + "\n")
    assert get_or_create_client_id() == zero


def test_reset_state_removes_both_files(fake_home):
    from vllm_mlx.telemetry.state import (
        client_id_path,
        consent_path,
        get_or_create_client_id,
        record_consent,
        reset_state,
    )

    record_consent(True, rapid_mlx_version="0.6.33")
    get_or_create_client_id()
    assert consent_path().exists()
    assert client_id_path().exists()
    reset_state()
    assert not consent_path().exists()
    assert not client_id_path().exists()
    # Idempotent — second reset_state on missing files must not raise.
    reset_state()


def test_consent_source_reports_origin(fake_home, monkeypatch):
    """The status command shows users *why* telemetry is in its current
    state — verify each source string is correctly reported."""
    from vllm_mlx.telemetry.state import consent_source, record_consent

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
    from vllm_mlx.telemetry.state import consent_path, get_consent_state

    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(":\n  not valid yaml :: at all")
    assert get_consent_state() is None


def test_consent_file_atomic_write(fake_home):
    """``record_consent`` writes via temp + rename so a SIGINT mid-write
    can't leave a half-file. The .tmp file should NOT be present after
    a successful write."""
    from vllm_mlx.telemetry.state import consent_path, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    leftover = consent_path().with_suffix(consent_path().suffix + ".tmp")
    assert not leftover.exists()


def test_record_consent_cleans_up_stale_tmp(fake_home):
    """Simulate an interrupted previous write by pre-planting a .tmp.
    record_consent must overwrite it cleanly and leave nothing behind."""
    import yaml

    from vllm_mlx.telemetry.state import (
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


def test_schema_version_mismatch_treated_as_unprompted(fake_home):
    """A consent file with a schema_version we don't recognize must
    be treated as 'never prompted' so the user gets re-asked under
    whatever the current disclosure copy is. Forward-compat for
    Phase 2+."""
    import yaml

    from vllm_mlx.telemetry.state import consent_path, get_consent_state

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
