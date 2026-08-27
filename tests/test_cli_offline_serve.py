# SPDX-License-Identifier: Apache-2.0
"""Tests for offline-serve refusal when a model is uncached (#2357).

``rapid-mlx serve <uncached-model>`` under ``HF_HUB_OFFLINE=1`` /
``TRANSFORMERS_OFFLINE=1`` used to fall through every download attempt (each
printing "First-time download" / "Pre-download skipped; server will retry"),
let the serve subprocess start, and end in misleading ``--mllm``/``--no-mllm``
lane advice even though neither flag can supply the missing checkpoint.

The fix (in ``_ensure_model_downloaded`` and ``main()``'s B2 gate) detects the
offline + uncached condition ONCE, states which repository is missing, points
to ``rapid-mlx pull`` and the expected cache location, and exits(1) before
server initialization — mirroring the TimeoutError / disk-space exits. A lane
override is never recommended when the checkpoint is simply absent. Cachedness
is judged by the single shared ``_cache_entry_is_runnable`` predicate, so a
fully-cached mflux / split-video / Whisper model is never refused.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from unittest.mock import patch

import pytest

from vllm_mlx import cli


def _make_serve_args(model: str) -> Namespace:
    """Minimal serve ``Namespace`` mirroring test_audio_alias_registry's helper
    so the audio-serve fork in ``serve_command`` is reachable under test."""
    return Namespace(
        model=model,
        _original_alias=model,
        embedding_model=None,
        served_model_name=None,
        no_mllm=True,
        mllm=False,
        max_tokens=None,
        api_key=None,
        timeout=60,
        max_request_bytes=None,
        cors_origins=None,
        rate_limit=0,
        log_level="INFO",
        host="127.0.0.1",
        port=8000,
        listen_fd=None,
        watchdog_ppid=None,
    )


def _uncached_probe(monkeypatch):
    """Force the runnability predicate to report "not cached runnable",
    so the model falls through to the offline refusal / download path."""
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: False)


def test_offline_hub_mode_detects_env_switches(monkeypatch):
    """Both offline switches flip the helper on; any truthy value counts."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert cli._offline_hub_mode_active() is True

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "true")
    assert cli._offline_hub_mode_active() is True

    # Both absent -> online.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert cli._offline_hub_mode_active() is False


def test_offline_uncached_serve_refuses_before_download(monkeypatch, capsys):
    """Offline + uncached must exit(1) with one actionable message and NOT
    attempt the download/mirror path (no repeated "First-time download")."""
    _uncached_probe(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)

    sentinel = []

    def _fail(*a, **k):
        sentinel.append(a)
        raise AssertionError("download/mirror path must not be reached offline")

    for name in ("_check_disk_space", "_try_mirror_prefetch"):
        monkeypatch.setattr(cli, name, _fail)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert exc.value.code == 1

    out = capsys.readouterr()
    assert "badorg/offline-missing-model is not cached" in out.err
    assert "network is unavailable (offline mode is enabled)" in out.err
    assert "rapid-mlx pull badorg/offline-missing-model" in out.err
    assert "cache location" in out.err
    # No repeated download phase, no "server will retry".
    assert "server will retry" not in out.err
    assert "First-time download" not in out.err
    assert sentinel == []  # neither disk-space nor mirror was attempted


def test_offline_refusal_counts_transformer_offline_too(monkeypatch, capsys):
    _uncached_probe(monkeypatch)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert exc.value.code == 1
    assert "not cached and the network is unavailable" in capsys.readouterr().err


def test_offline_local_path_is_noop_not_refused(monkeypatch, capsys):
    """A local path is a no-op even under offline mode — never refused."""
    import tempfile

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with tempfile.TemporaryDirectory() as d:
        cli._ensure_model_downloaded(d)  # os.path.exists -> early return
    assert "is not cached" not in capsys.readouterr().err


def test_offline_cached_repo_is_noop_not_refused(monkeypatch, capsys):
    """A fully-cached repo (any modality, judged by the single runnability
    probe core) never hits the offline refusal."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(cli, "_cache_runnability", lambda name: True)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: True)
    cli._ensure_model_downloaded("acme/already-cached")
    assert "is not cached" not in capsys.readouterr().err


def test_offline_cached_mflux_is_noop_not_refused(monkeypatch, capsys):
    """A fully-cached mflux checkpoint (no root ``model*.safetensors``, so a
    text-only ``is_repo_cached`` read misses it) must NOT be refused — the
    shared runnability probe core accepts it (codex #2357-P1)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(cli, "_cache_runnability", lambda name: True)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: True)
    cli._ensure_model_downloaded("acme/qwen-image-cached")
    assert "is not cached" not in capsys.readouterr().err


def test_disk_gate_systemexit_still_propagates(monkeypatch, capsys):
    """Below the offline refusal, a disk-space-gate ``SystemExit`` must still
    clear the spinner and propagate (not be swallowed by the runnability /
    cache refactor this file tests). Guards the re-raise boundary directly
    under ``_cache_entry_is_runnable``."""
    _uncached_probe(monkeypatch)  # not runnable, not offline -> reaches disk gate
    monkeypatch.setenv("HF_HUB_OFFLINE", "")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)
    monkeypatch.setattr(
        cli, "_check_disk_space", lambda *a, **k: (_ for _ in ()).throw(SystemExit(3))
    )
    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert exc.value.code == 3


class _DiskSpaceProbeError(Exception):
    """Raised from ``_check_disk_space`` to prove the download path was
    entered (not refused for offline). A real exception subclass, NOT
    ``StopIteration`` — under PEP 479 a StopIteration sentinel thrown through a
    generator becomes RuntimeError, so the assertion would fail on
    interpreter/configs where that wraps the throw."""


def test_online_uncached_still_attempts_download(monkeypatch, capsys):
    """Without offline switches, an uncached model still proceeds to the
    download path (no refusal) — connectivity may be available."""
    _uncached_probe(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)

    # The download path reaches _check_disk_space first; swallow it so the
    # only assertion is that we did NOT hard-refuse for offline.
    def _disk_gate_reached(*a, **k):
        raise _DiskSpaceProbeError("entered disk-space gate")

    monkeypatch.setattr(cli, "_check_disk_space", _disk_gate_reached)
    with pytest.raises(_DiskSpaceProbeError):
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert "is not cached and the network is unavailable" not in capsys.readouterr().err


def test_gate_refuses_offline_uncached_before_notices(monkeypatch, capsys):
    """``main()``'s B2 confirmation gate must refuse an offline + uncached
    serve BEFORE printing any "Resolving"/"About to download"/"Proceeding"
    notice. ``main()`` drives the gate with a patched isatty()==True and an
    uncached repo id; the offline refusal fires first (SystemExit 1), and the
    size-estimate + ``confirm_or_abort`` path is never reached (#2357-P2).
    """
    import vllm_mlx._download_gate as gate

    # Outer gate condition: text-only probe reports uncached (no root
    # ``model*.safetensors``). Offline refusal scope: the shared runnability
    # predicate also reports False, so the refusal must fire.
    monkeypatch.setattr(gate, "is_repo_cached", lambda name: False)
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: False)
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Interactive so the gate's confirmation branch is active.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # Auto-pull off by default; explicit in case a vendor sets it.
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)

    # The gate refuses first and exits(1), so serve_command is never reached
    # (nor is the spinner'd size estimate / confirm path, which live below the
    # offline short-circuit in main()). Guard both to catch a regression.
    with (
        patch.object(
            sys, "argv", ["rapid-mlx", "serve", "badorg/offline-missing-model"]
        ),
        patch.object(
            cli, "serve_command", side_effect=AssertionError("must not dispatch")
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 1

    out = capsys.readouterr()
    assert "badorg/offline-missing-model is not cached" in out.err
    assert "network is unavailable (offline mode is enabled)" in out.err
    # No contradictory download notice precedes the single refusal.
    assert "About to download" not in out.out
    assert "Proceeding" not in out.out


def test_serve_audio_alias_refuses_offline_uncached(monkeypatch, capsys):
    """``serve whisper`` (a short audio alias with no '/') never reaches
    main()'s B2 gate, and ``_serve_audio_mode`` loads weights lazily — so the
    audio fork itself must refuse an offline + uncached model BEFORE booting
    the audio server (codex #2357-P1-a)."""
    from vllm_mlx.audio import probe
    from vllm_mlx.audio.registry import AudioAliasEntry

    # ``serve_command`` gates audio aliases on ``require_audio_or_exit`` at the
    # top, which exits(2) when ``mlx_audio`` is absent (base install / the
    # no-MLX CI lane). Stub that availability seam so the test still reaches
    # the offline-refusal fork below even with no ``mlx-audio`` installed —
    # same technique neighbouring audio tests use (``_spy`` / probe stubs).
    monkeypatch.setattr(probe, "is_audio_model_alias", lambda _name: True)
    monkeypatch.setattr(probe, "require_audio_or_exit", lambda _name: None)

    # The resolved audio entry drives runnability + the refusal message.
    entry = AudioAliasEntry(
        alias="whisper",
        type="stt",
        hf_id="mlx-community/whisper-tiny-mlx",
        family="whisper",
    )
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    reached_audio_boot = []

    def _cannot_boot(*a, **k):
        reached_audio_boot.append(a)
        raise AssertionError("audio server must not boot for offline+uncached")

    with (
        patch.object(cli, "_resolve_audio_model_for_serve", return_value=entry),
        patch.object(cli, "_serve_audio_mode", side_effect=_cannot_boot),
    ):
        args = _make_serve_args("whisper")
        with pytest.raises(SystemExit) as exc:
            cli.serve_command(args)
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "mlx-community/whisper-tiny-mlx is not cached" in out.err
    assert "network is unavailable (offline mode is enabled)" in out.err
    assert reached_audio_boot == []


def test_gate_does_not_refuse_when_wan_local_dir_set(monkeypatch, capsys, tmp_path):
    """An offline user serving a Wan video alias with a valid
    ``RAPID_MLX_WAN_MODEL_DIR`` local checkpoint must NOT be refused — Wan's
    own lane loads from the local dir and never goes through
    ``_ensure_model_downloaded`` (codex #2357-P1-b)."""
    import vllm_mlx._download_gate as gate

    (tmp_path / "checkpoint").mkdir()
    monkeypatch.setattr(gate, "is_repo_cached", lambda name: False)
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: False)
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("RAPID_MLX_WAN_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)

    confirmed = []

    # Below the (skipped) offline refusal, the gate lands in the size-estimate
    # + confirm path. Stub both so the only assertion is that the refusal did
    # NOT fire; serve_command must still not be reached (gate self-skips or
    # confirm handles it), so we patch it to a sentinel just in case.
    monkeypatch.setattr(gate, "estimate_repo_size_bytes", lambda name: 1)
    monkeypatch.setattr(gate, "confirm_or_abort", lambda *a, **k: confirmed.append(a))
    dispatched = []
    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "Anes1032/Wan2.2-TI2V-5B-mlx-q8"],
        ),
        patch.object(cli, "serve_command", side_effect=dispatched.append),
    ):
        cli.main()

    out = capsys.readouterr()
    assert "is not cached and the network is unavailable" not in out.err
    assert confirmed != []  # the refusal was skipped; the confirm path ran
    assert dispatched != []  # … and serve still dispatched the Wan model


def test_gate_wan_dir_set_does_not_exempt_text_model(monkeypatch, capsys, tmp_path):
    """A stray ``RAPID_MLX_WAN_MODEL_DIR`` must NOT exempt an unrelated text
    model from the offline refusal — the exemption is scoped to the video-gen
    lane (codex #2357-P2)."""
    import vllm_mlx._download_gate as gate

    monkeypatch.setattr(gate, "is_repo_cached", lambda name: False)
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: False)
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("RAPID_MLX_WAN_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)

    with (
        patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "mlx-community/Qwen3.5-4B-4bit"],
        ),
        patch.object(
            cli, "serve_command", side_effect=AssertionError("must not dispatch")
        ),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "mlx-community/Qwen3.5-4B-4bit is not cached" in out.err
    assert "network is unavailable (offline mode is enabled)" in out.err


def test_gate_skips_offline_refusal_for_attached_client(monkeypatch, capsys):
    """A chat/bench attached client (--base-url/--port pointing at an existing
    server) must NOT be refused for a model absent from the local cache — the
    named model lives remotely and is never meant to be downloaded locally
    (codex #2357-P1)."""
    import vllm_mlx._download_gate as gate

    monkeypatch.setattr(gate, "is_repo_cached", lambda name: False)
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: False)
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    dispatched = []
    with (
        patch.object(
            sys,
            "argv",
            [
                "rapid-mlx",
                "chat",
                "mlx-community/Qwen3.5-4B-4bit",
                "--base-url",
                "http://127.0.0.1:9999",
            ],
        ),
        patch.object(cli, "chat_command", side_effect=dispatched.append),
    ):
        cli.main()

    out = capsys.readouterr()
    assert "is not cached and the network is unavailable" not in out.err
    assert dispatched != []  # attached client proceeded to chat_command


def test_offline_hub_mode_flags_parsed_independently(monkeypatch):
    """HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE are parsed independently and
    OR-ed: one switch being disabled must NOT mask the other being enabled.

    Regression for the parser folding both into a single string
    (``_is_true(get(A) or get(B))``) which turned ``HF_HUB_OFFLINE=0`` +
    ``TRANSFORMERS_OFFLINE=1`` into offline-INACTIVE.
    """
    # HF=0 would short-circuit the fold to "0"; TRANSFORMERS=1 must still win.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert cli._offline_hub_mode_active() is True

    # And the mirror: TRANSFORMERS=0 must not mask HF=1.
    monkeypatch.setenv("HF_HUB_OFFLINE", "yes")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    assert cli._offline_hub_mode_active() is True

    # Only when BOTH are disabled is offline truly inactive (no masking).
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "false")
    assert cli._offline_hub_mode_active() is False


def test_cache_probe_fault_fails_open_for_offline(monkeypatch):
    """A typed cachedness-probe fault (permission / malformed) is INCONCLUSIVE.

    It must NOT conclude "runnable" (would skip a needed download / show a
    broken checkmark), and it must NOT conclude "uncached" (would make the
    offline refusal fire on a model that may well be cached). The tri-state
    core returns ``None``; the boolean wrapper collapses ``None`` -> ``False``
    for skip-download/inventory callers; the offline-refusal caller (which
    compares ``is False``) must NOT refuse on ``None``.
    """
    # Induce an expected probe fault in resolve_audio_alias.
    monkeypatch.setattr(
        "vllm_mlx.audio.registry.resolve_audio_alias",
        lambda repo: (_ for _ in ()).throw(OSError("denied")),
    )
    # Tri-state core: inconclusive, not a baked True (regression) nor False.
    assert cli._cache_runnability("any/repo") is None
    # Boolean wrapper collapses None -> False so downloaders/inventory treat a
    # probe fault as NOT runnable (never skip a needed download / never list a
    # broken entry as runnable).
    assert cli._cache_entry_is_runnable("any/repo") is False

    # Same for a malformed-cache KeyError / structural ValueError.
    monkeypatch.setattr(
        "vllm_mlx.audio.registry.resolve_audio_alias",
        lambda repo: (_ for _ in ()).throw(KeyError("hdr")),
    )
    assert cli._cache_runnability("any/repo") is None
    assert cli._cache_entry_is_runnable("any/repo") is False

    # The offline-refusal caller equates a verdict with established-uncached
    # ONLY via ``is False``; on ``None`` it must NOT refuse. Simulate the B2
    # serve gate's condition for an offline + probe-faulting model.
    monkeypatch.setattr(
        "vllm_mlx.audio.registry.resolve_audio_alias",
        lambda repo: (_ for _ in ()).throw(ValueError("bad index")),
    )
    assert cli._cache_runnability("any/repo") is None
    assert cli._cache_runnability("any/repo") is not False


def test_ensure_model_downloaded_does_not_refuse_offline_on_probe_fault(
    monkeypatch, capsys
) -> None:
    """The real ``_ensure_model_downloaded`` refusal path must NOT fire when a
    cachedness probe faults (inconclusive), even offline.

    Codex R2: the boolean wrapper collapses ``None`` -> ``False``; the refusal
    must therefore key on ``is False`` in the production path, not on the
    boolean result, or a permission/malformed-cache fault would be reported as
    definitively uncached and refuse the serve. Setting offline + inducing the
    probe exception, the offline refusal (a ``SystemExit`` with the
    "not cached and the network is unavailable" message) must NOT occur; the
    function falls through to the normal online path.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Induce a probe fault: _cache_runnability -> resolve_audio_alias raises.
    monkeypatch.setattr(
        "vllm_mlx.audio.registry.resolve_audio_alias",
        lambda repo: (_ for _ in ()).throw(OSError("denied")),
    )
    # The refusal must not fire, so the function proceeds to the download path;
    # stub the post-refusal network steps so we only exercise the gate itself.
    monkeypatch.setattr(cli, "_check_disk_space", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: True)
    # No SystemExit (offline refusal) should be raised on the probe fault.
    cli._ensure_model_downloaded("some/repo")
    err = capsys.readouterr().err
    assert "is not cached and the network is unavailable" not in err
