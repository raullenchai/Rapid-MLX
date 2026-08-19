# SPDX-License-Identifier: Apache-2.0
"""First-run guide — install → first-token conversion.

Covers the ``vllm_mlx/first_run.py`` helpers and their three wiring points in
``vllm_mlx/cli.py``:

  * P0-1 — ``chat`` / ``run`` with no model → starter auto-select in
    ``main()`` (the known-good starter alias chosen, notice printed only on a
    TTY; the standard download gate is left untouched — the ~3.1 GB starter is
    under its 10 GiB confirm threshold — and non-interactive sessions fall
    through to that gate exactly as before this feature).
  * P0-2 — bare ``rapid-mlx`` in an interactive terminal → nameplate + exit 0;
    non-interactive → unchanged help + exit 1.
  * P0-3 — one-time "connect your agent" tip after the first chat that
    produced a response (marker + gating helpers).

Fully offline: no model load, no network, no real HF cache dependency — every
machine-state probe is monkeypatched.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import vllm_mlx.cli as cli
import vllm_mlx.first_run as fr


# ======================================================================
# P0-1: no-model → starter selection
# ======================================================================
def test_select_chat_default_cold_cache_returns_starter(monkeypatch):
    monkeypatch.setattr(fr, "cached_known_aliases", lambda: [])
    alias, already_cached = fr.select_chat_default()
    assert alias == fr.FIRST_RUN_MODEL
    assert already_cached is False


def test_select_chat_default_always_starter_never_arbitrary_cached(monkeypatch):
    # Even when other (possibly non-chat) models are cached, we pick the
    # known-good starter — never the most-recently-downloaded alias.
    monkeypatch.setattr(
        fr,
        "cached_known_aliases",
        lambda: [("embeddinggemma-300m-8bit", 200.0), ("gpt-oss-20b-mxfp4-q8", 100.0)],
    )
    alias, already_cached = fr.select_chat_default()
    assert alias == fr.FIRST_RUN_MODEL
    # Starter is not among the cached set → not already-downloaded.
    assert already_cached is False


def test_select_chat_default_reports_starter_cached(monkeypatch):
    # ``already_cached`` reflects whether the STARTER itself is downloaded.
    monkeypatch.setattr(
        fr,
        "cached_known_aliases",
        lambda: [(fr.FIRST_RUN_MODEL, 100.0), ("gpt-oss-20b-mxfp4-q8", 200.0)],
    )
    alias, already_cached = fr.select_chat_default()
    assert alias == fr.FIRST_RUN_MODEL
    assert already_cached is True


def test_cached_known_aliases_maps_sorts_and_drops_unmapped(monkeypatch):
    # size element is index 1 in the (repo, size, mtime) tuple.
    fake_rows = [
        ("org/Unmapped-Model", 0, 100.0),  # not in aliases.json → dropped
        ("mlx-community/Qwen3.5-4B-4bit", 0, 300.0),  # newest
        ("mlx-community/Gpt-Oss-20B", 0, 200.0),
    ]

    class _P:
        def __init__(self, hf):
            self.hf_path = hf

    fake_profiles = {
        "qwen3.5-4b-4bit": _P("mlx-community/Qwen3.5-4B-4bit"),
        "gpt-oss-20b-mxfp4-q8": _P("mlx-community/Gpt-Oss-20B"),
    }
    # String-form targets (not ``setattr(cli, ...)``): ``cached_known_aliases``
    # re-imports these from ``sys.modules`` at call time, and a sibling suite
    # (test_cli_argcomplete) pops+reimports ``vllm_mlx.cli``, so the module-level
    # ``cli`` reference here can go stale. Patching by dotted path re-resolves
    # the live module object and stays effective regardless of import churn.
    monkeypatch.setattr("vllm_mlx.cli._scan_hf_cache_models", lambda: fake_rows)
    monkeypatch.setattr("vllm_mlx.model_aliases.list_profiles", lambda: fake_profiles)

    rows = fr.cached_known_aliases()
    assert [a for a, _ in rows] == ["qwen3.5-4b-4bit", "gpt-oss-20b-mxfp4-q8"]


def test_cached_known_aliases_fail_silent(monkeypatch):
    def _boom():
        raise RuntimeError("broken cache dir")

    # Dotted-path target (see the note in the sibling test): survives the
    # ``vllm_mlx.cli`` pop+reimport that test_cli_argcomplete performs.
    monkeypatch.setattr("vllm_mlx.cli._scan_hf_cache_models", _boom)
    assert fr.cached_known_aliases() == []


# ======================================================================
# P0-2/P0-3: agent detection
# ======================================================================
class _Adapter:
    def __init__(self, detected):
        self._detected = detected

    def detect(self):
        return self._detected


def test_detected_agents_prefers_claude_code(monkeypatch):
    monkeypatch.setattr(
        "vllm_mlx.launch.ADAPTERS",
        {
            "continue-dev": _Adapter(True),
            "claude-code": _Adapter(True),
            "cline": _Adapter(False),
        },
    )
    agents = fr.detected_agents()
    assert agents[0] == "claude-code"  # preference order leads
    assert "continue-dev" in agents
    assert "cline" not in agents  # not detected
    assert fr.preferred_agent() == "claude-code"


def test_cursor_is_not_recommended_for_local_first_run(monkeypatch):
    monkeypatch.setattr(
        "vllm_mlx.launch.ADAPTERS",
        {"cursor": _Adapter(True), "claude-code": _Adapter(False)},
    )
    assert fr.detected_agents() == []
    assert fr.preferred_agent() is None


def test_detected_agents_detect_error_is_safe(monkeypatch):
    class _Raiser:
        def detect(self):
            raise OSError("probe blew up")

    monkeypatch.setattr("vllm_mlx.launch.ADAPTERS", {"claude-code": _Raiser()})
    assert fr.detected_agents() == []


# ======================================================================
# P0-2: nameplate
# ======================================================================
def test_nameplate_cold_cache_no_agent(monkeypatch):
    monkeypatch.setattr(fr, "cached_known_aliases", lambda: [])
    monkeypatch.setattr(fr, "preferred_agent", lambda: None)
    out = fr.build_nameplate("9.9.9")
    assert "Rapid-MLX 9.9.9" in out
    assert "rapid-mlx chat" in out
    assert "rapid-mlx recipe" in out
    assert "Smart + Fast" in out
    assert fr.FIRST_RUN_MODEL in out
    assert fr.FIRST_RUN_MODEL_SIZE in out
    assert "launch --all" in out  # generic signpost when no agent
    assert "Found in your cache" not in out


def test_nameplate_with_cache_and_agent(monkeypatch):
    # A cached (possibly non-chat) model is LISTED, but the chat suggestion
    # still points at the known-good starter — never the arbitrary cached alias.
    monkeypatch.setattr(fr, "cached_known_aliases", lambda: [("qwen3.5-9b-4bit", 10.0)])
    monkeypatch.setattr(fr, "preferred_agent", lambda: "claude-code")
    out = fr.build_nameplate("9.9.9")
    assert "Found in your cache: qwen3.5-9b-4bit" in out
    assert "rapid-mlx chat qwen3.5-9b-4bit" not in out
    assert fr.FIRST_RUN_MODEL in out  # chat target is the starter
    assert "rapid-mlx chat <model>" in out  # "pick a cached one" hint
    assert "launch claude-code" in out  # named agent signpost


def test_nameplate_starter_cached_says_already_downloaded(monkeypatch):
    monkeypatch.setattr(
        fr, "cached_known_aliases", lambda: [(fr.FIRST_RUN_MODEL, 10.0)]
    )
    monkeypatch.setattr(fr, "preferred_agent", lambda: None)
    out = fr.build_nameplate("9.9.9")
    assert "already downloaded" in out
    assert "launch --all" in out


# ======================================================================
# P0-3: one-time tip marker + text
# ======================================================================
def test_tip_marker_claimed_once_only(monkeypatch, tmp_path):
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(tmp_path / "state"))
    # The atomic claim wins exactly once (the exclusive-create), then every
    # subsequent claim loses — so a second concurrent session can't re-print.
    assert fr.claim_chat_agent_tip() is True
    assert fr.claim_chat_agent_tip() is False
    assert fr.claim_chat_agent_tip() is False


def test_tip_claim_failsafe_on_unwritable_state_dir(monkeypatch, tmp_path):
    # State dir sits under an existing FILE → the marker's parent mkdir raises
    # → claim returns False (never nag, never crash), not True.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(blocker / "state"))
    assert fr.claim_chat_agent_tip() is False


def test_state_dir_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom-state"
    monkeypatch.setenv("RAPID_MLX_STATE_DIR", str(target))
    assert fr._state_dir() == target


def test_tip_text_with_and_without_agent(monkeypatch):
    monkeypatch.setattr(fr, "preferred_agent", lambda: "claude-code")
    assert "launch claude-code" in fr.chat_agent_tip_text()
    monkeypatch.setattr(fr, "preferred_agent", lambda: None)
    assert "launch --all" in fr.chat_agent_tip_text()


# ======================================================================
# main() wiring — P0-1 auto-select
# ======================================================================
def _run_main_capture_chat(argv, *, stdin_tty=True):
    """Drive the REAL ``main()`` for a chat/run invocation, intercepting the
    dispatch to ``chat_command`` so nothing loads a model. Returns the captured
    ``args`` namespace (post auto-select + alias resolution)."""
    captured = {}

    def _capture(args):
        captured["args"] = args

    with (
        mock.patch.object(cli, "chat_command", _capture),
        mock.patch("vllm_mlx.telemetry.maybe_prompt_for_consent", return_value=False),
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        # Pretend everything is cached so the download-confirm gate never
        # blocks the test on an explicit-model run.
        mock.patch("vllm_mlx._download_gate.is_repo_cached", return_value=True),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=stdin_tty),
    ):
        cli.main()
    return captured.get("args")


def test_chat_no_model_auto_selects_starter(monkeypatch, capsys):
    monkeypatch.setattr(
        "vllm_mlx.first_run.select_chat_default",
        lambda: ("qwen3.5-4b-4bit", False),
    )
    args = _run_main_capture_chat(["chat"], stdin_tty=True)
    assert args is not None
    # The auto-selected alias flowed through normal resolution → _original_alias
    # holds the user-facing name, args.model the resolved HF path.
    assert getattr(args, "_original_alias", None) == "qwen3.5-4b-4bit"
    out = capsys.readouterr().out
    assert "No model specified — using qwen3.5-4b-4bit" in out


def test_run_alias_no_model_auto_selects_starter(monkeypatch, capsys):
    # ``run`` is an argparse alias of ``chat`` (Ollama compat) and the no-model
    # auto-select branch gates on {"chat", "run"}, so the alias must get the
    # same starter selection + notice, not just ``chat``.
    monkeypatch.setattr(
        "vllm_mlx.first_run.select_chat_default",
        lambda: ("qwen3.5-4b-4bit", False),
    )
    args = _run_main_capture_chat(["run"], stdin_tty=True)
    assert args is not None
    assert getattr(args, "_original_alias", None) == "qwen3.5-4b-4bit"
    assert "No model specified — using qwen3.5-4b-4bit" in capsys.readouterr().out


def test_chat_no_model_non_tty_starter_cached_proceeds_silently(monkeypatch, capsys):
    # Non-interactive + the starter already downloaded (zero download) →
    # proceeds silently (no notice, no prompt). The standard download gate's
    # own is_repo_cached check governs (helper stubs it cached).
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    monkeypatch.setattr(
        "vllm_mlx.first_run.select_chat_default",
        lambda: ("qwen3.5-4b-4bit", True),
    )
    args = _run_main_capture_chat(["chat"], stdin_tty=False)
    assert args is not None
    assert "No model specified" not in capsys.readouterr().out


def test_chat_no_model_non_tty_cold_cache_falls_through_silently(monkeypatch, capsys):
    # Non-interactive + a GENUINELY uncached starter (download gate ARMED:
    # is_repo_cached=False). The run must (a) NOT prompt — the confirm gate is
    # TTY-only — and (b) still fall through to dispatch, exactly as a bare
    # `rapid-mlx chat` behaved before this feature. Exiting 1 here would
    # silently regress documented scripted callers. Uses the gate probe (rather
    # than the everything-cached capture helper) so the cold-cache gate path is
    # actually exercised.
    monkeypatch.delenv("RAPID_MLX_AUTO_PULL", raising=False)
    confirm, dispatched = _run_main_gate_probe(
        ["chat"],
        auto_select_alias="qwen3.5-4b-4bit",
        starter_cached=False,
        stdin_tty=False,
    )
    assert confirm.called is False  # non-TTY → no y/N prompt
    assert dispatched is True  # fell through to chat_command, not exit 1
    out_err = capsys.readouterr()
    assert "No model specified" not in out_err.out  # notice is TTY-only
    assert "No model specified" not in out_err.err


def test_chat_explicit_model_skips_autoselect(monkeypatch, capsys):
    args = _run_main_capture_chat(["chat", "qwen3.5-9b-4bit"], stdin_tty=True)
    assert getattr(args, "_original_alias", None) == "qwen3.5-9b-4bit"
    assert "No model specified" not in capsys.readouterr().out


# ======================================================================
# main() wiring — download-confirm gate interaction
# ======================================================================
def _run_main_gate_probe(
    argv, *, auto_select_alias=None, starter_cached=False, stdin_tty=True
):
    """Drive ``main()`` with the download-confirm gate armed (model reported
    NOT cached by the gate's ``is_repo_cached``). Returns
    ``(confirm_or_abort_mock, dispatched)``, where ``dispatched`` is True iff
    the run reached ``chat_command`` (i.e. fell through to dispatch rather than
    aborting). ``starter_cached`` sets what ``select_chat_default`` reports
    about the starter (only affects the notice wording); ``stdin_tty`` toggles
    the interactivity the gate reads (the confirm prompt is TTY-only)."""
    confirm = mock.MagicMock()
    state = {"dispatched": False}

    def _dispatch(args):
        state["dispatched"] = True

    ctx = [
        mock.patch.object(cli, "chat_command", _dispatch),
        mock.patch("vllm_mlx.telemetry.maybe_prompt_for_consent", return_value=False),
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch("vllm_mlx._download_gate.is_repo_cached", return_value=False),
        mock.patch("vllm_mlx._download_gate.estimate_repo_size_bytes", return_value=0),
        mock.patch("vllm_mlx._download_gate.confirm_or_abort", confirm),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=stdin_tty),
    ]
    if auto_select_alias is not None:
        ctx.append(
            mock.patch(
                "vllm_mlx.first_run.select_chat_default",
                lambda: (auto_select_alias, starter_cached),
            )
        )
    with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7]:
        if auto_select_alias is not None:
            with ctx[8]:
                cli.main()
        else:
            cli.main()
    return confirm, state["dispatched"]


def test_autoselected_starter_uses_standard_download_gate():
    # Regression guard: the auto-selected starter must NOT special-case the
    # download gate. The starter is an unpinned Hugging Face repo whose
    # declared size we must actually verify — never assume — before waiving
    # confirmation, so it flows through ``estimate_repo_size_bytes`` +
    # ``confirm_or_abort`` exactly like an explicit model. (The "small model →
    # no prompt" decision belongs to confirm_or_abort's own 10 GiB threshold,
    # covered in test_download_gate — the ~3.1 GB starter is under it, so the
    # gate stays silent for it on its own, but the size is still checked.)
    # The redundant round-trip's *latency* is what we hide — behind a
    # "Resolving…" spinner, NOT by skipping the check.
    confirm, dispatched = _run_main_gate_probe(
        ["chat"], auto_select_alias="qwen3.5-4b-4bit"
    )
    assert confirm.called is True  # gate NOT bypassed
    assert dispatched is True  # and the run still reaches dispatch


def test_autoselected_starter_still_runs_disk_gate_in_prefetch(monkeypatch):
    # Belt-and-braces: the download-prep path always runs the DISK-SPACE gate
    # before pulling. ``_ensure_model_downloaded`` calls ``_check_disk_space``
    # (under the spinner) so a full disk still aborts cleanly regardless of the
    # confirm gate above.
    called = {"disk": False}

    def _fake_disk(model_name, force=False):
        called["disk"] = True

    # An HF-style repo id is not a local path, so the early ``os.path.exists``
    # return is naturally skipped — no need to patch os.
    monkeypatch.setattr(cli, "_check_disk_space", _fake_disk)
    monkeypatch.setattr(cli, "_try_mirror_prefetch", lambda *a, **k: True)
    monkeypatch.setattr("vllm_mlx._download_gate.is_repo_cached", lambda *a, **k: False)
    cli._ensure_model_downloaded("mlx-community/Qwen3.5-4B-MLX-4bit")
    assert called["disk"] is True


def test_explicit_uncached_model_still_confirms():
    # Control: an explicitly-typed, uncached model DOES hit the confirm gate.
    confirm, _dispatched = _run_main_gate_probe(["chat", "qwen3.5-9b-4bit"])
    assert confirm.called is True


# ======================================================================
# main() wiring — P0-2 bare-command nameplate branch
# ======================================================================
def _run_bare(*, stdout_tty, stdin_tty):
    with (
        mock.patch(
            "vllm_mlx.first_run.build_nameplate", return_value="NAMEPLATE-OK"
        ) as np,
        mock.patch(
            "vllm_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch.object(sys, "argv", ["rapid-mlx"]),
        mock.patch.object(sys.stdout, "isatty", return_value=stdout_tty),
        mock.patch.object(sys.stdin, "isatty", return_value=stdin_tty),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main()
    return np, exc.value.code


def test_bare_command_interactive_shows_nameplate(capsys):
    np, code = _run_bare(stdout_tty=True, stdin_tty=True)
    assert np.called is True
    assert code == 0
    assert "NAMEPLATE-OK" in capsys.readouterr().out


def test_bare_command_non_tty_falls_back_to_help():
    np, code = _run_bare(stdout_tty=False, stdin_tty=True)
    assert np.called is False  # nameplate never built off a TTY
    assert code == 1  # unchanged: help + exit 1
