# SPDX-License-Identifier: Apache-2.0
"""Tests for the model alias registry."""

import os

import pytest

from rapid_mlx.model_aliases import (
    RetiredModelAliasError,
    list_aliases,
    resolve_model,
    suggest_similar,
)


def test_known_alias_resolves():
    assert resolve_model("qwen3.5-9b-4bit") == "mlx-community/Qwen3.5-9B-4bit"
    assert resolve_model("llama3-3b-4bit") == "mlx-community/Llama-3.2-3B-Instruct-4bit"


def test_broken_ministral_3b_alias_is_rejected_before_loading():
    """#1367: the default VLM route hangs on its first text completion."""
    retired = "ministral-3b-4bit"
    assert retired not in list_aliases()
    with pytest.raises(RetiredModelAliasError, match="alias was retired"):
        resolve_model(retired)


def test_retired_alias_full_hf_path_remains_available_for_text_only_testing():
    hf_path = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
    assert resolve_model(hf_path) == hf_path


def test_local_directory_named_like_retired_alias_keeps_path_precedence(
    tmp_path, monkeypatch
):
    """A real local path is not alias support and keeps the path-first contract."""
    retired = "ministral-3b-4bit"
    (tmp_path / retired).mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_model(retired) == retired


def test_full_path_passes_through():
    assert resolve_model("mlx-community/Foo-Bar") == "mlx-community/Foo-Bar"
    assert resolve_model("/Users/me/local-model") == "/Users/me/local-model"


def test_unknown_name_passes_through():
    assert resolve_model("nonexistent-model") == "nonexistent-model"


def test_local_path_takes_priority_over_alias(tmp_path):
    """A local model directory matching an alias name should win."""
    local_dir = tmp_path / "qwen3.5-9b-4bit"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert resolve_model("qwen3.5-9b-4bit") == "qwen3.5-9b-4bit"
    finally:
        os.chdir(old_cwd)


def test_list_aliases_nonempty():
    aliases = list_aliases()
    assert len(aliases) >= 15
    assert "qwen3.5-9b-4bit" in aliases


def test_hermes_alias_not_llama():
    """Hermes-3 should be under its own name, not llama3-8b."""
    aliases = list_aliases()
    assert "llama3-8b" not in aliases
    assert "hermes3-8b-4bit" in aliases


def test_suggest_similar_stays_within_family():
    """Real reproduction: typing ``deepseek-v4-27b`` (non-existent variant)
    must suggest the deepseek-v4 family — NOT mix in deepseek-r1 which is
    a different model. Generic edit-distance ranking failed this; the
    family-aware filter exists to fix it."""
    suggestions = suggest_similar("deepseek-v4-27b")
    assert suggestions, "expected at least one suggestion"
    # Every suggestion must share the deepseek-v4 family — no deepseek-r1
    # bait-and-switch.
    for s in suggestions:
        assert s.startswith("deepseek-v4"), s


def test_suggest_similar_correctly_typo_for_close_size():
    """Typing ``qwen3.5-30b`` (typo for ``qwen3.5-35b-8bit``) should rank the
    correct alias first."""
    suggestions = suggest_similar("qwen3.5-30b")
    assert suggestions, "expected at least one suggestion"
    assert suggestions[0] == "qwen3.5-35b-8bit", suggestions


def test_suggest_similar_empty_for_nonsense():
    assert suggest_similar("xyzabc12345") == []


def test_suggest_similar_lets_legitimate_hf_ids_through():
    """Bare HF IDs must NOT match — otherwise the CLI fast-fail in
    ``main()`` would block legitimate single-segment HuggingFace
    repositories like ``gpt2`` and ``bert-base-uncased``."""
    assert suggest_similar("gpt2") == []
    assert suggest_similar("bert-base-uncased") == []


def test_suggest_similar_one_letter_no_match():
    """Single-character inputs must not match anything (would otherwise
    spuriously suggest with cutoff=0.5)."""
    assert suggest_similar("q") == []
    assert suggest_similar("g") == []


def test_suggest_similar_matches_partial_family_token():
    """A bare family name like ``hermes`` should suggest aliases that
    share that prefix (``hermes3-8b-4bit``), not return [] just because there's
    no exact ``hermes-foo`` separator pattern."""
    suggestions = suggest_similar("hermes")
    assert "hermes3-8b-4bit" in suggestions, suggestions


# --- Letter-only fallback (separator-mismatched names) ----------------


def test_suggest_similar_letter_fallback_handles_separator_mismatch():
    """Real bug from the field: ``rapid-mlx chat gemma4-27b`` returned
    zero suggestions because the strict family parser sees ``gemma4`` and
    no alias starts with ``gemma4`` (we have ``gemma-4-26b-4bit`` and
    ``gemma3-27b-4bit``). The letter-only fallback must catch this — extract
    ``gemma`` and match the whole gemma family."""
    suggestions = suggest_similar("gemma4-27b")
    assert suggestions, "letter-only fallback must produce gemma family suggestions"
    # All suggestions must be in the gemma family — no llama / qwen leakage.
    for s in suggestions:
        assert s.startswith("gemma"), s


def test_suggest_similar_letter_fallback_collapsed_separator():
    """User collapses our hyphen — ``mistral24b`` should still suggest
    ``mistral-24b-4bit``, not return []."""
    assert "mistral-24b-4bit" in suggest_similar("mistral24b")


def test_suggest_similar_letter_fallback_skips_legit_looking_names():
    """When the input has no size/quant suffix tokens (i.e., looks
    structurally like a legit single-segment HF repo ID), suggest_similar
    must return [] — not bait-and-switch ``gpt2`` to ``gpt-oss-20b-mxfp4-q8`` or
    ``qwen-coder`` to ``qwen3-coder-4bit``. The CLI layer's POPULAR_ALIASES
    fallback handles those cases at presentation time."""
    # ``gpt2`` has been pinned by test_suggest_similar_lets_legitimate_hf_ids_through;
    # this case adds the partial-family equivalent.
    assert suggest_similar("qwen-coder") == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("gemma4-27b", "gemma"),
        ("Gemma4-27b", "gemma"),  # lowercased
        ("gemma_4-27b", "gemma"),  # stops at non-letter
        ("mistral24b", "mistral"),
        ("qwen3.5-4b-4bit", "qwen"),  # stops at first digit
        ("123abc", ""),  # leading non-letter → empty
        ("", ""),  # empty input
        ("ab", "ab"),  # short prefix (caller enforces ≥3 minimum)
    ],
)
def test_letters_only_prefix(raw, expected):
    """Direct coverage for the letter-only family extraction. Caller
    (suggest_similar) enforces the 3-char minimum, so we don't here."""
    from rapid_mlx.model_aliases import _letters_only_prefix

    assert _letters_only_prefix(raw) == expected


def test_popular_aliases_curated_list_resolves():
    """Every entry in POPULAR_ALIASES (used as the user-facing
    'try one of these' fallback when zero fuzzy matches) must be a real
    alias in aliases.json — otherwise the error message is a lie."""
    from rapid_mlx.model_aliases import POPULAR_ALIASES

    aliases = list_aliases()
    missing = [a for a in POPULAR_ALIASES if a not in aliases]
    assert not missing, (
        f"POPULAR_ALIASES references non-existent aliases: {missing}. "
        f"Update rapid_mlx/model_aliases.py POPULAR_ALIASES tuple after "
        f"removing or renaming aliases."
    )


def test_draft_only_checkpoints_are_refused_with_the_precise_remedy():
    """The qwen3.6 MTP sidecars are qwen3_5_mtp checkpoints the pinned
    mlx-lm cannot serve as a primary; serving them used to die mid-load
    with a raw unsupported-architecture error. The gate must name the
    base alias that uses the draft automatically."""
    from rapid_mlx.model_aliases import draft_only_conflict

    conflict = draft_only_conflict("qwen3.6-35b-mtp-4bit")
    assert conflict is not None
    assert "qwen3.6-35b-mtp-4bit" in conflict
    assert "qwen3.6-35b-4bit" in conflict
    assert "draft checkpoint" in conflict


def test_draft_only_gate_also_covers_the_resolved_hf_path():
    from rapid_mlx.model_aliases import draft_only_conflict

    assert draft_only_conflict("mlx-community/Qwen3.6-27B-MTP-4bit") is not None


def test_embedded_mtp_checkpoints_stay_serable():
    """qwen3.8 embeds its own MTP head (hf_path == its mtp_draft_model);
    the gate must not touch it, or every qwen3.8 serve would break."""
    from rapid_mlx.model_aliases import draft_only_conflict

    assert draft_only_conflict("qwen3.8-27b-4bit") is None


def test_draft_gate_ignores_unknown_refs_and_ordinary_aliases():
    from rapid_mlx.model_aliases import draft_only_conflict

    assert draft_only_conflict("qwen3.5-4b-4bit") is None
    assert draft_only_conflict("owner/never-heard-of-it") is None
    assert draft_only_conflict(None) is None
    assert draft_only_conflict("") is None


def test_draft_gate_fails_open_when_the_catalog_cannot_be_read(monkeypatch):
    import rapid_mlx.model_aliases as model_aliases

    def broken_load():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(model_aliases, "_load", broken_load)
    assert model_aliases.draft_only_conflict("qwen3.6-35b-mtp-4bit") is None
    model_aliases.raise_if_draft_only_model("qwen3.6-35b-mtp-4bit")


def test_raise_if_draft_only_model_raises_the_typed_gate_error():
    from rapid_mlx.model_aliases import (
        DraftModelNotServableError,
        raise_if_draft_only_model,
    )

    with pytest.raises(DraftModelNotServableError, match="qwen3.6-35b-4bit"):
        raise_if_draft_only_model("qwen3.6-35b-mtp-4bit")


def test_cli_serve_draft_alias_fails_fast_at_resolve(monkeypatch, capsys):
    """End to end: serving the draft alias exits 1 with the remedy, records
    the resolve-stage failure, and never reaches a download or engine load."""
    import sys

    from rapid_mlx import cli

    emitted = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.6-35b-mtp-4bit"])
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda exc, *, alias_or_path, **kwargs: emitted.append(
            (exc, alias_or_path, kwargs)
        ),
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    stderr = capsys.readouterr().err
    assert "draft checkpoint" in stderr
    assert "qwen3.6-35b-4bit" in stderr
    assert emitted and emitted[0][1] == "qwen3.6-35b-mtp-4bit"
    assert emitted[0][2].get("failure_stage") == "resolve"


def test_cli_bench_draft_alias_fails_fast_with_resolve_telemetry(monkeypatch, capsys):
    import sys

    from rapid_mlx import cli

    emitted = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "bench", "qwen3.6-35b-mtp-4bit"])
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda exc, *, alias_or_path, **kwargs: emitted.append(
            (exc, alias_or_path, kwargs)
        ),
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    assert "draft checkpoint" in capsys.readouterr().err
    # bench telemeters prepare-stage load failures (bench_command's own
    # emit), so its resolve-stage gate refusal is recorded too.
    assert len(emitted) == 1
    assert emitted[0][1] == "qwen3.6-35b-mtp-4bit"
    assert emitted[0][2].get("failure_stage") == "resolve"


@pytest.mark.parametrize("command", ["chat", "run"])
def test_cli_local_chat_and_run_refuse_draft_before_dispatch(
    command, monkeypatch, capsys
):
    """The REPL prefetches before its child ``serve`` starts, so the parent
    entry point must reject a draft before chat_command can download it."""
    import sys

    from rapid_mlx import cli

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", command, "qwen3.6-35b-mtp-4bit"])
    monkeypatch.setattr(
        cli,
        "_ensure_model_downloaded",
        lambda *_a, **_kw: pytest.fail("draft checkpoint must not be downloaded"),
    )
    monkeypatch.setattr(
        cli,
        "chat_command",
        lambda _args: pytest.fail("draft checkpoint must not reach chat dispatch"),
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    assert "draft checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command,attachment",
    [
        ("chat", ["--base-url", "http://127.0.0.1:8123"]),
        ("run", ["--port", "8123"]),
    ],
)
def test_cli_attached_chat_and_run_do_not_apply_local_draft_gate(
    command, attachment, monkeypatch
):
    import sys

    from rapid_mlx import cli

    reached = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["rapid-mlx", command, "qwen3.6-35b-mtp-4bit", *attachment],
    )
    monkeypatch.setattr(cli, "chat_command", lambda args: reached.append(args.model))

    cli.main()

    assert reached == ["mlx-community/Qwen3.6-35B-A3B-MTP-4bit"]


def test_cli_pull_keeps_draft_sidecar_warmable(monkeypatch):
    """pull stays storage-only: pre-warming the sidecar must keep working."""
    import sys

    from rapid_mlx import cli

    pulled = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "pull", "qwen3.6-35b-mtp-4bit"])
    monkeypatch.setattr(
        "rapid_mlx.cli.pull_command",
        lambda args: pulled.append(args.model),
    )

    cli.main()

    assert pulled == ["mlx-community/Qwen3.6-35B-A3B-MTP-4bit"]


def test_draft_gate_ignores_profile_without_hf_path(monkeypatch):
    import types

    import rapid_mlx.model_aliases as model_aliases

    monkeypatch.setattr(
        model_aliases,
        "_load",
        lambda: {"weird": types.SimpleNamespace(hf_path=None, mtp_draft_model=None)},
    )
    assert model_aliases.draft_only_conflict("weird") is None


def test_draft_gate_skips_attached_remote_bench(monkeypatch, capsys):
    """bench --base-url targets an existing server: the named model lives
    remotely and must not be gated against the local catalog."""
    import sys

    from rapid_mlx import cli

    emitted = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rapid-mlx",
            "bench",
            "qwen3.6-35b-mtp-4bit",
            "--base-url",
            "http://127.0.0.1:8123",
        ],
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda exc, *, alias_or_path, **kwargs: emitted.append(kwargs),
    )
    reached = []
    monkeypatch.setattr(cli, "bench_command", lambda args: reached.append(args.model))

    cli.main()

    assert reached == ["mlx-community/Qwen3.6-35B-A3B-MTP-4bit"]
    assert "draft checkpoint" not in capsys.readouterr().err
    assert emitted == []


@pytest.mark.parametrize(
    "target", ["qwen3.6-35b-mtp-4bit", "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"]
)
def test_cli_serve_user_alias_to_a_draft_is_gated_on_the_resolved_path(
    monkeypatch, capsys, tmp_path, target
):
    """A user alias reaches the draft only through its resolved HF path; the
    gate must check that too, or the download + mid-load crash come back."""
    import json
    import sys

    from rapid_mlx import cli

    alias_file = tmp_path / "user-aliases.json"
    alias_file.write_text(
        json.dumps({"version": 1, "aliases": {"my-draft": target}}) + "\n"
    )
    monkeypatch.setenv("RAPID_MLX_USER_ALIASES_FILE", str(alias_file))
    emitted = []
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "my-draft"])
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda exc, *, alias_or_path, **kwargs: emitted.append((alias_or_path, kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "serve_command",
        lambda args: pytest.fail("the draft gate must refuse before serve"),
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    stderr = capsys.readouterr().err
    assert "draft checkpoint" in stderr
    assert "qwen3.6-35b-4bit" in stderr
    assert emitted == [("my-draft", {"failure_stage": "resolve"})]


def test_draft_gate_never_refuses_an_existing_local_path(monkeypatch, tmp_path):
    """resolve_model gives an existing local path precedence over every
    catalog spelling; a local model that shares a draft's name is served."""
    from rapid_mlx.model_aliases import draft_only_conflict

    monkeypatch.chdir(tmp_path)
    (tmp_path / "mlx-community" / "Qwen3.6-27B-MTP-4bit").mkdir(parents=True)
    (tmp_path / "qwen3.6-35b-mtp-4bit").mkdir()
    (tmp_path / "qwen3.6-35b-mtp-4bit" / "config.json").write_text("{}")

    assert draft_only_conflict("mlx-community/Qwen3.6-27B-MTP-4bit") is None
    assert draft_only_conflict("qwen3.6-35b-mtp-4bit") is None
    # A draft with no local twin is still refused.
    assert draft_only_conflict("mlx-community/Qwen3.6-35B-A3B-MTP-4bit") is not None


def test_cli_serve_external_root_model_named_like_draft_uses_local_path(
    monkeypatch, tmp_path
):
    """A complete external-root model wins during resolution even when its
    displayed name matches a catalog draft alias."""
    import sys

    from rapid_mlx import cli

    root = tmp_path / "models"
    model = root / "qwen3.6-35b-mtp-4bit"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("RAPID_MLX_EXTRA_MODEL_ROOTS", str(root))
    monkeypatch.setattr(
        "rapid_mlx.model_aliases._managed_hub_model_is_runnable", lambda _name: False
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.6-35b-mtp-4bit"])
    reached = []
    monkeypatch.setattr(
        cli,
        "serve_command",
        lambda args: reached.append(
            (args.model, getattr(args, "_original_alias", None))
        ),
    )

    cli.main()

    assert reached == [
        (os.path.realpath(model), "qwen3.6-35b-mtp-4bit"),
    ]


def test_cli_serve_draft_gate_terminates_server_start_at_resolve(monkeypatch):
    """The gate's SystemExit is caught by main()'s start-failure guard, which
    ends server_start_state at the current boundary — still ``resolve`` here
    (``preflight`` is only selected right before serve_command)."""
    import sys

    from rapid_mlx import cli
    from rapid_mlx.telemetry import server_start

    stages = []
    monkeypatch.setattr(server_start, "failed", lambda stage, **_: stages.append(stage))
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", "qwen3.6-35b-mtp-4bit"])

    with pytest.raises(SystemExit):
        cli.main()

    assert stages == ["resolve"]
