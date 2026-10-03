# SPDX-License-Identifier: Apache-2.0
"""A stray folder named like a catalog alias must not shadow the alias (#4097).

Contract: an explicit path spelling always means the filesystem, and so does
a bare token whose directory holds a model. A bare catalog-alias token whose
same-named directory is NOT a model (a bench output folder, an empty dir)
resolves to the catalog alias, with a one-line notice from the CLI.
"""

from __future__ import annotations

import os
import sys

import pytest

from rapid_mlx import model_aliases
from rapid_mlx.model_aliases import (
    draft_only_conflict,
    local_dir_shadows_alias,
    resolve_model,
)

ALIAS = "lfm2.5-2.6b-4bit"


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _hf_path(alias: str) -> str:
    return model_aliases._load()[alias].hf_path


def test_bench_output_folder_named_like_alias_resolves_to_the_alias(cwd):
    out = cwd / ALIAS
    out.mkdir()
    (out / "results.json").write_text("{}")

    assert local_dir_shadows_alias(ALIAS) is True
    assert resolve_model(ALIAS) == _hf_path(ALIAS)


def test_empty_folder_named_like_alias_resolves_to_the_alias(cwd):
    (cwd / ALIAS).mkdir()

    assert resolve_model(ALIAS) == _hf_path(ALIAS)


@pytest.mark.parametrize(
    "marker",
    [
        "config.json",
        "model_index.json",
        "split_model.json",
        "params.json",
        "model.safetensors",
        "model-q4.gguf",
        "weights.npz",
    ],
)
def test_local_model_folder_named_like_alias_keeps_path_precedence(cwd, marker):
    """Golden: a real local checkpoint is served in place, as before."""
    (cwd / ALIAS).mkdir()
    (cwd / ALIAS / marker).write_bytes(b"x")

    assert local_dir_shadows_alias(ALIAS) is False
    assert resolve_model(ALIAS) == ALIAS


@pytest.mark.parametrize("spelling", [f"./{ALIAS}", f"{ALIAS}/"])
def test_explicit_relative_spelling_selects_the_local_folder(cwd, spelling):
    """Golden: ``./alias`` (or ``alias/``) always means the filesystem."""
    (cwd / ALIAS).mkdir()

    assert local_dir_shadows_alias(spelling) is False
    assert resolve_model(spelling) == spelling


def test_absolute_path_selects_the_local_folder(cwd):
    target = cwd / ALIAS
    target.mkdir()

    assert resolve_model(str(target)) == str(target)


@pytest.mark.parametrize(
    "spelling", ["~/models/x", "~x", ".hidden", "../x", "/abs/x", "org/repo"]
)
def test_path_spellings_are_explicit(spelling):
    assert model_aliases._is_explicit_path_spelling(spelling) is True
    assert model_aliases._is_explicit_path_spelling(ALIAS) is False


def test_local_file_named_like_alias_keeps_path_precedence(cwd):
    (cwd / ALIAS).write_bytes(b"weights")

    assert resolve_model(ALIAS) == ALIAS


def test_non_alias_folder_is_untouched(cwd):
    (cwd / "my-model").mkdir()

    assert local_dir_shadows_alias("my-model") is False
    assert resolve_model("my-model") == "my-model"


def test_unreadable_folder_is_not_a_model(cwd, monkeypatch):
    (cwd / ALIAS).mkdir()
    real_listdir = os.listdir

    def listdir(path="."):
        if os.fspath(path) == ALIAS:
            raise PermissionError("denied")
        return real_listdir(path)

    monkeypatch.setattr(model_aliases.os, "listdir", listdir)
    assert local_dir_shadows_alias(ALIAS) is True


def test_registry_failure_keeps_historical_precedence(cwd, monkeypatch):
    (cwd / ALIAS).mkdir()

    def broken():
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(model_aliases, "_load", broken)
    assert local_dir_shadows_alias(ALIAS) is False


@pytest.mark.parametrize("value", [None, "", 3])
def test_non_string_or_empty_is_never_a_shadow(value):
    assert local_dir_shadows_alias(value) is False


def test_draft_gate_still_refuses_a_draft_alias_shadowed_by_a_stray_folder(cwd):
    (cwd / "qwen3.6-35b-mtp-4bit").mkdir()

    assert draft_only_conflict("qwen3.6-35b-mtp-4bit") is not None


def test_cli_serve_uses_the_alias_and_says_why(cwd, monkeypatch, capsys):
    from rapid_mlx import cli

    (cwd / ALIAS).mkdir()
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", ALIAS])
    reached = []
    monkeypatch.setattr(
        cli,
        "serve_command",
        lambda args: reached.append(
            (args.model, getattr(args, "_original_alias", None))
        ),
    )
    monkeypatch.setattr(
        "rapid_mlx.byom.preflight.run_cli_preflight", lambda *a, **k: None
    )

    cli.main()

    assert reached == [(_hf_path(ALIAS), ALIAS)]
    out = capsys.readouterr().out
    assert f"./{ALIAS} is not a model folder; using the catalog alias" in out
    assert f"Alias: {ALIAS} → {_hf_path(ALIAS)}" in out


def test_cli_serve_explicit_alias_without_local_folder_prints_no_notice(
    cwd, monkeypatch, capsys
):
    from rapid_mlx import cli

    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "serve", ALIAS])
    monkeypatch.setattr(cli, "serve_command", lambda args: None)
    monkeypatch.setattr(
        "rapid_mlx.byom.preflight.run_cli_preflight", lambda *a, **k: None
    )

    cli.main()

    assert "not a model folder" not in capsys.readouterr().out
