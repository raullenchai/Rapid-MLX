# SPDX-License-Identifier: Apache-2.0
"""Browser model links use exactly the existing repository-ID CLI paths."""

import sys

import pytest

from rapid_mlx.model_reference import normalize_model_reference

REPO = "mlx-community/Qwen3.5-9B-abliterated-4bit"


@pytest.mark.parametrize(
    "host",
    ["huggingface.co", "www.huggingface.co", "hf.co", "hf-mirror.com"],
)
@pytest.mark.parametrize("scheme", ["https://", "http://", ""])
@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "/",
        "/tree/main",
        "/tree/main/sub/path",
        "/blob/main/config.json",
        "/resolve/main/model.safetensors",
        "?download=true",
        "#files",
        "/tree/main/?download=true#files",
    ],
)
def test_browser_url_forms(host, scheme, suffix):
    notices = []
    assert (
        normalize_model_reference(
            f"{scheme}{host}/{REPO}{suffix}", notice=notices.append
        )
        == REPO
    )
    assert notices == [f"Using {REPO}"]


@pytest.mark.parametrize("wrapper", [" {} ", "'{}'", '"{}"', "<{}>", " '<{}>' "])
def test_copy_paste_wrappers(wrapper):
    assert normalize_model_reference(wrapper.format(f"https://hf.co/{REPO}")) == REPO


@pytest.mark.parametrize("route", ["tree", "blob", "resolve"])
@pytest.mark.parametrize("revision", ["dev", "a" * 40, "refs%2Fpr%2F1"])
def test_non_main_revision_notes_default_branch(route, revision):
    notices = []
    assert (
        normalize_model_reference(
            f"https://hf.co/{REPO}/{route}/{revision}/sub/file", notice=notices.append
        )
        == REPO
    )
    assert len(notices) == 2
    assert "default branch" in notices[1]


@pytest.mark.parametrize(
    "path,kind",
    [
        ("datasets/org/data", "dataset"),
        ("spaces/org/demo", "Space"),
        ("collections/org/models", "collection"),
        ("models?search=qwen", "model listing"),
    ],
)
def test_non_model_links(path, kind):
    with pytest.raises(ValueError, match=f"{kind} link, not a model"):
        normalize_model_reference(f"https://huggingface.co/{path}")


@pytest.mark.parametrize("path", ["", "org", "org/", "org//tree/main"])
def test_incomplete_model_links(path):
    with pytest.raises(ValueError, match="https://huggingface.co/org/name"):
        normalize_model_reference(f"https://huggingface.co/{path}")


@pytest.mark.parametrize(
    "name",
    [
        "qwen3.5-9b-4bit",
        REPO,
        "/tmp/model",
        "./hf.co/org/model",
        "~/models/test",
        "https://example.com/org/name",
        "[local-path",
        "https://huggingface.co.evil/org/name",
        "ftp://hf.co/org/name",
        " org/name ",
        "'qwen3.5-9b-4bit'",
    ],
)
def test_other_references_unchanged(name):
    notices = []
    assert normalize_model_reference(name, notice=notices.append) == name
    assert notices == []


def test_existing_url_shaped_local_path_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = f"hf.co/{REPO}"
    (tmp_path / name).mkdir(parents=True)
    assert normalize_model_reference(name) == name


def test_shared_resolver_accepts_urls():
    from rapid_mlx.model_aliases import resolve_hub_model, resolve_model

    for resolver in (resolve_model, resolve_hub_model):
        assert resolver(f"https://hf.co/{REPO}") == resolver(REPO)


@pytest.mark.parametrize(
    "command", ["serve", "chat", "run", "pull", "rm", "info", "bench"]
)
def test_cli_dispatch_matches_repo_id(command, monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.byom import preflight
    from rapid_mlx.telemetry import server_start

    reached, checked, attempted = [], [], []
    handler = "chat_command" if command == "run" else f"{command}_command"
    monkeypatch.setattr(
        cli,
        handler,
        lambda args: reached.append(
            (args.model, getattr(args, "_original_alias", None))
        ),
    )
    monkeypatch.setattr(
        preflight, "run_cli_preflight", lambda args, **kw: checked.append(args.model)
    )
    monkeypatch.setattr(
        server_start, "attempted", lambda model, **kw: attempted.append(model)
    )
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    for ref in (REPO, f"https://huggingface.co/{REPO}?token=private#files"):
        monkeypatch.setattr(sys, "argv", ["rapid-mlx", command, ref])
        cli.main()
    assert reached[0] == reached[1]
    assert reached[1][0] == REPO
    if command in {"serve", "pull"}:
        assert checked == [REPO, REPO]
    if command == "serve":
        assert attempted == [REPO, REPO]
    output = capsys.readouterr()
    assert output.err.count(f"Using {REPO}") == 1
    assert "token=private" not in output.out + output.err


def test_cli_invalid_link_stops_before_telemetry_and_dispatch(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.telemetry import server_start

    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "serve", "https://hf.co/datasets/org/data"]
    )
    monkeypatch.setattr(
        server_start,
        "attempted",
        lambda *a, **kw: pytest.fail("invalid URL reached telemetry"),
    )
    monkeypatch.setattr(
        cli, "serve_command", lambda args: pytest.fail("invalid URL dispatched")
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "dataset link, not a model" in capsys.readouterr().err


def test_cli_failure_telemetry_receives_repo_id(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.telemetry import model_events

    repo = "mlx-community/Qwen3.6-35B-A3B-MTP-4bit"
    emitted = []
    monkeypatch.setattr(
        sys, "argv", ["rapid-mlx", "serve", f"https://hf.co/{repo}?token=private"]
    )
    monkeypatch.setattr(
        model_events,
        "emit_model_serve_failed",
        lambda exc, *, alias_or_path, **kw: emitted.append(alias_or_path),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert emitted == [repo]
    assert "token=private" not in capsys.readouterr().err


def test_normalized_custom_repo_keeps_existing_telemetry_privacy(monkeypatch):
    from rapid_mlx.telemetry import model_id

    monkeypatch.setattr(model_id, "is_proven_public", lambda ref: False)
    repo = normalize_model_reference("https://hf.co/private-owner/private-model")
    assert model_id.telemetry_model_id(repo) == model_id.CUSTOM
