"""Cached-first CLI defaults: real tiny snapshots, no model loads or downloads."""

import importlib
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rapid_mlx import (
    cli,
    first_run,
    front_door,
)
from rapid_mlx import (
    local_models as lm,
)
from rapid_mlx import (
    model_aliases as ma,
)
from rapid_mlx.byom import preflight as pf


@pytest.fixture
def cache(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    # Match the cache redirection used by the existing cached-listing tests.
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    monkeypatch.setattr("huggingface_hub.constants.HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(pf, "supported_model_types", lambda: frozenset({"qwen2"}))
    monkeypatch.setattr(pf, "chat_model_types", lambda: frozenset({"qwen2"}))
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 16 << 30)
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 16)
    monkeypatch.delenv("RAPID_MLX_EXTRA_MODEL_ROOTS", raising=False)
    return hub


def write_model(
    directory,
    *,
    template=True,
    arch="Qwen2ForCausalLM",
    model_type="qwen2",
    weights=True,
):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [arch]})
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"} if template else {})
    )
    (directory / "tokenizer.json").write_text("{}")
    if weights:
        (directory / "model.safetensors").write_bytes(b"fake")
    return directory


def snapshot(cache, repo, **kwargs):
    root = cache / ("models--" + repo.replace("/", "--"))
    directory = write_model(root / "snapshots" / ("a" * 40), **kwargs)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("a" * 40)
    return directory


def local(model="org/chat", *, alias=None, fit="yes", mtime=0, external=False):
    return lm.LocalModel(alias or model, model, alias, external, 4, mtime, fit)


def capture_cli(monkeypatch, argv, *, tty=False):
    captured = []
    monkeypatch.setattr(cli, "serve_command", captured.append)
    monkeypatch.setattr(cli, "chat_command", captured.append)
    monkeypatch.setattr(cli, "_start_v2_lifecycle", lambda *a, **kw: None)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.consent_runtime.startup", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "rapid_mlx._version_check.prompt_upgrade_if_available", lambda: False
    )
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", *argv])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    cli.main()
    return captured[0]


def test_empty_and_catalog_cache(cache):
    assert lm.discover_runnable_chat_models() == []
    assert lm.select_default_local_model() is None
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    snapshot(cache, repo)
    found = lm.discover_runnable_chat_models()
    assert len(found) == 1
    assert found[0].model == first_run.FIRST_RUN_MODEL
    assert found[0].cataloged and not found[0].external
    assert found[0].fit == "yes"
    assert lm.select_default_local_model().model == first_run.FIRST_RUN_MODEL
    assert first_run.select_chat_default() == (first_run.FIRST_RUN_MODEL, True)


def test_serve_cached_catalog(cache, monkeypatch, capsys, stub_serve_port_resolution):
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    snapshot(cache, repo)
    args = capture_cli(monkeypatch, ["serve"])
    assert args.model == repo
    assert args._original_alias == first_run.FIRST_RUN_MODEL
    assert args._serve_default_selected and args._telemetry_auto_selected
    err = capsys.readouterr().err
    assert f"starting {first_run.FIRST_RUN_MODEL} (already on this Mac)." in err
    assert "Override: rapid-mlx serve <model>" in err
    assert "has not been catalog-tested" not in err


@pytest.mark.parametrize("external", [False, True])
def test_uncataloged_default_and_store_untouched_by_selection(
    cache, tmp_path, monkeypatch, capsys, stub_serve_port_resolution, external
):
    # Covers discovery, resolution and CLI selection (serve_command is stubbed);
    # loading from an external folder in place is exercised end to end.
    repo = "mlx-community/Uncataloged-chat-4bit"
    root = tmp_path / "external"
    if external:
        directory = write_model(root / repo)
        monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(root),))
    else:
        directory = snapshot(cache, repo)
    before = {
        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in directory.parent.rglob("*")
    }
    chosen = lm.select_default_local_model()
    assert chosen.model == repo and not chosen.cataloged
    assert chosen.external == external
    if external:
        assert ma.resolve_model(repo) == str(directory)
    args = capture_cli(monkeypatch, ["serve"])
    assert args.model == (str(directory) if external else repo)
    err = capsys.readouterr().err
    assert err.count("has not been catalog-tested here.") == 1
    assert repo in err
    after = {
        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in directory.parent.rglob("*")
    }
    assert after == before


@pytest.mark.parametrize(
    "repo,options",
    [
        ("org/embed-chat", {}),
        ("org/reranker", {}),
        ("org/whisper", {}),
        ("org/tts-chat", {}),
        ("org/vad", {}),
        ("org/tokenizer", {}),
        ("org/asr", {}),
        ("org/stt", {}),
        ("org/model-draft", {}),
        ("org/chat", {"template": False}),
        ("org/chat", {"arch": "BertModel"}),
        ("org/chat", {"model_type": "unsupported"}),
        ("org/chat", {"weights": False}),
    ],
)
def test_nonchat_and_incomplete_never_offered(cache, repo, options):
    snapshot(cache, repo, **options)
    assert lm.discover_runnable_chat_models() == []
    assert lm.select_default_local_model() is None


def test_mtp_primary_is_allowed(cache):
    snapshot(cache, "org/chat-mtp")
    assert lm.select_default_local_model().model == "org/chat-mtp"


def test_catalog_nonchat_and_hub_precedence(cache, tmp_path, monkeypatch):
    profiles = ma.list_profiles()
    nonchat = next(p for p in profiles.values() if p.modality != "text")
    snapshot(cache, nonchat.hf_path)
    repo = "org/chat"
    snapshot(cache, repo)
    root = tmp_path / "external"
    write_model(root / repo)
    monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(root),))
    models = lm.discover_runnable_chat_models()
    assert [m.repo for m in models] == [repo]
    assert not models[0].external


def test_fit_and_unknown_ram(cache, monkeypatch):
    snapshot(cache, "org/chat")
    assert lm.discover_runnable_chat_models(ram_bytes=4) == []
    assert lm.discover_runnable_chat_models(ram_bytes=1 << 30)[0].fit == "tight"
    assert lm.discover_runnable_chat_models(ram_bytes=0)[0].fit == "unknown"
    monkeypatch.setattr("rapid_mlx.recommendations.physical_ram_gb", lambda: 0)
    assert lm.select_default_local_model().fit == "unknown"


def test_policy_order(monkeypatch):
    import rapid_mlx.recommendations as rec

    rows = [
        local(alias="baseline", fit="tight"),
        local(alias="candidate"),
        local(alias="aaa"),
        local(mtime=999),
    ]
    monkeypatch.setattr(lm, "discover_runnable_chat_models", lambda _ram: rows)
    monkeypatch.setattr(rec, "starter_baseline_for_ram", lambda ram: "baseline")
    monkeypatch.setattr(
        rec, "starter_model_candidates", lambda ram: ("candidate", "aaa")
    )
    assert lm.select_default_local_model(16).alias == "baseline"
    rows.pop(0)
    assert lm.select_default_local_model(16).alias == "candidate"
    monkeypatch.setattr(rec, "starter_model_candidates", lambda ram: ())
    assert lm.select_default_local_model(16).alias == "aaa"
    rows[1] = replace(rows[1], fit="unknown")
    assert lm.select_default_local_model(16).alias == "candidate"
    rows[:] = [
        local("hub/older", mtime=2),
        local("ext/newer", mtime=3, external=True),
        local("hub/tight", mtime=999, fit="tight"),
    ]
    assert lm.select_default_local_model(16).model == "ext/newer"
    rows[0] = replace(rows[0], mtime=4)
    assert lm.select_default_local_model(16).model == "hub/older"
    rows[:] = [
        replace(local("org/transformers", mtime=9), mlx_native=False),
        local("org/older-mlx", mtime=1),
    ]
    assert lm.select_default_local_model(16).model == "org/older-mlx"
    rows[:] = []
    assert lm.select_default_local_model(16) is None


@pytest.mark.parametrize(
    ("repo", "quantized", "native"),
    [
        ("org/Plain-chat", False, False),
        ("org/Plain-chat-MLX", False, True),
        ("org/Quantized-chat", True, True),
    ],
)
def test_uncataloged_mlx_native_marker(cache, repo, quantized, native):
    directory = snapshot(cache, repo)
    if quantized:
        config = json.loads((directory / "config.json").read_text())
        config["quantization"] = {"bits": 4, "group_size": 64}
        (directory / "config.json").write_text(json.dumps(config))
    (found,) = lm.discover_runnable_chat_models(16 << 30)
    assert found.repo == repo
    assert found.mlx_native is native


def test_fail_silent(cache, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("broken cache")

    monkeypatch.setattr(cli, "_scan_hf_cache_models", fail)
    assert lm.discover_runnable_chat_models() == []
    assert lm.select_default_local_model() is None
    monkeypatch.setattr(lm, "discover_runnable_chat_models", fail)
    assert lm.select_default_local_model(16) is None


def test_broken_row_does_not_hide_other_models(cache, monkeypatch):
    snapshot(cache, "org/broken")
    snapshot(cache, "org/good")
    original = pf.inspect_local

    def inspect(path):
        if "models--org--broken" in path:
            raise OSError("unreadable")
        return original(path)

    monkeypatch.setattr(pf, "inspect_local", inspect)
    assert [m.model for m in lm.discover_runnable_chat_models()] == ["org/good"]


@pytest.mark.parametrize("command", ["chat", "run"])
@pytest.mark.parametrize("cataloged", [False, True])
def test_chat_cached_choice(cache, monkeypatch, capsys, command, cataloged):
    alias = "qwen3.5-9b-4bit" if cataloged else None
    repo = ma.list_profiles()[alias].hf_path if alias else "org/chat"
    snapshot(cache, repo)
    assert first_run.select_chat_default() == (alias or repo, True)
    args = capture_cli(monkeypatch, [command], tty=True)
    assert args.model == repo
    out = capsys.readouterr().out
    assert f"using {alias or repo} (already on this Mac)." in out
    assert ("has not been catalog-tested" in out) == (not cataloged)


def test_chat_cold_and_attached(cache, monkeypatch, capsys):
    assert first_run.select_chat_default() == (first_run.FIRST_RUN_MODEL, False)
    monkeypatch.setattr(
        lm,
        "select_default_local_model",
        lambda: pytest.fail("attached chat must not discover"),
    )
    args = capture_cli(monkeypatch, ["chat", "--port", "8001"], tty=True)
    assert args.model == ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    assert "No model specified" not in capsys.readouterr().out


@pytest.mark.parametrize("cataloged", [False, True])
def test_front_door_local_fallback(cache, monkeypatch, cataloged):
    alias = "qwen3.5-9b-4bit" if cataloged else None
    repo = ma.list_profiles()[alias].hf_path if alias else "org/chat"
    snapshot(cache, repo)
    monkeypatch.setattr(front_door, "_running_server", lambda: None)
    monkeypatch.setattr(first_run, "last_used_model", lambda: None)
    monkeypatch.setattr(first_run, "preferred_agent", lambda: None)
    monkeypatch.setattr(
        front_door, "_picks_for", lambda *a: [{"alias": "uncached", "role": "smart"}]
    )
    state = front_door.gather_state("test")
    assert state.selected == (alias or repo)
    assert state.local_default == (alias or repo)
    text = front_door.render_screen(state)
    assert (
        "(downloaded)" if cataloged else "(on this Mac · not catalog-tested)"
    ) in text
    state.cached.append(first_run.FIRST_RUN_MODEL)
    assert front_door._default_model(state) == first_run.FIRST_RUN_MODEL


@pytest.mark.parametrize(
    "command,status,auto,expected",
    [
        ("serve", "cached", False, True),
        ("serve", "no_verdict", False, True),
        ("serve", "cached", True, False),
        ("pull", "cached", False, False),
        ("serve", "skipped", False, False),
    ],
)
def test_preflight_uncataloged_note(
    monkeypatch, capsys, command, status, auto, expected
):
    monkeypatch.setattr(pf, "_target_status", lambda args: (status, None))
    args = SimpleNamespace(
        command=command, model="org/chat", _serve_default_selected=auto
    )
    pf.run_cli_preflight(args, spinner_factory=lambda _: nullcontext())
    assert (
        "Note: org/chat is not in the Rapid-MLX catalog" in capsys.readouterr().err
    ) == expected


def test_explicit_external_note(
    cache, tmp_path, monkeypatch, capsys, stub_serve_port_resolution
):
    root = tmp_path / "external"
    directory = write_model(root / "org/chat")
    monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(root),))
    args = capture_cli(monkeypatch, ["serve", "org/chat"])
    assert args.model == str(directory)
    assert (
        capsys.readouterr().err.count("Note: org/chat is not in the Rapid-MLX catalog")
        == 1
    )


def test_external_roots_order_and_errors(tmp_path, monkeypatch):
    configured = tmp_path / "configured"
    default = tmp_path / "default"
    configured.mkdir()
    default.mkdir()
    link = tmp_path / "link"
    link.symlink_to(configured)
    monkeypatch.setenv(
        "RAPID_MLX_EXTRA_MODEL_ROOTS", json.dumps(["", str(configured), str(link)])
    )
    monkeypatch.setattr(
        ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(default), str(tmp_path / "absent"))
    )
    assert ma.external_model_root_paths() == [str(configured), str(default)]
    assert cli._external_model_roots() == [str(configured), str(default)]
    original = ma.os.path.isdir

    def isdir(path):
        if path == str(default):
            raise OSError("unreadable")
        return original(path)

    monkeypatch.setattr(ma.os.path, "isdir", isdir)
    assert ma.external_model_root_paths() == [str(configured)]


@pytest.mark.parametrize(
    "failure", ["metadata", "snapshot", "directory", "inspection", "supported", "draft"]
)
def test_unreadable_or_unverified_chat(cache, monkeypatch, failure):
    from rapid_mlx import model_metadata

    repo = "org/chat"
    snapshot(cache, repo)
    if failure == "metadata":
        monkeypatch.setattr(
            model_metadata, "read_cached_model_metadata", lambda _: None
        )
    elif failure == "snapshot":
        monkeypatch.setattr(
            model_metadata,
            "read_cached_model_metadata",
            lambda _: SimpleNamespace(snapshot_dir=None),
        )
    elif failure == "directory":
        active_cli = importlib.import_module("rapid_mlx.cli")
        active_aliases = importlib.import_module("rapid_mlx.model_aliases")
        monkeypatch.setattr(active_cli, "_scan_hf_cache_models", lambda: [])
        monkeypatch.setattr(
            active_cli, "_scan_external_model_dirs", lambda: [(repo, 4, 0)]
        )
        monkeypatch.setattr(
            active_aliases, "_resolve_external_model_path", lambda _: None
        )
    elif failure == "inspection":
        monkeypatch.setattr(pf, "inspect_local", lambda _: None)
    elif failure == "supported":
        monkeypatch.setattr(pf, "supported_model_types", lambda: None)
    else:

        def reject(_):
            raise ma.DraftModelNotServableError("draft only")

        monkeypatch.setattr(ma, "raise_if_draft_only_model", reject)
    assert lm.discover_runnable_chat_models() == []


def test_catalog_oversize_dropped(cache):
    snapshot(cache, ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path)
    assert lm.discover_runnable_chat_models(ram_bytes=4) == []


@pytest.mark.parametrize(
    "architectures",
    [
        None,
        "Qwen2ForCausalLM",
        [],
        [None],
        ["Qwen2LMHeadModel"],
        ["Qwen2ForConditionalGeneration"],
    ],
)
def test_decoder_architecture_required(cache, architectures):
    path = snapshot(cache, "org/chat") / "config.json"
    path.write_text(json.dumps({"model_type": "qwen2", "architectures": architectures}))
    assert bool(lm.discover_runnable_chat_models()) == (
        isinstance(architectures, list)
        and architectures in (["Qwen2LMHeadModel"], ["Qwen2ForConditionalGeneration"])
    )


def test_cached_starter_does_not_scan_again(cache, monkeypatch):
    snapshot(cache, ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path)
    monkeypatch.setattr(
        lm, "select_default_local_model", lambda: pytest.fail("cached starter wins")
    )
    assert first_run.select_chat_default() == (first_run.FIRST_RUN_MODEL, True)


def test_unknown_ram_uses_16gb_policy(cache, monkeypatch):
    seen = []

    def baseline(ram):
        seen.append(ram)
        return "baseline"

    monkeypatch.setattr("rapid_mlx.recommendations.starter_baseline_for_ram", baseline)
    assert lm.select_default_local_model(0) is None
    assert seen == [16]


@pytest.mark.parametrize(
    "flags",
    [
        ["--cua-only"],
        ["--embedding-model", "embeddinggemma-300m-8bit"],
        ["--rerank-model", "org/rerank"],
    ],
)
def test_special_serve_modes_skip_discovery(
    cache, monkeypatch, flags, stub_serve_port_resolution
):
    monkeypatch.setattr(
        lm,
        "select_default_local_model",
        lambda: pytest.fail("special mode must not discover"),
    )
    args = capture_cli(monkeypatch, ["serve", *flags])
    assert not getattr(args, "_serve_default_selected", False)


def test_attached_base_url_skips_discovery(cache, monkeypatch):
    monkeypatch.setattr(
        lm,
        "select_default_local_model",
        lambda: pytest.fail("attached chat must not discover"),
    )
    args = capture_cli(monkeypatch, ["chat", "--base-url", "http://localhost:8001"])
    assert args.model == ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path


def test_explicit_model_and_telemetry_marker(
    cache, monkeypatch, capsys, stub_serve_port_resolution
):
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    snapshot(cache, repo)
    monkeypatch.setenv("RAPID_MLX_AUTO_SELECTED", "1")
    monkeypatch.setattr(
        lm,
        "select_default_local_model",
        lambda: pytest.fail("explicit model must not discover"),
    )
    args = capture_cli(monkeypatch, ["serve", first_run.FIRST_RUN_MODEL])
    assert args.model == repo and args._telemetry_auto_selected
    assert not getattr(args, "_serve_default_selected", False)
    assert "No model specified" not in capsys.readouterr().err


def test_first_catalog_alias_wins(cache, monkeypatch):
    snapshot(cache, "org/chat")
    profile = SimpleNamespace(hf_path="org/chat", modality="text")
    monkeypatch.setattr(
        ma, "list_profiles", lambda: {"first": profile, "second": profile}
    )
    assert [m.alias for m in lm.discover_runnable_chat_models()] == ["first"]


def test_catalog_draft_never_offered(cache, monkeypatch):
    snapshot(cache, ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path)

    def reject(_):
        raise ma.DraftModelNotServableError("draft only")

    monkeypatch.setattr(ma, "raise_if_draft_only_model", reject)
    assert lm.discover_runnable_chat_models() == []


def test_chat_cached_starter_notice(cache, monkeypatch, capsys):
    snapshot(cache, ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path)
    capture_cli(monkeypatch, ["chat"], tty=True)
    out = capsys.readouterr().out
    assert f"using {first_run.FIRST_RUN_MODEL} (already downloaded)." in out
    assert "catalog-tested" not in out


@pytest.mark.parametrize("explicit", [True, False])
def test_catalog_external_serves_without_note(
    cache, tmp_path, monkeypatch, capsys, stub_serve_port_resolution, explicit
):
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    root = tmp_path / "external"
    directory = write_model(root / repo)
    monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(root),))
    args = capture_cli(
        monkeypatch, ["serve", *([first_run.FIRST_RUN_MODEL] if explicit else [])]
    )
    assert args.model == str(directory)
    assert args._original_alias == first_run.FIRST_RUN_MODEL
    assert "catalog-tested" not in capsys.readouterr().err


def test_catalog_checkpoint_without_config_or_tokenizer_is_not_offered(cache):
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    directory = snapshot(cache, repo)
    (directory / "tokenizer.json").unlink()
    # Tokenizer metadata without a vocabulary is still an incomplete download.
    assert lm.discover_runnable_chat_models() == []
    (directory / "tokenizer.json").write_text("{}")
    (directory / "config.json").unlink()
    assert lm.discover_runnable_chat_models() == []


@pytest.mark.parametrize("method", ["awq", "gptq"])
def test_packed_quantization_is_not_offered(cache, method):
    directory = snapshot(cache, "org/chat-packed")
    config = json.loads((directory / "config.json").read_text())
    config["quantization_config"] = {"quant_method": method, "bits": 4}
    (directory / "config.json").write_text(json.dumps(config))
    assert lm.discover_runnable_chat_models() == []


def test_fit_uses_selected_checkpoint_not_whole_repo(cache):
    directory = snapshot(cache, "org/chat")
    # An old revision of the same repo makes the repo directory much larger
    # than the checkpoint that would actually load.
    old = directory.parent / ("b" * 40)
    old.mkdir()
    (old / "model.safetensors").write_bytes(b"x" * (64 << 10))
    (found,) = lm.discover_runnable_chat_models(ram_bytes=1024)
    assert found.size_bytes == 4


@pytest.mark.parametrize(
    ("vocab", "offered"),
    [
        (("tokenizer.model",), True),
        (("qwen.tiktoken",), True),
        (("vocab.json", "merges.txt"), True),
        (("vocab.json",), False),
    ],
)
def test_tokenizer_vocabulary_formats(cache, vocab, offered):
    directory = snapshot(cache, "org/chat")
    (directory / "tokenizer.json").unlink()
    for name in vocab:
        (directory / name).write_text("x")
    expected = ["org/chat"] if offered else []
    assert [m.repo for m in lm.discover_runnable_chat_models()] == expected


def test_external_checkpoint_without_vocabulary_is_not_offered(
    cache, tmp_path, monkeypatch
):
    root = tmp_path / "external"
    (write_model(root / "acme" / "chat") / "tokenizer.json").unlink()
    monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(root),))
    assert lm.discover_runnable_chat_models() == []


@pytest.mark.parametrize(
    "config_update",
    [
        {"quantization_config": {"quant_method": "awq", "bits": 4}},
        {"model_type": "unsupported", "architectures": ["NovelForCausalLM"]},
    ],
)
def test_cataloged_checkpoint_must_be_loadable(cache, config_update):
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    directory = snapshot(cache, repo)
    config = json.loads((directory / "config.json").read_text())
    config.update(config_update)
    (directory / "config.json").write_text(json.dumps(config))
    assert lm.discover_runnable_chat_models() == []


def test_incomplete_cached_starter_does_not_block_other_local_model(cache):
    starter = snapshot(cache, ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path)
    (starter / "tokenizer.json").unlink()
    snapshot(cache, "org/chat")
    assert lm.catalog_model_ready(first_run.FIRST_RUN_MODEL) is False
    assert first_run.select_chat_default() == ("org/chat", True)


def test_catalog_model_ready_edges(cache, monkeypatch):
    assert lm.catalog_model_ready("no-such-alias") is False
    monkeypatch.setattr(
        lm, "_judge_checkpoint", lambda *a, **kw: (_ for _ in ()).throw(OSError())
    )
    assert lm.catalog_model_ready(first_run.FIRST_RUN_MODEL) is False


def test_catalog_checkpoint_without_chat_template_is_not_offered(cache):
    repo = ma.list_profiles()[first_run.FIRST_RUN_MODEL].hf_path
    snapshot(cache, repo, template=False)
    assert lm.discover_runnable_chat_models() == []


def test_valid_external_copy_backs_up_rejected_hub_copy(cache, tmp_path, monkeypatch):
    repo = "org/chat"
    (snapshot(cache, repo) / "tokenizer.json").unlink()
    root = tmp_path / "external"
    directory = write_model(root / repo)
    monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(root),))
    (found,) = lm.discover_runnable_chat_models()
    assert found.external and found.repo == repo
    assert found.model == str(directory)


def test_serve_notes_wait_for_log_file(monkeypatch, capsys, tmp_path):
    from types import SimpleNamespace

    monkeypatch.delenv("RAPID_MLX_LOG_FILE", raising=False)
    plain = SimpleNamespace(log_file=None)
    lm.serve_startup_note(plain, "now")
    assert capsys.readouterr().err == "now\n"
    logged = SimpleNamespace(log_file=str(tmp_path / "serve.log"))
    lm.serve_startup_note(logged, "first")
    lm.serve_startup_note(logged, "second")
    assert capsys.readouterr().err == ""
    lm.flush_serve_startup_notes(logged)
    lm.flush_serve_startup_notes(plain)
    assert capsys.readouterr().err == "first\nsecond\n"


def test_external_duplicate_ids_judge_the_copy_serve_loads(
    cache, tmp_path, monkeypatch
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (write_model(first / "acme" / "chat") / "tokenizer.json").unlink()
    write_model(second / "acme" / "chat")
    monkeypatch.setattr(ma, "DEFAULT_EXTERNAL_MODEL_ROOTS", (str(first), str(second)))
    # serve resolves the id to the first root's copy, so the id is not offered
    # while that copy is unusable.
    assert ma.resolve_model("acme/chat") == str(first / "acme" / "chat")
    assert lm.discover_runnable_chat_models() == []
    write_model(first / "acme" / "chat")
    (found,) = lm.discover_runnable_chat_models()
    assert found.model == "acme/chat"


@pytest.mark.parametrize(
    ("repo", "offered"),
    [
        ("org/embedded-chat", True),
        ("asr-labs/chat", True),
        ("org/Qwen3-Embedding-4B", False),
        ("org/chat-reranker", False),
    ],
)
def test_name_segments_not_substrings(cache, repo, offered):
    snapshot(cache, repo)
    expected = [repo] if offered else []
    assert [m.repo for m in lm.discover_runnable_chat_models()] == expected


def test_speech_model_type_is_not_chat(cache, monkeypatch):
    # A speech checkpoint may have a supported loader and a chat-like head.
    monkeypatch.setattr(
        pf, "supported_model_types", lambda: frozenset({"qwen2", "speechy"})
    )
    snapshot(
        cache, "org/voice", model_type="speechy", arch="SpeechyForConditionalGeneration"
    )
    assert lm.discover_runnable_chat_models() == []
    monkeypatch.setattr(pf, "chat_model_types", lambda: None)
    snapshot(cache, "org/chat")
    assert lm.discover_runnable_chat_models() == []
