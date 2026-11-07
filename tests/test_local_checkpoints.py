"""Explicit local checkpoint references: offline, exact identity, no file ownership."""

import json

import pytest

from rapid_mlx import local_models as lm
from rapid_mlx.byom import preflight as pf
from rapid_mlx.local_checkpoints import inspect_local_checkpoints

ORIGINAL_DISCOVER = lm.discover_runnable_chat_models


def checkpoint(path, **config):
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                **config,
            }
        )
    )
    (path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}')
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"metadata-only-test-fixture")
    return path


@pytest.fixture(autouse=True)
def offline_inventory(monkeypatch):
    monkeypatch.setattr(lm, "discover_runnable_chat_models", lambda **kw: [])
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 16 << 30)
    monkeypatch.setattr(pf, "supported_model_types", lambda: frozenset({"qwen2"}))
    monkeypatch.setattr(pf, "chat_model_types", lambda: frozenset({"qwen2"}))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("download")
    )


@pytest.mark.parametrize(
    "name", ["same name", "tts", "embed", "draft", "trailing space "]
)
def test_explicit_directory_name_does_not_override_chat_metadata(tmp_path, name):
    path = checkpoint(tmp_path / name)
    row = inspect_local_checkpoints([str(path)])[0]
    assert row["ready"] is True
    assert row["launch_path"] == str(path.resolve())
    assert row["known_model"] is None
    assert "repo" not in row  # Folder names do not establish a Hub identity.


def test_exact_path_identity_and_internal_symlink(tmp_path):
    first = checkpoint(tmp_path / "a" / "same")
    second = checkpoint(tmp_path / "b" / "same")
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    (first / "tokenizer-alias.json").symlink_to(first / "tokenizer.json")
    rows = inspect_local_checkpoints([str(first), str(second), str(alias)])
    assert all(row["ready"] for row in rows)
    assert rows[0]["path"] == rows[2]["path"] != rows[1]["path"]


def test_symlink_escape_is_rejected_before_metadata_read(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "checkpoint")
    secret = tmp_path / "outside.json"
    secret.write_text('{"private":"not a model"}')
    (path / "config.json").unlink()
    (path / "config.json").symlink_to(secret)
    monkeypatch.setattr(
        "rapid_mlx._download_gate._snapshot_is_complete",
        lambda *a: pytest.fail("escaped read"),
    )
    assert inspect_local_checkpoints([str(path)])[0]["reason"] == "unsafe_symlink"


def test_existing_inventory_matches_exact_canonical_checkpoint_only(
    tmp_path, monkeypatch
):
    path = checkpoint(tmp_path / "managed" / "snapshot")
    other = checkpoint(tmp_path / "other" / "snapshot")
    model = lm.LocalModel(
        "known-alias", "org/model", "known-alias", False, 10, 0, "yes"
    )
    monkeypatch.setattr(lm, "discover_runnable_chat_models", lambda **kw: [model])
    monkeypatch.setattr(lm, "_checkpoint_dir", lambda *a: str(path))
    first, second = inspect_local_checkpoints([str(path), str(other)])
    assert first["known_model"] == "known-alias"
    assert second["known_model"] is None
    assert first["launch_path"] == str(path)


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("missing_weights", "incomplete_checkpoint"),
        ("malformed_config", "unreadable_metadata"),
        ("template", "missing_chat_template"),
        ("tokenizer", "missing_tokenizer"),
        ("packed", "unsupported_quantization"),
        ("embedding", "unsupported_chat_model"),
        ("draft_arch", "unsupported_chat_model"),
        ("unknown_arch", "unsupported_chat_model"),
    ],
)
def test_local_rejection_does_not_poison_mixed_batch(tmp_path, damage, reason):
    bad = checkpoint(tmp_path / "bad")
    good = checkpoint(tmp_path / "good")
    if damage == "missing_weights":
        (bad / "model.safetensors").unlink()
    if damage == "malformed_config":
        (bad / "config.json").write_text("{")
    if damage == "template":
        (bad / "tokenizer_config.json").write_text("{}")
    if damage == "tokenizer":
        (bad / "tokenizer.json").unlink()
    if damage in ("packed", "embedding", "draft_arch", "unknown_arch"):
        config = json.loads((bad / "config.json").read_text())
        if damage == "packed":
            config["quantization_config"] = {"quant_method": "awq"}
        if damage == "embedding":
            config["architectures"] = ["Qwen2ForTextEmbedding"]
        if damage == "draft_arch":
            config["model_type"] = "mtp_drafter"
        if damage == "unknown_arch":
            config["model_type"] = "not_installed"
        (bad / "config.json").write_text(json.dumps(config))
    rows = inspect_local_checkpoints([str(bad), str(good)])
    assert rows[0]["ready"] is False
    assert rows[0]["reason"] == reason
    assert rows[1]["ready"] is True


@pytest.mark.parametrize(
    "probe,reason",
    [
        ("runtime", "runtime_inventory_unavailable"),
        ("memory", "memory_unavailable"),
        ("inventory", "inventory_unavailable"),
    ],
)
def test_environment_failure_never_certifies_unknown_state(
    tmp_path, monkeypatch, probe, reason
):
    path = checkpoint(tmp_path / "checkpoint")
    if probe == "runtime":
        monkeypatch.setattr(pf, "supported_model_types", lambda: None)
    if probe == "memory":
        monkeypatch.setattr(pf, "physical_ram_bytes", lambda: None)
    if probe == "inventory":

        def broken(**kwargs):
            raise RuntimeError("unavailable")

        monkeypatch.setattr(lm, "discover_runnable_chat_models", broken)
    row = inspect_local_checkpoints([str(path)])[0]
    assert row["ready"] is False
    assert row["reason"] == reason


@pytest.mark.parametrize("kind", ["empty", "exception"])
def test_missing_diagnostics_and_inspection_exception_are_local(
    tmp_path, monkeypatch, kind
):
    bad = checkpoint(tmp_path / "bad")
    good = checkpoint(tmp_path / "good")
    original = lm._judge_checkpoint

    def inspect(repo, alias, **kwargs):
        if repo == str(bad):
            if kind == "exception":
                raise TypeError("malformed metadata")
            return None
        return original(repo, alias, **kwargs)

    monkeypatch.setattr(lm, "_judge_checkpoint", inspect)
    rows = inspect_local_checkpoints([str(bad), str(good)])
    assert rows[0]["reason"] == "inspection_failed"
    assert rows[1]["ready"] is True


def test_inspection_never_executes_checkpoint_python(tmp_path):
    marker = tmp_path / "executed"
    path = checkpoint(tmp_path / "checkpoint", auto_map={"AutoConfig": "model.Custom"})
    (path / "model.py").write_text(f"open({str(marker)!r}, 'w').write('executed')")
    assert inspect_local_checkpoints([str(path)])[0]["ready"] is True
    assert not marker.exists()


@pytest.mark.parametrize("name", ["ordinary-chat", "renamed", "tts"])
def test_backbone_typed_dflash_is_not_a_standalone_chat_checkpoint(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(pf, "supported_model_types", lambda: frozenset({"qwen3"}))
    monkeypatch.setattr(pf, "chat_model_types", lambda: frozenset({"qwen3"}))
    base = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    ordinary = checkpoint(tmp_path / "ordinary", **base)
    sidecar = checkpoint(
        tmp_path / name,
        **base,
        dflash_config={
            "mask_token_id": 151669,
            "target_layer_ids": [1, 8, 15],
            "num_target_layers": 32,
        },
    )
    positive, negative = inspect_local_checkpoints([str(ordinary), str(sidecar)])
    assert positive["ready"] is True
    assert negative["ready"] is False
    assert negative["reason"] == "unsupported_chat_model"


def test_strict_inventory_error_is_not_an_empty_inventory(monkeypatch):
    def unavailable():
        raise OSError("inventory unavailable")

    monkeypatch.setattr("rapid_mlx.cli._scan_hf_cache_models", unavailable)
    with pytest.raises(OSError, match="inventory unavailable"):
        ORIGINAL_DISCOVER(strict=True)
    assert ORIGINAL_DISCOVER() == []  # Existing callers retain their fallback.


def test_zero_weights_after_completeness_probe_fail_closed(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "checkpoint")
    (path / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(
        "rapid_mlx._download_gate._snapshot_is_complete", lambda path: True
    )
    assert (
        inspect_local_checkpoints([str(path)])[0]["reason"] == "incomplete_checkpoint"
    )


def test_memory_fit_refusal_is_not_a_download_fallback(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / "checkpoint")
    monkeypatch.setattr(pf, "physical_ram_bytes", lambda: 1)
    row = inspect_local_checkpoints([str(path)])[0]
    assert row["ready"] is False
    assert row["reason"] == "unsupported_or_oversize"


def test_cli_opt_in_and_default_json_contract(monkeypatch, capsys):
    from rapid_mlx import cli
    from rapid_mlx.cli_parser import build_parser

    parser = build_parser()
    monkeypatch.setattr(
        "rapid_mlx.local_checkpoints.inspect_local_checkpoints",
        lambda paths: [
            {"requested_path": p, "ready": False, "reason": "missing_directory"}
            for p in paths
        ],
    )
    cli.models_command(
        parser.parse_args(["models", "--inspect-local", "/missing path", "--json"])
    )
    assert (
        json.loads(capsys.readouterr().out)["local_checkpoints"][0]["reason"]
        == "missing_directory"
    )
    monkeypatch.setattr(
        cli, "_available_models_json_payload", lambda: {"old_shape": True}
    )
    cli.models_command(parser.parse_args(["models", "--json"]))
    assert json.loads(capsys.readouterr().out) == {"old_shape": True}
    for flags in (
        [],
        ["--cached", "--json"],
        ["--json", "--search", "qwen"],
        ["--json", "--modality", "text"],
    ):
        with pytest.raises(SystemExit, match="2"):
            cli.models_command(
                parser.parse_args(["models", "--inspect-local", "/missing", *flags])
            )
    assert "requires --json" in capsys.readouterr().err
    flags = [flag for _ in range(33) for flag in ("--inspect-local", "/missing")]
    with pytest.raises(SystemExit, match="2"):
        cli.models_command(parser.parse_args(["models", "--json", *flags]))


def test_bounds_and_invalid_paths(tmp_path):
    for paths in ([], ["/missing"] * 33):
        with pytest.raises(ValueError):
            inspect_local_checkpoints(paths)
    rows = inspect_local_checkpoints(
        [
            "relative",
            "/bad\npath",
            "/bad\ud800",
            "/" + "x" * 4096,
            str(tmp_path / "missing"),
        ]
    )
    assert [row["reason"] for row in rows] == ["invalid_path"] * 4 + [
        "missing_directory"
    ]
