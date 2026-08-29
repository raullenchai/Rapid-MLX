# SPDX-License-Identifier: Apache-2.0
"""Runtime-asset contracts for offline-capable audio pulls (#2646)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm_mlx import cli
from vllm_mlx.audio import registry, runtime_requirements


def test_kokoro_alias_and_hf_id_declare_voice_assets() -> None:
    expected = (
        registry.AudioRuntimeAsset(
            repo_id="prince-canuma/Kokoro-82M",
            allow_patterns=("voices/*.safetensors",),
        ),
    )

    assert registry.runtime_assets_for("kokoro") == expected
    assert registry.runtime_assets_for("mlx-community/Kokoro-82M-bf16") == expected
    assert registry.runtime_assets_for("whisper-small") == ()
    assert registry.runtime_assets_for("not-an-audio-model") == ()


def test_kokoro_alias_and_hf_id_declare_g2p_requirement() -> None:
    expected = (
        registry.AudioRuntimeRequirement(kind="spacy_pipeline", name="en_core_web_sm"),
    )

    assert registry.runtime_requirements_for("kokoro") == expected
    assert (
        registry.runtime_requirements_for("mlx-community/Kokoro-82M-bf16") == expected
    )
    assert registry.runtime_requirements_for("whisper-small") == ()
    assert registry.runtime_requirements_for("not-an-audio-model") == ()


def test_pull_downloads_primary_then_declared_runtime_assets(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None, str | None, str | None]] = []
    activations = 0
    preparations = []

    def fake_pull_repository(args, *, allow_patterns_override=None):
        calls.append(
            (
                args.model,
                allow_patterns_override,
                getattr(args, "bits", None),
                getattr(args, "format", None),
            )
        )

    def fake_activation() -> None:
        nonlocal activations
        activations += 1

    monkeypatch.setattr(cli, "_pull_repository", fake_pull_repository)
    monkeypatch.setattr(cli, "_emit_pull_activation", fake_activation)
    monkeypatch.setattr(
        runtime_requirements,
        "prepare_runtime_requirement",
        lambda requirement: preparations.append(requirement),
    )
    args = argparse.Namespace(
        model="mlx-community/Kokoro-82M-bf16",
        _original_alias="kokoro",
        bits=None,
        format=None,
    )

    cli.pull_command(args)

    assert calls == [
        ("mlx-community/Kokoro-82M-bf16", None, None, None),
        (
            "prince-canuma/Kokoro-82M",
            ["voices/*.safetensors"],
            None,
            None,
        ),
    ]
    assert activations == 1
    assert preparations == [
        registry.AudioRuntimeRequirement(kind="spacy_pipeline", name="en_core_web_sm")
    ]
    assert args.model == "mlx-community/Kokoro-82M-bf16"
    assert args._original_alias == "kokoro"


def test_pull_without_runtime_assets_keeps_single_repository(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **_kwargs: calls.append(args.model),
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)

    cli.pull_command(argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit"))

    assert calls == ["mlx-community/Qwen3-0.6B-4bit"]


def test_pull_does_not_download_primary_again_when_declared_as_runtime_asset(
    monkeypatch,
) -> None:
    """Malformed future metadata cannot turn one pull into a duplicate pull."""
    primary = "owner/model"
    calls: list[str] = []
    monkeypatch.setattr(
        registry,
        "runtime_assets_for",
        lambda _repo: (
            registry.AudioRuntimeAsset(
                repo_id=primary,
                allow_patterns=("voices/*.safetensors",),
            ),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda args, **_kwargs: calls.append(args.model),
    )
    monkeypatch.setattr(cli, "_emit_pull_activation", lambda: None)

    cli.pull_command(argparse.Namespace(model=primary))

    assert calls == [primary]


def test_runtime_asset_failure_does_not_report_successful_pull(monkeypatch) -> None:
    activations = 0

    def fake_pull_repository(args, **_kwargs):
        if args.model == "prince-canuma/Kokoro-82M":
            raise RuntimeError("asset download failed")

    def fake_activation() -> None:
        nonlocal activations
        activations += 1

    monkeypatch.setattr(cli, "_pull_repository", fake_pull_repository)
    monkeypatch.setattr(cli, "_emit_pull_activation", fake_activation)

    with pytest.raises(RuntimeError, match="asset download failed"):
        cli.pull_command(argparse.Namespace(model="mlx-community/Kokoro-82M-bf16"))

    assert activations == 0


def test_runtime_requirement_failure_does_not_report_successful_pull(
    monkeypatch, capsys
) -> None:
    activations = 0

    monkeypatch.setattr(cli, "_pull_repository", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runtime_requirements,
        "prepare_runtime_requirement",
        lambda requirement: (_ for _ in ()).throw(
            runtime_requirements.AudioRuntimePreparationError(
                "required pipeline is unavailable"
            )
        ),
    )

    def fake_activation() -> None:
        nonlocal activations
        activations += 1

    monkeypatch.setattr(cli, "_emit_pull_activation", fake_activation)

    with pytest.raises(SystemExit) as excinfo:
        cli.pull_command(argparse.Namespace(model="mlx-community/Kokoro-82M-bf16"))

    assert excinfo.value.code == 1
    assert activations == 0
    output = capsys.readouterr().out
    assert "Could not prepare audio runtime" in output
    assert "rapid-mlx[audio]" in output
    assert "rapid-mlx pull mlx-community/Kokoro-82M-bf16" in output


def test_main_pull_requirement_failure_is_actionable_without_traceback(
    monkeypatch, capsys
) -> None:
    requirement = registry.AudioRuntimeRequirement(
        kind="spacy_pipeline", name="en_core_web_sm"
    )
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    monkeypatch.setenv("RAPID_MLX_AUTO_PULL", "1")
    monkeypatch.setattr(sys, "argv", ["rapid-mlx", "--no-telemetry", "pull", "kokoro"])
    monkeypatch.setattr(cli, "_pull_repository", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry, "runtime_assets_for", lambda _name: ())
    monkeypatch.setattr(
        registry, "runtime_requirements_for", lambda _name: (requirement,)
    )
    monkeypatch.setattr(
        runtime_requirements,
        "prepare_runtime_requirement",
        lambda _requirement: (_ for _ in ()).throw(
            runtime_requirements.AudioRuntimePreparationError(
                "Could not prepare required spaCy pipeline 'en_core_web_sm'"
            )
        ),
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    output = capsys.readouterr().out
    assert "Could not prepare audio runtime for 'kokoro'" in output
    assert "rapid-mlx pull kokoro" in output
    assert "rapid-mlx[audio]" in output
    assert "Traceback" not in output


def test_runtime_asset_uses_normal_filtered_download_pipeline(
    monkeypatch, tmp_path, capsys
) -> None:
    seen: dict[str, object] = {}

    def fake_mirror(repo_id, *, allow_patterns=None, out=None):
        seen["mirror"] = (repo_id, allow_patterns)
        return False

    def fake_snapshot(repo_id, *, allow_patterns=None):
        seen["snapshot"] = (repo_id, allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(cli, "_try_mirror_prefetch", fake_mirror)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    monkeypatch.setattr(cli, "_blob_identifier", lambda _root: ())

    cli._pull_repository(
        argparse.Namespace(
            model="prince-canuma/Kokoro-82M",
            bits=None,
            format=None,
            json=True,
        ),
        allow_patterns_override=["voices/*.safetensors"],
    )

    expected = ("prince-canuma/Kokoro-82M", ["voices/*.safetensors"])
    assert seen["mirror"] == expected
    assert seen["snapshot"] == expected
    assert "runtime assets declared by the audio catalog" in capsys.readouterr().out


@pytest.mark.parametrize(
    "runtime_assets,error",
    [
        ([], "_runtime_assets must be an object"),
        ({"": []}, "family keys must be non-empty strings"),
        ({"kokoro": {}}, "_runtime_assets.kokoro must be an array"),
        ({"kokoro": ["bad"]}, "_runtime_assets.kokoro[0] must be an object"),
        (
            {"kokoro": [{"repo_id": "bad", "allow_patterns": ["*"]}]},
            "repo_id must be a HuggingFace namespace/name",
        ),
        (
            {
                "kokoro": [
                    {"repo_id": "owner/assets", "allow_patterns": ["*"]},
                    {"repo_id": "owner/assets", "allow_patterns": ["*"]},
                ]
            },
            "duplicate runtime asset",
        ),
        (
            {"kokoro": [{"repo_id": "owner/assets", "allow_patterns": [""]}]},
            "allow_patterns must be an array of non-empty strings",
        ),
        (
            {"unknown": [{"repo_id": "owner/assets", "allow_patterns": ["*"]}]},
            "runtime assets declared for unknown family",
        ),
    ],
)
def test_runtime_asset_registry_rejects_malformed_metadata(
    monkeypatch, tmp_path, runtime_assets, error
) -> None:
    registry_file = tmp_path / "aliases.json"
    registry_file.write_text(
        json.dumps(
            {
                "_runtime_assets": runtime_assets,
                "kokoro": {
                    "type": "tts",
                    "hf_id": "owner/kokoro",
                    "family": "kokoro",
                },
            }
        )
    )
    monkeypatch.setattr(registry, "_registry_path", lambda: str(registry_file))
    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_HF_ID_INDEX", {})
    monkeypatch.setattr(registry, "_RUNTIME_ASSETS", {})

    with pytest.raises(ValueError, match=re.escape(error)):
        registry._load_registry()


@pytest.mark.parametrize(
    "runtime_requirements_metadata,error",
    [
        ([], "_runtime_requirements must be an object"),
        ({"": []}, "family keys must be non-empty strings"),
        ({"kokoro": {}}, "_runtime_requirements.kokoro must be an array"),
        ({"kokoro": ["bad"]}, "_runtime_requirements.kokoro[0] must be an object"),
        (
            {"kokoro": [{"kind": "shell", "name": "anything"}]},
            "kind must be 'spacy_pipeline'",
        ),
        (
            {"kokoro": [{"kind": "spacy_pipeline", "name": "bad-name"}]},
            "name must be a Python package identifier",
        ),
        (
            {
                "kokoro": [
                    {"kind": "spacy_pipeline", "name": "en_core_web_sm"},
                    {"kind": "spacy_pipeline", "name": "en_core_web_sm"},
                ]
            },
            "duplicate runtime requirement",
        ),
        (
            {"unknown": [{"kind": "spacy_pipeline", "name": "en_core_web_sm"}]},
            "runtime requirements declared for unknown family",
        ),
    ],
)
def test_runtime_requirement_registry_rejects_malformed_metadata(
    monkeypatch, tmp_path, runtime_requirements_metadata, error
) -> None:
    registry_file = tmp_path / "aliases.json"
    registry_file.write_text(
        json.dumps(
            {
                "_runtime_requirements": runtime_requirements_metadata,
                "kokoro": {
                    "type": "tts",
                    "hf_id": "owner/kokoro",
                    "family": "kokoro",
                },
            }
        )
    )
    monkeypatch.setattr(registry, "_registry_path", lambda: str(registry_file))
    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_HF_ID_INDEX", {})
    monkeypatch.setattr(registry, "_RUNTIME_ASSETS", {})
    monkeypatch.setattr(registry, "_RUNTIME_REQUIREMENTS", {})

    with pytest.raises(ValueError, match=re.escape(error)):
        registry._load_registry()


def test_concurrent_first_lookup_publishes_runtime_assets_atomically(
    monkeypatch, tmp_path
) -> None:
    registry_file = tmp_path / "aliases.json"
    registry_file.write_text(
        json.dumps(
            {
                "_runtime_assets": {
                    "kokoro": [
                        {
                            "repo_id": "owner/voices",
                            "allow_patterns": ["voices/*.safetensors"],
                        }
                    ]
                },
                "kokoro": {
                    "type": "tts",
                    "hf_id": "owner/kokoro",
                    "family": "kokoro",
                },
            }
        )
    )
    monkeypatch.setattr(registry, "_registry_path", lambda: str(registry_file))
    monkeypatch.setattr(registry, "_REGISTRY", None)
    monkeypatch.setattr(registry, "_HF_ID_INDEX", {})
    monkeypatch.setattr(registry, "_RUNTIME_ASSETS", {})

    first_is_parsing = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    real_json_load = json.load
    load_calls = 0

    def blocking_json_load(stream):
        nonlocal load_calls
        load_calls += 1
        first_is_parsing.set()
        assert release_first.wait(timeout=2)
        return real_json_load(stream)

    def second_lookup():
        second_started.set()
        return registry.runtime_assets_for("kokoro")

    monkeypatch.setattr(registry.json, "load", blocking_json_load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(registry.runtime_assets_for, "kokoro")
        assert first_is_parsing.wait(timeout=2)
        second = executor.submit(second_lookup)
        assert second_started.wait(timeout=2)
        release_first.set()

        expected = (
            registry.AudioRuntimeAsset(
                repo_id="owner/voices",
                allow_patterns=("voices/*.safetensors",),
            ),
        )
        assert first.result(timeout=2) == expected
        assert second.result(timeout=2) == expected

    assert load_calls == 1
