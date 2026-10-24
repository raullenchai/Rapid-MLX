# SPDX-License-Identifier: Apache-2.0
"""Cached inventory joins every registry using exact artifact identities."""

from types import SimpleNamespace

import pytest

from rapid_mlx import cli
from rapid_mlx import model_aliases as aliases
from rapid_mlx.audio import registry


@pytest.fixture
def inventory(monkeypatch):
    import rapid_mlx.local_models as local_models

    monkeypatch.setattr(local_models, "discover_runnable_chat_models", lambda: [])
    monkeypatch.setattr(cli, "_scan_external_model_dirs", lambda: [])
    monkeypatch.setattr(cli, "_cached_repository_is_runnable", lambda repo: True)
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda alias: True)

    def scan(*repos):
        monkeypatch.setattr(
            cli, "_scan_hf_cache_models", lambda: [(repo, 1024, 0) for repo in repos]
        )

    return scan


@pytest.mark.parametrize(
    "alias,repo",
    [
        ("whisper", "mlx-community/whisper-large-v3-mlx"),
        ("kokoro", "mlx-community/Kokoro-82M-bf16"),
        ("qwen3-tts-4bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"),
    ],
)
def test_audio_catalog_alias_in_both_cached_views(inventory, capsys, alias, repo):
    inventory(repo)
    row = cli._cached_models_json_payload()["cached"][0]
    assert (row["alias"], row["repo"], row["subfolder"], row["state"]) == (
        alias,
        repo,
        None,
        "ok",
    )
    cli._print_cached_models()
    assert alias in capsys.readouterr().out


@pytest.mark.parametrize("modality", ["image-gen", "video-gen", "embedding"])
def test_other_catalog_sections_remain_mapped(inventory, monkeypatch, modality):
    monkeypatch.setattr(
        aliases,
        "list_profiles",
        lambda: {"media": aliases.AliasProfile(hf_path="org/media", modality=modality)},
    )
    inventory("org/media", "org/media-other", "ORG/media")
    rows = cli._cached_models_json_payload()["cached"]
    assert [(r["alias"], r["state"]) for r in rows] == [
        ("media", "ok"),
        (None, "unmapped"),
        (None, "unmapped"),
    ]


def test_audio_does_not_overwrite_existing_alias(inventory, monkeypatch):
    original = {"shared": aliases.AliasProfile(hf_path="org/text")}
    monkeypatch.setattr(aliases, "list_profiles", lambda: original)
    monkeypatch.setattr(
        registry,
        "list_audio_aliases",
        lambda: [SimpleNamespace(alias="shared", hf_id="org/audio")],
    )
    inventory("org/audio", "org/text")
    rows = cli._cached_models_json_payload()["cached"]
    assert [r["alias"] for r in rows] == [None, "shared"]
    assert list(original) == ["shared"]


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("subfolder", [None, "4bit"])
def test_exact_pinned_artifact(inventory, monkeypatch, capsys, complete, subfolder):
    profiles = {
        "pinned": aliases.AliasProfile(
            hf_path="org/model", revision="a" * 40, subfolder=subfolder
        ),
    }
    monkeypatch.setattr(aliases, "list_profiles", lambda: profiles)
    monkeypatch.setattr(aliases, "resolve_subfolder", lambda name: subfolder)
    monkeypatch.setattr(cli, "_cache_entry_is_runnable", lambda name: complete)
    calls = []

    def size(repo, folder, *, catalog_alias):
        calls.append((repo, folder, catalog_alias))
        return 512

    monkeypatch.setattr(cli, "_cached_subfolder_size", size)
    inventory("org/model")
    row = cli._cached_models_json_payload()["cached"][0]
    assert row["alias"] == ("pinned" if complete else None)
    assert row["subfolder"] == (subfolder if complete else None)
    assert calls == (
        [("org/model", "4bit", "pinned")] if complete and subfolder else []
    )
    cli._print_cached_models()
    assert ("pinned" in capsys.readouterr().out) == complete


@pytest.mark.parametrize(
    "external,runnable,state", [(False, False, "incomplete"), (True, True, "external")]
)
def test_audio_preserves_nonlaunchable_states(
    inventory, monkeypatch, capsys, external, runnable, state
):
    repo = "mlx-community/whisper-large-v3-mlx"
    inventory(*([] if external else [repo]))
    monkeypatch.setattr(cli, "_cached_repository_is_runnable", lambda repo: runnable)
    if external:
        monkeypatch.setattr(cli, "_scan_external_model_dirs", lambda: [(repo, 1024, 0)])
    row = cli._cached_models_json_payload()["cached"][0]
    assert (row["alias"], row["state"]) == (None, state)
    cli._print_cached_models()
    assert f"({state})" in capsys.readouterr().out
