"""#3116: sentence-embedding aliases must not be advertised as chat models.

embeddinggemma ships in ``aliases.json`` so ``--embedding-model`` can resolve
the short name, but the atomic catalog projected it as
``text_generation``/``chat`` — the only vocabulary it had — so every first-chat
picker (Desktop quickstart, Community Benchmark) offered it as a model that
could answer. These tests pin the new ``embedding`` modality end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_mlx import cli
from vllm_mlx.catalog.legacy import build_legacy_catalog_snapshot
from vllm_mlx.catalog.validation import (
    _TASK_OPERATIONS,
    CatalogValidationError,
    ContractValidator,
)
from vllm_mlx.model_aliases import resolve_profile
from vllm_mlx.routes import models as models_route

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_ALIASES = ("embeddinggemma-300m-6bit", "embeddinggemma-300m-8bit")


@pytest.mark.parametrize("alias", EMBEDDING_ALIASES)
def test_embedding_aliases_carry_the_embedding_modality(alias):
    profile = resolve_profile(alias)
    assert profile is not None
    assert profile.modality == "embedding"
    assert profile.supports_image_input is False


def test_legacy_projection_gives_embedding_aliases_their_own_task():
    snapshot = build_legacy_catalog_snapshot()
    ContractValidator().validate_catalog_snapshot(snapshot)
    aliases = {item["alias"]: item for item in snapshot["aliases"]}
    for alias in EMBEDDING_ALIASES:
        capabilities = aliases[alias]["capabilities"]
        assert capabilities["task_types"] == ["embedding"]
        assert capabilities["operation_modes"] == ["embed"]
        assert capabilities["runtime_adapter"] == "mlx_embeddings"
    # The chat vocabulary is untouched for a real chat alias.
    assert aliases["qwen3.5-4b-4bit"]["capabilities"]["task_types"] == [
        "text_generation"
    ]


def test_embedding_task_only_pairs_with_the_embed_operation():
    assert _TASK_OPERATIONS["embedding"] == {"embed"}
    snapshot = build_legacy_catalog_snapshot()
    alias = next(a for a in snapshot["aliases"] if a["alias"] == EMBEDDING_ALIASES[0])
    alias["capabilities"]["operation_modes"] = ["chat"]
    with pytest.raises(CatalogValidationError):
        ContractValidator().validate_catalog_snapshot(snapshot)


def test_alias_schema_and_proto_admit_the_embedding_vocabulary():
    for path in (
        ROOT / "proto" / "model-catalog" / "v2" / "model-alias.schema.json",
        ROOT / "vllm_mlx" / "catalog" / "schemas" / "model-alias.schema.json",
    ):
        schema = json.loads(path.read_text())
        capabilities = schema["properties"]["capabilities"]["properties"]
        assert "embedding" in capabilities["task_types"]["items"]["enum"]
        assert "embed" in capabilities["operation_modes"]["items"]["enum"]


def test_models_route_tags_embedding_alias_exclusively(monkeypatch):
    monkeypatch.setattr(models_route, "_locked_embedding_id", lambda: None)
    caps = models_route._detect_capabilities(
        "mlx-community/embeddinggemma-300m-6bit", profile_modality="embedding"
    )
    assert caps == ["embedding"]
    # Wire modality stays "text" (F-D01): the tag distinguishes the lane.
    assert (
        models_route._reported_modality(
            "mlx-community/embeddinggemma-300m-6bit", "embedding"
        )
        == "text"
    )


def test_serve_rejects_embedding_alias_with_the_embedding_model_hint(capsys):
    profile = SimpleNamespace(modality="embedding")
    with pytest.raises(SystemExit) as exc_info:
        cli._reject_embedding_alias_serve(profile, "embeddinggemma-300m-6bit")
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "sentence-embedding alias" in err
    assert "--embedding-model embeddinggemma-300m-6bit" in err


def test_serve_command_exits_before_any_model_work_for_embedding_alias(
    monkeypatch, capsys
):
    """codex #3128: pin the ``serve_command`` wiring, not just the helper.

    Anything past the guard is a bug here, so the first post-guard step the
    audio boot-check test also uses (the upgrade prompt) is turned into a
    tripwire.
    """
    from argparse import Namespace

    from vllm_mlx import _version_check

    def _past_the_guard(*_a, **_kw):
        raise AssertionError("serve_command ran past the embedding guard")

    monkeypatch.setattr(_version_check, "prompt_upgrade_if_available", _past_the_guard)
    args = Namespace(
        model="embeddinggemma-300m-6bit",
        embedding_model=None,
        no_mllm=False,
        mllm=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.serve_command(args)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "sentence-embedding alias" in err
    assert "--embedding-model embeddinggemma-300m-6bit" in err


@pytest.mark.parametrize(
    "profile",
    [None, SimpleNamespace(modality="text"), SimpleNamespace(modality="image-gen")],
)
def test_serve_guard_ignores_every_other_alias(profile):
    cli._reject_embedding_alias_serve(profile, "qwen3.5-4b-4bit")
