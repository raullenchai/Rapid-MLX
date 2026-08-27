# SPDX-License-Identifier: Apache-2.0
"""Hermetic coverage for offline model metadata inspection."""

import json
import sys
import types
from pathlib import Path

import pytest

from vllm_mlx import model_metadata as metadata


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_safetensors_header(path: Path, tensor_names) -> None:
    header = json.dumps({name: {} for name in tensor_names}).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header)


def test_readers_reject_missing_malformed_non_object_and_oversized_files(tmp_path):
    missing = tmp_path / "missing.json"
    assert metadata._read_json(None) is None
    assert metadata._read_json(missing) is None
    assert metadata._read_text(None) is None
    assert metadata._read_text(missing) is None

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert metadata._read_json(malformed) is None

    list_json = tmp_path / "list.json"
    _write_json(list_json, ["not", "an", "object"])
    assert metadata._read_json(list_json) is None

    huge = tmp_path / "huge.txt"
    huge.write_text("x" * (metadata.MAX_METADATA_FILE_BYTES + 1), encoding="utf-8")
    assert metadata._read_json(huge) is None
    assert metadata._read_text(huge) is None

    invalid_utf8 = tmp_path / "invalid-utf8.jinja"
    invalid_utf8.write_bytes(b"\xff")
    assert metadata._read_text(invalid_utf8) is None


def test_probe_oserror_is_treated_as_absent_not_propagated(tmp_path, monkeypatch):
    """codex #5: the ``is_file()`` / ``is_dir()`` PROBES are inside the OSError
    handler, so a stale/unmounted network share or permission-denied directory
    (where the probe itself raises ``OSError``) yields the documented file-absent
    result instead of crashing the routing path at model-load.
    """
    from pathlib import Path

    def _boom(self, *args, **kwargs):
        raise OSError("stale mount / permission denied")

    # A real path so the readers reach the probe (the ``path is None`` short
    # circuit runs first and would otherwise mask the probe).
    config = tmp_path / "config.json"
    _write_json(config, {"model_type": "qwen3"})
    template = tmp_path / "chat_template.jinja"
    template.write_text("hi", encoding="utf-8")

    monkeypatch.setattr(Path, "is_file", _boom)
    # ``is_file()`` raising must be swallowed → absent sentinel, not propagate.
    assert metadata._read_json(config) is None
    assert metadata._read_text(template) is None

    # ``is_dir()`` raising in the named-template scan and the local-dir entry
    # point must likewise be swallowed rather than escape.
    monkeypatch.setattr(Path, "is_dir", _boom)
    assert metadata._read_named_chat_templates(tmp_path) == {}
    assert metadata.read_local_model_metadata(str(tmp_path)) is None


def test_local_metadata_prefers_standalone_template_then_tokenizer_fallback(tmp_path):
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    _write_json(standalone / "config.json", {"model_type": "qwen3"})
    _write_json(standalone / "tokenizer_config.json", {"chat_template": "tokenizer"})
    (standalone / "chat_template.jinja").write_text("standalone", encoding="utf-8")

    result = metadata.read_local_model_metadata(str(standalone))

    assert result == metadata.ModelMetadata(
        config={"model_type": "qwen3"},
        chat_template="standalone",
        snapshot_dir=standalone,
        is_local=True,
    )

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    _write_json(fallback / "tokenizer_config.json", {"chat_template": "tokenizer"})
    fallback_result = metadata.read_local_model_metadata(str(fallback))

    assert fallback_result == metadata.ModelMetadata(
        config=None,
        chat_template="tokenizer",
        snapshot_dir=fallback,
        is_local=True,
    )
    assert metadata._chat_template(None) is None
    assert metadata.read_local_model_metadata(object()) is None


def test_named_template_directory_selects_tool_use_over_default(tmp_path):
    """A snapshot with named templates under ``additional_chat_templates/``
    (Transformers' modern layout) selects ``tool_use`` over ``default``.

    Mirrors ``PreTrainedTokenizerBase._from_pretrained``: ``chat_template.jinja``
    supplies the ``default`` template, and each ``<name>.jinja`` under
    ``additional_chat_templates/`` contributes a named template; ``tool_use``
    then wins over ``default`` for tool-carrying requests.
    """
    snap = tmp_path / "named-templates"
    snap.mkdir()
    _write_json(snap / "config.json", {"model_type": "qwen3"})
    # ``default`` comes from chat_template.jinja; the tokenizer_config entry must
    # NOT win (standalone files take priority in Transformers).
    _write_json(snap / "tokenizer_config.json", {"chat_template": "TOKENIZER_CONF"})
    (snap / "chat_template.jinja").write_text("DEFAULT_BODY", encoding="utf-8")
    template_dir = snap / "additional_chat_templates"
    template_dir.mkdir()
    (template_dir / "tool_use.jinja").write_text("TOOL_USE_BODY", encoding="utf-8")
    (template_dir / "rag.jinja").write_text("RAG_BODY", encoding="utf-8")

    result = metadata.read_local_model_metadata(str(snap))

    assert result is not None
    assert result.chat_template == "TOOL_USE_BODY"


def test_named_template_directory_default_only_flattens_to_string(tmp_path):
    """A lone ``default.jinja`` under the named directory (no ``tool_use``)
    flattens to that single template, matching Transformers' collapse of a
    lone ``default`` to a bare string."""
    snap = tmp_path / "default-only"
    snap.mkdir()
    template_dir = snap / "additional_chat_templates"
    template_dir.mkdir()
    (template_dir / "default.jinja").write_text("ONLY_DEFAULT", encoding="utf-8")

    assert metadata._chat_template(snap) == "ONLY_DEFAULT"


def test_named_tokenizer_templates_prefer_tool_use_then_default():
    assert (
        metadata._select_chat_template(
            {"chat_template": {"default": "default", "tool_use": "tool-use"}}
        )
        == "tool-use"
    )
    assert (
        metadata._select_chat_template({"chat_template": {"default": "default"}})
        == "default"
    )
    assert (
        metadata._select_chat_template({"chat_template": {"a": "one", "b": "two"}})
        is None
    )
    assert (
        metadata._select_chat_template(
            {
                "chat_template": [
                    {"name": "default", "template": "default"},
                    {"name": "tool_use", "template": "tool-use"},
                    {"name": 1, "template": "ignored"},
                ]
            }
        )
        == "tool-use"
    )


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("publisher/model", True),
        ("publisher/nested/model", True),
        ("plain-name", False),
        ("/absolute/model", False),
        ("./relative/model", False),
        ("../parent/model", False),
        ("~/cache/model", False),
    ],
)
def test_hub_repo_id_validation_rejects_path_lookalikes(model_name, expected):
    assert metadata._looks_like_hub_repo_id(model_name) is expected


def test_cached_file_handles_cache_hit_missing_and_lookup_error(monkeypatch, tmp_path):
    hit = tmp_path / "config.json"
    _write_json(hit, {"ok": True})
    no_exist = object()
    responses = {
        "config.json": str(hit),
        "missing.json": None,
        "no-exist.json": no_exist,
    }

    def lookup(repo_id, filename):
        if filename == "explode.json":
            raise RuntimeError("cache unavailable")
        return responses[filename]

    hub = types.ModuleType("huggingface_hub")
    hub._CACHED_NO_EXIST = no_exist
    hub.try_to_load_from_cache = lookup
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert metadata._cached_file("publisher/model", "config.json") == hit
    assert metadata._cached_file("publisher/model", "missing.json") is None
    assert metadata._cached_file("publisher/model", "no-exist.json") is None
    assert metadata._cached_file("publisher/model", "explode.json") is None
    assert metadata._cached_file("not-a-repo", "config.json") is None


def test_cached_file_handles_missing_huggingface_hub_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert metadata._cached_file("publisher/model", "config.json") is None


def test_cached_metadata_reads_standalone_template_and_tokenizer_fallback(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    standalone = snapshot / "chat_template.jinja"
    tokenizer = snapshot / "tokenizer_config.json"
    _write_json(config, {"model_type": "qwen3_5"})
    standalone.write_text("standalone", encoding="utf-8")
    _write_json(tokenizer, {"chat_template": "tokenizer"})
    paths = {
        "config.json": config,
        "chat_template.jinja": standalone,
        "tokenizer_config.json": tokenizer,
    }
    monkeypatch.setattr(
        metadata, "_cached_file", lambda name, filename: paths[filename]
    )

    result = metadata.read_cached_model_metadata("publisher/model")

    assert result == metadata.ModelMetadata(
        config={"model_type": "qwen3_5"},
        chat_template="standalone",
        snapshot_dir=snapshot,
    )

    standalone.unlink()
    paths["chat_template.jinja"] = None
    result = metadata.read_cached_model_metadata("publisher/model")
    assert result is not None
    assert result.chat_template == "tokenizer"


def test_cached_metadata_reads_one_snapshot_when_cache_refs_change(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "snapshot"
    stale = tmp_path / "stale"
    snapshot.mkdir()
    stale.mkdir()
    _write_json(snapshot / "config.json", {"model_type": "qwen3"})
    (snapshot / "chat_template.jinja").write_text("current", encoding="utf-8")
    (stale / "chat_template.jinja").write_text("stale", encoding="utf-8")
    calls = []

    def cached_file(name, filename):
        calls.append(filename)
        if filename == "config.json":
            return snapshot / filename
        return stale / filename

    monkeypatch.setattr(metadata, "_cached_file", cached_file)

    result = metadata.read_cached_model_metadata("publisher/model")

    assert result is not None
    assert result.snapshot_dir == snapshot
    assert result.chat_template == "current"
    assert calls == ["config.json"]


def test_cached_metadata_returns_none_without_any_cached_metadata(monkeypatch):
    monkeypatch.setattr(metadata, "_cached_file", lambda name, filename: None)

    assert metadata.read_cached_model_metadata("publisher/model") is None


def test_unreferenced_single_complete_snapshot_is_resolved_offline(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "models--publisher--model"
    snapshot = repo_root / "snapshots" / "immutable-revision"
    snapshot.mkdir(parents=True)
    _write_json(snapshot / "config.json", {"model_type": "qwen3_5"})
    (snapshot / "model.safetensors").write_bytes(b"complete")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    result = metadata.read_cached_model_metadata("publisher/model")

    assert result is not None
    assert result.snapshot_dir == snapshot
    assert result.config == {"model_type": "qwen3_5"}


def test_unreferenced_snapshot_fails_closed_when_partial_or_ambiguous(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "models--publisher--model"
    snapshots = repo_root / "snapshots"
    partial = snapshots / "partial"
    partial.mkdir(parents=True)
    _write_json(partial / "config.json", {"model_type": "qwen3_5"})
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert metadata.resolve_unreferenced_cached_snapshot("publisher/model") is None

    (partial / "model.safetensors").write_bytes(b"complete")
    second = snapshots / "second"
    second.mkdir()
    _write_json(second / "config.json", {"model_type": "qwen3_5"})
    (second / "model.safetensors").write_bytes(b"complete")

    assert metadata.resolve_unreferenced_cached_snapshot("publisher/model") is None


def test_unreferenced_snapshot_does_not_override_existing_main_ref(
    monkeypatch, tmp_path
):
    repo_root = tmp_path / "models--publisher--model"
    snapshot = repo_root / "snapshots" / "other-revision"
    snapshot.mkdir(parents=True)
    _write_json(snapshot / "config.json", {"model_type": "qwen3_5"})
    (snapshot / "model.safetensors").write_bytes(b"complete")
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("missing-revision", encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert metadata.resolve_unreferenced_cached_snapshot("publisher/model") is None


def test_unreferenced_snapshot_rejects_weight_link_outside_repo(monkeypatch, tmp_path):
    repo_root = tmp_path / "models--publisher--model"
    snapshot = repo_root / "snapshots" / "immutable-revision"
    snapshot.mkdir(parents=True)
    _write_json(snapshot / "config.json", {"model_type": "qwen3_5"})
    outside = tmp_path / "outside-model.safetensors"
    outside.write_bytes(b"complete")
    (snapshot / "model.safetensors").symlink_to(outside)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))

    assert metadata.resolve_unreferenced_cached_snapshot("publisher/model") is None


def test_read_model_metadata_prefers_local_directory_then_cache(monkeypatch, tmp_path):
    local = metadata.ModelMetadata({}, "local", tmp_path)
    cached = metadata.ModelMetadata({}, "cached", tmp_path)
    monkeypatch.setattr(metadata, "read_local_model_metadata", lambda name: local)
    monkeypatch.setattr(metadata, "read_cached_model_metadata", lambda name: cached)
    assert metadata.read_model_metadata("anything") is local

    monkeypatch.setattr(metadata, "read_local_model_metadata", lambda name: None)
    assert metadata.read_model_metadata("anything") is cached


def test_multimodal_config_and_sharded_weight_detection(tmp_path):
    assert metadata.config_indicates_multimodal(
        {"architectures": ["LlavaForConditionalGeneration"]}
    )
    assert metadata.config_indicates_multimodal({"audio_config": {}})
    assert not metadata.config_indicates_multimodal({"architectures": "not-a-list"})

    assert metadata.checkpoint_has_multimodal_weights(None) is None
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None

    # ARCHITECTURE-AGNOSTIC weight-evidence rule (round-5 regression fix,
    # restoring origin/main): a readable weight index with NO multimodal tensors
    # is a text-only verdict (``False``) for EVERY architecture, not just
    # Qwen3.5-MoE.  A language-only backbone (no vision/audio tensors) → text.
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "weight_map": {
                "language_model.model.layers.0.self_attn.q_proj.weight": (
                    "model.safetensors"
                ),
                "language_model.model.embed_tokens.weight": "model.safetensors",
                "language_model.lm_head.weight": "model.safetensors",
            }
        },
    )
    # No config, no vision tensors → text-only (False), architecture-agnostic.
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is False
    # Same with a VLM-capable arch declared: still text-only, because the
    # WEIGHTS carry no vision tensors (the #393 / codex #2 text-only-fork case).
    assert (
        metadata.checkpoint_has_multimodal_weights(
            tmp_path,
            {"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
        )
        is False
    )
    # A real vision tensor (``vision_encoder``) nested under ``language_model.``
    # is POSITIVE modality evidence → VLM (True).  Weight evidence, not the
    # namespace prefix, decides: a checkpoint that actually ships vision weights
    # is multimodal regardless of where they are nested.
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "weight_map": {
                "language_model.model.embed_tokens.weight": "model.safetensors",
                "language_model.vision_encoder.blocks.0.weight": "model.safetensors",
            }
        },
    )
    assert (
        metadata.checkpoint_has_multimodal_weights(
            tmp_path,
            {"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
        )
        is True
    )

    # A malformed weight_map (a list, not a dict) is unreadable evidence →
    # inconclusive (None), so the caller falls back to config/name heuristics.
    _write_json(tmp_path / "model.safetensors.index.json", {"weight_map": []})
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is None

    # An EMPTY-but-well-formed weight_map ({}) is a readable index with no
    # vision tensors → text-only (False), architecture-agnostic.
    _write_json(tmp_path / "model.safetensors.index.json", {"weight_map": {}})
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is False

    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"vision_tower.blocks.0.weight": "model.safetensors"}},
    )
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True

    # Single-file safetensors header path applies the SAME architecture-agnostic
    # rule: language-only header → text (False); vision tensor → VLM (True).
    (tmp_path / "model.safetensors.index.json").unlink()
    safetensors = tmp_path / "model.safetensors"
    _write_safetensors_header(safetensors, ["language_model.layers.0.weight"])
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is False

    _write_safetensors_header(safetensors, ["vision_tower.blocks.0.weight"])
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True

    _write_safetensors_header(safetensors, ["vision_encoder.blocks.0.weight"])
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True


def test_known_text_only_layout_rejects_modality_subtree_under_language_model():
    """FIX 3: the text-only allowlist is validated against precise text tensor
    paths, NOT the bare ``language_model.`` prefix.

    A modality subtree nested under ``language_model.`` (e.g.
    ``language_model.vision_encoder.blocks.0.weight``) must NOT be classified as
    text-only, or a real VLM whose vision encoder is namespaced under
    ``language_model.`` would be misrouted to the text loader.
    """
    cfg = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}

    # Precise real text layout (backbone + head) -> recognised as text-only.
    text_layout = {
        "language_model.model.embed_tokens.weight": "s",
        "language_model.model.layers.0.self_attn.q_proj.weight": "s",
        "language_model.model.norm.weight": "s",
        "language_model.lm_head.weight": "s",
    }
    assert metadata._known_text_only_weight_layout(text_layout, cfg) is True

    # Tied-embedding checkpoint (no ``lm_head``) is still text-only.
    tied_layout = {
        "language_model.model.embed_tokens.weight": "s",
        "language_model.model.layers.0.mlp.gate_proj.weight": "s",
    }
    assert metadata._known_text_only_weight_layout(tied_layout, cfg) is True

    # A vision subtree nested under ``language_model.`` is NOT text-only.
    nested_vision = {
        "language_model.model.embed_tokens.weight": "s",
        "language_model.vision_encoder.blocks.0.weight": "s",
    }
    assert metadata._known_text_only_weight_layout(nested_vision, cfg) is False

    # The bare-prefix form the OLD code accepted (``language_model.layers.*``
    # without the ``.model.`` segment) is not the real layout and is no longer
    # blindly trusted as text-only.
    bare_prefix = {"language_model.layers.0.weight": "s"}
    assert metadata._known_text_only_weight_layout(bare_prefix, cfg) is False

    # Wrong architecture -> never text-only regardless of tensor names.
    assert (
        metadata._known_text_only_weight_layout(
            text_layout, {"architectures": ["SomeOtherForCausalLM"]}
        )
        is False
    )


def test_qwen3_5_moe_text_tensor_allowlist_rejects_deep_modality_subtree():
    """codex #2: the qwen3.5-moe text allowlist enumerates the ACTUAL backbone
    children (``embed_tokens`` / ``layers`` / ``norm`` under
    ``language_model.model.``) + ``language_model.lm_head``, NOT the bare
    ``language_model.model.`` prefix.

    A modality subtree nested ONE LEVEL DEEPER — ``language_model.model
    .vision_encoder.*`` — matches the bare ``language_model.model.`` prefix the
    OLD code trusted and would let a genuine VLM reach an authoritative
    text-only verdict.  The tightened allowlist must reject it (inconclusive),
    while every real text tensor is still accepted.

    Tensor names are derived from mlx-lm ``models/qwen3_5.py``
    (``Qwen3_5TextModel`` → ``embed_tokens`` / ``layers`` / ``norm``; wrapper
    ``TextModel`` → ``lm_head``) — not invented.
    """
    # The exact codex #2 concern: deep vision subtree under ``.model.``.
    assert (
        metadata._is_qwen3_5_moe_text_tensor(
            "language_model.model.vision_encoder.blocks.0.weight"
        )
        is False
    )
    # Any other unknown modality subtree under ``.model.`` is likewise rejected.
    assert (
        metadata._is_qwen3_5_moe_text_tensor(
            "language_model.model.audio_tower.0.weight"
        )
        is False
    )

    # The real text tensors ARE still accepted.
    assert metadata._is_qwen3_5_moe_text_tensor(
        "language_model.model.embed_tokens.weight"
    )
    assert metadata._is_qwen3_5_moe_text_tensor(
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert metadata._is_qwen3_5_moe_text_tensor("language_model.model.norm.weight")
    assert metadata._is_qwen3_5_moe_text_tensor("language_model.lm_head.weight")
    # MoE expert tensors (the sanitizer emits ``...mlp.switch_mlp.*``) live
    # under ``language_model.model.layers.*`` and are still recognised.
    assert metadata._is_qwen3_5_moe_text_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    )

    # A whole checkpoint whose vision encoder nests under ``.model.`` is NOT a
    # text-only layout — the verdict must stay inconclusive (never text).
    cfg = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
    deep_vision_layout = {
        "language_model.model.embed_tokens.weight": "s",
        "language_model.model.layers.0.self_attn.q_proj.weight": "s",
        "language_model.model.vision_encoder.blocks.0.weight": "s",
    }
    assert metadata._known_text_only_weight_layout(deep_vision_layout, cfg) is False


def test_single_safetensors_glob_oserror_is_inconclusive(tmp_path, monkeypatch):
    """codex #5: file enumeration (``snapshot_dir.glob``) is inside the
    exception handler, so a permission error / stale mount / cache race yields
    the documented inconclusive result (``None``) instead of crashing routing.
    """
    from pathlib import Path

    def _boom(self, *args, **kwargs):
        raise OSError("stale mount / permission denied")

    monkeypatch.setattr(Path, "glob", _boom)

    # Direct helper: must not raise, returns inconclusive.
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None
    # Higher-level entry point (no index → falls through to the single-file
    # header path) must also stay inconclusive rather than crash.
    assert metadata.checkpoint_has_multimodal_weights(tmp_path, None) is None


def test_weight_index_has_independent_production_size_bound(tmp_path):
    weight_map = {
        "language_model." + "x" * metadata.MAX_METADATA_FILE_BYTES: "model.safetensors",
        "vision_tower.blocks.0.weight": "model.safetensors",
    }
    _write_json(tmp_path / "model.safetensors.index.json", {"weight_map": weight_map})

    assert (
        tmp_path / "model.safetensors.index.json"
    ).stat().st_size > metadata.MAX_METADATA_FILE_BYTES
    assert metadata.checkpoint_has_multimodal_weights(tmp_path) is True


def test_single_safetensors_header_rejects_corrupt_or_unsupported_shapes(tmp_path):
    model = tmp_path / "model.safetensors"

    model.write_bytes(b"tiny")
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    model.write_bytes((metadata.MAX_METADATA_FILE_BYTES + 1).to_bytes(8, "little"))
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    model.write_bytes((8).to_bytes(8, "little") + b"{}")
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    model.write_bytes((1).to_bytes(8, "little") + b"[")
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None

    header = json.dumps(["not", "an", "object"]).encode("utf-8")
    model.write_bytes(len(header).to_bytes(8, "little") + header)
    assert metadata._single_safetensors_has_multimodal_weights(tmp_path) is None
