# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for local config.json::model_file containment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rapid_mlx.utils.model_file_guard import validate_local_model_file


def _model_dir(tmp_path: Path, model_file: str | None) -> Path:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps({"model_file": model_file}), encoding="utf-8"
    )
    return model_root


def test_accepts_existing_nested_python_model_file(tmp_path: Path) -> None:
    model_root = _model_dir(tmp_path, "custom/model.py")
    custom_dir = model_root / "custom"
    custom_dir.mkdir()
    (custom_dir / "model.py").write_text("# local custom model\n", encoding="utf-8")

    validate_local_model_file(model_root)


def test_accepts_local_model_without_custom_model_file(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")

    validate_local_model_file(model_root)


def test_accepts_null_model_file(tmp_path: Path) -> None:
    model_root = _model_dir(tmp_path, None)

    validate_local_model_file(model_root)


def test_ignores_remote_model_id() -> None:
    validate_local_model_file("mlx-community/example-model-4bit")


def test_rejects_absolute_model_file(tmp_path: Path) -> None:
    model_root = _model_dir(tmp_path, str(tmp_path / "outside.py"))

    with pytest.raises(ValueError, match="relative Python file"):
        validate_local_model_file(model_root)


def test_rejects_parent_traversal(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    model_root = _model_dir(tmp_path, "../outside.py")

    with pytest.raises(ValueError, match="must stay inside"):
        validate_local_model_file(model_root)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    model_root = _model_dir(tmp_path, "custom.py")
    (model_root / "custom.py").symlink_to(outside)

    with pytest.raises(ValueError, match="must stay inside"):
        validate_local_model_file(model_root)


def test_rejects_missing_model_file(tmp_path: Path) -> None:
    model_root = _model_dir(tmp_path, "missing.py")

    with pytest.raises(ValueError, match="existing Python file"):
        validate_local_model_file(model_root)


def test_rejects_directory_model_file(tmp_path: Path) -> None:
    model_root = _model_dir(tmp_path, "custom.py")
    (model_root / "custom.py").mkdir()

    with pytest.raises(ValueError, match="regular Python file"):
        validate_local_model_file(model_root)


def test_rejects_non_python_model_file(tmp_path: Path) -> None:
    model_root = _model_dir(tmp_path, "custom.txt")
    (model_root / "custom.txt").write_text("not Python\n", encoding="utf-8")

    with pytest.raises(ValueError, match="relative Python file"):
        validate_local_model_file(model_root)


def test_shared_loader_validates_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rapid_mlx.utils import tokenizer

    events = []
    expected = (object(), object())

    monkeypatch.setattr(
        tokenizer,
        "validate_local_model_file",
        lambda model_name: events.append(("validate", model_name)),
    )
    monkeypatch.setattr(
        tokenizer,
        "_load_model_with_fallback_impl",
        lambda model_name, tokenizer_config=None: (
            events.append(("load", model_name)),
            expected,
        )[1],
    )
    monkeypatch.setattr(tokenizer, "_resolve_model_path", lambda model_name: None)
    monkeypatch.setattr(tokenizer, "_resolve_subfolder_checkpoint", lambda name: name)
    monkeypatch.setattr(tokenizer, "validate_model_config_file", lambda name: {})
    monkeypatch.setattr(tokenizer, "_post_load_ubc_evict", lambda model_name: None)

    assert tokenizer.load_model_with_fallback("local-model") is expected
    assert events == [("validate", "local-model"), ("load", "local-model")]


def test_shared_loader_pins_concrete_snapshot_on_loaded_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rapid_mlx.utils import tokenizer

    snapshot = tmp_path / "snapshots" / "immutable-revision"
    snapshot.mkdir(parents=True)

    class ImmutableModel:
        __slots__ = ()

    model = ImmutableModel()
    expected = (model, object())
    loaded = []

    monkeypatch.setattr(tokenizer, "_local_snapshot_if_cached", lambda name: name)
    monkeypatch.setattr(tokenizer, "_resolve_model_path", lambda name: snapshot)
    monkeypatch.setattr(tokenizer, "validate_local_model_file", lambda name: None)
    monkeypatch.setattr(tokenizer, "_model_requires_remote_code", lambda name: False)
    monkeypatch.setattr(
        tokenizer,
        "_load_model_with_fallback_impl",
        lambda name, tokenizer_config=None: (loaded.append(name), expected)[1],
    )
    monkeypatch.setattr(tokenizer, "_post_load_ubc_evict", lambda name: None)

    result = tokenizer.load_model_with_fallback("org/model", return_source=True)
    assert result[:2] == expected
    assert result[2] == str(snapshot)
    assert loaded == [str(snapshot)]


@pytest.mark.requires_mlx
@pytest.mark.parametrize("lazy", [False, True])
def test_text_dispatch_keeps_the_inspected_snapshot_when_remote_reference_moves(
    monkeypatch, tmp_path, lazy
):
    from types import SimpleNamespace

    import mlx_lm

    from rapid_mlx.utils import tokenizer

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text('{"model_type":"llama"}')
    (second / "config.json").write_text(
        '{"model_type":"llama","model_file":"custom.py"}'
    )
    remote_resolutions = []

    def resolve(name):
        if name == "org/moving-model":
            remote_resolutions.append(name)
            return first if len(remote_resolutions) == 1 else second
        return Path(name)

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(tokenizer, "_resolve_subfolder_checkpoint", lambda name: name)
    monkeypatch.setattr(tokenizer, "_local_snapshot_if_cached", lambda name: name)
    monkeypatch.setattr(tokenizer, "_resolve_model_path", resolve)
    loaded = []
    expected = (SimpleNamespace(), SimpleNamespace(chat_template="local"))

    def load(name, tokenizer_config=None, **kwargs):
        loaded.append(name)
        assert tokenizer_config["trust_remote_code"] is False
        return expected

    monkeypatch.setattr(mlx_lm, "load", load)
    monkeypatch.setattr(tokenizer, "_load_model_with_fallback_impl", load)
    for helper in [
        "_try_inject_mtp_post_load",
        "augment_eos_token_ids_from_generation_config",
        "repair_byte_level_decoder",
        "_post_load_ubc_evict",
    ]:
        monkeypatch.setattr(tokenizer, helper, lambda *args, **kwargs: None)
    result = tokenizer.load_model_with_fallback(
        "org/moving-model", lazy=lazy, return_source=True
    )
    assert result[:2] == expected
    assert result[2] == str(first)
    assert loaded == [str(first)]
    assert remote_resolutions == ["org/moving-model"]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_model_file_cannot_execute_without_consent(tmp_path, monkeypatch, lazy, remote):
    from rapid_mlx.utils import tokenizer

    root = _model_dir(tmp_path, "model.py")
    marker = tmp_path / "executed"
    (root / "model.py").write_text(f"open({str(marker)!r}, 'w').write('executed')")
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(tokenizer, "_local_snapshot_if_cached", lambda name: name)
    monkeypatch.setattr(tokenizer, "_resolve_model_path", lambda name: root)
    with pytest.raises(ValueError, match="explicitly opt in"):
        tokenizer.load_model_with_fallback(
            "org/repo" if remote else str(root), lazy=lazy
        )
    assert not marker.exists()


def test_checked_model_file_requires_consent_before_loader(tmp_path, monkeypatch):
    from rapid_mlx.model_load_errors import load_model_checked

    root = _model_dir(tmp_path, "model.py")
    (root / "model.py").write_text("# reviewed custom code")
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    calls = []

    def loader(path, **kwargs):
        calls.append(path)
        return object(), {}

    with pytest.raises(ValueError, match="explicitly opt in"):
        load_model_checked(loader, root)
    assert not calls
    load_model_checked(loader, root, trust_remote_code=True)
    assert calls == [root]
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    with pytest.raises(ValueError, match="explicitly opt in"):
        load_model_checked(loader, root, trust_remote_code=True)


def test_reviewed_runtime_overlay_ignores_repository_model_file(tmp_path, monkeypatch):
    from rapid_mlx.model_load_errors import load_model_checked

    root = _model_dir(tmp_path, "does-not-exist.py")
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    result = load_model_checked(
        lambda path, **kwargs: ("safe", kwargs["model_config"]),
        root,
        model_config={"model_file": None, "auto_map": None},
    )
    assert result == ("safe", {"model_file": None, "auto_map": None})


def test_hub_snapshot_accepts_same_repository_blob(tmp_path, monkeypatch):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--model"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    blobs = repo / "blobs"
    blobs.mkdir()
    (blobs / "hash").write_text("# custom model")
    (snapshot / "config.json").write_text('{"model_file":"model.py"}')
    (snapshot / "model.py").symlink_to(blobs / "hash")
    validate_local_model_file(snapshot)


@pytest.mark.requires_mlx
def test_vlm_policy_pins_snapshot_and_blocks_code(tmp_path, monkeypatch):
    import mlx_vlm
    import mlx_vlm.utils

    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    root = _model_dir(tmp_path, "model.py")
    (root / "model.py").write_text("# custom model")
    resolved = []
    calls = []

    def resolve(name, **kwargs):
        resolved.append((name, kwargs))
        return root

    monkeypatch.setattr(mlx_vlm.utils, "get_model_path", resolve)
    monkeypatch.setattr(mlx_vlm, "load", lambda *a, **k: calls.append((a, k)))
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(ValueError, match="explicitly opt in"):
        load_vlm_with_policy("org/repo", revision="pinned")
    assert calls == []
    load_vlm_with_policy("org/repo", revision="pinned", trust_remote_code=True)
    assert calls == [((str(root),), {"trust_remote_code": True})]
    assert resolved == [("org/repo", {"revision": "pinned"})] * 2


def test_checked_model_file_validates_effective_overlay(tmp_path, monkeypatch):
    from rapid_mlx.model_load_errors import load_model_checked

    root = _model_dir(tmp_path, None)
    (root / "config.json").write_text('{"model_type":"llama"}')
    outside = tmp_path / "outside.py"
    outside.write_text("# must not be loaded")
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(ValueError, match="must stay inside"):
        load_model_checked(
            lambda *args, **kwargs: pytest.fail("unsafe loader called"),
            root,
            trust_remote_code=True,
            model_config={"model_file": "../outside.py"},
        )


@pytest.mark.parametrize("subfolder", ["", "quantized"])
def test_configured_cache_repository_on_external_tier(tmp_path, monkeypatch, subfolder):
    import huggingface_hub.constants as constants

    cache = tmp_path / "cache"
    cache.mkdir()
    external = tmp_path / "cold" / "stored-repository"
    snapshot = external / "snapshots" / ("a" * 40)
    model = snapshot / subfolder
    model.mkdir(parents=True)
    blobs = external / "blobs"
    blobs.mkdir()
    code = blobs / "content-hash"
    code.write_text("# reviewed custom architecture")
    (model / "config.json").write_text('{"model_file":"model.py"}')
    (model / "model.py").symlink_to(code)
    entry = cache / "models--org--model"
    entry.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    validate_local_model_file(entry / "snapshots" / ("a" * 40) / subfolder)
    # The exception requires the configured cache entry, not its unregistered
    # cold-storage pathname or a lookalike directory outside the cache.
    with pytest.raises(ValueError, match="must stay inside"):
        validate_local_model_file(model)


def test_unresolved_checked_model_refuses_loader():
    from rapid_mlx.model_load_errors import load_model_checked

    with pytest.raises(ValueError, match="Cannot inspect"):
        load_model_checked(
            lambda *a, **k: pytest.fail("must not load"), "missing/repository"
        )


@pytest.mark.requires_mlx
def test_vlm_revision_subfolder_and_invalid_config(tmp_path, monkeypatch):
    import mlx_vlm.utils

    from rapid_mlx import model_aliases
    from rapid_mlx.utils import model_file_guard as guard
    from rapid_mlx.utils import tokenizer

    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: "a" * 40)
    with pytest.raises(ValueError, match="conflicting revision"):
        guard.resolve_vlm_checkpoint("org/repo", revision="b" * 40)
    monkeypatch.setattr(
        tokenizer, "_resolve_subfolder_checkpoint", lambda name: str(tmp_path)
    )
    assert guard.resolve_vlm_checkpoint("org/repo") == tmp_path
    monkeypatch.setattr(model_aliases, "resolve_revision", lambda name: None)
    monkeypatch.setattr(model_aliases, "resolve_subfolder", lambda name: "4bit")
    calls = []
    monkeypatch.setattr(
        mlx_vlm.utils, "get_model_path", lambda name, **kw: calls.append(kw) or tmp_path
    )
    assert guard.resolve_vlm_checkpoint("org/repo") == tmp_path / "4bit"
    assert calls == [{"revision": None, "allow_patterns": ["4bit/*"]}]
    (tmp_path / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="must contain an object"):
        guard.load_vlm_with_policy(tmp_path)


def test_unresolved_shared_loader_refuses_dispatch(monkeypatch):
    from rapid_mlx.utils import tokenizer

    for name in ["_resolve_subfolder_checkpoint", "_local_snapshot_if_cached"]:
        monkeypatch.setattr(tokenizer, name, lambda value: value)
    monkeypatch.setattr(tokenizer, "_resolve_model_path", lambda name: None)
    monkeypatch.setattr(tokenizer, "_rapid_owned_model_config", lambda name: None)
    with pytest.raises(ValueError, match="Cannot inspect"):
        tokenizer.load_model_with_fallback("missing/repository")


@pytest.mark.requires_mlx
def test_native_glm_requires_installed_gate(tmp_path, monkeypatch):
    from mlx_lm.models import deepseek_v32

    from rapid_mlx.utils import tokenizer

    (tmp_path / "config.json").write_text('{"model_type":"glm_moe_dsa"}')
    monkeypatch.setattr(tokenizer, "_install_dsv32_indexer_gate", lambda: None)
    monkeypatch.setattr(deepseek_v32, "_RAPID_MLX_INDEXER_GATE_INSTALLED", False)
    with pytest.raises(RuntimeError, match="adapter is unavailable"):
        tokenizer._uses_rapid_owned_runtime(str(tmp_path))


@pytest.mark.requires_mlx
def test_pflash_uses_checked_loader(tmp_path, monkeypatch):
    from rapid_mlx import engine_core
    from rapid_mlx.bench.pflash_replication import _build_engine
    from rapid_mlx.utils import tokenizer

    calls = []
    monkeypatch.setattr(
        tokenizer,
        "load_model_with_fallback",
        lambda path: calls.append(path) or ("model", "tokenizer"),
    )
    monkeypatch.setattr(engine_core, "AsyncEngineCore", lambda *args: args)
    model, token, engine = _build_engine(
        str(tmp_path), pflash_mode="off", keep_ratio=0.5
    )
    assert calls == [str(tmp_path)]
    assert (model, token) == ("model", "tokenizer")
    assert engine[:2] == (model, token)


def test_tokenizer_fallback_preserves_consent(monkeypatch):
    import sys
    from types import ModuleType

    from rapid_mlx.utils import tokenizer

    module = ModuleType("mlx_lm")
    module.load = lambda *a, **k: pytest.fail("high-level load must not run")
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setattr(tokenizer, "_register_vendored_archs", lambda: None)
    monkeypatch.setattr(
        tokenizer, "_neutralize_unbundled_template_types", lambda name, config: config
    )
    monkeypatch.setattr(tokenizer, "_needs_tokenizer_fallback", lambda name: True)
    calls = []
    monkeypatch.setattr(
        tokenizer,
        "_load_with_tokenizer_fallback",
        lambda name, **kw: calls.append(kw) or ("model", "tokenizer"),
    )
    assert tokenizer._load_model_with_fallback_impl(
        "local", {"trust_remote_code": False}
    ) == ("model", "tokenizer")
    assert calls == [{"enable_dspark": False, "trust_remote_code": False}]
