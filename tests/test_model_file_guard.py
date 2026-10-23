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


@pytest.mark.parametrize('subfolder', ['', 'quantized'])
def test_configured_cache_repository_on_external_tier(tmp_path, monkeypatch, subfolder):
    import huggingface_hub.constants as constants

    cache = tmp_path / 'cache'
    cache.mkdir()
    external = tmp_path / 'cold' / 'stored-repository'
    snapshot = external / 'snapshots' / ('a' * 40)
    model = snapshot / subfolder
    model.mkdir(parents=True)
    blobs = external / 'blobs'
    blobs.mkdir()
    code = blobs / 'content-hash'
    code.write_text('# reviewed custom architecture')
    (model / 'config.json').write_text('{"model_file":"model.py"}')
    (model / 'model.py').symlink_to(code)
    entry = cache / 'models--org--model'
    entry.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(cache))
    validate_local_model_file(entry / 'snapshots' / ('a' * 40) / subfolder)
    # The exception requires the configured cache entry, not its unregistered
    # cold-storage pathname or a lookalike directory outside the cache.
    with pytest.raises(ValueError, match='must stay inside'):
        validate_local_model_file(model)
