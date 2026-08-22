# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for local config.json::model_file containment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.utils.model_file_guard import validate_local_model_file


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
    from vllm_mlx.utils import tokenizer

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
    monkeypatch.setattr(tokenizer, "_post_load_ubc_evict", lambda model_name: None)

    assert tokenizer.load_model_with_fallback("local-model") is expected
    assert events == [("validate", "local-model"), ("load", "local-model")]


def test_shared_loader_pins_concrete_snapshot_on_loaded_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vllm_mlx.utils import tokenizer

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
