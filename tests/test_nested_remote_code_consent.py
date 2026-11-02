# SPDX-License-Identifier: Apache-2.0
"""Consent covers processor metadata at the paths forced-trust loaders open."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.utils.model_file_guard import RemoteCodeConsentRequired


def _write_metadata(directory, filename="tokenizer_config.json", **fields):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(fields), encoding="utf-8")


@pytest.fixture
def stt_loader(monkeypatch):
    from rapid_mlx.audio.stt import STTEngine

    native = ModuleType("mlx_audio.stt.utils")
    native.base_load_model = Mock(
        return_value=SimpleNamespace(config=SimpleNamespace(model_type="fun_asr_nano"))
    )
    exec(
        "def load_model(model_path, **kwargs):\n"
        "    return base_load_model(model_path=model_path, **kwargs)\n",
        native.__dict__,
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return STTEngine, native.base_load_model


@pytest.mark.parametrize("filename", ["tokenizer_config.json", "config.json"])
@pytest.mark.parametrize("location", ["child", "absolute", "dict_value"])
def test_stt_nested_metadata_blocks_before_native_dispatch(
    tmp_path, monkeypatch, stt_loader, filename, location
):
    engine_class, base = stt_loader
    root = tmp_path / "checkpoint"
    tokenizer = tmp_path / "outside" if location != "child" else root / "Qwen3-0.6B"
    config = {"model_type": "fun_asr_nano"}
    if location == "dict_value":
        config["text_config"] = {"tokenizer_path": str(tokenizer)}
    else:
        config["qwen_tokenizer_path"] = (
            "Qwen3-0.6B" if location == "child" else str(tokenizer)
        )
    _write_metadata(root, "config.json", **config)
    _write_metadata(tokenizer, filename, auto_map={"AutoTokenizer": ["tok.T", None]})
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    engine = engine_class(str(root), enable_vad_pretrim=False)
    with pytest.raises(RemoteCodeConsentRequired, match="declares auto_map") as exc:
        engine.load()
    assert ("Qwen3-0.6B" if location == "child" else "outside") in str(exc.value)
    base.assert_not_called()
    assert not engine._loaded

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    engine.load()
    base.assert_called_once_with(model_path=root)
    assert engine._loaded

    # Benign nested metadata must keep the native load path available.
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    _write_metadata(tokenizer, filename, tokenizer_class="Qwen2Tokenizer")
    base.reset_mock()
    engine_class(str(root), enable_vad_pretrim=False).load()
    base.assert_called_once_with(model_path=root)


@pytest.fixture(params=["policy", pytest.param("mllm", marks=pytest.mark.requires_mlx)])
def vlm_loader(tmp_path, monkeypatch, request):
    entrypoint = request.param
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    if entrypoint == "mllm":
        # Import the real runtime before installing loader doubles. Forcing MLX
        # imports to fail can leave dependent modules partially initialized.
        pytest.importorskip("mlx.core", exc_type=ImportError)
        pytest.importorskip("mlx_vlm", exc_type=ImportError)
        mllm = pytest.importorskip("rapid_mlx.models.mllm", exc_type=ImportError)
        from rapid_mlx.patches import (
            glm5_next_forget_gate_quant,
            glm5_next_processor,
            glm5_next_runtime,
        )

        monkeypatch.setattr(mllm, "_require_mlx_vlm", lambda: None)
        monkeypatch.setattr(
            glm5_next_runtime, "install_glm5_next_runtime_fix", lambda: None
        )
        monkeypatch.setattr(
            glm5_next_processor, "install_glm5_next_processor_patch", lambda: None
        )
        monkeypatch.setattr(
            glm5_next_forget_gate_quant,
            "install_glm5_next_forget_gate_quant_fix",
            lambda: None,
        )

    vlm = ModuleType("mlx_vlm")
    vlm.__path__ = []
    utils = ModuleType("mlx_vlm.utils")
    utils.get_model_path = lambda path, **kwargs: path

    def load_config(path, **kwargs):
        with (tmp_path / "config.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    utils.load_config = load_config
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    _write_metadata(tmp_path, "config.json", model_type="inkling")
    _write_metadata(tmp_path, auto_map={"AutoTokenizer": ["tok.T", None]})
    loader = Mock(
        return_value=(
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(tokenizer=SimpleNamespace()),
        )
    )
    vlm.load = loader
    if entrypoint == "mllm":

        def load(**kwargs):
            instance = mllm.MLXMultimodalLM(str(tmp_path), enable_cache=False, **kwargs)
            instance.load()
            assert instance._loaded

    else:

        def load(**kwargs):
            return load_vlm_with_policy(tmp_path, loader=loader, **kwargs)

    return load, loader


@pytest.mark.parametrize(
    "model_type", ["inkling", "got", "inkling_mm_model", "moondream1", "moondream2"]
)
def test_processor_consent_before_vlm_dispatch(
    tmp_path, monkeypatch, vlm_loader, model_type
):
    load, loader = vlm_loader
    _write_metadata(tmp_path, "config.json", model_type=model_type)
    external = model_type.startswith("moondream")
    _write_metadata(
        tmp_path,
        **({} if external else {"auto_map": {"AutoTokenizer": ["tok.T", None]}}),
    )
    reason = "external repository" if external else "auto_map"
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(RemoteCodeConsentRequired, match=reason):
        load()
    loader.assert_not_called()

    # Process opt-out wins even over an explicit API opt-in.
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    with pytest.raises(RemoteCodeConsentRequired, match=reason):
        load(trust_remote_code=True)
    loader.assert_not_called()

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE")
    load(trust_remote_code=True)
    assert loader.call_args.kwargs["trust_remote_code"] is True
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    load()
    assert loader.call_args.kwargs["trust_remote_code"] is True
    assert loader.call_count == 2


@pytest.mark.parametrize("field", ["auto_map", "model_file"])
def test_vlm_api_opt_out_overrides_environment(
    tmp_path, monkeypatch, vlm_loader, field
):
    load, loader = vlm_loader
    _write_metadata(tmp_path, **{field: "custom.py"})
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    with pytest.raises(RemoteCodeConsentRequired, match=field):
        load(trust_remote_code=False)
    loader.assert_not_called()


@pytest.mark.parametrize("trusted", [False, True])
def test_vlm_processor_policy_skips_io(tmp_path, monkeypatch, trusted):
    from pathlib import Path

    from rapid_mlx.utils import model_file_guard as guard

    fail = Mock(side_effect=AssertionError("processor metadata must not be read"))
    monkeypatch.setattr(guard, "validate_processor_code", fail)
    monkeypatch.setattr(Path, "open", fail)
    # Trusted forced/external types and reviewed non-forced types need no scan.
    types = ["qwen3_vl", "got", "moondream2"] if trusted else ["qwen3_vl"]
    for model_type in types:
        guard.validate_vlm_processor_policy(
            tmp_path,
            {"model_type": model_type, "auto_map": {"AutoTokenizer": "tok.T"}},
            trust_remote_code=trusted,
        )
    fail.assert_not_called()


def test_z_image_tokenizer_consent_before_vlm_dispatch(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    vlm = ModuleType("mlx_vlm")
    vlm.__path__ = []
    utils = ModuleType("mlx_vlm.utils")
    utils.get_model_path = lambda path, **kwargs: path

    def load_config(path, **kwargs):
        with (tmp_path / "config.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    utils.load_config = load_config
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    _write_metadata(tmp_path, "config.json", model_type="z_image")
    _write_metadata(tmp_path / "tokenizer", auto_map={"AutoTokenizer": ["tok.T", None]})
    loader = Mock(
        return_value=(
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(tokenizer=SimpleNamespace()),
        )
    )
    vlm.load = loader

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    with pytest.raises(RemoteCodeConsentRequired, match="auto_map"):
        load_vlm_with_policy(tmp_path, loader=loader)
    loader.assert_not_called()

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    load_vlm_with_policy(tmp_path, loader=loader)
    loader.assert_called_once_with(str(tmp_path), trust_remote_code=True)


@pytest.mark.parametrize(
    "filename",
    [
        "config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ],
)
@pytest.mark.parametrize("field", ["auto_map", "model_file"])
def test_stt_checks_unreferenced_immediate_directories(
    tmp_path, monkeypatch, stt_loader, filename, field
):
    engine_class, base = stt_loader
    _write_metadata(tmp_path, "config.json", model_type="fun_asr_nano")
    _write_metadata(tmp_path / "nested", filename, **{field: "custom.py"})
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    with pytest.raises(RemoteCodeConsentRequired, match="nested"):
        engine_class(str(tmp_path), enable_vad_pretrim=False).load()
    base.assert_not_called()


def test_processor_code_accepts_nonexistent_path(tmp_path):
    from rapid_mlx.utils.model_file_guard import validate_processor_code

    missing = tmp_path / "missing-checkpoint"
    assert not missing.exists()
    assert validate_processor_code(missing) is None


def test_directory_scan_bounds_and_realpath_deduplication(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import validate_processor_code

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    _write_metadata(tmp_path, "config.json", self_path=".")
    metadata = {"auto_map": {"AutoTokenizer": ["tok.T", None]}}
    _write_metadata(tmp_path / ".hidden", **metadata)
    _write_metadata(tmp_path / "visible" / "grandchild", **metadata)
    (tmp_path / "visible" / "back_to_root").symlink_to(
        tmp_path, target_is_directory=True
    )
    (tmp_path / "alias").symlink_to(tmp_path / "visible", target_is_directory=True)
    opened = []
    original = type(tmp_path).open

    def open_metadata(path, *args, **kwargs):
        opened.append(path.resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", open_metadata)
    validate_processor_code(tmp_path)
    assert opened.count((tmp_path / "visible" / "tokenizer_config.json").resolve()) == 1

    # A config-named path is checked even beyond the automatic scan depth.
    _write_metadata(tmp_path, "config.json", text_config={"path": "visible/grandchild"})
    with pytest.raises(RemoteCodeConsentRequired, match="visible/grandchild"):
        validate_processor_code(tmp_path)


def test_configured_directory_is_checked_beyond_scan_cap(tmp_path, monkeypatch):
    from pathlib import Path

    from rapid_mlx.utils.model_file_guard import validate_processor_code

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    directories = [tmp_path / f"tokenizer-{index}" for index in range(65)]
    for directory in directories:
        directory.mkdir()
    _write_metadata(tmp_path, "config.json", model_type="fun_asr_nano")
    _write_metadata(directories[-1], auto_map={"AutoTokenizer": ["tok.T", None]})
    original = Path.iterdir
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda path: iter(directories) if path == tmp_path else original(path),
    )
    validate_processor_code(tmp_path)

    _write_metadata(tmp_path, "config.json", qwen_tokenizer_path=directories[-1].name)
    with pytest.raises(RemoteCodeConsentRequired, match="tokenizer-64"):
        validate_processor_code(tmp_path)


def test_forced_trust_tokenizer_is_checked_beyond_scan_cap(tmp_path, monkeypatch):
    from pathlib import Path

    from rapid_mlx.utils.model_file_guard import validate_processor_code

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    for index in range(70):
        (tmp_path / f"a{index:02d}").mkdir()
    _write_metadata(tmp_path, "config.json", model_type="z_image")
    _write_metadata(tmp_path / "tokenizer", auto_map={"AutoTokenizer": ["tok.T", None]})
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter(sorted(original(path))))

    with pytest.raises(RemoteCodeConsentRequired, match="tokenizer.*auto_map"):
        validate_processor_code(tmp_path)


def test_speculative_config_paths_skip_invalid_strings(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import validate_processor_code

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    _write_metadata(
        tmp_path, "config.json", description="x" * 300, other_text="contains\x00nul"
    )

    validate_processor_code(tmp_path)
