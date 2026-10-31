# SPDX-License-Identifier: Apache-2.0
"""Consent at VLM processor load boundaries."""

import builtins
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rapid_mlx.utils.model_file_guard import validate_vlm_processor_code


def write_model(path, model_type, *, code=True):
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    if code:
        (path / "tokenizer_config.json").write_text(
            json.dumps({"auto_map": {"AutoTokenizer": "custom.Tokenizer"}})
        )


@pytest.fixture
def mllm_loader(monkeypatch):
    import mlx_vlm

    from rapid_mlx.models import mllm
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
    loader = Mock(
        return_value=(
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(tokenizer=SimpleNamespace()),
        )
    )
    monkeypatch.setattr(mlx_vlm, "load", loader)
    return mllm, loader


@pytest.mark.requires_mlx
def test_mllm_refuses_before_execution_and_retries(tmp_path, monkeypatch, mllm_loader):
    from rapid_mlx.utils import model_file_guard

    mllm, loader = mllm_loader
    write_model(tmp_path, "got")
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(f"open({str(marker)!r}, 'w').write('executed')")
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    instance = mllm.MLXMultimodalLM(str(tmp_path), enable_cache=False)
    with pytest.raises(ValueError, match="explicitly opt in"):
        instance.load()
    loader.assert_not_called()
    assert not instance._loaded and not marker.exists()
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    instance.load()
    assert instance._loaded
    loader.assert_called_once_with(str(tmp_path), trust_remote_code=True)
    monkeypatch.setattr(
        model_file_guard,
        "validate_vlm_processor_code",
        Mock(side_effect=AssertionError),
    )
    instance.load()
    assert loader.call_count == 1


@pytest.mark.requires_mlx
def test_real_got_processor_route_executes_only_with_consent(tmp_path, monkeypatch):
    import mlx_vlm.models.got.processing_got  # noqa: F401
    import transformers.dynamic_module_utils
    import transformers.utils
    import transformers.utils.hub

    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    model = tmp_path / "model"
    model.mkdir()
    marker = tmp_path / "executed"
    (model / "config.json").write_text(json.dumps({"model_type": "got"}))
    (model / "tokenizer_config.json").write_text(
        json.dumps(
            {"auto_map": {"AutoTokenizer": ["tok.T", None]}, "tokenizer_class": "T"}
        )
    )
    (model / "tok.py").write_text(
        "from transformers import PreTrainedTokenizer\n"
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "class T(PreTrainedTokenizer): pass\n"
    )
    modules = str(tmp_path / "modules")
    for module in (
        transformers.dynamic_module_utils,
        transformers.utils,
        transformers.utils.hub,
    ):
        monkeypatch.setattr(module, "HF_MODULES_CACHE", modules, raising=False)
    monkeypatch.syspath_prepend(modules)

    def loader(path, *, trust_remote_code):
        from mlx_vlm.utils import load_processor

        try:
            load_processor(Path(path), False, trust_remote_code=trust_remote_code)
        except Exception:
            pass
        return (object(), object())

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(ValueError, match="explicitly opt in"):
        load_vlm_with_policy(model, loader=loader)
    assert not marker.exists()

    loader(model, trust_remote_code=False)
    assert marker.exists()
    marker.unlink()
    # Force Transformers to re-execute the cached module for the consented load.
    with (model / "tok.py").open("a") as tokenizer_file:
        tokenizer_file.write("# consented reload\n")

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    load_vlm_with_policy(model, loader=loader)
    assert marker.exists()


@pytest.mark.requires_mlx
def test_policy_processor_validation_uses_disk_config(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    write_model(tmp_path, "got")
    loader = Mock()
    with pytest.raises(ValueError, match="explicitly opt in"):
        load_vlm_with_policy(
            tmp_path,
            loader=loader,
            model_config={"model_type": "qwen3_vl", "auto_map": None},
        )
    loader.assert_not_called()


@pytest.mark.requires_mlx
def test_inkling_refuses_before_loader(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    write_model(tmp_path, "inkling")
    loader = Mock()
    with pytest.raises(ValueError, match="explicitly opt in"):
        load_vlm_with_policy(tmp_path, loader=loader)
    loader.assert_not_called()


@pytest.mark.requires_mlx
@pytest.mark.parametrize("boundary", ["mllm", "policy"])
def test_auxiliary_requires_consent(tmp_path, monkeypatch, mllm_loader, boundary):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    mllm, loader = mllm_loader
    write_model(tmp_path, "moondream2", code=False)
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    instance = mllm.MLXMultimodalLM(str(tmp_path), enable_cache=False)
    call = (
        instance.load
        if boundary == "mllm"
        else lambda: load_vlm_with_policy(tmp_path, loader=loader)
    )
    with pytest.raises(ValueError, match="auxiliary repository.*explicitly opt in"):
        call()
    loader.assert_not_called()
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "1")
    call()
    loader.assert_called_once_with(str(tmp_path), trust_remote_code=True)


@pytest.mark.requires_mlx
@pytest.mark.parametrize(
    "environment,requested,allowed",
    [("1", None, True), (None, True, True), ("0", True, False)],
)
def test_policy_effective_consent(
    tmp_path, monkeypatch, environment, requested, allowed
):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    write_model(tmp_path, "got")
    if environment is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", environment)
    loader = Mock()
    if allowed:
        load_vlm_with_policy(tmp_path, loader=loader, trust_remote_code=requested)
        loader.assert_called_once_with(str(tmp_path), trust_remote_code=True)
    else:
        with pytest.raises(ValueError, match="explicitly opt in"):
            load_vlm_with_policy(tmp_path, loader=loader, trust_remote_code=requested)
        loader.assert_not_called()


@pytest.mark.requires_mlx
def test_forced_route_without_code_and_malformed_metadata(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    write_model(tmp_path, "got", code=False)
    loader = Mock()
    load_vlm_with_policy(tmp_path, loader=loader)
    loader.assert_called_once_with(str(tmp_path), trust_remote_code=False)
    loader.reset_mock()
    (tmp_path / "tokenizer_config.json").write_text("{")
    with pytest.raises(ValueError, match="tokenizer_config.json"):
        load_vlm_with_policy(tmp_path, loader=loader)
    loader.assert_not_called()


@pytest.mark.parametrize("model_type", ["qwen3_vl", "spark2_5", "kimi_k25", "lfm3_vl"])
def test_reviewed_routes_do_not_read_files(tmp_path, monkeypatch, model_type):
    write_model(tmp_path, model_type)
    config = {"model_type": model_type, "auto_map": {"AutoConfig": "custom.Config"}}
    monkeypatch.setattr(Path, "open", Mock(side_effect=AssertionError("file read")))
    monkeypatch.setattr(builtins, "open", Mock(side_effect=AssertionError("file read")))
    assert (
        validate_vlm_processor_code(tmp_path, config, trust_remote_code=False) is None
    )


@pytest.mark.parametrize(
    "filename",
    [
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ],
)
@pytest.mark.parametrize(
    "value", [{"auto_map": {"x": "y"}}, {"model_file": "custom.py"}, []]
)
def test_forced_route_checks_each_metadata_file(tmp_path, filename, value):
    (tmp_path / filename).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        validate_vlm_processor_code(
            tmp_path, {"model_type": "GOT"}, trust_remote_code=False
        )


@pytest.mark.requires_mlx
def test_local_loads_are_offline(tmp_path, monkeypatch):
    from rapid_mlx.utils.model_file_guard import load_vlm_with_policy

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setattr(
        socket.socket, "connect", Mock(side_effect=AssertionError("network"))
    )
    loader = Mock(return_value=(object(), object()))
    write_model(tmp_path, "got")
    with pytest.raises(ValueError, match="explicitly opt in"):
        load_vlm_with_policy(tmp_path, loader=loader)
    loader.assert_not_called()
    write_model(tmp_path, "qwen3_vl")
    assert load_vlm_with_policy(tmp_path, loader=loader) == loader.return_value
    loader.assert_called_once_with(str(tmp_path), trust_remote_code=False)


METADATA_NAMES = {
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
}


def forbid_metadata_opens(monkeypatch):
    path_open, builtin_open = Path.open, builtins.open
    seen = []

    def check(file):
        if isinstance(file, (str, bytes, Path)) and Path(file).name in METADATA_NAMES:
            seen.append(file)
            raise AssertionError(f"warm metadata read: {file}")

    def checked_path(self, *args, **kwargs):
        check(self)
        return path_open(self, *args, **kwargs)

    def checked_builtin(file, *args, **kwargs):
        check(file)
        return builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_path)
    monkeypatch.setattr(builtins, "open", checked_builtin)
    return seen


@pytest.mark.requires_mlx
@pytest.mark.asyncio
async def test_warm_mllm_engine_chat_skips_consent_io(
    tmp_path, monkeypatch, mllm_loader
):
    from unittest.mock import AsyncMock

    import mlx_vlm
    import mlx_vlm.utils

    from rapid_mlx.engine.batched import BatchedEngine
    from rapid_mlx.utils import model_file_guard

    mllm, loader = mllm_loader
    write_model(tmp_path, "qwen3_vl", code=False)
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    check = Mock(wraps=model_file_guard.validate_vlm_processor_code)
    monkeypatch.setattr(model_file_guard, "validate_vlm_processor_code", check)
    instance = mllm.MLXMultimodalLM(str(tmp_path), enable_cache=False)
    instance.load()
    check.assert_called_once()
    engine = BatchedEngine(str(tmp_path), force_mllm=True)
    engine._loaded = True
    engine._mllm_instance = instance
    engine._model, engine._processor = instance.model, instance.processor
    output = SimpleNamespace(
        output_text="hello",
        output_token_ids=[],
        prompt_tokens=1,
        completion_tokens=1,
        finish_reason="stop",
    )

    async def generate(**kwargs):
        kwargs["on_request_committed"]()
        return output

    scheduler = SimpleNamespace(
        config=SimpleNamespace(max_concurrent_requests=1),
        generate=AsyncMock(side_effect=generate),
    )
    engine._mllm_scheduler = scheduler
    monkeypatch.setattr(engine, "_apply_chat_template", lambda *a, **kw: "hello")
    monkeypatch.setattr(engine, "_prompt_parse_context", lambda *a, **kw: None)
    forbidden = Mock(side_effect=AssertionError("warm load boundary"))
    monkeypatch.setattr(model_file_guard, "validate_vlm_processor_code", forbidden)
    monkeypatch.setattr(model_file_guard, "resolve_vlm_checkpoint", forbidden)
    monkeypatch.setattr(mlx_vlm.utils, "load_config", forbidden)
    monkeypatch.setattr(mlx_vlm, "load", forbidden)
    seen = forbid_metadata_opens(monkeypatch)
    for _ in range(5):
        result = await engine.chat([{"role": "user", "content": "hello"}])
        assert result.text == "hello"
    assert scheduler.generate.await_count == 5
    forbidden.assert_not_called()
    assert not seen
    assert loader.call_count == 1


@pytest.mark.requires_mlx
def test_warm_stt_skips_vlm_consent_io(tmp_path, monkeypatch):
    import mlx_audio.stt.utils as native

    from rapid_mlx import model_downloads
    from rapid_mlx.audio.stt import STTEngine
    from rapid_mlx.utils import model_file_guard

    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    monkeypatch.setattr(model_downloads, "require_local", lambda name: None)
    monkeypatch.setattr(native, "get_model_path", lambda *a, **kw: tmp_path)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="sensevoice"),
        generate=Mock(
            return_value=SimpleNamespace(text="hello", segments=[], language="en")
        ),
    )
    loader = Mock(return_value=model)
    monkeypatch.setattr(native, "base_load_model", loader)
    engine = STTEngine("example/speech", enable_vad_pretrim=False)
    engine.load()
    forbidden = Mock(side_effect=AssertionError("warm VLM load boundary"))
    for name in (
        "validate_vlm_processor_code",
        "load_vlm_with_policy",
        "resolve_vlm_checkpoint",
    ):
        monkeypatch.setattr(model_file_guard, name, forbidden)
    seen = forbid_metadata_opens(monkeypatch)
    for _ in range(3):
        assert engine.transcribe("speech.wav").text == "hello"
    forbidden.assert_not_called()
    assert not seen
    assert model.generate.call_count == 3


def test_processor_guard_references_stay_at_load_boundaries():
    import ast

    root = Path(__file__).resolve().parents[1] / "rapid_mlx"
    references = []
    for file in root.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not (
                (
                    isinstance(node, ast.Name)
                    and node.id == "validate_vlm_processor_code"
                )
                or (
                    isinstance(node, ast.Attribute)
                    and node.attr == "validate_vlm_processor_code"
                )
                or (
                    isinstance(node, ast.alias)
                    and node.name == "validate_vlm_processor_code"
                )
                or (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "validate_vlm_processor_code"
                )
            ):
                continue
            ancestors = []
            current = node
            while current in parents:
                current = parents[current]
                if isinstance(
                    current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    ancestors.append(current.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ancestors.insert(0, node.name)
            references.append(
                (file.relative_to(root).as_posix(), tuple(reversed(ancestors)))
            )
    assert references
    assert all(
        (
            file == "utils/model_file_guard.py"
            and scope in {("validate_vlm_processor_code",), ("load_vlm_with_policy",)}
        )
        or (file == "models/mllm.py" and scope == ("MLXMultimodalLM", "load"))
        for file, scope in references
    ), references
