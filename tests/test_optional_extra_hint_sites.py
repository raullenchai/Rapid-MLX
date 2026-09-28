# SPDX-License-Identifier: Apache-2.0
"""Direct contracts for install-method-aware hints at lazy failure sites."""

from __future__ import annotations

import argparse
import ast
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException, Request


def test_hybrid_runtime_absent_and_broken_are_distinct(monkeypatch) -> None:
    from rapid_mlx.api import utils
    from rapid_mlx.models import mllm

    monkeypatch.setattr(utils, "is_mllm_model", lambda _name: True)
    monkeypatch.setattr(utils, "mllm_backbone_cache_mode", lambda _name: "arrays")
    monkeypatch.setattr(utils, "physical_ram_gb", lambda: 64.0)

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, None),
    )
    absent = utils.resolve_serving_lane_decision(
        "local/model", vision_min_memory_gb=32.0
    )
    assert absent.reason == "vision_runtime_absent"
    assert absent.auto_text_fallback is True

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.BROKEN, "PIL"),
    )
    broken = utils.resolve_serving_lane_decision(
        "local/model", vision_min_memory_gb=32.0
    )
    assert broken.is_mllm is True
    assert broken.reason == "vision_supported"


def test_hybrid_runtime_absence_is_resolved_without_memory_floor(monkeypatch) -> None:
    from rapid_mlx.api import utils
    from rapid_mlx.models import mllm

    monkeypatch.setattr(utils, "is_mllm_model", lambda _name: True)
    monkeypatch.setattr(utils, "mllm_backbone_cache_mode", lambda _name: "arrays")
    monkeypatch.setattr(utils, "mllm_hybrid_runtime_supported", lambda: False)
    monkeypatch.setattr(utils, "physical_ram_gb", lambda: 16.0)
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, None),
    )

    decision = utils.resolve_serving_lane_decision(
        "local/model", vision_min_memory_gb=None
    )

    assert decision.reason == "vision_hybrid_runtime_unsupported"
    assert decision.auto_text_fallback is True


def test_lazy_benchmark_hints_are_pinned(monkeypatch) -> None:
    from rapid_mlx import benchmark

    monkeypatch.setattr(benchmark, "Image", None)
    with pytest.raises(ImportError, match=r"rapid-mlx\[vision\]=="):
        benchmark.download_test_image("https://invalid.example")

    monkeypatch.setattr(benchmark, "cv2", None)
    with pytest.raises(ImportError, match=r"rapid-mlx\[vision\]=="):
        benchmark.create_test_video()
    with pytest.raises(ImportError, match=r"rapid-mlx\[vision\]=="):
        benchmark.get_video_info("missing.mp4")


def test_cli_broken_runtime_stays_on_vision_and_desktop_does_not_warn(
    monkeypatch, capsys
) -> None:
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile
    from rapid_mlx.models import mllm

    profile = resolve_profile("qwen3.5-4b-4bit")
    assert profile is not None
    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.BROKEN, "PIL"),
    )
    args = argparse.Namespace(
        model="qwen3.5-4b-4bit",
        mllm=False,
        no_mllm=False,
        spec_decode="none",
        enable_mtp=False,
        force_spec_decode=False,
    )
    assert cli._serve_will_run_on_mllm_lane(args) is True

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, None),
    )
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: True,
    )
    assert cli._warn_vision_text_only_degrade(profile, args=args) is False
    assert capsys.readouterr().err == ""


def test_desktop_suppresses_resolved_text_only_degrade_warning(
    monkeypatch, capsys
) -> None:
    from rapid_mlx import cli
    from rapid_mlx.model_aliases import resolve_profile

    profile = resolve_profile("qwen3.5-4b-4bit")
    assert profile is not None
    monkeypatch.setattr(
        cli, "_alias_text_degrades_without_vision", lambda *_a, **_kw: True
    )
    monkeypatch.setattr(
        "rapid_mlx.runtime.optional_runtime._running_in_desktop_sidecar",
        lambda: True,
    )

    assert cli._warn_vision_text_only_degrade(profile) is False
    assert capsys.readouterr().err == ""


def test_pull_is_storage_only_and_never_emits_serve_failure(monkeypatch) -> None:
    from rapid_mlx import cli

    pulled: list[str] = []
    monkeypatch.setattr(
        cli, "_pull_repository", lambda args, **_kw: pulled.append(args.model)
    )
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda *_a, **_kw: pytest.fail("pull must not emit model_serve_failed"),
    )
    monkeypatch.setattr(
        "rapid_mlx.models.mllm.require_mlx_vlm_or_exit",
        lambda *_a, **_kw: pytest.fail("pull must not gate on a serving runtime"),
    )

    cli.pull_command(argparse.Namespace(model="qwen3.8-27b-4bit"))

    assert pulled == ["qwen3.8-27b-4bit"]


def test_repair_command_detector_failures_fall_back_to_python(monkeypatch) -> None:
    from rapid_mlx import _version_check
    from rapid_mlx.runtime import optional_runtime

    monkeypatch.setattr(
        _version_check,
        "detect_install_method",
        lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    assert " -m pip install " in optional_runtime.optional_extra_repair_command(
        "vision"
    )

    exc = optional_runtime.OptionalRuntimeMissing(
        extra="vision",
        install_hint="hint",
        detail="missing",
        status="absent",
    )
    monkeypatch.setattr(optional_runtime, "_running_in_desktop_sidecar", lambda: False)
    monkeypatch.setattr("rapid_mlx.telemetry.server_start.failed", lambda _stage: None)
    monkeypatch.setattr(
        "rapid_mlx.telemetry.model_events.emit_model_serve_failed",
        lambda *_a, **_kw: None,
    )
    with pytest.raises(SystemExit, match="2"):
        optional_runtime.handle_optional_runtime_missing(exc)


def test_mllm_and_video_lazy_import_failures_use_pinned_hints(monkeypatch) -> None:
    from rapid_mlx.models import mllm as mllm_module
    from rapid_mlx.models.mllm import MLXMultimodalLM
    from rapid_mlx.runtime.video_lane import VideoEngine, VideoRuntimeError
    from rapid_mlx.speculative.dflash import runtime as dflash_runtime
    from rapid_mlx.video.engine import (
        VideoBackendUnavailableError,
        VideoGenerationEngine,
    )

    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    monkeypatch.setattr(mllm_module, "_require_mlx_vlm", lambda *_a, **_kw: None)
    with pytest.raises(ImportError, match=r"rapid-mlx\[vision\]=="):
        MLXMultimodalLM("local/model").load()

    monkeypatch.setattr(dflash_runtime, "have_runtime", lambda: False)
    with pytest.raises(RuntimeError, match=r"rapid-mlx\[dflash\]=="):
        dflash_runtime.load_runtime("local/drafter")

    monkeypatch.setitem(sys.modules, "mlx_video", None)
    monkeypatch.setattr(
        "rapid_mlx.runtime.video_lane._resolve_ffmpeg", lambda: "ffmpeg"
    )
    with pytest.raises(VideoRuntimeError, match=r"rapid-mlx\[video\]=="):
        VideoEngine("ltx-2.3").generate(
            prompt="test",
            output_path=Path("unused.mp4"),
            width=64,
            height=64,
            num_frames=1,
            fps=1,
            seed=1,
            image=None,
        )

    engine = VideoGenerationEngine("cog/model")
    with pytest.raises(VideoBackendUnavailableError, match=r"rapid-mlx\[video\]=="):
        engine._load_sync()


@pytest.mark.asyncio
async def test_http_lazy_failures_hide_local_python_paths(
    monkeypatch, tmp_path
) -> None:
    from rapid_mlx import server
    from rapid_mlx.api.models import (
        AudioMusicRequest,
        AudioSpeechRequest,
        EmbeddingRequest,
    )
    from rapid_mlx.config import get_config
    from rapid_mlx.routes import audio, embeddings, video

    async def fail_async(*_args, **_kwargs):
        raise ImportError("missing backend")

    monkeypatch.setattr(audio, "run_to_completion", fail_async)
    monkeypatch.setattr("rapid_mlx.audio.probe.require_mlx_audio_tts", lambda: None)
    monkeypatch.setattr(
        "rapid_mlx.audio.probe.require_kokoro_runtime", lambda *_a, **_kw: None
    )
    with pytest.raises(HTTPException) as speech:
        await audio.create_speech(AudioSpeechRequest(input="hello"))
    assert "python -m pip" in speech.value.detail
    assert sys.executable not in speech.value.detail

    with pytest.raises(HTTPException) as music:
        await audio.create_music(AudioMusicRequest(input="march", seconds=1))
    assert "python -m pip" in music.value.detail
    assert sys.executable not in music.value.detail

    cfg = get_config()
    monkeypatch.setattr(cfg, "embedding_model_locked", "local/embed")
    monkeypatch.setattr(cfg, "embedding_engine", object())
    monkeypatch.setattr(
        server,
        "load_embedding_model",
        lambda *_a, **_kw: (_ for _ in ()).throw(ImportError("missing")),
    )
    embedding_stub = types.ModuleType("rapid_mlx.embedding")
    embedding_stub.EMBEDDINGS_EXTRA_HTTP_INSTALL_HINT = "install embeddings"
    embedding_stub.EmbeddingInputTooLongError = type(
        "EmbeddingInputTooLongError", (Exception,), {}
    )
    monkeypatch.setitem(sys.modules, "rapid_mlx.embedding", embedding_stub)
    raw = Request({"type": "http", "headers": []})
    with pytest.raises(HTTPException) as embedding:
        await embeddings.create_embeddings(
            EmbeddingRequest(model="local/embed", input="hello"), raw
        )
    assert "python -m pip" in embedding.value.detail
    assert sys.executable not in embedding.value.detail

    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(HTTPException) as reference:
        video._validate_reference_image(tmp_path / "input.png")
    assert "python -m pip" in reference.value.detail
    assert sys.executable not in reference.value.detail


def test_http_response_hint_constants_explicitly_disable_paths() -> None:
    """Response constants must opt out even if the helper default changes."""
    constants = {
        Path(__file__).parents[1] / "rapid_mlx" / "embedding.py": {
            "EMBEDDINGS_EXTRA_HTTP_INSTALL_HINT"
        },
        Path(__file__).parents[1] / "rapid_mlx" / "audio" / "probe.py": {
            "_KOKORO_EXTRA_HINT"
        },
    }
    missing: list[str] = []
    for path, names in constants.items():
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            targets = getattr(node, "targets", [])
            assigned = {target.id for target in targets if isinstance(target, ast.Name)}
            for name in assigned & names:
                expression = ast.get_source_segment(source, node) or ""
                if "include_paths=False" not in expression:
                    missing.append(f"{path}:{node.lineno}:{name}")

    assert missing == []
