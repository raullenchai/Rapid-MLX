"""Server dependency errors point to the running Rapid-MLX environment."""

import asyncio
import io
import shlex
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.requires_mlx


@pytest.mark.parametrize("alignment", [False, True])
def test_audio_route_import_error_hint(monkeypatch, alignment):
    from fastapi import HTTPException, UploadFile

    from rapid_mlx.connect import package_install_command
    from rapid_mlx.routes import audio

    async def missing_audio(_file, _target):
        raise ImportError("audio runtime unavailable")

    monkeypatch.setattr(audio, "_stream_upload_to_tempfile", missing_audio)
    upload = UploadFile(filename="sample.wav", file=io.BytesIO(b"unused"))

    if alignment:
        call = audio._run_alignment_request(
            upload, audio.DEFAULT_ALIGNER_ALIAS, "hello", None, "json"
        )
    else:
        call = audio._run_stt_request(
            upload, "whisper-large-v3", None, "json", "transcribe"
        )

    with pytest.raises(HTTPException) as error:
        asyncio.run(call)

    assert error.value.status_code == 503
    assert package_install_command("mlx-audio") in error.value.detail


def test_guided_engine_unavailable_hint():
    from rapid_mlx.connect import package_install_command
    from rapid_mlx.engine.batched import BatchedEngine

    engine = SimpleNamespace(supports_guided_generation=False)
    with pytest.raises(RuntimeError) as error:
        asyncio.run(BatchedEngine.generate_with_schema(engine, [], {}))

    assert package_install_command("rapid-mlx[guided]") in str(error.value)


@pytest.mark.parametrize(
    "method", ["_prepare_native_video_inputs", "_generate_native_video"]
)
def test_native_video_upgrade_hint(monkeypatch, method):
    from rapid_mlx.models.mllm import MLXMultimodalLM

    monkeypatch.setitem(sys.modules, "mlx_vlm.video_generate", None)
    with pytest.raises(ImportError) as error:
        getattr(MLXMultimodalLM, method)(SimpleNamespace(), [])

    assert f"{shlex.quote(sys.executable)} -m pip install --upgrade mlx-vlm" in str(
        error.value
    )
