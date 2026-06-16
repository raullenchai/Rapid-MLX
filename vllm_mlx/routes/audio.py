# SPDX-License-Identifier: Apache-2.0
"""Audio endpoints (STT/TTS)."""

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from starlette.responses import Response

from ..middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

# Security: cap audio upload size to prevent memory-exhaustion DoS.
# 25 MB matches OpenAI's Whisper API limit and is far above any reasonable
# transcription payload (~25 min of 16 kHz mono WAV).
MAX_AUDIO_UPLOAD_SIZE = 25 * 1024 * 1024
_AUDIO_READ_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks

# Audio engines (lazy loaded, module-level to persist across requests)
_stt_engine = None
_tts_engine = None


@router.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """Transcribe audio to text (OpenAI Whisper API compatible)."""
    global _stt_engine

    try:
        from ..audio.stt import STTEngine

        model_map = {
            "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
            "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "whisper-medium": "mlx-community/whisper-medium-mlx",
            "whisper-small": "mlx-community/whisper-small-mlx",
            "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
            "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
        }
        model_name = model_map.get(model, model)

        if _stt_engine is None or _stt_engine.model_name != model_name:
            _stt_engine = STTEngine(model_name)
            _stt_engine.load()

        # Reject oversized uploads early via Content-Length if the client
        # advertises it. This avoids reading any bytes for the obvious DoS case.
        content_length = file.size
        if content_length is not None and content_length > MAX_AUDIO_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio upload too large: {content_length} bytes "
                    f"(max {MAX_AUDIO_UPLOAD_SIZE} bytes)"
                ),
            )

        # Stream the upload to a temp file in chunks, enforcing the cap as we
        # go. This bounds memory use even if the client lies about
        # Content-Length or sends a chunked transfer encoding.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            total = 0
            while True:
                chunk = await file.read(_AUDIO_READ_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_AUDIO_UPLOAD_SIZE:
                    tmp.close()
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Audio upload too large: exceeds "
                            f"{MAX_AUDIO_UPLOAD_SIZE} bytes"
                        ),
                    )
                tmp.write(chunk)

        try:
            result = _stt_engine.transcribe(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

        if response_format == "text":
            return result.text

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except HTTPException:
        # Preserve our own status codes (e.g. 413 for oversized uploads)
        # instead of downgrading them to 500 via the catch-all below.
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
    """Generate speech from text (OpenAI TTS API compatible)."""
    global _tts_engine

    try:
        from ..audio.tts import TTSEngine

        model_map = {
            "kokoro": "mlx-community/Kokoro-82M-bf16",
            "kokoro-4bit": "mlx-community/Kokoro-82M-4bit",
            "chatterbox": "mlx-community/chatterbox-turbo-fp16",
            "chatterbox-4bit": "mlx-community/chatterbox-turbo-4bit",
            "vibevoice": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            "voxcpm": "mlx-community/VoxCPM1.5",
        }
        model_name = model_map.get(model, model)

        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model."""
    from ..audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}
