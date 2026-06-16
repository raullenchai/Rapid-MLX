# SPDX-License-Identifier: Apache-2.0
"""Audio endpoints (STT/TTS)."""

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
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


def _reject_oversized_audio(advertised: int | None) -> None:
    """Reject obvious DoS uploads before doing any work.

    `advertised` is the client-supplied Content-Length (or `UploadFile.size`,
    which Starlette derives from it). If the client truthfully advertises a
    payload above the cap, fail with 413 *before* touching the engine or
    reading any bytes. Clients who omit/lie about Content-Length are caught
    later by the streaming counter in `_stream_upload_to_tempfile`.
    """
    if advertised is None:
        return
    if advertised > MAX_AUDIO_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio upload too large: {advertised} bytes "
                f"(max {MAX_AUDIO_UPLOAD_SIZE} bytes)"
            ),
        )


async def _stream_upload_to_tempfile(file: UploadFile, tmp) -> None:
    """Copy `file` into the open temp-file `tmp`, enforcing the size cap as
    we go. Raises HTTPException(413) the moment the cap is exceeded.

    Streaming in fixed-size chunks bounds peak memory to one chunk regardless
    of how much the client sends — defending against chunked-transfer clients
    that omit Content-Length entirely.
    """
    total = 0
    while True:
        chunk = await file.read(_AUDIO_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_AUDIO_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio upload too large: exceeds {MAX_AUDIO_UPLOAD_SIZE} bytes"
                ),
            )
        tmp.write(chunk)


@router.post("/v1/audio/transcriptions", dependencies=[Depends(verify_api_key)])
async def create_transcription(
    request: Request,
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """Transcribe audio to text (OpenAI Whisper API compatible)."""
    global _stt_engine

    # SECURITY: Enforce the upload-size cap *before* any expensive work
    # (engine import, model load, multipart materialization). We check the
    # raw Content-Length header straight from the request scope so that a
    # malicious client cannot trigger model loading just by advertising a
    # huge payload.
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is not None:
        try:
            _reject_oversized_audio(int(raw_content_length))
        except ValueError:
            # Malformed Content-Length — treat as unknown and rely on the
            # streaming cap below.
            pass
    # Belt-and-suspenders: Starlette's UploadFile may know the part size
    # even when the outer Content-Length is missing (e.g. multipart-only
    # length). Reject early if so.
    _reject_oversized_audio(file.size)

    tmp_path: str | None = None
    try:
        # SECURITY: Stream the upload to a bounded temp file *before* doing
        # anything expensive. Even a client that lies about / omits
        # Content-Length cannot force model load or import — they will hit
        # the streaming cap inside _stream_upload_to_tempfile() and get a
        # 413 long before the STTEngine block below runs.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            await _stream_upload_to_tempfile(file, tmp)

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

        result = _stt_engine.transcribe(tmp_path, language=language)

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
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            except OSError as cleanup_err:
                logger.warning(
                    "Failed to unlink temp audio file %s: %s", tmp_path, cleanup_err
                )


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
