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

# Security: cap audio upload size to prevent memory-/disk-exhaustion DoS.
# 25 MB matches OpenAI's Whisper API limit and is far above any reasonable
# transcription payload (~25 min of 16 kHz mono WAV). Multipart overhead
# (boundary, form fields) adds a few hundred bytes; we allow one MB of slack
# so a truthful 25 MB audio file isn't rejected at the request-level guard.
MAX_AUDIO_UPLOAD_SIZE = 25 * 1024 * 1024
_REQUEST_BODY_SLACK_BYTES = 1024 * 1024  # 1 MB headroom for multipart overhead
_AUDIO_READ_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks

# Audio engines (lazy loaded, module-level to persist across requests)
_stt_engine = None
_tts_engine = None


class AudioBodyLimitMiddleware:
    """ASGI middleware that bounds the request body of audio-upload
    routes BEFORE Starlette's multipart parser can spool it.

    Why ASGI middleware and not a FastAPI ``Depends``: when the route
    handler signature includes ``file: UploadFile``, Starlette's
    ``MultiPartParser`` runs as part of parameter resolution and reads
    the entire request body off the ``receive`` channel before any
    ``Depends`` callable is invoked. A ``Depends`` that inspects
    ``Content-Length`` therefore fires *after* the body has already been
    drained and spooled to ``SpooledTemporaryFile`` on disk —
    confirmed empirically with an ASGI ``receive`` probe.

    Running at the ASGI layer lets us short-circuit the receive loop:

    1. Honest-``Content-Length`` clients above the cap → 413 returned
       immediately with zero ``receive`` calls, so the body never lands
       on disk.

    2. Chunked / no-``Content-Length`` clients still flow through to
       the handler, where ``_stream_upload_to_tempfile`` enforces the
       per-file cap by counting bytes as it writes our own temp file.

    Scope is matched by path so unrelated routes pay zero overhead.
    The path set is intentionally narrow — only
    ``/v1/audio/transcriptions`` uploads a file; ``/v1/audio/speech``
    and ``/v1/audio/voices`` have small JSON bodies bounded by other
    means.
    """

    _GUARDED_PATHS: tuple[str, ...] = ("/v1/audio/transcriptions",)

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        if scope.get("path") not in self._GUARDED_PATHS:
            return await self.app(scope, receive, send)

        # Walk raw ASGI headers — case-insensitive byte keys.
        advertised: int | None = None
        for raw_name, raw_value in scope.get("headers", ()):
            if raw_name.lower() == b"content-length":
                try:
                    advertised = int(raw_value.decode("latin-1"))
                except (UnicodeDecodeError, ValueError):
                    advertised = None
                break

        limit = MAX_AUDIO_UPLOAD_SIZE + _REQUEST_BODY_SLACK_BYTES
        if advertised is not None and advertised > limit:
            await _send_413(
                send,
                (
                    f"Audio upload too large: request body {advertised} bytes "
                    f"(max {MAX_AUDIO_UPLOAD_SIZE} bytes per file)"
                ),
            )
            return

        await self.app(scope, receive, send)


async def _send_413(send, detail: str) -> None:
    """Emit a JSON 413 response from inside ASGI middleware.

    Hand-rolling the response (rather than raising ``HTTPException``)
    keeps the rejection self-contained inside the middleware — no
    FastAPI exception handlers or dependency machinery have to run, so
    the body is never read from ``receive``."""
    import json as _json

    body = _json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def install_audio_body_limit_middleware(app) -> None:
    """Attach :class:`AudioBodyLimitMiddleware` to an ``app``.

    Centralised so ``vllm_mlx.server`` and tests register the guard
    through one entry point — keeps the wiring discoverable from this
    module instead of buried in app-construction code."""
    app.add_middleware(AudioBodyLimitMiddleware)


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
    file: UploadFile,
    model: str = "whisper-large-v3",
    language: str | None = None,
    response_format: str = "json",
):
    """Transcribe audio to text (OpenAI Whisper API compatible).

    Two-layer size guard (defense in depth):

    1. :class:`AudioBodyLimitMiddleware` runs at the ASGI layer and
       rejects requests whose ``Content-Length`` exceeds the cap
       BEFORE Starlette's multipart parser drains the receive channel.
       Honest large uploads die there with zero disk I/O and no
       handler invocation.

    2. ``_stream_upload_to_tempfile`` (below) enforces the exact per-
       file cap while copying chunks into our own temp file. Catches
       chunked-transfer / no-``Content-Length`` clients that lied at
       layer 1: even if Starlette already spooled the body to its own
       ``SpooledTemporaryFile``, we refuse to copy more than the cap
       into ours and abort early before any STT engine import /
       ``.load()`` call happens.

    The 25 MB ceiling matches OpenAI's Whisper API and bounds the
    worst-case STT inference cost.
    """
    global _stt_engine

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
