# SPDX-License-Identifier: Apache-2.0
"""Audio endpoints (STT/TTS)."""

import logging
import os
import re
import tempfile

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
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

# OpenAI-style STT model alias → MLX repo. Promoted to module scope so
# the route can validate the model BEFORE streaming the upload (F-165):
# unknown names previously rode the body through the upload cap, then
# collapsed into a generic 500 "could not open/decode file" once
# ``STTEngine.load`` failed deep inside mlx-audio. Mirror the
# ``/v1/chat/completions`` and ``/v1/responses`` contract: validate the
# model name first and surface 404 with a distinct error type.
STT_MODEL_ALIASES: dict[str, str] = {
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}

# F-210: model strings must canonicalize to either a bare alias name
# (matches an entry in ``STT_MODEL_ALIASES``) or a single-slash
# HuggingFace-style ``<org>/<repo>`` id. Anything else — multi-slash
# paths (``foo/bar/baz``), all-slash strings (``////``), control
# characters, leading/trailing slashes, etc. — bypasses the alias lookup
# and trips a downstream codec-open failure that surfaces as a generic
# 500 ``transcription_failed``. Reject these BEFORE attempting decode so
# the canonical 404 ``model_not_found_error`` fires instead.
#
# Allowed characters mirror HuggingFace's repo-id conventions
# (alphanumeric, underscore, dot, hyphen). ``+`` is intentionally NOT
# allowed — HF repo ids are restricted to ``[A-Za-z0-9._-]`` (see
# huggingface_hub.utils.validate_repo_id).
#
# Codex r3 BLOCKING: the *total* repo_id length cap is 96 chars (not
# per-component). The per-component bound stays at 96 too because the
# total bound already implies it. Anchor the regex with the 96-char
# overall cap (enforced as a separate ``len(model) <= 96`` check below
# so the regex itself stays cheap to read).
_STT_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?$")
_HF_REPO_ID_MAX_LEN = 96


def _is_valid_repo_component(comp: str) -> bool:
    """Codex r2 / r3 BLOCKING follow-up: mirror HF's structural rules.

    A bare regex character-class check accepts strings that HF itself
    rejects (e.g. ``.hidden``, ``repo..name``, components starting/
    ending with ``.`` or ``-``, or ``.git`` suffix). Those still
    crash inside ``STTEngine.load`` as a 500 because the HF resolver
    fails the same way for them. Enforce the structural rules HF
    documents (``huggingface_hub.utils.validate_repo_id``).

    Codex r3 BLOCKING: ``.ipynb`` is NOT a HF-rejected suffix, only
    ``.git`` is. Removed the over-eager ``.ipynb`` check.
    """
    if not comp:
        return False
    if comp.startswith((".", "-")) or comp.endswith((".", "-")):
        return False
    # ``..`` is a parent-directory traversal sentinel; HF rejects it
    # to keep repo ids resolvable as filesystem paths.
    if ".." in comp:
        return False
    # ``--`` is rejected by HF's repo-id validator as well.
    if "--" in comp:
        return False
    # Only ``.git`` is explicitly reserved by HF (codex r3 — ``.ipynb``
    # was an over-rejection on my part).
    if comp.endswith(".git"):
        return False
    return True


def _resolve_stt_model(model: str) -> str:
    """Resolve an OpenAI-style STT model alias to the MLX repo path.

    Returns the resolved repo path for known aliases or passes through
    ``mlx-community/...`` / ``<org>/...`` style repo specs verbatim.
    Raises a 404 ``HTTPException`` for everything else so unknown
    ``model`` form fields don't reach the ``STTEngine.load`` call and
    collapse into a generic 500 "could not open/decode file" (F-165).

    Pass-through is intentionally restrictive — any string with a ``/``
    is treated as a HuggingFace-style repo id. Bare names without a
    slash that aren't in ``STT_MODEL_ALIASES`` are rejected up front.
    """
    if not isinstance(model, str) or not model:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "`model` must be a non-empty string",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "model",
                }
            },
        )
    if model in STT_MODEL_ALIASES:
        return STT_MODEL_ALIASES[model]

    # F-210: path-shaped / malformed model ids (``foo/bar/baz``,
    # ``////``, leading/trailing slashes, control chars) used to slip
    # past the simple ``"/" in model`` heuristic, then crash inside
    # ``STTEngine.load`` as a generic 500 ``transcription_failed``.
    # Canonicalize these to the same 404 ``model_not_found_error`` the
    # bogus-alias path returns (F-167 / PR #735) by enforcing the
    # repo-id regex BEFORE attempting any codec open.
    #
    # codex r2: char-class alone isn't enough — HF also rejects
    # ``..``/``--``/``.hidden``/``trailing-dot.``/``repo.git`` shapes.
    # Apply the regex (cheap fast-path), then the 96-char total cap
    # (codex r3 BLOCKING — was per-component, HF's actual rule is
    # ``len(repo_id) <= 96``), then per-component structural rules.
    _regex_ok = bool(_STT_MODEL_NAME_RE.fullmatch(model))
    _length_ok = len(model) <= _HF_REPO_ID_MAX_LEN
    _components_ok = (
        _regex_ok
        and _length_ok
        and all(_is_valid_repo_component(c) for c in model.split("/"))
    )
    if not _components_ok:
        available = ", ".join(sorted(STT_MODEL_ALIASES.keys()))
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": (
                        f"The model `{model}` does not exist. "
                        f"Available STT aliases: {available}"
                    ),
                    "type": "model_not_found_error",
                    "code": "model_not_found",
                    "param": "model",
                }
            },
        )

    if "/" in model:
        # Looks like a HuggingFace repo id — let STTEngine attempt to
        # load it. ImportError / model-load errors still surface, but
        # the client is explicitly opting in by passing a repo path.
        return model
    available = ", ".join(sorted(STT_MODEL_ALIASES.keys()))
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": (
                    f"The model `{model}` does not exist. "
                    f"Available STT aliases: {available}"
                ),
                "type": "model_not_found_error",
                "code": "model_not_found",
                "param": "model",
            }
        },
    )


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

    Running at the ASGI layer lets us short-circuit the receive loop
    in TWO complementary ways:

    1. **Honest-``Content-Length`` fast path** — if the advertised
       length exceeds the cap, return 413 immediately. Zero ``receive``
       calls, zero bytes on the server.

    2. **Chunked / no-``Content-Length`` slow path** — wrap ``receive``
       so it tallies streamed body bytes and returns a synthetic
       ``http.disconnect`` once the cap is exceeded. The middleware
       then emits 413. Starlette's multipart parser sees the
       disconnect, stops spooling, and unwinds — the server still
       lands at most ``MAX_AUDIO_UPLOAD_SIZE + slack`` bytes on disk
       (the threshold at which we trigger the abort), not the
       multi-GB body the attacker tried to send.

    Path scope is intentionally narrow — only
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

        limit = MAX_AUDIO_UPLOAD_SIZE + _REQUEST_BODY_SLACK_BYTES

        # Honest-Content-Length fast path: reject before any receive call.
        advertised: int | None = None
        for raw_name, raw_value in scope.get("headers", ()):
            if raw_name.lower() == b"content-length":
                try:
                    advertised = int(raw_value.decode("latin-1"))
                except (UnicodeDecodeError, ValueError):
                    advertised = None
                break

        if advertised is not None and advertised > limit:
            await _send_413(
                send,
                (
                    f"Audio upload too large: request body {advertised} bytes "
                    f"(max {MAX_AUDIO_UPLOAD_SIZE} bytes per file)"
                ),
            )
            return

        # Streaming slow path: wrap receive so chunked/lying clients
        # cannot bypass the cap by omitting Content-Length. We tally
        # bytes as they cross the receive channel and abort the request
        # the moment the running total exceeds the cap. The trip flag
        # ensures we emit exactly one 413, even if Starlette keeps
        # reading after we signal disconnect.
        tripped = {"value": False}
        total = {"bytes": 0}

        async def bounded_receive():
            if tripped["value"]:
                # Once we've decided to abort, signal disconnect so the
                # parser unwinds cleanly. (Starlette's MultiPartParser
                # honors ``http.disconnect`` by stopping its read loop.)
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg.get("type") == "http.request":
                body_len = len(msg.get("body", b"") or b"")
                total["bytes"] += body_len
                if total["bytes"] > limit:
                    tripped["value"] = True
                    return {"type": "http.disconnect"}
            return msg

        # Wrap send so that if the downstream app tries to emit a
        # response after we've tripped, we substitute our 413 instead.
        # This handles both the case where Starlette aborts on
        # disconnect (no downstream response) and the case where it
        # raises mid-stream (caught by FastAPI and turned into a 500
        # that we'd otherwise mask).
        sent_413 = {"value": False}

        async def guarded_send(msg):
            if tripped["value"] and not sent_413["value"]:
                sent_413["value"] = True
                await _send_413(
                    send,
                    (
                        f"Audio upload too large: streamed body exceeded "
                        f"{MAX_AUDIO_UPLOAD_SIZE} bytes per file"
                    ),
                )
                return
            if sent_413["value"]:
                # Downstream tried to send after we already wrote 413;
                # drop the message to avoid double-write.
                return
            await send(msg)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except Exception:
            # If we tripped the cap, the downstream app aborted because
            # of the synthetic http.disconnect we injected — translate
            # that into the documented 413. Otherwise it's a real
            # error; re-raise so it surfaces normally.
            if not tripped["value"]:
                raise

        # Send a fallback 413 if nothing was emitted: this catches both
        # (a) the silent-drop-on-disconnect path (Starlette returns
        #     cleanly without sending a response after seeing disconnect)
        # (b) the exception path swallowed above.
        if tripped["value"] and not sent_413["value"]:
            sent_413["value"] = True
            await _send_413(
                send,
                (
                    f"Audio upload too large: streamed body exceeded "
                    f"{MAX_AUDIO_UPLOAD_SIZE} bytes per file"
                ),
            )


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
    # ``model``, ``language``, ``response_format`` are sent as multipart
    # form fields by OpenAI-compatible clients (the official Whisper
    # API puts them in the ``multipart/form-data`` body). Pre-F-165
    # this route declared them as plain ``str`` parameters, which
    # FastAPI then resolves as query parameters — meaning a curl /
    # OpenAI-SDK client putting them in the body silently fell back to
    # the default ``whisper-large-v3`` and never reached
    # ``_resolve_stt_model``. To repair the OpenAI contract WITHOUT
    # breaking any pre-existing internal caller that still passes
    # ``?model=...`` on the query string (codex-bundled review on the
    # F-165 PR), accept both sources and prefer the form field when
    # it is provided. ``...`` (Ellipsis) is *not* used as a default —
    # leaving both unset still resolves to ``whisper-large-v3``.
    model_form: str | None = Form(None, alias="model"),
    language_form: str | None = Form(None, alias="language"),
    response_format_form: str | None = Form(None, alias="response_format"),
    model_query: str | None = Query(None, alias="model"),
    language_query: str | None = Query(None, alias="language"),
    response_format_query: str | None = Query(None, alias="response_format"),
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

    # Form wins over query when both are present (form is the OpenAI
    # contract; query is the pre-F-165 internal contract we're keeping
    # for back-compat). Defaults match the original signature.
    model = (
        model_form
        if model_form is not None
        else (model_query if model_query is not None else "whisper-large-v3")
    )
    language = language_form if language_form is not None else language_query
    response_format = (
        response_format_form
        if response_format_form is not None
        else (response_format_query if response_format_query is not None else "json")
    )

    # Resolve / validate the requested model BEFORE draining the upload.
    # Previously every failure mode (unknown alias, missing mlx-audio,
    # bad audio bytes) collapsed into a 500 "could not open/decode
    # file" because ``STTEngine.load`` for a bogus name raised generic
    # ``Exception`` caught by the catch-all below. Move the alias check
    # up front so unknown ``model`` form fields fail fast with a 404
    # "model_not_found_error" and never trigger a model load (F-165).
    model_name = _resolve_stt_model(model)

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
        # Preserve our own status codes (e.g. 413 for oversized uploads,
        # 404 for unknown STT alias) instead of downgrading them to 500
        # via the catch-all below.
        raise
    except Exception as e:
        # Full traceback goes to the operator log; the client sees a
        # generic message so we don't leak filesystem paths or
        # mlx-audio internals (mirrors the global server handler).
        logger.exception("Transcription failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Audio transcription failed",
                    "type": "api_error",
                    "code": "transcription_failed",
                    "param": None,
                }
            },
        )
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
