# SPDX-License-Identifier: Apache-2.0
"""Audio endpoints (STT/TTS)."""

import logging
import os
import re
import tempfile

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile
from starlette.responses import PlainTextResponse, Response

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


#: Default STT alias used both when the ``model`` form/query field is
#: omitted and when the caller passes the OpenAI-canonical ``"default"``
#: placeholder. Mirrors the ``/v1/chat/completions`` rule that maps
#: ``"default"`` to the boot-time CLI model — STT has no boot-time
#: model bound, so the route default is the closest equivalent.
DEFAULT_STT_ALIAS = "whisper-large-v3"


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

    R-03: ``"default"`` is the OpenAI-spec placeholder LangChain /
    LlamaIndex / openai-python emit when the caller hasn't picked a
    specific model id. Map it to :data:`DEFAULT_STT_ALIAS` so drop-in
    OpenAI-SDK code works against ``/v1/audio/transcriptions`` —
    rejecting ``"default"`` here breaks every OpenAI tutorial without
    a manual ``model=`` argument.
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
    if model == "default":
        return STT_MODEL_ALIASES[DEFAULT_STT_ALIAS]
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


def _reject_non_whisper_for_translation(model: str) -> None:
    """Codex r6 NIT: ``/v1/audio/translations`` promises English output.

    Only Whisper engines honor ``task="translate"`` (mlx_audio's
    Parakeet path ignores the kwarg and emits source-language text).
    Accepting a non-Whisper alias here would silently break the
    translations contract. Inspect the alias (after resolution to its
    upstream id, if applicable) and reject anything that is
    recognizably non-Whisper with a 400 distinct from the 404
    ``model_not_found`` envelope (the model is real, it's just the
    wrong engine for this route).

    Routing order: this helper handles ONLY models we positively
    recognize as non-Whisper (Parakeet aliases, or HF ids with
    ``parakeet``/other-engine markers). Unknown bare strings fall
    through to ``_resolve_stt_model``'s 404 ``model_not_found_error``
    so the envelope matches transcriptions. Empty / non-string ``model``
    likewise falls through to the 400 envelope ``_resolve_stt_model``
    emits.
    """
    if not isinstance(model, str) or not model:
        return
    # Resolve aliases first so callers that pass ``parakeet`` (alias)
    # vs ``mlx-community/parakeet-tdt-0.6b-v2`` (HF id) get the same
    # verdict.
    resolved = STT_MODEL_ALIASES.get(model, model)
    resolved_lc = resolved.lower()
    # Whisper-shaped → accept; the engine honors task=translate.
    if "whisper" in resolved_lc:
        return
    # Bare alias not in STT_MODEL_ALIASES → leave for _resolve_stt_model
    # to 404 (so the envelope matches transcriptions' unknown-model
    # path). HF-shaped ids (containing a ``/``) are pass-through in
    # _resolve_stt_model, so we MUST classify here: a parakeet/voxtral/
    # other-engine HF id would otherwise reach the engine and produce
    # source-language output.
    if "/" not in model and model not in STT_MODEL_ALIASES:
        return
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    f"The model `{model}` cannot perform translation. "
                    "`/v1/audio/translations` requires a Whisper engine "
                    "(only Whisper honors `task=translate`). Use "
                    "`/v1/audio/transcriptions` for source-language "
                    "output, or pass a Whisper alias such as "
                    "`whisper-large-v3`."
                ),
                "type": "invalid_request_error",
                "code": "invalid_model_for_translation",
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

    _GUARDED_PATHS: tuple[str, ...] = (
        "/v1/audio/transcriptions",
        # F-K-TRANSLATIONS-MISSING: the translations route mirrors the
        # transcriptions multipart contract — multipart parsing happens
        # the same way, so the body cap must guard both paths. Without
        # this entry an attacker could send a 1 GB ``.wav`` to
        # ``/v1/audio/translations`` and exhaust the worker before the
        # streaming cap inside the handler kicks in.
        "/v1/audio/translations",
    )

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


# ---------------------------------------------------------------------------
# R6-H2: STT ``response_format`` — was silently ignored pre-fix.
#
# Pre-r6-C the route only branched on ``response_format == "text"`` and
# fell through to a JSON envelope for everything else. Clients passing
# ``srt`` / ``vtt`` / ``verbose_json`` got a JSON body back regardless,
# silently breaking the OpenAI contract. The fix:
#
#   1. Accept the request only if ``response_format`` is one of the
#      OpenAI-documented five values — anything else → 400 with the
#      OpenAI-shaped envelope (``invalid_request_error``,
#      ``param="response_format"``). Saves the engine load.
#   2. After transcription, branch on the validated value and produce
#      the matching Content-Type / body — ``text/plain``, ``text/srt``,
#      ``text/vtt``, or ``application/json`` (default + verbose_json).
#
# SRT / VTT formatters work from ``result.segments`` when the STT
# engine reports them. If a backend doesn't (Parakeet today), the
# formatter falls back to a single cue spanning ``result.duration``
# so the client still gets a syntactically valid subtitle file.
# ---------------------------------------------------------------------------

#: OpenAI's documented set — keep the literal in sync with
#: ``test_stt_response_format.py`` so a drift here trips CI before
#: hitting prod.
_STT_RESPONSE_FORMATS: frozenset[str] = frozenset(
    ("json", "text", "srt", "vtt", "verbose_json")
)

#: Default when the caller omits the field. Mirrors OpenAI's behaviour.
_STT_DEFAULT_RESPONSE_FORMAT = "json"


def _validate_response_format(response_format: str | None) -> str:
    """Return the normalised response_format or raise 400.

    A ``None`` / empty value resolves to the documented default
    (``"json"``). Anything outside the OpenAI five-value set raises
    a 400 with the same OpenAI-shape envelope the rest of the route
    uses for ``invalid_request_error``. Performed BEFORE the upload
    drains so a typo (``"jsno"``) fails cheaply without touching the
    engine or temp file.
    """
    if response_format is None or response_format == "":
        return _STT_DEFAULT_RESPONSE_FORMAT
    if response_format not in _STT_RESPONSE_FORMATS:
        available = ", ".join(sorted(_STT_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"`response_format` must be one of: {available}; "
                        f"got {response_format!r}."
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "response_format",
                }
            },
        )
    return response_format


def _format_srt_timestamp(seconds: float) -> str:
    """Format a float second offset as the SRT timestamp ``HH:MM:SS,mmm``.

    SRT mandates comma as the millisecond separator (vs. VTT's dot).
    Clamp negative inputs to zero — a defective backend reporting
    negative timestamps would otherwise emit a malformed cue.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:  # rounding overflow
        millis = 0
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    """Format as the WebVTT timestamp ``HH:MM:SS.mmm``.

    Differs from SRT only in the millisecond separator (``.`` vs ``,``)
    and the lack of trailing index — share the bulk of the formatting
    with :func:`_format_srt_timestamp` to keep the two outputs in sync
    when one is patched.
    """
    return _format_srt_timestamp(seconds).replace(",", ".")


def _iter_segments_for_subtitles(result) -> list[tuple[float, float, str]]:
    """Normalise the engine's ``segments`` list into ``(start, end, text)``.

    Whisper-style engines report dicts with ``start``/``end``/``text``
    keys; future backends may report a dataclass. Walk both shapes and
    fall back to a single cue covering ``result.duration`` (or 0..0)
    when no segments are present so the SRT/VTT body is still valid.
    """
    segments = getattr(result, "segments", None) or []
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        if isinstance(seg, dict):
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            text = str(seg.get("text", "") or "").strip()
        else:
            start = float(getattr(seg, "start", 0.0) or 0.0)
            end = float(getattr(seg, "end", start) or start)
            text = str(getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        out.append((start, end, text))
    if not out:
        duration = float(getattr(result, "duration", 0.0) or 0.0)
        text = str(getattr(result, "text", "") or "").strip()
        out.append((0.0, duration, text))
    return out


def _build_srt_body(result) -> str:
    """Render a SubRip Subtitle (.srt) body from a transcription result."""
    cues = _iter_segments_for_subtitles(result)
    lines: list[str] = []
    for idx, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _build_vtt_body(result) -> str:
    """Render a WebVTT (.vtt) body from a transcription result.

    WebVTT starts with the mandatory ``WEBVTT`` header line followed
    by a blank line — clients that strip it (or some browsers that
    only support full WebVTT) will refuse to render the cues
    otherwise.
    """
    cues = _iter_segments_for_subtitles(result)
    lines: list[str] = ["WEBVTT", ""]
    for start, end, text in cues:
        lines.append(f"{_format_vtt_timestamp(start)} --> {_format_vtt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _build_verbose_json_body(result) -> dict:
    """Render the ``verbose_json`` body — text + language + duration + segments.

    Mirrors OpenAI's documented field set. ``segments`` is normalised
    to a list of dicts with the canonical key names; the engine's
    raw shape is whatever ``stt`` chose to expose (whisper-mlx ships
    dicts; future backends might ship objects).
    """
    cues = _iter_segments_for_subtitles(result)
    return {
        "task": "transcribe",
        "text": getattr(result, "text", ""),
        "language": getattr(result, "language", None),
        "duration": getattr(result, "duration", None),
        "segments": [
            {
                "id": idx,
                "start": start,
                "end": end,
                "text": text,
            }
            for idx, (start, end, text) in enumerate(cues)
        ],
    }


def _format_stt_response(result, response_format: str, task: str):
    """Branch on the validated ``response_format`` and produce a body.

    Centralised so the transcription and translation routes pick the
    same shape for the same value — a future change to one path
    automatically lands on the other.
    """
    if response_format == "text":
        return PlainTextResponse(getattr(result, "text", "") or "")
    if response_format == "srt":
        return PlainTextResponse(_build_srt_body(result), media_type="text/srt")
    if response_format == "vtt":
        return PlainTextResponse(_build_vtt_body(result), media_type="text/vtt")
    if response_format == "verbose_json":
        body = _build_verbose_json_body(result)
        # The verbose envelope carries an explicit ``task`` field —
        # translations should advertise themselves correctly even
        # when ``transcribe`` is the engine default.
        body["task"] = task
        return body
    # Default "json" envelope — keep the historical three fields so
    # any pre-fix client that already parses ``text``/``language``/
    # ``duration`` doesn't notice the upgrade.
    return {
        "text": getattr(result, "text", ""),
        "language": getattr(result, "language", None),
        "duration": getattr(result, "duration", None),
    }


# ---------------------------------------------------------------------------
# R6-H3: STT corrupted-upload envelope.
#
# Pre-fix every exception from the engine (including ffmpeg/librosa
# decode failures on garbage bytes) fell into the catch-all that
# returned 500 ``transcription_failed``. A corrupted upload is a
# CLIENT error — the OpenAI contract maps it to 400
# ``invalid_request_error`` with ``param="file"``. The fix introspects
# the exception (and its message) for the decode-failure shapes and
# re-maps them to the documented envelope.
# ---------------------------------------------------------------------------

#: Substrings that identify a decode/codec failure regardless of the
#: underlying library (mlx_audio surfaces multiple shapes — librosa,
#: soundfile, ffmpeg). Keep this list narrow so we don't accidentally
#: relabel a legitimate engine bug as a client error.
_DECODE_ERROR_HINTS: tuple[str, ...] = (
    "could not open",
    "could not decode",
    "format not recognised",
    "format not recognized",
    "unknown format",
    "no such format",
    "soundfile",
    "libsndfile",
    "audioread",
    "invalid audio",
    "unsupported audio",
    "could not load audio",
    "ffmpeg",
    "header is truncated",
    "error opening",
)


def _is_decode_error(exc: Exception) -> bool:
    """Return True iff ``exc`` looks like an audio-decode failure.

    Matches both the exception's class name (``DecodeError``,
    ``LibsndfileError``, ``SoundFileError``) and the message text
    against a curated hint list. Type-based matching is the strong
    signal — we use the substring check as a fallback because
    ``mlx_audio`` chains raw ``ValueError`` / ``RuntimeError`` for
    decode failures in some code paths.
    """
    if not isinstance(exc, Exception):
        return False
    cls_name = type(exc).__name__.lower()
    if any(tok in cls_name for tok in ("decode", "sndfile", "soundfile", "codec")):
        return True
    msg = str(exc).lower()
    return any(hint in msg for hint in _DECODE_ERROR_HINTS)


def _audio_decode_error_envelope(exc: Exception) -> HTTPException:
    """Build the OpenAI-shape 400 envelope for a decode failure.

    Keeps the original exception message in the envelope so the client
    can surface the actual decode reason ("Format not recognised" /
    "Header is truncated") without us leaking server paths or
    mlx_audio internals — we only include ``str(exc)``, never the
    traceback.
    """
    reason = str(exc).strip() or type(exc).__name__
    return HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": f"could not decode audio file: {reason}",
                "type": "invalid_request_error",
                "code": "invalid_audio_file",
                "param": "file",
            }
        },
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


async def _run_stt_request(
    file: UploadFile,
    model: str,
    language: str | None,
    response_format: str,
    task: str,
):
    """Shared STT pipeline used by both ``/v1/audio/transcriptions`` and
    ``/v1/audio/translations``.

    The two OpenAI endpoints have IDENTICAL multipart contracts — the
    only difference is the destination language: transcriptions keeps
    the source language (``task="transcribe"``), translations forces
    English output (``task="translate"``). Factoring the body into a
    helper keeps the size/probe/resolve/cleanup/envelope wiring in one
    place so a future fix to either path lands on both.

    F-K-TRANSLATIONS-MISSING: previously only the transcriptions route
    existed; ``/v1/audio/translations`` 404'd. Mirror the route via
    this helper and pass ``task="translate"`` so Whisper emits English.

    NOTE: callers are responsible for invoking ``require_mlx_audio_stt()``
    BEFORE this helper so the F-D05 source-grep regression guard
    (``test_audio_probe_consistency.py``) sees the probe call inside
    each route function body. Defense-in-depth: the helper also gates
    on the model alias resolver, which raises 4xx before any model
    load, so a missing probe call still fails closed — just not with
    the uniform 503 envelope the probe emits.
    """
    global _stt_engine

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

        result = _stt_engine.transcribe(tmp_path, language=language, task=task)

        # R6-H2: branch on the validated ``response_format`` so callers
        # that requested ``srt`` / ``vtt`` / ``verbose_json`` actually
        # get those shapes. Pre-fix only ``text`` had a non-JSON path;
        # everything else fell through to the JSON envelope.
        return _format_stt_response(result, response_format, task=task)

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
        # R6-H3: corrupted upload (raw garbage bytes, wrong codec,
        # truncated header) is a CLIENT error — surface a 400 with
        # the OpenAI-shape envelope so callers don't have to retry
        # the request on a 500 they're never going to recover from.
        # Decode errors must be detected BEFORE the generic 500
        # catch-all logs the trace as an unexpected backend bug.
        if _is_decode_error(e):
            logger.info("STT %s rejected corrupted upload: %s", task, e)
            raise _audio_decode_error_envelope(e)
        # Full traceback goes to the operator log; the client sees a
        # generic message so we don't leak filesystem paths or
        # mlx-audio internals (mirrors the global server handler).
        logger.exception("STT %s failed: %s", task, e)
        # F-K-WHISPER-500: when mlx_audio reports a structural
        # backend defect (missing processor wiring, broken model state)
        # surface 503 ``backend_unavailable`` instead of the generic
        # 500 ``transcription_failed``. The former tells clients the
        # backend is unhealthy and they should fall back to another
        # model; the latter implies the audio file was the problem.
        msg = str(e)
        if "Processor not found" in msg or "_processor" in msg:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": (
                            "Whisper backend is unhealthy: the configured "
                            f"model `{model_name}` could not load a "
                            "tokenizer/processor. Try `parakeet` or "
                            "`parakeet-v3` for the STT lane on this install, "
                            "or pin a Whisper variant whose mlx-community "
                            "repo ships processor files."
                        ),
                        "type": "backend_unavailable_error",
                        "code": "backend_unavailable",
                        "param": "model",
                    }
                },
            )
        code = "transcription_failed" if task == "transcribe" else "translation_failed"
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": (
                        "Audio transcription failed"
                        if task == "transcribe"
                        else "Audio translation failed"
                    ),
                    "type": "api_error",
                    "code": code,
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

    # R6-H2: reject unknown ``response_format`` values up front with a
    # 400 envelope so a typo (``"jsno"``) or unsupported value
    # (``"yaml"``) fails BEFORE we drain the upload, load the engine,
    # or run inference. Pre-fix the value silently fell through to the
    # JSON branch, masking client-side bugs as "STT lies about
    # response_format".
    response_format = _validate_response_format(response_format)

    # F-D05: STT-lane audio dep probe — same envelope as the TTS
    # lane shares. Fires BEFORE we spool any upload bytes so a broken
    # ``mlx_audio.stt`` install rejects cheaply (no temp file, no
    # read loop). Probed here (not inside ``_run_stt_request``) so
    # the source-grep guard in test_audio_probe_consistency.py sees
    # the ``require_mlx_audio`` call directly inside the route body.
    from ..audio.probe import require_mlx_audio_stt

    require_mlx_audio_stt()

    return await _run_stt_request(
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        task="transcribe",
    )


@router.post("/v1/audio/translations", dependencies=[Depends(verify_api_key)])
async def create_translation(
    file: UploadFile,
    # OpenAI's translations endpoint mirrors transcriptions but
    # OMITS the ``language`` field — the destination language is
    # always English. We still accept it on the form for clients
    # that share request-shaping code with transcriptions; it gets
    # ignored downstream because Whisper's ``translate`` task
    # always emits English regardless of the source-language hint.
    # F-K-TRANSLATIONS-MISSING.
    model_form: str | None = Form(None, alias="model"),
    response_format_form: str | None = Form(None, alias="response_format"),
    model_query: str | None = Query(None, alias="model"),
    response_format_query: str | None = Query(None, alias="response_format"),
):
    """Translate audio to English (OpenAI Whisper API compatible).

    F-K-TRANSLATIONS-MISSING: pre-fix this route was absent and
    OpenAI-SDK clients calling ``client.audio.translations.create(...)``
    saw a 404. Spec parity requires both transcriptions (source-
    language output) and translations (always-English output) — the
    only wire difference is that translations omits ``language`` from
    the form body. The underlying mlx-audio path is identical: Whisper
    accepts ``task="translate"`` which forces English emission.

    Codex r6 NIT: non-Whisper engines (Parakeet, future Voxtral, etc.)
    ignore the ``task="translate"`` flag, so accepting them here would
    silently return source-language audio under a contract that
    promises English. Reject non-Whisper aliases at the route boundary
    with a 400 ``invalid_model_for_translation`` so callers get a
    distinct, actionable error instead of mislabeled output.
    """
    model = (
        model_form
        if model_form is not None
        else (model_query if model_query is not None else "whisper-large-v3")
    )
    response_format = (
        response_format_form
        if response_format_form is not None
        else (response_format_query if response_format_query is not None else "json")
    )

    # R6-H2: validate ``response_format`` BEFORE the model-eligibility
    # check so a typo / unsupported value fails cheaply with the same
    # 400 envelope the transcriptions route uses. Mirrors the helper
    # used on the transcriptions route — the two paths share the
    # OpenAI five-value contract.
    response_format = _validate_response_format(response_format)

    # Codex r6 NIT: the translations contract guarantees English
    # output. Only Whisper engines honor ``task="translate"``; any
    # other STT alias would silently fall through to source-language
    # output. Reject up front with a clear envelope so callers know to
    # switch models (or fall back to /v1/audio/transcriptions if they
    # only need source-language text). Performed BEFORE the body probe
    # so a clearly-misrouted Parakeet request fails without touching
    # mlx_audio at all.
    _reject_non_whisper_for_translation(model)

    # F-D05: STT-lane audio dep probe (kept inside the route body so
    # the source-grep regression guard in
    # test_audio_probe_consistency.py picks it up — both the
    # transcriptions and translations routes share the STT lane).
    from ..audio.probe import require_mlx_audio_stt

    require_mlx_audio_stt()

    return await _run_stt_request(
        file=file,
        model=model,
        language=None,
        response_format=response_format,
        task="translate",
    )


@router.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    model: str = "kokoro",
    input: str = "",
    voice: str = "af_heart",
    speed: float = 1.0,
    response_format: str = "wav",
):
    """Generate speech from text (OpenAI TTS API compatible).

    F-D05: probe ``mlx_audio`` availability through the shared
    :func:`vllm_mlx.audio.probe.require_mlx_audio` helper so this
    route's 503 envelope matches ``/v1/audio/voices`` and
    ``/v1/audio/transcriptions``. Pre-fix each audio route hand-rolled
    its own check; the voices route never probed at all, the speech
    route only saw the ImportError, and operators couldn't tell from
    one endpoint that the others were also broken.
    """
    global _tts_engine

    # TTS-lane audio dep probe (F-D05 + codex r3 BLOCKING). Fires
    # BEFORE the lazy TTSEngine import — if the TTS sub-module of
    # mlx_audio is missing or broken at runtime, both ``/v1/audio/
    # speech`` and ``/v1/audio/voices`` return the SAME 503 envelope
    # with the actual failure reason embedded. A torn STT lane does
    # NOT 503 this route — the lane separation closes the codex r3
    # regression where a broken STT install would mask TTS-usable
    # installs as fully broken.
    from ..audio.probe import require_kokoro_runtime, require_mlx_audio_tts

    require_mlx_audio_tts()

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
        # R-03: ``"default"`` is the OpenAI-spec placeholder LangChain /
        # LlamaIndex / openai-python emit when the caller hasn't picked
        # a specific model id. Map it to the same alias as omitting the
        # field entirely (``kokoro`` — the route signature's default
        # value) so drop-in OpenAI-SDK code works without a manual
        # ``model=`` override.
        if model == "default":
            model = "kokoro"
        model_name = model_map.get(model, model)

        # F-K-KOKORO-MISAKI: Kokoro pulls ``misaki`` lazily inside
        # ``KokoroPipeline``; the TTS-lane probe above can't catch
        # the missing extra because ``mlx_audio.tts.generate``
        # imports cleanly without it. Gate the missing-extra at
        # this boundary so the request 503s with a clean envelope
        # BEFORE any weight load (mlx-community/Kokoro-82M-bf16 is
        # ~300 MB) or pipeline construction kicks off.
        if "kokoro" in model_name.lower():
            require_kokoro_runtime()

        if _tts_engine is None or _tts_engine.model_name != model_name:
            _tts_engine = TTSEngine(model_name)
            _tts_engine.load()

        audio = _tts_engine.generate(input, voice=voice, speed=speed)
        audio_bytes = _tts_engine.to_bytes(audio, format=response_format)

        content_type = (
            "audio/wav" if response_format == "wav" else f"audio/{response_format}"
        )
        return Response(content=audio_bytes, media_type=content_type)

    except HTTPException:
        # Preserve probe-emitted 503 (and any other explicit status)
        # rather than collapsing into the generic 500 catch-all below.
        raise
    except ImportError as e:
        # Defense in depth: if a future refactor introduces an import
        # path the probe doesn't cover (or the cached verdict is stale
        # in some edge case), still surface a meaningful 503 instead
        # of leaking a stack trace through the catch-all 500.
        raise HTTPException(
            status_code=503,
            detail=(
                f"mlx-audio import failed at runtime: {e}. "
                "Install with: pip install 'rapid-mlx[audio]'"
            ),
        )
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str = "kokoro"):
    """List available voices for a TTS model.

    F-D05: gates on the same :func:`require_mlx_audio` probe that
    ``/v1/audio/speech`` uses so callers can't get a 200 with a
    voice list while the very next ``speech`` call 503s on the same
    server. Pre-fix the voices route returned a static list without
    touching ``mlx_audio`` at all, so it advertised TTS-capability
    even when the engine wouldn't load.
    """
    # Probe FIRST, then import anything that depends on mlx_audio
    # transitively. Pre-fix this ordering wasn't a problem because
    # ``vllm_mlx.audio.tts`` doesn't import ``mlx_audio`` at module
    # level — but pinning the order in the route handler means a
    # future refactor that hoists an ``import mlx_audio`` to the top
    # of ``audio/tts.py`` (e.g. for type hints) can't accidentally
    # bypass the shared 503 envelope by failing at the route's import
    # statement before the probe even runs. Codex r1 BLOCKING on
    # PR #804. Codex r3 follow-up: probe the TTS lane SPECIFICALLY so
    # a torn STT install doesn't 503 voice listing.
    from ..audio.probe import require_mlx_audio_tts

    require_mlx_audio_tts()

    from ..audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

    if "kokoro" in model.lower():
        return {"voices": KOKORO_VOICES}
    elif "chatterbox" in model.lower():
        return {"voices": CHATTERBOX_VOICES}
    else:
        return {"voices": ["default"]}
