# SPDX-License-Identifier: Apache-2.0
"""Audio endpoints (STT/TTS)."""

import asyncio
import base64
import binascii
import io
import logging
import math
import os
import re
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, UploadFile
from starlette.responses import PlainTextResponse, Response

from ..api.models import AudioMusicRequest, AudioSpeechRequest
from ..middleware.auth import verify_api_key
from ._async_utils import run_to_completion

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Task #292: conditional audio-route registration.
#
# Pre-fix the router was unconditionally ``include_router``'d on every
# ``rapid-mlx serve <text-only-model>`` install (Bo R13/R14 fuzz wave).
# Hitting ``/v1/audio/transcriptions`` on a text-only server then either
# 500'd (no audio engine loaded → engine.load() crashes inside the
# handler) or returned a misleading ``model_not_found`` envelope — the
# server appeared to advertise capabilities it didn't have.
#
# The fix splits route registration into a deferred ``register_audio_routes``
# helper. ``vllm_mlx.server`` calls it only when the loaded model is
# audio-capable (resolved through :mod:`vllm_mlx.audio.registry`) OR the
# operator passed ``--enable-audio``. On text-only servers the router is
# never attached, so FastAPI's stock 404 fires for the audio paths — the
# customer-visible outcome the dogfood report asked for.
#
# Two flags drive the gate:
#
# * ``ServerConfig.enable_audio_lane`` — operator opt-in via the
#   ``--enable-audio`` CLI flag.
# * The loaded model alias / HF id matches an entry in
#   :mod:`vllm_mlx.audio.registry` — every audio-mode boot path (Bo
#   R10-C1 ``_serve_audio_mode``) already resolves through the registry,
#   so the same predicate fires here without a separate config knob.
#
# ``AudioBodyLimitMiddleware`` (25 MB multipart cap) stays installed
# unconditionally on the app — it early-returns for paths outside
# ``_GUARDED_PATHS`` so the per-request cost on text routes is a single
# tuple membership test. Keeping it install-time avoids the Starlette
# warning that fires on ``add_middleware`` after the middleware stack
# has been built.
# ---------------------------------------------------------------------------


def audio_routes_should_register(
    model_name: str | None,
    model_alias: str | None,
    enable_audio_lane: bool,
) -> bool:
    """Return True iff the audio router should be attached to the app.

    Truthy when any of:

    * ``enable_audio_lane`` is set (operator opt-in via ``--enable-audio``).
    * The loaded ``model_name`` resolves to a registered audio alias
      (per :func:`vllm_mlx.audio.registry.is_audio_name`).
    * The loaded ``model_alias`` resolves to a registered audio alias —
      covers ``rapid-mlx serve kokoro --served-model-name foo`` where
      ``model_name`` is the served alias and ``model_alias`` is the
      short-form audio alias.

    All other cases → False. Used by ``vllm_mlx.server.register_audio_routes``
    to decide whether to call ``app.include_router`` at app boot time.

    The registry lookup is intentionally tolerant: any failure
    (``[audio]`` extra not installed, malformed JSON) falls through to
    False so a torn registry can't accidentally re-enable the routes on
    a text-only server. A torn registry on an actual audio-server boot
    is already caught earlier by the registry's own loaders.
    """
    if enable_audio_lane:
        return True
    try:
        from ..audio.registry import is_audio_name
    except Exception:  # noqa: BLE001
        return False
    try:
        if model_name and is_audio_name(model_name):
            return True
        if model_alias and is_audio_name(model_alias):
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


# App-local sentinel attribute marking that
# :func:`register_audio_routes` has already attached its router. Codex
# r0 NIT: a route-table substring check (``"/v1/audio/"`` prefix) would
# false-positive on apps that mounted a CUSTOM ``/v1/audio/*`` route
# alongside our router — e.g. an operator-owned ``/v1/audio/health``
# probe added before the gate fires would silently block the helper from
# registering the canonical handlers. Stamping a dedicated attribute on
# the FastAPI app instead lets the helper key off its OWN prior call
# rather than any string-shaped collision.
_AUDIO_REGISTRATION_SENTINEL = "_rapid_mlx_audio_routes_registered"


def register_audio_routes(app) -> bool:
    """Idempotently attach the audio router to ``app``.

    Returns True if the router was attached on this call, False if it
    was already attached. The idempotency check looks at the app-local
    sentinel :data:`_AUDIO_REGISTRATION_SENTINEL` that this function
    stamps after a successful registration. We don't scan the route
    table for a ``/v1/audio/`` prefix because a custom operator-added
    audio sub-route (e.g. a private ``/v1/audio/health`` probe added
    before this helper runs) would false-positive and silently block
    the canonical handlers from mounting — codex r0 NIT.

    The 25 MB multipart cap (:func:`install_audio_body_limit_middleware`)
    is installed unconditionally at ``vllm_mlx.server`` import time —
    its inner dispatch early-returns for paths outside
    ``AudioBodyLimitMiddleware._GUARDED_PATHS`` so a text-only server
    pays one tuple-membership check per request. We don't toggle it
    here because ``add_middleware`` after the FastAPI middleware stack
    has been built raises a Starlette warning, and on the text path
    the stack is built before ``load_model`` returns.
    """
    if getattr(app, _AUDIO_REGISTRATION_SENTINEL, False):
        return False
    app.include_router(router)
    setattr(app, _AUDIO_REGISTRATION_SENTINEL, True)
    return True


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
_music_engine = None

# OpenAI-style STT model alias → MLX repo. Promoted to module scope so
# the route can validate the model BEFORE streaming the upload (F-165):
# unknown names previously rode the body through the upload cap, then
# collapsed into a generic 500 "could not open/decode file" once
# ``STTEngine.load`` failed deep inside mlx-audio. Mirror the
# ``/v1/chat/completions`` and ``/v1/responses`` contract: validate the
# model name first and surface 404 with a distinct error type.
# R10-C1: the STT alias table is now sourced from the central
# ``vllm_mlx.audio.registry`` (aliases.json). Pre-R10 this dict was
# inlined here and the boot path in ``serve_command`` had no resolver
# at all — short aliases like ``whisper-1`` 404'd at HF before reaching
# the audio engine. The registry is the SINGLE place a new audio
# model lands; ``rapid-mlx models`` and the boot guard read from the
# same JSON file.
#
# We freeze a snapshot here at import time so the route's hot path
# avoids the JSON round-trip per request. The registry is read-only at
# runtime so the snapshot can never drift from the file.
from ..audio.registry import stt_aliases as _stt_aliases_from_registry

STT_MODEL_ALIASES: dict[str, str] = dict(_stt_aliases_from_registry())

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

#: Registry alias used when a forced-alignment request (``text`` present)
#: omits ``model``. The ASR default (:data:`DEFAULT_STT_ALIAS`) is NOT an
#: aligner, so defaulting the alignment branch to it would fail deep
#: inside ``STTEngine.align`` rather than doing what the caller asked.
DEFAULT_ALIGNER_ALIAS = "qwen3-aligner"


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


def _is_aligner_model(model_name: str) -> bool:
    """Return True iff ``model_name`` is a forced-alignment model.

    Mirrors :attr:`vllm_mlx.audio.stt.STTEngine._is_aligner` — the
    ``"aligner"`` substring is the same signal the engine uses to pick
    the ``align()`` call surface (``mlx-community/Qwen3-ForcedAligner-
    0.6B-8bit`` and the ``qwen3-aligner`` / ``qwen3-forced-aligner``
    aliases all resolve to an id containing ``aligner``). Kept at module
    scope so the transcriptions route can fail-fast BEFORE draining the
    upload rather than surfacing the engine's own ValueError as a 500.
    """
    return isinstance(model_name, str) and "aligner" in model_name.lower()


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


def _reject_word_timestamps_for_non_whisper(
    model: str, timestamp_granularities: list[str] | None
) -> None:
    """Reject ``timestamp_granularities[]=word`` on non-Whisper engines.

    Only the Whisper family produces per-word timings: mlx-audio's whisper
    ``generate`` accepts ``word_timestamps=True`` and attaches a per-word
    list to each segment. Parakeet / other STT engines don't expose the
    flag, so honoring a ``word`` request against them would return an empty
    ``words`` array that FALSELY signals the granularity was fulfilled.
    Reject up front with a 400 (mirroring the Whisper-only routing the
    translations lane already uses) so callers get an actionable error
    instead of a misleading 200. ``segment`` granularity is unaffected —
    every STT engine reports segments.

    Classification mirrors ``_reject_non_whisper_for_translation``: alias
    is resolved to its upstream id first; unknown bare aliases fall through
    so ``_resolve_stt_model`` still emits the 404 ``model_not_found``
    envelope (keeping the error surface consistent with transcriptions).
    """
    if not timestamp_granularities or "word" not in timestamp_granularities:
        return
    if not isinstance(model, str) or not model:
        return
    resolved = STT_MODEL_ALIASES.get(model, model)
    if "whisper" in resolved.lower():
        return
    # Unknown bare alias → let ``_resolve_stt_model`` 404 it so the
    # envelope matches the unknown-model path rather than this 400.
    if "/" not in model and model not in STT_MODEL_ALIASES:
        return
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": (
                    f"The model `{model}` cannot produce word-level "
                    "timestamps. `timestamp_granularities[]=word` requires a "
                    "Whisper engine (e.g. `whisper-large-v3` or "
                    "`whisper-large-v3-turbo`). Drop `word` (segment-level "
                    "timestamps work on every STT engine) or switch to a "
                    "Whisper model."
                ),
                "type": "invalid_request_error",
                "code": "invalid_model_for_word_timestamps",
                "param": "timestamp_granularities",
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


#: OpenAI's documented ``timestamp_granularities[]`` values. ``word``
#: yields per-word timings; ``segment`` yields per-segment cues.
_STT_TIMESTAMP_GRANULARITIES: frozenset[str] = frozenset(("word", "segment"))


def _normalise_timestamp_granularities(
    values: list[str] | None,
) -> list[str] | None:
    """Validate + de-duplicate OpenAI ``timestamp_granularities[]`` values.

    Returns ``None`` when nothing usable was requested (field omitted or
    empty list) so downstream code can treat "no granularities" as the
    pre-feature default. Each value is lower-cased and checked against
    OpenAI's two-value set; an unknown value raises a 400 with the same
    ``invalid_request_error`` envelope the rest of the route uses, so a
    typo (``"words"``) fails cheaply BEFORE the upload drains — mirroring
    ``_validate_response_format``.
    """
    if not values:
        return None
    normalised: list[str] = []
    for raw in values:
        value = (raw or "").strip().lower()
        if value not in _STT_TIMESTAMP_GRANULARITIES:
            available = ", ".join(sorted(_STT_TIMESTAMP_GRANULARITIES))
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "`timestamp_granularities[]` values must each be "
                            f"one of: {available}; got {raw!r}."
                        ),
                        "type": "invalid_request_error",
                        "code": "invalid_request",
                        "param": "timestamp_granularities",
                    }
                },
            )
        if value not in normalised:
            normalised.append(value)
    return normalised or None


def _format_srt_timestamp(seconds: float) -> str:
    """Format a float second offset as the SRT timestamp ``HH:MM:SS,mmm``.

    SRT mandates comma as the millisecond separator (vs. VTT's dot).
    Clamp negative inputs to zero — a defective backend reporting
    negative timestamps would otherwise emit a malformed cue.

    Codex r1 BLOCKING: when ``round((seconds - int(seconds)) * 1000)``
    overflows to 1000 (e.g. ``59.9996`` rounds to ``60.000``), the
    naive ``secs += 1`` branch produced timestamps like
    ``00:00:60,000``, which subtitle parsers reject as invalid
    (seconds must be ``< 60``). Convert the entire timestamp to integer
    milliseconds first, then decompose hierarchically so each carry
    propagates all the way up to the hours digit — same approach used
    by ffmpeg's ``av_strerror`` formatter.
    """
    if seconds < 0:
        seconds = 0.0
    total_millis = int(round(seconds * 1000))
    hours, rem = divmod(total_millis, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, millis = divmod(rem, 1000)
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


def _finite_or_none(value) -> float | None:
    """Parse ``value`` as a finite float for a word-timestamp field.

    Returns the finite float when ``value`` parses cleanly, or ``None`` for
    anything the caller must drop: a missing (``None``) timing, a
    non-numeric value, or a NaN / infinity. Nothing is defaulted or
    fabricated — a word without a genuine, finite start AND end is omitted
    rather than pinned to a plausible-but-wrong ``0.0`` (which would
    mislead caption consumers) or serialised as non-JSON-safe
    ``NaN``/``Infinity``.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _iter_words_for_verbose(result) -> list[dict]:
    """Flatten the engine's per-segment ``words`` lists into OpenAI's
    top-level ``words`` array shape.

    OpenAI's ``verbose_json`` word object has EXACTLY three keys —
    ``word`` (str), ``start`` (float seconds), ``end`` (float seconds).
    Whisper (mlx-audio) attaches a ``words`` list of
    ``{word, start, end, probability}`` to each segment when
    ``word_timestamps=True``; we drop ``probability`` (not part of the
    OpenAI contract) and walk both dict- and object-shaped words so a
    future non-dict backend still renders. Segments without a ``words``
    list (e.g. a non-Whisper backend that can't emit per-word timings)
    contribute nothing — the caller then returns an empty ``words: []``
    array rather than raising, matching the "degrade, don't 500"
    contract for models that lack word timestamps.
    """
    segments = getattr(result, "segments", None) or []
    words_out: list[dict] = []
    for seg in segments:
        if isinstance(seg, dict):
            seg_words = seg.get("words")
        else:
            seg_words = getattr(seg, "words", None)
        if not isinstance(seg_words, list):
            continue
        for w in seg_words:
            if isinstance(w, dict):
                word = w.get("word")
                start = w.get("start")
                end = w.get("end")
            else:
                word = getattr(w, "word", None)
                start = getattr(w, "start", None)
                end = getattr(w, "end", None)
            if word is None:
                continue
            # mlx-audio's whisper carries the raw leading-space token
            # (`" Welcome"`, `" to"`); OpenAI's ``words`` array reports the
            # bare word with surrounding whitespace stripped. Match OpenAI
            # so caption/subtitle consumers don't have to re-trim. A token
            # that is pure whitespace after stripping contributes nothing.
            word_str = str(word).strip()
            if not word_str:
                continue
            # A word must carry a genuine finite start AND end to be placed
            # on a timeline. Drop any word with a missing / non-numeric /
            # NaN / inf timing instead of fabricating one, raising a 500, or
            # emitting JSON that standard parsers reject (NaN/inf are not
            # valid JSON). Whisper always supplies both, so the happy path
            # is unaffected.
            start_f = _finite_or_none(start)
            end_f = _finite_or_none(end)
            if start_f is None or end_f is None:
                continue
            words_out.append(
                {
                    "word": word_str,
                    "start": start_f,
                    "end": end_f,
                }
            )
    return words_out


def _build_verbose_json_body(result, timestamp_granularities=None) -> dict:
    """Render the ``verbose_json`` body — text + language + duration + segments.

    Mirrors OpenAI's documented field set. ``segments`` is normalised
    to a list of dicts with the canonical key names; the engine's
    raw shape is whatever ``stt`` chose to expose (whisper-mlx ships
    dicts; future backends might ship objects).

    ``timestamp_granularities`` maps OpenAI's ``timestamp_granularities[]``
    request field to the response shape:

    * ``None`` (field omitted) — the pre-feature default: emit
      ``segments`` only. Unchanged so existing clients see no drift.
    * contains ``"segment"`` — emit ``segments``.
    * contains ``"word"`` — additionally emit a top-level ``words``
      array of ``{word, start, end}``.

    Both keys can be present when both granularities are requested; a
    ``["word"]``-only request emits ``words`` without ``segments``,
    matching OpenAI's Whisper API.
    """
    include_segments = (
        timestamp_granularities is None or "segment" in timestamp_granularities
    )
    include_words = (
        timestamp_granularities is not None and "word" in timestamp_granularities
    )

    body: dict = {
        "task": "transcribe",
        "text": getattr(result, "text", ""),
        "language": getattr(result, "language", None),
        "duration": getattr(result, "duration", None),
    }
    if include_segments:
        cues = _iter_segments_for_subtitles(result)
        body["segments"] = [
            {
                "id": idx,
                "start": start,
                "end": end,
                "text": text,
            }
            for idx, (start, end, text) in enumerate(cues)
        ]
    if include_words:
        body["words"] = _iter_words_for_verbose(result)
    return body


def _format_stt_response(
    result, response_format: str, task: str, timestamp_granularities=None
):
    """Branch on the validated ``response_format`` and produce a body.

    Centralised so the transcription and translation routes pick the
    same shape for the same value — a future change to one path
    automatically lands on the other.

    ``timestamp_granularities`` only affects the ``verbose_json`` branch
    (OpenAI's word/segment timestamp switch); every other format ignores
    it, exactly as OpenAI's API does.
    """
    if response_format == "text":
        return PlainTextResponse(getattr(result, "text", "") or "")
    if response_format == "srt":
        return PlainTextResponse(_build_srt_body(result), media_type="text/srt")
    if response_format == "vtt":
        return PlainTextResponse(_build_vtt_body(result), media_type="text/vtt")
    if response_format == "verbose_json":
        body = _build_verbose_json_body(
            result, timestamp_granularities=timestamp_granularities
        )
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

#: Substrings that POSITIVELY identify a decode/codec failure on the
#: audio file itself (i.e. a CLIENT problem). Keep this list narrow so
#: we don't accidentally relabel a legitimate server-side bug as a
#: client error — codex r2 BLOCKING: broad tool-name tokens (``ffmpeg``,
#: ``audioread``) used in isolation flagged server misconfiguration
#: (missing/broken decoder binary) as a 400. Each hint here must
#: describe a FORMAT-shaped failure, not just mention a library name.
_DECODE_ERROR_HINTS: tuple[str, ...] = (
    "could not open",
    "could not decode",
    "format not recognised",
    "format not recognized",
    "unknown format",
    "no such format",
    "invalid audio",
    "unsupported audio",
    "could not load audio",
    "header is truncated",
    "error opening",
    # ``LibsndfileError: ...`` only fires on libsndfile-rejected bytes
    # (truncated header, unknown subtype) — it's a strong file-shape
    # signal, NOT a generic "soundfile imported" marker. Codex r2:
    # bare ``soundfile``/``libsndfile`` substrings could match a
    # ModuleNotFoundError, so qualify them.
    "libsndfile error",
    "soundfile.libsndfileerror",
)

#: Hints that indicate the decode tool itself is BROKEN on the server
#: (missing binary, version mismatch, library not found). These must
#: NOT downgrade to 400 — they're a server misconfiguration, the
#: client did nothing wrong. Codex r2 BLOCKING fix: pre-fix a missing
#: ``ffmpeg`` binary on the host would have produced "ffmpeg not found"
#: which matched the prior broad ``"ffmpeg"`` hint and got relabeled as
#: a client error.
_DECODE_SERVER_MISCONFIG_HINTS: tuple[str, ...] = (
    "not found",
    "not installed",
    "no module named",
    "no such file or directory",
    "executable not found",
    "command not found",
    "no module",
    "libsndfile not found",
)


def _is_decode_error(exc: Exception) -> bool:
    """Return True iff ``exc`` looks like an audio-decode failure CAUSED
    BY THE UPLOADED FILE (not by server misconfiguration).

    Matches both the exception's class name (``DecodeError``,
    ``LibsndfileError``, ``SoundFileError``) and the message text
    against a curated POSITIVE hint list. Type-based matching is the
    strong signal — the substring check is the fallback because
    ``mlx_audio`` chains raw ``ValueError`` / ``RuntimeError`` for
    decode failures in some code paths.

    Codex r2 BLOCKING: a message that matches a decode hint AND also
    matches a server-misconfig hint (``"ffmpeg binary not found"``,
    ``"libsndfile not installed"``) is NOT a client error. Bail out
    in that case so the envelope stays 500 ``transcription_failed`` —
    operators need to see those reports unmodified to know the host
    audio stack is broken.
    """
    if not isinstance(exc, Exception):
        return False
    msg = str(exc).lower()
    # Server-misconfig hints take precedence — even on a decode-shaped
    # class name (``ImportError`` doesn't but ``RuntimeError("ffmpeg
    # not found")`` could pattern-match the class-name path below).
    if any(hint in msg for hint in _DECODE_SERVER_MISCONFIG_HINTS):
        return False
    cls_name = type(exc).__name__.lower()
    if any(tok in cls_name for tok in ("decode", "sndfile", "soundfile", "codec")):
        return True
    return any(hint in msg for hint in _DECODE_ERROR_HINTS)


# Pattern for any string that LOOKS like a filesystem path. We strip
# these from the decode reason BEFORE returning it to the client so a
# librosa error of the form ``Error opening
# '/var/folders/xy/T/tmpXYZ.wav': ...`` doesn't leak the server's
# tempdir layout (codex r2 BLOCKING).
#
# Three shapes covered:
#   1. Absolute POSIX paths (``/var/...``, ``/Users/...``)
#   2. Windows-style absolute paths (``C:\Users\...``)
#   3. Paths quoted inside the message (``'/tmp/x.wav'``)
#
# Replacement is the literal ``<redacted>`` so the rest of the message
# (which often carries the actual format hint, e.g. ``Format not
# recognised``) survives unchanged.
_PATH_LIKE_RE = re.compile(
    # Quoted absolute paths first — librosa / soundfile wrap them in
    # single quotes. Match the entire quoted span (greedy through the
    # closing quote) so the path AND the quotes go together.
    r"""
    (?P<quoted>['"][/\\][^'"]*['"]) |   # '/abs/path' or "C:\path"
    (?P<unix>(?<![A-Za-z0-9_])/[A-Za-z0-9_./\-]+) |   # bare absolute POSIX
    (?P<win>(?<![A-Za-z0-9_])[A-Za-z]:\\[A-Za-z0-9_.\\\-]+)  # bare Win path
    """,
    re.VERBOSE,
)


def _sanitize_decode_reason(reason: str) -> str:
    """Strip filesystem paths from a decode error message.

    Codex r2 BLOCKING: librosa/ffmpeg/soundfile errors commonly echo
    the temp file path the route created (``/var/folders/.../tmpXYZ.wav``
    or even the original upload filename when the client controlled
    it). Echoing the server path is a low-severity infoleak — it
    discloses tempdir layout. Replace any path-shaped token with the
    literal ``<redacted>`` so the format-shape phrase (``Format not
    recognised``) still reaches the client.

    Also caps the overall length so a runaway exception (huge bytes
    quoted in the message) can't produce a multi-MB JSON envelope.
    """
    if not reason:
        return ""
    sanitized = _PATH_LIKE_RE.sub("<redacted>", reason)
    # Collapse multiple ``<redacted>`` in a row from overlapping matches.
    sanitized = re.sub(r"(<redacted>\s*){2,}", "<redacted> ", sanitized)
    # Length cap: keep messages bounded. 240 chars is plenty for any
    # legitimate decode-reason phrase + an embedded format name.
    if len(sanitized) > 240:
        sanitized = sanitized[:237] + "..."
    return sanitized.strip()


def _audio_decode_error_envelope(exc: Exception) -> HTTPException:
    """Build the OpenAI-shape 400 envelope for a decode failure.

    Codex r2 BLOCKING: the envelope previously included raw ``str(exc)``,
    which can carry server filesystem paths (temp file basenames,
    librosa-echoed locations). Run the reason through
    :func:`_sanitize_decode_reason` so the FORMAT hint survives but
    paths are redacted.

    The FULL exception (with paths) still goes to the operator log
    in the caller — only the sanitized form reaches the client.
    """
    raw = str(exc).strip() or type(exc).__name__
    safe = _sanitize_decode_reason(raw) or type(exc).__name__
    return HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": f"could not decode audio file: {safe}",
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
    timestamp_granularities: list[str] | None = None,
    context: str | None = None,
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

    # Forced-alignment routing (Qwen3-ForcedAligner). Requests that carry
    # a ``text`` field never reach this helper — ``create_transcription``
    # dispatches them to :func:`_run_alignment_request` instead. What is
    # left here is the incoherent combination that lands on the ASR path:
    # an aligner model with nothing to align to. Reject it BEFORE draining
    # the upload with a clean 400 so the caller gets an actionable message
    # instead of the engine's own ValueError collapsing into a generic 500
    # (``STTEngine.transcribe`` refuses aligner models — it cannot
    # recognize speech).
    if _is_aligner_model(model_name):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"model `{model}` is a forced-alignment model and "
                        "requires the known transcript in the `text` field. "
                        "Forced alignment returns per-character timings for "
                        "text you already have; it does not recognize speech. "
                        "Use a Whisper/Parakeet model for recognition."
                    ),
                    "type": "invalid_request_error",
                    "code": "alignment_text_required",
                    "param": "text",
                }
            },
        )

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

        # Same lock the alignment lane takes. ASR still runs INLINE on the
        # event loop, so this does not change how it executes — it makes
        # explicit the serialisation it already had implicitly, and closes
        # the window that offloading alignment would otherwise open: an ASR
        # request on the loop concurrent with an alignment render in the
        # executor means two multi-GB models resident and two callers
        # driving the accelerator. See _stt_lane_lock.
        async with _get_stt_lane_lock():
            if _stt_engine is None or _stt_engine.model_name != model_name:
                # Symmetric with the alignment path: one STT model resident.
                _evict_other_lane("asr")
                _stt_engine = None
                stt_engine = STTEngine(model_name)
                stt_engine.load()
                _stt_engine = stt_engine

            # Forward ``timestamp_granularities`` only when requested.
            # Keeping the default call shape unchanged preserves compatibility
            # with older STTEngine-shaped stubs and third-party engines.
            transcribe_kwargs: dict = {"language": language, "task": task}
            if timestamp_granularities:
                transcribe_kwargs["timestamp_granularities"] = timestamp_granularities
            # Only forward when non-empty: an empty hint still costs the decoder
            # attention, and omitting the kwarg keeps the call shape compatible
            # with STTEngine-shaped stubs and third-party engines.
            if context and context.strip():
                transcribe_kwargs["context"] = context.strip()
            result = _stt_engine.transcribe(tmp_path, **transcribe_kwargs)

        # R6-H2: branch on the validated ``response_format`` so callers
        # that requested ``srt`` / ``vtt`` / ``verbose_json`` actually
        # get those shapes. Pre-fix only ``text`` had a non-JSON path;
        # everything else fell through to the JSON envelope.
        return _format_stt_response(
            result,
            response_format,
            task=task,
            timestamp_granularities=timestamp_granularities,
        )

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


# ---------------------------------------------------------------------------
# Forced-alignment lane for ``/v1/audio/transcriptions`` + ``text``.
#
# Kept separate from ``_run_stt_request`` because the two lanes have
# different concurrency models: ASR still runs its engine call inline on
# the event loop, while alignment offloads to a worker thread (see the
# lock and engine-cache comments below).
# ---------------------------------------------------------------------------


#: Serialises the STT lane — BOTH transcription/translation and forced
#: alignment. One lock, not one per lane, and that matters.
#:
#: The two lanes share no Python state (``_stt_engine`` and
#: ``_aligner_engine`` are separate caches on purpose), but they share the
#: accelerator: each loads its own multi-GB model into unified memory and
#: runs MLX work against it. Before this change every audio lane executed
#: inline on the event loop, so they were mutually exclusive by accident.
#: Offloading alignment to a worker thread removes that accident — an ASR
#: request could then run on the loop while an alignment render is live in
#: the executor, with two models resident and two callers driving the GPU.
#: The lock restores the invariant deliberately instead of relying on the
#: event loop to provide it.
#:
#: Note this does not change how ASR EXECUTES: it still runs inline, so
#: taking the lock around it only makes explicit the serialisation it
#: already had. TTS, alignment, and music run in workers.
#:
#: An ``asyncio.Lock`` acquired ON THE EVENT LOOP, deliberately not a
#: ``threading.Lock`` held inside the worker. ``asyncio.to_thread`` uses
#: the shared default executor (min(32, cpu+4) threads), so queued
#: requests blocking on a thread-level lock would each pin an executor
#: thread just to wait — starving every other ``to_thread`` user in the
#: process (prefix-cache save, tool-grammar warmup). Waiting on the loop
#: instead costs a coroutine, not a thread, and only the request that
#: actually runs ever occupies an executor slot.
#:
class _CrossLoopAsyncLock:
    """Process-wide lock that never occupies the shared asyncio executor."""

    def __init__(self, thread_name_prefix: str = "rapid-mlx-audio-lock") -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix
        )

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        completed = threading.Event()

        def acquire() -> None:
            try:
                self._lock.acquire()
            finally:
                completed.set()

        waiter = loop.run_in_executor(self._executor, acquire)
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            # Cancelling an asyncio Future cannot stop a running
            # ``threading.Lock.acquire``. Drain the dedicated worker before
            # propagating so an abandoned acquisition never owns the lock.
            while not completed.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass
            self._lock.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._lock.release()


_stt_lane_lock = _CrossLoopAsyncLock("rapid-mlx-stt-lock")
_tts_lane_lock = _CrossLoopAsyncLock("rapid-mlx-tts-lock")
_music_lock = _CrossLoopAsyncLock("rapid-mlx-music-lock")


def _get_stt_lane_lock() -> _CrossLoopAsyncLock:
    """Return the process-wide STT-lane lock."""
    return _stt_lane_lock


def _get_tts_lane_lock() -> _CrossLoopAsyncLock:
    """Return the process-wide TTS-lane lock."""
    return _tts_lane_lock


def _get_music_lock() -> _CrossLoopAsyncLock:
    """Return the process-wide music-lane lock."""
    return _music_lock


#: Signatures of the two ``ValueError``s :meth:`STTEngine.align` raises
#: for caller mistakes (see its body): a model that isn't a forced
#: aligner, and empty/blank known text. Matched on message because the
#: engine raises bare ``ValueError`` for both; everything else that
#: surfaces as a ValueError is an internal fault, not a bad request.
#:
#: The route rejects both shapes up front (``_is_aligner_model`` before
#: the upload drains, blank ``text`` in ``create_transcription``), so
#: this classifier is a BACKSTOP: it keeps an engine-side rejection from
#: being dressed up as a generic 500 should the engine's own aligner
#: predicate ever diverge from the route's substring heuristic.
_CLIENT_ALIGNMENT_ERROR_SIGNATURES = (
    "requires a forced-aligner model",
    "requires non-empty known text",
)


def _is_client_alignment_error(exc: Exception) -> bool:
    """True if ``exc`` is an ``align()`` rejection the CALLER can fix."""
    message = str(exc).lower()
    return any(sig in message for sig in _CLIENT_ALIGNMENT_ERROR_SIGNATURES)


#: Cached forced-aligner engine — DELIBERATELY separate from
#: ``_stt_engine``.
#:
#: Sharing one global across both lanes is unsafe now that alignment runs
#: on a worker thread while ASR still runs on the event loop: an ASR
#: request can replace the engine in between the alignment path's cache
#: check and its ``align()`` call, and no lock held on only one side can
#: prevent that. A dedicated cache removes the shared mutable state
#: instead of trying to synchronise two lanes with different concurrency
#: models. It also stops the two lanes evicting each other's weights on
#: every alternating request, since an aligner and an ASR model are never
#: the same ``model_name``.
_aligner_engine = None


def _evict_other_lane(keep: str) -> None:
    """Release the STT lane's *other* cached engine before loading one.

    ``keep`` is ``"asr"`` or ``"aligner"``. Only ever called with the lane
    lock held, so the engine being dropped is guaranteed idle.

    Why this exists: separate caches per lane fix the race (an ASR request
    can no longer swap the engine under an in-flight alignment) but not the
    footprint — alternating requests would leave both models resident. MLX
    frees on refcount, so clearing the global is the release.
    """
    global _stt_engine, _aligner_engine

    if keep == "aligner" and _stt_engine is not None:
        logger.info(
            "Releasing ASR model %s to load the forced aligner "
            "(one STT model resident at a time)",
            getattr(_stt_engine, "model_name", "?"),
        )
        _stt_engine = None
    elif keep == "asr" and _aligner_engine is not None:
        logger.info(
            "Releasing forced aligner %s to load the ASR model "
            "(one STT model resident at a time)",
            getattr(_aligner_engine, "model_name", "?"),
        )
        _aligner_engine = None


def _align_blocking(
    model_name: str,
    audio_path: str,
    text: str,
    language: str | None,
):
    """Blocking half of the forced-alignment request — runs on a thread.

    Loads (or reuses) the aligner engine and runs
    :meth:`STTEngine.align`. Split out of :func:`_run_alignment_request`
    so the async handler can hand the seconds-long weight load + align
    to a worker thread instead of stalling the event loop.

    The caller holds :data:`_stt_lane_lock` for the whole call, so this
    body is already serialised — no thread-level locking here (see the
    lock's own comment for why it lives on the event loop).
    """
    global _aligner_engine

    from ..audio.stt import STTEngine

    # STTEngine.align defaults language to "Chinese"; only forward an
    # explicit caller value so the engine default stands otherwise.
    align_kwargs = {}
    if language:
        align_kwargs["language"] = language

    if _aligner_engine is None or _aligner_engine.model_name != model_name:
        # Drop the ASR lane's model before loading ours. The lock already
        # stops the two lanes RUNNING at once, but without this they both
        # stay resident after alternating requests — two multi-GB models in
        # unified memory for a server that can only use one at a time. The
        # caller holds the lane lock, so no ASR request is mid-flight and
        # this cannot pull weights out from under one.
        _evict_other_lane("aligner")
        # Also drop any PREVIOUS aligner (a different aligner alias) before
        # loading the replacement, so two multi-GB aligner models never sit
        # resident together during ``load()``. Inert under the current
        # single-aligner registry — the branch only re-enters when the cache
        # was already emptied (an ASR request evicted us), so there is nothing
        # to drop — but it keeps the "one STT model resident" invariant true if
        # a second aligner alias is ever registered. On a failed reload the
        # cache stays ``None`` and the next request reloads from disk, strictly
        # better than pinning a stale model.
        _aligner_engine = None
        # Load into a local first and publish only on success. Caching a
        # half-constructed engine would leave later requests matching on
        # ``model_name`` against an object whose weights never loaded.
        #
        # Named ``aligner``, not ``engine``: test_route_engine_contract
        # scans this module for ``engine.<method>()`` calls and requires
        # the method to exist on the LLM ``BaseEngine``. An ``STTEngine``
        # is a different hierarchy entirely, so the name would trip that
        # gate with a false positive.
        aligner = STTEngine(model_name)
        aligner.load()
        _aligner_engine = aligner
    return _aligner_engine.align(audio_path, text, **align_kwargs)


async def _run_alignment_request(
    file: UploadFile,
    model: str,
    text: str,
    language: str | None,
    response_format: str,
):
    """Forced-alignment pipeline for ``/v1/audio/transcriptions`` + ``text``.

    When the transcription request carries a ``text`` field the caller is
    asking for FORCED ALIGNMENT (align the KNOWN transcript to the audio,
    returning per-character/word timestamps) rather than ASR. This helper
    mirrors :func:`_run_stt_request`'s size/resolve/cleanup/envelope
    wiring but calls :meth:`STTEngine.align` and lets ``language`` fall
    back to the aligner's own ``"Chinese"`` default when omitted.

    The result's per-unit ``segments`` flow through the SAME
    ``_format_stt_response`` serializers ASR uses, so ``verbose_json`` /
    ``srt`` / ``vtt`` all work — the aligner emits exactly the
    ``{text, start, end}`` segment shape those formatters consume.
    """
    # Resolve the aligner model up front (404 for unknown aliases) BEFORE
    # draining the upload — same fail-fast ordering as the ASR path.
    model_name = _resolve_stt_model(model)

    # ``text`` only means anything to a forced aligner: a Whisper /
    # Parakeet engine cannot align, so silently ignoring the field would
    # mislead the caller into thinking alignment ran. Reject before the
    # upload drains rather than letting ``align()`` raise several
    # megabytes later.
    if not _is_aligner_model(model_name):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"the `text` field triggers forced alignment, which "
                        f"requires a forced-aligner model; `{model}` is not "
                        f"one. Pass `model={DEFAULT_ALIGNER_ALIAS}` (or another "
                        "aligner alias) to align `text` to the audio, or omit "
                        "`text` for normal speech-to-text."
                    ),
                    "type": "invalid_request_error",
                    "code": "alignment_model_required",
                    "param": "model",
                }
            },
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            await _stream_upload_to_tempfile(file, tmp)

        # Weight load + alignment are seconds of blocking compute, so run
        # them on a worker thread — an ``async def`` handler that calls
        # them inline stalls the whole event loop (every concurrent chat
        # completion, /healthz probe and SSE heartbeat) for the duration.
        #
        # Serialise BEFORE offloading: queueing on the loop costs a
        # coroutine, whereas queueing inside the worker would pin an
        # executor thread per waiter and starve every other to_thread user.
        # ``run_to_completion`` keeps the lock held and ``tmp_path`` alive
        # for exactly as long as the worker runs, even if the client
        # disconnects and cancels us mid-align.
        async with _get_stt_lane_lock():
            result = await run_to_completion(
                _align_blocking, model_name, tmp_path, text, language
            )

        return _format_stt_response(result, response_format, task="transcribe")

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="mlx-audio not installed. Install with: pip install mlx-audio",
        )
    except HTTPException:
        raise
    except Exception as e:
        # ONE handler, classified inside — deliberately not a chain of
        # ``except ValueError`` / ``except Exception`` clauses. A bare
        # ``raise`` in the narrower clause exits the whole try statement
        # rather than falling through to the broader one, so an
        # "unclassified → let the generic handler take it" flow silently
        # became a 500 with no envelope. Classifying in one place makes
        # the precedence explicit and testable.

        # 1. Corrupted / undecodable upload. Checked FIRST because some
        #    codec paths raise plain ValueError, which would otherwise be
        #    reported as a bad alignment request blaming ``model``/``text``
        #    and send the caller chasing a field that was never wrong.
        if _is_decode_error(e):
            logger.info("Forced alignment rejected corrupted upload: %s", e)
            raise _audio_decode_error_envelope(e)

        # 2. The two ``align()`` rejections the CALLER can fix: a model
        #    that isn't a forced aligner, or blank known text. Any other
        #    ValueError (weight loading, tokenizing, a reshape deep in the
        #    model) is an internal fault and must not be dressed up as a
        #    client error.
        if isinstance(e, ValueError) and _is_client_alignment_error(e):
            logger.info("Forced alignment rejected request: %s", e)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": "invalid_alignment_request",
                        "param": "text" if "text" in str(e).lower() else "model",
                    }
                },
            )

        # 3. Everything else is ours, not the caller's.
        logger.exception("Forced alignment failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Audio forced alignment failed",
                    "type": "api_error",
                    "code": "alignment_failed",
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
    # ``text`` is a rapid-mlx extension (NOT an OpenAI field) that turns
    # this route into the forced-alignment surface: when present, the
    # known transcript is aligned to the audio and per-character timings
    # come back in the segment shape verbose_json/srt/vtt already render.
    # ``model`` then defaults to :data:`DEFAULT_ALIGNER_ALIAS`. Accepted
    # on both form and query for parity with the other STT fields.
    text_form: str | None = Form(None, alias="text"),
    # Proper nouns / hotwords biasing the decoder. Off-spec for OpenAI (whose
    # Whisper API calls the equivalent ``prompt``), so it is accepted under a
    # backend-neutral name and mapped to each family's own kwarg in STTEngine.
    context_form: str | None = Form(None, alias="context"),
    # STT-word-timestamps: OpenAI serialises the array field as
    # ``timestamp_granularities[]`` (bracketed) in the multipart body.
    # We also accept the un-bracketed ``timestamp_granularities`` name
    # (some SDK/curl variants send it that way) and both spellings on the
    # query string, mirroring the form-over-query precedence the rest of
    # this route already uses for ``model``/``language``/``response_format``.
    timestamp_granularities_bracket_form: list[str] | None = Form(
        None, alias="timestamp_granularities[]"
    ),
    timestamp_granularities_plain_form: list[str] | None = Form(
        None, alias="timestamp_granularities"
    ),
    model_query: str | None = Query(None, alias="model"),
    language_query: str | None = Query(None, alias="language"),
    response_format_query: str | None = Query(None, alias="response_format"),
    text_query: str | None = Query(None, alias="text"),
    timestamp_granularities_bracket_query: list[str] | None = Query(
        None, alias="timestamp_granularities[]"
    ),
    timestamp_granularities_plain_query: list[str] | None = Query(
        None, alias="timestamp_granularities"
    ),
):
    """Transcribe audio to text (OpenAI Whisper API compatible).

    Forced-alignment extension: if the request carries a ``text`` field,
    the route aligns that KNOWN transcript to the uploaded audio
    (per-character/word timestamps, zero recognition error) via
    :meth:`STTEngine.align` instead of running ASR. When ``text`` is
    present but ``model`` is omitted it defaults to the registered
    aligner alias (:data:`DEFAULT_ALIGNER_ALIAS`), and the response
    defaults to ``verbose_json`` so the timestamped ``segments`` are
    actually in the body. When ``text`` is absent the behaviour is
    unchanged.

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
    #
    # ``model_merged`` / ``response_format_provided`` record whether the
    # caller explicitly supplied the field — they drive the alignment
    # defaults below, which only kick in when the field was omitted.
    model_merged = next(
        (v for v in (model_form, model_query) if isinstance(v, str)),
        None,
    )
    model_provided = model_merged is not None
    response_format_provided = any(
        isinstance(v, str) for v in (response_format_form, response_format_query)
    )
    # Select with an isinstance(str) check, NOT ``is not None``, for the same
    # direct-call reason as ``text`` below: a handler invoked as a plain
    # coroutine (not through FastAPI) receives any unpassed param as its
    # unresolved ``Form``/``Query`` sentinel — truthy and non-None but NOT a
    # string. Left as ``is not None`` that sentinel would flow through to the
    # ASR/alignment engines as a bogus ``language``. Treat a non-str as absent.
    language = next(
        (v for v in (language_form, language_query) if isinstance(v, str)),
        None,
    )
    response_format = next(
        (
            v
            for v in (response_format_form, response_format_query)
            if isinstance(v, str)
        ),
        "json",
    )
    # Forced-alignment transcript (rapid-mlx extension). Form wins over
    # query, matching the model/language/response_format precedence.
    #
    # Merge with an isinstance check, NOT ``is not None``. Several existing
    # tests (e.g. test_audio_upload_size_limit) call this handler directly
    # rather than through FastAPI, so any parameter they don't pass arrives
    # as its unresolved ``Form(None)`` / ``Query(None)`` object — truthy
    # and non-None, but NOT a string. The pre-existing params only ever get
    # compared against None so they tolerate that; ``text`` is inspected
    # with ``.strip()`` below, which would raise ``AttributeError: 'Form'
    # object has no attribute 'strip'``. Treat a non-str as absent.
    text = next(
        (v for v in (text_form, text_query) if isinstance(v, str)),
        None,
    )

    # PRESENCE, not truthiness, selects the alignment branch. A
    # whitespace-only ``text`` used to fall through to ASR whenever the
    # model wasn't an aligner, which silently answered a different
    # question than the caller asked — they wanted timestamps for a
    # transcript and got speech recognition instead, with a 200 that gives
    # no hint anything was ignored. It now 400s regardless of ``model``.
    #
    # A truly empty ``text=""`` is indistinguishable from an absent field
    # here — FastAPI coerces both to ``None`` for an ``Optional[str]``
    # form param — so it stays ASR. Only a non-empty-but-blank value is
    # rejectable, and it is the shape that signals real caller intent.
    if text is not None and not text.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "`text` was supplied but is blank. Send the known "
                        "transcript to align, or omit `text` entirely for "
                        "speech recognition."
                    ),
                    "type": "invalid_request_error",
                    "code": "alignment_text_required",
                    "param": "text",
                }
            },
        )
    is_alignment = text is not None

    if is_alignment:
        # Default to the registered aligner alias when the caller didn't
        # pick a model — the ASR default (whisper-large-v3) is not an
        # aligner and would fail deep in the engine.
        #
        # A WHITESPACE-ONLY model also takes the default here. FastAPI
        # already coerces a truly empty form field (``model=""``) to
        # ``None``, but ``model="   "`` — what you get from a form whose
        # input was spaced-out, or a shell ``-F "model=$UNSET "`` — comes
        # through verbatim and 404s in ``_resolve_stt_model`` as a
        # nonexistent alias. Treat it as unset so "just send audio +
        # text" keeps working. Scoped to this branch on purpose: the ASR
        # path's blank-model handling is long-standing contract.
        # isinstance guard for the same direct-call reason as ``text``.
        alignment_model_chosen = isinstance(model_merged, str) and bool(
            model_merged.strip()
        )
        model = model_merged if alignment_model_chosen else DEFAULT_ALIGNER_ALIAS
        # Default to verbose_json so the timestamped ``segments`` are in
        # the body; the plain ``json`` envelope drops them. An explicit
        # response_format (srt/vtt/text/json) is honoured as-is.
        # Whitespace-only is treated as unset for the same reason as
        # ``model`` above (it would otherwise 400 on the allowed set).
        if (
            not response_format_provided
            or not isinstance(response_format, str)
            or not response_format.strip()
        ):
            response_format = "verbose_json"
    else:
        model = model_merged if model_provided else DEFAULT_STT_ALIAS

    # R6-H2: reject unknown ``response_format`` values up front with a
    # 400 envelope so a typo (``"jsno"``) or unsupported value
    # (``"yaml"``) fails BEFORE we drain the upload, load the engine,
    # or run inference. Pre-fix the value silently fell through to the
    # JSON branch, masking client-side bugs as "STT lies about
    # response_format".
    response_format = _validate_response_format(response_format)

    # STT-word-timestamps: resolve the requested granularities from
    # whichever source carried them (bracketed form wins → plain form →
    # bracketed query → plain query), then validate the values up front so
    # a bad value (``"words"``) fails cheaply with a 400 before the upload
    # drains — same lifecycle as ``response_format`` above.
    #
    # Select the first NON-EMPTY list, NOT the first truthy value. Same
    # direct-call hazard the ``text`` merge above guards: a handler invoked
    # as a plain coroutine (not through FastAPI) receives any unpassed param
    # as its unresolved ``Form``/``Query`` sentinel — truthy and non-None but
    # NOT a list — so a raw ``or`` chain would forward that sentinel into the
    # normaliser and raise ``TypeError: 'Form' object is not iterable``. The
    # ``isinstance(v, list) and v`` guard keeps the original first-truthy
    # precedence (None / empty list / sentinel all skipped) while tolerating
    # the sentinel.
    _tg_source = next(
        (
            v
            for v in (
                timestamp_granularities_bracket_form,
                timestamp_granularities_plain_form,
                timestamp_granularities_bracket_query,
                timestamp_granularities_plain_query,
            )
            if isinstance(v, list) and v
        ),
        None,
    )
    timestamp_granularities = _normalise_timestamp_granularities(_tg_source)

    # OpenAI contract: ``timestamp_granularities[]`` is only meaningful
    # with ``response_format=verbose_json`` (the only shape that carries a
    # ``words``/``segments`` array). Reject the mismatch with a 400 instead
    # of silently ignoring the field, so a caller that forgot to set
    # ``verbose_json`` learns why their timestamps never arrived.
    if timestamp_granularities is not None and response_format != "verbose_json":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        "`timestamp_granularities[]` requires "
                        "`response_format=verbose_json`; got "
                        f"response_format={response_format!r}."
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": "timestamp_granularities",
                }
            },
        )

    # Word-level timings are a Whisper-only capability — reject the
    # ``word`` granularity on non-Whisper engines with a 400 rather than
    # returning an empty ``words`` array that falsely claims fulfillment.
    _reject_word_timestamps_for_non_whisper(model, timestamp_granularities)

    # F-D05: STT-lane audio dep probe — same envelope as the TTS
    # lane shares. Fires BEFORE we spool any upload bytes so a broken
    # ``mlx_audio.stt`` install rejects cheaply (no temp file, no
    # read loop). Probed here (not inside ``_run_stt_request``) so
    # the source-grep guard in test_audio_probe_consistency.py sees
    # the ``require_mlx_audio`` call directly inside the route body.
    from ..audio.probe import require_mlx_audio_stt

    require_mlx_audio_stt()

    if is_alignment:
        return await _run_alignment_request(
            file=file,
            model=model,
            text=text,
            language=language,
            response_format=response_format,
        )

    return await _run_stt_request(
        file=file,
        model=model,
        language=language,
        response_format=response_format,
        task="transcribe",
        timestamp_granularities=timestamp_granularities,
        context=context_form,
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


# R7-H3 (Bo 0.8.8 dogfood): TTS short-alias → HF repo map. Promoted
# from the inline ``model_map`` inside ``create_speech`` to a module-
# level constant so STT and TTS aliases live side-by-side and the
# unit tests can pin the table without crawling the handler body.
# Mirrors ``STT_MODEL_ALIASES`` (R-04 contract). Any future engine
# addition lands here once, not in the handler.
# R10-C1: TTS alias table now sourced from the central audio
# registry — same rationale as ``STT_MODEL_ALIASES`` above. The JSON
# file ships the verified HF id for every entry (including a real
# ``mlx-community/Kokoro-82M-8bit`` which now exists; pre-R10 we
# silently aliased it to the bf16 build because there was no 8bit
# repo at that time).
from ..audio.registry import tts_aliases as _tts_aliases_from_registry

TTS_MODEL_ALIASES: dict[str, str] = dict(_tts_aliases_from_registry())

#: Default TTS alias for empty / ``"default"`` requests. Mirrors the
#: ``DEFAULT_STT_ALIAS`` rule on the STT side — drop-in OpenAI-SDK code
#: that omits ``model=`` lands here.
DEFAULT_TTS_ALIAS = "kokoro"


# R8-H5 (Bo 0.8.9 dogfood): canonical IANA Content-Type per
# ``response_format``. Pre-fix the route inlined ``f"audio/{format}"``
# which both mislabeled the body (every non-wav format was actually
# WAV bytes — see ``TTSEngine.to_bytes`` for the encoder fix) AND
# emitted non-canonical types: ``audio/opus`` instead of ``audio/ogg``
# (Opus's IANA type is ``audio/ogg`` because the wire bytes are an
# OGG container with the Opus codec), ``audio/mp3`` instead of the
# IANA-canonical ``audio/mpeg``. The table below pairs each format
# with the type that matches what the encoder actually produces so
# browsers and ffmpeg agree on the container.
#
# Tied to :data:`vllm_mlx.api.models._TTS_ALLOWED_RESPONSE_FORMATS` —
# every key here MUST be in the allowed set so a request that passes
# the request-model validator always finds a Content-Type entry. The
# unit test ``test_audio_r8_a_bundle.py::TestTTSContentType`` pins
# both directions.
_TTS_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "mp3": "audio/mpeg",
    "pcm": "audio/pcm",
}

_MAX_TTS_REF_AUDIO_BYTES = 10 * 1024 * 1024


def _decode_tts_ref_audio(value: str) -> bytes:
    """Decode a base64 F5 reference clip with a bounded payload size."""
    encoded = value
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
        except ValueError as exc:
            raise ValueError("ref_audio data URL is malformed") from exc
        if ";base64" not in header:
            raise ValueError("ref_audio data URL must use base64 encoding")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ref_audio must be valid base64-encoded audio") from exc
    if not decoded:
        raise ValueError("ref_audio must not be empty")
    if len(decoded) > _MAX_TTS_REF_AUDIO_BYTES:
        raise ValueError("ref_audio exceeds the 10 MB decoded size limit")
    return decoded


def _served_tts_default() -> str | None:
    """The served TTS model's name, or ``None`` when no TTS model is served.

    ``rapid-mlx serve qwen3-tts-voicedesign`` stamps the served alias onto
    ``ServerConfig.model_alias`` and its HF id onto ``model_name`` (see the
    audio branch of ``cli.serve_command``). Without consulting it, the TTS
    routes fall back to :data:`DEFAULT_TTS_ALIAS` for an omitted ``model``, so a
    server started on one TTS model answers a bare ``/v1/audio/voices`` with
    Kokoro's voices — and a bare ``/v1/audio/speech`` tries to LOAD Kokoro —
    contradicting the ``Audio mode: <alias>`` banner it printed at boot.

    Returns ``None`` (caller falls back to :data:`DEFAULT_TTS_ALIAS`) whenever
    nothing served resolves to a TTS registry entry: a text model running with
    ``--enable-audio``, a served STT-only model, or the API-only test config
    whose ``model_alias`` / ``model_name`` are both unset.
    """
    from ..audio.registry import resolve_audio_alias
    from ..config import get_config

    cfg = get_config()
    # Check the alias AND the HF id independently, not ``alias or name``: under
    # ``--served-model-name foo`` the alias is the operator's opaque gateway
    # name (which resolves to nothing) while the real TTS model lives on
    # ``model_name``. Short-circuiting on the truthy-but-unrecognised alias
    # would wrongly fall back to Kokoro. First TTS hit wins.
    for served in (getattr(cfg, "model_alias", None), getattr(cfg, "model_name", None)):
        if not served:
            continue
        entry = resolve_audio_alias(served)
        if entry is not None and entry.type == "tts":
            return served
    return None


def _resolve_tts_model(model: str | None) -> str:
    """Map a TTS request-time alias to its HF repo id.

    Recognises:

    * ``None`` / ``""`` / ``"default"`` → the served TTS model when this
      process was started to serve one (:func:`_served_tts_default`), else
      :data:`DEFAULT_TTS_ALIAS`'s mapped HF id (R-03 OpenAI-canonical
      placeholder). Honouring the served model first is what makes
      ``rapid-mlx serve qwen3-tts-voicedesign`` answer a bare request with
      that model instead of silently loading Kokoro.
    * A short alias listed in :data:`TTS_MODEL_ALIASES` → its mapped
      HF id.
    * Anything else → pass through verbatim (a HuggingFace repo id
      the client is opting in to). Pre-fix the handler accepted the
      same shape inline; promotion to a helper keeps the contract
      identical without re-implementing the rule.

    R8-H4 (Bo 0.8.9 dogfood): lookup is case-insensitive so the
    brief's literal ``"kokoro-82m-8bit"`` and the SDK-style
    ``"Kokoro-82M-8bit"`` land on the same HF repo as the lowercase
    short form. HF repo ids (anything containing ``/``) keep their
    case verbatim — the case-insensitive lookup only fires for the
    short alias table, never for passthrough.
    """
    if not model or model == "default":
        served = _served_tts_default()
        if served is not None:
            # ``served`` is a concrete alias / HF id, never a placeholder, so
            # this reuses the exact mapping rule below without recursing again.
            return _resolve_tts_model(served)
        return TTS_MODEL_ALIASES[DEFAULT_TTS_ALIAS]
    return TTS_MODEL_ALIASES.get(model.lower(), model)


def _resolve_default_voice_literal(model_name: str, voice: str) -> str:
    """R11-B-F2/F3 (Bo 0.8.12 dogfood): map the literal ``"default"``
    voice to the registry's ``default_voice`` for ``model_name``.

    Pre-fix the route silently accepted ``voice`` omitted (the omitted
    case falls back to the Pydantic default ``"af_heart"``, which is
    valid for Kokoro), but the literal string ``"default"`` —
    which is exactly what naive callers and several SDK code samples
    send when they don't pick a voice — was rejected by
    :func:`_allowed_voices_for`'s allowlist. The asymmetry was a UX
    trap: "I sent the obvious value and got 400; what am I supposed
    to send?" The behaviour was especially confusing for kokoro where
    the registry already advertises ``default_voice="af_heart"``.

    Resolution rule: when ``voice`` is the literal string ``"default"``
    AND the resolved audio entry exposes a ``default_voice``, replace
    the literal with the registry value. Otherwise the literal passes
    through untouched — :func:`_allowed_voices_for` will let it through
    for unknown-family engines (their voice list IS ``["default"]``)
    and reject it for known families that don't ship a default. The
    "voice omitted → use ``af_heart``" path is unaffected because
    Pydantic populates the field default BEFORE this resolver runs.

    Resolution is keyed on the same registry helper the audio mode
    boot uses (``resolve_audio_alias``), so a registered HF id (``mlx-
    community/Kokoro-82M-bf16``) and its short alias (``kokoro``) both
    land on the same default. Names the registry doesn't know about
    (``mlx-community/Some-Future-TTS``) pass through unchanged.
    """
    if voice != "default":
        return voice
    try:
        from ..audio.registry import resolve_audio_alias
    except Exception:  # noqa: BLE001
        return voice
    entry = resolve_audio_alias(model_name)
    if entry is None or entry.default_voice is None:
        return voice
    return entry.default_voice


def _allowed_voices_for(model_name: str) -> list[str]:
    """Return the voice set the route should accept for ``model_name``.

    R8-M4 (Bo 0.8.9 dogfood): pre-fix the route handed ``voice``
    straight to ``mlx_audio.load_safetensors`` which 500'd on the
    missing safetensors file. The pre-flight check fires post alias
    resolution so the rule is "voice valid for whichever model the
    resolver picked", not "voice valid for the literal user-supplied
    name" — that way both ``kokoro`` and the full HF id
    ``mlx-community/Kokoro-82M-bf16`` honour the same voice list.

    R11-B-F1 (Bo 0.8.12 dogfood): pre-fix the per-family branch hard-
    coded the voice list to ``["default"]`` for everything except
    kokoro / chatterbox. That collapsed two real bugs into one
    end-to-end 500:

    * VibeVoice ships per-language voice caches (``en-Grace_woman.
      safetensors``, ``en-Mike_man.safetensors``, the eight non-
      English ``Spk0/Spk1`` pairs) and NO ``default.safetensors``,
      so ``voice="default"`` 500'd in
      ``mlx_audio.tts.models.vibevoice.Model.load_voice``.
    * A real file name like ``en-Grace_woman`` 400'd here because
      the static list only contained ``"default"``.

    The fix is to enumerate the snapshot's ``voices/`` dir at
    request time and use THAT list — see
    :func:`vllm_mlx.audio.tts._list_snapshot_voices`. This applies
    uniformly to every TTS family: chatterbox / voxcpm / dia ship a
    single ``default.safetensors`` so the enumeration returns the
    same ``["default"]`` the pre-fix static list had; kokoro ships
    50+ voice files (the pre-fix static list listed only 11) so the
    enumeration is a strict superset.

    Fallback path: when the snapshot isn't cached locally (first
    request, fresh install) the enumeration returns ``[]`` and we
    fall back to the per-family static list. That way the FIRST
    ``/v1/audio/speech`` call — which triggers the snapshot download
    via ``load_model`` — still passes voice validation against the
    registry default. After the snapshot lands the next request
    validates against the true voice set. The registry's
    ``default_voice`` MUST be a real voice that exists in the
    upstream snapshot so the cold-start path doesn't 500.
    """
    # Lazy import: ``vllm_mlx.audio.tts`` transitively pulls ``numpy``
    # which the API-only test runners don't install. Same lazy pattern
    # the route uses elsewhere.
    from ..audio.tts import (
        CHATTERBOX_VOICES,
        KOKORO_VOICES,
        QWEN3_TTS_VOICEDESIGN_VOICES,
        QWEN3_TTS_VOICES,
        _list_snapshot_voices,
        is_indextts_model,
        is_qwen3_tts_model,
        is_qwen3_voicedesign_model,
    )

    # Preferred path: enumerate the snapshot. Returns ``[]`` if the
    # repo isn't cached yet (local-only lookup; no HTTP). Falling back
    # to the static list keeps the first request alive — the engine
    # will pull the snapshot on ``load_model`` and subsequent calls
    # then validate against the true set.
    dynamic = _list_snapshot_voices(model_name)
    if dynamic:
        return dynamic

    name_lower = model_name.lower()
    if "kokoro" in name_lower:
        return list(KOKORO_VOICES)
    if "chatterbox" in name_lower:
        return list(CHATTERBOX_VOICES)
    if is_qwen3_voicedesign_model(model_name):
        # Qwen3-TTS VoiceDesign has NO named speakers — ``voice`` is ignored
        # and the whole voice is authored via ``instruct``. Advertise the
        # ``describe`` sentinel (mirrors F5's ``clone``) rather than the
        # CustomVoice speaker set. Checked BEFORE the general qwen3-tts branch
        # because a VoiceDesign id also matches the Qwen3-TTS family. Uses the
        # SAME shared classifier the engine's ``_is_qwen3_voicedesign`` does so
        # the two can't disagree (a mismatch would validate a VoiceDesign
        # request against CustomVoice speakers, or vice versa). The registry
        # ``default_voice`` for the VoiceDesign aliases is this same sentinel
        # so the voice-omitted / cold-start path validates.
        return list(QWEN3_TTS_VOICEDESIGN_VOICES)
    if is_qwen3_tts_model(model_name):
        # Qwen3-TTS CustomVoice ships baked-in named speakers and no
        # ``voices/`` snapshot dir, so the enumeration above always
        # returns ``[]`` and we serve the documented speaker set. The
        # registry ``default_voice`` (``Serena``) is a member of this
        # list so the cold-start / voice-omitted path validates.
        return list(QWEN3_TTS_VOICES)
    if is_indextts_model(model_name):
        return ["clone"]
    if "f5-tts" in name_lower or "f5_tts" in name_lower:
        # F5 conditions on a reference waveform rather than a named
        # safetensors voice. ``clone`` is the registry's UI/API sentinel;
        # when no custom reference is supplied the engine uses its packaged
        # reference voice.
        return ["clone"]
    if "vibevoice" in name_lower:
        # Cold-start fallback for VibeVoice — the canonical English
        # default is ``en-Grace_woman`` (per the upstream repo's
        # voice manifest). Listed alongside the other English voices
        # so the 400 envelope's ``Available:`` preview is informative
        # even before the snapshot has been downloaded.
        return [
            "en-Grace_woman",
            "en-Mike_man",
            "en-Carter_man",
            "en-Davis_man",
            "en-Emma_woman",
            "en-Frank_man",
        ]
    # Unknown family — accept ``"default"`` (the catch-all the engine
    # falls back to in :meth:`TTSEngine.get_voices`) so callers
    # passing a HF id we don't have a voice list for can still drive
    # the engine. Rejecting here would prematurely close the door on
    # third-party engines mlx-audio supports but rapid-mlx doesn't
    # ship metadata for.
    return ["default"]


def _is_clone_capable_model(model_name: str) -> bool:
    """Whether ``model_name`` can clone a voice from an inline reference.

    Four TTS families condition synthesis on a ``ref_audio`` reference
    clip sent on ``/v1/audio/speech``:

    * **F5-TTS** — always conditions on a reference waveform.
    * **Chatterbox** — optionally clones the reference timbre on top of
      its default voice (its engine branch forwards ``ref_audio``).
    * **IndexTTS** — requires a reference clip and has no named speakers.
    * **Qwen3-TTS Base** — the ``...-Base-...`` repo clones a voice
      zero-shot from the reference. Its CustomVoice sibling does NOT
      clone: it keeps a predefined named speaker and ignores
      ``ref_audio``.

    The verdict MUST stay in lock-step with
    :meth:`vllm_mlx.audio.tts.TTSEngine._detect_family`: a model this
    gate deems clone-capable while the engine classifies into a
    non-cloning family would skip voice validation here yet drop
    ``ref_audio`` in the engine — a silent 200 with the wrong (default)
    voice. So the F5 and Chatterbox checks use the SAME whole-id
    substrings the engine uses (``f5-tts``/``f5_tts`` and ``chatterbox``),
    NOT a broad ``f5`` token that would catch an unrelated ``org/f5-foo``
    the engine treats as Kokoro.

    Qwen3-TTS Base is classified on the repo NAME (last path component)
    split into ``-``/``_`` tokens — mirroring the Base-reject guard so
    an org segment (``customvoice-org/...``) or an unrelated ``base``
    elsewhere in the path can't flip the verdict. A repo name containing
    ``qwen3-tts`` is necessarily a substring of the full id, so whenever
    this returns True for Base the engine also detects ``qwen3_tts`` and
    forwards the reference — the dangerous direction cannot occur.
    """
    name_lower = model_name.lower()
    if "f5-tts" in name_lower or "f5_tts" in name_lower:
        return True
    if "chatterbox" in name_lower:
        return True
    if "indextts" in name_lower or "index-tts" in name_lower:
        return True
    repo = model_name.rsplit("/", 1)[-1].lower()
    tokens = set(re.split(r"[-_]", repo))
    is_qwen3 = "qwen3-tts" in repo or "qwen3_tts" in repo
    return is_qwen3 and "base" in tokens and "customvoice" not in tokens


def _generate_speech_blocking(
    model_name: str,
    input_text: str,
    response_format: str,
    gen_kwargs: dict,
    ref_bytes: bytes | None,
    ref_text: str | None,
    sample_rate: int | None,
    channels: int | None,
) -> tuple[bytes, int, int]:
    """Load, synthesize, and encode speech without blocking the event loop."""
    global _tts_engine

    from ..audio.tts import TTSEngine

    if _tts_engine is None or _tts_engine.model_name != model_name:
        tts_candidate = TTSEngine(model_name)
        tts_candidate.load()
        _tts_engine = tts_candidate

    kwargs = dict(gen_kwargs)
    if ref_bytes is not None:
        from .._tempfile_safe import managed_tempfile_path

        with managed_tempfile_path(prefix="tts-ref-", suffix=".wav") as ref_path:
            with open(ref_path, "wb") as ref_file:
                ref_file.write(ref_bytes)
            kwargs["ref_audio"] = ref_path.path
            if ref_text is not None:
                kwargs["ref_text"] = ref_text
            audio = _tts_engine.generate(input_text, **kwargs)
    else:
        audio = _tts_engine.generate(input_text, **kwargs)
    from ..audio.output_format import convert_audio_output

    converted, output_rate, output_channels = convert_audio_output(
        audio.audio,
        audio.sample_rate,
        sample_rate=sample_rate,
        channels=channels,
    )
    if sample_rate is not None or channels is not None:
        # Converted output is normalized to sample-first layout. A no-op
        # preserves the backend object and its duration metadata verbatim.
        audio.audio = converted
        audio.sample_rate = output_rate
        audio.duration = len(converted) / output_rate
    return (
        _tts_engine.to_bytes(audio, format=response_format),
        output_rate,
        output_channels,
    )


@router.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(request: AudioSpeechRequest = Body(...)):
    """Generate speech from text (OpenAI TTS API compatible).

    R7-M8 (Bo 0.8.8 dogfood): Bind a Pydantic :class:`AudioSpeechRequest`
    body model so the route honors the OpenAI JSON-body contract AND
    so empty / blank ``input`` raises a 400 ``invalid_request_error``
    with ``param="input"`` BEFORE the synthesis engine runs.
    Pre-fix the handler declared each field as a bare query parameter,
    which meant:

    * The JSON body Bo (and every OpenAI SDK) sends was silently
      dropped — ``input`` always fell back to its empty-string default,
      so every request synthesized the empty phoneme list and 500'd
      with ``No audio generated``.
    * There was nowhere to attach a ``min_length=1`` constraint,
      so the conflation with "engine genuinely failed" was structural.

    F-D05: probe ``mlx_audio`` availability through the shared
    :func:`vllm_mlx.audio.probe.require_mlx_audio` helper so this
    route's 503 envelope matches ``/v1/audio/voices`` and
    ``/v1/audio/transcriptions``.

    R7-H3 (Bo 0.8.8 dogfood): the catch-all now logs at ``exception``
    level (full traceback) and surfaces an OpenAI-shape envelope with
    ``type="api_error"``, ``code="tts_generation_failed"``. Pre-fix
    the catch-all collapsed every backend failure to a single-line
    ``logger.error`` + bare-string ``detail`` — the operator couldn't
    diagnose the upstream ``mlx_audio==0.4.4`` istftnet regression
    because the traceback never reached the log.
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

    # Pydantic already validated min_length / non-blank — past this
    # point the input is safe to forward.
    #
    # An OMITTED ``model`` follows the served model, not the Pydantic
    # placeholder default ("kokoro"): serving qwen3-tts-voicedesign and posting
    # a bare request should render with THAT model, not silently load Kokoro.
    # Detected via ``model_fields_set`` — the same mechanism the voice default
    # below uses — so an EXPLICIT ``"model": "kokoro"`` on a non-Kokoro server
    # is still honoured, while the omitted case flows through
    # ``_resolve_tts_model(None)`` to :func:`_served_tts_default`.
    model = request.model if "model" in request.model_fields_set else None
    input_text = request.input
    voice = request.voice
    speed = request.speed
    response_format = request.response_format
    sample_rate = request.sample_rate
    channels = request.channels
    instructions = request.instructions
    voice_seed = request.voice_seed
    ref_audio = request.ref_audio
    ref_text = request.ref_text
    exaggeration = request.exaggeration

    try:
        from ..audio.tts import (
            UnsupportedAudioFormatError,
            is_indextts_model,
            is_kokoro_family_model,
        )

        # R7-H3 follow-up: alias resolution lives in a shared helper
        # (see ``_resolve_tts_model``) so the bare alias / ``"default"``
        # / HF-path passthrough rule is one code path. Engines added in
        # the future land in :data:`TTS_MODEL_ALIASES` once, not in the
        # handler body.
        model_name = _resolve_tts_model(model)
        from ..audio.tts import is_qwen3_voicedesign_model

        if voice_seed is not None and not is_qwen3_voicedesign_model(model_name):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "voice_seed is supported only by "
                            "Qwen3-TTS VoiceDesign models"
                        ),
                        "type": "invalid_request_error",
                        "code": "unsupported_voice_seed",
                        "param": "voice_seed",
                    }
                },
            )

        # Zero-shot cloning from an inline ``ref_audio`` reference clip is
        # only wired for the clone-capable families (F5-TTS, Chatterbox, and
        # Qwen3-TTS Base — see ``_is_clone_capable_model``). A reference clip
        # aimed at any other model is rejected up front so the caller gets an
        # actionable 400 rather than the engine ignoring the clip and silently
        # synthesizing a default voice. (The shared ``AudioSpeechRequest``
        # validator still requires ``ref_audio``/``ref_text`` as a pair — the
        # F5 invariant — so a Chatterbox clone request supplies both even
        # though the Chatterbox engine branch consumes only the audio.)
        clone_capable = _is_clone_capable_model(model_name)
        inline_clone = ref_audio is not None and clone_capable
        if ref_audio is not None and not clone_capable:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            "ref_audio/ref_text voice cloning requires a "
                            "clone-capable model (F5-TTS, Chatterbox, "
                            "IndexTTS, or Qwen3-TTS Base)."
                        ),
                        "type": "invalid_request_error",
                        "code": "unsupported_voice_cloning",
                        "param": "model",
                    }
                },
            )

        if is_indextts_model(model_name) and ref_audio is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"model {model_name!r} is an IndexTTS "
                            "voice-cloning-only repo. Supply ref_audio with a "
                            "clean reference speech clip; ref_text is optional."
                        ),
                        "type": "invalid_request_error",
                        "code": "missing_reference_audio",
                        "param": "ref_audio",
                    }
                },
            )

        # Qwen3-TTS ships two shapes: CustomVoice (predefined speakers,
        # reference-free) and Base (voice-cloning ONLY, requires a reference
        # clip). A Base repo WITH an inline ``ref_audio`` is the correct
        # clone target and is served below. A Base repo WITHOUT a reference
        # cannot synthesize reference-free — it would otherwise reach the
        # CustomVoice speaker-validation + generate path and fail deep in
        # the engine ("Must provide one of ref_audio or ref_mel") as an
        # opaque 500. Reject the reference-free case up front with an
        # actionable 400 that points at both remedies (supply a reference,
        # or use a CustomVoice repo). Classify on the REPO NAME (last path
        # component), split into ``-``/``_``-delimited tokens, not a
        # whole-id substring: an org like ``customvoice-org/...`` or an
        # unrelated ``base`` elsewhere in the path must not flip the
        # decision, and a ``base`` token must be caught wherever it sits
        # (start/middle/end, hyphen or underscore delimited) —
        # ``...-0.6B-Base``, ``..._base_bf16``, etc.
        _repo = model_name.rsplit("/", 1)[-1].lower()
        _tokens = set(re.split(r"[-_]", _repo))
        _is_qwen3 = "qwen3-tts" in _repo or "qwen3_tts" in _repo
        if (
            _is_qwen3
            and "base" in _tokens
            and "customvoice" not in _tokens
            and ref_audio is None
        ):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": (
                            f"model {model_name!r} is a Qwen3-TTS Base "
                            "(voice-cloning-only) repo: it cannot synthesize "
                            "reference-free. Supply ref_audio (a clean 5-10s "
                            "reference clip) plus ref_text (its transcript) to "
                            "clone that voice, or use a CustomVoice repo (e.g. "
                            "the `qwen3-tts` alias) for reference-free "
                            "synthesis with a predefined speaker."
                        ),
                        "type": "invalid_request_error",
                        "code": "unsupported_model_variant",
                        "param": "model",
                    }
                },
            )

        # Voice selection + the speaker allowlist govern reference-free
        # (named-speaker) synthesis only. An inline-clone request is driven
        # by the reference clip, not a named speaker: a clone-capable Base
        # repo advertises the ``"clone"`` sentinel as its registry
        # ``default_voice`` (absent from the speaker allowlist), and F5
        # conditions on the waveform too — so running the allowlist here
        # would 400 the very request we intend to serve. Skip voice
        # resolution + validation entirely for a clone and OMIT ``voice``
        # from the generate call below — the timbre comes from ``ref_audio``.
        if not inline_clone:
            # R11-B-F3 (Bo 0.8.12 dogfood, PR #863): translate the literal
            # ``voice="default"`` to the registry's ``default_voice`` BEFORE
            # the allowlist check below. Pre-fix the obvious naive caller
            # value (``"default"``) was rejected by the kokoro allowlist
            # even though the registry already advertises
            # ``default_voice="af_heart"`` for it.
            #
            # R11-B-F1 (Bo 0.8.12 dogfood, this PR): the same resolver also
            # fires when ``voice`` was OMITTED from the JSON body. The
            # Pydantic model defaults ``voice`` to ``"af_heart"`` (kokoro's
            # canonical voice) for OpenAI-SDK parity. That default is
            # correct for kokoro but wrong for VibeVoice (no
            # ``af_heart.safetensors``) and Chatterbox/VoxCPM/Dia (expect
            # ``"default"``). The omitted-voice shape arrives here as
            # ``voice="af_heart"`` and 400'd against every non-kokoro
            # family. Treat the omitted-voice case the same way as the
            # literal ``"default"`` sentinel — both resolve to the registry
            # default. A client that EXPLICITLY sends ``voice="af_heart"``
            # against vibevoice keeps that value (the validator then
            # surfaces the 400 with the real available list).
            voice_omitted = "voice" not in request.model_fields_set
            if voice_omitted:
                voice = "default"
            voice = _resolve_default_voice_literal(model_name, voice)

            # R8-M4 (Bo 0.8.9 dogfood): validate ``voice`` against the
            # model's known voice set BEFORE we load weights. Pre-fix an
            # unknown name (drop-in OpenAI SDK code sending ``alloy`` /
            # ``nova`` / typo'd ``af_hart``) fell through to
            # ``mlx_audio.load_safetensors`` which 500'd on the missing
            # ``voices/<name>.safetensors`` file. The 500 envelope hid the
            # actual cause from the operator log AND from the caller. The
            # check fires post-resolution so a HF passthrough id (``mlx-
            # community/Kokoro-82M-bf16``) honours the same voice set as
            # the ``kokoro`` short alias — both go through the same model
            # family check.
            valid_voices = _allowed_voices_for(model_name)
            # Qwen3-TTS matches speaker names case-INsensitively (its engine
            # lowercases before the ``spk_id`` lookup) and the upstream docs
            # mix case ("serena" vs "Serena"). Normalize a case-insensitive
            # hit to the canonical spelling so ``serena`` / ``ono_anna``
            # aren't rejected as ``invalid_voice``; the engine then receives
            # the canonical form. Other families keep exact-match validation.
            if "qwen3-tts" in model_name.lower() or "qwen3_tts" in model_name.lower():
                _canonical = {v.lower(): v for v in valid_voices}
                voice = _canonical.get(voice.lower(), voice)
            if voice not in valid_voices:
                preview = ", ".join(valid_voices[:8])
                if len(valid_voices) > 8:
                    preview = f"{preview}, ..."
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": (
                                f"voice {voice!r} not recognized for model "
                                f"{model_name!r}. Available: {preview}."
                            ),
                            "type": "invalid_request_error",
                            "code": "invalid_voice",
                            "param": "voice",
                        }
                    },
                )

        # F-K-KOKORO-MISAKI: Kokoro pulls ``misaki`` lazily inside
        # ``KokoroPipeline``; the TTS-lane probe above can't catch
        # the missing extra because ``mlx_audio.tts.generate``
        # imports cleanly without it. Gate the missing-extra at
        # this boundary so the request 503s with a clean envelope
        # BEFORE any weight load (mlx-community/Kokoro-82M-bf16 is
        # ~300 MB) or pipeline construction kicks off.
        #
        # Gate on the engine's OWN family classifier, not a bare
        # ``"kokoro" in name`` substring: ``_detect_family`` DEFAULTS
        # every otherwise-unrecognized model to Kokoro, so a renamed /
        # third-party Kokoro repo would bypass a substring gate and hit
        # misaki's raw crash / uncaught ``SystemExit`` on first generate
        # (#1254). The classifier covers those too.
        if is_kokoro_family_model(model_name):
            # F-K-KOKORO-ESPEAK: the espeak readiness probe spawns
            # subprocesses and can block for seconds on a cold worker —
            # offload it so the async event loop stays responsive to other
            # in-flight requests (codex MAJOR).
            from fastapi.concurrency import run_in_threadpool

            # Pass the resolved voice so the spaCy en_core_web_sm gate applies
            # only to English voices (#1254) — Japanese/Mandarin/etc. Kokoro
            # voices use their own G2P and must not be forced through it.
            await run_in_threadpool(require_kokoro_runtime, voice)

        # Only forward ``instruct`` when the caller actually sent an
        # ``instructions`` field. Passing ``instruct=None`` is a no-op for
        # the real engine, but omitting the kwarg entirely keeps the call
        # shape backward-compatible with any generate() that predates the
        # emotion parameter (only Qwen3-TTS consumes it).
        #
        # OMIT ``voice`` for an inline clone — the reference clip selects
        # the timbre and the Base model ignores ``voice`` when a reference
        # is set (F5 has no named-speaker surface at all). Forwarding the
        # ``"clone"`` sentinel or a stray named speaker would be meaningless
        # and, for a strict engine, could raise. Reference-free synthesis
        # keeps the resolved named speaker.
        gen_kwargs = (
            {"speed": speed} if inline_clone else {"voice": voice, "speed": speed}
        )
        if instructions:
            gen_kwargs["instruct"] = instructions
        if voice_seed is not None:
            gen_kwargs["voice_seed"] = voice_seed
        # Only forward ``exaggeration`` when the caller actually sent it, so
        # the engine's own default holds otherwise. Like ``instruct`` it is
        # a no-op keyword for every non-Chatterbox family (``generate``
        # accepts it but only the Chatterbox branch forwards it to the
        # model), so a caller may send it against any model without a 400 —
        # matching OpenAI's ignore-unsupported-styling behaviour.
        if exaggeration is not None:
            gen_kwargs["exaggeration"] = exaggeration
        ref_bytes = None
        if ref_audio is not None:
            try:
                ref_bytes = _decode_tts_ref_audio(ref_audio)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                            "code": "invalid_ref_audio",
                            "param": "ref_audio",
                        }
                    },
                ) from exc
        try:
            async with _get_tts_lane_lock():
                audio_bytes, output_rate, output_channels = await run_to_completion(
                    _generate_speech_blocking,
                    model_name,
                    input_text,
                    response_format,
                    gen_kwargs,
                    ref_bytes,
                    ref_text,
                    sample_rate,
                    channels,
                )
        except UnsupportedAudioFormatError as e:
            # R8-H5 (Bo 0.8.9 dogfood): the encoder couldn't produce the
            # requested format (no codec / unknown name). Surface a 400
            # ``invalid_request_error`` with ``param="response_format"``
            # and the list of formats this build DOES support so the
            # caller can retry with a known-good value. Pre-fix the
            # route returned 200 with mislabeled WAV bytes.
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": "invalid_response_format",
                        "param": "response_format",
                    }
                },
            )

        # R8-H5: pick a Content-Type that actually matches the produced
        # bytes. Pre-fix the route blindly built ``audio/{format}``,
        # which both mislabelled the body (every non-wav format was
        # WAV) AND emitted non-canonical types (``audio/opus`` instead
        # of ``audio/ogg``). The mapping below pairs each
        # ``response_format`` with the IANA-canonical container type.
        content_type = _TTS_CONTENT_TYPES.get(
            response_format.lower(), "application/octet-stream"
        )
        headers = {
            "X-Audio-Sample-Rate": str(output_rate),
            "X-Audio-Channels": str(output_channels),
        }
        if voice_seed is not None:
            headers["X-Voice-Seed"] = str(voice_seed)
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers=headers,
        )

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
        # R7-H3: ``logger.exception`` writes the FULL traceback to the
        # operator log so future regressions in mlx_audio (or any other
        # upstream backend) leave enough breadcrumbs to root-cause from
        # the log alone. The pre-fix ``logger.error(f"...: {e}")`` only
        # captured the leaf exception's str() — operators chasing the
        # 0.4.4 istftnet shape mismatch never saw which istftnet line
        # raised, only the catch-all's ``No audio generated`` (which was
        # in fact the inner ``tts.py`` raise, NOT the upstream
        # broadcast_shapes error). Two-source confusion that the
        # traceback solves on its own.
        logger.exception("TTS generation failed: %s", e)
        # Replace the legacy bare-string ``detail`` with the OpenAI
        # envelope so clients can pattern-match on
        # ``error.type=="api_error"`` and ``error.code=="tts_generation_failed"``.
        # Mirrors the transcriptions ``code="transcription_failed"``
        # convention so cross-lane error handling stays uniform.
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Audio speech synthesis failed",
                    "type": "api_error",
                    "code": "tts_generation_failed",
                    "param": None,
                }
            },
        )


# ---------------------------------------------------------------------------
# Third audio lane: text→music / text→SFX (``/v1/audio/music``).
#
# Wired to ``vllm_mlx.audio.music.MusicEngine`` (MLX-native Stable Audio
# 3). OpenAI-flavored: JSON body in, WAV bytes out — the same
# request-in / audio-bytes-out shape as ``/v1/audio/speech``. The
# ``model`` field selects a DiT/decoder pairing via the alias table
# below; unknown values are rejected as caller errors.
# ---------------------------------------------------------------------------

#: ``model`` alias → ``(dit, decoder)`` pairing for SA3. ``medium`` is
#: higher quality (~3.9 GB peak); ``sm-music`` / ``sm-sfx`` are the fast
#: small variants (~1.7 GB, ~4x realtime). Kept local to the route (not
#: in the audio registry, whose closed schema is stt/tts-typed) so adding
#: a music variant is a one-line edit here. ``default`` maps to the
#: engine's own defaults.
MUSIC_MODEL_ALIASES: dict[str, tuple[str, str]] = {
    "medium": ("medium", "same-l"),
    "same-l": ("medium", "same-l"),
    "sm-music": ("sm-music", "same-s"),
    "sm-sfx": ("sm-sfx", "same-s"),
    "same-s": ("sm-music", "same-s"),
    "default": ("medium", "same-l"),
}

#: Defaults used when ``model`` is omitted or explicitly ``"default"``.
DEFAULT_MUSIC_DIT_DECODER: tuple[str, str] = ("medium", "same-l")


#: Serialises music generation. ``MusicEngine.generate`` shells out to the
#: vendored SA3 CLI, which peaks at ~3.9 GB (``medium``) — two concurrent
#: requests would run two such subprocesses and can exhaust unified memory
#: on a base-config Mac. Also guards the ``_music_engine`` global, mutated
#: from a worker thread.
#:
#: An ``asyncio.Lock`` acquired ON THE EVENT LOOP, deliberately not a
#: ``threading.Lock`` held inside the worker. ``asyncio.to_thread`` uses
#: the shared default executor (min(32, cpu+4) threads), so queued
#: requests blocking on a thread-level lock would each pin an executor
#: thread just to wait — starving every other ``to_thread`` user in the
#: process (prefix-cache save, tool-grammar warmup, the diffusion lane).
#: With a 900 s render ceiling that is a long time to hold threads other
#: subsystems need. Waiting on the loop instead costs a coroutine, and
#: only the request that actually runs ever occupies an executor slot.
#:
def _generate_music_blocking(
    dit: str,
    decoder: str,
    out_path: str,
    request: AudioMusicRequest,
) -> None:
    """Blocking half of ``/v1/audio/music`` — runs on a worker thread.

    ``MusicEngine.generate`` spawns the vendored Stable Audio 3 CLI and
    waits on it (default timeout 900 s). Calling that inline from the
    ``async def`` handler would stall the event loop for the entire
    render — every concurrent chat completion, ``/healthz`` probe and SSE
    heartbeat with it — so the handler hands it to
    :func:`asyncio.to_thread`.

    The caller holds :data:`_music_lock` for the whole call, so this body
    is already serialised — no thread-level locking here (see the lock's
    own comment for why it lives on the event loop).
    """
    global _music_engine

    from ..audio.music import MusicEngine

    if (
        _music_engine is None
        or _music_engine.dit != dit
        or _music_engine.decoder != decoder
    ):
        _music_engine = MusicEngine(dit=dit, decoder=decoder)

    _music_engine.generate(
        request.input,
        out_path,
        seconds=request.seconds,
        steps=request.steps,
        negative_prompt=request.negative_prompt,
        seed=request.seed,
    )


def _wav_has_audio_frames(payload: bytes) -> bool:
    """True if ``payload`` is a parseable WAV with at least one sample frame.

    Rejects both empty-output failure modes: SA3 exiting 0 having written
    only a RIFF header (~44 bytes, which passes a non-empty-bytes check
    but contains no audio), and output that isn't a readable WAV at all.

    Fail-CLOSED on a parse error, because the producer and this check use
    the same parser: SA3's ``save_wav`` writes 16-bit PCM via
    ``wave.open`` (see ``audio/sa3/scripts/sa3_mlx.py``). So anything
    ``wave`` can't read is not SA3 output, and handing it back under
    ``Content-Type: audio/wav`` would be mislabelling bytes rather than
    tolerating an exotic container.
    """
    try:
        with wave.open(io.BytesIO(payload), "rb") as w:
            return w.getnframes() > 0
    except (wave.Error, EOFError, OSError):
        return False


def _convert_music_wav(
    payload: bytes,
    sample_rate: int | None,
    channels: int | None,
) -> tuple[bytes, int, int]:
    """Convert SA3's PCM WAV while preserving its native format by default."""
    with wave.open(io.BytesIO(payload), "rb") as source:
        source_rate = source.getframerate()
        source_channels = source.getnchannels()
    if sample_rate is None and channels is None:
        # Keep the backend's byte-for-byte output, including any metadata
        # chunks a future SA3 encoder adds. Conversion is strictly opt-in.
        return payload, source_rate, source_channels

    import scipy.io.wavfile as wav

    from ..audio.output_format import convert_audio_output

    decoded_rate, encoded = wav.read(io.BytesIO(payload))
    if encoded.dtype == "int16":
        audio = encoded.astype("float32") / 32768.0
    elif encoded.dtype == "float32":
        audio = encoded
    else:
        raise ValueError(f"unsupported music WAV sample type: {encoded.dtype}")
    converted, output_rate, output_channels = convert_audio_output(
        audio,
        int(decoded_rate),
        sample_rate=sample_rate,
        channels=channels,
    )
    output = io.BytesIO()
    wav.write(
        output,
        output_rate,
        (converted * 32767.0).round().astype("int16"),
    )
    return output.getvalue(), output_rate, output_channels


def _resolve_music_model(model: str | None) -> tuple[str, str]:
    """Map a ``/v1/audio/music`` ``model`` alias to a ``(dit, decoder)`` pair.

    Recognises the :data:`MUSIC_MODEL_ALIASES` short names
    (case-insensitively). ``None`` / ``""`` / ``"default"`` select the
    defaults. Unknown values are rejected so a typo cannot silently select
    a different, substantially larger model.
    """
    if not model:
        return DEFAULT_MUSIC_DIT_DECODER
    alias = model.lower()
    if alias not in MUSIC_MODEL_ALIASES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": (
                        f"Unknown music model {model!r}. Available: "
                        f"{', '.join(sorted(MUSIC_MODEL_ALIASES))}."
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_model",
                    "param": "model",
                }
            },
        )
    return MUSIC_MODEL_ALIASES[alias]


@router.post("/v1/audio/music", dependencies=[Depends(verify_api_key)])
async def create_music(request: AudioMusicRequest = Body(...)):
    """Generate music / SFX from a text prompt.

    OpenAI-flavored: a JSON body (:class:`AudioMusicRequest`) in, WAV
    bytes out — the same request-in / audio-bytes-out contract as
    ``/v1/audio/speech``. Wired to
    :class:`vllm_mlx.audio.music.MusicEngine` (MLX-native Stable Audio
    3), which renders ``seconds`` of audio for ``input`` to a temp file
    that we stream back as ``audio/wav``.

    ``model`` selects the DiT/decoder pairing (see
    :data:`MUSIC_MODEL_ALIASES`); ``input`` (non-blank), ``seconds``
    (≤47s), ``steps``, ``negative_prompt`` and ``seed`` are validated by
    the request model. Backend failures surface a 500 with the OpenAI
    ``api_error`` envelope (``code="music_generation_failed"``), matching
    the speech route.
    """
    dit, decoder = _resolve_music_model(request.model)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name

        # Engine load + render are blocking (a subprocess, up to 900 s) —
        # off the event loop. Serialised on the loop before offloading so
        # queued renders don't each pin a shared executor thread.
        # See _generate_music_blocking and _music_lock.
        async with _get_music_lock():
            await run_to_completion(
                _generate_music_blocking, dit, decoder, tmp_path, request
            )

        audio_bytes = b""
        if os.path.exists(tmp_path):
            with open(tmp_path, "rb") as fh:
                audio_bytes = fh.read()

        # The SA3 CLI can exit 0 having written nothing, a 0-byte
        # placeholder, or a valid RIFF header with zero sample frames —
        # a sampler that bailed right after opening the file. All three
        # must fail loudly: handing back an HTTP 200 with an empty or
        # frameless ``audio/wav`` body reads to the caller as a
        # successfully generated silent clip, which is the failure mode
        # hardest to notice. A byte-count check alone misses the
        # header-only case (a bare WAV header is ~44 bytes), so count
        # actual frames.
        if not audio_bytes or not _wav_has_audio_frames(audio_bytes):
            raise RuntimeError(
                f"music engine produced no audio at {tmp_path} "
                f"(dit={dit!r}, decoder={decoder!r}, "
                f"{len(audio_bytes)} bytes)"
            )

        audio_bytes, output_rate, output_channels = await run_to_completion(
            _convert_music_wav,
            audio_bytes,
            request.sample_rate,
            request.channels,
        )

        # SA3 emits WAV; ``response_format`` is validated to ``wav`` by
        # the request model, so the Content-Type is always ``audio/wav``.
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Sample-Rate": str(output_rate),
                "X-Audio-Channels": str(output_channels),
            },
        )

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"music engine dependencies unavailable at runtime: {e}. "
                "Install with: pip install 'rapid-mlx[audio]'"
            ),
        )
    except Exception as e:
        # Mirror the speech route: full traceback to the operator log,
        # generic OpenAI-shape envelope to the client so we don't leak
        # subprocess/filesystem internals.
        logger.exception("Music generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Audio music generation failed",
                    "type": "api_error",
                    "code": "music_generation_failed",
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
                    "Failed to unlink temp music file %s: %s", tmp_path, cleanup_err
                )


@router.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: str | None = None):
    """List available voices for a TTS model.

    When ``model`` is omitted the listing follows the model this server is
    actually serving (:func:`_served_tts_default`), falling back to
    :data:`DEFAULT_TTS_ALIAS` only when no TTS model is served. Pre-fix the
    query defaulted to the literal ``"kokoro"``, so a server started on a
    different TTS model advertised Kokoro's voices here and then 400'd with
    ``invalid_voice`` when the caller sent one to ``/v1/audio/speech``.

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

    # R11-B-F1: route the listing through the SAME helper the
    # speech-route's voice validator uses, so a snapshot that ships
    # ``en-Grace_woman.safetensors`` doesn't show up as ``["default"]``
    # on ``/v1/audio/voices`` and then 400 with ``invalid_voice`` on
    # ``/v1/audio/speech``. The helper resolves the alias to its HF id
    # via the registry and falls back to the per-family static list
    # when the snapshot isn't cached locally — same contract as the
    # speech route.
    # ``None`` / ``""`` / ``"default"`` are all the omitted-model case — the
    # same placeholder set ``_resolve_tts_model`` collapses — so an explicit
    # ``?model=default`` selects the served model here exactly as it does on
    # /v1/audio/speech, rather than being handed to ``_allowed_voices_for``
    # verbatim.
    if not model or model == "default":
        model = _served_tts_default() or DEFAULT_TTS_ALIAS
    return {"voices": _allowed_voices_for(model)}
