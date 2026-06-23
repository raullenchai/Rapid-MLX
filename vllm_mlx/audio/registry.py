# SPDX-License-Identifier: Apache-2.0
"""R10-C1 audio model registry — single source of truth.

Pre-fix the audio alias surface was fragmented across three places:

* ``vllm_mlx.routes.audio.STT_MODEL_ALIASES`` / ``TTS_MODEL_ALIASES`` —
  request-time resolution for the route handlers.
* ``vllm_mlx.audio.probe._AUDIO_ALIAS_TOKENS`` — boot-guard substring
  classifier.
* ``vllm_mlx.cli.serve_command`` — no resolution at all; the alias
  fell through to ``_ensure_model_downloaded`` (HF 404) and then into
  ``mlx_lm.load_model`` (no safetensors). Every short alias 100% broken
  on 0.8.11 — Bo r10-R1 finding, predicted by codex r8-A r3.

R10-C1 consolidates the alias table into ``aliases.json`` and gives
every callsite the SAME lookup contract:

* :func:`resolve_audio_alias` — alias / HF id -> :class:`AudioAliasEntry`
  or ``None`` (for non-audio names).
* :func:`is_audio_name` — boolean form, used by ``serve_command`` to
  decide whether to fork into audio-mode.
* :func:`list_audio_aliases` — ordered alias listing for the
  ``rapid-mlx models`` table.

The registry is the ONLY place a new audio model lands. The route
alias tables (``STT_MODEL_ALIASES`` / ``TTS_MODEL_ALIASES``) are now
built from this registry at import time so a single JSON edit reaches
every consumer.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

AudioType = Literal["tts", "stt"]


@dataclass(frozen=True)
class AudioAliasEntry:
    """Resolved metadata for an audio alias.

    Fields:

    * ``alias`` — the registry key (short alias name, lowercase).
    * ``type`` — ``"tts"`` (text -> speech) or ``"stt"``
      (speech -> text). Drives the route binding decision in
      ``serve_command`` and the capability tag in ``rapid-mlx models``.
    * ``hf_id`` — the HuggingFace repo id the audio engine should
      load. Always single-slash org/name shape. Verified at registry
      introduction time via the HF API.
    * ``family`` — engine family (``kokoro`` / ``chatterbox`` /
      ``vibevoice`` / ``voxcpm`` / ``whisper`` / ``parakeet`` / ...).
      Used by the TTS lane to pick the voice list.
    * ``default_voice`` — TTS-only; first-choice voice for the engine.
      ``None`` for STT entries.
    * ``languages`` — STT-only; ``"multilingual"`` or a comma-separated
      ISO list. ``None`` for TTS entries.
    * ``notes`` — free-form operator-facing description (shown in
      ``rapid-mlx info <alias>``).
    """

    alias: str
    type: AudioType
    hf_id: str
    family: str
    default_voice: str | None = None
    languages: str | None = None
    notes: str = ""


# Lazy cache. Populated on first call to :func:`_load_registry`; reset
# via :func:`_reset_registry_cache` for tests that mutate the JSON.
_REGISTRY: dict[str, AudioAliasEntry] | None = None
# Reverse index: HF id (lowercase) -> alias key. Built alongside the
# forward index so :func:`resolve_audio_alias` can answer for full HF
# ids the same way it answers for short aliases.
_HF_ID_INDEX: dict[str, str] = {}


def _reset_registry_cache() -> None:
    """Test hook: clear the registry cache.

    Tests that swap the JSON file or patch the loader call this in
    their fixture so the next ``_load_registry`` call re-reads from
    disk. Production code never calls this.
    """
    global _REGISTRY
    _REGISTRY = None
    _HF_ID_INDEX.clear()


def _registry_path() -> str:
    return os.path.join(os.path.dirname(__file__), "aliases.json")


def _load_registry() -> dict[str, AudioAliasEntry]:
    """Parse ``aliases.json`` and return the alias -> entry map.

    The JSON file is committed alongside this module so the registry
    is read-only at runtime. Malformed entries fail fast at load time
    rather than at the first request — a typo'd ``hf_id`` would
    otherwise surface as a 404 deep in the audio loader.

    Keys beginning with ``_`` (e.g. ``_comment``) are skipped so the
    JSON file can carry inline documentation without polluting the
    alias surface.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    path = _registry_path()
    with open(path) as f:
        raw = json.load(f)

    entries: dict[str, AudioAliasEntry] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            # Inline doc / comment key — skip so it doesn't accidentally
            # register as an alias.
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"audio aliases.json: entry {key!r} must be an object, "
                f"got {type(value).__name__}"
            )
        try:
            kind = value["type"]
            hf_id = value["hf_id"]
            family = value["family"]
        except KeyError as e:
            raise ValueError(
                f"audio aliases.json: entry {key!r} missing required "
                f"field {e.args[0]!r}"
            ) from e
        if kind not in ("tts", "stt"):
            raise ValueError(
                f"audio aliases.json: entry {key!r} has invalid type "
                f"{kind!r}; must be 'tts' or 'stt'"
            )
        if "/" not in hf_id:
            raise ValueError(
                f"audio aliases.json: entry {key!r}.hf_id={hf_id!r} "
                "must be a HuggingFace ``org/name`` repo id"
            )
        entries[key] = AudioAliasEntry(
            alias=key,
            type=kind,
            hf_id=hf_id,
            family=family,
            default_voice=value.get("default_voice"),
            languages=value.get("languages"),
            notes=value.get("notes", ""),
        )

    _REGISTRY = entries
    # Reverse index keyed on the lowercased HF id so ``serve_command``
    # can route a request like ``rapid-mlx serve mlx-community/Kokoro-
    # 82M-bf16`` directly back to its registry entry (HF id case varies
    # across mlx-community uploads).
    _HF_ID_INDEX.clear()
    for alias, entry in entries.items():
        _HF_ID_INDEX.setdefault(entry.hf_id.lower(), alias)
    return entries


def resolve_audio_alias(name: str | None) -> AudioAliasEntry | None:
    """Return the registry entry for ``name``, or ``None`` if not audio.

    Resolution order (first hit wins):

    1. Direct short-alias lookup (case-insensitive) — ``kokoro``,
       ``whisper-large-v3``, ``parakeet-tdt-0.6b-v2``, ...
    2. Reverse HF-id lookup (case-insensitive) — ``mlx-community/
       Kokoro-82M-bf16`` returns the ``kokoro`` entry so audio-mode
       fires for full HF ids of audio models too. This is critical
       because users (and `rapid-mlx pull mlx-community/Kokoro-82M-bf16`
       output) routinely paste the full HF id into ``serve``.

    Non-audio names (text aliases, vision aliases, unknown HF ids)
    return ``None`` — the caller falls back to the text/vision path.

    Empty / non-string inputs short-circuit to ``None`` so callers
    don't need to defensively type-check before delegating.
    """
    if not isinstance(name, str) or not name:
        return None
    registry = _load_registry()
    lc = name.lower()
    # Short-alias direct hit.
    entry = registry.get(lc)
    if entry is not None:
        return entry
    # HF-id reverse lookup. ``_HF_ID_INDEX`` was populated alongside
    # ``_REGISTRY`` so the lookup is O(1) and case-insensitive.
    alias = _HF_ID_INDEX.get(lc)
    if alias is not None:
        return registry[alias]
    return None


def is_audio_name(name: str | None) -> bool:
    """Return True iff ``name`` resolves to a registered audio entry.

    Convenience wrapper around :func:`resolve_audio_alias` for the
    common boolean predicate. Used by ``serve_command`` to gate the
    audio-mode fork, and by the boot-time alias classifier as a
    REGISTRY-FIRST check (substring fallback still applies for HF ids
    of audio models that haven't been added to the registry yet —
    those route to the legacy substring path via
    :func:`vllm_mlx.audio.probe.is_audio_model_alias`).
    """
    return resolve_audio_alias(name) is not None


def list_audio_aliases() -> list[AudioAliasEntry]:
    """Return all audio aliases, sorted by name.

    Used by ``rapid-mlx models`` to render the audio section of the
    alias table. The sort is alphabetical so the ``kokoro*`` /
    ``whisper*`` / ``parakeet*`` groups cluster together visually.
    """
    return sorted(_load_registry().values(), key=lambda e: e.alias)


def stt_aliases() -> dict[str, str]:
    """Return ``{alias: hf_id}`` for every STT entry.

    Used by :mod:`vllm_mlx.routes.audio` to build its
    ``STT_MODEL_ALIASES`` table without duplicating the data. Bare
    ``dict`` rather than ``AudioAliasEntry`` so the route's existing
    consumers don't need to change shape.
    """
    return {e.alias: e.hf_id for e in _load_registry().values() if e.type == "stt"}


def tts_aliases() -> dict[str, str]:
    """Return ``{alias: hf_id}`` for every TTS entry.

    Counterpart to :func:`stt_aliases`. Same shape contract.
    """
    return {e.alias: e.hf_id for e in _load_registry().values() if e.type == "tts"}
