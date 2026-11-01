# SPDX-License-Identifier: Apache-2.0
"""R10-C1 audio model registry — single source of truth.

Pre-fix the audio alias surface was fragmented across three places:

* ``rapid_mlx.routes.audio.STT_MODEL_ALIASES`` / ``TTS_MODEL_ALIASES`` —
  request-time resolution for the route handlers.
* ``rapid_mlx.audio.probe._AUDIO_ALIAS_TOKENS`` — boot-guard substring
  classifier.
* ``rapid_mlx.cli.serve_command`` — no resolution at all; the alias
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
import re
import threading
from dataclasses import dataclass
from typing import Literal

from rapid_mlx.audio.runtime_catalog import SPACY_PIPELINE_PINS

logger = logging.getLogger(__name__)

AudioType = Literal["tts", "stt"]
AudioRuntimeRequirementKind = Literal["spacy_pipeline"]


@dataclass(frozen=True)
class AudioRuntimeAsset:
    """One external repository required by an audio family at inference time.

    Some audio runtimes keep reusable assets (for example voice packs) in a
    repository separate from the quantized checkpoint.  Keeping that
    relationship in the catalog lets ``rapid-mlx pull`` prepare a genuinely
    offline-runnable alias without teaching the downloader about individual
    model families.
    """

    repo_id: str
    allow_patterns: tuple[str, ...]
    revision: str | None = None


@dataclass(frozen=True)
class AudioRuntimeRequirement:
    """One typed environment requirement prepared by ``rapid-mlx pull``.

    The catalog stores data, not commands.  A closed ``kind`` set keeps the
    preparation surface auditable and prevents model metadata from becoming an
    arbitrary package-install hook.
    """

    kind: AudioRuntimeRequirementKind
    name: str


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
    revision: str | None = None
    catalog_hf_id: str | None = None


_LOCAL_MODEL_TYPE_FAMILIES = {"voxcpm2": "voxcpm", "chatterbox_turbo": "chatterbox"}


# Lazy cache. Populated on first call to :func:`_load_registry`. Reset
# only by re-running in a fresh process (e.g. per-test subprocess).
_REGISTRY: dict[str, AudioAliasEntry] | None = None
# Reverse index: HF id (lowercase) -> alias key. Built alongside the
# forward index so :func:`resolve_audio_alias` can answer for full HF
# ids the same way it answers for short aliases.
_HF_ID_INDEX: dict[str, str] = {}
_WHISPER_PROCESSORS: dict[str, AudioRuntimeAsset] = {}
# Transformers 5 processor loading reads the repo config.json. Optional probes
# for processor_config.json, chat_template.json, chat_template.jinja, and
# audio_tokenizer_config.json are absent in all pinned OpenAI Whisper repos
# and are tolerated when uncached.
_WHISPER_PROCESSOR_PATTERNS = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "config.json",
)
_RUNTIME_ASSETS: dict[str, tuple[AudioRuntimeAsset, ...]] = {}
_RUNTIME_REQUIREMENTS: dict[str, tuple[AudioRuntimeRequirement, ...]] = {}
_REGISTRY_LOCK = threading.Lock()
_LOCAL_AUDIO_ENTRIES: dict[str, AudioAliasEntry] = {}


def _revision(entry: dict) -> str | None:
    pin = entry.get("revision")
    if pin is not None and (
        not isinstance(pin, str) or not re.fullmatch(r"[0-9a-f]{40}", pin)
    ):
        raise ValueError("audio catalog revision must be a full 40-hex commit SHA")
    return pin


def resolve_audio_revision(name: str) -> str | None:
    registry = _load_registry()
    key = name.lower()
    entry = registry.get(key) or registry.get(_HF_ID_INDEX.get(key, key))
    if entry is not None:
        return entry.revision
    for assets in _RUNTIME_ASSETS.values():
        for asset in assets:
            if asset.repo_id.lower() == name.lower():
                return asset.revision
    return None


def _registry_path() -> str:
    return os.path.join(os.path.dirname(__file__), "aliases.json")


def _load_registry() -> dict[str, AudioAliasEntry]:
    """Return the registry, initializing its complete snapshot once."""

    if _REGISTRY is not None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None:
            return _REGISTRY
        return _load_registry_uncached()


def _load_registry_uncached() -> dict[str, AudioAliasEntry]:
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

    path = _registry_path()
    with open(path) as f:
        raw = json.load(f)

    raw_processors = raw.get("_whisper_processors", {})
    if not isinstance(raw_processors, dict):
        raise ValueError("audio aliases.json: _whisper_processors must be an object")
    processors: dict[str, AudioRuntimeAsset] = {}
    for hf_id, asset in raw_processors.items():
        if (
            not isinstance(hf_id, str)
            or hf_id.count("/") != 1
            or not isinstance(asset, dict)
        ):
            raise ValueError("audio aliases.json: invalid Whisper processor entry")
        repo_id = asset.get("repo_id")
        revision = _revision(asset)
        if not isinstance(repo_id, str) or repo_id.count("/") != 1 or revision is None:
            raise ValueError(
                "audio aliases.json: Whisper processors require a repo and revision"
            )
        if hf_id.lower() in processors:
            raise ValueError("audio aliases.json: duplicate Whisper processor entry")
        processors[hf_id.lower()] = AudioRuntimeAsset(
            repo_id, _WHISPER_PROCESSOR_PATTERNS, revision
        )

    runtime_assets: dict[str, tuple[AudioRuntimeAsset, ...]] = {}
    raw_runtime_assets = raw.get("_runtime_assets", {})
    if not isinstance(raw_runtime_assets, dict):
        raise ValueError("audio aliases.json: _runtime_assets must be an object")
    for family, assets in raw_runtime_assets.items():
        if not isinstance(family, str) or not family:
            raise ValueError(
                "audio aliases.json: _runtime_assets family keys must be non-empty strings"
            )
        if not isinstance(assets, list):
            raise ValueError(
                f"audio aliases.json: _runtime_assets.{family} must be an array"
            )
        parsed: list[AudioRuntimeAsset] = []
        seen_repos: set[str] = set()
        for index, asset in enumerate(assets):
            if not isinstance(asset, dict):
                raise ValueError(
                    f"audio aliases.json: _runtime_assets.{family}[{index}] "
                    "must be an object"
                )
            repo_id = asset.get("repo_id")
            patterns = asset.get("allow_patterns")
            if not isinstance(repo_id, str) or repo_id.count("/") != 1:
                raise ValueError(
                    f"audio aliases.json: _runtime_assets.{family}[{index}].repo_id "
                    "must be a HuggingFace namespace/name"
                )
            if repo_id in seen_repos:
                raise ValueError(
                    f"audio aliases.json: duplicate runtime asset {repo_id!r} "
                    f"for family {family!r}"
                )
            if (
                not isinstance(patterns, list)
                or not patterns
                or not all(isinstance(pattern, str) and pattern for pattern in patterns)
            ):
                raise ValueError(
                    f"audio aliases.json: _runtime_assets.{family}[{index}]."
                    "allow_patterns must be an array of non-empty strings"
                )
            seen_repos.add(repo_id)
            parsed.append(AudioRuntimeAsset(repo_id, tuple(patterns), _revision(asset)))
        runtime_assets[family] = tuple(parsed)

    runtime_requirements: dict[str, tuple[AudioRuntimeRequirement, ...]] = {}
    raw_runtime_requirements = raw.get("_runtime_requirements", {})
    if not isinstance(raw_runtime_requirements, dict):
        raise ValueError("audio aliases.json: _runtime_requirements must be an object")
    for family, requirements in raw_runtime_requirements.items():
        if not isinstance(family, str) or not family:
            raise ValueError(
                "audio aliases.json: _runtime_requirements family keys must be "
                "non-empty strings"
            )
        if not isinstance(requirements, list):
            raise ValueError(
                f"audio aliases.json: _runtime_requirements.{family} must be an array"
            )
        parsed_requirements: list[AudioRuntimeRequirement] = []
        seen_requirements: set[tuple[str, str]] = set()
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, dict):
                raise ValueError(
                    f"audio aliases.json: _runtime_requirements.{family}[{index}] "
                    "must be an object"
                )
            kind = requirement.get("kind")
            name = requirement.get("name")
            if kind != "spacy_pipeline":
                raise ValueError(
                    f"audio aliases.json: _runtime_requirements.{family}[{index}]."
                    "kind must be 'spacy_pipeline'"
                )
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"audio aliases.json: _runtime_requirements.{family}[{index}]."
                    "name must be a Python package identifier"
                )
            if name not in SPACY_PIPELINE_PINS:
                raise ValueError(
                    "audio aliases.json: runtime requirement is not in the "
                    "code-owned pipeline catalog"
                )
            key = (kind, name)
            if key in seen_requirements:
                raise ValueError(
                    f"audio aliases.json: duplicate runtime requirement {kind}:{name} "
                    f"for family {family!r}"
                )
            seen_requirements.add(key)
            parsed_requirements.append(AudioRuntimeRequirement(kind=kind, name=name))
        runtime_requirements[family] = tuple(parsed_requirements)

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
            revision=_revision(value),
        )

    known_families = {entry.family for entry in entries.values()}
    unknown_asset_families = set(runtime_assets) - known_families
    if unknown_asset_families:
        names = ", ".join(sorted(unknown_asset_families))
        raise ValueError(
            f"audio aliases.json: runtime assets declared for unknown family: {names}"
        )
    unknown_requirement_families = set(runtime_requirements) - known_families
    if unknown_requirement_families:
        names = ", ".join(sorted(unknown_requirement_families))
        raise ValueError(
            "audio aliases.json: runtime requirements declared for unknown family: "
            f"{names}"
        )
    # Reverse index keyed on the lowercased HF id so ``serve_command``
    # can route a request like ``rapid-mlx serve mlx-community/Kokoro-
    # 82M-bf16`` directly back to its registry entry (HF id case varies
    # across mlx-community uploads).
    hf_id_index: dict[str, str] = {}
    for alias, entry in entries.items():
        hf_id_index.setdefault(entry.hf_id.lower(), alias)
    # Publish one coherent snapshot while holding _REGISTRY_LOCK.  Readers use
    # _REGISTRY as the ready flag, so it must become visible only after every
    # companion index contains the matching data.
    _WHISPER_PROCESSORS.clear()
    _WHISPER_PROCESSORS.update(processors)
    _HF_ID_INDEX.clear()
    _HF_ID_INDEX.update(hf_id_index)
    _RUNTIME_ASSETS.clear()
    _RUNTIME_ASSETS.update(runtime_assets)
    _RUNTIME_REQUIREMENTS.clear()
    _RUNTIME_REQUIREMENTS.update(runtime_requirements)
    _REGISTRY = entries
    return entries


def resolve_local_audio_folder(path: str) -> AudioAliasEntry | None:
    """Recognize a local audio folder from config metadata or its name."""
    try:
        if not os.path.isdir(path):
            return None
        try:
            with open(os.path.join(path, "config.json"), "rb") as config_file:
                if os.fstat(config_file.fileno()).st_size > 1024 * 1024:
                    return None
                data = config_file.read(1024 * 1024)
            config = json.loads(data)
            configless = False
        except FileNotFoundError:
            config = {}
            configless = True
        if not isinstance(config, dict):
            return None
        families = {
            e.family.lower().replace("_", "-"): e for e in _load_registry().values()
        }
        model_type = config.get("model_type")
        normalized_type = (
            model_type.lower().replace("-", "_") if isinstance(model_type, str) else ""
        )
        normalized_type = _LOCAL_MODEL_TYPE_FAMILIES.get(
            normalized_type, normalized_type
        )
        template = (
            families.get(normalized_type.replace("_", "-"))
            if isinstance(model_type, str)
            else None
        )
        matches = [template] if template else []
        if not matches:
            architectures = config.get("architectures")
            if not isinstance(architectures, list):
                architectures = []
            if not isinstance(model_type, str):
                architectures = [*architectures, config.get("architecture")]
            matches = [
                entry
                for family, entry in families.items()
                if any(
                    isinstance(arch, str)
                    and re.sub(r"[-_.]", "", arch.lower()).startswith(
                        family.replace("-", "")
                    )
                    for arch in architectures
                )
            ]
        if not matches and "model_type" not in config:
            tokens = re.split(
                r"[-_. ]+", os.path.basename(os.path.normpath(path)).lower()
            )
            matches = [
                entry
                for family, entry in families.items()
                for parts in [family.split("-")]
                if any(
                    tokens[i : i + len(parts)] == parts
                    for i in range(len(tokens) - len(parts) + 1)
                )
            ]
        if len(matches) != 1:
            return None
        template = matches[0]
        if configless and (
            template.family != "f5"
            or not os.path.isfile(os.path.join(path, "vocab.txt"))
            or not any(
                os.path.isfile(os.path.join(path, name))
                and name.endswith(".safetensors")
                for name in os.listdir(path)
            )
        ):
            return None
        if template.family == "qwen3_tts":
            subtype = config.get("tts_model_type", "")
            subtype = (
                re.sub(r"[-_ .]", "", subtype.lower())
                if isinstance(subtype, str)
                else ""
            )
            variants = ("customvoice", "base", "voicedesign")
            if subtype not in variants:
                folder = re.sub(
                    r"[-_ .]", "", os.path.basename(os.path.normpath(path)).lower()
                )
                subtype = next(
                    (variant for variant in variants if variant in folder), ""
                )
            if subtype:
                template = next(
                    (
                        entry
                        for entry in _load_registry().values()
                        if entry.family == "qwen3_tts"
                        and subtype
                        in re.split(r"[-_.]", entry.hf_id.rsplit("/", 1)[-1].lower())
                    ),
                    template,
                )
        real_path = os.path.realpath(path)
        alias = (
            re.sub(
                r"[^a-z0-9._-]", "-", os.path.basename(os.path.normpath(path)).lower()
            ).strip(".-_")
            or "local-audio"
        )
        return AudioAliasEntry(
            alias=alias,
            type=template.type,
            hf_id=real_path,
            family=template.family,
            default_voice=template.default_voice,
            languages=template.languages,
            notes="local folder",
            catalog_hf_id=template.hf_id,
        )
    except Exception:
        return None


def register_served_local_audio_entry(
    entry: AudioAliasEntry, path: str | None = None
) -> None:
    """Register the served folder for this process only."""
    catalog = _load_registry()
    if entry.alias.lower() not in catalog:
        _LOCAL_AUDIO_ENTRIES[entry.alias.lower()] = entry
    _LOCAL_AUDIO_ENTRIES[entry.hf_id] = entry
    if path:
        if path.lower() not in catalog:
            _LOCAL_AUDIO_ENTRIES[path] = entry
        _LOCAL_AUDIO_ENTRIES[os.path.abspath(path)] = entry


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
    local = _LOCAL_AUDIO_ENTRIES.get(name) or _LOCAL_AUDIO_ENTRIES.get(name.lower())
    if local is not None:
        return local
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
    :func:`rapid_mlx.audio.probe.is_audio_model_alias`).
    """
    return resolve_audio_alias(name) is not None


def list_audio_aliases() -> list[AudioAliasEntry]:
    """Return all audio aliases, sorted by name.

    Used by ``rapid-mlx models`` to render the audio section of the
    alias table. The sort is alphabetical so the ``kokoro*`` /
    ``whisper*`` / ``parakeet*`` groups cluster together visually.
    """
    return sorted(_load_registry().values(), key=lambda e: e.alias)


def runtime_assets_for(name: str | None) -> tuple[AudioRuntimeAsset, ...]:
    """Return catalog-declared external runtime assets for an audio name.

    ``name`` accepts the same short-alias and full-HF-id forms as
    :func:`resolve_audio_alias`.  Non-audio names intentionally return an
    empty tuple so the general pull path stays unchanged.
    """

    entry = resolve_audio_alias(name)
    if entry is None:
        return ()
    assets = _RUNTIME_ASSETS.get(entry.family, ())
    processor = (
        whisper_processor_asset(entry.hf_id) if entry.family == "whisper" else None
    )
    return assets + (processor,) if processor else assets


def whisper_processor_asset(hf_id: str) -> AudioRuntimeAsset | None:
    """Return the pinned processor for a registered Whisper model ID."""
    _load_registry()
    return _WHISPER_PROCESSORS.get(hf_id.lower())


def runtime_requirements_for(
    name: str | None,
) -> tuple[AudioRuntimeRequirement, ...]:
    """Return typed environment requirements declared for an audio name."""

    entry = resolve_audio_alias(name)
    if entry is None:
        return ()
    return _RUNTIME_REQUIREMENTS.get(entry.family, ())


def stt_aliases() -> dict[str, str]:
    """Return ``{alias: hf_id}`` for every STT entry.

    Used by :mod:`rapid_mlx.routes.audio` to build its
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
