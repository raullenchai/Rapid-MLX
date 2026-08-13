# SPDX-License-Identifier: Apache-2.0
"""
Text-to-Speech (TTS) engine using mlx-audio.

Supports:
- Kokoro (fast, lightweight)
- Chatterbox (multilingual, expressive)
- IndexTTS (zero-shot voice cloning)
- VibeVoice (realtime, low latency)
- VoxCPM (Chinese/English, high quality)
"""

import importlib
import io
import json
import logging
import threading
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_QWEN_SAMPLER_PATCH_LOCK = threading.RLock()


@contextmanager
def _qwen_seeded_sampling(model, seed: int | None):
    """Give one Qwen generation a request-local RNG without global seeding."""
    if seed is None:
        yield
        return

    import mlx.core as mx

    module = importlib.import_module(type(model).__module__)
    original = module.categorical_sampling
    owner_thread = threading.get_ident()
    key = mx.random.key(seed)

    def dispatch(logits, temperature):
        nonlocal key
        if threading.get_ident() != owner_thread:
            return original(logits, temperature)
        keys = mx.random.split(key, 2)
        key, sample_key = keys[0], keys[1]
        return mx.random.categorical(logits * (1 / temperature), key=sample_key)

    # The public route already serializes TTS, while this lock also protects
    # callers embedding TTSEngine directly. Other inference threads may enter
    # the dispatcher concurrently; they fall through to the untouched sampler.
    with _QWEN_SAMPLER_PATCH_LOCK:
        module.categorical_sampling = dispatch
        try:
            yield
        finally:
            module.categorical_sampling = original


# Default models
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"

# F5-TTS normalizes the reference clip toward ``TARGET_RMS`` via
# ``aud * TARGET_RMS / rms``. A silent (rms==0) or near-silent reference would
# either divide by zero (NaN/inf) or get amplified by a huge factor, turning
# quantization noise / DC offset into full-scale garbage. Refs whose RMS falls
# below this floor (~-80 dBFS) are rejected with a clear error instead.
F5_MIN_REF_RMS = 1e-4

# Multi-channel references are downmixed to mono, but the channel count is left
# unbounded by the (frames-only) duration guard. Cap it so a pathological
# many-channel header can't force a disproportionately large decode/allocation
# before the downmix. Mono and stereo are the only realistic reference formats.
F5_MAX_REF_CHANNELS = 2

# Available voices per model family
KOKORO_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]

CHATTERBOX_VOICES = ["default"]  # Uses reference audio for voice

# Qwen3-TTS CustomVoice predefined speakers. The CustomVoice repos ship
# NO ``voices/`` snapshot dir (voices are baked-in named speakers, not
# per-voice safetensors), so ``_list_snapshot_voices`` returns ``[]`` and
# the route's ``_allowed_voices_for`` / ``get_voices`` fall back to this
# static list.
#
# The AUTHORITATIVE speaker set is the model config's
# ``talker_config.spk_id`` keys (see ``mlx_audio.tts.models.qwen3_tts``:
# ``supported_speakers = list(config.talker_config.spk_id.keys())``), which
# the engine matches case-INsensitively (``speaker.lower() in spk_id``).
# The upstream README under-documents it (lists only the Chinese + English
# speakers); the shipped ``1.7B-CustomVoice`` config actually carries nine,
# including the Japanese ``ono_anna`` and Korean ``sohee`` — omitting those
# would make the route 400 two valid voices. We list the canonical
# capitalized display forms here, grouped Chinese → English → JA → KO.
QWEN3_TTS_VOICES = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",  # Beijing dialect
    "Eric",  # Sichuan dialect
    "Ryan",
    "Aiden",
    "Ono_Anna",  # Japanese
    "Sohee",  # Korean
]

# Fallback voice description for the Qwen3-TTS *VoiceDesign* variant. Unlike
# CustomVoice (named speaker + optional ``instruct`` emotion), VoiceDesign has
# NO speakers — the whole voice is defined by a natural-language description
# carried in ``instruct``, which mlx_audio's ``generate_voice_design`` requires
# as a mandatory positional (no default). If a caller reaches a VoiceDesign
# model without supplying one we substitute this neutral narrator description
# rather than crash deep inside mlx_audio with a missing-arg TypeError.
QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT = (
    "A clear, natural narrator voice with a calm, neutral tone."
)

# Voice surface for the Qwen3-TTS *VoiceDesign* variant. VoiceDesign has NO
# named speakers — ``voice`` is ignored and the whole voice is authored in
# natural language via ``instruct``. Rather than advertise the nine CustomVoice
# speakers (which would mislead callers into thinking a picked speaker matters),
# it exposes a single ``describe`` sentinel, mirroring how the F5 family
# advertises ``clone`` for its reference-driven surface. The sentinel is also
# the registry ``default_voice`` for the VoiceDesign aliases so the
# voice-omitted / cold-start path validates without a real speaker name.
QWEN3_TTS_VOICEDESIGN_VOICES = ["describe"]
INDEXTTS_VOICES = ["clone"]


def is_indextts_model(model_name: str) -> bool:
    """True for IndexTTS checkpoints and aliases."""
    name_lower = model_name.lower()
    return "indextts" in name_lower or "index-tts" in name_lower


_INDEXTTS_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.model",
    "*.safetensors",
    "*.safetensors.index.json",
]


def _cached_snapshot_holds_indextts(snapshot: Path) -> bool:
    """True when a cached snapshot holds everything the IndexTTS load opens.

    ``huggingface_hub`` can only judge a local snapshot's completeness when it
    has a cached tree listing for the repo. A snapshot populated by the R2
    mirror has none, and is then returned sight-unseen — so check the files
    ourselves rather than accept a half-pulled checkpoint as usable.
    """

    def _present(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    if not all(
        _present(snapshot / name) for name in ("config.json", "tokenizer.model")
    ):
        return False

    index_path = snapshot / "model.safetensors.index.json"
    if _present(index_path):
        # Sharded checkpoint: the index names every shard the loader will open,
        # so "some safetensors exist" is not enough.
        try:
            with open(index_path) as index_file:
                weight_map = json.load(index_file).get("weight_map")
        except (OSError, json.JSONDecodeError, AttributeError):
            return False
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        shards = set(weight_map.values())
        if not all(isinstance(shard, str) for shard in shards):
            return False
        return all(
            Path(shard).name == shard and _present(snapshot / shard) for shard in shards
        )

    return any(_present(shard) for shard in snapshot.glob("*.safetensors"))


def _resolve_indextts_snapshot(model_name: str) -> Path:
    """Resolve the IndexTTS snapshot, preferring a warm cache over the network.

    ``snapshot_download`` resolves ``main`` → sha through the Hub on every call,
    even when every file is already on disk, and that lookup carries no timeout
    (``HfApi.repo_info`` passes ``timeout=None`` into an ``httpx.Client`` built
    with ``timeout=None``). On a poisoned-DNS network it therefore hangs in
    SYN_SENT rather than failing fast, stalling a load whose weights are already
    local. Try the cache first; reach for the network only when the cache cannot
    satisfy the load, so a cold or partial checkpoint still pulls as before.
    """
    from huggingface_hub import snapshot_download

    try:
        cached = Path(
            snapshot_download(
                model_name,
                allow_patterns=_INDEXTTS_ALLOW_PATTERNS,
                local_files_only=True,
            )
        )
    except Exception:
        cached = None
    if cached is not None and _cached_snapshot_holds_indextts(cached):
        return cached
    return Path(snapshot_download(model_name, allow_patterns=_INDEXTTS_ALLOW_PATTERNS))


def _load_indextts_model(model_name: str):
    """Load IndexTTS without mutating its incomplete cached config.

    The community checkpoints include ``tokenizer.model`` but omit the
    ``tokenizer_name`` field required by mlx-audio's ``ModelArgs``. Patch an
    in-memory copy and otherwise follow mlx-audio's normal load sequence.
    """
    import mlx.core as mx
    from mlx_audio.tts.models.indextts.indextts import Model

    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        model_path = local_path.resolve()
    else:
        model_path = _resolve_indextts_snapshot(model_name)
    with open(model_path / "config.json") as config_file:
        config = json.load(config_file)

    tokenizer_path = model_path / "tokenizer.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"IndexTTS tokenizer not found at {tokenizer_path}; "
            "the checkpoint must include tokenizer.model"
        )
    # mlx-audio treats tokenizer_name as a repo id first and a directory
    # second, appending ``tokenizer.model`` itself on the local fallback.
    config["tokenizer_name"] = str(model_path)

    model = Model(config)
    weights = {}
    for weight_path in sorted(model_path.glob("*.safetensors")):
        weights.update(mx.load(str(weight_path)))
    if not weights:
        raise FileNotFoundError(f"No IndexTTS safetensors found in {model_path}")
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def is_qwen3_tts_model(model_name: str) -> bool:
    """True for any Qwen3-TTS checkpoint (CustomVoice OR VoiceDesign).

    Shared classifier so the engine's family detection and the route's voice
    allowlist agree on exactly which ids are Qwen3-TTS. Matches the ``qwen3-tts``
    / ``qwen3_tts`` token anywhere in the id (the same full-id rule
    ``TTSEngine._detect_family`` has always used).
    """
    name_lower = model_name.lower()
    return "qwen3-tts" in name_lower or "qwen3_tts" in name_lower


def _qwen3_repo_component(model_name: str) -> str:
    """Extract the repo/model NAME component from any Qwen3-TTS identifier.

    The CustomVoice/VoiceDesign variant token lives in the repo NAME, so the
    variant check must look ONLY there — never at parent directories or the org
    namespace, which may coincidentally contain a token (``/srv/customvoice/
    Qwen3-TTS-VoiceDesign-bf16``, ``voicedesign-org/...-CustomVoice-...``).
    Handles the three id shapes the engine sees:

    * HuggingFace cache snapshot path — ``.../hub/models--<org>--<repo>/
      snapshots/<hash>``: decode the ``models--<org>--<repo>`` segment and
      return ``<repo>`` (the part after the LAST ``--``), so neither the
      ``snapshots/<hash>`` tail nor the org leaks in.
    * ``org/name`` HF id or a plain filesystem path — return the last path
      component (basename).
    * a bare alias/name with no separators — returned as-is.

    Trailing path separators are stripped first so a directory id written with a
    slash (``/models/Qwen3-TTS-VoiceDesign-bf16/``) still yields the repo name
    rather than an empty final component.
    """
    lowered = model_name.rstrip("/").lower()
    idx = lowered.find("models--")
    if idx != -1:
        segment = lowered[idx + len("models--") :].split("/", 1)[0]
        # ``segment`` is ``<org>--<repo>``; the repo is after the last ``--``.
        return segment.rsplit("--", 1)[-1]
    return lowered.rsplit("/", 1)[-1]


def is_qwen3_voicedesign_model(model_name: str) -> bool:
    """True for a Qwen3-TTS *VoiceDesign* checkpoint specifically.

    A single source of truth for the CustomVoice/VoiceDesign split, used by
    BOTH ``TTSEngine`` (which generate-arg shape + voice surface to use) and the
    route's ``_allowed_voices_for`` (which voice allowlist to validate against),
    so the two can never disagree — a mismatch would let a request synthesize
    as VoiceDesign while being validated against CustomVoice speakers (or vice
    versa).

    Authoritative path: resolve through the alias registry FIRST. Every
    supported input — the short aliases (``qwen3-tts-voicedesign``) and the
    canonical mlx-community HF ids — resolves to its clean canonical ``hf_id``
    (always ``mlx-community/Qwen3-TTS-...-VoiceDesign-...`` / ``...-CustomVoice-
    ...``), so the variant is read off registry metadata rather than whatever
    local filesystem path the model was loaded from. This mirrors how
    ``_list_snapshot_voices`` resolves aliases before touching the cache.

    Fallback path: for an UNREGISTERED id / bare local directory the registry
    can't help, so we read the ``customvoice`` / ``voicedesign`` variant token
    — mutually exclusive in the real repo names — from the REPO NAME component
    only (see :func:`_qwen3_repo_component`), never the parent dirs / org, with
    ``customvoice`` winning if both somehow appear (CustomVoice, a real speaker
    set, is the safe default). The truly authoritative signal for an
    unregistered checkpoint is its baked ``tts_model_type`` config, available
    only after the weights load; this pre-load name check is a deliberate
    best-effort so voice validation and the ``instruct`` fallback can run before
    a multi-GB download.
    """
    # Lazy import mirrors ``_list_snapshot_voices`` — the registry doesn't
    # import this module, and API-only runners without the audio extras can
    # still import ``tts``. Only the import is guarded (ImportError); a genuine
    # registry error (e.g. corrupt aliases.json) is left to propagate rather
    # than be masked by a silent heuristic fallback.
    name = model_name
    try:
        from .registry import resolve_audio_alias
    except ImportError:  # pragma: no cover — covered by ``[audio]``
        resolve_audio_alias = None  # type: ignore[assignment]
    if resolve_audio_alias is not None:
        entry = resolve_audio_alias(model_name)
        if entry is not None:
            name = entry.hf_id

    # Family test uses the whole id (``qwen3-tts`` may sit in the org); the
    # variant token is read only from the repo-name component.
    if not is_qwen3_tts_model(name):
        return False
    component = _qwen3_repo_component(name)
    if "customvoice" in component:
        return False
    return "voicedesign" in component


def _list_snapshot_voices(model_name: str) -> list[str]:
    """Return the safetensors voice files cached for ``model_name``.

    R11-B-F1 (Bo 0.8.12 dogfood): pre-fix the route hard-coded a
    static voice list per family — ``["default"]`` for everything
    except kokoro / chatterbox. VibeVoice's HF repo ships per-language
    voice caches (``en-Grace_woman.safetensors``, ``en-Mike_man.safe
    tensors``, the eight non-English ``Spk0/Spk1`` pairs, ...) and NO
    ``default.safetensors`` — so EVERY ``/v1/audio/speech`` call 500'd:
    ``voice="default"`` -> ``FileNotFoundError`` deep in
    ``mlx_audio.tts.models.vibevoice.Model.load_voice``; a real name
    like ``en-Grace_woman`` -> the route's 400 ``invalid_voice``
    because the static list only contained ``"default"``.

    The fix is structural — enumerate the snapshot's ``voices/`` dir at
    validation time and use THAT list, instead of pinning a per-family
    constant. The helper:

    * Returns ``[]`` when the snapshot isn't cached locally — the
      caller falls back to the static list so the FIRST request (which
      triggers the download via ``load_model``) still proceeds. Once
      the snapshot is on disk every subsequent request validates
      against the true voice set.
    * Accepts both short aliases (``"vibevoice"``) and HF ids
      (``"mlx-community/VibeVoice-Realtime-0.5B-4bit"``) by going
      through ``huggingface_hub.try_to_load_from_cache`` on
      ``config.json`` to resolve the snapshot path. We never trigger a
      download here — this is a synchronous request-path call and
      blocking on a 500 MB pull would convert "validate voice" into
      "wait two minutes".
    * Strips the ``.safetensors`` suffix and sorts the result so the
      400 envelope's ``Available:`` preview is deterministic.

    Applies UNIFORMLY to every TTS family (kokoro, chatterbox,
    voxcpm, dia, vibevoice, ...). Chatterbox/voxcpm/dia ship a single
    ``default.safetensors`` so the enumeration returns ``["default"]``
    — same behaviour as the pre-fix static list. Kokoro's snapshot
    actually ships 50+ voice files (the pre-fix static list named only
    11), so the enumeration is a strict superset there.

    Local-only by design (``local_files_only=True``): we MUST NOT
    issue an HTTP roundtrip from inside ``_allowed_voices_for`` — the
    pre-flight 400 lookup runs before any weight load and a network
    stall would convert "client sent invalid voice" into "server
    appears to hang then times out".
    """
    # Local imports keep ``vllm_mlx.audio.tts`` cheap to import on
    # API-only runners that don't install huggingface_hub (the audio
    # extras pin it, but the boot-guard path probes this module
    # without the extras).
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover — covered by ``[audio]``
        return []

    # Strip a leading registry alias to its HF id so both shapes
    # resolve to the same snapshot. The lazy import avoids a hard
    # circular: registry doesn't import this module.
    try:
        from .registry import resolve_audio_alias
    except ImportError:
        resolve_audio_alias = None  # type: ignore[assignment]

    hf_id = model_name
    if resolve_audio_alias is not None:
        entry = resolve_audio_alias(model_name)
        if entry is not None:
            hf_id = entry.hf_id

    # ``/`` is the HF-id separator — a bare alias that didn't resolve
    # in the registry (e.g. a typo) won't be in the cache either, so
    # short-circuit to ``[]`` and let the caller fall back to its
    # static list. The same guard catches HF passthrough ids that
    # haven't been downloaded yet.
    if "/" not in hf_id:
        return []

    # Pick a small file every snapshot has so ``try_to_load_from
    # _cache`` returns the snapshot's resolved path without triggering
    # a download. ``config.json`` is universal across mlx-audio TTS
    # repos (kokoro / chatterbox / vibevoice / voxcpm / dia all ship
    # one at the snapshot root).
    cached = try_to_load_from_cache(repo_id=hf_id, filename="config.json")
    if not cached or not isinstance(cached, str):
        # Snapshot not on disk yet — caller falls back to static.
        return []

    voices_dir = Path(cached).parent / "voices"
    if not voices_dir.is_dir():
        return []

    # ``.stem`` strips the ``.safetensors`` suffix; the registry
    # advertises voices by their bare name. Sorting makes the
    # ``Available:`` 400 preview deterministic so doc snapshots and
    # operator scripts don't churn.
    return sorted(p.stem for p in voices_dir.glob("*.safetensors"))


class UnsupportedAudioFormatError(Exception):
    """The requested TTS ``response_format`` cannot be encoded here.

    R8-H5 (Bo 0.8.9 dogfood): the legacy ``to_bytes`` ignored
    ``format`` and returned RIFF/WAV bytes for every value, so the
    route then set ``Content-Type: audio/{format}`` on bytes that
    started with ``RIFF…WAVE`` — a structural mislabel that broke
    every non-wav client. The encoder now raises this typed
    exception when the requested format isn't producible (no codec
    in libsndfile, no entry in the encoder table, etc.) so the route
    can translate it to a 400 ``invalid_request_error`` envelope
    listing the formats this build DOES support. The caller then
    retries with a known-good format instead of receiving a 500 or a
    mislabeled body.
    """

    def __init__(
        self,
        requested: str,
        supported: list[str],
        hint: str | None = None,
    ):
        self.requested = requested
        self.supported = supported
        self.hint = hint
        msg = (
            f"response_format={requested!r} is not supported by this "
            f"build. Supported formats: {', '.join(supported)}."
        )
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)


@dataclass
class AudioOutput:
    """Output from TTS generation."""

    audio: np.ndarray
    sample_rate: int
    duration: float


def detect_tts_family(model_name: str) -> str:
    """Module-level SSOT for TTS model-family detection.

    Kept at module scope (not only on ``TTSEngine``) so the audio route can
    ask the SAME question the engine answers at load time — e.g. whether a
    model resolves to the Kokoro family (which needs the misaki / espeak /
    spaCy runtime gate) — without constructing an engine. The
    ``TTSEngine._detect_family`` method delegates here so the two can never
    disagree (mirrors the ``is_qwen3_*`` shared classifiers above).
    """
    name_lower = model_name.lower()
    if "kokoro" in name_lower:
        return "kokoro"
    elif "chatterbox" in name_lower:
        return "chatterbox"
    elif "vibevoice" in name_lower:
        return "vibevoice"
    elif "voxcpm" in name_lower:
        return "voxcpm"
    elif "csm" in name_lower:
        return "csm"
    elif "cosyvoice" in name_lower:
        return "cosyvoice"
    elif is_qwen3_tts_model(model_name):
        # Both CustomVoice and VoiceDesign share the ``qwen3_tts`` family
        # (same mlx_audio loader + generate() entry). The VoiceDesign vs
        # CustomVoice split is a per-request generate-arg distinction, not
        # a separate loader — see ``_is_qwen3_voicedesign``.
        return "qwen3_tts"
    elif is_indextts_model(model_name):
        return "indextts"
    elif "f5-tts" in name_lower or "f5_tts" in name_lower:
        return "f5"
    else:
        return "kokoro"  # Default


def is_kokoro_family_model(model_name: str) -> bool:
    """True when ``model_name`` should be gated through the Kokoro runtime
    (misaki + espeak + spaCy G2P model, #1254).

    Uses the SAME classifier the engine loads with (:func:`detect_tts_family`,
    which :meth:`TTSEngine._detect_family` delegates to). The gate MUST agree
    with the engine: any model the engine loads/generates as Kokoro — including
    the name-based fallthrough default (a renamed / HF-path Kokoro repo, or any
    model the engine runs through the Kokoro generate path) — needs the Kokoro
    runtime, so it must be gated. Gating on a different (e.g. registry-only)
    view could skip the check for a model the engine still runs as Kokoro.
    """
    return detect_tts_family(model_name) == "kokoro"


class TTSEngine:
    """
    Text-to-Speech engine supporting multiple model families.

    Usage:
        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine.load()
        audio = engine.generate("Hello world!", voice="af_heart")
        engine.save(audio, "output.wav")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TTS_MODEL,
    ):
        """
        Initialize TTS engine.

        Args:
            model_name: HuggingFace model name. Supported families:
                - Kokoro: mlx-community/Kokoro-82M-bf16, Kokoro-82M-4bit
                - Chatterbox: mlx-community/chatterbox-turbo-fp16
                - VibeVoice: mlx-community/VibeVoice-Realtime-0.5B-4bit
                - VoxCPM: mlx-community/VoxCPM1.5
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._model_family = self._detect_family(model_name)

    def _is_qwen3_voicedesign(self) -> bool:
        """True for a Qwen3-TTS *VoiceDesign* checkpoint.

        VoiceDesign and CustomVoice share the ``qwen3_tts`` family (identical
        mlx_audio loader + ``generate()`` surface); they differ only in what
        ``generate()`` dispatches to — ``generate_voice_design`` (voice fully
        described by ``instruct``, no speakers) vs ``generate_custom_voice``
        (named speaker + optional ``instruct``). The mlx-community repos encode
        the variant in the id (``...-VoiceDesign-...``), so a substring check on
        the model name is enough to pick the right per-request arg shape without
        loading the weights. Delegates to the module-level
        :func:`is_qwen3_voicedesign_model` so the engine and the route's voice
        allowlist share ONE classifier and can never disagree.
        """
        return is_qwen3_voicedesign_model(self.model_name)

    def _detect_family(self, model_name: str) -> str:
        """Detect model family from name (delegates to the module SSOT so the
        engine and the audio route's Kokoro gate can never disagree)."""
        return detect_tts_family(model_name)

    def load(self) -> None:
        """Load the TTS model."""
        if self._loaded:
            return

        # F5-TTS is a standalone pure-MLX package (not mlx_audio's load_model
        # path): EN+ZH multilingual, zero-shot voice cloning, no torch. It fills
        # the Chinese expressive/cloneable gap Qwen3-TTS (flat) and Chatterbox
        # (English-only) leave open.
        if self._model_family == "f5":
            from f5_tts_mlx.cfm import F5TTS

            self.model = F5TTS.from_pretrained(self.model_name)
            self._loaded = True
            logger.info(f"TTS model loaded (f5): {self.model_name}")
            return

        if self._model_family == "indextts":
            self.model = _load_indextts_model(self.model_name)
            self._loaded = True
            logger.info(f"TTS model loaded (indextts): {self.model_name}")
            return

        try:
            from mlx_audio.tts.generate import load_model

            self.model = load_model(self.model_name)
            self._loaded = True
            logger.info(
                f"TTS model loaded: {self.model_name} (family: {self._model_family})"
            )
        except ImportError as e:
            logger.error(f"mlx-audio not installed: {e}")
            raise ImportError(
                "mlx-audio is required for TTS. Install with: pip install mlx-audio"
            ) from e

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        exaggeration: float | None = None,
        voice_seed: int | None = None,
    ) -> AudioOutput:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (model-specific)
            speed: Speech speed (0.5 to 2.0)
            lang_code: Language code (a=English, e=Spanish, f=French, etc.)
                Kokoro-style single-letter code. Ignored by families that
                auto-detect language (Qwen3-TTS).
            instruct: Optional emotion/style instruction (e.g. "Very happy
                and excited."). Honoured by the Qwen3-TTS family — it maps to
                that engine's ``instruct`` argument. For CustomVoice it
                modulates the emotional delivery of the predefined speaker
                (``voice``). For VoiceDesign it is the PRIMARY control: the
                full voice (timbre, gender, age, accent, emotion, prosody) is
                described here in natural language and ``voice`` is ignored;
                if omitted a neutral narrator description is substituted
                (``QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT``) so the mandatory
                arg is never missing. Other families ignore it (they have no
                emotion-control surface), so passing it is a no-op there
                rather than an error.
            voice_seed: Optional deterministic seed for Qwen3-TTS VoiceDesign.
                Reusing the same ``instruct`` and seed reproduces the same
                designed voice across calls. Rejected for other families by
                the API route.
            ref_audio: Optional path to a reference audio clip for zero-shot
                voice cloning. Used by the F5-TTS ``f5`` family to clone the
                clip's timbre; by the Chatterbox family (optional — clones the
                ref timbre on top of its default voice); and by Qwen3-TTS
                **Base** (optional — the Base variant ignores ``voice`` when
                ``ref_audio`` is set, while CustomVoice ignores ``ref_audio``
                and keeps its named speaker); and by IndexTTS (required — it
                has no predefined speakers). Use a clean 5-10s clip at the
                model's native sample rate. Ignored by families without a
                cloning surface.
            ref_text: Optional transcript of ``ref_audio`` (its exact spoken
                text). Paired with ``ref_audio`` for F5-TTS cloning and to
                anchor Qwen3-TTS Base cloning. Ignored by families that clone
                reference-free.
            exaggeration: Chatterbox emotion/intensity knob (0.0 neutral →
                ~1.0 very expressive). Only the Chatterbox family honours it;
                it drives that engine's ``exaggeration`` argument and is the
                lever that de-flattens the delivery. Other families ignore it.

        Returns:
            AudioOutput with audio data and metadata
        """
        if not self._loaded:
            self.load()

        if self._model_family == "f5":
            return self._generate_f5(text, ref_audio, ref_text, speed)

        try:
            import mlx.core as mx

            audio_chunks = []
            sample_rate = 24000  # Default for most models

            # Family-aware generate kwargs. Qwen3-TTS auto-detects the
            # language (``lang_code="auto"``) and accepts an ``instruct``
            # emotion/style argument the Kokoro-style path doesn't have;
            # forwarding Kokoro's single-letter ``lang_code="a"`` to it
            # would mis-hint the language. Every other family keeps the
            # pre-existing call shape unchanged.
            if self._model_family == "indextts":
                if not ref_audio:
                    raise ValueError(
                        "IndexTTS requires ref_audio (a reference speech clip "
                        "to clone); it has no predefined speakers."
                    )
                # Older supported mlx-audio releases require an mx.array;
                # newer releases also accept a path. Decode explicitly so the
                # route behaves identically across the supported range.
                from mlx_audio.tts.generate import load_audio

                ref_waveform = load_audio(
                    ref_audio,
                    sample_rate=getattr(self.model, "sample_rate", 24000),
                )
                gen_kwargs: dict = {"text": text, "ref_audio": ref_waveform}
            elif self._model_family == "chatterbox":
                # Chatterbox's ``generate`` steers expressiveness through
                # ``exaggeration`` and zero-shot cloning through
                # ``ref_audio``. It DOES also accept ``voice``/``speed``/
                # ``lang_code`` kwargs, but their Kokoro-oriented values on
                # this path (``voice="af_heart"``, ``lang_code="a"``) are
                # meaningless to it, so we deliberately do NOT forward them
                # and let the model's own defaults hold. ``exaggeration`` is
                # a real named parameter on both the non-turbo and turbo
                # repos (they load the same ``chatterbox.Model``), backed by
                # ``**kwargs`` on ``Model.generate`` and ``generate_audio``,
                # so forwarding it never raises TypeError on either variant.
                # Forward exactly those two knobs, each only when set.
                gen_kwargs: dict = {"text": text}
                if exaggeration is not None:
                    gen_kwargs["exaggeration"] = exaggeration
                if ref_audio:
                    gen_kwargs["ref_audio"] = ref_audio
            else:
                gen_kwargs = {"text": text, "voice": voice, "speed": speed}
                if self._model_family == "qwen3_tts":
                    gen_kwargs["lang_code"] = "auto"
                    if self._is_qwen3_voicedesign():
                        # VoiceDesign: the voice itself is authored in natural
                        # language via ``instruct`` (mandatory — the weights
                        # self-dispatch to ``generate_voice_design``, which
                        # drops ``voice`` and requires ``instruct``). Always
                        # forward a description, falling back to a neutral
                        # narrator so a missing one degrades gracefully instead
                        # of raising a TypeError deep in mlx_audio.
                        gen_kwargs["instruct"] = (
                            instruct or QWEN3_TTS_VOICEDESIGN_DEFAULT_INSTRUCT
                        )
                    elif instruct:
                        # CustomVoice: named speaker (``voice``) with optional
                        # ``instruct`` emotion/style modulation.
                        gen_kwargs["instruct"] = instruct
                    # Qwen3-TTS Base = zero-shot voice cloning: forward the
                    # reference clip (+ its transcript) when given. Base ignores
                    # ``voice`` while a ref is set; CustomVoice ignores
                    # ``ref_audio`` and keeps its named speaker — so forwarding
                    # only-when-set lets one family serve both variants.
                    if ref_audio:
                        gen_kwargs["ref_audio"] = ref_audio
                        if ref_text:
                            gen_kwargs["ref_text"] = ref_text
                else:
                    gen_kwargs["lang_code"] = lang_code

            if voice_seed is not None:
                if not self._is_qwen3_voicedesign():
                    raise ValueError(
                        "voice_seed is supported only by Qwen3-TTS VoiceDesign"
                    )
            with _qwen_seeded_sampling(self.model, voice_seed):
                for result in self.model.generate(**gen_kwargs):
                    audio_data = result.audio
                    if hasattr(result, "sample_rate"):
                        sample_rate = result.sample_rate

                    # Convert mlx array to numpy
                    if isinstance(audio_data, mx.array) or hasattr(
                        audio_data, "tolist"
                    ):
                        audio_np = np.array(audio_data.tolist(), dtype=np.float32)
                    else:
                        audio_np = np.array(audio_data, dtype=np.float32)
                    audio_chunks.append(audio_np)

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            # Concatenate all chunks
            full_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )
            duration = len(full_audio) / sample_rate

            return AudioOutput(
                audio=full_audio,
                sample_rate=sample_rate,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def _generate_f5(
        self,
        text: str,
        ref_audio: str | None,
        ref_text: str | None,
        speed: float,
    ) -> AudioOutput:
        """F5-TTS inference (pure MLX). Clones the reference clip's timbre and
        speaks ``text`` in it; with no ``ref_audio`` it uses the packaged default
        reference. Mirrors ``f5_tts_mlx.generate.generate`` but reuses the
        already-loaded model (no per-call reload)."""
        if (ref_audio is None) != (ref_text is None):
            raise ValueError("F5 voice cloning requires both ref_audio and ref_text.")

        import pkgutil

        import mlx.core as mx
        import soundfile as sf
        from f5_tts_mlx.generate import (
            FRAMES_PER_SEC,
            SAMPLE_RATE,
            TARGET_RMS,
            convert_char_to_pinyin,
            estimated_duration,
        )

        def read_reference(source):
            if hasattr(source, "seek"):
                source.seek(0)
            # Open the source once and validate metadata + read samples through
            # the SAME handle. Using ``sf.info(path)`` then ``sf.read(path)``
            # would open the path twice (TOCTOU): a file swapped between the two
            # opens could pass the sample-rate/frame/channel guards yet decode
            # something else. A single handle closes that gap and also bounds the
            # decode to the header's validated frames*channels.
            with sf.SoundFile(source) as handle:
                if handle.samplerate != SAMPLE_RATE:
                    raise ValueError(
                        f"F5 ref_audio must be {SAMPLE_RATE} Hz "
                        f"(got {handle.samplerate}); resample it."
                    )
                if handle.frames <= 0 or handle.frames > SAMPLE_RATE * 30:
                    raise ValueError("F5 ref_audio must be between 0 and 30 seconds.")
                # Reject pathological channel counts before decoding: the
                # duration guard bounds frames only, so a many-channel header
                # would otherwise force a frames*channels allocation.
                if handle.channels < 1 or handle.channels > F5_MAX_REF_CHANNELS:
                    raise ValueError(
                        f"F5 ref_audio must be mono or stereo "
                        f"(got {handle.channels} channels)."
                    )
                audio = handle.read()
            # soundfile returns shape (frames,) for mono and (frames, channels)
            # for multi-channel. F5's sample() expects batched mono, so downmix
            # any extra channels to a single mono track by averaging.
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio

        if ref_audio is not None:
            assert ref_text is not None  # paired-input validation above
            aud = read_reference(ref_audio)
            rtext = ref_text
        else:
            # packaged short reference (its timbre; content comes from `text`)
            data = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
            if data is None:
                raise RuntimeError("F5-TTS packaged reference audio is missing.")
            aud = read_reference(io.BytesIO(data))
            rtext = "Some call me nature, others call me mother nature."

        aud = mx.array(aud)
        rms = mx.sqrt(mx.mean(mx.square(aud)))
        mx.eval(rms)
        rms_value = float(rms)
        # A zero/near-silent reference divides by ~0 below (NaN/inf) or is
        # amplified into full-scale noise. Reject it with a clear error. Report
        # a non-finite RMS separately so the message doesn't nonsensically claim
        # NaN/inf is "below the floor".
        if not np.isfinite(rms_value):
            raise ValueError(
                "F5 ref_audio must contain finite, non-silent audio "
                f"(reference RMS is non-finite: {rms_value})."
            )
        if rms_value < F5_MIN_REF_RMS:
            raise ValueError(
                "F5 ref_audio must contain finite, non-silent audio "
                f"(reference RMS {rms_value:.2e} is below the "
                f"{F5_MIN_REF_RMS:.0e} floor)."
            )
        if rms_value < TARGET_RMS:
            aud = aud * TARGET_RMS / rms
        # explicit duration estimate — F5's auto heuristic can collapse to ~0s
        dur = int(estimated_duration(aud, rtext, text, speed) * FRAMES_PER_SEC)
        ptext = convert_char_to_pinyin([rtext + " " + text])
        # F5TTS.from_pretrained injects Vocos.decode as ``_vocoder``.
        # sample() therefore returns the decoded 1-D waveform (Vocos'
        # ISTFTHead squeezes the single batch axis), not mel frames.
        wave, _ = self.model.sample(
            mx.expand_dims(aud, axis=0),
            text=ptext,
            duration=dur,
            steps=8,
            speed=speed,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
        )
        if wave.ndim != 1:
            raise RuntimeError(
                f"F5-TTS returned an unexpected waveform shape: {wave.shape}"
            )
        wave = wave[aud.shape[0] :]  # trim the decoded reference prefix
        mx.eval(wave)
        out = np.array(wave, dtype=np.float32)
        return AudioOutput(
            audio=out, sample_rate=SAMPLE_RATE, duration=len(out) / SAMPLE_RATE
        )

    def stream_generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> Iterator[AudioOutput]:
        """
        Stream speech generation chunk by chunk.

        Args:
            text: Text to synthesize
            voice: Voice ID
            speed: Speech speed

        Yields:
            AudioOutput chunks
        """
        if not self._loaded:
            self.load()

        sample_rate = 24000

        for result in self.model.generate(
            text=text,
            voice=voice,
            speed=speed,
        ):
            audio_data = result.audio
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

            if hasattr(audio_data, "tolist"):
                audio_np = np.array(audio_data.tolist(), dtype=np.float32)
            else:
                audio_np = np.array(audio_data, dtype=np.float32)

            yield AudioOutput(
                audio=audio_np,
                sample_rate=sample_rate,
                duration=len(audio_np) / sample_rate,
            )

    def save(
        self,
        audio: AudioOutput,
        path: str | Path,
        format: str = "wav",
    ) -> None:
        """
        Save audio to file.

        Args:
            audio: AudioOutput to save
            path: Output file path
            format: Output format (wav, mp3)
        """
        try:
            from mlx_audio.tts import save_audio

            save_audio(audio.audio, str(path), sample_rate=audio.sample_rate)
            logger.info(f"Audio saved to {path}")
        except ImportError:
            # Fallback to scipy
            import scipy.io.wavfile as wav

            # Ensure audio is in correct format
            audio_int16 = (audio.audio * 32767).astype(np.int16)
            wav.write(str(path), audio.sample_rate, audio_int16)
            logger.info(f"Audio saved to {path} (scipy fallback)")

    def to_bytes(
        self,
        audio: AudioOutput,
        format: str = "wav",
    ) -> bytes:
        """
        Convert audio to bytes in the requested container format.

        R8-H5 (Bo 0.8.9 dogfood): pre-fix every call returned RIFF/WAV
        bytes regardless of ``format`` — the route then set
        ``Content-Type: audio/{format}`` so a client asking for
        ``response_format="mp3"`` got an ``audio/mp3``-labelled body
        whose magic was ``RIFF…WAVE``. Browsers and ffmpeg both reject
        the mismatch; OpenAI parity was structurally broken on every
        non-wav format. The handler now branches on ``format`` and
        encodes via the appropriate codec:

        * ``wav`` → Python's standard-library ``wave`` writer. Keeping this
          path independent of SciPy IO lets bounded clients retain only the
          ``scipy.signal`` resampling closure.
        * ``flac`` / ``ogg`` / ``opus`` → ``soundfile`` (libsndfile
          ≥1.0). Always shipped via ``rapid-mlx[audio]``.
        * ``mp3`` → ``soundfile`` when libsndfile ≥1.1 (the version
          that bundled the LAME-backed MP3 writer). Older builds raise
          ``LibsndfileError`` which the caller surfaces as a 400 with
          an actionable hint (see ``UnsupportedAudioFormatError``).
        * ``aac`` → not supported by libsndfile in any current release;
          raises :class:`UnsupportedAudioFormatError` so the route can
          translate to a 400 listing the formats this build supports.
          We do NOT silently relabel WAV bytes as ``audio/aac``.
        * ``pcm`` → raw little-endian int16 PCM (no container). Mirrors
          OpenAI's ``response_format="pcm"`` contract.

        Raises:
            UnsupportedAudioFormatError: when the requested format
                cannot be produced by the available encoder stack.
                The route catches this and emits a 400 envelope so
                the caller can fall back to a supported format rather
                than receive a mislabeled WAV.
        """
        fmt = (format or "wav").lower()
        audio_int16 = (np.clip(audio.audio, -1.0, 1.0) * 32767).astype(np.int16)

        if fmt == "wav":
            buffer = io.BytesIO()
            channels = 1 if audio_int16.ndim == 1 else audio_int16.shape[1]
            with wave.open(buffer, "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(2)
                output.setframerate(audio.sample_rate)
                output.writeframes(audio_int16.astype("<i2", copy=False).tobytes())
            return buffer.getvalue()

        if fmt == "pcm":
            # OpenAI ``response_format="pcm"`` is raw 16-bit signed LE
            # PCM at the source sample rate — no header, no container.
            # Pre-fix we wrapped the same bytes in RIFF/WAVE and
            # labeled them ``audio/pcm`` which any decoder following
            # the OpenAI contract would mis-parse as PCM headers.
            return audio_int16.tobytes()

        # soundfile-backed formats. ``flac``/``ogg``/``opus`` are
        # always supported; ``mp3`` depends on the libsndfile version
        # the wheel was built against. Surface a clean error if the
        # encoder isn't available so the route can emit a 400 listing
        # the supported set rather than a 500 stack trace.
        try:
            import soundfile as sf
        except ImportError as e:  # pragma: no cover — covered by extras
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=["wav", "pcm"],
                hint="Install with: pip install 'rapid-mlx[audio]'",
            ) from e

        # Map our OpenAI-style ``response_format`` values onto the
        # ``(format, subtype)`` pair ``soundfile`` expects. ``opus`` is
        # the OGG container with the Opus codec — same wire shape that
        # OpenAI returns. ``mp3`` only encodes when the underlying
        # libsndfile shipped the LAME writer; older wheels (<1.1) raise
        # ``LibsndfileError`` which we translate to the 400 hint.
        soundfile_targets: dict[str, tuple[str, str | None]] = {
            "flac": ("FLAC", None),
            "ogg": ("OGG", "VORBIS"),
            "opus": ("OGG", "OPUS"),
            "mp3": ("MP3", None),
        }
        target = soundfile_targets.get(fmt)
        if target is None:
            # Anything not in the table (``aac``, future formats, typos)
            # gets a structured rejection. The route maps this to a 400
            # envelope listing the formats we DID support so the caller
            # can retry with a known-good value.
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=sorted(["wav", "pcm", *soundfile_targets.keys()]),
            )

        container, subtype = target
        buffer = io.BytesIO()
        try:
            sf.write(
                buffer,
                audio_int16,
                audio.sample_rate,
                format=container,
                subtype=subtype,
            )
        except Exception as e:
            # libsndfile raises a typed ``LibsndfileError`` when the
            # codec isn't compiled in (most often ``mp3`` on macOS
            # wheels built against an older libsndfile). Re-raise as
            # the structured envelope error so the route emits 400 with
            # the supported-set hint instead of a 500 stack trace.
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=sorted(["wav", "pcm", *soundfile_targets.keys()]),
                hint=(
                    f"Encoder for {fmt!r} is not available in this "
                    f"libsndfile build ({e}). Upgrade libsndfile to "
                    "the latest release, or request a supported format."
                ),
            ) from e
        return buffer.getvalue()

    def get_voices(self) -> list:
        """Get available voices for current model."""
        if self._model_family == "kokoro":
            return KOKORO_VOICES
        elif self._model_family == "chatterbox":
            return CHATTERBOX_VOICES
        elif self._model_family == "qwen3_tts":
            # VoiceDesign has no speakers — advertise the ``describe``
            # sentinel instead of the CustomVoice speaker list so callers
            # aren't misled into thinking a picked speaker matters (the
            # voice is authored entirely via ``instruct``).
            if self._is_qwen3_voicedesign():
                return list(QWEN3_TTS_VOICEDESIGN_VOICES)
            return list(QWEN3_TTS_VOICES)
        elif self._model_family == "indextts":
            return list(INDEXTTS_VOICES)
        else:
            return ["default"]

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self._loaded = False
        logger.info("TTS model unloaded")


def generate_speech(
    text: str,
    model_name: str = DEFAULT_TTS_MODEL,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> AudioOutput:
    """
    Convenience function to generate speech without managing engine.

    Args:
        text: Text to synthesize
        model_name: Model to use
        voice: Voice ID
        speed: Speech speed

    Returns:
        AudioOutput
    """
    engine = TTSEngine(model_name)
    engine.load()
    return engine.generate(text, voice=voice, speed=speed)
