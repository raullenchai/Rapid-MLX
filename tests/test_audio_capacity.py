"""Capacity provenance for audio roles (#2305).

Issue #2305 forbids inferring capacity from model names or hashes. These tests
pin the three legitimate sources and, most importantly, that the name-parsing
heuristic used for arbitrary text checkpoints is NOT reachable from this path.
"""

from __future__ import annotations

import json
import os

import pytest

from vllm_mlx.runtime.audio_capacity import (
    AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES,
    resolve_audio_role_capacity,
)


def test_registered_alias_is_sized_from_the_catalog_manifest():
    from vllm_mlx.audio.registry import resolve_audio_alias
    from vllm_mlx.model_sizes import size_bytes

    capacity = resolve_audio_role_capacity("whisper-large-v3")

    entry = resolve_audio_alias("whisper-large-v3")
    assert capacity.capacity_source == "manifest"
    assert capacity.hf_id == entry.hf_id
    assert capacity.weight_bytes == size_bytes(entry.hf_id)
    assert capacity.reserved_bytes == (
        capacity.weight_bytes + AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES
    )


def test_every_registered_audio_alias_resolves_to_a_known_capacity():
    # The registry is the only place a new audio model lands. If one is added
    # without regenerating model_sizes.json, every load of it becomes an
    # unbudgeted admission — catch that here rather than in production.
    from vllm_mlx.audio.registry import list_audio_aliases

    unknown = [
        entry.alias
        for entry in list_audio_aliases()
        if not resolve_audio_role_capacity(entry.alias).is_known
    ]
    assert unknown == []


def test_hf_id_resolves_the_same_way_as_its_short_alias():
    by_alias = resolve_audio_role_capacity("kokoro")
    by_repo = resolve_audio_role_capacity(by_alias.hf_id)

    assert by_repo == by_alias


def test_unlisted_repo_falls_back_to_measuring_the_cached_snapshot(
    tmp_path, monkeypatch
):
    repo_id = "someone/unlisted-asr"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "abc123"
    blobs = repo_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("abc123")

    blob = blobs / "deadbeef"
    blob.write_bytes(b"x" * 4096)
    # HF snapshots are symlink trees into blobs/; one blob can back several
    # entries and must only be counted once.
    os.symlink(blob, snapshot / "weights.npz")
    os.symlink(blob, snapshot / "model.safetensors")
    (snapshot / "config.json").write_text(json.dumps({"a": 1}))

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 4096 + len(json.dumps({"a": 1}))
    assert capacity.reserved_bytes == (
        capacity.weight_bytes + AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES
    )


def test_uncached_unlisted_repo_reports_unknown_rather_than_guessing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity("someone/never-downloaded")

    assert capacity.capacity_source == "unknown"
    assert capacity.is_known is False
    # Zero, and the manager turns that into a rejection: a zero charge would
    # otherwise skip the ceiling check entirely.
    assert capacity.reserved_bytes == 0
    assert capacity.weight_bytes is None


def test_complete_snapshot_without_index_is_measured(tmp_path, monkeypatch):
    # Single-file (non-sharded) repos are complete with one non-empty
    # ``model*.safetensors`` and must still be sizable.
    repo_id = "someone/single-file-asr"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    (snapshot / "model.safetensors").write_bytes(b"x" * 2048)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 2048 + 2


def test_capacity_is_never_inferred_from_a_parameter_count_in_the_name(
    tmp_path, monkeypatch
):
    # ``estimate_model_bytes`` would happily size this from "7b" + "4bit".
    # #2305 forbids that for audio roles; the resolver must say "unknown".
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity("fictional/asr-7b-4bit")

    assert capacity.capacity_source == "unknown"
    assert capacity.reserved_bytes == 0


def test_partial_snapshot_is_not_reported_as_cached(tmp_path, monkeypatch):
    """A partial snapshot must not be sized from the downloaded fragment."""

    repo_id = "someone/interrupted-asr"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    # Index promises two shards; only one landed before the pull died.
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1024)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "unknown"
    assert capacity.reserved_bytes == 0


def test_snapshot_without_refs_main_is_not_measured(tmp_path, monkeypatch):
    """Only the revision the loader resolves may be measured.

    Without refs/main there is no way to know which snapshot snapshot_download
    would open, so an unrelated stale one must not stand in for it.
    """

    repo_id = "someone/no-refs"
    snapshot = tmp_path / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "old"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x" * 4096)
    (snapshot / "config.json").write_text("{}")

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    assert resolve_audio_role_capacity(repo_id).capacity_source == "unknown"


def test_runtime_overhead_does_not_claim_to_bound_decoded_audio():
    """The 25 MB upload cap bounds compressed bytes, not decoded samples.

    25 MB of 6 kbps Opus is ~9.7 hours, which is ~2.1 GiB of float32 at 16 kHz
    — far past the per-role allowance. The request-level duration limit is what
    actually bounds it, so assert that limit keeps the buffer within budget.
    """

    from vllm_mlx.runtime.audio_capacity import MAX_TRANSCRIPTION_SECONDS

    float32_at_16k = MAX_TRANSCRIPTION_SECONDS * 16_000 * 4
    assert float32_at_16k < AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES

    # And the cap the upload limit alone would permit does NOT fit, which is
    # why the duration bound has to exist.
    opus_6kbps_seconds = (25 * 1024 * 1024) / (6_000 / 8)
    assert opus_6kbps_seconds * 16_000 * 4 > AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES


def test_overlong_audio_is_rejected_before_the_engine_allocates(tmp_path):
    """Duration, not compressed size, is what bounds the decoded buffer."""

    import wave

    from fastapi import HTTPException

    from vllm_mlx.routes.audio import _audio_request_bytes
    from vllm_mlx.runtime.audio_capacity import MAX_TRANSCRIPTION_SECONDS

    def write_short_wav(path, rate=8000):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(b"\0\0")
        return path

    short = write_short_wav(tmp_path / "short.wav")
    # A tiny file charges a tiny amount rather than raising.
    assert _audio_request_bytes(str(short)) < 1024 * 1024

    long_path = tmp_path / "long.wav"
    rate = 8000
    frames = int((MAX_TRANSCRIPTION_SECONDS + 3600) * rate)
    with wave.open(str(long_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0")
    # Rewrite the frame count in the header rather than materializing hours of
    # silence on disk.
    with open(long_path, "r+b") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(40)
        handle.write((frames * 2).to_bytes(4, "little"))
        handle.seek(4)
        handle.write((size - 8).to_bytes(4, "little"))

    with pytest.raises(HTTPException) as excinfo:
        _audio_request_bytes(str(long_path))
    assert excinfo.value.status_code == 413
    assert excinfo.value.detail["error"]["code"] == "audio_too_long"


def test_unreadable_container_is_refused_not_charged_a_guess(tmp_path):
    """No sound upper bound exists for bytes we cannot identify.

    A bitrate floor is an assumption about encoders; a fixed "worst plausible
    layout" simultaneously under-charges anything longer or higher-rate than
    assumed and grossly over-charges a one-second file. Both were tried; both
    were wrong. Refusing is the only honest answer.
    """

    from fastapi import HTTPException

    from vllm_mlx.routes.audio import _audio_request_bytes

    opaque = tmp_path / "mystery.bin"
    opaque.write_bytes(b"\x00" * 4096)

    with pytest.raises(HTTPException) as excinfo:
        _audio_request_bytes(str(opaque))

    assert excinfo.value.status_code == 400
    # Reuses the lane's established "not usable audio" code — corrupted and
    # unidentifiable are the same failure from the caller's side.
    assert excinfo.value.detail["error"]["code"] == "invalid_audio_file"


def test_transcription_charge_scales_with_source_layout(tmp_path):
    """A 48 kHz stereo source costs far more than its 16 kHz mono result."""

    import wave

    from vllm_mlx.routes.audio import _audio_request_bytes

    def write_wav(path, rate, channels, seconds=2):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(b"\0\0" * rate * channels * seconds)
        return str(path)

    mono16 = write_wav(tmp_path / "mono16.wav", 16_000, 1)
    stereo48 = write_wav(tmp_path / "stereo48.wav", 48_000, 2)

    # Same duration, six times the source samples.
    assert _audio_request_bytes(stereo48) > _audio_request_bytes(mono16) * 3


def test_speech_input_is_length_bounded():
    """TTS output size grows with input length, so input must be bounded."""

    from pydantic import ValidationError

    from vllm_mlx.api.models import AudioSpeechRequest
    from vllm_mlx.runtime.audio_capacity import MAX_SPEECH_INPUT_CHARACTERS

    assert AudioSpeechRequest(input="x" * MAX_SPEECH_INPUT_CHARACTERS)
    with pytest.raises(ValidationError):
        AudioSpeechRequest(input="x" * (MAX_SPEECH_INPUT_CHARACTERS + 1))


def test_speech_buffer_scales_with_speed_not_just_characters():
    """20k chars is not a memory bound on its own — speed multiplies it."""

    from vllm_mlx.runtime.audio_capacity import (
        MAX_SPEECH_INPUT_CHARACTERS,
        speech_buffer_bytes,
    )

    at_normal = speech_buffer_bytes(MAX_SPEECH_INPUT_CHARACTERS, speed=1.0)
    at_slowest = speech_buffer_bytes(MAX_SPEECH_INPUT_CHARACTERS, speed=0.25)

    # speed=0.25 quadruples the duration of identical text.
    assert at_slowest == pytest.approx(at_normal * 4, rel=0.01)
    # And the worst case is multi-GiB, which is why it must be charged rather
    # than assumed to fit inside a fixed per-role allowance.
    assert at_slowest > 4 * 1024**3
    assert at_slowest > AUDIO_ROLE_RUNTIME_OVERHEAD_BYTES


def test_generation_ceiling_never_exceeds_what_the_reservation_bought():
    """The ledger must reserve for every second the engine is allowed to emit.

    An earlier revision let the ceiling run 2x the duration the reservation was
    sized from, reasoning that the 3.5x peak multiplier absorbed it. It does
    not: that multiplier counts concurrent COPIES of one waveform, not extra
    duration, so a request that generated up to its permitted ceiling peaked at
    ~2x its reservation. Both now derive from one bounded duration.
    """

    from vllm_mlx.runtime.audio_capacity import (
        _TTS_PEAK_MULTIPLIER,
        MAX_SPEECH_INPUT_CHARACTERS,
        max_output_seconds_for,
        speech_buffer_bytes,
    )

    for characters in (1, 500, MAX_SPEECH_INPUT_CHARACTERS):
        for speed in (0.25, 1.0, 2.0):
            text = "x" * characters
            allowed = max_output_seconds_for(text, speed=speed)
            reserved = speech_buffer_bytes(characters, speed=speed)

            # What the engine may actually produce, at the highest native rate
            # any registered engine emits, times the pipeline's live copies.
            worst_case = allowed * 44_100 * 4 * _TTS_PEAK_MULTIPLIER
            assert reserved >= worst_case * 0.999, (
                f"{characters} chars at speed={speed}: permitted "
                f"{allowed:.1f}s needs {worst_case:.0f}B but only "
                f"{reserved}B was reserved"
            )


def test_conversion_headroom_is_charged_for_the_permitted_duration():
    """The 96 kHz stereo conversion is sized from duration, so it must use the
    same bounded duration the engine is allowed to reach — not the raw
    prediction. Charging the prediction under-charged exactly the requests that
    resample the most."""

    from vllm_mlx.runtime.audio_capacity import (
        max_output_seconds_for,
        speech_buffer_bytes,
    )

    characters = 5_000
    allowed = max_output_seconds_for("x" * characters, speed=0.25)
    reserved = speech_buffer_bytes(
        characters, speed=0.25, sample_rate=96_000, channels=2
    )
    native = speech_buffer_bytes(characters, speed=0.25)

    # float32 converted array + int16 copy + encoded bytes, over the duration
    # the engine may actually reach.
    converted = allowed * 96_000 * 2 * 4 * 2
    assert reserved - native >= converted * 0.999


def test_tts_reference_charge_includes_encoded_decode_and_resample_buffers():
    from vllm_mlx.runtime.audio_capacity import tts_reference_buffer_bytes

    duration = 30.0
    encoded = 1_400_000
    compressed = 1_000_000
    reserved = tts_reference_buffer_bytes(
        duration,
        source_rate=48_000,
        source_channels=2,
        encoded_bytes=encoded,
        compressed_bytes=compressed,
    )

    source_float64 = duration * 48_000 * 2 * 8
    native_float32 = duration * 44_100 * 4
    assert reserved == int(
        source_float64 * 2 + native_float32 * 3 + encoded + compressed
    )


# ---------------------------------------------------------------------------
# The ceiling has to bound the ALLOCATION, not merely observe it afterwards.
# ---------------------------------------------------------------------------


#: The stride each single-yield family exposes on a LOADED model, as the
#: attribute path the engine reads. Mirrors the real checkpoints' defaults;
#: ``samples`` is what those values multiply out to.
_STRIDE_FIXTURES = {
    "voxcpm": {
        "samples": 4 * 1764,
        "rate": 44_100,
        "attrs": lambda: {
            "patch_size": 4,
            "audio_vae": type("V", (), {"hop_length": 1764})(),
        },
    },
    "qwen3_tts": {
        "samples": 1920,
        "rate": 24_000,
        "attrs": lambda: {
            "speech_tokenizer": type("S", (), {"decode_upsample_rate": 1920})()
        },
    },
    "chatterbox": {
        "samples": 960,  # 24 kHz / 25 Hz speech tokens
        "rate": 24_000,
        "attrs": lambda: {"sr": 24_000},
    },
    "vibevoice": {
        "samples": 8 * 5 * 5 * 4 * 2 * 2,
        "rate": 24_000,
        "attrs": lambda: {
            "config": type(
                "C",
                (),
                {
                    "acoustic_tokenizer_config": type(
                        "A",
                        (),
                        {
                            "decoder_ratios": None,
                            "encoder_ratios": [8, 5, 5, 4, 2, 2],
                        },
                    )()
                },
            )()
        },
    },
    "indextts": {
        "samples": 8 * 8 * 2 * 2,
        "rate": 24_000,
        "attrs": lambda: {
            "args": type(
                "A", (), {"bigvgan": type("B", (), {"upsample_rates": [8, 8, 2, 2]})()}
            )()
        },
    },
}


def _stride_engine(family, model_cls, **extra):
    """A ``TTSEngine`` whose model exposes ``family``'s real stride attributes.

    The engine measures every stride off the loaded checkpoint, so a double
    without them is refused — correctly, but it would test the refusal rather
    than the thing under test.
    """

    from vllm_mlx.audio.tts import TTSEngine

    fixture = _STRIDE_FIXTURES[family]
    attrs = {
        "generate": model_cls.generate,
        "sample_rate": fixture["rate"],
        **fixture["attrs"](),
        **extra,
    }
    engine = TTSEngine.__new__(TTSEngine)
    engine._model_family = family
    engine.model_name = f"test/{family}"
    engine._loaded = True
    engine.model = type("Fake", (), attrs)()
    return engine


class _RunawayModel:
    """TTS double that records calls and can ignore the duration estimate."""

    def __init__(
        self, chunks, sample_rate=24_000, signature_of=None, stride_attrs=None
    ):
        import inspect

        import numpy as np

        self._chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        self.sample_rate = sample_rate
        self.calls: list[dict] = []
        self.chunks_consumed = 0
        for name, value in (stride_attrs or {}).items():
            setattr(self, name, value)
        if signature_of is not None:
            # Per-instance, so borrowing never mutates the shared fake class.
            def generate(**kwargs):
                return self._generate(**kwargs)

            generate.__signature__ = inspect.signature(signature_of)
            self.generate = generate
        else:
            self.generate = self._generate

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        for chunk in self._chunks:
            self.chunks_consumed += 1
            yield type(
                "Result", (), {"audio": chunk, "sample_rate": self.sample_rate}
            )()


def _ceiling_samples(text, *, speed=1.0, sample_rate=24_000):
    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    return int(max_output_seconds_for(text, speed=speed) * sample_rate)


def test_an_oversized_chunk_is_truncated_not_merely_stopped_after():
    """One oversized chunk is sliced to the remaining budget."""

    import numpy as np

    from vllm_mlx.audio.tts import TTSEngine

    text = "hello"
    ceiling = _ceiling_samples(text)

    engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
    engine._loaded = True
    engine.model = _RunawayModel([np.ones(ceiling * 50, dtype=np.float32)])

    output = engine.generate(text)

    assert output.audio.size == ceiling
    assert engine.model.chunks_consumed == 1


def test_the_ceiling_is_denominated_in_seconds_not_a_fixed_rate_sample_count():
    """Different native rates receive the same duration ceiling."""

    import numpy as np

    from vllm_mlx.audio.tts import TTSEngine
    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    text = "hello"
    seconds = max_output_seconds_for(text)

    for rate in (24_000, 44_100):
        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        engine._loaded = True
        engine.model = _RunawayModel(
            [np.ones(int(seconds * rate * 20), dtype=np.float32)], sample_rate=rate
        )
        output = engine.generate(text)

        # Same DURATION at either rate, which is what the budget paid for.
        assert output.duration == pytest.approx(seconds, rel=0.01)
        assert output.audio.size == int(seconds * rate)


def test_accumulated_chunks_stop_at_the_ceiling():
    """A backend that streams past its budget stops being consumed."""

    import numpy as np

    from vllm_mlx.audio.tts import TTSEngine

    text = "hello"
    ceiling = _ceiling_samples(text)
    chunk = max(1, ceiling // 4)

    engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
    engine._loaded = True
    engine.model = _RunawayModel([np.ones(chunk, dtype=np.float32)] * 1000)

    output = engine.generate(text)

    assert output.audio.size <= ceiling
    # It stopped consuming rather than draining all 1000 chunks.
    assert engine.model.chunks_consumed < 1000


def test_single_yield_backends_are_bounded_before_they_generate():
    """Single-yield generation receives a token limit before allocation."""

    import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm
    import numpy as np

    from vllm_mlx.audio.tts import TTSEngine
    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    text = "hello"
    fixture = _STRIDE_FIXTURES["voxcpm"]

    engine = TTSEngine("mlx-community/VoxCPM1.5")
    assert engine._model_family == "voxcpm"
    engine._loaded = True
    # Borrow the real backend's signature so the budget is floored against the
    # default VoxCPM actually declares, not the fake's ``**kwargs``.
    engine.model = _RunawayModel(
        [np.ones(64, dtype=np.float32)],
        sample_rate=fixture["rate"],
        signature_of=voxcpm.Model.generate,
        stride_attrs=fixture["attrs"](),
    )

    engine.generate(text)

    (call,) = engine.model.calls
    budget = call["max_tokens"]
    # The default is 4096 tokens; a five-character request must be given far
    # less than that.
    assert budget < 4096
    # And the budget must be close to (never above) what the ledger reserved —
    # see the rounding test below for the direction that matters.
    permitted = budget * fixture["samples"] / fixture["rate"]
    reserved = max_output_seconds_for(text)
    assert permitted <= reserved
    assert permitted > reserved / 2


def test_every_single_yield_backend_is_bounded_before_it_generates():
    """Every complete-decode family stays within the request reservation."""

    import inspect as _inspect

    import mlx_audio.tts.models.chatterbox.chatterbox as chatterbox
    import mlx_audio.tts.models.indextts.indextts as indextts
    import mlx_audio.tts.models.qwen3_tts.qwen3_tts as qwen3_tts
    import mlx_audio.tts.models.vibevoice.vibevoice as vibevoice
    import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm

    from vllm_mlx.audio.tts import _SINGLE_YIELD_FAMILIES, _measured_samples_per_token
    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    backends = {
        "chatterbox": chatterbox.Model,
        "indextts": indextts.Model,
        "vibevoice": vibevoice.Model,
        "voxcpm": voxcpm.Model,
        "qwen3_tts": qwen3_tts.Model,
    }
    assert set(backends) == set(_SINGLE_YIELD_FAMILIES)
    assert set(backends) == set(_STRIDE_FIXTURES)

    for family, model_cls in backends.items():
        # Each really does yield exactly once per generate() body.
        source = _inspect.getsource(model_cls.generate)
        assert source.count("yield") >= 1

        engine = _stride_engine(family, model_cls)
        fixture = _STRIDE_FIXTURES[family]

        # The stride is measured off the model, never tabulated.
        per_token = _measured_samples_per_token(engine.model, family)
        assert per_token == fixture["samples"], family

        # A short request: the reservation, not the backend default, must bind.
        seconds = max_output_seconds_for("hello")
        budget = engine._token_budget_for(seconds)
        assert budget is not None, f"{family} generates unbounded"
        assert budget > 0
        permitted = budget * per_token / fixture["rate"]
        assert permitted <= seconds, (
            f"{family}: budget buys {permitted:.4f}s but only {seconds:.4f}s "
            "was reserved"
        )


def test_a_checkpoint_that_overrides_its_stride_is_sized_from_its_own_config():
    """Checkpoint stride overrides determine their own token budgets."""

    import mlx_audio.tts.models.vibevoice.vibevoice as vibevoice
    import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm

    from vllm_mlx.audio.tts import _measured_samples_per_token
    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    # VoxCPM with double the default patch_size.
    engine = _stride_engine("voxcpm", voxcpm.Model, patch_size=8)
    assert _measured_samples_per_token(engine.model, "voxcpm") == 8 * 1764

    # VibeVoice's decoder_ratios override its encoder_ratios when set.
    engine_vv = _stride_engine(
        "vibevoice",
        vibevoice.Model,
        config=type(
            "C",
            (),
            {
                "acoustic_tokenizer_config": type(
                    "A",
                    (),
                    {
                        "decoder_ratios": [8, 8, 4, 4],
                        "encoder_ratios": [8, 5, 5, 4, 2, 2],
                    },
                )()
            },
        )(),
    )
    assert _measured_samples_per_token(engine_vv.model, "vibevoice") == 8 * 8 * 4 * 4

    # And the budget shrinks accordingly, so the reservation still holds.
    seconds = max_output_seconds_for("hello" * 20)
    for engine_, per_token, rate in (
        (engine, 8 * 1764, 44_100),
        (engine_vv, 8 * 8 * 4 * 4, 24_000),
    ):
        budget = engine_._token_budget_for(seconds)
        assert budget * per_token / rate <= seconds


def test_a_single_yield_backend_that_cannot_be_bounded_is_refused():
    """Unboundable complete-decode backends fail closed on the model field."""

    import mlx_audio.tts.models.indextts.indextts as indextts

    from vllm_mlx.audio.tts import AudioGenerationUnboundedError, TTSEngine

    # (a) single-yield family whose stride cannot be measured off the model.
    engine = TTSEngine.__new__(TTSEngine)
    engine._model_family = "indextts"
    engine.model_name = "test/indextts"
    engine.model = type(
        "Fake", (), {"generate": indextts.Model.generate, "sample_rate": 24_000}
    )()
    with pytest.raises(AudioGenerationUnboundedError, match="stride") as excinfo:
        engine._token_budget_for(1.0)
    assert excinfo.value.param == "model"

    # (b) single-yield family that will not accept max_tokens. The waveform
    # must never be allocated, so the fake asserts if generate() is reached.
    class _NoMaxTokens:
        sample_rate = 24_000
        sr = 24_000

        def generate(self, *, text, voice=None, speed=1.0, lang_code=None):
            raise AssertionError("generation must be refused before it starts")
            yield  # pragma: no cover - generator marker

    engine = TTSEngine("mlx-community/chatterbox-turbo-fp16")
    assert engine._model_family == "chatterbox"
    engine._loaded = True
    engine.model = _NoMaxTokens()
    with pytest.raises(AudioGenerationUnboundedError, match="max_tokens") as excinfo:
        engine.generate("hello")
    assert excinfo.value.param == "model"


def test_a_reservation_too_small_for_one_decoder_step_is_refused():
    """A reservation smaller than one decoder step is rejected."""

    import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm

    from vllm_mlx.audio.tts import AudioGenerationUnboundedError
    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    engine = _stride_engine("voxcpm", voxcpm.Model)

    too_small = max_output_seconds_for("x", speed=4.0)
    assert too_small < 7056 / 44_100
    with pytest.raises(AudioGenerationUnboundedError, match="one decoder step") as exc:
        engine._token_budget_for(too_small)
    assert exc.value.param == "speed"

    # The same request at the default speed reserves enough and is allowed.
    assert engine._token_budget_for(max_output_seconds_for("x")) >= 1


def test_a_token_budget_never_loosens_the_backends_own_limit():
    """A request token budget only tightens a backend's own limit."""

    import mlx_audio.tts.models.qwen3_tts.qwen3_tts as qwen3_tts
    import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm

    from vllm_mlx.runtime.audio_capacity import (
        MAX_SPEECH_INPUT_CHARACTERS,
        max_output_seconds_for,
    )

    # The largest request the API accepts, at the slowest permitted speed.
    seconds = max_output_seconds_for("x" * MAX_SPEECH_INPUT_CHARACTERS, speed=0.25)

    for family, model_cls, default in (
        ("voxcpm", voxcpm.Model, 4096),
        ("qwen3_tts", qwen3_tts.Model, 4096),
    ):
        engine = _stride_engine(family, model_cls)
        assert engine._token_budget_for(seconds) == default


def test_the_token_budget_stays_inside_the_reservation():
    """Token-budget rounding never exceeds the reserved duration."""

    import mlx_audio.tts.models.qwen3_tts.qwen3_tts as qwen3_tts
    import mlx_audio.tts.models.voxcpm.voxcpm as voxcpm

    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    for family, model_cls in (
        ("voxcpm", voxcpm.Model),
        ("qwen3_tts", qwen3_tts.Model),
    ):
        fixture = _STRIDE_FIXTURES[family]
        per_token, rate = fixture["samples"], fixture["rate"]
        engine = _stride_engine(family, model_cls)

        for characters in (1, 7, 100, 3_000):
            seconds = max_output_seconds_for("x" * characters)
            budget = engine._token_budget_for(seconds)
            permitted = budget * per_token / rate
            assert permitted <= seconds, (
                f"{family} at {characters} chars: budget buys {permitted:.4f}s "
                f"but only {seconds:.4f}s was reserved"
            )


def _seed_cached_checkpoint(tmp_path, monkeypatch, repo_id, config, *, extra_json=None):
    """Write a HF-cache snapshot for ``repo_id`` containing ``config.json``."""

    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    (snapshot / "config.json").write_text(json.dumps(config))
    for relative_path, payload in (extra_json or {}).items():
        target = snapshot / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )
    return snapshot


def test_an_unservable_request_is_refused_before_the_weights_load(
    tmp_path, monkeypatch
):
    """Cached metadata can reject an unsafe request before loading weights."""

    from vllm_mlx.audio.tts import (
        AudioGenerationUnboundedError,
        precheck_generation_bounds,
    )

    repo = "mlx-community/VoxCPM1.5"
    _seed_cached_checkpoint(
        tmp_path,
        monkeypatch,
        repo,
        {
            "patch_size": 4,
            "audio_vae_config": {
                "sample_rate": 44_100,
                "encoder_rates": [2, 3, 6, 7, 7],
            },
        },
    )

    with pytest.raises(AudioGenerationUnboundedError, match="one decoder step") as exc:
        precheck_generation_bounds(repo, "x", speed=4.0)
    assert exc.value.param == "speed"

    # A request that DOES fit passes the gate and reaches the loader.
    precheck_generation_bounds(repo, "x", speed=1.0)


def test_the_preload_check_stays_silent_when_the_cache_cannot_answer(
    tmp_path, monkeypatch
):
    """Unknown cached metadata does not reject a cold start."""

    from vllm_mlx.audio.tts import precheck_generation_bounds

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    # Nothing cached at all.
    precheck_generation_bounds("mlx-community/VoxCPM1.5", "x", speed=4.0)

    # Cached, but the config carries no stride.
    _seed_cached_checkpoint(
        tmp_path, monkeypatch, "someone/mystery-tts", {"model_type": "voxcpm"}
    )
    precheck_generation_bounds("someone/mystery-tts", "x", speed=4.0)

    # A streaming family is never pre-checked: the per-chunk clamp bounds it.
    precheck_generation_bounds("mlx-community/Kokoro-82M-bf16", "x", speed=4.0)


def test_the_preload_check_reads_the_same_strides_as_the_loaded_model(
    tmp_path, monkeypatch
):
    """Cached and loaded probes use the same family stride definitions."""

    from vllm_mlx.audio.tts import _cached_sample_rate, _cached_samples_per_token

    cases = {
        "mlx-community/VoxCPM1.5": (
            "voxcpm",
            {
                "patch_size": 4,
                "audio_vae_config": {
                    "sample_rate": 44_100,
                    "encoder_rates": [2, 3, 6, 7, 7],
                },
            },
            {},
        ),
        "mlx-community/Qwen3-TTS-Flash": (
            "qwen3_tts",
            {"model_type": "qwen3_tts"},
            {
                "speech_tokenizer/config.json": {
                    "output_sample_rate": 24_000,
                    "decode_upsample_rate": 1920,
                }
            },
        ),
        "mlx-community/chatterbox-fp16": (
            "chatterbox",
            {"sample_rate": 24_000},
            {},
        ),
        "mlx-community/VibeVoice-Realtime-0.5B-4bit": (
            "vibevoice",
            {
                "sample_rate": 24_000,
                "acoustic_tokenizer_config": {"encoder_ratios": [8, 5, 5, 4, 2, 2]},
            },
            {},
        ),
        "mlx-community/IndexTTS-2-mlx": (
            "indextts",
            {"sample_rate": 24_000, "bigvgan": {"upsample_rates": [8, 8, 2, 2]}},
            {},
        ),
    }

    for repo, (family, config, extra_json) in cases.items():
        _seed_cached_checkpoint(
            tmp_path, monkeypatch, repo, config, extra_json=extra_json
        )
        assert (
            _cached_samples_per_token(repo, family)
            == _STRIDE_FIXTURES[family]["samples"]
        ), family
        assert _cached_sample_rate(repo, family) == _STRIDE_FIXTURES[family]["rate"]


def test_no_token_budget_is_sent_to_backends_that_reject_it():
    """Streaming backends without token-limit support use chunk clamping."""

    import numpy as np

    from vllm_mlx.audio.tts import _SINGLE_YIELD_FAMILIES, TTSEngine

    class _StrictModel:
        sample_rate = 24_000

        def __init__(self):
            self.calls: list[dict] = []

        def generate(self, *, text, voice=None, speed=1.0, lang_code=None):
            self.calls.append(
                {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "lang_code": lang_code,
                }
            )
            # Streamed in pieces, so the clamp bounds it as each arrives.
            for _ in range(1000):
                yield type(
                    "Result",
                    (),
                    {
                        "audio": np.ones(100_000, dtype=np.float32),
                        "sample_rate": 24_000,
                    },
                )()

    engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
    assert engine._model_family not in _SINGLE_YIELD_FAMILIES
    engine._loaded = True
    engine.model = _StrictModel()

    text = "hello"
    output = engine.generate(text)

    (call,) = engine.model.calls
    assert "max_tokens" not in call
    # The clamp is the bound for a streaming backend.
    assert output.audio.size == _ceiling_samples(text, sample_rate=24_000)


def test_waveform_conversion_does_not_allocate_a_python_list():
    """MLX waveform conversion avoids an intermediate Python list."""

    import tracemalloc

    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.audio.tts import _to_float32

    samples = 1_000_000
    waveform = mx.ones((samples,), dtype=mx.float32)
    mx.eval(waveform)
    float32_bytes = samples * 4

    tracemalloc.start()
    converted = _to_float32(waveform, mx)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert converted.dtype.name == "float32"
    assert converted.shape == (samples,)
    # One array-sized copy, with slack for numpy bookkeeping. The list-based
    # conversion measured >8x here.
    assert peak < float32_bytes * 2, (
        f"conversion peaked at {peak} bytes for a {float32_bytes}-byte waveform"
    )


def test_bfloat16_waveforms_convert_without_a_python_list():
    """bfloat16 is cast on MLX rather than through a Python list."""

    mx = pytest.importorskip("mlx.core")

    from vllm_mlx.audio.tts import _to_float32

    waveform = mx.ones((512,), dtype=mx.bfloat16)
    mx.eval(waveform)

    converted = _to_float32(waveform, mx)

    assert converted.dtype.name == "float32"
    assert converted.shape == (512,)


def test_custom_npz_repo_is_measured_without_registry_membership(tmp_path, monkeypatch):
    """NPZ completeness follows disk layout rather than registry membership."""

    repo_id = "someone/custom-asr-npz"
    repo_root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / "sha-a"
    snapshot.mkdir(parents=True)
    (repo_root / "refs").mkdir()
    (repo_root / "refs" / "main").write_text("sha-a")
    (snapshot / "config.json").write_text("{}")
    (snapshot / "weights.npz").write_bytes(b"x" * 8192)

    monkeypatch.setattr(
        "huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path), raising=False
    )

    capacity = resolve_audio_role_capacity(repo_id)

    assert capacity.capacity_source == "local_cache"
    assert capacity.weight_bytes == 8192 + 2
