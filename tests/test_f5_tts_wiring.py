"""Regression tests for the optional F5-TTS integration."""

from __future__ import annotations

import inspect
import io
import pkgutil
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from vllm_mlx.api.models import AudioSpeechRequest
from vllm_mlx.audio.tts import TTSEngine
from vllm_mlx.routes.audio import _allowed_voices_for, _decode_tts_ref_audio


def test_f5_family_is_detected() -> None:
    assert TTSEngine("lucasnewman/f5-tts-mlx")._model_family == "f5"
    assert _allowed_voices_for("lucasnewman/f5-tts-mlx") == ["clone"]


def test_installed_f5_sample_contract() -> None:
    f5 = pytest.importorskip("f5_tts_mlx.cfm")
    assert pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
    parameters = inspect.signature(f5.F5TTS.sample).parameters
    assert {"cond", "text", "duration", "steps", "speed", "cfg_strength"} <= set(
        parameters
    )


def test_f5_clone_requires_audio_and_transcript_together() -> None:
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    with pytest.raises(ValueError, match="both ref_audio and ref_text"):
        engine._generate_f5("hello", "/tmp/unused.wav", None, 1.0)


def test_f5_api_reference_is_base64_and_requires_a_pair() -> None:
    encoded = "data:audio/wav;base64,UklGRg=="
    request = AudioSpeechRequest(input="你好", ref_audio=encoded, ref_text="参考")
    assert _decode_tts_ref_audio(request.ref_audio or "") == b"RIFF"

    with pytest.raises(ValidationError, match="provided together"):
        AudioSpeechRequest(input="你好", ref_audio=encoded)
    with pytest.raises(ValueError, match="valid base64"):
        _decode_tts_ref_audio("not base64")


def test_f5_generation_uses_safe_in_memory_default_and_speed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mx = SimpleNamespace(
        array=lambda value: np.asarray(value),
        sqrt=np.sqrt,
        mean=np.mean,
        square=np.square,
        eval=lambda value: None,
        expand_dims=np.expand_dims,
    )
    mlx_module = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    for name, value in vars(fake_mx).items():
        setattr(mlx_core, name, value)
    mlx_module.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    generate_module = ModuleType("f5_tts_mlx.generate")
    generate_module.FRAMES_PER_SEC = 100
    generate_module.SAMPLE_RATE = 24_000
    generate_module.TARGET_RMS = 0.1
    generate_module.convert_char_to_pinyin = lambda texts: texts
    duration = MagicMock(return_value=2.0)
    generate_module.estimated_duration = duration
    package = ModuleType("f5_tts_mlx")
    monkeypatch.setitem(sys.modules, "f5_tts_mlx", package)
    monkeypatch.setitem(sys.modules, "f5_tts_mlx.generate", generate_module)

    monkeypatch.setattr("pkgutil.get_data", lambda *_: b"packaged-wave", raising=True)
    opened = _install_fake_soundfile(
        monkeypatch, channels=1, frames=240, data=np.ones(240, dtype=np.float32)
    )

    model = MagicMock()
    model.sample.return_value = (np.ones(480, dtype=np.float32), None)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.model = model

    output = engine._generate_f5("你好", None, None, 1.5)

    # The packaged default reference is read from an in-memory buffer (no
    # predictable temp path).
    assert isinstance(opened["source"], io.BytesIO)
    duration.assert_called_once()
    assert duration.call_args.args[-1] == 1.5
    assert model.sample.call_args.kwargs["speed"] == 1.5
    assert output.sample_rate == 24_000
    assert output.audio.shape == (240,)


def _install_fake_f5_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register lightweight fakes for ``mlx`` and ``f5_tts_mlx.generate`` so the
    F5 code path runs without real weights or the native mlx runtime."""
    fake_mx = ModuleType("mlx.core")
    fake_mx.array = np.asarray
    fake_mx.sqrt = np.sqrt
    fake_mx.mean = np.mean
    fake_mx.square = np.square
    fake_mx.eval = lambda value: None
    fake_mx.expand_dims = np.expand_dims
    mlx_module = ModuleType("mlx")
    mlx_module.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    generate_module = ModuleType("f5_tts_mlx.generate")
    generate_module.FRAMES_PER_SEC = 100
    generate_module.SAMPLE_RATE = 24_000
    generate_module.TARGET_RMS = 0.1
    generate_module.convert_char_to_pinyin = lambda texts: texts
    generate_module.estimated_duration = lambda *_: 1.0
    monkeypatch.setitem(sys.modules, "f5_tts_mlx", ModuleType("f5_tts_mlx"))
    monkeypatch.setitem(sys.modules, "f5_tts_mlx.generate", generate_module)


def _install_fake_soundfile(
    monkeypatch: pytest.MonkeyPatch,
    *,
    samplerate: int = 24_000,
    channels: int = 1,
    frames: int = 240,
    data: np.ndarray | None = None,
) -> dict:
    """Fake ``soundfile.SoundFile`` as a single-open context manager so the F5
    read path runs without a real file. The production code validates metadata
    and reads samples through one handle (no separate ``info``/``read`` opens);
    the fake mirrors that. Returns a dict whose ``source`` key records the
    object that was opened. ``data=None`` means ``read()`` must not be reached
    (e.g. a metadata guard should reject first)."""
    opened: dict = {}

    class _FakeSoundFile:
        def __init__(self, source, *args, **kwargs):
            opened["source"] = source
            if hasattr(source, "seek"):
                source.seek(0)
            self.samplerate = samplerate
            self.channels = channels
            self.frames = frames

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args, **kwargs):
            if data is None:
                raise AssertionError("SoundFile.read() should not be reached")
            return np.array(data)

    # ``soundfile`` belongs to the optional audio extra and is intentionally
    # absent from the core test environment. Install the fake module at the
    # import boundary instead of asking monkeypatch to import the real package
    # before replacing one attribute.
    fake_soundfile = ModuleType("soundfile")
    fake_soundfile.SoundFile = _FakeSoundFile
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    return opened


def test_f5_downmixes_stereo_reference_to_mono(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stereo reference is downmixed to mono and accepted (was rejected)."""
    _install_fake_f5_env(monkeypatch)

    # soundfile returns shape (frames, channels) for multi-channel audio. Use
    # distinct per-channel values so the test can tell averaging apart from
    # merely dropping a channel: L=0.2, R=0.4 -> mono mean 0.3. (0.3 exceeds
    # TARGET_RMS 0.1, so no RMS renormalization is applied and the conditioning
    # values stay 0.3.)
    stereo = np.stack(
        [np.full(240, 0.2, dtype=np.float32), np.full(240, 0.4, dtype=np.float32)],
        axis=1,
    )
    _install_fake_soundfile(monkeypatch, channels=2, frames=240, data=stereo)

    model = MagicMock()
    model.sample.return_value = (np.ones(480, dtype=np.float32), None)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.model = model

    output = engine._generate_f5("hello", "ref.wav", "reference", 1.0)

    # The reference handed to sample() must be batched mono: (1, frames), not
    # (1, frames, channels) — proving the stereo track was collapsed to mono.
    cond = np.asarray(model.sample.call_args.args[0])
    assert cond.shape == (1, 240)
    # It must be the per-channel *average* (0.3), not a dropped channel (0.2/0.4).
    assert np.allclose(cond, 0.3)
    assert output.audio.shape == (240,)


def test_f5_rejects_reference_with_too_many_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A header advertising more than stereo is rejected before decoding, so a
    pathological channel count can't force an oversized allocation."""
    _install_fake_f5_env(monkeypatch)

    # data=None => read() raises if reached; the channel guard must reject first.
    _install_fake_soundfile(monkeypatch, channels=64, frames=240, data=None)

    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    with pytest.raises(ValueError, match="mono or stereo"):
        engine._generate_f5("hello", "ref.wav", "reference", 1.0)


def test_f5_rejects_near_silent_reference_below_rms_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-silent reference (tiny positive RMS below the floor) is rejected.

    The prior guard only rejected ``rms <= 0``, so a near-silent clip slipped
    through and got amplified into full-scale noise. The RMS floor must reject
    it with a clear error.
    """
    _install_fake_f5_env(monkeypatch)

    # RMS == 5e-5, which is > 0 (so the old ``rms <= 0`` guard misses it) but
    # below the 1e-4 floor.
    near_silent = np.full(240, 5e-5, dtype=np.float32)
    _install_fake_soundfile(monkeypatch, channels=1, frames=240, data=near_silent)

    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    with pytest.raises(ValueError, match="non-silent"):
        engine._generate_f5("hello", "ref.wav", "reference", 1.0)


def test_f5_clamps_a_runaway_duration_estimate_to_the_output_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F5 samples its whole duration in ONE call, so the bound must land on the
    estimate (#2305).

    ``estimated_duration`` scales the reference clip's length by the
    ref_text-to-text character ratio, and both inputs are caller-controlled: a
    30-second reference (the maximum this route accepts) with a one-character
    ``ref_text`` produces a ratio large enough to turn a short ``text`` into
    minutes of audio. The request reservation is sized from the input text
    only, so an unclamped estimate is an unbudgeted allocation — and no
    per-chunk check downstream can intervene, because there are no chunks.
    """
    _install_fake_f5_env(monkeypatch)

    from vllm_mlx.runtime.audio_capacity import max_output_seconds_for

    # 30 s of reference at 24 kHz — the longest the guards permit.
    ref_frames = 24_000 * 30
    reference = np.full(ref_frames, 0.5, dtype=np.float32)
    _install_fake_soundfile(monkeypatch, channels=1, frames=ref_frames, data=reference)

    # A one-character ref_text against short text: the real heuristic's ratio
    # blows up here. Simulate the runaway directly.
    sys.modules["f5_tts_mlx.generate"].estimated_duration = lambda *_: 3600.0

    model = MagicMock()
    model.sample.return_value = (np.ones(ref_frames + 480, dtype=np.float32), None)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.model = model

    text = "hello"
    engine._generate_f5(text, "ref.wav", "x", 1.0)

    # FRAMES_PER_SEC is 100 in the fake env.
    permitted = max_output_seconds_for(text, speed=1.0) * 100
    ref_prefix = ref_frames / 24_000 * 100
    requested = model.sample.call_args.kwargs["duration"]

    # The unclamped estimate was 3600 s * 100 = 360_000 frames.
    assert requested < 360_000
    assert requested <= ref_prefix + permitted + 1


def test_f5_never_asks_for_less_than_the_reference_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clamp adds the reference prefix back rather than eating it out of
    the caller's budget: ``sample()`` decodes the prefix alongside the speech
    and ``_generate_f5`` trims it afterwards, so a ceiling applied to the total
    would leave no frames to generate into."""
    _install_fake_f5_env(monkeypatch)

    _install_fake_soundfile(
        monkeypatch, channels=1, frames=240, data=np.full(240, 0.5, dtype=np.float32)
    )
    sys.modules["f5_tts_mlx.generate"].estimated_duration = lambda *_: 0.0

    model = MagicMock()
    model.sample.return_value = (np.ones(480, dtype=np.float32), None)
    engine = TTSEngine("lucasnewman/f5-tts-mlx")
    engine.model = model

    engine._generate_f5("a", "ref.wav", "reference", 1.0)

    # 240 frames at 24 kHz = 0.01 s = 1 frame at FRAMES_PER_SEC=100.
    assert model.sample.call_args.kwargs["duration"] >= 2
