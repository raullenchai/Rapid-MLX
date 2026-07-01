# SPDX-License-Identifier: Apache-2.0
"""F-K-WHISPER-961 — Whisper silence hallucination guard.

kootenayalex reported (issue #961) that ``STTEngine.transcribe`` on
12 s of digital silence returns ``"Thank you."`` — a canonical Whisper
silence hallucination. The documented anti-hallucination guards
(``no_speech_threshold`` / ``logprob_threshold`` /
``hallucination_silence_threshold``) inside
``mlx_audio.stt.models.whisper.Model.generate`` are inert in practice:
on pure silence the model returns ``no_speech_prob ≈ 1e-11`` and
``avg_logprob ≈ -0.26``, and the AND-gate skip condition never fires.

The fix at ``vllm_mlx/audio/stt.py`` runs the bundled Silero VAD
(via ``mlx_audio.vad``) before Whisper:

* No speech → return an empty ``TranscriptionResult`` without invoking
  Whisper.
* Speech present → trim to the speech span, invoke Whisper on the
  trimmed waveform, then shift each returned segment (and per-word)
  timestamp back by the trim offset so callers see absolute times.

These tests stub the VAD + Whisper models to exercise the guard's
control flow deterministically without downloading multi-hundred-MB
weights on the CI runner.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Fake VAD + Whisper doubles. Both mimic the shape the production code
# consumes: ``vad.get_speech_timestamps(...)`` returns a list of
# ``{start,end}`` dicts (seconds), and ``whisper.generate(audio, ...)``
# returns an ``STTOutput``-shaped object with ``text``, ``segments``,
# ``language``.
# ---------------------------------------------------------------------------


class _FakeVAD:
    """VAD stub that returns a pre-configured speech-timestamp list.

    ``timestamps`` is a list of {"start": s, "end": s} dicts (seconds).
    The test controls the list to simulate every VAD outcome shape:
    empty (pure silence), single span (leading/trailing silence trim),
    multiple spans (mid-silence gap).
    """

    def __init__(self, timestamps: list[dict]):
        self._timestamps = timestamps
        self.call_count = 0
        self.last_kwargs: dict = {}

    def get_speech_timestamps(self, audio, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        # We copy so callers can't mutate our fixture in place.
        return [dict(ts) for ts in self._timestamps]


class _FakeWhisperResult:
    """Mimics ``mlx_audio.stt.models.whisper.STTOutput`` — a dataclass
    with ``text``, ``segments`` (list of dicts), ``language``."""

    def __init__(self, text: str, segments: list[dict], language: str = "en"):
        self.text = text
        self.segments = segments
        self.language = language


class _FakeWhisperModel:
    """Records every ``generate(...)`` call so the tests can assert both
    that Whisper was NOT called on pure silence AND that the correct
    audio input (trimmed waveform vs. original path) was passed."""

    def __init__(self, result: _FakeWhisperResult | None = None):
        # Default: a single 0.0-2.0s "hello world" segment.
        self._result = result or _FakeWhisperResult(
            text="hello world",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "hello world"},
            ],
            language="en",
        )
        self.calls: list[tuple[object, dict]] = []
        # Ensure the ``_ensure_whisper_processor`` gate is a no-op.
        self._processor = object()

    def generate(self, audio, **kwargs):
        # Copy so mutation on our side doesn't upset test assertions.
        self.calls.append((audio, dict(kwargs)))
        return self._result


class _FakeMxArray:
    """Lightweight stand-in for ``mx.array`` that supports the two
    operations the VAD helper touches: ``.shape`` and slice indexing.

    We deliberately avoid importing ``mlx.core`` in this test to keep
    the unit test isolated from Metal / MLX runtime state — the guard
    itself only uses ``waveform.shape[-1]`` and ``waveform[start:end]``.
    """

    def __init__(self, length_samples: int):
        self._length = length_samples

    @property
    def shape(self):  # noqa: ANN201
        return (self._length,)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self._length
            return _FakeMxArray(max(0, stop - start))
        raise TypeError(f"unsupported index: {key!r}")


# ---------------------------------------------------------------------------
# Fixture: build a fully-stubbed STTEngine that never touches the network
# or the mlx runtime, and expose the fake VAD + fake Whisper for
# per-test assertions.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_engine(monkeypatch, tmp_path):
    """Yields ``(engine, fake_vad, fake_whisper, fake_audio_path)``.

    * ``fake_vad.get_speech_timestamps`` returns whatever
      ``fake_vad._timestamps`` holds — mutate that list before calling
      ``engine.transcribe`` to steer the guard's branch.
    * ``fake_whisper.calls`` is the recorded generate() history.
    """
    from vllm_mlx.audio import stt as stt_mod

    # Reset the module-level VAD cache between tests so each fixture
    # gets a fresh singleton pinned to its own _FakeVAD.
    monkeypatch.setattr(stt_mod, "_VAD_MODEL_CACHE", None, raising=True)
    monkeypatch.setattr(stt_mod, "_VAD_IMPORT_UNAVAILABLE", False, raising=True)
    monkeypatch.setattr(stt_mod, "_VAD_LOAD_FAILURE_LOGGED", False, raising=True)
    # Ensure env override does not leak in from the host.
    monkeypatch.delenv("RAPID_MLX_STT_VAD_PRETRIM", raising=False)

    fake_vad = _FakeVAD(timestamps=[])
    fake_whisper = _FakeWhisperModel()

    def _fake_get_vad_model():
        return fake_vad

    def _fake_load_audio(_path):
        # 12 s of audio at 16 kHz for the pure-silence + short-tail
        # tests; individual tests can override by monkeypatching this
        # again with a different length.
        return _FakeMxArray(length_samples=16_000 * 12)

    # Patch the VAD lookup + audio decode inside stt.py.
    monkeypatch.setattr(stt_mod, "_get_vad_model", _fake_get_vad_model)

    # Route ``from mlx_audio.stt.utils import load_audio`` inside the
    # ``_maybe_vad_trim`` helper to our fake. We do this by patching
    # the module cache — the helper imports lazily on every call.
    import sys as _sys
    import types as _types

    fake_stt_utils = _types.ModuleType("mlx_audio.stt.utils")
    fake_stt_utils.load_audio = _fake_load_audio
    fake_stt_utils.load_model = lambda *_a, **_kw: fake_whisper
    monkeypatch.setitem(_sys.modules, "mlx_audio.stt.utils", fake_stt_utils)

    # Build the engine.
    eng = stt_mod.STTEngine("mlx-community/whisper-large-v3-turbo")
    eng.model = fake_whisper
    eng._loaded = True

    # A dummy file path — the fake load_audio ignores it, so it need
    # not exist on disk (but we materialise a stub to keep any
    # downstream ``str(Path)`` conversions predictable).
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFFsize")

    yield eng, fake_vad, fake_whisper, str(audio_path), _fake_load_audio, monkeypatch


# ---------------------------------------------------------------------------
# Test coverage — one class per behaviour, per the task's spec.
# ---------------------------------------------------------------------------


class TestSilenceReturnsEmpty:
    """Pure-silence input must NOT be handed to Whisper — the whole
    point of the guard. Whisper.generate must not be called; the
    returned result must be an empty ``TranscriptionResult``."""

    def test_pure_silence_returns_empty_string(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        # VAD reports no speech at all.
        fake_vad._timestamps = []

        result = eng.transcribe(path)

        assert result.text == ""
        assert result.segments == []
        assert result.language is None
        assert result.duration == 0.0
        # Critical: Whisper was NOT invoked. This is the whole
        # anti-hallucination invariant.
        assert fake_whisper.calls == []


class TestRealSpeechStillTranscribes:
    """Speech-containing input must reach Whisper and produce the same
    text as pre-fix — the guard must never eat legitimate content."""

    def test_real_speech_still_transcribes(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        # VAD reports a single 0-2 s speech span (matches the fake
        # Whisper result's segment span, so no trim/offset artefacts).
        fake_vad._timestamps = [{"start": 0.0, "end": 2.0}]

        result = eng.transcribe(path)

        assert result.text == "hello world"
        assert result.language == "en"
        assert result.segments and len(result.segments) == 1
        # Whisper WAS called — with the trimmed waveform (mx.array-
        # shaped), not the path string.
        assert len(fake_whisper.calls) == 1
        audio_in, _kwargs = fake_whisper.calls[0]
        assert not isinstance(audio_in, str)
        assert hasattr(audio_in, "shape")


class TestTrailingSilenceTrimmed:
    """Speech in [0, 2] + 10 s trailing silence must be trimmed so
    Whisper never sees the tail — the pre-fix bug was that Whisper
    would emit "Thank you." over the tail dead-air."""

    def test_speech_with_trailing_silence_stops_at_speech_end(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        # Real speech 0-2 s inside a 12 s clip → VAD reports the span.
        fake_vad._timestamps = [{"start": 0.0, "end": 2.0}]

        result = eng.transcribe(path)

        # Whisper received the trimmed waveform, not the original 12 s.
        audio_in, _ = fake_whisper.calls[0]
        assert audio_in.shape[0] <= int(16_000 * (2.0 + 0.4 + 0.01)), (
            "Trimmed waveform must not exceed [speech_end + 2 * pad]; "
            f"got {audio_in.shape[0]} samples"
        )
        # And the last segment.end stays close to the real speech end
        # — no hallucinated tail extending past 2.5 s.
        last_end = result.segments[-1]["end"]
        assert last_end < 2.0 + 0.5, f"tail hallucination: end={last_end}"


class TestVADDisabledViaEnv:
    """Env override ``RAPID_MLX_STT_VAD_PRETRIM=0`` must fully bypass
    the guard: VAD is never asked, Whisper receives the original path."""

    def test_vad_disabled_via_env_pass_through(self, stub_engine, monkeypatch):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        monkeypatch.setenv("RAPID_MLX_STT_VAD_PRETRIM", "0")

        result = eng.transcribe(path)

        # VAD was never called.
        assert fake_vad.call_count == 0
        # Whisper was called with the ORIGINAL path (pre-fix path).
        assert len(fake_whisper.calls) == 1
        audio_in, _kwargs = fake_whisper.calls[0]
        assert audio_in == path
        # And the transcription flowed through unchanged.
        assert result.text == "hello world"

    @pytest.mark.parametrize("value", ["false", "no", "off", "FALSE", "Off"])
    def test_env_disable_accepts_common_falsy_strings(
        self, stub_engine, monkeypatch, value
    ):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        monkeypatch.setenv("RAPID_MLX_STT_VAD_PRETRIM", value)

        eng.transcribe(path)

        assert fake_vad.call_count == 0, f"env value {value!r} should disable VAD"


class TestAbsoluteTimestampsPreserved:
    """When VAD trims leading silence, returned segment timestamps must
    be shifted back to absolute (file-relative) time. Otherwise a caller
    stitching multiple transcripts together would see spurious 0.0-s
    starts for every clip that had a silent lead-in."""

    def test_absolute_timestamps_preserved_when_vad_trims_leading_silence(
        self, stub_engine
    ):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        # VAD says speech starts 3.0 s in and ends 5.0 s in.
        fake_vad._timestamps = [{"start": 3.0, "end": 5.0}]
        # Whisper (on the trimmed span) reports segments starting at
        # 0.0 relative to its input.
        fake_whisper._result = _FakeWhisperResult(
            text="hello world",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "hello world"},
            ],
            language="en",
        )

        result = eng.transcribe(path)

        # Trim offset was ``max(0, 3.0 - 0.2) = 2.8``. Segment start
        # relative to the ORIGINAL file must be 0.0 + 2.8 = 2.8 s.
        seg = result.segments[0]
        assert seg["start"] >= 2.7 and seg["start"] <= 2.9, (
            f"segment.start must be shifted back to ~2.8 s; got {seg['start']}"
        )
        assert seg["end"] >= 4.7 and seg["end"] <= 4.9

    def test_word_level_timestamps_also_shifted(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        fake_vad._timestamps = [{"start": 3.0, "end": 5.0}]
        fake_whisper._result = _FakeWhisperResult(
            text="hello world",
            segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "hello"},
                        {"start": 0.5, "end": 2.0, "word": "world"},
                    ],
                },
            ],
        )

        result = eng.transcribe(path)

        # Codex r1 NIT: assert exact shifted values, not just lower
        # bounds — a double-shift bug would still pass a >= check.
        # Trim offset = max(0, 3.0 - 0.2) = 2.8.
        words = result.segments[0]["words"]
        assert words[0]["start"] == pytest.approx(2.8, abs=1e-6)
        assert words[0]["end"] == pytest.approx(3.3, abs=1e-6)
        assert words[1]["start"] == pytest.approx(3.3, abs=1e-6)
        assert words[1]["end"] == pytest.approx(4.8, abs=1e-6)


class TestKwargOverrideDisables:
    """The ``enable_vad_pretrim=False`` kwarg on ``STTEngine`` must fully
    bypass the guard even without an env override."""

    def test_enable_vad_pretrim_false_kwarg_disables(self, stub_engine, monkeypatch):
        _eng, fake_vad, fake_whisper, path, _load_audio, mp = stub_engine
        from vllm_mlx.audio import stt as stt_mod

        eng2 = stt_mod.STTEngine(
            "mlx-community/whisper-large-v3-turbo",
            enable_vad_pretrim=False,
        )
        eng2.model = fake_whisper
        eng2._loaded = True

        eng2.transcribe(path)

        assert fake_vad.call_count == 0
        # Whisper was called with the ORIGINAL path (pre-fix behaviour).
        audio_in, _ = fake_whisper.calls[0]
        assert audio_in == path


class TestVADImportFailureFallsBack:
    """If ``mlx_audio.vad`` is not importable (e.g. user installed
    rapid-mlx without the ``[audio]`` extra but supplied their own
    Whisper build), the guard must gracefully skip and pass through
    to the original path — never raise."""

    def test_vad_import_failure_falls_back(self, monkeypatch, tmp_path):
        # Rebuild a clean engine that goes through the REAL
        # ``_get_vad_model`` path, then break the ``mlx_audio.vad``
        # import so the helper hits the ``except ImportError`` branch.
        from vllm_mlx.audio import stt as stt_mod

        importlib.reload(stt_mod)
        monkeypatch.setattr(stt_mod, "_VAD_MODEL_CACHE", None)
        monkeypatch.setattr(stt_mod, "_VAD_IMPORT_UNAVAILABLE", False)
        monkeypatch.setattr(stt_mod, "_VAD_LOAD_FAILURE_LOGGED", False)
        monkeypatch.delenv("RAPID_MLX_STT_VAD_PRETRIM", raising=False)

        import sys as _sys

        # Kill any cached mlx_audio.vad and mask further imports.
        for name in list(_sys.modules):
            if name.startswith("mlx_audio.vad"):
                monkeypatch.delitem(_sys.modules, name, raising=False)

        import builtins

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "mlx_audio.vad" or name.startswith("mlx_audio.vad."):
                raise ImportError("simulated: mlx_audio.vad not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        fake_whisper = _FakeWhisperModel()
        eng = stt_mod.STTEngine("mlx-community/whisper-large-v3-turbo")
        eng.model = fake_whisper
        eng._loaded = True

        path = str(tmp_path / "audio.wav")
        (tmp_path / "audio.wav").write_bytes(b"RIFFsize")

        # Must not raise. Whisper receives the original path (fallback).
        result = eng.transcribe(path)

        assert result.text == "hello world"
        assert len(fake_whisper.calls) == 1
        audio_in, _ = fake_whisper.calls[0]
        assert audio_in == path


class TestMalformedVADOutputFallsBack:
    """Codex r1 BLOCKING #2 defense: if the VAD helper returns
    malformed timestamps (missing ``start`` / ``end`` keys, wrong types,
    empty dicts), the guard must fall back to unmodified transcription
    instead of raising ``KeyError`` out of ``transcribe()``."""

    def test_missing_start_key_falls_back(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        fake_vad._timestamps = [{"end": 2.0}]  # missing 'start'

        # Must not raise. Whisper is invoked with the original path
        # as the fallback (the reporter's pre-fix behaviour is
        # preferable to a 500).
        result = eng.transcribe(path)

        assert result.text == "hello world"
        audio_in, _ = fake_whisper.calls[0]
        assert audio_in == path

    def test_missing_end_key_falls_back(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        fake_vad._timestamps = [{"start": 1.0}]  # missing 'end'

        result = eng.transcribe(path)

        assert result.text == "hello world"
        audio_in, _ = fake_whisper.calls[0]
        assert audio_in == path

    def test_none_valued_timestamp_falls_back(self, stub_engine):
        eng, fake_vad, fake_whisper, path, *_ = stub_engine
        fake_vad._timestamps = [{"start": None, "end": 2.0}]  # TypeError

        result = eng.transcribe(path)

        assert result.text == "hello world"
        audio_in, _ = fake_whisper.calls[0]
        assert audio_in == path


class TestTransientLoadFailureRetries:
    """Codex r1 BLOCKING #1 defense: a transient ``vad_load(...)``
    failure must NOT permanently disable the guard for the process
    lifetime. Import failures are permanent (weights aren't installed)
    and DO stay cached, but transient load errors retry on every
    subsequent call.
    """

    def test_transient_load_failure_retries_on_next_call(self, monkeypatch):
        from vllm_mlx.audio import stt as stt_mod

        # Fresh module state.
        monkeypatch.setattr(stt_mod, "_VAD_MODEL_CACHE", None)
        monkeypatch.setattr(stt_mod, "_VAD_IMPORT_UNAVAILABLE", False)
        monkeypatch.setattr(stt_mod, "_VAD_LOAD_FAILURE_LOGGED", False)

        # Fake ``mlx_audio.vad.load`` that fails the first call, then
        # succeeds the second — mimics a network hiccup that recovers.
        import sys as _sys
        import types as _types

        state = {"calls": 0}

        def _flaky_load(_repo):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("simulated: HF 502")
            return _FakeVAD(timestamps=[])

        fake_vad_pkg = _types.ModuleType("mlx_audio.vad")
        fake_vad_pkg.load = _flaky_load
        monkeypatch.setitem(_sys.modules, "mlx_audio.vad", fake_vad_pkg)

        # First call — transient failure.
        assert stt_mod._get_vad_model() is None
        assert state["calls"] == 1
        # Permanent-unavailable flag must NOT be latched.
        assert stt_mod._VAD_IMPORT_UNAVAILABLE is False

        # Second call — retries, succeeds.
        vad = stt_mod._get_vad_model()
        assert vad is not None
        assert state["calls"] == 2

    def test_permanent_import_failure_is_cached(self, monkeypatch):
        """ImportError → cached, no retries (module isn't installed
        and can't become installed without a process restart)."""
        from vllm_mlx.audio import stt as stt_mod

        monkeypatch.setattr(stt_mod, "_VAD_MODEL_CACHE", None)
        monkeypatch.setattr(stt_mod, "_VAD_IMPORT_UNAVAILABLE", False)
        monkeypatch.setattr(stt_mod, "_VAD_LOAD_FAILURE_LOGGED", False)

        import sys as _sys

        # Ensure any cached mlx_audio.vad module is gone so the import
        # actually runs. Then block re-imports of it.
        for name in list(_sys.modules):
            if name.startswith("mlx_audio.vad"):
                monkeypatch.delitem(_sys.modules, name, raising=False)

        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "mlx_audio.vad" or name.startswith("mlx_audio.vad."):
                raise ImportError("simulated: not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)

        assert stt_mod._get_vad_model() is None
        assert stt_mod._VAD_IMPORT_UNAVAILABLE is True

        # Now un-block the import and call again — must NOT retry,
        # because permanent failures stay cached.
        monkeypatch.setattr(builtins, "__import__", real_import)
        assert stt_mod._get_vad_model() is None


class TestUpstreamWhisperInputContract:
    """Codex r1 NIT #3 defense: pin the upstream Whisper
    ``generate()`` input-shape contract so if
    ``mlx_audio.stt.models.whisper.Model._prepare_audio`` ever narrows
    its accepted input types we fail loudly here instead of in
    production. This tests the real installed ``mlx_audio`` (not our
    fake) but does NOT download weights.
    """

    def test_whisper_prepare_audio_accepts_np_and_mx(self):
        try:
            from mlx_audio.stt.models.whisper.whisper import Model
        except ImportError:
            pytest.skip("mlx_audio not installed; contract check n/a")

        import inspect
        import typing as _typing

        sig = inspect.signature(Model._prepare_audio)
        audio_param = sig.parameters.get("audio")
        assert audio_param is not None, (
            "Upstream ``Model._prepare_audio`` renamed the ``audio`` "
            "parameter — VAD trim path passes a positional array here."
        )
        annotation = audio_param.annotation
        # Accept both the raw ``Union[str, np.ndarray, mx.array]`` typing
        # and any subclass that still allows a bare ndarray/mx.array
        # positional. If the signature narrows to ``str`` only, this
        # would need to catch it — flag by asserting the annotation
        # isn't the bare string type.
        assert annotation is not str, (
            "Upstream ``Model._prepare_audio`` narrowed to str-only. "
            "The VAD trim path can no longer pass a trimmed waveform."
        )
        # Fallback: if the annotation is a Union, at least one of the
        # arms must be a non-``str`` type (numpy or mx.array).
        origin = _typing.get_origin(annotation)
        if origin is not None:
            args = _typing.get_args(annotation)
            non_str = [a for a in args if a is not str]
            assert non_str, (
                "Upstream ``Model._prepare_audio`` audio Union no longer "
                f"has a non-str arm: {annotation}"
            )


class TestParakeetEngineSkipsVAD:
    """Parakeet has its own silence behaviour and is out of scope for
    #961 — the guard must be gated so it never fires on non-Whisper
    engines regardless of the ``enable_vad_pretrim`` default."""

    def test_parakeet_engine_skips_vad(self, stub_engine):
        _eng, fake_vad, fake_whisper, path, _la, _mp = stub_engine
        from vllm_mlx.audio import stt as stt_mod

        eng_p = stt_mod.STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
        eng_p.model = fake_whisper
        eng_p._loaded = True
        assert eng_p._is_parakeet is True

        eng_p.transcribe(path)

        # VAD helper was NOT consulted for Parakeet.
        assert fake_vad.call_count == 0
        audio_in, kwargs = fake_whisper.calls[0]
        assert audio_in == path
        # Parakeet strips language / task kwargs per pre-fix behaviour.
        assert "language" not in kwargs
        assert "task" not in kwargs


# ---------------------------------------------------------------------------
# Sanity check on ``_vad_pretrim_disabled_by_env`` directly — the
# helper is small enough to unit-test in isolation.
# ---------------------------------------------------------------------------


class TestEnvHelper:
    @pytest.mark.parametrize(
        "val,expected",
        [
            (None, False),
            ("", False),
            ("1", False),
            ("true", False),
            ("yes", False),
            ("on", False),
            ("0", True),
            ("false", True),
            ("FALSE", True),
            ("no", True),
            ("off", True),
            (" 0 ", True),  # whitespace tolerated
        ],
    )
    def test_env_helper(self, monkeypatch, val, expected):
        from vllm_mlx.audio import stt as stt_mod

        if val is None:
            monkeypatch.delenv("RAPID_MLX_STT_VAD_PRETRIM", raising=False)
        else:
            monkeypatch.setenv("RAPID_MLX_STT_VAD_PRETRIM", val)
        assert stt_mod._vad_pretrim_disabled_by_env() is expected


# ---------------------------------------------------------------------------
# Sanity check on ``_shift_segment_time`` — pure function, easy to
# assert directly to catch off-by-one bugs before the integration path.
# ---------------------------------------------------------------------------


class TestShiftHelper:
    def test_shift_dict_segment(self):
        from vllm_mlx.audio.stt import _shift_segment_time

        seg = {"start": 1.0, "end": 2.0, "text": "hi"}
        _shift_segment_time(seg, 3.0)
        assert seg == {"start": 4.0, "end": 5.0, "text": "hi"}

    def test_shift_dict_segment_with_words(self):
        from vllm_mlx.audio.stt import _shift_segment_time

        seg = {
            "start": 1.0,
            "end": 2.0,
            "words": [
                {"start": 1.0, "end": 1.5, "word": "hi"},
                {"start": 1.5, "end": 2.0, "word": "there"},
            ],
        }
        _shift_segment_time(seg, 0.5)
        assert seg["start"] == 1.5
        assert seg["end"] == 2.5
        assert seg["words"][0]["start"] == 1.5
        assert seg["words"][1]["end"] == 2.5

    def test_shift_missing_keys_is_noop(self):
        from vllm_mlx.audio.stt import _shift_segment_time

        seg = {"text": "hi"}  # no start/end
        _shift_segment_time(seg, 5.0)
        assert seg == {"text": "hi"}

    def test_shift_object_segment(self):
        from vllm_mlx.audio.stt import _shift_segment_time

        class _Seg:
            start = 1.0
            end = 2.0

        s = _Seg()
        _shift_segment_time(s, 3.0)
        assert s.start == 4.0
        assert s.end == 5.0
