# SPDX-License-Identifier: Apache-2.0
"""Forced-alignment STT capability (Qwen3-ForcedAligner).

Hermetic: no real weights / network. A fake aligner model is injected at the
``STTEngine.model`` boundary; we assert the engine maps its per-character
``items`` onto the shared ``TranscriptionResult.segments`` shape and enforces
the align-vs-transcribe contract.
"""

from dataclasses import dataclass

import pytest

from vllm_mlx.audio.registry import resolve_audio_alias
from vllm_mlx.audio.stt import STTEngine, TranscriptionResult


@dataclass
class _FakeItem:
    text: str
    start_time: float
    end_time: float


class _FakeAlignResult:
    def __init__(self, text, items):
        self.text = text
        self.items = items


class _FakeAligner:
    """Stands in for mlx_audio's forced-aligner model."""

    def __init__(self):
        self.calls = []

    def generate(self, audio, text, language="English"):
        self.calls.append({"audio": audio, "text": text, "language": language})
        # emit one item per character with a simple monotonic 0.2s cadence
        items = [
            _FakeItem(ch, round(i * 0.2, 3), round((i + 1) * 0.2, 3))
            for i, ch in enumerate(text)
        ]
        return _FakeAlignResult(text=text, items=items)


def _aligner_engine():
    eng = STTEngine("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
    eng.model = _FakeAligner()
    eng._loaded = True  # skip real load_model
    return eng


def test_registry_exposes_forced_aligner():
    entry = resolve_audio_alias("qwen3-aligner")
    assert entry is not None
    assert entry.type == "stt"
    assert entry.family == "qwen3_aligner"
    assert entry.hf_id == "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    # long-form alias resolves to the same repo
    assert resolve_audio_alias("qwen3-forced-aligner").hf_id == entry.hf_id


def test_engine_detects_aligner_family():
    assert STTEngine("mlx-community/Qwen3-ForcedAligner-0.6B-8bit")._is_aligner
    assert not STTEngine("mlx-community/whisper-large-v3-mlx")._is_aligner


def test_align_returns_per_char_segments():
    eng = _aligner_engine()
    text = "临终前"
    res = eng.align("clip.wav", text, language="Chinese")
    assert isinstance(res, TranscriptionResult)
    assert res.text == text
    assert res.language == "Chinese"
    assert len(res.segments) == len(text)
    # segment shape matches the verbose_json/srt/vtt serializer contract
    first = res.segments[0]
    assert set(first) == {"text", "start", "end"}
    assert first == {"text": "临", "start": 0.0, "end": 0.2}
    # monotonic, and duration is the last char's end
    starts = [s["start"] for s in res.segments]
    assert starts == sorted(starts)
    assert res.duration == pytest.approx(0.6)
    # the known transcript was forwarded as an INPUT (no recognition)
    assert eng.model.calls[0]["text"] == text
    assert eng.model.calls[0]["language"] == "Chinese"


def test_transcribe_rejected_on_aligner():
    eng = _aligner_engine()
    with pytest.raises(ValueError, match="forced-alignment model"):
        eng.transcribe("clip.wav")


def test_align_requires_non_empty_text():
    eng = _aligner_engine()
    with pytest.raises(ValueError, match="non-empty"):
        eng.align("clip.wav", "   ")


def test_align_rejected_on_non_aligner():
    eng = STTEngine("mlx-community/whisper-large-v3-mlx")
    eng.model = object()
    eng._loaded = True
    with pytest.raises(ValueError, match="not one"):
        eng.align("clip.wav", "hello")
