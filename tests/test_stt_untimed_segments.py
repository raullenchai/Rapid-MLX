# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for transcripts without segment timestamps."""

from types import SimpleNamespace

from rapid_mlx.routes.audio import _build_srt_body, _build_verbose_json_body


def result(segments, duration=8.309625, text="hello world"):
    return SimpleNamespace(
        segments=segments, duration=duration, text=text, language="en"
    )


def test_untimed_segment_verbose_json():
    body = _build_verbose_json_body(
        result([{"start": 0, "end": 0, "text": "hello world"}])
    )
    assert body["segments"] == [
        {"id": 0, "start": 0.0, "end": 8.309625, "text": "hello world"}
    ]


def test_untimed_segment_srt():
    body = _build_srt_body(result([{"start": 0, "end": 0, "text": "hello world"}]))
    assert "00:00:00,000 --> 00:00:08,310" in body


def test_untimed_segment_preserves_whisper_fields():
    segments = [{"text": "hello world", "avg_logprob": -0.5}]
    body = _build_verbose_json_body(result(segments, duration=8.309625))
    assert body["segments"] == [
        {
            "id": 0,
            "start": 0.0,
            "end": 8.309625,
            "text": "hello world",
            "avg_logprob": -0.5,
        }
    ]


def test_multiple_untimed_segments_use_full_text():
    segments = [{"text": "hello"}, {"text": "world"}]
    body = _build_srt_body(result(segments, text="full transcript"))
    assert body == "1\n00:00:00,000 --> 00:00:08,310\nfull transcript\n"


def test_timed_segments_preserve_whisper_fields():
    segments = [{"start": 0, "end": 2, "text": "hello", "avg_logprob": -0.5}]
    body = _build_verbose_json_body(result(segments))
    assert body["segments"] == [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "hello", "avg_logprob": -0.5}
    ]


def test_untimed_segment_without_duration():
    body = _build_verbose_json_body(result([{"text": "hello"}], duration=None))
    assert body["segments"] == [{"id": 0, "start": 0.0, "end": 0.0, "text": "hello"}]
