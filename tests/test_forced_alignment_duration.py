import pytest

from tests.test_stt_language_hints import post, route, wav  # noqa: F401


def test_alignment_uses_clip_duration(route):  # noqa: F811
    client, _ = route
    response = post(
        client,
        "qwen3-aligner",
        wav(10),
        text="hello",
        language="zh",
        response_format="verbose_json",
    )
    assert response.status_code == 200, response.text
    assert response.json()["duration"] == pytest.approx(10.0)


def test_alignment_duration_falls_back_to_segments(monkeypatch):
    from types import SimpleNamespace

    from rapid_mlx.audio.stt import STTEngine

    engine = STTEngine("qwen3-aligner")
    engine._loaded = True
    engine.model = SimpleNamespace(
        generate=lambda **kwargs: SimpleNamespace(
            items=[SimpleNamespace(text="a", start_time=0, end_time=1)]
        )
    )
    monkeypatch.setattr(
        "rapid_mlx.audio.stt._audio_duration_seconds", lambda *args, **kwargs: None
    )
    result = engine.align("missing.wav", "a")
    assert result.duration == 1
    assert result.duration_from_segments is True
