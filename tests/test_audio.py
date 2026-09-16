# SPDX-License-Identifier: Apache-2.0
"""
Tests for audio support (STT, TTS, audio processing).

Note: Some tests require mlx-audio to be installed.
"""

import numpy as np
import pytest


class TestSTTEngine:
    """Tests for Speech-to-Text engine."""

    def test_init_whisper(self):
        """Test STT engine initialization with Whisper."""
        from rapid_mlx.audio.stt import STTEngine

        engine = STTEngine("mlx-community/whisper-large-v3-mlx")
        assert engine.model_name == "mlx-community/whisper-large-v3-mlx"
        assert engine._is_parakeet is False
        assert engine._loaded is False

    def test_init_parakeet(self):
        """Test STT engine initialization with Parakeet."""
        from rapid_mlx.audio.stt import STTEngine

        engine = STTEngine("mlx-community/parakeet-tdt-0.6b-v2")
        assert engine._is_parakeet is True

    def test_default_models(self):
        """Test default model constants."""
        from rapid_mlx.audio.stt import DEFAULT_PARAKEET_MODEL, DEFAULT_WHISPER_MODEL

        assert "whisper" in DEFAULT_WHISPER_MODEL.lower()
        assert "parakeet" in DEFAULT_PARAKEET_MODEL.lower()

    def test_transcription_result(self):
        """Test TranscriptionResult dataclass."""
        from rapid_mlx.audio.stt import TranscriptionResult

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.5,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.5


class TestTTSEngine:
    """Tests for Text-to-Speech engine."""

    def test_init_kokoro(self):
        """Test TTS engine initialization with Kokoro."""
        from rapid_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/Kokoro-82M-bf16")
        assert engine.model_name == "mlx-community/Kokoro-82M-bf16"
        assert engine._model_family == "kokoro"
        assert engine._loaded is False

    def test_init_chatterbox(self):
        """Test TTS engine initialization with Chatterbox."""
        from rapid_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/chatterbox-turbo-fp16")
        assert engine._model_family == "chatterbox"

    def test_init_vibevoice(self):
        """Test TTS engine initialization with VibeVoice."""
        from rapid_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VibeVoice-Realtime-0.5B-4bit")
        assert engine._model_family == "vibevoice"

    def test_init_voxcpm(self):
        """Test TTS engine initialization with VoxCPM."""
        from rapid_mlx.audio.tts import TTSEngine

        engine = TTSEngine("mlx-community/VoxCPM1.5")
        assert engine._model_family == "voxcpm"

    def test_available_voices(self):
        """Test voice lists."""
        from rapid_mlx.audio.tts import CHATTERBOX_VOICES, KOKORO_VOICES

        assert "af_heart" in KOKORO_VOICES
        assert len(KOKORO_VOICES) > 5
        assert "default" in CHATTERBOX_VOICES

    def test_get_voices(self):
        """Test get_voices method."""
        from rapid_mlx.audio.tts import TTSEngine

        kokoro = TTSEngine("mlx-community/Kokoro-82M-bf16")
        voices = kokoro.get_voices()
        assert "af_heart" in voices

    def test_audio_output(self):
        """Test AudioOutput dataclass."""
        from rapid_mlx.audio.tts import AudioOutput

        audio = np.zeros(24000, dtype=np.float32)
        output = AudioOutput(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
        )
        assert output.sample_rate == 24000
        assert output.duration == 1.0
        assert len(output.audio) == 24000


class TestAudioProcessor:
    """Tests for audio processor (SAM-Audio)."""

    def test_init(self):
        """Test audio processor initialization."""
        from rapid_mlx.audio.processor import AudioProcessor

        processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
        assert processor.model_name == "mlx-community/sam-audio-large-fp16"
        assert processor._loaded is False

    def test_default_model(self):
        """Test default SAM-Audio model."""
        from rapid_mlx.audio.processor import DEFAULT_SAM_MODEL

        assert "sam-audio" in DEFAULT_SAM_MODEL.lower()

    def test_separation_result(self):
        """Test SeparationResult dataclass."""
        from rapid_mlx.audio.processor import SeparationResult

        target = np.zeros(44100, dtype=np.float32)
        residual = np.zeros(44100, dtype=np.float32)

        result = SeparationResult(
            target=target,
            residual=residual,
            sample_rate=44100,
            peak_memory=1.5,
        )
        assert result.sample_rate == 44100
        assert result.peak_memory == 1.5
        assert len(result.target) == 44100


class TestAPIModels:
    """Tests for audio API models."""

    def test_audio_url(self):
        """Test AudioUrl model."""
        from rapid_mlx.api.models import AudioUrl

        url = AudioUrl(url="file://test.mp3")
        assert url.url == "file://test.mp3"

    def test_content_part_audio(self):
        """Test ContentPart with audio."""
        from rapid_mlx.api.models import ContentPart

        part = ContentPart(type="audio_url", audio_url={"url": "test.mp3"})
        assert part.type == "audio_url"
        # Pydantic converts dict to AudioUrl model
        assert part.audio_url.url == "test.mp3"

    def test_transcription_request(self):
        """Test AudioTranscriptionRequest model."""
        from rapid_mlx.api.models import AudioTranscriptionRequest

        req = AudioTranscriptionRequest(
            model="whisper-large-v3",
            language="en",
        )
        assert req.model == "whisper-large-v3"
        assert req.language == "en"
        assert req.response_format == "json"

    def test_speech_request(self):
        """Test AudioSpeechRequest model."""
        from rapid_mlx.api.models import AudioSpeechRequest

        req = AudioSpeechRequest(
            model="kokoro",
            input="Hello world",
            voice="af_heart",
            speed=1.2,
        )
        assert req.model == "kokoro"
        assert req.input == "Hello world"
        assert req.voice == "af_heart"
        assert req.speed == 1.2

    def test_transcription_response(self):
        """Test AudioTranscriptionResponse model."""
        from rapid_mlx.api.models import AudioTranscriptionResponse

        resp = AudioTranscriptionResponse(
            text="Hello world",
            language="en",
            duration=2.5,
        )
        assert resp.text == "Hello world"


class TestAudioImports:
    """Test that all audio modules can be imported."""

    def test_import_audio_module(self):
        """Test importing main audio module."""
        from rapid_mlx.audio import (
            AudioProcessor,
            STTEngine,
            TTSEngine,
        )

        assert STTEngine is not None
        assert TTSEngine is not None
        assert AudioProcessor is not None

    def test_import_api_models(self):
        """Test importing audio API models."""
        from rapid_mlx.api import (
            AudioTranscriptionRequest,
            AudioUrl,
        )

        assert AudioUrl is not None
        assert AudioTranscriptionRequest is not None


# Integration tests (require mlx-audio installed)
@pytest.mark.skip(reason="Requires mlx-audio and models downloaded")
class TestAudioIntegration:
    """Integration tests for audio (require models)."""

    def test_whisper_transcription(self):
        """Test Whisper transcription."""
        from rapid_mlx.audio import transcribe_audio

        result = transcribe_audio(
            "test_audio.wav",
            model_name="mlx-community/whisper-small-mlx",
        )
        assert result.text is not None

    def test_kokoro_tts(self):
        """Test Kokoro TTS generation."""
        from rapid_mlx.audio import generate_speech

        audio = generate_speech(
            "Hello world",
            model_name="mlx-community/Kokoro-82M-bf16",
            voice="af_heart",
        )
        assert audio.audio is not None
        assert audio.sample_rate > 0

    def test_sam_audio_separation(self):
        """Test SAM-Audio voice separation."""
        from rapid_mlx.audio import separate_voice

        target, residual = separate_voice(
            "test_audio.wav",
            model_name="mlx-community/sam-audio-small",
        )
        assert target is not None
        assert residual is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
