#!/usr/bin/env python3
"""
Live Speech Transcription - Real-time with Voice Activity Detection

Transcribes speech as you talk, detecting when you pause to process audio.
Much more natural than fixed-chunk transcription.

Usage:
    python examples/mic_live.py
    python examples/mic_live.py --model parakeet    # Faster for English

Requirements:
    pip install sounddevice soundfile numpy
"""

import argparse
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd

# Model aliases
MODEL_ALIASES = {
    "whisper-small": "mlx-community/whisper-small-mlx",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-turbo": "mlx-community/whisper-large-v3-turbo",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
}

SAMPLE_RATE = 16000


class LiveTranscriber:
    """Live transcription with voice activity detection."""

    def __init__(self, model_name: str, language: str = None):
        self.model_name = model_name
        self.language = language

        # VAD settings
        self.silence_threshold = 0.015  # Audio level threshold
        self.speech_pad_ms = 300  # Padding around speech
        self.min_speech_ms = 500  # Minimum speech duration
        self.silence_duration_ms = 700  # Silence to trigger transcription

        # State
        self.audio_buffer = deque(maxlen=SAMPLE_RATE * 30)  # 30s max
        self.is_speaking = False
        self.speech_start = 0
        self.last_speech_time = 0
        self.pending_audio = []

        # Threading
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False

        self.engine = None
        self.full_transcript = []

    def load_model(self):
        """Load STT model."""
        from vllm_mlx.audio.stt import STTEngine

        print(f"Loading: {self.model_name}")
        self.engine = STTEngine(self.model_name)
        self.engine.load()
        print("Ready!")

    def get_audio_level(self, audio):
        """Get RMS audio level."""
        return np.sqrt(np.mean(audio**2))

    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if self.running:
            self.audio_queue.put((time.time(), indata.copy().flatten()))

    def transcribe_audio(self, audio):
        """Transcribe audio array."""
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            sf.write(temp_path, audio, SAMPLE_RATE)
            result = self.engine.transcribe(temp_path, language=self.language)
            return result.text.strip()
        finally:
            os.unlink(temp_path)

    def process_audio_stream(self):
        """Process audio with VAD."""
        speech_buffer = []

        while self.running:
            try:
                timestamp, audio = self.audio_queue.get(timeout=0.05)
                level = self.get_audio_level(audio)

                is_speech = level > self.silence_threshold

                if is_speech:
                    if not self.is_speaking:
                        # Speech started
                        self.is_speaking = True
                        self.speech_start = timestamp
                        print("\r🎤 Listening...", end="", flush=True)

                    self.last_speech_time = timestamp
                    speech_buffer.extend(audio)

                elif self.is_speaking:
                    # Still collecting (might be brief pause)
                    speech_buffer.extend(audio)

                    silence_ms = (timestamp - self.last_speech_time) * 1000
                    speech_ms = (timestamp - self.speech_start) * 1000

                    # Check if silence long enough to trigger transcription
                    if (
                        silence_ms > self.silence_duration_ms
                        and speech_ms > self.min_speech_ms
                    ):
                        # Transcribe collected audio
                        audio_array = np.array(speech_buffer, dtype=np.float32)

                        print("\r⏳ Processing...", end="", flush=True)

                        text = self.transcribe_audio(audio_array)

                        if text and len(text) > 1:
                            self.full_transcript.append(text)
                            # Clear line and print result
                            print(f"\r\033[K💬 {text}")
                        else:
                            print("\r\033[K", end="")

                        # Reset
                        speech_buffer = []
                        self.is_speaking = False

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError: {e}")

        # Process remaining audio
        if speech_buffer and len(speech_buffer) > SAMPLE_RATE * 0.5:
            audio_array = np.array(speech_buffer, dtype=np.float32)
            text = self.transcribe_audio(audio_array)
            if text:
                self.full_transcript.append(text)
                print(f"\r\033[K💬 {text}")

    def run(self):
        """Start live transcription."""
        print()
        print("=" * 60)
        print(" 🎙️  LIVE TRANSCRIPTION")
        print("=" * 60)
        print()
        print(" Speak naturally - transcribes when you pause")
        print(" Press Ctrl+C to stop")
        print()
        print("-" * 60)
        print()

        self.running = True

        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        process_thread.start()

        # Start audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
        )

        try:
            with stream:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n")
            self.running = False
            process_thread.join(timeout=3)

        return self.full_transcript


def main():
    parser = argparse.ArgumentParser(description="Live Speech Transcription")
    parser.add_argument(
        "--model",
        "-m",
        default="whisper-small",
        help="Model (whisper-small, whisper-medium, parakeet)",
    )
    parser.add_argument("--language", "-l", help="Language code (en, es, etc.)")
    parser.add_argument(
        "--sensitivity",
        "-s",
        type=float,
        default=0.015,
        help="Mic sensitivity 0.01-0.05 (default: 0.015)",
    )
    args = parser.parse_args()

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║     🎙️  Live Speech Transcription - rapid-mlx          ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    model_name = MODEL_ALIASES.get(args.model, args.model)

    transcriber = LiveTranscriber(model_name=model_name, language=args.language)
    transcriber.silence_threshold = args.sensitivity

    transcriber.load_model()

    transcripts = transcriber.run()

    # Final summary
    print("-" * 60)
    print()
    print("📝 FULL TRANSCRIPT:")
    print()
    if transcripts:
        print(" " + " ".join(transcripts))
    else:
        print(" (No speech detected)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
