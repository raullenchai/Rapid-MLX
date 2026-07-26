#!/usr/bin/env python3
"""
Real-Time Microphone Transcription with Whisper - rapid-mlx

Transcribes speech in real-time as you speak using your Mac's microphone.

Usage:
    python examples/mic_realtime.py                      # Default (3s chunks)
    python examples/mic_realtime.py --chunk 5            # 5 second chunks
    python examples/mic_realtime.py --model parakeet     # Faster model

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

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd

# Model aliases
MODEL_ALIASES = {
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1


class RealtimeTranscriber:
    """Real-time audio transcription using Whisper."""

    def __init__(
        self, model_name: str, chunk_duration: float = 3.0, language: str = None
    ):
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.language = language
        self.sample_rate = SAMPLE_RATE

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_recording = False

        # Transcription
        self.engine = None
        self.transcriptions = []

    def load_model(self):
        """Load the STT model."""
        from vllm_mlx.audio.stt import STTEngine

        print(f"Loading model: {self.model_name}")
        self.engine = STTEngine(self.model_name)
        self.engine.load()
        print("Model ready!")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def transcribe_chunk(self, audio_data):
        """Transcribe a chunk of audio."""
        import soundfile as sf

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio_data, self.sample_rate)

        try:
            result = self.engine.transcribe(temp_path, language=self.language)
            return result.text.strip()
        finally:
            os.unlink(temp_path)

    def process_audio(self):
        """Process audio chunks in real-time."""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        buffer = np.array([], dtype=np.float32)

        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio from queue
                data = self.audio_queue.get(timeout=0.1)
                buffer = np.concatenate([buffer, data.flatten()])

                # Process when we have enough audio
                if len(buffer) >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]

                    # Check if audio has content (not silence)
                    if np.abs(chunk).max() > 0.01:
                        text = self.transcribe_chunk(chunk)
                        if text and text not in ["", " ", "."]:
                            self.transcriptions.append(text)
                            # Print transcription in real-time
                            print(f"\r\033[K  >> {text}", flush=True)
                            print()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError: {e}")

        # Process remaining buffer
        if len(buffer) > self.sample_rate * 0.5:  # At least 0.5s
            if np.abs(buffer).max() > 0.01:
                text = self.transcribe_chunk(buffer)
                if text and text not in ["", " ", "."]:
                    self.transcriptions.append(text)
                    print(f"\r\033[K  >> {text}", flush=True)
                    print()

    def run(self):
        """Start real-time transcription."""
        print()
        print("=" * 60)
        print(" Real-Time Transcription")
        print(f" Chunk size: {self.chunk_duration}s")
        print("=" * 60)
        print()
        print("Speak now! Press Ctrl+C to stop.")
        print()
        print("-" * 60)

        self.is_recording = True

        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        )

        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()

        try:
            with stream:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n")
            print("-" * 60)
            self.is_recording = False

            # Wait for processing to finish
            print("Processing remaining audio...")
            process_thread.join(timeout=5)

        return self.transcriptions


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Microphone Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/mic_realtime.py                   # 3 second chunks
    python examples/mic_realtime.py --chunk 2         # 2 second chunks (faster)
    python examples/mic_realtime.py --model parakeet  # Fast English model
    python examples/mic_realtime.py --language en     # Force English
        """,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="whisper-small",
        help="Model to use (default: whisper-small)",
    )
    parser.add_argument(
        "--chunk",
        "-c",
        type=float,
        default=3.0,
        help="Chunk duration in seconds (default: 3.0)",
    )
    parser.add_argument("--language", "-l", help="Language code (e.g., en, es)")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print(" Real-Time Microphone Transcription - rapid-mlx")
    print("=" * 60)
    print()

    if args.list_models:
        print("Available models:")
        for alias, full_name in MODEL_ALIASES.items():
            rec = " (recommended for real-time)" if alias == "whisper-small" else ""
            print(f"  {alias:20} -> {full_name}{rec}")
        return

    # Resolve model alias
    model_name = MODEL_ALIASES.get(args.model, args.model)

    # Create transcriber
    transcriber = RealtimeTranscriber(
        model_name=model_name, chunk_duration=args.chunk, language=args.language
    )

    # Load model
    transcriber.load_model()

    # Run transcription
    transcriptions = transcriber.run()

    # Show summary
    print()
    print("=" * 60)
    print(" FULL TRANSCRIPT")
    print("=" * 60)
    print()
    full_text = " ".join(transcriptions)
    print(full_text if full_text else "(No speech detected)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
