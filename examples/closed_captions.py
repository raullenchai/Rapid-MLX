#!/usr/bin/env python3
"""
Closed Captions (CC) - Real-time Subtitles

Ultra low-latency transcription for live subtitles/closed captions.
Small chunks, fast processing, continuous output.

Usage:
    python examples/closed_captions.py
    python examples/closed_captions.py --language es

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd

MODEL_ALIASES = {
    "whisper-small": "mlx-community/whisper-small-mlx",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-turbo": "mlx-community/whisper-large-v3-turbo",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
}

SAMPLE_RATE = 16000


class ClosedCaptions:
    """Real-time closed captions."""

    def __init__(self, model_name: str, language: str = None, chunk_sec: float = 1.5):
        self.model_name = model_name
        self.language = language
        self.chunk_sec = chunk_sec
        self.chunk_samples = int(SAMPLE_RATE * chunk_sec)

        self.audio_queue = queue.Queue()
        self.running = False
        self.engine = None

        # For display
        self.current_line = ""
        self.lines = []

    def load_model(self):
        from rapid_mlx.audio.stt import STTEngine

        self.engine = STTEngine(self.model_name)
        self.engine.load()

    def audio_callback(self, indata, frames, time_info, status):
        if self.running:
            self.audio_queue.put(indata.copy().flatten())

    def transcribe(self, audio):
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, audio, SAMPLE_RATE)
            result = self.engine.transcribe(path, language=self.language)
            return result.text.strip()
        finally:
            os.unlink(path)

    def display_caption(self, text):
        """Display caption like subtitles."""
        if not text or text in [".", ""]:
            return

        # Move cursor up and clear, then print new caption
        print(f"\r\033[K  {text}", flush=True)

    def process_loop(self):
        """Process audio continuously."""
        buffer = np.array([], dtype=np.float32)
        silence_threshold = 0.008

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.05)
                buffer = np.concatenate([buffer, chunk])

                # Process when buffer is full
                if len(buffer) >= self.chunk_samples:
                    audio = buffer[: self.chunk_samples]
                    buffer = buffer[self.chunk_samples // 2 :]  # 50% overlap

                    # Skip if too quiet
                    level = np.sqrt(np.mean(audio**2))
                    if level < silence_threshold:
                        continue

                    text = self.transcribe(audio)
                    self.display_caption(text)

            except queue.Empty:
                continue

    def run(self):
        print()
        print("┌" + "─" * 58 + "┐")
        print("│" + "  🎬 CLOSED CAPTIONS - Real-time Subtitles".center(58) + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print(f"  Chunk: {self.chunk_sec}s | Model: {self.model_name.split('/')[-1]}")
        print()
        print("  Ctrl+C para salir")
        print()
        print("─" * 60)
        print()

        self.running = True

        # Start processor
        processor = threading.Thread(target=self.process_loop, daemon=True)
        processor.start()

        # Audio stream
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
            self.running = False
            print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Closed Captions - Real-time Subtitles"
    )
    parser.add_argument("--model", "-m", default="whisper-large-v3")
    parser.add_argument("--language", "-l", default=None, help="es, en, etc.")
    parser.add_argument(
        "--chunk", "-c", type=float, default=3.0, help="Chunk size (default: 3.0s)"
    )
    args = parser.parse_args()

    model = MODEL_ALIASES.get(args.model, args.model)

    print("\n  Cargando modelo...")
    cc = ClosedCaptions(model, args.language, args.chunk)
    cc.load_model()
    print("  ¡Listo!")

    cc.run()


if __name__ == "__main__":
    main()
