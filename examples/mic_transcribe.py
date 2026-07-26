#!/usr/bin/env python3
"""
Live Microphone Transcription with Whisper - rapid-mlx

Records audio from your Mac's microphone and transcribes it using Whisper.

Usage:
    python examples/mic_transcribe.py                    # Record until Enter
    python examples/mic_transcribe.py --duration 5       # Record for 5 seconds
    python examples/mic_transcribe.py --model whisper-small  # Use smaller model
    python examples/mic_transcribe.py --continuous       # Continuous mode

Requirements:
    pip install sounddevice soundfile
"""

import argparse
import os
import sys
import tempfile
import threading

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model aliases
MODEL_ALIASES = {
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "whisper-large": "mlx-community/whisper-large-v3-mlx",
    "whisper-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}


def record_audio(duration=None, sample_rate=16000):
    """
    Record audio from microphone.

    Args:
        duration: Recording duration in seconds. If None, records until Enter.
        sample_rate: Audio sample rate (16000 Hz for Whisper)

    Returns:
        numpy array of audio data
    """
    import numpy as np
    import sounddevice as sd

    print()
    if duration:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()
    else:
        print("Recording... Press ENTER to stop.")
        print()

        # Record in chunks until Enter is pressed
        chunks = []
        stop_recording = threading.Event()

        def wait_for_enter():
            input()
            stop_recording.set()

        # Start thread waiting for Enter
        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()

        # Record in 0.5 second chunks
        chunk_duration = 0.5
        chunk_samples = int(chunk_duration * sample_rate)

        while not stop_recording.is_set():
            chunk = sd.rec(
                chunk_samples, samplerate=sample_rate, channels=1, dtype=np.float32
            )
            sd.wait()
            chunks.append(chunk)
            # Show recording indicator
            print(
                f"\r  Recording: {len(chunks) * chunk_duration:.1f}s",
                end="",
                flush=True,
            )

        print()  # New line after recording indicator
        audio = np.concatenate(chunks, axis=0) if chunks else np.array([])

    return audio.flatten(), sample_rate


def save_audio(audio, sample_rate, path):
    """Save audio to WAV file."""
    import soundfile as sf

    sf.write(path, audio, sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Live Microphone Transcription with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/mic_transcribe.py                     # Press Enter to stop
    python examples/mic_transcribe.py --duration 10       # Record 10 seconds
    python examples/mic_transcribe.py --model parakeet    # Fast English model
    python examples/mic_transcribe.py --continuous        # Keep transcribing
    python examples/mic_transcribe.py --save recording.wav  # Save audio
        """,
    )
    parser.add_argument(
        "--duration", "-d", type=float, help="Recording duration in seconds"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="whisper-small",
        help="Model: whisper-small, whisper-medium, whisper-large-v3, parakeet",
    )
    parser.add_argument(
        "--language", "-l", help="Language code (e.g., en, es). Auto-detect if not set"
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Continuous mode: keep recording and transcribing",
    )
    parser.add_argument("--save", "-s", help="Save recorded audio to this file")
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio input devices"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Microphone Transcription - rapid-mlx")
    print("=" * 60)
    print()

    # List devices
    if args.list_devices:
        import sounddevice as sd

        print("Audio Input Devices:")
        print(sd.query_devices())
        return

    # List models
    if args.list_models:
        print("Available models:")
        for alias, full_name in MODEL_ALIASES.items():
            print(f"  {alias:20} -> {full_name}")
        return

    # Resolve model alias
    model_name = MODEL_ALIASES.get(args.model, args.model)

    print(f"Model: {model_name}")
    print()

    # Load model first (so user doesn't wait after recording)
    print("Loading model...")
    from vllm_mlx.audio.stt import STTEngine

    engine = STTEngine(model_name)
    engine.load()
    print("Model ready!")
    print()

    try:
        while True:
            # Record audio
            audio, sample_rate = record_audio(duration=args.duration)

            if len(audio) == 0:
                print("No audio recorded.")
                if not args.continuous:
                    break
                continue

            duration = len(audio) / sample_rate
            print(f"Recorded {duration:.1f} seconds of audio")

            # Save to temp file for transcription
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            save_audio(audio, sample_rate, temp_path)

            # Also save permanently if requested
            if args.save:
                save_audio(audio, sample_rate, args.save)
                print(f"Audio saved to: {args.save}")

            # Transcribe
            print()
            print("Transcribing...")
            result = engine.transcribe(temp_path, language=args.language)

            # Clean up temp file
            os.unlink(temp_path)

            # Show result
            print()
            print("-" * 60)
            print("TRANSCRIPTION:")
            print("-" * 60)
            print()
            print(f"  {result.text}")
            print()
            print("-" * 60)

            if result.language:
                print(f"Detected language: {result.language}")

            if not args.continuous:
                break

            print()
            print("=" * 60)
            print(" Ready for next recording (Ctrl+C to exit)")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nExiting...")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
