#!/usr/bin/env python3
"""
Audio benchmarks for rapid-mlx.

Benchmarks STT (Speech-to-Text), TTS (Text-to-Speech), and audio processing.
"""

import os
import tempfile
import time

# Benchmark configurations
STT_MODELS = [
    ("mlx-community/whisper-tiny-mlx", "whisper-tiny"),
    ("mlx-community/whisper-small-mlx", "whisper-small"),
    ("mlx-community/whisper-medium-mlx", "whisper-medium"),
    ("mlx-community/whisper-large-v3-mlx", "whisper-large-v3"),
    ("mlx-community/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
    ("mlx-community/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2"),
    ("mlx-community/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3"),
]

TTS_MODELS = [
    ("mlx-community/Kokoro-82M-bf16", "kokoro"),
    ("mlx-community/Kokoro-82M-4bit", "kokoro-4bit"),
]

# Test inputs
TEST_TEXTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog. This is a test of text to speech synthesis.",
    "In a world where technology advances rapidly, artificial intelligence has become an integral part of our daily lives. From voice assistants to autonomous vehicles, AI systems are transforming how we work, communicate, and live.",
]


def generate_test_audio(duration_seconds: float = 5.0) -> str:
    """Generate a simple test audio file using TTS."""
    import wave

    import numpy as np

    # Create a simple sine wave tone
    sample_rate = 16000
    t = np.linspace(
        0, duration_seconds, int(sample_rate * duration_seconds), dtype=np.float32
    )
    # Mix of frequencies for more realistic audio
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 note
    audio += 0.1 * np.sin(2 * np.pi * 330 * t)  # E4 note
    audio = (audio * 32767).astype(np.int16)

    # Save to temp file
    fd, path = tempfile.mkstemp(suffix=".wav")
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())
    os.close(fd)
    return path


def benchmark_tts(
    model_name: str, alias: str, texts: list[str], voice: str = "af_heart"
):
    """Benchmark TTS model."""
    from vllm_mlx.audio.tts import TTSEngine

    print(f"\n{'=' * 60}")
    print(f"TTS Benchmark: {alias}")
    print(f"Model: {model_name}")
    print(f"Voice: {voice}")
    print(f"{'=' * 60}")

    # Load model
    print("Loading model...")
    load_start = time.time()
    engine = TTSEngine(model_name)
    engine.load()
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.2f}s")

    results = []
    for i, text in enumerate(texts):
        print(f"\nTest {i + 1}: {len(text)} characters")

        # Generate
        gen_start = time.time()
        output = engine.generate(text, voice=voice)
        gen_time = time.time() - gen_start

        # Calculate metrics
        chars_per_sec = len(text) / gen_time
        rtf = output.duration / gen_time  # Real-time factor

        print(f"  Generated: {output.duration:.2f}s audio in {gen_time:.2f}s")
        print(f"  Chars/sec: {chars_per_sec:.1f}")
        print(f"  RTF (real-time factor): {rtf:.2f}x")
        print(f"  Sample rate: {output.sample_rate} Hz")

        results.append(
            {
                "chars": len(text),
                "audio_duration": output.duration,
                "gen_time": gen_time,
                "chars_per_sec": chars_per_sec,
                "rtf": rtf,
            }
        )

    # Summary
    avg_chars_per_sec = sum(r["chars_per_sec"] for r in results) / len(results)
    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    print("\n--- Summary ---")
    print(f"Average chars/sec: {avg_chars_per_sec:.1f}")
    print(f"Average RTF: {avg_rtf:.2f}x")

    return {
        "model": alias,
        "load_time": load_time,
        "avg_chars_per_sec": avg_chars_per_sec,
        "avg_rtf": avg_rtf,
    }


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    import contextlib
    import wave

    # Try wav first
    if audio_path.endswith(".wav"):
        with contextlib.closing(wave.open(audio_path, "r")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)

    # For mp3 and other formats, use ffprobe if available
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def benchmark_stt(model_name: str, alias: str, audio_path: str):
    """Benchmark STT model."""
    from vllm_mlx.audio.stt import STTEngine

    print(f"\n{'=' * 60}")
    print(f"STT Benchmark: {alias}")
    print(f"Model: {model_name}")
    print(f"Audio: {audio_path}")
    print(f"{'=' * 60}")

    # Get audio duration first
    audio_duration = get_audio_duration(audio_path)

    # Load model
    print("Loading model...")
    load_start = time.time()
    engine = STTEngine(model_name)
    engine.load()
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.2f}s")

    # Transcribe
    print("\nTranscribing...")
    trans_start = time.time()
    result = engine.transcribe(audio_path)
    trans_time = time.time() - trans_start

    # Use detected duration or fallback to calculated
    duration = result.duration if result.duration else audio_duration

    # Calculate metrics
    rtf = duration / trans_time if duration and trans_time > 0 else 0

    print("\nResult:")
    print(
        f"  Text: {result.text[:100]}..."
        if len(result.text) > 100
        else f"  Text: {result.text}"
    )
    print(f"  Language: {result.language}")
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  Transcription time: {trans_time:.2f}s")
    print(f"  RTF (real-time factor): {rtf:.2f}x")

    return {
        "model": alias,
        "load_time": load_time,
        "audio_duration": duration,
        "trans_time": trans_time,
        "rtf": rtf,
    }


def check_whisper_backend():
    """
    Check whether the Whisper backend can be imported.

    Returns:
        (available: bool, reason: str)
    """
    try:
        import mlx_audio.stt.models.whisper  # noqa: F401

        return True, ""
    except Exception as e:
        return False, str(e)


def run_tts_benchmarks():
    """Run all TTS benchmarks."""
    print("\n" + "=" * 70)
    print(" TTS BENCHMARKS (Text-to-Speech)")
    print("=" * 70)

    results = []
    for model_name, alias in TTS_MODELS:
        try:
            result = benchmark_tts(model_name, alias, TEST_TEXTS)
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {alias}: {e}")
            continue

    # Print summary table
    if results:
        print("\n" + "=" * 70)
        print(" TTS BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Model':<25} {'Load (s)':<12} {'Chars/s':<12} {'RTF':<10}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['model']:<25} {r['load_time']:<12.2f} {r['avg_chars_per_sec']:<12.1f} {r['avg_rtf']:<10.2f}x"
            )

    return results


def run_stt_benchmarks(audio_path: str):
    """Run all STT benchmarks."""
    print("\n" + "=" * 70)
    print(" STT BENCHMARKS (Speech-to-Text)")
    print("=" * 70)

    whisper_available, whisper_error = check_whisper_backend()
    if not whisper_available:
        print("Warning: Whisper backend unavailable; skipping Whisper models.")
        print(f"Reason: {whisper_error}")

    results = []
    for model_name, alias in STT_MODELS:
        if alias.startswith("whisper") and not whisper_available:
            print(f"\nSkipping {alias}: Whisper backend unavailable")
            continue
        try:
            result = benchmark_stt(model_name, alias, audio_path)
            results.append(result)
        except Exception as e:
            print(f"\nError benchmarking {alias}: {e}")
            continue

    # Print summary table
    if results:
        print("\n" + "=" * 70)
        print(" STT BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Model':<25} {'Load (s)':<12} {'Trans (s)':<12} {'RTF':<10}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['model']:<25} {r['load_time']:<12.2f} {r['trans_time']:<12.2f} {r['rtf']:<10.2f}x"
            )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audio benchmarks for rapid-mlx")
    parser.add_argument("--tts", action="store_true", help="Run TTS benchmarks")
    parser.add_argument("--stt", action="store_true", help="Run STT benchmarks")
    parser.add_argument("--audio", type=str, help="Audio file for STT benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if args.all or (not args.tts and not args.stt):
        args.tts = True
        args.stt = True

    # Generate test audio if needed
    audio_path = args.audio
    if args.stt and not audio_path:
        print("Generating test audio file...")
        audio_path = generate_test_audio(10.0)
        print(f"Test audio: {audio_path}")

    try:
        if args.tts:
            run_tts_benchmarks()

        if args.stt:
            run_stt_benchmarks(audio_path)
    finally:
        # Cleanup generated audio
        if not args.audio and audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


if __name__ == "__main__":
    main()
