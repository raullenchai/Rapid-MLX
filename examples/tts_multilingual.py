#!/usr/bin/env python3
"""
Multilingual TTS Example - Text to Speech with multiple models and languages

Supported Models:
  - Kokoro: Fast, 82M params, 8 languages (en, es, fr, ja, zh, hi, it, pt)
  - Chatterbox: Expressive, voice cloning, 15+ languages
  - VibeVoice: Realtime, low latency, English
  - VoxCPM: High quality, Chinese/English
  - OuteTTS: Voice cloning, en/zh/ja/ko
  - Spark: Voice cloning, en/zh

Usage:
    python examples/tts_multilingual.py "Hello world"
    python examples/tts_multilingual.py "Hola mundo" --lang es
    python examples/tts_multilingual.py "Bonjour le monde" --lang fr --model kokoro
    python examples/tts_multilingual.py --list-models
    python examples/tts_multilingual.py --list-languages
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Model Registry
# =============================================================================

MODELS = {
    # Kokoro - Fast, multilingual
    "kokoro": {
        "path": "mlx-community/Kokoro-82M-bf16",
        "family": "kokoro",
        "languages": ["en", "es", "fr", "ja", "zh", "hi", "it", "pt"],
        "voices": [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ],
        "default_voice": "af_heart",
        "description": "Fast, lightweight (82M), 8 languages",
    },
    "kokoro-4bit": {
        "path": "mlx-community/Kokoro-82M-4bit",
        "family": "kokoro",
        "languages": ["en", "es", "fr", "ja", "zh", "hi", "it", "pt"],
        "voices": [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ],
        "default_voice": "af_heart",
        "description": "Quantized, lower memory",
    },
    # Chatterbox - Multilingual, expressive
    "chatterbox": {
        "path": "mlx-community/chatterbox-turbo-fp16",
        "family": "chatterbox",
        "languages": [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "zh",
            "ko",
            "ar",
            "hi",
            "nl",
            "pl",
            "tr",
        ],
        "voices": ["default"],
        "default_voice": "default",
        "description": "Expressive, 15+ languages, voice cloning capable",
    },
    "chatterbox-4bit": {
        "path": "mlx-community/chatterbox-turbo-4bit",
        "family": "chatterbox",
        "languages": [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "zh",
            "ko",
            "ar",
            "hi",
            "nl",
            "pl",
            "tr",
        ],
        "voices": ["default"],
        "default_voice": "default",
        "description": "Quantized chatterbox",
    },
    # VibeVoice - Realtime
    "vibevoice": {
        "path": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
        "family": "vibevoice",
        "languages": ["en"],
        "voices": ["default"],
        "default_voice": "default",
        "description": "Realtime, low latency, English only",
    },
    # VoxCPM - Chinese/English
    "voxcpm": {
        "path": "mlx-community/VoxCPM1.5",
        "family": "voxcpm",
        "languages": ["zh", "en"],
        "voices": ["default"],
        "default_voice": "default",
        "description": "High quality Chinese/English",
    },
}

# Language info
LANGUAGES = {
    "en": {"name": "English", "kokoro_code": "a"},
    "es": {"name": "Español", "kokoro_code": "e"},
    "fr": {"name": "Français", "kokoro_code": "f"},
    "de": {"name": "Deutsch", "kokoro_code": None},
    "it": {"name": "Italiano", "kokoro_code": "i"},
    "pt": {"name": "Português", "kokoro_code": "p"},
    "ja": {"name": "日本語", "kokoro_code": "j"},
    "zh": {"name": "中文", "kokoro_code": "z"},
    "ko": {"name": "한국어", "kokoro_code": None},
    "hi": {"name": "हिन्दी", "kokoro_code": "h"},
    "ru": {"name": "Русский", "kokoro_code": None},
    "ar": {"name": "العربية", "kokoro_code": None},
    "nl": {"name": "Nederlands", "kokoro_code": None},
    "pl": {"name": "Polski", "kokoro_code": None},
    "tr": {"name": "Türkçe", "kokoro_code": None},
}


def get_best_model_for_language(lang: str) -> str:
    """Get the best model for a given language."""
    lang = lang.lower()

    # Kokoro is best for these languages (fastest)
    kokoro_langs = ["en", "es", "fr", "ja", "zh", "hi", "it", "pt"]
    if lang in kokoro_langs:
        return "kokoro"

    # Chinese - VoxCPM is best
    if lang == "zh":
        return "voxcpm"

    # For other languages, use Chatterbox
    return "chatterbox"


def list_models():
    """Print available models."""
    print("\nAvailable TTS Models:")
    print("=" * 80)
    for name, info in MODELS.items():
        langs = ", ".join(info["languages"][:5])
        if len(info["languages"]) > 5:
            langs += f" (+{len(info['languages']) - 5} more)"
        print(f"\n  {name}")
        print(f"    Path: {info['path']}")
        print(f"    Languages: {langs}")
        print(f"    Voices: {len(info['voices'])}")
        print(f"    Description: {info['description']}")


def list_languages():
    """Print available languages and best models."""
    print("\nSupported Languages:")
    print("=" * 60)
    print(f"{'Code':<6} {'Language':<15} {'Best Model':<15} {'All Models'}")
    print("-" * 60)

    for code, info in sorted(LANGUAGES.items()):
        best = get_best_model_for_language(code)
        # Find all models supporting this language
        supporting = [name for name, m in MODELS.items() if code in m["languages"]]
        print(f"{code:<6} {info['name']:<15} {best:<15} {', '.join(supporting)}")


def generate_speech(
    text: str, model_name: str, lang: str, voice: str, speed: float, output: str
):
    """Generate speech using the specified model."""
    import wave

    import numpy as np
    from mlx_audio.tts.generate import load_model

    model_info = MODELS[model_name]
    model_path = model_info["path"]
    family = model_info["family"]

    print(f"\nModel: {model_name} ({model_path})")
    print(f"Family: {family}")
    print(f"Language: {LANGUAGES.get(lang, {}).get('name', lang)}")
    print(f"Voice: {voice}")
    print(f"Speed: {speed}x")
    print()

    # Load model
    print("Loading model...")
    start_load = time.time()
    model = load_model(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    # Prepare generation kwargs based on model family
    gen_kwargs = {
        "voice": voice,
        "speed": speed,
    }

    # Kokoro uses lang_code
    if family == "kokoro":
        lang_info = LANGUAGES.get(lang, {})
        kokoro_code = lang_info.get("kokoro_code", "a")
        if kokoro_code:
            gen_kwargs["lang_code"] = kokoro_code
        else:
            print(f"Warning: Language '{lang}' not supported by Kokoro, using English")
            gen_kwargs["lang_code"] = "a"

    # Generate
    print(f'\nGenerating: "{text}"')
    print()

    start_gen = time.time()
    audio_chunks = []
    sample_rate = 24000

    try:
        for result in model.generate(text, **gen_kwargs):
            audio_data = result.audio
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

            # Convert to numpy
            if hasattr(audio_data, "tolist"):
                audio_np = np.array(audio_data.tolist(), dtype=np.float32)
            else:
                audio_np = np.array(audio_data, dtype=np.float32)

            audio_chunks.append(audio_np)
    except Exception as e:
        print(f"Error during generation: {e}")
        print("\nTip: Some words may not be in the phoneme dictionary.")
        print("Try using common words in the selected language.")
        return None

    gen_time = time.time() - start_gen

    if not audio_chunks:
        print("Error: No audio generated")
        return None

    # Combine chunks
    full_audio = (
        np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    )
    duration = len(full_audio) / sample_rate

    print(f"Generated {duration:.2f}s audio in {gen_time:.2f}s")
    print(f"RTF (real-time factor): {duration / gen_time:.2f}x")

    # Save
    audio_int16 = (full_audio * 32767).astype(np.int16)
    with wave.open(output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"\nSaved to: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual TTS - Text to Speech with multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello world"                          # English with Kokoro
  %(prog)s "Hola mundo" --lang es                 # Spanish with Kokoro
  %(prog)s "Bonjour" --lang fr --model kokoro     # French with Kokoro
  %(prog)s "Hello" --model chatterbox             # English with Chatterbox
  %(prog)s --list-models                          # Show all models
  %(prog)s --list-languages                       # Show all languages
        """,
    )
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument(
        "--model",
        "-m",
        default="auto",
        help="Model: kokoro, chatterbox, vibevoice, voxcpm, or 'auto'",
    )
    parser.add_argument(
        "--lang", "-l", default="en", help="Language code: en, es, fr, ja, zh, etc."
    )
    parser.add_argument("--voice", "-v", default=None, help="Voice ID (model-specific)")
    parser.add_argument(
        "--speed", "-s", type=float, default=1.0, help="Speech speed 0.5-2.0"
    )
    parser.add_argument("--output", "-o", default="output.wav", help="Output file")
    parser.add_argument(
        "--play", "-p", action="store_true", help="Play audio after generation (macOS)"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--list-languages", action="store_true", help="List supported languages"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" Multilingual TTS - rapid-mlx")
    print("=" * 60)

    if args.list_models:
        list_models()
        return

    if args.list_languages:
        list_languages()
        return

    if not args.text:
        parser.print_help()
        return

    # Auto-select model based on language
    if args.model == "auto":
        args.model = get_best_model_for_language(args.lang)
        print(f"\nAuto-selected model: {args.model} (best for {args.lang})")

    # Validate model
    if args.model not in MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available: {', '.join(MODELS.keys())}")
        return

    model_info = MODELS[args.model]

    # Validate language
    if args.lang not in model_info["languages"]:
        print(
            f"Warning: Language '{args.lang}' not officially supported by {args.model}"
        )
        print(f"Supported: {', '.join(model_info['languages'])}")
        # Try anyway or switch model
        best = get_best_model_for_language(args.lang)
        if best != args.model:
            print(f"Suggestion: Use --model {best} for {args.lang}")

    # Default voice
    if args.voice is None:
        args.voice = model_info["default_voice"]

    # Generate
    output = generate_speech(
        text=args.text,
        model_name=args.model,
        lang=args.lang,
        voice=args.voice,
        speed=args.speed,
        output=args.output,
    )

    # Play
    if output and args.play:
        print("\nPlaying audio...")
        os.system(f"afplay {output}")


if __name__ == "__main__":
    main()
