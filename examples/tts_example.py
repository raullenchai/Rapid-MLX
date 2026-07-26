#!/usr/bin/env python3
"""
TTS Example - Text to Speech with rapid-mlx

Usage:
    python examples/tts_example.py "Hello, how are you?"
    python examples/tts_example.py "Welcome!" --voice am_michael
    python examples/tts_example.py "Hola, como estas?" --lang es
    python examples/tts_example.py --list-voices
    python examples/tts_example.py --list-languages
"""

import argparse
import os
import sys

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Language codes for Kokoro
LANGUAGES = {
    "a": "American English",
    "b": "British English",
    "e": "Español",
    "f": "Français",
    "i": "Italiano",
    "p": "Português (Brasil)",
    "j": "日本語 (Japanese)",
    "z": "中文 (Mandarin)",
    "h": "हिन्दी (Hindi)",
}

LANG_ALIASES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "spanish": "e",
    "fr": "f",
    "french": "f",
    "it": "i",
    "italian": "i",
    "pt": "p",
    "pt-br": "p",
    "portuguese": "p",
    "ja": "j",
    "japanese": "j",
    "zh": "z",
    "chinese": "z",
    "hi": "h",
    "hindi": "h",
}


def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech Example")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument(
        "--voice", "-v", default="af_heart", help="Voice ID (default: af_heart)"
    )
    parser.add_argument(
        "--lang",
        "-l",
        default="a",
        help="Language code: a=English, e/es=Spanish, f=French, etc.",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="Speech speed 0.5-2.0 (default: 1.0)",
    )
    parser.add_argument(
        "--output", "-o", default="output.wav", help="Output file (default: output.wav)"
    )
    parser.add_argument(
        "--model", "-m", default="mlx-community/Kokoro-82M-bf16", help="TTS model"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List available voices"
    )
    parser.add_argument(
        "--list-languages", action="store_true", help="List available languages"
    )
    parser.add_argument(
        "--play", "-p", action="store_true", help="Play audio after generation (macOS)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" TTS Example - rapid-mlx")
    print("=" * 60)
    print()

    # List languages
    if args.list_languages:
        print("Available languages:")
        for code, name in LANGUAGES.items():
            print(f"  {code}: {name}")
        print()
        print("Aliases:")
        for alias, code in sorted(LANG_ALIASES.items()):
            print(f"  --lang {alias} -> {code}")
        return

    from vllm_mlx.audio.tts import TTSEngine

    # Resolve language alias
    lang_code = args.lang.lower()
    lang_code = LANG_ALIASES.get(lang_code, lang_code)
    lang_name = LANGUAGES.get(lang_code, lang_code)

    # Initialize engine
    print(f"Model: {args.model}")
    engine = TTSEngine(args.model)
    engine.load()
    print(f"Model family: {engine._model_family}")
    print(f"Language: {lang_name} ({lang_code})")
    print()

    # List voices
    voices = engine.get_voices()
    print(f"Available voices ({len(voices)}):")
    for voice in voices:
        marker = " <--" if voice == args.voice else ""
        print(f"  - {voice}{marker}")
    print()

    if args.list_voices:
        return

    if not args.text:
        print("Error: No text provided. Use --help for usage.")
        return

    # Generate speech
    print(f'Text: "{args.text}"')
    print(f"Voice: {args.voice}")
    print(f"Language: {lang_name}")
    print(f"Speed: {args.speed}x")
    print()
    print("Generating...")

    try:
        output = engine.generate(
            args.text, voice=args.voice, speed=args.speed, lang_code=lang_code
        )
    except Exception as e:
        print(f"Error: {e}")
        print(
            "\nNote: Technical terms or made-up words may fail. Try common words in the selected language."
        )
        return

    print()
    print("Generated audio:")
    print(f"  Duration: {output.duration:.2f} seconds")
    print(f"  Sample rate: {output.sample_rate} Hz")
    print(f"  Samples: {len(output.audio):,}")
    print()

    # Save
    engine.save(output, args.output)
    print(f"Saved to: {args.output}")

    # Play on macOS
    if args.play:
        print("\nPlaying audio...")
        os.system(f"afplay {args.output}")


if __name__ == "__main__":
    main()
