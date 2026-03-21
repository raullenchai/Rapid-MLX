# Audio Support

vllm-mlx supports audio processing using [mlx-audio](https://github.com/Blaizzy/mlx-audio), providing:

- **STT (Speech-to-Text)**: Whisper, Parakeet
- **TTS (Text-to-Speech)**: Kokoro, Chatterbox, VibeVoice, VoxCPM
- **Audio Processing**: SAM-Audio (voice separation)

## Installation

```bash
# Core audio support
pip install mlx-audio>=0.2.9

# Required dependencies for TTS
pip install sounddevice soundfile scipy numba tiktoken misaki spacy num2words loguru phonemizer

# Download spacy English model
python -m spacy download en_core_web_sm

# For non-English TTS (Spanish, French, etc.), install espeak-ng:
# macOS
brew install espeak-ng

# Ubuntu/Debian
# sudo apt-get install espeak-ng
```

Or install all audio dependencies at once:

```bash
pip install rapid-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS, for non-English languages
```

## Quick Start

### Speech-to-Text (Transcription)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Transcribe audio file
with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
        language="en"  # optional
    )
print(transcript.text)
```

### Text-to-Speech (Generation)

```python
# Generate speech
audio = client.audio.speech.create(
    model="kokoro",
    input="Hello, how are you?",
    voice="af_heart",
    speed=1.0
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio.content)
```

### Voice Separation (SAM-Audio)

Isolate voice from background noise, music, or other sounds:

```python
from vllm_mlx.audio import AudioProcessor

# Load SAM-Audio model
processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
processor.load()

# Separate speech from audio
result = processor.separate("meeting_with_music.mp3", description="speech")

# Save isolated voice and background
processor.save(result.target, "voice_only.wav")
processor.save(result.residual, "background_only.wav")
```

**CLI Example:**
```bash
python examples/audio_separation_example.py meeting.mp3 --play
python examples/audio_separation_example.py song.mp3 --description music -o music.wav
```

### Drums Separation Demo

Isolate drums from a rock song using SAM-Audio:

| Audio | Description | Listen |
|-------|-------------|--------|
| Original | "Get Ready" by David Fesliyan (30s, royalty-free) | [🎵 rock_get_ready.mp3](../../examples/rock_get_ready.mp3) |
| Isolated Drums | Drums extracted by SAM-Audio | [🥁 drums_isolated.wav](../../examples/drums_isolated.wav) |
| Without Drums | Track with drums removed | [🎸 rock_no_drums.wav](../../examples/rock_no_drums.wav) |

```bash
# Isolate drums from rock song
python examples/audio_separation_example.py examples/rock_get_ready.mp3 \
  --description "drums" \
  --output drums_isolated.wav \
  --background rock_no_drums.wav
```

**Performance:** 30s audio processed in ~20 seconds on M4 Max.

## Supported Models

### STT Models (Speech-to-Text)

| Model | Alias | Languages | Speed | Quality |
|-------|-------|-----------|-------|---------|
| `mlx-community/whisper-large-v3-mlx` | `whisper-large-v3` | 99+ | Medium | Best |
| `mlx-community/whisper-large-v3-turbo` | `whisper-large-v3-turbo` | 99+ | Fast | Great |
| `mlx-community/whisper-medium-mlx` | `whisper-medium` | 99+ | Fast | Good |
| `mlx-community/whisper-small-mlx` | `whisper-small` | 99+ | Very Fast | OK |
| `mlx-community/parakeet-tdt-0.6b-v2` | `parakeet` | English | Fastest | Great |
| `mlx-community/parakeet-tdt-0.6b-v3` | `parakeet-v3` | English | Fastest | Best |

**Recommendation:**
- Multilingual: `whisper-large-v3`
- English only: `parakeet` (3x faster)

### TTS Models (Text-to-Speech)

#### Kokoro (Fast, Lightweight) - Recommended

| Model | Alias | Size | Languages |
|-------|-------|------|-----------|
| `mlx-community/Kokoro-82M-bf16` | `kokoro` | 82M | EN, ES, FR, JA, ZH, HI, IT, PT |
| `mlx-community/Kokoro-82M-4bit` | `kokoro-4bit` | 82M | EN, ES, FR, JA, ZH, HI, IT, PT |

**Voices (11):**
- Female American: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`
- Male American: `am_adam`, `am_michael`
- Female British: `bf_emma`, `bf_isabella`
- Male British: `bm_george`, `bm_lewis`

**Language Codes:**
| Code | Language | Code | Language |
|------|----------|------|----------|
| `a` / `en` | English (US) | `e` / `es` | Español |
| `b` / `en-gb` | English (UK) | `f` / `fr` | Français |
| `j` / `ja` | 日本語 | `z` / `zh` | 中文 |
| `i` / `it` | Italiano | `p` / `pt` | Português |
| `h` / `hi` | हिन्दी | | |

#### Chatterbox (Multilingual, Expressive)

| Model | Alias | Size | Languages |
|-------|-------|------|-----------|
| `mlx-community/chatterbox-turbo-fp16` | `chatterbox` | 134M | 15+ languages |
| `mlx-community/chatterbox-turbo-4bit` | `chatterbox-4bit` | 134M | 15+ languages |

**Supported Languages:** EN, ES, FR, DE, IT, PT, RU, JA, ZH, KO, AR, HI, NL, PL, TR

#### VibeVoice (Realtime)

| Model | Alias | Size | Use Case |
|-------|-------|------|----------|
| `mlx-community/VibeVoice-Realtime-0.5B-4bit` | `vibevoice` | 200M | Low latency, English |

#### VoxCPM (Chinese/English)

| Model | Alias | Size | Languages |
|-------|-------|------|-----------|
| `mlx-community/VoxCPM1.5` | `voxcpm` | 0.9B | ZH, EN |
| `mlx-community/VoxCPM1.5-4bit` | `voxcpm-4bit` | 200M | ZH, EN |

### Audio Processing Models

#### SAM-Audio (Voice Separation)

| Model | Size | Use Case |
|-------|------|----------|
| `mlx-community/sam-audio-large-fp16` | 3B | Best quality |
| `mlx-community/sam-audio-large` | 3B | Standard |
| `mlx-community/sam-audio-small-fp16` | 0.6B | Fast |
| `mlx-community/sam-audio-small` | 0.6B | Lightweight |

## API Reference

### POST /v1/audio/transcriptions

Transcribe audio to text (OpenAI Whisper API compatible).

**Parameters:**
- `file`: Audio file (mp3, wav, m4a, webm)
- `model`: Model name or alias
- `language`: Language code (optional, auto-detected)
- `response_format`: `json` or `text`

**Example:**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-large-v3
```

### POST /v1/audio/speech

Generate speech from text (OpenAI TTS API compatible).

**Parameters:**
- `model`: Model name or alias
- `input`: Text to synthesize
- `voice`: Voice ID
- `speed`: Speech speed (0.5 to 2.0)
- `response_format`: `wav`, `mp3`

**Example:**
```bash
curl http://localhost:8000/v1/audio/speech \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart"}' \
  -H "Content-Type: application/json" \
  --output speech.wav
```

### GET /v1/audio/voices

List available voices for a model.

**Example:**
```bash
curl http://localhost:8000/v1/audio/voices?model=kokoro
```

## CLI Examples

### Live Transcription / Closed Captions

Real-time speech-to-text transcription from your microphone:

```bash
# Closed captions with whisper-large-v3 (best quality)
python examples/closed_captions.py --language es --chunk 5

# Faster model for lower latency
python examples/closed_captions.py --language en --model whisper-turbo --chunk 3

# Basic mic transcription (record then transcribe)
python examples/mic_transcribe.py --language es

# Real-time chunked transcription
python examples/mic_realtime.py --language es --chunk 3

# Live transcription with voice activity detection
python examples/mic_live.py --language es
```

**Requirements:**
```bash
pip install sounddevice soundfile numpy
```

### Basic TTS

```bash
# Simple TTS example
python examples/tts_example.py "Hello, how are you?" --play

# With different voice
python examples/tts_example.py "Hello!" --voice am_michael --play

# Save to file
python examples/tts_example.py "Welcome to the demo" -o greeting.wav

# List available voices
python examples/tts_example.py --list-voices
```

### Multilingual TTS

```bash
# English (auto-selects best model)
python examples/tts_multilingual.py "Hello world" --play

# Spanish
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# French
python examples/tts_multilingual.py "Bonjour le monde" --lang fr --play

# Japanese
python examples/tts_multilingual.py "こんにちは" --lang ja --play

# Chinese
python examples/tts_multilingual.py "你好世界" --lang zh --play

# Use specific model
python examples/tts_multilingual.py "Hello" --model chatterbox --play

# List all models
python examples/tts_multilingual.py --list-models

# List all languages
python examples/tts_multilingual.py --list-languages
```

### Business Assistant Voice Examples

Pre-generated voice samples with **native voices** for common business use cases:

| Language | Voice | Message | Listen |
|----------|-------|---------|--------|
| 🇺🇸 English | af_heart | "Welcome to First National Bank. How may I assist you today?" | [▶️ assistant_bank_en.wav](../../examples/assistant_bank_en.wav) |
| 🇪🇸 Spanish | ef_dora | "Gracias por llamar a servicio al cliente. Un agente le atenderá pronto." | [▶️ assistant_service_es.wav](../../examples/assistant_service_es.wav) |
| 🇫🇷 French | ff_siwis | "Bienvenue. Votre appel est important pour nous." | [▶️ assistant_callcenter_fr.wav](../../examples/assistant_callcenter_fr.wav) |
| 🇨🇳 Chinese | zf_xiaobei | "欢迎致电技术支持中心。我们将竭诚为您服务。" | [▶️ assistant_support_zh.wav](../../examples/assistant_support_zh.wav) |

**Generate your own with native voices:**
```bash
# English - Bank assistant (native voice: af_heart)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Welcome to First National Bank. How may I assist you today?" \
  --voice af_heart --lang_code a --file_prefix assistant_bank_en

# Spanish - Customer service (native voice: ef_dora)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Gracias por llamar a servicio al cliente. Un agente le atendera pronto." \
  --voice ef_dora --lang_code e --file_prefix assistant_service_es

# French - Call center (native voice: ff_siwis)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "Bienvenue. Votre appel est important pour nous." \
  --voice ff_siwis --lang_code f --file_prefix assistant_callcenter_fr

# Chinese - Tech support (native voice: zf_xiaobei)
python -m mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 \
  --text "欢迎致电技术支持中心。我们将竭诚为您服务。" \
  --voice zf_xiaobei --lang_code z --file_prefix assistant_support_zh
```

### Native Voice Reference

| Language | Code | Voices |
|----------|------|--------|
| English (US) | `a` | af_heart, af_bella, af_nicole, am_adam, am_michael |
| English (UK) | `b` | bf_emma, bf_isabella, bm_george, bm_lewis |
| Spanish | `e` | ef_dora, em_alex, em_santa |
| French | `f` | ff_siwis |
| Chinese | `z` | zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zm_yunjian, zm_yunxi |
| Japanese | `j` | jf_alpha, jf_gongitsune, jm_kumo |
| Italian | `i` | if_sara, im_nicola |
| Portuguese | `p` | pf_dora, pm_alex |
| Hindi | `h` | hf_alpha, hf_beta, hm_omega |

## Python API

### Direct Usage (without server)

```python
from vllm_mlx.audio import STTEngine, TTSEngine, AudioProcessor

# Speech-to-Text
stt = STTEngine("mlx-community/whisper-large-v3-mlx")
stt.load()
result = stt.transcribe("audio.mp3")
print(result.text)

# Text-to-Speech
tts = TTSEngine("mlx-community/Kokoro-82M-bf16")
tts.load()
audio = tts.generate("Hello world", voice="af_heart")
tts.save(audio, "output.wav")

# Voice Separation
processor = AudioProcessor("mlx-community/sam-audio-large-fp16")
processor.load()
result = processor.separate("mixed_audio.mp3", description="speech")
processor.save(result.target, "voice_only.wav")
processor.save(result.residual, "background.wav")
```

### Convenience Functions

```python
from vllm_mlx.audio import transcribe_audio, generate_speech, separate_voice

# Quick transcription
result = transcribe_audio("audio.mp3")
print(result.text)

# Quick TTS
audio = generate_speech("Hello world", voice="af_heart")

# Quick voice separation
voice, background = separate_voice("mixed.mp3")
```

## Audio in Chat

Include audio in chat messages (transcribed automatically):

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize this audio"},
            {"type": "audio_url", "audio_url": {"url": "file://meeting.mp3"}}
        ]
    }]
)
```

## Benchmarks

Tested on Apple M2 Max (32GB).

### TTS Benchmarks (Kokoro-82M-bf16)

| Text Length | Audio Duration | Gen Time | RTF | Chars/sec |
|-------------|----------------|----------|-----|-----------|
| 25 chars | 1.95s | 0.43s | 4.6x | 58.5 |
| 88 chars | 6.00s | 0.32s | 18.6x | 272.4 |
| 117 chars | 7.92s | 0.27s | 29.0x | 427.4 |

**Summary:**
- Model load time: ~1.0s
- Average RTF: **17.4x** (17x faster than real-time)
- Average chars/sec: **252.8**

### STT Benchmarks

| Model | Load Time | Transcribe (6s audio) | RTF |
|-------|-----------|----------------------|-----|
| whisper-small | 0.25s | 0.20s | 30.2x |
| whisper-medium | 18.1s | 0.38s | 15.5x |
| whisper-large-v3 | ~30s | ~0.6s | ~10x |
| parakeet | ~0.5s | ~0.15s | ~40x |

**Notes:**
- RTF (Real-Time Factor) indicates how many times faster than real-time
- First load includes model download from HuggingFace
- Subsequent loads use cached models

### Recommendations by Use Case

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Fast English STT | `parakeet` | 40x RTF, low memory |
| Multilingual STT | `whisper-large-v3` | 99+ languages |
| Low-latency STT | `whisper-small` | 30x RTF, quick load |
| General TTS | `kokoro` | 17x RTF, good quality |
| Low memory TTS | `kokoro-4bit` | 4-bit quantized |

## Performance Tips

1. **Use Parakeet for English** - 40x faster than real-time
2. **Use 4-bit models** for lower memory usage
3. **Use SAM-Audio small** for faster voice separation
4. **Cache models** - engines are lazy-loaded and cached
5. **Pre-download models** to avoid first-run latency

## Troubleshooting

### mlx-audio not installed
```
pip install mlx-audio>=0.2.9
```

### Model download slow
Models are downloaded from HuggingFace on first use. Use `huggingface-cli download` to pre-download:
```bash
huggingface-cli download mlx-community/whisper-large-v3-mlx
huggingface-cli download mlx-community/Kokoro-82M-bf16
```

### Out of memory
Use smaller models or 4-bit quantized versions:
- `whisper-small-mlx` instead of `whisper-large-v3-mlx`
- `Kokoro-82M-4bit` instead of `Kokoro-82M-bf16`
- `sam-audio-small` instead of `sam-audio-large`

### Kokoro multilingual bug (mlx-audio 0.2.9)

If you get `ValueError: too many values to unpack` when using non-English languages (Spanish, Chinese, Japanese, etc.) with Kokoro, apply this fix:

```python
# Fix for mlx_audio/tts/models/kokoro/pipeline.py line 443
# Change:
#     ps, _ = self.g2p(chunk)
# To:
g2p_result = self.g2p(chunk)
ps = g2p_result[0] if isinstance(g2p_result, tuple) else g2p_result
```

**One-liner fix:**
```bash
python -c "
import os
path = os.path.join(os.path.dirname(__import__('mlx_audio').__file__), 'tts/models/kokoro/pipeline.py')
with open(path, 'r') as f: content = f.read()
old = '                    ps, _ = self.g2p(chunk)'
new = '''                    # Fix: handle both tuple (en) and string (zh/ja/es) returns from g2p
                    g2p_result = self.g2p(chunk)
                    ps = g2p_result[0] if isinstance(g2p_result, tuple) else g2p_result'''
if old in content:
    with open(path, 'w') as f: f.write(content.replace(old, new))
    print('Fix applied!')
"
```

This bug occurs because English g2p returns a tuple `(phonemes, tokens)` while other languages return just a string.
