# Audio Support

<!-- NOTE for doc sweeps: the Chinese/Japanese sample payloads in this guide
     are intentional — they are functional TTS inputs for zh/ja voices
     (qwen3-tts, f5-tts-zh, Kokoro zh/ja). English-only sweeps should skip
     them rather than flag them. -->

rapid-mlx supports audio processing using [mlx-audio](https://github.com/Blaizzy/mlx-audio), providing:

- **STT (Speech-to-Text)**: Whisper, Parakeet, SenseVoice
- **Forced alignment**: Qwen3-ForcedAligner (timings against a known transcript)
- **TTS (Text-to-Speech)**: Kokoro, Qwen3-TTS, Chatterbox, IndexTTS, VibeVoice, VoxCPM, F5-TTS, Dia
- **Zero-shot voice cloning**: IndexTTS, Qwen3-TTS Base, F5-TTS, Chatterbox
- **Audio Processing**: SAM-Audio (voice separation)

## Supported Aliases (R10-C1)

`rapid-mlx serve <alias>` recognizes the audio alias surface below and routes the request to the audio engines (skipping the text-LM loader). Both the short alias and the full HuggingFace id work — pasting a full HF id from `mlx-community/...` of an audio model takes the audio path automatically.

| Alias | Type | HuggingFace id |
| --- | --- | --- |
| `kokoro` (aka `kokoro-82m`, `kokoro-82m-bf16`) | TTS | `mlx-community/Kokoro-82M-bf16` |
| `kokoro-4bit` / `kokoro-82m-4bit` | TTS | `mlx-community/Kokoro-82M-4bit` |
| `kokoro-8bit` / `kokoro-82m-8bit` | TTS | `mlx-community/Kokoro-82M-8bit` |
| `chatterbox` | TTS | `mlx-community/chatterbox-turbo-fp16` |
| `chatterbox-4bit` | TTS | `mlx-community/chatterbox-turbo-4bit` |
| `vibevoice` / `vibevoice-realtime` | TTS | `mlx-community/VibeVoice-Realtime-0.5B-4bit` |
| `voxcpm` | TTS | `mlx-community/VoxCPM1.5` |
| `dia` | TTS | `mlx-community/Dia-1.6B-4bit` |
| `indextts` / `indextts-1.5` | TTS voice cloning | `mlx-community/IndexTTS-1.5` |
| `qwen3-tts` / `qwen3-tts-customvoice` | TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` |
| `qwen3-tts-6bit` | TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit` |
| `qwen3-tts-4bit` | TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` |
| `qwen3-tts-voicedesign` | TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` |
| `qwen3-tts-voicedesign-8bit` | TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` |
| `qwen3-tts-voicedesign-4bit` | TTS | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit` |
| `qwen3-tts-clone` | TTS voice cloning | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `f5-tts-zh` | TTS voice cloning | `lucasnewman/f5-tts-mlx` |
| `whisper` / `whisper-1` / `whisper-large-v3` | STT | `mlx-community/whisper-large-v3-mlx` |
| `whisper-large-v3-turbo` | STT | `mlx-community/whisper-large-v3-turbo` |
| `whisper-medium` | STT | `mlx-community/whisper-medium-mlx` |
| `whisper-small` | STT | `mlx-community/whisper-small-mlx` |
| `whisper-base` | STT | `mlx-community/whisper-base-mlx` |
| `whisper-tiny` | STT | `mlx-community/whisper-tiny-mlx` |
| `parakeet` / `parakeet-tdt-0.6b` / `parakeet-tdt-0.6b-v2` | STT | `mlx-community/parakeet-tdt-0.6b-v2` |
| `parakeet-v3` / `parakeet-tdt-0.6b-v3` | STT | `mlx-community/parakeet-tdt-0.6b-v3` |
| `sensevoice` / `sensevoice-small` | STT | `mlx-community/SenseVoiceSmall` |
| `qwen3-aligner` / `qwen3-forced-aligner` | Forced alignment | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` |

Run `rapid-mlx models` to see the full live list (the section header reads "Audio models" with `[audio:tts]` / `[audio:stt]` tags).

Audio engines load lazily on the first `/v1/audio/*` request — `rapid-mlx serve` returns as soon as the FastAPI app is bound, with no boot-time weight download.

### TTS quick start

```bash
# Boot the server with Kokoro
rapid-mlx serve kokoro
# Synthesize speech (OpenAI-compatible)
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello from rapid-mlx", "voice": "af_heart"}' \
  --output hello.wav
```

### STT quick start

```bash
# Boot with Whisper
rapid-mlx serve whisper-large-v3
# Transcribe (OpenAI-compatible)
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "model=whisper-large-v3" \
  -F "file=@speech.mp3"
```

### IndexTTS zero-shot voice cloning

IndexTTS has no predefined speakers. Send a base64-encoded reference clip
through the rapid-mlx `ref_audio` extension; unlike F5-TTS and Qwen3-TTS Base,
IndexTTS does not require a transcript of that clip.

```bash
rapid-mlx serve indextts

REF_AUDIO=$(base64 < reference.wav | tr -d '\n')
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"indextts\",\"input\":\"Hello in the cloned voice.\",\"ref_audio\":\"$REF_AUDIO\"}" \
  --output cloned.wav
```

### Qwen3-TTS: named speakers, described voices, or cloning

Qwen3-TTS ships as three checkpoints that behave differently enough to be worth
choosing between deliberately:

| Variant | Alias | How you control the voice |
| --- | --- | --- |
| CustomVoice | `qwen3-tts` | Pick a named speaker via `voice`; modulate emotion/style via `instructions` |
| VoiceDesign | `qwen3-tts-voicedesign` | Describe the whole voice in `instructions`; omit `voice` (only `describe` is accepted) |
| Base | `qwen3-tts-clone` | Clone from `ref_audio` + `ref_text`; no predefined speakers |

CustomVoice speakers are matched case-insensitively — Chinese: `Vivian`,
`Serena`, `Uncle_Fu`, `Dylan`, `Eric`; English: `Ryan`, `Aiden`; Japanese:
`Ono_Anna`; Korean: `Sohee`.

```bash
rapid-mlx serve qwen3-tts

curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","input":"欢迎收听。","voice":"Serena","instructions":"warm and unhurried"}' \
  --output narration.wav
```

VoiceDesign has **no named speakers at all** — the entire voice (timbre, gender,
age, accent, emotion, prosody) comes from `instructions`. Omit `voice`: the
endpoint validates it against the model's allowlist *before* generating, and for
VoiceDesign that allowlist is the single sentinel `describe`, so sending a speaker
name like `Serena` is rejected with `400 invalid_voice` rather than ignored.

```bash
rapid-mlx serve qwen3-tts-voicedesign

curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-voicedesign","input":"Chapter one.",
       "instructions":"a warm, low female narrator, calm and measured"}' \
  --output designed.wav
```

Base is the cloning variant. Unlike IndexTTS it needs the reference transcript
as well as the clip, which is what lets it hold one consistent branded narrator
across a whole channel:

```bash
rapid-mlx serve qwen3-tts-clone

REF_AUDIO=$(base64 < reference.wav | tr -d '\n')
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-tts-clone\",\"input\":\"新的一集开始了。\",\"ref_audio\":\"$REF_AUDIO\",\"ref_text\":\"参考音频的逐字文本\"}" \
  --output cloned.wav
```

### F5-TTS Chinese cloning

F5-TTS is pure MLX (no torch) and does EN+ZH zero-shot cloning from a 24 kHz
reference clip plus its transcript. It exists to fill the Chinese expressive gap:
Qwen3-TTS CustomVoice reads flat in Chinese, and Chatterbox cloning is
English-only. With no reference it falls back to a packaged default voice.

```bash
rapid-mlx serve f5-tts-zh

REF_AUDIO=$(base64 < reference_24k.wav | tr -d '\n')
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"f5-tts-zh\",\"input\":\"这是克隆出来的声音。\",\"ref_audio\":\"$REF_AUDIO\",\"ref_text\":\"参考音频的逐字文本\"}" \
  --output f5.wav
```

### SenseVoice — fast Asian-language ASR

SenseVoice Small (~234M, non-autoregressive CTC) is the strongest STT in the
registry for Chinese, Cantonese, Japanese and Korean.

The model itself also produces per-segment emotion and audio-event labels, but
`/v1/audio/transcriptions` does not currently surface them: the `json` envelope
returns `text`/`language`/`duration`, and `verbose_json` segments carry
`id`/`start`/`end`/`text` only. Expect a plain transcript from the HTTP API.

```bash
rapid-mlx serve sensevoice

curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=sensevoice
```

### Forced alignment (Qwen3-ForcedAligner)

Forced alignment is **not** recognition. You supply audio *and* the transcript
you already have, and get per-character start/end times back. Because it never
guesses at the words, it cannot mis-hear them — which is exactly what karaoke
captions and beat-synced editing need.

Pass a `text` field to the transcription endpoint to switch from recognition to
alignment:

```bash
rapid-mlx serve qwen3-aligner

curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=qwen3-aligner \
  -F text="the exact transcript of that audio"
```

From Python, the same path is `STTEngine.align(audio, text, language)`.

If `mlx-audio` is missing, the boot guard exits with rc=2 and the install hint:

```
error: model 'kokoro' is an audio alias and requires the optional `mlx-audio` dependency (shipped with the [audio] extra).
Install with: pip install 'rapid-mlx[audio]'
```

## Installation

```bash
# Core audio support — stay inside the supported range
# (mlx-audio 0.4.4 has a Kokoro istftnet regression; the pin excludes it)
pip install 'mlx-audio>=0.2.9,<0.4.4'

# Required dependencies for TTS
pip install sounddevice soundfile scipy misaki spacy num2words phonemizer-fork numba tiktoken loguru

# Download spacy English model
python -m spacy download en_core_web_sm

# For non-English TTS (Spanish, French, etc.), install espeak-ng:
# macOS
brew install espeak-ng

# Ubuntu/Debian
# sudo apt-get install espeak-ng
```

Or (recommended) install all audio dependencies at once — the `[audio]`
extra carries the supported `mlx-audio` version pin for you:

```bash
pip install 'rapid-mlx[audio]'
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

Isolate drums from a rock song using SAM-Audio. Point the example at any track
you have the rights to use — `--description` takes a free-form stem name
(`drums`, `bass`, `vocals`, `guitar`, …), and `--background` writes the
complementary mix:

```bash
python examples/audio_separation_example.py your_song.mp3 \
  --description "drums" \
  --output drums_isolated.wav \
  --background song_no_drums.wav
```

`--output` gets the isolated stem, `--background` gets the same track with that
stem removed. Add `--play` to audition the result without writing files.

> **Bring your own audio.** This repository ships no music. Stock "royalty-free"
> libraries usually still restrict redistribution and commercial use, so we do
> not bundle their files — use your own recordings, or a source you have cleared
> for your use. The [Free Music Archive](https://freemusicarchive.org/) hosts
> tracks under several licenses, so filter to CC0 / Public Domain and check the
> individual track's license before you rely on it; [MUSDB18](https://sigsep.github.io/datasets/musdb.html)
> is the standard source-separation benchmark but is research-licensed (not for
> commercial use). Whatever you pick, confirm its license covers what you plan
> to do.

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
| `mlx-community/SenseVoiceSmall` | `sensevoice` | zh, yue, ja, ko, en | Fastest | Great (Asian) |

**Recommendation:**
- Multilingual: `whisper-large-v3`
- English only: `parakeet` (3x faster)
- Chinese / Cantonese / Japanese / Korean: `sensevoice`
- You already have the transcript and need timings: `qwen3-aligner` (see Forced alignment below)

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

#### VoxCPM (English, experimental)

| Model | Alias | Size | Languages |
|-------|-------|------|-----------|
| `mlx-community/VoxCPM1.5` | `voxcpm` | 0.9B | EN (experimental) |

> **Do not use VoxCPM for Chinese.** Chinese output is currently broken in
> this MLX port — Chinese input produces Thai-script gibberish plus runaway
> generation. For Chinese TTS use `f5-tts-zh` (EN+ZH, cloneable) or
> `qwen3-tts` (named Chinese speakers) instead. English output works but is
> best-effort; report upstream if synthesis fails.

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
- `response_format`: `json` (default), `text`, `srt`, `vtt`, or `verbose_json`

**Example:**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=whisper-large-v3
```

#### Silence-hallucination guard (Whisper only)

Whisper is known to invent tokens like `"Thank you."` or
`"Thanks for watching!"` when handed pure-silence input, and to append
similar tails to legitimate clips that end in dead-air (issue #961).
Every Whisper request is therefore pre-filtered by Silero VAD
(bundled with the `[audio]` extra):

- **Pure-silence clips** — VAD detects no speech; the endpoint returns
  `{"text": "", "segments": [], "language": null}` and skips Whisper
  entirely (latency win: ~200 ms VAD vs multi-second decode).
- **Clips with leading / trailing silence** — VAD reports the speech
  span; Whisper only sees the trimmed waveform, and returned segment
  timestamps are shifted back to the original file's absolute time.
- **Sanity check** — if VAD reports no speech but the waveform still
  has non-trivial energy (RMS > ~-50 dBFS), the guard trusts the
  audio and falls back to Whisper. Real quiet / whispered speech is
  never silenced.

Operator controls:

| Knob | Type | Default | Effect |
| --- | --- | --- | --- |
| `RAPID_MLX_STT_VAD_PRETRIM` | env var | on | Set to `0`, `false`, `no`, or `off` to disable the guard run-wide and restore the pre-fix pass-through behaviour. |
| `STTEngine(..., enable_vad_pretrim=True)` | Python API | `True` | Same effect at the engine level; use when embedding `STTEngine` directly. |

The guard applies only to Whisper backends. Parakeet, Canary, and
other non-Whisper STT engines are never pre-filtered.

### POST /v1/audio/translations

Translate audio to English (OpenAI Whisper API compatible). Mirrors
`/v1/audio/transcriptions` — same `file`, `model`, and `response_format`
fields (all five formats above) — except the output is always English and
any `language` hint is ignored. Whisper models only: non-Whisper STT
aliases (e.g. `parakeet`, `sensevoice`) are rejected with
`400 invalid_model_for_translation` because they cannot force English
output.

**Example:**
```bash
curl http://localhost:8000/v1/audio/translations \
  -F file=@speech_fr.mp3 \
  -F model=whisper-large-v3
```

### POST /v1/audio/speech

Generate speech from text (OpenAI TTS API compatible).

**Parameters:**
- `model`: Model name or alias
- `input`: Text to synthesize
- `voice`: Voice ID
- `speed`: Speech speed (`0.25` to `4.0`)
- `response_format`: `wav`, `pcm`, `mp3`, `flac`, `ogg`, or `opus`
- `sample_rate`: Optional output rate (`8000` to `96000` Hz). Omit to keep
  the TTS model's native rate (commonly 24 kHz).
- `channels`: Optional output channel count, `1` or `2`. Omit to keep the
  model's native channel count (commonly mono).
- `voice_seed`: Optional unsigned 32-bit seed for
  `qwen3-tts-voicedesign`. Reuse the same `instructions` and seed to lock a
  designed voice across calls. The response echoes it in `X-Voice-Seed`.
- `exaggeration`: Optional emotion/intensity control for the Chatterbox
  family, `0.0` to `2.0` (`0.0` = flat/deadpan, ~`0.5` lively, up to ~`2.0`
  very theatrical). Every other TTS family ignores it, so it is safe to
  send unconditionally. Omit to keep the engine default.

MP3 accepts standard MPEG rates from 8–48 kHz; Opus accepts 8, 12, 16, 24,
or 48 kHz. WAV, PCM, FLAC, and Ogg/Vorbis accept any integer in the documented
8–96 kHz range. Invalid codec/rate combinations are rejected before synthesis.

**Example:**
```bash
curl http://localhost:8000/v1/audio/speech \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart",
       "sample_rate": 44100, "channels": 2}' \
  -H "Content-Type: application/json" \
  --output speech.wav
```

### POST /v1/audio/music

Generate music or sound effects from a text prompt, via the MLX-native
[Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-optimized)
engine vendored under `vllm_mlx/audio/sa3/`. Request-in / audio-bytes-out —
the same shape as `/v1/audio/speech`.

**Parameters (JSON body):**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | string | `"medium"` | DiT/decoder pairing: `medium` (higher quality, ~3.9 GB peak) or `sm-music` / `sm-sfx` (fast small, ~1.7 GB). Unknown values fall back to the engine defaults. |
| `input` | string | — (required) | Natural-language prompt. Non-blank, max 4096 chars (it becomes an argv element for the SA3 CLI). |
| `seconds` | number | `30.0` | Clip length. `0 < seconds <= 47` (the SA3 ceiling). NaN/inf rejected. |
| `steps` | integer | `8` | Pingpong sampling steps, `1..200`. |
| `negative_prompt` | string \| null | `null` | CFG negative branch (e.g. `"vocals, singing"`). Max 4096 chars. |
| `seed` | integer \| null | `null` | Fixed seed for reproducibility. |
| `response_format` | string | `"wav"` | Only `wav` is supported (SA3 renders WAV natively). |
| `sample_rate` | integer \| null | `null` | Output rate, `8000..96000` Hz. Omit to preserve SA3's native 44.1 kHz. |
| `channels` | `1` \| `2` \| null | `null` | Output channel count. Omit to preserve SA3's native stereo output. |

**Response:** `200 OK`, `Content-Type: audio/wav` — the raw WAV bytes.
Speech and music responses also include `X-Audio-Sample-Rate` and
`X-Audio-Channels`, which are especially useful for headerless PCM speech.

Errors: `400` for schema violations (blank or over-4096-char `input`,
`seconds > 47`, unsupported `response_format`); `500`
(`code="music_generation_failed"`) if the engine fails **or produces no
audio** — an SA3 run that exits cleanly without writing sample frames is
reported as a failure rather than returned as a silent clip; `503` if the
engine's runtime deps are unavailable.

> **On the status code:** a schema rejection is a FastAPI
> `RequestValidationError`, which the rapid-mlx server's global handler
> normalizes to **400** with a sanitized envelope (see
> `install_exception_handlers`). Stock FastAPI would emit 422 — you'll see
> 422 only if you mount the router on a bare app without those handlers,
> as the unit tests do.

Renders are serialized server-side: one SA3 subprocess at a time, so
concurrent callers queue instead of racing for unified memory. The render
runs off the event loop, so other requests (chat completions, health
probes) are unaffected while it works.

**Example:**
```bash
curl http://localhost:8000/v1/audio/music \
  -H "Content-Type: application/json" \
  -d '{
        "model": "medium",
        "input": "epic cinematic war drums, tense build-up",
        "seconds": 20,
        "steps": 8,
        "negative_prompt": "vocals, singing",
        "seed": 42,
        "sample_rate": 24000,
        "channels": 1
      }' \
  --output bgm.wav
```

Weights are fetched from HuggingFace on first use into the standard HF
cache (Stability Community License — free commercial use under $1M
revenue).

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
```bash
pip install 'rapid-mlx[audio]'
```
This installs `mlx-audio` with the supported version pin
(`mlx-audio>=0.2.9,<0.4.4` — 0.4.4 has a Kokoro istftnet regression that
breaks every Kokoro request).

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

### Kokoro multilingual bug (mlx-audio 0.2.9 only — historical)

mlx-audio 0.2.9 exactly had a Kokoro g2p bug: non-English languages
(Spanish, Chinese, Japanese, etc.) crashed with `ValueError: too many
values to unpack` because English g2p returned a tuple `(phonemes, tokens)`
while other languages returned just a string. Newer releases inside the
supported range (`mlx-audio>=0.2.9,<0.4.4`, e.g. 0.4.3 — what a fresh
`pip install 'rapid-mlx[audio]'` resolves to) no longer contain the buggy
unpacking. If you hit this error, upgrade `mlx-audio` within the pinned
range instead of hand-patching the package.
