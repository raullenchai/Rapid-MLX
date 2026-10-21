# Audio Benchmarks

## Speech-to-Text (STT) Benchmarks

### Running STT Benchmarks

```bash
# Run with default test audio
python examples/benchmark_audio.py --stt

# Run with your own audio file
python examples/benchmark_audio.py --stt --audio path/to/audio.wav
```

### Results (M4 Max, 128GB)

**Test audio:** 46.7 seconds of synthesized speech

| Model | Parameters | Load Time | Transcribe Time | RTF* |
|-------|------------|-----------|-----------------|------|
| whisper-tiny | 39M | 0.34s | 0.24s | **197x** |
| whisper-small | 244M | 0.18s | 0.47s | **98x** |
| whisper-medium | 769M | 0.35s | 1.15s | **41x** |
| whisper-large-v3 | 1.5B | 0.50s | 1.96s | **24x** |
| whisper-large-v3-turbo | 809M | 0.12s | 0.86s | **55x** |

*RTF = Real-Time Factor (higher is faster). RTF of 100x means 1 minute of audio transcribes in ~0.6 seconds.*

### Results (M1 Max, 64GB)

STT with Parakeet (default environment, Whisper unavailable due to numpy dependency mismatch):

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| parakeet-tdt-0.6b-v2 | 0.28s | 1.01s | **9.9x** |
| parakeet-tdt-0.6b-v3 | 0.30s | 0.19s | **52.7x** |

STT with Whisper (explicit `numpy==2.3.5` + `uv run --no-sync`):

| Model | Load Time | Transcribe Time | RTF |
|-------|-----------|-----------------|-----|
| whisper-tiny | 4.02s | 1.05s | **9.5x** |
| whisper-small | 10.15s | 1.03s | **9.7x** |
| whisper-medium | 22.96s | 2.20s | **4.6x** |
| whisper-large-v3 | 38.34s | 0.96s | **10.5x** |
| whisper-large-v3-turbo | 21.79s | 0.70s | **14.3x** |
| parakeet-tdt-0.6b-v2 | 0.47s | 0.18s | **54.4x** |
| parakeet-tdt-0.6b-v3 | 1.13s | 0.18s | **54.6x** |

### Model Recommendations

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Real-time transcription** | whisper-tiny | Fastest (197x RTF), low latency |
| **General use** | whisper-large-v3-turbo | Best balance of speed (55x) and quality |
| **Highest accuracy** | whisper-large-v3 | Most accurate, supports 99+ languages |
| **Low memory** | whisper-small | Good quality at 244M params |

### Transcription Quality

All models correctly transcribed the test audio. Example output:

```
Input text:
"Welcome to this comprehensive speech to text demonstration.
This audio sample is designed to test the accuracy and speed of various speech recognition models.
The quick brown fox jumps over the lazy dog..."

Whisper-large-v3 output:
"Welcome to this comprehensive speech to text demonstration.
This audio sample is designed to test the accuracy and speed of various speech recognition models.
The quick brown fox jumps over the lazy dog..." (identical)
```

### Supported Languages

Whisper models support 99+ languages including:
- English, Spanish, French, German, Italian, Portuguese
- Chinese (Mandarin, Cantonese), Japanese, Korean
- Arabic, Hindi, Russian, Turkish, Ukrainian
- And many more

## Text-to-Speech (TTS) Benchmarks

### Running TTS Benchmarks

```bash
python examples/benchmark_audio.py --tts
```

### Results (M4 Max, 128GB)

**Test:** Generate audio for 3 text samples (short, medium, long)

| Model | Load Time | Chars/sec | RTF* |
|-------|-----------|-----------|------|
| Kokoro-82M-bf16 | 0.8s | 350+ | **22x** |
| Kokoro-82M-4bit | 0.4s | 320+ | **20x** |

*RTF = Real-Time Factor. RTF of 22x means 1 second of audio generates in ~0.045 seconds.*

### TTS Results (M1 Max, 64GB)

| Model | Load Time | Avg Chars/s | Avg RTF |
|-------|-----------|-------------|---------|
| Kokoro-82M-bf16 | 2.81s | 176.0 | **11.9x** |
| Kokoro-82M-4bit | 0.22s | 225.6 | **15.5x** |

### TTS Quality

Kokoro produces natural-sounding speech with:
- 11 built-in voices (male and female)
- Support for 8 languages (English, Spanish, French, Japanese, Chinese, Italian, Portuguese, Hindi)
- 82M parameters, fast and lightweight

## Audio Processing Benchmarks

### SAM-Audio (Source Separation)

**Test:** Separate drums from 30-second rock song

| Metric | Value |
|--------|-------|
| Model | sam-audio-large-fp16 |
| Processing time | ~20s |
| Peak memory | ~27 GB |
| Output sample rate | 48000 Hz |

## Running All Audio Benchmarks

```bash
# Run all benchmarks
python examples/benchmark_audio.py --all

# Or run individually
python examples/benchmark_audio.py --stt
python examples/benchmark_audio.py --tts
```

## Available Models on mlx-community

### STT Models
- `mlx-community/whisper-tiny-mlx`
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-large-v3-mlx`
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/parakeet-tdt-0.6b-v2`
- `mlx-community/parakeet-tdt-0.6b-v3`

### TTS Models
- `mlx-community/Kokoro-82M-bf16` (recommended)
- `mlx-community/Kokoro-82M-4bit`
- `mlx-community/chatterbox-turbo-fp16`
- `mlx-community/VibeVoice-Realtime-0.5B-4bit`

### Audio Processing
- `mlx-community/sam-audio-large-fp16`
