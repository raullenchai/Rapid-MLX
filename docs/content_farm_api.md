# Content-Farm Engine API

OpenAI-style HTTP routes that expose the rapid-mlx content-farm engines
(music generation, forced alignment, and — soon — video generation) so
colleagues can integrate over a standard API.

All routes live on the rapid-mlx server (`vllm_mlx`). They follow the
same conventions as the existing `/v1/audio/*` OpenAI-compatible routes:
JSON or multipart request bodies, `Authorization: Bearer <api-key>` when
the server was started with `--api-key`, and OpenAI-shaped error
envelopes (`{"error": {"message", "type", "code", "param"}}`).

| Endpoint | Method | Status | Engine |
| --- | --- | --- | --- |
| `/v1/audio/music` | POST | **LIVE** | `MusicEngine` (Stable Audio 3, MLX-native) |
| `/v1/audio/transcriptions` (`text` field) | POST | **LIVE** | `STTEngine.align` (Qwen3 forced aligner) |
| `/v1/video/generations` | POST | **CONTRACT-ONLY** (returns 501) | `VideoEngine` — no backend yet (LTX-2.3 pending, see [The interface to implement](#the-interface-to-implement)) |

The audio routes are attached only when the server runs with an
audio-capable model or `--enable-audio`. The video route is always
registered (it has no engine to load) and returns HTTP 501 until a
backend is integrated.

---

## 1. `POST /v1/audio/music` — text → music / SFX (LIVE)

Generates music or sound effects from a text prompt via the MLX-native
Stable Audio 3 engine. Request-in / audio-bytes-out, the same shape as
`/v1/audio/speech`.

### Request (JSON body)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | string | `"medium"` | DiT/decoder pairing. `medium` (higher quality) or `sm-music` / `sm-sfx` (fast small). Unknown values fall back to engine defaults. |
| `input` | string | — (required) | Natural-language prompt. Non-blank, max 4096 chars (it becomes an argv element for the SA3 CLI). |
| `seconds` | number | `30.0` | Clip length. `0 < seconds <= 47` (SA3 ceiling). NaN/inf rejected. |
| `steps` | integer | `8` | Pingpong sampling steps. `1..200`. |
| `negative_prompt` | string \| null | `null` | CFG negative branch (e.g. `"vocals, singing"`). Max 4096 chars. |
| `seed` | integer \| null | `null` | Fixed seed for reproducibility. |
| `response_format` | string | `"wav"` | Only `wav` is supported. |

### Response

`200 OK`, `Content-Type: audio/wav` — the raw WAV bytes (same delivery
as `/v1/audio/speech`).

Errors: `400` for schema violations (blank or over-4096-char `input`,
`seconds > 47`, unsupported `response_format`); `500`
(`code="music_generation_failed"`) if the engine fails or produces no
audio; `503` if the engine's runtime deps are unavailable.

> **On the status code:** the schema rejection is a FastAPI
> `RequestValidationError`, which the rapid-mlx server's global handler
> normalizes to **400** with a sanitized envelope (see
> `install_exception_handlers` in `vllm_mlx/server.py`). Stock FastAPI
> would emit 422 — you'll see 422 only if you mount the router on a bare
> app without those handlers, as the unit tests do.

### curl

```bash
curl -s http://localhost:8000/v1/audio/music \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -d '{
        "model": "medium",
        "input": "epic cinematic war drums, tense build-up",
        "seconds": 20,
        "steps": 8,
        "negative_prompt": "vocals, singing",
        "seed": 42
      }' \
  --output bgm.wav
```

---

## 2. `POST /v1/audio/transcriptions` with a `text` field — forced alignment (LIVE)

The existing OpenAI Whisper-compatible transcription route. When the
request includes a **`text`** field, the route switches from ASR to
**forced alignment**: it aligns the *known* transcript in `text` to the
uploaded audio and returns per-unit (per-character for Chinese, per-word
for space-delimited languages) start/end timestamps with zero
recognition error. When `text` is absent, behavior is unchanged (normal
ASR).

### Request (multipart/form-data)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `file` | file | — (required) | The audio whose speech is `text`. |
| `text` | string | *(absent)* | **Presence switches to alignment.** The known transcript to align. |
| `model` | string | `qwen3-aligner` when `text` present, else `whisper-large-v3` | Must resolve to a forced-aligner model (family `qwen3_aligner`) for the alignment path. |
| `language` | string | aligner default (`Chinese`) | Alignment language, forwarded to the aligner. |
| `response_format` | string | `verbose_json` when `text` present, else `json` | One of `json`, `text`, `srt`, `vtt`, `verbose_json`. |

`model`/`language`/`response_format`/`text` are accepted on the
multipart form (OpenAI-style) or as query parameters (internal-caller
parity).

### Response

Shares the transcription serializers. `verbose_json` (the alignment
default) returns:

```json
{
  "task": "transcribe",
  "text": "临终前",
  "language": "Chinese",
  "duration": 0.6,
  "segments": [
    {"id": 0, "start": 0.0, "end": 0.2, "text": "临"},
    {"id": 1, "start": 0.2, "end": 0.4, "text": "终"},
    {"id": 2, "start": 0.4, "end": 0.6, "text": "前"}
  ]
}
```

`srt` / `vtt` return subtitle bodies built from the same segments; `text`
returns the transcript as `text/plain`.

Errors: `400` (`code="invalid_alignment_request"`) when the chosen model
is not a forced aligner or the `text` is empty/blank; `404` for an
unknown `model` alias; `400` (`invalid_audio_file`) for a corrupted
upload; `413` for uploads over 25 MB.

### curl

```bash
# Forced alignment: align a known transcript to the audio.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -F "file=@line.wav" \
  -F "text=临终前她握着我的手" \
  -F "language=Chinese" \
  -F "response_format=verbose_json"

# No text field → unchanged ASR behavior.
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -F "file=@clip.wav" \
  -F "model=whisper-large-v3"
```

---

## 3. `POST /v1/video/generations` — text → video / image → video (CONTRACT-ONLY)

The full request/response contract and the `VideoEngine` interface are
defined now so colleagues can integrate against a stable shape. **There
is no backend yet** — the route validates the request and then returns
**HTTP 501**. A future MLX-native LTX-2.3 backend implementing
`vllm_mlx.video.engine.VideoEngine` makes the route go live with no
change to the route handler.

One schema covers both modes: **text-to-video** when `image` is omitted,
**image-to-video** when `image` carries the conditioning first frame.

### Request (JSON body)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | string | `"ltx-2.3"` | Target backend. |
| `prompt` | string | — (required) | Natural-language description. Must be non-blank. |
| `image` | string \| null | `null` | Conditioning first frame for i2v: a base64 string / `data:` URI, or an `http(s)` URL. `null` → text-to-video. |
| `height` | integer | `704` | Output height, `1..4096`. |
| `width` | integer | `1216` | Output width, `1..4096`. |
| `num_frames` | integer | `97` | Frames to render, `1..4096`. |
| `frame_rate` | number | `25.0` | Playback fps, `0 < fps <= 240`. |
| `steps` | integer \| null | `null` | Denoising steps, `1..500` (`null` = backend default). |
| `seed` | integer \| null | `null` | Fixed seed. |
| `negative_prompt` | string \| null | `null` | CFG negative branch. Max 4096 chars. |
| `response_format` | string | `"mp4"` | Only `mp4` is supported. |

### Response (once a backend is integrated)

```json
{
  "created": 1730000000,
  "model": "ltx-2.3",
  "data": [
    {
      "b64_video": "AAAAIGZ0eXBpc29t...",
      "url": null,
      "audio": null,
      "format": "mp4",
      "width": 1216,
      "height": 704,
      "num_frames": 97,
      "frame_rate": 25.0
    }
  ]
}
```

Each `data[]` item populates exactly one of `b64_video` (inline base64
mp4) or `url`. The wired handler returns `b64_video` — a server-side
filesystem path is not a URL the client can fetch, and echoing one would
leak the server's layout. A backend that uploads to real object storage
can populate `url` instead. `audio` carries LTX-2.3's native soundtrack
(base64) when the backend emits one, else `null`.

### Current behavior (no backend)

`501 Not Implemented`:

```json
{
  "error": {
    "message": "video backend not yet integrated; see docs/content_farm_api.md",
    "type": "not_implemented_error",
    "code": "video_backend_not_implemented",
    "param": null
  }
}
```

Schema violations still fail at the request boundary (missing `prompt`,
`num_frames=0`, `frame_rate=NaN`, `response_format="webm"`, etc.) so you
can develop against the real wire contract today — as **400** on the
rapid-mlx server, per the note in §1.

### The interface to implement

A backend implements `vllm_mlx/video/engine.py::VideoEngine`:

```python
def generate(
    self,
    prompt: str,
    out_path: str | Path,
    *,
    image: str | None = None,
    height: int = 704,
    width: int = 1216,
    num_frames: int = 97,
    frame_rate: float = 25.0,
    steps: int | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> Path: ...
```

and registers a factory so `resolve_video_engine(model)` returns it:
assign `vllm_mlx.video.engine._VIDEO_ENGINE_FACTORY` to a callable
taking the requested `model` id and returning the engine. The route then
goes live with no handler change.

### curl

```bash
# Text-to-video (returns 501 today).
curl -s http://localhost:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -d '{
        "model": "ltx-2.3",
        "prompt": "a red fox trotting through fresh snow, cinematic",
        "height": 704,
        "width": 1216,
        "num_frames": 97,
        "frame_rate": 25
      }'

# Image-to-video (i2v) — condition on a first frame.
curl -s http://localhost:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -d '{
        "prompt": "the character turns and smiles",
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
        "num_frames": 65
      }'
```
