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
| `/v1/video/generations` | POST | **LIVE** when a backend is configured, else `501` | `WanVideoEngine` — Wan 2.1 / 2.2 via [mlx-video](https://github.com/Blaizzy/mlx-video) (see [§3](#3-post-v1videogenerations--text--video--image--video)) |

The audio routes are attached only when the server runs with an
audio-capable model or `--enable-audio`. The video route is always
registered and answers `501` until a video backend is configured — the
request is still fully validated, so the contract stays developable-against
on a text-only server.

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
| `text` | string | *(absent)* | **Presence switches to alignment.** The known transcript to align. A whitespace-only value is a `400`, never a silent downgrade to ASR. |
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
is not a forced aligner, or when `text` is whitespace-only (`param:
"text"`) — sending `text` means "I have the transcript, give me
timings", so the route refuses rather than quietly answering a different
question with ASR.

> One boundary worth knowing: a **truly empty** `text=""` is
> indistinguishable from an omitted field, because FastAPI coerces an
> empty form value to `None` for an optional parameter. `text=""`
> therefore behaves as "no `text`" and runs ASR. Any non-empty blank
> value (`"   "`) is rejected as above. Send no `text` field for ASR and
> a real transcript for alignment; don't rely on the empty string.

Also: `404` for an unknown `model` alias; `400`
(`invalid_audio_file`) for a corrupted upload; `413` for uploads over
25 MB.

A whitespace-only `model` or `response_format` is treated as unset and
takes the lane default (`qwen3-aligner` / `verbose_json`) rather than
failing — a truly empty form field is already coerced to absent by
FastAPI.

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

## 3. `POST /v1/video/generations` — text → video / image → video

One schema covers both modes: **text-to-video** when `image` is omitted,
**image-to-video** when `image` carries the conditioning first frame.

The built-in backend is **Wan 2.1 / 2.2** (Alibaba, Apache 2.0) running
MLX-natively through [mlx-video](https://github.com/Blaizzy/mlx-video) —
see [Enabling the Wan backend](#enabling-the-wan-backend). With nothing
configured the route answers `501` and the schema still validates, so
clients can build against the contract before any weights exist.

### Request (JSON body)

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | string | `"ltx-2.3"` | **Ignored.** rapid-mlx serves ONE checkpoint per process (as with the LLM lane), so this selects nothing — the served checkpoint is whatever `$RAPID_MLX_WAN_MODEL_DIR` names. The response reports the checkpoint that actually ran (e.g. `wan2.2-ti2v`), not this value. |
| `prompt` | string | — (required) | Natural-language description. Must be non-blank. |
| `image` | string \| null | `null` | Conditioning first frame for i2v. One of: a `data:image/*;base64,...` URI, an `http(s)://` URL, or a bare base64 payload. Max 12 MB of string and 64 MP decoded (checked from the header before decoding, so a compression bomb can't expand into memory). `null` → text-to-video. **The Wan backend accepts the two inline forms only** — see [Why remote image URLs are refused](#why-remote-image-urls-are-refused). |
| `height` | integer | `704` | Output height, `1..4096`. |
| `width` | integer | `1216` | Output width, `1..4096`. |
| `num_frames` | integer | `97` | Frames to render, `1..4096`. **Wan requires `4n+1`** (its latent temporal stride is 4) — `49`, `81`, `97` are valid, `80` is not. A violation is a `400` naming the nearest valid values. |
| `frame_rate` | number | `25.0` | Playback fps, `0 < fps <= 240`. **Wan ignores this**: the model emits frames at a fixed trained rate (16 fps for 2.1, 24 fps for 2.2) and fps is a container property, not a generation parameter. The response reports the clip's REAL rate — or `null` if the checkpoint doesn't declare one, rather than echoing a number nothing honoured. |
| `steps` | integer \| null | `null` | Denoising steps, `1..500` (`null` = the checkpoint's default: 50 for Wan2.1, 40 for Wan2.2). **This is the dominant cost** — see [Performance](#performance). |
| `seed` | integer \| null | `null` | Fixed seed. |
| `negative_prompt` | string \| null | `null` | CFG negative branch. Max 4096 chars. |
| `response_format` | string | `"mp4"` | Only `mp4` is supported. |

### Response

```json
{
  "created": 1730000000,
  "model": "wan2.2-ti2v",
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
can populate `url` instead. `audio` carries a native soundtrack (base64)
when a backend emits one — Wan 2.1/2.2 do not, so it is `null` today.

### Error codes

| Status | `code` | Meaning |
| --- | --- | --- |
| `501` | `video_backend_not_implemented` | No backend configured. The request was still fully validated — this is the contract check to develop against. |
| `503` | `video_backend_unavailable` | A backend IS configured but its runtime dependency is missing (or is the wrong package — see the warning below). The message carries the install command. |
| `400` | `invalid_video_request` | A model-specific constraint the generic schema can't express: Wan's `num_frames == 4n+1`; a resolution over the checkpoint's pixel-area ceiling; a remote `image` URL; an `image` on a `t2v`-only checkpoint or a missing one on an `i2v`-only checkpoint. |
| `400` | `invalid_request` | Schema violation (missing `prompt`, `num_frames=0`, `frame_rate=NaN`, `response_format="webm"`, an unsafe `image`…). 400 rather than 422 per the note in §1. |
| `500` | `video_generation_failed` | The backend ran but produced no output. |
| `500` | `video_too_large_to_inline` | The clip exceeds the 256 MB inline-base64 ceiling. |

The `501` body:

```json
{
  "error": {
    "message": "no video backend configured. Set $RAPID_MLX_WAN_MODEL_DIR to a converted MLX Wan 2.1/2.2 checkpoint to serve this route; see docs/content_farm_api.md",
    "type": "not_implemented_error",
    "code": "video_backend_not_implemented",
    "param": null
  }
}
```

### Enabling the Wan backend

**Step 1 — install the generation package.**

```bash
# Pinned to the commit this backend was developed and verified against.
pip install 'git+https://github.com/Blaizzy/mlx-video.git@87db56a51758fefb748a359b90a5283bb8ba4837'
```

Pinning matters more than usual here: because the PyPI name is taken by an
unrelated project there is no versioned release to depend on, so an
unpinned `main` could change `generate_video`'s signature under a working
install. `tests/test_wan_video_backend.py::TestUpstreamSignatureCompatibility`
asserts the keywords we pass still exist whenever the real package is
present, so drift fails a test instead of a production request.

> ⚠️ **Do NOT run `pip install mlx-video`.** That PyPI name belongs to an
> **unrelated project** (`AmiraniLabs/mlx-video`, a 5 KB video *loading*
> utility). It satisfies the `mlx_video` import name and then fails at call
> time in a thoroughly confusing way. This is also why rapid-mlx has no
> `[video]` pip extra: the package we need is only installable from git,
> which PyPI forbids as a direct reference in published metadata. The
> backend probes at runtime and tells you if the wrong one is installed.

**Step 2 — point the server at a converted MLX checkpoint.**

mlx-video needs weights in its own MLX layout (`model.safetensors` or
`{high,low}_noise_model.safetensors`, plus `t5_encoder.safetensors`,
`vae.safetensors`, `config.json`). Either convert the official PyTorch
release yourself:

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
python -m mlx_video.models.wan_2.convert \
    --input ./Wan2.2-TI2V-5B --output ./wan22-ti2v-5b-mlx \
    --quantize --bits 8 --group-size 64
```

…or use a pre-converted community upload (several exist on the Hub with
the exact layout above, ~18 GB for TI2V-5B at 8-bit). **We deliberately do
not ship an alias table pointing at those**: silently fetching multi-GB
weights from an unvetted third-party account on a user's first request is a
supply-chain decision, not a convenience. Name the directory you trust.

```bash
export RAPID_MLX_WAN_MODEL_DIR=/path/to/wan22-ti2v-5b-mlx
rapid-mlx serve <your-llm> --port 8000
```

**Optional tuning** (unset → the checkpoint's own defaults):

| Env var | Effect |
| --- | --- |
| `RAPID_MLX_WAN_STEPS` | Denoising steps. The single biggest cost lever. |
| `RAPID_MLX_WAN_SCHEDULER` | `unipc` (default), `dpm++`, `euler`. |
| `RAPID_MLX_WAN_TILING` | VAE decode tiling: `auto` (default), `none`, `aggressive`, … Lower memory at some speed cost. |
| `RAPID_MLX_WAN_LORA` | `path[:strength][,path[:strength]]`, applied to all models. |
| `RAPID_MLX_WAN_LORA_HIGH` / `_LOW` | Same, for the two halves of a dual-model (A14B) checkpoint. |

### Why remote image URLs are refused

The schema accepts an `http(s)://` URL for `image` because that is the
generic contract and a future backend may implement a safe loader. **The Wan
backend refuses them** and returns `400 invalid_video_request` asking you to
inline the frame instead.

This is deliberate. `image` is the only field a backend dereferences, so
fetching it would make the server's video route its sole outbound-request
primitive — an SSRF vector reaching loopback, RFC1918, and link-local
metadata endpoints, defeatable by DNS rebinding between validation and
connect, and by redirects to any of those. Doing it safely requires
socket-level control: resolve and re-check the address on every connection
*and* every redirect hop, plus size and time bounds. That is a subsystem,
not a helper, and one this backend has no way to exercise. A client can
always fetch the image itself and pass the bytes, so refusing costs little
and removes the whole class of problem.

Inline forms are decoded to a temporary file before rendering, because
mlx-video's I2V path is `PIL.Image.open(path)` — it accepts a filesystem
path and nothing else.

### Which Wan versions, and why not 2.7

Open weights stop at **Wan 2.2**. Wan 2.5, 2.6 and 2.7 are **API-only** —
no HuggingFace weights, no GitHub repo, no self-hosting. Verified against
the [`Wan-AI` HF org](https://huggingface.co/Wan-AI) (tops out at Wan2.2)
and the [`Wan-Video` GitHub org](https://github.com/Wan-Video) (only
`Wan2.1` and `Wan2.2` repos). Several SEO sites claim Wan 2.7 ships
Apache-2.0 open weights; that is false. A local inference server can only
serve what has weights, so this backend covers 2.1 and 2.2.

There is no Wan 2.3 or 2.4 — the series went 2.1 (Feb 2025) → 2.2
(Jul 2025) → 2.5-preview (Sep 2025) → 2.6 → 2.7 (Apr 2026).

| Variant | Params | Pipeline | Native |
| --- | --- | --- | --- |
| Wan2.1 T2V-1.3B | 1.3B | single | 480p, 16 fps |
| Wan2.1 T2V-14B | 14B | single | 720p, 16 fps |
| Wan2.2 TI2V-5B | 5B | single | 720p, 24 fps |
| Wan2.2 T2V-A14B | 27B (14B active) | dual (MoE) | 720p, 24 fps |
| Wan2.2 I2V-A14B | 27B (14B active) | dual (MoE) | 720p, 24 fps |

### Performance

Measured on an **M3 Ultra / 256 GB**, Wan2.2-TI2V-5B at 8-bit, `unipc`:

| Resolution | Clip | Steps | Wall clock | Peak memory |
| --- | --- | --- | --- | --- |
| 832×480 | 1.0 s | 8 | 46 s | 24.5 GB |
| 832×480 | 2.0 s | 8 | 77 s | 38.5 GB |
| 832×480 | 2.0 s | 40 (default) | 295 s | 38.5 GB |

**Steps dominate.** Going 40 → 8 steps cut a 2-second 480p clip from 295 s
to 77 s on identical hardware. If you want 720p to be practical, a
step-distilled LoRA (Wan2.2-Lightning, 4 steps) via
`RAPID_MLX_WAN_LORA_HIGH` / `_LOW` is the lever that matters far more than
resolution or quantization.

Stage breakdown for the 2 s / 8-step run: T5 encode 4 s, weight load 0.2 s,
denoise 54 s, VAE decode 19 s. Weights are loaded per request — a
resident-model mode is a follow-up.

> A widely-cited figure has Wan 2.2 taking **82 minutes** for a 2-second
> clip on an M1 Max. That measurement is ComfyUI / PyTorch-MPS with GGUF
> weights — a different runtime, different hardware and a dual-model 14B
> checkpoint. It is not comparable to the MLX numbers above, in either
> direction; quoted here only because it is the number most people find
> first when asking whether Wan runs on a Mac.

Rendering holds a process-wide lock: a video request is the heaviest thing
this server can be asked to do, and concurrent renders would multiply peak
unified memory. Queued requests wait on the event loop, so the LLM lane and
health probes stay responsive — but a Mac serving coding agents and video
from one process will contend for the GPU.

### Security note on `image`

`image` is the only field in this contract a backend **dereferences**,
which makes it the request's sole server-side fetch primitive. The schema
therefore restricts it at the boundary — so every future backend inherits
the restriction instead of each having to remember it:

- `data:` URIs must declare an `image/*` media type (`data:text/html;...`
  and unlabelled `data:;base64,...` are rejected).
- URL schemes are an allowlist of `http` / `https`. `file://`,
  `gopher://`, `ftp://` and friends are rejected — otherwise
  `file:///etc/passwd` is an arbitrary local-file read the moment a
  backend honours the field.
- A bare base64 payload (no scheme) is accepted as an inline frame.
- The whole string is capped at 12 MB.

- `http(s)` URLs must name a host: `https:///etc/passwd` (allowed scheme,
  empty host, absolute local path) and `http:frame.png` (opaque, no host)
  are rejected, since both are shapes a lenient fetcher resolves against
  the local filesystem.

**This is deliberately NOT complete SSRF defence, and the backend must
finish the job.** Schema validation runs before any network activity, so
it can only constrain the *shape* of the reference. An allowed
`https://` host can still resolve to a private address — `127.0.0.1`,
RFC1918, link-local metadata endpoints (`169.254.169.254`) — or DNS-rebind
between validation and connect, or redirect to any of those.

**A backend that dereferences `image` MUST**, at fetch time:

1. resolve the hostname and reject non-public addresses **per connection**
   (not once up front — that's the rebinding window),
2. re-apply the same check to **every redirect hop**, not just the initial
   URL, and
3. bound response size and time.

The route layer cannot do any of this for you: there is no fetch here to
hook, and doing it correctly requires control of the socket. Treat the
validation above as a shape filter that removes the trivially-wrong
inputs, not as an egress policy.

### Adding another backend

`WanVideoEngine` is one implementation, not a special case — the route
depends only on the Protocol, so a second backend (LTX-2, HunyuanVideo, …)
plugs in without touching `vllm_mlx/routes/video.py`.

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
assign `vllm_mlx.video.engine._VIDEO_ENGINE_FACTORY` to a callable taking
the requested `model` id and returning the engine. Add a `register()` to
`_autoregister()` in that module so the lane self-configures.

Two conventions worth following, both of which the route relies on:

* Raise **`vllm_mlx.video.engine.InvalidVideoRequestError`** for anything
  the CALLER can fix that the generic schema can't express (a frame count
  your model rejects, a resolution ceiling). The route maps it to
  `400 invalid_video_request`. It subclasses `ValueError`, but the route
  catches only the dedicated type on purpose — a bare `except ValueError`
  would also swallow corrupt weights, a bad LoRA and scheduler faults and
  report those as "your request is invalid", so raising plain `ValueError`
  gets you a `500`, not a `400`.
* Raise **`VideoBackendUnavailableError`** (or `ImportError`) for
  OPERATOR-fixable faults you can recognise up front — a missing
  dependency, a model directory that doesn't exist. Those become `503
  video_backend_unavailable` with your message. Raise them from your
  **factory** as well as from `generate`: the route resolves the engine
  outside the generation error mapping, so a factory that raises anything
  else produces an unstructured error.
  Note the Wan backend does *not* try to classify mid-render checkpoint
  load failures this way — telling "these weights are corrupt" apart from
  "this render failed" inside a third-party call is guesswork, so those
  surface as `500` (below) rather than being mislabelled `503`.
* Anything else you raise is treated as an internal fault and becomes a
  generic `500 video_generation_failed` with the traceback in the operator
  log and no detail leaked to the client.
* Expose `native_frame_rate` if your model can't vary fps. The route
  reports it instead of echoing the requested `frame_rate`, so the
  response describes the clip that actually came out. Backends that do
  honour arbitrary fps simply omit the attribute.

### curl

```bash
# Text-to-video. 49 frames = 2.0 s at Wan2.2's native 24 fps.
curl -s http://localhost:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RAPID_MLX_API_KEY" \
  -d '{
        "prompt": "a red fox trotting through fresh snow, cinematic",
        "height": 480,
        "width": 832,
        "num_frames": 49,
        "steps": 8
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
