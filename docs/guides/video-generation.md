# Video generation

Install Rapid-MLX on an Apple Silicon Mac, restart Terminal, and start a video model:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
# Restart Terminal, then:
rapid-mlx serve ltx-2.3-mlx-q4
```

The server prints the local endpoint. Submit a video request as shown below.

Rapid-MLX exposes LTX-2.3, LTX-2.5, MiniMax-H3, Pika H3 Flex, CogVideoX-Fun and Wan
through the asynchronous OpenAI-compatible Videos API.

The video runtime ships with the official installer and requires Python 3.12 or newer.

## Cancel a video job

`GET /v1/videos/capabilities` advertises
`"cancellation": {"queued": true, "in_progress": true}`.
`DELETE /v1/videos/{id}` on a queued job returns 200 with a cancellation error
and removes the record; subsequent requests return 404. On a running job it
returns 202 with `cancel_requested: true`. Cancellation is cooperative: the job
ends with status `failed` and error code `video_generation_cancelled`. Its record
remains available until deleted again. If successful completion wins the race,
the job completes and clears `cancel_requested`.

MiniMax-H3 (including Turbo), Pika H3 Flex, LTX-2.3, Wan and CogVideoX-Fun
check cancellation between denoise steps. LTX-2.5 terminates its generation
process. Lazy Ref2VA downloads check at chunk or file boundaries; a blocking
network read must return before a chunk check can run. Completed cached files
remain usable and interrupted transfers can resume.

During a lazy Ref2VA download, job status stays `in_progress`, the optional
`phase` field is `"downloading"`, and integer `progress` advances monotonically
from 1 to 20 in proportion to the missing asset bytes. `phase` is omitted after
the download, including on failure or cancellation. Completed generation reports
progress 100. The public status values do not change.

## Keep completed videos across restarts

By default, video jobs and MP4 files use a process-temporary directory and are
removed when the server exits. For a desktop app or another long-lived client,
pass an explicit artifact directory when starting the server:

```bash
rapid-mlx serve ltx-2.3-mlx-q4 \
  --video-output-dir /path/to/rapid-mlx-video-artifacts
```

Every completed job and its manifest are committed durably and atomically.
Starting a later server with the same directory restores up to the newest 100
completed jobs, so they remain available through `GET /v1/videos`,
`GET /v1/videos/{id}` and `GET /v1/videos/{id}/content`.
`DELETE /v1/videos/{id}` removes both the record and its MP4. Queued,
interrupted, failed, incomplete or malformed records are never restored as
completed work.

The metadata includes the prompt and generation settings. Choose a
user-private directory with enough free space; Rapid-MLX creates new job
directories with owner-only permissions but does not change the permissions of
an existing parent directory.

## Discover capabilities before serving

`rapid-mlx models --json` lists video aliases separately from other model
types. Each video entry includes `video_modes` (`text-to-video`,
`image-to-video`, or both) and `min_memory_gb`, so clients can choose a
compatible checkpoint before downloading or starting it. After the model is
serving, `GET /v1/videos/capabilities` remains the source for its live size,
duration, frame-rate, workload, reference-image, and optional-control limits.

## Motion and conditioning controls

`POST /v1/videos` accepts optional controls in addition to `seconds`:

- `fps` selects the output frame rate for LTX and CogVideoX-Fun. Wan uses the
  checkpoint's native frame rate and rejects a different value.
- `frames` overrides the frame count derived from `seconds`. LTX requires
  `8n+1` with a minimum of 9; Wan and CogVideoX-Fun require `4n+1` with a
  minimum of 5.
- `guidance_scale` (1–30) and `negative_prompt` are passed to backends with
  classifier-free guidance. LTX-2.5's distilled pipeline rejects both.
- `conditioning_strength` (0–1) controls how closely LTX image-to-video follows
  `input_reference`; it is rejected without a reference image and is not
  currently supported by Wan or CogVideoX-Fun.

For example, request a shorter, lower-frame-rate LTX image-to-video result with
weaker reference conditioning:

```bash
curl http://127.0.0.1:8000/v1/videos \
  -F model=ltx-2.3-mlx-q4 \
  -F 'prompt=the camera sweeps around the subject' \
  -F input_reference=@start.png \
  -F fps=12 \
  -F frames=17 \
  -F guidance_scale=4.5 \
  -F conditioning_strength=0.35 \
  -F 'negative_prompt=static camera, frozen subject'
```

## Wan 2.1 / 2.2

Wan uses the `mlx-video-with-audio` runtime included in the video extra. Four
converted Wan 2.2 checkpoints are registered; the 5B Q8 TI2V checkpoint is the
recommended starting point and supports both text-to-video and
image-to-video.

```bash
brew install ffmpeg
rapid-mlx serve wan2.2-ti2v-5b-q8
```

Create a one-second text-to-video job:

```bash
curl http://127.0.0.1:8000/v1/videos \
  -F model=wan2.2-ti2v-5b-q8 \
  -F 'prompt=a fox running through fresh snow, cinematic tracking shot' \
  -F seconds=1 \
  -F size=832x512 \
  -F seed=42
```

Add `-F input_reference=@start.png` for image-to-video when the served
checkpoint is TI2V or I2V. Poll and download the result through the same
`GET /v1/videos/{id}` and `GET /v1/videos/{id}/content` endpoints shown
below.

The backend reads the checkpoint's native frame rate (16 fps for Wan 2.1,
24 fps for Wan 2.2), enforces Wan's `4n+1` temporal shape and honors the
checkpoint's pixel-area ceiling. Converted local Wan 2.1/2.2 checkpoints can
override the selected Wan alias with `RAPID_MLX_WAN_MODEL_DIR`.

Optional process-level tuning:

```bash
RAPID_MLX_WAN_STEPS=8 \
RAPID_MLX_WAN_SCHEDULER=unipc \
RAPID_MLX_WAN_TILING=auto \
rapid-mlx serve wan2.2-ti2v-5b-q8
```

LoRAs use `path[:strength]` entries through `RAPID_MLX_WAN_LORA`; dual-model
checkpoints additionally accept `RAPID_MLX_WAN_LORA_HIGH` and
`RAPID_MLX_WAN_LORA_LOW`.

## LTX-2.3

MLX-native LTX-2.3. The Q4 checkpoint is a 22.8 GB download and wants at least
24 GB of unified memory; 32 GB or more is comfortable.

The checkpoint is an audio-video one and the pipeline runs
`generate_video_with_audio`, but the track it produces is silent. Rapid-MLX
remuxes it away, so **the MP4 you get back is video-only** — a silent audio
track is worse than none, since it makes downstream tools believe the clip has
sound. Expect no audio from any of the three backends.

```bash
brew install ffmpeg
rapid-mlx serve ltx-2.3-mlx-q4
```

```bash
curl http://localhost:8000/v1/videos \
  -F model=ltx-2.3-mlx-q4 \
  -F 'prompt=A fox running through fresh snow, cinematic tracking shot' \
  -F seconds=4 \
  -F size=768x512
```

LTX honours the `fps`, `frames` and `conditioning_strength` controls described
above; `frames` must satisfy LTX's own frame-count rule.

## LTX-2.5

LTX-2.5 uses a new Gemma 4 text encoder. Its runtime (`ltx_core_mlx` and
`ltx_pipelines_mlx`) ships inside rapid-mlx, so no separate checkout is needed:

```bash
brew install ffmpeg
rapid-mlx serve ltx-2.5-mlx-q8
```

`rapid-mlx pull ltx-2.5-mlx-q8` runs the same runtime preflight before it
downloads the weights, and `rapid-mlx models` marks video aliases that cannot
run on the current host.

The registered Q8 checkpoint is a 67.7 GB download and uses the low-RAM
distilled path by default. The weights are distributed under the LTX-2.x
Community License rather than an open-source license; review the checkpoint's
`LICENSE.md` before use, especially its commercial-use and generated-content
disclosure terms.

`RAPID_MLX_LTX25_RUNTIME` remains available for development against a pinned
`ltx-2-mlx` checkout. When set, it must point at an executable at the audited
revision and replaces the bundled runtime; an invalid override is reported as
an error instead of silently falling back. The bundled runtime carries
provenance metadata for its repository, commit and version, so a missing or
damaged copy is reported as a broken install rather than replaced by an
unverified copy.

Generations have a two-hour safety deadline; set
`RAPID_MLX_LTX25_TIMEOUT_SEC` to a larger value (minimum 60) for unusually
large jobs.

Create a job with the same API. LTX-2.5 generates synchronized audio and Rapid
preserves that audio track:

```bash
curl http://localhost:8000/v1/videos \
  -F model=ltx-2.5-mlx-q8 \
  -F 'prompt=A red fox trots across fresh snow at golden hour' \
  -F frames=97 \
  -F fps=24 \
  -F size=704x480
```

Text-to-video and image-to-video are supported. Frame counts must be `8n+1`;
dimensions are rounded up to the runtime's required 32-pixel boundary and
cropped back to the requested OpenAI size when necessary. The distilled path
does not expose `guidance_scale` or `negative_prompt`.

### Extend an LTX-2.5 video

`POST /v1/videos/extend` uses the existing MP4 as temporal context and returns
one MP4 containing the original and appended frames. It runs the runtime's
video extension pipeline with the source video as temporal context.

```bash
curl http://localhost:8000/v1/videos/extend \
  -F model=ltx-2.5-mlx-q8 \
  -F 'prompt=The camera continues tracking the rider as the dragon banks left' \
  -F input_video=@source.mp4 \
  -F extend_frames=16 \
  -F seed=42
```

The source must be an MP4 at exactly 24 fps, with dimensions from 256 to 1920
pixels in multiples of 32 and a decoded frame count of `8n+1` (at least 9).
Variable-rate files whose nominal and average frame rates disagree, corrupt
videos, and inconsistent frame-count metadata are rejected before generation.
`extend_frames` counts **output frames**, must be a multiple of 8, and accepts
8–48. The endpoint accepts up to 97 total output frames, 24 million
pixel-frames, and a 20 MB upload. The result uses the same asynchronous job and
download endpoints as `POST /v1/videos`. `GET /v1/videos/capabilities` reports
these extension limits. The uploaded source is removed after generation.

The server's `--max-request-bytes` also applies to uploads. To allow the full
20 MiB source plus multipart overhead, start the server with
`--max-request-bytes 23068672`.

## CogVideoX-Fun

Rapid-MLX can serve CogVideoX-Fun as an experimental, single-worker video
generation backend on Apple Silicon. The MVP supports one-second,
672×384 text-to-video jobs.

### Install

```bash
brew install ffmpeg
```

The pinned CogVideoX-Fun MLX runtime ships with Rapid-MLX; no source checkout
or `PYTHONPATH` modification is required.

Start the recommended q4 checkpoint:

```bash
rapid-mlx serve cogvideox-fun-5b-q4 --port 8000
```

The pipeline and weights load on the first job. The q4 checkpoint uses about
14.5 GB peak RSS; a Mac with at least 24 GB unified memory is recommended.

### Create and download a video

```bash
curl http://127.0.0.1:8000/v1/videos \
  -F model=cogvideox-fun-5b-q4 \
  -F 'prompt=a beautiful sunset over the ocean' \
  -F seconds=1 \
  -F size=672x384 \
  -F seed=42
```

The response contains a job ID. Poll it and download the completed MP4:

```bash
curl http://127.0.0.1:8000/v1/videos/video_ID
curl http://127.0.0.1:8000/v1/videos/video_ID/content --output result.mp4
```

Generation is serialized and defaults to 50 diffusion steps. On an M3 Ultra,
the q4 model produced a one-second 672×384 clip in about 338 seconds. Static
scenes were usable in testing; fast subject motion remains experimental.

## Pika H3 Flex

Serve `pika-h3-flex` with at least 192 GB of unified memory, with 256 GB recommended
for Ref2VA. A 512×512 5-second Ref2VA render with an image, a 4-second video,
and a 4-second audio reference peaked near 124 GiB of process memory; the
buffer cache is capped at 1 GiB during generation and reference videos are
encoded tile by tile. It has the same runtime
requirements as MiniMax-H3. The default FL2VA download is about 146.8 GB,
including the MiniMax-H3 base and FL2VA adapter. The first Ref2VA request downloads
about 69.0 GB more (a second transformer and Ref2VA adapter), which can take a
long time. Pre-fetch these assets with
`rapid-mlx pull pika-h3-flex --with-ref2va`. Cached assets are reused.
Generation uses
a fixed six-step renoise schedule, 24 fps video, and synchronized 32 kHz stereo
audio. Set `RAPID_MLX_URL` to your server's base URL for the examples below.

Use text alone, a first frame through `input_reference`, a last frame through
`last_frame`, or both fields for first+last-frame conditioning. Recommended
sizes are `832x480` (480p 16:9) and `480x832`. Durations are 5–15 seconds, rounded
up to the 17n+5 frame grid: 5 seconds gives 124 frames. The maximum is 345 frames,
including requests for 15 seconds. Explicit `frames` must follow the same grid.

```sh
rapid-mlx serve pika-h3-flex
curl "$RAPID_MLX_URL/v1/videos" \
  -F model=pika-h3-flex \
  -F 'prompt=A quiet shoreline at sunrise, with gentle waves' \
  -F seconds=5 -F size=832x480 \
  -F input_reference=@first.png -F last_frame=@last.png
```

Ref2VA generates video and synchronized audio from a prompt and ordered reference
files. Repeat the `references` multipart field; upload order is the packed
conditioning order:

```sh
curl "$RAPID_MLX_URL/v1/videos" \
  -F model=pika-h3-flex \
  -F 'prompt=The character walks along the shoreline as the waves break' \
  -F seconds=5 -F size=832x480 \
  -F references=@character.png \
  -F references=@motion.mp4 \
  -F references=@sound.wav
```

Supplying `references` selects Ref2VA. Without references, text alone selects T2VA,
and `input_reference` or `last_frame` selects FL2VA. References cannot be combined
with either frame field. Reference limits are:

- Up to 9 images, 3 videos, and 3 standalone audio clips, with 12 files total.
  At least one image or video is required.
- Images: JPEG, PNG, or WebP, up to 20 MB and 16 megapixels, with aspect ratios
  from 1:4 to 4:1.
- Videos: MP4, MOV, WebM, or Matroska (MKV), up to 100 MB and 3840 × 2160 source pixels.
- Audio: WAV, MP3, M4A, AAC, FLAC, or OGG, up to 20 MB.
- Each video and standalone audio clip must be 2–15 seconds. Video durations
  total at most 15 seconds; standalone audio durations independently total at
  most 15 seconds. Video soundtracks count toward neither duration total.

Media types are detected from file contents. Video and audio uploads require
`ffprobe`. Reference video is resampled to 24 fps, and audio to 32 kHz stereo,
before truncation to the requested output length. Reference image and video
sizes follow the generated canvas tier; image references may be upscaled.

Switching between FL2VA/T2VA and Ref2VA unloads one 33B transformer and loads the
other (about a minute; the text encoder and VAEs stay loaded); peak memory stays
in the same class as a single mode.

### License and attribution

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

FL2VA and Ref2VA adapters Copyright © 2026 Mellis, Inc. (Pika). It is a Model Derivative of MiniMax H3 distributed under the MiniMax H3 Community License Agreement.

The weights are downloaded from the upstream Hugging Face repositories and are never mirrored.

## HunyuanVideo 1.5 480p text-to-video

`hunyuan-video-1.5-480p` serves the non-distilled 8.3B DiT checkpoint
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, pinned at
`286be7ce72277246578a3e3cc2487e95ddae5bcf`. Use Apple Silicon with **64 GB or more
unified memory**. The checkpoint downloads about **53.4 GB** directly from Hugging
Face into the default cache; it is excluded from the mirror inventory.

```bash
rapid-mlx serve hunyuan-video-1.5-480p
curl http://localhost:8000/v1/videos \
  -F model=hunyuan-video-1.5-480p \
  -F 'prompt=A fox walking through a snowy forest' \
  -F size=848x480 -F frames=33 -F seed=42
```

The existing `/v1/videos` job, progress, cancellation, retrieval and MP4 endpoints
apply. Defaults are 848×480, 121 frames, 24 fps, 50 denoising steps, CFG 6 and flow
shift 5. Explicit frames must be `4n+1`, between 5 and 121. Width and height must
be multiples of 16, with short edge 256–480 and long edge at most 848. Portrait
480×848 is supported. FPS is fixed at 24. Negative prompts and CFG overrides use
the existing request fields. Image conditioning, extension, audio, 720p and
step/CFG-distilled checkpoints are not implemented or advertised.

The native MLX transformer and causal VAE decoder port the Diffusers model,
retaining checkpoint parameter names. Prompt conditioning reuses MLX-LM's Qwen2
text layers (the Qwen2.5-VL text-only path, taking hidden_states[-3]) and the
repository's configurable T5 primitives, with ByT5 glyph extraction and masking.
Both prompt encoders are released before denoising; the DiT is released before
loading the decoder. FP32 transformer shards are converted to BF16 one shard at
a time. The released VAE encoder is unused by T2V and is not loaded.

Decoding uses the reference 256-pixel spatial tiles with 25% overlap and blending.
Each tile retains the full temporal context, preserving causal attention and
first-frame upsampling. Layers and tiles are explicitly evaluated, decoded tiles
move to host memory, and the shared 1 GiB MLX buffer-cache cap bounds retained
buffers. The intended working set is in the roughly 36 GB model class; real
480p peak memory and performance still require the Mac smoke test below.

Scheduler, guider, DiT and VAE parameters come from checkpoint configs. Resolution,
task and distillation remain separate config concerns so later variants can use
this structure; unsupported configurations currently fail explicitly.

### Validation and real-weight smoke

The parity harness generates tiny random models in a separate Torch/Diffusers
process, then checks native MLX outputs in FP32. Torch is not a serving dependency.
It covers a masked joint transformer block, a full two-layer transformer in FP32
and BF16 (outputs compared in FP32), Qwen and ByT5 hidden states, 3-axis
RoPE, causal VAE decode including tiled blending, and every shifted Euler step.
VAE encode is not exercised because this backend does not use it.

```bash
python scripts/check_hunyuan_video15_parity.py \
  --reference-python /path/to/torch-diffusers-venv/bin/python \
  --json-output /private/tmp/hunyuan-video15-parity.json

python scripts/smoke_hunyuan_video15.py \
  --output /private/tmp/hunyuan-video15-smoke.mp4 \
  --frames 33 --steps 50 --seed 42
```

The optional parity and lifecycle tests can also run through pytest:

```bash
HUNYUAN_VIDEO15_REFERENCE_PYTHON=/path/to/torch-diffusers-venv/bin/python \
  python -m pytest tests/test_hunyuan_video15*.py -q
```

The smoke uses the serving worker and real weights, reports prompt/DiT load time,
denoise time, VAE load time, decode time and `mx.get_peak_memory()` in bytes.
`--model` can point at an existing Diffusers snapshot. `--steps 2` is a fast
plumbing check; use the default 50 for a model-quality check. No Metal inference,
real-weight output quality, throughput or 64 GB peak-memory qualification is
claimed from CPU tiny-model parity.

### License and attribution

The model weights use the **Tencent Hunyuan Community License Agreement**,
which excludes the **European Union, United Kingdom and South Korea**. Licensees
with more than 100 million monthly active users under the release-date test in
section 4 must obtain a separate Tencent license. The Acceptable Use Policy
(Exhibit A) and redistribution/service-provider disclosure obligations apply.
Read the bundled [license](../../rapid_mlx/video/hunyuan_video15/LICENSE-TENCENT)
and [NOTICE](../../rapid_mlx/video/hunyuan_video15/NOTICE) before use or deployment.
This integration follows the documentation and upstream-direct download policy
used for MiniMax H3 / Pika; it does not infer the operator's location.

Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement,
Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent
Hunyuan” are owned by Tencent or its affiliate.

Rapid-MLX provides this integration. Tencent is not affiliated with, associated
with, sponsoring, or endorsing Rapid-MLX. Operators offering it to end users must
identify their actual legal service provider as required by the license.
The native Diffusers ports retain their Apache 2.0 notices; model terms are
separate from those software notices.


Port provenance: Diffusers commit `8c54b38fea3d624d23d0e43c0c96037f080a79f9`
(`transformer_hunyuan_video15.py`, `autoencoder_kl_hunyuanvideo15.py` and
`pipelines/hunyuan_video1_5/`); official Tencent cross-check commit
`60783e704160023913bee78f0b47036d393d4dfa`.
Existing [distributed transformer](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py)
and [serving pipeline](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py)
implementations were inspected for established patterns. Their Torch and
distributed-kernel dependencies cannot run directly on MLX; the portable
Diffusers definitions remain this port's numeric reference.
