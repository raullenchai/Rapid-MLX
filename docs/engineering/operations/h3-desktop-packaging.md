# H3 Desktop packaging

The sidecar bundles the pinned H3 generation, processing, reference and weight
loading modules, both H3 adapters, the image-processing and text-encoder
closure, and the output encoder. The existing dependency lock supplies these
modules; no model assets are included. Output video uses the bundled encoder,
with system audio conversion for synchronized audio output.

Ref2VA always advertises image references. Video references use bundled FFmpeg
for all frames, preserving RGB order, native display resolution, autorotation,
and frame count; downstream sampling/resizing is unchanged. The adapter patches
both pinned decoder import sites before constructing a reference request. Array
inputs retain the pinned processor behavior. OpenCV is not used for reference
decoding and is deliberately excluded from the bundle; its Python dependency
remains available to source installations.

Timed limits require ffprobe and ffmpeg on PATH. The cached capability check
requires H.264 for video and AAC plus importable miniaudio/mlx_audio.audio_io
for audio. Missing or broken tools fail closed; unadvertised uploads return 503.
The sidecar puts `media-bin` first on PATH, with relative links to the two tools
in `bin`. Thus bundled tools take precedence over Homebrew in Desktop and
behavior matches advertised capabilities. `media-bin` contains no `rapid-mlx`,
so the CLI version check still discovers the external CLI. FFMPEG_BINARY and
FFPROBE_BINARY also point to the bundled binaries.

Supported reference containers are MP4/MOV/WebM/MKV video and
WAV/MP3/M4A/AAC/FLAC/OGG audio. Supported video codecs are H.264, HEVC, VP8 and
VP9. HEVC VideoToolbox acceleration also requires the native HEVC decoder for
bitstream parsing. AAC uses AudioToolbox (`aac_at`), not the native AAC decoder.
ALAC is not enabled. Codecs the runtime cannot decode are rejected at upload
with 400; the bundled build does not decode AV1. Pip installations use the
codec support of the FFmpeg on PATH, including AV1 when available. Video
soundtracks must also use a decodable audio codec.
A supported container does not imply support for every codec it can carry.

FFmpeg remains pinned to 7.1.5, static, with autodetection and networking
disabled, only file/pipe protocols, and no GPL, nonfree, or version3 features.
The source archive and LGPL notice continue to ship with the app. Both tools
must target macOS 14.0 and pass the same configuration license guard.

| Component | License | Why needed |
| --- | --- | --- |
| ffmpeg, ffprobe; avutil, avcodec, avformat, avfilter, avdevice core | LGPL-2.1+ | Decode, inspect streams and run bounded media pipelines; no input devices enabled |
| CoreFoundation, CoreMedia, CoreVideo, CoreServices, libSystem | Apple system frameworks/runtime | Media framework dependencies; no external codec libraries |
| swscale, swresample | LGPL-2.1+ | RGB conversion/scaling and PCM sample-format/rate conversion |
| file, pipe protocols | LGPL-2.1+ | Local files and subprocess pipes only |
| mov, rawvideo, matroska, wav, mp3, aac, flac, ogg demuxers | LGPL-2.1+ | Advertised reference containers and raw RGB encoder input |
| h264, hevc, vp8, vp9, rawvideo decoders | LGPL-2.1+ | Reference frames and native bitstream parsing |
| aac_at wrapper; AudioToolbox | LGPL-2.1+ wrapper / Apple system framework | AAC/M4A references and video soundtracks |
| mp3float, flac, vorbis, opus decoders | LGPL-2.1+ | Compressed audio references and soundtracks |
| pcm_s16le, pcm_s24le, pcm_s32le, pcm_f32le, pcm_u8 decoders | LGPL-2.1+ | WAV PCM variants |
| h264, hevc, vp8, vp9, aac, mpegaudio, flac, opus, vorbis parsers; automatically selected ac3 parser | LGPL-2.1+ | Packet framing and container dependencies; AC3 decoding is not enabled |
| Automatically selected aac_adtstoasc, vp9_superframe, vp9_superframe_split bitstream filters | LGPL-2.1+ | AAC remux and VP9 packet splitting; verified with `ffmpeg -bsfs` |
| h264_videotoolbox encoder; h264_videotoolbox, hevc_videotoolbox hwaccels; VideoToolbox | LGPL-2.1+ wrappers / Apple system framework | Existing output encoding and accelerated reference decode with software fallback |
| rawvideo, ppm, pcm_s16le, pcm_f32le encoders | LGPL-2.1+ | RGB frames and PCM audio pipes |
| mov, mp4, rawvideo, image2pipe, pcm_s16le, pcm_f32le muxers | LGPL-2.1+ | Output video, self-describing PPM frames and PCM (`-f s16le` / `-f f32le`) pipes |
| crop, format, scale, aresample, aformat, anull, null, transpose, hflip, vflip filters | LGPL-2.1+ | Existing crop/scale, audio conversion and display autorotation |
| Automatically selected rotate, trim, atrim, buffer, abuffer, buffersink, abuffersink | LGPL-2.1+ | FFmpeg CLI filter graph plumbing, rotation and trimming |

Standalone arm64 builds on macOS 26.5.2 with Apple clang 21.0.0, deployment
target 14.0, identical source SHA-256 and `strip -x` measured:

| Binary | Stripped bytes |
| --- | ---: |
| Previous ffmpeg | 3,680,744 |
| Expanded ffmpeg | 6,220,728 |
| New ffprobe | 6,161,912 |
| Total increase | 8,701,896 (8.30 MiB) |

These use the exact old/new step 1.5 configure flags and `make ffmpeg` /
`make ffmpeg ffprobe`; the increase is below the 10 MB target. No formats were
removed to meet it. The content-based Mach-O inventory signs regular files
once, ignores symlinks, and adds one binary (baseline 230). The application copy
preserves relative symlinks; signature verification still targets the real
executables and the app seal covers the links.

Desktop derives its importer and validation from live reference limits,
including formats, bytes, per-kind counts, total count and visual requirements.
Advertised duration ranges appear as hints; the server validates durations.
First/last frames and reference packs remain mutually exclusive.

The normal post-prune smoke imports the exact H3 symbols used by both adapters,
encodes silent and synchronized-audio video, and checks cropped output. With
only `media-bin` and system directories on PATH, it additionally probes a
16-frame H.264 clip, decodes its full RGB shape and FPS, decodes WAV and AAC M4A
at 32 kHz, decodes the synchronized MP4 soundtrack, and requires both timed
capability kinds. Missing tools/links or failed decode assertions fail the build.
This requires a native build environment with Apple media frameworks and MLX
available. Standalone FFmpeg validation and focused tests do not replace the
coordinator's signed sidecar build or full model generation.

From the repository root, using a configured development Python environment:

```sh
swift build --disable-sandbox --package-path apps/rapid-mac
swift test --disable-sandbox --package-path apps/rapid-mac --filter 'Video(H3|Client|GenViewModel|Foundation|FeatureGate)|ModelSize'
python3 apps/rapid-mac/scripts/check-localization.py
python3 -m pytest -q tests/test_video_capabilities.py tests/test_pika_h3_flex_video.py
python3 -m pytest -q tests/test_sidecar_h3_smoke.py
bash -n apps/rapid-mac/scripts/build-sidecar.sh
git diff --check
```
