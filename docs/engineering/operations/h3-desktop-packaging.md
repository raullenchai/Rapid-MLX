# H3 Desktop packaging

The sidecar bundles the pinned H3 generation, processing, reference and weight
loading modules, both H3 adapters, the image-processing and text-encoder
closure, and the output encoder. The existing dependency lock supplies these
modules; no model assets are included. Output video uses the bundled encoder,
with system audio conversion for synchronized audio output.

Ref2VA always advertises image references. Timed reference limits are advertised
only when their input decoders are present: video needs OpenCV and PATH-resolved
ffprobe/ffmpeg (including soundtrack decoding); audio needs miniaudio,
mlx_audio.audio_io and PATH-resolved ffprobe/ffmpeg. These pinned input paths do
not use the output-only FFMPEG_BINARY override. The signed bundle currently
lacks ffprobe and OpenCV, so it offers image references only. Unsupported timed
uploads return 503. Adding input decoders requires a separate packaging change.

Desktop derives its importer and validation from live reference limits,
including formats, bytes, per-kind counts, total count and visual requirements.
Advertised duration ranges appear as hints; the server validates durations.
First/last frames and reference packs remain mutually exclusive.

The normal sidecar build runs a post-pruning smoke check that imports the exact
H3 symbols used by both adapters, encodes silent and synchronized-audio video,
and checks cropped output. Missing imports or broken output encoding fail the
build before publication. The source smoke test verifies that this check covers
the adapters' imports and the audio encode path. It does not prove full model
generation; that requires model assets and sufficient unified memory. No signed
sidecar rebuild or full generation was performed for this change.

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
