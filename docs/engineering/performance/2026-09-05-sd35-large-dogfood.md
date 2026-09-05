# Stable Diffusion 3.5 Large Server and Desktop-path dogfood

## Outcome

The `sd35-large-4bit` alias completed real MLX-native text-to-image generation
through the OpenAI-compatible HTTP server. The runtime uses all three published
text encoders (CLIP-L, CLIP-G, and T5-XXL), emits true denoise progress, and
does not execute model-repository Python code.

## Environment

- Apple M3 Ultra, 256 GiB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.13
- MLX 0.32.1
- Transformers 5.15.1
- SentencePiece 0.2.2
- Pillow 12.3.0
- Hugging Face Hub 1.30.0
- Primary checkpoint revision: `0f92f6c2a9f9e1abc6738209e87ac22b049a7d26`
- Shared text-assets revision: `7b7a9946015fe6ae602464dfc026c19f6b6306f9`
- T5 tokenizer revision: `3db67ab1af984cf10548a73467f0e5bca2aaaeb2`
- Exact audited payload: 16,378,940,179 bytes (15.25 GiB)

## HTTP path

Server:

```bash
RAPID_MLX_NO_UPDATE_CHECK=1 RAPID_MLX_AUTO_PULL=1 \
  rapid-mlx serve sd35-large-4bit --host 127.0.0.1 --port 8135
```

Request (steps deliberately omitted to exercise the family default):

```bash
curl --fail-with-body --max-time 240 \
  -X POST http://127.0.0.1:8135/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"sd35-large-4bit","prompt":"Editorial photograph of a tiny red panda astronaut standing on the moon, reflective helmet visor, Earth in the background, detailed fur, cinematic natural light","negative_prompt":"blurry, deformed, text, watermark","size":"512x512","seed":42}'
```

Observed:

- HTTP 200 with a valid OpenAI-compatible `data[0].b64_json`
- 28 denoise steps selected by the server default
- decoded RGB PNG: 512×512, 457,015 bytes
- end-to-end request time: 62.17 seconds
- progress reached exactly 28/28 and reset to idle
- visual inspection: coherent red panda, space suit, moon surface, and Earth;
  no structural corruption or noise output

A second clean-process 512×512 / 28-step run completed in 57.29 seconds and
returned a 501,432-byte PNG. `/usr/bin/time -l` reported:

- maximum resident set size: 7,389,003,776 bytes (6.88 GiB)
- peak memory footprint: 16,667,122,480 bytes (15.52 GiB)
- swaps: 0

The catalog minimum is 32 GiB. Resident admission uses a conservative 20 GiB
charge: the measured 15.52 GiB peak plus allocator/output headroom. The 32 GiB
minimum also leaves room for the app, display compositor, and normal desktop
workloads on unified memory.

## Quality decision: keep T5 enabled

Before integration, the same 512×512 / 28-step prompt was tested without T5.
It produced a coherent astronaut but missed the requested red-panda subject.
The full three-encoder path produced the requested composition in 66.25 seconds
with a 4.76 GiB MLX allocator peak during the staged run. Rapid therefore keeps
T5 enabled by default and does not expose a lower-quality no-T5 mode in this PR.

After the allocator/cache cleanup fixes, a same-process 256×256 / 2-step
runtime smoke completed twice in 2.86 and 3.00 seconds with callbacks 1/2 and
2/2 on both runs. A fault-injection run then cancelled at the first denoise
callback; the same runtime instance recovered and completed the next render.
These short runs validate repeatability and cancellation recovery, not visual
quality or the product's fixed 28-step schedule.

## Safety, reproducibility, and packaging

- Every consumed file is revision-pinned and allowlisted. An incomplete
  primary or auxiliary snapshot fails before model construction.
- The runtime receives only verified local directories; `from_pretrained` is
  local-only and no remote model code is trusted or executed.
- The Desktop sidecar imports the SD3.5 adapter and SentencePiece during its
  build smoke, while continuing to assert that Torch, torchvision, and OpenCV
  are absent.
- The vendored numerical source retains its MIT license and exact provenance.
  No model weights are redistributed by Rapid-MLX.
- The model weights remain subject to the
  [Stability AI Community License](https://stability.ai/license), including its
  revenue threshold and Acceptable Use Policy. Users must confirm that those
  terms cover their deployment.

## Reference check (internal)

- The primary inference-server precedents were checked for Stable Diffusion 3
  pipeline structure, explicit stages, recommended sampling defaults, and
  gated-model handling. Rapid retained its existing single-flight image lane
  and OpenAI-compatible transport, adding only a family adapter.
- MLX-native implementations were checked next for tensor mapping, quantized
  checkpoint loading, staged encoder/model/VAE residency, and platform API
  compatibility. The selected runtime was validated on MLX 0.32.1 and adapted
  to current allocator and attention APIs.
- The Desktop already has the appropriate dedicated Images workflow and generic
  image-model catalog. The GUI change is intentionally limited to the model row
  and its 28-step progress seed; no new interaction model was introduced.

## Non-goals retained

This work does not add image-to-image, ControlNet, LoRA, arbitrary SD3-family
checkpoints, a no-T5 quality toggle, training, release, or deployment.
