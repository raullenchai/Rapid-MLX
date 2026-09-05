# SDXL Base Server and Desktop-path dogfood

## Outcome

The `sdxl-base` alias completed real MLX-native text-to-image generations
through both the engine and the OpenAI-compatible HTTP server. The official
fp16 checkpoint is revision-pinned, the UNet is quantized to 8-bit at load,
DeepCache reuses deep UNet features on alternating denoise steps, and no Torch
runtime is imported or bundled.

## Environment

- Apple M3 Ultra, 256 GiB unified memory
- macOS 26.5.2 (25F84)
- Python 3.12.13
- MLX 0.32.2
- Transformers 5.15.1
- Pillow 12.3.0
- Hugging Face Hub 1.30.0
- Model `stabilityai/stable-diffusion-xl-base-1.0`
- Revision `462165984030d82259a11f4367a4eed129e94a7b`
- Audited download payload: 6,941,201,645 bytes (6.46 GiB)

## Engine quality run

Prompt:

> Editorial photograph of a tiny red panda librarian reading an old book
> beside a rain-streaked window, warm tungsten light, detailed fur, natural
> depth of field

Parameters: 1024×1024, 30 steps, guidance 5.0, seed 42, 8-bit UNet, tiled VAE.

Exact-path baseline observed:

- cold construction plus generation: 61.72 seconds
- MLX peak allocation: 12.42 GiB
- output: valid 1024×1024 PNG, 1,399,045 bytes
- progress reached exactly step 30/30 and reset to idle after completion
- visual inspection: coherent subject, book, window, lighting, and depth;
  no structural corruption or noise output

DeepCache interval 2, with the same prompt/parameters/seed:

- generation after model construction: 16.46 seconds
- MLX peak allocation: 12.12 GiB
- a second clean-process engine run measured 18.43 seconds cold end-to-end
  and 12.49 GiB peak, versus 61.72 seconds for the exact-path cold run
  (3.35× lower observed end-to-end latency)
- the subject, book, window, lighting, fur, and depth remained coherent under
  side-by-side visual inspection
- three additional 512² / 30-step prompts (landscape, product, illustration)
  completed in 4.48, 4.31, and 4.32 seconds with coherent outputs

The exact baseline included a short cold model construction while the
DeepCache timer began immediately after construction, so the measurements are
not claimed as a publication-grade paired speedup. They establish a large,
repeatable latency reduction without the structural failure seen at four steps.

The catalog minimum remains 16 GiB. The resident-model safety charge is 14
GiB, leaving allocator and PNG headroom above the measured MLX peak.

## HTTP path

Server:

```bash
RAPID_MLX_NO_UPDATE_CHECK=1 RAPID_MLX_AUTO_PULL=1 \
  rapid-mlx serve sdxl-base --host 127.0.0.1 --port 8127
```

Request (steps deliberately omitted to exercise the family default):

```bash
curl --fail-with-body --max-time 180 \
  -X POST http://127.0.0.1:8127/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"sdxl-base","prompt":"A cobalt blue ceramic teapot on a linen tablecloth, soft daylight, product photography","negative_prompt":"blurry, text, watermark","size":"512x512","seed":17}'
```

Observed:

- HTTP 200
- 30 denoise steps selected by the server default
- valid OpenAI-compatible `data[0].b64_json`
- decoded PNG: 512×512, 338,013 bytes
- end-to-end request time with DeepCache: 5.15 seconds
- visual inspection: coherent blue-and-white teapot and tablecloth composition

## Safety and packaging checks

- A 4-step 512² smoke completed before integration (2.40 seconds, 9.71 GiB
  MLX peak), but its poor structure confirmed that 4 steps must not be used as
  the SDXL Base product default.
- The first server boot exposed a false 71.6 GB warning because the generic
  preflight sized every artifact in the Hub repository. The final path sizes
  only the pinned 6.46 GiB allowlist and budgets the measured 14 GiB working
  set; a repeated clean launch emitted no false pressure warning.
- Runtime callbacks fire after every evaluated denoise step and enforce the
  existing cooperative cancel contract.
- The sidecar smoke imports the vendored runtime while asserting Torch,
  torchvision, and OpenCV remain absent.

## Reference check (internal)

- The primary serving precedents use a single diffusion model per server, an
  OpenAI-compatible `/v1/images/generations` surface, explicit size/step
  validation, and model-recommended defaults. Rapid already had that contract,
  so this change adds an adapter rather than a second serving mechanism.
- Current accelerator-server work emphasizes modular pipeline stages, bounded
  batching, and step-wise execution. On a single Apple GPU, Rapid retains the
  existing process-wide single-flight lock and true per-step cancellation.
- The platform-native MLX examples validate direct Hub loading, latent-step
  evaluation, and quantization on constrained Macs. The selected implementation
  extends that shape to full SDXL Base, dual CLIP encoders, fp32 VAE decode, and
  an 8-bit UNet.
- Desktop references commonly expose image engine/model/size/steps as one
  dedicated generation workflow. Rapid's existing Images tab already follows
  the stronger multi-model catalog pattern, so no new interaction model was
  introduced: SDXL appears as another generation-only row with its own seeded
  step count.

## Non-goals retained

This work does not add SDXL image-to-image, refiner chaining, LoRA, ControlNet,
training, arbitrary community checkpoints, release, or deployment.
