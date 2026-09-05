# FLUX.1 schnell image-generation dogfood

## Scope

This run validates the pinned `flux-schnell` alias through Rapid-MLX's
OpenAI-compatible image endpoint. It is a functional and resource-sizing
dogfood pass, not a comparative quality benchmark.

## Environment

- Date: 2026-09-04 (America/Los_Angeles)
- Hardware: Apple M3 Ultra, 256 GiB unified memory
- OS: macOS 26.5.2 (25F84)
- Python: 3.12.13
- mflux: 0.19.1
- Rapid-MLX base commit: `594cd67b4e74d9da727733bf07da31c4afd3c57f`
- Model: `mflux-community/flux-1-schnell-mflux-q4`
- Revision: `bcdbe817ad51175959b2e691e64eca626db30558`
- Download footprint: 9,613,040,056 bytes (9.0 GiB as displayed)

## Reproduction

Start the source checkout under `/usr/bin/time`:

```bash
/usr/bin/time -l uv run rapid-mlx serve flux-schnell \
  --port 18085 --log-level INFO
```

From a second shell, submit the same default-size request used by Desktop:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:18085/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"flux-schnell","prompt":"A clean product photograph of a small brushed-aluminum robot beside an orange, soft studio light, crisp details","size":"1024x1024","n":1,"response_format":"b64_json","seed":42}'
```

## Result

- HTTP status: 200
- Output: one valid 1024×1024 RGB PNG
- Inference steps: 4
- Request wall time: 23.36 seconds
- Server maximum resident set size: 10,155,180,032 bytes (9.46 GiB)
- Warm-cache completeness gate: passed, including `text_encoder_2` and
  `tokenizer_2`

The measured peak supports a 9.5 GiB alias-only resident fallback charge. The
catalog requires 16 GiB of system memory to retain practical headroom on base
Apple Silicon machines.

Desktop validation used the source-built app's isolated `image-generation`
golden journey with its deterministic fake sidecar; real weights were exercised
through the Server request above. The full Images flow passed: model selection,
aspect changes, progress/finalizing states, generation, gallery/refinement
behavior, and the generation-only versus edit-capable picker split. Separate
catalog tests pin `flux-schnell` to that generation-only GUI path.
