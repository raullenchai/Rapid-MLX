# FLUX 3 Image Klein (next release)

`flux3-klein` resolves to `black-forest-labs/flux-3-image-klein`; it uses
16 steps, no guidance, and no negative prompt. `flux3-klein-base` resolves to
`black-forest-labs/flux-3-image-klein-base`; it uses 50 steps and guidance 3.5.
Both use 4-bit on-load quantization by default, matching FLUX 2 Klein.
The adapter feature-detects FLUX 3 runtime support in the supported mflux
0.23–0.24 range and reports a clear upgrade error if its classes are absent.
Explicit output dimensions must be positive multiples of 16.

The repositories are gated. Review and accept their terms on Hugging Face,
then run `hf auth login` before downloading. Rapid-MLX never accepts terms
on your behalf. The [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)
permits non-commercial use and commercial output generation by qualifying
users with less than US$5M gross annualized revenue, including affiliates,
subject to its terms; consult the full license for other uses.

```bash
rapid-mlx serve flux3-klein
curl http://localhost:8000/v1/images/edits \
  -F model=flux3-klein -F prompt='Combine these references' \
  -F image=@first.png -F image=@second.png
```

Both variants accept 1–10 references via repeated multipart `image` fields
or `image[]` fields. The first reference determines the edit canvas; as with
existing image edits, the API accepts `size` but leaves canvas sizing to the
backend. Uploads share the existing aggregate 25 MB request limit.

`rapid-mlx models --json` image rows and `/v1/models` model information expose
`max_reference_images`. Desktop uses it to cap attachments, defaulting to one
when older servers omit it. FLUX 2 Klein and Qwen Image Edit remain limited
to one reference. Qwen Image 2.1 accepts multiple references up to the installed
mflux `QwenImage21Edit` limit when that backend is present; older runtimes
without it retain one-reference img2img.
Single-reference Qwen 2.1 requests retain their existing img2img behavior.
Qwen Image 2.1 Turbo remains text-to-image only.

No real-weight qualification or performance claims accompany these aliases.
