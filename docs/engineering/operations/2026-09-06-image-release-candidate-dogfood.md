# Image release candidate dogfood — 2026-09-06

## Verdict

All ten built-in image aliases passed the exact-wheel API matrix on real
weights. The same wheel was then assembled into a release-mode Desktop
sidecar; that sidecar passed an offline real-weight generation smoke, and the
release-shaped app passed a real generate, denoise cancellation, recovery,
edit, picker-partition, aspect-ratio, gallery, and clean-quit walk.

This is a **commit-bound qualification receipt**, not authorization to publish
a release. The tested wheel was built from PR head `affc93e7264287ca74ad24a55d49106acb51125c`.
PR #3160 later merged as `05b6e0fc2bfad70acc238461fbdaba92a413350a`
after `main` had advanced, so later source trees and future release artifacts
must run their own exact-artifact matrix.

## Candidate and host

- Rapid-MLX: 0.13.4 at `affc93e7264287ca74ad24a55d49106acb51125c`
- Wheel: `rapid_mlx-0.13.4-py3-none-any.whl`
- Wheel SHA-256: `7267ffcff780356f4512c3c6b939c097b5c1bbd61a2c8c99ff810f150358d2e7`
- API environment: Python 3.12.14, MLX 0.32.2, mflux 0.19.1,
  Pillow 12.3.0
- Host: Apple M3 Ultra, 256 GiB physical unified memory, macOS 26.5.2
  (25F84), low-power mode off
- Requests omitted `steps`. Generation rows used 1024×1024. Edit rows used
  `docs/assets/logo.png`, with dimensions derived from the source image.
- RSS is the highest sampled resident size of the server process tree. The
  system swap counter was identical before and after every alias row.

## Exact model identities

| Alias | Repository | Immutable revision used | Catalog payload bytes |
| --- | --- | --- | ---: |
| `bonsai-image-4b-2bit` | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` | `2c24c81b934a658ba5590cf39088ba929985b4a8` | 3,888,262,196 |
| `flux-schnell` | `mflux-community/flux-1-schnell-mflux-q4` | `bcdbe817ad51175959b2e691e64eca626db30558` | 9,613,040,056 |
| `flux2-klein-4b` | `Runpod/FLUX.2-klein-4B-mflux-4bit` | `7ee1b3aa8178a1240050490072196a57da2bf2a9` | 4,619,695,783 |
| `flux2-klein-4b-bf16` | `mflux-community/flux2-klein-4b-mflux-bf16` | `4d8e1bae8eb47c7766705de2cda7dabd6cc4ba67` | 15,975,684,703 |
| `hidream-o1-dev` | `mlx-community/HiDream-O1-Image-Dev-mlx-bf16` | `33c7a00bce8e3410304f83ec408a15a1eb6782df` | 17,649,873,024 |
| `qwen-image` | `mflux-community/qwen-image-mflux-q6` | `c628fe4392d963557c3013c2709e6d3b67bca79d` | 31,013,313,085 |
| `qwen-image-edit` | `OsaurusAI/Qwen-Image-Edit-mflux-q8` | `a458969f2a612433cf036bfc3d8d818ceba29fab` | 37,472,689,129 |
| `sd35-large-4bit` | `argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized` | `0f92f6c2a9f9e1abc6738209e87ac22b049a7d26` | 16,378,940,179 total with auxiliaries |
| `sdxl-base` | `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` | 6,941,201,645 |
| `z-image-turbo` | `filipstrand/Z-Image-Turbo-mflux-4bit` | `b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b` (the sole cached snapshot) | 5,907,434,624 |

The SD3.5 total includes `argmaxinc/stable-diffusion` at
`7b7a9946015fe6ae602464dfc026c19f6b6306f9` and the
`google/t5-v1_1-xxl` tokenizer at
`3db67ab1af984cf10548a73467f0e5bca2aaaeb2`. Z-Image does not yet carry
a product-side revision pin; the table records the exact snapshot exercised
by this run rather than presenting the floating alias as content-addressed.

## API matrix

Each row started with no other candidate model resident. It required ready
image health, a cold default request, a warm default request, cancellation at
the first observed denoise step, a successful recovery request in the same
server, the capability-specific endpoint or 409 rejection, a clean stop, and
a new process with both `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
Every successful response contained exactly one decodable, non-uniform RGB
PNG of the expected dimensions.

| Alias | Operation | Cold (s) | Warm (s) | Cancel | Recovery (s) | Offline (s) | Peak RSS (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bonsai-image-4b-2bit` | Generate | 14.18 | 11.36 | 1/4 | 11.44 | 12.31 | 3.73 |
| `flux-schnell` | Generate | 28.05 | 20.43 | 1/4 | 20.40 | 21.25 | 10.17 |
| `flux2-klein-4b` | Generate + edit | 10.27 | 9.03 | 1/4 | 9.02 | 9.81 | 7.14 |
| `flux2-klein-4b-bf16` | Generate + edit | 24.58 | 8.92 | 1/4 | 8.93 | 9.74 | 22.71 |
| `hidream-o1-dev` | Generate | 29.40 | 26.15 | 1/28 | 26.15 | 27.22 | 18.43 |
| `qwen-image` | Generate | 216.95 | 212.79 | 1/20 | 212.69 | 216.05 | 31.53 |
| `qwen-image-edit` | Edit | 168.54 | 166.17 | 1/20 | 165.73 | 167.05 | 38.07 |
| `sd35-large-4bit` | Generate | 176.24 | 162.96 | 1/28 | 162.84 | 162.64 | 7.83 |
| `sdxl-base` | Generate | 17.59 | 15.93 | 1/30 | 15.91 | 16.74 | 2.95 |
| `z-image-turbo` | Generate | 35.65 | 32.32 | 1/8 | 32.55 | 33.85 | 9.93 |

The Klein q4 and BF16 edit outputs, and the Qwen Image Edit output, were
visually inspected. Their warm-sunset instructions were followed while the
source cheetah and its line-art identity remained recognizable. The Qwen Image
generation was also inspected: the red-panda subject, astronaut suit, lunar
surface, and Earth were coherent and matched the prompt.

### Blocking defect found and fixed

The first exact-wheel SD3.5 request failed before denoising because the secure
asset resolver correctly followed a Hugging Face snapshot symlink to an
extensionless blob, while MLX attempted to infer its format from the resolved
filename. PR #3160 keeps the contained resolved path and supplies
`format="safetensors"` explicitly. The focused tests passed 22/22, the full
suite passed 22,714 tests, hosted checks passed, and the fixed wheel completed
the SD3.5 row above.

## Desktop matrix

The release-mode app embedded the same wheel in a sidecar with Python 3.12.13,
MLX 0.32.2, mflux 0.19.0, and Pillow 12.3.0. The packaged sidecar tarball was
482 MiB raw / 160 MiB compressed with SHA-256
`419c6ccda9f735f91bb392cec66395b82814269f96e4a76d1290a13c66f9a2bc`.
It contained the expected 173 Mach-O files and the assembled app passed strict
deep ad-hoc signature verification. Ad-hoc signing is appropriate only for
this local dogfood; it is not a distribution-signing claim.

- The packaged sidecar completed an offline, pinned Klein q4 1024×1024
  generation in 12.11 seconds and returned a valid 1,163,257-byte PNG.
- The release smoke verified packaged resources and rendered a 1440×900 main
  window that remained alive for the sustained launch check.
- The generation picker exposed exactly nine aliases: all generation-capable
  rows and no edit-only row. The edit picker exposed exactly the two Klein
  variants plus `qwen-image-edit`, and no generation-only row.
- Picker copy matched the CLI catalog for download size, RAM floor, default
  512² resolution, step count, and runtime family.
- Choosing landscape changed the displayed request dimensions from 512×512 to
  512×384. A real render produced a selected gallery thumbnail.
- A 1024×768 render showed determinate `1 / 4` progress. Cancel added no gallery
  item; a subsequent render in the same server succeeded and retained both the
  new and earlier results.
- Editing that recovered image changed the aurora to a warm sunset while
  preserving the blue fox. The edit became the selected first gallery item
  and remained the edit source.
- Command-Q stopped the isolated app and its owned sidecar within roughly
  0.75 seconds; port 58150 had no listener afterward.

## Non-blocking follow-ups

- Pin `z-image-turbo` in the product revision table so future release runs do
  not need to derive identity from the local cache.
- The Qwen generation and edit servers emitted a Python resource-tracker
  warning about one semaphore cleaned during otherwise successful shutdown.
  No server PID remained, but the lifecycle should be made warning-clean.
- `release-smoke.sh` terminates the app process for cleanup. One run left its
  in-flight `models --cached --json` child behind; normal Command-Q in the
  isolated product walk stopped both app and sidecar cleanly. Harden the smoke
  cleanup to reap its catalog subprocess tree.
- This host is an M3 Ultra. The q4/BF16 rows above are not evidence for M1 or
  M2 performance or memory behavior; those machines require separate receipts.

## Release boundary

No tag, GitHub Release, notarized artifact, deployment, or updater mutation
was created. A human-authorized release must rebuild from its final source SHA
and rerun the exact-artifact acceptance contract in
`image-release-dogfood-matrix.md`.
