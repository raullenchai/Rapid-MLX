# LFM3-VL native serving qualification

**Status: numerical gate accepted; serving qualification passed for code commit `034a76324`.** The following describes the earlier Run B2, whose original numerical gate was superseded by the owner’s Run C decision. On a leased Mac Studio M5 Max (36 GB, macOS 27.0.1), Python 3.12 and MLX 0.32.4, a fresh base installation had the transitively installed optional vision package removed. A temporary HOME linked its Hugging Face cache to the machine's normal cache. The pinned 8-bit checkpoint was served on loopback port 18950 with telemetry disabled. The alias launch and probes used a working tree containing the integration changes in this branch; they were not run against a final committed head.

A 512x256 test image containing a red square, blue circle and the text HELLO received a correct description. Two ordered images were described correctly, as were two concurrently submitted image requests; the hybrid media lane serialized execution. Chinese text chat returned a sensible greeting, and an English text request returned a response. Streaming returned content deltas and `[DONE]`. A tool request returned a `get_weather` call with `city: Paris`. Invalid image data returned HTTP 400. Video initially returned 500; after the route fix the same probe returned HTTP 400 with `Video is not supported by LFM3-VL`.

One warm 256x256 image streaming sample measured 0.197 s from request start to first content delta; 28 completion tokens over 0.194 s from first delta to stream completion (approximately 139 subsequent tokens/s). This is one sample, not a benchmark distribution. At shutdown an `atexit` dump of loaded module names found no `mlx_vlm` modules; the package was absent from this environment.

The numerical comparison against the pinned bf16 reference did not meet the vision/projector cosine threshold: 0.999840706 and 0.999600738 versus 0.9999. A single 512-token square-image logits comparison passed its thresholds, and the same 8-bit fixture matched the pinned independent implementation exactly. Those isolated results do not satisfy the full fixture and cache gates. Investigate the first below-threshold vision layer (layer 11, zero-based) and rerun all numerical and end-to-end checks on the exact final commit before release. The remote copies and throwaway environments are removed after this run; model cache entries are retained.

## Run B3 follow-up

The revised float32 cosine comparison passed its numerical cosine thresholds on a square fixture: tower 0.999994764 and projector 0.999991315. Maximum absolute error divided by maximum absolute reference value was 0.014264 and 0.005748, respectively; the owner accepted the numerical qualification. The bf16 teacher-forced comparison still failed the 99.5% top-1 requirement on portrait and odd fixtures, each 508/512 (99.21875%), although mean KL passed (0.000163042 and 0.000325303). Square, landscape, two-image, and a 512-token continuation passed. The 8-bit implementation matched the pinned comparison exactly on tower, projector, and logits for all fixtures. Detailed per-layer measurements, fixture values, and saved comparison scripts are recorded in `/private/tmp/rapid-mlx-lfm3vl/correctness.md` and `/private/tmp/rapid-mlx-lfm3vl/verify/`. These are local qualification artifacts and must be retained outside the repository. This run does not qualify a final commit; do not merge on the basis of the earlier serving probes.

## Run C: replacement bf16 gate

The owner accepted mean KL ≤ 0.001 and a reference top-two margin ≤ 0.25 at every top-1 disagreement, including tied maxima. All six fixture KL values reported in the numerical record pass; every disagreement passes the margin gate. Positions are zero-based in the 512-token continuation. Square and long use the same input and are not independent. The reference logits were regenerated with the saved fixture scripts on the leased M5 Max and checked with the saved diagnostics script.

| Fixture | Position | Reference token | Native token | Reference top-2 margin |
| --- | ---: | ---: | ---: | ---: |
| square | 9 | 395 | 22312 | 0 |
| square | 70 | 229 | 124900 | 0 |
| portrait | 12 | 4145 | 6424 | 0.125 |
| portrait | 58 | 124900 | 229 | 0.125 |
| portrait | 82 | 124900 | 229 | 0.125 |
| portrait | 94 | 229 | 124900 | 0 |
| landscape | 106 | 229 | 124900 | 0 |
| landscape | 118 | 229 | 124900 | 0 |
| odd | 3 | 59061 | 18970 | 0.0625 |
| odd | 82 | 229 | 124900 | 0 |
| odd | 94 | 229 | 124900 | 0 |
| odd | 142 | 229 | 124900 | 0 |
| long | 9 | 395 | 22312 | 0 |
| long | 70 | 229 | 124900 | 0 |

## Run C serving qualification

On the leased M5 Max (36 GB, macOS 27.0.1), Python 3.12 and MLX 0.32.4, a fresh installation of code commit `3267f9957` served the pinned 8-bit alias after removing the transitively installed optional vision runtime. The image with printed HELLO, two ordered images, two concurrent image requests, English and Chinese text, SSE with `[DONE]`, and a Paris weather tool call all returned expected results. Invalid image data and video each returned HTTP 400. Hybrid requests use the serialized media lane.

One warm image stream had 0.123 s request-to-first-content latency and 37 completion tokens over 0.238 s from first content to stream end (approximately 152 subsequent tokens/s). This is a single sample. The shutdown module audit was empty, and a separate module discovery check returned `None` for `mlx_vlm`. The qualifying environment and remote checkout were removed after the run.

The same probe set was repeated after the dispatch review fixes on code commit `034a76324`, in another fresh Python 3.12 environment without the optional runtime. All probes again passed; invalid image and video returned 400, and the shutdown module list was empty. One warm image stream measured 0.145 s to first content and 37 completion tokens in 0.245 s from first content to completion (approximately 147 subsequent tokens/s). This is one sample.
