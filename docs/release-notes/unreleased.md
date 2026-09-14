<!-- Scratch space for the next release's notes. Append as you land work; in the
     version-bump PR, `git mv` this to vX.Y.Z.md and recreate this file empty.
     Whole-line HTML comments like this one are stripped before publishing.
     See README.md in this directory for what good notes look like. -->

Rapid-MLX now owns and has qualified the critical GLM-5.3 speculative-decoding
transaction, while keeping the released product fail-closed until its upstream
model/cache protocol is available in a tagged dependency.

## Highlights

**GLM-5.3 Flash acceleration is qualified without trading away its answers.**
On an M3 Ultra, the immutable Q4 target and matching 4-bit MTP sidecar raised
median category throughput from 26.58 to 35.54 tok/s (+33.7%); the median of
the six paired task speedups was +34.1%. Two repeated runs preserved all 12
reasoning traces and final answers byte for byte across coding, knowledge,
math, instruction following, creative constraints, and long-document
retrieval.

| Task | AR tok/s | MTP tok/s | Change |
| --- | ---: | ---: | ---: |
| Coding | 27.55 | 38.70 | +40.5% |
| Knowledge | 26.61 | 36.63 | +37.7% |
| Math | 26.56 | 34.68 | +30.6% |
| Instruction following | 25.69 | 35.71 | +39.0% |
| Creative constraints | 27.52 | 35.37 | +28.5% |
| Long-document retrieval | 10.62 | 11.74 | +10.6% |

Rapid now owns the speculative cache transaction, exposes the exact pair to
CLI and Desktop, keeps sampled requests on ordinary decoding, and fixes GLM-5
thought traces leaking into visible streamed answers. The currently packaged
mlx-vlm release does not yet expose the required cache protocol, so existing
installs safely remain on autoregressive decoding until a tagged compatible
dependency is shipped. ([#3462](https://github.com/raullenchai/Rapid-MLX/pull/3462),
[#3467](https://github.com/raullenchai/Rapid-MLX/pull/3467))
