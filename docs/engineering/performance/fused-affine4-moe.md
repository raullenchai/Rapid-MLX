# Selected affine-4 MoE decode

`RAPID_MLX_QWEN35_AFFINE4=1` enables selected-expert gate/up/SwiGLU,
down/weighted reduction, and parallel routing for qualified Qwen blocks.
The switch is opt-in; unset or `0` disables it. A false value also disables
execution on an already enrolled instance. Cross-generation throughput did
not meet the broad 10% improvement requirement for enabling it by default.

The loaders enroll each compatible block after a deterministic numerical
probe. The path requires 256 experts, top-8 normalized routing, BF16 input,
hidden size 2048, expert intermediate size 512, affine 4-bit expert banks,
group size 64, and BF16 quantization scales/biases. The shared gate can use
4-bit or 8-bit affine weights. Unsupported families, layouts, sharding,
prefill, wider decode batches, and multi-token speculative verification
use the existing implementation. The MTP backbone may use the kernel for
singleton calls; enabling the switch does not change speculative policy.

The full banks are explicit Metal inputs. A leading logical view is
insufficient to guarantee that all expert addresses remain valid under
future array materialization or buffer reuse, so production uses full banks.
MLX retains those inputs in the pending graph. Class closures capture only
the original method; block enrollment and dequantized shared-gate weights
belong to each model instance. GPU scratch outputs belong to each call.

The changed expert reduction order is approximate. On M5 Max, 50,000 fixed
teacher-forced next-token positions from public-domain prose, in independent
1024-position contexts, measured perplexity 4.88267571 (stock) versus
4.88472247 (fused): +0.0419%, within the 0.5% relative gate. Top-1 agreement
was 93.556%. Deterministic arithmetic, JSON, and Python-function prompts
passed on both paths. These results cover this checkpoint and corpus;
they do not establish exact-output parity or general task quality.

Checkpoint: `mlx-community/Qwen3.6-35B-A3B-4bit` at
`38740b847e4cb78f352aba30aa41c76e08e6eb46`, MLX 0.32.4.
Corpus: the fixed downloaded UTF-8 excerpt of
[public-domain prose](https://www.gutenberg.org/ebooks/2701.txt.utf-8), SHA-256
`4caa34e6e54399f212ff26f09e8bf48f8801e5a614f4714e84f6dff8d96e84da`.
Scored token stream SHA-256:
`577ad617d469d4db5f88e4ae974acd99d5d4f63615b536d3a9987035f17f048d`.
The exact excerpt must be retained to reproduce the corpus hash.

Run `scripts/qualify_qwen35_affine4.py --model <snapshot> --corpus <text>
--output <directory>` on an idle GPU. It scores both paths one token at a
time, ensuring that the B1 kernel is exercised, and stores NLLs, top-1 IDs,
functional outputs, hashes, and the qualification verdict.

## Serving qualification

Three valid fresh-process repetitions per arm, 564 prompt / 256 output tokens,
greedy, thinking disabled, idle-gated client timing, MLX 0.32.4.
M5 Max: 36 GiB/macOS 27.0.1; M4 Pro: 48 GiB/26.5.1;
M3 Ultra: 256 GiB/26.5.2. Stock means this kernel disabled.
Default MTP uses default policy parameters with a cached sidecar path.

| Chip | Mode | Stock tok/s (min–max) | Fused tok/s (min–max) | Change |
|---|---|---:|---:|---:|
| M5 Max | No spec | 151.05 (150.75–151.14) | 167.80 (167.65–167.87) | +11.09% |
| M5 Max | Default MTP | 161.70 (161.54–161.76) | 137.75 (137.69–138.09) | -14.81% |
| M4 Pro | No spec | 97.47 (97.32–97.53) | 97.53 (97.47–97.56) | +0.06% |
| M4 Pro | Default MTP | 94.27 (82.90–94.32) | 82.60 (82.58–82.72) | -12.38% |
| M3 Ultra | No spec | 136.13 (131.62–137.47) | 151.09 (146.34–158.15) | +10.99% |
| M3 Ultra | Default MTP | 148.62 (145.23–149.03) | 127.88 (127.12–129.30) | -13.96% |

MTP logs change from the default to steep verification-cost class on M5/M3.
The specialization accelerates singleton calls while wider verification falls
back. This interaction requires separate MTP qualification and blocks a
default-on recommendation. M4 no-spec is neutral.

M4 MTP confirmation: Stock 82.92 (82.82–82.93), fused 82.56 (82.33–82.60) tok/s; -0.43%, three additional valid repetitions per arm.

M3 no-spec confirmation after the initial wider spread: Stock 135.72 (135.71–137.78), fused 157.68 (156.98–158.70) tok/s; +16.18%, three additional valid repetitions per arm.

Dense q9 no-spec medians (stock/fused tok/s): M5 80.54/80.50,
M4 49.27/49.27, M3 111.52/111.53, with identical output digests on each chip.
Default-MTP dense controls have overlapping timing ranges, especially on M3;
they do not establish a tight sub-percent performance bound.
