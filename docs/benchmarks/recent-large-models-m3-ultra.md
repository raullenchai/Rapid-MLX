# Recent large models on M3 Ultra

This note is the evidence behind the concise large-model table in the project
README. It records a same-machine 0.13.3-to-0.13.4-candidate comparison for
Qwen3.8-27B and fresh context curves for Qwen3.8-Flash-Next and
GLM-5.3-Flash.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio (`Mac15,14`) |
| Chip | Apple M3 Ultra, 28 CPU cores |
| Unified memory | 256 GB |
| macOS | 26.5.2 (25F84) |
| Architecture | arm64 |
| Serving shape | Batch size 1; one model resident |
| Quantization | The Rapid-MLX 4-bit alias named in each row |
| Python | 3.12.13 |
| MLX / MLX-LM / MLX-VLM | 0.32.2 / 0.31.3 / 0.6.17 |
| 0.13.3 reference | `6f3f65b92eed6906421b5761686c0a2cd0923aa3` |
| 0.13.4 candidate measured | `615a8c5cd17b40db8d49e17d93c96f9094f23221` |

All rates are medians of three measured requests after the server reported
ready. TTFT is measured to the first visible streamed content, reasoning, or
tool delta. Prefill is server-reported prompt tokens divided by TTFT. Decode
excludes TTFT. The prefix cache is cleared before every timed request. MLX
active memory is allocator-active unified memory, not process RSS; RSS
materially undercounts Metal allocations. The three-run median also prevents
one first-request Metal compilation outlier from becoming the headline.

The Qwen context curves use temperature zero, thinking disabled, a cold prefix
cache for every request, and 256 requested decode tokens. The harness targets
128, 2,048, 8,192, and 32,768 prompt tokens; after applying the chat template,
the server reports 92, 2,012, 8,156, and 32,732 tokens.

## At a glance

| Model | Model shape | Measured workload | Median TTFT | Median prefill | Median decode | MLX memory observed through 32K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3.8-27b-4bit` | 27B dense | 8,156 → 256 | 25.058s | 325.5 tok/s | 38.97 tok/s | 22.1 GB active / 23.4 GB peak |
| `qwen3.8-flash-next-4bit` | 180B total / 6B active | 8,156 → 256 | 9.397s | 867.9 tok/s | 23.12 tok/s | 102.8 GB active / 148.1 GB peak |
| `glm5.3-flash-4bit` | 320B total / 18B active | 8,156 → 256 | 22.779s | 359.6 tok/s | 27.86 tok/s | 180.6 GB active / 195.6 GB peak |

The Flash-Next parameter total is 125B language-model parameters plus a 51B
n-gram embedding and 4B MTP head; 6B language-model parameters are active per
token. The GLM shape is 320B total and 18B active. Those figures describe the
upstream architectures; throughput and memory in this document are Rapid-MLX
measurements. See the official model cards for the
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), and
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) architecture
descriptions.

Flash-Next's fixed-K=1 MTP and prompt-lookup qualifications are separate
same-machine experiments. The table above reports the public alias's default
autoregressive path so it does not mix workload-specific acceleration with the
common path.

## Qwen3.8-27B context curve

Artifact: `rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX` at
`aa985c29ff5b334cbfdcbbc787d47e66e9d9e456`. Both releases used the public
`qwen3.8-27b-4bit` alias with no speculative-decoding flag. In 0.13.4 the
verified artifact automatically selects its compatible text lane and adaptive
MTP path; 0.13.3 served the ordinary path.

| Target (reported) prompt tokens | 0.13.3 TTFT | 0.13.4 TTFT | 0.13.3 decode | 0.13.4 decode | Decode speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.541s | 0.396s | 30.83 tok/s | 34.54 tok/s | 1.12× |
| 2,048 (2,012) | 6.185s | 5.937s | 28.99 tok/s | 44.44 tok/s | 1.53× |
| 8,192 (8,156) | 25.037s | 25.058s | 24.67 tok/s | 38.97 tok/s | 1.58× |
| 32,768 (32,732) | 110.251s | 108.432s | 16.58 tok/s | 39.12 tok/s | 2.36× |

The 0.13.4 prefill medians were 232.2, 338.9, 325.5, and 301.9 tok/s at
128, 2K, 8K, and 32K. Across the complete run, MTP recorded 983 accepted
drafts from 1,496 proposals (65.71%). Every draft is verified by the target;
the ratio is an efficiency signal, not a substitute for correctness testing.
The process reached about 22.1 GB MLX active memory during 32K decode and
reported a 23.4 GB allocator peak; after the request cache was released it
returned to 15.4 GB active.

The model-recommendation qualification uses a separate complete-process-tree
memory boundary. Do not compare its RSS number directly with the MLX allocator
figures above.

## Qwen3.8-Flash-Next context curve

Artifact: `rapid-mlx/Qwen3.8-Flash-Next-4bit` at
`dcf657e4acda2aae72da99cde65b6c491cd96998`. The rows below are the default
autoregressive public-alias path on the 0.13.4 candidate.

| Target (reported) prompt tokens | Median TTFT | Median prefill | Median decode |
| ---: | ---: | ---: | ---: |
| 128 (92) | 0.351s | 262.0 tok/s | 25.46 tok/s |
| 2,048 (2,012) | 2.299s | 875.1 tok/s | 24.07 tok/s |
| 8,192 (8,156) | 9.397s | 867.9 tok/s | 23.12 tok/s |
| 32,768 (32,732) | 45.731s | 715.8 tok/s | 21.46 tok/s |

MLX active memory was 102.8 GB after the sweep. The process also reported a
148.1 GB allocator peak inherited from model loading; it is not the steady
active footprint. The corresponding 0.13.3 medians were 25.40, 24.07, 23.15,
and 21.45 tok/s: the default path is effectively unchanged, so no 0.13.4
speedup is claimed for this row.

The separate fixed-K=1 MTP qualification measured decode at 34.85, 33.53,
32.20, and 28.82 tok/s at the same four prompt lengths: a 36–42% improvement
over its exact-run ordinary-decode baseline. All 45 functional outcomes
matched ordinary decode, with a 76.41% aggregate proposal acceptance ratio.
MTP added as much as 6.6 GB of active memory and increased 2K–32K TTFT by
5–8%, so it remains an explicit workload-dependent choice.

The model's quantized weights occupy about 99 GB before cache and allocator
headroom. A 192 GB Mac is the practical recommended tier. A 128 GB Mac is
tight and was not physically tested.

Full Flash-Next methodology and correctness evidence:

- [ordinary decode and QSA prefill](qwen38-flash-next-m3-ultra.md)
- [native MTP qualification](qwen38-flash-next-mtp-m3-ultra.md)

## GLM-5.3-Flash context curve

Artifact: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` at
`06d6c7530e8290e20fabdc37a825ce07bdfc490c`. The server ran with thinking
disabled and temperature zero. Speculative decoding remains disabled for this
alias because its separate qualification did not beat ordinary decoding.

| Target (reported) prompt tokens | Median TTFT | Median prefill | Median decode |
| ---: | ---: | ---: | ---: |
| 128 (128) | 0.819s | 156.3 tok/s | 32.51 tok/s |
| 2,048 (2,048) | 5.493s | 372.8 tok/s | 28.47 tok/s |
| 8,192 (8,192) | 22.779s | 359.6 tok/s | 27.86 tok/s |
| 32,768 (32,768) | 118.341s | 276.9 tok/s | 27.31 tok/s |

The first 128-token request paid a one-time Metal compilation cost; the
three-run median shown above reflects the other two consistent requests. After
the 32K sweep, the target process reported 180.6 GB MLX active memory and a
195.6 GB peak. The earlier 165.4 GB short-prompt measurement therefore must
not be used as a 32K sizing claim. The alias retains a 192 GB catalog floor for
shorter contexts, but this measured 32K workload needs the headroom of a 256 GB
Mac; it was not physically qualified on a 192 GB machine.

Resolve the exact measured checkpoint revision first, then serve that immutable
local snapshot. This avoids accidentally benchmarking a newer cached revision:

```bash
MODEL_DIR="$(
  python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="Vontra/GLM-5.3-Flash-MLX-4bit-MTP",
    revision="06d6c7530e8290e20fabdc37a825ce07bdfc490c",
))
PY
)"

HF_HUB_OFFLINE=1 rapid-mlx serve \
  "$MODEL_DIR" --served-model-name glm5.3-flash-4bit \
  --host 127.0.0.1 --port 8465 --no-thinking
```

The four context curves were recorded with the repository harness. Replace the
model, immutable tokenizer snapshot, PID, label, and output path for each
server process:

```bash
python .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8465/v1 \
  --model MODEL_ALIAS \
  --tokenizer-path IMMUTABLE_SNAPSHOT \
  --server-pid SERVER_PID \
  --label RESULT_LABEL \
  --rapid-sha EXACT_RAPID_SHA \
  --artifact-revision EXACT_ARTIFACT_REVISION \
  --output OUTPUT.json
```

The harness builds deterministic prompts at the four target lengths, requests
256 output tokens three times per length, parses the streamed final usage
event, and samples the complete process tree. `/v1/status` supplies the MLX
allocator readings used here. Do not run another model server concurrently.

The checkpoint contains a native MTP head, but the qualification experiment
did not produce a speedup: 32.00 tok/s ordinary decode versus 31.65 tok/s with
MTP for the sustained 512-token comparison, despite 72.97% acceptance and
5.71 GB additional active memory. GLM MTP is therefore disabled for this
alias; neither that no-go result nor an unqualified acceleration mode is used
in the README headline.
