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
| 0.13.4 Qwen3.8-27B candidate measured | `ede7158cc59b0b7002f05d8b34e985d9c4ae5206` |
| 0.13.4 Flash-Next / GLM candidate measured | `615a8c5cd17b40db8d49e17d93c96f9094f23221` |

All rates are medians of three measured requests after the server reported
ready. TTFT is measured to the first visible streamed content, reasoning, or
tool delta. Prefill is server-reported prompt tokens divided by TTFT. Decode
excludes TTFT and the first token already delivered at the TTFT boundary: the
reported rate is `(completion_tokens - 1) / (total_time - TTFT)`. The prefix
cache is cleared before every timed request. MLX active memory is
allocator-active unified memory, not process RSS; RSS
materially undercounts Metal allocations. The three-run median also prevents
one first-request Metal compilation outlier from becoming the headline.

The Qwen context curves use temperature zero, thinking disabled, a cold prefix
cache for every request, and 256 requested decode tokens. The harness targets
128, 2,048, 8,192, and 32,768 prompt tokens; after applying the chat template,
the server reports 92, 2,012, 8,156, and 32,732 tokens.

### Completion-length evidence

The harness records actual completion tokens and finish reason for every run.
The table below preserves those fields rather than assuming every 256-token
request reached the cap:

The committed [run-level CSV](results/2026-09-02-large-model-runs.csv) contains
all 60 timed rows, and its [metadata/status record](results/2026-09-02-large-model-metadata.json)
pins the Rapid/model revisions, source-artifact hashes, MLX allocator readings,
and MTP counters used below. Reviewers can recompute every median directly from
those two files.

| Result artifact | 128 | 2K | 8K | 32K |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B 0.13.3 | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` |
| Qwen3.8-27B 0.13.4 | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` |
| Flash-Next 0.13.3 | 232/232/232, `stop` | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` |
| Flash-Next 0.13.4 | 232/232/232, `stop` | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` |
| GLM-5.3-Flash 0.13.4 | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` | 256/256/256, `length` |

Flash-Next therefore has a 232-token decode denominator in its 128-target row;
the model emitted EOS consistently in both versions. Every version-comparison
row for Qwen3.8-27B and every 8K headline row completed the requested 256
tokens. SHA-256 fingerprints of the five source JSON artifacts, in table order,
are:

- `b29d4f75d275e7a714af3255e072e6f78a5046cb309eff85cf7a14e53bc052d5`
- `89b70e9746a9aaa4d7a1ba7b1d8bf80f19c72edc7e5c38cb8deb07be3ac010aa`
- `2de53fe3e8236b1fe4135f22811040d0e4a14edc4772bfaf46662d6ef332e925`
- `0b2245c6159a8b24714ff4637bc31f40c521422149ba2b496fa5fc10523368d6`
- `95cced14b5f73ffb23729e7bb89a4ea9c6354c9d19212333d32601963cf57718`

## At a glance

| Model | Model shape | Measured workload | Median TTFT | Median prefill | Median decode | MLX memory observed through 32K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3.8-27b-4bit` | 27B dense | 8,156 → 256 | 24.656s | 330.8 tok/s | 43.38 tok/s | 26.7 GB active / 27.1 GB peak |
| `qwen3.8-flash-next-4bit` | 180B total / 6B active | 8,156 → 256 | 9.397s | 867.9 tok/s | 23.03 tok/s | 102.8 GB active / 148.1 GB peak |
| `glm5.3-flash-4bit` | 320B total / 18B active | 8,192 → 256 | 22.779s | 359.6 tok/s | 27.75 tok/s | 180.6 GB active / 195.6 GB peak |

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
| 128 (92) | 0.541s | 0.511s | 30.71 tok/s | 43.93 tok/s | 1.43× |
| 2,048 (2,012) | 6.185s | 6.074s | 28.88 tok/s | 44.69 tok/s | 1.55× |
| 8,192 (8,156) | 25.037s | 24.656s | 24.58 tok/s | 43.38 tok/s | 1.77× |
| 32,768 (32,732) | 110.251s | 109.120s | 16.52 tok/s | 38.66 tok/s | 2.34× |

The 0.13.4 prefill medians were 180.0, 331.3, 330.8, and 300.0 tok/s at
128, 2K, 8K, and 32K. Across the complete run, MTP recorded 1,296 accepted
drafts from 2,033 proposals (63.75%). Every draft is verified by the target;
the ratio is an efficiency signal, not a substitute for correctness testing.
The process reached about 26.7 GB MLX active memory during 32K decode and
reported a 27.1 GB allocator peak. Immediately after the sweep, `/v1/status`
reported 20.0 GB active plus 6.5 GB allocator cache.

The model-recommendation qualification uses a separate complete-process-tree
memory boundary. Do not compare its RSS number directly with the MLX allocator
figures above. Its 32 GB recommendation is scoped to the approximately 8K
workload, where that qualification measured about 20 GB for the complete
process tree. The automatic-MTP 32K path was not physically qualified on a
32 GB Mac; this Studio sweep reached a 27.1 GB MLX allocator peak before
non-MLX process and macOS memory.

The version comparison used the same no-flag user command from fresh source
trees at the two exact Rapid-MLX commits in the environment table. Before
serving, the benchmark resolved the pinned revision and proved that the
offline alias selected that exact snapshot:

```bash
QWEN_SNAPSHOT="$(
  "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download

repo = "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"
revision = "aa985c29ff5b334cbfdcbbc787d47e66e9d9e456"
pinned = snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
default = snapshot_download(repo_id=repo, local_files_only=True)
if pinned != default:
    raise SystemExit(f"offline alias is not pinned: {default} != {pinned}")
print(pinned)
PY
)"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$SOURCE_TREE" \
  "$VENV/bin/python" -m vllm_mlx.cli serve qwen3.8-27b-4bit \
  --host 127.0.0.1 --port 8465 --no-thinking
```

On the 0.13.4 candidate, the alias's `mtp_draft_model` plus its `verified`
continuous-MTP qualification make that command select adaptive MTP. The
profile's `supports_spec_decode: false` disables the generic hybrid-model
suffix/draft verifier; it does not disable the separately admitted native-MTP
path. The same command on 0.13.3 selected ordinary decode. No speculative
flag was added to either benchmark process.

## Qwen3.8-Flash-Next context curve

Artifact: `rapid-mlx/Qwen3.8-Flash-Next-4bit` at
`dcf657e4acda2aae72da99cde65b6c491cd96998`. The rows below are the default
autoregressive public-alias path on the 0.13.4 candidate.

| Target (reported) prompt tokens | Median TTFT | Median prefill | Median decode |
| ---: | ---: | ---: | ---: |
| 128 (92) | 0.351s | 262.0 tok/s | 25.35 tok/s |
| 2,048 (2,012) | 2.299s | 875.1 tok/s | 23.98 tok/s |
| 8,192 (8,156) | 9.397s | 867.9 tok/s | 23.03 tok/s |
| 32,768 (32,732) | 45.731s | 715.8 tok/s | 21.37 tok/s |

MLX active memory was 102.8 GB after the sweep. The process also reported a
148.1 GB allocator peak inherited from model loading; it is not the steady
active footprint. The corresponding 0.13.3 medians were 25.29, 23.98, 23.06,
and 21.37 tok/s: the default path is effectively unchanged, so no 0.13.4
speedup is claimed for this row. The 128 row measures the 232 tokens emitted
before the model's consistent EOS; the other rows each emitted 256 tokens.

The separate fixed-K=1 MTP qualification measured decode at 34.71, 33.40,
32.07, and 28.71 tok/s at the same four prompt lengths: a 36–42% improvement
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
| 128 (128) | 0.819s | 156.3 tok/s | 32.38 tok/s |
| 2,048 (2,048) | 5.493s | 372.8 tok/s | 28.36 tok/s |
| 8,192 (8,192) | 22.779s | 359.6 tok/s | 27.75 tok/s |
| 32,768 (32,768) | 118.341s | 276.9 tok/s | 27.20 tok/s |

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

# Capture MLX allocator telemetry immediately after the completed sweep,
# before stopping the server. This is separate from benchmark.py's RSS sampler.
curl --silent http://127.0.0.1:8465/v1/status \
  | tee OUTPUT.status.json \
  | jq '{model, metal}'
```

The harness builds deterministic prompts at the four target lengths, requests
256 output tokens three times per length, parses the streamed final usage
event, and samples the complete process-tree RSS. The separate status command
captures the MLX allocator readings used here; for the GLM sweep it returned
`active_memory_gb: 180.6` and `peak_memory_gb: 195.58`. Do not run another
model server concurrently.

The checkpoint contains a native MTP head, but the qualification experiment
did not produce a speedup: 31.94 tok/s ordinary decode versus 31.59 tok/s with
MTP for the sustained 512-token comparison, despite 72.97% acceptance and
5.71 GB additional active memory. GLM MTP is therefore disabled for this
alias; neither that no-go result nor an unqualified acceleration mode is used
in the README headline.
