# Personal Intelligence top-model qualification

Date: 2026-09-15

## Product rule

Personal Intelligence is a model-local harness. It always runs on the model
already selected in Chat; Rapid must never start a second “agent model” behind
the user's back. A model/build is advertised in `/v1/models` and enabled in the
GUI only after that exact public identity, backing artifact, tool parser, and
harness profile have a passing receipt. Family resemblance is not evidence.

The support target below is the union of the two September 15 usage tables,
plus MiniCPM5-2B, the original 8/16 GB target. Quantizations are separate
qualification units because they can change tool-call reliability.

## Acceptance suite

Run the shipped suite against a live server, not Chat Completions in isolation:

```bash
python scripts/qualify_personal_intelligence.py MODEL \
  --base-url http://127.0.0.1:8000 \
  --seeds 11,22,33 \
  --hardware 'Mac model, chip, memory' \
  --os 'macOS version and build' \
  --runtime 'rapid-mlx, MLX, and mlx-lm versions' \
  --source-revision 'exact Git commit' \
  --server-command 'complete launch command and flags' \
  --output reports/benchmarks/personal-intelligence-MODEL.json
```

The five cases cover local-context recall, restraint when no tool is needed,
weather, deterministic search-to-browse, and prompt-injection resistance. A
build qualifies only at 15/15, with bounded tool calls and no hard run failure.
The script requires hardware, OS/runtime, source revision, and complete server
launch metadata; every committed JSON receipt is independently reproducible.

## Telemetry-driven build matrix

| Model | Build(s) in scope | Parser | Status |
|---|---|---|---|
| MiniCPM5-2B | `minicpm5-2b-4bit` / `openbmb/MiniCPM5-2B-MLX`; separately `mlx-community/MiniCPM5-2B-8bit` | `minicpm` | Q4 passed 15/15 locally; Q8 retains its earlier receipt |
| Qwen3.5-4B | `qwen3.5-4b-4bit`; `qwen3.5-4b-8bit` | `hermes` | Q4 passed 15/15 locally; Q8 pending exact-build run |
| Qwen3.5-9B | `qwen3.5-9b-4bit`; `qwen3.5-9b-8bit` | `hermes` | Q4 passed 15/15 on Mac Studio; Q8 pending |
| Qwen3.8-27B | `qwen3.8-27b-4bit` (Rapid MTP); raw `mlx-community/Qwen3.8-27B-4bit`; `qwen3.8-27b-mixed-3.5bpw`; `qwen3.8-27b-4bit-fp16` | `qwen3_coder_xml` | pending; M2 32 GB attempt was stopped before load when the runtime projected 140% memory use. No unsafe receipt was accepted |
| Qwen3.6-35B-A3B | `qwen3.6-35b-4bit`; `qwen3.6-35b-8bit` | `qwen3_coder_xml` | 8-bit passed 15/15 on Mac Studio; Q4 pending |
| Bonsai 27B | `bonsai-27b-2bit` | `hermes` | pending |
| Qwen3-Coder 30B | `qwen3-coder-30b-4bit` | `hermes` | pending |
| Ling 3.0 Tiny | `ling-3.0-tiny-4bit` | `glm47` | pending |
| Qwen3.6-27B | `qwen3.6-27b-4bit` | `qwen3_coder_xml` | pending (the cached Studio 8-bit build is not a substitute) |
| GPT-OSS 20B | `gpt-oss-20b` (`mlx-community/gpt-oss-20b-MXFP4-Q8`) | `harmony` | pending |
| LFM2.5 1.2B | `lfm2.5-1b-4bit` | `lfm` | passed 15/15 locally after deterministic search/browse staging |

The missing fourth Qwen3.8 build in the aggregate table is treated as the
Rapid FP16-MTP build above, matching the four shipped 27B artifact identities.
If telemetry identifies a different artifact, it must replace that row and get
its own receipt before exposure.

## Harness changes from physical dogfood

- Desktop now sends the selected model the user's custom instructions, memory,
  and recent completed turns as bounded transient context.
- The server owns the schemas and risk labels for `web_search`, `browse`, and
  `weather`; Desktop sends names only and remains the executor.
- Intent routing hides irrelevant tools. Ordinary recall and writing expose no
  live-data tool, weather exposes only weather, and web work starts with search.
- Search-to-browse is deterministic harness plumbing: Rapid extracts ranked
  HTTP(S) result lines and asks Desktop to browse them. It follows explicit
  pagination and reads up to three ranked pages for comparison tasks. Explicit
  no-network requests suppress the route. This removes fragile model-authored
  argument rounds without granting server-side execution.
- Tool arguments are checked against required and unknown schema fields before
  native dispatch. Invalid calls return a recoverable observation and do not
  reach the tool.

## Evidence

- `reports/benchmarks/personal-intelligence-minicpm5-2b-4bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.5-4b-4bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.5-9b-4bit.json`
- `reports/benchmarks/personal-intelligence-qwen3.6-35b-8bit.json`
- `reports/benchmarks/personal-intelligence-lfm2.5-1b-4bit.json`
- Further receipts are added only after the exact live build reaches 15/15.
