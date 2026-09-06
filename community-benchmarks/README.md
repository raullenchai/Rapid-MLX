# Community-submitted performance database

Real numbers from real users' Apple Silicon Macs running Rapid-MLX. Single-vendor benchmarks can only cover the hardware the vendor has — the headline table in the repo `README.md` was measured on an M3 Ultra 256 GB. This directory is how everyone else fills in their own row.

There are two flows. Both are consent-gated, both talk HTTPS to rapidmlx.com, and neither needs a git checkout, a GitHub account, or `gh`.

| | `rapid-mlx benchmark …` (0.13.4+) | `rapid-mlx bench <alias> --submit` (legacy) |
|---|---|---|
| What it measures | Registered protocols for text, image and video models | Text only |
| Record shape | Atomic `BenchmarkRun` (`proto/community-benchmark/v1`) | `schema.json` in this directory |
| Runs locally first | Yes — every run is archived under `~/.rapid-mlx/benchmarks/`; no benchmark data leaves the Mac until you run `share` (the model itself may be downloaded from Hugging Face during `run`, as with any other load) | No — the run and the submission are one command |
| Upload endpoint | `POST https://rapidmlx.com/api/benchmarks/atomic` | `POST https://rapidmlx.com/api/benchmarks` |
| Where it shows up | "Community Benchmark beta" on <https://rapidmlx.com/leaderboard> and, when the server assigns an identity, your contributor page | The comparable board on the same page |

The board's history (the `submissions/` directory here and `aggregated.json`) predates both HTTP flows: those rows arrived as pull requests. That path is gone; see [History](#history).

## Local-first flow: `rapid-mlx benchmark`

```console
$ rapid-mlx benchmark catalog                 # models with a registered protocol, with a fit column for this Mac
$ rapid-mlx benchmark plan qwen3.5-9b-4bit    # the exact workload, before anything runs
$ rapid-mlx benchmark run qwen3.5-9b-4bit     # measure; the result is saved locally, nothing is uploaded
Saved local result 174f47b7-dba9-4b68-aee2-e70fed6aa1ed
  pp512-tg128         45.8 tok/s decode   TTFT    813 ms   (5 rounds)
  pp2048-tg512        44.6 tok/s decode   TTFT   3199 ms   (5 rounds)
Nothing was uploaded.
Share it: rapid-mlx benchmark share 174f47b7-dba9-4b68-aee2-e70fed6aa1ed
$ rapid-mlx benchmark results                 # every local run
$ rapid-mlx benchmark inspect <run_id>        # the full record
$ rapid-mlx benchmark share <run_id>          # preview the exact upload, then y/N
```

The Desktop app's Community Benchmark page drives the same CLI (`catalog`, `run`, `results`, and the two-stage `share --preview` / `share --yes`, all with `--json`) and shows the same consent sheet before a share.

### What a run does

The registered text protocol (`rapid-community-speed` v2) is two fixed workloads, each 1 warmup + 5 measured rounds, greedy decoding, prefix cache off, one request at a time:

| Case | Prompt tokens | Output tokens |
|---|---|---|
| `pp512-tg128` | 512 | 128 |
| `pp2048-tg512` | 2048 | 512 |

Prompts are synthetic token sequences (`rapid-synthetic-token-corpus` v2, seeded per case), so no user content is ever measured or recorded. If the model is not in the local Hugging Face cache yet, `run` downloads it first — that network call is model loading, not a submission. The image and video protocols are a fixed prompt, seed and size (see `rapid-image-speed-v1.json` / `rapid-video-speed-v1.json` under `vllm_mlx/catalog/schemas/`). The protocol files are immutable; a new version is a new file and a new `protocol_version`.

Per round, a text measurement records `prompt_tokens`, `output_tokens`, `ttft_ms`, `decode_duration_ms`, `total_duration_ms` and `peak_active_memory_mib`. Decode throughput is derived by readers as `(output_tokens − 1) / (decode_duration_ms / 1000)` tokens per second — the first token lands at `ttft_ms` — which matches llama.cpp `tg` and vLLM TPOT semantics. The website uses this formula; the CLI summary printed by `benchmark run` uses the same one from the release after 0.13.4 (earlier releases divided by `output_tokens`, about 1% higher).

### What `share` sends

`rapid-mlx benchmark share <run_id>` prints the exact request body and asks for `y/N` (default no). `--preview` prints it without asking. The body is the archived run plus one field, `install_id`:

- `model` — the Hugging Face repo id (and subfolder), artifact format, and the quantization block. Releases up to 0.13.4 record the quantization as `unknown`; from the release after 0.13.4 it is read from the cached `config.json` (kind, method, bit width, group size) together with the resolved snapshot revision.
- `machine` — chip, unified memory, CPU/GPU core counts, macOS version, and the run conditions (AC/battery, Low Power Mode, thermal state, memory pressure, available memory). Releases up to 0.13.4 record these as `unknown`; from the release after 0.13.4 they are sampled before the model loads and again after the last measured round.
- `execution` — Rapid-MLX / MLX / Python versions, source revision when running from a checkout, and the execution fields (context length, speculative decoding, KV-cache mode/dtype, prefill backend); settings the runner did not observe are recorded explicitly as `unknown` / `null`, never guessed.
- `workload` — the protocol id, version and digest that produced the numbers.
- `measurements` — the raw per-round samples above.
- `install_id` — 12 hex characters generated once per install and stored in `~/.rapid-mlx/bench-install-id` (mode 0600). The server derives your public pseudonym (for example `northern-windy-numbat ·0a9`) from it. Delete the file to get a new identity.

**Never sent:** username, hostname, hardware serial or UUID, IP address (the endpoint observes the source IP for short-lived rate limiting and does not store it in the record), file paths, environment variables, prompts, model output.

On acceptance the server returns a receipt (`submission_id` = your `run_id`, the payload digest, and — when the server assigned one — your contributor identity). The CLI stores it under `~/.rapid-mlx/benchmarks/receipts/<run_id>.json` and prints your contributor URL, or the general leaderboard URL if the receipt carries no identity. Sharing the same run twice is idempotent: the server answers with the same receipt and `already_exists: true`.

### Contract

The wire format is JSON Schema 2020-12 with `additionalProperties: false` everywhere. The source of truth is `proto/` at the repo root (`proto/model-runtime/v1` for model identity, machine observation and execution config; `proto/community-benchmark/v1` for the run, the protocols and the receipt). Packaged copies live in `vllm_mlx/catalog/schemas/`; tests pin them byte-for-byte to `proto/`. Design notes: [`docs/engineering/decisions/2026-08-31-community-benchmark-wire-contract.md`](../docs/engineering/decisions/2026-08-31-community-benchmark-wire-contract.md) and [`…-community-benchmark-local-workspace.md`](../docs/engineering/decisions/2026-08-31-community-benchmark-local-workspace.md).

Public read surfaces: `GET https://rapidmlx.com/api/benchmarks/atomic/public` (privacy-safe projection; never returns `install_id` or digests) and `GET …/atomic/contributions` (paginated history, `?contributor=<slug>`). Raw records are admin-only.

## Legacy flow: `rapid-mlx bench <alias> --submit`

```console
$ rapid-mlx bench qwen3.5-9b-4bit --submit
```

Runs the same two-bucket workload (512/128 and 2048/512, 1 warmup + 5 rounds, greedy), pretty-prints the submission JSON, asks for `y/N`, saves a local copy, then POSTs it to `https://rapidmlx.com/api/benchmarks`. The payload is the shape in [`schema.json`](schema.json): `hardware`, `software`, `model`, `config`, `buckets.short` / `buckets.long` (median + raw rounds of `decode_tps`, `prefill_tps`, `ttft_ms`), `peak_ram_mb`, optional `--notes`. `--sampled` submits a second row at temp 0.7 / top_p 0.9. The hardware allowlist for this flow lives in `vllm_mlx/community_bench/hardware.py`.

Rows accepted here feed the comparable board (`GET https://rapidmlx.com/api/benchmarks`). The checked-in aggregator groups by `(chip, model alias, rapid_mlx_version)` with median + IQR per metric; memory size is recorded on every row but is not part of that key.

## Choosing between them

Use `rapid-mlx benchmark` unless you specifically want a row on the legacy comparable board. The local-first flow measures image and video models, keeps every run on disk so you can inspect it before deciding, gives you a contributor page when the server assigns an identity, and (from the changes referenced above) records the quantization and the machine conditions the numbers were produced under.

## History

Until mid-2026, `--submit` committed a JSON file into a checkout of this repository and opened a pull request; CI validated it (`.github/workflows/validate-community-submission.yml`) and [`scripts/aggregate.py`](scripts/aggregate.py) reduced `submissions/` into [`aggregated.json`](aggregated.json), which [`index.html`](index.html) renders. That path required a git checkout with a remote pointing at upstream — which nobody who ran `pip install rapid-mlx` or `brew install rapid-mlx` has — so the corpus stalled at 14 rows and the client moved to HTTP (#1403). The files here remain as the immutable record of those first submissions (all CC0), and the aggregator still works on them:

```bash
python community-benchmarks/scripts/aggregate.py          # regenerate aggregated.json
python community-benchmarks/scripts/aggregate.py --check  # CI freshness check
```

Nothing new is written to `submissions/` by either current flow.

## License

The files in this directory (`submissions/`, `aggregated.json`) are CC0 (`SPDX-License-Identifier: CC0-1.0`). The terms under which rapidmlx.com publishes rows submitted over HTTP are stated on the leaderboard page itself, not in the upload contract or the CLI consent text.
