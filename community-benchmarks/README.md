# Community-submitted performance database

Real numbers from real users' Apple Silicon Macs running Rapid-MLX. Single-vendor benchmarks can only cover the hardware the vendor has — the headline table in the repo `README.md` was measured on an M3 Ultra 256 GB. This directory is how everyone else fills in their own row.

## How submissions work

```
$ rapid-mlx bench qwen3.5-9b-4bit --submit
```

The CLI:

1. Detects your hardware via **non-privileged macOS interfaces only** (`sysctl`, `sw_vers`, `system_profiler`). See [What we collect](#what-we-collect) — the full whitelist is short and the schema enforces it.
2. Runs a standardized benchmark: 2 buckets (short / long), 5 measured rounds + 1 warmup, greedy decode. Numbers are directly comparable to llama.cpp's `llama-bench -p 512 -n 128 -r 5`.
3. Pretty-prints the exact JSON it's about to submit and asks for your `y/N` confirmation. **Nothing leaves your machine without that y.**
4. On `y`, opens a PR via your local `gh` CLI (uses your existing GitHub auth — no new token required). Branch and PR title are auto-generated.
5. A GitHub Action validates the schema + sanity-checks the numbers; on green, a maintainer merges.
6. The aggregator (`scripts/aggregate.py`) rebuilds `aggregated.json`, which the website reads.

## What we collect

Exactly these fields. Anything not on this list is **not** read, sent, or stored:

| Field | Source | Why |
|---|---|---|
| chip | `sysctl -n machdep.cpu.brand_string` | "Apple M4 Pro" — the bucketing key |
| ram_gb | `sysctl -n hw.memsize` | bucketing + headroom analysis |
| cpu_cores | `sysctl -n hw.ncpu` | distinguish M4 / M4 Pro / M4 Max within same chip family |
| gpu_cores | `system_profiler SPDisplaysDataType` (extract `Total Number of Cores`) | distinguishes 16-core vs 20-core M4 Pro |
| macos | `sw_vers -productVersion` | OS-level perf regressions |
| rapid_mlx | `vllm_mlx.__version__` | per-version regression tracking |
| mlx | `mlx.__version__` | underlying framework version |
| python | `sys.version_info[:3]` | rare but real perf differences |
| model.alias | CLI argument | what was benched |
| buckets | bench output | the actual numbers |
| notes | optional `--notes "..."` | "on battery", "fresh boot", etc. |
| submission_id | random uuid4 (truncated) | de-dup key for the aggregator |
| submitted_at | `datetime.utcnow().isoformat()` | when |

**Explicitly not collected**: username, hostname, hardware serial, hardware UUID, IP, MAC address, file paths, prompt text, model output, environment variables, any other data from your machine. The bench uses **synthetic random token sequences** (seeded per `schema_version`), so no user prompt or content ever enters the submission. The only network call is the `gh pr create` you explicitly authorize.

## What we DO with the data

- Aggregate into `aggregated.json` keyed by `(chip, ram_gb, cpu_cores, gpu_cores, model.alias, rapid_mlx_version, sampling)`. Per bucket: median + IQR + sample count. `cpu_cores`/`gpu_cores` are part of the key so the 16-core and 20-core M4 Pro SKUs (same brand string, different silicon) don't collapse.
- Render on `rapid-mlx.com/community-benchmarks` — sortable / filterable table.
- Per-contributor attribution: the PR author (your GitHub handle) is the only contributor signal; no other identity is tracked.
- License: all submissions are CC0 (`SPDX-License-Identifier: CC0-1.0`). The data is community-owned.

## Standardized bench config

Locked by `--submit`. If you want to tune knobs you have to drop `--submit` (the result then can't be uploaded — that's the contract).

| Parameter | Value | Source / rationale |
|---|---|---|
| Short bucket prefill | 512 random tokens | matches `llama-bench -p 512` default |
| Short bucket decode | 128 tokens | matches `llama-bench -n 128` default |
| Long bucket prefill | 2048 random tokens | covers long-context sensitivity |
| Long bucket decode | 512 tokens | matches Rapid-MLX's existing `long_decode_tps` |
| Rounds | 5 measured + 1 warmup discarded | `llama-bench -r 5` |
| Sampling | greedy (temp=0, top_p=1) | comparable to llama-bench / TGI / MLPerf |
| `decode_tps` formula | `output_tokens / (t_end − t_first_token)` | excludes prefill, matches vLLM TPOT + MLX-LM `generation_tps` |
| `prefill_tps` formula | `prompt_tokens / (t_first_token − t_start)` | matches llama.cpp `pp` |
| `ttft_ms` | `(t_first_token − t_request_in) * 1000` | end-to-end first-token latency |
| Reported per bucket | `decode_tps`, `prefill_tps`, `ttft_ms`, `peak_ram_mb` | direct overlap with llama-bench + AA + repo's existing `reports/benchmarks/*.json` |

You can additionally pass `--sampled` to run a **second** line at temp=0.7, top_p=0.9 (real-world sampling). Stored as a separate submission with `sampling="sampled"` — the aggregator buckets greedy and sampled separately so they're directly comparable to AA's user-facing numbers.

## Submission storage

Append-only. One file per submission under `submissions/`:

```
submissions/<YYYYMMDD>-<chip-slug>-<model-slug>-<submission_id>.json
```

Duplicate re-runs from the same machine are allowed and **encouraged** — more samples → tighter median. Each re-run is a fresh `rapid-mlx bench --submit` invocation that generates its own `submission_id`; copying an existing file under a new name does NOT add a second sample (the aggregator de-duplicates by `submission_id` so one machine can't multiply its vote). Outliers are kept in the raw store (auditable) but downweighted in the displayed median (IQR fences).

Submissions are **append-only**: PRs that delete or rename existing rows are rejected by CI. Apply corrections via a new submission rather than mutating history.

## For maintainers

- Schema: `schema.json` (JSON Schema draft 2020-12, additionalProperties: false everywhere).
- Aggregator: `scripts/aggregate.py` — pure stdlib, no deps. Rebuilds `aggregated.json` on every merge into `main`.
- CI: `.github/workflows/validate-community-submission.yml` validates incoming submissions and runs sanity checks (tps > 0, chip on whitelist, etc.). A maintainer reviews and merges; auto-merge can be added later if the false-positive rate stays low.
- Bumping `schema_version`: increment in `schema.json` + `vllm_mlx/community_bench/runner.py::SCHEMA_VERSION`. Old submissions are kept; the aggregator skips entries it doesn't understand.
