# Community-submitted performance database

Real numbers from real users' Apple Silicon Macs running Rapid-MLX. Single-vendor benchmarks can only cover the hardware the vendor has — the headline table in the repo `README.md` was measured on an M3 Ultra 256 GB. This directory is how everyone else fills in their own row.

## How submissions work

```
$ rapid-mlx bench qwen3.5-9b-4bit --submit
```

The CLI:

1. Detects your hardware via **non-privileged macOS interfaces only** (`sysctl`, `sw_vers`, `system_profiler`). See [What we collect](#what-we-collect) for the exact field list; the allowlist lives in `vllm_mlx/community_bench/hardware.py` and the schema constrains the recorded values.
2. Runs a standardized benchmark: 2 buckets (short / long), 5 measured rounds + 1 warmup, greedy decode. Numbers are directly comparable to llama.cpp's `llama-bench -p 512 -n 128 -r 5`.
3. Pretty-prints the exact JSON it's about to submit and asks for your `y/N` confirmation. **No submission JSON leaves your machine without that y.** Note that the model itself is fetched from HuggingFace before consent (same as any other `rapid-mlx bench` run) — that network call is part of loading the model, not the submission.
4. On `y`, runs `git push` to your `origin` remote on GitHub, then `gh pr create` against `raullenchai/Rapid-MLX`. Both use your existing git/gh credentials — no new token required. The commit is authored by whatever git author identity your repo has configured (`git config user.name` / `user.email`); that's how PRs work and is the only contributor identity attached to the row.
5. A GitHub Action validates the schema + sanity-checks the numbers; on green, a maintainer merges.

## Aggregator + page

Raw submissions are reduced to a sortable table by [`scripts/aggregate.py`](scripts/aggregate.py): groups by `(chip, model_alias, rapid_mlx_version)` — the axis the schema description names — and computes median + IQR per metric within each group. Output is the committed [`aggregated.json`](aggregated.json), which is what external consumers (the website's Performance tab; this directory's own [`index.html`](index.html)) fetch.

```bash
# Regenerate the aggregate after adding a submission
python community-benchmarks/scripts/aggregate.py

# Verify the on-disk aggregate matches what would be regenerated
# (the CI freshness check runs exactly this)
python community-benchmarks/scripts/aggregate.py --check
```

[`index.html`](index.html) is a single-file reference UI — no build step, no framework. Drop it (plus `aggregated.json`) onto any static host: GitHub Pages, S3, the rapidmlx.com Performance tab, anywhere. The page fetches `aggregated.json` from the same directory and renders a sortable filterable table.

GitHub Action [`aggregate-bench.yml`](../.github/workflows/aggregate-bench.yml) closes the loop: on PR, it verifies the committed `aggregated.json` is fresh against the current `submissions/`; on push-to-main, it regenerates and auto-commits if any submission slipped through without a regenerate.

## What we collect

The full payload is exactly the fields defined in `schema.json`. The schema's `additionalProperties: false` everywhere means a contributor *cannot* add fields beyond this list and have CI accept the row. The fields below are the complete contents:

| Field | Source | Why |
|---|---|---|
| schema_version | const `1` | so the aggregator can skip unknown versions |
| submission_id | random uuid4 (first 12 hex chars) | de-dup key |
| submitted_at | `datetime.now(timezone.utc).isoformat()` | when |
| hardware.chip | `sysctl -n machdep.cpu.brand_string` | "Apple M4 Pro" — the bucketing key |
| hardware.ram_gb | `sysctl -n hw.memsize` | bucketing + headroom analysis |
| hardware.cpu_cores | `sysctl -n hw.ncpu` | distinguish M4 / M4 Pro / M4 Max within same chip family |
| hardware.gpu_cores | `system_profiler SPDisplaysDataType` (extract `Total Number of Cores`) | distinguishes 16-core vs 20-core M4 Pro |
| software.macos | `sw_vers -productVersion` | OS-level perf regressions |
| software.rapid_mlx | `vllm_mlx.__version__` | per-version regression tracking |
| software.mlx | `mlx.__version__` | underlying framework version |
| software.python | `sys.version_info[:3]` | rare but real perf differences |
| model.alias | CLI argument | what was benched (whitelisted alias from `aliases.json`) |
| model.hf_path | resolved from alias | exact HF repo the alias points at |
| config.rounds / warmup_rounds / sampling / buckets_spec | locked by the standardized runner | comparability axes |
| config.prompt_hash | SHA256[:16] of the synthetic prompt seed | tampering check |
| buckets.short / buckets.long | bench output (decode_tps / prefill_tps / ttft_ms summary + 5 raw rounds each) | the actual numbers |
| peak_ram_mb | `mx.metal.get_peak_memory()` (best-effort; nullable) | headroom analysis |
| notes | optional `--notes "..."` | "on battery", "fresh boot", etc. |

**Explicitly not collected**: username, hostname, hardware serial, hardware UUID, IP, MAC address, file paths, prompt text, model output, environment variables, any other data from your machine. The bench uses **synthetic random token sequences** (seeded per `schema_version` + bucket length), so no user prompt or content ever enters the submission. The network calls are: (a) `gh pr create` against `raullenchai/Rapid-MLX` — explicitly consented; (b) `git push` to your `origin` remote — same consent prompt covers it; (c) HuggingFace model download as part of normal `rapid-mlx bench` model loading, identical to any other bench run.

## What we DO with the data

- Store each submission as a JSON file under `submissions/`. That's the raw store; future tooling reads from it.
- Per-contributor attribution: the PR author (your GitHub handle) is the only contributor signal; no other identity is tracked.
- License: all submissions are CC0 (`SPDX-License-Identifier: CC0-1.0`). The data is community-owned.
- Bucketing keys we plan to expose to future readers: `(hardware.chip, hardware.ram_gb, hardware.cpu_cores, hardware.gpu_cores, model.alias, software.rapid_mlx, config.sampling)`. The schema already carries each of these so we don't lose information now.

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
| `decode_tps` formula | `(output_tokens − 1) / (t_end − t_first_token)` | excludes prefill **and** the first token's decode (it lands at `t_first_token`); matches vLLM TPOT / llama.cpp `tg` semantics. For `N == 1` we fall back to `N / window` to avoid a 0/0. |
| `prefill_tps` formula | `prompt_tokens / (t_first_token − t_start)` | matches llama.cpp `pp` |
| `ttft_ms` | `(t_first_token − t_request_in) * 1000` | end-to-end first-token latency |
| Reported per bucket | `decode_tps`, `prefill_tps`, `ttft_ms` | direct overlap with llama-bench + AA + repo's existing `reports/benchmarks/*.json` |
| Reported per submission (top-level) | `peak_ram_mb` (nullable) | one value covers the whole bench, taken after warmup |

You can additionally pass `--sampled` to submit a **second** row at temp=0.7, top_p=0.9 (real-world sampling) right after the greedy submission. Stored as a separate submission with `sampling="sampled"` — `sampling` is part of every submission's bucket key so greedy and sampled rows never collapse into one number downstream. `--sampled` does *not* replace greedy; both rows are submitted (each with its own consent prompt).

## Submission storage

Append-only. One file per submission under `submissions/`:

```
submissions/<YYYYMMDD>-<chip-slug>-<model-slug>-<submission_id>.json
```

Duplicate re-runs from the same machine are allowed and **encouraged** — more samples → tighter median. Each re-run is a fresh `rapid-mlx bench --submit` invocation that generates its own `submission_id`; copying an existing file under a new name does NOT add a second sample (the validator rejects duplicate `submission_id`s so one machine can't multiply its vote). Outliers are kept in the raw store for full auditability.

Submissions are **append-only**: PRs that delete or rename existing rows are rejected by CI. Apply corrections via a new submission rather than mutating history.

## For maintainers

- Schema: `schema.json` (JSON Schema draft 2020-12, additionalProperties: false everywhere).
- CI: `.github/workflows/validate-community-submission.yml` validates incoming submissions and runs sanity checks (tps > 0, chip on whitelist, etc.). A maintainer reviews and merges; auto-merge can be added later if the false-positive rate stays low.
- Bumping `schema_version`: increment in `schema.json` + `vllm_mlx/community_bench/runner.py::SCHEMA_VERSION`. Old submissions are kept; readers should skip entries with an unknown version.
- Aggregator + website are explicitly deferred to a follow-up PR once the raw store has enough real submissions to design against.
