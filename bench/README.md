# bench/

Dev-only micro-benchmarks (not packaged with `pip install rapid-mlx`; for
end-to-end serving benchmarks use `rapid-mlx bench`).

- `bench_radix_vs_hash.py` — multi-tenant prefix-cache index bench (#303):
  N tenants sharing a system prompt, measuring index lookup/insert cost.
- `bench_spec_decode_mtp.py` — MTP speculative-decode bench (#302):
  product-level `mtp/none` landing comparison plus a same-generator
  `mtp/ar` diagnostic on a Qwen3.5/3.6 MTP checkpoint. Arms use discarded
  warm-ups, balanced cyclic ordering, and paired median/IQR reporting to
  reduce thermal and order bias; greedy runs can enforce token-stream
  equality with `--require-lossless`. Kernel warm-up resets the persistent
  auto-K controller; cold-start and explicit `--controller-warmup-generations`
  pre-calibrated runs are reported as separate phases.

  This is a single-stream, short-generation diagnostic, not a production
  workload qualification. The built-in 8-prompt/128-token matrix localizes
  generator and verification costs; it does not represent long-context coding,
  multi-turn agents, or concurrent scheduler traffic. Use `rapid-mlx bench` (or
  a captured serving workload) for end-to-end TTFT, latency, and throughput,
  and keep context length, output length, sampling parameters, concurrency,
  acceptance, and observed-K distributions in the report. A greedy throughput
  comparison with token mismatches is confounded; rerun it with
  `--require-lossless` before treating its speedup as controlled evidence.
- `repro_mtp_forced_k_parity.py` — opt-in real-weight correctness diagnostic
  for the MTP verify path. It compares stock `mlx_lm` AR, the same generator at
  fixed K=0, and fixed K=1, 2, and 3. It requires speculative attempts and
  verify calls in every K>0 arm and reports the first greedy token-stream
  divergence without making a throughput claim. The default model and matching
  sidecar require about 5 GB on disk:

  ```bash
  python3 bench/repro_mtp_forced_k_parity.py --format markdown
  ```

  Add `--require-parity` to return exit 1 on a divergence. Exit 2 always means
  the result is invalid because a requested speculative arm did not engage.
