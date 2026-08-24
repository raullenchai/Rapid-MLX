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

- `bench_spec_decode_mtp_server.py` — production-serving benchmark using
  NVIDIA SPEED-Bench workloads and Rapid's OpenAI-compatible streaming API.
  It captures TTFT, end-to-end latency, completion throughput, multi-turn
  context, raw request results, and before/after MTP Prometheus counters in an
  atomic JSON receipt with a SHA-256 sidecar. It does not import or compare
  against llama.cpp tooling.

  Capture baseline and MTP arms from separate, freshly started servers. Keep
  the model, sample IDs, sampling, context workload, output limit, and
  concurrency identical. The MTP arm's positive-control lane must use
  concurrency 1 and `--require-mtp-activity`:

  ```bash
  python3 bench/bench_spec_decode_mtp_server.py run \
    --base-url http://127.0.0.1:18000 \
    --target-revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
    --server-label baseline --sidecar none \
    --bench qualitative --category coding --limit 8 \
    --max-tokens 512 --concurrency 1 --warmup-samples 1 \
    --output artifacts/mtp-server-baseline.json

  python3 bench/bench_spec_decode_mtp_server.py run \
    --base-url http://127.0.0.1:18001 \
    --target-revision 8b2b98c00a6b4d291155e4890773ca8f769aee53 \
    --server-label mtp \
    --sidecar mlx-community/Qwen3.5-9B-MTP-4bit@REVISION \
    --bench qualitative --category coding --limit 8 \
    --max-tokens 512 --concurrency 1 --warmup-samples 1 \
    --require-mtp-activity \
    --output artifacts/mtp-server-mtp.json

  python3 bench/bench_spec_decode_mtp_server.py compare \
    --baseline artifacts/mtp-server-baseline.json \
    --mtp artifacts/mtp-server-mtp.json \
    --output artifacts/mtp-server-comparison.json
  ```

  Rapid currently enables MTP only when the scheduler's generation batch has
  one UID. Concurrency 2/4 cells therefore test production scheduler behavior
  and may mix ordinary-AR batches with moments where one request runs alone;
  they are not evidence of concurrent verification overlap. Report their
  attempts and K histograms as observed rather than assuming MTP engaged.
  The default SPEED-Bench dataset revision is pinned; pass an explicit
  `--dataset-revision` only when intentionally re-baselining the workload.
