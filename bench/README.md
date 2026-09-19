# bench/

Dev-only micro-benchmarks (not packaged with `pip install rapid-mlx`; for
end-to-end serving benchmarks use `rapid-mlx bench`).

- `bench_radix_vs_hash.py` — multi-tenant prefix-cache index bench (#303):
  N tenants sharing a system prompt, measuring index lookup/insert cost.
- `bench_spec_decode_mtp.py` — MTP speculative-decode bench (#302): decode
  tok/s of `--spec-decode mtp` vs `none` on a Qwen3.5/3.6 MTP checkpoint,
  interleaved runs to avoid thermal drift.
- `repro_mtp_forced_k_parity.py` — opt-in real-weight correctness
  diagnostic for #3295. It compares stock `mlx_lm` AR, the same Rapid
  generator parked at K=0, and fixed K=1/2/3 arms across the eight MTP bench
  prompts. It requires every speculative arm to engage and reports first-token
  divergence without treating byte inequality as a verifier failure:

  ```bash
  python3 bench/repro_mtp_forced_k_parity.py --format markdown
  ```
- `bench_ltx_video.py` — LTX-2.3/LTX-2.5 cold-start, per-step, process-tree
  RSS, MLX Metal allocator, and swap benchmark. It emits a JSON record plus a
  Markdown summary and runs each repetition in a fresh process:

  ```bash
  python3 bench/bench_ltx_video.py \
    --runtime mlx23 --model notapalindrome/ltx23-mlx-av-q4 \
    --frames 121 --size 768x512 --seed 42 --runs 3
  ```

  Reproducibility: a plain repo id is a mutable branch. Each report records
  the resolved model identity: local snapshot directories are pinned by a
  SHA-256 content digest (so an in-place rewrite changes the identity), and
  repo ids are resolved through the Hugging Face registry to an immutable
  commit before download. Integrity is re-checked with a stat fingerprint
  before and after every run. Per-run wall-clock deadline: `--deadline`
  (default 3600s) terminates a wedged worker.

  Measurement scope: worker events are HMAC-authenticated over a dedicated
  channel, so echoed prompt text cannot forge timings. Cold-start numbers
  are process cold starts (the preflight content digest cache-warms model
  files before run 1). `step_median_s` requires the runtime to expose
  per-step boundaries: LTX-2.5 emits authenticated sampler events, while
  MLX-2.3 currently has no per-step hook, so its step metrics are reported
  as `null` (follow-up: add a progress callback to `VideoEngine.generate`).
