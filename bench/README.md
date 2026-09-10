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
