# bench/

Dev-only micro-benchmarks (not packaged with `pip install rapid-mlx`; for
end-to-end serving benchmarks use `rapid-mlx bench`).

- `bench_radix_vs_hash.py` — multi-tenant prefix-cache index bench (#303):
  N tenants sharing a system prompt, measuring index lookup/insert cost.
- `bench_spec_decode_mtp.py` — MTP speculative-decode bench (#302): decode
  tok/s of `--spec-decode mtp` vs `none` on a Qwen3.5/3.6 MTP checkpoint,
  interleaved runs to avoid thermal drift.
