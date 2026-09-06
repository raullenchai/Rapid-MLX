# Runtime compatibility patches

Install-time patches must be wired from a module on the real production load
path, not only from an adapter or from their own test module. Each installer
therefore needs a subprocess regression test that:

1. starts a clean interpreter;
2. imports the production entrypoint first (currently
   `vllm_mlx.utils.tokenizer` for model-load patches); and
3. only then imports the patch's public `is_installed` probe and verifies both
   it and any upstream marker.

See `test_install_fires_on_real_serve_import_path` in
`tests/test_deepseek_v32_indexer_gate.py` and
`tests/test_qwen3_5_norm_shift.py` for the standing pattern.

`mla_absorbed_verify.py` carries the pending mlx-lm absorbed-MLA crossover
for short multi-token forwards. It is opt-in via
`RAPID_MLX_MLA_ABSORBED_VERIFY=1`, refuses unknown upstream method bodies,
and keeps detailed hot-path counters off unless
`RAPID_MLX_MLA_ABSORBED_VERIFY_STATS=1` is also set. It
requires a post-update cache of at least 1024 tokens, and retires itself
automatically once mlx-lm exposes the upstream helper. Quantized MLA caches
remain on mlx-lm's existing path and require separate qualification.
