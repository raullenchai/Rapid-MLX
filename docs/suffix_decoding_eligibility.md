# Suffix decoding

Suffix decoding is an optional setting for repeated text, such as edits that
copy much of a file. Start with a normal server. If your workload repeats
substantial text, run the same requests with suffix decoding and compare output
quality and latency:

```bash
rapid-mlx serve gemma-4-12b-4bit \
  --speculative-config '{"method":"suffix","num_speculative_tokens":8}'
```

Before trying another alias, run `rapid-mlx info <alias>` and check that
**Spec decode** says supported. Hybrid and recurrent architectures can be
ineligible even if `rapid-mlx recipe` recommends them for your memory tier;
the server skips suffix decoding for those models. Suffix decoding can slow
requests when drafts are rarely accepted, so leave it off unless your own
measurements show a benefit. Run `rapid-mlx serve --help` for the current flag.

## Measuring a model in the repository

The [suffix decoding benchmark](../scripts/bench_suffix_decoding_integrated.py)
compares several workloads with the feature on and off. It needs the
[development environment](../CONTRIBUTING.md) and a Mac with enough memory for
the selected model:

```bash
python3 scripts/bench_suffix_decoding_integrated.py \
  --model gemma-4-12b-4bit --runs 3 --max-tokens 256
```

Review the generated results before editing a model profile. The script's
`--update-profile` option prints a proposed profile update; it does not make a
workload-specific setting safe for every request.
