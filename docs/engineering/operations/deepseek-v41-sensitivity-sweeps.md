# DeepSeek V4.1 layer sensitivity sweeps

Paths in command examples use owner-supplied environment variables: export
`DSV41_SCRATCH_ROOT` for task scratch and `RD_DATA` for approved dataset exports.
Where used, set `SOURCE` to the immutable model snapshot, `GPTQ_OUTPUT` and
`RD_OUTPUT` to build destinations, `GPTQ_HOST`, `REMOTE_WORK`, `REMOTE_PYTHON`
for the remote worker, and `MLX_CPU_LIBRARY_DIR` to a matching CPU MLX library.
These paths must exist on the host running the corresponding command.

Run the retained projection sensitivity scan through the GPTQ entry point:

```sh
.venv/bin/python scripts/deepseek_v41_gptq.py scan \
  --source "$SOURCE" \
  --manifest ${DSV41_SCRATCH_ROOT}/gptq/source-manifest.json \
  --slices "$RD_DATA/calibration-slices.json" \
  --heldout-slices "$RD_DATA/heldout-slices.json" \
  --recipe ${DSV41_SCRATCH_ROOT}/gptq/full-recipe-studio.json \
  --scan ${DSV41_SCRATCH_ROOT}/gptq/sensitivity-scan
```

The scan is resumable at committed layer boundaries and records per-projection
RTN ranking proxies plus each layer's 4-bit block-output relative error. Use
`--through-layer` for a bounded pilot. The default path requires Metal; `--cpu`
exists only for tiny CPU qualification fixtures, not production measurements.
Treat sources, manifests, recipes and completed scan artifacts as immutable.
No inference-quality claim follows from the proxy alone: score every resulting
build on the logit rulers.

Collect a smaller candidate cache with the existing `collect --slices` manifest,
then compare it against a larger reference cache:

```sh
.venv/bin/python scripts/evaluate_deepseek_v41_rd.py compare \
  --reference /path/to/reference-cache --candidate /path/to/candidate-cache \
  --subset --output ${DSV41_SCRATCH_ROOT}/phase5/comparison.json
```

With `--subset`, candidate slices must be a nonempty subset of the reference.
Revision, tokenizer, per-slice corpus, logits hashes, teacher-forced token IDs,
and routing-mass checks still apply. The output records `compared_slices`; mean
and worst-domain KL describe only those slices. Without the flag, slice sets
must still match exactly.
