# DeepSeek V4.1 layer sensitivity sweeps

Create a forced-width allocation without rerunning RD optimization:

```sh
.venv/bin/python scripts/deepseek_v41_variant_alloc.py \
  --base /path/to/base-allocation.json \
  --costs /path/to/release-policy-costs.json \
  --set-bits 32,33,34=3/3/3 \
  --output /private/tmp/rapid-mlx-dsv41-1017/phase5/v-L32-34.json
```

Widths are gate/up/down, each from 2/3/4. Repeat `--set-bits` to override
additional layers; later overrides win. Expert IDs and draft policy are preserved.
The helper shares the allocator's equal-shaped projection cost function, including
router rows. It retains the base allocation's selected draft cost, since older
inventories price a different draft width. It prints decimal resident GB delta,
clears the enforced `budget_bytes`, and records the base target and identity in
`variant`. Base optimality estimates are removed. Payload accounting is updated;
shard/header accounting is deferred to the builder's exact source plan.

Plan the incremental build before writing model data:

```sh
.venv/bin/python scripts/repack_deepseek_v41_native.py \
  --source /path/to/immutable-release-checkpoint \
  --allocation /private/tmp/rapid-mlx-dsv41-1017/phase5/v-L32-34.json \
  --shared-engram /path/to/shared-engram \
  --reuse-from /path/to/base-build \
  --destination /path/to/variant-build --dry-run
```

The report lists `reuse.reused` and `reuse.rewritten` shard paths, including
Engram and the optional DSpark sidecar. Dry runs create no directories or model
files. Remove `--dry-run` only when ready to build. The destination and reuse
files must share a filesystem; the existing shared-Engram requirement also applies.
Only files with identical semantic tensor plans, exact serialized header bytes,
and expected size are hard-linked. Missing/mismatched shards are written normally.
A different source identity is rejected. Configs and indexes match fresh builds;
`reuse-provenance.json` separately records the reuse allocation/config hashes.

Treat sources and builds as immutable. Identity checks cover source config and
tensor metadata, not full source/reuse payload hashes. In-place edits to a linked
file affect every build sharing that inode. No new inference quality claim follows
from successful reuse: measure each variant's logits.

Collect a smaller candidate cache with the existing `collect --slices` manifest,
then compare it against a larger reference cache:

```sh
.venv/bin/python scripts/evaluate_deepseek_v41_rd.py compare \
  --reference /path/to/reference-cache --candidate /path/to/candidate-cache \
  --subset --output /private/tmp/rapid-mlx-dsv41-1017/phase5/comparison.json
```

With `--subset`, candidate slices must be a nonempty subset of the reference.
Revision, tokenizer, per-slice corpus, logits hashes, teacher-forced token IDs,
and routing-mass checks still apply. The output records `compared_slices`; mean
and worst-domain KL describe only those slices. Without the flag, slice sets
must still match exactly.
