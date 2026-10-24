# Recorded GPTQ recipe amendments

Paths in command examples use owner-supplied environment variables: export
`DSV41_SCRATCH_ROOT` for task scratch and `RD_DATA` for approved dataset exports.
Where used, set `SOURCE` to the immutable model snapshot, `GPTQ_OUTPUT` and
`RD_OUTPUT` to build destinations, `GPTQ_HOST`, `REMOTE_WORK`, `REMOTE_PYTHON`
for the remote worker, and `MLX_CPU_LIBRARY_DIR` to a matching CPU MLX library.
These paths must exist on the host running the corresponding command.

Stop the launcher at a committed layer boundary and hold its exclusive
`${DSV41_LAUNCHER_LOCK}` while applying code and amending metadata.
Set this variable to the same lock path used by the external launcher.
The `amend-recipe` command must not race a launcher or another writer.

```
python scripts/deepseek_v41_gptq.py amend-recipe \
  --parts PARTS --scratch STATE --recipe NEW.json \
  --from-layer F --reason 'Owner-approved precision and memory adjustment'
```

`F` must equal checkpoint `next_layer`. Only resident budget, memory/cache caps,
and gate/up override membership at or after `F` may change. Changes to earlier
membership, other recipe fields, an FFN phase, working directories, and
uncommitted next-layer outputs fail closed. Earlier layer artifacts are read and
hash-verified without rewriting them. Budget changes do not retroactively change
the historical allocation estimate; assembly recomputes actual payload costs.

`run.json` records the old recipe/identity, new recipe, boundary, reason and UTC
timestamp. It is replaced atomically before checkpoint identity is replaced
atomically. There is no cross-file atomic transaction: a crash between these
writes leaves readers refusing to continue. Retry the exact same amendment
(including reason) under the lock to repair checkpoint identity. Do not manually
replace identities. Backup copies are for diagnosis, not for rollback after any
new layer has started.

Completion records, native part readers and held-out prefix boundaries use the
identity in force at their layer. The quality prefix cache retains its original
directory, so completed prefixes remain reusable after earlier boundaries have
been reclaimed. Assembly preserves epoch history in `gptq-provenance.json` and
`config.json` and retains each layer's completion provenance.

The Studio launcher checks `FULL_RECIPE` (after its fixed memory caps) against
the recorded recipe in force at checkpoint `next_layer`. A stale Studio recipe
file may be replaced only after that comparison succeeds. Touch its
`STOP_FILE` (default `gptq/stop-at-boundary`) to exit between layers; remove the
sentinel before relaunch. A running Bash script must first be retired safely;
replacing its file does not retrofit sentinel checks into its current loop.

An original-reference process updates only the reference cache, with completion
markers and checkpoint writes after boundary generation. It does not advance the
quantization checkpoint. Interrupting one can leave incomplete reference work;
the operator procedure waits for either reference or run children to exit
naturally, avoiding that recovery path entirely.

The separate two-host fullrun driver's immutable `driver-inputs.json` binding
still rejects recipe/tool changes on resume. Its export checks understand layer
epochs, but this live Studio procedure uses the shell launcher exclusively; do
not switch this amended run to the remote driver.

The launcher is an owner-maintained external script and is not shipped here.
Its optional regression contract reads `DSV41_LAUNCHER` when explicitly set;
otherwise that single launcher test skips. The recipe amendment, epoch and
artifact checks run independently of the launcher.
