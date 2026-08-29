# Shared macOS host safety

These helpers keep release builds, GUI dogfood, and large-model verification
from interfering with one another on a shared Mac.

## Report host hygiene

Run the report from any registered Rapid-MLX worktree:

```bash
scripts/studio-hygiene.sh --repo /path/to/rapid-mlx
```

Dry-run is the default. The report identifies each registered worktree and why
it is retained, lists finished dogfood directories eligible for cleanup, and
marks the protected model inventory as untouchable. Model caches are never
deleted by this command.

`--apply` is destructive and is reserved for an explicitly authorized cleanup.
Even then, a worktree must be older than six hours, live below the configured
scratch root, be porcelain-clean, have an upstream, have no unpushed commits,
A branch with an open pull request is retained, and inability to read pull
request state makes every worktree ineligible. The script also requires no open
files, rechecks all conditions immediately before removal, and uses
`git worktree remove` rather than deleting a registered tree.
A dogfood directory needs both tool-owned and operator-finished markers.

## Run model loads

Wrap a direct model command so loads with an estimated working set above 20 GiB
share one host-wide lock:

```bash
python3 scripts/large-model-run.py --model qwen3.5-35b-4bit -- \
  rapid-mlx serve qwen3.5-35b-4bit
```

The wrapper estimates the working set from the checked-in alias and size
registries. Unknown models fail closed and require `--working-set-gb`. After
acquiring the lock, it checks available memory again and preserves a 4 GiB
reserve before executing the command. The dogfood MVP and generated Desktop
dogfood launchers already use this wrapper.

## Prepare GUI and VM lanes

`apps/rapid-mac/scripts/dogfood-host-precheck.sh` rejects a locked console or a
nonzero screensaver idle time before a local GUI lane starts. When given a
command after `--`, it keeps the host awake for exactly that process lifetime.
The AX smoke and golden-flow launchers invoke it automatically outside hosted
CI.

For a Tart guest, use a bounded guest-agent wait instead of a fixed sleep:

```bash
scripts/tart-guest-ready.sh --timeout 120 rapid-mac-ci
```

The command succeeds only after `tart exec` works, and otherwise exits with a
clear timeout.
