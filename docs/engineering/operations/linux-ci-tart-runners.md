# Self-hosted Linux CI runners (ephemeral tart VMs)

Linux CI jobs can run on Rapid-MLX's own Apple Silicon Macs instead of hosted
`ubuntu-latest` runners. Each job gets a fresh Ubuntu 24.04 arm64 VM that is
destroyed afterwards. This page covers setup, the supervisor, drain/pause,
rollback, and rebuilding the image.

## Shape

| Item | Value |
| --- | --- |
| Labels | `self-hosted`, `Linux`, `ARM64`, `rapidmlx-tart` |
| Routing switches | repository variables (JSON runner spec): `CI_LINUX_RUNNER` routes only the heavy `ci.yml` `test-matrix` shards; `CI_LINUX_LIGHT_RUNNER` routes every other Linux job (lint, guards, merge-on-green, and similar). Unset → `ubuntu-latest`. Fork heads always stay on hosted runners. |
| Pools | the CI Mac mini (2 slots); the M5 Max CI host (3 slots, joins when its lease frees) |
| VM size | 4 vCPU, 8 GB RAM, 30 GB sparse disk, copy-on-write clone of one golden image |
| Lifetime | one VM per job: clone → boot → JIT-register → run one job → destroy |
| Image | `rmx-golden`, built from `ghcr.io/cirruslabs/ubuntu:24.04`, pinned by digest in `build-golden.sh` |

What the golden image contains: build-essential, git, curl, jq, zstd, unzip, and
the GL/GLib shared libraries that OpenCV wheels need. It also has
`python`/`python3` (system 3.12 via `python-is-python3`, because utility jobs
call bare `python` without setup-python), the `gh` CLI, Node 22, a Swift
toolchain (the hosted image ships one, and some tests check for `swift`), uv,
and the GitHub Actions runner (linux-arm64).

Before routing a new class of job here, grep its `run:` steps for every tool it
calls without a setup action, and confirm the image provides each one.

Python 3.10–3.13 is pre-populated in `/opt/hostedtoolcache` in the hosted
layout, so `actions/setup-python` resolves versions without downloading. The
tiktoken vocab files are pre-seeded under `/opt/tiktoken-encodings` with
`TIKTOKEN_ENCODINGS_BASE` set, because the arm64 Linux `openai-harmony` wheel
cannot fetch them itself.

Light jobs default to hosted runners on purpose. They take seconds to a minute,
and queueing them behind 7–10 minute test shards on a five-slot pool causes
head-of-line blocking for merge gates. Set `CI_LINUX_LIGHT_RUNNER` only when
the pool has spare capacity.

## Security model

- The GitHub credential that mints JIT runner configs lives only on the host,
  at `~/ci-runners/secrets/gh-token` (mode 0600). The VM receives only the
  single-use JIT config, over stdin.
- VMs get no host mounts, no host credentials and no SSH keys, and each VM is
  deleted after one job. A runner is registered with a JIT config and never
  sees a second job.
- Fork pull requests never route here: the `runs-on` expression checks that
  the head repository is this repository. First-time external contributors
  also need workflow approval.

## Host layout

The scripts are versioned in
[`tools/ci-runners/tart/`](../../../tools/ci-runners/tart/), and that copy is
the source of truth. Each host runs a copy of them under `~/ci-runners`, which
also holds all runner state:

```
bin/tart -> ../tart.app/Contents/MacOS/tart   official signed release (Developer ID, notarized)
bin/common.sh  bin/slot.sh  bin/supervisor.sh  bin/build-golden.sh
guest/provision.sh      runs inside the VM when the golden image is built
host.env                MACHINE_ID, SLOTS, VM_CPU, VM_MEM_MB, VM_DISK_GB, MIN_FREE_GB, MC_DIRECT
secrets/                gh-token (+ machinecontrol.token on hosts that query Machine Control directly)
state/                  per-slot runner ids, Machine Control lease owner
logs/                   supervisor.log, slotN.runner.log, slotN.tart.log, build-golden.log
tart-home/              TART_HOME (VM images)
PAUSE                   present = drain
```

The supervisor is a per-user LaunchAgent, `~/Library/LaunchAgents/com.rapidmlx.ci-runners.plist`.
It has `RunAtLoad` and `KeepAlive` set, so it starts at login and restarts if it
exits. Virtualization needs a logged-in GUI session, so a rebooted host needs
an automatic or manual login before its slots come back.

## Supervisor behavior

The supervisor runs one loop per slot. Each loop works like this:

1. If draining (see below), wait.
2. If free disk is below `MIN_FREE_GB`, wait.
3. Otherwise run `tart clone rmx-golden rmx-slotN` and boot the VM headless.
   If a drain started during boot, destroy the VM without registering.
4. Call `POST /repos/{repo}/actions/runners/generate-jitconfig` with the labels
   above, and start `run.sh --jitconfig` in the VM through `tart exec`.
5. When the runner exits after its one job, stop and delete the VM, and remove
   any leftover registration. If GitHub cannot confirm the removal, the runner
   id goes to `state/slotN.orphans` and is retried on every loop.

On start, a loop destroys its stale VM and runner from before a crash or
reboot. A VM older than 7 hours is killed.

## Drain and pause

The host stops starting new VMs when either of these is true:

- `~/ci-runners/PAUSE` exists; or
- the host's Machine Control lease is held by any owner other than
  `rapid-mlx/harbor-ci-runners` (benchmarks get a quiet machine).

Running jobs always finish. Idle registered runners are deregistered and their
VMs destroyed within one 20 s poll, so a runner that was already idle can still
accept one job in that window. Drain is a quiesce, not a hard cutoff: wait for
`tart list` to show no `rmx-slot*` VMs before relying on a quiet machine. Machine Control state comes
from the API directly (`MC_DIRECT=1`). For hosts that cannot reach it, a
LaunchAgent on the Studio relays the lease owner into `state/mc-lease` every
20 s. Stale state (more than 10 min old) is treated as "not leased", so a
Machine Control outage does not stop CI.

```bash
touch ~/ci-runners/PAUSE     # drain this host
rm ~/ci-runners/PAUSE        # resume
tail -f ~/ci-runners/logs/supervisor.log
```

## Rollback

```bash
gh variable delete CI_LINUX_RUNNER -R raullenchai/Rapid-MLX         # test shards back to ubuntu-latest
gh variable delete CI_LINUX_LIGHT_RUNNER -R raullenchai/Rapid-MLX   # light jobs back (if it was set)
touch ~/ci-runners/PAUSE                                     # on each host
# Full stop and uninstall:
launchctl bootout gui/$(id -u)/com.rapidmlx.ci-runners
```

`bootout` sends TERM, and each slot destroys its VM. Remove `~/ci-runners`
after `tart list` shows no `rmx-slot*` VMs.

## Setup on a new host

1. Claim the host in Machine Control as `rapid-mlx/harbor-ci-runners`.
2. Install tart from the official release tarball. Verify the published
   SHA-256 checksum and run `spctl -a` on `tart.app`, then link the binary
   to `~/ci-runners/bin/tart`.
3. Copy `bin/` and `guest/` from `tools/ci-runners/tart/` into
   `~/ci-runners/`. Render the plist with
   `sed "s#__HOME__#$HOME#g" com.rapidmlx.ci-runners.plist.template > ~/Library/LaunchAgents/com.rapidmlx.ci-runners.plist`.
   Write `host.env` from `host.env.example`, and write the secrets with mode
   0600.
4. Run `bin/build-golden.sh`, then delete the cached OCI base image if disk is
   tight.
5. Load the supervisor with
   `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.rapidmlx.ci-runners.plist`.
6. Check that the runners appear online with the labels above.
7. Release the setup lease.

## Rebuilding the golden image

```bash
touch ~/ci-runners/PAUSE                 # optional; running clones are unaffected
~/ci-runners/bin/build-golden.sh         # builds rmx-golden-next, then swaps it in
rm ~/ci-runners/PAUSE
```

To change a tool version, edit `guest/provision.sh` (runner and uv versions
with their SHA-256, Node, Python, Swift).

The previous image is kept as `rmx-golden-prev` until the new one is renamed
into place, so a failed swap never leaves the host without an image.

The build shuts the guest down from inside, after `sync`. Stopping it from
outside can lose ext4 delayed-allocation writes, and that left zero-length
runner binaries in an early image. The build then aborts if it finds any
zero-length file in the runner or `/usr/local/bin`.

Validate a new image before relying on it. Open a throwaway draft PR that
hard-routes every `CI_LINUX_RUNNER` and `CI_LINUX_LIGHT_RUNNER` job to `rapidmlx-tart`, and run
merge-on-green from that branch with `workflow_dispatch` and `dry_run=true`.
Test shards alone are not enough: they use setup-python, while the utility jobs
(main guard, merge on green) call the system `python`.

## Validation record (2026-10-18)

A throwaway branch ran the `test-matrix` shard steps on the CI Mac mini (M2 Pro,
2 VMs in parallel). Job wall-clock includes checkout, setup-python, and pip:

| Shard | Tart VM | Hosted `ubuntu-latest` (last 3 `main` runs) |
| --- | --- | --- |
| 3.11 / 1 | 9 m 06 s | 10 m 00 s – 15 m 07 s |
| 3.12 / 2 | 7 m 45 s | 12 m 19 s – 14 m 22 s |

VM boot to runner online took 5–8 s.
