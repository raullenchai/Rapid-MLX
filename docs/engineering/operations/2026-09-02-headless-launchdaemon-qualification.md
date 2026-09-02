# Headless LaunchDaemon qualification — 2026-09-02

## Goal

Qualify the manual, phase-one deployment of `rapid-mlx serve` as a macOS
LaunchDaemon before publishing an operator guide or designing a
`rapid-mlx service` command. The target behavior is system boot, no GUI login,
crash recovery, explicit runtime ownership, and a safe update runbook.

## Initial compatibility host

- Host: `Raullen's Mac mini` (SSH alias `mac-mini`)
- Hardware: Apple M2 Pro, 32 GiB unified memory
- OS: macOS 26.5.2
- Disk free at admission: 65 GiB
- Auto-login: off or unset
- FileVault: **on**
- Idle sleep: disabled
- Automatic restart after power loss: disabled at admission
- Existing unrelated workloads were left running throughout qualification.

The host's SSH account could not run `sudo` non-interactively, so it was used
only for a user-domain compatibility rehearsal. No credentials were requested
or handled by the test operator.

## Baseline runtime

The existing curl-installer environment was:

- Rapid-MLX 0.12.15 at `~/.rapid-mlx`
- Python 3.13.2 arm64
- mlx 0.32.1
- mlx-lm 0.31.3
- mlx-vlm 0.6.15
- transformers 5.15.1
- empty Hugging Face model cache

The pre-upgrade package set was captured at
`/tmp/rapid-mlx-before-headless-test.txt` on the test host.

## Upgrade observations

Installing current `main` (commit `65a0fbc6`, package version 0.13.3) over the
compatible venv preserved both `mlx-vlm 0.6.15` and `transformers 5.15.1`.
That proves a base in-place upgrade does not remove arbitrary extras; it does
**not** make those retained versions compatible with the new release.

Reasserting the declared vision extra upgraded mlx-vlm to the required 0.6.17:

```bash
~/.rapid-mlx/bin/python -m pip install --upgrade '/tmp/rapid-mlx-headless-src[vision]'
```

This is why the public runbook records and explicitly reinstalls the
appliance's extras after every update.

Doctor from this exact `main` build correctly identified the incompatible
mlx-vlm version before the extra was reapplied. It also falsely reported the
importable core packages (`mlx`, `mlx-lm`, `transformers`, `fastapi`, and
`uvicorn`) as broken. Direct checks under the same
`~/.rapid-mlx/bin/python` imported those packages and resolved their metadata.
The false-negative Doctor result is a release blocker for claiming Doctor as a
headless post-update gate; it must be resolved or explicitly accounted for
before this guide is published as fully qualified.

## Non-privileged launchd qualification

Before requesting a privileged install, the same plist shape was loaded into
the logged-in user's `gui/<uid>` domain. `UserName` is ignored in that domain;
the purpose was to validate absolute paths, explicit HOME, cache resolution,
logs, throttling, KeepAlive, model loading, and API probes without touching the
system domain.

Configuration:

- Model: `bonsai-1.7b-2bit` (approximately 473 MiB)
- Served model name: `default`
- Address: `127.0.0.1:18000`
- Separate stdout/stderr logs under `~/Library/Logs/Rapid-MLX`
- KeepAlive: unconditional
- ThrottleInterval: 10 seconds

The first generated test plist accidentally retained the template executable
as a second argument. The process exited with an argparse error and launchd
retried at the configured ten-second interval. This confirmed throttling and
also established that generated arrays must be inspected with `plutil` before
bootstrap. The corrected array was:

```text
rapid-mlx serve bonsai-1.7b-2bit --served-model-name default \
  --host 127.0.0.1 --port 18000 --max-num-seqs 4
```

After correction, all four smoke stages passed:

1. registered job and expected process owner;
2. `/livez`;
3. `/readyz` plus `/v1/models`; and
4. a real one-token `/v1/chat/completions` request.

KeepAlive was then tested by sending SIGTERM to PID 55169. launchd created PID
57116, readiness returned, and the full four-stage smoke passed again. No GUI
or unrelated service was stopped.

## System-domain qualification host

The privileged and reboot gates were run on a separate, otherwise clean test
machine:

- Hardware: Apple M4 Pro, 48 GiB unified memory
- OS: macOS 26.5.1
- FileVault: off
- Automatic login: off
- Rapid-MLX before admission: not installed
- Disk free at admission: approximately 403 GiB

The curl installer was used to establish Rapid-MLX 0.13.3 and its pinned
Python 3.12 runtime. Because that host timed out while downloading the Python
artifact directly from GitHub, the exact installer-pinned artifact was
downloaded on the operator machine, checksum-verified, and copied to the test
host. This was a network transport workaround, not a runtime substitution.

The root-owned plist was installed with mode 0644, loaded as
`system/com.rapidmlx.server`, and verified to run as the unprivileged service
account. The qualification model was `bonsai-1.7b-2bit`, served only on
`127.0.0.1:18000`.

All smoke stages passed in the system domain:

1. launchd registration, PID, and process owner;
2. `/livez`;
3. `/readyz` and `/v1/models`; and
4. a real one-token chat completion.

KeepAlive was tested by asking launchd to send SIGTERM. The service restarted
from PID 1905 to PID 1974, became ready, and passed the full smoke test again.

## Update and extras qualification

The clean installation did not include the optional vision extra. After
installing `rapid-mlx[vision]`, the layered runtime contained `mlx-vlm 0.6.17`
and `transformers 5.15.1`. A subsequent base-package upgrade preserved both
versions. The system service was kept booted out during the update, loaded
again afterwards, and passed the completion smoke test.

Doctor returned a non-zero result even though direct imports of `mlx`,
`mlx_lm`, `transformers`, `fastapi`, `uvicorn`, `PIL`, and `mlx_vlm` succeeded
under the exact application interpreter and the server completed requests.
It incorrectly classified those imports as broken. This is a separate Doctor
probe defect; until it is fixed, Doctor is useful evidence but cannot be the
sole post-update gate. The API smoke test remains authoritative for this
deployment procedure.

## Reboot qualification

The M4 Pro host was restarted with the system service loaded. After SSH became
available again:

- the kernel boot time had advanced;
- `/dev/console` was owned by `root`;
- `uptime` reported zero logged-in users;
- the LaunchDaemon was running as the service account with PID 336; and
- all four smoke stages, including a real completion, passed.

This qualifies normal boot recovery without automatic login. The test also
confirmed that `/tmp` is not durable across reboot, so operator tooling and
backups must not be stored there for an appliance deployment.

`pmset -a sleep 0 autorestart 1` was configured and inspected, but an actual AC
power interruption was not performed because the host had no remotely
controlled power source. The guide therefore treats power-loss recovery as a
site-specific acceptance test. FileVault must also be considered separately:
the initial M2 Pro host could not provide unattended cold-boot recovery while
its data volume required preboot unlock.

## Cleanup

The user-domain rehearsal job was removed with:

```bash
launchctl bootout gui/$(id -u)/com.rapidmlx.server
```

After the M4 Pro qualification, the system job was booted out, its test plist
was removed, port 18000 was confirmed closed, and the admission power settings
were restored (`sleep 1`, `autorestart 0`). Rapid-MLX and the small cached model
were retained for follow-up testing. No persistent daemon or power-policy
change was left on either host.
