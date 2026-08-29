# Self-hosted macOS GUI runner

The `mini-gui` runner supplies the logged-in Aqua session needed to build the
Desktop test app and run its GUI golden-flow gate. It is repository-scoped and
is not a general model or release runner. The generic Swift build,
`test-apple-silicon`, L1 smoke, and release jobs stay hosted because the mini is
reserved for deterministic GUI work and does not replace those lanes' build,
GPU, performance, or release contracts.

## Security boundary

Rapid-MLX is public. Never run code from a fork on project hardware. The
workflow selector sends every external head to `macos-15` before it considers
runner status. Only same-repository integration candidates, merge groups, and
main pushes are eligible for `mini-gui`.

The selector needs a repository-scoped, fine-grained token with only this
repository and **Administration: read** permission so it can read runner
online state. Store it as the Actions secret `MINI_GUI_RUNNER_READ_TOKEN`.
Missing credentials, an API error, or no online matching runner all select the
hosted fallback. Never reuse a release or publishing token for this purpose.

## One-time installation

1. Log in locally as the account that owns the interactive GUI session. Keep
   that account logged in and the screen unlocked; SSH alone does not create
   the GUI session.
2. In repository **Settings → Actions → Runners**, choose **New self-hosted
   runner**, macOS, ARM64. Use only the time-limited registration command shown
   by the web UI. Do not request a token through the API, paste it into chat,
   save it in a file, or commit it.
3. Install the current ARM64 runner under `~/actions-runner-mini-gui` and verify
   the archive SHA-256 against the release asset digest before extraction.
4. From that directory, run the UI-provided configuration command with:

   ```bash
   ./config.sh \
     --url https://github.com/raullenchai/Rapid-MLX \
     --name mini-gui \
     --labels mini-gui \
     --work _work \
     --unattended
   ```

   Insert the UI's `--token` argument in the private shell. The runner also
   receives the default `self-hosted`, `macOS`, and `ARM64` labels.
5. Install the service as the logged-in user, never with `sudo`. Point the
   official service installer at the repository's interactive template so the
   listener holds display/system sleep assertions for its whole lifetime:

   ```bash
   cp /path/to/Rapid-MLX/scripts/actions-runner-mini-gui.plist.template \
     ~/actions-runner-mini-gui/bin/actions.runner.mini-gui.plist.template
   cd ~/actions-runner-mini-gui
   export GITHUB_ACTIONS_RUNNER_SERVICE_TEMPLATE="$PWD/bin/actions.runner.mini-gui.plist.template"
   ./svc.sh install
   ./svc.sh start
   ./svc.sh status
   ```

   The generated file must live in `~/Library/LaunchAgents`; a system daemon
   cannot provide the logged-in Aqua/TCC context.

## Preflight and health

Before accepting GUI jobs, verify:

```bash
stat -f %Su /dev/console
pgrep -x Dock
cd ~/actions-runner-mini-gui && ./svc.sh status
pmset -g assertions
```

The console user must be the runner user, Dock must exist, the service must be
started, and `caffeinate` must hold display/system assertions. The workflow's
Accessibility preflight remains fail-closed: a locked/missing GUI session or a
missing TCC grant is a host failure, not an application failure.

Grant Accessibility to the runner's interactive parent only through macOS
System Settings. Do not edit the TCC database directly.

## Dogfood coexistence

The mini remains available for daily development. Pause the runner before
interactive development or dogfood needs exclusive use of its GUI session:

```bash
cd ~/actions-runner-mini-gui && ./svc.sh stop
```

The next workflow evaluates the runner as offline and selects hosted macOS.
Resume after dogfood and confirm it returns online:

```bash
cd ~/actions-runner-mini-gui && ./svc.sh start
cd ~/actions-runner-mini-gui && ./svc.sh status
```

Pause before dispatching the candidate: host selection is made once at workflow
start, so stopping the service after selection can leave a selected job queued
for the mini. Stopping the service also does not cancel a job already running.
Let that job finish or cancel the workflow through the normal CI controls before
stopping it.

## Verification and recovery

For the first live proof, use an internal integration candidate and confirm both
`gui-app-build` and every `gui-golden-flows` matrix job report runner
`mini-gui`; the `desktop-tests` aggregate must remain green. Then stop the
service before dispatching another internal candidate and confirm both jobs
report hosted `macos-15`.

If the mini is unhealthy, leave the service stopped and use the hosted fallback.
Do not weaken the GUI preflight or branch/fork guard to make a run start.
