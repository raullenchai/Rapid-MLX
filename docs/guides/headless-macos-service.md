# Headless macOS service

Run Rapid-MLX as a system `launchd` daemon when a Mac is used as an unattended
inference appliance. A system daemon starts after macOS boots and does not
depend on a graphical login. This is the macOS equivalent of an enabled
`systemd` service.

> **Operational scope:** `rapid-mlx service` (the supported way to set this
> up) installs and manages the daemon for you in one command. The manual
> runbook below remains as the fallback / recovery path and as the reference
> for what the CLI produces. Test the procedure while you still have physical
> or out-of-band access to the Mac.

## Quickstart: `rapid-mlx service`

Create a dedicated, non-administrator service account (shown as `serveuser`)
and install Rapid-MLX there with the one-line installer, exactly as in
[section 1](#1-prepare-the-service-account) below. Download the intended
model once as that account. Then, from an administrator session:

```bash
sudo rapid-mlx service install \
  --service-user serveuser \
  --model qwen3.5-4b-4bit \
  --host 127.0.0.1 --port 8000
```

`install` validates the service account, writes a deterministic,
root-owned plist to `/Library/LaunchDaemons/com.rapidmlx.server.plist`,
and bootstraps it into the system domain. It refuses to install onto an
administrator or system account, refuses to embed a secret
(`--api-key`) into the definition, and refuses when a server already
answers on the target port (so it cannot silently race a Desktop-managed
instance). Rehearse first without touching the machine:

```bash
rapid-mlx service install --service-user serveuser \
  --model qwen3.5-4b-4bit --dry-run
```

Pass advanced non-secret server options after a `--` separator, for example
`-- --max-num-seqs 4`. Use the service command's own `--host` and `--port`
options for binding; secret-bearing flags are intentionally rejected. The
installed plist contains only a stable `service run --config ...` invocation.
The versioned, root-owned JSON definition under
`/Library/Application Support/Rapid-MLX/Services/` is the effective source of
truth. It is readable for diagnostics but not writable without root and is
strictly validated to exclude secrets. Keeping it outside the Python runtime
also lets a venv rebuild leave service configuration untouched.

Change an installed definition as a two-step transaction. `configure` validates
and stages a candidate but does not disturb the running server. `apply` swaps
the candidate into place, restarts, and requires `/readyz`; if bootstrap or
readiness fails it restores the previous config and service.

An installation created by the original argv-in-plist service release has no
versioned config yet. Migrate it once with `service uninstall` followed by
`service install`; this preserves its models, caches, credentials, and logs.

```bash
sudo rapid-mlx service configure \
  --model qwen3.5-9b-4bit --port 9000 -- --max-num-seqs 8
sudo rapid-mlx service apply --dry-run
sudo rapid-mlx service apply
rapid-mlx service config
```

To clear advanced serve flags, use `service configure --clear-serve-args`.
Changing the service account or executable still requires uninstall/install,
because those are security boundaries rather than runtime preferences.

For API authentication, send the key over stdin to a private credential file.
It never appears in argv, the plist, shell history, `service config`, or status:

```bash
security find-generic-password -w -s rapid-mlx-api-key | \
  sudo rapid-mlx service credential set
sudo rapid-mlx service restart
rapid-mlx service credential status
```

Use `sudo rapid-mlx service credential unset` followed by restart to disable
authentication. The service refuses to start if the credential is a symlink,
owned by another uid, or accessible by group/others.

Day-to-day operations:

```bash
rapid-mlx service status                 # registration / pid / health / logs
rapid-mlx service status --json          # machine-readable
rapid-mlx service logs                   # tail daemon logs
rapid-mlx service logs --follow          # stream, across KeepAlive restarts
sudo rapid-mlx service restart           # kickstart + wait until healthy
```

The stable runtime captures server stdout and stderr through bounded rotating
logs. Defaults are 100 MiB per active stream, five backups, and seven-day
retention. Stage different limits with `service configure --log-max-mb`,
`--log-backup-count`, and `--log-retention-days`, then apply them normally.

Upgrade the service with the same health gate and rollback behavior. The
command freezes the working environment before stopping the server, runs the
package upgrade as the service account, diagnoses it with `doctor`, and only
accepts it after launchd `/readyz` succeeds. On failure it restores the frozen
environment and starts the previous service.

```bash
sudo rapid-mlx service upgrade --dry-run
sudo rapid-mlx service upgrade --version 0.13.5 --extras vision,embeddings
```

Declaring extras reasserts the appliance's intended optional features. The
rollback requirements snapshot is mode 0600 under the service config directory.

Remove the service (models, cache, and logs are left in place):

```bash
sudo rapid-mlx service uninstall         # bootout + remove plist
rapid-mlx service uninstall --dry-run    # print the removal steps first
```

A full appliance acceptance test is [`scripts/headless_service_smoke.sh`](../scripts/headless_service_smoke.sh),
which checks registration, process owner, liveness, readiness, model
inventory, and a real one-token completion. Run it after any reboot test.

> The CLI is macOS-only and requires an existing least-privilege account —
> it intentionally does **not** create system users. Programmatic account
> creation is out of scope for the first supported release.

## Before you begin

You need:

- an Apple silicon Mac running macOS 13 or later;
- a dedicated, non-administrator local account, shown below as `serveuser`;
- Rapid-MLX installed by that account with the one-line installer;
- the model downloaded once before the machine becomes headless;
- administrator access for `/Library/LaunchDaemons` and `launchctl`; and
- a recovery path if the service or network configuration is wrong.

### FileVault changes the boot guarantee

A LaunchDaemon does run without a GUI login, but only after macOS has booted
and the data volume is available. On a FileVault-protected Mac, a cold boot
after power loss can stop at the preboot unlock screen. No user daemon, SSH
server, or Rapid-MLX process can run before that unlock.

If recovery after a complete power loss must be unattended, decide explicitly
whether to disable FileVault or provide a physical/out-of-band unlock process.
Do not disable disk encryption merely to follow this guide. An authenticated
restart used by some managed updates is not a substitute for testing the cold
boot behavior of your own machine.

## 1. Prepare the service account

Log in as the service account while the machine still has a display. Run every
command in this section as that account, without `sudo`:

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
rapid-mlx doctor
rapid-mlx models
```

Start the intended model once so its weights are present in this account's
cache, then stop the foreground server:

```bash
rapid-mlx serve qwen3.5-4b-4bit --host 127.0.0.1 --port 8000
```

The daemon uses these service-account paths:

| Path | Purpose |
|---|---|
| `/Users/serveuser/.local/bin/rapid-mlx` | Stable CLI symlink created by the installer |
| `/Users/serveuser/.rapid-mlx/` | Rapid-MLX virtual environment and app state |
| `/Users/serveuser/.rapid-mlx-python/` | Standalone base Python, when the installer supplied one |
| `/Users/serveuser/.cache/huggingface/hub/` | Default model cache resolved from `HOME` |
| `/Users/serveuser/Library/Logs/Rapid-MLX/` | Recommended daemon logs |

Create the log directory as the service account:

```bash
mkdir -p "$HOME/Library/Logs/Rapid-MLX"
chmod 750 "$HOME/Library/Logs/Rapid-MLX"
```

## 2. Configure the LaunchDaemon

Copy the repository template and replace every occurrence of `serveuser`. Also
select the model and launch flags appropriate for the machine:

```bash
cp examples/launchd/com.rapidmlx.server.plist /tmp/com.rapidmlx.server.plist
sed -i '' 's/serveuser/YOUR_SERVICE_ACCOUNT/g' /tmp/com.rapidmlx.server.plist
plutil -lint /tmp/com.rapidmlx.server.plist
```

`launchd` does not expand `$HOME` or `~` in plist values. Keep executable,
working-directory, cache, and log paths absolute. The explicit `HOME` variable
is required: a process in the system launchd domain does not inherit the GUI
session's shell environment, and Hugging Face otherwise cannot reliably find
the service account's cache.

Switch to a separate administrator session for the remaining system setup.
Keep `serveuser` in every absolute path; do not substitute the administrator's
home directory. Install the validated file:

```bash
sudo install -o root -g wheel -m 644 \
  /tmp/com.rapidmlx.server.plist \
  /Library/LaunchDaemons/com.rapidmlx.server.plist
sudo launchctl bootstrap system \
  /Library/LaunchDaemons/com.rapidmlx.server.plist
```

The template uses unconditional `KeepAlive`. A crash or unexpected clean exit
is restarted, while `ThrottleInterval` prevents a broken configuration from
spawning more than once every ten seconds. Always use `bootout` before planned
maintenance so KeepAlive does not fight the operator.

Inspect the service and logs:

```bash
sudo launchctl print system/com.rapidmlx.server
tail -F /Users/serveuser/Library/Logs/Rapid-MLX/server.stdout.log \
        /Users/serveuser/Library/Logs/Rapid-MLX/server.stderr.log
```

`launchctl print` is diagnostic output, not a stable machine-readable API.

## 3. Network security

The template binds to `127.0.0.1` by default. That is safe for a local reverse
proxy or an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 serveuser@your-mac
```

Do not put an API key in `ProgramArguments` or the plist's
`EnvironmentVariables`. Arguments are visible to other local processes, and
launchd can display configured environment values through `launchctl print`
even when the plist itself is mode 0600.

For direct LAN or internet access, keep Rapid-MLX on loopback and put a
separately managed TLS-terminating, authenticating reverse proxy in front of
it. Restrict the proxy with the host or network firewall and a trusted-host
allowlist. This template deliberately does not provide a public-listener
recipe.

## 4. Verify the running appliance

Run the repository smoke test on the Mac. It checks the system-domain job,
process owner, liveness, readiness, model inventory, and one real completion:

```bash
export RAPID_MLX_SERVICE_USER=serveuser
export RAPID_MLX_SMOKE_MODEL=default
./scripts/headless_service_smoke.sh
```

For an authenticated service, provide the key through the environment. The
script puts it in a mode-600 temporary curl configuration, not argv or output:

```bash
export RAPID_MLX_API_KEY='your-key'
./scripts/headless_service_smoke.sh
```

Test KeepAlive without rebooting:

```bash
old_pid=$(launchctl print system/com.rapidmlx.server | \
  awk -F'= ' '/^[[:space:]]*pid =/{print $2; exit}')
sudo launchctl kill SIGTERM system/com.rapidmlx.server
for _ in {1..30}; do
  new_pid=$(launchctl print system/com.rapidmlx.server 2>/dev/null | \
    awk -F'= ' '/^[[:space:]]*pid =/{print $2; exit}')
  [ -n "$new_pid" ] && [ "$new_pid" != "$old_pid" ] && break
  sleep 1
done
[ -n "${new_pid:-}" ] && [ "$new_pid" != "$old_pid" ] || exit 1
./scripts/headless_service_smoke.sh
```

The smoke test waits up to 120 seconds for readiness and must pass again with a
new process. To test the actual requirement, disable automatic login, reboot
while physical recovery is available, leave the Mac at the login window, and
run the same smoke remotely.

For desktop Macs that must restart when power returns, review the current
settings and then enable Apple's power-loss restart option:

```bash
pmset -g custom
sudo pmset -a sleep 0 autorestart 1
```

This setting cannot bypass FileVault's preboot unlock.

## 5. Update without fighting KeepAlive

Run this section from the administrator session. Commands that modify the
application runtime explicitly switch back to `serveuser`; this prevents an
update from accidentally installing into the administrator's home. Record the
working state first. Include every optional extra the service uses; `pip freeze`
records versions but cannot reconstruct which extras you intended:

```bash
sudo -u serveuser -H /Users/serveuser/.local/bin/rapid-mlx --version
sudo -u serveuser -H /bin/bash -c \
  '/Users/serveuser/.rapid-mlx/bin/python -m pip freeze > "$HOME/rapid-mlx-before-upgrade.txt"'
sudo cp -p /Library/LaunchDaemons/com.rapidmlx.server.plist \
  /Library/LaunchDaemons/com.rapidmlx.server.plist.pre-upgrade
```

Then use this order:

```bash
sudo launchctl bootout system/com.rapidmlx.server
curl -fsSL https://rapidmlx.com/install.sh | sudo -u serveuser -H /bin/bash

# Re-assert the extras required by this appliance. Examples:
sudo -u serveuser -H /Users/serveuser/.rapid-mlx/bin/python \
  -m pip install --upgrade 'rapid-mlx[vision]'

sudo -u serveuser -H /Users/serveuser/.rapid-mlx/bin/rapid-mlx doctor || \
  echo 'Doctor reported issues; keep the daemon stopped and continue validation.'
plutil -lint /Library/LaunchDaemons/com.rapidmlx.server.plist
sudo launchctl bootstrap system \
  /Library/LaunchDaemons/com.rapidmlx.server.plist
./scripts/headless_service_smoke.sh
```

Investigate every Doctor issue, but do not use Doctor alone to accept or reject
the update. Some 0.13.3 runtime layouts can produce false import failures. The
API smoke test is the actionable gate: it proves that the daemon can load the
configured model and serve a request in its real launchd environment.

The installer upgrades a compatible `~/.rapid-mlx` environment in place, so
unrelated installed packages normally remain. It deliberately rebuilds the
venv when its Python is missing, broken, non-native, or too old; a rebuild does
not preserve optional extras. Re-installing the declared extras and running
Doctor are therefore required parts of the runbook, not optional cleanup.

If validation fails, keep the daemon booted out, inspect the stderr log, and
restore the previous application version plus the same extras before loading
the service again:

```bash
sudo launchctl bootout system/com.rapidmlx.server 2>/dev/null || true
sudo -u serveuser -H /Users/serveuser/.rapid-mlx/bin/python -m pip install \
  --force-reinstall -r /Users/serveuser/rapid-mlx-before-upgrade.txt
sudo cp -p /Library/LaunchDaemons/com.rapidmlx.server.plist.pre-upgrade \
  /Library/LaunchDaemons/com.rapidmlx.server.plist
sudo -u serveuser -H /Users/serveuser/.rapid-mlx/bin/rapid-mlx doctor || true
sudo launchctl bootstrap system \
  /Library/LaunchDaemons/com.rapidmlx.server.plist
./scripts/headless_service_smoke.sh
```

## 6. Stop or remove the service

Stop it for maintenance:

```bash
sudo launchctl bootout system/com.rapidmlx.server
```

Remove it permanently:

```bash
sudo launchctl bootout system/com.rapidmlx.server 2>/dev/null || true
sudo rm /Library/LaunchDaemons/com.rapidmlx.server.plist
```

Removing the plist does not delete the service account, model cache, virtual
environment, or logs.

## Known limits

- `rapid-mlx service` requires a pre-existing non-administrator service
  account and an administrator to run mutating commands; it does not create
  system users.
- Runtime stdout/stderr are bounded automatically, but an external collector is
  still recommended for durable centralized history and alerting.
- A full application update causes downtime while the daemon is booted out.
- FileVault-protected cold boots require an unlock before the system can reach
  the state in which this daemon runs.
- Enabling `autorestart` is not proof of recovery after an AC outage. Perform a
  controlled site acceptance test if power-loss recovery is a requirement.
