# Headless macOS service

Run Rapid-MLX as a system `launchd` daemon when a Mac is used as an unattended
inference appliance. A system daemon starts after macOS boots and does not
depend on a graphical login. This is the macOS equivalent of an enabled
`systemd` service.

> **Operational scope:** this guide uses the existing CLI and Apple's
> `launchd`. Rapid-MLX does not yet install or manage the daemon for you. Test
> the procedure while you still have physical or out-of-band access to the Mac.

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

- There is no `rapid-mlx service install` command yet; this is an operator-run
  deployment.
- Rapid-MLX does not rotate the two log files. Configure your existing log
  collector or rotation policy and monitor free disk space.
- A full application update causes downtime while the daemon is booted out.
- FileVault-protected cold boots require an unlock before the system can reach
  the state in which this daemon runs.
- Enabling `autorestart` is not proof of recovery after an AC outage. Perform a
  controlled site acceptance test if power-loss recovery is a requirement.
