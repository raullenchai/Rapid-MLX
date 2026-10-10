# Share Compute with QuickSilver

Share Compute lets your Mac serve live QuickSilver requests and earn QuickSilver
API credit. Credit pays for API usage; it is not a cash payout. You need an
Apple Silicon Mac, a supported model that fits its memory, and a QuickSilver
provider key (`qsppk-`). The model weights are downloaded once if needed.

## Desktop

1. In **Settings → Experimental**, enable **Share Compute**.
2. Open **Share Compute → Live Pool** to see supported models and choose one
   that is available on your Mac. Download it if the app asks you to.
3. On **Share**, click **Connect & serve**. Give each Mac on the same provider
   account a distinct worker name. Paste your provider key in the confirmation
   sheet and click **Connect & Share**. Rapid sends this key directly to the
   local provider process; it does not save the provider key in app settings.
4. Wait for **Your Mac is contributing** and **Provider: Connected**. The Mac
   can receive work while the app and share session remain running. Use
   **Stop sharing** to leave the pool. Rapid requests a restart of your previous
   local model; check that it becomes ready before using it again.

For earnings, open **Credits** and save a separate QuickSilver read-only key
(`qsprk-`). Rapid stores it in this Mac's Keychain. The credit ledger shows
settled accounting windows across all nodes on your provider account, while
**Local activity** records only sessions on this Mac. A request can succeed
before its credit appears: settlement closes a window later. Calls paid for
by the node owner's own account do not earn contributor credit.

## Terminal

An interactive first run keeps the provider key out of the command line and
shell history:

```bash
rapid-mlx share qwen3.8-27b --quicksilver --worker "$(hostname -s)"
```

Enter the `qsppk-` key when prompted. Later runs reuse the local node
registration. Choose a different `--worker` on each Mac. Stop with Ctrl-C.
Only models enabled in QuickSilver's current pool can receive requests; the
Desktop **Live Pool** view shows the current availability.

Pool mode always disables optional disk caches. Before starting the local
server, it removes the selected model's old automatic prompt snapshots,
including interrupted saves, from `~/.cache/rapid-mlx/prefix_cache/`. Other
models and downloaded weights are preserved. Stop any other server using that
model before sharing; a busy cache, ambiguous legacy ownership, or failed cleanup prevents the node from
joining the pool. Local serving outside pool mode can still persist prompt
caches; new snapshots use private directories and files (0700 and 0600).
Snapshots whose model ownership cannot be established from their full
fingerprint require manual review and removal before sharing can start.

The API and relay use verified TLS with system roots and the bundled CA list.
Certificate verification failures stop registration or relay connection
immediately. Update Python's CA certificates and `certifi`, or use
`SSL_CERT_FILE` for a trusted private CA bundle, then restart sharing.

After registering once, install and start a per-user macOS background service:

```bash
rapid-mlx share qwen3.8-27b --quicksilver --worker "$(hostname -s)" --install-service
```

The job uses the installed `rapid-mlx` launcher, reuses the private node
registration, and starts automatically. Reinstalling replaces and restarts the
job. A failed `launchctl bootstrap` is reported as an error; the written plist
remains available for diagnosis. The command prints how to stop the job.
