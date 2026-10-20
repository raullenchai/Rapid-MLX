# Share Compute with QuickSilver

Share Compute lets your Mac serve live QuickSilver requests and earn QuickSilver
API credit. Credit pays for API usage; it is not a cash payout. You need an
Apple Silicon Mac, a supported model that fits its memory, and a QuickSilver
provider key (`qsppk-`). The model weights are downloaded once if needed.

## Desktop

1. In **Settings → Experimental**, enable **Share Compute**.
2. Open **Share Compute → Live Pool** to see supported models and choose one
   that is available on your Mac. Download it if the app asks you to.
3. In **Settings → Experimental → Share Compute**, give this Mac a worker
   name and save your `qsppk-` provider key. Use a different name on each Mac.
   The key is stored in this Mac's Keychain, separately from the ledger key;
   app preferences contain no secret. You can also paste and save the key in
   the first connection sheet. Saving settings does not start sharing.
4. On **Share**, click **Start sharing** (or **Set up sharing** for first setup). Choose the supported model shown in Live Pool.
   Rapid unloads the current local model and runs an exclusive pool process:
   the sharing model is unavailable to local chat. Chat explains that sharing
   must stop before a local model can start; it does not offer a blocked Start.
   Repeat sessions start using
   the saved account or existing node, without another key-entry sheet.
5. Wait for **Your Mac is contributing** and **Provider: Connected**. The Mac
   can receive work while the app and share session remain running. Use
   **Stop sharing** to go offline and unload the sharing model. Your previous
   model stays unloaded, leaving the machine's resources available. Starting
   again reuses the same valid node registration; stopping is not deregistration.
   You can replace or remove the saved provider key in Settings while offline.
   Replacing the account key invalidates the old local registration proof before
   writing the new Keychain entry, so the next join registers the new account.
   If the login Keychain is missing, locked, or denies this app access, Rapid
   reports the problem without opening a system authentication/reset dialog.
   A valid cached node can still reconnect without reading the provider key.
   Removing the saved key does not revoke remote credentials; use QuickSilver's
   dashboard for revocation. Local node credentials are retained, but the GUI
   requires a provider key again after its registration proof is invalidated.

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

Enter the `qsppk-` key when prompted. `--worker` is this Mac's node name;
when omitted, it defaults to the hostname. Use a different name on each Mac.
The terminal reports registration or cached-node reuse, download progress,
model startup, warm-up, pool connection, and local request activity. Keep the
command running while online. Ctrl-C stops sharing and unloads its worker;
run the same command to return online using the cached node, without a key
prompt. `--reregister` repairs an expired or changed registration binding; it
is not required for ordinary offline/online use. The recommended CLI flow uses the interactive prompt, keeping the key out of
argv and shell history. GUI Keychain settings are not imported into CLI.
The Qwen3.8 27B pool defaults to the qualified TensorFold runtime (48 GB+ unified memory) in both CLI
and Desktop, including in-memory message-boundary prefix reuse. Desktop bundles
that runtime. For the curl installer (`curl -fsSL https://rapidmlx.com/install.sh | bash`),
install its qualified revision once into Rapid’s environment:

```bash
~/.rapid-mlx/bin/python -m pip install 'mlx==0.32.3' 'mlx-audio==0.5.3' \
  'tensorfold @ git+https://github.com/ashhart/TensorFold.git@cb2ebf0540f42604e2759b2ddef497861e928248'
```

For other Python installs (including pip and uv tool), use the same Python that
runs `rapid-mlx`; the runtime error prints the exact install command. The audio
pin keeps the base audio stack compatible with the qualified MLX version.
Homebrew does not bundle TensorFold and manages its own MLX dependency; use the
curl or Python install for this lane rather than modifying the Homebrew environment.

An unavailable or incompatible runtime prevents joining. The local TensorFold
backend serves text; requests requiring unsupported tools, media, grammar, or
sampling features return a retryable capacity refusal in pool mode, allowing
QuickSilver to use a compatible provider. Standalone TensorFold serving retains
its explicit 400 refusal for unsupported features. Reuse is local to the node: a new prefix,
restart, cache eviction, or node switch can still require cold prefill.
The qualified 27B backend advertises one concurrent request, matching its
single inference lane; `--max-num-seqs` values above one are rejected before
registration.
On a Mac below the TensorFold memory floor that still has enough memory for
`qwen3.8-27b-4bit` (the 32 GB recommendation tier and up), the CLI serves the
27B pool with `qwen3.8-27b-4bit` (MTP runtime, ordinary pool concurrency)
instead and prints one line naming the runtime it chose. Smaller Macs are
refused by the TensorFold memory-floor check instead.
An existing registration bound to the old ordinary 27B alias needs one fresh
registration for TensorFold (CLI: add `--reregister`; Desktop: reconnect using
the saved provider key, or save one in Settings if this is the first setup). Node credentials remain bound to the selected alias.
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
