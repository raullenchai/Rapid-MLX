# Rapid-MLX Desktop — Privacy Policy

Last updated: 2026-08-18.

Rapid-MLX Desktop ("the App") is a local-first SwiftUI Mac client for the
`rapid-mlx` inference server. We designed it so that your prompts,
attachments, and model responses **never leave your Mac** unless you
explicitly send them somewhere (e.g. via the web-search tool, which
sends the search query to a third-party search provider — Keenable's
keyless endpoint by default; you can switch the backend or add your
own key in Settings → Tools).

## What stays on your Mac

* Chat history (conversations, messages, attachments, folders) — stored
  under `~/Library/Application Support/Rapid/conversations.json` and
  never transmitted off-device.
* API keys for tools (Keenable / Parallel / Tavily / Brave) — stored
  in macOS Keychain.
* Server settings, picker state, window layout — stored in
  UserDefaults under `com.rapidmlx.rapid`.
* The engine's **prompt prefix cache** — the embedded `rapid-mlx` server
  keeps reusable KV state derived from your prompts on disk (under
  `~/.cache/rapid-mlx/prefix_cache/`) so repeated context prefills
  faster. It stays on-device and is derived data, not transcript text,
  but it is computed from your prompts: you can turn it off per model in
  Settings → Performance, or for any server with
  `rapid-mlx serve --disable-prefix-cache`, and delete the directory at
  any time.

## What we collect (telemetry)

Anonymous, opt-in usage telemetry. Default: **off until you make an
explicit choice** in the first-run disclosure. You can change that choice
at any time in Settings → Privacy. When enabled, the desktop app and its
embedded `rapid-mlx` engine use the same consent record and random client ID,
so one Mac is not counted as two installs. We collect:

* `session_start` — once per app launch. Includes:
  * `client_id` — random UUID stored locally at
    `~/.rapid-mlx/telemetry-client-id`; never tied to an account, device
    serial, or hardware identifier. It is shared only between the desktop app
    and embedded engine for anonymous de-duplication.
  * `session_id` — random UUID, regenerated on every launch.
  * App version, macOS version, CPU architecture, Apple chip family (or the
    coarse label `Intel` on legacy Macs), and memory tier rounded to the nearest
    GiB. No serial number, exact Intel CPU SKU, clock speed, or exact byte count.
* `error` — when the app crashes or hits an unhandled exception.
  Includes:
  * Error type + message (e.g. `EXC_BAD_ACCESS`, `Bundle.module
    fatalError`).
  * Truncated stack trace (top 30 frames, file/line, no captured
    locals).
  * App version + macOS version + a short context label (e.g.
    `chat_send`, `download_install`).
* Embedded-engine `session_start`, `session_end`, `request`, and `error`
  events — only after the same opt-in. Depending on the bundled engine
  version, these can include:
  * Rapid-MLX version, macOS version, CPU architecture, chip family, memory
    tier, and Python version.
  * Public model aliases, subcommand and feature/flag names (never values).
  * Request endpoint, streaming/tool-use booleans, HTTP status, and coarse
    buckets for token counts, time to first token, and decode speed.
  * Closed-set error categories and non-reversible stack fingerprints; no
    exception message or raw traceback.

Anonymous telemetry does **not** collect:

* Your name, email, IP address, or device identifier.
* Prompts, model responses, attachments, unredacted user paths, or anything
  you type into the app. Crash diagnostics can contain paths after usernames
  and temporary-container identifiers have been replaced with `<redacted>`.
* Prompt or response contents from traffic to your `rapid-mlx` server.
* Tool API keys.

The telemetry endpoint is `https://telemetry.rapidmlx.com/v1/events`.
The receiving Cloudflare Worker strips client IPs before writing to
storage. Source is open: `github.com/raullenchai/rapidmlx.com` under
`telemetry-worker/`.

## Opt out

Settings → Privacy → "Send anonymous usage data" → off. Takes effect
immediately for both the desktop app and its embedded engine; no further
events are sent. Already-sent events cannot be retroactively deleted because
they are not associated with your identity, but the rolling 30-day raw-event
storage window means they age out. Running `rapid-mlx telemetry reset` removes
the shared consent and random client ID; the desktop app will ask again rather
than restoring the old ID.

## Feedback

The App contains **no feedback form and no crash-reporting SDK** (earlier
releases embedded Sentry for an explicit feedback form; it has been removed
and the App links no Sentry code). To report a bug or request a feature,
open an issue at
[github.com/raullenchai/Rapid-MLX/issues](https://github.com/raullenchai/Rapid-MLX/issues)
— what you share there is governed by GitHub, not by the App.

## Local inference server (`rapid-mlx`) — trust boundary

Rapid-MLX Desktop is a **client**. The actual inference work — loading the
model, tokenizing your prompt, generating the response — runs inside a
separate `rapid-mlx` process on your Mac. The app talks to it over
`http://127.0.0.1:<port>` (loopback only; never exposed off-device by
Rapid-MLX Desktop itself).

**Every prompt, every attachment, every chat turn flows through the
`rapid-mlx` binary running on your Mac.** That binary loads the model
from disk, has access to its own working directory, and can write to
its own log files. A malicious build of `rapid-mlx` could:

* Read every prompt and response that flows through the chat surface.
* Exfiltrate prompt content over any outbound network connection it
  chooses to open. (Rapid-MLX Desktop does not restrict the
  subprocess's network access.)
* Return tampered model output — inject false answers, or leak
  prompts back through carefully crafted replies.
* Read locally cached model weights on disk.

### How Rapid-MLX Desktop resolves the `rapid-mlx` binary

`ServerLocator` resolves the binary in two tiers (see
`Sources/Rapid/Server/ServerLocator.swift`):

1. **`RAPID_BIN`** environment variable — explicit dev/test override;
   when present in the app's launch environment it wins
   unconditionally, so power users can point the app at a checkout.
2. **Managed sidecars — newest version wins.** The two app-managed
   slots are compared by their `VERSION` files when both exist:
   * **Runtime-override** —
     `~/Library/Application Support/Rapid/runtime-override/rapid-mlx/bin/rapid-mlx`,
     the canonical install path: the slim-DMG bootstrapper and the
     sidecar updater publish the engine here atomically.
   * **Bundled sidecar** —
     `Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx`,
     shipped inside the notarised app bundle on full-bundle builds.

   The runtime override wins when it is the same version or newer; a
   stale or unversioned runtime override cannot shadow a newer,
   versioned sidecar shipped by an app update.

**A `rapid-mlx` on your `$PATH` (Homebrew, pipx, uv, anything else) is
intentionally never consulted.** The PATH fallback was removed in the
v0.8.10 cutover: the desktop and CLI versions would drift silently, and
the app's "up to date" claim would lie about whichever copy actually
answered. A hostile shim on `$PATH` therefore never serves your prompts.
The honest residual risk is the runtime-override slot itself. It is an
**executable-code trust boundary**, not ordinary app data: it lives in
your user-writable Application Support directory, and whatever binary
sits there is what the app launches and hands every prompt to. Malware
that already has user-level write access could plant a binary there
and gain code execution with your privileges the next time the app
starts — reading every prompt and response, your files, and anything
else your account can reach.

### Verifying what's actually running

* The only mutable **app-managed** slot is
  `~/Library/Application Support/Rapid/runtime-override/` — inspect it
  (or delete it; the app re-provisions from the official channel).
* **`RAPID_BIN` outranks both managed slots**, so also confirm it is
  not set in the environment the app launches from. Anything already
  running as your user — you, a shell profile, a launch agent, or
  user-level malware via `launchctl setenv` — can set it, so treat an
  unexpected `RAPID_BIN` as a red flag, not a curiosity: check with
  `launchctl getenv RAPID_BIN`, unset it (`launchctl unsetenv
  RAPID_BIN`, and remove it from any shell/launch-agent config), and
  relaunch.
* **Source is open** at
  [github.com/raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX);
  releases publish wheels + PyPI artifacts. Inspect the workflow
  that built a release if you need to audit the supply chain.
* Want the CLI on your `$PATH` for terminal use? `brew install
  rapid-mlx` (Homebrew core) — it is a separate install and does not
  affect which binary the desktop app runs.

For reports of malicious `rapid-mlx` distributions, follow the
disclosure process in `SECURITY.md`. The bundled sidecar is in scope
for Rapid-MLX Desktop's security report channel; user-installed
distributions are out of scope but we will help triage.

## Third-party services

* **Telemetry collector** — Cloudflare Workers + R2 storage. Subject
  to Cloudflare's data processing terms.
* **Auto-update channel** — signed production builds use Sparkle to poll
  `https://dl.rapidmlx.com/appcast.xml` every six hours. Sparkle sends a
  standard app/version user agent but no system profile
  (`SUSendsSystemProfile=false`), unique identifier, chat content, or usage
  data. It verifies every update archive with the Ed25519 public key embedded
  in the signed app, downloads in the background when enabled, and installs
  the prepared update when Rapid-MLX next quits normally.

  During the migration, ``UpdateChecker`` also polls
  `https://rapidmlx.com/api/desktop-update?v=<app-version>` on launch
  and every 6 hours while running. This is a thin Cloudflare Worker
  that returns the **byte-identical** release manifest the public R2
  object (`https://dl.rapidmlx.com/latest.json`) serves, while also
  counting the poll as an aggregate active-install signal. **The only
  data the request sends is the app's own version** (the `?v=` query,
  the same value already in the `User-Agent: Rapid-Desktop/<version>`
  header) — no device fingerprint, no unique ID, nothing about your
  usage. The Worker records only aggregate counts (version and
  Cloudflare-derived country) and **never stores your IP address**.
  You can turn the check off entirely: set the environment variable
  `RAPIDMLX_NO_UPDATE_CHECK=1` (or the cross-tool `DO_NOT_TRACK=1`),
  which skips the request completely — no poll, no signal. No GitHub
  API call, no PAT, no proxy of GitHub — those were the v0.5 shape and
  were retired in v0.6.12 (PR rapid-desktop#225 + rapidmlx.com#8). When a newer release is available the app
  surfaces the existing in-app version status. **The app never downloads or
  installs an update itself.** Everything past "a newer version exists" is
  Sparkle's, and only on builds carrying an injected Sparkle public key
  (signed releases):
  * **Ed25519 signature over every downloaded update.** Sparkle fetches its
    own appcast over HTTPS and refuses any payload whose EdDSA signature does
    not verify against the public key baked into the app bundle at build time.
  * **Apple code-signing continuity.** Before replacing the bundle Sparkle
    checks that the update is signed by the same identity as the running app.
    Production releases are **Developer ID Application signed** (team
    `73WQ7ZGSWC`, MachineFi Inc.), hardened-runtime, notarised and stapled,
    so an update signed by any other team is rejected.
  * **Installed on quit, never over a running bundle**, and Sparkle handles
    the authorisation prompt itself when `/Applications` is not writable.
  * **Unsigned and local development builds have no updater at all.** With no
    public key there is nothing to verify against, so those builds only report
    version status; there is no in-app download or install path to fall back
    to.

  If you want to side-step the in-app updater entirely you can
  download the DMG from the
  [GitHub Releases page](https://github.com/raullenchai/Rapid-MLX/releases/latest)
  and verify its origin yourself before installing.
* **Search and tool providers** — when the model calls the
  web-search tool, the search query (never your whole conversation)
  is sent to the configured provider per their own privacy policy.
  The default backend is Keenable's keyless endpoint
  (`api.keenable.ai`) — no account, no API key; the request carries
  the query plus standard connection metadata (such as your IP
  address), nothing more. You can switch to Parallel, Tavily,
  Brave, or the DuckDuckGo scrape — or add your own key — in
  Settings → Tools. When Keenable is unreachable the tool retries the
  query against DuckDuckGo. Rapid-MLX Desktop is a passthrough.

## Contact

privacy@rapidmlx.com
