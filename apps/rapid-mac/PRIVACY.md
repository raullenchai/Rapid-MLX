# Rapid-MLX Desktop — Privacy Policy

Last updated: 2026-07-28.

Rapid-MLX Desktop ("the App") is a local-first SwiftUI Mac client for the
`rapid-mlx` inference server. We designed it so that your prompts,
attachments, and model responses **never leave your Mac** unless you
explicitly send them somewhere (e.g. via the web-search tool, which
calls a third-party search provider you configured).

## What stays on your Mac

* Chat history (sessions, messages, attachments) — stored under
  `~/Library/Application Support/Rapid/sessions.json` and never
  transmitted off-device.
* API keys for tools (Brave / Tavily / etc.) — stored in macOS
  Keychain.
* Server settings, picker state, window layout — stored in
  UserDefaults under `com.rapidmlx.rapid`.

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
  * App version + macOS version + CPU architecture.
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

## Feedback you choose to submit

The Help menu and menu-bar tray include bug-report and feature-request forms.
Nothing is sent merely by opening a form. When you press **Send Feedback**, the
App sends the following to Sentry:

* The feedback text you entered and whether it is a bug report or feature
  request.
* Your email address, only if you choose to provide it.
* Standard Sentry app and device context: Rapid-MLX version and runtime
  details; macOS version; Mac model, architecture, and processor count;
  available and app memory; thermal and low-power state; and locale, calendar,
  24-hour-format preference, and time zone.

The form never reads or attaches prompts, model responses, chat history,
attachments, API keys, local file paths, screenshots, or server traffic. It
does not ask for your name. User-submitted feedback is separate from automatic
telemetry, so the anonymous-usage toggle does not block a submission you
explicitly initiate.

Sentry receives the same shared user label (`feedback`) for every submission.
The App sets this non-unique value to prevent the SDK from creating a distinct
per-install identifier.

The Sentry SDK is configured exclusively for these submissions. Its automatic
crash handling, app-hang tracking, performance tracing, sessions, metrics,
failed-request capture, swizzling, and breadcrumbs are disabled. Sentry is a
third-party processor and, like any network service, receives connection
metadata such as the source IP while handling the request. Sentry's processing
is governed by its privacy policy: https://sentry.io/privacy/.

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

`ServerLocator` picks a binary in this fixed order (see
`Sources/Rapid/Server/ServerLocator.swift`):

1. **`RAPID_BIN`** environment variable — for tests / dev overrides.
2. **In-app update override** — the in-app updater (Phase 4, not yet
   wired in the current release) and the slim-DMG bootstrapper (live
   since v0.8.12) both drop a newer `rapid-mlx` at
   `~/Library/Application Support/Rapid/runtime-override/rapid-mlx/bin/rapid-mlx`
   (the `rapid-mlx/` wrapper directory is the top-level entry of the
   sidecar tarball, preserved through extract + atomic publish; fixed
   in #430). Slot wins over the bundled sidecar when populated.
3. **Bundled sidecar** — `Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx/bin/rapid-mlx`,
   shipped inside the notarised app bundle once Phase 5 (CI release
   integration) lands. **Default winner when present.** The
   `ServerLocator` code already prepends this candidate (see
   `Sources/Rapid/Server/ServerLocator.swift` Phase 1 work); the
   binary itself is not yet shipping in the current release stream,
   so the chain falls through to slot 4 until then.
4. **PATH / Homebrew / pipx** — checked after the bundled slot, **or**
   when you opt out of the bundled binary via Settings → rapid-mlx →
   "Use my own rapid-mlx install". **In the current release stream
   this is the de-facto default**, since the bundled-sidecar slot
   above is empty until Phase 5 ships.

This means: **once Phase 5 ships, a hostile shim on your `$PATH`
will not serve your prompts by default.** The bundled binary,
signed and notarised as part of the Rapid-MLX Desktop release, will
take precedence over PATH / Homebrew. Until that release, the
chain currently resolves to whichever `rapid-mlx` your `$PATH` /
Homebrew Cellar exposes — so prefer the official Homebrew tap
(see "Verifying what's actually running" below) and avoid putting
arbitrary `rapid-mlx` shadows on your `$PATH`.

### Verifying what's actually running

* **Settings → rapid-mlx** surfaces the resolved binary path and the
  resolution source (bundled / user-installed). If the source isn't
  "bundled" and you didn't intend to opt out, flip the toggle back.
* **If you opted into a user-installed binary**, prefer the official
  Homebrew tap:
  `brew tap raullenchai/rapid-mlx && brew install raullenchai/rapid-mlx/rapid-mlx`.
  Homebrew enforces the formula's checksum, so a tampered download
  doesn't install silently.
* **Source is open** at
  [github.com/raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX);
  releases publish wheels + PyPI artifacts. Inspect the workflow
  that built a release if you need to audit the supply chain.
* **Audit your `$PATH`** with `which -a rapid-mlx` — relevant only if
  you've opted into the user-installed slot. The first hit wins from
  that slot.

For reports of malicious `rapid-mlx` distributions, follow the
disclosure process in `SECURITY.md`. The bundled sidecar is in scope
for Rapid-MLX Desktop's security report channel; user-installed
distributions are out of scope but we will help triage.

## Third-party services

* **Sentry Feedback** — receives only feedback you explicitly submit, the
  optional email you enter, and the standard app/device context listed above.
  Automatic Sentry diagnostics are disabled.
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
* **Tool providers you configure** — when you call a search / file /
  web tool, your prompt content is sent directly to that provider
  (Brave, Tavily, etc.) per their own privacy policy. Rapid-MLX Desktop
  is a passthrough.

## Contact

privacy@machinefi.com
