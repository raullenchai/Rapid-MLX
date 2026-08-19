# Rapid-MLX Desktop

A native SwiftUI Mac app for local LLMs, built on
[rapid-mlx](https://github.com/raullenchai/Rapid-MLX). It bundles the
inference engine as a sidecar — no separate install, no Python to manage —
and gives you a ChatGPT / Ollama-style experience that runs entirely on your
Apple-Silicon Mac. Your weights, your prompts, your data, on device.

Open source (Apache-2.0), part of the [`raullenchai/Rapid-MLX`](https://github.com/raullenchai/Rapid-MLX)
monorepo.

## Install

Download the latest `.dmg` from
[GitHub Releases](https://github.com/raullenchai/Rapid-MLX/releases?q=rapid-mac),
open it, and drag **Rapid-MLX Desktop** to Applications.

> The `rapid-mlx` name belongs to the **engine**, not this app:
> `pip install rapid-mlx` and `brew install rapid-mlx` (in homebrew-core) both
> install the command-line inference server. The desktop app is distributed
> **only** as a direct download — it bundles the engine, so there is nothing
> else to install.

## What it does

- **Chat, local-first.** Streaming responses with a `thinking` / `content`
  split for reasoning models (Qwen 3.5 / 3.6, GLM 4.7, DeepSeek V4). Markdown
  and code rendering, copy-as-markdown, copy-code.
- **Document analysis in chat.** Attach text-based PDF, CSV, and TXT files to a
  normal conversation, then summarize, compare, or ask follow-up questions.
  Text extraction and parsing happen locally before the model sees the source.
- **Two-step model lifecycle.** The composer's model button names one action
  at a time: an uncached model shows **Download** (fetch the weights only),
  the same button becomes **Start** once they are on disk, and a running model
  shows **Stop model**. Browse cached vs. all models with per-alias size +
  memory hints and switch freely.
- **Connect your agents.** The bundled server speaks the OpenAI and Anthropic
  wire formats on `127.0.0.1`, so any editor or agent that accepts a custom
  base URL can use your local model for free. The Launch section gives you
  copy-paste config per tool.
- **Conversation history.** A sidebar of past chats, persisted privately on
  device (owner-only permissions), with a fresh "New chat" a click away.
- **Images tab.** A dedicated image-generation surface with the same compose
  box as chat — pick an image model, prompt, and generate locally.
- **Audio & voice tab.** Transcription, voice listing, and speech synthesis,
  driven by the sidecar's local OpenAI-compatible audio routes.
- **Built-in tools with approval.** Web search, page browsing, and weather are
  available to the model as tools; browsing is gated behind an approval prompt
  before any network fetch.
- **MCP connectors.** Attach external Model Context Protocol servers to expose
  their tools to your local model.
- **Folders & export.** Organize conversations into folders in the sidebar and
  export a conversation to a file via a save panel.
- **Custom instructions.** A persistent system-prompt preface applied across
  conversations.
- **Per-model performance settings.** Per-alias runtime overrides (persisted
  across launches) resolved into the sidecar's `serve` arguments at spawn.
- **Bundled engine.** The `rapid-mlx` engine ships inside the app as a
  sidecar; the app spawns and supervises it, reaps it cleanly on quit, and
  surfaces Downloading / Preparing / Warming-up phases so a slow first load
  never looks like a crash.

## Privacy

- **Opt-in telemetry.** Off until you accept on first run, toggleable in
  Settings → Privacy. Events go to `telemetry.rapidmlx.com` (a Cloudflare
  Worker → R2) under one shared anonymous client ID so the app and the
  embedded engine never double-count an install. Paths are PII-redacted
  (`/Users/<name>/` scrubbed); chat content and attachments are never sent.
- **Self-update.** Signed releases use Sparkle to check an EdDSA-signed appcast,
  download updates in the background, and install them when Rapid-MLX quits.
  During the migration the existing six-hour version-status request remains;
  it sends only the app version and records aggregate counts without storing
  the IP. Opt out of both requests with `RAPIDMLX_NO_UPDATE_CHECK=1` or
  `DO_NOT_TRACK=1`.

See [PRIVACY.md](PRIVACY.md) for the full field list, reset behavior, and
retention.

## Build from source

```bash
bash scripts/build.sh
open "build/Rapid-MLX Desktop.app"
```

`build.sh` compiles the SwiftUI executable and bundles the `rapid-mlx`
sidecar (a python-build-standalone CPython + the engine + its deps) into
`Contents/Resources/rapid-mlx/`. Set `SKIP_SIDECAR=1` for a fast dev build
without the ~5–10 min sidecar step. Requires a recent Swift toolchain on
macOS 14+.

## Test

```bash
bash scripts/smoke.sh      # fast end-to-end smoke against a fake rapid-mlx
swift test                 # unit + snapshot suite

# With a built app already running, exercise real macOS Accessibility UI.
# Requires Peekaboo + jq and never starts or downloads a model.
bash scripts/gui-ax-smoke.sh
```

`gui-ax-smoke.sh` is the deterministic layer of GUI dogfood: it reads the
Accessibility tree, targets stable identifiers, invokes native AX actions,
and saves both JSON trees and screenshots under `/tmp`. Coordinate input is
reserved for explicitly logged fallbacks when a system/SwiftUI surface does
not expose a usable AX action. Exploratory agents can use the same Peekaboo
commands, while assertions in this script remain model-independent.

## Architecture

```
RapidApp (SwiftUI Scene root)
├── ContentView              — NavigationSplitView: sidebar + detail
│   ├── SidebarView          — New Chat · Launch · conversation history
│   ├── ChatView             — single-column transcript + compose box
│   │   └── ModelPickerBar   — inline model picker (Download / Start / Stop)
│   └── ConnectToolsView     — "Launch": copy-paste tool configs
└── MenuBarController        — AppKit NSStatusItem tray: icon + update CTA + Quit
                               (no SwiftUI MenuBarExtra — its glyph fails to
                               render on macOS 26, see #502)
```

State holders are `@Observable` classes injected via SwiftUI `environment`
(`ServerManager`, `ChatViewModel`, `DownloadManager`, `QuickstartCoordinator`,
`SamplingConfig`, `AppearanceConfig`, `SettingsRouter`, `UpdateChecker`, …).
The bundled sidecar is resolved by `ServerLocator`
(`RAPID_BIN` override → app-managed runtime-override → the bundled engine).

## Releasing

Pushing a `rapid-mac-v*` tag triggers
[`.github/workflows/rapid-mac-release.yml`](../../.github/workflows/rapid-mac-release.yml):

1. Imports the `Developer ID Application` certificate into a temp keychain
2. `bash scripts/build.sh` → `build/Rapid-MLX Desktop.app` (signed, hardened runtime)
3. `xcrun notarytool submit` via the App Store Connect API key
4. `xcrun stapler staple` on the `.app`
5. `bash scripts/dmg.sh` → `rapid-mlx-desktop.dmg`, notarised + stapled
6. Attaches the `.dmg` to the GitHub Release for the tag

Required repo secrets: `MACOS_DEVID_CERT_P12_BASE64`,
`MACOS_DEVID_CERT_PASSWORD`, `APPLE_TEAM_ID`, `AC_API_KEY_ID`,
`AC_API_ISSUER_ID`, `AC_API_KEY_P8_BASE64` (see [RELEASING.md](RELEASING.md)).
`workflow_dispatch` runs the same build but uploads a workflow artifact
(no Release) for signing dry-runs.

## Security

Reporting a vulnerability: see [SECURITY.md](SECURITY.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
Third-party components: [THIRD_PARTY.md](THIRD_PARTY.md).
