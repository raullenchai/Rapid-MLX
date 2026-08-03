# Rapid-MLX Desktop — Third-Party Acknowledgements

Rapid-MLX Desktop incorporates open-source software from the following
projects. Each is used under the terms of its original license. The
full license texts are reproduced in the linked repositories.

## Swift packages

* **swift-testing** — Apple Inc. — Apache License 2.0
  https://github.com/swiftlang/swift-testing
  Used by the test target only.

* **ViewInspector** — Alexey Naumov — MIT License
  https://github.com/nalexn/ViewInspector
  Used by the test target only — SwiftUI view-tree introspection.

* **swift-markdown-ui** — Guillermo González Real — MIT License
  https://github.com/gonzalezreal/swift-markdown-ui
  Block-level markdown rendering for assistant messages.

* **sentry-cocoa** — Functional Software, Inc. dba Sentry — MIT License
  https://github.com/getsentry/sentry-cocoa
  Delivery of feedback that a user explicitly submits from the app.

_Sparkle is on the roadmap as the eventual auto-update framework
(see [issue #16](https://github.com/raullenchai/Rapid-MLX/issues/16))
but is not bundled in the current release stream. The shipped
updater is the in-tree `Sources/Rapid/Updater/UpdateChecker.swift`
+ `Installer.swift` pair, which polls a Cloudflare Worker proxy of
GitHub Releases and surfaces an "Update available" CTA — see
[PRIVACY.md](PRIVACY.md#third-party-services). This file will list
Sparkle once it actually lands in `Package.resolved`._

## Assets

* **Cheetah mascot** — derived from the `rapidmlx.com` landing-page
  assets. © 2026 MachineFi. Embedded in the app bundle (not the
  source tree under an OSS license).

## Server dependencies (informational)

The `rapid-mlx` server that Rapid-MLX Desktop talks to is a separate
project licensed under Apache 2.0:
https://github.com/raullenchai/Rapid-MLX

It in turn vendors mlx-lm, mlx-vlm, and openai-harmony — each under
their respective Apache 2.0 / MIT licenses. Those are NOT bundled
inside Rapid-MLX Desktop and are installed separately via `pip` /
`pipx` / `brew`.

For the complete machine-readable resolution of pinned versions,
see `Package.resolved` at the repository root.
