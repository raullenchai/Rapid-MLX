# Explicit trust for executable inputs

Owner: Atlas, with Vector (model loading) and Harbor/Pixel (distribution).
Audit baseline: `366975c0f438d83d305df694e6a2cbf106880f88`.

The October 22 report identified five real trust gaps. A model's `model_file`
can execute Python even when its path stays inside the model directory;
upstream mlx-lm and mlx-vlm do not govern this import with the tokenizer's
`trust_remote_code` flag. Tokenizer repository code was enabled by default.
CLI upgrades executed a mutable website script, and package-index failures
could silently switch installation to Git source. Desktop discovery accepted
a user-writable runtime merely because it was executable. Python build
archives were structurally checked but not checked against a reviewed digest.
Finally, the CLI's open-ended mlx-arsenal requirement differed from the desktop
pin. These are execution/trust weaknesses, not evidence that a package or
repository has been compromised.

The chosen boundaries are:

- Model repository Python requires explicit consent. Path containment remains
  a separate restriction and does not constitute consent. Rapid-owned model
  adapters can disable repository selectors and use their reviewed implementation.
- CLI upgrades use the existing environment's package manager. Installation
  failures remain failures; intentional Git installs identify a full commit.
- Current desktop releases use their bundled sidecar. The release workflow
  builds full apps, and the old bootstrap coordinator is absent from this
  baseline. An automatically selected legacy runtime in Application Support
  therefore has no place in current release discovery. `RAPID_BIN` remains an
  explicit developer choice.
- The embedded Python archive must match a digest recorded in the build recipe,
  including on cache hits. CLI and desktop use the same reviewed arsenal version.

These controls do not sandbox explicitly trusted Python or guarantee safety of
all transitive dependencies. A package version or Git commit pin is not a
publisher signature. Removing automatic writable-runtime discovery closes that
specific selection path; it does not claim protection from every attacker
already executing as the same macOS user.

Compatibility changes are intentional: custom-code models need opt-in, source
installations need a commit, and legacy slim app users must install the current
full desktop app. Production deployment and release authorization remain human
decisions. See the operational installer and sidecar guidance for migration and
rollback details.
