# Operations knowledge

Store repeatable runbooks here. Production-facing procedures must state
prerequisites, verification, observability, rollback, and authorization gates.

- [Merge on green](merge-on-green.md) — exact-head authorization, full main
  validation, freezes, recovery and rollback.
- [Image release dogfood matrix](image-release-dogfood-matrix.md) — exact-candidate
  real-weight, API, recovery, and Desktop acceptance for every built-in image
  alias.
- [2026-09-06 image release candidate dogfood](2026-09-06-image-release-candidate-dogfood.md)
  — commit-bound 10-alias API matrix and release-shaped Desktop receipt.
- [Headless LaunchDaemon qualification](2026-09-02-headless-launchdaemon-qualification.md)
  — physical-host boot, no-login, keepalive, restart, and rollback evidence for
  the always-on service.
- [Self-hosted Linux CI runners](linux-ci-tart-runners.md) — ephemeral tart
  VMs behind `CI_LINUX_RUNNER` / `CI_LINUX_LIGHT_RUNNER`: supervisor, drain/pause, rollback, and golden
  image rebuilds.
- [Model mirror operations](model-mirror.md) — drift auditing, intentional
  Hugging Face fallback, and safe R2 resync procedure.
