# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx service`` — supported headless macOS service lifecycle.

Codifies the documented launchd contract for running Rapid-MLX as an
unattended system daemon (the macOS analogue of an enabled ``systemd``
service) behind commands an operator can run:

* ``rapid-mlx service install`` — validate a least-privilege service
  account and model, write a deterministic LaunchDaemon plist to
  ``/Library/LaunchDaemons``, and bootstrap it.
* ``rapid-mlx service status`` — launchd state / PID / endpoint health /
  owner / log paths, in human or ``--json`` form.
* ``rapid-mlx service logs`` — tail the daemon stdout/stderr logs.
* ``rapid-mlx service restart`` — kickstart the daemon and wait for
  readiness.
* ``rapid-mlx service configure/apply`` — stage and health-gate versioned
  configuration with automatic rollback.
* ``rapid-mlx service credential`` — store API authentication outside the
  plist and process arguments.
* ``rapid-mlx service upgrade`` — snapshot the Python environment and restore
  it if the upgraded launchd service fails readiness.
* ``rapid-mlx service uninstall`` — ``bootout`` + remove the plist
  without touching models, cache, or logs.

This package is distinct from :mod:`vllm_mlx.service` (the engine's shared
helpers + post-processing pipeline). See GH issue #2859 for motivation and
the acceptance contract. The manual runbook that this command makes
unnecessary remains documented in
``docs/guides/headless-macos-service.md`` as a fallback/recovery path.
"""
