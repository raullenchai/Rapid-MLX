# SPDX-License-Identifier: Apache-2.0
"""One-time newsletter pointer printed under the interactive serve banner.

``rapid-mlx serve`` is the CLI's most-used entry point. After the ready
banner, a human at a terminal sees a single secondary line pointing at the
landing page where they can subscribe in the browser. The CLI never asks
for or stores an email address.

The pointer is deliberately conservative:

* once per machine, tracked by an empty marker file in the same
  ``~/.rapid-mlx/`` state directory as the other one-time tips
  (``rapid_mlx/first_run.py``);
* interactive terminals only — piped/redirected output, ``CI``, Desktop
  sidecars, ``rapid-mlx launch``/``start``/chat children and supervised
  servers stay byte-clean;
* never on a run that already showed the update-available notice (one
  promotion at a time);
* no network calls, no telemetry, no new flags or env vars, and every
  filesystem failure is swallowed so the server boot is never affected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEWSLETTER_URL = "https://rapidmlx.com/newsletter"
NEWSLETTER_TEXT = (
    f"  New models that run on your Mac, the day they drop: {NEWSLETTER_URL}"
)

# Empty marker file, in the same ``~/.rapid-mlx/`` state dir as the other
# one-time CLI tips. Presence means "already shown on this machine".
_MARKER_NAME = "newsletter_hint_shown"

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def marker_path() -> Path:
    """Return the one-time marker path via the existing state-dir helper."""
    from rapid_mlx.first_run import _state_dir

    return _state_dir() / _MARKER_NAME


def newsletter_line(*, color: bool) -> str:
    """Return the secondary banner line, optionally dimmed."""
    return f"{_DIM}{NEWSLETTER_TEXT}{_RESET}" if color else NEWSLETTER_TEXT


def should_show_newsletter_hint(
    *,
    stdout_isatty: bool,
    ci: bool,
    managed_spawn: bool,
    update_notice_shown: bool,
    marker_exists: bool,
) -> bool:
    """Pure gate for the one-time newsletter pointer.

    Every exclusion is fail-closed toward silence: a script, CI runner,
    Desktop/launch/chat child, a run that already showed the update notice,
    or a machine that has seen the pointer before all keep the terminal
    output unchanged.
    """
    if not stdout_isatty or ci or managed_spawn:
        return False
    if update_notice_shown or marker_exists:
        return False
    return True


def _stdout_isatty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _ci() -> bool:
    return bool(os.environ.get("CI"))


def _managed_spawn() -> bool:
    """Whether this serve was spawned by Desktop or another managed parent.

    Reuses the existing process-role detector: it treats Desktop sidecars
    (``RAPID_MLX_PROCESS_ROLE``) and parent-watchdog children
    (``RAPID_MLX_WATCHDOG_PPID``, stamped by ``launch``, ``start``, chat,
    ``run``, ``share`` and ``bench``) as ``SIDECAR``. If the detector itself
    fails, stay silent rather than guess.
    """
    try:
        from rapid_mlx.telemetry.consent_decision import ProcessRole
        from rapid_mlx.telemetry.consent_runtime import detect_role

        return detect_role() is ProcessRole.SIDECAR
    except Exception:
        return True


def _marker_exists() -> bool:
    try:
        return marker_path().exists()
    except Exception:
        return False


def _color_enabled() -> bool:
    """Dim only on a real terminal and only when color is not opted out."""
    if os.environ.get("NO_COLOR"):
        return False
    return os.environ.get("TERM", "") != "dumb"


def _claim_marker() -> None:
    """Create the marker after the line was printed; exclusive create wins."""
    marker = marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "x"):
        pass


def maybe_print_newsletter_hint(*, update_notice_shown: bool = False) -> bool:
    """Print the pointer once per machine; return whether it was printed.

    The marker is written only after the print succeeded. A marker write
    failure is silent and never affects ``serve``; it only means a later
    run may try again. Never raises.
    """
    try:
        if not should_show_newsletter_hint(
            stdout_isatty=_stdout_isatty(),
            ci=_ci(),
            managed_spawn=_managed_spawn(),
            update_notice_shown=update_notice_shown,
            marker_exists=_marker_exists(),
        ):
            return False
        print(newsletter_line(color=_color_enabled()), flush=True)
    except Exception:
        return False

    try:
        _claim_marker()
    except Exception:
        pass
    return True
