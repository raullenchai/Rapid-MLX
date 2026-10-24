# SPDX-License-Identifier: Apache-2.0
"""One-time pointer under the interactive serve ready banner."""

import fcntl
import os
import sys

from rapid_mlx import update_refresh
from rapid_mlx.first_run import _state_dir
from rapid_mlx.runtime.optional_runtime import _running_in_desktop_sidecar
from rapid_mlx.telemetry.consent_decision import ProcessRole
from rapid_mlx.telemetry.consent_runtime import detect_role

NEWSLETTER_TEXT = (
    "  New models that run on your Mac, the day they drop: "
    "https://rapidmlx.com/newsletter"
)


def _interactive_stdout() -> bool:
    try:
        return sys.stdout.isatty() and os.tcgetpgrp(sys.stdout.fileno()) == os.getpgrp()
    except Exception:
        return False


def maybe_print_newsletter_hint(*, update_notice_shown: bool = False) -> None:
    """Print after readiness, then mark it shown; never interfere with serve."""
    try:
        if (
            update_notice_shown
            or "CI" in os.environ
            or not _interactive_stdout()
            or detect_role() is ProcessRole.SIDECAR
            or _running_in_desktop_sidecar()
        ):
            return

        marker = _state_dir() / "newsletter_hint_shown"
        marker.parent.mkdir(parents=True, exist_ok=True)
        # Serialize first serves so both cannot print before either marks it.
        with (marker.parent / "newsletter_hint.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if marker.exists() or update_refresh.notice_printed():
                return
            color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
            line = f"\x1b[2m{NEWSLETTER_TEXT}\x1b[0m" if color else NEWSLETTER_TEXT
            print(line, flush=True)
            try:
                marker.touch(exist_ok=True)
            except OSError:
                pass
    except Exception:
        pass
