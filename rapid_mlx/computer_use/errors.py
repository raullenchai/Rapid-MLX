"""Typed computer-use errors with agent-facing recovery hints.

Every failure surfaced to an agent carries a machine-readable code and a
short recovery hint so the agent can self-correct instead of retrying
blindly. Codes follow a taxonomy of actionable failure classes.
"""

from __future__ import annotations

RECOVERY_HINTS: dict[str, tuple[str, ...]] = {
    "app_not_found": (
        "Run `computer list-apps --json` to see running apps,",
        "then retry with the exact name, bundle id, or pid:N.",
    ),
    "window_not_found": (
        "Run `computer list-windows --app <app> --json` and target",
        "an existing window, or bring the app to the foreground.",
    ),
    "element_not_found": (
        "The AX tree shifted; re-run `computer get-app-state --app <app>`",
        "and use a fresh element index from the new snapshot.",
    ),
    "value_not_settable": (
        "This element rejects direct AX value writes;",
        "use `computer type-text` after clicking it, or pick a text-field element.",
    ),
    "screenshot_failed": (
        "Grant Screen Recording permission to the host terminal app",
        "or retry with --no-screenshot to use the AX tree only.",
    ),
    "permission_denied": (
        "Grant Accessibility permission to the host terminal app",
        "(System Settings > Privacy & Security > Accessibility),",
        "then retry. `computer permissions --json` reports status.",
    ),
    "automation_permission_required": (
        "Allow Rapid-MLX to control the selected browser in System Settings",
        "> Privacy & Security > Automation, then retry the task.",
    ),
    "window_stale": ("The window was replaced or moved; re-run get-app-state.",),
    "action_timeout": (
        "The app did not settle in time; retry once, then re-observe",
        "with get-app-state instead of repeating the same action blindly.",
    ),
    "unsupported_key": (
        "press-key accepts a single key (Return, Escape, Tab, Delete,",
        "arrows, Space, a-z, 0-9); use hotkey for modifier chords like Cmd+A.",
    ),
    "invalid_argument": ("Fix the flagged argument and retry.",),
    "stale_snapshot": (
        "The cached snapshot expired; re-run get-app-state for fresh indexes.",
    ),
    "stale_observation": (
        "The observation is too old or lacks a stable window identity;",
        "re-run get-app-state and retry with its window_id and element index.",
    ),
    "target_drift": (
        "The selected window moved, resized, or lost focus;",
        "re-run list-windows and get-app-state before retrying.",
    ),
    "target_stale": (
        "The planned control changed before input could be dispatched;",
        "re-observe the selected window and create a fresh plan.",
    ),
    "target_occluded": (
        "Another window covers the action point; bring the selected window",
        "to the foreground, re-run get-app-state, and retry.",
    ),
    "unsupported_platform": ("Run computer-use on a Mac with PyObjC installed.",),
}


class ComputerUseError(Exception):
    """Typed computer-use failure with an agent-facing recovery hint."""

    def __init__(
        self, code: str, message: str, recovery: tuple[str, ...] | None = None
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.recovery = recovery or RECOVERY_HINTS.get(code, ())

    def to_payload(self) -> dict:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "recovery": list(self.recovery),
            },
        }
