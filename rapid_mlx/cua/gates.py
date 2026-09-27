"""Consent gates: credential guard, sign-in gate, destructive-action gate.

Mirrors Orca's ACTION_POLICY idea in file-sentinel form: the agent never
touches credentials or payments; sign-in pages pause for a human APPROVE
marker; the run directory keeps the audit trail.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from rapid_mlx.cua.planner import SENSITIVE_RE

SIGN_IN_RE = re.compile(r"\b(sign.?in|log.?in|log.?on|登录|登入)\b", re.IGNORECASE)
FORBIDDEN_COMMERCE_RE = re.compile(
    r"\b(add.?to.?cart|checkout|check.?out|place.?order|buy.?now|支付|下单|购买)\b",
    re.IGNORECASE,
)


class ConsentError(RuntimeError):
    """Raised when a plan crosses a hard consent boundary."""


def check_plan_consents(plan: dict, target_label: str = "") -> None:
    """Hard stops. Nothing here is configurable — credentials are off-limits."""
    haystack = " ".join(
        [
            str(plan.get("step_instruction", "")),
            str(plan.get("text", "")),
            target_label,
        ]
    )
    if SENSITIVE_RE.search(haystack):
        raise ConsentError(
            "plan references credentials or payment secrets "
            f"(step: {plan.get('step_instruction', '')[:80]!r})"
        )
    if plan.get("action") == "fill" and FORBIDDEN_COMMERCE_RE.search(haystack):
        raise ConsentError(
            "fill must not target cart/checkout/payment controls "
            f"(label: {target_label[:80]!r})"
        )


def looks_like_sign_in(snapshot: dict) -> bool:
    parts = [
        f"{e.get('label', '')} {e.get('role', '')}"
        for e in snapshot.get("elements", [])
    ]
    haystack = " ".join(parts)
    return bool(SIGN_IN_RE.search(haystack)) and "AXSecureTextField" in haystack


def wait_for_human(run_dir: Path, marker: str, timeout: float) -> bool:
    """File-sentinel human gate. Returns True when approved in time."""
    path = run_dir / marker
    deadline = time.monotonic() + timeout
    print(f"[human-gate] waiting up to {int(timeout)}s: touch {path}", flush=True)
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(1.0)
    return False
