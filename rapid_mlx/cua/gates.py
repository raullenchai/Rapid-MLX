"""Consent gates: hard stops and approval for consequential actions.

The agent never touches credentials or payments, and never clicks or fills
cart/checkout controls (research-only v1). Sign-in pages pause for a human
APPROVE marker; the run directory keeps the audit trail.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path

from rapid_mlx.cua.planner import SENSITIVE_RE

SIGN_IN_RE = re.compile(
    r"(?:\b(?:sign.?in|log.?in|log.?on)\b|登录|登入)", re.IGNORECASE
)
FORBIDDEN_COMMERCE_RE = re.compile(
    r"(?:\b(?:add.?to.?cart|checkout|check.?out|place.?order|buy.?now)\b|支付|下单|购买)",
    re.IGNORECASE,
)

# These labels describe controls which commit an effect outside the local draft.
# Keep this deterministic and deliberately conservative: an uncertain commit is
# safer to pause than to infer from application-specific behavior.
CONSEQUENTIAL_RE = re.compile(
    r"(?:\b(?:send|submit|post|publish|delete|remove|trash|discard|erase|"
    r"confirm|book|reserve|schedule|share|upload|reply|comment|invite|"
    r"unsubscribe|enviar|publicar|eliminar|envoyer|publier|supprimer|senden|"
    r"veröffentlichen|löschen)\b|发送|提交|发布|删除|移除|确认|预订|预约|分享|"
    r"上传|回复|评论|邀请|送信|投稿|公開|削除|보내기|게시|삭제)",
    re.IGNORECASE,
)
READ_ONLY_SUBMIT_RE = re.compile(
    r"(?:\b(?:search|find|filter|look\s*up|buscar|rechercher|suchen)\b|"
    r"搜索|查找|筛选|検索|검색)",
    re.IGNORECASE,
)
KEYBOARD_ACTIVATION_KEYS = {"enter", "return", "space"}


class ConsentError(RuntimeError):
    """Raised when a plan crosses a hard consent boundary."""


@dataclass(frozen=True)
class ApprovalRequirement:
    """A human approval request bound to one proposed action and target."""

    kind: str
    action: str
    target: str
    instruction: str

    @property
    def reason(self) -> str:
        return (
            f"{self.kind}: action={self.action}; target={self.target!r}; "
            f"proposed={self.instruction!r}"
        )


def is_keyboard_activation(plan: dict) -> bool:
    """Whether a press can activate the currently focused control."""
    return plan.get("action") == "press" and str(plan.get("key", "")).casefold() in (
        KEYBOARD_ACTIVATION_KEYS
    )


def consequential_action(
    plan: dict,
    target_label: str = "",
    *,
    target_role: str = "",
    target_parent_role: str = "",
    app_name: str = "",
) -> ApprovalRequirement | None:
    """Return an exact approval request for an externally committing action."""
    action = str(plan.get("action", ""))
    instruction = str(plan.get("step_instruction", "")).strip()
    if action == "save":
        target = target_label.strip() or "selected document"
        return ApprovalRequirement(
            kind="external_commit",
            action=action,
            target=target[:160],
            instruction=instruction[:240],
        )
    if action not in {"click", "press"}:
        return None
    if is_keyboard_activation(plan):
        label = target_label.strip()
        role = target_role.strip()
        parent_role = target_parent_role.strip()
        key = str(plan.get("key", "")).casefold()
        finder_inline_rename = (
            app_name.casefold() == "finder"
            and key in {"enter", "return"}
            and (
                (role == "AXTextField" and parent_role == "AXCell")
                or (role == "AXRow" and parent_role == "AXOutline")
            )
        )
        read_only_search = role == "AXSearchField" and bool(
            READ_ONLY_SUBMIT_RE.search(label)
        )
        if finder_inline_rename or read_only_search:
            return None
        if not label or not CONSEQUENTIAL_RE.search(f"{label} {instruction}"):
            return ApprovalRequirement(
                kind="external_commit",
                action=action,
                target=label[:160] or "focused control (unverified)",
                instruction=instruction[:240],
            )
    target = target_label.strip() or f"element {plan.get('element_index', -1)}"
    haystack = f"{target_label} {instruction}"
    if not CONSEQUENTIAL_RE.search(haystack):
        return None
    # Search submission is a read-only navigation flow even when a planner
    # uses the generic word "submit" in its instruction.
    if READ_ONLY_SUBMIT_RE.search(target_label) and not CONSEQUENTIAL_RE.search(
        target_label
    ):
        return None
    return ApprovalRequirement(
        kind="external_commit",
        action=action,
        target=target[:160],
        instruction=instruction[:240],
    )


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
    if plan.get("action") in {
        "fill",
        "click",
        "press",
    } and FORBIDDEN_COMMERCE_RE.search(haystack):
        raise ConsentError(
            f"{plan.get('action')} must not target cart/checkout/payment "
            f"controls (label: {target_label[:80]!r})"
        )


def looks_like_sign_in(snapshot: dict) -> bool:
    parts = [
        f"{e.get('label', '')} {e.get('role', '')}"
        for e in snapshot.get("elements", [])
    ]
    haystack = " ".join(parts)
    return bool(SIGN_IN_RE.search(haystack)) and "AXSecureTextField" in haystack


async def wait_for_human(
    run_dir: Path, marker: str, timeout: float, *, reason: str = ""
) -> bool:
    """File-sentinel human gate. Returns True when approved in time."""
    path = run_dir / marker
    path.unlink(missing_ok=True)
    deadline = time.monotonic() + timeout
    detail = f" for {reason}" if reason else ""
    print(
        f"[human-gate] waiting up to {int(timeout)}s{detail}: touch {path}",
        flush=True,
    )
    while time.monotonic() < deadline:
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        await asyncio.sleep(1.0)
    return False
