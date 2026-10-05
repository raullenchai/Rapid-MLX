"""What the agent must not do on its own in a handed-over browser.

Pure classifiers over observed rows (role, subrole, label), so they can be
tested without a screen:

* secrets the user types themselves: passwords, one-time codes, card
  security codes (the agent hands these over instead of typing them);
* card numbers in anything the agent types (refused outright);
* buttons that commit money or an irreversible action (they need the
  user's approval of what is on screen at that moment).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
SECURE_LABEL = "[secure text redacted]"

_SECRET_LABEL = re.compile(
    r"password|passcode|\bpin\b|one[- ]time|verification code|security code|"
    r"\bcvv\b|\bcvc\b|\b2fa\b|two[- ]factor|authenticat(?:ion|or) code|"
    r"social security|\bssn\b",
    re.I,
)

_COMMIT_LABEL = re.compile(
    r"\b(?:place|submit|complete|confirm|finish)\s+(?:my\s+|your\s+)?"
    r"(?:order|payment|purchase|booking|transfer)\b"
    r"|\b(?:pay|buy|book|order)\s+now\b"
    r"|\bmake\s+(?:a\s+)?payment\b"
    r"|\bsend\s+(?:money|payment)\b"
    r"|\b(?:transfer|donate|subscribe|enroll)\b",
    re.I,
)
_AMOUNT = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d{2})?")
_PRICED_VERB = re.compile(
    r"\b(?:upgrade|join|pay|buy|purchase|subscribe|donate|renew|tip|add funds)\b", re.I
)
_PRESS_ROLES = {"AXButton", "AXLink", "AXMenuItem", "AXMenuButton", "AXPopUpButton"}
_TOTAL_WORDS = re.compile(
    r"\b(?:(?:sub)?totals?|amounts?|charged?|due|fees?|pay from|paid with|date)\b", re.I
)
_VALUE = re.compile(r"\d")
_METHOD = re.compile(r"ending (?:in )?\d{4}|••\s?\d{4}", re.I)
_CARD_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def needs_human_input(role: str, subrole: str, label: str) -> str | None:
    """Why only the user should type into this field, or None."""
    if (
        subrole == "AXSecureTextField"
        or role == "AXSecureTextField"
        or label == SECURE_LABEL
    ):
        return "a password field"
    if role in TEXT_ROLES and _SECRET_LABEL.search(label or ""):
        return f'a secret field ("{label[:60]}")'
    return None


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def contains_card_number(text: str) -> bool:
    for match in _CARD_RUN.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return True
    return False


def is_money_commit(role: str, label: str) -> bool:
    """Whether pressing this control commits money or an irreversible step."""
    if role not in _PRESS_ROLES or not label:
        return False
    if _COMMIT_LABEL.search(label):
        return True
    return bool(_AMOUNT.search(label) and _PRICED_VERB.search(label))


def amounts(texts: list[str]) -> tuple[str, ...]:
    """Every currency amount on screen, in order (the approval binds to it)."""
    return tuple(
        m.group().replace(" ", "") for t in texts for m in _AMOUNT.finditer(t or "")
    )


def money_context(
    texts: list[str], choices: Sequence[str] = (), limit: int = 10
) -> list[str]:
    """The lines a person needs to approve a charge: totals, fees, method,
    and the options chosen on the page (delivery window, tip, plan)."""
    # Tables put the name and the value in separate cells: join a label
    # with the value that follows it ("Total charged" + "$142.37").
    texts = [t for t in texts if t and t.strip()]  # cells and groups have no text
    joined: list[str] = []
    i = 0
    while i < len(texts):
        t = texts[i]
        nxt = texts[i + 1] if i + 1 < len(texts) else ""
        if (
            _TOTAL_WORDS.search(t)
            and not _AMOUNT.search(t)
            and not _METHOD.search(t)
            and len(t) < 40
            and _VALUE.search(nxt)
        ):
            joined.append(f"{t}: {nxt}")
            i += 2
            continue
        joined.append(t)
        i += 1
    picked = [
        t
        for t in joined
        if (": " in t and _TOTAL_WORDS.search(t.split(": ")[0]))
        or (_TOTAL_WORDS.search(t) and _AMOUNT.search(t))
        or _METHOD.search(t)
    ]
    if not picked:
        picked = [t for t in joined if _AMOUNT.search(t)]
    seen: list[str] = []
    for t in [*picked, *(f"Chosen: {c}" for c in choices)]:
        if t[:120] not in seen:
            seen.append(t[:120])
    return seen[:limit]
