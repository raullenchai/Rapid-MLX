"""What perception keeps out of the model's context.

Pure classifiers over observed rows (role, subrole, label), so they can be
tested without a screen. They decide what an observation shows, never what
the agent may do: whether to confirm with the user before an action (a
payment, a deletion) is the brain's decision, made from the user's
instructions, and it asks in its reply. Secrets the user types themselves
(passwords, one-time codes, card security codes) are named but their values
are never read back into an observation; ``handoff`` gives the user the
window to type them.
"""

from __future__ import annotations

import re

TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
SECURE_LABEL = "[secure text redacted]"
# Shown instead of what the user typed into a field only they may fill.
USER_VALUE = "[entered by the user]"

_SECRET_LABEL = re.compile(
    r"password|passcode|\bpin\b|one[- ]time|verification code|security code|"
    r"\bcvv\b|\bcvc\b|\b2fa\b|two[- ]factor|authenticat(?:ion|or) code|"
    r"social security|\bssn\b|card number|\bcard no\b|credit card|debit card|"
    r"\botp\b|\bmfa\b|\bsms code\b|(?:sign[- ]in|login|access) code|two[- ]step|2[- ]step",
    re.I,
)

# Prefix or suffix symbol or code, any grouping and decimal separator
# ("$1,204.50", "€12,50", "12,50 €", "1.234,56 EUR", "R$ 30", "CHF 9.50").
_CURRENCY_SYMBOLS = "$€£¥₹₩₽₺₪₫฿₱₦₴₡"
_CURRENCY_CODES = (
    "USD|EUR|GBP|CAD|AUD|NZD|JPY|CNY|HKD|SGD|CHF|INR|KRW|BRL|MXN|SEK|NOK|DKK|"
    "PLN|CZK|HUF|ZAR|TRY|ILS|AED|SAR|THB|PHP|IDR|MYR|TWD|RUB|UAH|NGN"
)
# Dollars of other countries keep their letters: "R$ 30" and "$30" differ.
_DOLLAR_PREFIXES = "R|US|C|CA|A|AU|NZ|HK|S|SG|MX|NT"
_NUMBER = r"\d(?:[\d.,]*\d)?"
_CURRENCY = (
    rf"(?:(?<![A-Za-z])(?:{_DOLLAR_PREFIXES})\$"
    rf"|[{_CURRENCY_SYMBOLS}]|\b(?:{_CURRENCY_CODES})\b)"
)
# A suffix currency followed by a number is that number's prefix
# ("Qty 2 €5" is €5, not "2 €").
_AMOUNT = re.compile(rf"{_CURRENCY}\s?{_NUMBER}|{_NUMBER}\s?{_CURRENCY}(?!\s?\d)")


def user_only_field(role: str, subrole: str, label: str) -> bool:
    """Whether this field holds a secret only the user types.

    Its value is never shown in an observation: a password field's contents
    are never read, and what was typed into a code or card field reads as
    ``USER_VALUE``.
    """
    if (
        subrole == "AXSecureTextField"
        or role == "AXSecureTextField"
        or label == SECURE_LABEL
    ):
        return True
    return role in TEXT_ROLES and bool(_SECRET_LABEL.search(label or ""))


def is_price(text: str) -> bool:
    """Whether ``text`` shows a currency amount (a tile's price, not its name)."""
    return bool(_AMOUNT.search(text or ""))
