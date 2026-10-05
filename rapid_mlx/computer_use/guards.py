"""What the agent must not do on its own in a handed-over browser.

Pure classifiers over observed rows (role, subrole, label), so they can be
tested without a screen:

* secrets the user types themselves: passwords, one-time codes, card
  security codes (the agent hands these over instead of typing them);
* card numbers in anything the agent types (refused outright);
* controls that commit money (they need the user's approval of what is on
  screen at that moment). Deleting or cancelling is not gated here: whether
  the user asked for it is the planner's call, made from their request.

The vocabularies cover en, es, pt, de, fr, it, nl, zh-Hans/zh-Hant, ja and
ko. They gate the press that commits, not the way to it: "Checkout",
"Proceed to checkout" and "Continue to payment" open the page whose button
pays, and that button is gated. A verb that is also free ("Subscribe",
"Transfer", "Book") commits money only with a price on it or around it.
A label outside these languages is not recognised; an amount on a
"pay"-like control in one of them still is.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
SECURE_LABEL = "[secure text redacted]"
# Shown instead of what the user typed into a field only they may fill.
USER_VALUE = "[entered by the user]"

# Field names in the languages a shopper meets most (en, es, pt, de, fr, it,
# nl, zh-Hans/zh-Hant, ja, ko): passwords, one-time and verification codes,
# PINs, card numbers and security codes. CJK words have no \b around them.
_SECRET_LABEL = re.compile(
    r"password|passcode|\bpin\b|one[- ]time|verification code|security code|"
    r"\bcvv\b|\bcvc\b|\b2fa\b|two[- ]factor|authenticat(?:ion|or) code|"
    r"social security|\bssn\b|card number|\bcard no\b|credit card|debit card|"
    r"\botp\b|\bmfa\b|\bsms code\b|(?:sign[- ]in|login|access) code|two[- ]step|2[- ]step|"
    # "Enter the 6-digit code", "six digit PIN"
    r"\b(?:\d{1,2}|four|five|six|seven|eight)[- ]?digits?\s+(?:code|pin|passcode)\b|"
    # es / pt
    r"contraseña|\bsenha\b|"
    r"c[óo]digo\s+(?:de\s+)?(?:verificaci[óo]n|verifica[çc][ãa]o|seguran[çc]a|seguridad|"
    r"acceso|acesso|confirmaci[óo]n|confirma[çc][ãa]o|un solo uso|uso [úu]nico|sms)|"
    r"c[óo]digo de \d{1,2} d[íi]gitos|"
    r"n[úu]mero\s+(?:de\s+(?:la\s+)?|do\s+)?(?:tarjeta|cart[ãa]o)|"
    # de
    r"passwort|kennwort|bestätigungscode|verifizierungscode|sicherheitscode|"
    r"einmal(?:code|passwort|kennwort)|\btan\b|\d{1,2}-stellige[nr]?\s+(?:code|pin)|"
    r"karten(?:nummer|prüfnummer)|prüf(?:ziffer|nummer)|"
    # fr
    r"mot de passe|code\s+(?:de\s+)?(?:vérification|sécurité|confirmation|validation)|"
    r"code à usage unique|code à \d{1,2} chiffres|num[ée]ro\s+de\s+(?:la\s+)?carte|cryptogramme|"
    # it / nl
    r"codice\s+(?:di\s+)?(?:verifica|sicurezza|conferma)|codice monouso|"
    r"numero\s+(?:della\s+)?carta|wachtwoord|verificatiecode|beveiligingscode|kaartnummer|"
    # zh-Hans / zh-Hant
    r"密码|密碼|验证码|驗證碼|校验码|校驗碼|动态码|動態碼|安全码|安全碼|"
    r"卡号|卡號|pin(?:码|碼)|"
    # ja
    r"パスワード|暗証番号|認証コード|確認コード|ワンタイム|セキュリティ(?:ー)?コード|カード番号|"
    # ko
    r"비밀번호|인증\s?번호|인증\s?코드|보안\s?코드|카드\s?번호|"
    r"\d{1,2}\s?(?:位|桁の?|자리)\s?(?:数字|数字の)?(?:码|碼|コード|코드|번호)",
    re.I,
)

# Pressing these commits money by itself, whatever is on screen. Moving to a
# checkout or payment page ("Checkout", "Proceed to checkout", "Continue to
# payment", "Zur Kasse", "去结算") commits nothing and is not gated: the
# button that pays on that page is.
_COMMIT_LABEL = re.compile(
    r"\b(?:place|submit|complete|confirm|finish)\s+(?:my\s+|your\s+|the\s+)?"
    r"(?:order|payment|purchase|booking|transfer)\b"
    r"|\b(?:pay|buy|book|order|purchase)\s+now\b"
    r"|\bconfirm\s+(?:and|&)\s+(?:pay|buy|book|order|purchase|subscribe)\b"
    r"|\bmake\s+(?:a\s+)?payment\b"
    r"|\bsend\s+(?:money|payment)\b"
    r"|\b(?:donate|enroll)\b"
    r"|^\s*(?:pay|buy|book|purchase|reserve)\s*$"
    # es / pt
    r"|\b(?:comprar|pagar|reservar)\s+(?:ahora|ya|agora)\b"
    r"|\b(?:realizar|hacer|confirmar|finalizar|fazer|completar|enviar)\s+"
    r"(?:el\s+|la\s+|o\s+|a\s+|mi\s+|meu\s+|minha\s+)?(?:pedido|compra|pago|pagamento)\b"
    r"|\bconfirmar\s+[ye]\s+pagar\b|^\s*(?:comprar|pagar)\s*$"
    # de
    r"|\b(?:zahlungspflichtig|kostenpflichtig)\s+(?:bestellen|abschließen|buchen)\b"
    r"|\bjetzt\s+(?:kaufen|bezahlen|bestellen|buchen|zahlen)\b"
    r"|\bbestellung\s+(?:abschicken|absenden|aufgeben|abschließen)\b"
    r"|\bkauf\s+abschließen\b|\bbestellen\s+und\s+bezahlen\b"
    r"|^\s*(?:kaufen|bezahlen|bestellen)\s*$"
    # fr
    r"|\bacheter\s+maintenant\b|\bpayer\s+maintenant\b"
    r"|\b(?:passer|confirmer|valider)\s+(?:la\s+|ma\s+|votre\s+)?commande\b"
    r"|\b(?:confirmer|valider)\s+(?:l['’]\s?achat|le\s+paiement|et\s+payer)\b"
    r"|^\s*(?:acheter|payer|commander)\s*$"
    # it / nl
    r"|\b(?:acquista|compra|paga)\s+ora\b"
    r"|\b(?:effettua|conferma|invia|completa)\s+(?:l['’]\s?)?(?:ordine|acquisto|pagamento)\b"
    r"|\bconferma\s+e\s+paga\b|^\s*(?:acquista|paga)\s*$"
    r"|\bnu\s+(?:kopen|betalen)\b|\bbestelling\s+plaatsen\b|^\s*(?:betalen|kopen)\s*$"
    # zh-Hans / zh-Hant
    r"|立即(?:购买|購買|支付|付款|下单|下單)|马上(?:购买|支付)|馬上(?:購買|支付)"
    r"|(?:提交|确认|確認)(?:订单|訂單)|(?:确认|確認)(?:支付|付款|购买|購買)"
    r"|去(?:支付|付款)|^\s*(?:支付|付款|购买|購買|下单|下單)\s*$"
    # ja
    r"|注文を確定|注文確定|購入を確定|購入確定|支払いを確定|今すぐ(?:購入|買う)"
    r"|^\s*(?:注文する|購入する|支払う|購入)\s*$"
    # ko
    r"|구매\s?하기|바로\s?구매|결제\s?하기|주문\s?하기|지금\s?구매|(?:주문|구매|결제)\s?확정"
    r"|^\s*(?:결제|구매)\s*$",
    re.I,
)
# Verbs that commit money only when a price is on the button or around it
# ("Subscribe $9.99/mo", a "Book" next to the fare): bare, they also name a
# newsletter, a channel, a settings pane ("Transfer or Reset") or a demo.
_PRICED_VERB = re.compile(
    r"\b(?:upgrade|join|pay|buy|purchase|subscribe|donate|renew|tip|add funds|"
    r"transfer|book|reserve|rent|"
    r"pagar|comprar|suscribirse|suscribir|assinar|doar|donar|renovar|reservar|"
    r"zahlen|bezahlen|kaufen|abonnieren|spenden|buchen|"
    r"payer|acheter|s['’]abonner|réserver|"
    r"paga|pagare|acquista|abbonati|dona|prenota)\b"
    r"|支付|付款|购买|購買|订阅|訂閱|充值|預訂|预订|購入|支払|申し込|결제|구매|구독|후원|예약",
    re.I,
)
# A priced verb with these is free (a newsletter, a demo call), unless the
# button itself carries a price.
_FREE_TARGET = re.compile(
    r"\b(?:newsletters?|e-?mails?|updates|notifications?|alerts?|channels?|podcasts?|"
    r"feeds?|rss|calendars?|demo|call|consultation|free|for free|gratis|gratuit)\b"
    r"|免费|免費|無料|무료",
    re.I,
)
_CHOICE = re.compile(r"\b(?:or|oder|ou)\b|或|または|또는|/", re.I)
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
# Not inside a word: "kr 99" is an amount, "Rskr 99" is not.
_NOT_LETTER_BEFORE = r"(?<![^\W\d_])"
_NOT_LETTER_AFTER = r"(?![^\W\d_])"
# Local abbreviations written before or after the number ("Rs. 500", "kr 99",
# "99 kr", "zł 10", "Rp 15.000") and CJK unit characters ("199元", "500円").
_LOCAL_CURRENCY = (
    rf"{_NOT_LETTER_BEFORE}(?:Rs\.?|kr\.?|zł|Kč|RM|Rp|lei){_NOT_LETTER_AFTER}|[元円원]"
)
_CURRENCY = (
    rf"(?:(?<![A-Za-z])(?:{_DOLLAR_PREFIXES})\$"
    rf"|[{_CURRENCY_SYMBOLS}]|\b(?:{_CURRENCY_CODES})\b|{_LOCAL_CURRENCY})"
)
# Currencies written out, after the number only ("209 US dollars",
# "20 euros"). Pounds are left out: "5 pounds" is as often a weight.
_CURRENCY_WORDS = (
    r"(?i:(?:US|U\.S\.|Canadian|Australian|New Zealand|Hong Kong|Singapore|Mexican)"
    r"\s+)?(?i:dollars?|euros?|yen|yuan|rupees?|pesos?|reais|francs?|kronor|kroner|"
    rf"zlotys?|złotych|rubles?|roubles?|rand){_NOT_LETTER_AFTER}"
)
_SUFFIX = rf"(?:{_CURRENCY}|{_CURRENCY_WORDS})"
# A suffix currency followed by a number is that number's prefix
# ("Qty 2 €5" is €5, not "2 €").
_AMOUNT = re.compile(rf"{_CURRENCY}\s?{_NUMBER}|{_NUMBER}\s?{_SUFFIX}(?!\s?\d)")
# A cell holding only an amount ("$7.49", "12,50 €", "$65.00/yr").
_BARE_AMOUNT = re.compile(
    rf"^\s*[-−+]?(?:{_CURRENCY}\s?{_NUMBER}|{_NUMBER}\s?{_SUFFIX})"
    r"(?:\s*/\s*[A-Za-z]{1,5})?\s*$"
)
# Verbs that commit money with a price around the button, not only on it:
# a "Book" next to the fare, a "Subscribe" under "$9.99/mo".
_NEARBY_PRICED_VERB = re.compile(
    r"\b(?:subscribe|transfer|book|reserve|renew|suscribirse|suscribir|assinar|"
    r"reservar|abonnieren|buchen|s['’]abonner|réserver|abbonati|prenota)\b"
    r"|订阅|訂閱|预订|預訂|申し込|구독|예약",
    re.I,
)
# Roles the session presses that carry a control's name. A clickable text,
# group or cell ("Place order" as a styled div) is pressed too, but only a
# short name reads as a control rather than as content about one.
_PRESS_ROLES = {"AXButton", "AXLink", "AXMenuItem", "AXMenuButton", "AXPopUpButton"}
MAX_CONTROL_WORDS = 6
MAX_CONTROL_CHARS = 48
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


def _names_a_control(role: str, label: str) -> bool:
    """Whether pressing ``role`` labelled ``label`` presses that label.

    A text field is clicked to type into it, never to commit. Outside the
    button roles, a long label is content (a paragraph that mentions the
    button), not the name of what is pressed.
    """
    if not label or role in TEXT_ROLES:
        return False
    return role in _PRESS_ROLES or (
        len(label) <= MAX_CONTROL_CHARS and len(label.split()) <= MAX_CONTROL_WORDS
    )


def is_money_commit(role: str, label: str, nearby: Sequence[str] = ()) -> bool:
    """Whether pressing this control commits money.

    Called only for the control being pressed, whatever its role.
    ``nearby``: the priced lines around it (its order summary, the fare next
    to it); "Book" or "Subscribe" spends money only with a price on it or
    there.
    """
    if not _names_a_control(role, label):
        return False
    if _COMMIT_LABEL.search(label):
        return True
    if _AMOUNT.search(label):
        return bool(_PRICED_VERB.search(label))
    # A short call to act ("Book", "Subscribe to Premium"), not a menu of
    # choices ("Transfer or Reset") or a sentence about something else.
    return bool(
        _NEARBY_PRICED_VERB.search(label)
        and len(label.split()) <= 3
        and not _CHOICE.search(label)
        and not _FREE_TARGET.search(label)
        and amounts(list(nearby))
    )


def amounts(texts: list[str]) -> tuple[str, ...]:
    """Every currency amount on screen, in order (the approval binds to it)."""
    return tuple(
        m.group().replace(" ", "") for t in texts for m in _AMOUNT.finditer(t or "")
    )


def _join_cells(texts: Sequence[str]) -> list[str]:
    """Texts with each name joined to the value in the cell after it.

    Tables put the name and the value in separate cells ("Total charged" +
    "$142.37", "Bananas" + "$7.49"); an amount alone means nothing.
    """
    texts = [t for t in texts if t and t.strip()]  # cells and groups have no text
    joined: list[str] = []
    i = 0
    while i < len(texts):
        t = texts[i]
        nxt = texts[i + 1] if i + 1 < len(texts) else ""
        if (
            not _AMOUNT.search(t)
            and not _METHOD.search(t)
            and len(t) < 40
            and (
                (_TOTAL_WORDS.search(t) and _VALUE.search(nxt))
                or _BARE_AMOUNT.match(nxt)
            )
        ):
            joined.append(f"{t}: {nxt}")
            i += 2
            continue
        joined.append(t)
        i += 1
    return joined


def priced_lines(texts: Sequence[str], limit: int = 8) -> list[str]:
    """The lines of ``texts`` that carry an amount, each with its name."""
    return [t for t in _join_cells(texts) if _AMOUNT.search(t)][:limit]


def _same_line(text: str) -> str:
    # "Order total $48.14" (a heading) and "Order total: $48.14" (two cells)
    # say the same thing.
    return re.sub(r"[\s:]+", " ", text).strip().casefold()


def money_context(
    texts: list[str],
    choices: Sequence[str] = (),
    limit: int = 12,
    *,
    items: Sequence[str] = (),
    heading: str = "",
) -> list[str]:
    """The lines a person needs to approve a charge: what it is for (the
    button's section), totals, fees, method, the options chosen on the page
    (delivery window, tip, plan), and the priced lines next to the button."""
    joined = _join_cells(texts)
    picked = [
        t
        for t in joined
        if (": " in t and _TOTAL_WORDS.search(t.split(": ")[0]))
        or (_TOTAL_WORDS.search(t) and _AMOUNT.search(t))
        or _METHOD.search(t)
    ]
    if not picked:
        picked = [t for t in joined if _AMOUNT.search(t)]
    lines = [f"For: {heading}"] if heading else []
    lines += [*picked, *(f"Chosen: {c}" for c in choices)]
    lines += [f"Item: {t}" for t in items]
    seen: list[str] = []
    keys: set[str] = set()
    for t in lines:
        key = _same_line(t.removeprefix("Item: "))
        if key not in keys:
            keys.add(key)
            seen.append(t[:120])
    return seen[:limit]
