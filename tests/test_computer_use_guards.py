from rapid_mlx.computer_use import guards


def test_secret_fields_are_for_the_user():
    assert guards.needs_human_input("AXTextField", "AXSecureTextField", "Password")
    assert guards.needs_human_input("AXTextField", "", "[secure text redacted]")
    assert guards.needs_human_input("AXTextField", "", "Verification code")
    assert guards.needs_human_input("AXTextField", "", "Security code (CVV)")
    assert guards.needs_human_input("AXTextField", "", "Username or email") is None
    assert guards.needs_human_input("AXTextArea", "", "Type a message") is None
    # A heading that mentions a code is not an input.
    assert (
        guards.needs_human_input("AXStaticText", "", "Enter your verification code")
        is None
    )


def test_card_numbers_need_a_valid_checksum():
    assert guards.contains_card_number("my card is 4242 4242 4242 4242")
    assert guards.contains_card_number("4111-1111-1111-1111")
    assert not guards.contains_card_number("order VM-20419, phone 650 555 0142")
    assert not guards.contains_card_number("1234 5678 9012 3456")  # fails Luhn


def test_money_commits():
    commit = lambda label, role="AXButton": guards.is_money_commit(role, label)  # noqa: E731
    for label in (
        "Place order",
        "Submit payment",
        "Pay now",
        "Buy now",
        "Confirm purchase",
        "Upgrade for $65",
        "Enroll now",
        "Make a payment",
        "Donate",
    ):
        assert commit(label), label
    for label in (
        "Add to cart",
        "Pay bill",
        "Continue to checkout",
        "Continue",
        "Replace with 2% Reduced Fat Milk, 1 gal × 2 – $6.19",
        "Yes, refund to my card",
        "Accept store credit",
    ):
        assert not commit(label), label
    # Whatever is pressed is checked: a "Place order" div commits as surely
    # as a button. A paragraph that mentions the button, or a field, does not.
    for role in ("AXStaticText", "AXGroup", "AXCheckBox", "AXCell"):
        assert commit("Place order", role=role), role
    assert not commit(
        "By clicking Place order you agree to the conditions of use and sale.",
        role="AXStaticText",
    )
    assert commit(
        "By clicking Place order you agree to the conditions of use and sale."
    )  # a button's whole name is still its name
    assert not commit("Place order", role="AXTextField")


def test_money_context_joins_table_cells():
    texts = [
        "Account",
        "",
        "4418-2207-91",
        "Amount",
        "",
        "$142.37",
        "Convenience fee",
        "$0.00",
        "Total charged",
        "",
        "$142.37",
        "Pay from",
        "",
        "Checking ending 6789",
        "Payment date",
        "",
        "2026-10-03",
    ]
    context = guards.money_context(texts)
    assert "Total charged: $142.37" in context
    assert "Amount: $142.37" in context
    assert "Pay from: Checking ending 6789" in context
    assert "Payment date: 2026-10-03" in context
    assert not any(line.startswith("Checking ending 6789:") for line in context)
    assert guards.amounts(texts) == ("$142.37", "$0.00", "$142.37")


def test_money_context_skips_headings_and_lists_choices():
    texts = [
        "Delivering to 94301",
        "Checkout",
        "Checkout",
        "Choose a delivery window",
        "Choose a delivery window",
        "Sat, Oct 10",
        "12–2 PM",
        "Payment",
        "Visa ending 4242",
        "Driver tip $5",
        "Order total $48.14",
        "Order total",
        "$48.14",
        "Place order",
    ]
    context = guards.money_context(texts, ["Sat, Oct 10 12–2 PM", "Driver tip: $5"])
    # The heading "Order total $48.14" and the cells "Order total" + "$48.14"
    # say the same thing once.
    assert context == [
        "Visa ending 4242",
        "Order total $48.14",
        "Chosen: Sat, Oct 10 12–2 PM",
        "Chosen: Driver tip: $5",
    ]


def test_secure_roles_and_labels_on_non_text_controls():
    assert guards.needs_human_input("AXSecureTextField", "", "") == "a password field"
    assert guards.needs_human_input("AXComboBox", "", "Enter PIN").startswith(
        "a secret field"
    )
    assert guards.needs_human_input("AXSearchField", "", "Search orders") is None
    assert guards.needs_human_input("AXButton", "", "Forgot password?") is None
    for label in ("OTP", "MFA code", "Sign-in code", "2-step code", "SMS code"):
        assert guards.needs_human_input("AXTextField", "", label), label


def test_card_number_bounds():
    assert guards.contains_card_number("378282246310005")  # 15-digit Amex
    assert not guards.contains_card_number("")
    assert not guards.contains_card_number(None)
    assert not guards.contains_card_number("tracking 1Z 4242 4242 42")  # too short
    # A card inside a longer digit run is still refused.
    assert guards.contains_card_number("4242 4242 4242 4242 4242 4242")


def test_priced_actions_and_amounts():
    assert guards.is_money_commit("AXLink", "Renew for £9.99")
    assert guards.is_money_commit("AXMenuItem", "Tip $3")
    assert not guards.is_money_commit("AXButton", "View plan $9.99")
    assert not guards.is_money_commit("AXButton", "")
    assert guards.amounts(["€12,50", "12,99 €", "1.234,56 EUR", "USD 9"]) == (
        "€12,50",
        "12,99€",
        "1.234,56EUR",
        "USD9",
    )
    assert guards.amounts(
        ["R$ 30,00", "CHF 9.50", "₹1,299", "Qty 2 €5", "3 items"]
    ) == (
        "R$30,00",
        "CHF9.50",
        "₹1,299",
        "€5",
    )
    # Other countries' dollars keep their letters, so they never bind as $.
    assert guards.amounts(["US$ 5", "C$5", "30 R$", "BAR$5"]) == (
        "US$5",
        "C$5",
        "30R$",
        "$5",
    )
    assert guards.amounts(["Total $1,204.50 and €3", "", None, "£ 7"]) == (
        "$1,204.50",
        "€3",
        "£7",
    )


def test_money_context_falls_back_to_amounts_dedupes_and_caps():
    # "Coffee" is not a fee.
    texts = ["Coffee beans $18.00", "Coffee beans $18.00", "Mug $9.00", "Ships Friday"]
    assert guards.money_context(texts) == ["Coffee beans $18.00", "Mug $9.00"]
    many = [f"Fee {i}: $1.0{i}" for i in range(9)]
    assert len(guards.money_context(many, ["Plan: Pro"] * 3, limit=4)) == 4
    assert guards.money_context(["Total", "$5.00"], ["Plan: Pro", "Plan: Pro"]) == [
        "Total: $5.00",
        "Chosen: Plan: Pro",
    ]


def test_priced_lines_name_each_amount_and_context_says_what_it_is_for():
    # A product name and its price in separate cells read as one line; an
    # amount with nothing before it stays as it is.
    texts = ["Bananas", "$0.29", "$65.00/yr", "Ships Friday", "Eggs", "12,50 €"]
    assert guards.priced_lines(texts) == [
        "Bananas: $0.29",
        "$65.00/yr",
        "Eggs: 12,50 €",
    ]
    assert guards.priced_lines(texts, limit=1) == ["Bananas: $0.29"]
    context = guards.money_context(
        ["Order total", "$48.14", "Order total $48.14"],
        items=["Bananas: $0.29", "Order total $48.14"],
        heading="Your order",
    )
    # The same line said twice (two cells, or a heading) is listed once.
    assert context == ["For: Your order", "Order total: $48.14", "Item: Bananas: $0.29"]


# -- held-out generality: other languages, other phrasings, other roles ------


def test_secret_fields_in_other_languages_and_phrasings():
    for label in (
        "Contraseña",
        "Senha",
        "Passwort",
        "Kennwort",
        "Mot de passe",
        "Wachtwoord",
        "密码",
        "密碼",
        "パスワード",
        "비밀번호",
        "Código de verificación",
        "Código de segurança",
        "Bestätigungscode",
        "Sicherheitscode",
        "Code de vérification",
        "Codice di verifica",
        "验证码",
        "驗證碼",
        "認証コード",
        "인증번호",
        "Kartennummer",
        "Numéro de carte",
        "Número de tarjeta",
        "Número do cartão",
        "Numero della carta",
        "卡号",
        "カード番号",
        "카드 번호",
        "暗証番号",
        "PIN码",
        "安全码",
        "Enter the 6-digit code we sent",
        "six digit code",
        "6-stelliger Code",
        "Saisissez le code à 6 chiffres",
        "Código de 6 dígitos",
        "6桁のコード",
        "6자리 코드",
    ):
        assert guards.needs_human_input("AXTextField", "", label), label
    for label in (
        "Palabra clave",
        "Promo code",
        "Gift card code",
        "Zip code",
        "Enter your 10-digit phone number",
        "Nachricht",
        "地址",
        "メッセージ",
        "검색",
        "Nombre",
    ):
        assert guards.needs_human_input("AXTextField", "", label) is None, label


def test_commit_buttons_in_other_languages():
    for label in (
        "Comprar ahora",
        "Realizar pedido",
        "Confirmar compra",
        "Finalizar compra",
        "Comprar agora",
        "Pagar R$ 30",
        "Jetzt kaufen",
        "Zahlungspflichtig bestellen",
        "Bestellung abschicken",
        "Kaufen",
        "Payer 25,00 €",
        "Passer la commande",
        "Confirmer la commande",
        "Acheter maintenant",
        "Acquista ora",
        "Conferma ordine",
        "Bestelling plaatsen",
        "立即购买",
        "立即購買",
        "提交订单",
        "确认付款",
        "支付 ¥199",
        "注文を確定する",
        "今すぐ購入",
        "구매하기",
        "결제하기",
        "Pay 20 dollars",
    ):
        assert guards.is_money_commit("AXButton", label), label
    for label in (
        "In den Warenkorb",
        "Zur Kasse",
        "Añadir al carrito",
        "Weiter einkaufen",
        "Bestellung überprüfen",
        "Ver pedido",
        "加入购物车",
        "去结算",
        "カートに入れる",
        "注文履歴",
        "장바구니 담기",
        "Abonnieren",
        "订阅",
    ):
        assert not guards.is_money_commit("AXButton", label), label


def test_commits_need_money_not_just_a_verb():
    commit = guards.is_money_commit
    # Bare, these verbs are a settings pane, a newsletter, a channel.
    for label in ("Transfer or Reset", "Subscribe to our newsletter", "Subscribe"):
        assert not commit("AXButton", label), label
    for label in ("Confirm and pay", "Complete purchase", "Book", "Purchase", "Pay"):
        assert commit("AXButton", label), label
    # Moving to the page that pays commits nothing; its pay button is gated.
    for label in (
        "Checkout",
        "Check out",
        "Proceed to checkout",
        "Continue to payment",
        "Book a demo",
    ):
        assert not commit("AXButton", label), label
    # A price on the button, or around it, makes these verbs spend.
    assert commit("AXButton", "Subscribe for $9.99/mo")
    assert commit("AXButton", "Subscribe", ["Premium: $9.99/mo"])
    assert commit("AXButton", "Transfer", ["Amount: 500 dollars"])
    assert commit("AXButton", "Book", ["From 209 US dollars"])
    assert commit("AXButton", "Subscribe to Premium", ["$9.99/mo"])
    assert not commit("AXButton", "Subscribe to our newsletter", ["$9.99"])
    # A choice of panes, or a long name, is not a call to pay.
    assert not commit("AXButton", "Transfer or Reset", ["$9.99"])
    assert not commit("AXButton", "Book your next stay with us", ["$9.99"])
    assert not commit("AXButton", "Transfer", ["Ships Friday"])
    assert not commit("AXButton", "Monday, October 26, 2026 , 209 US dollars")


def test_written_out_and_local_amounts():
    assert guards.amounts(
        [
            "From 209 US dollars.",
            "20 dollars",
            "10 Euros",
            "Rs. 500",
            "kr 99",
            "99 kr",
            "zł 10",
            "10 zł",
            "199元",
            "500円",
            "10,000원",
        ]
    ) == (
        "209USdollars",
        "20dollars",
        "10Euros",
        "Rs.500",
        "kr99",
        "99kr",
        "zł10",
        "10zł",
        "199元",
        "500円",
        "10,000원",
    )
    # Inside a word, as a weight, or as a verb, it is not money.
    assert guards.amounts(["Rskr 9", "5 pounds", "Try 3 times", "10 dollarstore"]) == ()
    assert guards.priced_lines(["Fare", "209 US dollars"]) == ["Fare: 209 US dollars"]
    assert guards.money_context(["Total", "Rs. 500"]) == ["Total: Rs. 500"]
