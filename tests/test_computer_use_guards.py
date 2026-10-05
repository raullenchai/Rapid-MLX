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
    assert not commit("Place order", role="AXStaticText")


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
    assert context == [
        "Visa ending 4242",
        "Order total $48.14",
        "Order total: $48.14",
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
