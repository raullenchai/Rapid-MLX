from rapid_mlx.computer_use import privacy


def test_secret_fields_are_the_users():
    only = privacy.user_only_field
    assert only("AXTextField", "AXSecureTextField", "Password")
    assert only("AXTextField", "", "[secure text redacted]")
    assert only("AXTextField", "", "Verification code")
    assert only("AXTextField", "", "Security code (CVV)")
    assert not only("AXTextField", "", "Username or email")
    assert not only("AXTextArea", "", "Type a message")
    # A heading that mentions a code is not an input.
    assert not only("AXStaticText", "", "Enter your verification code")


def test_secure_roles_and_labels_on_non_text_controls():
    only = privacy.user_only_field
    assert only("AXSecureTextField", "", "")
    assert only("AXComboBox", "", "Enter PIN")
    assert not only("AXSearchField", "", "Search orders")
    assert not only("AXButton", "", "Forgot password?")
    for label in ("OTP", "MFA code", "Sign-in code", "2-step code", "SMS code"):
        assert only("AXTextField", "", label), label


def test_prices_in_any_locale():
    for text in (
        "$7.49",
        "€12,50",
        "12,99 €",
        "1.234,56 EUR",
        "USD 9",
        "R$ 30,00",
        "CHF 9.50",
        "₹1,299",
        "US$ 5",
        "£ 7",
        "Qty 2 €5",
    ):
        assert privacy.is_price(text), text
    for text in ("3 items", "Bananas", "", None, "Order VM-20419"):
        assert not privacy.is_price(text), text
