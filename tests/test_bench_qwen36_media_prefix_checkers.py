"""Contracts for the media-prefix harness's open-ended ``any`` checker.

The ``any`` checker gates turns where no fixed answer exists, but it is
still grounded two ways: ``min_words`` plus a distinct-vocabulary floor
rejects degenerate loops (``foo foo foo foo foo``), and ``required_any``
anchors the semantics — at least one alternative term list must be fully
present, so fluent-but-unrelated output fails.
"""

from __future__ import annotations

from scripts.bench_qwen36_media_prefix import _checker_pass


def test_any_requires_min_words():
    assert not _checker_pass({"type": "any", "min_words": 5}, "one two three")
    assert _checker_pass({"type": "any", "min_words": 3}, "one two three four")


def test_any_rejects_repeated_single_token():
    # Word count alone accepts this; the distinct-vocabulary bar must not.
    assert not _checker_pass({"type": "any", "min_words": 5}, "foo foo foo foo foo")


def test_any_accepts_normal_prose():
    text = (
        "The desktop shows the model picker with qwen3.6 selected and the "
        "status bar reports ready, so pressing next moves to the runtime step."
    )
    assert _checker_pass({"type": "any", "min_words": 5}, text)


def test_any_default_has_no_floor():
    assert _checker_pass({"type": "any"}, "ok")
    assert _checker_pass({}, "ok")


def test_any_distinct_floor_scales_with_min_words():
    # Floor is max(2, min_words // 2) distinct words: 4 distinct words pass
    # a min_words=8 checker even with repeats, 3 do not.
    four = "alpha beta gamma delta alpha beta gamma delta"
    three = "alpha beta gamma alpha beta gamma alpha beta gamma"
    assert _checker_pass({"type": "any", "min_words": 8}, four)
    assert not _checker_pass({"type": "any", "min_words": 8}, three)


def test_any_required_any_requires_an_alternative():
    checker = {"type": "any", "min_words": 3, "required_any": [["ready"], ["idle"]]}
    assert _checker_pass(checker, "The status chip reads Ready and green.")
    assert _checker_pass(checker, "The chip says IDLE right now.")
    # Fluent but unrelated: correct length, none of the anchors.
    assert not _checker_pass(checker, "The weather is nice today, honestly.")


def test_any_required_any_alternative_must_match_completely():
    # An alternative is satisfied only when every term in it is present.
    checker = {"type": "any", "required_any": [["new", "chat"]]}
    assert _checker_pass(checker, "press new chat to begin")
    assert not _checker_pass(checker, "press the new button to begin the setup")


def test_any_without_required_any_stays_word_floor_only():
    assert _checker_pass({"type": "any", "min_words": 2}, "anything at all")
    assert not _checker_pass({"type": "any", "min_words": 3}, "too short")


def test_required_terms_match_on_token_boundaries():
    # Substring matching admits wrong answers: "bright side" contains
    # "right" but is not the right side.
    checker = {"type": "terms", "required": ["right"]}
    assert _checker_pass(checker, "The Stop button is on the right side.")
    assert not _checker_pass(checker, "The Stop button is on the bright side.")
    # Punctuation at token edges does not block the match.
    assert _checker_pass(checker, "The Stop button sits on the right.")


def test_required_terms_reject_negated_mentions():
    # "not ready" must not satisfy a "ready" requirement.
    checker = {"type": "terms", "required": ["ready"]}
    assert _checker_pass(checker, "The status chip reads Ready.")
    assert not _checker_pass(checker, "The chip is not ready at all.")
    assert not _checker_pass(checker, "There is no ready indicator here.")
    # A negated compound does not satisfy the positive term either.
    assert not _checker_pass(checker, "The dark-themed chip is not ready.")
    # ``forbidden`` stays negation-blind: claiming the chip is "not ready"
    # on a screen whose chip says otherwise is still wrong content.
    assert not _checker_pass(
        {"type": "terms", "required": ["idle"], "forbidden": ["ready"]},
        "The chip is not ready.",
    )


def test_compound_forms_tokenize_consistently():
    # Slash/hyphen compounds split the same on both sides: "dark-themed"
    # satisfies "dark", and "skip"/"next" satisfies both terms.
    assert _checker_pass(
        {"type": "terms", "required": ["dark"]},
        "The dark-themed interface is high contrast.",
    )
    assert _checker_pass(
        {"type": "terms", "required": ["skip", "next"]},
        'The action bar reads "skip"/"next".',
    )
    # A compound hyphenated with the term is a match; an unrelated token
    # sharing a substring is not ("bright" is not "right").
    assert not _checker_pass(
        {"type": "terms", "required": ["right"]}, "bright-themed buttons everywhere"
    )


def test_structural_constraints_enforce_prompt_wording():
    # "Two short lines, nothing else": a one-line keyword blob fails even
    # with the right terms.
    two_lines = {
        "type": "terms",
        "required": ["qwen3.6-27b", "ready"],
        "min_lines": 2,
        "max_lines": 2,
    }
    assert _checker_pass(two_lines, "qwen3.6-27b\nReady")
    assert not _checker_pass(two_lines, "qwen3.6-27b Ready and stop too")
    assert not _checker_pass(two_lines, "qwen3.6-27b\nReady\nStop")
    # "Reply with only its exact label": a verbose answer fails.
    label_only = {"type": "terms", "required": ["skip"], "max_words": 3}
    assert _checker_pass(label_only, "Skip")
    assert not _checker_pass(label_only, "The Skip button is in the corner")
    # "at least 200 words": keyword-laden filler without the terms fails
    # on terms; fluent filler without structure passes only the floor.
    long_ask = {"type": "terms", "required": ["pick a model"], "min_words": 200}
    assert not _checker_pass(long_ask, "Pick a model, press Next or Skip.")
    filler = "Model selection interfaces matter in daily workflows. " * 40
    assert not _checker_pass(long_ask, filler)
    grounded = filler + "The primary heading asks the user to pick a model."
    assert _checker_pass(long_ask, grounded)
    # Line/word bounds apply to json_shape and any checkers too.
    assert not _checker_pass(
        {"type": "json_shape", "keys": ["a"], "max_lines": 1}, '{"a": 1}\nextra'
    )
    assert not _checker_pass({"type": "any", "min_words": 3, "max_lines": 1}, "a\nb\nc")
