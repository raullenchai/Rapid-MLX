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
