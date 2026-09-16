"""Contracts for the media-prefix harness's open-ended ``any`` checker.

The ``any`` checker gates turns where no term checker can be semantic. A
bare word count is gameable — a degenerate loop like ``foo foo foo foo
foo`` passes — so the checker also requires the vocabulary to spread.
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
