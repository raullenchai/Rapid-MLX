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


def test_json_shape_requires_whole_response_to_parse():
    # Prose wrapped around an all-null payload must not satisfy a
    # "JSON only" ask: the stripped response itself must parse.
    checker = {
        "type": "json_shape",
        "keys": ["model", "status", "action"],
        "required": ["qwen3.6-27b", "ready", "stop"],
    }
    assert _checker_pass(
        checker, '{"model": "qwen3.6-27b", "status": "Ready", "action": "Stop"}'
    )
    assert not _checker_pass(
        checker,
        'qwen3.6 ready stop\n{"model": null, "status": null, "action": null}',
    )
    assert not _checker_pass(checker, 'Sure! {"model": "qwen3.6-27b"}')
    # Fenced JSON still parses.
    assert _checker_pass(
        checker,
        '```json\n{"model": "qwen3.6-27b", "status": "Ready", "action": "Stop"}\n```',
    )


def test_json_shape_field_terms_pin_designated_values():
    # ``required`` alone accepts wrong values scattered in prose; the
    # designated fields must carry the values.
    checker = {
        "type": "json_shape",
        "keys": ["model", "status"],
        "field_terms": {"model": ["qwen3.6-27b"], "status": ["ready"]},
    }
    assert _checker_pass(checker, '{"model": "qwen3.6-27b", "status": "Ready"}')
    assert not _checker_pass(checker, '{"model": null, "status": "Ready"}')
    assert not _checker_pass(
        checker, '{"model": "some-other-model", "status": "Ready"}'
    )
    assert not _checker_pass(checker, '{"status": "Ready"}')


def test_json_shape_list_expect_validates_items_in_order():
    # Buttons: exactly the three visible labels, in order.
    buttons = {
        "type": "json_shape",
        "keys": ["title", "buttons"],
        "list_keys": ["buttons"],
        "list_expect": {
            "buttons": [["new chat"], ["search chats"], ["start chatting"]]
        },
    }
    assert _checker_pass(
        buttons,
        '{"title": "T", "buttons": ["New chat", "Search chats", "Start chatting"]}',
    )
    assert not _checker_pass(buttons, '{"title": "T", "buttons": ["garbage"]}')
    assert not _checker_pass(
        buttons, '{"title": "T", "buttons": ["New chat", "Search chats"]}'
    )
    assert not _checker_pass(
        buttons,
        '{"title": "T", "buttons": ["Search chats", "New chat", "Start chatting"]}',
    )


def test_json_shape_list_len_and_bars_in_order():
    # Two model bars, each with its expected model and status in order.
    bars = {
        "type": "json_shape",
        "keys": ["bars"],
        "list_keys": ["bars"],
        "item_keys": {"bars": ["model", "status"]},
        "list_len": {"bars": 2},
        "list_expect": {
            "bars": [
                {"model": ["qwen3.6-27b"], "status": ["ready"]},
                {"model": ["fake-alias"], "status": ["idle"]},
            ]
        },
    }
    good = '{"bars": [{"model": "qwen3.6-27b", "status": "Ready"}, {"model": "fake-alias", "status": "Idle"}]}'
    assert _checker_pass(bars, good)
    # Missing the second bar.
    assert not _checker_pass(
        bars, '{"bars": [{"model": "qwen3.6-27b", "status": "Ready"}]}'
    )
    # Wrong status on the second bar.
    assert not _checker_pass(
        bars,
        '{"bars": [{"model": "qwen3.6-27b", "status": "Ready"}, {"model": "fake-alias", "status": "Starting"}]}',
    )
    # Null values fail.
    assert not _checker_pass(
        bars,
        '{"bars": [{"model": null, "status": null}, {"model": null, "status": null}]}',
    )


def test_negation_window_catches_split_negation():
    # Negation across a clause: "not currently ready" must not satisfy
    # "ready" (the guard scans the three tokens before the match).
    checker = {"type": "terms", "required": ["ready"]}
    assert _checker_pass(checker, "The chip is Ready.")
    assert not _checker_pass(checker, "The chip is not currently ready.")
    assert not _checker_pass(checker, "It does not say ready anywhere.")
    # Four tokens away is outside the window: no longer a negation of the
    # matched phrase ("not X Y Z ready" reads as a new clause).
    assert _checker_pass(checker, "not installed yet, though ready")


def _pass(turn, ttft_s, cached_tokens):
    return {"turn": turn, "ttft_s": ttft_s, "cached_tokens": cached_tokens}


def test_resume_gate_counts_expected_samples_per_pass():
    from scripts.bench_qwen36_media_prefix import _resume_gate_samples

    # Turn 1 is the documented store turn; turns 2+ are resume slots.
    # Gating is per measured sample: a slot whose cached_tokens median is
    # positive because only one of two passes resumed is still a miss for
    # the cold pass (with slots+medians this would falsely count as
    # covered).
    auto_passes = [
        [_pass(0, 1.0, 0), _pass(1, 1.0, 0), _pass(2, 0.5, 64), _pass(3, 0.4, 64)],
        [_pass(0, 1.0, 0), _pass(1, 1.0, 0), _pass(2, 0.9, 0), _pass(3, 0.4, 64)],
    ]
    expected, resumed, misses = _resume_gate_samples(auto_passes)
    assert (expected, resumed) == (4, 3)
    assert misses == [{"pass": 1, "turn": 2}]


def test_resume_regressions_compare_each_resumed_sample_to_baseline():
    from scripts.bench_qwen36_media_prefix import _resume_regressions

    # Baseline median TTFT per turn (from the off phase).
    baseline = {2: 0.5, 3: 0.5}
    auto_passes = [
        [_pass(1, 9.9, 0), _pass(2, 0.54, 64), _pass(3, 0.9, 64)],
        [_pass(1, 9.9, 0), _pass(2, 0.46, 64), _pass(3, 0.2, 64)],
    ]
    flagged = _resume_regressions(auto_passes, baseline, margin=1.15)
    # Pass 0's turn-3 sample (0.9s) is 1.8x the baseline median; the same
    # slot's other pass (0.2s) is fine — per-sample gating flags the slow
    # one instead of letting the median hide it. Turn-2 samples straddle
    # the margin without crossing it. Store-turn samples (turn 1) never
    # count regardless of latency, and cold samples are never compared.
    assert flagged == [
        {
            "pass": 0,
            "turn": 3,
            "ttft_s": 0.9,
            "baseline_median_ttft_s": 0.5,
            "cached_tokens": 64,
        }
    ]
    # A resume slower than a missing baseline entry cannot be judged.
    assert _resume_regressions([[_pass(7, 9.9, 64)]], {}, margin=1.15) == []


def test_line_expect_pins_image_order():
    # Two-toolbar turns report one line per bar; swapping the bars is a
    # wrong answer even though every term is present somewhere.
    checker = {
        "type": "terms",
        "required": ["ready", "idle"],
        "min_lines": 2,
        "max_lines": 2,
        "line_expect": [["ready"], ["idle"]],
    }
    assert _checker_pass(checker, "Ready\nIdle")
    assert not _checker_pass(checker, "Idle\nReady")
    # A missing second line fails; line_expect pins which line carries the
    # terms, not line exclusivity — but the terms must be on THEIR line.
    assert not _checker_pass(checker, "Ready")
    assert _checker_pass(checker, "Ready\nIdle and download available")
    assert not _checker_pass(checker, "Idle and download available\nReady")


def test_required_any_groups_require_every_group():
    # A summary of two prior answers must anchor BOTH: a flat
    # required_any let a summary drop an entire answer.
    checker = {
        "type": "any",
        "min_words": 5,
        "required_any_groups": [[["qwen3.6"], ["ready"]], [["icon"], ["info"]]],
    }
    both = "The model is qwen3.6-27b with a Ready chip and an info icon."
    assert _checker_pass(checker, both)
    # Covers answer 1 only.
    assert not _checker_pass(checker, "The model is qwen3.6-27b and its chip is Ready.")
    # Covers answer 2 only.
    assert not _checker_pass(checker, "A circular information icon is visible.")


def test_json_shape_still_parses_fenced_payload():
    # The JSON turns ask for a fenced code block, so a fence wrapping
    # exactly one JSON object is the requested markup, not prose.
    checker = {"type": "json_shape", "keys": ["a"], "required": ["x"]}
    assert _checker_pass(checker, '```json\n{"a": "x"}\n```')
    # Prose outside the fence still fails the whole-payload parse.
    assert not _checker_pass(checker, 'Here you go:\n```json\n{"a": "x"}\n```')
    assert not _checker_pass(checker, '```json\n{"a": "x"}\n```\nHope that helps!')
