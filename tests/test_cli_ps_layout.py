# SPDX-License-Identifier: Apache-2.0
"""``rapid-mlx ps`` MODEL/UPTIME column must not collide (#1999).

Serving a local *path* (a converted model) is the normal case, and those
paths routinely exceed the 40-char MODEL column. The old code let them run
straight into UPTIME with no separator; ``_elide_front`` caps the shown model
at 38 chars, keeping the distinctive tail, so the ``<40`` pad always leaves at
least two spaces before UPTIME.
"""

from __future__ import annotations

from vllm_mlx.cli import _elide_front


def test_short_model_is_unchanged():
    assert _elide_front("qwen3.5-4b-4bit", 38) == "qwen3.5-4b-4bit"


def test_exactly_at_width_is_unchanged():
    s = "x" * 38
    assert _elide_front(s, 38) == s


def test_long_path_is_front_elided_and_keeps_the_tail():
    model = "/Users/raullenstudio/mtplx-research/Qwen3.8-27B-MTPLX-Optimized-Speed"
    shown = _elide_front(model, 38)
    assert len(shown) == 38
    assert shown.startswith("…")
    # The distinctive tail (the model name) survives.
    assert shown.endswith("Optimized-Speed")


def test_never_exceeds_width_even_for_tiny_widths():
    # Contract: the result is at most `width` chars, including the degenerate
    # widths that leave no room for the ellipsis.
    for w in (0, 1, 2, 3):
        assert len(_elide_front("abcdefgh", w)) <= w
    assert _elide_front("abcdefgh", 0) == ""
    assert _elide_front("abcdefgh", 1) == "h"
    assert _elide_front("abcdefgh", 2) == "…h"


def test_pad_leaves_a_gap_before_uptime():
    """The rendered ``{model:<40}{uptime:<10}`` must keep >= 2 spaces between
    the (elided) model and the uptime for every input length."""
    for model in ("short", "x" * 40, "/very/long/" + "y" * 80):
        shown = _elide_front(model, 38)
        row = f"  {'12345':<8}{'8123':<8}{shown:<40}{'5h20m':<10}"
        assert "5h20m" in row
        # index where uptime starts minus where the model text ends
        uptime_start = row.index("5h20m")
        model_end = 2 + 8 + 8 + len(shown)
        assert uptime_start - model_end >= 2, f"columns collide: {row!r}"
