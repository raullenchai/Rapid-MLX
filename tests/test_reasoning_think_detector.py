# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``ThinkDetector`` — runtime autonomous-think routing.

The detector is the migration target for the name-regex parser dispatch
in ``model_auto_config.py``. It makes the routing decision from the
model's emitted bytes (does the response open with ``<think>``?), not
from the model name — so once aliases declare ``can_emit_think``, the
detector replaces every name-regex entry that today maps a model family
to a specific reasoning parser.

These tests pin the detector contract; the dispatch wiring itself is a
follow-up.
"""

from __future__ import annotations

from vllm_mlx.reasoning.think_detector import (
    ThinkDetector,
    looks_like_autonomous_think,
)


def test_looks_like_autonomous_think_simple():
    assert looks_like_autonomous_think("<think>")
    assert looks_like_autonomous_think("<think>Step 1: ...")


def test_looks_like_autonomous_think_with_leading_whitespace():
    """Distilled-on-Qwen models often emit a couple of space tokens
    before the opener."""
    assert looks_like_autonomous_think("  <think>")
    assert looks_like_autonomous_think("\n\n<think>")


def test_looks_like_autonomous_think_negatives():
    assert not looks_like_autonomous_think("")
    assert not looks_like_autonomous_think("The answer is 42")
    assert not looks_like_autonomous_think("Okay let me think.")
    # ``<think>`` deep inside a response is NOT an autonomous opener.
    long_prefix = "A" * 200 + "<think>"
    assert not looks_like_autonomous_think(long_prefix)


def test_detector_routes_to_thinking_on_opener():
    det = ThinkDetector()
    assert det.feed("<think>")
    assert det.is_decided


def test_detector_commits_to_content_after_threshold():
    det = ThinkDetector()
    # First 30 chars don't open with <think>
    assert not det.feed("The answer is 42 right now")
    assert not det.is_decided
    # Past the inspection window
    assert not det.feed("A" * (ThinkDetector.INSPECT_BYTES + 5))
    assert det.is_decided


def test_detector_decision_is_sticky():
    det = ThinkDetector()
    det.feed("<think>reasoning here")
    # Even if subsequent text doesn't look thinking-shaped, the routing
    # stays.
    assert det.feed("regular content from now on")


def test_detector_reset():
    det = ThinkDetector()
    det.feed("<think>")
    det.reset()
    assert not det.is_decided
    assert not det.feed("plain text without opener" + "A" * 100)
