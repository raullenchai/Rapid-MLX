# SPDX-License-Identifier: Apache-2.0
"""Always-on unit tests for the output-coherence primitives (#1247).

These are pure — no server, no MLX — so they run on every PR on a GitHub-hosted
runner and lock in the *gate logic itself*: that known garbage is flagged, that
real prose is not, and that a coherent-but-wrong answer fails its golden case.
The serve-path half (``evals/coherence_gate.py``) runs on the Apple-Silicon
release gauntlet where a real model is served.

Acceptance for #1247 is "test the gate": the exact class that shipped in #1234
(garbage output) and a coherent-but-wrong regression must both be rejected here.
"""

from __future__ import annotations

import sys

import httpx
import pytest

from evals import coherence_gate
from vllm_mlx.coherence import (
    GOLDEN,
    GoldenCase,
    evaluate_case,
    evaluate_concluded,
    is_degenerate_completion,
    looks_like_garbage,
    strip_thinking,
)

# ── real, coherent outputs that must NOT be flagged as garbage ──────────────
COHERENT = [
    "The capital of Japan is Tokyo.",
    "391",
    "Tokyo",
    "banana",
    # Short legitimate answers — regression guard for the dogfood false
    # positive where "7" tripped char-dominance (1/1 > 0.5). A valid single-
    # token answer must never be judged garbage.
    "7",
    "42",
    "10000",
    "100000",
    "Alabama",
    "9",
    "Yes",
    "No",
    "OK",
    "The ocean is a vast body of saltwater that covers most of Earth's "
    "surface. It teems with life, from microscopic plankton to enormous whales.",
    "A CPU executes instructions: it fetches, decodes, and runs the operations "
    "that make up a program, coordinating the rest of the computer.",
    "Yes — pedestrians are visible walking along the sidewalk on the left.",
]

# ── shipped / plausible garbage that MUST be flagged ────────────────────────
GARBAGE = [
    "",
    "   ",
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",  # prefix-cache poison (#gotchas)
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # doubled-norm degenerate single token
    "。。。。。。。。。。。。。。。。。。。。。。",  # CJK fill
    "the the the the the the the the the the the the the",  # bigram loop
    "cat cat cat cat cat cat cat cat cat cat cat cat",  # word loop
    "Sure! " + "!" * 40,  # long single-char run embedded in otherwise-ok text
    "11111111111111111111",  # long numeric single-token collapse
]


@pytest.mark.parametrize("text", COHERENT)
def test_coherent_text_not_flagged(text: str) -> None:
    is_garbage, reason = looks_like_garbage(text)
    assert not is_garbage, f"false positive on {text[:50]!r}: {reason}"


@pytest.mark.parametrize("text", GARBAGE)
def test_garbage_text_flagged(text: str) -> None:
    is_garbage, _ = looks_like_garbage(text)
    assert is_garbage, f"missed garbage: {text[:50]!r}"


# ── is_degenerate_completion: the boolean #1250 telemetry canary ────────────
@pytest.mark.parametrize("text", COHERENT)
def test_is_degenerate_false_on_coherent(text: str) -> None:
    assert is_degenerate_completion(text) is False


@pytest.mark.parametrize("text", [g for g in GARBAGE if g.strip()])
def test_is_degenerate_true_on_nonempty_garbage(text: str) -> None:
    assert is_degenerate_completion(text) is True


def test_is_degenerate_false_on_empty_is_absence_not_garbage() -> None:
    """Empty / whitespace / None is absence of output — a SEPARATE signal (the
    zero completion-token bucket), not degeneracy. Keep this bool a clean
    "non-empty content looks like garbage" indicator so it never double-counts
    empties with the token-bucket signal."""
    assert is_degenerate_completion("") is False
    assert is_degenerate_completion("   ") is False
    assert is_degenerate_completion(None) is False


def test_evaluate_case_accepts_correct_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    passed, _ = evaluate_case(case, "**Tokyo.**")
    assert passed


def test_evaluate_case_rejects_coherent_but_wrong() -> None:
    """A fluent, non-garbage, but WRONG answer must fail — this is the class no
    perf/import gate catches."""
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    passed, reason = evaluate_case(case, "Tokyo is incorrect; Osaka is the capital.")
    assert not passed
    assert "exact match" in reason


def test_evaluate_case_rejects_garbage_for_golden_prompt() -> None:
    """The exact #1234 failure: the model emits garbage for a golden prompt. The
    BLOCKING layer rejects it deterministically — it is simply not an exact
    match for the expected token — with NO reliance on the heuristic garbage
    detector (which is advisory-only)."""
    case = next(c for c in GOLDEN if c.id == "arithmetic")
    passed, reason = evaluate_case(case, "!!!!!!!!!!!!!!!!!!!!")
    assert not passed
    assert "exact match" in reason


def test_blocking_layer_catches_what_the_detector_cannot() -> None:
    """The review's core point: a frequency heuristic cannot separate diverse
    token soup from prose, so the advisory detector FALSE-GREENS on it — but the
    deterministic golden check rejects it anyway (not an exact match). This is
    exactly why the golden answers are the blocking layer and the detector is
    advisory-only."""
    token_soup = "Ocean qzxv blorp fnarg glip. Water wug zibble plonk traz."
    # The advisory detector honestly cannot tell this from prose:
    assert looks_like_garbage(token_soup)[0] is False
    # But the blocking golden check rejects it — not an exact "Tokyo":
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    assert not evaluate_case(case, token_soup)[0]


def test_evaluate_case_arithmetic_wrong_number_fails() -> None:
    case = next(c for c in GOLDEN if c.id == "arithmetic")
    assert evaluate_case(case, "391.")[0]
    assert not evaluate_case(case, "1391")[0]


def test_evaluate_case_days_rejects_number_containing_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "days-in-week")
    assert not evaluate_case(case, "17")[0]


def test_no_think_leak_case_rejects_raw_reasoning_tag() -> None:
    case = next(c for c in GOLDEN if c.id == "no-think-leak")
    # Correct answer but raw reasoning markers leaked into the visible output.
    leaked = "<think>France's capital is Paris</think> Paris"
    passed, reason = evaluate_case(case, leaked)
    assert not passed
    assert "think" in reason.lower() or "reasoning" in reason.lower()
    # Clean answer passes.
    assert evaluate_case(case, "Paris")[0]


# ── reasoning-distill concluded-answer scoring (#1323) ──────────────────────
def test_strip_thinking_removes_tagged_reasoning_blocks() -> None:
    assert strip_thinking("<think>let me think</think> Tokyo") == "Tokyo"
    assert strip_thinking("<reasoning>trace</reasoning> blue") == "blue"
    # A trailing unclosed thought with no concluded answer strips to empty.
    assert strip_thinking("<think>partial trace") == ""


def test_evaluate_concluded_accepts_tagged_concluded_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    passed, reason = evaluate_concluded(
        case, "<think>Japan is in East Asia, let me recall.</think> Tokyo"
    )
    assert passed
    assert "concluded" in reason


def test_evaluate_concluded_rejects_coherent_but_wrong() -> None:
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    assert not evaluate_concluded(case, "<think>reasoning</think> Osaka")[0]


def test_evaluate_concluded_accepts_only_a_terminal_boxed_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "arithmetic")
    real_deepseek_shape = (
        "To find the product, multiply and add the partial results. "
        "Therefore the product is:\n\\[\n\\boxed{391}\n\\]"
    )
    passed, reason = evaluate_concluded(case, real_deepseek_shape)
    assert passed
    assert "terminal boxed" in reason

    # Merely mentioning the expected result before a different conclusion must
    # not turn the strict golden gate green.
    assert not evaluate_concluded(case, r"Maybe \boxed{391}, but final: 392")[0]
    assert not evaluate_concluded(case, r"Therefore: \boxed{392}")[0]

    word_case = next(c for c in GOLDEN if c.id == "capital-japan")
    assert evaluate_concluded(word_case, r"Therefore: \boxed{\text{Tokyo}}")[0]
    assert evaluate_concluded(word_case, r"Therefore: \boxed{\mathrm{Tokyo}}")[0]
    assert not evaluate_concluded(
        word_case, r"Maybe \boxed{\text{Tokyo}}, but final: Osaka"
    )[0]


def test_evaluate_concluded_rejects_pure_reasoning_no_answer() -> None:
    case = next(c for c in GOLDEN if c.id == "arithmetic")
    passed, reason = evaluate_concluded(case, "<think>17 times 23 is 391...")
    assert not passed
    assert "no concluded answer" in reason


def test_reasoning_distill_untagged_prose_does_not_false_pass() -> None:
    """Untagged chain-of-thought prose that never reaches the terse token must
    still fail the concluded check — the gate strips tags, not prose."""
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    assert not evaluate_concluded(
        case,
        "Okay, so I need to figure out the capital of Japan. Japan is a "
        "country in East Asia. Its largest city is Tokyo, which is where the "
        "government sits. I'm fairly sure the capital is Tok",
    )[0]


@pytest.mark.parametrize(
    ("thinking", "expected_max_tokens"), [(False, 32), (True, 512)]
)
def test_gate_gives_reasoning_distill_enough_budget_to_reach_its_conclusion(
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
    expected_max_tokens: int,
) -> None:
    case = next(c for c in GOLDEN if c.id == "capital-japan")
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "Tokyo"}}]}

    def fake_post(_url: str, *, json: dict[str, object], timeout: float) -> Response:
        captured.update(json)
        assert timeout == 120.0
        return Response()

    monkeypatch.setattr(coherence_gate.httpx, "post", fake_post)
    assert (
        coherence_gate._generate(
            "http://localhost:8000/v1", case, timeout=120.0, thinking=thinking
        )
        == "Tokyo"
    )
    assert captured["max_tokens"] == expected_max_tokens
    assert captured["enable_thinking"] is thinking


def test_gate_returns_infrastructure_code_for_midrun_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coherence_gate, "_server_reachable", lambda _url: True)

    def fail_transport(*_args: object, **_kwargs: object) -> str:
        raise httpx.ConnectError("server stopped")

    monkeypatch.setattr(coherence_gate, "_generate", fail_transport)
    monkeypatch.setattr(sys, "argv", ["coherence_gate.py"])
    assert coherence_gate.main() == 2


def test_gate_turns_invalid_response_content_into_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coherence_gate, "_server_reachable", lambda _url: True)
    monkeypatch.setattr(
        coherence_gate, "_generate", lambda *_args, **_kwargs: ["not", "text"]
    )
    monkeypatch.setattr(sys, "argv", ["coherence_gate.py"])
    assert coherence_gate.main() == 1


def test_gate_returns_infrastructure_code_for_invalid_server_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coherence_gate, "_server_reachable", lambda _url: True)

    def fail_protocol(*_args: object, **_kwargs: object) -> str:
        raise coherence_gate.InvalidServerResponseError("malformed response")

    monkeypatch.setattr(coherence_gate, "_generate", fail_protocol)
    monkeypatch.setattr(sys, "argv", ["coherence_gate.py"])
    assert coherence_gate.main() == 2


def test_golden_set_is_all_deterministic_blocking() -> None:
    """Every golden case must be a deterministic, exact-answer predicate — no
    heuristic / open-ended cases in the blocking set (review convergence)."""
    assert len(GOLDEN) >= 5
    valid_kinds = {"exact", "no_think_leak"}  # deterministic only
    ids = set()
    for c in GOLDEN:
        assert isinstance(c, GoldenCase)
        assert c.kind in valid_kinds, f"{c.id}: non-deterministic kind {c.kind!r}"
        assert c.id not in ids, f"duplicate case id {c.id!r}"
        ids.add(c.id)
        assert c.expect, f"{c.id}: a deterministic case needs a non-empty expect"
        assert c.max_tokens > 0
