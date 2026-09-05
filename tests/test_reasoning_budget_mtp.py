# SPDX-License-Identifier: Apache-2.0
"""``ReasoningBudgetLogitsProcessor`` as a transactional MTP processor (#3044).

The continuous MTP lane admits a request only when its logits-processor row
is, object for object, the scheduler's ``_mtp_safe_logits_processors``; a
processor with per-request mutable state qualifies only by exposing the
snapshot / restore / apply contract ``spec_decode/mtp/generator.py`` drives
(see ``GrammarLogitsProcessor`` and ``AgentRepetitionLogitsProcessor``).
Before #3044 the thinking budget exposed none of it, so every request that
carried a ``reasoning_max_tokens`` cap (every graded ``reasoning_effort`` on
a template without native levels) silently fell back to ordinary decode.

These tests are pure phase-machine checks (no mlx): the budget's ``free`` and
``generation`` phases return the incoming logits object untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.api.reasoning_budget import ReasoningBudgetLogitsProcessor

THINK_END = 99
THINK_START = 50
PROMPT = [1, 2, 3]
LOGITS = SimpleNamespace(shape=(1, 128))


def _tentative(*ids: int) -> SimpleNamespace:
    """The verifier's draft-prefix array: only ``size`` and ``tolist`` matter."""
    return SimpleNamespace(size=len(ids), tolist=lambda: list(ids))


def test_budget_exposes_the_generator_transaction_contract():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 3)
    assert callable(proc.mtp_apply)
    assert callable(proc.mtp_snapshot_state)
    assert callable(proc.mtp_restore_state)


def test_snapshot_restore_round_trips_every_phase_counter():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 3, seeded_thinking=True)
    proc(PROMPT, LOGITS)
    proc(PROMPT + [10], LOGITS)
    boundary = proc.mtp_snapshot_state()
    proc(PROMPT + [10, 11, THINK_END, 12], LOGITS)
    assert proc._ended is True
    proc.mtp_restore_state(boundary)
    assert (proc._prompt_len, proc._committed, proc._think_count) == (3, 4, 1)
    assert proc._started is True
    assert proc._ended is False
    assert proc._force_logged is False
    assert proc.mtp_snapshot_state() == boundary


def test_mtp_apply_walks_the_draft_prefix_tentatively_and_restore_rolls_back():
    """Verification rows share ordinary decode's cumulative history; the
    draft prefix advances the counters only until the generator restores the
    target-accepted boundary."""
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 4, seeded_thinking=True)
    proc(PROMPT, LOGITS)  # baseline the prompt like the first decode step
    boundary = proc.mtp_snapshot_state()
    # Row 0: the last committed token only. Rows 1..2: one and two drafts.
    assert proc.mtp_apply(PROMPT + [10], _tentative(), LOGITS) is LOGITS
    assert proc.mtp_apply(PROMPT + [10, 11], _tentative(11), LOGITS) is LOGITS
    assert proc.mtp_apply(PROMPT + [10, 11, 12], _tentative(11, 12), LOGITS) is LOGITS
    assert proc._think_count == 3
    # One more tentative token would spend the budget (phase only: the force
    # row itself needs real mlx logits and is covered by the mlx suite).
    assert proc._phase(PROMPT + [10, 11, 12, 13]) == "force"
    # Target rejected every draft: back to the pre-proposal boundary.
    proc.mtp_restore_state(boundary)
    assert proc._think_count == 0
    assert proc._committed == len(PROMPT)
    assert proc._phase(PROMPT + [10]) == "free"


def test_restore_to_an_accepted_row_keeps_only_that_prefix():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 4, seeded_thinking=True)
    proc(PROMPT, LOGITS)
    states = []
    for row in ([10], [10, 11], [10, 11, 12]):
        proc.mtp_apply(PROMPT + row, _tentative(*row[1:]), LOGITS)
        states.append(proc.mtp_snapshot_state())
    # One draft accepted: the boundary after row 1 becomes committed state.
    proc.mtp_restore_state(states[1])
    assert proc._think_count == 2
    assert proc._committed == len(PROMPT) + 2
    # The next ordinary steps continue from there: 4/4 spent → force.
    assert proc._phase(PROMPT + [10, 11, 13]) == "free"
    assert proc._phase(PROMPT + [10, 11, 13, 14]) == "force"


def test_mtp_apply_uses_the_cumulative_history_not_the_tentative_arg():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 3, seeded_thinking=True)
    proc(PROMPT, LOGITS)
    poisoned = SimpleNamespace(size=5, tolist=lambda: [THINK_END] * 5)
    proc.mtp_apply(PROMPT + [10], poisoned, LOGITS)
    assert proc._think_count == 1
    assert proc._ended is False


def test_generation_phase_is_inert_under_mtp_too():
    proc = ReasoningBudgetLogitsProcessor(THINK_END, 3, seeded_thinking=True)
    proc(PROMPT, LOGITS)
    proc(PROMPT + [THINK_END], LOGITS)
    assert proc._ended is True
    assert proc.mtp_apply(PROMPT + [THINK_END, 20], _tentative(20), LOGITS) is LOGITS


def test_emitting_template_transitions_inside_a_draft_row_and_rolls_back():
    proc = ReasoningBudgetLogitsProcessor(
        THINK_END, 3, think_start_id=THINK_START, seeded_thinking=False
    )
    proc(PROMPT, LOGITS)
    boundary = proc.mtp_snapshot_state()
    proc.mtp_apply(PROMPT + [THINK_START, 10], _tentative(10), LOGITS)
    assert proc._started is True
    assert proc._think_count == 1
    proc.mtp_restore_state(boundary)
    assert proc._started is False
    assert proc._think_count == 0
