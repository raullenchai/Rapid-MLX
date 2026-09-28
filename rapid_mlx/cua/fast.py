"""Fast local thinking: laya outcome ranking + no-progress detection.

The fast path never talks to the cloud. laya (on-device) routes each action
outcome in ~100 ms; the NoProgressTracker catches fixation loops that small
local planners are prone to and injects a recovery hint into the next
slow-thinking prompt.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import httpx

from rapid_mlx.cua.planner import assert_loopback_url

RANKER_MODEL = "convaiinnovations/laya"


class FastOutcomeRanker:
    """Cheap structured outcome routing via the local System One ranker."""

    LABELS = {
        "success": "The requested computer action succeeded.",
        "no_effect": "The requested computer action had no effect.",
        "wrong_effect": "The app changed, but not as requested.",
    }

    def __init__(self, url: str, model: str = RANKER_MODEL, timeout: float = 10.0):
        self.url = assert_loopback_url(url)
        self.model = model
        # The ranker is local-only; never inherit an ambient HTTP proxy.
        self.client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def close(self) -> None:
        await self.client.aclose()

    async def rank(self, context: str, answers: list[str]) -> tuple[list[dict], float]:
        started = __import__("time").perf_counter()
        response = await self.client.post(
            self.url,
            json={
                "model": self.model,
                "context": context,
                "answers": answers,
                "temperature": 1,
            },
        )
        if response.is_error:
            raise RuntimeError(
                f"fast ranker HTTP {response.status_code}: {response.text[:500]}"
            )
        return response.json()["ranked"], __import__("time").perf_counter() - started

    async def assess(self, goal: str, plan: dict, delta: dict) -> tuple[dict, float]:
        context = (
            f"Overall goal: {goal}\n"
            f"Requested action: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Observed structured delta: {json.dumps(delta, ensure_ascii=False)}\n"
            "Rank the descriptions by how accurately they describe the outcome."
        )
        ranked, latency = await self.rank(context, list(self.LABELS.values()))
        reverse = {value: key for key, value in self.LABELS.items()}
        winner = ranked[0]
        return (
            {
                "outcome": reverse[winner["candidate"]],
                "confidence": float(winner["prob"]),
                "source": "system-one-rank",
            },
            latency,
        )


@dataclass
class NoProgressTracker:
    """Detect planner fixation: repeated identical steps or stalled outcomes."""

    repeat_limit: int = 3
    stall_limit: int = 3
    bad_outcomes: set = field(
        default_factory=lambda: {"no_effect", "wrong_effect", "uncertain"}
    )
    instruction_counts: Counter = field(default_factory=Counter)
    consecutive_bad: int = 0
    interventions: int = 0
    last_intervention_was_consecutive: bool = False

    def record(self, plan: dict, outcome: str) -> None:
        instruction = str(plan.get("step_instruction", "")).strip().lower()
        if instruction:
            self.instruction_counts[instruction] += 1
        if outcome in self.bad_outcomes:
            self.consecutive_bad += 1
        else:
            self.consecutive_bad = 0

    def should_intervene(self) -> bool:
        if self.consecutive_bad >= self.stall_limit:
            return True
        if self.instruction_counts:
            _, count = self.instruction_counts.most_common(1)[0]
            if count >= self.repeat_limit:
                return True
        return False

    def take_hint(self, snapshot: dict) -> str:
        """Build the recovery hint and reset one intervention cycle."""
        self.interventions += 1
        top_instruction, top_count = (
            self.instruction_counts.most_common(1)[0]
            if self.instruction_counts
            else ("(none)", 0)
        )
        stalled = self.consecutive_bad >= self.stall_limit
        labels = [
            f"[{e['index']}] {e['label'][:60]}"
            for e in snapshot.get("elements", [])[:12]
        ]
        page_signature = "\n".join(labels) if labels else "(empty snapshot)"
        repeated = top_count >= self.repeat_limit
        parts = [
            "PROGRESS CHECK: the last steps made no visible progress.",
        ]
        if repeated:
            parts.append(
                f'You have issued the same step {top_count} times: "{top_instruction}". '
                "It is not working; stop retrying it."
            )
        if stalled:
            parts.append(
                f"{self.consecutive_bad} consecutive actions had no or wrong effect."
            )
        parts.append(
            "Re-evaluate from the CURRENT snapshot below. Consider: scroll to "
            "reveal content, press Enter after focusing a field, a different "
            "element, or done with an honest blocker summary."
        )
        parts.append(f"Current top elements:\n{page_signature}")
        # reset for the next evaluation cycle
        self.consecutive_bad = 0
        self.instruction_counts.clear()
        return "\n".join(parts)

    def exhausted(self, max_interventions: int = 2) -> bool:
        return self.interventions >= max_interventions
