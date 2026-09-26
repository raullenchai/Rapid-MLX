"""Run the small CUA next-action ranking probe against System One."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from rapid_general_cue import _loopback_url

QUESTION = (
    "Choose the single next desktop action that most directly and safely "
    "advances the user goal from the described screen."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=lambda value: _loopback_url(value, label="System One URL"),
        default="http://127.0.0.1:8700/v1/rank",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("RAPID_MLX_SYSTEM_ONE_API_KEY")
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("jev_candidate_cases.json"),
    )
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    correct = 0
    latencies = []
    for case in cases:
        request = urllib.request.Request(
            args.url,
            data=json.dumps(
                {
                    "context": case["context"],
                    "question": QUESTION,
                    "answers": case["answers"],
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}),
            },
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        winner = payload["ranked"][0]["candidate"]
        predicted = case["answers"].index(winner)
        passed = predicted == case["expected"]
        correct += int(passed)
        print(
            f"{case['id']}: {'PASS' if passed else 'FAIL'} "
            f"top={predicted} prob={payload['ranked'][0]['prob']:.4f} "
            f"wall={latency_ms:.1f}ms"
        )
    print(
        f"top1={correct}/{len(cases)} "
        f"mean_wall_ms={sum(latencies) / len(latencies):.1f}"
    )


if __name__ == "__main__":
    main()
