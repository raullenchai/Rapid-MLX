import argparse
import json
import time
from pathlib import Path

import httpx

def prompt(repetitions: int, nonce: str) -> str:
    filler = "The blue harbor has three boats, and the red harbor has five boats. "
    return (
        f"Experiment {nonce}. Remember: the secret code is ZEBRA-4417. "
        + filler * repetitions
        + "\nAnswer with only the secret code."
    )


def run_one(
    client: httpx.Client,
    base_url: str,
    model: str,
    arm: str,
    repetitions: int,
    nonce: str,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(repetitions, nonce)}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 24,
    }
    t0 = time.monotonic()
    first = None
    pieces = []
    usage = None
    with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=body, timeout=180
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                token = choice.get("delta", {}).get("content") or ""
                if token:
                    if first is None:
                        first = time.monotonic()
                    pieces.append(token)
    done = time.monotonic()
    return {
        "arm": arm,
        "repetitions": repetitions,
        "nonce": nonce,
        "ttft_s": round(first - t0, 3) if first else None,
        "total_s": round(done - t0, 3),
        "reply": "".join(pieces),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18940")
    parser.add_argument("--reps", type=int, nargs="+", default=[8, 170, 650])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    with httpx.Client() as client, args.out.open("a") as output:
        for repetitions in args.reps:
            for index in range(args.rounds):
                nonce = f"ane-probe-{repetitions}-{index}"
                row = run_one(
                    client, args.base_url, args.model, args.arm, repetitions, nonce
                )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
