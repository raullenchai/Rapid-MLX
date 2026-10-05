import argparse
import json
import time
from pathlib import Path

import httpx

FILLER = "The blue harbor has three boats, and the red harbor has five boats. " * 170
TASKS = {
    "arithmetic": (
        FILLER + "\nWhat is 37 times 48? Answer with only the integer.",
        lambda reply: reply.strip() == "1776",
    ),
    "code": (
        FILLER
        + "\nWrite a Python function named dedupe that removes duplicate integers "
        "from a list while preserving their first-seen order. Return code only.",
        lambda reply: "def dedupe" in reply and "return" in reply,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18940")
    args = parser.parse_args()
    with httpx.Client(timeout=180) as client, args.out.open("a") as output:
        for name, (prompt, check) in TASKS.items():
            t0 = time.monotonic()
            response = client.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": 0,
                    "max_tokens": 128,
                },
            )
            response.raise_for_status()
            payload = response.json()
            reply = payload["choices"][0]["message"]["content"] or ""
            row = {
                "arm": args.arm,
                "task": name,
                "success": check(reply),
                "reply": reply,
                "elapsed_s": round(time.monotonic() - t0, 3),
                "usage": payload.get("usage"),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(name, row["success"], row["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
