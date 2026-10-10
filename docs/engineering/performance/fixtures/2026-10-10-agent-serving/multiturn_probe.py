import json
import sys
from probe import ROOT, clear, messages, request, status

label = sys.argv[1]
out = {"label": label, "sessions": []}
for session in range(3):
    clear()
    history = messages(4096, f"growing-session-{session}")
    rows = []
    for turn in range(5):
        row = request(history, 64)
        row["turn"] = turn + 1
        rows.append(row)
        history.append({"role": "assistant", "content": row["output_text"]})
        history.append(
            {
                "role": "user",
                "content": f"Turn {turn + 2}: Explain another property of the arithmetic records. Use one paragraph.",
            }
        )
        print(
            json.dumps(
                {
                    "label": label,
                    "session": session,
                    "turn": turn + 1,
                    "ttft_ms": round(row["ttft_ms"], 1),
                    "usage": row["usage"],
                }
            ),
            flush=True,
        )
    out["sessions"].append(rows)
    (ROOT / (label + "-multiturn.json")).write_text(json.dumps(out, indent=2))
out["status"] = status()
(ROOT / (label + "-multiturn.json")).write_text(json.dumps(out, indent=2))
