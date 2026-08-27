#!/usr/bin/env python3
"""Supplementary GSM8K re-score from saved lm_eval samples (both models, same rule).

The stock `flexible-extract` filter takes the LAST number in the reply, which
penalises chat-style answers that restate context after the answer
("**$64** for the 16 glasses" -> 16) and decimal currency ("$26.00" -> 26.00 != 26).
Rule here: prefer the last **bold** number in the reply; else fall back to the
last number. Normalise by stripping $ and , and trailing .00. This is NOT the
harness metric; it is reported only as a caveat next to the harness number.
"""

import glob
import json
import re
import sys

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def norm(s):
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def extract(resp):
    bold = re.findall(r"\*\*([^*]*?)\*\*", resp)
    for b in reversed(bold):
        nums = re.findall(NUM, b)
        if nums:
            return norm(nums[-1])
    nums = re.findall(NUM, resp)
    return norm(nums[-1]) if nums else ""


for model_dir in sys.argv[1:]:
    files = sorted(
        glob.glob(
            f"/private/tmp/atlas-flash-eval/results/{model_dir}/gsm8k_cot_zeroshot/*/samples_*.jsonl"
        )
    )
    if not files:
        print(model_dir, "no samples")
        continue
    n = ok = harness_ok = 0
    for line in open(files[-1]):
        d = json.loads(line)
        if d["filter"] != "flexible-extract":
            continue
        n += 1
        harness_ok += int(d["exact_match"])
        tgt = norm(d["target"].split("#### ")[-1])
        ok += int(extract(d["resps"][0][0]) == tgt)
    print(
        f"{model_dir}: N={n} harness flexible-extract={100 * harness_ok / n:.1f} bold-aware re-score={100 * ok / n:.1f}"
    )
