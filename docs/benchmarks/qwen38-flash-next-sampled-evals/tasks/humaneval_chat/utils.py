"""Filters for the chat-completions-safe HumanEval variant.

`pass_at_k` is copied verbatim from lm_eval/tasks/humaneval/utils.py so the
metric is identical to the stock task. `build_predictions_chat` replaces
`build_predictions_instruct`: instead of assuming an assistant prefill, it
extracts the fenced code block that defines the entry point from a normal chat
reply and prepends the original prompt (imports + signature + docstring; a
function whose body is only a docstring is valid Python, so redefining it below
is harmless).
"""

import re

import evaluate as hf_evaluate

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.S)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\r?\n")


def _extract_code(resp: str, entry_point: str) -> str:
    blocks = _FENCE.findall(resp)
    if blocks:
        defining = [b for b in blocks if f"def {entry_point}" in b]
        return defining[0] if defining else max(blocks, key=len)
    # Unterminated fence (hit max_gen_toks) or no fence at all.
    m = _OPEN_FENCE.search(resp)
    if m:
        return resp[m.end() :]
    return resp


def build_predictions_chat(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    out = []
    for resp, doc in zip(resps, docs):
        preds = []
        for r in resp:
            code = _extract_code(r, doc["entry_point"])
            preds.append(doc["prompt"] + "\n" + code)
        out.append(preds)
    return out
