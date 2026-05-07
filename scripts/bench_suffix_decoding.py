#!/usr/bin/env python3
"""SuffixDecoding PoC benchmark — vanilla vs adaptive suffix-tree drafter.

Runs both decoding paths on the same prompt and reports decode TPS,
acceptance rate, and effective speedup. The PoC is single-request mode
only — no BatchedEngine integration. We're answering one question:

    On agent / code workloads, does adaptive suffix decoding actually
    deliver the 1.5-3x decode speedup the SuffixDecoding paper claims
    on Apple Silicon?

If yes (>= 1.5x on the agent workload, no slowdown on the chat
baseline), we proceed to a full BatchedEngine integration in a
follow-up PR. If no, we kill the project here and pick another
optimization from the survey.

Usage:
    python3.12 scripts/bench_suffix_decoding.py
    python3.12 scripts/bench_suffix_decoding.py \\
        --model mlx-community/Qwen3.5-4B-MLX-4bit \\
        --max-tokens 256 \\
        --json /tmp/sufdec.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as mlx_cache

from vllm_mlx.speculative.suffix_decoding import SuffixDecodingDrafter

# ---------- Workloads ------------------------------------------------------
#
# Each workload mirrors a real-world request shape that we know exercises
# different drafter regimes:
#
#  - "chat":     A normal chat prompt. Drafter has near-zero edit-prompt
#                signal — this is the regression-floor test.
#  - "code_edit": "Re-emit this function with one-line change". The model's
#                 output echoes ~80% of the prompt, so the drafter should
#                 hit very high acceptance rates.
#  - "tool_loop": Repeated tool-call structure (function_name + JSON
#                 schema). A pattern the agent LLM emits over and over;
#                 SuffixDecoding's bread and butter.

CHAT_PROMPT = (
    "You are a helpful assistant.\n\n"
    "User: Briefly explain what speculative decoding is, in two sentences.\n\n"
    "Assistant:"
)

CODE_EDIT_PROMPT = """You are a code refactoring assistant. Re-emit the entire function below \
with ONE change: rename the local variable `total` to `accumulator`. \
Do not change anything else. Reply with the full revised function and nothing else.

```python
def compute_score(records, weights):
    if not records:
        return 0.0
    total = 0.0
    for record, weight in zip(records, weights):
        total += float(record.value) * float(weight)
    if total < 0:
        return 0.0
    return total
```

Revised function:
```python
"""

TOOL_LOOP_PROMPT = """You are a function-calling agent. The available tool is:

{
  "name": "get_weather",
  "description": "Get the current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    },
    "required": ["location"]
  }
}

Emit a tool call for each of the following cities, one per line, in the format:
<tool_call>{"name": "get_weather", "arguments": {"location": "<CITY>"}}</tool_call>

Cities: Tokyo, Paris, London, New York, Berlin, Madrid, Rome, Sydney, Toronto, Beijing.
Begin:
"""

# An agentic ReAct trace — the model has just emitted Thought/Action/
# Observation rounds and is asked to continue. The phrase shape repeats
# heavily, which is the SuffixDecoding sweet spot.
AGENT_REACT_PROMPT = """You are a ReAct agent. You answer by alternating between
Thought, Action, and Observation. The available tool is `search(query)`.

Question: What is the population of the capital of France?

Thought 1: I need to find the capital of France first.
Action 1: search("capital of France")
Observation 1: Paris is the capital of France.

Thought 2: Now I need the population of Paris.
Action 2: search("population of Paris")
Observation 2: Paris has a population of about 2.1 million.

Thought 3: I have the answer.
Action 3: finish("about 2.1 million")

Question: What is the population of the capital of Germany?

Thought 1: I need to find the capital of Germany first.
Action 1: search("capital of Germany")
Observation 1: Berlin is the capital of Germany.

Thought 2:"""

# Structured emit: JSON array of similar objects. Drafter should hit
# very high acceptance on schema keys and structural punctuation.
JSON_ARRAY_PROMPT = """Emit a JSON array of 8 user records. Each record has \
exactly these fields: "id" (integer), "name" (string), "email" (string), \
"role" (one of "admin", "user", "guest"), "active" (boolean). \
The id must increment from 1. Output ONLY the JSON array, no commentary.

["""

# Pure summarize — minimal redundancy. This is the regression-floor test:
# the drafter should NOT slow this down meaningfully (no signal to lock
# onto, drafts get rejected fast).
SUMMARIZE_PROMPT = """Summarize the following passage in a single short paragraph:

The Apollo program was a series of human spaceflight missions undertaken by NASA \
between 1961 and 1972, with the goal of landing astronauts on the Moon. Apollo 11 \
achieved this goal in July 1969, when Neil Armstrong and Buzz Aldrin became the first \
humans to walk on the lunar surface. The program continued through Apollo 17 in 1972, \
returning a wealth of lunar samples and scientific data, before being discontinued due \
to budgetary pressures and shifting national priorities.

Summary:"""


WORKLOADS = {
    "chat": CHAT_PROMPT,
    "code_edit": CODE_EDIT_PROMPT,
    "tool_loop": TOOL_LOOP_PROMPT,
    "agent_react": AGENT_REACT_PROMPT,
    "json_array": JSON_ARRAY_PROMPT,
    "summarize": SUMMARIZE_PROMPT,
}


# ---------- Bench harness --------------------------------------------------


@dataclass
class RunResult:
    workload: str
    mode: str  # "vanilla" or "suffix"
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float
    drafter_stats: dict | None = None
    sample_text: str = ""
    out_tokens: list[int] = field(default_factory=list)
    stopped_on_eos: bool = False

    @property
    def tps(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.completion_tokens / self.wall_time_s


@dataclass
class WorkloadResult:
    workload: str
    vanilla: RunResult
    suffix: RunResult
    speedup: float = 0.0
    # Token-level correctness: compare token IDs up to the shorter of the
    # two stop points. Under greedy, this MUST be zero — non-zero means a
    # real correctness regression in the suffix path.
    common_len: int = 0
    token_diffs_in_common: int = 0
    runs: list[RunResult] = field(default_factory=list)


def _run_vanilla(model, tokenizer, prompt: str, max_tokens: int) -> RunResult:
    """Greedy decode, one token per forward, no spec. The baseline."""
    prompt_ids = tokenizer.encode(prompt)
    cache_state = mlx_cache.make_prompt_cache(model)
    y = mx.array(prompt_ids, mx.uint32)

    # Prefill
    logits = model(y[None], cache=cache_state)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(next_tok)

    out = [next_tok]
    eos_tokens = (
        tokenizer.eos_token_ids
        if hasattr(tokenizer, "eos_token_ids")
        else {tokenizer.eos_token_id}
    )

    stopped_on_eos = next_tok in eos_tokens
    t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        if next_tok in eos_tokens:
            stopped_on_eos = True
            break
        logits = model(mx.array([next_tok], mx.uint32)[None], cache=cache_state)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mx.eval(next_tok)
        out.append(next_tok)
    dt = time.perf_counter() - t0

    text = tokenizer.decode(out)
    return RunResult(
        workload="",
        mode="vanilla",
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(out),
        wall_time_s=dt,
        sample_text=text[:200],
        out_tokens=list(out),
        stopped_on_eos=stopped_on_eos,
    )


def _run_suffix(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    *,
    max_draft: int,
    max_suffix: int,
    min_conf: float,
) -> RunResult:
    """Greedy decode with adaptive suffix-tree spec. Same model, same
    sampler — outputs must match vanilla token-for-token under greedy."""
    prompt_ids = tokenizer.encode(prompt)
    cache_state = mlx_cache.make_prompt_cache(model)

    drafter = SuffixDecodingDrafter(
        max_draft_tokens=max_draft,
        max_suffix_len=max_suffix,
        min_confidence=min_conf,
    )
    drafter.add_prompt_tokens(prompt_ids)

    y = mx.array(prompt_ids, mx.uint32)
    logits = model(y[None], cache=cache_state)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(next_tok)
    out = [next_tok]
    drafter.add_generated_token(next_tok)

    eos_tokens = (
        tokenizer.eos_token_ids
        if hasattr(tokenizer, "eos_token_ids")
        else {tokenizer.eos_token_id}
    )

    stopped_on_eos = next_tok in eos_tokens
    t0 = time.perf_counter()
    while len(out) < max_tokens:
        if next_tok in eos_tokens:
            stopped_on_eos = True
            break

        draft = drafter.get_draft()
        if not draft:
            # No draft — vanilla single-token step
            logits = model(mx.array([next_tok], mx.uint32)[None], cache=cache_state)
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            mx.eval(next_tok)
            out.append(next_tok)
            drafter.add_generated_token(next_tok)
            continue

        # Verify [last, draft_1, ..., draft_k] in one forward; the model
        # produces logits for k+1 positions (including the post-last-draft
        # position which is the "always-emitted" next token if all drafts
        # accept).
        verify_input = mx.array([next_tok, *draft], mx.uint32)[None]
        verify_logits = model(verify_input, cache=cache_state)
        # Take argmax at every position. Position 0 corresponds to the
        # token AFTER `next_tok` (i.e. the verifier's own first prediction);
        # we compare it to draft[0]. Position i corresponds to the token
        # AFTER draft[i-1].
        verify_pred = mx.argmax(verify_logits[0, :, :], axis=-1)
        mx.eval(verify_pred)
        verify_pred_list = verify_pred.tolist()  # length k+1

        n_accepted = 0
        eos_in_draft = False
        for i, dt_tok in enumerate(draft):
            if verify_pred_list[i] == dt_tok:
                n_accepted += 1
                out.append(dt_tok)
                drafter.add_generated_token(dt_tok)
                # Conditioning on tokens past EOS is undefined behavior
                # for most chat models — must stop here for greedy parity
                # with the vanilla path.
                if dt_tok in eos_tokens:
                    eos_in_draft = True
                    break
                if len(out) >= max_tokens:
                    break
            else:
                break

        drafter.record_acceptance(n_accepted)

        if eos_in_draft:
            stopped_on_eos = True
            # Cache may have stale KV entries for un-accepted drafts after
            # the EOS, but since we're stopping, no need to trim.
            break

        if len(out) >= max_tokens:
            break

        # The next "primary" token is verify_pred at the first rejection
        # site (or position k if all drafts accepted).
        next_tok = verify_pred_list[n_accepted]
        out.append(next_tok)
        drafter.add_generated_token(next_tok)

        # Trim KV cache for rejected drafts. We fed k+1 input tokens but
        # only kept (n_accepted + 1) of the resulting positions; drop the
        # remaining (k - n_accepted) from the KV cache so the next forward
        # is consistent.
        rejected = len(draft) - n_accepted
        if rejected > 0:
            mlx_cache.trim_prompt_cache(cache_state, rejected)

    dt = time.perf_counter() - t0
    text = tokenizer.decode(out)
    return RunResult(
        workload="",
        mode="suffix",
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(out),
        wall_time_s=dt,
        drafter_stats=drafter.stats_dict(),
        sample_text=text[:200],
        out_tokens=list(out),
        stopped_on_eos=stopped_on_eos,
    )


def _bench_one_model(
    model_id: str,
    workloads: list[str],
    max_tokens: int,
    max_draft: int,
    max_suffix: int,
    min_conf: float,
) -> dict[str, WorkloadResult]:
    print(f"\n=== model: `{model_id}` ===")
    print("Loading...")
    model, tokenizer = load(model_id)
    print("Loaded.")

    results: dict[str, WorkloadResult] = {}
    for name in workloads:
        prompt = WORKLOADS[name]
        print(f"\n## workload: {name}")

        # Warmup with a tiny vanilla run so the first real run isn't
        # paying for model JIT / weight load.
        _ = _run_vanilla(model, tokenizer, "Hi", 8)

        v = _run_vanilla(model, tokenizer, prompt, max_tokens)
        v.workload = name
        s = _run_suffix(
            model,
            tokenizer,
            prompt,
            max_tokens,
            max_draft=max_draft,
            max_suffix=max_suffix,
            min_conf=min_conf,
        )
        s.workload = name

        speedup = s.tps / v.tps if v.tps > 0 else 0.0

        # Token-level correctness check. Under greedy, the two paths must
        # produce identical token IDs up to the shorter common length.
        # Anything else is a real correctness regression.
        common = min(len(v.out_tokens), len(s.out_tokens))
        diffs = sum(
            1 for a, b in zip(v.out_tokens[:common], s.out_tokens[:common]) if a != b
        )
        results[name] = WorkloadResult(
            workload=name,
            vanilla=v,
            suffix=s,
            speedup=speedup,
            common_len=common,
            token_diffs_in_common=diffs,
            runs=[v, s],
        )
        print(
            f"  vanilla:  {v.tps:6.1f} tok/s  "
            f"({v.completion_tokens} tok in {v.wall_time_s:.2f}s, "
            f"eos={v.stopped_on_eos})"
        )
        print(
            f"  suffix:   {s.tps:6.1f} tok/s  "
            f"({s.completion_tokens} tok in {s.wall_time_s:.2f}s, "
            f"eos={s.stopped_on_eos})"
        )
        if s.drafter_stats:
            ds = s.drafter_stats
            print(
                f"   ↳ drafter: {ds['total_drafts_proposed']} proposals, "
                f"{ds['total_draft_tokens_proposed']} tokens proposed, "
                f"{ds['total_draft_tokens_accepted']} accepted "
                f"(rate {ds['acceptance_rate']:.0%}, "
                f"+{ds['mean_accepted_per_step']:.2f}/step)"
            )
        ok = "✓" if diffs == 0 else "✗"
        print(
            f"  **speedup: {speedup:.2f}x**  "
            f"(token diffs in common-prefix [{common}]: {diffs} {ok})"
        )

    # Drop model refs so MLX can release weights before next model loads.
    del model
    del tokenizer
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Single model id. If omitted, --models takes precedence.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Multi-model sweep — runs all workloads for each model in turn.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation length per run",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=list(WORKLOADS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--max-draft", type=int, default=8)
    parser.add_argument("--max-suffix", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--json", default=None, help="Write raw results JSON")
    args = parser.parse_args()

    if "all" in args.workloads:
        wl_names = list(WORKLOADS.keys())
    else:
        wl_names = args.workloads

    if args.models:
        model_ids = args.models
    elif args.model:
        model_ids = [args.model]
    else:
        model_ids = ["mlx-community/Qwen3-0.6B-8bit"]

    print("# SuffixDecoding PoC benchmark — multi-model sweep")
    print()
    print(f"- models: {model_ids}")
    print(f"- workloads: {wl_names}")
    print(f"- max_tokens: {args.max_tokens}")
    print(
        f"- drafter: max_draft={args.max_draft}, max_suffix={args.max_suffix}, "
        f"min_conf={args.min_conf}"
    )

    all_results: dict[str, dict[str, WorkloadResult]] = {}
    for mid in model_ids:
        try:
            all_results[mid] = _bench_one_model(
                mid,
                wl_names,
                args.max_tokens,
                args.max_draft,
                args.max_suffix,
                args.min_conf,
            )
        except Exception as e:  # noqa: BLE001
            print(f"!! model `{mid}` failed: {e!r}")
            all_results[mid] = {}

    # Aggregated cross-model summary
    print("\n\n# Cross-model summary")
    print()
    print(
        "| model | workload | vanilla tok/s | suffix tok/s | speedup "
        "| accepted/step | tok-diff |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for mid, results in all_results.items():
        for name, r in results.items():
            accept = (
                r.suffix.drafter_stats["mean_accepted_per_step"]
                if r.suffix.drafter_stats
                else 0
            )
            tag = mid.split("/")[-1]
            ok = (
                "0 ✓"
                if r.token_diffs_in_common == 0
                else f"{r.token_diffs_in_common} ✗"
            )
            print(
                f"| {tag} | {name} | {r.vanilla.tps:.1f} | {r.suffix.tps:.1f} | "
                f"{r.speedup:.2f}x | {accept:.2f} | {ok} |"
            )

    if args.json:
        out = {
            "max_tokens": args.max_tokens,
            "drafter_config": {
                "max_draft": args.max_draft,
                "max_suffix": args.max_suffix,
                "min_conf": args.min_conf,
            },
            "models": {
                mid: {
                    "workloads": {
                        name: {
                            "vanilla_tps": r.vanilla.tps,
                            "suffix_tps": r.suffix.tps,
                            "speedup": r.speedup,
                            "vanilla_completion_tokens": r.vanilla.completion_tokens,
                            "suffix_completion_tokens": r.suffix.completion_tokens,
                            "vanilla_stopped_on_eos": r.vanilla.stopped_on_eos,
                            "suffix_stopped_on_eos": r.suffix.stopped_on_eos,
                            "common_len": r.common_len,
                            "token_diffs_in_common": r.token_diffs_in_common,
                            "drafter_stats": r.suffix.drafter_stats,
                            "vanilla_text": r.vanilla.sample_text,
                            "suffix_text": r.suffix.sample_text,
                        }
                        for name, r in results.items()
                    }
                }
                for mid, results in all_results.items()
            },
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote raw results: {args.json}")


if __name__ == "__main__":
    main()
