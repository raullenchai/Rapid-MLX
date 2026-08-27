#!/usr/bin/env python3
"""Run the fixed 20-case Qwen4-Exp panel for sequential greedy parity."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from qwen4_exp_real_parity import _load

ORDINARY_CASES = {
    "R1": "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. Give the ball's cost in cents and briefly justify it.",
    "R2": "Three boxes are labeled Apples, Oranges, and Mixed, but every label is wrong. You may draw one fruit from one box. Which box do you draw from, and how do you relabel all three?",
    "R3": "A train travels 120 km at 60 km/h and returns 120 km at 40 km/h. What is its average speed for the whole trip? Show the decisive calculation.",
    "R4": "Find the smallest positive integer n such that n leaves remainder 1 when divided by 2, 3, 4, 5, and 6, and is divisible by 7. Give n and a short verification.",
    "C1": "Write a Python function merge_intervals(items) that merges overlapping closed integer intervals. Return only one fenced Python block. Example: [(1,3),(2,6),(8,10)] -> [(1,6),(8,10)]. Do not mutate the input.",
    "C2": "Write a Python function first_unique(s) returning the first Unicode character that occurs exactly once, or None. Return only one fenced Python block. It must handle the empty string.",
    "C3": "Write a Python function topo_sort(nodes, edges) returning a deterministic topological order, choosing the lexicographically smallest available node; raise ValueError on a cycle. Return only one fenced Python block.",
    "C4": "Write a Python function json_pointer_get(doc, pointer) implementing RFC 6901 lookup for objects and arrays, including ~0 and ~1 unescaping and the empty pointer. Return only one fenced Python block.",
    "Z1": "用中文简要解释为什么蒙提霍尔问题中换门的胜率是三分之二。限四句话。",
    "Z2": "把下面这句话翻译成自然、专业的中文，不要添加解释：The cache must preserve explicit operator overrides while choosing a safe automatic default.",
    "Z3": "某商品先涨价20%，再降价20%。请用中文说明最终价格相对原价的变化，并给出百分比。",
    "Z4": "请写一个恰好包含三项的编号清单，说明本地大模型服务启动失败时应先检查什么。每项不超过十五个汉字。",
}

TOOLS = {
    "T1": (
        "What is the weather in Tokyo in Celsius?",
        "get_weather",
        {"city": {"type": "string"}, "unit": {"type": "string"}},
        ["city", "unit"],
    ),
    "T2": (
        "Look up the current price for AAPL.",
        "get_stock_price",
        {"symbol": {"type": "string"}},
        ["symbol"],
    ),
    "T3": (
        "Schedule a 30 minute meeting called Release audit at 2026-09-02T10:30:00-07:00 with alice@example.com and bob@example.com.",
        "schedule_meeting",
        {
            "title": {"type": "string"},
            "start": {"type": "string"},
            "duration_minutes": {"type": "integer"},
            "participants": {"type": "array", "items": {"type": "string"}},
        },
        ["title", "start", "duration_minutes", "participants"],
    ),
    "T4": (
        "Search the local docs for hybrid prefix cache admission and return the top 3 matches.",
        "search_docs",
        {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        ["query", "top_k"],
    ),
}


def _recall_document(rows: int = 800) -> str:
    facts = {
        37: "Project Alder's launch code is HARBOR-7319.",
        173: "The north warehouse closes every Tuesday at 18:40.",
        401: "Sample R-88 must be stored at minus 17 degrees Celsius.",
        733: "Dr. Lin's emergency extension is 6042.",
    }
    lines = []
    for index in range(rows):
        lines.append(
            f"Record {index:04d}: inventory batch {10000 + index} was inspected "
            "and archived under routine policy."
        )
        if index in facts:
            lines.append(facts[index])
    return "\n".join(lines)


def _cases(recall_rows: int = 800):
    for case_id, prompt in ORDINARY_CASES.items():
        yield case_id, prompt, None
    for case_id, (prompt, name, properties, required) in TOOLS.items():
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Invoke {name}.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        yield case_id, prompt, [tool]
    document = _recall_document(recall_rows)
    questions = {
        "L1": "What is Project Alder's launch code? Return only the code.",
        "L2": "When does the north warehouse close on Tuesdays? Return only the time.",
        "L3": "At what temperature must sample R-88 be stored? Return only the temperature and unit.",
        "L4": "What is Dr. Lin's emergency extension? Return only the digits.",
    }
    for case_id, question in questions.items():
        yield case_id, f"Use only the document below.\n\n{document}\n\n{question}", None


def _capture_wrapper(original, captured):
    def capture(self, *call_args, **call_kwargs):
        value = original(self, *call_args, **call_kwargs)
        captured.append(value)
        return value

    return capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("rapid", "upstream"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--case-id")
    parser.add_argument("--case-prefix")
    parser.add_argument("--logits-output", type=Path)
    parser.add_argument("--layers-output", type=Path)
    parser.add_argument("--qsa-mask-output", type=Path)
    parser.add_argument("--full-layers", action="store_true")
    parser.add_argument("--disable-upstream-fused-rope", action="store_true")
    parser.add_argument("--recall-rows", type=int, default=800)
    args = parser.parse_args()

    _holder, layers, language = _load(args.checkpoint.resolve(), args.backend)
    if args.disable_upstream_fused_rope:
        if args.backend != "upstream":
            parser.error("--disable-upstream-fused-rope requires --backend upstream")
        for layer in layers:
            attention = getattr(layer, "self_attn", None)
            rotary = getattr(attention, "rotary_emb", None)
            if rotary is not None:
                rotary.fused_apply = False
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(args.checkpoint.resolve())
    eos_ids = set(tokenizer.eos_token_ids)
    results = {}
    diagnostic_logits = {}
    diagnostic_layers = {}
    diagnostic_qsa_masks = {}
    for case_index, (case_id, prompt, tools) in enumerate(_cases(args.recall_rows)):
        if case_index >= args.max_cases:
            break
        if args.case_id is not None and case_id != args.case_id:
            continue
        if args.case_prefix is not None and not case_id.startswith(args.case_prefix):
            continue
        template_args = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }
        if tools is not None:
            template_args["tools"] = tools
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **template_args
        )
        cache = language.make_cache()
        started = time.monotonic()
        captured_layers = []
        captured_qsa_masks = []
        layer_class = type(layers[0])
        original_layer_call = layer_class.__call__
        indexer_class = type(layers[3].self_attn.indexer)
        original_indexer_call = indexer_class.__call__

        def capture_indexer(
            self,
            *call_args,
            _original=original_indexer_call,
            _captured=captured_qsa_masks,
            **call_kwargs,
        ):
            value = _original(self, *call_args, **call_kwargs)
            if value is not None:
                array = np.asarray(value)
                _captured.append((array.shape, np.packbits(array, axis=-1)))
            return value

        if args.layers_output is not None:
            layer_class.__call__ = _capture_wrapper(
                original_layer_call, captured_layers
            )
        if args.qsa_mask_output is not None:
            indexer_class.__call__ = capture_indexer
        try:
            output = language(mx.array([input_ids], dtype=mx.int32), cache=cache)
        finally:
            layer_class.__call__ = original_layer_call
            indexer_class.__call__ = original_indexer_call
        if captured_layers:
            diagnostic_layers[case_id] = np.stack(
                [
                    np.asarray(
                        (value if args.full_layers else value[:, -1, :]).astype(
                            mx.float32
                        )
                    )[0]
                    for value in captured_layers
                ]
            )
        if captured_qsa_masks:
            diagnostic_qsa_masks[case_id] = captured_qsa_masks
        logits = output if args.backend == "rapid" else output.logits
        generated = []
        case_logits = []
        for _ in range(args.tokens):
            if args.logits_output is not None:
                case_logits.append(np.asarray(logits[:, -1, :].astype(mx.float32))[0])
            token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(token)
            if token in eos_ids:
                break
            output = language(mx.array([[token]], dtype=mx.int32), cache=cache)
            logits = output if args.backend == "rapid" else output.logits
        elapsed = time.monotonic() - started
        results[case_id] = {
            "prompt_tokens": len(input_ids),
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated),
            "wall_s": round(elapsed, 3),
        }
        if case_logits:
            diagnostic_logits[case_id] = np.stack(case_logits)
        print(
            json.dumps(
                {
                    "case": case_id,
                    "prompt_tokens": len(input_ids),
                    "generated": len(generated),
                    "wall_s": round(elapsed, 3),
                }
            ),
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "tokens": args.tokens,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        del cache, output, logits
        gc.collect()
        mx.clear_cache()

    if args.logits_output is not None:
        args.logits_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.logits_output, **diagnostic_logits)
    if args.layers_output is not None:
        args.layers_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.layers_output, **diagnostic_layers)
    if args.qsa_mask_output is not None:
        args.qsa_mask_output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        metadata = {}
        for case_id, masks in diagnostic_qsa_masks.items():
            metadata[case_id] = [shape for shape, _packed in masks]
            for layer_index, (_shape, packed) in enumerate(masks):
                arrays[f"{case_id}_{layer_index}"] = packed
        arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez(args.qsa_mask_output, **arrays)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
