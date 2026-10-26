"""CPU-only full-postprocessor scaling probe (fixture construction excluded)."""

import json
import statistics
import time

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.service.postprocessor import StreamingPostProcessor


def measure(size, repetitions=3, *, verify=True):
    raw = (
        '<tool_call>{"name":"run_shell_command","arguments":{"command":"'
        + "x" * size
        + '"}}</tool_call>'
    )
    chunks = [raw[i : i + 5] for i in range(0, len(raw), 5)]
    outputs = []
    text = ""
    for i, chunk in enumerate(chunks):
        text += chunk
        outputs.append(
            GenerationOutput(
                text=text, new_text=chunk, completion_tokens=i + 1, finished=False
            )
        )
    outputs.append(
        GenerationOutput(text=text, new_text="", finished=True, finish_reason="stop")
    )
    samples = []
    for _ in range(repetitions):
        cfg = reset_config()
        cfg.tool_call_parser = "hermes"
        cfg.enable_auto_tool_choice = True
        cfg.no_thinking = True
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
        }
        processor = StreamingPostProcessor(
            cfg, tools_requested=True, enable_thinking=False, request=request
        )
        events = []
        start = time.process_time_ns()
        for output in outputs:
            events.extend(processor.process_chunk(output))
        events.extend(processor.finalize())
        samples.append((time.process_time_ns() - start) / len(chunks))
        calls = {}
        for event in events:
            for call in event.tool_calls or []:
                target = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                for field in target:
                    target[field] += call.get("function", {}).get(field, "")
        valid_call = False
        if len(calls) == 1:
            call = calls[0]
            try:
                valid_call = call["name"] == "run_shell_command" and json.loads(
                    call["arguments"]
                ) == {"command": "x" * size}
            except ValueError:
                pass
        if verify:
            assert valid_call, "benchmark must retain one complete call"

    reset_config()
    return {
        "call_valid": valid_call,
        "bytes": size,
        "chunks": len(chunks),
        "ns_per_chunk": statistics.median(samples),
    }


def scaling():
    return [measure(size) for size in (1024, 4096, 16384, 65536)]


if __name__ == "__main__":
    print(json.dumps(scaling(), indent=2))
