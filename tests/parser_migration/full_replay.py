"""Archived focused and registry replay. Usage: full_replay.py TREE MODE OUTPUT."""

import itertools
import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[2] not in {"focused", "matrix"}:
        raise ValueError("expected TREE focused|matrix OUTPUT")
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    from harness import route
    from rapid_mlx.reasoning import list_parsers
    from rapid_mlx.tool_parsers import ToolParserManager

    mode = sys.argv[2]
    out = Path(sys.argv[3])
    rawcall = '<tool_call>{"name":"run_shell_command","arguments":{"command":"pwd"}}</tool_call>'
    native = {
        "hermes": rawcall,
        "glm47": "<tool_call>run_shell_command<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
        "mistral": '[TOOL_CALLS] [{"name":"run_shell_command","arguments":{"command":"pwd"}}]',
        "qwen3_coder_xml": "<tool_call><function=run_shell_command><parameter=command>pwd</parameter></function></tool_call>",
        "harmony": '<|start|>assistant to=run_shell_command<|channel|>commentary<|message|>{"command":"pwd"}<|call|>',
        "gemma4": "<|tool_call>call:run_shell_command{command:<|string|>pwd<|string|>}<tool_call|>",
        "deepseek_v3": '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>run_shell_command\n```json\n{"command":"pwd"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    }
    scenarios = []
    if mode == "matrix":
        for r, t in itertools.product(
            [None] + list_parsers(), ToolParserManager.list_registered()
        ):
            scenarios.append((r, t, "plain", "Hello world. This is a streamed answer."))
    else:
        for r, t in [
            ("qwen3", "hermes"),
            ("qwen3", "glm47"),
            ("deepseek_r1", "deepseek_v3"),
            ("gpt_oss", "harmony"),
            ("harmony", "harmony"),
            ("gemma4", "gemma4"),
            (None, "mistral"),
            (None, "qwen3_coder_xml"),
        ]:
            for name, raw in [
                ("plain", "Hello world. This is a streamed answer."),
                ("explicit", "<think>Secret.</think>Answer"),
                ("implicit", "Secret.</think>Answer"),
                ("nested", "<think>Secret.<think>Nested</think>Answer</think>Tail"),
                ("native", native[t]),
                ("promoted", "<think>Secret." + native[t] + "</think>Answer"),
                (
                    "orphan",
                    "<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>Answer",
                ),
                (
                    "malformed",
                    '<tool_call>{"name":"run_shell_command","arguments":oops}</tool_call>Answer',
                ),
                ("literal", "Example <tool_call> and <|im_end|> are literal markers."),
                (
                    "fenced",
                    '```json\n{"name":"run_shell_command","arguments":{"command":"pwd"}}\n```'
                    + native[t],
                ),
                (
                    "harmony",
                    "<|start|>assistant<|channel|>analysis<|message|>Secret<|end|><|start|>assistant<|channel|>final<|message|>Answer<|return|>",
                ),
                ("gemma", "<|channel>thought\nSecret<channel|>Answer"),
            ]:
                scenarios.append((r, t, name, raw))
    done = (
        {tuple(json.loads(line)["key"]) for line in out.read_text().splitlines()}
        if out.exists()
        else set()
    )
    prior = os.environ.get("PRIOR")
    if prior:
        done.update(
            tuple(json.loads(line)["key"])
            for line in Path(prior).read_text().splitlines()
        )
    with out.open("a") as f:
        for scenario_i, (r, t, name, raw) in enumerate(scenarios):
            if scenario_i % int(os.environ.get("SHARDS", "1")) != int(
                os.environ.get("SHARD", "0")
            ):
                continue
            for thinking, tools_mode, surface, stream in itertools.product(
                [False, True],
                ["absent", "enabled"],
                ["chat", "responses", "messages"],
                [False, True],
            ):
                splits = (
                    ["chars"]
                    if mode == "matrix" or not stream
                    else ["chars", "whole", "seven"]
                )
                for split in splits:
                    chunks = (
                        list(raw)
                        if split == "chars"
                        else [raw]
                        if split == "whole"
                        else [raw[i : i + 7] for i in range(0, len(raw), 7)]
                    )
                    if (
                        r,
                        t,
                        name,
                        thinking,
                        tools_mode,
                        surface,
                        stream,
                        split,
                    ) in done:
                        continue
                    result = route(
                        r,
                        t,
                        raw,
                        surface,
                        stream,
                        chunks,
                        thinking,
                        tools_mode=tools_mode,
                    )
                    f.write(
                        json.dumps(
                            {
                                "key": [
                                    r,
                                    t,
                                    name,
                                    thinking,
                                    tools_mode,
                                    surface,
                                    stream,
                                    split,
                                ],
                                "result": result,
                            }
                        )
                        + "\n"
                    )
            f.flush()
    print(out, "done", flush=True)
