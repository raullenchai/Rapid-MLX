# SPDX-License-Identifier: Apache-2.0
"""Identify CLI serve invocations without importing the inference runtime."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

_ENTRYPOINTS = {"rapid-mlx", "rmlx", "vllm-mlx"}
_GLOBAL_FLAGS = {"--no-banner", "--no-telemetry", "--disable-version-check"}


def serve_subcommand_index(argv: Sequence[str]) -> int | None:
    """Return the actual ``serve`` subcommand position, or ``None``.

    Match the executable or Python's script/module operand, never arbitrary
    argument text. uv/pipx console scripts work like ordinary Python scripts;
    uv and caffeinate parent processes are excluded to avoid duplicate PIDs.
    Python's -c and stdin modes must never be mistaken for script execution.
    """
    if not argv:
        return None
    executable = os.path.basename(argv[0])
    index = 1
    if executable not in _ENTRYPOINTS:
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable.lower()):
            return None
        while index < len(argv):
            token = argv[index]
            # ``-m`` may close a cluster of no-argument flags (``-um``).
            module_option = re.fullmatch(r"-[bBdEiIOPqRsSuvx]*m(.*)", token)
            if module_option:
                if not module_option.group(1):
                    index += 1
                    if index == len(argv):
                        return None
                    module = argv[index]
                else:
                    module = module_option.group(1)
                if not any(
                    module == package or module.startswith(package + ".")
                    for package in ("rapid_mlx", "vllm_mlx")
                ):
                    return None
                index += 1
                break
            if token in {"-W", "-X", "--check-hash-based-pycs"}:
                index += 2
                continue
            if token.startswith(("-W", "-X")) or re.fullmatch(
                r"-[bBdEiIOPqRsSuvx]+", token
            ):
                index += 1
                continue
            if token == "--":
                index += 1
            if index == len(argv) or os.path.basename(argv[index]) not in _ENTRYPOINTS:
                return None
            index += 1
            break
        else:
            return None
    while index < len(argv) and argv[index] in _GLOBAL_FLAGS:
        index += 1
    return index if index < len(argv) and argv[index] == "serve" else None
