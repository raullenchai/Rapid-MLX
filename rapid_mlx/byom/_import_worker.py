# SPDX-License-Identifier: Apache-2.0
"""Child-process steps of ``rapid-mlx import`` (convert, smoke test).

Run in a child interpreter (see ``imports.WORKER_BOOTSTRAP``) so a Ctrl-C,
crash or OOM in MLX can only ever take down this process; the parent owns the
temporary directory and removes it. Nothing here touches the import cache.
"""

from __future__ import annotations


def convert(source: str, out: str, bits: int, group_size: int) -> None:
    from mlx_lm import convert as mlx_convert

    mlx_convert(
        hf_path=source,
        mlx_path=out,
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
    )


def smoke(out: str) -> None:
    """Load the converted model and generate one token."""
    from mlx_lm import generate, load

    model, tokenizer = load(out)
    text = generate(model, tokenizer, prompt="Hello", max_tokens=1, verbose=False)
    if not isinstance(text, str):
        raise SystemExit("smoke test produced no text")


def main(argv: list[str]) -> int:
    step, *rest = argv
    if step == "convert":
        source, out, bits, group_size = rest
        convert(source, out, int(bits), int(group_size))
    elif step == "smoke":
        (out,) = rest
        smoke(out)
    else:
        raise SystemExit(f"unknown step {step!r}")
    return 0
