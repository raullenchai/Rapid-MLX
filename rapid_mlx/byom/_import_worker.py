# SPDX-License-Identifier: Apache-2.0
"""Child-process steps of ``rapid-mlx import`` (convert, smoke test).

Run in a child interpreter (see ``imports.WORKER_BOOTSTRAP``) so a Ctrl-C,
crash or OOM in MLX can only ever take down this process; the parent owns the
temporary directory and removes it. Nothing here touches the import cache.
"""

from __future__ import annotations


def convert(source: str, out: str, bits: int, group_size: int) -> None:
    from ..utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    from mlx_lm import convert as mlx_convert
    from mlx_lm.utils import _download

    from ..model_load_errors import validate_model_config_file
    from ..utils.model_file_guard import (
        remote_code_enabled,
        validate_local_model_file,
        validate_model_code,
    )

    # Conversion executes the source model too. Inspect the exact snapshot
    # passed to the converter before its loader can import repository Python.
    source_path = _download(source)
    config = validate_model_config_file(source_path)
    trust = remote_code_enabled()
    validate_model_code(config or {}, trust_remote_code=trust)
    validate_local_model_file(source_path)

    mlx_convert(
        hf_path=str(source_path),
        mlx_path=out,
        quantize=True,
        q_bits=bits,
        q_group_size=group_size,
        trust_remote_code=trust,
    )


def smoke(out: str) -> None:
    """Load the converted model and generate one token."""
    from ..utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    from mlx_lm import generate

    from ..utils.tokenizer import load_model_with_fallback

    loaded = load_model_with_fallback(out)
    model, tokenizer = loaded[:2]
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
