#!/usr/bin/env python3
"""
Tool-parser matrix coverage audit.

Catches the bug class surfaced by #425 (jpcarranza94) and the v0.6.62 #429
meta-fix: a ``--tool-call-parser`` value is registered in
``ToolParserManager`` but never exercised end-to-end via the actual
CLI-flag → server-boot → wire-format flow. Unit tests cover the parser
class in isolation; the parity test (``tests/test_tool_call_streaming_parity.py``)
covers the stream/non-stream symmetry; this audit covers the THIRD surface
— integration matrix coverage.

The bug class: a parser name that's auto-detected for some models works
in practice (because every user happens to use the auto-routed default
that pairs with a tested model). But a parser name that's only reachable
via an EXPLICIT ``--tool-call-parser X`` CLI choice and never exercised
in the matrix can silently mis-handle the wire format the user's actual
model emits — exactly what happened to ``qwen3_xml`` in #425.

Source of truth: ``scripts/pr_validate/golden_models.yaml::overrides``.
Each override carries an ``args: [--enable-auto-tool-choice, --tool-call-parser, X, ...]``
list; the parser names in those lists are the matrix-tested set. Every
registered ``ToolParserManager`` name must either appear there or be in
``MATRIX_EXEMPT`` with a documented reason.

Exit 0 = clean. Exit 1 = uncovered parsers + actionable diff printed.

Run via ``python3 scripts/audit_tool_parser_coverage.py`` or as part of
``tests/test_tool_parser_coverage.py`` (the test layer that gates CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_MODELS = REPO_ROOT / "scripts" / "pr_validate" / "golden_models.yaml"


# Explicit exemptions. Each entry MUST carry a reason — review forces the
# author to think about WHY no matrix coverage is needed, rather than
# silently pad this dict.
#
# Categories:
#   - aliases of a covered parser (same class via register_module list)
#   - meta-parsers that don't have a wire format (auto / generic routers)
#   - TODOs: parser is registered but no model is in the matrix yet;
#     ticket should be filed and referenced here
MATRIX_EXEMPT: dict[str, str] = {
    # Meta-parsers / routers — no wire format of their own.
    "auto": "auto-routing meta-parser, not a wire-format parser",
    "generic": "generic fallback router, not a wire-format parser",
    # Aliases of parsers that ARE in the matrix via canonical name.
    "qwen3_coder": "alias of hermes (qwen3_coder model series uses hermes parser)",
    "nous": "alias of hermes (NousResearch/Hermes series)",
    "qwen": "JSON-body Qwen variant; matrix covers Qwen3.6 via hermes which handles both bodies",
    "qwen3": "alias of qwen",
    "gemma_4": "alias of gemma4",
    "gpt-oss": "alias of harmony",
    "seed": "alias of seed_oss",
    "gpt_oss": "alias of seed_oss",
    "kimi_k2": "alias of kimi",
    "moonshot": "alias of kimi",
    "llama3": "alias of llama",
    "llama4": "alias of llama",
    "deepseek_r1": "alias of deepseek",
    "deepseek_r1_0528": "alias of deepseek_v3 (R12-5)",
    "minimax_m2": "alias of minimax",
    "nemotron3": "alias of nemotron",
    "granite3": "alias of granite",
    "glm4": "alias of glm47",
    "meetkai": "alias of functionary",
    # TODOs — parser registered, no canonical model in matrix yet. Each
    # entry is a follow-up ticket. Adding a real golden_models.yaml entry
    # for any of these REMOVES the corresponding TODO line (failure mode:
    # leaving the TODO after adding matrix coverage is harmless — the
    # audit still passes because matrix > exempt).
    "gemma4": "TODO: re-add gemma4 to golden_models when mlx-community ships a tighter instruction-tuned variant (see golden_models.yaml lines 125-141)",
    "qwen3_xml": "TODO: parser registered to QwenToolParser (JSON body) but name implies XML body; covered by hermes BARE_FUNCTION_PATTERN in practice — fix or deprecate registration in follow-up to #426",
    "qwen3_coder_xml": "TODO: add Qwen3-Coder XML body model to golden_models when capacity allows",
    "glm47": "TODO: add GLM-4.7/5 model to golden_models",
    "granite": "TODO: add Granite4 H-Tiny/Small model to golden_models",
    "llama": "TODO: add Llama 3.3/4 instruct model to golden_models",
    "minimax": "TODO: add MiniMax-M2/M2.5 model to golden_models",
    "mistral": "TODO: add Mistral-Small/Large model to golden_models",
    "nemotron": "TODO: add Nemotron model to golden_models",
    "kimi": "TODO: add Kimi-K2/K2.6 model to golden_models",
    "deepseek": "TODO: add DeepSeek-V2 / R1-distill model to golden_models",
    "deepseek_v3": "TODO: add DeepSeek-V3 / R1-0528-Qwen3-8B model to golden_models (R12-5)",
    "deepseek_v31": "TODO: add DeepSeek-V3.1 thinking-channel model to golden_models",
    "functionary": "TODO: add Functionary-medium model to golden_models",
    "xlam": "TODO: add xLAM model to golden_models",
    "seed_oss": "TODO: add Seed-OSS model to golden_models",
    # UI-TARS (ByteDance) — GUI-agent VLM. Adding a real golden_models
    # entry requires running a Qwen2-VL / Qwen2.5-VL backbone in the
    # pr_validate matrix (mlx-vlm path, not the plain mlx-lm path the
    # other matrix entries use). Track as TODO until vision-VLM matrix
    # support lands.
    "ui_tars": "TODO: add UI-TARS-1.5-7B-4bit to golden_models once the VLM matrix supports mlx-vlm-backed Computer-Use models",
    "ui-tars": "alias of ui_tars (kebab-case spelling)",
    "uitars": "alias of ui_tars (no-separator spelling)",
}


def _load_yaml(path: Path) -> dict:
    """Parse golden_models.yaml. PyYAML is required (in test deps)."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "PyYAML required to parse golden_models.yaml — "
            "`pip install pyyaml` or use the test extras"
        ) from e
    return yaml.safe_load(path.read_text())


def parsers_in_matrix(golden_models_path: Path) -> set[str]:
    """Extract every ``--tool-call-parser X`` value from golden_models.yaml
    overrides. The override args list is the source of truth — that's the
    exact CLI flag the pr_validate matrix passes to the server.
    """
    cfg = _load_yaml(golden_models_path)
    parsers: set[str] = set()
    overrides = (cfg.get("overrides") or {}).values()
    for override in overrides:
        args = (override or {}).get("args", []) or []
        for i, a in enumerate(args):
            if a == "--tool-call-parser" and i + 1 < len(args):
                parsers.add(args[i + 1])
    return parsers


def registered_parsers() -> set[str]:
    """Read ``ToolParserManager.tool_parsers`` registry. Importing this
    pulls in the full tool-parser module tree, which on this repo is
    Linux-safe (no mlx imports). The CLI fidelity audit deliberately uses
    AST to avoid that; we can rely on the import here because the parser
    tree is dependency-free.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from vllm_mlx.tool_parsers import (
        ToolParserManager,  # type: ignore[import-not-found]
    )

    return set(ToolParserManager.tool_parsers)


def audit() -> tuple[set[str], set[str], set[str]]:
    """Return (registered, matrix_covered, gaps).

    ``gaps`` is the actionable set — registered parsers with neither
    matrix coverage nor an explicit ``MATRIX_EXEMPT`` entry. Empty set
    means the audit passes.
    """
    registered = registered_parsers()
    matrix = parsers_in_matrix(GOLDEN_MODELS)
    exempt = set(MATRIX_EXEMPT)
    gaps = registered - matrix - exempt
    return registered, matrix, gaps


def main() -> int:
    registered, matrix, gaps = audit()

    if not gaps:
        print(
            f"OK: {len(registered)} registered tool parser(s) covered "
            f"({len(matrix)} via matrix, {len(MATRIX_EXEMPT)} exempt)."
        )
        return 0

    print(f"FAIL: {len(gaps)} registered tool parser(s) without coverage:")
    for parser_name in sorted(gaps):
        print(f"  - {parser_name}")
    print()
    print("Action:")
    print(
        "  - Add a ``--tool-call-parser`` override to "
        "``scripts/pr_validate/golden_models.yaml`` that exercises this "
        "parser end-to-end."
    )
    print(
        "  - OR add the parser to ``MATRIX_EXEMPT`` in this script with "
        "a documented reason (alias / TODO with ticket / etc.)."
    )
    print()
    print(
        "Background: every ``--tool-call-parser X`` value users can pass "
        "must have integration matrix coverage OR an explicit exemption. "
        "See #425 (jpcarranza94) for the bug class this gates — "
        "``qwen3_xml`` was registered but never matrix-tested; the wire-"
        "format mismatch only surfaced in production."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
