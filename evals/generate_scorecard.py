#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate SCORECARD.md from eval result JSON files.

Reads all JSON files in evals/results/ and produces a human-readable
comparison table.

Usage:
    python evals/generate_scorecard.py
    python evals/generate_scorecard.py --output evals/SCORECARD.md
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).parent
RESULTS_DIR = EVALS_DIR / "results"


def load_results() -> list[dict]:
    """Load all result JSON files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.name
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f.name}: {e}")
    return results


def fmt_pct(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.0%}"
    return str(val)


def fmt_num(val, suffix="") -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        if val == int(val):
            return f"{int(val)}{suffix}"
        return f"{val:.1f}{suffix}"
    return str(val)


def fmt_ms(val_s) -> str:
    """Format seconds as milliseconds string (e.g. 0.687 → '687ms')."""
    if val_s is None:
        return "—"
    if isinstance(val_s, (int, float)):
        ms = val_s * 1000
        if ms >= 1000:
            return f"{ms / 1000:.1f}s"
        return f"{ms:.0f}ms"
    return str(val_s)


def fmt_ram(val_gb) -> str:
    """Format RAM in GB (e.g. 23.4 → '23.4 GB')."""
    if val_gb is None:
        return "—"
    if isinstance(val_gb, (int, float)):
        return f"{val_gb:.1f} GB"
    return str(val_gb)


def calc_avg(*scores) -> str:
    """Calculate average of non-None scores, return as percentage string."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return "—"
    avg = sum(valid) / len(valid)
    return f"{avg:.0%}"


def generate_scorecard(results: list[dict]) -> str:
    lines = []
    lines.append("# rapid-mlx Model Scorecard")
    lines.append("")
    lines.append(
        f"*Auto-generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*"
    )
    lines.append("")

    # Extract hardware from first result for header note
    hw_labels = {r.get("hardware", "") for r in results if r.get("hardware")}
    hw_note = ", ".join(sorted(hw_labels)) if hw_labels else "Apple Silicon"
    lines.append(f"> **Tested on**: {hw_note}")
    lines.append(">")
    lines.append(
        "> **Methodology**: All suites use `enable_thinking: false`. Cache cleared between suites. See [README](README.md) for details."
    )
    lines.append("")
    lines.append("## Comparison Table")
    lines.append("")

    # Header — new layout with RAM, TTFT, Avg; without Hardware, Parser
    cols = [
        "Model",
        "Quant",
        "RAM",
        "TTFT",
        "Decode (s)",
        "Decode (l)",
        "Tools",
        "Coding",
        "Reasoning",
        "General",
        "Avg",
        "Date",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for r in results:
        speed = r.get("speed", {})
        tool = r.get("tool_calling", {})
        coding = r.get("coding", {})
        reasoning = r.get("reasoning", {})
        general = r.get("general", {})

        # Quality scores for Avg calculation
        tool_score = tool.get("score")
        coding_score = coding.get("score")
        reasoning_score = reasoning.get("score")
        general_score = general.get("score")

        row = [
            r.get("model", "?"),
            r.get("quantization", "—") or "—",
            fmt_ram(speed.get("ram_active_gb")),
            fmt_ms(speed.get("ttft_cold_s")),
            fmt_num(speed.get("decode_short_tps"), " t/s"),
            fmt_num(speed.get("decode_long_tps"), " t/s"),
            fmt_pct(tool_score),
            fmt_pct(coding_score),
            fmt_pct(reasoning_score),
            fmt_pct(general_score),
            calc_avg(tool_score, coding_score, reasoning_score, general_score),
            r.get("date", "—"),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Detail sections per model
    lines.append("## Details")
    lines.append("")

    for r in results:
        lines.append(f"### {r.get('model', '?')}")
        lines.append("")
        lines.append(f"- **Hardware**: {r.get('hardware', '?')}")
        lines.append(f"- **Parser**: {r.get('parser', 'auto')}")
        if r.get("server_flags"):
            lines.append(f"- **Server flags**: `{r['server_flags']}`")
        lines.append(f"- **Date**: {r.get('date', '?')}")

        speed = r.get("speed", {})
        if speed:
            lines.append(
                f"- **TTFT**: cold={fmt_ms(speed.get('ttft_cold_s'))}, warm={fmt_ms(speed.get('ttft_warm_s'))}"
            )
            lines.append(
                f"- **Decode**: short={fmt_num(speed.get('decode_short_tps'), ' t/s')}, long={fmt_num(speed.get('decode_long_tps'), ' t/s')}"
            )
            if speed.get("ram_active_gb") is not None:
                ram_line = f"- **RAM**: active={fmt_ram(speed.get('ram_active_gb'))}"
                if speed.get("ram_peak_gb") is not None:
                    ram_line += f", peak={fmt_ram(speed.get('ram_peak_gb'))}"
                lines.append(ram_line)

        for suite_name in ["tool_calling", "coding", "reasoning", "general"]:
            suite = r.get(suite_name, {})
            if suite:
                lines.append(
                    f"- **{suite_name.replace('_', ' ').title()}**: {fmt_pct(suite.get('score'))} ({suite.get('passed', '?')}/{suite.get('total', '?')})"
                )

        if r.get("total_eval_time_s"):
            lines.append(f"- **Eval time**: {r['total_eval_time_s']}s")

        lines.append("")

    # How to contribute
    lines.append("---")
    lines.append("")
    lines.append("## How to Add Your Results")
    lines.append("")
    lines.append(
        "1. Start rapid-mlx with your model: `rapid-mlx serve <model> --port 8000`"
    )
    lines.append(
        '2. Run the eval: `python evals/run_eval.py --model "<model-name>" --quantization <quant>`'
    )
    lines.append("3. Your results are saved to `evals/results/<model>.json`")
    lines.append("4. Regenerate this table: `python evals/generate_scorecard.py`")
    lines.append("5. Submit a PR with your JSON file!")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SCORECARD.md from eval results"
    )
    parser.add_argument(
        "--output",
        default=str(EVALS_DIR / "SCORECARD.md"),
        help="Output file (default: evals/SCORECARD.md)",
    )
    args = parser.parse_args()

    results = load_results()
    if not results:
        print("No result files found in evals/results/")
        print("Run an eval first: python evals/run_eval.py --model <name>")
        return

    print(f"Found {len(results)} result file(s)")

    scorecard = generate_scorecard(results)
    Path(args.output).write_text(scorecard + "\n")
    print(f"Scorecard written to: {args.output}")


if __name__ == "__main__":
    main()
