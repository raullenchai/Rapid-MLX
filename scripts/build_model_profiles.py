#!/usr/bin/env python3.12
"""Aggregate per-alias profile data into one canonical artifact.

Reads three sources of truth for each alias in ``vllm_mlx/aliases.json``:

  1. ``aliases.json`` itself     — config: parsers, hybrid flag, spec_decode flag,
                                    ``suffix_decoding_tier`` (when set in SSOT)
  2. ``evals/results/suffix_<alias>_v2.json``
                                  — fresh post-PR-#308 bench: vanilla_tps,
                                    suffix_tps, per-workload speedups, tier
  3. ``evals/results/<model-stem>.json``
                                  — community/dev eval scorecard: speed
                                    (TTFT, decode TPS, RAM), tool-calling
                                    score, coding/reasoning/general scores

Produces:

  - ``evals/results/model_profiles.json`` — machine-readable master artifact
  - ``docs/model_profiles.md`` — human-readable per-alias table

The script is idempotent — re-run after the SuffixDecoding sweep finishes
to pick up new bench files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIASES_PATH = REPO_ROOT / "vllm_mlx" / "aliases.json"
EVALS_DIR = REPO_ROOT / "evals" / "results"
OUT_JSON = EVALS_DIR / "model_profiles.json"
OUT_MD = REPO_ROOT / "docs" / "model_profiles.md"


# Manual mapping from alias → eval-scorecard filename stem (without .json).
# Most aliases don't have eval data; this only covers the ~20 we've benched.
# Filename derives from the model directory name on disk, not the alias —
# a rename-safe mapping table is simpler than fuzzy matching across renamed quants.
EVAL_FILE_MAP: dict[str, str] = {
    "qwen3.5-4b": "qwen3.5-4b-4bit",
    "qwen3.5-9b": "qwen3.5-9b-4bit",
    "qwen3.5-27b": "qwen3.5-27b-4bit",
    "qwen3.5-35b": "qwen3.5-35b-a3b-4bit",
    "qwen3.5-122b": "qwen3.5-122b-a10b-mxfp4",
    "qwen3.5-122b-8bit": "qwen3.5-122b-a10b-8bit",
    "qwen3.6-27b": "qwen3.6-27b-4bit",
    "qwen3.6-27b-8bit": "qwen3.6-27b-8bit",
    "qwen3-coder-30b": "qwen3-coder-next-4bit",
    "hermes3-8b": "hermes-3-llama-3.1-8b-4bit",
    "gpt-oss-20b": "gpt-oss-20b-mxfp4-q8",
    "minimax-m2.5": "minimax-m2.5-4bit",
    "mistral-24b": "mistral-small-3.2-4bit",
    "devstral-24b": "devstral-small-2-4bit",
    "glm4.5-air": "glm-4.5-air-4bit",
    "glm4.7-9b": "glm-4.7-flash-8bit",
}


@dataclass
class SuffixData:
    tier: str | None = None
    vanilla_tps: dict[str, float] = field(default_factory=dict)
    suffix_tps: dict[str, float] = field(default_factory=dict)
    speedups: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    source_file: str | None = None


@dataclass
class SpeedData:
    ttft_cold_s: float | None = None
    ttft_warm_s: float | None = None
    decode_short_tps: float | None = None
    decode_long_tps: float | None = None
    ram_active_gb: float | None = None
    ram_peak_gb: float | None = None
    bench_date: str | None = None
    source_file: str | None = None


@dataclass
class EvalScores:
    tool_calling: dict[str, Any] | None = None
    coding: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    general: dict[str, Any] | None = None
    eval_date: str | None = None


@dataclass
class Recipe:
    enable_suffix_decoding: bool
    enable_spec_decoding: bool
    recommended_flags: list[str]
    warnings: list[str]


@dataclass
class Profile:
    alias: str
    hf_path: str
    tool_call_parser: str | None
    reasoning_parser: str | None
    is_hybrid: bool
    supports_spec_decode: bool
    suffix: SuffixData
    speed: SpeedData
    evals: EvalScores
    recipe: Recipe


def _load_suffix(alias: str) -> SuffixData:
    """Pick the freshest bench file. v2 (post-fix) wins; v1 is legacy."""
    path_v2 = EVALS_DIR / f"suffix_{alias}_v2.json"
    path_v1 = EVALS_DIR / f"suffix_{alias}.json"
    path = path_v2 if path_v2.exists() else (path_v1 if path_v1.exists() else None)
    if path is None:
        return SuffixData()

    raw = json.loads(path.read_text())
    return SuffixData(
        tier=raw.get("tier"),
        vanilla_tps=raw.get("vanilla_tps") or {},
        suffix_tps=raw.get("suffix_tps") or {},
        speedups=raw.get("speedup") or {},
        skipped=raw.get("skipped") or {},
        source_file=str(path.relative_to(REPO_ROOT)),
    )


def _load_speed_and_evals(alias: str) -> tuple[SpeedData, EvalScores]:
    stem = EVAL_FILE_MAP.get(alias)
    if not stem:
        return SpeedData(), EvalScores()
    path = EVALS_DIR / f"{stem}.json"
    if not path.exists():
        return SpeedData(), EvalScores()

    raw = json.loads(path.read_text())
    speed_raw = raw.get("speed") or {}
    speed = SpeedData(
        ttft_cold_s=speed_raw.get("ttft_cold_s"),
        ttft_warm_s=speed_raw.get("ttft_warm_s"),
        decode_short_tps=speed_raw.get("decode_short_tps"),
        decode_long_tps=speed_raw.get("decode_long_tps"),
        ram_active_gb=speed_raw.get("ram_active_gb"),
        ram_peak_gb=speed_raw.get("ram_peak_gb"),
        bench_date=raw.get("date"),
        source_file=str(path.relative_to(REPO_ROOT)),
    )

    def _scorecard(key: str) -> dict[str, Any] | None:
        block = raw.get(key)
        if not isinstance(block, dict):
            return None
        # Drop ``details`` — too verbose for the aggregate, full data
        # is one file lookup away via ``source_file``.
        return {k: v for k, v in block.items() if k != "details"}

    evals = EvalScores(
        tool_calling=_scorecard("tool_calling"),
        coding=_scorecard("coding"),
        reasoning=_scorecard("reasoning"),
        general=_scorecard("general"),
        eval_date=raw.get("date"),
    )
    return speed, evals


def _derive_recipe(
    alias: str,
    profile_cfg: dict[str, Any],
    suffix: SuffixData,
) -> Recipe:
    """Pick the optimal flag set given everything we know about this alias.

    Rules:
      - suffix-decoding ON when tier in {neutral, structured, agent}
      - suffix-decoding OFF when tier=avoid (verify-forward overhead > gain)
      - spec-decoding OFF on hybrid (Mamba/SSM mix breaks draft replay)
      - spec-decoding ON only when ``supports_spec_decode`` AND not hybrid
      - flags include ``--enable-auto-tool-choice`` whenever a parser is set
    """
    tool_parser = profile_cfg.get("tool_call_parser")
    reasoning = profile_cfg.get("reasoning_parser")
    is_hybrid = bool(profile_cfg.get("is_hybrid", False))
    supports_spec = bool(profile_cfg.get("supports_spec_decode", False))

    enable_suffix = suffix.tier in {"neutral", "structured", "agent"}
    enable_spec = supports_spec and not is_hybrid

    flags: list[str] = []
    if tool_parser:
        flags.append("--enable-auto-tool-choice")
        flags.append(f"--tool-call-parser {tool_parser}")
    if reasoning:
        flags.append(f"--reasoning-parser {reasoning}")
    if enable_suffix:
        flags.append("--enable-suffix-decoding")
    # We don't surface --enable-spec-decode in the recipe yet because the
    # serve-time auto-apply for spec lives in a separate (unmerged) PR;
    # callers can set it themselves based on ``supports_spec_decode``.

    warnings: list[str] = []
    if is_hybrid:
        warnings.append(
            "hybrid model: spec-decode and suffix-decode unavailable "
            "(Mamba/SSM mix breaks drafter)"
        )
    if suffix.tier is None:
        warnings.append("suffix-decoding tier unknown — bench has not run yet")
    elif suffix.tier == "unknown":
        warnings.append(
            "suffix-decoding bench produced no usable data (most workloads "
            "rejected by reliability gates) — re-bench needed"
        )
    if suffix.tier == "avoid":
        warnings.append(
            "suffix-decoding hurts on this model (verify-forward overhead "
            "exceeds drafter gain) — leave disabled"
        )

    return Recipe(
        enable_suffix_decoding=enable_suffix,
        enable_spec_decoding=enable_spec,
        recommended_flags=flags,
        warnings=warnings,
    )


def build_profiles() -> list[Profile]:
    aliases_raw: dict[str, dict[str, Any]] = json.loads(ALIASES_PATH.read_text())
    profiles: list[Profile] = []
    for alias in sorted(aliases_raw):
        cfg = aliases_raw[alias]
        suffix = _load_suffix(alias)
        # Prefer SSOT tier from aliases.json if present (it's authoritative
        # once the bench data has been promoted into the registry).
        if cfg.get("suffix_decoding_tier") and not suffix.tier:
            suffix.tier = cfg["suffix_decoding_tier"]
        speed, evals = _load_speed_and_evals(alias)
        recipe = _derive_recipe(alias, cfg, suffix)
        profiles.append(
            Profile(
                alias=alias,
                hf_path=cfg["hf_path"],
                tool_call_parser=cfg.get("tool_call_parser"),
                reasoning_parser=cfg.get("reasoning_parser"),
                is_hybrid=bool(cfg.get("is_hybrid", False)),
                supports_spec_decode=bool(cfg.get("supports_spec_decode", False)),
                suffix=suffix,
                speed=speed,
                evals=evals,
                recipe=recipe,
            )
        )
    return profiles


def write_json(profiles: Iterable[Profile]) -> None:
    payload = {
        "_schema_version": 1,
        "_generator": "scripts/build_model_profiles.py",
        "profiles": [asdict(p) for p in profiles],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _fmt_tier(tier: str | None) -> str:
    if tier is None:
        return "—"
    return {
        "agent": "🟢 agent",
        "structured": "🟢 structured",
        "neutral": "⚪ neutral",
        "avoid": "🔴 avoid",
        "unknown": "❓ unknown",
        "n/a": "n/a (hybrid)",
    }.get(tier, tier)


def _fmt_tps(tps_dict: dict[str, float]) -> str:
    if not tps_dict:
        return "—"
    keys = ("chat", "json_array", "tool_loop", "code_edit")
    parts = []
    for k in keys:
        if k in tps_dict and tps_dict[k] is not None:
            parts.append(f"{tps_dict[k]:.0f}")
        else:
            parts.append("—")
    return "/".join(parts)


def _fmt_speedups(speedups: dict[str, float]) -> str:
    if not speedups:
        return "—"
    keys = ("chat", "json_array", "tool_loop", "code_edit")
    parts = []
    for k in keys:
        v = speedups.get(k)
        if v is None:
            parts.append("—")
        else:
            arrow = "↑" if v > 1.05 else ("↓" if v < 0.95 else "·")
            parts.append(f"{v:.2f}{arrow}")
    return " / ".join(parts)


def write_md(profiles: list[Profile]) -> None:
    lines: list[str] = []
    lines.append("# Model Profiles")
    lines.append("")
    lines.append(
        "Auto-generated by `scripts/build_model_profiles.py`. "
        "Do not edit by hand — re-run the script to refresh."
    )
    lines.append("")
    lines.append("Sources:")
    lines.append("- `vllm_mlx/aliases.json` — config (parsers, hybrid, spec-decode)")
    lines.append(
        "- `evals/results/suffix_<alias>_v2.json` — SuffixDecoding bench "
        "(vanilla TPS, per-workload speedups, tier classification)"
    )
    lines.append(
        "- `evals/results/<model-stem>.json` — community eval scorecards "
        "(TTFT, long-context TPS, tool-calling/coding/reasoning scores)"
    )
    lines.append("")

    # ---- summary header table ----
    lines.append("## Summary")
    lines.append("")
    lines.append("Workload columns are `chat / json_array / tool_loop / code_edit`.")
    lines.append("")
    lines.append(
        "| Alias | Hybrid | SpecDec | Suffix tier | Vanilla TPS | Suffix speedup |"
    )
    lines.append(
        "|-------|--------|---------|-------------|-------------|----------------|"
    )
    for p in profiles:
        hybrid = "✓" if p.is_hybrid else ""
        spec = "✓" if p.supports_spec_decode else ""
        # On hybrid models the framework auto-renders tier as n/a; show that.
        display_tier = "n/a" if p.is_hybrid else p.suffix.tier
        lines.append(
            f"| `{p.alias}` | {hybrid} | {spec} | {_fmt_tier(display_tier)} | "
            f"{_fmt_tps(p.suffix.vanilla_tps)} | "
            f"{_fmt_speedups(p.suffix.speedups)} |"
        )
    lines.append("")

    # ---- per-alias detail ----
    lines.append("## Per-alias detail")
    lines.append("")
    for p in profiles:
        lines.append(f"### `{p.alias}`")
        lines.append("")
        lines.append(f"- **HF path:** `{p.hf_path}`")
        lines.append(
            f"- **Parsers:** tool=`{p.tool_call_parser or '—'}`, "
            f"reasoning=`{p.reasoning_parser or '—'}`"
        )
        lines.append(
            f"- **Architecture:** hybrid={p.is_hybrid}, "
            f"supports_spec_decode={p.supports_spec_decode}"
        )

        # SuffixDecoding
        if p.suffix.tier is not None:
            lines.append(
                f"- **SuffixDecoding:** tier=`{p.suffix.tier}`"
                + (
                    f" (source: `{p.suffix.source_file}`)"
                    if p.suffix.source_file
                    else ""
                )
            )
            if p.suffix.vanilla_tps:
                lines.append(
                    f"  - vanilla TPS (chat/json/tool/code): "
                    f"`{_fmt_tps(p.suffix.vanilla_tps)}`"
                )
            if p.suffix.speedups:
                lines.append(
                    f"  - speedup: `{_fmt_speedups(p.suffix.speedups)}`"
                )
            if p.suffix.skipped:
                skipped_str = ", ".join(
                    f"{k}: {v}" for k, v in sorted(p.suffix.skipped.items())
                )
                lines.append(f"  - skipped workloads: {skipped_str}")
        else:
            lines.append("- **SuffixDecoding:** no bench data")

        # Speed (from eval scorecard)
        s = p.speed
        if any(
            v is not None
            for v in (
                s.ttft_warm_s,
                s.decode_short_tps,
                s.decode_long_tps,
                s.ram_active_gb,
            )
        ):
            ttft = f"{s.ttft_warm_s:.3f}s" if s.ttft_warm_s is not None else "—"
            short = (
                f"{s.decode_short_tps:.0f}" if s.decode_short_tps is not None else "—"
            )
            long_ = (
                f"{s.decode_long_tps:.0f}" if s.decode_long_tps is not None else "—"
            )
            ram = f"{s.ram_active_gb:.1f}GB" if s.ram_active_gb is not None else "—"
            lines.append(
                f"- **Speed (eval bench):** TTFT warm `{ttft}`, "
                f"decode short/long `{short}/{long_}` tok/s, RAM active `{ram}`"
                + (f" — {s.bench_date}" if s.bench_date else "")
            )

        # Eval scores
        ev = p.evals
        score_bits: list[str] = []
        for label, block in (
            ("tool", ev.tool_calling),
            ("coding", ev.coding),
            ("reasoning", ev.reasoning),
            ("general", ev.general),
        ):
            if block:
                score_bits.append(
                    f"{label} {block['passed']}/{block['total']} "
                    f"({block['score']:.2f})"
                )
        if score_bits:
            lines.append(f"- **Eval scores:** {' · '.join(score_bits)}")

        # Recipe
        r = p.recipe
        lines.append(
            f"- **Recipe:** suffix-decode={'on' if r.enable_suffix_decoding else 'off'}, "
            f"spec-decode={'on' if r.enable_spec_decoding else 'off'}"
        )
        if r.recommended_flags:
            lines.append(
                f"  - flags: `{' '.join(r.recommended_flags)}`"
            )
        for w in r.warnings:
            lines.append(f"  - ⚠ {w}")

        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


_ALIAS_RE = re.compile(r"^[a-z][a-z0-9.\-]*$")


def _validate_aliases(aliases_raw: dict[str, Any]) -> None:
    for alias in aliases_raw:
        if not _ALIAS_RE.match(alias):
            raise ValueError(
                f"alias {alias!r} does not match expected slug pattern; "
                "fix it in vllm_mlx/aliases.json before regenerating"
            )


def main() -> int:
    aliases_raw = json.loads(ALIASES_PATH.read_text())
    _validate_aliases(aliases_raw)
    profiles = build_profiles()
    write_json(profiles)
    write_md(profiles)
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)} ({len(profiles)} aliases)")
    print(f"wrote {OUT_MD.relative_to(REPO_ROOT)}")

    # Quick stats summary on stdout for sanity-check after runs.
    total = len(profiles)
    has_suffix = sum(1 for p in profiles if p.suffix.tier is not None)
    has_evals = sum(1 for p in profiles if p.evals.tool_calling is not None)
    has_speed = sum(1 for p in profiles if p.speed.decode_long_tps is not None)
    print(
        f"  coverage: suffix_tier={has_suffix}/{total} · "
        f"eval_scorecard={has_evals}/{total} · "
        f"speed_bench={has_speed}/{total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
