# SPDX-License-Identifier: Apache-2.0
"""Tests for SuffixDecoding tier classification (#269)."""

from __future__ import annotations

from vllm_mlx.model_auto_config import (
    ModelConfig,
    _suffix_tier_cell,
    classify_suffix_decoding_tier,
    format_profile_table,
    suffix_decoding_hint,
)


class TestClassifyTier:
    """Pure-function tests for the tier boundary logic."""

    def test_empty_input_is_unknown(self):
        assert classify_suffix_decoding_tier({}) == "unknown"

    def test_qwen3_06b_is_agent(self):
        # Real numbers from evals/results/SUFFIX_POC_REPORT.md
        speedup = {
            "chat": 1.05,
            "json_array": 1.41,
            "tool_loop": 4.60,
            "code_edit": 1.58,
        }
        assert classify_suffix_decoding_tier(speedup) == "agent"

    def test_qwen3_14b_is_agent(self):
        # tool_loop 1.99x (>=1.8) AND others all >= 0.95
        speedup = {
            "chat": 0.98,
            "json_array": 1.20,
            "tool_loop": 1.99,
            "code_edit": 1.30,
        }
        assert classify_suffix_decoding_tier(speedup) == "agent"

    def test_llama_no_tool_rlhf_is_structured_when_some_workload_wins(self):
        speedup = {
            "chat": 0.97,
            "json_array": 1.55,
            "tool_loop": 1.27,
            "code_edit": 1.40,
        }
        # tool_loop < 1.8 → not AGENT.  max=1.55 >= 1.5 AND min=0.97 >= 0.90 → STRUCTURED.
        assert classify_suffix_decoding_tier(speedup) == "structured"

    def test_no_workload_wins_but_no_regression_is_neutral(self):
        speedup = {"chat": 0.99, "tool_loop": 1.02, "json_array": 1.00}
        assert classify_suffix_decoding_tier(speedup) == "neutral"

    def test_any_regression_is_avoid(self):
        # tool_loop wins big but chat regresses → AVOID. The regression
        # dominates because we don't know the user's traffic mix.
        speedup = {"chat": 0.78, "tool_loop": 3.0, "json_array": 1.5}
        assert classify_suffix_decoding_tier(speedup) == "avoid"

    def test_avoid_threshold_is_strict_lt_0_85(self):
        # <0.85 → AVOID (regression dominates).
        assert classify_suffix_decoding_tier({"chat": 0.84}) == "avoid"
        # 0.85 escapes the AVOID-from-low-lo trip but doesn't clear
        # STRUCTURED's 0.90 floor either, so it falls through into the
        # mixed-signal AVOID at the end. That's the intended behavior:
        # being "just above the regression threshold" is not enough to
        # recommend the flag.
        assert classify_suffix_decoding_tier({"chat": 0.85, "x": 1.6}) == "avoid"
        # Clearing the STRUCTURED floor (0.90) flips to STRUCTURED.
        assert classify_suffix_decoding_tier({"chat": 0.90, "x": 1.6}) == "structured"

    def test_agent_requires_others_dont_regress(self):
        # tool_loop=2x but code_edit dropped to 0.92 → fails AGENT's
        # ``min(others) >= 0.95`` gate. Falls through positive buckets
        # and lands on AVOID rather than silently shipping a regression.
        speedup = {"tool_loop": 2.0, "chat": 1.0, "code_edit": 0.92, "json_array": 1.1}
        # 0.92 >= 0.85 so not AVOID-from-regression… but 0.92 < 0.95 so
        # not AGENT… max=2.0 >= 1.5 AND min=0.92 >= 0.90 → STRUCTURED.
        assert classify_suffix_decoding_tier(speedup) == "structured"

    def test_agent_requires_tool_loop_specifically(self):
        # max workload is "json_array" 2x; tool_loop is 1.0. Even though
        # max > 1.8 globally, AGENT specifically needs *tool_loop* to win
        # (otherwise the agent label is misleading).
        speedup = {"tool_loop": 1.0, "chat": 1.0, "json_array": 2.0, "code_edit": 1.0}
        assert classify_suffix_decoding_tier(speedup) == "structured"

    def test_single_workload_dict(self):
        # A degenerate single-workload bench shouldn't crash; if the only
        # workload is tool_loop ≥ 1.8x, return AGENT regardless of
        # ``min(others)`` (which is empty).
        assert classify_suffix_decoding_tier({"tool_loop": 2.5}) == "agent"


class TestSuffixDecodingHint:
    """The startup hint surfaces only AGENT/STRUCTURED/AVOID — not
    UNKNOWN or NEUTRAL — to keep ``rapid-mlx serve`` startup quiet."""

    def test_no_cfg_no_hint(self):
        assert suffix_decoding_hint(None) is None

    def test_unknown_is_silent(self):
        cfg = ModelConfig(suffix_decoding_tier="unknown")
        assert suffix_decoding_hint(cfg) is None

    def test_neutral_is_silent(self):
        cfg = ModelConfig(suffix_decoding_tier="neutral")
        assert suffix_decoding_hint(cfg) is None

    def test_agent_recommends(self):
        cfg = ModelConfig(
            suffix_decoding_tier="agent",
            suffix_bench_speedup={"chat": 1.05, "tool_loop": 4.6},
        )
        hint = suffix_decoding_hint(cfg)
        assert hint is not None
        assert "recommended" in hint.lower()
        assert "4.6" in hint  # surfaces the actual tool win

    def test_structured_recommends(self):
        cfg = ModelConfig(
            suffix_decoding_tier="structured",
            suffix_bench_speedup={"chat": 0.97, "json_array": 1.55},
        )
        hint = suffix_decoding_hint(cfg)
        assert hint is not None
        assert "structured" in hint.lower() or "may help" in hint.lower()

    def test_avoid_warns(self):
        cfg = ModelConfig(
            suffix_decoding_tier="avoid",
            suffix_bench_speedup={"chat": 0.78},
        )
        hint = suffix_decoding_hint(cfg)
        assert hint is not None
        assert "regress" in hint.lower() or "avoid" in hint.lower()

    def test_hybrid_no_hint_even_if_tier_is_agent(self):
        """If the safety gate ``supports_spec_decode=False`` is set, we
        must not nudge the user toward a flag that's silently ignored."""
        cfg = ModelConfig(
            supports_spec_decode=False,
            suffix_decoding_tier="agent",
            suffix_bench_speedup={"tool_loop": 2.5},
        )
        assert suffix_decoding_hint(cfg) is None


class TestProfileTableCell:
    """The Level 2 ``rapid-mlx info`` table must surface the tier without
    crashing on edge cases (no bench data, hybrid models, missing fields)."""

    def test_unknown_mentions_bench_script(self):
        cfg = ModelConfig()  # default unknown, empty dict
        table = format_profile_table("test/Model", cfg)
        assert "Suffix tier" in table
        assert "unknown" in table.lower()
        # The boxed table truncates the script name to keep alignment,
        # but a discoverable prefix must remain so the user knows where
        # to look. The un-truncated helper (called without ``max_width``)
        # must still emit the full path for log scrapers / non-boxed
        # callers.
        assert "bench_suffix_decod" in table
        assert "bench_suffix_decoding_integrated" in _suffix_tier_cell(cfg)

    def test_hybrid_says_n_a(self):
        cfg = ModelConfig(supports_spec_decode=False, is_hybrid=True)
        table = format_profile_table("test/HybridModel", cfg)
        assert "n/a" in table.lower()
        assert "hybrid" in table.lower()

    def test_agent_shows_tool_loop_speedup(self):
        cfg = ModelConfig(
            suffix_decoding_tier="agent",
            suffix_bench_speedup={"chat": 1.05, "tool_loop": 4.6, "json_array": 1.41},
        )
        table = format_profile_table("test/AgentModel", cfg)
        # Should pick the peak workload (tool_loop here).
        assert "tool_loop" in table
        assert "4.6" in table

    def test_avoid_shows_worst_workload(self):
        cfg = ModelConfig(
            suffix_decoding_tier="avoid",
            suffix_bench_speedup={"chat": 0.78, "tool_loop": 1.5},
        )
        table = format_profile_table("test/AvoidModel", cfg)
        # AVOID surfaces the regression, not the peak.
        assert "chat" in table
        assert "0.78" in table

    def test_long_avoid_note_fits_box_when_max_width_set(self):
        """Regression: long ``avoid`` notes (e.g. ``gemma-4-26b``) used
        to overflow the right ``│`` border. Truncation must keep the
        tier word + numeric speedup whole while shortening the trailing
        rationale to ``…)``."""
        cfg = ModelConfig(
            suffix_decoding_tier="avoid",
            suffix_bench_speedup={"json_array": 0.20},
        )
        # 41 cols == the value-column width inside the rapid-mlx info
        # box (inner=60 minus the 17-char key field and 2-char ``: ``).
        cell = _suffix_tier_cell(cfg, max_width=41)
        assert len(cell) <= 41
        assert cell.startswith("avoid (json_array 0.20x")
        assert cell.endswith("…)")

    def test_short_tier_note_is_not_wrongly_truncated(self):
        """Tier notes that already fit must pass through untouched —
        the truncator should be a no-op on the common case."""
        cfg = ModelConfig(supports_spec_decode=False, is_hybrid=True)
        cell = _suffix_tier_cell(cfg, max_width=41)
        assert cell == "n/a (hybrid arch — spec decode off)"
        assert "…" not in cell

    def test_table_rows_all_same_width_for_long_avoid(self):
        """Box-frame alignment invariant: every bordered row must end at
        the same column. Pre-fix the ``Suffix tier`` row for an alias
        like ``gemma-4-26b`` would render past the right ``│``."""
        cfg = ModelConfig(
            suffix_decoding_tier="avoid",
            suffix_bench_speedup={"json_array": 0.20},
        )
        table = format_profile_table("mlx-community/gemma-4-26b-a4b-it-4bit", cfg)
        widths = {
            len(line)
            for line in table.splitlines()
            if line.startswith(("│", "┌", "└"))
        }
        assert len(widths) == 1, (
            f"All rows must be same printable width, got: {widths}\n{table}"
        )


class TestModelConfigDefaults:
    """Backward compatibility: existing ``ModelConfig`` constructions
    must not regress when the new fields default in."""

    def test_default_tier_is_unknown(self):
        assert ModelConfig().suffix_decoding_tier == "unknown"

    def test_default_speedup_dict_is_empty_not_none(self):
        # ``None`` would crash hint/table code that does ``or {}``;
        # default-empty-dict avoids the conditional and keeps semantics
        # ("we have no data" reads cleanly as ``not cfg.suffix_bench_speedup``).
        cfg = ModelConfig()
        assert cfg.suffix_bench_speedup == {}

    def test_each_modelconfig_gets_its_own_dict(self):
        """``field(default_factory=dict)`` regression test — using a
        bare ``dict()`` literal as the default would share the same
        dict across all ModelConfig instances."""
        a = ModelConfig()
        b = ModelConfig()
        a.suffix_bench_speedup["chat"] = 1.0
        assert b.suffix_bench_speedup == {}, (
            "ModelConfig instances must not share dict state"
        )
