# SPDX-License-Identifier: Apache-2.0
"""Tests for the recipe system: hardware detection, model recipes, engine."""

import json
import subprocess
import sys

import pytest

from vllm_mlx.recipes.engine import (
    _round_down_context,
    compute_recommendation,
    format_models_table,
    format_recommendation,
)
from vllm_mlx.recipes.hardware import (
    HARDWARE_PROFILES,
    HardwareProfile,
    detect_hardware,
)
from vllm_mlx.recipes.models import (
    MODEL_RECIPES,
    get_recipe,
    search_recipes,
)

# ── Hardware profiles ────────────────────────────────────────────


class TestHardwareProfiles:
    def test_profiles_not_empty(self):
        assert len(HARDWARE_PROFILES) > 20

    def test_all_profiles_have_required_fields(self):
        for hp in HARDWARE_PROFILES.values():
            assert hp.memory_gb > 0
            assert hp.bandwidth_gbs > 0
            assert hp.generation in ("m1", "m2", "m3", "m4")
            assert hp.chip
            assert hp.name

    def test_usable_memory(self):
        hp = HARDWARE_PROFILES["m4-max-128"]
        assert hp.usable_memory_gb == 124  # 128 - 4

    def test_known_profiles_exist(self):
        assert "m3-ultra-256" in HARDWARE_PROFILES
        assert "m4-max-128" in HARDWARE_PROFILES
        assert "m1-8" in HARDWARE_PROFILES

    def test_bandwidth_ranges(self):
        for hp in HARDWARE_PROFILES.values():
            assert 50 <= hp.bandwidth_gbs <= 1000, f"{hp.id} bandwidth out of range"

    def test_detect_hardware_returns_profile(self):
        """On macOS, should detect hardware. On other OS, returns None."""
        import platform

        result = detect_hardware()
        if platform.system() == "Darwin":
            assert result is not None
            assert isinstance(result, HardwareProfile)
        else:
            assert result is None


# ── Model recipes ────────────────────────────────────────────────


class TestModelRecipes:
    def test_recipes_not_empty(self):
        assert len(MODEL_RECIPES) >= 15

    def test_all_recipes_have_required_fields(self):
        for r in MODEL_RECIPES.values():
            assert r.model_id, f"{r.id} missing model_id"
            assert r.model_memory_gb > 0, f"{r.id} missing memory"
            assert r.kv_per_1k_tokens_mb > 0, f"{r.id} missing kv rate"
            assert r.measured_tps > 0, f"{r.id} missing tps"
            assert r.provider, f"{r.id} missing provider"
            assert r.architecture in (
                "dense",
                "moe",
                "moe-hybrid",
            ), f"{r.id} bad arch"

    def test_tool_parser_set_when_tool_calling(self):
        for r in MODEL_RECIPES.values():
            if r.tool_calling:
                assert r.tool_parser is not None, (
                    f"{r.id} has tool_calling but no parser"
                )

    def test_reasoning_parser_set_when_reasoning(self):
        for r in MODEL_RECIPES.values():
            if r.reasoning:
                assert r.reasoning_parser is not None, (
                    f"{r.id} has reasoning but no parser"
                )

    def test_bandwidth_efficiency_range(self):
        for r in MODEL_RECIPES.values():
            assert 0.3 <= r.bandwidth_efficiency <= 0.8, (
                f"{r.id} efficiency {r.bandwidth_efficiency} out of range"
            )

    def test_get_recipe_by_short_id(self):
        r = get_recipe("qwen3.5-35b")
        assert r is not None
        assert r.id == "qwen3.5-35b"

    def test_get_recipe_by_hf_id(self):
        r = get_recipe("mlx-community/Qwen3.5-4B-MLX-4bit")
        assert r is not None
        assert r.id == "qwen3.5-4b"

    def test_get_recipe_fuzzy(self):
        r = get_recipe("phi4")
        assert r is not None
        assert "phi4" in r.id

    def test_get_recipe_not_found(self):
        assert get_recipe("nonexistent-model-xyz") is None

    def test_search_all(self):
        results = search_recipes()
        assert len(results) == len(MODEL_RECIPES)

    def test_search_by_provider(self):
        results = search_recipes("Qwen")
        assert len(results) >= 5
        assert all(
            "qwen" in r.provider.lower() or "qwen" in r.id.lower() for r in results
        )

    def test_search_no_results(self):
        results = search_recipes("zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_qwen36_27b_recipe_exists(self):
        r = get_recipe("qwen3.6-27b")
        assert r is not None
        assert r.architecture == "dense"
        assert r.parameter_count == "27B"
        assert r.tool_calling is True
        assert r.reasoning is True

    def test_qwen36_35b_recipe_exists(self):
        r = get_recipe("qwen3.6-35b")
        assert r is not None
        assert r.architecture == "moe-hybrid"
        assert r.active_parameters == "3B"

    def test_qwen36_6bit_recipe_exists(self):
        r = get_recipe("qwen3.6-35b-6bit")
        assert r is not None
        assert r.quantization == "6bit"
        assert r.model_memory_gb > get_recipe("qwen3.6-35b").model_memory_gb

    def test_qwen36_search(self):
        results = search_recipes("qwen3.6")
        assert len(results) >= 3  # 27b, 35b-4bit, 35b-6bit


# ── Recipe engine ────────────────────────��───────────────────────


class TestRecipeEngine:
    @pytest.fixture
    def m4_max_64(self):
        return HARDWARE_PROFILES["m4-max-64"]

    @pytest.fixture
    def m1_8(self):
        return HARDWARE_PROFILES["m1-8"]

    @pytest.fixture
    def m3_ultra_256(self):
        return HARDWARE_PROFILES["m3-ultra-256"]

    @pytest.fixture
    def small_model(self):
        return get_recipe("qwen3.5-4b")

    @pytest.fixture
    def large_model(self):
        return get_recipe("qwen3.5-122b")

    def test_small_model_fits_everywhere(self, small_model, m1_8):
        rec = compute_recommendation(small_model, m1_8)
        assert rec.fits
        assert rec.max_context_tokens > 0
        assert rec.estimated_tps > 0

    def test_large_model_oom_on_small_hw(self, large_model, m1_8):
        rec = compute_recommendation(large_model, m1_8)
        assert not rec.fits
        assert rec.status == "oom"

    def test_large_model_fits_on_ultra(self, large_model, m3_ultra_256):
        rec = compute_recommendation(large_model, m3_ultra_256)
        assert rec.fits
        assert rec.status == "comfortable"

    def test_estimated_tps_scales_with_bandwidth(self, small_model):
        rec_ultra = compute_recommendation(
            small_model, HARDWARE_PROFILES["m3-ultra-256"]
        )
        rec_m4max = compute_recommendation(small_model, HARDWARE_PROFILES["m4-max-64"])
        # M4 Max has ~68% of Ultra bandwidth → tps should be ~68%
        ratio = rec_m4max.estimated_tps / rec_ultra.estimated_tps
        assert 0.5 < ratio < 0.9

    def test_command_contains_model_id(self, small_model, m4_max_64):
        rec = compute_recommendation(small_model, m4_max_64)
        assert small_model.model_id in rec.command

    def test_command_contains_parsers(self, small_model, m4_max_64):
        rec = compute_recommendation(small_model, m4_max_64)
        assert "--tool-call-parser hermes" in rec.command
        assert "--reasoning-parser qwen3" in rec.command

    def test_turboquant_recommended_when_tight(self):
        """Model that barely fits should get TurboQuant for longer context."""
        model = get_recipe("qwen3.5-35b")
        # M4 Pro 48GB: 48 - 20 - 4 = 24GB available, should fit but tight for long ctx
        hw = HARDWARE_PROFILES["m4-pro-48"]
        rec = compute_recommendation(model, hw, desired_context=65536)
        # With 24GB available and 32MB/1K tokens, max ~750K without turbo
        # Should not need turbo here since it fits
        assert rec.fits

    def test_desired_context_caps_command(self, small_model, m3_ultra_256):
        rec = compute_recommendation(small_model, m3_ultra_256, desired_context=8192)
        assert "--max-kv-cache-tokens 8192" in rec.command

    def test_round_down_context(self):
        assert _round_down_context(100000) == 65536
        assert _round_down_context(65536) == 65536
        assert _round_down_context(32768) == 32768
        assert _round_down_context(50000) == 32768
        assert _round_down_context(1000) == 2048  # minimum

    def test_format_recommendation_not_empty(self, small_model, m4_max_64):
        rec = compute_recommendation(small_model, m4_max_64)
        text = format_recommendation(rec)
        assert len(text) > 100
        assert small_model.name in text
        assert m4_max_64.name in text

    def test_format_recommendation_oom(self, large_model, m1_8):
        rec = compute_recommendation(large_model, m1_8)
        text = format_recommendation(rec)
        assert "OOM" in text

    def test_format_models_table(self, m4_max_64):
        recipes = search_recipes()
        text = format_models_table(recipes, m4_max_64)
        assert "M4 Max 64GB" in text
        assert "qwen3.5-4b" in text

    def test_format_models_table_no_hardware(self):
        recipes = search_recipes()
        text = format_models_table(recipes, None)
        assert "Available Models" in text


# ── CLI integration ──────────────────────────────────────────────


class TestCLIIntegration:
    def test_models_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "vllm_mlx.cli", "models", "--all-hardware"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Available Models" in result.stdout
        assert "qwen3.5-4b" in result.stdout

    def test_recipe_command(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm_mlx.cli",
                "recipe",
                "qwen3.5-4b",
                "--hardware",
                "m4-max-64",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Qwen3.5 4B" in result.stdout
        assert "rapid-mlx serve" in result.stdout

    def test_recipe_json_output(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm_mlx.cli",
                "recipe",
                "qwen3.5-4b",
                "--hardware",
                "m4-max-64",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout.strip())
        assert data["fits"] is True
        assert data["estimated_tps"] > 0

    def test_recipe_not_found(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm_mlx.cli",
                "recipe",
                "nonexistent",
                "--hardware",
                "m4-max-64",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_recipe_oom(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm_mlx.cli",
                "recipe",
                "qwen3.5-122b",
                "--hardware",
                "m1-8",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "OOM" in result.stdout

    def test_models_with_hardware(self):
        result = subprocess.run(
            [sys.executable, "-m", "vllm_mlx.cli", "models", "--hardware", "m4-max-64"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "M4 Max 64GB" in result.stdout
        assert "OOM" in result.stdout  # 122B should show OOM
