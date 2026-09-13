"""Contracts for the Qwen3.5-family paired benchmark."""

from __future__ import annotations

import pytest

from scripts.benchmark_qwen35_fused_gdn_decode import _coefficient_of_variation


def test_coefficient_of_variation_accepts_stable_samples():
    assert _coefficient_of_variation([114.2, 114.8, 114.5]) < 0.01


def test_coefficient_of_variation_exposes_noisy_host():
    assert _coefficient_of_variation([19.7, 93.1, 23.5, 112.9]) > 0.5


def test_coefficient_of_variation_rejects_zero_mean():
    assert _coefficient_of_variation([0.0, 0.0]) == pytest.approx(float("inf"))
