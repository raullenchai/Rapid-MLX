from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import mlx.core as mx


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_deepseek_v41_dspark.py"
    spec = importlib.util.spec_from_file_location("benchmark_deepseek_v41_dspark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_requires_explicit_checkpoint_runtime_trust(tmp_path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_deepseek_v41_dspark.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--target",
            str(tmp_path / "target"),
            "--overlay",
            str(tmp_path / "overlay"),
            "--checkpoint-runtime",
            str(tmp_path / "runtime"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--trust-checkpoint-runtime" in result.stderr


def test_greedy_prefix_stops_at_accepted_eos() -> None:
    module = _load_script()
    eos_id = 1
    candidate = [9, eos_id, 7]
    logits = mx.zeros((1, 3, 10))
    logits[0, 0, eos_id] = 1
    logits[0, 1, 7] = 1

    committed, mismatch, hit_eos, accepted = module._match_greedy_prefix(
        candidate, logits, eos_id, False
    )

    assert committed == [9, eos_id]
    assert mismatch is None
    assert hit_eos is True
    assert accepted == 1
