from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_benchmark_requires_explicit_checkpoint_runtime_trust(tmp_path) -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "benchmark_deepseek_v41_dspark.py"
    )
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
