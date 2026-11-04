# SPDX-License-Identifier: Apache-2.0
"""#5577 e2e: an OpenCode session that outgrows the memory-fitted window compacts.

The server runs a small model under a tight Metal budget (an explicit
``--gpu-memory-utilization`` sized so ``max_model_len`` lands near 23k tokens,
the same ratio as a 27B model on a 32 GB Mac). OpenCode is configured through
the real ``launch opencode --isolated`` writer, then reads more file content
than fits. Passing means the client compacted (proactively or on a classified
``context_length_exceeded``), never saw a retryable 503, did not loop on
rejections, and kept working afterwards.

On origin/main before #5577 this fails: OpenCode is written with no context
limit, so it never compacts, and the over-memory prompt surfaces as a 503 /
mid-stream internal error that the client cannot act on.

Run manually (needs Metal, the cached model and ``opencode`` on PATH)::

    RAPID_MLX_E2E_OPENCODE=1 pytest --run-slow -m slow \
        tests/integrations/test_opencode_memory_compaction.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import pytest

# ``slow`` (not ``integration``): this test starts its own server.
pytestmark = [pytest.mark.slow, pytest.mark.real_hf_cache]

MODEL = "qwen3.5-9b-4bit"
PORT = int(os.environ.get("RAPID_MLX_E2E_PORT", "8463"))
# Device-specific: 0.025 of a 239 GB working set ≈ 6 GB (5.2 GB weights).
UTILIZATION = os.environ.get("RAPID_MLX_E2E_GPU_UTIL", "0.025")


def _models() -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/v1/models", timeout=2
        ) as response:
            return json.loads(response.read())
    except OSError:
        return None


@pytest.fixture
def server(tmp_path):
    if not os.environ.get("RAPID_MLX_E2E_OPENCODE"):
        pytest.skip("set RAPID_MLX_E2E_OPENCODE=1 to run the real-model e2e")
    if shutil.which("opencode") is None:
        pytest.skip("opencode is not installed")
    log_path = tmp_path / "serve.log"
    env = dict(os.environ, DO_NOT_TRACK="1")
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rapid_mlx.cli",
                "serve",
                MODEL,
                "--port",
                str(PORT),
                "--gpu-memory-utilization",
                UTILIZATION,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        deadline = time.time() + 300
        while _models() is None:
            assert proc.poll() is None, log_path.read_text()[-2000:]
            assert time.time() < deadline, "server did not start"
            time.sleep(2)
        yield log_path
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def test_opencode_compacts_instead_of_looping(server, tmp_path):
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    env = dict(os.environ, HOME=str(home), DO_NOT_TRACK="1")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rapid_mlx.cli",
            "launch",
            "opencode",
            "--isolated",
            "--model",
            MODEL,
            "--port",
            str(PORT),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    config_path = next((home / ".rapid-mlx" / "connect").rglob("opencode.json"))
    config = json.loads(config_path.read_text())
    provider_name, provider = next(iter(config["provider"].items()))
    limit = next(iter(provider["models"].values()))["limit"]
    entry = next(e for e in _models()["data"] if e["id"] == MODEL)
    ceiling = entry["max_model_len"]
    assert ceiling < entry["context_window"], "budget is not memory-bound"
    assert limit["context"] == int(ceiling * 0.9)
    assert limit["output"] < limit["context"]

    words = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "theta",
        "kappa",
        "lambda",
        "sigma",
    ]
    for i in range(10):
        body = "\n".join(
            " ".join(words[(i + j + k) % len(words)] for k in range(14)) + f" line{j}"
            for j in range(160)
        )
        (work / f"notes_{i:02d}.txt").write_text(
            f"# notes {i}\n{body}\nSECRET_{i:02d} = {1000 + 37 * i}\n"
        )
    offset = server.stat().st_size
    run = subprocess.run(
        [
            "opencode",
            "run",
            "--format",
            "json",
            "--model",
            f"{provider_name}/{MODEL}",
            "There are 10 files notes_00.txt .. notes_09.txt in the current "
            "directory. Use the read tool to read each file IN FULL, one file per "
            "tool call, in order. Each file ends with a line 'SECRET_NN = value'. "
            "After reading all ten, reply with the ten SECRET lines. Do not use "
            "grep, bash or any other tool.",
        ],
        cwd=work,
        env=dict(
            env,
            # OpenCode resolves its project directory from PWD, not getcwd().
            PWD=str(work),
            OPENCODE_CONFIG=str(config_path),
            OPENCODE_CONFIG_CONTENT=config_path.read_text(),
        ),
        capture_output=True,
        text=True,
        timeout=1500,
    )
    (tmp_path / "opencode.jsonl").write_text(run.stdout)
    statuses = re.findall(
        r'"POST /v1/[^"]+" (\d{3})', server.read_text(errors="replace")[offset:]
    )
    last_error = max(
        (i for i, status in enumerate(statuses) if status != "200"), default=-1
    )
    assert "503" not in statuses, statuses
    assert statuses.count("400") <= 3, statuses
    assert "compaction" in run.stdout, run.stdout[-2000:]
    assert "200" in statuses[last_error + 1 :], statuses
