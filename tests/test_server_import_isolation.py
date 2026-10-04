"""Guard: ``import rapid_mlx.server`` must not load optional heavy lanes.

The text server is the core product. Vision, audio, image, video, the
System-One decision server and the 14k-line CLI command module are opt-in
(separate extras or separate entry points) and are imported lazily when a
request or flag actually needs them. Loading any of them at server import
time costs cold-start latency and memory for every text-only user, and makes
the base install depend on extras it does not declare.

Runs in a fresh interpreter so modules imported by other tests in this
session cannot mask (or fake) a leak.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

# Module prefixes that must stay unloaded after ``import rapid_mlx.server``.
FORBIDDEN_AT_SERVER_IMPORT = (
    # Optional third-party runtimes (extras or not dependencies at all).
    # ``torch`` is deliberately absent: when it is installed (the ``vision``
    # extra pulls it in), ``transformers.modeling_rope_utils`` imports it at
    # module scope, which Rapid-MLX cannot avoid while using transformers.
    "mlx_vlm",
    "mlx_audio",
    "diffusers",
    "cv2",
    "mflux",
    "mlx_video",
    "videox_fun_mlx",
    # Rapid-MLX lanes that are only needed for non-text models.
    "rapid_mlx.image",
    "rapid_mlx.video",
    "rapid_mlx.models.mllm",
    "rapid_mlx.models.mlx_vlm_vendored",
    "rapid_mlx.mllm_batch_generator",
    "rapid_mlx.system_one",
    # The CLI command module; the server must not depend on it.
    "rapid_mlx.cli",
)


def _loaded_after_server_import() -> list[str]:
    env = dict(os.environ, RAPID_MLX_TELEMETRY="0")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, sys; import rapid_mlx.server; "
                "print(json.dumps(sorted(sys.modules)))",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "`import rapid_mlx.server` did not finish within 180s; stderr "
            f"tail: {str(exc.stderr or '')[-2000:]}"
        )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_server_import_does_not_load_optional_lanes() -> None:
    loaded = _loaded_after_server_import()
    leaked = sorted(
        module
        for module in loaded
        for prefix in FORBIDDEN_AT_SERVER_IMPORT
        if module == prefix or module.startswith(prefix + ".")
    )
    assert not leaked, (
        "`import rapid_mlx.server` eagerly loaded optional modules; import "
        f"them lazily where they are used instead: {leaked[:10]}"
    )
