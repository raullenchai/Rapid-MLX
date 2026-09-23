# SPDX-License-Identifier: Apache-2.0
"""Stable stderr contract for deterministic optional-runtime preflights."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections import namedtuple

import pytest

MARKER_PREFIX = "RAPID_MLX_STARTUP_FAILURE:"


def _run_guard(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_video_extra_guard_emits_one_stderr_marker_without_changing_cli_error() -> None:
    """The released Wan failure keeps its human text and rc, plus one marker."""
    result = _run_guard(
        """
        from unittest.mock import patch
        from rapid_mlx.runtime.video_lane import require_video_runtime_or_exit

        with patch("importlib.util.find_spec", return_value=None), \\
             patch("rapid_mlx.runtime.video_lane._resolve_ffmpeg", return_value="ffmpeg"):
            require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        """
    )

    marker = f"{MARKER_PREFIX} runtime_extra_missing extra=video"
    human = (
        "\n  Error: video generation requires the `rapid-mlx[video]` Python extra.\n\n"
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == human + marker + "\n"
    assert result.stderr.count(marker) == 1


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        (
            """
            from unittest.mock import patch
            from rapid_mlx.runtime.image_lane import require_image_runtime_or_exit

            with patch("importlib.util.find_spec", return_value=None):
                require_image_runtime_or_exit("flux2-klein-4b")
            """,
            f"{MARKER_PREFIX} runtime_extra_missing extra=image",
        ),
        (
            """
            from unittest.mock import patch
            from rapid_mlx.audio.probe import require_audio_or_exit

            with patch("importlib.util.find_spec", return_value=None):
                require_audio_or_exit("kokoro")
            """,
            f"{MARKER_PREFIX} runtime_extra_missing extra=audio",
        ),
        (
            """
            from unittest.mock import patch
            from rapid_mlx.models.mllm import (
                VisionRuntimeStatus,
                require_mlx_vlm_or_exit,
            )

            with patch(
                "rapid_mlx.models.mllm.vision_runtime_status",
                return_value=(VisionRuntimeStatus.ABSENT, None),
            ):
                require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
            """,
            f"{MARKER_PREFIX} runtime_extra_missing extra=vision",
        ),
    ],
)
def test_sibling_extra_guards_emit_closed_marker_once(source: str, marker: str) -> None:
    result = _run_guard(source)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.count(MARKER_PREFIX) == 1
    assert result.stderr.splitlines()[-1] == marker


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        (
            """
            from collections import namedtuple
            from unittest.mock import patch
            import rapid_mlx.runtime.video_lane as lane

            Version = namedtuple("Version", "major minor")
            with patch.object(lane.sys, "version_info", Version(3, 10)):
                lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
            """,
            f"{MARKER_PREFIX} python_version_unsupported extra=video",
        ),
        (
            """
            from unittest.mock import patch
            import rapid_mlx.runtime.video_lane as lane

            with patch.object(lane, "_default_video_runtime_requirements", return_value=[]), \\
                 patch.object(lane, "_resolve_ffmpeg", return_value=None):
                lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
            """,
            f"{MARKER_PREFIX} runtime_dependency_missing extra=video",
        ),
        (
            """
            from unittest.mock import patch
            from rapid_mlx.models.mllm import (
                VisionRuntimeStatus,
                require_mlx_vlm_or_exit,
            )

            with patch(
                "rapid_mlx.models.mllm.vision_runtime_status",
                return_value=(VisionRuntimeStatus.INCOMPATIBLE, "0.0"),
            ):
                require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
            """,
            f"{MARKER_PREFIX} runtime_incompatible extra=vision",
        ),
        (
            """
            from unittest.mock import patch
            from rapid_mlx.models.mllm import (
                VisionRuntimeStatus,
                require_mlx_vlm_or_exit,
            )

            with patch(
                "rapid_mlx.models.mllm.vision_runtime_status",
                return_value=(VisionRuntimeStatus.BROKEN, "PIL"),
            ):
                require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
            """,
            f"{MARKER_PREFIX} runtime_broken extra=vision",
        ),
    ],
)
def test_other_preflight_failures_use_only_closed_reason_tokens(
    source: str, marker: str
) -> None:
    result = _run_guard(source)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.count(MARKER_PREFIX) == 1
    assert result.stderr.splitlines()[-1] == marker


def test_video_guard_marker_branches_are_covered_in_process(
    monkeypatch, capsys
) -> None:
    from rapid_mlx.runtime import video_lane

    monkeypatch.setattr(video_lane, "_is_ltx25_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_is_cogvideox_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        video_lane,
        "_default_video_runtime_requirements",
        lambda _name: ["rapid-mlx[video]"],
    )
    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
    assert "runtime_extra_missing extra=video" in capsys.readouterr().err

    monkeypatch.setattr(
        video_lane, "_default_video_runtime_requirements", lambda _name: []
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: None)
    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
    assert "runtime_dependency_missing extra=video" in capsys.readouterr().err

    version = namedtuple("Version", "major minor")
    monkeypatch.setattr(video_lane.sys, "version_info", version(3, 10))
    with pytest.raises(SystemExit, match="2"):
        video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
    assert "python_version_unsupported extra=video" in capsys.readouterr().err


def test_image_and_audio_marker_branches_are_covered_in_process(
    monkeypatch, capsys
) -> None:
    from rapid_mlx.audio import probe
    from rapid_mlx.runtime import image_lane

    monkeypatch.setattr(image_lane.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(SystemExit, match="2"):
        image_lane.require_image_runtime_or_exit("flux2-klein-4b")
    assert "runtime_extra_missing extra=image" in capsys.readouterr().err

    version = namedtuple("Version", "major minor")
    monkeypatch.setattr(image_lane.sys, "version_info", version(3, 10))
    with pytest.raises(SystemExit, match="2"):
        image_lane.require_image_runtime_or_exit("flux2-klein-4b")
    assert "python_version_unsupported extra=image" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        probe.require_audio_or_exit("kokoro")
    assert "runtime_extra_missing extra=audio" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status_name", "expected_reason"),
    [
        ("ABSENT", "runtime_extra_missing"),
        ("BROKEN", "runtime_broken"),
        ("INCOMPATIBLE", "runtime_incompatible"),
    ],
)
def test_vision_marker_branches_are_covered_in_process(
    monkeypatch, capsys, status_name: str, expected_reason: str
) -> None:
    from rapid_mlx.models import mllm

    status = getattr(mllm.VisionRuntimeStatus, status_name)
    monkeypatch.setattr(mllm, "vision_runtime_status", lambda: (status, "PIL"))
    with pytest.raises(SystemExit, match="2"):
        mllm.require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
    assert f"{expected_reason} extra=vision" in capsys.readouterr().err


def test_text_diffusion_missing_vision_marker_is_covered(monkeypatch, capsys) -> None:
    from rapid_mlx.models import mllm

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, None),
    )
    with pytest.raises(SystemExit, match="2"):
        mllm.require_mlx_vlm_or_exit("diffusion-gemma-26b", text_diffusion=True)
    assert "runtime_extra_missing extra=vision" in capsys.readouterr().err
