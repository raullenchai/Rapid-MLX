# SPDX-License-Identifier: Apache-2.0
"""Stable stderr contract for deterministic optional-runtime preflights."""

from __future__ import annotations

import shlex
import subprocess
import sys
import textwrap
from collections import namedtuple

import pytest

MARKER_PREFIX = "RAPID-MLX-STARTUP-FAILURE:"
VISION_PYTHON = shlex.quote(sys.executable)
VISION_INSTALL_HINT = (
    "Install the validated vision stack into this runtime with:\n"
    f"    {VISION_PYTHON} -m pip install --upgrade --force-reinstall "
    "'rapid-mlx[vision]'\n"
    "or repair mlx-vlm directly (pinned to Rapid-MLX's validated set):\n"
    f"    {VISION_PYTHON} -m pip install --upgrade --force-reinstall "
    "'mlx-vlm==0.7.2'"
)


def _run_guard(source: str) -> subprocess.CompletedProcess[str]:
    guarded = (
        "try:\n"
        + textwrap.indent(textwrap.dedent(source), "    ")
        + "\nexcept Exception as exc:\n"
        + "    from rapid_mlx.cli import _handle_optional_runtime_missing\n"
        + "    _handle_optional_runtime_missing(exc)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", guarded],
        check=False,
        capture_output=True,
        text=True,
    )


def test_video_extra_guard_emits_one_stderr_marker_without_changing_cli_error() -> None:
    """The released Wan failure keeps its human text and rc, plus one marker."""
    result = _run_guard(
        """
        from collections import namedtuple
        from unittest.mock import patch
        import rapid_mlx.runtime.video_lane as lane

        Version = namedtuple("Version", "major minor")
        with patch.object(lane.sys, "version_info", Version(3, 11)), \\
             patch("importlib.util.find_spec", return_value=None), \\
             patch.object(lane, "_resolve_ffmpeg", return_value="ffmpeg"):
            lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        """
    )

    marker = f"{MARKER_PREFIX} runtime_extra_missing extra=video"
    human = (
        "\n  Error: video generation requires the `rapid-mlx[video]` Python extra.\n\n"
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(human)
    assert "pip install 'rapid-mlx[video]'" in result.stderr
    assert result.stderr.endswith(marker + "\n")
    assert result.stderr.count(marker) == 1


@pytest.mark.parametrize(
    ("source", "human", "marker"),
    [
        (
            """
            from collections import namedtuple
            from unittest.mock import patch
            import rapid_mlx.runtime.image_lane as lane

            Version = namedtuple("Version", "major minor")
            with patch.object(lane.sys, "version_info", Version(3, 11)), \\
                 patch("importlib.util.find_spec", return_value=None):
                lane.require_image_runtime_or_exit("flux2-klein-4b")
            """,
            "\n  Error: image generation requires the `rapid-mlx[image]` "
            "Python extra (`pip install 'rapid-mlx[image]'`).\n\n",
            f"{MARKER_PREFIX} runtime_extra_missing extra=image",
        ),
        (
            """
            from unittest.mock import patch
            from rapid_mlx.audio.probe import require_audio_or_exit

            with patch("importlib.util.find_spec", return_value=None):
                require_audio_or_exit("kokoro")
            """,
            "error: model 'kokoro' is an audio alias and requires the optional "
            "`mlx-audio` dependency (shipped with the [audio] extra).\n"
            "Install with: pip install 'rapid-mlx[audio]'\n",
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
            "error: model 'ui-tars-1.5-7b-4bit' is a vision/multimodal alias "
            "and requires the optional `mlx-vlm` dependency (shipped with the "
            "[vision] extra).\n"
            + VISION_INSTALL_HINT
            + "\nOr, if this checkpoint has a text-capable backbone and you "
            "only need text output, `--no-mllm` boots the text-only lane "
            "straight from the base wheel (no mlx-vlm, drops image/vision "
            "input).\n",
            f"{MARKER_PREFIX} runtime_extra_missing extra=vision",
        ),
    ],
)
def test_sibling_extra_guards_emit_closed_marker_once(
    source: str, human: str, marker: str
) -> None:
    result = _run_guard(source)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(human)
    assert result.stderr.endswith(marker + "\n")


@pytest.mark.parametrize(
    ("source", "human", "marker"),
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
            "\n  Error: video generation requires Python 3.11 or newer "
            "(current: 3.10). Rapid-MLX core still supports Python 3.10, but "
            "the upstream mlx-video runtime does not.\n\n",
            f"{MARKER_PREFIX} python_version_unsupported extra=video",
        ),
        (
            """
            from collections import namedtuple
            from unittest.mock import patch
            import rapid_mlx.runtime.video_lane as lane

            Version = namedtuple("Version", "major minor")
            with patch.object(lane.sys, "version_info", Version(3, 11)), \\
                 patch.object(lane, "_default_video_runtime_requirements", return_value=[]), \\
                 patch.object(lane, "_resolve_ffmpeg", return_value=None):
                lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
            """,
            "\n  Error: video generation requires ffmpeg (`brew install ffmpeg`).\n\n",
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
            "error: model 'ui-tars-1.5-7b-4bit' requires the Rapid-MLX vision "
            "lane, but mlx-vlm '0.0' is incompatible; this release validates "
            "exactly 0.7.2. This is a vision-runtime compatibility error, not "
            "a Metal out-of-memory error.\n" + VISION_INSTALL_HINT + "\n",
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
            "error: model 'ui-tars-1.5-7b-4bit' is a vision/multimodal alias, "
            "but the vision runtime cannot load.\n"
            "`mlx-vlm` is installed but its dependency 'PIL' is not, so the "
            "vision runtime cannot load. "
            + VISION_INSTALL_HINT
            + "\nAlternatively, repair just the missing dependency in this "
            "runtime:\n"
            f"    {VISION_PYTHON} -m pip install pillow\n",
            f"{MARKER_PREFIX} runtime_broken extra=vision",
        ),
    ],
)
def test_other_preflight_failures_use_only_closed_reason_tokens(
    source: str, human: str, marker: str
) -> None:
    result = _run_guard(source)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith(human)
    assert result.stderr.endswith(marker + "\n")


def test_healthy_vision_status_has_no_failure_output() -> None:
    result = _run_guard(
        """
        from unittest.mock import patch
        from rapid_mlx.models.mllm import (
            VisionRuntimeStatus,
            require_mlx_vlm_or_exit,
        )

        with patch(
            "rapid_mlx.models.mllm.vision_runtime_status",
            return_value=(VisionRuntimeStatus.OK, None),
        ):
            require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit")
        """
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def _handle_guard(call) -> None:
    from rapid_mlx.cli import _handle_optional_runtime_missing
    from rapid_mlx.runtime.optional_runtime import OptionalRuntimeMissing

    try:
        call()
    except OptionalRuntimeMissing as exc:
        _handle_optional_runtime_missing(exc)


def test_video_guard_marker_branches_are_covered_in_process(
    monkeypatch, capsys
) -> None:
    from rapid_mlx.runtime import video_lane

    version = namedtuple("Version", "major minor")
    monkeypatch.setattr(video_lane.sys, "version_info", version(3, 11))
    monkeypatch.setattr(video_lane, "_is_ltx25_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_is_cogvideox_name", lambda _name: False)
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        video_lane,
        "_default_video_runtime_requirements",
        lambda _name: ["rapid-mlx[video]"],
    )
    with pytest.raises(SystemExit, match="2"):
        _handle_guard(
            lambda: video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        )
    assert "runtime_extra_missing extra=video" in capsys.readouterr().err

    monkeypatch.setattr(
        video_lane, "_default_video_runtime_requirements", lambda _name: []
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: None)
    with pytest.raises(SystemExit, match="2"):
        _handle_guard(
            lambda: video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        )
    assert "runtime_dependency_missing extra=video" in capsys.readouterr().err

    monkeypatch.setattr(video_lane.sys, "version_info", version(3, 10))
    with pytest.raises(SystemExit, match="2"):
        _handle_guard(
            lambda: video_lane.require_video_runtime_or_exit("wan2.2-ti2v-5b-q8")
        )
    assert "python_version_unsupported extra=video" in capsys.readouterr().err


def test_image_and_audio_marker_branches_are_covered_in_process(
    monkeypatch, capsys
) -> None:
    from rapid_mlx.audio import probe
    from rapid_mlx.runtime import image_lane

    version = namedtuple("Version", "major minor")
    monkeypatch.setattr(image_lane.sys, "version_info", version(3, 11))
    monkeypatch.setattr(image_lane.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(SystemExit, match="2"):
        _handle_guard(
            lambda: image_lane.require_image_runtime_or_exit("flux2-klein-4b")
        )
    assert "runtime_extra_missing extra=image" in capsys.readouterr().err

    monkeypatch.setattr(image_lane.sys, "version_info", version(3, 10))
    with pytest.raises(SystemExit, match="2"):
        _handle_guard(
            lambda: image_lane.require_image_runtime_or_exit("flux2-klein-4b")
        )
    assert "python_version_unsupported extra=image" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        _handle_guard(lambda: probe.require_audio_or_exit("kokoro"))
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
        _handle_guard(lambda: mllm.require_mlx_vlm_or_exit("ui-tars-1.5-7b-4bit"))
    assert f"{expected_reason} extra=vision" in capsys.readouterr().err


def test_text_diffusion_missing_vision_marker_is_covered(monkeypatch, capsys) -> None:
    from rapid_mlx.models import mllm

    monkeypatch.setattr(
        mllm,
        "vision_runtime_status",
        lambda: (mllm.VisionRuntimeStatus.ABSENT, None),
    )
    with pytest.raises(SystemExit, match="2"):
        _handle_guard(
            lambda: mllm.require_mlx_vlm_or_exit(
                "diffusion-gemma-26b", text_diffusion=True
            )
        )
    assert "runtime_extra_missing extra=vision" in capsys.readouterr().err
