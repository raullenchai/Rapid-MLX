# SPDX-License-Identifier: Apache-2.0
"""The shared video runtime preflight wired into ``pull`` and ``models``.

The serve path already refuses a video alias whose runtime cannot work here.
This module pins the two other surfaces that must agree with it: ``pull`` runs
the same check *before* downloading weights, and the model catalog marks video
aliases that cannot run on this host. There is deliberately no static
capability matrix — both surfaces call the same import probe.
"""

from __future__ import annotations

import argparse
import io
import sys
from unittest.mock import patch

import pytest

from rapid_mlx import cli
from rapid_mlx.runtime import video_lane
from rapid_mlx.video import ltx25

LTX25_ALIAS = "ltx-2.5-mlx-q8"
LTX25_HF_PATH = "MrMofer/ltx-2.5-mlx-q8"


class _DownloadReachedError(Exception):
    """Sentinel proving ``pull`` passed the preflight into the download path."""


def _runtime(
    state: str,
    source: str | None = None,
    executable: str | None = None,
    detail: str | None = None,
    reason: str | None = None,
) -> ltx25.LTX25Runtime:
    return ltx25.LTX25Runtime(
        state=state,
        source=source,
        executable=executable,
        detail=detail,
        reason=reason,
    )


def _pull(args: argparse.Namespace) -> None:
    """Run ``pull_command`` with telemetry and activation emission disabled."""
    with patch.object(cli, "_emit_pull_activation"):
        cli.pull_command(args)


def _pull_model(model: str) -> argparse.Namespace:
    return argparse.Namespace(model=model)


def _capture_models_output() -> str:
    buffer = io.StringIO()
    with (
        patch.object(sys, "stdout", buffer),
        patch("rapid_mlx._version_check.print_staleness_warning_if_any"),
    ):
        cli.models_command(None)
    return buffer.getvalue()


# --- pull: preflight runs before the download ------------------------------


def test_pull_ltx25_absent_refuses_before_download(monkeypatch, capsys) -> None:
    """A damaged bundled runtime fails the pull instead of downloading 24 GB."""
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: _runtime("absent", "vendored"))
    monkeypatch.setattr(
        video_lane, "_resolve_ffmpeg", lambda: "/opt/homebrew/bin/ffmpeg"
    )
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **k: pytest.fail("must not download without a runtime"),
    )

    with pytest.raises(SystemExit) as exc:
        _pull(_pull_model(LTX25_ALIAS))

    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "LTX-2.5 runtime shipped with rapid-mlx" in out
    assert "ships with rapid-mlx but is missing or damaged" in out
    assert "git clone" not in out


def test_pull_ltx25_shadowed_copy_uses_removal_hint(monkeypatch, capsys) -> None:
    """A shadowing copy is fixed by removing the conflicting path.

    Reinstalling rapid-mlx cannot help: the bundle is intact, another copy
    simply wins the import path.
    """
    shadowed = (
        "ltx_pipelines_mlx resolves outside this rapid-mlx installation "
        "(/opt/other/ltx_pipelines_mlx); a shadowing copy is not trusted."
    )
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime("corrupt", "vendored", detail=shadowed, reason="shadowed"),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **k: pytest.fail("must not download without a runtime"),
    )

    with pytest.raises(SystemExit) as exc:
        _pull(_pull_model(LTX25_ALIAS))

    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "a different copy shadows it" in out
    assert "Remove the conflicting LTX-2.5 copy from PYTHONPATH" in out
    assert "/opt/other/ltx_pipelines_mlx" in out
    assert "reinstall" not in out.lower()
    assert "Repair the install" not in out


def test_pull_ltx25_damaged_copy_uses_repair_hint(monkeypatch, capsys) -> None:
    """A modified or partially installed bundle is repaired by reinstalling."""
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime(
            "corrupt",
            "vendored",
            detail="ltx_core_mlx files no longer match the audited snapshot.",
            reason="damaged",
        ),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **k: pytest.fail("must not download without a runtime"),
    )

    with pytest.raises(SystemExit) as exc:
        _pull(_pull_model(LTX25_ALIAS))

    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "ships with rapid-mlx but is missing or damaged" in out
    assert "Repair the install" in out
    assert "PYTHONPATH" not in out


def test_pull_ltx25_override_invalid_never_claims_bundled_runtime(
    monkeypatch, capsys
) -> None:
    """A bad explicit override is fixed at the override, not by reinstalling."""
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime(
            "override_invalid",
            "override",
            detail="RAPID_MLX_LTX25_RUNTIME does not point at the pinned revision.",
        ),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **k: pytest.fail("must not download without a runtime"),
    )

    with pytest.raises(SystemExit) as exc:
        _pull(_pull_model(LTX25_ALIAS))

    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "RAPID_MLX_LTX25_RUNTIME" in out
    assert ltx25.LTX25_RUNTIME_COMMIT in out
    assert "ships with rapid-mlx" not in out


def test_pull_ltx25_python_floor_uses_upgrade_hint(monkeypatch, capsys) -> None:
    """On the 3.10 floor the hint upgrades Python, not the same interpreter."""
    monkeypatch.setattr(ltx25, "ltx25_runtime", lambda: _runtime("python_unsupported"))
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **k: pytest.fail("must not download without a runtime"),
    )

    with pytest.raises(SystemExit) as exc:
        _pull(_pull_model(LTX25_ALIAS))

    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "Python 3.11 or newer" in out
    assert "uv tool install --force --python 3.12" in out
    assert "pip install --upgrade --force-reinstall" not in out
    assert "ships with rapid-mlx" not in out


def test_pull_ltx25_ffmpeg_only_uses_ffmpeg_hint(monkeypatch, capsys) -> None:
    """A trusted runtime with no ffmpeg fails with an ffmpeg-only fix."""
    monkeypatch.setattr(
        ltx25,
        "ltx25_runtime",
        lambda: _runtime("trusted", "vendored", executable=sys.executable),
    )
    monkeypatch.setattr(video_lane, "_resolve_ffmpeg", lambda: None)
    monkeypatch.setattr(
        cli,
        "_pull_repository",
        lambda *a, **k: pytest.fail("must not download without a runtime"),
    )

    with pytest.raises(SystemExit) as exc:
        _pull(_pull_model(LTX25_ALIAS))

    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "brew install ffmpeg" in out
    assert "ships with rapid-mlx" not in out


def test_pull_ltx25_preflights_with_provisioning(monkeypatch) -> None:
    """``pull`` uses the provisioning form, like ``serve``, and only then downloads."""
    seen: dict[str, object] = {}

    def fake_issue(model_name: str, *, provision: bool = True):
        seen["model_name"] = model_name
        seen["provision"] = provision
        return None

    monkeypatch.setattr(video_lane, "video_runtime_issue", fake_issue)

    def fake_pull(args, **_kwargs):
        raise _DownloadReachedError

    monkeypatch.setattr(cli, "_pull_repository", fake_pull)

    with pytest.raises(_DownloadReachedError):
        _pull(_pull_model(LTX25_ALIAS))

    assert seen == {"model_name": LTX25_HF_PATH, "provision": True}


def test_pull_non_video_alias_skips_video_preflight(monkeypatch) -> None:
    """Text pulls never touch the video lane."""
    monkeypatch.setattr(
        video_lane,
        "video_runtime_issue",
        lambda *a, **k: pytest.fail("text pulls must not run the video preflight"),
    )

    def fake_pull(args, **_kwargs):
        raise _DownloadReachedError

    monkeypatch.setattr(cli, "_pull_repository", fake_pull)

    with pytest.raises(_DownloadReachedError):
        _pull(_pull_model("mlx-community/Qwen3-0.6B-4bit"))


def test_pull_unknown_model_skips_video_preflight(monkeypatch) -> None:
    """A repo id outside the catalog has no profile to preflight."""
    monkeypatch.setattr(
        video_lane,
        "video_runtime_issue",
        lambda *a, **k: pytest.fail("unknown repos have no video profile"),
    )

    def fake_pull(args, **_kwargs):
        raise _DownloadReachedError

    monkeypatch.setattr(cli, "_pull_repository", fake_pull)

    with pytest.raises(_DownloadReachedError):
        _pull(_pull_model("acme/not-in-the-catalog"))


# --- models: catalog marks what cannot run here ----------------------------


def test_models_json_marks_only_video_aliases(monkeypatch) -> None:
    """``runtime_ready`` is a video-only key; other modalities stay byte-shape."""
    monkeypatch.setattr(
        cli,
        "_video_alias_runtime_ready",
        lambda alias: alias == LTX25_ALIAS,
    )

    payload = cli._available_models_json_payload()

    video = payload["video"]
    assert video, "expected at least one video alias"
    for entry in video:
        assert isinstance(entry["runtime_ready"], bool)
        assert entry["runtime_ready"] is (entry["alias"] == LTX25_ALIAS)
    for modality in ("text", "image", "audio"):
        for entry in payload[modality]:
            assert "runtime_ready" not in entry


def test_models_text_table_shows_video_runtime_column(monkeypatch) -> None:
    """The video table tells a human which aliases need setup."""
    monkeypatch.setattr(
        cli,
        "_video_alias_runtime_ready",
        lambda alias: alias == LTX25_ALIAS,
    )

    out = _capture_models_output()
    head, _, video_section = out.partition("Video models (")
    assert video_section, "expected a 'Video models (N aliases)' section"
    lines = video_section.splitlines()
    header_index = next(index for index, line in enumerate(lines) if "Runtime" in line)
    assert header_index < next(
        index for index, line in enumerate(lines) if "[video:gen]" in line
    )

    rows = {
        line.split()[0]: line
        for line in video_section.splitlines()
        if "[video:gen]" in line
    }
    assert "ready" in rows[LTX25_ALIAS]
    assert all(
        "needs setup" in line for alias, line in rows.items() if alias != LTX25_ALIAS
    )


def test_models_text_table_marks_every_video_alias_needs_setup(monkeypatch) -> None:
    """When the lane is unavailable on this host every row says so."""
    monkeypatch.setattr(cli, "_video_alias_runtime_ready", lambda _alias: False)

    out = _capture_models_output()
    _, _, video_section = out.partition("Video models (")
    rows = [line for line in video_section.splitlines() if "[video:gen]" in line]
    assert rows
    assert all("needs setup" in line for line in rows)


# --- hint text contract ----------------------------------------------------


def test_ltx25_python_floor_hint_has_no_same_interpreter_pip_reinstall() -> None:
    """The floor hint must not suggest pip-installing into Python 3.10."""
    from rapid_mlx.runtime.base_runtime import python_upgrade_hint

    hint = python_upgrade_hint()
    assert "Python 3.11 or newer" in hint
    assert "uv tool install --force --python 3.12" in hint
    assert "pip install" not in hint


def test_ltx25_ffmpeg_hint_is_specific_to_encoding() -> None:
    """ffmpeg-only failures get an ffmpeg fix, not a reinstall."""
    assert "ffmpeg" in video_lane._LTX25_FFMPEG_HINT
    assert "brew install ffmpeg" in video_lane._LTX25_FFMPEG_HINT
    assert "ships with rapid-mlx" not in video_lane._LTX25_FFMPEG_HINT
