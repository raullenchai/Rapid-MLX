# SPDX-License-Identifier: Apache-2.0
"""Hermetic tests for ``vllm_mlx.audio.music.MusicEngine`` (PR #1307).

No mlx, no weights, no network: the two external boundaries — the
``sa3_mlx.py`` subprocess and ``huggingface_hub.hf_hub_download`` — are
faked, so this suite runs on the Linux CI runner.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Load music.py directly, bypassing ``vllm_mlx.audio.__init__`` (which pulls in
# numpy/mlx via the STT/TTS lanes). music.py itself is stdlib-only, so this keeps
# the suite runnable on the mlx-free Linux CI runner.
_MUSIC_PY = Path(__file__).resolve().parents[1] / "vllm_mlx" / "audio" / "music.py"
_spec = importlib.util.spec_from_file_location("_vllm_mlx_music_under_test", _MUSIC_PY)
music = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(music)
MusicEngine = music.MusicEngine


def _argv_to_map(cmd: list[str]) -> dict[str, str]:
    """``--flag=value`` argv -> dict (MusicEngine uses the = form)."""
    out = {}
    for tok in cmd:
        if tok.startswith("--") and "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


@pytest.fixture
def no_weight_fetch(monkeypatch):
    """Pretend the SA3 weights are already materialized."""
    monkeypatch.setattr(MusicEngine, "_ensure_weights", lambda self: None)


@pytest.fixture
def fake_run(monkeypatch):
    """Capture argv; create the requested --out file so the check passes."""
    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        args = _argv_to_map(cmd)
        if "--out" in args:
            p = Path(args["--out"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"RIFF____WAVE")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(music.subprocess, "run", _run)
    return calls


# ---------------------------------------------------------------- argv contract


def test_generate_builds_expected_argv(tmp_path, no_weight_fetch, fake_run):
    out = tmp_path / "bgm.wav"
    got = MusicEngine().generate("epic war drums", out, seconds=12.5, steps=6)

    assert got == out.resolve()
    cmd, kwargs = fake_run[0]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("sa3_mlx.py")
    args = _argv_to_map(cmd)
    assert args["--prompt"] == "epic war drums"
    assert args["--dit"] == music.DEFAULT_DIT
    assert args["--decoder"] == music.DEFAULT_DECODER
    assert args["--steps"] == "6"
    assert args["--seconds"] == "12.50"
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 900
    # never through a shell
    assert not kwargs.get("shell", False)


def test_optional_flags_omitted_by_default(tmp_path, no_weight_fetch, fake_run):
    MusicEngine().generate("pad", tmp_path / "a.wav")
    cmd = fake_run[0][0]
    assert not any(t.startswith("--negative-prompt") for t in cmd)
    assert not any(t.startswith("--seed") for t in cmd)


def test_optional_flags_forwarded(tmp_path, no_weight_fetch, fake_run):
    MusicEngine(dit="sm-sfx", decoder="same-s").generate(
        "door slam", tmp_path / "b.wav", negative_prompt="vocals", seed=42
    )
    args = _argv_to_map(fake_run[0][0])
    assert args["--negative-prompt"] == "vocals"
    assert args["--seed"] == "42"
    assert args["--dit"] == "sm-sfx"
    assert args["--decoder"] == "same-s"


def test_prompt_starting_with_dash_is_not_parsed_as_flag(
    tmp_path, no_weight_fetch, fake_run
):
    """Regression: bare ``--prompt X`` would let "--play" become an option."""
    MusicEngine().generate("--play loud", tmp_path / "c.wav")
    cmd = fake_run[0][0]
    assert "--prompt=--play loud" in cmd
    # the prompt must never appear as its own bare argv token
    assert "--play loud" not in cmd


# ------------------------------------------------------------------- out_path


def test_relative_out_path_is_absolutized(
    tmp_path, monkeypatch, no_weight_fetch, fake_run
):
    """sa3_mlx.py re-roots a relative --out under its own vendored output/ dir."""
    monkeypatch.chdir(tmp_path)
    got = MusicEngine().generate("x", "rel.wav")

    assert got.is_absolute()
    assert got == (tmp_path / "rel.wav").resolve()
    assert _argv_to_map(fake_run[0][0])["--out"] == str(got)


def test_parent_directory_is_created(tmp_path, no_weight_fetch, fake_run):
    got = MusicEngine().generate("x", tmp_path / "deep" / "nested" / "d.wav")
    assert got.exists()


# --------------------------------------------------------------- error surface


def test_subprocess_failure_raises_runtime_error_with_stderr(
    tmp_path, monkeypatch, no_weight_fetch
):
    def _boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(3, cmd, b"", b"CUDA is not a thing here")

    monkeypatch.setattr(music.subprocess, "run", _boom)
    with pytest.raises(RuntimeError, match="exit 3") as ei:
        MusicEngine().generate("x", tmp_path / "e.wav")
    assert "CUDA is not a thing here" in str(ei.value)


def test_timeout_is_distinguishable_from_failure(
    tmp_path, monkeypatch, no_weight_fetch
):
    def _slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 900)

    monkeypatch.setattr(music.subprocess, "run", _slow)
    # TimeoutError is NOT a RuntimeError -> callers can tell the two apart
    with pytest.raises(TimeoutError, match="exceeded 900s"):
        MusicEngine().generate("x", tmp_path / "f.wav", timeout=900)


def test_missing_output_is_reported(tmp_path, monkeypatch, no_weight_fetch):
    monkeypatch.setattr(
        music.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""),
    )
    with pytest.raises(RuntimeError, match="produced no output"):
        MusicEngine().generate("x", tmp_path / "g.wav")


# ------------------------------------------------------------ weight resolution


@pytest.mark.parametrize("bad_dit", ["sm-music-xl", "", "large"])
def test_unknown_dit_rejected(bad_dit):
    with pytest.raises(ValueError, match="unknown dit"):
        MusicEngine(dit=bad_dit)._ensure_weights()


def test_unknown_decoder_rejected():
    with pytest.raises(ValueError, match="unknown decoder"):
        MusicEngine(decoder="same-xl")._ensure_weights()


def test_every_preset_maps_to_a_weight_file():
    """The presets advertised in the class docstring must resolve."""
    for dit in ("medium", "sm-music", "sm-sfx"):
        assert dit in music._DIT_NPZ
    for dec in ("same-l", "same-s"):
        assert dec in music._DECODER_NPZ
    assert music.DEFAULT_DIT in music._DIT_NPZ
    assert music.DEFAULT_DECODER in music._DECODER_NPZ


def test_ensure_weights_downloads_only_missing(tmp_path, monkeypatch):
    """DiT + decoder + shared T5Gemma are fetched; present files are skipped."""
    mlx_dir = tmp_path / "mlx"
    mlx_dir.mkdir()
    # t5gemma already on disk -> must NOT be re-downloaded
    (mlx_dir / "t5gemma_f16.npz").write_bytes(b"cached")
    monkeypatch.setattr(music, "_SA3_MLX_DIR", mlx_dir)

    requested = []

    def _dl(repo_id, filename):
        requested.append((repo_id, filename))
        src = tmp_path / "hf" / filename
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"tensors")
        return str(src)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=_dl),
    )

    MusicEngine(dit="sm-music", decoder="same-s")._ensure_weights()

    assert requested == [
        ("stabilityai/stable-audio-3-optimized", "MLX/dit_sm-music_f16.npz"),
        ("stabilityai/stable-audio-3-optimized", "MLX/same_s_decoder_f32.npz"),
    ]
    assert (mlx_dir / "dit_sm-music_f16.npz").exists()
    assert (mlx_dir / "t5gemma_f16.npz").read_bytes() == b"cached"


def test_ensure_weights_replaces_dangling_symlink(tmp_path, monkeypatch):
    """A dangling pointer must be replaced, not treated as present."""
    mlx_dir = tmp_path / "mlx"
    mlx_dir.mkdir()
    dangling = mlx_dir / "t5gemma_f16.npz"
    dangling.symlink_to(tmp_path / "nope" / "gone.npz")
    assert dangling.is_symlink() and not dangling.exists()
    monkeypatch.setattr(music, "_SA3_MLX_DIR", mlx_dir)

    def _dl(repo_id, filename):
        src = tmp_path / "hf" / filename
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"real")
        return str(src)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=_dl)
    )

    MusicEngine()._ensure_weights()
    assert dangling.exists()
    assert dangling.read_bytes() == b"real"


def test_no_weights_committed_to_git():
    """Guard: real tensors (or local-cache symlinks) must never land in git."""
    mlx_dir = Path(music.__file__).parent / "sa3" / "models" / "mlx"
    strays = list(mlx_dir.glob("*.npz")) if mlx_dir.exists() else []
    # Downloaded weights are gitignored; this only fails if someone commits one.
    for p in strays:
        assert not p.is_symlink() or p.exists(), f"dangling committed pointer: {p}"
