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
    assert args["--seconds"] == "12.5"
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


def test_stale_output_is_not_reported_as_success(
    tmp_path, monkeypatch, no_weight_fetch
):
    """A previous wav at out_path must not mask a no-op generation."""
    stale = tmp_path / "h.wav"
    stale.write_bytes(b"OLD AUDIO")
    monkeypatch.setattr(
        music.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""),
    )
    with pytest.raises(RuntimeError, match="produced no output"):
        MusicEngine().generate("x", stale)
    assert not stale.exists()


def test_existing_output_is_overwritten(tmp_path, no_weight_fetch, fake_run):
    """The happy path still replaces an existing file."""
    out = tmp_path / "i.wav"
    out.write_bytes(b"OLD AUDIO")
    got = MusicEngine().generate("x", out)
    assert got.read_bytes() == b"RIFF____WAVE"


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


def test_ensure_weights_fetches_missing_and_returns_cache_paths(tmp_path, monkeypatch):
    """DiT + decoder are fetched to the HF cache; a real vendored file is used
    as-is; the returned map points at the cache paths, never the package dir."""
    mlx_dir = tmp_path / "mlx"
    mlx_dir.mkdir()
    # t5gemma already present as a real vendored file -> must NOT be re-downloaded
    (mlx_dir / "t5gemma_f16.npz").write_bytes(b"cached")
    monkeypatch.setattr(music, "_SA3_MLX_DIR", mlx_dir)

    requested = []

    def _dl(repo_id, filename):
        requested.append((repo_id, filename))
        src = tmp_path / "hf" / filename  # stand-in for the HF cache
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"tensors")
        return str(src)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=_dl),
    )

    resolved = MusicEngine(dit="sm-music", decoder="same-s")._ensure_weights()

    # Only the two absent files hit the network; the present t5gemma is skipped.
    assert requested == [
        ("stabilityai/stable-audio-3-optimized", "MLX/dit_sm-music_f16.npz"),
        ("stabilityai/stable-audio-3-optimized", "MLX/same_s_decoder_f32.npz"),
    ]
    # Downloaded files resolve to the HF cache, NOT the (possibly read-only)
    # vendored package dir.
    assert (
        resolved["dit_sm-music_f16.npz"]
        == tmp_path / "hf" / "MLX" / "dit_sm-music_f16.npz"
    )
    assert (
        resolved["same_s_decoder_f32.npz"]
        == tmp_path / "hf" / "MLX" / "same_s_decoder_f32.npz"
    )
    # The already-vendored real file is used in place.
    assert resolved["t5gemma_f16.npz"] == mlx_dir / "t5gemma_f16.npz"
    # Nothing was written into the package dir for the fetched components.
    assert not (mlx_dir / "dit_sm-music_f16.npz").exists()
    assert not (mlx_dir / "same_s_decoder_f32.npz").exists()
    assert sorted(p.name for p in mlx_dir.iterdir()) == ["t5gemma_f16.npz"]


def test_ensure_weights_does_not_write_into_readonly_package_dir(tmp_path, monkeypatch):
    """Regression for the read-only-prod crash: when the vendored package dir
    is read-only (as under a pip/site-packages install), resolving weights must
    still succeed by loading from the HF cache rather than symlinking in."""
    mlx_dir = tmp_path / "mlx"
    mlx_dir.mkdir()
    monkeypatch.setattr(music, "_SA3_MLX_DIR", mlx_dir)

    def _dl(repo_id, filename):
        src = tmp_path / "hf" / filename
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"real")
        return str(src)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=_dl)
    )

    # Make the package dir read-only — any attempt to symlink/copy a weight
    # into it would raise PermissionError.
    mlx_dir.chmod(0o500)
    try:
        resolved = MusicEngine()._ensure_weights()  # medium / same-l
    finally:
        mlx_dir.chmod(0o700)  # restore so tmp cleanup can remove it

    # All three components resolved to the writable cache, none into the dir.
    assert set(resolved) == {
        "dit_medium_f16.npz",
        "same_l_decoder_f32.npz",
        "t5gemma_f16.npz",
    }
    for name, path in resolved.items():
        assert path == tmp_path / "hf" / "MLX" / name
    assert list(mlx_dir.iterdir()) == []  # package dir untouched


# --------------------------------------------------------------- seconds guard


@pytest.mark.parametrize("bad", [0, 0.0, -1, -12.5])
def test_generate_rejects_nonpositive_seconds(tmp_path, bad):
    """seconds <= 0 must fail fast (empty/degenerate output otherwise)."""
    with pytest.raises(ValueError, match="seconds must be in"):
        MusicEngine().generate("x", tmp_path / "a.wav", seconds=bad)


@pytest.mark.parametrize("bad", [47.01, 60, 1000])
def test_generate_rejects_too_long_seconds(tmp_path, bad):
    """seconds beyond SA3's ~47s support must be rejected (memory blowup)."""
    with pytest.raises(ValueError, match="seconds must be in"):
        MusicEngine().generate("x", tmp_path / "a.wav", seconds=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_generate_rejects_nonfinite_seconds(tmp_path, bad):
    """NaN/inf slip past a plain range check — must be rejected explicitly."""
    with pytest.raises(ValueError, match="seconds must be in"):
        MusicEngine().generate("x", tmp_path / "a.wav", seconds=bad)


def test_generate_rejects_seconds_before_touching_output(tmp_path):
    """Validation happens before the destination is deleted — a bad seconds
    value must not destroy a pre-existing file."""
    out = tmp_path / "keep.wav"
    out.write_bytes(b"PRIOR")
    with pytest.raises(ValueError, match="seconds must be in"):
        MusicEngine().generate("x", out, seconds=0)
    assert out.read_bytes() == b"PRIOR"


def test_generate_accepts_boundary_seconds(tmp_path, no_weight_fetch, fake_run):
    """The 47s upper bound and small positive values are accepted."""
    MusicEngine().generate("x", tmp_path / "a.wav", seconds=47.0)
    assert _argv_to_map(fake_run[0][0])["--seconds"] == "47.0"
    MusicEngine().generate("x", tmp_path / "b.wav", seconds=0.5)
    assert _argv_to_map(fake_run[1][0])["--seconds"] == "0.5"


def test_generate_forwards_full_seconds_precision(tmp_path, no_weight_fetch, fake_run):
    """Regression: a fixed 2-decimal format collapsed small valid durations to
    ``0.00`` (empty clip). The full float must be forwarded verbatim."""
    MusicEngine().generate("x", tmp_path / "tiny.wav", seconds=0.003)
    assert _argv_to_map(fake_run[0][0])["--seconds"] == "0.003"


def test_generate_unlinks_symlink_destination_not_its_target(
    tmp_path, no_weight_fetch, fake_run
):
    """Regression: resolving the full out_path would dereference a final
    symlink, so unlink would delete the link's TARGET. The link itself must be
    removed and the target left intact."""
    target = tmp_path / "precious.wav"
    target.write_bytes(b"DO NOT DELETE")
    link = tmp_path / "out.wav"
    link.symlink_to(target)

    got = MusicEngine().generate("x", link)

    # The symlink was replaced (fake_run writes a fresh file at the link path),
    # and the original target was never touched.
    assert target.read_bytes() == b"DO NOT DELETE"
    assert not got.is_symlink()
    assert got.read_bytes() == b"RIFF____WAVE"


def test_no_weights_tracked_by_git():
    """Guard: no ``*.npz`` may be TRACKED under sa3/models/mlx.

    Weights are multi-hundred-MB and are downloaded at runtime. Committing
    either a real tensor or a symlink into someone's local HuggingFace cache
    (the original state of this PR) must fail here. Asked of git rather than
    the filesystem, so a developer's downloaded weights don't trip it.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")

    proc = subprocess.run(
        ["git", "ls-files", "--", "vllm_mlx/audio/sa3/models/mlx"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"git ls-files unavailable: {proc.stderr.strip()}")

    tracked = [ln for ln in proc.stdout.split() if ln.endswith(".npz")]
    assert tracked == [], f"weights must not be committed to git: {tracked}"
