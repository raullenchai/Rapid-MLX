"""Sanity-check every shipped npz and every CLI config of sa3_mlx.py.

Run from inside the sa3_mlx/ folder:
    python test_all_configs.py
"""

from __future__ import annotations
import os, subprocess, sys, time, wave, tempfile
from pathlib import Path
import numpy as np


REPO = Path(__file__).resolve().parent.parent  # project root (scripts/ is one level down)
PY   = sys.executable


# ─── expected weight files ────────────────────────────────────────────────

EXPECTED_NPZ = {
    "models/mlx/dit_sm-music_f16.npz":     {"min_keys": 400, "must_have_prefix": "cond."},
    "models/mlx/dit_sm-sfx_f16.npz":       {"min_keys": 400, "must_have_prefix": "cond."},
    "models/mlx/dit_medium_f16.npz":       {"min_keys": 500, "must_have_prefix": "cond."},
    "models/mlx/same_s_encoder_f32.npz":   {"min_keys": 100},
    "models/mlx/same_s_decoder_f32.npz":   {"min_keys": 100},
    "models/mlx/same_l_encoder_f32.npz":   {"min_keys": 200},
    "models/mlx/same_l_decoder_f32.npz":   {"min_keys": 200},
    "models/mlx/t5gemma_f16.npz":          {"min_keys": 130, "must_have_key": "TOKENIZER_MODEL"},
}


# ─── helpers ──────────────────────────────────────────────────────────────

def check_npz(path: Path, spec: dict) -> tuple[bool, str]:
    """Open the npz and verify it has the expected shape/keys."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        return False, f"load failed: {e}"
    keys = list(z.files)
    if len(keys) < spec["min_keys"]:
        return False, f"only {len(keys)} keys (expected ≥ {spec['min_keys']})"
    if "must_have_prefix" in spec:
        prefix = spec["must_have_prefix"]
        if not any(k.startswith(prefix) for k in keys):
            return False, f"no keys with prefix {prefix!r}"
    if "must_have_key" in spec:
        if spec["must_have_key"] not in keys:
            return False, f"missing key {spec['must_have_key']!r}"
    # Sanity: actually read one tensor without error
    try:
        for k in keys[:3]:
            _ = np.asarray(z[k])
    except Exception as e:
        return False, f"tensor read failed: {e}"
    return True, f"{len(keys)} keys"


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sr  = w.getframerate()
        n   = w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, nch).T.astype(np.float32) / 32767.0
    return pcm, sr


def run_cli(name: str, extra_args: list[str], expected_seconds: float) -> tuple[bool, str]:
    """Run sa3_mlx.py with the given args; sanity-check the output WAV."""
    out = Path(tempfile.gettempdir()) / f"sa3_test_{name}.wav"
    cmd = [PY, str(REPO / "scripts" / "sa3_mlx.py"),
           "--seed", "42",
           "--out", str(out)] + extra_args
    if "--steps" not in extra_args:                    # default to 4 steps for speed
        cmd += ["--steps", "4"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                              cwd=str(REPO))
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after 180s"
    dt = time.time() - t0
    if proc.returncode != 0:
        last_err = proc.stderr.strip().split("\n")[-1] if proc.stderr.strip() else "(no stderr)"
        return False, f"exit {proc.returncode}: {last_err[:120]}"
    if not out.exists():
        return False, "no output WAV produced"

    pcm, sr = read_wav(str(out))
    # Quality gates: correct duration, non-NaN, not all-silent.
    actual_secs = pcm.shape[-1] / sr
    if abs(actual_secs - expected_seconds) > 0.1:
        return False, f"duration {actual_secs:.2f}s ≠ requested {expected_seconds:.2f}s"
    if not np.isfinite(pcm).all():
        return False, "audio contains NaN/Inf"
    peak = float(np.abs(pcm).max())
    rms  = float(np.sqrt((pcm ** 2).mean()))
    if peak < 0.005 or rms < 0.0005:
        return False, f"effectively silent (peak={peak:.4f} rms={rms:.4f})"

    out.unlink(missing_ok=True)
    return True, f"{dt:.1f}s  peak={peak:.3f} rms={rms:.3f}"


# ─── tests ────────────────────────────────────────────────────────────────

def test_weights():
    print("\n[ phase 1 ] verify every shipped npz loads cleanly\n")
    fails = []
    for relpath, spec in EXPECTED_NPZ.items():
        full = REPO / relpath
        if not full.exists():
            print(f"  ✗ {relpath:42s}  FILE MISSING")
            fails.append(relpath)
            continue
        ok, info = check_npz(full, spec)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {relpath:42s}  {info}")
        if not ok:
            fails.append(relpath)
    return fails


def test_configs():
    print("\n[ phase 2 ] end-to-end CLI configurations (steps=4, 3s clips)\n")
    SECONDS = "3"
    EXPECTED_SEC = 3.0

    # Will use this as init-audio for audio-to-audio + inpaint tests.
    # Generate one short clip with sm-music first.
    init_wav = Path(tempfile.gettempdir()) / "sa3_test_init.wav"
    print("  [setup] generating a small init audio for a2a/inpaint tests …")
    subprocess.run([PY, str(REPO / "scripts" / "sa3_mlx.py"),
                    "--prompt", "drums",
                    "--dit", "sm-music", "--decoder", "same-s",
                    "--seconds", SECONDS, "--seed", "1", "--steps", "4",
                    "--out", str(init_wav)],
                   capture_output=True, check=True, cwd=str(REPO))
    print(f"  [setup] init audio at {init_wav}\n")

    matrix = [
        # text-to-audio: each DiT × natural decoder
        ("t2a-music",       ["--prompt", "lofi house",   "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS]),
        ("t2a-sfx",         ["--prompt", "rain on roof", "--dit", "sm-sfx",   "--decoder", "same-s", "--seconds", SECONDS]),
        ("t2a-medium",      ["--prompt", "piano solo",   "--dit", "medium",   "--decoder", "same-l", "--seconds", SECONDS]),

        # cross-pairings (DiT/decoder mismatch — should still produce audio, may sound odd)
        ("cross-music-L",   ["--prompt", "lofi",         "--dit", "sm-music", "--decoder", "same-l", "--seconds", SECONDS]),
        ("cross-medium-S",  ["--prompt", "piano",        "--dit", "medium",   "--decoder", "same-s", "--seconds", SECONDS]),

        # mode: audio-to-audio
        ("a2a-music",       ["--prompt", "jazz fusion",  "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--init-audio", str(init_wav), "--init-noise-level", "0.6"]),
        ("a2a-medium",      ["--prompt", "ambient",      "--dit", "medium",   "--decoder", "same-l", "--seconds", SECONDS,
                             "--init-audio", str(init_wav), "--init-noise-level", "0.6"]),

        # mode: inpainting
        ("inpaint-music",   ["--prompt", "guitar solo",  "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--init-audio", str(init_wav), "--inpaint-range", "1,2"]),

        # CFG variants
        ("cfg3-music",      ["--prompt", "techno beat",  "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--cfg", "3.0"]),
        ("cfg3-neg",        ["--prompt", "techno beat",  "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--cfg", "3.0", "--negative-prompt", "vocals, drums"]),
        ("cfg0.5",          ["--prompt", "techno",       "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--cfg", "0.5"]),
        ("cfg-apg0-vanilla",["--prompt", "techno",       "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--cfg", "3.0", "--apg", "0.0"]),

        # dtype: fp32 DiT
        ("fp32-music",      ["--prompt", "lofi",         "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--dit-dtype", "fp32"]),

        # different step counts
        ("steps1",          ["--prompt", "lofi",         "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--steps", "1"]),
        ("steps16",         ["--prompt", "lofi",         "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--steps", "16"]),

        # --no-free-models
        ("no-free",         ["--prompt", "lofi",         "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS,
                             "--no-free-models"]),

        # empty prompt (unconditional)
        ("empty-prompt",    ["--prompt", "",             "--dit", "sm-music", "--decoder", "same-s", "--seconds", SECONDS]),
    ]

    fails = []
    for name, args in matrix:
        ok, info = run_cli(name, args, EXPECTED_SEC)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name:24s}  {info}")
        if not ok:
            fails.append((name, info))

    init_wav.unlink(missing_ok=True)
    return fails


def main():
    print("=" * 68)
    print("sa3_mlx full-stack self-test")
    print("=" * 68)
    npz_fails    = test_weights()
    config_fails = test_configs()

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    if not npz_fails and not config_fails:
        print("✓ ALL PASS")
        return 0
    print(f"npz failures   : {len(npz_fails)}")
    for f in npz_fails: print(f"    - {f}")
    print(f"config failures: {len(config_fails)}")
    for name, info in config_fails: print(f"    - {name}: {info}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
