"""Pre-encode audio files into SAME latents for LoRA finetuning (MLX, torch-free).

Offline audio → latent CLI mirroring underfit's ``dataset_processing/pre_encode.py``
conventions, ported to the standalone MLX runtime: single-process, soundfile-based
loading (no torch / torchaudio), and ``models.defs.audio_encoding.encode_audio``
doing the padding / chunking / masking.

Every audio file under ``--audio-dir`` (recursive) becomes a ``.npy`` + ``.json``
pair at the same relative path under ``--output-dir``:

    <output-dir>/<relpath-with-npy-suffix>       latents [D, T] float32
    <output-dir>/<relpath-with-json-suffix>      metadata (underfit field parity)

Conventions matched to underfit:
  * Whole files, no clip splitting; cap at ``--max-duration`` seconds (default
    600) with the sample cap aligned DOWN to a multiple of 4096 (ds_ratio).
  * 44.1 kHz stereo — mono is duplicated to stereo, >2 channels keep the first
    two, other sample rates are linearly resampled. NO loudness/peak norm.
  * Encoder runs fp32; ``chunked=True`` for clips longer than 30 s.
  * Metadata: ``path, relpath, src_relpath, seconds_total`` (rounded 3dp),
    ``seconds_start: 0, audio_samples, latent_shape``, latent-resolution 0/1
    ``padding_mask``, merged with tags extracted from sidecars.
  * Skip files whose outputs already exist unless ``--overwrite``.
  * ``<output-dir>/details.json`` records the global encode params.

Latent scaling note: the MLX SAME encoders output DiT-space (softnorm) latents
directly — the equivalent of torch's unscaled-latents-then-divide-by-scale
convention with scale 1.0 — so latents are saved as-is and the trainer uses
them directly (no pretransform-scale division at train time).

Tag extraction (ported from underfit's ``extract_tags``): JSON sidecar
(``<stem>.json`` next to the audio, or ``<parent>/json/<stem>.json``) keeps all
non-empty string/number values; else a ``.txt`` sidecar (same dir or
``<parent>/txt/``) whose whole stripped content becomes the ``prompt`` tag.
Embedded ID3/Vorbis tags are a DELIBERATE OMISSION — they need an extra
dependency (``audio_metadata``), and sidecars are the underfit-recommended
path anyway.

Usage:
    python pre_encode_mlx.py --audio-dir ~/clips --output-dir ~/latents \
        --codec same-s [--max-duration 600] [--overwrite]

``--codec same-s`` pairs with the sm-music / sm-sfx DiTs, ``same-l`` with medium.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent  # optimized/mlx (scripts/ is one level down)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import mlx.core as mx  # noqa: E402

from models.defs.audio_encoding import (  # noqa: E402
    SAME_ENCODER_PAD_MODULO,
    SAMPLES_PER_LATENT,
    encode_audio,
)

SAMPLE_RATE = 44100
DS_RATIO = SAMPLES_PER_LATENT  # 4096 samples per latent
CHUNKED_THRESHOLD_SECONDS = 30.0

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
_MIN_AUDIO_SIZE = 4096  # skip files smaller than this (macOS resource forks, corrupt)


# ---------------------------------------------------------------------------
# Audio discovery
# ---------------------------------------------------------------------------


def find_audio_files(root) -> list[Path]:
    """Recursively find audio files under ``root``, sorted by path.

    Skips macOS resource forks (``._*``) and files too small to be real audio.
    """
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("._"):
                continue
            fp = Path(dirpath) / fn
            if fp.suffix.lower() in AUDIO_EXTS:
                try:
                    if fp.stat().st_size < _MIN_AUDIO_SIZE:
                        continue
                except OSError:
                    continue
                files.append(fp)
    files.sort()
    return files


# ---------------------------------------------------------------------------
# Audio loading (torch-free)
# ---------------------------------------------------------------------------


def load_audio(path) -> np.ndarray:
    """Load an audio file → ``(2, T)`` float32 at 44.1 kHz.

    soundfile decodes; mono is duplicated to stereo, >2 channels keep the
    first two; non-44.1 kHz input is linearly resampled at
    align_corners=False positions (same pattern as sa3_gradio.read_audio_any).
    No loudness/peak normalization.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, ch)
    y = data.T
    y = np.stack([y[0], y[0]]) if y.shape[0] == 1 else y[:2]
    if sr != SAMPLE_RATE:
        in_len = y.shape[-1]
        new_len = int(round(in_len * SAMPLE_RATE / sr))
        scale = in_len / new_len
        pos = np.clip((np.arange(new_len) + 0.5) * scale - 0.5, 0, in_len - 1)
        y = np.stack([np.interp(pos, np.arange(in_len), ch) for ch in y])
    return np.ascontiguousarray(y, dtype=np.float32)


def max_samples_for_duration(max_duration: float) -> int:
    """Sample cap for ``max_duration`` seconds, aligned DOWN to ds_ratio (4096)."""
    return (int(max_duration * SAMPLE_RATE) // DS_RATIO) * DS_RATIO


# ---------------------------------------------------------------------------
# Sidecar tag extraction (ported from underfit pre_encode.extract_tags;
# embedded ID3/Vorbis tags deliberately omitted — see module docstring)
# ---------------------------------------------------------------------------


def extract_tags(filepath) -> dict:
    """Extract tag fields for an audio file from sidecar files.

    Priority:
      1. JSON sidecar (``<stem>.json`` same dir, or ``<parent>/json/<stem>.json``)
         — all non-empty string/number values kept (custom keys preserved).
      2. Plain ``.txt`` sidecar (same dir or ``<parent>/txt/``) — whole stripped
         content becomes the ``prompt`` tag.
    Returns ``{}`` when neither sidecar exists (no embedded-tag fallback).
    """
    fp = Path(filepath)
    stem = fp.stem

    # 1. JSON sidecar — same dir, then sibling json/ dir
    sidecar_candidates = [
        fp.with_suffix(".json"),
        fp.parent.parent / "json" / (stem + ".json"),
    ]
    for sidecar in sidecar_candidates:
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    sc = json.load(f)
                tags_out = {
                    k: str(v)
                    for k, v in sc.items()
                    if v and isinstance(v, (str, int, float))
                }
                if tags_out:
                    return tags_out
            except Exception:
                pass

    # 2. Plain .txt sidecar (SA3 convention) — same dir, then sibling txt/ dir.
    txt_candidates = [
        fp.with_suffix(".txt"),
        fp.parent.parent / "txt" / (stem + ".txt"),
    ]
    for txt in txt_candidates:
        if txt.exists():
            try:
                content = txt.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    return {"prompt": content}
            except Exception:
                pass

    return {}


# ---------------------------------------------------------------------------
# Core encode
# ---------------------------------------------------------------------------


def encode_file(
    encoder,
    audio_np: np.ndarray,
    *,
    pad_modulo: int,
    actual_samples: int | None = None,
):
    """Encode one ``(2, T)`` float32 waveform → (latents, padding_mask, seconds).

    Returns:
        latents_np:   ``[D, T_latent]`` float32 (softnorm/DiT space, saved as-is)
        padding_mask: latent-resolution 0/1 int list, length ``T_latent`` —
                      zeros over the alignment-padding tail
        seconds_total: ``round(actual_samples / 44100, 3)``

    ``chunked=True`` is used automatically for clips longer than 30 s.
    """
    audio_np = np.asarray(audio_np)
    if audio_np.ndim != 2 or audio_np.shape[0] != 2:
        raise ValueError(f"Expected audio shaped (2, T), got {audio_np.shape}.")
    if actual_samples is None:
        actual_samples = int(audio_np.shape[-1])
    duration = actual_samples / SAMPLE_RATE
    encoded = encode_audio(
        encoder,
        audio_np[None, ...],
        valid_sample_lengths=[actual_samples],
        pad_modulo=pad_modulo,
        chunked=duration > CHUNKED_THRESHOLD_SECONDS,
    )
    latents_np = np.asarray(encoded.latents.astype(mx.float32))[0]  # [D, T]
    padding_mask = [int(v) for v in np.asarray(encoded.padding_mask)[0]]
    seconds_total = round(actual_samples / SAMPLE_RATE, 3)
    return latents_np, padding_mask, seconds_total


def build_metadata(
    fpath,
    rel,
    out_rel,
    *,
    actual_samples: int,
    latent_shape,
    padding_mask,
    seconds_total,
) -> dict:
    """Underfit-parity metadata dict for one encoded file (tags merged last)."""
    meta = {
        "path": str(fpath),
        "relpath": str(out_rel),
        "src_relpath": str(rel),
        "seconds_total": seconds_total,
        "seconds_start": 0,
        "audio_samples": int(actual_samples),
        "latent_shape": [int(s) for s in latent_shape],
        "padding_mask": padding_mask,
    }
    meta.update(extract_tags(fpath))
    return meta


def run(
    audio_dir,
    output_dir,
    codec: str,
    encoder,
    *,
    max_duration: float = 600.0,
    overwrite: bool = False,
    exclude: set | None = None,
) -> dict:
    """Encode every audio file under ``audio_dir`` into ``output_dir``.

    Returns stats: ``{"encoded", "skipped", "errors", "count"}``.
    """
    if codec not in SAME_ENCODER_PAD_MODULO:
        raise ValueError(f"Unknown codec {codec!r}; expected same-s or same-l.")
    pad_modulo = SAME_ENCODER_PAD_MODULO[codec]

    audio_dir = Path(audio_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    files = find_audio_files(audio_dir)
    if exclude:
        files = [f for f in files if str(f.relative_to(audio_dir)) not in exclude]
    max_samples = max_samples_for_duration(max_duration)

    output_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "codec": codec,
        "sample_rate": SAMPLE_RATE,
        "max_duration": max_duration,
        "max_samples": max_samples,
        "pad_modulo": pad_modulo,
        "audio_dir": str(audio_dir),
        "count": len(files),
    }
    with open(output_dir / "details.json", "w") as f:
        json.dump(details, f, indent=2)

    total = len(files)
    encoded = skipped = errors = 0
    t0 = time.time()
    for index, fpath in enumerate(files, 1):
        rel = fpath.relative_to(audio_dir)
        out_rel = rel.with_suffix(".npy")
        npy_path = output_dir / out_rel
        json_path = npy_path.with_suffix(".json")
        tag = f"[{index}/{total}]"
        if npy_path.exists() and json_path.exists() and not overwrite:
            skipped += 1
            print(f"  {tag} SKIP {out_rel} (exists)")
            continue
        try:
            audio = load_audio(fpath)
            actual_samples = int(audio.shape[-1])
            if actual_samples > max_samples:
                audio = audio[:, :max_samples]
                actual_samples = max_samples

            latents_np, padding_mask, seconds_total = encode_file(
                encoder, audio, pad_modulo=pad_modulo, actual_samples=actual_samples
            )

            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(npy_path), latents_np)
            meta = build_metadata(
                fpath,
                rel,
                out_rel,
                actual_samples=actual_samples,
                latent_shape=latents_np.shape,
                padding_mask=padding_mask,
                seconds_total=seconds_total,
            )
            with open(json_path, "w") as f:
                json.dump(meta, f)

            encoded += 1
            shape_str = "x".join(str(s) for s in latents_np.shape)
            print(f"  {tag} {out_rel}  {seconds_total:.1f}s -> [{shape_str}]")
        except Exception as e:
            errors += 1
            print(f"  {tag} ERROR {rel}: {e}")

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(
        f"\nDone in {mins}m{secs:02d}s: {encoded} encoded, {skipped} skipped, "
        f"{errors} errors  (of {total})"
    )
    print(f"Output: {output_dir}")
    return {"encoded": encoded, "skipped": skipped, "errors": errors, "count": total}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_codec_encoder(codec: str):
    """Load the real SAME encoder for ``codec`` in fp32. Returns (encoder, pad_modulo)."""
    from sa3_mlx import load_encoder  # lazy: pulls weight-download machinery

    return load_encoder(codec, mx.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-encode audio into SAME latents for MLX LoRA finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio-dir", "-i", required=True,
        help="Directory of audio files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output root; each audio file becomes <output-dir>/<relpath>.npy + .json",
    )
    parser.add_argument(
        "--codec", required=True, choices=sorted(SAME_ENCODER_PAD_MODULO),
        help="Autoencoder: same-s pairs with sm-music/sm-sfx, same-l with medium",
    )
    parser.add_argument(
        "--max-duration", type=float, default=600.0,
        help="Crop files longer than N seconds (default: 600; aligned down to 4096 samples)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-encode files whose outputs already exist",
    )
    parser.add_argument(
        "--exclude-file", type=str, default=None,
        help="Text file of relpaths (one per line, relative to --audio-dir) to skip",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    if not audio_dir.is_dir():
        print(f"Error: not a directory: {audio_dir}")
        sys.exit(1)
    files = find_audio_files(audio_dir)
    if not files:
        print(f"No audio files found in {audio_dir}")
        sys.exit(1)

    exclude = set()
    if args.exclude_file and Path(args.exclude_file).is_file():
        exclude = {ln.strip() for ln in Path(args.exclude_file).read_text().splitlines()
                   if ln.strip()}
    n_before = len(files)
    if exclude:
        files = [f for f in files if str(f.relative_to(audio_dir)) not in exclude]
    if not files:
        print(f"All {n_before} file(s) excluded by {args.exclude_file}")
        sys.exit(1)

    max_samples = max_samples_for_duration(args.max_duration)
    import mlx.core as mx
    print(f"Engine: MLX  ·  device: {mx.default_device()}")
    print(f"Codec:  {args.codec} (pad_modulo={SAME_ENCODER_PAD_MODULO[args.codec]}, fp32)")
    print(f"Input:  {audio_dir} ({len(files)} audio files"
          + (f", {n_before - len(files)} excluded)" if exclude else ")"))
    print(f"Output: {Path(args.output_dir).expanduser().resolve()}")
    print(f"Max:    {max_samples / SAMPLE_RATE:.1f}s ({max_samples:,} samples)\n")

    print("Loading encoder...")
    encoder, pad_modulo = load_codec_encoder(args.codec)
    assert pad_modulo == SAME_ENCODER_PAD_MODULO[args.codec]

    run(
        audio_dir,
        args.output_dir,
        args.codec,
        encoder,
        max_duration=args.max_duration,
        overwrite=args.overwrite,
        exclude=exclude,
    )


if __name__ == "__main__":
    main()
