#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``vllm_mlx/model_sizes.json`` — the checked-in download-size
manifest that powers the ``Size`` column in ``rapid-mlx models`` and the
size line in ``rapid-mlx info <alias>``.

Why a manifest instead of a live lookup?
    ``rapid-mlx models`` lists ~165 aliases from the pure declarative
    registry with *zero* network calls (it is instant and works offline).
    Sizing them live would mean ~180 HuggingFace ``model_info`` round-trips
    per invocation — slow, rate-limited, and flaky. Instead we resolve the
    weight+tokenizer footprint of every distinct ``hf_path`` ONCE, here, and
    check the result in. The runtime just reads the JSON.

Keyed by ``hf_path`` (not alias) so aliases that share a repo dedupe and the
map survives alias renames. Sizes are the same weight+tokenizer footprint the
download-confirmation gate reports (``estimate_repo_size_bytes``), so the
listing and the ``[Y/n]`` prompt agree.

Every alias ``hf_path`` (text) and audio ``hf_id`` gets an entry. Repos whose
size could not be resolved (gated / 404 / HF outage) are recorded as ``null``
so the key is still present — the coverage test stays green and the runtime
renders ``—`` for them.

Usage:
    python3.12 scripts/gen_model_sizes.py            # regenerate in place
    python3.12 scripts/gen_model_sizes.py --check     # fail if stale (CI)

Run this whenever ``aliases.json`` or ``audio/aliases.json`` change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "vllm_mlx" / "model_sizes.json"

# Running ``python3.12 scripts/gen_model_sizes.py`` puts ``scripts/`` on
# sys.path[0], so a bare ``import vllm_mlx`` would resolve to an OLDER
# pip-installed copy in site-packages (silently sizing a stale alias set and
# dropping newly-added models). Force the local checkout to win.
sys.path.insert(0, str(REPO_ROOT))

# Parallelism for the first sizing pass. Each call is capped at 5s by
# ``estimate_repo_size_bytes``; a modest pool keeps the whole sweep well
# under a minute without hammering the Hub.
_MAX_WORKERS = 4

# Every alias points at a repo with real weights, so the smallest legitimate
# footprint (whisper-tiny) is still tens of MiB. A result under this floor
# means the heavy ``?blobs=true`` response came back WITHOUT per-file blob
# sizes (we then counted only ``config.json`` — a few hundred bytes). This is
# a same-process artefact of the concurrent bulk sweep: a FRESH interpreter
# always resolves these repos correctly, so the repair pass re-probes each
# suspect in a subprocess (see ``_build`` / ``_size_via_subprocess``).
_MIN_PLAUSIBLE_BYTES = 1 * 1024 * 1024  # 1 MiB
_REPAIR_ATTEMPTS = 2


def _collect_hf_paths() -> list[str]:
    """Every distinct repo id referenced by the text and audio registries."""
    from vllm_mlx.model_aliases import list_profiles

    paths: set[str] = {p.hf_path for p in list_profiles().values()}

    # Audio registry is optional (import may fail on a text-only checkout);
    # a missing audio section must not abort the text manifest.
    try:
        from vllm_mlx.audio.registry import list_audio_aliases

        paths.update(e.hf_id for e in list_audio_aliases())
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"  warn: audio registry unavailable ({exc}); skipping", file=sys.stderr)

    return sorted(paths)


def _size_one(repo_id: str) -> tuple[str, int | None]:
    from vllm_mlx._download_gate import estimate_repo_size_bytes

    return repo_id, estimate_repo_size_bytes(repo_id)


def _looks_partial(size: int | None) -> bool:
    """True if ``size`` is None or so small the blob metadata must be missing."""
    return size is None or size < _MIN_PLAUSIBLE_BYTES


def _size_via_subprocess(repo_id: str) -> int | None:
    """Re-probe ``repo_id`` in a FRESH interpreter.

    The bulk concurrent pass occasionally records a partial size (blob
    metadata missing from the ``?blobs=true`` response) — a same-process
    artefact that a clean interpreter never reproduces. Shelling out
    guarantees a fresh HTTP session.
    """
    code = (
        "from vllm_mlx._download_gate import estimate_repo_size_bytes;"
        f"v=estimate_repo_size_bytes({repo_id!r});"
        "print(v if v is not None else '')"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO_ROOT,
        ).stdout.strip()
        return int(out) if out else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _build() -> dict[str, int | None]:
    repos = _collect_hf_paths()
    print(f"Sizing {len(repos)} repos via HuggingFace…", file=sys.stderr)
    sizes: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for repo_id, size in pool.map(_size_one, repos):
            sizes[repo_id] = size
            marker = "—" if size is None else f"{size / 1024**3:.1f} GiB"
            print(f"  {marker:>12}  {repo_id}", file=sys.stderr)

    # Repair pass: re-probe every implausibly-small result in a fresh
    # subprocess. A repo that stays partial across every attempt is recorded
    # as null (genuinely unresolvable — e.g. a 404 or gated repo) rather than
    # a misleading few-hundred-byte value the runtime would render as "0 MiB".
    suspects = [r for r, s in sizes.items() if _looks_partial(s)]
    if suspects:
        print(
            f"Repairing {len(suspects)} suspect(s) via fresh subprocesses…",
            file=sys.stderr,
        )
    for repo_id in suspects:
        best: int | None = None
        for _ in range(_REPAIR_ATTEMPTS):
            size = _size_via_subprocess(repo_id)
            if size is not None and (best is None or size > best):
                best = size
            if not _looks_partial(best):
                break
        sizes[repo_id] = None if _looks_partial(best) else best
        marker = (
            "—" if sizes[repo_id] is None else f"{sizes[repo_id] / 1024**3:.1f} GiB"
        )
        print(f"  repaired {marker:>12}  {repo_id}", file=sys.stderr)

    return dict(sorted(sizes.items()))


def _render(sizes: dict[str, int | None]) -> str:
    payload = {
        "__doc__": (
            "Download footprint (weight+tokenizer bytes) per hf_path. "
            "Generated by scripts/gen_model_sizes.py — do not hand-edit. "
            "null = size could not be resolved at generation time."
        ),
        "sizes": sizes,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the on-disk manifest differs from a fresh build",
    )
    args = ap.parse_args()

    rendered = _render(_build())

    if args.check:
        current = MANIFEST_PATH.read_text() if MANIFEST_PATH.exists() else ""
        if current != rendered:
            print(
                "model_sizes.json is stale — run "
                "`python3.12 scripts/gen_model_sizes.py`",
                file=sys.stderr,
            )
            return 1
        print("model_sizes.json is up to date.", file=sys.stderr)
        return 0

    MANIFEST_PATH.write_text(rendered)
    print(f"Wrote {MANIFEST_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
