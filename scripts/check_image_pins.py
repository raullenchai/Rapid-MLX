"""Read-only upstream resolution check, independent of mirror drift findings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if sys.path[:1] != [str(ROOT)]:
    sys.path.insert(0, str(ROOT))


def image_pins() -> list[tuple[str, str]]:
    """Include auxiliary assets even when they have no catalog alias."""
    from rapid_mlx._download_gate import (
        IMAGE_MODEL_REVISIONS,
        IMAGE_MODEL_RUNTIME_ASSETS,
    )

    pins = set(IMAGE_MODEL_REVISIONS.items())
    for assets in IMAGE_MODEL_RUNTIME_ASSETS.values():
        pins.update((repo, revision) for repo, revision, _ in assets)
    return sorted(pins)


def main() -> int:
    from huggingface_hub import model_info
    from huggingface_hub.errors import HfHubHTTPError, RevisionNotFoundError

    from rapid_mlx._download_gate import _HF_RESOLVE_TIMEOUT_SECONDS, call_with_deadline

    failed = False
    for repo, revision in image_pins():
        try:
            info = call_with_deadline(
                model_info, _HF_RESOLVE_TIMEOUT_SECONDS, repo, revision=revision
            )
            if info.sha != revision:
                raise ValueError(f"resolved to {info.sha}")
        except RevisionNotFoundError:
            print(f"DANGLING IMAGE PIN: {repo}@{revision}", file=sys.stderr)
            failed = True
        except (HfHubHTTPError, OSError, TimeoutError, ValueError) as error:
            print(
                f"IMAGE PIN CHECK FAILED: {repo}@{revision}: {error}", file=sys.stderr
            )
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
