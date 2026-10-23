"""CLI phase markers for long-running pipeline stages.

Writes brief ``[phase] ...`` / ``[phase] done in X.Ys`` lines to ``stderr``
so the user sees progress through silent stages (Gemma encode, DiT load,
VAE decode) without polluting ``stdout`` for callers that pipe it.

Gated by a single ``verbose`` flag plumbed from CLI ``--quiet``.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def phase(label: str, *, verbose: bool = True) -> Iterator[None]:
    """Print ``[label]...`` on enter and ``[label] done in X.Ys`` on exit.

    No-op when ``verbose=False``. Output goes to ``stderr`` so stdout
    stays clean for callers that pipe pipeline output.
    """
    if not verbose:
        yield
        return

    print(f"[{label}] ...", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[{label}] done in {dt:.1f}s", file=sys.stderr, flush=True)
