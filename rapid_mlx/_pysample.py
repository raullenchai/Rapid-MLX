"""In-process sampling profiler (debug aid, env-gated).

``RAPID_PYSAMPLE=/path/out.txt`` starts a daemon thread that samples
``sys._current_frames()`` every ~5ms and aggregates innermost stack
signatures per thread. The report is rewritten every 10s and at exit.

Motivation: py-spy needs root on macOS; this answers "which Python code
runs on the event loop while the decode thread sits in next()" without
attaching a debugger. Overhead ~1-2%.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from collections import Counter

_DEPTH = 6
_INTERVAL_S = 0.005
_REPORT_EVERY_S = 10.0


def _signature(frame) -> str:
    parts = []
    f = frame
    while f is not None and len(parts) < _DEPTH:
        code = f.f_code
        parts.append(f"{os.path.basename(code.co_filename)}:{code.co_name}")
        f = f.f_back
    return " < ".join(parts)


def install() -> bool:
    out_path = os.environ.get("RAPID_PYSAMPLE")
    if not out_path:
        return False

    counters: dict[str, Counter] = {}
    names: dict[int, str] = {}
    self_ident: list[int] = []
    # The atexit report races the daemon sampler over ``counters``
    # (RuntimeError: dict changed size during iteration — codex r1 on
    # #1878): snapshot under a lock shared with the sampling loop.
    lock = threading.Lock()
    _warned: list[int] = []

    def _report():
        # The whole snapshot + write runs under the lock: the periodic
        # report and the atexit report target the same file, and
        # interleaved writers would corrupt it (codex r2 NIT).
        with lock:
            snapshot = {tname: ctr.copy() for tname, ctr in counters.items()}
            try:
                with open(out_path, "w") as fh:
                    for tname, ctr in sorted(snapshot.items()):
                        total = sum(ctr.values())
                        fh.write(f"== thread {tname} samples={total}\n")
                        for sig, n in ctr.most_common(30):
                            fh.write(f"{n:8d} {100.0 * n / total:5.1f}% {sig}\n")
                        fh.write("\n")
            except OSError as exc:
                # Warn once — silently dropping every report would leave
                # an operator believing profiling is active while an
                # unwritable RAPID_PYSAMPLE path produces nothing
                # (codex r4 NIT).
                if not _warned:
                    _warned.append(1)
                    print(
                        f"[pysample] cannot write report to {out_path}: {exc}",
                        file=sys.stderr,
                    )

    def _loop():
        self_ident.append(threading.get_ident())
        last_report = time.monotonic()
        while True:
            time.sleep(_INTERVAL_S)
            for t in threading.enumerate():
                names[t.ident] = t.name
            frames = sys._current_frames()
            with lock:
                for ident, frame in frames.items():
                    if ident in self_ident:
                        continue
                    tname = names.get(ident, str(ident))
                    counters.setdefault(tname, Counter())[_signature(frame)] += 1
            now = time.monotonic()
            if now - last_report >= _REPORT_EVERY_S:
                last_report = now
                _report()

    threading.Thread(target=_loop, name="rapid-pysample", daemon=True).start()
    atexit.register(_report)
    return True
