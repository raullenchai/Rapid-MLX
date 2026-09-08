# SPDX-License-Identifier: Apache-2.0
"""Fresh-process report collector for ``rapid-mlx doctor --json``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .cli import report_document
from .env_health import run_all


def _selection(request: dict[str, object], key: str) -> set[str] | None:
    value = request.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{key} selection is invalid")
    return set(value)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        return 2
    request_path, result_path = map(Path, args[:2])
    try:
        keepalive_fd = int(args[2])
        status_fd = int(args[3])
    except ValueError:
        return 2
    staging_path = result_path.with_suffix(".json.tmp")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or set(request) != {"only", "skip", "deep"}:
            raise TypeError("request fields are invalid")
        if not isinstance(request["deep"], bool):
            raise TypeError("deep flag is invalid")
        report = run_all(
            only=_selection(request, "only"),
            skip=_selection(request, "skip"),
            deep=request["deep"],
        )
        message: dict[str, object] = {"ok": True, "report": report_document(report)}
    except Exception as exc:  # noqa: BLE001 - parent renders schema-valid failure
        message = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        with staging_path.open("w", encoding="utf-8") as stream:
            json.dump(message, stream)
            stream.flush()
            os.fsync(stream.fileno())
        staging_path.replace(result_path)
    except OSError:
        for fd in (keepalive_fd, status_fd):
            try:
                os.close(fd)
            except OSError:
                pass
        return 1
    # Retain the process-group leader until the parent has consumed the
    # result. If the parent dies, its pipe closes and this read returns EOF.
    try:
        os.read(keepalive_fd, 1)
    except OSError:
        pass
    finally:
        os.close(keepalive_fd)
        os.close(status_fd)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())
