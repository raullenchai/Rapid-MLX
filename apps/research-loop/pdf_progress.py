"""Bounded parser child with page progress, run via local-search's RSS watchdog."""

import io
import json
import math
import resource
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rapid_mlx.local_search.extraction import MAX_BYTES, Unit
from rapid_mlx.local_search.parsing import PARSER_MEMORY_BYTES, parser_timeout_seconds


def main():
    if sys.platform != "darwin":
        resource.setrlimit(
            resource.RLIMIT_AS, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES)
        )
    try:
        from pypdf import PdfReader

        data = sys.stdin.buffer.read(MAX_BYTES + 1)
        timeout = math.ceil(parser_timeout_seconds(len(data)))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        if len(data) > MAX_BYTES:
            raise ValueError("size cap")
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("password protected")
        units, total = [], 0
        progress = Path(sys.argv[1])
        for i, page in enumerate(reader.pages, 1):
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps([i, len(reader.pages)]))
            temporary.replace(progress)
            text = page.extract_text() or ""
            total += len(text.encode("utf-8"))
            if total > MAX_BYTES:
                raise ValueError("expanded size cap")
            if text.strip():
                units.append(Unit(text=text, page=i))
        if sum(c.isalnum() for u in units for c in u.text) < 3:
            raise ValueError("No readable text")
        result = dict(units=[asdict(u) for u in units])
    except Exception as exc:
        result = dict(error=str(exc))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
