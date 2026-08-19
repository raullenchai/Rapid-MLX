"""Pin docs/guides/tool-calling.md's Supported Parsers table to the registry.

The table is hand-maintained and has drifted before (the 2026-08-18 doc
audit found ~12 registered parsers missing). This pins it both ways: every
distinct registered parser class has exactly one row, and every row's
primary name is a registered name backed by a distinct class.
"""

import re
from pathlib import Path

from vllm_mlx.tool_parsers import ToolParserManager

DOC = Path(__file__).resolve().parent.parent / "docs" / "guides" / "tool-calling.md"


def _doc_primary_names() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    section = re.search(r"## Supported Parsers(.*?)\n## ", text, re.S)
    assert section, "Supported Parsers section not found in tool-calling.md"
    primaries = re.findall(r"^\|\s*`([a-z0-9_.\-]+)`", section.group(1), re.M)
    return primaries


def _registry_classes() -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {}
    for name, cls in ToolParserManager.tool_parsers.items():
        classes.setdefault(cls.__name__, []).append(name)
    return classes


def test_table_covers_every_registered_parser_class() -> None:
    classes = _registry_classes()
    doc_primaries = _doc_primary_names()

    assert len(doc_primaries) == len(set(doc_primaries)), (
        f"duplicate rows in the doc table: {doc_primaries}"
    )
    assert len(doc_primaries) == len(classes), (
        f"doc table has {len(doc_primaries)} rows but the registry has "
        f"{len(classes)} distinct parser classes"
    )

    registered_names = set(ToolParserManager.tool_parsers)
    unknown = [p for p in doc_primaries if p not in registered_names]
    assert not unknown, f"doc rows not in the registry: {unknown}"

    # Each row must map to a DIFFERENT class — a table that lists two
    # aliases of one parser as separate rows while omitting another class
    # would still pass the count check without this.
    row_classes = {ToolParserManager.tool_parsers[p].__name__ for p in doc_primaries}
    assert len(row_classes) == len(classes), (
        f"doc rows map to {len(row_classes)} classes, registry has "
        f"{len(classes)}: missing "
        f"{sorted(set(classes) - row_classes)}"
    )
