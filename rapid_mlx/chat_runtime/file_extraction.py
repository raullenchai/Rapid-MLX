# SPDX-License-Identifier: Apache-2.0
"""Shared, bounded text extraction for workspace reads and explicit attachments.

No filesystem authority is granted here: callers supply bytes they are permitted
to read. Office files are parsed as bounded XML, with no macros or external links.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path

from rapid_mlx.local_search import extraction

MAX_INPUT_BYTES = extraction.MAX_BYTES
# The CSV parser's field limit is process-global; bound it to the input cap.
csv.field_size_limit(MAX_INPUT_BYTES)
MAX_TEXT_BYTES = 1024 * 1024
MAX_TABLE_ROWS = 1000
MAX_TABLE_CELLS = extraction.XLSX_MAX_CELLS_PER_SHEET
_TRUNCATED = "\n[Extraction truncated at 1 MiB; split the source into smaller files.]"


class DocumentError(ValueError):
    """A stable machine-readable extraction failure, safe to return to the model."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _bounded(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) <= MAX_TEXT_BYTES:
        return text
    room = MAX_TEXT_BYTES - len(_TRUNCATED.encode("utf-8"))
    return data[:room].decode("utf-8", errors="ignore") + _TRUNCATED


def _cell(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _csv_text(data: bytes, suffix: str) -> str:
    text = data.decode("utf-8-sig")
    if any(ord(c) < 32 and c not in "\n\r\t\f" for c in text):
        raise DocumentError("binary_unsupported", "Binary content is not supported.")
    rows = []
    cells = 0
    for index, row in enumerate(
        csv.reader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ","), 1
    ):
        if index > MAX_TABLE_ROWS or cells + len(row) > MAX_TABLE_CELLS:
            rows.append(
                "[Table truncated at 1000 rows or 10000 cells; "
                "split the source to read the remaining rows.]"
            )
            break
        cells += len(row)
        rows.append("\t".join(_cell(value) for value in row))
    return "\n".join(rows)


def _xlsx_text(data: bytes) -> str:
    # The indexing extractor intentionally flattens cells. Chat needs the sheet
    # names and cell coordinates to retain sparse columns and row relationships.
    ns = extraction._SHEET
    parts = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml = extraction._OfficeXML(archive, MAX_INPUT_BYTES)
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = [
                "".join(t.text or "" for t in item.iter(ns + "t"))
                for item in xml.read("xl/sharedStrings.xml").iter(ns + "si")
            ]
        workbook = "xl/workbook.xml"
        relationships = xml.relationships(workbook)
        sheets = list(xml.read(workbook).iter(ns + "sheet"))
        if len(sheets) > extraction.MAX_DOCUMENT_PAGES:
            raise extraction.SkippedError("expanded size cap")
        for sheet in sheets:
            parts.append(f"Sheet: {_cell(sheet.attrib.get('name', 'Unnamed'))}")
            parts.append("Row\tCells (column=value)")
            target = relationships[sheet.attrib[extraction._REL_ID]][1]
            count = 0
            for row_index, row in enumerate(xml.read(target).iter(ns + "row"), 1):
                cells = list(row.iter(ns + "c"))
                if row_index > MAX_TABLE_ROWS or count + len(cells) > MAX_TABLE_CELLS:
                    parts.append(
                        "[Sheet truncated at 1000 rows or 10000 cells; "
                        "split the source to read the remaining rows.]"
                    )
                    break
                count += len(cells)
                values = []
                for column_index, cell in enumerate(cells, 1):
                    value = cell.findtext(ns + "v", "")
                    kind = cell.attrib.get("t")
                    if kind == "s":
                        value = shared[int(value)] if value else ""
                    elif kind == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(ns + "t"))
                    elif kind == "b":
                        value = "TRUE" if value == "1" else "FALSE"
                    if cell.find(ns + "f") is not None and not value:
                        value = "[formula has no cached value]"
                    coordinate = cell.attrib.get("r", "")
                    match = re.fullmatch(r"([A-Z]+)[0-9]+", coordinate)
                    column = match[1] if match else str(column_index)
                    values.append(f"{column}={_cell(value)}")
                parts.append(
                    f"{row.attrib.get('r', str(row_index))}\t" + "\t".join(values)
                )
    return "\n".join(parts)


def extract_document(data: bytes, name: str) -> str:
    """Extract UTF-8 text; retain page/slide labels and spreadsheet coordinates.

    Input is capped at 20 MiB; output at 1 MiB, with an explicit marker. Tables
    retain at most 1000 rows/10000 cells per sheet. Formulas are not evaluated.
    Images are deferred until model capability and persisted image parts agree.
    """
    if len(data) > MAX_INPUT_BYTES:
        raise DocumentError("file_too_large", "Files are limited to 20 MiB.")
    path = Path(name)
    suffix = path.suffix.lower()
    if suffix in extraction.IMAGE_SUFFIXES:
        raise DocumentError(
            "image_unsupported",
            "Image input is not supported by this model/tool pipeline yet; "
            "provide a text description or attach extracted text.",
        )
    try:
        if suffix == ".xlsx":
            text = _xlsx_text(data)
        elif suffix in {".csv", ".tsv"}:
            text = _csv_text(data, suffix)
        elif suffix in {".pdf", ".docx", ".pptx", ".html", ".htm"}:
            units = extraction.extract(path, data)
            label = "Slide" if suffix == ".pptx" else "Page"
            text = "\n\n".join(
                f"{label} {unit.page}\n{unit.text}" if unit.page else unit.text
                for unit in units
            )
        else:
            # Plain code/text need not have a known extension. Never decode a
            # binary with replacement characters and present it as document text.
            text = data.decode("utf-8-sig")
            if any(ord(c) < 32 and c not in "\n\r\t\f" for c in text):
                raise DocumentError(
                    "binary_unsupported", "Binary content is not supported."
                )
        return _bounded(text)
    except DocumentError:
        raise
    except UnicodeDecodeError:
        raise DocumentError(
            "binary_unsupported", "The file is not UTF-8 text."
        ) from None
    except extraction.SkippedError as error:
        reason = str(error)
        code = "file_too_large" if "size cap" in reason else "document_unsupported"
        raise DocumentError(code, extraction.skip_reason(reason)) from None
    except Exception:
        # Parser exceptions can contain document contents; expose a stable error.
        raise DocumentError(
            "document_invalid", "The document could not be extracted."
        ) from None


if __name__ == "__main__":
    import json
    import sys

    try:
        result: dict[str, str | dict[str, str]] = {
            "text": extract_document(
                sys.stdin.buffer.read(MAX_INPUT_BYTES + 1), sys.argv[1]
            )
        }
    except DocumentError as error:
        result = {"error": {"code": error.code, "message": error.message}}
    print(json.dumps(result, ensure_ascii=False))
