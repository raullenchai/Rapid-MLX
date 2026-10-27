# SPDX-License-Identifier: Apache-2.0
"""Small generated fixtures for the shared chat document extraction boundary."""

import io
import zipfile

import pytest

from rapid_mlx.chat_runtime.file_extraction import (
    MAX_INPUT_BYTES,
    MAX_TEXT_BYTES,
    DocumentError,
    extract_document,
)

WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION = "http://schemas.openxmlformats.org/presentationml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def archive(parts):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as output:
        for name, value in parts.items():
            output.writestr(name, value)
    return stream.getvalue()


def workbook(rows, shared=""):
    parts = {
        "xl/workbook.xml": (
            f'<workbook xmlns="{SHEET}" xmlns:r="{REL}"><sheets>'
            '<sheet name="Budget" r:id="sheet1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<Relationships><Relationship Id="sheet1" '
            'Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
        ),
        "xl/worksheets/sheet1.xml": f'<worksheet xmlns="{SHEET}"><sheetData>{rows}</sheetData></worksheet>',
    }
    if shared:
        parts["xl/sharedStrings.xml"] = f'<sst xmlns="{SHEET}">{shared}</sst>'
    return archive(parts)


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("code.py", b"def target():\n    return 1\n", "def target():"),
        ("config.json", b'{"value": 3}', '"value": 3'),
        ("config.yaml", b"value: 3\n", "value: 3"),
        ("UNKNOWN", b"plain text", "plain text"),
        ("names.csv", b'name,total\n"widget, large",25\n', "widget, large\t25"),
        ("names.tsv", b"name\ttotal\nwidget\t25", "widget\t25"),
        (
            "page.html",
            b"<h1>Report</h1><script>secret script</script><p>A &amp; B</p>",
            "A & B",
        ),
    ],
)
def test_text_formats(name, data, expected):
    result = extract_document(data, name)
    assert expected in result
    assert "secret script" not in result


def test_csv_accepts_large_field_and_converts_parser_error(monkeypatch):
    import csv

    value = "x" * 150000
    assert extract_document(value.encode(), "large.csv") == value

    def fail(*args, **kwargs):
        raise csv.Error("private content")

    monkeypatch.setattr(csv, "reader", fail)
    with pytest.raises(DocumentError) as error:
        extract_document(b"x", "bad.csv")
    assert error.value.code == "document_invalid"


def test_docx():
    data = archive(
        {
            "word/document.xml": f'<w:document xmlns:w="{WORD}"><w:body><w:p><w:r><w:t>Quarterly report</w:t></w:r></w:p></w:body></w:document>'
        }
    )
    assert extract_document(data, "report.docx") == "Quarterly report"


def test_xlsx_sparse_columns_shared_strings_and_formula():
    data = workbook(
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1"><v>25</v></c></row>'
        '<row r="2"><c r="A2" t="inlineStr"><is><t>Other</t></is></c><c r="C2"><v>75</v></c></row>'
        '<row r="3"><c r="C3"><f>SUM(C1:C2)</f><v>100</v></c></row>'
        '<row r="4"><c r="C4"><f>SUM(C1:C2)</f></c></row>',
        "<si><t>Widget</t></si>",
    )
    result = extract_document(data, "budget.xlsx")
    assert "Sheet: Budget" in result
    assert "1\tA=Widget\tC=25" in result
    assert "2\tA=Other\tC=75" in result
    assert "3\tC=100" in result
    assert "formula has no cached value" in result


def test_pptx_slide_order():
    data = archive(
        {
            "ppt/presentation.xml": f'<p:presentation xmlns:p="{PRESENTATION}" xmlns:r="{REL}"><p:sldIdLst><p:sldId r:id="second"/><p:sldId r:id="first"/></p:sldIdLst></p:presentation>',
            "ppt/_rels/presentation.xml.rels": '<Relationships><Relationship Id="first" Type="slide" Target="slides/slide1.xml"/><Relationship Id="second" Type="slide" Target="slides/slide2.xml"/></Relationships>',
            "ppt/slides/slide1.xml": f'<a:p xmlns:a="{DRAWING}"><a:r><a:t>End</a:t></a:r></a:p>',
            "ppt/slides/slide2.xml": f'<a:p xmlns:a="{DRAWING}"><a:r><a:t>Start</a:t></a:r></a:p>',
        }
    )
    assert extract_document(data, "slides.pptx") == "Slide 1\nStart\n\nSlide 2\nEnd"


def pdf_fixture():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Quarterly revenue is 100 dollars.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def test_pdf():
    assert "Page 1\nQuarterly revenue is 100 dollars." in extract_document(
        pdf_fixture(), "report.pdf"
    )


def test_caps_and_table_markers():
    result = extract_document(("é" * MAX_TEXT_BYTES).encode(), "large.txt")
    assert len(result.encode()) <= MAX_TEXT_BYTES
    assert "Extraction truncated" in result
    result = extract_document(b"name,total\n" + b"widget,1\n" * 1001, "large.csv")
    assert "Table truncated" in result
    data = workbook(
        "".join(
            f'<row r="{i}"><c r="A{i}"><v>{i}</v></c></row>' for i in range(1, 1002)
        )
    )
    result = extract_document(data, "large.xlsx")
    assert "1000\tA=1000" in result
    assert "1001\tA=1001" not in result
    assert "Sheet truncated" in result
    with pytest.raises(DocumentError, match="20 MiB") as error:
        extract_document(b"x" * (MAX_INPUT_BYTES + 1), "large.txt")
    assert error.value.code == "file_too_large"


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("image.png", b"PNG", "image_unsupported"),
        ("binary.txt", b"a\0b", "binary_unsupported"),
        ("binary.dat", b"\xff\xfe", "binary_unsupported"),
        ("broken.docx", b"invalid", "document_invalid"),
        ("broken.pdf", b"invalid", "document_invalid"),
    ],
)
def test_typed_errors(name, content, code):
    with pytest.raises(DocumentError) as error:
        extract_document(content, name)
    assert error.value.code == code


def test_ooxml_entities_rejected():
    data = archive(
        {
            "word/document.xml": f'<!DOCTYPE foo [<!ENTITY x "secret">]><w:document xmlns:w="{WORD}"><w:p><w:t>&x;</w:t></w:p></w:document>'
        }
    )
    with pytest.raises(DocumentError) as error:
        extract_document(data, "unsafe.docx")
    assert error.value.code == "document_unsupported"
