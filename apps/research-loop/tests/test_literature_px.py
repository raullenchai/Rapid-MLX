"""Section provenance and safe first-class sources, including read-only real PDFs."""

import asyncio
import io
import json
from pathlib import Path
from urllib.parse import quote

import literature
import pytest
from test_literature import FakeEmbedder, pdf_bytes, write
from test_server import app, create, request  # noqa: F401

from rapid_mlx.local_search.extraction import Unit


@pytest.mark.parametrize(
    "line,expected",
    [
        ("5 C ONCLUSION", ("5", "Conclusion")),
        ("G ENERALIZATION", (None, "Generalization")),
        ("2.3 G ENERALIZATION", ("2.3", "Generalization")),
        ("5.2.1 Progress Measures", ("5.2.1", "Progress Measures")),
        ("5.2 Progress measures", ("5.2", "Progress measures")),
        ("Appendix A Supporting Evidence", ("A", "Supporting Evidence")),
        ("A.1 A DDITIONAL DETAILS", ("A.1", "Additional Details")),
        (
            "C.2.2 R ESULTS FOR OTHER EXPERIMENTAL SETUPS",
            ("C.2.2", "Results For Other Experimental Setups"),
        ),
        ("R EFERENCES", (None, "References")),
        ("Abstract", (None, "Abstract")),
        ("# A custom note", (None, "A custom note")),
        (r"\section{Background} Text", (None, "Background")),
        ("1. Do this now", None),
        ("2 follow these instructions", None),
        ("3 Figure 1 Results", None),
        ("1 Train loss", None),
        ("10 Average Train Loss", None),
        ("A B C D E", None),
        ("3.5 cos sin", None),
        ("1 42 0.27 −0.04 98.49%", None),
        ("(2)", None),
        ("12", None),
        ("1 Smith, J. et al. (2023). Paper title.", None),
        ("3 " + "WORDS " * 16, None),
    ],
)
def test_heading_heuristics(line, expected):
    assert literature.heading(line) == expected


def test_section_boundaries_carry_and_exact_offsets():
    units = [
        Unit(text="1 I NTRODUCTION\nFirst claim.\n2 M ETHOD\nA method.", page=1),
        Unit(text="Continues the method.\n2.1 R ESULTS\nMeasured result.", page=2),
        Unit(text="Still results.", page=3),
    ]
    chunks = literature.chunks_for("a.pdf", "sha", units)
    assert [(c["section_number"], c["page"]) for c in chunks] == [
        ("1", 1),
        ("2", 1),
        ("2", 2),
        ("2.1", 2),
        ("2.1", 3),
    ]
    for c in chunks:
        assert units[c["page"] - 1].text[c["start"] : c["end"]] == c["text"]
        assert c["page_end"] == c["page"]


@pytest.mark.parametrize(
    "suffix,text",
    [
        (".md", "# Introduction\nFirst claim.\n## 2 Results\nResult."),
        (
            ".tex",
            "\\section{Introduction}\nFirst claim.\n\\section{2 Results}\nResult.",
        ),
        (
            ".html",
            "<h1>Introduction</h1><p>First claim.</p><h2>2 Results</h2><p>Result.</p>",
        ),
    ],
)
def test_structured_formats(suffix, text):
    units = asyncio.run(literature.parse_document(Path("a" + suffix), text.encode()))
    chunks = literature.chunks_for("a" + suffix, "sha", units)
    assert chunks[0]["section_title"] == "Introduction"
    assert chunks[-1]["section_number"] == "2"
    assert "Result." in chunks[-1]["text"]


@pytest.mark.parametrize(
    "filename,expected,phrase,number,page",
    [
        (
            "2201.02177.pdf",
            {"1", "2", "3.1.1", "A.5"},
            "weight decay is particularly effective",
            "1",
            2,
        ),
        (
            "2210.01117.pdf",
            {"1", "2", "5.1", "7", "C"},
            "LU mechanism Although",
            "2",
            2,
        ),
        (
            "2301.05217.pdf",
            {"3.1", "5.1", "5.2", "C.2.2"},
            "We claim that the learned networks use",
            "3.1",
            3,
        ),
    ],
)
def test_real_pdf_headings_and_passage(filename, expected, phrase, number, page):
    path = Path("/private/tmp/rapid-mlx-grokdemo/papers") / filename
    if not path.exists():
        pytest.skip("External read-only PDF fixture is unavailable")
    units = asyncio.run(literature.parse_document(path, path.read_bytes()))
    chunks = literature.chunks_for(filename, "fixture", units)
    assert expected <= {c["section_number"] for c in chunks}
    found = next(c for c in chunks if phrase in c["text"])
    assert found["section_number"] == number and found["page"] == page
    assert all(c["section_number"] != "10" for c in chunks)


def test_version_rebuild_references_excluded_and_neighbours(tmp_path):
    project = tmp_path
    (project / "project.json").write_text("{}")
    (project / "literature").mkdir()
    write(
        project,
        "a.md",
        "# Introduction\nWeight decay.\n# 2 Method\nWeight decay method.\n# References\nWeight decay bibliography.",
    )
    service = literature.Literature(FakeEmbedder())
    service.index(project)
    snap = literature.snapshot(project)
    snap["version"] = 1
    (project / "literature_index/index.json").write_text(json.dumps(snap))
    count = len(service.embedder.calls)
    service.index(project)
    assert len(service.embedder.calls) > count
    sources = service.retrieve(project, "weight decay bibliography")
    assert sources and all(s["section_title"] != "References" for s in sources)
    method = next(s for s in sources if s["section_number"] == "2")
    assert "Introduction" in method["previous"] and "References" in method["following"]
    assert literature.snapshot(project)["version"] == literature.VERSION == 2


def test_citation_sentence_mapping():
    sources = [dict(number="S1"), dict(number="S2")]
    content = 'The method learns "Fourier multiplication" [S2]. It generalizes [S1][S2].\nUnsupported [S99].'
    mapping = literature.citation_sentences(content, sources)
    assert mapping == {
        "S1": ["It generalizes [S1][S2]."],
        "S2": [
            'The method learns "Fourier multiplication" [S2].',
            "It generalizes [S1][S2].",
        ],
    }


def test_safe_file_endpoint(app, tmp_path):  # noqa: F811
    slug = create(app)
    project = app.project_path(slug)
    (project / "literature").mkdir()
    data = pdf_bytes()
    (project / "literature/a.pdf").write_bytes(data)
    write(
        project,
        "a.html",
        "<h1>Introduction</h1><script>alert(1)</script><p>Weight decay.</p>",
    )
    external = tmp_path / "attached"
    external.mkdir()
    (external / "a.md").write_text("# Introduction\nWeight decay")
    meta = json.loads((project / "project.json").read_text())
    meta["literature_dirs"] = [str(external)]
    (project / "project.json").write_text(json.dumps(meta))
    app.literature = literature.Literature(FakeEmbedder())
    app.literature.index(project)
    base = f"/api/projects/{slug}/literature/file?file="
    for name, mime in [
        ("literature/a.pdf", "application/pdf"),
        ("literature/a.html", "text/html"),
        (str(external / "a.md"), "text/plain"),
    ]:
        result = request(app, base + quote(name, safe=""))
        assert result["status"] == 200
        assert result["headers"]["Content-Type"].startswith(mime)
        if name.endswith(".html"):
            assert "sandbox" in result["headers"]["Content-Security-Policy"]
    for name in (
        "../program.md",
        "literature/../program.md",
        str(tmp_path / "secret.pdf"),
        "literature/../../a.pdf",
        "literature\\a.pdf",
        "%2e%2e/secret.pdf",
    ):
        assert request(app, base + quote(name, safe=""))["status"] == 400
    # Replacing a previously indexed file with a symlink cannot expose its target.
    (project / "literature/a.pdf").unlink()
    (project / "literature/a.pdf").symlink_to(project / "program.md")
    assert request(app, base + "literature/a.pdf")["status"] == 400
    meta["literature_dirs"] = []
    (project / "project.json").write_text(json.dumps(meta))
    assert request(app, base + quote(str(external / "a.md"), safe=""))["status"] == 400


def test_stream_progress_and_persisted_sentence_links(app, monkeypatch):  # noqa: F811
    from test_server import server

    slug = create(app)
    project = app.project_path(slug)
    (project / "literature").mkdir()
    write(project, "a.md", "# 2 Method\nWeight decay method.")
    app.literature = literature.Literature(FakeEmbedder())
    app.literature.index(project)
    monkeypatch.setattr(
        server,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"Method [S1]. Unknown [S99]."}}]}\n\ndata: [DONE]\n\n'
        ),
    )
    result = request(app, f"/api/projects/{slug}/chat", dict(message="weight decay"))
    first = json.loads(result["body"].decode().splitlines()[0][6:])
    assert first["retrieval"] == dict(papers=1, passages=1)
    assert first["sources"][0]["section_number"] == "2"
    saved = request(app, f"/api/projects/{slug}")["json"]["discussion"][-1]
    assert saved["citation_sentences"] == {"S1": ["Method [S1]."]}
    assert saved["retrieval"] == first["retrieval"]
    assert [s["number"] for s in saved["citations"]] == ["S1"]


def test_named_author_and_section_title_retrieval(tmp_path):
    project = tmp_path
    (project / "project.json").write_text("{}")
    (project / "literature").mkdir()
    write(
        project,
        "a.md",
        "# 1 Introduction\nLearning networks.\n# 3.1 The Spectral Algorithm\nAlgorithm details.",
    )
    write(
        project, "b.md", "# Introduction\nOther cites AuthorA algorithm network learns."
    )
    service = literature.Literature(FakeEmbedder())
    service.index(project)
    snap = literature.snapshot(project)
    for filename, author in [("a.md", "AuthorA"), ("b.md", "AuthorB")]:
        snap["files"]["literature/" + filename]["metadata"] = {
            "first_author_surname": author
        }
    (project / "literature_index/index.json").write_text(json.dumps(snap))
    sources = service.retrieve(
        project,
        "What algorithm does AuthorA identify and which section describes it?",
        limit=1,
    )
    assert sources and all(s["file"] == "literature/a.md" for s in sources)
    assert any(s["section_number"] == "3.1" for s in sources)


def test_wrapped_pdf_heading():
    chunks = literature.chunks_for(
        "a.pdf",
        "sha",
        [
            Unit(
                text="C.2 A DDITIONAL RESULTS FOR THE\nMODULAR ADDITION TASK\nWe describe results.",
                page=3,
            )
        ],
    )
    assert (
        chunks[0]["section_title"] == "Additional Results For The Modular Addition Task"
    )


def test_grouped_markers_normalize_and_keep_all_valid_sources():
    sources = [dict(number="S5"), dict(number="S7")]
    content = "Algorithm [S5, S7]. Invalid [S99]. Mixed [S7,S99]."
    normalized = literature.normalize_citations(content, sources)
    assert normalized == "Algorithm [S5][S7]. Invalid . Mixed [S7]."
    assert literature.cited(content, sources) == sources
    assert literature.citation_sentences(normalized, sources)["S7"] == [
        "Algorithm [S5][S7].",
        "Mixed [S7].",
    ]
