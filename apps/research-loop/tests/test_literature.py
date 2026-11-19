"""CPU-only extraction, incremental snapshots, hybrid retrieval and source contracts."""

import asyncio
import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import hypotheses
import literature
import pytest
from test_server import app, create, request, upstream  # noqa: F401


class FakeEmbedder:
    identity = ["fake", "v1"]

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.extend(texts)
        return [
            [1, 0]
            if any(w in t.casefold() for w in ("decay", "regularisation"))
            else [0, 1]
            for t in texts
        ]


@pytest.fixture
def library(tmp_path):
    (tmp_path / "project.json").write_text("{}")
    (tmp_path / "literature").mkdir()
    service = literature.Literature(FakeEmbedder())
    return tmp_path, service


def write(project, name, text):
    (project / "literature" / name).write_text(text)


def wait(service, project):
    for _ in range(400):
        result = service.status(project)
        if result["state"] != "indexing":
            return result
        time.sleep(0.01)
    pytest.fail("Index did not finish")


def test_incremental_index_new_changed_removed_and_restart(library):
    project, service = library
    write(project, "decay.md", "# Decay\nRegularisation may help.")
    write(project, "budget.txt", "Hold the budget fixed.")
    service.index(project)
    first = literature.snapshot(project)
    assert len(first["chunks"]) == 2 and len(service.embedder.calls) == 2
    service.index(project)
    assert len(service.embedder.calls) == 2
    write(project, "decay.md", "# Decay\nRegularisation can be tested across seeds.")
    write(project, "new.tex", r"\section{Comparison} fixed budget")
    (project / "literature/budget.txt").unlink()
    service.index(project)
    assert len(service.embedder.calls) == 4
    data = literature.snapshot(project)
    assert len(data["files"]) == 2
    assert data["chunks"][0]["id"] != first["chunks"][0]["id"]
    restarted = literature.Literature(FakeEmbedder())
    restarted.index(project)
    assert not restarted.embedder.calls
    assert restarted.status(project)["passages"] == 2
    assert all(
        c["text"] == c["text"].strip() and c["end"] - c["start"] == len(c["text"])
        for c in data["chunks"]
    )


@pytest.mark.parametrize(
    "suffix,text,expected",
    [
        (".md", "# Note\nDecay matters", "Decay matters"),
        (".txt", "Decay matters", "Decay matters"),
        (".tex", r"\section{Decay} matters", r"\section{Decay}"),
        (".html", "<h1>Decay</h1><p>matters</p><script>bad()</script>", "Decay"),
    ],
)
def test_extraction_chunk_locators(suffix, text, expected):
    units = asyncio.run(literature.parse_document(Path("note" + suffix), text.encode()))
    chunks = literature.chunks_for("note" + suffix, "hash", units)
    assert expected in chunks[0]["text"]
    assert chunks[0]["section"]
    if suffix == ".html":
        assert "bad()" not in chunks[0]["text"]


def test_pdf_pages_survive_bounded_parser():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in ("Weight decay", "Fixed budget"):
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
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    units = asyncio.run(literature.parse_document(Path("paper.pdf"), output.getvalue()))
    chunks = literature.chunks_for("paper.pdf", "hash", units)
    assert [c["page"] for c in chunks] == [1, 2]
    assert chunks[1]["text"] == "Fixed budget"


def test_hybrid_ranking_dedup_and_silence(library):
    project, service = library
    write(
        project,
        "decay.md",
        "# Decay\n" + ("Weight decay regularisation has tradeoffs. " * 100),
    )
    write(project, "budget.md", "Fixed budget comparison.")
    service.index(project)
    sources = service.retrieve(project, "decay regularisation")
    assert sources[0]["file"] == "literature/decay.md"
    assert 1 <= len(sources) <= 6
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            assert a["file"] != b["file"] or max(a["start"], b["start"]) >= min(
                a["end"], b["end"]
            )
    # Exact lexical evidence still finds a passage with a poor semantic score.
    assert service.retrieve(project, "budget")[0]["file"] == "literature/budget.md"
    service.embedder = lambda texts: [[-1, 0] for _ in texts]
    service.embedder.identity = ["fake", "v1"]
    assert service.retrieve(project, "xylophone") == []


def test_read_only_folders_errors_and_failed_changed_file_removed(library, tmp_path):
    project, service = library
    external = tmp_path / "external"
    external.mkdir()
    path = external / "chosen.md"
    path.write_text("Weight decay hypothesis")
    (project / "project.json").write_text(
        json.dumps(dict(literature_dirs=[str(external)]))
    )
    write(project, "broken.txt", "valid")
    service.index(project)
    before = path.read_bytes()
    write(project, "broken.txt", "\x00bad")
    service.index(project)
    status = service.status(project)
    assert status["errors"]["literature/broken.txt"]
    assert not any(
        c["file"].endswith("broken.txt") for c in literature.snapshot(project)["chunks"]
    )
    assert path.read_bytes() == before
    path.unlink()
    service.index(project)
    assert not literature.snapshot(project)["chunks"]


def test_background_progress_and_bad_embedding(library):
    project, service = library
    write(project, "note.md", "Weight decay")
    service.embedder = lambda texts: [[float("nan")]]
    service.start(project)
    status = wait(service, project)
    assert status["progress"] == status["total"] == 1
    assert status["errors"] and status["passages"] == 0


def test_configuration_change_reembeds(library):
    project, service = library
    write(project, "note.md", "Weight decay")
    service.index(project)
    service.embedder.identity = ["fake", "v2"]
    with pytest.raises(ValueError, match="re-index"):
        service.retrieve(project, "decay")
    service.index(project)
    assert len(service.embedder.calls) == 2


def test_not_configured_and_no_mlx(library):
    project, _ = library
    service = literature.Literature()
    assert service.status(project)["state"] == "not configured"
    assert service.retrieve(project, "anything") == []
    with pytest.raises(ValueError, match="not configured"):
        service.start(project)
    code = f"import sys; sys.path.insert(0, {str(Path(literature.__file__).parent)!r}); import literature; assert 'mlx.core' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_citation_filter_and_hypothesis_append_compatibility(library):
    project, service = library
    write(project, "note.md", "Weight decay")
    service.index(project)
    sources = service.retrieve(project, "decay")
    assert literature.cited("Claims [S1] [S99] [S1]", sources) == sources
    assert literature.cited("No source cited", sources) == []
    old = hypotheses.add(project, dict(statement="Old claim"))
    new = hypotheses.add(
        project, dict(statement="New claim", sources=sources), "discussion"
    )
    assert "sources" not in old and new["sources"] == sources
    assert hypotheses.read_rows(project / "hypotheses.jsonl")[1]["sources"] == sources
    with pytest.raises(ValueError, match="sources"):
        hypotheses.add(project, dict(statement="Invalid", sources=["bad"]))


def test_literature_api_disabled_upload_index_remove_and_folder(app, tmp_path):  # noqa: F811
    slug = create(app)
    base = f"/api/projects/{slug}"
    project = app.project_path(slug)
    assert request(app, base + "/literature")["json"]["state"] == "not configured"
    assert request(app, base + "/literature/index", {})["status"] == 400
    app.literature = literature.Literature(FakeEmbedder())
    result = request(
        app,
        base + "/literature/upload",
        dict(name="note.md", content=base64.b64encode(b"Weight decay").decode()),
    )
    assert result["status"] == 202
    assert wait(app.literature, project)["passages"] == 1
    for name in ("../bad.md", "bad.py"):
        assert (
            request(app, base + "/literature/upload", dict(name=name, content="eA=="))[
                "status"
            ]
            == 400
        )
    assert (
        request(app, base + "/literature/remove", dict(name="note.md"))["status"] == 202
    )
    assert wait(app.literature, project)["passages"] == 0
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    (chosen / "source.md").write_text("Regularisation")
    assert (
        request(app, base + "/literature/folder", dict(path=str(chosen)))["status"]
        == 202
    )
    assert wait(app.literature, project)["passages"] == 1
    assert json.loads((project / "project.json").read_text())["literature_dirs"] == [
        str(chosen)
    ]
    assert (
        request(app, base + "/literature/remove-folder", dict(path=str(chosen)))[
            "status"
        ]
        == 202
    )
    assert (
        wait(app.literature, project)["passages"] == 0
        and (chosen / "source.md").exists()
    )


def test_discuss_context_citation_persistence_and_hypothesis_sources(app, upstream):  # noqa: F811
    url, received = upstream
    app.models["q35"] = url
    app.literature = literature.Literature(FakeEmbedder())
    slug = create(app)
    project = app.project_path(slug)
    (project / "literature").mkdir()
    write(project, "note.md", "Weight decay hypothesis")
    app.literature.index(project)
    # The existing stub deliberately cites nothing: retrieval is not citation evidence.
    request(app, f"/api/projects/{slug}/chat", dict(message="decay", model="q35"))
    assert "[S1]" in received[-1]["messages"][0]["content"]
    saved = request(app, f"/api/projects/{slug}")["json"]["discussion"][-1]
    assert (
        saved["citations"] == []
        and saved["sources_note"] == "No provided sources cited."
    )
    sources = app.literature.retrieve(project, "decay")
    from test_server import server

    server.append_message(project, "assistant", "Prediction [S1]", citations=sources)
    response = request(
        app,
        f"/api/projects/{slug}/hypotheses",
        dict(statement="Claim", source="discussion", discussion_index=2),
    )
    assert (
        response["status"] == 201
        and response["json"]["hypothesis"]["sources"] == sources
    )
    assert (
        request(
            app,
            f"/api/projects/{slug}/hypotheses",
            dict(statement="Claim", discussion_index=0),
        )["status"]
        == 400
    )


def test_stream_citations_are_saved_and_unknown_markers_excluded(app, monkeypatch):  # noqa: F811
    from test_server import server

    slug = create(app)
    project = app.project_path(slug)
    (project / "literature").mkdir()
    write(project, "note.md", "Weight decay hypothesis")
    app.literature = literature.Literature(FakeEmbedder())
    app.literature.index(project)

    def reply(req, **kwargs):
        assert "[S1]" in json.loads(req.data)["messages"][0]["content"]
        return io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"Prediction [S1] and invalid [S99]."}}]}\n\ndata: [DONE]\n\n'
        )

    monkeypatch.setattr(server, "urlopen", reply)
    assert (
        request(app, f"/api/projects/{slug}/chat", dict(message="decay"))["status"]
        == 200
    )
    saved = hypotheses.read_rows(project / "discussion.jsonl")[-1]
    assert [s["number"] for s in saved["citations"]] == ["S1"]
    assert saved["citations"][0]["snippet"] == "Weight decay hypothesis"
    assert saved["sources_note"] == ""


@pytest.mark.parametrize("researcher_claim", [True, False])
def test_loop_planning_has_top_three_context_and_journal_ids(
    tmp_path, monkeypatch, researcher_claim
):
    from test_hypotheses import run_loop

    original_init = literature.Literature.__init__
    monkeypatch.setattr(
        literature.Literature,
        "__init__",
        lambda self, embedder=None: original_init(self, FakeEmbedder()),
    )

    def configure(project):
        (project / "project.json").write_text("{}")
        (project / "literature").mkdir()
        write(project, "decay.md", "Weight decay can be compared at a fixed budget.")
        literature.Literature().index(project)
        if researcher_claim:
            hypotheses.add(project, dict(statement="Weight decay reduces score"))

    def proposal(project, prompt, edits):
        assert "[S1]" in prompt and "fixed budget" in prompt
        return dict(
            hypothesis_id="h1" if researcher_claim else None,
            hypothesis="Weight decay reduces score",
            why="Regularisation",
            prediction="Lower score",
            change="VALUE 10 -> 8",
        )

    project, events, prompts = run_loop(
        tmp_path, monkeypatch, proposal, rounds=1, configure=configure
    )
    event = next(e for e in events if e["event"] == "literature_context")
    assert event["source_ids"] == [event["sources"][0]["chunk_id"]]
    assert len(event["sources"]) <= 3
    plan_prompts = [p for p in prompts if "Use experiment_plan:" in p]
    assert plan_prompts and "[S1]" in plan_prompts[0]
    if not researcher_claim:
        assert (
            hypotheses.read_rows(project / "hypotheses.jsonl")[0]["sources"]
            == event["sources"]
        )


def test_repeated_passages_keep_exact_contiguous_spans():
    text = "Repeated research phrase. " * 300
    units = asyncio.run(literature.parse_document(Path("repeat.md"), text.encode()))
    chunks = literature.chunks_for("repeat.md", "hash", units)
    assert len(chunks) > 3
    for previous, current in zip(chunks, chunks[1:]):
        assert previous["end"] <= current["start"]
    assert all(text[c["start"] : c["end"]] == c["text"] for c in chunks)
