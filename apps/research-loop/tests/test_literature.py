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


def test_two_turn_sources_never_rebind_and_transfer_to_hypothesis(app, monkeypatch):  # noqa: F811
    from test_server import server

    slug = create(app)
    project = app.project_path(slug)
    app.literature = literature.Literature(FakeEmbedder())
    a = dict(
        number="S1",
        file="paper-a.pdf",
        chunk_id="a",
        snippet="Claim A",
        page=1,
        section=None,
    )
    b = dict(
        number="S1",
        file="paper-b.pdf",
        chunk_id="b",
        snippet="Claim B",
        page=2,
        section=None,
    )
    retrieved = iter([[a], [b]])
    monkeypatch.setattr(app.literature, "retrieve", lambda *args: next(retrieved))
    prompts = []

    def reply(req, **kwargs):
        prompts.append(json.loads(req.data))
        return io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"Claim A [S1]"}}]}\n\ndata: [DONE]\n\n'
        )

    monkeypatch.setattr(server, "urlopen", reply)
    for _ in range(2):
        assert (
            request(app, f"/api/projects/{slug}/chat", dict(message="Why?"))["status"]
            == 200
        )
    saved = hypotheses.read_rows(project / "discussion.jsonl")
    assert saved[1]["citations"] == saved[3]["citations"] == [a]
    system = prompts[1]["messages"][0]["content"]
    assert "[S1] paper-a.pdf" in system and "[S2] paper-b.pdf" in system
    assert prompts[1]["messages"][2]["content"] == "Claim A [S1]"
    assert prompts[1]["max_tokens"] == 1600
    assert system.endswith("Stop when the answer is complete.")
    assert "aim under 250 words" in system
    result = request(
        app,
        f"/api/projects/{slug}/hypotheses",
        dict(statement="Claim A", discussion_index=3),
    )
    assert result["json"]["hypothesis"]["sources"] == [a]


def test_prompt_source_collisions_unknown_and_duplicate_snapshots():
    a = dict(number="S1", file="a", chunk_id="a", snippet="A")
    b = dict(number="S1", file="b", chunk_id="b", snippet="B")
    history = [
        dict(role="assistant", content="A [S1] [S2]", citations=[a]),
        dict(role="assistant", content="B [S1]", citations=[b]),
    ]
    messages, sources = literature.prompt_sources(history, [a, b])
    assert messages[0]["content"] == "A [S1] [source unavailable]"
    assert messages[1]["content"] == "B [S2]"
    assert len(sources) == 2
    assert history[1]["content"] == "B [S1]"


def pdf_bytes(metadata=None):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    if metadata:
        writer.add_metadata(metadata)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_metadata_preferred_and_model_fallback():
    from rapid_mlx.local_search.extraction import Unit

    units = [Unit(text="Published 2022\nPaper", page=1)]

    def forbidden(text):
        pytest.fail("Usable metadata must avoid model request")

    result = literature.describe_pdf(
        pdf_bytes({"/Title": "Grokking: details", "/Author": "Alethea Power"}),
        units,
        forbidden,
    )
    assert result["label"] == "Power et al. 2022 · Grokking"
    seen = []

    def resolver(text):
        seen.append(text)
        return dict(title="Grokking", first_author_surname="Power", year="2022")

    result = literature.describe_pdf(pdf_bytes(), units, resolver)
    assert seen == [units[0].text] and result["pages"] == 1
    assert result["label"] == "Power et al. 2022 · Grokking"
    assert literature.describe_pdf(pdf_bytes(), units, lambda text: {})["label"] is None


def test_metadata_cached_by_hash_and_labels_survive_restart(library, monkeypatch):
    from rapid_mlx.local_search.extraction import Unit

    project, service = library
    data = pdf_bytes()
    (project / "literature/a.pdf").write_bytes(data)

    async def parse(*args):
        return [Unit(text="Weight decay 2022", page=1)]

    monkeypatch.setattr(literature, "parse_document", parse)
    calls = []

    def describe(*args):
        calls.append(1)
        return dict(label="Power et al. 2022 · Grokking", pages=1)

    monkeypatch.setattr(literature, "describe_pdf", describe)
    service.index(project)
    (project / "literature/b.pdf").write_bytes(data)
    service.index(project)
    literature.Literature(service.embedder).index(project)
    assert calls == [1]
    (project / "project.json").write_text(
        json.dumps(dict(literature_labels={"literature/a.pdf": "My paper"}))
    )
    restarted = literature.Literature(service.embedder)
    assert restarted.status(project)["files"][0]["label"] == "My paper"
    assert restarted.retrieve(project, "decay")[0]["label"] == "My paper"


def test_edit_paper_label_api(app):  # noqa: F811
    slug = create(app)
    project = app.project_path(slug)
    (project / "literature").mkdir()
    write(project, "note.md", "Weight decay")
    app.literature = literature.Literature(FakeEmbedder())
    app.literature.index(project)
    base = f"/api/projects/{slug}/literature/label"
    for payload in [
        dict(file="missing", label="X"),
        dict(file="literature/note.md", label=""),
    ]:
        assert request(app, base, payload)["status"] == 400
    result = request(app, base, dict(file="literature/note.md", label="Research note"))
    assert result["status"] == 200
    assert result["json"]["files"][0]["label"] == "Research note"
    assert json.loads((project / "project.json").read_text())["literature_labels"] == {
        "literature/note.md": "Research note"
    }


def test_metadata_request_strict_bounded_and_configured_endpoint(monkeypatch):
    received = []

    class Opener:
        def open(self, request, timeout):
            received.append((request, timeout))
            content = json.dumps(
                dict(title="Grokking", first_author_surname="Power", year="2022")
            )
            return io.BytesIO(
                json.dumps(dict(choices=[dict(message=dict(content=content))])).encode()
            )

    monkeypatch.setattr(literature, "build_opener", lambda *args: Opener())
    result = literature.HTTPMetadata("http://127.0.0.1:8482/v1", "q35")("x" * 20000)
    req, timeout = received[0]
    body = json.loads(req.data)
    assert req.full_url == "http://127.0.0.1:8482/v1/chat/completions"
    assert body["model"] == "q35" and not body["stream"]
    assert body["max_tokens"] == 200 and timeout == 180
    assert body["response_format"]["json_schema"]["strict"] is True
    assert len(body["messages"][-1]["content"]) == 10000
    assert result["first_author_surname"] == "Power"


@pytest.fixture
def http_embeddings(monkeypatch):
    config = {"dimensions": 2, "calls": []}

    class Opener:
        def open(self, request, timeout):
            body = json.loads(request.data)
            config["calls"].append(body)
            return io.StringIO(
                json.dumps(
                    {
                        "data": [
                            {
                                "index": i,
                                "embedding": [1] + [0] * (config["dimensions"] - 1),
                            }
                            for i, _ in enumerate(body["input"])
                        ]
                    }
                )
            )

    monkeypatch.setattr(literature, "build_opener", lambda *args: Opener())
    return config


def test_embedding_identity_standalone_hub_copy_reuses_index(tmp_path, http_embeddings):
    import shutil

    project = tmp_path / "standalone"
    project.mkdir()
    (project / "project.json").write_text("{}")
    (project / "literature").mkdir()
    write(project, "grokking.txt", "Weight decay improves generalization.")
    direct = literature.HTTPEmbedder("http://standalone/v1", "org/embed-model")
    literature.Literature(direct).index(project)
    assert direct.identity == ["org/embed-model", 2]
    copied = tmp_path / "hub"
    shutil.copytree(project, copied)
    index = copied / "literature_index/index.json"
    original = index.read_bytes()
    hub = literature.HTTPEmbedder("http://hub/hub/v1", "engine/org/embed-model")
    service = literature.Literature(hub)
    assert service.retrieve(copied, "Weight decay")
    assert hub.identity == direct.identity
    assert index.read_bytes() == original
    http_embeddings["calls"].clear()
    service.index(copied)
    assert not http_embeddings["calls"]  # Neither re-extraction nor embedding rebuild.
    other = literature.Literature(
        literature.HTTPEmbedder("http://other/v1", "org/different")
    )
    with pytest.raises(ValueError, match="configuration changed"):
        other.retrieve(copied, "Weight decay")
    http_embeddings["dimensions"] = 3
    changed = literature.Literature(
        literature.HTTPEmbedder("http://hub/hub/v1", "engine/org/embed-model")
    )
    with pytest.raises(ValueError, match="configuration changed"):
        changed.retrieve(copied, "Weight decay")


@pytest.mark.parametrize(
    "url,model",
    [
        ("http://standalone/v1", "org/embed-model"),
        ("http://old-hub/hub/v1", "old-engine/org/embed-model"),
    ],
)
def test_legacy_embedding_identity_reuses_without_rewriting(
    library, http_embeddings, url, model
):
    project, _ = library
    write(project, "note.txt", "Weight decay improves generalization.")
    service = literature.Literature(
        literature.HTTPEmbedder("http://new-hub/hub/v1", "new-engine/org/embed-model")
    )
    service.index(project)
    path = project / "literature_index/index.json"
    data = json.loads(path.read_text())
    data["identity"] = [url, model]
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    assert service.retrieve(project, "Weight decay")
    assert path.read_bytes() == original
    http_embeddings["calls"].clear()
    service.index(project)
    assert not http_embeddings["calls"]
    assert literature.snapshot(project)["identity"] == ["org/embed-model", 2]


@pytest.mark.parametrize(
    "model,dimensions", [("org/different", 2), ("org/embed-model", 3)]
)
def test_embedding_identity_change_rebuilds_index(
    library, http_embeddings, model, dimensions
):
    project, _ = library
    write(project, "note.txt", "Weight decay improves generalization.")
    original = literature.Literature(
        literature.HTTPEmbedder("http://direct/v1", "org/embed-model")
    )
    original.index(project)
    http_embeddings["calls"].clear()
    http_embeddings["dimensions"] = dimensions
    changed = literature.Literature(
        literature.HTTPEmbedder("http://hub/hub/v1", "engine/" + model)
    )
    changed.index(project)
    data = literature.snapshot(project)
    assert data["identity"] == [model, dimensions]
    assert len(data["vectors"][0]) == dimensions
    assert any(
        "title:" in text for call in http_embeddings["calls"] for text in call["input"]
    )
    assert changed.retrieve(project, "Weight decay")
