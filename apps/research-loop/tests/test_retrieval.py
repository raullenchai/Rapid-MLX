"""Round 6 retrieval, identity, evidence and progress regressions."""

import asyncio
import io
import json
from pathlib import Path

import hub_client
import literature
import pytest
import retrieval
import retrieval_eval
from test_literature import FakeEmbedder, http_embeddings, library, write  # noqa: F401


def fixture():
    chunks = [
        dict(
            id=str(i),
            file=file,
            text=text,
            section_title=title,
            section_number=section,
            page=i + 1,
            start=i * 100,
            end=i * 100 + 80,
        )
        for i, (file, text, title, section) in enumerate(
            [
                (
                    "a",
                    "Weight decay regularization accelerates learning.",
                    "Method",
                    "2",
                ),
                ("a", "Weight decay regularization near duplicate.", "Method", "2"),
                (
                    "b",
                    "Fourier multiplication builds useful algorithms.",
                    "Algorithm",
                    "3.1",
                ),
                ("b", "Weight decay citation bibliography.", "References", None),
                ("c", "A different optimization mechanism.", "Results", "4"),
            ]
        )
    ]
    return dict(
        chunks=chunks,
        vectors=[[1, 0], [1, 0], [0.8, 0.6], [1, 0], [0, 1]],
        files={
            "a": dict(label="Power 2022", metadata=dict(first_author_surname="Power")),
            "b": dict(label="Nanda 2023"),
            "c": dict(label="Liu 2023"),
        },
    )


def test_bm25_rrf_and_mmr():
    data = fixture()
    stats = retrieval.bm25_index(data["chunks"], data["files"])
    assert stats["df"]["weight"] == 3
    assert retrieval.lexical("Power", stats, [0, 1, 2, 4]) == [0, 1]
    assert retrieval.lexical("Algorithm", stats, [0, 1, 2, 4])[0] == 2
    score = retrieval.rrf([[2, 0], [0, 2]])
    assert score[0] == score[2] == 1 / 61 + 1 / 62
    selected = retrieval.retrieve(
        data,
        "weight decay",
        lambda qs: [[1, 0]] * len(qs),
        rerank=lambda q, passages: [1] * len(passages),
    )
    # Adjacent same-paper passages may both be relevant; diversity is opt-in.
    assert selected[:2] == [0, 1]
    assert 3 not in selected
    scoped = retrieval.retrieve(
        data, "Power weight decay", lambda qs: [[1, 0]] * len(qs)
    )
    assert scoped == [0, 1]
    assert retrieval.retrieve(data, "unknown", lambda qs: [[-1, 0]] * len(qs)) == []


def test_one_multquery_call_and_original_preserved():
    data, calls, embedded = fixture(), [], []

    def chat(q):
        calls.append(q)
        return dict(queries=["Fourier multiplication", "optimization mechanism"])

    def embed(qs):
        embedded.extend(qs)
        return [[1, 0]] * len(qs)

    question = "What new experiments could explain why learning emerges so late after memorization?"
    assert retrieval.retrieve(data, question, embed, chat, multiquery=True)
    assert calls == [question] and len(embedded) == 3
    assert question in embedded[0] and "Fourier multiplication" in embedded[1]
    for q in (
        "weight decay",
        "How does Power explain the very late emergence of useful learning?",
        "What does the paper say about why generalization can emerge so late?",
        "Where is section 3 and what does it establish about generalization?",
    ):
        retrieval.retrieve(data, q, embed, chat, multiquery=True)
    assert calls == [question]
    assert retrieval.queries(data, question, lambda q: "not JSON") == [question]
    assert retrieval.queries(
        data, question, lambda q: dict(queries=["", None, "useful"])
    ) == [question, "useful"]


def test_rerank_top_twenty_and_before_mmr():
    data = fixture()
    called = []

    def rerank(query, passages):
        called.append((query, passages))
        return [100 if "Fourier" in p else 0 for p in passages]

    selected = retrieval.retrieve(
        data, "weight decay", lambda qs: [[1, 0]] * len(qs), rerank=rerank
    )
    assert selected[0] == 2 and len(called) == 1 and len(called[0][1]) <= 20
    data["chunks"] = [
        dict(data["chunks"][0], id=str(i), start=i * 100, end=i * 100 + 80)
        for i in range(30)
    ]
    data["vectors"] = [[1, 0]] * 30
    retrieval.retrieve(data, "weight decay", lambda qs: [[1, 0]], rerank=rerank)
    assert len(called[-1][1]) == 20

    def unavailable(*args):
        raise OSError("unavailable")

    assert retrieval.retrieve(
        data, "weight decay", lambda qs: [[1, 0]], rerank=unavailable
    )


def test_eval_plumbing_fixture():
    data = fixture()
    data["bm25"] = retrieval.bm25_index(data["chunks"], data["files"])
    result = retrieval_eval.evaluate(
        data, lambda qs: [[1, 0]] * len(qs), cases=[("Power weight decay", "a", "2")]
    )
    for rows in result.values():
        assert (
            retrieval_eval.metrics(rows)["recall@6"]
            == retrieval_eval.metrics(rows)["MRR"]
            == 1
        )
        assert rows[0]["rank"] == 1
    assert len(retrieval_eval.CASES) == 15
    assert {p for _, p, _ in retrieval_eval.CASES} == {
        "2201.02177",
        "2210.01117",
        "2301.05217",
    }


def test_query_chat_schema_priority_and_rerank_http(monkeypatch):
    calls = []

    class Opener:
        def open(self, request, timeout):
            body = json.loads(request.data)
            calls.append((request, body, timeout))
            if request.full_url.endswith("/rerank"):
                return io.StringIO(
                    json.dumps(
                        dict(
                            results=[
                                dict(index=1, relevance_score=0.2),
                                dict(index=0, relevance_score=0.9),
                            ]
                        )
                    )
                )
            return io.StringIO(
                json.dumps(
                    dict(
                        choices=[
                            dict(
                                message=dict(
                                    content=json.dumps(dict(queries=["a", "b"]))
                                )
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr(literature, "build_opener", lambda *a: Opener())
    monkeypatch.setenv("RAPID_HUB_URL", "http://hub")
    monkeypatch.setenv("RAPID_HUB_TOKEN", "secret")
    monkeypatch.setenv("RAPID_PRIORITY", "background")
    with hub_client.project_scope("study"):
        assert literature.HTTPQueryChat("http://hub/hub/v1", "chat/id")("question") == {
            "queries": ["a", "b"]
        }
        assert literature.HTTPReranker("http://hub/hub/v1", "rank/id")(
            "question", ["a", "b"]
        ) == [0.9, 0.2]
    req, body, timeout = calls[0]
    assert (
        body["response_format"]["json_schema"]["strict"] is True
        and body["max_tokens"] == 180
    )
    assert timeout == 30 and req.get_header("X-rapid-priority") == "interactive"
    assert req.get_header("X-rapid-project") == "study"
    monkeypatch.setattr(
        hub_client,
        "call",
        lambda path: dict(data=[dict(engine="rank", id="id", capabilities=["rerank"])]),
    )
    assert literature.configured_reranker().model == "rank/id"
    monkeypatch.setattr(hub_client, "call", lambda path: dict(data=[]))
    assert literature.configured_reranker() is None
    monkeypatch.delenv("RAPID_HUB_URL")
    monkeypatch.setenv("RAPID_RERANK_URL", "http://standalone/v1")
    monkeypatch.setenv("RAPID_RERANK_MODEL", "ranker")
    assert literature.configured_reranker().model == "ranker"


def test_resolved_identity_aliases_and_model_drift(library, http_embeddings):  # noqa: F811
    project, _ = library
    write(project, "note.txt", "Weight decay")
    http_embeddings["model"] = "mlx-community/embeddinggemma-2-4bit"
    direct = literature.HTTPEmbedder("http://standalone/v1", "embeddinggemma-2-4bit")
    literature.Literature(direct).index(project)
    hub = literature.HTTPEmbedder(
        "http://hub/hub/v1", "engine/mlx-community/embeddinggemma-2-4bit"
    )
    assert literature.Literature(hub).retrieve(project, "decay")
    assert hub.identity == direct.identity == [http_embeddings["model"], 2]
    default = literature.HTTPEmbedder("http://standalone/v1", "default")
    assert default.identity == direct.identity
    http_embeddings["model"] = "another/actual-model"
    with pytest.raises(ValueError, match="model or dimensions changed"):
        default(["query"])
    other = literature.Literature(
        literature.HTTPEmbedder("http://standalone/v1", "default")
    )
    with pytest.raises(ValueError, match="configuration changed"):
        other.retrieve(project, "decay")
    http_embeddings["model"] = "default"
    with pytest.raises(ValueError, match="resolved model"):
        literature.HTTPEmbedder("http://standalone/v1", "default")(["query"])


def test_late_query_dimension_drift(library, http_embeddings):  # noqa: F811
    project, _ = library
    write(project, "note.txt", "Weight decay")
    embed = literature.HTTPEmbedder("http://standalone/v1", "org/embed-model")
    service = literature.Literature(embed)
    service.index(project)
    http_embeddings["dimensions"] = 3
    with pytest.raises(ValueError, match="dimensions changed"):
        service.retrieve(project, "decay")


def test_index_progress_and_bm25_snapshot(library, monkeypatch):  # noqa: F811
    project, service = library
    write(project, "note.txt", "Weight decay")
    seen = []

    async def parse(path, data, progress):
        from rapid_mlx.local_search.extraction import Unit

        for i in (1, 2):
            progress(i, 2)
            seen.append(service.status(project))
        return [Unit(text="Weight decay", page=2)]

    monkeypatch.setattr(literature, "parse_document", parse)

    def embed(texts):
        seen.append(service.status(project))
        return [[1, 0]] * len(texts)

    service.embedder = embed
    service.index(project)
    assert [(s["phase"], s["current"], s["count"]) for s in seen] == [
        ("reading", 1, 2),
        ("reading", 2, 2),
        ("embedding", 0, 1),
    ]
    assert (
        seen[0]["paper"] == "note.txt" and service.status(project)["state"] == "ready"
    )
    assert literature.snapshot(project)["bm25"]["df"]["decay"] == 1


def test_real_pdf_progress_child(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in (
        "Weight decay improves learning.",
        "Fourier multiplication algorithm.",
    ):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                )
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    seen = []
    units = asyncio.run(
        literature.parse_document(
            Path("paper.pdf"), output.getvalue(), lambda i, n: seen.append((i, n))
        )
    )
    assert [(u.page, u.text.strip()) for u in units] == [
        (1, "Weight decay improves learning."),
        (2, "Fourier multiplication algorithm."),
    ]
    assert seen and seen[-1] == (2, 2)


def test_new_hypothesis_retrieves_its_statement(tmp_path, monkeypatch):
    import hypotheses
    from test_hypotheses import run_loop

    calls = []
    original = literature.Literature.__init__
    monkeypatch.setattr(
        literature.Literature,
        "__init__",
        lambda self, embedder=None: original(self, FakeEmbedder()),
    )

    def retrieve(self, project, query, goal="", limit=6):
        calls.append(query)
        return [
            dict(
                number="S1",
                file="paper.pdf",
                chunk_id="specific" if query == "New precise claim" else "planning",
                snippet=query,
                label="Paper 2022",
                section="§2 Method",
                page=3,
            )
        ]

    monkeypatch.setattr(literature.Literature, "retrieve", retrieve)

    def proposal(project, prompt, edits):
        return dict(
            hypothesis_id=None,
            hypothesis="New precise claim",
            why="reason",
            prediction="Lower score",
            change="VALUE 10 -> 8",
        )

    project, events, _ = run_loop(tmp_path, monkeypatch, proposal, rounds=1)
    claim = hypotheses.read_rows(project / "hypotheses.jsonl")[0]
    assert "New precise claim" in calls
    assert claim["sources"][0]["chunk_id"] == "specific"
    assert (
        claim["sources"][0]["label"],
        claim["sources"][0]["section"],
        claim["sources"][0]["page"],
    ) == ("Paper 2022", "§2 Method", 3)


@pytest.mark.parametrize(
    "identity", [["default", 2], ["org/embed-model", 2], ["http://old/v1", "default"]]
)
def test_ambiguous_legacy_identity_requires_reindex(library, http_embeddings, identity):  # noqa: F811
    project, _ = library
    write(project, "note.txt", "Weight decay")
    service = literature.Literature(
        literature.HTTPEmbedder("http://standalone/v1", "org/embed-model")
    )
    service.index(project)
    path = project / "literature_index/index.json"
    data = json.loads(path.read_text())
    data.pop("identity_resolved")
    data["identity"] = identity
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        service.retrieve(project, "weight decay")
    assert path.read_bytes() == original
    service.index(project)
    assert service.retrieve(project, "weight decay")


def test_missing_resolved_response_model_rejected(monkeypatch):
    class Opener:
        def open(self, *args, **kwargs):
            return io.StringIO('{"data":[{"index":0,"embedding":[1,0]}]}')

    monkeypatch.setattr(literature, "build_opener", lambda *a: Opener())
    with pytest.raises(ValueError, match="resolved model"):
        literature.HTTPEmbedder("http://standalone/v1", "default")(["query"])


def test_rewrites_add_lexical_candidates_with_low_dense_scores():
    data = fixture()
    question = "What new experiments might explain delayed emergence following extensive training across networks?"
    embed = lambda qs: [[-1, 0]] * len(qs)
    assert retrieval.retrieve(data, question, embed) == []
    result = retrieval.retrieve(
        data,
        question,
        embed,
        lambda q: {"queries": ["Fourier multiplication", "optimization mechanism"]},
        bm25=True,
        multiquery=True,
    )
    assert set(result) == {2, 4}


def test_stopwords_literals_and_old_snapshot_rebuild():
    assert retrieval.terms("What is beta2 0.98 in section A.1.2, teacher-student?") == [
        "beta2",
        "0.98",
        "section",
        "a.1.2",
        "teacher-student",
    ]
    data = fixture()
    stats = retrieval.bm25_index(data["chunks"], data["files"])
    assert retrieval.lexical("what does the", stats, range(5)) == []
    stale = dict(stats, version=1, documents=[])
    options = dict(bm25=True, stats=stale)
    assert retrieval.retrieve(data, "Fourier", lambda qs: [[-1, 0]], **options) == [2]


def test_original_query_dominates_rewrite_votes():
    # Several identical drifting rewrites must share one budget, not cast more votes.
    original = retrieval.rrf([[0, 1], [1, 0]], [1, 0.2])
    repeated = retrieval.rrf([[0, 1], [1, 0], [1, 0]], [1, 0.1, 0.1])
    assert original == repeated and original[0] > original[1]
    data = fixture()
    calls = []
    question = "What new experiments could explain why learning emerges so late after memorization?"
    retrieval.retrieve(
        data, question, lambda qs: [[1, 0]], chat=lambda q: calls.append(q)
    )
    assert calls == []  # Unproven components are off by default.


def test_equal_rerank_preserves_ranking_and_same_paper_passages():
    data = fixture()
    data["vectors"][0] = [0.8, 0.6]
    data["vectors"][1] = [1, 0]
    embed = lambda qs: [[1, 0]]
    expected = retrieval.retrieve(data, "weight decay", embed)
    assert expected[:2] == [1, 0]
    assert (
        retrieval.retrieve(
            data, "weight decay", embed, rerank=lambda q, ps: [7] * len(ps)
        )
        == expected
    )
    # Even experimental MMR never treats adjacency as full redundancy.
    assert retrieval.retrieve(data, "weight decay", embed, mmr=True)[:2] == [1, 0]


def test_dense_admission_is_not_fused_score_admission():
    data = fixture()
    # No dense candidate can pass cosine .5. Literal evidence remains admissible.
    embed = lambda qs: [[-1, 0]]
    assert retrieval.retrieve(data, "Fourier", embed, bm25=False) == []
    assert retrieval.retrieve(data, "Fourier", embed, bm25=True) == [2]
    assert retrieval.rrf([[2]])[2] < 0.5


def test_eval_frozen_split_and_all_ablations():
    questions = retrieval_eval.QUESTIONS
    assert len(questions) == 30
    assert [
        (c["query"], c["paper"], c["section"]) for c in questions[:15]
    ] == retrieval_eval.CASES
    assert [c["split"] for c in questions] == ["dev"] * 20 + ["held-out"] * 10
    assert sum(c["subset"] == "lexical" for c in questions[20:]) == 7
    result = retrieval_eval.evaluate(
        fixture(),
        lambda qs: [[1, 0]] * len(qs),
        rerank=lambda q, ps: [1] * len(ps),
        cases=[("Power weight decay", "a", "2")],
    )
    for mode in ("dense", "dense+bm25", "dense+bm25+multiquery", "full"):
        assert mode in result and mode + "+rerank" in result
    assert retrieval_eval.metrics(
        [dict(rank=1), dict(rank=3), dict(rank=6), dict(rank=None)]
    ) == {
        "recall@1": 0.25,
        "recall@3": 0.5,
        "recall@6": 0.75,
        "MRR": (1 + 1 / 3 + 1 / 6) / 4,
    }


def test_eval_disk_cache_batch_independence_and_offline(tmp_path):
    calls = []
    cache = retrieval_eval.DiskCache(tmp_path, "model-and-prompt")

    def embed(texts):
        calls.append(texts)
        return [[1, 0] for _ in texts]

    assert cache.embedder(embed)(["a", "b"]) == [[1, 0], [1, 0]]
    assert cache.embedder(embed)(["b", "c", "a"]) == [[1, 0]] * 3
    assert calls == [["a", "b"], ["c"]]
    cache.call("rewrite", ["prompt-v1", "q"], lambda: dict(queries=["rewrite"]))
    offline = retrieval_eval.DiskCache(tmp_path, "model-and-prompt", offline=True)
    assert offline.embedder(None)(["b", "a"]) == [[1, 0]] * 2
    assert offline.call("rewrite", ["prompt-v1", "q"], None) == dict(
        queries=["rewrite"]
    )
    with pytest.raises(RuntimeError, match="cache miss"):
        offline.call("rewrite", ["prompt-v2", "q"], None)
    with pytest.raises(RuntimeError, match="cache miss"):
        offline.embedder(None)(["missing"])
    with pytest.raises(RuntimeError, match="cache miss"):
        retrieval_eval.DiskCache(tmp_path, "different-model", offline=True).embedder(
            None
        )(["a"])


def test_application_default_keeps_coverage_and_cosine_admission(monkeypatch):
    data = fixture()
    seen = []
    real = literature.semantic_ranking

    def admit(scores, rows, background):
        seen.extend(s for _, s in scores)
        return real(scores, rows, background)

    monkeypatch.setattr(literature, "semantic_ranking", admit)
    # A lexical-only source remains available even when dense cosine rejects it.
    assert literature.baseline_rank(data, "Fourier", lambda qs: [[-1, 0]]) == [2]
    assert seen and all(-1 <= score <= 1 for score in seen)
    assert min(seen) == -1
    assert literature.baseline_rank(data, "Fourier", None, limit=0) == []
