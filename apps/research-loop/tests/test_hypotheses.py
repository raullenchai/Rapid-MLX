"""Hypothesis evidence, project persistence, and offline loop scheduling regressions."""

import io
import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hypotheses
import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        ([], "untested"),
        (["testing"], "testing"),
        (["no effect"], "testing"),
        (["worse"], "testing"),
        (["crashed"], "testing"),
        (["improved"], "supported"),
        (["supported"], "supported"),
        (["worse", "no effect"], "refuted"),
        (["refuted", "crashed"], "refuted"),
        (["no effect", "no effect"], "inconclusive"),
        (["crashed", "crashed"], "inconclusive"),
        (["crashed", "testing"], "testing"),
        (["crashed", "improved"], "supported"),
    ],
)
def test_status(verdicts, expected):
    assert hypotheses.status([{"verdict": v} for v in verdicts], {}) == expected


def test_configured_threshold_and_invalid_limits():
    assert (
        hypotheses.status([{"verdict": "worse"}], {"experiments_per_hypothesis": 1})
        == "refuted"
    )
    assert (
        hypotheses.status([{"verdict": "worse"}] * 2, {"experiments_per_hypothesis": 3})
        == "testing"
    )
    for value in (0, -1, True, 2.5, "2"):
        with pytest.raises(ValueError):
            hypotheses.limit({"experiments_per_hypothesis": value})


def write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_project_store_concurrent_process_safe_ids_and_derived_status(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(
            pool.map(
                lambda i: hypotheses.add(
                    tmp_path,
                    {
                        "statement": f"Claim {i}",
                        "why": "Reason",
                        "prediction": "Better score",
                    },
                ),
                range(30),
            )
        )
    assert {r["id"] for r in rows} == {f"h{i}" for i in range(1, 31)}
    assert all(
        "status" not in r for r in rows
    )  # Status cannot be asserted by a writer.
    claims = hypotheses.collect(tmp_path, {})
    assert all(h["status"] == "untested" for h in claims)
    with (tmp_path / "hypotheses.jsonl").open("a") as stream:
        stream.write('{"id":"partial')
    assert len(hypotheses.collect(tmp_path, {})) == 30
    assert (
        hypotheses.add(tmp_path, {"statement": "After interrupted append"})["id"]
        == "h31"
    )
    assert len(hypotheses.collect(tmp_path, {})) == 31


def test_legacy_dedupe_across_runs_mixed_events_and_no_rewrites(tmp_path):
    stored = hypotheses.add(tmp_path, {"statement": "New claim"})
    first = tmp_path / "runs" / "old" / "events.jsonl"
    second = tmp_path / "runs" / "live" / "events.jsonl"
    write_events(
        first,
        [
            {"event": "idea", "round": 1, "hypothesis": "Same claim"},
            {"event": "result", "round": 1, "verdict": "refuted", "value": 20},
            {"event": "idea", "round": 2, "hypothesis": "Same claim"},
            {"event": "result", "round": 2, "verdict": "no effect", "value": 10},
        ],
    )
    write_events(
        second,
        [
            {
                "event": "idea",
                "round": 1,
                "hypothesis_id": stored["id"],
                "hypothesis": "New claim",
                "change": "X 1 -> 2",
            },
            {
                "event": "result",
                "round": 1,
                "hypothesis_id": stored["id"],
                "verdict": "improved",
                "value": 8,
            },
            {"event": "idea", "round": 2, "hypothesis": "Same claim"},
        ],
    )
    before = [p.read_bytes() for p in (first, second)]
    claims = hypotheses.collect(tmp_path, {})
    assert len(claims) == 2
    assert claims[0]["id"] == stored["id"] and claims[0]["status"] == "supported"
    legacy = claims[1]
    assert legacy["statement"] == "Same claim" and legacy["status"] == "refuted"
    assert len(legacy["experiments"]) == 3
    assert legacy["id"] == hypotheses.legacy_id("Same claim")
    assert before == [p.read_bytes() for p in (first, second)]


def test_migrate_project_queue_idempotently_without_moving_original(tmp_path):
    queue = tmp_path / "ideas.jsonl"
    write_events(queue, [{"hypothesis": "Claim", "change": "X 1 -> 2"}])
    before = queue.read_bytes()
    hypotheses.migrate_queue(tmp_path)
    hypotheses.migrate_queue(tmp_path)
    assert queue.read_bytes() == before
    claims = hypotheses.collect(tmp_path, {})
    assert len(claims) == 1 and claims[0]["statement"] == "Claim"
    assert claims[0]["source"] == "you"


def test_symlink_store_rejected(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("")
    (tmp_path / "hypotheses.jsonl").symlink_to(elsewhere)
    with pytest.raises(ValueError):
        hypotheses.add(tmp_path, {"statement": "Claim"})
    with pytest.raises(ValueError):
        hypotheses.collect(tmp_path, {})
    assert elsewhere.read_text() == ""


def run_loop(tmp_path, monkeypatch, proposal, rounds=2, configure=None):
    project = tmp_path / "project"
    project.mkdir()
    task = dict(
        title="Offline study",
        editable="measure.py",
        metric="score",
        direction="min",
        noise=0.5,
        fixed=["^BUDGET = 10$"],
        seeds=[0, 1],
    )
    (project / "task.json").write_text(json.dumps(task))
    (project / "program.md").write_text(
        "Improve the score; leave evaluation and budget fixed."
    )
    (project / "measure.py").write_text(
        'import json\nBUDGET = 10\nVALUE = 10\nprint("RESULT " + json.dumps({"score": VALUE}))\n'
    )
    if configure:
        configure(project)
    run = project / "runs" / "test"
    prompts = []
    edits = 0

    def reply(request, **kwargs):
        nonlocal edits
        prompt = json.loads(request.data)["messages"][-1]["content"]
        prompts.append(prompt)
        import re

        body = json.loads(request.data)
        kind = body["response_format"]["json_schema"]["name"]
        if kind == "edit":
            edits += 1
            content = dict(
                base_sha256=re.search(r"base_sha256: ([a-f0-9]+)", prompt)[1],
                edits=[
                    dict(
                        search_lines=["VALUE = 10"],
                        replace_lines=[f"VALUE = {10.1 if edits == 1 else 8}"],
                    )
                ],
            )
        elif kind == "conclusion":
            hid = re.search(r"conclusion for (h\d+)", prompt)[1]
            content = dict(hypothesis_id=hid, summary_md="Evidence recorded.")
        elif kind == "run_report":
            content = dict(
                summary_md="Run complete.", next_steps_md="Repeat with more seeds."
            )
        else:
            old = proposal(project, prompt, edits)
            content = dict(
                hypothesis_id=old["hypothesis_id"],
                new_hypothesis=dict(
                    statement=old.get("hypothesis", ""),
                    why=old["why"],
                    prediction=old["prediction"],
                )
                if old["hypothesis_id"] is None
                else None,
                change=old["change"],
                prediction=old["prediction"],
            )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(content)},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", reply)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(ROOT / "loop.py"), str(project), str(run), "--rounds", str(rounds)],
    )
    runpy.run_path(str(ROOT / "loop.py"), run_name="__main__")
    return project, hypotheses.read_rows(run / "events.jsonl"), prompts


def test_one_hypothesis_many_experiments_and_live_researcher_priority(
    tmp_path, monkeypatch
):
    def configure(project):
        hypotheses.add(
            project,
            {
                "statement": "Regularization improves score",
                "why": "Less overfitting",
                "prediction": "Lower score",
            },
        )

    def proposal(project, prompt, edits):
        hid = "h1"
        schema = json.loads(prompt.split("Reply with JSON matching this schema: ")[-1])
        assert schema["properties"]["hypothesis_id"]["enum"] == [
            "h1" if edits == 0 else "h2"
        ]
        if edits == 0:
            # Add while the loop is already loaded, exactly as a web mutation does.
            hypotheses.add(
                project, {"statement": "Smoothing improves score"}, "discussion"
            )
        if (
            "Researcher priority: you MUST test h2" in prompt
            and "Previous proposal rejected" in prompt
        ):
            hid = "h2"
        return dict(
            hypothesis_id=hid,
            why="Less overfitting",
            prediction="Lower score",
            change=f"VALUE 10 -> {10.1 if edits == 0 else 8}",
        )

    project, events, prompts = run_loop(
        tmp_path, monkeypatch, proposal, configure=configure
    )
    proposals = [e for e in events if e["event"] == "idea"]
    assert [e["hypothesis_id"] for e in proposals] == ["h1", "h2"]
    assert any("Previous proposal rejected" in p for p in prompts)
    assert all(
        e.get("hypothesis_id")
        for e in events
        if e.get("round", 0) > 0 and e["event"] != "hypothesis_error"
    )
    claims = hypotheses.collect(project, {})
    assert [h["status"] for h in claims] == ["inconclusive", "supported"]


def test_same_claim_second_experiment_can_support_it(tmp_path, monkeypatch):
    def configure(project):
        hypotheses.add(project, {"statement": "Regularization improves score"})

    def proposal(project, prompt, edits):
        schema = json.loads(prompt.split("Reply with JSON matching this schema: ")[-1])
        assert schema["properties"]["hypothesis_id"]["enum"] == (
            ["h1"] if edits == 0 else ["h1", None]
        )
        assert f"h1 ({edits}/2 experiments)" in prompt
        return dict(
            hypothesis_id="h1",
            why="Less overfitting",
            prediction="Lower score",
            change=f"VALUE 10 -> {10.1 if edits == 0 else 8}",
        )

    project, events, _ = run_loop(tmp_path, monkeypatch, proposal, configure=configure)
    experiments = [e for e in events if e["event"] == "result" and e.get("round")]
    assert [e["verdict"] for e in experiments] == ["no effect", "improved"]
    assert all(e["hypothesis_id"] == "h1" for e in experiments)
    assert len(hypotheses.collect(project, {})[0]["experiments"]) == 2
    assert hypotheses.collect(project, {})[0]["status"] == "supported"
    for event in [e for e in events if e["event"] == "result"]:
        assert len(event.get("result", {}).get("per_seed", [])) == 2


@pytest.mark.parametrize(
    "invalid", ["concluded", "duplicate", "unknown", "long", "same claim"]
)
def test_loop_rejects_invalid_proposals_without_running(tmp_path, monkeypatch, invalid):
    def configure(project):
        hypotheses.add(project, {"statement": "Existing claim"}, "model")
        write_events(
            project / "runs" / "prior" / "events.jsonl",
            [
                dict(
                    event="idea",
                    round=1,
                    hypothesis_id="h1",
                    hypothesis="Existing claim",
                    change="VALUE 10 -> 8",
                ),
                dict(
                    event="result",
                    round=1,
                    verdict="improved",
                    hypothesis_id="h1",
                    value=8,
                ),
            ],
        )

    def proposal(project, prompt, edits):
        schema = json.loads(prompt.split("Reply with JSON matching this schema: ")[-1])
        assert schema["properties"]["hypothesis_id"]["enum"] == [None]
        assert "Concluded hypotheses (do not test): h1 (supported)" in prompt
        data = dict(
            hypothesis_id=None,
            hypothesis="A distinct claim",
            why="Reason",
            prediction="Lower score",
            change="VALUE 10 -> 9",
        )
        if invalid == "concluded":
            data["hypothesis_id"] = "h1"
        elif invalid == "duplicate":
            data["change"] = " VALUE 10 -> 8 "
        elif invalid == "unknown":
            data["hypothesis_id"] = "h99"
        elif invalid == "long":
            data["hypothesis"] = "word " * 41
        else:
            data["hypothesis"] = "Existing claim"
        return data

    _, events, prompts = run_loop(
        tmp_path, monkeypatch, proposal, rounds=1, configure=configure
    )
    assert any(e["event"] == "hypothesis_error" for e in events)
    assert not any(e.get("round", 0) > 0 and e["event"] == "run_start" for e in events)
    assert sum("Open hypotheses:" in p for p in prompts) == 2


def test_model_creates_claim_and_reuses_it_for_second_experiment(tmp_path, monkeypatch):
    def proposal(project, prompt, edits):
        return dict(
            hypothesis_id=None if edits == 0 else "h1",
            hypothesis="Smoothing improves score",
            why="Less variance",
            prediction="Lower score",
            change=f"VALUE 10 -> {10.1 if edits == 0 else 8}",
        )

    project, events, _ = run_loop(tmp_path, monkeypatch, proposal)
    claims = hypotheses.collect(project, {})
    assert len(claims) == 1 and claims[0]["id"] == "h1"
    assert claims[0]["source"] == "model" and claims[0]["status"] == "supported"
    assert [e["hypothesis_id"] for e in events if e["event"] == "idea"] == ["h1", "h1"]


def test_legacy_project_queue_visible_without_writes(tmp_path):
    write_events(tmp_path / "ideas.jsonl", [{"hypothesis": "An untested claim"}])
    claims = hypotheses.collect(tmp_path, {})
    assert len(claims) == 1 and claims[0]["status"] == "untested"
    assert not (tmp_path / "hypotheses.jsonl").exists()


def test_legacy_identity_matches_browser_utf8_contract():
    assert hypotheses.legacy_id("Stronger decay α 💡") == "legacy-04a44bf9e6cd4edc"


@pytest.mark.parametrize("field", ["page", "page_end"])
@pytest.mark.parametrize(
    "value", ['1"><img src=x onerror=alert(1)>', "2", 0, -1, True, 1.5]
)
def test_source_pages_require_positive_integers(tmp_path, field, value):
    source = dict(
        number="S1", file="paper.pdf", chunk_id="c", snippet="Evidence", page=1
    )
    source[field] = value
    with pytest.raises(ValueError, match="sources"):
        hypotheses.add(tmp_path, dict(statement="Claim", sources=[source]))
    assert not (tmp_path / "hypotheses.jsonl").exists()


def test_source_pages_optional_or_positive(tmp_path):
    for pages in ({}, {"page": None, "page_end": None}, {"page": 1, "page_end": 2}):
        source = dict(
            number="S1", file="paper.pdf", chunk_id="c", snippet="Evidence", **pages
        )
        assert hypotheses.add(tmp_path, dict(statement="Claim", sources=[source]))[
            "sources"
        ] == [source]
