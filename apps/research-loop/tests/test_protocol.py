"""Protocol boundaries: durable state, unsafe producers, incomplete evidence and transport."""

import io
import json
import re
import subprocess
import sys
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hypotheses
import protocol as p
import pytest

TASK = dict(
    editable="measure.py", metric="score", direction="max", noise=0.5, seeds=[0, 1]
)


def register(store, content=b"value\n1\n", **extra):
    return store.register(
        content,
        **(
            dict(
                id="dataset",
                kind="data",
                role="curve",
                title="Measurements",
                path="data.csv",
                owner=dict(scope="run", run=store.run.name),
            )
            | extra
        ),
    )


def test_journal_parallel_seq_and_partial_recovery(tmp_path):
    journal = p.Journal(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: journal.append("tick", n=n), range(40)))
    rows = p.events(tmp_path)
    assert [e["seq"] for e in rows] == list(range(1, 41))
    with (tmp_path / "events.jsonl").open("ab") as stream:
        stream.write(b'{"event":"uncommitted')
    assert len(p.events(tmp_path)) == 40
    assert journal.append("done")["seq"] == 41
    assert len(p.events(tmp_path)) == 41


def test_registration_idempotent_revision_and_atomic_publication(tmp_path, monkeypatch):
    store = p.Artifacts(tmp_path)
    original = store.journal._append

    def published(stream, event, data):
        if event == "artifact":
            a = data["artifact"]
            assert p.digest((tmp_path / a["path"]).read_bytes()) == a["sha256"]
        return original(stream, event, data)

    monkeypatch.setattr(store.journal, "_append", published)
    first = register(store)
    assert register(store) == first
    assert len(p.events(tmp_path)) == 1
    second = register(store, b"value\n2\n")
    assert second["revision"] == 2 and second["supersedes"] == {
        "id": "dataset",
        "revision": 1,
    }
    assert first["path"] != second["path"]
    assert p.artifact_content(tmp_path, first) == b"value\n1\n"
    assert not list(tmp_path.rglob(".artifact-*"))


def test_quota_rejects_before_copy_and_counts_old_revisions(tmp_path, monkeypatch):
    store = p.Artifacts(tmp_path)
    first = register(store)
    monkeypatch.setattr(p, "RUN_LIMIT", first["bytes"] + 2)
    with pytest.raises(ValueError, match="Per-run quota"):
        register(store, b"value\n22\n")
    assert len(list((tmp_path / "artifacts").iterdir())) == 1
    monkeypatch.setattr(p, "ARTIFACT_LIMIT", 2)
    with pytest.raises(ValueError, match="Per-artifact quota"):
        register(store)
    assert [e["event"] for e in p.events(tmp_path)] == [
        "artifact",
        "artifact_rejected",
        "artifact_rejected",
    ]


@pytest.mark.parametrize(
    "name", ["../outside.csv", "/absolute.csv", "sub/../../escape.csv"]
)
def test_path_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        register(p.Artifacts(tmp_path), path=name)


def test_manifest_symlink_mime_and_copy_isolation(tmp_path):
    store = p.Artifacts(tmp_path / "run")
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "coords.csv").write_text("x,y,z\n1,2,3\n")
    (producer / "fake.png").write_text("not an image")
    (producer / "link.csv").symlink_to(producer / "coords.csv")
    (producer / "unused.csv").write_text("ignored")
    entries = [
        dict(kind="data", path="link.csv", title="Symlink"),
        dict(kind="image", path="fake.png", title="Fake image"),
        dict(kind="data", path="coords.csv", title="Coordinates"),
    ]
    (producer / "manifest.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries)
    )
    rows = store.manifest(producer, dict(scope="run", run="run"), "script")
    assert len(rows) == 1 and rows[0]["id"] == "script-2"
    (producer / "coords.csv").write_text("later changes do not affect copy")
    assert p.artifact_content(store.run, rows[0]) == b"x,y,z\n1,2,3\n"
    assert sum(e["event"] == "artifact_rejected" for e in p.events(store.run)) >= 2


@pytest.mark.parametrize(
    "content,name,kind",
    [
        (b"\x00binary", "source.py", "code"),
        (b"<svg/>", "fake.csv", "data"),
        (b'{"x": []}\n', "nested.jsonl", "data"),
        (b"text", "fake.jpg", "image"),
        (b"text", "data.json", "data"),
        (b"report", "report.html", "report"),
    ],
)
def test_mime_rejected(tmp_path, content, name, kind):
    with pytest.raises(ValueError):
        register(p.Artifacts(tmp_path), content, path=name, kind=kind)


def test_chart_schema_and_refs(tmp_path):
    store = p.Artifacts(tmp_path)
    data = register(store, b"t,v,low,high\n1,2,1,3\n")
    spec = p.default_chart(dict(x="t", series=[dict(key="v")]), data["id"])
    spec.update(
        band=dict(low="low", high="high"), reference=dict(value=2, label="Reference")
    )
    a = register(
        store,
        json.dumps(spec).encode(),
        id="chart",
        kind="chart",
        path="chart.json",
        refs=[data["id"]],
    )
    assert a["kind"] == "chart"
    for change in (
        dict(url="https://external.invalid"),
        dict(mark="area"),
        dict(data=dict(url="https://external.invalid")),
        dict(
            encoding=dict(x=dict(field="t", expression="alert(1)"), y=dict(field="v"))
        ),
        dict(layer=[dict(transform=[])]),
    ):
        with pytest.raises(ValueError):
            p.validate_chart(dict(spec, **change))
    missing = dict(spec, encoding=dict(x=dict(field="t"), y=dict(field="absent")))
    with pytest.raises(ValueError, match="field absent"):
        register(
            store,
            json.dumps(missing).encode(),
            id="missing",
            kind="chart",
            path="chart.json",
            refs=[data["id"]],
        )
    with pytest.raises(ValueError, match="registered data"):
        register(
            store,
            json.dumps(spec).encode(),
            id="no-ref",
            kind="chart",
            path="chart.json",
        )


def edit(source, blocks, **extra):
    return dict(
        base_sha256=p.digest(source.encode()),
        edits=[dict(search_lines=old, replace_lines=new) for old, new in blocks],
        **extra,
    )


def test_edits_atomic_unique_overlap_hash_syntax_fixed():
    source = "BUDGET = 10\nx = 1\ny = 2\n"
    assert p.apply_edits(
        source, edit(source, [(["x = 1"], ["x = 3"])]), ["^BUDGET = 10$"]
    ) == source.replace("x = 1", "x = 3")
    cases = [
        edit(source, [(["missing"], [])]),
        edit(source, [(["x = 1"], ["x = ("])]),
        edit(source, [(["BUDGET = 10"], ["BUDGET = 20"])]),
        edit(source, [(["x = 1", "y = 2"], []), (["y = 2"], [])]),
    ]
    for message in cases:
        with pytest.raises(ValueError):
            p.apply_edits(source, message, ["^BUDGET = 10$"])
        assert source == "BUDGET = 10\nx = 1\ny = 2\n"
    with pytest.raises(ValueError, match="base_sha256"):
        p.apply_edits(source, dict(cases[0], base_sha256="bad"))
    with pytest.raises(ValueError, match="2 times"):
        p.apply_edits(
            "x = 1\nx = 1\n", edit("x = 1\nx = 1\n", [(["x = 1"], ["x = 2"])])
        )


def test_loop_edit_retry_and_crash_repair_use_exact_base(tmp_path, monkeypatch):
    import runpy

    source = (
        'VALUE = 10\nprint("RESULT " + __import__("json").dumps({"score": VALUE}))\n'
    )
    crashed = source.replace("VALUE = 10", 'raise RuntimeError("candidate crash")')
    (tmp_path / "measure.py").write_text(source)
    (tmp_path / "program.md").write_text("Lower the score.")
    (tmp_path / "task.json").write_text(
        json.dumps(dict(TASK, seeds=[0], fixed=[], title="Offline study"))
    )
    run = tmp_path / "runs" / "test"
    edit_prompts = []

    def upstream(request, **kwargs):
        body = json.loads(request.data)
        kind = body["response_format"]["json_schema"]["name"]
        prompt = body["messages"][-1]["content"]
        if kind == "experiment_plan":
            return response(
                dict(
                    hypothesis_id=None,
                    new_hypothesis=dict(
                        statement="Lower values improve score",
                        why="Test",
                        prediction="Lower",
                    ),
                    change="Lower VALUE",
                    prediction="Lower",
                )
            )
        if kind == "edit":
            edit_prompts.append(prompt)
            attempt = len(edit_prompts)
            base = source if attempt <= 2 else crashed
            assert f"base_sha256: {p.digest(base.encode())}" in prompt
            assert f"BEGIN CURRENT BASE SOURCE\n{base}END CURRENT BASE SOURCE" in prompt
            if attempt % 2:
                return response(edit(base, [(["VALUE = 999"], ["VALUE = 8"])]))
            assert "Nearest matching lines" in prompt
            assert "1: " + base.splitlines()[0] in prompt
            return response(
                edit(
                    base,
                    [
                        (
                            [base.splitlines()[0]],
                            [crashed.splitlines()[0] if attempt == 2 else "VALUE = 8"],
                        )
                    ],
                )
            )
        if kind == "conclusion":
            return response(dict(hypothesis_id="h1", summary_md="Evidence recorded"))
        return response(dict(summary_md="Finished", next_steps_md="Repeat"))

    monkeypatch.setattr("urllib.request.urlopen", upstream)
    monkeypatch.setattr(
        sys, "argv", ["loop.py", str(tmp_path), str(run), "--rounds", "1"]
    )
    runpy.run_path(str(Path(__file__).parents[1] / "loop.py"), run_name="__main__")
    rows = p.events(run)
    assert len(edit_prompts) == 4
    assert sum(e["event"] == "model_rejected" for e in rows) == 2
    records = [e["record"] for e in rows if e["event"] == "result_record"]
    assert records[-2]["verdict"] == "crashed"
    assert records[-1]["aggregate"] == 8


@pytest.mark.parametrize("words", [16, 30, 40])
def test_loop_accepts_natural_length_claims(tmp_path, monkeypatch, words):
    from test_hypotheses import run_loop

    def proposal(project, prompt, edits):
        assert "one sentence, at most 30 words" in prompt
        return dict(
            hypothesis_id=None,
            hypothesis="word " * words,
            why="Reason",
            prediction="Lower",
            change="VALUE 10 -> 8",
        )

    _, rows, _ = run_loop(tmp_path, monkeypatch, proposal, rounds=1)
    assert not any(e["event"] == "hypothesis_error" for e in rows)
    assert any(e["event"] == "run_start" and e["round"] == 1 for e in rows)


def test_plan_schema_is_per_call_and_validated_on_fallback(monkeypatch):
    schemas = [
        p.experiment_plan_schema(["h1", "h3"]),
        p.experiment_plan_schema(["h2"], priority="h2"),
        p.experiment_plan_schema([]),
    ]
    assert [s["properties"]["hypothesis_id"]["enum"] for s in schemas] == [
        ["h1", "h3", None],
        ["h2"],
        [None],
    ]
    assert "enum" not in p.SCHEMAS["experiment_plan"]["properties"]["hypothesis_id"]
    calls = []

    def upstream(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        if len(calls) == 1:
            assert body["response_format"]["json_schema"]["schema"] == schemas[0]
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "unsupported",
                {},
                io.BytesIO(b"json_schema unsupported"),
            )
        hid = "closed" if len(calls) == 2 else "h3"
        return response(
            dict(
                hypothesis_id=hid,
                new_hypothesis=None,
                change="Change",
                prediction="Lower",
            )
        )

    monkeypatch.setattr("urllib.request.urlopen", upstream)
    model = p.Model("http://local/v1", "test", lambda *a, **kw: None)
    assert (
        model.message("experiment_plan", "Choose", schema=schemas[0])["hypothesis_id"]
        == "h3"
    )
    assert len(calls) == 3
    assert "enum" in calls[-1]["messages"][-1]["content"]


def response(value, finish="stop"):
    return io.BytesIO(
        json.dumps(
            dict(
                choices=[
                    dict(finish_reason=finish, message=dict(content=json.dumps(value)))
                ]
            )
        ).encode()
    )


def test_model_fallback_truncation_retry_and_schema_check(monkeypatch):
    calls, logs = [], []
    good = dict(statement="Claim", why="Reason", prediction="Higher")

    def upstream(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "unsupported",
                {},
                io.BytesIO(b"response_format unsupported"),
            )
        if len(calls) == 2:
            return response(good, "length")
        assert "finish_reason" in body["messages"][-1]["content"]
        return response(good)

    monkeypatch.setattr("urllib.request.urlopen", upstream)
    client = p.Model(
        "http://local/v1",
        "test",
        lambda event, **data: logs.append(dict(event=event, **data)),
    )
    assert client.message("hypothesis", "Propose a claim") == good
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    assert any(e["event"] == "model_format_fallback" for e in logs)
    assert any(e["event"] == "model_rejected" and "length" in e["reason"] for e in logs)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: response(dict(good, status="supported")),
    )
    with pytest.raises(ValueError, match="unknown fields"):
        client.message("hypothesis", "No status field allowed")


def test_result_missing_seeds_and_conflicting_reducer():
    record = p.frozen_record(1, 0, TASK, "code", hypothesis_id="h1")
    incomplete = p.result_record(record, {0: dict(score=10)}, parent_value=2)
    assert incomplete["verdict"] == "incomplete" and incomplete["missing_seeds"] == [1]
    assert hypotheses.status([incomplete], TASK, ended=True) == "inconclusive"
    complete = p.result_record(
        record, {0: dict(score=10), 1: dict(score=12)}, parent_value=2
    )
    assert complete["verdict"] == "improved" and complete["aggregate"] == 11
    conflict = hypotheses.evidence([complete, dict(verdict="worse")], TASK)
    assert conflict["status"] == "inconclusive" and conflict["conflicting_evidence"]
    assert (
        hypotheses.status([dict(verdict="worse", missing_seeds=[1])], TASK, ended=True)
        == "inconclusive"
    )
    for verdict, status in [
        ("worse", "refuted"),
        ("no effect", "inconclusive"),
        ("crashed", "inconclusive"),
    ]:
        evidence = hypotheses.evidence(
            [
                dict(
                    verdict=verdict,
                    run_ended=True,
                    run_ended_reason="stopped by researcher",
                )
            ],
            TASK,
        )
        assert evidence["status"] == status and "stopped" in evidence["reason"]


def test_latest_attempt_only_and_frozen_state_wins_over_compat_events(tmp_path):
    hypotheses.add(tmp_path, dict(statement="Claim"))
    run = tmp_path / "runs" / "run"
    journal = p.Journal(run)
    for attempt, value in [(0, 10), (1, 1)]:
        record = p.frozen_record(1, attempt, TASK, str(value), hypothesis_id="h1")
        journal.append("experiment_record", record=record)
        journal.append(
            "result_record",
            record=p.result_record(
                record, {0: dict(score=value), 1: dict(score=value)}, parent_value=5
            ),
        )
    journal.append("result", round=1, verdict="improved", hypothesis_id="h1")
    journal.append("done", reason="stopped")
    claim = hypotheses.collect(tmp_path, TASK)[0]
    assert len(claim["experiments"]) == 1
    assert claim["status"] == "refuted" and claim["experiments"][0]["attempt"] == 1


def test_legacy_live_adapter_stable_seq_no_rewrite_and_bundle(tmp_path):
    run = tmp_path
    lines = [
        dict(event="baseline", result=dict(score=2), metric="score"),
        dict(event="idea", round=1, hypothesis="Claim", parent=0),
        dict(event="curve", round=1, seed=0, step=1, score=3),
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    (run / "round0_measure.py").write_text("x = 1\n")
    (run / "round1_measure.py").write_text("x = 2\n")
    before = {f.name: f.read_bytes() for f in run.iterdir()}
    task = dict(TASK, curve=dict(x="step", series=[dict(key="score")]))
    view = p.snapshot(run, task)
    assert view["legacy"] and view["watermark"] == 3
    assert {a["kind"] for a in view["artifacts"]} == {"code", "data", "chart"}
    assert view == p.snapshot(run, task)
    archive = zipfile.ZipFile(io.BytesIO(p.bundle(run, view)))
    assert {"PROTOCOL.md", "envelopes.json", "records.json", "ledger.csv"} <= set(
        archive.namelist()
    )
    assert before == {f.name: f.read_bytes() for f in run.iterdir()}
    with (run / "events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(event="curve", round=1, seed=0, step=2, score=4)) + "\n"
        )
    later = p.snapshot(run, task)
    assert later["watermark"] == 4
    assert [e["seq"] for e in later["events"][:3]] == [1, 2, 3]
    with pytest.raises(ValueError, match="read-only"):
        p.Journal(run).append("no writes")


def test_templates_fixed_lines_units_and_script_artifacts(tmp_path):
    root = Path(__file__).parents[1] / "templates"
    for name in ("no-three-ap", "lj-cluster"):
        task = json.loads((root / name / "task.json").read_text())
        source = (root / name / task["editable"]).read_text()
        assert all(re.search(pattern, source, re.M) for pattern in task["fixed"])
    chemistry = root / "lj-cluster" / "cluster.py"
    result = subprocess.run(
        [sys.executable, str(chemistry)],
        env={**__import__("os").environ, "ARTIFACT_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    store = p.Artifacts(tmp_path / "run")
    items = store.manifest(tmp_path, dict(scope="run", run="run"), "chem")
    assert [a["kind"] for a in items] == ["data", "image", "report"]
    assert len(p.data_rows(p.artifact_content(store.run, items[0]), "text/csv")) == 13
    assert len(p.artifact_content(store.run, items[2]).decode().splitlines()) == 3


def test_unfinished_repair_cannot_inherit_success(tmp_path):
    hypotheses.add(tmp_path, dict(statement="Claim"))
    journal = p.Journal(tmp_path / "runs" / "run")
    first = p.frozen_record(1, 0, TASK, "x = 1", hypothesis_id="h1")
    journal.append("experiment_record", record=first)
    journal.append(
        "result_record",
        record=p.result_record(
            first, {0: dict(score=10), 1: dict(score=10)}, parent_value=1
        ),
    )
    journal.append("experiment_record", record=dict(first, attempt=1))
    journal.append("done", reason="interrupted")
    claim = hypotheses.collect(tmp_path, TASK)[0]
    assert claim["status"] == "inconclusive" and claim["missing_evidence"]
    assert claim["experiments"][0]["attempt"] == 1


def test_manifest_chart_can_reference_supplied_prefix(tmp_path):
    directory = tmp_path / "script"
    directory.mkdir()
    (directory / "data.csv").write_text("x,y\n1,2\n")
    spec = p.default_chart(dict(x="x", series=[dict(key="y")]), "prefix-0")
    (directory / "chart.json").write_text(json.dumps(spec))
    rows = [
        dict(kind="data", path="data.csv", title="Data"),
        dict(kind="chart", path="chart.json", title="Chart", refs=["prefix-0"]),
    ]
    (directory / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    store = p.Artifacts(tmp_path / "run")
    artifacts = store.manifest(directory, dict(scope="run", run="run"), "prefix")
    assert [a["kind"] for a in artifacts] == ["data", "chart"]


def test_legacy_prompt_parse_still_validates_and_edits_atomically():
    source = "x = 1\n"
    prompt = "base_sha256: " + p.digest(source.encode())
    legacy = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"
    message = p.parse_message(legacy, "edit", False, prompt)
    assert p.apply_edits(source, message) == "x = 2\n"
    plan = p.parse_message(
        'Here is a plan: {"hypothesis_id":null,"hypothesis":"Claim","why":"Reason","change":"Change","prediction":"Higher"}',
        "experiment_plan",
        False,
        "",
    )
    p.validate(plan, p.SCHEMAS["experiment_plan"])
    assert plan["new_hypothesis"]["statement"] == "Claim"
    with pytest.raises(ValueError):
        p.parse_message(legacy, "edit", True, prompt)


def test_hypothesis_conclusions_across_runs_and_bad_envelopes(tmp_path):
    hypotheses.add(tmp_path, dict(statement="Claim"))
    store = p.Artifacts(tmp_path / "runs" / "run")
    artifact = store.register(
        b"Conclusion",
        id="conclusion",
        kind="report",
        role="conclusion",
        title="Conclusion",
        path="conclusion.md",
        owner=dict(scope="hypothesis", run="run", hypothesis_id="h1"),
    )
    store.journal.append("artifact", artifact=None)
    store.journal.append("artifact", artifact={"owner": "invalid"})
    claim = hypotheses.collect(tmp_path, TASK)[0]
    assert claim["reports"] == [artifact]
    view = p.snapshot(store.run, TASK)
    assert all(a["kind"] == "file" for a in view["artifacts"][1:])


def test_legacy_replay_uses_reducer_at_cursor(tmp_path):
    import server

    project = tmp_path
    hypotheses.add(project, dict(statement="Claim"))
    run = project / "runs" / "old"
    run.mkdir(parents=True)
    rows = [
        dict(event="idea", round=1, hypothesis_id="h1", hypothesis="Claim"),
        dict(event="result", round=1, verdict="worse"),
        dict(event="done", reason="finished"),
    ]
    path = run / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in rows))
    frames = list(server.event_stream(path, project=project, task=TASK))
    evidence = [json.loads(frame.split(b"data: ")[1]) for frame in frames]
    assert [e["hypotheses"][0]["status"] for e in evidence] == [
        "testing",
        "testing",
        "refuted",
    ]
    assert list(server.event_stream(path, after=3)) == []
    with path.open("ab") as stream:
        stream.write(b'{"uncommitted')
    before = path.read_bytes()
    with pytest.raises(ValueError, match="read-only"):
        p.Journal(run).append("reject")
    assert path.read_bytes() == before


def test_math_template_enforces_seconds_even_if_search_ignores_deadline(tmp_path):
    source = (Path(__file__).parents[1] / "templates/no-three-ap/search.py").read_text()
    start, end = source.index("def search("), source.index("START = ")
    source = (
        source[:start]
        + "def search(rng, deadline):\n    report([0])\n    while True:\n        pass\n\n\n"
        + source[end:]
    )
    source = source.replace("BUDGET_S = 6.0", "BUDGET_S = 0.03")
    path = tmp_path / "search.py"
    path.write_text(source)
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=3
    )
    assert result.returncode == 0, result.stderr
    curve, final = result.stdout.splitlines()
    point = json.loads(curve)
    assert 0 <= point["elapsed_s"] < 0.03 and "t" not in point
    measured = json.loads(final.removeprefix("RESULT "))
    assert measured["size"] == 1 and 0.02 <= measured["seconds"] < 1
