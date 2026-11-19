"""Minimal autoresearch loop on MLX.

Discussion -> hypotheses -> experiments -> numerical evidence for each hypothesis.
A task is a folder with program.md (the goal, in words), task.json (metric, budget, fixed lines)
and one editable file. Everything the loop does is appended to <run_dir>/events.jsonl.

usage: python loop.py <task_dir> <run_dir> --base-url http://127.0.0.1:8000/v1 --model default --rounds 10
Human input: add claims to the project hypotheses.jsonl via the web API;
create <run_dir>/PAUSE to pause between steps, <run_dir>/STOP to stop after the current step.
"""

import argparse
import difflib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import hypotheses
import literature
import protocol

ap = argparse.ArgumentParser()
ap.add_argument("task")
ap.add_argument("run_dir")
ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--model", default="default")
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--embeddings")
ap.add_argument("--embedding-model", default=literature.DEFAULT_MODEL)
args = ap.parse_args()
CHILD_GROUPS = set()


def hub_shutdown(signum, frame):
    """Hub terminates the plugin group; experiments have their own groups."""
    for pid in tuple(CHILD_GROUPS):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


if os.environ.get("RAPID_HUB_URL"):
    signal.signal(signal.SIGTERM, hub_shutdown)

library = literature.Literature(
    literature.HTTPEmbedder(args.embeddings, args.embedding_model)
    if args.embeddings
    else None
)

task_cfg = json.load(open(f"{args.task}/task.json"))
program = open(f"{args.task}/program.md").read()
EDITABLE, METRIC, NOISE = task_cfg["editable"], task_cfg["metric"], task_cfg["noise"]
SIGN = 1 if task_cfg["direction"] == "max" else -1
out = os.path.abspath(args.run_dir)
os.makedirs(out, exist_ok=True)
best_src = open(f"{args.task}/{EDITABLE}").read()
# Project hypotheses stay available across runs and are reread before each experiment.
hypotheses.limit(task_cfg)
hypotheses.migrate_queue(args.task)
history, round_hypotheses = [], {}
attempts, plans, result_records = {}, {}, {}
journal = protocol.Journal(out)
artifacts = protocol.Artifacts(out)
SYSTEM = "You are a careful ML research assistant running small experiments on a laptop. Be concrete and brief."


DEFAULT_TIMEOUT_S = 300  # per seed; override with task.json timeout_s


def log(event, **kw):
    if kw.get("round") in round_hypotheses:
        kw["hypothesis_id"] = round_hypotheses[kw["round"]]
    journal.append(event, **kw)
    if event != "curve":
        print(event, json.dumps(kw)[:240], flush=True)


def gate():
    while os.path.exists(f"{out}/PAUSE"):
        time.sleep(1)
    if os.path.exists(f"{out}/STOP"):
        log("research_ended", best=best, reason="stopped by researcher")
        reports()
        log("done", best=best, reason="stopped by researcher")
        sys.exit(0)


model = protocol.Model(args.base_url, args.model, log, SYSTEM)


def edit(src, instruction, rnd, kind):
    prompt = (
        f"{program}\n{instruction}\n"
        f"Edit the exact current base {EDITABLE} below. All SEARCH blocks must be copied "
        "from this source, including blank lines and indentation. Rejected edits are never applied.\n"
        f"base_sha256: {protocol.digest(src.encode())}\n"
        f"BEGIN CURRENT BASE SOURCE\n{src}END CURRENT BASE SOURCE\n"
        "Return edit line arrays, preserving indentation. Each search block must match uniquely."
    )
    try:
        new = model.message(
            "edit",
            prompt,
            check=lambda message: protocol.apply_edits(
                src, message, task_cfg["fixed"], EDITABLE
            ),
            retries=2,
        )
    except ValueError as exc:
        log("edit_rejected", round=rnd, reason=str(exc))
        return None
    diff = "".join(
        difflib.unified_diff(
            src.splitlines(True), new.splitlines(True), f"a/{EDITABLE}", f"b/{EDITABLE}"
        )
    )
    log(kind, round=rnd, diff=diff, prompt_id=model.prompt_id)
    return new


def owner(rnd, seed=None):
    row = dict(
        scope="experiment" if seed is None else "seed",
        run=Path(out).name,
        experiment=rnd,
        hypothesis_id=round_hypotheses.get(rnd),
        attempt=attempts[rnd],
    )
    if seed is not None:
        row["seed"] = seed
    return row


def attach(
    content,
    id,
    kind,
    title,
    path,
    rnd,
    seed=None,
    role="attachment",
    refs=(),
    producer="harness",
):
    try:
        return artifacts.register(
            content,
            id=id,
            kind=kind,
            title=title,
            path=path,
            owner=owner(rnd, seed),
            role=role,
            refs=refs,
            producer=producer,
            provenance=dict(
                model_id=args.model,
                prompt_id=model.prompt_id,
                code_sha256=plans[rnd]["code_sha256"],
                env=protocol.runtime_env(task_cfg),
            ),
        )
    except (ValueError, OSError):
        return None


def run(src, rnd, execute=True):
    attempt = attempts[rnd] = attempts.get(rnd, -1) + 1
    plan = plans[rnd] = protocol.frozen_record(
        rnd,
        attempt,
        task_cfg,
        src,
        hypothesis_id=round_hypotheses.get(rnd),
        parent_experiment=parent if rnd else None,
        change=idea["change"] if rnd else "Baseline",
        model_id=args.model,
        prompt_id=model.prompt_id,
    )
    log("experiment_record", round=rnd, record=plan)
    if not execute:
        record = protocol.result_record(
            plan, {}, task_cfg["seeds"], best if rnd else None
        )
        result_records[rnd] = record
        log("result_record", round=rnd, record=record)
        return None, "Could not apply the change"
    prefix = f"e{rnd}-a{attempt}"
    path = f"{out}/round{rnd}_{EDITABLE}"
    Path(path).write_text(src)
    parent_code = f"e{parent}-a{attempts.get(parent, 0)}-code" if rnd else None
    attach(
        src.encode(),
        f"{prefix}-code",
        "code",
        EDITABLE,
        EDITABLE,
        rnd,
        role="primary_code",
        refs=[parent_code] if parent_code else [],
    )
    if rnd:
        diff = "".join(
            difflib.unified_diff(
                best_src.splitlines(True),
                src.splitlines(True),
                f"experiment-{parent}",
                f"experiment-{rnd}",
            )
        )
        attach(
            diff.encode(),
            f"{prefix}-diff",
            "code",
            f"Diff vs experiment {parent}",
            "change.diff",
            rnd,
            role="diff",
            refs=[parent_code],
        )
    values, crashed, errors = {}, [], []
    for seed in task_cfg["seeds"]:
        log("run_start", round=rnd, seed=seed, attempt=attempt)
        with tempfile.TemporaryDirectory(prefix="seed-", dir=out) as directory:
            env = {
                **os.environ,
                **task_cfg.get("env", {}),
                "SEED": str(seed),
                "CURVE_OUT": f"{directory}/curve.json",
                "ARTIFACT_DIR": directory,
                "ARTIFACT_ID_PREFIX": f"{prefix}-s{seed}-script",
            }
            p = subprocess.Popen(
                [args.python, path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                cwd=out,
                start_new_session=True,
            )
            CHILD_GROUPS.add(p.pid)
            # Hard wall-clock limit: a script that ignores its own budget is
            # killed (with its whole process group) and recorded as crashed.
            limit = float(task_cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
            timed_out = threading.Event()

            def kill_overrun(p=p, timed_out=timed_out):
                timed_out.set()
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

            watchdog = threading.Timer(limit, kill_overrun)
            watchdog.daemon = True
            watchdog.start()
            stderr_parts, stdout_parts = [], []
            # Drain both pipes concurrently, retaining bounded logs in memory.
            counts = {"stdout": 0, "stderr": 0}

            def drain(p=p, counts=counts, stderr_parts=stderr_parts):
                for line in iter(lambda stream=p.stderr: stream.readline(65536), ""):
                    counts["stderr"] += len(line.encode())
                    if counts["stderr"] <= protocol.ARTIFACT_LIMIT + 1:
                        stderr_parts.append(line)

            thread = threading.Thread(target=drain)
            thread.start()
            result, points = None, []
            for line in iter(lambda stream=p.stdout: stream.readline(65536), ""):
                counts["stdout"] += len(line.encode())
                if counts["stdout"] <= protocol.ARTIFACT_LIMIT + 1:
                    stdout_parts.append(line)
                try:
                    if line.startswith("RESULT "):
                        candidate = json.loads(line[7:])
                        json.dumps(candidate, allow_nan=False)
                        value = candidate.get(METRIC)
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                        ):
                            result = candidate
                    elif line.startswith("{"):
                        point = json.loads(line)
                        if isinstance(point, dict) and all(
                            not isinstance(v, (dict, list)) for v in point.values()
                        ):
                            json.dumps(point, allow_nan=False)
                            points.append(point)
                            log(
                                "curve",
                                round=rnd,
                                seed=seed,
                                point=point,
                                **{
                                    k: v
                                    for k, v in point.items()
                                    if k not in ("round", "seed", "event", "point")
                                },
                            )
                except (ValueError, TypeError, AttributeError):
                    pass
            p.wait()
            CHILD_GROUPS.discard(p.pid)
            watchdog.cancel()
            thread.join()
            if timed_out.is_set():
                result = None
                stderr_parts.append(
                    f"TimeoutError: experiment exceeded the {limit:g} s wall-clock limit\n"
                )
            for role, parts in (("stdout", stdout_parts), ("stderr", stderr_parts)):
                if counts[role] > protocol.ARTIFACT_LIMIT:
                    log(
                        "artifact_rejected",
                        round=rnd,
                        seed=seed,
                        reason=f"{role} exceeds 5 MB quota",
                    )
                else:
                    attach(
                        "".join(parts).encode(),
                        f"{prefix}-s{seed}-{role}",
                        "log",
                        f"{role} · seed {seed}",
                        f"{role}.log",
                        rnd,
                        seed,
                        role,
                    )
            if points:
                data_id = f"{prefix}-s{seed}-curve"
                data = attach(
                    ("\n".join(json.dumps(p) for p in points) + "\n").encode(),
                    data_id,
                    "data",
                    f"Measurements · seed {seed}",
                    "curve.jsonl",
                    rnd,
                    seed,
                    "curve",
                )
                spec = protocol.default_chart(task_cfg.get("curve", {}), data_id)
                if data and spec:
                    attach(
                        json.dumps(spec).encode(),
                        f"{data_id}-chart",
                        "chart",
                        f"Curve · seed {seed}",
                        "curve.chart.json",
                        rnd,
                        seed,
                        "curve",
                        [data_id],
                    )
            artifacts.manifest(
                directory,
                owner(rnd, seed),
                f"{prefix}-s{seed}-script",
                dict(
                    model_id=args.model,
                    prompt_id=model.prompt_id,
                    code_sha256=plan["code_sha256"],
                    env=protocol.runtime_env(task_cfg),
                ),
            )
            if p.returncode or result is None:
                crashed.append(seed)
                errors.append(
                    "".join(stderr_parts)[-2000:] or "Missing valid RESULT metric"
                )
                log("run_error", round=rnd, seed=seed, error=errors[-1])
            else:
                values[seed] = result
                log("run_end", round=rnd, seed=seed, result=result)
    record = protocol.result_record(plan, values, crashed, best if rnd else None)
    result_records[rnd] = record
    log("result_record", round=rnd, record=record)
    if record["verdict"] in ("crashed", "incomplete"):
        return None, "\n".join(errors) or "Could not execute experiment"
    return {METRIC: record["aggregate"], "per_seed": list(values.values())}, None


def history_text():
    def fields(r):
        return ", ".join(
            f"{k} {v}"
            for k, v in r.items()
            if isinstance(v, (int, float, type(None)))
            and not isinstance(v, bool)
            and k not in ("seconds", "steps")
        )

    base = "; ".join(fields(r) for r in baseline_seeds)
    lines = [
        f"- Baseline (unchanged file): {METRIC} {baseline_value} | per seed: {base}"
    ]
    if not history:
        return lines[0]

    def detail(h):
        seeds = h.get("per_seed") or []
        extra = "; ".join(fields(r) for r in seeds)
        return f" | per seed: {extra}" if extra else ""

    return "\n".join(
        lines
        + [
            f"- Experiment {h['round']} ({h['source']}): {h['hypothesis']} | change: {h['change']} | {METRIC}: {h.get('value')} (best before: {h['best_before']}) | verdict: {h['verdict']}{detail(h)}"
            for h in history
        ]
    )


def next_experiment(rnd):
    # Include this run even for direct CLI callers who place it outside project/runs.
    claims = hypotheses.collect(args.task, task_cfg, [Path(out) / "events.jsonl"])
    capacity = hypotheses.limit(task_cfg)
    open_claims = [
        h
        for h in claims
        if h["status"] in ("untested", "testing") and len(h["experiments"]) < capacity
    ]
    pending = next(
        (
            h
            for h in open_claims
            if h["source"] in ("you", "discussion") and h["status"] == "untested"
        ),
        None,
    )
    sources = []
    if library.embedder:
        try:
            query = (
                pending["statement"]
                if pending
                else "; ".join(h["statement"] for h in open_claims) or program
            )
            sources = library.retrieve(args.task, query, program, limit=3)
        except (ValueError, OSError, KeyError) as exc:
            log("literature_error", round=rnd, error=str(exc))
    if sources:
        log(
            "literature_context",
            round=rnd,
            source_ids=[s["chunk_id"] for s in sources],
            sources=sources,
        )
    tried = {norm(e.get("change", "")) for h in claims for e in h["experiments"]}
    if pending and pending.get("change") and norm(pending["change"]) not in tried:
        return (
            dict(
                hypothesis_id=pending["id"],
                hypothesis=pending["statement"],
                why=pending["why"],
                prediction=pending["prediction"],
                change=pending["change"],
                source=pending["source"],
                literature_sources=sources,
            ),
            0,
            {},
        )
    open_text = (
        "; ".join(
            f"{h['id']} ({len(h['experiments'])}/{capacity} experiments): {h['statement']}"
            f" | why: {h['why']} | prediction: {h['prediction']}"
            for h in open_claims
        )
        or "none"
    )
    concluded_text = (
        "; ".join(
            f"{h['id']} ({h['status']}): {h['statement']}"
            for h in claims
            if h not in open_claims
        )
        or "none"
    )
    prompt = (
        f"{program}\n\nCurrent best {EDITABLE}:\n```python\n{best_src}\n```\n\nBest {METRIC} so far: {best}\n\nExperiments so far:\n{history_text()}\n\n"
        f"Open hypotheses: {open_text}\nConcluded hypotheses (do not test): {concluded_text}\n"
        f"Each hypothesis permits at most {capacity} experiments. "
        "A hypothesis is a testable claim about the system, with why and prediction, not a code edit. "
        "Write a new claim as one sentence, at most 30 words. "
        "Choose one new experiment for an open (untested/testing) hypothesis, or a new hypothesis and its first experiment. "
        "Never test a concluded (supported/refuted/inconclusive) hypothesis. Never repeat a tested change. "
        "Experiment verdicts: improved, worse, no effect, crashed. "
        "Use experiment_plan: hypothesis_id for an existing claim, or null with new_hypothesis."
    )
    if library.embedder:
        prompt += literature.context(sources)
    if pending:
        prompt += f"\nResearcher priority: you MUST test {pending['id']}: {pending['statement']}. Return that hypothesis_id."
    tried = {norm(e.get("change", "")) for h in claims for e in h["experiments"]}

    def check(proposal):
        hid = proposal["hypothesis_id"]
        claim = next((h for h in claims if h["id"] == hid), None) if hid else None
        if pending and hid != pending["id"]:
            raise ValueError("Test the researcher-priority hypothesis first.")
        if hid and (
            not claim
            or claim["status"] not in ("untested", "testing")
            or len(claim["experiments"]) >= hypotheses.limit(task_cfg)
        ):
            raise ValueError(
                "Choose an open hypothesis with remaining experiment capacity."
            )
        if not proposal["change"].strip() or norm(proposal["change"]) in tried:
            raise ValueError(
                "Change must be nonempty and different from every tested change."
            )
        if not hid:
            new = proposal["new_hypothesis"]
            if (
                not new
                or not new["statement"].strip()
                or len(new["statement"].split()) > 40
            ):
                raise ValueError(
                    "A new hypothesis must be a claim of at most 40 words."
                )
            if any(norm(h["statement"]) == norm(new["statement"]) for h in claims):
                raise ValueError("This hypothesis already exists.")
        elif proposal["new_hypothesis"] is not None:
            raise ValueError(
                "Existing hypothesis plans must set new_hypothesis to null."
            )
        return proposal

    started = time.time()
    proposal = model.message(
        "experiment_plan",
        prompt,
        check=check,
        schema=protocol.experiment_plan_schema(
            [h["id"] for h in open_claims], pending["id"] if pending else None
        ),
    )
    hid = proposal["hypothesis_id"]
    claim = (
        next((h for h in claims if h["id"] == hid), None)
        if hid
        else hypotheses.add(
            args.task, dict(proposal["new_hypothesis"], sources=sources), "model"
        )
    )
    log("hypothesis", hypothesis=claim)
    return (
        dict(
            proposal,
            hypothesis_id=claim["id"],
            hypothesis=claim["statement"],
            why=claim["why"],
            source=claim["source"],
            literature_sources=sources,
        ),
        round(time.time() - started, 1),
        {},
    )


def norm(change):
    return re.sub(r"\s+", "", str(change)).lower()


def reports():
    claims = hypotheses.collect(args.task, task_cfg, [Path(out) / "events.jsonl"])
    for claim in claims:
        evidence = [e for e in claim["experiments"] if e["run"] == Path(out).name]
        if not evidence:
            continue
        try:

            def same_hypothesis(message, claim=claim):
                if message["hypothesis_id"] != claim["id"]:
                    raise ValueError(
                        "conclusion hypothesis_id must match requested claim"
                    )
                return message

            report = model.message(
                "conclusion",
                f"Write a concise evidence-based conclusion for {claim['id']}. Status is computed by the harness; do not override it.\n{json.dumps(claim)}",
                check=same_hypothesis,
            )
            artifacts.register(
                report["summary_md"].encode(),
                id=f"{claim['id']}-conclusion",
                kind="report",
                role="conclusion",
                title="Hypothesis conclusion",
                path="conclusion.md",
                producer="model",
                owner=dict(
                    scope="hypothesis", run=Path(out).name, hypothesis_id=claim["id"]
                ),
                provenance=dict(
                    model_id=args.model,
                    prompt_id=model.prompt_id,
                    code_sha256=None,
                    env=protocol.runtime_env(task_cfg),
                ),
            )
        except Exception as exc:
            log("report_error", hypothesis_id=claim["id"], error=str(exc))
    try:
        report = model.message(
            "run_report",
            f"Summarize the completed research run, evidence limitations and next steps.\n{history_text()}",
        )
        artifacts.register(
            (
                report["summary_md"] + "\n\nNext steps\n\n" + report["next_steps_md"]
            ).encode(),
            id="run-report",
            kind="report",
            role="summary",
            title="Run report",
            path="report.md",
            producer="model",
            owner=dict(scope="run", run=Path(out).name),
            provenance=dict(
                model_id=args.model,
                prompt_id=model.prompt_id,
                code_sha256=protocol.digest(best_src.encode()),
                env=protocol.runtime_env(task_cfg),
            ),
        )
    except Exception as exc:
        log("report_error", error=str(exc))


best_result, err = run(best_src, 0)
if not best_result:
    log("run_error", round=0, error=err)
    sys.exit(1)
best = best_result[METRIC]
baseline_value, baseline_seeds = best, best_result["per_seed"]
log("baseline", result=best_result, metric=METRIC, task=task_cfg["title"])
parent = 0  # round whose code the next experiment starts from

for rnd in range(1, args.rounds + 1):
    gate()
    try:
        idea, secs, usage = next_experiment(rnd)
    except Exception as e:
        log("hypothesis_error", round=rnd, error=str(e)[:300])
        continue
    round_hypotheses[rnd] = idea["hypothesis_id"]
    log(
        "idea",
        round=rnd,
        parent=parent,
        seconds=secs,
        usage=usage,
        **{
            k: idea.get(k, "")
            for k in ("hypothesis", "why", "change", "prediction", "source")
        },
    )
    gate()
    experiment_sources = []
    if library.embedder:
        try:
            experiment_sources = library.retrieve(
                args.task, idea["hypothesis"], program, limit=3
            )
            log(
                "literature_context",
                round=rnd,
                phase="implementation",
                source_ids=[s["chunk_id"] for s in experiment_sources],
                sources=experiment_sources,
            )
        except (ValueError, OSError, KeyError) as exc:
            log("literature_error", round=rnd, error=str(exc))
    src = edit(
        best_src,
        f"Implement exactly this change and nothing else: {idea['change'] or idea['hypothesis']}\n(Hypothesis: {idea['hypothesis']})"
        + (literature.context(experiment_sources) if library.embedder else ""),
        rnd,
        "code",
    )
    entry = {
        "hypothesis_id": idea["hypothesis_id"],
        "round": rnd,
        "parent": parent,
        "source": idea["source"],
        "hypothesis": idea["hypothesis"],
        "change": idea["change"],
        "best_before": best,
    }
    if not src:
        run(best_src, rnd, execute=False)
        entry["verdict"] = "crashed"
        history.append(entry)
        log("result", round=rnd, verdict="crashed", reason="could not apply the change")
        continue
    result, err = run(src, rnd)
    for fix in range(2):
        if result:
            break
        log("run_error", round=rnd, attempt=fix, error=err)
        fixed = edit(
            src,
            f"Running the crashed candidate shown as CURRENT BASE SOURCE below failed with:\n```\n{err}\n```\nFix that candidate without changing the hypothesis being tested ({idea['hypothesis']}).",
            rnd,
            "fix",
        )
        if not fixed:
            break
        src = fixed
        result, err = run(src, rnd)
    if not result:
        entry["verdict"] = result_records[rnd]["verdict"]
        history.append(entry)
        log(
            "result",
            round=rnd,
            verdict=result_records[rnd]["verdict"],
            error=(err or "")[-300:],
        )
        continue
    value = result[METRIC]
    delta = SIGN * (value - best)
    verdict = (
        "improved" if delta > NOISE else ("worse" if delta < -NOISE else "no effect")
    )
    entry.update(value=value, verdict=verdict, per_seed=result["per_seed"])
    history.append(entry)
    log(
        "result",
        round=rnd,
        result=result,
        value=value,
        best_before=best,
        delta=round(value - best, 4),
        verdict=verdict,
    )
    if verdict == "improved":
        best, best_src, parent = value, src, rnd
        open(f"{out}/best_{EDITABLE}", "w").write(src)
    if SIGN * best >= SIGN * task_cfg.get("goal", float("inf") * SIGN):
        log("goal_reached", round=rnd, best=best)
        break


json.dump(history, open(f"{out}/history.json", "w"), indent=1)
log("research_ended", best=best, reason="experiment budget exhausted or goal reached")
reports()
log("done", best=best)
