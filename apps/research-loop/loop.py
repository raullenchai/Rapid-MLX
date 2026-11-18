"""Minimal autoresearch loop on MLX.

idea -> code (small edits) -> run (fixed budget, several seeds) -> verdict by the numbers -> next idea.
A task is a folder with program.md (the goal, in words), task.json (metric, budget, fixed lines)
and one editable file. Everything the loop does is appended to <run_dir>/events.jsonl.

usage: python loop.py <task_dir> <run_dir> --base-url http://127.0.0.1:8000/v1 --model default --rounds 10
Human input: append {"hypothesis": "...", "change": "..."} lines to <run_dir>/inbox.jsonl (change optional);
create <run_dir>/PAUSE to pause between steps, <run_dir>/STOP to stop after the current step.
"""
import argparse, atexit, difflib, json, os, re, statistics, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("run_dir")
ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1"); ap.add_argument("--model", default="default")
ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--python", default=sys.executable)
args = ap.parse_args()

task_cfg = json.load(open(f"{args.task}/task.json"))
program = open(f"{args.task}/program.md").read()
EDITABLE, METRIC, NOISE = task_cfg["editable"], task_cfg["metric"], task_cfg["noise"]
SIGN = 1 if task_cfg["direction"] == "max" else -1
out = os.path.abspath(args.run_dir)
os.makedirs(out, exist_ok=True)
best_src = open(f"{args.task}/{EDITABLE}").read()
# Atomically take the project queue once; ideas arriving during a run use its inbox.
queued_path = f"{args.task}/ideas.jsonl"
if os.path.isfile(queued_path):
    os.replace(queued_path, f"{out}/queued_ideas.jsonl")
    with open(f"{out}/inbox.jsonl", "a") as inbox:
        for line in open(f"{out}/queued_ideas.jsonl"):
            if line.strip():
                idea = json.loads(line)
                idea["source"] = "you"
                inbox.write(json.dumps(idea) + "\n")
history, inbox_seen = [], 0
def retain_untried_ideas():
    """Carry unconsumed human ideas into the next run, including after a stop."""
    path = f"{out}/inbox.jsonl"
    if os.path.isfile(path):
        remaining = [line for line in open(path).read().splitlines() if line.strip()][inbox_seen:]
        if remaining:
            with open(queued_path, "a") as queue:
                queue.write("\n".join(remaining) + "\n")

atexit.register(retain_untried_ideas)
SYSTEM = "You are a careful ML research assistant running small experiments on a laptop. Be concrete and brief."

def log(event, **kw):
    rec = {"ts": round(time.time(), 3), "event": event, **kw}
    with open(f"{out}/events.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if event != "curve":
        print(event, json.dumps(kw)[:240], flush=True)

def gate():
    while os.path.exists(f"{out}/PAUSE"):
        time.sleep(1)
    if os.path.exists(f"{out}/STOP"):
        log("done", best=best, reason="stopped by researcher"); sys.exit(0)

def chat(prompt, max_tokens=4000):
    body = json.dumps({"model": args.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.6}).encode()
    req = urllib.request.Request(f"{args.base_url}/chat/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0]["message"].get("content") or "", round(time.time() - t, 1), r.get("usage", {})

def last_json(text):
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{.*\})", text, re.S)
    return json.loads(blocks[-1])

EDIT_RULES = ("Reply only with edit blocks, one per change, exactly in this form:\n"
              "<<<<<<< SEARCH\n<exact lines from the current file>\n=======\n<replacement lines>\n>>>>>>> REPLACE\n"
              "SEARCH text must match the file exactly (including indentation). Keep edits minimal.")

def apply_edits(src, text):
    blocks = re.findall(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n?>>>>>>> REPLACE", text, re.S)
    if not blocks:
        return None, "no edit blocks found"
    for old, new in blocks:
        if src.count(old) != 1:
            return None, f"SEARCH text found {src.count(old)} times (must be exactly once):\n{old[:300]}"
        src = src.replace(old, new)
    bad = [f for f in task_cfg["fixed"] if not re.search(f, src, re.M)]
    if bad:
        return None, f"edits changed lines that are fixed by the task: {bad}"
    return src, None

def edit(src, instruction, rnd, kind):
    """Ask for edits; retry once with the error. Returns new source or None."""
    prompt = f"{program}\n\nCurrent {EDITABLE}:\n```python\n{src}\n```\n\n{instruction}\n\n{EDIT_RULES}"
    for attempt in range(2):
        text, secs, usage = chat(prompt)
        new, err = apply_edits(src, text)
        if new:
            diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True), f"a/{EDITABLE}", f"b/{EDITABLE}"))
            log(kind, round=rnd, attempt=attempt, seconds=secs, usage=usage, diff=diff)
            return new
        log("edit_rejected", round=rnd, attempt=attempt, reason=err)
        prompt += f"\n\nYour previous reply could not be applied: {err}\nTry again."
    return None

def run(src, rnd):
    path = f"{out}/round{rnd}_{EDITABLE}"
    open(path, "w").write(src)
    per_seed = []
    for seed in task_cfg["seeds"]:
        gate()
        log("run_start", round=rnd, seed=seed)
        env = {**os.environ, **task_cfg.get("env", {}), "SEED": str(seed), "CURVE_OUT": f"{out}/round{rnd}_seed{seed}_curve.json"}
        p = subprocess.Popen([args.python, path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=out)
        result, points = None, []
        for line in p.stdout:
            line = line.strip()
            if line.startswith("RESULT "):
                result = json.loads(line[7:])
            elif line.startswith("{"):
                try:
                    point = json.loads(line); points.append(point); log("curve", round=rnd, seed=seed, **point)
                except json.JSONDecodeError: pass
        p.wait()
        if points and not os.path.exists(env["CURVE_OUT"]):
            json.dump(points, open(env["CURVE_OUT"], "w"))
        if p.returncode or result is None:
            return None, p.stderr.read()[-2000:]
        per_seed.append(result)
        log("run_end", round=rnd, seed=seed, result=result)
    return {METRIC: round(statistics.mean(r[METRIC] for r in per_seed), 4), "per_seed": per_seed}, None

def history_text():
    def fields(r):
        return ", ".join(f"{k} {v}" for k, v in r.items() if isinstance(v, (int, float, type(None))) and not isinstance(v, bool) and k not in ("seconds", "steps"))
    base = "; ".join(fields(r) for r in baseline_seeds)
    lines = [f"- Baseline (unchanged file): {METRIC} {baseline_value} | per seed: {base}"]
    if not history:
        return lines[0]
    def detail(h):
        seeds = h.get("per_seed") or []
        extra = "; ".join(fields(r) for r in seeds)
        return f" | per seed: {extra}" if extra else ""
    return "\n".join(lines + [f"- Round {h['round']} ({h['source']}): {h['hypothesis']} | change: {h['change']} | {METRIC}: {h.get('value')} (best before: {h['best_before']}) | verdict: {h['verdict']}{detail(h)}" for h in history])

def next_idea(rnd):
    global inbox_seen
    path = f"{out}/inbox.jsonl"
    if os.path.exists(path):
        lines = [l for l in open(path).read().splitlines() if l.strip()]
        if len(lines) > inbox_seen:
            idea = json.loads(lines[inbox_seen]); inbox_seen += 1
            idea.setdefault("change", ""); idea.setdefault("why", "Proposed by the researcher."); idea.setdefault("prediction", "")
            idea["source"] = "you"
            return idea, 0, {}
    prompt = (f"{program}\n\nCurrent best {EDITABLE}:\n```python\n{best_src}\n```\n\nBest {METRIC} so far: {best}\n\nExperiments so far:\n{history_text()}\n\n"
              "Propose the single most promising next hypothesis. Learn from the results above: do not repeat a refuted idea. "
              'Reply with one JSON object in a ```json block: {"hypothesis": "<one testable sentence, at most 15 words>", "why": "<at most 30 words>", '
              '"change": "<the exact change, as short as possible, e.g. WEIGHT_DECAY 0.01 -> 1.0>", "prediction": "<expected effect on the metric, at most 12 words>"}')
    text, secs, usage = chat(prompt, 2000)
    idea = last_json(text); idea["source"] = "model"
    tried = {norm(h["change"]) for h in history}
    if norm(idea.get("change", "")) in tried:  # one retry: the model must not repeat an experiment
        text, secs2, usage = chat(prompt + f"\n\nYou proposed `{idea['change']}`, which was already tested. Propose a different experiment.", 2000)
        idea = last_json(text); idea["source"] = "model"; secs += secs2
    return idea, secs, usage

def norm(change):
    return re.sub(r"\s+", "", str(change)).lower()

best_result, err = run(best_src, 0)
if not best_result:
    log("run_error", round=0, error=err); sys.exit(1)
best = best_result[METRIC]
baseline_value, baseline_seeds = best, best_result["per_seed"]
log("baseline", result=best_result, metric=METRIC, task=task_cfg["title"])
parent = 0  # round whose code the next experiment starts from

for rnd in range(1, args.rounds + 1):
    gate()
    try:
        idea, secs, usage = next_idea(rnd)
    except Exception as e:
        log("idea_error", round=rnd, error=str(e)[:300]); continue
    log("idea", round=rnd, parent=parent, seconds=secs, usage=usage, **{k: idea.get(k, "") for k in ("hypothesis", "why", "change", "prediction", "source")})
    gate()
    src = edit(best_src, f"Implement exactly this change and nothing else: {idea['change'] or idea['hypothesis']}\n(Hypothesis: {idea['hypothesis']})", rnd, "code")
    entry = {"round": rnd, "parent": parent, "source": idea["source"], "hypothesis": idea["hypothesis"], "change": idea["change"], "best_before": best}
    if not src:
        entry["verdict"] = "not run"; history.append(entry); log("result", round=rnd, verdict="not run", reason="could not apply the change"); continue
    result, err = run(src, rnd)
    for fix in range(2):
        if result:
            break
        log("run_error", round=rnd, attempt=fix, error=err)
        fixed = edit(src, f"Running this file failed with:\n```\n{err}\n```\nFix the error without changing the idea being tested ({idea['hypothesis']}).", rnd, "fix")
        if not fixed:
            break
        src = fixed
        result, err = run(src, rnd)
    if not result:
        entry["verdict"] = "crashed"; history.append(entry); log("result", round=rnd, verdict="crashed", error=(err or "")[-300:]); continue
    value = result[METRIC]
    delta = SIGN * (value - best)
    verdict = "supported" if delta > NOISE else ("refuted" if delta < -NOISE else "no effect")
    entry.update(value=value, verdict=verdict, per_seed=result["per_seed"]); history.append(entry)
    log("result", round=rnd, result=result, value=value, best_before=best, delta=round(value - best, 4), verdict=verdict)
    if verdict == "supported":
        best, best_src, parent = value, src, rnd
        open(f"{out}/best_{EDITABLE}", "w").write(src)
    if SIGN * best >= SIGN * task_cfg.get("goal", float("inf") * SIGN):
        log("goal_reached", round=rnd, best=best); break
json.dump(history, open(f"{out}/history.json", "w"), indent=1)
log("done", best=best)
