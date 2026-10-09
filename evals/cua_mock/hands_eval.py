"""Held-out check of the hands and the perception layer, with no model in it.

The mock tasks in TASKS.md mix planning, negotiation and judgement, so a
failed run does not say which layer failed. Here a fixed script plays the
brain: it names every target by role and label, resolves it from the
observation it was just given, and a page-side oracle says what really
happened. A failure is then the hands' or the perception's, and the step
record says which.

    python evals/cua_mock/hands_eval.py [--reps 20] [--only form,twin] [--out report.json]

Needs Google Chrome and the Accessibility grant; it opens its own windows on
127.0.0.1 and closes them. Fixtures (sites/hands/):

* form     long form: exact values (Unicode, several lines), a field inside
           a nested scroller, a popup, one submission and no other field touched;
* dynamic  a page that moves by itself: a click that does nothing is not
           reported as confirmed, a late total is read before the commit;
* board    a drag reorders a list; a painted (canvas) control is absent from
           the observation, so nothing is sent at it;
* twin     two windows with one title: input reaches only the one named.
* textedit a native document: exact text on disk without Save being asked
           for, a Save panel opened in the background and cancelled, a
           draft closed and deleted, nothing else written.
* finder   a folder window: a file scrolled out sideways is brought into
           view and selected (its sibling is not), and Escape gets the
           window back from the inline rename editor with nothing renamed.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rapid_mlx.computer_use import backend, perception
from rapid_mlx.computer_use.errors import ComputerUseError

HERE = Path(__file__).resolve().parent
APP = "Google Chrome"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
NOTES = "Línea uno — café\n第二行 🚀\nlast line"


class StepError(Exception):
    """A step did not do what the script needed; ``layer`` says whose it is."""

    def __init__(self, layer: str, message: str):
        super().__init__(message)
        self.layer = layer


class Run:
    """One repetition of one fixture: a session, its windows, its step log."""

    def __init__(self, base: str, scratch: str = ""):
        self.base = base
        self.scratch = scratch  # a folder of this run's own for files
        self.session = perception.PerceptionSession()
        self.steps: list[dict] = []
        self.windows: list[str] = []
        self.apps: dict[str, str] = {}  # window id -> the app it belongs to

    # -- the oracle ------------------------------------------------------------

    def _post(self, path: str) -> None:
        req = urllib.request.Request(
            self.base + path, b"{}", {"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5).read()

    def events(self, kind: str | None = None) -> list[dict]:
        with urllib.request.urlopen(self.base + "/api/state", timeout=5) as response:
            events = json.loads(response.read())["events"]
        return [e for e in events if kind is None or e.get("type") == kind]

    # -- windows ---------------------------------------------------------------

    def open(self, page: str, title: str) -> str:
        """Open ``page`` in a window of its own and return that window's id."""
        before = {str(w["window_id"]) for w in _windows()}
        subprocess.run(
            [CHROME, "--new-window", f"{self.base}/hands/{page}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for window in _windows():
                wid = str(window["window_id"])
                if wid not in before and title in str(window.get("title", "")):
                    self.windows.append(wid)
                    return wid
            time.sleep(0.2)
        raise StepError("setup", f"no new window titled {title!r} for {page}")

    def adopt(self, app: str, title: str) -> str:
        """The id of ``app``'s window titled ``title``, once it is there."""
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for window in _windows(app):
                if window.get("title") == title:
                    wid = str(window["window_id"])
                    try:
                        # Listed a moment before its accessibility window is.
                        self.session.observe(app, wid)
                    except ComputerUseError:
                        continue
                    self.windows.append(wid)
                    self.apps[wid] = app
                    return wid
            time.sleep(0.2)
        raise StepError("setup", f"{app} shows no window titled {title!r}")

    def close_all(self) -> None:
        for wid in self.windows:
            try:
                # The close button, not Cmd+W: a window a failed step left
                # behind may not hold the focus a chord needs.
                rows = self.session.observe(self.apps.get(wid, APP), wid).rows
                button = next(r for r in rows if r.subrole == "AXCloseButton")
                self.session.act("click", button.ref)
            except (ComputerUseError, StopIteration):
                pass
        self.windows.clear()
        self.session.release_awake()

    # -- what the script can do --------------------------------------------------

    def observe(self, wid: str) -> perception.Observation:
        obs = self.session.observe(self.apps.get(wid, APP), wid)
        self.steps.append(
            {
                "op": "observe",
                "observation": obs.obs_id,
                "rows": len(obs.rows),
                "chars": len(obs.render()),
                "truncated": obs.truncated,
                "ms": obs.elapsed_ms,
            }
        )
        return obs

    def find(
        self, obs: perception.Observation, role: str, label: str
    ) -> perception.Row:
        """The one row with this role and label; none or several is a
        perception failure (the script, like a brain, has nothing else)."""
        rows = [r for r in obs.rows if r.role == role and r.label == label]
        if len(rows) != 1:
            raise StepError(
                "perception", f"{len(rows)} rows are {role} {label!r} in {obs.obs_id}"
            )
        return rows[0]

    def act(
        self, wid: str, op: str, role: str | None = None, label: str = "", **kw: Any
    ) -> dict:
        obs = self.observe(wid)
        row = self.find(obs, role, label) if role else None
        if "to" in kw:
            kw["to"] = self.find(obs, *kw["to"]).ref
        if row is None:
            kw["window_id"] = wid
        step: dict[str, Any] = {"op": op, "target": f"{role} {label!r}" if role else ""}
        try:
            out = self.session.act(op, row.ref if row else None, **kw)
        except ComputerUseError as exc:
            step["error"] = exc.code
            self.steps.append(step)
            raise StepError("delivery", f"{op} {label!r}: {exc.code}: {exc.message}")
        receipt = out["receipt"]
        step.update(
            {
                key: receipt[key]
                for key in ("effect", "settled", "acted_ms", "route", "mode", "error")
                if key in receipt
            }
        )
        step["frontmost_changed"] = "frontmost_changed" in receipt
        self.steps.append(step)
        return out

    def wait(
        self, wid: str, text: str, timeout: float = 10.0
    ) -> perception.Observation:
        started = time.monotonic()
        out = self.session.wait(wid, until_text=text, timeout=timeout)
        self.steps.append(
            {
                "op": "wait",
                "target": text,
                "met": out["met"],
                "ms": round((time.monotonic() - started) * 1000),
            }
        )
        if not out["met"]:
            raise StepError("verification", f"{text!r} never showed")
        return out["observation"]


def _windows(app: str = APP) -> list[dict]:
    try:
        return backend.list_windows(app)
    except ComputerUseError:
        return []


def _confirmed(out: dict, what: str) -> None:
    effect = out["receipt"]["effect"]
    if effect != "confirmed":
        raise StepError("verification", f"{what}: effect {effect}, not confirmed")


def _expect(ok: bool, layer: str, message: str) -> None:
    if not ok:
        raise StepError(layer, message)


# -- fixtures ------------------------------------------------------------------


def form(run: Run, rng: random.Random) -> None:
    company = f"Ørsted & Søn {rng.randrange(1000, 9999)}"
    taxid = f"TX-{rng.randrange(10**6, 10**7)}"
    country = rng.choice(["Canada", "Chile", "China"])
    wid = run.open("form.html", "Supplier intake")
    run.wait(wid, "Submit intake")
    wanted = [
        ("AXTextField", "Company name", company),
        ("AXTextField", "Contact email", "ap@example.test"),
        ("AXTextArea", "Notes", NOTES),
        ("AXTextField", "Tax ID", taxid),
    ]
    rng.shuffle(wanted)
    for role, label, text in wanted:
        _confirmed(run.act(wid, "fill", role, label, text=text), f"fill {label}")
    run.act(wid, "click", "AXPopUpButton", "Country", menu_item=country)
    run.act(wid, "click", "AXCheckBox", "Accept terms")
    run.act(wid, "click", "AXButton", "Submit intake")
    run.wait(wid, "Intake received")
    sent = run.events("form_submit")
    _expect(len(sent) == 1, "delivery", f"{len(sent)} submissions, not 1")
    got = sent[0]
    want = {
        "company": company,
        "email": "ap@example.test",
        "notes": NOTES,
        "country": country,
        "taxid": taxid,
        "terms": True,
        "other": [],
    }
    wrong = {k: got.get(k) for k, v in want.items() if got.get(k) != v}
    _expect(not wrong, "delivery", f"submitted values differ: {wrong}")


def dynamic(run: Run, rng: random.Random) -> None:
    wid = run.open("dynamic.html", "Checkout review")
    run.wait(wid, "Place order")
    # The clock and the viewer count change whatever is done: a button that
    # does nothing must not come back as confirmed.
    idle = run.act(wid, "click", "AXButton", "Refresh estimate")
    _expect(
        idle["receipt"]["effect"] != "confirmed",
        "verification",
        "a click with no effect of its own was reported confirmed",
    )
    express = rng.random() < 0.5
    total = "$58.00" if express else "$48.00"
    if express:
        run.act(wid, "click", "AXButton", "Add express shipping")
        run.wait(wid, total)
    obs = run.observe(wid)
    _expect(
        any(r.label == total for r in obs.rows),
        "perception",
        f"the total {total} is not in the observation",
    )
    place = run.find(obs, "AXButton", "Place order")
    if "disabled" in place.states:
        # Enabled a moment after load: wait for it rather than press blind.
        deadline = time.monotonic() + 5
        while "disabled" in place.states and time.monotonic() < deadline:
            time.sleep(0.3)
            place = run.find(run.observe(wid), "AXButton", "Place order")
    _expect("disabled" not in place.states, "perception", "Place order stays disabled")
    run.act(wid, "click", "AXButton", "Place order")
    run.wait(wid, "Order placed")
    placed = run.events("order_placed")
    _expect(len(placed) == 1, "delivery", f"{len(placed)} orders, not 1")
    _expect(
        placed[0]["total"] == (58 if express else 48),
        "verification",
        f"charged {placed[0]['total']} with {total} read",
    )
    _expect(len(run.events("estimate_clicked")) == 1, "delivery", "estimate click lost")


def board(run: Run, rng: random.Random) -> None:
    wid = run.open("board.html", "Launch board")
    run.wait(wid, "Delta")
    names = ["Alpha", "Bravo", "Charlie", "Delta"]
    moved, onto = rng.sample(names, 2)
    while names.index(onto) == names.index(moved) + 1:  # already before it
        moved, onto = rng.sample(names, 2)
    want = [n for n in names if n != moved]
    want.insert(want.index(onto), moved)
    run.act(wid, "drag", "AXStaticText", moved, to=("AXStaticText", onto))
    run.wait(wid, ", ".join(want))
    orders = [e["order"] for e in run.events("reordered")]
    _expect(orders == [want], "delivery", f"orders {orders}, wanted [{want}]")
    # The painted control has no row: the script has nothing to aim at, and
    # sending a guess is exactly what must not happen.
    obs = run.observe(wid)
    _expect(
        not any("Ignite" in r.label for r in obs.rows),
        "perception",
        "a painted control was reported as a row",
    )
    run.steps.append({"op": "unsupported", "target": "canvas 'Ignite'"})
    _expect(not run.events("canvas_click"), "delivery", "input reached the canvas")


def twin(run: Run, rng: random.Random) -> None:
    first = run.open("twin.html?copy=A", "Twin notes")
    second = run.open("twin.html?copy=B", "Twin notes")
    by_copy: dict[str, str] = {}
    for wid in (first, second):
        obs = run.wait(wid, "Copy ")
        copy = next(r.label[-1] for r in obs.rows if r.label.startswith("Copy "))
        by_copy[copy] = wid
    _expect(sorted(by_copy) == ["A", "B"], "perception", f"copies seen: {by_copy}")
    target = rng.choice(["A", "B"])
    other = "B" if target == "A" else "A"
    text = f"note {rng.randrange(10**5)} für {target}"
    wid = by_copy[target]
    _confirmed(run.act(wid, "fill", "AXTextField", "Note", text=text), "fill Note")
    # Keys take the route a click does not: the same question again.
    run.act(wid, "key", "AXTextField", "Note", key="Tab")
    run.act(wid, "click", "AXButton", "Save note")
    run.wait(wid, f"Saved in copy {target}")
    saved = run.events("twin_saved")
    _expect(
        [(e["copy"], e["value"]) for e in saved] == [(target, text)],
        "delivery",
        f"saved {saved}",
    )
    stray = [
        e
        for kind in ("twin_input", "twin_key", "twin_saved")
        for e in run.events(kind)
        if e["copy"] == other
    ]
    _expect(not stray, "targeting", f"copy {other} received input: {stray[:3]}")
    stolen = [s for s in run.steps if s.get("frontmost_changed")]
    _expect(not stolen, "delivery", f"{len(stolen)} steps changed the frontmost app")


def textedit(run: Run, rng: random.Random) -> None:
    folder = Path(run.scratch) / "textedit"
    folder.mkdir(parents=True, exist_ok=True)
    # Never removed here: a file taken from under an open document makes
    # TextEdit put up an alert that outlives the run.
    before = {f.name for f in folder.iterdir()}
    name = f"memo-{rng.randrange(10**6)}.txt"
    path = folder / name
    path.write_text("first draft\n", encoding="utf-8")
    subprocess.run(["open", "-g", "-a", "TextEdit", str(path)], check=True, timeout=20)
    wid = run.adopt("TextEdit", name)
    text = f"{NOTES}\nref {rng.randrange(10**6)}"
    _confirmed(run.act(wid, "fill", "AXTextArea", "first draft", text=text), "fill")
    if not _settles(lambda: path.read_text(encoding="utf-8") == text, 3):
        run.act(wid, "key", key="cmd+s")
    _expect(
        _settles(lambda: path.read_text(encoding="utf-8") == text, 5),
        "delivery",
        f"the file holds {path.read_text(encoding='utf-8')!r}",
    )
    # A new document: its Save panel opens, and Cancel leaves nothing behind.
    made = run.act(wid, "key", key="cmd+n")["receipt"].get("new_windows") or []
    _expect(len(made) == 1, "delivery", f"New made {len(made)} windows")
    draft = made[0]
    run.windows.append(draft)
    run.apps[draft] = "TextEdit"
    obs = run.observe(draft)
    body = next((r for r in obs.rows if r.role == "AXTextArea"), None)
    _expect(body is not None, "perception", "the new document has no text area")
    _confirmed(run.session.act("fill", body.ref, text="unsaved body"), "fill draft")
    run.act(draft, "key", key="cmd+s")
    obs = run.wait(draft, "Save As:")
    _expect(
        any(r.role == "AXSheet" for r in obs.rows), "perception", "no sheet is reported"
    )
    run.act(draft, "click", "AXButton", "Cancel")
    _expect(
        _settles(lambda: not _has(run.observe(draft), "AXSheet"), 5),
        "verification",
        "the Save panel stays after Cancel",
    )
    run.act(draft, "key", key="cmd+w")
    run.wait(draft, "Delete")
    gone = run.act(draft, "click", "AXButton", "Delete")
    _expect(gone["observation"].closed, "verification", "the draft window stays")
    run.windows.remove(draft)
    shut = run.act(wid, "key", key="cmd+w")
    _expect(shut["observation"].closed, "verification", "the memo window stays")
    run.windows.remove(wid)
    stray = sorted({f.name for f in folder.iterdir()} - before - {name})
    _expect(not stray, "delivery", f"the run also wrote {stray}")
    _expect(path.read_text(encoding="utf-8") == text, "delivery", "the memo changed")
    stolen = [s for s in run.steps if s.get("frontmost_changed")]
    _expect(not stolen, "delivery", f"{len(stolen)} steps changed the frontmost app")


def finder(run: Run, rng: random.Random) -> None:
    name = f"shelf-{rng.randrange(10**6)}"
    folder = Path(run.scratch) / "finder" / name
    folder.mkdir(parents=True, exist_ok=True)
    files = {f"{word}-{rng.randrange(1000)}.txt": word for word in ("alpha", "bravo")}
    for file, body in files.items():
        (folder / file).write_text(body, encoding="utf-8")
    subprocess.run(["open", "-g", str(folder)], check=True, timeout=20)
    wid = run.adopt("Finder", name)
    picked, other = rng.sample(sorted(files), 2)

    def named(label: str) -> perception.Row:
        return run.find(run.observe(wid), "AXTextField", label)

    # A column view opens with the new folder pushed out to the right.
    for _ in range(4):
        if named(picked).on_screen:
            break
        run.act(wid, "scroll", direction="right", pages=3)
    _expect(named(picked).on_screen, "perception", f"{picked} never came into view")
    run.act(wid, "click", "AXTextField", picked)
    _expect(
        "selected" in named(picked).states, "verification", "the file is not selected"
    )
    _expect(
        "selected" not in named(other).states, "targeting", "its sibling is selected"
    )
    # Return opens the inline rename editor, which takes the window's keys;
    # Escape must give the window back with nothing renamed.
    run.act(wid, "key", "AXTextField", picked, key="Return")
    run.act(wid, "key", key="Escape")
    _expect(
        _settles(lambda: named(picked).on_screen, 3),
        "perception",
        "the file row is gone",
    )
    shut = run.act(wid, "key", key="cmd+w")
    _expect(shut["observation"].closed, "verification", "the folder window stays")
    run.windows.remove(wid)
    held = {f.name: f.read_text(encoding="utf-8") for f in folder.iterdir()}
    _expect(held == files, "delivery", f"the folder holds {held}")


def _settles(check: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.25)
    return check()


def _has(obs: perception.Observation, role: str) -> bool:
    return any(row.role == role for row in obs.rows)


FIXTURES: dict[str, Callable[[Run, random.Random], None]] = {
    "form": form,
    "dynamic": dynamic,
    "board": board,
    "twin": twin,
    "textedit": textedit,
    "finder": finder,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--only", default=",".join(FIXTURES))
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--scratch",
        default="/private/tmp/cua-hands-eval",
        help="folder the native fixtures create (and empty) for their files",
    )
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    server = subprocess.Popen(
        [sys.executable, str(HERE / "server.py"), "--port", str(args.port)]
    )
    results: list[dict] = []
    try:
        time.sleep(0.8)
        for name in args.only.split(","):
            for rep in range(args.reps):
                Path(args.scratch).mkdir(parents=True, exist_ok=True)
                run = Run(base, args.scratch)
                run._post("/api/reset")
                rng = random.Random(f"{args.seed}:{name}:{rep}")
                verdict: dict[str, Any] = {"fixture": name, "rep": rep, "ok": True}
                started = time.monotonic()
                try:
                    FIXTURES[name](run, rng)
                except StepError as exc:
                    verdict.update(ok=False, layer=exc.layer, why=str(exc))
                except ComputerUseError as exc:
                    verdict.update(
                        ok=False, layer="perception", why=f"{exc.code}: {exc.message}"
                    )
                finally:
                    run.close_all()
                verdict["s"] = round(time.monotonic() - started, 1)
                verdict["steps"] = run.steps
                results.append(verdict)
                mark = "ok  " if verdict["ok"] else "FAIL"
                why = (
                    "" if verdict["ok"] else f"  [{verdict['layer']}] {verdict['why']}"
                )
                print(f"{mark} {name} #{rep} {verdict['s']}s{why}", flush=True)
    finally:
        server.terminate()
    failed = [r for r in results if not r["ok"]]
    for name in dict.fromkeys(r["fixture"] for r in results):
        mine = [r for r in results if r["fixture"] == name]
        acts = [s for r in mine for s in r["steps"] if "acted_ms" in s]
        seen = [s for r in mine for s in r["steps"] if s["op"] == "observe"]
        print(
            f"{name}: {sum(r['ok'] for r in mine)}/{len(mine)} passed; "
            f"{len(acts)} actions, median {_median(s['acted_ms'] for s in acts)} ms; "
            f"observations up to {max((s['chars'] for s in seen), default=0)} chars"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return 1 if failed else 0


def _median(values: Any) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0


if __name__ == "__main__":
    sys.exit(main())
