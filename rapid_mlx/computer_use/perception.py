"""Perception session: stable refs, observation ids, deltas, receipts.

One ``PerceptionSession`` tracks what the model has seen:

* every element gets a session-wide ref (``e12``) bound to its live AX
  element (CFEqual) and its role, so refs survive re-observation,
  scrolling and edits (a renamed element keeps its ref and its new name is
  reported as a change);
* every observation gets an id (``o7``); an action must name refs from the
  latest observation of its window, otherwise it is refused as stale;
* an observation lists what changed since the previous one of the same
  window (added / removed / changed, capped), and marks new rows with ``*``
  (browser-use) while the full list stays the body (Playwright and Muse
  both fall back to the full tree);
* an action waits for the window to settle (two equal samples), observes,
  and returns a receipt: effect, the changes it caused, window and focus
  transitions;
* an action whose outcome is unknown blocks further input until the window
  is observed again (Muse's "unresolved outcome" rule, relaxed: a fresh
  observation clears it).
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import backend, guards
from .errors import ComputerUseError

MAX_CHANGE_LINES = 20
MAX_LABEL_CHARS = 160  # a product tile button carries name, size and price
MAX_TEXT_CHARS = 300  # static text is content, not a control name
SETTLE_POLL_S = 0.1
SETTLE_MIN_S = 0.3
SETTLE_CAP_S = 1.5
SETTLE_CAP_SLOW_S = 5.0
_SLOW_KEYS = {"return", "enter", "cmd+r", "cmd+l", "cmd+n", "cmd+t", "cmd+w"}
# Keys that press the focused control.
_ACTIVATING_KEYS = {"return", "enter", "space", " ", "kp_enter"}
_PASTE = re.compile(
    r"^(?:(?:shift|option|alt|ctrl)\+)*(?:cmd|command)\+(?:(?:shift|option|alt)\+)*v$",
    re.I,
)
WAIT_POLL_S = 0.5
# A bot reply lands in pieces (bubble, then text, then quick replies).
WAIT_QUIET_S = 2.5
MAX_FOLD_SECTIONS = 12
# Refusals that mean "the page moved under the snapshot", not "the target is gone".
_SHIFTED = {"element_not_found", "stale_observation"}
MAX_FIND_HITS = 25
# Live AX elements remembered for ref stability; past this, elements no
# current observation shows are forgotten.
MAX_REMEMBERED_ELEMENTS = 20000
_BROWSER_POPUPS = {"Chrome", "Tab search", "View site information"}
# A chat partner's typing indicator: the reply is not there yet.
_TYPING = re.compile(r"\bis typing|\btyping(?:…|\.\.\.)|\bis thinking\b", re.I)
OPEN_URL_WAIT_S = 8.0
# Browser-extension toolbar buttons: ~25 rows of noise on every page.
_EXTENSION_BUTTON = re.compile(r"(has|wants) access to this site$|^extensions$", re.I)
_ADDRESS_BAR = re.compile(
    r"address and search|address bar|smart search|search or enter", re.I
)


def _short_role(role: str) -> str:
    return (role[2:] if role.startswith("AX") else role).lower() or "?"


@dataclass
class Row:
    ref: str
    role: str
    label: str
    value: str | None
    states: tuple[str, ...]
    index: int
    center: tuple[int, int]
    new: bool = False
    subrole: str = ""
    on_screen: bool = True  # inside the window's visible area (not scrolled away)
    sliver: bool = False  # visually hidden (clipped to 1 px), not just scrolled away

    def is_browser_noise(self) -> bool:
        return self.role == "AXPopUpButton" and bool(
            _EXTENSION_BUTTON.search(self.label)
        )

    def signature(self) -> tuple:
        return (self.role, self.label, self.value, self.states)

    def render(self) -> str:
        text = f"{'*' if self.new else ' '}{self.ref} {_short_role(self.role)}"
        if self.label:
            cap = MAX_TEXT_CHARS if self.role == "AXStaticText" else MAX_LABEL_CHARS
            text += f' "{self.label[:cap]}"'
        if self.value is not None:
            text += f" = {self.value[:MAX_LABEL_CHARS]!r}"
        if self.states:
            text += f" [{','.join(self.states)}]"
        return text


@dataclass
class Observation:
    obs_id: str
    app: str
    window_id: str
    title: str
    snapshot: dict
    rows: list[Row]
    changes: list[str]
    change_counts: tuple[int, int, int]
    previous: str | None
    elapsed_ms: int
    truncated: bool
    window_ids: list[str] = field(default_factory=list)

    def by_ref(self) -> dict[str, Row]:
        return {row.ref: row for row in self.rows}

    def render(self, *, full: bool = True, everything: bool = False) -> str:
        head = (
            f"observation {self.obs_id} · {self.app} · window {self.window_id}"
            f' "{self.title}" · {len(self.rows)} elements · {self.elapsed_ms} ms'
        )
        if self.truncated:
            head += " · TRUNCATED (budget or cap reached; scroll or narrow)"
        lines = [head]
        if self.previous is not None:
            added, removed, changed = self.change_counts
            lines.append(
                f"changes since {self.previous}: +{added} -{removed} ~{changed}"
            )
            lines.extend(f"  {line}" for line in self.changes)
        if full:
            lines.append("elements:")
            rows = [row for row in self.rows if not row.is_browser_noise()]
            noise = len(self.rows) - len(rows)
            if everything:
                lines.extend(row.render() for row in rows)
            else:
                # The model reads what is on screen, as the user would; a long
                # page (a buy box after 1,400 carousel nodes) is still walked
                # whole, so off-screen rows stay addressable and searchable.
                lines.extend(row.render() for row in rows if row.on_screen)
                off = [row for row in rows if not row.on_screen]
                if off:
                    sections: list[str] = []
                    for row in off:
                        if (
                            row.role == "AXHeading"
                            and not row.sliver
                            and row.label
                            and row.label not in sections
                        ):
                            sections.append(row.label[:60])
                    hint = f"  ({len(off)} more elements off screen"
                    if sections:
                        shown = " · ".join(
                            f'"{s}"' for s in sections[:MAX_FOLD_SECTIONS]
                        )
                        more = len(sections) - MAX_FOLD_SECTIONS
                        hint += f"; sections: {shown}" + (
                            f" +{more} more" if more > 0 else ""
                        )
                    lines.append(
                        hint + "; scroll, or observe with find=<text> or all=true)"
                    )
            if noise:
                lines.append(f"  ({noise} browser-extension buttons hidden)")
        return "\n".join(lines)

    def find(self, text: str) -> str:
        """Rows anywhere in the window (on or off screen) whose text matches."""
        needle = text.lower()
        hits = [
            row
            for row in self.rows
            if not row.is_browser_noise()
            and needle in f"{row.label} {row.value or ''}".lower()
        ]
        lines = [f"find {text!r} in {self.obs_id}: {len(hits)} matches"]
        lines.extend(
            row.render() + ("" if row.on_screen else "  (off screen)")
            for row in hits[:MAX_FIND_HITS]
        )
        if len(hits) > MAX_FIND_HITS:
            lines.append(f"  … {len(hits) - MAX_FIND_HITS} more; narrow the text")
        return "\n".join(lines)

    def texts(self) -> list[str]:
        return [
            row.label if row.value is None else f"{row.label} {row.value}".strip()
            for row in self.rows
            if not row.is_browser_noise()
        ]

    def choices(self) -> list[str]:
        """Options chosen on the page: checked radios and boxes, menu values.

        A radio's label is often only part of the choice ("12–2 PM" under a
        "Sat, Oct 10" legend), so it is prefixed with the nearest text before
        its group.
        """
        out: list[str] = []
        radio_labels = {r.label for r in self.rows if r.role == "AXRadioButton"}
        for i, row in enumerate(self.rows):
            if (
                row.is_browser_noise()
                or "checked" not in row.states
                and row.role != "AXPopUpButton"
            ):
                continue
            if row.role == "AXRadioButton":
                legend = next(
                    (
                        r.label
                        for r in reversed(self.rows[:i])
                        if r.role in {"AXStaticText", "AXHeading"}
                        and r.label
                        and not any(r.label in lab for lab in radio_labels)
                    ),
                    "",
                )
                out.append(f"{legend} {row.label}".strip())
            elif row.role == "AXCheckBox":
                out.append(f"[x] {row.label}")
            elif (
                row.role == "AXPopUpButton"
                and row.value
                and row.label not in _BROWSER_POPUPS
            ):
                out.append(f"{row.label}: {row.value}")
        return out


class PerceptionSession:
    def __init__(self) -> None:
        self._ref_ids = itertools.count(1)
        self._obs_ids = itertools.count(1)
        # live AX element -> (ref, role, label)
        self._refs: dict[object, tuple[str, str, str]] = {}
        self._latest: dict[str, Observation] = {}  # window id -> observation
        self._unresolved: dict[str, str] = {}  # window id -> description
        self._awake: subprocess.Popen | None = None
        self._handed_from: int | None = None  # pid in front before take_front
        # One lock serializes AX work: a handoff waits on one thread while the
        # human channel (approvals, human input) arrives on another.
        self._lock = threading.RLock()
        self._approval_ids = itertools.count(1)
        self._pending: dict[str, dict] = {}  # approval id -> what was asked
        self._approved: dict[str, dict] = {}  # approval id -> granted, unused
        self._with_human: dict[str, str] = {}  # window id -> why
        self._human_done: set[str] = set()
        self.on_event = None  # host hook: callable(kind, payload)

    # -- keeping the screen on ----------------------------------------------

    def keep_awake(self) -> None:
        """Hold a display-sleep assertion for as long as this process lives.

        With the display asleep every window leaves the screen and nothing
        can be read or operated (measured: observe fails until it wakes), so
        a task holds the screen on, as a video player does. ``caffeinate -w``
        drops the assertion by itself when this process exits.
        """
        if self._awake is not None and self._awake.poll() is None:
            return
        try:
            self._awake = subprocess.Popen(
                ["/usr/bin/caffeinate", "-d", "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            # No caffeinate: a sleeping display is still reported by
            # display_asleep, so the task fails loudly rather than blindly.
            self._awake = None

    def release_awake(self) -> None:
        awake, self._awake = self._awake, None
        if awake is None or awake.poll() is not None:
            return
        awake.terminate()
        try:
            awake.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            awake.kill()
            awake.wait()

    # -- handover ----------------------------------------------------------

    def take_front(self, app: str, window_id: str | int) -> dict:
        """Bring one exact window to the front for a handed-over task.

        When the user hands the Mac over, the agent works in the foreground:
        Chromium stops exposing a page that is off screen (another Space,
        minimized), so the task window has to be the visible one. Matches the
        AX window by its CG window id, never by frame (browser windows share
        one size), activates the app and raises that window. The app that was
        in front is remembered for ``hand_back``.
        """
        from . import ax_driver, background_input

        cg_id = backend._cg_window_id(window_id)
        front = _frontmost_pid()
        if self._handed_from is None and front is not None:
            self._handed_from = front
        app_element, info = backend._resolve_app(app, activate=True)

        def matching() -> list:
            return [
                w
                for w in ax_driver._app_windows(app_element)
                if background_input.ax_window_id(w) == cg_id
            ]

        matches = matching()
        if not matches and info.get("pid") is not None:
            # On another Space the window is missing from AXWindows; it is
            # still reachable by remote token.
            ax_driver.discover_remote_windows(int(info["pid"]), {cg_id})
            matches = matching()
        if len(matches) != 1:
            raise ComputerUseError(
                "window_not_found", f"no window {window_id} in {app} to bring forward"
            )
        window = matches[0]
        if ax_driver._get(window, "AXMinimized"):
            ax_driver.AXUIElementSetAttributeValue(window, "AXMinimized", False)
        ax_driver.AXUIElementSetAttributeValue(window, "AXMain", True)
        ax_driver.AXUIElementPerformAction(window, "AXRaise")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if ax_driver.window_is_onscreen(cg_id) and _frontmost_pid() == info.get(
                "pid"
            ):
                return {"window_id": f"cg:{cg_id}", "front": True}
            time.sleep(0.1)
        return {"window_id": f"cg:{cg_id}", "front": False}

    def hand_back(self) -> dict:
        """Return the front to the app the user had before ``take_front``."""
        pid, self._handed_from = self._handed_from, None
        restored = pid is not None and backend._activate_app(pid)
        self.release_awake()
        return {"restored_pid": pid, "restored": restored}

    # -- what only the user may do -----------------------------------------

    def _guard(
        self, op: str, obs: Observation, row: Row | None, kw: dict
    ) -> tuple | None:
        """Raise if the agent may not do this; return an approval it used."""
        if obs.window_id in self._with_human:
            raise ComputerUseError(
                "with_human",
                f"the user is working in window {obs.window_id} "
                f"({self._with_human[obs.window_id]}); wait for the handoff to finish",
            )
        # A printable key is typing too: a password spelled key by key is
        # still the user's.
        key_text = _key_text(kw.get("key")) if op == "key" else None
        # A paste enters text the agent never saw (the user's clipboard).
        pasting = op == "key" and bool(_PASTE.match(str(kw.get("key", ""))))
        typing = op == "type" or key_text is not None or pasting
        if op == "fill" or typing:
            # Typed text goes to the focused element whatever ref was named,
            # so a focused secret field is guarded the same as a named one.
            targets = [row] if row is not None else []
            if typing:
                targets += [r for r in obs.rows if "focused" in r.states]
            for target in targets:
                why = guards.needs_human_input(
                    target.role, target.subrole, target.label
                )
                if why:
                    raise ComputerUseError(
                        "needs_human",
                        f"{target.ref} is {why}; the user types it. Use handoff "
                        "with a reason, then continue when it returns.",
                    )
            if typing:
                # Focus can move after the observation (a page that focuses
                # its password box on load): ask the app what has it now.
                why = _focused_secret(obs.snapshot.get("app") or {})
                if why:
                    raise ComputerUseError(
                        "needs_human",
                        f"the focused field is {why}; the user types it. Use "
                        "handoff with a reason, then continue when it returns.",
                    )
            text = key_text if key_text is not None else str(kw.get("text", ""))
            held = [text]
            if typing:
                # Typing appends: a card number split across calls is still a
                # card number in the field. The text may land in the named
                # field or the focused one, so each is checked.
                held += [
                    field.value + text
                    for field in targets
                    if field.value and field.value != guards.USER_VALUE
                ]
            if any(guards.contains_card_number(t) for t in held):
                raise ComputerUseError(
                    "sensitive_data",
                    "the text contains a card number; it was not typed",
                )
        # Pressing a control is a click, its AX action, or an activating key.
        # A key goes to the focused control whatever ref was named, so the
        # focused controls are guarded as well as the named one.
        activating_key = op == "key" and _activating(kw.get("key"))
        if op in {"click", "action"} or activating_key:
            pressed = [row] if row is not None else []
            if activating_key:
                pressed += [r for r in obs.rows if "focused" in r.states]
            for target in pressed:
                if guards.is_money_commit(target.role, target.label):
                    return self._require_approval(obs, target)
        return None

    def _require_approval(self, obs: Observation, row: Row) -> tuple[str, dict]:
        """Let a commit through only with the user's approval of this screen.

        The approval binds to the window, the control and every amount on
        screen: if a total changes after the user approved, it is a new
        question. An approval is used once.
        """
        texts = obs.texts()
        key = _approval_key(obs, row)
        for aid, grant in list(self._approved.items()):
            if grant["key"] == key:
                del self._approved[aid]
                self._emit("approval_used", {"id": aid, "label": row.label})
                return aid, grant
        asked = next((a for a, p in self._pending.items() if p["key"] == key), None)
        aid = asked if asked is not None else f"a{next(self._approval_ids)}"
        if asked is None:
            self._pending[aid] = {
                "key": key,
                "label": row.label,
                "window": obs.window_id,
                "title": obs.title,
                "context": guards.money_context(texts, obs.choices())
                + (
                    ["(the page was only partly read: check every amount on screen)"]
                    if obs.truncated
                    else []
                ),
            }
            self._emit("approval_requested", {"id": aid, **_public(self._pending[aid])})
        pending = self._pending[aid]
        raise ComputerUseError(
            "needs_approval",
            f'"{row.label}" spends money or cannot be undone. Show the user what it '
            f"commits and wait for approval {aid}; then click it again. On screen: "
            + " | ".join(pending["context"] or ["(no amounts shown)"]),
        )

    # The methods below are the user's channel. A host wires them to the
    # person (a dialog, the app UI, a CLI); they are never model tools.

    def pending_approvals(self) -> dict:
        return {aid: _public(p) for aid, p in self._pending.items()}

    def approve(self, approval_id: str) -> dict:
        with self._lock:
            pending = self._pending.pop(approval_id, None)
            if pending is None:
                raise ComputerUseError(
                    "invalid_argument", f"no pending approval {approval_id}"
                )
            self._approved[approval_id] = pending
            self._emit("approved", {"id": approval_id, "label": pending["label"]})
            return {"approved": approval_id, "label": pending["label"]}

    def deny(self, approval_id: str) -> dict:
        with self._lock:
            pending = self._pending.pop(approval_id, None)
            if pending is None:
                raise ComputerUseError(
                    "invalid_argument", f"no pending approval {approval_id}"
                )
            self._emit("denied", {"id": approval_id})
            return {"denied": approval_id, "label": pending["label"]}

    def human_act(self, op: str, ref: str | None = None, **kwargs: Any) -> dict:
        """Input the user makes through the host (stands in for their keyboard)."""
        with self._lock:
            row = self._resolve(ref)[1] if ref else None
            self._emit("human_input", {"op": op, "label": row.label if row else ""})
            return self._act(op, ref, by_human=True, **kwargs)

    def human_done(self, window_id: str | int) -> None:
        with self._lock:
            known = self._known_window(window_id)
            self._human_done.add(known if known is not None else str(window_id))

    # -- waiting -----------------------------------------------------------

    def wait(
        self,
        window_id: str,
        *,
        until_text: str | None = None,
        until_gone: str | None = None,
        timeout: float = 30.0,
        _until_human: bool = False,
    ) -> dict:
        """Observe until the page shows (or stops showing) a text, or changes.

        With neither condition, returns once the page changed and then stayed
        quiet for ``WAIT_QUIET_S`` with no typing indicator showing (a chat
        reply arrives as "typing…", then the message). The returned
        observation's changes are relative to where the wait started, so a
        reply that arrived in several steps reads as one change.
        """
        with self._lock:
            start = self._window_obs(window_id)
        app, wid = start.app, start.window_id
        deadline = time.monotonic() + timeout
        met = False
        changed_at: float | None = None
        shown = _content(start)
        while True:
            with self._lock:
                obs = self._observe(app, wid)
            page = "\n".join(obs.texts()).lower()
            if until_text is not None:
                met = until_text.lower() in page
            elif until_gone is not None:
                met = until_gone.lower() not in page
            else:
                # Only new or changed text counts: focus, a cleared input box
                # or a button state after sending is not the reply.
                now = time.monotonic()
                content = _content(obs)
                if content != shown:
                    shown = content
                    changed_at = now
                met = (
                    not _until_human
                    and changed_at is not None
                    and now - changed_at >= WAIT_QUIET_S
                    and not _TYPING.search(page)
                )
            if _until_human and wid in self._human_done:
                self._human_done.discard(wid)
                met = True
            if met or time.monotonic() >= deadline:
                break
            time.sleep(WAIT_POLL_S)
        with self._lock:
            before = set(start.by_ref())
            for row in obs.rows:
                row.new = row.ref not in before
            obs.changes, obs.change_counts = _diff(start, obs.rows)
            obs.previous = start.obs_id
        return {"met": met, "observation": obs}

    def handoff(
        self,
        window_id: str,
        reason: str,
        *,
        until_text: str | None = None,
        until_gone: str | None = None,
        timeout: float = 600.0,
    ) -> dict:
        """Give the user the front to do what only they may do, then resume.

        Brings the window forward, tells the user why, blocks agent input to
        that window, and returns when the page shows the condition (or the
        user says they are done). The agent never sees what they typed into
        a field only they may fill.
        """
        with self._lock:
            obs = self._window_obs(window_id)
            front = self.take_front(obs.app, obs.window_id) or {}
            self._with_human[obs.window_id] = reason
            self._human_done.discard(obs.window_id)
        self._emit("handoff", {"window": obs.window_id, "reason": reason})
        # Name the window, so a user whose front did not switch knows where.
        _notify("Your turn", f"{reason} ({obs.title})" if obs.title else reason)
        try:
            out = self.wait(
                obs.window_id,
                until_text=until_text,
                until_gone=until_gone,
                timeout=timeout,
                _until_human=True,
            )
        finally:
            with self._lock:
                self._with_human.pop(obs.window_id, None)
        self._emit("handoff_done", {"window": obs.window_id, "met": out["met"]})
        out["front"] = bool(front.get("front"))
        return out

    # -- browsing ----------------------------------------------------------

    def open_url(self, window_id: str, url: str) -> dict:
        """Load a URL in a browser window through its address bar."""
        with self._lock:
            start = self._window_obs(window_id)
            obs = self._observe(start.app, start.window_id)
            bar = next(
                (
                    row
                    for row in obs.rows
                    if row.role in {"AXTextField", "AXComboBox"}
                    and _ADDRESS_BAR.search(row.label)
                ),
                None,
            )
            if bar is None:
                raise ComputerUseError(
                    "invalid_argument", "no address bar in this window"
                )
            old_title = obs.title
            # The address bar takes keyboard focus only when pressed; a value
            # written without focus is not what Return submits.
            # One op: its own steps do not gate each other (pressing a bar
            # that already has focus shows no change); the load check below
            # is what verifies it.
            for op, ref, kw in (
                ("click", bar.ref, {}),
                ("fill", bar.ref, {"text": url}),
                # Inline autocomplete selects a longer address the typed one
                # is a prefix of (amazon.com/ -> amazon.com/your-orders):
                # drop the selected completion so Return opens this URL.
                ("key", None, {"key": "forwarddelete", "window_id": obs.window_id}),
                ("key", None, {"key": "Return", "window_id": obs.window_id}),
            ):
                self._unresolved.pop(obs.window_id, None)
                step = self._act(op, ref, by_human=False, **kw)["receipt"]
                if step["effect"] == "refused":
                    # Never press Return on an address the bar did not take.
                    error = step.get("error") or {}
                    raise ComputerUseError(
                        str(error.get("code") or "action_failed"),
                        f"open_url stopped at {step['action']}: "
                        + str(error.get("message") or "refused"),
                    )
            self._unresolved.pop(obs.window_id, None)
        deadline = time.monotonic() + OPEN_URL_WAIT_S
        while True:
            with self._lock:
                now = self._observe(obs.app, obs.window_id)
            loaded = now.title != old_title and any(
                row.role not in {"AXButton", "AXPopUpButton", "AXTextField", "AXGroup"}
                and row.ref not in obs.by_ref()
                for row in now.rows
            )
            if loaded or time.monotonic() >= deadline:
                break
            time.sleep(WAIT_POLL_S)
        with self._lock:
            settled_obs, settled = self._settle(obs.app, obs.window_id, slow=False)
            before = set(obs.by_ref())
            for row in settled_obs.rows:
                row.new = row.ref not in before
            settled_obs.changes, settled_obs.change_counts = _diff(
                obs, settled_obs.rows
            )
            settled_obs.previous = obs.obs_id
            if not loaded:
                # No load seen: like any action without a visible outcome,
                # it blocks further input until the window is observed.
                self._unresolved[obs.window_id] = f"open_url {url}"
        receipt = {
            "action": f"open_url {url}",
            "effect": "confirmed" if loaded else "unverifiable",
            "settled": settled,
            "observation": settled_obs.obs_id,
            "title": settled_obs.title,
        }
        if not loaded:
            receipt["unresolved"] = (
                "load not confirmed; observe before sending more input"
            )
        return {"receipt": receipt, "observation": settled_obs}

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event is not None:
            try:
                self.on_event(kind, payload)
            except Exception:  # noqa: BLE001 - a host hook must not break a task
                pass

    # -- observing ---------------------------------------------------------

    def observe(self, app: str, window_id: str | int | None = None) -> Observation:
        with self._lock:
            return self._observe(app, window_id)

    def _observe(self, app: str, window_id: str | int | None = None) -> Observation:
        self.keep_awake()
        _require_display_awake()
        started = time.perf_counter()
        snapshot = backend.get_app_state(
            app,
            screenshot=False,
            use_cache=False,
            window_id=window_id,
            activate=backend.OBSERVE_BY_ROUTE,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        wid = str(snapshot["window_id"])
        previous = self._latest.get(wid)
        rows = self._rows(snapshot, previous)
        changes, counts = _diff(previous, rows)
        obs = Observation(
            obs_id=f"o{next(self._obs_ids)}",
            app=str(snapshot["app"].get("name") or app),
            window_id=wid,
            title=str(snapshot.get("window", {}).get("title") or ""),
            snapshot=snapshot,
            rows=rows,
            changes=changes,
            change_counts=counts,
            previous=previous.obs_id if previous else None,
            elapsed_ms=elapsed_ms,
            truncated=bool(snapshot.get("truncated")),
            window_ids=list(snapshot.get("visible_window_ids") or []),
        )
        self._latest[wid] = obs
        self._unresolved.pop(wid, None)
        return obs

    def _rows(self, snapshot: dict, previous: Observation | None) -> list[Row]:
        lives = backend.live_elements(snapshot) or []
        seen_before = set(previous.by_ref()) if previous else set()
        rows = []
        used: set[str] = set()
        for position, element in enumerate(snapshot["elements"]):
            live = lives[position] if position < len(lives) else None
            role = str(element.get("role") or "")
            label = str(element.get("label") or "")
            ref = None
            if live is not None:
                known = self._refs.get(live)
                # Same live element, same role: same ref, so a changed text
                # reads as a change rather than a removal plus an addition.
                if known is not None and known[1] == role and known[0] not in used:
                    ref = known[0]
                    self._refs[live] = (ref, role, label)
            if ref is None:
                ref = f"e{next(self._ref_ids)}"
                if live is not None:
                    self._refs[live] = (ref, role, label)
            used.add(ref)
            value = element.get("value")
            if (
                isinstance(value, str)
                and value
                and guards.needs_human_input(
                    role, str(element.get("subrole") or ""), label
                )
            ):
                # What the user typed into a secret field is theirs.
                value = guards.USER_VALUE
            center = element.get("center") or [0, 0]
            # The driver clips frames to the window: a scrolled-away element
            # (below the fold, a carousel's hidden slide) comes back empty, and
            # a "visually hidden" one (skip links, clipped to 1 px) is a sliver.
            width, height = (
                float(element.get("width") or 0),
                float(element.get("height") or 0),
            )
            on_screen = width > 1 and height > 1
            sliver = not on_screen and width > 0 and height > 0
            rows.append(
                Row(
                    ref=ref,
                    role=role,
                    label=label,
                    value=value if isinstance(value, str) else None,
                    states=tuple(element.get("states") or ()),
                    index=int(element["index"]),
                    center=(int(center[0]), int(center[1])),
                    new=previous is not None and ref not in seen_before,
                    subrole=str(element.get("subrole") or ""),
                    on_screen=on_screen,
                    sliver=sliver,
                )
            )
        if len(self._refs) > MAX_REMEMBERED_ELEMENTS:
            # This window's previous observation is being replaced.
            current = set(used)
            for obs in self._latest.values():
                if obs is not previous:
                    current.update(row.ref for row in obs.rows)
            self._refs = {
                live: entry for live, entry in self._refs.items() if entry[0] in current
            }
        return rows

    # -- acting ------------------------------------------------------------

    def _resolve(self, ref: str) -> tuple[Observation, Row]:
        for obs in self._latest.values():
            row = obs.by_ref().get(ref)
            if row is not None:
                return obs, row
        latest = ", ".join(f"{o.obs_id} (window {w})" for w, o in self._latest.items())
        raise ComputerUseError(
            "stale_ref",
            f"{ref} is not in the latest observation of any window; "
            f"latest: {latest or 'none'}. Observe and use a current ref.",
        )

    def _known_window(self, window_id: str | int) -> str | None:
        """The observed window ``window_id`` names ("cg:12" or 12), or None."""
        if str(window_id) in self._latest:
            return str(window_id)
        try:
            wanted = backend._cg_window_id(window_id)
        except ComputerUseError:
            return None
        for wid in self._latest:
            try:
                if backend._cg_window_id(wid) == wanted:
                    return wid
            except ComputerUseError:
                continue
        return None

    def _window_obs(self, window_id: str | int | None) -> Observation:
        if window_id is None:
            if len(self._latest) == 1:
                return next(iter(self._latest.values()))
            raise ComputerUseError(
                "invalid_argument",
                "observe the target window first (or pass window_id)",
            )
        known = self._known_window(window_id)
        if known is not None:
            return self._latest[known]
        # A window not observed yet: observe it through the app that owns it.
        # A named window is never swapped for another one.
        owner = _window_owner(window_id)
        if owner is not None:
            obs = self._observe(owner, window_id)
            if self._known_window(window_id) == obs.window_id:
                return obs
        raise ComputerUseError(
            "window_not_found",
            f"window {window_id} is not observed and could not be found",
        )

    def act(self, op: str, ref: str | None = None, **kwargs: Any) -> dict:
        with self._lock:
            return self._act(op, ref, by_human=False, **kwargs)

    def _act(self, op: str, ref: str | None, *, by_human: bool, **kwargs: Any) -> dict:
        if ref is not None:
            obs, row = self._resolve(ref)
        else:
            obs, row = self._window_obs(kwargs.pop("window_id", None)), None
        if obs.window_id in self._with_human and not by_human:
            self._guard(op, obs, row, kwargs)  # raises with_human
        # The gate stops the agent sending blind input; the user sees the screen.
        if obs.window_id in self._unresolved and not by_human:
            raise ComputerUseError(
                "unresolved_outcome",
                f"an earlier action ({self._unresolved[obs.window_id]}) has an "
                f"unresolved outcome; observe window {obs.window_id} before "
                "sending more input",
            )
        self.keep_awake()
        _require_display_awake()
        # Last before acting, so an approval is used only by an attempt.
        used = None if by_human else self._guard(op, obs, row, kwargs)
        app, snapshot, wid = obs.app, obs.snapshot, obs.window_id
        front_before = _frontmost_bundle()
        started = time.perf_counter()
        error = None
        result: dict = {}
        try:
            try:
                result = self._dispatch(op, app, snapshot, wid, row, kwargs)
            except ComputerUseError as exc:
                # A live page (a rotating carousel, a feed) shifts the walk's
                # indexes between observing and acting. The ref follows the
                # element itself, so when it is still there unchanged, act on
                # a fresh snapshot once instead of refusing.
                if exc.code not in _SHIFTED or row is None:
                    raise
                fresh = self._observe(app, wid)
                again = fresh.by_ref().get(row.ref)
                if again is None or (again.role, again.label) != (row.role, row.label):
                    raise
                if used is not None and _approval_key(fresh, again) != used[1]["key"]:
                    # The approval covered the screen the user saw; a shifted
                    # page with other amounts is a new question.
                    raise
                obs, row, snapshot = fresh, again, fresh.snapshot
                result = self._dispatch(op, app, snapshot, wid, row, kwargs)
        except ComputerUseError as exc:
            error = exc
            if used is not None:
                # The click never happened (stale snapshot, drift): the user's
                # approval still stands for the same screen.
                self._approved[used[0]] = used[1]
                self._emit("approval_kept", {"id": used[0], "label": used[1]["label"]})
        acted_ms = round((time.perf_counter() - started) * 1000)
        slow = op == "key" and str(kwargs.get("key", "")).lower() in _SLOW_KEYS
        # Background typing into a hidden renderer is consumed long after it
        # was posted; settle on the requested text rather than on quiet.
        want = None
        if op == "fill" and ref is not None:
            want = (ref, str(kwargs["text"]))
        elif op == "click" and ref is not None and kwargs.get("menu_item"):
            want = (ref, str(kwargs["menu_item"]))
        after, settled = self._settle(
            app, wid, slow=slow or want is not None, want=want
        )
        description = f"{op} {ref or ''}".strip()
        added, removed, changed = after.change_counts
        target_after = after.by_ref().get(ref) if ref is not None else None
        target_changed = (
            None
            if row is None
            else target_after is None or target_after.signature() != row.signature()
        )
        observed_match = (
            want is not None
            and target_after is not None
            and target_after.value is not None
            and target_after.value.strip() == want[1].strip()
        )
        if observed_match:
            # The observation shows exactly what was asked for, whatever the
            # transport reported (it can refuse after its first route landed).
            effect = "confirmed"
        elif error:
            effect = "refused"
        else:
            effect = str(result.get("effect") or "unverifiable")
            if effect != "confirmed":
                if target_changed:
                    effect = "target_changed"
                elif added or removed or changed:
                    effect = "other_changes_only"
                elif effect == "unverifiable":
                    effect = "no_visible_change"
        # Something else on the page changing is an outcome (a chat message
        # sent from an already-focused button); only silence is unresolved.
        if effect in {"suspected_noop", "no_visible_change"} or (
            effect not in {"refused", "confirmed", "other_changes_only"} and not settled
        ):
            self._unresolved[wid] = description
        new_windows = sorted(set(after.window_ids) - set(obs.window_ids))
        front_after = _frontmost_bundle()
        receipt = {
            "action": description,
            "effect": effect,
            "settled": settled,
            "acted_ms": acted_ms,
            "observation": after.obs_id,
            "changes": after.change_counts,
        }
        if target_changed is not None:
            receipt["target_changed"] = target_changed
        if error:
            receipt["error"] = {"code": error.code, "message": error.message}
        if result.get("verification"):
            receipt["verification"] = result["verification"]
        if result.get("warnings"):
            receipt["warnings"] = result["warnings"]
        for key in ("menu", "focus_restored", "mode"):
            if key in result:
                receipt[key] = result[key]
        if new_windows:
            receipt["new_windows"] = new_windows
        if front_after != front_before:
            receipt["frontmost_changed"] = [front_before, front_after]
        if wid in self._unresolved:
            receipt["unresolved"] = (
                "outcome not confirmed; observe before sending more input"
            )
        return {"receipt": receipt, "observation": after}

    def _dispatch(
        self, op: str, app: str, snapshot: dict, wid: str, row: Row | None, kw: dict
    ) -> dict:
        index = row.index if row is not None else None
        if op == "click":
            return backend.click(
                app,
                element_index=index,
                click_count=int(kw.get("count", 1)),
                mouse_button=str(kw.get("button", "left")),
                menu_item=kw.get("menu_item"),
                modifiers=kw.get("modifiers"),
                expected_snapshot=snapshot,
                window_id=wid,
            )
        if op in {"fill", "action"} and index is None:
            raise ComputerUseError("invalid_argument", f"{op} needs a ref")
        if op == "fill" and index is not None:
            return backend.set_value(
                app, index, str(kw["text"]), expected_snapshot=snapshot, window_id=wid
            )
        if op == "type":
            return backend.type_text(app, str(kw["text"]), window_id=wid)
        if op == "key":
            try:
                return backend.press_key(
                    app,
                    str(kw["key"]),
                    window_id=wid,
                    expected_snapshot=snapshot if index is not None else None,
                    element_index=index,
                )
            except ComputerUseError as exc:
                # Return is bound to the focused element, and a web field
                # often has no focus of its own yet (or its wrapper has it):
                # put the caret in the field, then press at window level.
                if (
                    exc.code != "target_drift"
                    or row is None
                    or row.role not in guards.TEXT_ROLES
                    or str(kw["key"]).lower() not in {"return", "enter"}
                ):
                    raise
                backend.click(
                    app, element_index=index, expected_snapshot=snapshot, window_id=wid
                )
                return backend.press_key(app, str(kw["key"]), window_id=wid)
        if op == "scroll":
            x, y = row.center if row is not None else (None, None)
            return backend.scroll(
                app,
                str(kw.get("direction", "down")),
                pages=float(kw.get("pages", 1)),
                x=x,
                y=y,
                window_id=wid,
                expected_snapshot=snapshot,
            )
        if op == "action" and index is not None:
            return backend.perform_secondary_action(
                app, index, str(kw["name"]), window_id=wid
            )
        raise ComputerUseError("invalid_argument", f"unknown op {op!r}")

    def _settle(
        self,
        app: str,
        wid: str,
        *,
        slow: bool,
        want: tuple[str, str] | None = None,
    ) -> tuple[Observation, bool]:
        """Observe until two consecutive samples match (or the cap).

        With ``want`` (ref, text) the target must also show that text.
        """
        cap = SETTLE_CAP_SLOW_S if slow else SETTLE_CAP_S
        started = time.monotonic()
        time.sleep(SETTLE_POLL_S)
        baseline = self._latest[wid]
        last_signature: tuple | None = None
        while True:
            snapshot = backend.get_app_state(
                app,
                screenshot=False,
                use_cache=False,
                window_id=wid,
                activate=backend.OBSERVE_BY_ROUTE,
            )
            signature: tuple | None = tuple(
                (
                    e.get("role"),
                    e.get("label"),
                    e.get("value"),
                    tuple(e.get("states") or ()),
                )
                for e in snapshot["elements"]
            )
            elapsed = time.monotonic() - started
            if want is not None and not _shows(baseline, snapshot, want):
                signature = None  # not there yet: never counts as settled
            if (
                signature is not None
                and signature == last_signature
                and elapsed >= SETTLE_MIN_S
            ):
                settled = True
                break
            if elapsed >= cap:
                settled = False
                break
            last_signature = signature
            time.sleep(SETTLE_POLL_S)
        # Turn the settled snapshot into the observation the receipt cites,
        # diffed against the observation the action was taken from.
        self._latest[wid] = baseline
        obs = self._observe_snapshot(app, snapshot, baseline)
        return obs, settled

    def _observe_snapshot(
        self, app: str, snapshot: dict, previous: Observation
    ) -> Observation:
        rows = self._rows(snapshot, previous)
        changes, counts = _diff(previous, rows)
        obs = Observation(
            obs_id=f"o{next(self._obs_ids)}",
            app=previous.app,
            window_id=previous.window_id,
            title=str(snapshot.get("window", {}).get("title") or ""),
            snapshot=snapshot,
            rows=rows,
            changes=changes,
            change_counts=counts,
            previous=previous.obs_id,
            elapsed_ms=0,
            truncated=bool(snapshot.get("truncated")),
            window_ids=list(snapshot.get("visible_window_ids") or []),
        )
        self._latest[previous.window_id] = obs
        return obs


def _diff(
    previous: Observation | None, rows: list[Row]
) -> tuple[list[str], tuple[int, int, int]]:
    if previous is None:
        return [], (0, 0, 0)
    before = previous.by_ref()
    now = {row.ref: row for row in rows}
    added = [row for row in rows if row.ref not in before]
    removed = [row for ref, row in before.items() if ref not in now]
    changed = [
        (before[row.ref], row)
        for row in rows
        if row.ref in before and before[row.ref].signature() != row.signature()
    ]
    lines: list[str] = []
    for old, new in changed:
        parts = []
        if old.value != new.value:
            parts.append(f"value {old.value!r} -> {new.value!r}")
        if old.states != new.states:
            parts.append(f"[{','.join(old.states)}] -> [{','.join(new.states)}]")
        if old.label != new.label:
            parts.append(f'label "{old.label[:40]}" -> "{new.label[:40]}"')
        lines.append(
            f'~ {new.ref} {_short_role(new.role)} "{new.label[:40]}" '
            + "; ".join(parts)
        )
    lines.extend(f"+ {row.render().strip()}" for row in added)
    lines.extend(f"- {row.render().strip()}" for row in removed)
    if len(lines) > MAX_CHANGE_LINES:
        extra = len(lines) - MAX_CHANGE_LINES
        lines = lines[:MAX_CHANGE_LINES] + [f"... {extra} more changes"]
    return lines, (len(added), len(removed), len(changed))


def _shows(baseline: Observation, snapshot: dict, want: tuple[str, str]) -> bool:
    """Whether the element behind ``want[0]`` shows the text ``want[1]``."""
    row = baseline.by_ref().get(want[0])
    if row is None:
        return False
    target = None
    for element in snapshot["elements"]:
        if (
            element.get("role") == row.role
            and element.get("center")
            and (tuple(element["center"]) == row.center)
        ):
            target = element
            break
    value = target.get("value") if target else None
    return isinstance(value, str) and value.strip() == want[1].strip()


def _require_display_awake() -> None:
    try:
        import Quartz  # type: ignore[import-untyped]

        asleep = bool(Quartz.CGDisplayIsAsleep(Quartz.CGMainDisplayID()))
    except Exception:  # pragma: no cover
        return
    if asleep:
        raise ComputerUseError(
            "display_asleep",
            "the screen is off, so no window can be seen or operated; "
            "wake the Mac (and keep it awake) to continue",
        )


def _content(obs: Observation) -> frozenset[tuple[str, str]]:
    """What a page says, apart from what is typed into it."""
    return frozenset(
        (row.role, row.label)
        for row in obs.rows
        if row.label
        and row.role not in guards.TEXT_ROLES
        and not row.is_browser_noise()
    )


def _split_key(key: object) -> tuple[frozenset[str], str]:
    """``(modifiers, base key)`` of a chord: "cmd+shift+Return" ->
    ({"cmd", "shift"}, "Return"); a bare "+" is its own base key."""
    name = str(key or "")
    head, sep, base = name.rpartition("+")
    if sep and not base:  # "cmd++" or "+": the base key is "+"
        head, base = head[:-1] if head.endswith("+") else head, "+"
    modifiers = frozenset(m.lower() for m in head.split("+") if m)
    return modifiers, base


# Modifiers that still type a character: shift+7 is "&", option+a is "å".
_TYPING_MODIFIERS = frozenset({"shift", "option", "alt"})


def _key_text(key: object) -> str | None:
    """The character pressing ``key`` types ("a", "shift+7", "option+a"), or
    None. With option the character differs from the base key, which stands
    in for it."""
    modifiers, base = _split_key(key)
    if not modifiers <= _TYPING_MODIFIERS:
        return None
    return base if len(base) == 1 and base.isprintable() else None


def _activating(key: object) -> bool:
    """Whether ``key`` presses the focused control, with any modifiers."""
    return _split_key(key)[1].lower() in _ACTIVATING_KEYS


def _focused_secret(app_info: dict) -> str | None:
    """Why the app's focused element is for the user only, read live.

    Reads only role, subrole and naming attributes, never a value.
    """
    if app_info.get("pid") is None:
        return None
    unchecked = "a field whose focus could not be checked"
    try:
        from . import ax_driver

        app_element = backend._pid_app_element(app_info)
        readable, focused = ax_driver._get_checked(app_element, "AXFocusedUIElement")
        if not readable:
            return unchecked
        if focused is None:
            return None  # nothing has focus: typed text goes nowhere
        names: dict[str, str] = {}
        for attribute in (
            "AXRole",
            "AXSubrole",
            "AXDescription",
            "AXTitle",
            "AXPlaceholderValue",
        ):
            readable, value = ax_driver._get_checked(focused, attribute)
            if not readable:
                return unchecked
            names[attribute] = value.strip() if isinstance(value, str) else ""
        role, subrole = names["AXRole"], names["AXSubrole"]
        if "AXSecureTextField" in (role, subrole):
            return guards.needs_human_input(role, subrole, "")
        label = next(
            (
                names[a]
                for a in ("AXDescription", "AXTitle", "AXPlaceholderValue")
                if names[a]
            ),
            "",
        )
        return guards.needs_human_input(role, subrole, label)
    except Exception:  # noqa: BLE001 - focus could not be inspected: fail closed
        return unchecked


def _window_owner(window_id: str | int) -> str | None:
    """``pid:N`` of the app that owns a CG window, or None."""
    try:
        from Quartz import (  # type: ignore[import-untyped]
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListOptionAll,
        )

        wanted = backend._cg_window_id(window_id)
        for window in (
            CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID) or []
        ):
            if int(window.get("kCGWindowNumber", -1)) == wanted:
                return f"pid:{int(window['kCGWindowOwnerPID'])}"
    except Exception:  # noqa: BLE001 - no CG window list, or a malformed id
        return None
    return None


def _frontmost_pid() -> int | None:
    # NSWorkspace's frontmost app is refreshed by run-loop notifications, so a
    # long-lived process without a run loop reads a stale value; ask the
    # WindowServer instead.
    try:
        from . import background_input

        pid = background_input.front_pid()
        if pid is not None:
            return pid
        from AppKit import NSWorkspace  # type: ignore[import-untyped]

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None
    except Exception:  # pragma: no cover
        return None


def _frontmost_bundle() -> str | None:
    pid = _frontmost_pid()
    if pid is None:
        return None
    try:
        from AppKit import NSRunningApplication  # type: ignore[import-untyped]

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(app.bundleIdentifier()) if app is not None else None
    except Exception:  # pragma: no cover
        return None


def _approval_key(obs: Observation, row: Row) -> tuple:
    """What an approval binds to: window, control, every amount on screen and
    every chosen option (delivery slot, plan, payment method, ...)."""
    return (
        obs.window_id,
        row.ref,
        row.label,
        guards.amounts(obs.texts()),
        tuple(obs.choices()),
    )


def _public(pending: dict) -> dict:
    return {k: v for k, v in pending.items() if k != "key"}


def _notify(title: str, message: str) -> None:
    """Tell the person at the Mac that it is their turn (sound + banner)."""
    script = f'display notification {_applescript_string(message)} with title {_applescript_string(title)} sound name "Glass"'
    try:
        subprocess.Popen(
            ["/usr/bin/osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:  # pragma: no cover
        pass


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
